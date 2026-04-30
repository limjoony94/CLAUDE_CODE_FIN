"""R26 Grid Bot — Main Loop.

Per cycle (5min poll):
1. Fetch 1h candles (last 800 bars for 30d ATR median).
2. Compute ranging filter from latest bar.
3. If no active grid + ranging → setup grid.
4. If active grid:
   - Check fills (grid orders + TP orders)
   - Trend exit: |price - init_mid| > 1.5% AND not ranging → force close all
   - Max lifetime: > 168h → force close (max grid lifetime)
5. Hard halts: daily loss / 30d loss / API errors / emergency adverse.
"""
import logging
import time
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional

import ccxt
import pandas as pd

from .config import load_config, load_api_keys
from .indicators import compute_ranging_filter, is_ranging_now, compute_trend_exit_signal
from .grid import GridManager, StateManager, GridState

logger = logging.getLogger('r26_grid.bot')


class R26GridBot:
    def __init__(self, config_path: str = 'config/r26_grid_config.yaml'):
        self.config = load_config(config_path)
        self.api_keys = load_api_keys(self.config['api_keys_path'])

        self.symbol = self.config['exchange']['symbol']
        self.timeframe = self.config['exchange']['timeframe']

        # Risk halts
        self.consecutive_api_errors = 0
        self.start_capital = None
        self.start_ts = datetime.now(timezone.utc)

        self.exchange = self._init_exchange()
        self.sm = StateManager(self.config['logging']['state_path'])
        self.state = self.sm.load()
        # Pass notional_callback for per-TP compound + journal_path for trade journal
        self.grid = GridManager(
            self.exchange, self.symbol, self.state, self.sm,
            spacing_pct=self.config['strategy']['grid_spacing_pct'],
            notional_callback=self._compute_per_level_notional,
            journal_path=self.config['logging'].get('journal_path')
        )
        # Reconcile state vs exchange (catches orphans from crash + race)
        self._reconcile_state_exchange()
        # Cycle counter for periodic balance snapshot
        self._cycle_count = 0
        self._last_cycle_summary = None

    def _init_exchange(self) -> ccxt.bingx:
        ex = ccxt.bingx({
            'apiKey': self.api_keys['api_key'],
            'secret': self.api_keys['secret'],
            'enableRateLimit': True,
            'options': {
                'defaultType': 'swap',
                'recvWindow': 10000,
            },
        })

        # Verify hedge mode (BUG#66 protection)
        if self.config['exchange'].get('hedge_mode_check', True):
            try:
                pos_mode = ex.fetch_position_mode()
                if pos_mode and pos_mode.get('hedged') is True:
                    logger.error("Account is in Hedge mode! R26 requires One-Way mode. "
                                  "Attempting auto-correction...")
                    ex.set_position_mode(hedged=False)
                    pos_mode = ex.fetch_position_mode()
                    if pos_mode and pos_mode.get('hedged') is True:
                        logger.critical("Failed to set One-Way mode. ABORTING.")
                        sys.exit(1)
                    logger.info("Position mode corrected to One-Way")
            except Exception as e:
                logger.warning(f"Position mode check failed (non-fatal): {e}")

        # Set leverage
        ex_lev = self.config['exchange']['exchange_leverage']
        try:
            ex.set_leverage(ex_lev, self.symbol, params={'side': 'BOTH'})
            logger.info(f"Exchange leverage set to {ex_lev}× for {self.symbol}")
        except Exception as e:
            logger.error(f"Failed to set leverage {ex_lev}×: {e}. Continuing — verify manually.")

        return ex

    def _reconcile_state_exchange(self):
        """On startup, compare state JSON to actual exchange state.

        Detects:
        1. State levels with order_id NOT in exchange open orders → order was
           filled or canceled while bot was offline
        2. Exchange open orders NOT in state → orphan orders, cancel them
        3. Exchange positions NOT in state.open_positions → orphan positions,
           emergency close them (no TP, can't predict behavior)
        """
        if not self.state.active:
            logger.info("State inactive — skipping reconciliation")
            return

        try:
            open_orders = self.exchange.fetch_open_orders(self.symbol)
            open_order_ids = {str(o.get('id')) for o in open_orders}
            logger.info(f"Reconciliation: {len(open_orders)} open orders on exchange")

            # Check state levels — order_id should be in exchange open orders
            state_order_ids = set()
            stale_levels = []
            for level in (self.state.buy_levels + self.state.sell_levels):
                if level.order_id and not level.filled:
                    state_order_ids.add(str(level.order_id))
                    if str(level.order_id) not in open_order_ids:
                        stale_levels.append(level)

            for level in stale_levels:
                logger.warning(f"Stale state level (order missing from exchange): "
                                f"side={level.side} lv={level.level_idx} "
                                f"order_id={level.order_id} — clearing order_id, "
                                f"force_close_all triggered")
            if stale_levels:
                logger.warning("Triggering force_close_all due to stale state — "
                                "safer than running with unknown order status")
                self.grid.force_close_all(reason='RECONCILE_STALE_STATE')
                return

            # Check for orphan orders on exchange (in exchange, not in state)
            orphan_orders = [oid for oid in open_order_ids if oid not in state_order_ids]
            if orphan_orders:
                logger.warning(f"Orphan orders on exchange (not tracked by bot): "
                                f"{orphan_orders} — cancelling")
                for oid in orphan_orders:
                    try:
                        self.exchange.cancel_order(oid, self.symbol)
                    except Exception as e:
                        logger.error(f"Failed to cancel orphan {oid}: {e}")

            # Check positions
            positions = self.exchange.fetch_positions([self.symbol])
            actual_positions = [p for p in positions if abs(float(p.get('contracts') or 0)) > 0]
            tracked_qty = sum(abs(p.qty_btc) for p in self.state.open_positions)
            actual_qty = sum(abs(float(p.get('contracts') or 0)) for p in actual_positions)
            logger.info(f"Reconciliation: tracked positions qty={tracked_qty:.6f}, "
                          f"exchange actual qty={actual_qty:.6f}")
            if abs(tracked_qty - actual_qty) > 0.0005:  # 0.0005 BTC tolerance
                logger.warning(f"Position qty mismatch: state {tracked_qty:.6f} vs "
                                f"exchange {actual_qty:.6f}. force_close_all to clean state")
                self.grid.force_close_all(reason='RECONCILE_POS_MISMATCH')

        except Exception as e:
            logger.error(f"Reconciliation failed (non-fatal, continuing): {e}")

    def fetch_recent_candles(self) -> pd.DataFrame:
        """Fetch last N bars of timeframe candles for ranging filter."""
        n = self.config['exchange']['candle_bars_fetch']
        try:
            ohlcv = self.exchange.fetch_ohlcv(self.symbol, self.timeframe, limit=n)
            df = pd.DataFrame(ohlcv, columns=['ts_ms', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['ts_ms'], unit='ms', utc=True)
            self.consecutive_api_errors = 0
            return df
        except Exception as e:
            self.consecutive_api_errors += 1
            logger.error(f"fetch_ohlcv failed (errors {self.consecutive_api_errors}): {e}")
            raise

    def get_account_equity(self) -> float:
        """Fetch USDT TRUE equity (cash + unrealized PnL, excl. reserved margin).

        BingX `total` field via CCXT = `availableMargin` (cash − reservations).
        For halt logic we need TRUE equity from raw response: info.data.balance.equity.
        """
        try:
            balance = self.exchange.fetch_balance({'type': 'swap'})
            # Try BingX raw equity field
            info = balance.get('info', {})
            data = info.get('data', {})
            bal = data.get('balance', {})
            equity_str = bal.get('equity')
            if equity_str is not None:
                return float(equity_str)
            # Fallback to CCXT 'total' if raw not available
            usdt_total = balance.get('USDT', {}).get('total', 0.0)
            logger.warning(f"BingX raw equity not found; falling back to CCXT total={usdt_total}")
            return float(usdt_total)
        except Exception as e:
            logger.error(f"fetch_balance failed: {e}")
            return -1.0

    def check_halts(self) -> Optional[str]:
        """Return halt reason string if should sys.exit, else None."""
        # API errors
        if self.consecutive_api_errors >= self.config['risk']['halt_consecutive_api_errors']:
            return f"CONSECUTIVE_API_ERRORS_{self.consecutive_api_errors}"

        equity = self.get_account_equity()
        if equity < 0:
            return None  # API failure handled by api_errors

        if self.start_capital is None or self.start_capital <= 0:
            self.start_capital = equity
            return None

        # Daily loss check
        loss_pct = (self.start_capital - equity) / self.start_capital * 100
        if loss_pct > self.config['risk']['halt_daily_loss_pct']:
            return f"DAILY_LOSS_{loss_pct:.2f}_PCT"

        # 30d loss (treat as cum since start for now)
        if loss_pct > self.config['risk']['halt_30d_loss_pct']:
            return f"30D_LOSS_{loss_pct:.2f}_PCT"

        # Emergency adverse on open position
        emergency_threshold = self.config['risk']['halt_emergency_adverse_pct']
        try:
            ticker = self.exchange.fetch_ticker(self.symbol)
            current_price = ticker.get('last') or ticker.get('close')
            if current_price:
                for pos in self.state.open_positions:
                    if pos.side == 'long':
                        adverse = (pos.entry_price - current_price) / pos.entry_price * 100
                    else:
                        adverse = (current_price - pos.entry_price) / pos.entry_price * 100
                    if adverse > emergency_threshold:
                        return f"EMERGENCY_ADVERSE_{adverse:.2f}_PCT_pos_side_{pos.side}"
        except Exception as e:
            logger.warning(f"Ticker fetch for emergency check failed: {e}")

        return None

    def cycle(self):
        """One bot cycle."""
        # 1. Fetch recent candles
        df = self.fetch_recent_candles()
        if df.empty or len(df) < self.config['bot']['warmup_bars']:
            logger.warning(f"Insufficient candles: {len(df)} (need {self.config['bot']['warmup_bars']})")
            return

        latest = df.iloc[-1]
        current_price = float(latest['close'])
        current_ts_ms = int(latest['ts_ms'])

        # 2. Compute ranging filter
        ranging = is_ranging_now(
            df,
            atr_period=self.config['strategy']['atr_period'],
            lookback_bars=self.config['strategy']['atr_pct_median_lookback_bars']
        )

        # Compact cycle heartbeat: log only on state change OR every N cycles
        self._cycle_count += 1
        cycle_summary = (round(current_price, 0), ranging, self.state.active,
                          len(self.state.open_positions))
        snapshot_n = self.config['logging'].get('balance_snapshot_every_n_cycles', 12)
        compact_mode = self.config['logging'].get('cycle_heartbeat_compact', True)
        should_log_cycle = (
            (not compact_mode) or
            (cycle_summary != self._last_cycle_summary) or
            (self._cycle_count % snapshot_n == 0)
        )
        if should_log_cycle:
            equity = self.get_account_equity() if self._cycle_count % snapshot_n == 0 else None
            extra = f", equity=${equity:.2f}" if equity and equity > 0 else ""
            logger.info(f"Cycle: price={current_price:.2f}, ranging={ranging}, "
                         f"grid_active={self.state.active}, "
                         f"open_pos={len(self.state.open_positions)}{extra}")
        self._last_cycle_summary = cycle_summary

        # 3. Halt check (BEFORE any new orders)
        halt_reason = self.check_halts()
        if halt_reason:
            logger.critical(f"HALT TRIGGERED: {halt_reason}")
            self.grid.force_close_all(reason=f"HALT_{halt_reason}")
            logger.critical("Halt complete. Exiting.")
            sys.exit(1)

        # 4. Active grid management
        if self.state.active:
            # Check fills + TPs
            self.grid.check_fills(current_price)
            self.grid.check_tp_fills()

            # Trend exit signal
            need_trend_exit = compute_trend_exit_signal(
                current_price, self.state.init_mid,
                self.config['strategy']['trend_exit_distance_pct'],
                ranging
            )
            if need_trend_exit:
                logger.warning(f"Trend exit triggered: price={current_price:.2f}, "
                                f"init_mid={self.state.init_mid:.2f}, "
                                f"dist={abs(current_price - self.state.init_mid)/self.state.init_mid*100:.2f}%, "
                                f"ranging={ranging}")
                self.grid.force_close_all(reason='TREND_EXIT')
                return

            # Max lifetime check
            elapsed_bars = (current_ts_ms - self.state.init_ts_ms) / (60 * 60 * 1000)  # 1h bars
            if elapsed_bars > self.config['strategy']['max_grid_lifetime_bars']:
                logger.warning(f"Max grid lifetime reached: {elapsed_bars:.0f}h")
                self.grid.force_close_all(reason='MAX_LIFETIME')
                return

        # 5. Setup new grid if no active grid + ranging
        if not self.state.active and ranging:
            per_level_notional = self._compute_per_level_notional()
            logger.info(f"Setting up new grid at mid={current_price:.2f}, "
                         f"per_level_notional=${per_level_notional:.2f}")
            self.grid.setup_grid(
                init_mid=current_price,
                ts_ms=current_ts_ms,
                levels_each_side=self.config['strategy']['grid_levels_each_side'],
                per_level_notional_usd=per_level_notional
            )

    def _compute_per_level_notional(self) -> float:
        """Compute per-level notional based on auto-size mode.

        If auto_size_from_balance: per_level = balance × util × leverage / total_levels
        Else: use fixed per_level_notional_usd from config.
        """
        risk_cfg = self.config['risk']
        if risk_cfg.get('auto_size_from_balance', False):
            equity = self.get_account_equity()
            if equity <= 0:
                logger.error(f"Cannot fetch balance for auto-size; "
                              f"falling back to fixed ${risk_cfg['per_level_notional_usd']}")
                return float(risk_cfg['per_level_notional_usd'])
            util = risk_cfg.get('balance_utilization_pct', 100) / 100
            # USE trading_leverage (actual sizing), NOT exchange_leverage (max permission)
            leverage = self.config['exchange']['trading_leverage']
            total_levels = 2 * self.config['strategy']['grid_levels_each_side']
            per_level = (equity * util * leverage) / total_levels
            logger.info(f"Auto-sized per_level_notional=${per_level:.2f} "
                          f"from balance ${equity:.2f} × util {util*100:.0f}% × "
                          f"trading_leverage {leverage}× / {total_levels} levels")
            return per_level
        return float(risk_cfg['per_level_notional_usd'])

    def run(self):
        """Main loop."""
        poll_interval = self.config['bot']['poll_interval_seconds']
        logger.info(f"R26 Grid Bot starting. Poll interval: {poll_interval}s")
        logger.info(f"Symbol: {self.symbol}, Leverage: {self.config['exchange']['exchange_leverage']}x")
        logger.info(f"Per-level notional: ${self.config['risk']['per_level_notional_usd']}")

        while True:
            try:
                self.cycle()
            except KeyboardInterrupt:
                logger.info("KeyboardInterrupt — shutting down")
                break
            except Exception as e:
                logger.exception(f"Cycle error: {e}")
                self.consecutive_api_errors += 1
            time.sleep(poll_interval)
