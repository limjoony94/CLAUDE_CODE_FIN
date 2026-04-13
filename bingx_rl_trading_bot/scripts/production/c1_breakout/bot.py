"""
C1 Breakout v2.5 — Bot Main Loop
==================================
BTC/USDT 15m, N=1, One-Way mode, crash-safe.
Exchange leverage 10x, trading leverage 3x.

v2.5 (2026-04-13): 30-Cycle critical review
  - check_exit priority: SL before Emergency (realistic order)
  - Backtest fractal lookback unified to 10 (match production)
  - Leverage: exchange 10x / trading 3x separation
  - Trail orders reduceOnly (prevent reverse position)
  - Ghost trade lookup 24h window
  - Leverage verification on startup (abort if mismatch)
  - Halt brakes removed (per user)

BUG#35 (2026-04-14): TRAILING_STOP_MARKET priceRate fix
  - Root cause: 'priceRate' param bypasses CCXT ÷100 conversion
    CCXT converts priceRate internally BUT extend(request,params) overwrites it
    with original priceRate=0.9 → BingX interprets as 90% callback
  - Fix: use 'trailingPercent' (in CCXT omit list, survives conversion)
  - Impact: trigger was at best×0.1 (~$7,212) instead of best×0.991 (~$71,778)
"""

import os
import sys
import json
import time
import logging
import ccxt
import yaml
import requests
from datetime import datetime, timedelta
from pathlib import Path

from .signals import C1BreakoutSignal
from .indicators import compute_atr, compute_channel, compute_fractal_swings
from .config import load_config

logger = logging.getLogger('c1_breakout')


class TimeSyncBingX(ccxt.bingx):
    """BingX with automatic server time synchronization."""
    _time_offset = 0
    _last_sync = 0

    def milliseconds(self):
        # Re-sync every 5 minutes
        now = time.time()
        if now - self._last_sync > 300:
            try:
                resp = requests.get(
                    'https://open-api.bingx.com/openApi/swap/v2/server/time',
                    timeout=5)
                server_ms = resp.json().get('data', {}).get('serverTime', 0)
                if server_ms:
                    self._time_offset = server_ms - int(now * 1000)
                    self._last_sync = now
                    logger.debug(f"Time sync: offset={self._time_offset}ms")
            except Exception:
                pass
        return int(time.time() * 1000) + self._time_offset


class C1BreakoutBot:
    def __init__(self, config_path='config/c1_breakout_config.yaml'):
        self.config = load_config(config_path)
        self.signal = C1BreakoutSignal(self.config['strategy'])
        self.max_positions = self.config['strategy'].get('max_positions', 1)

        self.state_path = 'results/c1_breakout_state.json'
        self.positions = []
        self.trade_history = []
        self.bars_since_last_exit = 999  # BUG#16: min_bars_between enforcement
        # BUG#35: Force trail re-placement on first cycle after restart
        # Ensures any wrong priceRate (90%) orders are replaced with correct trailingPercent (0.9%)
        self._force_trail_reset = True

        self._init_exchange()
        self._load_state()
        self._sync_exchange()

    @property
    def size_pct(self):
        return 100.0 / self.max_positions

    def _init_exchange(self):
        api_path = 'config/api_keys.yaml'
        if not os.path.exists(api_path):
            logger.warning("No API keys — dry run"); self.exchange = None; return
        with open(api_path) as f:
            keys = yaml.safe_load(f)
        bk = keys.get('bingx', keys)
        if isinstance(bk, dict) and 'mainnet' in bk: bk = bk['mainnet']
        api_key = bk.get('api_key', '')
        secret = bk.get('secret_key', '')
        if api_key and secret:
            self.exchange = TimeSyncBingX({
                'apiKey': api_key, 'secret': secret,
                'options': {'defaultType': 'swap'},
            })
            # Force initial time sync
            self.exchange.milliseconds()
            logger.info(f"Exchange connected (time offset: {self.exchange._time_offset}ms)")
            # Set leverage on exchange — verify it matches config
            target_lev = self.config['exchange'].get('leverage', 1)
            try:
                self.exchange.set_leverage(target_lev, self.config['exchange']['symbol'],
                                           params={'side': 'BOTH'})
                logger.info(f"Leverage set to {target_lev}x")
            except Exception as e:
                # set_leverage can fail if position exists — verify current leverage
                logger.warning(f"set_leverage({target_lev}x) failed: {e}")
                try:
                    raw = self.exchange.fetch_positions([self.config['exchange']['symbol']])
                    cur_lev = None
                    for ep in raw:
                        lev_val = ep.get('leverage') or ep.get('info', {}).get('leverage')
                        if lev_val is not None:
                            cur_lev = int(float(lev_val))
                            break
                    if cur_lev == target_lev:
                        logger.info(f"Leverage already {cur_lev}x — OK")
                    elif cur_lev is not None:
                        logger.error(f"CRITICAL: Exchange leverage={cur_lev}x != config={target_lev}x — aborting")
                        sys.exit(1)
                    else:
                        # No positions → can't verify, set_leverage likely failed transiently
                        logger.warning(f"Cannot verify leverage (no positions) — assuming {target_lev}x OK")
                except Exception as e2:
                    logger.warning(f"Leverage verification failed: {e2} — will retry on next restart")
        else:
            logger.warning("API keys empty — dry run"); self.exchange = None

    def _load_state(self):
        if not os.path.exists(self.state_path): return
        try:
            with open(self.state_path) as f:
                state = json.load(f)
        except (json.JSONDecodeError, ValueError) as e:
            # BUG#25: Corrupted state file — start fresh, positions recovered via orphan sync
            logger.error(f"State file corrupted: {e} — starting fresh")
            return
        self.positions = state.get('positions') or []
        self.trade_history = state.get('trade_history', [])
        self.bars_since_last_exit = state.get('bars_since_last_exit', 999)
        logger.info(f"State: {len(self.positions)} pos, {len(self.trade_history)} trades")

    def _save_state(self):
        Path(self.state_path).parent.mkdir(parents=True, exist_ok=True)
        # BUG#25: Atomic write — write to temp, then rename (crash-safe)
        tmp_path = self.state_path + '.tmp'
        data = {
            'positions': self.positions,
            'trade_history': self.trade_history[-500:],
            'bars_since_last_exit': self.bars_since_last_exit,
            'updated': datetime.utcnow().isoformat(),
        }
        with open(tmp_path, 'w') as f:
            json.dump(data, f, indent=2)
        # Atomic rename (on Windows: replaces existing)
        os.replace(tmp_path, self.state_path)

    # ── Exchange Sync (SAFE: no ghost on API error) ───────

    def _get_live_positions(self):
        """Returns {direction: info} or None on API error."""
        if not self.exchange: return {}
        try:
            raw = self.exchange.fetch_positions([self.config['exchange']['symbol']])
            live = {}
            for ep in raw:
                contracts = float(ep.get('contracts') or 0)
                if contracts > 0:
                    side = (ep.get('side') or '').upper()
                    if side in ('LONG', 'SHORT'):
                        live[side] = {
                            'entry_price': float(ep.get('entryPrice') or 0),
                            'contracts': contracts,
                        }
                    elif side:
                        logger.warning(f"Unknown position side '{side}' with {contracts} contracts")
            return live
        except Exception as e:
            logger.error(f"Position fetch FAILED: {e}")
            return None  # None = API error, distinct from {} = no positions

    def _sync_exchange(self):
        live = self._get_live_positions()

        # API error → DO NOT touch local state
        if live is None:
            logger.warning("Sync SKIPPED — API error (will retry next cycle)")
            return

        logger.info(f"Sync: exchange={list(live.keys())}, "
                    f"local={[p['direction'] for p in self.positions]}")
        changed = False

        # Ghost: local has it, exchange doesn't → closed by exchange SL/Trail
        for i in range(len(self.positions) - 1, -1, -1):
            pos = self.positions[i]
            d = pos['direction']
            if d not in live:
                # BUG#21: Try to get actual exit from recent trades
                exit_price, reason = self._resolve_ghost_exit(pos)

                if d == 'LONG':
                    est_pnl = (exit_price / pos['entry_price'] - 1) * 100
                else:
                    est_pnl = (1 - exit_price / pos['entry_price']) * 100

                trading_lev = self.config['exchange'].get('trading_leverage',
                              self.config['exchange'].get('leverage', 1))
                pnl = (est_pnl - 0.10) * trading_lev
                self.trade_history.append({
                    'direction': d, 'entry_price': pos['entry_price'],
                    'exit_price': exit_price,
                    'pnl_pct': round(pnl, 4), 'reason': reason,
                    'bars_held': pos.get('bars_held', 0),
                    'exit_time': datetime.utcnow().isoformat(),
                })
                self.bars_since_last_exit = 0

                logger.warning(f"GHOST: {d} @ ${pos['entry_price']:.2f} → "
                               f"{reason} exit=${exit_price:.2f} PnL={pnl:+.2f}%")
                self.positions.pop(i); changed = True

        # Orphan: exchange has it, local doesn't
        local_dirs = {p['direction'] for p in self.positions}
        for side, info in live.items():
            if side not in local_dirs and len(self.positions) < self.max_positions:
                logger.warning(f"ORPHAN: {side} @ ${info['entry_price']:.2f} adopted")
                ep = info['entry_price']
                emg = self.config['strategy']['emergency_sl_pct']
                sl = ep * (1-emg/100) if side=='LONG' else ep * (1+emg/100)
                self.positions.append({
                    'direction': side, 'entry_price': ep, 'sl_price': sl,
                    'best_price': ep, 'entry_time': datetime.utcnow().isoformat(),
                    'bars_held': 0, 'size_pct': self.size_pct,
                })
                changed = True

        # Clean orphan orders when no positions exist
        if not live and not self.positions:
            try:
                orders = self.exchange.fetch_open_orders(self.config['exchange']['symbol'])
                for order in orders:
                    self.exchange.cancel_order(order['id'], self.config['exchange']['symbol'])
                if orders:
                    logger.info(f"Cleaned {len(orders)} orphan orders")
            except Exception:
                pass

        if changed: self._save_state()
        else: logger.info("Sync: OK")

    def _resolve_ghost_exit(self, pos):
        """Determine actual exit price and reason for a ghost position.

        Tries fetch_my_trades first (accurate), falls back to SL price estimate.
        Returns (exit_price, reason).
        """
        d = pos['direction']
        entry_price = pos['entry_price']
        sl_price = pos['sl_price']

        # Try to find the closing trade from exchange history
        try:
            symbol = self.config['exchange']['symbol']
            # Fetch recent trades (last 24 hours — covers long bot downtime)
            since = int((time.time() - 86400) * 1000)
            trades = self.exchange.fetch_my_trades(symbol, since=since, limit=50)
            # Find closing trade: opposite side of position
            close_side = 'sell' if d == 'LONG' else 'buy'
            for t in reversed(trades):
                if (t.get('side', '').lower() == close_side
                        and float(t.get('amount', 0)) > 0):
                    exit_price = float(t['price'])
                    # Determine reason by comparing exit to SL
                    if d == 'LONG':
                        near_sl = abs(exit_price - sl_price) / entry_price < 0.003
                    else:
                        near_sl = abs(exit_price - sl_price) / entry_price < 0.003
                    reason = 'EXCHANGE_SL' if near_sl else 'EXCHANGE_TRAIL'
                    logger.info(f"Ghost resolved via trade history: exit=${exit_price:.2f}")
                    return exit_price, reason
        except Exception as e:
            logger.debug(f"fetch_my_trades failed: {e}")

        # Fallback: use SL price (conservative)
        return sl_price, 'EXCHANGE_SL'

    # ── Candle Fetch ──────────────────────────────────────

    def fetch_candles(self):
        if not self.exchange: return None
        try:
            ohlcv = self.exchange.fetch_ohlcv(
                self.config['exchange']['symbol'], '15m',
                limit=self.config['exchange'].get('candle_bars_fetch', 100))  # BUG#42: was config['bot'] — wrong section
            if not ohlcv or len(ohlcv) < 30: return None
            # Stale data guard: check if last completed bar timestamp is new
            last_ts = ohlcv[-2][0]  # n-2 = last completed bar
            if hasattr(self, '_last_bar_ts') and last_ts == self._last_bar_ts:
                logger.warning(f"Stale candle data (same bar ts={last_ts}) — skip")
                return None
            self._last_bar_ts = last_ts
            return {
                'open': [x[1] for x in ohlcv], 'high': [x[2] for x in ohlcv],
                'low': [x[3] for x in ohlcv], 'close': [x[4] for x in ohlcv],
            }
        except Exception as e:
            logger.error(f"Candle fetch: {e}"); return None

    # ── Core Logic ────────────────────────────────────────

    def process_candles(self, candles):
        n = len(candles['close']); bar = n - 2
        if bar < 25: return
        cfg = self.config['strategy']
        atr = compute_atr(candles['high'], candles['low'], candles['close'], cfg['atr_period'])
        ch_h, ch_l = compute_channel(candles['high'], candles['low'], cfg['channel_period'])
        sw_l, sw_h = compute_fractal_swings(candles['high'], candles['low'])
        cur_atr = atr[bar]

        # Track bars since last exit (BUG#16)
        self.bars_since_last_exit += 1

        # Exits + Trail Update
        for i in range(len(self.positions) - 1, -1, -1):
            pos = self.positions[i]
            pos['bars_held'] = pos.get('bars_held', 0) + 1
            if pos['direction'] == 'LONG':
                pos['best_price'] = max(pos.get('best_price', pos['entry_price']), candles['high'][bar])
            else:
                pos['best_price'] = min(pos.get('best_price', pos['entry_price']), candles['low'][bar])

            # Bot trail check (backtest-identical logic)
            ex = self.signal.check_exit(
                pos['direction'], pos['entry_price'], pos['best_price'],
                candles['high'][bar], candles['low'][bar], candles['close'][bar],
                pos['sl_price'], cur_atr, pos['bars_held'])
            if ex:
                self._do_close(i, ex)
            elif self.exchange:
                # Update exchange trailing stop with current ATR (backtest parity)
                self._update_exchange_trail(pos, candles['close'][bar], cur_atr)

        # Entries — enforce min_bars_between (BUG#16: backtest parity)
        min_bars = cfg.get('min_bars_between', 2)
        if (len(self.positions) < self.max_positions
                and self.bars_since_last_exit >= min_bars):
            sig = self.signal.check_entry(
                candles['open'][bar], candles['high'][bar], candles['low'][bar],
                candles['close'][bar], ch_h[bar], ch_l[bar], cur_atr, sw_l[bar], sw_h[bar])
            if sig:
                self._do_open(sig, candles['close'][bar], cur_atr)
        self._save_state()

    def _do_open(self, signal, price, atr_val):
        d = signal['direction']; sl = signal['sl_price']
        pos = {
            'direction': d, 'entry_price': price, 'sl_price': sl,
            'best_price': price, 'entry_time': datetime.utcnow().isoformat(),
            'bars_held': 0, 'size_pct': self.size_pct,
        }
        self.positions.append(pos)
        logger.info(f"ENTRY {d} @ ${price:.2f} | SL=${sl:.2f} ({signal['sl_pct']:.2f}%) | ATR=${atr_val:.2f}")

        if self.exchange:
            success = self._exchange_open(d, price, sl, atr_val)
            if not success:
                logger.error("ORDER FAILED — rollback")
                self.positions.pop()

    def _do_close(self, idx, exit_signal):
        pos = self.positions[idx]; d = pos['direction']; xp = exit_signal['exit_price']
        if d == 'LONG': raw_pct = (xp / pos['entry_price'] - 1) * 100
        else: raw_pct = (1 - xp / pos['entry_price']) * 100
        trading_lev = self.config['exchange'].get('trading_leverage',
                      self.config['exchange'].get('leverage', 1))
        pnl = (raw_pct - 0.10) * trading_lev

        self.trade_history.append({
            'direction': d, 'entry_price': pos['entry_price'], 'exit_price': xp,
            'pnl_pct': round(pnl, 4), 'reason': exit_signal['reason'],
            'bars_held': pos['bars_held'], 'exit_time': datetime.utcnow().isoformat(),
        })
        self.bars_since_last_exit = 0  # BUG#16: reset for min_bars_between

        logger.info(f"EXIT {d} {exit_signal['reason']} | PnL={pnl:+.2f}% | "
                    f"Hold={pos['bars_held']}b")
        if self.exchange: self._exchange_close(d)
        self.positions.pop(idx)

    # ── Exchange Orders (One-Way, TimeSynced) ─────────────

    def _get_balance(self):
        try:
            bal = self.exchange.fetch_balance()
            usdt = float(bal.get('USDT', {}).get('free', 0))
            logger.info(f"Balance: ${usdt:.2f}")
            return usdt
        except Exception as e:
            logger.error(f"Balance: {e}"); return 0

    def _calc_amount(self, price):
        usdt = self._get_balance()
        if usdt < 10:  # BUG#22: minimum $10 balance required
            logger.warning(f"Balance too low: ${usdt:.2f}")
            return 0
        # trading_leverage = actual sizing (3x), exchange leverage = max allowed (10x)
        trading_lev = self.config['exchange'].get('trading_leverage',
                      self.config['exchange'].get('leverage', 1))
        # BUG#31: Use 98% of balance — leave 2% for fees + maintenance margin
        qty = usdt * 0.98 * self.size_pct / 100.0 * trading_lev / price
        return round(qty, 4)

    def _exchange_open(self, direction, price, sl_price, atr_val):
        """MARKET entry + SL + Trailing Stop. Returns True/False."""
        symbol = self.config['exchange']['symbol']
        side = 'buy' if direction == 'LONG' else 'sell'
        qty = self._calc_amount(price)
        if qty <= 0: logger.error("No balance"); return False
        fill_price = price
        market_filled = False

        try:
            # 1. MARKET entry
            order = self.exchange.create_order(symbol, 'market', side, qty,
                params={'positionSide': 'BOTH'})
            market_filled = True
            # BUG#18: Get actual fill price and update entry_price
            fill_price = float(order.get('average') or order.get('price') or price)
            filled_qty = float(order.get('filled') or order.get('amount') or qty)
            if fill_price > 0 and self.positions:
                self.positions[-1]['entry_price'] = fill_price
                slip = (fill_price / price - 1) * 100
                if abs(slip) > 0.01:
                    logger.info(f"Slippage: {slip:+.3f}% (signal={price:.1f} fill={fill_price:.1f})")
            logger.info(f"MARKET {direction} qty={filled_qty} fill=${fill_price:.1f}")

            # 2. SL (STOP_MARKET) — use actual filled qty (BUG#28)
            sl_side = 'sell' if direction == 'LONG' else 'buy'
            sl_result = self.exchange.create_order(symbol, 'STOP_MARKET', sl_side, filled_qty,
                params={'positionSide': 'BOTH', 'stopPrice': round(sl_price, 1),
                        'reduceOnly': True})
            if self.positions:
                self.positions[-1]['sl_order_id'] = sl_result.get('id', '')
            logger.info(f"SL @ ${sl_price:.1f}")

            # 3. Trail TP — exchange native TRAILING_STOP_MARKET (backup)
            # Bot's check_exit is primary (backtest math, 15m check).
            # Exchange trail is backup when bot is down.
            trail_K = self.config['strategy'].get('trail_K', 2.5)
            ref_price = fill_price if fill_price > 0 else price
            atr_pct = atr_val / ref_price * 100
            callback = round(max(0.1, min(5.0, trail_K * atr_pct)), 1)
            try:
                tp_side = 'sell' if direction == 'LONG' else 'buy'
                activate = round(ref_price * (1 + 0.001) if direction == 'LONG'
                                 else ref_price * (1 - 0.001), 1)
                self.exchange.create_order(
                    symbol, 'TRAILING_STOP_MARKET', tp_side, filled_qty,
                    params={'positionSide': 'BOTH',
                            'activatePrice': activate,
                            'trailingPercent': callback,  # BUG#35: priceRate bypasses CCXT ÷100 conversion
                            'reduceOnly': True})
                pos_obj = self.positions[-1] if self.positions else None
                if pos_obj:
                    pos_obj['last_callback'] = callback
                logger.info(f"Trail TP: callback={callback}% activate=${activate:.1f}")
            except Exception as e:
                logger.warning(f"Trail TP on exchange failed (bot will manage): {e}")

            return True
        except Exception as e:
            logger.error(f"Order error: {e}")
            # BUG#26: If MARKET filled but SL failed, close the position immediately
            if market_filled:
                logger.error("MARKET filled but SL/Trail failed — emergency close")
                try:
                    close_side = 'sell' if direction == 'LONG' else 'buy'
                    live = self._get_live_positions()
                    if live and direction in live:
                        self.exchange.create_order(symbol, 'market', close_side,
                            live[direction]['contracts'],
                            params={'positionSide': 'BOTH', 'reduceOnly': True})
                        logger.info("Emergency close completed")
                except Exception as e2:
                    logger.error(f"Emergency close FAILED: {e2} — orphan on exchange!")
            return False

    def _calc_trail_trigger_price(self, pos, cur_atr):
        """Compute exact trail trigger price using backtest-identical math.

        Not used for exchange orders (exchange uses native TRAILING_STOP_MARKET).
        Kept for analysis/debugging. Bot's check_exit handles trail internally.

        Returns trigger price or None if trail not yet active.
        """
        import math
        d = pos['direction']
        ep = pos['entry_price']
        bp = pos.get('best_price', ep)
        trail_K = self.config['strategy'].get('trail_K', 2.5)
        activation = self.config['strategy'].get('trail_activation_pct', 0.05)

        if math.isnan(cur_atr) or cur_atr <= 0:
            return None

        # Check activation
        if d == 'LONG':
            best_pnl = (bp / ep - 1) * 100
        else:
            best_pnl = (1 - bp / ep) * 100
        if best_pnl <= activation:
            return None

        k_atr = trail_K * cur_atr

        if d == 'LONG':
            # cur² - best·cur + k_atr·entry = 0
            # Upper root = trigger price (below best, above entry)
            disc = bp * bp - 4 * k_atr * ep
            if disc < 0:
                return None
            trigger = (bp + math.sqrt(disc)) / 2
            # Trail only tightens: trigger must be above fractal SL
            if trigger <= pos['sl_price']:
                return None
            return round(trigger, 1)
        else:
            # cur² - best·cur - k_atr·entry = 0
            # Upper root = trigger price (above best, for SHORT = price going up)
            disc = bp * bp + 4 * k_atr * ep
            trigger = (bp + math.sqrt(disc)) / 2
            # Trail only tightens: trigger must be below fractal SL
            if trigger >= pos['sl_price']:
                return None
            return round(trigger, 1)

    def _update_exchange_trail(self, pos, cur_price, cur_atr):
        """Update exchange TRAILING_STOP_MARKET callback rate with current ATR.

        Also verifies fractal SL STOP still exists — re-places if missing.
        Bot's check_exit is the primary trail mechanism (backtest math).
        Exchange TRAILING_STOP_MARKET is backup when bot is down.
        """
        try:
            import math
            if math.isnan(cur_atr) or cur_atr <= 0:
                return

            symbol = self.config['exchange']['symbol']
            trail_K = self.config['strategy'].get('trail_K', 2.5)
            new_callback = round(max(0.1, min(5.0, trail_K * cur_atr / cur_price * 100)), 1)

            orders = self.exchange.fetch_open_orders(symbol)

            # ── Verify SL STOP exists — re-place if missing ──
            sl_order_id = pos.get('sl_order_id', '')
            sl_found = any(o.get('id', '') == sl_order_id for o in orders) if sl_order_id else False
            if not sl_found:
                # BUG#36: Price-based fallback — must update sl_order_id to prevent accidental cancellation
                sl_price = pos['sl_price']
                for o in orders:
                    if 'TRAILING' in ((o.get('info') or {}).get('type', '') or '').upper():
                        continue
                    sp = float(o.get('stopPrice') or o.get('info', {}).get('stopPrice') or 0)
                    if abs(sp - sl_price) < 1.0:
                        sl_found = True
                        pos['sl_order_id'] = o.get('id', sl_order_id)  # Update ID to prevent cancel
                        logger.info(f"SL found by price (ID mismatch) — updated sl_order_id")
                        break
            if not sl_found:
                live = self._get_live_positions()
                if live and pos['direction'] in live:
                    sl_side = 'sell' if pos['direction'] == 'LONG' else 'buy'
                    sl_result = self.exchange.create_order(
                        symbol, 'STOP_MARKET', sl_side, live[pos['direction']]['contracts'],
                        params={'positionSide': 'BOTH',
                                'stopPrice': round(pos['sl_price'], 1),
                                'reduceOnly': True})
                    pos['sl_order_id'] = sl_result.get('id', '')
                    logger.warning(f"SL STOP was missing — re-placed @ ${pos['sl_price']:.1f}")

            # ── Update TRAILING_STOP_MARKET callback ──
            old_callback = pos.get('last_callback', 0)
            # BUG#35: On first cycle after restart, force re-placement to fix wrong priceRate orders
            force_reset = getattr(self, '_force_trail_reset', False)
            if force_reset:
                self._force_trail_reset = False  # Only once per bot session
                logger.info("Trail: forcing re-placement (BUG#35 priceRate fix)")
            elif abs(new_callback - old_callback) < 0.1:
                return  # No meaningful change

            # Cancel existing trailing stop (TRAILING_STOP_MARKET or legacy STOP_MARKET trail)
            for order in orders:
                oid = order.get('id', '')
                if oid == pos.get('sl_order_id', ''):
                    continue  # Never cancel the fractal SL
                o_type = (order.get('info') or {}).get('type', '') or order.get('type', '')
                is_trailing = 'TRAILING' in o_type.upper()
                # Also cancel non-SL STOP_MARKET (legacy managed trail from v2.5 transition)
                is_non_sl_stop = o_type.upper() == 'STOP_MARKET'
                if is_trailing or is_non_sl_stop:
                    try:
                        self.exchange.cancel_order(oid, symbol)
                    except Exception:
                        pass

            # Place new TRAILING_STOP_MARKET with updated callback
            live = self._get_live_positions()
            if not live or pos['direction'] not in live:
                return

            qty = live[pos['direction']]['contracts']
            tp_side = 'sell' if pos['direction'] == 'LONG' else 'buy'
            activate = round(cur_price * (1 + 0.001) if pos['direction'] == 'LONG'
                             else cur_price * (1 - 0.001), 1)

            self.exchange.create_order(
                symbol, 'TRAILING_STOP_MARKET', tp_side, qty,
                params={'positionSide': 'BOTH',
                        'activatePrice': activate,
                        'trailingPercent': new_callback,  # BUG#35: use trailingPercent (CCXT omits it after ÷100)
                        'reduceOnly': True})

            pos['last_callback'] = new_callback
            logger.info(f"Trail TP updated: callback={new_callback}% (ATR=${cur_atr:.0f})")

        except Exception as e:
            logger.warning(f"Trail update failed: {e}")

    def _exchange_close(self, direction):
        try:
            symbol = self.config['exchange']['symbol']
            side = 'sell' if direction == 'LONG' else 'buy'
            live = self._get_live_positions()
            if live and direction in live:
                self.exchange.create_order(symbol, 'market', side,
                    live[direction]['contracts'],
                    params={'positionSide': 'BOTH', 'reduceOnly': True})
            # Cancel all remaining orders (individually, so one failure doesn't stop others)
            try:
                for order in self.exchange.fetch_open_orders(symbol):
                    try:
                        self.exchange.cancel_order(order['id'], symbol)
                    except Exception:
                        pass
            except Exception:
                pass
            logger.info(f"Closed {direction}")
        except Exception as e:
            logger.error(f"Close: {e}")

    # ── Main Loop ─────────────────────────────────────────

    def _wait_for_candle(self):
        now = datetime.utcnow()
        next_min = 15 - (now.minute % 15)
        if next_min == 15: next_min = 0
        wait = next_min * 60 - now.second + 5
        # BUG#37: was `<= 5` — skipped valid bars when 0-5 sec past boundary
        # Fix: only skip when already past the 5-sec buffer window (wait < 0)
        if wait <= 0: wait += 900
        logger.info(f"Next: {(now+timedelta(seconds=wait)).strftime('%H:%M:%S')} UTC ({wait}s)")
        time.sleep(wait)

    def run(self):
        ex_lev = self.config['exchange'].get('leverage', 1)
        tr_lev = self.config['exchange'].get('trading_leverage', ex_lev)
        logger.info(f"C1 Breakout v2.5 (N={self.max_positions}, exchange={ex_lev}x, trading={tr_lev}x)")
        self._last_hourly = -1

        while True:
            try:
                self._wait_for_candle()
                self._sync_exchange()  # BUG#38: unconditional — orphan detect works even when no local positions
                candles = self.fetch_candles()
                if candles: self.process_candles(candles)
                logger.info(f"Cycle: pos={len(self.positions)}")

                # Hourly status summary
                now = datetime.utcnow()
                if now.hour != self._last_hourly:
                    self._last_hourly = now.hour
                    trades_today = [t for t in self.trade_history
                                    if t.get('exit_time', '')[:10] == now.strftime('%Y-%m-%d')]
                    wins = sum(1 for t in trades_today if t['pnl_pct'] > 0)
                    total = len(trades_today)
                    wr = (wins / total * 100) if total else 0
                    pos_info = ', '.join(
                        f"{p['direction']} @{p['entry_price']:.0f} ({p['bars_held']}b)"
                        for p in self.positions) or 'none'
                    logger.info(f"HOURLY | trades={total} WR={wr:.0f}% "
                                f"pos=[{pos_info}]")
            except KeyboardInterrupt:
                logger.info("Stopped"); self._save_state(); break
            except Exception as e:
                logger.error(f"Loop: {e}", exc_info=True); self._save_state(); time.sleep(60)
