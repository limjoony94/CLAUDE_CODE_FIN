"""
C1 Breakout v2.6 — Bot Main Loop
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

BUG#36 (2026-04-14): Ghost resolution timestamp filter
  - Root cause: _resolve_ghost_exit searched all recent sells regardless of timing
    LONG close → SHORT open (also 'sell') → bot crash → restart resolves ghost
    using SHORT entry price as LONG exit price
  - Fix: only match closing trades AFTER position entry_time

BUG#37 (2026-04-14): Trail replacement direction (tighten-only)
  - Root cause: trail replaced on ANY callback change, including when ATR rises
    Cancel old TRAILING_STOP_MARKET (tracking best_price) + re-place at lower
    activatePrice = cur_price×1.001 → BingX tracking resets → protection gap
  - Fix: only replace when new_callback < old_callback (tighten only)
    Forced reset on startup still replaces regardless

BUG#38 (2026-04-14): Insufficient margin retry
  - Root cause: single attempt at full size, no fallback on 101253 error
    After large loss, balance may be marginally insufficient at 98% sizing
  - Fix: retry with 95% then 90% sizing on Insufficient margin error

BUG#43 (2026-04-14): _force_trail_reset when no position
  - Root cause: flag always True at startup, persists when no position exists
    New entry → first trail update forces cancel+re-place of just-placed trail
    → 15 min protection gap where BingX tracking is reset
  - Fix: clear flag after _load_state() if self.positions is empty

BUG#44 (2026-04-14): Balance cache in retry loop
  - Root cause: _calc_amount calls _get_balance() (API call) on every retry
    3 API calls in rapid succession for same balance value
  - Fix: cache balance once before retry loop in _exchange_open

BUG#45 (2026-04-15): Ghost exit_time uses detection time, not exchange time
  - Root cause: trade_history recorded datetime.utcnow() (ghost detection)
    instead of actual exchange execution timestamp from fetch_my_trades
    Up to 15 min error — affects daily PnL aggregation at day boundaries
  - Fix: _resolve_ghost_exit returns exchange trade timestamp; ghost handler uses it

BUG#46 (2026-04-15): Trail update resets BingX best_price tracking
  - Root cause: TRAILING_STOP_MARKET cancel+re-place destroys BingX's native
    best_price memory. New order needs activatePrice re-reached before tracking.
  - Fix: asymmetric update policy:
    LOOSEN only (ATR↑ > 0.1pp): re-place with wider callback. Tracking resets
      but wider trail is far from best → low premature trigger risk.
      Prevents frozen tight trail from cutting winners short.
    TIGHTEN never: bot's check_exit handles tightening with current ATR
      (backtest-identical, runs every 15m). Exchange tracking preserved.

BUG#48 (2026-04-17): Orphan adoption discarded actual SL (Opus 4.7 review)
  - Root cause: _sync_exchange orphan path always wrote emergency_sl_pct (3%)
    into local sl_price. If original fractal SL was 0.5%, on restart the bot
    would treat SL as 3% and _update_exchange_trail would place a NEW SL at
    3% (loose) since the 0.5% exchange SL didn't match.
  - Fix: _resolve_orphan_sl() queries exchange for live reduceOnly STOP orders
    on the correct side, picks the tightest (closest to entry), and restores
    both sl_price and sl_order_id. 3% fallback only when truly no SL exists
    (genuine crash between MARKET fill and SL placement).

BUG#49 (2026-04-17): fill_price slippage could push sl_pct out of bounds
  - Root cause: signal SL was sized against bar_close; fill_price differs by
    slippage. After BUG#18 entry_price update, sl_pct relative to fill could
    fall below sl_min_pct or exceed sl_max_pct without any warning.
  - Fix: warn-only — fractal SL is absolute (market structure), but log when
    actual sl_pct vs fill violates configured bounds for monitoring.

BUG#50 (2026-04-17): Ghost exit reason used stale best_price (classification only)
  - Root cause: BUG#47 distance heuristic compared exit price to est_trail
    derived from last_callback × local best_price. During offline periods,
    actual extremes moved beyond bot's state → misclassification.
  - Fix: prefer explicit trade.info.orderType (STOP_MARKET vs TRAILING_STOP_MARKET).
    Distance-based fallback retained for legacy trades without order type info.

BUG#51 (2026-04-17): Silent candle fetch outage skipped tighten logic
  - Root cause: fetch_candles failure → process_candles skipped → check_exit
    skipped → ATR-based trail tightening not running. Exchange TRAILING stays
    at initial callback (BUG#46 policy) so tightening is bot's job; sustained
    outage leaves winners exposed.
  - Fix: _candle_fail_streak counter; warn at streak >= 3 with open position,
    >= 6 without — operator-visible signal for monitoring intervention.

BUG#52 (2026-04-17): No validation of leverage relationship
  - Root cause: trading_leverage > leverage silently allowed at config load.
    Actual sizing (qty = balance × trading_lev / price) exceeding exchange cap
    triggers immediate liquidation risk.
  - Fix: load_config raises ValueError when trading_leverage > leverage or
    either is non-positive. Also validates sl_min_pct < sl_max_pct.

BUG#53 (2026-04-17): Channel sanity check missing
  - Root cause: if channel_high <= channel_low (flat/inverted data), the
    breakout condition (close > channel_high OR close < channel_low) could
    still fire pathologically.
  - Fix: reject channel_high <= channel_low explicitly in check_entry.
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
        # Only meaningful when a position already exists at startup
        self._force_trail_reset = True

        self._init_exchange()
        self._load_state()

        # Clear force_trail_reset if no positions — avoids unnecessary trail
        # re-placement on the first entry after restart (BUG#43 protection gap)
        if not self.positions:
            self._force_trail_reset = False

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
        # Enrich positions with monitoring-friendly trail estimate
        positions_out = []
        for pos in self.positions:
            p = dict(pos)
            cb = p.get('last_callback', 0)
            bp = p.get('best_price', p.get('entry_price', 0))
            if cb > 0 and bp > 0:
                d = p.get('direction', 'LONG')
                p['trail_estimate'] = round(
                    bp * (1 - cb / 100) if d == 'LONG' else bp * (1 + cb / 100), 1)
                p['trail_type'] = 'TRAILING_STOP_MARKET'
            positions_out.append(p)
        data = {
            'positions': positions_out,
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
                exit_price, reason, exit_ts_ms = self._resolve_ghost_exit(pos)

                if d == 'LONG':
                    est_pnl = (exit_price / pos['entry_price'] - 1) * 100
                else:
                    est_pnl = (1 - exit_price / pos['entry_price']) * 100

                trading_lev = self.config['exchange'].get('trading_leverage',
                              self.config['exchange'].get('leverage', 1))
                pnl = (est_pnl - 0.10) * trading_lev
                # BUG#45: use actual exchange exit timestamp, fallback to detection time
                if exit_ts_ms:
                    exit_time = datetime.utcfromtimestamp(exit_ts_ms / 1000).isoformat()
                else:
                    exit_time = datetime.utcnow().isoformat()
                self.trade_history.append({
                    'direction': d, 'entry_price': pos['entry_price'],
                    'exit_price': exit_price,
                    'pnl_pct': round(pnl, 4), 'reason': reason,
                    'bars_held': pos.get('bars_held', 0),
                    'exit_time': exit_time,
                })
                self.bars_since_last_exit = 0

                logger.warning(f"GHOST: {d} @ ${pos['entry_price']:.2f} → "
                               f"{reason} exit=${exit_price:.2f} PnL={pnl:+.2f}%")
                self.positions.pop(i); changed = True

        # Orphan: exchange has it, local doesn't
        # BUG#48: restore actual SL from exchange instead of 3% emergency fallback
        #   Old: always used emergency_sl_pct (3%) → tighter original SL was discarded
        #   New: query exchange for live STOP_MARKET order; fallback to 3% only if
        #        genuinely no SL exists (crash between MARKET fill and SL placement)
        local_dirs = {p['direction'] for p in self.positions}
        for side, info in live.items():
            if side not in local_dirs and len(self.positions) < self.max_positions:
                ep = info['entry_price']
                resolved_sl, resolved_id = self._resolve_orphan_sl(side, ep)
                if resolved_sl is not None:
                    sl = resolved_sl
                    sl_src = f"exchange (id={resolved_id[:8]}...)" if resolved_id else "exchange"
                else:
                    emg = self.config['strategy']['emergency_sl_pct']
                    sl = ep * (1-emg/100) if side=='LONG' else ep * (1+emg/100)
                    resolved_id = ''
                    sl_src = f"emergency {emg}% fallback"
                logger.warning(
                    f"ORPHAN: {side} @ ${ep:.2f} adopted | SL=${sl:.2f} ({sl_src})")
                self.positions.append({
                    'direction': side, 'entry_price': ep, 'sl_price': sl,
                    'sl_order_id': resolved_id,
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

    def _resolve_orphan_sl(self, side, entry_price):
        """Query exchange for existing SL STOP order on orphan position.

        BUG#48: Restores actual SL (fractal-based) during crash recovery instead of
        forcing 3% emergency fallback. Only uses fallback when genuinely no SL exists
        (true orphan = crash between MARKET fill and SL placement).

        Filters:
          - reduceOnly = True (closing order)
          - side opposite to position (sell for LONG, buy for SHORT)
          - non-TRAILING (fractal SL is STOP_MARKET)
          - stopPrice on correct side of entry (LONG: below, SHORT: above)

        Selection: closest to entry (most conservative / already partially triggered).

        Returns:
            (stop_price, order_id) if found, (None, None) otherwise.
        """
        if not self.exchange:
            return None, None
        try:
            symbol = self.config['exchange']['symbol']
            orders = self.exchange.fetch_open_orders(symbol)
        except Exception as e:
            logger.warning(f"Orphan SL lookup failed: {e}")
            return None, None

        close_side = 'sell' if side == 'LONG' else 'buy'
        candidates = []
        for o in orders:
            info = o.get('info') or {}
            # Type: BingX may put it on o.type or info.type, various casings
            otype = str(info.get('type') or o.get('type') or '').upper()
            if 'TRAILING' in otype:
                continue  # trail is backup, not fractal SL
            if 'STOP' not in otype:
                continue
            # Side check (tolerate upper/lower case from different exchanges)
            o_side = str(o.get('side') or info.get('side') or '').lower()
            if o_side != close_side:
                continue
            # reduceOnly: tolerate bool/int/str representations
            ro_raw = o.get('reduceOnly')
            if ro_raw is None:
                ro_raw = info.get('reduceOnly')
            if ro_raw is None:
                ro_raw = info.get('reduceonly')
            ro = str(ro_raw).lower() in ('true', '1', 'yes')
            if not ro:
                continue
            # Stop price: BingX uses several aliases
            sp_raw = (o.get('stopPrice')
                      or info.get('stopPrice')
                      or info.get('stopprice')
                      or info.get('triggerPrice')
                      or info.get('stop_price'))
            try:
                sp = float(sp_raw)
            except (TypeError, ValueError):
                continue
            if sp <= 0:
                continue
            # Directional sanity: LONG SL below entry, SHORT SL above entry
            if side == 'LONG' and sp >= entry_price:
                continue
            if side == 'SHORT' and sp <= entry_price:
                continue
            candidates.append((sp, o.get('id', '')))

        if not candidates:
            return None, None
        # Most conservative: closest to entry
        if side == 'LONG':
            # LONG SL below entry → higher stopPrice = closer to entry = tighter
            candidates.sort(key=lambda x: -x[0])
        else:
            # SHORT SL above entry → lower stopPrice = closer to entry = tighter
            candidates.sort(key=lambda x: x[0])
        if len(candidates) > 1:
            logger.warning(
                f"Orphan SL: {len(candidates)} STOP candidates — picking tightest")
        return candidates[0]

    def _resolve_ghost_exit(self, pos):
        """Determine actual exit price and reason for a ghost position.

        Tries fetch_my_trades first (accurate), falls back to SL price estimate.
        Returns (exit_price, reason).
        """
        d = pos['direction']
        entry_price = pos['entry_price']
        sl_price = pos['sl_price']

        # BUG#36: parse entry_time to filter trades — avoids matching a subsequent
        # open in the same direction (e.g. LONG close → SHORT open both use 'sell')
        entry_ts_ms = 0
        try:
            entry_str = pos.get('entry_time', '')
            if entry_str:
                from datetime import timezone
                dt = datetime.fromisoformat(entry_str.replace('Z', '+00:00'))
                if dt.tzinfo is None:
                    dt = dt.replace(tzinfo=timezone.utc)
                entry_ts_ms = int(dt.timestamp() * 1000)
        except Exception:
            pass

        # Try to find the closing trade from exchange history
        try:
            symbol = self.config['exchange']['symbol']
            # Fetch recent trades (last 24 hours — covers long bot downtime)
            since = int((time.time() - 86400) * 1000)
            trades = self.exchange.fetch_my_trades(symbol, since=since, limit=50)
            # Find closing trade: opposite side of position, AFTER position entry
            close_side = 'sell' if d == 'LONG' else 'buy'
            for t in reversed(trades):
                if t.get('side', '').lower() != close_side:
                    continue
                if float(t.get('amount', 0)) <= 0:
                    continue
                # BUG#36: skip trades at or before entry (could be prior position's close)
                t_ts = int(t.get('timestamp') or t.get('info', {}).get('time') or 0)
                if entry_ts_ms > 0 and t_ts <= entry_ts_ms:
                    continue
                exit_price = float(t['price'])
                # BUG#50: prefer authoritative order type from exchange trade info.
                #   Old (BUG#47): distance comparison using bot's last_callback/best_price.
                #     Stale during offline periods (price made new extremes bot didn't see).
                #   New: inspect trade.info.orderType for explicit STOP_MARKET / TRAILING_STOP_MARKET.
                #     Fallback to distance comparison only when order type missing.
                t_info = t.get('info') or {}
                raw_type = str(
                    t_info.get('orderType') or t_info.get('type')
                    or t.get('type') or '').upper()
                reason = None
                if 'TRAILING' in raw_type:
                    reason = 'EXCHANGE_TRAIL'
                elif 'STOP' in raw_type:
                    reason = 'EXCHANGE_SL'
                if reason is None:
                    # Fallback: distance comparison (legacy BUG#47 logic)
                    dist_sl = abs(exit_price - sl_price)
                    trail_cb = pos.get('last_callback', 0)
                    bp = pos.get('best_price', entry_price)
                    if trail_cb > 0 and bp > 0:
                        if d == 'LONG':
                            est_trail = bp * (1 - trail_cb / 100)
                        else:
                            est_trail = bp * (1 + trail_cb / 100)
                        dist_trail = abs(exit_price - est_trail)
                        reason = 'EXCHANGE_SL' if dist_sl < dist_trail else 'EXCHANGE_TRAIL'
                    else:
                        reason = 'EXCHANGE_SL' if dist_sl / entry_price < 0.003 else 'EXCHANGE_TRAIL'
                # BUG#45: return actual exit timestamp from exchange (not detection time)
                exit_ts = t_ts if t_ts > 0 else None
                logger.info(f"Ghost resolved via trade history: exit=${exit_price:.2f}")
                return exit_price, reason, exit_ts
        except Exception as e:
            logger.debug(f"fetch_my_trades failed: {e}")

        # Fallback: use SL price (conservative)
        return sl_price, 'EXCHANGE_SL', None

    # ── Candle Fetch ──────────────────────────────────────

    def fetch_candles(self):
        """Fetch 15m OHLCV. Returns None on failure/stale — caller tracks via
        _candle_fail_streak.

        BUG#51: consecutive failures are monitored so operator can detect silent
        tighten-logic outage (check_exit skipped → ATR↓ tighten never runs;
        exchange TRAILING stays at initial callback per BUG#46 policy).
        """
        if not self.exchange:
            self._candle_fail_streak = getattr(self, '_candle_fail_streak', 0) + 1
            return None
        try:
            ohlcv = self.exchange.fetch_ohlcv(
                self.config['exchange']['symbol'], '15m',
                limit=self.config['exchange'].get('candle_bars_fetch', 100))  # BUG#42: was config['bot'] — wrong section
            if not ohlcv or len(ohlcv) < 30:
                self._candle_fail_streak = getattr(self, '_candle_fail_streak', 0) + 1
                return None
            # Stale data guard: check if last completed bar timestamp is new
            last_ts = ohlcv[-2][0]  # n-2 = last completed bar
            if hasattr(self, '_last_bar_ts') and last_ts == self._last_bar_ts:
                logger.warning(f"Stale candle data (same bar ts={last_ts}) — skip")
                self._candle_fail_streak = getattr(self, '_candle_fail_streak', 0) + 1
                return None
            self._last_bar_ts = last_ts
            self._candle_fail_streak = 0  # reset on success
            return {
                'open': [x[1] for x in ohlcv], 'high': [x[2] for x in ohlcv],
                'low': [x[3] for x in ohlcv], 'close': [x[4] for x in ohlcv],
            }
        except Exception as e:
            self._candle_fail_streak = getattr(self, '_candle_fail_streak', 0) + 1
            logger.error(f"Candle fetch: {e}")
            return None

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

    def _calc_amount(self, price, scale=1.0):
        usdt = self._get_balance()
        if usdt < 10:  # BUG#22: minimum $10 balance required
            logger.warning(f"Balance too low: ${usdt:.2f}")
            return 0
        # trading_leverage = actual sizing (3x), exchange leverage = max allowed (10x)
        trading_lev = self.config['exchange'].get('trading_leverage',
                      self.config['exchange'].get('leverage', 1))
        # BUG#31: Use 98% of balance — leave 2% for fees + maintenance margin
        # scale param allows retry with reduced sizing (BUG#38)
        qty = usdt * 0.98 * scale * self.size_pct / 100.0 * trading_lev / price
        return round(qty, 4)

    def _exchange_open(self, direction, price, sl_price, atr_val):
        """MARKET entry + SL + Trailing Stop. Returns True/False."""
        symbol = self.config['exchange']['symbol']
        side = 'buy' if direction == 'LONG' else 'sell'
        fill_price = price
        market_filled = False
        qty = 0

        try:
            # 1. MARKET entry (BUG#38: retry on insufficient margin with smaller size)
            # BUG#44: cache balance once — avoid 3× API calls on retry
            usdt = self._get_balance()
            if usdt < 10:
                logger.error(f"No balance (${usdt:.2f})"); return False
            trading_lev = self.config['exchange'].get('trading_leverage',
                          self.config['exchange'].get('leverage', 1))
            order = None
            for attempt, scale in enumerate((1.0, 0.95, 0.90), 1):
                attempt_qty = round(usdt * 0.98 * scale * self.size_pct / 100.0
                                    * trading_lev / price, 4)
                if attempt_qty <= 0:
                    break
                try:
                    order = self.exchange.create_order(symbol, 'market', side, attempt_qty,
                        params={'positionSide': 'BOTH'})
                    qty = attempt_qty
                    break
                except Exception as e_inner:
                    if '101253' in str(e_inner) and attempt < 3:
                        logger.warning(f"Insufficient margin (attempt {attempt}, "
                                       f"${usdt:.1f}×{scale}) — retrying smaller")
                        continue
                    raise  # re-raise non-margin errors or final attempt
            if order is None:
                raise RuntimeError("All sizing attempts failed")
            market_filled = True
            # BUG#18: Get actual fill price and update entry_price
            fill_price = float(order.get('average') or order.get('price') or price)
            filled_qty = float(order.get('filled') or order.get('amount') or qty)
            if fill_price > 0 and self.positions:
                self.positions[-1]['entry_price'] = fill_price
                slip = (fill_price / price - 1) * 100
                if abs(slip) > 0.01:
                    logger.info(f"Slippage: {slip:+.3f}% (signal={price:.1f} fill={fill_price:.1f})")
                # BUG#49: validate sl_pct against fill_price (not signal close)
                # SL price is fractal-based absolute — keep as-is, but warn if slippage
                # pushed actual sl_pct outside configured min/max bounds.
                cfg_s = self.config['strategy']
                actual_sl_pct = abs(fill_price - sl_price) / fill_price * 100
                lo = cfg_s.get('sl_min_pct', 0.15)
                hi = cfg_s.get('sl_max_pct', 3.0)
                if actual_sl_pct < lo or actual_sl_pct > hi:
                    logger.warning(
                        f"SL bounds after slippage: sl_pct={actual_sl_pct:.3f}% "
                        f"outside [{lo}%, {hi}%] — fill=${fill_price:.1f} "
                        f"sl=${sl_price:.1f} slip={slip:+.3f}%")
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
        """Verify exchange orders and manage trail protection.

        BUG#46: Trail is placed ONCE at entry and NEVER updated.
        Reason: BingX TRAILING_STOP_MARKET cancel+re-place RESETS best_price tracking.
        Each update destroys the trail's memory of the highest/lowest price seen,
        creating a protection gap where activatePrice must be re-reached.

        The exchange trail is crash protection (fires intrabar).
        The bot's check_exit is the primary mechanism (backtest-identical, current ATR).

        This function only:
        1. Verifies fractal SL STOP exists — re-places if missing
        2. Verifies trail exists — re-places if missing (same callback as entry)
        3. On first cycle after restart with legacy orders: one-time forced reset
        """
        try:
            import math
            if math.isnan(cur_atr) or cur_atr <= 0:
                return

            symbol = self.config['exchange']['symbol']
            orders = self.exchange.fetch_open_orders(symbol)

            # ── 1. Verify SL STOP exists — re-place if missing ──
            sl_order_id = pos.get('sl_order_id', '')
            sl_found = any(o.get('id', '') == sl_order_id for o in orders) if sl_order_id else False
            if not sl_found:
                sl_price = pos['sl_price']
                for o in orders:
                    if 'TRAILING' in ((o.get('info') or {}).get('type', '') or '').upper():
                        continue
                    sp = float(o.get('stopPrice') or o.get('info', {}).get('stopPrice') or 0)
                    if abs(sp - sl_price) < 1.0:
                        sl_found = True
                        pos['sl_order_id'] = o.get('id', sl_order_id)
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

            # ── 2. Verify trail exists — re-place if missing ──
            trail_exists = any(
                'TRAILING' in ((o.get('info') or {}).get('type', '') or '').upper()
                for o in orders)

            # BUG#35: On first cycle after restart, force re-placement for legacy priceRate orders
            force_reset = getattr(self, '_force_trail_reset', False)
            if force_reset:
                self._force_trail_reset = False
                if trail_exists:
                    # Cancel existing trail (may have wrong priceRate from old code)
                    for order in orders:
                        oid = order.get('id', '')
                        if oid == pos.get('sl_order_id', ''):
                            continue
                        o_type = (order.get('info') or {}).get('type', '') or order.get('type', '')
                        if 'TRAILING' in o_type.upper():
                            try:
                                self.exchange.cancel_order(oid, symbol)
                            except Exception:
                                pass
                    trail_exists = False
                    logger.info("Trail: forcing re-placement (BUG#35 legacy priceRate fix)")

            # ── 3. Re-place trail if missing, or LOOSEN if ATR rose significantly ──
            trail_K = self.config['strategy'].get('trail_K', 2.5)
            new_callback = round(max(0.1, min(5.0, trail_K * cur_atr / cur_price * 100)), 1)
            old_callback = pos.get('last_callback', 0)

            need_replace = False
            if not trail_exists:
                need_replace = True
                logger.warning(f"Trail STOP missing — will re-place")
            elif new_callback > old_callback + 0.1:
                # BUG#46: LOOSEN ONLY — ATR rose → exchange trail is too tight
                # → would cause premature exit (not matching backtest behavior)
                # Cancel old trail and re-place with wider callback.
                # Tracking resets, but wider trail is far from best → low trigger risk.
                # TIGHTENING is handled by bot's check_exit (current ATR, every 15m).
                for order in orders:
                    oid = order.get('id', '')
                    if oid == pos.get('sl_order_id', ''):
                        continue
                    o_type = (order.get('info') or {}).get('type', '') or order.get('type', '')
                    if 'TRAILING' in o_type.upper():
                        try:
                            self.exchange.cancel_order(oid, symbol)
                        except Exception:
                            pass
                need_replace = True
                logger.info(f"Trail LOOSEN: {old_callback}%→{new_callback}% (ATR=${cur_atr:.0f})")

            if need_replace:
                callback = new_callback if new_callback > 0 else old_callback
                if callback <= 0:
                    callback = round(max(0.1, min(5.0, trail_K * cur_atr / cur_price * 100)), 1)
                live = self._get_live_positions()
                if live and pos['direction'] in live:
                    qty = live[pos['direction']]['contracts']
                    tp_side = 'sell' if pos['direction'] == 'LONG' else 'buy'
                    activate = round(cur_price * (1 + 0.001) if pos['direction'] == 'LONG'
                                     else cur_price * (1 - 0.001), 1)
                    self.exchange.create_order(
                        symbol, 'TRAILING_STOP_MARKET', tp_side, qty,
                        params={'positionSide': 'BOTH',
                                'activatePrice': activate,
                                'trailingPercent': callback,
                                'reduceOnly': True})
                    pos['last_callback'] = callback
                    bp = pos.get('best_price', cur_price)
                    est = round(bp * (1 - callback/100) if pos['direction'] == 'LONG'
                                else bp * (1 + callback/100), 1)
                    logger.info(f"Trail placed: callback={callback}% est≈${est:.1f}")

            # BUG#46 policy:
            # - LOOSEN (ATR↑): re-place → tracking resets but trail is far → safe
            # - TIGHTEN (ATR↓): DON'T re-place → bot check_exit handles it → tracking preserved

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
        logger.info(f"C1 Breakout v2.6 (N={self.max_positions}, exchange={ex_lev}x, trading={tr_lev}x)")
        self._last_hourly = -1

        while True:
            try:
                self._wait_for_candle()
                self._sync_exchange()  # BUG#38: unconditional — orphan detect works even when no local positions
                candles = self.fetch_candles()
                if candles:
                    self.process_candles(candles)
                else:
                    # BUG#51: warn on sustained candle failures — check_exit skipped,
                    # ATR-tighten trail backup is not running (exchange trail is fixed callback).
                    streak = getattr(self, '_candle_fail_streak', 0)
                    if streak >= 3 and self.positions:
                        logger.warning(
                            f"Candle fetch failed {streak}x in a row WITH open position — "
                            f"bot's tighten logic not running; relying on exchange SL/TRAIL only")
                    elif streak >= 6:
                        logger.warning(f"Candle fetch failed {streak}x in a row (no position)")
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
                    pos_parts = []
                    for p in self.positions:
                        cb = p.get('last_callback', 0)
                        bp = p.get('best_price', p.get('entry_price', 0))
                        if cb > 0 and bp > 0:
                            d = p.get('direction', 'LONG')
                            est = round(bp * (1 - cb/100) if d == 'LONG' else bp * (1 + cb/100), 0)
                            pos_parts.append(
                                f"{p['direction']} @{p['entry_price']:.0f} "
                                f"best={bp:.0f} trail≈{est:.0f} ({p['bars_held']}b)")
                        else:
                            pos_parts.append(
                                f"{p['direction']} @{p['entry_price']:.0f} ({p['bars_held']}b)")
                    pos_info = ', '.join(pos_parts) or 'none'
                    logger.info(f"HOURLY | trades={total} WR={wr:.0f}% "
                                f"pos=[{pos_info}]")
            except KeyboardInterrupt:
                logger.info("Stopped"); self._save_state(); break
            except Exception as e:
                logger.error(f"Loop: {e}", exc_info=True); self._save_state(); time.sleep(60)
