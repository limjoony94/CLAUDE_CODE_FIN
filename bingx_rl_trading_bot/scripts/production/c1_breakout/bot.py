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

BUG#65 (2026-04-18): Capture actual MARKET close fill price
  - Root cause: _do_close recorded exit_price as the theoretical trigger from
    check_exit, ignoring actual MARKET fill from _exchange_close. For TRAIL_TP
    exits, recorded PnL was slightly overstated (no sell-side slippage modeled).
  - Fix: _exchange_close returns actual fill price (via order['average'] or
    fetch_my_trades fallback). _do_close uses it for accurate PnL recording.
    Also records exit_slippage_pct for monitoring. Falls back to theoretical
    on API error (safe degradation).

BUG#64 (2026-04-18): best_price sync with fill_price at entry
  - Root cause: _do_open set best_price = signal_price, _exchange_open updated
    entry_price = fill_price, but best_price stayed at signal. With slippage,
    best_pnl at entry was non-zero (sometimes negative for LONG with positive slip).
    Backtest has best_price = entry_price exactly → best_pnl = 0 at entry.
  - Fix: after fill_price update, also sync best_price = fill_price.
    Now matches backtest initialization (best_pnl = 0 at entry).

BUG#62 (2026-04-18): activatePrice aligned with trail_activation_pct
  - Root cause: TRAILING_STOP_MARKET used activatePrice = entry × 1.001 (0.1%),
    2x stricter than backtest's trail_activation_pct (0.05%).
    Live trail activated later than backtest would — minor but real divergence.
  - Fix: activatePrice = entry × (1 ± trail_activation_pct/100) → 0.05% match.

BUG#63 (2026-04-18): Best-price-driven trail tighten (backtest parity)
  - Root cause: BUG#61 baton-touch STOP_MARKET is static — never updated as
    best_price climbs, even when backtest re-evaluates trail every bar.
    Live trail fell behind backtest in trending markets.
  - Fix: each cycle, compute exact backtest trigger. If tighter than current
    STOP_MARKET trigger, cancel+replace at new price. Honors backtest's
    "re-check every bar" semantics. Threshold: 0.05% of trigger price (to avoid
    excessive churn on small best_price fluctuations).

BUG#61 (2026-04-18): Trail update baton-touch on LOOSEN
  - Root cause: BUG#46 LOOSEN still reset BingX tracking. New TRAILING order
    with activatePrice = cur_price × 1.001 meant:
      1. Lost historical best_price ($77K → forgotten)
      2. Needs activatePrice to be re-reached before resuming tracking
      3. If price dropped below activatePrice immediately, no protection
  - Fix: on LOOSEN, use BATON-TOUCH:
      1. Compute EXACT trigger via _calc_trail_trigger_price (backtest formula)
         cur² - best·cur + trail_K·ATR·entry = 0 → upper root is trigger
      2. Place as STOP_MARKET (fixed trigger, not TRAILING)
      3. This preserves the level where trail WOULD be if tracking had continued
    Track trail_order_id in pos state to properly identify/verify/cancel.
    Pre-activation case (best_pnl ≤ trail_activation_pct) still uses
    TRAILING_STOP_MARKET since baton-touch undefined before activation.
    BUG#61b: use exact quadratic formula (100% parity with signals.py).

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

BUG#54 (2026-04-17): bars_since_last_exit ignored wall-clock during outages
  - Root cause: counter only incremented per processed candle cycle. After a
    2h outage, restart saw saved_counter=0 and first cycle made it 1 — bot
    believed only 1 bar had passed since last exit. min_bars_between=2 still
    blocked entry once, so operational impact was minor, but bookkeeping was
    incorrect and diverged from backtest semantics.
  - Fix: persist last_exit_time on every close (ghost + normal). On _load_state
    compute elapsed_bars = (now - last_exit_time)/15min and take max with saved
    counter. Never regresses (prefers higher of the two).

BUG#55 (2026-04-17): Partial fill on MARKET was silent
  - Root cause: BingX market orders rarely partial but thin-liquidity moments
    can leave filled_qty < requested. SL/Trail already size to filled_qty
    (BUG#28), so exchange protection stays correct, but operator had no signal.
  - Fix: log warning when filled_qty < requested_qty × 0.99 with shortfall %.

BUG#56 (2026-04-17): trade_history grew unbounded in memory
  - Root cause: _save_state writes trade_history[-500:] to disk, but in-memory
    list kept appending. After months of operation memory footprint grew
    linearly (~220KB/year at 3 trades/day — negligible but asymmetric).
  - Fix: after each exit, trim in-memory list to 500 when it exceeds 1000.

BUG#57 (2026-04-17): datetime.utcnow() deprecated in Python 3.12+
  - Root cause: naive UTC construction deprecated; Python 3.14+ may remove.
    Would start emitting DeprecationWarnings in 3.13 logs.
  - Fix: _utc_now() / _utc_now_naive_iso() helpers. Internal arithmetic uses
    aware UTC; serialization uses .replace(tzinfo=None).isoformat() to match
    existing state.json format (backward-compatible).

BUG#58 (2026-04-17): state.json I/O on OneDrive-synced path could crash loop
  - Root cause: state_path lives under OneDrive sync folder. Sync-triggered
    file locks (ERROR_SHARING_VIOLATION) would raise from os.replace and
    propagate to main loop. A missed save is recoverable; a crashed loop is not.
  - Fix: try/except around json.dump and os.replace. Log warning, clean up
    .tmp on failure. Next cycle retries. Positions still re-hydrated via
    orphan detection on restart if state is stale.

BUG#59 (2026-04-17): _update_exchange_trail silent failures
  - Root cause: the outer try/except logged a single-line warning on every
    failure. Sustained structural problems (API permissions revoked, rate
    limited) produced one warning per cycle — easy to miss in log noise.
    With an open position, both SL verification AND trail re-placement run
    inside this function, so sustained failure = no exchange-side protection
    maintenance.
  - Fix: _trail_update_fail_streak counter. ≥3 consecutive failures elevate
    the warning to note that SL verification + tighten backup are not running.

BUG#60 (2026-04-17): check_exit trail path unsafe on bad price data
  - Root cause: trail_dist_pct = trail_K × atr / current_close × 100. If
    current_close ≤ 0 (bad candle) → ZeroDivisionError. If current_close is
    NaN → NaN propagation silently skips trail without logging, leaving
    position unprotected on the bot side (exchange still has orders).
  - Fix: add `not math.isnan(current_close) and current_close > 0` to trail
    activation guard alongside existing ATR guard.

BUG#61 (2026-04-17): TimeSyncBingX accepted any server time blindly
  - Root cause: BingX serverTime response trusted without sanity check. A
    bad response (e.g. 0, year-2000, year-3000) would set _time_offset
    wildly wrong, breaking every signed API request thereafter with
    timestamp errors.
  - Fix: _MAX_OFFSET_MS = 60_000 cap. Reject offsets beyond ±60s; keep
    previous valid offset (or 0 at startup). Log warning on rejection.
"""

import os
import sys
import json
import time
import logging
import ccxt
import yaml
import requests
from datetime import datetime, timedelta, timezone
from pathlib import Path


# BUG#57: datetime.utcnow() deprecated in Python 3.12+.
# _utc_now() returns UTC-aware datetime; callers use .replace(tzinfo=None) to
# preserve historical naive isoformat (backward-compatible with existing state.json).
def _utc_now():
    return datetime.now(timezone.utc)


def _utc_now_naive_iso():
    return datetime.now(timezone.utc).replace(tzinfo=None).isoformat()

from .signals import C1BreakoutSignal
from .indicators import compute_atr, compute_channel, compute_fractal_swings
from .config import load_config

logger = logging.getLogger('c1_breakout')


class TimeSyncBingX(ccxt.bingx):
    """BingX with automatic server time synchronization."""
    _time_offset = 0
    _last_sync = 0

    # BUG#61: clamp offset to ±60s. Larger deviation likely indicates malformed
    # server response or system clock drift; applying it would break every signed
    # request with timestamp errors. Ignoring keeps the previous valid offset (or 0).
    _MAX_OFFSET_MS = 60_000

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
                    new_offset = server_ms - int(now * 1000)
                    if abs(new_offset) > self._MAX_OFFSET_MS:
                        logger.warning(
                            f"Time sync: offset={new_offset}ms exceeds "
                            f"±{self._MAX_OFFSET_MS}ms — rejecting, keeping "
                            f"previous offset={self._time_offset}ms")
                    else:
                        self._time_offset = new_offset
                        self._last_sync = now
                        logger.debug(f"Time sync: offset={new_offset}ms")
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
        # BUG#54: track last exit wall-clock for accurate bars_since_last_exit
        # reconstruction on restart (elapsed time divided by 15min bar).
        self.last_exit_time = None  # datetime or None
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

        # BUG#54: reconstruct bars_since_last_exit from elapsed wall-clock time.
        # Saved counter reflects only bars processed during bot uptime; a 2h outage
        # would leave counter undercounted and briefly bias entry permission timing.
        # Prefer max(saved_counter, elapsed_bars) — conservative (never entry-blocks
        # legitimate waits, and catches up after extended downtime).
        last_exit_str = state.get('last_exit_time')
        if last_exit_str:
            try:
                from datetime import timezone
                le = datetime.fromisoformat(last_exit_str.replace('Z', '+00:00'))
                if le.tzinfo is None:
                    le = le.replace(tzinfo=timezone.utc)
                now_utc = datetime.now(timezone.utc)
                elapsed_sec = (now_utc - le).total_seconds()
                if elapsed_sec > 0:
                    elapsed_bars = int(elapsed_sec // 900)  # 15 min per bar
                    if elapsed_bars > self.bars_since_last_exit:
                        logger.info(
                            f"bars_since_last_exit reconciled: "
                            f"{self.bars_since_last_exit}→{elapsed_bars} "
                            f"(elapsed {elapsed_sec/60:.1f} min since last exit)")
                        self.bars_since_last_exit = elapsed_bars
                self.last_exit_time = le
            except Exception as e:
                logger.debug(f"last_exit_time parse failed: {e}")
        logger.info(f"State: {len(self.positions)} pos, {len(self.trade_history)} trades")

    def _save_state(self):
        """Persist bot state to disk.

        BUG#25: Atomic write (tmp → rename) is crash-safe.
        BUG#58: state_path is under OneDrive-synced folder — sync can briefly lock
        files during upload. Wrap in try/except to avoid crashing the main loop:
        a single failed save is recoverable (next cycle retries). True data loss
        only occurs if the bot dies before the next save AND the last persisted
        state is stale — acceptable since positions are re-hydrated via orphan
        detection on restart.
        """
        Path(self.state_path).parent.mkdir(parents=True, exist_ok=True)
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
            # BUG#54: save last_exit_time for elapsed-bar reconciliation on restart.
            # Naive iso format (matches exit_time/entry_time/updated convention).
            'last_exit_time': (self.last_exit_time.replace(tzinfo=None).isoformat()
                               if self.last_exit_time else None),
            'updated': _utc_now_naive_iso(),
        }
        try:
            with open(tmp_path, 'w') as f:
                json.dump(data, f, indent=2)
            # Atomic rename (on Windows: replaces existing)
            os.replace(tmp_path, self.state_path)
        except (OSError, PermissionError) as e:
            # BUG#58: OneDrive sync lock or transient I/O — warn, do not crash.
            # Attempt cleanup of tmp file to prevent accumulation.
            logger.warning(f"State save skipped (I/O): {e} — will retry next cycle")
            try:
                if os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass

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
                from datetime import timezone
                if exit_ts_ms:
                    exit_dt = datetime.fromtimestamp(exit_ts_ms / 1000, tz=timezone.utc)
                else:
                    exit_dt = datetime.now(timezone.utc)
                exit_time = exit_dt.replace(tzinfo=None).isoformat()  # preserve naive format
                self.trade_history.append({
                    'direction': d, 'entry_price': pos['entry_price'],
                    'exit_price': exit_price,
                    'pnl_pct': round(pnl, 4), 'reason': reason,
                    'bars_held': pos.get('bars_held', 0),
                    'exit_time': exit_time,
                })
                self.bars_since_last_exit = 0
                self.last_exit_time = exit_dt  # BUG#54: accurate restart reconciliation
                # BUG#56: cap in-memory trade_history
                if len(self.trade_history) > 1000:
                    self.trade_history = self.trade_history[-500:]

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
                    'best_price': ep, 'entry_time': _utc_now_naive_iso(),
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
            # Regime filter (trend) — skip entry in low-trend (choppy) regimes.
            # Disabled by default; enable after 30-day LIVE validation.
            tf_cfg = cfg.get('trend_filter', {}) or {}
            if tf_cfg.get('enabled', False):
                lb = tf_cfg.get('lookback_bars', 192)
                min_trend = tf_cfg.get('min_abs_trend_pct', 1.0)
                if bar >= lb:
                    c_past = candles['close'][bar - lb]
                    c_now = candles['close'][bar]
                    if c_past > 0:
                        trend_pct = abs((c_now / c_past - 1) * 100)
                        if trend_pct < min_trend:
                            logger.info(
                                f"Trend filter skip: |trend|={trend_pct:.2f}% "
                                f"< {min_trend}% (lb={lb})")
                            self._save_state()
                            return

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
            'best_price': price, 'entry_time': _utc_now_naive_iso(),
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
        trading_lev = self.config['exchange'].get('trading_leverage',
                      self.config['exchange'].get('leverage', 1))

        from datetime import timezone
        exit_dt = datetime.now(timezone.utc)

        # BUG#65: close on exchange FIRST, capture actual fill price
        actual_fill = None
        actual_ts = None
        if self.exchange:
            actual_fill, actual_ts = self._exchange_close(d)

        # Use actual fill if available (more accurate than theoretical)
        recorded_exit = actual_fill if actual_fill and actual_fill > 0 else xp
        if d == 'LONG':
            raw_pct = (recorded_exit / pos['entry_price'] - 1) * 100
        else:
            raw_pct = (1 - recorded_exit / pos['entry_price']) * 100
        pnl = (raw_pct - 0.10) * trading_lev

        # Slippage tracking
        slip = 0.0
        if actual_fill and actual_fill > 0 and xp > 0:
            if d == 'LONG':
                slip = (actual_fill / xp - 1) * 100  # negative = sold below trigger
            else:
                slip = (1 - actual_fill / xp) * 100  # negative = bought above trigger
            if abs(slip) > 0.02:
                logger.info(f"Exit slippage: {slip:+.3f}% (trigger=${xp:.1f} fill=${actual_fill:.1f})")

        record_exit_time = exit_dt
        if actual_ts:
            try:
                record_exit_time = datetime.utcfromtimestamp(actual_ts / 1000)
            except Exception:
                pass

        self.trade_history.append({
            'direction': d, 'entry_price': pos['entry_price'],
            'exit_price': recorded_exit,
            'pnl_pct': round(pnl, 4), 'reason': exit_signal['reason'],
            'bars_held': pos['bars_held'],
            'exit_time': record_exit_time.replace(tzinfo=None).isoformat()
                         if record_exit_time.tzinfo else record_exit_time.isoformat(),
            'exit_slippage_pct': round(slip, 4) if actual_fill else None,
        })
        self.bars_since_last_exit = 0  # BUG#16: reset for min_bars_between
        self.last_exit_time = exit_dt  # BUG#54: accurate restart reconciliation
        # BUG#56: cap in-memory trade_history (disk was already capped at 500)
        if len(self.trade_history) > 1000:
            self.trade_history = self.trade_history[-500:]

        logger.info(f"EXIT {d} {exit_signal['reason']} | PnL={pnl:+.2f}% | "
                    f"Hold={pos['bars_held']}b")
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
            # BUG#55: detect partial fill (thin liquidity / extreme volatility).
            # SL/Trail are sized to filled_qty (BUG#28), so exchange-side protection
            # stays correct. Log warns operator to investigate order book conditions.
            requested_qty = float(qty) if qty else filled_qty
            if requested_qty > 0 and filled_qty < requested_qty * 0.99:
                shortfall = (requested_qty - filled_qty) / requested_qty * 100
                logger.warning(
                    f"Partial fill: filled={filled_qty} / requested={requested_qty} "
                    f"(shortfall {shortfall:.2f}%) — SL/Trail sized to actual fill")
            if fill_price > 0 and self.positions:
                self.positions[-1]['entry_price'] = fill_price
                # BUG#64: sync best_price with fill_price for backtest parity.
                # _do_open initialized best_price = signal_price. With slippage,
                # best_price could be LOWER than entry_price (for LONG with positive
                # slip), causing negative best_pnl at entry — diverges from backtest
                # where best_pnl = 0 at entry.
                self.positions[-1]['best_price'] = fill_price
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
                # BUG#62: activatePrice aligned with backtest trail_activation_pct
                # Previous 0.001 (0.1%) was 2x stricter than backtest (0.05%)
                # Now matches: activates when price moves trail_activation_pct from entry
                act_pct = self.config['strategy'].get('trail_activation_pct', 0.05) / 100
                activate = round(ref_price * (1 + act_pct) if direction == 'LONG'
                                 else ref_price * (1 - act_pct), 1)
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
            # BUG#61: trail may be TRAILING_STOP_MARKET (pre-activation) or
            # STOP_MARKET with tracked trail_order_id (baton-touched after LOOSEN).
            trail_order_id = pos.get('trail_order_id', '')
            trail_exists_by_id = (trail_order_id and any(
                o.get('id', '') == trail_order_id for o in orders))
            trail_exists_by_type = any(
                'TRAILING' in ((o.get('info') or {}).get('type', '') or '').upper()
                for o in orders)
            trail_exists = trail_exists_by_id or trail_exists_by_type

            # BUG#35: On first cycle after restart, force re-placement for legacy priceRate orders
            force_reset = getattr(self, '_force_trail_reset', False)
            if force_reset:
                self._force_trail_reset = False
                if trail_exists:
                    # Cancel existing trail (TRAILING or baton-touch STOP_MARKET)
                    for order in orders:
                        oid = order.get('id', '')
                        if oid == pos.get('sl_order_id', ''):
                            continue
                        o_type = (order.get('info') or {}).get('type', '') or order.get('type', '')
                        is_trailing = 'TRAILING' in o_type.upper()
                        is_baton_stop = (trail_order_id and oid == trail_order_id)
                        if is_trailing or is_baton_stop:
                            try:
                                self.exchange.cancel_order(oid, symbol)
                            except Exception:
                                pass
                    pos['trail_order_id'] = ''  # clear stale ID
                    trail_exists = False
                    logger.info("Trail: forcing re-placement (BUG#35 legacy priceRate fix)")

            # ── 3. Re-place trail if missing, LOOSEN on ATR rise, or TIGHTEN on best rise ──
            trail_K = self.config['strategy'].get('trail_K', 2.5)
            new_callback = round(max(0.1, min(5.0, trail_K * cur_atr / cur_price * 100)), 1)
            old_callback = pos.get('last_callback', 0)

            # BUG#63: Detect best_price-driven tighten opportunity.
            # Backtest re-evaluates trail every bar with current best_price.
            # Live baton-touch STOP_MARKET is static; only updating on callback
            # change misses the best_price-driven tightening that backtest does.
            # Fix: if current computed trigger is tighter than existing trail,
            # update it (honors backtest "re-check every bar" semantics).
            prev_trigger = pos.get('trail_trigger', 0)
            bp_for_calc = pos.get('best_price', cur_price)
            d_for_calc = pos.get('direction', 'LONG')
            act_chk = self.config['strategy'].get('trail_activation_pct', 0.05)
            if d_for_calc == 'LONG':
                best_pnl_chk = (bp_for_calc / pos.get('entry_price', cur_price) - 1) * 100
            else:
                best_pnl_chk = (1 - bp_for_calc / pos.get('entry_price', cur_price)) * 100

            should_tighten = False
            if trail_exists and prev_trigger > 0 and best_pnl_chk > act_chk:
                # Compute current theoretical trigger with exact backtest formula
                exact_trig = self._calc_trail_trigger_price(pos, cur_atr)
                if exact_trig is not None:
                    # For LONG: tighter = higher trigger. For SHORT: tighter = lower.
                    if d_for_calc == 'LONG' and exact_trig > prev_trigger + max(5.0, prev_trigger * 0.0005):
                        should_tighten = True
                    elif d_for_calc == 'SHORT' and exact_trig < prev_trigger - max(5.0, prev_trigger * 0.0005):
                        should_tighten = True

            need_replace = False
            if not trail_exists:
                need_replace = True
                logger.warning(f"Trail STOP missing — will re-place")
            elif should_tighten:
                # BUG#63: TIGHTEN on best_price rise — backtest matches this behavior
                for order in orders:
                    oid = order.get('id', '')
                    if oid == pos.get('sl_order_id', ''):
                        continue
                    o_type = (order.get('info') or {}).get('type', '') or order.get('type', '')
                    is_trailing = 'TRAILING' in o_type.upper()
                    is_baton_stop = (trail_order_id and oid == trail_order_id)
                    if is_trailing or is_baton_stop:
                        try:
                            self.exchange.cancel_order(oid, symbol)
                        except Exception:
                            pass
                pos['trail_order_id'] = ''
                need_replace = True
                logger.info(f"Trail TIGHTEN: best-driven trigger update (prev=${prev_trigger:.1f})")
            elif new_callback > old_callback + 0.1:
                # BUG#46 + BUG#61: LOOSEN — ATR rose → exchange trail is too tight
                # Cancel old trail (TRAILING or baton-touch STOP_MARKET) and re-place
                # using baton-touch from best_price (BUG#61 below in need_replace path).
                for order in orders:
                    oid = order.get('id', '')
                    if oid == pos.get('sl_order_id', ''):
                        continue
                    o_type = (order.get('info') or {}).get('type', '') or order.get('type', '')
                    # Cancel TRAILING or the tracked baton-touch STOP_MARKET
                    is_trailing = 'TRAILING' in o_type.upper()
                    is_baton_stop = (trail_order_id and oid == trail_order_id)
                    if is_trailing or is_baton_stop:
                        try:
                            self.exchange.cancel_order(oid, symbol)
                        except Exception:
                            pass
                # Clear old trail_order_id since we cancelled it
                pos['trail_order_id'] = ''
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
                    bp = pos.get('best_price', cur_price)
                    ep = pos.get('entry_price', cur_price)
                    d = pos['direction']

                    # BUG#61: Baton-touch — preserve best_price tracking on LOOSEN.
                    # Previous TRAILING_STOP_MARKET cancel+re-place RESETS BingX's
                    # internal best_price tracking. New order needs activatePrice to
                    # be re-reached before tracking resumes.
                    # Fix: compute exact trigger from LOCAL best_price × (1±callback/100)
                    # and place as STOP_MARKET (fixed trigger), NOT TRAILING_STOP_MARKET.
                    # This "hands off" the trail level — continuing from where the
                    # old trail would have been, based on the historical best_price.
                    if d == 'LONG':
                        best_profit_pct = (bp / ep - 1) * 100
                    else:
                        best_profit_pct = (1 - bp / ep) * 100
                    activation = self.config['strategy'].get('trail_activation_pct', 0.05)

                    if best_profit_pct > activation:
                        # Trail activation threshold met → use baton-touch STOP_MARKET
                        # BUG#61b: Use EXACT backtest formula via _calc_trail_trigger_price
                        # (solves quadratic cur² - best·cur + trail_K·ATR·entry = 0)
                        # instead of approximation best × (1 - callback/100).
                        # Ensures 100% parity with signals.py check_exit math.
                        exact_trigger = self._calc_trail_trigger_price(pos, cur_atr)
                        if exact_trigger is not None:
                            baton_trigger = exact_trigger
                        elif d == 'LONG':
                            baton_trigger = round(bp * (1 - callback/100), 1)
                        else:
                            baton_trigger = round(bp * (1 + callback/100), 1)
                        # Sanity: trigger must be on correct side of current price
                        ok = (d == 'LONG' and baton_trigger < cur_price) or \
                             (d == 'SHORT' and baton_trigger > cur_price)
                        if ok:
                            result = self.exchange.create_order(
                                symbol, 'STOP_MARKET', tp_side, qty,
                                params={'positionSide': 'BOTH',
                                        'stopPrice': baton_trigger,
                                        'reduceOnly': True})
                            pos['last_callback'] = callback
                            pos['trail_order_id'] = result.get('id', '')
                            pos['trail_trigger'] = baton_trigger
                            logger.info(f"Trail BATON-TOUCH: STOP_MARKET @${baton_trigger:.1f} "
                                        f"(best=${bp:.1f} × (1∓{callback}%))")
                        else:
                            # Trigger would be on wrong side — fall back to TRAILING
                            # BUG#62: activatePrice aligned with trail_activation_pct
                            act_pct = activation / 100
                            activate = round(cur_price * (1 + act_pct) if d == 'LONG'
                                             else cur_price * (1 - act_pct), 1)
                            self.exchange.create_order(
                                symbol, 'TRAILING_STOP_MARKET', tp_side, qty,
                                params={'positionSide': 'BOTH',
                                        'activatePrice': activate,
                                        'trailingPercent': callback,
                                        'reduceOnly': True})
                            pos['last_callback'] = callback
                            est = round(bp * (1 - callback/100) if d == 'LONG'
                                        else bp * (1 + callback/100), 1)
                            logger.info(f"Trail placed (fallback TRAILING): callback={callback}% est≈${est:.1f}")
                    else:
                        # Not yet activated — use TRAILING_STOP_MARKET
                        # BUG#62: activatePrice aligned with trail_activation_pct (backtest parity)
                        act_pct = activation / 100
                        activate = round(cur_price * (1 + act_pct) if d == 'LONG'
                                         else cur_price * (1 - act_pct), 1)
                        self.exchange.create_order(
                            symbol, 'TRAILING_STOP_MARKET', tp_side, qty,
                            params={'positionSide': 'BOTH',
                                    'activatePrice': activate,
                                    'trailingPercent': callback,
                                    'reduceOnly': True})
                        pos['last_callback'] = callback
                        est = round(bp * (1 - callback/100) if d == 'LONG'
                                    else bp * (1 + callback/100), 1)
                        logger.info(f"Trail placed (TRAILING, pre-activation): callback={callback}% est≈${est:.1f}")

            # BUG#46 + BUG#61 policy:
            # - LOOSEN (ATR↑): baton-touch via STOP_MARKET at best_price × callback
            #   (preserves historical best tracking, no reset)
            # - TIGHTEN (ATR↓): DON'T re-place → bot check_exit handles it
            # - Pre-activation (best_pnl ≤ 0.05%): use TRAILING_STOP_MARKET (BingX native)
            self._trail_update_fail_streak = 0  # success resets streak (BUG#59)

        except Exception as e:
            # BUG#59: track sustained trail-update failures. Silent warnings per-cycle
            # could mask a structural issue (API permissions, rate limit). With open
            # position, ≥3 consecutive failures mean SL verification + trail re-placement
            # are both blocked — operator should investigate.
            self._trail_update_fail_streak = getattr(
                self, '_trail_update_fail_streak', 0) + 1
            streak = self._trail_update_fail_streak
            if streak >= 3:
                logger.warning(
                    f"Trail update failed {streak}x in a row: {e} — "
                    f"SL verification + tighten backup are NOT running")
            else:
                logger.warning(f"Trail update failed: {e}")

    def _exchange_close(self, direction):
        """Close position via MARKET + cancel remaining orders.

        BUG#65: return actual fill price (and timestamp) from the MARKET close
        so _do_close can record realistic exit_price instead of the theoretical
        trigger price. Fallback to None on error — _do_close will keep computed.
        """
        try:
            symbol = self.config['exchange']['symbol']
            side = 'sell' if direction == 'LONG' else 'buy'
            live = self._get_live_positions()
            actual_fill = None
            actual_ts = None
            if live and direction in live:
                order = self.exchange.create_order(symbol, 'market', side,
                    live[direction]['contracts'],
                    params={'positionSide': 'BOTH', 'reduceOnly': True})
                # Try order's reported fill first
                try:
                    avg = order.get('average') or order.get('price')
                    if avg and float(avg) > 0:
                        actual_fill = float(avg)
                    ts = order.get('timestamp') or order.get('info', {}).get('updateTime')
                    if ts:
                        actual_ts = int(ts)
                except Exception:
                    pass
                # If average not in immediate response, query fetch_my_trades
                if actual_fill is None:
                    try:
                        since = int((time.time() - 60) * 1000)  # last 60 sec
                        trades = self.exchange.fetch_my_trades(symbol, since=since, limit=10)
                        for t in reversed(trades):
                            if t.get('side', '').lower() == side and float(t.get('amount', 0)) > 0:
                                actual_fill = float(t['price'])
                                actual_ts = int(t.get('timestamp') or 0)
                                break
                    except Exception:
                        pass
            # Cancel all remaining orders (individually, so one failure doesn't stop others)
            try:
                for order in self.exchange.fetch_open_orders(symbol):
                    try:
                        self.exchange.cancel_order(order['id'], symbol)
                    except Exception:
                        pass
            except Exception:
                pass
            logger.info(f"Closed {direction}" + (f" @${actual_fill:.1f}" if actual_fill else ""))
            return actual_fill, actual_ts
        except Exception as e:
            logger.error(f"Close: {e}")
            return None, None

    # ── Main Loop ─────────────────────────────────────────

    def _wait_for_candle(self):
        now = _utc_now().replace(tzinfo=None)  # naive UTC for arithmetic
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
                now = _utc_now().replace(tzinfo=None)
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
