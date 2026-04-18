"""
Shake-out Pattern Verification — Live vs Backtest (Apr 10-17, 2026)
=====================================================================
Question: During BTC's run from $70K to $75K+, the bot repeatedly took
positions that got "shaken out" — stopped out on SL, or trailed out for
tiny/zero profit only to re-enter at a higher price. The user wants to
know: is this STRATEGY behavior (same thing happens in backtest), or is
this an EXECUTION problem (intrabar trail firing in live but not
reproducible at bar-close in backtest)?

The critical asymmetry in scripts/production/c1_breakout/signals.py:
  - SL / Emergency: checked against current_high / current_low (intrabar)
  - TRAIL_TP: checked against current_close ONLY (bar-close)

Live exchange behaviour:
  - STOP_MARKET SL:           intrabar   (matches backtest)
  - TRAILING_STOP_MARKET:     intrabar   (DOES NOT match backtest)

So live trail exits can fire on a wick that the bar-close backtest never sees.
This script fetches exact BingX 15m swap candles, runs production signal code,
and produces a per-trade comparison quantifying where live ≠ backtest and why.

Output: results/shake_out_pattern_verification.json
"""
import json
import math
import os
import sys
from datetime import datetime, timezone

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, REPO_ROOT)

from scripts.production.c1_breakout.indicators import (
    compute_atr, compute_channel, compute_fractal_swings
)
from scripts.production.c1_breakout.signals import C1BreakoutSignal

# ── Config (production-identical, from c1_breakout_config.yaml) ──
CONFIG = {
    'channel_period': 15,
    'body_min_ratio': 0.4,
    'atr_period': 14,
    'trail_K': 2.5,
    'max_sl_atr': 3.3,
    'emergency_sl_pct': 3.0,
    'max_hold_bars': 192,
    'sl_min_pct': 0.15,
    'sl_max_pct': 3.0,
    'min_bars_between': 2,
    'trail_activation_pct': 0.05,
}
FRACTAL_LOOKBACK = 10
LEVERAGE = 3
FEE_RT_PCT = 0.10  # round-trip taker fee as a % of notional (live has leveraged-notional fee)

# ── Live trades (from results/c1_breakout_state.json, trade_history) ──
# id is sequential index; trade #1 is the legacy orphan at 71100 (pre-v2.5 dust) — skip it
# and focus on v2.5+/v2.6 (trade #2 onward, Apr 12).
LIVE_TRADES = [
    {'id': 2,  'dir': 'SHORT', 'entry': 70560.0,  'exit': 71080.3, 'pnl_3x': -2.5122, 'reason': 'EXCHANGE_SL',    'bars': 16, 'exit_time': '2026-04-13T13:00:06'},
    {'id': 3,  'dir': 'LONG',  'entry': 71668.6,  'exit': 74026.6, 'pnl_3x':  9.5704, 'reason': 'EXCHANGE_TRAIL', 'bars': 44, 'exit_time': '2026-04-14T01:09:59'},
    {'id': 4,  'dir': 'SHORT', 'entry': 74226.0,  'exit': 74782.5, 'pnl_3x': -2.5492, 'reason': 'EXCHANGE_SL',    'bars':  4, 'exit_time': '2026-04-14T07:15:06'},
    {'id': 5,  'dir': 'SHORT', 'entry': 74300.0,  'exit': 74630.0, 'pnl_3x': -1.6324, 'reason': 'EXCHANGE_TRAIL', 'bars': 10, 'exit_time': '2026-04-14T13:30:06'},
    {'id': 6,  'dir': 'LONG',  'entry': 75112.0,  'exit': 75448.0, 'pnl_3x':  1.0420, 'reason': 'EXCHANGE_TRAIL', 'bars':  3, 'exit_time': '2026-04-14T14:45:06'},
    {'id': 7,  'dir': 'LONG',  'entry': 74521.9,  'exit': 74521.9, 'pnl_3x': -0.3000, 'reason': 'TRAIL_TP',       'bars': 12, 'exit_time': '2026-04-15T03:45:06'},
    {'id': 8,  'dir': 'SHORT', 'entry': 73948.3,  'exit': 73948.3, 'pnl_3x': -0.3000, 'reason': 'TRAIL_TP',       'bars': 12, 'exit_time': '2026-04-15T08:30:05'},
    {'id': 9,  'dir': 'LONG',  'entry': 74305.7,  'exit': 73854.8, 'pnl_3x': -2.1205, 'reason': 'EXCHANGE_SL',    'bars':  3, 'exit_time': '2026-04-15T13:43:35'},
    {'id': 10, 'dir': 'LONG',  'entry': 74361.5,  'exit': 74622.4, 'pnl_3x':  0.7526, 'reason': 'EXCHANGE_TRAIL', 'bars':  4, 'exit_time': '2026-04-15T20:11:57'},
    {'id': 11, 'dir': 'SHORT', 'entry': 74649.9,  'exit': 74685.6, 'pnl_3x': -0.4435, 'reason': 'EXCHANGE_TRAIL', 'bars': 16, 'exit_time': '2026-04-16T12:02:22'},
    {'id': 12, 'dir': 'LONG',  'entry': 74958.2,  'exit': 74442.4, 'pnl_3x': -2.3644, 'reason': 'EXCHANGE_TRAIL', 'bars':  0, 'exit_time': '2026-04-16T13:42:28'},
    {'id': 13, 'dir': 'SHORT', 'entry': 73653.8,  'exit': 74058.7, 'pnl_3x': -1.9492, 'reason': 'EXCHANGE_TRAIL', 'bars':  3, 'exit_time': '2026-04-16T14:49:45'},
    {'id': 14, 'dir': 'LONG',  'entry': 75055.5,  'exit': 75055.5, 'pnl_3x': -0.3000, 'reason': 'TRAIL_TP',       'bars': 16, 'exit_time': '2026-04-16T23:00:06'},
    {'id': 15, 'dir': 'SHORT', 'entry': 74680.0,  'exit': 74680.0, 'pnl_3x': -0.3000, 'reason': 'TRAIL_TP',       'bars': 22, 'exit_time': '2026-04-17T07:00:05'},
    {'id': 16, 'dir': 'LONG',  'entry': 75315.2,  'exit': 75630.1, 'pnl_3x':  0.9543, 'reason': 'EXCHANGE_TRAIL', 'bars':  3, 'exit_time': '2026-04-17T09:33:06'},
    {'id': 17, 'dir': 'LONG',  'entry': 76905.5,  'exit': 76159.5, 'pnl_3x': -3.2101, 'reason': 'EXCHANGE_TRAIL', 'bars':  1, 'exit_time': '2026-04-17T13:32:01'},
]

# Eval window: Apr 12 00:00 (first v2.5+ entry) to Apr 17 23:59
# Warmup: fetch back to Apr 10 to have indicator history
WARMUP_START_UTC = datetime(2026, 4, 10, 0, 0, tzinfo=timezone.utc)
EVAL_START_UTC   = datetime(2026, 4, 12, 0, 0, tzinfo=timezone.utc)
EVAL_END_UTC     = datetime(2026, 4, 17, 23, 59, tzinfo=timezone.utc)


def fetch_candles():
    """Fetch BTC-USDT 15m swap candles from BingX."""
    import ccxt
    exchange = ccxt.bingx({'options': {'defaultType': 'swap'}, 'enableRateLimit': True})

    start_ms = int(WARMUP_START_UTC.timestamp() * 1000)
    end_ms   = int(EVAL_END_UTC.timestamp() * 1000)

    all_candles = []
    since = start_ms
    while since < end_ms:
        batch = exchange.fetch_ohlcv('BTC-USDT', '15m', since=since, limit=1000)
        if not batch:
            break
        all_candles.extend(batch)
        last = batch[-1][0]
        if last <= since:
            break
        since = last + 1

    # Dedup + filter + sort
    seen = set(); unique = []
    for c in all_candles:
        if c[0] not in seen:
            seen.add(c[0]); unique.append(c)
    unique = [c for c in unique if start_ms <= c[0] <= end_ms]
    unique.sort(key=lambda c: c[0])
    return unique


def run_backtest(candles):
    """Run C1 Breakout v2 using production signal/indicator modules."""
    sig = C1BreakoutSignal(CONFIG)

    ts     = [c[0] for c in candles]
    opens  = [c[1] for c in candles]
    highs  = [c[2] for c in candles]
    lows   = [c[3] for c in candles]
    closes = [c[4] for c in candles]
    n = len(candles)

    atr = compute_atr(highs, lows, closes, CONFIG['atr_period'])
    ch_high, ch_low = compute_channel(highs, lows, CONFIG['channel_period'])
    sw_low, sw_high = compute_fractal_swings(highs, lows, FRACTAL_LOOKBACK)

    eval_start_ms = int(EVAL_START_UTC.timestamp() * 1000)

    trades = []
    position = None
    last_exit_bar = -999

    for i in range(1, n - 1):  # need i+1 for entry
        if position is not None:
            entry_bar = position['entry_bar']
            bars_held = i - entry_bar

            # best_price tracks running extreme (intrabar)
            if position['dir'] == 'LONG':
                position['best_price'] = max(position['best_price'], highs[i])
            else:
                position['best_price'] = min(position['best_price'], lows[i])

            atr_here = atr[i] if not math.isnan(atr[i]) else atr[i - 1]
            exit_res = sig.check_exit(
                direction=position['dir'],
                entry_price=position['entry_price'],
                best_price=position['best_price'],
                current_high=highs[i], current_low=lows[i], current_close=closes[i],
                sl_price=position['sl_price'],
                atr_val=atr_here, bars_held=bars_held,
            )

            if exit_res is not None:
                exit_price = exit_res['exit_price']
                reason = exit_res['reason']
                if position['dir'] == 'LONG':
                    raw_pnl = (exit_price / position['entry_price'] - 1) * 100
                else:
                    raw_pnl = (1 - exit_price / position['entry_price']) * 100
                pnl_3x = raw_pnl * LEVERAGE - FEE_RT_PCT * LEVERAGE

                trades.append({
                    'signal_bar_ts': ts[position['signal_bar']],
                    'signal_bar_dt': datetime.fromtimestamp(ts[position['signal_bar']]/1000, tz=timezone.utc).strftime('%Y-%m-%d %H:%M UTC'),
                    'entry_bar_ts': ts[entry_bar],
                    'entry_bar_dt': datetime.fromtimestamp(ts[entry_bar]/1000, tz=timezone.utc).strftime('%Y-%m-%d %H:%M UTC'),
                    'exit_bar_ts': ts[i],
                    'exit_bar_dt': datetime.fromtimestamp(ts[i]/1000, tz=timezone.utc).strftime('%Y-%m-%d %H:%M UTC'),
                    'dir': position['dir'],
                    'entry_price': round(position['entry_price'], 2),
                    'exit_price': round(exit_price, 2),
                    'sl_price': round(position['sl_price'], 2),
                    'best_price': round(position['best_price'], 2),
                    'raw_pnl_pct': round(raw_pnl, 4),
                    'pnl_3x_pct': round(pnl_3x, 4),
                    'reason': reason,
                    'bars_held': bars_held,
                })
                last_exit_bar = i
                position = None

        # Entry — only inside evaluation window
        if position is None and ts[i] >= eval_start_ms:
            if i - last_exit_bar < CONFIG['min_bars_between']:
                continue
            if math.isnan(ch_high[i]) or math.isnan(ch_low[i]) or math.isnan(atr[i]):
                continue
            if math.isnan(sw_low[i]) or math.isnan(sw_high[i]):
                continue

            entry_sig = sig.check_entry(
                bar_open=opens[i], bar_high=highs[i], bar_low=lows[i], bar_close=closes[i],
                channel_high=ch_high[i], channel_low=ch_low[i],
                atr_val=atr[i],
                last_swing_low=sw_low[i], last_swing_high=sw_high[i],
            )
            if entry_sig is None:
                continue

            entry_bar_idx = i + 1
            entry_price = opens[entry_bar_idx]
            if entry_sig['direction'] == 'LONG':
                best_price = max(entry_price, highs[entry_bar_idx])
            else:
                best_price = min(entry_price, lows[entry_bar_idx])
            position = {
                'dir': entry_sig['direction'],
                'entry_price': entry_price,
                'sl_price': entry_sig['sl_price'],
                'best_price': best_price,
                'entry_bar': entry_bar_idx,
                'signal_bar': i,
            }

            # Check immediate exit on entry bar (same-bar SL/trail)
            atr_here = atr[entry_bar_idx] if not math.isnan(atr[entry_bar_idx]) else atr[i]
            exit_res = sig.check_exit(
                direction=position['dir'],
                entry_price=position['entry_price'],
                best_price=position['best_price'],
                current_high=highs[entry_bar_idx],
                current_low=lows[entry_bar_idx],
                current_close=closes[entry_bar_idx],
                sl_price=position['sl_price'],
                atr_val=atr_here, bars_held=1,
            )
            if exit_res is not None:
                exit_price = exit_res['exit_price']
                reason = exit_res['reason']
                if position['dir'] == 'LONG':
                    raw_pnl = (exit_price / position['entry_price'] - 1) * 100
                else:
                    raw_pnl = (1 - exit_price / position['entry_price']) * 100
                pnl_3x = raw_pnl * LEVERAGE - FEE_RT_PCT * LEVERAGE
                trades.append({
                    'signal_bar_ts': ts[i],
                    'signal_bar_dt': datetime.fromtimestamp(ts[i]/1000, tz=timezone.utc).strftime('%Y-%m-%d %H:%M UTC'),
                    'entry_bar_ts': ts[entry_bar_idx],
                    'entry_bar_dt': datetime.fromtimestamp(ts[entry_bar_idx]/1000, tz=timezone.utc).strftime('%Y-%m-%d %H:%M UTC'),
                    'exit_bar_ts': ts[entry_bar_idx],
                    'exit_bar_dt': datetime.fromtimestamp(ts[entry_bar_idx]/1000, tz=timezone.utc).strftime('%Y-%m-%d %H:%M UTC'),
                    'dir': position['dir'],
                    'entry_price': round(position['entry_price'], 2),
                    'exit_price': round(exit_price, 2),
                    'sl_price': round(position['sl_price'], 2),
                    'best_price': round(position['best_price'], 2),
                    'raw_pnl_pct': round(raw_pnl, 4),
                    'pnl_3x_pct': round(pnl_3x, 4),
                    'reason': reason,
                    'bars_held': 1,
                })
                last_exit_bar = entry_bar_idx
                position = None

    return trades, ts, opens, highs, lows, closes


def match_trades(bt_trades, live_trades):
    """Match live trades to backtest trades.

    Primary key: live entry-price within ±0.5% of backtest entry (trade on
    same breakout). We also check direction and time proximity.
    """
    matches = []
    unmatched_live = []
    unmatched_bt = list(range(len(bt_trades)))  # indices

    for live in live_trades:
        best_j = None; best_score = 1e18
        for j, bt in enumerate(bt_trades):
            if bt['dir'] != live['dir']:
                continue
            price_diff_pct = abs(bt['entry_price'] - live['entry']) / live['entry'] * 100
            if price_diff_pct > 1.0:
                continue
            # Use absolute price-pct diff as primary score
            score = price_diff_pct
            if score < best_score:
                best_score = score; best_j = j

        if best_j is None:
            unmatched_live.append({**live, 'reason_no_match': 'no same-dir bt trade within 1.0% of live entry'})
            continue

        bt = bt_trades[best_j]
        # Exit reason alignment
        live_reason_simplified = 'SL' if 'SL' in live['reason'] else ('TRAIL' if 'TRAIL' in live['reason'] else live['reason'])
        bt_reason_simplified = 'SL' if bt['reason'] in ('SL', 'EMERGENCY') else ('TRAIL' if bt['reason'] == 'TRAIL_TP' else bt['reason'])
        reason_match = live_reason_simplified == bt_reason_simplified

        # PnL delta
        pnl_delta = bt['pnl_3x_pct'] - live['pnl_3x']
        entry_delta_pct = (bt['entry_price'] - live['entry']) / live['entry'] * 100
        exit_delta_pct  = (bt['exit_price']  - live['exit'])  / live['exit']  * 100

        matches.append({
            'live_id': live['id'],
            'live_dir': live['dir'],
            'live_entry': live['entry'],
            'live_exit': live['exit'],
            'live_pnl_3x': live['pnl_3x'],
            'live_reason': live['reason'],
            'live_bars': live['bars'],
            'bt_dir': bt['dir'],
            'bt_entry': bt['entry_price'],
            'bt_exit': bt['exit_price'],
            'bt_pnl_3x': bt['pnl_3x_pct'],
            'bt_reason': bt['reason'],
            'bt_bars': bt['bars_held'],
            'bt_signal_dt': bt['signal_bar_dt'],
            'bt_entry_dt': bt['entry_bar_dt'],
            'bt_exit_dt': bt['exit_bar_dt'],
            'entry_price_delta_pct': round(entry_delta_pct, 3),
            'exit_price_delta_pct': round(exit_delta_pct, 3),
            'pnl_3x_delta': round(pnl_delta, 3),
            'reason_match': reason_match,
            'bars_delta': bt['bars_held'] - live['bars'],
        })
        if best_j in unmatched_bt:
            unmatched_bt.remove(best_j)

    unmatched_bt_trades = [bt_trades[j] for j in unmatched_bt]
    return matches, unmatched_live, unmatched_bt_trades


def analyze_intrabar_vs_close(bt_trades, ts, opens, highs, lows, closes, live_trades):
    """For each live trade, locate its entry bar in the fetched candles and
    re-simulate what bar-close logic would have done bar-by-bar.

    Produces a per-trade 'close path' timeline so we can see whether a live
    exchange-side trail fired on a wick that the bar-close backtest never
    reached.
    """
    # Build timestamp → index lookup
    results = []
    for live in live_trades:
        live_exit_time = datetime.fromisoformat(live['exit_time'].replace('Z', '+00:00') if 'Z' in live['exit_time'] else live['exit_time'])
        if live_exit_time.tzinfo is None:
            live_exit_time = live_exit_time.replace(tzinfo=timezone.utc)
        live_exit_ms = int(live_exit_time.timestamp() * 1000)

        # Locate bar containing the live exit (bar whose start <= exit_time < bar_start+900s)
        exit_bar_idx = None
        for i, t in enumerate(ts):
            if t <= live_exit_ms < t + 15 * 60 * 1000:
                exit_bar_idx = i; break
        if exit_bar_idx is None:
            results.append({'live_id': live['id'], 'note': 'exit bar not located in fetched candles'})
            continue

        # Locate entry bar by matching entry-price
        entry_bar_idx = None
        # scan up to 5 bars before exit looking for a bar whose open is near live entry
        for i in range(max(1, exit_bar_idx - live['bars'] - 2),
                       min(len(ts), exit_bar_idx + 1)):
            if abs(opens[i] - live['entry']) / live['entry'] < 0.005:
                entry_bar_idx = i; break
        if entry_bar_idx is None:
            # fall back: assume entry bar = exit_bar - bars_held
            entry_bar_idx = max(1, exit_bar_idx - max(1, live['bars']))

        # Re-simulate bar-close logic starting from entry bar + 1
        # Track best_price intrabar, but only CLOSE would have exited via trail
        sig = C1BreakoutSignal(CONFIG)
        atr_full = compute_atr(highs, lows, closes, CONFIG['atr_period'])

        entry_price = live['entry']
        if live['dir'] == 'LONG':
            best_price = max(entry_price, highs[entry_bar_idx])
        else:
            best_price = min(entry_price, lows[entry_bar_idx])

        # SL: re-compute using the same signal-bar swing (we don't have it —
        # approximate from max_sl_atr cap using live entry)
        atr_entry = atr_full[entry_bar_idx] if not math.isnan(atr_full[entry_bar_idx]) else atr_full[entry_bar_idx - 1]
        # We can't perfectly reproduce fractal SL without signal-bar context,
        # so use max_sl_atr cap as a conservative upper bound on SL distance.
        if live['dir'] == 'LONG':
            approx_sl = entry_price - CONFIG['max_sl_atr'] * atr_entry
        else:
            approx_sl = entry_price + CONFIG['max_sl_atr'] * atr_entry

        # Bar-close walk
        path = []
        bar_close_exit_bar = None
        bar_close_exit_reason = None
        bar_close_exit_price = None

        for j in range(entry_bar_idx, min(exit_bar_idx + 2, len(ts))):
            bars_held = j - entry_bar_idx
            if live['dir'] == 'LONG':
                best_price = max(best_price, highs[j])
                worst_pnl_intrabar = (lows[j] / entry_price - 1) * 100
                close_pnl = (closes[j] / entry_price - 1) * 100
                best_pnl = (best_price / entry_price - 1) * 100
            else:
                best_price = min(best_price, lows[j])
                worst_pnl_intrabar = (1 - highs[j] / entry_price) * 100
                close_pnl = (1 - closes[j] / entry_price) * 100
                best_pnl = (1 - best_price / entry_price) * 100

            atr_j = atr_full[j] if (j < len(atr_full) and not math.isnan(atr_full[j])) else atr_entry
            trail_dist_pct = CONFIG['trail_K'] * atr_j / closes[j] * 100 if closes[j] > 0 else float('inf')
            trail_armed = best_pnl > CONFIG['trail_activation_pct']
            drawdown_from_best = best_pnl - close_pnl
            trail_would_fire_bar_close = trail_armed and drawdown_from_best >= trail_dist_pct

            path.append({
                'bar_idx': j,
                'bar_dt': datetime.fromtimestamp(ts[j]/1000, tz=timezone.utc).strftime('%m-%d %H:%M'),
                'bars_held': bars_held,
                'high': round(highs[j], 1),
                'low': round(lows[j], 1),
                'close': round(closes[j], 1),
                'best_price': round(best_price, 1),
                'worst_pnl_intrabar_pct': round(worst_pnl_intrabar, 3),
                'close_pnl_pct': round(close_pnl, 3),
                'best_pnl_pct': round(best_pnl, 3),
                'trail_dist_pct': round(trail_dist_pct, 3),
                'trail_armed': trail_armed,
                'trail_fires_bar_close': trail_would_fire_bar_close,
            })

            if bar_close_exit_bar is None:
                if live['dir'] == 'LONG' and lows[j] <= approx_sl:
                    bar_close_exit_bar = j; bar_close_exit_reason = 'SL_approx'; bar_close_exit_price = approx_sl
                elif live['dir'] == 'SHORT' and highs[j] >= approx_sl:
                    bar_close_exit_bar = j; bar_close_exit_reason = 'SL_approx'; bar_close_exit_price = approx_sl
                elif worst_pnl_intrabar <= -CONFIG['emergency_sl_pct']:
                    bar_close_exit_bar = j; bar_close_exit_reason = 'EMERGENCY'
                    if live['dir'] == 'LONG':
                        bar_close_exit_price = entry_price * (1 - CONFIG['emergency_sl_pct']/100)
                    else:
                        bar_close_exit_price = entry_price * (1 + CONFIG['emergency_sl_pct']/100)
                elif trail_would_fire_bar_close:
                    realized = max(0, best_pnl - trail_dist_pct)
                    if live['dir'] == 'LONG':
                        bar_close_exit_price = entry_price * (1 + realized/100)
                    else:
                        bar_close_exit_price = entry_price * (1 - realized/100)
                    bar_close_exit_bar = j; bar_close_exit_reason = 'TRAIL_TP'

        # Diagnose divergence
        live_exit_bar_offset = exit_bar_idx - entry_bar_idx
        if bar_close_exit_bar is None:
            divergence = 'bar_close_never_exits_in_window'
            bar_close_pnl_3x = None
        else:
            bar_close_offset = bar_close_exit_bar - entry_bar_idx
            if live['dir'] == 'LONG':
                bc_raw = (bar_close_exit_price / entry_price - 1) * 100
            else:
                bc_raw = (1 - bar_close_exit_price / entry_price) * 100
            bar_close_pnl_3x = bc_raw * LEVERAGE - FEE_RT_PCT * LEVERAGE
            if bar_close_offset == live_exit_bar_offset:
                divergence = 'same_bar_exit'
            elif bar_close_offset < live_exit_bar_offset:
                divergence = 'bar_close_exits_earlier'
            else:
                divergence = 'bar_close_exits_later'

        results.append({
            'live_id': live['id'],
            'live_dir': live['dir'],
            'live_entry': live['entry'],
            'live_exit': live['exit'],
            'live_pnl_3x': live['pnl_3x'],
            'live_reason': live['reason'],
            'live_bars_held': live['bars'],
            'entry_bar_idx': entry_bar_idx,
            'entry_bar_dt': datetime.fromtimestamp(ts[entry_bar_idx]/1000, tz=timezone.utc).strftime('%m-%d %H:%M'),
            'live_exit_bar_idx': exit_bar_idx,
            'live_exit_bar_offset': live_exit_bar_offset,
            'bar_close_exit_bar_offset': (bar_close_exit_bar - entry_bar_idx) if bar_close_exit_bar is not None else None,
            'bar_close_exit_reason': bar_close_exit_reason,
            'bar_close_exit_price': round(bar_close_exit_price, 2) if bar_close_exit_price else None,
            'bar_close_pnl_3x': round(bar_close_pnl_3x, 3) if bar_close_pnl_3x is not None else None,
            'divergence': divergence,
            'bar_path': path,
            'note': f"approx_sl used (real fractal SL not reconstructable without signal-bar context)",
        })
    return results


def shake_out_sequence(trades, key_entry='entry_price'):
    """Return the sequence of entry prices in order — to visualize the
    'shaken out then re-enter higher' pattern."""
    return [{'num': i + 1,
             'dir': t.get('dir') or t.get('direction'),
             'entry': t[key_entry] if key_entry in t else t.get('entry'),
             'exit': t.get('exit_price') or t.get('exit'),
             'pnl_3x': t.get('pnl_3x_pct') or t.get('pnl_3x')}
            for i, t in enumerate(trades)]


def main():
    print("=" * 80)
    print("SHAKE-OUT PATTERN VERIFICATION — LIVE vs BACKTEST (Apr 12-17, 2026)")
    print("=" * 80)
    print()
    print("Fetching BTC-USDT 15m swap candles from BingX...")
    candles = fetch_candles()
    print(f"Fetched {len(candles)} candles")
    if candles:
        first_dt = datetime.fromtimestamp(candles[0][0]/1000, tz=timezone.utc)
        last_dt  = datetime.fromtimestamp(candles[-1][0]/1000, tz=timezone.utc)
        print(f"  First: {first_dt} | Last: {last_dt}")
    print()

    print("Running C1 Breakout v2 backtest (production signal code)...")
    bt_trades, ts, opens, highs, lows, closes = run_backtest(candles)
    print(f"Backtest produced {len(bt_trades)} trades")
    print()

    # ── Side-by-side table ──
    print("=" * 110)
    print("BACKTEST TRADES (Apr 12-17)")
    print("=" * 110)
    hdr = f"{'#':>3} {'Dir':>5} {'Signal dt':>17} {'Entry dt':>17} {'Exit dt':>17} {'Entry$':>9} {'Exit$':>9} {'3xPnL%':>8} {'Reason':>10} {'Bars':>4}"
    print(hdr)
    print('-' * len(hdr))
    for i, t in enumerate(bt_trades, 1):
        print(f"{i:>3} {t['dir']:>5} {t['signal_bar_dt'][:16]:>17} {t['entry_bar_dt'][:16]:>17} {t['exit_bar_dt'][:16]:>17} "
              f"{t['entry_price']:>9.1f} {t['exit_price']:>9.1f} {t['pnl_3x_pct']:>+8.3f} {t['reason']:>10} {t['bars_held']:>4}")
    print()

    # ── Match against live ──
    matches, unmatched_live, unmatched_bt = match_trades(bt_trades, LIVE_TRADES)

    print("=" * 110)
    print(f"LIVE ↔ BACKTEST MATCHING — {len(matches)} matches, {len(unmatched_live)} unmatched live, {len(unmatched_bt)} unmatched bt")
    print("=" * 110)
    hdr = f"{'Live#':>5} {'Dir':>5} {'LiveEntry':>10} {'LiveExit':>10} {'LivePnL3x':>10} {'LiveRsn':>10} | {'BTEntry':>10} {'BTExit':>10} {'BTPnL3x':>10} {'BTRsn':>10} | {'Δentry%':>8} {'Δexit%':>8} {'ΔPnL':>8} {'RsnOK':>6}"
    print(hdr); print('-' * len(hdr))
    for m in matches:
        print(f"{m['live_id']:>5} {m['live_dir']:>5} {m['live_entry']:>10.1f} {m['live_exit']:>10.1f} {m['live_pnl_3x']:>+10.3f} {m['live_reason']:>10} | "
              f"{m['bt_entry']:>10.1f} {m['bt_exit']:>10.1f} {m['bt_pnl_3x']:>+10.3f} {m['bt_reason']:>10} | "
              f"{m['entry_price_delta_pct']:>+8.3f} {m['exit_price_delta_pct']:>+8.3f} {m['pnl_3x_delta']:>+8.3f} {str(m['reason_match']):>6}")
    print()

    if unmatched_live:
        print("UNMATCHED LIVE TRADES (backtest produced no matching same-dir trade within 1% entry):")
        for u in unmatched_live:
            print(f"  Live #{u['id']}: {u['dir']} @{u['entry']} → {u['exit']} pnl3x={u['pnl_3x']:+.3f}% reason={u['reason']}")
        print()

    if unmatched_bt:
        print("EXTRA BACKTEST TRADES (no live counterpart):")
        for t in unmatched_bt:
            print(f"  BT {t['dir']} @{t['entry_price']} → {t['exit_price']} pnl3x={t['pnl_3x_pct']:+.3f}% reason={t['reason']} signal={t['signal_bar_dt']}")
        print()

    # ── Intrabar vs close analysis ──
    print("=" * 110)
    print("INTRABAR vs BAR-CLOSE DIAGNOSIS (per live trade)")
    print("=" * 110)
    ib = analyze_intrabar_vs_close(bt_trades, ts, opens, highs, lows, closes, LIVE_TRADES)
    for r in ib:
        if 'divergence' not in r:
            print(f"Live #{r['live_id']}: {r.get('note')}"); continue
        tag = {'same_bar_exit': 'OK', 'bar_close_exits_earlier': 'BT-EARLY',
               'bar_close_exits_later': 'BT-LATE',
               'bar_close_never_exits_in_window': 'BT-NEVER'}[r['divergence']]
        bc_pnl = f"{r['bar_close_pnl_3x']:+.3f}%" if r['bar_close_pnl_3x'] is not None else 'none'
        bc_off = r['bar_close_exit_bar_offset'] if r['bar_close_exit_bar_offset'] is not None else 'N/A'
        print(f"Live #{r['live_id']:>2} {r['live_dir']:>5} @{r['live_entry']:.1f}: "
              f"live exit bar={r['live_exit_bar_offset']}, bar-close exit bar={bc_off} ({r['bar_close_exit_reason']}, {bc_pnl})  "
              f"[{tag}]")
    print()

    # ── Shake-out sequence ──
    print("=" * 110)
    print("SHAKE-OUT SEQUENCE — entry prices in chronological order")
    print("=" * 110)
    print(f"{'#':>3} {'Dir':>5} {'LiveEntry':>10} {'LiveExit':>10} {'LivePnL3x':>10} | {'BTEntry':>10} {'BTExit':>10} {'BTPnL3x':>10}")
    # Zip live with matched BT
    live_by_id = {l['id']: l for l in LIVE_TRADES}
    match_by_live = {m['live_id']: m for m in matches}
    for live in LIVE_TRADES:
        m = match_by_live.get(live['id'])
        if m:
            print(f"{live['id']:>3} {live['dir']:>5} {live['entry']:>10.1f} {live['exit']:>10.1f} {live['pnl_3x']:>+10.3f} | "
                  f"{m['bt_entry']:>10.1f} {m['bt_exit']:>10.1f} {m['bt_pnl_3x']:>+10.3f}")
        else:
            print(f"{live['id']:>3} {live['dir']:>5} {live['entry']:>10.1f} {live['exit']:>10.1f} {live['pnl_3x']:>+10.3f} | (no bt match)")
    print()

    # Detect shake-out pattern: consecutive same-direction entries at rising/falling prices
    def count_shake_out_patterns(trade_list, price_key='entry'):
        shakes = []
        for i in range(len(trade_list) - 1):
            cur = trade_list[i]; nxt = trade_list[i + 1]
            if cur.get('dir', cur.get('direction')) == 'LONG' and nxt.get('dir', nxt.get('direction')) == 'LONG':
                cur_e = cur.get(price_key, cur.get('entry_price'))
                nxt_e = nxt.get(price_key, nxt.get('entry_price'))
                cur_pnl = cur.get('pnl_3x', cur.get('pnl_3x_pct'))
                if nxt_e > cur_e and cur_pnl <= 0:
                    shakes.append((cur, nxt, 'LONG re-entry higher after loss'))
            if cur.get('dir', cur.get('direction')) == 'SHORT' and nxt.get('dir', nxt.get('direction')) == 'SHORT':
                cur_e = cur.get(price_key, cur.get('entry_price'))
                nxt_e = nxt.get(price_key, nxt.get('entry_price'))
                cur_pnl = cur.get('pnl_3x', cur.get('pnl_3x_pct'))
                if nxt_e < cur_e and cur_pnl <= 0:
                    shakes.append((cur, nxt, 'SHORT re-entry lower after loss'))
        return shakes

    live_shakes = count_shake_out_patterns(LIVE_TRADES, 'entry')
    bt_shakes = count_shake_out_patterns(bt_trades, 'entry_price')
    print(f"Shake-out chains (same-dir consecutive loss→re-entry-further): live={len(live_shakes)}, bt={len(bt_shakes)}")
    for c, n, desc in live_shakes:
        print(f"  LIVE: #{c['id']}→#{n['id']} {desc} ({c['entry']:.0f} {c['pnl_3x']:+.2f}% → {n['entry']:.0f})")
    for c, n, desc in bt_shakes:
        print(f"  BT:   {desc} ({c['entry_price']:.0f} {c['pnl_3x_pct']:+.2f}% → {n['entry_price']:.0f})")
    print()

    # Zero-PnL trails
    zero_live = [l for l in LIVE_TRADES if l['reason'] == 'TRAIL_TP' and abs(l['pnl_3x'] + 0.3) < 0.01]
    zero_bt = [t for t in bt_trades if t['reason'] == 'TRAIL_TP' and abs(t['pnl_3x_pct'] + 0.3) < 0.05]
    print(f"Zero-profit TRAIL_TP exits (−0.30% = 0% gross, only fees): live={len(zero_live)}, bt={len(zero_bt)}")
    for z in zero_live:
        print(f"  LIVE #{z['id']}: {z['dir']} @{z['entry']} (bars={z['bars']})")
    for z in zero_bt:
        print(f"  BT:        {z['dir']} @{z['entry_price']} (bars={z['bars_held']}, signal={z['signal_bar_dt']})")
    print()

    # ── Summary metrics ──
    live_total_3x = sum(l['pnl_3x'] for l in LIVE_TRADES)
    bt_total_3x = sum(t['pnl_3x_pct'] for t in bt_trades)
    print("=" * 110)
    print("AGGREGATE")
    print("=" * 110)
    print(f"  Live: {len(LIVE_TRADES)} trades, total 3x PnL = {live_total_3x:+.3f}%, "
          f"wins = {sum(1 for l in LIVE_TRADES if l['pnl_3x'] > 0)}")
    print(f"  BT:   {len(bt_trades)} trades, total 3x PnL = {bt_total_3x:+.3f}%, "
          f"wins = {sum(1 for t in bt_trades if t['pnl_3x_pct'] > 0)}")
    print(f"  Match rate: {len(matches)}/{len(LIVE_TRADES)} live trades matched to backtest")
    print(f"  Reason agreement among matched: {sum(1 for m in matches if m['reason_match'])}/{len(matches)}")
    print()

    # Save JSON
    output = {
        'metadata': {
            'script': 'shake_out_pattern_verification.py',
            'date_run': datetime.now().isoformat(),
            'eval_start': str(EVAL_START_UTC),
            'eval_end': str(EVAL_END_UTC),
            'data_source': 'BingX swap 15m BTC-USDT via ccxt',
            'config': CONFIG,
            'leverage': LEVERAGE,
            'fee_rt_pct': FEE_RT_PCT,
            'note_on_mainnet': 'Bot trades on mainnet API keys; ccxt bingx swap fetch reads same public data',
        },
        'live_trades': LIVE_TRADES,
        'backtest_trades': bt_trades,
        'matches': matches,
        'unmatched_live': unmatched_live,
        'unmatched_bt': unmatched_bt,
        'intrabar_vs_close_analysis': ib,
        'shake_out_analysis': {
            'live_shake_chains': len(live_shakes),
            'bt_shake_chains': len(bt_shakes),
        },
        'zero_profit_trails': {
            'live_count': len(zero_live),
            'bt_count': len(zero_bt),
        },
        'aggregate': {
            'live_total_pnl_3x': round(live_total_3x, 3),
            'bt_total_pnl_3x': round(bt_total_3x, 3),
            'match_rate': f"{len(matches)}/{len(LIVE_TRADES)}",
            'reason_agreement': f"{sum(1 for m in matches if m['reason_match'])}/{len(matches)}",
        },
    }
    out_path = os.path.join(REPO_ROOT, 'results', 'shake_out_pattern_verification.json')
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"Results saved to: {out_path}")


if __name__ == '__main__':
    main()
