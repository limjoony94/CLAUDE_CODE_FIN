#!/usr/bin/env python3
"""
Time-of-Day Trading Performance Study (v1.53.0, 303d data)
============================================================

Analyze whether specific hours-of-day have significantly different
win rates or PnL, and test whether time-based entry filtering improves
risk-adjusted returns.

Phase 1: Hour-of-day trade distribution and WR heatmap
Phase 2: Block worst N hours — sweep N=1..6
Phase 3: Block worst hours per direction (LONG vs SHORT separately)
Phase 4: WF 3-fold validation of best configs

Standard Research Protocol: LEVERAGE=3, FEE×LEV, ATR-scaled, Compound
"""

import os
import sys
import json
import warnings
from datetime import datetime

import numpy as np
import pandas as pd

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
sys.path.insert(0, _PROJECT_ROOT)

from scripts.analysis.stack_resolution_study import (
    load_and_classify, compute_atr_ratio, compute_ema_slope,
    find_neutral_window, calc_stats,
    DATA_FILE, PATTERNS_FILE,
    LEVERAGE, FEE_PCT, SLIPPAGE_BUFFER, N_SLOTS,
    MDD_FULL_BELOW, MDD_MIN_ABOVE, MDD_MIN_SCALE,
    ATR_PERIOD, ATR_WINDOW,
)

warnings.filterwarnings('ignore')

OUTPUT_FILE = os.path.join(_PROJECT_ROOT, 'results', 'time_of_day_study.json')

DIRECTION_CAP = 7
CASCADE_TIGHTEN_PCT = 85
CASCADE_MULT = 1.0 - CASCADE_TIGHTEN_PCT / 100.0
MOM_THRESHOLD = 1.5
MOM_LOOKBACK = 3
MOM_COOLDOWN = 12
TIMEOUT_BARS = 288
AGG_COUNTER = 8.0
AGG_WITH = 15.0
ATR_LO = 0.5
ATR_HI = 1.5
EARLY_CONFIRM = 3
EARLY_MIN_PROFIT = 0.3


def clamp(value, lo, hi):
    return max(lo, min(hi, value))


def portfolio_sim_tod(
    signal_tuples, opens, highs, lows, closes, type_codes, n_bars,
    atr_ratio, ema_slope, start_bar, end_bar,
    blocked_hours=None, blocked_hours_long=None, blocked_hours_short=None,
    timestamps=None,
):
    """Portfolio sim v1.53.0 with time-of-day filtering.

    blocked_hours: set of UTC hours (0-23) to block ALL entries
    blocked_hours_long: set of UTC hours to block LONG entries only
    blocked_hours_short: set of UTC hours to block SHORT entries only
    timestamps: array of datetime for each bar (for hour extraction)
    """
    fee = FEE_PCT * LEVERAGE
    size_pct = 100.0 / N_SLOTS

    blocked_hours = blocked_hours or set()
    blocked_hours_long = blocked_hours_long or set()
    blocked_hours_short = blocked_hours_short or set()

    positions = []
    trades = []
    equity = 100.0
    peak_equity = 100.0
    mom_pause_until = {'LONG': -1, 'SHORT': -1}

    sig_by_bar = {}
    for s_bar, pat, direction, tp, sl in signal_tuples:
        if start_bar <= s_bar < end_bar:
            sig_by_bar.setdefault(s_bar, []).append((pat, direction, tp, sl))

    for bar in range(start_bar, end_bar):
        cur_open = opens[bar]
        cur_high = highs[bar]
        cur_low = lows[bar]
        cur_close = closes[bar]
        cur_atr = atr_ratio[bar] if bar < len(atr_ratio) else 1.0
        vol_mult = clamp(cur_atr, ATR_LO, ATR_HI)

        # ---- Momentum guard ----
        if bar >= MOM_LOOKBACK:
            pct_move = (cur_close - closes[bar - MOM_LOOKBACK]) / closes[bar - MOM_LOOKBACK] * 100
            if abs(pct_move) >= MOM_THRESHOLD:
                block_dir = 'SHORT' if pct_move > 0 else 'LONG'
                mom_pause_until[block_dir] = bar + MOM_COOLDOWN

        # ---- Check exits ----
        closed_slots = []
        for i, pos in enumerate(positions):
            age = bar - pos['entry_bar']

            # Timeout
            if age >= TIMEOUT_BARS:
                exit_pnl = (cur_open - pos['entry_price']) / pos['entry_price'] * 100
                if pos['direction'] == 'SHORT':
                    exit_pnl = -exit_pnl
                exit_pnl = exit_pnl * LEVERAGE - fee
                closed_slots.append((i, exit_pnl, 'TIMEOUT'))
                continue

            # TP/SL intrabar
            tp_dist = abs(pos['tp_price'] - cur_open)
            sl_dist = abs(pos['sl_price'] - cur_open)

            if pos['direction'] == 'LONG':
                tp_hit = cur_high >= pos['tp_price']
                sl_hit = cur_low <= pos['sl_price']
            else:
                tp_hit = cur_low <= pos['tp_price']
                sl_hit = cur_high >= pos['sl_price']

            if tp_hit and sl_hit:
                if tp_dist <= sl_dist:
                    exit_pnl = pos['tp_pct'] * LEVERAGE - fee
                    closed_slots.append((i, exit_pnl, 'TP'))
                else:
                    exit_pnl = -pos['sl_pct'] * LEVERAGE - fee
                    closed_slots.append((i, exit_pnl, 'SL'))
            elif tp_hit:
                exit_pnl = pos['tp_pct'] * LEVERAGE - fee
                closed_slots.append((i, exit_pnl, 'TP'))
            elif sl_hit:
                exit_pnl = -pos['sl_pct'] * LEVERAGE - fee
                closed_slots.append((i, exit_pnl, 'SL'))
                continue

            # Early exit
            if pos['direction'] == 'LONG':
                cur_pnl_pct = (cur_close - pos['entry_price']) / pos['entry_price'] * 100
            else:
                cur_pnl_pct = (pos['entry_price'] - cur_close) / pos['entry_price'] * 100

            if cur_pnl_pct >= EARLY_MIN_PROFIT and age >= EARLY_CONFIRM:
                cancel_type = 'BD' if pos['direction'] == 'LONG' else 'BU'
                consecutive = 0
                for k in range(max(0, bar - EARLY_CONFIRM + 1), bar + 1):
                    if k < len(type_codes) and type_codes[k] == cancel_type:
                        consecutive += 1
                    else:
                        consecutive = 0
                if consecutive >= EARLY_CONFIRM:
                    exit_pnl = cur_pnl_pct * LEVERAGE - fee
                    closed_slots.append((i, exit_pnl, f'EARLY_{cancel_type}'))

        # Process closes
        for idx, exit_pnl, reason in sorted(closed_slots, key=lambda x: x[0], reverse=True):
            pos_closed = positions.pop(idx)
            slot_pnl = exit_pnl * size_pct / 100.0

            # MDD sizing
            dd = (peak_equity - equity) / peak_equity * 100 if peak_equity > 0 else 0
            if dd <= MDD_FULL_BELOW:
                mdd_scale = 1.0
            elif dd >= MDD_MIN_ABOVE:
                mdd_scale = MDD_MIN_SCALE
            else:
                mdd_scale = 1.0 - (1.0 - MDD_MIN_SCALE) * (dd - MDD_FULL_BELOW) / (MDD_MIN_ABOVE - MDD_FULL_BELOW)
            slot_pnl *= mdd_scale

            equity += equity * slot_pnl / 100.0
            if equity > peak_equity:
                peak_equity = equity
            trades.append({
                'pnl': slot_pnl, 'reason': reason,
                'direction': pos_closed['direction'],
                'pattern': pos_closed['pattern'],
                'entry_bar': pos_closed['entry_bar'],
                'exit_bar': bar,
            })

            # Cascade SL
            if reason == 'SL':
                for p in positions:
                    if p['direction'] == pos_closed['direction']:
                        p['sl_pct'] *= CASCADE_MULT

        # ---- New entries ----
        if bar in sig_by_bar:
            for pat, direction, tp_base, sl_base in sig_by_bar[bar]:
                if len(positions) >= N_SLOTS:
                    break

                # Time-of-day filter
                if timestamps is not None and bar < len(timestamps):
                    hour = pd.Timestamp(timestamps[bar]).hour
                    if hour in blocked_hours:
                        continue
                    if direction == 'LONG' and hour in blocked_hours_long:
                        continue
                    if direction == 'SHORT' and hour in blocked_hours_short:
                        continue

                # Momentum guard
                if mom_pause_until.get(direction, -1) > bar:
                    continue

                # Direction cap
                dir_count = sum(1 for p in positions if p['direction'] == direction)
                if dir_count >= DIRECTION_CAP:
                    continue

                # Duplicate pattern check
                active_pats = {p['pattern'] for p in positions}
                if pat in active_pats:
                    continue

                # Aggregate risk cap
                slope_val = ema_slope[bar] if bar < len(ema_slope) else 0
                is_up = slope_val > 0
                is_counter = (direction == 'SHORT' and is_up) or (direction == 'LONG' and not is_up)
                cap = AGG_COUNTER if is_counter else AGG_WITH
                agg_exp = sum(
                    p['sl_pct'] * vol_mult * LEVERAGE / N_SLOTS
                    for p in positions if p['direction'] == direction
                )
                new_exp = sl_base * vol_mult * LEVERAGE / N_SLOTS
                if agg_exp + new_exp > cap:
                    continue

                entry_price = opens[bar + 1] if bar + 1 < n_bars else cur_close
                tp_pct = tp_base * vol_mult
                sl_pct = sl_base * vol_mult

                # Proportional cap
                max_sl = 13.0 / LEVERAGE / (100.0 / N_SLOTS) * 100.0
                if sl_pct > max_sl:
                    ratio = max_sl / sl_pct
                    sl_pct = max_sl
                    tp_pct *= ratio

                if direction == 'LONG':
                    tp_price = entry_price * (1 + tp_pct / 100)
                    sl_price = entry_price * (1 - sl_pct / 100)
                else:
                    tp_price = entry_price * (1 - tp_pct / 100)
                    sl_price = entry_price * (1 + sl_pct / 100)

                positions.append({
                    'pattern': pat, 'direction': direction,
                    'entry_bar': bar + 1, 'entry_price': entry_price,
                    'tp_price': tp_price, 'sl_price': sl_price,
                    'tp_pct': tp_pct, 'sl_pct': sl_pct,
                })

    return trades, equity, peak_equity


def compute_stats(trades, equity, peak_equity):
    """Compute stats from portfolio sim output."""
    if not trades:
        return {'trades': 0, 'wr': 0.0, 'pnl': 0.0, 'mdd': 0.0, 'pnl_mdd': 0.0}
    n = len(trades)
    wins = sum(1 for t in trades if t['pnl'] > 0)
    pnl = equity - 100.0
    mdd = (peak_equity - min(equity, peak_equity)) / peak_equity * 100 if peak_equity > 0 else 0

    # Recompute MDD from trade sequence
    eq = 100.0
    pk = eq
    max_dd = 0.0
    sorted_t = sorted(trades, key=lambda x: x['entry_bar'])
    for t in sorted_t:
        eq += eq * t['pnl'] / 100.0
        if eq > pk:
            pk = eq
        dd = (pk - eq) / pk * 100
        if dd > max_dd:
            max_dd = dd

    return {
        'trades': n,
        'wr': wins / n * 100,
        'pnl': pnl,
        'mdd': max_dd if max_dd > 0 else 0.01,
        'pnl_mdd': pnl / max_dd if max_dd > 0 else pnl / 0.01,
    }


def build_signal_index(df, patterns_info):
    """Build signal tuples from classified data."""
    type_codes_list = df['type_code'].tolist()
    n = len(type_codes_list)
    signals = []
    for i in range(2, n - 1):
        pat3 = f"{type_codes_list[i-2]}-{type_codes_list[i-1]}-{type_codes_list[i]}"
        if pat3 in patterns_info:
            info = patterns_info[pat3]
            signals.append((i, pat3, info['direction'], info['tp'], info['sl']))
    return signals


def expanding_window_wf(df, patterns_info, timestamps, n_folds=3, **sim_kwargs):
    """Expanding window WF with time-of-day filtering."""
    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    type_codes = df['type_code'].values
    n_bars = len(df)

    atr_ratio = compute_atr_ratio(df)
    ema_slope = compute_ema_slope(closes)

    signals = build_signal_index(df, patterns_info)
    fold_size = n_bars // (n_folds + 1)

    results = []
    for fold in range(n_folds):
        is_end = fold_size * (fold + 2)
        oos_start = is_end
        oos_end = min(is_end + fold_size, n_bars)

        trades, eq, pk = portfolio_sim_tod(
            signals, opens, highs, lows, closes, type_codes, n_bars,
            atr_ratio, ema_slope, oos_start, oos_end,
            timestamps=timestamps,
            **sim_kwargs,
        )
        pnl = eq - 100.0
        n_trades = len(trades)
        wins = sum(1 for t in trades if t['pnl'] > 0)
        wr = wins / n_trades * 100 if n_trades > 0 else 0
        mdd = (pk - min(eq, pk)) / pk * 100 if pk > 0 else 0
        results.append({'fold': fold + 1, 'pnl': pnl, 'wr': wr, 'trades': n_trades, 'mdd': mdd})

    return results


def main():
    print("=" * 70)
    print("Time-of-Day Trading Performance Study")
    print("=" * 70)

    # Load data
    df = load_and_classify(DATA_FILE)
    with open(PATTERNS_FILE) as f:
        pdata = json.load(f)
    patterns_info = {}
    long_pats = set(pdata.get('patterns', {}).get('long', []))
    short_pats = set(pdata.get('patterns', {}).get('short', []))
    tpsl = pdata.get('patterns_tpsl', {})
    for pat in long_pats:
        if pat in tpsl:
            patterns_info[pat] = {'direction': 'LONG', 'tp': tpsl[pat][0], 'sl': tpsl[pat][1]}
    for pat in short_pats:
        if pat in tpsl:
            patterns_info[pat] = {'direction': 'SHORT', 'tp': tpsl[pat][0], 'sl': tpsl[pat][1]}
    print(f"Data: {len(df)} bars, Patterns: {len(patterns_info)}")

    # Neutral window
    closes_full = df['close'].values
    start_idx, end_idx = find_neutral_window(closes_full)
    df_neutral = df.iloc[start_idx:end_idx].reset_index(drop=True)
    n_bars = len(df_neutral)
    print(f"Neutral window: {n_bars} bars (idx {start_idx}-{end_idx})")

    # Extract timestamps
    if 'timestamp' in df_neutral.columns:
        timestamps = pd.to_datetime(df_neutral['timestamp'])
    elif 'open_time' in df_neutral.columns:
        timestamps = pd.to_datetime(df_neutral['open_time'], unit='ms')
    else:
        # Assume 5m bars starting from some reference
        print("WARNING: No timestamp column, using index-based hours")
        timestamps = pd.date_range('2025-05-01', periods=n_bars, freq='5min')

    timestamps_arr = timestamps.to_numpy()
    hours = np.array([pd.Timestamp(t).hour for t in timestamps_arr])

    opens = df_neutral['open'].values
    highs = df_neutral['high'].values
    lows = df_neutral['low'].values
    closes = df_neutral['close'].values
    type_codes = df_neutral['candle_type'].values

    atr_ratio = compute_atr_ratio(df_neutral)
    ema_slope = compute_ema_slope(df_neutral['close'].values)

    signals = build_signal_index(df_neutral, patterns_info)

    # ================================================================
    # Phase 1: Hour-of-day analysis
    # ================================================================
    print("\n" + "=" * 70)
    print("Phase 1: Hour-of-Day Trade Distribution & WR")
    print("=" * 70)

    # Run baseline to get all trades with entry bar info
    baseline_trades, baseline_eq, baseline_pk = portfolio_sim_tod(
        signals, opens, highs, lows, closes, type_codes, n_bars,
        atr_ratio, ema_slope, 0, n_bars,
        timestamps=timestamps_arr,
    )
    baseline_stats = compute_stats(baseline_trades, baseline_eq, baseline_pk)
    print(f"\nBaseline: {baseline_stats['trades']} trades, WR {baseline_stats['wr']:.1f}%, "
          f"PnL {baseline_stats['pnl']:+.1f}%, MDD {baseline_stats['mdd']:.1f}%, "
          f"PnL/MDD {baseline_stats['pnl_mdd']:.1f}")

    # Analyze by hour
    hour_stats = {}
    for t in baseline_trades:
        entry_bar = t['entry_bar']
        if entry_bar < len(timestamps_arr):
            h = pd.Timestamp(timestamps_arr[entry_bar]).hour
        else:
            continue
        if h not in hour_stats:
            hour_stats[h] = {'wins': 0, 'losses': 0, 'pnl': 0.0, 'trades': []}
        if t['pnl'] > 0:
            hour_stats[h]['wins'] += 1
        else:
            hour_stats[h]['losses'] += 1
        hour_stats[h]['pnl'] += t['pnl']
        hour_stats[h]['trades'].append(t)

    print(f"\n{'Hour':>4s} {'Trades':>7s} {'WR':>6s} {'PnL%':>8s} {'AvgPnL':>8s} {'L_WR':>6s} {'S_WR':>6s}")
    print("-" * 55)

    hour_wr = {}
    hour_pnl = {}
    for h in range(24):
        if h in hour_stats:
            s = hour_stats[h]
            n = s['wins'] + s['losses']
            wr = s['wins'] / n * 100 if n > 0 else 0
            avg_pnl = s['pnl'] / n if n > 0 else 0
            hour_wr[h] = wr
            hour_pnl[h] = s['pnl']

            # Direction split
            long_trades = [t for t in s['trades'] if t['direction'] == 'LONG']
            short_trades = [t for t in s['trades'] if t['direction'] == 'SHORT']
            l_wr = sum(1 for t in long_trades if t['pnl'] > 0) / len(long_trades) * 100 if long_trades else 0
            s_wr = sum(1 for t in short_trades if t['pnl'] > 0) / len(short_trades) * 100 if short_trades else 0
            l_str = f"{l_wr:5.1f}%" if long_trades else "   N/A"
            s_str = f"{s_wr:5.1f}%" if short_trades else "   N/A"

            marker = " ◀ WORST" if wr < 55 and n >= 20 else (" ★ BEST" if wr > 75 and n >= 20 else "")
            print(f"{h:4d} {n:7d} {wr:5.1f}% {s['pnl']:+7.1f}% {avg_pnl:+7.3f}% {l_str} {s_str}{marker}")
        else:
            print(f"{h:4d}       0    N/A      N/A      N/A    N/A    N/A")

    # Rank hours by WR (minimum 20 trades for significance)
    significant_hours = {h: wr for h, wr in hour_wr.items()
                         if h in hour_stats and hour_stats[h]['wins'] + hour_stats[h]['losses'] >= 20}
    sorted_hours = sorted(significant_hours.items(), key=lambda x: x[1])

    print(f"\nSignificant hours (≥20 trades): {len(significant_hours)}")
    print(f"Worst hours: {[(h, f'{wr:.1f}%') for h, wr in sorted_hours[:5]]}")
    print(f"Best hours: {[(h, f'{wr:.1f}%') for h, wr in sorted_hours[-5:]]}")

    # ================================================================
    # Phase 2: Block worst N hours — sweep
    # ================================================================
    print("\n" + "=" * 70)
    print("Phase 2: Block Worst N Hours (uniform)")
    print("=" * 70)

    phase2_results = []
    for n_block in range(1, 7):
        worst_n = set(h for h, _ in sorted_hours[:n_block])
        trades, eq, pk = portfolio_sim_tod(
            signals, opens, highs, lows, closes, type_codes, n_bars,
            atr_ratio, ema_slope, 0, n_bars,
            blocked_hours=worst_n,
            timestamps=timestamps_arr,
        )
        stats = compute_stats(trades, eq, pk)
        delta_pnl_mdd = stats['pnl_mdd'] - baseline_stats['pnl_mdd']
        phase2_results.append({
            'blocked': sorted(worst_n),
            'n_block': n_block,
            **stats,
            'delta_pnl_mdd': delta_pnl_mdd,
        })
        marker = " ★" if delta_pnl_mdd > 5 else ""
        print(f"Block {n_block}h {sorted(worst_n)}: "
              f"trades={stats['trades']}, WR={stats['wr']:.1f}%, "
              f"PnL={stats['pnl']:+.1f}%, MDD={stats['mdd']:.1f}%, "
              f"PnL/MDD={stats['pnl_mdd']:.1f} (Δ{delta_pnl_mdd:+.1f}){marker}")

    # ================================================================
    # Phase 3: Direction-specific hour blocking
    # ================================================================
    print("\n" + "=" * 70)
    print("Phase 3: Direction-Specific Hour Blocking")
    print("=" * 70)

    # Find worst hours per direction
    dir_hour_stats = {'LONG': {}, 'SHORT': {}}
    for t in baseline_trades:
        entry_bar = t['entry_bar']
        if entry_bar < len(timestamps_arr):
            h = pd.Timestamp(timestamps_arr[entry_bar]).hour
        else:
            continue
        d = t['direction']
        if h not in dir_hour_stats[d]:
            dir_hour_stats[d][h] = {'wins': 0, 'losses': 0, 'pnl': 0.0}
        if t['pnl'] > 0:
            dir_hour_stats[d][h]['wins'] += 1
        else:
            dir_hour_stats[d][h]['losses'] += 1
        dir_hour_stats[d][h]['pnl'] += t['pnl']

    phase3_results = []
    for d in ['LONG', 'SHORT']:
        print(f"\n--- {d} Direction ---")
        sig_h = {}
        for h, s in dir_hour_stats[d].items():
            n = s['wins'] + s['losses']
            if n >= 10:
                sig_h[h] = s['wins'] / n * 100

        sorted_d = sorted(sig_h.items(), key=lambda x: x[1])
        print(f"  Worst {d} hours: {[(h, f'{wr:.1f}%') for h, wr in sorted_d[:5]]}")

        for n_block in [1, 2, 3]:
            worst_n = set(h for h, _ in sorted_d[:n_block])
            kwargs = {}
            if d == 'LONG':
                kwargs['blocked_hours_long'] = worst_n
            else:
                kwargs['blocked_hours_short'] = worst_n

            trades, eq, pk = portfolio_sim_tod(
                signals, opens, highs, lows, closes, type_codes, n_bars,
                atr_ratio, ema_slope, 0, n_bars,
                timestamps=timestamps_arr,
                **kwargs,
            )
            stats = compute_stats(trades, eq, pk)
            delta = stats['pnl_mdd'] - baseline_stats['pnl_mdd']
            phase3_results.append({
                'direction': d,
                'blocked': sorted(worst_n),
                'n_block': n_block,
                **stats,
                'delta_pnl_mdd': delta,
            })
            marker = " ★" if delta > 5 else ""
            print(f"  Block {n_block}h {sorted(worst_n)}: "
                  f"trades={stats['trades']}, WR={stats['wr']:.1f}%, "
                  f"PnL/MDD={stats['pnl_mdd']:.1f} (Δ{delta:+.1f}){marker}")

    # ================================================================
    # Phase 4: WF Validation
    # ================================================================
    print("\n" + "=" * 70)
    print("Phase 4: WF 3-fold Validation")
    print("=" * 70)

    # Collect candidates: baseline + top 3 from phase 2 + top 2 from phase 3
    candidates = [
        {'name': 'baseline', 'kwargs': {}},
    ]

    # Top phase 2 configs
    p2_sorted = sorted(phase2_results, key=lambda x: x['pnl_mdd'], reverse=True)
    for cfg in p2_sorted[:3]:
        candidates.append({
            'name': f"block_{cfg['n_block']}h",
            'kwargs': {'blocked_hours': set(cfg['blocked'])},
        })

    # Top phase 3 configs (best per direction)
    p3_sorted = sorted(phase3_results, key=lambda x: x['pnl_mdd'], reverse=True)
    seen = set()
    for cfg in p3_sorted:
        key = f"{cfg['direction']}_{cfg['n_block']}"
        if key not in seen and len(candidates) < 8:
            seen.add(key)
            kw = {}
            if cfg['direction'] == 'LONG':
                kw['blocked_hours_long'] = set(cfg['blocked'])
            else:
                kw['blocked_hours_short'] = set(cfg['blocked'])
            candidates.append({
                'name': f"block_{cfg['direction'][:1]}_{cfg['n_block']}h",
                'kwargs': kw,
            })

    print(f"\nValidating {len(candidates)} candidates...")
    wf_results = []
    for cand in candidates:
        folds = expanding_window_wf(
            df_neutral, patterns_info, timestamps_arr,
            n_folds=3, **cand['kwargs'],
        )
        min_fold = min(f['pnl'] for f in folds)
        all_pass = all(f['pnl'] > 0 for f in folds)
        wf_results.append({
            'name': cand['name'],
            'folds': folds,
            'min_fold': min_fold,
            'all_pass': all_pass,
        })
        status = "PASS" if all_pass else "FAIL"
        fold_str = " | ".join(f"F{f['fold']}: {f['pnl']:+.1f}%" for f in folds)
        print(f"  {cand['name']:25s}: {status} | min={min_fold:+.1f}% | {fold_str}")

    # ================================================================
    # Verdict
    # ================================================================
    print("\n" + "=" * 70)
    print("VERDICT")
    print("=" * 70)

    baseline_wf = next(r for r in wf_results if r['name'] == 'baseline')
    best_alt = None
    best_delta = 0
    for r in wf_results:
        if r['name'] == 'baseline':
            continue
        if r['all_pass']:
            delta = r['min_fold'] - baseline_wf['min_fold']
            if delta > best_delta:
                best_delta = delta
                best_alt = r

    if best_alt and best_delta > 5:
        print(f"ADOPT: {best_alt['name']} (min fold Δ{best_delta:+.1f}%)")
    else:
        print(f"KEEP BASELINE — no time filter improves OOS min fold by >5%")
        if best_alt:
            print(f"  Best alternative: {best_alt['name']} (Δ{best_delta:+.1f}%)")

    # Save results
    output = {
        'timestamp': datetime.now().isoformat(),
        'data_bars': n_bars,
        'patterns': len(patterns_info),
        'baseline': baseline_stats,
        'hour_analysis': {str(h): {
            'trades': hour_stats[h]['wins'] + hour_stats[h]['losses'],
            'wr': hour_wr.get(h, 0),
            'pnl': hour_stats[h]['pnl'],
        } for h in sorted(hour_stats.keys())},
        'phase2_block_hours': phase2_results,
        'phase3_direction': phase3_results,
        'wf_validation': [{
            'name': r['name'],
            'folds': r['folds'],
            'min_fold': r['min_fold'],
            'all_pass': r['all_pass'],
        } for r in wf_results],
        'verdict': best_alt['name'] if best_alt and best_delta > 5 else 'KEEP_BASELINE',
    }
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved: {OUTPUT_FILE}")


if __name__ == '__main__':
    main()
