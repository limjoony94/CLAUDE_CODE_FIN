#!/usr/bin/env python3
"""
MDD Sizing Parameter Sweep Study (v1.46.0 baseline)
=====================================================

MDD Sizing was set in v1.35.2 (full_below=3%, min_above=15%, min_scale=0.25)
based on edge half-life 30d argument. Since then, Cascade SL (v1.41.0→v1.45.0)
fundamentally changed MDD profile — typical MDD is now 3-5% vs 15-25% pre-Cascade.

This study sweeps all 3 MDD sizing parameters to find current optimum.

Phase 1: full_size_below_dd sweep (1-8%)
Phase 2: min_size_above_dd sweep (8-25%)
Phase 3: min_scale sweep (0.1-0.5)
Phase 4: Grid search top combos from Phase 1-3
Phase 5: WF 3-fold validation

Standard Research Protocol: LEVERAGE=3, FEE×LEV, Timeout DROP, ATR-scaled, Compound

Author: Research Agent
Date: 2026-03-05
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
    LEVERAGE, FEE_PCT, SLIPPAGE_BUFFER, TIMEOUT_BARS, N_SLOTS,
    ATR_CLAMP_LO, ATR_CLAMP_HI, BARS_PER_DAY,
    EARLY_CONFIRM, EARLY_MIN_PROFIT,
    clamp,
)

warnings.filterwarnings('ignore')

OUTPUT_FILE = os.path.join(_PROJECT_ROOT, 'results', 'mdd_sizing_sweep_study.json')

# v1.46.0 fixed params
DIRECTION_CAP = 7
CASCADE_TIGHTEN_PCT = 85
CASCADE_MULT = 1.0 - CASCADE_TIGHTEN_PCT / 100.0
AGG_COUNTER_CAP = 5.0
AGG_WITH_CAP = 15.0
MOM_THRESHOLD = 1.0
MOM_LOOKBACK = 3
MOM_COOLDOWN = 12


def portfolio_sim_mdd(
    signal_tuples, opens, highs, lows, closes, type_codes, n_bars,
    atr_ratio, ema_slope, start_bar, end_bar,
    mdd_full_below=3.0, mdd_min_above=15.0, mdd_min_scale=0.25,
):
    """Portfolio sim v1.46.0 with configurable MDD sizing params."""
    fee = FEE_PCT * LEVERAGE
    size_pct = 100.0 / N_SLOTS

    positions = []
    trades = []
    equity = 100.0
    peak_equity = 100.0
    mom_pause_until = {'LONG': -1, 'SHORT': -1}

    sig_by_bar = {}
    for s_bar, pat, direction, tp, sl in signal_tuples:
        if start_bar <= s_bar < end_bar:
            sig_by_bar.setdefault(s_bar, []).append((pat, direction, tp, sl))

    # Track sizing decisions for analysis
    scale_events = []

    for bar in range(start_bar, end_bar):
        if bar >= n_bars - 1:
            break

        closed_slots = set()
        sl_exits = []

        h_bar = highs[bar] if bar < n_bars else 0
        l_bar = lows[bar] if bar < n_bars else 0
        o_bar = opens[bar] if bar < n_bars else 0

        for pos in positions:
            entry_bar = pos['entry_bar']
            if bar < entry_bar:
                continue
            entry_p = pos['entry_price']
            if entry_p <= 0:
                continue

            direction = pos['direction']
            eff_tp = pos['eff_tp_pct']
            eff_sl = pos['eff_sl_pct']
            sm = pos['size_mult']
            hold = bar - entry_bar

            exit_price = None
            reason = None

            if hold >= TIMEOUT_BARS:
                closed_slots.add(pos['slot'])
                continue

            if hold >= EARLY_CONFIRM:
                exit_types = ['BD'] if direction == 'LONG' else ['BU']
                candle_ok = True
                for k in range(EARLY_CONFIRM):
                    cb = bar - 1 - k
                    if cb < 0 or cb >= len(type_codes) or type_codes[cb] not in exit_types:
                        candle_ok = False
                        break
                if candle_ok:
                    cp = closes[bar - 1] if bar - 1 < n_bars else entry_p
                    if direction == 'LONG':
                        unr = (cp / entry_p - 1) * 100 * LEVERAGE
                    else:
                        unr = (1 - cp / entry_p) * 100 * LEVERAGE
                    if unr >= EARLY_MIN_PROFIT:
                        exit_price = cp
                        reason = 'EARLY'

            if reason is None:
                if direction == 'LONG':
                    tp_p = entry_p * (1 + eff_tp / 100)
                    sl_p = entry_p * (1 - eff_sl / 100)
                    hit_tp = h_bar >= tp_p
                    hit_sl = l_bar <= sl_p
                else:
                    tp_p = entry_p * (1 - eff_tp / 100)
                    sl_p = entry_p * (1 + eff_sl / 100)
                    hit_tp = l_bar <= tp_p
                    hit_sl = h_bar >= sl_p

                if hit_tp and hit_sl:
                    if abs(tp_p - o_bar) <= abs(sl_p - o_bar):
                        exit_price, reason = tp_p, 'TP'
                    else:
                        exit_price, reason = sl_p, 'SL'
                elif hit_tp:
                    exit_price, reason = tp_p, 'TP'
                elif hit_sl:
                    exit_price, reason = sl_p, 'SL'

            if reason is None:
                continue

            if direction == 'LONG':
                pnl = (exit_price / entry_p - 1) * 100 * LEVERAGE
            else:
                pnl = (1 - exit_price / entry_p) * 100 * LEVERAGE
            pnl -= fee

            pnl_portfolio = pnl * (size_pct / 100) * sm
            trades.append({
                'entry_bar': entry_bar, 'exit_bar': bar, 'pnl_slot': pnl,
                'reason': reason, 'pattern': pos['pattern'],
                'direction': direction, 'pnl_portfolio': pnl_portfolio,
            })
            closed_slots.add(pos['slot'])

            if reason == 'SL':
                sl_exits.append(pos)

        positions = [p for p in positions if p['slot'] not in closed_slots]

        # Cascade SL tightening (v1.45.0: t85)
        if sl_exits:
            for sl_pos in sl_exits:
                sl_dir = sl_pos['direction']
                for pos in positions:
                    if pos['direction'] == sl_dir:
                        pos['eff_sl_pct'] *= CASCADE_MULT

        equity_delta = sum(t['pnl_portfolio'] for t in trades if t['exit_bar'] == bar)
        equity *= (1 + equity_delta / 100)
        if equity > peak_equity:
            peak_equity = equity

        # Momentum guard (v1.46.0: lb3/cd12)
        if bar >= MOM_LOOKBACK:
            pa = closes[bar - MOM_LOOKBACK]
            if pa > 0:
                pct = (closes[bar] / pa - 1) * 100
                if pct > MOM_THRESHOLD:
                    mom_pause_until['SHORT'] = max(mom_pause_until['SHORT'], bar + MOM_COOLDOWN)
                elif pct < -MOM_THRESHOLD:
                    mom_pause_until['LONG'] = max(mom_pause_until['LONG'], bar + MOM_COOLDOWN)

        if bar not in sig_by_bar:
            continue

        for pat, direction, tp_pct, sl_pct in sig_by_bar[bar]:
            if len(positions) >= N_SLOTS:
                continue
            dir_count = sum(1 for p in positions if p['direction'] == direction)
            if dir_count >= DIRECTION_CAP:
                continue
            if any(p['pattern'] == pat for p in positions):
                continue

            entry_bar = bar + 1
            if entry_bar >= n_bars:
                continue
            entry_price = opens[entry_bar]
            if entry_price <= 0:
                continue

            if bar < mom_pause_until.get(direction, -1):
                continue

            # MDD sizing (configurable)
            sm = 1.0
            if peak_equity > 0:
                dd_pct = (peak_equity - equity) / peak_equity * 100
                if dd_pct <= mdd_full_below:
                    mdd_scale = 1.0
                elif dd_pct >= mdd_min_above:
                    mdd_scale = mdd_min_scale
                else:
                    mdd_scale = 1.0 - (1.0 - mdd_min_scale) * (
                        dd_pct - mdd_full_below) / (mdd_min_above - mdd_full_below)
                sm *= mdd_scale

                if mdd_scale < 1.0:
                    scale_events.append((bar, dd_pct, mdd_scale))

            if bar < len(atr_ratio) and not np.isnan(atr_ratio[bar]):
                r = clamp(atr_ratio[bar], ATR_CLAMP_LO, ATR_CLAMP_HI)
            else:
                r = 1.0

            eff_tp = tp_pct * r + SLIPPAGE_BUFFER
            eff_sl = max(0.1, sl_pct * r - SLIPPAGE_BUFFER)

            # AggRisk cap (v1.44.0: 5/15)
            slope = ema_slope[bar] if bar < len(ema_slope) else 0
            is_uptrend = slope > 0
            is_counter = ((direction == 'SHORT' and is_uptrend) or
                          (direction == 'LONG' and not is_uptrend))
            cap_pct = AGG_COUNTER_CAP if is_counter else AGG_WITH_CAP

            existing = sum(
                p['eff_sl_pct'] * (1.0 / N_SLOTS) * LEVERAGE * p['size_mult']
                for p in positions if p['direction'] == direction
            )
            new_exp = eff_sl * (1.0 / N_SLOTS) * LEVERAGE * sm
            if existing + new_exp > cap_pct:
                continue

            positions.append({
                'slot': f"{pat}_{bar}",
                'entry_bar': entry_bar,
                'entry_price': entry_price,
                'direction': direction,
                'pattern': pat,
                'eff_tp_pct': eff_tp,
                'eff_sl_pct': eff_sl,
                'size_mult': sm,
            })

    # Force-close remaining
    for pos in positions:
        eb = pos['entry_bar']
        if eb >= n_bars:
            continue
        ep = pos['entry_price']
        if ep <= 0:
            continue
        exit_bar = min(end_bar - 1, n_bars - 1)
        exit_price = opens[exit_bar]
        if pos['direction'] == 'LONG':
            pnl = (exit_price / ep - 1) * 100 * LEVERAGE
        else:
            pnl = (1 - exit_price / ep) * 100 * LEVERAGE
        pnl -= fee
        sm = pos['size_mult']
        trades.append({
            'entry_bar': eb, 'exit_bar': exit_bar, 'pnl_slot': pnl,
            'reason': 'FORCE_CLOSE', 'pattern': pos['pattern'],
            'direction': pos['direction'],
            'pnl_portfolio': pnl * (size_pct / 100) * sm,
        })

    return trades, len(scale_events)


def expanding_window_wf(signal_tuples, opens, highs, lows, closes, type_codes,
                        n_bars, atr_ratio, ema_slope, neutral_start, neutral_end,
                        mdd_full_below, mdd_min_above, mdd_min_scale, n_folds=3):
    """3-fold expanding window WF."""
    total_bars = neutral_end - neutral_start
    fold_size = total_bars // (n_folds + 1)

    results = []
    for fold in range(n_folds):
        is_end = neutral_start + fold_size * (fold + 1)
        oos_start = is_end
        oos_end = min(is_end + fold_size, neutral_end)
        if oos_start >= oos_end:
            continue

        oos_trades, _ = portfolio_sim_mdd(
            signal_tuples, opens, highs, lows, closes, type_codes, n_bars,
            atr_ratio, ema_slope, oos_start, oos_end,
            mdd_full_below=mdd_full_below, mdd_min_above=mdd_min_above,
            mdd_min_scale=mdd_min_scale,
        )

        stats = calc_stats(oos_trades)
        results.append(stats['pnl'])

    return results


def main():
    print("=" * 90)
    print("MDD SIZING PARAMETER SWEEP STUDY (v1.46.0 baseline)")
    print("=" * 90)

    # Load data
    print("\nLoading data...")
    df = load_and_classify(DATA_FILE)
    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    type_codes = df['rctype'].values
    n_bars = len(df)
    atr_ratio = compute_atr_ratio(df)
    ema_slope = compute_ema_slope(closes)
    neutral_start, neutral_end = find_neutral_window(closes)
    print(f"  Neutral: {neutral_start}-{neutral_end} ({neutral_end - neutral_start} bars)")

    # Load patterns
    with open(PATTERNS_FILE) as f:
        pat_data = json.load(f)
    pats_raw = pat_data['patterns']
    tpsl = pat_data.get('patterns_tpsl', {})
    patterns = {}
    for pat_name in pats_raw.get('long', []):
        tp_sl = tpsl.get(pat_name, [2.0, 3.0])
        patterns[pat_name] = {'direction': 'LONG', 'tp_pct': tp_sl[0], 'sl_pct': tp_sl[1]}
    for pat_name in pats_raw.get('short', []):
        tp_sl = tpsl.get(pat_name, [2.0, 3.0])
        patterns[pat_name] = {'direction': 'SHORT', 'tp_pct': tp_sl[0], 'sl_pct': tp_sl[1]}
    print(f"  Loaded {len(patterns)} patterns")

    # Build signals
    signal_tuples = []
    for i in range(2, n_bars):
        triplet = f"{type_codes[i-2]}-{type_codes[i-1]}-{type_codes[i]}"
        if triplet in patterns:
            info = patterns[triplet]
            signal_tuples.append((i, triplet, info['direction'], info['tp_pct'], info['sl_pct']))
    neutral_signals = [(b, p, d, tp, sl) for b, p, d, tp, sl in signal_tuples
                       if neutral_start <= b < neutral_end]
    print(f"  Signals: {len(neutral_signals)} in neutral window")

    # ==================== BASELINE ====================
    print("\n" + "=" * 90)
    print("BASELINE: Current config (full_below=3%, min_above=15%, min_scale=0.25)")
    print("=" * 90)

    base_trades, base_scales = portfolio_sim_mdd(
        neutral_signals, opens, highs, lows, closes, type_codes, n_bars,
        atr_ratio, ema_slope, neutral_start, neutral_end,
        mdd_full_below=3.0, mdd_min_above=15.0, mdd_min_scale=0.25,
    )
    base_stats = calc_stats(base_trades)
    print(f"  PnL: {base_stats['pnl']:+.1f}%, MDD: {base_stats['mdd']:.2f}%, "
          f"PnL/MDD: {base_stats['pnl'] / max(base_stats['mdd'], 0.01):.1f}, "
          f"WR: {base_stats['wr']:.1f}%, Trades: {base_stats['trades']}, "
          f"Scale events: {base_scales}")

    # Also test OFF to understand if MDD sizing helps at all
    off_trades, _ = portfolio_sim_mdd(
        neutral_signals, opens, highs, lows, closes, type_codes, n_bars,
        atr_ratio, ema_slope, neutral_start, neutral_end,
        mdd_full_below=100.0, mdd_min_above=101.0, mdd_min_scale=1.0,  # effectively OFF
    )
    off_stats = calc_stats(off_trades)
    print(f"  [OFF] PnL: {off_stats['pnl']:+.1f}%, MDD: {off_stats['mdd']:.2f}%, "
          f"PnL/MDD: {off_stats['pnl'] / max(off_stats['mdd'], 0.01):.1f}")

    all_results = {}

    # ==================== PHASE 1: full_size_below_dd sweep ====================
    print("\n" + "=" * 90)
    print("PHASE 1: full_size_below_dd sweep (min_above=15, min_scale=0.25 fixed)")
    print("=" * 90)

    full_below_vals = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0]
    print(f"\n{'full_below':>12} {'PnL%':>8} {'MDD%':>8} {'P/M':>8} {'WR%':>6} "
          f"{'Trades':>7} {'Scales':>7}")
    print("-" * 65)

    p1_results = {}
    for fb in full_below_vals:
        trades, n_sc = portfolio_sim_mdd(
            neutral_signals, opens, highs, lows, closes, type_codes, n_bars,
            atr_ratio, ema_slope, neutral_start, neutral_end,
            mdd_full_below=fb, mdd_min_above=15.0, mdd_min_scale=0.25,
        )
        stats = calc_stats(trades)
        pm = stats['pnl'] / max(stats['mdd'], 0.01)
        p1_results[fb] = {'pnl': stats['pnl'], 'mdd': stats['mdd'], 'pm': pm,
                          'wr': stats['wr'], 'trades': stats['trades'], 'scales': n_sc}
        marker = ' ← current' if fb == 3.0 else ''
        print(f"{fb:>12.1f} {stats['pnl']:>+8.1f} {stats['mdd']:>8.2f} "
              f"{pm:>8.1f} {stats['wr']:>6.1f} {stats['trades']:>7} {n_sc:>7}{marker}")

    # ==================== PHASE 2: min_size_above_dd sweep ====================
    print("\n" + "=" * 90)
    print("PHASE 2: min_size_above_dd sweep (full_below=3, min_scale=0.25 fixed)")
    print("=" * 90)

    min_above_vals = [8.0, 10.0, 12.0, 15.0, 20.0, 25.0]
    print(f"\n{'min_above':>12} {'PnL%':>8} {'MDD%':>8} {'P/M':>8} {'WR%':>6} "
          f"{'Trades':>7} {'Scales':>7}")
    print("-" * 65)

    p2_results = {}
    for ma in min_above_vals:
        trades, n_sc = portfolio_sim_mdd(
            neutral_signals, opens, highs, lows, closes, type_codes, n_bars,
            atr_ratio, ema_slope, neutral_start, neutral_end,
            mdd_full_below=3.0, mdd_min_above=ma, mdd_min_scale=0.25,
        )
        stats = calc_stats(trades)
        pm = stats['pnl'] / max(stats['mdd'], 0.01)
        p2_results[ma] = {'pnl': stats['pnl'], 'mdd': stats['mdd'], 'pm': pm,
                          'wr': stats['wr'], 'trades': stats['trades'], 'scales': n_sc}
        marker = ' ← current' if ma == 15.0 else ''
        print(f"{ma:>12.1f} {stats['pnl']:>+8.1f} {stats['mdd']:>8.2f} "
              f"{pm:>8.1f} {stats['wr']:>6.1f} {stats['trades']:>7} {n_sc:>7}{marker}")

    # ==================== PHASE 3: min_scale sweep ====================
    print("\n" + "=" * 90)
    print("PHASE 3: min_scale sweep (full_below=3, min_above=15 fixed)")
    print("=" * 90)

    min_scale_vals = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
    print(f"\n{'min_scale':>12} {'PnL%':>8} {'MDD%':>8} {'P/M':>8} {'WR%':>6} "
          f"{'Trades':>7} {'Scales':>7}")
    print("-" * 65)

    p3_results = {}
    for ms in min_scale_vals:
        trades, n_sc = portfolio_sim_mdd(
            neutral_signals, opens, highs, lows, closes, type_codes, n_bars,
            atr_ratio, ema_slope, neutral_start, neutral_end,
            mdd_full_below=3.0, mdd_min_above=15.0, mdd_min_scale=ms,
        )
        stats = calc_stats(trades)
        pm = stats['pnl'] / max(stats['mdd'], 0.01)
        p3_results[ms] = {'pnl': stats['pnl'], 'mdd': stats['mdd'], 'pm': pm,
                          'wr': stats['wr'], 'trades': stats['trades'], 'scales': n_sc}
        marker = ' ← current' if ms == 0.25 else ''
        print(f"{ms:>12.2f} {stats['pnl']:>+8.1f} {stats['mdd']:>8.2f} "
              f"{pm:>8.1f} {stats['wr']:>6.1f} {stats['trades']:>7} {n_sc:>7}{marker}")

    # ==================== PHASE 4: Grid search top combos ====================
    print("\n" + "=" * 90)
    print("PHASE 4: Grid search (top candidates from Phase 1-3)")
    print("=" * 90)

    # Pick top 3 from each phase + OFF
    # Sort each phase by PnL/MDD
    top_fb = sorted(p1_results.keys(), key=lambda k: p1_results[k]['pm'], reverse=True)[:3]
    top_ma = sorted(p2_results.keys(), key=lambda k: p2_results[k]['pm'], reverse=True)[:3]
    top_ms = sorted(p3_results.keys(), key=lambda k: p3_results[k]['pm'], reverse=True)[:3]

    print(f"\n  Top full_below: {top_fb}")
    print(f"  Top min_above:  {top_ma}")
    print(f"  Top min_scale:  {top_ms}")

    grid_results = {}
    configs = []
    for fb in top_fb:
        for ma in top_ma:
            for ms in top_ms:
                if fb >= ma:  # invalid: full_below must be < min_above
                    continue
                configs.append((fb, ma, ms))

    # Add current + OFF
    configs.append((3.0, 15.0, 0.25))  # current
    configs.append((100.0, 101.0, 1.0))  # OFF
    configs = list(set(configs))
    configs.sort()

    print(f"\n  Grid: {len(configs)} configs")
    print(f"\n{'Config':>25} {'PnL%':>8} {'MDD%':>8} {'P/M':>8} {'WR%':>6} "
          f"{'Trades':>7} {'Scales':>7}")
    print("-" * 80)

    for fb, ma, ms in configs:
        trades, n_sc = portfolio_sim_mdd(
            neutral_signals, opens, highs, lows, closes, type_codes, n_bars,
            atr_ratio, ema_slope, neutral_start, neutral_end,
            mdd_full_below=fb, mdd_min_above=ma, mdd_min_scale=ms,
        )
        stats = calc_stats(trades)
        pm = stats['pnl'] / max(stats['mdd'], 0.01)
        label = f"fb{fb:.0f}_ma{ma:.0f}_ms{ms:.2f}"
        if fb >= 100:
            label = "OFF"
        elif fb == 3.0 and ma == 15.0 and ms == 0.25:
            label = "CURRENT"
        grid_results[label] = {
            'full_below': fb, 'min_above': ma, 'min_scale': ms,
            'pnl': round(stats['pnl'], 1), 'mdd': round(stats['mdd'], 2),
            'pnl_mdd': round(pm, 1), 'wr': round(stats['wr'], 1),
            'trades': stats['trades'], 'scales': n_sc,
        }
        marker = ''
        if fb == 3.0 and ma == 15.0 and ms == 0.25:
            marker = ' ← current'
        elif fb >= 100:
            marker = ' ← OFF'
        print(f"{label:>25} {stats['pnl']:>+8.1f} {stats['mdd']:>8.2f} "
              f"{pm:>8.1f} {stats['wr']:>6.1f} {stats['trades']:>7} {n_sc:>7}{marker}")

    # ==================== PHASE 5: WF validation ====================
    print("\n" + "=" * 90)
    print("PHASE 5: WF 3-fold validation (top 5 + current + OFF)")
    print("=" * 90)

    # Rank by PnL/MDD, take top 5
    ranked = sorted(grid_results.items(), key=lambda x: x[1]['pnl_mdd'], reverse=True)
    wf_candidates = []
    seen = set()
    for label, r in ranked:
        key = (r['full_below'], r['min_above'], r['min_scale'])
        if key not in seen:
            seen.add(key)
            wf_candidates.append((label, r))
        if len(wf_candidates) >= 5:
            break
    # Ensure current and OFF are included
    for label, r in grid_results.items():
        key = (r['full_below'], r['min_above'], r['min_scale'])
        if key not in seen and label in ('CURRENT', 'OFF'):
            seen.add(key)
            wf_candidates.append((label, r))

    wf_results = {}
    print(f"\n{'Config':>25} {'F1':>8} {'F2':>8} {'F3':>8} {'Avg':>8} {'Min':>8} {'WF':>6}")
    print("-" * 72)

    for label, r in wf_candidates:
        folds = expanding_window_wf(
            neutral_signals, opens, highs, lows, closes, type_codes, n_bars,
            atr_ratio, ema_slope, neutral_start, neutral_end,
            mdd_full_below=r['full_below'], mdd_min_above=r['min_above'],
            mdd_min_scale=r['min_scale'],
        )

        if len(folds) == 3:
            avg = np.mean(folds)
            mn = min(folds)
            n_pass = sum(1 for f in folds if f > 0)
            wf_str = f"{n_pass}/3 {'P' if n_pass == 3 else 'F'}"
        else:
            avg = mn = 0
            n_pass = 0
            wf_str = "N/A"

        wf_results[label] = {
            'full_below': r['full_below'], 'min_above': r['min_above'],
            'min_scale': r['min_scale'],
            'folds': [round(f, 1) for f in folds],
            'avg': round(avg, 1), 'min': round(mn, 1), 'n_pass': n_pass,
            'is_pnl_mdd': r['pnl_mdd'],
        }

        marker = ' ← current' if label == 'CURRENT' else (' ← OFF' if label == 'OFF' else '')
        fold_strs = [f"{f:>+8.1f}" for f in folds]
        print(f"{label:>25} {''.join(fold_strs)} {avg:>+8.1f} {mn:>+8.1f} {wf_str:>6}{marker}")

    # ==================== SYNTHESIS ====================
    print("\n" + "=" * 90)
    print("SYNTHESIS")
    print("=" * 90)

    current_is = grid_results.get('CURRENT', {})
    current_wf = wf_results.get('CURRENT', {})
    print(f"\nCurrent (3/15/0.25):")
    print(f"  IS PnL/MDD: {current_is.get('pnl_mdd', 'N/A')}")
    print(f"  IS Scale events: {current_is.get('scales', 'N/A')}")
    print(f"  OOS avg: {current_wf.get('avg', 'N/A')}, min: {current_wf.get('min', 'N/A')}")

    off_is = grid_results.get('OFF', {})
    off_wf = wf_results.get('OFF', {})
    print(f"\nOFF:")
    print(f"  IS PnL/MDD: {off_is.get('pnl_mdd', 'N/A')}")
    print(f"  OOS avg: {off_wf.get('avg', 'N/A')}, min: {off_wf.get('min', 'N/A')}")

    # Find best WF-passing
    best_label = None
    best_min = -999
    for label, wr in wf_results.items():
        if wr.get('n_pass', 0) == 3 and wr['min'] > best_min:
            best_min = wr['min']
            best_label = label

    if best_label:
        best_wf = wf_results[best_label]
        best_is = grid_results.get(best_label, {})
        print(f"\nBest WF-passing by min fold: {best_label}")
        print(f"  Config: fb={best_wf['full_below']}, ma={best_wf['min_above']}, ms={best_wf['min_scale']}")
        print(f"  IS PnL/MDD: {best_is.get('pnl_mdd', 'N/A')}")
        print(f"  OOS avg: {best_wf['avg']}, min: {best_wf['min']}")

        if best_label == 'CURRENT':
            print("\n>>> RECOMMEND: KEEP current MDD sizing (3/15/0.25)")
        elif best_label == 'OFF':
            print("\n>>> RECOMMEND: DISABLE MDD sizing (not contributing)")
        else:
            delta_min = best_wf['min'] - current_wf.get('min', 0)
            delta_is = best_is.get('pnl_mdd', 0) - current_is.get('pnl_mdd', 0)
            print(f"\n  Min fold delta: {delta_min:+.1f}%")
            print(f"  IS PnL/MDD delta: {delta_is:+.1f}")
            if delta_min > 5 or delta_is > 20:
                print(f"\n>>> RECOMMEND: Switch to {best_label}")
            else:
                print(f"\n>>> RECOMMEND: Marginal improvement, consider keeping current")

    # Save
    output = {
        'timestamp': datetime.now().isoformat(),
        'version': 'v1.46.0',
        'baseline': {
            'pnl': round(base_stats['pnl'], 1),
            'mdd': round(base_stats['mdd'], 2),
            'pnl_mdd': round(base_stats['pnl'] / max(base_stats['mdd'], 0.01), 1),
            'scale_events': base_scales,
        },
        'off': {
            'pnl': round(off_stats['pnl'], 1),
            'mdd': round(off_stats['mdd'], 2),
            'pnl_mdd': round(off_stats['pnl'] / max(off_stats['mdd'], 0.01), 1),
        },
        'phase1_full_below': {str(k): v for k, v in p1_results.items()},
        'phase2_min_above': {str(k): v for k, v in p2_results.items()},
        'phase3_min_scale': {str(k): v for k, v in p3_results.items()},
        'grid_results': grid_results,
        'wf_results': wf_results,
        'recommended': best_label,
    }
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved: {OUTPUT_FILE}")


if __name__ == '__main__':
    main()
