#!/usr/bin/env python3
"""
SHORT Pattern Restoration + Compact TP/SL Study
================================================

Problem 1 - Trade Frequency:
  22 patterns produce ~1.0 trade/day OOS (N=9 slots heavily underused).
  34 SHORT patterns dropped from 59-pat set due to STRONG score < 3.
  Same situation as LONG restoration: MAE/MFE TP/SL was suboptimal,
  Phase 4 WR Excess grid search can elevate to STRONG.

Problem 2 - TP/SL Too Wide:
  Current TP 1.0-3.5%, SL 1.0-4.0% → trades take very long to resolve.
  Many trades immediately underwater after entry.
  Tighter (compact) TP/SL → faster resolution, more slot turnover.

Six Phases:
  1. Re-evaluate 34 dropped SHORTs with 4-phase STRONG classification
  2. Compact TP/SL re-evaluation for ALL patterns (standard vs compact grid)
  3. Holding time analysis (bars-to-exit for wide vs compact TP/SL)
  4. Portfolio comparison: wide vs compact vs mixed
  5. WF 3-fold validation of best combination
  6. Final recommendation

Standard Research Protocol:
  - Entry: next bar open after signal
  - Exit: intrabar high/low (distance-based, same-bar from bar OPEN)
  - Sizing: compound (1/N per slot)
  - Fee: FEE_PCT * LEVERAGE = 0.30% capital-space
  - Timeout: 864 bars (72h) -> DROP
  - WF: 3-fold expanding window within 270d overlap of 720d data
  - classify_candle: production import (NEVER self-implement)
"""

import os
import sys
import json
import time
import logging

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
sys.path.insert(0, _PROJECT_ROOT)

from scripts.analysis.multi_position_diversification_study import (
    load_and_classify, load_patterns, build_signal_index,
    compute_atr_ratio, calc_stats, find_overlap_bar,
    FEE_PCT, LEVERAGE, FEE, MAX_BARS, BARS_PER_DAY,
    ATR_PERIOD, ATR_WINDOW, CLAMP_LO, CLAMP_HI,
    SLIPPAGE_BUFFER,
)
from scripts.analysis.tpsl_regime_bias_study import (
    backtest_pattern_trades, compute_wr_and_edge,
    build_custom_schedule, simulate_hedge_simple,
    run_wf_3fold, load_patterns_from_file,
)
from scripts.analysis.fifo_vs_hedge_backtest_study import (
    ActivePosition, pnl_mdd_ratio,
)
from scripts.analysis.long_restore_and_direction_cap_study import (
    simulate_hedge_with_cap, run_wf_3fold_with_cap,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger('short_restore_compact')

# ---------------------------------------------------------------------------
# File paths
# ---------------------------------------------------------------------------
DATA_720D = os.path.join(_PROJECT_ROOT, 'data', 'btc_5m_720days_binance.csv')
PATTERNS_FILE = os.path.join(_PROJECT_ROOT, 'results', 'dynamic_patterns.json')
PATTERNS_59_BACKUP = os.path.join(_PROJECT_ROOT, 'results', 'dynamic_patterns_59pat_backup.json')
OUTPUT_FILE = os.path.join(_PROJECT_ROOT, 'results', 'short_restore_tpsl_compact_study.json')

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TIMEOUT_T864 = 864
MAX_POSITIONS = 9
DIRECTION_CAP = 6

# Standard grid (same as LONG study)
TP_GRID_WIDE = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 3.5]
SL_GRID_WIDE = [1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 3.5, 4.0]

# Compact grid (tighter TP/SL for faster resolution)
TP_GRID_COMPACT = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]
SL_GRID_COMPACT = [1.0, 1.25, 1.5, 1.75, 2.0, 2.5]

# Symmetric test levels
SYMMETRIC_LEVELS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]


# ===================================================================
# PHASE 1: Re-evaluate 34 dropped SHORTs with STRONG classification
# ===================================================================

def phase1_reevaluate_shorts(dropped_patterns, sig_index, opens, highs, lows,
                             n_bars, atr_ratio, overlap_bar):
    """
    Same 4-phase STRONG classification as LONG restoration study,
    but for SHORT patterns. Uses standard wide grid.
    """
    print("\n" + "=" * 110)
    print(f"  PHASE 1: RE-EVALUATE {len(dropped_patterns)} DROPPED SHORT PATTERNS")
    print(f"  Wide Grid: TP {TP_GRID_WIDE} x SL {SL_GRID_WIDE}")
    print(f"  Symmetric: {SYMMETRIC_LEVELS}")
    print(f"  Timeout: {TIMEOUT_T864} bars, ATR-scaled")
    print("=" * 110)

    t0 = time.time()
    results = []

    for p in dropped_patterns:
        pattern = p['pattern']
        bars = sig_index.get(pattern)
        if not bars:
            continue

        direction_int = -1  # All SHORT
        orig_tp, orig_sl = p['tp'], p['sl']

        # --- P1: Original MAE/MFE TP/SL WR Excess ---
        trades_orig = backtest_pattern_trades(
            bars, direction_int, orig_tp, orig_sl,
            opens, highs, lows, n_bars,
            atr_ratio=atr_ratio, use_atr=True,
            lo_bar=overlap_bar, timeout_bars=TIMEOUT_T864)
        orig_wr, orig_edge, orig_n = compute_wr_and_edge(trades_orig)
        random_wr_orig = orig_sl / (orig_tp + orig_sl) * 100
        p1_excess = orig_wr - random_wr_orig
        p1_pass = p1_excess > 5.0

        # --- P4: Grid search for optimal WR Excess ---
        best_excess = -999
        best_tp, best_sl = orig_tp, orig_sl
        best_wr, best_edge, best_n = 0, 0, 0

        for tp_test in TP_GRID_WIDE:
            for sl_test in SL_GRID_WIDE:
                trades = backtest_pattern_trades(
                    bars, direction_int, tp_test, sl_test,
                    opens, highs, lows, n_bars,
                    atr_ratio=atr_ratio, use_atr=True,
                    lo_bar=overlap_bar, timeout_bars=TIMEOUT_T864)
                wr, edge, n = compute_wr_and_edge(trades)
                if n < 10:
                    continue
                random_wr = sl_test / (tp_test + sl_test) * 100
                excess = wr - random_wr
                if excess > best_excess:
                    best_excess = excess
                    best_tp, best_sl = tp_test, sl_test
                    best_wr, best_edge, best_n = wr, edge, n

        p4_pass = best_excess > 5.0

        # --- P2: Direction flip at optimal TP/SL ---
        flip_trades = backtest_pattern_trades(
            bars, 1, best_tp, best_sl,  # flip to LONG
            opens, highs, lows, n_bars,
            atr_ratio=atr_ratio, use_atr=True,
            lo_bar=overlap_bar, timeout_bars=TIMEOUT_T864)
        flip_wr, _, _ = compute_wr_and_edge(flip_trades)
        random_wr_opt = best_sl / (best_tp + best_sl) * 100
        p2_genuine = flip_wr < random_wr_opt

        # --- P3: Symmetric TP=SL test ---
        sym_above = 0
        sym_total = 0
        sym_details = []
        for level in SYMMETRIC_LEVELS:
            trades = backtest_pattern_trades(
                bars, direction_int, level, level,
                opens, highs, lows, n_bars,
                atr_ratio=atr_ratio, use_atr=True,
                lo_bar=overlap_bar, timeout_bars=TIMEOUT_T864)
            wr, _, n = compute_wr_and_edge(trades)
            if n > 0:
                sym_total += 1
                above = wr > 50.0
                if above:
                    sym_above += 1
                sym_details.append({'level': level, 'wr': round(wr, 1), 'n': n, 'above_50': above})

        p3_all_above = sym_above == sym_total and sym_total > 0

        # --- Score ---
        score = sum([p1_pass, p2_genuine, p3_all_above, p4_pass])
        if score >= 3:
            overall = 'STRONG'
        elif score >= 2:
            overall = 'MODERATE'
        elif score >= 1:
            overall = 'WEAK'
        else:
            overall = 'NONE'

        results.append({
            'pattern': pattern, 'direction': 'SHORT',
            'orig_tp': orig_tp, 'orig_sl': orig_sl,
            'orig_wr': round(orig_wr, 1), 'orig_n': orig_n,
            'p1_excess': round(p1_excess, 1), 'p1_pass': p1_pass,
            'optimal_tp': best_tp, 'optimal_sl': best_sl,
            'optimal_wr': round(best_wr, 1), 'optimal_excess': round(best_excess, 1),
            'optimal_edge': round(best_edge, 3), 'optimal_n': best_n,
            'p4_pass': p4_pass,
            'flip_wr': round(flip_wr, 1), 'random_wr_opt': round(random_wr_opt, 1),
            'p2_genuine': p2_genuine,
            'sym_above': sym_above, 'sym_total': sym_total,
            'sym_details': sym_details, 'p3_all_above': p3_all_above,
            'score': score, 'overall': overall,
        })

    elapsed = time.time() - t0
    results.sort(key=lambda x: -x['score'])

    # Print results
    print(f"\n  {'#':>2} {'Pattern':<12} | {'OrigTP':>6} {'OrigSL':>6} {'P1Exc':>6} | "
          f"{'OptTP':>5} {'OptSL':>5} {'OptExc':>6} | {'FlipWR':>6} {'Sym':>5} | "
          f"{'P1':>3} {'P2':>3} {'P3':>3} {'P4':>3} {'Score':>5} {'Class':>8}")
    print("  " + "-" * 105)

    for i, r in enumerate(results, 1):
        p1 = 'Y' if r['p1_pass'] else 'N'
        p2 = 'Y' if r['p2_genuine'] else 'N'
        p3 = 'Y' if r['p3_all_above'] else 'N'
        p4 = 'Y' if r['p4_pass'] else 'N'
        print(f"  {i:2d} {r['pattern']:<12} | {r['orig_tp']:6.2f} {r['orig_sl']:6.2f} "
              f"{r['p1_excess']:>+5.1f}% | "
              f"{r['optimal_tp']:5.2f} {r['optimal_sl']:5.2f} {r['optimal_excess']:>+5.1f}% | "
              f"{r['flip_wr']:5.1f}% {r['sym_above']:2d}/{r['sym_total']:1d} | "
              f"  {p1}   {p2}   {p3}   {p4}   {r['score']:3d}   {r['overall']:>8}")

    strong = sum(1 for r in results if r['overall'] == 'STRONG')
    moderate = sum(1 for r in results if r['overall'] == 'MODERATE')
    print(f"\n  STRONG: {strong}/{len(dropped_patterns)}, MODERATE: {moderate}/{len(dropped_patterns)}  ({elapsed:.1f}s)")

    return results


# ===================================================================
# PHASE 2: Compact TP/SL re-evaluation for ALL patterns
# ===================================================================

def phase2_compact_tpsl(all_patterns, sig_index, opens, highs, lows,
                        n_bars, atr_ratio, overlap_bar):
    """
    For each pattern in the full set, find optimal TP/SL under BOTH grids:
      - Wide: current grid (TP max 3.5, SL max 4.0)
      - Compact: tighter grid (TP max 2.0, SL max 2.5)
    Compare WR Excess, edge, holding time.
    """
    print("\n" + "=" * 110)
    print(f"  PHASE 2: COMPACT vs WIDE TP/SL RE-EVALUATION ({len(all_patterns)} patterns)")
    print(f"  Wide:    TP {TP_GRID_WIDE} x SL {SL_GRID_WIDE}")
    print(f"  Compact: TP {TP_GRID_COMPACT} x SL {SL_GRID_COMPACT}")
    print("=" * 110)

    t0 = time.time()
    results = []

    for pat in all_patterns:
        pattern = pat['pattern']
        direction = pat['direction']
        direction_int = 1 if direction == 'LONG' else -1
        bars = sig_index.get(pattern)
        if not bars:
            continue

        row = {'pattern': pattern, 'direction': direction}

        # Run both grids
        for grid_label, tp_grid, sl_grid in [
            ('wide', TP_GRID_WIDE, SL_GRID_WIDE),
            ('compact', TP_GRID_COMPACT, SL_GRID_COMPACT),
        ]:
            best_excess = -999
            best_tp, best_sl = 1.0, 1.0
            best_wr, best_edge, best_n = 0, 0, 0

            for tp_test in tp_grid:
                for sl_test in sl_grid:
                    trades = backtest_pattern_trades(
                        bars, direction_int, tp_test, sl_test,
                        opens, highs, lows, n_bars,
                        atr_ratio=atr_ratio, use_atr=True,
                        lo_bar=overlap_bar, timeout_bars=TIMEOUT_T864)
                    wr, edge, n = compute_wr_and_edge(trades)
                    if n < 10:
                        continue
                    random_wr = sl_test / (tp_test + sl_test) * 100
                    excess = wr - random_wr
                    if excess > best_excess:
                        best_excess = excess
                        best_tp, best_sl = tp_test, sl_test
                        best_wr, best_edge, best_n = wr, edge, n

            strong = best_excess > 5.0
            row[f'{grid_label}_tp'] = best_tp
            row[f'{grid_label}_sl'] = best_sl
            row[f'{grid_label}_wr'] = round(best_wr, 1)
            row[f'{grid_label}_excess'] = round(best_excess, 1)
            row[f'{grid_label}_edge'] = round(best_edge, 3)
            row[f'{grid_label}_n'] = best_n
            row[f'{grid_label}_strong'] = strong

            # Compute median holding time with best TP/SL
            trades_best = backtest_pattern_trades(
                bars, direction_int, best_tp, best_sl,
                opens, highs, lows, n_bars,
                atr_ratio=atr_ratio, use_atr=True,
                lo_bar=overlap_bar, timeout_bars=TIMEOUT_T864)
            if trades_best:
                hold_bars = [t['exit_bar'] - t['entry_bar'] for t in trades_best if t.get('reason') != 'timeout']
                row[f'{grid_label}_hold_median'] = int(np.median(hold_bars)) if hold_bars else 0
                row[f'{grid_label}_hold_mean'] = round(np.mean(hold_bars), 0) if hold_bars else 0
            else:
                row[f'{grid_label}_hold_median'] = 0
                row[f'{grid_label}_hold_mean'] = 0

        results.append(row)

    elapsed = time.time() - t0

    # Print comparison
    print(f"\n  {'Pattern':<12} {'Dir':>5} | {'W_TP':>5} {'W_SL':>5} {'W_Exc':>6} {'W_Hold':>6} | "
          f"{'C_TP':>5} {'C_SL':>5} {'C_Exc':>6} {'C_Hold':>6} | {'Faster':>7}")
    print("  " + "-" * 95)

    for r in results:
        w_hold = r.get('wide_hold_median', 0)
        c_hold = r.get('compact_hold_median', 0)
        faster = f"{w_hold - c_hold:+d}" if (w_hold and c_hold) else "?"
        w_s = '*' if r.get('wide_strong') else ' '
        c_s = '*' if r.get('compact_strong') else ' '
        print(f"  {r['pattern']:<12} {r['direction']:>5} | "
              f"{r.get('wide_tp',0):5.2f} {r.get('wide_sl',0):5.2f} "
              f"{r.get('wide_excess',0):>+5.1f}{w_s} {w_hold:5d}b | "
              f"{r.get('compact_tp',0):5.2f} {r.get('compact_sl',0):5.2f} "
              f"{r.get('compact_excess',0):>+5.1f}{c_s} {c_hold:5d}b | {faster:>7}")

    # Summary stats
    wide_strong = sum(1 for r in results if r.get('wide_strong'))
    comp_strong = sum(1 for r in results if r.get('compact_strong'))
    wide_holds = [r['wide_hold_median'] for r in results if r.get('wide_hold_median', 0) > 0]
    comp_holds = [r['compact_hold_median'] for r in results if r.get('compact_hold_median', 0) > 0]

    print(f"\n  STRONG patterns: Wide={wide_strong}, Compact={comp_strong}")
    if wide_holds:
        print(f"  Median holding time: Wide={np.median(wide_holds):.0f}b ({np.median(wide_holds)*5/60:.1f}h), "
              f"Compact={np.median(comp_holds):.0f}b ({np.median(comp_holds)*5/60:.1f}h)")
    print(f"  ({elapsed:.1f}s)")

    return results


# ===================================================================
# PHASE 3: Holding Time Analysis
# ===================================================================

def phase3_holding_time(results_phase2):
    """
    Summarize holding time differences between wide and compact TP/SL.
    """
    print("\n" + "=" * 110)
    print("  PHASE 3: HOLDING TIME ANALYSIS")
    print("=" * 110)

    wide_holds = []
    comp_holds = []
    both_strong = []

    for r in results_phase2:
        w_h = r.get('wide_hold_median', 0)
        c_h = r.get('compact_hold_median', 0)
        if w_h > 0:
            wide_holds.append(w_h)
        if c_h > 0:
            comp_holds.append(c_h)
        if r.get('wide_strong') and r.get('compact_strong'):
            both_strong.append({
                'pattern': r['pattern'],
                'wide_hold': w_h,
                'compact_hold': c_h,
                'wide_excess': r.get('wide_excess', 0),
                'compact_excess': r.get('compact_excess', 0),
                'speedup': w_h / c_h if c_h > 0 else 0,
            })

    print(f"\n  Wide TP/SL holding time (median bars): "
          f"p25={np.percentile(wide_holds, 25):.0f}, "
          f"median={np.median(wide_holds):.0f}, "
          f"p75={np.percentile(wide_holds, 75):.0f}, "
          f"mean={np.mean(wide_holds):.0f}")
    print(f"  Compact TP/SL holding time (median bars): "
          f"p25={np.percentile(comp_holds, 25):.0f}, "
          f"median={np.median(comp_holds):.0f}, "
          f"p75={np.percentile(comp_holds, 75):.0f}, "
          f"mean={np.mean(comp_holds):.0f}")

    if both_strong:
        speedups = [b['speedup'] for b in both_strong]
        print(f"\n  Patterns STRONG in both grids: {len(both_strong)}")
        print(f"  Speed-up (wide/compact): median={np.median(speedups):.1f}x, mean={np.mean(speedups):.1f}x")

        # WR Excess comparison for patterns strong in both
        w_exc = [b['wide_excess'] for b in both_strong]
        c_exc = [b['compact_excess'] for b in both_strong]
        print(f"  WR Excess: Wide avg={np.mean(w_exc):.1f}pp, Compact avg={np.mean(c_exc):.1f}pp, "
              f"diff={np.mean(w_exc)-np.mean(c_exc):+.1f}pp")

    summary = {
        'wide_hold_median': round(float(np.median(wide_holds)), 1),
        'wide_hold_hours': round(float(np.median(wide_holds)) * 5 / 60, 1),
        'compact_hold_median': round(float(np.median(comp_holds)), 1),
        'compact_hold_hours': round(float(np.median(comp_holds)) * 5 / 60, 1),
        'both_strong_count': len(both_strong),
        'median_speedup': round(float(np.median(speedups)), 1) if both_strong else 0,
    }
    return summary


# ===================================================================
# PHASE 4: Portfolio Comparison (Wide vs Compact vs Mixed)
# ===================================================================

def phase4_portfolio_comparison(pattern_configs, sig_index, opens, highs, lows,
                                n_bars, atr_ratio, overlap_bar):
    """
    Compare portfolio performance under different TP/SL regimes.
    pattern_configs: dict of {label: pattern_list}
      Each pattern has: pattern, direction, tp, sl
    """
    print("\n" + "=" * 110)
    print("  PHASE 4: PORTFOLIO COMPARISON")
    print(f"  Hedge N={MAX_POSITIONS}, cap={DIRECTION_CAP}, T864, ATR-scaled")
    print("=" * 110)

    all_results = {}

    for label, pats in pattern_configs.items():
        sched = build_custom_schedule(pats, sig_index, atr_ratio, opens, n_bars)
        sim = simulate_hedge_with_cap(
            MAX_POSITIONS, DIRECTION_CAP, sched, opens, highs, lows,
            overlap_bar, len(opens), timeout_bars=TIMEOUT_T864)
        port = [(r['entry_bar'], r['exit_bar'], r['scaled_pnl']) for r in sim]
        stats = calc_stats(port)

        # Count directions
        long_t = sum(1 for r in sim if r.get('direction') == 1)
        short_t = sum(1 for r in sim if r.get('direction') == -1)

        # Holding time
        hold_bars = [r['exit_bar'] - r['entry_bar'] for r in sim]
        hold_med = int(np.median(hold_bars)) if hold_bars else 0

        n_long = sum(1 for p in pats if p['direction'] == 'LONG')
        n_short = sum(1 for p in pats if p['direction'] == 'SHORT')

        row = {
            'patterns': len(pats),
            'n_long': n_long,
            'n_short': n_short,
            'trades': stats['trades'],
            'wr': stats['wr'],
            'pf': stats.get('pf', 0),
            'add_pnl': stats.get('add_pnl', 0),
            'add_edge': stats.get('add_edge', 0),
            'add_mdd': stats.get('add_mdd', 0),
            'cmp_pnl': stats.get('cmp_pnl', 0),
            'cmp_mdd': stats.get('cmp_mdd', 0),
            'pnl_mdd_ratio': round(stats.get('cmp_pnl', 0) / max(stats.get('cmp_mdd', 0.1), 0.1), 2),
            'long_trades': long_t,
            'short_trades': short_t,
            'hold_median_bars': hold_med,
            'hold_median_hours': round(hold_med * 5 / 60, 1),
            'trades_per_day': round(stats['trades'] / (270 if stats['trades'] > 0 else 1), 2),
        }
        all_results[label] = row

        print(f"\n  [{label}] {len(pats)}pat ({n_long}L+{n_short}S)")
        print(f"    Trades: {row['trades']} ({row['trades_per_day']}/day), "
              f"L/S={long_t}/{short_t}")
        print(f"    WR: {row['wr']:.1f}%, PF: {row['pf']:.2f}")
        print(f"    PnL: +{row['cmp_pnl']:.1f}%, MDD: {row['cmp_mdd']:.1f}%, "
              f"PnL/MDD: {row['pnl_mdd_ratio']:.2f}")
        print(f"    Hold median: {hold_med}b ({row['hold_median_hours']:.1f}h)")

    return all_results


# ===================================================================
# PHASE 5: WF 3-fold Validation
# ===================================================================

def phase5_wf_validation(best_configs, sig_index, opens, highs, lows,
                         n_bars, atr_ratio, overlap_bar):
    """Run WF 3-fold for the top configurations."""
    print("\n" + "=" * 110)
    print("  PHASE 5: WF 3-FOLD VALIDATION")
    print("=" * 110)

    wf_results = {}
    for label, pats in best_configs.items():
        sched = build_custom_schedule(pats, sig_index, atr_ratio, opens, n_bars)
        folds, verdict = run_wf_3fold_with_cap(
            MAX_POSITIONS, DIRECTION_CAP, sched, opens, highs, lows,
            overlap_bar, n_bars, timeout_bars=TIMEOUT_T864)
        wf_results[label] = {
            'folds': [{'fold': f['fold'], 'oos_trades': f['oos_trades'],
                        'oos_wr': f['oos_wr'], 'oos_pnl': f['oos_pnl'],
                        'oos_mdd': f['oos_mdd'], 'pass': str(f['pass'])}
                       for f in folds],
            'verdict': verdict,
        }
        print(f"\n  [{label}] {verdict}")
        for f in folds:
            print(f"    Fold {f['fold']}: trades={f['oos_trades']}, WR={f['oos_wr']:.1f}%, "
                  f"PnL=+{f['oos_pnl']:.1f}%, MDD={f['oos_mdd']:.1f}%")

    return wf_results


# ===================================================================
# MAIN
# ===================================================================

def main():
    t_start = time.time()

    # --- Load data ---
    print("Loading 720-day data...")
    df = load_and_classify(DATA_720D)
    opens = df['open'].values.astype(np.float64)
    highs = df['high'].values.astype(np.float64)
    lows = df['low'].values.astype(np.float64)
    closes = df['close'].values.astype(np.float64)
    types = df['rctype'].values
    n_bars = len(df)
    print(f"  {n_bars} bars loaded")

    # Build signal index
    sig_index = build_signal_index(types, n_bars)
    atr_ratio = compute_atr_ratio(highs, lows, closes, ATR_PERIOD, ATR_WINDOW)
    overlap_bar = find_overlap_bar(df)
    print(f"  Overlap bar: {overlap_bar}, Post-overlap: {n_bars - overlap_bar} bars")

    # --- Load current 22 patterns ---
    with open(PATTERNS_FILE) as f:
        p22_data = json.load(f)
    current_patterns = []
    for key, det in p22_data.get('pattern_details', {}).items():
        tpsl = p22_data.get('patterns_tpsl', {}).get(det['pattern'], [1.0, 1.0])
        current_patterns.append({
            'pattern': det['pattern'],
            'direction': det['direction'],
            'tp': tpsl[0], 'sl': tpsl[1],
        })

    # --- Load 59-pat backup to find dropped SHORTs ---
    with open(PATTERNS_59_BACKUP) as f:
        p59_data = json.load(f)

    current_pat_names = {p['pattern'] for p in current_patterns}
    dropped_shorts = []
    for key, det in p59_data.get('pattern_details', {}).items():
        if det['direction'] == 'SHORT' and det['pattern'] not in current_pat_names:
            dropped_shorts.append({
                'pattern': det['pattern'],
                'direction': 'SHORT',
                'tp': det['tp'], 'sl': det['sl'],
            })

    # Also find the 13 existing SHORTs in current set (for reference)
    current_shorts = {p['pattern'] for p in current_patterns if p['direction'] == 'SHORT'}
    print(f"  Current: {len(current_patterns)} patterns ({len(current_patterns)-len(current_shorts)}L + {len(current_shorts)}S)")
    print(f"  Dropped SHORTs to evaluate: {len(dropped_shorts)}")

    # ================================================================
    # PHASE 1: Re-evaluate dropped SHORTs
    # ================================================================
    phase1_results = phase1_reevaluate_shorts(
        dropped_shorts, sig_index, opens, highs, lows,
        n_bars, atr_ratio, overlap_bar)

    strong_shorts = [r for r in phase1_results if r['overall'] == 'STRONG']
    print(f"\n  >> {len(strong_shorts)} STRONG SHORTs restored")

    # Build restored SHORT pattern list
    restored_short_pats = [{
        'pattern': r['pattern'],
        'direction': 'SHORT',
        'tp': r['optimal_tp'], 'sl': r['optimal_sl'],
    } for r in strong_shorts]

    # Build expanded ALL patterns = current 22 + restored SHORTs (wide TP/SL)
    expanded_wide = list(current_patterns) + restored_short_pats
    print(f"  Expanded wide: {len(expanded_wide)} patterns "
          f"({sum(1 for p in expanded_wide if p['direction']=='LONG')}L + "
          f"{sum(1 for p in expanded_wide if p['direction']=='SHORT')}S)")

    # ================================================================
    # PHASE 2: Compact TP/SL re-evaluation
    # ================================================================
    phase2_results = phase2_compact_tpsl(
        expanded_wide, sig_index, opens, highs, lows,
        n_bars, atr_ratio, overlap_bar)

    # Build compact pattern set (only patterns STRONG under compact grid)
    compact_pats = []
    for r in phase2_results:
        if r.get('compact_strong'):
            compact_pats.append({
                'pattern': r['pattern'],
                'direction': r['direction'],
                'tp': r['compact_tp'], 'sl': r['compact_sl'],
            })

    # Build mixed set: use compact TP/SL when available, else wide
    mixed_pats = []
    for r in phase2_results:
        if r.get('compact_strong'):
            mixed_pats.append({
                'pattern': r['pattern'],
                'direction': r['direction'],
                'tp': r['compact_tp'], 'sl': r['compact_sl'],
            })
        elif r.get('wide_strong'):
            mixed_pats.append({
                'pattern': r['pattern'],
                'direction': r['direction'],
                'tp': r['wide_tp'], 'sl': r['wide_sl'],
            })

    print(f"\n  Expanded wide:    {len(expanded_wide)} patterns (all use wide TP/SL)")
    print(f"  Compact only:     {len(compact_pats)} patterns (STRONG under compact grid)")
    print(f"  Mixed (best-fit): {len(mixed_pats)} patterns (compact preferred, wide fallback)")

    # ================================================================
    # PHASE 3: Holding time analysis
    # ================================================================
    hold_summary = phase3_holding_time(phase2_results)

    # ================================================================
    # PHASE 4: Portfolio comparison
    # ================================================================
    configs = {
        'baseline_22_wide': current_patterns,
        'expanded_wide': expanded_wide,
        'compact_only': compact_pats,
        'mixed': mixed_pats,
    }
    phase4_results = phase4_portfolio_comparison(
        configs, sig_index, opens, highs, lows,
        n_bars, atr_ratio, overlap_bar)

    # ================================================================
    # PHASE 5: WF for top configurations
    # ================================================================
    # Pick top 2 by PnL/MDD for WF
    ranked = sorted(phase4_results.items(), key=lambda x: -x[1]['pnl_mdd_ratio'])
    top_configs = {}
    for label, _ in ranked[:3]:
        top_configs[label] = configs[label]

    phase5_results = phase5_wf_validation(
        top_configs, sig_index, opens, highs, lows,
        n_bars, atr_ratio, overlap_bar)

    # ================================================================
    # PHASE 6: Final Recommendation
    # ================================================================
    print("\n" + "=" * 110)
    print("  PHASE 6: FINAL RECOMMENDATION")
    print("=" * 110)

    # Find best WF PASS config
    best_label = None
    best_ratio = 0
    for label, wf in phase5_results.items():
        if 'PASS' in wf['verdict']:
            ratio = phase4_results[label]['pnl_mdd_ratio']
            if ratio > best_ratio:
                best_ratio = ratio
                best_label = label

    if best_label:
        best = phase4_results[best_label]
        print(f"\n  BEST: {best_label}")
        print(f"    Patterns: {best['patterns']} ({best['n_long']}L + {best['n_short']}S)")
        print(f"    Trades: {best['trades']} ({best['trades_per_day']}/day)")
        print(f"    WR: {best['wr']:.1f}%, PnL: +{best['cmp_pnl']:.1f}%, MDD: {best['cmp_mdd']:.1f}%")
        print(f"    PnL/MDD: {best_ratio:.2f}")
        print(f"    Holding: {best['hold_median_bars']}b ({best['hold_median_hours']:.1f}h)")
        print(f"    WF: {phase5_results[best_label]['verdict']}")

        vs_baseline = phase4_results.get('baseline_22_wide', {})
        if vs_baseline:
            print(f"\n  vs Baseline (22pat wide):")
            print(f"    Trades: {vs_baseline['trades']} -> {best['trades']} "
                  f"({best['trades']/max(vs_baseline['trades'],1)*100-100:+.0f}%)")
            print(f"    Hold: {vs_baseline['hold_median_bars']}b -> {best['hold_median_bars']}b "
                  f"({best['hold_median_bars']-vs_baseline['hold_median_bars']:+d}b)")
            print(f"    PnL/MDD: {vs_baseline['pnl_mdd_ratio']:.2f} -> {best_ratio:.2f}")
    else:
        print("\n  No WF PASS configuration found!")
        best_label = ranked[0][0]

    # --- Save results ---
    elapsed_total = time.time() - t_start

    output = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'study': 'SHORT Pattern Restoration + Compact TP/SL',
        'data_bars': n_bars,
        'overlap_bar': overlap_bar,
        'dropped_shorts_evaluated': len(dropped_shorts),
        'parameters': {
            'max_positions': MAX_POSITIONS,
            'direction_cap': DIRECTION_CAP,
            'timeout': TIMEOUT_T864,
            'tp_grid_wide': TP_GRID_WIDE,
            'sl_grid_wide': SL_GRID_WIDE,
            'tp_grid_compact': TP_GRID_COMPACT,
            'sl_grid_compact': SL_GRID_COMPACT,
            'leverage': LEVERAGE,
            'fee': FEE,
        },
        'phase1_short_reevaluation': phase1_results,
        'phase2_compact_comparison': [{k: v for k, v in r.items() if k != 'sym_details'}
                                       for r in phase2_results],
        'phase3_holding_time': hold_summary,
        'phase4_portfolio': phase4_results,
        'phase5_wf': phase5_results,
        'recommendation': {
            'best_config': best_label,
            'strong_shorts_restored': len(strong_shorts),
            'strong_short_patterns': [
                {'pattern': r['pattern'], 'tp': r['optimal_tp'], 'sl': r['optimal_sl'],
                 'excess': r['optimal_excess'], 'score': r['score']}
                for r in strong_shorts
            ],
        },
        'elapsed_seconds': round(elapsed_total, 1),
    }

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved to {OUTPUT_FILE}")
    print(f"  Total elapsed: {elapsed_total:.1f}s")


if __name__ == '__main__':
    main()
