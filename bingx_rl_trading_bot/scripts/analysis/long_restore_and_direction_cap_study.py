#!/usr/bin/env python3
"""
LONG Pattern Restoration + Direction Cap Study
===============================================

Problem 1 - Direction Bias:
  Current 15 patterns = 2L + 13S (87% SHORT).
  59-pat backup has 12 LONGs, all with WR Excess > 20pp (genuine edge).
  10 LONGs dropped because they didn't reach STRONG (3-4 phase score)
  in tpsl_regime_bias_study.py.  Re-evaluate with Phase 4 Optimal TP/SL.

Problem 2 - Portfolio Direction Concentration:
  Hedge N=9 allows all 9 slots to be same direction.
  No direction cap exists in _route_signal() or simulate_hedge_simple().

Five Phases:
  1. Re-evaluate 10 dropped LONGs with 4-phase STRONG classification
  2. Expanded portfolio construction + comparison
  3. Direction cap simulation sweep
  4. WF 3-fold validation of best combination
  5. Final recommendation + production change list

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

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger('long_restore_cap')

# ---------------------------------------------------------------------------
# File paths
# ---------------------------------------------------------------------------
DATA_720D = os.path.join(_PROJECT_ROOT, 'data', 'btc_5m_720days_binance.csv')
PATTERNS_FILE = os.path.join(_PROJECT_ROOT, 'results', 'dynamic_patterns.json')
PATTERNS_59_BACKUP = os.path.join(_PROJECT_ROOT, 'results', 'dynamic_patterns_59pat_backup.json')
OUTPUT_FILE = os.path.join(_PROJECT_ROOT, 'results', 'long_restore_and_direction_cap_study.json')

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TIMEOUT_T864 = 864
MAX_POSITIONS = 9

# 10 dropped LONG patterns (in 59-pat but not in current 15-pat)
DROPPED_LONG_NAMES = [
    'BD-BD-BU', 'BD-DN-MU', 'DN-ST-H', 'H-ST-DN', 'H-U-MU',
    'MD-BD-DN', 'MD-DN-BU', 'MD-DN-DN', 'MD-MU-ST', 'U-MD-H',
]

# Phase 1 grid search
TP_GRID = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 3.5]
SL_GRID_EFF = [1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 3.5, 4.0]

# Phase 1 symmetric test levels
SYMMETRIC_LEVELS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]

# Phase 3 direction caps to sweep
DIRECTION_CAPS = [5, 6, 7, 8, 9]


# ===================================================================
# PHASE 1: Re-evaluate 10 dropped LONGs
# ===================================================================

def phase1_reevaluate(dropped_patterns, sig_index, opens, highs, lows,
                      n_bars, atr_ratio, overlap_bar):
    """
    For each dropped LONG pattern, run 4-phase STRONG classification:
      P1: Original MAE/MFE TP/SL WR Excess > 5pp
      P2: Direction Flip at optimal TP/SL -> actual_wr < RW WR = GENUINE
      P3: Symmetric TP=SL all levels WR > 50%
      P4: Grid-search optimal WR Excess > 5pp
    Score 3-4 = STRONG, 2 = MODERATE, 1 = WEAK, 0 = NONE
    """
    print("\n" + "=" * 110)
    print("  PHASE 1: RE-EVALUATE 10 DROPPED LONG PATTERNS")
    print(f"  Grid: TP {TP_GRID} x SL {SL_GRID_EFF}")
    print(f"  Symmetric: {SYMMETRIC_LEVELS}")
    print(f"  Timeout: {TIMEOUT_T864} bars, ATR-scaled")
    print("=" * 110)

    t0 = time.time()
    results = []

    for p in dropped_patterns:
        pattern = p['pattern']
        bars = sig_index.get(pattern)
        if not bars:
            logger.warning(f"  No signals for {pattern}")
            continue

        direction_int = 1  # All LONG
        orig_tp, orig_sl = p['tp'], p['sl']

        # --- P1: Original TP/SL WR Excess ---
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

        for tp_test in TP_GRID:
            for sl_test in SL_GRID_EFF:
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
            bars, -1, best_tp, best_sl,
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
            'pattern': pattern, 'direction': 'LONG',
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
    print("  " + "-" * 100)

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
    print(f"\n  STRONG: {strong}/10, MODERATE: {moderate}/10  ({elapsed:.1f}s)")

    return results


# ===================================================================
# PHASE 2: Expanded Portfolio Comparison
# ===================================================================

def phase2_expanded_portfolio(current_patterns, restored_longs, sig_index,
                              opens, highs, lows, n_bars, atr_ratio, overlap_bar):
    """
    Compare:
      - baseline: current 15 patterns (2L + 13S)
      - expanded: current 15 + STRONG restored LONGs
    Both with Hedge N=9, T864.
    """
    expanded = list(current_patterns) + restored_longs
    n_long_exp = sum(1 for p in expanded if p['direction'] == 'LONG')
    n_short_exp = sum(1 for p in expanded if p['direction'] == 'SHORT')

    print("\n" + "=" * 110)
    print("  PHASE 2: EXPANDED PORTFOLIO COMPARISON")
    print(f"  Baseline: {len(current_patterns)} patterns (2L + 13S)")
    print(f"  Expanded: {len(expanded)} patterns ({n_long_exp}L + {n_short_exp}S)")
    print(f"  Hedge N={MAX_POSITIONS}, T864, ATR-scaled")
    print("=" * 110)

    scenarios = {}
    for label, pats in [('baseline_15', current_patterns), ('expanded', expanded)]:
        sched = build_custom_schedule(pats, sig_index, atr_ratio, opens, n_bars)
        sim_results = simulate_hedge_simple(
            MAX_POSITIONS, sched, opens, highs, lows,
            overlap_bar, n_bars, TIMEOUT_T864)
        portfolio = [(r['entry_bar'], r['exit_bar'], r['scaled_pnl']) for r in sim_results]
        stats = calc_stats(portfolio)
        stats['pnl_mdd_ratio'] = pnl_mdd_ratio(stats['cmp_pnl'], stats['cmp_mdd'])
        stats['long_trades'] = sum(1 for r in sim_results if r['direction'] == 1)
        stats['short_trades'] = sum(1 for r in sim_results if r['direction'] == -1)
        stats['n_patterns'] = len(pats)
        scenarios[label] = stats

    # Print comparison
    print(f"\n  {'Metric':>16} | {'Baseline(15)':>14} | {'Expanded':>14}")
    print("  " + "-" * 52)
    for key in ['n_patterns', 'trades', 'wr', 'pf', 'cmp_pnl', 'cmp_mdd',
                'pnl_mdd_ratio', 'long_trades', 'short_trades']:
        v1 = scenarios['baseline_15'].get(key, 0)
        v2 = scenarios['expanded'].get(key, 0)
        if isinstance(v1, float):
            print(f"  {key:>16} | {v1:14.1f} | {v2:14.1f}")
        else:
            print(f"  {key:>16} | {v1:14d} | {v2:14d}")

    return scenarios, expanded


# ===================================================================
# PHASE 3: Direction Cap Simulation
# ===================================================================

def simulate_hedge_with_cap(max_positions, max_same_dir, schedule,
                            opens, highs, lows, lo_bar, hi_bar,
                            timeout_bars=None):
    """
    Like simulate_hedge_simple but with direction cap.
    max_same_dir: max positions allowed in same direction.
    """
    slots = {}
    slot_counter = 0
    results = []
    weight = 1.0 / max_positions
    n_bars = len(opens)

    def _close_slot(sid, exit_bar, exit_price, reason):
        pos = slots.pop(sid)
        pm = pos.direction * (exit_price / pos.entry_price - 1) * 100
        pnl = pm * LEVERAGE - FEE
        scaled = pnl * weight
        results.append({
            'entry_bar': pos.entry_bar, 'exit_bar': exit_bar,
            'scaled_pnl': scaled, 'raw_pnl': pnl,
            'reason': reason, 'direction': pos.direction,
            'pattern': pos.pattern,
        })

    for bar in range(lo_bar, hi_bar):
        if bar >= n_bars:
            break

        # Check TP/SL
        sids_to_close = []
        for sid, pos in slots.items():
            h, l, o = highs[bar], lows[bar], opens[bar]
            if pos.direction == 1:
                tp_hit = h >= pos.tp_price
                sl_hit = l <= pos.sl_price
            else:
                tp_hit = l <= pos.tp_price
                sl_hit = h >= pos.sl_price

            if tp_hit and sl_hit:
                tp_dist = abs(pos.tp_price - o)
                sl_dist = abs(pos.sl_price - o)
                if tp_dist <= sl_dist:
                    tp_hit, sl_hit = True, False
                else:
                    tp_hit, sl_hit = False, True

            if tp_hit:
                sids_to_close.append((sid, bar, pos.tp_price, 'TP'))
            elif sl_hit:
                sids_to_close.append((sid, bar, pos.sl_price, 'SL'))

        for sid, ebar, eprice, reason in sids_to_close:
            if sid in slots:
                _close_slot(sid, ebar, eprice, reason)

        # Check timeouts
        if timeout_bars and timeout_bars > 0:
            for sid in list(slots):
                if sid not in slots:
                    continue
                pos = slots[sid]
                if bar - pos.entry_bar >= timeout_bars:
                    slots.pop(sid)  # DROP

        # Process signals
        signals = schedule.get(bar)
        if not signals:
            continue

        active_patterns = {s.pattern for s in slots.values()}
        signals_sorted = sorted(signals, key=lambda s: -s[1])

        chosen = None
        for pat_name, direction, tp, sl, entry_price, sig_bar in signals_sorted:
            if pat_name in active_patterns:
                continue
            chosen = (pat_name, direction, tp, sl, entry_price, sig_bar)
            break

        if chosen is None:
            continue

        pat_name, sig_dir, tp, sl, entry_price, sig_bar = chosen

        # Direction cap check
        same_dir_count = sum(1 for s in slots.values() if s.direction == sig_dir)
        if same_dir_count >= max_same_dir:
            continue  # SKIP

        if len(slots) < max_positions:
            slot_counter += 1
            sid = f's{slot_counter}'
            pos = ActivePosition(pat_name, sig_dir, entry_price, tp, sl, bar, sig_bar)
            slots[sid] = pos

    # Close remaining at end
    for sid in list(slots):
        exit_price = opens[min(hi_bar - 1, n_bars - 1)]
        _close_slot(sid, hi_bar - 1, exit_price, 'END')

    return results


def run_wf_3fold_with_cap(max_positions, max_same_dir, schedule,
                          opens, highs, lows, overlap_bar, n_bars,
                          timeout_bars=None):
    """3-fold expanding window WF with direction cap."""
    avail_bars = n_bars - overlap_bar
    quarter = avail_bars // 4
    folds = [
        (overlap_bar, overlap_bar + quarter,
         overlap_bar + quarter, overlap_bar + 2 * quarter),
        (overlap_bar, overlap_bar + 2 * quarter,
         overlap_bar + 2 * quarter, overlap_bar + 3 * quarter),
        (overlap_bar, overlap_bar + 3 * quarter,
         overlap_bar + 3 * quarter, min(overlap_bar + avail_bars, n_bars)),
    ]
    wf_results = []
    for fold_idx, (is_lo, is_hi, oos_lo, oos_hi) in enumerate(folds):
        oos_res = simulate_hedge_with_cap(
            max_positions, max_same_dir, schedule, opens, highs, lows,
            oos_lo, oos_hi, timeout_bars)
        oos_port = [(r['entry_bar'], r['exit_bar'], r['scaled_pnl']) for r in oos_res]
        oos_stats = calc_stats(oos_port)
        wf_results.append({
            'fold': fold_idx + 1,
            'oos_trades': oos_stats['trades'],
            'oos_wr': oos_stats['wr'],
            'oos_pnl': oos_stats['cmp_pnl'],
            'oos_mdd': oos_stats['cmp_mdd'],
            'pass': oos_stats['cmp_pnl'] > 0,
        })
    pass_count = sum(1 for f in wf_results if f['pass'])
    return wf_results, f"{'PASS' if pass_count == 3 else 'FAIL'} ({pass_count}/3)"


def phase3_direction_cap(pattern_sets, sig_index, opens, highs, lows,
                         n_bars, atr_ratio, overlap_bar):
    """
    Sweep direction caps for each pattern set.
    pattern_sets: dict of {label: pattern_list}
    """
    print("\n" + "=" * 110)
    print("  PHASE 3: DIRECTION CAP SIMULATION")
    print(f"  Caps: {DIRECTION_CAPS} (9 = unlimited)")
    print(f"  Hedge N={MAX_POSITIONS}, T864, ATR-scaled")
    print("=" * 110)

    all_results = {}

    for label, pats in pattern_sets.items():
        sched = build_custom_schedule(pats, sig_index, atr_ratio, opens, n_bars)
        cap_results = {}

        for cap in DIRECTION_CAPS:
            sim_results = simulate_hedge_with_cap(
                MAX_POSITIONS, cap, sched, opens, highs, lows,
                overlap_bar, n_bars, TIMEOUT_T864)
            portfolio = [(r['entry_bar'], r['exit_bar'], r['scaled_pnl']) for r in sim_results]
            stats = calc_stats(portfolio)
            stats['pnl_mdd_ratio'] = pnl_mdd_ratio(stats['cmp_pnl'], stats['cmp_mdd'])
            stats['long_trades'] = sum(1 for r in sim_results if r['direction'] == 1)
            stats['short_trades'] = sum(1 for r in sim_results if r['direction'] == -1)
            cap_results[cap] = stats

        all_results[label] = cap_results

    # Print results
    for label, cap_results in all_results.items():
        n_pats = len(pattern_sets[label])
        print(f"\n  --- {label} ({n_pats} patterns) ---")
        print(f"  {'Cap':>4} | {'Trades':>7} {'WR':>6} {'PF':>6} | "
              f"{'CmpPnL':>9} {'CmpMDD':>8} {'PnL/MDD':>8} | "
              f"{'Long':>5} {'Short':>6} {'L%':>5}")
        print("  " + "-" * 80)
        for cap in DIRECTION_CAPS:
            s = cap_results[cap]
            total_dir = s['long_trades'] + s['short_trades']
            l_pct = s['long_trades'] / max(total_dir, 1) * 100
            cap_label = f"{cap}" if cap < 9 else "9(no)"
            print(f"  {cap_label:>4} | {s['trades']:7d} {s['wr']:5.1f}% {s['pf']:6.2f} | "
                  f"{s['cmp_pnl']:>+9.1f} {s['cmp_mdd']:8.1f} {s['pnl_mdd_ratio']:8.2f} | "
                  f"{s['long_trades']:5d} {s['short_trades']:6d} {l_pct:4.1f}%")

    return all_results


# ===================================================================
# PHASE 4: WF 3-fold Validation
# ===================================================================

def phase4_wf_validation(best_pats, best_cap, sig_index,
                         opens, highs, lows, n_bars, atr_ratio, overlap_bar):
    """WF 3-fold for the best combination."""
    print("\n" + "=" * 110)
    print("  PHASE 4: WF 3-FOLD VALIDATION")
    n_l = sum(1 for p in best_pats if p['direction'] == 'LONG')
    n_s = sum(1 for p in best_pats if p['direction'] == 'SHORT')
    print(f"  Patterns: {len(best_pats)} ({n_l}L + {n_s}S), Cap: {best_cap}")
    print("=" * 110)

    sched = build_custom_schedule(best_pats, sig_index, atr_ratio, opens, n_bars)

    # Run WF with cap
    wf_cap, verdict_cap = run_wf_3fold_with_cap(
        MAX_POSITIONS, best_cap, sched, opens, highs, lows,
        overlap_bar, n_bars, TIMEOUT_T864)

    # Also run without cap for reference
    wf_nocap, verdict_nocap = run_wf_3fold(
        MAX_POSITIONS, sched, opens, highs, lows,
        overlap_bar, n_bars, TIMEOUT_T864)

    print(f"\n  {'Fold':>4} | {'With Cap':>12} {'PnL':>8} {'MDD':>8} | "
          f"{'No Cap':>12} {'PnL':>8} {'MDD':>8}")
    print("  " + "-" * 65)
    for fc, fn in zip(wf_cap, wf_nocap):
        pc = 'PASS' if fc['pass'] else 'FAIL'
        pn = 'PASS' if fn['pass'] else 'FAIL'
        print(f"  {fc['fold']:4d} | {pc:>12} {fc['oos_pnl']:>+8.1f} {fc['oos_mdd']:8.1f} | "
              f"{pn:>12} {fn['oos_pnl']:>+8.1f} {fn['oos_mdd']:8.1f}")

    print(f"\n  With Cap {best_cap}: {verdict_cap}")
    print(f"  No Cap:        {verdict_nocap}")

    return {
        'with_cap': {'folds': wf_cap, 'verdict': verdict_cap},
        'no_cap': {'folds': wf_nocap, 'verdict': verdict_nocap},
    }


# ===================================================================
# PHASE 5: Final Recommendation
# ===================================================================

def phase5_recommendation(p1_results, p2_scenarios, p3_results, p4_wf,
                          current_patterns, restored_longs, best_cap):
    """Final recommendation and production change list."""
    print("\n" + "=" * 110)
    print("  PHASE 5: FINAL RECOMMENDATION")
    print("=" * 110)

    strong_patterns = [r for r in p1_results if r['overall'] == 'STRONG']
    moderate_patterns = [r for r in p1_results if r['overall'] == 'MODERATE']

    print(f"\n  Restored STRONG LONGs: {len(strong_patterns)}")
    for r in strong_patterns:
        print(f"    {r['pattern']}: TP={r['optimal_tp']:.2f} SL={r['optimal_sl']:.2f} "
              f"WR Excess={r['optimal_excess']:+.1f}pp Score={r['score']}/4")

    if moderate_patterns:
        print(f"\n  MODERATE LONGs (not restored): {len(moderate_patterns)}")
        for r in moderate_patterns:
            print(f"    {r['pattern']}: Score={r['score']}/4")

    # Expanded vs baseline
    base = p2_scenarios.get('baseline_15', {})
    exp = p2_scenarios.get('expanded', {})
    if base and exp:
        pnl_diff = exp.get('cmp_pnl', 0) - base.get('cmp_pnl', 0)
        mdd_diff = exp.get('cmp_mdd', 0) - base.get('cmp_mdd', 0)
        print(f"\n  Portfolio Impact:")
        print(f"    PnL: {base.get('cmp_pnl', 0):+.1f}% -> {exp.get('cmp_pnl', 0):+.1f}% ({pnl_diff:+.1f}%)")
        print(f"    MDD: {base.get('cmp_mdd', 0):.1f}% -> {exp.get('cmp_mdd', 0):.1f}% ({mdd_diff:+.1f}%)")

    # Direction cap
    print(f"\n  Best Direction Cap: {best_cap}")

    # WF
    wf_cap_verdict = p4_wf.get('with_cap', {}).get('verdict', 'N/A')
    print(f"  WF Verdict: {wf_cap_verdict}")

    # Production changes
    print(f"\n  --- PRODUCTION CHANGES ---")
    if strong_patterns:
        print(f"  1. Add {len(strong_patterns)} LONG patterns to dynamic_patterns.json:")
        for r in strong_patterns:
            print(f"     {r['pattern']}_LONG: TP={r['optimal_tp']:.2f} SL={r['optimal_sl']:.2f}")
    if best_cap < MAX_POSITIONS:
        print(f"  2. Add direction cap = {best_cap} to bot.py _route_signal()")
        print(f"     (max {best_cap} positions in same direction)")
    else:
        print(f"  2. No direction cap needed (cap={best_cap} = unlimited)")

    recommendation = {
        'strong_longs_restored': len(strong_patterns),
        'strong_patterns': [{
            'pattern': r['pattern'], 'tp': r['optimal_tp'], 'sl': r['optimal_sl'],
            'excess': r['optimal_excess'], 'score': r['score'],
        } for r in strong_patterns],
        'direction_cap': best_cap,
        'wf_verdict': wf_cap_verdict,
        'pnl_change': round(pnl_diff, 1) if base and exp else 0,
        'mdd_change': round(mdd_diff, 1) if base and exp else 0,
    }
    return recommendation


# ===================================================================
# MAIN
# ===================================================================

def main():
    t0 = time.time()

    print("=" * 110)
    print("  LONG PATTERN RESTORATION + DIRECTION CAP STUDY")
    print("=" * 110)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    print("\n--- Loading 720d data ---")
    df = load_and_classify(DATA_720D)
    n_bars = len(df)
    opens = df['open'].values.astype(np.float64)
    highs = df['high'].values.astype(np.float64)
    lows = df['low'].values.astype(np.float64)
    closes = df['close'].values.astype(np.float64)
    types = df['rctype'].values

    overlap_bar = find_overlap_bar(df)
    avail_days = (n_bars - overlap_bar) / BARS_PER_DAY
    logger.info(f"720d: {n_bars} bars, overlap: {overlap_bar}, avail: {avail_days:.1f}d")

    sig_index = build_signal_index(types, n_bars)
    atr_ratio = compute_atr_ratio(highs, lows, closes, ATR_PERIOD, ATR_WINDOW)

    # ------------------------------------------------------------------
    # Load patterns
    # ------------------------------------------------------------------
    current_patterns = load_patterns()
    logger.info(f"Current: {len(current_patterns)} patterns")

    patterns_59 = load_patterns_from_file(PATTERNS_59_BACKUP)
    logger.info(f"59-pat backup: {len(patterns_59)} patterns")

    # Identify dropped LONGs
    current_long_names = {p['pattern'] for p in current_patterns if p['direction'] == 'LONG'}
    dropped_longs = [p for p in patterns_59
                     if p['direction'] == 'LONG' and p['pattern'] in DROPPED_LONG_NAMES]
    logger.info(f"Dropped LONGs to evaluate: {len(dropped_longs)}")

    for p in dropped_longs:
        print(f"  {p['pattern']}: TP={p['tp']:.2f} SL={p['sl']:.2f}")

    output = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'study': 'LONG Pattern Restoration + Direction Cap',
        'data_bars': n_bars,
        'overlap_bar': overlap_bar,
        'current_patterns': len(current_patterns),
        'dropped_longs_evaluated': len(dropped_longs),
        'parameters': {
            'max_positions': MAX_POSITIONS,
            'timeout': TIMEOUT_T864,
            'tp_grid': TP_GRID,
            'sl_grid_eff': SL_GRID_EFF,
            'symmetric_levels': SYMMETRIC_LEVELS,
            'direction_caps': DIRECTION_CAPS,
            'leverage': LEVERAGE,
            'fee': FEE,
        },
    }

    # ------------------------------------------------------------------
    # PHASE 1
    # ------------------------------------------------------------------
    p1_results = phase1_reevaluate(
        dropped_longs, sig_index, opens, highs, lows,
        n_bars, atr_ratio, overlap_bar)
    output['phase1_reevaluation'] = p1_results

    # Build restored LONG list (STRONG only) with optimal TP/SL
    restored_longs = []
    for r in p1_results:
        if r['overall'] == 'STRONG':
            restored_longs.append({
                'pattern': r['pattern'],
                'direction': 'LONG',
                'tp': r['optimal_tp'],
                'sl': r['optimal_sl'],
            })
    output['restored_longs'] = restored_longs
    logger.info(f"Restored STRONG LONGs: {len(restored_longs)}")

    # ------------------------------------------------------------------
    # PHASE 2
    # ------------------------------------------------------------------
    p2_scenarios, expanded_patterns = phase2_expanded_portfolio(
        current_patterns, restored_longs, sig_index,
        opens, highs, lows, n_bars, atr_ratio, overlap_bar)
    output['phase2_portfolio_comparison'] = p2_scenarios

    # ------------------------------------------------------------------
    # PHASE 3
    # ------------------------------------------------------------------
    pattern_sets = {
        'baseline_15': current_patterns,
        'expanded': expanded_patterns,
    }
    p3_results = phase3_direction_cap(
        pattern_sets, sig_index, opens, highs, lows,
        n_bars, atr_ratio, overlap_bar)
    output['phase3_direction_cap'] = {
        label: {str(cap): stats for cap, stats in cap_results.items()}
        for label, cap_results in p3_results.items()
    }

    # Find best combo: maximize PnL/MDD among WF-passing scenarios
    # We'll check WF for top candidates
    best_label = None
    best_cap = MAX_POSITIONS  # default: no cap
    best_ratio = -999

    for label, cap_results in p3_results.items():
        for cap, stats in cap_results.items():
            ratio = stats['pnl_mdd_ratio']
            if ratio > best_ratio:
                best_ratio = ratio
                best_label = label
                best_cap = cap

    logger.info(f"Best combo (pre-WF): {best_label} cap={best_cap} ratio={best_ratio:.2f}")

    # Use expanded set for WF if it's better
    best_pats = pattern_sets.get(best_label, expanded_patterns)

    # ------------------------------------------------------------------
    # PHASE 4
    # ------------------------------------------------------------------
    p4_wf = phase4_wf_validation(
        best_pats, best_cap, sig_index,
        opens, highs, lows, n_bars, atr_ratio, overlap_bar)
    output['phase4_wf_validation'] = p4_wf

    # If WF fails with best cap, try other caps
    wf_verdict = p4_wf['with_cap']['verdict']
    if '3/3' not in wf_verdict:
        logger.info(f"Best cap {best_cap} failed WF, trying alternatives...")
        sched = build_custom_schedule(best_pats, sig_index, atr_ratio, opens, n_bars)
        for cap in DIRECTION_CAPS:
            if cap == best_cap:
                continue
            wf_folds, verdict = run_wf_3fold_with_cap(
                MAX_POSITIONS, cap, sched, opens, highs, lows,
                overlap_bar, n_bars, TIMEOUT_T864)
            if '3/3' in verdict:
                best_cap = cap
                p4_wf['with_cap'] = {'folds': wf_folds, 'verdict': verdict}
                logger.info(f"  Cap {cap} passes WF: {verdict}")
                break

    # Also try baseline patterns if expanded fails
    if '3/3' not in p4_wf['with_cap']['verdict'] and best_label == 'expanded':
        logger.info("Expanded fails WF, trying baseline_15...")
        best_pats = current_patterns
        best_label = 'baseline_15'
        sched = build_custom_schedule(best_pats, sig_index, atr_ratio, opens, n_bars)
        for cap in DIRECTION_CAPS:
            wf_folds, verdict = run_wf_3fold_with_cap(
                MAX_POSITIONS, cap, sched, opens, highs, lows,
                overlap_bar, n_bars, TIMEOUT_T864)
            if '3/3' in verdict:
                best_cap = cap
                p4_wf['with_cap'] = {'folds': wf_folds, 'verdict': verdict}
                p4_wf['fallback'] = f'baseline_15 cap={cap}'
                logger.info(f"  Baseline cap={cap} passes WF")
                break

    output['best_combination'] = {
        'pattern_set': best_label,
        'n_patterns': len(best_pats),
        'direction_cap': best_cap,
        'wf_verdict': p4_wf['with_cap']['verdict'],
    }

    # ------------------------------------------------------------------
    # PHASE 5
    # ------------------------------------------------------------------
    recommendation = phase5_recommendation(
        p1_results, p2_scenarios, p3_results, p4_wf,
        current_patterns, restored_longs, best_cap)
    output['phase5_recommendation'] = recommendation

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    elapsed = time.time() - t0
    print(f"\n  Total elapsed: {elapsed:.1f}s")
    print(f"  Results saved to {OUTPUT_FILE}")
    print("=" * 110)


if __name__ == '__main__':
    main()
