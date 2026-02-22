#!/usr/bin/env python3
"""
ATR-Scaled Backtest Study — Scanner vs Production Condition Gap Analysis
========================================================================

Core Question:
  Scanner(pattern_scanner.py)는 ATR scaling 없이 고정 TP/SL로 패턴을 발굴·선별한다.
  하지만 Production은 ATR-scaled TP/SL(ATR14/median576, clamp [0.6,1.7])로 실행한다.
  즉 **발굴 조건과 실행 조건이 다르다**. 이 gap이 패턴 선별/성과에 영향을 주는가?

Four Phases:
  1. Individual Pattern — Fixed vs ATR-Scaled
     15개 현재 패턴: fixed vs ATR-scaled 백테스트 비교 (WR, edge, trades, WR Excess)
  2. Pattern Selection Impact (59 patterns re-evaluation)
     59개 MAE/MFE 패턴을 ATR-scaled로 재평가, 품질 필터 적용 후 패턴 셋 비교
  3. Portfolio Comparison (Hedge N=5, WF 3-fold)
     Fixed vs ATR-scaled 포트폴리오, T864 timeout, WF 3-fold 비교
  4. ATR Distribution Analysis
     270d 구간 ATR ratio 분포, 시간대별 변화

Standard Research Protocol:
  - Entry: next bar open after signal
  - Exit: intrabar high/low (distance-based, same-bar resolution from bar OPEN)
  - Sizing: compound (1/N per slot)
  - Fee: FEE_PCT * LEVERAGE = 0.30% capital-space
  - Timeout: MAX_BARS = 288 (24h) -> DROP from PnL
  - MC: sign randomization, 3 seeds (42, 123, 7777), max p < 0.01
  - WF: 3-fold expanding window within 270d overlap of 720d data
  - classify_candle: production import (NEVER self-implement)

Data:
  - btc_5m_720days_binance.csv (720d with overlap region)

Patterns:
  - Current 15 patterns from results/dynamic_patterns.json
  - 59-pattern backup from results/dynamic_patterns_59pat_backup.json
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
    build_signal_schedule, ActivePosition, pnl_mdd_ratio,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger('atr_scaled_backtest')

# ---------------------------------------------------------------------------
# File paths
# ---------------------------------------------------------------------------
DATA_720D = os.path.join(_PROJECT_ROOT, 'data', 'btc_5m_720days_binance.csv')
PATTERNS_FILE = os.path.join(_PROJECT_ROOT, 'results', 'dynamic_patterns.json')
PATTERNS_59_BACKUP = os.path.join(_PROJECT_ROOT, 'results', 'dynamic_patterns_59pat_backup.json')
OUTPUT_FILE = os.path.join(_PROJECT_ROOT, 'results', 'atr_scaled_backtest_study.json')

# ---------------------------------------------------------------------------
# MC test (from scanner)
# ---------------------------------------------------------------------------
MC_SIMS = 5000
MC_SEEDS = [42, 123, 7777]


def mc_test(pnls):
    """Multi-seed sign randomization MC test. Returns max p-value (conservative)."""
    if len(pnls) < 5:
        return 1.0
    pnls_arr = np.array(pnls)
    actual = np.sum(pnls_arr)
    p_vals = []
    for seed in MC_SEEDS:
        rng = np.random.RandomState(seed)
        signs = rng.choice([-1, 1], size=(MC_SIMS, len(pnls_arr)))
        rand_sums = signs @ pnls_arr
        p_vals.append(float(np.mean(rand_sums >= actual)))
    return max(p_vals)


# ---------------------------------------------------------------------------
# Quality filter thresholds (same as production selection)
# ---------------------------------------------------------------------------
EDGE_THRESHOLD = 21.8    # pp
WR_THRESHOLD = 60.0      # %
SL_MIN = 1.0             # %
MC_P_THRESHOLD = 0.01
MIN_TRADES = 25
WR_EXCESS_THRESHOLD = 5.0  # pp

# Portfolio parameters
TIMEOUT_T864 = 864
MAX_POSITIONS = 5


# ===================================================================
# Build fixed (no-ATR) schedule — same as build_signal_schedule but
# forces r=1.0 (no ATR scaling), still applies slippage buffer
# ===================================================================

def build_fixed_schedule(patterns, sig_index, opens, n_bars):
    """Build signal schedule WITHOUT ATR scaling (scanner conditions)."""
    schedule = {}
    for p in patterns:
        bars = sig_index.get(p['pattern'])
        if not bars:
            continue
        direction = 1 if p['direction'] == 'LONG' else -1
        base_tp, base_sl = p['tp'], p['sl']

        for sig_bar in bars:
            entry_bar = sig_bar + 1
            if entry_bar >= n_bars:
                continue
            entry_price = opens[entry_bar]
            if entry_price <= 0:
                continue

            # Fixed: r = 1.0 always (scanner condition)
            tp = base_tp * 1.0 + SLIPPAGE_BUFFER
            sl = max(0.1, base_sl * 1.0 - SLIPPAGE_BUFFER)

            if entry_bar not in schedule:
                schedule[entry_bar] = []
            schedule[entry_bar].append((p['pattern'], direction, tp, sl, entry_price, sig_bar))

    return schedule


# ===================================================================
# PHASE 1: Individual Pattern — Fixed vs ATR-Scaled
# ===================================================================

def phase1_individual_comparison(patterns, sig_index, opens, highs, lows,
                                  n_bars, atr_ratio, overlap_bar):
    """
    For each of 15 patterns: fixed vs ATR-scaled backtest.
    Compare WR, edge, trades, WR Excess.
    """
    print("\n" + "=" * 120)
    print("  PHASE 1: INDIVIDUAL PATTERN — FIXED vs ATR-SCALED")
    print("  Fixed: scanner condition (r=1.0)")
    print("  ATR-Scaled: production condition (ATR14/median576, clamp[0.6,1.7])")
    print("=" * 120)

    results = []

    for p in patterns:
        bars = sig_index.get(p['pattern'])
        if not bars:
            continue

        direction_int = 1 if p['direction'] == 'LONG' else -1
        tp, sl = p['tp'], p['sl']
        random_wr = sl / (tp + sl) * 100

        # Fixed backtest (scanner condition)
        fixed_trades = backtest_pattern_trades(
            bars, direction_int, tp, sl,
            opens, highs, lows, n_bars,
            atr_ratio=None, use_atr=False,
            lo_bar=overlap_bar, timeout_bars=MAX_BARS)
        fixed_wr, fixed_edge, fixed_n = compute_wr_and_edge(fixed_trades)
        fixed_excess = fixed_wr - random_wr

        # ATR-scaled backtest (production condition)
        atr_trades = backtest_pattern_trades(
            bars, direction_int, tp, sl,
            opens, highs, lows, n_bars,
            atr_ratio=atr_ratio, use_atr=True,
            lo_bar=overlap_bar, timeout_bars=MAX_BARS)
        atr_wr, atr_edge, atr_n = compute_wr_and_edge(atr_trades)
        atr_excess = atr_wr - random_wr

        # Compute compound PnL for both
        def compound_pnl(trades):
            if not trades:
                return 0.0
            eq = 1.0
            for t in trades:
                eq *= (1 + t['pnl'] / 100)
            return (eq - 1) * 100

        fixed_pnl = compound_pnl(fixed_trades)
        atr_pnl = compound_pnl(atr_trades)

        results.append({
            'pattern': p['pattern'],
            'direction': p['direction'],
            'tp': tp,
            'sl': sl,
            'random_wr': round(random_wr, 1),
            # Fixed
            'fixed_trades': fixed_n,
            'fixed_wr': round(fixed_wr, 1),
            'fixed_edge': round(fixed_edge, 3),
            'fixed_excess': round(fixed_excess, 1),
            'fixed_pnl': round(fixed_pnl, 1),
            # ATR-scaled
            'atr_trades': atr_n,
            'atr_wr': round(atr_wr, 1),
            'atr_edge': round(atr_edge, 3),
            'atr_excess': round(atr_excess, 1),
            'atr_pnl': round(atr_pnl, 1),
            # Deltas
            'wr_delta': round(atr_wr - fixed_wr, 1),
            'edge_delta': round(atr_edge - fixed_edge, 3),
            'excess_delta': round(atr_excess - fixed_excess, 1),
            'trades_delta': atr_n - fixed_n,
            'pnl_delta': round(atr_pnl - fixed_pnl, 1),
        })

    results.sort(key=lambda x: -abs(x['wr_delta']))

    # Print table
    print(f"\n  {'#':>2} {'Pattern':<12} {'Dir':>5} | "
          f"{'FxTr':>4} {'FxWR':>6} {'FxEdge':>7} {'FxExc':>6} {'FxPnL':>8} | "
          f"{'AtrTr':>5} {'AtrWR':>6} {'AtrEdg':>7} {'AtrExc':>6} {'AtrPnL':>8} | "
          f"{'dWR':>5} {'dEdge':>7} {'dExc':>5} {'dPnL':>7}")
    print("  " + "-" * 140)

    for i, r in enumerate(results, 1):
        print(f"  {i:2d} {r['pattern']:<12} {r['direction']:>5} | "
              f"{r['fixed_trades']:4d} {r['fixed_wr']:5.1f}% {r['fixed_edge']:>+7.3f} "
              f"{r['fixed_excess']:>+5.1f} {r['fixed_pnl']:>+7.1f}% | "
              f"{r['atr_trades']:5d} {r['atr_wr']:5.1f}% {r['atr_edge']:>+7.3f} "
              f"{r['atr_excess']:>+5.1f} {r['atr_pnl']:>+7.1f}% | "
              f"{r['wr_delta']:>+4.1f} {r['edge_delta']:>+7.3f} "
              f"{r['excess_delta']:>+4.1f} {r['pnl_delta']:>+6.1f}%")

    # Summary
    avg_wr_delta = np.mean([r['wr_delta'] for r in results])
    avg_edge_delta = np.mean([r['edge_delta'] for r in results])
    avg_excess_delta = np.mean([r['excess_delta'] for r in results])
    avg_pnl_delta = np.mean([r['pnl_delta'] for r in results])
    avg_trades_delta = np.mean([r['trades_delta'] for r in results])

    print(f"\n  Summary ({len(results)} patterns):")
    print(f"    Avg WR delta (ATR - Fixed):     {avg_wr_delta:+.1f}pp")
    print(f"    Avg edge delta:                 {avg_edge_delta:+.3f}%/trade")
    print(f"    Avg WR Excess delta:            {avg_excess_delta:+.1f}pp")
    print(f"    Avg PnL delta:                  {avg_pnl_delta:+.1f}%")
    print(f"    Avg trades delta:               {avg_trades_delta:+.1f}")

    patterns_wr_improved = sum(1 for r in results if r['wr_delta'] > 0)
    patterns_wr_worse = sum(1 for r in results if r['wr_delta'] < 0)
    patterns_wr_same = sum(1 for r in results if r['wr_delta'] == 0)
    print(f"    WR improved: {patterns_wr_improved}, worse: {patterns_wr_worse}, same: {patterns_wr_same}")

    return results


# ===================================================================
# PHASE 2: Pattern Selection Impact (59 patterns re-evaluation)
# ===================================================================

def phase2_pattern_selection(sig_index, opens, highs, lows, n_bars,
                              atr_ratio, overlap_bar, current_patterns):
    """
    Re-evaluate 59 MAE/MFE patterns under ATR-scaled conditions.
    Apply quality filters, compare resulting set with current 15.
    """
    print("\n" + "=" * 120)
    print("  PHASE 2: PATTERN SELECTION IMPACT (59 patterns, ATR-scaled re-evaluation)")
    print(f"  Filters: Edge>={EDGE_THRESHOLD}pp, WR>={WR_THRESHOLD}%, SL>={SL_MIN}%, "
          f"MC<{MC_P_THRESHOLD}, min_trades>={MIN_TRADES}, WR_Excess>{WR_EXCESS_THRESHOLD}pp")
    print("=" * 120)

    if not os.path.exists(PATTERNS_59_BACKUP):
        print(f"  ERROR: 59-pattern backup not found at {PATTERNS_59_BACKUP}")
        return None

    patterns_59 = load_patterns_from_file(PATTERNS_59_BACKUP)
    logger.info(f"Loaded {len(patterns_59)} patterns from 59-pat backup")

    fixed_results = []
    atr_results = []

    for p in patterns_59:
        bars = sig_index.get(p['pattern'])
        if not bars:
            continue

        direction_int = 1 if p['direction'] == 'LONG' else -1
        tp, sl = p['tp'], p['sl']
        random_wr = sl / (tp + sl) * 100

        # --- Fixed backtest ---
        f_trades = backtest_pattern_trades(
            bars, direction_int, tp, sl,
            opens, highs, lows, n_bars,
            atr_ratio=None, use_atr=False,
            lo_bar=overlap_bar, timeout_bars=MAX_BARS)
        f_wr, f_edge, f_n = compute_wr_and_edge(f_trades)
        f_excess = f_wr - random_wr

        # MC test for fixed
        f_pnls = [t['pnl'] for t in f_trades]
        f_mc_p = mc_test(f_pnls) if f_pnls else 1.0

        # Edge in pp: use WR-based edge (baseline = 50%)
        f_edge_pp = f_wr - 50.0

        fixed_results.append({
            'pattern': p['pattern'],
            'direction': p['direction'],
            'tp': tp, 'sl': sl,
            'trades': f_n, 'wr': round(f_wr, 1),
            'edge_pp': round(f_edge_pp, 1),
            'wr_excess': round(f_excess, 1),
            'mc_p': round(f_mc_p, 4),
            'edge_per_trade': round(f_edge, 3),
        })

        # --- ATR-scaled backtest ---
        a_trades = backtest_pattern_trades(
            bars, direction_int, tp, sl,
            opens, highs, lows, n_bars,
            atr_ratio=atr_ratio, use_atr=True,
            lo_bar=overlap_bar, timeout_bars=MAX_BARS)
        a_wr, a_edge, a_n = compute_wr_and_edge(a_trades)
        a_excess = a_wr - random_wr

        # MC test for ATR-scaled
        a_pnls = [t['pnl'] for t in a_trades]
        a_mc_p = mc_test(a_pnls) if a_pnls else 1.0

        a_edge_pp = a_wr - 50.0

        atr_results.append({
            'pattern': p['pattern'],
            'direction': p['direction'],
            'tp': tp, 'sl': sl,
            'trades': a_n, 'wr': round(a_wr, 1),
            'edge_pp': round(a_edge_pp, 1),
            'wr_excess': round(a_excess, 1),
            'mc_p': round(a_mc_p, 4),
            'edge_per_trade': round(a_edge, 3),
        })

    # Apply quality filters
    def apply_filters(results_list):
        passed = []
        for r in results_list:
            if r['sl'] < SL_MIN:
                continue
            if r['trades'] < MIN_TRADES:
                continue
            if r['edge_pp'] < EDGE_THRESHOLD:
                continue
            if r['wr'] < WR_THRESHOLD:
                continue
            if r['mc_p'] >= MC_P_THRESHOLD:
                continue
            if r['wr_excess'] <= WR_EXCESS_THRESHOLD:
                continue
            passed.append(r)
        return passed

    fixed_passed = apply_filters(fixed_results)
    atr_passed = apply_filters(atr_results)

    fixed_names = {r['pattern'] + '_' + r['direction'] for r in fixed_passed}
    atr_names = {r['pattern'] + '_' + r['direction'] for r in atr_passed}
    current_names = {p['pattern'] + '_' + p['direction'] for p in current_patterns}

    overlap_fa = fixed_names & atr_names
    only_fixed = fixed_names - atr_names
    only_atr = atr_names - fixed_names

    overlap_fc = fixed_names & current_names
    overlap_ac = atr_names & current_names

    print(f"\n  59-pattern re-evaluation results:")
    print(f"    Fixed passed filters:  {len(fixed_passed)} patterns")
    print(f"    ATR passed filters:    {len(atr_passed)} patterns")
    print(f"    Current (15):          {len(current_names)} patterns")
    print(f"\n    Fixed ∩ ATR:           {len(overlap_fa)}")
    print(f"    Fixed only:            {len(only_fixed)} -> {sorted(only_fixed)}")
    print(f"    ATR only:              {len(only_atr)} -> {sorted(only_atr)}")
    print(f"\n    Fixed ∩ Current(15):   {len(overlap_fc)}")
    print(f"    ATR ∩ Current(15):     {len(overlap_ac)}")
    print(f"    Current not in Fixed:  {sorted(current_names - fixed_names)}")
    print(f"    Current not in ATR:    {sorted(current_names - atr_names)}")

    # Detail table for ATR-passed patterns
    print(f"\n  ATR-scaled passed patterns ({len(atr_passed)}):")
    print(f"  {'#':>2} {'Pattern':<12} {'Dir':>5} | {'TP':>5} {'SL':>5} | "
          f"{'Tr':>4} {'WR':>6} {'EdgePP':>7} {'Excess':>7} {'MC':>6} | {'In15':>4}")
    print("  " + "-" * 85)

    atr_passed.sort(key=lambda x: -x['wr_excess'])
    for i, r in enumerate(atr_passed, 1):
        key = r['pattern'] + '_' + r['direction']
        in_15 = "YES" if key in current_names else "no"
        print(f"  {i:2d} {r['pattern']:<12} {r['direction']:>5} | "
              f"{r['tp']:5.2f} {r['sl']:5.2f} | "
              f"{r['trades']:4d} {r['wr']:5.1f}% {r['edge_pp']:>+6.1f} "
              f"{r['wr_excess']:>+6.1f} {r['mc_p']:6.4f} | {in_15:>4}")

    return {
        'fixed_total': len(fixed_results),
        'atr_total': len(atr_results),
        'fixed_passed': len(fixed_passed),
        'atr_passed': len(atr_passed),
        'fixed_passed_names': sorted(fixed_names),
        'atr_passed_names': sorted(atr_names),
        'overlap_fixed_atr': len(overlap_fa),
        'only_fixed': sorted(only_fixed),
        'only_atr': sorted(only_atr),
        'fixed_overlap_current': len(overlap_fc),
        'atr_overlap_current': len(overlap_ac),
        'current_not_in_fixed': sorted(current_names - fixed_names),
        'current_not_in_atr': sorted(current_names - atr_names),
        'atr_passed_details': atr_passed,
        'fixed_passed_details': fixed_passed,
    }


# ===================================================================
# PHASE 3: Portfolio Comparison (Hedge N=5, WF)
# ===================================================================

def phase3_portfolio_comparison(patterns, sig_index, opens, highs, lows,
                                 n_bars, atr_ratio, overlap_bar):
    """
    Compare Hedge N=5 portfolios:
      A) Fixed TP/SL (scanner condition)
      B) ATR-scaled TP/SL (production condition)
    Both with T864 timeout. WF 3-fold validation.
    """
    print("\n" + "=" * 120)
    print("  PHASE 3: PORTFOLIO COMPARISON (Hedge N=5, WF 3-fold)")
    print("  Scenario A: Fixed TP/SL (scanner, r=1.0)")
    print("  Scenario B: ATR-Scaled TP/SL (production, ATR14/median576)")
    print(f"  Both: T864 timeout ({TIMEOUT_T864} bars), N={MAX_POSITIONS}")
    print("=" * 120)

    # Build schedules
    fixed_sched = build_fixed_schedule(patterns, sig_index, opens, n_bars)
    atr_sched = build_custom_schedule(patterns, sig_index, atr_ratio, opens, n_bars)

    scenarios = {}

    for label, sched, timeout in [
        ('fixed_no_timeout', fixed_sched, None),
        ('fixed_T864', fixed_sched, TIMEOUT_T864),
        ('atr_no_timeout', atr_sched, None),
        ('atr_T864', atr_sched, TIMEOUT_T864),
    ]:
        # Full overlap simulation
        results = simulate_hedge_simple(
            MAX_POSITIONS, sched, opens, highs, lows,
            overlap_bar, n_bars, timeout)
        portfolio = [(r['entry_bar'], r['exit_bar'], r['scaled_pnl']) for r in results]
        stats = calc_stats(portfolio)
        ratio = pnl_mdd_ratio(stats['cmp_pnl'], stats['cmp_mdd'])

        # WF 3-fold
        wf_folds, wf_verdict = run_wf_3fold(
            MAX_POSITIONS, sched, opens, highs, lows,
            overlap_bar, n_bars, timeout)

        scenarios[label] = {
            **stats,
            'pnl_mdd_ratio': ratio,
            'wf_verdict': wf_verdict,
            'wf_folds': wf_folds,
            'timeout': timeout,
        }

    # Print comparison
    print(f"\n  {'Scenario':>20} | {'Trades':>7} {'WR':>6} {'PF':>6} | "
          f"{'CmpPnL':>9} {'CmpMDD':>8} {'PnL/MDD':>8} | {'WF':>12}")
    print("  " + "-" * 95)

    for label in ['fixed_no_timeout', 'fixed_T864', 'atr_no_timeout', 'atr_T864']:
        s = scenarios[label]
        print(f"  {label:>20} | {s['trades']:7d} {s['wr']:5.1f}% {s['pf']:6.2f} | "
              f"{s['cmp_pnl']:>+9.1f} {s['cmp_mdd']:8.1f} {s['pnl_mdd_ratio']:8.2f} | "
              f"{s['wf_verdict']:>12}")

    # WF detail for T864 scenarios
    for label in ['fixed_T864', 'atr_T864']:
        s = scenarios[label]
        print(f"\n  WF Detail: {label}")
        print(f"    {'Fold':>4} | {'OOS Tr':>7} {'OOS WR':>7} {'OOS PnL':>9} {'OOS MDD':>8} {'Pass':>5}")
        for f in s['wf_folds']:
            print(f"    {f['fold']:4d} | {f['oos_trades']:7d} {f['oos_wr']:6.1f}% "
                  f"{f['oos_pnl']:>+9.1f} {f['oos_mdd']:8.1f} "
                  f"{'PASS' if f['pass'] else 'FAIL':>5}")

    # Verdict
    fixed_t864 = scenarios['fixed_T864']
    atr_t864 = scenarios['atr_T864']
    pnl_diff = atr_t864['cmp_pnl'] - fixed_t864['cmp_pnl']
    mdd_diff = atr_t864['cmp_mdd'] - fixed_t864['cmp_mdd']
    ratio_diff = atr_t864['pnl_mdd_ratio'] - fixed_t864['pnl_mdd_ratio']

    print(f"\n  --- T864 COMPARISON VERDICT ---")
    print(f"    Fixed T864:  PnL {fixed_t864['cmp_pnl']:+.1f}%, MDD {fixed_t864['cmp_mdd']:.1f}%, "
          f"PnL/MDD {fixed_t864['pnl_mdd_ratio']:.2f}, WF {fixed_t864['wf_verdict']}")
    print(f"    ATR T864:    PnL {atr_t864['cmp_pnl']:+.1f}%, MDD {atr_t864['cmp_mdd']:.1f}%, "
          f"PnL/MDD {atr_t864['pnl_mdd_ratio']:.2f}, WF {atr_t864['wf_verdict']}")
    print(f"    Deltas:      PnL {pnl_diff:+.1f}%, MDD {mdd_diff:+.1f}%, PnL/MDD {ratio_diff:+.2f}")

    if atr_t864['pnl_mdd_ratio'] > fixed_t864['pnl_mdd_ratio']:
        print(f"    RESULT: ATR scaling IMPROVES risk-adjusted performance")
    elif atr_t864['pnl_mdd_ratio'] < fixed_t864['pnl_mdd_ratio']:
        print(f"    RESULT: ATR scaling REDUCES risk-adjusted performance")
    else:
        print(f"    RESULT: ATR scaling has NO IMPACT on risk-adjusted performance")

    return scenarios


# ===================================================================
# PHASE 4: ATR Distribution Analysis
# ===================================================================

def phase4_atr_distribution(atr_ratio, overlap_bar, n_bars):
    """
    Analyze ATR ratio distribution over the 270d overlap period.
    """
    print("\n" + "=" * 120)
    print("  PHASE 4: ATR RATIO DISTRIBUTION ANALYSIS (270d overlap)")
    print(f"  ATR scaling: ATR({ATR_PERIOD})/median({ATR_WINDOW}), clamp [{CLAMP_LO}, {CLAMP_HI}]")
    print("=" * 120)

    # Get overlap region ATR ratios
    overlap_atr = atr_ratio[overlap_bar:]
    valid_mask = ~np.isnan(overlap_atr)
    valid_atr = overlap_atr[valid_mask]

    if len(valid_atr) == 0:
        print("  ERROR: No valid ATR ratios in overlap region")
        return None

    # Basic statistics
    stats = {
        'count': len(valid_atr),
        'nan_count': int(np.sum(~valid_mask)),
        'mean': round(float(np.mean(valid_atr)), 4),
        'median': round(float(np.median(valid_atr)), 4),
        'std': round(float(np.std(valid_atr)), 4),
        'min': round(float(np.min(valid_atr)), 4),
        'max': round(float(np.max(valid_atr)), 4),
        'p5': round(float(np.percentile(valid_atr, 5)), 4),
        'p10': round(float(np.percentile(valid_atr, 10)), 4),
        'p25': round(float(np.percentile(valid_atr, 25)), 4),
        'p75': round(float(np.percentile(valid_atr, 75)), 4),
        'p90': round(float(np.percentile(valid_atr, 90)), 4),
        'p95': round(float(np.percentile(valid_atr, 95)), 4),
    }

    # Clamped statistics (what production actually uses)
    clamped = np.clip(valid_atr, CLAMP_LO, CLAMP_HI)
    clamped_stats = {
        'mean': round(float(np.mean(clamped)), 4),
        'median': round(float(np.median(clamped)), 4),
        'std': round(float(np.std(clamped)), 4),
        'min': round(float(np.min(clamped)), 4),
        'max': round(float(np.max(clamped)), 4),
        'pct_at_lo_clamp': round(float(np.mean(valid_atr <= CLAMP_LO)) * 100, 1),
        'pct_at_hi_clamp': round(float(np.mean(valid_atr >= CLAMP_HI)) * 100, 1),
        'pct_within_clamps': round(float(np.mean((valid_atr > CLAMP_LO) & (valid_atr < CLAMP_HI))) * 100, 1),
    }

    # Time-based analysis: split into thirds
    third = len(overlap_atr) // 3
    periods = [
        ('first_third', overlap_atr[:third]),
        ('second_third', overlap_atr[third:2 * third]),
        ('last_third', overlap_atr[2 * third:]),
    ]
    period_stats = {}
    for name, data in periods:
        v = data[~np.isnan(data)]
        if len(v) > 0:
            period_stats[name] = {
                'mean': round(float(np.mean(v)), 4),
                'median': round(float(np.median(v)), 4),
                'std': round(float(np.std(v)), 4),
                'bars': len(v),
                'days': round(len(v) / BARS_PER_DAY, 1),
            }

    # Histogram buckets
    buckets = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.5, 1.7, 2.0, 99.0]
    histogram = []
    for i in range(len(buckets) - 1):
        count = int(np.sum((valid_atr >= buckets[i]) & (valid_atr < buckets[i + 1])))
        pct = round(count / len(valid_atr) * 100, 1)
        histogram.append({
            'range': f"{buckets[i]:.1f}-{buckets[i+1]:.1f}",
            'count': count,
            'pct': pct,
        })

    # TP/SL scaling impact analysis
    # For a reference pattern with TP=2.0%, SL=3.0%
    ref_tp, ref_sl = 2.0, 3.0
    scaled_tps = ref_tp * clamped
    scaled_sls = ref_sl * clamped

    scaling_impact = {
        'ref_tp': ref_tp,
        'ref_sl': ref_sl,
        'scaled_tp_mean': round(float(np.mean(scaled_tps)), 3),
        'scaled_tp_min': round(float(np.min(scaled_tps)), 3),
        'scaled_tp_max': round(float(np.max(scaled_tps)), 3),
        'scaled_sl_mean': round(float(np.mean(scaled_sls)), 3),
        'scaled_sl_min': round(float(np.min(scaled_sls)), 3),
        'scaled_sl_max': round(float(np.max(scaled_sls)), 3),
        'tp_compression': round((1 - np.mean(scaled_tps) / ref_tp) * 100, 1),
        'sl_compression': round((1 - np.mean(scaled_sls) / ref_sl) * 100, 1),
    }

    # Print results
    print(f"\n  Raw ATR Ratio Statistics (270d overlap):")
    print(f"    Count:   {stats['count']} valid bars ({stats['nan_count']} NaN)")
    print(f"    Mean:    {stats['mean']:.4f}")
    print(f"    Median:  {stats['median']:.4f}")
    print(f"    Std:     {stats['std']:.4f}")
    print(f"    Range:   [{stats['min']:.4f}, {stats['max']:.4f}]")
    print(f"    P5/P95:  [{stats['p5']:.4f}, {stats['p95']:.4f}]")
    print(f"    P10/P90: [{stats['p10']:.4f}, {stats['p90']:.4f}]")
    print(f"    P25/P75: [{stats['p25']:.4f}, {stats['p75']:.4f}]")

    print(f"\n  Clamped [{CLAMP_LO}, {CLAMP_HI}] Statistics:")
    print(f"    Mean:    {clamped_stats['mean']:.4f}")
    print(f"    Median:  {clamped_stats['median']:.4f}")
    print(f"    At lo clamp ({CLAMP_LO}): {clamped_stats['pct_at_lo_clamp']:.1f}%")
    print(f"    At hi clamp ({CLAMP_HI}): {clamped_stats['pct_at_hi_clamp']:.1f}%")
    print(f"    Within clamps:    {clamped_stats['pct_within_clamps']:.1f}%")

    print(f"\n  Time Period Analysis:")
    for name, ps in period_stats.items():
        print(f"    {name:15s}: mean={ps['mean']:.4f}, median={ps['median']:.4f}, "
              f"std={ps['std']:.4f} ({ps['days']:.0f}d)")

    print(f"\n  ATR Ratio Histogram:")
    print(f"    {'Range':>10} | {'Count':>7} | {'Pct':>6} | {'Bar':>30}")
    print(f"    {'-'*10}-+-{'-'*7}-+-{'-'*6}-+-{'-'*30}")
    for h in histogram:
        bar = '#' * int(h['pct'] / 2)
        print(f"    {h['range']:>10} | {h['count']:7d} | {h['pct']:5.1f}% | {bar}")

    print(f"\n  TP/SL Scaling Impact (ref TP={ref_tp}%, SL={ref_sl}%):")
    print(f"    Scaled TP: mean {scaling_impact['scaled_tp_mean']:.3f}%, "
          f"range [{scaling_impact['scaled_tp_min']:.3f}, {scaling_impact['scaled_tp_max']:.3f}]")
    print(f"    Scaled SL: mean {scaling_impact['scaled_sl_mean']:.3f}%, "
          f"range [{scaling_impact['scaled_sl_min']:.3f}, {scaling_impact['scaled_sl_max']:.3f}]")
    print(f"    TP compression from nominal: {scaling_impact['tp_compression']:+.1f}%")
    print(f"    SL compression from nominal: {scaling_impact['sl_compression']:+.1f}%")

    if abs(clamped_stats['mean'] - 1.0) < 0.05:
        print(f"\n    INTERPRETATION: Mean ATR ratio ~1.0 — ATR scaling is nearly neutral on average")
    elif clamped_stats['mean'] < 1.0:
        print(f"\n    INTERPRETATION: Mean ATR ratio < 1.0 — ATR scaling COMPRESSES TP/SL "
              f"(low vol environment, tighter stops)")
    else:
        print(f"\n    INTERPRETATION: Mean ATR ratio > 1.0 — ATR scaling EXPANDS TP/SL "
              f"(high vol environment, wider stops)")

    return {
        'raw_stats': stats,
        'clamped_stats': clamped_stats,
        'period_stats': period_stats,
        'histogram': histogram,
        'scaling_impact': scaling_impact,
    }


# ===================================================================
# MAIN
# ===================================================================

def main():
    t0 = time.time()

    print("=" * 120)
    print("  ATR-SCALED BACKTEST STUDY")
    print("  Scanner vs Production Condition Gap Analysis")
    print("=" * 120)

    # ------------------------------------------------------------------
    # Load 720d data
    # ------------------------------------------------------------------
    print("\n--- Loading 720d data ---")
    df_720 = load_and_classify(DATA_720D)
    n_bars = len(df_720)
    opens = df_720['open'].values.astype(np.float64)
    highs = df_720['high'].values.astype(np.float64)
    lows = df_720['low'].values.astype(np.float64)
    closes = df_720['close'].values.astype(np.float64)
    types = df_720['rctype'].values

    overlap_bar = find_overlap_bar(df_720)
    avail_bars = n_bars - overlap_bar
    avail_days = avail_bars / BARS_PER_DAY
    logger.info(f"720d data: {n_bars} bars, overlap bar: {overlap_bar}, "
                f"avail: {avail_bars} bars ({avail_days:.1f} days)")

    # ------------------------------------------------------------------
    # Load patterns and build indices
    # ------------------------------------------------------------------
    patterns = load_patterns()
    logger.info(f"Loaded {len(patterns)} patterns from {PATTERNS_FILE}")

    sig_index = build_signal_index(types, n_bars)
    atr_ratio = compute_atr_ratio(highs, lows, closes, ATR_PERIOD, ATR_WINDOW)

    # ------------------------------------------------------------------
    # Output structure
    # ------------------------------------------------------------------
    output = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'study': 'ATR-Scaled Backtest Study',
        'data_bars': n_bars,
        'overlap_bar': overlap_bar,
        'overlap_days': round(avail_days, 1),
        'n_patterns': len(patterns),
        'parameters': {
            'leverage': LEVERAGE,
            'fee_pct': FEE_PCT,
            'fee': FEE,
            'max_bars': MAX_BARS,
            'slippage_buffer': SLIPPAGE_BUFFER,
            'atr_period': ATR_PERIOD,
            'atr_window': ATR_WINDOW,
            'clamp_lo': CLAMP_LO,
            'clamp_hi': CLAMP_HI,
            'timeout_t864': TIMEOUT_T864,
            'max_positions': MAX_POSITIONS,
            'mc_sims': MC_SIMS,
            'mc_seeds': MC_SEEDS,
            'quality_filters': {
                'edge_threshold': EDGE_THRESHOLD,
                'wr_threshold': WR_THRESHOLD,
                'sl_min': SL_MIN,
                'mc_p_threshold': MC_P_THRESHOLD,
                'min_trades': MIN_TRADES,
                'wr_excess_threshold': WR_EXCESS_THRESHOLD,
            },
        },
    }

    # ------------------------------------------------------------------
    # PHASE 1: Individual Pattern Comparison
    # ------------------------------------------------------------------
    p1_results = phase1_individual_comparison(
        patterns, sig_index, opens, highs, lows,
        n_bars, atr_ratio, overlap_bar)
    output['phase1_individual'] = p1_results

    # ------------------------------------------------------------------
    # PHASE 2: Pattern Selection Impact
    # ------------------------------------------------------------------
    p2_results = phase2_pattern_selection(
        sig_index, opens, highs, lows, n_bars,
        atr_ratio, overlap_bar, patterns)
    output['phase2_selection'] = p2_results

    # ------------------------------------------------------------------
    # PHASE 3: Portfolio Comparison
    # ------------------------------------------------------------------
    p3_results = phase3_portfolio_comparison(
        patterns, sig_index, opens, highs, lows,
        n_bars, atr_ratio, overlap_bar)
    output['phase3_portfolio'] = p3_results

    # ------------------------------------------------------------------
    # PHASE 4: ATR Distribution
    # ------------------------------------------------------------------
    p4_results = phase4_atr_distribution(atr_ratio, overlap_bar, n_bars)
    output['phase4_atr_distribution'] = p4_results

    # ------------------------------------------------------------------
    # FINAL SUMMARY
    # ------------------------------------------------------------------
    print("\n" + "=" * 120)
    print("  FINAL SUMMARY")
    print("=" * 120)

    # Phase 1
    if p1_results:
        avg_wr_d = np.mean([r['wr_delta'] for r in p1_results])
        avg_edge_d = np.mean([r['edge_delta'] for r in p1_results])
        avg_pnl_d = np.mean([r['pnl_delta'] for r in p1_results])
        print(f"\n  Phase 1 — Individual Pattern (Fixed vs ATR):")
        print(f"    Avg WR delta:   {avg_wr_d:+.1f}pp")
        print(f"    Avg edge delta: {avg_edge_d:+.3f}%/trade")
        print(f"    Avg PnL delta:  {avg_pnl_d:+.1f}%")
        direction = "ATR better" if avg_edge_d > 0 else "Fixed better" if avg_edge_d < 0 else "Equal"
        print(f"    Direction:      {direction}")

    # Phase 2
    if p2_results:
        print(f"\n  Phase 2 — Pattern Selection (59-pat re-evaluation):")
        print(f"    Fixed passed: {p2_results['fixed_passed']}")
        print(f"    ATR passed:   {p2_results['atr_passed']}")
        print(f"    Overlap:      {p2_results['overlap_fixed_atr']}")
        if p2_results['only_atr']:
            print(f"    ATR-only patterns (would be ADDED):   {p2_results['only_atr']}")
        if p2_results['only_fixed']:
            print(f"    Fixed-only patterns (would be LOST):  {p2_results['only_fixed']}")
        selection_impact = "DIFFERENT" if p2_results['only_atr'] or p2_results['only_fixed'] else "SAME"
        print(f"    Selection impact: {selection_impact}")

    # Phase 3
    if p3_results:
        ft = p3_results.get('fixed_T864', {})
        at = p3_results.get('atr_T864', {})
        print(f"\n  Phase 3 — Portfolio (Hedge N=5, T864):")
        print(f"    Fixed:  PnL {ft.get('cmp_pnl', 0):+.1f}%, MDD {ft.get('cmp_mdd', 0):.1f}%, "
              f"PnL/MDD {ft.get('pnl_mdd_ratio', 0):.2f}, WF {ft.get('wf_verdict', 'N/A')}")
        print(f"    ATR:    PnL {at.get('cmp_pnl', 0):+.1f}%, MDD {at.get('cmp_mdd', 0):.1f}%, "
              f"PnL/MDD {at.get('pnl_mdd_ratio', 0):.2f}, WF {at.get('wf_verdict', 'N/A')}")

    # Phase 4
    if p4_results:
        cs = p4_results['clamped_stats']
        print(f"\n  Phase 4 — ATR Distribution:")
        print(f"    Mean clamped ratio: {cs['mean']:.4f}")
        print(f"    At lo clamp: {cs['pct_at_lo_clamp']:.1f}%, at hi clamp: {cs['pct_at_hi_clamp']:.1f}%")
        print(f"    Within clamps: {cs['pct_within_clamps']:.1f}%")

    # Overall verdict
    print(f"\n  --- OVERALL VERDICT ---")
    if p1_results and p3_results:
        avg_wr_d = np.mean([r['wr_delta'] for r in p1_results])
        ft = p3_results.get('fixed_T864', {})
        at = p3_results.get('atr_T864', {})
        ratio_diff = at.get('pnl_mdd_ratio', 0) - ft.get('pnl_mdd_ratio', 0)

        if abs(avg_wr_d) < 2.0 and abs(ratio_diff) < 1.0:
            print(f"    ATR scaling has MINIMAL impact on pattern-level and portfolio-level performance.")
            print(f"    The scanner-to-production gap is NOT a significant concern.")
        elif ratio_diff > 1.0:
            print(f"    ATR scaling SIGNIFICANTLY IMPROVES performance.")
            print(f"    Consider integrating ATR scaling into the scanner.")
        elif ratio_diff < -1.0:
            print(f"    ATR scaling SIGNIFICANTLY REDUCES performance.")
            print(f"    Review ATR scaling parameters or consider disabling.")
        else:
            print(f"    ATR scaling has MODERATE impact. Further investigation recommended.")

    output['summary'] = {
        'phase1_avg_wr_delta': round(float(np.mean([r['wr_delta'] for r in p1_results])), 1) if p1_results else None,
        'phase1_avg_edge_delta': round(float(np.mean([r['edge_delta'] for r in p1_results])), 3) if p1_results else None,
        'phase2_fixed_passed': p2_results['fixed_passed'] if p2_results else None,
        'phase2_atr_passed': p2_results['atr_passed'] if p2_results else None,
        'phase2_selection_changed': bool(p2_results and (p2_results['only_atr'] or p2_results['only_fixed'])),
        'phase3_fixed_t864_pnl': p3_results.get('fixed_T864', {}).get('cmp_pnl') if p3_results else None,
        'phase3_atr_t864_pnl': p3_results.get('atr_T864', {}).get('cmp_pnl') if p3_results else None,
        'phase4_mean_atr_ratio': p4_results['clamped_stats']['mean'] if p4_results else None,
    }

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    elapsed = time.time() - t0
    print(f"\n  Elapsed: {elapsed:.1f}s")
    print(f"  Results saved to {OUTPUT_FILE}")
    print("=" * 120)


if __name__ == '__main__':
    main()
