#!/usr/bin/env python3
"""
TP/SL Regime Bias Study — Per-Pattern Directional Edge Decomposition
=====================================================================

Core Question:
  For each of the 15 WR-Excess-filtered patterns, does the current TP/SL
  setting produce wins due to market regime (bear bias) or genuine
  pattern-based directional edge?

Key Insight:
  Random Walk WR = SL/(TP+SL).  With TP<SL (R:R<1), this exceeds 50%.
  A pattern's ACTUAL WR must exceed this Random Walk WR for the pattern
  to have genuine predictive power.  WR Excess = Actual WR - Random WR.

Five Phases:
  1. Current TP/SL Decomposition — per-pattern actual vs random WR, regime WR
  2. Direction Flip Test — if edge is directional, flipping should destroy it
  3. Symmetric TP/SL Test — TP=SL at multiple levels, random WR is always 50%
  4. WR Excess Maximizing Grid — find TP/SL that maximizes genuine edge
  5. Regime-Robust Portfolio — portfolio of Phase 4 optimal + WF validation

Standard Research Protocol:
  - Entry: next bar open after signal
  - Exit: intrabar high/low (distance-based, same-bar resolution from bar OPEN)
  - Sizing: compound (1/N per slot)
  - Fee: FEE_PCT * LEVERAGE = 0.30% capital-space
  - Timeout: MAX_BARS = 288 (24h) -> DROP from PnL
  - WF: 3-fold expanding window within 270d overlap of 720d data
  - classify_candle: production import (NEVER self-implement)

Data:
  - Primary: btc_5m_270days_reclassified.csv (270d ground truth)
  - WF: btc_5m_720days_binance.csv (720d with overlap region)

Patterns:
  - Current 15 patterns from results/dynamic_patterns.json
  - Comparison: 59-pattern backup from results/dynamic_patterns_59pat_backup.json
"""

import os
import sys
import json
import time
import logging

from multiprocessing import Pool, cpu_count

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
from scripts.analysis.fifo_vs_hedge_backtest_study import (
    build_signal_schedule, ActivePosition, pnl_mdd_ratio,
)

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger('tpsl_regime_bias')

# ---------------------------------------------------------------------------
# File paths
# ---------------------------------------------------------------------------
DATA_270D = os.path.join(_PROJECT_ROOT, 'data', 'btc_5m_270days_reclassified.csv')
DATA_720D = os.path.join(_PROJECT_ROOT, 'data', 'btc_5m_720days_binance.csv')
PATTERNS_FILE = os.path.join(_PROJECT_ROOT, 'results', 'dynamic_patterns.json')
PATTERNS_59_BACKUP = os.path.join(_PROJECT_ROOT, 'results', 'dynamic_patterns_59pat_backup.json')
OUTPUT_FILE = os.path.join(_PROJECT_ROOT, 'results', 'tpsl_regime_bias_study.json')

# ---------------------------------------------------------------------------
# Regime parameters
# ---------------------------------------------------------------------------
REGIME_WINDOW = 240      # 240 bars = 20 hours
REGIME_BULL_THRESH = 1.0  # > +1% = BULL
REGIME_BEAR_THRESH = -1.0 # < -1% = BEAR

# Phase 3: Symmetric TP/SL levels
SYMMETRIC_LEVELS = [1.0, 1.5, 2.0, 2.5, 3.0]

# Phase 4: Grid search dimensions
TP_GRID = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 3.5]
SL_GRID = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 3.5, 4.0]

# Phase 5: Timeout for portfolio
TIMEOUT_T864 = 864  # 3 days


# ===================================================================
# UTILITY: Per-trade backtest (single pattern, arbitrary TP/SL)
# ===================================================================

def backtest_pattern_trades(sig_bars, direction_int, tp_pct, sl_pct,
                            opens, highs, lows, n_bars,
                            atr_ratio=None, use_atr=False,
                            lo_bar=0, hi_bar=None,
                            timeout_bars=None):
    """
    Backtest a pattern's signals with given TP/SL.

    Returns list of dicts: {sig_bar, entry_bar, exit_bar, pnl, reason, entry_price}
    Timeout trades are EXCLUDED from PnL (DROP protocol).
    """
    if hi_bar is None:
        hi_bar = n_bars
    is_long = (direction_int == 1)
    results = []

    for sig_bar in sig_bars:
        if sig_bar < lo_bar or sig_bar >= hi_bar:
            continue
        entry_bar = sig_bar + 1
        if entry_bar >= n_bars:
            continue
        entry_price = opens[entry_bar]
        if entry_price <= 0:
            continue

        # ATR scaling
        if use_atr and atr_ratio is not None and sig_bar < len(atr_ratio) and not np.isnan(atr_ratio[sig_bar]):
            r = max(CLAMP_LO, min(CLAMP_HI, atr_ratio[sig_bar]))
        else:
            r = 1.0

        eff_tp = tp_pct * r + SLIPPAGE_BUFFER
        eff_sl = max(0.1, sl_pct * r - SLIPPAGE_BUFFER)

        if is_long:
            tp_price = entry_price * (1 + eff_tp / 100)
            sl_price = entry_price * (1 - eff_sl / 100)
        else:
            tp_price = entry_price * (1 - eff_tp / 100)
            sl_price = entry_price * (1 + eff_sl / 100)

        max_exit = min(entry_bar + 1 + (timeout_bars if timeout_bars else MAX_BARS), n_bars)

        resolved = False
        for j in range(entry_bar + 1, max_exit):
            h, l, o = highs[j], lows[j], opens[j]
            if is_long:
                tp_hit = h >= tp_price
                sl_hit = l <= sl_price
            else:
                tp_hit = l <= tp_price
                sl_hit = h >= sl_price

            if tp_hit and sl_hit:
                # Same-bar resolution: distance from bar OPEN
                tp_dist = abs(tp_price - o)
                sl_dist = abs(sl_price - o)
                if tp_dist <= sl_dist:
                    tp_hit, sl_hit = True, False
                else:
                    tp_hit, sl_hit = False, True

            if tp_hit:
                pnl = eff_tp * LEVERAGE - FEE
                results.append({
                    'sig_bar': sig_bar, 'entry_bar': entry_bar, 'exit_bar': j,
                    'pnl': pnl, 'reason': 'TP', 'entry_price': entry_price,
                })
                resolved = True
                break
            elif sl_hit:
                pnl = -eff_sl * LEVERAGE - FEE
                results.append({
                    'sig_bar': sig_bar, 'entry_bar': entry_bar, 'exit_bar': j,
                    'pnl': pnl, 'reason': 'SL', 'entry_price': entry_price,
                })
                resolved = True
                break

        if not resolved:
            # TIMEOUT -> DROP (exclude from PnL per protocol)
            pass

    return results


def compute_wr_and_edge(trades):
    """Compute WR and per-trade edge from trade list."""
    if not trades:
        return 0.0, 0.0, 0
    wins = sum(1 for t in trades if t['pnl'] > 0)
    total = len(trades)
    wr = wins / total * 100
    edge = sum(t['pnl'] for t in trades) / total
    return wr, edge, total


# ===================================================================
# REGIME CLASSIFICATION
# ===================================================================

def classify_regime(closes, window=REGIME_WINDOW,
                    bull_thresh=REGIME_BULL_THRESH,
                    bear_thresh=REGIME_BEAR_THRESH):
    """
    Classify each bar's regime based on rolling return.

    Returns numpy array of regime labels: 'BULL', 'BEAR', 'NEUTRAL'
    """
    n = len(closes)
    regimes = np.empty(n, dtype='U7')
    regimes[:] = 'NEUTRAL'

    for i in range(window, n):
        ret = (closes[i] / closes[i - window] - 1) * 100
        if ret > bull_thresh:
            regimes[i] = 'BULL'
        elif ret < bear_thresh:
            regimes[i] = 'BEAR'
        else:
            regimes[i] = 'NEUTRAL'

    return regimes


# ===================================================================
# PHASE 1: Current TP/SL Decomposition
# ===================================================================

def phase1_decomposition(patterns, sig_index, opens, highs, lows, closes,
                         n_bars, atr_ratio, regimes, overlap_bar):
    """
    For each pattern: actual WR, random WR, WR excess, and regime-split WR.
    """
    print("\n" + "=" * 110)
    print("  PHASE 1: CURRENT TP/SL DECOMPOSITION (per-pattern)")
    print("  Random Walk WR = SL/(TP+SL) -- expected WR under symmetric random walk")
    print("  Regime = rolling 240-bar return: >+1% BULL, <-1% BEAR, else NEUTRAL")
    print("=" * 110)

    results = []

    for p in patterns:
        bars = sig_index.get(p['pattern'])
        if not bars:
            continue

        direction_int = 1 if p['direction'] == 'LONG' else -1
        tp, sl = p['tp'], p['sl']
        rr = tp / sl if sl > 0 else 999.0
        random_wr = sl / (tp + sl) * 100

        # Full backtest (overlap region, no timeout -> DROP)
        trades = backtest_pattern_trades(
            bars, direction_int, tp, sl,
            opens, highs, lows, n_bars,
            atr_ratio=atr_ratio, use_atr=True,
            lo_bar=overlap_bar, timeout_bars=MAX_BARS)

        actual_wr, edge, total = compute_wr_and_edge(trades)
        excess = actual_wr - random_wr

        # Regime-split WR
        regime_trades = {'BULL': [], 'BEAR': [], 'NEUTRAL': []}
        for t in trades:
            regime = regimes[t['sig_bar']] if t['sig_bar'] < len(regimes) else 'NEUTRAL'
            regime_trades[regime].append(t)

        regime_wrs = {}
        for regime_name in ['BULL', 'BEAR', 'NEUTRAL']:
            rt = regime_trades[regime_name]
            if rt:
                w = sum(1 for x in rt if x['pnl'] > 0)
                regime_wrs[regime_name] = round(w / len(rt) * 100, 1)
            else:
                regime_wrs[regime_name] = None

        results.append({
            'pattern': p['pattern'],
            'direction': p['direction'],
            'tp': tp,
            'sl': sl,
            'rr': round(rr, 3),
            'trades': total,
            'actual_wr': round(actual_wr, 1),
            'random_wr': round(random_wr, 1),
            'wr_excess': round(excess, 1),
            'edge_per_trade': round(edge, 3),
            'bull_wr': regime_wrs['BULL'],
            'bear_wr': regime_wrs['BEAR'],
            'neutral_wr': regime_wrs['NEUTRAL'],
            'bull_trades': len(regime_trades['BULL']),
            'bear_trades': len(regime_trades['BEAR']),
            'neutral_trades': len(regime_trades['NEUTRAL']),
        })

    results.sort(key=lambda x: -x['wr_excess'])

    # Print table
    print(f"\n  {'#':>2} {'Pattern':<12} {'Dir':>5} | {'TP':>5} {'SL':>5} {'R:R':>5} | "
          f"{'Tr':>4} {'ActWR':>6} {'RndWR':>6} {'Excess':>7} | "
          f"{'BullWR':>7} {'BearWR':>7} {'NeutWR':>7} | {'Edge':>7}")
    print("  " + "-" * 105)

    for i, r in enumerate(results, 1):
        bull_str = f"{r['bull_wr']:6.1f}%" if r['bull_wr'] is not None else "   N/A"
        bear_str = f"{r['bear_wr']:6.1f}%" if r['bear_wr'] is not None else "   N/A"
        neut_str = f"{r['neutral_wr']:6.1f}%" if r['neutral_wr'] is not None else "   N/A"

        print(f"  {i:2d} {r['pattern']:<12} {r['direction']:>5} | "
              f"{r['tp']:5.2f} {r['sl']:5.2f} {r['rr']:5.3f} | "
              f"{r['trades']:4d} {r['actual_wr']:5.1f}% {r['random_wr']:5.1f}% {r['wr_excess']:>+6.1f}% | "
              f"{bull_str} {bear_str} {neut_str} | "
              f"{r['edge_per_trade']:>+7.3f}")

    # Summary stats
    avg_excess = np.mean([r['wr_excess'] for r in results])
    avg_rr = np.mean([r['rr'] for r in results])
    total_trades = sum(r['trades'] for r in results)
    print(f"\n  Summary: {len(results)} patterns, {total_trades} trades, "
          f"avg R:R {avg_rr:.3f}, avg WR excess {avg_excess:+.1f}pp")

    return results


# ===================================================================
# PHASE 2: Direction Flip Test
# ===================================================================

def phase2_direction_flip(patterns, sig_index, opens, highs, lows, n_bars,
                          atr_ratio, overlap_bar):
    """
    Flip each pattern's direction. If edge is genuine, flipped should lose.
    """
    print("\n" + "=" * 110)
    print("  PHASE 2: DIRECTION FLIP TEST (per-pattern)")
    print("  Flip LONG->SHORT, SHORT->LONG, keep same TP/SL")
    print("  If flipped WR drops below Random Walk WR -> edge IS direction-specific (genuine)")
    print("  If flipped WR stays above -> TP/SL setting doing the work (regime bias)")
    print("=" * 110)

    results = []

    for p in patterns:
        bars = sig_index.get(p['pattern'])
        if not bars:
            continue

        original_dir = 1 if p['direction'] == 'LONG' else -1
        flipped_dir = -original_dir
        tp, sl = p['tp'], p['sl']
        random_wr = sl / (tp + sl) * 100

        # Original
        orig_trades = backtest_pattern_trades(
            bars, original_dir, tp, sl,
            opens, highs, lows, n_bars,
            atr_ratio=atr_ratio, use_atr=True,
            lo_bar=overlap_bar, timeout_bars=MAX_BARS)
        orig_wr, orig_edge, orig_n = compute_wr_and_edge(orig_trades)

        # Flipped
        flip_trades = backtest_pattern_trades(
            bars, flipped_dir, tp, sl,
            opens, highs, lows, n_bars,
            atr_ratio=atr_ratio, use_atr=True,
            lo_bar=overlap_bar, timeout_bars=MAX_BARS)
        flip_wr, flip_edge, flip_n = compute_wr_and_edge(flip_trades)

        # Diagnosis
        wr_drop = orig_wr - flip_wr
        flip_below_random = flip_wr < random_wr
        verdict = "GENUINE" if flip_below_random else "REGIME_SUSPECT"

        # Compute compound PnL for flipped direction
        if flip_trades:
            eq = 1.0
            for t in flip_trades:
                eq *= (1 + t['pnl'] / 100)
            flip_pnl = (eq - 1) * 100
        else:
            flip_pnl = 0.0

        results.append({
            'pattern': p['pattern'],
            'direction': p['direction'],
            'tp': tp,
            'sl': sl,
            'random_wr': round(random_wr, 1),
            'orig_trades': orig_n,
            'orig_wr': round(orig_wr, 1),
            'orig_edge': round(orig_edge, 3),
            'flip_trades': flip_n,
            'flip_wr': round(flip_wr, 1),
            'flip_edge': round(flip_edge, 3),
            'flip_pnl': round(flip_pnl, 1),
            'wr_drop': round(wr_drop, 1),
            'flip_below_random': flip_below_random,
            'verdict': verdict,
        })

    results.sort(key=lambda x: -x['wr_drop'])

    # Print table
    print(f"\n  {'#':>2} {'Pattern':<12} {'Dir':>5} | {'RndWR':>6} | "
          f"{'OrigTr':>6} {'OrigWR':>7} {'OrigEdge':>9} | "
          f"{'FlipTr':>6} {'FlipWR':>7} {'FlipEdge':>9} {'FlipPnL':>8} | "
          f"{'WRDrop':>7} {'Verdict':>15}")
    print("  " + "-" * 120)

    genuine_count = 0
    for i, r in enumerate(results, 1):
        if r['verdict'] == 'GENUINE':
            genuine_count += 1
        marker = " ***" if r['verdict'] == 'GENUINE' else ""
        print(f"  {i:2d} {r['pattern']:<12} {r['direction']:>5} | {r['random_wr']:5.1f}% | "
              f"{r['orig_trades']:6d} {r['orig_wr']:6.1f}% {r['orig_edge']:>+9.3f} | "
              f"{r['flip_trades']:6d} {r['flip_wr']:6.1f}% {r['flip_edge']:>+9.3f} {r['flip_pnl']:>+7.1f}% | "
              f"{r['wr_drop']:>+6.1f}% {r['verdict']:>15}{marker}")

    print(f"\n  Genuine direction edge: {genuine_count}/{len(results)} patterns")
    print(f"  Regime-suspect: {len(results) - genuine_count}/{len(results)} patterns")

    return results


# ===================================================================
# PHASE 3: Symmetric TP/SL Test
# ===================================================================

def phase3_symmetric_test(patterns, sig_index, opens, highs, lows, n_bars,
                          atr_ratio, overlap_bar):
    """
    Set TP = SL at multiple levels. Random Walk WR at TP=SL is always 50%.
    If actual WR > 50% -> genuine directional edge independent of R:R.
    """
    print("\n" + "=" * 110)
    print("  PHASE 3: SYMMETRIC TP/SL TEST (TP = SL)")
    print("  Random Walk WR for TP=SL is always 50%")
    print("  Actual WR > 50% -> genuine directional prediction power")
    print(f"  Levels tested: {SYMMETRIC_LEVELS}%")
    print("=" * 110)

    results = []

    for p in patterns:
        bars = sig_index.get(p['pattern'])
        if not bars:
            continue

        direction_int = 1 if p['direction'] == 'LONG' else -1
        pattern_results = {
            'pattern': p['pattern'],
            'direction': p['direction'],
            'original_tp': p['tp'],
            'original_sl': p['sl'],
            'symmetric_tests': [],
        }

        for level in SYMMETRIC_LEVELS:
            trades = backtest_pattern_trades(
                bars, direction_int, level, level,
                opens, highs, lows, n_bars,
                atr_ratio=atr_ratio, use_atr=True,
                lo_bar=overlap_bar, timeout_bars=MAX_BARS)

            wr, edge, total = compute_wr_and_edge(trades)

            pattern_results['symmetric_tests'].append({
                'level': level,
                'trades': total,
                'wr': round(wr, 1),
                'edge': round(edge, 3),
                'above_50': wr > 50.0,
            })

        # Average WR across all symmetric levels
        wrs = [t['wr'] for t in pattern_results['symmetric_tests'] if t['trades'] > 0]
        pattern_results['avg_symmetric_wr'] = round(np.mean(wrs), 1) if wrs else 0.0
        pattern_results['all_above_50'] = all(t['above_50'] for t in pattern_results['symmetric_tests'] if t['trades'] > 0)
        pattern_results['count_above_50'] = sum(1 for t in pattern_results['symmetric_tests'] if t['above_50'] and t['trades'] > 0)
        pattern_results['total_levels_tested'] = sum(1 for t in pattern_results['symmetric_tests'] if t['trades'] > 0)

        results.append(pattern_results)

    results.sort(key=lambda x: -x['avg_symmetric_wr'])

    # Print header
    level_headers = "  ".join(f"{l:5.1f}%" for l in SYMMETRIC_LEVELS)
    print(f"\n  {'#':>2} {'Pattern':<12} {'Dir':>5} | {level_headers} | {'AvgWR':>6} {'#>50':>4}")
    print("  " + "-" * (42 + len(SYMMETRIC_LEVELS) * 8))

    for i, r in enumerate(results, 1):
        level_strs = []
        for t in r['symmetric_tests']:
            if t['trades'] == 0:
                level_strs.append("   N/A")
            else:
                marker = "+" if t['above_50'] else " "
                level_strs.append(f"{marker}{t['wr']:5.1f}")
        level_line = "  ".join(level_strs)

        verdict = " ***" if r['all_above_50'] else ""
        print(f"  {i:2d} {r['pattern']:<12} {r['direction']:>5} | {level_line} | "
              f"{r['avg_symmetric_wr']:5.1f}% {r['count_above_50']:3d}/{r['total_levels_tested']}{verdict}")

    genuine = sum(1 for r in results if r['all_above_50'])
    partial = sum(1 for r in results if r['count_above_50'] > 0 and not r['all_above_50'])
    none_ = sum(1 for r in results if r['count_above_50'] == 0)

    print(f"\n  All levels above 50%: {genuine}/{len(results)} (strong directional edge)")
    print(f"  Partial above 50%:   {partial}/{len(results)} (conditional edge)")
    print(f"  Never above 50%:     {none_}/{len(results)} (no directional edge at symmetric TP/SL)")

    return results


# ===================================================================
# PHASE 4: WR Excess Maximizing Grid Search
# ===================================================================

def phase4_grid_search(patterns, sig_index, opens, highs, lows, n_bars,
                       atr_ratio, overlap_bar):
    """
    Grid search TP x SL for each pattern, find the setting that maximizes WR Excess.

    Runs sequentially -- benchmarked at ~1500 backtests in 3s for 15 patterns,
    far faster than multiprocessing overhead of pickling 207k-element arrays.
    """
    # Compute effective SL grid (skip < 1.0% per execution risk rule)
    sl_grid_eff = [s for s in SL_GRID if s >= 1.0]
    combos_per_pat = len(TP_GRID) * len(sl_grid_eff)

    print("\n" + "=" * 110)
    print("  PHASE 4: WR EXCESS MAXIMIZING TP/SL GRID SEARCH")
    print(f"  TP grid: {TP_GRID}")
    print(f"  SL grid (>=1.0%): {sl_grid_eff}")
    print(f"  Combos per pattern: {combos_per_pat}")
    print(f"  Total combos: {combos_per_pat * len(patterns)}")
    print("=" * 110)

    t0 = time.time()
    all_results = []
    current_excess_map = {}  # pattern -> current TP/SL excess
    total_work = 0

    for p in patterns:
        bars = sig_index.get(p['pattern'])
        if not bars:
            continue

        direction_int = 1 if p['direction'] == 'LONG' else -1

        # First: compute current TP/SL excess (may not be on grid)
        cur_random_wr = p['sl'] / (p['tp'] + p['sl']) * 100
        cur_trades = backtest_pattern_trades(
            bars, direction_int, p['tp'], p['sl'],
            opens, highs, lows, n_bars,
            atr_ratio=atr_ratio, use_atr=True,
            lo_bar=overlap_bar, timeout_bars=MAX_BARS)
        cur_wr, cur_edge, cur_n = compute_wr_and_edge(cur_trades)
        current_excess_map[p['pattern']] = round(cur_wr - cur_random_wr, 1)

        # Then: grid search
        for tp_test in TP_GRID:
            for sl_test in sl_grid_eff:
                random_wr = sl_test / (tp_test + sl_test) * 100

                trades = backtest_pattern_trades(
                    bars, direction_int, tp_test, sl_test,
                    opens, highs, lows, n_bars,
                    atr_ratio=atr_ratio, use_atr=True,
                    lo_bar=overlap_bar, timeout_bars=MAX_BARS)

                actual_wr, edge, total = compute_wr_and_edge(trades)
                wr_excess = actual_wr - random_wr

                all_results.append({
                    'pattern': p['pattern'],
                    'direction': p['direction'],
                    'tp_test': tp_test,
                    'sl_test': sl_test,
                    'rr_test': round(tp_test / sl_test, 3) if sl_test > 0 else 999,
                    'trades': total,
                    'actual_wr': round(actual_wr, 1),
                    'random_wr': round(random_wr, 1),
                    'wr_excess': round(wr_excess, 1),
                    'edge_per_trade': round(edge, 3),
                    'base_tp': p['tp'],
                    'base_sl': p['sl'],
                })
                total_work += 1

    elapsed = time.time() - t0
    print(f"  {total_work} backtests completed in {elapsed:.1f}s ({total_work/max(elapsed,0.001):.0f}/sec)")

    # Organize results by pattern
    pattern_grid = {}
    for r in all_results:
        pat = r['pattern']
        if pat not in pattern_grid:
            pattern_grid[pat] = []
        pattern_grid[pat].append(r)

    # Find optimal TP/SL for each pattern (maximize WR Excess with min_trades filter)
    optimal_results = []
    for p in patterns:
        grid = pattern_grid.get(p['pattern'], [])
        if not grid:
            continue

        # Filter: min 10 trades for grid search (lower than production since this is exploration)
        valid = [r for r in grid if r['trades'] >= 10]
        if not valid:
            # Fall back to current TP/SL
            optimal_results.append({
                'pattern': p['pattern'],
                'direction': p['direction'],
                'current_tp': p['tp'],
                'current_sl': p['sl'],
                'current_rr': round(p['tp'] / p['sl'], 3) if p['sl'] > 0 else 999,
                'optimal_tp': p['tp'],
                'optimal_sl': p['sl'],
                'optimal_rr': round(p['tp'] / p['sl'], 3) if p['sl'] > 0 else 999,
                'optimal_excess': 0.0,
                'optimal_edge': 0.0,
                'optimal_trades': 0,
                'optimal_wr': 0.0,
                'current_excess': 0.0,
                'improvement': 0.0,
                'grid_combos_tested': len(grid),
                'top5_grid': [],
            })
            continue

        # Sort by WR excess, then by edge as tiebreaker
        valid.sort(key=lambda x: (-x['wr_excess'], -x['edge_per_trade']))
        best = valid[0]

        # Use pre-computed current TP/SL excess (not from grid, since current TP/SL may not be on grid)
        current_excess = current_excess_map.get(p['pattern'], 0.0)

        # Top 5 grid results
        top5 = valid[:5]

        optimal_results.append({
            'pattern': p['pattern'],
            'direction': p['direction'],
            'current_tp': p['tp'],
            'current_sl': p['sl'],
            'current_rr': round(p['tp'] / p['sl'], 3) if p['sl'] > 0 else 999,
            'optimal_tp': best['tp_test'],
            'optimal_sl': best['sl_test'],
            'optimal_rr': best['rr_test'],
            'optimal_excess': best['wr_excess'],
            'optimal_edge': best['edge_per_trade'],
            'optimal_trades': best['trades'],
            'optimal_wr': best['actual_wr'],
            'current_excess': round(current_excess, 1),
            'improvement': round(best['wr_excess'] - current_excess, 1),
            'grid_combos_tested': len(valid),
            'top5_grid': [{
                'tp': r['tp_test'], 'sl': r['sl_test'],
                'rr': r['rr_test'], 'wr': r['actual_wr'],
                'random_wr': r['random_wr'], 'excess': r['wr_excess'],
                'edge': r['edge_per_trade'], 'trades': r['trades'],
            } for r in top5],
        })

    optimal_results.sort(key=lambda x: -x['optimal_excess'])

    # Print results
    print(f"\n  {'#':>2} {'Pattern':<12} {'Dir':>5} | "
          f"{'CurTP':>5} {'CurSL':>5} {'CurR:R':>6} {'CurExc':>7} | "
          f"{'OptTP':>5} {'OptSL':>5} {'OptR:R':>6} {'OptExc':>7} {'OptWR':>6} {'OptEdg':>7} | "
          f"{'Impr':>6}")
    print("  " + "-" * 115)

    for i, r in enumerate(optimal_results, 1):
        changed = r['optimal_tp'] != r['current_tp'] or r['optimal_sl'] != r['current_sl']
        marker = " *" if changed else ""
        print(f"  {i:2d} {r['pattern']:<12} {r['direction']:>5} | "
              f"{r['current_tp']:5.2f} {r['current_sl']:5.2f} {r['current_rr']:6.3f} {r['current_excess']:>+6.1f}% | "
              f"{r['optimal_tp']:5.2f} {r['optimal_sl']:5.2f} {r['optimal_rr']:6.3f} {r['optimal_excess']:>+6.1f}% "
              f"{r['optimal_wr']:5.1f}% {r['optimal_edge']:>+7.3f} | "
              f"{r['improvement']:>+5.1f}%{marker}")

    changed_count = sum(1 for r in optimal_results
                        if r['optimal_tp'] != r['current_tp']
                        or r['optimal_sl'] != r['current_sl'])
    avg_improvement = np.mean([r['improvement'] for r in optimal_results])
    print(f"\n  Changed TP/SL: {changed_count}/{len(optimal_results)} patterns")
    print(f"  Avg WR Excess improvement: {avg_improvement:+.1f}pp")

    # Show top 5 for each pattern
    print("\n  -- Top 5 grid results per pattern --")
    for r in optimal_results:
        print(f"\n  {r['pattern']} ({r['direction']}):")
        for j, g in enumerate(r['top5_grid'], 1):
            print(f"    {j}. TP={g['tp']:.2f} SL={g['sl']:.2f} R:R={g['rr']:.3f} "
                  f"WR={g['wr']:.1f}% RndWR={g['random_wr']:.1f}% "
                  f"Excess={g['excess']:+.1f}% Edge={g['edge']:+.3f} Trades={g['trades']}")

    return optimal_results, pattern_grid


# ===================================================================
# PHASE 5: Regime-Robust Portfolio with WF
# ===================================================================

def build_custom_schedule(patterns_custom, sig_index, atr_ratio, opens, n_bars):
    """
    Build signal schedule from custom pattern list with potentially modified TP/SL.
    patterns_custom: list of dicts with pattern, direction, tp, sl
    """
    schedule = {}
    for p in patterns_custom:
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

            if atr_ratio is not None and sig_bar < len(atr_ratio) and not np.isnan(atr_ratio[sig_bar]):
                r = max(CLAMP_LO, min(CLAMP_HI, atr_ratio[sig_bar]))
            else:
                r = 1.0

            tp = base_tp * r + SLIPPAGE_BUFFER
            sl = max(0.1, base_sl * r - SLIPPAGE_BUFFER)

            if entry_bar not in schedule:
                schedule[entry_bar] = []
            schedule[entry_bar].append((p['pattern'], direction, tp, sl, entry_price, sig_bar))

    return schedule


def simulate_hedge_simple(max_positions, schedule, opens, highs, lows,
                          lo_bar, hi_bar, timeout_bars=None):
    """
    Simplified hedge simulation (direction-agnostic).
    Returns list of result dicts.
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
                    # TIMEOUT -> DROP (do not record, just remove)
                    slots.pop(sid)

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


def run_wf_3fold(max_positions, schedule, opens, highs, lows,
                 overlap_bar, n_bars, timeout_bars=None):
    """3-fold expanding window WF within overlap region."""
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
        oos_res = simulate_hedge_simple(
            max_positions, schedule, opens, highs, lows,
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


def phase5_portfolio_wf(optimal_results, patterns_original, sig_index,
                        opens, highs, lows, n_bars, atr_ratio, overlap_bar):
    """
    Build portfolio from Phase 4 optimal TP/SL and run WF validation.
    Also run current TP/SL portfolio for comparison.
    """
    print("\n" + "=" * 110)
    print("  PHASE 5: REGIME-ROBUST PORTFOLIO + WF VALIDATION")
    print("  Compare: Current TP/SL vs Phase 4 WR-Excess-Optimal TP/SL")
    print(f"  Timeout: T864 ({TIMEOUT_T864} bars = 3 days) for portfolio simulation")
    print(f"  WF: 3-fold expanding window within 270d overlap")
    print("=" * 110)

    # Build optimal pattern list
    optimal_patterns = []
    for r in optimal_results:
        optimal_patterns.append({
            'pattern': r['pattern'],
            'direction': r['direction'],
            'tp': r['optimal_tp'],
            'sl': r['optimal_sl'],
        })

    max_positions = 5
    scenarios = {}

    for label, pats, timeout in [
        ('current_NT', patterns_original, None),
        ('current_T864', patterns_original, TIMEOUT_T864),
        ('optimal_NT', optimal_patterns, None),
        ('optimal_T864', optimal_patterns, TIMEOUT_T864),
    ]:
        sched = build_custom_schedule(pats, sig_index, atr_ratio, opens, n_bars)
        results = simulate_hedge_simple(
            max_positions, sched, opens, highs, lows,
            overlap_bar, n_bars, timeout)
        portfolio = [(r['entry_bar'], r['exit_bar'], r['scaled_pnl']) for r in results]
        stats = calc_stats(portfolio)
        ratio = pnl_mdd_ratio(stats['cmp_pnl'], stats['cmp_mdd'])

        wf_folds, wf_verdict = run_wf_3fold(
            max_positions, sched, opens, highs, lows,
            overlap_bar, n_bars, timeout)

        scenarios[label] = {
            **stats,
            'pnl_mdd_ratio': ratio,
            'wf_verdict': wf_verdict,
            'wf_folds': wf_folds,
            'n_patterns': len(pats),
            'timeout': timeout,
        }

    # Print comparison
    print(f"\n  {'Scenario':>16} | {'#Pat':>4} {'Trades':>7} | {'WR':>6} {'PF':>6} | "
          f"{'CmpPnL':>9} {'CmpMDD':>8} {'PnL/MDD':>8} | {'WF':>12}")
    print("  " + "-" * 95)

    for label in ['current_NT', 'current_T864', 'optimal_NT', 'optimal_T864']:
        s = scenarios[label]
        print(f"  {label:>16} | {s['n_patterns']:4d} {s['trades']:7d} | "
              f"{s['wr']:5.1f}% {s['pf']:6.2f} | "
              f"{s['cmp_pnl']:>+9.1f} {s['cmp_mdd']:8.1f} {s['pnl_mdd_ratio']:8.2f} | "
              f"{s['wf_verdict']:>12}")

    # WF detail
    for label in ['current_T864', 'optimal_T864']:
        s = scenarios[label]
        print(f"\n  WF Detail: {label}")
        print(f"    {'Fold':>4} | {'OOS Tr':>7} {'OOS WR':>7} {'OOS PnL':>9} {'OOS MDD':>8} {'Pass':>5}")
        for f in s['wf_folds']:
            print(f"    {f['fold']:4d} | {f['oos_trades']:7d} {f['oos_wr']:6.1f}% "
                  f"{f['oos_pnl']:>+9.1f} {f['oos_mdd']:8.1f} "
                  f"{'PASS' if f['pass'] else 'FAIL':>5}")

    # Verdict
    cur_t864 = scenarios['current_T864']
    opt_t864 = scenarios['optimal_T864']
    pnl_diff = opt_t864['cmp_pnl'] - cur_t864['cmp_pnl']

    print(f"\n  --- PORTFOLIO COMPARISON VERDICT ---")
    print(f"    Current T864: PnL {cur_t864['cmp_pnl']:+.1f}%, MDD {cur_t864['cmp_mdd']:.1f}%, "
          f"WF {cur_t864['wf_verdict']}")
    print(f"    Optimal T864: PnL {opt_t864['cmp_pnl']:+.1f}%, MDD {opt_t864['cmp_mdd']:.1f}%, "
          f"WF {opt_t864['wf_verdict']}")
    print(f"    PnL difference: {pnl_diff:+.1f}%")

    if '3/3' in opt_t864['wf_verdict'] and opt_t864['cmp_pnl'] > cur_t864['cmp_pnl']:
        print(f"    RECOMMENDATION: Optimal TP/SL is better and passes WF")
    elif '3/3' not in opt_t864['wf_verdict']:
        print(f"    CAUTION: Optimal TP/SL fails WF -- likely overfit to grid search")
    else:
        print(f"    NOTE: Optimal TP/SL passes WF but does not improve PnL")

    return scenarios


# ===================================================================
# 59-PATTERN COMPARISON
# ===================================================================

def load_patterns_from_file(filepath):
    """Load patterns from a dynamic_patterns JSON file."""
    with open(filepath) as f:
        data = json.load(f)
    return [{'pattern': v['pattern'], 'direction': v['direction'],
             'tp': float(v['tp']), 'sl': float(v['sl'])}
            for v in data['pattern_details'].values()]


# ===================================================================
# MAIN
# ===================================================================

def main():
    t0 = time.time()

    print("=" * 110)
    print("  TP/SL REGIME BIAS STUDY")
    print("  Per-Pattern Directional Edge Decomposition")
    print("=" * 110)

    # ------------------------------------------------------------------
    # Load 720d data (for WF with overlap region)
    # ------------------------------------------------------------------
    print("\n--- Loading 720d data for WF validation ---")
    df_720 = load_and_classify(DATA_720D)
    n_bars_720 = len(df_720)
    opens_720 = df_720['open'].values.astype(np.float64)
    highs_720 = df_720['high'].values.astype(np.float64)
    lows_720 = df_720['low'].values.astype(np.float64)
    closes_720 = df_720['close'].values.astype(np.float64)
    types_720 = df_720['rctype'].values

    overlap_bar = find_overlap_bar(df_720)
    avail_bars = n_bars_720 - overlap_bar
    avail_days = avail_bars / BARS_PER_DAY
    logger.info(f"720d data: {n_bars_720} bars, overlap bar: {overlap_bar}, "
                f"avail: {avail_bars} bars ({avail_days:.1f} days)")

    # ------------------------------------------------------------------
    # Load patterns and build indices
    # ------------------------------------------------------------------
    patterns = load_patterns()
    logger.info(f"Loaded {len(patterns)} patterns from {PATTERNS_FILE}")

    sig_index_720 = build_signal_index(types_720, n_bars_720)
    atr_ratio_720 = compute_atr_ratio(highs_720, lows_720, closes_720, ATR_PERIOD, ATR_WINDOW)

    # Classify regimes
    regimes_720 = classify_regime(closes_720)
    regime_counts = {
        'BULL': int(np.sum(regimes_720 == 'BULL')),
        'BEAR': int(np.sum(regimes_720 == 'BEAR')),
        'NEUTRAL': int(np.sum(regimes_720 == 'NEUTRAL')),
    }
    total_regime = sum(regime_counts.values())
    logger.info(f"Regime distribution (720d): BULL={regime_counts['BULL']} ({regime_counts['BULL']/total_regime*100:.1f}%), "
                f"BEAR={regime_counts['BEAR']} ({regime_counts['BEAR']/total_regime*100:.1f}%), "
                f"NEUTRAL={regime_counts['NEUTRAL']} ({regime_counts['NEUTRAL']/total_regime*100:.1f}%)")

    # Regime distribution in overlap region
    overlap_regimes = regimes_720[overlap_bar:]
    overlap_regime_counts = {
        'BULL': int(np.sum(overlap_regimes == 'BULL')),
        'BEAR': int(np.sum(overlap_regimes == 'BEAR')),
        'NEUTRAL': int(np.sum(overlap_regimes == 'NEUTRAL')),
    }
    total_ol = sum(overlap_regime_counts.values())
    print(f"\n  Regime distribution (270d overlap):")
    for k, v in overlap_regime_counts.items():
        print(f"    {k}: {v} bars ({v/total_ol*100:.1f}%)")

    output = {
        'timestamp': pd.Timestamp.now().isoformat(),
        'study': 'TP/SL Regime Bias Study',
        'data_720d_bars': n_bars_720,
        'overlap_bar': overlap_bar,
        'overlap_days': round(avail_days, 1),
        'n_patterns': len(patterns),
        'patterns_file': os.path.basename(PATTERNS_FILE),
        'regime_distribution_overlap': overlap_regime_counts,
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
            'regime_window': REGIME_WINDOW,
            'regime_bull_thresh': REGIME_BULL_THRESH,
            'regime_bear_thresh': REGIME_BEAR_THRESH,
            'symmetric_levels': SYMMETRIC_LEVELS,
            'tp_grid': TP_GRID,
            'sl_grid': SL_GRID,
            'timeout_t864': TIMEOUT_T864,
        },
    }

    # ------------------------------------------------------------------
    # PHASE 1: Current TP/SL Decomposition
    # ------------------------------------------------------------------
    p1_results = phase1_decomposition(
        patterns, sig_index_720,
        opens_720, highs_720, lows_720, closes_720,
        n_bars_720, atr_ratio_720, regimes_720, overlap_bar)
    output['phase1_decomposition'] = p1_results

    # ------------------------------------------------------------------
    # PHASE 2: Direction Flip Test
    # ------------------------------------------------------------------
    p2_results = phase2_direction_flip(
        patterns, sig_index_720,
        opens_720, highs_720, lows_720, n_bars_720,
        atr_ratio_720, overlap_bar)
    output['phase2_direction_flip'] = p2_results

    # ------------------------------------------------------------------
    # PHASE 3: Symmetric TP/SL Test
    # ------------------------------------------------------------------
    p3_results = phase3_symmetric_test(
        patterns, sig_index_720,
        opens_720, highs_720, lows_720, n_bars_720,
        atr_ratio_720, overlap_bar)
    output['phase3_symmetric_test'] = p3_results

    # ------------------------------------------------------------------
    # PHASE 4: Grid Search (multiprocessing)
    # ------------------------------------------------------------------
    p4_optimal, p4_grid = phase4_grid_search(
        patterns, sig_index_720,
        opens_720, highs_720, lows_720, n_bars_720,
        atr_ratio_720, overlap_bar)
    output['phase4_grid_search'] = p4_optimal
    # Note: full grid too large to save; only optimal results saved

    # ------------------------------------------------------------------
    # PHASE 5: Portfolio + WF
    # ------------------------------------------------------------------
    p5_scenarios = phase5_portfolio_wf(
        p4_optimal, patterns, sig_index_720,
        opens_720, highs_720, lows_720, n_bars_720,
        atr_ratio_720, overlap_bar)
    output['phase5_portfolio_wf'] = p5_scenarios

    # ------------------------------------------------------------------
    # CROSS-STUDY: 59-pattern backup comparison
    # ------------------------------------------------------------------
    print("\n" + "=" * 110)
    print("  BONUS: 59-PATTERN BACKUP COMPARISON")
    print("=" * 110)

    if os.path.exists(PATTERNS_59_BACKUP):
        patterns_59 = load_patterns_from_file(PATTERNS_59_BACKUP)
        logger.info(f"Loaded {len(patterns_59)} patterns from 59-pat backup")

        # Simple stats: how many of the 15 patterns are in the 59-pat set?
        current_names = {p['pattern'] + '_' + p['direction'] for p in patterns}
        backup_names = {p['pattern'] + '_' + p['direction'] for p in patterns_59}
        overlap_pats = current_names & backup_names
        only_in_15 = current_names - backup_names
        only_in_59 = backup_names - current_names

        print(f"\n  Current (15 patterns): {len(current_names)}")
        print(f"  Backup (59 patterns):  {len(backup_names)}")
        print(f"  Overlap:               {len(overlap_pats)}")
        print(f"  Only in 15:            {len(only_in_15)} -> {sorted(only_in_15)}")
        print(f"  Only in 59:            {len(only_in_59)}")

        # TP/SL comparison for overlapping patterns
        if overlap_pats:
            print(f"\n  TP/SL comparison for overlapping patterns:")
            print(f"  {'Pattern+Dir':<20} | {'15p TP':>6} {'15p SL':>6} | {'59p TP':>6} {'59p SL':>6} | {'TP diff':>7} {'SL diff':>7}")
            print("  " + "-" * 80)

            pat15_map = {p['pattern'] + '_' + p['direction']: p for p in patterns}
            pat59_map = {p['pattern'] + '_' + p['direction']: p for p in patterns_59}

            for key in sorted(overlap_pats):
                p15 = pat15_map[key]
                p59 = pat59_map[key]
                tp_diff = p15['tp'] - p59['tp']
                sl_diff = p15['sl'] - p59['sl']
                print(f"  {key:<20} | {p15['tp']:6.2f} {p15['sl']:6.2f} | "
                      f"{p59['tp']:6.2f} {p59['sl']:6.2f} | "
                      f"{tp_diff:>+6.2f} {sl_diff:>+6.2f}")

        output['bonus_59pat_comparison'] = {
            'overlap_count': len(overlap_pats),
            'only_in_15': sorted(only_in_15),
            'only_in_59_count': len(only_in_59),
        }
    else:
        print(f"  59-pattern backup not found at {PATTERNS_59_BACKUP}")
        output['bonus_59pat_comparison'] = None

    # ------------------------------------------------------------------
    # FINAL SUMMARY
    # ------------------------------------------------------------------
    print("\n" + "=" * 110)
    print("  FINAL SUMMARY")
    print("=" * 110)

    # Phase 1 summary
    avg_excess_p1 = np.mean([r['wr_excess'] for r in p1_results])
    genuine_p1 = sum(1 for r in p1_results if r['wr_excess'] > 5)
    print(f"\n  Phase 1 - TP/SL Decomposition:")
    print(f"    Avg WR Excess over random walk: {avg_excess_p1:+.1f}pp")
    print(f"    Patterns with >5pp excess: {genuine_p1}/{len(p1_results)}")

    # Phase 2 summary
    genuine_p2 = sum(1 for r in p2_results if r['verdict'] == 'GENUINE')
    print(f"\n  Phase 2 - Direction Flip:")
    print(f"    Genuine directional edge: {genuine_p2}/{len(p2_results)}")
    print(f"    Regime-suspect: {len(p2_results) - genuine_p2}/{len(p2_results)}")

    # Phase 3 summary
    all_above = sum(1 for r in p3_results if r['all_above_50'])
    partial_above = sum(1 for r in p3_results if r['count_above_50'] > 0 and not r['all_above_50'])
    print(f"\n  Phase 3 - Symmetric TP/SL:")
    print(f"    Strong directional edge (all levels >50%): {all_above}/{len(p3_results)}")
    print(f"    Conditional edge (some levels >50%): {partial_above}/{len(p3_results)}")

    # Phase 4 summary
    avg_opt_excess = np.mean([r['optimal_excess'] for r in p4_optimal])
    avg_cur_excess = np.mean([r['current_excess'] for r in p4_optimal])
    changed = sum(1 for r in p4_optimal
                  if r['optimal_tp'] != r['current_tp']
                  or r['optimal_sl'] != r['current_sl'])
    print(f"\n  Phase 4 - Grid Search Optimal:")
    print(f"    Avg current WR Excess: {avg_cur_excess:+.1f}pp")
    print(f"    Avg optimal WR Excess: {avg_opt_excess:+.1f}pp")
    print(f"    Patterns with changed TP/SL: {changed}/{len(p4_optimal)}")

    # Phase 5 summary
    cur = p5_scenarios.get('current_T864', {})
    opt = p5_scenarios.get('optimal_T864', {})
    print(f"\n  Phase 5 - Portfolio WF (T864):")
    print(f"    Current: PnL {cur.get('cmp_pnl', 0):+.1f}%, MDD {cur.get('cmp_mdd', 0):.1f}%, "
          f"WF {cur.get('wf_verdict', 'N/A')}")
    print(f"    Optimal: PnL {opt.get('cmp_pnl', 0):+.1f}%, MDD {opt.get('cmp_mdd', 0):.1f}%, "
          f"WF {opt.get('wf_verdict', 'N/A')}")

    # Cross-phase pattern classification
    print(f"\n  --- CROSS-PHASE PATTERN CLASSIFICATION ---")
    print(f"  {'Pattern':<12} {'Dir':>5} | {'P1:Excess':>9} {'P2:Flip':>10} {'P3:Sym':>8} {'P4:OptExc':>9} | {'Overall':>10}")
    print("  " + "-" * 80)

    # Build lookup maps
    p2_map = {r['pattern']: r for r in p2_results}
    p3_map = {r['pattern']: r for r in p3_results}
    p4_map = {r['pattern']: r for r in p4_optimal}

    classification_results = []
    for r in p1_results:
        pat = r['pattern']
        p2r = p2_map.get(pat, {})
        p3r = p3_map.get(pat, {})
        p4r = p4_map.get(pat, {})

        # Scoring: each phase contributes a signal
        p1_genuine = r['wr_excess'] > 5
        p2_genuine = p2r.get('verdict') == 'GENUINE'
        p3_genuine = p3r.get('all_above_50', False)
        p4_positive = p4r.get('optimal_excess', 0) > 5

        score = sum([p1_genuine, p2_genuine, p3_genuine, p4_positive])
        if score >= 3:
            overall = 'STRONG'
        elif score >= 2:
            overall = 'MODERATE'
        elif score >= 1:
            overall = 'WEAK'
        else:
            overall = 'NONE'

        classification_results.append({
            'pattern': pat,
            'direction': r['direction'],
            'p1_excess': r['wr_excess'],
            'p2_verdict': p2r.get('verdict', 'N/A'),
            'p3_all_above_50': p3r.get('all_above_50', False),
            'p4_optimal_excess': p4r.get('optimal_excess', 0),
            'score': score,
            'overall': overall,
        })

        p2_str = p2r.get('verdict', 'N/A')[:7]
        p3_str = "YES" if p3r.get('all_above_50', False) else "NO"
        p4_str = f"{p4r.get('optimal_excess', 0):+.1f}pp"

        print(f"  {pat:<12} {r['direction']:>5} | "
              f"{r['wr_excess']:>+8.1f}% {p2_str:>10} {p3_str:>8} {p4_str:>9} | "
              f"{overall:>10}")

    output['cross_phase_classification'] = classification_results

    strong = sum(1 for c in classification_results if c['overall'] == 'STRONG')
    moderate = sum(1 for c in classification_results if c['overall'] == 'MODERATE')
    weak = sum(1 for c in classification_results if c['overall'] == 'WEAK')
    none_ = sum(1 for c in classification_results if c['overall'] == 'NONE')

    print(f"\n  Classification summary:")
    print(f"    STRONG (3-4 phases confirm genuine edge): {strong}")
    print(f"    MODERATE (2 phases confirm): {moderate}")
    print(f"    WEAK (1 phase confirms): {weak}")
    print(f"    NONE (0 phases confirm): {none_}")

    output['summary'] = {
        'phase1_avg_excess': round(avg_excess_p1, 1),
        'phase2_genuine_count': genuine_p2,
        'phase3_all_above_50_count': all_above,
        'phase4_avg_optimal_excess': round(avg_opt_excess, 1),
        'classification_strong': strong,
        'classification_moderate': moderate,
        'classification_weak': weak,
        'classification_none': none_,
    }

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    elapsed = time.time() - t0
    print(f"\n  Elapsed: {elapsed:.1f}s")
    print(f"  Results saved to {OUTPUT_FILE}")
    print("=" * 110)


if __name__ == '__main__':
    main()
