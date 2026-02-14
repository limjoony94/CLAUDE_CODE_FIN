#!/usr/bin/env python3
"""
Statistical Rigor Study — 다중검정 보정 + 엄격한 통계 기준 연구

핵심 문제:
  3456개 조합(1728 패턴 × 2 방향) 스캔 시 MC < 0.01 → ~35개 거짓 양성 예상.
  현재 51패턴 중 상당수가 우연의 산물일 가능성.

5-Phase:
  1. Multiple Testing Correction (Bonferroni + BH FDR)
  2. Min Trades Sensitivity (20/30/50/80)
  3. Tighter MC Threshold (0.01/0.005/0.001)
  4. Bootstrap Edge CI (95% lower bound > 0)
  5. Combined Strictness Scenarios + Portfolio WF
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
sys.path.insert(0, _PROJECT_ROOT)

from scripts.production.pattern_5m.indicators import classify_candle
from scripts.production.pattern_5m.constants import AVG_BODY_WINDOW

# ============================================================
# Constants
# ============================================================
FEE_PCT = 0.10
LEVERAGE = 3
MAX_BARS = 500
UNI_TP = 2.0
UNI_SL = 3.0
BASELINE_WR = UNI_SL / (UNI_TP + UNI_SL) * 100  # 60.0%

DATA_FILE = os.path.join(_PROJECT_ROOT, 'data', 'btc_5m_270days_reclassified.csv')
RESULTS_DIR = os.path.join(_PROJECT_ROOT, 'results')


# ============================================================
# Core Functions
# ============================================================

def load_and_classify(data_file):
    df = pd.read_csv(data_file)
    df['body'] = abs(df['close'] - df['open'])
    df['avg_body'] = df['body'].rolling(AVG_BODY_WINDOW, min_periods=1).mean()
    types = []
    for i, row in df.iterrows():
        ct = classify_candle(row, df.at[i, 'avg_body'])
        types.append(ct.value)
    df['candle_type'] = types
    return df


def build_signal_index(types, n):
    idx = {}
    for i in range(2, n):
        pat = f"{types[i-2]}-{types[i-1]}-{types[i]}"
        if pat not in idx:
            idx[pat] = []
        idx[pat].append(i)
    return idx


def bt_signals(signal_bars, direction, tp_pct, sl_pct, opens, highs, lows, n_bars):
    trades = []
    for idx in signal_bars:
        if idx + 1 >= n_bars:
            continue
        entry = opens[idx + 1]
        if entry <= 0:
            continue
        eb = idx + 1
        if direction == 'LONG':
            tpp = entry * (1 + tp_pct / 100)
            slp = entry * (1 - sl_pct / 100)
        else:
            tpp = entry * (1 - tp_pct / 100)
            slp = entry * (1 + sl_pct / 100)
        for j in range(idx + 2, min(idx + 2 + MAX_BARS, n_bars)):
            if direction == 'LONG':
                ht = highs[j] >= tpp
                hs = lows[j] <= slp
            else:
                ht = lows[j] <= tpp
                hs = highs[j] >= slp
            if ht and hs:
                bo = opens[j]
                dist_tp = abs(tpp - bo)
                dist_sl = abs(slp - bo)
                pnl = (tp_pct if dist_tp <= dist_sl else -sl_pct) * LEVERAGE - FEE_PCT
                trades.append((eb, j, pnl))
                break
            elif ht:
                trades.append((eb, j, tp_pct * LEVERAGE - FEE_PCT))
                break
            elif hs:
                trades.append((eb, j, -sl_pct * LEVERAGE - FEE_PCT))
                break
    return trades


def mc_test(pnls, n_sims=10000):
    """Sign randomization MC test with higher precision (10k sims)."""
    if len(pnls) < 5:
        return 1.0
    pnls_arr = np.array(pnls)
    actual = np.sum(pnls_arr)
    signs = np.random.choice([-1, 1], size=(n_sims, len(pnls_arr)))
    rand_sums = signs @ pnls_arr
    p = float(np.mean(rand_sums >= actual))
    return max(p, 1.0 / n_sims)  # floor at 1/n_sims for precision


def mc_test_high_precision(pnls, n_sims=50000):
    """High-precision MC for Bonferroni-level testing."""
    if len(pnls) < 5:
        return 1.0
    pnls_arr = np.array(pnls)
    actual = np.sum(pnls_arr)
    # Process in chunks to avoid memory issues
    count = 0
    chunk = 10000
    for _ in range(0, n_sims, chunk):
        cs = min(chunk, n_sims - _)
        signs = np.random.choice([-1, 1], size=(cs, len(pnls_arr)))
        rand_sums = signs @ pnls_arr
        count += np.sum(rand_sums >= actual)
    p = count / n_sims
    return max(p, 1.0 / n_sims)


def portfolio_1pos(all_trades):
    if not all_trades:
        return []
    all_trades.sort(key=lambda x: x[0])
    filtered = []
    last_exit = -1
    for eb, xb, pnl in all_trades:
        if eb > last_exit:
            filtered.append((eb, xb, pnl))
            last_exit = xb
    return filtered


def calc_simple_stats(trades):
    if not trades:
        return {'pnl': 0, 'trades': 0, 'wr': 0, 'mdd': 0, 'pf': 0}
    pnls = [t[2] for t in trades]
    cum = 0
    peak = 0
    mdd = 0
    wins = []
    losses = []
    for p in pnls:
        cum += p
        if cum > peak:
            peak = cum
        dd = peak - cum
        if dd > mdd:
            mdd = dd
        if p > 0:
            wins.append(p)
        else:
            losses.append(p)
    wsum = sum(wins)
    lsum = sum(abs(x) for x in losses)
    return {
        'pnl': round(cum, 1),
        'trades': len(pnls),
        'wr': round(len(wins) / len(pnls) * 100, 1) if pnls else 0,
        'mdd': round(mdd, 1),
        'pf': round(wsum / lsum, 2) if lsum > 0 else 999,
    }


def discover_patterns_for_wf(types, opens, highs, lows, n, start_idx, end_idx,
                              min_trades, mc_threshold, edge_threshold):
    """Discover patterns in region with given thresholds."""
    signal_index = {}
    for i in range(max(2, start_idx), end_idx):
        pat = f"{types[i-2]}-{types[i-1]}-{types[i]}"
        if pat not in signal_index:
            signal_index[pat] = []
        signal_index[pat].append(i)

    selected = {}
    for pat_name, sigs in signal_index.items():
        for direction in ['LONG', 'SHORT']:
            if len(sigs) < min_trades:
                continue
            trades = bt_signals(sigs, direction, UNI_TP, UNI_SL, opens, highs, lows, n)
            if len(trades) < min_trades:
                continue
            pnls = [t[2] for t in trades]
            wr = len([p for p in pnls if p > 0]) / len(pnls) * 100
            edge = wr - BASELINE_WR
            if edge < edge_threshold:
                continue
            p = mc_test(pnls)
            if p >= mc_threshold:
                continue
            key = f"{pat_name}_{direction}"
            selected[key] = {'pattern': pat_name, 'direction': direction}
    return selected


def portfolio_wf(types, opens, highs, lows, n, signal_index,
                 min_trades, mc_threshold, edge_threshold, pattern_whitelist=None):
    """3-fold portfolio WF with given thresholds. If pattern_whitelist given, intersect."""
    n_folds = 3
    fold_size = n // n_folds
    oos_results = []

    for fold in range(n_folds):
        oos_start = fold * fold_size
        oos_end = (fold + 1) * fold_size if fold < n_folds - 1 else n

        # IS discovery
        is_selected = {}
        for is_start, is_end in ([(0, oos_start)] if oos_start > 0 else []) + ([(oos_end, n)] if oos_end < n else []):
            found = discover_patterns_for_wf(types, opens, highs, lows, n,
                                             is_start, is_end,
                                             min_trades, mc_threshold, edge_threshold)
            is_selected.update(found)

        # Intersect with whitelist if provided
        if pattern_whitelist is not None:
            is_selected = {k: v for k, v in is_selected.items() if k in pattern_whitelist}

        # OOS portfolio
        oos_trades = []
        for key, info in is_selected.items():
            pat = info['pattern']
            direction = info['direction']
            if pat in signal_index:
                sigs = [s for s in signal_index[pat] if oos_start <= s < oos_end]
                if sigs:
                    trades = bt_signals(sigs, direction, UNI_TP, UNI_SL, opens, highs, lows, n)
                    oos_trades.extend(trades)

        port = portfolio_1pos(oos_trades)
        pnls = [t[2] for t in port]
        oos_pnl = sum(pnls) if pnls else 0
        oos_wr = (len([p for p in pnls if p > 0]) / len(pnls) * 100) if pnls else 0

        oos_results.append({
            'is_patterns': len(is_selected),
            'oos_trades': len(port),
            'oos_pnl': round(oos_pnl, 1),
            'oos_wr': round(oos_wr, 1),
            'oos_edge': round(oos_wr - BASELINE_WR, 1),
        })

    positive = len([r for r in oos_results if r['oos_pnl'] > 0])
    avg_edge = np.mean([r['oos_edge'] for r in oos_results])
    avg_trades = np.mean([r['oos_trades'] for r in oos_results])
    total_oos_pnl = sum(r['oos_pnl'] for r in oos_results)

    return {
        'folds': oos_results,
        'positive': positive,
        'avg_oos_edge': round(avg_edge, 1),
        'avg_oos_trades': round(avg_trades, 1),
        'total_oos_pnl': round(total_oos_pnl, 1),
    }


# ============================================================
# Phase 1: Multiple Testing Correction
# ============================================================

def phase1_multiple_testing(types, opens, highs, lows, n, signal_index):
    """Apply Bonferroni and BH FDR corrections to all pattern p-values."""
    print("\n" + "=" * 70)
    print("Phase 1: Multiple Testing Correction")
    print("=" * 70)

    # Scan ALL patterns (min_trades=1 to get full count) but only keep those with enough trades
    all_results = []
    n_tested = 0

    for pat_name, sigs in signal_index.items():
        for direction in ['LONG', 'SHORT']:
            if len(sigs) < 10:  # absolute minimum
                continue
            trades = bt_signals(sigs, direction, UNI_TP, UNI_SL, opens, highs, lows, n)
            if len(trades) < 10:
                continue

            n_tested += 1
            pnls = [t[2] for t in trades]
            wr = len([p for p in pnls if p > 0]) / len(pnls) * 100
            edge = wr - BASELINE_WR

            if edge <= 0:
                continue

            # High-precision MC
            p = mc_test_high_precision(pnls, n_sims=50000)

            key = f"{pat_name}_{direction}"
            all_results.append({
                'key': key,
                'pattern': pat_name,
                'direction': direction,
                'trades': len(trades),
                'wr': round(wr, 1),
                'edge': round(edge, 1),
                'mc_p': p,
            })

    print(f"\n  Total pattern-direction combinations tested: {n_tested}")
    print(f"  Patterns with positive edge: {len(all_results)}")

    # Sort by p-value
    all_results.sort(key=lambda x: x['mc_p'])

    # Bonferroni correction
    bonf_threshold = 0.01 / n_tested
    bonf_pass = [r for r in all_results if r['mc_p'] <= bonf_threshold]

    # Benjamini-Hochberg FDR (q=0.05)
    fdr_q = 0.05
    bh_pass = []
    m = len(all_results)
    for i, r in enumerate(all_results):
        rank = i + 1
        bh_threshold = fdr_q * rank / m
        if r['mc_p'] <= bh_threshold:
            bh_pass.append(r)
        else:
            break  # BH step-up: stop at first failure

    # Standard MC < 0.01 (no correction)
    raw_pass = [r for r in all_results if r['mc_p'] < 0.01]

    print(f"\n  Bonferroni threshold: {bonf_threshold:.6f} (0.01 / {n_tested})")
    print(f"  Bonferroni pass: {len(bonf_pass)} patterns")

    print(f"\n  BH FDR (q=0.05) pass: {len(bh_pass)} patterns")

    print(f"\n  Raw MC < 0.01 (uncorrected): {len(raw_pass)} patterns")

    # Expected false positives
    expected_fp_raw = n_tested * 0.01
    print(f"\n  Expected false positives (raw MC<0.01): ~{expected_fp_raw:.0f}")
    print(f"  Expected false positives (Bonferroni): <1")
    print(f"  Expected false positives (BH FDR q=0.05): ~{len(bh_pass)*0.05:.1f}")

    # Apply min_trades=20 filter
    bonf_20 = [r for r in bonf_pass if r['trades'] >= 20]
    bh_20 = [r for r in bh_pass if r['trades'] >= 20]
    raw_20 = [r for r in raw_pass if r['trades'] >= 20]

    print(f"\n  With min_trades >= 20:")
    print(f"    Bonferroni: {len(bonf_20)} patterns")
    print(f"    BH FDR: {len(bh_20)} patterns")
    print(f"    Raw MC<0.01: {len(raw_20)} patterns")

    # Show Bonferroni patterns
    if bonf_20:
        print(f"\n  --- Bonferroni-surviving patterns (min_trades>=20) ---")
        print(f"  {'Pattern':<25} {'Trades':>6} {'WR':>7} {'Edge':>6} {'MC_p':>10}")
        for r in bonf_20:
            print(f"  {r['key']:<25} {r['trades']:>6} {r['wr']:>6.1f}% {r['edge']:>5.1f} {r['mc_p']:>10.6f}")

    # Show BH FDR patterns
    if bh_20:
        print(f"\n  --- BH FDR-surviving patterns (min_trades>=20) ---")
        print(f"  {'Pattern':<25} {'Trades':>6} {'WR':>7} {'Edge':>6} {'MC_p':>10}")
        for r in bh_20:
            print(f"  {r['key']:<25} {r['trades']:>6} {r['wr']:>6.1f}% {r['edge']:>5.1f} {r['mc_p']:>10.6f}")

    return {
        'n_tested': n_tested,
        'bonf_threshold': bonf_threshold,
        'bonf_pass_count': len(bonf_20),
        'bh_fdr_pass_count': len(bh_20),
        'raw_pass_count': len(raw_20),
        'expected_fp_raw': round(expected_fp_raw, 0),
        'bonf_patterns': [r['key'] for r in bonf_20],
        'bh_fdr_patterns': [r['key'] for r in bh_20],
        'all_results': all_results,
    }


# ============================================================
# Phase 2: Bootstrap Edge Confidence Intervals
# ============================================================

def phase2_bootstrap_ci(types, opens, highs, lows, n, signal_index):
    """Bootstrap 95% CI for each pattern's edge. Keep only if CI lower bound > 0."""
    print("\n" + "=" * 70)
    print("Phase 2: Bootstrap Edge Confidence Intervals")
    print("=" * 70)

    # Load current patterns
    with open(os.path.join(RESULTS_DIR, 'dynamic_patterns.json'), 'r') as f:
        dp = json.load(f)
    current = dp['pattern_details']

    n_boot = 5000
    ci_results = []

    for key, info in sorted(current.items(), key=lambda x: x[1]['trades'], reverse=True):
        pat = info['pattern']
        direction = info['direction']
        if pat not in signal_index:
            continue

        sigs = signal_index[pat]
        trades = bt_signals(sigs, direction, UNI_TP, UNI_SL, opens, highs, lows, n)
        if not trades:
            continue

        pnls = np.array([t[2] for t in trades])
        actual_wr = np.mean(pnls > 0) * 100
        actual_edge = actual_wr - BASELINE_WR

        # Bootstrap WR
        boot_wrs = []
        for _ in range(n_boot):
            sample = np.random.choice(pnls, size=len(pnls), replace=True)
            wr = np.mean(sample > 0) * 100
            boot_wrs.append(wr)

        boot_wrs = np.array(boot_wrs)
        boot_edges = boot_wrs - BASELINE_WR
        ci_lower = np.percentile(boot_edges, 2.5)
        ci_upper = np.percentile(boot_edges, 97.5)

        ci_results.append({
            'key': key,
            'trades': len(trades),
            'edge': round(actual_edge, 1),
            'ci_lower': round(ci_lower, 1),
            'ci_upper': round(ci_upper, 1),
            'ci_excludes_zero': bool(ci_lower > 0),
        })

    # Print results
    pass_ci = [r for r in ci_results if r['ci_excludes_zero']]
    fail_ci = [r for r in ci_results if not r['ci_excludes_zero']]

    print(f"\n  CI excludes zero (edge significant): {len(pass_ci)}/{len(ci_results)}")
    print(f"  CI includes zero (edge not significant): {len(fail_ci)}/{len(ci_results)}")

    if fail_ci:
        print(f"\n  --- Patterns where 95% CI includes zero ---")
        print(f"  {'Pattern':<25} {'Trades':>6} {'Edge':>6} {'CI_Low':>7} {'CI_Up':>7}")
        for r in sorted(fail_ci, key=lambda x: x['ci_lower']):
            print(f"  {r['key']:<25} {r['trades']:>6} {r['edge']:>5.1f} {r['ci_lower']:>6.1f} {r['ci_upper']:>6.1f}")

    return {
        'n_patterns': len(ci_results),
        'pass_ci': len(pass_ci),
        'fail_ci': len(fail_ci),
        'pass_patterns': [r['key'] for r in pass_ci],
        'fail_patterns': [r['key'] for r in fail_ci],
        'details': ci_results,
    }


# ============================================================
# Phase 3: Combined Scenarios + Portfolio WF
# ============================================================

def phase3_combined_scenarios(types, opens, highs, lows, n, signal_index,
                              p1_results, p2_results):
    """Test various strictness combinations with portfolio WF."""
    print("\n" + "=" * 70)
    print("Phase 3: Combined Strictness Scenarios + Portfolio WF")
    print("=" * 70)

    bonf_set = set(p1_results['bonf_patterns'])
    bh_set = set(p1_results['bh_fdr_patterns'])
    ci_pass_set = set(p2_results['pass_patterns'])

    # Load current pattern details for portfolio building
    with open(os.path.join(RESULTS_DIR, 'dynamic_patterns.json'), 'r') as f:
        dp = json.load(f)
    current_keys = set(dp['pattern_details'].keys())

    scenarios = [
        ('A_CURRENT_51', 'Current 51 (MC<0.01, min_trades>=20)', 20, 0.01, 5.0, None),
        ('B_MC_005', 'Tighter MC<0.005', 20, 0.005, 5.0, None),
        ('C_MC_001', 'Tighter MC<0.001', 20, 0.001, 5.0, None),
        ('D_TRADES_30', 'min_trades>=30', 30, 0.01, 5.0, None),
        ('E_TRADES_50', 'min_trades>=50', 50, 0.01, 5.0, None),
        ('F_BH_FDR', 'BH FDR q=0.05 corrected', 20, 0.01, 5.0, bh_set),
        ('G_BONFERRONI', 'Bonferroni corrected', 20, 0.01, 5.0, bonf_set),
        ('H_CI_PASS', 'Bootstrap 95% CI > 0', 20, 0.01, 5.0, ci_pass_set),
        ('I_BH_CI', 'BH FDR + Bootstrap CI', 20, 0.01, 5.0, bh_set & ci_pass_set),
        ('J_EDGE_10', 'edge>=10pp', 20, 0.01, 10.0, None),
        ('K_STRICT', 'MC<0.005 + trades>=30 + edge>=8pp', 30, 0.005, 8.0, None),
    ]

    print(f"\n  {'Scenario':<18} {'Desc':<35} {'IS_Pats':>7} {'Trades':>7} {'WR':>6} {'PnL':>9} {'MDD':>7} {'PnL/MDD':>8} | {'WF':>4} {'OOS_E':>6} {'OOS_T':>6}")
    print("  " + "-" * 130)

    all_scenario_results = {}

    for name, desc, mt, mc_th, edge_th, whitelist in scenarios:
        # IS scan
        all_pats = {}
        for pat_name, sigs in signal_index.items():
            for direction in ['LONG', 'SHORT']:
                if len(sigs) < mt:
                    continue
                trades = bt_signals(sigs, direction, UNI_TP, UNI_SL, opens, highs, lows, n)
                if len(trades) < mt:
                    continue
                pnls = [t[2] for t in trades]
                wr = len([p for p in pnls if p > 0]) / len(pnls) * 100
                edge = wr - BASELINE_WR
                if edge < edge_th:
                    continue
                p = mc_test(pnls)
                if p >= mc_th:
                    continue
                key = f"{pat_name}_{direction}"
                all_pats[key] = {'pattern': pat_name, 'direction': direction, 'trades_raw': trades}

        # Apply whitelist if given
        if whitelist is not None:
            all_pats = {k: v for k, v in all_pats.items() if k in whitelist}

        # IS portfolio stats
        all_trades = []
        for info in all_pats.values():
            all_trades.extend(info['trades_raw'])
        port = portfolio_1pos(all_trades)
        stats = calc_simple_stats(port)
        pnl_mdd = round(stats['pnl'] / stats['mdd'], 1) if stats['mdd'] > 0 else 999

        # Portfolio WF
        wf = portfolio_wf(types, opens, highs, lows, n, signal_index,
                         mt, mc_th, edge_th,
                         pattern_whitelist=whitelist)

        all_scenario_results[name] = {
            'desc': desc,
            'is_patterns': len(all_pats),
            'is_stats': stats,
            'is_pnl_mdd': pnl_mdd,
            'wf': wf,
        }

        print(f"  {name:<18} {desc[:34]:<35} {len(all_pats):>7} {stats['trades']:>7} {stats['wr']:>5.1f}% {stats['pnl']:>8.1f}% {stats['mdd']:>6.1f}% {pnl_mdd:>7.1f}x | {wf['positive']:>1}/3 {wf['avg_oos_edge']:>5.1f} {wf['avg_oos_trades']:>5.0f}")

    # Rank by OOS edge
    print(f"\n  Ranking by OOS Edge (WF average):")
    ranked = sorted(all_scenario_results.items(), key=lambda x: x[1]['wf']['avg_oos_edge'], reverse=True)
    for i, (name, res) in enumerate(ranked, 1):
        wf = res['wf']
        flag = " ***" if wf['positive'] < 3 else ""
        print(f"    {i}. {name}: OOS Edge {wf['avg_oos_edge']}pp, WF {wf['positive']}/3, IS {res['is_patterns']} patterns{flag}")

    # Find optimal: highest OOS edge with WF 3/3
    best_3_3 = [(name, res) for name, res in ranked if res['wf']['positive'] == 3]
    if best_3_3:
        best_name, best = best_3_3[0]
        print(f"\n  *** BEST (WF 3/3): {best_name} ***")
        print(f"      OOS Edge: {best['wf']['avg_oos_edge']}pp")
        print(f"      IS: {best['is_patterns']} patterns, PnL {best['is_stats']['pnl']:.1f}%, MDD {best['is_stats']['mdd']:.1f}%")

    return all_scenario_results


# ============================================================
# Phase 4: Edge Decay Analysis (IS → OOS)
# ============================================================

def phase4_edge_decay(types, opens, highs, lows, n, signal_index):
    """Measure edge decay from IS to OOS across different strictness levels."""
    print("\n" + "=" * 70)
    print("Phase 4: Edge Decay Analysis (IS → OOS)")
    print("=" * 70)

    configs = [
        ('MC<0.01 mt20', 20, 0.01, 5.0),
        ('MC<0.005 mt20', 20, 0.005, 5.0),
        ('MC<0.001 mt20', 20, 0.001, 5.0),
        ('MC<0.01 mt30', 30, 0.01, 5.0),
        ('MC<0.01 mt50', 50, 0.01, 5.0),
        ('MC<0.005 mt30', 30, 0.005, 5.0),
    ]

    n_folds = 3
    fold_size = n // n_folds

    print(f"\n  {'Config':<18} {'IS_Edge':>8} {'OOS_Edge':>9} {'Decay':>7} {'Decay%':>7} {'IS_WR':>7} {'OOS_WR':>8}")
    print("  " + "-" * 70)

    decay_results = []

    for label, mt, mc_th, edge_th in configs:
        is_edges = []
        oos_edges = []
        is_wrs = []
        oos_wrs = []

        for fold in range(n_folds):
            oos_start = fold * fold_size
            oos_end = (fold + 1) * fold_size if fold < n_folds - 1 else n

            # IS discovery
            is_selected = {}
            for is_start, is_end in ([(0, oos_start)] if oos_start > 0 else []) + ([(oos_end, n)] if oos_end < n else []):
                found = discover_patterns_for_wf(types, opens, highs, lows, n,
                                                 is_start, is_end, mt, mc_th, edge_th)
                is_selected.update(found)

            if not is_selected:
                continue

            # IS stats (from IS region)
            is_trades = []
            for key, info in is_selected.items():
                pat = info['pattern']
                direction = info['direction']
                if pat in signal_index:
                    is_sigs = [s for s in signal_index[pat] if s < oos_start or s >= oos_end]
                    if is_sigs:
                        trades = bt_signals(is_sigs, direction, UNI_TP, UNI_SL, opens, highs, lows, n)
                        is_trades.extend(trades)

            is_port = portfolio_1pos(is_trades)
            is_pnls = [t[2] for t in is_port]
            if is_pnls:
                is_wr = len([p for p in is_pnls if p > 0]) / len(is_pnls) * 100
                is_edges.append(is_wr - BASELINE_WR)
                is_wrs.append(is_wr)

            # OOS stats
            oos_trades = []
            for key, info in is_selected.items():
                pat = info['pattern']
                direction = info['direction']
                if pat in signal_index:
                    oos_sigs = [s for s in signal_index[pat] if oos_start <= s < oos_end]
                    if oos_sigs:
                        trades = bt_signals(oos_sigs, direction, UNI_TP, UNI_SL, opens, highs, lows, n)
                        oos_trades.extend(trades)

            oos_port = portfolio_1pos(oos_trades)
            oos_pnls = [t[2] for t in oos_port]
            if oos_pnls:
                oos_wr = len([p for p in oos_pnls if p > 0]) / len(oos_pnls) * 100
                oos_edges.append(oos_wr - BASELINE_WR)
                oos_wrs.append(oos_wr)

        if is_edges and oos_edges:
            avg_is = np.mean(is_edges)
            avg_oos = np.mean(oos_edges)
            decay = avg_is - avg_oos
            decay_pct = (decay / avg_is * 100) if avg_is > 0 else 0
            avg_is_wr = np.mean(is_wrs)
            avg_oos_wr = np.mean(oos_wrs)

            decay_results.append({
                'config': label,
                'is_edge': round(avg_is, 1),
                'oos_edge': round(avg_oos, 1),
                'decay_pp': round(decay, 1),
                'decay_pct': round(decay_pct, 1),
                'is_wr': round(avg_is_wr, 1),
                'oos_wr': round(avg_oos_wr, 1),
            })

            print(f"  {label:<18} {avg_is:>7.1f}pp {avg_oos:>8.1f}pp {decay:>6.1f}pp {decay_pct:>6.1f}% {avg_is_wr:>6.1f}% {avg_oos_wr:>7.1f}%")

    if decay_results:
        min_decay = min(decay_results, key=lambda x: x['decay_pct'])
        print(f"\n  Least edge decay: {min_decay['config']} ({min_decay['decay_pct']:.1f}% decay)")

    return decay_results


# ============================================================
# Main
# ============================================================

def main():
    np.random.seed(42)
    t0 = time.time()

    print("=" * 70)
    print("Statistical Rigor Study — v1.28.3 Multi-Testing Correction")
    print("Universal TP 2.0% / SL 3.0% | LEVERAGE 3x")
    print("=" * 70)

    print("\nLoading and classifying data...")
    df = load_and_classify(DATA_FILE)
    types = df['candle_type'].tolist()
    n = len(types)
    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    signal_index = build_signal_index(types, n)
    print(f"Data: {n} bars, {len(signal_index)} unique patterns")

    # Phase 1
    t1 = time.time()
    p1 = phase1_multiple_testing(types, opens, highs, lows, n, signal_index)
    print(f"  [Phase 1 time: {time.time()-t1:.1f}s]")

    # Phase 2
    t2 = time.time()
    p2 = phase2_bootstrap_ci(types, opens, highs, lows, n, signal_index)
    print(f"  [Phase 2 time: {time.time()-t2:.1f}s]")

    # Phase 3
    t3 = time.time()
    p3 = phase3_combined_scenarios(types, opens, highs, lows, n, signal_index, p1, p2)
    print(f"  [Phase 3 time: {time.time()-t3:.1f}s]")

    # Phase 4
    t4 = time.time()
    p4 = phase4_edge_decay(types, opens, highs, lows, n, signal_index)
    print(f"  [Phase 4 time: {time.time()-t4:.1f}s]")

    # ============================================================
    # Final Summary
    # ============================================================
    total_time = time.time() - t0
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    print(f"\n  1. Multiple Testing:")
    print(f"     - {p1['n_tested']} combinations tested → ~{p1['expected_fp_raw']:.0f} expected false positives at MC<0.01")
    print(f"     - Bonferroni (α={p1['bonf_threshold']:.6f}): {p1['bonf_pass_count']} patterns survive")
    print(f"     - BH FDR (q=0.05): {p1['bh_fdr_pass_count']} patterns survive")
    print(f"     - Raw MC<0.01: {p1['raw_pass_count']} patterns")

    print(f"\n  2. Bootstrap CI:")
    print(f"     - 95% CI excludes zero: {p2['pass_ci']}/{p2['n_patterns']} patterns")
    print(f"     - CI includes zero: {p2['fail_ci']} patterns (edge not significant)")

    # Find best scenario from Phase 3
    best_3_3 = [(name, res) for name, res in p3.items()
                if res['wf']['positive'] == 3]
    if best_3_3:
        best_3_3.sort(key=lambda x: x[1]['wf']['avg_oos_edge'], reverse=True)
        bn, br = best_3_3[0]
        print(f"\n  3. Best scenario (WF 3/3): {bn}")
        print(f"     - OOS Edge: {br['wf']['avg_oos_edge']}pp, IS Patterns: {br['is_patterns']}")

    if p4:
        min_decay = min(p4, key=lambda x: x['decay_pct'])
        print(f"\n  4. Edge decay: least decay at {min_decay['config']} ({min_decay['decay_pct']:.1f}%)")

    print(f"\n  Total time: {total_time:.1f}s")

    # Save
    output = {
        'version': 'v1.28.3',
        'date': '2026-02-14',
        'phase1_multiple_testing': {
            'n_tested': p1['n_tested'],
            'bonf_threshold': p1['bonf_threshold'],
            'bonf_pass': p1['bonf_patterns'],
            'bh_fdr_pass': p1['bh_fdr_patterns'],
            'expected_fp_raw': p1['expected_fp_raw'],
        },
        'phase2_bootstrap_ci': {
            'pass_count': p2['pass_ci'],
            'fail_count': p2['fail_ci'],
            'fail_patterns': p2['fail_patterns'],
            'details': p2['details'],
        },
        'phase3_scenarios': {
            name: {
                'desc': res['desc'],
                'is_patterns': res['is_patterns'],
                'is_pnl': res['is_stats']['pnl'],
                'is_mdd': res['is_stats']['mdd'],
                'is_pnl_mdd': res['is_pnl_mdd'],
                'wf_positive': res['wf']['positive'],
                'wf_avg_oos_edge': res['wf']['avg_oos_edge'],
                'wf_avg_oos_trades': res['wf']['avg_oos_trades'],
            }
            for name, res in p3.items()
        },
        'phase4_edge_decay': p4,
    }

    out_path = os.path.join(RESULTS_DIR, 'statistical_rigor_study.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\n  Results saved: {out_path}")


if __name__ == '__main__':
    main()
