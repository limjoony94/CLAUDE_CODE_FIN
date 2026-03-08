#!/usr/bin/env python3
"""
Distribution-Based TP Hit Probability Study

5-Phase comparison: log-normal MFE distribution vs empirical percentile for TP derivation.

Phase 1: Distribution Fitting Diagnostics
Phase 2: TP Candidate Comparison
Phase 3: Full Scan Comparison (IS)
Phase 4: WF 3-fold OOS Validation
Phase 5: Sensitivity Analysis

Standard Research Protocol:
- Entry: signal bar +1 open
- Exit: intrabar high/low (distance-based)
- Sizing: compound (복리)
- Fees: 0.05% × 2 = 0.10%
- Slippage: 0.02% buffer
- MC: sign randomization (10k sims)
- WF: expanding window

Usage:
    cd bingx_rl_trading_bot
    python scripts/analysis/distribution_tp_study.py
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
from collections import defaultdict

# Project root
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
sys.path.insert(0, _PROJECT_ROOT)

from scripts.scanner.pattern_scanner import (
    load_and_classify, build_signal_index, find_neutral_window,
    compute_excursions, compute_atr_ratio, derive_tp_sl,
    fit_mfe_distribution, derive_tp_from_distribution,
    grid_search_mae_mfe, bt_signals_atr, bt_signals,
    portfolio_1pos, calc_stats, mc_test, expanding_window_wf,
    scan_patterns_mae_mfe,
    TP_PERCENTILES, SL_PERCENTILES, DIST_HIT_PROBS,
    MAX_BARS, LEVERAGE, FEE_PCT, MAX_BASELINE_WR,
    DEFAULT_ATR_PERIOD, DEFAULT_ATR_WINDOW,
    DEFAULT_ATR_CLAMP_LO, DEFAULT_ATR_CLAMP_HI,
)
from scripts.production.pattern_5m.indicators import classify_candle

try:
    from scipy.stats import lognorm, gamma, weibull_min, kstest
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False
    print("ERROR: scipy required for this study. Install: pip install scipy")
    sys.exit(1)


# ============================================================
# Config
# ============================================================
DATA_FILE = os.path.join(_PROJECT_ROOT, 'data', 'btc_5m_270days_reclassified.csv')
OUTPUT_FILE = os.path.join(_PROJECT_ROOT, 'results', 'distribution_tp_study.json')

EDGE_THRESHOLD = 18.0
MC_THRESHOLD = 0.01
MIN_TRADES = 25
NEUTRAL_TOL = 1.0
HOLDOUT_DAYS = 7
WF_FOLDS = 3
ATR_PERIOD = DEFAULT_ATR_PERIOD
ATR_WINDOW = DEFAULT_ATR_WINDOW
CLAMP_LO = DEFAULT_ATR_CLAMP_LO
CLAMP_HI = DEFAULT_ATR_CLAMP_HI


def print_header(phase_num, title):
    print(f"\n{'='*70}")
    print(f"  Phase {phase_num}: {title}")
    print(f"{'='*70}\n")


# ============================================================
# Phase 1: Distribution Fitting Diagnostics
# ============================================================
def phase1_fitting_diagnostics(df, signal_index, atr_ratio):
    print_header(1, "Distribution Fitting Diagnostics")

    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    n = len(df)

    results = []
    fit_counts = defaultdict(int)  # quality -> count
    alt_better_count = 0
    total_tested = 0

    for pat_name, sigs in signal_index.items():
        for direction in ['LONG', 'SHORT']:
            if len(sigs) < MIN_TRADES:
                continue

            excursions = compute_excursions(sigs, direction, opens, highs, lows, n, MAX_BARS)
            if len(excursions) < MIN_TRADES:
                continue

            mfes = [e['mfe_pct'] for e in excursions]
            positive_mfes = [m for m in mfes if m > 0]
            if len(positive_mfes) < 10:
                continue

            total_tested += 1
            arr = np.array(positive_mfes)

            # Log-normal fit
            ln_info = fit_mfe_distribution(mfes)
            if ln_info is None:
                fit_counts['failed'] += 1
                continue

            fit_counts[ln_info['fit_quality']] += 1

            # Alternative distributions
            try:
                g_shape, g_loc, g_scale = gamma.fit(arr, floc=0)
                g_ks, g_pval = kstest(arr, 'gamma', args=(g_shape, 0, g_scale))
            except Exception:
                g_pval = 0

            try:
                w_c, w_loc, w_scale = weibull_min.fit(arr, floc=0)
                w_ks, w_pval = kstest(arr, 'weibull_min', args=(w_c, 0, w_scale))
            except Exception:
                w_pval = 0

            best_dist = 'lognorm'
            best_pval = ln_info['ks_pval']
            if g_pval > best_pval:
                best_dist = 'gamma'
                best_pval = g_pval
            if w_pval > best_pval:
                best_dist = 'weibull'
                best_pval = w_pval

            if best_dist != 'lognorm':
                alt_better_count += 1

            results.append({
                'pattern': pat_name, 'direction': direction,
                'n_mfe': len(positive_mfes),
                'mfe_median': round(np.median(arr), 3),
                'ln_ks_pval': ln_info['ks_pval'],
                'ln_quality': ln_info['fit_quality'],
                'gamma_pval': round(g_pval, 4),
                'weibull_pval': round(w_pval, 4),
                'best_dist': best_dist,
            })

    print(f"Total pattern-direction combos tested: {total_tested}")
    print(f"\nLog-normal fit quality distribution:")
    for q in ['good', 'acceptable', 'poor', 'failed']:
        cnt = fit_counts.get(q, 0)
        pct = cnt / total_tested * 100 if total_tested > 0 else 0
        print(f"  {q:12s}: {cnt:4d} ({pct:.1f}%)")

    good_pct = fit_counts.get('good', 0) / total_tested * 100 if total_tested > 0 else 0
    accept_pct = (fit_counts.get('good', 0) + fit_counts.get('acceptable', 0)) / total_tested * 100 if total_tested > 0 else 0
    print(f"\n  Good+Acceptable: {accept_pct:.1f}%")

    print(f"\nAlternative distribution better: {alt_better_count}/{total_tested} "
          f"({alt_better_count/total_tested*100:.1f}%)")

    # KS p-value distribution for log-normal fits
    pvals = [r['ln_ks_pval'] for r in results if r['ln_ks_pval'] is not None]
    if pvals:
        print(f"\nKS p-value distribution (log-normal):")
        print(f"  Min: {min(pvals):.4f}")
        print(f"  Median: {np.median(pvals):.4f}")
        print(f"  Mean: {np.mean(pvals):.4f}")
        print(f"  >0.05: {sum(1 for p in pvals if p > 0.05)}/{len(pvals)} "
              f"({sum(1 for p in pvals if p > 0.05)/len(pvals)*100:.1f}%)")
        print(f"  >0.01: {sum(1 for p in pvals if p > 0.01)}/{len(pvals)} "
              f"({sum(1 for p in pvals if p > 0.01)/len(pvals)*100:.1f}%)")

    return {'fit_counts': dict(fit_counts), 'total_tested': total_tested,
            'alt_better_pct': round(alt_better_count/total_tested*100, 1) if total_tested > 0 else 0,
            'good_acceptable_pct': round(accept_pct, 1)}


# ============================================================
# Phase 2: TP Candidate Comparison
# ============================================================
def phase2_tp_candidate_comparison(df, signal_index, atr_ratio):
    print_header(2, "TP Candidate Comparison")

    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    n = len(df)

    comparisons = []
    dist_wider_count = 0
    total_compared = 0

    for pat_name, sigs in signal_index.items():
        for direction in ['LONG', 'SHORT']:
            if len(sigs) < MIN_TRADES:
                continue

            excursions = compute_excursions(sigs, direction, opens, highs, lows, n, MAX_BARS)
            if len(excursions) < MIN_TRADES:
                continue

            mfes = [e['mfe_pct'] for e in excursions]
            positive_mfes = [m for m in mfes if m > 0]
            if len(positive_mfes) < 10:
                continue

            # Percentile TPs
            pctile_tps = []
            for p in TP_PERCENTILES:
                tp_val = round(float(np.percentile(mfes, p)), 2)
                pctile_tps.append(tp_val)

            # Distribution TPs
            dist_info = fit_mfe_distribution(mfes)
            dist_tps = []
            if dist_info is not None:
                dist_candidates = derive_tp_from_distribution(dist_info, DIST_HIT_PROBS)
                dist_tps = [tp for _, tp in dist_candidates]
                total_compared += 1
                if len(dist_tps) > len(pctile_tps):
                    dist_wider_count += 1

            comparisons.append({
                'pattern': pat_name, 'direction': direction,
                'n_pctile_tps': len(pctile_tps),
                'pctile_tp_range': [min(pctile_tps), max(pctile_tps)] if pctile_tps else [],
                'n_dist_tps': len(dist_tps),
                'dist_tp_range': [min(dist_tps), max(dist_tps)] if dist_tps else [],
                'dist_fit': dist_info is not None,
            })

    print(f"Pattern-direction combos compared: {total_compared}")
    print(f"Distribution provides wider TP range: {dist_wider_count}/{total_compared} "
          f"({dist_wider_count/total_compared*100:.1f}%)" if total_compared > 0 else "N/A")

    # Aggregate stats
    p_counts = [c['n_pctile_tps'] for c in comparisons if c['n_pctile_tps'] > 0]
    d_counts = [c['n_dist_tps'] for c in comparisons if c['n_dist_tps'] > 0]
    print(f"\nPercentile TP candidates per pattern: {np.mean(p_counts):.1f} avg "
          f"(always {len(TP_PERCENTILES)})")
    print(f"Distribution TP candidates per pattern: {np.mean(d_counts):.1f} avg "
          f"(max {len(DIST_HIT_PROBS)})" if d_counts else "Distribution TPs: N/A")

    return {'total_compared': total_compared,
            'dist_wider_pct': round(dist_wider_count/total_compared*100, 1) if total_compared > 0 else 0,
            'avg_dist_candidates': round(np.mean(d_counts), 1) if d_counts else 0}


# ============================================================
# Phase 3: Full Scan Comparison (IS)
# ============================================================
def phase3_full_scan_comparison(df, signal_index, atr_ratio):
    print_header(3, "Full Scan Comparison (IS)")
    t0 = time.time()

    # Percentile scan (baseline)
    print("Scanning with percentile method...")
    t1 = time.time()
    result_pctile = scan_patterns_mae_mfe(
        df, edge_threshold=EDGE_THRESHOLD, mc_threshold=MC_THRESHOLD,
        min_trades=MIN_TRADES, signal_index=signal_index,
        max_baseline_wr=MAX_BASELINE_WR,
        atr_ratio=atr_ratio, clamp_lo=CLAMP_LO, clamp_hi=CLAMP_HI,
        tp_method='percentile',
    )
    t_pctile = time.time() - t1

    # Distribution scan
    print("Scanning with distribution method...")
    t2 = time.time()
    result_dist = scan_patterns_mae_mfe(
        df, edge_threshold=EDGE_THRESHOLD, mc_threshold=MC_THRESHOLD,
        min_trades=MIN_TRADES, signal_index=signal_index,
        max_baseline_wr=MAX_BASELINE_WR,
        atr_ratio=atr_ratio, clamp_lo=CLAMP_LO, clamp_hi=CLAMP_HI,
        tp_method='distribution',
    )
    t_dist = time.time() - t2

    # Compare
    n_pctile = len(result_pctile['long_patterns']) + len(result_pctile['short_patterns'])
    n_dist = len(result_dist['long_patterns']) + len(result_dist['short_patterns'])

    ps_p = result_pctile['portfolio_stats']
    ps_d = result_dist['portfolio_stats']

    print(f"\n{'Metric':<25} {'Percentile':>12} {'Distribution':>12} {'Delta':>10}")
    print("-" * 62)
    print(f"{'Patterns':<25} {n_pctile:>12} {n_dist:>12} {n_dist-n_pctile:>+10}")
    n_lp = len(result_pctile['long_patterns'])
    n_sp = len(result_pctile['short_patterns'])
    n_ld = len(result_dist['long_patterns'])
    n_sd = len(result_dist['short_patterns'])
    print(f"{'  LONG':<25} {n_lp:>12} {n_ld:>12} {n_ld-n_lp:>+10}")
    print(f"{'  SHORT':<25} {n_sp:>12} {n_sd:>12} {n_sd-n_sp:>+10}")
    print(f"{'Trades':<25} {ps_p['trades']:>12} {ps_d['trades']:>12} {ps_d['trades']-ps_p['trades']:>+10}")
    print(f"{'WR %':<25} {ps_p['wr']:>12.1f} {ps_d['wr']:>12.1f} {ps_d['wr']-ps_p['wr']:>+10.1f}")
    print(f"{'PnL %':<25} {ps_p['pnl']:>12.1f} {ps_d['pnl']:>12.1f} {ps_d['pnl']-ps_p['pnl']:>+10.1f}")
    print(f"{'MDD %':<25} {ps_p['mdd']:>12.1f} {ps_d['mdd']:>12.1f} {ps_d['mdd']-ps_p['mdd']:>+10.1f}")
    pnl_mdd_p = ps_p['pnl'] / ps_p['mdd'] if ps_p['mdd'] > 0 else 0
    pnl_mdd_d = ps_d['pnl'] / ps_d['mdd'] if ps_d['mdd'] > 0 else 0
    print(f"{'PnL/MDD':<25} {pnl_mdd_p:>12.1f} {pnl_mdd_d:>12.1f} {pnl_mdd_d-pnl_mdd_p:>+10.1f}")
    print(f"{'PF':<25} {ps_p['pf']:>12.2f} {ps_d['pf']:>12.2f} {ps_d['pf']-ps_p['pf']:>+10.2f}")
    print(f"{'Scan time (s)':<25} {t_pctile:>12.1f} {t_dist:>12.1f}")

    # TP/SL distribution comparison
    if result_pctile['tp_distribution'] and result_dist['tp_distribution']:
        print(f"\nTP range:")
        tp_p = result_pctile['tp_distribution']
        tp_d = result_dist['tp_distribution']
        print(f"  Percentile:    [{tp_p['min']:.2f}, {tp_p['max']:.2f}], median={tp_p['median']:.2f}")
        print(f"  Distribution:  [{tp_d['min']:.2f}, {tp_d['max']:.2f}], median={tp_d['median']:.2f}")

    # Hit prob stats for distribution patterns
    hit_probs = [v.get('tp_hit_prob') for v in result_dist['pattern_details'].values()
                 if v.get('tp_hit_prob') is not None]
    if hit_probs:
        print(f"\nDistribution TP hit probabilities (selected patterns):")
        print(f"  Min: {min(hit_probs):.2f}, Median: {np.median(hit_probs):.2f}, "
              f"Max: {max(hit_probs):.2f}")

    return {
        'percentile': {
            'patterns': n_pctile, 'long': n_lp, 'short': n_sp,
            **ps_p, 'pnl_mdd': round(pnl_mdd_p, 1), 'time_sec': round(t_pctile, 1),
        },
        'distribution': {
            'patterns': n_dist, 'long': n_ld, 'short': n_sd,
            **ps_d, 'pnl_mdd': round(pnl_mdd_d, 1), 'time_sec': round(t_dist, 1),
            'hit_probs': [round(h, 2) for h in hit_probs] if hit_probs else [],
        },
    }


# ============================================================
# Phase 4: WF 3-fold OOS Validation
# ============================================================
def phase4_wf_validation(df, signal_index, atr_ratio, closes_arr):
    print_header(4, f"WF {WF_FOLDS}-fold OOS Validation")

    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    n = len(df)

    results = {}
    for method in ['percentile', 'distribution']:
        print(f"\n--- {method.upper()} ---")
        t0 = time.time()
        wf = expanding_window_wf(
            signal_index, opens, highs, lows, n,
            n_folds=WF_FOLDS, mode='mae_mfe',
            min_trades=MIN_TRADES, edge_threshold=EDGE_THRESHOLD,
            mc_threshold=MC_THRESHOLD, max_baseline_wr=MAX_BASELINE_WR,
            closes=closes_arr, atr_period=ATR_PERIOD, atr_window=ATR_WINDOW,
            clamp_lo=CLAMP_LO, clamp_hi=CLAMP_HI, use_atr=True,
            tp_method=method,
        )
        elapsed = time.time() - t0

        print(f"  Time: {elapsed:.1f}s")
        print(f"  Positive folds: {wf['positive_folds']}/{WF_FOLDS}")
        print(f"  Total OOS PnL: {wf['total_oos_pnl']}%")
        print(f"  Total OOS trades: {wf['total_oos_trades']}")
        print(f"  Stable patterns: {wf['stable_pattern_count']}")

        for f in wf['folds']:
            status = "✅" if f['oos_positive'] else "❌"
            print(f"    Fold {f['fold']}: IS {f['is_patterns']}pat "
                  f"({f['is_long']}L+{f['is_short']}S) | "
                  f"OOS {f['oos_trades']}tr WR={f['oos_wr']}% "
                  f"PnL={f['oos_pnl']}% {status}")

        wf_pass = wf['positive_folds'] == WF_FOLDS
        results[method] = {
            'wf_pass': wf_pass,
            'positive_folds': wf['positive_folds'],
            'total_oos_pnl': wf['total_oos_pnl'],
            'total_oos_trades': wf['total_oos_trades'],
            'stable_patterns': wf['stable_pattern_count'],
            'folds': wf['folds'],
            'time_sec': round(elapsed, 1),
        }

    # GO/STOP judgment
    p_pass = results['percentile']['wf_pass']
    d_pass = results['distribution']['wf_pass']
    p_pnl = results['percentile']['total_oos_pnl']
    d_pnl = results['distribution']['total_oos_pnl']

    print(f"\n{'='*50}")
    print(f"  JUDGMENT")
    print(f"{'='*50}")

    if d_pass and d_pnl >= p_pnl:
        verdict = 'GO'
        reason = (f"Distribution OOS {d_pnl}% >= Percentile {p_pnl}%, "
                  f"WF {'PASS' if d_pass else 'FAIL'}")
    elif d_pass and d_pnl < p_pnl:
        verdict = 'INFORMATIONAL'
        reason = (f"Distribution WF PASS but OOS {d_pnl}% < Percentile {p_pnl}% "
                  f"(keep as option)")
    elif not d_pass and p_pass:
        verdict = 'STOP'
        reason = f"Distribution WF FAIL, Percentile WF PASS"
    elif not d_pass and not p_pass:
        verdict = 'STOP'
        reason = f"Both WF FAIL"
    else:
        verdict = 'STOP'
        reason = f"Percentile WF FAIL, unusual case"

    print(f"\n  Verdict: **{verdict}**")
    print(f"  Reason: {reason}")
    print(f"\n  Percentile: WF {'PASS' if p_pass else 'FAIL'}, OOS PnL {p_pnl}%")
    print(f"  Distribution: WF {'PASS' if d_pass else 'FAIL'}, OOS PnL {d_pnl}%")

    results['verdict'] = verdict
    results['reason'] = reason
    return results


# ============================================================
# Phase 5: Sensitivity Analysis
# ============================================================
def phase5_sensitivity(df, signal_index, atr_ratio):
    print_header(5, "Sensitivity Analysis")

    results = {}

    # 5a: DIST_HIT_PROBS resolution
    print("--- 5a: Hit probability resolution ---")
    resolutions = {
        '6pt': [0.50, 0.60, 0.70, 0.80, 0.90, 0.95],
        '11pt': DIST_HIT_PROBS,
        '20pt': [round(0.40 + i * 0.03, 2) for i in range(20)],
    }

    res_results = {}
    for name, probs in resolutions.items():
        r = scan_patterns_mae_mfe(
            df, edge_threshold=EDGE_THRESHOLD, mc_threshold=MC_THRESHOLD,
            min_trades=MIN_TRADES, signal_index=signal_index,
            max_baseline_wr=MAX_BASELINE_WR,
            atr_ratio=atr_ratio, clamp_lo=CLAMP_LO, clamp_hi=CLAMP_HI,
            tp_method='distribution', dist_hit_probs=probs,
        )
        n_pat = len(r['long_patterns']) + len(r['short_patterns'])
        ps = r['portfolio_stats']
        pnl_mdd = ps['pnl'] / ps['mdd'] if ps['mdd'] > 0 else 0
        res_results[name] = {
            'n_probs': len(probs), 'patterns': n_pat,
            'pnl': ps['pnl'], 'mdd': ps['mdd'], 'pnl_mdd': round(pnl_mdd, 1),
            'wr': ps['wr'], 'trades': ps['trades'],
        }
        print(f"  {name} ({len(probs)} points): {n_pat} pat, "
              f"PnL {ps['pnl']}%, MDD {ps['mdd']}%, PnL/MDD {pnl_mdd:.1f}")

    results['resolution'] = res_results

    # 5b: MAX_BASELINE_WR sensitivity
    print("\n--- 5b: MAX_BASELINE_WR sensitivity ---")
    wr_results = {}
    for bwr in [65.0, 70.0, 75.0]:
        r = scan_patterns_mae_mfe(
            df, edge_threshold=EDGE_THRESHOLD, mc_threshold=MC_THRESHOLD,
            min_trades=MIN_TRADES, signal_index=signal_index,
            max_baseline_wr=bwr,
            atr_ratio=atr_ratio, clamp_lo=CLAMP_LO, clamp_hi=CLAMP_HI,
            tp_method='distribution',
        )
        n_pat = len(r['long_patterns']) + len(r['short_patterns'])
        ps = r['portfolio_stats']
        pnl_mdd = ps['pnl'] / ps['mdd'] if ps['mdd'] > 0 else 0
        wr_results[str(bwr)] = {
            'patterns': n_pat, 'pnl': ps['pnl'], 'mdd': ps['mdd'],
            'pnl_mdd': round(pnl_mdd, 1), 'wr': ps['wr'],
        }
        print(f"  BWR {bwr}%: {n_pat} pat, PnL {ps['pnl']}%, "
              f"MDD {ps['mdd']}%, PnL/MDD {pnl_mdd:.1f}")

    results['baseline_wr'] = wr_results

    # 5c: Edge threshold sensitivity
    print("\n--- 5c: Edge threshold sensitivity ---")
    edge_results = {}
    for eth in [15.0, 18.0, 20.0, 21.8]:
        r = scan_patterns_mae_mfe(
            df, edge_threshold=eth, mc_threshold=MC_THRESHOLD,
            min_trades=MIN_TRADES, signal_index=signal_index,
            max_baseline_wr=MAX_BASELINE_WR,
            atr_ratio=atr_ratio, clamp_lo=CLAMP_LO, clamp_hi=CLAMP_HI,
            tp_method='distribution',
        )
        n_pat = len(r['long_patterns']) + len(r['short_patterns'])
        ps = r['portfolio_stats']
        pnl_mdd = ps['pnl'] / ps['mdd'] if ps['mdd'] > 0 else 0
        edge_results[str(eth)] = {
            'patterns': n_pat, 'pnl': ps['pnl'], 'mdd': ps['mdd'],
            'pnl_mdd': round(pnl_mdd, 1), 'wr': ps['wr'],
        }
        print(f"  Edge {eth}pp: {n_pat} pat, PnL {ps['pnl']}%, "
              f"MDD {ps['mdd']}%, PnL/MDD {pnl_mdd:.1f}")

    results['edge_threshold'] = edge_results
    return results


# ============================================================
# Main
# ============================================================
def main():
    print("=" * 70)
    print("  Distribution-Based TP Hit Probability Study")
    print("  Standard Research Protocol | MAE/MFE + Log-Normal Fit")
    print("=" * 70)

    t_start = time.time()

    # Load data
    print(f"\nLoading data: {DATA_FILE}")
    df = load_and_classify(DATA_FILE)
    print(f"Total bars: {len(df)}")

    # Neutral window
    closes_all = df['close'].values
    nw = find_neutral_window(closes_all, tol_pct=NEUTRAL_TOL)
    if nw is not None:
        ns, ne = nw
        drift = (closes_all[ne] / closes_all[ns] - 1) * 100
        neutral_days = (ne - ns) / 288
        df = df.iloc[ns:ne + 1].reset_index(drop=True)
        print(f"Neutral window: {neutral_days:.0f}d (bars {ns}-{ne}), "
              f"drift {drift:+.2f}%")
    else:
        print("WARNING: No neutral window found, using full data")

    # Holdout split
    holdout_bars = HOLDOUT_DAYS * 288
    if holdout_bars < len(df) - 288:
        df_holdout = df.iloc[-holdout_bars:].reset_index(drop=True)
        df = df.iloc[:-holdout_bars].reset_index(drop=True)
        print(f"Holdout: {HOLDOUT_DAYS}d ({holdout_bars} bars), "
              f"IS: {len(df)} bars")

    # Signal index + ATR
    types = df['candle_type'].tolist()
    n = len(types)
    signal_index = build_signal_index(types, n)
    print(f"Unique patterns: {len(signal_index)}")

    highs_arr = df['high'].values
    lows_arr = df['low'].values
    closes_arr = df['close'].values
    atr_ratio = compute_atr_ratio(highs_arr, lows_arr, closes_arr,
                                   ATR_PERIOD, ATR_WINDOW)

    # Run phases
    all_results = {
        'config': {
            'data_file': os.path.basename(DATA_FILE),
            'edge_threshold': EDGE_THRESHOLD,
            'mc_threshold': MC_THRESHOLD,
            'min_trades': MIN_TRADES,
            'neutral_tol': NEUTRAL_TOL,
            'holdout_days': HOLDOUT_DAYS,
            'wf_folds': WF_FOLDS,
            'dist_hit_probs': DIST_HIT_PROBS,
            'tp_percentiles': TP_PERCENTILES,
        },
    }

    all_results['phase1'] = phase1_fitting_diagnostics(df, signal_index, atr_ratio)
    all_results['phase2'] = phase2_tp_candidate_comparison(df, signal_index, atr_ratio)
    all_results['phase3'] = phase3_full_scan_comparison(df, signal_index, atr_ratio)
    all_results['phase4'] = phase4_wf_validation(df, signal_index, atr_ratio, closes_arr)
    all_results['phase5'] = phase5_sensitivity(df, signal_index, atr_ratio)

    total_time = time.time() - t_start
    all_results['total_time_sec'] = round(total_time, 1)

    # Save results
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

    class NumpyEncoder(json.JSONEncoder):
        def default(self, obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, (np.bool_,)):
                return bool(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return super().default(obj)

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(all_results, f, indent=2, cls=NumpyEncoder)
    print(f"\nResults saved to {OUTPUT_FILE}")

    # Final summary
    print(f"\n{'='*70}")
    print(f"  STUDY COMPLETE — {total_time:.0f}s")
    print(f"{'='*70}")
    print(f"\n  Phase 1: Log-normal fit — "
          f"{all_results['phase1']['good_acceptable_pct']:.0f}% good/acceptable")
    print(f"  Phase 2: Distribution wider TP range — "
          f"{all_results['phase2']['dist_wider_pct']:.0f}%")

    p3 = all_results['phase3']
    print(f"  Phase 3: IS — Pctile {p3['percentile']['pnl']}% vs "
          f"Dist {p3['distribution']['pnl']}%")

    p4 = all_results['phase4']
    print(f"  Phase 4: WF — Verdict: **{p4['verdict']}**")
    print(f"    Pctile OOS: {p4['percentile']['total_oos_pnl']}% "
          f"({'PASS' if p4['percentile']['wf_pass'] else 'FAIL'})")
    print(f"    Dist OOS:   {p4['distribution']['total_oos_pnl']}% "
          f"({'PASS' if p4['distribution']['wf_pass'] else 'FAIL'})")


if __name__ == '__main__':
    main()
