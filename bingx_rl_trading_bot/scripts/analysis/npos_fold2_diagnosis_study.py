#!/usr/bin/env python3
"""
Fold 2 Near-PASS Diagnosis & Improvement Study
===============================================
Problem: N-pos WF Fold 2 OOS PnL = +2.91% (dangerously close to FAIL).
         Fold 1: +22.81%, Fold 3: +12.10% — Fold 2 is the weak link.

Fold 2: IS=[0, 36048), OOS=[36048, 54072) — ~62.5 days OOS
        IS patterns: 93 (65L+28S) — heavy LONG bias

Approach:
  Phase 1: Diagnosis — WHY is Fold 2 marginal?
    - Market regime in OOS period (price action, trend)
    - Filter breakdown: which filter blocks most in this period?
    - Win/Loss distribution: are there correlated losses?
    - LONG vs SHORT performance decomposition

  Phase 2: Hypothesis Testing
    H1: Relaxed agg_risk_cap (counter 5%/with 10%) — less blocking, more trades
    H2: WF fold count 3→4 (smaller OOS windows, less regime exposure)
    H3: WF fold count 3→5
    H4: Adaptive agg_risk — wider cap in lower-conviction regime
    H5: Loss burst brake integration (production has it, scanner N-pos doesn't)

  Phase 3: Robustness — does the fix hold across ALL folds?

Standard Research Protocol:
  - Entry: next-bar open
  - Exit: intrabar distance-based (from bar open)
  - Fee: 0.10% × 3 = 0.30%
  - Slippage: 0.02% buffer
  - Timeout: 864 bars (DROP)
  - Sizing: compound, 1/N
"""

import sys
import os
import time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from scripts.scanner.pattern_scanner import (
    load_and_classify,
    compute_atr_ratio,
    compute_ema_slope,
    build_signal_index,
    scan_universe_range,
    portfolio_npos,
    calc_stats_compound,
    portfolio_1pos,
    expanding_window_wf,
    find_neutral_window,
    LEVERAGE, FEE_PCT, TIMEOUT_BARS,
    DEFAULT_N_SLOTS, DEFAULT_DIRECTION_CAP, DEFAULT_REGIME_MULT,
    DEFAULT_AGG_RISK_COUNTER, DEFAULT_AGG_RISK_WITH,
    DEFAULT_MOMENTUM_LOOKBACK, DEFAULT_MOMENTUM_THRESHOLD, DEFAULT_MOMENTUM_COOLDOWN,
    DEFAULT_ATR_CLAMP_LO, DEFAULT_ATR_CLAMP_HI,
)

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', '..',
                         'data', 'btc_5m_270days_reclassified.csv')

BASELINE_NPOS_KWARGS = {
    'n_slots': DEFAULT_N_SLOTS,
    'direction_cap': DEFAULT_DIRECTION_CAP,
    'regime_mult': DEFAULT_REGIME_MULT,
    'agg_risk_counter': DEFAULT_AGG_RISK_COUNTER,
    'agg_risk_with': DEFAULT_AGG_RISK_WITH,
    'momentum_lookback': DEFAULT_MOMENTUM_LOOKBACK,
    'momentum_threshold': DEFAULT_MOMENTUM_THRESHOLD,
    'momentum_cooldown': DEFAULT_MOMENTUM_COOLDOWN,
    'clamp_lo': DEFAULT_ATR_CLAMP_LO,
    'clamp_hi': DEFAULT_ATR_CLAMP_HI,
    'timeout_bars': TIMEOUT_BARS,
}


def run_npos_wf(df, n_folds, npos_kwargs, label=""):
    """Run expanding window WF with N-pos evaluation."""
    closes_full = df['close'].values

    # Find neutral window and slice data (mirrors scanner main())
    nw = find_neutral_window(closes_full, tol_pct=1.0)
    bar_start, bar_end = nw

    df_nw = df.iloc[bar_start:bar_end].reset_index(drop=True)
    nw_opens = df_nw['open'].values
    nw_highs = df_nw['high'].values
    nw_lows = df_nw['low'].values
    nw_closes = df_nw['close'].values
    neutral_bars = len(df_nw)

    nw_types = df_nw['candle_type'].values
    signal_index = build_signal_index(nw_types, neutral_bars)

    result = expanding_window_wf(
        signal_index, nw_opens, nw_highs, nw_lows, neutral_bars,
        n_folds=n_folds,
        min_trades=25, edge_threshold=18.0, mc_threshold=0.01,
        uni_tp=2.0, uni_sl=3.0, mode='mae_mfe',
        use_atr=True, closes=nw_closes,
        atr_period=14, atr_window=576,
        clamp_lo=npos_kwargs.get('clamp_lo', 0.6),
        clamp_hi=npos_kwargs.get('clamp_hi', 1.7),
        max_baseline_wr=70.0,
        use_npos=True, npos_kwargs=npos_kwargs,
    )

    return result


def phase1_diagnosis(df):
    """Phase 1: Diagnose WHY Fold 2 is marginal."""
    print("=" * 70)
    print("PHASE 1: FOLD 2 DIAGNOSIS")
    print("=" * 70)

    n = len(df)
    closes = df['close'].values
    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values

    # Find neutral window
    nw = find_neutral_window(closes, tol_pct=1.0)
    bar_start, bar_end = nw
    neutral_n = bar_end - bar_start

    seg = neutral_n // 4  # 3 folds => 4 segments
    fold2_is_end = 2 * seg
    fold2_oos_start = fold2_is_end
    fold2_oos_end = 3 * seg

    # Map back to original data indices
    abs_oos_start = bar_start + fold2_oos_start
    abs_oos_end = bar_start + fold2_oos_end

    print(f"\nNeutral window: bars [{bar_start}, {bar_end}) = {neutral_n} bars")
    print(f"Fold 2 OOS: bars [{fold2_oos_start}, {fold2_oos_end}) in neutral")
    print(f"            bars [{abs_oos_start}, {abs_oos_end}) in absolute")

    # Price action in each fold OOS
    print(f"\n--- Price Action by Fold OOS ---")
    for fold_i in range(3):
        oos_s = (fold_i + 1) * seg + bar_start
        oos_e = (fold_i + 2) * seg + bar_start if fold_i < 2 else bar_end
        p_start = closes[oos_s]
        p_end = closes[min(oos_e - 1, n - 1)]
        pct = (p_end / p_start - 1) * 100
        print(f"  Fold {fold_i+1}: price {p_start:.0f} -> {p_end:.0f} ({pct:+.1f}%)")

    # Run baseline WF with detailed per-fold stats
    print(f"\n--- Baseline N-pos WF (3-fold) with blocked breakdown ---")
    result = run_npos_wf(df, 3, BASELINE_NPOS_KWARGS, "baseline")

    for f in result['folds']:
        blocked = f.get('blocked', {})
        print(f"\n  Fold {f['fold']}:")
        print(f"    IS patterns: {f['is_long']}L + {f['is_short']}S = {f['is_long']+f['is_short']}")
        print(f"    OOS: {f['oos_trades']} trades, WR {f['oos_wr']:.1f}%, PnL {f['oos_pnl']:+.2f}%")
        print(f"    Blocked: {blocked}")
        if 'max_corr_loss' in f:
            print(f"    Max corr loss: {f['max_corr_loss']:.2f}%")

    # Detailed Fold 2 analysis: LONG vs SHORT decomposition
    print(f"\n--- Fold 2 LONG vs SHORT decomposition ---")

    # Use neutral-window-relative indexing for scan
    nw_opens = opens[bar_start:bar_end]
    nw_highs = highs[bar_start:bar_end]
    nw_lows = lows[bar_start:bar_end]
    nw_closes = closes[bar_start:bar_end]

    nw_df = df.iloc[bar_start:bar_end].reset_index(drop=True)
    nw_types = nw_df['candle_type'].values
    nw_signal_index = build_signal_index(nw_types, neutral_n)
    nw_atr = compute_atr_ratio(nw_highs, nw_lows, nw_closes, 14, window=576)

    is_patterns = scan_universe_range(
        nw_signal_index, nw_opens, nw_highs, nw_lows, neutral_n,
        bar_start=0, bar_end=fold2_is_end, mode='mae_mfe',
        min_trades=25, edge_threshold=18.0, mc_threshold=0.01,
        atr_ratio=nw_atr, clamp_lo=0.6, clamp_hi=1.7,
        max_baseline_wr=70.0,
    )

    # Separate LONG and SHORT signals
    long_signals = []
    short_signals = []
    for r in is_patterns:
        for s in nw_signal_index.get(r['pattern'], []):
            if fold2_oos_start <= s < fold2_oos_end:
                tup = (s, r['pattern'], r['direction'], r['tp'], r['sl'])
                if r['direction'] == 'LONG':
                    long_signals.append(tup)
                else:
                    short_signals.append(tup)

    nw_ema = compute_ema_slope(nw_closes)

    for label, sigs in [("LONG-only", long_signals), ("SHORT-only", short_signals),
                        ("COMBINED", long_signals + short_signals)]:
        if not sigs:
            print(f"  {label}: no signals")
            continue
        trades, stats = portfolio_npos(
            sigs, nw_opens, nw_highs, nw_lows, nw_closes, neutral_n,
            nw_atr, nw_ema, fold2_oos_start, fold2_oos_end,
            **BASELINE_NPOS_KWARGS,
        )
        cs = calc_stats_compound(trades)
        print(f"  {label}: {len(sigs)} signals, {cs['trades']} trades, "
              f"WR {cs['wr']:.1f}%, PnL {cs['pnl']:+.2f}%, "
              f"MDD {cs['mdd']:.1f}%")
        print(f"    Blocked: {stats.get('blocked', {})}")

    return result


def phase2_hypotheses(df):
    """Phase 2: Test improvement hypotheses."""
    print("\n" + "=" * 70)
    print("PHASE 2: HYPOTHESIS TESTING")
    print("=" * 70)

    results = {}

    # H0: Baseline (3-fold, current params)
    print("\n--- H0: Baseline (3-fold, current params) ---")
    t0 = time.time()
    r = run_npos_wf(df, 3, BASELINE_NPOS_KWARGS, "H0")
    dt = time.time() - t0
    results['H0_baseline'] = r
    _print_wf_summary(r, dt)

    # H1: Relaxed agg_risk_cap (5/10 instead of 3/7)
    print("\n--- H1: Relaxed agg_risk_cap (counter=5, with=10) ---")
    kw = {**BASELINE_NPOS_KWARGS, 'agg_risk_counter': 5.0, 'agg_risk_with': 10.0}
    t0 = time.time()
    r = run_npos_wf(df, 3, kw, "H1")
    dt = time.time() - t0
    results['H1_relaxed_agg'] = r
    _print_wf_summary(r, dt)

    # H2: 4-fold WF (smaller OOS segments)
    print("\n--- H2: 4-fold WF ---")
    t0 = time.time()
    r = run_npos_wf(df, 4, BASELINE_NPOS_KWARGS, "H2")
    dt = time.time() - t0
    results['H2_4fold'] = r
    _print_wf_summary(r, dt)

    # H3: 5-fold WF
    print("\n--- H3: 5-fold WF ---")
    t0 = time.time()
    r = run_npos_wf(df, 5, BASELINE_NPOS_KWARGS, "H3")
    dt = time.time() - t0
    results['H3_5fold'] = r
    _print_wf_summary(r, dt)

    # H4: No regime sizing (mult=1.0) — test if regime sizing hurts Fold 2
    print("\n--- H4: No regime sizing (mult=1.0) ---")
    kw = {**BASELINE_NPOS_KWARGS, 'regime_mult': 1.0}
    t0 = time.time()
    r = run_npos_wf(df, 3, kw, "H4")
    dt = time.time() - t0
    results['H4_no_regime'] = r
    _print_wf_summary(r, dt)

    # H5: Relaxed agg_risk + 0.5 regime mult (middle ground)
    print("\n--- H5: Moderate agg_risk (4/8) + regime_mult=0.5 ---")
    kw = {**BASELINE_NPOS_KWARGS, 'agg_risk_counter': 4.0, 'agg_risk_with': 8.0,
          'regime_mult': 0.5}
    t0 = time.time()
    r = run_npos_wf(df, 3, kw, "H5")
    dt = time.time() - t0
    results['H5_moderate'] = r
    _print_wf_summary(r, dt)

    # H6: No agg_risk_cap at all (bound check)
    print("\n--- H6: No agg_risk_cap (disable) ---")
    kw = {**BASELINE_NPOS_KWARGS, 'agg_risk_counter': 999.0, 'agg_risk_with': 999.0}
    t0 = time.time()
    r = run_npos_wf(df, 3, kw, "H6")
    dt = time.time() - t0
    results['H6_no_agg'] = r
    _print_wf_summary(r, dt)

    # H7: No momentum guard
    print("\n--- H7: No momentum guard ---")
    kw = {**BASELINE_NPOS_KWARGS, 'momentum_lookback': 0, 'momentum_threshold': 999.0,
          'momentum_cooldown': 0}
    t0 = time.time()
    r = run_npos_wf(df, 3, kw, "H7")
    dt = time.time() - t0
    results['H7_no_momentum'] = r
    _print_wf_summary(r, dt)

    # H8: Direction cap 9 (effectively no cap within N=9)
    print("\n--- H8: Direction cap=9 (no cap) ---")
    kw = {**BASELINE_NPOS_KWARGS, 'direction_cap': 9}
    t0 = time.time()
    r = run_npos_wf(df, 3, kw, "H8")
    dt = time.time() - t0
    results['H8_no_dircap'] = r
    _print_wf_summary(r, dt)

    # H9: Minimal filters (no regime, no agg, no momentum — only dir_cap + timeout)
    print("\n--- H9: Minimal filters (dir_cap=7, timeout only) ---")
    kw = {**BASELINE_NPOS_KWARGS, 'regime_mult': 1.0,
          'agg_risk_counter': 999.0, 'agg_risk_with': 999.0,
          'momentum_lookback': 0, 'momentum_threshold': 999.0, 'momentum_cooldown': 0}
    t0 = time.time()
    r = run_npos_wf(df, 3, kw, "H9")
    dt = time.time() - t0
    results['H9_minimal'] = r
    _print_wf_summary(r, dt)

    return results


def phase3_summary(results):
    """Phase 3: Summary comparison table."""
    print("\n" + "=" * 70)
    print("PHASE 3: COMPARISON SUMMARY")
    print("=" * 70)

    print(f"\n{'Hypothesis':<25} {'Folds':>6} {'PASS':>5} {'Min OOS':>9} "
          f"{'Total PnL':>10} {'Avg WR':>7} {'Verdict':>8}")
    print("-" * 75)

    for label, r in sorted(results.items()):
        n_folds = r['n_folds']
        pos = r['positive_folds']
        total_pnl = r['total_oos_pnl']
        min_oos = min(f['oos_pnl'] for f in r['folds'])
        avg_wr = np.mean([f['oos_wr'] for f in r['folds']])
        all_pass = pos == n_folds
        verdict = "PASS" if all_pass else "FAIL"

        print(f"  {label:<23} {n_folds:>5}  {pos}/{n_folds}  "
              f"{min_oos:>+8.2f}%  {total_pnl:>+9.1f}%  {avg_wr:>6.1f}%  "
              f"{'✅' if all_pass else '❌'} {verdict}")

    # Find best improvement for Fold 2 stability
    print(f"\n--- Fold-by-Fold Detail (3-fold hypotheses only) ---")
    print(f"{'Hypothesis':<25}", end="")
    for i in range(3):
        print(f"  {'Fold '+str(i+1):>10}", end="")
    print()
    print("-" * 60)
    for label, r in sorted(results.items()):
        if r['n_folds'] != 3:
            continue
        print(f"  {label:<23}", end="")
        for f in r['folds']:
            pnl = f['oos_pnl']
            marker = " ⚠️" if 0 < pnl < 5 else ""
            print(f"  {pnl:>+8.2f}%{marker}", end="")
        print()

    # Recommendation
    print(f"\n--- RECOMMENDATION ---")
    # Find scenario with best minimum fold PnL while all PASS
    best = None
    best_min = -999
    for label, r in results.items():
        if r['positive_folds'] == r['n_folds']:
            min_pnl = min(f['oos_pnl'] for f in r['folds'])
            if min_pnl > best_min:
                best_min = min_pnl
                best = label
    if best:
        r = results[best]
        print(f"  Best min-fold safety: {best} (min OOS PnL = {best_min:+.2f}%)")
        print(f"  Total OOS PnL: {r['total_oos_pnl']:+.1f}%, "
              f"{r['positive_folds']}/{r['n_folds']} PASS")
    else:
        print("  No hypothesis achieves all-fold PASS!")


def _print_wf_summary(r, elapsed):
    """Print WF result summary."""
    for f in r['folds']:
        blocked = f.get('blocked', {})
        bl_str = ", ".join(f"{k}={v}" for k, v in sorted(blocked.items()) if v > 0)
        print(f"  Fold {f['fold']}: IS {f['is_long']}L+{f['is_short']}S, "
              f"OOS {f['oos_trades']}tr WR {f['oos_wr']:.1f}% "
              f"PnL {f['oos_pnl']:+.2f}% | {bl_str}")
    print(f"  => {r['positive_folds']}/{r['n_folds']} PASS, "
          f"total OOS PnL {r['total_oos_pnl']:+.1f}%, "
          f"{r['stable_pattern_count']} stable | {elapsed:.0f}s")


def main():
    print("Fold 2 Near-PASS Diagnosis & Improvement Study")
    print("=" * 70)
    t_start = time.time()

    df = load_and_classify(DATA_PATH)
    print(f"Loaded {len(df)} candles")

    # Phase 1
    baseline = phase1_diagnosis(df)

    # Phase 2
    all_results = phase2_hypotheses(df)

    # Phase 3
    phase3_summary(all_results)

    elapsed = time.time() - t_start
    print(f"\nTotal elapsed: {elapsed:.0f}s ({elapsed/60:.1f}min)")


if __name__ == '__main__':
    main()
