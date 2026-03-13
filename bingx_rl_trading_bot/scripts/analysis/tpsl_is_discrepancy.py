#!/usr/bin/env python3
"""IS Discrepancy Investigation

Why tpsl_full_reopt shows 75.9x for TP×0.5 SL×1.0
while tpsl_deep_crossval showed 113.0x?

Hypotheses:
H1: Pattern set difference (full_reopt loads 131 from JSON, deep_crossval re-discovers)
H2: Signal construction difference (build_signals vs scanner's own signal pipeline)
H3: Factor application difference (pre vs post ATR scaling)
H4: Classification difference (load_and_classify vs CSV candle_type column)

Test each hypothesis by running the same N-pos simulation with controlled variations.
"""

import sys, os, json, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from scripts.scanner.pattern_scanner import (
    build_signal_index, portfolio_npos, calc_stats_compound,
    compute_atr_ratio, compute_ema_slope, load_and_classify,
    scan_patterns_mae_mfe,
    FEE_PCT, LEVERAGE, MAX_BARS, DEFAULT_N_SLOTS, DEFAULT_DIRECTION_CAP,
    DEFAULT_REGIME_MULT, DEFAULT_AGG_RISK_COUNTER, DEFAULT_AGG_RISK_WITH,
    DEFAULT_MOMENTUM_LOOKBACK, DEFAULT_MOMENTUM_THRESHOLD, DEFAULT_MOMENTUM_COOLDOWN,
    DEFAULT_CASCADE_TIGHTEN_PCT, DEFAULT_ATR_CLAMP_LO, DEFAULT_ATR_CLAMP_HI,
    DEFAULT_ATR_PERIOD, DEFAULT_ATR_WINDOW, TIMEOUT_BARS,
    NPOS_EMA_PERIOD, NPOS_EMA_LOOKBACK, DEFAULT_EDGE_THRESHOLD,
)

DATA_FILE = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'btc_5m_270days_reclassified.csv')
PATTERNS_FILE = os.path.join(os.path.dirname(__file__), '..', '..', 'results', 'dynamic_patterns.json')
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), '..', '..', 'results', 'tpsl_is_discrepancy.json')


def run_npos(signal_tuples, opens, highs, lows, closes, n, atr_ratio, ema_slope,
             start_bar, end_bar):
    trades, stats = portfolio_npos(
        signal_tuples, opens, highs, lows, closes, n, atr_ratio, ema_slope,
        start_bar, end_bar,
        n_slots=DEFAULT_N_SLOTS, direction_cap=DEFAULT_DIRECTION_CAP,
        regime_mult=DEFAULT_REGIME_MULT,
        agg_risk_counter=DEFAULT_AGG_RISK_COUNTER,
        agg_risk_with=DEFAULT_AGG_RISK_WITH,
        momentum_lookback=DEFAULT_MOMENTUM_LOOKBACK,
        momentum_threshold=DEFAULT_MOMENTUM_THRESHOLD,
        momentum_cooldown=DEFAULT_MOMENTUM_COOLDOWN,
        cascade_tighten_pct=DEFAULT_CASCADE_TIGHTEN_PCT,
        timeout_bars=TIMEOUT_BARS)
    c = calc_stats_compound(trades)
    mdd = stats.get('mdd_mtm', c.get('mdd', 0))
    c['mdd_mtm'] = round(mdd, 2)
    c['pnl_mdd'] = round(c['pnl'] / mdd, 1) if mdd > 0 else round(c['pnl'], 1)
    c['blocked'] = stats.get('blocked', {})
    return c


def main():
    t0 = time.time()
    results = {'study': 'tpsl_is_discrepancy', 'date': '2026-03-12'}

    # Load data using scanner's own load_and_classify
    df = load_and_classify(DATA_FILE)
    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    n = len(df)

    signal_index = build_signal_index(df['candle_type'].tolist(), n)
    atr_ratio = compute_atr_ratio(highs, lows, closes,
                                   atr_period=DEFAULT_ATR_PERIOD, window=DEFAULT_ATR_WINDOW)
    ema_slope = compute_ema_slope(closes, period=NPOS_EMA_PERIOD, lookback=NPOS_EMA_LOOKBACK)

    start_bar = 576
    end_bar = n

    # Load pre-computed patterns from JSON
    with open(PATTERNS_FILE) as f:
        pdata = json.load(f)
    json_details = pdata.get('pattern_details', {})

    print(f"Data: {n} bars, {n/288:.0f} days")
    print(f"JSON patterns: {len(json_details)}")
    print(f"Signal index patterns: {len(signal_index)}")

    # ================================================================
    # Test 1: JSON patterns with original TP/SL (no scaling) — should match ~88.5x
    # ================================================================
    print("\n" + "="*60)
    print("TEST 1: JSON patterns, original TP/SL (TP×1.0)")
    print("="*60)

    sig_tuples_orig = []
    for key, info in json_details.items():
        pat = info['pattern']
        direction = info['direction']
        tp = info['tp']
        sl = info['sl']
        if pat in signal_index:
            for sig_bar in signal_index[pat]:
                sig_tuples_orig.append((sig_bar, pat, direction, tp, sl))

    print(f"Signal tuples: {len(sig_tuples_orig)}")
    if sig_tuples_orig:
        stats1 = run_npos(sig_tuples_orig, opens, highs, lows, closes, n,
                          atr_ratio, ema_slope, start_bar, end_bar)
        print(f"Result: trades={stats1['trades']} WR={stats1['wr']}% "
              f"PnL={stats1['pnl']:+.1f}% MDD={stats1['mdd_mtm']:.2f}% "
              f"PnL/MDD={stats1['pnl_mdd']:.1f}x")
        print(f"Expected: ~88.5x (v1.56.0 IS)")
        results['test1_json_orig'] = stats1
    else:
        print("NO SIGNALS — pattern names don't match signal_index")
        # Debug
        json_pats = set(info['pattern'] for info in json_details.values())
        si_pats = set(signal_index.keys())
        overlap = json_pats & si_pats
        print(f"JSON pattern names sample: {list(json_pats)[:5]}")
        print(f"Signal index names sample: {list(si_pats)[:5]}")
        print(f"Overlap: {len(overlap)}")
        results['test1_debug'] = {
            'json_pats_sample': list(json_pats)[:5],
            'si_pats_sample': list(si_pats)[:5],
            'overlap': len(overlap),
        }

    # ================================================================
    # Test 2: JSON patterns with TP×0.5 (v1.57.0 scaling)
    # ================================================================
    print("\n" + "="*60)
    print("TEST 2: JSON patterns, TP×0.5 (v1.57.0)")
    print("="*60)

    sig_tuples_scaled = []
    for key, info in json_details.items():
        pat = info['pattern']
        direction = info['direction']
        tp = round(max(0.3, info['tp'] * 0.5), 3)
        sl = info['sl']
        if pat in signal_index:
            for sig_bar in signal_index[pat]:
                sig_tuples_scaled.append((sig_bar, pat, direction, tp, sl))

    if sig_tuples_scaled:
        stats2 = run_npos(sig_tuples_scaled, opens, highs, lows, closes, n,
                          atr_ratio, ema_slope, start_bar, end_bar)
        print(f"Result: trades={stats2['trades']} WR={stats2['wr']}% "
              f"PnL={stats2['pnl']:+.1f}% MDD={stats2['mdd_mtm']:.2f}% "
              f"PnL/MDD={stats2['pnl_mdd']:.1f}x")
        print(f"Expected: ~113.0x (v1.57.0 deep_crossval IS)")
        results['test2_json_scaled'] = stats2

    # ================================================================
    # Test 3: Re-discover patterns using scanner, then N-pos
    # ================================================================
    print("\n" + "="*60)
    print("TEST 3: Re-discover patterns via scanner MAE/MFE")
    print("="*60)

    scan_result = scan_patterns_mae_mfe(
        df, edge_threshold=18.0, mc_threshold=0.01, min_trades=25,
        signal_index=signal_index, concurrency=0,
        atr_ratio=atr_ratio, clamp_lo=DEFAULT_ATR_CLAMP_LO, clamp_hi=DEFAULT_ATR_CLAMP_HI,
    )

    scanned_tpsl = scan_result['patterns_tpsl']
    scanned_details = scan_result.get('pattern_details', {})
    print(f"Scanner found: {len(scanned_tpsl)} patterns")

    # Compare pattern sets
    json_keys = set(f"{info['pattern']}_{info['direction']}" for info in json_details.values())
    scan_keys = set(scanned_details.keys())
    overlap = json_keys & scan_keys
    only_json = json_keys - scan_keys
    only_scan = scan_keys - json_keys
    print(f"JSON: {len(json_keys)}, Scanner: {len(scan_keys)}, Overlap: {len(overlap)}")
    print(f"Only in JSON: {len(only_json)}, Only in Scanner: {len(only_scan)}")

    # N-pos with scanned patterns (original TP/SL)
    sig_tuples_scan_orig = []
    for key, info in scanned_details.items():
        pat = info['pattern']
        direction = info['direction']
        tp = info['tp']
        sl = info['sl']
        if pat in signal_index:
            for sig_bar in signal_index[pat]:
                sig_tuples_scan_orig.append((sig_bar, pat, direction, tp, sl))

    stats3_orig = run_npos(sig_tuples_scan_orig, opens, highs, lows, closes, n,
                            atr_ratio, ema_slope, start_bar, end_bar)
    print(f"Scanner orig: trades={stats3_orig['trades']} PnL/MDD={stats3_orig['pnl_mdd']:.1f}x")

    # N-pos with scanned patterns, TP×0.5
    sig_tuples_scan_scaled = []
    for key, info in scanned_details.items():
        pat = info['pattern']
        direction = info['direction']
        tp = round(max(0.3, info['tp'] * 0.5), 3)
        sl = info['sl']
        if pat in signal_index:
            for sig_bar in signal_index[pat]:
                sig_tuples_scan_scaled.append((sig_bar, pat, direction, tp, sl))

    stats3_scaled = run_npos(sig_tuples_scan_scaled, opens, highs, lows, closes, n,
                              atr_ratio, ema_slope, start_bar, end_bar)
    print(f"Scanner TP×0.5: trades={stats3_scaled['trades']} PnL/MDD={stats3_scaled['pnl_mdd']:.1f}x")

    results['test3_scanner'] = {
        'n_scanned': len(scanned_tpsl),
        'overlap_with_json': len(overlap),
        'only_json': len(only_json),
        'only_scan': len(only_scan),
        'scan_orig': stats3_orig,
        'scan_scaled': stats3_scaled,
    }

    # ================================================================
    # Test 4: Compare TP/SL values — JSON vs fresh scan
    # ================================================================
    print("\n" + "="*60)
    print("TEST 4: TP/SL Value Comparison (JSON vs Fresh Scan)")
    print("="*60)

    tp_diffs = []
    sl_diffs = []
    for key in overlap:
        json_info = json_details[key]
        scan_info = scanned_details[key]
        tp_diff = json_info['tp'] - scan_info['tp']
        sl_diff = json_info['sl'] - scan_info['sl']
        tp_diffs.append(tp_diff)
        sl_diffs.append(sl_diff)

    if tp_diffs:
        print(f"TP diff (JSON - Scan): mean={np.mean(tp_diffs):.3f} "
              f"std={np.std(tp_diffs):.3f} max_abs={max(abs(d) for d in tp_diffs):.3f}")
        print(f"SL diff (JSON - Scan): mean={np.mean(sl_diffs):.3f} "
              f"std={np.std(sl_diffs):.3f} max_abs={max(abs(d) for d in sl_diffs):.3f}")
        exact_match = sum(1 for d in tp_diffs if abs(d) < 0.001)
        print(f"Exact TP match: {exact_match}/{len(tp_diffs)} ({exact_match/len(tp_diffs)*100:.1f}%)")

        results['test4_tpsl_comparison'] = {
            'tp_diff_mean': round(float(np.mean(tp_diffs)), 3),
            'tp_diff_std': round(float(np.std(tp_diffs)), 3),
            'sl_diff_mean': round(float(np.mean(sl_diffs)), 3),
            'sl_diff_std': round(float(np.std(sl_diffs)), 3),
            'exact_match_rate': round(exact_match / len(tp_diffs) * 100, 1),
        }

    # ================================================================
    # Test 5: full_reopt's approach — apply_factors then build_signals
    # ================================================================
    print("\n" + "="*60)
    print("TEST 5: Reproduce full_reopt methodology")
    print("="*60)

    # This mimics exactly what tpsl_full_reopt does
    selected_fullreopt = {}
    for key, info in json_details.items():
        pat = info['pattern']
        direction = info['direction']
        # TP×0.5 SL×1.0
        tp = round(max(0.3, info['tp'] * 0.5), 3)
        sl = round(max(0.5, info['sl'] * 1.0), 3)
        full_key = f"{pat}_{direction}"
        selected_fullreopt[full_key] = {'tp': tp, 'sl': sl}

    # Build signals the full_reopt way (rsplit on key)
    sig_tuples_reopt = []
    for key, tpsl in selected_fullreopt.items():
        parts = key.rsplit('_', 1)
        pat_name, direction = parts[0], parts[1]
        if pat_name in signal_index:
            for sig_bar in signal_index[pat_name]:
                sig_tuples_reopt.append((sig_bar, pat_name, direction, tpsl['tp'], tpsl['sl']))

    print(f"full_reopt signal tuples: {len(sig_tuples_reopt)}")
    print(f"test2 signal tuples: {len(sig_tuples_scaled)}")

    if sig_tuples_reopt:
        stats5 = run_npos(sig_tuples_reopt, opens, highs, lows, closes, n,
                          atr_ratio, ema_slope, start_bar, end_bar)
        print(f"full_reopt PnL/MDD: {stats5['pnl_mdd']:.1f}x")
        results['test5_fullreopt'] = stats5

    # Check for key format differences
    # The key in json_details is "BD-BD-BU_LONG", rsplit('_', 1) on full_key = "BD-BD-BU_LONG"
    # gives pat_name="BD-BD-BU", direction="LONG" — correct
    # But what if patterns have underscores in their names?
    underscore_pats = [k for k in selected_fullreopt.keys() if k.count('_') > 1]
    if underscore_pats:
        print(f"\nWARNING: Patterns with multiple underscores: {underscore_pats[:5]}")
        # These would be mis-parsed by rsplit('_', 1)
        # e.g. "ST_U_LONG" would give pat_name="ST_U", direction="LONG" — WRONG if pattern is "ST" direction "U_LONG"
        results['underscore_issue'] = underscore_pats[:10]

    # ================================================================
    # Summary
    # ================================================================
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)

    print(f"\n  Test 1 (JSON orig TP×1.0):     {results.get('test1_json_orig', {}).get('pnl_mdd', 'N/A')}x")
    print(f"  Test 2 (JSON TP×0.5):           {results.get('test2_json_scaled', {}).get('pnl_mdd', 'N/A')}x")
    print(f"  Test 3 (Scanner orig):          {stats3_orig['pnl_mdd']}x")
    print(f"  Test 3 (Scanner TP×0.5):        {stats3_scaled['pnl_mdd']}x")
    print(f"  Test 5 (full_reopt repro):      {results.get('test5_fullreopt', {}).get('pnl_mdd', 'N/A')}x")
    print(f"  Expected (deep_crossval):       113.0x")
    print(f"  Expected (v1.56.0 baseline):    88.5x")

    elapsed = time.time() - t0
    results['elapsed_seconds'] = round(elapsed, 1)

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved: {OUTPUT_FILE}")
    print(f"Elapsed: {elapsed:.0f}s")


if __name__ == '__main__':
    main()
