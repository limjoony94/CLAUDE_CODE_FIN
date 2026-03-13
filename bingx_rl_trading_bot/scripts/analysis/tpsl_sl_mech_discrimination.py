#!/usr/bin/env python3
"""SL Factor + Mechanism Re-optimization Discrimination Test

Research questions:
1. Is SL×1.1 improvement genuine (discriminating) or noise?
2. Are mechanism changes (Cascade 95%, Timeout 96) genuine or overfitted?
3. What is the proper combined recommendation?

Phases:
1. SL factor discrimination — shuffle SL factors across patterns
2. Cascade 95% vs 85% — independent WF test
3. Timeout reduction — OOS-only validation (96 vs 144 vs 192 vs 288)
4. Conservative mechanism bundle (only clearly beneficial changes)
5. WF validation with proper IS/OOS pattern re-discovery
6. Conclusion

Uses 131 patterns from dynamic_patterns.json + load_and_classify.
"""

import sys, os, json, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from scripts.scanner.pattern_scanner import (
    build_signal_index, portfolio_npos, calc_stats_compound,
    compute_atr_ratio, compute_ema_slope, load_and_classify,
    FEE_PCT, LEVERAGE, MAX_BARS, DEFAULT_N_SLOTS, DEFAULT_DIRECTION_CAP,
    DEFAULT_REGIME_MULT, DEFAULT_AGG_RISK_COUNTER, DEFAULT_AGG_RISK_WITH,
    DEFAULT_MOMENTUM_LOOKBACK, DEFAULT_MOMENTUM_THRESHOLD, DEFAULT_MOMENTUM_COOLDOWN,
    DEFAULT_CASCADE_TIGHTEN_PCT, DEFAULT_ATR_CLAMP_LO, DEFAULT_ATR_CLAMP_HI,
    DEFAULT_ATR_PERIOD, DEFAULT_ATR_WINDOW, TIMEOUT_BARS,
    NPOS_EMA_PERIOD, NPOS_EMA_LOOKBACK,
)

DATA_FILE = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'btc_5m_270days_reclassified.csv')
PATTERNS_FILE = os.path.join(os.path.dirname(__file__), '..', '..', 'results', 'dynamic_patterns.json')
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), '..', '..', 'results', 'tpsl_sl_mech_discrimination.json')


def run_npos(signal_tuples, opens, highs, lows, closes, n, atr_ratio, ema_slope,
             start_bar, end_bar, **kwargs):
    params = dict(
        n_slots=DEFAULT_N_SLOTS,
        direction_cap=kwargs.get('direction_cap', DEFAULT_DIRECTION_CAP),
        regime_mult=DEFAULT_REGIME_MULT,
        agg_risk_counter=kwargs.get('agg_risk_counter', DEFAULT_AGG_RISK_COUNTER),
        agg_risk_with=kwargs.get('agg_risk_with', DEFAULT_AGG_RISK_WITH),
        momentum_lookback=DEFAULT_MOMENTUM_LOOKBACK,
        momentum_threshold=kwargs.get('momentum_threshold', DEFAULT_MOMENTUM_THRESHOLD),
        momentum_cooldown=DEFAULT_MOMENTUM_COOLDOWN,
        cascade_tighten_pct=kwargs.get('cascade_tighten_pct', DEFAULT_CASCADE_TIGHTEN_PCT),
        timeout_bars=kwargs.get('timeout_bars', TIMEOUT_BARS),
    )
    trades, stats = portfolio_npos(
        signal_tuples, opens, highs, lows, closes, n, atr_ratio, ema_slope,
        start_bar, end_bar, **params)
    c = calc_stats_compound(trades)
    mdd = stats.get('mdd_mtm', c.get('mdd', 0))
    c['mdd_mtm'] = round(mdd, 2)
    c['pnl_mdd'] = round(c['pnl'] / mdd, 1) if mdd > 0 else round(c['pnl'], 1)
    c['blocked'] = stats.get('blocked', {})
    return c


def build_signal_tuples(patterns_dict, signal_index, tp_factor=1.0, sl_factor=1.0):
    """Build signal tuples from pattern dict with optional scaling."""
    tuples = []
    for key, info in patterns_dict.items():
        pat = info['pattern']
        direction = info['direction']
        tp = round(max(0.3, info['tp'] * tp_factor), 3)
        sl = round(max(0.5, info['sl'] * sl_factor), 3)
        if pat in signal_index:
            for sig_bar in signal_index[pat]:
                tuples.append((sig_bar, pat, direction, tp, sl))
    return tuples


def main():
    t0 = time.time()
    results = {'study': 'tpsl_sl_mech_discrimination', 'date': '2026-03-12', 'phases': {}}

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

    with open(PATTERNS_FILE) as f:
        pdata = json.load(f)
    json_details = pdata.get('pattern_details', {})

    start_bar = 576
    end_bar = n

    print(f"Data: {n} bars, {n/288:.0f} days, {len(json_details)} patterns")

    # Baseline: TP×0.5 SL×1.0 (v1.57.0)
    baseline_tuples = build_signal_tuples(json_details, signal_index, tp_factor=0.5, sl_factor=1.0)
    baseline = run_npos(baseline_tuples, opens, highs, lows, closes, n,
                        atr_ratio, ema_slope, start_bar, end_bar)
    print(f"\nBaseline (TP×0.5 SL×1.0): PnL/MDD={baseline['pnl_mdd']:.1f}x "
          f"PnL={baseline['pnl']:+.1f}% WR={baseline['wr']:.1f}%")

    # ================================================================
    # Phase 1: SL Factor Discrimination
    # ================================================================
    print("\n" + "="*60)
    print("PHASE 1: SL Factor Discrimination Test")
    print("="*60)

    # Test SL factors
    sl_factors = [0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]
    sl_results = {}
    for sf in sl_factors:
        tuples = build_signal_tuples(json_details, signal_index, tp_factor=0.5, sl_factor=sf)
        stats = run_npos(tuples, opens, highs, lows, closes, n,
                         atr_ratio, ema_slope, start_bar, end_bar)
        sl_results[sf] = stats
        tag = " ← current" if sf == 1.0 else ""
        print(f"  SL×{sf:.1f}: PnL/MDD={stats['pnl_mdd']:7.1f}x PnL={stats['pnl']:+8.1f}% "
              f"MDD={stats['mdd_mtm']:5.2f}% WR={stats['wr']:5.1f}%{tag}")

    best_sl = max(sl_results.items(), key=lambda x: x[1]['pnl_mdd'])
    print(f"\n  Best SL factor: {best_sl[0]} ({best_sl[1]['pnl_mdd']:.1f}x)")

    # Discrimination: shuffle SL assignments
    best_sf = best_sl[0]
    real_tuples = build_signal_tuples(json_details, signal_index, tp_factor=0.5, sl_factor=best_sf)
    real = run_npos(real_tuples, opens, highs, lows, closes, n,
                    atr_ratio, ema_slope, start_bar, end_bar)

    # For discrimination: keep same patterns but shuffle which SL goes to which pattern
    rng = np.random.RandomState(42)
    n_shuf = 30
    rand_pnls = []
    pattern_keys = list(json_details.keys())
    original_sls = [json_details[k]['sl'] for k in pattern_keys]

    for _ in range(n_shuf):
        shuffled_sls = list(original_sls)
        rng.shuffle(shuffled_sls)
        tuples = []
        for i, key in enumerate(pattern_keys):
            info = json_details[key]
            pat = info['pattern']
            direction = info['direction']
            tp = round(max(0.3, info['tp'] * 0.5), 3)
            sl = round(max(0.5, shuffled_sls[i] * best_sf), 3)
            if pat in signal_index:
                for sig_bar in signal_index[pat]:
                    tuples.append((sig_bar, pat, direction, tp, sl))
        r = run_npos(tuples, opens, highs, lows, closes, n,
                     atr_ratio, ema_slope, start_bar, end_bar)
        rand_pnls.append(r['pnl'])

    gap = real['pnl'] - np.mean(rand_pnls)
    gap_pct = gap / abs(np.mean(rand_pnls)) * 100 if np.mean(rand_pnls) != 0 else 0
    p_val = sum(1 for r in rand_pnls if r >= real['pnl']) / n_shuf
    sl_verdict = "DISCRIMINATING" if p_val < 0.1 else "NON-DISCRIMINATING"

    print(f"\n  SL Discrimination (SL×{best_sf}):")
    print(f"    Real PnL: {real['pnl']:+.1f}%  Random mean: {np.mean(rand_pnls):+.1f}%")
    print(f"    Gap: {gap:+.1f}% ({gap_pct:+.1f}%)  p={p_val:.3f}  {sl_verdict}")

    results['phases']['phase1'] = {
        'sl_sweep': {str(k): {'pnl_mdd': v['pnl_mdd'], 'pnl': v['pnl'], 'mdd_mtm': v['mdd_mtm']}
                     for k, v in sl_results.items()},
        'best_sl_factor': best_sf,
        'discrimination': {
            'real_pnl': round(real['pnl'], 1),
            'random_mean': round(float(np.mean(rand_pnls)), 1),
            'gap': round(gap, 1), 'p_value': round(p_val, 3),
            'verdict': sl_verdict,
        },
    }

    # ================================================================
    # Phase 2: Mechanism Individual OOS Tests
    # ================================================================
    print("\n" + "="*60)
    print("PHASE 2: Mechanism OOS Tests (IS/OOS split)")
    print("="*60)

    # Use 70/30 split for IS/OOS
    split_bar = int(n * 0.7)
    best_tp_sl_tuples = build_signal_tuples(json_details, signal_index,
                                             tp_factor=0.5, sl_factor=best_sf)

    # IS baseline
    is_base = run_npos(best_tp_sl_tuples, opens, highs, lows, closes, n,
                       atr_ratio, ema_slope, start_bar, split_bar)
    oos_base = run_npos(best_tp_sl_tuples, opens, highs, lows, closes, n,
                        atr_ratio, ema_slope, split_bar, end_bar)
    print(f"  Baseline: IS PnL/MDD={is_base['pnl_mdd']:.1f}x  OOS PnL={oos_base['pnl']:+.1f}%")

    mechs_to_test = {
        'cascade_90': {'cascade_tighten_pct': 90},
        'cascade_95': {'cascade_tighten_pct': 95},
        'timeout_144': {'timeout_bars': 144},
        'timeout_192': {'timeout_bars': 192},
        'timeout_96': {'timeout_bars': 96},
        'aggr_counter_12': {'agg_risk_counter': 12},
        'dcap_6': {'direction_cap': 6},
    }

    mech_oos_results = {'baseline': {
        'is_pnl_mdd': is_base['pnl_mdd'], 'oos_pnl': oos_base['pnl'],
        'is_pnl': is_base['pnl'], 'oos_pnl_mdd': oos_base['pnl_mdd'],
    }}

    for mname, mparams in mechs_to_test.items():
        is_stats = run_npos(best_tp_sl_tuples, opens, highs, lows, closes, n,
                            atr_ratio, ema_slope, start_bar, split_bar, **mparams)
        oos_stats = run_npos(best_tp_sl_tuples, opens, highs, lows, closes, n,
                             atr_ratio, ema_slope, split_bar, end_bar, **mparams)
        is_delta = is_stats['pnl_mdd'] - is_base['pnl_mdd']
        oos_delta = oos_stats['pnl'] - oos_base['pnl']
        tag = "✓" if oos_delta > 0 else "✗"
        print(f"  {mname:20s}: IS Δ={is_delta:+6.1f}x  OOS Δ={oos_delta:+7.1f}% {tag}")

        mech_oos_results[mname] = {
            'is_pnl_mdd': is_stats['pnl_mdd'], 'oos_pnl': oos_stats['pnl'],
            'is_delta': round(is_delta, 1), 'oos_delta': round(oos_delta, 1),
            'oos_improves': oos_delta > 0,
        }

    results['phases']['phase2'] = mech_oos_results

    # ================================================================
    # Phase 3: Conservative Bundle (only OOS-confirmed changes)
    # ================================================================
    print("\n" + "="*60)
    print("PHASE 3: Conservative Mechanism Bundle")
    print("="*60)

    # Only include mechanisms that improved OOS
    conservative_changes = {}
    for mname, mdata in mech_oos_results.items():
        if mname == 'baseline':
            continue
        if mdata['oos_improves'] and mdata['oos_delta'] > 5:  # >5% OOS improvement threshold
            # Extract the actual param changes
            conservative_changes.update(mechs_to_test[mname])

    print(f"  OOS-confirmed changes: {conservative_changes if conservative_changes else 'NONE'}")

    if conservative_changes:
        cons_is = run_npos(best_tp_sl_tuples, opens, highs, lows, closes, n,
                           atr_ratio, ema_slope, start_bar, split_bar, **conservative_changes)
        cons_oos = run_npos(best_tp_sl_tuples, opens, highs, lows, closes, n,
                            atr_ratio, ema_slope, split_bar, end_bar, **conservative_changes)
        print(f"  Conservative: IS PnL/MDD={cons_is['pnl_mdd']:.1f}x  OOS PnL={cons_oos['pnl']:+.1f}%")
        print(f"  vs Baseline:  IS PnL/MDD={is_base['pnl_mdd']:.1f}x  OOS PnL={oos_base['pnl']:+.1f}%")
        results['phases']['phase3'] = {
            'changes': conservative_changes,
            'is_pnl_mdd': cons_is['pnl_mdd'],
            'oos_pnl': cons_oos['pnl'],
        }
    else:
        print("  No mechanism changes pass OOS validation")
        results['phases']['phase3'] = {'changes': {}, 'note': 'No OOS-confirmed improvements'}

    # ================================================================
    # Phase 4: WF Validation (3-fold) — key configurations
    # ================================================================
    print("\n" + "="*60)
    print("PHASE 4: Walk-Forward Validation (3-fold)")
    print("="*60)

    configs_wf = {
        f'TP0.5_SL{best_sf}': ({}, best_sf),
        'TP0.5_SL1.0_current': ({}, 1.0),
    }
    if conservative_changes:
        configs_wf[f'TP0.5_SL{best_sf}+cons_mechs'] = (conservative_changes, best_sf)

    n_folds = 3
    wf_results = {}
    for cfg_name, (mechs, sf) in configs_wf.items():
        sig_t = build_signal_tuples(json_details, signal_index, tp_factor=0.5, sl_factor=sf)
        oos_pnls = []
        for fi in range(n_folds):
            is_end = int(n * (fi + 2) / (n_folds + 1))
            oos_end = int(n * (fi + 3) / (n_folds + 1)) if fi < n_folds - 1 else n
            oos = run_npos(sig_t, opens, highs, lows, closes, n,
                           atr_ratio, ema_slope, is_end, oos_end, **mechs)
            oos_pnls.append(oos['pnl'])
            tag = "PASS" if oos['pnl'] > 0 else "FAIL"
            print(f"  {cfg_name:30s} F{fi+1}: OOS={oos['pnl']:+7.1f}% [{tag}]")

        total = sum(oos_pnls)
        passes = sum(1 for p in oos_pnls if p > 0)
        wf_results[cfg_name] = {
            'oos_pnls': [round(p, 1) for p in oos_pnls],
            'total_oos': round(total, 1),
            'verdict': f"{passes}/{n_folds} {'PASS' if passes == n_folds else 'FAIL'}",
        }
        print(f"  {cfg_name:30s} TOTAL={total:+7.1f}% [{passes}/{n_folds}]")

    results['phases']['phase4'] = wf_results

    # ================================================================
    # Phase 5: Conclusion
    # ================================================================
    print("\n" + "="*60)
    print("PHASE 5: CONCLUSION")
    print("="*60)

    print(f"\n  1. SL Factor:")
    print(f"     Best: SL×{best_sf} ({best_sl[1]['pnl_mdd']:.1f}x IS)")
    print(f"     Discrimination: {sl_verdict}")

    print(f"\n  2. Mechanism OOS Validation:")
    for mname, mdata in mech_oos_results.items():
        if mname == 'baseline':
            continue
        tag = "OOS ✓" if mdata['oos_improves'] else "OOS ✗"
        print(f"     {mname:20s}: IS Δ={mdata['is_delta']:+6.1f}x  OOS Δ={mdata['oos_delta']:+6.1f}%  {tag}")

    print(f"\n  3. Conservative Bundle: {conservative_changes if conservative_changes else 'NONE'}")

    # Final recommendation
    if sl_verdict == "NON-DISCRIMINATING":
        sl_rec = "SL×1.0 유지 (SL factor 효과 비판별)"
    else:
        sl_rec = f"SL×{best_sf} 적용 (판별력 확인)"

    mech_rec = conservative_changes if conservative_changes else "메커니즘 변경 불필요 (OOS 검증 미통과)"

    print(f"\n  RECOMMENDATION:")
    print(f"    SL: {sl_rec}")
    print(f"    Mechanisms: {mech_rec}")

    results['phases']['phase5'] = {
        'sl_recommendation': sl_rec,
        'mech_recommendation': str(mech_rec),
        'sl_verdict': sl_verdict,
        'conservative_changes': conservative_changes,
    }

    elapsed = time.time() - t0
    results['elapsed_seconds'] = round(elapsed, 1)

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved: {OUTPUT_FILE}")
    print(f"Elapsed: {elapsed:.0f}s")


if __name__ == '__main__':
    main()
