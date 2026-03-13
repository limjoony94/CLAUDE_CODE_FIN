#!/usr/bin/env python3
"""
Corrected WF Validation — Fixed Fold Boundary Bug
===================================================
Previous scripts had `is_end = int(n * (fi + 2) / (n_folds + 1))`
which made Fold 3 OOS empty (is_end = n → zero OOS bars).

Correct formula: is_end = int(n * (fi + 1) / (n_folds + 1))

Re-validates:
  1. Current v1.58.0 (TP×0.5 SL×1.1)
  2. SL×1.0 baseline (for comparison)
  3. Conservative mechanism bundle (cascade 95%, timeout 96, aggr 12%)
  4. Discrimination test for SL×1.1 (shuffle SL assignments)
"""

import sys, os, json, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from scripts.scanner.pattern_scanner import (
    load_and_classify, build_signal_index, portfolio_npos, calc_stats_compound,
    compute_atr_ratio, compute_ema_slope,
    DEFAULT_N_SLOTS, DEFAULT_DIRECTION_CAP, DEFAULT_REGIME_MULT,
    DEFAULT_AGG_RISK_COUNTER, DEFAULT_AGG_RISK_WITH,
    DEFAULT_MOMENTUM_LOOKBACK, DEFAULT_MOMENTUM_THRESHOLD, DEFAULT_MOMENTUM_COOLDOWN,
    DEFAULT_CASCADE_TIGHTEN_PCT, TIMEOUT_BARS,
)

DATA_FILE = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'btc_5m_270days_reclassified.csv')
PATTERNS_JSON = os.path.join(os.path.dirname(__file__), '..', '..', 'results', 'dynamic_patterns.json')

def load_all():
    df = load_and_classify(DATA_FILE)
    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    types = df['candle_type'].values
    n = len(df)
    atr_ratio = compute_atr_ratio(highs, lows, closes)
    ema_slope = compute_ema_slope(closes)
    signal_index = build_signal_index(types, n)

    with open(PATTERNS_JSON) as f:
        data = json.load(f)
    json_details = data.get('pattern_details', {})

    return opens, highs, lows, closes, n, atr_ratio, ema_slope, signal_index, json_details

def build_signal_tuples(json_details, signal_index, tp_factor=0.5, sl_factor=1.0):
    tuples = []
    for key, det in json_details.items():
        pat = det['pattern']
        direction = det['direction']
        tp = round(max(0.3, det['tp'] * tp_factor), 3)
        sl = round(max(0.5, det['sl'] * sl_factor), 3)
        if pat in signal_index:
            for sig_bar in signal_index[pat]:
                tuples.append((sig_bar, pat, direction, tp, sl))
    return sorted(tuples, key=lambda x: x[0])

def run_npos(signal_tuples, opens, highs, lows, closes, n, atr_ratio, ema_slope,
             start_bar, end_bar, **kwargs):
    if not signal_tuples:
        return {'trades': 0, 'wr': 0.0, 'pnl': 0.0, 'mdd': 0.0, 'pnl_mdd': 0.0}
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
    active = [t for t in trades if not t.get('drop', False)]
    c = calc_stats_compound(active)
    mdd = stats.get('mdd_mtm', c.get('mdd', 0))
    c['mdd_mtm'] = round(mdd, 2)
    c['pnl_mdd'] = round(c['pnl'] / mdd, 1) if mdd > 0 else round(c['pnl'], 1)
    return c

def expanding_window_folds(n, n_folds=3):
    """CORRECT expanding window: is_end = n*(fi+1)/(nf+1), oos_end = n*(fi+2)/(nf+1)."""
    folds = []
    for fi in range(n_folds):
        is_end = int(n * (fi + 1) / (n_folds + 1))
        oos_end = int(n * (fi + 2) / (n_folds + 1)) if fi < n_folds - 1 else n
        folds.append((is_end, oos_end))
    return folds

def wf_validate(signal_tuples, opens, highs, lows, closes, n, atr_ratio, ema_slope,
                n_folds=3, label="", **mechs):
    folds = expanding_window_folds(n, n_folds)
    oos_pnls = []
    for fi, (is_end, oos_end) in enumerate(folds):
        oos = run_npos(signal_tuples, opens, highs, lows, closes, n,
                       atr_ratio, ema_slope, is_end, oos_end, **mechs)
        oos_pnls.append(oos['pnl'])
        tag = "PASS" if oos['pnl'] > 0 else "FAIL"
        print(f"  {label:35s} F{fi+1}: OOS={oos['pnl']:+7.1f}% ({oos['trades']}t, WR={oos['wr']:.1f}%) [{tag}]")
    total = sum(oos_pnls)
    passes = sum(1 for p in oos_pnls if p > 0)
    print(f"  {label:35s} TOTAL={total:+7.1f}% [{passes}/{n_folds}]")
    return {'oos_pnls': [round(p, 1) for p in oos_pnls], 'total': round(total, 1), 'passes': passes}

def main():
    t0 = time.time()
    opens, highs, lows, closes, n, atr_ratio, ema_slope, signal_index, json_details = load_all()
    print(f"Data: {n} bars, {n/288:.0f} days, {len(json_details)} patterns")
    results = {}

    # ================================================================
    # PHASE 1: IS Performance Comparison
    # ================================================================
    print("\n" + "="*60)
    print("PHASE 1: IS Performance (full data)")
    print("="*60)

    configs = {
        'TP0.5_SL1.0 (v1.57.0)':  (0.5, 1.0, {}),
        'TP0.5_SL1.1 (v1.58.0)':  (0.5, 1.1, {}),
        'TP0.5_SL1.1+cascade95':  (0.5, 1.1, {'cascade_tighten_pct': 95}),
        'TP0.5_SL1.1+timeout96':  (0.5, 1.1, {'timeout_bars': 96}),
        'TP0.5_SL1.1+cons_bundle': (0.5, 1.1, {'cascade_tighten_pct': 95, 'timeout_bars': 96, 'agg_risk_counter': 12.0}),
    }

    is_results = {}
    for name, (tp_f, sl_f, mechs) in configs.items():
        sigs = build_signal_tuples(json_details, signal_index, tp_factor=tp_f, sl_factor=sl_f)
        stats = run_npos(sigs, opens, highs, lows, closes, n, atr_ratio, ema_slope, 0, n, **mechs)
        print(f"  {name:35s}: {stats['trades']}t WR={stats['wr']:.1f}% PnL={stats['pnl']:+.1f}% MDD={stats['mdd_mtm']:.2f}% PnL/MDD={stats['pnl_mdd']:.1f}x")
        is_results[name] = {'trades': stats['trades'], 'wr': round(stats['wr'], 1),
                           'pnl': round(stats['pnl'], 1), 'mdd': stats['mdd_mtm'],
                           'pnl_mdd': stats['pnl_mdd']}
    results['is'] = is_results

    # ================================================================
    # PHASE 2: Corrected 3-Fold WF Validation
    # ================================================================
    print("\n" + "="*60)
    print("PHASE 2: Corrected 3-Fold WF (expanding window)")
    print("="*60)

    wf_results = {}
    for name, (tp_f, sl_f, mechs) in configs.items():
        sigs = build_signal_tuples(json_details, signal_index, tp_factor=tp_f, sl_factor=sl_f)
        wf = wf_validate(sigs, opens, highs, lows, closes, n, atr_ratio, ema_slope,
                         n_folds=3, label=name, **mechs)
        wf_results[name] = wf
        print()
    results['wf_3fold'] = wf_results

    # ================================================================
    # PHASE 3: 5-Fold WF for Robustness
    # ================================================================
    print("="*60)
    print("PHASE 3: 5-Fold WF (robustness check)")
    print("="*60)

    wf5_results = {}
    for name in ['TP0.5_SL1.0 (v1.57.0)', 'TP0.5_SL1.1 (v1.58.0)']:
        tp_f, sl_f, mechs = configs[name]
        sigs = build_signal_tuples(json_details, signal_index, tp_factor=tp_f, sl_factor=sl_f)
        wf = wf_validate(sigs, opens, highs, lows, closes, n, atr_ratio, ema_slope,
                         n_folds=5, label=name, **mechs)
        wf5_results[name] = wf
        print()
    results['wf_5fold'] = wf5_results

    # ================================================================
    # PHASE 4: SL×1.1 Discrimination (shuffle SL assignments)
    # ================================================================
    print("="*60)
    print("PHASE 4: SL×1.1 Discrimination Test")
    print("="*60)

    real_sigs = build_signal_tuples(json_details, signal_index, tp_factor=0.5, sl_factor=1.1)
    real_stats = run_npos(real_sigs, opens, highs, lows, closes, n, atr_ratio, ema_slope, 0, n)
    real_pnl = real_stats['pnl']

    # Shuffle SL values across patterns (keep TP fixed)
    np.random.seed(42)
    n_shuffles = 30
    random_pnls = []
    sl_values = [round(max(0.5, det['sl'] * 1.1), 3) for det in json_details.values()]

    for si in range(n_shuffles):
        shuffled_sls = sl_values.copy()
        np.random.shuffle(shuffled_sls)
        # Rebuild signals with shuffled SLs
        shuffled_tuples = []
        for idx, (key, det) in enumerate(json_details.items()):
            pat = det['pattern']
            direction = det['direction']
            tp = round(max(0.3, det['tp'] * 0.5), 3)
            sl = shuffled_sls[idx % len(shuffled_sls)]
            if pat in signal_index:
                for sig_bar in signal_index[pat]:
                    shuffled_tuples.append((sig_bar, pat, direction, tp, sl))
        shuffled_tuples.sort(key=lambda x: x[0])
        s_stats = run_npos(shuffled_tuples, opens, highs, lows, closes, n,
                          atr_ratio, ema_slope, 0, n)
        random_pnls.append(s_stats['pnl'])

    random_mean = np.mean(random_pnls)
    gap = real_pnl - random_mean
    gap_pct = gap / abs(random_mean) * 100 if random_mean != 0 else 0
    p_value = np.mean([1 for rp in random_pnls if rp >= real_pnl])

    print(f"  Real PnL: {real_pnl:+.1f}%  Random mean: {random_mean:+.1f}%")
    print(f"  Gap: {gap:+.1f}% ({gap_pct:+.1f}%)  p={p_value:.3f}  {'DISCRIMINATING' if p_value < 0.05 else 'NON-DISCRIMINATING'}")

    results['discrimination'] = {
        'real_pnl': round(real_pnl, 1),
        'random_mean': round(random_mean, 1),
        'gap': round(gap, 1),
        'gap_pct': round(gap_pct, 1),
        'p_value': round(p_value, 3),
        'verdict': 'DISCRIMINATING' if p_value < 0.05 else 'NON-DISCRIMINATING',
    }

    # ================================================================
    # SUMMARY
    # ================================================================
    print(f"\n{'='*60}")
    print("SUMMARY")
    print("="*60)

    print(f"\n  IS Performance:")
    for name, r in is_results.items():
        print(f"    {name:35s}: PnL/MDD={r['pnl_mdd']:.1f}x")

    print(f"\n  3-Fold WF (CORRECTED):")
    for name, r in wf_results.items():
        print(f"    {name:35s}: {r['oos_pnls']} Total={r['total']:+.1f}% [{r['passes']}/3]")

    print(f"\n  SL Discrimination: {results['discrimination']['verdict']}")
    print(f"    Gap: {results['discrimination']['gap']:+.1f}% ({results['discrimination']['gap_pct']:+.1f}%)")

    # Recommendation
    best_wf = max(wf_results.items(), key=lambda x: x[1]['total'])
    print(f"\n  RECOMMENDATION: {best_wf[0]}")
    print(f"    WF Total: {best_wf[1]['total']:+.1f}%, Passes: {best_wf[1]['passes']}/3")

    out = os.path.join(os.path.dirname(__file__), '..', '..', 'results', 'wf_corrected_validation.json')
    with open(out, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved: {os.path.abspath(out)}")
    print(f"Elapsed: {time.time()-t0:.0f}s")

if __name__ == '__main__':
    main()
