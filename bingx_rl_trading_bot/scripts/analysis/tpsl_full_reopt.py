#!/usr/bin/env python3
"""Full Stack Re-optimization Under TP Scaling

Three research questions:
1. TP×SL joint factor grid — is TP-only scaling optimal, or does SL also benefit from scaling?
2. Mechanism re-optimization — do Cascade/Timeout/AggRisk/Momentum/DirCap optimal values
   change when TP is halved?
3. Joint factor + mechanism — combined optimization

Phases:
1. TP×SL factor 2D grid search (N-pos)
2. Mechanism sweep under best TP×SL factor
   - Cascade SL tighten_pct: [70, 75, 80, 85, 90, 95]
   - Timeout bars: [144, 192, 240, 288, 384]
   - AggRisk counter: [5, 8, 10, 12]
   - AggRisk with: [10, 15, 20]
   - Momentum threshold: [1.0, 1.5, 2.0]
   - Direction cap: [5, 6, 7, 8]
3. Top mechanisms combined
4. WF validation
5. Discrimination test
6. Conclusion

Standard Research Protocol.
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
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), '..', '..', 'results', 'tpsl_full_reopt.json')


def load_data():
    print("Loading data...")
    df = load_and_classify(DATA_FILE)
    print(f"Data: {len(df)} bars, {len(df)/288:.0f} days")
    return df


def load_patterns():
    """Load current dynamic_patterns.json (original TP/SL before scaling)."""
    with open(PATTERNS_FILE) as f:
        data = json.load(f)
    details = data.get('pattern_details', {})
    patterns = {}
    for key, info in details.items():
        pat = info['pattern']
        direction = info['direction']
        tp = info['tp']
        sl = info['sl']
        patterns[f"{pat}_{direction}"] = {'tp': tp, 'sl': sl}
    print(f"Loaded {len(patterns)} patterns from dynamic_patterns.json")
    return patterns


def build_signals(patterns, signal_index):
    """Build signal tuples for portfolio_npos with given TP/SL."""
    tuples = []
    for key, tpsl in patterns.items():
        parts = key.rsplit('_', 1)
        pat_name, direction = parts[0], parts[1]
        if pat_name in signal_index:
            for sig_bar in signal_index[pat_name]:
                tuples.append((sig_bar, pat_name, direction, tpsl['tp'], tpsl['sl']))
    return tuples


def run_npos(signal_tuples, opens, highs, lows, closes, n, atr_ratio, ema_slope,
             start_bar, end_bar, **kwargs):
    """Run portfolio_npos with overridable parameters."""
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


def apply_factors(patterns, tp_factor, sl_factor):
    """Apply TP and SL scaling factors."""
    scaled = {}
    for key, tpsl in patterns.items():
        new_tp = round(max(0.3, tpsl['tp'] * tp_factor), 3)
        new_sl = round(max(0.5, tpsl['sl'] * sl_factor), 3)
        scaled[key] = {'tp': new_tp, 'sl': new_sl}
    return scaled


def main():
    t0 = time.time()
    results = {'study': 'tpsl_full_reopt', 'date': '2026-03-12', 'phases': {}}

    df = load_data()
    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    n = len(df)

    signal_index = build_signal_index(df['candle_type'].tolist(), n)
    atr_ratio = compute_atr_ratio(highs, lows, closes,
                                   atr_period=DEFAULT_ATR_PERIOD, window=DEFAULT_ATR_WINDOW)
    ema_slope = compute_ema_slope(closes, period=NPOS_EMA_PERIOD, lookback=NPOS_EMA_LOOKBACK)

    base_patterns = load_patterns()
    start_bar = 576
    end_bar = n

    # ================================================================
    # Phase 1: TP × SL Factor 2D Grid
    # ================================================================
    print("\n" + "="*60)
    print("PHASE 1: TP × SL Factor 2D Grid Search")
    print("="*60)

    tp_factors = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
    sl_factors = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2]

    grid_2d = {}
    best_combo = None
    best_pnl_mdd = -9999

    print(f"\n{'':8s}", end='')
    for sf in sl_factors:
        print(f"  SL×{sf:.1f}", end='')
    print()
    print("-" * (8 + len(sl_factors) * 8))

    for tf in tp_factors:
        print(f"TP×{tf:.1f} |", end='')
        for sf in sl_factors:
            scaled = apply_factors(base_patterns, tf, sf)
            sig_tuples = build_signals(scaled, signal_index)
            stats = run_npos(sig_tuples, opens, highs, lows, closes, n,
                             atr_ratio, ema_slope, start_bar, end_bar)

            key = f"TP{tf}_SL{sf}"
            grid_2d[key] = {
                'tp_factor': tf, 'sl_factor': sf,
                'stats': stats,
            }
            pnl_mdd = stats['pnl_mdd']
            print(f" {pnl_mdd:6.1f}x", end='')

            if pnl_mdd > best_pnl_mdd:
                best_pnl_mdd = pnl_mdd
                best_combo = (tf, sf)
        print()

    print(f"\nBest: TP×{best_combo[0]} SL×{best_combo[1]} = {best_pnl_mdd:.1f}x")

    # Current v1.57.0 baseline
    baseline_key = "TP0.5_SL1.0"
    baseline_pnl_mdd = grid_2d[baseline_key]['stats']['pnl_mdd']
    print(f"Current (TP×0.5, SL×1.0): {baseline_pnl_mdd:.1f}x")
    improvement = (best_pnl_mdd - baseline_pnl_mdd) / baseline_pnl_mdd * 100
    print(f"Improvement: {improvement:+.1f}%")

    results['phases']['phase1'] = {
        'grid_2d': {k: {'tp_factor': v['tp_factor'], 'sl_factor': v['sl_factor'],
                        'pnl': v['stats']['pnl'], 'pnl_mdd': v['stats']['pnl_mdd'],
                        'mdd_mtm': v['stats']['mdd_mtm'], 'wr': v['stats']['wr'],
                        'trades': v['stats']['trades']}
                    for k, v in grid_2d.items()},
        'best_combo': {'tp_factor': best_combo[0], 'sl_factor': best_combo[1],
                       'pnl_mdd': best_pnl_mdd},
        'baseline_pnl_mdd': baseline_pnl_mdd,
        'improvement_vs_baseline': round(improvement, 1),
    }

    # ================================================================
    # Phase 2: Mechanism Sweep Under Best Factor
    # ================================================================
    print("\n" + "="*60)
    print(f"PHASE 2: Mechanism Sweep (TP×{best_combo[0]} SL×{best_combo[1]})")
    print("="*60)

    best_patterns = apply_factors(base_patterns, best_combo[0], best_combo[1])
    best_sig_tuples = build_signals(best_patterns, signal_index)

    # Also run with v1.57.0 factor for comparison
    v157_patterns = apply_factors(base_patterns, 0.5, 1.0)
    v157_sig_tuples = build_signals(v157_patterns, signal_index)

    # Baseline with current mechanisms
    mech_baseline = run_npos(best_sig_tuples, opens, highs, lows, closes, n,
                              atr_ratio, ema_slope, start_bar, end_bar)
    print(f"Mechanism baseline (defaults): PnL/MDD={mech_baseline['pnl_mdd']:.1f}x "
          f"PnL={mech_baseline['pnl']:+.1f}%")

    mech_results = {}

    # 2a. Cascade SL tighten_pct
    print("\n  [2a] Cascade SL tighten_pct:")
    cascade_sweep = {}
    for val in [0, 50, 70, 75, 80, 85, 90, 95]:
        stats = run_npos(best_sig_tuples, opens, highs, lows, closes, n,
                         atr_ratio, ema_slope, start_bar, end_bar,
                         cascade_tighten_pct=val)
        cascade_sweep[val] = stats
        tag = " ← current" if val == 85 else ""
        print(f"    {val:3d}%: PnL/MDD={stats['pnl_mdd']:7.1f}x PnL={stats['pnl']:+8.1f}% "
              f"MDD={stats['mdd_mtm']:5.2f}% WR={stats['wr']:5.1f}%{tag}")

    best_cascade = max(cascade_sweep.items(), key=lambda x: x[1]['pnl_mdd'])
    print(f"    Best: {best_cascade[0]}% ({best_cascade[1]['pnl_mdd']:.1f}x)")
    mech_results['cascade'] = {
        'sweep': {str(k): {'pnl_mdd': v['pnl_mdd'], 'pnl': v['pnl'], 'mdd_mtm': v['mdd_mtm']}
                  for k, v in cascade_sweep.items()},
        'best': best_cascade[0], 'best_pnl_mdd': best_cascade[1]['pnl_mdd'],
    }

    # 2b. Timeout bars
    print("\n  [2b] Timeout bars:")
    timeout_sweep = {}
    for val in [96, 144, 192, 240, 288, 384, 576]:
        stats = run_npos(best_sig_tuples, opens, highs, lows, closes, n,
                         atr_ratio, ema_slope, start_bar, end_bar,
                         timeout_bars=val)
        timeout_sweep[val] = stats
        tag = " ← current" if val == 288 else ""
        print(f"    {val:3d}: PnL/MDD={stats['pnl_mdd']:7.1f}x PnL={stats['pnl']:+8.1f}% "
              f"MDD={stats['mdd_mtm']:5.2f}%{tag}")

    best_timeout = max(timeout_sweep.items(), key=lambda x: x[1]['pnl_mdd'])
    print(f"    Best: {best_timeout[0]} ({best_timeout[1]['pnl_mdd']:.1f}x)")
    mech_results['timeout'] = {
        'sweep': {str(k): {'pnl_mdd': v['pnl_mdd'], 'pnl': v['pnl'], 'mdd_mtm': v['mdd_mtm']}
                  for k, v in timeout_sweep.items()},
        'best': best_timeout[0], 'best_pnl_mdd': best_timeout[1]['pnl_mdd'],
    }

    # 2c. AggRisk counter
    print("\n  [2c] AggRisk counter:")
    aggr_counter_sweep = {}
    for val in [3, 5, 8, 10, 12, 15]:
        stats = run_npos(best_sig_tuples, opens, highs, lows, closes, n,
                         atr_ratio, ema_slope, start_bar, end_bar,
                         agg_risk_counter=val)
        aggr_counter_sweep[val] = stats
        tag = " ← current" if val == 8 else ""
        print(f"    {val:3d}%: PnL/MDD={stats['pnl_mdd']:7.1f}x PnL={stats['pnl']:+8.1f}%{tag}")

    best_aggr_c = max(aggr_counter_sweep.items(), key=lambda x: x[1]['pnl_mdd'])
    print(f"    Best: {best_aggr_c[0]}% ({best_aggr_c[1]['pnl_mdd']:.1f}x)")
    mech_results['agg_risk_counter'] = {
        'sweep': {str(k): {'pnl_mdd': v['pnl_mdd'], 'pnl': v['pnl']}
                  for k, v in aggr_counter_sweep.items()},
        'best': best_aggr_c[0],
    }

    # 2d. AggRisk with
    print("\n  [2d] AggRisk with:")
    aggr_with_sweep = {}
    for val in [8, 10, 15, 20, 25]:
        stats = run_npos(best_sig_tuples, opens, highs, lows, closes, n,
                         atr_ratio, ema_slope, start_bar, end_bar,
                         agg_risk_with=val)
        aggr_with_sweep[val] = stats
        tag = " ← current" if val == 15 else ""
        print(f"    {val:3d}%: PnL/MDD={stats['pnl_mdd']:7.1f}x PnL={stats['pnl']:+8.1f}%{tag}")

    best_aggr_w = max(aggr_with_sweep.items(), key=lambda x: x[1]['pnl_mdd'])
    print(f"    Best: {best_aggr_w[0]}% ({best_aggr_w[1]['pnl_mdd']:.1f}x)")
    mech_results['agg_risk_with'] = {
        'sweep': {str(k): {'pnl_mdd': v['pnl_mdd'], 'pnl': v['pnl']}
                  for k, v in aggr_with_sweep.items()},
        'best': best_aggr_w[0],
    }

    # 2e. Momentum threshold
    print("\n  [2e] Momentum threshold:")
    mom_sweep = {}
    for val in [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]:
        stats = run_npos(best_sig_tuples, opens, highs, lows, closes, n,
                         atr_ratio, ema_slope, start_bar, end_bar,
                         momentum_threshold=val)
        mom_sweep[val] = stats
        tag = " ← current" if val == 1.5 else ""
        print(f"    {val:.1f}%: PnL/MDD={stats['pnl_mdd']:7.1f}x PnL={stats['pnl']:+8.1f}%{tag}")

    best_mom = max(mom_sweep.items(), key=lambda x: x[1]['pnl_mdd'])
    print(f"    Best: {best_mom[0]}% ({best_mom[1]['pnl_mdd']:.1f}x)")
    mech_results['momentum'] = {
        'sweep': {str(k): {'pnl_mdd': v['pnl_mdd'], 'pnl': v['pnl']}
                  for k, v in mom_sweep.items()},
        'best': best_mom[0],
    }

    # 2f. Direction cap
    print("\n  [2f] Direction cap:")
    dcap_sweep = {}
    for val in [4, 5, 6, 7, 8, 9]:
        stats = run_npos(best_sig_tuples, opens, highs, lows, closes, n,
                         atr_ratio, ema_slope, start_bar, end_bar,
                         direction_cap=val)
        dcap_sweep[val] = stats
        tag = " ← current" if val == 7 else ""
        print(f"    {val}: PnL/MDD={stats['pnl_mdd']:7.1f}x PnL={stats['pnl']:+8.1f}%{tag}")

    best_dcap = max(dcap_sweep.items(), key=lambda x: x[1]['pnl_mdd'])
    print(f"    Best: {best_dcap[0]} ({best_dcap[1]['pnl_mdd']:.1f}x)")
    mech_results['direction_cap'] = {
        'sweep': {str(k): {'pnl_mdd': v['pnl_mdd'], 'pnl': v['pnl']}
                  for k, v in dcap_sweep.items()},
        'best': best_dcap[0],
    }

    results['phases']['phase2'] = {
        'factor_used': {'tp': best_combo[0], 'sl': best_combo[1]},
        'mechanism_baseline': {'pnl_mdd': mech_baseline['pnl_mdd'], 'pnl': mech_baseline['pnl']},
        'mechanisms': mech_results,
    }

    # ================================================================
    # Phase 3: Combined Best Mechanisms
    # ================================================================
    print("\n" + "="*60)
    print("PHASE 3: Combined Best Mechanisms")
    print("="*60)

    # Collect changes from baseline
    changes = {}
    if best_cascade[0] != 85:
        changes['cascade_tighten_pct'] = best_cascade[0]
    if best_timeout[0] != 288:
        changes['timeout_bars'] = best_timeout[0]
    if best_aggr_c[0] != 8:
        changes['agg_risk_counter'] = best_aggr_c[0]
    if best_aggr_w[0] != 15:
        changes['agg_risk_with'] = best_aggr_w[0]
    if best_mom[0] != 1.5:
        changes['momentum_threshold'] = best_mom[0]
    if best_dcap[0] != 7:
        changes['direction_cap'] = best_dcap[0]

    print(f"  Changes from current: {changes if changes else 'NONE'}")

    # Run combined
    combined_stats = run_npos(best_sig_tuples, opens, highs, lows, closes, n,
                               atr_ratio, ema_slope, start_bar, end_bar, **changes)
    print(f"  Combined: PnL/MDD={combined_stats['pnl_mdd']:.1f}x "
          f"PnL={combined_stats['pnl']:+.1f}% MDD={combined_stats['mdd_mtm']:.2f}%")
    print(f"  vs Baseline (current mechs): {mech_baseline['pnl_mdd']:.1f}x")

    combined_impr = (combined_stats['pnl_mdd'] - mech_baseline['pnl_mdd']) / mech_baseline['pnl_mdd'] * 100
    print(f"  Improvement: {combined_impr:+.1f}%")

    # Also run v1.57.0 baseline for reference
    v157_baseline = run_npos(v157_sig_tuples, opens, highs, lows, closes, n,
                              atr_ratio, ema_slope, start_bar, end_bar)
    print(f"\n  v1.57.0 (TP×0.5 SL×1.0 current mechs): PnL/MDD={v157_baseline['pnl_mdd']:.1f}x")

    # Run v1.57.0 + changed mechanisms
    if changes:
        v157_changed = run_npos(v157_sig_tuples, opens, highs, lows, closes, n,
                                 atr_ratio, ema_slope, start_bar, end_bar, **changes)
        print(f"  v1.57.0 + changed mechs: PnL/MDD={v157_changed['pnl_mdd']:.1f}x")
    else:
        v157_changed = v157_baseline

    results['phases']['phase3'] = {
        'changes': changes,
        'combined_stats': {'pnl_mdd': combined_stats['pnl_mdd'], 'pnl': combined_stats['pnl'],
                           'mdd_mtm': combined_stats['mdd_mtm'], 'wr': combined_stats['wr'],
                           'trades': combined_stats['trades']},
        'baseline_stats': {'pnl_mdd': mech_baseline['pnl_mdd'], 'pnl': mech_baseline['pnl']},
        'improvement': round(combined_impr, 1),
        'v157_baseline': {'pnl_mdd': v157_baseline['pnl_mdd'], 'pnl': v157_baseline['pnl']},
        'v157_changed': {'pnl_mdd': v157_changed['pnl_mdd'], 'pnl': v157_changed['pnl']},
    }

    # ================================================================
    # Phase 4: WF Validation
    # ================================================================
    print("\n" + "="*60)
    print("PHASE 4: Walk-Forward Validation")
    print("="*60)

    # Configurations to validate
    configs = {
        'v1.57.0_current': (v157_sig_tuples, {}),
        f'best_factor_TP{best_combo[0]}_SL{best_combo[1]}': (best_sig_tuples, {}),
    }
    if changes:
        configs[f'best_factor+mechs'] = (best_sig_tuples, changes)
        configs['v1.57.0+mechs'] = (v157_sig_tuples, changes)

    n_folds = 3
    wf_results = {}

    for cfg_name, (sig_tuples, mechs) in configs.items():
        oos_pnls = []
        for f_idx in range(n_folds):
            is_end = int(n * (f_idx + 2) / (n_folds + 1))
            oos_end = int(n * (f_idx + 3) / (n_folds + 1)) if f_idx < n_folds - 1 else n
            oos_stats = run_npos(sig_tuples, opens, highs, lows, closes, n,
                                  atr_ratio, ema_slope, is_end, oos_end, **mechs)
            oos_pnls.append(oos_stats['pnl'])
            tag = "PASS" if oos_stats['pnl'] > 0 else "FAIL"
            print(f"  {cfg_name:30s} F{f_idx+1}: OOS={oos_stats['pnl']:+7.1f}% [{tag}]")

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
    # Phase 5: Discrimination Test
    # ================================================================
    print("\n" + "="*60)
    print("PHASE 5: Discrimination Test (Best Config)")
    print("="*60)

    # Test: best factor config vs random factor shuffle
    best_cfg_name = max(
        [(k, v) for k, v in configs.items()],
        key=lambda x: run_npos(x[1][0], opens, highs, lows, closes, n,
                                atr_ratio, ema_slope, start_bar, end_bar,
                                **x[1][1])['pnl_mdd']
    )[0]
    best_cfg_tuples, best_cfg_mechs = configs[best_cfg_name]

    real = run_npos(best_cfg_tuples, opens, highs, lows, closes, n,
                     atr_ratio, ema_slope, start_bar, end_bar, **best_cfg_mechs)

    rng = np.random.RandomState(42)
    n_shuf = 20
    rand_pnls = []
    # Shuffle TP/SL assignments across patterns
    best_factor_patterns = apply_factors(base_patterns, best_combo[0], best_combo[1])
    for _ in range(n_shuf):
        keys = list(best_factor_patterns.keys())
        vals = list(best_factor_patterns.values())
        rng.shuffle(vals)
        shuf_pats = dict(zip(keys, vals))
        shuf_tuples = build_signals(shuf_pats, signal_index)
        r = run_npos(shuf_tuples, opens, highs, lows, closes, n,
                      atr_ratio, ema_slope, start_bar, end_bar, **best_cfg_mechs)
        rand_pnls.append(r['pnl'])

    gap = real['pnl'] - np.mean(rand_pnls)
    gap_pct = gap / abs(np.mean(rand_pnls)) * 100 if np.mean(rand_pnls) != 0 else 0
    p_val = sum(1 for r in rand_pnls if r >= real['pnl']) / n_shuf
    verdict = "DISCRIMINATING" if p_val < 0.1 else "NON-DISCRIMINATING"

    print(f"  Config: {best_cfg_name}")
    print(f"  Real PnL: {real['pnl']:+.1f}%  Random mean: {np.mean(rand_pnls):+.1f}%")
    print(f"  Gap: {gap:+.1f}% ({gap_pct:+.1f}%)  p={p_val:.3f}  {verdict}")

    results['phases']['phase5'] = {
        'config': best_cfg_name,
        'real_pnl': round(real['pnl'], 1),
        'random_mean': round(float(np.mean(rand_pnls)), 1),
        'gap': round(gap, 1), 'gap_pct': round(gap_pct, 1),
        'p_value': round(p_val, 3), 'verdict': verdict,
    }

    # ================================================================
    # Phase 6: Conclusion
    # ================================================================
    print("\n" + "="*60)
    print("PHASE 6: CONCLUSION")
    print("="*60)

    # Summary table
    print("\n  Strategy Summary:")
    print(f"  {'Config':35s} {'IS PnL/MDD':>12s} {'OOS Total':>10s} {'WF':>8s}")
    print(f"  {'-'*35} {'-'*12} {'-'*10} {'-'*8}")

    for cfg_name in configs:
        if cfg_name in wf_results:
            is_stats = run_npos(configs[cfg_name][0], opens, highs, lows, closes, n,
                                 atr_ratio, ema_slope, start_bar, end_bar,
                                 **configs[cfg_name][1])
            wf_info = wf_results[cfg_name]
            print(f"  {cfg_name:35s} {is_stats['pnl_mdd']:10.1f}x {wf_info['total_oos']:+9.1f}% "
                  f"{wf_info['verdict']:>8s}")

    # Key conclusions
    print("\n  KEY FINDINGS:")
    print(f"  1. Best TP×SL factor: TP×{best_combo[0]} SL×{best_combo[1]}")
    if best_combo[1] != 1.0:
        print(f"     → SL scaling DOES help (SL×{best_combo[1]} vs SL×1.0)")
    else:
        print(f"     → SL scaling NOT needed (SL×1.0 is optimal)")

    if changes:
        print(f"  2. Mechanism changes needed: {changes}")
    else:
        print(f"  2. Mechanism changes: NONE — current values are optimal under new TP")

    total_impr_vs_v157 = (combined_stats['pnl_mdd'] - v157_baseline['pnl_mdd']) / v157_baseline['pnl_mdd'] * 100
    print(f"  3. Total improvement vs v1.57.0: {total_impr_vs_v157:+.1f}%")

    if total_impr_vs_v157 < 10:
        final_rec = "KEEP v1.57.0 — improvements are marginal (<10%)"
    else:
        final_rec = f"UPDATE — TP×{best_combo[0]} SL×{best_combo[1]}"
        if changes:
            final_rec += f" + mechanism changes: {changes}"

    print(f"\n  RECOMMENDATION: {final_rec}")

    results['phases']['phase6'] = {
        'best_tp_factor': best_combo[0],
        'best_sl_factor': best_combo[1],
        'mechanism_changes': changes,
        'total_improvement_vs_v157': round(total_impr_vs_v157, 1),
        'recommendation': final_rec,
    }

    elapsed = time.time() - t0
    results['elapsed_seconds'] = round(elapsed, 1)

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved: {OUTPUT_FILE}")
    print(f"Elapsed: {elapsed:.0f}s")


if __name__ == '__main__':
    main()
