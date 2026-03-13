#!/usr/bin/env python3
"""
TP Uniform vs Per-Pattern Scaled Comparison
============================================
Question: Scanner가 per-pattern MAE/MFE로 최적화한 TP에 ×0.5를 곱하는 것이 더 좋은가,
         아니면 모든 패턴에 동일한 고정 TP를 적용하는 것이 더 좋은가?

Comparison:
  A) Per-pattern TP×0.5 (현재 v1.58.0) — 패턴별 MAE/MFE 최적 TP에 0.5 곱함
  B) Uniform flat TP — 모든 패턴에 동일한 TP 적용 (SL은 per-pattern 유지)
  C) Per-pattern TP×factor sweep — 다양한 factor로 비교

Also validates TP/SL factor direction (not swapped).
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

def run_npos(signal_tuples, opens, highs, lows, closes, n, atr_ratio, ema_slope,
             start_bar, end_bar):
    if not signal_tuples:
        return {'trades': 0, 'wr': 0.0, 'pnl': 0.0, 'mdd': 0.0, 'pnl_mdd': 0.0}
    trades, stats = portfolio_npos(
        signal_tuples, opens, highs, lows, closes, n, atr_ratio, ema_slope,
        start_bar, end_bar,
        n_slots=DEFAULT_N_SLOTS, direction_cap=DEFAULT_DIRECTION_CAP,
        regime_mult=DEFAULT_REGIME_MULT,
        agg_risk_counter=DEFAULT_AGG_RISK_COUNTER, agg_risk_with=DEFAULT_AGG_RISK_WITH,
        momentum_lookback=DEFAULT_MOMENTUM_LOOKBACK,
        momentum_threshold=DEFAULT_MOMENTUM_THRESHOLD,
        momentum_cooldown=DEFAULT_MOMENTUM_COOLDOWN,
        cascade_tighten_pct=DEFAULT_CASCADE_TIGHTEN_PCT,
        timeout_bars=TIMEOUT_BARS,
    )
    active = [t for t in trades if not t.get('drop', False)]
    c = calc_stats_compound(active)
    mdd = stats.get('mdd_mtm', c.get('mdd', 0))
    c['mdd_mtm'] = round(mdd, 2)
    c['pnl_mdd'] = round(c['pnl'] / mdd, 1) if mdd > 0 else round(c['pnl'], 1)
    return c

def build_signals_scaled(json_details, signal_index, tp_factor=1.0, sl_factor=1.0):
    """Per-pattern TP×factor, SL×factor."""
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

def build_signals_uniform_tp(json_details, signal_index, uniform_tp, sl_factor=1.0):
    """Uniform TP for ALL patterns, SL per-pattern × factor."""
    tuples = []
    for key, det in json_details.items():
        pat = det['pattern']
        direction = det['direction']
        tp = uniform_tp
        sl = round(max(0.5, det['sl'] * sl_factor), 3)
        if pat in signal_index:
            for sig_bar in signal_index[pat]:
                tuples.append((sig_bar, pat, direction, tp, sl))
    return sorted(tuples, key=lambda x: x[0])

def expanding_window_folds(n, n_folds=3):
    folds = []
    for fi in range(n_folds):
        is_end = int(n * (fi + 1) / (n_folds + 1))
        oos_end = int(n * (fi + 2) / (n_folds + 1)) if fi < n_folds - 1 else n
        folds.append((is_end, oos_end))
    return folds

def wf_oos_total(signal_tuples, opens, highs, lows, closes, n, atr_ratio, ema_slope, n_folds=3):
    folds = expanding_window_folds(n, n_folds)
    oos_pnls = []
    for is_end, oos_end in folds:
        oos = run_npos(signal_tuples, opens, highs, lows, closes, n,
                       atr_ratio, ema_slope, is_end, oos_end)
        oos_pnls.append(oos['pnl'])
    return oos_pnls

def main():
    t0 = time.time()
    opens, highs, lows, closes, n, atr_ratio, ema_slope, signal_index, json_details = load_all()

    # Show TP distribution for context
    tps_orig = [det['tp'] for det in json_details.values()]
    tps_scaled = [round(max(0.3, det['tp'] * 0.5), 3) for det in json_details.values()]
    print(f"Data: {n} bars, {len(json_details)} patterns")
    print(f"Original TP range: {min(tps_orig):.2f} - {max(tps_orig):.2f}% (mean={np.mean(tps_orig):.2f}%)")
    print(f"Scaled TP×0.5:     {min(tps_scaled):.3f} - {max(tps_scaled):.3f}% (mean={np.mean(tps_scaled):.3f}%)")

    results = {}

    # ================================================================
    # PHASE 1: TP/SL Factor Direction Verification
    # ================================================================
    print(f"\n{'='*60}")
    print("PHASE 1: TP/SL Factor Direction Verification")
    print("="*60)

    # Verify: TP×0.5 should shrink TP, SL×1.1 should widen SL
    sample_key = list(json_details.keys())[0]
    sample = json_details[sample_key]
    orig_tp, orig_sl = sample['tp'], sample['sl']
    scaled_tp = round(max(0.3, orig_tp * 0.5), 3)
    scaled_sl = round(max(0.5, orig_sl * 1.1), 3)
    print(f"  Sample pattern: {sample_key}")
    print(f"  Original:  TP={orig_tp:.2f}%  SL={orig_sl:.2f}%")
    print(f"  After scaling (TP×0.5, SL×1.1): TP={scaled_tp:.3f}%  SL={scaled_sl:.3f}%")
    print(f"  TP shrunk: {scaled_tp < orig_tp} ({'✅ CORRECT' if scaled_tp < orig_tp else '❌ WRONG'})")
    print(f"  SL widened: {scaled_sl > orig_sl} ({'✅ CORRECT' if scaled_sl > orig_sl else '❌ WRONG'})")

    # Cross-check: what if factors were swapped?
    swapped_tp = round(max(0.3, orig_tp * 1.1), 3)
    swapped_sl = round(max(0.5, orig_sl * 0.5), 3)
    print(f"\n  If swapped (TP×1.1, SL×0.5): TP={swapped_tp:.3f}%  SL={swapped_sl:.3f}%")
    print(f"  (This would WIDEN TP and SHRINK SL — opposite intent)")

    sigs_correct = build_signals_scaled(json_details, signal_index, tp_factor=0.5, sl_factor=1.1)
    sigs_swapped = build_signals_scaled(json_details, signal_index, tp_factor=1.1, sl_factor=0.5)
    is_correct = run_npos(sigs_correct, opens, highs, lows, closes, n, atr_ratio, ema_slope, 0, n)
    is_swapped = run_npos(sigs_swapped, opens, highs, lows, closes, n, atr_ratio, ema_slope, 0, n)
    print(f"\n  Correct (TP×0.5 SL×1.1): PnL/MDD={is_correct['pnl_mdd']:.1f}x  PnL={is_correct['pnl']:+.1f}%  WR={is_correct['wr']:.1f}%")
    print(f"  Swapped (TP×1.1 SL×0.5): PnL/MDD={is_swapped['pnl_mdd']:.1f}x  PnL={is_swapped['pnl']:+.1f}%  WR={is_swapped['wr']:.1f}%")

    results['factor_verification'] = {
        'correct': {'pnl_mdd': is_correct['pnl_mdd'], 'pnl': round(is_correct['pnl'], 1)},
        'swapped': {'pnl_mdd': is_swapped['pnl_mdd'], 'pnl': round(is_swapped['pnl'], 1)},
        'tp_shrunk': scaled_tp < orig_tp,
        'sl_widened': scaled_sl > orig_sl,
    }

    # ================================================================
    # PHASE 2: Per-Pattern Scaled vs Uniform Flat TP (IS)
    # ================================================================
    print(f"\n{'='*60}")
    print("PHASE 2: Per-Pattern Scaled vs Uniform Flat TP (IS)")
    print("="*60)

    # Per-pattern TP×factor sweep
    print("\n  A) Per-pattern TP×factor (SL×1.1):")
    tp_factor_results = {}
    for tp_f in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]:
        sigs = build_signals_scaled(json_details, signal_index, tp_factor=tp_f, sl_factor=1.1)
        s = run_npos(sigs, opens, highs, lows, closes, n, atr_ratio, ema_slope, 0, n)
        marker = " ← current" if tp_f == 0.5 else ""
        print(f"    TP×{tp_f}: {s['trades']}t WR={s['wr']:.1f}% PnL={s['pnl']:+.1f}% MDD={s['mdd_mtm']:.2f}% PnL/MDD={s['pnl_mdd']:.1f}x{marker}")
        tp_factor_results[f'TPx{tp_f}'] = {
            'trades': s['trades'], 'wr': round(s['wr'], 1),
            'pnl': round(s['pnl'], 1), 'mdd': s['mdd_mtm'], 'pnl_mdd': s['pnl_mdd']}

    # Uniform flat TP sweep (same TP for all patterns)
    print("\n  B) Uniform flat TP (SL per-pattern ×1.1):")
    uniform_results = {}
    # Use the mean of scaled TPs as center, sweep around it
    mean_scaled_tp = np.mean(tps_scaled)
    for utp in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, round(mean_scaled_tp, 2)]:
        sigs = build_signals_uniform_tp(json_details, signal_index, uniform_tp=utp, sl_factor=1.1)
        s = run_npos(sigs, opens, highs, lows, closes, n, atr_ratio, ema_slope, 0, n)
        marker = " ← mean of per-pat scaled" if abs(utp - mean_scaled_tp) < 0.01 else ""
        print(f"    TP={utp:.2f}%: {s['trades']}t WR={s['wr']:.1f}% PnL={s['pnl']:+.1f}% MDD={s['mdd_mtm']:.2f}% PnL/MDD={s['pnl_mdd']:.1f}x{marker}")
        uniform_results[f'TP_flat_{utp}'] = {
            'trades': s['trades'], 'wr': round(s['wr'], 1),
            'pnl': round(s['pnl'], 1), 'mdd': s['mdd_mtm'], 'pnl_mdd': s['pnl_mdd']}

    results['is_comparison'] = {'per_pattern_scaled': tp_factor_results, 'uniform_flat': uniform_results}

    # ================================================================
    # PHASE 3: WF Comparison — Best per-pattern vs Best uniform
    # ================================================================
    print(f"\n{'='*60}")
    print("PHASE 3: WF Validation — Top Candidates")
    print("="*60)

    # Find best per-pattern factor
    best_pp = max(tp_factor_results.items(), key=lambda x: x[1]['pnl_mdd'])
    # Find best uniform
    best_uni = max(uniform_results.items(), key=lambda x: x[1]['pnl_mdd'])

    candidates = {
        f'PP_{best_pp[0]}_SL1.1': ('scaled', float(best_pp[0].replace('TPx', '')), 1.1),
        'PP_TPx0.5_SL1.1 (current)': ('scaled', 0.5, 1.1),
        f'UNI_{best_uni[0]}_SL1.1': ('uniform', float(best_uni[0].replace('TP_flat_', '')), 1.1),
        'UNI_TP0.5_SL1.1': ('uniform', 0.5, 1.1),
    }

    wf_comp = {}
    for name, (mode, tp_val, sl_f) in candidates.items():
        if mode == 'scaled':
            sigs = build_signals_scaled(json_details, signal_index, tp_factor=tp_val, sl_factor=sl_f)
        else:
            sigs = build_signals_uniform_tp(json_details, signal_index, uniform_tp=tp_val, sl_factor=sl_f)

        oos_pnls = wf_oos_total(sigs, opens, highs, lows, closes, n, atr_ratio, ema_slope, n_folds=3)
        total = sum(oos_pnls)
        passes = sum(1 for p in oos_pnls if p > 0)
        oos_pnls_r = [round(p, 1) for p in oos_pnls]
        print(f"  {name:35s}: {oos_pnls_r} Total={total:+.1f}% [{passes}/3]")
        wf_comp[name] = {'oos_pnls': oos_pnls_r, 'total': round(total, 1), 'passes': passes}

    results['wf_comparison'] = wf_comp

    # ================================================================
    # PHASE 4: Discrimination — Per-pattern scaling vs Uniform
    # ================================================================
    print(f"\n{'='*60}")
    print("PHASE 4: Per-Pattern vs Uniform Discrimination")
    print("="*60)

    # Test if per-pattern TP assignments matter (shuffle TP across patterns)
    real_sigs = build_signals_scaled(json_details, signal_index, tp_factor=0.5, sl_factor=1.1)
    real_stats = run_npos(real_sigs, opens, highs, lows, closes, n, atr_ratio, ema_slope, 0, n)

    np.random.seed(42)
    n_shuffles = 30
    random_pnls = []
    tp_values = [round(max(0.3, det['tp'] * 0.5), 3) for det in json_details.values()]

    for _ in range(n_shuffles):
        shuffled_tps = tp_values.copy()
        np.random.shuffle(shuffled_tps)
        shuffled_tuples = []
        for idx, (key, det) in enumerate(json_details.items()):
            pat = det['pattern']
            direction = det['direction']
            tp = shuffled_tps[idx % len(shuffled_tps)]
            sl = round(max(0.5, det['sl'] * 1.1), 3)
            if pat in signal_index:
                for sig_bar in signal_index[pat]:
                    shuffled_tuples.append((sig_bar, pat, direction, tp, sl))
        shuffled_tuples.sort(key=lambda x: x[0])
        s = run_npos(shuffled_tuples, opens, highs, lows, closes, n, atr_ratio, ema_slope, 0, n)
        random_pnls.append(s['pnl'])

    gap = real_stats['pnl'] - np.mean(random_pnls)
    gap_pct = gap / abs(np.mean(random_pnls)) * 100 if np.mean(random_pnls) != 0 else 0
    p_val = np.mean([1 for rp in random_pnls if rp >= real_stats['pnl']])

    print(f"  Real PnL (per-pattern TP×0.5): {real_stats['pnl']:+.1f}%")
    print(f"  Shuffled TP mean: {np.mean(random_pnls):+.1f}%")
    print(f"  Gap: {gap:+.1f}% ({gap_pct:+.1f}%)  p={p_val:.3f}")
    print(f"  → {'DISCRIMINATING: Per-pattern TP matters' if p_val < 0.05 else 'NON-DISC: Uniform TP is equally good'}")

    results['tp_assignment_disc'] = {
        'real_pnl': round(real_stats['pnl'], 1),
        'shuffled_mean': round(np.mean(random_pnls), 1),
        'gap': round(gap, 1),
        'gap_pct': round(gap_pct, 1),
        'p_value': round(p_val, 3),
        'verdict': 'DISCRIMINATING' if p_val < 0.05 else 'NON-DISCRIMINATING',
    }

    # ================================================================
    # SUMMARY
    # ================================================================
    print(f"\n{'='*60}")
    print("SUMMARY")
    print("="*60)

    best_pp_is = max(tp_factor_results.items(), key=lambda x: x[1]['pnl_mdd'])
    best_uni_is = max(uniform_results.items(), key=lambda x: x[1]['pnl_mdd'])
    best_wf = max(wf_comp.items(), key=lambda x: x[1]['total'])

    print(f"\n  Best per-pattern scaled (IS): {best_pp_is[0]} → PnL/MDD={best_pp_is[1]['pnl_mdd']:.1f}x")
    print(f"  Best uniform flat (IS):       {best_uni_is[0]} → PnL/MDD={best_uni_is[1]['pnl_mdd']:.1f}x")
    print(f"  Best WF OOS total:            {best_wf[0]} → {best_wf[1]['total']:+.1f}%")
    print(f"  TP assignment discrimination:  {results['tp_assignment_disc']['verdict']}")

    if results['tp_assignment_disc']['verdict'] == 'NON-DISCRIMINATING':
        print(f"\n  → Per-pattern TP 할당은 의미 없음. Uniform flat TP가 동등하거나 더 나을 수 있음.")
        print(f"     Uniform이 더 간단하고 설명 가능 → 검토 필요")
    else:
        print(f"\n  → Per-pattern TP 할당이 유의미. 패턴별 MAE/MFE 최적화 + scaling 유지 권장")

    out = os.path.join(os.path.dirname(__file__), '..', '..', 'results', 'tp_uniform_vs_scaled.json')
    with open(out, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved: {os.path.abspath(out)}")
    print(f"Elapsed: {time.time()-t0:.0f}s")

if __name__ == '__main__':
    main()
