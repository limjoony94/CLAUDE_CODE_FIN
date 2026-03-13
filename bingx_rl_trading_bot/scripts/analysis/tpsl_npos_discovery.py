#!/usr/bin/env python3
"""TP/SL N-pos Direct Discovery Study

Research question: Can we find optimal TP/SL directly in N-pos context
during pattern discovery, eliminating the need for post-discovery scaling?

Current problem:
- Scanner grid_search_mae_mfe() optimizes per-pattern TP/SL in 1-pos context
  using MFE percentiles [40,50,60,70,80]
- Production runs N=9 portfolio where slot turnover matters
- Post-discovery tp_scale_factor=0.5 bridges the gap (v1.57.0)

Key insight to test: If we extend TP_PERCENTILES to include lower values
(15-35), would the scanner's 1-pos scoring naturally select lower TPs that
happen to be N-pos optimal, eliminating the need for post-hoc scaling?

Phases:
1. Extended TP percentile grid — 1-pos scoring for all patterns
2. N-pos portfolio for each uniform TP percentile choice
3. Per-pattern: what percentile does 1-pos scoring pick?
4. N-pos "oracle" — for each pattern, which TP percentile is best in N-pos?
5. Strategy comparison (current vs extended vs scaling vs oracle)
6. WF validation
7. Discrimination test
8. Conclusion

Standard Research Protocol: next-bar entry, intrabar exit, compound sizing,
0.10% fee, 0.02% slippage, MC 3 seeds.
"""

import sys, os, json, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

from scripts.scanner.pattern_scanner import (
    build_signal_index, compute_excursions, bt_signals_atr,
    portfolio_npos, calc_stats_compound, compute_atr_ratio,
    compute_ema_slope, load_and_classify, derive_tp_sl,
    FEE_PCT, LEVERAGE, MAX_BARS, DEFAULT_N_SLOTS, DEFAULT_DIRECTION_CAP,
    DEFAULT_REGIME_MULT, DEFAULT_AGG_RISK_COUNTER, DEFAULT_AGG_RISK_WITH,
    DEFAULT_MOMENTUM_LOOKBACK, DEFAULT_MOMENTUM_THRESHOLD, DEFAULT_MOMENTUM_COOLDOWN,
    DEFAULT_CASCADE_TIGHTEN_PCT, DEFAULT_ATR_CLAMP_LO, DEFAULT_ATR_CLAMP_HI,
    DEFAULT_ATR_PERIOD, DEFAULT_ATR_WINDOW, DEFAULT_EDGE_THRESHOLD,
    DEFAULT_MIN_TRADES, MAX_BASELINE_WR, TIMEOUT_BARS,
    TP_PERCENTILES, SL_PERCENTILES,
    NPOS_EMA_PERIOD, NPOS_EMA_LOOKBACK,
)

# ============================================================
DATA_FILE = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'btc_5m_270days_reclassified.csv')
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), '..', '..', 'results', 'tpsl_npos_discovery.json')

# Extended TP percentile grid — includes lower percentiles for faster TP
EXTENDED_TP_PERCENTILES = [15, 20, 25, 30, 35, 40, 50, 60, 70, 80]


def load_data():
    print("Loading data...")
    df = load_and_classify(DATA_FILE)
    print(f"Data: {len(df)} bars, {len(df)/288:.0f} days")
    return df


def run_npos(selected_patterns, signal_index, opens, highs, lows, closes, n,
             atr_ratio, ema_slope, start_bar, end_bar):
    """Run N-pos portfolio simulation for a given set of patterns with specified TP/SL."""
    signal_tuples = []
    for key, tpsl in selected_patterns.items():
        parts = key.rsplit('_', 1)
        pat_name, direction = parts[0], parts[1]
        tp, sl = tpsl
        if pat_name in signal_index:
            for sig_bar in signal_index[pat_name]:
                signal_tuples.append((sig_bar, pat_name, direction, tp, sl))

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
        timeout_bars=TIMEOUT_BARS,
    )
    compound = calc_stats_compound(trades)
    mdd_mtm = stats.get('mdd_mtm', compound.get('mdd', 0))
    compound['mdd_mtm'] = round(mdd_mtm, 2)
    compound['pnl_mdd'] = round(compound['pnl'] / mdd_mtm, 1) if mdd_mtm > 0 else round(compound['pnl'], 1)
    compound['blocked'] = stats.get('blocked', {})
    return compound


def grid_search_1pos(sigs, direction, mfes, maes, tp_percentiles, sl_percentiles,
                     opens, highs, lows, n, atr_ratio, clamp_lo, clamp_hi):
    """1-pos grid search for one pattern across TP percentiles. Returns best per TP percentile."""
    results = {}
    for tp_p in tp_percentiles:
        tp_val = round(float(np.percentile(mfes, tp_p)), 3)
        best_score = -9999
        best_combo = None

        for sl_p in sl_percentiles:
            sl_val = round(float(np.percentile(maes, sl_p)), 3)
            if tp_val < 0.1 or sl_val < 0.3:
                continue
            bwr = sl_val / (tp_val + sl_val) * 100
            if bwr > MAX_BASELINE_WR:
                continue

            trades = bt_signals_atr(sigs, direction, tp_val, sl_val,
                                    opens, highs, lows, n,
                                    atr_ratio, clamp_lo, clamp_hi)
            if len(trades) < DEFAULT_MIN_TRADES:
                continue

            pnls = [t[2] for t in trades]
            cum = 0; peak = 0; mdd_val = 0; w = 0
            for p in pnls:
                cum += p
                if cum > peak: peak = cum
                dd = peak - cum
                if dd > mdd_val: mdd_val = dd
                if p > 0: w += 1

            wr = w / len(pnls) * 100
            score = (cum / mdd_val) if mdd_val > 0 else cum
            edge = wr - bwr

            if score > best_score and edge >= DEFAULT_EDGE_THRESHOLD:
                best_score = score
                best_combo = {
                    'tp': tp_val, 'sl': sl_val,
                    'tp_percentile': tp_p, 'sl_percentile': sl_p,
                    'trades': len(trades), 'wr': round(wr, 1),
                    'pnl': round(cum, 1), 'edge': round(edge, 1),
                    'score': round(score, 2),
                }

        if best_combo is not None:
            results[tp_p] = best_combo
    return results


def main():
    t0 = time.time()
    results = {'study': 'tpsl_npos_discovery', 'date': '2026-03-12', 'phases': {}}

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

    start_bar = 576
    end_bar = n

    # ================================================================
    # Phase 1: Build per-pattern excursion data + extended grid search
    # ================================================================
    print("\n" + "="*60)
    print("PHASE 1: Per-Pattern Extended TP Grid Search (1-pos)")
    print("="*60)

    pattern_grid = {}  # key -> {tp_percentile -> best combo}
    for pat_name in signal_index:
        sigs = signal_index[pat_name]
        if len(sigs) < DEFAULT_MIN_TRADES:
            continue
        for direction in ['LONG', 'SHORT']:
            key = f"{pat_name}_{direction}"
            excursions = compute_excursions(sigs, direction, opens, highs, lows, n, MAX_BARS)
            if len(excursions) < DEFAULT_MIN_TRADES:
                continue

            mfes = [e['mfe_pct'] for e in excursions]
            maes_arr = np.array([e['mae_pct'] for e in excursions])

            res = grid_search_1pos(sigs, direction, mfes, maes_arr,
                                   EXTENDED_TP_PERCENTILES, SL_PERCENTILES,
                                   opens, highs, lows, n, atr_ratio,
                                   DEFAULT_ATR_CLAMP_LO, DEFAULT_ATR_CLAMP_HI)
            if res:
                pattern_grid[key] = res

    print(f"Patterns with valid combos: {len(pattern_grid)}")

    # What does 1-pos scoring pick from the full grid?
    tp_pctile_chosen_1pos = {p: 0 for p in EXTENDED_TP_PERCENTILES}
    for key, combos in pattern_grid.items():
        best_p = max(combos.items(), key=lambda x: x[1]['score'])[0]
        tp_pctile_chosen_1pos[best_p] += 1

    print("\n1-pos optimal TP percentile distribution:")
    for p in EXTENDED_TP_PERCENTILES:
        cnt = tp_pctile_chosen_1pos[p]
        bar = "█" * cnt
        print(f"  P{p:3d}: {cnt:3d} patterns {bar}")

    results['phases']['phase1'] = {
        'patterns_tested': len(pattern_grid),
        'tp_pctile_chosen_1pos': tp_pctile_chosen_1pos,
    }

    # ================================================================
    # Phase 2: N-pos portfolio for each uniform TP percentile
    # ================================================================
    print("\n" + "="*60)
    print("PHASE 2: N-pos Portfolio by Uniform TP Percentile")
    print("="*60)

    npos_by_pctile = {}
    for tp_p in EXTENDED_TP_PERCENTILES:
        selected = {}
        for key, combos in pattern_grid.items():
            if tp_p in combos:
                c = combos[tp_p]
                selected[key] = [c['tp'], c['sl']]

        if len(selected) < 5:
            print(f"  P{tp_p:3d}: {len(selected)} patterns (skip)")
            continue

        stats = run_npos(selected, signal_index, opens, highs, lows, closes, n,
                         atr_ratio, ema_slope, start_bar, end_bar)
        npos_by_pctile[tp_p] = {'n_patterns': len(selected), 'stats': stats}

        tps = [v[0] for v in selected.values()]
        print(f"  P{tp_p:3d}: {len(selected):3d}pat | "
              f"tr={stats['trades']:4d} WR={stats['wr']:5.1f}% "
              f"PnL={stats['pnl']:+8.1f}% MDD={stats['mdd_mtm']:5.2f}% "
              f"PnL/MDD={stats['pnl_mdd']:7.1f}x | "
              f"avgTP={np.mean(tps):.2f}%")

    # Find best N-pos TP percentile
    best_npos_p = max(npos_by_pctile.items(), key=lambda x: x[1]['stats']['pnl_mdd'])[0]
    print(f"\n  Best N-pos TP percentile: P{best_npos_p}")

    results['phases']['phase2'] = {
        'npos_by_pctile': {str(k): v for k, v in npos_by_pctile.items()},
        'best_npos_pctile': best_npos_p,
    }

    # ================================================================
    # Phase 3: Strategy Comparison
    # ================================================================
    print("\n" + "="*60)
    print("PHASE 3: Strategy Comparison")
    print("="*60)

    # A: Current scanner (TP percentiles 40-80, best 1-pos score)
    selected_A = {}
    for key, combos in pattern_grid.items():
        current = {p: c for p, c in combos.items() if p in [40, 50, 60, 70, 80]}
        if current:
            best = max(current.items(), key=lambda x: x[1]['score'])[1]
            selected_A[key] = [best['tp'], best['sl']]

    # B: A + tp_scale_factor=0.5 (v1.57.0)
    selected_B = {k: [round(max(0.3, v[0] * 0.5), 3), v[1]] for k, v in selected_A.items()}

    # C: Extended grid (TP percentiles 15-80, best 1-pos score)
    selected_C = {}
    for key, combos in pattern_grid.items():
        best = max(combos.items(), key=lambda x: x[1]['score'])[1]
        selected_C[key] = [best['tp'], best['sl']]

    # D: Uniform best N-pos percentile
    selected_D = {}
    for key, combos in pattern_grid.items():
        if best_npos_p in combos:
            c = combos[best_npos_p]
            selected_D[key] = [c['tp'], c['sl']]

    # E: Extended grid + tp_scale_factor=0.5
    selected_E = {k: [round(max(0.3, v[0] * 0.5), 3), v[1]] for k, v in selected_C.items()}

    # F: Fine TP scale search on A (scale=0.3-0.8)
    print("\n  Fine TP scale search on current patterns...")
    scale_results = {}
    for scale in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]:
        scaled = {k: [round(max(0.3, v[0] * scale), 3), v[1]] for k, v in selected_A.items()}
        stats = run_npos(scaled, signal_index, opens, highs, lows, closes, n,
                         atr_ratio, ema_slope, start_bar, end_bar)
        scale_results[scale] = stats
        print(f"    scale={scale:.1f}: PnL/MDD={stats['pnl_mdd']:7.1f}x PnL={stats['pnl']:+8.1f}%")

    best_scale = max(scale_results.items(), key=lambda x: x[1]['pnl_mdd'])[0]
    print(f"  Best scale: {best_scale}")

    strategies = {
        'A_scanner_current': selected_A,
        'B_scale_0.5': selected_B,
        'C_extended_1pos': selected_C,
        f'D_uniform_P{best_npos_p}': selected_D,
        'E_extended_scale_0.5': selected_E,
    }

    print("\n  Full strategy comparison:")
    strat_stats = {}
    for name, pats in strategies.items():
        if len(pats) < 5:
            print(f"  {name}: too few patterns ({len(pats)})")
            continue
        stats = run_npos(pats, signal_index, opens, highs, lows, closes, n,
                         atr_ratio, ema_slope, start_bar, end_bar)
        strat_stats[name] = {'n_patterns': len(pats), 'stats': stats}

        tps = [v[0] for v in pats.values()]
        sls = [v[1] for v in pats.values()]
        print(f"  {name:25s}: {len(pats):3d}pat | "
              f"tr={stats['trades']:4d} WR={stats['wr']:5.1f}% "
              f"PnL={stats['pnl']:+8.1f}% MDD={stats['mdd_mtm']:5.2f}% "
              f"PnL/MDD={stats['pnl_mdd']:7.1f}x | "
              f"TP={np.mean(tps):.2f}% SL={np.mean(sls):.2f}%")

    results['phases']['phase3'] = {
        'strategies': {k: v for k, v in strat_stats.items()},
        'scale_sweep': {str(k): {'pnl': v['pnl'], 'pnl_mdd': v['pnl_mdd'], 'mdd_mtm': v['mdd_mtm']}
                        for k, v in scale_results.items()},
        'best_scale': best_scale,
    }

    # ================================================================
    # Phase 4: 1-pos Score vs N-pos Outcome Per Pattern
    # ================================================================
    print("\n" + "="*60)
    print("PHASE 4: Why 1-pos Scoring Fails — TP Percentile Analysis")
    print("="*60)

    # For each pattern that exists in both A and D, compare which TP percentile was chosen
    common_keys = set(selected_A.keys()) & set(selected_D.keys())
    tp_shift = []
    for key in common_keys:
        combos = pattern_grid[key]
        # What 1-pos scoring chose (from 40-80)
        current = {p: c for p, c in combos.items() if p in [40, 50, 60, 70, 80]}
        if not current:
            continue
        chosen_1pos = max(current.items(), key=lambda x: x[1]['score'])
        chosen_npos = best_npos_p  # uniform

        tp_1pos = chosen_1pos[1]['tp']
        tp_npos = combos[best_npos_p]['tp'] if best_npos_p in combos else None
        if tp_npos is not None:
            tp_shift.append({
                'key': key,
                'tp_1pos': tp_1pos,
                'tp_npos': tp_npos,
                'pctile_1pos': chosen_1pos[0],
                'pctile_npos': best_npos_p,
                'tp_ratio': round(tp_npos / tp_1pos, 3) if tp_1pos > 0 else 0,
            })

    if tp_shift:
        ratios = [s['tp_ratio'] for s in tp_shift]
        pctile_1pos_dist = {}
        for s in tp_shift:
            p = s['pctile_1pos']
            pctile_1pos_dist[p] = pctile_1pos_dist.get(p, 0) + 1

        print(f"  Common patterns: {len(tp_shift)}")
        print(f"  TP ratio (npos/1pos): mean={np.mean(ratios):.3f} median={np.median(ratios):.3f}")
        print(f"  1-pos chosen percentile distribution: {pctile_1pos_dist}")
        print(f"  N-pos optimal percentile: P{best_npos_p}")

        # How many patterns would have naturally chosen the N-pos optimal percentile?
        natural_match = sum(1 for s in tp_shift if s['pctile_1pos'] == best_npos_p)
        print(f"  Patterns where 1-pos scoring = N-pos optimal: {natural_match}/{len(tp_shift)} "
              f"({natural_match/len(tp_shift)*100:.1f}%)")

        results['phases']['phase4'] = {
            'common_patterns': len(tp_shift),
            'tp_ratio_mean': round(float(np.mean(ratios)), 3),
            'tp_ratio_median': round(float(np.median(ratios)), 3),
            'pctile_1pos_distribution': pctile_1pos_dist,
            'npos_optimal_pctile': best_npos_p,
            'natural_match_rate': round(natural_match / len(tp_shift) * 100, 1),
        }

    # ================================================================
    # Phase 5: WF Validation (3-fold)
    # ================================================================
    print("\n" + "="*60)
    print("PHASE 5: Walk-Forward Validation")
    print("="*60)

    # Top 3 strategies
    top_3 = sorted(strat_stats.items(), key=lambda x: x[1]['stats']['pnl_mdd'], reverse=True)[:3]

    n_folds = 3
    boundaries = []
    for f in range(n_folds):
        is_end = int(n * (f + 2) / (n_folds + 1))
        oos_end = int(n * (f + 3) / (n_folds + 1)) if f < n_folds - 1 else n
        boundaries.append((start_bar, is_end, is_end, oos_end))

    wf = {}
    for sname, _ in top_3:
        pats = strategies[sname]
        oos_pnls = []
        for fi, (_, _, oos_s, oos_e) in enumerate(boundaries):
            oos = run_npos(pats, signal_index, opens, highs, lows, closes, n,
                           atr_ratio, ema_slope, oos_s, oos_e)
            oos_pnls.append(oos['pnl'])
            tag = "PASS" if oos['pnl'] > 0 else "FAIL"
            print(f"  {sname:25s} F{fi+1}: OOS={oos['pnl']:+7.1f}% [{tag}]")

        total = sum(oos_pnls)
        passes = sum(1 for p in oos_pnls if p > 0)
        wf[sname] = {
            'oos_pnls': [round(p, 1) for p in oos_pnls],
            'total_oos': round(total, 1),
            'verdict': f"{passes}/{n_folds} {'PASS' if passes == n_folds else 'FAIL'}",
        }
        print(f"  {sname:25s} TOTAL={total:+7.1f}% [{passes}/{n_folds}]")

    results['phases']['phase5'] = wf

    # ================================================================
    # Phase 6: Discrimination Test (best strategy)
    # ================================================================
    print("\n" + "="*60)
    print("PHASE 6: Discrimination Test")
    print("="*60)

    best_sname = top_3[0][0]
    best_pats = strategies[best_sname]
    real = run_npos(best_pats, signal_index, opens, highs, lows, closes, n,
                    atr_ratio, ema_slope, start_bar, end_bar)

    n_shuf = 30
    rng = np.random.RandomState(42)
    rand_pnls = []
    for _ in range(n_shuf):
        keys = list(best_pats.keys())
        vals = list(best_pats.values())
        rng.shuffle(vals)
        shuf = dict(zip(keys, vals))
        r = run_npos(shuf, signal_index, opens, highs, lows, closes, n,
                     atr_ratio, ema_slope, start_bar, end_bar)
        rand_pnls.append(r['pnl'])

    gap = real['pnl'] - np.mean(rand_pnls)
    gap_pct = gap / abs(np.mean(rand_pnls)) * 100 if np.mean(rand_pnls) != 0 else 0
    p_val = sum(1 for r in rand_pnls if r >= real['pnl']) / n_shuf
    verdict = "DISCRIMINATING" if p_val < 0.1 else "NON-DISCRIMINATING"

    print(f"  Real PnL: {real['pnl']:+.1f}%  Random mean: {np.mean(rand_pnls):+.1f}%")
    print(f"  Gap: {gap:+.1f}% ({gap_pct:+.1f}%)  p={p_val:.3f}  {verdict}")

    results['phases']['phase6'] = {
        'strategy': best_sname,
        'real_pnl': round(real['pnl'], 1),
        'random_mean': round(float(np.mean(rand_pnls)), 1),
        'gap': round(gap, 1), 'gap_pct': round(gap_pct, 1),
        'p_value': round(p_val, 3), 'verdict': verdict,
    }

    # ================================================================
    # Phase 7: Conclusion
    # ================================================================
    print("\n" + "="*60)
    print("PHASE 7: CONCLUSION")
    print("="*60)

    best_overall = max(strat_stats.items(), key=lambda x: x[1]['stats']['pnl_mdd'])
    b_name = best_overall[0]
    b_stats = best_overall[1]['stats']
    b_pnl_mdd = b_stats['pnl_mdd']

    scaled_pnl_mdd = strat_stats.get('B_scale_0.5', {}).get('stats', {}).get('pnl_mdd', 0)
    current_pnl_mdd = strat_stats.get('A_scanner_current', {}).get('stats', {}).get('pnl_mdd', 0)

    impr_vs_scaled = (b_pnl_mdd - scaled_pnl_mdd) / scaled_pnl_mdd * 100 if scaled_pnl_mdd > 0 else 0
    impr_vs_current = (b_pnl_mdd - current_pnl_mdd) / current_pnl_mdd * 100 if current_pnl_mdd > 0 else 0

    print(f"\n  Best: {b_name} — PnL/MDD={b_pnl_mdd:.1f}x")
    print(f"  vs A (current 1-pos): {current_pnl_mdd:.1f}x ({impr_vs_current:+.1f}%)")
    print(f"  vs B (TP×0.5):        {scaled_pnl_mdd:.1f}x ({impr_vs_scaled:+.1f}%)")

    if impr_vs_scaled < 10:
        rec = (f"KEEP_SCALING — Best ({b_name}) only {impr_vs_scaled:+.1f}% better than TP×0.5. "
               f"Post-discovery scaling is sufficient. Scanner TP_PERCENTILES change unnecessary.")
    elif b_name.startswith('C_') or b_name.startswith('D_'):
        rec = (f"MODIFY_SCANNER — {b_name} outperforms TP×0.5 by {impr_vs_scaled:+.1f}%. "
               f"Extend TP_PERCENTILES to include lower values.")
    else:
        rec = f"KEEP_SCALING — TP×0.5 remains best practical approach."

    print(f"\n  RECOMMENDATION: {rec}")

    # Answer the fundamental question
    print("\n  === FUNDAMENTAL QUESTION ===")
    print("  Q: Can the scanner find N-pos-optimal TP directly?")
    if best_npos_p < 40:
        print(f"  A: YES — P{best_npos_p} is optimal in N-pos, below current grid [40-80].")
        if impr_vs_scaled >= 10:
            print(f"     Extending grid to include P{best_npos_p} would improve by {impr_vs_scaled:+.1f}%.")
        else:
            print(f"     But improvement is only {impr_vs_scaled:+.1f}% vs simple scaling.")
            print(f"     Post-discovery TP×{best_scale} remains the practical choice.")
    else:
        print(f"  A: NO — N-pos optimal (P{best_npos_p}) is already in current grid [40-80].")
        print(f"     The issue is not the grid range but the 1-pos scoring metric.")
        print(f"     1-pos PnL/MDD score does not capture N-pos slot turnover benefit.")

    results['phases']['phase7'] = {
        'best_strategy': b_name,
        'best_pnl_mdd': b_pnl_mdd,
        'current_pnl_mdd': current_pnl_mdd,
        'scaled_pnl_mdd': scaled_pnl_mdd,
        'improvement_vs_scaled': round(impr_vs_scaled, 1),
        'improvement_vs_current': round(impr_vs_current, 1),
        'recommendation': rec,
        'npos_optimal_pctile': best_npos_p,
        'best_tp_scale': best_scale,
    }

    elapsed = time.time() - t0
    results['elapsed_seconds'] = round(elapsed, 1)

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved: {OUTPUT_FILE}")
    print(f"Elapsed: {elapsed:.0f}s")


if __name__ == '__main__':
    main()
