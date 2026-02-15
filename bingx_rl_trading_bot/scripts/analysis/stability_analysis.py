#!/usr/bin/env python3
"""
TP/SL Stability Analysis — 안정/불안정 패턴 분류 및 하이브리드 전략 설계

목적:
1. 47패턴을 TP/SL 안정도별 분류 (Stable / Moderate / Unstable)
2. Hybrid 전략 설계: Stable→per-pattern, Unstable→universal
3. Expanding Window WF로 Hybrid vs Universal vs Per-Pattern 비교
4. 최적 stability threshold 탐색

Production classify_candle 사용, LEVERAGE=3, timeout DROP
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

FEE_PCT = 0.10
LEVERAGE = 3
MAX_BARS = 500
MC_SIMS = 5000
MC_SEEDS = [42, 123, 7777]

DATA_FILE = os.path.join(_PROJECT_ROOT, 'data', 'btc_5m_270days_reclassified.csv')
DYNAMIC_FILE = os.path.join(_PROJECT_ROOT, 'results', 'dynamic_patterns.json')
OUTPUT_FILE = os.path.join(_PROJECT_ROOT, 'results', 'stability_analysis.json')

TP_COARSE = [0.5, 0.7, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.1, 2.5, 3.0]
SL_COARSE = [0.5, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]

UNIVERSAL_TP = 2.0
UNIVERSAL_SL = 3.0


def load_data():
    df = pd.read_csv(DATA_FILE)
    with open(DYNAMIC_FILE) as f:
        dp = json.load(f)
    return df, dp


def classify_all(df):
    df['body'] = abs(df['close'] - df['open'])
    df['avg_body'] = df['body'].rolling(AVG_BODY_WINDOW, min_periods=1).mean()
    types = []
    for i, row in df.iterrows():
        ct = classify_candle(row, df.at[i, 'avg_body'])
        types.append(ct.value)
    return types, df['open'].values, df['high'].values, df['low'].values, df['close'].values


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


def calc_compound(trades):
    if not trades:
        return {'pnl': 0, 'trades': 0, 'wr': 0, 'mdd': 0, 'pf': 0, 'pnl_mdd': 0,
                'avg_hold': 0, 'max_consec_loss': 0}
    equity = 100.0
    peak = 100.0
    mdd = 0.0
    wins = []
    losses = []
    holds = []
    cl = 0
    max_cl = 0
    for eb, xb, p in trades:
        equity *= (1 + p / 100)
        if equity > peak:
            peak = equity
        dd = (peak - equity) / peak * 100
        if dd > mdd:
            mdd = dd
        if p > 0:
            wins.append(p)
            cl = 0
        else:
            losses.append(p)
            cl += 1
            if cl > max_cl:
                max_cl = cl
        holds.append((xb - eb) * 5 / 60)
    total_pnl = equity - 100
    wsum = sum(wins)
    lsum = sum(abs(x) for x in losses)
    return {
        'pnl': round(total_pnl, 1),
        'trades': len(trades),
        'wr': round(len(wins) / len(trades) * 100, 1),
        'mdd': round(mdd, 1),
        'pf': round(wsum / lsum, 2) if lsum > 0 else 999,
        'pnl_mdd': round(total_pnl / mdd, 1) if mdd > 0 else 999,
        'avg_hold': round(np.mean(holds), 1) if holds else 0,
        'max_consec_loss': max_cl,
    }


def grid_search_pattern(signal_bars, direction, opens, highs, lows, n_bars, min_trades=10):
    best = None
    best_score = -9999
    for tp in TP_COARSE:
        for sl in SL_COARSE:
            if sl < 0.5:
                continue
            trades = bt_signals(signal_bars, direction, tp, sl, opens, highs, lows, n_bars)
            if len(trades) < min_trades:
                continue
            stats = calc_compound(trades)
            score = stats['pnl_mdd'] if stats['mdd'] > 0 else stats['pnl']
            if score > best_score:
                best_score = score
                best = {'tp': tp, 'sl': sl, 'trades': stats['trades'],
                        'wr': stats['wr'], 'pnl': stats['pnl'],
                        'mdd': stats['mdd'], 'pnl_mdd': stats['pnl_mdd'], 'pf': stats['pf']}
    return best


def mc_test_multi_seed(pnls, seeds=MC_SEEDS, n_sims=MC_SIMS):
    if len(pnls) < 5:
        return 1.0
    pnls_arr = np.array(pnls)
    actual = np.sum(pnls_arr)
    p_values = []
    for seed in seeds:
        rng = np.random.RandomState(seed)
        signs = rng.choice([-1, 1], size=(n_sims, len(pnls_arr)))
        rand_sums = signs @ pnls_arr
        p_values.append(float(np.mean(rand_sums >= actual)))
    return max(p_values)


def compute_stability(signal_index, pat, direction, opens, highs, lows, n_bars, n_folds=4):
    """
    Compute TP/SL stability across 4 expanding windows.
    More folds = finer stability measurement.
    Returns stability metrics.
    """
    fold_size = n_bars // (n_folds + 1)
    fold_optima = []

    for fold in range(n_folds):
        train_end = (fold + 1) * fold_size
        if pat not in signal_index:
            continue
        train_bars = [b for b in signal_index[pat] if b < train_end]
        if len(train_bars) < 10:
            fold_optima.append(None)
            continue
        opt = grid_search_pattern(train_bars, direction, opens, highs, lows, n_bars, min_trades=10)
        fold_optima.append(opt)

    valid = [o for o in fold_optima if o is not None]
    if len(valid) < 2:
        return {
            'fold_optima': fold_optima,
            'n_valid_folds': len(valid),
            'tp_range': None, 'sl_range': None,
            'tp_std': None, 'sl_std': None,
            'category': 'INSUFFICIENT_DATA',
            'stability_score': 0,
        }

    tps = [o['tp'] for o in valid]
    sls = [o['sl'] for o in valid]
    tp_range = max(tps) - min(tps)
    sl_range = max(sls) - min(sls)
    tp_std = float(np.std(tps))
    sl_std = float(np.std(sls))

    # Stability score: lower = more stable (0 = perfectly stable)
    # Normalize: TP range / TP_max_possible + SL range / SL_max_possible
    stability_score = tp_range / 3.0 + sl_range / 4.0  # max TP=3.0, max SL=4.0

    if tp_range <= 0.3 and sl_range <= 0.5:
        category = 'STABLE'
    elif tp_range <= 0.7 and sl_range <= 1.5:
        category = 'MODERATE'
    else:
        category = 'UNSTABLE'

    # Check if last 2 folds converge (recent stability)
    if len(valid) >= 2:
        last2_tp_range = abs(valid[-1]['tp'] - valid[-2]['tp'])
        last2_sl_range = abs(valid[-1]['sl'] - valid[-2]['sl'])
        recent_convergence = last2_tp_range <= 0.3 and last2_sl_range <= 0.5
    else:
        recent_convergence = False
        last2_tp_range = None
        last2_sl_range = None

    return {
        'fold_optima': [{'tp': o['tp'], 'sl': o['sl'], 'pnl': o['pnl'],
                         'wr': o['wr'], 'trades': o['trades']} if o else None for o in fold_optima],
        'n_valid_folds': len(valid),
        'tp_range': round(tp_range, 2),
        'sl_range': round(sl_range, 2),
        'tp_std': round(tp_std, 3),
        'sl_std': round(sl_std, 3),
        'stability_score': round(stability_score, 3),
        'category': category,
        'recent_convergence': recent_convergence,
        'last2_tp_range': round(last2_tp_range, 2) if last2_tp_range is not None else None,
        'last2_sl_range': round(last2_sl_range, 2) if last2_sl_range is not None else None,
    }


def expanding_wf_hybrid(signal_index, long_pats, short_pats,
                        opens, highs, lows, n_bars, n_folds=3,
                        stable_pats=None, mode='universal'):
    """
    Expanding Window WF with hybrid mode support.
    mode: 'universal' | 'per_pattern' | 'hybrid'
    stable_pats: set of patterns that use per-pattern TP/SL (hybrid mode)
    """
    fold_size = n_bars // (n_folds + 1)
    results = []

    for fold in range(n_folds):
        train_end = (fold + 1) * fold_size
        test_start = train_end
        test_end = (fold + 2) * fold_size if fold < n_folds - 1 else n_bars

        all_test_trades = []

        for pat in long_pats:
            if pat not in signal_index:
                continue
            test_bars = [b for b in signal_index[pat] if test_start <= b < test_end]
            if not test_bars:
                continue

            use_pp = False
            if mode == 'per_pattern':
                use_pp = True
            elif mode == 'hybrid' and stable_pats and pat in stable_pats:
                use_pp = True

            if use_pp:
                train_bars = [b for b in signal_index[pat] if b < train_end]
                if len(train_bars) >= 10:
                    opt = grid_search_pattern(train_bars, 'LONG', opens, highs, lows, n_bars, min_trades=10)
                    if opt:
                        trades = bt_signals(test_bars, 'LONG', opt['tp'], opt['sl'],
                                            opens, highs, lows, n_bars)
                        all_test_trades.extend(trades)
                        continue
                # Fallback to universal
                trades = bt_signals(test_bars, 'LONG', UNIVERSAL_TP, UNIVERSAL_SL,
                                    opens, highs, lows, n_bars)
            else:
                trades = bt_signals(test_bars, 'LONG', UNIVERSAL_TP, UNIVERSAL_SL,
                                    opens, highs, lows, n_bars)
            all_test_trades.extend(trades)

        for pat in short_pats:
            if pat not in signal_index:
                continue
            test_bars = [b for b in signal_index[pat] if test_start <= b < test_end]
            if not test_bars:
                continue

            use_pp = False
            if mode == 'per_pattern':
                use_pp = True
            elif mode == 'hybrid' and stable_pats and pat in stable_pats:
                use_pp = True

            if use_pp:
                train_bars = [b for b in signal_index[pat] if b < train_end]
                if len(train_bars) >= 10:
                    opt = grid_search_pattern(train_bars, 'SHORT', opens, highs, lows, n_bars, min_trades=10)
                    if opt:
                        trades = bt_signals(test_bars, 'SHORT', opt['tp'], opt['sl'],
                                            opens, highs, lows, n_bars)
                        all_test_trades.extend(trades)
                        continue
                trades = bt_signals(test_bars, 'SHORT', UNIVERSAL_TP, UNIVERSAL_SL,
                                    opens, highs, lows, n_bars)
            else:
                trades = bt_signals(test_bars, 'SHORT', UNIVERSAL_TP, UNIVERSAL_SL,
                                    opens, highs, lows, n_bars)
            all_test_trades.extend(trades)

        portfolio = portfolio_1pos(all_test_trades)
        stats = calc_compound(portfolio)
        results.append({
            'fold': fold,
            'train_range': f'0-{train_end}',
            'test_range': f'{test_start}-{test_end}',
            **stats,
        })

    return results


def main():
    t0 = time.time()
    df, dp = load_data()
    types, opens, highs, lows, closes = classify_all(df)
    n_bars = len(df)

    long_pats = dp['patterns']['long']
    short_pats = dp['patterns']['short']
    current_tpsl = dp.get('patterns_tpsl', {})
    total_pats = len(long_pats) + len(short_pats)

    # Build signal index
    signal_index = {}
    for i in range(2, n_bars):
        pat = f"{types[i-2]}-{types[i-1]}-{types[i]}"
        if pat not in signal_index:
            signal_index[pat] = []
        signal_index[pat].append(i)

    print('=' * 95)
    print('TP/SL Stability Analysis & Hybrid Strategy Design')
    print(f'Patterns: {total_pats} (L:{len(long_pats)} S:{len(short_pats)}) | Data: {n_bars} bars (270d)')
    print('4-fold expanding stability | Hybrid WF comparison | Threshold sweep')
    print('=' * 95)

    # ══════════════════════════════════════════════════════════════
    # Phase 1: Compute stability for ALL 47 patterns (4 expanding folds)
    # ══════════════════════════════════════════════════════════════
    print('\n' + '─' * 95)
    print('Phase 1: TP/SL Stability Analysis (4 expanding folds)')
    print('─' * 95)

    all_stability = {}
    all_pats_dirs = [(p, 'LONG') for p in long_pats] + [(p, 'SHORT') for p in short_pats]

    for pat, direction in all_pats_dirs:
        stab = compute_stability(signal_index, pat, direction, opens, highs, lows, n_bars, n_folds=4)
        all_stability[pat] = {**stab, 'direction': direction}

    # Categorize
    stable = {p: s for p, s in all_stability.items() if s['category'] == 'STABLE'}
    moderate = {p: s for p, s in all_stability.items() if s['category'] == 'MODERATE'}
    unstable = {p: s for p, s in all_stability.items() if s['category'] == 'UNSTABLE'}
    insufficient = {p: s for p, s in all_stability.items() if s['category'] == 'INSUFFICIENT_DATA'}

    print(f'\n  Category Distribution:')
    print(f'    STABLE   (TP rng<=0.3, SL rng<=0.5): {len(stable)}')
    print(f'    MODERATE (TP rng<=0.7, SL rng<=1.5): {len(moderate)}')
    print(f'    UNSTABLE (beyond moderate):           {len(unstable)}')
    print(f'    INSUFFICIENT_DATA:                    {len(insufficient)}')

    # Print all patterns sorted by stability score
    sorted_pats = sorted(all_stability.items(),
                         key=lambda x: x[1]['stability_score'] if x[1]['stability_score'] else 999)

    print(f'\n  {"Pattern":<18} {"Dir":>5} {"Cat":>10} {"Score":>6} {"TP_rng":>6} {"SL_rng":>6} '
          f'{"Recent":>7} {"Fold TPs":>28} {"Fold SLs":>28}')
    print('  ' + '-' * 135)

    for pat, s in sorted_pats:
        fo = s['fold_optima']
        tp_strs = []
        sl_strs = []
        for o in fo:
            if o:
                tp_strs.append(f'{o["tp"]:.1f}')
                sl_strs.append(f'{o["sl"]:.1f}')
            else:
                tp_strs.append('N/A')
                sl_strs.append('N/A')

        tp_rng = f'{s["tp_range"]:.1f}' if s['tp_range'] is not None else 'N/A'
        sl_rng = f'{s["sl_range"]:.1f}' if s['sl_range'] is not None else 'N/A'
        score = f'{s["stability_score"]:.3f}' if s['stability_score'] else 'N/A'
        conv = 'Y' if s.get('recent_convergence') else 'N'

        print(f'  {pat:<18} {s["direction"]:>5} {s["category"]:>10} {score:>6} {tp_rng:>5}% {sl_rng:>5}% '
              f'{conv:>7} {",".join(tp_strs):>28} {",".join(sl_strs):>28}')

    # ══════════════════════════════════════════════════════════════
    # Phase 2: Threshold Sweep — find optimal stability cutoff
    # ══════════════════════════════════════════════════════════════
    print('\n' + '─' * 95)
    print('Phase 2: Stability Threshold Sweep (Expanding WF)')
    print('  Testing: Universal / Stable-only / Moderate+ / All Per-Pattern / Recent-Converged')
    print('─' * 95)

    # Define scenario sets
    scenarios = {}

    # A: Universal (baseline)
    scenarios['A_UNIVERSAL'] = {'mode': 'universal', 'stable_pats': None, 'desc': 'All Universal TP 2.0/SL 3.0'}

    # B: Only STABLE patterns get per-pattern
    stable_set = set(stable.keys())
    scenarios['B_STABLE_ONLY'] = {'mode': 'hybrid', 'stable_pats': stable_set,
                                   'desc': f'STABLE({len(stable_set)}) per-pattern, rest universal'}

    # C: STABLE + MODERATE get per-pattern
    moderate_set = stable_set | set(moderate.keys())
    scenarios['C_MODERATE_PLUS'] = {'mode': 'hybrid', 'stable_pats': moderate_set,
                                    'desc': f'STABLE+MODERATE({len(moderate_set)}) per-pattern, rest universal'}

    # D: Recent converging (last 2 folds agree)
    recent_set = set()
    for p, s in all_stability.items():
        if s.get('recent_convergence'):
            recent_set.add(p)
    scenarios['D_RECENT_CONVERGE'] = {'mode': 'hybrid', 'stable_pats': recent_set,
                                      'desc': f'Recent-converging({len(recent_set)}) per-pattern, rest universal'}

    # E: All per-pattern
    scenarios['E_ALL_PP'] = {'mode': 'per_pattern', 'stable_pats': None, 'desc': 'All 47 per-pattern'}

    # F: STABLE + recent-converging
    stable_recent = stable_set | recent_set
    scenarios['F_STABLE_RECENT'] = {'mode': 'hybrid', 'stable_pats': stable_recent,
                                     'desc': f'STABLE+Recent({len(stable_recent)}) per-pattern, rest universal'}

    # G: Low stability score (< 0.5)
    low_score_set = set()
    for p, s in all_stability.items():
        if s['stability_score'] is not None and s['stability_score'] < 0.5:
            low_score_set.add(p)
    scenarios['G_LOW_SCORE'] = {'mode': 'hybrid', 'stable_pats': low_score_set,
                                'desc': f'Score<0.5({len(low_score_set)}) per-pattern, rest universal'}

    # H: Very low stability score (< 0.3)
    vlow_score_set = set()
    for p, s in all_stability.items():
        if s['stability_score'] is not None and s['stability_score'] < 0.3:
            vlow_score_set.add(p)
    scenarios['H_VLOW_SCORE'] = {'mode': 'hybrid', 'stable_pats': vlow_score_set,
                                  'desc': f'Score<0.3({len(vlow_score_set)}) per-pattern, rest universal'}

    wf_results = {}
    print(f'\n  {"Scenario":<22} {"PP#":>4} {"F0 PnL":>10} {"F1 PnL":>10} {"F2 PnL":>10} '
          f'{"Avg OOS":>10} {"Pos":>4} {"F0 WR":>6} {"F1 WR":>6} {"F2 WR":>6} {"Avg MDD":>8}')
    print('  ' + '-' * 110)

    for name in sorted(scenarios.keys()):
        sc = scenarios[name]
        folds = expanding_wf_hybrid(signal_index, long_pats, short_pats,
                                    opens, highs, lows, n_bars, n_folds=3,
                                    stable_pats=sc['stable_pats'], mode=sc['mode'])
        avg_oos = np.mean([f['pnl'] for f in folds])
        pos = sum(1 for f in folds if f['pnl'] > 0)
        avg_mdd = np.mean([f['mdd'] for f in folds])
        pp_count = len(sc['stable_pats']) if sc['stable_pats'] else (47 if sc['mode'] == 'per_pattern' else 0)

        wf_results[name] = {
            'desc': sc['desc'],
            'pp_count': pp_count,
            'folds': folds,
            'avg_oos_pnl': round(avg_oos, 1),
            'positive_folds': pos,
            'avg_mdd': round(avg_mdd, 1),
        }

        pnls = [f['pnl'] for f in folds]
        wrs = [f['wr'] for f in folds]
        print(f'  {name:<22} {pp_count:>4} {pnls[0]:>+9.1f}% {pnls[1]:>+9.1f}% {pnls[2]:>+9.1f}% '
              f'{avg_oos:>+9.1f}% {pos:>3}/3 {wrs[0]:>5.1f}% {wrs[1]:>5.1f}% {wrs[2]:>5.1f}% {avg_mdd:>7.1f}%')

    # ══════════════════════════════════════════════════════════════
    # Phase 3: Best Scenario MC Validation
    # ══════════════════════════════════════════════════════════════
    print('\n' + '─' * 95)
    print('Phase 3: Best Scenario MC Validation')
    print('─' * 95)

    # Find best scenario (highest avg OOS PnL)
    best_name = max(wf_results.keys(), key=lambda k: wf_results[k]['avg_oos_pnl'])
    best = wf_results[best_name]

    print(f'  Best scenario: {best_name} — {best["desc"]}')
    print(f'  Avg OOS PnL: {best["avg_oos_pnl"]:+.1f}% | Positive: {best["positive_folds"]}/3')

    # In-sample portfolio for best scenario
    best_sc = scenarios[best_name]
    all_trades_best = []
    for pat in long_pats:
        if pat not in signal_index:
            continue
        use_pp = False
        if best_sc['mode'] == 'per_pattern':
            use_pp = True
        elif best_sc['mode'] == 'hybrid' and best_sc['stable_pats'] and pat in best_sc['stable_pats']:
            use_pp = True

        if use_pp and pat in current_tpsl:
            tp, sl = current_tpsl[pat]
            trades = bt_signals(signal_index[pat], 'LONG', tp, sl, opens, highs, lows, n_bars)
        else:
            trades = bt_signals(signal_index[pat], 'LONG', UNIVERSAL_TP, UNIVERSAL_SL,
                                opens, highs, lows, n_bars)
        all_trades_best.extend(trades)

    for pat in short_pats:
        if pat not in signal_index:
            continue
        use_pp = False
        if best_sc['mode'] == 'per_pattern':
            use_pp = True
        elif best_sc['mode'] == 'hybrid' and best_sc['stable_pats'] and pat in best_sc['stable_pats']:
            use_pp = True

        if use_pp and pat in current_tpsl:
            tp, sl = current_tpsl[pat]
            trades = bt_signals(signal_index[pat], 'SHORT', tp, sl, opens, highs, lows, n_bars)
        else:
            trades = bt_signals(signal_index[pat], 'SHORT', UNIVERSAL_TP, UNIVERSAL_SL,
                                opens, highs, lows, n_bars)
        all_trades_best.extend(trades)

    best_portfolio = portfolio_1pos(all_trades_best)
    best_stats = calc_compound(best_portfolio)
    best_pnls = [t[2] for t in best_portfolio]
    best_mc = mc_test_multi_seed(best_pnls)

    print(f'  In-sample: Tr={best_stats["trades"]} WR={best_stats["wr"]}% '
          f'PnL=+{best_stats["pnl"]:.1f}% MDD={best_stats["mdd"]:.1f}%')
    print(f'  Multi-seed MC p={best_mc:.6f} {"PASS" if best_mc < 0.01 else "FAIL"}')

    # Also MC for universal baseline
    all_trades_uni = []
    for pat in long_pats:
        if pat in signal_index:
            trades = bt_signals(signal_index[pat], 'LONG', UNIVERSAL_TP, UNIVERSAL_SL,
                                opens, highs, lows, n_bars)
            all_trades_uni.extend(trades)
    for pat in short_pats:
        if pat in signal_index:
            trades = bt_signals(signal_index[pat], 'SHORT', UNIVERSAL_TP, UNIVERSAL_SL,
                                opens, highs, lows, n_bars)
            all_trades_uni.extend(trades)

    uni_portfolio = portfolio_1pos(all_trades_uni)
    uni_stats = calc_compound(uni_portfolio)
    uni_pnls = [t[2] for t in uni_portfolio]
    uni_mc = mc_test_multi_seed(uni_pnls)

    print(f'\n  Universal baseline: Tr={uni_stats["trades"]} WR={uni_stats["wr"]}% '
          f'PnL=+{uni_stats["pnl"]:.1f}% MDD={uni_stats["mdd"]:.1f}%')
    print(f'  Multi-seed MC p={uni_mc:.6f} {"PASS" if uni_mc < 0.01 else "FAIL"}')

    # ══════════════════════════════════════════════════════════════
    # Phase 4: Detailed Pattern Classification
    # ══════════════════════════════════════════════════════════════
    print('\n' + '─' * 95)
    print('Phase 4: Pattern Classification Summary')
    print('─' * 95)

    for cat_name, cat_dict in [('STABLE', stable), ('MODERATE', moderate),
                                ('UNSTABLE', unstable), ('INSUFFICIENT_DATA', insufficient)]:
        if not cat_dict:
            continue
        print(f'\n  [{cat_name}] — {len(cat_dict)} patterns')
        for pat in sorted(cat_dict.keys()):
            s = cat_dict[pat]
            fo = s['fold_optima']
            tp_strs = [f'{o["tp"]:.1f}' if o else 'N/A' for o in fo]
            sl_strs = [f'{o["sl"]:.1f}' if o else 'N/A' for o in fo]
            cur = current_tpsl.get(pat, [None, None])
            cur_str = f'{cur[0]:.1f}/{cur[1]:.1f}' if cur[0] else 'N/A'
            conv = 'Conv' if s.get('recent_convergence') else ''
            print(f'    {pat:<18} {s["direction"]:>5} Score={s["stability_score"]:.3f} '
                  f'TP=[{",".join(tp_strs)}] SL=[{",".join(sl_strs)}] '
                  f'Current={cur_str} {conv}')

    # ══════════════════════════════════════════════════════════════
    # Phase 5: Final Recommendation
    # ══════════════════════════════════════════════════════════════
    print('\n' + '=' * 95)
    print('FINAL RECOMMENDATION')
    print('=' * 95)

    # Rank scenarios
    ranked = sorted(wf_results.items(), key=lambda x: x[1]['avg_oos_pnl'], reverse=True)

    print(f'\n  Scenario Ranking (by Expanding WF OOS PnL):')
    for rank, (name, res) in enumerate(ranked, 1):
        marker = '<-- BEST' if rank == 1 else ''
        print(f'    {rank}. {name:<22} Avg OOS: {res["avg_oos_pnl"]:>+8.1f}% | '
              f'Pos: {res["positive_folds"]}/3 | PP: {res["pp_count"]} | '
              f'MDD: {res["avg_mdd"]:.1f}% {marker}')

    winner = ranked[0][0]
    runner_up = ranked[1][0] if len(ranked) > 1 else None
    uni_result = wf_results['A_UNIVERSAL']

    print(f'\n  Winner: {winner}')
    print(f'  vs Universal: {wf_results[winner]["avg_oos_pnl"]:+.1f}% vs {uni_result["avg_oos_pnl"]:+.1f}%')

    if winner == 'A_UNIVERSAL':
        action = 'ROLLBACK to Universal TP 2.0/SL 3.0 — per-pattern adds no OOS value'
    elif wf_results[winner]['avg_oos_pnl'] > uni_result['avg_oos_pnl'] * 1.05:
        action = f'SWITCH to {winner} — {wf_results[winner]["avg_oos_pnl"] - uni_result["avg_oos_pnl"]:+.1f}% OOS improvement'
    else:
        action = f'MARGINAL — {winner} only {wf_results[winner]["avg_oos_pnl"] - uni_result["avg_oos_pnl"]:+.1f}% better, consider risk'

    print(f'  Action: {action}')

    if winner != 'A_UNIVERSAL' and scenarios[winner]['stable_pats']:
        print(f'\n  Patterns using per-pattern TP/SL in {winner}:')
        for p in sorted(scenarios[winner]['stable_pats']):
            s = all_stability[p]
            cur = current_tpsl.get(p, [None, None])
            print(f'    {p:<18} {s["direction"]:>5} TP={cur[0]}% SL={cur[1]}%')

    # ── Save ──
    elapsed = time.time() - t0
    output = {
        'analysis': 'stability_analysis',
        'version': 'v1.28.5',
        'data_bars': n_bars,
        'patterns': total_pats,
        'elapsed_sec': round(elapsed, 1),
        'stability': {
            'categories': {
                'STABLE': list(stable.keys()),
                'MODERATE': list(moderate.keys()),
                'UNSTABLE': list(unstable.keys()),
                'INSUFFICIENT_DATA': list(insufficient.keys()),
            },
            'all_patterns': {p: {k: v for k, v in s.items() if k != 'fold_optima'}
                             for p, s in all_stability.items()},
        },
        'scenarios': {name: {k: v for k, v in res.items() if k != 'folds'}
                      for name, res in wf_results.items()},
        'scenario_folds': {name: res['folds'] for name, res in wf_results.items()},
        'best_scenario': {
            'name': winner,
            'desc': wf_results[winner]['desc'],
            'avg_oos_pnl': wf_results[winner]['avg_oos_pnl'],
            'vs_universal': round(wf_results[winner]['avg_oos_pnl'] - uni_result['avg_oos_pnl'], 1),
        },
        'mc_validation': {
            'best_mc_p': best_mc,
            'universal_mc_p': uni_mc,
        },
        'recommendation': action,
    }

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output, f, indent=2)
    print(f'\nResults saved: {OUTPUT_FILE}')
    print(f'Elapsed: {elapsed:.0f}s')


if __name__ == '__main__':
    main()
