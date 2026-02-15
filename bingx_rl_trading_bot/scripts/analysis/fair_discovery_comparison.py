#!/usr/bin/env python3
"""
Fair Discovery Comparison — 패턴 발굴 단계부터 TP/SL 분리 비교

핵심 가설: 현재 47패턴은 Universal TP 2.0/SL 3.0 기준으로 발굴되었으므로,
Universal이 유리한 것은 순환논리.

공정한 비교:
  A. Universal Discovery: TP 2.0/SL 3.0으로 패턴 발굴 → Universal로 트레이딩 (현재)
  B. Per-Pattern Discovery: 각 패턴 최적 TP/SL로 발굴 → 최적 TP/SL로 트레이딩
  C. Cross-test: A의 패턴을 B의 TP/SL로, B의 패턴을 A의 TP/SL로

모든 비교는 Expanding Window WF로 검증.

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
EDGE_THRESHOLD = 10.0
MC_THRESHOLD = 0.01
MIN_TRADES = 20

UNIVERSAL_TP = 2.0
UNIVERSAL_SL = 3.0

TP_GRID = [0.5, 0.7, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.1, 2.5, 3.0]
SL_GRID = [0.5, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
MAX_BASELINE_WR = 70.0  # baseline WR > 70% → distance effect dominant, skip

DATA_FILE = os.path.join(_PROJECT_ROOT, 'data', 'btc_5m_270days_reclassified.csv')
OUTPUT_FILE = os.path.join(_PROJECT_ROOT, 'results', 'fair_discovery_comparison.json')


def classify_all(df):
    df['body'] = abs(df['close'] - df['open'])
    df['avg_body'] = df['body'].rolling(AVG_BODY_WINDOW, min_periods=1).mean()
    types = []
    for i, row in df.iterrows():
        ct = classify_candle(row, df.at[i, 'avg_body'])
        types.append(ct.value)
    return types


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
        return {'pnl': 0, 'trades': 0, 'wr': 0, 'mdd': 0, 'pf': 0, 'pnl_mdd': 0, 'max_cl': 0}
    equity = 100.0
    peak = 100.0
    mdd = 0.0
    wins = []
    losses = []
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
        'max_cl': max_cl,
    }


def mc_test_multi(pnls):
    if len(pnls) < 5:
        return 1.0
    pnls_arr = np.array(pnls)
    actual = np.sum(pnls_arr)
    p_vals = []
    for seed in MC_SEEDS:
        rng = np.random.RandomState(seed)
        signs = rng.choice([-1, 1], size=(MC_SIMS, len(pnls_arr)))
        rand_sums = signs @ pnls_arr
        p_vals.append(float(np.mean(rand_sums >= actual)))
    return max(p_vals)


def grid_search_best(signal_bars, direction, opens, highs, lows, n_bars, min_tr=20,
                     max_baseline_wr=None):
    """Find best TP/SL by PnL/MDD score.
    max_baseline_wr: if set, skip TP/SL combos where SL/(TP+SL)*100 > max_baseline_wr
    (i.e., skip distance-dependent patterns where TP << SL).
    """
    best = None
    best_score = -9999
    for tp in TP_GRID:
        for sl in SL_GRID:
            if sl < 0.5:
                continue
            if max_baseline_wr is not None:
                bwr = sl / (tp + sl) * 100
                if bwr > max_baseline_wr:
                    continue
            trades = bt_signals(signal_bars, direction, tp, sl, opens, highs, lows, n_bars)
            if len(trades) < min_tr:
                continue
            stats = calc_compound(trades)
            score = stats['pnl_mdd'] if stats['mdd'] > 0 else stats['pnl']
            if score > best_score:
                best_score = score
                best = {'tp': tp, 'sl': sl, **stats}
    return best


# ══════════════════════════════════════════════════════════════
# Discovery Methods
# ══════════════════════════════════════════════════════════════

def discover_universal(signal_index, opens, highs, lows, n_bars, bar_limit=None):
    """
    Method A: Discover patterns using Universal TP/SL.
    edge = WR - baseline_wr, where baseline_wr = SL/(TP+SL)
    """
    baseline_wr = UNIVERSAL_SL / (UNIVERSAL_TP + UNIVERSAL_SL) * 100
    selected = {}

    for pat_name, sigs in signal_index.items():
        if bar_limit:
            sigs = [s for s in sigs if s < bar_limit]
        if len(sigs) < MIN_TRADES:
            continue
        for direction in ['LONG', 'SHORT']:
            trades = bt_signals(sigs, direction, UNIVERSAL_TP, UNIVERSAL_SL,
                                opens, highs, lows, n_bars)
            if len(trades) < MIN_TRADES:
                continue
            pnls = [t[2] for t in trades]
            wr = len([p for p in pnls if p > 0]) / len(pnls) * 100
            edge = wr - baseline_wr
            if edge < EDGE_THRESHOLD:
                continue
            mc_p = mc_test_multi(pnls)
            if mc_p >= MC_THRESHOLD:
                continue
            key = f'{pat_name}_{direction}'
            selected[key] = {
                'pattern': pat_name, 'direction': direction,
                'tp': UNIVERSAL_TP, 'sl': UNIVERSAL_SL,
                'trades': len(trades), 'wr': round(wr, 1),
                'edge': round(edge, 1), 'mc_p': round(mc_p, 4),
            }
    return selected


def discover_per_pattern(signal_index, opens, highs, lows, n_bars, bar_limit=None):
    """
    Method B: Discover patterns using per-pattern optimal TP/SL.
    For each pattern+direction, find optimal TP/SL via grid search,
    then check if edge >= threshold with THAT TP/SL.

    IMPORTANT: Only considers TP/SL combos where baseline WR <= MAX_BASELINE_WR.
    This prevents discovering patterns where TP << SL (distance effect dominant).
    e.g., TP=0.5/SL=4.0 → baseline=88.9% → skipped because edge is just noise.
    """
    selected = {}

    for pat_name, sigs in signal_index.items():
        if bar_limit:
            sigs = [s for s in sigs if s < bar_limit]
        if len(sigs) < MIN_TRADES:
            continue
        for direction in ['LONG', 'SHORT']:
            opt = grid_search_best(sigs, direction, opens, highs, lows, n_bars,
                                   min_tr=MIN_TRADES, max_baseline_wr=MAX_BASELINE_WR)
            if opt is None:
                continue

            tp, sl = opt['tp'], opt['sl']
            trades = bt_signals(sigs, direction, tp, sl, opens, highs, lows, n_bars)
            if len(trades) < MIN_TRADES:
                continue

            pnls = [t[2] for t in trades]
            wr = len([p for p in pnls if p > 0]) / len(pnls) * 100
            baseline_wr = sl / (tp + sl) * 100
            edge = wr - baseline_wr

            if edge < EDGE_THRESHOLD:
                continue

            mc_p = mc_test_multi(pnls)
            if mc_p >= MC_THRESHOLD:
                continue

            key = f'{pat_name}_{direction}'
            selected[key] = {
                'pattern': pat_name, 'direction': direction,
                'tp': tp, 'sl': sl,
                'trades': len(trades), 'wr': round(wr, 1),
                'edge': round(edge, 1), 'mc_p': round(mc_p, 4),
                'baseline_wr': round(baseline_wr, 1),
            }
    return selected


def portfolio_backtest(selected, signal_index, opens, highs, lows, n_bars,
                       bar_start=None, bar_end=None, use_universal=False):
    """Run portfolio backtest for selected patterns."""
    all_trades = []
    for key, info in selected.items():
        pat = info['pattern']
        direction = info['direction']
        if pat not in signal_index:
            continue
        sigs = signal_index[pat]
        if bar_start is not None or bar_end is not None:
            sigs = [s for s in sigs
                    if (bar_start is None or s >= bar_start) and
                       (bar_end is None or s < bar_end)]
        if not sigs:
            continue

        if use_universal:
            tp, sl = UNIVERSAL_TP, UNIVERSAL_SL
        else:
            tp, sl = info['tp'], info['sl']

        trades = bt_signals(sigs, direction, tp, sl, opens, highs, lows, n_bars)
        all_trades.extend(trades)

    portfolio = portfolio_1pos(all_trades)
    return portfolio


def expanding_wf_discovery(signal_index, opens, highs, lows, n_bars, n_folds=3,
                           method='universal'):
    """
    Expanding Window WF with discovery at each fold.
    Train on [0..split], discover patterns, test on [split..split+fold_size].
    """
    fold_size = n_bars // (n_folds + 1)
    results = []

    for fold in range(n_folds):
        train_end = (fold + 1) * fold_size
        test_start = train_end
        test_end = (fold + 2) * fold_size if fold < n_folds - 1 else n_bars

        # Discover on training period
        if method == 'universal':
            selected = discover_universal(signal_index, opens, highs, lows, n_bars,
                                          bar_limit=train_end)
            use_uni = True
        elif method == 'per_pattern':
            selected = discover_per_pattern(signal_index, opens, highs, lows, n_bars,
                                            bar_limit=train_end)
            use_uni = False
        elif method == 'pp_discovery_uni_trade':
            # Discover with per-pattern, but trade with universal
            selected = discover_per_pattern(signal_index, opens, highs, lows, n_bars,
                                            bar_limit=train_end)
            use_uni = True
        elif method == 'uni_discovery_pp_trade':
            # Discover with universal, but trade with per-pattern optimization
            selected = discover_universal(signal_index, opens, highs, lows, n_bars,
                                          bar_limit=train_end)
            # Re-optimize TP/SL for each selected pattern on training data
            for key in selected:
                pat = selected[key]['pattern']
                direction = selected[key]['direction']
                sigs = [s for s in signal_index.get(pat, []) if s < train_end]
                if len(sigs) >= 10:
                    opt = grid_search_best(sigs, direction, opens, highs, lows, n_bars,
                                           min_tr=10, max_baseline_wr=MAX_BASELINE_WR)
                    if opt:
                        selected[key]['tp'] = opt['tp']
                        selected[key]['sl'] = opt['sl']
            use_uni = False
        else:
            raise ValueError(f'Unknown method: {method}')

        n_discovered = len(selected)

        # Test on OOS period
        portfolio = portfolio_backtest(selected, signal_index, opens, highs, lows, n_bars,
                                       bar_start=test_start, bar_end=test_end,
                                       use_universal=use_uni)
        stats = calc_compound(portfolio)

        results.append({
            'fold': fold,
            'train_range': f'0-{train_end}',
            'test_range': f'{test_start}-{test_end}',
            'discovered': n_discovered,
            **stats,
        })

    return results


def main():
    t0 = time.time()
    print('=' * 100)
    print('Fair Discovery Comparison — 패턴 발굴 방법론별 공정 비교')
    print('=' * 100)

    df = pd.read_csv(DATA_FILE)
    types = classify_all(df)
    n_bars = len(df)
    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values

    signal_index = {}
    for i in range(2, n_bars):
        pat = f'{types[i-2]}-{types[i-1]}-{types[i]}'
        if pat not in signal_index:
            signal_index[pat] = []
        signal_index[pat].append(i)

    print(f'Data: {n_bars} bars | Unique patterns: {len(signal_index)}')
    print(f'Edge threshold: {EDGE_THRESHOLD}pp | MC: {MC_THRESHOLD} (3-seed) | Min trades: {MIN_TRADES}')
    print(f'Max baseline WR for PP discovery: {MAX_BASELINE_WR}% (R:R >= {(100-MAX_BASELINE_WR)/MAX_BASELINE_WR:.2f})')
    print(f'Universal baseline WR: {UNIVERSAL_SL/(UNIVERSAL_TP+UNIVERSAL_SL)*100:.1f}%')

    # ══════════════════════════════════════════════════════════════
    # Phase 1: Full-period discovery comparison
    # ══════════════════════════════════════════════════════════════
    print('\n' + '─' * 100)
    print('Phase 1: Full-period Discovery (In-sample)')
    print('─' * 100)

    uni_disc = discover_universal(signal_index, opens, highs, lows, n_bars)
    pp_disc = discover_per_pattern(signal_index, opens, highs, lows, n_bars)

    uni_pats = set(uni_disc.keys())
    pp_pats = set(pp_disc.keys())
    overlap = uni_pats & pp_pats
    uni_only = uni_pats - pp_pats
    pp_only = pp_pats - uni_pats

    print(f'\n  Universal discovery: {len(uni_disc)} patterns')
    print(f'  Per-pattern discovery: {len(pp_disc)} patterns')
    print(f'  Overlap: {len(overlap)} | Uni-only: {len(uni_only)} | PP-only: {len(pp_only)}')

    if uni_only:
        print(f'\n  Patterns ONLY found by Universal (missed by per-pattern):')
        for k in sorted(uni_only):
            info = uni_disc[k]
            print(f'    {k:<25} WR={info["wr"]}% edge={info["edge"]}pp trades={info["trades"]} MC={info["mc_p"]}')

    if pp_only:
        print(f'\n  Patterns ONLY found by Per-pattern (missed by Universal): {len(pp_only)} total')
        print(f'  (showing top 30 by edge):')
        sorted_pp = sorted(pp_only, key=lambda k: pp_disc[k]['edge'], reverse=True)
        for k in sorted_pp[:30]:
            info = pp_disc[k]
            print(f'    {k:<25} TP={info["tp"]} SL={info["sl"]} WR={info["wr"]}% '
                  f'edge={info["edge"]}pp BE_WR={info["baseline_wr"]}% trades={info["trades"]}')

    # In-sample backtest
    print(f'\n  In-sample portfolio comparison:')
    uni_is = portfolio_backtest(uni_disc, signal_index, opens, highs, lows, n_bars, use_universal=True)
    uni_is_stats = calc_compound(uni_is)
    pp_is = portfolio_backtest(pp_disc, signal_index, opens, highs, lows, n_bars, use_universal=False)
    pp_is_stats = calc_compound(pp_is)

    print(f'    Universal:   Tr={uni_is_stats["trades"]} WR={uni_is_stats["wr"]}% '
          f'PnL=+{uni_is_stats["pnl"]:.1f}% MDD={uni_is_stats["mdd"]:.1f}% '
          f'PnL/MDD={uni_is_stats["pnl_mdd"]:.1f}x PF={uni_is_stats["pf"]}')
    print(f'    Per-pattern: Tr={pp_is_stats["trades"]} WR={pp_is_stats["wr"]}% '
          f'PnL=+{pp_is_stats["pnl"]:.1f}% MDD={pp_is_stats["mdd"]:.1f}% '
          f'PnL/MDD={pp_is_stats["pnl_mdd"]:.1f}x PF={pp_is_stats["pf"]}')

    # ══════════════════════════════════════════════════════════════
    # Phase 2: Expanding Window WF — 4 methods compared
    # ══════════════════════════════════════════════════════════════
    print('\n' + '─' * 100)
    print('Phase 2: Expanding Window WF (3-fold) — 4 Discovery×Trading Methods')
    print('─' * 100)

    methods = {
        'A_UniDisc_UniTrade': 'universal',
        'B_PPDisc_PPTrade': 'per_pattern',
        'C_PPDisc_UniTrade': 'pp_discovery_uni_trade',
        'D_UniDisc_PPTrade': 'uni_discovery_pp_trade',
    }

    all_wf_results = {}
    print(f'\n  {"Method":<25} {"Disc":>5} {"F0 PnL":>10} {"F1 PnL":>10} {"F2 PnL":>10} '
          f'{"Avg OOS":>10} {"Pos":>4} {"Avg WR":>7} {"Avg MDD":>8} {"Avg Disc":>8}')
    print('  ' + '-' * 105)

    for name, method in methods.items():
        folds = expanding_wf_discovery(signal_index, opens, highs, lows, n_bars,
                                       n_folds=3, method=method)
        avg_oos = np.mean([f['pnl'] for f in folds])
        pos = sum(1 for f in folds if f['pnl'] > 0)
        avg_wr = np.mean([f['wr'] for f in folds])
        avg_mdd = np.mean([f['mdd'] for f in folds])
        avg_disc = np.mean([f['discovered'] for f in folds])
        disc_method = method.split('_')[0] if '_' in method else method

        all_wf_results[name] = {
            'method': method,
            'folds': folds,
            'avg_oos_pnl': round(avg_oos, 1),
            'positive_folds': pos,
            'avg_wr': round(avg_wr, 1),
            'avg_mdd': round(avg_mdd, 1),
            'avg_discovered': round(avg_disc, 1),
        }

        pnls = [f['pnl'] for f in folds]
        print(f'  {name:<25} {disc_method:>5} {pnls[0]:>+9.1f}% {pnls[1]:>+9.1f}% {pnls[2]:>+9.1f}% '
              f'{avg_oos:>+9.1f}% {pos:>3}/3 {avg_wr:>6.1f}% {avg_mdd:>7.1f}% {avg_disc:>7.1f}')

    # ══════════════════════════════════════════════════════════════
    # Phase 3: Overlap analysis per WF fold
    # ══════════════════════════════════════════════════════════════
    print('\n' + '─' * 100)
    print('Phase 3: Discovery Overlap Analysis per WF Fold')
    print('─' * 100)

    fold_size = n_bars // 4
    for fold in range(3):
        train_end = (fold + 1) * fold_size
        uni_d = discover_universal(signal_index, opens, highs, lows, n_bars, bar_limit=train_end)
        pp_d = discover_per_pattern(signal_index, opens, highs, lows, n_bars, bar_limit=train_end)

        uni_set = set(uni_d.keys())
        pp_set = set(pp_d.keys())
        overlap_f = uni_set & pp_set
        uni_only_f = uni_set - pp_set
        pp_only_f = pp_set - uni_set

        print(f'\n  Fold {fold} (train 0-{train_end}):')
        print(f'    Universal: {len(uni_d)} | Per-Pattern: {len(pp_d)} | '
              f'Overlap: {len(overlap_f)} | Uni-only: {len(uni_only_f)} | PP-only: {len(pp_only_f)}')

        if pp_only_f:
            print(f'    PP-only patterns: {len(pp_only_f)} (top 10 by edge):')
            sorted_ppf = sorted(pp_only_f, key=lambda k: pp_d[k]['edge'], reverse=True)
            for k in sorted_ppf[:10]:
                info = pp_d[k]
                print(f'      {k:<25} TP={info["tp"]} SL={info["sl"]} edge={info["edge"]}pp '
                      f'WR={info["wr"]}% (BE={info["baseline_wr"]}%)')

    # ══════════════════════════════════════════════════════════════
    # Phase 4: TP/SL Distribution by Discovery Method
    # ══════════════════════════════════════════════════════════════
    print('\n' + '─' * 100)
    print('Phase 4: Per-Pattern TP/SL Distribution (full-period discovery)')
    print('─' * 100)

    pp_tps = [pp_disc[k]['tp'] for k in pp_disc]
    pp_sls = [pp_disc[k]['sl'] for k in pp_disc]

    if pp_tps:
        print(f'  TP: min={min(pp_tps):.1f} med={np.median(pp_tps):.1f} '
              f'mean={np.mean(pp_tps):.1f} max={max(pp_tps):.1f}')
        print(f'  SL: min={min(pp_sls):.1f} med={np.median(pp_sls):.1f} '
              f'mean={np.mean(pp_sls):.1f} max={max(pp_sls):.1f}')

        # How many are close to Universal?
        near_uni = sum(1 for k in pp_disc
                       if abs(pp_disc[k]['tp'] - UNIVERSAL_TP) <= 0.3 and
                          abs(pp_disc[k]['sl'] - UNIVERSAL_SL) <= 0.5)
        print(f'  Near Universal (TP±0.3, SL±0.5): {near_uni}/{len(pp_disc)} '
              f'({near_uni/len(pp_disc)*100:.0f}%)')

        # Baseline WR distribution
        pp_bwrs = [pp_disc[k]['baseline_wr'] for k in pp_disc]
        print(f'  Baseline WR: min={min(pp_bwrs):.1f}% med={np.median(pp_bwrs):.1f}% '
              f'mean={np.mean(pp_bwrs):.1f}% max={max(pp_bwrs):.1f}%')

        # R:R distribution
        pp_rrs = [pp_disc[k]['tp'] / pp_disc[k]['sl'] for k in pp_disc if pp_disc[k]['sl'] > 0]
        print(f'  R:R (TP/SL): min={min(pp_rrs):.2f} med={np.median(pp_rrs):.2f} '
              f'mean={np.mean(pp_rrs):.2f} max={max(pp_rrs):.2f}')

        # TP/SL frequency
        from collections import Counter
        tp_counts = Counter(pp_tps)
        sl_counts = Counter(pp_sls)
        print(f'  TP top-3: {tp_counts.most_common(3)}')
        print(f'  SL top-3: {sl_counts.most_common(3)}')

    # ══════════════════════════════════════════════════════════════
    # Phase 5: MC validation of best method
    # ══════════════════════════════════════════════════════════════
    print('\n' + '─' * 100)
    print('Phase 5: MC Validation')
    print('─' * 100)

    best_name = max(all_wf_results.keys(), key=lambda k: all_wf_results[k]['avg_oos_pnl'])
    best = all_wf_results[best_name]
    print(f'  Best method: {best_name} (Avg OOS: {best["avg_oos_pnl"]:+.1f}%)')

    # MC on each method's in-sample portfolio
    for name in sorted(all_wf_results.keys()):
        method = methods[name]
        if 'per_pattern' in method or 'pp' in method.lower():
            disc = pp_disc
            use_uni = 'uni_trade' in method.lower() or method == 'pp_discovery_uni_trade'
        else:
            disc = uni_disc
            use_uni = 'uni' in method.lower() and 'pp_trade' not in method.lower()

        if name == 'A_UniDisc_UniTrade':
            portfolio = portfolio_backtest(uni_disc, signal_index, opens, highs, lows, n_bars, use_universal=True)
        elif name == 'B_PPDisc_PPTrade':
            portfolio = portfolio_backtest(pp_disc, signal_index, opens, highs, lows, n_bars, use_universal=False)
        elif name == 'C_PPDisc_UniTrade':
            portfolio = portfolio_backtest(pp_disc, signal_index, opens, highs, lows, n_bars, use_universal=True)
        elif name == 'D_UniDisc_PPTrade':
            # Need to re-optimize for uni-discovered patterns
            pp_from_uni = {}
            for key, info in uni_disc.items():
                pat = info['pattern']
                direction = info['direction']
                sigs = signal_index.get(pat, [])
                if len(sigs) >= 10:
                    opt = grid_search_best(sigs, direction, opens, highs, lows, n_bars,
                                           min_tr=10, max_baseline_wr=MAX_BASELINE_WR)
                    if opt:
                        pp_from_uni[key] = {**info, 'tp': opt['tp'], 'sl': opt['sl']}
                    else:
                        pp_from_uni[key] = info
                else:
                    pp_from_uni[key] = info
            portfolio = portfolio_backtest(pp_from_uni, signal_index, opens, highs, lows, n_bars, use_universal=False)
        else:
            continue

        pnls = [t[2] for t in portfolio]
        mc_p = mc_test_multi(pnls)
        stats = calc_compound(portfolio)
        print(f'  {name}: IS PnL=+{stats["pnl"]:.1f}% MC p={mc_p:.6f} {"PASS" if mc_p < 0.01 else "FAIL"}')

    # ══════════════════════════════════════════════════════════════
    # Final Summary
    # ══════════════════════════════════════════════════════════════
    print('\n' + '=' * 100)
    print('FINAL SUMMARY')
    print('=' * 100)

    ranked = sorted(all_wf_results.items(), key=lambda x: x[1]['avg_oos_pnl'], reverse=True)
    print(f'\n  Ranking by Expanding WF OOS PnL:')
    for rank, (name, res) in enumerate(ranked, 1):
        marker = '<-- BEST' if rank == 1 else ''
        print(f'    {rank}. {name:<25} Avg OOS: {res["avg_oos_pnl"]:>+8.1f}% | '
              f'Pos: {res["positive_folds"]}/3 | Avg Disc: {res["avg_discovered"]:.0f} | '
              f'WR: {res["avg_wr"]:.1f}% | MDD: {res["avg_mdd"]:.1f}% {marker}')

    uni_oos = all_wf_results['A_UniDisc_UniTrade']['avg_oos_pnl']
    pp_oos = all_wf_results['B_PPDisc_PPTrade']['avg_oos_pnl']
    diff = pp_oos - uni_oos

    print(f'\n  Key Comparison:')
    print(f'    A (Uni→Uni):  {uni_oos:+.1f}%')
    print(f'    B (PP→PP):    {pp_oos:+.1f}%')
    print(f'    Difference:   {diff:+.1f}% ({diff/abs(uni_oos)*100:+.1f}% relative)')

    print(f'\n  Pattern Discovery:')
    print(f'    Universal discovers: {len(uni_disc)} patterns')
    print(f'    Per-pattern discovers: {len(pp_disc)} patterns')
    print(f'    PP-only (hidden from Universal): {len(pp_only)} patterns')

    if pp_oos > uni_oos:
        verdict = f'Per-Pattern Discovery SUPERIOR ({diff:+.1f}% OOS) — current Universal-based selection is suboptimal'
    elif abs(diff) < uni_oos * 0.1:
        verdict = f'COMPARABLE (diff {diff:+.1f}%, <10% relative) — neither method clearly dominates'
    else:
        verdict = f'Universal Discovery SUPERIOR ({-diff:+.1f}% OOS) — per-pattern discovery adds overfitting'

    print(f'\n  VERDICT: {verdict}')

    # Save
    elapsed = time.time() - t0
    output = {
        'analysis': 'fair_discovery_comparison',
        'data_bars': n_bars,
        'elapsed_sec': round(elapsed, 1),
        'discovery': {
            'universal': {
                'count': len(uni_disc),
                'patterns': {k: {kk: vv for kk, vv in v.items()} for k, v in uni_disc.items()},
            },
            'per_pattern': {
                'count': len(pp_disc),
                'patterns': {k: {kk: vv for kk, vv in v.items()} for k, v in pp_disc.items()},
            },
            'overlap': len(overlap),
            'uni_only': list(uni_only),
            'pp_only': list(pp_only),
        },
        'expanding_wf': {name: {k: v for k, v in res.items()} for name, res in all_wf_results.items()},
        'ranking': [{'rank': i+1, 'name': name, 'avg_oos': res['avg_oos_pnl']}
                    for i, (name, res) in enumerate(ranked)],
        'verdict': verdict,
    }

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output, f, indent=2)
    print(f'\nResults saved: {OUTPUT_FILE}')
    print(f'Elapsed: {elapsed:.0f}s')


if __name__ == '__main__':
    main()
