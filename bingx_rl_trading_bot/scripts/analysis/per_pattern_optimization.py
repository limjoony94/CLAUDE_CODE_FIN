#!/usr/bin/env python3
"""
Per-Pattern TP/SL Optimization vs Universal — 현재 47패턴 기준

현재 프로덕션 47패턴에 대해:
1. Universal TP 2.1/SL 3.0 백테스트 (baseline)
2. 개별 패턴별 TP/SL grid search 최적화
3. 최적화된 per-pattern TP/SL 포트폴리오 백테스트
4. 3-fold WF OOS 비교
5. 결과 JSON 저장

⚠️ Production classify_candle 사용, LEVERAGE=3, timeout DROP
"""

import os
import sys
import json
import numpy as np
import pandas as pd

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
sys.path.insert(0, _PROJECT_ROOT)

from scripts.production.pattern_5m.indicators import classify_candle
from scripts.production.pattern_5m.constants import AVG_BODY_WINDOW

# ── Constants ──
FEE_PCT = 0.10
LEVERAGE = 3
MAX_BARS = 500
MC_SIMS = 5000
DATA_FILE = os.path.join(_PROJECT_ROOT, 'data', 'btc_5m_270days_reclassified.csv')
PATTERNS_FILE = os.path.join(_PROJECT_ROOT, 'results', 'dynamic_patterns.json')
OUTPUT_FILE = os.path.join(_PROJECT_ROOT, 'results', 'per_pattern_optimization.json')

# Grid search ranges
TP_VALUES = [0.5, 0.7, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.1, 2.5, 3.0]
SL_VALUES = [0.5, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]

UNIVERSAL_TP = 2.1
UNIVERSAL_SL = 3.0


def load_data():
    df = pd.read_csv(DATA_FILE)
    with open(PATTERNS_FILE) as f:
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
    """Backtest: entry at next bar open, intrabar exit, timeout DROP."""
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
    """1-position-at-a-time filter."""
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
    """Compound PnL, MDD, WR, PF, hold stats."""
    if not trades:
        return {'pnl': 0, 'trades': 0, 'wr': 0, 'mdd': 0, 'pf': 0, 'pnl_mdd': 0,
                'avg_hold': 0, 'med_hold': 0}
    equity = 100.0
    peak = 100.0
    mdd = 0.0
    wins = []
    losses = []
    holds = []
    for eb, xb, p in trades:
        equity *= (1 + p / 100)
        if equity > peak:
            peak = equity
        dd = (peak - equity) / peak * 100
        if dd > mdd:
            mdd = dd
        if p > 0:
            wins.append(p)
        else:
            losses.append(p)
        holds.append((xb - eb) * 5 / 60)  # hours

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
        'med_hold': round(np.median(holds), 1) if holds else 0,
    }


def grid_search_pattern(signal_bars, direction, opens, highs, lows, n_bars):
    """Grid search optimal TP/SL for a single pattern. Maximize compound PnL/MDD."""
    best = None
    best_score = -9999

    for tp in TP_VALUES:
        for sl in SL_VALUES:
            if sl < 0.5:
                continue
            trades = bt_signals(signal_bars, direction, tp, sl, opens, highs, lows, n_bars)
            if len(trades) < 10:
                continue
            stats = calc_compound(trades)
            if stats['trades'] < 10:
                continue

            score = stats['pnl_mdd'] if stats['mdd'] > 0 else stats['pnl']

            if score > best_score:
                best_score = score
                best = {
                    'tp': tp, 'sl': sl,
                    'trades': stats['trades'],
                    'wr': stats['wr'],
                    'pnl': stats['pnl'],
                    'mdd': stats['mdd'],
                    'pnl_mdd': stats['pnl_mdd'],
                    'pf': stats['pf'],
                    'avg_hold': stats['avg_hold'],
                }

    return best


def mc_test(pnls, n_sims=MC_SIMS):
    if len(pnls) < 5:
        return 1.0
    pnls_arr = np.array(pnls)
    actual = np.sum(pnls_arr)
    signs = np.random.choice([-1, 1], size=(n_sims, len(pnls_arr)))
    rand_sums = signs @ pnls_arr
    return float(np.mean(rand_sums >= actual))


def walkforward_3fold(signal_index, patterns_long, patterns_short,
                      opens, highs, lows, n_bars, use_per_pattern=False,
                      per_pattern_tpsl=None):
    """3-fold walk-forward. Returns list of fold results."""
    fold_size = n_bars // 3
    results = []

    for fold in range(3):
        test_start = fold * fold_size
        test_end = (fold + 1) * fold_size if fold < 2 else n_bars
        train_ranges = []
        if fold > 0:
            train_ranges.append((0, fold * fold_size))
        if fold < 2:
            train_ranges.append(((fold + 1) * fold_size, n_bars))

        # Get train signal bars (in train range)
        def in_train(bar):
            for s, e in train_ranges:
                if s <= bar < e:
                    return True
            return False

        def in_test(bar):
            return test_start <= bar < test_end

        if use_per_pattern and per_pattern_tpsl:
            # Per-pattern: optimize on train, apply to test
            fold_train_optimal = {}
            for pat_key, direction in [(p + '_LONG', 'LONG') for p in patterns_long] + \
                                      [(p + '_SHORT', 'SHORT') for p in patterns_short]:
                pat = pat_key.rsplit('_', 1)[0]
                if pat not in signal_index:
                    continue
                train_bars = [b for b in signal_index[pat] if in_train(b)]
                if len(train_bars) < 10:
                    continue
                opt = grid_search_pattern(train_bars, direction, opens, highs, lows, n_bars)
                if opt:
                    fold_train_optimal[pat_key] = opt

            # Test with optimized TP/SL
            all_test_trades = []
            for pat_key, opt in fold_train_optimal.items():
                pat = pat_key.rsplit('_', 1)[0]
                direction = pat_key.rsplit('_', 1)[1]
                if pat not in signal_index:
                    continue
                test_bars = [b for b in signal_index[pat] if in_test(b)]
                trades = bt_signals(test_bars, direction, opt['tp'], opt['sl'],
                                    opens, highs, lows, n_bars)
                all_test_trades.extend(trades)

        else:
            # Universal TP/SL
            all_test_trades = []
            for pat in patterns_long:
                if pat not in signal_index:
                    continue
                test_bars = [b for b in signal_index[pat] if in_test(b)]
                trades = bt_signals(test_bars, 'LONG', UNIVERSAL_TP, UNIVERSAL_SL,
                                    opens, highs, lows, n_bars)
                all_test_trades.extend(trades)
            for pat in patterns_short:
                if pat not in signal_index:
                    continue
                test_bars = [b for b in signal_index[pat] if in_test(b)]
                trades = bt_signals(test_bars, 'SHORT', UNIVERSAL_TP, UNIVERSAL_SL,
                                    opens, highs, lows, n_bars)
                all_test_trades.extend(trades)

        portfolio = portfolio_1pos(all_test_trades)
        stats = calc_compound(portfolio)
        results.append({
            'fold': fold,
            'test_range': f'{test_start}-{test_end}',
            **stats
        })

    return results


def main():
    np.random.seed(42)
    df, dp = load_data()
    types, opens, highs, lows, closes = classify_all(df)
    n_bars = len(df)

    long_pats = dp['patterns']['long']
    short_pats = dp['patterns']['short']
    total_pats = len(long_pats) + len(short_pats)

    # Build signal index
    signal_index = {}
    for i in range(2, n_bars):
        pat = f"{types[i-2]}-{types[i-1]}-{types[i]}"
        if pat not in signal_index:
            signal_index[pat] = []
        signal_index[pat].append(i)

    print('=' * 80)
    print(f'Per-Pattern TP/SL Optimization vs Universal (TP {UNIVERSAL_TP}/SL {UNIVERSAL_SL})')
    print(f'패턴: {total_pats}개 (L:{len(long_pats)} S:{len(short_pats)}) | 데이터: {n_bars}봉 (270일)')
    print('=' * 80)

    # ── Phase 1: Universal Baseline ──
    print('\n── Phase 1: Universal Baseline ──')
    all_uni_trades = []
    for pat in long_pats:
        if pat in signal_index:
            trades = bt_signals(signal_index[pat], 'LONG', UNIVERSAL_TP, UNIVERSAL_SL,
                                opens, highs, lows, n_bars)
            all_uni_trades.extend(trades)
    for pat in short_pats:
        if pat in signal_index:
            trades = bt_signals(signal_index[pat], 'SHORT', UNIVERSAL_TP, UNIVERSAL_SL,
                                opens, highs, lows, n_bars)
            all_uni_trades.extend(trades)

    uni_portfolio = portfolio_1pos(all_uni_trades)
    uni_stats = calc_compound(uni_portfolio)
    uni_pnls = [t[2] for t in uni_portfolio]
    uni_mc = mc_test(uni_pnls)

    print(f'  Trades: {uni_stats["trades"]} | WR: {uni_stats["wr"]}% | '
          f'PnL: +{uni_stats["pnl"]}% | MDD: {uni_stats["mdd"]}% | '
          f'PnL/MDD: {uni_stats["pnl_mdd"]}x | PF: {uni_stats["pf"]} | MC: {uni_mc}')
    print(f'  Avg Hold: {uni_stats["avg_hold"]}h | Med Hold: {uni_stats["med_hold"]}h')

    # ── Phase 2: Per-Pattern Grid Search ──
    print('\n── Phase 2: Per-Pattern Grid Search Optimization ──')
    print(f'  Grid: TP {TP_VALUES} x SL {SL_VALUES} = {len(TP_VALUES)*len(SL_VALUES)} combos/pattern')

    per_pattern_results = {}
    all_pp_trades = []

    for pat in long_pats:
        if pat not in signal_index:
            continue
        opt = grid_search_pattern(signal_index[pat], 'LONG', opens, highs, lows, n_bars)
        if opt:
            per_pattern_results[f'{pat}_LONG'] = opt
            trades = bt_signals(signal_index[pat], 'LONG', opt['tp'], opt['sl'],
                                opens, highs, lows, n_bars)
            all_pp_trades.extend(trades)

    for pat in short_pats:
        if pat not in signal_index:
            continue
        opt = grid_search_pattern(signal_index[pat], 'SHORT', opens, highs, lows, n_bars)
        if opt:
            per_pattern_results[f'{pat}_SHORT'] = opt
            trades = bt_signals(signal_index[pat], 'SHORT', opt['tp'], opt['sl'],
                                opens, highs, lows, n_bars)
            all_pp_trades.extend(trades)

    pp_portfolio = portfolio_1pos(all_pp_trades)
    pp_stats = calc_compound(pp_portfolio)
    pp_pnls = [t[2] for t in pp_portfolio]
    pp_mc = mc_test(pp_pnls)

    print(f'\n  최적화된 패턴: {len(per_pattern_results)}/{total_pats}')
    print(f'  Trades: {pp_stats["trades"]} | WR: {pp_stats["wr"]}% | '
          f'PnL: +{pp_stats["pnl"]}% | MDD: {pp_stats["mdd"]}% | '
          f'PnL/MDD: {pp_stats["pnl_mdd"]}x | PF: {pp_stats["pf"]} | MC: {pp_mc}')
    print(f'  Avg Hold: {pp_stats["avg_hold"]}h | Med Hold: {pp_stats["med_hold"]}h')

    # ── Phase 3: 패턴별 상세 ──
    print('\n── Phase 3: 패턴별 최적 TP/SL ──')
    print(f'  {"Pattern":<22} {"Dir":>5} {"TP":>5} {"SL":>5} {"R:R":>5} '
          f'{"Tr":>4} {"WR%":>6} {"PnL%":>8} {"MDD%":>6} {"P/M":>6} {"Hold":>6}')
    print('  ' + '-' * 86)

    tp_dist = []
    sl_dist = []
    for pk in sorted(per_pattern_results.keys()):
        r = per_pattern_results[pk]
        pat, d = pk.rsplit('_', 1)
        rr = r['tp'] / r['sl'] if r['sl'] > 0 else 999
        tp_dist.append(r['tp'])
        sl_dist.append(r['sl'])
        print(f'  {pat:<22} {d:>5} {r["tp"]:>5.1f} {r["sl"]:>5.1f} {rr:>5.2f} '
              f'{r["trades"]:>4} {r["wr"]:>5.1f}% {r["pnl"]:>7.1f}% '
              f'{r["mdd"]:>5.1f}% {r["pnl_mdd"]:>5.1f}x {r["avg_hold"]:>5.1f}h')

    print(f'\n  TP 분포: min={min(tp_dist):.1f} med={np.median(tp_dist):.1f} '
          f'mean={np.mean(tp_dist):.1f} max={max(tp_dist):.1f}')
    print(f'  SL 분포: min={min(sl_dist):.1f} med={np.median(sl_dist):.1f} '
          f'mean={np.mean(sl_dist):.1f} max={max(sl_dist):.1f}')

    # Count by TP size
    tight = sum(1 for t in tp_dist if t <= 1.0)
    medium = sum(1 for t in tp_dist if 1.0 < t <= 2.0)
    wide = sum(1 for t in tp_dist if t > 2.0)
    print(f'  TP 크기: tight(≤1.0) {tight} | medium(1.0-2.0) {medium} | wide(>2.0) {wide}')

    # ── Phase 4: 직접 비교 ──
    print('\n── Phase 4: Universal vs Per-Pattern 직접 비교 ──')
    print(f'  {"":>15} {"Universal":>15} {"Per-Pattern":>15} {"차이":>15}')
    print('  ' + '-' * 60)

    rows = [
        ('Trades', uni_stats['trades'], pp_stats['trades']),
        ('WR%', uni_stats['wr'], pp_stats['wr']),
        ('PnL%', uni_stats['pnl'], pp_stats['pnl']),
        ('MDD%', uni_stats['mdd'], pp_stats['mdd']),
        ('PnL/MDD', uni_stats['pnl_mdd'], pp_stats['pnl_mdd']),
        ('PF', uni_stats['pf'], pp_stats['pf']),
        ('Avg Hold(h)', uni_stats['avg_hold'], pp_stats['avg_hold']),
        ('Med Hold(h)', uni_stats['med_hold'], pp_stats['med_hold']),
    ]
    for label, u, p in rows:
        if isinstance(u, float):
            diff = p - u
            pct = (p / u - 1) * 100 if u != 0 else 0
            print(f'  {label:>15} {u:>15.1f} {p:>15.1f} {diff:>+10.1f} ({pct:>+.0f}%)')
        else:
            diff = p - u
            print(f'  {label:>15} {u:>15} {p:>15} {diff:>+10}')

    # ── Phase 5: 3-Fold Walk-Forward ──
    print('\n── Phase 5: 3-Fold Walk-Forward OOS 비교 ──')

    print('\n  [Universal TP/SL WF]')
    uni_wf = walkforward_3fold(signal_index, long_pats, short_pats,
                               opens, highs, lows, n_bars,
                               use_per_pattern=False)
    uni_oos_positive = 0
    for f in uni_wf:
        marker = '✅' if f['pnl'] > 0 else '❌'
        print(f'    Fold {f["fold"]}: Trades={f["trades"]} WR={f["wr"]}% '
              f'PnL={f["pnl"]:+.1f}% MDD={f["mdd"]:.1f}% {marker}')
        if f['pnl'] > 0:
            uni_oos_positive += 1
    uni_avg_oos = np.mean([f['pnl'] for f in uni_wf])
    print(f'    Avg OOS PnL: {uni_avg_oos:+.1f}% | Positive: {uni_oos_positive}/3')

    print('\n  [Per-Pattern Optimized WF]')
    pp_wf = walkforward_3fold(signal_index, long_pats, short_pats,
                              opens, highs, lows, n_bars,
                              use_per_pattern=True,
                              per_pattern_tpsl=per_pattern_results)
    pp_oos_positive = 0
    for f in pp_wf:
        marker = '✅' if f['pnl'] > 0 else '❌'
        print(f'    Fold {f["fold"]}: Trades={f["trades"]} WR={f["wr"]}% '
              f'PnL={f["pnl"]:+.1f}% MDD={f["mdd"]:.1f}% {marker}')
        if f['pnl'] > 0:
            pp_oos_positive += 1
    pp_avg_oos = np.mean([f['pnl'] for f in pp_wf])
    print(f'    Avg OOS PnL: {pp_avg_oos:+.1f}% | Positive: {pp_oos_positive}/3')

    # ── Phase 6: 최종 판정 ──
    print('\n' + '=' * 80)
    print('최종 비교')
    print('=' * 80)
    print(f'  {"":>20} {"Universal":>15} {"Per-Pattern":>15}')
    print(f'  {"In-Sample PnL":>20} {uni_stats["pnl"]:>+14.1f}% {pp_stats["pnl"]:>+14.1f}%')
    print(f'  {"In-Sample PnL/MDD":>20} {uni_stats["pnl_mdd"]:>14.1f}x {pp_stats["pnl_mdd"]:>14.1f}x')
    print(f'  {"WF OOS Avg PnL":>20} {uni_avg_oos:>+14.1f}% {pp_avg_oos:>+14.1f}%')
    print(f'  {"WF Positive Folds":>20} {uni_oos_positive:>14}/3 {pp_oos_positive:>14}/3')
    print(f'  {"MC p-value":>20} {uni_mc:>14.4f} {pp_mc:>14.4f}')
    print(f'  {"Avg Hold":>20} {uni_stats["avg_hold"]:>13.1f}h {pp_stats["avg_hold"]:>13.1f}h')

    oos_winner = 'Universal' if uni_avg_oos > pp_avg_oos else 'Per-Pattern'
    is_winner = 'In-Sample' if pp_stats['pnl'] > uni_stats['pnl'] else 'Universal'
    print(f'\n  In-Sample 우위: {is_winner}')
    print(f'  OOS 우위: {oos_winner}')

    if pp_avg_oos > uni_avg_oos and pp_oos_positive >= 2:
        verdict = 'Per-Pattern BETTER on OOS — 전환 고려'
    elif uni_avg_oos > pp_avg_oos and uni_oos_positive >= 2:
        verdict = 'Universal BETTER on OOS — 현행 유지'
    else:
        verdict = 'INCONCLUSIVE — 추가 데이터 필요'
    print(f'  판정: {verdict}')

    # ── Save Results ──
    output = {
        'analysis': 'per_pattern_optimization_vs_universal',
        'patterns': total_pats,
        'data_bars': n_bars,
        'universal': {
            'tp': UNIVERSAL_TP, 'sl': UNIVERSAL_SL,
            'in_sample': uni_stats,
            'mc_p': uni_mc,
            'wf_folds': uni_wf,
            'wf_avg_oos_pnl': round(uni_avg_oos, 1),
            'wf_positive_folds': uni_oos_positive,
        },
        'per_pattern': {
            'in_sample': pp_stats,
            'mc_p': pp_mc,
            'pattern_details': per_pattern_results,
            'tp_distribution': {
                'min': round(min(tp_dist), 1), 'median': round(float(np.median(tp_dist)), 1),
                'mean': round(float(np.mean(tp_dist)), 1), 'max': round(max(tp_dist), 1),
            },
            'sl_distribution': {
                'min': round(min(sl_dist), 1), 'median': round(float(np.median(sl_dist)), 1),
                'mean': round(float(np.mean(sl_dist)), 1), 'max': round(max(sl_dist), 1),
            },
            'wf_folds': pp_wf,
            'wf_avg_oos_pnl': round(pp_avg_oos, 1),
            'wf_positive_folds': pp_oos_positive,
        },
        'verdict': verdict,
    }

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output, f, indent=2)
    print(f'\n결과 저장: {OUTPUT_FILE}')


if __name__ == '__main__':
    main()
