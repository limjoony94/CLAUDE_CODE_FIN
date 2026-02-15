#!/usr/bin/env python3
"""
Per-Pattern TP/SL Deep Validation — v1.28.5 심층 검증

기존 per_pattern_optimization.py의 k-fold WF 문제점 수정:
1. Expanding Window WF (IS=[0..T], OOS=[T..T+1]) — 미래 데이터 누출 방지
2. Multi-seed MC (3 seeds, max p-value < 0.01)
3. Sensitivity analysis (±0.2% TP/SL 변동 시 PnL 변화)
4. Fine grid (0.1% resolution) re-optimization
5. Per-pattern WF stability (fold간 최적 TP/SL 일관성)
6. Portfolio risk metrics (consecutive losses, recovery)

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

# ── Constants ──
FEE_PCT = 0.10
LEVERAGE = 3
MAX_BARS = 500
MC_SIMS = 5000
MC_SEEDS = [42, 123, 7777]  # Multi-seed MC

DATA_FILE = os.path.join(_PROJECT_ROOT, 'data', 'btc_5m_270days_reclassified.csv')
DYNAMIC_FILE = os.path.join(_PROJECT_ROOT, 'results', 'dynamic_patterns.json')
OUTPUT_FILE = os.path.join(_PROJECT_ROOT, 'results', 'per_pattern_deep_validation.json')

# Fine grid for re-optimization (0.1% resolution)
TP_FINE = [round(x * 0.1, 1) for x in range(3, 35)]  # 0.3 ~ 3.4
SL_FINE = [round(x * 0.1, 1) for x in range(5, 45)]  # 0.5 ~ 4.4

# Coarse grid (same as original, for comparison)
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
                'avg_hold': 0, 'med_hold': 0, 'max_consec_loss': 0}
    equity = 100.0
    peak = 100.0
    mdd = 0.0
    wins = []
    losses = []
    holds = []
    consec_loss = 0
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
            consec_loss = 0
        else:
            losses.append(p)
            consec_loss += 1
            if consec_loss > max_cl:
                max_cl = consec_loss
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
        'med_hold': round(np.median(holds), 1) if holds else 0,
        'max_consec_loss': max_cl,
    }


def mc_test_multi_seed(pnls, seeds=MC_SEEDS, n_sims=MC_SIMS):
    """Multi-seed MC: returns max p-value across seeds (conservative)."""
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
    return max(p_values)  # Conservative: worst seed


def grid_search_pattern(signal_bars, direction, opens, highs, lows, n_bars,
                        tp_grid=None, sl_grid=None, min_trades=20):
    """Grid search optimal TP/SL for a single pattern."""
    if tp_grid is None:
        tp_grid = TP_FINE
    if sl_grid is None:
        sl_grid = SL_FINE
    best = None
    best_score = -9999

    for tp in tp_grid:
        for sl in sl_grid:
            if sl < 0.5:
                continue
            trades = bt_signals(signal_bars, direction, tp, sl, opens, highs, lows, n_bars)
            if len(trades) < min_trades:
                continue
            stats = calc_compound(trades)
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
                }
    return best


def expanding_wf(signal_index, long_pats, short_pats,
                 opens, highs, lows, n_bars, n_folds=3,
                 use_per_pattern=False, fixed_tpsl=None):
    """
    Expanding Window Walk-Forward (correct methodology).
    IS = [0..split], OOS = [split..split+fold_size]
    Split advances each fold.
    """
    fold_size = n_bars // (n_folds + 1)
    results = []

    for fold in range(n_folds):
        train_end = (fold + 1) * fold_size
        test_start = train_end
        test_end = (fold + 2) * fold_size if fold < n_folds - 1 else n_bars

        if use_per_pattern:
            # Optimize on train period, test on OOS
            all_test_trades = []
            for pat in long_pats:
                if pat not in signal_index:
                    continue
                train_bars = [b for b in signal_index[pat] if b < train_end]
                if len(train_bars) < 10:
                    # Not enough train data, use universal
                    test_bars = [b for b in signal_index[pat] if test_start <= b < test_end]
                    trades = bt_signals(test_bars, 'LONG', UNIVERSAL_TP, UNIVERSAL_SL,
                                        opens, highs, lows, n_bars)
                    all_test_trades.extend(trades)
                    continue
                opt = grid_search_pattern(train_bars, 'LONG', opens, highs, lows, n_bars,
                                          tp_grid=TP_COARSE, sl_grid=SL_COARSE, min_trades=10)
                if opt:
                    test_bars = [b for b in signal_index[pat] if test_start <= b < test_end]
                    trades = bt_signals(test_bars, 'LONG', opt['tp'], opt['sl'],
                                        opens, highs, lows, n_bars)
                    all_test_trades.extend(trades)

            for pat in short_pats:
                if pat not in signal_index:
                    continue
                train_bars = [b for b in signal_index[pat] if b < train_end]
                if len(train_bars) < 10:
                    test_bars = [b for b in signal_index[pat] if test_start <= b < test_end]
                    trades = bt_signals(test_bars, 'SHORT', UNIVERSAL_TP, UNIVERSAL_SL,
                                        opens, highs, lows, n_bars)
                    all_test_trades.extend(trades)
                    continue
                opt = grid_search_pattern(train_bars, 'SHORT', opens, highs, lows, n_bars,
                                          tp_grid=TP_COARSE, sl_grid=SL_COARSE, min_trades=10)
                if opt:
                    test_bars = [b for b in signal_index[pat] if test_start <= b < test_end]
                    trades = bt_signals(test_bars, 'SHORT', opt['tp'], opt['sl'],
                                        opens, highs, lows, n_bars)
                    all_test_trades.extend(trades)
        else:
            # Universal or fixed TP/SL
            tp_use = fixed_tpsl[0] if fixed_tpsl else UNIVERSAL_TP
            sl_use = fixed_tpsl[1] if fixed_tpsl else UNIVERSAL_SL
            all_test_trades = []
            for pat in long_pats:
                if pat in signal_index:
                    test_bars = [b for b in signal_index[pat] if test_start <= b < test_end]
                    trades = bt_signals(test_bars, 'LONG', tp_use, sl_use,
                                        opens, highs, lows, n_bars)
                    all_test_trades.extend(trades)
            for pat in short_pats:
                if pat in signal_index:
                    test_bars = [b for b in signal_index[pat] if test_start <= b < test_end]
                    trades = bt_signals(test_bars, 'SHORT', tp_use, sl_use,
                                        opens, highs, lows, n_bars)
                    all_test_trades.extend(trades)

        portfolio = portfolio_1pos(all_test_trades)
        stats = calc_compound(portfolio)
        be_wr = sl_use / (tp_use + sl_use) * 100 if not use_per_pattern and fixed_tpsl is None else 0
        oos_edge = stats['wr'] - be_wr if be_wr > 0 else 0

        results.append({
            'fold': fold,
            'train_range': f'0-{train_end}',
            'test_range': f'{test_start}-{test_end}',
            **stats,
            'oos_edge_pp': round(oos_edge, 1),
        })

    return results


def sensitivity_analysis(signal_bars, direction, base_tp, base_sl, opens, highs, lows, n_bars):
    """Check PnL sensitivity to ±0.1, ±0.2% TP/SL changes."""
    results = {}
    for dtp in [-0.2, -0.1, 0, 0.1, 0.2]:
        for dsl in [-0.2, -0.1, 0, 0.1, 0.2]:
            tp = round(base_tp + dtp, 1)
            sl = round(base_sl + dsl, 1)
            if tp <= 0 or sl < 0.5:
                continue
            trades = bt_signals(signal_bars, direction, tp, sl, opens, highs, lows, n_bars)
            stats = calc_compound(trades)
            key = f'dTP{dtp:+.1f}_dSL{dsl:+.1f}'
            results[key] = {
                'tp': tp, 'sl': sl,
                'trades': stats['trades'],
                'wr': stats['wr'],
                'pnl': stats['pnl'],
                'mdd': stats['mdd'],
                'pnl_mdd': stats['pnl_mdd'],
            }
    return results


def tp_sl_stability_across_folds(signal_index, pat, direction, opens, highs, lows, n_bars, n_folds=3):
    """Check if optimal TP/SL is consistent across expanding window folds."""
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
        opt = grid_search_pattern(train_bars, direction, opens, highs, lows, n_bars,
                                  tp_grid=TP_COARSE, sl_grid=SL_COARSE, min_trades=10)
        fold_optima.append(opt)

    return fold_optima


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

    print('=' * 90)
    print(f'Per-Pattern TP/SL Deep Validation (v1.28.5)')
    print(f'Patterns: {total_pats} (L:{len(long_pats)} S:{len(short_pats)}) | Data: {n_bars} bars (270d)')
    print(f'Expanding Window WF | Multi-seed MC | Sensitivity | Fine Grid')
    print('=' * 90)

    # ══════════════════════════════════════════════════════════════
    # Phase 1: Expanding Window WF — Universal vs Per-Pattern
    # ══════════════════════════════════════════════════════════════
    print('\n' + '─' * 90)
    print('Phase 1: Expanding Window WF (3-fold) — Universal vs Per-Pattern')
    print('─' * 90)

    print('\n  [Universal TP 2.0 / SL 3.0]')
    uni_ewf = expanding_wf(signal_index, long_pats, short_pats,
                           opens, highs, lows, n_bars, n_folds=3,
                           use_per_pattern=False)
    uni_oos_pos = 0
    for f in uni_ewf:
        marker = '+' if f['pnl'] > 0 else 'X'
        print(f'    Fold {f["fold"]}: IS=[{f["train_range"]}] OOS=[{f["test_range"]}] | '
              f'Tr={f["trades"]} WR={f["wr"]}% PnL={f["pnl"]:+.1f}% MDD={f["mdd"]:.1f}% '
              f'Edge={f["oos_edge_pp"]:+.1f}pp [{marker}]')
        if f['pnl'] > 0:
            uni_oos_pos += 1
    uni_avg_oos = np.mean([f['pnl'] for f in uni_ewf])
    uni_avg_edge = np.mean([f['oos_edge_pp'] for f in uni_ewf])
    print(f'    Avg OOS PnL: {uni_avg_oos:+.1f}% | Avg Edge: {uni_avg_edge:+.1f}pp | '
          f'Positive: {uni_oos_pos}/3')

    print('\n  [Per-Pattern Optimized (expanding train)]')
    pp_ewf = expanding_wf(signal_index, long_pats, short_pats,
                          opens, highs, lows, n_bars, n_folds=3,
                          use_per_pattern=True)
    pp_oos_pos = 0
    for f in pp_ewf:
        marker = '+' if f['pnl'] > 0 else 'X'
        print(f'    Fold {f["fold"]}: IS=[{f["train_range"]}] OOS=[{f["test_range"]}] | '
              f'Tr={f["trades"]} WR={f["wr"]}% PnL={f["pnl"]:+.1f}% MDD={f["mdd"]:.1f}% [{marker}]')
        if f['pnl'] > 0:
            pp_oos_pos += 1
    pp_avg_oos = np.mean([f['pnl'] for f in pp_ewf])
    print(f'    Avg OOS PnL: {pp_avg_oos:+.1f}% | Positive: {pp_oos_pos}/3')

    ewf_verdict = 'Per-Pattern WINS' if pp_avg_oos > uni_avg_oos and pp_oos_pos >= 2 else \
                  'Universal WINS' if uni_avg_oos > pp_avg_oos and uni_oos_pos >= 2 else 'INCONCLUSIVE'
    print(f'\n  Expanding WF Verdict: {ewf_verdict}')
    print(f'  Universal Avg OOS: {uni_avg_oos:+.1f}% vs Per-Pattern Avg OOS: {pp_avg_oos:+.1f}%')

    # ══════════════════════════════════════════════════════════════
    # Phase 2: Multi-Seed MC Test
    # ══════════════════════════════════════════════════════════════
    print('\n' + '─' * 90)
    print('Phase 2: Multi-Seed MC Test (3 seeds, conservative max p-value)')
    print('─' * 90)

    # Full portfolio MC with current per-pattern TP/SL
    all_pp_trades = []
    for pat in long_pats:
        if pat in signal_index and pat in current_tpsl:
            tp, sl = current_tpsl[pat]
            trades = bt_signals(signal_index[pat], 'LONG', tp, sl, opens, highs, lows, n_bars)
            all_pp_trades.extend(trades)
    for pat in short_pats:
        if pat in signal_index and pat in current_tpsl:
            tp, sl = current_tpsl[pat]
            trades = bt_signals(signal_index[pat], 'SHORT', tp, sl, opens, highs, lows, n_bars)
            all_pp_trades.extend(trades)

    pp_portfolio = portfolio_1pos(all_pp_trades)
    pp_pnls = [t[2] for t in pp_portfolio]
    pp_mc_multi = mc_test_multi_seed(pp_pnls)
    pp_stats = calc_compound(pp_portfolio)

    print(f'  Portfolio: {pp_stats["trades"]} trades, WR {pp_stats["wr"]}%, '
          f'PnL +{pp_stats["pnl"]:.1f}%, MDD {pp_stats["mdd"]:.1f}%')
    print(f'  Multi-seed MC p-value (max): {pp_mc_multi:.6f} '
          f'{"PASS" if pp_mc_multi < 0.01 else "FAIL"}')

    # Per-pattern MC
    print('\n  Per-pattern MC test (multi-seed):')
    pat_mc_results = {}
    mc_fail_count = 0
    for pat in long_pats:
        if pat not in signal_index or pat not in current_tpsl:
            continue
        tp, sl = current_tpsl[pat]
        trades = bt_signals(signal_index[pat], 'LONG', tp, sl, opens, highs, lows, n_bars)
        pnls = [t[2] for t in trades]
        mc_p = mc_test_multi_seed(pnls)
        pat_mc_results[pat] = {'mc_p': mc_p, 'trades': len(trades), 'dir': 'LONG'}
        if mc_p >= 0.01:
            mc_fail_count += 1
            print(f'    FAIL: {pat} (LONG) p={mc_p:.4f} trades={len(trades)}')

    for pat in short_pats:
        if pat not in signal_index or pat not in current_tpsl:
            continue
        tp, sl = current_tpsl[pat]
        trades = bt_signals(signal_index[pat], 'SHORT', tp, sl, opens, highs, lows, n_bars)
        pnls = [t[2] for t in trades]
        mc_p = mc_test_multi_seed(pnls)
        pat_mc_results[pat] = {'mc_p': mc_p, 'trades': len(trades), 'dir': 'SHORT'}
        if mc_p >= 0.01:
            mc_fail_count += 1
            print(f'    FAIL: {pat} (SHORT) p={mc_p:.4f} trades={len(trades)}')

    mc_pass = len(pat_mc_results) - mc_fail_count
    print(f'  Individual MC: {mc_pass}/{len(pat_mc_results)} PASS, {mc_fail_count} FAIL')

    # ══════════════════════════════════════════════════════════════
    # Phase 3: Fine Grid Re-optimization
    # ══════════════════════════════════════════════════════════════
    print('\n' + '─' * 90)
    print('Phase 3: Fine Grid Re-optimization (0.1% resolution)')
    print(f'  Grid: TP {len(TP_FINE)} values x SL {len(SL_FINE)} values = {len(TP_FINE)*len(SL_FINE)} combos/pattern')
    print('─' * 90)

    fine_results = {}
    changed_count = 0
    all_fine_trades = []

    for pat in long_pats:
        if pat not in signal_index:
            continue
        opt = grid_search_pattern(signal_index[pat], 'LONG', opens, highs, lows, n_bars,
                                  tp_grid=TP_FINE, sl_grid=SL_FINE, min_trades=20)
        if opt:
            cur = current_tpsl.get(pat, [None, None])
            changed = (cur[0] != opt['tp'] or cur[1] != opt['sl']) if cur[0] else True
            fine_results[pat] = {**opt, 'direction': 'LONG', 'changed': changed,
                                 'prev_tp': cur[0], 'prev_sl': cur[1]}
            if changed:
                changed_count += 1
            trades = bt_signals(signal_index[pat], 'LONG', opt['tp'], opt['sl'],
                                opens, highs, lows, n_bars)
            all_fine_trades.extend(trades)

    for pat in short_pats:
        if pat not in signal_index:
            continue
        opt = grid_search_pattern(signal_index[pat], 'SHORT', opens, highs, lows, n_bars,
                                  tp_grid=TP_FINE, sl_grid=SL_FINE, min_trades=20)
        if opt:
            cur = current_tpsl.get(pat, [None, None])
            changed = (cur[0] != opt['tp'] or cur[1] != opt['sl']) if cur[0] else True
            fine_results[pat] = {**opt, 'direction': 'SHORT', 'changed': changed,
                                 'prev_tp': cur[0], 'prev_sl': cur[1]}
            if changed:
                changed_count += 1
            trades = bt_signals(signal_index[pat], 'SHORT', opt['tp'], opt['sl'],
                                opens, highs, lows, n_bars)
            all_fine_trades.extend(trades)

    fine_portfolio = portfolio_1pos(all_fine_trades)
    fine_stats = calc_compound(fine_portfolio)

    print(f'\n  Fine grid: {len(fine_results)} patterns optimized, {changed_count} changed vs coarse grid')
    print(f'  Fine Portfolio: Tr={fine_stats["trades"]} WR={fine_stats["wr"]}% '
          f'PnL=+{fine_stats["pnl"]:.1f}% MDD={fine_stats["mdd"]:.1f}% '
          f'PnL/MDD={fine_stats["pnl_mdd"]:.1f}x')
    print(f'  vs Current (coarse): Tr={pp_stats["trades"]} WR={pp_stats["wr"]}% '
          f'PnL=+{pp_stats["pnl"]:.1f}% MDD={pp_stats["mdd"]:.1f}% '
          f'PnL/MDD={pp_stats["pnl_mdd"]:.1f}x')

    # Show changed patterns
    if changed_count > 0:
        print(f'\n  Changed patterns (fine vs coarse):')
        print(f'  {"Pattern":<20} {"Dir":>5} {"Prev TP":>7} {"Fine TP":>7} {"Prev SL":>7} {"Fine SL":>7} '
              f'{"PnL%":>8} {"PnL/MDD":>8}')
        for pat in sorted(fine_results.keys()):
            r = fine_results[pat]
            if r['changed']:
                print(f'  {pat:<20} {r["direction"]:>5} {r["prev_tp"]:>7.1f} {r["tp"]:>7.1f} '
                      f'{r["prev_sl"]:>7.1f} {r["sl"]:>7.1f} {r["pnl"]:>7.1f}% {r["pnl_mdd"]:>7.1f}x')

    # ══════════════════════════════════════════════════════════════
    # Phase 4: Sensitivity Analysis (top-10 + worst-3 patterns)
    # ══════════════════════════════════════════════════════════════
    print('\n' + '─' * 90)
    print('Phase 4: Sensitivity Analysis (±0.2% TP/SL shifts)')
    print('─' * 90)

    # Select patterns for sensitivity: top-5 by trade count + bottom-3 by PnL
    pat_by_trades = sorted(fine_results.items(), key=lambda x: x[1]['trades'], reverse=True)
    pat_by_pnl = sorted(fine_results.items(), key=lambda x: x[1]['pnl'])
    selected_pats = []
    seen = set()
    for p, _ in pat_by_trades[:5]:
        if p not in seen:
            selected_pats.append(p)
            seen.add(p)
    for p, _ in pat_by_pnl[:3]:
        if p not in seen:
            selected_pats.append(p)
            seen.add(p)

    sensitivity_results = {}
    for pat in selected_pats:
        r = fine_results[pat]
        direction = r['direction']
        if pat not in signal_index:
            continue
        sens = sensitivity_analysis(signal_index[pat], direction, r['tp'], r['sl'],
                                    opens, highs, lows, n_bars)
        sensitivity_results[pat] = sens

        # Find PnL range
        pnl_values = [s['pnl'] for s in sens.values()]
        base_pnl = sens.get('dTP+0.0_dSL+0.0', {}).get('pnl', 0)
        if pnl_values:
            pnl_min = min(pnl_values)
            pnl_max = max(pnl_values)
            pnl_range = pnl_max - pnl_min
            stability = 'STABLE' if pnl_range < base_pnl * 0.3 else \
                        'MODERATE' if pnl_range < base_pnl * 0.6 else 'FRAGILE'
            print(f'  {pat} ({direction}): TP={r["tp"]}% SL={r["sl"]}% | '
                  f'PnL range [{pnl_min:+.1f}%, {pnl_max:+.1f}%] | '
                  f'Variation: {pnl_range:.1f}% | {stability}')

    # ══════════════════════════════════════════════════════════════
    # Phase 5: TP/SL Stability Across Expanding WF Folds
    # ══════════════════════════════════════════════════════════════
    print('\n' + '─' * 90)
    print('Phase 5: TP/SL Stability Across Expanding WF Folds')
    print('─' * 90)

    stability_results = {}
    stable_count = 0
    unstable_count = 0

    all_pats_dirs = [(p, 'LONG') for p in long_pats] + [(p, 'SHORT') for p in short_pats]
    for pat, direction in all_pats_dirs:
        fold_optima = tp_sl_stability_across_folds(
            signal_index, pat, direction, opens, highs, lows, n_bars, n_folds=3)

        valid_optima = [o for o in fold_optima if o is not None]
        if len(valid_optima) < 2:
            continue

        tps = [o['tp'] for o in valid_optima]
        sls = [o['sl'] for o in valid_optima]
        tp_std = np.std(tps)
        sl_std = np.std(sls)
        tp_range = max(tps) - min(tps)
        sl_range = max(sls) - min(sls)
        is_stable = tp_range <= 0.5 and sl_range <= 1.0

        stability_results[pat] = {
            'direction': direction,
            'fold_optima': [{'tp': o['tp'], 'sl': o['sl']} if o else None for o in fold_optima],
            'tp_range': round(tp_range, 1),
            'sl_range': round(sl_range, 1),
            'tp_std': round(tp_std, 2),
            'sl_std': round(sl_std, 2),
            'stable': is_stable,
        }
        if is_stable:
            stable_count += 1
        else:
            unstable_count += 1

    print(f'  Stable (TP range <=0.5, SL range <=1.0): {stable_count}/{stable_count + unstable_count}')
    print(f'  Unstable: {unstable_count}/{stable_count + unstable_count}')

    # Show unstable patterns
    if unstable_count > 0:
        print(f'\n  Unstable patterns (TP/SL varies significantly across folds):')
        print(f'  {"Pattern":<20} {"Dir":>5} {"TP range":>8} {"SL range":>8} {"Fold TPs":>25} {"Fold SLs":>25}')
        for pat in sorted(stability_results.keys()):
            r = stability_results[pat]
            if not r['stable']:
                tps = [f'{o["tp"]:.1f}' if o else 'N/A' for o in r['fold_optima']]
                sls = [f'{o["sl"]:.1f}' if o else 'N/A' for o in r['fold_optima']]
                print(f'  {pat:<20} {r["direction"]:>5} {r["tp_range"]:>7.1f}% {r["sl_range"]:>7.1f}% '
                      f'{",".join(tps):>25} {",".join(sls):>25}')

    # ══════════════════════════════════════════════════════════════
    # Phase 6: Portfolio Risk Metrics
    # ══════════════════════════════════════════════════════════════
    print('\n' + '─' * 90)
    print('Phase 6: Portfolio Risk Metrics')
    print('─' * 90)

    # Current per-pattern portfolio
    pnl_series = [t[2] for t in pp_portfolio]
    if pnl_series:
        avg_win = np.mean([p for p in pnl_series if p > 0]) if any(p > 0 for p in pnl_series) else 0
        avg_loss = np.mean([abs(p) for p in pnl_series if p <= 0]) if any(p <= 0 for p in pnl_series) else 0
        max_single_loss = max(abs(p) for p in pnl_series if p <= 0) if any(p <= 0 for p in pnl_series) else 0
        max_single_win = max(p for p in pnl_series if p > 0) if any(p > 0 for p in pnl_series) else 0

        # Consecutive losses
        max_cl = 0
        cl = 0
        for p in pnl_series:
            if p <= 0:
                cl += 1
                if cl > max_cl:
                    max_cl = cl
            else:
                cl = 0

        # Win streaks
        max_ws = 0
        ws = 0
        for p in pnl_series:
            if p > 0:
                ws += 1
                if ws > max_ws:
                    max_ws = ws
            else:
                ws = 0

        # Drawdown recovery (bars)
        equity_curve = [100.0]
        for p in pnl_series:
            equity_curve.append(equity_curve[-1] * (1 + p / 100))
        peak = equity_curve[0]
        in_dd = False
        dd_start = 0
        max_recovery = 0
        for i, eq in enumerate(equity_curve):
            if eq >= peak:
                if in_dd:
                    recovery_len = i - dd_start
                    if recovery_len > max_recovery:
                        max_recovery = recovery_len
                    in_dd = False
                peak = eq
            elif not in_dd:
                in_dd = True
                dd_start = i

        print(f'  Total Trades: {pp_stats["trades"]}')
        print(f'  Win Rate: {pp_stats["wr"]}%')
        print(f'  Avg Win: +{avg_win:.2f}% | Avg Loss: -{avg_loss:.2f}%')
        print(f'  Max Single Win: +{max_single_win:.2f}% | Max Single Loss: -{max_single_loss:.2f}%')
        print(f'  Max Consecutive Losses: {max_cl}')
        print(f'  Max Win Streak: {max_ws}')
        print(f'  Max Drawdown: {pp_stats["mdd"]:.1f}%')
        print(f'  Longest Recovery: {max_recovery} trades')
        print(f'  Profit Factor: {pp_stats["pf"]}')
        print(f'  Max Daily Loss Risk: {max_single_loss:.2f}% '
              f'{"< 13% SAFE" if max_single_loss < 13 else ">= 13% DANGER"}')

    # ══════════════════════════════════════════════════════════════
    # Phase 7: Final Summary & Recommendations
    # ══════════════════════════════════════════════════════════════
    print('\n' + '=' * 90)
    print('FINAL SUMMARY')
    print('=' * 90)

    print(f'\n  1. Expanding Window WF (correct methodology):')
    print(f'     Universal:    Avg OOS {uni_avg_oos:+.1f}% | Positive {uni_oos_pos}/3')
    print(f'     Per-Pattern:  Avg OOS {pp_avg_oos:+.1f}% | Positive {pp_oos_pos}/3')
    print(f'     Verdict: {ewf_verdict}')

    print(f'\n  2. Multi-Seed MC (conservative):')
    print(f'     Portfolio p={pp_mc_multi:.6f} {"PASS" if pp_mc_multi < 0.01 else "FAIL"}')
    print(f'     Individual: {mc_pass}/{len(pat_mc_results)} PASS, {mc_fail_count} FAIL')

    print(f'\n  3. Fine Grid (0.1% resolution):')
    print(f'     {changed_count}/{len(fine_results)} patterns have different optima vs coarse grid')
    print(f'     Fine PnL: +{fine_stats["pnl"]:.1f}% vs Coarse PnL: +{pp_stats["pnl"]:.1f}%')

    print(f'\n  4. TP/SL Stability:')
    print(f'     {stable_count}/{stable_count + unstable_count} patterns have stable TP/SL across folds')

    print(f'\n  5. Risk:')
    if pnl_series:
        print(f'     Max Consec Loss: {max_cl} | Max Single Loss: -{max_single_loss:.2f}%')
        print(f'     MDD: {pp_stats["mdd"]:.1f}%')

    # Overall assessment
    issues = []
    if ewf_verdict != 'Per-Pattern WINS':
        issues.append(f'Expanding WF: {ewf_verdict}')
    if pp_mc_multi >= 0.01:
        issues.append(f'Portfolio MC FAIL (p={pp_mc_multi:.4f})')
    if mc_fail_count > 0:
        issues.append(f'{mc_fail_count} individual patterns MC FAIL')
    if stable_count < (stable_count + unstable_count) * 0.5:
        issues.append(f'<50% patterns have stable TP/SL')

    if not issues:
        overall = 'STRONG — Per-pattern optimization validated'
    elif len(issues) <= 1:
        overall = f'MODERATE — {issues[0]}'
    else:
        overall = f'WEAK — {len(issues)} issues: {"; ".join(issues)}'

    print(f'\n  OVERALL: {overall}')

    # ── Save ──
    elapsed = time.time() - t0
    output = {
        'analysis': 'per_pattern_deep_validation',
        'version': 'v1.28.5',
        'data_bars': n_bars,
        'patterns': total_pats,
        'elapsed_sec': round(elapsed, 1),
        'expanding_wf': {
            'universal': {
                'folds': uni_ewf,
                'avg_oos_pnl': round(uni_avg_oos, 1),
                'positive_folds': uni_oos_pos,
                'avg_edge_pp': round(uni_avg_edge, 1),
            },
            'per_pattern': {
                'folds': pp_ewf,
                'avg_oos_pnl': round(pp_avg_oos, 1),
                'positive_folds': pp_oos_pos,
            },
            'verdict': ewf_verdict,
        },
        'mc_test': {
            'portfolio_mc_p': pp_mc_multi,
            'portfolio_pass': pp_mc_multi < 0.01,
            'individual_pass': mc_pass,
            'individual_fail': mc_fail_count,
            'individual_total': len(pat_mc_results),
            'failed_patterns': {p: r for p, r in pat_mc_results.items() if r['mc_p'] >= 0.01},
        },
        'fine_grid': {
            'changed_count': changed_count,
            'total': len(fine_results),
            'fine_stats': fine_stats,
            'current_stats': {k: v for k, v in pp_stats.items()},
            'changed_patterns': {p: {k: v for k, v in r.items() if k != 'changed'}
                                 for p, r in fine_results.items() if r['changed']},
        },
        'sensitivity': sensitivity_results,
        'stability': {
            'stable_count': stable_count,
            'unstable_count': unstable_count,
            'unstable_patterns': {p: r for p, r in stability_results.items() if not r['stable']},
        },
        'risk': {
            'max_consec_loss': max_cl if pnl_series else 0,
            'max_single_loss': round(max_single_loss, 2) if pnl_series else 0,
            'mdd': pp_stats['mdd'],
            'profit_factor': pp_stats['pf'],
        },
        'overall': overall,
        'issues': issues,
    }

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output, f, indent=2)
    print(f'\nResults saved: {OUTPUT_FILE}')
    print(f'Elapsed: {elapsed:.0f}s')


if __name__ == '__main__':
    main()
