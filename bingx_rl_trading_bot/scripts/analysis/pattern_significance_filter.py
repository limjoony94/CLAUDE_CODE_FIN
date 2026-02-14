#!/usr/bin/env python3
"""
Pattern Statistical Significance Filter — v1.28.2 패턴 필터링 연구

현재 66개 패턴의 통계적 강도를 분석하고, 다양한 필터 기준별
포트폴리오 성과를 비교하여 최적 필터를 결정.

핵심 문제: min_trades=10은 MC test 신뢰도 부족 → 과적합 위험
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from collections import defaultdict

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
sys.path.insert(0, _PROJECT_ROOT)

from scripts.production.pattern_5m.indicators import classify_candle
from scripts.production.pattern_5m.constants import AVG_BODY_WINDOW

# ============================================================
# Constants
# ============================================================
FEE_PCT = 0.10
LEVERAGE = 3
MAX_BARS = 500
MC_SIMS = 5000
UNI_TP = 2.0
UNI_SL = 3.0
BASELINE_WR = UNI_SL / (UNI_TP + UNI_SL) * 100  # 60.0%

DATA_FILE = os.path.join(_PROJECT_ROOT, 'data', 'btc_5m_270days_reclassified.csv')
RESULTS_DIR = os.path.join(_PROJECT_ROOT, 'results')

# ============================================================
# Core backtest functions (from scanner)
# ============================================================

def load_and_classify(data_file):
    df = pd.read_csv(data_file)
    df['body'] = abs(df['close'] - df['open'])
    df['avg_body'] = df['body'].rolling(AVG_BODY_WINDOW, min_periods=1).mean()
    types = []
    for i, row in df.iterrows():
        ct = classify_candle(row, df.at[i, 'avg_body'])
        types.append(ct.value)
    df['candle_type'] = types
    return df


def build_signal_index(types, n):
    idx = {}
    for i in range(2, n):
        pat = f"{types[i-2]}-{types[i-1]}-{types[i]}"
        if pat not in idx:
            idx[pat] = []
        idx[pat].append(i)
    return idx


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


def mc_test(pnls, n_sims=MC_SIMS):
    if len(pnls) < 5:
        return 1.0
    pnls_arr = np.array(pnls)
    actual = np.sum(pnls_arr)
    signs = np.random.choice([-1, 1], size=(n_sims, len(pnls_arr)))
    rand_sums = signs @ pnls_arr
    return float(np.mean(rand_sums >= actual))


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


def calc_compound_stats(trades):
    """Compound 방식 통계 계산."""
    if not trades:
        return {'pnl': 0, 'trades': 0, 'wr': 0, 'mdd': 0, 'pf': 0, 'pnl_per_trade': 0}
    pnls = [t[2] for t in trades]
    equity = 100.0
    peak = 100.0
    mdd = 0
    wins = []
    losses = []
    for p in pnls:
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
    total_pnl = equity - 100.0
    wsum = sum(wins)
    lsum = sum(abs(x) for x in losses)
    return {
        'pnl': round(total_pnl, 1),
        'trades': len(pnls),
        'wr': round(len(wins) / len(pnls) * 100, 1) if pnls else 0,
        'mdd': round(mdd, 1),
        'pf': round(wsum / lsum, 2) if lsum > 0 else 999,
        'pnl_per_trade': round(total_pnl / len(pnls), 2) if pnls else 0,
    }


# ============================================================
# Walk-Forward per-pattern validation
# ============================================================

def wf_validate_pattern(types, opens, highs, lows, n, pat_name, direction, n_folds=3):
    """개별 패턴의 Walk-Forward 검증 (3-fold)."""
    fold_size = n // n_folds
    oos_results = []

    for fold in range(n_folds):
        oos_start = fold * fold_size
        oos_end = (fold + 1) * fold_size if fold < n_folds - 1 else n

        # Build signal index for full data
        all_sigs = []
        for i in range(2, n):
            pat = f"{types[i-2]}-{types[i-1]}-{types[i]}"
            if pat == pat_name:
                all_sigs.append(i)

        # Split into IS and OOS
        is_sigs = [s for s in all_sigs if s < oos_start or s >= oos_end]
        oos_sigs = [s for s in all_sigs if oos_start <= s < oos_end]

        if len(oos_sigs) < 3:
            continue

        # IS에서 MC 통과하는지 확인
        is_trades = bt_signals(is_sigs, direction, UNI_TP, UNI_SL, opens, highs, lows, n)
        if len(is_trades) < 5:
            oos_results.append({'oos_pnl': 0, 'oos_wr': 0, 'oos_trades': 0, 'is_pass': False})
            continue

        is_pnls = [t[2] for t in is_trades]
        is_wr = len([p for p in is_pnls if p > 0]) / len(is_pnls) * 100
        is_edge = is_wr - BASELINE_WR

        if is_edge < 5.0:
            oos_results.append({'oos_pnl': 0, 'oos_wr': 0, 'oos_trades': 0, 'is_pass': False})
            continue

        is_mc = mc_test(is_pnls)
        if is_mc >= 0.01:
            oos_results.append({'oos_pnl': 0, 'oos_wr': 0, 'oos_trades': 0, 'is_pass': False})
            continue

        # OOS 성과
        oos_trades = bt_signals(oos_sigs, direction, UNI_TP, UNI_SL, opens, highs, lows, n)
        if not oos_trades:
            oos_results.append({'oos_pnl': 0, 'oos_wr': 0, 'oos_trades': 0, 'is_pass': True})
            continue

        oos_pnls = [t[2] for t in oos_trades]
        oos_pnl = sum(oos_pnls)
        oos_wr = len([p for p in oos_pnls if p > 0]) / len(oos_pnls) * 100

        oos_results.append({
            'oos_pnl': round(oos_pnl, 2),
            'oos_wr': round(oos_wr, 1),
            'oos_trades': len(oos_trades),
            'is_pass': True,
        })

    if not oos_results:
        return {'wf_folds': 0, 'wf_positive': 0, 'avg_oos_wr': 0, 'avg_oos_pnl': 0}

    is_pass_folds = [r for r in oos_results if r['is_pass']]
    positive_folds = len([r for r in is_pass_folds if r['oos_pnl'] > 0])
    avg_oos_wr = np.mean([r['oos_wr'] for r in is_pass_folds]) if is_pass_folds else 0
    avg_oos_pnl = np.mean([r['oos_pnl'] for r in is_pass_folds]) if is_pass_folds else 0

    return {
        'wf_folds': len(is_pass_folds),
        'wf_positive': positive_folds,
        'avg_oos_wr': round(avg_oos_wr, 1),
        'avg_oos_pnl': round(avg_oos_pnl, 2),
    }


# ============================================================
# Main Analysis
# ============================================================

def main():
    np.random.seed(42)

    # Load data
    print("=" * 70)
    print("Pattern Statistical Significance Filter — v1.28.2")
    print("=" * 70)

    print("\n[1/5] Loading and classifying data...")
    df = load_and_classify(DATA_FILE)
    types = df['candle_type'].tolist()
    n = len(types)
    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values

    # Load current patterns
    with open(os.path.join(RESULTS_DIR, 'dynamic_patterns.json'), 'r') as f:
        current = json.load(f)

    details = current['pattern_details']
    print(f"Current patterns: {len(details)}")

    # ============================================================
    # Phase 1: Distribution Analysis
    # ============================================================
    print("\n[2/5] Distribution Analysis...")
    print("-" * 70)

    # Sort by trades
    sorted_pats = sorted(details.items(), key=lambda x: x[1]['trades'])

    trade_counts = [v['trades'] for v in details.values()]
    edges = [v['edge'] for v in details.values()]
    mc_ps = [v['mc_p'] for v in details.values()]

    print(f"\nTrade Count Distribution:")
    print(f"  Min: {min(trade_counts)}, Max: {max(trade_counts)}, Median: {np.median(trade_counts):.0f}")
    print(f"  <15: {len([t for t in trade_counts if t < 15])}")
    print(f"  <20: {len([t for t in trade_counts if t < 20])}")
    print(f"  <30: {len([t for t in trade_counts if t < 30])}")
    print(f"  >=30: {len([t for t in trade_counts if t >= 30])}")

    print(f"\nEdge Distribution (pp above baseline WR {BASELINE_WR:.0f}%):")
    print(f"  Min: {min(edges):.1f}, Max: {max(edges):.1f}, Median: {np.median(edges):.1f}")
    print(f"  <8pp: {len([e for e in edges if e < 8])}")
    print(f"  <10pp: {len([e for e in edges if e < 10])}")
    print(f"  >=10pp: {len([e for e in edges if e >= 10])}")

    print(f"\nMC p-value Distribution:")
    print(f"  Min: {min(mc_ps):.4f}, Max: {max(mc_ps):.4f}, Median: {np.median(mc_ps):.4f}")
    print(f"  <0.001: {len([p for p in mc_ps if p < 0.001])}")
    print(f"  <0.005: {len([p for p in mc_ps if p < 0.005])}")
    print(f"  <0.01: {len([p for p in mc_ps if p < 0.01])}")

    # Low-trade patterns detail
    print(f"\n--- Patterns with <20 trades ---")
    print(f"{'Pattern':<25} {'Trades':>6} {'WR':>6} {'Edge':>6} {'MC_p':>8}")
    for key, v in sorted_pats:
        if v['trades'] < 20:
            print(f"{key:<25} {v['trades']:>6} {v['wr']:>5.1f}% {v['edge']:>5.1f} {v['mc_p']:>8.4f}")

    # ============================================================
    # Phase 2: Walk-Forward per-pattern validation
    # ============================================================
    print(f"\n[3/5] Walk-Forward Validation (3-fold, per-pattern)...")
    print("-" * 70)

    wf_results = {}
    for key, v in sorted(details.items(), key=lambda x: x[1]['trades'], reverse=True):
        pat_name = v['pattern']
        direction = v['direction']
        wf = wf_validate_pattern(types, opens, highs, lows, n, pat_name, direction)
        wf_results[key] = wf

    # Print WF results
    print(f"\n{'Pattern':<25} {'Trades':>6} {'Edge':>6} {'MC_p':>8} {'WF':>5} {'OOS_WR':>7} {'OOS_PnL':>8} {'Verdict':>8}")
    print("-" * 80)

    verdicts = {}
    for key, v in sorted(details.items(), key=lambda x: x[1]['trades'], reverse=True):
        wf = wf_results[key]
        wf_str = f"{wf['wf_positive']}/{wf['wf_folds']}"

        # Verdict
        if wf['wf_folds'] == 0:
            verdict = "NO_DATA"
        elif wf['wf_positive'] >= 2 and wf['avg_oos_wr'] > BASELINE_WR:
            verdict = "PASS"
        elif wf['wf_positive'] >= 1:
            verdict = "WEAK"
        else:
            verdict = "FAIL"
        verdicts[key] = verdict

        flag = " ***" if verdict == "FAIL" else (" *" if verdict == "WEAK" else "")
        print(f"{key:<25} {v['trades']:>6} {v['edge']:>5.1f} {v['mc_p']:>8.4f} {wf_str:>5} {wf['avg_oos_wr']:>6.1f}% {wf['avg_oos_pnl']:>7.1f}% {verdict:>8}{flag}")

    # Summary
    pass_count = len([v for v in verdicts.values() if v == "PASS"])
    weak_count = len([v for v in verdicts.values() if v == "WEAK"])
    fail_count = len([v for v in verdicts.values() if v == "FAIL"])
    nodata_count = len([v for v in verdicts.values() if v == "NO_DATA"])
    print(f"\nWF Summary: PASS={pass_count}, WEAK={weak_count}, FAIL={fail_count}, NO_DATA={nodata_count}")

    # ============================================================
    # Phase 3: Filter Scenarios
    # ============================================================
    print(f"\n[4/5] Filter Scenarios — Portfolio Comparison...")
    print("-" * 70)

    signal_index = build_signal_index(types, n)

    scenarios = {
        'A_CURRENT_66': {
            'desc': 'Current 66 patterns (min_trades=10)',
            'filter': lambda k, v, wf: True,
        },
        'B_MIN_TRADES_20': {
            'desc': 'min_trades >= 20',
            'filter': lambda k, v, wf: v['trades'] >= 20,
        },
        'C_MIN_TRADES_30': {
            'desc': 'min_trades >= 30',
            'filter': lambda k, v, wf: v['trades'] >= 30,
        },
        'D_WF_PASS_ONLY': {
            'desc': 'WF PASS only (2+/3 positive + OOS WR > baseline)',
            'filter': lambda k, v, wf: wf.get(k, {}).get('wf_positive', 0) >= 2 and wf.get(k, {}).get('avg_oos_wr', 0) > BASELINE_WR,
        },
        'E_WF_PASS_OR_WEAK': {
            'desc': 'WF PASS or WEAK (exclude FAIL+NO_DATA)',
            'filter': lambda k, v, wf: verdicts.get(k) in ('PASS', 'WEAK'),
        },
        'F_STRICT': {
            'desc': 'min_trades>=20 AND MC<0.005 AND edge>=8pp',
            'filter': lambda k, v, wf: v['trades'] >= 20 and v['mc_p'] < 0.005 and v['edge'] >= 8.0,
        },
        'G_WF_PASS_TRADES20': {
            'desc': 'WF PASS AND min_trades>=20',
            'filter': lambda k, v, wf: verdicts.get(k) == 'PASS' and v['trades'] >= 20,
        },
    }

    scenario_results = {}
    print(f"\n{'Scenario':<25} {'Desc':<45} {'Pats':>5} {'Trades':>7} {'WR':>6} {'PnL':>9} {'MDD':>7} {'PnL/MDD':>8} {'PF':>5} {'$/trade':>8}")
    print("-" * 135)

    for name, sc in scenarios.items():
        # Filter patterns
        selected_keys = [k for k, v in details.items()
                        if sc['filter'](k, v, wf_results)]

        # Build portfolio
        all_trades = []
        long_pats = []
        short_pats = []
        for key in selected_keys:
            v = details[key]
            pat_name = v['pattern']
            direction = v['direction']
            if pat_name in signal_index:
                sigs = signal_index[pat_name]
                trades = bt_signals(sigs, direction, UNI_TP, UNI_SL, opens, highs, lows, n)
                all_trades.extend(trades)
                if direction == 'LONG':
                    long_pats.append(pat_name)
                else:
                    short_pats.append(pat_name)

        portfolio = portfolio_1pos(all_trades)
        stats = calc_compound_stats(portfolio)

        pnl_mdd = round(stats['pnl'] / stats['mdd'], 1) if stats['mdd'] > 0 else 999

        scenario_results[name] = {
            'patterns': len(selected_keys),
            'long': len(long_pats),
            'short': len(short_pats),
            'stats': stats,
            'pnl_mdd': pnl_mdd,
            'selected_keys': selected_keys,
        }

        print(f"{name:<25} {sc['desc'][:44]:<45} {len(selected_keys):>5} {stats['trades']:>7} {stats['wr']:>5.1f}% {stats['pnl']:>8.1f}% {stats['mdd']:>6.1f}% {pnl_mdd:>7.1f}x {stats['pf']:>5.2f} {stats['pnl_per_trade']:>7.2f}%")

    # ============================================================
    # Phase 4: Recommendation
    # ============================================================
    print(f"\n[5/5] Recommendation...")
    print("-" * 70)

    # Rank by PnL/MDD ratio
    ranked = sorted(scenario_results.items(), key=lambda x: x[1]['pnl_mdd'], reverse=True)

    print("\nRanking by PnL/MDD ratio:")
    for i, (name, res) in enumerate(ranked, 1):
        pats = res['patterns']
        stats = res['stats']
        print(f"  {i}. {name}: PnL/MDD={res['pnl_mdd']}x, PnL={stats['pnl']:.1f}%, MDD={stats['mdd']:.1f}%, {pats} patterns ({res['long']}L+{res['short']}S)")

    # Best scenario
    best_name, best = ranked[0]
    print(f"\n*** BEST: {best_name} ***")
    print(f"  Patterns: {best['patterns']} ({best['long']}L+{best['short']}S)")
    print(f"  Trades: {best['stats']['trades']}, WR: {best['stats']['wr']}%")
    print(f"  PnL: {best['stats']['pnl']:.1f}%, MDD: {best['stats']['mdd']:.1f}%, PnL/MDD: {best['pnl_mdd']}x")

    # Show which patterns are in best but not in strictest, and vice versa
    print(f"\n--- Patterns removed by best filter vs current ---")
    current_keys = set(scenario_results['A_CURRENT_66']['selected_keys'])
    best_keys = set(best['selected_keys'])
    removed = current_keys - best_keys
    if removed:
        print(f"  Removed ({len(removed)}):")
        for k in sorted(removed):
            v = details[k]
            wf = wf_results.get(k, {})
            print(f"    {k}: trades={v['trades']}, edge={v['edge']}, mc_p={v['mc_p']}, WF={wf.get('wf_positive',0)}/{wf.get('wf_folds',0)}")

    # Save results
    output = {
        'analysis_date': '2026-02-14',
        'current_patterns': len(details),
        'baseline_wr': BASELINE_WR,
        'uni_tp': UNI_TP,
        'uni_sl': UNI_SL,
        'wf_results': wf_results,
        'verdicts': verdicts,
        'scenario_summary': {
            name: {
                'patterns': res['patterns'],
                'long': res['long'],
                'short': res['short'],
                'trades': res['stats']['trades'],
                'wr': res['stats']['wr'],
                'pnl': res['stats']['pnl'],
                'mdd': res['stats']['mdd'],
                'pnl_mdd': res['pnl_mdd'],
                'pf': res['stats']['pf'],
                'pnl_per_trade': res['stats']['pnl_per_trade'],
            }
            for name, res in scenario_results.items()
        },
        'best_scenario': best_name,
        'best_patterns': sorted(best['selected_keys']),
        'removed_patterns': sorted(list(removed)),
    }

    out_path = os.path.join(RESULTS_DIR, 'pattern_significance_filter.json')
    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    print(f"\nResults saved: {out_path}")


if __name__ == '__main__':
    main()
