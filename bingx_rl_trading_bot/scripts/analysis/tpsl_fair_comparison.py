#!/usr/bin/env python3
"""
TP/SL Fair Comparison — 각 TP/SL별 패턴 재발굴 + 포트폴리오 비교

핵심: 동일 패턴에 TP/SL만 바꾸는 것이 아니라,
각 TP/SL 설정으로 독립적으로 패턴을 스캔하고,
해당 패턴 세트로 포트폴리오를 구성하여 비교.

Phase 1: 각 TP/SL로 패턴 스캔 (edge≥10pp, MC<0.01, trades≥20)
Phase 2: 포트폴리오 비교 (1-at-a-time, compound PnL, MDD)
Phase 3: 홀딩 시간 + 놓친 시그널 분석
Phase 4: 3-fold WF OOS 비교
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

from scripts.scanner.pattern_scanner import (
    load_and_classify, scan_patterns, portfolio_1pos, calc_stats,
    build_signal_index, bt_signals, mc_test
)

DATA_FILE = os.path.join(_PROJECT_ROOT, 'data', 'btc_5m_270days_reclassified.csv')
RESULTS_DIR = os.path.join(_PROJECT_ROOT, 'results')
LEVERAGE = 3
FEE_PCT = 0.10
MAX_BARS = 500

# TP/SL configurations to compare
CONFIGS = [
    (0.5, 0.75),
    (0.7, 1.0),
    (0.8, 1.2),
    (1.0, 1.5),
    (1.2, 1.8),
    (1.4, 2.0),
    (1.6, 2.4),
    (1.8, 2.7),
    (2.0, 3.0),
    (2.5, 3.5),
    (3.0, 4.0),
]

EDGE_THRESHOLD = 10.0
MC_THRESHOLD = 0.01
MIN_TRADES = 20


def compound_pnl_mdd(trades):
    """Compound equity curve from trade list."""
    if not trades:
        return 0.0, 0.0, 0.0
    pnls = [t[2] for t in trades]
    equity = 100.0
    peak = 100.0
    mdd = 0.0
    for p in pnls:
        equity *= (1 + p / 100)
        if equity > peak:
            peak = equity
        dd = (peak - equity) / peak * 100
        if dd > mdd:
            mdd = dd
    return equity - 100, mdd, equity


def holding_stats(trades):
    """Calculate holding time statistics."""
    if not trades:
        return {}
    bars = [xb - eb + 1 for eb, xb, _ in trades]
    hours = [b * 5 / 60 for b in bars]
    return {
        'avg_bars': np.mean(bars),
        'med_bars': np.median(bars),
        'p95_bars': np.percentile(bars, 95),
        'max_bars': max(bars),
        'avg_hours': np.mean(hours),
        'med_hours': np.median(hours),
        'p95_hours': np.percentile(hours, 95),
    }


def count_missed_signals(portfolio_trades, all_signal_bars):
    """Count signals that occur during portfolio holding periods."""
    if not portfolio_trades or not all_signal_bars:
        return 0, 0.0
    all_signal_bars_set = sorted(all_signal_bars)
    total_missed = 0
    for eb, xb, _ in portfolio_trades:
        for sb in all_signal_bars_set:
            if sb > xb:
                break
            if sb >= eb:
                total_missed += 1
    avg_missed = total_missed / len(portfolio_trades) if portfolio_trades else 0
    return total_missed, avg_missed


def wf_3fold(df, tp, sl, edge_threshold, mc_threshold, min_trades):
    """3-fold Walk-Forward: train on 2/3, test on 1/3."""
    n = len(df)
    fold_size = n // 3
    results = []

    for fold in range(3):
        oos_start = fold * fold_size
        oos_end = oos_start + fold_size if fold < 2 else n

        # IS = everything except OOS
        is_df = pd.concat([df.iloc[:oos_start], df.iloc[oos_end:]], ignore_index=True)
        oos_df = df.iloc[oos_start:oos_end].reset_index(drop=True)

        # Scan patterns on IS
        is_result = scan_patterns(is_df, tp, sl, edge_threshold, mc_threshold, min_trades)
        long_pats = set(is_result.get('long_patterns', []))
        short_pats = set(is_result.get('short_patterns', []))

        if not long_pats and not short_pats:
            results.append({
                'fold': fold + 1, 'is_patterns': 0,
                'oos_trades': 0, 'oos_wr': 0, 'oos_edge': 0, 'oos_pnl': 0
            })
            continue

        # Test on OOS
        oos_types = oos_df['candle_type'].tolist()
        oos_opens = oos_df['open'].values
        oos_highs = oos_df['high'].values
        oos_lows = oos_df['low'].values
        oos_n = len(oos_df)
        oos_sig_idx = build_signal_index(oos_types, oos_n)

        be_wr = sl / (tp + sl) * 100
        all_oos_trades = []

        for pat_name in oos_sig_idx:
            for direction, pat_set in [('LONG', long_pats), ('SHORT', short_pats)]:
                if pat_name not in pat_set:
                    continue
                sigs = oos_sig_idx[pat_name]
                trades = bt_signals(sigs, direction, tp, sl, oos_opens, oos_highs, oos_lows, oos_n)
                all_oos_trades.extend(trades)

        port = portfolio_1pos(all_oos_trades)
        if not port:
            results.append({
                'fold': fold + 1,
                'is_patterns': len(long_pats) + len(short_pats),
                'oos_trades': 0, 'oos_wr': 0, 'oos_edge': 0, 'oos_pnl': 0
            })
            continue

        oos_pnls = [t[2] for t in port]
        oos_wr = len([p for p in oos_pnls if p > 0]) / len(oos_pnls) * 100
        oos_edge = oos_wr - be_wr
        oos_pnl_compound, oos_mdd, _ = compound_pnl_mdd(port)

        results.append({
            'fold': fold + 1,
            'is_patterns': len(long_pats) + len(short_pats),
            'oos_trades': len(port),
            'oos_wr': round(oos_wr, 1),
            'oos_edge': round(oos_edge, 1),
            'oos_pnl': round(oos_pnl_compound, 1),
            'oos_mdd': round(oos_mdd, 1),
        })

    return results


def main():
    t0 = time.time()

    print('=' * 80)
    print('TP/SL Fair Comparison — 각 TP/SL별 패턴 독립 발굴 + 포트폴리오 비교')
    print(f'기준: edge≥{EDGE_THRESHOLD}pp, MC<{MC_THRESHOLD}, trades≥{MIN_TRADES}')
    print('=' * 80)

    # Load data once
    df = load_and_classify(DATA_FILE)
    n_bars = len(df)
    total_hours = n_bars * 5 / 60
    total_days = total_hours / 24

    # ── Phase 1 + 2: 각 TP/SL별 스캔 + 포트폴리오 ──
    print(f'\nPhase 1+2: 패턴 발굴 + 포트폴리오 비교 ({len(CONFIGS)} configs)')
    print('-' * 80)

    all_results = []

    for tp, sl in CONFIGS:
        be_wr = sl / (tp + sl) * 100
        win_pnl = tp * LEVERAGE - FEE_PCT
        loss_pnl = sl * LEVERAGE + FEE_PCT

        # Scan patterns for this TP/SL
        scan_result = scan_patterns(df, tp, sl, EDGE_THRESHOLD, MC_THRESHOLD, MIN_TRADES)
        long_pats = set(scan_result.get('long_patterns', []))
        short_pats = set(scan_result.get('short_patterns', []))
        n_patterns = len(long_pats) + len(short_pats)

        if n_patterns == 0:
            all_results.append({
                'tp': tp, 'sl': sl, 'be_wr': be_wr,
                'n_patterns': 0, 'n_long': 0, 'n_short': 0,
                'trades': 0, 'wr': 0, 'edge': 0,
                'pnl_simple': 0, 'pnl_compound': 0, 'mdd': 0,
                'pnl_mdd': 0, 'pf': 0,
                'avg_hold_h': 0, 'med_hold_h': 0, 'p95_hold_h': 0,
                'missed_avg': 0, 'occupancy': 0,
                'trades_per_day': 0, 'win_pnl': win_pnl, 'loss_pnl': loss_pnl,
            })
            continue

        # Build portfolio trades
        types = df['candle_type'].tolist()
        opens = df['open'].values
        highs = df['high'].values
        lows = df['low'].values
        sig_idx = build_signal_index(types, n_bars)

        all_trades = []
        all_signal_entries = []

        for pat_name in sig_idx:
            for direction, pat_set in [('LONG', long_pats), ('SHORT', short_pats)]:
                if pat_name not in pat_set:
                    continue
                sigs = sig_idx[pat_name]
                trades = bt_signals(sigs, direction, tp, sl, opens, highs, lows, n_bars)
                all_trades.extend(trades)
                all_signal_entries.extend([t[0] for t in trades])

        port = portfolio_1pos(all_trades)
        stats = calc_stats(port)
        pnl_compound, mdd_compound, _ = compound_pnl_mdd(port)
        hold = holding_stats(port)

        # Missed signals
        total_missed, avg_missed = count_missed_signals(port, sorted(set(all_signal_entries)))

        # Occupancy
        bars_in_trade = sum(xb - eb + 1 for eb, xb, _ in port)
        occupancy = bars_in_trade / n_bars * 100

        # PnL/MDD
        pnl_mdd = pnl_compound / mdd_compound if mdd_compound > 0 else float('inf')

        r = {
            'tp': tp, 'sl': sl, 'be_wr': round(be_wr, 1),
            'n_patterns': n_patterns,
            'n_long': len(long_pats), 'n_short': len(short_pats),
            'trades': stats['trades'], 'wr': stats['wr'],
            'edge': round(stats['wr'] - be_wr, 1),
            'pnl_simple': stats['pnl'], 'pnl_compound': round(pnl_compound, 1),
            'mdd': round(mdd_compound, 1),
            'pnl_mdd': round(pnl_mdd, 1),
            'pf': stats['pf'],
            'avg_hold_h': round(hold.get('avg_hours', 0), 1),
            'med_hold_h': round(hold.get('med_hours', 0), 1),
            'p95_hold_h': round(hold.get('p95_hours', 0), 1),
            'missed_avg': round(avg_missed, 1),
            'occupancy': round(occupancy, 1),
            'trades_per_day': round(stats['trades'] / total_days, 2),
            'win_pnl': round(win_pnl, 2), 'loss_pnl': round(loss_pnl, 2),
        }
        all_results.append(r)

    # ── Print Phase 1+2 Results ──
    print(f'\n{"TP/SL":>8} {"BE%":>5} {"Pats":>5} {"L/S":>7} {"Trades":>7} '
          f'{"WR%":>6} {"Edge":>6} {"PnL%":>10} {"MDD%":>7} {"PnL/MDD":>8} '
          f'{"PF":>5} {"T/day":>6}')
    print('-' * 95)

    for r in all_results:
        tpsl = f'{r["tp"]}/{r["sl"]}'
        ls = f'{r["n_long"]}/{r["n_short"]}'
        pnl_str = f'{r["pnl_compound"]:.0f}' if abs(r['pnl_compound']) < 100000 else f'{r["pnl_compound"]:.0e}'
        print(f'{tpsl:>8} {r["be_wr"]:>4.0f}% {r["n_patterns"]:>5} {ls:>7} {r["trades"]:>7} '
              f'{r["wr"]:>5.1f}% {r["edge"]:>5.1f} {pnl_str:>10} {r["mdd"]:>6.1f}% {r["pnl_mdd"]:>7.1f}x '
              f'{r["pf"]:>5.2f} {r["trades_per_day"]:>5.2f}')

    # ── Phase 3: 홀딩 시간 비교 ──
    print(f'\n{"TP/SL":>8} {"AvgHold":>9} {"MedHold":>9} {"P95Hold":>9} '
          f'{"Miss/T":>8} {"Occupy":>8} {"Win$":>7} {"Loss$":>7}')
    print('-' * 75)

    for r in all_results:
        if r['trades'] == 0:
            continue
        tpsl = f'{r["tp"]}/{r["sl"]}'
        print(f'{tpsl:>8} {r["avg_hold_h"]:>7.1f}h {r["med_hold_h"]:>7.1f}h {r["p95_hold_h"]:>7.1f}h '
              f'{r["missed_avg"]:>7.1f} {r["occupancy"]:>6.1f}% {r["win_pnl"]:>6.2f}% {r["loss_pnl"]:>6.2f}%')

    # ── Phase 4: WF 3-fold (top configs only) ──
    # Select configs with patterns > 0
    wf_configs = [(r['tp'], r['sl']) for r in all_results if r['n_patterns'] >= 5]

    if wf_configs:
        print(f'\n{"="*80}')
        print(f'Phase 4: 3-fold Walk-Forward OOS (패턴≥5인 {len(wf_configs)} configs)')
        print(f'{"="*80}')

        print(f'\n{"TP/SL":>8} {"F1 Edge":>8} {"F2 Edge":>8} {"F3 Edge":>8} '
              f'{"Pos/3":>6} {"AvgEdge":>8} {"Verdict":>10}')
        print('-' * 65)

        wf_results = []
        for tp, sl in wf_configs:
            folds = wf_3fold(df, tp, sl, EDGE_THRESHOLD, MC_THRESHOLD, MIN_TRADES)
            edges = [f['oos_edge'] for f in folds if f['oos_trades'] > 0]
            positive = sum(1 for e in edges if e > 0)
            avg_edge = np.mean(edges) if edges else 0

            verdict = 'STRONG' if positive == 3 else ('WEAK' if positive == 2 else 'FAIL')
            tpsl = f'{tp}/{sl}'

            f_edges = []
            for f in folds:
                if f['oos_trades'] > 0:
                    f_edges.append(f'{f["oos_edge"]:>+.1f}')
                else:
                    f_edges.append('  N/A')

            while len(f_edges) < 3:
                f_edges.append('  N/A')

            print(f'{tpsl:>8} {f_edges[0]:>8} {f_edges[1]:>8} {f_edges[2]:>8} '
                  f'{positive:>4}/3 {avg_edge:>+7.1f} {verdict:>10}')

            wf_results.append({
                'tp': tp, 'sl': sl,
                'folds': folds,
                'positive': positive,
                'avg_edge': round(avg_edge, 1),
                'verdict': verdict,
            })

    # ── Summary ──
    print(f'\n{"="*80}')
    print('Summary')
    print(f'{"="*80}')

    # Find best by PnL/MDD
    valid = [r for r in all_results if r['trades'] > 0 and r['mdd'] > 0]
    if valid:
        best_pnl_mdd = max(valid, key=lambda x: x['pnl_mdd'])
        best_pnl = max(valid, key=lambda x: x['pnl_compound'])
        most_trades = max(valid, key=lambda x: x['trades'])
        shortest_hold = min(valid, key=lambda x: x['avg_hold_h']) if valid else None

        print(f'  Best PnL/MDD: TP {best_pnl_mdd["tp"]}/SL {best_pnl_mdd["sl"]} '
              f'({best_pnl_mdd["pnl_mdd"]:.1f}x, {best_pnl_mdd["n_patterns"]} patterns)')
        print(f'  Best PnL:     TP {best_pnl["tp"]}/SL {best_pnl["sl"]} '
              f'({best_pnl["pnl_compound"]:.0f}%, {best_pnl["n_patterns"]} patterns)')
        print(f'  Most trades:  TP {most_trades["tp"]}/SL {most_trades["sl"]} '
              f'({most_trades["trades"]} trades, {most_trades["trades_per_day"]:.2f}/day)')
        if shortest_hold:
            print(f'  Shortest hold: TP {shortest_hold["tp"]}/SL {shortest_hold["sl"]} '
                  f'({shortest_hold["avg_hold_h"]:.1f}h avg)')

    # Save results
    output = {
        'date': '2026-02-15',
        'edge_threshold': EDGE_THRESHOLD,
        'mc_threshold': MC_THRESHOLD,
        'min_trades': MIN_TRADES,
        'configs': all_results,
    }
    if wf_configs:
        output['wf_results'] = wf_results

    out_file = os.path.join(RESULTS_DIR, 'tpsl_fair_comparison.json')
    with open(out_file, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f'\n결과 저장: {out_file}')

    elapsed = time.time() - t0
    print(f'소요 시간: {elapsed:.0f}초')


if __name__ == '__main__':
    main()
