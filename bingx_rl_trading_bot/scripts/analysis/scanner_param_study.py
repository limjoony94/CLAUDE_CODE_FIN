#!/usr/bin/env python3
"""
Scanner Parameter Sensitivity Study — PP Discovery

Tests edge_threshold × min_trades combinations with 3-fold expanding WF
to find optimal scanner parameters for OOS performance.

Efficient approach:
1. Grid search ONCE per fold with min_trades=20 (lowest)
2. Post-filter with different edge/min_trades thresholds (cheap)
3. Backtest selected patterns on OOS with IS-optimal TP/SL

WF folds (expanding, 4 quarters → 3 OOS folds):
  Fold 1: IS = Q1,         OOS = Q2
  Fold 2: IS = Q1+Q2,      OOS = Q3
  Fold 3: IS = Q1+Q2+Q3,   OOS = Q4
"""

import os
import sys
import json
import time

import numpy as np
import pandas as pd
from datetime import datetime

# Project setup
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
sys.path.insert(0, _PROJECT_ROOT)

from scripts.scanner.pattern_scanner import (
    load_and_classify, build_signal_index, bt_signals, mc_test,
    grid_search_best, portfolio_1pos, calc_stats,
    MAX_BASELINE_WR,
)

# ============================================================
# Study Parameters
# ============================================================
EDGE_THRESHOLDS = [10, 12, 15, 18, 20, 25]
MIN_TRADES_LIST = [20, 25, 30, 40]
MC_THRESHOLD = 0.01
N_FOLDS = 3

DATA_FILE = os.path.join(_PROJECT_ROOT, 'data', 'btc_5m_270days_reclassified.csv')
OUTPUT_FILE = os.path.join(_PROJECT_ROOT, 'results', 'scanner_param_study.json')


# ============================================================
# Core Functions
# ============================================================

def discover_pp_all(types, opens, highs, lows, n, signal_index, min_tr=20):
    """Run PP grid search on IS data. Cache all candidate results."""
    cache = {}
    for pat_name in signal_index:
        for direction in ['LONG', 'SHORT']:
            sigs = signal_index[pat_name]
            if len(sigs) < min_tr:
                continue

            opt = grid_search_best(sigs, direction, opens, highs, lows, n, min_tr=min_tr)
            if opt is None:
                continue

            tp, sl = opt['tp'], opt['sl']
            trades = bt_signals(sigs, direction, tp, sl, opens, highs, lows, n)
            if len(trades) < min_tr:
                continue

            pnls = [t[2] for t in trades]
            wr = len([p for p in pnls if p > 0]) / len(pnls) * 100
            baseline_wr = sl / (tp + sl) * 100
            edge = wr - baseline_wr

            key = f"{pat_name}_{direction}"
            cache[key] = {
                'pattern': pat_name,
                'direction': direction,
                'tp': tp, 'sl': sl,
                'trades': len(trades),
                'wr': round(wr, 1),
                'edge': round(edge, 1),
                'baseline_wr': round(baseline_wr, 1),
                'pnls': pnls,
                'trades_raw': trades,
            }
    return cache


def filter_and_evaluate(cache, edge_threshold, min_trades):
    """Filter cached results and run MC. Returns selected patterns + IS stats."""
    selected = {}
    for key, info in cache.items():
        if info['trades'] < min_trades:
            continue
        if info['edge'] < edge_threshold:
            continue
        p = mc_test(info['pnls'])
        if p >= MC_THRESHOLD:
            continue
        selected[key] = {
            'pattern': info['pattern'], 'direction': info['direction'],
            'tp': info['tp'], 'sl': info['sl'],
            'trades': info['trades'], 'wr': info['wr'],
            'edge': info['edge'], 'mc_p': round(p, 4),
            'trades_raw': info['trades_raw'],
        }

    # Portfolio stats (IS)
    all_trades = []
    for v in selected.values():
        all_trades.extend(v['trades_raw'])
    portfolio_trades = portfolio_1pos(all_trades)
    is_stats = calc_stats(portfolio_trades)
    return selected, is_stats


def backtest_oos(selected, oos_df):
    """Backtest IS-discovered patterns on OOS data with IS-optimal TP/SL."""
    types = oos_df['candle_type'].tolist()
    n = len(types)
    opens = oos_df['open'].values
    highs = oos_df['high'].values
    lows = oos_df['low'].values

    signal_index = build_signal_index(types, n)

    all_trades = []
    for key, info in selected.items():
        pat_name = info['pattern']
        direction = info['direction']
        tp, sl = info['tp'], info['sl']

        if pat_name not in signal_index:
            continue

        sigs = signal_index[pat_name]
        trades = bt_signals(sigs, direction, tp, sl, opens, highs, lows, n)
        all_trades.extend(trades)

    portfolio_trades = portfolio_1pos(all_trades)
    oos_stats = calc_stats(portfolio_trades)
    return oos_stats


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 70)
    print("Scanner Parameter Sensitivity Study — PP Discovery")
    print(f"Edge thresholds: {EDGE_THRESHOLDS}")
    print(f"Min trades: {MIN_TRADES_LIST}")
    print(f"Total combos: {len(EDGE_THRESHOLDS) * len(MIN_TRADES_LIST)}")
    print("=" * 70)

    df = load_and_classify(DATA_FILE)
    n_total = len(df)
    quarter = n_total // 4

    # Define WF folds
    folds = []
    for f in range(N_FOLDS):
        is_end = quarter * (f + 1)
        oos_end = quarter * (f + 2)
        folds.append((0, is_end, is_end, oos_end))
        print(f"  Fold {f+1}: IS [0:{is_end}] ({is_end} bars), OOS [{is_end}:{oos_end}] ({oos_end - is_end} bars)")

    # Run study
    combo_results = {f"edge{e}_mt{m}": {'edge': e, 'mt': m, 'folds': []}
                     for e in EDGE_THRESHOLDS for m in MIN_TRADES_LIST}

    for fold_i, (is_s, is_e, oos_s, oos_e) in enumerate(folds):
        fold_n = fold_i + 1
        is_df = df.iloc[is_s:is_e].reset_index(drop=True)
        oos_df = df.iloc[oos_s:oos_e].reset_index(drop=True)

        print(f"\n--- Fold {fold_n}: IS {len(is_df)} bars, OOS {len(oos_df)} bars ---")

        # Grid search on IS (one-time, with min_tr=20)
        types_is = is_df['candle_type'].tolist()
        n_is = len(types_is)
        sig_idx = build_signal_index(types_is, n_is)

        t0 = time.time()
        cache = discover_pp_all(
            types_is,
            is_df['open'].values, is_df['high'].values, is_df['low'].values,
            n_is, sig_idx, min_tr=min(MIN_TRADES_LIST),
        )
        elapsed = time.time() - t0
        print(f"  Grid search: {len(cache)} candidates in {elapsed:.1f}s")

        # Test each parameter combo
        for edge_th in EDGE_THRESHOLDS:
            for min_tr in MIN_TRADES_LIST:
                combo_key = f"edge{edge_th}_mt{min_tr}"

                selected, is_stats = filter_and_evaluate(cache, edge_th, min_tr)
                n_pat = len(selected)
                n_long = sum(1 for v in selected.values() if v['direction'] == 'LONG')
                n_short = n_pat - n_long

                if n_pat == 0:
                    combo_results[combo_key]['folds'].append({
                        'fold': fold_n, 'patterns': 0, 'long': 0, 'short': 0,
                        'is': {'pnl': 0, 'trades': 0, 'wr': 0, 'mdd': 0, 'pf': 0},
                        'oos': {'pnl': 0, 'trades': 0, 'wr': 0, 'mdd': 0, 'pf': 0},
                    })
                    continue

                oos_stats = backtest_oos(selected, oos_df)

                combo_results[combo_key]['folds'].append({
                    'fold': fold_n,
                    'patterns': n_pat, 'long': n_long, 'short': n_short,
                    'is': is_stats,
                    'oos': oos_stats,
                })

    # ============================================================
    # Aggregate & Print
    # ============================================================
    print("\n" + "=" * 100)
    print("RESULTS — sorted by OOS PnL/MDD")
    print("=" * 100)

    summary = []
    for combo_key, data in combo_results.items():
        folds_data = data['folds']
        avg_pat = np.mean([f['patterns'] for f in folds_data])

        oos_pnls = [f['oos']['pnl'] for f in folds_data]
        oos_wrs = [f['oos']['wr'] for f in folds_data if f['oos']['trades'] > 0]
        oos_mdds = [f['oos']['mdd'] for f in folds_data if f['oos']['mdd'] > 0]
        oos_trades = [f['oos']['trades'] for f in folds_data]
        is_pnls = [f['is']['pnl'] for f in folds_data]

        n_pos = sum(1 for p in oos_pnls if p > 0)
        avg_oos_pnl = np.mean(oos_pnls)
        avg_oos_wr = np.mean(oos_wrs) if oos_wrs else 0
        avg_oos_mdd = np.mean(oos_mdds) if oos_mdds else 0
        avg_oos_tr = np.mean(oos_trades)
        avg_is_pnl = np.mean(is_pnls)
        pnl_mdd = avg_oos_pnl / avg_oos_mdd if avg_oos_mdd > 0 else 0

        summary.append({
            'combo': combo_key,
            'edge': data['edge'],
            'min_trades': data['mt'],
            'avg_patterns': round(avg_pat, 0),
            'oos_positive': f"{n_pos}/{len(folds_data)}",
            'avg_oos_pnl': round(avg_oos_pnl, 1),
            'total_oos_pnl': round(sum(oos_pnls), 1),
            'avg_oos_wr': round(avg_oos_wr, 1),
            'avg_oos_mdd': round(avg_oos_mdd, 1),
            'avg_oos_trades': round(avg_oos_tr, 0),
            'oos_pnl_mdd': round(pnl_mdd, 2),
            'avg_is_pnl': round(avg_is_pnl, 1),
            'folds': folds_data,
        })

    summary.sort(key=lambda x: x['oos_pnl_mdd'], reverse=True)

    hdr = (f"{'Combo':<18} {'Edge':>4} {'MT':>3} {'Pat':>4} {'OOS+':>5} "
           f"{'OOS PnL':>8} {'OOS WR':>7} {'OOS MDD':>8} {'PnL/MDD':>8} "
           f"{'OOS Tr':>6} {'IS PnL':>8}")
    print(hdr)
    print("-" * len(hdr))
    for s in summary:
        print(f"{s['combo']:<18} {s['edge']:>4} {s['min_trades']:>3} "
              f"{s['avg_patterns']:>4.0f} {s['oos_positive']:>5} "
              f"{s['avg_oos_pnl']:>8.1f} {s['avg_oos_wr']:>6.1f}% "
              f"{s['avg_oos_mdd']:>8.1f} {s['oos_pnl_mdd']:>8.2f} "
              f"{s['avg_oos_trades']:>5.0f} {s['avg_is_pnl']:>8.1f}")

    # Save
    output = {
        'study': 'scanner_parameter_sensitivity',
        'generated_at': datetime.now().isoformat(timespec='seconds'),
        'data_file': os.path.basename(DATA_FILE),
        'data_bars': n_total,
        'parameters': {
            'edge_thresholds': EDGE_THRESHOLDS,
            'min_trades_list': MIN_TRADES_LIST,
            'mc_threshold': MC_THRESHOLD,
            'max_baseline_wr': MAX_BASELINE_WR,
        },
        'wf_folds': N_FOLDS,
        'results': summary,
    }
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved: {OUTPUT_FILE}")

    # Best combo
    best = summary[0]
    print(f"\n{'='*70}")
    print(f"BEST by OOS PnL/MDD: {best['combo']}")
    print(f"  Edge >= {best['edge']}pp | Min trades >= {best['min_trades']}")
    print(f"  Patterns: {best['avg_patterns']:.0f} | OOS+: {best['oos_positive']}")
    print(f"  OOS PnL: {best['avg_oos_pnl']:.1f}% | WR: {best['avg_oos_wr']:.1f}% | "
          f"MDD: {best['avg_oos_mdd']:.1f}% | PnL/MDD: {best['oos_pnl_mdd']:.2f}")


if __name__ == '__main__':
    main()
