#!/usr/bin/env python3
"""
TP/SL Walk-Forward Frontier Study
==================================
For each TP/SL combination, run 3-fold WF to find which ones
actually survive OOS. This answers:
  "Which TP/SL preserves enough edge after WR degradation?"

Key insight from v1.29.0 failure:
  - BE WR = SL/(TP+SL). Smaller TP/SL → higher BE WR → less OOS margin.
  - OOS WR drop is ~12pp regardless of TP/SL.
  - Need: OOS WR - BE WR > 0 for profitability.

Sweep: TP 0.50~3.00%, SL 0.50~3.00%, step 0.10% (676 combos)
Each combo: 3-fold WF with MC validation on train, compound returns on test.

Uses production classify_candle().
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

# ============================================================
# Constants
# ============================================================
LEVERAGE = 3
FEE_PCT = 0.10
MAX_BARS = 500
MC_SIMS = 2000        # Reduced for speed (676 combos × 3 folds)
EDGE_THRESHOLD = 5.0
MC_THRESHOLD = 0.01
MIN_TRADES = 10

DATA_FILE = os.path.join(_PROJECT_ROOT, 'data', 'btc_5m_270days_reclassified.csv')
OUTPUT_FILE = os.path.join(_PROJECT_ROOT, 'results', 'tpsl_wf_frontier.json')

# Sweep range (0.20% step for speed; 196 combos)
TP_RANGE = np.arange(0.60, 3.05, 0.20)
SL_RANGE = np.arange(0.60, 3.05, 0.20)


# ============================================================
# Data Loading
# ============================================================
def load_and_classify(data_file):
    print(f"Loading {os.path.basename(data_file)}...")
    df = pd.read_csv(data_file)
    df['body'] = abs(df['close'] - df['open'])
    df['avg_body'] = df['body'].rolling(AVG_BODY_WINDOW, min_periods=1).mean()
    types = []
    for i, row in df.iterrows():
        ct = classify_candle(row, df.at[i, 'avg_body'])
        types.append(ct.value)
    df['candle_type'] = types
    print(f"  {len(df)} candles classified")
    return df


# ============================================================
# Core Functions
# ============================================================
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


def compound_equity(trades):
    """Returns (final_pnl%, max_dd%, n_trades, wr)."""
    if not trades:
        return 0, 0, 0, 0
    equity = 100.0
    peak = 100.0
    mdd = 0
    wins = 0
    for _, _, pnl_pct in trades:
        equity *= (1 + pnl_pct / 100)
        if equity > peak:
            peak = equity
        dd = (peak - equity) / peak * 100
        if dd > mdd:
            mdd = dd
        if pnl_pct > 0:
            wins += 1
    n = len(trades)
    return equity - 100, mdd, n, wins / n * 100 if n else 0


# ============================================================
# Walk-Forward for a single TP/SL
# ============================================================
def wf_single(signal_index, opens, highs, lows, n_total, tp, sl, n_folds=3):
    """Run 3-fold WF for given TP/SL. Returns summary dict."""
    fold_size = n_total // n_folds
    baseline_wr = sl / (tp + sl) * 100

    oos_pnls = []
    oos_wrs = []
    oos_trades_total = 0
    oos_edges = []

    for fold in range(n_folds):
        test_start = fold * fold_size
        test_end = (fold + 1) * fold_size if fold < n_folds - 1 else n_total

        # --- Train: discover patterns on non-test bars ---
        selected = {}
        for pat_name, all_sigs in signal_index.items():
            for direction in ['LONG', 'SHORT']:
                train_sigs = [s for s in all_sigs if s < test_start or s >= test_end]
                if len(train_sigs) < MIN_TRADES:
                    continue
                trades = bt_signals(train_sigs, direction, tp, sl, opens, highs, lows, n_total)
                if len(trades) < MIN_TRADES:
                    continue
                pnls = [t[2] for t in trades]
                wr = len([p for p in pnls if p > 0]) / len(pnls) * 100
                edge = wr - baseline_wr
                if edge < EDGE_THRESHOLD:
                    continue
                p = mc_test(pnls)
                if p >= MC_THRESHOLD:
                    continue
                key = f"{pat_name}_{direction}"
                selected[key] = {'pattern': pat_name, 'direction': direction}

        # --- Test: apply to held-out fold ---
        test_trades = []
        for key, info in selected.items():
            pat_name = info['pattern']
            direction = info['direction']
            if pat_name not in signal_index:
                continue
            test_sigs = [s for s in signal_index[pat_name] if s >= test_start and s < test_end]
            if not test_sigs:
                continue
            trades = bt_signals(test_sigs, direction, tp, sl, opens, highs, lows, n_total)
            test_trades.extend(trades)

        test_portfolio = portfolio_1pos(test_trades)
        pnl, mdd, n_trades, wr = compound_equity(test_portfolio)
        oos_pnls.append(pnl)
        oos_wrs.append(wr)
        oos_trades_total += n_trades
        oos_edges.append(wr - baseline_wr)

    avg_oos_pnl = np.mean(oos_pnls) if oos_pnls else 0
    avg_oos_wr = np.mean(oos_wrs) if oos_wrs else 0
    avg_oos_edge = np.mean(oos_edges) if oos_edges else 0
    positive_folds = sum(1 for p in oos_pnls if p > 0)

    return {
        'tp': round(tp, 2),
        'sl': round(sl, 2),
        'be_wr': round(baseline_wr, 1),
        'avg_oos_pnl': round(avg_oos_pnl, 1),
        'avg_oos_wr': round(avg_oos_wr, 1),
        'avg_oos_edge': round(avg_oos_edge, 1),
        'positive_folds': positive_folds,
        'n_folds': n_folds,
        'oos_trades': oos_trades_total,
        'fold_pnls': [round(p, 1) for p in oos_pnls],
        'fold_wrs': [round(w, 1) for w in oos_wrs],
    }


# ============================================================
# Main
# ============================================================
def main():
    t0 = time.time()
    print("=" * 60)
    print("TP/SL WALK-FORWARD FRONTIER")
    print(f"TP: {TP_RANGE[0]:.1f}-{TP_RANGE[-1]:.1f}%, SL: {SL_RANGE[0]:.1f}-{SL_RANGE[-1]:.1f}%")
    print(f"Step: 0.10%, Combos: {len(TP_RANGE)}×{len(SL_RANGE)} = {len(TP_RANGE)*len(SL_RANGE)}")
    print("=" * 60)

    df = load_and_classify(DATA_FILE)
    types = df['candle_type'].tolist()
    n = len(types)
    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values

    # Build signal index once (shared across all combos)
    signal_index = build_signal_index(types, n)
    print(f"Signal index: {len(signal_index)} patterns")

    # Run WF for each TP/SL combo
    results = []
    total = len(TP_RANGE) * len(SL_RANGE)
    done = 0

    for tp in TP_RANGE:
        for sl in SL_RANGE:
            r = wf_single(signal_index, opens, highs, lows, n, tp, sl)
            results.append(r)
            done += 1
            if done % 50 == 0 or done == total:
                elapsed = time.time() - t0
                eta = elapsed / done * (total - done)
                print(f"  [{done}/{total}] {elapsed:.0f}s elapsed, ~{eta:.0f}s remaining")

    # ============================================================
    # Analysis
    # ============================================================
    print(f"\n{'='*60}")
    print("ANALYSIS")
    print(f"{'='*60}")

    # Filter: 3/3 positive folds
    perfect = [r for r in results if r['positive_folds'] == 3]
    good = [r for r in results if r['positive_folds'] >= 2]

    print(f"\nTotal combos: {len(results)}")
    print(f"3/3 positive folds: {len(perfect)}")
    print(f">=2/3 positive folds: {len(good)}")

    # Rankings
    # 1. By OOS PnL (3/3 only)
    if perfect:
        by_pnl = sorted(perfect, key=lambda x: x['avg_oos_pnl'], reverse=True)
        print(f"\n--- Top 10 by OOS PnL (3/3 folds positive) ---")
        print(f"{'TP':>5} {'SL':>5} {'BE_WR':>6} {'OOS_WR':>7} {'Edge':>6} {'OOS_PnL':>9} {'Trades':>7}")
        for r in by_pnl[:10]:
            print(f"{r['tp']:5.2f} {r['sl']:5.2f} {r['be_wr']:6.1f}% {r['avg_oos_wr']:6.1f}% "
                  f"{r['avg_oos_edge']:5.1f}pp {r['avg_oos_pnl']:8.1f}% {r['oos_trades']:6d}")

    # 2. By OOS Edge (safety margin)
    if perfect:
        by_edge = sorted(perfect, key=lambda x: x['avg_oos_edge'], reverse=True)
        print(f"\n--- Top 10 by OOS Edge (safety margin, 3/3) ---")
        print(f"{'TP':>5} {'SL':>5} {'BE_WR':>6} {'OOS_WR':>7} {'Edge':>6} {'OOS_PnL':>9} {'Trades':>7}")
        for r in by_edge[:10]:
            print(f"{r['tp']:5.2f} {r['sl']:5.2f} {r['be_wr']:6.1f}% {r['avg_oos_wr']:6.1f}% "
                  f"{r['avg_oos_edge']:5.1f}pp {r['avg_oos_pnl']:8.1f}% {r['oos_trades']:6d}")

    # 3. By OOS Edge × PnL (balanced)
    if perfect:
        for r in perfect:
            r['score'] = r['avg_oos_edge'] * max(r['avg_oos_pnl'], 0) / 100
        by_score = sorted(perfect, key=lambda x: x['score'], reverse=True)
        print(f"\n--- Top 10 by Edge×PnL score (3/3) ---")
        print(f"{'TP':>5} {'SL':>5} {'BE_WR':>6} {'OOS_WR':>7} {'Edge':>6} {'OOS_PnL':>9} {'Score':>7}")
        for r in by_score[:10]:
            print(f"{r['tp']:5.2f} {r['sl']:5.2f} {r['be_wr']:6.1f}% {r['avg_oos_wr']:6.1f}% "
                  f"{r['avg_oos_edge']:5.1f}pp {r['avg_oos_pnl']:8.1f}% {r['score']:6.1f}")

    # 4. Heatmap data: OOS PnL and OOS Edge
    n_tp = len(TP_RANGE)
    n_sl = len(SL_RANGE)
    hm_pnl = np.zeros((n_tp, n_sl))
    hm_edge = np.zeros((n_tp, n_sl))
    hm_folds = np.zeros((n_tp, n_sl))
    hm_wr = np.zeros((n_tp, n_sl))

    for r in results:
        tp_idx = round((r['tp'] - TP_RANGE[0]) / 0.10)
        sl_idx = round((r['sl'] - SL_RANGE[0]) / 0.10)
        if 0 <= tp_idx < n_tp and 0 <= sl_idx < n_sl:
            hm_pnl[tp_idx, sl_idx] = r['avg_oos_pnl']
            hm_edge[tp_idx, sl_idx] = r['avg_oos_edge']
            hm_folds[tp_idx, sl_idx] = r['positive_folds']
            hm_wr[tp_idx, sl_idx] = r['avg_oos_wr']

    # 5. BE WR analysis: find the frontier where edge turns negative
    print(f"\n--- BE WR vs OOS Edge Analysis ---")
    print(f"{'BE_WR':>7} {'Avg OOS Edge':>13} {'3/3 count':>10} {'Best TP/SL':>12}")
    be_wr_buckets = defaultdict(list)
    for r in results:
        bucket = round(r['be_wr'] / 5) * 5
        be_wr_buckets[bucket].append(r)
    for bw in sorted(be_wr_buckets.keys()):
        items = be_wr_buckets[bw]
        avg_edge = np.mean([r['avg_oos_edge'] for r in items])
        n_perfect = sum(1 for r in items if r['positive_folds'] == 3)
        best = max(items, key=lambda x: x['avg_oos_pnl'])
        print(f"{bw:6.0f}% {avg_edge:12.1f}pp {n_perfect:9d} {best['tp']:.1f}/{best['sl']:.1f}")

    # Current config marker
    current = next((r for r in results if abs(r['tp'] - 2.10) < 0.01 and abs(r['sl'] - 3.00) < 0.01), None)
    if current:
        print(f"\n--- Current Config (TP 2.10/SL 3.00) ---")
        print(f"  OOS PnL: {current['avg_oos_pnl']}%, OOS WR: {current['avg_oos_wr']}%, "
              f"Edge: {current['avg_oos_edge']}pp, Folds: {current['positive_folds']}/3")
        print(f"  Fold PnLs: {current['fold_pnls']}")

    # Save output
    elapsed = time.time() - t0
    output = {
        'meta': {
            'data_file': os.path.basename(DATA_FILE),
            'data_bars': len(df),
            'tp_range': [round(x, 2) for x in TP_RANGE],
            'sl_range': [round(x, 2) for x in SL_RANGE],
            'n_combos': len(results),
            'n_folds': 3,
            'mc_sims': MC_SIMS,
            'elapsed_seconds': round(elapsed, 1),
        },
        'summary': {
            'total_combos': len(results),
            'perfect_3_3': len(perfect),
            'good_2_3': len(good),
        },
        'top_by_oos_pnl': by_pnl[:20] if perfect else [],
        'top_by_oos_edge': by_edge[:20] if perfect else [],
        'top_by_score': by_score[:20] if perfect else [],
        'current_210_300': current,
        'heatmap_oos_pnl': hm_pnl.tolist(),
        'heatmap_oos_edge': hm_edge.tolist(),
        'heatmap_positive_folds': hm_folds.tolist(),
        'heatmap_oos_wr': hm_wr.tolist(),
        'all_results': results,
    }

    os.makedirs(os.path.dirname(os.path.abspath(OUTPUT_FILE)), exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"COMPLETE — {elapsed:.0f}s ({elapsed/60:.1f} min)")
    print(f"Output: {OUTPUT_FILE}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
