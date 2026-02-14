#!/usr/bin/env python3
"""
TP/SL Walk-Forward — Targeted Candidates
==========================================
Run 3-fold WF for ~15 specific TP/SL candidates to find optimal config.

Key insight: BE WR = SL/(TP+SL). Need OOS WR > BE WR by enough margin.
OOS WR drops ~12pp from in-sample.

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

LEVERAGE = 3
FEE_PCT = 0.10
MAX_BARS = 500
MC_SIMS = 3000
EDGE_THRESHOLD = 5.0
MC_THRESHOLD = 0.01
MIN_TRADES = 10

DATA_FILE = os.path.join(_PROJECT_ROOT, 'data', 'btc_5m_270days_reclassified.csv')
OUTPUT_FILE = os.path.join(_PROJECT_ROOT, 'results', 'tpsl_wf_targeted.json')

# Targeted TP/SL candidates (covering the full range with strategic picks)
CANDIDATES = [
    # Small TP/SL (high trade count, low MDD, but narrow margin)
    (0.80, 1.20),  # Failed WF (baseline)
    (0.80, 1.50),  # Same TP, wider SL → lower BE WR
    (0.80, 2.00),  # Same TP, much wider SL
    (1.00, 1.50),  # Slightly larger
    (1.00, 2.00),  # TP 1.0, generous SL
    # Medium TP/SL
    (1.20, 1.80),  # Balanced medium
    (1.20, 2.40),  # Medium TP, wide SL
    (1.50, 2.00),  # Medium-high
    (1.50, 2.50),  # Medium TP, wide SL
    (1.50, 3.00),  # Medium TP, widest SL
    # Large TP/SL (fewer trades, higher per-trade edge)
    (2.00, 3.00),  # Near current
    (2.10, 3.00),  # Current config
    (2.50, 3.00),  # Wider TP
    (2.00, 2.00),  # Equal TP/SL
    (3.00, 3.00),  # Maximum equal
]


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


def wf_single(signal_index, opens, highs, lows, n_total, tp, sl, n_folds=3):
    fold_size = n_total // n_folds
    baseline_wr = sl / (tp + sl) * 100

    oos_pnls = []
    oos_wrs = []
    oos_trades_total = 0
    oos_mdds = []
    fold_details = []

    for fold in range(n_folds):
        test_start = fold * fold_size
        test_end = (fold + 1) * fold_size if fold < n_folds - 1 else n_total

        # Train: discover patterns on non-test bars
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

        # Test: apply on held-out fold
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
        oos_mdds.append(mdd)
        oos_trades_total += n_trades

        # Train stats for reference
        train_trades = []
        for key, info in selected.items():
            pat_name = info['pattern']
            direction = info['direction']
            if pat_name not in signal_index:
                continue
            train_sigs = [s for s in signal_index[pat_name] if s < test_start or s >= test_end]
            t = bt_signals(train_sigs, direction, tp, sl, opens, highs, lows, n_total)
            train_trades.extend(t)
        train_portfolio = portfolio_1pos(train_trades)
        train_pnl, train_mdd, train_n, train_wr = compound_equity(train_portfolio)

        fold_details.append({
            'fold': fold + 1,
            'patterns': len(selected),
            'train_trades': train_n,
            'train_wr': round(train_wr, 1),
            'train_pnl': round(train_pnl, 1),
            'test_trades': n_trades,
            'test_wr': round(wr, 1),
            'test_pnl': round(pnl, 1),
            'test_mdd': round(mdd, 1),
            'wr_drop': round(train_wr - wr, 1),
        })

    avg_oos_pnl = np.mean(oos_pnls)
    avg_oos_wr = np.mean(oos_wrs)
    avg_oos_mdd = np.mean(oos_mdds)
    positive_folds = sum(1 for p in oos_pnls if p > 0)

    # Expected per-trade values
    avg_win = tp * LEVERAGE - FEE_PCT
    avg_loss = sl * LEVERAGE + FEE_PCT
    oos_expected = avg_oos_wr / 100 * avg_win - (1 - avg_oos_wr / 100) * avg_loss
    single_sl_loss = sl * LEVERAGE + FEE_PCT  # Max single-trade loss %

    return {
        'tp': round(tp, 2),
        'sl': round(sl, 2),
        'be_wr': round(baseline_wr, 1),
        'avg_oos_pnl': round(avg_oos_pnl, 1),
        'avg_oos_wr': round(avg_oos_wr, 1),
        'avg_oos_edge': round(avg_oos_wr - baseline_wr, 1),
        'avg_oos_mdd': round(avg_oos_mdd, 1),
        'positive_folds': positive_folds,
        'oos_trades': oos_trades_total,
        'oos_expected_per_trade': round(oos_expected, 3),
        'single_sl_loss': round(single_sl_loss, 2),
        'avg_win': round(avg_win, 2),
        'avg_loss': round(avg_loss, 2),
        'folds': fold_details,
    }


def main():
    t0 = time.time()
    print("=" * 60)
    print("TP/SL WALK-FORWARD — TARGETED CANDIDATES")
    print(f"Candidates: {len(CANDIDATES)}")
    print("=" * 60)

    df = load_and_classify(DATA_FILE)
    types = df['candle_type'].tolist()
    n = len(types)
    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    signal_index = build_signal_index(types, n)
    print(f"Signal index: {len(signal_index)} patterns\n")

    results = []
    for i, (tp, sl) in enumerate(CANDIDATES):
        print(f"[{i+1}/{len(CANDIDATES)}] TP {tp:.2f} / SL {sl:.2f}  (BE WR: {sl/(tp+sl)*100:.1f}%)")
        r = wf_single(signal_index, opens, highs, lows, n, tp, sl)
        results.append(r)

        # Quick summary
        status = "PASS" if r['positive_folds'] == 3 else f"FAIL ({r['positive_folds']}/3)"
        print(f"  → {status} | OOS WR {r['avg_oos_wr']:.1f}% | Edge {r['avg_oos_edge']:.1f}pp | "
              f"PnL {r['avg_oos_pnl']:.1f}% | MDD {r['avg_oos_mdd']:.1f}% | "
              f"E[trade] {r['oos_expected_per_trade']:.3f}%")
        for fd in r['folds']:
            print(f"    Fold{fd['fold']}: {fd['patterns']}pat, "
                  f"Train WR {fd['train_wr']}%→Test WR {fd['test_wr']}% (drop {fd['wr_drop']}pp), "
                  f"PnL {fd['test_pnl']:.1f}%")
        print()

    # ============================================================
    # Summary Table
    # ============================================================
    print("\n" + "=" * 100)
    print("SUMMARY TABLE")
    print("=" * 100)
    print(f"{'TP':>5} {'SL':>5} {'BE_WR':>6} {'OOS_WR':>7} {'Edge':>6} {'Folds':>6} "
          f"{'OOS_PnL':>9} {'OOS_MDD':>8} {'E[trade]':>9} {'SL_Loss':>8} {'Verdict':>10}")
    print("-" * 100)

    for r in results:
        verdict = "PASS" if r['positive_folds'] == 3 else "FAIL"
        if r['positive_folds'] == 3 and r['avg_oos_edge'] >= 5:
            verdict = "STRONG"
        elif r['positive_folds'] == 3:
            verdict = "PASS"
        elif r['positive_folds'] == 2:
            verdict = "WEAK"
        else:
            verdict = "FAIL"

        marker = " <<<" if abs(r['tp'] - 2.10) < 0.01 and abs(r['sl'] - 3.00) < 0.01 else ""
        print(f"{r['tp']:5.2f} {r['sl']:5.2f} {r['be_wr']:5.1f}% {r['avg_oos_wr']:6.1f}% "
              f"{r['avg_oos_edge']:5.1f}pp {r['positive_folds']:3d}/3 "
              f"{r['avg_oos_pnl']:8.1f}% {r['avg_oos_mdd']:7.1f}% "
              f"{r['oos_expected_per_trade']:8.3f}% {r['single_sl_loss']:7.2f}% "
              f"{verdict:>8}{marker}")

    # Best by different criteria
    pass_results = [r for r in results if r['positive_folds'] == 3]
    if pass_results:
        best_pnl = max(pass_results, key=lambda x: x['avg_oos_pnl'])
        best_edge = max(pass_results, key=lambda x: x['avg_oos_edge'])
        best_mdd = min(pass_results, key=lambda x: x['avg_oos_mdd'])
        best_expected = max(pass_results, key=lambda x: x['oos_expected_per_trade'])

        print(f"\n--- Best by Criteria (3/3 PASS only) ---")
        print(f"  Best OOS PnL:      TP {best_pnl['tp']}/{best_pnl['sl']} → {best_pnl['avg_oos_pnl']:.1f}%")
        print(f"  Best OOS Edge:     TP {best_edge['tp']}/{best_edge['sl']} → {best_edge['avg_oos_edge']:.1f}pp")
        print(f"  Lowest OOS MDD:    TP {best_mdd['tp']}/{best_mdd['sl']} → {best_mdd['avg_oos_mdd']:.1f}%")
        print(f"  Best E[per trade]: TP {best_expected['tp']}/{best_expected['sl']} → {best_expected['oos_expected_per_trade']:.3f}%")

    # Save
    elapsed = time.time() - t0
    output = {
        'meta': {
            'data_file': os.path.basename(DATA_FILE),
            'data_bars': n,
            'candidates': len(CANDIDATES),
            'elapsed_seconds': round(elapsed, 1),
        },
        'results': results,
        'pass_count': len(pass_results) if pass_results else 0,
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
