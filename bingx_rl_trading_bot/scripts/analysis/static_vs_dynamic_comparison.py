#!/usr/bin/env python3
"""
Static vs Dynamic Backtest Comparison

Runs IDENTICAL backtest engine on both:
  A) Static: 51 patterns + per-pattern TP/SL (constants.py)
  B) Dynamic: 75 patterns + Universal TP 2.0/SL 3.0 (dynamic_patterns.json)

Uses production classify_candle(), LEVERAGE=3, timeout trades DROPPED.
"""

import os
import sys
import json
import numpy as np
import pandas as pd

# Project root
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
sys.path.insert(0, _PROJECT_ROOT)

from scripts.production.pattern_5m.indicators import classify_candle
from scripts.production.pattern_5m.constants import (
    AVG_BODY_WINDOW,
    VALIDATED_LONG_PATTERNS,
    VALIDATED_SHORT_PATTERNS,
    PATTERN_OPTIMAL_TPSL,
    PATTERN_STATS,
)

# ============================================================
# Constants
# ============================================================
FEE_PCT = 0.10   # 0.05% x 2
LEVERAGE = 3
MAX_BARS = 500

DATA_FILE = os.path.join(_PROJECT_ROOT, 'data', 'btc_5m_270days_reclassified.csv')
DYNAMIC_JSON = os.path.join(_PROJECT_ROOT, 'results', 'dynamic_patterns.json')


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
    """Backtest. Entry: next bar open. Exit: intrabar. Timeout DROPPED."""
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
                bo = opens[j]; pnl = (tp_pct if abs(tpp - bo) <= abs(slp - bo) else -sl_pct) * LEVERAGE - FEE_PCT
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


def calc_stats(trades):
    if not trades:
        return {'pnl': 0, 'trades': 0, 'wr': 0, 'mdd': 0, 'pf': 0, 'avg_pnl': 0, 'max_consec_loss': 0}
    pnls = [t[2] for t in trades]
    cum = 0
    peak = 0
    mdd = 0
    wins = []
    losses = []
    consec_loss = 0
    max_consec = 0
    for p in pnls:
        cum += p
        if cum > peak:
            peak = cum
        dd = peak - cum
        if dd > mdd:
            mdd = dd
        if p > 0:
            wins.append(p)
            consec_loss = 0
        else:
            losses.append(p)
            consec_loss += 1
            if consec_loss > max_consec:
                max_consec = consec_loss
    wsum = sum(wins)
    lsum = sum(abs(x) for x in losses)
    return {
        'pnl': round(cum, 1),
        'trades': len(pnls),
        'wr': round(len(wins) / len(pnls) * 100, 1) if pnls else 0,
        'mdd': round(mdd, 1),
        'pf': round(wsum / lsum, 2) if lsum > 0 else 999,
        'avg_pnl': round(cum / len(pnls), 2) if pnls else 0,
        'max_consec_loss': max_consec,
    }


def run_static(signal_index, opens, highs, lows, n):
    """Static: 51 patterns + per-pattern TP/SL from constants.py"""
    all_trades = []

    for pat in VALIDATED_LONG_PATTERNS:
        if pat not in signal_index:
            continue
        tp, sl = PATTERN_OPTIMAL_TPSL.get(pat, (1.0, 1.0))
        trades = bt_signals(signal_index[pat], 'LONG', tp, sl, opens, highs, lows, n)
        all_trades.extend(trades)

    for pat in VALIDATED_SHORT_PATTERNS:
        if pat not in signal_index:
            continue
        tp, sl = PATTERN_OPTIMAL_TPSL.get(pat, (1.0, 1.0))
        trades = bt_signals(signal_index[pat], 'SHORT', tp, sl, opens, highs, lows, n)
        all_trades.extend(trades)

    return portfolio_1pos(all_trades)


def run_dynamic(signal_index, opens, highs, lows, n, dyn_json):
    """Dynamic: patterns from JSON + Universal TP/SL"""
    uni_tp = dyn_json['universal_tp']
    uni_sl = dyn_json['universal_sl']
    all_trades = []

    for pat in dyn_json['patterns']['long']:
        if pat not in signal_index:
            continue
        trades = bt_signals(signal_index[pat], 'LONG', uni_tp, uni_sl, opens, highs, lows, n)
        all_trades.extend(trades)

    for pat in dyn_json['patterns']['short']:
        if pat not in signal_index:
            continue
        trades = bt_signals(signal_index[pat], 'SHORT', uni_tp, uni_sl, opens, highs, lows, n)
        all_trades.extend(trades)

    return portfolio_1pos(all_trades)


def run_overlap_only(signal_index, opens, highs, lows, n, dyn_json):
    """Overlap only: 22 shared patterns with BOTH TP/SL variants"""
    static_long = set(VALIDATED_LONG_PATTERNS)
    static_short = set(VALIDATED_SHORT_PATTERNS)
    dyn_long = set(dyn_json['patterns']['long'])
    dyn_short = set(dyn_json['patterns']['short'])

    overlap_long = static_long & dyn_long
    overlap_short = static_short & dyn_short

    uni_tp = dyn_json['universal_tp']
    uni_sl = dyn_json['universal_sl']

    # Static TP/SL on overlap patterns
    trades_s = []
    for pat in overlap_long:
        if pat not in signal_index:
            continue
        tp, sl = PATTERN_OPTIMAL_TPSL.get(pat, (1.0, 1.0))
        trades_s.extend(bt_signals(signal_index[pat], 'LONG', tp, sl, opens, highs, lows, n))
    for pat in overlap_short:
        if pat not in signal_index:
            continue
        tp, sl = PATTERN_OPTIMAL_TPSL.get(pat, (1.0, 1.0))
        trades_s.extend(bt_signals(signal_index[pat], 'SHORT', tp, sl, opens, highs, lows, n))

    # Universal TP/SL on overlap patterns
    trades_d = []
    for pat in overlap_long:
        if pat not in signal_index:
            continue
        trades_d.extend(bt_signals(signal_index[pat], 'LONG', uni_tp, uni_sl, opens, highs, lows, n))
    for pat in overlap_short:
        if pat not in signal_index:
            continue
        trades_d.extend(bt_signals(signal_index[pat], 'SHORT', uni_tp, uni_sl, opens, highs, lows, n))

    return overlap_long, overlap_short, portfolio_1pos(trades_s), portfolio_1pos(trades_d)


def main():
    import argparse
    parser = argparse.ArgumentParser(description='Static vs Dynamic comparison')
    parser.add_argument('--data', default=DATA_FILE, help='OHLCV CSV path')
    parser.add_argument('--dynamic', default=DYNAMIC_JSON, help='Dynamic patterns JSON')
    args = parser.parse_args()

    data_file = args.data
    dyn_file = args.dynamic
    label = os.path.basename(data_file).replace('.csv', '')

    print("=" * 70)
    print(f"  STATIC vs DYNAMIC BACKTEST COMPARISON — {label}")
    print("=" * 70)

    # Load data
    print(f"\nLoading and classifying data: {data_file}")
    df = load_and_classify(data_file)
    types = df['candle_type'].tolist()
    n = len(types)
    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values

    signal_index = build_signal_index(types, n)
    print(f"  {n} bars, {len(signal_index)} unique 3-candle patterns")

    # Load dynamic JSON
    with open(dyn_file, 'r') as f:
        dyn_json = json.load(f)

    # Run backtests
    print("\nRunning backtests...")
    static_trades = run_static(signal_index, opens, highs, lows, n)
    dynamic_trades = run_dynamic(signal_index, opens, highs, lows, n, dyn_json)

    s = calc_stats(static_trades)
    d = calc_stats(dynamic_trades)

    # Overlap analysis
    ol, os_set, overlap_static_trades, overlap_dynamic_trades = run_overlap_only(
        signal_index, opens, highs, lows, n, dyn_json
    )
    os_stats = calc_stats(overlap_static_trades)
    od_stats = calc_stats(overlap_dynamic_trades)

    # Pattern counts
    n_static = len(VALIDATED_LONG_PATTERNS) + len(VALIDATED_SHORT_PATTERNS)
    n_dynamic = len(dyn_json['patterns']['long']) + len(dyn_json['patterns']['short'])
    n_overlap = len(ol) + len(os_set)

    # Print comparison
    print("\n" + "=" * 70)
    print("  RESULTS")
    print("=" * 70)

    print(f"\n{'Metric':<22} {'Static (51p)':<16} {'Dynamic (75p)':<16} {'Delta':<12}")
    print("-" * 70)
    print(f"{'Patterns':<22} {n_static:<16} {n_dynamic:<16} {n_dynamic - n_static:+d}")
    print(f"{'TP/SL Mode':<22} {'Per-pattern':<16} {'Universal 2/3':<16}")
    print(f"{'Trades':<22} {s['trades']:<16} {d['trades']:<16} {d['trades'] - s['trades']:+d}")
    print(f"{'Win Rate %':<22} {s['wr']:<16} {d['wr']:<16} {d['wr'] - s['wr']:+.1f}pp")
    print(f"{'PnL %':<22} {s['pnl']:<16} {d['pnl']:<16} {d['pnl'] - s['pnl']:+.1f}")
    print(f"{'MDD %':<22} {s['mdd']:<16} {d['mdd']:<16} {d['mdd'] - s['mdd']:+.1f}")

    s_ratio = round(s['pnl'] / s['mdd'], 1) if s['mdd'] > 0 else 999
    d_ratio = round(d['pnl'] / d['mdd'], 1) if d['mdd'] > 0 else 999
    print(f"{'PnL/MDD':<22} {s_ratio:<16} {d_ratio:<16} {d_ratio - s_ratio:+.1f}")
    print(f"{'Profit Factor':<22} {s['pf']:<16} {d['pf']:<16} {d['pf'] - s['pf']:+.2f}")
    print(f"{'Avg PnL/Trade %':<22} {s['avg_pnl']:<16} {d['avg_pnl']:<16} {d['avg_pnl'] - s['avg_pnl']:+.2f}")
    print(f"{'Max Consec Loss':<22} {s['max_consec_loss']:<16} {d['max_consec_loss']:<16}")

    # Overlap comparison (same patterns, different TP/SL)
    print(f"\n{'='*70}")
    print(f"  OVERLAP ANALYSIS ({n_overlap} shared patterns)")
    print(f"{'='*70}")
    print(f"\n{'Metric':<22} {'Per-pattern TP/SL':<20} {'Universal 2/3':<16} {'Delta':<12}")
    print("-" * 70)
    print(f"{'Trades':<22} {os_stats['trades']:<20} {od_stats['trades']:<16} {od_stats['trades'] - os_stats['trades']:+d}")
    print(f"{'Win Rate %':<22} {os_stats['wr']:<20} {od_stats['wr']:<16} {od_stats['wr'] - os_stats['wr']:+.1f}pp")
    print(f"{'PnL %':<22} {os_stats['pnl']:<20} {od_stats['pnl']:<16} {od_stats['pnl'] - os_stats['pnl']:+.1f}")
    print(f"{'MDD %':<22} {os_stats['mdd']:<20} {od_stats['mdd']:<16} {od_stats['mdd'] - os_stats['mdd']:+.1f}")

    os_ratio = round(os_stats['pnl'] / os_stats['mdd'], 1) if os_stats['mdd'] > 0 else 999
    od_ratio = round(od_stats['pnl'] / od_stats['mdd'], 1) if od_stats['mdd'] > 0 else 999
    print(f"{'PnL/MDD':<22} {os_ratio:<20} {od_ratio:<16} {od_ratio - os_ratio:+.1f}")
    print(f"{'Profit Factor':<22} {os_stats['pf']:<20} {od_stats['pf']:<16} {od_stats['pf'] - os_stats['pf']:+.2f}")

    # Overlap patterns detail
    print(f"\n  Overlap LONG ({len(ol)}): {sorted(ol)}")
    print(f"  Overlap SHORT ({len(os_set)}): {sorted(os_set)}")

    # Static-only and Dynamic-only
    static_only_l = set(VALIDATED_LONG_PATTERNS) - set(dyn_json['patterns']['long'])
    static_only_s = set(VALIDATED_SHORT_PATTERNS) - set(dyn_json['patterns']['short'])
    dyn_only_l = set(dyn_json['patterns']['long']) - set(VALIDATED_LONG_PATTERNS)
    dyn_only_s = set(dyn_json['patterns']['short']) - set(VALIDATED_SHORT_PATTERNS)

    print(f"\n{'='*70}")
    print(f"  PATTERN BREAKDOWN")
    print(f"{'='*70}")
    print(f"\n  Static only ({len(static_only_l) + len(static_only_s)}):")
    print(f"    LONG  ({len(static_only_l)}): {sorted(static_only_l)}")
    print(f"    SHORT ({len(static_only_s)}): {sorted(static_only_s)}")
    print(f"\n  Dynamic only ({len(dyn_only_l) + len(dyn_only_s)}):")
    print(f"    LONG  ({len(dyn_only_l)}): {sorted(dyn_only_l)}")
    print(f"    SHORT ({len(dyn_only_s)}): {sorted(dyn_only_s)}")

    # Summary verdict
    print(f"\n{'='*70}")
    print(f"  VERDICT")
    print(f"{'='*70}")
    if s['pnl'] > d['pnl']:
        pnl_winner = "STATIC"
    else:
        pnl_winner = "DYNAMIC"
    if s_ratio > d_ratio:
        risk_winner = "STATIC"
    else:
        risk_winner = "DYNAMIC"

    print(f"\n  PnL Winner (in-sample 270d): {pnl_winner}")
    print(f"  Risk-Adjusted Winner:        {risk_winner}")
    print(f"\n  NOTE: This is IN-SAMPLE comparison.")
    print(f"  v7 True Walk-Forward showed Universal TP/SL >> per-pattern on OOS.")
    print(f"  Per-pattern TP/SL overfits in-sample (look-ahead bias 7.3%).")
    print(f"  Dynamic system trades MORE patterns = more diversification potential.")


if __name__ == '__main__':
    main()
