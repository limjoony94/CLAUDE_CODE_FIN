#!/usr/bin/env python3
"""
Extended Rediscovery v3: Cross-Exchange Diagnostic
===================================================
v1: 720일 Binance에서 새 패턴 발굴 → 모든 OOS 음수
v2: TP/SL 3가지 전략 → 모든 OOS 음수, MFE≈MAE

핵심 질문: BingX에서 작동하는 51패턴이 같은 기간 Binance에서도 작동하는가?

Phase 1: Data Alignment & Classification Comparison
Phase 2: BingX Patterns on Both Exchanges (same overlap period)
Phase 3: BingX Patterns on Full 720d Binance (time-period breakdown)
Phase 4: Signal-Level Divergence Analysis
Phase 5: MFE/MAE on Matched Signals
Phase 6: Control — Full BingX 270d

Diagnosis:
- BX+ BN- same period → EXCHANGE ARTIFACT
- BX+ BN+ same period, BN- full → TIME-LIMITED EDGE
- BX+ BN+ full → REAL EDGE

Optimized: numpy vectorization, flush=True
"""

import sys
import json
import time
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from collections import defaultdict

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RESULTS_DIR = PROJECT_ROOT / "results"

# Import production classify_candle (same as reference research scripts)
sys.path.insert(0, str(SCRIPT_DIR.parent / 'production'))
from pattern_5m.indicators import classify_candle

FEE_PCT = 0.10
LEVERAGE = 3
MAX_BARS = 500

# v1.27.3 — 51 patterns (32L + 19S, U-H-BU removed v1.27.2)
PATTERNS = {
    # LONG (32)
    "BD-BD-U":   ("LONG", 1.4, 3.0),
    "BD-MU-BD":  ("LONG", 0.7, 0.7),
    "BD-ST-U":   ("LONG", 1.4, 2.0),
    "BU-BU-BD":  ("LONG", 2.1, 3.0),
    "D-MU-U":    ("LONG", 1.05, 2.0),
    "DN-BD-BD":  ("LONG", 1.4, 1.5),
    "DN-DF-MU":  ("LONG", 1.05, 3.0),
    "DN-DF-ST":  ("LONG", 1.4, 3.0),
    "DN-DN-H":   ("LONG", 0.7, 2.5),
    "DN-MD-DN":  ("LONG", 1.5, 3.0),
    "GS-ST-ST":  ("LONG", 1.05, 2.0),
    "H-BU-BU":   ("LONG", 1.4, 3.0),
    "H-MU-MD":   ("LONG", 0.7, 2.0),
    "IH-MD-MD":  ("LONG", 0.7, 2.0),
    "IH-ST-MU":  ("LONG", 0.35, 2.5),
    "MD-BU-MD":  ("LONG", 1.4, 2.5),
    "MD-DN-MU":  ("LONG", 1.0, 1.0),
    "MD-H-MD":   ("LONG", 0.7, 0.7),
    "MD-MD-ST":  ("LONG", 1.05, 2.0),
    "MD-ST-BD":  ("LONG", 1.0, 0.5),
    "MD-ST-MD":  ("LONG", 2.0, 2.0),
    "MU-BD-ST":  ("LONG", 1.4, 3.0),
    "MU-DF-U":   ("LONG", 1.05, 2.0),
    "MU-H-MU":   ("LONG", 1.5, 0.7),
    "MU-IH-DN":  ("LONG", 1.4, 2.0),
    "MU-U-H":    ("LONG", 1.75, 3.0),
    "U-H-MU":    ("LONG", 1.5, 2.0),
    "U-MD-GS":   ("LONG", 0.35, 2.5),
    "U-MD-MD":   ("LONG", 1.5, 0.7),
    "U-MU-H":    ("LONG", 1.05, 3.0),
    "U-MU-IH":   ("LONG", 2.1, 2.5),
    "U-ST-DF":   ("LONG", 1.4, 3.0),
    # SHORT (19)
    "BD-BU-DN":  ("SHORT", 3.0, 3.0),
    "BD-D-D":    ("SHORT", 1.05, 3.0),
    "BD-U-H":    ("SHORT", 2.5, 3.0),
    "BU-MD-MD":  ("SHORT", 3.0, 2.0),
    "BU-ST-GS":  ("SHORT", 0.7, 2.5),
    "D-BD-ST":   ("SHORT", 1.75, 3.0),
    "D-DN-DN":   ("SHORT", 2.5, 3.0),
    "DN-BD-BU":  ("SHORT", 1.75, 3.0),
    "DN-D-BD":   ("SHORT", 1.75, 1.0),
    "DN-DF-DN":  ("SHORT", 2.0, 2.0),
    "H-U-BD":    ("SHORT", 2.1, 2.0),
    "IH-ST-ST":  ("SHORT", 1.4, 3.0),
    "MD-MD-MD":  ("SHORT", 3.0, 3.0),
    "ST-BD-BU":  ("SHORT", 2.1, 3.0),
    "ST-DN-BU":  ("SHORT", 1.4, 3.0),
    "ST-DN-U":   ("SHORT", 2.1, 3.0),
    "ST-MU-ST":  ("SHORT", 1.4, 3.0),
    "U-GS-DN":   ("SHORT", 3.0, 3.0),
    "U-ST-DN":   ("SHORT", 1.4, 3.0),
}


# ── Candle classification (uses PRODUCTION classify_candle — identical to reference scripts)
def classify_candles(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    n = len(df)

    body_abs_arr = np.abs(closes - opens)
    avg_body_20_arr = pd.Series(body_abs_arr).rolling(20).mean().values  # min_periods=20 (default)

    types = []
    for i in range(n):
        row = pd.Series({'open': opens[i], 'high': highs[i], 'low': lows[i], 'close': closes[i]})
        avg_b = avg_body_20_arr[i] if not pd.isna(avg_body_20_arr[i]) else 1.0
        types.append(classify_candle(row, avg_b).value)

    df['type_code'] = types
    df['pattern_3'] = [None, None] + [
        f"{types[i-2]}-{types[i-1]}-{types[i]}" for i in range(2, n)
    ]
    return df


# ── Portfolio backtest (1-position-at-a-time, REFERENCE protocol) ──
# Matches collect_trades_1pos() from tp_sl_reoptimization_v1270.py exactly:
# - Collect ALL possible trades first, then filter non-overlapping
# - Timeout trades are DROPPED (not counted) — same as reference
# - Bar range: idx+2 to idx+2+MAX_BARS (same as reference)
def backtest_portfolio(df, patterns_dict, max_bars=500):
    highs = df['high'].values
    lows = df['low'].values
    opens = df['open'].values
    closes = df['close'].values
    types = df['type_code'].values
    n = len(df)

    # Step 1: Collect ALL possible trades (including overlapping)
    raw_trades = []
    for i in range(2, n):
        pat = f"{types[i-2]}-{types[i-1]}-{types[i]}"
        if pat not in patterns_dict:
            continue
        if i + 1 >= n:
            continue
        entry = opens[i + 1]
        if entry <= 0:
            continue
        direction, tp_pct, sl_pct = patterns_dict[pat]
        entry_bar = i + 1

        if direction == 'LONG':
            tpp = entry * (1 + tp_pct / 100)
            slp = entry * (1 - sl_pct / 100)
        else:
            tpp = entry * (1 - tp_pct / 100)
            slp = entry * (1 + sl_pct / 100)

        for j in range(i + 2, min(i + 2 + max_bars, n)):
            ht = (highs[j] >= tpp) if direction == 'LONG' else (lows[j] <= tpp)
            hs = (lows[j] <= slp) if direction == 'LONG' else (highs[j] >= slp)
            if ht and hs:
                pnl = (tp_pct if abs(tpp - entry) <= abs(slp - entry) else -sl_pct) * LEVERAGE - FEE_PCT
                exit_type = 'tp' if abs(tpp - entry) <= abs(slp - entry) else 'sl'
                raw_trades.append((entry_bar, j, pnl, pat, direction, exit_type))
                break
            elif ht:
                raw_trades.append((entry_bar, j, tp_pct * LEVERAGE - FEE_PCT, pat, direction, 'tp'))
                break
            elif hs:
                raw_trades.append((entry_bar, j, -sl_pct * LEVERAGE - FEE_PCT, pat, direction, 'sl'))
                break
        # If no TP/SL hit within max_bars → trade is DROPPED (same as reference)

    # Step 2: Sort by entry bar, filter non-overlapping (1-position-at-a-time)
    raw_trades.sort(key=lambda x: x[0])
    filtered = []
    last_exit = -1
    for eb, xb, pnl, pat, direction, exit_type in raw_trades:
        if eb > last_exit:
            filtered.append({'pattern': pat, 'direction': direction,
                             'pnl': pnl, 'exit': exit_type, 'bar': xb,
                             'entry_bar': eb, 'exit_bar': xb})
            last_exit = xb

    return filtered


def calc_metrics(trades):
    if not trades:
        return {'pnl': 0, 'trades': 0, 'wr': 0, 'mdd': 0, 'pf': 0,
                'max_consec_loss': 0, 'avg_win': 0, 'avg_loss': 0}
    pnls = [t['pnl'] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    cum = np.cumsum(pnls)
    peak = np.maximum.accumulate(cum)
    dd = peak - cum
    mdd = float(np.max(dd)) if len(dd) > 0 else 0

    max_cl = cl = 0
    for p in pnls:
        if p <= 0:
            cl += 1
            max_cl = max(max_cl, cl)
        else:
            cl = 0

    gross_win = sum(wins) if wins else 0
    gross_loss = abs(sum(losses)) if losses else 0.001

    return {
        'pnl': round(float(np.sum(pnls)), 2),
        'trades': len(pnls),
        'wr': round(len(wins) / len(pnls) * 100, 1) if pnls else 0,
        'mdd': round(mdd, 2),
        'pf': round(gross_win / gross_loss, 2) if gross_loss > 0 else 999,
        'max_consec_loss': max_cl,
        'avg_win': round(float(np.mean(wins)), 3) if wins else 0,
        'avg_loss': round(float(np.mean(losses)), 3) if losses else 0,
    }


def main():
    t0 = time.time()
    results = {
        'meta': {
            'script': 'extended_rediscovery_v3_cross_exchange.py',
            'timestamp': datetime.now().isoformat(),
            'patterns': len(PATTERNS),
            'max_bars': MAX_BARS,
            'fee_pct': FEE_PCT,
        }
    }

    # ── Load data ──
    print("=" * 60, flush=True)
    print("Phase 0: Loading data", flush=True)
    print("=" * 60, flush=True)

    bingx_path = DATA_DIR / "btc_5m_270days_reclassified.csv"
    binance_path = DATA_DIR / "btc_5m_720days_binance.csv"

    df_bx = pd.read_csv(bingx_path, parse_dates=['timestamp'])
    df_bn = pd.read_csv(binance_path, parse_dates=['timestamp'])

    print(f"BingX:   {len(df_bx):>7} bars, {df_bx['timestamp'].min()} ~ {df_bx['timestamp'].max()}", flush=True)
    print(f"Binance: {len(df_bn):>7} bars, {df_bn['timestamp'].min()} ~ {df_bn['timestamp'].max()}", flush=True)

    results['meta']['bingx_bars'] = len(df_bx)
    results['meta']['binance_bars'] = len(df_bn)
    results['meta']['bingx_range'] = f"{df_bx['timestamp'].min()} ~ {df_bx['timestamp'].max()}"
    results['meta']['binance_range'] = f"{df_bn['timestamp'].min()} ~ {df_bn['timestamp'].max()}"

    # Reclassify both with identical algorithm
    print("\nReclassifying both datasets with identical algorithm...", flush=True)
    df_bx = classify_candles(df_bx)
    df_bn = classify_candles(df_bn)

    # ── Phase 1: Overlap & Classification Comparison ──
    print("\n" + "=" * 60, flush=True)
    print("Phase 1: Data Alignment & Classification Comparison", flush=True)
    print("=" * 60, flush=True)

    bx_start = df_bx['timestamp'].min()
    bx_end = df_bx['timestamp'].max()
    bn_start = df_bn['timestamp'].min()
    bn_end = df_bn['timestamp'].max()

    overlap_start = max(bx_start, bn_start)
    overlap_end = min(bx_end, bn_end)
    overlap_days = (overlap_end - overlap_start).days

    print(f"Overlap period: {overlap_start} ~ {overlap_end}", flush=True)
    print(f"Overlap days: {overlap_days}", flush=True)

    bx_ov = df_bx[(df_bx['timestamp'] >= overlap_start) & (df_bx['timestamp'] <= overlap_end)].reset_index(drop=True)
    bn_ov = df_bn[(df_bn['timestamp'] >= overlap_start) & (df_bn['timestamp'] <= overlap_end)].reset_index(drop=True)

    print(f"BingX overlap bars:  {len(bx_ov)}", flush=True)
    print(f"Binance overlap bars: {len(bn_ov)}", flush=True)

    # Merge on timestamp for bar-by-bar comparison
    merged = pd.merge(
        bx_ov[['timestamp', 'open', 'high', 'low', 'close', 'type_code', 'pattern_3']],
        bn_ov[['timestamp', 'open', 'high', 'low', 'close', 'type_code', 'pattern_3']],
        on='timestamp', suffixes=('_bx', '_bn')
    )
    print(f"Matched timestamps: {len(merged)} / {len(bx_ov)}", flush=True)

    # Classification comparison
    type_match = (merged['type_code_bx'] == merged['type_code_bn']).mean() * 100
    pat_match_rate = merged.apply(
        lambda r: r['pattern_3_bx'] == r['pattern_3_bn']
        if pd.notna(r['pattern_3_bx']) and pd.notna(r['pattern_3_bn']) else False,
        axis=1
    ).mean() * 100

    print(f"\nCandle type match rate:    {type_match:.1f}%", flush=True)
    print(f"3-candle pattern match rate: {pat_match_rate:.1f}%", flush=True)

    # Price divergence
    price_diff = np.abs(merged['close_bx'] - merged['close_bn']) / merged['close_bx'] * 100
    print(f"\nPrice divergence (close): median {price_diff.median():.4f}%, "
          f"P95 {price_diff.quantile(0.95):.4f}%, max {price_diff.max():.4f}%", flush=True)

    # OHLC divergence
    for col in ['open', 'high', 'low', 'close']:
        diff = np.abs(merged[f'{col}_bx'] - merged[f'{col}_bn']) / merged[f'{col}_bx'] * 100
        print(f"  {col:>5} diff: median {diff.median():.4f}%, P95 {diff.quantile(0.95):.4f}%", flush=True)

    # Per-type mismatch breakdown
    type_confusion = defaultdict(lambda: defaultdict(int))
    for _, row in merged.iterrows():
        type_confusion[row['type_code_bx']][row['type_code_bn']] += 1

    mismatches = []
    for bx_type, bn_types in sorted(type_confusion.items()):
        total = sum(bn_types.values())
        for bn_type, count in sorted(bn_types.items(), key=lambda x: -x[1]):
            if bx_type != bn_type and count / total > 0.03:
                mismatches.append((bx_type, bn_type, count, count / total * 100))

    if mismatches:
        print(f"\nMajor classification mismatches (>3% of type):", flush=True)
        for bx_t, bn_t, cnt, pct in sorted(mismatches, key=lambda x: -x[2])[:20]:
            print(f"  BingX {bx_t:>3} → Binance {bn_t:>3}: {cnt:>5} bars ({pct:.1f}%)", flush=True)

    results['phase1'] = {
        'overlap_period': f"{overlap_start} ~ {overlap_end}",
        'overlap_days': overlap_days,
        'bx_overlap_bars': len(bx_ov),
        'bn_overlap_bars': len(bn_ov),
        'matched_timestamps': len(merged),
        'type_match_rate': round(type_match, 2),
        'pattern_match_rate': round(pat_match_rate, 2),
        'price_divergence': {
            'median_pct': round(float(price_diff.median()), 4),
            'p95_pct': round(float(price_diff.quantile(0.95)), 4),
            'max_pct': round(float(price_diff.max()), 4),
        },
        'major_mismatches': [
            {'bx': m[0], 'bn': m[1], 'count': m[2], 'pct': round(m[3], 1)}
            for m in sorted(mismatches, key=lambda x: -x[2])[:20]
        ],
    }

    # ── Phase 2: Same-Period Comparison ──
    print("\n" + "=" * 60, flush=True)
    print("Phase 2: BingX Patterns on Both Exchanges (Same Period)", flush=True)
    print("=" * 60, flush=True)

    trades_bx = backtest_portfolio(bx_ov, PATTERNS, MAX_BARS)
    metrics_bx = calc_metrics(trades_bx)
    print(f"\n[BingX overlap]  {metrics_bx['trades']:>4} trades, "
          f"WR {metrics_bx['wr']:>5.1f}%, PnL {metrics_bx['pnl']:>+8.1f}%, "
          f"MDD {metrics_bx['mdd']:>6.1f}%, PF {metrics_bx['pf']:.2f}", flush=True)

    trades_bn = backtest_portfolio(bn_ov, PATTERNS, MAX_BARS)
    metrics_bn = calc_metrics(trades_bn)
    print(f"[Binance overlap] {metrics_bn['trades']:>4} trades, "
          f"WR {metrics_bn['wr']:>5.1f}%, PnL {metrics_bn['pnl']:>+8.1f}%, "
          f"MDD {metrics_bn['mdd']:>6.1f}%, PF {metrics_bn['pf']:.2f}", flush=True)

    # Per-pattern comparison
    bx_pat_pnl = defaultdict(list)
    bn_pat_pnl = defaultdict(list)
    for t in trades_bx:
        bx_pat_pnl[t['pattern']].append(t['pnl'])
    for t in trades_bn:
        bn_pat_pnl[t['pattern']].append(t['pnl'])

    all_pats = sorted(set(list(bx_pat_pnl.keys()) + list(bn_pat_pnl.keys())))
    print(f"\nPer-pattern comparison ({len(all_pats)} patterns traded):", flush=True)
    print(f"{'Pattern':<15} {'BX_Tr':>5} {'BX_WR':>6} {'BX_PnL':>8} "
          f"{'BN_Tr':>5} {'BN_WR':>6} {'BN_PnL':>8} {'Gap':>7}", flush=True)
    print("-" * 70, flush=True)

    pat_comparison = []
    for pat in all_pats:
        bp = bx_pat_pnl.get(pat, [])
        np_ = bn_pat_pnl.get(pat, [])
        bx_s = sum(bp)
        bn_s = sum(np_)
        bx_wr = sum(1 for p in bp if p > 0) / len(bp) * 100 if bp else 0
        bn_wr = sum(1 for p in np_ if p > 0) / len(np_) * 100 if np_ else 0
        gap = bn_s - bx_s

        print(f"{pat:<15} {len(bp):>5} {bx_wr:>5.1f}% {bx_s:>+7.1f}% "
              f"{len(np_):>5} {bn_wr:>5.1f}% {bn_s:>+7.1f}% {gap:>+6.1f}%", flush=True)
        pat_comparison.append({
            'pattern': pat, 'direction': PATTERNS.get(pat, ('?',))[0],
            'bx_trades': len(bp), 'bx_wr': round(bx_wr, 1), 'bx_pnl': round(bx_s, 2),
            'bn_trades': len(np_), 'bn_wr': round(bn_wr, 1), 'bn_pnl': round(bn_s, 2),
            'pnl_gap': round(gap, 2),
        })

    bx_better = sum(1 for p in pat_comparison if p['bx_pnl'] > p['bn_pnl'])
    bn_better = sum(1 for p in pat_comparison if p['bn_pnl'] > p['bx_pnl'])
    print(f"\nBingX better: {bx_better}, Binance better: {bn_better}", flush=True)

    results['phase2'] = {
        'bingx_metrics': metrics_bx,
        'binance_metrics': metrics_bn,
        'pnl_gap': round(metrics_bn['pnl'] - metrics_bx['pnl'], 2),
        'wr_gap': round(metrics_bn['wr'] - metrics_bx['wr'], 1),
        'bx_better_count': bx_better,
        'bn_better_count': bn_better,
        'per_pattern': pat_comparison,
    }

    # ── Phase 3: Full Binance 720d + Time Breakdown ──
    print("\n" + "=" * 60, flush=True)
    print("Phase 3: BingX Patterns on Full 720d Binance", flush=True)
    print("=" * 60, flush=True)

    trades_full = backtest_portfolio(df_bn, PATTERNS, MAX_BARS)
    metrics_full = calc_metrics(trades_full)
    print(f"[Binance 720d FULL] {metrics_full['trades']:>4} trades, "
          f"WR {metrics_full['wr']:>5.1f}%, PnL {metrics_full['pnl']:>+8.1f}%, "
          f"MDD {metrics_full['mdd']:>6.1f}%, PF {metrics_full['pf']:.2f}", flush=True)

    # Time period breakdown
    df_bn_before = df_bn[df_bn['timestamp'] < overlap_start].reset_index(drop=True)
    df_bn_after = df_bn[df_bn['timestamp'] > overlap_end].reset_index(drop=True)

    period_results = {}

    if len(df_bn_before) > 100:
        trades_before = backtest_portfolio(df_bn_before, PATTERNS, MAX_BARS)
        metrics_before = calc_metrics(trades_before)
        pre_days = (overlap_start - bn_start).days
        print(f"\n[Binance PRE-overlap]  {pre_days:>3}d: {metrics_before['trades']:>4} trades, "
              f"WR {metrics_before['wr']:>5.1f}%, PnL {metrics_before['pnl']:>+8.1f}%", flush=True)
        period_results['pre_overlap'] = {'days': pre_days, **metrics_before}

    print(f"[Binance OVERLAP]     {overlap_days:>3}d: {metrics_bn['trades']:>4} trades, "
          f"WR {metrics_bn['wr']:>5.1f}%, PnL {metrics_bn['pnl']:>+8.1f}%", flush=True)
    period_results['overlap'] = {'days': overlap_days, **metrics_bn}

    if len(df_bn_after) > 100:
        trades_after = backtest_portfolio(df_bn_after, PATTERNS, MAX_BARS)
        metrics_after = calc_metrics(trades_after)
        post_days = (bn_end - overlap_end).days
        print(f"[Binance POST-overlap] {post_days:>3}d: {metrics_after['trades']:>4} trades, "
              f"WR {metrics_after['wr']:>5.1f}%, PnL {metrics_after['pnl']:>+8.1f}%", flush=True)
        period_results['post_overlap'] = {'days': post_days, **metrics_after}

    # Quarterly breakdown on full Binance data
    print(f"\nQuarterly breakdown (Binance):", flush=True)
    df_bn['quarter'] = df_bn['timestamp'].dt.to_period('Q').astype(str)
    quarters = sorted(df_bn['quarter'].unique())
    quarterly = []
    for q in quarters:
        df_q = df_bn[df_bn['quarter'] == q].reset_index(drop=True)
        if len(df_q) < 100:
            continue
        tq = backtest_portfolio(df_q, PATTERNS, MAX_BARS)
        mq = calc_metrics(tq)
        print(f"  {q}: {mq['trades']:>4} trades, WR {mq['wr']:>5.1f}%, "
              f"PnL {mq['pnl']:>+8.1f}%", flush=True)
        quarterly.append({'quarter': q, **mq})

    results['phase3'] = {
        'full_720d': metrics_full,
        'periods': period_results,
        'quarterly': quarterly,
    }

    # ── Phase 4: Signal-Level Divergence ──
    print("\n" + "=" * 60, flush=True)
    print("Phase 4: Signal-Level Divergence Analysis", flush=True)
    print("=" * 60, flush=True)

    pat_set = set(PATTERNS.keys())

    bx_signals = set()
    bn_signals = set()

    bx_ts = bx_ov['timestamp'].values
    bn_ts = bn_ov['timestamp'].values
    bx_p3 = bx_ov['pattern_3'].values
    bn_p3 = bn_ov['pattern_3'].values

    for i in range(len(bx_ov)):
        p = bx_p3[i]
        if p is not None and not (isinstance(p, float) and np.isnan(p)) and p in pat_set:
            bx_signals.add((str(bx_ts[i]), p))

    for i in range(len(bn_ov)):
        p = bn_p3[i]
        if p is not None and not (isinstance(p, float) and np.isnan(p)) and p in pat_set:
            bn_signals.add((str(bn_ts[i]), p))

    both = bx_signals & bn_signals
    bx_only = bx_signals - bn_signals
    bn_only = bn_signals - bx_signals

    print(f"BingX signals:  {len(bx_signals)}", flush=True)
    print(f"Binance signals: {len(bn_signals)}", flush=True)
    print(f"Both (same time + pattern): {len(both)}", flush=True)
    print(f"BingX only: {len(bx_only)}", flush=True)
    print(f"Binance only: {len(bn_only)}", flush=True)

    signal_overlap_rate = len(both) / max(len(bx_signals), 1) * 100
    print(f"Signal overlap rate: {signal_overlap_rate:.1f}%", flush=True)

    # Analyze BingX-only divergence reasons
    ts_to_bn_pat = {}
    for i in range(len(bn_ov)):
        ts_to_bn_pat[str(bn_ts[i])] = bn_p3[i]

    mismatch_reasons = defaultdict(int)
    bx_only_patterns = defaultdict(int)
    for ts, pat in bx_only:
        bn_pat = ts_to_bn_pat.get(ts)
        if bn_pat is None or (isinstance(bn_pat, float) and np.isnan(bn_pat)):
            mismatch_reasons['timestamp_missing'] += 1
        elif bn_pat != pat:
            mismatch_reasons['different_classification'] += 1
            bx_only_patterns[pat] += 1
        else:
            mismatch_reasons['unknown'] += 1

    print(f"\nBingX-only signal reasons:", flush=True)
    for reason, count in sorted(mismatch_reasons.items(), key=lambda x: -x[1]):
        print(f"  {reason}: {count}", flush=True)

    # Which patterns are most affected?
    if bx_only_patterns:
        print(f"\nMost divergent patterns (BingX-only count):", flush=True)
        for pat, cnt in sorted(bx_only_patterns.items(), key=lambda x: -x[1])[:10]:
            total_bx = sum(1 for ts, p in bx_signals if p == pat)
            print(f"  {pat:<15}: {cnt} BX-only / {total_bx} total ({cnt/max(total_bx,1)*100:.0f}%)", flush=True)

    results['phase4'] = {
        'bx_signals': len(bx_signals),
        'bn_signals': len(bn_signals),
        'both_signals': len(both),
        'bx_only': len(bx_only),
        'bn_only': len(bn_only),
        'signal_overlap_rate': round(signal_overlap_rate, 1),
        'mismatch_reasons': dict(mismatch_reasons),
    }

    # ── Phase 5: MFE/MAE on Matched Signals ──
    print("\n" + "=" * 60, flush=True)
    print("Phase 5: MFE/MAE on Matched Signals", flush=True)
    print("=" * 60, flush=True)

    bx_ts_idx = {str(bx_ts[i]): i for i in range(len(bx_ov))}
    bn_ts_idx = {str(bn_ts[i]): i for i in range(len(bn_ov))}

    bx_highs = bx_ov['high'].values
    bx_lows = bx_ov['low'].values
    bx_opens = bx_ov['open'].values
    bn_highs = bn_ov['high'].values
    bn_lows = bn_ov['low'].values
    bn_opens = bn_ov['open'].values

    horizons = [10, 50, 100, 200, 500]
    mfe_mae = {h: {'bx_mfe': [], 'bx_mae': [], 'bn_mfe': [], 'bn_mae': []} for h in horizons}

    for ts, pat in both:
        bx_idx = bx_ts_idx.get(ts)
        bn_idx = bn_ts_idx.get(ts)
        if bx_idx is None or bn_idx is None:
            continue

        direction = PATTERNS[pat][0]

        for hz in horizons:
            # BingX
            bx_ei = bx_idx + 1
            if bx_ei >= len(bx_ov):
                continue
            bx_ep = bx_opens[bx_ei]
            bx_end = min(bx_ei + hz, len(bx_ov))
            if bx_ep <= 0 or bx_end <= bx_ei:
                continue

            if direction == 'LONG':
                bx_mfe = (np.max(bx_highs[bx_ei:bx_end]) - bx_ep) / bx_ep * 100
                bx_mae = (bx_ep - np.min(bx_lows[bx_ei:bx_end])) / bx_ep * 100
            else:
                bx_mfe = (bx_ep - np.min(bx_lows[bx_ei:bx_end])) / bx_ep * 100
                bx_mae = (np.max(bx_highs[bx_ei:bx_end]) - bx_ep) / bx_ep * 100

            # Binance
            bn_ei = bn_idx + 1
            if bn_ei >= len(bn_ov):
                continue
            bn_ep = bn_opens[bn_ei]
            bn_end = min(bn_ei + hz, len(bn_ov))
            if bn_ep <= 0 or bn_end <= bn_ei:
                continue

            if direction == 'LONG':
                bn_mfe = (np.max(bn_highs[bn_ei:bn_end]) - bn_ep) / bn_ep * 100
                bn_mae = (bn_ep - np.min(bn_lows[bn_ei:bn_end])) / bn_ep * 100
            else:
                bn_mfe = (bn_ep - np.min(bn_lows[bn_ei:bn_end])) / bn_ep * 100
                bn_mae = (np.max(bn_highs[bn_ei:bn_end]) - bn_ep) / bn_ep * 100

            mfe_mae[hz]['bx_mfe'].append(bx_mfe)
            mfe_mae[hz]['bx_mae'].append(bx_mae)
            mfe_mae[hz]['bn_mfe'].append(bn_mfe)
            mfe_mae[hz]['bn_mae'].append(bn_mae)

    print(f"{'Hz':>6} {'BX_MFE':>8} {'BX_MAE':>8} {'BX_diff':>8} "
          f"{'BN_MFE':>8} {'BN_MAE':>8} {'BN_diff':>8} {'N':>5}", flush=True)
    print("-" * 68, flush=True)

    mfe_mae_results = {}
    for hz in horizons:
        d = mfe_mae[hz]
        n = len(d['bx_mfe'])
        if n == 0:
            continue
        bx_mfe_med = float(np.median(d['bx_mfe']))
        bx_mae_med = float(np.median(d['bx_mae']))
        bn_mfe_med = float(np.median(d['bn_mfe']))
        bn_mae_med = float(np.median(d['bn_mae']))
        print(f"{hz:>6} {bx_mfe_med:>7.3f}% {bx_mae_med:>7.3f}% {bx_mfe_med-bx_mae_med:>+7.3f}% "
              f"{bn_mfe_med:>7.3f}% {bn_mae_med:>7.3f}% {bn_mfe_med-bn_mae_med:>+7.3f}% {n:>5}", flush=True)

        mfe_mae_results[str(hz)] = {
            'n': n,
            'bx_mfe_p50': round(bx_mfe_med, 4), 'bx_mae_p50': round(bx_mae_med, 4),
            'bx_diff': round(bx_mfe_med - bx_mae_med, 4),
            'bn_mfe_p50': round(bn_mfe_med, 4), 'bn_mae_p50': round(bn_mae_med, 4),
            'bn_diff': round(bn_mfe_med - bn_mae_med, 4),
        }

    results['phase5'] = mfe_mae_results

    # ── Phase 6: Control — Full BingX 270d ──
    print("\n" + "=" * 60, flush=True)
    print("Phase 6: Control — Full BingX 270d", flush=True)
    print("=" * 60, flush=True)

    trades_bx_full = backtest_portfolio(df_bx, PATTERNS, MAX_BARS)
    metrics_bx_full = calc_metrics(trades_bx_full)
    print(f"[BingX 270d FULL] {metrics_bx_full['trades']:>4} trades, "
          f"WR {metrics_bx_full['wr']:>5.1f}%, PnL {metrics_bx_full['pnl']:>+8.1f}%, "
          f"MDD {metrics_bx_full['mdd']:>6.1f}%, PF {metrics_bx_full['pf']:.2f}", flush=True)
    print(f"(Control: should approximate v1.27.1 backtest ~+956%)", flush=True)

    results['phase6'] = {
        'bingx_270d_full': metrics_bx_full,
    }

    # ── Verdict ──
    print("\n" + "=" * 60, flush=True)
    print("VERDICT", flush=True)
    print("=" * 60, flush=True)

    bx_pnl = metrics_bx['pnl']
    bn_pnl = metrics_bn['pnl']
    bn_full_pnl = metrics_full['pnl']

    if bx_pnl > 50 and bn_pnl < -10:
        diagnosis = "EXCHANGE ARTIFACT — Edge exists only on BingX, not Binance"
    elif bx_pnl > 50 and bn_pnl > 0 and bn_full_pnl < -10:
        diagnosis = "TIME-LIMITED EDGE — Works on both for same period, decays on extended"
    elif bx_pnl > 50 and bn_pnl > 0 and bn_full_pnl > 0:
        diagnosis = "REAL EDGE — Works on both exchanges and extended period"
    elif bx_pnl <= 50 and bn_pnl <= 0:
        diagnosis = "OVERLAP TOO SHORT — Both exchanges show weak results for overlap period"
    else:
        diagnosis = (f"MIXED — BX_overlap {bx_pnl:+.1f}%, BN_overlap {bn_pnl:+.1f}%, "
                     f"BN_full {bn_full_pnl:+.1f}%")

    print(f"\nBingX 270d full:    {metrics_bx_full['pnl']:>+8.1f}%", flush=True)
    print(f"BingX overlap:      {bx_pnl:>+8.1f}%", flush=True)
    print(f"Binance overlap:    {bn_pnl:>+8.1f}%", flush=True)
    print(f"Binance 720d full:  {bn_full_pnl:>+8.1f}%", flush=True)
    print(f"Signal overlap:     {signal_overlap_rate:>7.1f}%", flush=True)
    print(f"Type match rate:    {type_match:>7.1f}%", flush=True)
    print(f"\nDIAGNOSIS: {diagnosis}", flush=True)

    results['verdict'] = {
        'bx_270d_full_pnl': metrics_bx_full['pnl'],
        'bx_overlap_pnl': bx_pnl,
        'bn_overlap_pnl': bn_pnl,
        'bn_720d_full_pnl': bn_full_pnl,
        'signal_overlap_rate': round(signal_overlap_rate, 1),
        'type_match_rate': round(type_match, 1),
        'diagnosis': diagnosis,
    }

    elapsed = time.time() - t0
    results['meta']['elapsed_sec'] = round(elapsed, 1)
    print(f"\nTotal time: {elapsed:.1f}s", flush=True)

    out_path = RESULTS_DIR / "extended_rediscovery_v3_cross_exchange.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Results saved: {out_path}", flush=True)


if __name__ == '__main__':
    main()
