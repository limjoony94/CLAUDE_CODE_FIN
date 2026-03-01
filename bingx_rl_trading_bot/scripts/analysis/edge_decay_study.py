#!/usr/bin/env python3
"""
Edge Decay Time-Series Study — 패턴 edge의 시간 감쇠 측정

핵심 질문:
  1. 현행 130패턴의 WR은 시간에 따라 어떻게 변하는가?
  2. Neutral window(~257d) 내부 vs 이후 구간에서 edge 차이는?
  3. 방향별(LONG/SHORT) edge decay 패턴은?
  4. 30-trade 롤링 WR 궤적은?
  5. BTC 가격 변화와 WR의 상관관계는?

방법:
  297d 데이터 + dynamic_patterns.json 130패턴.
  N=1 portfolio backtest → 시간별 WR 분석.

Protocol:
  - Production classify_candle() via scanner
  - LEVERAGE = 3, FEE = 0.10% × LEVERAGE
  - ATR-scaled TP/SL (scanner bt_signals_atr)
  - Timeout DROP (864 bars)
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
from collections import defaultdict
from datetime import datetime, timedelta

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
sys.path.insert(0, _PROJECT_ROOT)

import scripts.scanner.pattern_scanner as scanner

RESULT_FILE = os.path.join(_PROJECT_ROOT, 'results', 'edge_decay_study.json')
DATA_FILE = os.path.join(_PROJECT_ROOT, 'data', 'btc_5m_270days_reclassified.csv')
PATTERNS_FILE = os.path.join(_PROJECT_ROOT, 'results', 'dynamic_patterns.json')
LEVERAGE = scanner.LEVERAGE
FEE_PCT = scanner.FEE_PCT
BARS_PER_DAY = 288
TIMEOUT_BARS = 864


def load_patterns():
    """Load current 130 patterns from dynamic_patterns.json."""
    with open(PATTERNS_FILE) as f:
        data = json.load(f)
    patterns = []
    for key, info in data.get('pattern_details', {}).items():
        patterns.append({
            'pattern': info['pattern'],
            'direction': info['direction'],
            'tp': info['tp'],
            'sl': info['sl'],
        })
    return patterns, data.get('meta', {})


def generate_all_trades(patterns, signal_index, opens, highs, lows, closes,
                        n_bars, use_atr=True):
    """Generate all trades for given patterns using scanner backtest engine.

    Returns list of dicts with bar, direction, pnl, pattern, etc.
    """
    # Precompute ATR ratio array once
    atr_ratio = None
    if use_atr and hasattr(scanner, 'compute_atr_ratio'):
        atr_ratio = scanner.compute_atr_ratio(highs, lows, closes)

    all_trades = []

    for pat in patterns:
        sigs = signal_index.get(pat['pattern'], [])
        if not sigs:
            continue

        direction = pat['direction']
        tp_pct = pat['tp']
        sl_pct = pat['sl']

        # Use scanner's backtest engine
        if use_atr and atr_ratio is not None:
            raw_trades = scanner.bt_signals_atr(
                sigs, direction, tp_pct, sl_pct,
                opens, highs, lows, n_bars, atr_ratio)
        else:
            raw_trades = scanner.bt_signals(
                sigs, direction, tp_pct, sl_pct,
                opens, highs, lows, n_bars)

        for entry_bar, exit_bar, pnl in raw_trades:
            all_trades.append({
                'bar': entry_bar,
                'exit_bar': exit_bar,
                'direction': direction,
                'pnl': round(pnl, 4),
                'pattern': pat['pattern'],
                'tp': tp_pct,
                'sl': sl_pct,
                'hold_bars': exit_bar - entry_bar,
                'exit_reason': 'TP' if pnl > 0 else 'SL',
                'atr_ratio': round(float(atr_ratio[entry_bar]), 4) if atr_ratio is not None and entry_bar < len(atr_ratio) else 1.0,
            })

    # Sort by bar index
    all_trades.sort(key=lambda t: t['bar'])
    return all_trades


def main():
    t0 = time.time()
    results = {'meta': {}, 'sections': {}}

    # ── Load data ──
    print("Loading data...")
    df = scanner.load_and_classify(DATA_FILE)
    types = df['candle_type'].tolist()
    signal_index = scanner.build_signal_index(types, len(df))
    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    n_bars = len(df)

    # Parse timestamps for date mapping
    timestamps = pd.to_datetime(df['timestamp'])
    bar_to_date = timestamps.dt.date.values

    patterns, pat_meta = load_patterns()
    neutral_window = pat_meta.get('neutral_window', {})
    neutral_start = neutral_window.get('start_bar', 0)
    neutral_end = neutral_window.get('end_bar', n_bars)

    print(f"  {n_bars} bars ({n_bars/BARS_PER_DAY:.0f} days)")
    print(f"  {len(patterns)} patterns")
    print(f"  Neutral window: bar {neutral_start}→{neutral_end} "
          f"({(neutral_end-neutral_start)/BARS_PER_DAY:.0f}d)")

    results['meta'] = {
        'data_file': os.path.basename(DATA_FILE),
        'bars': n_bars,
        'days': round(n_bars / BARS_PER_DAY, 1),
        'patterns': len(patterns),
        'neutral_window_bars': [neutral_start, neutral_end],
        'timeout_bars': TIMEOUT_BARS,
        'leverage': LEVERAGE,
    }

    # ── Generate all trades ──
    print("\nGenerating trades (ATR-scaled)...")
    trades = generate_all_trades(
        patterns, signal_index, opens, highs, lows, closes, n_bars, use_atr=True)
    print(f"  Total trades: {len(trades)}")
    wins = sum(1 for t in trades if t['pnl'] > 0)
    print(f"  WR: {wins/len(trades)*100:.1f}% ({wins}W/{len(trades)-wins}L)")

    # ═══════════════════════════════════════════════════════════
    # 1. MONTHLY WR BREAKDOWN
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*65}")
    print("1. MONTHLY WR BREAKDOWN")
    print(f"{'='*65}")

    monthly = defaultdict(lambda: {'wins': 0, 'total': 0, 'pnls': [], 'long_w': 0, 'long_t': 0, 'short_w': 0, 'short_t': 0})
    for t in trades:
        date = bar_to_date[min(t['bar'], n_bars - 1)]
        month_key = f"{date.year}-{date.month:02d}"
        monthly[month_key]['total'] += 1
        if t['pnl'] > 0:
            monthly[month_key]['wins'] += 1
        monthly[month_key]['pnls'].append(t['pnl'])
        if t['direction'] == 'LONG':
            monthly[month_key]['long_t'] += 1
            if t['pnl'] > 0:
                monthly[month_key]['long_w'] += 1
        else:
            monthly[month_key]['short_t'] += 1
            if t['pnl'] > 0:
                monthly[month_key]['short_w'] += 1

    print(f"  {'Month':<10} {'Trades':>6} {'WR%':>6} {'PnL%':>8} {'L-WR%':>7} {'S-WR%':>7} {'ATR':>5}")
    print(f"  {'-'*55}")

    monthly_output = []
    for month in sorted(monthly.keys()):
        m = monthly[month]
        wr = m['wins'] / m['total'] * 100
        pnl = sum(m['pnls'])
        l_wr = m['long_w'] / m['long_t'] * 100 if m['long_t'] > 0 else 0
        s_wr = m['short_w'] / m['short_t'] * 100 if m['short_t'] > 0 else 0
        # Average ATR ratio for the month
        month_trades = [t for t in trades if bar_to_date[min(t['bar'], n_bars-1)].strftime('%Y-%m') == month]
        avg_atr = np.mean([t['atr_ratio'] for t in month_trades]) if month_trades else 1.0

        print(f"  {month:<10} {m['total']:>6} {wr:>6.1f} {pnl:>+8.1f} "
              f"{l_wr:>6.1f}% {s_wr:>6.1f}% {avg_atr:>5.2f}")
        monthly_output.append({
            'month': month, 'trades': m['total'], 'wr': round(wr, 1),
            'pnl': round(pnl, 1), 'long_wr': round(l_wr, 1), 'short_wr': round(s_wr, 1),
            'avg_atr': round(avg_atr, 3),
        })
    results['sections']['monthly'] = monthly_output

    # ═══════════════════════════════════════════════════════════
    # 2. 15-DAY PERIOD WR
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*65}")
    print("2. 15-DAY PERIOD WR")
    print(f"{'='*65}")

    period_bars = 15 * BARS_PER_DAY
    n_periods = n_bars // period_bars
    period_output = []

    print(f"  {'Period':<8} {'Days':<12} {'Trades':>6} {'WR%':>6} {'Edge':>7} {'PnL%':>8}")
    print(f"  {'-'*55}")

    for p in range(n_periods):
        p_start = p * period_bars
        p_end = min((p + 1) * period_bars, n_bars)
        p_trades = [t for t in trades if p_start <= t['bar'] < p_end]

        if not p_trades:
            continue

        w = sum(1 for t in p_trades if t['pnl'] > 0)
        wr = w / len(p_trades) * 100
        pnl = sum(t['pnl'] for t in p_trades)

        # Average baseline WR
        avg_baseline = np.mean([
            pat['sl'] / (pat['tp'] + pat['sl']) * 100
            for pat in patterns
        ])
        edge = wr - avg_baseline

        start_date = bar_to_date[p_start]
        end_date = bar_to_date[min(p_end - 1, n_bars - 1)]
        day_label = f"{start_date}"

        print(f"  {p+1:<8} {str(day_label):<12} {len(p_trades):>6} {wr:>6.1f} "
              f"{edge:>+6.1f}pp {pnl:>+8.1f}")

        period_output.append({
            'period': p + 1, 'start_date': str(start_date), 'end_date': str(end_date),
            'trades': len(p_trades), 'wr': round(wr, 1), 'edge': round(edge, 1),
            'pnl': round(pnl, 1),
        })
    results['sections']['periods_15d'] = period_output

    # ═══════════════════════════════════════════════════════════
    # 3. ROLLING 30-TRADE WR TRAJECTORY
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*65}")
    print("3. ROLLING 30-TRADE WR TRAJECTORY")
    print(f"{'='*65}")

    window = 30
    rolling_wrs = []
    for i in range(window, len(trades) + 1):
        chunk = trades[i - window:i]
        w = sum(1 for t in chunk if t['pnl'] > 0)
        wr = w / window * 100
        bar = chunk[-1]['bar']
        date = bar_to_date[min(bar, n_bars - 1)]
        rolling_wrs.append({
            'trade_num': i, 'wr': round(wr, 1), 'bar': int(bar), 'date': str(date),
        })

    if rolling_wrs:
        wrs_arr = [r['wr'] for r in rolling_wrs]
        print(f"  Rolling WR (window={window}): mean {np.mean(wrs_arr):.1f}%, "
              f"std {np.std(wrs_arr):.1f}%, "
              f"min {min(wrs_arr):.1f}%, max {max(wrs_arr):.1f}%")

        # First/last quartile
        q1_wrs = wrs_arr[:len(wrs_arr)//4]
        q4_wrs = wrs_arr[-len(wrs_arr)//4:]
        print(f"  First quartile avg: {np.mean(q1_wrs):.1f}%")
        print(f"  Last quartile avg:  {np.mean(q4_wrs):.1f}%")
        print(f"  Decay: {np.mean(q1_wrs) - np.mean(q4_wrs):+.1f}pp "
              f"(positive = edge decaying)")

        # Print every 50th point
        print(f"\n  {'Trade#':>7} {'Date':<12} {'Roll-WR':>8}")
        for r in rolling_wrs[::50]:
            print(f"  {r['trade_num']:>7} {r['date']:<12} {r['wr']:>7.1f}%")
        # Always show last
        if rolling_wrs and rolling_wrs[-1] not in rolling_wrs[::50]:
            r = rolling_wrs[-1]
            print(f"  {r['trade_num']:>7} {r['date']:<12} {r['wr']:>7.1f}%")

    results['sections']['rolling_wr'] = {
        'window': window,
        'mean': round(np.mean(wrs_arr), 1) if rolling_wrs else 0,
        'std': round(np.std(wrs_arr), 1) if rolling_wrs else 0,
        'min': round(min(wrs_arr), 1) if rolling_wrs else 0,
        'max': round(max(wrs_arr), 1) if rolling_wrs else 0,
        'first_quartile_avg': round(np.mean(q1_wrs), 1) if rolling_wrs else 0,
        'last_quartile_avg': round(np.mean(q4_wrs), 1) if rolling_wrs else 0,
        'trajectory': rolling_wrs[::10],  # Every 10th for JSON size
    }

    # ═══════════════════════════════════════════════════════════
    # 4. NEUTRAL WINDOW vs POST-NEUTRAL
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*65}")
    print("4. NEUTRAL WINDOW (IS) vs POST-NEUTRAL (pseudo-OOS)")
    print(f"{'='*65}")

    in_neutral = [t for t in trades if neutral_start <= t['bar'] < neutral_end]
    post_neutral = [t for t in trades if t['bar'] >= neutral_end]

    sections = [
        ('Neutral (IS)', in_neutral),
        ('Post-Neutral', post_neutral),
    ]

    neutral_output = {}
    for label, section_trades in sections:
        if not section_trades:
            print(f"\n  {label}: no trades")
            continue
        w = sum(1 for t in section_trades if t['pnl'] > 0)
        wr = w / len(section_trades) * 100
        pnl = sum(t['pnl'] for t in section_trades)
        l_t = [t for t in section_trades if t['direction'] == 'LONG']
        s_t = [t for t in section_trades if t['direction'] == 'SHORT']
        l_wr = sum(1 for t in l_t if t['pnl'] > 0) / len(l_t) * 100 if l_t else 0
        s_wr = sum(1 for t in s_t if t['pnl'] > 0) / len(s_t) * 100 if s_t else 0

        print(f"\n  {label}:")
        print(f"    Trades: {len(section_trades)} ({len(l_t)}L + {len(s_t)}S)")
        print(f"    WR: {wr:.1f}% (LONG {l_wr:.1f}%, SHORT {s_wr:.1f}%)")
        print(f"    PnL: {pnl:+.1f}%")

        neutral_output[label] = {
            'trades': len(section_trades), 'wr': round(wr, 1),
            'long_trades': len(l_t), 'short_trades': len(s_t),
            'long_wr': round(l_wr, 1), 'short_wr': round(s_wr, 1),
            'pnl': round(pnl, 1),
        }

    if 'Neutral (IS)' in neutral_output and 'Post-Neutral' in neutral_output:
        wr_drop = neutral_output['Neutral (IS)']['wr'] - neutral_output['Post-Neutral']['wr']
        print(f"\n  WR drop: {wr_drop:+.1f}pp (neutral→post)")
        neutral_output['wr_drop_pp'] = round(wr_drop, 1)

    results['sections']['neutral_comparison'] = neutral_output

    # ═══════════════════════════════════════════════════════════
    # 5. DIRECTION-SPECIFIC DECAY
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*65}")
    print("5. DIRECTION-SPECIFIC EDGE OVER TIME")
    print(f"{'='*65}")

    for direction in ['LONG', 'SHORT']:
        d_trades = [t for t in trades if t['direction'] == direction]
        if len(d_trades) < window:
            print(f"\n  {direction}: insufficient trades ({len(d_trades)})")
            continue

        # Split into thirds
        third = len(d_trades) // 3
        thirds = [
            ('First third', d_trades[:third]),
            ('Middle third', d_trades[third:2*third]),
            ('Last third', d_trades[2*third:]),
        ]

        print(f"\n  {direction} ({len(d_trades)} trades):")
        for label, chunk in thirds:
            w = sum(1 for t in chunk if t['pnl'] > 0)
            wr = w / len(chunk) * 100
            pnl = sum(t['pnl'] for t in chunk)
            start_date = bar_to_date[min(chunk[0]['bar'], n_bars-1)]
            end_date = bar_to_date[min(chunk[-1]['bar'], n_bars-1)]
            print(f"    {label:<14} {len(chunk):>4} trades | WR {wr:>5.1f}% | "
                  f"PnL {pnl:>+7.1f}% | {start_date}→{end_date}")

    results['sections']['direction_decay'] = {}
    for direction in ['LONG', 'SHORT']:
        d_trades = [t for t in trades if t['direction'] == direction]
        third = max(1, len(d_trades) // 3)
        chunks = [d_trades[:third], d_trades[third:2*third], d_trades[2*third:]]
        results['sections']['direction_decay'][direction] = [
            {'segment': i+1, 'trades': len(c),
             'wr': round(sum(1 for t in c if t['pnl'] > 0) / len(c) * 100, 1) if c else 0,
             'pnl': round(sum(t['pnl'] for t in c), 1)}
            for i, c in enumerate(chunks) if c
        ]

    # ═══════════════════════════════════════════════════════════
    # 6. PER-PATTERN STABILITY
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*65}")
    print("6. PER-PATTERN STABILITY (first-half vs last-half WR)")
    print(f"{'='*65}")

    pat_stability = {}
    for pat in patterns:
        key = f"{pat['pattern']}_{pat['direction']}"
        p_trades = [t for t in trades
                    if t['pattern'] == pat['pattern'] and t['direction'] == pat['direction']]
        if len(p_trades) < 4:
            continue

        mid = len(p_trades) // 2
        first = p_trades[:mid]
        second = p_trades[mid:]

        wr1 = sum(1 for t in first if t['pnl'] > 0) / len(first) * 100
        wr2 = sum(1 for t in second if t['pnl'] > 0) / len(second) * 100
        delta = wr2 - wr1

        pat_stability[key] = {
            'total': len(p_trades),
            'first_half_wr': round(wr1, 1),
            'second_half_wr': round(wr2, 1),
            'wr_delta': round(delta, 1),
        }

    # Sort by delta (most decaying first)
    sorted_pats = sorted(pat_stability.items(), key=lambda x: x[1]['wr_delta'])

    print(f"\n  Most DECAYING patterns (WR drops):")
    print(f"  {'Pattern':<20} {'Trades':>6} {'1st-WR':>7} {'2nd-WR':>7} {'Delta':>7}")
    for key, s in sorted_pats[:10]:
        print(f"  {key:<20} {s['total']:>6} {s['first_half_wr']:>6.1f}% "
              f"{s['second_half_wr']:>6.1f}% {s['wr_delta']:>+6.1f}pp")

    print(f"\n  Most STABLE/IMPROVING patterns:")
    for key, s in sorted_pats[-5:]:
        print(f"  {key:<20} {s['total']:>6} {s['first_half_wr']:>6.1f}% "
              f"{s['second_half_wr']:>6.1f}% {s['wr_delta']:>+6.1f}pp")

    # Statistics
    deltas = [s['wr_delta'] for s in pat_stability.values()]
    decaying = sum(1 for d in deltas if d < -5)
    stable = sum(1 for d in deltas if -5 <= d <= 5)
    improving = sum(1 for d in deltas if d > 5)
    print(f"\n  Distribution: {decaying} decaying (<-5pp), {stable} stable (±5pp), "
          f"{improving} improving (>+5pp)")
    print(f"  Mean delta: {np.mean(deltas):+.1f}pp, median: {np.median(deltas):+.1f}pp")

    results['sections']['pattern_stability'] = {
        'n_patterns': len(pat_stability),
        'decaying': decaying, 'stable': stable, 'improving': improving,
        'mean_delta': round(float(np.mean(deltas)), 1),
        'median_delta': round(float(np.median(deltas)), 1),
        'worst_10': [{**{'key': k}, **v} for k, v in sorted_pats[:10]],
        'best_5': [{**{'key': k}, **v} for k, v in sorted_pats[-5:]],
    }

    # ═══════════════════════════════════════════════════════════
    # 7. BTC PRICE vs WR CORRELATION
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*65}")
    print("7. BTC PRICE CHANGE vs WR CORRELATION")
    print(f"{'='*65}")

    # 15-day BTC returns vs 15-day WR
    btc_returns = []
    period_wrs_for_corr = []
    for p in range(n_periods):
        p_start = p * period_bars
        p_end = min((p + 1) * period_bars, n_bars) - 1
        btc_ret = (closes[p_end] - closes[p_start]) / closes[p_start] * 100
        p_trades_period = [t for t in trades if p_start <= t['bar'] < p_end + 1]
        if len(p_trades_period) >= 5:
            w = sum(1 for t in p_trades_period if t['pnl'] > 0)
            wr = w / len(p_trades_period) * 100
            btc_returns.append(btc_ret)
            period_wrs_for_corr.append(wr)

    if len(btc_returns) >= 5:
        corr = np.corrcoef(btc_returns, period_wrs_for_corr)[0, 1]
        print(f"  BTC 15d return vs WR correlation: {corr:+.3f}")
        print(f"  (negative = bot performs worse when BTC rises)")

        # Direction breakdown
        for direction in ['LONG', 'SHORT']:
            dir_wrs = []
            for p in range(n_periods):
                p_start = p * period_bars
                p_end = min((p + 1) * period_bars, n_bars) - 1
                d_trades = [t for t in trades
                            if p_start <= t['bar'] < p_end + 1 and t['direction'] == direction]
                if len(d_trades) >= 3:
                    w = sum(1 for t in d_trades if t['pnl'] > 0)
                    dir_wrs.append(w / len(d_trades) * 100)
                else:
                    dir_wrs.append(None)

            valid = [(r, w) for r, w in zip(btc_returns, dir_wrs)
                     if w is not None and r is not None]
            if len(valid) >= 5:
                r_arr, w_arr = zip(*valid)
                dir_corr = np.corrcoef(r_arr, w_arr)[0, 1]
                print(f"  {direction} WR vs BTC return: {dir_corr:+.3f}")
    else:
        corr = 0
        print(f"  Insufficient data for correlation")

    results['sections']['btc_correlation'] = {
        'btc_return_vs_wr_corr': round(float(corr), 3) if btc_returns else 0,
        'n_periods': len(btc_returns),
    }

    # ═══════════════════════════════════════════════════════════
    # 8. EDGE HALF-LIFE ESTIMATION
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*65}")
    print("8. EDGE HALF-LIFE ESTIMATION")
    print(f"{'='*65}")

    if rolling_wrs:
        avg_baseline = np.mean([
            pat['sl'] / (pat['tp'] + pat['sl']) * 100
            for pat in patterns
        ])
        print(f"  Average baseline (random walk) WR: {avg_baseline:.1f}%")

        # Rolling edge = rolling WR - baseline
        rolling_edges = [r['wr'] - avg_baseline for r in rolling_wrs]
        initial_edge = np.mean(rolling_edges[:len(rolling_edges)//4])
        half_edge = initial_edge / 2

        half_life_trade = None
        for i, e in enumerate(rolling_edges):
            if e < half_edge:
                half_life_trade = i + window
                break

        first_neg_trade = None
        for i, e in enumerate(rolling_edges):
            if e <= 0:
                first_neg_trade = i + window
                break

        print(f"  Initial edge (first quartile): {initial_edge:+.1f}pp")
        print(f"  Final edge (last quartile): "
              f"{np.mean(rolling_edges[-len(rolling_edges)//4:]):+.1f}pp")

        if half_life_trade:
            hl_bar = trades[min(half_life_trade, len(trades)-1)]['bar']
            hl_days = hl_bar / BARS_PER_DAY
            print(f"  Half-life: trade #{half_life_trade} (~{hl_days:.0f} days into data)")
        else:
            print(f"  Half-life: NOT REACHED (edge sustained)")

        if first_neg_trade:
            neg_bar = trades[min(first_neg_trade, len(trades)-1)]['bar']
            neg_days = neg_bar / BARS_PER_DAY
            print(f"  First negative edge: trade #{first_neg_trade} (~{neg_days:.0f} days)")
        else:
            print(f"  First negative edge: NOT REACHED")

        results['sections']['half_life'] = {
            'baseline_wr': round(avg_baseline, 1),
            'initial_edge_pp': round(initial_edge, 1),
            'final_edge_pp': round(float(np.mean(rolling_edges[-len(rolling_edges)//4:])), 1),
            'half_life_trade': half_life_trade,
            'first_neg_trade': first_neg_trade,
        }

    # ═══════════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*65}")
    print("SUMMARY")
    print(f"{'='*65}")

    total_wr = wins / len(trades) * 100
    print(f"  Total trades: {len(trades)}")
    print(f"  Overall WR: {total_wr:.1f}%")

    if monthly_output:
        first_month_wr = monthly_output[0]['wr']
        last_month_wr = monthly_output[-1]['wr']
        print(f"  First month WR: {first_month_wr:.1f}%")
        print(f"  Last month WR: {last_month_wr:.1f}%")
        print(f"  Monthly WR trend: {first_month_wr - last_month_wr:+.1f}pp decay")

    if 'Neutral (IS)' in neutral_output and 'Post-Neutral' in neutral_output:
        print(f"  Neutral WR: {neutral_output['Neutral (IS)']['wr']:.1f}%")
        print(f"  Post-neutral WR: {neutral_output['Post-Neutral']['wr']:.1f}%")

    if rolling_wrs:
        print(f"  Rolling WR range: {min(wrs_arr):.1f}% — {max(wrs_arr):.1f}%")

    print(f"  Pattern stability: {decaying} decaying, {stable} stable, {improving} improving")

    elapsed = time.time() - t0
    results['meta']['elapsed_seconds'] = round(elapsed, 1)
    print(f"\n  Time: {elapsed:.1f}s")

    # Save
    def json_safe(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (pd.Timestamp, datetime)):
            return str(obj)
        raise TypeError(f"Not serializable: {type(obj)}")

    with open(RESULT_FILE, 'w') as f:
        json.dump(results, f, indent=2, default=json_safe)
    print(f"  Saved to {RESULT_FILE}")


if __name__ == '__main__':
    main()
