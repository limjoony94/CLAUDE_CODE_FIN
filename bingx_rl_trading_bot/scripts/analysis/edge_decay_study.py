#!/usr/bin/env python3
"""
Edge Decay Study — 패턴 edge의 시간 감쇠 측정

핵심 질문:
  1. 발굴 시점 이후 edge(WR - baseline WR)는 어떻게 변하는가?
  2. Edge half-life는 얼마인가? (= re-scan 주기 결정 근거)
  3. 어떤 패턴 특성이 decay 저항성과 관련있는가?

방법:
  720일 Binance 데이터에서 Expanding Window 사용.
  IS 크기를 90/135/180/225/270일로 다양화.
  각 IS에서 발굴된 패턴의 OOS 성과를 30일 단위로 측정.

Protocol:
  - Production classify_candle() via scanner
  - LEVERAGE = 3, FEE = 0.10%
  - Per-pattern grid search (current scanner mode)
  - MC < 0.01, edge >= 21.8pp, min_trades >= 25
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
from collections import Counter

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
sys.path.insert(0, _PROJECT_ROOT)

import scripts.scanner.pattern_scanner as scanner

RESULT_FILE = os.path.join(_PROJECT_ROOT, 'results', 'edge_decay_study.json')
DATA_720D = os.path.join(_PROJECT_ROOT, 'data', 'btc_5m_720days_binance.csv')
LEVERAGE = scanner.LEVERAGE
FEE_PCT = scanner.FEE_PCT
EDGE_THRESHOLD = 21.8
BARS_PER_DAY = 288  # 24h * 60/5


def measure_oos_window(patterns, signal_index, opens, highs, lows, n_bars,
                       window_start, window_end):
    """Measure pattern portfolio performance in a specific bar range.

    Returns dict with WR, edge, PnL, MDD, trades, per-pattern detail.
    """
    all_trades = []
    pat_detail = {}

    for pat in patterns:
        sigs = [s for s in signal_index[pat['pattern']]
                if window_start <= s < window_end]
        if not sigs:
            continue

        trades = scanner.bt_signals(
            sigs, pat['direction'], pat['tp'], pat['sl'],
            opens, highs, lows, window_end
        )
        if not trades:
            continue

        pnls = [t[2] for t in trades]
        wr = len([p for p in pnls if p > 0]) / len(pnls) * 100
        baseline_wr = pat['sl'] / (pat['tp'] + pat['sl']) * 100
        edge = wr - baseline_wr

        pat_detail[f"{pat['pattern']}_{pat['direction']}"] = {
            'trades': len(trades), 'wr': round(wr, 1),
            'edge': round(edge, 1), 'pnl': round(sum(pnls), 1),
        }

        all_trades.extend(trades)

    if not all_trades:
        return {
            'trades': 0, 'wr': 0, 'edge': 0, 'pnl': 0, 'mdd': 0, 'pf': 0,
            'active_patterns': 0, 'pat_detail': {},
        }

    # Portfolio: 1-pos-at-a-time
    port = scanner.portfolio_1pos(all_trades)
    stats = scanner.calc_stats(port)

    # Calculate portfolio-level edge
    avg_baseline = np.mean([
        pat['sl'] / (pat['tp'] + pat['sl']) * 100
        for pat in patterns
    ])
    port_edge = stats['wr'] - avg_baseline

    return {
        'trades': stats['trades'],
        'wr': stats['wr'],
        'edge': round(port_edge, 1),
        'pnl': stats['pnl'],
        'mdd': stats['mdd'],
        'pf': stats['pf'],
        'active_patterns': len(pat_detail),
        'pat_detail': pat_detail,
    }


def main():
    t0 = time.time()
    results = {'meta': {}, 'decay_curves': {}, 'summary': {}}

    # Load 720-day data
    print("Loading 720-day Binance data...")
    df = scanner.load_and_classify(DATA_720D)
    types = df['candle_type'].tolist()
    signal_index = scanner.build_signal_index(types, len(df))
    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    n_bars = len(df)
    days_total = n_bars / BARS_PER_DAY

    print(f"  {n_bars} bars, {days_total:.0f} days")
    results['meta'] = {
        'data': '720d_binance',
        'bars': n_bars,
        'days': round(days_total, 1),
        'edge_threshold': EDGE_THRESHOLD,
        'leverage': LEVERAGE,
        'fee_pct': FEE_PCT,
    }

    # IS windows: 90, 135, 180, 225, 270 days
    is_days_list = [90, 135, 180, 225, 270]
    oos_chunk_days = 30  # measure in 30-day windows
    oos_chunk_bars = oos_chunk_days * BARS_PER_DAY

    print(f"\n  IS windows: {is_days_list} days")
    print(f"  OOS measurement: {oos_chunk_days}-day windows")
    print(f"  Edge threshold: {EDGE_THRESHOLD}pp")

    # ========================================
    # Main Study: Decay Curves
    # ========================================
    for is_days in is_days_list:
        is_bars = is_days * BARS_PER_DAY
        if is_bars >= n_bars:
            print(f"\n  [SKIP] IS={is_days}d — exceeds data")
            continue

        oos_available_bars = n_bars - is_bars
        oos_available_days = oos_available_bars / BARS_PER_DAY
        n_windows = int(oos_available_days // oos_chunk_days)

        print(f"\n{'='*60}")
        print(f"IS = {is_days} days ({is_bars} bars)")
        print(f"  OOS available: {oos_available_days:.0f} days → {n_windows} x {oos_chunk_days}d windows")
        print(f"{'='*60}")

        # Discover patterns on IS
        print(f"  Discovering patterns (per_pattern grid search)...")
        is_patterns = scanner.scan_universe_range(
            signal_index, opens, highs, lows, n_bars,
            bar_start=0, bar_end=is_bars, mode='per_pattern',
            min_trades=scanner.DEFAULT_MIN_TRADES,
            edge_threshold=EDGE_THRESHOLD,
            mc_threshold=scanner.DEFAULT_MC_THRESHOLD,
            max_baseline_wr=scanner.MAX_BASELINE_WR,
        )

        # Dedup: best edge per pattern (same as scanner)
        best_by_pat = {}
        for r in is_patterns:
            key = f"{r['pattern']}_{r['direction']}"
            if key not in best_by_pat or r['edge'] > best_by_pat[key]['edge']:
                best_by_pat[key] = r
        is_patterns = list(best_by_pat.values())

        n_long = sum(1 for p in is_patterns if p['direction'] == 'LONG')
        n_short = sum(1 for p in is_patterns if p['direction'] == 'SHORT')
        print(f"  Found {len(is_patterns)} patterns ({n_long}L + {n_short}S)")

        if not is_patterns:
            print(f"  [SKIP] No patterns found")
            continue

        # IS performance (sanity check)
        is_perf = measure_oos_window(
            is_patterns, signal_index, opens, highs, lows, n_bars,
            window_start=0, window_end=is_bars
        )
        print(f"  IS perf: WR {is_perf['wr']}%, edge {is_perf['edge']:+.1f}pp, "
              f"PnL {is_perf['pnl']:+.1f}%, trades {is_perf['trades']}")

        # Measure OOS in 30-day windows
        windows = []
        print(f"\n  {'Window':<12} {'Days':<10} {'WR':>6} {'Edge':>7} {'PnL':>8} "
              f"{'MDD':>6} {'Trades':>7} {'Active':>7}")
        print(f"  {'-'*65}")

        for w in range(n_windows):
            w_start = is_bars + w * oos_chunk_bars
            w_end = min(is_bars + (w + 1) * oos_chunk_bars, n_bars)
            days_since = (w + 1) * oos_chunk_days  # midpoint days since IS end

            perf = measure_oos_window(
                is_patterns, signal_index, opens, highs, lows, n_bars,
                window_start=w_start, window_end=w_end
            )

            window_result = {
                'window': w + 1,
                'days_since_discovery': days_since,
                'bar_range': [int(w_start), int(w_end)],
                **{k: v for k, v in perf.items() if k != 'pat_detail'},
            }
            windows.append(window_result)

            # Print row
            label = f"OOS {w+1}"
            day_range = f"+{(w)*oos_chunk_days}-{(w+1)*oos_chunk_days}d"
            print(f"  {label:<12} {day_range:<10} {perf['wr']:>5.1f}% {perf['edge']:>+6.1f}pp "
                  f"{perf['pnl']:>+7.1f}% {perf['mdd']:>5.1f}% {perf['trades']:>6} "
                  f"{perf['active_patterns']:>6}/{len(is_patterns)}")

        # Decay statistics
        edges = [w['edge'] for w in windows if w['trades'] > 0]
        wrs = [w['wr'] for w in windows if w['trades'] > 0]
        pnls = [w['pnl'] for w in windows]

        # Edge half-life: first window where edge < IS_edge/2
        is_edge = is_perf['edge']
        half_edge = is_edge / 2
        half_life_days = None
        for w in windows:
            if w['trades'] > 0 and w['edge'] < half_edge:
                half_life_days = w['days_since_discovery']
                break

        # First negative edge window
        first_neg_days = None
        for w in windows:
            if w['trades'] > 0 and w['edge'] <= 0:
                first_neg_days = w['days_since_discovery']
                break

        # Cumulative OOS PnL
        cum_oos_pnl = round(sum(pnls), 1)
        positive_windows = sum(1 for p in pnls if p > 0)

        decay_summary = {
            'is_days': is_days,
            'is_patterns': len(is_patterns),
            'is_long': n_long, 'is_short': n_short,
            'is_edge': round(is_edge, 1),
            'is_wr': is_perf['wr'],
            'is_pnl': is_perf['pnl'],
            'oos_windows': n_windows,
            'oos_edges': [round(e, 1) for e in edges],
            'oos_wrs': [round(w, 1) for w in wrs],
            'oos_pnls': [round(p, 1) for p in pnls],
            'cum_oos_pnl': cum_oos_pnl,
            'positive_windows': positive_windows,
            'edge_half_life_days': half_life_days,
            'first_neg_edge_days': first_neg_days,
            'avg_oos_edge': round(float(np.mean(edges)), 1) if edges else 0,
            'median_oos_edge': round(float(np.median(edges)), 1) if edges else 0,
            'edge_decay_rate_pp_per_month': round(
                (is_edge - (float(np.mean(edges)) if edges else 0)) /
                (n_windows * oos_chunk_days / 30) if n_windows > 0 else 0, 2
            ),
        }

        print(f"\n  --- Decay Summary (IS={is_days}d) ---")
        print(f"  IS edge: {is_edge:+.1f}pp | Avg OOS edge: {decay_summary['avg_oos_edge']:+.1f}pp")
        print(f"  Edge half-life: {half_life_days}d" if half_life_days else "  Edge half-life: not reached")
        print(f"  First negative edge: {first_neg_days}d" if first_neg_days else "  First negative edge: not reached")
        print(f"  Positive windows: {positive_windows}/{n_windows}")
        print(f"  Cumulative OOS PnL: {cum_oos_pnl:+.1f}%")
        print(f"  Edge decay rate: {decay_summary['edge_decay_rate_pp_per_month']:+.2f}pp/month")

        results['decay_curves'][f'IS_{is_days}d'] = {
            'summary': decay_summary,
            'windows': windows,
        }

    # ========================================
    # Cross-IS Comparison
    # ========================================
    print(f"\n{'='*60}")
    print("CROSS-IS COMPARISON")
    print(f"{'='*60}")

    print(f"\n  {'IS':<8} {'Patterns':>8} {'IS Edge':>8} {'Avg OOS':>8} "
          f"{'Half-life':>10} {'Neg Edge':>10} {'OOS PnL':>8} {'Pos Win':>8}")
    print(f"  {'-'*72}")

    best_config = None
    best_cum_pnl = -9999

    for key in sorted(results['decay_curves'].keys()):
        s = results['decay_curves'][key]['summary']
        hl = f"{s['edge_half_life_days']}d" if s['edge_half_life_days'] else "never"
        neg = f"{s['first_neg_edge_days']}d" if s['first_neg_edge_days'] else "never"
        print(f"  {s['is_days']:<8} {s['is_patterns']:>8} {s['is_edge']:>+7.1f}pp "
              f"{s['avg_oos_edge']:>+7.1f}pp {hl:>10} {neg:>10} "
              f"{s['cum_oos_pnl']:>+7.1f}% {s['positive_windows']}/{s['oos_windows']}")

        if s['cum_oos_pnl'] > best_cum_pnl:
            best_cum_pnl = s['cum_oos_pnl']
            best_config = s

    # ========================================
    # Re-scan Recommendation
    # ========================================
    print(f"\n{'='*60}")
    print("RE-SCAN FREQUENCY RECOMMENDATION")
    print(f"{'='*60}")

    # Aggregate decay rates across all IS windows
    all_half_lives = [
        results['decay_curves'][k]['summary']['edge_half_life_days']
        for k in results['decay_curves']
        if results['decay_curves'][k]['summary']['edge_half_life_days'] is not None
    ]
    all_neg_edges = [
        results['decay_curves'][k]['summary']['first_neg_edge_days']
        for k in results['decay_curves']
        if results['decay_curves'][k]['summary']['first_neg_edge_days'] is not None
    ]
    all_decay_rates = [
        results['decay_curves'][k]['summary']['edge_decay_rate_pp_per_month']
        for k in results['decay_curves']
    ]

    if all_half_lives:
        median_hl = int(np.median(all_half_lives))
        print(f"\n  Edge half-life (median): {median_hl} days")
    else:
        median_hl = None
        print(f"\n  Edge half-life: NOT REACHED in any IS configuration")

    if all_neg_edges:
        median_neg = int(np.median(all_neg_edges))
        print(f"  First negative edge (median): {median_neg} days")
    else:
        median_neg = None
        print(f"  First negative edge: NOT REACHED")

    avg_decay = round(float(np.mean(all_decay_rates)), 2)
    print(f"  Average decay rate: {avg_decay:+.2f}pp/month")

    # Recommendation logic
    if median_hl and median_hl <= 60:
        rec = f"Aggressive: re-scan every {max(30, median_hl // 2)} days"
    elif median_hl and median_hl <= 120:
        rec = f"Moderate: re-scan every {median_hl // 2}-{median_hl} days"
    elif median_neg and median_neg <= 180:
        rec = f"Conservative: re-scan every {median_neg // 2} days"
    else:
        rec = "Edge is durable — re-scan every 90-180 days or when live WR drops"

    print(f"\n  RECOMMENDATION: {rec}")

    results['summary'] = {
        'median_half_life_days': median_hl,
        'median_first_neg_edge_days': median_neg,
        'avg_decay_rate_pp_per_month': avg_decay,
        'recommendation': rec,
        'best_is_config': best_config,
    }

    elapsed = time.time() - t0
    results['meta']['total_time_seconds'] = round(elapsed, 1)

    print(f"\n  Total time: {elapsed / 60:.1f} minutes")

    # Save
    def json_safe(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Not JSON serializable: {type(obj)}")

    with open(RESULT_FILE, 'w') as f:
        json.dump(results, f, indent=2, default=json_safe)
    print(f"  Results saved to {RESULT_FILE}")


if __name__ == '__main__':
    main()
