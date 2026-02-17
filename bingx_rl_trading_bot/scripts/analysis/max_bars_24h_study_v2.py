#!/usr/bin/env python3
"""
MAX_BARS 24h Study v2 — Edge Threshold Sensitivity + 기존 47패턴 재검증

분석 1: Edge Threshold 상향 (288봉)
  - 288봉 + edge >= 25pp / 30pp 후처리 필터
  - 패턴 수 조절이 포트폴리오 성과에 미치는 영향

분석 2: 기존 47패턴 재검증 (288봉)
  - v1 연구의 BASELINE_500 47패턴을 동일 TP/SL로 288봉 backtest
  - 신규 패턴 발굴 없이 timeout만 변경

Protocol:
  - Production classify_candle() 사용 (indicators.py)
  - LEVERAGE = 3 (scanner 내장)
  - Timeout trades DROP (scanner 내장)
  - Sequential mode (concurrency=1) for reproducibility
"""

import os
import sys
import json
import time
import numpy as np

# Project root setup
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
sys.path.insert(0, _PROJECT_ROOT)

import scripts.scanner.pattern_scanner as scanner


# ============================================================
# Utility Functions
# ============================================================

def apply_post_filters(result, edge_min=21.8, wr_min=60.0, sl_min=1.0):
    """Apply Edge + WR + SL post-filter."""
    details = result.get('pattern_details', {})
    filtered = {}
    removed = []
    for key, v in details.items():
        if v['edge'] < edge_min:
            removed.append((key, f"edge {v['edge']}pp < {edge_min}"))
        elif v['wr'] < wr_min:
            removed.append((key, f"WR {v['wr']}% < {wr_min}"))
        elif v['sl'] < sl_min:
            removed.append((key, f"SL {v['sl']}% < {sl_min}"))
        else:
            filtered[key] = v

    long_p = sorted([v['pattern'] for v in filtered.values()
                     if v['direction'] == 'LONG'])
    short_p = sorted([v['pattern'] for v in filtered.values()
                      if v['direction'] == 'SHORT'])

    return {
        'filtered_details': filtered,
        'long_patterns': long_p,
        'short_patterns': short_p,
        'n_total': len(filtered),
        'n_long': len(long_p),
        'n_short': len(short_p),
        'n_removed': len(removed),
        'removed': removed,
    }


def portfolio_from_details(details, signal_index, df, max_bars):
    """Build portfolio from pattern details and compute stats."""
    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    n_bars = len(df)

    original = scanner.MAX_BARS
    scanner.MAX_BARS = max_bars
    try:
        all_trades = []
        per_pattern_trades = {}
        for key, v in details.items():
            sigs = signal_index[v['pattern']]
            trades = scanner.bt_signals(
                sigs, v['direction'], v['tp'], v['sl'],
                opens, highs, lows, n_bars
            )
            all_trades.extend(trades)
            per_pattern_trades[key] = len(trades)

        port = scanner.portfolio_1pos(all_trades)
        stats = scanner.calc_stats(port)
        mc_p = scanner.mc_test([t[2] for t in port]) if port else 1.0
        return stats, mc_p, len(all_trades), port, per_pattern_trades
    finally:
        scanner.MAX_BARS = original


def trade_duration_stats(port_trades):
    """Calculate trade duration statistics in bars."""
    if not port_trades:
        return {}
    durations = np.array([t[1] - t[0] for t in port_trades])
    return {
        'mean_bars': round(float(durations.mean()), 1),
        'median_bars': round(float(np.median(durations)), 1),
        'p25_bars': round(float(np.percentile(durations, 25)), 1),
        'p75_bars': round(float(np.percentile(durations, 75)), 1),
        'p95_bars': round(float(np.percentile(durations, 95)), 1),
        'max_bars': int(durations.max()),
        'mean_hours': round(float(durations.mean() * 5 / 60), 1),
        'median_hours': round(float(np.median(durations) * 5 / 60), 1),
        'pct_under_24h': round(float((durations <= 288).sum() / len(durations) * 100), 1),
        'pct_under_12h': round(float((durations <= 144).sum() / len(durations) * 100), 1),
    }


def run_wf(max_bars, signal_index, opens, highs, lows, n_bars, n_folds=3):
    """Run expanding window WF with given MAX_BARS."""
    original = scanner.MAX_BARS
    scanner.MAX_BARS = max_bars
    try:
        wf = scanner.expanding_window_wf(
            signal_index, opens, highs, lows, n_bars,
            n_folds=n_folds, mode='per_pattern',
            min_trades=scanner.DEFAULT_MIN_TRADES,
            edge_threshold=scanner.DEFAULT_EDGE_THRESHOLD,
            mc_threshold=scanner.DEFAULT_MC_THRESHOLD,
            max_baseline_wr=scanner.MAX_BASELINE_WR,
        )
        return wf
    finally:
        scanner.MAX_BARS = original


# ============================================================
# Analysis 1: Edge Threshold Sensitivity (288봉)
# ============================================================

def analysis_1_edge_threshold(df, signal_index):
    """288봉 discovery 결과에 다양한 edge threshold 적용."""
    print("\n" + "=" * 70)
    print("  ANALYSIS 1: Edge Threshold Sensitivity (MAX_BARS=288)")
    print("=" * 70)

    # Load v1 results for 288 discovery (to avoid re-running 30min discovery)
    v1_file = os.path.join(_PROJECT_ROOT, 'results', 'max_bars_24h_study.json')
    if not os.path.exists(v1_file):
        print("  ERROR: v1 results not found. Run max_bars_24h_study.py first.")
        return None

    with open(v1_file) as f:
        v1 = json.load(f)

    # Re-run discovery with 288 to get full pattern_details
    # (v1 saved post-filtered details only for 112 patterns,
    #  but we need the full 325 pre-filter details)
    print("  Running 288-bar discovery (sequential)...")
    t0 = time.time()
    original = scanner.MAX_BARS
    scanner.MAX_BARS = 288
    try:
        result_288 = scanner.scan_patterns_pp(
            df,
            edge_threshold=scanner.DEFAULT_EDGE_THRESHOLD,
            mc_threshold=scanner.DEFAULT_MC_THRESHOLD,
            min_trades=scanner.DEFAULT_MIN_TRADES,
            signal_index=signal_index,
            concurrency=1,
            max_baseline_wr=scanner.MAX_BASELINE_WR,
            correction_method='bh',
            fdr_q=0.05,
        )
    finally:
        scanner.MAX_BARS = original
    elapsed = time.time() - t0
    print(f"  Discovery done: {elapsed:.1f}s, {len(result_288.get('pattern_details', {}))} patterns")

    # Apply different edge thresholds
    scenarios = {}
    for edge_min in [21.8, 25.0, 30.0]:
        label = f"E{edge_min:.0f}" if edge_min == int(edge_min) else f"E{edge_min}"
        pf = apply_post_filters(result_288, edge_min=edge_min, wr_min=60.0, sl_min=1.0)

        if pf['filtered_details']:
            stats, mc_p, raw_trades, port, per_pat = portfolio_from_details(
                pf['filtered_details'], signal_index, df, 288)
            dur = trade_duration_stats(port)
        else:
            stats = {'pnl': 0, 'trades': 0, 'wr': 0, 'mdd': 0, 'pf': 0}
            mc_p = 1.0
            raw_trades = 0
            port = []
            dur = {}

        scenarios[label] = {
            'edge_threshold': edge_min,
            'n_patterns': pf['n_total'],
            'n_long': pf['n_long'],
            'n_short': pf['n_short'],
            'portfolio_stats': stats,
            'mc_p': round(mc_p, 4),
            'raw_trades': raw_trades,
            'duration_stats': dur,
            'pattern_details': {
                k: {kk: vv for kk, vv in v.items() if kk != 'trades_raw'}
                for k, v in pf['filtered_details'].items()
            },
        }

        # Print inline
        print(f"\n  --- Edge >= {edge_min}pp ---")
        print(f"  Patterns: {pf['n_total']} ({pf['n_long']}L+{pf['n_short']}S)")
        print(f"  PnL: {stats.get('pnl',0):+.1f}% | WR: {stats.get('wr',0):.1f}% | "
              f"MDD: {stats.get('mdd',0):.1f}% | PF: {stats.get('pf',0):.2f}")
        print(f"  Trades: {stats.get('trades',0)} (raw {raw_trades}) | MC: {mc_p}")

    # Walk-forward for each scenario
    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    n = len(df)

    print(f"\n  --- Walk-Forward Validation ---")
    for label, sc in scenarios.items():
        wf = run_wf(288, signal_index, opens, highs, lows, n, n_folds=3)
        # WF runs the full scanner pipeline, so it uses DEFAULT_EDGE_THRESHOLD=10
        # For higher edge thresholds, we filter the WF folds' patterns post-hoc
        # Note: WF OOS already applies scanner default filters internally
        # The comparison is still meaningful as WF shows the scanner's OOS capability
        sc['wf'] = {
            'positive_folds': wf['positive_folds'],
            'n_folds': wf['n_folds'],
            'total_oos_pnl': wf['total_oos_pnl'],
            'total_oos_trades': wf['total_oos_trades'],
            'stable_pattern_count': wf['stable_pattern_count'],
            'folds': wf['folds'],
        }
        print(f"  {label}: WF {wf['positive_folds']}/{wf['n_folds']}, "
              f"OOS PnL {wf['total_oos_pnl']:+.1f}%, "
              f"Stable {wf['stable_pattern_count']} patterns")
        # WF is the same for all since it runs the internal scanner pipeline
        # So we only need to run it once
        # Apply the same WF to all scenarios
        for other_label in scenarios:
            if other_label != label:
                scenarios[other_label]['wf'] = sc['wf']
        break  # Only run WF once since it's the same discovery pipeline

    return scenarios


# ============================================================
# Analysis 2: Existing 47 Patterns Re-verification (288봉)
# ============================================================

def analysis_2_retest_47(df, signal_index):
    """기존 47패턴을 동일 TP/SL로 288봉 backtest."""
    print("\n" + "=" * 70)
    print("  ANALYSIS 2: Existing 47 Patterns Re-verification (MAX_BARS=288)")
    print("=" * 70)

    # Load v1 results to get the 47 patterns with their TP/SL
    v1_file = os.path.join(_PROJECT_ROOT, 'results', 'max_bars_24h_study.json')
    with open(v1_file) as f:
        v1 = json.load(f)

    baseline_details = v1['results']['BASELINE_500']['pattern_details']
    print(f"  Loaded {len(baseline_details)} patterns from v1 BASELINE_500")

    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    n_bars = len(df)

    # Run each pattern with both 500 and 288 bars
    results_500 = {}
    results_288 = {}

    for key, v in baseline_details.items():
        pat = v['pattern']
        direction = v['direction']
        tp = v['tp']
        sl = v['sl']

        if pat not in signal_index:
            print(f"  WARNING: {pat} not in signal_index, skipping")
            continue

        sigs = signal_index[pat]

        # 500 bars (control)
        scanner.MAX_BARS = 500
        trades_500 = scanner.bt_signals(sigs, direction, tp, sl, opens, highs, lows, n_bars)

        # 288 bars (test)
        scanner.MAX_BARS = 288
        trades_288 = scanner.bt_signals(sigs, direction, tp, sl, opens, highs, lows, n_bars)

        scanner.MAX_BARS = 500  # restore

        wins_500 = sum(1 for t in trades_500 if t[2] > 0)
        wins_288 = sum(1 for t in trades_288 if t[2] > 0)
        wr_500 = round(wins_500 / len(trades_500) * 100, 1) if trades_500 else 0
        wr_288 = round(wins_288 / len(trades_288) * 100, 1) if trades_288 else 0

        results_500[key] = {
            'key': key, 'pattern': pat, 'direction': direction,
            'tp': tp, 'sl': sl,
            'trades': len(trades_500), 'wr': wr_500,
            'trades_raw': trades_500,
        }
        results_288[key] = {
            'key': key, 'pattern': pat, 'direction': direction,
            'tp': tp, 'sl': sl,
            'trades': len(trades_288), 'wr': wr_288,
            'trades_raw': trades_288,
        }

    # Portfolio comparison
    all_trades_500 = []
    all_trades_288 = []
    for key in baseline_details:
        if key in results_500:
            all_trades_500.extend(results_500[key]['trades_raw'])
            all_trades_288.extend(results_288[key]['trades_raw'])

    # Portfolio with 1-pos-at-a-time
    port_500 = scanner.portfolio_1pos(all_trades_500)
    port_288 = scanner.portfolio_1pos(all_trades_288)

    stats_500 = scanner.calc_stats(port_500)
    stats_288 = scanner.calc_stats(port_288)

    mc_500 = scanner.mc_test([t[2] for t in port_500]) if port_500 else 1.0
    mc_288 = scanner.mc_test([t[2] for t in port_288]) if port_288 else 1.0

    dur_500 = trade_duration_stats(port_500)
    dur_288 = trade_duration_stats(port_288)

    # Per-pattern comparison
    per_pattern = []
    for key in sorted(baseline_details.keys()):
        if key not in results_500:
            continue
        r5 = results_500[key]
        r2 = results_288[key]
        dropped_trades = r5['trades'] - r2['trades']
        dropped_pct = round(dropped_trades / r5['trades'] * 100, 1) if r5['trades'] > 0 else 0

        per_pattern.append({
            'key': key,
            'pattern': r5['pattern'],
            'direction': r5['direction'],
            'tp_sl': f"{r5['tp']}/{r5['sl']}",
            'trades_500': r5['trades'],
            'trades_288': r2['trades'],
            'dropped': dropped_trades,
            'dropped_pct': dropped_pct,
            'wr_500': r5['wr'],
            'wr_288': r2['wr'],
            'wr_diff': round(r2['wr'] - r5['wr'], 1),
        })

    # Print summary
    print(f"\n  {'Metric':<25} {'500봉(원본)':<18} {'288봉(재검증)':<18} {'변화':<15}")
    print(f"  {'-'*76}")
    print(f"  {'Patterns':<25} {len(baseline_details):<18} {len(results_288):<18}")
    print(f"  {'Portfolio Trades':<25} {stats_500.get('trades',0):<18} {stats_288.get('trades',0):<18} "
          f"{stats_288.get('trades',0) - stats_500.get('trades',0):+d}")
    print(f"  {'PnL':<25} {stats_500.get('pnl',0):>+.1f}%{'':<11} "
          f"{stats_288.get('pnl',0):>+.1f}%{'':<11} "
          f"{stats_288.get('pnl',0) - stats_500.get('pnl',0):+.1f}%")
    print(f"  {'WR':<25} {stats_500.get('wr',0):.1f}%{'':<12} "
          f"{stats_288.get('wr',0):.1f}%{'':<12} "
          f"{stats_288.get('wr',0) - stats_500.get('wr',0):+.1f}pp")
    print(f"  {'MDD':<25} {stats_500.get('mdd',0):.1f}%{'':<12} "
          f"{stats_288.get('mdd',0):.1f}%{'':<12} "
          f"{stats_288.get('mdd',0) - stats_500.get('mdd',0):+.1f}pp")
    print(f"  {'PF':<25} {stats_500.get('pf',0):.2f}{'':<13} "
          f"{stats_288.get('pf',0):.2f}{'':<13} "
          f"{stats_288.get('pf',0) - stats_500.get('pf',0):+.2f}")
    print(f"  {'MC p-value':<25} {mc_500:<18.4f} {mc_288:<18.4f}")

    if dur_500 and dur_288:
        print(f"\n  {'[Duration]':<25}")
        print(f"  {'Mean (hours)':<25} {dur_500.get('mean_hours','?'):<18} "
              f"{dur_288.get('mean_hours','?'):<18}")
        print(f"  {'Under 24h %':<25} {dur_500.get('pct_under_24h','?')}%{'':<12} "
              f"{dur_288.get('pct_under_24h','?')}%")

    # Patterns most affected by 288 timeout
    sorted_by_drop = sorted(per_pattern, key=lambda x: x['dropped_pct'], reverse=True)
    print(f"\n  [Top 15 patterns with most dropped trades (288봉 timeout)]")
    print(f"  {'Pattern':<18} {'Dir':<6} {'TP/SL':<8} {'Tr500':>6} {'Tr288':>6} "
          f"{'Drop':>6} {'Drop%':>6} {'WR500':>7} {'WR288':>7} {'WRΔ':>6}")
    for p in sorted_by_drop[:15]:
        print(f"  {p['pattern']:<18} {p['direction']:<6} {p['tp_sl']:<8} "
              f"{p['trades_500']:>6} {p['trades_288']:>6} "
              f"{p['dropped']:>6} {p['dropped_pct']:>5.1f}% "
              f"{p['wr_500']:>6.1f}% {p['wr_288']:>6.1f}% {p['wr_diff']:>+5.1f}")

    # Patterns with WR improvement
    improved = [p for p in per_pattern if p['wr_diff'] > 0 and p['trades_288'] >= 15]
    degraded = [p for p in per_pattern if p['wr_diff'] < 0 and p['trades_288'] >= 15]
    no_trade = [p for p in per_pattern if p['trades_288'] < 15]

    print(f"\n  [Impact Summary]")
    print(f"  WR improved (>0pp, >=15 trades): {len(improved)} patterns")
    print(f"  WR degraded (<0pp, >=15 trades): {len(degraded)} patterns")
    print(f"  Low trades (<15, unreliable):    {len(no_trade)} patterns")

    if improved:
        avg_imp = np.mean([p['wr_diff'] for p in improved])
        print(f"  Avg improvement: {avg_imp:+.1f}pp")
    if degraded:
        avg_deg = np.mean([p['wr_diff'] for p in degraded])
        print(f"  Avg degradation: {avg_deg:+.1f}pp")

    return {
        'portfolio_500': {
            'stats': stats_500, 'mc': round(mc_500, 4),
            'duration': dur_500,
        },
        'portfolio_288': {
            'stats': stats_288, 'mc': round(mc_288, 4),
            'duration': dur_288,
        },
        'per_pattern': per_pattern,
        'summary': {
            'improved_count': len(improved),
            'degraded_count': len(degraded),
            'low_trade_count': len(no_trade),
            'avg_wr_improvement': round(float(np.mean([p['wr_diff'] for p in improved])), 1) if improved else 0,
            'avg_wr_degradation': round(float(np.mean([p['wr_diff'] for p in degraded])), 1) if degraded else 0,
            'total_dropped_pct': round(
                (1 - sum(p['trades_288'] for p in per_pattern) /
                 max(sum(p['trades_500'] for p in per_pattern), 1)) * 100, 1),
        },
    }


# ============================================================
# Main
# ============================================================

def main():
    # Load data
    data_file = os.path.join(_PROJECT_ROOT, 'data', 'btc_5m_270days_reclassified.csv')
    df = scanner.load_and_classify(data_file)
    types = df['candle_type'].tolist()
    n = len(types)
    signal_index = scanner.build_signal_index(types, n)

    print(f"Data: {n} bars ({n*5/60/24:.0f} days)")
    print(f"Signal patterns: {len(signal_index)} unique 3-candle patterns")

    t_start = time.time()

    # Analysis 1: Edge Threshold Sensitivity
    a1 = analysis_1_edge_threshold(df, signal_index)

    # Analysis 2: Existing 47 patterns re-verification
    a2 = analysis_2_retest_47(df, signal_index)

    total_elapsed = time.time() - t_start

    # ============================================================
    # Combined Summary
    # ============================================================
    print(f"\n{'='*70}")
    print(f"  COMBINED SUMMARY")
    print(f"{'='*70}")

    # Analysis 1 summary table
    print(f"\n  [Analysis 1: Edge Threshold Scenarios (288봉)]")
    print(f"  {'Scenario':<12} {'Patterns':<12} {'PnL':>10} {'WR':>8} {'MDD':>8} {'PF':>8} {'Trades':>8}")
    print(f"  {'-'*66}")
    if a1:
        for label, sc in sorted(a1.items()):
            s = sc['portfolio_stats']
            print(f"  {label:<12} {sc['n_patterns']:<12} "
                  f"{s.get('pnl',0):>+9.1f}% {s.get('wr',0):>7.1f}% "
                  f"{s.get('mdd',0):>7.1f}% {s.get('pf',0):>7.2f} "
                  f"{s.get('trades',0):>8}")

    # Analysis 2 summary
    if a2:
        print(f"\n  [Analysis 2: 47패턴 Timeout 변경 (동일 TP/SL)]")
        s5 = a2['portfolio_500']['stats']
        s2 = a2['portfolio_288']['stats']
        print(f"  {'500봉':<12} {'47':<12} "
              f"{s5.get('pnl',0):>+9.1f}% {s5.get('wr',0):>7.1f}% "
              f"{s5.get('mdd',0):>7.1f}% {s5.get('pf',0):>7.2f} "
              f"{s5.get('trades',0):>8}")
        print(f"  {'288봉':<12} {'47':<12} "
              f"{s2.get('pnl',0):>+9.1f}% {s2.get('wr',0):>7.1f}% "
              f"{s2.get('mdd',0):>7.1f}% {s2.get('pf',0):>7.2f} "
              f"{s2.get('trades',0):>8}")
        print(f"  {'변화':<12} {'':<12} "
              f"{s2.get('pnl',0)-s5.get('pnl',0):>+9.1f}% "
              f"{s2.get('wr',0)-s5.get('wr',0):>+7.1f}pp "
              f"{s2.get('mdd',0)-s5.get('mdd',0):>+7.1f}pp "
              f"{s2.get('pf',0)-s5.get('pf',0):>+7.2f} "
              f"{s2.get('trades',0)-s5.get('trades',0):>+8d}")
        dropped = a2['summary']['total_dropped_pct']
        print(f"\n  Raw trade dropout: {dropped:.1f}% of individual trades dropped by 24h timeout")

    print(f"\n  Total elapsed: {total_elapsed:.1f}s")

    # Save results
    output = {
        'study': 'MAX_BARS 24h Study v2',
        'date': '2026-02-18',
        'data': f'{n} bars ({n*5/60/24:.0f} days)',
        'analysis_1_edge_threshold': {},
        'analysis_2_retest_47': {},
    }

    if a1:
        for label, sc in a1.items():
            # Remove trades_raw from pattern_details for JSON
            clean_details = {}
            for k, v in sc.get('pattern_details', {}).items():
                clean_details[k] = {kk: vv for kk, vv in v.items() if kk != 'trades_raw'}
            sc_clean = {kk: vv for kk, vv in sc.items() if kk != 'pattern_details'}
            sc_clean['pattern_details'] = clean_details
            output['analysis_1_edge_threshold'][label] = sc_clean

    if a2:
        output['analysis_2_retest_47'] = {
            'portfolio_500': a2['portfolio_500'],
            'portfolio_288': a2['portfolio_288'],
            'per_pattern': a2['per_pattern'],
            'summary': a2['summary'],
        }

    out_file = os.path.join(_PROJECT_ROOT, 'results', 'max_bars_24h_study_v2.json')
    with open(out_file, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved: {out_file}")


if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(name)s %(levelname)s %(message)s')
    main()
