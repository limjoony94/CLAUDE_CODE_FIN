#!/usr/bin/env python3
"""
MAX_BARS 24h Study — 타임아웃을 500봉(41.7h) → 288봉(24h)으로 줄이면?

연구 목적:
  24시간 내 TP/SL 미달성 트레이드를 DROP하면 패턴 발굴 결과가 어떻게 변하는지 확인.
  현재 MAX_BARS=500과 MAX_BARS=288을 동일 조건으로 비교.

비교 항목:
  1. 발견 패턴 수 (필터 전/후)
  2. 포트폴리오 성과 (PnL, WR, MDD, PF)
  3. 패턴 중복률 (500봉에서도 발견되는 패턴 비율)
  4. 트레이드 수 변화 (timeout으로 DROP되는 비율)
  5. 3-fold Expanding Window WF 검증
  6. Edge>=21.8pp + WR>=60% + SL>=1.0% 후처리 필터 적용

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

# Project root setup
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
sys.path.insert(0, _PROJECT_ROOT)

import scripts.scanner.pattern_scanner as scanner


def run_discovery(max_bars, df, signal_index, label):
    """Run PP discovery with given MAX_BARS value."""
    original = scanner.MAX_BARS
    scanner.MAX_BARS = max_bars
    try:
        print(f"\n{'='*60}")
        print(f"  {label}: MAX_BARS={max_bars} ({max_bars*5/60:.1f}h)")
        print(f"{'='*60}")

        t0 = time.time()
        result = scanner.scan_patterns_pp(
            df,
            edge_threshold=scanner.DEFAULT_EDGE_THRESHOLD,
            mc_threshold=scanner.DEFAULT_MC_THRESHOLD,
            min_trades=scanner.DEFAULT_MIN_TRADES,
            signal_index=signal_index,
            concurrency=1,  # Sequential for reproducibility
            max_baseline_wr=scanner.MAX_BASELINE_WR,
            correction_method='bh',
            fdr_q=0.05,
        )
        elapsed = time.time() - t0
        print(f"  Elapsed: {elapsed:.1f}s")
        return result
    finally:
        scanner.MAX_BARS = original


def apply_post_filters(result, edge_min=21.8, wr_min=60.0, sl_min=1.0):
    """Apply Edge>=21.8pp + WR>=60% + SL>=1.0% post-filter."""
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


def portfolio_from_filtered(filtered_details, signal_index, df, max_bars):
    """Build portfolio from filtered patterns and compute stats."""
    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    n_bars = len(df)

    original = scanner.MAX_BARS
    scanner.MAX_BARS = max_bars
    try:
        all_trades = []
        for key, v in filtered_details.items():
            sigs = signal_index[v['pattern']]
            trades = scanner.bt_signals(
                sigs, v['direction'], v['tp'], v['sl'],
                opens, highs, lows, n_bars
            )
            all_trades.extend(trades)

        port = scanner.portfolio_1pos(all_trades)
        stats = scanner.calc_stats(port)
        mc_p = scanner.mc_test([t[2] for t in port]) if port else 1.0
        return stats, mc_p, len(all_trades), port
    finally:
        scanner.MAX_BARS = original


def trade_duration_stats(port_trades):
    """Calculate trade duration statistics in bars."""
    if not port_trades:
        return {}
    durations = [t[1] - t[0] for t in port_trades]
    import numpy as np
    durations = np.array(durations)
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


def main():
    # Load data
    data_file = os.path.join(_PROJECT_ROOT, 'data', 'btc_5m_270days_reclassified.csv')
    df = scanner.load_and_classify(data_file)
    types = df['candle_type'].tolist()
    n = len(types)
    signal_index = scanner.build_signal_index(types, n)
    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values

    print(f"Data: {n} bars ({n*5/60/24:.0f} days)")
    print(f"Signal patterns: {len(signal_index)} unique 3-candle patterns")

    # ============================================================
    # Phase 1: Full discovery comparison
    # ============================================================
    results = {}
    for label, max_bars in [('BASELINE_500', 500), ('TEST_288', 288)]:
        result = run_discovery(max_bars, df, signal_index, label)

        # Pre-filter stats
        pre_n = len(result.get('pattern_details', {}))
        pre_long = len(result.get('long_patterns', []))
        pre_short = len(result.get('short_patterns', []))

        # Post-filter
        pf = apply_post_filters(result)

        # Portfolio from post-filtered patterns
        if pf['filtered_details']:
            stats, mc_p, raw_trades, port = portfolio_from_filtered(
                pf['filtered_details'], signal_index, df, max_bars)
            dur = trade_duration_stats(port)
        else:
            stats = {'pnl': 0, 'trades': 0, 'wr': 0, 'mdd': 0, 'pf': 0}
            mc_p = 1.0
            raw_trades = 0
            port = []
            dur = {}

        results[label] = {
            'max_bars': max_bars,
            'hours': round(max_bars * 5 / 60, 1),
            'pre_filter': {
                'n_patterns': pre_n, 'n_long': pre_long, 'n_short': pre_short,
                'portfolio_stats': result.get('portfolio_stats', {}),
                'portfolio_mc': result.get('portfolio_mc', 1.0),
                'tp_dist': result.get('tp_distribution', {}),
                'sl_dist': result.get('sl_distribution', {}),
            },
            'post_filter': {
                'n_patterns': pf['n_total'],
                'n_long': pf['n_long'],
                'n_short': pf['n_short'],
                'n_removed': pf['n_removed'],
                'patterns': pf['long_patterns'] + pf['short_patterns'],
                'portfolio_stats': stats,
                'portfolio_mc': round(mc_p, 4),
                'raw_trades': raw_trades,
                'duration_stats': dur,
            },
            'pattern_details': {
                k: {kk: vv for kk, vv in v.items() if kk != 'trades_raw'}
                for k, v in pf['filtered_details'].items()
            },
        }

    # ============================================================
    # Phase 2: Pattern overlap analysis
    # ============================================================
    base_pats = set(results['BASELINE_500']['post_filter']['patterns'])
    test_pats = set(results['TEST_288']['post_filter']['patterns'])
    overlap = base_pats & test_pats
    only_base = base_pats - test_pats
    only_test = test_pats - base_pats

    overlap_analysis = {
        'baseline_count': len(base_pats),
        'test_count': len(test_pats),
        'overlap_count': len(overlap),
        'overlap_pct_of_baseline': round(len(overlap) / len(base_pats) * 100, 1) if base_pats else 0,
        'overlap_pct_of_test': round(len(overlap) / len(test_pats) * 100, 1) if test_pats else 0,
        'only_in_baseline': sorted(only_base),
        'only_in_test': sorted(only_test),
        'common': sorted(overlap),
    }

    # ============================================================
    # Phase 3: Walk-Forward Validation for 288
    # ============================================================
    print(f"\n{'='*60}")
    print(f"  Walk-Forward Validation (3-fold Expanding)")
    print(f"{'='*60}")

    wf_results = {}
    for label, max_bars in [('BASELINE_500', 500), ('TEST_288', 288)]:
        print(f"\n  --- {label} WF ---")
        wf = run_wf(max_bars, signal_index, opens, highs, lows, n, n_folds=3)
        wf_results[label] = wf
        for f in wf['folds']:
            status = "PASS" if f['oos_positive'] else "FAIL"
            print(f"    Fold {f['fold']}: IS {f['is_patterns']}pat | "
                  f"OOS {f['oos_trades']}tr WR {f['oos_wr']}% "
                  f"PnL {f['oos_pnl']}% [{status}]")
        print(f"    Total: {wf['positive_folds']}/{wf['n_folds']} positive, "
              f"OOS PnL {wf['total_oos_pnl']}%")

    # ============================================================
    # Phase 4: Per-pattern comparison (overlapping patterns)
    # ============================================================
    per_pattern_comparison = []
    base_details = results['BASELINE_500'].get('pattern_details', {})
    test_details = results['TEST_288'].get('pattern_details', {})
    for pat in sorted(overlap):
        # Find the pattern in details (key format: PATTERN_DIRECTION)
        b = None
        t = None
        for k, v in base_details.items():
            if v['pattern'] == pat:
                b = v
                break
        for k, v in test_details.items():
            if v['pattern'] == pat:
                t = v
                break
        if b and t:
            per_pattern_comparison.append({
                'pattern': pat,
                'direction': b['direction'],
                'base_tp_sl': f"{b['tp']}/{b['sl']}",
                'test_tp_sl': f"{t['tp']}/{t['sl']}",
                'base_trades': b['trades'],
                'test_trades': t['trades'],
                'trade_diff': t['trades'] - b['trades'],
                'base_wr': b['wr'],
                'test_wr': t['wr'],
                'wr_diff': round(t['wr'] - b['wr'], 1),
                'base_edge': b['edge'],
                'test_edge': t['edge'],
                'edge_diff': round(t['edge'] - b['edge'], 1),
                'tp_sl_changed': b['tp'] != t['tp'] or b['sl'] != t['sl'],
            })

    # ============================================================
    # Summary
    # ============================================================
    summary = {
        'study': 'MAX_BARS 24h Study',
        'date': '2026-02-18',
        'data': f'{n} bars ({n*5/60/24:.0f} days)',
        'comparison': {
            'baseline': 'MAX_BARS=500 (41.7h)',
            'test': 'MAX_BARS=288 (24.0h)',
        },
        'results': results,
        'overlap_analysis': overlap_analysis,
        'walk_forward': {k: {kk: vv for kk, vv in v.items()}
                        for k, v in wf_results.items()},
        'per_pattern_comparison': per_pattern_comparison,
    }

    # Save results
    out_file = os.path.join(_PROJECT_ROOT, 'results', 'max_bars_24h_study.json')
    with open(out_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nResults saved: {out_file}")

    # ============================================================
    # Print Summary Table
    # ============================================================
    print(f"\n{'='*70}")
    print(f"  SUMMARY: MAX_BARS=500 vs MAX_BARS=288")
    print(f"{'='*70}")

    b = results['BASELINE_500']
    t = results['TEST_288']

    print(f"\n  {'Metric':<30} {'500봉(41.7h)':<20} {'288봉(24.0h)':<20}")
    print(f"  {'-'*70}")

    # Pre-filter
    print(f"  {'[Pre-filter]':<30}")
    print(f"  {'Patterns':<30} {b['pre_filter']['n_patterns']:<20} "
          f"{t['pre_filter']['n_patterns']:<20}")
    bp = b['pre_filter']['portfolio_stats']
    tp = t['pre_filter']['portfolio_stats']
    print(f"  {'Portfolio PnL':<30} {bp.get('pnl',0):>+.1f}%{'':<13} "
          f"{tp.get('pnl',0):>+.1f}%")
    print(f"  {'Portfolio WR':<30} {bp.get('wr',0):.1f}%{'':<14} "
          f"{tp.get('wr',0):.1f}%")
    print(f"  {'Portfolio Trades':<30} {bp.get('trades',0):<20} "
          f"{tp.get('trades',0):<20}")

    # Post-filter
    print(f"\n  {'[Post-filter: E>=21.8+WR>=60+SL>=1.0]':<30}")
    print(f"  {'Patterns':<30} "
          f"{b['post_filter']['n_patterns']} ({b['post_filter']['n_long']}L+{b['post_filter']['n_short']}S){'':<5} "
          f"{t['post_filter']['n_patterns']} ({t['post_filter']['n_long']}L+{t['post_filter']['n_short']}S)")
    bs = b['post_filter']['portfolio_stats']
    ts = t['post_filter']['portfolio_stats']
    print(f"  {'PnL':<30} {bs.get('pnl',0):>+.1f}%{'':<13} "
          f"{ts.get('pnl',0):>+.1f}%")
    print(f"  {'WR':<30} {bs.get('wr',0):.1f}%{'':<14} "
          f"{ts.get('wr',0):.1f}%")
    print(f"  {'MDD':<30} {bs.get('mdd',0):.1f}%{'':<14} "
          f"{ts.get('mdd',0):.1f}%")
    print(f"  {'PF':<30} {bs.get('pf',0):.2f}{'':<15} "
          f"{ts.get('pf',0):.2f}")
    print(f"  {'Trades':<30} {bs.get('trades',0):<20} "
          f"{ts.get('trades',0):<20}")
    print(f"  {'MC p-value':<30} {b['post_filter']['portfolio_mc']:<20} "
          f"{t['post_filter']['portfolio_mc']:<20}")

    # Duration stats
    bd = b['post_filter'].get('duration_stats', {})
    td = t['post_filter'].get('duration_stats', {})
    if bd:
        print(f"\n  {'[Trade Duration]':<30}")
        print(f"  {'Mean (hours)':<30} {bd.get('mean_hours','?'):<20} "
              f"{td.get('mean_hours','?'):<20}")
        print(f"  {'Median (hours)':<30} {bd.get('median_hours','?'):<20} "
              f"{td.get('median_hours','?'):<20}")
        print(f"  {'Under 24h %':<30} {bd.get('pct_under_24h','?')}%{'':<14} "
              f"{td.get('pct_under_24h','?')}%")

    # Overlap
    oa = overlap_analysis
    print(f"\n  {'[Pattern Overlap]':<30}")
    print(f"  {'Common patterns':<30} {oa['overlap_count']} / {oa['baseline_count']} baseline "
          f"({oa['overlap_pct_of_baseline']}%)")
    if oa['only_in_baseline']:
        print(f"  {'Only in 500봉':<30} {', '.join(oa['only_in_baseline'][:10])}")
    if oa['only_in_test']:
        print(f"  {'Only in 288봉':<30} {', '.join(oa['only_in_test'][:10])}")

    # WF
    print(f"\n  {'[Walk-Forward 3-fold]':<30}")
    bwf = wf_results['BASELINE_500']
    twf = wf_results['TEST_288']
    print(f"  {'Positive folds':<30} {bwf['positive_folds']}/{bwf['n_folds']}{'':<14} "
          f"{twf['positive_folds']}/{twf['n_folds']}")
    print(f"  {'Total OOS PnL':<30} {bwf['total_oos_pnl']:>+.1f}%{'':<13} "
          f"{twf['total_oos_pnl']:>+.1f}%")
    print(f"  {'Total OOS Trades':<30} {bwf['total_oos_trades']:<20} "
          f"{twf['total_oos_trades']:<20}")
    print(f"  {'Stable patterns':<30} {bwf['stable_pattern_count']:<20} "
          f"{twf['stable_pattern_count']:<20}")

    # TP/SL changes for overlapping patterns
    if per_pattern_comparison:
        changed = [p for p in per_pattern_comparison if p['tp_sl_changed']]
        print(f"\n  {'[Overlapping Pattern TP/SL Changes]':<30}")
        print(f"  {len(overlap)} common patterns, {len(changed)} with different TP/SL")
        if changed:
            for p in changed[:10]:
                print(f"    {p['pattern']:<12} {p['direction']:<6} "
                      f"TP/SL {p['base_tp_sl']} → {p['test_tp_sl']} "
                      f"| WR {p['base_wr']}→{p['test_wr']}% "
                      f"| Trades {p['base_trades']}→{p['test_trades']}")

    # Conclusion
    pnl_diff = ts.get('pnl', 0) - bs.get('pnl', 0)
    mdd_diff = ts.get('mdd', 0) - bs.get('mdd', 0)
    pat_diff = t['post_filter']['n_patterns'] - b['post_filter']['n_patterns']

    print(f"\n  {'='*70}")
    print(f"  CONCLUSION:")
    print(f"  24h timeout vs 41.7h timeout:")
    print(f"    Patterns: {pat_diff:+d} ({b['post_filter']['n_patterns']} → {t['post_filter']['n_patterns']})")
    print(f"    PnL:      {pnl_diff:+.1f}% ({bs.get('pnl',0):+.1f} → {ts.get('pnl',0):+.1f})")
    print(f"    MDD:      {mdd_diff:+.1f}% ({bs.get('mdd',0):.1f} → {ts.get('mdd',0):.1f})")
    wf_status = "PASS" if twf['positive_folds'] >= 2 else "FAIL"
    print(f"    WF:       {twf['positive_folds']}/{twf['n_folds']} [{wf_status}]")
    print(f"  {'='*70}")


if __name__ == '__main__':
    import logging
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(name)s %(levelname)s %(message)s')
    main()
