#!/usr/bin/env python3
"""
Strategy Improvement Study — v1.28.24 기반 4대 개선 방향 연구

종합 평가에서 도출된 주의점 기반:
  Study 1: WF-Stable 패턴만 사용 (57→? stable patterns)
  Study 2: Direction Balance (SHORT 80% 편향 완화)
  Study 3: Timeout Trade Behavior (DROP vs HOLD 괴리 정량화)
  Study 4: Edge Threshold 재최적화 (288봉 기준)

Protocol:
  - Production classify_candle() (indicators.py) via scanner
  - LEVERAGE = 3 (scanner 내장)
  - Timeout trades DROP (scanner 내장)
  - Expanding Window WF (3-fold)
  - Multi-seed MC test
  - Compound returns for MDD
"""

import os
import sys
import json
import time
import numpy as np
from collections import Counter

# Project root setup
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
sys.path.insert(0, _PROJECT_ROOT)

import scripts.scanner.pattern_scanner as scanner

RESULT_FILE = os.path.join(_PROJECT_ROOT, 'results', 'strategy_improvement_study.json')


# ============================================================
# Shared Utilities
# ============================================================

def portfolio_backtest(details, signal_index, df, max_bars):
    """Run portfolio backtest for given pattern details."""
    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    n_bars = len(df)

    original = scanner.MAX_BARS
    scanner.MAX_BARS = max_bars
    try:
        all_trades = []
        for key, v in details.items():
            pat = v['pattern']
            if pat not in signal_index:
                continue
            sigs = signal_index[pat]
            trades = scanner.bt_signals(
                sigs, v['direction'], v['tp'], v['sl'],
                opens, highs, lows, n_bars
            )
            all_trades.extend(trades)

        port = scanner.portfolio_1pos(all_trades)
        stats = scanner.calc_stats(port)

        # MC test (scanner.mc_test already uses multi-seed internally)
        pnls = [t[2] for t in port]
        mc_p = scanner.mc_test(pnls) if pnls else 1.0

        # Duration stats
        dur = {}
        if port:
            durations = np.array([t[1] - t[0] for t in port])
            dur = {
                'mean_bars': round(float(durations.mean()), 1),
                'median_bars': round(float(np.median(durations)), 1),
                'mean_hours': round(float(durations.mean() * 5 / 60), 1),
                'pct_under_24h': round(float((durations <= 288).sum() / len(durations) * 100), 1),
                'pct_under_12h': round(float((durations <= 144).sum() / len(durations) * 100), 1),
            }

        return {
            'stats': stats,
            'mc_p': round(mc_p, 4),
            'raw_trades': len(all_trades),
            'port_trades': len(port),
            'duration': dur,
            'port': port,  # for further analysis (not saved to JSON)
        }
    finally:
        scanner.MAX_BARS = original


def run_expanding_wf(signal_index, df, max_bars, n_folds=3,
                     edge_threshold=None, post_filter_fn=None):
    """Run expanding window WF, optionally with post-filter on IS patterns."""
    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    n_bars = len(df)

    original = scanner.MAX_BARS
    scanner.MAX_BARS = max_bars
    try:
        if post_filter_fn is None:
            # Standard WF
            wf = scanner.expanding_window_wf(
                signal_index, opens, highs, lows, n_bars,
                n_folds=n_folds, mode='per_pattern',
                min_trades=scanner.DEFAULT_MIN_TRADES,
                edge_threshold=edge_threshold or scanner.DEFAULT_EDGE_THRESHOLD,
                mc_threshold=scanner.DEFAULT_MC_THRESHOLD,
                max_baseline_wr=scanner.MAX_BASELINE_WR,
            )
            return wf
        else:
            # Custom WF with post-filter applied to IS patterns
            return custom_wf_with_filter(
                signal_index, opens, highs, lows, n_bars,
                n_folds, edge_threshold or scanner.DEFAULT_EDGE_THRESHOLD,
                post_filter_fn
            )
    finally:
        scanner.MAX_BARS = original


def custom_wf_with_filter(signal_index, opens, highs, lows, n_bars,
                          n_folds, edge_threshold, post_filter_fn):
    """WF where IS patterns are further filtered by post_filter_fn."""
    seg_size = n_bars // (n_folds + 1)
    folds = []
    pattern_appearances = Counter()

    for fold in range(n_folds):
        is_end = (fold + 1) * seg_size
        oos_start = is_end
        oos_end = (fold + 2) * seg_size if fold < n_folds - 1 else n_bars

        # Fresh scan on IS range
        is_patterns = scanner.scan_universe_range(
            signal_index, opens, highs, lows, n_bars,
            bar_start=0, bar_end=is_end, mode='per_pattern',
            min_trades=scanner.DEFAULT_MIN_TRADES,
            edge_threshold=edge_threshold,
            mc_threshold=scanner.DEFAULT_MC_THRESHOLD,
            max_baseline_wr=scanner.MAX_BASELINE_WR,
        )

        # Apply post-filter
        is_patterns = post_filter_fn(is_patterns)

        for r in is_patterns:
            pattern_appearances[(r['pattern'], r['direction'])] += 1

        # OOS backtest
        oos_trades = []
        for r in is_patterns:
            oos_sigs = [s for s in signal_index[r['pattern']]
                        if oos_start <= s < oos_end]
            oos_trades.extend(scanner.bt_signals(
                oos_sigs, r['direction'], r['tp'], r['sl'],
                opens, highs, lows, oos_end
            ))

        oos_port = scanner.portfolio_1pos(oos_trades)
        oos_stats = scanner.calc_stats(oos_port)
        n_long = sum(1 for r in is_patterns if r['direction'] == 'LONG')
        n_short = sum(1 for r in is_patterns if r['direction'] == 'SHORT')

        folds.append({
            'fold': fold + 1,
            'is_bars': is_end,
            'oos_bars': oos_end - oos_start,
            'is_patterns': len(is_patterns),
            'is_long': n_long, 'is_short': n_short,
            'oos_trades': oos_stats['trades'],
            'oos_wr': oos_stats['wr'],
            'oos_pnl': oos_stats['pnl'],
            'oos_mdd': oos_stats['mdd'],
            'oos_pf': oos_stats['pf'],
            'oos_positive': oos_stats['pnl'] > 0,
        })

    stable = sorted(
        f"{pat}_{d}"
        for (pat, d), cnt in pattern_appearances.items()
        if cnt == n_folds
    )
    positive_folds = sum(1 for f in folds if f['oos_positive'])
    total_oos_pnl = round(sum(f['oos_pnl'] for f in folds), 1)
    total_oos_trades = sum(f['oos_trades'] for f in folds)

    return {
        'n_folds': n_folds,
        'folds': folds,
        'positive_folds': positive_folds,
        'total_oos_pnl': total_oos_pnl,
        'total_oos_trades': total_oos_trades,
        'stable_pattern_count': len(stable),
        'stable_patterns': stable,
    }


def print_scenario(name, stats, mc_p, n_pat, n_long, n_short, raw_trades=None, dur=None):
    """Pretty-print scenario results."""
    print(f"\n  [{name}]")
    print(f"  Patterns: {n_pat} ({n_long}L + {n_short}S)")
    print(f"  PnL: {stats.get('pnl',0):+.1f}% | WR: {stats.get('wr',0):.1f}% | "
          f"MDD: {stats.get('mdd',0):.1f}% | PF: {stats.get('pf',0):.2f}")
    pnl_mdd = stats['pnl'] / stats['mdd'] if stats.get('mdd', 0) > 0 else 0
    print(f"  PnL/MDD: {pnl_mdd:.1f}x | Trades: {stats.get('trades',0)} | MC: {mc_p}")
    if raw_trades:
        print(f"  Raw trades (all): {raw_trades}")
    if dur:
        print(f"  Duration: mean {dur.get('mean_hours',0):.1f}h | "
              f"<24h: {dur.get('pct_under_24h',0):.0f}% | <12h: {dur.get('pct_under_12h',0):.0f}%")


def print_wf(wf, name=""):
    """Pretty-print WF results."""
    label = f" ({name})" if name else ""
    print(f"  WF{label}: {wf['positive_folds']}/{wf['n_folds']} positive, "
          f"OOS PnL {wf['total_oos_pnl']:+.1f}%, "
          f"{wf['total_oos_trades']} trades, "
          f"Stable: {wf['stable_pattern_count']}")
    for f in wf['folds']:
        print(f"    Fold {f['fold']}: WR={f['oos_wr']:.1f}%, PnL={f['oos_pnl']:+.1f}%, "
              f"MDD={f['oos_mdd']:.1f}%, PF={f['oos_pf']:.2f}, Trades={f['oos_trades']}")


# ============================================================
# Study 1: WF-Stable Pattern Subset
# ============================================================

def study_1_wf_stable(df, signal_index, all_details):
    """WF-Stable 패턴만으로 포트폴리오 구성 vs 전체 112.

    가설: Stable subset은 MDD 감소 + OOS 일관성 증가.
    리스크: 패턴 수 감소 → PnL 하락 가능.
    """
    print("\n" + "=" * 70)
    print("  STUDY 1: WF-Stable Pattern Subset")
    print("  가설: WF-Stable 패턴만 사용하면 OOS 안정성 증가")
    print("=" * 70)

    # Step 1: Run WF to identify stable patterns
    print("\n  Step 1: Expanding WF (3-fold) for stable pattern identification...")
    t0 = time.time()
    wf_full = run_expanding_wf(signal_index, df, 288, n_folds=3,
                               edge_threshold=scanner.DEFAULT_EDGE_THRESHOLD)
    elapsed = time.time() - t0
    print(f"  WF done in {elapsed:.1f}s")
    print_wf(wf_full, "Full 288-bar")

    stable_keys = set(wf_full.get('stable_patterns', []))
    print(f"\n  Stable patterns: {len(stable_keys)}")

    # Step 2: Filter all_details to stable only
    stable_details = {k: v for k, v in all_details.items() if k in stable_keys}
    unstable_details = {k: v for k, v in all_details.items() if k not in stable_keys}

    n_stable_long = sum(1 for v in stable_details.values() if v['direction'] == 'LONG')
    n_stable_short = sum(1 for v in stable_details.values() if v['direction'] == 'SHORT')
    n_unstable_long = sum(1 for v in unstable_details.values() if v['direction'] == 'LONG')
    n_unstable_short = sum(1 for v in unstable_details.values() if v['direction'] == 'SHORT')

    # Step 3: Backtest both
    print("\n  Step 2: Portfolio backtests...")
    bt_full = portfolio_backtest(all_details, signal_index, df, 288)
    bt_stable = portfolio_backtest(stable_details, signal_index, df, 288)
    bt_unstable = portfolio_backtest(unstable_details, signal_index, df, 288)

    print_scenario("A: Full 112", bt_full['stats'], bt_full['mc_p'],
                   len(all_details),
                   sum(1 for v in all_details.values() if v['direction'] == 'LONG'),
                   sum(1 for v in all_details.values() if v['direction'] == 'SHORT'),
                   bt_full['raw_trades'], bt_full['duration'])

    print_scenario("B: Stable Only", bt_stable['stats'], bt_stable['mc_p'],
                   len(stable_details), n_stable_long, n_stable_short,
                   bt_stable['raw_trades'], bt_stable['duration'])

    print_scenario("C: Unstable Only", bt_unstable['stats'], bt_unstable['mc_p'],
                   len(unstable_details), n_unstable_long, n_unstable_short,
                   bt_unstable['raw_trades'], bt_unstable['duration'])

    # Step 4: WF for stable-only (using post-filter that keeps only edge>=21.8)
    print("\n  Step 3: WF for Stable-Only subset...")
    # WF with edge >= 21.8 post-filter
    def edge_21_8_filter(is_patterns):
        return [p for p in is_patterns if p.get('edge', 0) >= 21.8]

    wf_stable_filtered = run_expanding_wf(
        signal_index, df, 288, n_folds=3,
        edge_threshold=scanner.DEFAULT_EDGE_THRESHOLD,
        post_filter_fn=edge_21_8_filter
    )
    print_wf(wf_stable_filtered, "Edge>=21.8 post-filter")

    result = {
        'full_112': {
            'n_patterns': len(all_details),
            'n_long': sum(1 for v in all_details.values() if v['direction'] == 'LONG'),
            'n_short': sum(1 for v in all_details.values() if v['direction'] == 'SHORT'),
            'stats': bt_full['stats'],
            'mc_p': bt_full['mc_p'],
            'duration': bt_full['duration'],
        },
        'stable_only': {
            'n_patterns': len(stable_details),
            'n_long': n_stable_long, 'n_short': n_stable_short,
            'stats': bt_stable['stats'],
            'mc_p': bt_stable['mc_p'],
            'duration': bt_stable['duration'],
            'pattern_keys': sorted(stable_keys),
        },
        'unstable_only': {
            'n_patterns': len(unstable_details),
            'n_long': n_unstable_long, 'n_short': n_unstable_short,
            'stats': bt_unstable['stats'],
            'mc_p': bt_unstable['mc_p'],
            'duration': bt_unstable['duration'],
        },
        'wf_full': {k: v for k, v in wf_full.items() if k != 'stable_patterns'},
        'wf_stable_filtered': {k: v for k, v in wf_stable_filtered.items() if k != 'stable_patterns'},
        'wf_stable_filtered_stable_patterns': wf_stable_filtered.get('stable_patterns', []),
    }

    return result


# ============================================================
# Study 2: Direction Balance
# ============================================================

def study_2_direction_balance(df, signal_index, all_details):
    """SHORT 편향 완화 시나리오 비교.

    현재: 22L + 90S (80% SHORT)
    시나리오:
      A: 전체 112 (baseline)
      B: SHORT 상위 N개만 (LONG 전체 유지, SHORT를 edge 기준으로 정렬 후 top-N)
      C: LONG:SHORT = 1:2 비율 (22L + 44S)
      D: LONG:SHORT = 1:3 비율 (22L + 66S)
      E: LONG only (22L)
      F: SHORT only (90S)
    """
    print("\n" + "=" * 70)
    print("  STUDY 2: Direction Balance")
    print("  가설: SHORT 편향 완화로 BTC 상승장 robustness 개선")
    print("=" * 70)

    long_details = {k: v for k, v in all_details.items() if v['direction'] == 'LONG'}
    short_details = {k: v for k, v in all_details.items() if v['direction'] == 'SHORT'}

    # Sort SHORT by edge (descending)
    short_sorted = sorted(short_details.items(), key=lambda x: x[1]['edge'], reverse=True)

    scenarios = {}

    # A: Full 112 (baseline)
    scenarios['A_Full_112'] = {
        'details': all_details,
        'desc': 'Full 112 (22L+90S)',
    }

    # B: LONG all + SHORT top 44 (1:2 ratio)
    top_44_short = dict(short_sorted[:44])
    b_details = {**long_details, **top_44_short}
    scenarios['B_Ratio_1to2'] = {
        'details': b_details,
        'desc': f'1:2 ratio ({len(long_details)}L+44S)',
    }

    # C: LONG all + SHORT top 66 (1:3 ratio)
    top_66_short = dict(short_sorted[:66])
    c_details = {**long_details, **top_66_short}
    scenarios['C_Ratio_1to3'] = {
        'details': c_details,
        'desc': f'1:3 ratio ({len(long_details)}L+66S)',
    }

    # D: LONG all + SHORT top 22 (1:1 ratio)
    top_22_short = dict(short_sorted[:22])
    d_details = {**long_details, **top_22_short}
    scenarios['D_Ratio_1to1'] = {
        'details': d_details,
        'desc': f'1:1 ratio ({len(long_details)}L+22S)',
    }

    # E: LONG only
    scenarios['E_Long_Only'] = {
        'details': long_details,
        'desc': f'LONG only ({len(long_details)}L)',
    }

    # F: SHORT only
    scenarios['F_Short_Only'] = {
        'details': short_details,
        'desc': f'SHORT only ({len(short_details)}S)',
    }

    results = {}
    for label, sc in scenarios.items():
        d = sc['details']
        bt = portfolio_backtest(d, signal_index, df, 288)
        n_l = sum(1 for v in d.values() if v['direction'] == 'LONG')
        n_s = sum(1 for v in d.values() if v['direction'] == 'SHORT')
        print_scenario(f"{label}: {sc['desc']}", bt['stats'], bt['mc_p'],
                       len(d), n_l, n_s, bt['raw_trades'], bt['duration'])

        results[label] = {
            'description': sc['desc'],
            'n_patterns': len(d),
            'n_long': n_l, 'n_short': n_s,
            'stats': bt['stats'],
            'mc_p': bt['mc_p'],
            'duration': bt['duration'],
        }

    # WF for top 2 non-baseline scenarios (by PnL/MDD)
    print("\n  WF validation for top scenarios...")
    ranked = sorted(
        [(k, v) for k, v in results.items() if k != 'A_Full_112'],
        key=lambda x: (x[1]['stats']['pnl'] / x[1]['stats']['mdd']
                        if x[1]['stats']['mdd'] > 0 else 0),
        reverse=True
    )

    for label, _ in ranked[:2]:
        sc = scenarios[label]
        d = sc['details']
        # Create a filter function that keeps only the patterns in this scenario
        pat_keys = set()
        for k, v in d.items():
            pat_keys.add((v['pattern'], v['direction']))

        def make_filter(keys):
            def f(is_patterns):
                return [p for p in is_patterns
                        if (p['pattern'], p['direction']) in keys
                        and p.get('edge', 0) >= 21.8]
            return f

        wf = run_expanding_wf(
            signal_index, df, 288, n_folds=3,
            edge_threshold=scanner.DEFAULT_EDGE_THRESHOLD,
            post_filter_fn=make_filter(pat_keys)
        )
        print_wf(wf, label)
        results[label]['wf'] = {k: v for k, v in wf.items() if k != 'stable_patterns'}

    return results


# ============================================================
# Study 3: Timeout Trade Behavior
# ============================================================

def study_3_timeout_behavior(df, signal_index, all_details):
    """DROP 메커니즘의 실제 영향 정량화.

    Scanner: 288봉 내 미결 → DROP
    Production: TP/SL까지 무한 보유

    분석:
      a) 288봉 timeout 거래 수 (패턴별)
      b) Timeout 거래를 500/1000봉까지 연장 시 TP/SL 도달률
      c) Timeout 거래의 시장가 청산 PnL 분포
      d) DROP vs HOLD-to-resolution 포트폴리오 비교
    """
    print("\n" + "=" * 70)
    print("  STUDY 3: Timeout Trade Behavior Analysis")
    print("  Scanner DROP vs Production HOLD 괴리 정량화")
    print("=" * 70)

    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    n_bars = len(df)

    # For each pattern: run at 288 bars AND 2000 bars (essentially unlimited)
    per_pattern_analysis = {}
    total_288_trades = 0
    total_2000_trades = 0
    all_trades_288 = []
    all_trades_2000 = []
    timeout_pnls = []  # PnL at bar 288 for timeout trades

    for key, v in all_details.items():
        pat = v['pattern']
        if pat not in signal_index:
            continue
        sigs = signal_index[pat]
        direction = v['direction']
        tp = v['tp']
        sl = v['sl']

        # 288 bars (standard)
        scanner.MAX_BARS = 288
        trades_288 = scanner.bt_signals(sigs, direction, tp, sl, opens, highs, lows, n_bars)

        # 2000 bars (essentially unlimited — ~7 days)
        scanner.MAX_BARS = 2000
        trades_2000 = scanner.bt_signals(sigs, direction, tp, sl, opens, highs, lows, n_bars)

        scanner.MAX_BARS = 288  # restore

        all_trades_288.extend(trades_288)
        all_trades_2000.extend(trades_2000)
        total_288_trades += len(trades_288)
        total_2000_trades += len(trades_2000)

        # Find timeout trades: in 2000-bar but NOT in 288-bar
        # Trades are (entry_bar, exit_bar, pnl), matched by entry_bar
        entry_bars_288 = set(t[0] for t in trades_288)
        entry_bars_2000 = set(t[0] for t in trades_2000)

        # Trades that exist in 2000 but were dropped in 288
        timeout_entries = entry_bars_2000 - entry_bars_288
        # Also: trades that exist in both but with different exit (shorter in 288)
        # Actually, 288 just drops trades that don't resolve in 288 bars
        # So timeout = entries in 2000 with duration > 288

        timeout_count = 0
        resolved_within_288 = 0
        resolved_289_to_500 = 0
        resolved_501_to_1000 = 0
        resolved_1001_plus = 0
        timeout_trade_pnls = []

        for t in trades_2000:
            duration = t[1] - t[0]
            if duration <= 288:
                resolved_within_288 += 1
            else:
                timeout_count += 1
                if duration <= 500:
                    resolved_289_to_500 += 1
                elif duration <= 1000:
                    resolved_501_to_1000 += 1
                else:
                    resolved_1001_plus += 1
                timeout_trade_pnls.append(t[2])

                # Calculate PnL at bar 288 (what forced close would give)
                entry_bar = t[0]
                bar_288 = entry_bar + 288
                if bar_288 < n_bars:
                    price_at_288 = opens[bar_288]  # use open of bar 288
                    entry_price = opens[entry_bar + 1]  # next bar open (entry)
                    if direction == 'LONG':
                        forced_pnl = (price_at_288 / entry_price - 1) * 100 * scanner.LEVERAGE
                    else:
                        forced_pnl = (1 - price_at_288 / entry_price) * 100 * scanner.LEVERAGE
                    forced_pnl -= 0.10  # fee
                    timeout_pnls.append(forced_pnl)

        wins_timeout = sum(1 for p in timeout_trade_pnls if p > 0)
        wr_timeout = round(wins_timeout / timeout_count * 100, 1) if timeout_count > 0 else 0

        per_pattern_analysis[key] = {
            'pattern': pat, 'direction': direction,
            'tp': tp, 'sl': sl,
            'trades_288': len(trades_288),
            'trades_2000': len(trades_2000),
            'timeout_count': timeout_count,
            'timeout_pct': round(timeout_count / len(trades_2000) * 100, 1) if trades_2000 else 0,
            'timeout_wr': wr_timeout,
            'resolved_289_500': resolved_289_to_500,
            'resolved_501_1000': resolved_501_to_1000,
            'resolved_1001_plus': resolved_1001_plus,
        }

    # Portfolio comparison: 288 vs 2000 (unlimited)
    port_288 = scanner.portfolio_1pos(all_trades_288)
    port_2000 = scanner.portfolio_1pos(all_trades_2000)
    stats_288 = scanner.calc_stats(port_288)
    stats_2000 = scanner.calc_stats(port_2000)

    # Timeout trade aggregate stats
    total_timeout = sum(v['timeout_count'] for v in per_pattern_analysis.values())
    total_resolved = sum(v['trades_2000'] for v in per_pattern_analysis.values())
    avg_timeout_pct = round(total_timeout / total_resolved * 100, 1) if total_resolved > 0 else 0

    print(f"\n  --- Aggregate Statistics ---")
    print(f"  Total trades (288-bar DROP): {total_288_trades}")
    print(f"  Total trades (2000-bar/unlimited): {total_2000_trades}")
    print(f"  Timeout trades (>288 bars): {total_timeout} ({avg_timeout_pct}%)")

    if timeout_pnls:
        tp_arr = np.array(timeout_pnls)
        wins_forced = sum(1 for p in tp_arr if p > 0)
        print(f"\n  --- Timeout Trade Forced-Close Analysis ---")
        print(f"  If closed at 288 bars (market price):")
        print(f"    N={len(tp_arr)}, Win%={wins_forced/len(tp_arr)*100:.1f}%")
        print(f"    Mean PnL: {tp_arr.mean():+.2f}%")
        print(f"    Median PnL: {np.median(tp_arr):+.2f}%")
        print(f"    Std: {tp_arr.std():.2f}%")
        print(f"    Min/Max: {tp_arr.min():+.2f}% / {tp_arr.max():+.2f}%")

    # Resolution timing for timeout trades
    total_289_500 = sum(v['resolved_289_500'] for v in per_pattern_analysis.values())
    total_501_1000 = sum(v['resolved_501_1000'] for v in per_pattern_analysis.values())
    total_1001_plus = sum(v['resolved_1001_plus'] for v in per_pattern_analysis.values())

    if total_timeout > 0:
        print(f"\n  --- Timeout Resolution Timing ---")
        print(f"  289-500 bars (24-42h):  {total_289_500} ({total_289_500/total_timeout*100:.1f}%)")
        print(f"  501-1000 bars (42-83h): {total_501_1000} ({total_501_1000/total_timeout*100:.1f}%)")
        print(f"  1001+ bars (>83h):      {total_1001_plus} ({total_1001_plus/total_timeout*100:.1f}%)")

    print(f"\n  --- Portfolio: DROP (288) vs HOLD (2000) ---")
    print(f"  {'Metric':<20} {'DROP (288)':<18} {'HOLD (2000)':<18}")
    print(f"  {'-'*56}")
    for metric in ['pnl', 'trades', 'wr', 'mdd', 'pf']:
        v288 = stats_288.get(metric, 0)
        v2000 = stats_2000.get(metric, 0)
        if metric == 'pnl':
            print(f"  {'PnL':<20} {v288:>+.1f}%{'':<10} {v2000:>+.1f}%")
        elif metric == 'trades':
            print(f"  {'Trades':<20} {v288:<18} {v2000:<18}")
        elif metric == 'wr':
            print(f"  {'WR':<20} {v288:.1f}%{'':<12} {v2000:.1f}%")
        elif metric == 'mdd':
            print(f"  {'MDD':<20} {v288:.1f}%{'':<12} {v2000:.1f}%")
        elif metric == 'pf':
            print(f"  {'PF':<20} {v288:.2f}{'':<13} {v2000:.2f}")

    # Top patterns by timeout rate
    sorted_by_timeout = sorted(per_pattern_analysis.values(),
                                key=lambda x: x['timeout_pct'], reverse=True)
    print(f"\n  --- Top 15 Patterns by Timeout Rate ---")
    print(f"  {'Pattern':<18} {'Dir':<6} {'TP/SL':<8} {'Tr288':>6} {'Tr2K':>6} "
          f"{'TO':>4} {'TO%':>6} {'TO_WR':>6}")
    for p in sorted_by_timeout[:15]:
        print(f"  {p['pattern']:<18} {p['direction']:<6} "
              f"{p['tp']}/{p['sl']:<5} "
              f"{p['trades_288']:>6} {p['trades_2000']:>6} "
              f"{p['timeout_count']:>4} {p['timeout_pct']:>5.1f}% "
              f"{p['timeout_wr']:>5.1f}%")

    result = {
        'aggregate': {
            'trades_288': total_288_trades,
            'trades_2000': total_2000_trades,
            'timeout_count': total_timeout,
            'timeout_pct': avg_timeout_pct,
        },
        'forced_close_stats': {
            'n': len(timeout_pnls),
            'win_pct': round(sum(1 for p in timeout_pnls if p > 0) / len(timeout_pnls) * 100, 1) if timeout_pnls else 0,
            'mean_pnl': round(float(np.mean(timeout_pnls)), 2) if timeout_pnls else 0,
            'median_pnl': round(float(np.median(timeout_pnls)), 2) if timeout_pnls else 0,
        },
        'resolution_timing': {
            '289_500_bars': total_289_500,
            '501_1000_bars': total_501_1000,
            '1001_plus_bars': total_1001_plus,
        },
        'portfolio_drop_288': stats_288,
        'portfolio_hold_2000': stats_2000,
        'top_timeout_patterns': [
            {k: v for k, v in p.items()}
            for p in sorted_by_timeout[:15]
        ],
    }

    return result


# ============================================================
# Study 4: Edge Threshold Re-optimization
# ============================================================

def study_4_edge_threshold(df, signal_index, all_details_by_edge):
    """288봉 기준 Edge threshold 최적화.

    21.8pp는 500봉 시대의 값. 288봉에서 최적 threshold는?
    WF OOS 기준으로 판단.
    """
    print("\n" + "=" * 70)
    print("  STUDY 4: Edge Threshold Re-optimization (288 bars)")
    print("  기존 21.8pp → 288봉 기준 최적값 탐색 (WF OOS 기준)")
    print("=" * 70)

    # Test range: 18, 20, 21.8, 23, 25, 27, 30
    thresholds = [18.0, 20.0, 21.8, 23.0, 25.0, 27.0, 30.0]

    results = {}
    for edge_min in thresholds:
        # Filter patterns by edge threshold
        filtered = {k: v for k, v in all_details_by_edge.items()
                    if v['edge'] >= edge_min and v['wr'] >= 60.0 and v['sl'] >= 1.0}

        if not filtered:
            print(f"\n  Edge >= {edge_min}pp: 0 patterns (SKIP)")
            results[f'E{edge_min}'] = {'n_patterns': 0, 'skip': True}
            continue

        n_l = sum(1 for v in filtered.values() if v['direction'] == 'LONG')
        n_s = sum(1 for v in filtered.values() if v['direction'] == 'SHORT')

        bt = portfolio_backtest(filtered, signal_index, df, 288)
        print_scenario(f"Edge >= {edge_min}pp", bt['stats'], bt['mc_p'],
                       len(filtered), n_l, n_s, bt['raw_trades'], bt['duration'])

        results[f'E{edge_min}'] = {
            'edge_threshold': edge_min,
            'n_patterns': len(filtered),
            'n_long': n_l, 'n_short': n_s,
            'stats': bt['stats'],
            'mc_p': bt['mc_p'],
            'duration': bt['duration'],
        }

    # WF for thresholds with sufficient patterns
    print("\n  --- WF Validation ---")
    for edge_min in thresholds:
        label = f'E{edge_min}'
        if results.get(label, {}).get('skip'):
            continue
        if results[label]['n_patterns'] < 5:
            print(f"  {label}: too few patterns ({results[label]['n_patterns']}), skip WF")
            continue

        def make_edge_filter(threshold):
            def f(is_patterns):
                return [p for p in is_patterns if p.get('edge', 0) >= threshold]
            return f

        wf = run_expanding_wf(
            signal_index, df, 288, n_folds=3,
            edge_threshold=scanner.DEFAULT_EDGE_THRESHOLD,
            post_filter_fn=make_edge_filter(edge_min)
        )
        print_wf(wf, f"Edge>={edge_min}pp")
        results[label]['wf'] = {k: v for k, v in wf.items() if k != 'stable_patterns'}

    # Find optimal threshold by WF OOS PnL/trade
    print("\n  --- Threshold Comparison ---")
    print(f"  {'Edge':<8} {'Pat':>5} {'IS PnL':>10} {'IS MDD':>8} {'PnL/MDD':>9} "
          f"{'WF Pos':>7} {'WF OOS':>10} {'WF Tr':>7}")
    print(f"  {'-'*70}")
    for edge_min in thresholds:
        label = f'E{edge_min}'
        r = results.get(label, {})
        if r.get('skip'):
            continue
        wf = r.get('wf', {})
        print(f"  {edge_min:>5.1f}pp {r['n_patterns']:>5} "
              f"{r['stats']['pnl']:>+9.1f}% {r['stats']['mdd']:>7.1f}% "
              f"{r['stats']['pnl']/r['stats']['mdd'] if r['stats']['mdd']>0 else 0:>8.1f}x "
              f"{wf.get('positive_folds','?'):>4}/{wf.get('n_folds','?')} "
              f"{wf.get('total_oos_pnl','?'):>+9}% "
              f"{wf.get('total_oos_trades','?'):>7}")

    return results


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 70)
    print("  Strategy Improvement Study")
    print("  Based on v1.28.24 (112 patterns, MAX_BARS=288)")
    print("=" * 70)

    t_start = time.time()

    # Load data
    print("\n  Loading data...")
    df = scanner.load_and_classify(scanner.DEFAULT_DATA_FILE)
    types = df['candle_type'].values
    signal_index = scanner.build_signal_index(types, len(df))
    print(f"  Data: {len(df)} bars ({len(df)*5/60/24:.0f} days)")
    print(f"  Patterns in index: {len(signal_index)}")

    # Load current 112 pattern details
    dp_file = os.path.join(_PROJECT_ROOT, 'results', 'dynamic_patterns.json')
    with open(dp_file) as f:
        dp = json.load(f)
    all_details = dp['pattern_details']
    print(f"  Current patterns: {len(all_details)}")

    # For Study 4, we need ALL patterns from scanner (not just post-filtered 112)
    # Re-run discovery to get full set
    print("\n  Running full 288-bar discovery for Study 4...")
    t0 = time.time()
    original = scanner.MAX_BARS
    scanner.MAX_BARS = 288
    try:
        full_scan = scanner.scan_patterns_pp(
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
    all_scan_details = full_scan.get('pattern_details', {})
    print(f"  Full scan: {len(all_scan_details)} patterns in {time.time()-t0:.1f}s")

    # Run studies
    results = {
        'study': 'Strategy Improvement Study',
        'date': time.strftime('%Y-%m-%d'),
        'baseline': '112 patterns, MAX_BARS=288, Edge>=21.8pp',
    }

    results['study_1_wf_stable'] = study_1_wf_stable(df, signal_index, all_details)
    results['study_2_direction_balance'] = study_2_direction_balance(df, signal_index, all_details)
    results['study_3_timeout_behavior'] = study_3_timeout_behavior(df, signal_index, all_details)
    results['study_4_edge_threshold'] = study_4_edge_threshold(df, signal_index, all_scan_details)

    elapsed = time.time() - t_start
    results['total_time_seconds'] = round(elapsed, 1)

    # Save results
    with open(RESULT_FILE, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n  Results saved: {RESULT_FILE}")
    print(f"  Total time: {elapsed:.1f}s ({elapsed/60:.1f}min)")


if __name__ == '__main__':
    main()
