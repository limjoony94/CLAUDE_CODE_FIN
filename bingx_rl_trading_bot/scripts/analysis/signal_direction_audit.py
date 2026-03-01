#!/usr/bin/env python3
"""
Signal Direction Audit — 패턴 신호의 방향 예측력 근본 검증

핵심 질문:
  1. LONG 패턴 발화 후 실제로 가격이 올라가는가?
  2. SHORT 패턴 발화 후 실제로 가격이 내려가는가?
  3. 시간에 따라 방향 예측력이 변하는가?
  4. 최근 60일에서도 예측력이 유효한가?
  5. TP/SL 없이 fixed-horizon return 기준으로도 edge가 있는가?

방법:
  - 130패턴 신호 발화 후 fixed-horizon 수익률 측정 (TP/SL 무관)
  - 5/10/30/60/288 bar horizon (5min ~ 24h)
  - 방향 정확도: 예측 방향으로 가격이 이동한 비율
  - 전체 기간 vs 최근 60일 vs 분기별 비교
  - ALL 1,728 패턴 universe도 검증 (우리 130만 특별한지)

Protocol: Production classify_candle(), no TP/SL bias, pure direction signal quality
"""

import os
import sys
import json
import time
import numpy as np
import pandas as pd
from collections import defaultdict

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
sys.path.insert(0, _PROJECT_ROOT)

import scripts.scanner.pattern_scanner as scanner

RESULT_FILE = os.path.join(_PROJECT_ROOT, 'results', 'signal_direction_audit.json')
DATA_FILE = os.path.join(_PROJECT_ROOT, 'data', 'btc_5m_270days_reclassified.csv')
PATTERNS_FILE = os.path.join(_PROJECT_ROOT, 'results', 'dynamic_patterns.json')
BARS_PER_DAY = 288

# Fixed horizons to measure (in bars)
HORIZONS = [5, 10, 30, 60, 288]  # 25min, 50min, 2.5h, 5h, 24h
RECENT_DAYS = 60  # "recent" = last 60 days


def load_patterns():
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
    return patterns


def measure_direction_accuracy(signal_bars, direction, opens, closes, n_bars, horizons):
    """For each signal, measure if price moved in predicted direction at each horizon.

    Returns dict of horizon → {correct, total, returns[]}.
    """
    results = {h: {'correct': 0, 'total': 0, 'returns': []} for h in horizons}

    for sig in signal_bars:
        entry_bar = sig + 1
        if entry_bar >= n_bars:
            continue
        entry_price = opens[entry_bar]
        if entry_price <= 0:
            continue

        for h in horizons:
            exit_bar = entry_bar + h
            if exit_bar >= n_bars:
                continue
            exit_price = closes[exit_bar]

            # Raw return in predicted direction
            if direction == 'LONG':
                ret = (exit_price - entry_price) / entry_price * 100
            else:
                ret = (entry_price - exit_price) / entry_price * 100

            results[h]['total'] += 1
            results[h]['returns'].append(ret)
            if ret > 0:
                results[h]['correct'] += 1

    return results


def main():
    t0 = time.time()
    output = {'study': 'signal_direction_audit', 'sections': {}}

    print("=" * 70)
    print("SIGNAL DIRECTION AUDIT — 패턴 방향 예측력 근본 검증")
    print("=" * 70)

    # Load data
    df = scanner.load_and_classify(DATA_FILE)
    types = df['candle_type'].tolist()
    signal_index = scanner.build_signal_index(types, len(df))
    opens = df['open'].values
    closes = df['close'].values
    highs = df['high'].values
    lows = df['low'].values
    n_bars = len(df)
    timestamps = pd.to_datetime(df['timestamp'])

    patterns = load_patterns()
    recent_bar = n_bars - RECENT_DAYS * BARS_PER_DAY

    print(f"  {n_bars} bars ({n_bars/BARS_PER_DAY:.0f} days)")
    print(f"  {len(patterns)} patterns (our selection)")
    print(f"  Recent cutoff: bar {recent_bar} ({RECENT_DAYS} days)")
    print(f"  Horizons: {HORIZONS} bars")

    output['meta'] = {
        'bars': n_bars, 'patterns': len(patterns),
        'recent_days': RECENT_DAYS, 'recent_bar': recent_bar,
        'horizons': HORIZONS,
    }

    # ═══════════════════════════════════════════════════════════
    # 1. OVERALL DIRECTION ACCURACY (our 130 patterns)
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("1. DIRECTION ACCURACY — 130 Patterns, Full Period")
    print(f"{'='*70}")

    # Aggregate all signals by direction
    long_signals = []
    short_signals = []
    for pat in patterns:
        sigs = signal_index.get(pat['pattern'], [])
        if pat['direction'] == 'LONG':
            long_signals.extend(sigs)
        else:
            short_signals.extend(sigs)

    long_signals.sort()
    short_signals.sort()
    print(f"  LONG signals: {len(long_signals)}")
    print(f"  SHORT signals: {len(short_signals)}")

    # Measure direction accuracy at each horizon
    print(f"\n  {'Horizon':<12} {'LONG Acc':>9} {'LONG Ret':>9} {'SHORT Acc':>10} {'SHORT Ret':>10} {'Combined':>9}")
    print(f"  {'-'*64}")

    overall_output = []
    for h in HORIZONS:
        long_acc = measure_direction_accuracy(long_signals, 'LONG', opens, closes, n_bars, [h])
        short_acc = measure_direction_accuracy(short_signals, 'SHORT', opens, closes, n_bars, [h])

        l = long_acc[h]
        s = short_acc[h]
        l_pct = l['correct'] / l['total'] * 100 if l['total'] > 0 else 0
        s_pct = s['correct'] / s['total'] * 100 if s['total'] > 0 else 0
        l_ret = np.mean(l['returns']) if l['returns'] else 0
        s_ret = np.mean(s['returns']) if s['returns'] else 0

        combined_correct = l['correct'] + s['correct']
        combined_total = l['total'] + s['total']
        combined_pct = combined_correct / combined_total * 100 if combined_total > 0 else 0

        h_label = f"{h*5}min" if h < 288 else f"{h*5//60}h"
        print(f"  {h:>4}b ({h_label:<5}) {l_pct:>8.1f}% {l_ret:>+8.3f}% "
              f"{s_pct:>9.1f}% {s_ret:>+9.3f}% {combined_pct:>8.1f}%")

        overall_output.append({
            'horizon_bars': h, 'horizon_label': h_label,
            'long_accuracy': round(l_pct, 1), 'long_mean_return': round(l_ret, 4),
            'long_total': l['total'],
            'short_accuracy': round(s_pct, 1), 'short_mean_return': round(s_ret, 4),
            'short_total': s['total'],
            'combined_accuracy': round(combined_pct, 1),
        })
    output['sections']['overall'] = overall_output

    # ═══════════════════════════════════════════════════════════
    # 2. RECENT 60 DAYS vs EARLIER
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"2. DIRECTION ACCURACY — Recent {RECENT_DAYS}d vs Earlier")
    print(f"{'='*70}")

    periods = [
        ('Earlier', 0, recent_bar),
        (f'Recent {RECENT_DAYS}d', recent_bar, n_bars),
    ]

    period_output = {}
    for period_name, bar_start, bar_end in periods:
        print(f"\n  --- {period_name} (bar {bar_start}→{bar_end}) ---")

        for direction, dir_signals in [('LONG', long_signals), ('SHORT', short_signals)]:
            period_sigs = [s for s in dir_signals if bar_start <= s < bar_end]
            if not period_sigs:
                print(f"    {direction}: no signals")
                continue

            acc = measure_direction_accuracy(period_sigs, direction, opens, closes, n_bars, HORIZONS)

            row = []
            for h in HORIZONS:
                a = acc[h]
                pct = a['correct'] / a['total'] * 100 if a['total'] > 0 else 0
                ret = np.mean(a['returns']) if a['returns'] else 0
                row.append((h, pct, ret, a['total']))

            h_label = '  '.join(f"{r[1]:.0f}%" for r in row)
            print(f"    {direction} ({len(period_sigs)} signals): acc @ horizons = {h_label}")

            period_output[f"{period_name}_{direction}"] = [
                {'horizon': r[0], 'accuracy': round(r[1], 1),
                 'mean_return': round(r[2], 4), 'n_signals': r[3]}
                for r in row
            ]
    output['sections']['period_comparison'] = period_output

    # ═══════════════════════════════════════════════════════════
    # 3. QUARTERLY DIRECTION ACCURACY
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("3. QUARTERLY DIRECTION ACCURACY (288-bar / 24h horizon)")
    print(f"{'='*70}")

    quarter_bars = 90 * BARS_PER_DAY
    n_quarters = n_bars // quarter_bars + 1
    h_main = 288  # 24h horizon

    quarterly_output = []
    print(f"  {'Quarter':<10} {'LONG Acc':>9} {'LONG N':>7} {'SHORT Acc':>10} {'SHORT N':>8} {'BTC Ret':>8}")
    print(f"  {'-'*57}")

    for q in range(n_quarters):
        q_start = q * quarter_bars
        q_end = min((q + 1) * quarter_bars, n_bars)
        if q_end - q_start < BARS_PER_DAY:
            continue

        btc_ret = (closes[min(q_end - 1, n_bars - 1)] - closes[q_start]) / closes[q_start] * 100

        q_date = timestamps.iloc[q_start].strftime('%Y-%m')

        results_q = {}
        for direction, dir_signals in [('LONG', long_signals), ('SHORT', short_signals)]:
            q_sigs = [s for s in dir_signals if q_start <= s < q_end]
            if not q_sigs:
                results_q[direction] = {'accuracy': 0, 'n': 0}
                continue
            acc = measure_direction_accuracy(q_sigs, direction, opens, closes, n_bars, [h_main])
            a = acc[h_main]
            pct = a['correct'] / a['total'] * 100 if a['total'] > 0 else 0
            results_q[direction] = {'accuracy': round(pct, 1), 'n': a['total']}

        print(f"  {q_date:<10} {results_q['LONG']['accuracy']:>8.1f}% {results_q['LONG']['n']:>6} "
              f"{results_q['SHORT']['accuracy']:>9.1f}% {results_q['SHORT']['n']:>7} "
              f"{btc_ret:>+7.1f}%")

        quarterly_output.append({
            'quarter': q_date, 'btc_return': round(btc_ret, 1),
            'long_accuracy': results_q['LONG']['accuracy'],
            'long_n': results_q['LONG']['n'],
            'short_accuracy': results_q['SHORT']['accuracy'],
            'short_n': results_q['SHORT']['n'],
        })
    output['sections']['quarterly'] = quarterly_output

    # ═══════════════════════════════════════════════════════════
    # 4. PER-PATTERN DIRECTION ACCURACY (our 130, recent period)
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"4. PER-PATTERN ACCURACY (24h horizon, recent {RECENT_DAYS}d)")
    print(f"{'='*70}")

    h_audit = 288
    pat_audit = []

    for pat in patterns:
        sigs = signal_index.get(pat['pattern'], [])
        recent_sigs = [s for s in sigs if s >= recent_bar]
        all_sigs = sigs

        if not recent_sigs:
            continue

        # Recent accuracy
        acc_recent = measure_direction_accuracy(
            recent_sigs, pat['direction'], opens, closes, n_bars, [h_audit])
        r = acc_recent[h_audit]
        recent_pct = r['correct'] / r['total'] * 100 if r['total'] > 0 else 0
        recent_ret = np.mean(r['returns']) if r['returns'] else 0

        # Full-period accuracy
        acc_all = measure_direction_accuracy(
            all_sigs, pat['direction'], opens, closes, n_bars, [h_audit])
        a = acc_all[h_audit]
        all_pct = a['correct'] / a['total'] * 100 if a['total'] > 0 else 0

        pat_audit.append({
            'pattern': pat['pattern'],
            'direction': pat['direction'],
            'recent_signals': r['total'],
            'recent_accuracy': round(recent_pct, 1),
            'recent_mean_return': round(recent_ret, 4),
            'full_accuracy': round(all_pct, 1),
            'full_signals': a['total'],
            'accuracy_change': round(recent_pct - all_pct, 1),
        })

    pat_audit.sort(key=lambda x: x['recent_accuracy'])

    print(f"\n  {'Pattern':<18} {'Dir':>5} {'Rec-N':>5} {'Rec-Acc':>8} {'Rec-Ret':>9} {'Full-Acc':>9} {'Δ':>7}")
    print(f"  {'-'*67}")

    # Show worst 15
    print("  --- WORST (방향 예측력 없음/역) ---")
    for p in pat_audit[:15]:
        print(f"  {p['pattern']:<18} {p['direction']:>5} {p['recent_signals']:>5} "
              f"{p['recent_accuracy']:>7.1f}% {p['recent_mean_return']:>+8.3f}% "
              f"{p['full_accuracy']:>8.1f}% {p['accuracy_change']:>+6.1f}")

    print(f"\n  --- BEST (방향 예측력 유지) ---")
    for p in pat_audit[-10:]:
        print(f"  {p['pattern']:<18} {p['direction']:>5} {p['recent_signals']:>5} "
              f"{p['recent_accuracy']:>7.1f}% {p['recent_mean_return']:>+8.3f}% "
              f"{p['full_accuracy']:>8.1f}% {p['accuracy_change']:>+6.1f}")

    # Direction summary
    long_pats = [p for p in pat_audit if p['direction'] == 'LONG']
    short_pats = [p for p in pat_audit if p['direction'] == 'SHORT']
    long_good = sum(1 for p in long_pats if p['recent_accuracy'] > 50)
    short_good = sum(1 for p in short_pats if p['recent_accuracy'] > 50)

    print(f"\n  LONG: {long_good}/{len(long_pats)} patterns with >50% accuracy (recent)")
    print(f"  SHORT: {short_good}/{len(short_pats)} patterns with >50% accuracy (recent)")

    output['sections']['per_pattern'] = pat_audit

    # ═══════════════════════════════════════════════════════════
    # 5. UNIVERSE COMPARISON — Our 130 vs Random vs All
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("5. UNIVERSE COMPARISON — Selected vs All Patterns")
    print(f"{'='*70}")

    # All possible 3-candle patterns in the data
    all_patterns_in_data = set(signal_index.keys())
    our_patterns = set(f"{p['pattern']}" for p in patterns)

    print(f"  Unique 3-candle patterns in data: {len(all_patterns_in_data)}")
    print(f"  Our selected patterns: {len(our_patterns)}")

    # For each pattern in the universe, measure direction accuracy both ways
    universe_long_accs = []
    universe_short_accs = []
    our_long_accs = []
    our_short_accs = []

    h_univ = 288

    for pat_name in all_patterns_in_data:
        sigs = signal_index[pat_name]
        recent_sigs = [s for s in sigs if s >= recent_bar]
        if len(recent_sigs) < 3:
            continue

        # Test as LONG
        acc_long = measure_direction_accuracy(recent_sigs, 'LONG', opens, closes, n_bars, [h_univ])
        l = acc_long[h_univ]
        if l['total'] >= 3:
            l_pct = l['correct'] / l['total'] * 100
            universe_long_accs.append(l_pct)

        # Test as SHORT
        acc_short = measure_direction_accuracy(recent_sigs, 'SHORT', opens, closes, n_bars, [h_univ])
        s = acc_short[h_univ]
        if s['total'] >= 3:
            s_pct = s['correct'] / s['total'] * 100
            universe_short_accs.append(s_pct)

    # Our patterns
    for p in pat_audit:
        if p['direction'] == 'LONG':
            our_long_accs.append(p['recent_accuracy'])
        else:
            our_short_accs.append(p['recent_accuracy'])

    print(f"\n  Universe (recent {RECENT_DAYS}d, 24h horizon):")
    if universe_long_accs:
        print(f"    ALL patterns as LONG: mean acc {np.mean(universe_long_accs):.1f}%, "
              f"median {np.median(universe_long_accs):.1f}%, std {np.std(universe_long_accs):.1f}%")
    if universe_short_accs:
        print(f"    ALL patterns as SHORT: mean acc {np.mean(universe_short_accs):.1f}%, "
              f"median {np.median(universe_short_accs):.1f}%, std {np.std(universe_short_accs):.1f}%")

    print(f"\n  Our selected (recent {RECENT_DAYS}d, 24h horizon):")
    if our_long_accs:
        print(f"    Our LONG (n={len(our_long_accs)}): mean acc {np.mean(our_long_accs):.1f}%, "
              f"median {np.median(our_long_accs):.1f}%")
    if our_short_accs:
        print(f"    Our SHORT (n={len(our_short_accs)}): mean acc {np.mean(our_short_accs):.1f}%, "
              f"median {np.median(our_short_accs):.1f}%")

    # Are our patterns better than random selection?
    if our_long_accs and universe_long_accs:
        diff_l = np.mean(our_long_accs) - np.mean(universe_long_accs)
        print(f"\n    Our LONG vs Universe: {diff_l:+.1f}pp "
              f"({'better' if diff_l > 0 else 'WORSE'})")
    if our_short_accs and universe_short_accs:
        diff_s = np.mean(our_short_accs) - np.mean(universe_short_accs)
        print(f"    Our SHORT vs Universe: {diff_s:+.1f}pp "
              f"({'better' if diff_s > 0 else 'WORSE'})")

    output['sections']['universe'] = {
        'universe_long_mean': round(float(np.mean(universe_long_accs)), 1) if universe_long_accs else 0,
        'universe_short_mean': round(float(np.mean(universe_short_accs)), 1) if universe_short_accs else 0,
        'our_long_mean': round(float(np.mean(our_long_accs)), 1) if our_long_accs else 0,
        'our_short_mean': round(float(np.mean(our_short_accs)), 1) if our_short_accs else 0,
    }

    # ═══════════════════════════════════════════════════════════
    # 6. MFE/MAE ANALYSIS (recent period)
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print(f"6. MFE/MAE — Maximum Excursions (recent {RECENT_DAYS}d)")
    print(f"{'='*70}")

    mfe_mae_output = {}
    for direction, dir_signals in [('LONG', long_signals), ('SHORT', short_signals)]:
        recent_sigs = [s for s in dir_signals if s >= recent_bar]
        if not recent_sigs:
            continue

        mfes = []
        maes = []
        for sig in recent_sigs:
            entry_bar = sig + 1
            if entry_bar >= n_bars:
                continue
            entry_price = opens[entry_bar]
            if entry_price <= 0:
                continue

            # Look ahead 288 bars (24h)
            end = min(entry_bar + 288, n_bars)
            if entry_bar >= end:
                continue

            window_highs = highs[entry_bar:end]
            window_lows = lows[entry_bar:end]

            if direction == 'LONG':
                mfe = (max(window_highs) - entry_price) / entry_price * 100
                mae = (entry_price - min(window_lows)) / entry_price * 100
            else:
                mfe = (entry_price - min(window_lows)) / entry_price * 100
                mae = (max(window_highs) - entry_price) / entry_price * 100

            mfes.append(mfe)
            maes.append(mae)

        if mfes:
            print(f"\n  {direction} ({len(mfes)} signals, 24h window):")
            print(f"    MFE: mean {np.mean(mfes):.3f}%, "
                  f"median {np.median(mfes):.3f}%, P25 {np.percentile(mfes, 25):.3f}%")
            print(f"    MAE: mean {np.mean(maes):.3f}%, "
                  f"median {np.median(maes):.3f}%, P25 {np.percentile(maes, 25):.3f}%")
            print(f"    MFE/MAE ratio: {np.mean(mfes)/np.mean(maes):.3f} "
                  f"({'favorable' if np.mean(mfes) > np.mean(maes) else 'UNFAVORABLE'})")

            # What TP/SL levels are achievable?
            for tp_pct in [0.5, 1.0, 1.5, 2.0]:
                hit_rate = sum(1 for m in mfes if m >= tp_pct) / len(mfes) * 100
                print(f"    MFE >= {tp_pct}%: {hit_rate:.1f}% of signals")

            mfe_mae_output[direction] = {
                'n': len(mfes),
                'mfe_mean': round(float(np.mean(mfes)), 4),
                'mfe_median': round(float(np.median(mfes)), 4),
                'mae_mean': round(float(np.mean(maes)), 4),
                'mae_median': round(float(np.median(maes)), 4),
                'mfe_mae_ratio': round(float(np.mean(mfes) / np.mean(maes)), 3) if np.mean(maes) > 0 else 0,
            }
    output['sections']['mfe_mae'] = mfe_mae_output

    # ═══════════════════════════════════════════════════════════
    # 7. BREAKEVEN ANALYSIS — What WR is needed?
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("7. BREAKEVEN ANALYSIS — 현행 TP/SL에서 필요 WR vs 실제 방향 예측력")
    print(f"{'='*70}")

    # Our patterns' average TP/SL
    tps = [p['tp'] for p in patterns]
    sls = [p['sl'] for p in patterns]
    avg_tp = np.mean(tps)
    avg_sl = np.mean(sls)
    breakeven_wr = avg_sl / (avg_tp + avg_sl) * 100

    # Actual recent direction accuracy at TP-horizon
    # Approximate TP horizon: how many bars to typically reach TP distance?
    # Use our recent MFE data: what fraction hits TP within 24h?
    print(f"  Average TP: {avg_tp:.2f}% | Average SL: {avg_sl:.2f}%")
    print(f"  R:R ratio: {avg_tp/avg_sl:.3f}")
    print(f"  Breakeven WR (random walk): {breakeven_wr:.1f}%")

    # Recent period TP/SL hit rates from actual scanner backtest
    print(f"\n  Recent period direction accuracy (24h horizon):")
    if our_long_accs:
        print(f"    LONG: {np.mean(our_long_accs):.1f}% "
              f"({'above' if np.mean(our_long_accs) > 50 else 'BELOW'} 50% → "
              f"{'HAS' if np.mean(our_long_accs) > 50 else 'NO'} directional edge)")
    if our_short_accs:
        print(f"    SHORT: {np.mean(our_short_accs):.1f}% "
              f"({'above' if np.mean(our_short_accs) > 50 else 'BELOW'} 50% → "
              f"{'HAS' if np.mean(our_short_accs) > 50 else 'NO'} directional edge)")

    # What needs to be true for the system to work:
    # WR > breakeven_wr  AND  direction accuracy > 50%
    print(f"\n  System requirements:")
    print(f"    ① Direction accuracy > 50% (net upward for LONG, net downward for SHORT)")
    print(f"    ② Actual WR > {breakeven_wr:.1f}% (breakeven with R:R {avg_tp/avg_sl:.3f})")

    output['sections']['breakeven'] = {
        'avg_tp': round(avg_tp, 3), 'avg_sl': round(avg_sl, 3),
        'rr_ratio': round(avg_tp / avg_sl, 4),
        'breakeven_wr': round(breakeven_wr, 1),
    }

    # ═══════════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════════
    print(f"\n{'='*70}")
    print("SUMMARY & VERDICT")
    print(f"{'='*70}")

    # Determine verdict
    long_has_edge = bool(our_long_accs and np.mean(our_long_accs) > 52)
    short_has_edge = bool(our_short_accs and np.mean(our_short_accs) > 52)

    if long_has_edge and short_has_edge:
        verdict = "BOTH directions maintain edge — rescan may recover performance"
    elif short_has_edge and not long_has_edge:
        verdict = "SHORT-only edge — consider SHORT-only operation or LONG pattern redesign"
    elif long_has_edge and not short_has_edge:
        verdict = "LONG-only edge — consider LONG-only operation or SHORT pattern redesign"
    else:
        verdict = "NO directional edge detected — fundamental system redesign needed"

    print(f"\n  LONG direction prediction (recent {RECENT_DAYS}d):")
    if our_long_accs:
        print(f"    Mean accuracy: {np.mean(our_long_accs):.1f}% "
              f"(need >50%, {'PASS' if long_has_edge else 'FAIL'})")
    print(f"  SHORT direction prediction (recent {RECENT_DAYS}d):")
    if our_short_accs:
        print(f"    Mean accuracy: {np.mean(our_short_accs):.1f}% "
              f"(need >50%, {'PASS' if short_has_edge else 'FAIL'})")

    print(f"\n  VERDICT: {verdict}")

    # Additional recommendations
    if not long_has_edge:
        print(f"\n  ⚠ LONG patterns have lost directional prediction ability")
        print(f"    - {len(long_pats)} LONG patterns, "
              f"only {long_good} ({long_good/len(long_pats)*100:.0f}%) above 50%")
        long_below = [p for p in long_pats if p['recent_accuracy'] <= 50]
        if long_below:
            print(f"    - Worst LONG patterns should be removed or direction-flipped")
    if not short_has_edge:
        print(f"\n  ⚠ SHORT patterns have lost directional prediction ability")
        short_below = [p for p in short_pats if p['recent_accuracy'] <= 50]
        if short_below:
            print(f"    - {len(short_below)}/{len(short_pats)} SHORT below 50% accuracy")

    output['sections']['verdict'] = {
        'long_has_edge': long_has_edge,
        'short_has_edge': short_has_edge,
        'verdict': verdict,
        'long_patterns_above_50': long_good,
        'long_patterns_total': len(long_pats),
        'short_patterns_above_50': short_good,
        'short_patterns_total': len(short_pats),
    }

    elapsed = time.time() - t0
    output['elapsed_seconds'] = round(elapsed, 1)
    print(f"\n  Time: {elapsed:.1f}s")

    def json_safe(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        raise TypeError(f"Not serializable: {type(obj)}")

    with open(RESULT_FILE, 'w') as f:
        json.dump(output, f, indent=2, default=json_safe)
    print(f"  Saved to {RESULT_FILE}")


if __name__ == '__main__':
    main()
