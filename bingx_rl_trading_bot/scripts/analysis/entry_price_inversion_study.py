#!/usr/bin/env python3
"""
Entry Price Inversion Study — LONG이 SHORT보다 높은 가격에 진입하는 현상 분석

관찰: 현재 라이브 포지션에서 LONG 평단 > SHORT 평단 ($67,225 vs $66,626)
  - LONG은 local peak에서 진입 (높은 가격에서 매수)
  - SHORT는 local trough에서 진입 (낮은 가격에서 매도)
  → 양쪽 모두 불리한 진입 = 구조적 문제?

분석:
  Phase 1: 전체 297d에서 모든 패턴 신호의 진입가 비교
           LONG 신호 시점의 가격 vs SHORT 신호 시점의 가격
  Phase 2: 개별 패턴별 진입가 분석 — 어떤 패턴이 peak/trough에서 발화하는지
  Phase 3: 진입가 상대 위치 분석 — rolling window 내 percentile
           (0% = 윈도우 최저, 100% = 최고)
  Phase 4: 수익성과의 상관관계 — 높은 곳에서 LONG 진입 시 WR 영향
  Phase 5: 3-candle 패턴의 구조적 의미 — 패턴이 추세 추종인지 역추세인지

Protocol: Production classify_candle(), ATR-scaled TP/SL, compound portfolio
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from collections import defaultdict

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
sys.path.insert(0, _PROJECT_ROOT)

import scripts.scanner.pattern_scanner as scanner

RESULT_FILE = os.path.join(_PROJECT_ROOT, 'results', 'entry_price_inversion_study.json')
DATA_FILE = os.path.join(_PROJECT_ROOT, 'data', 'btc_5m_270days_reclassified.csv')
PATTERNS_FILE = os.path.join(_PROJECT_ROOT, 'results', 'dynamic_patterns.json')
BARS_PER_DAY = 288
LEVERAGE = 3
FEE_PCT = 0.10

# Rolling windows for percentile analysis
PERCENTILE_WINDOWS = [12, 36, 72, 288]  # 1h, 3h, 6h, 24h


def load_data():
    df = scanner.load_and_classify(DATA_FILE)
    # scanner produces 'candle_type' column with short codes (BD, BU, etc.)
    if 'candle_type' not in df.columns:
        raise RuntimeError("load_and_classify did not produce 'candle_type' column")
    return df


def load_patterns():
    with open(PATTERNS_FILE) as f:
        data = json.load(f)
    patterns = []
    for key, info in data.get('pattern_details', {}).items():
        patterns.append({
            'pattern': info['pattern'],
            'direction': info['direction'],
            'tp_pct': info.get('tp_pct', 2.0),
            'sl_pct': info.get('sl_pct', 3.0),
        })
    return patterns


def find_signals(df, patterns):
    """Find all signal locations for each pattern."""
    codes = df['candle_type'].values
    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values
    n = len(df)

    signals = []  # (bar_idx, pattern, direction, entry_price)
    pat_lookup = {}
    for p in patterns:
        key = p['pattern']
        if key not in pat_lookup:
            pat_lookup[key] = []
        pat_lookup[key].append(p)

    for i in range(2, n - 1):
        triplet = f"{codes[i-2]}-{codes[i-1]}-{codes[i]}"
        if triplet in pat_lookup:
            entry_price = opens[i + 1]  # next bar open
            for p in pat_lookup[triplet]:
                signals.append({
                    'bar': i + 1,
                    'pattern': triplet,
                    'direction': p['direction'],
                    'entry_price': entry_price,
                    'tp_pct': p['tp_pct'],
                    'sl_pct': p['sl_pct'],
                })
    return signals


def phase1_aggregate_entry_prices(signals, closes):
    """Phase 1: Aggregate entry price comparison LONG vs SHORT."""
    print("\n" + "="*70)
    print("PHASE 1: Aggregate Entry Price — LONG vs SHORT")
    print("="*70)

    long_prices = [s['entry_price'] for s in signals if s['direction'] == 'LONG']
    short_prices = [s['entry_price'] for s in signals if s['direction'] == 'SHORT']

    print(f"\nTotal signals: {len(signals)} ({len(long_prices)} LONG, {len(short_prices)} SHORT)")
    print(f"\nLONG  avg entry: ${np.mean(long_prices):,.1f}  (median ${np.median(long_prices):,.1f})")
    print(f"SHORT avg entry: ${np.mean(short_prices):,.1f}  (median ${np.median(short_prices):,.1f})")
    diff = np.mean(long_prices) - np.mean(short_prices)
    diff_pct = diff / np.mean(short_prices) * 100
    print(f"Difference: ${diff:+,.1f} ({diff_pct:+.3f}%)")

    # Per-quarter analysis
    n_bars = len(closes)
    quarter_size = n_bars // 4
    print(f"\nPer-Quarter Entry Price Difference:")
    print(f"{'Quarter':<10} {'LONG avg':>12} {'SHORT avg':>12} {'Diff':>10} {'Diff%':>8}")
    print("-" * 60)

    results = {'long_avg': np.mean(long_prices), 'short_avg': np.mean(short_prices),
               'diff': diff, 'diff_pct': diff_pct, 'quarters': []}

    for q in range(4):
        q_start = q * quarter_size
        q_end = (q + 1) * quarter_size if q < 3 else n_bars
        q_long = [s['entry_price'] for s in signals
                  if s['direction'] == 'LONG' and q_start <= s['bar'] < q_end]
        q_short = [s['entry_price'] for s in signals
                   if s['direction'] == 'SHORT' and q_start <= s['bar'] < q_end]
        if q_long and q_short:
            q_diff = np.mean(q_long) - np.mean(q_short)
            q_diff_pct = q_diff / np.mean(q_short) * 100
            print(f"Q{q+1:<9} ${np.mean(q_long):>11,.1f} ${np.mean(q_short):>11,.1f} "
                  f"${q_diff:>+9,.1f} {q_diff_pct:>+7.3f}%")
            results['quarters'].append({
                'q': q+1, 'long_avg': np.mean(q_long), 'short_avg': np.mean(q_short),
                'diff': q_diff, 'diff_pct': q_diff_pct,
                'n_long': len(q_long), 'n_short': len(q_short),
            })

    return results


def phase2_per_pattern_entry_prices(signals, closes):
    """Phase 2: Per-pattern entry price analysis."""
    print("\n" + "="*70)
    print("PHASE 2: Per-Pattern Entry Price")
    print("="*70)

    # Group signals by pattern
    pattern_signals = defaultdict(list)
    for s in signals:
        key = (s['pattern'], s['direction'])
        pattern_signals[key].append(s['entry_price'])

    results = []
    for (pat, direction), prices in sorted(pattern_signals.items()):
        avg_price = np.mean(prices)
        results.append({
            'pattern': pat,
            'direction': direction,
            'avg_entry': avg_price,
            'n_signals': len(prices),
        })

    # Sort by avg_entry descending
    results.sort(key=lambda x: x['avg_entry'], reverse=True)

    print(f"\n{'Pattern':<15} {'Dir':<6} {'Avg Entry':>12} {'Signals':>8}")
    print("-" * 45)
    for r in results[:10]:
        print(f"{r['pattern']:<15} {r['direction']:<6} ${r['avg_entry']:>11,.1f} {r['n_signals']:>8}")
    print("...")
    for r in results[-10:]:
        print(f"{r['pattern']:<15} {r['direction']:<6} ${r['avg_entry']:>11,.1f} {r['n_signals']:>8}")

    # Count how many LONG patterns have above-median entry
    all_entries = [r['avg_entry'] for r in results]
    median_entry = np.median(all_entries)
    long_above = sum(1 for r in results if r['direction'] == 'LONG' and r['avg_entry'] > median_entry)
    long_total = sum(1 for r in results if r['direction'] == 'LONG')
    short_below = sum(1 for r in results if r['direction'] == 'SHORT' and r['avg_entry'] < median_entry)
    short_total = sum(1 for r in results if r['direction'] == 'SHORT')

    print(f"\nMedian entry: ${median_entry:,.1f}")
    print(f"LONG above median: {long_above}/{long_total} ({long_above/max(1,long_total)*100:.0f}%)")
    print(f"SHORT below median: {short_below}/{short_total} ({short_below/max(1,short_total)*100:.0f}%)")

    return results


def phase3_entry_percentile(signals, closes, highs, lows):
    """Phase 3: Entry price as percentile within rolling window."""
    print("\n" + "="*70)
    print("PHASE 3: Entry Price Percentile (within rolling window)")
    print("="*70)
    print("0% = window lowest, 100% = window highest")

    n = len(closes)
    results = {}

    for window in PERCENTILE_WINDOWS:
        window_hours = window * 5 / 60
        long_pctiles = []
        short_pctiles = []

        for s in signals:
            bar = s['bar']
            entry = s['entry_price']

            # Get rolling window min/max
            start = max(0, bar - window)
            window_high = np.max(highs[start:bar+1])
            window_low = np.min(lows[start:bar+1])

            if window_high > window_low:
                pctile = (entry - window_low) / (window_high - window_low) * 100
            else:
                pctile = 50.0

            if s['direction'] == 'LONG':
                long_pctiles.append(pctile)
            else:
                short_pctiles.append(pctile)

        l_avg = np.mean(long_pctiles) if long_pctiles else 0
        s_avg = np.mean(short_pctiles) if short_pctiles else 0

        print(f"\nWindow: {window} bars ({window_hours:.0f}h)")
        print(f"  LONG  entry percentile: {l_avg:.1f}%  (ideally <50% = buy low)")
        print(f"  SHORT entry percentile: {s_avg:.1f}%  (ideally >50% = sell high)")
        print(f"  Inverted? LONG {'>50' if l_avg > 50 else '≤50'}, SHORT {'<50' if s_avg < 50 else '≥50'}")

        results[f'window_{window}'] = {
            'window_bars': window,
            'window_hours': window_hours,
            'long_avg_pctile': l_avg,
            'short_avg_pctile': s_avg,
            'inverted': l_avg > 50 and s_avg < 50,
            'long_pctiles_p25_p50_p75': [
                np.percentile(long_pctiles, 25),
                np.percentile(long_pctiles, 50),
                np.percentile(long_pctiles, 75),
            ] if long_pctiles else [],
            'short_pctiles_p25_p50_p75': [
                np.percentile(short_pctiles, 25),
                np.percentile(short_pctiles, 50),
                np.percentile(short_pctiles, 75),
            ] if short_pctiles else [],
        }

    return results


def phase4_entry_pctile_vs_outcome(signals, opens, highs, lows, closes):
    """Phase 4: Does entry percentile affect trade outcome?"""
    print("\n" + "="*70)
    print("PHASE 4: Entry Percentile vs Trade Outcome")
    print("="*70)

    n = len(closes)
    # Use 288-bar (24h) window for percentile
    window = 288

    # Compute ATR ratio for each signal
    atr_ratio = scanner.compute_atr_ratio(highs, lows, closes)

    results = {'long': [], 'short': []}

    for s in signals:
        bar = s['bar']
        entry = s['entry_price']
        direction = s['direction']
        tp_pct = s['tp_pct']
        sl_pct = s['sl_pct']

        # Percentile
        start = max(0, bar - window)
        window_high = np.max(highs[start:bar+1])
        window_low = np.min(lows[start:bar+1])
        if window_high > window_low:
            pctile = (entry - window_low) / (window_high - window_low) * 100
        else:
            pctile = 50.0

        # ATR scale TP/SL
        ar = atr_ratio[bar] if bar < len(atr_ratio) else 1.0
        ar = max(0.6, min(1.7, ar))
        tp = tp_pct * ar / 100
        sl = sl_pct * ar / 100

        # Simulate trade
        if direction == 'LONG':
            tp_price = entry * (1 + tp)
            sl_price = entry * (1 - sl)
        else:
            tp_price = entry * (1 - tp)
            sl_price = entry * (1 + sl)

        outcome = None
        for j in range(bar, min(bar + 864, n)):
            if direction == 'LONG':
                if lows[j] <= sl_price:
                    outcome = 'SL'
                    break
                if highs[j] >= tp_price:
                    outcome = 'TP'
                    break
            else:
                if highs[j] >= sl_price:
                    outcome = 'SL'
                    break
                if lows[j] <= tp_price:
                    outcome = 'TP'
                    break

        if outcome is None:
            outcome = 'TIMEOUT'

        is_win = outcome == 'TP'
        results[direction.lower()].append({
            'pctile': pctile,
            'win': is_win,
            'pattern': s['pattern'],
        })

    # Analyze: bin by percentile and compute WR
    print(f"\n{'Pctile Bin':<15} {'LONG WR':>10} {'LONG N':>8} {'SHORT WR':>10} {'SHORT N':>8}")
    print("-" * 55)

    summary = []
    bins = [(0, 25), (25, 50), (50, 75), (75, 100)]
    for lo, hi in bins:
        l_trades = [t for t in results['long'] if lo <= t['pctile'] < hi]
        s_trades = [t for t in results['short'] if lo <= t['pctile'] < hi]
        l_wr = np.mean([t['win'] for t in l_trades]) * 100 if l_trades else 0
        s_wr = np.mean([t['win'] for t in s_trades]) * 100 if s_trades else 0
        print(f"{lo:>3}-{hi:<3}%       {l_wr:>9.1f}% {len(l_trades):>8} {s_wr:>9.1f}% {len(s_trades):>8}")
        summary.append({
            'bin': f'{lo}-{hi}%',
            'long_wr': l_wr, 'long_n': len(l_trades),
            'short_wr': s_wr, 'short_n': len(s_trades),
        })

    # Key insight: is there a correlation?
    for direction in ['long', 'short']:
        pctiles = [t['pctile'] for t in results[direction]]
        wins = [1 if t['win'] else 0 for t in results[direction]]
        if pctiles:
            corr = np.corrcoef(pctiles, wins)[0, 1]
            print(f"\n{direction.upper()} pctile-win correlation: r={corr:.3f}")
            # For LONG: negative corr = buying high is bad (expected)
            # For SHORT: positive corr = selling high is good (expected)

    return {'bins': summary, 'raw_counts': {
        'long': len(results['long']), 'short': len(results['short'])
    }}


def phase5_pattern_structure(signals, opens, highs, lows, closes):
    """Phase 5: Are patterns trend-following or mean-reverting?"""
    print("\n" + "="*70)
    print("PHASE 5: Pattern Structure — Trend Following vs Mean Reverting")
    print("="*70)
    print("Measure: price change in the 3 candles BEFORE entry")
    print("  → Positive pre-move + LONG signal = trend-following")
    print("  → Negative pre-move + LONG signal = mean-reverting")

    results = {'long_premoves': [], 'short_premoves': [], 'patterns': {}}

    for s in signals:
        bar = s['bar']
        if bar < 4:
            continue

        # Pre-move: price change over the 3 pattern candles
        pre_open = opens[bar - 3]
        pre_close = closes[bar - 1]
        if pre_open <= 0:
            continue
        pre_move_pct = (pre_close / pre_open - 1) * 100

        direction = s['direction']
        pattern = s['pattern']

        if direction == 'LONG':
            results['long_premoves'].append(pre_move_pct)
        else:
            results['short_premoves'].append(pre_move_pct)

        key = (pattern, direction)
        if key not in results['patterns']:
            results['patterns'][key] = []
        results['patterns'][key].append(pre_move_pct)

    long_pm = results['long_premoves']
    short_pm = results['short_premoves']

    print(f"\nLONG signals ({len(long_pm)} total):")
    print(f"  Pre-move avg: {np.mean(long_pm):+.3f}%")
    print(f"  Pre-move > 0 (trend-following): {sum(1 for x in long_pm if x > 0)}/{len(long_pm)} "
          f"({sum(1 for x in long_pm if x > 0)/len(long_pm)*100:.1f}%)")
    if np.mean(long_pm) > 0:
        print(f"  → TREND-FOLLOWING: LONGs fire after UP moves (buying high)")
    else:
        print(f"  → MEAN-REVERTING: LONGs fire after DOWN moves (buying dip)")

    print(f"\nSHORT signals ({len(short_pm)} total):")
    print(f"  Pre-move avg: {np.mean(short_pm):+.3f}%")
    print(f"  Pre-move < 0 (trend-following): {sum(1 for x in short_pm if x < 0)}/{len(short_pm)} "
          f"({sum(1 for x in short_pm if x < 0)/len(short_pm)*100:.1f}%)")
    if np.mean(short_pm) < 0:
        print(f"  → TREND-FOLLOWING: SHORTs fire after DOWN moves (selling low)")
    else:
        print(f"  → MEAN-REVERTING: SHORTs fire after UP moves (selling high)")

    # Per-pattern breakdown
    print(f"\nTop 10 most trend-following LONG patterns (highest pre-move):")
    long_pats = [(k, np.mean(v)) for k, v in results['patterns'].items() if k[1] == 'LONG']
    long_pats.sort(key=lambda x: x[1], reverse=True)
    print(f"{'Pattern':<15} {'Dir':<6} {'PreMove':>10} {'Signals':>8}")
    print("-" * 45)
    for (pat, d), pm in long_pats[:10]:
        n = len(results['patterns'][(pat, d)])
        label = "↑trend" if pm > 0 else "↓reversal"
        print(f"{pat:<15} {d:<6} {pm:>+9.3f}% {n:>8}  {label}")

    print(f"\nTop 10 most trend-following SHORT patterns (lowest pre-move):")
    short_pats = [(k, np.mean(v)) for k, v in results['patterns'].items() if k[1] == 'SHORT']
    short_pats.sort(key=lambda x: x[1])
    print(f"{'Pattern':<15} {'Dir':<6} {'PreMove':>10} {'Signals':>8}")
    print("-" * 45)
    for (pat, d), pm in short_pats[:10]:
        n = len(results['patterns'][(pat, d)])
        label = "↓trend" if pm < 0 else "↑reversal"
        print(f"{pat:<15} {d:<6} {pm:>+9.3f}% {n:>8}  {label}")

    # Summary
    long_tf = sum(1 for _, pm in long_pats if pm > 0)
    short_tf = sum(1 for _, pm in short_pats if pm < 0)
    print(f"\nSummary:")
    print(f"  LONG trend-following: {long_tf}/{len(long_pats)} "
          f"({long_tf/max(1,len(long_pats))*100:.0f}%)")
    print(f"  SHORT trend-following: {short_tf}/{len(short_pats)} "
          f"({short_tf/max(1,len(short_pats))*100:.0f}%)")

    both_tf = long_tf + short_tf
    total = len(long_pats) + len(short_pats)
    if both_tf > total * 0.6:
        print(f"\n⚠️  MAJORITY TREND-FOLLOWING ({both_tf}/{total} = {both_tf/total*100:.0f}%)")
        print(f"   This explains the entry price inversion: LONGs buy after up, SHORTs sell after down")
    elif both_tf < total * 0.4:
        print(f"\n✅ MAJORITY MEAN-REVERTING ({total-both_tf}/{total})")
    else:
        print(f"\n⚖️  MIXED: {both_tf}/{total} trend-following")

    return {
        'long_premove_avg': np.mean(long_pm),
        'short_premove_avg': np.mean(short_pm),
        'long_trend_following_pct': long_tf / max(1, len(long_pats)) * 100,
        'short_trend_following_pct': short_tf / max(1, len(short_pats)) * 100,
    }


def main():
    print("Entry Price Inversion Study")
    print("=" * 70)

    # Load data
    df = load_data()
    n = len(df)
    print(f"Data: {n} bars ({n/BARS_PER_DAY:.0f} days)")

    opens = df['open'].values
    highs = df['high'].values
    lows = df['low'].values
    closes = df['close'].values

    # Load patterns
    patterns = load_patterns()
    print(f"Patterns: {len(patterns)} ({sum(1 for p in patterns if p['direction']=='LONG')}L + "
          f"{sum(1 for p in patterns if p['direction']=='SHORT')}S)")

    # Find all signals
    signals = find_signals(df, patterns)
    print(f"Total signals: {len(signals)}")

    all_results = {}

    # Phase 1: Aggregate entry prices
    all_results['phase1'] = phase1_aggregate_entry_prices(signals, closes)

    # Phase 2: Per-pattern entry prices
    all_results['phase2'] = phase2_per_pattern_entry_prices(signals, closes)

    # Phase 3: Entry percentile
    all_results['phase3'] = phase3_entry_percentile(signals, closes, highs, lows)

    # Phase 4: Percentile vs outcome
    all_results['phase4'] = phase4_entry_pctile_vs_outcome(signals, opens, highs, lows, closes)

    # Phase 5: Pattern structure
    all_results['phase5'] = phase5_pattern_structure(signals, opens, highs, lows, closes)

    # Final summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)

    p1 = all_results['phase1']
    p3 = all_results['phase3']
    p5 = all_results['phase5']

    print(f"\n1. Entry Price Inversion: LONG ${p1['long_avg']:,.0f} vs SHORT ${p1['short_avg']:,.0f} "
          f"(diff {p1['diff_pct']:+.3f}%)")

    # Check if inversion exists in 24h window
    w288 = p3.get('window_288', {})
    if w288:
        print(f"2. 24h Window Percentile: LONG {w288['long_avg_pctile']:.1f}% / "
              f"SHORT {w288['short_avg_pctile']:.1f}%")
        if w288.get('inverted'):
            print(f"   → CONFIRMED: LONG buys HIGH, SHORT sells LOW within 24h window")
        else:
            print(f"   → NOT inverted in 24h window")

    print(f"3. Pattern Structure:")
    print(f"   LONG pre-move: {p5['long_premove_avg']:+.3f}% "
          f"({'trend-following' if p5['long_premove_avg'] > 0 else 'mean-reverting'})")
    print(f"   SHORT pre-move: {p5['short_premove_avg']:+.3f}% "
          f"({'trend-following' if p5['short_premove_avg'] < 0 else 'mean-reverting'})")
    print(f"   LONG TF: {p5['long_trend_following_pct']:.0f}% | "
          f"SHORT TF: {p5['short_trend_following_pct']:.0f}%")

    # Diagnosis
    print(f"\n{'='*70}")
    print("DIAGNOSIS")
    print(f"{'='*70}")
    if p5['long_premove_avg'] > 0 and p5['short_premove_avg'] < 0:
        print("Both directions are TREND-FOLLOWING:")
        print("  - LONG fires after price rises → buys at local high")
        print("  - SHORT fires after price falls → sells at local low")
        print("  → In a mean-reverting 5m market, this creates adverse selection")
        print("  → FIX: Consider entry timing improvement or post-entry confirmation")
    elif p5['long_premove_avg'] > 0:
        print("LONG is trend-following but SHORT is mean-reverting:")
        print("  → LONG entries at peaks are the primary problem")
    elif p5['short_premove_avg'] < 0:
        print("SHORT is trend-following but LONG is mean-reverting:")
        print("  → SHORT entries at troughs are the primary problem")
    else:
        print("Both directions are mean-reverting — entry price inversion has other cause")

    # Save results
    with open(RESULT_FILE, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {RESULT_FILE}")


if __name__ == '__main__':
    main()
