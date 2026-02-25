#!/usr/bin/env python3
"""
MTF Direction Study — Phase 1 & 2
==================================
Multi-Timeframe Direction Filter Research

Phase 1: Aggregate 5m → 15m / 1h + classify (production classify_candle)
Phase 2: 1h 3-candle pattern directional accuracy measurement

Standard Research Protocol:
- Production classify_candle import
- LEVERAGE = 3, FEE_PCT = 0.10%
- Entry at next bar open
- Expanding Window WF (Phase 5, later)

Usage:
  cd bingx_rl_trading_bot
  python scripts/analysis/mtf_direction_study.py
  python scripts/analysis/mtf_direction_study.py --horizons 1,3,6,12,24
  python scripts/analysis/mtf_direction_study.py --min-occ 30 --min-acc 0.53
"""

import sys
import os
import json
import argparse
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

# Production import
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.production.pattern_5m.indicators import classify_candle
from scripts.production.pattern_5m.constants import AVG_BODY_WINDOW

# ============================================================
# Constants
# ============================================================
LEVERAGE = 3
FEE_PCT = 0.10  # 0.05% * 2 sides
MAX_BARS_5M = 288  # 24h at 5m

DEFAULT_HORIZONS_1H = [1, 3, 6, 12, 24]  # hours
MIN_OCCURRENCES = 50
MIN_ACCURACY = 0.55
SIGNIFICANCE_LEVEL = 0.01

BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / 'data'
RESULTS_DIR = BASE_DIR / 'results'


# ============================================================
# Phase 1: Data Aggregation
# ============================================================

def load_5m_data(filepath=None):
    """Load 5m OHLCV data."""
    if filepath is None:
        filepath = DATA_DIR / 'btc_5m_720days_binance.csv'
    df = pd.read_csv(filepath, parse_dates=['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    days = (df['timestamp'].max() - df['timestamp'].min()).days
    print(f"[Phase 1] Loaded 5m: {len(df):,} rows, {days}d "
          f"({df['timestamp'].min().date()} → {df['timestamp'].max().date()})")
    return df


def aggregate_to_higher_tf(df_5m, tf_minutes):
    """Aggregate 5m candles to higher timeframe OHLCV."""
    df = df_5m[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
    df = df.set_index('timestamp')

    rule = f'{tf_minutes}min'
    agg = df.resample(rule).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()

    agg = agg.reset_index()
    print(f"[Phase 1] Aggregated to {tf_minutes}m: {len(agg):,} rows")
    return agg


def classify_dataframe(df):
    """Apply production classify_candle to any OHLCV dataframe.

    Replicates production indicators.py logic:
    1. body, body_abs, avg_body_20
    2. classify_candle() per row → type_code
    3. Build pattern_3 from 3 consecutive type_codes
    """
    df = df.copy()

    # Body calculations
    df['body'] = df['close'] - df['open']
    df['body_abs'] = df['body'].abs()
    df['avg_body_20'] = df['body_abs'].rolling(AVG_BODY_WINDOW, min_periods=1).mean()

    # Classify each candle (production function)
    avg_body_vals = df['avg_body_20'].values
    type_codes = []
    for i, row in enumerate(df.itertuples(index=False)):
        avg_b = avg_body_vals[i]
        if pd.isna(avg_b):
            avg_b = 1.0
        ct = classify_candle(row, avg_b)
        type_codes.append(ct.value)  # e.g. 'BU', 'DN', 'ST'

    df['type_code'] = type_codes

    # Build 3-candle patterns
    tc = df['type_code'].values
    patterns = [None, None] + [
        f"{tc[i-2]}-{tc[i-1]}-{tc[i]}" for i in range(2, len(tc))
    ]
    df['pattern_3'] = patterns

    return df


def verify_aggregation(df_5m, df_1h, n_checks=5):
    """Verify OHLCV aggregation integrity with random samples."""
    rng = np.random.default_rng(42)
    valid_idx = range(10, len(df_1h) - 10)
    checks = rng.choice(valid_idx, size=min(n_checks, len(valid_idx)), replace=False)

    for idx in checks:
        ts = df_1h['timestamp'].iloc[idx]
        mask = (df_5m['timestamp'] >= ts) & (df_5m['timestamp'] < ts + pd.Timedelta(hours=1))
        sub = df_5m.loc[mask]
        if len(sub) == 0:
            continue
        assert abs(sub['high'].max() - df_1h['high'].iloc[idx]) < 0.01, \
            f"High mismatch at {ts}"
        assert abs(sub['low'].min() - df_1h['low'].iloc[idx]) < 0.01, \
            f"Low mismatch at {ts}"
        assert abs(sub['open'].iloc[0] - df_1h['open'].iloc[idx]) < 0.01, \
            f"Open mismatch at {ts}"

    print(f"[Phase 1] Aggregation integrity: {n_checks} samples verified")


def merge_1h_pattern_onto_5m(df_5m, df_1h):
    """Merge completed 1h pattern onto 5m dataframe.

    A 1h candle at 14:00 covers 14:00-14:55 (12 × 5m candles).
    It's completed at 14:55 (close). The pattern_3 formed from it
    is available from the NEXT hour (15:00 onward).
    So: 5m candles from 15:00 get the 14:00 1h pattern.
    """
    df_1h_lookup = df_1h[['timestamp', 'pattern_3', 'type_code']].copy()
    df_1h_lookup = df_1h_lookup.rename(columns={
        'pattern_3': 'pattern_3_1h',
        'type_code': 'type_code_1h'
    })
    # Pattern available after 1h candle closes
    df_1h_lookup['available_from'] = df_1h_lookup['timestamp'] + pd.Timedelta(hours=1)

    df_out = df_5m.sort_values('timestamp').copy()
    df_1h_sorted = df_1h_lookup.sort_values('available_from')

    df_out = pd.merge_asof(
        df_out,
        df_1h_sorted[['available_from', 'pattern_3_1h', 'type_code_1h']],
        left_on='timestamp',
        right_on='available_from',
        direction='backward'
    )

    n_mapped = df_out['pattern_3_1h'].notna().sum()
    print(f"[Phase 1] 5m→1h mapping: {n_mapped:,}/{len(df_out):,} "
          f"({n_mapped/len(df_out)*100:.1f}%) rows with 1h pattern")
    return df_out


# ============================================================
# Phase 2: 1h Directional Accuracy
# ============================================================

def measure_directional_accuracy(df_1h, horizons_hours):
    """Measure directional accuracy for each 1h 3-candle pattern.

    For each pattern occurrence at bar i:
      For each horizon h (in 1h bars):
        future_close = close[i + h]
        move = (future_close - close[i]) / close[i]
        direction = sign(move)

    Returns: {pattern: {horizon: {up, down, total, moves}}}
    """
    close = df_1h['close'].values
    patterns = df_1h['pattern_3'].values
    n = len(df_1h)

    results = {}
    for i in range(2, n):
        pat = patterns[i]
        if not pat or pd.isna(pat):
            continue

        if pat not in results:
            results[pat] = {
                h: {'up': 0, 'down': 0, 'total': 0, 'moves': []}
                for h in horizons_hours
            }

        entry_close = close[i]
        if entry_close <= 0:
            continue

        for h in horizons_hours:
            future_idx = i + h
            if future_idx >= n:
                continue

            future_close = close[future_idx]
            move_pct = (future_close - entry_close) / entry_close * 100

            results[pat][h]['total'] += 1
            results[pat][h]['moves'].append(move_pct)
            if future_close > entry_close:
                results[pat][h]['up'] += 1
            elif future_close < entry_close:
                results[pat][h]['down'] += 1

    return results


def binomial_p_value(successes, total):
    """One-sided binomial test: P(X >= successes | n=total, p=0.5)."""
    from scipy.stats import binom
    # P(X >= successes) = 1 - P(X <= successes - 1)
    p = 1 - binom.cdf(successes - 1, total, 0.5)
    return p


def analyze_directional_results(results, min_occ, min_acc, sig_level):
    """Analyze and rank directional patterns."""
    analysis = []

    for pat, horizons in results.items():
        for h, data in horizons.items():
            total = data['total']
            if total < min_occ:
                continue

            up_rate = data['up'] / total
            down_rate = data['down'] / total

            if up_rate >= down_rate:
                best_dir = 'LONG'
                accuracy = up_rate
                successes = data['up']
            else:
                best_dir = 'SHORT'
                accuracy = down_rate
                successes = data['down']

            p_value = binomial_p_value(successes, total)

            moves = np.array(data['moves'])
            avg_move = np.mean(moves)
            # For SHORT direction, a negative avg_move is good
            directional_move = avg_move if best_dir == 'LONG' else -avg_move

            analysis.append({
                'pattern': pat,
                'horizon_h': h,
                'direction': best_dir,
                'accuracy': round(accuracy * 100, 2),
                'total': total,
                'up': data['up'],
                'down': data['down'],
                'avg_move_pct': round(avg_move, 4),
                'dir_move_pct': round(directional_move, 4),
                'p_value': round(p_value, 6),
                'significant': p_value < sig_level,
                'strong': accuracy >= min_acc and p_value < sig_level
            })

    return sorted(analysis, key=lambda x: (-x['accuracy'], -x['total']))


def print_phase2_report(analysis, horizons_hours, min_occ, min_acc, sig_level):
    """Print Phase 2 report and return summary."""
    print("\n" + "=" * 80)
    print("PHASE 2 RESULTS: 1H PATTERN DIRECTIONAL ACCURACY")
    print("=" * 80)

    # --- Summary by horizon ---
    horizon_summary = {}
    for h in horizons_hours:
        h_results = [a for a in analysis if a['horizon_h'] == h]
        sig_h = [a for a in h_results if a['significant']]
        strong_h = [a for a in h_results if a['strong']]
        avg_acc = np.mean([a['accuracy'] for a in h_results]) if h_results else 0
        max_acc = max([a['accuracy'] for a in h_results]) if h_results else 0
        horizon_summary[h] = {
            'total_patterns': len(h_results),
            'significant': len(sig_h),
            'strong': len(strong_h),
            'avg_accuracy': round(avg_acc, 1),
            'max_accuracy': round(max_acc, 1)
        }

    print(f"\n{'Horizon':>8} | {'Tested':>6} | {'Sig(p<{:.2f})'.format(sig_level):>12} "
          f"| {'Strong(>{:.0f}%)'.format(min_acc*100):>12} | {'AvgAcc':>7} | {'MaxAcc':>7}")
    print("-" * 65)
    for h in horizons_hours:
        s = horizon_summary[h]
        print(f"{h:>6}h | {s['total_patterns']:>6} | {s['significant']:>12} "
              f"| {s['strong']:>12} | {s['avg_accuracy']:>6.1f}% | {s['max_accuracy']:>6.1f}%")

    # --- Strong patterns detail ---
    strong = [a for a in analysis if a['strong']]
    if strong:
        print(f"\n--- STRONG DIRECTIONAL PATTERNS "
              f"(acc > {min_acc*100:.0f}%, p < {sig_level}) ---")
        print(f"{'Pattern':>15} | {'Dir':>5} | {'Hr':>3} | {'Acc%':>6} "
              f"| {'N':>5} | {'Up':>4} | {'Dn':>4} | {'AvgMove%':>9} | {'p-val':>8}")
        print("-" * 80)
        for a in strong[:40]:
            print(f"{a['pattern']:>15} | {a['direction']:>5} | {a['horizon_h']:>2}h "
                  f"| {a['accuracy']:>5.1f}% | {a['total']:>5} | {a['up']:>4} "
                  f"| {a['down']:>4} | {a['avg_move_pct']:>+8.3f}% | {a['p_value']:>8.4f}")
    else:
        print(f"\n!! NO STRONG PATTERNS FOUND "
              f"(accuracy > {min_acc*100:.0f}% & p < {sig_level})")

    # --- Unique strong patterns (best horizon per pattern) ---
    unique_strong = {}
    for a in strong:
        p = a['pattern']
        if p not in unique_strong or a['accuracy'] > unique_strong[p]['accuracy']:
            unique_strong[p] = a

    print(f"\nUnique strong patterns: {len(unique_strong)}")
    if unique_strong:
        by_dir = {}
        for p, a in unique_strong.items():
            d = a['direction']
            by_dir.setdefault(d, []).append(a)
        for d in ['LONG', 'SHORT']:
            pats = by_dir.get(d, [])
            print(f"  {d}: {len(pats)} patterns")

    return horizon_summary, strong, unique_strong


def go_no_go_decision(horizon_summary, unique_strong):
    """Phase 2 Go/No-Go decision."""
    print("\n" + "=" * 80)
    print("GO / NO-GO DECISION")
    print("=" * 80)

    # Find best horizon by strong count
    best_h = max(horizon_summary.items(), key=lambda x: x[1]['strong'])
    best_horizon = best_h[0]
    n_strong = best_h[1]['strong']
    n_unique = len(unique_strong)

    if n_unique >= 10:
        decision = "GO"
        msg = (f"GO — {n_unique} unique strong patterns "
               f"(best horizon: {best_horizon}h with {n_strong} entries)")
        action = "Proceed to Phase 3: MTF Filter Backtest"
    elif n_unique >= 3:
        decision = "CONDITIONAL_GO"
        msg = (f"CONDITIONAL GO — {n_unique} unique strong patterns "
               f"(limited but testable)")
        action = ("Proceed to Phase 3 with caution. "
                  "Also test 15m patterns as alternative.")
    else:
        decision = "NO_GO"
        msg = f"NO-GO — Only {n_unique} unique strong patterns"
        action = ("1h patterns lack directional power. "
                  "Try: (a) 15m patterns, (b) relax thresholds, "
                  "(c) different pattern definition.")

    print(f"\n  Decision: {decision}")
    print(f"  Reason: {msg}")
    print(f"  Action: {action}")

    return decision, best_horizon


# ============================================================
# Phase 1h Distribution Analysis
# ============================================================

def analyze_1h_type_distribution(df_1h):
    """Analyze 1h candle type and pattern distributions."""
    print("\n--- 1H CANDLE TYPE DISTRIBUTION ---")
    tc = df_1h['type_code'].value_counts()
    print(tc.to_string())

    print(f"\n--- 1H PATTERN DISTRIBUTION (top 20) ---")
    pc = df_1h['pattern_3'].value_counts()
    print(f"Total unique: {len(pc)}")
    print(f"Patterns >= 50 occurrences: {(pc >= 50).sum()}")
    print(f"Patterns >= 100 occurrences: {(pc >= 100).sum()}")
    print(pc.head(20).to_string())

    return tc, pc


# ============================================================
# Main
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(description='MTF Direction Study — Phase 1 & 2')
    parser.add_argument('--data', type=str, default=None,
                        help='5m data CSV path (default: btc_5m_720days_binance.csv)')
    parser.add_argument('--horizons', type=str, default='1,3,6,12,24',
                        help='Comma-separated prediction horizons in hours')
    parser.add_argument('--min-occ', type=int, default=MIN_OCCURRENCES,
                        help=f'Min pattern occurrences (default: {MIN_OCCURRENCES})')
    parser.add_argument('--min-acc', type=float, default=MIN_ACCURACY,
                        help=f'Min directional accuracy (default: {MIN_ACCURACY})')
    parser.add_argument('--sig-level', type=float, default=SIGNIFICANCE_LEVEL,
                        help=f'Significance level (default: {SIGNIFICANCE_LEVEL})')
    parser.add_argument('--save-data', action='store_true',
                        help='Save aggregated CSV files to data/')
    return parser.parse_args()


def main():
    args = parse_args()
    horizons = [int(h) for h in args.horizons.split(',')]

    print("=" * 80)
    print("MTF DIRECTION STUDY — Phase 1 & 2")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print(f"Horizons: {horizons}h | Min occurrences: {args.min_occ} "
          f"| Min accuracy: {args.min_acc*100:.0f}% | p < {args.sig_level}")
    print("=" * 80)

    # ================================================================
    # PHASE 1: DATA PREPARATION
    # ================================================================
    print("\n" + "=" * 80)
    print("PHASE 1: DATA AGGREGATION + CLASSIFICATION")
    print("=" * 80)

    # Load 5m
    df_5m = load_5m_data(args.data)

    # Aggregate to 1h
    df_1h = aggregate_to_higher_tf(df_5m, 60)

    # Classify 1h candles (production classify_candle)
    print("[Phase 1] Classifying 1h candles...")
    df_1h = classify_dataframe(df_1h)

    # Verify aggregation
    verify_aggregation(df_5m, df_1h)

    # 1h distribution analysis
    tc_dist, pat_dist = analyze_1h_type_distribution(df_1h)

    # Aggregate to 15m (for future use / alternative)
    df_15m = aggregate_to_higher_tf(df_5m, 15)
    print("[Phase 1] Classifying 15m candles...")
    df_15m = classify_dataframe(df_15m)

    # Merge 1h pattern onto 5m
    print("[Phase 1] Merging 1h patterns onto 5m data...")
    df_5m_mtf = merge_1h_pattern_onto_5m(df_5m, df_1h)

    # Optionally save aggregated data
    if args.save_data:
        df_1h.to_csv(DATA_DIR / 'btc_1h_720days.csv', index=False)
        df_15m.to_csv(DATA_DIR / 'btc_15m_720days.csv', index=False)
        df_5m_mtf.to_csv(DATA_DIR / 'btc_5m_720days_mtf.csv', index=False)
        print("[Phase 1] Saved aggregated CSV files to data/")

    # ================================================================
    # PHASE 2: 1H DIRECTIONAL ACCURACY
    # ================================================================
    print("\n" + "=" * 80)
    print("PHASE 2: 1H PATTERN DIRECTIONAL ACCURACY")
    print("=" * 80)

    # Measure directional accuracy
    print("[Phase 2] Measuring directional accuracy across horizons...")
    dir_results = measure_directional_accuracy(df_1h, horizons)

    # Analyze
    analysis = analyze_directional_results(
        dir_results, args.min_occ, args.min_acc, args.sig_level
    )

    # Report
    horizon_summary, strong_list, unique_strong = print_phase2_report(
        analysis, horizons, args.min_occ, args.min_acc, args.sig_level
    )

    # Go/No-Go
    decision, best_horizon = go_no_go_decision(horizon_summary, unique_strong)

    # ================================================================
    # SAVE RESULTS
    # ================================================================
    output = {
        'study': 'mtf_direction_study',
        'date': datetime.now().isoformat(),
        'phases': ['phase_1', 'phase_2'],
        'parameters': {
            'horizons_hours': horizons,
            'min_occurrences': args.min_occ,
            'min_accuracy': args.min_acc,
            'significance_level': args.sig_level,
            'leverage': LEVERAGE,
            'fee_pct': FEE_PCT,
        },
        'phase_1': {
            '5m_rows': len(df_5m),
            '1h_rows': len(df_1h),
            '15m_rows': len(df_15m),
            '5m_mtf_rows': len(df_5m_mtf),
            '1h_unique_patterns': len(pat_dist),
            '1h_patterns_min_occ': int((pat_dist >= args.min_occ).sum()),
            '1h_type_distribution': tc_dist.to_dict(),
        },
        'phase_2': {
            'horizon_summary': {
                str(h): info for h, info in horizon_summary.items()
            },
            'total_strong': len(strong_list),
            'unique_strong': len(unique_strong),
            'strong_patterns': [
                {k: v for k, v in a.items() if k != 'moves'}
                for a in strong_list
            ],
            'unique_strong_detail': {
                pat: {k: v for k, v in info.items() if k != 'moves'}
                for pat, info in unique_strong.items()
            },
        },
        'decision': {
            'go_no_go': decision,
            'best_horizon': best_horizon,
        }
    }

    out_file = RESULTS_DIR / 'mtf_direction_study.json'
    with open(out_file, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved: {out_file}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"  1h data: {len(df_1h):,} candles ({len(pat_dist)} unique patterns)")
    print(f"  Tested: {(pat_dist >= args.min_occ).sum()} patterns "
          f"(>= {args.min_occ} occurrences)")
    print(f"  Strong: {len(unique_strong)} unique patterns "
          f"(acc > {args.min_acc*100:.0f}%, p < {args.sig_level})")
    print(f"  Best horizon: {best_horizon}h")
    print(f"  Decision: {decision}")
    print("=" * 80)

    return output


if __name__ == '__main__':
    main()
