"""
Pattern Discovery - Exhaustive Search for New Trading Patterns
Searches all 1,728 possible 3-candle patterns and validates promising candidates.

Usage:
    python scripts/analysis/pattern_discovery.py
"""

import pandas as pd
import numpy as np
from itertools import product
from datetime import datetime
from typing import List, Dict, Tuple, Optional
import warnings
import sys
from pathlib import Path

# Import canonical classify_candle from production
sys.path.insert(0, str(Path(__file__).parent.parent))
from scripts.production.pattern_5m.indicators import classify_candle

warnings.filterwarnings('ignore')

# ============================================================
# Configuration
# ============================================================

# 12-type candle classification
CANDLE_TYPES = ['D', 'DF', 'GS', 'H', 'IH', 'ST', 'MU', 'MD', 'BU', 'BD', 'U', 'DN']

# Current production patterns (exclude from discovery)
PRODUCTION_LONG = ['U-BU-U', 'ST-BD-DN']
PRODUCTION_SHORT = ['BD-BD-BD', 'DN-DN-BD', 'MU-ST-DN', 'IH-DN-DN', 'BD-ST-DN', 'BU-U-DN', 'D-DN-BD']
PRODUCTION_PATTERNS = set(PRODUCTION_LONG + PRODUCTION_SHORT)

# Validation thresholds
MIN_TRADES = 20           # Minimum sample size
MIN_WIN_RATE = 55.0       # Minimum win rate %
MIN_EDGE = 0.10           # Minimum edge (expected value)
MIN_WF_PASS = 3           # Minimum WF folds (out of 5)

# TP/SL grid for optimization
TP_GRID = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
SL_GRID = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5]

# Trading parameters
LEVERAGE = 3
FEE_PCT = 0.10  # Round-trip

# ============================================================
# Candle Classification (from production code)
# ============================================================

def classify_candles(df: pd.DataFrame) -> pd.DataFrame:
    """Classify candles into 12 types using production canonical source."""
    df = df.copy()

    # Compute avg_body_20 for normalization
    df['body_abs'] = (df['close'] - df['open']).abs()
    df['avg_body_20'] = df['body_abs'].rolling(20).mean()

    # Use production classify_candle (row, avg_body_20) — 2 args
    types = []
    for i, row in df.iterrows():
        avg_b = df.at[i, 'avg_body_20']
        if pd.isna(avg_b):
            avg_b = 1.0
        types.append(classify_candle(row, avg_b).value)
    df['candle_type'] = types

    # Create pattern string (last 3 candles)
    df['pattern'] = df['candle_type'].shift(2) + '-' + df['candle_type'].shift(1) + '-' + df['candle_type']

    return df


def add_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicators for context analysis."""
    df = df.copy()

    # RSI
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, 1)
    df['rsi'] = 100 - (100 / (1 + rs))

    # EMA for trend
    df['ema20'] = df['close'].ewm(span=20).mean()
    df['trend'] = np.where(df['close'] > df['ema20'], 'UP', 'DN')

    return df


# ============================================================
# Backtesting Functions
# ============================================================

def backtest_pattern(
    df: pd.DataFrame,
    pattern: str,
    direction: str,
    tp_pct: float,
    sl_pct: float,
    leverage: int = LEVERAGE
) -> Dict:
    """Backtest a single pattern with given TP/SL."""

    # Find pattern matches
    signals = df[df['pattern'] == pattern].copy()

    if len(signals) < 5:
        return {'trades': 0, 'wr': 0, 'pnl': 0, 'edge': 0}

    results = []

    for idx in signals.index:
        pos = df.index.get_loc(idx)
        if pos + 1 >= len(df):
            continue

        entry_price = df.iloc[pos + 1]['open']

        # Simulate trade
        for i in range(pos + 1, min(pos + 100, len(df))):
            bar = df.iloc[i]

            if direction == 'LONG':
                tp_price = entry_price * (1 + tp_pct / 100)
                sl_price = entry_price * (1 - sl_pct / 100)

                if bar['high'] >= tp_price:
                    pnl = (tp_pct * leverage) - FEE_PCT
                    results.append({'outcome': 'WIN', 'pnl': pnl})
                    break
                elif bar['low'] <= sl_price:
                    pnl = -(sl_pct * leverage) - FEE_PCT
                    results.append({'outcome': 'LOSS', 'pnl': pnl})
                    break
            else:  # SHORT
                tp_price = entry_price * (1 - tp_pct / 100)
                sl_price = entry_price * (1 + sl_pct / 100)

                if bar['low'] <= tp_price:
                    pnl = (tp_pct * leverage) - FEE_PCT
                    results.append({'outcome': 'WIN', 'pnl': pnl})
                    break
                elif bar['high'] >= sl_price:
                    pnl = -(sl_pct * leverage) - FEE_PCT
                    results.append({'outcome': 'LOSS', 'pnl': pnl})
                    break

    if not results:
        return {'trades': 0, 'wr': 0, 'pnl': 0, 'edge': 0}

    trades = len(results)
    wins = sum(1 for r in results if r['outcome'] == 'WIN')
    wr = (wins / trades) * 100 if trades > 0 else 0
    total_pnl = sum(r['pnl'] for r in results)
    edge = total_pnl / trades if trades > 0 else 0

    return {
        'trades': trades,
        'wr': wr,
        'pnl': total_pnl,
        'edge': edge
    }


def walk_forward_validation(
    df: pd.DataFrame,
    pattern: str,
    direction: str,
    tp_pct: float,
    sl_pct: float,
    n_folds: int = 5
) -> Tuple[int, List[float]]:
    """Run walk-forward validation."""

    fold_size = len(df) // n_folds
    fold_pnls = []

    for i in range(n_folds):
        start_idx = i * fold_size
        end_idx = (i + 1) * fold_size if i < n_folds - 1 else len(df)

        fold_df = df.iloc[start_idx:end_idx].copy()
        fold_df = fold_df.reset_index(drop=True)

        result = backtest_pattern(fold_df, pattern, direction, tp_pct, sl_pct)
        fold_pnls.append(result['pnl'])

    profitable_folds = sum(1 for pnl in fold_pnls if pnl > 0)

    return profitable_folds, fold_pnls


def find_optimal_tpsl(
    df: pd.DataFrame,
    pattern: str,
    direction: str
) -> Tuple[float, float, Dict]:
    """Find optimal TP/SL via grid search."""

    best_score = -999
    best_tp, best_sl = 1.5, 2.0
    best_result = None

    for tp in TP_GRID:
        for sl in SL_GRID:
            result = backtest_pattern(df, pattern, direction, tp, sl)

            if result['trades'] < 10:
                continue

            # Score: prioritize WR and edge
            score = result['wr'] * 0.5 + result['edge'] * 50

            if score > best_score:
                best_score = score
                best_tp, best_sl = tp, sl
                best_result = result

    return best_tp, best_sl, best_result or {'trades': 0, 'wr': 0, 'pnl': 0, 'edge': 0}


# ============================================================
# Pattern Discovery
# ============================================================

def generate_all_patterns() -> List[str]:
    """Generate all 1,728 possible 3-candle patterns."""
    patterns = []
    for c1, c2, c3 in product(CANDLE_TYPES, repeat=3):
        patterns.append(f"{c1}-{c2}-{c3}")
    return patterns


def discover_patterns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Exhaustive search for promising patterns.
    Tests all 1,728 patterns in both LONG and SHORT directions.
    """

    all_patterns = generate_all_patterns()
    print(f"Total patterns to test: {len(all_patterns)} x 2 directions = {len(all_patterns) * 2}")

    results = []

    for i, pattern in enumerate(all_patterns):
        if (i + 1) % 200 == 0:
            print(f"Progress: {i + 1}/{len(all_patterns)} patterns...")

        for direction in ['LONG', 'SHORT']:
            # Skip production patterns
            if pattern in PRODUCTION_PATTERNS:
                continue

            # Quick filter: check sample size first
            signal_count = len(df[df['pattern'] == pattern])
            if signal_count < MIN_TRADES:
                continue

            # Find optimal TP/SL
            opt_tp, opt_sl, opt_result = find_optimal_tpsl(df, pattern, direction)

            if opt_result['trades'] < MIN_TRADES:
                continue

            # Walk-forward validation
            wf_pass, wf_pnls = walk_forward_validation(
                df, pattern, direction, opt_tp, opt_sl
            )

            results.append({
                'pattern': pattern,
                'direction': direction,
                'trades': opt_result['trades'],
                'wr': opt_result['wr'],
                'pnl': opt_result['pnl'],
                'edge': opt_result['edge'],
                'opt_tp': opt_tp,
                'opt_sl': opt_sl,
                'wf_pass': wf_pass,
                'wf_pnls': wf_pnls,
                'is_production': pattern in PRODUCTION_PATTERNS
            })

    results_df = pd.DataFrame(results)
    return results_df


def filter_candidates(results_df: pd.DataFrame) -> pd.DataFrame:
    """Filter and rank promising pattern candidates."""

    # Apply validation criteria
    candidates = results_df[
        (results_df['trades'] >= MIN_TRADES) &
        (results_df['wr'] >= MIN_WIN_RATE) &
        (results_df['edge'] >= MIN_EDGE) &
        (results_df['wf_pass'] >= MIN_WF_PASS) &
        (~results_df['is_production'])
    ].copy()

    # Calculate composite score
    candidates['score'] = (
        candidates['trades'].clip(upper=100) / 100 * 20 +  # Sample size (max 20)
        candidates['wr'] / 100 * 30 +                       # Win rate (max 30)
        candidates['edge'].clip(upper=1.0) * 30 +           # Edge (max 30)
        candidates['wf_pass'] / 5 * 20                      # WF consistency (max 20)
    )

    # Sort by score
    candidates = candidates.sort_values('score', ascending=False)

    return candidates


# ============================================================
# Main Execution
# ============================================================

def main():
    print("=" * 70)
    print("Pattern Discovery - Exhaustive Search for New Trading Patterns")
    print("=" * 70)

    # Load data
    data_path = "data/btc_5m_90days_validation.csv"
    print(f"\nLoading data: {data_path}")

    try:
        df = pd.read_csv(data_path)
    except FileNotFoundError:
        print(f"ERROR: Data file not found: {data_path}")
        return

    print(f"Loaded {len(df)} bars")
    print(f"Period: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")

    # Classify candles
    print("\nClassifying candles...")
    df = classify_candles(df)
    df = add_indicators(df)

    # Show candle type distribution
    type_dist = df['candle_type'].value_counts()
    print("\nCandle type distribution:")
    for ct, count in type_dist.items():
        print(f"  {ct}: {count} ({count/len(df)*100:.1f}%)")

    # Discover patterns
    print("\n" + "=" * 70)
    print("Starting exhaustive pattern search...")
    print("=" * 70)

    results_df = discover_patterns(df)

    if results_df.empty:
        print("No patterns found with sufficient data.")
        return

    print(f"\nTotal patterns tested with sufficient data: {len(results_df)}")

    # Filter candidates
    print("\n" + "=" * 70)
    print("Filtering candidates...")
    print(f"Criteria: trades >= {MIN_TRADES}, WR >= {MIN_WIN_RATE}%, edge >= {MIN_EDGE}, WF >= {MIN_WF_PASS}/5")
    print("=" * 70)

    candidates = filter_candidates(results_df)

    if candidates.empty:
        print("\nNo candidates passed all validation criteria.")
        print("\nRelaxing criteria for analysis...")

        # Show best patterns even if they don't pass
        top_by_wr = results_df.nlargest(10, 'wr')
        print("\nTop 10 by Win Rate:")
        for _, row in top_by_wr.iterrows():
            print(f"  {row['pattern']} {row['direction']}: {row['wr']:.1f}% WR, {row['trades']} trades, WF {row['wf_pass']}/5")

        return results_df

    print(f"\n✅ Found {len(candidates)} promising new patterns!")

    # Display top candidates
    print("\n" + "=" * 70)
    print("TOP NEW PATTERN CANDIDATES")
    print("=" * 70)

    for i, (_, row) in enumerate(candidates.head(20).iterrows(), 1):
        print(f"\n[{i}] {row['pattern']} ({row['direction']})")
        print(f"    Trades: {row['trades']}, WR: {row['wr']:.1f}%, Edge: {row['edge']:.2f}")
        print(f"    Optimal TP/SL: {row['opt_tp']:.1f}% / {row['opt_sl']:.1f}%")
        print(f"    WF: {row['wf_pass']}/5, Score: {row['score']:.1f}")
        print(f"    PnL: {row['pnl']:.2f}%")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"results/pattern_discovery_{timestamp}.csv"
    candidates.to_csv(output_file, index=False)
    print(f"\n✅ Results saved to: {output_file}")

    # Generate recommended additions
    print("\n" + "=" * 70)
    print("RECOMMENDED NEW PATTERNS FOR PRODUCTION")
    print("=" * 70)

    # Top candidates for each direction
    long_candidates = candidates[candidates['direction'] == 'LONG'].head(5)
    short_candidates = candidates[candidates['direction'] == 'SHORT'].head(5)

    print("\nNEW LONG PATTERNS:")
    if long_candidates.empty:
        print("  (No new LONG patterns found)")
    else:
        for _, row in long_candidates.iterrows():
            print(f"  '{row['pattern']}': ({row['opt_tp']}, {row['opt_sl']}),  # WR {row['wr']:.1f}%, WF {row['wf_pass']}/5")

    print("\nNEW SHORT PATTERNS:")
    if short_candidates.empty:
        print("  (No new SHORT patterns found)")
    else:
        for _, row in short_candidates.iterrows():
            print(f"  '{row['pattern']}': ({row['opt_tp']}, {row['opt_sl']}),  # WR {row['wr']:.1f}%, WF {row['wf_pass']}/5")

    return candidates


if __name__ == "__main__":
    import os
    os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    results = main()
