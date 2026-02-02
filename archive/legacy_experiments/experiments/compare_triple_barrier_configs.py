"""
Comprehensive Triple Barrier Configuration Comparison

Tests multiple parameter combinations to find optimal Exit labeling approach:
1. Barrier ratio variations (1:1, 1.5:1, 2:1, 3:1)
2. Risk scoring methods (binary, time-weighted, P&L-weighted)
3. Percentile cutoffs (10th, 15th, 20th)

Goal: Find configuration that achieves 10-20% exit rate target
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from tqdm import tqdm
from itertools import product

# Paths
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data"
FEATURES_DIR = DATA_DIR / "features"
LABELS_DIR = DATA_DIR / "labels"
LABELS_DIR.mkdir(parents=True, exist_ok=True)

# Input file
FEATURES_FILE = FEATURES_DIR / "BTCUSDT_5m_features_90days_complete_20251106_164542.csv"

# Output files
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_COMPARISON = LABELS_DIR / f"triple_barrier_comparison_{TIMESTAMP}.csv"

# Configuration Space
BARRIER_CONFIGS = [
    # (stop_multiplier, profit_multiplier, description)
    (1.0, 1.0, "1:1 R/R (1.0 ATR)"),
    (1.5, 1.5, "1:1 R/R (1.5 ATR)"),
    (1.0, 1.5, "1.5:1 R/R"),
    (1.0, 2.0, "2:1 R/R"),
    (0.5, 1.5, "3:1 R/R"),
]

SCORING_METHODS = [
    "binary",        # Current: stop=10, timeout=5, profit=0
    "time_weighted", # Earlier hits = higher risk
    "pnl_weighted",  # Larger loss = higher risk
]

PERCENTILES = [10, 15, 20]  # Worst N% labeled as exits

TIME_LIMIT = 60  # 60 candles = 5 hours


def calculate_risk_score(outcome, hit_index, time_limit, pnl_pct, method='binary'):
    """
    Calculate risk score based on outcome and method

    Methods:
    - binary: stop=10, timeout=5, profit=0
    - time_weighted: score * (1 - time_to_hit/time_limit)
    - pnl_weighted: score based on P&L magnitude
    """
    if method == 'binary':
        if outcome == 'stop':
            return 10.0
        elif outcome == 'timeout':
            return 5.0
        else:  # profit
            return 0.0

    elif method == 'time_weighted':
        # Base score
        if outcome == 'stop':
            base_score = 10.0
        elif outcome == 'timeout':
            base_score = 5.0
        else:  # profit
            return 0.0

        # Weight by how quickly it hit (earlier = worse)
        if hit_index >= 0:
            time_factor = 1.0 - (hit_index / time_limit)
            return base_score * time_factor
        else:
            return base_score

    elif method == 'pnl_weighted':
        # Score based on P&L magnitude
        if outcome == 'profit':
            return 0.0
        elif outcome == 'timeout':
            # Neutral timeout = medium risk
            return 5.0
        else:  # stop
            # Larger loss = higher risk (scale 5-15)
            loss_magnitude = abs(pnl_pct) * 100  # Convert to percentage
            score = 5.0 + min(loss_magnitude * 2, 10.0)  # Cap at 15
            return score


def check_barriers(entry_price, future_prices, profit_barrier, stop_barrier, side='LONG'):
    """
    Simulate forward to find which barrier hits first

    Returns:
        outcome: 'profit', 'stop', or 'timeout'
        hit_index: candle index where barrier was hit (or -1 if timeout)
        pnl_pct: P&L percentage at hit point
    """
    for i, price in enumerate(future_prices):
        if side == 'LONG':
            if price >= profit_barrier:
                pnl_pct = (price - entry_price) / entry_price
                return 'profit', i, pnl_pct
            elif price <= stop_barrier:
                pnl_pct = (price - entry_price) / entry_price
                return 'stop', i, pnl_pct
        else:  # SHORT
            if price <= profit_barrier:
                pnl_pct = (entry_price - price) / entry_price
                return 'profit', i, pnl_pct
            elif price >= stop_barrier:
                pnl_pct = (entry_price - price) / entry_price
                return 'stop', i, pnl_pct

    # Timeout
    final_price = future_prices[-1] if len(future_prices) > 0 else entry_price
    if side == 'LONG':
        pnl_pct = (final_price - entry_price) / entry_price
    else:
        pnl_pct = (entry_price - final_price) / entry_price

    return 'timeout', -1, pnl_pct


def test_configuration(df, side, stop_mult, profit_mult, scoring_method, percentile):
    """
    Test a single Triple Barrier configuration

    Returns:
        dict with configuration results
    """
    df = df.copy()

    # Calculate risk scores for all candles
    risk_scores = []
    outcomes = []
    pnl_list = []

    for i in range(len(df) - TIME_LIMIT):
        current_price = df.iloc[i]['close']
        current_atr = df.iloc[i].get('atr', current_price * 0.01)

        # Calculate dynamic barriers
        if side == 'LONG':
            stop_barrier = current_price - (current_atr * stop_mult)
            profit_barrier = current_price + (current_atr * profit_mult)
        else:  # SHORT
            stop_barrier = current_price + (current_atr * stop_mult)
            profit_barrier = current_price - (current_atr * profit_mult)

        # Get future prices
        future_prices = df.iloc[i+1:i+1+TIME_LIMIT]['close'].values

        # Check which barrier hits first
        outcome, hit_index, pnl_pct = check_barriers(
            current_price, future_prices,
            profit_barrier, stop_barrier,
            side
        )

        # Calculate risk score
        risk_score = calculate_risk_score(
            outcome, hit_index, TIME_LIMIT, pnl_pct, scoring_method
        )

        risk_scores.append(risk_score)
        outcomes.append(outcome)
        pnl_list.append(pnl_pct)

    # Last TIME_LIMIT candles: cannot simulate forward
    for _ in range(TIME_LIMIT):
        risk_scores.append(calculate_risk_score('timeout', -1, TIME_LIMIT, 0.0, scoring_method))
        outcomes.append('timeout')
        pnl_list.append(0.0)

    # Calculate quantile threshold
    risk_threshold = np.percentile(risk_scores, 100 - percentile)

    # Apply quantile-based labeling
    exit_labels = [1 if score >= risk_threshold else 0 for score in risk_scores]

    # Statistics
    total = len(exit_labels)
    exits = sum(exit_labels)
    exit_pct = (exits / total) * 100

    # Outcome statistics
    profit_count = outcomes.count('profit')
    stop_count = outcomes.count('stop')
    timeout_count = outcomes.count('timeout')

    # Average P&L
    avg_pnl = np.mean(pnl_list) * 100

    # Risk score distribution
    risk_q25 = np.percentile(risk_scores, 25)
    risk_median = np.percentile(risk_scores, 50)
    risk_q75 = np.percentile(risk_scores, 75)

    return {
        'side': side,
        'stop_mult': stop_mult,
        'profit_mult': profit_mult,
        'rr_ratio': f"{profit_mult/stop_mult:.1f}:1",
        'scoring_method': scoring_method,
        'percentile': percentile,
        'exit_rate': exit_pct,
        'profit_hit_pct': profit_count / total * 100,
        'stop_hit_pct': stop_count / total * 100,
        'timeout_pct': timeout_count / total * 100,
        'avg_pnl': avg_pnl,
        'risk_threshold': risk_threshold,
        'risk_q25': risk_q25,
        'risk_median': risk_median,
        'risk_q75': risk_q75,
    }


def main():
    print("="*80)
    print("TRIPLE BARRIER CONFIGURATION COMPARISON")
    print("="*80)

    print(f"\n📂 Input: {FEATURES_FILE.name}")
    print(f"📤 Output: {OUTPUT_COMPARISON.name}")

    print(f"\n⚙️ CONFIGURATION SPACE:")
    print(f"   Barrier Configs: {len(BARRIER_CONFIGS)}")
    print(f"   Scoring Methods: {len(SCORING_METHODS)}")
    print(f"   Percentiles: {len(PERCENTILES)}")
    print(f"   Total Combinations: {len(BARRIER_CONFIGS) * len(SCORING_METHODS) * len(PERCENTILES) * 2} (×2 for LONG/SHORT)")

    # Load features
    print(f"\n{'='*80}")
    print("LOADING DATA")
    print(f"{'='*80}")

    print(f"\n📖 Loading features...")
    df = pd.read_csv(FEATURES_FILE)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    print(f"   Rows: {len(df):,}")
    print(f"   Period: {df['timestamp'].min()} to {df['timestamp'].max()}")

    # Check for ATR column
    if 'atr' not in df.columns:
        print(f"\n⚠️  Warning: 'atr' column not found, will use 1% default")

    # Run all configurations
    print(f"\n{'='*80}")
    print("TESTING CONFIGURATIONS")
    print(f"{'='*80}")

    results = []

    # Generate all combinations
    total_configs = len(BARRIER_CONFIGS) * len(SCORING_METHODS) * len(PERCENTILES) * 2

    with tqdm(total=total_configs, desc="Testing configurations") as pbar:
        for side in ['LONG', 'SHORT']:
            for (stop_mult, profit_mult, desc) in BARRIER_CONFIGS:
                for scoring_method in SCORING_METHODS:
                    for percentile in PERCENTILES:
                        # Test configuration
                        result = test_configuration(
                            df, side, stop_mult, profit_mult,
                            scoring_method, percentile
                        )
                        results.append(result)
                        pbar.update(1)

    # Convert to DataFrame
    df_results = pd.DataFrame(results)

    # Save results
    print(f"\n{'='*80}")
    print("SAVING RESULTS")
    print(f"{'='*80}")

    df_results.to_csv(OUTPUT_COMPARISON, index=False)
    print(f"\n✅ Saved: {OUTPUT_COMPARISON}")

    # Analysis
    print(f"\n{'='*80}")
    print("ANALYSIS")
    print(f"{'='*80}")

    # Target: 10-20% exit rate
    target_min = 10.0
    target_max = 20.0

    # Filter configurations within target range
    df_target = df_results[
        (df_results['exit_rate'] >= target_min) &
        (df_results['exit_rate'] <= target_max)
    ]

    print(f"\n📊 Configurations within target (10-20% exit rate): {len(df_target)}/{len(df_results)}")

    if len(df_target) > 0:
        print(f"\n🎯 TOP 5 CONFIGURATIONS (Closest to 15% midpoint):")
        df_target['distance_to_midpoint'] = abs(df_target['exit_rate'] - 15.0)
        df_top = df_target.sort_values('distance_to_midpoint').head(5)

        for idx, row in df_top.iterrows():
            print(f"\n   Config #{idx+1}:")
            print(f"      Side: {row['side']}")
            print(f"      Barriers: {row['rr_ratio']} ({row['stop_mult']} ATR stop, {row['profit_mult']} ATR profit)")
            print(f"      Scoring: {row['scoring_method']}")
            print(f"      Percentile: {row['percentile']}th")
            print(f"      Exit Rate: {row['exit_rate']:.2f}%")
            print(f"      Outcomes: Profit {row['profit_hit_pct']:.1f}%, Stop {row['stop_hit_pct']:.1f}%, Timeout {row['timeout_pct']:.1f}%")
            print(f"      Avg P&L: {row['avg_pnl']:.2f}%")
    else:
        print(f"\n⚠️  No configurations achieved target 10-20% exit rate")
        print(f"\n   Closest configurations:")
        df_results['distance_to_range'] = df_results['exit_rate'].apply(
            lambda x: min(abs(x - target_min), abs(x - target_max))
        )
        df_closest = df_results.sort_values('distance_to_range').head(5)

        for idx, row in df_closest.iterrows():
            print(f"\n   Config #{idx+1}:")
            print(f"      Side: {row['side']}")
            print(f"      Barriers: {row['rr_ratio']} ({row['stop_mult']} ATR stop, {row['profit_mult']} ATR profit)")
            print(f"      Scoring: {row['scoring_method']}")
            print(f"      Percentile: {row['percentile']}th")
            print(f"      Exit Rate: {row['exit_rate']:.2f}%")
            print(f"      Outcomes: Profit {row['profit_hit_pct']:.1f}%, Stop {row['stop_hit_pct']:.1f}%, Timeout {row['timeout_pct']:.1f}%")

    # Summary statistics
    print(f"\n{'='*80}")
    print("SUMMARY STATISTICS")
    print(f"{'='*80}")

    print(f"\n📊 Exit Rate Distribution:")
    print(f"   Min: {df_results['exit_rate'].min():.2f}%")
    print(f"   25th percentile: {df_results['exit_rate'].quantile(0.25):.2f}%")
    print(f"   Median: {df_results['exit_rate'].median():.2f}%")
    print(f"   75th percentile: {df_results['exit_rate'].quantile(0.75):.2f}%")
    print(f"   Max: {df_results['exit_rate'].max():.2f}%")

    print(f"\n📊 Barrier Outcome Averages:")
    print(f"   Profit Hit: {df_results['profit_hit_pct'].mean():.2f}%")
    print(f"   Stop Hit: {df_results['stop_hit_pct'].mean():.2f}%")
    print(f"   Timeout: {df_results['timeout_pct'].mean():.2f}%")

    print(f"\n{'='*80}")
    print("COMPARISON COMPLETE")
    print(f"{'='*80}")

    print(f"\n📋 NEXT STEPS:")
    print(f"   1. Review {OUTPUT_COMPARISON.name} for detailed results")
    print(f"   2. Select optimal configuration based on:")
    print(f"      - Exit rate closest to 10-20% target")
    print(f"      - Good balance of profit/stop outcomes")
    print(f"      - Reasonable average P&L")
    print(f"   3. Generate final Exit labels with selected config")
    print(f"   4. Retrain Exit models with 171 features")


if __name__ == "__main__":
    main()
