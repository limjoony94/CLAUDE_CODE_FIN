"""
Diagnostic Tool: Analyze Each Labeling Criterion Independently

Purpose: Understand why multi-criteria approach generates 0 labels

Approach:
1. Test each criterion independently
2. Measure how restrictive each criterion is
3. Test combinations with OR logic
4. Find optimal balance

Author: Claude Code
Date: 2025-10-16
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import pickle

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.production.train_xgboost_improved_v3_phase2 import calculate_features
from scripts.production.advanced_technical_features import AdvancedTechnicalFeatures
from src.features.sell_signal_features import SellSignalFeatures

DATA_DIR = PROJECT_ROOT / "data" / "historical"
MODELS_DIR = PROJECT_ROOT / "models"


def load_data():
    """Load and prepare data"""
    print("Loading data...")
    df = pd.read_csv(DATA_DIR / "BTCUSDT_5m_max.csv")
    df = calculate_features(df)
    adv = AdvancedTechnicalFeatures(lookback_sr=50, lookback_trend=20)
    df = adv.calculate_all_features(df)
    sell = SellSignalFeatures()
    df = sell.calculate_all_features(df)
    df = df.ffill().dropna()
    print(f"✅ {len(df):,} candles loaded\n")
    return df


def simulate_trades(df, threshold=0.70):
    """Simulate LONG trades for labeling"""
    print(f"Simulating LONG trades (threshold={threshold})...")

    # Load LONG entry model
    with open(MODELS_DIR / "xgboost_v4_phase4_advanced_lookahead3_thresh0.pkl", 'rb') as f:
        model = pickle.load(f)
    with open(MODELS_DIR / "xgboost_v4_phase4_advanced_lookahead3_thresh0_scaler.pkl", 'rb') as f:
        scaler = pickle.load(f)
    with open(MODELS_DIR / "xgboost_v4_phase4_advanced_lookahead3_thresh0_features.txt", 'r') as f:
        features = [line.strip() for line in f]

    trades = []
    for i in range(len(df) - 96):
        row = df[features].iloc[i:i+1].values
        if np.isnan(row).any():
            continue

        row_scaled = scaler.transform(row)
        prob = model.predict_proba(row_scaled)[0][1]

        if prob >= threshold:
            trades.append({
                'entry_idx': i,
                'entry_price': df['close'].iloc[i],
                'entry_prob': prob
            })

    print(f"✅ Simulated {len(trades):,} LONG trades\n")
    return trades


def test_criterion_1_profit(df, trades, threshold=0.003):
    """
    Criterion 1: Profit Only

    Simple: Just check if current profit > threshold
    """
    print("="*80)
    print(f"CRITERION 1: Profit > {threshold*100:.1f}%")
    print("="*80)

    labels = np.zeros(len(df))

    for trade in trades:
        entry_idx = trade['entry_idx']
        entry_price = trade['entry_price']
        max_holding = min(entry_idx + 96, len(df))

        for i in range(entry_idx + 6, max_holding - 48):
            current_price = df['close'].iloc[i]
            profit = (current_price - entry_price) / entry_price

            if profit >= threshold:
                labels[i] = 1

    positive = labels.sum()
    positive_rate = positive / len(labels)

    print(f"Results:")
    print(f"  Positive labels: {int(positive):,}")
    print(f"  Positive rate: {positive_rate*100:.2f}%")
    print(f"  Labels per trade: {positive/len(trades):.1f}")
    print()

    return labels, {
        'criterion': 'profit',
        'threshold': threshold,
        'positive': int(positive),
        'positive_rate': positive_rate
    }


def test_criterion_2_lead_peak(df, trades, lead_min=3, lead_max=24, peak_threshold=0.002):
    """
    Criterion 2: Lead-Time Peak Detection

    Check if there's a peak ahead (lead_min to lead_max candles)
    """
    print("="*80)
    print(f"CRITERION 2: Lead-Time Peak Detection")
    print(f"  Lead window: {lead_min}-{lead_max} candles")
    print(f"  Peak threshold: {peak_threshold*100:.2f}%")
    print("="*80)

    labels = np.zeros(len(df))

    for trade in trades:
        entry_idx = trade['entry_idx']
        entry_price = trade['entry_price']
        max_holding = min(entry_idx + 96, len(df))

        for i in range(entry_idx + 6, max_holding - 48):
            current_price = df['close'].iloc[i]

            # Check future window for peak
            future_window = df['close'].iloc[i+lead_min:i+lead_max+1]
            if len(future_window) == 0:
                continue

            future_max = future_window.max()
            peak_distance = (future_max - current_price) / current_price

            if peak_distance > peak_threshold:
                # Verify it's a peak (price falls after)
                peak_idx = i + lead_min + future_window.idxmax()
                if peak_idx + 7 < len(df):
                    post_peak = df['close'].iloc[peak_idx+1:peak_idx+7].mean()
                    if post_peak < future_max * 0.997:  # Falls 0.3% after peak
                        labels[i] = 1

    positive = labels.sum()
    positive_rate = positive / len(labels)

    print(f"Results:")
    print(f"  Positive labels: {int(positive):,}")
    print(f"  Positive rate: {positive_rate*100:.2f}%")
    print(f"  Labels per trade: {positive/len(trades):.1f}")
    print()

    return labels, {
        'criterion': 'lead_peak',
        'lead_min': lead_min,
        'lead_max': lead_max,
        'peak_threshold': peak_threshold,
        'positive': int(positive),
        'positive_rate': positive_rate
    }


def test_criterion_3_relative(df, trades, profit_threshold=0.003, tolerance=0.001):
    """
    Criterion 3: Relative Performance

    Check if current exit beats future exits (within tolerance)
    Requires profit + relative comparison
    """
    print("="*80)
    print(f"CRITERION 3: Relative Performance")
    print(f"  Profit threshold: {profit_threshold*100:.1f}%")
    print(f"  Relative tolerance: {tolerance*100:.2f}%")
    print("="*80)

    labels = np.zeros(len(df))

    for trade in trades:
        entry_idx = trade['entry_idx']
        entry_price = trade['entry_price']
        max_holding = min(entry_idx + 96, len(df))

        for i in range(entry_idx + 6, max_holding - 48):
            current_price = df['close'].iloc[i]
            profit = (current_price - entry_price) / entry_price

            # Must have profit
            if profit < profit_threshold:
                continue

            # Compare to future exits
            future_prices = df['close'].iloc[i+1:i+25]
            if len(future_prices) == 0:
                continue

            future_profits = (future_prices - entry_price) / entry_price
            best_future_profit = future_profits.max()

            # Current exit should be within tolerance of best future
            if profit >= best_future_profit - tolerance:
                labels[i] = 1

    positive = labels.sum()
    positive_rate = positive / len(labels)

    print(f"Results:")
    print(f"  Positive labels: {int(positive):,}")
    print(f"  Positive rate: {positive_rate*100:.2f}%")
    print(f"  Labels per trade: {positive/len(trades):.1f}")
    print()

    return labels, {
        'criterion': 'relative',
        'profit_threshold': profit_threshold,
        'tolerance': tolerance,
        'positive': int(positive),
        'positive_rate': positive_rate
    }


def test_combination_2of3(labels1, labels2, labels3):
    """
    Test: Any 2 of 3 criteria (OR logic with scoring)
    """
    print("="*80)
    print("COMBINATION: Any 2 of 3 Criteria (Scoring)")
    print("="*80)

    # Score: 1 point per criterion satisfied
    scores = labels1 + labels2 + labels3

    # Accept if score >= 2 (any 2 criteria)
    labels_2of3 = (scores >= 2).astype(int)

    positive = labels_2of3.sum()
    positive_rate = positive / len(labels_2of3)

    print(f"Results:")
    print(f"  Positive labels: {int(positive):,}")
    print(f"  Positive rate: {positive_rate*100:.2f}%")
    print()

    # Breakdown by score
    print(f"Label Distribution by Score:")
    for score in [0, 1, 2, 3]:
        count = (scores == score).sum()
        pct = count / len(scores) * 100
        print(f"  Score {score}: {count:,} ({pct:.2f}%)")
    print()

    return labels_2of3, {
        'criterion': '2of3',
        'positive': int(positive),
        'positive_rate': positive_rate
    }


def test_combination_1of3(labels1, labels2, labels3):
    """
    Test: Any 1 of 3 criteria (Maximum OR)
    """
    print("="*80)
    print("COMBINATION: Any 1 of 3 Criteria (OR)")
    print("="*80)

    # Accept if ANY criterion is satisfied
    labels_1of3 = ((labels1 > 0) | (labels2 > 0) | (labels3 > 0)).astype(int)

    positive = labels_1of3.sum()
    positive_rate = positive / len(labels_1of3)

    print(f"Results:")
    print(f"  Positive labels: {int(positive):,}")
    print(f"  Positive rate: {positive_rate*100:.2f}%")
    print()

    return labels_1of3, {
        'criterion': '1of3',
        'positive': int(positive),
        'positive_rate': positive_rate
    }


def test_combination_all3(labels1, labels2, labels3):
    """
    Test: All 3 criteria (AND logic - original approach)
    """
    print("="*80)
    print("COMBINATION: All 3 Criteria (AND) - Original")
    print("="*80)

    # All must be satisfied
    labels_all3 = ((labels1 > 0) & (labels2 > 0) & (labels3 > 0)).astype(int)

    positive = labels_all3.sum()
    positive_rate = positive / len(labels_all3)

    print(f"Results:")
    print(f"  Positive labels: {int(positive):,}")
    print(f"  Positive rate: {positive_rate*100:.2f}%")
    print()

    return labels_all3, {
        'criterion': 'all3',
        'positive': int(positive),
        'positive_rate': positive_rate
    }


def main():
    print("="*80)
    print("Labeling Criteria Diagnostic Tool")
    print("="*80)
    print()

    # Load data
    df = load_data()

    # Simulate trades
    trades = simulate_trades(df, threshold=0.70)

    # Test each criterion independently
    labels_profit, stats_profit = test_criterion_1_profit(
        df, trades, threshold=0.003  # 0.3%
    )

    labels_peak, stats_peak = test_criterion_2_lead_peak(
        df, trades,
        lead_min=3,
        lead_max=24,
        peak_threshold=0.002  # 0.2%
    )

    labels_relative, stats_relative = test_criterion_3_relative(
        df, trades,
        profit_threshold=0.003,  # 0.3%
        tolerance=0.001  # 0.1%
    )

    # Test combinations
    labels_1of3, stats_1of3 = test_combination_1of3(
        labels_profit, labels_peak, labels_relative
    )

    labels_2of3, stats_2of3 = test_combination_2of3(
        labels_profit, labels_peak, labels_relative
    )

    labels_all3, stats_all3 = test_combination_all3(
        labels_profit, labels_peak, labels_relative
    )

    # Summary
    print("="*80)
    print("SUMMARY: Labeling Approaches Comparison")
    print("="*80)
    print()

    all_stats = [
        stats_profit,
        stats_peak,
        stats_relative,
        stats_1of3,
        stats_2of3,
        stats_all3
    ]

    print(f"{'Approach':<30} {'Positive Labels':<20} {'Positive Rate':<15}")
    print("-"*80)

    for stats in all_stats:
        criterion = stats['criterion'].upper()
        positive = stats['positive']
        rate = stats['positive_rate']
        print(f"{criterion:<30} {positive:>15,}   {rate:>12.2%}")

    print()
    print("="*80)
    print("RECOMMENDATIONS")
    print("="*80)
    print()

    # Target: 10-20% positive rate
    target_min = 0.10
    target_max = 0.20

    recommendations = []

    for stats in all_stats:
        rate = stats['positive_rate']
        criterion = stats['criterion']

        if target_min <= rate <= target_max:
            recommendations.append(f"✅ {criterion.upper()}: {rate:.2%} (IDEAL RANGE)")
        elif rate < target_min:
            recommendations.append(f"⚠️ {criterion.upper()}: {rate:.2%} (too strict)")
        else:
            recommendations.append(f"⚠️ {criterion.upper()}: {rate:.2%} (too loose)")

    for rec in recommendations:
        print(rec)

    print()
    print("="*80)
    print("NEXT STEPS")
    print("="*80)
    print()

    # Find best approach
    best_approach = None
    best_distance = float('inf')

    for stats in all_stats:
        rate = stats['positive_rate']
        # Distance from ideal center (15%)
        distance = abs(rate - 0.15)

        if distance < best_distance:
            best_distance = distance
            best_approach = stats

    print(f"Recommended Approach: {best_approach['criterion'].upper()}")
    print(f"  Positive Rate: {best_approach['positive_rate']:.2%}")
    print(f"  Positive Labels: {best_approach['positive']:,}")
    print()

    if best_approach['criterion'] == '1of3':
        print("Implementation: Use OR logic (any 1 of 3 criteria)")
    elif best_approach['criterion'] == '2of3':
        print("Implementation: Use scoring system (any 2 of 3 criteria)")
    elif best_approach['criterion'] == 'profit':
        print("Implementation: Use simple profit-based labeling")
    elif best_approach['criterion'] == 'lead_peak':
        print("Implementation: Use lead-time peak detection only")
    elif best_approach['criterion'] == 'relative':
        print("Implementation: Use relative performance only")
    else:
        print("Implementation: Current AND logic (all 3 criteria)")

    print()
    print("="*80)


if __name__ == "__main__":
    main()
