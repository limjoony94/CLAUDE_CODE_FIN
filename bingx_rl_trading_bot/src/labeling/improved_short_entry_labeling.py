"""
Improved SHORT Entry Labeling with 2of3 Scoring System

Inspired by EXIT model's success with 2of3 scoring (13.93% positive rate)

Problem with Peak/Trough labeling:
  - Too strict conditions → 1.00% signal rate
  - Low win rate (20%)
  - Minimal contribution (~5%)

Solution: Flexible 2of3 Scoring
  - Criterion 1: 수익 가능성 (0.5% 이상 하락)
  - Criterion 2: 빠른 하락 (3-24 candles 내 저점)
  - Criterion 3: 상대적 우위 (지금 진입 > 나중 진입)

  Label 1 if score >= 2 (any 2 of 3)
"""

import pandas as pd
import numpy as np


def create_improved_short_entry_labels(
    df,
    lookahead=24,  # 2 hours (24 * 5min)
    profit_threshold=0.005,  # 0.5% minimum profit potential
    lead_min=3,  # Minimum lead-time (15 min)
    lead_max=24,  # Maximum lead-time (2 hours)
    relative_delay=12  # Compare with 1h delayed entry
):
    """
    Improved SHORT Entry Labels with 2of3 Scoring

    Args:
        df: DataFrame with OHLCV data
        lookahead: How far to look ahead for troughs (candles)
        profit_threshold: Minimum profit % for good SHORT entry
        lead_min: Minimum candles to trough
        lead_max: Maximum candles to trough
        relative_delay: Delay candles for comparison

    Returns:
        labels: numpy array of 0/1 labels
    """
    print("\n" + "="*80)
    print("Improved SHORT Entry Labeling (2of3 Scoring)")
    print("="*80)
    print(f"Lookahead: {lookahead} candles ({lookahead*5/60:.1f} hours)")
    print(f"Profit threshold: {profit_threshold*100:.1f}%")
    print(f"Lead-time window: {lead_min}-{lead_max} candles ({lead_min*5/60:.1f}-{lead_max*5/60:.1f}h)")
    print(f"Relative comparison: {relative_delay} candles ({relative_delay*5/60:.1f}h)")

    labels = []
    label_details = []

    for i in range(len(df)):
        # Can't label near the end
        if i >= len(df) - lookahead:
            labels.append(0)
            label_details.append({'reason': 'insufficient_future_data'})
            continue

        current_price = df['close'].iloc[i]

        # Get future window
        future_window = df.iloc[i+1:i+1+lookahead]  # Start from next candle
        future_lows = future_window['low'].values

        if len(future_lows) == 0:
            labels.append(0)
            label_details.append({'reason': 'no_future_data'})
            continue

        # Find minimum price in future window
        min_idx = np.argmin(future_lows)
        min_price = future_lows[min_idx]
        candles_to_min = min_idx + 1  # +1 because we started from next candle

        # Calculate potential SHORT profit
        # SHORT profit = (entry_price - exit_price) / entry_price
        potential_profit = (current_price - min_price) / current_price

        # Criterion 1: Profit potential (0.5%+ drop expected)
        has_profit = potential_profit >= profit_threshold

        # Criterion 2: Lead-time quality (trough within 3-24 candles)
        has_lead_time = (lead_min <= candles_to_min <= lead_max)

        # Criterion 3: Beats delayed entry
        beats_delayed = False
        if i + relative_delay < len(df):
            delayed_entry_price = df['close'].iloc[i + relative_delay]
            # Check if delayed entry can still catch the trough
            delayed_profit = (delayed_entry_price - min_price) / delayed_entry_price
            beats_delayed = potential_profit > delayed_profit
        else:
            # If can't delay, assume current is better
            beats_delayed = has_profit

        # 2of3 Scoring
        score = sum([has_profit, has_lead_time, beats_delayed])
        label = 1 if score >= 2 else 0

        labels.append(label)
        label_details.append({
            'has_profit': has_profit,
            'has_lead_time': has_lead_time,
            'beats_delayed': beats_delayed,
            'score': score,
            'potential_profit': potential_profit,
            'candles_to_min': candles_to_min,
            'min_price': min_price
        })

    labels = np.array(labels)

    # Analysis
    positive_count = np.sum(labels == 1)
    positive_rate = positive_count / len(labels) * 100

    # Detailed breakdown
    has_profit_count = sum(1 for d in label_details if d.get('has_profit', False))
    has_lead_count = sum(1 for d in label_details if d.get('has_lead_time', False))
    beats_delayed_count = sum(1 for d in label_details if d.get('beats_delayed', False))

    score_2_count = sum(1 for d in label_details if d.get('score', 0) == 2)
    score_3_count = sum(1 for d in label_details if d.get('score', 0) == 3)

    print(f"\nCriterion Statistics:")
    print(f"  Has profit (>={profit_threshold*100:.1f}%): {has_profit_count:,} ({has_profit_count/len(labels)*100:.1f}%)")
    print(f"  Has lead-time ({lead_min}-{lead_max}): {has_lead_count:,} ({has_lead_count/len(labels)*100:.1f}%)")
    print(f"  Beats delayed: {beats_delayed_count:,} ({beats_delayed_count/len(labels)*100:.1f}%)")

    print(f"\nScoring Distribution:")
    print(f"  Score 2/3: {score_2_count:,} ({score_2_count/len(labels)*100:.1f}%)")
    print(f"  Score 3/3: {score_3_count:,} ({score_3_count/len(labels)*100:.1f}%)")
    print(f"  Total positive (score>=2): {positive_count:,} ({positive_rate:.2f}%)")

    print(f"\n✅ Positive rate: {positive_rate:.2f}%")

    if positive_rate < 5.0:
        print(f"⚠️  WARNING: Positive rate very low ({positive_rate:.2f}%)")
        print(f"   Consider: lower profit_threshold or wider lead_time window")
    elif positive_rate > 25.0:
        print(f"⚠️  WARNING: Positive rate very high ({positive_rate:.2f}%)")
        print(f"   Consider: higher profit_threshold or stricter lead_time window")
    else:
        print(f"✅ Positive rate in good range (5-25%)")

    return labels


def diagnose_short_entry_labeling(df):
    """
    Diagnose SHORT Entry labeling with different parameters

    Test multiple configurations to find optimal settings
    """
    print("\n" + "="*80)
    print("SHORT Entry Labeling Diagnosis")
    print("="*80)

    configs = [
        # (profit_threshold, lead_min, lead_max, description)
        (0.003, 3, 24, "Conservative (0.3%, 3-24 candles)"),
        (0.005, 3, 24, "Baseline (0.5%, 3-24 candles)"),
        (0.007, 3, 24, "Strict (0.7%, 3-24 candles)"),
        (0.005, 3, 48, "Extended window (0.5%, 3-48 candles)"),
        (0.005, 6, 24, "Delayed entry (0.5%, 6-24 candles)"),
    ]

    results = []

    for profit_th, lead_min, lead_max, desc in configs:
        print(f"\n{'='*80}")
        print(f"Testing: {desc}")
        print(f"{'='*80}")

        labels = create_improved_short_entry_labels(
            df,
            lookahead=max(lead_max + 12, 24),
            profit_threshold=profit_th,
            lead_min=lead_min,
            lead_max=lead_max,
            relative_delay=12
        )

        positive_rate = np.sum(labels) / len(labels) * 100

        results.append({
            'config': desc,
            'positive_rate': positive_rate,
            'positive_count': np.sum(labels),
            'profit_threshold': profit_th,
            'lead_min': lead_min,
            'lead_max': lead_max
        })

    # Summary
    print("\n" + "="*80)
    print("DIAGNOSIS SUMMARY")
    print("="*80)
    print(f"\n{'Configuration':<50} {'Positive Rate':<15} {'Count':<10}")
    print("-"*80)

    for r in results:
        print(f"{r['config']:<50} {r['positive_rate']:>6.2f}% {r['positive_count']:>10,}")

    # Recommendation
    print("\n" + "="*80)
    print("RECOMMENDATION")
    print("="*80)

    # Find config with positive rate in 10-20% range (ideal for ML)
    ideal_configs = [r for r in results if 10.0 <= r['positive_rate'] <= 20.0]

    if ideal_configs:
        best = ideal_configs[0]
        print(f"✅ Recommended: {best['config']}")
        print(f"   Positive rate: {best['positive_rate']:.2f}%")
        print(f"   Profit threshold: {best['profit_threshold']*100:.1f}%")
        print(f"   Lead-time: {best['lead_min']}-{best['lead_max']} candles")
    else:
        # Find closest to 15%
        best = min(results, key=lambda r: abs(r['positive_rate'] - 15.0))
        print(f"⚠️  No config in ideal range (10-20%)")
        print(f"   Closest: {best['config']}")
        print(f"   Positive rate: {best['positive_rate']:.2f}%")
        print(f"   Consider: adjust parameters to reach 10-20% range")

    return results


if __name__ == "__main__":
    # Test with synthetic data
    print("="*80)
    print("Testing Improved SHORT Entry Labeling")
    print("="*80)

    # Create downtrend data
    np.random.seed(42)
    n = 1000

    # Simulate BTC-like price movement with some downtrends
    price = 100000.0
    prices = []

    for i in range(n):
        # Add some downtrend periods
        if i % 200 < 50:  # Downtrend every 200 candles
            change = np.random.normal(-0.002, 0.005)  # Downward bias
        else:
            change = np.random.normal(0.001, 0.005)  # Slight upward bias

        price = price * (1 + change)
        prices.append(price)

    prices = np.array(prices)

    df = pd.DataFrame({
        'open': prices,
        'high': prices * (1 + abs(np.random.normal(0, 0.001, n))),
        'low': prices * (1 - abs(np.random.normal(0, 0.001, n))),
        'close': prices,
        'volume': np.random.normal(1000, 100, n)
    })

    print(f"\nSynthetic data: {len(df)} candles")
    print(f"Price range: ${df['close'].min():.2f} - ${df['close'].max():.2f}")

    # Test baseline
    labels = create_improved_short_entry_labels(df)

    print(f"\n✅ Labeling test complete!")
    print(f"   Labels created: {len(labels):,}")
    print(f"   Positive labels: {np.sum(labels):,} ({np.sum(labels)/len(labels)*100:.1f}%)")
