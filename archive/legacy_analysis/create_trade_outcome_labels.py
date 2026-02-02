"""
Create Trade Outcome Labels for Optimization
=============================================

Simplified labeling logic based on actual price movements.
No Exit model needed - uses forward-looking price action.

Criteria (based on Walk-Forward Decoupled logic):
- Leveraged P&L > 2% within reasonable hold time
- No catastrophic loss (< -3% balance = -0.75% price with 4x leverage)
- Reasonable hold time (< 60 candles = 5 hours)

LONG Label = 1 if:
  - Price reaches +0.5% (2% leveraged) within 60 candles
  - AND never drops below -0.75% (3% leveraged loss)

SHORT Label = 1 if:
  - Price drops -0.5% (2% leveraged gain) within 60 candles
  - AND never rises above +0.75% (3% leveraged loss)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Configuration (matching Walk-Forward Decoupled - stricter criteria)
LEVERAGE = 4
TARGET_PROFIT_PCT = 0.01   # 1.0% price = 4% leveraged
MAX_LOSS_PCT = 0.0075      # 0.75% price = 3% leveraged
MAX_HOLD_CANDLES = 60      # 5 hours


def label_long_entries(df):
    """
    Label LONG entry points based on forward price action

    Good LONG entry = Price reaches +0.5% without hitting -0.75% within 60 candles
    """
    print("\n🔵 Labeling LONG Entries...")

    labels = np.zeros(len(df))
    good_count = 0

    for i in range(len(df) - MAX_HOLD_CANDLES):
        entry_price = df['close'].iloc[i]
        future_prices = df['close'].iloc[i+1:i+MAX_HOLD_CANDLES+1].values

        if len(future_prices) == 0:
            continue

        # Calculate returns
        returns = (future_prices - entry_price) / entry_price

        # Check for target hit
        target_hit = np.any(returns >= TARGET_PROFIT_PCT)

        # Check for stop loss hit BEFORE target
        if target_hit:
            target_idx = np.where(returns >= TARGET_PROFIT_PCT)[0][0]
            returns_before_target = returns[:target_idx+1]
            stop_hit = np.any(returns_before_target <= -MAX_LOSS_PCT)
        else:
            stop_hit = np.any(returns <= -MAX_LOSS_PCT)

        # Label as good if target hit and stop not hit first
        if target_hit and not stop_hit:
            labels[i] = 1
            good_count += 1

    positive = (labels == 1).sum()
    positive_pct = positive / len(labels) * 100

    print(f"  Good LONG entries: {positive:,} ({positive_pct:.2f}%)")
    print(f"  Bad LONG entries: {len(labels) - positive:,} ({100-positive_pct:.2f}%)")

    return labels


def label_short_entries(df):
    """
    Label SHORT entry points based on forward price action

    Good SHORT entry = Price drops -0.5% without rising +0.75% within 60 candles
    """
    print("\n🔴 Labeling SHORT Entries...")

    labels = np.zeros(len(df))
    good_count = 0

    for i in range(len(df) - MAX_HOLD_CANDLES):
        entry_price = df['close'].iloc[i]
        future_prices = df['close'].iloc[i+1:i+MAX_HOLD_CANDLES+1].values

        if len(future_prices) == 0:
            continue

        # Calculate returns (negative for SHORT)
        returns = (entry_price - future_prices) / entry_price

        # Check for target hit
        target_hit = np.any(returns >= TARGET_PROFIT_PCT)

        # Check for stop loss hit BEFORE target
        if target_hit:
            target_idx = np.where(returns >= TARGET_PROFIT_PCT)[0][0]
            returns_before_target = returns[:target_idx+1]
            stop_hit = np.any(returns_before_target <= -MAX_LOSS_PCT)
        else:
            stop_hit = np.any(returns <= -MAX_LOSS_PCT)

        # Label as good if target hit and stop not hit first
        if target_hit and not stop_hit:
            labels[i] = 1
            good_count += 1

    positive = (labels == 1).sum()
    positive_pct = positive / len(labels) * 100

    print(f"  Good SHORT entries: {positive:,} ({positive_pct:.2f}%)")
    print(f"  Bad SHORT entries: {len(labels) - positive:,} ({100-positive_pct:.2f}%)")

    return labels


def main():
    """Generate trade outcome labels"""

    print("="*80)
    print("Trade Outcome Label Generation")
    print("="*80)

    print(f"\nConfiguration:")
    print(f"  Leverage: {LEVERAGE}x")
    print(f"  Target Profit: {TARGET_PROFIT_PCT*100:.2f}% price = {TARGET_PROFIT_PCT*LEVERAGE*100:.1f}% leveraged")
    print(f"  Max Loss: {MAX_LOSS_PCT*100:.2f}% price = {MAX_LOSS_PCT*LEVERAGE*100:.1f}% leveraged")
    print(f"  Max Hold: {MAX_HOLD_CANDLES} candles ({MAX_HOLD_CANDLES/12:.1f} hours)")

    # Load data
    DATA_DIR = PROJECT_ROOT / "data" / "historical"
    data_file = DATA_DIR / "BTCUSDT_5m_max.csv"

    print(f"\nLoading data from: {data_file}")
    df = pd.read_csv(data_file)

    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])

    print(f"Loaded {len(df):,} candles")
    print(f"Period: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")

    # Generate labels
    print("\n" + "="*80)
    print("Generating Labels")
    print("="*80)

    long_labels = label_long_entries(df)
    short_labels = label_short_entries(df)

    # Add to dataframe
    df['signal_long'] = long_labels
    df['signal_short'] = short_labels

    # Save labeled data
    LABELS_DIR = PROJECT_ROOT / "data" / "labels"
    LABELS_DIR.mkdir(exist_ok=True)

    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    output_file = LABELS_DIR / f"trade_outcome_labels_{timestamp}.csv"
    df[['timestamp', 'close', 'signal_long', 'signal_short']].to_csv(output_file, index=False)

    print(f"\n{'='*80}")
    print(f"Labels saved to: {output_file}")
    print(f"{'='*80}")

    # Summary statistics
    print(f"\n📊 Label Statistics:")
    print(f"\n  LONG:")
    print(f"    Positive: {long_labels.sum():,} ({long_labels.mean()*100:.2f}%)")
    print(f"    Negative: {(long_labels == 0).sum():,} ({(1-long_labels.mean())*100:.2f}%)")

    print(f"\n  SHORT:")
    print(f"    Positive: {short_labels.sum():,} ({short_labels.mean()*100:.2f}%)")
    print(f"    Negative: {(short_labels == 0).sum():,} ({(1-short_labels.mean())*100:.2f}%)")

    print(f"\n  Expected Positive Rate: 5-15%")
    print(f"  Actual LONG: {long_labels.mean()*100:.2f}%")
    print(f"  Actual SHORT: {short_labels.mean()*100:.2f}%")

    if 5 <= long_labels.mean()*100 <= 15 and 5 <= short_labels.mean()*100 <= 15:
        print(f"\n  ✅ Label distribution looks good!")
    else:
        print(f"\n  ⚠️  Label distribution outside expected range")
        print(f"     Consider adjusting TARGET_PROFIT_PCT or MAX_LOSS_PCT")

    return output_file


if __name__ == "__main__":
    label_file = main()
    print(f"\n✅ Trade outcome labels generated successfully!")
    print(f"   Use this file for optimization: {label_file}")
