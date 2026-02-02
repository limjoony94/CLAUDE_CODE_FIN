"""
Create Patience-Based Exit Labels
===================================

Problem: 59% of trades exit too early (0-5 candles) with only 48.49% WR
Solution: Create patience-based exit labels that wait for optimal timing

Analysis from Enhanced Baseline:
  0-5 candles:  48.49% WR, +0.1151% avg (728 trades) ❌
  10-20 candles: 69.40% WR, +0.9075% avg (134 trades) ✅
  50+ candles:  71.64% WR, +1.1479% avg (67 trades) ✅

Strategy:
  - Old Label: exit_signal = 1 when profit > threshold
  - New Label: exit_signal = 1 when profit > threshold AND hold_time >= min_hold
  - Parameters: profit_threshold = 0.5%, min_hold = 10 candles

Expected Impact: Win Rate 48% → 65%+, Avg profit +0.1% → +0.9%+

Created: 2025-10-30
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
from datetime import datetime

# Configuration
LEVERAGE = 4
PROFIT_THRESHOLD = 0.005  # 0.5% leveraged profit
MIN_HOLD_TIME = 10  # 10 candles (50 minutes)
MAX_HOLD_TIME = 120  # 120 candles (10 hours)
STOP_LOSS = -0.03  # -3% leveraged loss

DATA_DIR = PROJECT_ROOT / "data" / "features"
LABELS_DIR = PROJECT_ROOT / "data" / "labels"
LABELS_DIR.mkdir(exist_ok=True)

print("=" * 80)
print("PATIENCE-BASED EXIT LABELS GENERATION")
print("=" * 80)
print()
print("Strategy: Encourage longer holds for better profits")
print(f"  Profit Threshold: {PROFIT_THRESHOLD*100:.2f}%")
print(f"  Min Hold Time: {MIN_HOLD_TIME} candles ({MIN_HOLD_TIME/12:.2f} hours)")
print(f"  Max Hold Time: {MAX_HOLD_TIME} candles ({MAX_HOLD_TIME/12:.2f} hours)")
print(f"  Stop Loss: {STOP_LOSS*100:.2f}%")
print()

# Load dataset
print("-" * 80)
print("Loading Dataset...")
print("-" * 80)

df = pd.read_csv(DATA_DIR / "BTCUSDT_5m_features.csv")
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values('timestamp').reset_index(drop=True)

print(f"✅ Loaded {len(df):,} candles")
print(f"   Date range: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")
print(f"   Days: {(df['timestamp'].max() - df['timestamp'].min()).days}")
print()

# Generate patience-based exit labels
print("-" * 80)
print("Generating Patience-Based Exit Labels...")
print("-" * 80)
print()

def create_patience_exit_labels(df, side, profit_threshold, min_hold):
    """
    Create exit labels with patience requirement

    Exit = 1 when:
      1. Profit > threshold AND hold_time >= min_hold (patience exit)
      2. Loss < stop_loss (emergency exit)
      3. Hold time >= max_hold (timeout exit)
    """
    labels = np.zeros(len(df), dtype=int)

    total_entries = 0
    patience_exits = 0
    emergency_exits = 0
    timeout_exits = 0

    for entry_idx in range(len(df) - MAX_HOLD_TIME):
        entry_price = df['close'].iloc[entry_idx]

        # Simulate hold
        for hold_time in range(1, MAX_HOLD_TIME + 1):
            exit_idx = entry_idx + hold_time
            if exit_idx >= len(df):
                break

            exit_price = df['close'].iloc[exit_idx]

            # Calculate P&L
            if side == 'LONG':
                price_pnl = (exit_price - entry_price) / entry_price
            else:  # SHORT
                price_pnl = (entry_price - exit_price) / entry_price

            leveraged_pnl = price_pnl * LEVERAGE

            # Check exit conditions
            exit_triggered = False

            # 1. Stop Loss (immediate exit)
            if leveraged_pnl <= STOP_LOSS:
                labels[exit_idx] = 1
                emergency_exits += 1
                exit_triggered = True

            # 2. Max Hold (timeout exit)
            elif hold_time >= MAX_HOLD_TIME:
                labels[exit_idx] = 1
                timeout_exits += 1
                exit_triggered = True

            # 3. Patience Exit (profit + hold time requirement)
            elif leveraged_pnl >= profit_threshold and hold_time >= min_hold:
                labels[exit_idx] = 1
                patience_exits += 1
                exit_triggered = True

            # Exit simulation if triggered
            if exit_triggered:
                total_entries += 1
                break

    print(f"  {side} Exit Labels:")
    print(f"    Total Entries Simulated: {total_entries:,}")
    print(f"    Patience Exits: {patience_exits:,} ({patience_exits/total_entries*100:.1f}%)")
    print(f"    Emergency Exits: {emergency_exits:,} ({emergency_exits/total_entries*100:.1f}%)")
    print(f"    Timeout Exits: {timeout_exits:,} ({timeout_exits/total_entries*100:.1f}%)")
    print(f"    Total Exit Signals: {labels.sum():,} ({labels.sum()/len(labels)*100:.2f}%)")
    print()

    return labels

# Create LONG exit labels
print("Creating LONG Patience Exit Labels...")
long_exit_labels = create_patience_exit_labels(df, 'LONG', PROFIT_THRESHOLD, MIN_HOLD_TIME)

# Create SHORT exit labels
print("Creating SHORT Patience Exit Labels...")
short_exit_labels = create_patience_exit_labels(df, 'SHORT', PROFIT_THRESHOLD, MIN_HOLD_TIME)

# Add labels to dataframe
df['long_exit_patience'] = long_exit_labels
df['short_exit_patience'] = short_exit_labels

# Save labeled dataset
print("-" * 80)
print("Saving Labeled Dataset...")
print("-" * 80)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = LABELS_DIR / f"exit_labels_patience_{timestamp}.csv"

df.to_csv(output_file, index=False)

print(f"✅ Saved: {output_file}")
print(f"   Columns added: long_exit_patience, short_exit_patience")
print()

# Statistics
print("-" * 80)
print("Label Statistics")
print("-" * 80)
print()

print(f"LONG Exit Labels:")
print(f"  Total Samples: {len(df):,}")
print(f"  Exit Signals: {long_exit_labels.sum():,} ({long_exit_labels.sum()/len(df)*100:.2f}%)")
print(f"  No Exit: {(long_exit_labels == 0).sum():,} ({(long_exit_labels == 0).sum()/len(df)*100:.2f}%)")
print()

print(f"SHORT Exit Labels:")
print(f"  Total Samples: {len(df):,}")
print(f"  Exit Signals: {short_exit_labels.sum():,} ({short_exit_labels.sum()/len(df)*100:.2f}%)")
print(f"  No Exit: {(short_exit_labels == 0).sum():,} ({(short_exit_labels == 0).sum()/len(df)*100:.2f}%)")
print()

print("=" * 80)
print("PATIENCE EXIT LABELS GENERATION COMPLETE")
print("=" * 80)
print()
print("Next Step: Train Exit models with these patience-based labels")
print(f"  Command: python scripts/experiments/retrain_patience_exit_models.py")
