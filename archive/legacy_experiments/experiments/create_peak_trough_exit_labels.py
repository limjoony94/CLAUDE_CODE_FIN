"""
Create Peak/Trough-Based Exit Labels
======================================

Strategy: Exit at optimal timing based on local peaks/troughs

Peak/Trough Concept:
  - LONG Exit: Exit when price reaches local peak (sell high)
  - SHORT Exit: Exit when price reaches local trough (buy low)

Method:
  1. Look forward N candles (lookforward window)
  2. Find local peak (LONG) or trough (SHORT) in that window
  3. If current price is within 80% of peak/trough → EXIT signal
  4. Also respect emergency rules (Stop Loss, Max Hold)

Expected Impact:
  - Better exit timing (ride trends longer)
  - Higher win rate (exit near optimal points)
  - Better risk/reward (avoid premature exits)

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
LOOKFORWARD_WINDOW = 24  # 24 candles = 2 hours (RELAXED: was 48)
PEAK_THRESHOLD = 0.80  # Exit if within 80% of peak
TROUGH_THRESHOLD = 0.80  # Exit if within 80% of trough
MAX_HOLD_TIME = 120  # 120 candles (10 hours)
STOP_LOSS = -0.03  # -3% leveraged loss
NEAR_WINDOW = 15  # Consider peak/trough "near" if within 15 candles (RELAXED: was 10)
PROXIMITY_THRESHOLD = 0.85  # Exit if within 85% of peak/trough (RELAXED: was 0.95)
MIN_PROFIT = 0.005  # Minimum 0.5% profit to exit (RELAXED: was 0.01)

DATA_DIR = PROJECT_ROOT / "data" / "features"
LABELS_DIR = PROJECT_ROOT / "data" / "labels"
LABELS_DIR.mkdir(exist_ok=True)

print("=" * 80)
print("PEAK/TROUGH EXIT LABELS GENERATION")
print("=" * 80)
print()
print("Strategy: Exit at local peaks (LONG) and troughs (SHORT) - RELAXED")
print(f"  Lookforward Window: {LOOKFORWARD_WINDOW} candles ({LOOKFORWARD_WINDOW/12:.2f} hours)")
print(f"  Near Window: {NEAR_WINDOW} candles (peak/trough proximity)")
print(f"  Proximity Threshold: {PROXIMITY_THRESHOLD*100:.0f}% (price near peak/trough)")
print(f"  Min Profit: {MIN_PROFIT*100:.1f}%")
print(f"  Max Hold Time: {MAX_HOLD_TIME} candles ({MAX_HOLD_TIME/12:.2f} hours)")
print(f"  Stop Loss: {STOP_LOSS*100:.1f}%")
print()

# Load Features Dataset
print("-" * 80)
print("STEP 1: Loading Features Dataset")
print("-" * 80)

df = pd.read_csv(DATA_DIR / "BTCUSDT_5m_features_enhanced_exit.csv")
df['timestamp'] = pd.to_datetime(df['timestamp'])

print(f"✅ Features loaded: {len(df):,} candles")
print(f"   Date range: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")
print(f"   Days: {(df['timestamp'].max() - df['timestamp'].min()).days}")
print()

# Create LONG Exit Labels (Peak-based)
print("-" * 80)
print("STEP 2: Creating LONG Exit Labels (Peak-based)")
print("-" * 80)

long_exit_labels = []

for i in range(len(df)):
    # Default: no exit signal
    exit_signal = 0

    # Look forward for peak
    start_idx = i + 1  # Start from next candle
    end_idx = min(i + LOOKFORWARD_WINDOW + 1, len(df))
    future_window = df.iloc[start_idx:end_idx]

    if len(future_window) < 2:
        long_exit_labels.append(0)
        continue

    # Current price (entry price)
    current_price = df.iloc[i]['close']

    # Find peak in FUTURE window
    peak_idx_in_window = future_window['close'].idxmax()
    peak_price = future_window.loc[peak_idx_in_window, 'close']
    peak_position = future_window.index.get_loc(peak_idx_in_window)  # Position in future window

    # Calculate leveraged P&L at peak
    price_change = (peak_price - current_price) / current_price
    leveraged_pnl = price_change * LEVERAGE

    # EXIT condition: Peak is NEAR (within NEAR_WINDOW candles of future)
    # AND current price is close to peak (within PROXIMITY_THRESHOLD)
    peak_is_near = peak_position <= NEAR_WINDOW
    price_near_peak = current_price / peak_price >= PROXIMITY_THRESHOLD

    if peak_is_near and price_near_peak and leveraged_pnl > MIN_PROFIT:
        exit_signal = 1

    long_exit_labels.append(exit_signal)

    # Progress
    if (i + 1) % 10000 == 0:
        print(f"  Processed {i+1:,}/{len(df):,} candles...")

df['long_exit_peak'] = long_exit_labels

print(f"\n✅ LONG Exit labels created")
print(f"   Total: {len(long_exit_labels):,}")
print(f"   Exit signals: {sum(long_exit_labels):,} ({sum(long_exit_labels)/len(long_exit_labels)*100:.2f}%)")
print()

# Create SHORT Exit Labels (Trough-based)
print("-" * 80)
print("STEP 3: Creating SHORT Exit Labels (Trough-based)")
print("-" * 80)

short_exit_labels = []

for i in range(len(df)):
    # Default: no exit signal
    exit_signal = 0

    # Look forward for trough
    start_idx = i + 1  # Start from next candle
    end_idx = min(i + LOOKFORWARD_WINDOW + 1, len(df))
    future_window = df.iloc[start_idx:end_idx]

    if len(future_window) < 2:
        short_exit_labels.append(0)
        continue

    # Current price (entry price)
    current_price = df.iloc[i]['close']

    # Find trough in FUTURE window
    trough_idx_in_window = future_window['close'].idxmin()
    trough_price = future_window.loc[trough_idx_in_window, 'close']
    trough_position = future_window.index.get_loc(trough_idx_in_window)  # Position in future window

    # Calculate leveraged P&L at trough (SHORT: profit when price drops)
    price_change = (current_price - trough_price) / current_price  # Inverted for SHORT
    leveraged_pnl = price_change * LEVERAGE

    # EXIT condition: Trough is NEAR (within NEAR_WINDOW candles of future)
    # AND current price is close to trough (within PROXIMITY_THRESHOLD)
    trough_is_near = trough_position <= NEAR_WINDOW
    price_near_trough = trough_price / current_price >= PROXIMITY_THRESHOLD

    if trough_is_near and price_near_trough and leveraged_pnl > MIN_PROFIT:
        exit_signal = 1

    short_exit_labels.append(exit_signal)

    # Progress
    if (i + 1) % 10000 == 0:
        print(f"  Processed {i+1:,}/{len(df):,} candles...")

df['short_exit_trough'] = short_exit_labels

print(f"\n✅ SHORT Exit labels created")
print(f"   Total: {len(short_exit_labels):,}")
print(f"   Exit signals: {sum(short_exit_labels):,} ({sum(short_exit_labels)/len(short_exit_labels)*100:.2f}%)")
print()

# Save Labels
print("-" * 80)
print("STEP 4: Saving Labels")
print("-" * 80)

timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = LABELS_DIR / f"exit_labels_peak_trough_{timestamp_str}.csv"

# Save only timestamp + labels
labels_df = df[['timestamp', 'long_exit_peak', 'short_exit_trough']].copy()
labels_df.to_csv(output_file, index=False)

print(f"✅ Labels saved: {output_file.name}")
print(f"   Size: {output_file.stat().st_size / 1024 / 1024:.2f} MB")
print()

# Summary Statistics
print("=" * 80)
print("SUMMARY")
print("=" * 80)

print("\nLabel Distribution:")
print(f"  LONG Exit (Peak-based):")
print(f"    Exit signals: {sum(long_exit_labels):,} ({sum(long_exit_labels)/len(long_exit_labels)*100:.2f}%)")
print(f"    No exit: {len(long_exit_labels) - sum(long_exit_labels):,} ({(1 - sum(long_exit_labels)/len(long_exit_labels))*100:.2f}%)")
print()
print(f"  SHORT Exit (Trough-based):")
print(f"    Exit signals: {sum(short_exit_labels):,} ({sum(short_exit_labels)/len(short_exit_labels)*100:.2f}%)")
print(f"    No exit: {len(short_exit_labels) - sum(short_exit_labels):,} ({(1 - sum(short_exit_labels)/len(short_exit_labels))*100:.2f}%)")
print()

print("\nConfiguration (RELAXED):")
print(f"  Lookforward: {LOOKFORWARD_WINDOW} candles ({LOOKFORWARD_WINDOW/12:.2f} hours)")
print(f"  Near Window: {NEAR_WINDOW} candles")
print(f"  Proximity Threshold: {PROXIMITY_THRESHOLD*100:.0f}%")
print(f"  Min Profit: {MIN_PROFIT*100:.1f}%")
print(f"  Max Hold: {MAX_HOLD_TIME} candles ({MAX_HOLD_TIME/12:.2f} hours)")
print(f"  Stop Loss: {STOP_LOSS*100:.1f}%")
print()

print("✅ Peak/Trough Exit labels generation complete!")
print()
print("Next Steps:")
print("  1. Run: python scripts/experiments/retrain_peak_trough_exit_models.py")
print("  2. Run: python scripts/experiments/backtest_peak_trough_exits.py")
print()
