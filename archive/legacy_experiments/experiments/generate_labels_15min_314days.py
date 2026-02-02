"""
Generate Entry and Exit Labels for 15-Minute 314-Day Dataset
=============================================================

Purpose: Create accurate trade outcome labels for 15-min candles

Labeling Strategy:
1. Entry Labels (LONG/SHORT):
   - LONG signal: Price rises X% within N candles (5% in 8 candles = 2 hours)
   - SHORT signal: Price falls X% within N candles (5% in 8 candles = 2 hours)

2. Exit Labels (patience-based):
   - LONG exit: Price stops rising or starts declining
   - SHORT exit: Price stops falling or starts rising

Input: BTCUSDT_15m_raw_314days_20251106_143614.csv
Output:
  - entry_labels_15min_314days_{timestamp}.csv
  - exit_labels_15min_314days_{timestamp}.csv

Created: 2025-11-06 15:30 KST
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
from datetime import datetime

# Paths
DATA_DIR = PROJECT_ROOT / "data" / "features"
LABELS_DIR = PROJECT_ROOT / "data" / "labels"
LABELS_DIR.mkdir(parents=True, exist_ok=True)

# Input file
INPUT_FILE = DATA_DIR / "BTCUSDT_15m_raw_314days_20251106_143614.csv"

# Labeling parameters (adjusted for 15-min candles)
ENTRY_THRESHOLD = 0.02  # 2% price movement (less extreme for 15-min)
ENTRY_LOOKFORWARD = 20  # 20 candles = 5 hours (15min × 20 = 300 min)
EXIT_PATIENCE = 8       # 8 candles = 2 hours for exit signals

print("="*80)
print("GENERATING ENTRY AND EXIT LABELS FOR 15-MIN CANDLES")
print("="*80)
print()
print(f"📂 Input: {INPUT_FILE.name}")
print(f"📊 Timeframe: 15-minute candles")
print()
print("Labeling Parameters:")
print(f"  Entry Threshold: {ENTRY_THRESHOLD:.1%}")
print(f"  Entry Lookforward: {ENTRY_LOOKFORWARD} candles (2 hours)")
print(f"  Exit Patience: {EXIT_PATIENCE} candles (1 hour)")
print()

# Load raw data
print("📖 Loading raw OHLCV data...")
df = pd.read_csv(INPUT_FILE)
df['timestamp'] = pd.to_datetime(df['timestamp'])

print(f"  Rows: {len(df):,}")
print(f"  Period: {df['timestamp'].min()} to {df['timestamp'].max()}")
print(f"  Days: {(df['timestamp'].max() - df['timestamp'].min()).days}")
print()

# ============================================================================
# PART 1: GENERATE ENTRY LABELS
# ============================================================================

print("="*80)
print("PART 1: GENERATING ENTRY LABELS")
print("="*80)
print()

# Calculate future returns
print("📊 Calculating future price movements...")
df['future_max'] = df['high'].rolling(window=ENTRY_LOOKFORWARD, min_periods=1).max().shift(-ENTRY_LOOKFORWARD)
df['future_min'] = df['low'].rolling(window=ENTRY_LOOKFORWARD, min_periods=1).min().shift(-ENTRY_LOOKFORWARD)

# Calculate returns
df['future_max_return'] = (df['future_max'] - df['close']) / df['close']
df['future_min_return'] = (df['future_min'] - df['close']) / df['close']

# Generate labels
df['signal_long'] = (df['future_max_return'] >= ENTRY_THRESHOLD).astype(float)
df['signal_short'] = (df['future_min_return'] <= -ENTRY_THRESHOLD).astype(float)

# Remove last N rows (no future data)
df_entry = df[:-ENTRY_LOOKFORWARD].copy()

print(f"✅ Entry labels generated!")
print(f"  Valid rows: {len(df_entry):,} (lost {ENTRY_LOOKFORWARD} to lookforward)")
print()
print(f"📊 LONG Signal Distribution:")
print(f"  Total: {df_entry['signal_long'].sum():,} ({df_entry['signal_long'].mean():.2%})")
print()
print(f"📊 SHORT Signal Distribution:")
print(f"  Total: {df_entry['signal_short'].sum():,} ({df_entry['signal_short'].mean():.2%})")
print()

# Save Entry Labels
timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
entry_labels_file = LABELS_DIR / f"entry_labels_15min_314days_{timestamp_str}.csv"

df_entry_output = df_entry[['timestamp', 'close', 'signal_long', 'signal_short']].copy()
df_entry_output.to_csv(entry_labels_file, index=False)

print(f"💾 Entry labels saved: {entry_labels_file.name}")
print(f"  File size: {entry_labels_file.stat().st_size / 1024:.2f} KB")
print()

# ============================================================================
# PART 2: GENERATE EXIT LABELS
# ============================================================================

print("="*80)
print("PART 2: GENERATING EXIT LABELS")
print("="*80)
print()

print("📊 Calculating patience-based exit signals...")

# Initialize exit labels
df['long_exit_patience'] = 0.0
df['short_exit_patience'] = 0.0

# For LONG exits: detect when price stops rising
for i in range(EXIT_PATIENCE, len(df) - EXIT_PATIENCE):
    # Look at next EXIT_PATIENCE candles
    future_prices = df['close'].iloc[i+1:i+1+EXIT_PATIENCE].values
    current_price = df['close'].iloc[i]

    # LONG exit: if future prices are declining or stagnant
    if len(future_prices) == EXIT_PATIENCE:
        # Check if price peaked and is declining
        max_future = future_prices.max()
        last_future = future_prices[-1]

        # Exit if:
        # 1. Price rose then fell back
        # 2. Last future price < max future price
        if max_future > current_price and last_future < max_future * 0.99:
            df.loc[df.index[i], 'long_exit_patience'] = 1.0

        # SHORT exit: if future prices are rising
        min_future = future_prices.min()

        # Exit if:
        # 1. Price fell then rose back
        # 2. Last future price > min future price
        if min_future < current_price and last_future > min_future * 1.01:
            df.loc[df.index[i], 'short_exit_patience'] = 1.0

# Remove last N rows (no future data)
df_exit = df[:-EXIT_PATIENCE].copy()

print(f"✅ Exit labels generated!")
print(f"  Valid rows: {len(df_exit):,} (lost {EXIT_PATIENCE} to lookforward)")
print()
print(f"📊 LONG Exit Distribution:")
print(f"  Total: {df_exit['long_exit_patience'].sum():,} ({df_exit['long_exit_patience'].mean():.2%})")
print()
print(f"📊 SHORT Exit Distribution:")
print(f"  Total: {df_exit['short_exit_patience'].sum():,} ({df_exit['short_exit_patience'].mean():.2%})")
print()

# Save Exit Labels (simple format: timestamp + 2 labels)
exit_labels_file = LABELS_DIR / f"exit_labels_15min_314days_{timestamp_str}.csv"

df_exit_output = df_exit[['timestamp', 'long_exit_patience', 'short_exit_patience']].copy()
df_exit_output.to_csv(exit_labels_file, index=False)

print(f"💾 Exit labels saved: {exit_labels_file.name}")
print(f"  File size: {exit_labels_file.stat().st_size / 1024:.2f} KB")
print()

# ============================================================================
# SUMMARY
# ============================================================================

print("="*80)
print("LABEL GENERATION COMPLETE")
print("="*80)
print()
print(f"✅ Entry Labels:")
print(f"   File: {entry_labels_file.name}")
print(f"   Rows: {len(df_entry_output):,}")
print(f"   LONG signals: {df_entry_output['signal_long'].sum():,} ({df_entry_output['signal_long'].mean():.2%})")
print(f"   SHORT signals: {df_entry_output['signal_short'].sum():,} ({df_entry_output['signal_short'].mean():.2%})")
print()
print(f"✅ Exit Labels:")
print(f"   File: {exit_labels_file.name}")
print(f"   Rows: {len(df_exit_output):,}")
print(f"   LONG exits: {df_exit_output['long_exit_patience'].sum():,} ({df_exit_output['long_exit_patience'].mean():.2%})")
print(f"   SHORT exits: {df_exit_output['short_exit_patience'].sum():,} ({df_exit_output['short_exit_patience'].mean():.2%})")
print()

# Quality checks
print("="*80)
print("QUALITY CHECKS")
print("="*80)
print()

# Check label distribution balance
long_ratio = df_entry_output['signal_long'].mean()
short_ratio = df_entry_output['signal_short'].mean()

print("📊 Entry Label Balance:")
if 0.01 <= long_ratio <= 0.20 and 0.01 <= short_ratio <= 0.20:
    print(f"   ✅ GOOD: LONG {long_ratio:.2%}, SHORT {short_ratio:.2%}")
    print(f"   Labels are reasonably balanced (1-20% range)")
else:
    print(f"   ⚠️  WARNING: LONG {long_ratio:.2%}, SHORT {short_ratio:.2%}")
    print(f"   Labels may be imbalanced (<1% or >20%)")
print()

exit_long_ratio = df_exit_output['long_exit_patience'].mean()
exit_short_ratio = df_exit_output['short_exit_patience'].mean()

print("📊 Exit Label Balance:")
if 0.05 <= exit_long_ratio <= 0.50 and 0.05 <= exit_short_ratio <= 0.50:
    print(f"   ✅ GOOD: LONG {exit_long_ratio:.2%}, SHORT {exit_short_ratio:.2%}")
    print(f"   Exit signals are reasonably frequent (5-50% range)")
else:
    print(f"   ⚠️  WARNING: LONG {exit_long_ratio:.2%}, SHORT {exit_short_ratio:.2%}")
    print(f"   Exit signals may be too rare (<5%) or too frequent (>50%)")
print()

print("="*80)
print("NEXT STEPS")
print("="*80)
print()
print("1. Review label distributions above")
print("2. Proceed to model training with Enhanced 5-Fold CV")
print("3. Use these labels for 314-day 15-min dataset training")
print()
print("✅ Label generation complete!")
print()
