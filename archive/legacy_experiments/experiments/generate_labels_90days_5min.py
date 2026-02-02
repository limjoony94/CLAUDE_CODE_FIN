"""
Generate Entry and Exit Labels for 90-Day 5-Min Dataset
========================================================

Purpose: Create labels for 90-day 5-min complete feature dataset

Entry Labels:
- LONG: 3% rise within 6 candles (30 minutes @ 5-min)
- SHORT: 3% fall within 6 candles (30 minutes @ 5-min)

Exit Labels:
- Patience-based: Hold for profit, exit on reversal
- Max hold: 8 candles (40 minutes @ 5-min)

Input: BTCUSDT_5m_features_90days_complete_20251106_164542.csv
Output: entry_labels_90days_5min_{timestamp}.csv
        exit_labels_90days_5min_{timestamp}.csv

Created: 2025-11-06 16:45 KST
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
INPUT_FILE = DATA_DIR / "BTCUSDT_5m_features_90days_complete_20251106_164542.csv"

# Entry label parameters (5-min candles)
LONG_THRESHOLD = 0.03  # 3% rise
SHORT_THRESHOLD = 0.03  # 3% fall
LOOKFORWARD = 6  # 30 minutes @ 5-min

# Exit label parameters (5-min candles)
MAX_HOLD = 8  # 40 minutes @ 5-min
MIN_PROFIT_EXIT = 0.01  # 1% minimum profit to consider exit

print("=" * 80)
print("GENERATING LABELS FOR 90-DAY 5-MIN DATASET")
print("=" * 80)
print()
print(f"📂 Input: {INPUT_FILE.name}")
print(f"📊 Entry Thresholds: ±{LONG_THRESHOLD*100}% in {LOOKFORWARD} candles ({LOOKFORWARD*5} min)")
print(f"⏱️ Exit Max Hold: {MAX_HOLD} candles ({MAX_HOLD*5} min)")
print()

# Check if input file exists
if not INPUT_FILE.exists():
    print(f"❌ ERROR: Input file not found: {INPUT_FILE}")
    sys.exit(1)

# Load features
print("📖 Loading feature data...")
df = pd.read_csv(INPUT_FILE)
df['timestamp'] = pd.to_datetime(df['timestamp'])

print(f"   Rows: {len(df):,}")
print(f"   Period: {df['timestamp'].min()} to {df['timestamp'].max()}")
print(f"   Days: {(df['timestamp'].max() - df['timestamp'].min()).days}")
print()

# ============================================================================
# ENTRY LABELS
# ============================================================================

print("=" * 80)
print("GENERATING ENTRY LABELS")
print("=" * 80)
print()

def generate_entry_labels(df, long_threshold=0.03, short_threshold=0.03, lookforward=6):
    """Generate entry labels for LONG and SHORT positions"""

    n = len(df)
    long_labels = np.zeros(n, dtype=int)
    short_labels = np.zeros(n, dtype=int)

    for i in range(n - lookforward):
        current_price = df.iloc[i]['close']

        # Look ahead for price movements
        future_prices = df.iloc[i+1:i+1+lookforward]['close'].values

        if len(future_prices) == 0:
            continue

        # LONG label: Price rises by threshold
        max_future = np.max(future_prices)
        if (max_future - current_price) / current_price >= long_threshold:
            long_labels[i] = 1

        # SHORT label: Price falls by threshold
        min_future = np.min(future_prices)
        if (current_price - min_future) / current_price >= short_threshold:
            short_labels[i] = 1

    return long_labels, short_labels

print(f"⚙️ Calculating entry labels...")
print(f"   LONG threshold: {LONG_THRESHOLD*100}% rise in {LOOKFORWARD} candles")
print(f"   SHORT threshold: {SHORT_THRESHOLD*100}% fall in {LOOKFORWARD} candles")
print()

long_labels, short_labels = generate_entry_labels(
    df,
    long_threshold=LONG_THRESHOLD,
    short_threshold=SHORT_THRESHOLD,
    lookforward=LOOKFORWARD
)

# Create entry labels dataframe
df_entry_labels = df[['timestamp', 'close']].copy()
df_entry_labels['long_entry_label'] = long_labels
df_entry_labels['short_entry_label'] = short_labels

# Statistics
long_count = np.sum(long_labels)
short_count = np.sum(short_labels)
long_pct = long_count / len(df) * 100
short_pct = short_count / len(df) * 100

print(f"✅ Entry labels generated:")
print(f"   LONG entries: {long_count:,} ({long_pct:.2f}%)")
print(f"   SHORT entries: {short_count:,} ({short_pct:.2f}%)")
print(f"   Total labeled: {long_count + short_count:,} ({(long_pct + short_pct):.2f}%)")
print()

# ============================================================================
# EXIT LABELS
# ============================================================================

print("=" * 80)
print("GENERATING EXIT LABELS")
print("=" * 80)
print()

def generate_exit_labels(df, max_hold=8, min_profit=0.01):
    """Generate exit labels for LONG and SHORT positions (patience-based)"""

    n = len(df)
    long_exit_labels = np.zeros(n, dtype=int)
    short_exit_labels = np.zeros(n, dtype=int)

    for i in range(n - max_hold):
        current_price = df.iloc[i]['close']

        # Look ahead for exit opportunities
        future_window = df.iloc[i+1:i+1+max_hold]

        if len(future_window) == 0:
            continue

        # LONG exit: Find profitable exit or reversal
        for j, (idx, row) in enumerate(future_window.iterrows()):
            future_price = row['close']
            profit_pct = (future_price - current_price) / current_price

            # Exit if profit >= min_profit and next candle reverses
            if profit_pct >= min_profit:
                if j + 1 < len(future_window):
                    next_price = future_window.iloc[j + 1]['close']
                    if next_price < future_price:  # Reversal detected
                        long_exit_labels[i + j + 1] = 1
                        break
                else:
                    # End of window, exit
                    long_exit_labels[i + j + 1] = 1
                    break

        # SHORT exit: Find profitable exit or reversal
        for j, (idx, row) in enumerate(future_window.iterrows()):
            future_price = row['close']
            profit_pct = (current_price - future_price) / current_price

            # Exit if profit >= min_profit and next candle reverses
            if profit_pct >= min_profit:
                if j + 1 < len(future_window):
                    next_price = future_window.iloc[j + 1]['close']
                    if next_price > future_price:  # Reversal detected
                        short_exit_labels[i + j + 1] = 1
                        break
                else:
                    # End of window, exit
                    short_exit_labels[i + j + 1] = 1
                    break

    return long_exit_labels, short_exit_labels

print(f"⚙️ Calculating exit labels (patience-based)...")
print(f"   Max hold: {MAX_HOLD} candles ({MAX_HOLD*5} min)")
print(f"   Min profit: {MIN_PROFIT_EXIT*100}%")
print()

long_exit_labels, short_exit_labels = generate_exit_labels(
    df,
    max_hold=MAX_HOLD,
    min_profit=MIN_PROFIT_EXIT
)

# Create exit labels dataframe
df_exit_labels = df[['timestamp', 'close']].copy()
df_exit_labels['long_exit_label'] = long_exit_labels
df_exit_labels['short_exit_label'] = short_exit_labels

# Statistics
long_exit_count = np.sum(long_exit_labels)
short_exit_count = np.sum(short_exit_labels)
long_exit_pct = long_exit_count / len(df) * 100
short_exit_pct = short_exit_count / len(df) * 100

print(f"✅ Exit labels generated:")
print(f"   LONG exits: {long_exit_count:,} ({long_exit_pct:.2f}%)")
print(f"   SHORT exits: {short_exit_count:,} ({short_exit_pct:.2f}%)")
print(f"   Total labeled: {long_exit_count + short_exit_count:,} ({(long_exit_pct + short_exit_pct):.2f}%)")
print()

# ============================================================================
# SAVE LABELS
# ============================================================================

print("=" * 80)
print("SAVING LABELS")
print("=" * 80)
print()

timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')

# Save entry labels
entry_file = LABELS_DIR / f"entry_labels_90days_5min_{timestamp_str}.csv"
df_entry_labels.to_csv(entry_file, index=False)
print(f"💾 Entry labels saved: {entry_file.name}")
print(f"   Size: {entry_file.stat().st_size / 1024 / 1024:.2f} MB")
print()

# Save exit labels
exit_file = LABELS_DIR / f"exit_labels_90days_5min_{timestamp_str}.csv"
df_exit_labels.to_csv(exit_file, index=False)
print(f"💾 Exit labels saved: {exit_file.name}")
print(f"   Size: {exit_file.stat().st_size / 1024 / 1024:.2f} MB")
print()

# ============================================================================
# SUMMARY
# ============================================================================

print("=" * 80)
print("LABEL GENERATION SUMMARY")
print("=" * 80)
print()

print(f"📊 Dataset:")
print(f"   Period: {df['timestamp'].min()} to {df['timestamp'].max()}")
print(f"   Rows: {len(df):,}")
print(f"   Days: {(df['timestamp'].max() - df['timestamp'].min()).days}")
print()

print(f"📈 Entry Labels:")
print(f"   LONG: {long_count:,} ({long_pct:.2f}%) - {LONG_THRESHOLD*100}% rise in {LOOKFORWARD*5} min")
print(f"   SHORT: {short_count:,} ({short_pct:.2f}%) - {SHORT_THRESHOLD*100}% fall in {LOOKFORWARD*5} min")
print()

print(f"🚪 Exit Labels:")
print(f"   LONG: {long_exit_count:,} ({long_exit_pct:.2f}%) - patience-based, max {MAX_HOLD*5} min")
print(f"   SHORT: {short_exit_count:,} ({short_exit_pct:.2f}%) - patience-based, max {MAX_HOLD*5} min")
print()

print("=" * 80)
print("NEXT STEPS")
print("=" * 80)
print()
print("1. Train 4 models with Enhanced 5-Fold CV:")
print("   → LONG Entry (90 days @ 5-min)")
print("   → SHORT Entry (90 days @ 5-min)")
print("   → LONG Exit (90 days @ 5-min)")
print("   → SHORT Exit (90 days @ 5-min)")
print()
print("2. Split: 61 days training + 28 days validation")
print("3. Backtest on 28-day out-of-sample period")
print("4. Compare vs 52-day models (current production)")
print()
print("✅ Label generation complete!")
print(f"   Entry labels: {entry_file}")
print(f"   Exit labels: {exit_file}")
print()
