"""
Generate RELAXED Entry and Exit Labels for 90-Day 5-Min Dataset
================================================================

Purpose: Create relaxed labels to ensure sufficient positive samples

ADJUSTED PARAMETERS for 5-min candles:
- Entry: 2% movement in 12 candles (60 min = 1 hour)
- Exit: Patience-based, max 16 candles (80 min)

Rationale:
- Original 3% in 30min too strict → only 10 labels total
- Need ~5-10% positive labels for model training
- 2% in 60min more realistic for 5-min timeframe

Input: BTCUSDT_5m_features_90days_complete_20251106_164542.csv
Output: entry_labels_90days_5min_relaxed_{timestamp}.csv
        exit_labels_90days_5min_relaxed_{timestamp}.csv

Created: 2025-11-06 17:05 KST
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

# MAXIMUM RELAXED entry label parameters (5-min candles)
LONG_THRESHOLD = 0.015  # 1.5% rise (was 3% → 2%)
SHORT_THRESHOLD = 0.015  # 1.5% fall (was 3% → 2%)
LOOKFORWARD = 24  # 120 minutes @ 5-min (was 6 = 30 min → 12 = 60 min)

# RELAXED exit label parameters (5-min candles)
MAX_HOLD = 16  # 80 minutes @ 5-min (was 8 = 40 min)
MIN_PROFIT_EXIT = 0.005  # 0.5% minimum profit (was 1%)

print("=" * 80)
print("GENERATING RELAXED LABELS FOR 90-DAY 5-MIN DATASET")
print("=" * 80)
print()
print(f"📂 Input: {INPUT_FILE.name}")
print(f"📊 Entry Thresholds: ±{LONG_THRESHOLD*100}% in {LOOKFORWARD} candles ({LOOKFORWARD*5} min)")
print(f"   Previous: ±3.0% in 6 candles (30 min) → Only 10 labels total")
print(f"   Adjusted: ±{LONG_THRESHOLD*100}% in {LOOKFORWARD} candles ({LOOKFORWARD*5} min)")
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
# ENTRY LABELS (RELAXED)
# ============================================================================

print("=" * 80)
print("GENERATING ENTRY LABELS (RELAXED)")
print("=" * 80)
print()

def generate_entry_labels(df, long_threshold=0.02, short_threshold=0.02, lookforward=12):
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

if long_pct < 1.0 or short_pct < 1.0:
    print(f"⚠️  WARNING: Label distribution still very low (<1%)")
    print(f"   Consider further relaxing parameters:")
    print(f"   - Lower threshold: 1.5% instead of {LONG_THRESHOLD*100}%")
    print(f"   - Longer window: 24 candles (2 hours) instead of {LOOKFORWARD}")
    print()

# ============================================================================
# EXIT LABELS (RELAXED)
# ============================================================================

print("=" * 80)
print("GENERATING EXIT LABELS (RELAXED)")
print("=" * 80)
print()

def generate_exit_labels(df, max_hold=16, min_profit=0.005):
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
entry_file = LABELS_DIR / f"entry_labels_90days_5min_relaxed_{timestamp_str}.csv"
df_entry_labels.to_csv(entry_file, index=False)
print(f"💾 Entry labels saved: {entry_file.name}")
print(f"   Size: {entry_file.stat().st_size / 1024 / 1024:.2f} MB")
print()

# Save exit labels
exit_file = LABELS_DIR / f"exit_labels_90days_5min_relaxed_{timestamp_str}.csv"
df_exit_labels.to_csv(exit_file, index=False)
print(f"💾 Exit labels saved: {exit_file.name}")
print(f"   Size: {exit_file.stat().st_size / 1024 / 1024:.2f} MB")
print()

# ============================================================================
# COMPARISON WITH ORIGINAL STRICT LABELS
# ============================================================================

print("=" * 80)
print("COMPARISON: RELAXED vs STRICT LABELS")
print("=" * 80)
print()

print(f"📊 Strict Labels (3% in 30min):")
print(f"   LONG: 3 (0.01%)")
print(f"   SHORT: 7 (0.03%)")
print(f"   Total: 10 (0.04%) ❌ TOO SPARSE")
print()

print(f"📊 Relaxed Labels ({LONG_THRESHOLD*100}% in {LOOKFORWARD*5}min):")
print(f"   LONG: {long_count:,} ({long_pct:.2f}%)")
print(f"   SHORT: {short_count:,} ({short_pct:.2f}%)")
print(f"   Total: {long_count + short_count:,} ({(long_pct + short_pct):.2f}%)")
print()

improvement_factor = (long_count + short_count) / 10
print(f"📈 Improvement: {improvement_factor:.1f}x more labels")
print()

# Check if still need more relaxation
if (long_pct + short_pct) < 5.0:
    print(f"⚠️  RECOMMENDATION: Labels still sparse (<5% total)")
    print(f"   For optimal training, consider:")
    print(f"   - Target: 5-10% positive labels")
    print(f"   - Option A: 1.5% threshold in 60min")
    print(f"   - Option B: 2.0% threshold in 120min")
    print()
elif (long_pct + short_pct) > 15.0:
    print(f"⚠️  WARNING: Labels might be too relaxed (>15% total)")
    print(f"   May include low-quality signals")
    print()
else:
    print(f"✅ Label distribution looks good (5-15% range)")
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

print(f"📈 Entry Labels (RELAXED):")
print(f"   LONG: {long_count:,} ({long_pct:.2f}%) - {LONG_THRESHOLD*100}% rise in {LOOKFORWARD*5} min")
print(f"   SHORT: {short_count:,} ({short_pct:.2f}%) - {SHORT_THRESHOLD*100}% fall in {LOOKFORWARD*5} min")
print()

print(f"🚪 Exit Labels (RELAXED):")
print(f"   LONG: {long_exit_count:,} ({long_exit_pct:.2f}%) - patience-based, max {MAX_HOLD*5} min")
print(f"   SHORT: {short_exit_count:,} ({short_exit_pct:.2f}%) - patience-based, max {MAX_HOLD*5} min")
print()

print("=" * 80)
print("NEXT STEPS")
print("=" * 80)
print()
print("1. If labels sufficient (5-10%), proceed with training")
print("2. If still sparse (<5%), further relax parameters")
print("3. Train 4 models with Enhanced 5-Fold CV")
print("4. Validate on 28-day out-of-sample period")
print()
print("✅ Relaxed label generation complete!")
print(f"   Entry labels: {entry_file}")
print(f"   Exit labels: {exit_file}")
print()
