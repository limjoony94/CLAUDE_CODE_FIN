"""
Calculate Complete Features on 90-Day 5-Minute Dataset
========================================================

Purpose: Calculate ALL features (Entry + Exit) on 90-day 5-min data

Features:
- Entry: 85 (LONG), 79 (SHORT) from production_features_v1.py
- Exit: 27 (both) from production_exit_features_v1.py

Total: ~191 features (complete feature set)

Input: BTCUSDT_5m_raw_90days_20251106_163815.csv
Output: BTCUSDT_5m_features_90days_complete_{timestamp}.csv

Created: 2025-11-06 16:40 KST
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from datetime import datetime

# Import both feature calculators
from scripts.production.production_features_v1 import calculate_all_features_enhanced_v2
from scripts.production.production_exit_features_v1 import prepare_exit_features

# Paths
DATA_DIR = PROJECT_ROOT / "data" / "features"
INPUT_FILE = DATA_DIR / "BTCUSDT_5m_raw_90days_20251106_163815.csv"

print("=" * 80)
print("CALCULATING COMPLETE FEATURES ON 90-DAY 5-MINUTE DATASET")
print("=" * 80)
print()
print(f"📂 Input: {INPUT_FILE.name}")
print(f"📊 Timeframe: 5-minute candles")
print(f"✨ Features: Entry (85/79) + Exit (27) = ~191 total")
print()

# Check if input file exists
if not INPUT_FILE.exists():
    print(f"❌ ERROR: Input file not found: {INPUT_FILE}")
    sys.exit(1)

# Load raw data
print("📖 Loading raw OHLCV data...")
df_raw = pd.read_csv(INPUT_FILE)
df_raw['timestamp'] = pd.to_datetime(df_raw['timestamp'])

print(f"   Rows: {len(df_raw):,}")
print(f"   Period: {df_raw['timestamp'].min()} to {df_raw['timestamp'].max()}")
print(f"   Days: {(df_raw['timestamp'].max() - df_raw['timestamp'].min()).days}")
print(f"   Price Range: ${df_raw['low'].min():,.2f} - ${df_raw['high'].max():,.2f}")
print()

# Calculate Entry features (phase='phase1' for training data)
print("⚙️ Calculating Entry features...")
print("   Phase: 'phase1' (training data)")
print("   Source: production_features_v1.py")
print()

df_entry = calculate_all_features_enhanced_v2(df_raw.copy(), phase='phase1')

print(f"✅ Entry features calculated!")
print(f"   Input rows: {len(df_raw):,}")
print(f"   Output rows: {len(df_entry):,}")
print(f"   Lost to lookback: {len(df_raw) - len(df_entry):,} rows")
print(f"   Features: {len(df_entry.columns)} total")
print()

# Calculate Exit features (on top of Entry features)
print("⚙️ Calculating Exit features...")
print("   Source: production_exit_features_v1.py")
print("   Note: Adds exit-specific features on top of entry features")
print()

df_complete = prepare_exit_features(df_entry.copy())

print(f"✅ Exit features calculated!")
print(f"   Output rows: {len(df_complete):,}")
print(f"   Total features: {len(df_complete.columns)}")
print()

# Data quality check
print("🔍 Feature Quality Check:")
print(f"   NaN values: {df_complete.isna().sum().sum():,}")
print(f"   Inf values: {df_complete.isin([float('inf'), float('-inf')]).sum().sum():,}")
print()

if df_complete.isna().sum().sum() > 0:
    print("⚠️ WARNING: NaN values detected, filling with 0...")
    df_complete = df_complete.fillna(0)
    print("   ✅ NaN values filled")
    print()

if df_complete.isin([float('inf'), float('-inf')]).sum().sum() > 0:
    print("⚠️ WARNING: Inf values detected, clipping...")
    df_complete = df_complete.replace([float('inf'), float('-inf')], 0)
    print("   ✅ Inf values clipped")
    print()

# Save
timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
output_file = DATA_DIR / f"BTCUSDT_5m_features_90days_complete_{timestamp_str}.csv"

print(f"💾 Saving complete feature set to: {output_file.name}")
df_complete.to_csv(output_file, index=False)
print(f"   File size: {output_file.stat().st_size / 1024 / 1024:.2f} MB")
print()

# Training/Validation Split Info
from datetime import timedelta
train_end_timestamp = df_complete['timestamp'].max() - timedelta(days=28)
train_df = df_complete[df_complete['timestamp'] <= train_end_timestamp]
val_df = df_complete[df_complete['timestamp'] > train_end_timestamp]

print("=" * 80)
print("TRAIN/VALIDATION SPLIT PREVIEW")
print("=" * 80)
print()
print(f"📚 Training Set (first ~61 days):")
print(f"   Period: {train_df['timestamp'].min()} to {train_df['timestamp'].max()}")
print(f"   Rows: {len(train_df):,}")
print(f"   Days: {(train_df['timestamp'].max() - train_df['timestamp'].min()).days}")
print()
print(f"✅ Validation Set (last 28 days):")
print(f"   Period: {val_df['timestamp'].min()} to {val_df['timestamp'].max()}")
print(f"   Rows: {len(val_df):,}")
print(f"   Days: {(val_df['timestamp'].max() - val_df['timestamp'].min()).days}")
print()

# Feature list summary
print("=" * 80)
print("FEATURE SUMMARY")
print("=" * 80)
print()
print(f"Total Features: {len(df_complete.columns)}")
print()
print("Key Feature Groups:")
print("  - Price/Volume: open, high, low, close, volume")
print("  - Moving Averages: sma_20, sma_50, ema_12, ema_26, ma_200")
print("  - Momentum: rsi_14, macd, macd_signal, momentum_10")
print("  - Volatility: atr_14, bb_width, volatility_20")
print("  - Volume Profile: vwap, vp_support, vp_resistance")
print("  - Exit Features: volume_surge, price_acceleration, etc.")
print()

print("=" * 80)
print("NEXT STEPS")
print("=" * 80)
print()
print("1. Generate entry and exit labels for 90-day period")
print("2. Retrain 4 models with Enhanced 5-Fold CV:")
print("   → LONG Entry (90 days @ 5-min)")
print("   → SHORT Entry (90 days @ 5-min)")
print("   → LONG Exit (90 days @ 5-min)")
print("   → SHORT Exit (90 days @ 5-min)")
print()
print("3. Split: 61 days training + 28 days validation")
print("4. Compare vs 52-day models (current production)")
print()
print("✅ Feature calculation complete!")
print(f"   File: {output_file}")
print()
