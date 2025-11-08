"""
Calculate Features on 90-Day Dataset for Full Model Retraining
===============================================================

Purpose: Calculate all 195 features (including NEW SHORT-specific features)

Input: BTCUSDT_5m_raw_90days_20251105_010144.csv (25,920 candles)
Output: Feature-engineered dataset ready for training

Features: 195 total
  - 79 base features (used by current models)
  - 85 LONG Entry features (Enhanced 5-Fold CV)
  - 89 SHORT Entry features (includes 10 NEW SHORT-specific features)
  - 27 Exit features each (LONG/SHORT)

Created: 2025-11-05 01:05 KST
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from datetime import datetime

# Import production feature calculation
from scripts.production.production_features_v1 import calculate_all_features_enhanced_v2

# Paths
DATA_DIR = PROJECT_ROOT / "data" / "features"
INPUT_FILE = DATA_DIR / "BTCUSDT_5m_raw_90days_20251105_010144.csv"

print("="*80)
print("CALCULATING FEATURES ON 90-DAY DATASET")
print("="*80)
print()
print(f"📂 Input: {INPUT_FILE.name}")
print(f"📊 Feature Calculator: calculate_all_features_enhanced_v2 (production)")
print(f"✨ Includes: 10 NEW SHORT-specific features")
print()

# Load raw data
print("📖 Loading raw OHLCV data...")
df_raw = pd.read_csv(INPUT_FILE)
df_raw['timestamp'] = pd.to_datetime(df_raw['timestamp'])

print(f"   Rows: {len(df_raw):,}")
print(f"   Period: {df_raw['timestamp'].min()} to {df_raw['timestamp'].max()}")
print(f"   Days: {(df_raw['timestamp'].max() - df_raw['timestamp'].min()).days}")
print()

# Calculate features (phase='phase1' for training data)
print("⚙️ Calculating features (this may take 2-3 minutes)...")
print("   Phase: 'phase1' (training data)")
print()

df_features = calculate_all_features_enhanced_v2(df_raw.copy(), phase='phase1')

print()
print(f"✅ Features calculated!")
print(f"   Output rows: {len(df_features):,}")
print(f"   Features: {len(df_features.columns)} total")
print()

# Data quality check
print("🔍 Feature Quality Check:")
print(f"   NaN values: {df_features.isna().sum().sum():,}")
print(f"   Inf values: {df_features.isin([float('inf'), float('-inf')]).sum().sum():,}")
print()

if df_features.isna().sum().sum() > 0:
    print("⚠️ WARNING: NaN values detected, filling with 0...")
    df_features = df_features.fillna(0)
    print("   ✅ NaN values filled")
    print()

if df_features.isin([float('inf'), float('-inf')]).sum().sum() > 0:
    print("⚠️ WARNING: Inf values detected, clipping...")
    df_features = df_features.replace([float('inf'), float('-inf')], 0)
    print("   ✅ Inf values clipped")
    print()

# Save
timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
output_file = DATA_DIR / f"BTCUSDT_5m_features_90days_{timestamp_str}.csv"

print(f"💾 Saving features to: {output_file.name}")
df_features.to_csv(output_file, index=False)
print(f"   File size: {output_file.stat().st_size / 1024 / 1024:.2f} MB")
print()

# Training/Validation Split Info
from datetime import timedelta
train_end_timestamp = df_features['timestamp'].max() - timedelta(days=28)
train_df = df_features[df_features['timestamp'] <= train_end_timestamp]
val_df = df_features[df_features['timestamp'] > train_end_timestamp]

print("="*80)
print("TRAIN/VALIDATION SPLIT")
print("="*80)
print()
print(f"📚 Training Set (first 62 days):")
print(f"   Period: {train_df['timestamp'].min()} to {train_df['timestamp'].max()}")
print(f"   Rows: {len(train_df):,}")
print(f"   Days: {(train_df['timestamp'].max() - train_df['timestamp'].min()).days}")
print()
print(f"✅ Validation Set (last 28 days):")
print(f"   Period: {val_df['timestamp'].min()} to {val_df['timestamp'].max()}")
print(f"   Rows: {len(val_df):,}")
print(f"   Days: {(val_df['timestamp'].max() - val_df['timestamp'].min()).days}")
print()

# Feature list
print("="*80)
print("FEATURE SUMMARY")
print("="*80)
print()
print(f"Total Features: {len(df_features.columns)}")
print()
print("Key Feature Groups:")
print("  - Price/Volume: open, high, low, close, volume")
print("  - Moving Averages: sma_20, sma_50, ema_12, ema_26, ma_200")
print("  - Momentum: rsi_14, macd, macd_signal, momentum_10")
print("  - Volatility: atr_14, bb_width, volatility_20")
print("  - Volume Profile: vwap, vp_support, vp_resistance")
print("  - SHORT-specific (NEW): downtrend_strength, price_below_ma200_pct, etc.")
print()

print("="*80)
print("NEXT STEPS")
print("="*80)
print()
print("1. Create retraining script with 62d train + 28d validation split")
print()
print("2. Retrain 4 models:")
print("   → LONG Entry (62 days)")
print("   → SHORT Entry (62 days, with NEW features)")
print("   → LONG Exit (62 days)")
print("   → SHORT Exit (62 days)")
print()
print("3. Backtest on validation period (28 days)")
print()
print("4. Compare vs current models:")
print("   → Current LONG: 104 days training, 85 features")
print("   → Current SHORT: 35 days training, 79 features (OLD)")
print("   → New LONG: 62 days training, 85 features")
print("   → New SHORT: 62 days training, 89 features (NEW)")
print()
print("5. Decide deployment based on validation performance")
print()
print("✅ Feature calculation complete!")
print(f"   File: {output_file}")
print()
