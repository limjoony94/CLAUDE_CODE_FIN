"""
Calculate Features on 314-Day 15-Minute Dataset
================================================

Purpose: Calculate all 195 features on maximum historical 15-min candle data

Input: BTCUSDT_15m_raw_314days_20251106_143614.csv (30,240 candles @ 15-min)
Output: Feature-engineered dataset ready for training

Features: 195 total
  - 79 base features (used by current models)
  - 85 LONG Entry features (Enhanced 5-Fold CV)
  - 89 SHORT Entry features (includes 10 NEW SHORT-specific features)
  - 27 Exit features each (LONG/SHORT)

Dataset Span: Dec 26, 2024 - Nov 6, 2025 (314 days)
Includes: 2024 bear market ($74K-$80K) + full 2025 data

Created: 2025-11-06 14:40 KST
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
INPUT_FILE = DATA_DIR / "BTCUSDT_15m_raw_314days_20251106_143614.csv"

print("="*80)
print("CALCULATING FEATURES ON 314-DAY 15-MINUTE DATASET")
print("="*80)
print()
print(f"📂 Input: {INPUT_FILE.name}")
print(f"📊 Timeframe: 15-minute candles (3× longer history vs 5-min)")
print(f"📊 Feature Calculator: calculate_all_features_enhanced_v2 (production)")
print(f"✨ Includes: 10 NEW SHORT-specific features")
print()

# Check if input file exists
if not INPUT_FILE.exists():
    print(f"❌ ERROR: Input file not found: {INPUT_FILE}")
    print("   Please run fetch_15min_historical_max.py first")
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

# Calculate features (phase='phase1' for training data)
print("⚙️ Calculating features (this may take 3-5 minutes)...")
print("   Phase: 'phase1' (training data)")
print("   Note: 15-min timeframe may require different lookback periods")
print()

df_features = calculate_all_features_enhanced_v2(df_raw.copy(), phase='phase1')

print()
print(f"✅ Features calculated!")
print(f"   Input rows: {len(df_raw):,}")
print(f"   Output rows: {len(df_features):,}")
print(f"   Lost to lookback: {len(df_raw) - len(df_features):,} rows")
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
output_file = DATA_DIR / f"BTCUSDT_15m_features_314days_{timestamp_str}.csv"

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
print(f"📚 Training Set (first 286 days):")
print(f"   Period: {train_df['timestamp'].min()} to {train_df['timestamp'].max()}")
print(f"   Rows: {len(train_df):,}")
print(f"   Days: {(train_df['timestamp'].max() - train_df['timestamp'].min()).days}")
print(f"   Price Range: ${train_df['close'].min():,.2f} - ${train_df['close'].max():,.2f}")
print()
print(f"✅ Validation Set (last 28 days):")
print(f"   Period: {val_df['timestamp'].min()} to {val_df['timestamp'].max()}")
print(f"   Rows: {len(val_df):,}")
print(f"   Days: {(val_df['timestamp'].max() - val_df['timestamp'].min()).days}")
print(f"   Price Range: ${val_df['close'].min():,.2f} - ${val_df['close'].max():,.2f}")
print()

# Market regime analysis
print("="*80)
print("MARKET REGIME COVERAGE")
print("="*80)
print()
print("📊 Full Dataset Statistics:")
print(f"   Min Price: ${df_features['close'].min():,.2f} (Bear market low)")
print(f"   Max Price: ${df_features['close'].max():,.2f} (Bull market high)")
print(f"   Mean Price: ${df_features['close'].mean():,.2f}")
print(f"   Std Dev: ${df_features['close'].std():,.2f}")
print()
print("🔍 Regime Identification:")

# Define regimes based on price levels
bear_market = df_features[df_features['close'] < 80000]
transition = df_features[(df_features['close'] >= 80000) & (df_features['close'] < 100000)]
bull_market = df_features[df_features['close'] >= 100000]

print(f"   Bear Market (<$80K): {len(bear_market):,} candles ({len(bear_market)/len(df_features)*100:.1f}%)")
print(f"   Transition ($80K-$100K): {len(transition):,} candles ({len(transition)/len(df_features)*100:.1f}%)")
print(f"   Bull Market (>$100K): {len(bull_market):,} candles ({len(bull_market)/len(df_features)*100:.1f}%)")
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
print("1. Create retraining script with 286d train + 28d validation split")
print()
print("2. Retrain 4 models with 15-min candles:")
print("   → LONG Entry (286 days, 15-min)")
print("   → SHORT Entry (286 days, 15-min, with NEW features)")
print("   → LONG Exit (286 days, 15-min)")
print("   → SHORT Exit (286 days, 15-min)")
print()
print("3. Backtest on validation period (28 days)")
print()
print("4. Compare vs current 5-min models:")
print("   → Current: 52 days @ 5-min (15,003 candles)")
print("   → New: 286 days @ 15-min (~27,500 candles)")
print("   → Advantage: 5.5× longer training period")
print()
print("5. Assess if 15-min timeframe better captures market regime changes")
print()
print("✅ Feature calculation complete!")
print(f"   File: {output_file}")
print()
