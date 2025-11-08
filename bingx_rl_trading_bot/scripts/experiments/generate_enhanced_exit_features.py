"""
Generate Enhanced Exit Features
================================

Add 15 enhanced exit features to the base 495-day dataset for Patience Exit model training.

Problem: Patience exit models trained with only 6 features (close, volume, ma_20, rsi, macd, macd_signal)
Solution: Add 15 missing features to match the 21 features used by threshold_075 Exit models

Missing Features (15):
  1. volume_surge
  2. price_acceleration
  3. price_vs_ma20
  4. price_vs_ma50
  5. volatility_20
  6. rsi_slope
  7. rsi_overbought
  8. rsi_oversold
  9. rsi_divergence
  10. macd_histogram_slope
  11. macd_crossover
  12. macd_crossunder
  13. bb_position
  14. higher_high
  15. near_support

Created: 2025-10-30
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import talib
from datetime import datetime

# Directories
DATA_DIR = PROJECT_ROOT / "data" / "features"
OUTPUT_DIR = PROJECT_ROOT / "data" / "features"
OUTPUT_DIR.mkdir(exist_ok=True)

print("=" * 80)
print("GENERATE ENHANCED EXIT FEATURES")
print("=" * 80)
print()
print("Task: Add 15 missing features to base dataset (495 days)")
print("Target: 21 total exit features for Patience Exit model retraining")
print()

# Load base dataset
print("-" * 80)
print("STEP 1: Loading Base Dataset")
print("-" * 80)

base_file = DATA_DIR / "BTCUSDT_5m_features.csv"
df = pd.read_csv(base_file)
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values('timestamp').reset_index(drop=True)

print(f"✅ Loaded: {len(df):,} candles")
print(f"   Date range: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")
print(f"   Days: {(df['timestamp'].max() - df['timestamp'].min()).days}")
print()

# Check existing features
print("-" * 80)
print("STEP 2: Checking Existing Features")
print("-" * 80)

existing_features = set(df.columns)
required_base_features = ['close', 'volume', 'ma_20', 'rsi', 'macd', 'macd_signal']

print("Existing base features:")
for feat in required_base_features:
    status = "✅" if feat in existing_features else "❌"
    print(f"  {status} {feat}")

# Verify all base features exist
missing_base = [f for f in required_base_features if f not in existing_features]
if missing_base:
    print(f"\n❌ ERROR: Missing base features: {missing_base}")
    print("   Cannot proceed without base features")
    sys.exit(1)

print("\n✅ All base features present")
print()

# Generate enhanced exit features
print("-" * 80)
print("STEP 3: Generating Enhanced Exit Features (15 features)")
print("-" * 80)
print()

# ========================================================================
# 1-2. Volume Features (2 features)
# ========================================================================
print("Calculating volume features (2)...")

# volume_surge: High volume flag (> 1.5× 20-period average)
volume_ma20 = df['volume'].rolling(20).mean()
df['volume_surge'] = (df['volume'] > volume_ma20 * 1.5).astype(float)

print("  ✅ volume_surge")

# ========================================================================
# 3-5. Price Features (3 features)
# ========================================================================
print("\nCalculating price features (3)...")

# price_vs_ma20: Price relative to 20-period MA
ma_20 = df['close'].rolling(20).mean()
df['price_vs_ma20'] = (df['close'] - ma_20) / ma_20

# price_vs_ma50: Price relative to 50-period MA
ma_50 = df['close'].rolling(50).mean()
df['price_vs_ma50'] = (df['close'] - ma_50) / ma_50

# price_acceleration: Second derivative of price (rate of change of momentum)
df['price_acceleration'] = df['close'].diff().diff()

print("  ✅ price_vs_ma20")
print("  ✅ price_vs_ma50")
print("  ✅ price_acceleration")

# ========================================================================
# 6. Volatility Feature (1 feature)
# ========================================================================
print("\nCalculating volatility feature (1)...")

# volatility_20: 20-period standard deviation of returns
df['returns'] = df['close'].pct_change()
df['volatility_20'] = df['returns'].rolling(20).std()

print("  ✅ volatility_20")

# ========================================================================
# 7-10. RSI Features (4 features)
# ========================================================================
print("\nCalculating RSI features (4)...")

# rsi_slope: Rate of change of RSI over 3 candles
df['rsi_slope'] = df['rsi'].diff(3) / 3

# rsi_overbought: RSI > 70 flag
df['rsi_overbought'] = (df['rsi'] > 70).astype(float)

# rsi_oversold: RSI < 30 flag
df['rsi_oversold'] = (df['rsi'] < 30).astype(float)

# rsi_divergence: Price vs RSI direction mismatch
price_change_5 = df['close'].diff(5)
rsi_change_5 = df['rsi'].diff(5)
df['rsi_divergence'] = (
    ((price_change_5 > 0) & (rsi_change_5 < 0)) |  # Bearish divergence
    ((price_change_5 < 0) & (rsi_change_5 > 0))    # Bullish divergence
).astype(float)

print("  ✅ rsi_slope")
print("  ✅ rsi_overbought")
print("  ✅ rsi_oversold")
print("  ✅ rsi_divergence")

# ========================================================================
# 11-13. MACD Features (3 features)
# ========================================================================
print("\nCalculating MACD features (3)...")

# First, ensure we have macd_histogram (might not exist in base)
if 'macd_histogram' not in df.columns:
    # Recalculate MACD to get histogram
    macd, macd_signal_calc, macd_hist = talib.MACD(
        df['close'], fastperiod=12, slowperiod=26, signalperiod=9
    )
    df['macd_histogram'] = macd_hist
    print("  ℹ️  Calculated macd_histogram from MACD")

# macd_histogram_slope: Rate of change of MACD histogram
df['macd_histogram_slope'] = df['macd_histogram'].diff(3) / 3

# macd_crossover: MACD crosses above signal line
df['macd_crossover'] = (
    (df['macd'] > df['macd_signal']) &
    (df['macd'].shift(1) <= df['macd_signal'].shift(1))
).astype(float)

# macd_crossunder: MACD crosses below signal line
df['macd_crossunder'] = (
    (df['macd'] < df['macd_signal']) &
    (df['macd'].shift(1) >= df['macd_signal'].shift(1))
).astype(float)

print("  ✅ macd_histogram_slope")
print("  ✅ macd_crossover")
print("  ✅ macd_crossunder")

# ========================================================================
# 14. Bollinger Band Feature (1 feature)
# ========================================================================
print("\nCalculating Bollinger Band feature (1)...")

# Calculate Bollinger Bands if not present
if 'bb_upper' not in df.columns or 'bb_lower' not in df.columns:
    bb_upper, bb_middle, bb_lower = talib.BBANDS(
        df['close'], timeperiod=20, nbdevup=2, nbdevdn=2
    )
    df['bb_upper'] = bb_upper
    df['bb_lower'] = bb_lower
    print("  ℹ️  Calculated Bollinger Bands")

# bb_position: Normalized position within Bollinger Bands (0-1)
df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
df['bb_position'] = df['bb_position'].fillna(0.5)  # Handle division by zero

print("  ✅ bb_position")

# ========================================================================
# 15-16. Price Pattern Features (2 features)
# ========================================================================
print("\nCalculating price pattern features (2)...")

# higher_high: Current high > previous high
df['higher_high'] = (df['high'] > df['high'].shift(1)).astype(float)

# near_support: Price near recent low (simplified version)
# Use 20-period rolling min as support level
support_level = df['low'].rolling(20).min()
df['near_support'] = (df['close'] < support_level * 1.02).astype(float)

print("  ✅ higher_high")
print("  ✅ near_support")

print()

# Clean NaN values
print("-" * 80)
print("STEP 4: Cleaning NaN Values")
print("-" * 80)

rows_before = len(df)
df = df.dropna().reset_index(drop=True)
rows_after = len(df)
rows_lost = rows_before - rows_after

print(f"Rows before: {rows_before:,}")
print(f"Rows after: {rows_after:,}")
print(f"Rows lost: {rows_lost:,} ({rows_lost/rows_before*100:.1f}%)")
print()

# Verify all 21 exit features are present
print("-" * 80)
print("STEP 5: Verifying All 21 Exit Features")
print("-" * 80)

required_exit_features = [
    'close', 'volume', 'volume_surge', 'price_acceleration',
    'ma_20', 'price_vs_ma20', 'price_vs_ma50', 'volatility_20',
    'rsi', 'rsi_slope', 'rsi_overbought', 'rsi_oversold', 'rsi_divergence',
    'macd', 'macd_signal', 'macd_histogram_slope',
    'macd_crossover', 'macd_crossunder',
    'bb_position', 'higher_high', 'near_support'
]

print(f"Required exit features: {len(required_exit_features)}")
print()

all_present = True
for feat in required_exit_features:
    if feat in df.columns:
        print(f"  ✅ {feat}")
    else:
        print(f"  ❌ {feat} - MISSING!")
        all_present = False

if not all_present:
    print("\n❌ ERROR: Some features are missing!")
    sys.exit(1)

print("\n✅ All 21 exit features verified")
print()

# Save enhanced dataset
print("-" * 80)
print("STEP 6: Saving Enhanced Dataset")
print("-" * 80)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = OUTPUT_DIR / f"BTCUSDT_5m_features_enhanced_exit_{timestamp}.csv"

# Also save without timestamp for easy reference
output_file_latest = OUTPUT_DIR / "BTCUSDT_5m_features_enhanced_exit.csv"

df.to_csv(output_file, index=False)
df.to_csv(output_file_latest, index=False)

print(f"✅ Saved: {output_file.name}")
print(f"✅ Saved: {output_file_latest.name}")
print(f"   Rows: {len(df):,}")
print(f"   Features: {len(df.columns)}")
print()

# Summary
print("=" * 80)
print("ENHANCED EXIT FEATURES GENERATION COMPLETE")
print("=" * 80)
print()
print("Results:")
print(f"  Input: BTCUSDT_5m_features.csv ({rows_before:,} rows)")
print(f"  Output: {output_file_latest.name} ({rows_after:,} rows)")
print(f"  Features added: 15")
print(f"  Total exit features: 21")
print()
print("Next Steps:")
print("  1. Update retrain_patience_exit_models.py to use enhanced dataset")
print("  2. Retrain Patience Exit models with 21 features")
print("  3. Run backtest validation")
print()
