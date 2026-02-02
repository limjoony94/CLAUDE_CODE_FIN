"""
Add Missing Exit Features to CSV
==================================

Adds ONLY the 15 missing features needed by Exit models.
Avoids duplicate columns by not recalculating existing features.

Missing features (15 total):
  volume_surge, price_acceleration, price_vs_ma20, price_vs_ma50,
  volatility_20, rsi_slope, rsi_overbought, rsi_oversold, rsi_divergence,
  macd_histogram_slope, macd_crossover, macd_crossunder,
  bb_position, higher_high, near_support

Created: 2025-10-29
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np

DATA_DIR = PROJECT_ROOT / "data" / "features"
INPUT_FILE = DATA_DIR / "BTCUSDT_5m_features.csv"
OUTPUT_FILE = DATA_DIR / "BTCUSDT_5m_features_exit_ready.csv"

print("="*80)
print("ADDING MISSING EXIT FEATURES TO CSV")
print("="*80)
print()

# Load existing CSV
print(f"Loading: {INPUT_FILE}")
df = pd.read_csv(INPUT_FILE)
print(f"✅ Loaded {len(df):,} rows, {len(df.columns)} columns")
print()

# Calculate missing features
print("Calculating 15 missing features...")
print()

# 1. volume_surge
print("  1/15 volume_surge")
df['volume_surge'] = df['volume'] / df['volume'].rolling(20).mean()

# 2. price_acceleration
print("  2/15 price_acceleration")
df['price_acceleration'] = df['close'].diff().diff()

# 3. price_vs_ma20
print("  3/15 price_vs_ma20")
df['price_vs_ma20'] = (df['close'] - df['ma_20']) / df['ma_20']

# 4. price_vs_ma50
print("  4/15 price_vs_ma50")
if 'ma_50' not in df.columns:
    df['ma_50'] = df['close'].rolling(50).mean()
df['price_vs_ma50'] = (df['close'] - df['ma_50']) / df['ma_50']

# 5. volatility_20
print("  5/15 volatility_20")
df['volatility_20'] = df['close'].pct_change().rolling(20).std()

# 6. rsi_slope
print("  6/15 rsi_slope")
df['rsi_slope'] = df['rsi'].diff()

# 7. rsi_overbought
print("  7/15 rsi_overbought")
df['rsi_overbought'] = (df['rsi'] > 70).astype(float)

# 8. rsi_oversold
print("  8/15 rsi_oversold")
df['rsi_oversold'] = (df['rsi'] < 30).astype(float)

# 9. rsi_divergence
print("  9/15 rsi_divergence")
price_change_5 = df['close'].diff(5)
rsi_change_5 = df['rsi'].diff(5)
df['rsi_divergence'] = (
    ((price_change_5 > 0) & (rsi_change_5 < 0)) |  # Bearish divergence
    ((price_change_5 < 0) & (rsi_change_5 > 0))    # Bullish divergence
).astype(float)

# 10. macd_histogram_slope
print(" 10/15 macd_histogram_slope")
macd_histogram = df['macd'] - df['macd_signal']
df['macd_histogram_slope'] = macd_histogram.diff()

# 11. macd_crossover
print(" 11/15 macd_crossover")
df['macd_crossover'] = (
    (df['macd'] > df['macd_signal']) &
    (df['macd'].shift(1) <= df['macd_signal'].shift(1))
).astype(float)

# 12. macd_crossunder
print(" 12/15 macd_crossunder")
df['macd_crossunder'] = (
    (df['macd'] < df['macd_signal']) &
    (df['macd'].shift(1) >= df['macd_signal'].shift(1))
).astype(float)

# 13. bb_position
print(" 13/15 bb_position")
if 'bb_upper' in df.columns and 'bb_lower' in df.columns:
    bb_range = df['bb_upper'] - df['bb_lower']
    df['bb_position'] = (df['close'] - df['bb_lower']) / (bb_range + 1e-10)
else:
    # Calculate Bollinger Bands
    bb_ma = df['close'].rolling(20).mean()
    bb_std = df['close'].rolling(20).std()
    df['bb_upper'] = bb_ma + (2 * bb_std)
    df['bb_lower'] = bb_ma - (2 * bb_std)
    bb_range = df['bb_upper'] - df['bb_lower']
    df['bb_position'] = (df['close'] - df['bb_lower']) / (bb_range + 1e-10)

# 14. higher_high
print(" 14/15 higher_high")
df['higher_high'] = (df['high'] > df['high'].shift(1)).astype(float)

# 15. near_support
print(" 15/15 near_support")
support = df['low'].rolling(20).min()
df['near_support'] = (df['low'] < support.shift(1) * 1.01).astype(float)

print()
print("✅ All 15 features calculated")
print()

# Clean NaN values
print("Cleaning NaN values...")
df = df.ffill().bfill().fillna(0)
print("✅ NaN values cleaned")
print()

# Verify all Exit features now present
exit_features = [
    'close', 'volume', 'volume_surge', 'price_acceleration',
    'ma_20', 'price_vs_ma20', 'price_vs_ma50', 'volatility_20',
    'rsi', 'rsi_slope', 'rsi_overbought', 'rsi_oversold', 'rsi_divergence',
    'macd', 'macd_signal', 'macd_histogram_slope',
    'macd_crossover', 'macd_crossunder',
    'bb_position', 'higher_high', 'near_support'
]

missing = [f for f in exit_features if f not in df.columns]
if missing:
    print(f"⚠️  Still missing: {missing}")
else:
    print(f"✅ All 21 Exit features verified present!")
print()

# Save to new file
print(f"Saving to: {OUTPUT_FILE}")
df.to_csv(OUTPUT_FILE, index=False)
print(f"✅ Saved {len(df):,} rows, {len(df.columns)} columns")
print()

print("="*80)
print("COMPLETE - CSV READY FOR BACKTEST")
print("="*80)
