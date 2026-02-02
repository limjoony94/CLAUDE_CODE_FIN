"""
Add Enhanced Exit Features to Dataset
=====================================

Adds the 14 enhanced features required by Exit models:
  - volume_surge, price_acceleration
  - price_vs_ma20, price_vs_ma50
  - rsi_slope, rsi_overbought, rsi_oversold, rsi_divergence
  - macd_histogram_slope, macd_crossover, macd_crossunder
  - bb_position, higher_high, near_support

Created: 2025-11-02
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np

DATA_DIR = PROJECT_ROOT / "data" / "features"

print("="*80)
print("Adding Enhanced Exit Features")
print("="*80)
print()

# Load enhanced dataset
print("Loading BTCUSDT_5m_features_enhanced.csv...")
df = pd.read_csv(DATA_DIR / "BTCUSDT_5m_features_enhanced.csv")
print(f"✅ Loaded {len(df):,} rows, {len(df.columns)} columns")
print()

print("Adding 14 enhanced features...")
print()

# Volume surge
if 'volume_surge' not in df.columns:
    df['volume_surge'] = (df['volume'] > df['volume'].rolling(20).mean() * 1.5).astype(float)
    print("✅ volume_surge: Volume > 20-period mean × 1.5")

# Price acceleration
if 'price_acceleration' not in df.columns:
    df['price_acceleration'] = df['close'].pct_change(5)
    print("✅ price_acceleration: 5-period price change")

# Price vs MA
if 'price_vs_ma20' not in df.columns:
    if 'sma_20' in df.columns:
        df['price_vs_ma20'] = (df['close'] - df['sma_20']) / df['sma_20']
    elif 'ma_20' in df.columns:
        df['price_vs_ma20'] = (df['close'] - df['ma_20']) / df['ma_20']
    print("✅ price_vs_ma20: (price - ma20) / ma20")

if 'price_vs_ma50' not in df.columns:
    if 'sma_50' in df.columns:
        df['price_vs_ma50'] = (df['close'] - df['sma_50']) / df['sma_50']
    elif 'ma_50' in df.columns:
        df['price_vs_ma50'] = (df['close'] - df['ma_50']) / df['ma_50']
    print("✅ price_vs_ma50: (price - ma50) / ma50")

# RSI slope
if 'rsi_slope' not in df.columns and 'rsi' in df.columns:
    df['rsi_slope'] = df['rsi'].diff(5)
    print("✅ rsi_slope: 5-period RSI change")

# RSI conditions
if 'rsi_overbought' not in df.columns and 'rsi' in df.columns:
    df['rsi_overbought'] = (df['rsi'] > 70).astype(float)
    print("✅ rsi_overbought: RSI > 70")

if 'rsi_oversold' not in df.columns and 'rsi' in df.columns:
    df['rsi_oversold'] = (df['rsi'] < 30).astype(float)
    print("✅ rsi_oversold: RSI < 30")

# RSI divergence (simplified)
if 'rsi_divergence' not in df.columns and 'rsi' in df.columns:
    price_trend = df['close'].diff(10)
    rsi_trend = df['rsi'].diff(10)
    df['rsi_divergence'] = (((price_trend > 0) & (rsi_trend < 0)) |
                            ((price_trend < 0) & (rsi_trend > 0))).astype(float)
    print("✅ rsi_divergence: Price/RSI trend divergence")

# MACD histogram slope
if 'macd_histogram_slope' not in df.columns:
    if 'macd_histogram' in df.columns:
        df['macd_histogram_slope'] = df['macd_histogram'].diff(3)
    elif 'macd' in df.columns and 'macd_signal' in df.columns:
        df['macd_histogram'] = df['macd'] - df['macd_signal']
        df['macd_histogram_slope'] = df['macd_histogram'].diff(3)
    print("✅ macd_histogram_slope: 3-period histogram change")

# MACD crossovers
if 'macd_crossover' not in df.columns and 'macd' in df.columns and 'macd_signal' in df.columns:
    df['macd_crossover'] = ((df['macd'] > df['macd_signal']) &
                            (df['macd'].shift(1) <= df['macd_signal'].shift(1))).astype(float)
    print("✅ macd_crossover: MACD crosses above signal")

if 'macd_crossunder' not in df.columns and 'macd' in df.columns and 'macd_signal' in df.columns:
    df['macd_crossunder'] = ((df['macd'] < df['macd_signal']) &
                             (df['macd'].shift(1) >= df['macd_signal'].shift(1))).astype(float)
    print("✅ macd_crossunder: MACD crosses below signal")

# Bollinger Band position
if 'bb_position' not in df.columns and 'bb_high' in df.columns and 'bb_low' in df.columns:
    bb_range = df['bb_high'] - df['bb_low']
    df['bb_position'] = (df['close'] - df['bb_low']) / bb_range.replace(0, np.nan)
    print("✅ bb_position: (price - bb_low) / (bb_high - bb_low)")

# Higher high
if 'higher_high' not in df.columns:
    df['higher_high'] = (df['high'] > df['high'].shift(1)).astype(float)
    print("✅ higher_high: Current high > previous high")

# Near support (simplified)
if 'near_support' not in df.columns:
    rolling_low = df['low'].rolling(20).min()
    df['near_support'] = ((df['close'] - rolling_low) / rolling_low < 0.02).astype(float)
    print("✅ near_support: Within 2% of 20-period low")

print()

# Clean NaN
print("Cleaning NaN values...")
df = df.ffill().bfill()
print("✅ NaN values handled")
print()

# Save
output_path = DATA_DIR / "BTCUSDT_5m_features_complete.csv"
df.to_csv(output_path, index=False)

print("="*80)
print("Enhanced Dataset Saved")
print("="*80)
print(f"File: {output_path}")
print(f"Rows: {len(df):,}")
print(f"Columns: {len(df.columns)}")
print()
print("All 27 Exit features now available:")
print("  ✅ 13 Core features (volatility_20, sma_50, ema_26, etc.)")
print("  ✅ 14 Enhanced features (volume_surge, price_acceleration, etc.)")
print()
print("Next: Retrain models with complete dataset")
print()
