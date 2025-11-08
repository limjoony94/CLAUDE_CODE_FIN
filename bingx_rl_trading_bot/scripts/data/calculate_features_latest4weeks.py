"""
Calculate Features for Latest 4-Week Data
==========================================
User Request: "최신 4주로 새로 받아서 백테스트를 진행해 주세요"
Purpose: Calculate technical indicators for fresh 4-week data before backtest
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

print("=" * 80)
print("CALCULATING FEATURES FOR LATEST 4-WEEK DATA")
print("=" * 80)
print()

# Find latest raw data file
raw_files = list(DATA_DIR.glob("BTCUSDT_5m_raw_latest4weeks_*.csv"))
if not raw_files:
    print("❌ No raw data file found!")
    sys.exit(1)

latest_raw = sorted(raw_files)[-1]
print(f"Loading: {latest_raw.name}")
print()

# Load raw data
df = pd.read_csv(latest_raw)
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Convert OHLCV to float
for col in ['open', 'high', 'low', 'close', 'volume']:
    df[col] = df[col].astype(float)

print(f"✓ Loaded {len(df):,} candles")
print(f"✓ Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
print()

print("Calculating technical indicators...")

# ================================
# MOVING AVERAGES
# ================================
df['sma_20'] = df['close'].rolling(20).mean()
df['sma_50'] = df['close'].rolling(50).mean()
df['sma_200'] = df['close'].rolling(200).mean()
df['ema_12'] = df['close'].ewm(span=12).mean()
df['ema_26'] = df['close'].ewm(span=26).mean()

# ================================
# RSI
# ================================
delta = df['close'].diff()
gain = delta.where(delta > 0, 0)
loss = -delta.where(delta < 0, 0)
avg_gain = gain.rolling(14).mean()
avg_loss = loss.rolling(14).mean()
rs = avg_gain / avg_loss
df['rsi'] = 100 - (100 / (1 + rs))

# ================================
# MACD
# ================================
df['macd'] = df['ema_12'] - df['ema_26']
df['macd_signal'] = df['macd'].ewm(span=9).mean()
df['macd_histogram'] = df['macd'] - df['macd_signal']

# ================================
# BOLLINGER BANDS
# ================================
df['bb_middle'] = df['close'].rolling(20).mean()
bb_std = df['close'].rolling(20).std()
df['bb_high'] = df['bb_middle'] + (bb_std * 2)
df['bb_low'] = df['bb_middle'] - (bb_std * 2)

# ================================
# VOLUME METRICS
# ================================
df['volume_sma'] = df['volume'].rolling(20).mean()
df['volume_ratio'] = df['volume'] / df['volume_sma']

# ================================
# PRICE CHANGES
# ================================
df['price_change'] = df['close'].pct_change()
df['price_change_5'] = df['close'].pct_change(5)
df['price_change_20'] = df['close'].pct_change(20)

# ================================
# ENHANCED EXIT FEATURES
# ================================
print("Calculating enhanced exit features...")

# Volume surge
df['volume_surge'] = (df['volume'] / df['volume'].rolling(20).mean() - 1).fillna(0)

# Price acceleration
df['price_acceleration'] = df['close'].diff(2).fillna(0)

# Price vs MA
df['price_vs_ma20'] = ((df['close'] - df['sma_20']) / df['sma_20']).fillna(0)
df['price_vs_ma50'] = ((df['close'] - df['sma_50']) / df['sma_50']).fillna(0)

# Volatility
df['volatility_20'] = df['close'].pct_change().rolling(20).std().fillna(0)

# RSI features
df['rsi_slope'] = df['rsi'].diff(5).fillna(0)
df['rsi_overbought'] = (df['rsi'] > 70).astype(int)
df['rsi_oversold'] = (df['rsi'] < 30).astype(int)
df['rsi_divergence'] = (df['rsi'].diff() * df['close'].pct_change() < 0).astype(int)

# MACD features
df['macd_histogram_slope'] = (df['macd'] - df['macd_signal']).diff(3).fillna(0)
df['macd_crossover'] = ((df['macd'] > df['macd_signal']) & (df['macd'].shift(1) <= df['macd_signal'].shift(1))).astype(int)
df['macd_crossunder'] = ((df['macd'] < df['macd_signal']) & (df['macd'].shift(1) >= df['macd_signal'].shift(1))).astype(int)

# Bollinger Bands features
bb_range = df['bb_high'] - df['bb_low']
bb_range = bb_range.replace(0, 1e-10)
df['bb_position'] = ((df['close'] - df['bb_low']) / bb_range).fillna(0.5)

# Price patterns
df['higher_high'] = (df['high'] > df['high'].shift(1)).astype(int)

# Support/Resistance
support_level = df['low'].rolling(20).min()
df['near_support'] = (df['close'] <= support_level * 1.01).astype(int)

print("✓ All features calculated")
print()

# Drop NaN rows (from indicators)
df_clean = df.dropna().reset_index(drop=True)

print(f"✓ Clean data: {len(df_clean):,} candles (dropped {len(df) - len(df_clean):,} NaN rows)")
print()

# Save features
timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = DATA_DIR / f"BTCUSDT_5m_features_latest4weeks_{timestamp_str}.csv"
df_clean.to_csv(output_file, index=False)

print(f"✓ Saved features: {output_file.name}")
print()

# Display summary
print("=" * 80)
print("FEATURE CALCULATION COMPLETE")
print("=" * 80)
print(f"Total Features: {len(df_clean.columns)}")
print(f"Clean Candles: {len(df_clean):,}")
print(f"Date Range: {df_clean['timestamp'].min()} to {df_clean['timestamp'].max()}")
print()
print(f"Next Step: Run backtest using {output_file.name}")
print("=" * 80)
