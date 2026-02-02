"""
Analyze Validation Period Market Conditions

Investigates why both Phase 2 Enhanced and Current Production models
failed catastrophically on Sep 28 - Oct 26, 2025 validation period.

Compares:
- Training period: Jul 14 - Sep 28 (21,940 candles)
- Validation period: Sep 28 - Oct 26 (8,064 candles)
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Paths
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data" / "features"

print("=" * 80)
print("VALIDATION PERIOD MARKET CONDITIONS ANALYSIS")
print("=" * 80)
print()

# Load complete dataset
df = pd.read_csv(DATA_DIR / "BTCUSDT_5m_features_complete.csv")
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Split into training and validation
validation_candles = 28 * 24 * 60 // 5  # 8,064 candles
train_end_idx = len(df) - validation_candles

df_train = df.iloc[:train_end_idx].copy()
df_val = df.iloc[train_end_idx:].copy()

print("Data Splits:")
print(f"  Training: {len(df_train):,} candles ({df_train['timestamp'].min()} to {df_train['timestamp'].max()})")
print(f"  Validation: {len(df_val):,} candles ({df_val['timestamp'].min()} to {df_val['timestamp'].max()})")
print()

# Calculate key statistics
def analyze_period(df, label):
    print("-" * 80)
    print(f"{label}")
    print("-" * 80)

    # Price statistics
    print("\nPrice Behavior:")
    print(f"  Start Price: ${df['close'].iloc[0]:,.2f}")
    print(f"  End Price: ${df['close'].iloc[-1]:,.2f}")
    print(f"  Price Change: {(df['close'].iloc[-1] / df['close'].iloc[0] - 1) * 100:+.2f}%")
    print(f"  Min Price: ${df['close'].min():,.2f}")
    print(f"  Max Price: ${df['close'].max():,.2f}")
    print(f"  Price Range: {(df['close'].max() / df['close'].min() - 1) * 100:.2f}%")

    # Volatility
    print("\nVolatility:")
    print(f"  Std Dev: ${df['close'].std():,.2f}")
    print(f"  % Std Dev: {df['close'].std() / df['close'].mean() * 100:.2f}%")
    print(f"  ATR Mean: ${df['atr'].mean():.2f}")
    print(f"  ATR Max: ${df['atr'].max():.2f}")

    # Trend indicators
    print("\nTrend Indicators:")
    print(f"  RSI Mean: {df['rsi'].mean():.2f}")
    print(f"  RSI > 70 (Overbought): {(df['rsi'] > 70).sum()} candles ({(df['rsi'] > 70).sum() / len(df) * 100:.1f}%)")
    print(f"  RSI < 30 (Oversold): {(df['rsi'] < 30).sum()} candles ({(df['rsi'] < 30).sum() / len(df) * 100:.1f}%)")

    # MACD
    macd_positive = (df['macd'] > df['macd_signal']).sum()
    print(f"  MACD > Signal: {macd_positive} candles ({macd_positive / len(df) * 100:.1f}%)")

    # Moving averages
    if 'sma_20' in df.columns and 'sma_50' in df.columns:
        price_above_ma20 = (df['close'] > df['sma_20']).sum()
        price_above_ma50 = (df['close'] > df['sma_50']).sum()
        print(f"  Price > SMA20: {price_above_ma20} candles ({price_above_ma20 / len(df) * 100:.1f}%)")
        print(f"  Price > SMA50: {price_above_ma50} candles ({price_above_ma50 / len(df) * 100:.1f}%)")

    # Volume
    print("\nVolume:")
    print(f"  Volume Mean: {df['volume'].mean():,.0f}")
    print(f"  Volume Std: {df['volume'].std():,.0f}")
    print(f"  High Volume (>1.5x mean): {(df['volume'] > df['volume'].mean() * 1.5).sum()} candles")

    # Candle patterns
    print("\nCandle Patterns:")
    green_candles = (df['close'] > df['open']).sum()
    red_candles = (df['close'] < df['open']).sum()
    print(f"  Green Candles: {green_candles} ({green_candles / len(df) * 100:.1f}%)")
    print(f"  Red Candles: {red_candles} ({red_candles / len(df) * 100:.1f}%)")

    # Large moves
    df['pct_change'] = df['close'].pct_change() * 100
    large_up = (df['pct_change'] > 1).sum()
    large_down = (df['pct_change'] < -1).sum()
    print(f"  Large Up Moves (>1%): {large_up} ({large_up / len(df) * 100:.2f}%)")
    print(f"  Large Down Moves (<-1%): {large_down} ({large_down / len(df) * 100:.2f}%)")

    print()

# Analyze both periods
analyze_period(df_train, "TRAINING PERIOD (Jul 14 - Sep 28)")
analyze_period(df_val, "VALIDATION PERIOD (Sep 28 - Oct 26)")

# Direct comparison
print("=" * 80)
print("KEY DIFFERENCES (Validation vs Training)")
print("=" * 80)
print()

metrics = {
    'Price Change %': (df_val['close'].iloc[-1] / df_val['close'].iloc[0] - 1) * 100 - (df_train['close'].iloc[-1] / df_train['close'].iloc[0] - 1) * 100,
    'Volatility (Std Dev)': df_val['close'].std() - df_train['close'].std(),
    'RSI Mean': df_val['rsi'].mean() - df_train['rsi'].mean(),
    'ATR Mean': df_val['atr'].mean() - df_train['atr'].mean(),
    'Volume Mean': df_val['volume'].mean() - df_train['volume'].mean(),
    'Green Candles %': (df_val['close'] > df_val['open']).sum() / len(df_val) * 100 - (df_train['close'] > df_train['open']).sum() / len(df_train) * 100,
}

for metric, diff in metrics.items():
    print(f"{metric:25s}: {diff:+.2f}")

print()
print("=" * 80)
print("CONCLUSION")
print("=" * 80)
print()

# Determine if validation period is anomalous
vol_ratio = df_val['close'].std() / df_train['close'].std()
rsi_diff = abs(df_val['rsi'].mean() - df_train['rsi'].mean())
atr_ratio = df_val['atr'].mean() / df_train['atr'].mean()

if vol_ratio > 1.5 or rsi_diff > 10 or atr_ratio > 1.5:
    print("⚠️  VALIDATION PERIOD IS SIGNIFICANTLY DIFFERENT FROM TRAINING")
    print(f"   - Volatility Ratio: {vol_ratio:.2f}x")
    print(f"   - RSI Difference: {rsi_diff:.2f} points")
    print(f"   - ATR Ratio: {atr_ratio:.2f}x")
    print()
    print("This explains why both model sets failed badly:")
    print("  1. Models trained on one market regime")
    print("  2. Validation period is different market regime")
    print("  3. Walk-Forward Decoupled prevented overfitting to training data")
    print("  4. But models still can't generalize to extreme conditions")
    print()
    print("Recommendation:")
    print("  - DO NOT DEPLOY either model set")
    print("  - Need more robust training that handles regime changes")
    print("  - Consider ensemble or regime-detection models")
else:
    print("✅ VALIDATION PERIOD IS SIMILAR TO TRAINING")
    print(f"   - Volatility Ratio: {vol_ratio:.2f}x")
    print(f"   - RSI Difference: {rsi_diff:.2f} points")
    print(f"   - ATR Ratio: {atr_ratio:.2f}x")
    print()
    print("This suggests overfitting despite Walk-Forward Decoupled:")
    print("  1. Models memorizing training patterns")
    print("  2. Not learning generalizable features")
    print("  3. Need feature engineering or different approach")
