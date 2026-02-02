"""
Add 15-minute Long-term Features to 5-minute Data

목표: Bull market detection 개선
- 5분 캔들 유지 (기본 timeframe)
- 15분 long-term features 추가 (EMA, Trend 등)
- Bull -4.45% → 0%+ 목표
"""

import pandas as pd
import numpy as np
import ta
from pathlib import Path


def add_15m_features(df_5m, df_15m):
    """
    Add 15-minute long-term features to 5-minute data

    Args:
        df_5m: 5-minute candlestick data
        df_15m: 15-minute candlestick data

    Returns:
        df_5m with 15m features added
    """
    print("Adding 15-minute long-term features...")

    # Calculate 15m long-term indicators
    df_15m = df_15m.copy()

    # Long-term EMAs
    df_15m['ema_50_15m'] = ta.trend.ema_indicator(df_15m['close'], window=50)
    df_15m['ema_200_15m'] = ta.trend.ema_indicator(df_15m['close'], window=200)

    # EMA crossover
    df_15m['ema_cross_15m'] = (df_15m['ema_50_15m'] > df_15m['ema_200_15m']).astype(int)

    # EMA distance (trend strength)
    df_15m['ema_dist_15m'] = ((df_15m['ema_50_15m'] - df_15m['ema_200_15m']) / df_15m['ema_200_15m']) * 100

    # Long-term RSI
    df_15m['rsi_15m'] = ta.momentum.rsi(df_15m['close'], window=14)

    # Long-term MACD
    macd_15m = ta.trend.MACD(df_15m['close'])
    df_15m['macd_15m'] = macd_15m.macd()
    df_15m['macd_signal_15m'] = macd_15m.macd_signal()
    df_15m['macd_diff_15m'] = macd_15m.macd_diff()

    # Trend strength (ADX)
    df_15m['adx_15m'] = ta.trend.adx(df_15m['high'], df_15m['low'], df_15m['close'], window=14)

    # Volatility
    df_15m['volatility_15m'] = df_15m['close'].pct_change().rolling(20).std()

    # Momentum
    df_15m['momentum_15m'] = df_15m['close'].pct_change(periods=10) * 100

    # Support/Resistance levels
    df_15m['support_15m'] = df_15m['low'].rolling(50).min()
    df_15m['resistance_15m'] = df_15m['high'].rolling(50).max()
    df_15m['price_position_15m'] = ((df_15m['close'] - df_15m['support_15m']) /
                                     (df_15m['resistance_15m'] - df_15m['support_15m']) * 100)

    # Select 15m features to merge
    features_15m = [
        'ema_50_15m', 'ema_200_15m', 'ema_cross_15m', 'ema_dist_15m',
        'rsi_15m', 'macd_15m', 'macd_signal_15m', 'macd_diff_15m',
        'adx_15m', 'volatility_15m', 'momentum_15m',
        'support_15m', 'resistance_15m', 'price_position_15m'
    ]

    df_15m_features = df_15m[['timestamp'] + features_15m].copy()

    # Convert timestamp to datetime for merging
    df_5m['timestamp'] = pd.to_datetime(df_5m['timestamp'])
    df_15m_features['timestamp'] = pd.to_datetime(df_15m_features['timestamp'])

    # Merge: for each 5m candle, use the corresponding 15m features
    # (15m timestamp is the start of each 15m period)
    df_5m['timestamp_15m'] = df_5m['timestamp'].dt.floor('15min')
    df_merged = df_5m.merge(
        df_15m_features,
        left_on='timestamp_15m',
        right_on='timestamp',
        how='left',
        suffixes=('', '_15m_dup')
    )

    # Drop duplicate timestamp column
    df_merged = df_merged.drop(['timestamp_15m', 'timestamp_15m_dup'], axis=1, errors='ignore')

    print(f"✅ Added {len(features_15m)} features from 15m data")
    print(f"   Total features: {len(df_merged.columns)}")

    return df_merged


if __name__ == "__main__":
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    DATA_DIR = PROJECT_ROOT / "data" / "historical"

    # Load data
    print("Loading data...")
    df_5m = pd.read_csv(DATA_DIR / "BTCUSDT_5m_max.csv")
    df_15m = pd.read_csv(DATA_DIR / "BTCUSDT_15m.csv")

    print(f"5m data: {len(df_5m)} candles")
    print(f"15m data: {len(df_15m)} candles")

    # Add 15m features
    df_enhanced = add_15m_features(df_5m, df_15m)

    # Check for NaN
    nan_count = df_enhanced.isna().sum().sum()
    print(f"\nNaN values: {nan_count}")

    if nan_count > 0:
        print("Dropping NaN rows...")
        df_enhanced = df_enhanced.dropna()
        print(f"Final data: {len(df_enhanced)} candles")

    # Save
    output_file = DATA_DIR / "BTCUSDT_5m_with_15m_features.csv"
    df_enhanced.to_csv(output_file, index=False)
    print(f"\n✅ Saved enhanced data: {output_file}")

    # Show sample
    print("\nSample of 15m features:")
    feature_cols = [c for c in df_enhanced.columns if '_15m' in c]
    print(df_enhanced[['timestamp', 'close'] + feature_cols[:5]].tail())
