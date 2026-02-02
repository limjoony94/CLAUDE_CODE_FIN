"""
Multi-Timeframe Feature Calculator
===================================

Aggregates 5m candles to 15m, 1h, 4h timeframes and calculates indicators.
Returns 30 multi-timeframe features merged back to 5m base timeframe.

Created: 2025-10-29
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
from ta.trend import ADXIndicator
from ta.volatility import AverageTrueRange

def calculate_atr(df, period=14):
    """Calculate Average True Range"""
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift())
    low_close = abs(df['low'] - df['close'].shift())

    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)

    atr = true_range.rolling(period).mean()
    return atr

def aggregate_to_timeframe(df, timeframe='15min', prefix='mtf_15m'):
    """
    Resample 5m data to higher timeframe and calculate indicators.

    Args:
        df: DataFrame with 5m OHLCV data (with timestamp as index)
        timeframe: pandas resample string ('15min', '1h', '4h')
        prefix: prefix for feature names

    Returns:
        DataFrame with aggregated features (indexed by timestamp)
    """
    # Resample OHLCV (df already has timestamp as index)
    df_agg = df.resample(timeframe).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    })

    # Calculate indicators on aggregated timeframe

    # 1. Trend Strength: (close - sma_50) / atr_14
    df_agg['sma_20'] = df_agg['close'].rolling(20).mean()
    df_agg['sma_50'] = df_agg['close'].rolling(50).mean()
    df_agg['atr_14'] = calculate_atr(df_agg, 14)
    df_agg[f'{prefix}_trend_strength'] = (df_agg['close'] - df_agg['sma_50']) / df_agg['atr_14']

    # 2. Momentum: recent return
    if timeframe == '15min':
        lookback = 4  # 1 hour in 15m candles
    elif timeframe == '1h' or timeframe == '60min':
        lookback = 4  # 4 hours in 1h candles
    elif timeframe == '4h' or timeframe == '240min':
        lookback = 6  # 24 hours in 4h candles
    else:
        lookback = 12  # default 1 hour

    df_agg[f'{prefix}_momentum'] = df_agg['close'].pct_change(lookback)

    # 3. Volume Trend: (volume / ma_20) - 1
    df_agg['volume_ma_20'] = df_agg['volume'].rolling(20).mean()
    df_agg[f'{prefix}_volume_trend'] = (df_agg['volume'] / df_agg['volume_ma_20']) - 1

    # 4. Volatility: atr / close
    df_agg[f'{prefix}_volatility'] = df_agg['atr_14'] / df_agg['close']

    # 5. Price Acceleration: momentum.diff(3)
    df_agg[f'{prefix}_price_acceleration'] = df_agg[f'{prefix}_momentum'].diff(3)

    # 6. Trend Consistency: % time above sma_20 in last N candles
    df_agg[f'{prefix}_trend_consistency'] = (df_agg['close'] > df_agg['sma_20']).rolling(12).mean()

    # Additional features specific to timeframe

    if timeframe == '15min':
        # Breakout: distance from 4-hour high
        df_agg[f'{prefix}_breakout'] = (df_agg['high'] - df_agg['high'].rolling(16).max()) / df_agg['high'].rolling(16).max()

        # Trend alignment: close > sma_20 > sma_50
        df_agg[f'{prefix}_trend_alignment'] = (
            (df_agg['close'] > df_agg['sma_20']) &
            (df_agg['sma_20'] > df_agg['sma_50'])
        ).astype(float)

    elif timeframe == '1h' or timeframe == '60min':
        # Trend direction: 1 if sma_10 > sma_20, else -1
        df_agg['sma_10'] = df_agg['close'].rolling(10).mean()
        df_agg[f'{prefix}_trend_direction'] = np.where(
            df_agg['sma_10'] > df_agg['sma_20'], 1, -1
        )

        # Volume profile: volume percentile
        df_agg[f'{prefix}_volume_profile'] = df_agg['volume'].rolling(24).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else 0.5
        )

        # Volatility regime: atr percentile
        df_agg[f'{prefix}_volatility_regime'] = df_agg['atr_14'].rolling(24).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else 0.5
        )

        # Price position in range
        df_agg[f'{prefix}_price_position'] = (
            (df_agg['close'] - df_agg['low'].rolling(24).min()) /
            (df_agg['high'].rolling(24).max() - df_agg['low'].rolling(24).min())
        ).fillna(0.5)

    elif timeframe == '4h' or timeframe == '240min':
        # Support distance: % from 8-day low
        df_agg[f'{prefix}_support_distance'] = (
            (df_agg['close'] - df_agg['low'].rolling(48).min()) / df_agg['close']
        )

        # Resistance distance: % to 8-day high
        df_agg[f'{prefix}_resistance_distance'] = (
            (df_agg['high'].rolling(48).max() - df_agg['close']) / df_agg['close']
        )

        # Volume strength: recent vs 1-week average
        df_agg[f'{prefix}_volume_strength'] = (
            df_agg['volume'].rolling(48).mean() / df_agg['volume'].rolling(168).mean()
        )

        # Trend persistence: % time in uptrend over 2 days
        df_agg['sma_6'] = df_agg['close'].rolling(6).mean()
        df_agg[f'{prefix}_trend_persistence'] = (
            (df_agg['close'] > df_agg['sma_6']).rolling(12).sum() / 12
        )

    # Select only feature columns (not intermediate calculations)
    feature_cols = [col for col in df_agg.columns if col.startswith(prefix)]

    return df_agg[feature_cols]

def calculate_5m_features(df, prefix='mtf_5m'):
    """Calculate features directly on 5m timeframe"""

    # 1. Trend Strength
    df['sma_50'] = df['close'].rolling(50).mean()
    df['atr_14'] = calculate_atr(df, 14)
    df[f'{prefix}_trend_strength'] = (df['close'] - df['sma_50']) / df['atr_14']

    # 2. Momentum: 1-hour return (12 candles)
    df[f'{prefix}_momentum'] = df['close'].pct_change(12)

    # 3. Volume Trend
    df['volume_ma_20'] = df['volume'].rolling(20).mean()
    df[f'{prefix}_volume_trend'] = (df['volume'] / df['volume_ma_20']) - 1

    # 4. Volatility
    df[f'{prefix}_volatility'] = df['atr_14'] / df['close']

    # 5. Price Acceleration
    df[f'{prefix}_price_acceleration'] = df[f'{prefix}_momentum'].diff(3)

    # 6. Trend Consistency
    df['sma_20'] = df['close'].rolling(20).mean()
    df[f'{prefix}_trend_consistency'] = (df['close'] > df['sma_20']).rolling(12).mean()

    # Drop intermediate columns
    feature_cols = [col for col in df.columns if col.startswith(prefix)]
    return df[['timestamp'] + feature_cols]

def calculate_cross_timeframe_alignment(df):
    """Calculate alignment features across all timeframes"""

    # All trends aligned
    df['mtf_all_trends_aligned'] = (
        (df['mtf_5m_trend_strength'] > 0) &
        (df['mtf_15m_trend_strength'] > 0) &
        (df['mtf_1h_trend_strength'] > 0) &
        (df['mtf_4h_trend_strength'] > 0)
    ).astype(float)

    # Momentum divergence: short-term vs long-term
    df['mtf_momentum_divergence'] = df['mtf_5m_momentum'] - df['mtf_1h_momentum']

    # Volatility expansion: short-term vs long-term
    df['mtf_volatility_expansion'] = df['mtf_5m_volatility'] / df['mtf_4h_volatility']

    # Volume acceleration: short-term vs long-term
    df['mtf_volume_acceleration'] = df['mtf_5m_volume_trend'] - df['mtf_1h_volume_trend']

    # Trend reversal signal: short pullback in uptrend
    df['mtf_trend_reversal_signal'] = (
        (df['mtf_5m_trend_strength'] < 0) &
        (df['mtf_15m_trend_strength'] < 0) &
        (df['mtf_1h_trend_strength'] > 0)
    ).astype(float)

    # Breakout confluence: count of timeframes breaking highs
    df['mtf_breakout_confluence'] = (
        (df['mtf_5m_momentum'] > 0.02).astype(int) +
        (df['mtf_15m_momentum'] > 0.02).astype(int) +
        (df['mtf_1h_momentum'] > 0.02).astype(int) +
        (df['mtf_4h_momentum'] > 0.02).astype(int)
    )

    return df

def calculate_all_multitimeframe_features(df):
    """
    Main function: Calculate all 30 multi-timeframe features.

    Args:
        df: DataFrame with 5m OHLCV data (timestamp, open, high, low, close, volume)

    Returns:
        DataFrame with original data + 30 multi-timeframe features
    """
    print("Calculating Multi-Timeframe Features...")
    print(f"Input: {len(df):,} candles")

    # Ensure timestamp is datetime and set as index
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    # 1. 5-minute features (6 features) - calculate before setting index
    print("  - 5m features...")
    df_5m = calculate_5m_features(df.copy(), 'mtf_5m')
    df = df.merge(df_5m[['timestamp'] + [c for c in df_5m.columns if c.startswith('mtf_5m')]], on='timestamp')

    # Now set timestamp as index for resampling
    df = df.set_index('timestamp')

    # 2. 15-minute aggregation (8 features: 6 standard + 2 extra)
    print("  - 15m aggregation...")
    df_15m = aggregate_to_timeframe(df, '15min', 'mtf_15m')
    df = df.join(df_15m, how='left').ffill()

    # 3. 1-hour aggregation (10 features: 6 standard + 4 extra)
    print("  - 1h aggregation...")
    df_1h = aggregate_to_timeframe(df, '1h', 'mtf_1h')
    df = df.join(df_1h, how='left').ffill()

    # 4. 4-hour aggregation (10 features: 6 standard + 4 extra)
    print("  - 4h aggregation...")
    df_4h = aggregate_to_timeframe(df, '4h', 'mtf_4h')
    df = df.join(df_4h, how='left').ffill()

    # 5. Cross-timeframe alignment (6 features)
    print("  - Cross-timeframe alignment...")
    df = calculate_cross_timeframe_alignment(df)

    # Reset index
    df = df.reset_index()

    # Verify feature count
    mtf_features = [col for col in df.columns if col.startswith('mtf_')]
    print(f"\n✅ Generated {len(mtf_features)} multi-timeframe features")

    # List all features
    print("\nFeature Categories:")
    print(f"  5m features: {len([c for c in mtf_features if 'mtf_5m' in c])}")
    print(f"  15m features: {len([c for c in mtf_features if 'mtf_15m' in c])}")
    print(f"  1h features: {len([c for c in mtf_features if 'mtf_1h' in c])}")
    print(f"  4h features: {len([c for c in mtf_features if 'mtf_4h' in c])}")
    print(f"  Cross-TF features: {len([c for c in mtf_features if 'mtf_' in c and not any(x in c for x in ['5m', '15m', '1h', '4h'])])}")

    return df

if __name__ == '__main__':
    print("="*80)
    print("MULTI-TIMEFRAME FEATURE CALCULATOR")
    print("="*80)
    print()

    # Load data
    data_file = PROJECT_ROOT / "data" / "features" / "BTCUSDT_5m_features.csv"
    print(f"Loading: {data_file.name}")

    df = pd.read_csv(data_file)
    print(f"✅ Loaded {len(df):,} candles")
    print()

    # Calculate features
    df_with_mtf = calculate_all_multitimeframe_features(df)

    # Save
    output_file = PROJECT_ROOT / "data" / "features" / "BTCUSDT_5m_multitimeframe_features.csv"
    df_with_mtf.to_csv(output_file, index=False)
    print(f"\n✅ Saved: {output_file.name}")
    print(f"   Columns: {len(df_with_mtf.columns)}")
    print(f"   Rows: {len(df_with_mtf):,}")
    print()

    # Sample output
    print("Sample MTF Features (last row):")
    mtf_cols = [col for col in df_with_mtf.columns if col.startswith('mtf_')]
    print(df_with_mtf[mtf_cols].iloc[-1].to_string())
    print()
