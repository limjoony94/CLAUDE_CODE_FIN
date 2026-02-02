"""
Market Regime Detection Feature Calculator
==========================================

Calculates 25 regime detection features:
- Trend Regime (8 features): ADX, trend strength, exhaustion
- Volatility Regime (8 features): ATR percentile, expansion/contraction
- Volume Regime (9 features): Buying/selling pressure, divergences

Created: 2025-10-29
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
from scipy import stats

def calculate_atr(df, period=14):
    """Calculate Average True Range"""
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift())
    low_close = abs(df['low'] - df['close'].shift())

    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = ranges.max(axis=1)

    atr = true_range.rolling(period).mean()
    return atr

def calculate_adx(df, period=14):
    """Calculate ADX, +DI, -DI"""
    # True Range
    df['tr'] = calculate_atr(df, 1)  # 1-period ATR is just TR

    # Directional Movement
    df['high_diff'] = df['high'] - df['high'].shift()
    df['low_diff'] = df['low'].shift() - df['low']

    df['plus_dm'] = np.where(
        (df['high_diff'] > df['low_diff']) & (df['high_diff'] > 0),
        df['high_diff'],
        0
    )
    df['minus_dm'] = np.where(
        (df['low_diff'] > df['high_diff']) & (df['low_diff'] > 0),
        df['low_diff'],
        0
    )

    # Smoothed values
    df['plus_dm_smooth'] = df['plus_dm'].rolling(period).sum()
    df['minus_dm_smooth'] = df['minus_dm'].rolling(period).sum()
    df['tr_smooth'] = df['tr'].rolling(period).sum()

    # Directional Indicators
    df['plus_di'] = 100 * df['plus_dm_smooth'] / df['tr_smooth']
    df['minus_di'] = 100 * df['minus_dm_smooth'] / df['tr_smooth']

    # DX and ADX
    df['dx'] = 100 * abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di'])
    df['adx'] = df['dx'].rolling(period).mean()

    return df['adx'], df['plus_di'], df['minus_di']

def calculate_trend_regime_features(df):
    """
    Calculate 8 trend regime features.

    Returns: DataFrame with regime_trend_* features
    """
    print("  Calculating Trend Regime features...")

    # 1. Trend Strength (ADX)
    df['adx'], df['plus_di'], df['minus_di'] = calculate_adx(df, 14)
    df['regime_trend_strength'] = df['adx']

    # 2. Trend Direction (+DI vs -DI)
    df['regime_trend_direction'] = np.where(
        df['plus_di'] > df['minus_di'], 1, -1
    )

    # 3. Trend Slope (slope of 50-period SMA over 20 periods)
    df['sma_50'] = df['close'].rolling(50).mean()
    df['regime_trend_slope'] = df['sma_50'].diff(20) / 20

    # 4. Trend Consistency (% of last 50 candles above/below SMA)
    df['regime_trend_consistency'] = (df['close'] > df['sma_50']).rolling(50).mean()

    # 5. Trend Age (candles since last trend reversal)
    trend_signal = (df['sma_50'].diff() > 0).astype(int).diff()
    df['regime_trend_age'] = (trend_signal != 0).cumsum()
    df['regime_trend_age'] = df.groupby('regime_trend_age').cumcount()

    # 6. Trend Exhaustion (RSI extreme + volume decline)
    if 'rsi' in df.columns:
        rsi_extreme = ((df['rsi'] > 70) | (df['rsi'] < 30)).astype(float)
    else:
        # Calculate simple RSI if not present
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = -delta.where(delta < 0, 0).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        rsi = 100 - (100 / (1 + rs))
        rsi_extreme = ((rsi > 70) | (rsi < 30)).astype(float)

    volume_decline = (df['volume'] < df['volume'].rolling(20).mean()).astype(float)
    df['regime_trend_exhaustion'] = rsi_extreme * volume_decline

    # 7. Channel Position (Donchian Channel)
    df['donchian_high'] = df['high'].rolling(20).max()
    df['donchian_low'] = df['low'].rolling(20).min()
    df['regime_channel_position'] = (
        (df['close'] - df['donchian_low']) /
        (df['donchian_high'] - df['donchian_low'])
    ).fillna(0.5)

    # 8. Breakout Potential (Bollinger Band squeeze + low ATR)
    if 'bb_upper' in df.columns and 'bb_lower' in df.columns:
        bb_width = (df['bb_upper'] - df['bb_lower']) / df['close']
    else:
        # Calculate Bollinger Bands
        bb_ma = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        bb_width = (2 * bb_std) / df['close']

    df['atr_14'] = calculate_atr(df, 14)
    atr_percentile = df['atr_14'].rolling(100).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else 0.5
    )

    df['regime_breakout_potential'] = (
        (bb_width < bb_width.rolling(50).quantile(0.2)) &
        (atr_percentile < 0.3)
    ).astype(float)

    # Select regime features
    regime_cols = [col for col in df.columns if col.startswith('regime_trend_')]
    return df[regime_cols]

def calculate_volatility_regime_features(df):
    """
    Calculate 8 volatility regime features.

    Returns: DataFrame with regime_volatility_* features
    """
    print("  Calculating Volatility Regime features...")

    # Calculate ATR if not present
    if 'atr_14' not in df.columns:
        df['atr_14'] = calculate_atr(df, 14)

    # 1. Volatility Level (ATR percentile)
    df['regime_volatility_level'] = df['atr_14'].rolling(100).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else 0.5
    ) * 100  # 0-100 scale

    # 2. Volatility Trend (ATR diff over 10 periods)
    df['regime_volatility_trend'] = df['atr_14'].diff(10)

    # 3. Volatility Skew (upside vs downside volatility)
    df['returns'] = df['close'].pct_change()
    upside_vol = df['returns'].where(df['returns'] > 0, 0).rolling(20).std()
    downside_vol = df['returns'].where(df['returns'] < 0, 0).rolling(20).std()
    df['regime_volatility_skew'] = upside_vol / downside_vol.replace(0, np.nan)

    # 4. GARCH Forecast (Simple GARCH(1,1) approximation)
    # Simplified: recent variance + persistence
    variance = df['returns'].rolling(20).var()
    df['regime_garch_forecast'] = 0.1 * variance + 0.9 * variance.shift()

    # 5. Realized vs Implied (recent ATR vs historical average)
    df['regime_realized_vs_implied'] = df['atr_14'] / df['atr_14'].rolling(100).mean()

    # 6. Intraday Range (high-low / open)
    df['regime_intraday_range'] = (df['high'] - df['low']) / df['open']

    # 7. Overnight Gap (open - prev_close)
    df['regime_overnight_gap'] = df['open'] - df['close'].shift()

    # 8. Volatility Regime Change (5m vol > 2x 4h vol)
    # Use existing multi-timeframe features if available
    if 'mtf_5m_volatility' in df.columns and 'mtf_4h_volatility' in df.columns:
        df['regime_volatility_regime_change'] = (
            df['mtf_5m_volatility'] > 2 * df['mtf_4h_volatility']
        ).astype(float)
    else:
        # Fallback: recent vs long-term
        df['regime_volatility_regime_change'] = (
            df['atr_14'] > 2 * df['atr_14'].rolling(96).mean()
        ).astype(float)

    # Select regime features
    regime_cols = [col for col in df.columns if col.startswith('regime_volatility_')]
    return df[regime_cols]

def calculate_volume_regime_features(df):
    """
    Calculate 9 volume regime features.

    Returns: DataFrame with regime_volume_* features
    """
    print("  Calculating Volume Regime features...")

    # 1. Volume Level (percentile)
    df['regime_volume_level'] = df['volume'].rolling(100).apply(
        lambda x: pd.Series(x).rank(pct=True).iloc[-1] if len(x) > 0 else 0.5
    ) * 100  # 0-100 scale

    # 2. Volume Trend (volume MA diff over 10 periods)
    df['volume_ma_20'] = df['volume'].rolling(20).mean()
    df['regime_volume_trend'] = df['volume_ma_20'].diff(10)

    # 3. Volume Concentration (top 20% candles' volume / total)
    df['regime_volume_concentration'] = df['volume'].rolling(20).apply(
        lambda x: x.nlargest(int(len(x) * 0.2)).sum() / x.sum() if len(x) > 0 else 0.5
    )

    # 4. Volume-Price Correlation
    df['returns'] = df['close'].pct_change()
    df['regime_volume_price_correlation'] = df['volume'].rolling(20).corr(df['returns'].abs())

    # 5. Buying Pressure (proxy using close position in candle)
    df['close_position'] = (df['close'] - df['low']) / (df['high'] - df['low']).replace(0, np.nan)
    df['regime_buying_pressure'] = (df['close_position'] * df['volume']).rolling(20).mean()

    # 6. Selling Pressure
    df['regime_selling_pressure'] = ((1 - df['close_position']) * df['volume']).rolling(20).mean()

    # 7. Volume Divergence (price trend vs volume trend)
    price_trend = df['close'].diff(20)
    volume_trend = df['volume'].diff(20)
    df['regime_volume_divergence'] = np.where(
        (price_trend > 0) & (volume_trend < 0), -1,  # Bearish divergence
        np.where((price_trend < 0) & (volume_trend > 0), 1, 0)  # Bullish divergence
    )

    # 8. Large Trade Frequency (count of volume > 2x average)
    df['regime_large_trade_frequency'] = (
        df['volume'] > 2 * df['volume_ma_20']
    ).rolling(20).sum()

    # 9. Volume Profile Shape (kurtosis of volume distribution)
    df['regime_volume_profile_shape'] = df['volume'].rolling(50).apply(
        lambda x: stats.kurtosis(x, fisher=True, nan_policy='omit') if len(x) > 10 else 0
    )

    # Select regime features
    regime_cols = [col for col in df.columns if col.startswith('regime_volume_')]
    return df[regime_cols]

def calculate_all_regime_features(df):
    """
    Main function: Calculate all 25 market regime features.

    Args:
        df: DataFrame with OHLCV data

    Returns:
        DataFrame with original data + 25 regime features
    """
    print("Calculating Market Regime Features...")
    print(f"Input: {len(df):,} candles")

    # Calculate trend regime features (8)
    trend_features = calculate_trend_regime_features(df.copy())
    df = pd.concat([df, trend_features], axis=1)

    # Calculate volatility regime features (8)
    vol_features = calculate_volatility_regime_features(df.copy())
    df = pd.concat([df, vol_features], axis=1)

    # Calculate volume regime features (9)
    volume_features = calculate_volume_regime_features(df.copy())
    df = pd.concat([df, volume_features], axis=1)

    # Verify feature count
    regime_features = [col for col in df.columns if col.startswith('regime_')]
    print(f"\n✅ Generated {len(regime_features)} market regime features")

    # List categories
    print("\nFeature Categories:")
    print(f"  Trend regime: {len([c for c in regime_features if 'regime_trend_' in c])}")
    print(f"  Volatility regime: {len([c for c in regime_features if 'regime_volatility_' in c])}")
    print(f"  Volume regime: {len([c for c in regime_features if 'regime_volume_' in c])}")

    return df

if __name__ == '__main__':
    print("="*80)
    print("MARKET REGIME DETECTION FEATURE CALCULATOR")
    print("="*80)
    print()

    # Load data with multi-timeframe features
    data_file = PROJECT_ROOT / "data" / "features" / "BTCUSDT_5m_multitimeframe_features.csv"
    print(f"Loading: {data_file.name}")

    df = pd.read_csv(data_file)
    print(f"✅ Loaded {len(df):,} candles with {len(df.columns)} columns")
    print()

    # Calculate regime features
    df_with_regime = calculate_all_regime_features(df)

    # Save
    output_file = PROJECT_ROOT / "data" / "features" / "BTCUSDT_5m_regime_features.csv"
    df_with_regime.to_csv(output_file, index=False)
    print(f"\n✅ Saved: {output_file.name}")
    print(f"   Columns: {len(df_with_regime.columns)}")
    print(f"   Rows: {len(df_with_regime):,}")
    print()

    # Sample output
    print("Sample Regime Features (last row):")
    regime_cols = [col for col in df_with_regime.columns if col.startswith('regime_')]
    print(df_with_regime[regime_cols].iloc[-1].to_string())
    print()
