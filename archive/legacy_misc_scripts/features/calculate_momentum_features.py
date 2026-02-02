"""
Momentum Quality & Persistence Feature Calculator
================================================

Calculates 20 momentum features:
- Momentum Strength (7 features): ROC across multiple horizons
- Momentum Persistence (7 features): Consecutive moves, trending
- Momentum Exhaustion (6 features): RSI divergence, volume divergence

Created: 2025-10-29
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np

def calculate_momentum_strength_features(df):
    """
    Calculate 7 momentum strength features.

    Returns: DataFrame with momentum_* features
    """
    print("  Calculating Momentum Strength features...")

    # 1-3. Rate of Change across multiple horizons
    df['momentum_roc_1h'] = df['close'].pct_change(12)  # 1 hour
    df['momentum_roc_4h'] = df['close'].pct_change(48)  # 4 hours
    df['momentum_roc_24h'] = df['close'].pct_change(288)  # 24 hours

    # 4. Momentum Acceleration (second derivative)
    df['momentum_acceleration'] = df['momentum_roc_1h'].diff(6)

    # 5. Momentum Magnitude (average absolute change)
    df['returns'] = df['close'].pct_change()
    df['momentum_magnitude'] = df['returns'].abs().rolling(20).mean()

    # 6. Momentum Asymmetry (up vs down move strength)
    positive_returns = df['returns'].where(df['returns'] > 0, 0)
    negative_returns = df['returns'].where(df['returns'] < 0, 0).abs()
    df['momentum_asymmetry'] = (
        positive_returns.rolling(20).mean() /
        negative_returns.rolling(20).mean().replace(0, np.nan)
    ).fillna(1.0)

    # 7. Momentum Efficiency (net change / sum of absolute changes)
    def calculate_efficiency(prices):
        if len(prices) < 2:
            return 0.5
        net_change = abs(prices.iloc[-1] - prices.iloc[0])
        path_length = abs(prices.diff()).sum()
        return net_change / path_length if path_length > 0 else 0.5

    df['momentum_efficiency'] = df['close'].rolling(20).apply(calculate_efficiency)

    # Select momentum features
    momentum_cols = [col for col in df.columns if col.startswith('momentum_') and '_strength' not in col and '_persistence' not in col and '_exhaustion' not in col]
    return df[momentum_cols]

def calculate_momentum_persistence_features(df):
    """
    Calculate 7 momentum persistence features.

    Returns: DataFrame with momentum_* features
    """
    print("  Calculating Momentum Persistence features...")

    # Helper: consecutive counts
    def count_consecutive(series):
        """Count current streak of True values"""
        if len(series) == 0:
            return 0
        count = 0
        for val in reversed(series.values):
            if val:
                count += 1
            else:
                break
        return count

    # 1. Consecutive Up Moves
    df['up_move'] = df['close'].diff() > 0
    df['momentum_consecutive_up'] = df['up_move'].rolling(20).apply(count_consecutive)

    # 2. Consecutive Down Moves
    df['down_move'] = df['close'].diff() < 0
    df['momentum_consecutive_down'] = df['down_move'].rolling(20).apply(count_consecutive)

    # 3. Persistence Score (correlation of returns with lagged returns)
    df['returns'] = df['close'].pct_change()
    df['momentum_persistence_score'] = df['returns'].rolling(20).apply(
        lambda x: x.autocorr(lag=1) if len(x) > 1 else 0
    )

    # 4. Trend Days (days since last trend reversal)
    # Use moving average crossover as trend signal
    df['sma_10'] = df['close'].rolling(10).mean()
    df['sma_20'] = df['close'].rolling(20).mean()
    trend_change = (df['sma_10'] > df['sma_20']).astype(int).diff().abs()
    df['momentum_trend_days'] = trend_change.cumsum()
    df['momentum_trend_days'] = df.groupby('momentum_trend_days').cumcount()

    # 5. Higher Highs (count in last 20 candles)
    df['momentum_higher_highs'] = (
        df['high'] > df['high'].shift(1)
    ).rolling(20).sum()

    # 6. Lower Lows (count in last 20 candles)
    df['momentum_lower_lows'] = (
        df['low'] < df['low'].shift(1)
    ).rolling(20).sum()

    # 7. ZigZag Pattern (detect higher-high/higher-low pattern)
    # Simplified: uptrend if higher highs > lower lows
    df['momentum_zigzag_pattern'] = (
        df['momentum_higher_highs'] > df['momentum_lower_lows']
    ).astype(float)

    # Select momentum features
    momentum_cols = [col for col in df.columns if col.startswith('momentum_') and any(x in col for x in ['consecutive', 'persistence', 'trend_days', 'higher', 'lower', 'zigzag'])]
    return df[momentum_cols]

def calculate_momentum_exhaustion_features(df):
    """
    Calculate 6 momentum exhaustion features.

    Returns: DataFrame with momentum_* features
    """
    print("  Calculating Momentum Exhaustion features...")

    # Calculate RSI if not present
    if 'rsi' not in df.columns:
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = -delta.where(delta < 0, 0).rolling(14).mean()
        rs = gain / loss.replace(0, np.nan)
        df['rsi'] = 100 - (100 / (1 + rs))

    # 1. RSI Extreme
    df['momentum_rsi_extreme'] = (
        ((df['rsi'] > 70) | (df['rsi'] < 30))
    ).astype(float)

    # 2. RSI Divergence (price new high but RSI not new high)
    price_high = df['close'] == df['close'].rolling(20).max()
    rsi_high = df['rsi'] == df['rsi'].rolling(20).max()
    df['momentum_rsi_divergence'] = (price_high & ~rsi_high).astype(float)

    # 3. Volume Divergence (price new high but volume declining)
    volume_declining = df['volume'] < df['volume'].rolling(20).mean()
    df['momentum_volume_divergence'] = (price_high & volume_declining).astype(float)

    # 4. Parabolic Extension (distance from parabolic SAR approximation)
    # Simplified: distance from 20-period moving average as % of ATR
    df['sma_20'] = df['close'].rolling(20).mean()
    if 'atr_14' not in df.columns:
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = ranges.max(axis=1)
        df['atr_14'] = true_range.rolling(14).mean()

    df['momentum_parabolic_extension'] = (
        abs(df['close'] - df['sma_20']) / df['atr_14']
    )

    # 5. Climax Volume (volume > 3x average + price reversal)
    volume_climax = df['volume'] > 3 * df['volume'].rolling(20).mean()
    price_reversal = (
        (df['close'].diff() > 0) & (df['close'].diff().shift(-1) < 0)
    )
    df['momentum_climax_volume'] = (volume_climax & price_reversal).astype(float)

    # 6. Exhaustion Score (composite of above signals)
    df['momentum_exhaustion_score'] = (
        df['momentum_rsi_extreme'] +
        df['momentum_rsi_divergence'] +
        df['momentum_volume_divergence'] +
        (df['momentum_parabolic_extension'] > 2).astype(float) +
        df['momentum_climax_volume']
    ) / 5  # Normalize to 0-1

    # Select momentum features
    momentum_cols = [col for col in df.columns if col.startswith('momentum_') and any(x in col for x in ['rsi', 'volume_divergence', 'parabolic', 'climax', 'exhaustion'])]
    return df[momentum_cols]

def calculate_all_momentum_features(df):
    """
    Main function: Calculate all 20 momentum quality features.

    Args:
        df: DataFrame with OHLCV data

    Returns:
        DataFrame with original data + 20 momentum features
    """
    print("Calculating Momentum Quality Features...")
    print(f"Input: {len(df):,} candles")

    # Calculate momentum strength features (7)
    strength_features = calculate_momentum_strength_features(df.copy())
    df = pd.concat([df, strength_features], axis=1)

    # Calculate momentum persistence features (7)
    persistence_features = calculate_momentum_persistence_features(df.copy())
    df = pd.concat([df, persistence_features], axis=1)

    # Calculate momentum exhaustion features (6)
    exhaustion_features = calculate_momentum_exhaustion_features(df.copy())
    df = pd.concat([df, exhaustion_features], axis=1)

    # Verify feature count
    momentum_features = [col for col in df.columns if col.startswith('momentum_')]
    print(f"\n✅ Generated {len(momentum_features)} momentum quality features")

    # List categories
    print("\nFeature Categories:")
    strength_cols = ['roc_1h', 'roc_4h', 'roc_24h', 'acceleration', 'magnitude', 'asymmetry', 'efficiency']
    persistence_cols = ['consecutive', 'persistence_score', 'trend_days', 'higher', 'lower', 'zigzag']
    exhaustion_cols = ['rsi', 'divergence', 'parabolic', 'climax', 'exhaustion']

    print(f"  Strength: {len([c for c in momentum_features if any(x in c for x in strength_cols)])}")
    print(f"  Persistence: {len([c for c in momentum_features if any(x in c for x in persistence_cols)])}")
    print(f"  Exhaustion: {len([c for c in momentum_features if any(x in c for x in exhaustion_cols)])}")

    return df

if __name__ == '__main__':
    print("="*80)
    print("MOMENTUM QUALITY & PERSISTENCE FEATURE CALCULATOR")
    print("="*80)
    print()

    # Load data with regime features
    data_file = PROJECT_ROOT / "data" / "features" / "BTCUSDT_5m_regime_features.csv"
    print(f"Loading: {data_file.name}")

    df = pd.read_csv(data_file)
    print(f"✅ Loaded {len(df):,} candles with {len(df.columns)} columns")
    print()

    # Calculate momentum features
    df_with_momentum = calculate_all_momentum_features(df)

    # Save
    output_file = PROJECT_ROOT / "data" / "features" / "BTCUSDT_5m_momentum_features.csv"
    df_with_momentum.to_csv(output_file, index=False)
    print(f"\n✅ Saved: {output_file.name}")
    print(f"   Columns: {len(df_with_momentum.columns)}")
    print(f"   Rows: {len(df_with_momentum):,}")
    print()

    # Sample output
    print("Sample Momentum Features (last row):")
    momentum_cols = [col for col in df_with_momentum.columns if col.startswith('momentum_')]
    print(df_with_momentum[momentum_cols].iloc[-1].to_string())
    print()
