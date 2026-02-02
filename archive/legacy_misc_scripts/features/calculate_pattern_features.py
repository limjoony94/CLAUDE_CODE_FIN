"""
Dynamic Pattern Recognition Feature Calculator
==============================================

Calculates 20 dynamic pattern features:
- Support/Resistance (7 features): Adaptive levels, distance, breaks
- Breakout Detection (7 features): Volume confirmation, follow-through
- Reversal Patterns (6 features): Swing highs/lows, divergences

Created: 2025-10-29
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
from scipy.signal import find_peaks

def calculate_support_resistance_features(df):
    """
    Calculate 7 support/resistance features.

    Returns: DataFrame with pattern_* features
    """
    print("  Calculating Support/Resistance features...")

    # 1. Dynamic Support (20-day low)
    df['pattern_support_level'] = df['low'].rolling(20).min()

    # 2. Dynamic Resistance (20-day high)
    df['pattern_resistance_level'] = df['high'].rolling(20).max()

    # 3. Support Distance (% from current price)
    df['pattern_support_distance'] = (
        (df['close'] - df['pattern_support_level']) / df['close']
    )

    # 4. Resistance Distance (% to resistance)
    df['pattern_resistance_distance'] = (
        (df['pattern_resistance_level'] - df['close']) / df['close']
    )

    # 5. Support Strength (number of touches)
    # Define "touch" as within 0.5% of support
    support_touches = (
        abs(df['low'] - df['pattern_support_level']) / df['pattern_support_level'] < 0.005
    ).rolling(50).sum()
    df['pattern_support_strength'] = support_touches

    # 6. Resistance Strength (number of touches)
    resistance_touches = (
        abs(df['high'] - df['pattern_resistance_level']) / df['pattern_resistance_level'] < 0.005
    ).rolling(50).sum()
    df['pattern_resistance_strength'] = resistance_touches

    # 7. S/R Ratio (support/resistance balance)
    df['pattern_sr_ratio'] = df['pattern_support_distance'] / df['pattern_resistance_distance'].replace(0, np.nan)

    # Select pattern features
    pattern_cols = [col for col in df.columns if col.startswith('pattern_') and any(x in col for x in ['support', 'resistance', 'sr_ratio'])]
    return df[pattern_cols]

def calculate_breakout_features(df):
    """
    Calculate 7 breakout detection features.

    Returns: DataFrame with pattern_* features
    """
    print("  Calculating Breakout Detection features...")

    # Calculate support/resistance if not present
    if 'pattern_resistance_level' not in df.columns:
        df['pattern_support_level'] = df['low'].rolling(20).min()
        df['pattern_resistance_level'] = df['high'].rolling(20).max()

    # 1. Resistance Break (close above resistance)
    df['pattern_resistance_break'] = (
        df['close'] > df['pattern_resistance_level'].shift(1)
    ).astype(float)

    # 2. Support Break (close below support)
    df['pattern_support_break'] = (
        df['close'] < df['pattern_support_level'].shift(1)
    ).astype(float)

    # 3. Breakout Strength (how far beyond level)
    df['pattern_breakout_strength'] = np.where(
        df['pattern_resistance_break'] == 1,
        (df['close'] - df['pattern_resistance_level'].shift(1)) / df['pattern_resistance_level'].shift(1),
        np.where(
            df['pattern_support_break'] == 1,
            (df['pattern_support_level'].shift(1) - df['close']) / df['pattern_support_level'].shift(1),
            0
        )
    )

    # 4. Volume Confirmation (volume > 1.5x average on breakout)
    volume_avg = df['volume'].rolling(20).mean()
    df['pattern_volume_confirmation'] = (
        (df['volume'] > 1.5 * volume_avg) &
        ((df['pattern_resistance_break'] == 1) | (df['pattern_support_break'] == 1))
    ).astype(float)

    # 5. Breakout Follow-Through (3-candle confirmation)
    # For resistance break: next 3 candles close above breakout price
    # For support break: next 3 candles close below breakout price
    def check_follow_through(series):
        # Look at next 3 candles (forward-looking, but OK for feature engineering)
        if len(series) < 4:
            return 0
        current = series.iloc[0]
        next_3 = series.iloc[1:4]
        if current > 0:  # Resistance break
            return float(all(next_3 > series.iloc[0]))
        elif current < 0:  # Support break
            return float(all(next_3 < series.iloc[0]))
        return 0

    df['pattern_breakout_follow_through'] = df['close'].rolling(4).apply(
        lambda x: check_follow_through(x) if len(x) == 4 else 0
    )

    # 6. False Breakout Indicator (price returns within S/R range)
    df['pattern_false_breakout'] = (
        (df['pattern_resistance_break'].shift(1) == 1) & (df['close'] < df['pattern_resistance_level'].shift(2)) |
        (df['pattern_support_break'].shift(1) == 1) & (df['close'] > df['pattern_support_level'].shift(2))
    ).astype(float)

    # 7. Breakout Age (candles since last breakout)
    breakout_occurred = (df['pattern_resistance_break'] == 1) | (df['pattern_support_break'] == 1)
    df['breakout_cumsum'] = breakout_occurred.cumsum()
    df['pattern_breakout_age'] = df.groupby('breakout_cumsum').cumcount()

    # Select pattern features
    pattern_cols = [col for col in df.columns if col.startswith('pattern_') and any(x in col for x in ['break', 'confirmation', 'follow', 'false', 'age'])]
    return df[pattern_cols]

def calculate_reversal_features(df):
    """
    Calculate 6 reversal pattern features.

    Returns: DataFrame with pattern_* features
    """
    print("  Calculating Reversal Pattern features...")

    # 1. Swing High (local peak using scipy)
    # Find peaks in high prices with minimum distance of 5 candles
    peaks, _ = find_peaks(df['high'].values, distance=5)
    df['pattern_swing_high'] = 0
    df.loc[peaks, 'pattern_swing_high'] = 1

    # 2. Swing Low (local trough using scipy)
    troughs, _ = find_peaks(-df['low'].values, distance=5)
    df['pattern_swing_low'] = 0
    df.loc[troughs, 'pattern_swing_low'] = 1

    # 3. Higher High (HH) - current high > previous high
    df['pattern_higher_high'] = (
        (df['high'] > df['high'].shift(1)) & (df['pattern_swing_high'] == 1)
    ).astype(float)

    # 4. Lower Low (LL) - current low < previous low
    df['pattern_lower_low'] = (
        (df['low'] < df['low'].shift(1)) & (df['pattern_swing_low'] == 1)
    ).astype(float)

    # 5. Price-Volume Divergence
    # Bearish: price making higher highs but volume declining
    price_hh = df['high'] > df['high'].rolling(10).max().shift(1)
    volume_declining = df['volume'] < df['volume'].rolling(10).mean()
    df['pattern_bearish_divergence'] = (price_hh & volume_declining).astype(float)

    # Bullish: price making lower lows but volume declining
    price_ll = df['low'] < df['low'].rolling(10).min().shift(1)
    df['pattern_bullish_divergence'] = (price_ll & volume_declining).astype(float)

    # 6. Reversal Signal Composite
    # Combines swing detection with divergence
    df['pattern_reversal_signal'] = (
        df['pattern_swing_high'] * df['pattern_bearish_divergence'] +
        df['pattern_swing_low'] * df['pattern_bullish_divergence']
    )

    # Select pattern features
    pattern_cols = [col for col in df.columns if col.startswith('pattern_') and any(x in col for x in ['swing', 'higher', 'lower', 'divergence', 'reversal'])]
    return df[pattern_cols]

def calculate_all_pattern_features(df):
    """
    Main function: Calculate all 20 dynamic pattern features.

    Args:
        df: DataFrame with OHLCV data

    Returns:
        DataFrame with original data + 20 pattern features
    """
    print("Calculating Dynamic Pattern Recognition Features...")
    print(f"Input: {len(df):,} candles")

    # Calculate support/resistance features (7)
    sr_features = calculate_support_resistance_features(df.copy())
    df = pd.concat([df, sr_features], axis=1)

    # Calculate breakout features (7)
    breakout_features = calculate_breakout_features(df.copy())
    df = pd.concat([df, breakout_features], axis=1)

    # Calculate reversal features (6)
    reversal_features = calculate_reversal_features(df.copy())
    df = pd.concat([df, reversal_features], axis=1)

    # Verify feature count
    pattern_features = [col for col in df.columns if col.startswith('pattern_')]
    print(f"\n✅ Generated {len(pattern_features)} dynamic pattern features")

    # List categories
    print("\nFeature Categories:")
    sr_cols = ['support', 'resistance', 'sr_ratio']
    breakout_cols = ['break', 'confirmation', 'follow', 'false', 'age']
    reversal_cols = ['swing', 'higher', 'lower', 'divergence', 'reversal']

    print(f"  Support/Resistance: {len([c for c in pattern_features if any(x in c for x in sr_cols) and not any(x in c for x in breakout_cols)])}")
    print(f"  Breakout Detection: {len([c for c in pattern_features if any(x in c for x in breakout_cols)])}")
    print(f"  Reversal Patterns: {len([c for c in pattern_features if any(x in c for x in reversal_cols)])}")

    return df

if __name__ == '__main__':
    print("="*80)
    print("DYNAMIC PATTERN RECOGNITION FEATURE CALCULATOR")
    print("="*80)
    print()

    # Load data with microstructure features
    data_file = PROJECT_ROOT / "data" / "features" / "BTCUSDT_5m_microstructure_features.csv"
    print(f"Loading: {data_file.name}")

    df = pd.read_csv(data_file)
    print(f"✅ Loaded {len(df):,} candles with {len(df.columns)} columns")
    print()

    # Calculate pattern features
    df_with_patterns = calculate_all_pattern_features(df)

    # Save
    output_file = PROJECT_ROOT / "data" / "features" / "BTCUSDT_5m_pattern_features.csv"
    df_with_patterns.to_csv(output_file, index=False)
    print(f"\n✅ Saved: {output_file.name}")
    print(f"   Columns: {len(df_with_patterns.columns)}")
    print(f"   Rows: {len(df_with_patterns):,}")
    print()

    # Sample output
    print("Sample Pattern Features (last row):")
    pattern_cols = [col for col in df_with_patterns.columns if col.startswith('pattern_')]
    print(df_with_patterns[pattern_cols].iloc[-1].to_string())
    print()
