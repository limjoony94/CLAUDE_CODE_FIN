"""
Reversal Detection Features for SHORT Exit Model
=================================================

Specialized features to detect price reversals (bounces) during SHORT positions:
1. Price momentum reversal detection
2. Volume spike on reversal
3. RSI divergence (real implementation)
4. Support level bounce detection
5. Consecutive green candles (SHORT danger)
6. Buy pressure indicators

These features help identify when SHORT position should exit before losing profits.

Author: Claude Code
Date: 2025-10-18
"""

import pandas as pd
import numpy as np


def calculate_momentum_reversal(df, lookback=5):
    """
    Detect momentum reversal: downtrend changing to uptrend

    Logic:
    - Calculate price momentum (rate of change)
    - Detect when negative momentum starts becoming less negative
    - Signal strength: how quickly momentum is reversing

    Returns:
        reversal_signal: 0-1, higher = stronger reversal
    """
    # Price momentum (% change over lookback)
    momentum = df['close'].pct_change(lookback)

    # Momentum acceleration (second derivative)
    momentum_accel = momentum.diff()

    # Reversal signal: momentum is negative but accelerating upward
    reversal = np.where(
        (momentum < 0) & (momentum_accel > 0),
        momentum_accel / momentum.abs().replace(0, 1),  # Normalized strength
        0
    )

    # Clip to 0-1 range
    reversal_signal = np.clip(reversal, 0, 1)

    return reversal_signal


def calculate_volume_spike_on_bounce(df, lookback=20, spike_threshold=1.5):
    """
    Detect volume spike during price bounce (reversal)

    Logic:
    - High volume during price increase = strong buying pressure
    - This often signals SHORT position should exit

    Returns:
        spike_signal: 0-1, higher = stronger spike
    """
    # Average volume
    avg_volume = df['volume'].rolling(lookback).mean()

    # Price direction (1 = up, 0 = down)
    price_up = (df['close'] > df['close'].shift(1)).astype(int)

    # Volume ratio
    volume_ratio = df['volume'] / avg_volume.replace(0, 1)

    # Spike signal: high volume + price up
    spike_signal = np.where(
        (price_up == 1) & (volume_ratio > spike_threshold),
        np.clip((volume_ratio - spike_threshold) / 2, 0, 1),  # Normalized
        0
    )

    return spike_signal


def calculate_rsi_divergence_real(df, lookback=10):
    """
    Real RSI divergence detection

    Bullish Divergence (SHORT exit signal):
    - Price makes lower low
    - RSI makes higher low
    - Indicates weakening downtrend

    Returns:
        divergence_signal: 0-1, higher = stronger divergence
    """
    divergence_signal = np.zeros(len(df))

    # Need at least lookback candles
    for i in range(lookback, len(df)):
        # Current values
        current_price = df['close'].iloc[i]
        current_rsi = df['rsi'].iloc[i]

        # Find previous low in lookback window
        lookback_window = df['close'].iloc[i-lookback:i]
        if len(lookback_window) == 0:
            continue

        prev_low_idx = lookback_window.idxmin()
        prev_low_price = df['close'].iloc[prev_low_idx]
        prev_low_rsi = df['rsi'].iloc[prev_low_idx]

        # Check for bullish divergence
        # Price: lower low (current < previous)
        # RSI: higher low (current > previous)
        if current_price < prev_low_price and current_rsi > prev_low_rsi:
            # Calculate divergence strength
            price_diff = (prev_low_price - current_price) / prev_low_price
            rsi_diff = (current_rsi - prev_low_rsi) / 100

            # Stronger divergence = larger differences
            strength = min(price_diff + rsi_diff, 1.0)
            divergence_signal[i] = strength

    return divergence_signal


def calculate_support_bounce(df, support_lookback=100, bounce_threshold=0.01):
    """
    Detect bounce from support level

    Logic:
    - Identify recent support (lowest low in lookback)
    - When price approaches support and bounces = exit signal

    Returns:
        bounce_signal: 0-1, higher = stronger bounce from support
    """
    bounce_signal = np.zeros(len(df))

    for i in range(support_lookback, len(df)):
        # Find support level (recent low)
        recent_lows = df['low'].iloc[i-support_lookback:i]
        support_level = recent_lows.min()

        # Current price relative to support
        current_price = df['close'].iloc[i]
        distance_to_support = (current_price - support_level) / support_level

        # Bounce: price near support (<1% above) and moving up
        price_moving_up = df['close'].iloc[i] > df['close'].iloc[i-1]

        if distance_to_support < bounce_threshold and price_moving_up:
            # Signal strength: closer to support = stronger
            strength = 1.0 - (distance_to_support / bounce_threshold)
            bounce_signal[i] = np.clip(strength, 0, 1)

    return bounce_signal


def calculate_consecutive_green_candles(df, max_count=5):
    """
    Count consecutive green (bullish) candles

    For SHORT: Multiple green candles = danger, should exit

    Returns:
        green_count_normalized: 0-1, normalized by max_count
    """
    # Green candle: close > open
    is_green = (df['close'] > df['open']).astype(int)

    # Count consecutive greens
    consecutive_count = np.zeros(len(df))
    count = 0

    for i in range(len(df)):
        if is_green.iloc[i] == 1:
            count += 1
        else:
            count = 0
        consecutive_count[i] = count

    # Normalize by max_count
    green_count_normalized = np.clip(consecutive_count / max_count, 0, 1)

    return green_count_normalized


def calculate_buy_pressure(df, lookback=5):
    """
    Calculate buy pressure indicator

    Logic:
    - Compare closes to opens over recent candles
    - High closes relative to opens = buy pressure
    - For SHORT: high buy pressure = exit signal

    Returns:
        buy_pressure: 0-1, higher = stronger buy pressure
    """
    # Candle body strength: (close - open) / (high - low)
    # Positive = bullish, negative = bearish
    body_ratio = (df['close'] - df['open']) / (df['high'] - df['low']).replace(0, 1)

    # Average body ratio over lookback
    avg_body_ratio = body_ratio.rolling(lookback).mean()

    # Buy pressure: normalized positive body ratio
    buy_pressure = np.clip((avg_body_ratio + 1) / 2, 0, 1)  # Map [-1,1] to [0,1]

    return buy_pressure


def calculate_short_liquidation_cascade(df, lookback=10, threshold=0.02):
    """
    Detect potential short liquidation cascade

    Logic:
    - Rapid price increase (>2% in short time) can trigger cascading short liquidations
    - This creates even stronger upward pressure
    - For SHORT position: URGENT exit signal

    Returns:
        cascade_signal: 0-1, higher = stronger cascade risk
    """
    # Price change over lookback
    price_change = df['close'].pct_change(lookback)

    # Cascade risk: rapid upward move
    cascade_signal = np.where(
        price_change > threshold,
        np.clip((price_change - threshold) / threshold, 0, 1),  # Normalized strength
        0
    )

    return cascade_signal


def add_all_reversal_features(df):
    """
    Add all reversal detection features to DataFrame

    Args:
        df: DataFrame with OHLCV and basic indicators (RSI, etc.)

    Returns:
        df: DataFrame with additional reversal features
    """
    print("\nCalculating reversal detection features...")

    # 1. Momentum reversal
    df['price_momentum_reversal'] = calculate_momentum_reversal(df, lookback=5)

    # 2. Volume spike on bounce
    df['volume_spike_on_bounce'] = calculate_volume_spike_on_bounce(df, lookback=20, spike_threshold=1.5)

    # 3. RSI divergence (real)
    df['rsi_divergence_real'] = calculate_rsi_divergence_real(df, lookback=10)

    # 4. Support bounce
    df['support_bounce'] = calculate_support_bounce(df, support_lookback=100, bounce_threshold=0.01)

    # 5. Consecutive green candles
    df['consecutive_green_candles'] = calculate_consecutive_green_candles(df, max_count=5)

    # 6. Buy pressure
    df['buy_pressure'] = calculate_buy_pressure(df, lookback=5)

    # 7. Short liquidation cascade risk
    df['short_liquidation_cascade'] = calculate_short_liquidation_cascade(df, lookback=10, threshold=0.02)

    # Composite reversal signal (weighted average)
    df['reversal_composite'] = (
        df['price_momentum_reversal'] * 0.2 +
        df['volume_spike_on_bounce'] * 0.15 +
        df['rsi_divergence_real'] * 0.25 +
        df['support_bounce'] * 0.15 +
        df['consecutive_green_candles'] * 0.1 +
        df['buy_pressure'] * 0.1 +
        df['short_liquidation_cascade'] * 0.05
    )

    print("✅ Reversal features calculated:")
    print("   - price_momentum_reversal")
    print("   - volume_spike_on_bounce")
    print("   - rsi_divergence_real")
    print("   - support_bounce")
    print("   - consecutive_green_candles")
    print("   - buy_pressure")
    print("   - short_liquidation_cascade")
    print("   - reversal_composite (weighted)")

    # Clean NaN
    reversal_cols = [
        'price_momentum_reversal', 'volume_spike_on_bounce', 'rsi_divergence_real',
        'support_bounce', 'consecutive_green_candles', 'buy_pressure',
        'short_liquidation_cascade', 'reversal_composite'
    ]

    for col in reversal_cols:
        df[col] = df[col].fillna(0)

    return df


# Example usage
if __name__ == "__main__":
    print("Reversal Detection Features Module")
    print("="*50)
    print("\nThis module provides specialized features for detecting")
    print("price reversals during SHORT positions.")
    print("\nFeatures:")
    print("  1. price_momentum_reversal - downtrend reversing")
    print("  2. volume_spike_on_bounce - high volume + price up")
    print("  3. rsi_divergence_real - price low + RSI high (bullish)")
    print("  4. support_bounce - bounce from support level")
    print("  5. consecutive_green_candles - multiple bullish candles")
    print("  6. buy_pressure - buying strength indicator")
    print("  7. short_liquidation_cascade - rapid upward move")
    print("  8. reversal_composite - weighted combination")
    print("\nUsage:")
    print("  from reversal_detection_features import add_all_reversal_features")
    print("  df = add_all_reversal_features(df)")
