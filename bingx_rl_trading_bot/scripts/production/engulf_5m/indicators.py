"""
Engulf 5m Bot - Technical Indicators
Calculate technical indicators and signal patterns.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any

logger = logging.getLogger('engulf_5m')


def calculate_indicators(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Calculate all technical indicators for signal detection.

    Args:
        df: DataFrame with OHLCV data
        config: Bot configuration

    Returns:
        DataFrame with calculated indicators
    """
    df = df.copy()
    strategy = config.get('strategy', {})

    # Basic candle properties
    df = _calculate_candle_properties(df)

    # Volume indicators
    df = _calculate_volume_indicators(df, strategy)

    # Engulfing pattern detection
    df = _detect_engulfing_patterns(df, strategy)

    # ATR for volatility
    df = _calculate_atr(df)

    return df


def _calculate_candle_properties(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate basic candle properties."""
    df['body'] = df['close'] - df['open']
    df['prev_body'] = df['body'].shift(1)
    df['candle_range'] = df['high'] - df['low']
    df['body_ratio'] = abs(df['body']) / df['candle_range'].replace(0, np.nan)
    df['body_pct'] = abs(df['body']) / df['open'] * 100
    df['prev_candle_range'] = df['candle_range'].shift(1)
    df['prev_body_ratio'] = abs(df['prev_body']) / df['prev_candle_range'].replace(0, np.nan)

    return df


def _calculate_volume_indicators(df: pd.DataFrame, strategy: Dict[str, Any]) -> pd.DataFrame:
    """Calculate volume-based indicators."""
    volume_ma_period = strategy.get('volume_ma_period', 20)
    df['volume_ma'] = df['volume'].rolling(window=volume_ma_period).mean()
    df['volume_ratio'] = df['volume'] / df['volume_ma'].replace(0, np.nan)

    return df


def _detect_engulfing_patterns(df: pd.DataFrame, strategy: Dict[str, Any]) -> pd.DataFrame:
    """
    Detect bullish and bearish engulfing patterns.

    Gap condition (open vs prev_close) filters for stronger engulfing patterns.
    Despite seeming inappropriate for 24/7 markets, it improved WR 60.5% vs 43.7%.
    """
    # Base conditions for bullish engulfing
    bullish_base = (
        (df['body'] > 0) &                        # Current candle is bullish
        (df['prev_body'] < 0) &                   # Previous candle was bearish
        (df['close'] > df['open'].shift(1)) &    # Close above prev open (engulfs)
        (df['open'] < df['close'].shift(1))      # Gap: open < prev close (quality filter)
    )

    # Base conditions for bearish engulfing
    bearish_base = (
        (df['body'] < 0) &                        # Current candle is bearish
        (df['prev_body'] > 0) &                   # Previous candle was bullish
        (df['close'] < df['open'].shift(1)) &    # Close below prev open (engulfs)
        (df['open'] > df['close'].shift(1))      # Gap: open > prev close (quality filter)
    )

    # Apply optional filters
    bullish_base, bearish_base = _apply_body_filter(df, strategy, bullish_base, bearish_base)
    bullish_base, bearish_base = _apply_prev_body_filter(df, strategy, bullish_base, bearish_base)
    bullish_base, bearish_base = _apply_volume_filter(df, strategy, bullish_base, bearish_base)
    bullish_base, bearish_base = _apply_body_pct_filter(df, strategy, bullish_base, bearish_base)

    df['bullish_engulf'] = bullish_base
    df['bearish_engulf'] = bearish_base

    return df


def _apply_body_filter(
    df: pd.DataFrame,
    strategy: Dict[str, Any],
    bullish: pd.Series,
    bearish: pd.Series
) -> tuple:
    """Apply body ratio filter if enabled."""
    if strategy.get('use_body_filter', False):
        min_body_ratio = strategy.get('min_body_ratio', 0.5)
        body_filter = df['body_ratio'] >= min_body_ratio
        bullish = bullish & body_filter
        bearish = bearish & body_filter

    return bullish, bearish


def _apply_prev_body_filter(
    df: pd.DataFrame,
    strategy: Dict[str, Any],
    bullish: pd.Series,
    bearish: pd.Series
) -> tuple:
    """Apply previous body ratio filter if enabled."""
    if strategy.get('use_prev_body_filter', False):
        min_prev_body_ratio = strategy.get('min_prev_body_ratio', 0.3)
        prev_body_filter = df['prev_body_ratio'] >= min_prev_body_ratio
        bullish = bullish & prev_body_filter
        bearish = bearish & prev_body_filter

    return bullish, bearish


def _apply_volume_filter(
    df: pd.DataFrame,
    strategy: Dict[str, Any],
    bullish: pd.Series,
    bearish: pd.Series
) -> tuple:
    """Apply volume filter if enabled."""
    if strategy.get('use_volume_filter', False):
        min_volume_ratio = strategy.get('min_volume_ratio', 1.0)
        volume_filter = df['volume_ratio'] >= min_volume_ratio
        bullish = bullish & volume_filter
        bearish = bearish & volume_filter

    return bullish, bearish


def _apply_body_pct_filter(
    df: pd.DataFrame,
    strategy: Dict[str, Any],
    bullish: pd.Series,
    bearish: pd.Series
) -> tuple:
    """Apply body percentage filter if enabled."""
    if strategy.get('use_body_pct_filter', False):
        min_body_pct = strategy.get('min_body_pct', 0.24)
        body_pct_filter = df['body_pct'] >= min_body_pct
        bullish = bullish & body_pct_filter
        bearish = bearish & body_pct_filter

    return bullish, bearish


def _calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Calculate Average True Range (ATR).

    Args:
        df: DataFrame with OHLCV data
        period: ATR period (default: 14)

    Returns:
        DataFrame with ATR column
    """
    df['tr'] = np.maximum(
        df['high'] - df['low'],
        np.maximum(
            abs(df['high'] - df['close'].shift(1)),
            abs(df['low'] - df['close'].shift(1))
        )
    )
    df['atr'] = df['tr'].rolling(window=period).mean()

    return df


def get_volatility_multiplier(df: pd.DataFrame, config: Dict[str, Any]) -> float:
    """
    Calculate volatility-based TP/SL multiplier.

    Args:
        df: DataFrame with ATR data
        config: Bot configuration

    Returns:
        Multiplier value (default 1.0)
    """
    strategy = config.get('strategy', {})
    vol_adaptive = strategy.get('vol_adaptive', {})

    if not vol_adaptive.get('enabled', False):
        return 1.0

    lookback = vol_adaptive.get('lookback', 75)
    thresholds = vol_adaptive.get('thresholds', [0.3, 0.6, 0.9])
    multipliers = vol_adaptive.get('multipliers', [0.8, 1.0, 1.15, 1.4])

    if 'atr' not in df.columns or len(df) < lookback:
        return 1.0

    recent_atr = df['atr'].iloc[-lookback:].dropna()
    if len(recent_atr) < lookback // 2:
        return 1.0

    current_atr = df['atr'].iloc[-1]
    if pd.isna(current_atr):
        return 1.0

    atr_min = recent_atr.min()
    atr_max = recent_atr.max()

    if atr_max == atr_min:
        percentile = 0.5
    else:
        percentile = (current_atr - atr_min) / (atr_max - atr_min)

    # Determine multiplier based on percentile
    if percentile <= thresholds[0]:
        mult = multipliers[0]
    elif percentile <= thresholds[1]:
        mult = multipliers[1]
    elif percentile <= thresholds[2]:
        mult = multipliers[2]
    else:
        mult = multipliers[3]

    return mult
