"""
Pattern 5m Bot - Technical Indicators
Candle classification and pattern detection.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Any

from .constants import (
    CandleType,
    AVG_BODY_WINDOW,
    DOJI_BODY_RATIO_THRESHOLD,
    WICK_DOMINANCE_THRESHOLD,
    MARUBOZU_WICK_RATIO_THRESHOLD,
    HAMMER_WICK_TO_BODY_RATIO,
    HAMMER_OPPOSITE_WICK_RATIO,
    SPINNING_TOP_BODY_NORM,
    SPINNING_TOP_WICK_RATIO,
    BIG_CANDLE_NORM_THRESHOLD,
)

logger = logging.getLogger('pattern_5m')


def classify_candle(row: pd.Series, avg_body_20: float) -> CandleType:
    """
    Classify a single candle into one of 12 types.

    Args:
        row: OHLCV row from DataFrame
        avg_body_20: 20-period average absolute body size

    Returns:
        CandleType enum value
    """
    o, h, l, c = row['open'], row['high'], row['low'], row['close']
    body = c - o
    body_abs = abs(body)
    range_hl = h - l

    if range_hl == 0:
        return CandleType.DOJI

    upper_wick = h - max(o, c)
    lower_wick = min(o, c) - l
    body_ratio = body_abs / range_hl

    # Marubozu (checked first - very small wicks)
    total_wick_ratio = (upper_wick + lower_wick) / range_hl
    if total_wick_ratio < MARUBOZU_WICK_RATIO_THRESHOLD:
        return CandleType.MARUBOZU_UP if body > 0 else CandleType.MARUBOZU_DOWN

    # Hammer / Inverted Hammer (checked before DOJI family)
    # These have extreme wick-to-body ratios and should take priority
    if body_abs > 0:
        lower_to_body = lower_wick / body_abs
        upper_to_body = upper_wick / body_abs
        if lower_to_body > HAMMER_WICK_TO_BODY_RATIO and upper_to_body < HAMMER_OPPOSITE_WICK_RATIO:
            return CandleType.HAMMER
        if upper_to_body > HAMMER_WICK_TO_BODY_RATIO and lower_to_body < HAMMER_OPPOSITE_WICK_RATIO:
            return CandleType.INV_HAMMER

    # Doji family (checked after HAMMER to avoid misclassification)
    if body_ratio < DOJI_BODY_RATIO_THRESHOLD:
        lower_ratio = lower_wick / range_hl
        upper_ratio = upper_wick / range_hl
        if lower_ratio > WICK_DOMINANCE_THRESHOLD:
            return CandleType.DRAGONFLY
        elif upper_ratio > WICK_DOMINANCE_THRESHOLD:
            return CandleType.GRAVESTONE
        return CandleType.DOJI

    # Spinning Top
    norm_body = body_abs / avg_body_20 if avg_body_20 > 0 else 1.0
    if norm_body < SPINNING_TOP_BODY_NORM and body_abs > 0:
        if lower_wick >= SPINNING_TOP_WICK_RATIO * body_abs and upper_wick >= SPINNING_TOP_WICK_RATIO * body_abs:
            return CandleType.SPINNING_TOP

    # Big or Medium
    if norm_body > BIG_CANDLE_NORM_THRESHOLD:
        return CandleType.BIG_UP if body > 0 else CandleType.BIG_DOWN
    else:
        return CandleType.MED_UP if body > 0 else CandleType.MED_DOWN


def calculate_indicators(df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
    """
    Calculate all indicators needed for pattern detection.

    Adds columns:
    - body, body_abs, avg_body_20: Body metrics
    - candle_type, type_code: Classification
    - pattern_3: 3-candle pattern string

    Args:
        df: DataFrame with OHLCV data
        config: Bot configuration

    Returns:
        DataFrame with indicator columns added
    """
    if len(df) < AVG_BODY_WINDOW + 5:
        logger.warning(f"Insufficient data for classification: {len(df)} bars")
        return df

    df = df.copy()

    # Body calculations
    df['body'] = df['close'] - df['open']
    df['body_abs'] = df['body'].abs()
    df['avg_body_20'] = df['body_abs'].rolling(AVG_BODY_WINDOW).mean()

    # Classify each candle
    # For bars 0-19: avg_body_20 is NaN, so use default=1.0
    # This preserves range-based types (DOJI, HAMMER, DRAGONFLY, GRAVESTONE, MARUBOZU)
    # while norm_body-dependent types (SPINNING_TOP, BIG) default conservatively
    candle_types = []
    for i in range(len(df)):
        avg_b = df.iloc[i]['avg_body_20']
        if pd.isna(avg_b):
            avg_b = 1.0  # default: preserves range-based classification
        candle_types.append(classify_candle(df.iloc[i], avg_b))

    df['candle_type'] = candle_types
    df['type_code'] = [ct.value for ct in candle_types]

    # Build 3-candle patterns
    patterns = []
    for i in range(len(df)):
        if i < 2:
            patterns.append(None)
        else:
            p = f"{df.iloc[i-2]['type_code']}-{df.iloc[i-1]['type_code']}-{df.iloc[i]['type_code']}"
            patterns.append(p)
    df['pattern_3'] = patterns

    return df


def get_current_pattern(df: pd.DataFrame) -> str:
    """
    Get the current 3-candle pattern from the last completed candle.

    Args:
        df: DataFrame with pattern_3 column

    Returns:
        Pattern string or empty string if not available
    """
    if 'pattern_3' not in df.columns or len(df) < 3:
        return ""

    pattern = df.iloc[-2].get('pattern_3', '')  # Last completed candle
    return pattern if pattern else ""


def get_pattern_description(pattern: str) -> str:
    """
    Get human-readable description of a pattern.

    Args:
        pattern: Pattern string (e.g., "MU-U-DN")

    Returns:
        Description string
    """
    descriptions = {
        # LONG patterns (10) - v1.25.0
        "MD-BU-U": "Marubozu Down → Big Up → Up (LONG reversal momentum)",
        "MU-MU-U": "Marubozu Up → Marubozu Up → Up (LONG triple momentum)",
        "MU-U-MU": "Marubozu Up → Up → Marubozu Up (LONG momentum continuation)",
        "BU-BU-BD": "Big Up → Big Up → Big Down (LONG pullback entry)",
        "ST-H-DN": "Spinning → Hammer → Down (LONG hammer reversal)",
        "ST-MU-U": "Spinning → Marubozu Up → Up (LONG breakout)",
        "DN-IH-ST": "Down → Inv Hammer → Spinning (LONG indecision reversal)",
        "IH-DN-DN": "Inv Hammer → Down → Down (LONG oversold bounce)",
        "MD-DN-MU": "Marubozu Down → Down → Marubozu Up (LONG V-reversal)",
        "BD-ST-U": "Big Down → Spinning → Up (LONG bottom reversal)",
        # SHORT patterns (10) - v1.25.0
        "MD-ST-ST": "Marubozu Down → Spinning → Spinning (SHORT continuation)",
        "U-MU-BU": "Up → Marubozu Up → Big Up (SHORT exhaustion top)",
        "MU-BU-DN": "Marubozu Up → Big Up → Down (SHORT reversal)",
        "ST-H-U": "Spinning → Hammer → Up (SHORT trap setup)",
        "ST-DN-H": "Spinning → Down → Hammer (SHORT continuation)",
        "MD-MU-U": "Marubozu Down → Marubozu Up → Up (SHORT failed bounce)",
        "BU-U-ST": "Big Up → Up → Spinning (SHORT momentum exhaustion)",
        "H-DN-ST": "Hammer → Down → Spinning (SHORT breakdown)",
        "DN-BD-BU": "Down → Big Down → Big Up (SHORT bounce trap)",
        "DN-BU-U": "Down → Big Up → Up (SHORT dead cat bounce)",
    }
    return descriptions.get(pattern, pattern)


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
