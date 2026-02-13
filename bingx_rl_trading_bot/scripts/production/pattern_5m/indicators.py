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


def classify_candle(row, avg_body_20: float) -> CandleType:
    """
    Classify a single candle into one of 12 types.

    Args:
        row: OHLCV row (pd.Series or namedtuple with open/high/low/close attrs)
        avg_body_20: 20-period average absolute body size

    Returns:
        CandleType enum value
    """
    o, h, l, c = row.open, row.high, row.low, row.close
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
    Calculate all indicators needed for pattern detection and context filtering.

    Adds columns:
    - body, body_abs, avg_body_20: Body metrics
    - candle_type, type_code: Classification
    - pattern_3: 3-candle pattern string
    - rsi: 14-period RSI
    - atr, atr_pct: 14-period ATR and ATR%

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

    # Classify each candle using itertuples (faster than df.iloc[i])
    avg_body_vals = df['avg_body_20'].values
    candle_types = []
    for i, row in enumerate(df.itertuples(index=False)):
        avg_b = avg_body_vals[i]
        if pd.isna(avg_b):
            avg_b = 1.0
        candle_types.append(classify_candle(row, avg_b))

    df['candle_type'] = candle_types
    df['type_code'] = [ct.value for ct in candle_types]

    # Build 3-candle patterns using pre-extracted array
    type_codes = df['type_code'].values
    patterns = [None, None] + [
        f"{type_codes[i-2]}-{type_codes[i-1]}-{type_codes[i]}"
        for i in range(2, len(type_codes))
    ]
    df['pattern_3'] = patterns

    # RSI (14-period) — used by context filters
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / (loss + 1e-10)
    df['rsi'] = 100 - (100 / (1 + rs))

    # ATR and ATR% (14-period) — used by context filters and volatility multiplier
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift(1)).abs()
    low_close = (df['low'] - df['close'].shift(1)).abs()
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    df['atr'] = tr.rolling(14).mean()
    df['atr_pct'] = df['atr'] / df['close'] * 100

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
