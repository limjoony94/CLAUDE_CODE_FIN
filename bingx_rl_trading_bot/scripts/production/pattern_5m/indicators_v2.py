"""
Pattern 5m Bot - Improved Technical Indicators v2
ATR-normalized candle classification with classic pattern detection.

Key Improvements over v1:
1. ATR-based normalization (adaptive to market volatility)
2. 36-type classification (Size × Wick × Direction)
3. Quantified classic patterns (Engulfing, Hammer, Three Inside)
4. Context filters (position, trend)
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, Tuple, Optional
from enum import Enum

logger = logging.getLogger('pattern_5m')


# ============================================================
# Constants for ATR-Normalized Classification
# ============================================================

# Body Size thresholds (ATR ratio)
BODY_TINY_MAX = 0.2      # < 0.2 ATR = Doji family
BODY_SMALL_MAX = 0.5     # 0.2-0.5 ATR = Small
BODY_MEDIUM_MAX = 1.0    # 0.5-1.0 ATR = Medium
# > 1.0 ATR = Large

# Wick classification thresholds
WICK_RATIO_THRESHOLD = 2.0    # One wick > 2x other = dominant
NO_WICK_THRESHOLD = 0.15      # Total wick < 15% range = no wick

# Direction threshold
DIRECTION_THRESHOLD = 0.03    # 3% of range for neutral

# Context lookback
POSITION_LOOKBACK = 20
TREND_EMA_PERIOD = 20

# Pattern validation
MIN_ENGULF_BODY_ATR = 0.3     # Minimum body size for engulfing
ENGULF_RATIO = 1.2            # Current body > prev body * ratio
HAMMER_WICK_RATIO = 2.0       # Lower wick > 2x body
HAMMER_OPPOSITE_MAX = 0.3     # Opposite wick < 0.3x body
MIN_HAMMER_BODY_ATR = 0.15    # Minimum body for hammer


class BodySize(Enum):
    """4-level body size classification."""
    TINY = "T"      # Doji family
    SMALL = "S"     # Small body
    MEDIUM = "M"    # Medium body
    LARGE = "L"     # Large body (Marubozu family)


class WickType(Enum):
    """3-level wick balance classification."""
    FULL = "F"      # No/minimal wicks (full body)
    TOP = "T"       # Top wick dominant
    BOT = "B"       # Bottom wick dominant
    BAL = "X"       # Balanced wicks


class Direction(Enum):
    """3-level direction classification."""
    UP = "U"        # Bullish
    DN = "D"        # Bearish
    NEU = "N"       # Neutral (Doji)


# ============================================================
# ATR Calculation
# ============================================================

def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Calculate Average True Range.

    Args:
        df: DataFrame with high, low, close columns
        period: ATR period (default 14)

    Returns:
        Series with ATR values
    """
    high = df['high']
    low = df['low']
    close = df['close']

    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))

    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()

    return atr


# ============================================================
# V2 Candle Classification
# ============================================================

def classify_body_size(body_abs: float, atr: float) -> BodySize:
    """
    Classify body size relative to ATR.

    Args:
        body_abs: Absolute body size
        atr: Current ATR value

    Returns:
        BodySize enum
    """
    if atr <= 0:
        return BodySize.MEDIUM

    ratio = body_abs / atr

    if ratio < BODY_TINY_MAX:
        return BodySize.TINY
    elif ratio < BODY_SMALL_MAX:
        return BodySize.SMALL
    elif ratio < BODY_MEDIUM_MAX:
        return BodySize.MEDIUM
    else:
        return BodySize.LARGE


def classify_wick_type(
    upper_wick: float,
    lower_wick: float,
    range_hl: float
) -> WickType:
    """
    Classify wick balance.

    Args:
        upper_wick: Upper wick size
        lower_wick: Lower wick size
        range_hl: High-Low range

    Returns:
        WickType enum
    """
    if range_hl <= 0:
        return WickType.BAL

    total_wick = upper_wick + lower_wick

    # Full body (no wicks)
    if total_wick < NO_WICK_THRESHOLD * range_hl:
        return WickType.FULL

    # Wick dominance
    if upper_wick > WICK_RATIO_THRESHOLD * lower_wick and lower_wick > 0:
        return WickType.TOP
    elif lower_wick > WICK_RATIO_THRESHOLD * upper_wick and upper_wick > 0:
        return WickType.BOT
    elif upper_wick > 0 and lower_wick == 0:
        return WickType.TOP
    elif lower_wick > 0 and upper_wick == 0:
        return WickType.BOT
    else:
        return WickType.BAL


def classify_direction(body: float, range_hl: float) -> Direction:
    """
    Classify candle direction.

    Args:
        body: Signed body (close - open)
        range_hl: High-Low range

    Returns:
        Direction enum
    """
    if range_hl <= 0:
        return Direction.NEU

    body_ratio = body / range_hl

    if body_ratio > DIRECTION_THRESHOLD:
        return Direction.UP
    elif body_ratio < -DIRECTION_THRESHOLD:
        return Direction.DN
    else:
        return Direction.NEU


def classify_candle_v2(row: pd.Series, atr: float) -> str:
    """
    Classify a candle using the V2 system (ATR-normalized).

    Returns a 3-character code: Size-Wick-Direction
    Examples:
    - "L-F-U" = Large, Full body, Up (Bullish Marubozu)
    - "T-B-N" = Tiny, Bottom wick, Neutral (Dragonfly Doji)
    - "M-T-D" = Medium, Top wick, Down (Shooting Star type)

    Args:
        row: OHLCV row
        atr: Current ATR value

    Returns:
        3-character classification code
    """
    o, h, l, c = row['open'], row['high'], row['low'], row['close']

    body = c - o
    body_abs = abs(body)
    range_hl = h - l

    if range_hl == 0:
        return "T-X-N"  # Doji with no range

    upper_wick = h - max(o, c)
    lower_wick = min(o, c) - l

    # Classify each dimension
    size = classify_body_size(body_abs, atr)
    wick = classify_wick_type(upper_wick, lower_wick, range_hl)
    direction = classify_direction(body, range_hl)

    return f"{size.value}-{wick.value}-{direction.value}"


# ============================================================
# Classic Pattern Detection (Quantified)
# ============================================================

def is_bullish_engulfing(
    curr: pd.Series,
    prev: pd.Series,
    atr: float
) -> bool:
    """
    Detect Bullish Engulfing pattern with ATR normalization.

    Conditions:
    1. Previous candle is bearish
    2. Current candle is bullish
    3. Current body engulfs previous body
    4. Current body > MIN_ENGULF_BODY_ATR * ATR
    5. Current body > previous body * ENGULF_RATIO
    """
    # Basic conditions
    prev_bearish = prev['close'] < prev['open']
    curr_bullish = curr['close'] > curr['open']

    if not (prev_bearish and curr_bullish):
        return False

    # Engulfing condition
    engulfs = (
        curr['close'] > prev['open'] and
        curr['open'] < prev['close']
    )

    if not engulfs:
        return False

    # Size filters
    curr_body = abs(curr['close'] - curr['open'])
    prev_body = abs(prev['close'] - prev['open'])

    if atr > 0:
        size_filter = curr_body >= MIN_ENGULF_BODY_ATR * atr
    else:
        size_filter = True

    ratio_filter = prev_body > 0 and curr_body >= prev_body * ENGULF_RATIO

    return size_filter and ratio_filter


def is_bearish_engulfing(
    curr: pd.Series,
    prev: pd.Series,
    atr: float
) -> bool:
    """
    Detect Bearish Engulfing pattern with ATR normalization.
    """
    # Basic conditions
    prev_bullish = prev['close'] > prev['open']
    curr_bearish = curr['close'] < curr['open']

    if not (prev_bullish and curr_bearish):
        return False

    # Engulfing condition
    engulfs = (
        curr['close'] < prev['open'] and
        curr['open'] > prev['close']
    )

    if not engulfs:
        return False

    # Size filters
    curr_body = abs(curr['close'] - curr['open'])
    prev_body = abs(prev['close'] - prev['open'])

    if atr > 0:
        size_filter = curr_body >= MIN_ENGULF_BODY_ATR * atr
    else:
        size_filter = True

    ratio_filter = prev_body > 0 and curr_body >= prev_body * ENGULF_RATIO

    return size_filter and ratio_filter


def is_hammer(
    row: pd.Series,
    atr: float,
    recent_low: float
) -> bool:
    """
    Detect Hammer pattern at support level.

    Conditions:
    1. Lower wick > 2x body
    2. Upper wick < 0.3x body
    3. Body > MIN_HAMMER_BODY_ATR * ATR
    4. At recent low (within 1% of lookback low)
    """
    body = abs(row['close'] - row['open'])
    lower_wick = min(row['open'], row['close']) - row['low']
    upper_wick = row['high'] - max(row['open'], row['close'])

    if body == 0:
        return False

    # Shape conditions
    shape_ok = (
        lower_wick >= HAMMER_WICK_RATIO * body and
        upper_wick <= HAMMER_OPPOSITE_MAX * body
    )

    if not shape_ok:
        return False

    # Size condition
    if atr > 0:
        size_ok = body >= MIN_HAMMER_BODY_ATR * atr
    else:
        size_ok = True

    # Position condition (at recent low)
    at_bottom = row['low'] <= recent_low * 1.01

    return size_ok and at_bottom


def is_shooting_star(
    row: pd.Series,
    atr: float,
    recent_high: float
) -> bool:
    """
    Detect Shooting Star pattern at resistance level.
    """
    body = abs(row['close'] - row['open'])
    upper_wick = row['high'] - max(row['open'], row['close'])
    lower_wick = min(row['open'], row['close']) - row['low']

    if body == 0:
        return False

    # Shape conditions
    shape_ok = (
        upper_wick >= HAMMER_WICK_RATIO * body and
        lower_wick <= HAMMER_OPPOSITE_MAX * body
    )

    if not shape_ok:
        return False

    # Size condition
    if atr > 0:
        size_ok = body >= MIN_HAMMER_BODY_ATR * atr
    else:
        size_ok = True

    # Position condition (at recent high)
    at_top = row['high'] >= recent_high * 0.99

    return size_ok and at_top


def is_three_inside_up(
    c1: pd.Series,
    c2: pd.Series,
    c3: pd.Series,
    atr: float
) -> bool:
    """
    Detect Three Inside Up pattern.

    1. First: Large bearish candle
    2. Second: Small bullish candle inside first
    3. Third: Closes above first's open
    """
    # First candle: large bearish
    first_bearish = c1['close'] < c1['open']
    first_body = abs(c1['close'] - c1['open'])
    first_large = atr > 0 and first_body >= 0.7 * atr

    if not (first_bearish and first_large):
        return False

    # Second candle: bullish inside first
    second_bullish = c2['close'] > c2['open']
    inside = (
        c2['high'] < c1['high'] and
        c2['low'] > c1['low']
    )

    if not (second_bullish and inside):
        return False

    # Third candle: breaks above first's open
    third_breakout = c3['close'] > c1['open']

    return third_breakout


def is_three_inside_down(
    c1: pd.Series,
    c2: pd.Series,
    c3: pd.Series,
    atr: float
) -> bool:
    """
    Detect Three Inside Down pattern.
    """
    # First candle: large bullish
    first_bullish = c1['close'] > c1['open']
    first_body = abs(c1['close'] - c1['open'])
    first_large = atr > 0 and first_body >= 0.7 * atr

    if not (first_bullish and first_large):
        return False

    # Second candle: bearish inside first
    second_bearish = c2['close'] < c2['open']
    inside = (
        c2['high'] < c1['high'] and
        c2['low'] > c1['low']
    )

    if not (second_bearish and inside):
        return False

    # Third candle: breaks below first's open
    third_breakout = c3['close'] < c1['open']

    return third_breakout


# ============================================================
# Main Indicator Calculation
# ============================================================

def calculate_indicators_v2(
    df: pd.DataFrame,
    config: Dict[str, Any]
) -> pd.DataFrame:
    """
    Calculate all indicators for V2 pattern detection.

    Adds columns:
    - atr: 14-period ATR
    - candle_code: V2 classification (e.g., "M-B-U")
    - pattern_2: 2-candle pattern code
    - pattern_3: 3-candle pattern code
    - classic_pattern: Detected classic pattern name
    - recent_high, recent_low: For context
    - ema_20: Trend filter

    Args:
        df: DataFrame with OHLCV data
        config: Bot configuration

    Returns:
        DataFrame with indicator columns added
    """
    if len(df) < 30:
        logger.warning(f"Insufficient data for V2 classification: {len(df)} bars")
        return df

    df = df.copy()

    # Calculate ATR
    df['atr'] = calculate_atr(df, period=14)

    # Calculate EMA for trend
    df['ema_20'] = df['close'].ewm(span=TREND_EMA_PERIOD, adjust=False).mean()

    # Calculate context: recent high/low
    df['recent_high'] = df['high'].rolling(window=POSITION_LOOKBACK).max()
    df['recent_low'] = df['low'].rolling(window=POSITION_LOOKBACK).min()

    # Classify each candle using V2 system
    candle_codes = []
    for i in range(len(df)):
        if i < 14 or pd.isna(df.iloc[i]['atr']):
            candle_codes.append("M-X-N")  # Default
        else:
            candle_codes.append(
                classify_candle_v2(df.iloc[i], df.iloc[i]['atr'])
            )

    df['candle_code'] = candle_codes

    # Build 2-candle and 3-candle patterns
    pattern_2 = []
    pattern_3 = []

    for i in range(len(df)):
        if i < 1:
            pattern_2.append(None)
        else:
            p2 = f"{df.iloc[i-1]['candle_code']}_{df.iloc[i]['candle_code']}"
            pattern_2.append(p2)

        if i < 2:
            pattern_3.append(None)
        else:
            p3 = f"{df.iloc[i-2]['candle_code']}_{df.iloc[i-1]['candle_code']}_{df.iloc[i]['candle_code']}"
            pattern_3.append(p3)

    df['pattern_2'] = pattern_2
    df['pattern_3'] = pattern_3

    # Detect classic patterns
    classic_patterns = []

    for i in range(len(df)):
        pattern = None

        if i < 2 or pd.isna(df.iloc[i]['atr']):
            classic_patterns.append(None)
            continue

        atr = df.iloc[i]['atr']
        curr = df.iloc[i]
        prev = df.iloc[i-1]

        # Check Engulfing
        if is_bullish_engulfing(curr, prev, atr):
            pattern = "BULL_ENGULF"
        elif is_bearish_engulfing(curr, prev, atr):
            pattern = "BEAR_ENGULF"

        # Check Hammer/Shooting Star (if not engulfing)
        if pattern is None:
            recent_low = df.iloc[i]['recent_low']
            recent_high = df.iloc[i]['recent_high']

            if not pd.isna(recent_low) and is_hammer(curr, atr, recent_low):
                pattern = "HAMMER"
            elif not pd.isna(recent_high) and is_shooting_star(curr, atr, recent_high):
                pattern = "SHOOT_STAR"

        # Check Three Inside patterns
        if pattern is None and i >= 2:
            c1, c2, c3 = df.iloc[i-2], df.iloc[i-1], df.iloc[i]

            if is_three_inside_up(c1, c2, c3, atr):
                pattern = "THREE_IN_UP"
            elif is_three_inside_down(c1, c2, c3, atr):
                pattern = "THREE_IN_DN"

        classic_patterns.append(pattern)

    df['classic_pattern'] = classic_patterns

    # Add trend context
    df['above_ema'] = df['close'] > df['ema_20']

    return df


def get_signal_v2(
    df: pd.DataFrame,
    config: Dict[str, Any]
) -> Tuple[Optional[str], str]:
    """
    Generate trading signal using V2 classification.

    Returns:
        Tuple of (direction, reason) or (None, "")
    """
    if len(df) < 3:
        return None, ""

    # Get latest completed candle (index -2, as -1 is forming)
    idx = -2
    row = df.iloc[idx]

    classic = row.get('classic_pattern')
    candle_code = row.get('candle_code', '')
    above_ema = row.get('above_ema', True)

    # Priority 1: Classic patterns
    if classic == "BULL_ENGULF":
        return "LONG", f"Bullish Engulfing ({candle_code})"

    if classic == "BEAR_ENGULF":
        return "SHORT", f"Bearish Engulfing ({candle_code})"

    if classic == "HAMMER":
        return "LONG", f"Hammer at support ({candle_code})"

    if classic == "SHOOT_STAR":
        return "SHORT", f"Shooting Star at resistance ({candle_code})"

    if classic == "THREE_IN_UP":
        return "LONG", f"Three Inside Up ({candle_code})"

    if classic == "THREE_IN_DN":
        return "SHORT", f"Three Inside Down ({candle_code})"

    # No classic pattern detected
    return None, ""


# ============================================================
# Analysis Utilities
# ============================================================

def analyze_candle_distribution(df: pd.DataFrame) -> Dict[str, int]:
    """
    Analyze distribution of candle codes in data.

    Returns dict of code -> count
    """
    if 'candle_code' not in df.columns:
        return {}

    return df['candle_code'].value_counts().to_dict()


def analyze_pattern_performance(
    df: pd.DataFrame,
    tp_pct: float = 2.5,
    sl_pct: float = 2.0,
    leverage: int = 3
) -> pd.DataFrame:
    """
    Analyze performance of each classic pattern.

    Returns DataFrame with pattern performance metrics.
    """
    results = []

    patterns = df[df['classic_pattern'].notna()].copy()

    for pattern_name in patterns['classic_pattern'].unique():
        pattern_signals = patterns[patterns['classic_pattern'] == pattern_name]

        wins, losses = 0, 0
        total_pnl = 0

        for idx in pattern_signals.index:
            i = df.index.get_loc(idx)

            if i + 1 >= len(df):
                continue

            # Determine direction
            if pattern_name in ['BULL_ENGULF', 'HAMMER', 'THREE_IN_UP']:
                direction = 'LONG'
            else:
                direction = 'SHORT'

            entry_price = df.iloc[i + 1]['open']

            if direction == 'LONG':
                tp_price = entry_price * (1 + tp_pct / 100)
                sl_price = entry_price * (1 - sl_pct / 100)
            else:
                tp_price = entry_price * (1 - tp_pct / 100)
                sl_price = entry_price * (1 + sl_pct / 100)

            # Find exit
            for j in range(i + 1, min(i + 501, len(df))):
                high = df.iloc[j]['high']
                low = df.iloc[j]['low']

                if direction == 'LONG':
                    if low <= sl_price:
                        losses += 1
                        total_pnl -= sl_pct * leverage
                        break
                    if high >= tp_price:
                        wins += 1
                        total_pnl += tp_pct * leverage
                        break
                else:
                    if high >= sl_price:
                        losses += 1
                        total_pnl -= sl_pct * leverage
                        break
                    if low <= tp_price:
                        wins += 1
                        total_pnl += tp_pct * leverage
                        break

        total = wins + losses
        wr = wins / total * 100 if total > 0 else 0
        avg_pnl = total_pnl / total if total > 0 else 0

        results.append({
            'pattern': pattern_name,
            'trades': total,
            'wins': wins,
            'losses': losses,
            'win_rate': wr,
            'total_pnl': total_pnl,
            'avg_pnl': avg_pnl
        })

    return pd.DataFrame(results)
