"""
Pattern 5m Bot - Signal Detection
3-candle pattern detection and classification.
"""

import logging
import pandas as pd
from typing import Dict, Any, Optional, Tuple

import numpy as np

from .indicators import classify_candle, calculate_indicators
from .constants import (
    CandleType,
    VALIDATED_LONG_PATTERNS,
    VALIDATED_SHORT_PATTERNS,
    DOJI_BODY_RATIO_THRESHOLD,
    WICK_DOMINANCE_THRESHOLD,
    MARUBOZU_WICK_RATIO_THRESHOLD,
    HAMMER_WICK_TO_BODY_RATIO,
    HAMMER_OPPOSITE_WICK_RATIO,
    SPINNING_TOP_BODY_NORM,
    SPINNING_TOP_WICK_RATIO,
    BIG_CANDLE_NORM_THRESHOLD,
    AVG_BODY_WINDOW,
    PATTERN_STATS,
    CONFIDENCE_WEIGHT_CLARITY,
    CONFIDENCE_WEIGHT_HISTORICAL,
    CONFIDENCE_WEIGHT_REGIME,
    CONFIDENCE_LOG_FILE,
    EARLY_EXIT_CONFIG,
    EARLY_EXIT_BEARISH_TYPES,
    EARLY_EXIT_BULLISH_TYPES,
    EARLY_EXIT_CONFIRM_CANDLES,
    EARLY_EXIT_MIN_PROFIT_PCT,
    # v1.7 Context Filters
    PATTERN_CONTEXT_FILTERS,
    CONTEXT_FILTER_ENABLED,
    CONTEXT_PREFERRED_BONUS,
    RSI_OVERSOLD_THRESHOLD,
    RSI_OVERBOUGHT_THRESHOLD,
    VOL_LOW_QUANTILE,
    VOL_HIGH_QUANTILE,
    TREND_LOOKBACK_BARS,
    # v1.18 Regime Detection
    MarketRegime,
    REGIME_DETECTION_ENABLED,
    REGIME_LOOKBACK_BARS,
    REGIME_TREND_THRESHOLD,
    REGIME_VOL_THRESHOLD,
    REGIME_PATTERNS,
    DEFAULT_REGIME,
)

logger = logging.getLogger('pattern_5m')


def add_candle_classification(df: pd.DataFrame, config: dict = None) -> pd.DataFrame:
    """
    Add candle type classification to DataFrame.

    Delegates to indicators.calculate_indicators() — the single source of truth
    for candle classification. This wrapper is kept for backward compatibility.

    Args:
        df: DataFrame with OHLCV data
        config: Optional bot config (passed to calculate_indicators)

    Returns:
        DataFrame with added classification columns
    """
    return calculate_indicators(df, config or {})


def detect_market_regime(df: pd.DataFrame) -> MarketRegime:
    """
    Detect current market regime (v1.18).

    Uses:
    - Price change over lookback period for trend detection
    - ATR% for volatility detection

    Regimes:
    - BULL: price_change > 2%
    - BEAR: price_change < -2%
    - SIDEWAYS: -2% <= price_change <= 2%

    Research: regime_adaptive_research.py
    Distribution: SIDEWAYS 85.8%, BEAR 8.5%, BULL 5.4%

    Args:
        df: DataFrame with OHLCV data (needs at least REGIME_LOOKBACK_BARS + 14 bars)

    Returns:
        MarketRegime enum value
    """
    if not REGIME_DETECTION_ENABLED:
        return DEFAULT_REGIME

    min_bars = REGIME_LOOKBACK_BARS + 14  # Need lookback + ATR period
    if len(df) < min_bars:
        logger.debug(f"Not enough data for regime detection ({len(df)} < {min_bars}), using {DEFAULT_REGIME.value}")
        return DEFAULT_REGIME

    try:
        # Calculate price change over lookback period
        current_close = df.iloc[-2]['close']  # Last complete candle
        past_close = df.iloc[-2 - REGIME_LOOKBACK_BARS]['close']
        price_change = (current_close - past_close) / past_close * 100

        # Determine regime based on price change
        if price_change > REGIME_TREND_THRESHOLD:
            regime = MarketRegime.BULL
        elif price_change < -REGIME_TREND_THRESHOLD:
            regime = MarketRegime.BEAR
        else:
            regime = MarketRegime.SIDEWAYS

        logger.debug(
            f"📊 Regime detected: {regime.value} | "
            f"Price change ({REGIME_LOOKBACK_BARS} bars): {price_change:+.2f}%"
        )

        return regime

    except Exception as e:
        logger.warning(f"Regime detection error: {e}, using {DEFAULT_REGIME.value}")
        return DEFAULT_REGIME


def calculate_context(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Calculate market context for pattern filtering (v1.7).

    Context features:
    - rsi_zone: 'OS' (oversold <30), 'OB' (overbought >70), 'N' (neutral)
    - vol: 'L' (low), 'M' (medium), 'H' (high) based on ATR percentile
    - trend: 'UP' or 'DN' based on close vs close[TREND_LOOKBACK_BARS]

    Args:
        df: DataFrame with OHLCV data (needs at least 200 bars for ATR quantiles)

    Returns:
        Dict with context values: {'rsi_zone': str, 'vol': str, 'trend': str}
    """
    if len(df) < max(TREND_LOOKBACK_BARS + 1, 15):
        return {'rsi_zone': 'N', 'vol': 'M', 'trend': 'N'}

    current = df.iloc[-2]  # Last complete candle

    # Use pre-computed RSI from calculate_indicators() (avoids redundant computation)
    current_rsi = current.get('rsi', 50.0) if 'rsi' in df.columns else 50.0
    if pd.isna(current_rsi):
        current_rsi = 50.0

    if current_rsi < RSI_OVERSOLD_THRESHOLD:
        rsi_zone = 'OS'
    elif current_rsi > RSI_OVERBOUGHT_THRESHOLD:
        rsi_zone = 'OB'
    else:
        rsi_zone = 'N'

    # Use pre-computed ATR% from calculate_indicators() (avoids redundant computation)
    if 'atr_pct' in df.columns:
        current_atr_pct = current.get('atr_pct', 0)
        if pd.isna(current_atr_pct):
            current_atr_pct = 0
    else:
        current_atr_pct = 0

    # Calculate quantiles from recent data (use last 200 bars for stability)
    lookback = min(200, len(df) - 1)
    if 'atr_pct' in df.columns:
        recent_atr = df['atr_pct'].iloc[-lookback:-1]
    else:
        recent_atr = pd.Series([current_atr_pct])
    q_low = recent_atr.quantile(VOL_LOW_QUANTILE)
    q_high = recent_atr.quantile(VOL_HIGH_QUANTILE)

    if current_atr_pct < q_low:
        vol = 'L'
    elif current_atr_pct > q_high:
        vol = 'H'
    else:
        vol = 'M'

    # Trend (compare current close to close N bars ago)
    current_close = current['close']
    past_close = df.iloc[-2 - TREND_LOOKBACK_BARS]['close']
    trend = 'UP' if current_close > past_close else 'DN'

    return {'rsi_zone': rsi_zone, 'vol': vol, 'trend': trend, 'rsi': current_rsi}


def check_context_filter(
    pattern: str,
    context: Dict[str, Any]
) -> Tuple[bool, float, str]:
    """
    Check if pattern passes context filter (v1.7).

    Args:
        pattern: 3-candle pattern string (e.g., 'U-DN-DN')
        context: Context dict from calculate_context()

    Returns:
        Tuple of (passes_filter, confidence_bonus, filter_reason)
        - passes_filter: True if signal should be taken
        - confidence_bonus: Additional confidence (0.0 to CONTEXT_PREFERRED_BONUS)
        - filter_reason: Description of filter result
    """
    if not CONTEXT_FILTER_ENABLED:
        return True, 0.0, "context_disabled"

    filter_config = PATTERN_CONTEXT_FILTERS.get(pattern)
    if not filter_config:
        return True, 0.0, "no_filter"

    # Check required filters (must pass)
    required = filter_config.get('required', {})
    for ctx_key, allowed_values in required.items():
        current_value = context.get(ctx_key)
        if current_value not in allowed_values:
            return False, 0.0, f"required_{ctx_key}={current_value}_not_in_{allowed_values}"

    # Check excluded filters (must NOT match) - v1.8
    excluded = filter_config.get('excluded', {})
    for ctx_key, excluded_values in excluded.items():
        current_value = context.get(ctx_key)
        if current_value in excluded_values:
            return False, 0.0, f"excluded_{ctx_key}={current_value}_in_{excluded_values}"

    # Check preferred filters (bonus if matched)
    confidence_bonus = 0.0
    preferred = filter_config.get('preferred', {})
    matched_preferred = []
    for ctx_key, preferred_values in preferred.items():
        current_value = context.get(ctx_key)
        if current_value in preferred_values:
            confidence_bonus += CONTEXT_PREFERRED_BONUS
            matched_preferred.append(f"{ctx_key}={current_value}")

    if matched_preferred:
        return True, confidence_bonus, f"preferred_match:{','.join(matched_preferred)}"

    return True, 0.0, "passed"


def calculate_candle_clarity(row: pd.Series, avg_body_20: float) -> float:
    """
    Calculate how clearly a candle fits its classification.

    Returns a score from 0.0 to 1.0 indicating classification confidence.
    Higher scores mean the candle is a more "textbook" example of its type.
    """
    o, h, l, c = row['open'], row['high'], row['low'], row['close']
    body = c - o
    body_abs = abs(body)
    range_hl = h - l

    if range_hl == 0:
        return 0.5  # Neutral for zero-range candles

    upper_wick = h - max(o, c)
    lower_wick = min(o, c) - l
    body_ratio = body_abs / range_hl

    candle_type = row.get('candle_type')
    if candle_type is None:
        return 0.5

    # Calculate clarity based on how strongly the candle matches its type
    if candle_type == CandleType.DOJI:
        # Clearer if body_ratio is closer to 0
        return max(0.3, 1.0 - (body_ratio / DOJI_BODY_RATIO_THRESHOLD))

    elif candle_type in (CandleType.DRAGONFLY, CandleType.GRAVESTONE):
        # Clearer if dominant wick is larger
        dominant_wick_ratio = max(lower_wick, upper_wick) / range_hl
        return min(1.0, dominant_wick_ratio / WICK_DOMINANCE_THRESHOLD)

    elif candle_type in (CandleType.MARUBOZU_UP, CandleType.MARUBOZU_DOWN):
        # Clearer if wicks are smaller
        total_wick_ratio = (upper_wick + lower_wick) / range_hl
        return max(0.5, 1.0 - (total_wick_ratio / MARUBOZU_WICK_RATIO_THRESHOLD))

    elif candle_type in (CandleType.HAMMER, CandleType.INV_HAMMER):
        # Clearer if wick-to-body ratio is higher
        if body_abs > 0:
            wick_to_body = max(lower_wick, upper_wick) / body_abs
            return min(1.0, wick_to_body / (HAMMER_WICK_TO_BODY_RATIO * 1.5))
        return 0.5

    elif candle_type == CandleType.SPINNING_TOP:
        # Clearer if both wicks are balanced and body is small
        norm_body = body_abs / avg_body_20 if avg_body_20 > 0 else 1.0
        balance = 1.0 - abs(upper_wick - lower_wick) / max(upper_wick + lower_wick, 0.001)
        return min(1.0, (1.0 - norm_body) * 0.5 + balance * 0.5)

    elif candle_type in (CandleType.BIG_UP, CandleType.BIG_DOWN):
        # Clearer if normalized body is larger
        norm_body = body_abs / avg_body_20 if avg_body_20 > 0 else 1.0
        return min(1.0, norm_body / (BIG_CANDLE_NORM_THRESHOLD * 1.5))

    else:  # MED_UP, MED_DOWN
        return 0.6  # Medium candles have moderate clarity


def calculate_pattern_confidence(
    df: pd.DataFrame,
    pattern: str,
    idx: int = -2
) -> Tuple[float, Dict[str, float]]:
    """
    Calculate overall pattern confidence score.

    Components:
    - Classification clarity: How clearly each candle fits its type (40%)
    - Historical win rate: Pattern's historical performance (30%)
    - Regime alignment: Placeholder for future regime-based scoring (30%)

    Args:
        df: DataFrame with candle classification
        pattern: 3-candle pattern string
        idx: Index of the current (third) candle

    Returns:
        Tuple of (confidence_score, component_dict)
    """
    # Get the 3 candles of the pattern
    if len(df) < abs(idx) + 2:
        return 0.5, {"clarity": 0.5, "historical": 0.5, "regime": 0.5}

    c1_idx = idx - 2
    c2_idx = idx - 1
    c3_idx = idx

    # Component 1: Classification clarity (average of 3 candles)
    avg_body = df.iloc[c3_idx].get('avg_body_20', df['body'].abs().mean())
    clarity_scores = [
        calculate_candle_clarity(df.iloc[c1_idx], avg_body),
        calculate_candle_clarity(df.iloc[c2_idx], avg_body),
        calculate_candle_clarity(df.iloc[c3_idx], avg_body),
    ]
    clarity = sum(clarity_scores) / 3

    # Component 2: Historical win rate (normalized: 50%=0, 70%=1)
    # PATTERN_STATS WR is in percentage (57-98), convert to ratio for normalization
    pattern_stats = PATTERN_STATS.get(pattern, {})
    hist_wr = pattern_stats.get("wr", 50.0)
    historical = min(1.0, max(0.0, (hist_wr / 100.0 - 0.50) / 0.20))

    # Component 3: Regime alignment (placeholder - use neutral 0.6 for now)
    # Future: integrate with MTF trend analysis
    regime = 0.6

    # Weighted combination
    confidence = (
        CONFIDENCE_WEIGHT_CLARITY * clarity +
        CONFIDENCE_WEIGHT_HISTORICAL * historical +
        CONFIDENCE_WEIGHT_REGIME * regime
    )

    components = {
        "clarity": round(clarity, 3),
        "historical": round(historical, 3),
        "regime": round(regime, 3),
    }

    return round(confidence, 3), components


def check_entry_signal(
    df: pd.DataFrame,
    state: Dict[str, Any],
    config: Dict[str, Any]
) -> Tuple[Optional[str], Optional[str]]:
    """
    Check for entry signals based on 3-candle patterns.

    v1.18: Regime-Adaptive strategy - use regime-specific patterns
    v1.7: Added context filter support (RSI/Vol/Trend)

    Args:
        df: DataFrame with calculated indicators and pattern classification
        state: Bot state dictionary
        config: Bot configuration

    Returns:
        Tuple of (signal, reason) or (None, None) if no signal
    """
    if len(df) < 5:
        return None, None

    # Ensure classification is done (single source: indicators.calculate_indicators)
    if 'pattern_3' not in df.columns:
        df = calculate_indicators(df, config)

    current = df.iloc[-2]  # Last complete candle
    current_timestamp = current.get('timestamp', df.index[-2])

    # Prevent duplicate signals on same candle
    last_signal_candle = state.get('last_signal_candle_timestamp')
    if last_signal_candle and last_signal_candle == current_timestamp:
        return None, None

    pattern = current.get('pattern_3')
    if pattern is None:
        return None, None

    signal = None
    reason = None
    regime_tp_sl = None  # v1.18: Regime-specific TP/SL

    # v1.18: Detect market regime and use regime-specific patterns
    if REGIME_DETECTION_ENABLED:
        regime = detect_market_regime(df)
        regime_name = regime.value

        # Get patterns for current regime
        regime_patterns = REGIME_PATTERNS.get(regime_name, {})

        if pattern in regime_patterns:
            direction, tp, sl = regime_patterns[pattern]
            signal = direction
            reason = f"Pattern: {pattern} ({direction}) | Regime: {regime_name}"
            regime_tp_sl = (tp, sl)  # Store for later use
            state['current_regime'] = regime_name
            state['regime_tp_sl'] = regime_tp_sl

            logger.info(
                f"📊 Regime-Adaptive Signal: {pattern} → {direction} | "
                f"Regime: {regime_name} | TP={tp}%, SL={sl}%"
            )
    else:
        # Fallback to v1.17 behavior (non-regime-adaptive)
        # Check LONG patterns
        long_patterns = config.get('strategy', {}).get('long_patterns', VALIDATED_LONG_PATTERNS)
        if pattern in long_patterns:
            signal = 'LONG'
            reason = f"Pattern: {pattern} (LONG)"

        # Check SHORT patterns
        short_patterns = config.get('strategy', {}).get('short_patterns', VALIDATED_SHORT_PATTERNS)
        if pattern in short_patterns:
            signal = 'SHORT'
            reason = f"Pattern: {pattern} (SHORT)"

    if signal:
        # v1.7: Context filter check
        context = calculate_context(df)
        passes_filter, ctx_bonus, filter_reason = check_context_filter(pattern, context)

        if not passes_filter:
            logger.info(
                f"🚫 Context filter rejected: {pattern} | "
                f"RSI={context.get('rsi', 0):.1f} ({context['rsi_zone']}) | "
                f"Vol={context['vol']} | Trend={context['trend']} | "
                f"Reason: {filter_reason}"
            )
            return None, None

        # Log context info
        logger.info(
            f"✅ Context check passed: {pattern} | "
            f"RSI={context.get('rsi', 0):.1f} ({context['rsi_zone']}) | "
            f"Vol={context['vol']} | Trend={context['trend']} | "
            f"Bonus={ctx_bonus:.2f} | {filter_reason}"
        )

        state['last_signal_candle_timestamp'] = current_timestamp
        confidence, conf_components = calculate_pattern_confidence(df, pattern)

        # Add context bonus to confidence
        confidence = min(1.0, confidence + ctx_bonus)
        conf_components['context_bonus'] = ctx_bonus

        _log_signal_details(df, current, pattern, signal, confidence, conf_components)
        _save_confidence_to_csv(current_timestamp, pattern, signal, confidence, conf_components)

    return signal, reason


def _log_signal_details(
    df: pd.DataFrame,
    current: pd.Series,
    pattern: str,
    signal: str,
    confidence: float = 0.0,
    conf_components: Optional[Dict[str, float]] = None
) -> None:
    """Log detailed signal information including confidence score."""
    logger.info(
        f"📊 Pattern Signal: {pattern} → {signal} | Confidence: {confidence:.3f}"
    )

    # Log confidence components
    if conf_components:
        logger.info(
            f"📊 Confidence breakdown: "
            f"clarity={conf_components.get('clarity', 0):.3f}, "
            f"historical={conf_components.get('historical', 0):.3f}, "
            f"regime={conf_components.get('regime', 0):.3f}"
        )

    logger.info(
        f"📊 Current candle: O={current['open']:.2f} "
        f"H={current['high']:.2f} L={current['low']:.2f} "
        f"C={current['close']:.2f}"
    )

    # Log 3 candle sequence
    if len(df) >= 4:
        c1 = df.iloc[-4]  # First candle of pattern
        c2 = df.iloc[-3]  # Second candle of pattern
        c3 = df.iloc[-2]  # Third candle of pattern (current)
        logger.info(
            f"📊 3-candle sequence: "
            f"[{c1.get('type_code', '?')}] O={c1['open']:.0f} C={c1['close']:.0f} | "
            f"[{c2.get('type_code', '?')}] O={c2['open']:.0f} C={c2['close']:.0f} | "
            f"[{c3.get('type_code', '?')}] O={c3['open']:.0f} C={c3['close']:.0f}"
        )


def _save_confidence_to_csv(
    timestamp: Any,
    pattern: str,
    signal: str,
    confidence: float,
    conf_components: Dict[str, float]
) -> None:
    """
    Save confidence data to CSV for future analysis.

    This accumulates data to validate if confidence correlates with trade outcomes.
    """
    import os
    from datetime import datetime

    try:
        # Get project root (relative to this file)
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
        csv_path = os.path.join(project_root, CONFIDENCE_LOG_FILE)

        # Ensure directory exists
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)

        # Check if file exists (to write header)
        file_exists = os.path.exists(csv_path)

        # Format timestamp
        if isinstance(timestamp, (int, float)):
            ts_str = datetime.fromtimestamp(timestamp / 1000).strftime('%Y-%m-%d %H:%M:%S')
        elif hasattr(timestamp, 'strftime'):
            ts_str = timestamp.strftime('%Y-%m-%d %H:%M:%S')
        else:
            ts_str = str(timestamp)

        # Write to CSV
        with open(csv_path, 'a', encoding='utf-8') as f:
            if not file_exists:
                f.write("timestamp,pattern,signal,confidence,clarity,historical,regime,outcome\n")
            f.write(
                f"{ts_str},{pattern},{signal},{confidence:.3f},"
                f"{conf_components.get('clarity', 0):.3f},"
                f"{conf_components.get('historical', 0):.3f},"
                f"{conf_components.get('regime', 0):.3f},"
                f"\n"  # outcome left empty, to be filled after trade closes
            )

        logger.debug(f"📝 Confidence logged to {csv_path}")

    except Exception as e:
        logger.warning(f"Failed to save confidence to CSV: {e}")


def check_cooldown(state: Dict[str, Any], config: Dict[str, Any]) -> bool:
    """
    Check if cooldown period has passed since last signal.

    Args:
        state: Bot state dictionary
        config: Bot configuration

    Returns:
        True if cooldown passed (can trade), False otherwise
    """
    from datetime import datetime

    cooldown = config.get('strategy', {}).get('cooldown_candles', 0)
    if cooldown == 0:
        return True

    if not state.get('last_signal_time'):
        return True

    cooldown_minutes = cooldown * 5  # 5-minute candles
    last_signal = datetime.fromisoformat(state['last_signal_time'])
    elapsed = (datetime.now() - last_signal).total_seconds() / 60

    return elapsed >= cooldown_minutes


def check_daily_loss_limit(state: Dict[str, Any], config: Dict[str, Any]) -> bool:
    """
    Check if daily loss limit has been reached.

    Args:
        state: Bot state dictionary
        config: Bot configuration

    Returns:
        True if daily loss limit reached (should pause trading)
    """
    from .state import reset_daily_stats_if_needed

    reset_daily_stats_if_needed(state)
    max_loss = config.get('risk', {}).get('max_daily_loss_pct', 10.0)
    return state.get('daily_pnl', 0) <= -max_loss


def check_consecutive_loss_limit(state: Dict[str, Any]) -> bool:
    """
    Check if consecutive loss limit has been reached.

    v1.27.0: Pause trading after MAX_CONSECUTIVE_LOSSES consecutive SL hits.
    Resets on next winning trade or daily reset.

    Args:
        state: Bot state dictionary

    Returns:
        True if consecutive loss limit reached (should pause trading)
    """
    from .constants import MAX_CONSECUTIVE_LOSSES
    return state.get('consecutive_losses', 0) >= MAX_CONSECUTIVE_LOSSES


def check_early_exit_signal(
    position: Dict[str, Any],
    df: pd.DataFrame,
    current_price: float,
    config: Dict[str, Any],
) -> Tuple[bool, int, Optional[str], Optional[int]]:
    """
    Check if early exit should trigger based on reversal candle patterns.

    ultra_conservative mode (v1.3):
    - LONG exits on 3 consecutive BD (Big Down) candles
    - SHORT exits on 3 consecutive BU (Big Up) candles
    - Only if unrealized PnL >= 0.3%

    v1.14.1 BugFix: Track last_counted_candle_ts to prevent double-counting
    the same candle when function is called multiple times per candle period.

    Research: EARLY_EXIT_SIGNAL_RESEARCH_20260123.md
    Result: +93% improvement over baseline

    Args:
        position: Position dictionary with direction, entry_price, reversal_count,
                  last_counted_candle_ts
        df: DataFrame with candle classification (must have type_code column)
        current_price: Current market price
        config: Bot configuration

    Returns:
        Tuple of (should_exit, new_reversal_count, exit_reason, last_counted_candle_ts)
    """
    # Check if early exit is enabled
    if not EARLY_EXIT_CONFIG.get('enabled', True):
        return False, position.get('reversal_count', 0), None, position.get('last_counted_candle_ts')

    # Ensure we have classification data
    if 'type_code' not in df.columns or len(df) < 2:
        return False, position.get('reversal_count', 0), None, position.get('last_counted_candle_ts')

    # Ensure we have a valid current price
    if current_price is None or current_price <= 0:
        return False, position.get('reversal_count', 0), None, position.get('last_counted_candle_ts')

    # Get current candle's type code (last completed candle)
    current_candle = df.iloc[-2]
    type_code = current_candle.get('type_code', '')

    # v1.14.1: Get candle timestamp to prevent double-counting
    candle_ts = None
    if 'timestamp' in df.columns:
        candle_ts = int(df.iloc[-2]['timestamp'])
    elif df.index.name == 'timestamp' or isinstance(df.index, pd.DatetimeIndex):
        candle_ts = int(df.index[-2].timestamp() * 1000) if hasattr(df.index[-2], 'timestamp') else None

    direction = position.get('direction', '')
    entry_price = position.get('entry_price', 0)
    leverage = config.get('leverage', 3)
    reversal_count = position.get('reversal_count', 0)
    last_counted_ts = position.get('last_counted_candle_ts')

    # v1.14.1: Skip if we already counted this candle
    if candle_ts is not None and last_counted_ts is not None and candle_ts == last_counted_ts:
        # Same candle as before, don't recount
        return False, reversal_count, None, last_counted_ts

    # Calculate unrealized PnL
    if direction == 'LONG':
        unrealized_pnl = (current_price / entry_price - 1) * 100 * leverage
        reversal_types = EARLY_EXIT_BEARISH_TYPES
        exit_reason_prefix = 'EARLY_BD'
    elif direction == 'SHORT':
        unrealized_pnl = (entry_price / current_price - 1) * 100 * leverage
        reversal_types = EARLY_EXIT_BULLISH_TYPES
        exit_reason_prefix = 'EARLY_BU'
    else:
        return False, reversal_count, None, last_counted_ts

    # Check if current candle is a reversal signal
    if type_code in reversal_types:
        reversal_count += 1
        logger.debug(
            f"🔍 Early exit check: {type_code} detected | "
            f"count={reversal_count}/{EARLY_EXIT_CONFIRM_CANDLES} | "
            f"PnL={unrealized_pnl:+.2f}% (min={EARLY_EXIT_MIN_PROFIT_PCT}%) | "
            f"candle_ts={candle_ts}"
        )

        # Check if conditions are met
        if (reversal_count >= EARLY_EXIT_CONFIRM_CANDLES
                and unrealized_pnl >= EARLY_EXIT_MIN_PROFIT_PCT):
            logger.info(
                f"🚨 Early exit signal! {reversal_count}x {type_code} confirmed | "
                f"PnL={unrealized_pnl:+.2f}% >= {EARLY_EXIT_MIN_PROFIT_PCT}%"
            )
            return True, reversal_count, exit_reason_prefix, candle_ts

    else:
        # Reset counter if non-reversal candle
        if reversal_count > 0:
            logger.debug(f"🔄 Reversal count reset (non-reversal candle: {type_code})")
        reversal_count = 0

    return False, reversal_count, None, candle_ts


def get_candle_type_for_price(df: pd.DataFrame) -> str:
    """
    Get the type code of the last completed candle.

    Args:
        df: DataFrame with candle classification

    Returns:
        Type code string (e.g., 'BD', 'BU', 'U', etc.)
    """
    if 'type_code' not in df.columns or len(df) < 2:
        return 'UNKNOWN'

    return df.iloc[-2].get('type_code', 'UNKNOWN')
