"""
Engulf 5m Bot - Signal Detection
Entry signal detection and validation.
"""

import logging
import pandas as pd
from typing import Dict, Any, Optional, Tuple

from .utils.logging_config import log_signal_conditions

logger = logging.getLogger('engulf_5m')


def check_entry_signal(
    df: pd.DataFrame,
    state: Dict[str, Any],
    config: Dict[str, Any]
) -> Tuple[Optional[str], Optional[str]]:
    """
    Check for entry signals on the last completed candle.

    Args:
        df: DataFrame with calculated indicators
        state: Bot state dictionary
        config: Bot configuration

    Returns:
        Tuple of (signal, reason) or (None, None) if no signal
    """
    current = df.iloc[-2]  # Last complete candle
    current_timestamp = df.iloc[-2]['timestamp']

    # Prevent duplicate signals on same candle
    last_signal_candle = state.get('last_signal_candle_timestamp')
    if last_signal_candle and last_signal_candle == current_timestamp:
        return None, None

    # Debug logging
    if config.get('debug_mode', False):
        log_signal_conditions(logger, df, config)

    signal = None
    reason = None

    if current['bullish_engulf']:
        signal = 'LONG'
        body_pct = current['body_pct']
        reason = f"Bullish Engulfing (body={body_pct:.2f}%)"
    elif current['bearish_engulf']:
        signal = 'SHORT'
        body_pct = current['body_pct']
        reason = f"Bearish Engulfing (body={body_pct:.2f}%)"

    if signal:
        state['last_signal_candle_timestamp'] = current_timestamp
        _log_signal_details(df, current, current_timestamp)

    return signal, reason


def _log_signal_details(
    df: pd.DataFrame,
    current: pd.Series,
    timestamp: int
) -> None:
    """Log detailed signal OHLCV data for debugging."""
    prev = df.iloc[-3]  # Previous candle (for engulfing comparison)

    logger.info(
        f"📊 Signal OHLCV - Current: O={current['open']:.2f} "
        f"H={current['high']:.2f} L={current['low']:.2f} "
        f"C={current['close']:.2f} ts={timestamp}"
    )
    logger.info(
        f"📊 Signal OHLCV - Prev: O={prev['open']:.2f} "
        f"H={prev['high']:.2f} L={prev['low']:.2f} "
        f"C={prev['close']:.2f}"
    )
    logger.info(
        f"📊 Conditions - body_pct={current['body_pct']:.3f}%, "
        f"prev_body_pct={abs(current['prev_body'])/prev['open']*100:.3f}%, "
        f"engulf={current['close']:.2f} vs prev_open={prev['open']:.2f}"
    )


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

    cooldown = config['strategy'].get('cooldown_candles', 0)
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
    max_loss = config['risk']['max_daily_loss_pct']
    return state.get('daily_pnl', 0) <= -max_loss
