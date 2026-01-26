"""
Pattern 5m Bot - 3-Candle Pattern Trading Strategy

A production trading bot that uses validated 3-candle patterns for entry signals.
Based on walk-forward validated patterns with:
- 9 LONG patterns (best: MU-U-DN)
- 4 SHORT patterns (best: MU-ST-ST)

Backtest Results (104 days, 3x leverage):
- Combined (9 patterns): +219.4% compound
- vs Buy & Hold: +302.0% excess return
"""

from .bot import run_bot
from .signals import check_entry_signal, add_candle_classification
from .indicators import calculate_indicators, classify_candle
from .constants import (
    BOT_NAME,
    BOT_VERSION,
    CandleType,
    VALIDATED_LONG_PATTERNS,
    VALIDATED_SHORT_PATTERNS,
)

__all__ = [
    'run_bot',
    'check_entry_signal',
    'add_candle_classification',
    'calculate_indicators',
    'classify_candle',
    'BOT_NAME',
    'BOT_VERSION',
    'CandleType',
    'VALIDATED_LONG_PATTERNS',
    'VALIDATED_SHORT_PATTERNS',
]
