"""
Pattern 5m Bot - 3-Candle Pattern Trading Strategy

A production trading bot that uses validated 3-candle patterns for BTC-USDT entry signals.
Dynamic pattern selection via scanner CLI with per-pattern TP/SL grid search.
Quality filter: Edge>=18pp + WR>=60% + SL>=1.0% + MC<0.01 + min_trades>=25 + Holdout 7d.
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
