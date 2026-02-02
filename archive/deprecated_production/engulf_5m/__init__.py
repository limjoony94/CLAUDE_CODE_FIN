"""
Engulf 5m Bot - Package Initialization
Modular trading bot for engulfing pattern detection.
"""

from .constants import BOT_NAME, BOT_VERSION
from .bot import run_bot

__all__ = [
    'BOT_NAME',
    'BOT_VERSION',
    'run_bot',
]

__version__ = BOT_VERSION
