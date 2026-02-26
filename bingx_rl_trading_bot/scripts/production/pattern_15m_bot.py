#!/usr/bin/env python3
"""
Pattern 15m Bot - Entry Point

15-minute candle pattern trading strategy (mixed strategy complement to 5m bot).
Uses the same pattern_5m module with PATTERN_BOT_TF=15m environment override.

See CLAUDE.md for current version, pattern list, and configuration.

Usage:
  python pattern_15m_bot.py

  # Or with explicit env var:
  PATTERN_BOT_TF=15m python pattern_15m_bot.py
"""

import os
import sys

# Set timeframe BEFORE any imports from the pattern module
# This controls all paths, timing constants, and bot identification
os.environ['PATTERN_BOT_TF'] = '15m'

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)
))))

from scripts.production.pattern_5m.bot import run_bot


if __name__ == '__main__':
    run_bot()
