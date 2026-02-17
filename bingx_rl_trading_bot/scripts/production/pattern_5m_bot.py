#!/usr/bin/env python3
"""
Pattern 5m Bot - Entry Point

3-candle pattern trading strategy with dynamic pattern selection.
See CLAUDE.md for current version, pattern list, and configuration.

Usage:
  python pattern_5m_bot.py
"""

import sys
import os

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)
))))

from scripts.production.pattern_5m.bot import run_bot


if __name__ == '__main__':
    run_bot()
