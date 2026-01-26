#!/usr/bin/env python3
"""
Pattern 5m Bot - Entry Point

3-candle pattern trading strategy using validated patterns.

Validated Patterns:
  LONG: MU-U-DN, DN-MD-BD, DF-U-U, BU-ST-ST, MU-DN-MU
  SHORT: MU-ST-ST, U-MU-ST, IH-DN-DN, D-ST-U

Backtest Results (104 days, 3x leverage):
  - Combined: +219.4% compound
  - vs Buy & Hold: +302.0% excess return

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
