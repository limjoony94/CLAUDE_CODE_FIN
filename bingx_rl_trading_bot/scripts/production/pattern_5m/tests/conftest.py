"""Pytest configuration for pattern_5m tests.

Sets up sys.path to allow imports from bingx_rl_trading_bot package.
"""
import sys
from pathlib import Path

# Add CLAUDE_CODE_FIN to sys.path to enable:
# from bingx_rl_trading_bot.scripts.production.pattern_5m import ...
#
# Path resolution:
# __file__ = .../CLAUDE_CODE_FIN/bingx_rl_trading_bot/scripts/production/pattern_5m/tests/conftest.py
# .parents[4] = bingx_rl_trading_bot/
# .parents[5] = CLAUDE_CODE_FIN/
PROJECT_ROOT = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(PROJECT_ROOT))
