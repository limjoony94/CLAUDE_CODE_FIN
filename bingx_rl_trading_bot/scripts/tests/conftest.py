"""Shared fixtures for C1 Breakout test suite.

Permanent storage of critical-evaluation cases discovered during code review.
Each test case encodes one criticism angle:
  A. Edge conditions (null/empty/extreme values)
  B. Backtest parity (live-only changes must not alter backtest semantics)
  C. Bug interaction (new fixes don't break old ones)
  D. Rollback safety (failures degrade gracefully)
"""
import sys
from pathlib import Path

# Ensure bingx_rl_trading_bot/ is on sys.path so tests can import
# scripts.production.c1_breakout.* without a package install.
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pytest
from unittest.mock import MagicMock


@pytest.fixture
def default_strategy_config():
    """Minimal strategy config matching production defaults."""
    return {
        'channel_period': 15,
        'body_min_ratio': 0.4,
        'atr_period': 14,
        'trail_K': 2.5,
        'max_sl_atr': 3.3,
        'emergency_sl_pct': 3.0,
        'max_hold_bars': 192,
        'sl_min_pct': 0.15,
        'sl_max_pct': 3.0,
        'min_bars_between': 2,
        'trail_activation_pct': 0.05,
        'max_positions': 1,
    }


@pytest.fixture
def full_config():
    """Complete config dict suitable for C1BreakoutBot (exchange mocked)."""
    return {
        'strategy': {
            'channel_period': 15, 'body_min_ratio': 0.4, 'atr_period': 14,
            'trail_K': 2.5, 'max_sl_atr': 3.3, 'emergency_sl_pct': 3.0,
            'max_hold_bars': 192, 'sl_min_pct': 0.15, 'sl_max_pct': 3.0,
            'min_bars_between': 2, 'trail_activation_pct': 0.05,
            'max_positions': 1,
        },
        'exchange': {
            'symbol': 'BTC-USDT', 'timeframe': '15m',
            'leverage': 10, 'trading_leverage': 3,
            'position_mode': 'One-Way', 'candle_bars_fetch': 100,
        },
        'bot': {'poll_interval_seconds': 900, 'warmup_bars': 25},
        'risk': {},
    }


@pytest.fixture
def mock_bot(full_config, tmp_path):
    """Bare-bones C1BreakoutBot without __init__ side effects.

    Bypasses _init_exchange / _load_state / _sync_exchange so tests can
    set precise state. Exchange is MagicMock for controlled API responses.
    """
    from scripts.production.c1_breakout.bot import C1BreakoutBot
    b = C1BreakoutBot.__new__(C1BreakoutBot)
    b.config = full_config
    b.state_path = str(tmp_path / 'state.json')
    b.positions = []
    b.trade_history = []
    b.bars_since_last_exit = 999
    b.last_exit_time = None
    b.max_positions = full_config['strategy']['max_positions']
    b.exchange = MagicMock()
    b._force_trail_reset = False
    return b


@pytest.fixture
def sample_candles():
    """18 bars of monotonic uptrend — guarantees ATR/Channel/Fractal valid."""
    return {
        'high': [100 + i for i in range(18)],
        'low': [99 + i for i in range(18)],
        'close': [99.5 + i for i in range(18)],
        'open': [99.5 + i for i in range(18)],
    }
