"""
C1 Breakout v2 — Configuration
"""

import os
import copy
import yaml

DEFAULT_CONFIG = {
    'strategy': {
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
        'backstop_tp_atr': 0,  # Disabled — trail has no cap
    },
    'risk': {},
    'exchange': {
        'symbol': 'BTC-USDT',
        'timeframe': '15m',
        'leverage': 1,
        'trading_leverage': 1,  # Actual position sizing leverage (default = same as leverage)
        'position_mode': 'One-Way',
        'candle_bars_fetch': 100,
    },
    'bot': {
        'poll_interval_seconds': 900,  # 15 minutes
        'warmup_bars': 25,
    },
}


def load_config(path: str = 'config/c1_breakout_config.yaml') -> dict:
    """Load config from YAML, falling back to defaults."""
    config = copy.deepcopy(DEFAULT_CONFIG)

    if os.path.exists(path):
        with open(path, 'r') as f:
            user_config = yaml.safe_load(f) or {}
        # Deep merge
        for section in config:
            if section in user_config and isinstance(user_config[section], dict):
                config[section].update(user_config[section])
    else:
        import logging
        logging.getLogger('c1_breakout').warning(
            f"Config file not found: {path} — using defaults (leverage=1)")

    return config
