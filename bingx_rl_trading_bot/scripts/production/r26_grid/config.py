"""R26 Grid Bot — Configuration loader."""
import os
import copy
import yaml

DEFAULT_CONFIG = {
    'strategy': {
        'grid_spacing_pct': 0.30,
        'grid_levels_each_side': 5,
        'trend_exit_distance_pct': 1.5,
        'max_grid_lifetime_bars': 168,
        'atr_period': 20,
        'atr_pct_median_lookback_bars': 720,
    },
    'risk': {
        'auto_size_from_balance': False,
        'balance_utilization_pct': 100,
        'per_level_notional_usd': 150,
        'halt_daily_loss_pct': 3.0,
        'halt_30d_loss_pct': 10.0,
        'halt_consecutive_api_errors': 10,
        'halt_emergency_adverse_pct': 9.0,
    },
    'exchange': {
        'symbol': 'BTC-USDT',
        'timeframe': '1h',
        'exchange_leverage': 10,
        'trading_leverage': 10,
        'position_mode': 'One-Way',
        'candle_bars_fetch': 800,
        'hedge_mode_check': True,
    },
    'bot': {
        'poll_interval_seconds': 300,
        'warmup_bars': 720,
    },
    'logging': {
        'log_path': 'logs/r26_grid.log',
        'state_path': 'results/r26_grid_state.json',
        'log_rotate_daily': True,
        'log_retention_days': 30,
    },
    'api_keys_path': 'config/api_keys.yaml',
}


def load_config(path: str = 'config/r26_grid_config.yaml') -> dict:
    """Load YAML config with deep-merge over defaults + validation."""
    config = copy.deepcopy(DEFAULT_CONFIG)

    if os.path.exists(path):
        with open(path, 'r') as f:
            user_config = yaml.safe_load(f) or {}
        for section in config:
            if isinstance(config[section], dict) and section in user_config:
                if isinstance(user_config[section], dict):
                    config[section].update(user_config[section])
            elif section in user_config:
                config[section] = user_config[section]
    else:
        import logging
        logging.getLogger('r26_grid').warning(
            f"Config file not found: {path} — using defaults")

    # Validation
    s = config['strategy']
    e = config['exchange']
    r = config['risk']

    if s['grid_spacing_pct'] <= 0 or s['grid_spacing_pct'] > 5:
        raise ValueError(f"grid_spacing_pct {s['grid_spacing_pct']}% out of range (0, 5]")
    if s['grid_levels_each_side'] < 1 or s['grid_levels_each_side'] > 20:
        raise ValueError(f"grid_levels_each_side {s['grid_levels_each_side']} out of range [1, 20]")
    if s['trend_exit_distance_pct'] < s['grid_spacing_pct'] * s['grid_levels_each_side']:
        # Trend exit should be at least grid extent (R26 BT uses equal: 1.5% = 0.30% × 5)
        raise ValueError(
            f"trend_exit_distance ({s['trend_exit_distance_pct']}%) must be >= "
            f"grid extent ({s['grid_spacing_pct'] * s['grid_levels_each_side']}%)")

    ex_lev = e.get('exchange_leverage', 10)
    tr_lev = e.get('trading_leverage', ex_lev)
    if tr_lev > ex_lev:
        raise ValueError(
            f"Config error: trading_leverage ({tr_lev}x) must not exceed "
            f"exchange_leverage ({ex_lev}x)")
    if tr_lev <= 0 or ex_lev <= 0:
        raise ValueError(f"Leverage must be positive (ex={ex_lev}, tr={tr_lev})")

    if r['halt_daily_loss_pct'] <= 0 or r['halt_daily_loss_pct'] > 50:
        raise ValueError(f"halt_daily_loss_pct {r['halt_daily_loss_pct']}% out of sane range")
    if not r.get('auto_size_from_balance', False):
        if r['per_level_notional_usd'] <= 0:
            raise ValueError(f"per_level_notional_usd must be positive: {r['per_level_notional_usd']}")
    util = r.get('balance_utilization_pct', 100)
    if util <= 0 or util > 100:
        raise ValueError(f"balance_utilization_pct {util}% out of range (0, 100]")

    return config


def load_api_keys(path: str = 'config/api_keys.yaml') -> dict:
    """Load BingX API keys (matches C1 pattern).

    Supports two formats:
      A: {'bingx': {'api_key': ..., 'secret': ...}}
      B: {'bingx': {'mainnet': {'api_key': ..., 'secret_key': ...}}}  (C1 format)

    Returns: {'api_key': ..., 'secret': ...}
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"API keys file not found: {path}")
    with open(path, 'r') as f:
        keys = yaml.safe_load(f) or {}
    bk = keys.get('bingx', keys)
    # If nested mainnet/testnet, prefer mainnet
    if isinstance(bk, dict) and 'mainnet' in bk:
        bk = bk['mainnet']
    api_key = bk.get('api_key', '')
    secret = bk.get('secret_key', bk.get('secret', ''))
    if not api_key or not secret:
        raise ValueError(f"API keys missing api_key/secret in {path}")
    return {'api_key': api_key, 'secret': secret}
