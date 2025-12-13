#!/usr/bin/env python3
"""
ADX Trend + Supertrend Trail Bot v1.0
=====================================
Date: 2025-12-13
Based on Dynamic Stop-Loss Research

Research Results:
- Full Period Return: +1276.6%
- Max Drawdown: 21.6%
- Win Rate: 70.3%
- Risk-Adjusted Return: 59.06
- Trades: 622 (1.98/day)
- Walk-Forward: 8/8 windows positive (100%)
- Monthly: 11/11 months positive (100%)

Configuration:
- Entry: ADX Trend (+DI/-DI crossover when ADX > 20)
- TP: 2.0% (fixed)
- SL: Supertrend trailing line (dynamic)
- Cooldown: 6 candles (1.5 hours on 15m)
"""

import os
import sys
import json
import time
import hmac
import hashlib
import logging
import psutil
import requests
from datetime import datetime, timedelta
from typing import Dict, Optional, List, Any
from logging.handlers import RotatingFileHandler

import ccxt
import pandas as pd
import numpy as np
import yaml

# =============================================================================
# Project Paths
# =============================================================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CONFIG_DIR = os.path.join(PROJECT_ROOT, 'config')
CONFIG_FILE = os.path.join(CONFIG_DIR, 'adx_supertrend_trail_config.yaml')

# =============================================================================
# Configuration Loading
# =============================================================================
def load_bot_config() -> Dict:
    """Load bot configuration from YAML file"""
    if os.path.exists(CONFIG_FILE):
        with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    return {}

_config = load_bot_config()

# =============================================================================
# Strategy Parameters
# =============================================================================
_strategy = _config.get('strategy', {})
ADX_THRESHOLD = _strategy.get('adx_threshold', 20)
SUPERTREND_PERIOD = _strategy.get('supertrend_period', 14)
SUPERTREND_MULTIPLIER = _strategy.get('supertrend_multiplier', 3.0)

# Exit Parameters
_exit = _config.get('exit', {})
TAKE_PROFIT_PCT = _exit.get('take_profit_pct', 2.0)
COOLDOWN_CANDLES = _exit.get('cooldown_candles', 6)
MAX_HOLD_CANDLES = _exit.get('max_hold_candles', 192)  # 48 hours

# SL Distance Limits (skip signals with extreme SL distances)
MIN_SL_DISTANCE_PCT = _exit.get('min_sl_distance_pct', 0.3)
MAX_SL_DISTANCE_PCT = _exit.get('max_sl_distance_pct', 5.0)

# Leverage & Position
_leverage = _config.get('leverage', {})
EXCHANGE_LEVERAGE = _leverage.get('exchange_leverage', 20)
EFFECTIVE_LEVERAGE = _leverage.get('effective_leverage', 4)
LEVERAGE = EFFECTIVE_LEVERAGE
POSITION_SIZE_PCT = _leverage.get('position_size_pct', 0.95)
POSITION_MODE = _leverage.get('position_mode', 'one-way')

def get_position_side(direction: str) -> str:
    """Get position side based on position mode"""
    if POSITION_MODE == 'hedge':
        return 'LONG' if direction == 'LONG' else 'SHORT'
    else:
        return 'BOTH'

# Trading
_trading = _config.get('trading', {})
TAKER_FEE_RATE = _trading.get('taker_fee_rate', 0.05) / 100
TRADING_FEE_PCT = TAKER_FEE_RATE * 100
SYMBOL = _trading.get('symbol', 'BTC-USDT')
TIMEFRAME = _trading.get('timeframe', '15m')
CHECK_INTERVAL_SECONDS = _trading.get('check_interval_seconds', 60)

# Backup
_backup = _config.get('backup', {})
BACKUP_ENABLED = _backup.get('enabled', True)
MAX_BACKUPS = _backup.get('max_backups', 10)
BACKUP_ON_TRADE = _backup.get('backup_on_trade', True)
BACKUP_INTERVAL_MINUTES = _backup.get('backup_interval_minutes', 30)

# Sync
_sync = _config.get('sync', {})
SYNC_ENABLED = _sync.get('enabled', True)
DEEP_SYNC_INTERVAL_MINUTES = _sync.get('deep_sync_interval_minutes', 5)
QUICK_SYNC_ON_LOOP = _sync.get('quick_sync_on_loop', True)
VERIFY_ORDERS_ON_SYNC = _sync.get('verify_orders_on_sync', True)
AUTO_RECREATE_ORDERS = _sync.get('auto_recreate_orders', True)

# API Retry
_api_retry = _config.get('api_retry', {})
API_MAX_RETRIES = _api_retry.get('max_retries', 3)
API_BASE_DELAY = _api_retry.get('base_delay_seconds', 1.0)
API_MAX_DELAY = _api_retry.get('max_delay_seconds', 30.0)
API_EXPONENTIAL_BASE = _api_retry.get('exponential_base', 2.0)

# Health Check
_health = _config.get('health_check', {})
HEALTH_CHECK_ENABLED = _health.get('enabled', True)
HEALTH_CHECK_INTERVAL = _health.get('interval_minutes', 10) * 60
HEALTH_CHECK_MAX_FAILURES = _health.get('max_consecutive_failures', 5)

# Error Handling
_error = _config.get('error_handling', {})
RECOVERABLE_ERRORS = _error.get('recoverable_errors', [
    'TIMEOUT', 'CONNECTION_RESET', 'TEMPORARY_UNAVAILABLE',
    'RATE_LIMIT_EXCEEDED', 'INTERNAL_SERVER_ERROR'
])
CRITICAL_ERRORS = _error.get('critical_errors', [
    'INVALID_API_KEY', 'INSUFFICIENT_BALANCE', 'POSITION_NOT_FOUND', 'ORDER_REJECTED'
])
ON_CRITICAL_ERROR = _error.get('on_critical_error', 'pause')
PAUSE_DURATION_MINUTES = _error.get('pause_duration_minutes', 5)

# Files
STATE_FILE = os.path.join(PROJECT_ROOT, 'results', 'adx_supertrend_trail_bot_state.json')
BACKUP_DIR = os.path.join(PROJECT_ROOT, 'results', 'backups')
LOCK_FILE = os.path.join(PROJECT_ROOT, 'results', 'adx_supertrend_trail_bot.lock')
LOG_DIR = os.path.join(PROJECT_ROOT, 'logs')

# Create directories
os.makedirs(BACKUP_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# =============================================================================
# API Keys
# =============================================================================
def load_api_keys() -> Dict:
    """Load API keys from config/api_keys.yaml"""
    api_keys_file = os.path.join(PROJECT_ROOT, 'config', 'api_keys.yaml')
    if os.path.exists(api_keys_file):
        with open(api_keys_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            return config.get('bingx', {}).get('mainnet', {})
    return {}

_api_config = load_api_keys()
API_KEY = _api_config.get('api_key', os.getenv('BINGX_API_KEY', ''))
API_SECRET = _api_config.get('secret_key', os.getenv('BINGX_API_SECRET', ''))
USE_TESTNET = False

# =============================================================================
# Error Handling & Health Check
# =============================================================================
_consecutive_failures = 0
_last_health_check_time = 0
_bot_paused = False
_pause_until = 0

class BotError(Exception):
    def __init__(self, message: str, error_type: str = 'UNKNOWN', is_recoverable: bool = True):
        super().__init__(message)
        self.error_type = error_type
        self.is_recoverable = is_recoverable

def classify_error(error: Exception) -> tuple:
    error_str = str(error).upper()
    for critical in CRITICAL_ERRORS:
        if critical in error_str:
            return 'critical', critical
    for recoverable in RECOVERABLE_ERRORS:
        if recoverable in error_str:
            return 'recoverable', recoverable
    return 'unknown', 'UNKNOWN'

def calculate_retry_delay(attempt: int) -> float:
    delay = min(API_BASE_DELAY * (API_EXPONENTIAL_BASE ** attempt), API_MAX_DELAY)
    jitter = delay * 0.1 * (2 * np.random.random() - 1)
    return delay + jitter

def api_retry(func):
    def wrapper(*args, **kwargs):
        last_error = None
        for attempt in range(API_MAX_RETRIES):
            try:
                result = func(*args, **kwargs)
                if attempt > 0:
                    logger.info(f"API call succeeded after {attempt + 1} attempts")
                return result
            except Exception as e:
                last_error = e
                error_class, error_type = classify_error(e)
                if error_class == 'critical':
                    logger.error(f"Critical error in {func.__name__}: {e}")
                    raise BotError(str(e), error_type, is_recoverable=False)
                if attempt < API_MAX_RETRIES - 1:
                    delay = calculate_retry_delay(attempt)
                    logger.warning(f"API call failed (attempt {attempt + 1}/{API_MAX_RETRIES}): {e}. Retrying in {delay:.1f}s")
                    time.sleep(delay)
        raise last_error
    return wrapper

def handle_critical_error(error: Exception):
    global _bot_paused, _pause_until
    logger.error(f"Critical error occurred: {error}")
    if ON_CRITICAL_ERROR == 'pause':
        _bot_paused = True
        _pause_until = time.time() + (PAUSE_DURATION_MINUTES * 60)
        logger.warning(f"Bot paused for {PAUSE_DURATION_MINUTES} minutes")
    elif ON_CRITICAL_ERROR == 'stop':
        logger.error("Bot stopping due to critical error")
        sys.exit(1)

def check_bot_paused() -> bool:
    global _bot_paused, _pause_until
    if _bot_paused:
        if time.time() >= _pause_until:
            _bot_paused = False
            logger.info("Bot resuming after pause")
            return False
        return True
    return False

def run_health_check(exchange) -> bool:
    global _consecutive_failures, _last_health_check_time
    if not HEALTH_CHECK_ENABLED:
        return True
    if time.time() - _last_health_check_time < HEALTH_CHECK_INTERVAL:
        return True
    _last_health_check_time = time.time()
    try:
        checks_passed = 0
        checks_total = 2
        try:
            exchange.fetch_ticker(SYMBOL)
            checks_passed += 1
        except Exception as e:
            logger.warning(f"Health check - API connection failed: {e}")
        try:
            balance = get_account_balance(exchange)
            if balance.get('free', 0) > 0:
                checks_passed += 1
        except Exception as e:
            logger.warning(f"Health check - Balance check failed: {e}")
        if checks_passed == checks_total:
            _consecutive_failures = 0
            logger.info(f"Health check passed ({checks_passed}/{checks_total})")
            return True
        else:
            _consecutive_failures += 1
            logger.warning(f"Health check partial ({checks_passed}/{checks_total})")
            if _consecutive_failures >= HEALTH_CHECK_MAX_FAILURES:
                handle_critical_error(Exception(f"Health check failed {_consecutive_failures} times"))
            return False
    except Exception as e:
        _consecutive_failures += 1
        logger.error(f"Health check error: {e}")
        return False

def reset_failure_counter():
    global _consecutive_failures
    _consecutive_failures = 0

# =============================================================================
# Raw API Functions
# =============================================================================
BINGX_BASE_URL = 'https://open-api.bingx.com'

def _sign_params(params: Dict) -> str:
    query_string = '&'.join([f"{k}={v}" for k, v in sorted(params.items())])
    signature = hmac.new(API_SECRET.encode(), query_string.encode(), hashlib.sha256).hexdigest()
    return f"{query_string}&signature={signature}"

def _raw_api_request(method: str, endpoint: str, params: Dict) -> Dict:
    params['timestamp'] = int(time.time() * 1000)
    signed_params = _sign_params(params)
    headers = {'X-BX-APIKEY': API_KEY}
    url = f"{BINGX_BASE_URL}{endpoint}?{signed_params}"
    if method == 'GET':
        response = requests.get(url, headers=headers, timeout=10)
    else:
        response = requests.post(url, headers=headers, timeout=10)
    return response.json()

def get_open_orders_raw(symbol: str) -> List[Dict]:
    params = {'symbol': symbol.replace('-', '')}
    result = _raw_api_request('GET', '/openApi/swap/v2/trade/openOrders', params)
    if result.get('code') == 0:
        return result.get('data', {}).get('orders', [])
    return []

def cancel_order_raw(symbol: str, order_id: str) -> bool:
    params = {'symbol': symbol.replace('-', ''), 'orderId': order_id}
    result = _raw_api_request('POST', '/openApi/swap/v2/trade/order', params)
    return result.get('code') == 0

# =============================================================================
# Logging Setup
# =============================================================================
LOG_MAX_BYTES = 10 * 1024 * 1024
LOG_BACKUP_COUNT = 5
LOG_LEVEL = getattr(logging, _config.get('logging', {}).get('level', 'INFO'))
LOG_TO_CONSOLE = _config.get('logging', {}).get('log_to_console', True)

logger = logging.getLogger('ADXSupertrendTrailBot')
logger.setLevel(LOG_LEVEL)

log_formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

log_file = os.path.join(LOG_DIR, 'adx_supertrend_trail_bot.log')
file_handler = RotatingFileHandler(log_file, maxBytes=LOG_MAX_BYTES, backupCount=LOG_BACKUP_COUNT)
file_handler.setFormatter(log_formatter)
logger.addHandler(file_handler)

if LOG_TO_CONSOLE:
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(log_formatter)
    logger.addHandler(console_handler)

daily_log_file = os.path.join(LOG_DIR, f'adx_supertrend_trail_bot_{datetime.now().strftime("%Y%m%d")}.log')
daily_handler = logging.FileHandler(daily_log_file)
daily_handler.setFormatter(log_formatter)
logger.addHandler(daily_handler)

# =============================================================================
# State Management
# =============================================================================
_last_backup_time = time.time()

def load_state() -> Dict:
    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, 'r') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading state: {e}")
            return recover_from_backup()
    return {
        'position': None,
        'last_trade_candle': 0,
        'candle_count': 0,
        'total_trades': 0,
        'total_pnl': 0.0,
        'winning_trades': 0,
        'initial_balance': None,
        'created_at': datetime.now().isoformat()
    }

def save_state(state: Dict, create_backup: bool = False):
    global _last_backup_time
    state['updated_at'] = datetime.now().isoformat()
    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2, default=str)
    if BACKUP_ENABLED and create_backup:
        create_backup_file(state)
        _last_backup_time = time.time()

def create_backup_file(state: Dict):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_file = os.path.join(BACKUP_DIR, f'adx_supertrend_trail_state_{timestamp}.json')
    with open(backup_file, 'w') as f:
        json.dump(state, f, indent=2, default=str)
    cleanup_old_backups()

def cleanup_old_backups():
    backups = sorted([f for f in os.listdir(BACKUP_DIR) if f.startswith('adx_supertrend_trail_state_')])
    while len(backups) > MAX_BACKUPS:
        oldest = backups.pop(0)
        os.remove(os.path.join(BACKUP_DIR, oldest))

def recover_from_backup() -> Dict:
    backups = sorted([f for f in os.listdir(BACKUP_DIR) if f.startswith('adx_supertrend_trail_state_')], reverse=True)
    for backup in backups:
        try:
            with open(os.path.join(BACKUP_DIR, backup), 'r') as f:
                state = json.load(f)
            logger.warning(f"Recovered state from backup: {backup}")
            return state
        except Exception as e:
            logger.error(f"Failed to recover from {backup}: {e}")
    return load_state()

def check_periodic_backup(state: Dict):
    global _last_backup_time
    if not BACKUP_ENABLED:
        return
    if time.time() - _last_backup_time >= BACKUP_INTERVAL_MINUTES * 60:
        create_backup_file(state)
        _last_backup_time = time.time()
        logger.info("Periodic backup created")

# =============================================================================
# Duplicate Process Check
# =============================================================================
BOT_SCRIPT_NAME = 'adx_supertrend_trail_bot.py'

def find_duplicate_bot_processes() -> List[psutil.Process]:
    duplicates = []
    current_pid = os.getpid()
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            cmdline = proc.info.get('cmdline', [])
            if cmdline and any(BOT_SCRIPT_NAME in str(c) for c in cmdline):
                if proc.info['pid'] != current_pid:
                    duplicates.append(proc)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return duplicates

def check_and_handle_duplicates():
    duplicates = find_duplicate_bot_processes()
    if duplicates:
        logger.warning(f"Found {len(duplicates)} duplicate bot process(es)")
        for proc in duplicates:
            logger.warning(f"  - PID {proc.pid}")
        logger.info("Auto-killing duplicate processes...")
        for proc in duplicates:
            try:
                proc.terminate()
                logger.info(f"Terminated duplicate process: {proc.pid}")
            except Exception as e:
                logger.error(f"Failed to terminate {proc.pid}: {e}")

# =============================================================================
# Technical Indicators
# =============================================================================
def calculate_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Calculate ADX, +DI, -DI, and Supertrend indicators"""
    df = df.copy()

    # True Range
    tr = pd.DataFrame({
        'hl': df['high'] - df['low'],
        'hc': abs(df['high'] - df['close'].shift(1)),
        'lc': abs(df['low'] - df['close'].shift(1))
    }).max(axis=1)

    # ATR
    df['atr'] = tr.ewm(span=14, adjust=False).mean()

    # +DM, -DM
    plus_dm = df['high'].diff()
    minus_dm = -df['low'].diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

    # +DI, -DI
    atr_14 = tr.ewm(span=14, adjust=False).mean()
    df['plus_di'] = 100 * (plus_dm.ewm(span=14, adjust=False).mean() / atr_14)
    df['minus_di'] = 100 * (minus_dm.ewm(span=14, adjust=False).mean() / atr_14)

    # ADX
    dx = 100 * abs(df['plus_di'] - df['minus_di']) / (df['plus_di'] + df['minus_di'] + 1e-10)
    df['adx'] = dx.ewm(span=14, adjust=False).mean()

    # Supertrend (for trailing SL)
    hl2 = (df['high'] + df['low']) / 2
    upper = hl2 + SUPERTREND_MULTIPLIER * df['atr']
    lower = hl2 - SUPERTREND_MULTIPLIER * df['atr']

    supertrend = np.zeros(len(df))
    direction = np.zeros(len(df))

    for i in range(1, len(df)):
        if df['close'].iloc[i] > upper.iloc[i-1]:
            direction[i] = 1
        elif df['close'].iloc[i] < lower.iloc[i-1]:
            direction[i] = -1
        else:
            direction[i] = direction[i-1]

        if direction[i] == 1:
            supertrend[i] = max(lower.iloc[i], supertrend[i-1] if direction[i-1] == 1 else lower.iloc[i])
        else:
            supertrend[i] = min(upper.iloc[i], supertrend[i-1] if direction[i-1] == -1 else upper.iloc[i])

    df['supertrend'] = supertrend
    df['st_direction'] = direction

    return df

# =============================================================================
# Entry Signal (ADX Trend)
# =============================================================================
def check_entry_signal(df: pd.DataFrame, state: Dict) -> tuple:
    """
    Check for ADX Trend entry signal (+DI/-DI crossover when ADX > threshold)

    Entry Logic:
    - LONG: +DI crosses above -DI when ADX > 20
    - SHORT: -DI crosses above +DI when ADX > 20

    Returns:
        signal: 1 (LONG), -1 (SHORT), 0 (no signal)
        reason: String explaining the signal
        sl_price: Supertrend line price for stop-loss
    """
    if len(df) < 200:
        return 0, "Insufficient data", None

    row = df.iloc[-1]
    prev_row = df.iloc[-2]

    # Skip if indicators not ready
    if pd.isna(row['adx']) or pd.isna(row['plus_di']) or pd.isna(row['supertrend']):
        return 0, "Indicators not ready", None

    # Check cooldown
    candle_count = state.get('candle_count', 0)
    last_trade_candle = state.get('last_trade_candle', 0)
    if last_trade_candle > 0 and candle_count - last_trade_candle < COOLDOWN_CANDLES:
        remaining = COOLDOWN_CANDLES - (candle_count - last_trade_candle)
        return 0, f"Cooldown: {remaining} candles", None

    adx = row['adx']
    plus_di = row['plus_di']
    minus_di = row['minus_di']
    prev_plus_di = prev_row['plus_di']
    prev_minus_di = prev_row['minus_di']
    price = row['close']
    supertrend = row['supertrend']

    # Calculate SL distance
    sl_distance_pct = abs(price - supertrend) / price * 100

    # Skip if SL distance is out of range
    if sl_distance_pct < MIN_SL_DISTANCE_PCT or sl_distance_pct > MAX_SL_DISTANCE_PCT:
        return 0, f"SL distance {sl_distance_pct:.2f}% out of range [{MIN_SL_DISTANCE_PCT}-{MAX_SL_DISTANCE_PCT}%]", None

    # ADX Trend signals
    if adx > ADX_THRESHOLD:
        # +DI crosses above -DI = LONG
        if prev_plus_di < prev_minus_di and plus_di > minus_di:
            reason = f"LONG: +DI crosses above -DI (ADX={adx:.1f}), Price={price:.1f}, ST_SL={supertrend:.1f} ({sl_distance_pct:.2f}%)"
            return 1, reason, supertrend

        # -DI crosses above +DI = SHORT
        elif prev_minus_di < prev_plus_di and minus_di > plus_di:
            reason = f"SHORT: -DI crosses above +DI (ADX={adx:.1f}), Price={price:.1f}, ST_SL={supertrend:.1f} ({sl_distance_pct:.2f}%)"
            return -1, reason, supertrend

    return 0, f"No signal: ADX={adx:.1f}, +DI={plus_di:.1f}, -DI={minus_di:.1f}, Price={price:.1f}", None

# =============================================================================
# Exchange Operations
# =============================================================================
@api_retry
def fetch_candles(exchange, symbol: str, timeframe: str, limit: int = 500) -> pd.DataFrame:
    ohlcv = exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    return df

@api_retry
def get_account_balance(exchange) -> Dict:
    balance = exchange.fetch_balance()
    usdt_balance = balance.get('USDT', {})
    return {
        'total': usdt_balance.get('total', 0),
        'free': usdt_balance.get('free', 0),
        'used': usdt_balance.get('used', 0)
    }

@api_retry
def get_current_position(exchange, symbol: str) -> Optional[Dict]:
    positions = exchange.fetch_positions([symbol])
    for pos in positions:
        if pos['symbol'] == symbol and abs(pos['contracts']) > 0:
            return {
                'side': pos['side'],
                'size': abs(pos['contracts']),
                'entry_price': pos['entryPrice'],
                'unrealized_pnl': pos.get('unrealizedPnl', 0),
                'mark_price': pos.get('markPrice', 0)
            }
    return None

@api_retry
def open_position(exchange, state: Dict, direction: str, balance: Dict, entry_price: float, sl_price: float) -> bool:
    """Open a new position with Supertrend trailing stop"""
    try:
        available = balance['free']
        position_value = available * POSITION_SIZE_PCT * EFFECTIVE_LEVERAGE
        quantity = position_value / entry_price
        quantity = round(quantity, 3)

        if quantity < 0.001:
            logger.warning(f"Position size too small: {quantity}")
            return False

        side = 'buy' if direction == 'LONG' else 'sell'
        position_side = get_position_side(direction)

        # Place market order
        params = {'positionSide': position_side}
        order = exchange.create_market_order(SYMBOL, side, quantity, params=params)

        logger.info(f"Opened {direction} position: {quantity} @ ~{entry_price:.1f}")

        # Calculate TP price (fixed percentage)
        if direction == 'LONG':
            tp_price = round(entry_price * (1 + TAKE_PROFIT_PCT / 100), 1)
        else:
            tp_price = round(entry_price * (1 - TAKE_PROFIT_PCT / 100), 1)

        # Place TP order
        try:
            tp_side = 'sell' if direction == 'LONG' else 'buy'
            tp_params = {'positionSide': position_side, 'stopPrice': tp_price}
            exchange.create_order(SYMBOL, 'TAKE_PROFIT_MARKET', tp_side, quantity, None, tp_params)
            logger.info(f"TP order placed @ {tp_price:.1f}")
        except Exception as e:
            logger.error(f"Failed to place TP order: {e}")

        # Place initial SL order at Supertrend line
        sl_price = round(sl_price, 1)
        try:
            sl_side = 'sell' if direction == 'LONG' else 'buy'
            sl_params = {'positionSide': position_side, 'stopPrice': sl_price}
            exchange.create_order(SYMBOL, 'STOP_MARKET', sl_side, quantity, None, sl_params)
            logger.info(f"Initial SL order placed @ {sl_price:.1f} (Supertrend)")
        except Exception as e:
            logger.error(f"Failed to place SL order: {e}")

        # Update state
        state['position'] = {
            'direction': direction,
            'entry_price': entry_price,
            'quantity': quantity,
            'entry_time': datetime.now().isoformat(),
            'entry_candle': state.get('candle_count', 0),
            'tp_price': tp_price,
            'sl_price': sl_price,
            'initial_sl_price': sl_price  # Remember initial SL for reference
        }
        state['last_trade_candle'] = state.get('candle_count', 0)

        if BACKUP_ON_TRADE:
            save_state(state, create_backup=True)
        else:
            save_state(state)

        return True

    except Exception as e:
        logger.error(f"Failed to open position: {e}")
        return False

@api_retry
def close_position(exchange, state: Dict, reason: str, current_price: float) -> bool:
    """Close current position"""
    try:
        position = state.get('position')
        if not position:
            return False

        direction = position['direction']
        quantity = position['quantity']
        entry_price = position['entry_price']

        # Cancel existing orders
        try:
            open_orders = get_open_orders_raw(SYMBOL)
            for order in open_orders:
                cancel_order_raw(SYMBOL, order['orderId'])
        except Exception as e:
            logger.warning(f"Error canceling orders: {e}")

        # Close position
        close_side = 'sell' if direction == 'LONG' else 'buy'
        position_side = get_position_side(direction)

        params = {'positionSide': position_side}
        order = exchange.create_market_order(SYMBOL, close_side, quantity, params=params)

        # Calculate PnL
        if direction == 'LONG':
            pnl_pct = (current_price - entry_price) / entry_price * 100
        else:
            pnl_pct = (entry_price - current_price) / entry_price * 100

        pnl_with_leverage = pnl_pct * LEVERAGE
        fees = TRADING_FEE_PCT * 2 * LEVERAGE
        net_pnl = pnl_with_leverage - fees

        # Update statistics
        state['total_trades'] = state.get('total_trades', 0) + 1
        state['total_pnl'] = state.get('total_pnl', 0) + net_pnl
        if net_pnl > 0:
            state['winning_trades'] = state.get('winning_trades', 0) + 1

        win_rate = state['winning_trades'] / state['total_trades'] * 100 if state['total_trades'] > 0 else 0

        logger.info(f"Closed {direction}: {reason}, PnL: {net_pnl:+.2f}% | Total: {state['total_trades']} trades, WR: {win_rate:.1f}%, Cumulative: {state['total_pnl']:+.2f}%")

        state['position'] = None

        if BACKUP_ON_TRADE:
            save_state(state, create_backup=True)
        else:
            save_state(state)

        return True

    except Exception as e:
        logger.error(f"Failed to close position: {e}")
        return False

def update_trailing_sl(exchange, state: Dict, df: pd.DataFrame) -> bool:
    """Update trailing stop-loss based on Supertrend line"""
    position = state.get('position')
    if not position:
        return False

    direction = position['direction']
    current_sl = position.get('sl_price', 0)
    current_supertrend = df['supertrend'].iloc[-1]

    # Only trail in favorable direction
    new_sl = None
    if direction == 'LONG':
        # For LONG, only trail up
        if current_supertrend > current_sl:
            new_sl = round(current_supertrend, 1)
    else:
        # For SHORT, only trail down
        if current_supertrend < current_sl:
            new_sl = round(current_supertrend, 1)

    if new_sl and new_sl != current_sl:
        try:
            quantity = position['quantity']
            position_side = get_position_side(direction)

            # Cancel existing SL order
            open_orders = get_open_orders_raw(SYMBOL)
            for order in open_orders:
                if 'STOP' in str(order.get('type', '')) and 'TAKE_PROFIT' not in str(order.get('type', '')):
                    cancel_order_raw(SYMBOL, order['orderId'])

            # Place new SL order
            sl_side = 'sell' if direction == 'LONG' else 'buy'
            sl_params = {'positionSide': position_side, 'stopPrice': new_sl}
            exchange.create_order(SYMBOL, 'STOP_MARKET', sl_side, quantity, None, sl_params)

            position['sl_price'] = new_sl
            save_state(state)
            logger.info(f"Trailing SL updated: {current_sl:.1f} → {new_sl:.1f}")
            return True

        except Exception as e:
            logger.error(f"Failed to update trailing SL: {e}")

    return False

def check_position_exit(exchange, state: Dict, current_price: float) -> bool:
    """Check if position should be exited"""
    position = state.get('position')
    if not position:
        return False

    direction = position['direction']
    entry_price = position['entry_price']
    sl_price = position.get('sl_price', 0)
    tp_price = position.get('tp_price', 0)

    # Calculate current PnL
    if direction == 'LONG':
        pnl_pct = (current_price - entry_price) / entry_price * 100
        hit_tp = current_price >= tp_price
        hit_sl = current_price <= sl_price
    else:
        pnl_pct = (entry_price - current_price) / entry_price * 100
        hit_tp = current_price <= tp_price
        hit_sl = current_price >= sl_price

    # Check TP
    if hit_tp:
        return close_position(exchange, state, f"TP hit ({pnl_pct:.2f}%)", current_price)

    # Check SL (Supertrend trail)
    if hit_sl:
        return close_position(exchange, state, f"Supertrend SL hit ({pnl_pct:.2f}%)", current_price)

    # Check max hold
    candle_count = state.get('candle_count', 0)
    entry_candle = position.get('entry_candle', 0)
    hold_candles = candle_count - entry_candle

    if hold_candles >= MAX_HOLD_CANDLES:
        return close_position(exchange, state, f"Max hold ({hold_candles} candles)", current_price)

    return False

# =============================================================================
# Sync Functions
# =============================================================================
def quick_sync_with_exchange(exchange, state: Dict) -> Dict:
    if not QUICK_SYNC_ON_LOOP:
        return state
    try:
        exchange_pos = get_current_position(exchange, SYMBOL)
        state_pos = state.get('position')
        if exchange_pos and not state_pos:
            logger.warning("Found orphaned exchange position, syncing...")
            state['position'] = {
                'direction': 'LONG' if exchange_pos['side'] == 'long' else 'SHORT',
                'entry_price': exchange_pos['entry_price'],
                'quantity': exchange_pos['size'],
                'entry_time': datetime.now().isoformat(),
                'entry_candle': state.get('candle_count', 0),
                'synced_from_exchange': True
            }
            save_state(state)
        elif state_pos and not exchange_pos:
            logger.warning("State position not found on exchange, clearing...")
            state['position'] = None
            save_state(state)
        return state
    except Exception as e:
        logger.error(f"Quick sync error: {e}")
        return state

_last_deep_sync_time = 0

def deep_sync_with_exchange(exchange, state: Dict) -> Dict:
    global _last_deep_sync_time
    if not SYNC_ENABLED:
        return state
    if time.time() - _last_deep_sync_time < DEEP_SYNC_INTERVAL_MINUTES * 60:
        return state
    _last_deep_sync_time = time.time()
    logger.info("Running deep sync with exchange...")
    try:
        state = quick_sync_with_exchange(exchange, state)
        logger.info("Deep sync completed")
        return state
    except Exception as e:
        logger.error(f"Deep sync error: {e}")
        return state

# =============================================================================
# Main Bot Loop
# =============================================================================
def run_bot():
    """Main bot loop"""
    logger.info("=" * 60)
    logger.info("ADX Trend + Supertrend Trail Bot v1.0 Starting")
    logger.info("=" * 60)
    logger.info(f"Strategy: ADX Trend (threshold={ADX_THRESHOLD})")
    logger.info(f"Entry: +DI/-DI crossover when ADX > {ADX_THRESHOLD}")
    logger.info(f"Exit: TP={TAKE_PROFIT_PCT}%, SL=Supertrend Trail")
    logger.info(f"Leverage: {EFFECTIVE_LEVERAGE}x (Exchange: {EXCHANGE_LEVERAGE}x)")
    logger.info("=" * 60)

    check_and_handle_duplicates()

    exchange = ccxt.bingx({
        'apiKey': API_KEY,
        'secret': API_SECRET,
        'enableRateLimit': True,
        'options': {
            'defaultType': 'swap',
            'adjustForTimeDifference': True
        }
    })

    if USE_TESTNET:
        exchange.set_sandbox_mode(True)
        logger.info("Using TESTNET")

    try:
        exchange.set_leverage(EXCHANGE_LEVERAGE, SYMBOL)
        logger.info(f"Leverage set to {EXCHANGE_LEVERAGE}x")
    except Exception as e:
        logger.warning(f"Could not set leverage: {e}")

    state = load_state()
    logger.info(f"State loaded: {state.get('total_trades', 0)} trades, PnL: {state.get('total_pnl', 0):+.2f}%")

    if state.get('initial_balance') is None:
        balance = get_account_balance(exchange)
        state['initial_balance'] = balance['total']
        save_state(state)
        logger.info(f"Initial balance: ${balance['total']:.2f}")

    logger.info("Bot started successfully")

    while True:
        try:
            if check_bot_paused():
                time.sleep(10)
                continue

            if not run_health_check(exchange):
                time.sleep(30)
                continue

            df = fetch_candles(exchange, SYMBOL, TIMEFRAME, limit=500)
            df = calculate_indicators(df)

            current_price = df['close'].iloc[-1]
            state['candle_count'] = state.get('candle_count', 0) + 1

            state = quick_sync_with_exchange(exchange, state)
            state = deep_sync_with_exchange(exchange, state)

            # If in position: update trailing SL and check exit
            if state.get('position'):
                update_trailing_sl(exchange, state, df)
                check_position_exit(exchange, state, current_price)

            # If not in position: check for entry
            if not state.get('position'):
                signal, reason, sl_price = check_entry_signal(df, state)

                if signal != 0:
                    logger.info(f"Signal: {reason}")
                    balance = get_account_balance(exchange)
                    direction = 'LONG' if signal == 1 else 'SHORT'

                    if open_position(exchange, state, direction, balance, current_price, sl_price):
                        logger.info(f"Position opened: {direction}")
                else:
                    logger.info(f"Checking: {reason}")

            check_periodic_backup(state)
            reset_failure_counter()
            time.sleep(CHECK_INTERVAL_SECONDS)

        except BotError as e:
            if not e.is_recoverable:
                handle_critical_error(e)
            else:
                logger.error(f"Recoverable error: {e}")
                time.sleep(30)
        except KeyboardInterrupt:
            logger.info("Bot stopped by user")
            save_state(state, create_backup=True)
            break
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            time.sleep(60)


if __name__ == "__main__":
    run_bot()
