"""
RSI Trend Filter Bot v1.0
Strategy: RSI(14) 40/60 with EMA100 trend filter
Validated: Walk-Forward 6/7 (86%), PnL +120.8%, Sharpe 1.31, p=0.013

Entry:
  LONG:  Close > EMA(100) AND RSI(14) crosses above 40
  SHORT: Close < EMA(100) AND RSI(14) crosses below 60

Exit:
  TP: 3.0%
  SL: 2.0%
  Leverage: 4x
"""

import ccxt
import pandas as pd
import numpy as np
import yaml
import json
import time
import logging
import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

# ============================================================
# CONFIGURATION
# ============================================================

BOT_NAME = "rsi_trend_filter_bot"
BOT_VERSION = "1.0"
CONFIG_FILE = "config/rsi_trend_filter_config.yaml"
STATE_FILE = "results/rsi_trend_filter_bot_state.json"
LOG_DIR = "logs"

# Default configuration (overridden by config file)
DEFAULT_CONFIG = {
    'symbol': 'BTC-USDT',
    'timeframe': '15m',
    'leverage': 4,
    'position_size_pct': 95,  # % of available margin
    'strategy': {
        'rsi_period': 14,
        'rsi_long_threshold': 40,
        'rsi_short_threshold': 60,
        'ema_period': 100,
        'tp_pct': 3.0,
        'sl_pct': 2.0,
        'cooldown_candles': 4,
    },
    'risk': {
        'max_daily_loss_pct': 15,
        'max_position_size_usd': 10000,
    },
    'api': {
        'retry_attempts': 3,
        'retry_delay': 2,
    }
}

# ============================================================
# LOGGING SETUP
# ============================================================

def setup_logging():
    """Setup logging configuration"""
    os.makedirs(LOG_DIR, exist_ok=True)
    log_file = os.path.join(LOG_DIR, f"{BOT_NAME}_{datetime.now().strftime('%Y%m%d')}.log")

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s | %(levelname)s | %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

logger = setup_logging()

# ============================================================
# CONFIGURATION LOADING
# ============================================================

def load_config():
    """Load configuration from YAML file"""
    config = DEFAULT_CONFIG.copy()

    if os.path.exists(CONFIG_FILE):
        try:
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                file_config = yaml.safe_load(f)
                if file_config:
                    # Deep merge
                    for key, value in file_config.items():
                        if isinstance(value, dict) and key in config:
                            config[key].update(value)
                        else:
                            config[key] = value
            logger.info(f"Config loaded from {CONFIG_FILE}")
        except Exception as e:
            logger.warning(f"Failed to load config: {e}, using defaults")
    else:
        logger.info("Config file not found, using defaults")

    return config

# ============================================================
# EXCHANGE SETUP
# ============================================================

def create_exchange():
    """Create and configure exchange connection"""
    with open('config/api_keys.yaml', 'r') as f:
        api_config = yaml.safe_load(f)

    mainnet_config = api_config.get('bingx', {}).get('mainnet', {})

    exchange = ccxt.bingx({
        'apiKey': mainnet_config.get('api_key'),
        'secret': mainnet_config.get('secret_key'),
        'enableRateLimit': True,
        'options': {
            'defaultType': 'swap',
            'adjustForTimeDifference': True,
        }
    })

    return exchange

# ============================================================
# STATE MANAGEMENT
# ============================================================

def load_state():
    """Load bot state from file"""
    default_state = {
        'position': None,
        'last_signal_time': None,
        'daily_pnl': 0,
        'daily_trades': 0,
        'total_trades': 0,
        'total_pnl': 0,
        'winning_trades': 0,
        'last_rsi': None,
        'created_at': datetime.now().isoformat(),
        'updated_at': datetime.now().isoformat(),
    }

    if os.path.exists(STATE_FILE):
        try:
            with open(STATE_FILE, 'r') as f:
                state = json.load(f)
                # Reset daily stats if new day
                if state.get('last_trade_date') != datetime.now().strftime('%Y-%m-%d'):
                    state['daily_pnl'] = 0
                    state['daily_trades'] = 0
                    state['last_trade_date'] = datetime.now().strftime('%Y-%m-%d')
                return state
        except Exception as e:
            logger.error(f"Failed to load state: {e}")

    return default_state

def save_state(state):
    """Save bot state to file"""
    state['updated_at'] = datetime.now().isoformat()
    state['last_trade_date'] = datetime.now().strftime('%Y-%m-%d')

    os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)

    # Backup existing state
    if os.path.exists(STATE_FILE):
        backup_file = f"{STATE_FILE}.backup"
        try:
            with open(STATE_FILE, 'r') as f:
                with open(backup_file, 'w') as bf:
                    bf.write(f.read())
        except:
            pass

    with open(STATE_FILE, 'w') as f:
        json.dump(state, f, indent=2, default=str)

# ============================================================
# INDICATOR CALCULATIONS
# ============================================================

def calculate_rsi(df, period=14):
    """Calculate RSI indicator"""
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def calculate_ema(df, period=100):
    """Calculate EMA indicator"""
    return df['close'].ewm(span=period, adjust=False).mean()

def calculate_indicators(df, config):
    """Calculate all required indicators"""
    strategy = config['strategy']

    df['rsi'] = calculate_rsi(df, strategy['rsi_period'])
    df['ema'] = calculate_ema(df, strategy['ema_period'])

    return df

# ============================================================
# SIGNAL GENERATION
# ============================================================

def check_entry_signal(df, state, config):
    """
    Check for entry signals

    LONG:  Close > EMA(100) AND RSI crosses above 40
    SHORT: Close < EMA(100) AND RSI crosses below 60
    """
    strategy = config['strategy']

    current = df.iloc[-1]
    previous = df.iloc[-2]

    close = current['close']
    ema = current['ema']
    rsi = current['rsi']
    rsi_prev = previous['rsi']

    # Store RSI for monitoring
    state['last_rsi'] = float(rsi)

    signal = None
    reason = None

    # Check trend direction
    above_ema = close > ema

    # LONG signal: Above EMA + RSI crosses above threshold
    if above_ema:
        if rsi > strategy['rsi_long_threshold'] and rsi_prev <= strategy['rsi_long_threshold']:
            signal = 'LONG'
            reason = f"RSI crossed above {strategy['rsi_long_threshold']} (RSI={rsi:.1f}), above EMA{strategy['ema_period']}"

    # SHORT signal: Below EMA + RSI crosses below threshold
    else:
        if rsi < strategy['rsi_short_threshold'] and rsi_prev >= strategy['rsi_short_threshold']:
            signal = 'SHORT'
            reason = f"RSI crossed below {strategy['rsi_short_threshold']} (RSI={rsi:.1f}), below EMA{strategy['ema_period']}"

    return signal, reason

# ============================================================
# ORDER EXECUTION
# ============================================================

def get_position_size(exchange, config):
    """Calculate position size based on available margin"""
    try:
        balance = exchange.fetch_balance()
        available = float(balance.get('USDT', {}).get('free', 0))

        # Calculate position size
        size_pct = config['position_size_pct'] / 100
        max_size = config['risk']['max_position_size_usd']

        position_value = min(available * size_pct, max_size)

        # Get current price
        ticker = exchange.fetch_ticker(config['symbol'])
        price = ticker['last']

        # Calculate quantity (with leverage)
        quantity = (position_value * config['leverage']) / price

        # Round to appropriate precision
        quantity = round(quantity, 3)

        return quantity, available

    except Exception as e:
        logger.error(f"Failed to calculate position size: {e}")
        return None, None

def open_position(exchange, state, config, signal, reason):
    """Open a new position"""
    symbol = config['symbol']
    leverage = config['leverage']
    strategy = config['strategy']

    try:
        # Set leverage
        try:
            exchange.set_leverage(leverage, symbol)
        except Exception as e:
            logger.warning(f"Set leverage warning: {e}")

        # Get position size
        quantity, available = get_position_size(exchange, config)
        if quantity is None or quantity <= 0:
            logger.warning("Invalid position size, skipping")
            return False

        # Get current price for TP/SL calculation
        ticker = exchange.fetch_ticker(symbol)
        entry_price = ticker['last']

        # Calculate TP/SL prices
        direction = 1 if signal == 'LONG' else -1
        tp_price = round(entry_price * (1 + direction * strategy['tp_pct'] / 100), 1)
        sl_price = round(entry_price * (1 - direction * strategy['sl_pct'] / 100), 1)

        # Place market order
        side = 'buy' if signal == 'LONG' else 'sell'

        logger.info(f"Opening {signal} position: {quantity} {symbol} @ ~${entry_price:.1f}")
        logger.info(f"TP: ${tp_price:.1f} ({strategy['tp_pct']}%) | SL: ${sl_price:.1f} ({strategy['sl_pct']}%)")

        order = exchange.create_market_order(
            symbol=symbol,
            side=side,
            amount=quantity,
            params={'positionSide': 'LONG' if signal == 'LONG' else 'SHORT'}
        )

        # Update state
        state['position'] = {
            'direction': signal,
            'entry_price': entry_price,
            'quantity': quantity,
            'tp_price': tp_price,
            'sl_price': sl_price,
            'entry_time': datetime.now().isoformat(),
            'reason': reason,
            'order_id': order.get('id'),
        }
        state['last_signal_time'] = datetime.now().isoformat()

        save_state(state)
        logger.info(f"Position opened successfully: {order.get('id')}")

        # Place TP/SL orders
        place_tp_sl_orders(exchange, state, config)

        return True

    except Exception as e:
        logger.error(f"Failed to open position: {e}")
        return False

def place_tp_sl_orders(exchange, state, config):
    """Place TP and SL orders for the position"""
    if not state.get('position'):
        return

    position = state['position']
    symbol = config['symbol']

    try:
        direction = position['direction']
        quantity = position['quantity']
        tp_price = position['tp_price']
        sl_price = position['sl_price']

        position_side = 'LONG' if direction == 'LONG' else 'SHORT'
        close_side = 'sell' if direction == 'LONG' else 'buy'

        # Place TP order
        try:
            tp_order = exchange.create_order(
                symbol=symbol,
                type='TAKE_PROFIT_MARKET',
                side=close_side,
                amount=quantity,
                params={
                    'positionSide': position_side,
                    'stopPrice': tp_price,
                    'closePosition': True,
                }
            )
            position['tp_order_id'] = tp_order.get('id')
            logger.info(f"TP order placed: {tp_order.get('id')} @ ${tp_price}")
        except Exception as e:
            logger.warning(f"TP order failed: {e}")

        # Place SL order
        try:
            sl_order = exchange.create_order(
                symbol=symbol,
                type='STOP_MARKET',
                side=close_side,
                amount=quantity,
                params={
                    'positionSide': position_side,
                    'stopPrice': sl_price,
                    'closePosition': True,
                }
            )
            position['sl_order_id'] = sl_order.get('id')
            logger.info(f"SL order placed: {sl_order.get('id')} @ ${sl_price}")
        except Exception as e:
            logger.warning(f"SL order failed: {e}")

        save_state(state)

    except Exception as e:
        logger.error(f"Failed to place TP/SL orders: {e}")

def check_position_status(exchange, state, config):
    """Check if position is still open and update state"""
    if not state.get('position'):
        return

    symbol = config['symbol']
    position = state['position']

    try:
        # Fetch current positions
        positions = exchange.fetch_positions([symbol])

        # Find our position
        position_side = 'long' if position['direction'] == 'LONG' else 'short'
        current_pos = None

        for pos in positions:
            if pos.get('side') == position_side and float(pos.get('contracts', 0)) > 0:
                current_pos = pos
                break

        # Position closed
        if current_pos is None or float(current_pos.get('contracts', 0)) == 0:
            # Calculate PnL
            ticker = exchange.fetch_ticker(symbol)
            exit_price = ticker['last']

            direction = 1 if position['direction'] == 'LONG' else -1
            pnl_pct = direction * (exit_price / position['entry_price'] - 1) * 100 * config['leverage']
            pnl_pct -= 2 * 0.05 * config['leverage']  # Fees

            # Determine exit reason
            if position['direction'] == 'LONG':
                if exit_price >= position['tp_price']:
                    exit_reason = 'TP'
                elif exit_price <= position['sl_price']:
                    exit_reason = 'SL'
                else:
                    exit_reason = 'UNKNOWN'
            else:
                if exit_price <= position['tp_price']:
                    exit_reason = 'TP'
                elif exit_price >= position['sl_price']:
                    exit_reason = 'SL'
                else:
                    exit_reason = 'UNKNOWN'

            logger.info(f"Position closed: {exit_reason} | PnL: {pnl_pct:+.2f}%")

            # Update stats
            state['total_trades'] += 1
            state['total_pnl'] += pnl_pct
            state['daily_trades'] += 1
            state['daily_pnl'] += pnl_pct

            if pnl_pct > 0:
                state['winning_trades'] += 1

            state['last_trade'] = {
                'direction': position['direction'],
                'entry_price': position['entry_price'],
                'exit_price': exit_price,
                'pnl_pct': pnl_pct,
                'exit_reason': exit_reason,
                'closed_at': datetime.now().isoformat(),
            }

            state['position'] = None
            save_state(state)

            return True

        return False

    except Exception as e:
        logger.error(f"Failed to check position status: {e}")
        return False

def close_position_market(exchange, state, config, reason="Manual"):
    """Close position at market price"""
    if not state.get('position'):
        return False

    position = state['position']
    symbol = config['symbol']

    try:
        direction = position['direction']
        quantity = position['quantity']

        close_side = 'sell' if direction == 'LONG' else 'buy'
        position_side = 'LONG' if direction == 'LONG' else 'SHORT'

        # Cancel existing orders first
        try:
            exchange.cancel_all_orders(symbol)
        except:
            pass

        # Place market close order
        order = exchange.create_market_order(
            symbol=symbol,
            side=close_side,
            amount=quantity,
            params={'positionSide': position_side}
        )

        logger.info(f"Position closed: {reason}")

        # Update state
        ticker = exchange.fetch_ticker(symbol)
        exit_price = ticker['last']

        dir_mult = 1 if direction == 'LONG' else -1
        pnl_pct = dir_mult * (exit_price / position['entry_price'] - 1) * 100 * config['leverage']
        pnl_pct -= 2 * 0.05 * config['leverage']

        state['total_trades'] += 1
        state['total_pnl'] += pnl_pct
        state['daily_trades'] += 1
        state['daily_pnl'] += pnl_pct

        if pnl_pct > 0:
            state['winning_trades'] += 1

        state['position'] = None
        save_state(state)

        logger.info(f"Closed at ${exit_price:.1f} | PnL: {pnl_pct:+.2f}%")

        return True

    except Exception as e:
        logger.error(f"Failed to close position: {e}")
        return False

# ============================================================
# MAIN LOOP
# ============================================================

def check_cooldown(state, config):
    """Check if cooldown period has passed"""
    if not state.get('last_signal_time'):
        return True

    cooldown_minutes = config['strategy']['cooldown_candles'] * 15  # 15m candles
    last_signal = datetime.fromisoformat(state['last_signal_time'])
    elapsed = (datetime.now() - last_signal).total_seconds() / 60

    return elapsed >= cooldown_minutes

def check_daily_loss_limit(state, config):
    """Check if daily loss limit is reached"""
    max_loss = config['risk']['max_daily_loss_pct']
    return state.get('daily_pnl', 0) <= -max_loss

def run_bot():
    """Main bot loop"""
    logger.info(f"=" * 60)
    logger.info(f"RSI Trend Filter Bot v{BOT_VERSION} Starting")
    logger.info(f"=" * 60)

    # Load configuration and state
    config = load_config()
    state = load_state()

    logger.info(f"Symbol: {config['symbol']}")
    logger.info(f"Strategy: RSI({config['strategy']['rsi_period']}) "
                f"{config['strategy']['rsi_long_threshold']}/{config['strategy']['rsi_short_threshold']} "
                f"EMA{config['strategy']['ema_period']}")
    logger.info(f"TP/SL: {config['strategy']['tp_pct']}% / {config['strategy']['sl_pct']}%")
    logger.info(f"Leverage: {config['leverage']}x")

    # Create exchange connection
    exchange = create_exchange()

    # Test connection
    try:
        balance = exchange.fetch_balance()
        available = float(balance.get('USDT', {}).get('free', 0))
        logger.info(f"Connected! Available balance: ${available:.2f}")
    except Exception as e:
        logger.error(f"Connection failed: {e}")
        return

    # Main loop
    iteration = 0
    while True:
        try:
            iteration += 1

            # Fetch OHLCV data
            ohlcv = exchange.fetch_ohlcv(config['symbol'], config['timeframe'], limit=200)
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')

            # Calculate indicators
            df = calculate_indicators(df, config)

            current_price = df['close'].iloc[-1]
            current_rsi = df['rsi'].iloc[-1]
            current_ema = df['ema'].iloc[-1]

            # Log status periodically
            if iteration % 10 == 1:
                pos_status = f"Position: {state['position']['direction']}" if state.get('position') else "No position"
                logger.info(f"Price: ${current_price:.1f} | RSI: {current_rsi:.1f} | EMA: ${current_ema:.1f} | {pos_status}")

            # Check daily loss limit
            if check_daily_loss_limit(state, config):
                logger.warning("Daily loss limit reached, pausing trading")
                time.sleep(300)
                continue

            # Check position status if we have one
            if state.get('position'):
                position_closed = check_position_status(exchange, state, config)
                if not position_closed:
                    # Position still open, wait for next iteration
                    time.sleep(60)
                    continue

            # Check for new entry signal
            if not state.get('position') and check_cooldown(state, config):
                signal, reason = check_entry_signal(df, state, config)

                if signal:
                    logger.info(f"Signal detected: {signal} - {reason}")
                    open_position(exchange, state, config, signal, reason)

            # Save state
            save_state(state)

            # Wait before next iteration
            time.sleep(60)  # Check every minute

        except KeyboardInterrupt:
            logger.info("Bot stopped by user")
            break
        except Exception as e:
            logger.error(f"Error in main loop: {e}")
            time.sleep(30)

    # Cleanup
    logger.info("Bot shutdown complete")

# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == '__main__':
    run_bot()
