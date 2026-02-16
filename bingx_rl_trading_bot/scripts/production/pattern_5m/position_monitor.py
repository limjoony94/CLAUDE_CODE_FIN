"""
Pattern 5m Bot - Position Monitoring
Functions for monitoring and syncing trading positions.
"""

import time
import logging
from datetime import datetime
from typing import Dict, Any, Optional

import ccxt

from .constants import (
    PRICE_TOLERANCE_PCT,
    TP_LOWER_MULT,
    TP_UPPER_MULT,
    SL_LOWER_MULT,
    SL_UPPER_MULT,
    QTY_REDUCTION_THRESHOLD,
    EXIT_PRICE_RETRY_DELAY,
    EXIT_PRICE_INITIAL_DELAY,
    MAX_EXIT_PRICE_RETRIES,
)
from .models import APICache, CircuitBreaker, PerformanceMetrics
from .exchange import fetch_ticker_cached, fetch_positions_cached
from .state import save_state

logger = logging.getLogger('pattern_5m')


def sync_position_with_exchange(
    exchange: ccxt.bingx,
    state: Dict[str, Any],
    config: Dict[str, Any],
    cache: APICache,
    circuit_breaker: Optional[CircuitBreaker] = None,
    metrics: Optional[PerformanceMetrics] = None,
) -> bool:
    """
    Synchronize local state with exchange position.

    Args:
        exchange: Exchange instance
        state: Bot state dictionary
        config: Bot configuration
        cache: APICache instance
        circuit_breaker: Optional CircuitBreaker
        metrics: Optional PerformanceMetrics

    Returns:
        True if sync was needed
    """
    # Import here to avoid circular dependency
    from .position_close import record_closed_position, recover_position_to_state

    symbol = config['symbol']

    try:
        positions = fetch_positions_cached(exchange, symbol, cache, force_refresh=True,
                                           circuit_breaker=circuit_breaker, metrics=metrics)

        exchange_long = None
        exchange_short = None

        for pos in positions:
            contracts = float(pos.get('contracts', 0))
            if contracts > 0:
                if pos.get('side') == 'long':
                    exchange_long = pos
                elif pos.get('side') == 'short':
                    exchange_short = pos

        state_position = state.get('position')

        logger.debug(
            f"Position sync: exchange_long={'yes' if exchange_long else 'no'}, "
            f"exchange_short={'yes' if exchange_short else 'no'}, "
            f"state_position={'yes' if state_position else 'no'}"
        )

        # State has position but exchange doesn't
        if state_position and not exchange_long and not exchange_short:
            logger.warning("State has position but exchange doesn't")
            actual_exit = get_actual_exit_price(exchange, state, config)
            if actual_exit:
                record_closed_position(exchange, state, config, actual_exit['price'],
                                      actual_exit['reason'], cache, metrics)
            else:
                record_closed_position(exchange, state, config, state_position['entry_price'],
                                      'EXTERNAL', cache, metrics)
            return True

        # Exchange has position but state doesn't
        if not state_position:
            if exchange_long:
                logger.info("Exchange has LONG position but state doesn't - recovering")
                recover_position_to_state(state, config, exchange_long, 'LONG', exchange, cache)
                return True
            elif exchange_short:
                logger.info("Exchange has SHORT position but state doesn't - recovering")
                recover_position_to_state(state, config, exchange_short, 'SHORT', exchange, cache)
                return True

        # Direction mismatch: state says one direction but exchange has the other
        if state_position:
            state_dir = state_position.get('direction', '')
            expected_exchange = exchange_long if state_dir == 'LONG' else exchange_short
            opposite_exchange = exchange_short if state_dir == 'LONG' else exchange_long
            if not expected_exchange and opposite_exchange:
                opposite_dir = 'SHORT' if state_dir == 'LONG' else 'LONG'
                logger.error(
                    f"Direction mismatch: state={state_dir} but exchange has {opposite_dir}. "
                    f"Closing state position, recovering exchange position."
                )
                actual_exit = get_actual_exit_price(exchange, state, config)
                exit_price = actual_exit['price'] if actual_exit else state_position['entry_price']
                exit_reason = 'DIRECTION_MISMATCH'
                record_closed_position(exchange, state, config, exit_price,
                                      exit_reason, cache, metrics)
                recover_position_to_state(state, config, opposite_exchange, opposite_dir, exchange, cache)
                return True

        logger.info("Position sync completed - state matches exchange")
        return False

    except ccxt.NetworkError as e:
        logger.error(f"Failed to sync position (network error): {e}")
        return False
    except ccxt.ExchangeError as e:
        logger.error(f"Failed to sync position (exchange error): {e}")
        return False
    except Exception as e:
        logger.exception(f"Failed to sync position: {e}")
        return False


def get_actual_exit_price(
    exchange: ccxt.bingx,
    state: Dict[str, Any],
    config: Dict[str, Any],
) -> Optional[Dict]:
    """
    Get actual exit price from trade history.

    Args:
        exchange: Exchange instance
        state: Bot state dictionary
        config: Bot configuration

    Returns:
        Dict with 'price' and 'reason', or None
    """
    symbol = config['symbol']
    position = state.get('position')
    if not position:
        return None

    try:
        trades = exchange.fetch_my_trades(symbol, limit=20)
        close_side = 'sell' if position['direction'] == 'LONG' else 'buy'

        # Convert entry_time to epoch ms for timezone-safe comparison
        entry_time_str = position.get('entry_time', '')
        entry_ts = 0
        if entry_time_str:
            try:
                entry_ts = datetime.fromisoformat(entry_time_str).timestamp() * 1000
            except (ValueError, TypeError):
                entry_ts = 0

        for trade in reversed(trades):
            if trade.get('side') == close_side:
                trade_ts = trade.get('timestamp', 0)  # CCXT provides epoch ms
                if trade_ts > entry_ts:
                    filled_price = float(trade.get('price', 0))
                    if filled_price > 0:
                        reason = _infer_exit_reason(filled_price, position)
                        return {'price': filled_price, 'reason': reason}
    except ccxt.NetworkError as e:
        logger.debug(f"fetch_my_trades failed (network error): {e}")
    except ccxt.ExchangeError as e:
        logger.debug(f"fetch_my_trades failed (exchange error): {e}")
    except Exception as e:
        logger.debug(f"fetch_my_trades failed: {e}")

    return None


def _infer_exit_reason(filled_price: float, position: Dict) -> str:
    """Infer exit reason from filled price."""
    tp = position.get('tp_price', 0)
    sl = position.get('sl_price', 0)
    direction = position.get('direction', '')

    if direction == 'LONG':
        if tp and abs(filled_price - tp) / tp < PRICE_TOLERANCE_PCT:
            return 'TP'
        elif sl and abs(filled_price - sl) / sl < PRICE_TOLERANCE_PCT:
            return 'SL'
    elif direction == 'SHORT':
        if tp and abs(filled_price - tp) / tp < PRICE_TOLERANCE_PCT:
            return 'TP'
        elif sl and abs(filled_price - sl) / sl < PRICE_TOLERANCE_PCT:
            return 'SL'

    return 'MARKET'


def check_position_status(
    exchange: ccxt.bingx,
    state: Dict[str, Any],
    config: Dict[str, Any],
    cache: APICache,
    circuit_breaker: Optional[CircuitBreaker] = None,
    metrics: Optional[PerformanceMetrics] = None,
) -> bool:
    """
    Check if position is still open or has been closed.

    Args:
        exchange: Exchange instance
        state: Bot state dictionary
        config: Bot configuration
        cache: APICache instance
        circuit_breaker: Optional CircuitBreaker
        metrics: Optional PerformanceMetrics

    Returns:
        True if position was closed
    """
    if not state.get('position'):
        return False

    symbol = config['symbol']
    position = state['position']

    try:
        positions = fetch_positions_cached(exchange, symbol, cache, force_refresh=True,
                                           circuit_breaker=circuit_breaker, metrics=metrics)
        position_side = 'long' if position['direction'] == 'LONG' else 'short'
        opposite_side = 'short' if position_side == 'long' else 'long'
        current_pos = None
        opposite_pos = None

        for pos in positions:
            contracts = float(pos.get('contracts', 0))
            if contracts > 0:
                if pos.get('side') == position_side:
                    current_pos = pos
                elif pos.get('side') == opposite_side:
                    opposite_pos = pos

        # Direction mismatch: expected side empty but opposite side has position
        if current_pos is None and opposite_pos is not None:
            from .position_close import recover_position_to_state
            actual_dir = opposite_side.upper()
            logger.error(
                f"Direction mismatch: state={position['direction']} but "
                f"exchange has {actual_dir} position. "
                f"Closing state position and recovering exchange position immediately."
            )
            # Close the stale local state first
            _handle_position_closed(exchange, state, config, position, cache, metrics)
            # Immediately recover the actual exchange position
            recover_position_to_state(state, config, opposite_pos, actual_dir, exchange, cache)
            return True

        # Handle scale-out partial fills
        scale_out_enabled = position.get('scale_out_enabled', False)
        if scale_out_enabled and current_pos:
            _check_scale_out_fills(state, position, current_pos)

        # Position is closed
        if current_pos is None or float(current_pos.get('contracts', 0)) == 0:
            return _handle_position_closed(exchange, state, config, position, cache, metrics)

        return False

    except ccxt.NetworkError as e:
        logger.error(f"Failed to check position status (network error): {e}")
        return False
    except ccxt.ExchangeError as e:
        logger.error(f"Failed to check position status (exchange error): {e}")
        return False
    except Exception as e:
        logger.exception(f"Failed to check position status: {e}")
        return False


def _check_scale_out_fills(
    state: Dict[str, Any],
    position: Dict,
    current_pos: Dict,
) -> None:
    """Check and update scale-out stage fills."""
    current_qty = float(current_pos.get('contracts', 0))
    prev_remaining = position.get('remaining_quantity', position['quantity'])

    if current_qty < prev_remaining * QTY_REDUCTION_THRESHOLD:
        position['remaining_quantity'] = current_qty

        total_qty = position['quantity']
        stages = position.get('scale_out_stages', [])
        cumulative_filled = 0.0

        for stage in stages:
            cumulative_filled += stage['quantity']
            filled_threshold = total_qty - cumulative_filled

            if not stage['filled'] and current_qty <= filled_threshold + 0.0001:
                stage['filled'] = True
                logger.info(f"📈 Scale-out Stage {stage['stage']} filled: {stage['quantity']}")

                # Check if rotation is enabled and TP1 (stage 1) just filled
                if stage['stage'] == 1 and position.get('rotation_enabled', False):
                    position['is_partial'] = True
                    logger.info(f"🔄 Rotation: Position is now partial ({current_qty:.4f}), looking for refill signals")

        save_state(state)


def _handle_position_closed(
    exchange: ccxt.bingx,
    state: Dict[str, Any],
    config: Dict[str, Any],
    position: Dict,
    cache: APICache,
    metrics: Optional[PerformanceMetrics],
) -> bool:
    """Handle position closure detection."""
    # Import here to avoid circular dependency
    from .position_close import record_closed_position

    logger.info(
        f"🔍 Position closure detected: {position['direction']} "
        f"(entry=${position['entry_price']:.1f}), fetching exit details..."
    )
    time.sleep(EXIT_PRICE_INITIAL_DELAY)

    actual_exit = None
    for retry in range(MAX_EXIT_PRICE_RETRIES):
        actual_exit = get_actual_exit_price(exchange, state, config)
        if actual_exit:
            break
        if retry < MAX_EXIT_PRICE_RETRIES - 1:
            time.sleep(EXIT_PRICE_RETRY_DELAY)

    if actual_exit:
        exit_price = actual_exit['price']
        exit_reason = actual_exit['reason']
    else:
        ticker = fetch_ticker_cached(exchange, config['symbol'], cache, force_refresh=True)
        exit_price = ticker['last']
        exit_reason = _infer_exit_from_price(exit_price, position)

    # Adjust reason for scale-out
    scale_out_enabled = position.get('scale_out_enabled', False)
    if scale_out_enabled:
        stages = position.get('scale_out_stages', [])
        filled_stages = sum(1 for s in stages if s['filled'])
        if filled_stages == len(stages):
            exit_reason = 'TP_SCALEOUT'
        elif exit_reason == 'SL':
            exit_reason = f'SL_AFTER_{filled_stages}_STAGES'

    record_closed_position(exchange, state, config, exit_price, exit_reason, cache, metrics)
    return True


def _infer_exit_from_price(exit_price: float, position: Dict) -> str:
    """Infer exit reason from current price vs TP/SL levels."""
    tp = position.get('tp_price', 0)
    sl = position.get('sl_price', 0)
    direction = position.get('direction', '')

    if direction == 'LONG':
        if exit_price >= tp * TP_LOWER_MULT:
            return 'TP'
        elif exit_price <= sl * SL_UPPER_MULT:
            return 'SL'
    else:  # SHORT
        if exit_price <= tp * TP_UPPER_MULT:
            return 'TP'
        elif exit_price >= sl * SL_LOWER_MULT:
            return 'SL'

    return 'UNKNOWN'