"""
Engulf 5m Bot - Position Opening
Functions for opening new trading positions.
"""

import time
import logging
import pandas as pd
from datetime import datetime
from typing import Dict, Any, Optional, List

import ccxt

from .constants import (
    SLIPPAGE_BUFFER_PCT,
    ENTRY_PRICE_FETCH_DELAY,
    PRICE_ROUND_DECIMALS,
    QUANTITY_ROUND_DECIMALS,
)
from .models import APICache, CircuitBreaker, PerformanceMetrics
from .exchange import fetch_ticker_cached, fetch_positions_cached, fetch_balance_cached
from .indicators import get_volatility_multiplier
from .state import save_state
from .orders import place_tp_sl_orders

logger = logging.getLogger('engulf_5m')


def get_position_size(
    exchange: ccxt.bingx,
    config: Dict[str, Any],
    cache: APICache,
    circuit_breaker: Optional[CircuitBreaker] = None,
    metrics: Optional[PerformanceMetrics] = None,
) -> tuple:
    """
    Calculate position size based on available balance.

    Args:
        exchange: Exchange instance
        config: Bot configuration
        cache: APICache instance
        circuit_breaker: Optional CircuitBreaker
        metrics: Optional PerformanceMetrics

    Returns:
        Tuple of (quantity, available_balance) or (None, None) on error
    """
    try:
        balance = fetch_balance_cached(exchange, cache, force_refresh=True,
                                       circuit_breaker=circuit_breaker, metrics=metrics)
        available = float(balance.get('USDT', {}).get('free', 0))

        size_pct = config['position_size_pct'] / 100
        max_size = config['risk']['max_position_size_usd']
        position_value = min(available * size_pct, max_size)

        ticker = fetch_ticker_cached(exchange, config['symbol'], cache, force_refresh=True,
                                     circuit_breaker=circuit_breaker, metrics=metrics)
        price = ticker['last']

        quantity = (position_value * config['leverage']) / price
        quantity = round(quantity, QUANTITY_ROUND_DECIMALS)

        return quantity, available
    except ccxt.NetworkError as e:
        logger.error(f"Failed to calculate position size (network error): {e}")
        return None, None
    except ccxt.ExchangeError as e:
        logger.error(f"Failed to calculate position size (exchange error): {e}")
        return None, None
    except Exception as e:
        logger.error(f"Failed to calculate position size: {e}")
        return None, None


def open_position(
    exchange: ccxt.bingx,
    state: Dict[str, Any],
    config: Dict[str, Any],
    signal: str,
    reason: str,
    cache: APICache,
    circuit_breaker: Optional[CircuitBreaker] = None,
    metrics: Optional[PerformanceMetrics] = None,
    df: Optional[pd.DataFrame] = None,
) -> bool:
    """
    Open a new trading position.

    Args:
        exchange: Exchange instance
        state: Bot state dictionary
        config: Bot configuration
        signal: 'LONG' or 'SHORT'
        reason: Signal reason string
        cache: APICache instance
        circuit_breaker: Optional CircuitBreaker
        metrics: Optional PerformanceMetrics
        df: Optional DataFrame for volatility calculation

    Returns:
        True if position opened successfully
    """
    symbol = config['symbol']
    leverage = config['leverage']
    exchange_leverage = config.get('exchange_leverage', leverage)
    strategy = config['strategy']

    try:
        # Check for existing position on exchange
        if not _verify_no_existing_position(exchange, state, config, cache, circuit_breaker, metrics):
            return False

        # Set leverage
        _set_leverage(exchange, symbol, exchange_leverage)

        # Calculate position size
        quantity, available = get_position_size(exchange, config, cache, circuit_breaker, metrics)
        if quantity is None or quantity <= 0:
            logger.warning("Invalid position size, skipping")
            return False

        # Get estimated price
        ticker = fetch_ticker_cached(exchange, symbol, cache, force_refresh=True,
                                     circuit_breaker=circuit_breaker, metrics=metrics)
        estimated_price = ticker['last']

        # Execute market order
        side = 'buy' if signal == 'LONG' else 'sell'
        logger.info(f"Opening {signal} position: {quantity} {symbol} @ ~${estimated_price:.1f}")

        order = exchange.create_market_order(
            symbol=symbol,
            side=side,
            amount=quantity,
            params={'positionSide': 'BOTH'}
        )

        # Get actual fill price
        actual_entry_price, actual_quantity = _get_actual_fill_price(
            exchange, order, signal, symbol, estimated_price, quantity, cache, circuit_breaker, metrics
        )

        # Invalidate cache after order
        cache.invalidate_all()

        logger.info(f"Actual fill: ${actual_entry_price:.1f} (qty: {actual_quantity})")

        # Calculate TP/SL
        direction = 1 if signal == 'LONG' else -1
        vol_mult = 1.0
        if df is not None:
            vol_mult = get_volatility_multiplier(df, config)
            if vol_mult != 1.0:
                logger.info(f"Vol-Adaptive: ATR multiplier = {vol_mult:.2f}x")

        tp_price, sl_price, tp_pct_adjusted, sl_pct_adjusted = _calculate_tp_sl(
            actual_entry_price, direction, strategy, vol_mult
        )

        logger.info(f"TP: ${tp_price:.1f} ({tp_pct_adjusted:.2f}%) | SL: ${sl_price:.1f} ({sl_pct_adjusted:.2f}%)")

        # Handle scale-out if enabled
        scale_out_stages = _setup_scale_out(
            strategy, actual_entry_price, actual_quantity, direction, tp_pct_adjusted
        )

        # Update state
        state['position'] = {
            'direction': signal,
            'entry_price': actual_entry_price,
            'estimated_entry': estimated_price,
            'quantity': actual_quantity,
            'remaining_quantity': actual_quantity,
            'tp_price': tp_price,
            'sl_price': sl_price,
            'vol_mult': vol_mult,
            'scale_out_enabled': bool(scale_out_stages),
            'scale_out_stages': scale_out_stages,
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

    except ccxt.NetworkError as e:
        logger.error(f"Failed to open position (network error): {e}")
        return False
    except ccxt.ExchangeError as e:
        logger.error(f"Failed to open position (exchange error): {e}")
        return False
    except ccxt.InsufficientFunds as e:
        logger.error(f"Failed to open position (insufficient funds): {e}")
        return False
    except Exception as e:
        logger.error(f"Failed to open position: {e}")
        return False


def _verify_no_existing_position(
    exchange: ccxt.bingx,
    state: Dict[str, Any],
    config: Dict[str, Any],
    cache: APICache,
    circuit_breaker: Optional[CircuitBreaker],
    metrics: Optional[PerformanceMetrics],
) -> bool:
    """Verify no position exists before opening new one."""
    # Import here to avoid circular dependency
    from .position_monitor import sync_position_with_exchange

    try:
        positions = fetch_positions_cached(exchange, config['symbol'], cache, force_refresh=True,
                                           circuit_breaker=circuit_breaker, metrics=metrics)
        for pos in positions:
            if abs(float(pos.get('contracts', 0))) > 0:
                logger.warning("Position already exists on exchange")
                sync_position_with_exchange(exchange, state, config, cache, circuit_breaker, metrics)
                return False
    except ccxt.NetworkError as e:
        logger.warning(f"Could not verify exchange position (network error): {e}")
    except ccxt.ExchangeError as e:
        logger.warning(f"Could not verify exchange position (exchange error): {e}")
    except Exception as e:
        logger.warning(f"Could not verify exchange position: {e}")
    return True


def _set_leverage(exchange: ccxt.bingx, symbol: str, leverage: int) -> None:
    """Set leverage for the symbol."""
    try:
        exchange.set_leverage(leverage, symbol)
    except ccxt.ExchangeError as e:
        if 'No need to change' in str(e) or 'same' in str(e).lower():
            pass  # Already set
        else:
            logger.warning(f"Set leverage warning (exchange error): {e}")
    except Exception as e:
        logger.warning(f"Set leverage warning: {e}")


def _get_actual_fill_price(
    exchange: ccxt.bingx,
    order: Dict,
    signal: str,
    symbol: str,
    estimated_price: float,
    quantity: float,
    cache: APICache,
    circuit_breaker: Optional[CircuitBreaker],
    metrics: Optional[PerformanceMetrics],
) -> tuple:
    """Get actual fill price from order or positions."""
    actual_entry_price = float(order.get('average', order.get('price', estimated_price)))
    actual_quantity = float(order.get('filled', quantity))

    if actual_entry_price == 0 or actual_entry_price == estimated_price:
        time.sleep(ENTRY_PRICE_FETCH_DELAY)
        try:
            positions = fetch_positions_cached(exchange, symbol, cache, force_refresh=True,
                                               circuit_breaker=circuit_breaker, metrics=metrics)
            pos_side = 'long' if signal == 'LONG' else 'short'
            for pos in positions:
                if pos.get('side') == pos_side and float(pos.get('contracts', 0)) > 0:
                    actual_entry_price = float(pos.get('entryPrice', estimated_price))
                    actual_quantity = float(pos.get('contracts', actual_quantity))
                    break
        except ccxt.NetworkError as e:
            logger.warning(f"Could not fetch actual entry price (network error): {e}")
            actual_entry_price = estimated_price
        except ccxt.ExchangeError as e:
            logger.warning(f"Could not fetch actual entry price (exchange error): {e}")
            actual_entry_price = estimated_price
        except Exception as e:
            logger.warning(f"Could not fetch actual entry price: {e}")
            actual_entry_price = estimated_price

    return actual_entry_price, actual_quantity


def _calculate_tp_sl(
    entry_price: float,
    direction: int,
    strategy: Dict[str, Any],
    vol_mult: float,
) -> tuple:
    """Calculate TP and SL prices."""
    base_tp_pct = strategy['tp_pct']
    base_sl_pct = strategy['sl_pct']
    tp_pct_adjusted = (base_tp_pct * vol_mult) - SLIPPAGE_BUFFER_PCT
    sl_pct_adjusted = (base_sl_pct * vol_mult) - SLIPPAGE_BUFFER_PCT

    tp_price = round(entry_price * (1 + direction * tp_pct_adjusted / 100), PRICE_ROUND_DECIMALS)
    sl_price = round(entry_price * (1 - direction * sl_pct_adjusted / 100), PRICE_ROUND_DECIMALS)

    return tp_price, sl_price, tp_pct_adjusted, sl_pct_adjusted


def _setup_scale_out(
    strategy: Dict[str, Any],
    entry_price: float,
    quantity: float,
    direction: int,
    tp_pct_adjusted: float,
) -> List[Dict]:
    """Setup scale-out stages if enabled."""
    scale_out_config = strategy.get('scale_out', {})
    scale_out_enabled = scale_out_config.get('enabled', False)
    scale_out_stages = []

    if scale_out_enabled:
        stages_config = scale_out_config.get('stages', [])
        allocated_qty = 0.0

        for i, (pct, tp_mult) in enumerate(stages_config):
            stage_tp_pct = tp_pct_adjusted * tp_mult
            stage_tp_price = round(entry_price * (1 + direction * stage_tp_pct / 100), PRICE_ROUND_DECIMALS)

            if i == len(stages_config) - 1:
                stage_qty = round(quantity - allocated_qty, QUANTITY_ROUND_DECIMALS)
            else:
                stage_qty = round(quantity * pct, QUANTITY_ROUND_DECIMALS)
                allocated_qty += stage_qty

            scale_out_stages.append({
                'stage': i + 1,
                'pct': pct,
                'tp_mult': tp_mult,
                'tp_price': stage_tp_price,
                'quantity': stage_qty,
                'filled': False,
                'order_id': None,
            })
            logger.info(f"  Stage {i+1}: {pct*100:.0f}% @ ${stage_tp_price:.1f}")

    return scale_out_stages