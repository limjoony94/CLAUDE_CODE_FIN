"""
Pattern 5m Bot - Position Opening
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
    ROTATION_ENABLED,
    PATTERN_OPTIMAL_TPSL,
)
from .models import APICache, CircuitBreaker, PerformanceMetrics
from .exchange import fetch_ticker_cached, fetch_positions_cached, fetch_balance_cached, verify_position_mode
from .indicators import get_volatility_multiplier
from .state import save_state
from .orders import place_tp_sl_orders

logger = logging.getLogger('pattern_5m')


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
            logger.warning(f"Invalid position size (qty={quantity}, balance=${available}), skipping")
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

        # v1.6: Extract pattern from reason for pattern-specific TP/SL
        pattern = None
        if reason and 'Pattern:' in reason:
            pattern = reason.split('Pattern:')[-1].strip().split()[0]

        # v1.18: Get regime-specific TP/SL from state (set by check_entry_signal)
        regime_tp_sl = state.get('regime_tp_sl')
        current_regime = state.get('current_regime', 'UNKNOWN')

        tp_price, sl_price, tp_pct_adjusted, sl_pct_adjusted = calculate_tp_sl(
            actual_entry_price, direction, strategy, vol_mult, pattern, regime_tp_sl
        )

        # v1.18: Log regime-specific TP/SL usage
        if regime_tp_sl:
            logger.info(f"[v1.18] Regime-specific TP/SL: {current_regime} | {pattern} → TP={regime_tp_sl[0]}%, SL={regime_tp_sl[1]}%")
        # v1.6: Log pattern-specific TP/SL usage (fallback)
        elif pattern and pattern in PATTERN_OPTIMAL_TPSL:
            opt_tp, opt_sl = PATTERN_OPTIMAL_TPSL[pattern]
            logger.info(f"[v1.6] Pattern-specific TP/SL: {pattern} → TP={opt_tp}%, SL={opt_sl}%")
        logger.info(f"TP: ${tp_price:.1f} ({tp_pct_adjusted:.2f}%) | SL: ${sl_price:.1f} ({sl_pct_adjusted:.2f}%)")

        # Handle scale-out if enabled
        scale_out_stages = setup_scale_out(
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
            # Rotation (순환매) fields
            'rotation_enabled': ROTATION_ENABLED and bool(scale_out_stages),
            'is_partial': False,
            'avg_entry_price': actual_entry_price,
            'total_entries': 1,
            'refill_entries': [],
            # v1.18: Regime-adaptive fields
            'market_regime': current_regime,
            'regime_tp_sl': regime_tp_sl,
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
    except ccxt.InsufficientFunds as e:
        logger.error(f"Failed to open position (insufficient funds): {e}")
        return False
    except ccxt.ExchangeError as e:
        error_msg = str(e)
        # Auto-recover from Hedge mode error
        if 'Hedge mode' in error_msg or '109400' in error_msg:
            logger.warning(f"⚠️ Hedge mode detected, attempting to switch to One-Way mode...")
            try:
                exchange.set_position_mode(hedged=False, symbol=config['symbol'])
                logger.info("✅ Switched to One-Way mode, please retry the signal")
            except Exception as mode_err:
                logger.error(f"Failed to switch position mode: {mode_err}")
        else:
            logger.error(f"Failed to open position (exchange error): {e}")
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
        return True  # Verified: no existing position
    except ccxt.NetworkError as e:
        logger.warning(f"Could not verify exchange position (network error): {e}")
    except ccxt.ExchangeError as e:
        logger.warning(f"Could not verify exchange position (exchange error): {e}")
    except Exception as e:
        logger.warning(f"Could not verify exchange position: {e}")
    return False  # Cannot verify — block new position for safety


def _set_leverage(exchange: ccxt.bingx, symbol: str, leverage: int) -> None:
    """Set leverage for the symbol."""
    try:
        exchange.set_leverage(leverage, symbol, params={'side': 'BOTH'})
    except ccxt.ExchangeError as e:
        if 'No need to change' in str(e) or 'same' in str(e).lower():
            pass  # Already set
        else:
            logger.warning(f"Set leverage warning (exchange error): {e}")
    except ccxt.NetworkError as e:
        logger.warning(f"Set leverage warning (network error): {e}")
    except (ValueError, TypeError) as e:
        logger.warning(f"Set leverage warning (invalid value): {e}")


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


def calculate_tp_sl(
    entry_price: float,
    direction: int,
    strategy: Dict[str, Any],
    vol_mult: float,
    pattern: Optional[str] = None,
    regime_tp_sl: Optional[tuple] = None,
) -> tuple:
    """Calculate TP and SL prices — single source of truth.

    Used by: open_position, refill_position, recover_position_to_state,
             recalculate_position_orders.

    v1.18: Priority: regime_tp_sl > PATTERN_OPTIMAL_TPSL > strategy defaults
    v1.6: Uses pattern-specific TP/SL from PATTERN_OPTIMAL_TPSL if available.
    """
    # v1.18: Check for regime-specific TP/SL first (highest priority)
    if regime_tp_sl:
        base_tp_pct, base_sl_pct = regime_tp_sl
        logger.debug(f"Using regime-specific TP/SL: TP={base_tp_pct}%, SL={base_sl_pct}%")
    # v1.6: Check for pattern-specific optimal TP/SL
    elif pattern and pattern in PATTERN_OPTIMAL_TPSL:
        base_tp_pct, base_sl_pct = PATTERN_OPTIMAL_TPSL[pattern]
        logger.debug(f"Using pattern-specific TP/SL for {pattern}: TP={base_tp_pct}%, SL={base_sl_pct}%")
    else:
        base_tp_pct = strategy['tp_pct']
        base_sl_pct = strategy['sl_pct']

    tp_pct_adjusted = (base_tp_pct * vol_mult) + SLIPPAGE_BUFFER_PCT  # TP: add slippage (target further)
    sl_pct_adjusted = (base_sl_pct * vol_mult) - SLIPPAGE_BUFFER_PCT  # SL: subtract slippage (tighter)

    tp_price = round(entry_price * (1 + direction * tp_pct_adjusted / 100), PRICE_ROUND_DECIMALS)
    sl_price = round(entry_price * (1 - direction * sl_pct_adjusted / 100), PRICE_ROUND_DECIMALS)

    return tp_price, sl_price, tp_pct_adjusted, sl_pct_adjusted


def setup_scale_out(
    strategy: Dict[str, Any],
    entry_price: float,
    quantity: float,
    direction: int,
    tp_pct_adjusted: float,
) -> List[Dict]:
    """Setup scale-out stages if enabled — single source of truth.

    Used by: open_position, refill_position, recover_position_to_state,
             recalculate_position_orders.
    """
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


def refill_position(
    exchange: ccxt.bingx,
    state: Dict[str, Any],
    config: Dict[str, Any],
    reason: str,
    cache: APICache,
    circuit_breaker: Optional[CircuitBreaker] = None,
    metrics: Optional[PerformanceMetrics] = None,
    df: Optional[pd.DataFrame] = None,
) -> bool:
    """
    Refill a partial position back to full size (rotation/순환매).

    Args:
        exchange: Exchange instance
        state: Bot state dictionary
        config: Bot configuration
        reason: Refill reason string
        cache: APICache instance
        circuit_breaker: Optional CircuitBreaker
        metrics: Optional PerformanceMetrics
        df: Optional DataFrame for volatility calculation

    Returns:
        True if refill successful
    """
    symbol = config['symbol']
    strategy = config['strategy']
    position = state.get('position')

    if not position:
        logger.warning("No position to refill")
        return False

    if not position.get('is_partial', False):
        logger.warning("Position is not partial, cannot refill")
        return False

    try:
        # Calculate refill quantity
        original_qty = position.get('quantity', 0)
        remaining_qty = position.get('remaining_quantity', 0)
        refill_qty = round(original_qty - remaining_qty, QUANTITY_ROUND_DECIMALS)

        if refill_qty <= 0:
            logger.warning(f"Invalid refill quantity: {refill_qty}")
            return False

        # Get current price
        ticker = fetch_ticker_cached(exchange, symbol, cache, force_refresh=True,
                                     circuit_breaker=circuit_breaker, metrics=metrics)
        estimated_price = ticker['last']

        # Execute market order for refill
        side = 'buy' if position['direction'] == 'LONG' else 'sell'
        logger.info(f"🔄 Refilling {position['direction']}: {refill_qty} {symbol} @ ~${estimated_price:.1f}")

        order = exchange.create_market_order(
            symbol=symbol,
            side=side,
            amount=refill_qty,
            params={'positionSide': 'BOTH'}
        )

        # Get actual fill price
        actual_fill_price, actual_refill_qty = _get_actual_fill_price(
            exchange, order, position['direction'], symbol,
            estimated_price, refill_qty, cache, circuit_breaker, metrics
        )

        cache.invalidate_all()

        logger.info(f"Refill executed: ${actual_fill_price:.1f} (qty: {actual_refill_qty})")

        # Calculate new average entry price
        old_entry = position.get('avg_entry_price', 0)
        if old_entry == 0:
            old_entry = position.get('entry_price', 0)

        old_cost = old_entry * remaining_qty
        new_cost = actual_fill_price * actual_refill_qty
        total_qty = remaining_qty + actual_refill_qty
        new_avg_entry = (old_cost + new_cost) / total_qty

        logger.info(f"New average entry: ${new_avg_entry:.1f} (was ${old_entry:.1f})")

        # Recalculate TP/SL based on new average entry
        direction = 1 if position['direction'] == 'LONG' else -1
        vol_mult = position.get('vol_mult', 1.0)

        # Extract pattern from position reason for per-pattern TP/SL
        pattern = None
        pos_reason = position.get('reason', '')
        if pos_reason and 'Pattern:' in pos_reason:
            pattern = pos_reason.split('Pattern:')[-1].strip().split()[0]
        regime_tp_sl = position.get('regime_tp_sl')

        tp_price, sl_price, tp_pct_adjusted, sl_pct_adjusted = calculate_tp_sl(
            new_avg_entry, direction, strategy, vol_mult, pattern, regime_tp_sl
        )

        logger.info(f"New TP: ${tp_price:.1f} | New SL: ${sl_price:.1f}")

        # Update position state
        position['remaining_quantity'] = round(total_qty, QUANTITY_ROUND_DECIMALS)
        position['avg_entry_price'] = new_avg_entry
        position['tp_price'] = tp_price
        position['sl_price'] = sl_price
        position['is_partial'] = False  # No longer partial
        position['total_entries'] = position.get('total_entries', 1) + 1

        # Record refill entry
        refill_entries = position.get('refill_entries', [])
        refill_entries.append({
            'price': actual_fill_price,
            'quantity': actual_refill_qty,
            'time': datetime.now().isoformat(),
            'reason': reason,
        })
        position['refill_entries'] = refill_entries

        # Reset scale-out stages for new full position
        scale_out_stages = setup_scale_out(
            strategy, new_avg_entry, total_qty, direction, tp_pct_adjusted
        )
        position['scale_out_stages'] = scale_out_stages
        position['scale_out_enabled'] = bool(scale_out_stages)

        state['last_signal_time'] = datetime.now().isoformat()
        save_state(state)

        logger.info(f"✅ Position refilled: entries={position['total_entries']}, avg=${new_avg_entry:.1f}")

        # Cancel old TP/SL and place new orders
        _cancel_existing_tp_sl(exchange, position, symbol)
        place_tp_sl_orders(exchange, state, config)

        return True

    except ccxt.NetworkError as e:
        logger.error(f"Failed to refill position (network error): {e}")
        return False
    except ccxt.InsufficientFunds as e:
        logger.error(f"Failed to refill position (insufficient funds): {e}")
        return False
    except ccxt.ExchangeError as e:
        logger.error(f"Failed to refill position (exchange error): {e}")
        return False
    except Exception as e:
        logger.error(f"Failed to refill position: {e}")
        return False


def _cancel_existing_tp_sl(
    exchange: ccxt.bingx,
    position: Dict[str, Any],
    symbol: str,
) -> None:
    """Cancel existing TP/SL orders before placing new ones."""
    orders_to_cancel = []

    if position.get('sl_order_id'):
        orders_to_cancel.append(position['sl_order_id'])

    # Cancel scale-out TP orders
    for stage in position.get('scale_out_stages', []):
        if stage.get('order_id') and not stage.get('filled'):
            orders_to_cancel.append(stage['order_id'])

    for order_id in orders_to_cancel:
        try:
            exchange.cancel_order(order_id, symbol)
            logger.debug(f"Cancelled order {order_id}")
        except ccxt.OrderNotFound:
            pass  # Already cancelled or filled
        except ccxt.NetworkError as e:
            logger.warning(f"Failed to cancel order {order_id} (network error): {e}")
        except ccxt.ExchangeError as e:
            logger.warning(f"Failed to cancel order {order_id} (exchange error): {e}")
        except Exception as e:
            logger.warning(f"Failed to cancel order {order_id}: {e}")