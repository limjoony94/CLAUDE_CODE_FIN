"""
Pattern 5m Bot - Order Management
Place and manage TP/SL orders.
"""

import logging
from typing import Dict, Any, List, Optional

import ccxt

from .state import save_state
from .constants import PATTERN_OPTIMAL_TPSL, QUANTITY_ROUND_DECIMALS
from .utils import extract_pattern_name

logger = logging.getLogger('pattern_5m')

# Sentinel value: TP/SL order exists on exchange but local ID is unknown (crash recovery)
_EXCHANGE_MANAGED = "EXCHANGE_MANAGED"


def _get_position_side(config: Dict[str, Any], direction: str) -> str:
    """Get positionSide param for BingX API. Always 'BOTH' (One-Way mode)."""
    return 'BOTH'


def place_tp_sl_orders(
    exchange: ccxt.bingx,
    state: Dict[str, Any],
    config: Dict[str, Any],
    position: Optional[Dict] = None,
) -> None:
    """
    Place TP and SL orders for a position slot.

    Args:
        exchange: Exchange instance
        state: Bot state dictionary
        config: Bot configuration
        position: Specific position slot dict (if None, uses first active position)
    """
    if position is None:
        positions = state.get('positions') or {}
        if not positions:
            return
        position = next(iter(positions.values()))

    symbol = config['symbol']

    try:
        direction = position['direction']
        quantity = position.get('remaining_quantity', position['quantity'])
        tp_price = position['tp_price']
        sl_price = position['sl_price']
        close_side = 'sell' if direction == 'LONG' else 'buy'
        position_side = _get_position_side(config, direction)

        scale_out_enabled = position.get('scale_out_enabled', False)
        scale_out_stages = position.get('scale_out_stages', [])

        # Handle scale-out TP orders
        if scale_out_enabled and scale_out_stages:
            _place_scale_out_orders(exchange, position, symbol, close_side, scale_out_stages, position_side)
        else:
            _place_single_tp_order(exchange, position, symbol, close_side, quantity, tp_price, position_side)

        # Always place SL order
        _place_sl_order(exchange, position, symbol, close_side, quantity, sl_price, position_side)

        save_state(state)

    except ccxt.NetworkError as e:
        logger.error(f"Network error placing TP/SL orders: {e}")
    except ccxt.ExchangeError as e:
        logger.error(f"Exchange error placing TP/SL orders: {e}")
    except Exception as e:
        logger.exception(f"Failed to place TP/SL orders: {e}")


def _place_scale_out_orders(
    exchange: ccxt.bingx,
    position: Dict,
    symbol: str,
    close_side: str,
    scale_out_stages: List[Dict],
    position_side: str = 'BOTH',
) -> None:
    """Place staged TP orders for scale-out."""
    logger.info(f"📈 Scale-out: placing {len(scale_out_stages)} staged TP orders")

    for idx, stage in enumerate(scale_out_stages):
        try:
            stage_qty = stage['quantity']
            stage_tp_price = stage['tp_price']

            order_params = {
                'positionSide': position_side,
                'stopPrice': stage_tp_price,
            }

            tp_order = exchange.create_order(
                symbol=symbol,
                type='TAKE_PROFIT_MARKET',
                side=close_side,
                amount=stage_qty,
                params=order_params
            )
            stage['order_id'] = tp_order.get('id')
            logger.info(f"  Stage {stage['stage']} TP: {tp_order.get('id')} @ ${stage_tp_price:.1f}")

        except ccxt.InsufficientFunds as e:
            logger.warning(f"Stage {stage['stage']} TP order failed (insufficient funds): {e}")
        except ccxt.ExchangeError as e:
            logger.warning(f"Stage {stage['stage']} TP order failed (exchange error): {e}")
        except Exception as e:
            logger.warning(f"Stage {stage['stage']} TP order failed: {e}")


def _place_single_tp_order(
    exchange: ccxt.bingx,
    position: Dict,
    symbol: str,
    close_side: str,
    quantity: float,
    tp_price: float,
    position_side: str = 'BOTH',
) -> None:
    """Place a single TP order."""
    try:
        tp_order = exchange.create_order(
            symbol=symbol,
            type='TAKE_PROFIT_MARKET',
            side=close_side,
            amount=quantity,
            params={
                'positionSide': position_side,
                'stopPrice': tp_price,
            }
        )
        position['tp_order_id'] = tp_order.get('id')
        logger.info(f"TP order placed: {tp_order.get('id')} @ ${tp_price} (qty: {quantity})")
    except ccxt.InsufficientFunds as e:
        logger.warning(f"TP order failed (insufficient funds): {e}")
    except ccxt.ExchangeError as e:
        error_msg = str(e)
        if '110407' in error_msg:
            position['tp_order_id'] = _EXCHANGE_MANAGED
            logger.info("TP order already exists on exchange — marking as managed")
        elif '110413' in error_msg:
            position['tp_order_id'] = _EXCHANGE_MANAGED
            logger.warning("TP price already exceeded — marking as managed, position_monitor will handle")
        else:
            logger.warning(f"TP order failed (exchange error): {e}")
    except Exception as e:
        logger.warning(f"TP order failed: {e}")


def _place_sl_order(
    exchange: ccxt.bingx,
    position: Dict,
    symbol: str,
    close_side: str,
    quantity: float,
    sl_price: float,
    position_side: str = 'BOTH',
) -> None:
    """Place a SL order."""
    try:
        sl_order = exchange.create_order(
            symbol=symbol,
            type='STOP_MARKET',
            side=close_side,
            amount=quantity,
            params={
                'positionSide': position_side,
                'stopPrice': sl_price,
            }
        )
        position['sl_order_id'] = sl_order.get('id')
        logger.info(f"SL order placed: {sl_order.get('id')} @ ${sl_price} (qty: {quantity})")
    except ccxt.InsufficientFunds as e:
        logger.warning(f"SL order failed (insufficient funds): {e}")
    except ccxt.ExchangeError as e:
        error_msg = str(e)
        if '110406' in error_msg:
            position['sl_order_id'] = _EXCHANGE_MANAGED
            logger.info("SL order already exists on exchange — marking as managed")
        else:
            logger.warning(f"SL order failed (exchange error): {e}")
    except Exception as e:
        logger.warning(f"SL order failed: {e}")


def adjust_tpsl_to_config(
    exchange: ccxt.bingx,
    state: Dict[str, Any],
    config: Dict[str, Any],
) -> bool:
    """
    Adjust TP/SL orders to match current config on bot startup.

    Iterates all active position slots and checks if TP/SL prices match
    the expected values from the current config. Supports all modes:
    - Dynamic per-pattern: uses _dynamic_patterns_tpsl from config
    - Dynamic universal: uses _dynamic_tp/_dynamic_sl from config
    - Static: uses PATTERN_OPTIMAL_TPSL from constants

    If they differ, cancels the old orders and places new ones.

    Args:
        exchange: Exchange instance
        state: Bot state dictionary
        config: Bot configuration

    Returns:
        True if any adjustment was made, False otherwise
    """
    positions = state.get('positions') or {}
    if not positions:
        return False

    symbol = config['symbol']
    any_adjusted = False

    for position in positions.values():
        adjusted = _adjust_single_position_tpsl(exchange, position, state, config, symbol)
        any_adjusted = any_adjusted or adjusted

    # Always verify emergency SL matches current slot SL prices.
    # This covers: (1) TP/SL adjustments above, (2) stale emergency SL
    # from previous session, (3) state corruption recovery scenarios.
    _verify_emergency_sl(exchange, state, config)

    return any_adjusted


def _adjust_single_position_tpsl(
    exchange: ccxt.bingx,
    position: Dict,
    state: Dict[str, Any],
    config: Dict[str, Any],
    symbol: str,
) -> bool:
    """Adjust TP/SL for a single position slot.

    Determines expected TP/SL percentages based on the current config mode
    (dynamic per-pattern, dynamic universal, or static), then compares
    against the position's current TP/SL prices and re-places if needed.
    """
    # Resolve expected TP/SL percentages based on mode (before accessing entry_price)
    expected_tp_pct, expected_sl_pct, label = _resolve_expected_tpsl(position, config)
    if expected_tp_pct is None:
        return False

    entry_price = position['entry_price']
    direction = position['direction']

    # Calculate expected prices
    if direction == 'LONG':
        expected_tp_price = entry_price * (1 + expected_tp_pct / 100)
        expected_sl_price = entry_price * (1 - expected_sl_pct / 100)
    else:  # SHORT
        expected_tp_price = entry_price * (1 - expected_tp_pct / 100)
        expected_sl_price = entry_price * (1 + expected_sl_pct / 100)

    # Round to appropriate precision
    expected_tp_price = round(expected_tp_price, 1)
    expected_sl_price = round(expected_sl_price, 1)

    # Check if adjustment is needed (tolerance: $1)
    current_tp = position.get('tp_price', 0)
    current_sl = position.get('sl_price', 0)

    tp_diff = abs(current_tp - expected_tp_price)
    sl_diff = abs(current_sl - expected_sl_price)

    if tp_diff <= 1.0 and sl_diff <= 1.0:
        logger.debug(f"TP/SL already match config for {label}")
        return False

    slot_id = position.get('slot_id', 'unknown')
    logger.info(f"🔧 TP/SL adjustment needed for {label} (slot {slot_id}):")
    logger.info(f"   Current: TP=${current_tp:.1f}, SL=${current_sl:.1f}")
    logger.info(f"   Expected: TP=${expected_tp_price:.1f} ({expected_tp_pct}%), SL=${expected_sl_price:.1f} ({expected_sl_pct}%)")

    try:
        # Cancel existing orders
        _cancel_existing_tpsl_orders(exchange, position, symbol)

        # Update position with new prices
        position['tp_price'] = expected_tp_price
        position['sl_price'] = expected_sl_price

        # Place new orders (use remaining_quantity for scale-out partial fills)
        quantity = position.get('remaining_quantity', position['quantity'])
        close_side = 'sell' if direction == 'LONG' else 'buy'
        position_side = _get_position_side(config, direction)

        # Place TP order
        _place_single_tp_order(exchange, position, symbol, close_side, quantity, expected_tp_price, position_side)

        # Place SL order
        _place_sl_order(exchange, position, symbol, close_side, quantity, expected_sl_price, position_side)

        save_state(state)
        logger.info(f"✅ TP/SL adjusted successfully for {label} (slot {slot_id})")
        return True

    except Exception as e:
        logger.exception(f"Failed to adjust TP/SL: {e}")
        return False


def _resolve_expected_tpsl(
    position: Dict,
    config: Dict[str, Any],
) -> tuple:
    """Resolve expected TP/SL percentages based on config mode.

    Returns:
        (tp_pct, sl_pct, label) or (None, None, None) if pattern not found.
    """
    # Dynamic per-pattern mode: lookup from _dynamic_patterns_tpsl
    if config.get('_dynamic_tpsl_per_pattern'):
        pattern_name = extract_pattern_name(position.get('reason', ''))
        if not pattern_name:
            pattern_name = position.get('pattern_name')
        if not pattern_name:
            logger.debug("No pattern found in position, skipping TP/SL adjustment")
            return None, None, None
        pp_tpsl = config.get('_dynamic_patterns_tpsl', {})
        if pattern_name not in pp_tpsl:
            logger.debug(f"Pattern {pattern_name} not in dynamic per-pattern dict, skipping")
            return None, None, None
        tp_pct, sl_pct = pp_tpsl[pattern_name]
        return tp_pct, sl_pct, pattern_name

    # Dynamic universal mode: use config universal values
    if config.get('_dynamic_tpsl_universal'):
        tp_pct = config.get('_dynamic_tp', 0)
        sl_pct = config.get('_dynamic_sl', 0)
        if tp_pct <= 0 or sl_pct <= 0:
            return None, None, None
        return tp_pct, sl_pct, 'universal'

    # Static mode: use PATTERN_OPTIMAL_TPSL from constants
    pattern_name = extract_pattern_name(position.get('reason', ''))
    if not pattern_name:
        pattern_name = position.get('pattern_name')
    if not pattern_name:
        logger.debug("No pattern found in position, skipping TP/SL adjustment")
        return None, None, None
    if pattern_name not in PATTERN_OPTIMAL_TPSL:
        logger.debug(f"Pattern {pattern_name} not in PATTERN_OPTIMAL_TPSL, skipping")
        return None, None, None
    tp_pct, sl_pct = PATTERN_OPTIMAL_TPSL[pattern_name]
    return tp_pct, sl_pct, pattern_name


def _cancel_existing_tpsl_orders(
    exchange: ccxt.bingx,
    position: Dict,
    symbol: str,
    open_orders: Optional[List] = None,
) -> None:
    """Cancel existing TP/SL orders before placing new ones."""
    tp_order_id = position.get('tp_order_id')
    sl_order_id = position.get('sl_order_id')

    try:
        if open_orders is None:
            open_orders = exchange.fetch_open_orders(symbol)
        open_order_ids = {o.get('id') for o in open_orders}

        if tp_order_id and tp_order_id != _EXCHANGE_MANAGED and tp_order_id in open_order_ids:
            exchange.cancel_order(tp_order_id, symbol)
            logger.info(f"   Cancelled old TP order: {tp_order_id}")

        if sl_order_id and sl_order_id != _EXCHANGE_MANAGED and sl_order_id in open_order_ids:
            exchange.cancel_order(sl_order_id, symbol)
            logger.info(f"   Cancelled old SL order: {sl_order_id}")

        # Also cancel scale-out orders if any
        for stage in position.get('scale_out_stages', []):
            stage_id = stage.get('order_id')
            if stage_id and stage_id in open_order_ids and not stage.get('filled'):
                exchange.cancel_order(stage_id, symbol)
                logger.info(f"   Cancelled scale-out order: {stage_id}")

    except Exception as e:
        logger.warning(f"Error cancelling existing orders: {e}")


def verify_tp_sl_orders(
    exchange: ccxt.bingx,
    state: Dict[str, Any],
    config: Dict[str, Any],
) -> None:
    """
    Verify TP/SL orders exist and re-place if missing.

    Iterates all active position slots and verifies each one.

    Args:
        exchange: Exchange instance
        state: Bot state dictionary
        config: Bot configuration
    """
    positions = state.get('positions') or {}
    if not positions:
        return

    symbol = config['symbol']

    try:
        open_orders = exchange.fetch_open_orders(symbol)
        open_order_ids = {o.get('id'): o for o in open_orders}
        state_changed = False

        # Guard: skip directions where exchange qty < local sum
        # (per-slot fill detected — check_position_status will handle)
        skip_directions = set()
        try:
            exchange_positions = exchange.fetch_positions([symbol])
            exchange_qty_map = {}
            for pos in exchange_positions:
                contracts = float(pos.get('contracts', 0))
                if contracts > 0:
                    side = pos.get('side')
                    dir_label = 'LONG' if side == 'long' else 'SHORT'
                    exchange_qty_map[dir_label] = contracts

            for dir_label in ('LONG', 'SHORT'):
                local_qty = sum(
                    p.get('quantity', 0) for p in positions.values()
                    if p.get('direction') == dir_label
                )
                if local_qty <= 0:
                    continue
                exchange_qty = exchange_qty_map.get(dir_label, 0)
                if exchange_qty < local_qty - 0.0001:
                    skip_directions.add(dir_label)
                    logger.info(
                        f"⚠️ Verify: skipping {dir_label} "
                        f"(exchange={exchange_qty:.4f} < local={local_qty:.4f}) "
                        f"— check_position_status will handle"
                    )
        except Exception as e:
            logger.debug(f"Verify qty guard: fetch_positions failed: {e}")

        for position in positions.values():
            if position.get('direction') in skip_directions:
                continue
            position_side = _get_position_side(config, position['direction'])
            scale_out_enabled = position.get('scale_out_enabled', False)
            scale_out_stages = position.get('scale_out_stages', [])

            # Verify TP orders (scale-out or single)
            if scale_out_enabled and scale_out_stages:
                state_changed = _verify_scale_out_orders(
                    exchange, position, symbol, scale_out_stages, open_order_ids, position_side
                ) or state_changed
            else:
                state_changed = _verify_single_tp_order(
                    exchange, position, symbol, open_order_ids, position_side
                ) or state_changed

            # Verify SL order
            state_changed = _verify_sl_order(
                exchange, position, symbol, open_order_ids, position_side
            ) or state_changed

        if state_changed:
            save_state(state)

    except ccxt.NetworkError as e:
        logger.warning(f"Could not verify TP/SL orders (network): {e}")
    except ccxt.ExchangeError as e:
        logger.warning(f"Could not verify TP/SL orders (exchange): {e}")
    except Exception as e:
        logger.warning(f"Could not verify TP/SL orders: {e}")


def _verify_single_tp_order(
    exchange: ccxt.bingx,
    position: Dict,
    symbol: str,
    open_order_ids: Dict,
    position_side: str = 'BOTH',
) -> bool:
    """Verify and re-place missing single TP order (non-scale-out mode).

    Handles three cases:
    - tp_order_id exists but order is missing from exchange
    - tp_order_id was never set (initial placement failed)
    - tp_order_id is EXCHANGE_MANAGED (confirmed on exchange, ID unknown)
    """
    tp_order_id = position.get('tp_order_id')
    tp_price = position.get('tp_price')
    state_changed = False

    if not tp_price or tp_price <= 0:
        logger.warning("Cannot verify TP order: tp_price is missing or invalid")
        return False

    # Skip if already confirmed on exchange (crash recovery, unknown ID)
    if tp_order_id == _EXCHANGE_MANAGED:
        return False

    needs_tp = False
    if not tp_order_id:
        logger.warning("TP order was never placed — placing now")
        needs_tp = True
    elif tp_order_id not in open_order_ids:
        logger.warning("TP order missing from exchange, re-placing...")
        needs_tp = True

    if needs_tp:
        try:
            close_side = 'sell' if position['direction'] == 'LONG' else 'buy'
            tp_order = exchange.create_order(
                symbol=symbol,
                type='TAKE_PROFIT_MARKET',
                side=close_side,
                amount=position.get('remaining_quantity', position['quantity']),
                params={
                    'positionSide': position_side,
                    'stopPrice': position['tp_price'],
                }
            )
            position['tp_order_id'] = tp_order.get('id')
            logger.info(f"TP order {'placed' if not tp_order_id else 're-placed'}: {tp_order.get('id')}")
            state_changed = True
        except ccxt.ExchangeError as e:
            error_msg = str(e)
            if '110407' in error_msg:
                position['tp_order_id'] = _EXCHANGE_MANAGED
                logger.info("TP order confirmed on exchange (crash recovery — ID unknown, marking as managed)")
                state_changed = True
            elif '110413' in error_msg:
                position['tp_order_id'] = _EXCHANGE_MANAGED
                logger.warning("TP price already exceeded current price — skipping TP placement, position_monitor will handle")
                state_changed = True
            else:
                logger.error(f"Failed to place TP order (exchange error): {e}")
        except Exception as e:
            logger.exception(f"Failed to place TP order: {e}")

    return state_changed


def _verify_scale_out_orders(
    exchange: ccxt.bingx,
    position: Dict,
    symbol: str,
    scale_out_stages: List[Dict],
    open_order_ids: Dict,
    position_side: str = 'BOTH',
) -> bool:
    """Verify and re-place missing scale-out TP orders."""
    close_side = 'sell' if position['direction'] == 'LONG' else 'buy'
    state_changed = False

    for idx, stage in enumerate(scale_out_stages):
        if stage.get('filled', False):
            continue

        stage_order_id = stage.get('order_id')

        # Re-place if order was never placed (order_id=None) or missing from exchange
        if not stage_order_id or stage_order_id not in open_order_ids:
            action = "was never placed" if not stage_order_id else "missing from exchange"
            logger.warning(f"Stage {stage['stage']} TP order {action}, re-placing...")
            try:
                order_params = {'positionSide': position_side, 'stopPrice': stage['tp_price']}

                tp_order = exchange.create_order(
                    symbol=symbol,
                    type='TAKE_PROFIT_MARKET',
                    side=close_side,
                    amount=stage['quantity'],
                    params=order_params
                )
                stage['order_id'] = tp_order.get('id')
                logger.info(f"Stage {stage['stage']} TP re-placed: {tp_order.get('id')}")
                state_changed = True
            except ccxt.ExchangeError as e:
                logger.error(f"Failed to re-place Stage {stage['stage']} TP (exchange error): {e}")
            except Exception as e:
                logger.exception(f"Failed to re-place Stage {stage['stage']} TP: {e}")

    return state_changed


def _verify_sl_order(
    exchange: ccxt.bingx,
    position: Dict,
    symbol: str,
    open_order_ids: Dict,
    position_side: str = 'BOTH',
) -> bool:
    """Verify and re-place missing SL order.

    Handles three cases:
    - sl_order_id exists but order is missing from exchange (cancelled/expired)
    - sl_order_id was never set (initial placement failed) — CRITICAL safety gap
    - sl_order_id is EXCHANGE_MANAGED (confirmed on exchange, ID unknown)
    """
    sl_order_id = position.get('sl_order_id')
    sl_price = position.get('sl_price')
    state_changed = False

    if not sl_price or sl_price <= 0:
        logger.error("Cannot verify SL order: sl_price is missing or invalid — position UNPROTECTED")
        return False

    # Skip if already confirmed on exchange (crash recovery, unknown ID)
    if sl_order_id == _EXCHANGE_MANAGED:
        return False

    needs_sl = False
    if not sl_order_id:
        logger.warning("SL order was never placed — placing now for position protection")
        needs_sl = True
    elif sl_order_id not in open_order_ids:
        logger.warning("SL order missing from exchange, re-placing...")
        needs_sl = True

    if needs_sl:
        try:
            close_side = 'sell' if position['direction'] == 'LONG' else 'buy'
            sl_order = exchange.create_order(
                symbol=symbol,
                type='STOP_MARKET',
                side=close_side,
                amount=position.get('remaining_quantity', position['quantity']),
                params={
                    'positionSide': position_side,
                    'stopPrice': position['sl_price'],
                }
            )
            position['sl_order_id'] = sl_order.get('id')
            logger.info(f"SL order {'placed' if not sl_order_id else 're-placed'}: {sl_order.get('id')}")
            state_changed = True
        except ccxt.ExchangeError as e:
            error_msg = str(e)
            if '110406' in error_msg:
                position['sl_order_id'] = _EXCHANGE_MANAGED
                logger.info("SL order confirmed on exchange (crash recovery — ID unknown, marking as managed)")
                state_changed = True
            else:
                logger.error(f"Failed to place SL order (exchange error): {e}")
        except Exception as e:
            logger.exception(f"Failed to place SL order: {e}")

    return state_changed


def cancel_remaining_orders(
    exchange: ccxt.bingx,
    state: Dict[str, Any],
    config: Dict[str, Any],
    position: Optional[Dict] = None,
) -> None:
    """
    Cancel all remaining orders for a position slot (or all slots).

    Args:
        exchange: Exchange instance
        state: Bot state dictionary
        config: Bot configuration
        position: Specific position slot dict. If None, cancels orders for all slots.
    """
    if position is not None:
        slots = [position]
    else:
        slots = list((state.get('positions') or {}).values())

    if not slots:
        return

    symbol = config['symbol']

    # Collect all order IDs to cancel across all target slots
    orders_to_cancel = []
    for slot in slots:
        tp_order_id = slot.get('tp_order_id')
        sl_order_id = slot.get('sl_order_id')

        if tp_order_id and tp_order_id != _EXCHANGE_MANAGED:
            orders_to_cancel.append(('TP', tp_order_id))
        if sl_order_id and sl_order_id != _EXCHANGE_MANAGED:
            orders_to_cancel.append(('SL', sl_order_id))

        for stage in slot.get('scale_out_stages', []):
            if stage.get('order_id') and not stage.get('filled'):
                orders_to_cancel.append(('Scale-out TP', stage['order_id']))

    if not orders_to_cancel:
        return

    try:
        open_orders = exchange.fetch_open_orders(symbol)
        open_order_ids = {o.get('id') for o in open_orders}

        for order_type, order_id in orders_to_cancel:
            if order_id in open_order_ids:
                try:
                    exchange.cancel_order(order_id, symbol)
                    logger.info(f"🗑️ Cancelled {order_type} order: {order_id}")
                except ccxt.OrderNotFound:
                    logger.debug(f"{order_type} order already filled/cancelled: {order_id}")
                except ccxt.ExchangeError as e:
                    logger.warning(f"⚠️ Failed to cancel {order_type} order {order_id}: {e}")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to cancel {order_type} order {order_id}: {e}")

    except ccxt.NetworkError as e:
        logger.error(f"Failed to cancel remaining orders (network): {e}")
    except ccxt.ExchangeError as e:
        logger.error(f"Failed to cancel remaining orders (exchange): {e}")
    except Exception as e:
        logger.exception(f"Failed to cancel remaining orders: {e}")


def _verify_emergency_sl(
    exchange: ccxt.bingx,
    state: Dict[str, Any],
    config: Dict[str, Any],
) -> None:
    """Verify emergency SL price matches current slot SL prices.

    Compares the expected worst SL price against the live emergency SL order.
    Re-places if they differ (e.g., after TP/SL adjustment or state recovery).
    """
    positions = state.get('positions') or {}
    if not positions:
        return

    expected_price = _get_worst_sl_price(state)
    if not expected_price:
        return

    order_id = state.get('emergency_sl_order_id')
    if not order_id or order_id == _EXCHANGE_MANAGED:
        # No existing emergency SL or exchange-managed → place fresh
        logger.info(f"🛡️ Emergency SL missing or exchange-managed — placing at ${expected_price:.1f}")
        place_emergency_sl(exchange, state, config)
        return

    # Check if the existing order matches expected price
    try:
        symbol = config.get('symbol', 'BTC-USDT')
        open_orders = exchange.fetch_open_orders(symbol)
        current_price = None
        for o in open_orders:
            if o.get('id') == order_id:
                current_price = float(o.get('stopPrice', 0) or o.get('price', 0))
                break

        if current_price is None:
            # Order not found on exchange — re-place
            logger.warning(f"🛡️ Emergency SL order {order_id} not found on exchange — re-placing")
            state['emergency_sl_order_id'] = None
            place_emergency_sl(exchange, state, config)
            return

        if abs(current_price - expected_price) > 1.0:
            logger.info(
                f"🛡️ Emergency SL price mismatch: current=${current_price:.1f}, "
                f"expected=${expected_price:.1f} — updating"
            )
            update_emergency_sl(exchange, state, config)
        else:
            logger.debug(f"Emergency SL verified: ${current_price:.1f} matches expected ${expected_price:.1f}")

    except Exception as e:
        logger.warning(f"Failed to verify emergency SL, re-placing: {e}")
        update_emergency_sl(exchange, state, config)


# ─── Emergency SL (v1.29.0: whole-position safety net) ───

def _get_worst_sl_price(state: Dict[str, Any]) -> Optional[float]:
    """Get the worst (most protective) SL price across all active slots.

    LONG: min(all sl_prices) - buffer  (lowest = worst case)
    SHORT: max(all sl_prices) + buffer (highest = worst case)
    """
    from .constants import EMERGENCY_SL_BUFFER_PCT

    positions = state.get('positions') or {}
    if not positions:
        return None

    direction = state.get('active_direction')
    sl_prices = [s.get('sl_price', 0) for s in positions.values() if s.get('sl_price')]
    if not sl_prices:
        return None

    if direction == 'LONG':
        worst = min(sl_prices)
        return round(worst * (1 - EMERGENCY_SL_BUFFER_PCT), 1)
    else:  # SHORT
        worst = max(sl_prices)
        return round(worst * (1 + EMERGENCY_SL_BUFFER_PCT), 1)


def place_emergency_sl(
    exchange: ccxt.bingx,
    state: Dict[str, Any],
    config: Dict[str, Any],
) -> None:
    """Place a closePosition emergency SL protecting the entire NET position."""
    positions = state.get('positions') or {}
    if not positions:
        return

    worst_sl = _get_worst_sl_price(state)
    if not worst_sl:
        logger.warning("Cannot place emergency SL: no valid SL prices in slots")
        return

    direction = state.get('active_direction')
    close_side = 'sell' if direction == 'LONG' else 'buy'
    symbol = config['symbol']

    total_qty = round(
        sum(p.get('quantity', 0) for p in positions.values()),
        QUANTITY_ROUND_DECIMALS,
    )
    if total_qty <= 0:
        logger.warning("Cannot place emergency SL: total quantity is 0")
        return

    try:
        # Use amount-based STOP_MARKET (same proven pattern as per-slot SL orders)
        # Note: closePosition:True + amount causes BingX API rejection
        order = exchange.create_order(
            symbol=symbol,
            type='STOP_MARKET',
            side=close_side,
            amount=total_qty,
            params={
                'positionSide': 'BOTH',
                'stopPrice': worst_sl,
            }
        )
        state['emergency_sl_order_id'] = order.get('id')
        save_state(state)
        logger.info(f"🛡️ Emergency SL placed: {order.get('id')} @ ${worst_sl:.1f} (qty={total_qty})")
    except ccxt.ExchangeError as e:
        error_msg = str(e)
        if '110406' in error_msg:
            state['emergency_sl_order_id'] = _EXCHANGE_MANAGED
            logger.info("Emergency SL already exists on exchange — marking as managed")
        else:
            logger.error(f"Failed to place emergency SL: {e}")
    except Exception as e:
        logger.exception(f"Failed to place emergency SL: {e}")


def cancel_emergency_sl(
    exchange: ccxt.bingx,
    state: Dict[str, Any],
    config: Dict[str, Any],
) -> None:
    """Cancel the emergency SL order."""
    order_id = state.get('emergency_sl_order_id')
    if not order_id or order_id == _EXCHANGE_MANAGED:
        state['emergency_sl_order_id'] = None
        return

    symbol = config['symbol']
    try:
        exchange.cancel_order(order_id, symbol)
        logger.info(f"🛡️ Emergency SL cancelled: {order_id}")
    except ccxt.OrderNotFound:
        logger.debug(f"Emergency SL already gone: {order_id}")
    except Exception as e:
        logger.warning(f"Failed to cancel emergency SL {order_id}: {e}")
    finally:
        state['emergency_sl_order_id'] = None


def update_emergency_sl(
    exchange: ccxt.bingx,
    state: Dict[str, Any],
    config: Dict[str, Any],
) -> None:
    """Re-calculate and re-place emergency SL after slot add/remove.

    Cancels old SL first, then places new one. Brief gap is acceptable
    because per-slot SLs still protect during the transition. Without
    closePosition, overlapping STOP_MARKET orders risk double-execution.
    """
    positions = state.get('positions') or {}
    if not positions:
        cancel_emergency_sl(exchange, state, config)
        return

    old_order_id = state.get('emergency_sl_order_id')
    # Cancel old emergency SL first (per-slot SLs still protect during brief gap)
    if old_order_id and old_order_id != _EXCHANGE_MANAGED:
        try:
            symbol = config.get('symbol', 'BTC-USDT')
            exchange.cancel_order(old_order_id, symbol)
            logger.debug(f"Cancelled old emergency SL {old_order_id}")
        except (ccxt.OrderNotFound, ccxt.InvalidOrder):
            logger.debug(f"Old emergency SL already gone: {old_order_id}")
        except Exception as e:
            logger.warning(f"Failed to cancel old emergency SL {old_order_id}: {e}")
    state['emergency_sl_order_id'] = None
    # Place new emergency SL
    place_emergency_sl(exchange, state, config)