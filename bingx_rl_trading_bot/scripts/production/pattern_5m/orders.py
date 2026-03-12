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

# Buffer for SL recalculation when original SL is already breached (v1.35.4)
_SL_BREACH_BUFFER_PCT = 0.003  # 0.3% from current price


def _recalculate_breached_sl(
    exchange: ccxt.bingx, symbol: str, direction: str, old_sl: float,
) -> Optional[float]:
    """Recalculate SL when original price is already past current market price.

    For SHORT: SL (buy stop) must be above current price.
    For LONG: SL (sell stop) must be below current price.
    When the stored SL violates this, place SL at current_price ± buffer.

    Returns new SL price, or None if current price cannot be fetched.
    """
    try:
        ticker = exchange.fetch_ticker(symbol)
        current_price = ticker['last']
    except Exception as e:
        logger.error(f"Cannot fetch price for SL recalculation: {e}")
        return None

    if direction == 'SHORT':
        new_sl = round(current_price * (1 + _SL_BREACH_BUFFER_PCT), 1)
    else:
        new_sl = round(current_price * (1 - _SL_BREACH_BUFFER_PCT), 1)

    logger.warning(
        f"⚠️ SL BREACHED: {direction} SL ${old_sl:.1f} already past current ${current_price:.1f}. "
        f"Adjusting to ${new_sl:.1f} ({_SL_BREACH_BUFFER_PCT * 100:.1f}% from current)"
    )
    return new_sl


def _get_position_side(config: Dict[str, Any], direction: str) -> str:
    """Get positionSide param for BingX API.

    One-Way mode: always 'BOTH'.
    Hedge mode: 'LONG' or 'SHORT' matching direction.
    """
    if config.get('position_mode') == 'hedge':
        return 'LONG' if direction == 'LONG' else 'SHORT'
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
        elif '110412' in error_msg or '110411' in error_msg:
            # SL price already breached — recalculate from current price (v1.35.4, v1.36.6: +110411)
            direction = position.get('direction', '')
            new_sl = _recalculate_breached_sl(exchange, symbol, direction, sl_price)
            if new_sl:
                position['sl_price'] = new_sl
                try:
                    sl_order = exchange.create_order(
                        symbol=symbol, type='STOP_MARKET', side=close_side,
                        amount=quantity,
                        params={'positionSide': position_side, 'stopPrice': new_sl},
                    )
                    position['sl_order_id'] = sl_order.get('id')
                    logger.info(f"SL order placed (breach-adjusted): {sl_order.get('id')} @ ${new_sl}")
                except Exception as retry_e:
                    logger.error(f"SL order retry failed after breach adjustment: {retry_e}")
        else:
            logger.warning(f"SL order failed (exchange error): {e}")
    except Exception as e:
        logger.warning(f"SL order failed: {e}")


def update_single_sl(
    exchange: ccxt.bingx,
    position: Dict,
    config: Dict[str, Any],
    new_sl_price: float,
) -> bool:
    """Place new SL first, then cancel old one — eliminating protection gap.

    Used by cascade SL tightening (v1.41.0) to update a single slot's
    SL order after a correlated SL exit.

    v1.56.2: Place-first, Cancel-after pattern (matching update_emergency_sl).
    Returns True if the new SL was placed (or marked managed).
    """
    symbol = config['symbol']
    direction = position.get('direction', '')
    close_side = 'sell' if direction == 'LONG' else 'buy'
    position_side = _get_position_side(config, direction)
    quantity = position.get('remaining_quantity', position.get('quantity', 0))

    old_sl_id = position.get('sl_order_id')

    # Place new SL FIRST — position is protected immediately
    position['sl_order_id'] = None
    position['sl_price'] = new_sl_price
    _place_sl_order(exchange, position, symbol, close_side, quantity, new_sl_price, position_side)

    new_placed = position.get('sl_order_id') is not None

    # THEN cancel old SL — protection gap = 0
    if old_sl_id and old_sl_id != _EXCHANGE_MANAGED:
        try:
            exchange.cancel_order(old_sl_id, symbol)
            logger.info(f"CASCADE: Cancelled old SL {old_sl_id} (new SL active)")
        except (ccxt.OrderNotFound, ccxt.InvalidOrder):
            logger.debug(f"CASCADE: Old SL {old_sl_id} already gone")
        except Exception as e:
            logger.warning(f"CASCADE: Failed to cancel old SL {old_sl_id}: {e}")

    return new_placed


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

    Uses calculate_tp_sl() — the single source of truth — to determine expected
    TP/SL prices (with ATR scaling and slippage buffer), then compares against
    the position's current TP/SL prices and re-places if needed.
    """
    from .position_open import calculate_tp_sl  # local import to avoid circular

    # Resolve pattern (needed for per-pattern and static modes, optional for universal)
    pattern = position.get('pattern_name') or extract_pattern_name(position.get('reason', ''))

    # Verify pattern/mode compatibility before proceeding
    if config.get('_dynamic_tpsl_per_pattern'):
        pp_tpsl = config.get('_dynamic_patterns_tpsl', {})
        if not pattern or pattern not in pp_tpsl:
            # v1.59.2: Don't skip — calculate_tp_sl uses median fallback for unknown patterns.
            # Skipping leaves positions with dangerously tight defaults (1%/1%) forever.
            logger.info(
                f"Pattern {'missing' if not pattern else f'{pattern} not in dict'} — "
                f"TP/SL adjustment will use median fallback"
            )
    elif config.get('_dynamic_tpsl_universal'):
        pass  # Universal mode doesn't require a pattern
    else:
        # Static mode: need pattern in PATTERN_OPTIMAL_TPSL
        if not pattern:
            logger.debug("No pattern found in position, skipping TP/SL adjustment")
            return False
        if pattern not in PATTERN_OPTIMAL_TPSL:
            logger.debug(f"Pattern {pattern} not in PATTERN_OPTIMAL_TPSL, skipping adjustment")
            return False

    entry_price = position['entry_price']
    direction = position['direction']
    dir_mult = 1 if direction == 'LONG' else -1
    vol_mult = position.get('vol_mult', 1.0)
    strategy = config.get('strategy', {})

    # Use single source of truth for TP/SL (applies ATR scaling + slippage buffer)
    expected_tp_price, expected_sl_price, tp_pct_adj, sl_pct_adj = calculate_tp_sl(
        entry_price, dir_mult, strategy, vol_mult=vol_mult, pattern=pattern, config=config
    )

    # Check if adjustment is needed (tolerance: $1)
    current_tp = position.get('tp_price', 0)
    current_sl = position.get('sl_price', 0)

    tp_diff = abs(current_tp - expected_tp_price)
    sl_diff = abs(current_sl - expected_sl_price)

    if tp_diff <= 1.0 and sl_diff <= 1.0:
        logger.debug(f"TP/SL already match config for {pattern} (vol_mult={vol_mult:.4f})")
        return False

    slot_id = position.get('slot_id', 'unknown')
    logger.info(f"🔧 TP/SL adjustment needed for {pattern} (slot {slot_id}):")
    logger.info(f"   Current: TP=${current_tp:.1f}, SL=${current_sl:.1f}")
    logger.info(f"   Expected: TP=${expected_tp_price:.1f} ({tp_pct_adj:.2f}%), SL=${expected_sl_price:.1f} ({sl_pct_adj:.2f}%)")
    logger.info(f"   vol_mult={vol_mult:.4f}")

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
        logger.info(f"✅ TP/SL adjusted successfully for {pattern} (slot {slot_id})")
        return True

    except Exception as e:
        logger.exception(f"Failed to adjust TP/SL: {e}")
        return False

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
            elif '110412' in error_msg or '110411' in error_msg:
                # SL price already breached — recalculate from current price (v1.35.4, v1.36.6: +110411)
                direction = position.get('direction', '')
                new_sl = _recalculate_breached_sl(exchange, symbol, direction, position['sl_price'])
                if new_sl:
                    position['sl_price'] = new_sl
                    try:
                        close_side = 'sell' if direction == 'LONG' else 'buy'
                        sl_order = exchange.create_order(
                            symbol=symbol, type='STOP_MARKET', side=close_side,
                            amount=position.get('remaining_quantity', position['quantity']),
                            params={'positionSide': position_side, 'stopPrice': new_sl},
                        )
                        position['sl_order_id'] = sl_order.get('id')
                        logger.info(f"SL order placed (breach-adjusted): {sl_order.get('id')} @ ${new_sl}")
                        state_changed = True
                    except Exception as retry_e:
                        logger.error(f"SL order retry failed after breach adjustment: {retry_e}")
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


def _verify_emergency_sl_for_direction(
    exchange: ccxt.bingx,
    state: Dict[str, Any],
    config: Dict[str, Any],
    direction: str,
    open_orders: List[Dict],
) -> None:
    """Verify emergency SL for one direction against live orders."""
    expected_price = _get_worst_sl_price_for_direction(state, direction)
    if not expected_price:
        return

    order_id = _get_emergency_sl_id(state, config, direction)
    if not order_id or order_id == _EXCHANGE_MANAGED:
        logger.info(f"🛡️ Emergency SL ({direction}) missing — placing at ${expected_price:.1f}")
        _place_emergency_sl_for_direction(exchange, state, config, direction)
        return

    current_price = None
    for o in open_orders:
        if o.get('id') == order_id:
            current_price = float(o.get('stopPrice', 0) or o.get('price', 0))
            break

    if current_price is None:
        logger.warning(f"🛡️ Emergency SL ({direction}) order {order_id} not found — re-placing")
        _set_emergency_sl_id(state, config, direction, None)
        _place_emergency_sl_for_direction(exchange, state, config, direction)
        return

    if abs(current_price - expected_price) > 1.0:
        logger.info(
            f"🛡️ Emergency SL ({direction}) price mismatch: "
            f"current=${current_price:.1f}, expected=${expected_price:.1f} — updating"
        )
        update_emergency_sl(exchange, state, config)
    else:
        logger.debug(f"Emergency SL ({direction}) verified: ${current_price:.1f}")


def _verify_emergency_sl(
    exchange: ccxt.bingx,
    state: Dict[str, Any],
    config: Dict[str, Any],
) -> None:
    """Verify emergency SL price(s) match current slot SL prices.

    Hedge mode: verifies per-direction.
    One-Way mode: verifies single emergency SL.
    """
    positions = state.get('positions') or {}
    if not positions:
        return

    try:
        symbol = config.get('symbol', 'BTC-USDT')
        open_orders = exchange.fetch_open_orders(symbol)
    except Exception as e:
        logger.warning(f"Failed to fetch open orders for emergency SL verify: {e}")
        return

    if config.get('position_mode') == 'hedge':
        active_dirs = {p.get('direction') for p in positions.values() if p.get('direction')}
        for d in active_dirs:
            try:
                _verify_emergency_sl_for_direction(exchange, state, config, d, open_orders)
            except Exception as e:
                logger.warning(f"Failed to verify emergency SL ({d}), re-placing: {e}")
                update_emergency_sl(exchange, state, config)
    else:
        direction = state.get('active_direction')
        if not direction:
            return
        try:
            _verify_emergency_sl_for_direction(exchange, state, config, direction, open_orders)
        except Exception as e:
            logger.warning(f"Failed to verify emergency SL, re-placing: {e}")
            update_emergency_sl(exchange, state, config)


# ─── Emergency SL (v1.30.0: per-direction safety net for hedge mode) ───

def _get_worst_sl_price_for_direction(
    state: Dict[str, Any], direction: str,
) -> Optional[float]:
    """Get worst SL price for a specific direction's slots.

    LONG: min(sl_prices) - buffer  (lowest = worst case)
    SHORT: max(sl_prices) + buffer (highest = worst case)
    """
    from .constants import EMERGENCY_SL_BUFFER_PCT

    positions = state.get('positions') or {}
    sl_prices = [
        s.get('sl_price', 0) for s in positions.values()
        if s.get('sl_price') and s.get('direction') == direction
    ]
    if not sl_prices:
        return None

    if direction == 'LONG':
        worst = min(sl_prices)
        return round(worst * (1 - EMERGENCY_SL_BUFFER_PCT), 1)
    else:  # SHORT
        worst = max(sl_prices)
        return round(worst * (1 + EMERGENCY_SL_BUFFER_PCT), 1)


def _get_worst_sl_price(state: Dict[str, Any]) -> Optional[float]:
    """Get worst SL price across all active slots (One-Way compat wrapper)."""
    direction = state.get('active_direction')
    if not direction:
        return None
    return _get_worst_sl_price_for_direction(state, direction)


def _place_emergency_sl_for_direction(
    exchange: ccxt.bingx,
    state: Dict[str, Any],
    config: Dict[str, Any],
    direction: str,
) -> None:
    """Place a single emergency SL for one direction."""
    positions = state.get('positions') or {}
    dir_slots = [p for p in positions.values() if p.get('direction') == direction]
    if not dir_slots:
        return

    worst_sl = _get_worst_sl_price_for_direction(state, direction)
    if not worst_sl:
        logger.warning(f"Cannot place emergency SL ({direction}): no valid SL prices")
        return

    close_side = 'sell' if direction == 'LONG' else 'buy'
    position_side = _get_position_side(config, direction)
    symbol = config['symbol']

    total_qty = round(
        sum(p.get('quantity', 0) for p in dir_slots),
        QUANTITY_ROUND_DECIMALS,
    )
    if total_qty <= 0:
        logger.warning(f"Cannot place emergency SL ({direction}): total quantity is 0")
        return

    try:
        order = exchange.create_order(
            symbol=symbol,
            type='STOP_MARKET',
            side=close_side,
            amount=total_qty,
            params={
                'positionSide': position_side,
                'stopPrice': worst_sl,
                'closePosition': 'true',
            }
        )
        order_id = order.get('id')
        _set_emergency_sl_id(state, config, direction, order_id)
        save_state(state)
        logger.info(f"🛡️ Emergency SL ({direction}) placed: {order_id} @ ${worst_sl:.1f} (closePosition=true, qty={total_qty})")
    except ccxt.ExchangeError as e:
        error_msg = str(e)
        if '110406' in error_msg:
            _set_emergency_sl_id(state, config, direction, _EXCHANGE_MANAGED)
            logger.info(f"Emergency SL ({direction}) already exists on exchange — marking as managed")
        elif '110412' in error_msg or '110411' in error_msg:
            # Emergency SL price already breached — recalculate (v1.35.4, v1.36.6: +110411)
            new_sl = _recalculate_breached_sl(exchange, symbol, direction, worst_sl)
            if new_sl:
                # Update all slots' sl_price so future calculations use the adjusted value
                for slot in dir_slots:
                    slot['sl_price'] = new_sl
                try:
                    order = exchange.create_order(
                        symbol=symbol, type='STOP_MARKET', side=close_side,
                        amount=total_qty,
                        params={
                            'positionSide': position_side,
                            'stopPrice': new_sl,
                            'closePosition': 'true',
                        },
                    )
                    order_id = order.get('id')
                    _set_emergency_sl_id(state, config, direction, order_id)
                    save_state(state)
                    logger.info(f"🛡️ Emergency SL ({direction}) placed (breach-adjusted): {order_id} @ ${new_sl:.1f}")
                except Exception as retry_e:
                    logger.critical(
                        f"EMERGENCY SL ({direction}) PLACEMENT FAILED — POSITION UNPROTECTED! "
                        f"Slots: {len(dir_slots)}, TotalQty: {total_qty}. Error: {retry_e}"
                    )
        elif '110424' in error_msg:
            # Order size > available amount — per-slot SLs already cover position
            logger.warning(
                f"Emergency SL ({direction}) rejected (110424 size>available). "
                f"Per-slot SLs active — marking as exchange-managed."
            )
            _set_emergency_sl_id(state, config, direction, _EXCHANGE_MANAGED)
        else:
            logger.critical(
                f"EMERGENCY SL ({direction}) PLACEMENT FAILED — POSITION UNPROTECTED! "
                f"Slots: {len(dir_slots)}, TotalQty: {total_qty}. Error: {e}"
            )
    except Exception as e:
        logger.critical(
            f"EMERGENCY SL ({direction}) PLACEMENT FAILED — POSITION UNPROTECTED! "
            f"Slots: {len(dir_slots)}, TotalQty: {total_qty}. Error: {e}"
        )


def _set_emergency_sl_id(
    state: Dict[str, Any], config: Dict[str, Any],
    direction: str, order_id: Optional[str],
) -> None:
    """Store emergency SL order ID in the appropriate state field."""
    if config.get('position_mode') == 'hedge':
        sl_orders = state.setdefault('emergency_sl_orders', {})
        sl_orders[direction] = order_id
    else:
        state['emergency_sl_order_id'] = order_id


def _get_emergency_sl_id(
    state: Dict[str, Any], config: Dict[str, Any], direction: str,
) -> Optional[str]:
    """Read emergency SL order ID from the appropriate state field."""
    if config.get('position_mode') == 'hedge':
        return (state.get('emergency_sl_orders') or {}).get(direction)
    return state.get('emergency_sl_order_id')


def place_emergency_sl(
    exchange: ccxt.bingx,
    state: Dict[str, Any],
    config: Dict[str, Any],
) -> None:
    """Place emergency SL(s) protecting all active positions.

    Hedge mode: places one SL per active direction.
    One-Way mode: places one SL for the single active direction.
    """
    positions = state.get('positions') or {}
    if not positions:
        return

    if config.get('position_mode') == 'hedge':
        active_dirs = {p.get('direction') for p in positions.values() if p.get('direction')}
        for d in active_dirs:
            _place_emergency_sl_for_direction(exchange, state, config, d)
    else:
        direction = state.get('active_direction')
        if direction:
            _place_emergency_sl_for_direction(exchange, state, config, direction)


def _cancel_emergency_sl_for_direction(
    exchange: ccxt.bingx,
    state: Dict[str, Any],
    config: Dict[str, Any],
    direction: str,
) -> None:
    """Cancel emergency SL for one direction."""
    order_id = _get_emergency_sl_id(state, config, direction)
    if not order_id or order_id == _EXCHANGE_MANAGED:
        _set_emergency_sl_id(state, config, direction, None)
        return

    symbol = config['symbol']
    try:
        exchange.cancel_order(order_id, symbol)
        logger.info(f"🛡️ Emergency SL ({direction}) cancelled: {order_id}")
    except ccxt.OrderNotFound:
        logger.debug(f"Emergency SL ({direction}) already gone: {order_id}")
    except Exception as e:
        logger.warning(f"Failed to cancel emergency SL ({direction}) {order_id}: {e}")
    finally:
        _set_emergency_sl_id(state, config, direction, None)


def cancel_emergency_sl(
    exchange: ccxt.bingx,
    state: Dict[str, Any],
    config: Dict[str, Any],
) -> None:
    """Cancel all emergency SL orders."""
    if config.get('position_mode') == 'hedge':
        for direction in ['LONG', 'SHORT']:
            _cancel_emergency_sl_for_direction(exchange, state, config, direction)
    else:
        _cancel_emergency_sl_for_direction(exchange, state, config,
                                            state.get('active_direction') or 'LONG')
        state['emergency_sl_order_id'] = None


def update_emergency_sl(
    exchange: ccxt.bingx,
    state: Dict[str, Any],
    config: Dict[str, Any],
) -> None:
    """Re-calculate and re-place emergency SL(s) after slot add/remove.

    Place-first, cancel-after: new closePosition SL is placed before
    cancelling old one, eliminating the protection gap (v1.36.6).
    """
    positions = state.get('positions') or {}
    if not positions:
        cancel_emergency_sl(exchange, state, config)
        return

    if config.get('position_mode') == 'hedge':
        active_dirs = {p.get('direction') for p in positions.values() if p.get('direction')}
        # Cancel removed directions
        for direction in ['LONG', 'SHORT']:
            if direction not in active_dirs:
                _cancel_emergency_sl_for_direction(exchange, state, config, direction)
        # Place-first, cancel-after for active directions
        for direction in active_dirs:
            old_id = _get_emergency_sl_id(state, config, direction)
            # Place new SL first (closePosition=true, no qty conflict)
            _set_emergency_sl_id(state, config, direction, None)
            _place_emergency_sl_for_direction(exchange, state, config, direction)
            # Then cancel old SL (protection gap = 0)
            if old_id and old_id != _EXCHANGE_MANAGED:
                try:
                    exchange.cancel_order(old_id, config.get('symbol', 'BTC-USDT'))
                    logger.debug(f"Cancelled old emergency SL ({direction}) {old_id}")
                except (ccxt.OrderNotFound, ccxt.InvalidOrder):
                    logger.debug(f"Old emergency SL ({direction}) already gone: {old_id}")
                except Exception as e:
                    logger.warning(f"Failed to cancel old emergency SL ({direction}) {old_id}: {e}")
    else:
        # One-Way: same as before
        old_order_id = state.get('emergency_sl_order_id')
        if old_order_id and old_order_id != _EXCHANGE_MANAGED:
            try:
                exchange.cancel_order(old_order_id, config.get('symbol', 'BTC-USDT'))
                logger.debug(f"Cancelled old emergency SL {old_order_id}")
            except (ccxt.OrderNotFound, ccxt.InvalidOrder):
                logger.debug(f"Old emergency SL already gone: {old_order_id}")
            except Exception as e:
                logger.warning(f"Failed to cancel old emergency SL {old_order_id}: {e}")
        state['emergency_sl_order_id'] = None
        place_emergency_sl(exchange, state, config)