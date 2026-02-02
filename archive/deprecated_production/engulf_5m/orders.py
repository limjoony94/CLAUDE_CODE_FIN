"""
Engulf 5m Bot - Order Management
Place and manage TP/SL orders.
"""

import logging
from typing import Dict, Any, List

import ccxt

from .state import save_state

logger = logging.getLogger('engulf_5m')


def place_tp_sl_orders(
    exchange: ccxt.bingx,
    state: Dict[str, Any],
    config: Dict[str, Any],
) -> None:
    """
    Place TP and SL orders for the current position.

    Args:
        exchange: Exchange instance
        state: Bot state dictionary
        config: Bot configuration
    """
    if not state.get('position'):
        return

    position = state['position']
    symbol = config['symbol']

    try:
        direction = position['direction']
        quantity = position['quantity']
        tp_price = position['tp_price']
        sl_price = position['sl_price']
        close_side = 'sell' if direction == 'LONG' else 'buy'

        scale_out_enabled = position.get('scale_out_enabled', False)
        scale_out_stages = position.get('scale_out_stages', [])

        # Handle scale-out TP orders
        if scale_out_enabled and scale_out_stages:
            _place_scale_out_orders(exchange, position, symbol, close_side, scale_out_stages)
        else:
            _place_single_tp_order(exchange, position, symbol, close_side, quantity, tp_price)

        # Always place SL order
        _place_sl_order(exchange, position, symbol, close_side, quantity, sl_price)

        save_state(state)

    except ccxt.NetworkError as e:
        logger.error(f"Network error placing TP/SL orders: {e}")
    except ccxt.ExchangeError as e:
        logger.error(f"Exchange error placing TP/SL orders: {e}")
    except Exception as e:
        logger.error(f"Failed to place TP/SL orders: {e}")


def _place_scale_out_orders(
    exchange: ccxt.bingx,
    position: Dict,
    symbol: str,
    close_side: str,
    scale_out_stages: List[Dict],
) -> None:
    """Place staged TP orders for scale-out."""
    logger.info(f"📈 Scale-out: placing {len(scale_out_stages)} staged TP orders")
    num_stages = len(scale_out_stages)

    for idx, stage in enumerate(scale_out_stages):
        try:
            stage_qty = stage['quantity']
            stage_tp_price = stage['tp_price']
            is_last_stage = (idx == num_stages - 1)

            order_params = {
                'positionSide': 'BOTH',
                'stopPrice': stage_tp_price,
            }
            if is_last_stage:
                order_params['closePosition'] = True

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
) -> None:
    """Place a single TP order."""
    try:
        tp_order = exchange.create_order(
            symbol=symbol,
            type='TAKE_PROFIT_MARKET',
            side=close_side,
            amount=quantity,
            params={
                'positionSide': 'BOTH',
                'stopPrice': tp_price,
                'closePosition': True,
            }
        )
        position['tp_order_id'] = tp_order.get('id')
        logger.info(f"TP order placed: {tp_order.get('id')} @ ${tp_price}")
    except ccxt.InsufficientFunds as e:
        logger.warning(f"TP order failed (insufficient funds): {e}")
    except ccxt.ExchangeError as e:
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
) -> None:
    """Place a SL order."""
    try:
        sl_order = exchange.create_order(
            symbol=symbol,
            type='STOP_MARKET',
            side=close_side,
            amount=quantity,
            params={
                'positionSide': 'BOTH',
                'stopPrice': sl_price,
                'closePosition': True,
            }
        )
        position['sl_order_id'] = sl_order.get('id')
        logger.info(f"SL order placed: {sl_order.get('id')} @ ${sl_price}")
    except ccxt.InsufficientFunds as e:
        logger.warning(f"SL order failed (insufficient funds): {e}")
    except ccxt.ExchangeError as e:
        logger.warning(f"SL order failed (exchange error): {e}")
    except Exception as e:
        logger.warning(f"SL order failed: {e}")


def verify_tp_sl_orders(
    exchange: ccxt.bingx,
    state: Dict[str, Any],
    config: Dict[str, Any],
) -> None:
    """
    Verify TP/SL orders exist and re-place if missing.

    Args:
        exchange: Exchange instance
        state: Bot state dictionary
        config: Bot configuration
    """
    if not state.get('position'):
        return

    position = state['position']
    symbol = config['symbol']
    scale_out_enabled = position.get('scale_out_enabled', False)
    scale_out_stages = position.get('scale_out_stages', [])

    try:
        open_orders = exchange.fetch_open_orders(symbol)
        open_order_ids = {o.get('id'): o for o in open_orders}
        state_changed = False

        # Verify scale-out orders
        if scale_out_enabled and scale_out_stages:
            state_changed = _verify_scale_out_orders(
                exchange, position, symbol, scale_out_stages, open_order_ids
            ) or state_changed

        # Verify SL order
        state_changed = _verify_sl_order(
            exchange, position, symbol, open_order_ids
        ) or state_changed

        if state_changed:
            save_state(state)

    except ccxt.NetworkError as e:
        logger.debug(f"Could not verify TP/SL orders (network): {e}")
    except ccxt.ExchangeError as e:
        logger.debug(f"Could not verify TP/SL orders (exchange): {e}")
    except Exception as e:
        logger.debug(f"Could not verify TP/SL orders: {e}")


def _verify_scale_out_orders(
    exchange: ccxt.bingx,
    position: Dict,
    symbol: str,
    scale_out_stages: List[Dict],
    open_order_ids: Dict,
) -> bool:
    """Verify and re-place missing scale-out TP orders."""
    num_stages = len(scale_out_stages)
    close_side = 'sell' if position['direction'] == 'LONG' else 'buy'
    state_changed = False

    for idx, stage in enumerate(scale_out_stages):
        if stage.get('filled', False):
            continue

        stage_order_id = stage.get('order_id')
        is_last_stage = (idx == num_stages - 1)

        if stage_order_id and stage_order_id not in open_order_ids:
            logger.warning(f"Stage {stage['stage']} TP order missing, re-placing...")
            try:
                order_params = {'positionSide': 'BOTH', 'stopPrice': stage['tp_price']}
                if is_last_stage:
                    order_params['closePosition'] = True

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
                logger.error(f"Failed to re-place Stage {stage['stage']} TP: {e}")

    return state_changed


def _verify_sl_order(
    exchange: ccxt.bingx,
    position: Dict,
    symbol: str,
    open_order_ids: Dict,
) -> bool:
    """Verify and re-place missing SL order."""
    sl_order_id = position.get('sl_order_id')
    state_changed = False

    if sl_order_id and sl_order_id not in open_order_ids:
        logger.warning("SL order missing, re-placing...")
        try:
            close_side = 'sell' if position['direction'] == 'LONG' else 'buy'
            sl_order = exchange.create_order(
                symbol=symbol,
                type='STOP_MARKET',
                side=close_side,
                amount=position['quantity'],
                params={
                    'positionSide': 'BOTH',
                    'stopPrice': position['sl_price'],
                    'closePosition': True,
                }
            )
            position['sl_order_id'] = sl_order.get('id')
            logger.info(f"SL order re-placed: {sl_order.get('id')}")
            state_changed = True
        except ccxt.ExchangeError as e:
            logger.error(f"Failed to re-place SL order (exchange error): {e}")
        except Exception as e:
            logger.error(f"Failed to re-place SL order: {e}")

    return state_changed


def cancel_remaining_orders(
    exchange: ccxt.bingx,
    state: Dict[str, Any],
    config: Dict[str, Any],
) -> None:
    """
    Cancel all remaining orders for the position.

    Args:
        exchange: Exchange instance
        state: Bot state dictionary
        config: Bot configuration
    """
    position = state.get('position')
    if not position:
        return

    symbol = config['symbol']
    tp_order_id = position.get('tp_order_id')
    sl_order_id = position.get('sl_order_id')

    # Collect scale-out order IDs
    scale_out_tp_ids = []
    for stage in position.get('scale_out_stages', []):
        if stage.get('order_id') and not stage.get('filled'):
            scale_out_tp_ids.append(stage['order_id'])

    if not tp_order_id and not sl_order_id and not scale_out_tp_ids:
        return

    try:
        open_orders = exchange.fetch_open_orders(symbol)
        open_order_ids = [o.get('id') for o in open_orders]

        # Cancel scale-out TP orders
        for so_tp_id in scale_out_tp_ids:
            if so_tp_id in open_order_ids:
                try:
                    exchange.cancel_order(so_tp_id, symbol)
                    logger.info(f"🗑️ Cancelled Scale-out TP order: {so_tp_id}")
                except ccxt.OrderNotFound:
                    logger.debug(f"Scale-out TP order already cancelled: {so_tp_id}")
                except ccxt.ExchangeError as e:
                    logger.warning(f"Failed to cancel Scale-out TP order: {e}")
                except Exception as e:
                    logger.warning(f"Failed to cancel Scale-out TP order: {e}")

        # Cancel regular TP order
        if tp_order_id and tp_order_id in open_order_ids:
            try:
                exchange.cancel_order(tp_order_id, symbol)
                logger.info(f"🗑️ Cancelled TP order: {tp_order_id}")
            except ccxt.OrderNotFound:
                pass
            except ccxt.ExchangeError:
                pass
            except Exception:
                pass

        # Cancel SL order
        if sl_order_id and sl_order_id in open_order_ids:
            try:
                exchange.cancel_order(sl_order_id, symbol)
                logger.info(f"🗑️ Cancelled SL order: {sl_order_id}")
            except ccxt.OrderNotFound:
                pass
            except ccxt.ExchangeError:
                pass
            except Exception:
                pass

    except ccxt.NetworkError as e:
        logger.error(f"Failed to cancel remaining orders (network): {e}")
    except ccxt.ExchangeError as e:
        logger.error(f"Failed to cancel remaining orders (exchange): {e}")
    except Exception as e:
        logger.error(f"Failed to cancel remaining orders: {e}")