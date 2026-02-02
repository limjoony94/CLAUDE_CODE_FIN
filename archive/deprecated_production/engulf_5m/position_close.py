"""
Engulf 5m Bot - Position Closing and Recovery
Functions for closing positions and crash recovery.
"""

import logging
from datetime import datetime
from typing import Dict, Any, Optional, List

import ccxt

from .constants import (
    SLIPPAGE_BUFFER_PCT,
    FEE_PCT,
    QTY_TOLERANCE,
    PRICE_ROUND_DECIMALS,
    QUANTITY_ROUND_DECIMALS,
)
from .models import APICache, CircuitBreaker, PerformanceMetrics
from .exchange import fetch_positions_cached
from .state import save_state
from .orders import place_tp_sl_orders, cancel_remaining_orders

logger = logging.getLogger('engulf_5m')


def record_closed_position(
    exchange: ccxt.bingx,
    state: Dict[str, Any],
    config: Dict[str, Any],
    exit_price: float,
    exit_reason: str,
    cache: APICache,
    metrics: Optional[PerformanceMetrics] = None,
) -> None:
    """
    Record a closed position and update statistics.

    Args:
        exchange: Exchange instance
        state: Bot state dictionary
        config: Bot configuration
        exit_price: Exit price
        exit_reason: Reason for exit (TP, SL, etc.)
        cache: APICache instance
        metrics: Optional PerformanceMetrics
    """
    from .state import save_metrics

    position = state.get('position')
    if not position:
        return

    if exchange:
        cancel_remaining_orders(exchange, state, config)

    direction = 1 if position['direction'] == 'LONG' else -1
    pnl_pct = direction * (exit_price / position['entry_price'] - 1) * 100 * config['leverage']
    pnl_pct -= 2 * FEE_PCT * config['leverage']

    logger.info(f"Position closed: {exit_reason} | Exit: ${exit_price:.1f} | PnL: {pnl_pct:+.2f}%")

    # Update metrics
    if metrics:
        metrics.update_trade(pnl_pct)

    # Update state statistics
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

    state['last_signal_time'] = datetime.now().isoformat()
    state['position'] = None
    save_state(state, is_trade_close=True)

    # Save metrics immediately after trade close
    if metrics:
        save_metrics(metrics)
        logger.info(f"📊 Metrics saved: {metrics.total_trades} trades, {metrics.actual_win_rate:.1f}% WR")

    # Invalidate position cache
    cache.set_positions([])


def recover_position_to_state(
    state: Dict[str, Any],
    config: Dict[str, Any],
    exchange_pos: Dict,
    direction: str,
    exchange: Optional[ccxt.bingx] = None,
    cache: Optional[APICache] = None,
) -> None:
    """
    Recover position from exchange to local state.

    Args:
        state: Bot state dictionary
        config: Bot configuration
        exchange_pos: Exchange position dictionary
        direction: 'LONG' or 'SHORT'
        exchange: Optional exchange instance for placing TP/SL
        cache: Optional APICache instance
    """
    strategy = config['strategy']
    entry_price = float(exchange_pos.get('entryPrice', 0))
    quantity = float(exchange_pos.get('contracts', 0))
    dir_mult = 1 if direction == 'LONG' else -1

    vol_mult = 1.0
    base_tp_pct = strategy['tp_pct']
    base_sl_pct = strategy['sl_pct']
    tp_pct_adjusted = (base_tp_pct * vol_mult) - SLIPPAGE_BUFFER_PCT
    sl_pct_adjusted = (base_sl_pct * vol_mult) - SLIPPAGE_BUFFER_PCT

    tp_price = round(entry_price * (1 + dir_mult * tp_pct_adjusted / 100), PRICE_ROUND_DECIMALS)
    sl_price = round(entry_price * (1 - dir_mult * sl_pct_adjusted / 100), PRICE_ROUND_DECIMALS)

    logger.info(f"Recovered {direction} position: entry=${entry_price:.1f}")

    # Setup scale-out if enabled
    scale_out_stages = _setup_scale_out_for_recovery(strategy, entry_price, quantity, dir_mult, tp_pct_adjusted)

    state['position'] = {
        'direction': direction,
        'entry_price': entry_price,
        'quantity': quantity,
        'remaining_quantity': quantity,
        'tp_price': tp_price,
        'sl_price': sl_price,
        'vol_mult': vol_mult,
        'scale_out_enabled': bool(scale_out_stages),
        'scale_out_stages': scale_out_stages,
        'entry_time': datetime.now().isoformat(),
        'reason': 'Recovered from exchange',
        'recovered': True,
        'needs_tpsl': True,
    }
    save_state(state)

    if exchange:
        place_tp_sl_orders(exchange, state, config)
        if state['position'].get('tp_order_id') or state['position'].get('sl_order_id') or scale_out_stages:
            state['position']['needs_tpsl'] = False
            save_state(state)


def _setup_scale_out_for_recovery(
    strategy: Dict[str, Any],
    entry_price: float,
    quantity: float,
    direction: int,
    tp_pct_adjusted: float,
) -> List[Dict]:
    """Setup scale-out stages for recovery."""
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


def recover_from_crash(
    exchange: ccxt.bingx,
    state: Dict[str, Any],
    config: Dict[str, Any],
    cache: APICache,
    circuit_breaker: Optional[CircuitBreaker] = None,
    metrics: Optional[PerformanceMetrics] = None,
) -> bool:
    """
    Comprehensive crash recovery - reconcile exchange with local state.

    Args:
        exchange: Exchange instance
        state: Bot state dictionary
        config: Bot configuration
        cache: APICache instance
        circuit_breaker: Optional CircuitBreaker
        metrics: Optional PerformanceMetrics

    Returns:
        True if recovery action was taken
    """
    # Import here to avoid circular dependency
    from .position_monitor import get_actual_exit_price

    symbol = config['symbol']
    logger.info("🔄 Running crash recovery check...")

    try:
        positions = fetch_positions_cached(exchange, symbol, cache, force_refresh=True,
                                           circuit_breaker=circuit_breaker, metrics=metrics)

        exchange_position = None
        for pos in positions:
            contracts = float(pos.get('contracts', 0))
            if contracts > 0:
                exchange_position = pos
                break

        local_position = state.get('position')

        # Case 1: Exchange has position, local doesn't
        if exchange_position and not local_position:
            logger.warning("🔧 Recovery: Found orphan position on exchange")
            direction = 'LONG' if exchange_position.get('side') == 'long' else 'SHORT'
            recover_position_to_state(state, config, exchange_position, direction, exchange, cache)
            return True

        # Case 2: Local has position, exchange doesn't
        if local_position and not exchange_position:
            logger.warning("🔧 Recovery: Local position not found on exchange")
            actual_exit = get_actual_exit_price(exchange, state, config)
            if actual_exit:
                record_closed_position(exchange, state, config, actual_exit['price'],
                                      actual_exit['reason'], cache, metrics)
            else:
                record_closed_position(exchange, state, config, local_position['entry_price'],
                                      'CRASH_RECOVERY', cache, metrics)
            return True

        # Case 3: Both have positions - verify they match
        if exchange_position and local_position:
            ex_qty = float(exchange_position.get('contracts', 0))
            local_qty = local_position.get('quantity', 0)

            if abs(ex_qty - local_qty) > QTY_TOLERANCE:
                logger.warning(f"🔧 Recovery: Quantity mismatch (exchange={ex_qty}, local={local_qty})")
                local_position['quantity'] = ex_qty
                local_position['remaining_quantity'] = ex_qty
                save_state(state)
                return True

        logger.info("✅ Crash recovery check passed - state is consistent")
        return False

    except ccxt.NetworkError as e:
        logger.error(f"Crash recovery failed (network error): {e}")
        return False
    except ccxt.ExchangeError as e:
        logger.error(f"Crash recovery failed (exchange error): {e}")
        return False
    except Exception as e:
        logger.error(f"Crash recovery failed: {e}")
        return False