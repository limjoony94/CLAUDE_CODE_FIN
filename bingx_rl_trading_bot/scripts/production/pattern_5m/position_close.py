"""
Pattern 5m Bot - Position Closing and Recovery
Functions for closing positions and crash recovery.
"""

import logging
import os
from datetime import datetime
from typing import Dict, Any, Optional, List

import ccxt

import re

from .constants import (
    FEE_PCT,
    QTY_TOLERANCE,
    ROTATION_ENABLED,
    CONFIDENCE_LOG_FILE,
)
from .position_open import calculate_tp_sl, setup_scale_out
from .models import APICache, CircuitBreaker, PerformanceMetrics
from .exchange import fetch_positions_cached
from .state import save_state
from .orders import place_tp_sl_orders, cancel_remaining_orders

logger = logging.getLogger('pattern_5m')


def detect_ghost_positions(
    exchange: ccxt.bingx,
    state: Dict[str, Any],
    config: Dict[str, Any],
    cache: APICache,
    circuit_breaker: Optional[CircuitBreaker] = None,
    metrics: Optional[PerformanceMetrics] = None,
) -> None:
    """
    Ghost position detection - warn if exchange position not tracked locally.

    Ghost positions는 거래소에 있지만 local state에 없는 포지션입니다.
    이는 봇 재시작, crash, 또는 manual trading으로 발생할 수 있습니다.

    Args:
        exchange: Exchange instance
        state: Bot state dictionary
        config: Bot configuration
        cache: APICache instance
        circuit_breaker: Optional CircuitBreaker
        metrics: Optional PerformanceMetrics
    """
    symbol = config['symbol']

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

        # Ghost position: exchange has position but local state doesn't
        if exchange_position and not local_position:
            direction = 'LONG' if exchange_position.get('side') == 'long' else 'SHORT'
            qty = float(exchange_position.get('contracts', 0))
            entry_price = float(exchange_position.get('entryPrice', 0))

            logger.warning(
                f"👻 GHOST POSITION DETECTED on exchange:\n"
                f"   Symbol: {symbol}\n"
                f"   Direction: {direction}\n"
                f"   Quantity: {qty}\n"
                f"   Entry Price: {entry_price}\n"
                f"   → This position is not tracked in local state\n"
                f"   → Crash recovery will attempt to reconcile"
            )
    except Exception as e:
        logger.error(f"❌ Ghost position detection failed: {e}")


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

    # Calculate price-basis PnL (without leverage) for log clarity
    price_pnl_pct = direction * (exit_price / position['entry_price'] - 1) * 100
    price_pnl_pct -= 2 * FEE_PCT

    # Extract pattern name from reason
    pattern_match = re.search(r'Pattern:\s*(\S+)', position.get('reason', ''))
    pattern_name = pattern_match.group(1) if pattern_match else 'N/A'

    # Calculate hold time
    entry_time_str = position.get('entry_time', '')
    hold_minutes = 0
    if entry_time_str:
        try:
            entry_dt = datetime.fromisoformat(entry_time_str)
            hold_minutes = int((datetime.now() - entry_dt).total_seconds() / 60)
        except (ValueError, TypeError):
            pass

    logger.info(
        f"🏁 TRADE CLOSED | {position['direction']} {pattern_name} | "
        f"Entry: ${position['entry_price']:.1f} → Exit: ${exit_price:.1f} | "
        f"PnL: {pnl_pct:+.2f}% (lev) / {price_pnl_pct:+.2f}% (price) | "
        f"TP: ${position.get('tp_price', 0):.1f} SL: ${position.get('sl_price', 0):.1f} | "
        f"Reason: {exit_reason} | Hold: {hold_minutes}m"
    )

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
        state['consecutive_losses'] = 0
    else:
        state['consecutive_losses'] = state.get('consecutive_losses', 0) + 1

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

    # Update confidence log with trade outcome
    outcome = "WIN" if pnl_pct > 0 else "LOSS"
    _update_confidence_log_outcome(position.get('entry_time'), outcome, pnl_pct)

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

    # Use shared calculate_tp_sl (supports per-pattern TP/SL via needs_tpsl)
    tp_price, sl_price, tp_pct_adjusted, _ = calculate_tp_sl(
        entry_price, dir_mult, strategy, vol_mult=1.0
    )

    logger.info(f"Recovered {direction} position: entry=${entry_price:.1f}")

    # Setup scale-out if enabled
    scale_out_stages = setup_scale_out(strategy, entry_price, quantity, dir_mult, tp_pct_adjusted)

    state['position'] = {
        'direction': direction,
        'entry_price': entry_price,
        'quantity': quantity,
        'remaining_quantity': quantity,
        'tp_price': tp_price,
        'sl_price': sl_price,
        'vol_mult': 1.0,  # Recovery uses default volatility multiplier
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


def recalculate_position_orders(
    exchange: ccxt.bingx,
    state: Dict[str, Any],
    config: Dict[str, Any],
    new_quantity: float,
) -> bool:
    """
    Recalculate scale-out stages and update TP/SL orders when position quantity changes.

    Args:
        exchange: Exchange instance
        state: Bot state dictionary
        config: Bot configuration
        new_quantity: New position quantity from exchange

    Returns:
        True if recalculation was successful
    """
    position = state.get('position')
    if not position:
        return False

    strategy = config['strategy']
    entry_price = position.get('entry_price', 0)
    direction = 1 if position['direction'] == 'LONG' else -1
    vol_mult = position.get('vol_mult', 1.0)

    # Cancel existing orders FIRST (before modifying scale_out_stages)
    try:
        cancel_remaining_orders(exchange, state, config)
    except Exception as e:
        logger.warning(f"Failed to cancel existing orders: {e}")

    # Update quantities
    position['quantity'] = new_quantity
    position['remaining_quantity'] = new_quantity

    # Recalculate TP/SL using shared function (supports per-pattern values)
    reason = position.get('reason', '')
    pattern_match = re.search(r'Pattern:\s*(\S+)', reason)
    pattern_name = pattern_match.group(1) if pattern_match else None
    regime_tp_sl = position.get('regime_tp_sl')

    tp_price, sl_price, tp_pct_adjusted, _ = calculate_tp_sl(
        entry_price, direction, strategy, vol_mult, pattern_name, regime_tp_sl
    )

    position['tp_price'] = tp_price
    position['sl_price'] = sl_price

    # Recalculate scale-out stages (after cancelling old orders)
    scale_out_stages = setup_scale_out(
        strategy, entry_price, new_quantity, direction, tp_pct_adjusted
    )
    position['scale_out_stages'] = scale_out_stages
    position['scale_out_enabled'] = bool(scale_out_stages)

    # Update rotation fields
    position['rotation_enabled'] = ROTATION_ENABLED and bool(scale_out_stages)
    position['is_partial'] = False

    logger.info(f"🔧 Recalculated position: qty={new_quantity}, stages={len(scale_out_stages)}")

    # Place new orders
    try:
        position['needs_tpsl'] = True
        save_state(state)

        place_tp_sl_orders(exchange, state, config)
        if position.get('sl_order_id') or scale_out_stages:
            position['needs_tpsl'] = False
            save_state(state)
            logger.info("✅ TP/SL orders updated")
            return True
    except Exception as e:
        logger.error(f"Failed to update TP/SL orders: {e}")
        save_state(state)
        return False

    return True


def close_position_market(
    exchange: ccxt.bingx,
    state: Dict[str, Any],
    config: Dict[str, Any],
    cache: APICache,
    exit_reason: str = 'MARKET',
    metrics: Optional[PerformanceMetrics] = None,
) -> bool:
    """
    Close position with market order (used for early exit).

    Args:
        exchange: Exchange instance
        state: Bot state dictionary
        config: Bot configuration
        cache: APICache instance
        exit_reason: Reason for closing (e.g., 'EARLY_BD', 'EARLY_BU')
        metrics: Optional PerformanceMetrics

    Returns:
        True if market close was successful
    """
    position = state.get('position')
    if not position:
        logger.warning("No position to close")
        return False

    symbol = config['symbol']
    direction = position.get('direction', '')
    quantity = position.get('remaining_quantity', position.get('quantity', 0))

    if quantity <= 0:
        logger.warning("Position quantity is zero")
        return False

    try:
        # Cancel existing TP/SL orders first
        cancel_remaining_orders(exchange, state, config)

        # Determine close side
        close_side = 'sell' if direction == 'LONG' else 'buy'

        logger.info(f"🚨 Early exit: closing {direction} {quantity} {symbol} @ market ({exit_reason})")

        # Place market order to close
        order = exchange.create_order(
            symbol=symbol,
            type='market',
            side=close_side,
            amount=quantity,
            params={'positionSide': 'BOTH'}  # One-way mode
        )

        if order:
            # Get actual fill price
            fill_price = float(order.get('average', 0)) or float(order.get('price', 0))

            if fill_price <= 0:
                # Fallback to ticker price
                from .exchange import fetch_ticker_cached
                ticker = fetch_ticker_cached(exchange, symbol, cache, force_refresh=True)
                fill_price = ticker['last']

            logger.info(f"✅ Early exit executed @ ${fill_price:.1f}")

            # Record the closed position
            record_closed_position(
                exchange, state, config, fill_price, exit_reason, cache, metrics
            )
            return True

    except ccxt.InsufficientFunds as e:
        logger.error(f"Insufficient funds for market close: {e}")
    except ccxt.InvalidOrder as e:
        logger.error(f"Invalid order for market close: {e}")
    except ccxt.NetworkError as e:
        logger.error(f"Network error during market close: {e}")
    except ccxt.ExchangeError as e:
        logger.error(f"Exchange error during market close: {e}")
    except Exception as e:
        logger.exception(f"Failed to close position: {e}")

    return False


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

    # Phase 3: Ghost position detection (log warnings, don't block)
    detect_ghost_positions(exchange, state, config, cache, circuit_breaker, metrics)

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
                # Recalculate scale-out stages and TP/SL orders with new quantity
                recalculate_position_orders(exchange, state, config, ex_qty)
                return True

            # Case 4: Check scale-out stages sum matches position quantity
            scale_out_stages = local_position.get('scale_out_stages', [])
            if scale_out_stages:
                stages_sum = sum(s.get('quantity', 0) for s in scale_out_stages if not s.get('filled', False))
                if abs(stages_sum - ex_qty) > QTY_TOLERANCE:
                    logger.warning(f"🔧 Recovery: Scale-out sum mismatch (stages={stages_sum:.4f}, position={ex_qty:.4f})")
                    recalculate_position_orders(exchange, state, config, ex_qty)
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


def _update_confidence_log_outcome(
    entry_time: Optional[str],
    outcome: str,
    pnl_pct: float
) -> None:
    """
    Update the most recent confidence log entry with trade outcome.

    This allows correlation of confidence scores with actual trade results.

    Args:
        entry_time: Position entry timestamp (ISO format)
        outcome: "WIN" or "LOSS"
        pnl_pct: Profit/loss percentage
    """
    import csv

    try:
        csv_path = CONFIDENCE_LOG_FILE  # Already absolute path from constants

        if not os.path.exists(csv_path):
            logger.debug("Confidence log file not found, skipping outcome update")
            return

        # Read all rows
        rows = []
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            rows = list(reader)

        if len(rows) < 2:
            return

        # Find the most recent row with empty outcome (last row typically)
        header = rows[0]
        outcome_idx = header.index('outcome') if 'outcome' in header else -1

        if outcome_idx == -1:
            logger.warning("Confidence log missing 'outcome' column")
            return

        # Update the last row with empty outcome
        for i in range(len(rows) - 1, 0, -1):
            if len(rows[i]) > outcome_idx and rows[i][outcome_idx].strip() == '':
                rows[i][outcome_idx] = f"{outcome}:{pnl_pct:+.2f}%"
                break

        # Write back
        with open(csv_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f)
            writer.writerows(rows)

        logger.debug(f"📝 Confidence log updated with outcome: {outcome}")

    except Exception as e:
        logger.warning(f"Failed to update confidence log outcome: {e}")