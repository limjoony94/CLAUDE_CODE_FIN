"""
Pattern 5m Bot - Position Closing and Recovery
Functions for closing positions and crash recovery.
"""

import logging
import os
import time
from datetime import datetime
from typing import Dict, Any, Optional, Tuple

import ccxt

from .constants import (
    FEE_PCT,
    QTY_TOLERANCE,
    ROTATION_ENABLED,
    CONFIDENCE_LOG_FILE,
)
from .position_open import calculate_tp_sl, setup_scale_out
from .models import APICache, CircuitBreaker, PerformanceMetrics
from .exchange import fetch_positions_cached, fetch_ticker_cached
from .state import save_state, save_metrics
from .orders import place_tp_sl_orders, cancel_remaining_orders
from .utils import extract_pattern_name

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
        logger.exception(f"❌ Ghost position detection failed: {e}")


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
    position = state.get('position')
    if not position:
        return

    if exchange:
        cancel_remaining_orders(exchange, state, config)

    direction = 1 if position['direction'] == 'LONG' else -1

    if not position.get('entry_price') or position['entry_price'] <= 0:
        logger.error(f"Invalid entry_price={position.get('entry_price')} — cannot calculate PnL, recording 0%")
        pnl_pct = 0.0
        price_pnl_pct = 0.0
    else:
        pnl_pct = direction * (exit_price / position['entry_price'] - 1) * 100 * config['leverage']
        pnl_pct -= 2 * FEE_PCT * config['leverage']
        # Calculate price-basis PnL (without leverage) for log clarity
        price_pnl_pct = direction * (exit_price / position['entry_price'] - 1) * 100
        price_pnl_pct -= 2 * FEE_PCT

    # Extract pattern name from reason
    pattern_name = extract_pattern_name(position.get('reason', '')) or 'N/A'

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

    # Invalidate position cache (don't set empty — forces fresh fetch next time)
    cache.invalidate_all()


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

    # Preserve pattern_name from previous state if available (before overwriting)
    old_position = state.get('position', {})
    old_pattern_name = extract_pattern_name(old_position.get('reason', '')) or old_position.get('pattern_name')

    # Try to read TP/SL from existing exchange orders (preserves per-pattern values)
    tp_from_exchange, sl_from_exchange = _read_tpsl_from_exchange_orders(
        exchange, config.get('symbol', 'BTC-USDT'), direction
    )

    if tp_from_exchange and sl_from_exchange:
        tp_price = tp_from_exchange
        sl_price = sl_from_exchange
        # Estimate tp_pct_adjusted for scale-out calculation
        tp_pct_adjusted = abs(tp_price / entry_price - 1) * 100 if entry_price > 0 else 1.0
        logger.info(
            f"Recovered {direction} position: entry=${entry_price:.1f} | "
            f"TP/SL from exchange orders: TP=${tp_price:.1f}, SL=${sl_price:.1f}"
        )
    else:
        # Fallback: calculate from config defaults (pass pattern for per-pattern lookup)
        tp_price, sl_price, tp_pct_adjusted, _ = calculate_tp_sl(
            entry_price, dir_mult, strategy, vol_mult=1.0, pattern=old_pattern_name, config=config
        )
        logger.info(
            f"Recovered {direction} position: entry=${entry_price:.1f} | "
            f"TP/SL from config{f' (pattern={old_pattern_name})' if old_pattern_name else ' defaults'}: "
            f"TP=${tp_price:.1f}, SL=${sl_price:.1f}"
        )

    # Setup scale-out if enabled
    scale_out_stages = setup_scale_out(strategy, entry_price, quantity, dir_mult, tp_pct_adjusted)

    reason = f"Recovered from exchange ({old_pattern_name})" if old_pattern_name else 'Recovered from exchange'
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
        'reason': reason,
        'pattern_name': old_pattern_name or None,
        'recovered': True,
        'needs_tpsl': True,
    }
    save_state(state)

    if exchange:
        place_tp_sl_orders(exchange, state, config)
        if state['position'].get('tp_order_id') or state['position'].get('sl_order_id') or scale_out_stages:
            state['position']['needs_tpsl'] = False
            save_state(state)


def _read_tpsl_from_exchange_orders(
    exchange: Optional[ccxt.bingx],
    symbol: str,
    direction: str,
) -> Tuple[Optional[float], Optional[float]]:
    """
    Read TP and SL prices from existing exchange open orders.

    Args:
        exchange: Exchange instance (may be None)
        symbol: Trading symbol
        direction: 'LONG' or 'SHORT'

    Returns:
        Tuple of (tp_price, sl_price) — either may be None if not found
    """
    if not exchange:
        return None, None

    try:
        open_orders = exchange.fetch_open_orders(symbol)
        tp_price = None
        sl_price = None

        for order in open_orders:
            order_type = (order.get('type') or '').upper()
            # CCXT normalizes stopPrice; BingX also provides it in info
            stop_price = float(
                order.get('stopPrice')
                or (order.get('info') or {}).get('stopPrice')
                or 0
            )
            if stop_price <= 0:
                continue

            if 'TAKE_PROFIT' in order_type:
                tp_price = stop_price
            elif 'STOP' in order_type and 'TAKE' not in order_type:
                sl_price = stop_price

        # Sanity check: TP and SL must be on correct sides of entry
        # (skip validation if we don't have both — partial is still useful)
        if tp_price and sl_price:
            logger.debug(f"Exchange orders found: TP=${tp_price:.1f}, SL=${sl_price:.1f}")

        return tp_price, sl_price

    except Exception as e:
        logger.debug(f"Could not read TP/SL from exchange orders: {e}")
        return None, None


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
    pattern_name = extract_pattern_name(reason) or None
    regime_tp_sl = position.get('regime_tp_sl')

    tp_price, sl_price, tp_pct_adjusted, _ = calculate_tp_sl(
        entry_price, direction, strategy, vol_mult, pattern_name, regime_tp_sl, config
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
            logger.info("TP/SL orders updated after recalculation")
            return True
        else:
            logger.warning("TP/SL orders placed but SL not confirmed — needs_tpsl remains True")
            save_state(state)
            return False
    except Exception as e:
        logger.exception(f"Failed to update TP/SL orders: {e}")
        save_state(state)
        return False


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
        t_close = time.time()
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
                ticker = fetch_ticker_cached(exchange, symbol, cache, force_refresh=True)
                fill_price = ticker['last']

            close_latency_ms = (time.time() - t_close) * 1000
            logger.info(f"✅ Early exit executed @ ${fill_price:.1f} [{close_latency_ms:.0f}ms]")

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

    # TP/SL was cancelled but market close failed — re-place both to protect position
    try:
        from .orders import _place_sl_order, _place_single_tp_order
        close_side = 'sell' if direction == 'LONG' else 'buy'
        sl_price = position.get('sl_price')
        tp_price = position.get('tp_price')
        if sl_price and quantity > 0:
            logger.warning(f"⚠️ Re-placing SL @ ${sl_price} after failed market close")
            _place_sl_order(exchange, position, symbol, close_side, quantity, sl_price)
        if tp_price and quantity > 0:
            logger.warning(f"⚠️ Re-placing TP @ ${tp_price} after failed market close")
            _place_single_tp_order(exchange, position, symbol, close_side, quantity, tp_price)
        save_state(state)
    except Exception as restore_e:
        logger.error(f"Failed to re-place TP/SL after market close failure: {restore_e}")

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
                # Use current ticker as fallback (more accurate than entry_price which gives PnL=0%)
                try:
                    ticker = fetch_ticker_cached(exchange, config['symbol'], cache, force_refresh=True)
                    fallback_price = ticker['last']
                except Exception:
                    fallback_price = local_position['entry_price']
                record_closed_position(exchange, state, config, fallback_price,
                                      'CRASH_RECOVERY', cache, metrics)
            return True

        # Case 3: Both exist — check direction mismatch first
        if exchange_position and local_position:
            ex_side = exchange_position.get('side', '')  # 'long' or 'short'
            local_dir = local_position.get('direction', '')  # 'LONG' or 'SHORT'
            expected_side = 'long' if local_dir == 'LONG' else 'short'

            if ex_side != expected_side:
                actual_dir = 'LONG' if ex_side == 'long' else 'SHORT'
                logger.error(
                    f"🔧 Recovery: Direction mismatch (local={local_dir}, exchange={actual_dir}). "
                    f"Closing local state, recovering exchange position."
                )
                actual_exit = get_actual_exit_price(exchange, state, config)
                exit_price = actual_exit['price'] if actual_exit else local_position['entry_price']
                record_closed_position(exchange, state, config, exit_price,
                                      'DIRECTION_MISMATCH', cache, metrics)
                recover_position_to_state(state, config, exchange_position, actual_dir, exchange, cache)
                return True

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
        logger.exception(f"Crash recovery failed: {e}")
        return False


def _update_confidence_log_outcome(
    entry_time: Optional[str],
    outcome: str,
    pnl_pct: float
) -> None:
    """
    Update the most recent confidence log entry with trade outcome.

    Reads only the last few KB of the file to find the target row,
    then rewrites only if an empty-outcome row is found.
    """
    try:
        csv_path = CONFIDENCE_LOG_FILE

        if not os.path.exists(csv_path):
            return

        outcome_value = f"{outcome}:{pnl_pct:+.2f}%"

        # Read only the tail of the file to find the last empty-outcome row
        file_size = os.path.getsize(csv_path)
        read_size = min(4096, file_size)  # Last 4KB is sufficient

        with open(csv_path, 'r+', encoding='utf-8') as f:
            # Seek to near end of file
            if file_size > read_size:
                f.seek(file_size - read_size)
                # Skip partial first line
                f.readline()
                tail_start = f.tell()
            else:
                tail_start = 0
                f.seek(0)

            tail_lines = f.readlines()

            if not tail_lines:
                return

            # Search from end for row with trailing comma
            for i in range(len(tail_lines) - 1, -1, -1):
                line = tail_lines[i].rstrip('\n').rstrip('\r')
                if line.endswith(','):
                    tail_lines[i] = line + outcome_value + '\n'
                    # Rewrite only the tail portion
                    f.seek(tail_start)
                    f.writelines(tail_lines)
                    f.truncate()
                    return

    except Exception as e:
        logger.warning(f"Failed to update confidence log outcome: {e}")