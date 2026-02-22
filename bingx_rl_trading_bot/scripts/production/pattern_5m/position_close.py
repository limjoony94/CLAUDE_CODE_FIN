"""
Pattern 5m Bot - Position Closing and Recovery
Functions for closing positions and crash recovery.
"""

import logging
import os
import time
import uuid
from datetime import datetime
from typing import Dict, Any, Optional, Tuple

import ccxt

from .constants import (
    FEE_PCT,
    QTY_TOLERANCE,
    QUANTITY_ROUND_DECIMALS,
    ROTATION_ENABLED,
    CONFIDENCE_LOG_FILE,
)
from .position_open import calculate_tp_sl, setup_scale_out
from .models import APICache, CircuitBreaker, PerformanceMetrics
from .exchange import fetch_positions_cached, fetch_ticker_cached, fetch_balance_cached
from .state import save_state, save_metrics
from .orders import (
    place_tp_sl_orders, cancel_remaining_orders, update_emergency_sl,
    cancel_emergency_sl, _get_position_side,
)
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

        # Build exchange map by direction
        exchange_map = {}
        for pos in positions:
            contracts = float(pos.get('contracts', 0))
            if contracts > 0:
                exchange_map[pos.get('side')] = pos

        # Build set of directions tracked by bot
        bot_slots = state.get('positions') or {}
        bot_tracked_sides = set()
        for slot in bot_slots.values():
            d = slot.get('direction', '')
            bot_tracked_sides.add('long' if d == 'LONG' else 'short')

        # Ghost: exchange has position in a direction bot doesn't track
        for side, ex_pos in exchange_map.items():
            if side not in bot_tracked_sides:
                direction = 'LONG' if side == 'long' else 'SHORT'
                qty = float(ex_pos.get('contracts', 0))
                entry_price = float(ex_pos.get('entryPrice', 0))

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


def calculate_pnl(
    entry_price: float,
    exit_price: float,
    direction: int,
    leverage: int,
) -> Tuple[float, float]:
    """
    Calculate leveraged and price-basis PnL percentage.

    Args:
        entry_price: Entry price (must be > 0)
        exit_price: Exit price
        direction: 1 for LONG, -1 for SHORT
        leverage: Leverage multiplier

    Returns:
        Tuple of (pnl_pct_leveraged, pnl_pct_price)
    """
    price_move = direction * (exit_price / entry_price - 1) * 100
    pnl_pct = price_move * leverage - 2 * FEE_PCT * leverage
    price_pnl_pct = price_move - 2 * FEE_PCT
    return pnl_pct, price_pnl_pct


def record_closed_position(
    exchange: ccxt.bingx,
    state: Dict[str, Any],
    config: Dict[str, Any],
    exit_price: float,
    exit_reason: str,
    cache: APICache,
    metrics: Optional[PerformanceMetrics] = None,
    position: Optional[Dict] = None,
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
        position: Specific slot to close (if None, uses first slot)
    """
    if position is None:
        positions = state.get('positions') or {}
        if not positions:
            return
        position = next(iter(positions.values()))

    if exchange:
        cancel_remaining_orders(exchange, state, config, position=position)

    direction = 1 if position['direction'] == 'LONG' else -1

    if not position.get('entry_price') or position['entry_price'] <= 0:
        logger.error(f"Invalid entry_price={position.get('entry_price')} — cannot calculate PnL, recording 0%")
        pnl_pct = 0.0
        price_pnl_pct = 0.0
    else:
        pnl_pct, price_pnl_pct = calculate_pnl(
            entry_price=position['entry_price'],
            exit_price=exit_price,
            direction=direction,
            leverage=config['leverage'],
        )

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

    # Scale PnL to portfolio basis (1/N sizing)
    max_positions = config.get('max_positions', 1)
    portfolio_pnl_pct = pnl_pct / max_positions

    logger.info(
        f"🏁 TRADE CLOSED | {position['direction']} {pattern_name} | "
        f"Entry: ${position['entry_price']:.1f} → Exit: ${exit_price:.1f} | "
        f"PnL: {pnl_pct:+.2f}% (slot) / {portfolio_pnl_pct:+.2f}% (portfolio) | "
        f"TP: ${position.get('tp_price', 0):.1f} SL: ${position.get('sl_price', 0):.1f} | "
        f"Reason: {exit_reason} | Hold: {hold_minutes}m"
    )

    # Update metrics (portfolio-scaled PnL)
    if metrics:
        metrics.update_trade(portfolio_pnl_pct)

    # Update state statistics (portfolio-scaled PnL)
    state['total_trades'] += 1
    state['total_pnl'] += portfolio_pnl_pct
    state['daily_trades'] += 1
    state['daily_pnl'] += portfolio_pnl_pct

    if pnl_pct >= 0:
        state['winning_trades'] += 1
        state['consecutive_losses'] = 0
    else:
        state['consecutive_losses'] = state.get('consecutive_losses', 0) + 1

    state['last_trade'] = {
        'direction': position['direction'],
        'entry_price': position['entry_price'],
        'exit_price': exit_price,
        'pnl_pct': pnl_pct,
        'portfolio_pnl_pct': portfolio_pnl_pct,
        'exit_reason': exit_reason,
        'closed_at': datetime.now().isoformat(),
    }

    state['last_signal_time'] = datetime.now().isoformat()

    # Remove this slot from positions dict
    slot_id = position.get('slot_id')
    positions = state.get('positions') or {}
    if slot_id and slot_id in positions:
        del positions[slot_id]
    elif slot_id is None:
        # Legacy fallback: remove first slot matching direction
        for sid, s in list(positions.items()):
            if s.get('direction') == position.get('direction'):
                del positions[sid]
                break
    state['has_position'] = len(positions) > 0
    # Update active_direction
    if not positions:
        state['active_direction'] = None

    # v1.29.0: Update emergency SL after slot removal
    if exchange and positions:
        update_emergency_sl(exchange, state, config)
    elif exchange and not positions:
        cancel_emergency_sl(exchange, state, config)

    save_state(state, is_trade_close=True)

    # Save metrics immediately after trade close
    if metrics:
        save_metrics(metrics)
        logger.info(f"📊 Metrics saved: {metrics.total_trades} trades, {metrics.actual_win_rate:.1f}% WR")

    # Update confidence log with trade outcome
    outcome = "WIN" if pnl_pct >= 0 else "LOSS"
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
    saved_tpsl_pairs: Optional[list] = None,
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
        saved_tpsl_pairs: Optional pre-saved per-slot (tp, sl) tuples from
            _snapshot_all_tpsl() — used when orders were cancelled before this call
    """
    strategy = config['strategy']
    entry_price = float(exchange_pos.get('entryPrice', 0))
    quantity = float(exchange_pos.get('contracts', 0))
    dir_mult = 1 if direction == 'LONG' else -1

    # Try to find pattern_name from existing slots for this direction
    old_pattern_name = None
    for slot in (state.get('positions') or {}).values():
        if slot.get('direction') == direction:
            old_pattern_name = extract_pattern_name(slot.get('reason', '')) or slot.get('pattern_name')
            if old_pattern_name:
                break

    if not old_pattern_name:
        logger.warning(
            f"⚠️ Recovery: pattern_name unavailable for {direction} — "
            f"per-pattern TP/SL adjustment will use defaults until manually restored"
        )

    # Determine TP/SL source (priority: saved snapshot > exchange orders > config)
    if saved_tpsl_pairs:
        # Use first pair as default fallback for slots beyond saved count
        default_tp, default_sl = saved_tpsl_pairs[0]
        if not default_sl:
            _, default_sl, _, _ = calculate_tp_sl(
                entry_price, dir_mult, strategy, vol_mult=1.0, pattern=old_pattern_name, config=config
            )
        tp_pct_adjusted = abs(default_tp / entry_price - 1) * 100 if entry_price > 0 else 1.0
        logger.info(
            f"Recovered {direction} position: entry=${entry_price:.1f} | "
            f"Using {len(saved_tpsl_pairs)} saved TP/SL pair(s)"
        )
    else:
        # Try to read TP/SL from existing exchange orders (preserves per-pattern values)
        tp_from_exchange, sl_from_exchange = _read_tpsl_from_exchange_orders(
            exchange, config.get('symbol', 'BTC-USDT'), direction
        )

        if tp_from_exchange and sl_from_exchange:
            default_tp = tp_from_exchange
            default_sl = sl_from_exchange
            tp_pct_adjusted = abs(default_tp / entry_price - 1) * 100 if entry_price > 0 else 1.0
            logger.info(
                f"Recovered {direction} position: entry=${entry_price:.1f} | "
                f"TP/SL from exchange orders: TP=${default_tp:.1f}, SL=${default_sl:.1f}"
            )
        else:
            # Fallback: calculate from config defaults (pass pattern for per-pattern lookup)
            default_tp, default_sl, tp_pct_adjusted, _ = calculate_tp_sl(
                entry_price, dir_mult, strategy, vol_mult=1.0, pattern=old_pattern_name, config=config
            )
            logger.info(
                f"Recovered {direction} position: entry=${entry_price:.1f} | "
                f"TP/SL from config{f' (pattern={old_pattern_name})' if old_pattern_name else ' defaults'}: "
                f"TP=${default_tp:.1f}, SL=${default_sl:.1f}"
            )

    # Determine how many recovery slots to create (N=1 for single-position mode)
    max_positions = config.get('max_positions', 1)
    n_slots = _calculate_recovery_slot_count(
        exchange, cache, config, quantity, entry_price, max_positions, state
    )
    per_slot_qty = round(quantity / n_slots, QUANTITY_ROUND_DECIMALS)

    reason = f"Recovered from exchange ({old_pattern_name})" if old_pattern_name else 'Recovered from exchange'
    positions = state.setdefault('positions', {})
    new_slot_ids = []

    for i in range(n_slots):
        # Last slot gets remainder to avoid rounding loss
        if i == n_slots - 1:
            slot_qty = round(quantity - per_slot_qty * (n_slots - 1), QUANTITY_ROUND_DECIMALS)
        else:
            slot_qty = per_slot_qty

        # Per-slot TP/SL from saved snapshot (if available)
        if saved_tpsl_pairs and i < len(saved_tpsl_pairs):
            slot_tp = saved_tpsl_pairs[i][0] or default_tp
            slot_sl = saved_tpsl_pairs[i][1] or default_sl
        else:
            slot_tp = default_tp
            slot_sl = default_sl

        slot_tp_pct = abs(slot_tp / entry_price - 1) * 100 if entry_price > 0 else 1.0
        scale_out_stages = setup_scale_out(strategy, entry_price, slot_qty, dir_mult, slot_tp_pct)
        slot_id = uuid.uuid4().hex[:8]
        recovered_slot = {
            'slot_id': slot_id,
            'direction': direction,
            'entry_price': entry_price,
            'quantity': slot_qty,
            'remaining_quantity': slot_qty,
            'tp_price': slot_tp,
            'sl_price': slot_sl,
            'vol_mult': 1.0,
            'scale_out_enabled': bool(scale_out_stages),
            'scale_out_stages': scale_out_stages,
            'entry_time': datetime.now().isoformat(),
            'reason': reason,
            'pattern_name': old_pattern_name or None,
            'recovered': True,
            'needs_tpsl': True,
        }
        positions[slot_id] = recovered_slot
        new_slot_ids.append(slot_id)

    # One-Way mode: track direction; Hedge: no single-direction constraint
    if config.get('position_mode') != 'hedge':
        state['active_direction'] = direction
    state['has_position'] = True
    logger.info(f"Recovery: created {n_slots} slot(s) from {quantity:.4f} total qty")
    save_state(state)

    if exchange:
        for sid in new_slot_ids:
            slot = positions[sid]
            place_tp_sl_orders(exchange, state, config, position=slot)
            if slot.get('tp_order_id') or slot.get('sl_order_id') or slot.get('scale_out_stages'):
                slot['needs_tpsl'] = False
        save_state(state)


def _calculate_recovery_slot_count(
    exchange: Optional[ccxt.bingx],
    cache: Optional[APICache],
    config: Dict[str, Any],
    quantity: float,
    entry_price: float,
    max_positions: int,
    state: Dict[str, Any],
) -> int:
    """Calculate how many virtual slots to create during orphan recovery.

    Uses current equity to estimate expected per-slot quantity,
    then divides exchange quantity by that estimate.
    Falls back to 1 slot if calculation fails.
    """
    if max_positions <= 1 or not exchange or not cache:
        return 1

    existing_slots = len(state.get('positions') or {})
    available_slots = max(1, max_positions - existing_slots)

    try:
        balance = fetch_balance_cached(exchange, cache, force_refresh=True)
        total_equity = float(balance.get('USDT', {}).get('total', 0))

        if total_equity <= 0 or entry_price <= 0:
            return 1

        size_pct = config.get('position_size_pct', 95) / 100
        leverage = config.get('leverage', 3)
        per_slot_equity = total_equity * size_pct / max_positions
        expected_slot_qty = round(
            (per_slot_equity * leverage) / entry_price,
            QUANTITY_ROUND_DECIMALS,
        )

        if expected_slot_qty <= 0:
            return 1

        n_slots = max(1, min(available_slots, round(quantity / expected_slot_qty)))
        logger.info(
            f"Recovery slot calc: qty={quantity:.4f}, "
            f"expected_per_slot={expected_slot_qty:.4f}, "
            f"available={available_slots} → {n_slots} slot(s)"
        )
        return n_slots

    except Exception as e:
        logger.warning(f"Could not calculate recovery slot count, using 1: {e}")
        return 1


def _read_tpsl_from_exchange_orders(
    exchange: Optional[ccxt.bingx],
    symbol: str,
    direction: str,
) -> Tuple[Optional[float], Optional[float]]:
    """
    Read TP and SL prices from existing exchange open orders.

    With multi-slot, multiple TP/SL orders may exist. Returns the widest
    (most protective) pair based on direction:
    - LONG: highest TP, lowest SL
    - SHORT: lowest TP, highest SL

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
        tp_prices = []
        sl_prices = []

        for order in open_orders:
            # In hedge mode, filter by positionSide to avoid mixing directions
            order_pos_side = (
                (order.get('info') or {}).get('positionSide', 'BOTH')
            ).upper()
            if order_pos_side != 'BOTH' and order_pos_side != direction:
                continue

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
                tp_prices.append(stop_price)
            elif 'STOP' in order_type and 'TAKE' not in order_type:
                sl_prices.append(stop_price)

        # Select widest pair based on direction for maximum protection
        tp_price = None
        sl_price = None
        if tp_prices:
            tp_price = max(tp_prices) if direction == 'LONG' else min(tp_prices)
        if sl_prices:
            sl_price = min(sl_prices) if direction == 'LONG' else max(sl_prices)

        if tp_prices or sl_prices:
            tp_str = f"${tp_price:.1f}" if tp_price else "None"
            sl_str = f"${sl_price:.1f}" if sl_price else "None"
            logger.debug(
                f"Exchange orders found: {len(tp_prices)} TP, {len(sl_prices)} SL → "
                f"TP={tp_str}, SL={sl_str}"
            )

        return tp_price, sl_price

    except Exception as e:
        logger.debug(f"Could not read TP/SL from exchange orders: {e}")
        return None, None


def _snapshot_all_tpsl(
    exchange: Optional[ccxt.bingx],
    symbol: str,
    direction: str,
) -> list:
    """Snapshot per-slot TP/SL prices from exchange orders before cancellation.

    Called BEFORE Phase 1 order cancellation to preserve per-pattern TP/SL values.
    Returns list of (tp_price, sl_price) tuples, one per detected slot.
    Emergency SL is filtered out (identified by largest amount when SL count > TP count).

    Args:
        exchange: Exchange instance (may be None)
        symbol: Trading symbol
        direction: 'LONG' or 'SHORT'

    Returns:
        List of (tp_price, sl_price) tuples. Empty list if no TP orders found.
    """
    if not exchange:
        return []

    try:
        open_orders = exchange.fetch_open_orders(symbol)
        tp_entries = []  # (price, amount)
        sl_entries = []  # (price, amount)

        for order in open_orders:
            # In hedge mode, filter by positionSide to avoid mixing directions
            order_pos_side = (
                (order.get('info') or {}).get('positionSide', 'BOTH')
            ).upper()
            if order_pos_side != 'BOTH' and order_pos_side != direction:
                continue

            order_type = (order.get('type') or '').upper()
            stop_price = float(
                order.get('stopPrice')
                or (order.get('info') or {}).get('stopPrice')
                or 0
            )
            amount = float(order.get('amount', 0))
            if stop_price <= 0:
                continue

            if 'TAKE_PROFIT' in order_type:
                tp_entries.append((stop_price, amount))
            elif 'STOP' in order_type and 'TAKE' not in order_type:
                sl_entries.append((stop_price, amount))

        if not tp_entries:
            return []

        # Filter emergency SL: if more SLs than TPs, remove the one with largest amount
        per_slot_sls = list(sl_entries)
        if len(sl_entries) > len(tp_entries):
            sl_by_amount = sorted(sl_entries, key=lambda x: x[1], reverse=True)
            emergency = sl_by_amount[0]
            per_slot_sls = sl_by_amount[1:]
            logger.debug(
                f"Snapshot: filtered emergency SL price=${emergency[0]:.1f} "
                f"amount={emergency[1]:.4f}"
            )

        # Sort for consistent pairing by "risk profile width"
        # Pair widest TP with widest SL (both farther from entry)
        tp_prices = [p for p, _ in tp_entries]
        sl_prices = [p for p, _ in per_slot_sls]

        if direction == 'SHORT':
            # SHORT: lower TP = wider (farther below entry), higher SL = wider
            tp_prices.sort()            # ascending = widest first
            sl_prices.sort(reverse=True)  # descending = widest first
        else:
            # LONG: higher TP = wider (farther above entry), lower SL = wider
            tp_prices.sort(reverse=True)  # descending = widest first
            sl_prices.sort()              # ascending = widest first

        # Pair by index (widest-with-widest)
        n_pairs = min(len(tp_prices), len(sl_prices))
        pairs = [(tp_prices[i], sl_prices[i]) for i in range(n_pairs)]

        # Append unpaired TPs with None SL
        for i in range(n_pairs, len(tp_prices)):
            pairs.append((tp_prices[i], None))

        if pairs:
            logger.info(
                f"Snapshot: {len(tp_entries)} TP, {len(sl_entries)} SL orders "
                f"→ {len(pairs)} pair(s) saved before cancellation"
            )

        return pairs

    except Exception as e:
        logger.debug(f"Could not snapshot TP/SL from exchange: {e}")
        return []


def recalculate_position_orders(
    exchange: ccxt.bingx,
    state: Dict[str, Any],
    config: Dict[str, Any],
    new_quantity: float,
    position: Optional[Dict] = None,
) -> bool:
    """
    Recalculate scale-out stages and update TP/SL orders when position quantity changes.

    Args:
        exchange: Exchange instance
        state: Bot state dictionary
        config: Bot configuration
        new_quantity: New position quantity from exchange
        position: Specific slot to recalculate (if None, uses first slot)

    Returns:
        True if recalculation was successful
    """
    if position is None:
        positions = state.get('positions') or {}
        if not positions:
            return False
        position = next(iter(positions.values()))

    strategy = config['strategy']
    entry_price = position.get('entry_price', 0)
    direction = 1 if position['direction'] == 'LONG' else -1
    vol_mult = position.get('vol_mult', 1.0)

    # Cancel existing orders FIRST (before modifying scale_out_stages)
    try:
        cancel_remaining_orders(exchange, state, config, position=position)
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

        place_tp_sl_orders(exchange, state, config, position=position)
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
    position: Optional[Dict] = None,
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
        position: Specific slot to close (if None, uses first slot)

    Returns:
        True if market close was successful
    """
    if position is None:
        positions = state.get('positions') or {}
        if not positions:
            logger.warning("No position to close")
            return False
        position = next(iter(positions.values()))

    symbol = config['symbol']
    direction = position.get('direction', '')
    quantity = position.get('remaining_quantity', position.get('quantity', 0))

    if quantity <= 0:
        logger.warning("Position quantity is zero")
        return False

    try:
        # Cancel existing TP/SL orders first
        cancel_remaining_orders(exchange, state, config, position=position)

        # Determine close side and positionSide
        close_side = 'sell' if direction == 'LONG' else 'buy'
        position_side = _get_position_side(config, direction)

        logger.info(f"🚨 Early exit: closing {direction} {quantity} {symbol} @ market ({exit_reason})")

        # Place market order to close
        t_close = time.time()
        order = exchange.create_order(
            symbol=symbol,
            type='market',
            side=close_side,
            amount=quantity,
            params={'positionSide': position_side}
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
                exchange, state, config, fill_price, exit_reason, cache, metrics,
                position=position,
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
        logger.warning("⚠️ Re-placing TP/SL after failed market close")
        place_tp_sl_orders(exchange, state, config, position=position)
    except Exception as restore_e:
        logger.error(f"Failed to re-place TP/SL after market close failure: {restore_e}")

    return False


def _cancel_all_symbol_orders(exchange: ccxt.bingx, symbol: str) -> int:
    """Cancel all open orders for a symbol (clean slate for recovery).

    Returns number of orders cancelled.
    """
    try:
        open_orders = exchange.fetch_open_orders(symbol)
        cancelled = 0
        for order in open_orders:
            try:
                exchange.cancel_order(order['id'], symbol)
                cancelled += 1
            except (ccxt.OrderNotFound, ccxt.InvalidOrder):
                pass
            except Exception as e:
                logger.debug(f"Failed to cancel order {order.get('id')}: {e}")
        if cancelled:
            logger.info(f"🧹 Cancelled {cancelled} open orders for clean recovery")
        return cancelled
    except Exception as e:
        logger.warning(f"Failed to fetch/cancel orders for recovery: {e}")
        return 0


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

    Phase 1: Cancel all open orders (clean slate, prevents orphaned order conflicts)
    Phase 2: Ghost position detection
    Phase 3: Reconcile exchange vs local state (cases 1-3)
    Phase 4: Re-place TP/SL for all surviving slots + emergency SL

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

    # Phase 0: Snapshot per-slot TP/SL from exchange BEFORE cancellation
    tpsl_snapshots = {}  # direction → list of (tp, sl) pairs
    for dir_label in ('LONG', 'SHORT'):
        pairs = _snapshot_all_tpsl(exchange, symbol, dir_label)
        if pairs:
            tpsl_snapshots[dir_label] = pairs

    # Phase 1: Cancel all open orders (prevents orphaned TP/SL conflicts)
    _cancel_all_symbol_orders(exchange, symbol)
    # Clear stale order IDs from all slots (orders are now cancelled)
    for slot in (state.get('positions') or {}).values():
        slot['tp_order_id'] = None
        slot['sl_order_id'] = None
    state['emergency_sl_order_id'] = None
    state['emergency_sl_orders'] = {}

    # Phase 2: Ghost position detection (log warnings, don't block)
    detect_ghost_positions(exchange, state, config, cache, circuit_breaker, metrics)

    try:
        exchange_positions = fetch_positions_cached(exchange, symbol, cache, force_refresh=True,
                                                     circuit_breaker=circuit_breaker, metrics=metrics)

        # Build exchange map by direction
        exchange_map = {}  # 'long'/'short' → exchange position dict
        for pos in exchange_positions:
            contracts = float(pos.get('contracts', 0))
            if contracts > 0:
                exchange_map[pos.get('side')] = pos

        bot_slots = state.get('positions') or {}
        recovery_needed = False

        # Phase 3: Check each direction
        for dir_label, dir_key in [('LONG', 'long'), ('SHORT', 'short')]:
            dir_slots = {sid: s for sid, s in bot_slots.items() if s.get('direction') == dir_label}
            exchange_pos = exchange_map.get(dir_key)

            # Case 1: Exchange has position, bot has no slots for this direction → orphan
            if exchange_pos and not dir_slots:
                logger.warning(f"🔧 Recovery: Found orphan {dir_label} position on exchange")
                recover_position_to_state(
                    state, config, exchange_pos, dir_label, exchange, cache,
                    saved_tpsl_pairs=tpsl_snapshots.get(dir_label),
                )
                recovery_needed = True
                continue

            # Case 2: Bot has slots, exchange has no position for this direction → closed
            if dir_slots and not exchange_pos:
                logger.warning(f"🔧 Recovery: {len(dir_slots)} {dir_label} slot(s) not found on exchange")
                for slot in list(dir_slots.values()):
                    actual_exit = get_actual_exit_price(exchange, state, config, position=slot)
                    if actual_exit:
                        record_closed_position(exchange, state, config, actual_exit['price'],
                                              actual_exit['reason'], cache, metrics, position=slot)
                    else:
                        try:
                            ticker = fetch_ticker_cached(exchange, config['symbol'], cache, force_refresh=True)
                            fallback_price = ticker['last']
                        except Exception:
                            fallback_price = slot['entry_price']
                        record_closed_position(exchange, state, config, fallback_price,
                                              'CRASH_RECOVERY', cache, metrics, position=slot)
                recovery_needed = True
                continue

            # Case 3: Both exist — check quantity mismatch
            if exchange_pos and dir_slots:
                ex_qty = float(exchange_pos.get('contracts', 0))
                local_qty_sum = sum(s.get('quantity', 0) for s in dir_slots.values())

                if abs(ex_qty - local_qty_sum) > QTY_TOLERANCE:
                    logger.warning(
                        f"🔧 Recovery: {dir_label} qty mismatch "
                        f"(exchange={ex_qty:.4f}, local_sum={local_qty_sum:.4f})"
                    )
                    if ex_qty < local_qty_sum:
                        # Some slots closed externally — remove oldest FIFO
                        sorted_slots = sorted(dir_slots.items(), key=lambda x: x[1].get('entry_time', ''))
                        remaining = local_qty_sum
                        for sid, slot in sorted_slots:
                            if abs(remaining - ex_qty) <= QTY_TOLERANCE:
                                break
                            slot_qty = slot.get('quantity', 0)
                            if remaining - slot_qty >= ex_qty - QTY_TOLERANCE:
                                actual_exit = get_actual_exit_price(exchange, state, config, position=slot)
                                if actual_exit:
                                    record_closed_position(exchange, state, config, actual_exit['price'],
                                                          actual_exit['reason'], cache, metrics, position=slot)
                                else:
                                    try:
                                        ticker = fetch_ticker_cached(exchange, config['symbol'], cache, force_refresh=True)
                                        fallback_price = ticker['last']
                                    except Exception:
                                        fallback_price = slot.get('entry_price', 0)
                                    record_closed_position(exchange, state, config, fallback_price,
                                                          'CRASH_RECOVERY', cache, metrics, position=slot)
                                remaining -= slot_qty
                            else:
                                break
                    else:
                        # Exchange has more qty — create new slot if room, else absorb
                        diff = round(ex_qty - local_qty_sum, QUANTITY_ROUND_DECIMALS)
                        max_pos = config.get('max_positions', 1)
                        available = max_pos - len(dir_slots)

                        if available > 0 and diff > 0:
                            # Use TP/SL from existing slots (same direction)
                            ref_slot = next(iter(dir_slots.values()))
                            ep = float(exchange_pos.get('entryPrice', ref_slot.get('entry_price', 0)))
                            tp_p = ref_slot.get('tp_price')
                            sl_p = ref_slot.get('sl_price')
                            if not tp_p or not sl_p:
                                dm = 1 if dir_label == 'LONG' else -1
                                ref_pat = (
                                    extract_pattern_name(ref_slot.get('reason', ''))
                                    or ref_slot.get('pattern_name')
                                )
                                tp_p, sl_p, _, _ = calculate_tp_sl(
                                    ep, dm, config['strategy'],
                                    vol_mult=1.0, pattern=ref_pat, config=config,
                                )
                            new_sid = uuid.uuid4().hex[:8]
                            ref_pattern = (
                                extract_pattern_name(ref_slot.get('reason', ''))
                                or ref_slot.get('pattern_name')
                            )
                            new_slot = {
                                'slot_id': new_sid,
                                'direction': dir_label,
                                'entry_price': ep,
                                'quantity': diff,
                                'remaining_quantity': diff,
                                'tp_price': tp_p,
                                'sl_price': sl_p,
                                'vol_mult': 1.0,
                                'scale_out_enabled': False,
                                'scale_out_stages': [],
                                'entry_time': datetime.now().isoformat(),
                                'reason': f'Recovered from exchange ({ref_pattern})' if ref_pattern else 'Recovered from exchange',
                                'pattern_name': ref_pattern,
                                'recovered': True,
                                'needs_tpsl': True,
                            }
                            state.setdefault('positions', {})[new_sid] = new_slot
                            logger.info(f"Created recovery slot {new_sid} for +{diff:.4f} excess qty")
                        else:
                            first_slot = next(iter(dir_slots.values()))
                            orig_rem = first_slot.get('remaining_quantity', first_slot['quantity'])
                            first_slot['quantity'] = round(first_slot['quantity'] + diff, QUANTITY_ROUND_DECIMALS)
                            first_slot['remaining_quantity'] = round(orig_rem + diff, QUANTITY_ROUND_DECIMALS)
                            logger.info(f"Absorbed +{diff} qty into slot {first_slot.get('slot_id')}")
                    recovery_needed = True

        # Phase 4: Re-place TP/SL for all surviving slots (orders were cancelled in Phase 1)
        remaining_slots = state.get('positions') or {}
        if remaining_slots:
            for slot in remaining_slots.values():
                if not slot.get('tp_order_id') or not slot.get('sl_order_id'):
                    place_tp_sl_orders(exchange, state, config, position=slot)
            update_emergency_sl(exchange, state, config)

        save_state(state)

        if not recovery_needed:
            logger.info("✅ Crash recovery check passed - state is consistent")

        return recovery_needed

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