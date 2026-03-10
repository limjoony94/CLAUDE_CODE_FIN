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
from .orders import _EXCHANGE_MANAGED, update_single_sl

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

    Compares bot's position slots against exchange positions per direction.
    Handles: orphan recovery, external closures, direction mismatches.

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
    from .position_close import record_closed_position, recover_position_to_state

    symbol = config['symbol']

    try:
        exchange_positions = fetch_positions_cached(exchange, symbol, cache, force_refresh=True,
                                                     circuit_breaker=circuit_breaker, metrics=metrics)

        exchange_map = {}  # 'long'/'short' → exchange position dict
        for pos in exchange_positions:
            contracts = float(pos.get('contracts', 0))
            if contracts > 0:
                exchange_map[pos.get('side')] = pos

        bot_slots = state.get('positions') or {}
        bot_long_slots = [s for s in bot_slots.values() if s.get('direction') == 'LONG']
        bot_short_slots = [s for s in bot_slots.values() if s.get('direction') == 'SHORT']

        logger.debug(
            f"Position sync: exchange_long={'yes' if 'long' in exchange_map else 'no'}, "
            f"exchange_short={'yes' if 'short' in exchange_map else 'no'}, "
            f"bot_slots={len(bot_slots)} (L:{len(bot_long_slots)} S:{len(bot_short_slots)})"
        )

        sync_needed = False

        # Check each direction
        for dir_label, dir_key, bot_dir_slots in [
            ('LONG', 'long', bot_long_slots),
            ('SHORT', 'short', bot_short_slots),
        ]:
            exchange_pos = exchange_map.get(dir_key)

            # Bot has slots but exchange has no position for this direction → externally closed
            if bot_dir_slots and not exchange_pos:
                logger.warning(f"Bot has {len(bot_dir_slots)} {dir_label} slot(s) but exchange has none — closing")
                for slot in list(bot_dir_slots):
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
                                              'EXTERNAL', cache, metrics, position=slot)
                sync_needed = True

        # Check for orphan exchange positions (exchange has position, no bot slots)
        for dir_label, dir_key, bot_dir_slots in [
            ('LONG', 'long', bot_long_slots),
            ('SHORT', 'short', bot_short_slots),
        ]:
            exchange_pos = exchange_map.get(dir_key)
            if exchange_pos and not bot_dir_slots:
                logger.info(f"Exchange has {dir_label} position but no bot slots — recovering")
                recover_position_to_state(state, config, exchange_pos, dir_label, exchange, cache)
                sync_needed = True

        if not sync_needed:
            logger.info("Position sync completed - state matches exchange")

        return sync_needed

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
    position: Optional[Dict] = None,
) -> Optional[Dict]:
    """
    Get actual exit price from trade history.

    Args:
        exchange: Exchange instance
        state: Bot state dictionary
        config: Bot configuration
        position: Specific position slot (if None, uses first active slot)

    Returns:
        Dict with 'price' and 'reason', or None
    """
    symbol = config['symbol']
    if position is None:
        positions = state.get('positions') or {}
        if not positions:
            return None
        position = next(iter(positions.values()))

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

        # In hedge mode, also filter by positionSide to avoid
        # confusing a SHORT open with a LONG close (both are 'sell')
        expected_pos_side = position.get('direction', '').upper()  # 'LONG' or 'SHORT'

        for trade in reversed(trades):
            if trade.get('side') == close_side:
                # Hedge mode: check positionSide matches
                trade_pos_side = (
                    (trade.get('info') or {}).get('positionSide', 'BOTH')
                ).upper()
                if trade_pos_side != 'BOTH' and trade_pos_side != expected_pos_side:
                    continue

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
    """Infer exit reason from filled price proximity to TP/SL.

    v1.55.1: Widened thresholds to reduce MARKET/UNKNOWN misclassifications.
    - Near-SL zone: 40% → 60% of entry-to-SL range
    - TP tolerance: 0.3% → 0.5% (via PRICE_TOLERANCE_PCT constant)
    - Near-TP zone: 30% → 50% of entry-to-TP range
    - CASCADE_SL: also detects via cascade_tightened flag and prior SL levels
    Priority: TP > SL (exact) > EMERGENCY_SL > CASCADE_SL (flagged) > SL (near) > CASCADE_SL (inferred) > MARKET.
    """
    tp = position.get('tp_price', 0)
    sl = position.get('sl_price', 0)
    entry = position.get('entry_price', 0)
    direction = position.get('direction', '')

    if tp and abs(filled_price - tp) / tp < PRICE_TOLERANCE_PCT:
        return 'TP'
    if sl and abs(filled_price - sl) / sl < PRICE_TOLERANCE_PCT:
        return 'SL'

    # If price is beyond SL (worse), likely emergency SL or cascade liquidation
    if direction == 'LONG' and sl and filled_price < sl:
        return 'EMERGENCY_SL'
    if direction == 'SHORT' and sl and filled_price > sl:
        return 'EMERGENCY_SL'

    # v1.55.1: CASCADE_SL detection — if position was cascade-tightened and
    # exit is in the loss direction, classify as CASCADE_SL immediately.
    # Also check against prior SL levels stored during cascade tightening.
    if position.get('cascade_tightened'):
        if direction == 'LONG' and entry > 0 and filled_price < entry:
            # Check against prior SL levels (cascade may have moved SL multiple times)
            prior_sls = position.get('cascade_prior_sls') or []
            for prior_sl in prior_sls:
                if prior_sl > 0 and abs(filled_price - prior_sl) / prior_sl < PRICE_TOLERANCE_PCT:
                    return 'CASCADE_SL'
            # General loss-side exit on cascade-tightened position
            if sl and sl < filled_price < entry:
                return 'CASCADE_SL'
        if direction == 'SHORT' and entry > 0 and filled_price > entry:
            prior_sls = position.get('cascade_prior_sls') or []
            for prior_sl in prior_sls:
                if prior_sl > 0 and abs(filled_price - prior_sl) / prior_sl < PRICE_TOLERANCE_PCT:
                    return 'CASCADE_SL'
            if sl and entry < filled_price < sl:
                return 'CASCADE_SL'

    # v1.55.1: Near-SL classification — if price is between entry and SL,
    # and closer to SL than to entry (within SL-side 60% of range), classify as SL.
    # This captures slippage-affected SL fills and cascade-tightened SL hits.
    if entry > 0 and sl > 0:
        sl_to_entry = abs(entry - sl)
        if sl_to_entry > 0:
            if direction == 'LONG' and sl < filled_price < entry:
                sl_proximity = abs(filled_price - sl) / sl_to_entry
                if sl_proximity < 0.6:  # within 60% of SL→entry range from SL
                    return 'SL'
            if direction == 'SHORT' and entry < filled_price < sl:
                sl_proximity = abs(filled_price - sl) / sl_to_entry
                if sl_proximity < 0.6:
                    return 'SL'

    # Cascade SL inference: price between entry and SL (loss direction) but not
    # near current SL — likely hit a pre-cascade SL or exchange closed during
    # cascade chain.  Only classify as CASCADE_SL when position was losing.
    if entry > 0 and sl > 0:
        if direction == 'LONG' and sl < filled_price < entry:
            return 'CASCADE_SL'
        if direction == 'SHORT' and entry < filled_price < sl:
            return 'CASCADE_SL'

    # v1.55.1: Near-TP classification — profitable exit near TP
    if entry > 0 and tp > 0:
        tp_to_entry = abs(tp - entry)
        if tp_to_entry > 0:
            if direction == 'LONG' and filled_price > entry:
                tp_proximity = abs(filled_price - tp) / tp_to_entry
                if tp_proximity < 0.5:  # within 50% of entry→TP range from TP
                    return 'TP'
            if direction == 'SHORT' and filled_price < entry:
                tp_proximity = abs(filled_price - tp) / tp_to_entry
                if tp_proximity < 0.5:
                    return 'TP'

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
    Check if any position slot has been closed.

    Iterates all active slots, groups by direction, and checks
    against exchange positions.

    Args:
        exchange: Exchange instance
        state: Bot state dictionary
        config: Bot configuration
        cache: APICache instance
        circuit_breaker: Optional CircuitBreaker
        metrics: Optional PerformanceMetrics

    Returns:
        True if any position was closed
    """
    bot_slots = state.get('positions') or {}
    if not bot_slots:
        return False

    symbol = config['symbol']

    try:
        exchange_positions = fetch_positions_cached(exchange, symbol, cache, force_refresh=True,
                                                     circuit_breaker=circuit_breaker, metrics=metrics)
        exchange_map = {}  # 'long'/'short' → exchange position dict
        for pos in exchange_positions:
            contracts = float(pos.get('contracts', 0))
            if contracts > 0:
                exchange_map[pos.get('side')] = pos

        any_closed = False

        # Group bot slots by direction
        for dir_label, dir_key in [('LONG', 'long'), ('SHORT', 'short')]:
            dir_slots = [s for s in bot_slots.values() if s.get('direction') == dir_label]
            if not dir_slots:
                continue

            exchange_pos = exchange_map.get(dir_key)

            # All slots for this direction are closed (exchange has no position)
            if exchange_pos is None or float(exchange_pos.get('contracts', 0)) == 0:
                # v1.55.0: Mass closure verification — if 3+ slots appear closed
                # simultaneously, re-fetch positions to guard against API glitches
                # (03-05 incident: network error returned empty positions → false closure)
                if len(dir_slots) >= 3:
                    logger.warning(
                        f"⚠️ {len(dir_slots)} {dir_label} slots appear closed — "
                        f"verifying with fresh API call..."
                    )
                    time.sleep(1)
                    try:
                        verify_positions = fetch_positions_cached(
                            exchange, symbol, cache, force_refresh=True,
                            circuit_breaker=circuit_breaker, metrics=metrics
                        )
                        for vp in verify_positions:
                            if (float(vp.get('contracts', 0)) > 0
                                    and vp.get('side') == dir_key):
                                logger.warning(
                                    f"⚠️ FALSE ALARM: {dir_label} position still exists "
                                    f"on exchange after re-check — skipping closure"
                                )
                                exchange_pos = vp
                                break
                    except Exception as e:
                        logger.warning(
                            f"⚠️ Mass closure verification failed: {e} — "
                            f"deferring to next cycle"
                        )
                        continue  # Skip this direction entirely, retry next loop

                if exchange_pos is None or float(exchange_pos.get('contracts', 0)) == 0:
                    if len(dir_slots) > 1:
                        total_qty = sum(s.get('quantity', 0) for s in dir_slots)
                        logger.warning(
                            f"⚠️ CASCADE: {len(dir_slots)} {dir_label} slots closed simultaneously "
                            f"(qty={total_qty:.4f}). Likely emergency SL or exchange liquidation."
                        )
                    for slot in list(dir_slots):
                        _handle_position_closed(exchange, state, config, slot, cache, metrics)
                    any_closed = True
                continue

            # Per-slot TP/SL fill detection (N>1 multi-position)
            # Exchange has position but with fewer contracts than local sum
            # → some individual slot TP/SL orders were filled
            exchange_qty = float(exchange_pos.get('contracts', 0))
            local_qty_sum = sum(s.get('quantity', 0) for s in dir_slots)

            if exchange_qty < local_qty_sum - 0.0001:
                try:
                    open_orders = exchange.fetch_open_orders(symbol)
                    open_order_ids = {o.get('id') for o in open_orders}
                except Exception as e:
                    logger.debug(f"Per-slot fill detection: failed to fetch open orders: {e}")
                    open_order_ids = None

                if open_order_ids is not None:
                    for slot in list(dir_slots):
                        tp_id = slot.get('tp_order_id')
                        sl_id = slot.get('sl_order_id')

                        # Skip EXCHANGE_MANAGED slots
                        if tp_id == _EXCHANGE_MANAGED or sl_id == _EXCHANGE_MANAGED:
                            continue

                        # Need at least one tracked order to detect fill
                        if not tp_id and not sl_id:
                            continue

                        tp_on_exchange = tp_id in open_order_ids if tp_id else False
                        sl_on_exchange = sl_id in open_order_ids if sl_id else False

                        slot_closed = False
                        if tp_id and not tp_on_exchange and sl_on_exchange:
                            logger.info(
                                f"🔍 Per-slot fill: {dir_label} slot {slot.get('slot_id')} "
                                f"TP order {tp_id} filled (SL {sl_id} still active)"
                            )
                            slot_closed = True
                        elif sl_id and not sl_on_exchange and tp_on_exchange:
                            logger.info(
                                f"🔍 Per-slot fill: {dir_label} slot {slot.get('slot_id')} "
                                f"SL order {sl_id} filled (TP {tp_id} still active)"
                            )
                            slot_closed = True
                        elif tp_id and not tp_on_exchange and sl_id and not sl_on_exchange:
                            logger.info(
                                f"🔍 Per-slot fill: {dir_label} slot {slot.get('slot_id')} "
                                f"both orders gone (TP {tp_id}, SL {sl_id})"
                            )
                            slot_closed = True

                        if slot_closed:
                            _handle_position_closed(
                                exchange, state, config, slot, cache, metrics
                            )
                            any_closed = True

            # Handle scale-out partial fills per slot
            for slot in dir_slots:
                scale_out_enabled = slot.get('scale_out_enabled', False)
                if scale_out_enabled:
                    _check_scale_out_fills(state, slot, exchange_pos)

        return any_closed

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
    """Handle position slot closure detection."""
    from .position_close import record_closed_position

    slot_id = position.get('slot_id', 'unknown')
    logger.info(
        f"🔍 Position closure detected: {position['direction']} slot {slot_id} "
        f"(entry=${position['entry_price']:.1f}), fetching exit details..."
    )
    time.sleep(EXIT_PRICE_INITIAL_DELAY)

    actual_exit = None
    for retry in range(MAX_EXIT_PRICE_RETRIES):
        actual_exit = get_actual_exit_price(exchange, state, config, position=position)
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

    # v1.41.0: Cascade SL tightening — tighten same-dir SLs before slot removal
    if exit_reason in ('SL', 'EMERGENCY_SL', 'CASCADE_SL') or exit_reason.startswith('SL_AFTER_'):
        _cascade_tighten_sls(exchange, state, config, position)

    record_closed_position(exchange, state, config, exit_price, exit_reason, cache, metrics, position=position)
    return True


def _cascade_tighten_sls(
    exchange: ccxt.bingx,
    state: Dict[str, Any],
    config: Dict[str, Any],
    closed_position: Dict,
) -> None:
    """Tighten SL orders on remaining same-direction positions after an SL exit.

    v1.41.0: Correlated loss study H5_Cascade_t75 — reduce SL distance
    to (1 - tighten_pct/100) of current distance. Cascading: each
    subsequent SL exit recalculates from current sl_price.
    """
    cascade_cfg = config.get('risk', {}).get('cascade_sl_tightening', {})
    if not cascade_cfg.get('enabled', False):
        return

    tighten_pct = cascade_cfg.get('tighten_pct', 75)
    keep_ratio = 1.0 - tighten_pct / 100.0  # e.g. 75% → keep 0.25

    direction = closed_position.get('direction', '')
    closed_slot = closed_position.get('slot_id', 'unknown')
    positions = state.get('positions') or {}

    same_dir = [
        (sid, pos) for sid, pos in positions.items()
        if pos.get('direction') == direction and sid != closed_slot
    ]

    if not same_dir:
        return

    logger.info(
        f"🔗 CASCADE SL: {direction} SL exit (slot {closed_slot}) — "
        f"tightening {len(same_dir)} same-dir positions to {keep_ratio:.0%} SL distance"
    )

    for sid, pos in same_dir:
        entry = pos.get('entry_price', 0)
        old_sl = pos.get('sl_price', 0)
        if entry <= 0 or old_sl <= 0:
            continue

        old_dist = abs(entry - old_sl)
        new_dist = old_dist * keep_ratio

        if direction == 'LONG':
            new_sl = round(entry - new_dist, 1)
        else:
            new_sl = round(entry + new_dist, 1)

        logger.info(
            f"  CASCADE slot {sid}: SL ${old_sl:.1f} → ${new_sl:.1f} "
            f"(dist {old_dist:.1f} → {new_dist:.1f})"
        )

        # v1.55.1: Track cascade tightening for exit classification
        pos['cascade_tightened'] = True
        prior_sls = pos.get('cascade_prior_sls') or []
        prior_sls.append(old_sl)
        pos['cascade_prior_sls'] = prior_sls

        success = update_single_sl(exchange, pos, config, new_sl)
        if not success:
            pos['sl_price'] = old_sl
            logger.critical(
                f"  CASCADE slot {sid}: SL update FAILED — reverted to ${old_sl:.1f}. "
                f"POSITION MAY BE UNPROTECTED until next verify cycle."
            )

    save_state(state)


def _infer_exit_from_price(exit_price: float, position: Dict) -> str:
    """Infer exit reason from current price vs TP/SL levels.

    v1.55.1: Widened thresholds to match _infer_exit_reason().
    - Near-SL zone: 40% → 60%
    - Near-TP zone: 30% → 50%
    - CASCADE_SL: also detects via cascade_tightened flag
    """
    tp = position.get('tp_price', 0)
    sl = position.get('sl_price', 0)
    entry = position.get('entry_price', 0)
    direction = position.get('direction', '')

    if direction == 'LONG':
        if tp > 0 and exit_price >= tp * TP_LOWER_MULT:
            return 'TP'
        elif sl > 0 and exit_price <= sl * SL_UPPER_MULT:
            return 'SL'
    else:  # SHORT
        if tp > 0 and exit_price <= tp * TP_UPPER_MULT:
            return 'TP'
        elif sl > 0 and exit_price >= sl * SL_LOWER_MULT:
            return 'SL'

    # v1.55.1: CASCADE_SL detection via flag
    if position.get('cascade_tightened'):
        if direction == 'LONG' and entry > 0 and exit_price < entry:
            prior_sls = position.get('cascade_prior_sls') or []
            for prior_sl in prior_sls:
                if prior_sl > 0 and abs(exit_price - prior_sl) / prior_sl < PRICE_TOLERANCE_PCT:
                    return 'CASCADE_SL'
            if sl and sl < exit_price < entry:
                return 'CASCADE_SL'
        if direction == 'SHORT' and entry > 0 and exit_price > entry:
            prior_sls = position.get('cascade_prior_sls') or []
            for prior_sl in prior_sls:
                if prior_sl > 0 and abs(exit_price - prior_sl) / prior_sl < PRICE_TOLERANCE_PCT:
                    return 'CASCADE_SL'
            if sl and entry < exit_price < sl:
                return 'CASCADE_SL'

    # v1.55.1: Near-SL classification (within 60% of SL-to-entry distance)
    if entry > 0 and sl > 0:
        sl_to_entry = abs(entry - sl)
        if sl_to_entry > 0:
            if direction == 'LONG' and sl < exit_price < entry:
                sl_proximity = abs(exit_price - sl) / sl_to_entry
                if sl_proximity < 0.6:
                    return 'SL'
                return 'CASCADE_SL'
            if direction == 'SHORT' and entry < exit_price < sl:
                sl_proximity = abs(exit_price - sl) / sl_to_entry
                if sl_proximity < 0.6:
                    return 'SL'
                return 'CASCADE_SL'

    # v1.55.1: Near-TP classification (within 50% of TP-to-entry distance)
    if entry > 0 and tp > 0:
        tp_to_entry = abs(tp - entry)
        if tp_to_entry > 0:
            if direction == 'LONG' and exit_price > entry:
                tp_proximity = abs(exit_price - tp) / tp_to_entry
                if tp_proximity < 0.5:
                    return 'TP'
            if direction == 'SHORT' and exit_price < entry:
                tp_proximity = abs(exit_price - tp) / tp_to_entry
                if tp_proximity < 0.5:
                    return 'TP'

    return 'UNKNOWN'