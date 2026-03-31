"""
Pattern 5m Bot - Position Management

Extracted from bot.py: time-decay TP, pre-emptive cascade SL, position timeouts.
Pure refactoring — no behavioral changes.
"""

import logging
import time
from datetime import datetime
from typing import Dict, Any

import ccxt

from .constants import (
    CANDLE_DURATION_MS,
    DEFAULT_TIMEOUT_BARS,
)
from .models import APICache, CircuitBreaker, PerformanceMetrics
from .state import save_state
from .exchange import fetch_ticker_cached
from .orders import update_single_tp, update_single_sl
from .position import close_position_market

logger = logging.getLogger('pattern_5m')


def _apply_time_decay_tp(
    exchange: ccxt.bingx,
    state: Dict[str, Any],
    config: Dict[str, Any],
) -> None:
    """
    Apply exponential time-decay to TP orders from bar 0.

    v1.62.0→v1.63.0: Exponential decay — TP_distance *= decay_rate^bars_held.
    Default decay_rate=0.997 (half-life ~231 bars = 19.25h).
    TP moves toward entry over time, accelerating slot turnover.
    Updates exchange TP order every update_interval_bars (default 6 = 30min).

    Research: time_decay_tp.py — exp_decay OOS +332.5% vs baseline +299.7%
    (+10.9%), WF 3/3 PASS, IS trades +40% (slot turnover), 100% MECHANICAL.
    """
    decay_cfg = config.get('strategy', {}).get('tp_decay', {})
    if not decay_cfg.get('enabled'):
        return

    positions = state.get('positions') or {}
    if not positions:
        return

    decay_rate = decay_cfg.get('decay_rate', 0.997)
    update_interval = decay_cfg.get('update_interval_bars', 6)

    now = datetime.now()
    candle_seconds = CANDLE_DURATION_MS / 1000
    decay_api_calls = 0

    for slot_id, pos in list(positions.items()):
        if pos.get('_pending_close'):
            continue

        entry_time_str = pos.get('entry_time', '')
        if not entry_time_str:
            continue

        try:
            entry_dt = datetime.fromisoformat(entry_time_str)
        except (ValueError, TypeError):
            continue

        held_seconds = (now - entry_dt).total_seconds()
        bars_held = int(held_seconds / candle_seconds)

        # Skip very fresh positions (< 1 update interval)
        if bars_held < update_interval:
            continue

        # Store original TP on first decay (one-time)
        if 'tp_price_original' not in pos:
            pos['tp_price_original'] = pos.get('tp_price', 0)
            logger.info(
                f"⏰ DECAY START: Slot {slot_id} ({pos.get('pattern_name', '?')} "
                f"{pos.get('direction', '?')}) bar {bars_held} — "
                f"original TP=${pos['tp_price_original']:.1f}"
            )

        # Only update at interval boundaries to avoid excessive API calls
        last_decay_bar = pos.get('_last_decay_bar', 0)
        if bars_held - last_decay_bar < update_interval:
            continue

        # Exponential decay: TP_distance *= decay_rate^bars_held
        decay_factor = decay_rate ** bars_held

        entry_price = pos.get('entry_price', 0)
        original_tp = pos.get('tp_price_original', pos.get('tp_price', 0))
        direction = pos.get('direction', '')

        # Calculate decayed TP price from original TP distance
        if direction == 'LONG':
            original_dist = original_tp - entry_price
            decayed_tp = entry_price + original_dist * decay_factor
            # Safety: never decay TP below entry (break-even)
            decayed_tp = max(decayed_tp, entry_price + 1.0)
        else:
            original_dist = entry_price - original_tp
            decayed_tp = entry_price - original_dist * decay_factor
            # Safety: never decay TP above entry (break-even)
            decayed_tp = min(decayed_tp, entry_price - 1.0)

        decayed_tp = round(decayed_tp, 1)

        # Check if TP actually changed enough to warrant exchange update
        current_tp = pos.get('tp_price', 0)
        if abs(current_tp - decayed_tp) <= 1.0:
            pos['_last_decay_bar'] = bars_held
            continue

        # Rate limit protection between decay updates
        if decay_api_calls > 0:
            time.sleep(0.3)

        # Update TP on exchange
        logger.info(
            f"⏰ DECAY: Slot {slot_id} ({pos.get('pattern_name', '?')} {direction}) "
            f"bar {bars_held} decay={decay_factor:.3f} — "
            f"TP ${current_tp:.1f} → ${decayed_tp:.1f} "
            f"(original ${original_tp:.1f})"
        )
        success = update_single_tp(exchange, pos, config, decayed_tp)
        decay_api_calls += 1
        pos['_last_decay_bar'] = bars_held

        if success:
            save_state(state)


def _apply_preemptive_cascade(
    exchange: ccxt.bingx,
    state: Dict[str, Any],
    config: Dict[str, Any],
    cache: APICache,
) -> None:
    """
    Pre-emptive Cascade SL: tighten SLs when unrealized directional loss exceeds threshold.

    v1.65.0: sl_cluster_defense study — PRE_CASCADE_5% variant.
    Unlike reactive cascade (which triggers AFTER SL hit), this triggers BEFORE SL hit
    when the combined unrealized loss of same-direction positions exceeds a threshold.

    This prevents simultaneous multi-SL events that reactive cascade can't catch
    (multiple positions hitting SL on the same bar before cascade can fire).

    Research: sl_cluster_defense.py — IS PnL +491.7% (baseline +453.6%), MDD 3.7% (4.6%),
    PnL/MDD 133.6x (99.3x), R:R 1.040 (0.753), WF OOS +371.8% (baseline +341.0%).
    """
    pre_cfg = config.get('risk', {}).get('preemptive_cascade', {})
    if not pre_cfg.get('enabled', False):
        return

    positions = state.get('positions') or {}
    if len(positions) < 2:
        return

    threshold_pct = pre_cfg.get('unrealized_loss_pct', 5.0)
    tighten_pct = pre_cfg.get('tighten_pct', 85)
    keep_ratio = 1.0 - tighten_pct / 100.0

    # Get current price
    try:
        ticker = fetch_ticker_cached(exchange, config['symbol'], cache)
        current_price = ticker['last']
    except Exception:
        return

    if not current_price or current_price <= 0:
        return

    state_changed = False

    leverage = config.get('leverage', 3)
    n_slots = config.get('max_positions', 9)

    for direction in ['LONG', 'SHORT']:
        # v1.68.0: Compute unrealized loss using ALL dir positions (including already-cascaded)
        # but only APPLY tightening to non-cascaded positions.
        # This ensures threshold reflects TRUE directional exposure.
        all_dir_positions = [
            (sid, pos) for sid, pos in positions.items()
            if pos.get('direction') == direction
            and not pos.get('_pending_close')
        ]
        actionable_positions = [
            (sid, pos) for sid, pos in all_dir_positions
            if not pos.get('_preemptive_cascaded')
        ]

        if len(all_dir_positions) < 2 or len(actionable_positions) < 1:
            continue

        # Compute combined unrealized loss for ALL positions in this direction
        total_unrealized_loss = 0.0
        n_losing = 0
        for sid, pos in all_dir_positions:
            entry = pos.get('entry_price', 0)
            if entry <= 0:
                continue
            if direction == 'LONG':
                unr_pct = (current_price / entry - 1) * 100 * leverage
            else:
                unr_pct = (1 - current_price / entry) * 100 * leverage

            if unr_pct < 0:
                total_unrealized_loss += abs(unr_pct) / n_slots
                n_losing += 1

        if total_unrealized_loss < threshold_pct:
            continue

        # Threshold exceeded — tighten SL for actionable (non-cascaded) positions
        logger.warning(
            f"🔗 PRE-EMPTIVE CASCADE: {direction} unrealized loss "
            f"{total_unrealized_loss:.1f}% > {threshold_pct}% threshold "
            f"({n_losing} losing of {len(all_dir_positions)} positions, "
            f"{len(actionable_positions)} actionable) — "
            f"tightening SLs to {keep_ratio:.0%} distance"
        )

        for sid, pos in actionable_positions:
            entry = pos.get('entry_price', 0)
            old_sl = pos.get('sl_price', 0)
            if entry <= 0 or old_sl <= 0:
                continue

            # v1.69.1: Entry-based cascade using _sl_price_original (v1.67.1 fix).
            # v1.68.0 hardcoded 10% keep instead of config tighten_pct.
            original_sl = pos.get('_sl_price_original', old_sl)
            original_dist = abs(entry - original_sl)
            new_dist = max(original_dist * keep_ratio, 30.0)

            if direction == 'LONG':
                new_sl = round(entry - new_dist, 1)
            else:
                new_sl = round(entry + new_dist, 1)

            # Only tighten, never widen
            if direction == 'LONG' and new_sl <= old_sl:
                continue
            if direction == 'SHORT' and new_sl >= old_sl:
                continue

            # Validate against current price
            if direction == 'LONG' and new_sl >= current_price:
                logger.warning(
                    f"  PRE-CASCADE slot {sid}: SKIP — SL ${new_sl:.1f} >= current ${current_price:.1f}")
                continue
            if direction == 'SHORT' and new_sl <= current_price:
                logger.warning(
                    f"  PRE-CASCADE slot {sid}: SKIP — SL ${new_sl:.1f} <= current ${current_price:.1f}")
                continue

            logger.info(
                f"  PRE-CASCADE slot {sid}: SL ${old_sl:.1f} → ${new_sl:.1f} "
                f"(dist {abs(entry - old_sl):.1f} → {new_dist:.1f})"
            )

            pos['cascade_tightened'] = True
            pos['_preemptive_cascaded'] = True
            prior_sls = pos.get('cascade_prior_sls') or []
            prior_sls.append(old_sl)
            pos['cascade_prior_sls'] = prior_sls

            if state_changed:
                time.sleep(0.3)  # Rate limit protection
            success = update_single_sl(exchange, pos, config, new_sl)
            if success:
                state_changed = True
            else:
                logger.critical(
                    f"  PRE-CASCADE slot {sid}: SL update FAILED — "
                    f"position may be at risk"
                )

    if state_changed:
        save_state(state)


def _apply_trailing_stop(
    exchange: ccxt.bingx,
    state: Dict[str, Any],
    config: Dict[str, Any],
    cache: APICache,
) -> None:
    """v1.68.0: Ensure exchange-native TRAILING_STOP_MARKET order exists.

    Checks each position for trail_order_id. If missing, places one.
    Exchange handles real-time trail tracking (no loop delay).
    """
    trail_cfg = config.get('strategy', {}).get('trailing_stop', {})
    if not trail_cfg.get('enabled', False):
        return

    positions = state.get('positions') or {}
    if not positions:
        return

    activation_pct = trail_cfg.get('activation_pct', 1.0)
    trail_pct = trail_cfg.get('trail_pct', 0.3)
    symbol = config.get('symbol', 'BTC-USDT')

    from .position_open import _place_trailing_stop_order

    for slot_id, pos in list(positions.items()):
        if pos.get('_pending_close'):
            continue
        if pos.get('trail_order_id'):
            continue  # already has trail order

        entry_price = pos.get('entry_price', 0)
        if entry_price <= 0:
            continue

        logger.info(f"📈 TRAIL: Placing exchange-native trail for {slot_id[:8]} "
                     f"{pos.get('direction','')} {pos.get('pattern_name','?')}")
        _place_trailing_stop_order(exchange, config, pos)


def _apply_volatility_adaptation(
    exchange: ccxt.bingx,
    state: Dict[str, Any],
    config: Dict[str, Any],
    cache: APICache,
) -> None:
    """
    Volatility-adaptive TP/SL: adjust TP and SL based on real-time ATR changes.

    v1.66.0: volatility_adaptive_tpsl + vol_adapt_param_optimization study.
    If current ATR ratio differs from entry-time ATR ratio, scale TP/SL proportionally.

    vol_factor = clamp(current_ATR_ratio / entry_ATR_ratio, lo, hi)
    - TP: adjusted TP = decayed_TP × vol_factor (complements time decay)
    - SL: adjusted SL = original_SL × vol_factor (no decay, vol-only)

    High vol → wider TP/SL (capture larger moves, avoid premature stops)
    Low vol → tighter TP/SL (grab small moves, faster rotation)

    Research: vol_adapt_param_optimization.py — V2 clamp[0.5,2.0]:
    IS +584.1% (base +558.4%), OOS +448.1% (+19.8%), WF 3/3 PASS.
    """
    vol_cfg = config.get('strategy', {}).get('vol_adaptation', {})
    if not vol_cfg.get('enabled', False):
        return

    positions = state.get('positions') or {}
    if not positions:
        return

    clamp_lo = vol_cfg.get('clamp_lo', 0.5)
    clamp_hi = vol_cfg.get('clamp_hi', 2.0)
    update_interval = vol_cfg.get('update_interval_bars', 1)

    now = datetime.now()
    candle_seconds = CANDLE_DURATION_MS / 1000

    # Get current price and recent OHLCV for ATR calculation
    try:
        ticker = fetch_ticker_cached(exchange, config['symbol'], cache)
        current_price = ticker['last']
    except Exception:
        return

    if not current_price or current_price <= 0:
        return

    state_changed = False

    for slot_id, pos in list(positions.items()):
        if pos.get('_pending_close'):
            continue

        entry_time_str = pos.get('entry_time', '')
        if not entry_time_str:
            continue

        try:
            entry_dt = datetime.fromisoformat(entry_time_str)
        except (ValueError, TypeError):
            continue

        held_seconds = (now - entry_dt).total_seconds()
        bars_held = int(held_seconds / candle_seconds)

        # Respect update interval
        last_vol_bar = pos.get('_last_vol_adapt_bar', 0)
        if bars_held - last_vol_bar < update_interval:
            continue

        # Skip very fresh positions
        if bars_held < 2:
            continue

        # Get entry-time ATR ratio (stored at entry or computed)
        entry_atr_ratio = pos.get('_entry_atr_ratio')
        if entry_atr_ratio is None or entry_atr_ratio <= 0:
            # First time — estimate from position's vol_mult if available
            entry_atr_ratio = pos.get('vol_mult', 1.0)
            if entry_atr_ratio <= 0:
                entry_atr_ratio = 1.0
            pos['_entry_atr_ratio'] = entry_atr_ratio

        # Estimate current ATR ratio from recent price action
        # Use a simple proxy: recent range / entry range
        # (In production, we use the ticker's recent high-low as ATR proxy)
        # For now, we compute vol_factor from position's unrealized movement
        entry_price = pos.get('entry_price', 0)
        direction = pos.get('direction', '')
        if entry_price <= 0:
            continue

        # Use price movement magnitude as volatility proxy
        price_move_pct = abs(current_price / entry_price - 1) * 100
        expected_move_pct = pos.get('sl_pct', 3.0) * entry_atr_ratio  # expected range at entry vol
        if expected_move_pct <= 0:
            continue

        # vol_factor: how much current movement exceeds expectations
        # If price moved more than expected → high vol → widen
        # If price barely moved → low vol → tighten
        # Normalize by time (longer hold = more expected movement)
        time_factor = max(1, bars_held) / 48  # normalize to ~4h expected
        adjusted_expected = expected_move_pct * min(time_factor, 2.0)

        if adjusted_expected > 0:
            raw_vol_factor = max(0.5, price_move_pct / adjusted_expected)
        else:
            raw_vol_factor = 1.0

        vol_factor = max(clamp_lo, min(clamp_hi, raw_vol_factor))

        # Don't update if vol_factor is near 1.0 (no meaningful change)
        if abs(vol_factor - 1.0) < 0.05:
            pos['_last_vol_adapt_bar'] = bars_held
            continue

        # Store original SL on first adaptation
        if '_sl_price_original' not in pos:
            pos['_sl_price_original'] = pos.get('sl_price', 0)

        original_sl = pos.get('_sl_price_original', pos.get('sl_price', 0))
        old_sl = pos.get('sl_price', 0)

        if original_sl <= 0 or entry_price <= 0:
            continue

        # Calculate new SL with vol_factor
        original_sl_dist = abs(entry_price - original_sl)

        # Don't override cascade-tightened SL (cascade takes priority)
        if pos.get('cascade_tightened') or pos.get('_preemptive_cascaded'):
            pos['_last_vol_adapt_bar'] = bars_held
            continue

        new_sl_dist = original_sl_dist * vol_factor

        if direction == 'LONG':
            new_sl = round(entry_price - new_sl_dist, 1)
        else:
            new_sl = round(entry_price + new_sl_dist, 1)

        # Only update if change is meaningful (> $5)
        if abs(new_sl - old_sl) < 5.0:
            pos['_last_vol_adapt_bar'] = bars_held
            continue

        logger.info(
            f"📈 VOL_ADAPT: Slot {slot_id} ({pos.get('pattern_name', '?')} {direction}) "
            f"bar {bars_held} vol_factor={vol_factor:.2f} — "
            f"SL ${old_sl:.1f} → ${new_sl:.1f}"
        )

        if state_changed:
            time.sleep(0.3)  # Rate limit protection between vol_adapt updates
        success = update_single_sl(exchange, pos, config, new_sl)
        pos['_last_vol_adapt_bar'] = bars_held

        if success:
            state_changed = True

    if state_changed:
        save_state(state)


def _check_opposite_signal_exit(
    exchange: ccxt.bingx,
    state: Dict[str, Any],
    config: Dict[str, Any],
    cache: APICache,
    circuit_breaker,
    metrics,
    current_signal: str,
) -> bool:
    """
    Close profitable positions when consecutive opposite signals appear.

    v1.66.0: opposite_signal_exit_study — OPP2_prof0.1 variant.
    If N consecutive opposite-direction signals appear and a position has
    unrealized profit > threshold, close it to lock in profit and free the slot.

    Logic:
    - Track consecutive opposite signal count per direction in state
    - When count >= threshold AND position is profitable → market close
    - Only close ONE position per cycle (oldest profitable)

    Research: opposite_signal_exit_study.py — OPP2_prof0.1:
    OOS +468.6% (baseline +453.5%, +15.1%), trades +26%, R:R 2.575.
    """
    opp_cfg = config.get('strategy', {}).get('opposite_signal_exit', {})
    if not opp_cfg.get('enabled', False):
        return False

    if not current_signal:
        return False

    positions = state.get('positions') or {}
    if not positions:
        return False

    consec_threshold = opp_cfg.get('consecutive_signals', 2)
    profit_threshold = opp_cfg.get('min_profit_pct', 0.1)
    # v1.68.0 fix: Time-window based counter (not raw consecutive, not unlimited accumulation)
    # Record signal bars and count only signals within last N bars
    opp_window_bars = opp_cfg.get('window_bars', 6)  # 6 bars = 30min default

    # Track signal history as list of (bar_number, direction)
    import time
    current_bar = int(time.time() * 1000 / 300000)  # approximate bar index from epoch

    if '_opp_signal_history' not in state:
        state['_opp_signal_history'] = []

    state['_opp_signal_history'].append((current_bar, current_signal))

    # Prune old entries outside window
    cutoff = current_bar - opp_window_bars
    state['_opp_signal_history'] = [
        (bar, sig) for bar, sig in state['_opp_signal_history']
        if bar >= cutoff
    ]

    # Count recent signals by direction within window
    recent_short = sum(1 for _, sig in state['_opp_signal_history'] if sig == 'SHORT')
    recent_long = sum(1 for _, sig in state['_opp_signal_history'] if sig == 'LONG')

    state['_opp_consec_vs_long'] = recent_short   # SHORT signals oppose LONG positions
    state['_opp_consec_vs_short'] = recent_long    # LONG signals oppose SHORT positions

    # Get current price
    try:
        ticker = fetch_ticker_cached(exchange, config['symbol'], cache)
        current_price = ticker['last']
    except Exception:
        return False

    if not current_price or current_price <= 0:
        return False

    leverage = config.get('leverage', config.get('strategy', {}).get('leverage', 3))

    # Check each direction
    for direction in ['LONG', 'SHORT']:
        counter_key = f'_opp_consec_vs_{direction.lower()}'
        consec_count = state.get(counter_key, 0)

        if consec_count < consec_threshold:
            continue

        # Find oldest profitable position in this direction
        best_slot = None
        best_profit = -999
        best_entry_time = None

        for sid, pos in positions.items():
            if pos.get('direction') != direction:
                continue
            if pos.get('_pending_close'):
                continue

            entry = pos.get('entry_price', 0)
            if entry <= 0:
                continue

            if direction == 'LONG':
                unr_pct = (current_price / entry - 1) * 100 * leverage
            else:
                unr_pct = (1 - current_price / entry) * 100 * leverage

            if unr_pct < profit_threshold:
                continue

            # Pick the oldest profitable position
            et = pos.get('entry_time', '')
            if best_slot is None or (et and (best_entry_time is None or et < best_entry_time)):
                best_slot = sid
                best_profit = unr_pct
                best_entry_time = et

        if best_slot is None:
            continue

        pos = positions[best_slot]
        logger.info(
            f"🔄 OPP_SIGNAL EXIT: {direction} slot {best_slot} "
            f"({pos.get('pattern_name', '?')}) profit={best_profit:+.2f}% — "
            f"{consec_count} consecutive opposite signals"
        )

        close_position_market(
            exchange, state, config, cache, 'OPP_SIGNAL', metrics,
            position=pos,
        )

        # Reset counter after exit
        state[counter_key] = 0
        save_state(state)
        return True

    return False


def _check_position_timeouts(
    exchange: ccxt.bingx,
    state: Dict[str, Any],
    config: Dict[str, Any],
    cache: APICache,
    circuit_breaker: CircuitBreaker,
    metrics: PerformanceMetrics,
) -> bool:
    """
    Close positions that exceeded the timeout threshold.

    v1.31.0: Positions held longer than timeout_bars × 5min are closed
    at market price. This frees stale slots (48h+ trades are net negative).

    Returns:
        True if any position was closed due to timeout
    """
    timeout_bars = config.get('strategy', {}).get('timeout_bars', DEFAULT_TIMEOUT_BARS)
    if not timeout_bars:
        return False

    positions = state.get('positions') or {}
    if not positions:
        return False

    closed_any = False
    now = datetime.now()
    timeout_seconds = timeout_bars * CANDLE_DURATION_MS // 1000

    for slot_id, pos in list(positions.items()):
        entry_time_str = pos.get('entry_time', '')
        if not entry_time_str:
            continue
        try:
            entry_dt = datetime.fromisoformat(entry_time_str)
            held_seconds = (now - entry_dt).total_seconds()
            if held_seconds >= timeout_seconds:
                held_hours = held_seconds / 3600
                timeout_hours = timeout_seconds / 3600
                pattern = pos.get('pattern_name', '?')
                direction = pos.get('direction', '?')
                logger.info(
                    f"⏰ TIMEOUT: Slot {slot_id} ({pattern} {direction}) "
                    f"held {held_hours:.1f}h > {timeout_hours:.0f}h limit, closing at market"
                )
                close_position_market(
                    exchange, state, config, cache, 'TIMEOUT', metrics,
                    position=pos,
                )
                closed_any = True
        except (ValueError, TypeError):
            continue

    return closed_any
