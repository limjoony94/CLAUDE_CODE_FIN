"""
Pattern 5m Bot - Position Opening
Functions for opening new trading positions.
"""

import time
import uuid
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
from .exchange import fetch_ticker_cached, fetch_positions_cached, fetch_balance_cached
from .indicators import get_volatility_multiplier
from .state import save_state
from .utils import extract_pattern_name
from .orders import place_tp_sl_orders, cancel_remaining_orders, update_emergency_sl, _get_position_side

logger = logging.getLogger('pattern_5m')


def get_position_size(
    exchange: ccxt.bingx,
    config: Dict[str, Any],
    cache: APICache,
    circuit_breaker: Optional[CircuitBreaker] = None,
    metrics: Optional[PerformanceMetrics] = None,
    state: Optional[Dict[str, Any]] = None,
    signal: Optional[str] = None,
    df: Optional[pd.DataFrame] = None,
    equity_curve_scale: float = 1.0,
) -> tuple:
    """
    Calculate position size based on available balance.

    Args:
        exchange: Exchange instance
        config: Bot configuration
        cache: APICache instance
        circuit_breaker: Optional CircuitBreaker
        metrics: Optional PerformanceMetrics
        state: Optional bot state dict (for MDD sizing, v1.34.0)
        signal: Optional 'LONG' or 'SHORT' (for regime sizing, v1.35.3)
        df: Optional DataFrame with close prices (for regime sizing, v1.35.3)
        equity_curve_scale: Size multiplier from equity curve trading (v1.40.0)

    Returns:
        Tuple of (quantity, available_balance, price) or (None, None, None) on error
    """
    try:
        balance = fetch_balance_cached(exchange, cache, force_refresh=True,
                                       circuit_breaker=circuit_breaker, metrics=metrics)
        available = float(balance.get('USDT', {}).get('free', 0))
        total_equity = float(balance.get('USDT', {}).get('total', available))

        size_pct = config['position_size_pct'] / 100
        max_size = config['risk']['max_position_size_usd']
        max_positions = config.get('max_positions', 1)

        # MDD dynamic sizing (v1.34.0)
        mdd_scale = 1.0
        mdd_cfg = config.get('risk', {}).get('mdd_sizing', {})
        if mdd_cfg.get('enabled', False) and state:
            peak = state.get('peak_equity', 0)
            if peak > 0 and total_equity < peak:
                dd_pct = (peak - total_equity) / peak * 100
                full_below = mdd_cfg.get('full_size_below_dd', 5.0)
                min_above = mdd_cfg.get('min_size_above_dd', 20.0)
                min_scale = mdd_cfg.get('min_scale', 0.25)
                if dd_pct <= full_below:
                    mdd_scale = 1.0
                elif dd_pct >= min_above:
                    mdd_scale = min_scale
                else:
                    mdd_scale = 1.0 - (1.0 - min_scale) * (dd_pct - full_below) / (min_above - full_below)
                if mdd_scale < 1.0:
                    logger.info(f"MDD sizing: DD={dd_pct:.1f}%, scale={mdd_scale:.2f}")

        # Regime-aware sizing (v1.35.3)
        regime_scale = 1.0
        regime_cfg = config.get('risk', {}).get('regime_sizing', {})
        if regime_cfg.get('enabled', False) and signal and df is not None and len(df) > 0:
            ema_period = regime_cfg.get('ema_period', 20)
            lookback = regime_cfg.get('lookback', 5)
            counter_mult = regime_cfg.get('counter_mult', 0.3)
            if len(df) >= ema_period + lookback:
                closes = df['close'].values
                # EMA calculation
                ema = pd.Series(closes).ewm(span=ema_period, adjust=False).mean().values
                # Slope: compare current EMA vs lookback bars ago
                slope = ema[-1] - ema[-1 - lookback]
                is_uptrend = slope > 0
                is_counter = (is_uptrend and signal == 'SHORT') or (not is_uptrend and signal == 'LONG')
                if is_counter:
                    regime_scale = counter_mult
                    logger.info(f"Regime sizing: {'UP' if is_uptrend else 'DOWN'} trend, {signal} is counter-regime, scale={regime_scale:.2f}")

        # 1/N sizing: each slot gets equity/max_positions
        per_slot_equity = total_equity * size_pct / max_positions * mdd_scale * regime_scale * equity_curve_scale
        position_value = min(per_slot_equity, available * size_pct, max_size)

        ticker = fetch_ticker_cached(exchange, config['symbol'], cache, force_refresh=True,
                                     circuit_breaker=circuit_breaker, metrics=metrics)
        price = ticker['last']

        effective_leverage = config['leverage']
        quantity = (position_value * effective_leverage) / price
        quantity = round(quantity, QUANTITY_ROUND_DECIMALS)

        return quantity, available, price
    except ccxt.NetworkError as e:
        logger.error(f"Failed to calculate position size (network error): {e}")
        return None, None, None
    except ccxt.ExchangeError as e:
        logger.error(f"Failed to calculate position size (exchange error): {e}")
        return None, None, None
    except Exception as e:
        logger.exception(f"Failed to calculate position size: {e}")
        return None, None, None


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
    leverage_override: Optional[float] = None,
    equity_curve_scale: float = 1.0,
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
        leverage_override: Optional adaptive leverage (v1.39.0)
        equity_curve_scale: Size multiplier from equity curve trading (v1.40.0)

    Returns:
        True if position opened successfully
    """
    symbol = config['symbol']
    leverage = leverage_override or config['leverage']
    exchange_leverage = config.get('exchange_leverage', leverage)
    strategy = config['strategy']

    try:
        # Check slot availability (multi-position)
        max_positions = config.get('max_positions', 1)
        active_positions = state.get('positions') or {}
        if len(active_positions) >= max_positions:
            logger.info(f"All {max_positions} slots occupied ({len(active_positions)} active), skipping signal")
            return False

        # Set leverage
        _set_leverage(exchange, symbol, exchange_leverage, config)

        # Calculate position size (also returns current price to avoid double ticker fetch)
        quantity, available, estimated_price = get_position_size(
            exchange, config, cache, circuit_breaker, metrics, state=state, signal=signal, df=df,
            equity_curve_scale=equity_curve_scale,
        )
        if quantity is None or quantity <= 0:
            logger.warning(f"Invalid position size (qty={quantity}, balance=${available}), skipping")
            return False

        # Log balance and position sizing details
        effective_leverage = leverage  # leverage already resolved from leverage_override or config
        position_value = quantity * estimated_price / effective_leverage
        size_pct = config['position_size_pct']
        logger.info(f"Balance: ${available:.2f} | Size: ${position_value:.2f} ({size_pct}%) | Notional: ${quantity * estimated_price:.2f} ({effective_leverage:.1f}x)")

        # Execute market order
        t_open = time.time()
        side = 'buy' if signal == 'LONG' else 'sell'
        logger.info(f"Opening {signal} position: {quantity} {symbol} @ ~${estimated_price:.1f}")

        position_side = _get_position_side(config, signal)
        order = exchange.create_market_order(
            symbol=symbol,
            side=side,
            amount=quantity,
            params={'positionSide': position_side}
        )

        # Get actual fill price
        actual_entry_price, actual_quantity = _get_actual_fill_price(
            exchange, order, signal, symbol, estimated_price, quantity, cache, circuit_breaker, metrics
        )

        # Invalidate cache after order
        cache.invalidate_all()

        open_latency_ms = (time.time() - t_open) * 1000
        logger.info(f"Actual fill: ${actual_entry_price:.1f} (qty: {actual_quantity}) [{open_latency_ms:.0f}ms]")

        # Calculate TP/SL
        direction = 1 if signal == 'LONG' else -1
        vol_mult = 1.0
        if df is not None:
            vol_mult = get_volatility_multiplier(df, config)
            if vol_mult != 1.0:
                logger.info(f"Vol-Adaptive: ATR multiplier = {vol_mult:.2f}x")

        # v1.6: Extract pattern from reason for pattern-specific TP/SL
        pattern = extract_pattern_name(reason) or None

        # v1.18: Get regime-specific TP/SL from state (set by check_entry_signal)
        regime_tp_sl = state.get('regime_tp_sl')
        current_regime = state.get('current_regime', 'UNKNOWN')

        tp_price, sl_price, tp_pct_adjusted, sl_pct_adjusted = calculate_tp_sl(
            actual_entry_price, direction, strategy, vol_mult, pattern, regime_tp_sl, config
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

        # Create new slot and append to positions list
        slot_id = uuid.uuid4().hex[:8]
        new_slot = {
            'slot_id': slot_id,
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
            'pattern_name': pattern,
            # Rotation (순환매) fields
            'rotation_enabled': ROTATION_ENABLED and bool(scale_out_stages),
            'is_partial': False,
            'avg_entry_price': actual_entry_price,
            'total_entries': 1,
            'refill_entries': [],
            # v1.18: Regime-adaptive fields
            'market_regime': current_regime,
            'regime_tp_sl': regime_tp_sl,
            # v1.39.0: Per-slot effective leverage for adaptive leverage
            'effective_leverage': effective_leverage,
            # v1.40.0: Equity curve scale factor
            'equity_curve_scale': equity_curve_scale,
        }
        state['positions'][slot_id] = new_slot
        # One-Way mode: track single direction constraint; Hedge: no constraint
        if config.get('position_mode') != 'hedge':
            state['active_direction'] = signal
        state['last_signal_time'] = datetime.now().isoformat()

        save_state(state)
        logger.info(f"Position opened (slot {slot_id}): {order.get('id')} [{len(state['positions'])}/{max_positions} slots]")

        # Place TP/SL orders for this slot
        place_tp_sl_orders(exchange, state, config, position=new_slot)

        # CRITICAL: Verify SL was placed — position is unprotected without it
        if not new_slot.get('sl_order_id'):
            logger.warning(f"SL order not placed for slot {slot_id} — retrying...")
            for retry in range(2):
                place_tp_sl_orders(exchange, state, config, position=new_slot)
                if new_slot.get('sl_order_id'):
                    logger.info(f"SL order placed on retry {retry + 1}")
                    break
            if not new_slot.get('sl_order_id'):
                logger.error(f"CRITICAL: SL order failed for slot {slot_id} — position UNPROTECTED until next verify cycle")

        # v1.29.0: Update emergency SL to cover all active slots
        update_emergency_sl(exchange, state, config)

        return True

    except ccxt.NetworkError as e:
        logger.error(f"Failed to open position (network error): {e}")
        return False
    except ccxt.InsufficientFunds as e:
        logger.error(f"Failed to open position (insufficient funds): {e}")
        return False
    except ccxt.ExchangeError as e:
        error_msg = str(e)
        # Auto-recover from position mode mismatch (use config setting)
        if 'Hedge mode' in error_msg or '109400' in error_msg or 'position mode' in error_msg.lower():
            is_hedge = config.get('position_mode') == 'hedge'
            target_label = 'Hedge' if is_hedge else 'One-Way'
            logger.warning(f"Position mode mismatch, attempting to switch to {target_label}...")
            try:
                exchange.set_position_mode(hedged=is_hedge, symbol=config['symbol'])
                logger.info(f"Switched to {target_label} mode, please retry the signal")
            except Exception as mode_err:
                logger.error(f"Failed to switch position mode: {mode_err}")
        else:
            logger.error(f"Failed to open position (exchange error): {e}")
        return False
    except Exception as e:
        logger.exception(f"Failed to open position: {e}")
        return False


def _check_slot_available(state: Dict[str, Any], config: Dict[str, Any]) -> bool:
    """Check if a slot is available for a new position."""
    max_positions = config.get('max_positions', 1)
    active = len(state.get('positions') or {})
    return active < max_positions


def _set_leverage(exchange: ccxt.bingx, symbol: str, leverage: int, config: Optional[Dict[str, Any]] = None) -> None:
    """Set leverage for the symbol.

    One-Way mode: BOTH side only.
    Hedge mode: LONG and SHORT sides separately.
    """
    is_hedge = (config or {}).get('position_mode') == 'hedge'
    sides = ['LONG', 'SHORT'] if is_hedge else ['BOTH']
    for side in sides:
        try:
            exchange.set_leverage(leverage, symbol, params={'side': side})
        except ccxt.ExchangeError as e:
            if 'No need to change' in str(e) or 'same' in str(e).lower():
                logger.debug(f"Leverage already set to {leverage}x (side={side})")
            else:
                logger.warning(f"Set leverage warning (side={side}, exchange error): {e}")
        except ccxt.NetworkError as e:
            logger.warning(f"Set leverage warning (side={side}, network error): {e}")
        except (ValueError, TypeError) as e:
            logger.warning(f"Set leverage warning (side={side}, invalid value): {e}")


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
                    exchange_total_qty = float(pos.get('contracts', actual_quantity))
                    actual_quantity = min(exchange_total_qty, quantity)  # Cap to order qty
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


def _effective_vol_mult(vol_mult: float, base_sl_pct: float, config: Optional[Dict[str, Any]]) -> float:
    """Compute effective vol_mult so scaled SL never exceeds daily loss limit.

    Caps the multiplier itself (not the SL result) to preserve TP/SL ratio.
    max_mult = (max_daily_loss_pct / leverage) / base_sl_pct
    effective = min(vol_mult, max_mult)

    This ensures TP and SL always scale proportionally — no R:R distortion.
    """
    if not config or base_sl_pct <= 0:
        return vol_mult
    leverage = config.get('leverage', 3)
    max_daily_loss = config.get('risk', {}).get('max_daily_loss_pct', 13)
    max_sl_pct = max_daily_loss / leverage
    max_mult = max_sl_pct / base_sl_pct
    if vol_mult > max_mult:
        logger.info(f"Vol mult capped: {vol_mult:.3f}x → {max_mult:.3f}x (SL {base_sl_pct:.2f}% × {max_mult:.3f} = {base_sl_pct * max_mult:.2f}%, limit {max_sl_pct:.2f}%)")
        return max_mult
    return vol_mult


def calculate_tp_sl(
    entry_price: float,
    direction: int,
    strategy: Dict[str, Any],
    vol_mult: float,
    pattern: Optional[str] = None,
    regime_tp_sl: Optional[tuple] = None,
    config: Optional[Dict[str, Any]] = None,
) -> tuple:
    """Calculate TP and SL prices — single source of truth.

    Used by: open_position, refill_position, recover_position_to_state,
             recalculate_position_orders.

    Priority: dynamic_universal > regime_tp_sl > PATTERN_OPTIMAL_TPSL > strategy defaults
    """
    # Dynamic Per-Pattern TP/SL mode (highest priority)
    if config and config.get('_dynamic_tpsl_per_pattern'):
        pp_tpsl = config.get('_dynamic_patterns_tpsl', {})
        if pattern and pattern in pp_tpsl:
            base_tp_pct = pp_tpsl[pattern][0]
            base_sl_pct = pp_tpsl[pattern][1]
            logger.debug(f"Using dynamic per-pattern TP/SL: {pattern} → TP={base_tp_pct}%, SL={base_sl_pct}%")
        else:
            base_tp_pct = strategy['tp_pct']
            base_sl_pct = strategy['sl_pct']
            logger.warning(f"Pattern {pattern} not in dynamic per-pattern dict, using defaults")

        eff_mult = _effective_vol_mult(vol_mult, base_sl_pct, config)
        tp_pct_adjusted = (base_tp_pct * eff_mult) + SLIPPAGE_BUFFER_PCT
        sl_pct_adjusted = max(0.1, (base_sl_pct * eff_mult) - SLIPPAGE_BUFFER_PCT)
        tp_price = round(entry_price * (1 + direction * tp_pct_adjusted / 100), PRICE_ROUND_DECIMALS)
        sl_price = round(entry_price * (1 - direction * sl_pct_adjusted / 100), PRICE_ROUND_DECIMALS)
        return tp_price, sl_price, tp_pct_adjusted, sl_pct_adjusted

    # Dynamic Universal TP/SL mode
    if config and config.get('_dynamic_tpsl_universal'):
        base_tp_pct = config['_dynamic_tp']
        base_sl_pct = config['_dynamic_sl']
        logger.debug(f"Using dynamic universal TP/SL: TP={base_tp_pct}%, SL={base_sl_pct}%")

        eff_mult = _effective_vol_mult(vol_mult, base_sl_pct, config)
        tp_pct_adjusted = (base_tp_pct * eff_mult) + SLIPPAGE_BUFFER_PCT
        sl_pct_adjusted = max(0.1, (base_sl_pct * eff_mult) - SLIPPAGE_BUFFER_PCT)
        tp_price = round(entry_price * (1 + direction * tp_pct_adjusted / 100), PRICE_ROUND_DECIMALS)
        sl_price = round(entry_price * (1 - direction * sl_pct_adjusted / 100), PRICE_ROUND_DECIMALS)
        return tp_price, sl_price, tp_pct_adjusted, sl_pct_adjusted

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

    eff_mult = _effective_vol_mult(vol_mult, base_sl_pct, config)
    tp_pct_adjusted = (base_tp_pct * eff_mult) + SLIPPAGE_BUFFER_PCT  # TP: add slippage (target further)
    sl_pct_adjusted = (base_sl_pct * eff_mult) - SLIPPAGE_BUFFER_PCT  # SL: subtract slippage (tighter)
    sl_pct_adjusted = max(0.1, sl_pct_adjusted)  # Floor guard: prevent negative/tiny SL

    tp_price = round(entry_price * (1 + direction * tp_pct_adjusted / 100), PRICE_ROUND_DECIMALS)
    sl_price = round(entry_price * (1 - direction * sl_pct_adjusted / 100), PRICE_ROUND_DECIMALS)

    logger.debug(f"TP/SL calculated: entry=${entry_price:.1f} dir={direction} | TP=${tp_price:.1f} ({tp_pct_adjusted:.2f}%) SL=${sl_price:.1f} ({sl_pct_adjusted:.2f}%)")

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

    # Find the slot to refill (by slot_id if provided in reason, else first partial)
    positions = state.get('positions') or {}
    position = None
    for pos in positions.values():
        if pos.get('is_partial', False):
            position = pos
            break

    if not position:
        logger.warning("No partial position to refill")
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

        position_side = _get_position_side(config, position['direction'])
        order = exchange.create_market_order(
            symbol=symbol,
            side=side,
            amount=refill_qty,
            params={'positionSide': position_side}
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
        if total_qty <= 0:
            logger.error(f"Invalid total_qty={total_qty} after refill — aborting")
            return False
        new_avg_entry = (old_cost + new_cost) / total_qty

        logger.info(f"New average entry: ${new_avg_entry:.1f} (was ${old_entry:.1f})")

        # Recalculate TP/SL based on new average entry
        direction = 1 if position['direction'] == 'LONG' else -1
        vol_mult = position.get('vol_mult', 1.0)

        # Extract pattern from position reason for per-pattern TP/SL
        pattern = extract_pattern_name(position.get('reason', '')) or None
        regime_tp_sl = position.get('regime_tp_sl')

        tp_price, sl_price, tp_pct_adjusted, sl_pct_adjusted = calculate_tp_sl(
            new_avg_entry, direction, strategy, vol_mult, pattern, regime_tp_sl, config
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

        # Cancel old TP/SL and place new orders for this slot
        cancel_remaining_orders(exchange, state, config, position=position)
        place_tp_sl_orders(exchange, state, config, position=position)

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
        logger.exception(f"Failed to refill position: {e}")
        return False
