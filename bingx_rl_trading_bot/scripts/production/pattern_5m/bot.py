"""
Pattern 5m Bot - Main Bot Logic
The main trading loop and bot orchestration.

v1.25.1: Candle-aligned smart sleep replaces fixed-interval polling.
- Trading Window (first 30s after candle close): signal/exit check
- Maintenance Window (rest of candle): monitoring, sync, health, metrics
- Signal latency: ~5-8s after candle close (was 17-50s)
"""

import os
import sys
import json
import math
import time
import signal
import logging
import numpy as np
import pandas as pd
from collections import namedtuple
from datetime import datetime
from typing import Dict, Any, Optional

import ccxt

from .constants import (
    BOT_NAME,
    BOT_VERSION,
    PROJECT_ROOT,
    CONFIG_FILE,
    STATE_FILE,
    METRICS_FILE,
    DYNAMIC_PATTERNS_FILE,
    MAX_OHLCV_CANDLES,
    DEFAULT_SLEEP_INTERVAL,
    DAILY_LOSS_PAUSE_SECONDS,
    CANDLE_SETTLE_SECONDS,
    CANDLE_DURATION_MS,
    POSITION_SYNC_INTERVAL_MINUTES,
    TRADING_WINDOW_SECONDS,
    POSITION_MONITOR_INTERVAL,
    MAX_MAINTENANCE_SLEEP,
    TP_SL_VERIFY_INTERVAL_SECONDS,
    LOG_STATUS_INTERVAL_SECONDS,
    METRICS_SAVE_INTERVAL_SECONDS,
    VALIDATED_LONG_PATTERNS,
    VALIDATED_SHORT_PATTERNS,
    DEFAULT_TIMEOUT_BARS,
    SLIPPAGE_BUFFER_PCT,
)
from .models import APICache, CircuitBreaker, PerformanceMetrics
from .config import load_config, validate_config, load_dynamic_patterns
from .state import (
    load_state,
    save_state,
    load_metrics,
    save_metrics,
    sync_metrics_with_state,
    update_peak_equity,
)
from .exchange import (
    create_exchange,
    verify_position_mode,
    set_margin_mode,
    fetch_ohlcv,
    fetch_ticker_cached,
    fetch_balance_cached,
    health_check,
    set_shutdown_checker,
)
from .indicators import calculate_indicators, get_volatility_multiplier
from .signals import check_entry_signal, check_cooldown, check_daily_loss_limit, check_early_exit_signal
from .position import (
    open_position,
    check_position_status,
    sync_position_with_exchange,
    recover_from_crash,
    close_position_market,
)
from .orders import (
    verify_tp_sl_orders, adjust_tpsl_to_config,
    _get_emergency_sl_id, _place_emergency_sl_for_direction,
    _EXCHANGE_MANAGED,
    update_single_tp,
    update_single_sl,
)
from .position_management import (
    _apply_time_decay_tp,
    _apply_preemptive_cascade,
    _apply_volatility_adaptation,
    _apply_trailing_stop,
    _check_opposite_signal_exit,
    _check_position_timeouts,
)
from .utils import extract_pattern_name
from .utils.lock import acquire_lock, release_lock
from .utils.logging_config import setup_logging

logger = logging.getLogger('pattern_5m')

# Global shutdown flag
shutdown_requested = False

# Candle timing helper
CandleTiming = namedtuple('CandleTiming', [
    'seconds_into_candle', 'seconds_until_close', 'candle_id', 'in_trading_window'
])


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    global shutdown_requested
    logger.info(f"\n🛑 Received signal {signum}, initiating graceful shutdown...")
    shutdown_requested = True


def _get_candle_timing():
    """
    Calculate current position within the 5-minute candle cycle.

    Returns:
        CandleTiming namedtuple with seconds_into_candle, seconds_until_close,
        candle_id (unique per candle), and in_trading_window flag.
    """
    now_ms = int(time.time() * 1000)
    candle_progress_ms = now_ms % CANDLE_DURATION_MS
    seconds_into_candle = candle_progress_ms / 1000
    seconds_until_close = (CANDLE_DURATION_MS - candle_progress_ms) / 1000
    candle_id = now_ms // CANDLE_DURATION_MS
    in_trading_window = seconds_into_candle < TRADING_WINDOW_SECONDS

    return CandleTiming(
        seconds_into_candle=seconds_into_candle,
        seconds_until_close=seconds_until_close,
        candle_id=candle_id,
        in_trading_window=in_trading_window,
    )


def _interruptible_sleep(duration):
    """
    Sleep that checks shutdown_requested every 1s for fast shutdown.

    Args:
        duration: Total sleep duration in seconds
    """
    global shutdown_requested
    end_time = time.time() + duration
    while time.time() < end_time and not shutdown_requested:
        remaining = end_time - time.time()
        time.sleep(min(1.0, max(0.0, remaining)))


def _calculate_sleep_duration(has_position, last_processed_candle_id):
    """
    Calculate smart sleep duration based on candle timing and position state.

    Args:
        has_position: Whether bot currently has an open position
        last_processed_candle_id: ID of the last candle that was processed

    Returns:
        Sleep duration in seconds
    """
    timing = _get_candle_timing()

    # In trading window with unprocessed candle → wake immediately
    if timing.in_trading_window and timing.candle_id != last_processed_candle_id:
        return 0

    # Time until next candle's trading window
    next_trading_start = timing.seconds_until_close + CANDLE_SETTLE_SECONDS

    if has_position:
        return min(POSITION_MONITOR_INTERVAL, next_trading_start)
    else:
        return min(MAX_MAINTENANCE_SLEEP, next_trading_start)


def run_bot(config_file: str = CONFIG_FILE) -> None:
    """
    Main entry point for the Pattern 5m trading bot.

    Args:
        config_file: Path to configuration YAML file (absolute path from constants.py)
    """
    # v1.25.5: Use absolute paths from constants.py (CWD-independent)
    # Change to project root for any relative path operations in dependencies
    os.chdir(PROJECT_ROOT)

    # All paths are now absolute from constants.py
    config_path = config_file
    state_path = STATE_FILE
    metrics_path = METRICS_FILE

    # Load configuration
    config = load_config(config_path)
    config = load_dynamic_patterns(config)
    validate_config(config)

    # Setup logging
    setup_logging(
        debug_mode=config.get('debug_mode', False),
        json_format=config.get('json_logging', False)
    )

    logger.info(f"{'='*60}")
    logger.info(f"🚀 {BOT_NAME} v{BOT_VERSION} starting...")
    logger.info(f"{'='*60}")
    logger.info(f"Symbol: {config['symbol']} | TF: {config['timeframe']} | Lev: {config['leverage']}x")

    # Log active pattern summary for quick reference
    _log_pattern_summary(config)

    # Acquire lock
    lock_acquired = acquire_lock()
    if not lock_acquired:
        logger.error("❌ Failed to acquire lock - another instance may be running")
        sys.exit(1)

    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    # Register shutdown checker so API retry/circuit breaker sleeps are interruptible
    set_shutdown_checker(lambda: shutdown_requested)

    try:
        _run_bot_main(config, state_path, metrics_path)
    except KeyboardInterrupt:
        logger.info("🛑 Keyboard interrupt received")
    except Exception as e:
        logger.exception(f"❌ Fatal error: {e}")
        raise
    finally:
        release_lock()
        logger.info("🔒 Lock released, bot stopped")


def _check_scan_staleness(config: Dict[str, Any]) -> None:
    """Check dynamic_patterns.json age, warn if stale (v1.34.0)."""
    try:
        with open(DYNAMIC_PATTERNS_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)
        generated_at_str = data.get('generated_at', '')
        if not generated_at_str:
            logger.warning("dynamic_patterns.json missing 'generated_at' field")
            return
        generated_at = datetime.fromisoformat(generated_at_str)
        age_days = (datetime.now() - generated_at).days
        interval = config.get('strategy', {}).get('rescan_interval_days', 90)
        if age_days > interval:
            logger.warning(
                f"Pattern scan is {age_days} days old (threshold: {interval}d). "
                f"Consider re-scanning: python scripts/scanner/pattern_scanner.py"
            )
        else:
            logger.info(f"Pattern scan age: {age_days}d (threshold: {interval}d)")
    except Exception as e:
        logger.warning(f"Could not check scan staleness: {e}")


def _run_bot_main(
    config: Dict[str, Any],
    state_path: str,
    metrics_path: str,
) -> None:
    """
    Main bot loop with candle-aligned timing.

    Two-phase loop per candle:
    - Trading Window (first 30s after candle close): signal/exit check
    - Maintenance Window (rest of candle): monitoring, sync, health, metrics

    Signal latency: ~5-8s after candle close (was 17-50s with fixed polling).
    """
    global shutdown_requested

    # Initialize components
    exchange = create_exchange()
    cache = APICache()
    circuit_breaker = CircuitBreaker()

    # Load or create metrics
    metrics = load_metrics(metrics_path)
    if metrics is None:
        metrics = PerformanceMetrics(session_start=datetime.now().isoformat())
        logger.info("📊 Created new performance metrics")

    # Load state
    state = load_state(state_path)

    # Sync metrics with state
    metrics = sync_metrics_with_state(metrics, state)

    # Verify exchange settings
    _verify_exchange_settings(exchange, config)

    # Check scan staleness (v1.34.0)
    _check_scan_staleness(config)

    # Crash recovery
    recover_from_crash(exchange, state, config, cache, circuit_breaker, metrics)

    # Check if existing positions' TP/SL match current config
    if state.get('positions'):
        adjust_tpsl_to_config(exchange, state, config)

    # Candle-aligned loop state
    last_processed_candle_id = -1
    last_sync_time = 0
    last_health_check_time = 0
    last_tp_sl_verify_time = 0.0
    last_log_status_time = 0.0
    last_metrics_save_time = 0.0

    logger.info("✅ Bot initialized (candle-aligned loop), starting main loop...")

    while not shutdown_requested:
        try:
            # Check daily loss limit
            if check_daily_loss_limit(state, config):
                logger.warning(f"⚠️ Daily loss limit reached, pausing {DAILY_LOSS_PAUSE_SECONDS}s")
                _interruptible_sleep(DAILY_LOSS_PAUSE_SECONDS)
                continue

            timing = _get_candle_timing()
            has_position = bool(state.get('positions') or {})
            now = time.time()

            # === TRADING WINDOW: First 30s after candle close ===
            if timing.in_trading_window and timing.candle_id != last_processed_candle_id:
                _wait_for_candle_settle(config)

                # 1. Check all active slots for TP/SL hit
                if has_position:
                    position_closed = check_position_status(
                        exchange, state, config, cache, circuit_breaker, metrics
                    )
                    if not position_closed:
                        # Check early exit for all active slots
                        _process_existing_positions(
                            exchange, state, config, cache, circuit_breaker, metrics
                        )

                # 1b. Check position timeouts (v1.31.0: close stale positions)
                if state.get('positions'):
                    _check_position_timeouts(
                        exchange, state, config, cache, circuit_breaker, metrics
                    )

                # 1b2. Apply time-decay TP (v1.62.0: reduce TP after 12h)
                if state.get('positions'):
                    _apply_time_decay_tp(exchange, state, config)

                # 1b3. Pre-emptive cascade SL (v1.65.0: tighten SLs before cluster hit)
                if state.get('positions'):
                    _apply_preemptive_cascade(exchange, state, config, cache)

                # 1b4. Trailing stop (v1.68.0: lock profit after activation)
                if state.get('positions'):
                    _apply_trailing_stop(exchange, state, config, cache)

                # 1b5. Volatility-adaptive SL (v1.66.0: adjust SL based on real-time ATR)
                if state.get('positions'):
                    _apply_volatility_adaptation(exchange, state, config, cache)

                # 1c. Proactive emergency SL health check (v1.36.6)
                _ensure_emergency_sl_exists(exchange, state, config)

                # 1e. Record loss for burst brake (v1.37.0)
                _record_loss_for_burst_brake(state, config)

                # 1f. Update rolling WR tracker for adaptive leverage (v1.39.0)
                _record_trade_for_rolling_wr(state, config)

                # 1g. Update equity curve tracker (v1.40.0)
                _record_trade_for_equity_curve(state, config)

                # 2. Check for new entry signal (always, if slots available)
                _process_entry_signal(
                    exchange, state, config, cache, circuit_breaker, metrics
                )

                # 2b. Opposite signal early exit (v1.66.0)
                # Uses last_signal stored by _process_entry_signal
                last_sig = state.get('_last_detected_signal')
                if state.get('positions') and last_sig:
                    _check_opposite_signal_exit(
                        exchange, state, config, cache, circuit_breaker, metrics,
                        last_sig,
                    )

                last_processed_candle_id = timing.candle_id

            else:
                # === MAINTENANCE WINDOW ===
                if has_position:
                    position_closed = check_position_status(
                        exchange, state, config, cache, circuit_breaker, metrics
                    )
                    if position_closed:
                        has_position = bool(state.get('positions') or {})
                        _record_loss_for_burst_brake(state, config)  # v1.37.0
                        _record_trade_for_rolling_wr(state, config)  # v1.39.0
                        _record_trade_for_equity_curve(state, config)  # v1.40.0

                # Proactive emergency SL health check (v1.36.6)
                _ensure_emergency_sl_exists(exchange, state, config)

                # Position sync (clock-aligned, every 5 min)
                last_sync_time = _maybe_sync_position(
                    exchange, state, config, cache, circuit_breaker, metrics, last_sync_time
                )

                # Health check (clock-aligned, every 30 min)
                last_health_check_time = _maybe_run_health_check(
                    exchange, config, cache, circuit_breaker, metrics, state_path,
                    last_health_check_time
                )

                # Time-based maintenance tasks
                has_position = bool(state.get('positions') or {})
                if has_position:
                    if now - last_tp_sl_verify_time >= TP_SL_VERIFY_INTERVAL_SECONDS:
                        verify_tp_sl_orders(exchange, state, config)
                        last_tp_sl_verify_time = now

                    if now - last_log_status_time >= LOG_STATUS_INTERVAL_SECONDS:
                        for slot in (state.get('positions') or {}).values():
                            if not slot.get('_pending_close'):
                                _log_position_status(slot, config, cache, exchange)
                        last_log_status_time = now
                else:
                    if now - last_log_status_time >= LOG_STATUS_INTERVAL_SECONDS:
                        _log_waiting_status(state, metrics, cache, exchange, config)
                        last_log_status_time = now

                if now - last_metrics_save_time >= METRICS_SAVE_INTERVAL_SECONDS:
                    save_metrics(metrics, metrics_path)
                    # Update peak equity for MDD sizing (v1.34.0)
                    try:
                        bal = fetch_balance_cached(exchange, cache, force_refresh=True,
                                                   circuit_breaker=circuit_breaker, metrics=metrics)
                        equity = float(bal.get('USDT', {}).get('total', 0))
                        if equity > 0:
                            update_peak_equity(state, equity)
                    except Exception as e:
                        logger.debug(f"Peak equity update skipped: {e}")  # non-critical
                    last_metrics_save_time = now

            # Smart sleep — refresh in case position changed during this iteration
            has_position = bool(state.get('positions') or {})
            sleep_duration = _calculate_sleep_duration(has_position, last_processed_candle_id)
            _interruptible_sleep(sleep_duration)

        except ccxt.NetworkError as e:
            logger.warning(f"⚠️ Network error: {e}")
            circuit_breaker.record_failure()
            _interruptible_sleep(DEFAULT_SLEEP_INTERVAL)
        except ccxt.ExchangeError as e:
            logger.error(f"❌ Exchange error: {e}")
            _interruptible_sleep(DEFAULT_SLEEP_INTERVAL)
        except Exception as e:
            logger.exception(f"❌ Unexpected error: {e}")
            _interruptible_sleep(DEFAULT_SLEEP_INTERVAL)

    # Graceful shutdown
    logger.info("Performing graceful shutdown...")
    save_state(state, state_path)
    save_metrics(metrics, metrics_path)
    logger.info("✅ State and metrics saved")


def _verify_exchange_settings(exchange: ccxt.bingx, config: Dict[str, Any]) -> None:
    """Verify and configure exchange settings."""
    if not verify_position_mode(exchange, config):
        logger.error("❌ Position mode verification failed")
        raise RuntimeError("Position mode verification failed")

    set_margin_mode(exchange, config)


def _should_sync_now(last_sync_time: float, interval_minutes: int) -> bool:
    """
    Check if sync should run at clock-aligned intervals.

    Args:
        last_sync_time: Timestamp of last sync (0 for first run)
        interval_minutes: Sync interval in minutes (e.g., 5 for every 5 minutes)

    Returns:
        True if should sync now
    """
    current_dt = datetime.now()
    current_minute = current_dt.minute

    # First run: always sync
    if last_sync_time == 0:
        return True

    # Check if current minute is at sync interval (00, 05, 10, ...)
    if current_minute % interval_minutes != 0:
        return False

    # Prevent duplicate sync in same minute
    last_sync_dt = datetime.fromtimestamp(last_sync_time)
    if last_sync_dt.hour == current_dt.hour and last_sync_dt.minute == current_minute:
        return False

    return True


def _maybe_sync_position(
    exchange: ccxt.bingx,
    state: Dict[str, Any],
    config: Dict[str, Any],
    cache: APICache,
    circuit_breaker: CircuitBreaker,
    metrics: PerformanceMetrics,
    last_sync_time: float,
) -> float:
    """
    Clock-aligned position synchronization.

    Runs immediately on first call, then at 5-minute intervals (00, 05, 10, ...).

    Returns:
        Updated last_sync_time
    """
    if _should_sync_now(last_sync_time, POSITION_SYNC_INTERVAL_MINUTES):
        sync_position_with_exchange(exchange, state, config, cache, circuit_breaker, metrics)
        return time.time()

    return last_sync_time


def _process_existing_positions(
    exchange: ccxt.bingx,
    state: Dict[str, Any],
    config: Dict[str, Any],
    cache: APICache,
    circuit_breaker: CircuitBreaker,
    metrics: PerformanceMetrics,
) -> bool:
    """
    Check for early exit signals on all active slots.

    Called during trading window after candle settle.
    Position status check (TP/SL hit) is handled by caller.

    Returns:
        True if any trading action occurred (early exit), False otherwise
    """
    positions = state.get('positions') or {}
    if not positions:
        return False

    # Fetch indicators once for all slots
    df = _fetch_and_calculate_indicators(exchange, config)
    if df is None:
        return False

    any_action = False
    try:
        ticker = fetch_ticker_cached(exchange, config['symbol'], cache)
        current_price = ticker['last']

        for position in list(positions.values()):  # copy — dict may mutate during iteration
            should_exit, new_reversal_count, exit_reason, last_counted_ts = check_early_exit_signal(
                position, df, current_price, config
            )

            # Update reversal count and candle timestamp in slot state
            state_changed = False
            if position.get('reversal_count', 0) != new_reversal_count:
                position['reversal_count'] = new_reversal_count
                state_changed = True
            if last_counted_ts is not None and position.get('last_counted_candle_ts') != last_counted_ts:
                position['last_counted_candle_ts'] = last_counted_ts
                state_changed = True
            if state_changed:
                save_state(state)

            # Execute early exit if triggered
            if should_exit and exit_reason:
                slot_id = position.get('slot_id', '?')
                logger.info(f"🚨 Early exit triggered for slot {slot_id}: {exit_reason}")
                success = close_position_market(
                    exchange, state, config, cache, exit_reason, metrics,
                    position=position,
                )
                if success:
                    logger.info(f"✅ Early exit completed for slot {slot_id}: {exit_reason}")
                    any_action = True

    except Exception as e:
        logger.warning(f"Early exit check failed: {e}")

    return any_action


def _ensure_emergency_sl_exists(
    exchange: ccxt.bingx,
    state: Dict[str, Any],
    config: Dict[str, Any],
) -> None:
    """Proactive check: re-place emergency SL if missing for any active direction.

    v1.59.5: Also resolves EXCHANGE_MANAGED state by calling verify
    to find actual order and adopt or cancel-replace it.
    """
    if config.get('position_mode') != 'hedge':
        return
    positions = state.get('positions') or {}
    if not positions:
        return
    active_dirs = {p['direction'] for p in positions.values() if p.get('direction')}
    for direction in active_dirs:
        sl_id = _get_emergency_sl_id(state, config, direction)
        if not sl_id:
            logger.warning(f"🛡️ Emergency SL ({direction}) absent — re-placing proactively")
            _place_emergency_sl_for_direction(exchange, state, config, direction)
        elif sl_id == _EXCHANGE_MANAGED:
            # v1.59.5: EXCHANGE_MANAGED = BingX has its own closePosition order.
            # Per-slot individual SLs provide real protection.
            # Don't retry — BingX may report different stopPrice or hide the order.
            pass



def _estimate_new_sl_pct(
    config: Dict[str, Any],
    pattern_name: Optional[str],
    df: Optional[pd.DataFrame],
) -> float:
    """Estimate effective SL% for a new position (pre-entry).

    Uses pattern's base SL from config + ATR vol multiplier + proportional cap.
    """
    # Get base SL from dynamic per-pattern config
    base_sl = config.get('strategy', {}).get('sl_pct', 1.0)
    if config.get('_dynamic_tpsl_per_pattern') and pattern_name:
        pp_tpsl = config.get('_dynamic_patterns_tpsl', {})
        if pattern_name in pp_tpsl:
            base_sl = pp_tpsl[pattern_name][1]
    elif config.get('_dynamic_tpsl_universal'):
        base_sl = config.get('_dynamic_sl', base_sl)

    # ATR vol multiplier
    vol_mult = 1.0
    if df is not None:
        vol_mult = get_volatility_multiplier(df, config)

    # Proportional cap (same logic as _effective_vol_mult in position_open.py)
    if base_sl > 0:
        leverage = config.get('leverage', 3)
        max_daily_loss = config.get('risk', {}).get('max_daily_loss_pct', 13)
        max_sl_pct = max_daily_loss / leverage
        max_mult = max_sl_pct / base_sl
        vol_mult = min(vol_mult, max_mult)

    return max(0.1, base_sl * vol_mult - SLIPPAGE_BUFFER_PCT)


def _check_momentum_guard(
    state: Dict[str, Any],
    config: Dict[str, Any],
    signal_direction: str,
    df: Optional[pd.DataFrame],
) -> bool:
    """Check if recent price momentum makes counter-direction entry risky.

    v1.36.2: Pauses counter-direction entries during strong short-term moves.
    E.g., if BTC rose >1% in 30min, pause SHORT for 30min.

    Research: entry_improvement_study.py H2 — PnL/MDD 8.19 vs baseline 7.51 (+9%).
    Blocks only ~2.3% of signals but avoids correlated losses during spikes.

    Returns:
        True if momentum guard blocks this entry (skip entry)
    """
    mg_cfg = config.get('risk', {}).get('momentum_guard', {})
    if not mg_cfg.get('enabled', False):
        return False

    lookback_bars = mg_cfg.get('lookback_bars', 6)
    threshold_pct = mg_cfg.get('threshold_pct', 1.0)
    cooldown_bars = mg_cfg.get('cooldown_bars', 6)

    if df is None or len(df) < lookback_bars + 1:
        return False

    closes = df['close'].values
    current_price = closes[-1]
    past_price = closes[-(lookback_bars + 1)]
    if past_price <= 0:
        return False

    pct_change = (current_price / past_price - 1) * 100

    # Strong upward move → block SHORT
    if signal_direction == 'SHORT' and pct_change > threshold_pct:
        logger.info(f"⚡ Momentum guard: BTC {pct_change:+.2f}% in {lookback_bars*5}min "
                     f"(>{threshold_pct}%), blocking SHORT entry")
        # Store cooldown expiry in state for future checks
        state['_momentum_cooldown_short'] = time.time() + cooldown_bars * CANDLE_DURATION_MS // 1000
        save_state(state)  # v1.56.2: persist cooldown across crashes
        return True

    # Strong downward move → block LONG
    if signal_direction == 'LONG' and pct_change < -threshold_pct:
        logger.info(f"⚡ Momentum guard: BTC {pct_change:+.2f}% in {lookback_bars*5}min "
                     f"(<-{threshold_pct}%), blocking LONG entry")
        state['_momentum_cooldown_long'] = time.time() + cooldown_bars * CANDLE_DURATION_MS // 1000
        save_state(state)  # v1.56.2: persist cooldown across crashes
        return True

    # Check if still in cooldown from previous trigger
    cooldown_key = f'_momentum_cooldown_{signal_direction.lower()}'
    cooldown_until = state.get(cooldown_key, 0)
    if time.time() < cooldown_until:
        remaining = int(cooldown_until - time.time())
        logger.info(f"⚡ Momentum guard: {signal_direction} still in cooldown ({remaining}s remaining)")
        return True

    return False


def _record_trade_for_rolling_wr(
    state: Dict[str, Any],
    config: Dict[str, Any],
) -> None:
    """Record closed trade in rolling WR tracker for adaptive leverage.

    v1.39.0: Called after any position close event (same pattern as
    _record_loss_for_burst_brake). Reads last_trade from state to avoid
    double-counting.
    """
    al_cfg = config.get('strategy', {}).get('adaptive_leverage', {})
    if not al_cfg.get('enabled', False):
        return

    last_trade = state.get('last_trade')
    if not last_trade:
        return

    closed_at = last_trade.get('closed_at', '')
    if not closed_at:
        return

    # Avoid double-counting
    last_recorded = state.get('_rolling_wr_last_recorded', '')
    if closed_at == last_recorded:
        return

    pnl_pct = last_trade.get('pnl_pct', 0)
    direction = last_trade.get('direction', '')
    # Extract pattern and TP/SL info
    pattern_name = ''
    tp_pct, sl_pct = 1.0, 1.0
    # These are set in the last_trade by record_closed_position; fallback to defaults
    if not pattern_name:
        pattern_name = 'N/A'

    _update_rolling_wr_tracker(state, pnl_pct, direction, pattern_name, tp_pct, sl_pct)
    state['_rolling_wr_last_recorded'] = closed_at


def _record_loss_for_burst_brake(
    state: Dict[str, Any],
    config: Dict[str, Any],
) -> None:
    """Record a loss trade in the burst tracker for Loss Burst Brake.

    v1.37.0: After 'threshold' same-direction losses within 'window_bars',
    block that direction for 'block_bars'. Research: portfolio_exposure_study.py
    H6b — MDD -36%, burst count -36%, max daily loss -76%.

    Call after any position close event.
    """
    brake_cfg = config.get('risk', {}).get('loss_burst_brake', {})
    if not brake_cfg.get('enabled', False):
        return

    last_trade = state.get('last_trade')
    if not last_trade:
        return

    portfolio_pnl = last_trade.get('portfolio_pnl_pct', last_trade.get('pnl_pct', 0))
    if portfolio_pnl > 0:
        return  # not a loss

    direction = last_trade.get('direction', '')
    if not direction:
        return

    closed_at = last_trade.get('closed_at', '')
    if not closed_at:
        return

    # Avoid double-counting: check if this trade was already recorded
    last_recorded = state.get('_loss_burst_last_recorded', '')
    if closed_at == last_recorded:
        return

    # Record loss timestamp
    tracker = state.get('_loss_burst_tracker') or {}
    if direction not in tracker:
        tracker[direction] = []
    tracker[direction].append(time.time())
    state['_loss_burst_tracker'] = tracker
    state['_loss_burst_last_recorded'] = closed_at

    logger.info(
        f"🛑 Loss burst tracker: recorded {direction} loss "
        f"(pnl={portfolio_pnl:+.2f}%, {len(tracker[direction])} recent)"
    )


def _check_loss_burst_brake(
    state: Dict[str, Any],
    config: Dict[str, Any],
    signal_direction: str,
) -> bool:
    """Check if recent same-direction losses should block entry.

    v1.37.0: If 'threshold' same-direction losses occurred within 'window_bars',
    block that direction for 'block_bars'.

    Returns:
        True if loss burst brake blocks this entry (skip entry)
    """
    brake_cfg = config.get('risk', {}).get('loss_burst_brake', {})
    if not brake_cfg.get('enabled', False):
        return False

    threshold = brake_cfg.get('threshold', 2)
    candle_seconds = CANDLE_DURATION_MS // 1000
    window_seconds = brake_cfg.get('window_bars', 288) * candle_seconds  # bars → seconds
    block_seconds = brake_cfg.get('block_bars', 144) * candle_seconds

    # Check active block
    blocked_key = f'_loss_burst_blocked_until_{signal_direction.lower()}'
    blocked_until = state.get(blocked_key, 0)
    if time.time() < blocked_until:
        remaining = int(blocked_until - time.time())
        remaining_min = remaining // 60
        logger.info(
            f"🛑 Loss burst brake: {signal_direction} blocked "
            f"({remaining_min}min remaining)"
        )
        return True

    # Count recent same-direction losses within window
    tracker = state.get('_loss_burst_tracker') or {}
    dir_losses = tracker.get(signal_direction, [])

    # Clean old entries outside window
    cutoff = time.time() - window_seconds
    dir_losses = [ts for ts in dir_losses if ts > cutoff]

    # Update tracker in state
    if '_loss_burst_tracker' not in state:
        state['_loss_burst_tracker'] = {}
    state['_loss_burst_tracker'][signal_direction] = dir_losses

    if len(dir_losses) >= threshold:
        # Activate block
        state[blocked_key] = time.time() + block_seconds
        state['_loss_burst_tracker'][signal_direction] = []  # reset after activation
        block_hours = block_seconds / 3600
        window_hours = window_seconds / 3600
        logger.warning(
            f"🛑 Loss burst brake ACTIVATED: {len(dir_losses)} {signal_direction} losses "
            f"in {window_hours:.0f}h → blocking {signal_direction} for {block_hours:.0f}h"
        )
        return True

    return False


def _clamp(value: float, lo: float, hi: float) -> float:
    """Clamp value between lo and hi."""
    return max(lo, min(hi, value))


def _leverage_wr_confidence(
    rolling_wr: float, rolling_edge: float,
    expected_wr: float, ref_edge: float,
    min_lev: float, max_lev: float,
) -> float:
    """H5: WR Confidence × Edge Quality leverage."""
    if expected_wr <= 0:
        return min_lev
    wr_conf = _clamp(rolling_wr / expected_wr, 0, 1)
    edge_q = _clamp(rolling_edge / ref_edge, 0, 1.5) if ref_edge > 0 else 0.5
    combined = _clamp(wr_conf * edge_q, 0, 1)
    return min_lev + (max_lev - min_lev) * combined


def _leverage_kelly(
    rolling_wr: float, avg_win: float, avg_loss: float, max_lev: float,
) -> float:
    """H6: Half-Kelly leverage."""
    if avg_loss <= 0:
        return max_lev
    b = avg_win / avg_loss  # R:R ratio
    f = (rolling_wr * (1 + b) - 1) / b  # Kelly fraction
    return _clamp(f * 0.5 * max_lev, 1.0, max_lev)


def _leverage_breakeven(
    rolling_wr: float, tp_pct: float, sl_pct: float,
    min_lev: float, max_lev: float,
) -> float:
    """H7: Breakeven-Aware leverage."""
    if (tp_pct + sl_pct) <= 0:
        return min_lev
    be_wr = sl_pct / (tp_pct + sl_pct)
    margin = rolling_wr - be_wr
    if margin <= 0:
        return min_lev
    max_margin = 0.30
    return min_lev + (max_lev - min_lev) * _clamp(margin / max_margin, 0, 1)


def _leverage_exp_decay(
    rolling_wr: float, expected_wr: float,
    min_lev: float, max_lev: float,
) -> float:
    """H9: Exponential Decay (Conservative) leverage."""
    gap = max(expected_wr - rolling_wr, 0)
    decay = math.exp(-gap / 0.10)  # 10pp gap → 0.37, 20pp → 0.14
    return min_lev + (max_lev - min_lev) * decay


def _compute_adaptive_leverage(
    state: Dict[str, Any],
    config: Dict[str, Any],
    signal_direction: str,
    pattern_name: Optional[str],
) -> float:
    """Compute dynamic leverage based on rolling WR vs expected.

    v1.39.0: Returns config['leverage'] when adaptive_leverage is disabled.
    """
    al_cfg = config.get('strategy', {}).get('adaptive_leverage', {})
    if not al_cfg.get('enabled', False):
        return config.get('leverage', 3)

    min_lev = al_cfg.get('min_leverage', 1.0)
    max_lev = al_cfg.get('max_leverage', 3.0)
    window = al_cfg.get('window', 10)
    method = al_cfg.get('method', 'wr_confidence')
    expected_wr = config.get('_npos_portfolio_wr', al_cfg.get('expected_wr', 73.2) / 100)
    ref_edge = config.get('_npos_ref_edge', al_cfg.get('ref_edge', 0.126) / 100)

    tracker = state.get('rolling_wr_tracker') or {}
    recent = tracker.get('recent_trades', [])
    if len(recent) < 3:  # minimum trades for meaningful WR
        logger.debug(f"Adaptive leverage: insufficient trades ({len(recent)}), using min={min_lev}")
        return min_lev

    # Calculate rolling stats from recent trades
    recent_window = recent[-window:]
    wins = sum(1 for t in recent_window if t[0] >= 0)
    rolling_wr = wins / len(recent_window)
    rolling_edge = np.mean([t[0] for t in recent_window]) / 100  # pct → fraction

    # Get per-pattern R:R for breakeven methods
    tp_pct, sl_pct = 1.0, 1.0  # defaults
    if pattern_name:
        tpsl = config.get('_dynamic_patterns_tpsl', {}).get(pattern_name, [1.0, 1.0])
        tp_pct, sl_pct = tpsl[0] / 100, tpsl[1] / 100

    if method == 'wr_confidence':
        result = _leverage_wr_confidence(rolling_wr, rolling_edge, expected_wr, ref_edge, min_lev, max_lev)
    elif method == 'kelly':
        wins_pnl = [t[0] for t in recent_window if t[0] > 0]
        losses_pnl = [t[0] for t in recent_window if t[0] < 0]
        avg_win = np.mean(wins_pnl) if wins_pnl else 0
        avg_loss = abs(np.mean(losses_pnl)) if losses_pnl else 1
        result = _leverage_kelly(rolling_wr, avg_win, avg_loss, max_lev)
    elif method == 'breakeven':
        result = _leverage_breakeven(rolling_wr, tp_pct, sl_pct, min_lev, max_lev)
    elif method == 'combined':
        wr_score = _leverage_wr_confidence(rolling_wr, rolling_edge, expected_wr, ref_edge, 0, 1)
        be_score = _leverage_breakeven(rolling_wr, tp_pct, sl_pct, 0, 1)
        blend = 0.6 * wr_score + 0.4 * be_score
        result = min_lev + (max_lev - min_lev) * _clamp(blend, 0, 1)
    elif method == 'exp_decay':
        result = _leverage_exp_decay(rolling_wr, expected_wr, min_lev, max_lev)
    else:
        result = config['leverage']

    logger.info(
        f"📊 Adaptive leverage: method={method}, rolling_WR={rolling_wr:.1%} "
        f"(w={len(recent_window)}), edge={rolling_edge:.4f}, → {result:.2f}x"
    )
    return result


def _update_rolling_wr_tracker(
    state: Dict[str, Any],
    pnl_pct: float,
    direction: str,
    pattern_name: str,
    tp_pct: float,
    sl_pct: float,
) -> None:
    """Update rolling WR tracker after position close.

    v1.39.0: Keeps last 50 trades for rolling window calculations.
    """
    tracker = state.get('rolling_wr_tracker') or {'recent_trades': []}
    recent = tracker.get('recent_trades', [])
    recent.append([pnl_pct, direction, pattern_name, tp_pct, sl_pct])
    # Keep last 50 trades (5× max window)
    if len(recent) > 50:
        recent = recent[-50:]
    tracker['recent_trades'] = recent
    state['rolling_wr_tracker'] = tracker


def _check_equity_curve_sizing(
    state: Dict[str, Any],
    config: Dict[str, Any],
    direction: str,
) -> float:
    """Return size multiplier based on per-direction equity curve vs SMA.

    v1.40.0: Equity Curve Trading (Combo2-H1). When a direction's cumulative
    PnL falls below its SMA(ema_trades), reduce that direction's position
    size by size_mult (e.g. 0.5). LONG/SHORT tracked independently.

    Returns 1.0 (normal) or config size_mult when equity < SMA.
    """
    ec_cfg = config.get('risk', {}).get('equity_curve_trading', {})
    if not ec_cfg.get('enabled', False):
        return 1.0

    ema_trades = ec_cfg.get('ema_trades', 30)
    size_mult = ec_cfg.get('size_mult', 0.5)

    tracker = state.get('equity_curve_tracker') or {}
    key = 'long_cum_pnls' if direction == 'LONG' else 'short_cum_pnls'
    cum_pnls = tracker.get(key, [])

    if len(cum_pnls) < ema_trades:
        return 1.0  # not enough data yet

    # SMA of last ema_trades values
    window = cum_pnls[-ema_trades:]
    sma = sum(window) / ema_trades
    current = cum_pnls[-1]

    if current < sma:
        logger.info(
            f"📉 Equity curve sizing: {direction} cum_pnl={current:.2f}% < SMA({ema_trades})={sma:.2f}% → scale={size_mult}"
        )
        return size_mult
    return 1.0


def _check_correlation_aware_entry(
    state: Dict[str, Any],
    config: Dict[str, Any],
    signal_direction: str,
    df: Optional[pd.DataFrame],
) -> bool:
    """Return True if entry should be BLOCKED (correlation guard).

    v1.40.0: Correlation-Aware Entry (Combo2-H6). Block when same-direction
    ratio >= threshold AND new entry is counter-regime.

    Returns:
        True if entry should be blocked
    """
    ca_cfg = config.get('risk', {}).get('correlation_aware_entry', {})
    if not ca_cfg.get('enabled', False):
        return False

    positions = state.get('positions') or {}
    if len(positions) < 2:
        return False

    dir_pct = ca_cfg.get('dir_pct_threshold', 0.70)

    same_dir = sum(1 for p in positions.values() if p.get('direction') == signal_direction)
    ratio = same_dir / len(positions)

    if ratio < dir_pct:
        return False

    # Check if counter-regime using existing regime_sizing EMA params
    regime_cfg = config.get('risk', {}).get('regime_sizing', {})
    ema_period = regime_cfg.get('ema_period', 20)
    lookback = regime_cfg.get('lookback', 5)

    if df is not None and len(df) >= ema_period + lookback:
        closes = df['close'].values
        ema = pd.Series(closes).ewm(span=ema_period, adjust=False).mean().values
        slope = ema[-1] - ema[-1 - lookback]
        is_uptrend = slope > 0
        is_counter = (
            (is_uptrend and signal_direction == 'SHORT') or
            (not is_uptrend and signal_direction == 'LONG')
        )
        if is_counter:
            logger.info(
                f"🛡️ Correlation guard: {signal_direction} dir_ratio={ratio:.0%} >= {dir_pct:.0%}, "
                f"counter-regime ({'UP' if is_uptrend else 'DOWN'}) → BLOCK"
            )
            return True

    return False


def _update_equity_curve_tracker(
    state: Dict[str, Any],
    direction: str,
    portfolio_pnl_pct: float,
) -> None:
    """Record cumulative PnL after trade close for equity curve trading.

    v1.40.0: Per-direction tracking — LONG/SHORT independent curves.
    Each entry is the running cumulative sum of portfolio PnL for that direction.
    """
    tracker = state.setdefault('equity_curve_tracker', {
        'long_cum_pnls': [],
        'short_cum_pnls': [],
    })
    key = 'long_cum_pnls' if direction == 'LONG' else 'short_cum_pnls'
    cum_pnls = tracker.get(key, [])
    prev = cum_pnls[-1] if cum_pnls else 0.0
    cum_pnls.append(round(prev + portfolio_pnl_pct, 4))
    # Keep last 100 entries (3x+ of max EMA period)
    if len(cum_pnls) > 100:
        cum_pnls = cum_pnls[-100:]
    tracker[key] = cum_pnls


def _record_trade_for_equity_curve(
    state: Dict[str, Any],
    config: Dict[str, Any],
) -> None:
    """Record trade for equity curve tracker after position close.

    v1.40.0: Called from trading/maintenance window after trade exit detection.
    Similar pattern to _record_loss_for_burst_brake and _record_trade_for_rolling_wr.
    """
    ec_cfg = config.get('risk', {}).get('equity_curve_trading', {})
    if not ec_cfg.get('enabled', False):
        return

    last_trade = state.get('last_trade')
    if not last_trade:
        return

    closed_at = last_trade.get('closed_at', '')
    if not closed_at:
        return

    # Avoid double-counting
    last_recorded = state.get('_equity_curve_last_recorded', '')
    if closed_at == last_recorded:
        return

    direction = last_trade.get('direction', '')
    if not direction:
        return

    portfolio_pnl = last_trade.get('portfolio_pnl_pct', last_trade.get('pnl_pct', 0))
    _update_equity_curve_tracker(state, direction, portfolio_pnl)
    state['_equity_curve_last_recorded'] = closed_at


def _check_aggregate_risk_cap(
    state: Dict[str, Any],
    config: Dict[str, Any],
    signal_direction: str,
    pattern_name: Optional[str],
    df: Optional[pd.DataFrame],
) -> bool:
    """Check if new entry would exceed aggregate directional SL exposure cap.

    v1.35.5: Limits correlated risk from multiple same-direction positions.
    Uses EMA(20) regime to set tighter cap for counter-regime entries.

    Aggregate SL exposure = sum(sl_dist_pct * (1/N) * leverage) per direction.
    Research: dynamic_3_7 reduced MDD by 52% (13.2%→6.3%) with WF 3/3 PASS.

    Returns:
        True if cap would be exceeded (skip entry)
    """
    cap_cfg = config.get('risk', {}).get('aggregate_risk_cap', {})
    if not cap_cfg.get('enabled', False):
        return False

    counter_cap = cap_cfg.get('counter_cap', 3.0)
    with_cap = cap_cfg.get('with_cap', 7.0)
    leverage = config.get('leverage', 3)
    max_positions = config.get('max_positions', 9)

    # Determine regime from EMA slope (reuse regime_sizing params)
    regime_cfg = config.get('risk', {}).get('regime_sizing', {})
    ema_period = regime_cfg.get('ema_period', 20)
    lookback = regime_cfg.get('lookback', 5)

    is_uptrend = False
    if df is not None and len(df) >= ema_period + lookback:
        closes = df['close'].values
        ema = pd.Series(closes).ewm(span=ema_period, adjust=False).mean().values
        slope = ema[-1] - ema[-1 - lookback]
        is_uptrend = slope > 0

    is_counter = (
        (signal_direction == 'SHORT' and is_uptrend) or
        (signal_direction == 'LONG' and not is_uptrend)
    )
    cap = counter_cap if is_counter else with_cap

    # Compute aggregate SL exposure from existing positions
    # v1.60.0: Exclude pending-close slots
    positions = state.get('positions') or {}
    agg_exposure = 0.0
    for pos in positions.values():
        if pos.get('_pending_close'):
            continue
        if pos.get('direction') != signal_direction:
            continue
        entry = pos.get('entry_price', 0)
        sl = pos.get('sl_price', 0)
        if entry <= 0:
            continue
        sl_dist_pct = abs(entry - sl) / entry * 100
        slot_size = 1.0 / max_positions
        slot_lev = pos.get('effective_leverage', leverage)
        agg_exposure += sl_dist_pct * slot_size * slot_lev

    # Compute new position's SL exposure
    new_sl_pct = _estimate_new_sl_pct(config, pattern_name, df)
    new_exposure = new_sl_pct * (1.0 / max_positions) * leverage

    total = agg_exposure + new_exposure
    if total > cap:
        regime_str = 'UP' if is_uptrend else 'DOWN'
        logger.info(
            f"🛡️ Aggregate risk cap: {signal_direction} exposure {agg_exposure:.2f}% "
            f"+ new {new_exposure:.2f}% = {total:.2f}% > cap {cap:.1f}% "
            f"(regime={regime_str}, {'counter' if is_counter else 'with'})"
        )
        return True

    return False


def _route_signal(
    state: Dict[str, Any],
    config: Dict[str, Any],
    signal_direction: str,
) -> str:
    """
    Route a trading signal to the appropriate action.

    Hedge mode: direction-agnostic, OPEN if slots available, else SKIP.
    One-Way mode: same-direction OPEN, opposite → CLOSE_OLDEST (FIFO).

    Returns:
        'OPEN' — open new slot
        'CLOSE_OLDEST' — close oldest opposite-direction slot (One-Way FIFO)
        'SKIP' — ignore signal (slots full)
    """
    positions = state.get('positions') or {}
    max_pos = config.get('max_positions', 1)

    if not positions:
        return 'OPEN'

    is_hedge = config.get('position_mode') == 'hedge'

    if is_hedge:
        # Hedge mode: mixed directions allowed, check slot count + direction cap
        # v1.60.0: Exclude pending-close slots from counts
        active_positions = {
            sid: p for sid, p in positions.items()
            if not p.get('_pending_close')
        }
        if len(active_positions) >= max_pos:
            return 'SKIP'
        direction_cap = config.get('strategy', {}).get('direction_cap', max_pos)
        same_dir_count = sum(
            1 for p in active_positions.values()
            if p.get('direction') == signal_direction
        )
        if same_dir_count >= direction_cap:
            return 'SKIP'
        return 'OPEN'

    # One-Way mode: same-direction only + FIFO close on opposite
    active_dir = state.get('active_direction')
    if signal_direction == active_dir:
        if len(positions) < max_pos:
            return 'OPEN'
        return 'SKIP'

    # Opposite direction
    return 'CLOSE_OLDEST'


def _get_oldest_slot(state: Dict[str, Any]) -> Optional[Dict]:
    """Get the oldest active slot (by entry_time, FIFO)."""
    positions = state.get('positions') or {}
    if not positions:
        return None
    return min(positions.values(), key=lambda s: s.get('entry_time', ''))


def _process_entry_signal(
    exchange: ccxt.bingx,
    state: Dict[str, Any],
    config: Dict[str, Any],
    cache: APICache,
    circuit_breaker: CircuitBreaker,
    metrics: PerformanceMetrics,
) -> bool:
    """
    Check for entry signals and route based on slot availability.

    Called during trading window. Handles OPEN, CLOSE_OLDEST, and SKIP.

    Returns:
        True if trading action occurred, False otherwise
    """
    # Note: No early return for full slots — _route_signal handles SKIP vs CLOSE_OLDEST

    # Check cooldown
    if not check_cooldown(state, config):
        return False

    # Fetch and calculate indicators
    df = _fetch_and_calculate_indicators(exchange, config)
    if df is None:
        return False

    # Check for entry signal
    signal_result, reason = check_entry_signal(df, state, config)

    # v1.66.0: Store detected signal direction for opposite-signal exit
    state['_last_detected_signal'] = signal_result  # None if no signal

    if not signal_result:
        return False

    logger.info(f"🎯 Signal detected: {signal_result} | {reason}")

    action = _route_signal(state, config, signal_result)

    if action == 'OPEN':
        # v1.36.2: Momentum guard — pause counter-dir entries during strong moves
        if _check_momentum_guard(state, config, signal_result, df):
            return False

        # v1.37.0: Loss burst brake — block direction after burst same-dir losses
        if _check_loss_burst_brake(state, config, signal_result):
            return False

        # v1.40.0: Correlation-aware entry guard — block counter-regime when dir-concentrated
        if _check_correlation_aware_entry(state, config, signal_result, df):
            return False

        # v1.35.5: Aggregate directional risk cap check
        pattern_name = extract_pattern_name(reason)
        if _check_aggregate_risk_cap(state, config, signal_result, pattern_name, df):
            return False

        # v1.39.0: Compute adaptive leverage before opening
        adaptive_lev = _compute_adaptive_leverage(state, config, signal_result, pattern_name)

        # v1.40.0: Equity curve sizing — reduce size when direction's curve < SMA
        ec_mult = _check_equity_curve_sizing(state, config, signal_result)

        success = open_position(
            exchange, state, config, signal_result, reason,
            cache, circuit_breaker, metrics, df,
            leverage_override=adaptive_lev,
            equity_curve_scale=ec_mult,
        )
        if success:
            logger.info(f"✅ Position opened: {signal_result} (slot {len(state.get('positions') or {})})")
            return True
        return False

    elif action == 'CLOSE_OLDEST':
        oldest = _get_oldest_slot(state)
        if oldest:
            slot_id = oldest.get('slot_id', '?')
            logger.info(f"🔄 Closing oldest slot {slot_id} for opposite signal")
            close_position_market(
                exchange, state, config, cache, 'OPPOSITE_SIGNAL', metrics,
                position=oldest,
            )
            # Re-route: if slots now empty, immediately open opposite direction
            re_action = _route_signal(state, config, signal_result)
            if re_action == 'OPEN':
                logger.info(f"🔄 Re-routing: slots empty → immediate {signal_result} entry")
                success = open_position(
                    exchange, state, config, signal_result, reason,
                    cache, circuit_breaker, metrics, df
                )
                if success:
                    logger.info(f"✅ Reverse entry: {signal_result} (slot {len(state.get('positions') or {})})")
            return True

    # SKIP: do nothing
    return False


def _wait_for_candle_settle(config: Dict[str, Any]) -> None:
    """Wait for new candle data to settle."""
    now_ms = int(time.time() * 1000)
    candle_progress = now_ms % CANDLE_DURATION_MS
    time_since_close = candle_progress / 1000

    if time_since_close < CANDLE_SETTLE_SECONDS:
        wait_time = CANDLE_SETTLE_SECONDS - time_since_close
        logger.debug(f"Waiting {wait_time:.1f}s for candle to settle")
        _interruptible_sleep(wait_time)


def _fetch_and_calculate_indicators(
    exchange: ccxt.bingx,
    config: Dict[str, Any],
) -> Optional[pd.DataFrame]:
    """Fetch OHLCV data and calculate indicators."""
    try:
        t0 = time.time()
        ohlcv = fetch_ohlcv(
            exchange,
            config['symbol'],
            config['timeframe'],
            MAX_OHLCV_CANDLES
        )

        if not ohlcv or len(ohlcv) < 20:
            logger.warning("Insufficient OHLCV data")
            return None

        df = pd.DataFrame(
            ohlcv,
            columns=['timestamp', 'open', 'high', 'low', 'close', 'volume']
        )
        df = calculate_indicators(df, config)
        elapsed_ms = (time.time() - t0) * 1000
        logger.debug(f"OHLCV fetch + indicators: {elapsed_ms:.0f}ms ({len(ohlcv)} candles)")

        return df

    except ccxt.NetworkError as e:
        logger.error(f"Failed to fetch indicators (network error): {e}")
        return None
    except ccxt.ExchangeError as e:
        logger.error(f"Failed to fetch indicators (exchange error): {e}")
        return None
    except Exception as e:
        logger.exception(f"Failed to fetch/calculate indicators: {e}")
        return None


def _log_position_status(
    position: Dict[str, Any],
    config: Dict[str, Any],
    cache: APICache,
    exchange: ccxt.bingx,
) -> None:
    """Log current position status with TP/SL progress."""
    try:
        ticker = fetch_ticker_cached(exchange, config['symbol'], cache)
        current_price = ticker['last']
        direction = 1 if position['direction'] == 'LONG' else -1
        pnl_pct = direction * (current_price / position['entry_price'] - 1) * 100 * config['leverage']

        # Extract pattern name
        pattern_name = extract_pattern_name(position.get('reason', ''))

        # Calculate TP/SL progress (0% = at entry, 100% = at target)
        entry = position['entry_price']
        tp = position.get('tp_price', 0)
        sl = position.get('sl_price', 0)

        tp_progress = 0
        sl_progress = 0
        if tp and entry and tp != entry:
            tp_progress = max(0, direction * (current_price - entry) / (direction * (tp - entry)) * 100)
        if sl and entry and sl != entry:
            sl_progress = max(0, -direction * (current_price - entry) / (-direction * (sl - entry)) * 100)

        dir_label = position['direction']
        pattern_str = f" {pattern_name}" if pattern_name else ""

        logger.info(
            f"📊 {dir_label}{pattern_str} | "
            f"${entry:.1f} → ${current_price:.1f} | "
            f"PnL: {pnl_pct:+.2f}% (lev) | "
            f"TP {tp_progress:.0f}% done | SL {sl_progress:.0f}%"
        )
    except ccxt.NetworkError as e:
        logger.debug(f"Could not log position status (network error): {e}")
    except Exception as e:
        logger.debug(f"Could not log position status: {e}")


def _log_waiting_status(
    state: Dict[str, Any],
    metrics: PerformanceMetrics,
    cache: APICache,
    exchange: ccxt.bingx,
    config: Dict[str, Any],
) -> None:
    """Log waiting for signal status."""
    try:
        ticker = fetch_ticker_cached(exchange, config['symbol'], cache)
        current_price = ticker['last']

        daily_pnl = state.get('daily_pnl', 0)
        daily_trades = state.get('daily_trades', 0)

        # Direction concentration of open positions (v1.36.5)
        positions = state.get('positions', {})
        n_long = sum(1 for p in positions.values() if p.get('direction') == 'LONG')
        n_short = sum(1 for p in positions.values() if p.get('direction') == 'SHORT')
        pos_info = f" | Pos: {n_long}L/{n_short}S" if positions else ""

        logger.info(
            f"⏳ Waiting | BTC: ${current_price:.0f} | "
            f"Daily: {daily_pnl:+.2f}% ({daily_trades} trades) | "
            f"Total: {metrics.total_trades} trades, {metrics.actual_win_rate:.1f}% WR"
            f"{pos_info}"
        )

        # Consecutive losses warning (v1.36.5)
        consec = state.get('consecutive_losses', 0)
        if consec >= 5:
            logger.warning(
                f"⚠️  {consec} CONSECUTIVE LOSSES — "
                f"MDD sizing active, check direction concentration"
            )
    except ccxt.NetworkError as e:
        logger.debug(f"Could not log waiting status (network error): {e}")
    except Exception as e:
        logger.debug(f"Could not log waiting status: {e}")


def _maybe_run_health_check(
    exchange: ccxt.bingx,
    config: Dict[str, Any],
    cache: APICache,
    circuit_breaker: CircuitBreaker,
    metrics: PerformanceMetrics,
    state_file: str,
    last_health_check_time: float,
    interval_minutes: int = 30,
) -> float:
    """
    Clock-aligned health check.

    Runs immediately on first call, then at specified intervals (default 30 min: 00, 30).

    Returns:
        Updated last_health_check_time
    """
    if _should_sync_now(last_health_check_time, interval_minutes):
        _run_health_check(exchange, config, cache, circuit_breaker, metrics, state_file)
        return time.time()

    return last_health_check_time


def _run_health_check(
    exchange: ccxt.bingx,
    config: Dict[str, Any],
    cache: APICache,
    circuit_breaker: CircuitBreaker,
    metrics: PerformanceMetrics,
    state_file: str,
) -> None:
    """Run and log health check."""
    try:
        health = health_check(
            exchange, config, cache, circuit_breaker, metrics, state_file
        )
        status = health.get('status', 'unknown')

        if status != 'healthy':
            logger.warning(f"⚠️ Health check: {status}")
            for check_name, check_result in health.get('checks', {}).items():
                if check_result.get('status') != 'ok':
                    logger.warning(f"  - {check_name}: {check_result}")
        else:
            logger.debug("✅ Health check passed")
    except Exception as e:
        logger.warning(f"Health check failed: {e}")


def _log_pattern_summary(config: Dict[str, Any]) -> None:
    """Log active pattern count and TP/SL range at startup."""
    long_patterns = config.get('strategy', {}).get('long_patterns', VALIDATED_LONG_PATTERNS)
    short_patterns = config.get('strategy', {}).get('short_patterns', VALIDATED_SHORT_PATTERNS)

    source = 'dynamic' if config.get('_dynamic_tpsl_per_pattern') or config.get('_dynamic_tpsl_universal') else 'static'
    tp_sl_mode = 'per-pattern' if config.get('_dynamic_tpsl_per_pattern') else 'universal' if config.get('_dynamic_tpsl_universal') else 'static'

    logger.info(
        f"Patterns: {len(long_patterns)}L + {len(short_patterns)}S = "
        f"{len(long_patterns) + len(short_patterns)} ({source}, {tp_sl_mode})"
    )

    if config.get('_dynamic_tpsl_per_pattern'):
        tpsl = config.get('_dynamic_patterns_tpsl', {})
        if tpsl:
            tps = [v[0] for v in tpsl.values()]
            sls = [v[1] for v in tpsl.values()]
            logger.info(
                f"TP range: {min(tps):.1f}%~{max(tps):.1f}% | "
                f"SL range: {min(sls):.1f}%~{max(sls):.1f}%"
            )
    elif config.get('_dynamic_tpsl_universal'):
        logger.info(f"Universal TP: {config['_dynamic_tp']}% | SL: {config['_dynamic_sl']}%")


# Entry point for direct execution
if __name__ == '__main__':
    run_bot()
