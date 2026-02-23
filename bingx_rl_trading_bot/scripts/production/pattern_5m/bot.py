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
import time
import signal
import logging
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
    MAX_OHLCV_CANDLES,
    DEFAULT_SLEEP_INTERVAL,
    DAILY_LOSS_PAUSE_SECONDS,
    CONSECUTIVE_LOSS_PAUSE_SECONDS,
    MAX_CONSECUTIVE_LOSSES,
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
)
from .models import APICache, CircuitBreaker, PerformanceMetrics
from .config import load_config, validate_config, load_dynamic_patterns
from .state import (
    load_state,
    save_state,
    load_metrics,
    save_metrics,
    sync_metrics_with_state,
)
from .exchange import (
    create_exchange,
    verify_position_mode,
    set_margin_mode,
    fetch_ohlcv,
    fetch_ticker_cached,
    health_check,
    set_shutdown_checker,
)
from .indicators import calculate_indicators
from .signals import check_entry_signal, check_cooldown, check_daily_loss_limit, check_consecutive_loss_limit, check_early_exit_signal
from .position import (
    open_position,
    check_position_status,
    sync_position_with_exchange,
    recover_from_crash,
    close_position_market,
)
from .orders import verify_tp_sl_orders, adjust_tpsl_to_config
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

    # Crash recovery
    recover_from_crash(exchange, state, config, cache, circuit_breaker, metrics)

    # Check if existing positions' TP/SL match current config
    if state.get('positions') or {}:
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

            # Check consecutive loss limit (v1.27.0)
            if check_consecutive_loss_limit(state) and not (state.get('positions') or {}):
                consec = state.get('consecutive_losses', 0)
                logger.warning(f"⚠️ {consec} consecutive losses (limit={MAX_CONSECUTIVE_LOSSES}), pausing {CONSECUTIVE_LOSS_PAUSE_SECONDS}s")
                _interruptible_sleep(CONSECUTIVE_LOSS_PAUSE_SECONDS)
                state['consecutive_losses'] = 0  # Reset after pause
                save_state(state)  # Persist reset to prevent re-pause on crash
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
                if state.get('positions') or {}:
                    _check_position_timeouts(
                        exchange, state, config, cache, circuit_breaker, metrics
                    )

                # 2. Check for new entry signal (always, if slots available)
                _process_entry_signal(
                    exchange, state, config, cache, circuit_breaker, metrics
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
                            _log_position_status(slot, config, cache, exchange)
                        last_log_status_time = now
                else:
                    if now - last_log_status_time >= LOG_STATUS_INTERVAL_SECONDS:
                        _log_waiting_status(state, metrics, cache, exchange, config)
                        last_log_status_time = now

                if now - last_metrics_save_time >= METRICS_SAVE_INTERVAL_SECONDS:
                    save_metrics(metrics, metrics_path)
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
    timeout_seconds = timeout_bars * 300  # 5min = 300s per bar

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
        if len(positions) >= max_pos:
            return 'SKIP'
        direction_cap = config.get('strategy', {}).get('direction_cap', max_pos)
        same_dir_count = sum(
            1 for p in positions.values()
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
    if not signal_result:
        return False

    logger.info(f"🎯 Signal detected: {signal_result} | {reason}")

    action = _route_signal(state, config, signal_result)

    if action == 'OPEN':
        success = open_position(
            exchange, state, config, signal_result, reason,
            cache, circuit_breaker, metrics, df
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

        logger.info(
            f"⏳ Waiting | BTC: ${current_price:.0f} | "
            f"Daily: {daily_pnl:+.2f}% ({daily_trades} trades) | "
            f"Total: {metrics.total_trades} trades, {metrics.actual_win_rate:.1f}% WR"
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
