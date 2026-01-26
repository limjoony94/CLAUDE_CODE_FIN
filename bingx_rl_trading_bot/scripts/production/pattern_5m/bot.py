"""
Pattern 5m Bot - Main Bot Logic
The main trading loop and bot orchestration.
"""

import os
import sys
import time
import signal
import logging
import pandas as pd
from datetime import datetime
from typing import Dict, Any, Optional

import ccxt

from .constants import (
    BOT_NAME,
    BOT_VERSION,
    CONFIG_FILE,
    STATE_FILE,
    METRICS_FILE,
    MAX_OHLCV_CANDLES,
    DEFAULT_SLEEP_INTERVAL,
    POSITION_CHECK_SLEEP,
    DAILY_LOSS_PAUSE_SECONDS,
    CANDLE_SETTLE_SECONDS,
    CANDLE_DURATION_MS,
    TP_SL_CHECK_INTERVAL,
    LOG_STATUS_INTERVAL,
    METRICS_SAVE_INTERVAL,
    POSITION_SYNC_INTERVAL_MINUTES,
    DEFAULT_HEALTH_CHECK_INTERVAL,
)
from .models import APICache, CircuitBreaker, PerformanceMetrics
from .config import load_config, validate_config
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
)
from .indicators import calculate_indicators
from .signals import check_entry_signal, check_cooldown, check_daily_loss_limit, check_early_exit_signal
from .position import (
    open_position,
    check_position_status,
    sync_position_with_exchange,
    recover_from_crash,
    close_position_market,
)
from .orders import verify_tp_sl_orders, adjust_tpsl_to_config
from .utils.lock import acquire_lock, release_lock
from .utils.logging_config import setup_logging

logger = logging.getLogger('pattern_5m')

# Global shutdown flag
shutdown_requested = False


def signal_handler(signum, frame):
    """Handle shutdown signals gracefully."""
    global shutdown_requested
    logger.info(f"\n🛑 Received signal {signum}, initiating graceful shutdown...")
    shutdown_requested = True


def run_bot(config_file: str = CONFIG_FILE) -> None:
    """
    Main entry point for the Pattern 5m trading bot.

    Args:
        config_file: Path to configuration YAML file
    """
    # Resolve paths
    script_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)
    ))))
    os.chdir(script_dir)

    config_path = os.path.join(script_dir, config_file)
    state_path = os.path.join(script_dir, STATE_FILE)
    metrics_path = os.path.join(script_dir, METRICS_FILE)

    # Load configuration
    config = load_config(config_path)
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

    # Acquire lock
    lock_acquired = acquire_lock()
    if not lock_acquired:
        logger.error("❌ Failed to acquire lock - another instance may be running")
        sys.exit(1)

    # Setup signal handlers
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

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
    Main bot loop implementation.

    Args:
        config: Bot configuration dictionary
        state_path: Path to state JSON file
        metrics_path: Path to metrics JSON file
    """
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

    # Check if existing position's TP/SL matches current config
    if state.get('position'):
        adjust_tpsl_to_config(exchange, state, config)

    # Main loop counters
    iteration = 0
    last_sync_time = time.time()
    health_check_interval = config.get('health_check_interval', DEFAULT_HEALTH_CHECK_INTERVAL)

    logger.info("✅ Bot initialized, starting main loop...")

    while not shutdown_requested:
        try:
            iteration += 1

            # Check daily loss limit
            if check_daily_loss_limit(state, config):
                logger.warning(f"⚠️ Daily loss limit reached, pausing {DAILY_LOSS_PAUSE_SECONDS}s")
                time.sleep(DAILY_LOSS_PAUSE_SECONDS)
                continue

            # Periodic position sync
            last_sync_time = _maybe_sync_position(
                exchange, state, config, cache, circuit_breaker, metrics, last_sync_time
            )

            # Process position or look for signals
            if state.get('position'):
                _process_existing_position(
                    exchange, state, config, cache, circuit_breaker, metrics, iteration
                )
            else:
                _process_no_position(
                    exchange, state, config, cache, circuit_breaker, metrics, iteration
                )

            # Periodic health check
            if iteration % health_check_interval == 0:
                _run_health_check(exchange, config, cache, circuit_breaker, metrics, state_path)

            # Periodic metrics save
            if iteration % METRICS_SAVE_INTERVAL == 0:
                save_metrics(metrics, metrics_path)

        except ccxt.NetworkError as e:
            logger.warning(f"⚠️ Network error: {e}")
            circuit_breaker.record_failure()
            time.sleep(DEFAULT_SLEEP_INTERVAL)
        except ccxt.ExchangeError as e:
            logger.error(f"❌ Exchange error: {e}")
            time.sleep(DEFAULT_SLEEP_INTERVAL)
        except Exception as e:
            logger.exception(f"❌ Unexpected error: {e}")
            time.sleep(DEFAULT_SLEEP_INTERVAL)

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
    Periodic position synchronization.

    Returns:
        Updated last_sync_time
    """
    current_time = time.time()
    sync_interval = POSITION_SYNC_INTERVAL_MINUTES * 60

    if (current_time - last_sync_time) >= sync_interval:
        sync_position_with_exchange(exchange, state, config, cache, circuit_breaker, metrics)
        return current_time

    return last_sync_time


def _process_existing_position(
    exchange: ccxt.bingx,
    state: Dict[str, Any],
    config: Dict[str, Any],
    cache: APICache,
    circuit_breaker: CircuitBreaker,
    metrics: PerformanceMetrics,
    iteration: int,
) -> None:
    """Process when a position exists."""
    from .state import save_state

    position = state['position']

    # Check if position is still open
    position_closed = check_position_status(
        exchange, state, config, cache, circuit_breaker, metrics
    )

    if position_closed:
        logger.info("📤 Position closed")
        return

    # === EARLY EXIT CHECK (v1.3) ===
    # Check for reversal signals that should trigger early exit
    # This runs before refill logic to ensure early exit takes priority
    _wait_for_candle_settle(config)
    df = _fetch_and_calculate_indicators(exchange, config)

    if df is not None:
        # Get current price for PnL calculation
        try:
            ticker = fetch_ticker_cached(exchange, config['symbol'], cache)
            current_price = ticker['last']

            # Check early exit signal (v1.14.1: now returns last_counted_candle_ts)
            should_exit, new_reversal_count, exit_reason, last_counted_ts = check_early_exit_signal(
                position, df, current_price, config
            )

            # Update reversal count and candle timestamp in position state
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
                logger.info(f"🚨 Early exit triggered: {exit_reason}")
                success = close_position_market(
                    exchange, state, config, cache, exit_reason, metrics
                )
                if success:
                    logger.info(f"✅ Early exit completed: {exit_reason}")
                return

        except Exception as e:
            logger.warning(f"Early exit check failed: {e}")

    # Note: Rotation/refill feature disabled for pattern_5m bot
    # The check_refill_signal function is not implemented in this module
    # If rotation is needed, implement check_refill_signal in signals.py

    # Log position status periodically
    if iteration % LOG_STATUS_INTERVAL == 0:
        _log_position_status(position, config, cache, exchange)

    # Verify TP/SL orders exist
    if iteration % TP_SL_CHECK_INTERVAL == 0:
        verify_tp_sl_orders(exchange, state, config)

    time.sleep(POSITION_CHECK_SLEEP)


def _process_no_position(
    exchange: ccxt.bingx,
    state: Dict[str, Any],
    config: Dict[str, Any],
    cache: APICache,
    circuit_breaker: CircuitBreaker,
    metrics: PerformanceMetrics,
    iteration: int,
) -> None:
    """Process when no position exists - look for entry signals."""
    # Check cooldown
    if not check_cooldown(state, config):
        time.sleep(DEFAULT_SLEEP_INTERVAL)
        return

    # Wait for candle to settle
    _wait_for_candle_settle(config)

    # Fetch and calculate indicators
    df = _fetch_and_calculate_indicators(exchange, config)
    if df is None:
        time.sleep(DEFAULT_SLEEP_INTERVAL)
        return

    # Check for entry signal
    signal, reason = check_entry_signal(df, state, config)

    if signal:
        logger.info(f"🎯 Signal detected: {signal} | {reason}")
        success = open_position(
            exchange, state, config, signal, reason,
            cache, circuit_breaker, metrics, df
        )
        if success:
            logger.info(f"✅ Position opened: {signal}")
    else:
        # Log status periodically
        if iteration % LOG_STATUS_INTERVAL == 0:
            _log_waiting_status(state, metrics, cache, exchange, config)

    time.sleep(DEFAULT_SLEEP_INTERVAL)


def _wait_for_candle_settle(config: Dict[str, Any]) -> None:
    """Wait for new candle data to settle."""
    now_ms = int(time.time() * 1000)
    candle_progress = now_ms % CANDLE_DURATION_MS
    time_since_close = candle_progress / 1000

    if time_since_close < CANDLE_SETTLE_SECONDS:
        wait_time = CANDLE_SETTLE_SECONDS - time_since_close
        logger.debug(f"Waiting {wait_time:.1f}s for candle to settle")
        time.sleep(wait_time)


def _fetch_and_calculate_indicators(
    exchange: ccxt.bingx,
    config: Dict[str, Any],
) -> Optional[pd.DataFrame]:
    """Fetch OHLCV data and calculate indicators."""
    try:
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

        return df

    except ccxt.NetworkError as e:
        logger.error(f"Failed to fetch indicators (network error): {e}")
        return None
    except ccxt.ExchangeError as e:
        logger.error(f"Failed to fetch indicators (exchange error): {e}")
        return None
    except Exception as e:
        logger.error(f"Failed to fetch/calculate indicators: {e}")
        return None


def _log_position_status(
    position: Dict[str, Any],
    config: Dict[str, Any],
    cache: APICache,
    exchange: ccxt.bingx,
) -> None:
    """Log current position status."""
    try:
        ticker = fetch_ticker_cached(exchange, config['symbol'], cache)
        current_price = ticker['last']
        direction = 1 if position['direction'] == 'LONG' else -1
        pnl_pct = direction * (current_price / position['entry_price'] - 1) * 100 * config['leverage']

        logger.info(
            f"📊 Position: {position['direction']} | "
            f"Entry: ${position['entry_price']:.1f} | "
            f"Current: ${current_price:.1f} | "
            f"PnL: {pnl_pct:+.2f}%"
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
        logger.debug(f"Health check failed: {e}")


# Entry point for direct execution
if __name__ == '__main__':
    run_bot()