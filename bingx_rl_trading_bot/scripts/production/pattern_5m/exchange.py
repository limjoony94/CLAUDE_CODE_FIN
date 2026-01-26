"""
Engulf 5m Bot - Exchange Interface
Exchange setup, API calls, and caching logic.
"""

import time
import logging
import yaml
import ccxt
from functools import wraps
from typing import Dict, List, Any, Optional, Callable, TypeVar

T = TypeVar('T')

from .constants import (
    API_KEYS_FILE,
    API_MAX_ATTEMPTS,
    API_BASE_DELAY,
    API_MAX_DELAY,
)
from .models import APICache, CircuitBreaker, PerformanceMetrics

logger = logging.getLogger('pattern_5m')


def create_exchange(api_keys_file: str = API_KEYS_FILE) -> ccxt.bingx:
    """
    Create and configure BingX exchange instance.

    Args:
        api_keys_file: Path to API keys YAML file

    Returns:
        Configured ccxt.bingx exchange instance
    """
    with open(api_keys_file, 'r') as f:
        api_config = yaml.safe_load(f)

    mainnet_config = api_config.get('bingx', {}).get('mainnet', {})

    exchange = ccxt.bingx({
        'apiKey': mainnet_config.get('api_key'),
        'secret': mainnet_config.get('secret_key'),
        'enableRateLimit': True,
        'options': {
            'defaultType': 'swap',
            'adjustForTimeDifference': True,
            'recvWindow': 60000,
        }
    })

    # Synchronize server time
    _sync_server_time(exchange)

    return exchange


def _sync_server_time(exchange: ccxt.bingx) -> None:
    """Synchronize local time with exchange server time."""
    try:
        exchange.load_time_difference()
        time_diff = exchange.options.get('timeDifference', 0)
        if time_diff != 0:
            logger.info(f"✅ Server time synchronized (offset: {time_diff}ms)")
            original_milliseconds = exchange.milliseconds

            def adjusted_milliseconds() -> int:
                return original_milliseconds() - time_diff

            exchange.milliseconds = adjusted_milliseconds
    except ccxt.NetworkError as e:
        logger.warning(f"⚠️ Could not sync server time (network error): {e}")
    except ccxt.ExchangeError as e:
        logger.warning(f"⚠️ Could not sync server time (exchange error): {e}")
    except Exception as e:
        logger.warning(f"⚠️ Could not sync server time: {e}")


def verify_position_mode(exchange: ccxt.bingx, config: Dict[str, Any]) -> bool:
    """
    Verify and set position mode (one-way or hedge).

    Args:
        exchange: Exchange instance
        config: Bot configuration

    Returns:
        True if position mode is correctly set
    """
    symbol = config['symbol']
    expected_mode = config.get('position_mode', 'one-way').lower()
    logger.info(f"Verifying position mode (expected: {expected_mode})...")

    try:
        positions = exchange.fetch_positions([symbol])
        has_long = any(p.get('side') == 'long' and float(p.get('contracts', 0)) > 0 for p in positions)
        has_short = any(p.get('side') == 'short' and float(p.get('contracts', 0)) > 0 for p in positions)

        if has_long and has_short:
            logger.error("🔴 CRITICAL: Both LONG and SHORT positions exist!")
            return False

        if expected_mode == 'one-way':
            try:
                exchange.set_position_mode(hedged=False, symbol=symbol)
                logger.info("✅ Position mode set to One-Way")
            except ccxt.ExchangeError as e:
                if 'No need to change' in str(e) or 'same' in str(e).lower():
                    logger.info("✅ Position mode already One-Way")

        return True
    except ccxt.NetworkError as e:
        logger.error(f"Failed to verify position mode (network error): {e}")
        return False
    except ccxt.ExchangeError as e:
        logger.error(f"Failed to verify position mode (exchange error): {e}")
        return False
    except Exception as e:
        logger.error(f"Failed to verify position mode: {e}")
        return False


def set_margin_mode(exchange: ccxt.bingx, config: Dict[str, Any]) -> bool:
    """
    Set margin mode (crossed or isolated).

    Args:
        exchange: Exchange instance
        config: Bot configuration

    Returns:
        True if margin mode is correctly set
    """
    symbol = config['symbol']
    margin_mode = config.get('margin_mode', 'crossed').upper()
    logger.info(f"Setting margin mode to {margin_mode}...")

    try:
        exchange.set_margin_mode(margin_mode, symbol)
        logger.info(f"✅ Margin mode set to {margin_mode}")
        return True
    except ccxt.ExchangeError as e:
        if 'No need to change' in str(e) or 'same' in str(e).lower():
            logger.info(f"✅ Margin mode already {margin_mode}")
            return True
        return False
    except ccxt.NetworkError as e:
        logger.warning(f"Failed to set margin mode (network error): {e}")
        return False
    except Exception as e:
        logger.warning(f"Failed to set margin mode: {e}")
        return False


# ============================================================
# API RETRY DECORATOR
# ============================================================

def api_retry(
    max_attempts: int = API_MAX_ATTEMPTS,
    base_delay: int = API_BASE_DELAY,
    max_delay: int = API_MAX_DELAY,
    circuit_breaker: Optional[CircuitBreaker] = None,
    metrics: Optional[PerformanceMetrics] = None,
) -> Callable:
    """
    Decorator for API calls with retry logic and circuit breaker.

    Args:
        max_attempts: Maximum retry attempts
        base_delay: Base delay between retries (seconds)
        max_delay: Maximum delay between retries (seconds)
        circuit_breaker: Optional CircuitBreaker instance
        metrics: Optional PerformanceMetrics instance

    Returns:
        Decorated function
    """
    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            # Check circuit breaker
            if circuit_breaker and not circuit_breaker.can_execute():
                wait_time = circuit_breaker.get_wait_time()
                logger.warning(f"Circuit breaker OPEN, waiting {wait_time:.1f}s")
                time.sleep(wait_time)

            last_exception = None
            start_time = time.time()

            for attempt in range(max_attempts):
                try:
                    result = func(*args, **kwargs)

                    # Record success
                    if metrics:
                        latency_ms = (time.time() - start_time) * 1000
                        metrics.update_api_latency(latency_ms)
                    if circuit_breaker:
                        circuit_breaker.record_success()

                    return result

                except ccxt.RateLimitExceeded as e:
                    delay = min(base_delay * (2 ** attempt), max_delay)
                    logger.warning(f"Rate limit exceeded, retrying in {delay}s (attempt {attempt + 1}/{max_attempts})")
                    time.sleep(delay)
                    last_exception = e
                    if circuit_breaker:
                        circuit_breaker.record_failure()

                except ccxt.NetworkError as e:
                    delay = min(base_delay * (2 ** attempt), max_delay)
                    logger.warning(f"Network error, retrying in {delay}s (attempt {attempt + 1}/{max_attempts}): {e}")
                    time.sleep(delay)
                    last_exception = e
                    if circuit_breaker:
                        circuit_breaker.record_failure()

                except ccxt.ExchangeError as e:
                    # Don't retry exchange errors (likely permanent)
                    if circuit_breaker:
                        circuit_breaker.record_failure()
                    raise e

                except Exception as e:
                    delay = min(base_delay * (2 ** attempt), max_delay)
                    logger.warning(f"API error, retrying in {delay}s (attempt {attempt + 1}/{max_attempts}): {e}")
                    time.sleep(delay)
                    last_exception = e
                    if circuit_breaker:
                        circuit_breaker.record_failure()

            raise last_exception

        return wrapper
    return decorator


# ============================================================
# CACHED API CALLS
# ============================================================

def fetch_ticker_cached(
    exchange: ccxt.bingx,
    symbol: str,
    cache: APICache,
    force_refresh: bool = False,
    circuit_breaker: Optional[CircuitBreaker] = None,
    metrics: Optional[PerformanceMetrics] = None,
) -> Dict[str, Any]:
    """
    Fetch ticker with caching support.

    Args:
        exchange: Exchange instance
        symbol: Trading symbol
        cache: APICache instance
        force_refresh: Force refresh even if cached
        circuit_breaker: Optional CircuitBreaker
        metrics: Optional PerformanceMetrics

    Returns:
        Ticker data dictionary
    """
    if not force_refresh:
        cached = cache.get_ticker()
        if cached:
            logger.debug("Using cached ticker")
            return cached

    @api_retry(circuit_breaker=circuit_breaker, metrics=metrics)
    def _fetch() -> Dict[str, Any]:
        return exchange.fetch_ticker(symbol)

    ticker = _fetch()
    cache.set_ticker(ticker)
    return ticker


def fetch_balance_cached(
    exchange: ccxt.bingx,
    cache: APICache,
    force_refresh: bool = False,
    circuit_breaker: Optional[CircuitBreaker] = None,
    metrics: Optional[PerformanceMetrics] = None,
) -> Dict[str, Any]:
    """
    Fetch balance with caching support.

    Args:
        exchange: Exchange instance
        cache: APICache instance
        force_refresh: Force refresh even if cached
        circuit_breaker: Optional CircuitBreaker
        metrics: Optional PerformanceMetrics

    Returns:
        Balance data dictionary
    """
    if not force_refresh:
        cached = cache.get_balance()
        if cached:
            logger.debug("Using cached balance")
            return cached

    @api_retry(circuit_breaker=circuit_breaker, metrics=metrics)
    def _fetch() -> Dict[str, Any]:
        return exchange.fetch_balance()

    balance = _fetch()
    cache.set_balance(balance)
    return balance


def fetch_positions_cached(
    exchange: ccxt.bingx,
    symbol: str,
    cache: APICache,
    force_refresh: bool = False,
    circuit_breaker: Optional[CircuitBreaker] = None,
    metrics: Optional[PerformanceMetrics] = None,
) -> List[Dict[str, Any]]:
    """
    Fetch positions with caching support.

    Args:
        exchange: Exchange instance
        symbol: Trading symbol
        cache: APICache instance
        force_refresh: Force refresh even if cached
        circuit_breaker: Optional CircuitBreaker
        metrics: Optional PerformanceMetrics

    Returns:
        List of position dictionaries
    """
    if not force_refresh:
        cached = cache.get_positions()
        if cached is not None:
            logger.debug("Using cached positions")
            return cached

    @api_retry(circuit_breaker=circuit_breaker, metrics=metrics)
    def _fetch() -> List[Dict[str, Any]]:
        return exchange.fetch_positions([symbol])

    positions = _fetch()
    cache.set_positions(positions)
    return positions


def fetch_ohlcv(
    exchange: ccxt.bingx,
    symbol: str,
    timeframe: str,
    limit: int = 100,
) -> List[List[Any]]:
    """
    Fetch OHLCV candlestick data.

    Args:
        exchange: Exchange instance
        symbol: Trading symbol
        timeframe: Candle timeframe (e.g., '5m')
        limit: Number of candles to fetch

    Returns:
        List of OHLCV data (each item: [timestamp, open, high, low, close, volume])
    """
    return exchange.fetch_ohlcv(symbol, timeframe, limit=limit)


# ============================================================
# HEALTH CHECK
# ============================================================

def health_check(
    exchange: ccxt.bingx,
    config: Dict[str, Any],
    cache: APICache,
    circuit_breaker: CircuitBreaker,
    metrics: PerformanceMetrics,
    state_file: str,
) -> Dict[str, Any]:
    """
    Comprehensive health check for bot components.

    Args:
        exchange: Exchange instance
        config: Bot configuration
        cache: APICache instance
        circuit_breaker: CircuitBreaker instance
        metrics: PerformanceMetrics instance
        state_file: Path to state file

    Returns:
        Health status dictionary
    """
    import os
    from .constants import STATE_STALE_THRESHOLD_SECONDS

    health = {
        'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'status': 'healthy',
        'checks': {}
    }

    # Check API connectivity
    try:
        start = time.time()
        ticker = fetch_ticker_cached(exchange, config['symbol'], cache, force_refresh=True)
        latency = (time.time() - start) * 1000
        health['checks']['api_connectivity'] = {
            'status': 'ok',
            'latency_ms': round(latency, 1),
            'price': ticker['last']
        }
    except ccxt.NetworkError as e:
        health['checks']['api_connectivity'] = {'status': 'error', 'message': f'Network: {e}'}
        health['status'] = 'degraded'
    except ccxt.ExchangeError as e:
        health['checks']['api_connectivity'] = {'status': 'error', 'message': f'Exchange: {e}'}
        health['status'] = 'degraded'
    except Exception as e:
        health['checks']['api_connectivity'] = {'status': 'error', 'message': str(e)}
        health['status'] = 'degraded'

    # Check circuit breaker
    health['checks']['circuit_breaker'] = {
        'status': 'ok' if not circuit_breaker.is_open else 'tripped',
        'failure_count': circuit_breaker.failure_count,
        'is_open': circuit_breaker.is_open
    }
    if circuit_breaker.is_open:
        health['status'] = 'degraded'

    # Check metrics
    health['checks']['metrics'] = {
        'total_trades': metrics.total_trades,
        'win_rate': round(metrics.actual_win_rate, 1),
        'total_pnl': round(metrics.total_pnl_pct, 2),
        'avg_latency_ms': round(metrics.avg_api_latency_ms, 1)
    }

    # Check state file
    try:
        if os.path.exists(state_file):
            mtime = os.path.getmtime(state_file)
            age_seconds = time.time() - mtime
            health['checks']['state_file'] = {
                'status': 'ok' if age_seconds < STATE_STALE_THRESHOLD_SECONDS else 'stale',
                'age_seconds': round(age_seconds, 1)
            }
        else:
            health['checks']['state_file'] = {'status': 'missing'}
    except (IOError, OSError) as e:
        health['checks']['state_file'] = {'status': 'error', 'message': f'I/O error: {e}'}
    except Exception as e:
        health['checks']['state_file'] = {'status': 'error', 'message': str(e)}

    return health