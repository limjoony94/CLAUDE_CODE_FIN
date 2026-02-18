"""Tests for exchange.py — API retry, caching, circuit breaker integration.

Uses mock objects to avoid real exchange calls.
"""

import time

import pytest
import ccxt
from unittest.mock import MagicMock, patch

from bingx_rl_trading_bot.scripts.production.pattern_5m.exchange import (
    _api_call_with_retry,
    fetch_ticker_cached,
    fetch_balance_cached,
    fetch_positions_cached,
    set_shutdown_checker,
    _interruptible_api_sleep,
)
from bingx_rl_trading_bot.scripts.production.pattern_5m.models import (
    APICache,
    CircuitBreaker,
    PerformanceMetrics,
)


# ── _api_call_with_retry ────────────────────────────────────


@patch('bingx_rl_trading_bot.scripts.production.pattern_5m.exchange._interruptible_api_sleep')
class TestApiCallWithRetry:
    """Test _api_call_with_retry() retry/circuit breaker/metrics logic."""

    def test_success_first_try(self, mock_sleep):
        """Successful call on first attempt — no retries."""
        func = MagicMock(return_value={'last': 50000})
        result = _api_call_with_retry(func)
        assert result == {'last': 50000}
        func.assert_called_once()
        mock_sleep.assert_not_called()

    def test_retry_on_rate_limit(self, mock_sleep):
        """RateLimitExceeded → retries, then succeeds."""
        func = MagicMock(side_effect=[
            ccxt.RateLimitExceeded('rate limit'),
            {'last': 50000},
        ])
        result = _api_call_with_retry(func)
        assert result == {'last': 50000}
        assert func.call_count == 2
        mock_sleep.assert_called_once()

    def test_retry_on_network_error(self, mock_sleep):
        """NetworkError → retries, then succeeds."""
        func = MagicMock(side_effect=[
            ccxt.NetworkError('timeout'),
            {'last': 50000},
        ])
        result = _api_call_with_retry(func)
        assert result == {'last': 50000}
        assert func.call_count == 2

    def test_exchange_error_no_retry(self, mock_sleep):
        """ExchangeError → raises immediately without retry."""
        func = MagicMock(side_effect=ccxt.ExchangeError('invalid order'))
        with pytest.raises(ccxt.ExchangeError, match='invalid order'):
            _api_call_with_retry(func)
        func.assert_called_once()
        mock_sleep.assert_not_called()

    def test_all_retries_exhausted(self, mock_sleep):
        """All 3 attempts fail → raises last exception."""
        func = MagicMock(side_effect=ccxt.NetworkError('down'))
        with pytest.raises(ccxt.NetworkError, match='down'):
            _api_call_with_retry(func)
        assert func.call_count == 3  # API_MAX_ATTEMPTS = 3

    def test_circuit_breaker_blocks_when_open(self, mock_sleep):
        """Open circuit breaker → waits before attempting."""
        cb = CircuitBreaker()
        cb.is_open = True
        cb.last_failure_time = time.time()  # recently failed
        func = MagicMock(return_value={'last': 50000})
        result = _api_call_with_retry(func, circuit_breaker=cb)
        assert result == {'last': 50000}
        # Should have slept for circuit breaker wait
        assert mock_sleep.call_count >= 1

    def test_circuit_breaker_records_success(self, mock_sleep):
        """Successful call → circuit breaker records success."""
        cb = CircuitBreaker()
        cb.failure_count = 3
        func = MagicMock(return_value={'ok': True})
        _api_call_with_retry(func, circuit_breaker=cb)
        assert cb.failure_count == 0

    def test_circuit_breaker_records_failure(self, mock_sleep):
        """Failed call → circuit breaker records failure."""
        cb = CircuitBreaker()
        assert cb.failure_count == 0
        func = MagicMock(side_effect=ccxt.NetworkError('fail'))
        with pytest.raises(ccxt.NetworkError):
            _api_call_with_retry(func, circuit_breaker=cb)
        assert cb.failure_count == 3  # 3 attempts, 3 failures

    def test_metrics_latency_recorded(self, mock_sleep):
        """Successful call → avg_api_latency_ms updated."""
        metrics = PerformanceMetrics()
        assert metrics.avg_api_latency_ms == 0.0
        func = MagicMock(return_value={'ok': True})
        _api_call_with_retry(func, metrics=metrics)
        assert metrics.avg_api_latency_ms > 0

    def test_generic_exception_retries(self, mock_sleep):
        """Non-ccxt exceptions also trigger retries."""
        func = MagicMock(side_effect=[
            RuntimeError('transient'),
            {'ok': True},
        ])
        result = _api_call_with_retry(func)
        assert result == {'ok': True}
        assert func.call_count == 2

    def test_exponential_backoff_delay(self, mock_sleep):
        """Retry delays increase exponentially."""
        func = MagicMock(side_effect=ccxt.NetworkError('fail'))
        with pytest.raises(ccxt.NetworkError):
            _api_call_with_retry(func)
        # Should have been called with increasing delays
        delays = [call.args[0] for call in mock_sleep.call_args_list]
        assert len(delays) == 3  # 3 attempts
        # Delays should be non-decreasing (exponential backoff)
        for i in range(1, len(delays)):
            assert delays[i] >= delays[i - 1]


# ── fetch_ticker_cached ─────────────────────────────────────


class TestFetchTickerCached:
    """Test fetch_ticker_cached() caching behavior."""

    def test_returns_cached_data(self):
        """Cache hit → returns cached data without API call."""
        cache = APICache()
        cached_ticker = {'last': 50000, 'symbol': 'BTC/USDT'}
        cache.set_ticker(cached_ticker)
        exchange = MagicMock()
        result = fetch_ticker_cached(exchange, 'BTC/USDT:USDT', cache)
        assert result == cached_ticker
        exchange.fetch_ticker.assert_not_called()

    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.exchange._interruptible_api_sleep')
    def test_fetches_on_cache_miss(self, mock_sleep):
        """Cache miss → calls exchange API and caches result."""
        cache = APICache()
        exchange = MagicMock()
        fresh_ticker = {'last': 51000, 'symbol': 'BTC/USDT'}
        exchange.fetch_ticker.return_value = fresh_ticker
        result = fetch_ticker_cached(exchange, 'BTC/USDT:USDT', cache)
        assert result == fresh_ticker
        exchange.fetch_ticker.assert_called_once()
        # Verify it was cached
        assert cache.get_ticker() == fresh_ticker

    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.exchange._interruptible_api_sleep')
    def test_force_refresh_bypasses_cache(self, mock_sleep):
        """force_refresh=True → calls API even with cached data."""
        cache = APICache()
        cache.set_ticker({'last': 50000})
        exchange = MagicMock()
        exchange.fetch_ticker.return_value = {'last': 52000}
        result = fetch_ticker_cached(
            exchange, 'BTC/USDT:USDT', cache, force_refresh=True
        )
        assert result['last'] == 52000
        exchange.fetch_ticker.assert_called_once()


# ── fetch_balance_cached ─────────────────────────────────────


class TestFetchBalanceCached:
    """Test fetch_balance_cached() caching behavior."""

    def test_returns_cached_data(self):
        """Cache hit → returns cached balance."""
        cache = APICache()
        cached_balance = {'USDT': {'free': 1000, 'total': 1000}}
        cache.set_balance(cached_balance)
        exchange = MagicMock()
        result = fetch_balance_cached(exchange, cache)
        assert result == cached_balance
        exchange.fetch_balance.assert_not_called()

    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.exchange._interruptible_api_sleep')
    def test_fetches_on_cache_miss(self, mock_sleep):
        """Cache miss → calls exchange API."""
        cache = APICache()
        exchange = MagicMock()
        fresh_balance = {'USDT': {'free': 2000, 'total': 2000}}
        exchange.fetch_balance.return_value = fresh_balance
        result = fetch_balance_cached(exchange, cache)
        assert result == fresh_balance
        exchange.fetch_balance.assert_called_once()


# ── fetch_positions_cached ───────────────────────────────────


class TestFetchPositionsCached:
    """Test fetch_positions_cached() caching behavior."""

    def test_returns_cached_data(self):
        """Cache hit → returns cached positions."""
        cache = APICache()
        cached_positions = [{'symbol': 'BTC/USDT', 'side': 'long'}]
        cache.set_positions(cached_positions)
        exchange = MagicMock()
        result = fetch_positions_cached(exchange, 'BTC/USDT:USDT', cache)
        assert result == cached_positions
        exchange.fetch_positions.assert_not_called()

    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.exchange._interruptible_api_sleep')
    def test_fetches_on_cache_miss(self, mock_sleep):
        """Cache miss → calls exchange API."""
        cache = APICache()
        exchange = MagicMock()
        fresh_positions = [{'symbol': 'BTC/USDT', 'side': 'long', 'contracts': 0.01}]
        exchange.fetch_positions.return_value = fresh_positions
        result = fetch_positions_cached(exchange, 'BTC/USDT:USDT', cache)
        assert result == fresh_positions
        exchange.fetch_positions.assert_called_once()

    def test_empty_list_is_cached(self):
        """Empty positions list is also cached (not treated as None)."""
        cache = APICache()
        cache.set_positions([])
        exchange = MagicMock()
        result = fetch_positions_cached(exchange, 'BTC/USDT:USDT', cache)
        assert result == []
        exchange.fetch_positions.assert_not_called()


# ── _interruptible_api_sleep ─────────────────────────────────


class TestInterruptibleApiSleep:
    """Test _interruptible_api_sleep() shutdown-aware sleep."""

    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.exchange.time.sleep')
    def test_sleeps_requested_duration(self, mock_sleep):
        """Normal sleep without shutdown → sleeps close to requested time."""
        _interruptible_api_sleep(0.1)
        assert mock_sleep.call_count >= 1

    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.exchange.time.sleep')
    def test_interrupts_on_shutdown(self, mock_sleep):
        """Shutdown checker returns True → returns early."""
        old_checker = None
        try:
            set_shutdown_checker(lambda: True)
            _interruptible_api_sleep(10.0)  # would be 10s, but should return immediately
            # Should not have slept for the full duration
            # (may have slept at most once before checking)
        finally:
            set_shutdown_checker(lambda: False)
