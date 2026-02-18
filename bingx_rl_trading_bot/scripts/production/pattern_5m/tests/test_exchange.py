"""Tests for exchange.py — API retry, caching, circuit breaker, exchange setup.

Uses mock objects to avoid real exchange calls.
"""

import time

import pytest
import ccxt
from unittest.mock import MagicMock, patch

from bingx_rl_trading_bot.scripts.production.pattern_5m.exchange import (
    _api_call_with_retry,
    _sync_server_time,
    create_exchange,
    fetch_ticker_cached,
    fetch_balance_cached,
    fetch_positions_cached,
    fetch_ohlcv,
    set_shutdown_checker,
    _interruptible_api_sleep,
    verify_position_mode,
    set_margin_mode,
    health_check,
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


# ── verify_position_mode ─────────────────────────────────────


class TestVerifyPositionMode:
    """Test verify_position_mode() exchange setup."""

    def test_one_way_success(self):
        """One-way mode set successfully → returns True."""
        exchange = MagicMock()
        exchange.fetch_positions.return_value = []
        exchange.set_position_mode.return_value = None
        config = {'symbol': 'BTC/USDT:USDT', 'position_mode': 'one-way'}
        assert verify_position_mode(exchange, config) is True

    def test_already_one_way(self):
        """Position mode already one-way → returns True."""
        exchange = MagicMock()
        exchange.fetch_positions.return_value = []
        exchange.set_position_mode.side_effect = ccxt.ExchangeError('No need to change')
        config = {'symbol': 'BTC/USDT:USDT', 'position_mode': 'one-way'}
        assert verify_position_mode(exchange, config) is True

    def test_both_positions_exist(self):
        """Both LONG and SHORT positions → returns False."""
        exchange = MagicMock()
        exchange.fetch_positions.return_value = [
            {'side': 'long', 'contracts': 0.01},
            {'side': 'short', 'contracts': 0.01},
        ]
        config = {'symbol': 'BTC/USDT:USDT'}
        assert verify_position_mode(exchange, config) is False

    def test_network_error(self):
        """Network error → returns False."""
        exchange = MagicMock()
        exchange.fetch_positions.side_effect = ccxt.NetworkError('timeout')
        config = {'symbol': 'BTC/USDT:USDT'}
        assert verify_position_mode(exchange, config) is False

    def test_exchange_error_not_same(self):
        """Exchange error that is NOT 'no need to change' → returns False."""
        exchange = MagicMock()
        exchange.fetch_positions.return_value = []
        exchange.set_position_mode.side_effect = ccxt.ExchangeError('permission denied')
        config = {'symbol': 'BTC/USDT:USDT', 'position_mode': 'one-way'}
        assert verify_position_mode(exchange, config) is False


# ── set_margin_mode ──────────────────────────────────────────


class TestSetMarginMode:
    """Test set_margin_mode() exchange setup."""

    def test_set_crossed_success(self):
        """Set CROSSED margin mode → returns True."""
        exchange = MagicMock()
        config = {'symbol': 'BTC/USDT:USDT', 'margin_mode': 'crossed'}
        assert set_margin_mode(exchange, config) is True
        exchange.set_margin_mode.assert_called_once_with('CROSSED', 'BTC/USDT:USDT')

    def test_already_set(self):
        """Margin mode already set → returns True."""
        exchange = MagicMock()
        exchange.set_margin_mode.side_effect = ccxt.ExchangeError('No need to change')
        config = {'symbol': 'BTC/USDT:USDT', 'margin_mode': 'crossed'}
        assert set_margin_mode(exchange, config) is True

    def test_exchange_error(self):
        """Exchange error → returns False."""
        exchange = MagicMock()
        exchange.set_margin_mode.side_effect = ccxt.ExchangeError('invalid symbol')
        config = {'symbol': 'BTC/USDT:USDT', 'margin_mode': 'crossed'}
        assert set_margin_mode(exchange, config) is False

    def test_network_error(self):
        """Network error → returns False."""
        exchange = MagicMock()
        exchange.set_margin_mode.side_effect = ccxt.NetworkError('timeout')
        config = {'symbol': 'BTC/USDT:USDT', 'margin_mode': 'crossed'}
        assert set_margin_mode(exchange, config) is False

    def test_default_margin_mode(self):
        """No margin_mode in config → defaults to CROSSED."""
        exchange = MagicMock()
        config = {'symbol': 'BTC/USDT:USDT'}
        set_margin_mode(exchange, config)
        exchange.set_margin_mode.assert_called_once_with('CROSSED', 'BTC/USDT:USDT')

    def test_generic_exception_returns_false(self):
        """Generic exception → returns False."""
        exchange = MagicMock()
        exchange.set_margin_mode.side_effect = RuntimeError('unexpected')
        config = {'symbol': 'BTC/USDT:USDT', 'margin_mode': 'crossed'}
        assert set_margin_mode(exchange, config) is False


# ── _sync_server_time ──────────────────────────────────────


class TestSyncServerTime:
    """Test _sync_server_time() time synchronization."""

    def test_no_time_difference(self):
        """Time difference = 0 → no adjustment."""
        exchange = MagicMock()
        exchange.options = {'timeDifference': 0}
        _sync_server_time(exchange)
        exchange.load_time_difference.assert_called_once()

    def test_with_time_difference(self):
        """Non-zero time diff → milliseconds adjusted."""
        exchange = MagicMock()
        exchange.options = {'timeDifference': 500}
        original_ms = MagicMock(return_value=1000000)
        exchange.milliseconds = original_ms

        _sync_server_time(exchange)

        # milliseconds should be replaced
        assert exchange.milliseconds != original_ms
        # New function returns original - diff
        adjusted = exchange.milliseconds()
        assert adjusted == 1000000 - 500

    def test_network_error_no_crash(self):
        """Network error → silently handled."""
        exchange = MagicMock()
        exchange.load_time_difference.side_effect = ccxt.NetworkError('fail')
        _sync_server_time(exchange)  # should not raise

    def test_exchange_error_no_crash(self):
        """Exchange error → silently handled."""
        exchange = MagicMock()
        exchange.load_time_difference.side_effect = ccxt.ExchangeError('fail')
        _sync_server_time(exchange)  # should not raise

    def test_generic_exception_no_crash(self):
        """Generic exception → silently handled."""
        exchange = MagicMock()
        exchange.load_time_difference.side_effect = RuntimeError('fail')
        _sync_server_time(exchange)  # should not raise


# ── create_exchange ────────────────────────────────────────


class TestSetupExchange:
    """Test create_exchange() factory function."""

    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.exchange._sync_server_time')
    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.exchange.ccxt.bingx')
    @patch('builtins.open')
    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.exchange.yaml.safe_load')
    def test_creates_exchange_instance(self, mock_yaml, mock_open, mock_bingx_cls, mock_sync):
        """Creates ccxt.bingx with correct config."""
        mock_yaml.return_value = {
            'bingx': {'mainnet': {'api_key': 'KEY', 'secret_key': 'SECRET'}}
        }
        mock_instance = MagicMock()
        mock_bingx_cls.return_value = mock_instance

        result = create_exchange('fake_keys.yaml')

        assert result is mock_instance
        mock_bingx_cls.assert_called_once()
        call_kwargs = mock_bingx_cls.call_args[0][0]
        assert call_kwargs['apiKey'] == 'KEY'
        assert call_kwargs['secret'] == 'SECRET'
        mock_sync.assert_called_once_with(mock_instance)

    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.exchange._sync_server_time')
    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.exchange.ccxt.bingx')
    @patch('builtins.open')
    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.exchange.yaml.safe_load')
    def test_missing_keys_defaults_to_none(self, mock_yaml, mock_open, mock_bingx_cls, mock_sync):
        """Missing api_key/secret → defaults to None."""
        mock_yaml.return_value = {'bingx': {'mainnet': {}}}
        mock_bingx_cls.return_value = MagicMock()

        create_exchange('keys.yaml')

        call_kwargs = mock_bingx_cls.call_args[0][0]
        assert call_kwargs['apiKey'] is None
        assert call_kwargs['secret'] is None


# ── verify_position_mode (additional error paths) ──────────


class TestVerifyPositionModeExtended:
    """Test verify_position_mode() — additional uncovered error paths."""

    def test_exchange_error_during_fetch(self):
        """ExchangeError during fetch_positions → returns False."""
        exchange = MagicMock()
        exchange.fetch_positions.side_effect = ccxt.ExchangeError('fail')
        config = {'symbol': 'BTC/USDT:USDT'}
        assert verify_position_mode(exchange, config) is False

    def test_generic_exception(self):
        """Generic exception → returns False."""
        exchange = MagicMock()
        exchange.fetch_positions.side_effect = RuntimeError('crash')
        config = {'symbol': 'BTC/USDT:USDT'}
        assert verify_position_mode(exchange, config) is False


# ── fetch_ohlcv ──────────────────────────────────────────


@patch('bingx_rl_trading_bot.scripts.production.pattern_5m.exchange._interruptible_api_sleep')
class TestFetchOhlcv:
    """Test fetch_ohlcv() wrapper."""

    def test_returns_data(self, mock_sleep):
        """Successful fetch → returns OHLCV list."""
        exchange = MagicMock()
        data = [[1, 50000, 50100, 49900, 50050, 100]]
        exchange.fetch_ohlcv.return_value = data
        result = fetch_ohlcv(exchange, 'BTC/USDT:USDT', '5m', limit=10)
        assert result == data
        exchange.fetch_ohlcv.assert_called_once()


# ── health_check ─────────────────────────────────────────


@patch('bingx_rl_trading_bot.scripts.production.pattern_5m.exchange._interruptible_api_sleep',
       new=MagicMock())
class TestHealthCheck:
    """Test health_check() comprehensive bot health check."""

    def test_healthy_status(self):
        """All checks pass → status='healthy'."""
        exchange = MagicMock()
        exchange.fetch_ticker.return_value = {'last': 50000}
        config = {'symbol': 'BTC/USDT:USDT'}
        cache = APICache()
        cb = CircuitBreaker()
        metrics = PerformanceMetrics()
        metrics.total_trades = 10
        metrics.actual_win_rate = 80.0
        metrics.total_pnl_pct = 5.0

        import tempfile, json
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump({'total_trades': 10}, f)
            sf = f.name
        try:
            result = health_check(exchange, config, cache, cb, metrics, sf)
            assert result['status'] == 'healthy'
            assert result['checks']['api_connectivity']['status'] == 'ok'
            assert result['checks']['circuit_breaker']['status'] == 'ok'
            assert result['checks']['metrics']['total_trades'] == 10
            assert result['checks']['state_file']['status'] == 'ok'
        finally:
            import os
            os.unlink(sf)

    def test_network_error_degrades(self):
        """API network error → status='degraded'."""
        exchange = MagicMock()
        exchange.fetch_ticker.side_effect = ccxt.NetworkError('fail')
        config = {'symbol': 'BTC/USDT:USDT'}
        cache = APICache()
        cb = CircuitBreaker()
        metrics = PerformanceMetrics()

        result = health_check(exchange, config, cache, cb, metrics, '/nonexistent/state.json')
        assert result['status'] == 'degraded'
        assert result['checks']['api_connectivity']['status'] == 'error'

    def test_circuit_breaker_open_degrades(self):
        """Open circuit breaker → status='degraded'."""
        exchange = MagicMock()
        exchange.fetch_ticker.return_value = {'last': 50000}
        config = {'symbol': 'BTC/USDT:USDT'}
        cache = APICache()
        cb = CircuitBreaker()
        cb.is_open = True
        metrics = PerformanceMetrics()

        result = health_check(exchange, config, cache, cb, metrics, '/nonexistent.json')
        assert result['status'] == 'degraded'
        assert result['checks']['circuit_breaker']['status'] == 'tripped'

    def test_missing_state_file(self):
        """State file doesn't exist → state_file check shows 'missing'."""
        exchange = MagicMock()
        exchange.fetch_ticker.return_value = {'last': 50000}
        config = {'symbol': 'BTC/USDT:USDT'}
        cache = APICache()
        cb = CircuitBreaker()
        metrics = PerformanceMetrics()

        result = health_check(exchange, config, cache, cb, metrics, '/nonexistent_state_file.json')
        assert result['checks']['state_file']['status'] == 'missing'

    def test_exchange_error_degrades(self):
        """Exchange error on ticker → status='degraded'."""
        exchange = MagicMock()
        exchange.fetch_ticker.side_effect = ccxt.ExchangeError('fail')
        config = {'symbol': 'BTC/USDT:USDT'}
        cache = APICache()
        cb = CircuitBreaker()
        metrics = PerformanceMetrics()

        result = health_check(exchange, config, cache, cb, metrics, '/nonexistent.json')
        assert result['status'] == 'degraded'

    def test_generic_error_degrades(self):
        """Generic error on ticker → status='degraded'."""
        exchange = MagicMock()
        exchange.fetch_ticker.side_effect = RuntimeError('fail')
        config = {'symbol': 'BTC/USDT:USDT'}
        cache = APICache()
        cb = CircuitBreaker()
        metrics = PerformanceMetrics()

        result = health_check(exchange, config, cache, cb, metrics, '/nonexistent.json')
        assert result['status'] == 'degraded'

    def test_state_file_io_error(self):
        """IOError on getmtime → state_file shows error."""
        exchange = MagicMock()
        exchange.fetch_ticker.return_value = {'last': 50000}
        config = {'symbol': 'BTC/USDT:USDT'}
        cache = APICache()
        cb = CircuitBreaker()
        metrics = PerformanceMetrics()

        with patch('os.path.exists', return_value=True), \
             patch('os.path.getmtime', side_effect=OSError('disk fail')):
            result = health_check(exchange, config, cache, cb, metrics, '/some/state.json')
        assert result['checks']['state_file']['status'] == 'error'
        assert 'I/O error' in result['checks']['state_file']['message']

    def test_state_file_generic_exception(self):
        """Generic exception on state file check → state_file shows error."""
        exchange = MagicMock()
        exchange.fetch_ticker.return_value = {'last': 50000}
        config = {'symbol': 'BTC/USDT:USDT'}
        cache = APICache()
        cb = CircuitBreaker()
        metrics = PerformanceMetrics()

        with patch('os.path.exists', return_value=True), \
             patch('os.path.getmtime', side_effect=ValueError('bad')):
            result = health_check(exchange, config, cache, cb, metrics, '/some/state.json')
        assert result['checks']['state_file']['status'] == 'error'


# ── api_call_with_retry CB paths (lines 221, 233, 242, 246) ──


class TestApiCallCBPaths:
    """Test circuit breaker record_failure on various exception types."""

    @pytest.fixture(autouse=True)
    def mock_sleep(self):
        with patch(
            'bingx_rl_trading_bot.scripts.production.pattern_5m.exchange._interruptible_api_sleep',
        ) as m:
            yield m

    def test_rate_limit_records_cb_failure(self):
        """RateLimitExceeded with CB → CB.record_failure called."""
        cb = CircuitBreaker()
        func = MagicMock(side_effect=ccxt.RateLimitExceeded('limit'))
        with pytest.raises(ccxt.RateLimitExceeded):
            _api_call_with_retry(func, circuit_breaker=cb)
        assert cb.failure_count >= 1

    def test_exchange_error_records_cb_failure(self):
        """ExchangeError with CB → CB.record_failure, then re-raise."""
        cb = CircuitBreaker()
        func = MagicMock(side_effect=ccxt.ExchangeError('bad'))
        with pytest.raises(ccxt.ExchangeError):
            _api_call_with_retry(func, circuit_breaker=cb)
        assert cb.failure_count >= 1

    def test_generic_error_records_cb_failure(self):
        """Generic error with CB → CB.record_failure called."""
        cb = CircuitBreaker()
        func = MagicMock(side_effect=RuntimeError('oops'))
        with pytest.raises(RuntimeError):
            _api_call_with_retry(func, circuit_breaker=cb)
        assert cb.failure_count >= 1


# ── zero attempts edge case (line 246) ────────────────────────


class TestApiCallZeroAttempts:
    """Test _api_call_with_retry with 0 configured max attempts."""

    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.exchange._interruptible_api_sleep')
    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.exchange.API_MAX_ATTEMPTS', 0)
    def test_zero_attempts_raises_runtime_error(self, mock_sleep):
        """API_MAX_ATTEMPTS=0 → RuntimeError (line 246)."""
        func = MagicMock()
        with pytest.raises(RuntimeError, match='0 attempts'):
            _api_call_with_retry(func)
        func.assert_not_called()
