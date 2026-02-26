"""Tests for bot.py — Early Exit triggers, main loop logic with exchange mock."""

import pytest
import time
import pandas as pd
import numpy as np
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime

from bingx_rl_trading_bot.scripts.production.pattern_5m.signals import check_early_exit_signal
from bingx_rl_trading_bot.scripts.production.pattern_5m.constants import (
    CandleType,
    CANDLE_DURATION_MS,
    TRADING_WINDOW_SECONDS,
    CANDLE_SETTLE_SECONDS,
    POSITION_MONITOR_INTERVAL,
    MAX_MAINTENANCE_SLEEP,
)
from bingx_rl_trading_bot.scripts.production.pattern_5m.models import PerformanceMetrics
from bingx_rl_trading_bot.scripts.production.pattern_5m.bot import (
    _get_candle_timing,
    _interruptible_sleep,
    _calculate_sleep_duration,
    _should_sync_now,
    _verify_exchange_settings,
    _process_entry_signal,
    _process_existing_positions,
    _fetch_and_calculate_indicators,
    _log_position_status,
    _log_waiting_status,
    _maybe_run_health_check,
    _maybe_sync_position,
    _wait_for_candle_settle,
    _log_pattern_summary,
    _run_health_check,
    _run_bot_main,
    run_bot,
    signal_handler,
    CandleTiming,
)
from bingx_rl_trading_bot.scripts.production.pattern_5m.models import APICache, CircuitBreaker


# ── Helper Fixtures ───────────────────────────────────────────

@pytest.fixture
def make_df_with_types():
    """Factory to create DataFrame with candle types."""
    def _make(types: list, n_bars: int = None):
        """
        Args:
            types: List of CandleType enum values (e.g., [CandleType.BD, CandleType.BD])
            n_bars: Total bars (will pad with MED_UP if needed)
        """
        if n_bars and len(types) < n_bars:
            types = types + [CandleType.MED_UP] * (n_bars - len(types))

        # Create realistic OHLCV data
        n = len(types)
        base_price = 50000
        opens = [base_price + i * 10 for i in range(n)]
        closes = opens.copy()
        highs = [o + 50 for o in opens]
        lows = [o - 50 for o in opens]

        timestamps = pd.date_range('2025-01-01', periods=n, freq='5min')
        df = pd.DataFrame({
            'open': opens,
            'high': highs,
            'low': lows,
            'close': closes,
            'candle_type': types,
            'type_code': [t.value for t in types],
            'timestamp': (timestamps.astype('int64') // 10**6).tolist()  # Unix ms
        })
        return df

    return _make


@pytest.fixture
def long_position():
    """Create a sample LONG position."""
    return {
        'symbol': 'BTC-USDT',
        'direction': 'LONG',
        'entry_price': 50000.0,
        'quantity': 0.01,
        'entry_time': datetime.now().isoformat(),
        'early_exit_candle_count': {}
    }


@pytest.fixture
def short_position():
    """Create a sample SHORT position."""
    return {
        'symbol': 'BTC-USDT',
        'direction': 'SHORT',
        'entry_price': 50000.0,
        'quantity': 0.01,
        'entry_time': datetime.now().isoformat(),
        'early_exit_candle_count': {}
    }


@pytest.fixture
def default_config():
    """Create default bot configuration."""
    return {
        'symbol': 'BTC-USDT',
        'timeframe': '5m',
        'leverage': 3,
        'strategy': {
            'early_exit': {
                'enabled': True,
                'bearish_types': ['BD'],
                'bullish_types': ['BU'],
                'confirm_candles': 3,
                'min_profit_pct': 0.3
            }
        }
    }


# ── Early Exit Signal Tests ───────────────────────────────────

class TestEarlyExitSignal:
    """Test check_early_exit_signal() with various candle sequences."""

    def test_early_exit_not_enabled(self, make_df_with_types, long_position, default_config):
        """Early exit disabled → should always return (False, None)."""
        config = default_config.copy()
        config['strategy']['early_exit']['enabled'] = False

        df = make_df_with_types([CandleType.BIG_DOWN, CandleType.BIG_DOWN, CandleType.BIG_DOWN])
        current_price = 50500.0  # 1% profit

        should_exit, _, _, _ = check_early_exit_signal(long_position, df, current_price, config)

        assert should_exit is False

    def test_long_early_exit_3bd_with_profit(self, make_df_with_types, long_position, default_config):
        """LONG: 3 consecutive BD + profit >= 0.3% → should exit."""
        df = make_df_with_types([CandleType.BIG_DOWN, CandleType.BIG_DOWN, CandleType.BIG_DOWN])
        current_price = 50200.0  # +0.4% profit (above 0.3%)

        # Simulate 2 previous BD candles already seen
        long_position['reversal_count'] = 2

        should_exit, _, _, _ = check_early_exit_signal(long_position, df, current_price, default_config)

        assert should_exit is True

    def test_long_early_exit_3bd_no_profit(self, make_df_with_types, long_position, default_config):
        """LONG: 3 consecutive BD but profit < 0.3% → should NOT exit."""
        df = make_df_with_types([CandleType.BIG_DOWN, CandleType.BIG_DOWN, CandleType.BIG_DOWN])
        current_price = 50100.0  # +0.2% profit (below 0.3%)

        should_exit, _, _, _ = check_early_exit_signal(long_position, df, current_price, default_config)

        assert should_exit is False

    def test_long_early_exit_only_2bd(self, make_df_with_types, long_position, default_config):
        """LONG: Only 2 BD (< confirm_candles=3) → should NOT exit."""
        df = make_df_with_types([CandleType.BIG_DOWN, CandleType.BIG_DOWN, CandleType.MED_UP])
        current_price = 50500.0  # high profit

        should_exit, _, _, _ = check_early_exit_signal(long_position, df, current_price, default_config)

        assert should_exit is False

    def test_long_early_exit_interrupted_sequence(self, make_df_with_types, long_position, default_config):
        """LONG: BD sequence interrupted by non-BD → should NOT exit."""
        df = make_df_with_types([CandleType.BIG_DOWN, CandleType.MED_UP, CandleType.BIG_DOWN, CandleType.BIG_DOWN])
        current_price = 50500.0

        should_exit, _, _, _ = check_early_exit_signal(long_position, df, current_price, default_config)

        assert should_exit is False

    def test_short_early_exit_3bu_with_profit(self, make_df_with_types, short_position, default_config):
        """SHORT: 3 consecutive BU + profit >= 0.3% → should exit."""
        df = make_df_with_types([CandleType.BIG_UP, CandleType.BIG_UP, CandleType.BIG_UP])
        current_price = 49800.0  # +0.4% profit (entry=50000, current lower)

        # Simulate 2 previous BU candles already seen
        short_position['reversal_count'] = 2

        should_exit, _, _, _ = check_early_exit_signal(short_position, df, current_price, default_config)

        assert should_exit is True

    def test_short_early_exit_3bu_no_profit(self, make_df_with_types, short_position, default_config):
        """SHORT: 3 consecutive BU but profit < 0.3% → should NOT exit."""
        df = make_df_with_types([CandleType.BIG_UP, CandleType.BIG_UP, CandleType.BIG_UP])
        current_price = 49900.0  # +0.2% profit (below threshold)

        should_exit, _, _, _ = check_early_exit_signal(short_position, df, current_price, default_config)

        assert should_exit is False

    def test_early_exit_candle_deduplication(self, make_df_with_types, long_position, default_config):
        """Early exit should not trigger twice for same candle."""
        df = make_df_with_types([CandleType.BIG_DOWN, CandleType.BIG_DOWN, CandleType.BIG_DOWN])
        current_price = 50200.0

        # Simulate 2 previous BD candles already seen
        long_position['reversal_count'] = 2

        # First check - should exit
        should_exit1, _, _, last_ts1 = check_early_exit_signal(long_position, df, current_price, default_config)
        assert should_exit1 is True

        # Mark this candle as counted
        long_position['last_counted_candle_ts'] = last_ts1

        # Second check with same candle - should NOT exit again
        should_exit2, _, _, _ = check_early_exit_signal(long_position, df, current_price, default_config)
        assert should_exit2 is False

    def test_early_exit_with_loss(self, make_df_with_types, long_position, default_config):
        """Early exit with loss (PnL < 0) → should NOT exit."""
        df = make_df_with_types([CandleType.BIG_DOWN, CandleType.BIG_DOWN, CandleType.BIG_DOWN])
        current_price = 49000.0  # -2% loss

        should_exit, _, _, _ = check_early_exit_signal(long_position, df, current_price, default_config)

        assert should_exit is False


# ── Early Exit Edge Cases ─────────────────────────────────────

class TestEarlyExitEdgeCases:
    """Test early exit edge cases and error handling."""

    def test_early_exit_empty_dataframe(self, long_position, default_config):
        """Empty DataFrame → should return (False, None)."""
        df = pd.DataFrame()
        current_price = 50000.0

        should_exit, _, _, _ = check_early_exit_signal(long_position, df, current_price, default_config)

        assert should_exit is False

    def test_early_exit_short_dataframe(self, make_df_with_types, long_position, default_config):
        """DataFrame with < confirm_candles → should return (False, None)."""
        df = make_df_with_types([CandleType.BIG_DOWN])  # only 1 candle
        current_price = 50200.0

        should_exit, _, _, _ = check_early_exit_signal(long_position, df, current_price, default_config)

        assert should_exit is False

    def test_early_exit_missing_candle_type_column(self, long_position, default_config):
        """DataFrame missing 'candle_type' column → should return (False, None)."""
        df = pd.DataFrame({
            'open': [50000],
            'close': [50100],
            'timestamp': [pd.Timestamp.now()]
        })
        current_price = 50200.0

        should_exit, _, _, _ = check_early_exit_signal(long_position, df, current_price, default_config)

        assert should_exit is False

    def test_early_exit_none_current_price(self, make_df_with_types, long_position, default_config):
        """current_price=None → should return (False, None)."""
        df = make_df_with_types([CandleType.BIG_DOWN, CandleType.BIG_DOWN, CandleType.BIG_DOWN])
        current_price = None

        should_exit, _, _, _ = check_early_exit_signal(long_position, df, current_price, default_config)

        assert should_exit is False


# ── Main Loop Integration Tests ──────────────────────────────

class TestMainLoopIntegration:
    """Test main loop logic with mocked exchange."""

    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.bot.create_exchange')
    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.bot._fetch_and_calculate_indicators')
    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.bot.fetch_ticker_cached')
    def test_process_position_calls_early_exit(
        self,
        mock_fetch_ticker,
        mock_fetch_indicators,
        mock_create_exchange,
        make_df_with_types,
        long_position,
        default_config
    ):
        """Main loop should check early exit when position exists."""
        from bingx_rl_trading_bot.scripts.production.pattern_5m.bot import _process_existing_positions
        from bingx_rl_trading_bot.scripts.production.pattern_5m.models import APICache, CircuitBreaker

        # Setup mocks
        mock_exchange = Mock()
        mock_create_exchange.return_value = mock_exchange

        # Mock indicators to return 3 BIG_DOWN candles
        df = make_df_with_types([CandleType.BIG_DOWN, CandleType.BIG_DOWN, CandleType.BIG_DOWN], n_bars=50)
        mock_fetch_indicators.return_value = df

        # Mock ticker for current price
        mock_fetch_ticker.return_value = {'last': 50200.0}

        long_position['slot_id'] = 'slot_1'
        state = {
            'positions': {'slot_1': long_position},
            'active_direction': 'LONG',
        }
        cache = APICache()
        circuit_breaker = CircuitBreaker()
        metrics = PerformanceMetrics(session_start=datetime.now().isoformat())

        # Run the function (should trigger early exit logic)
        _process_existing_positions(
            mock_exchange, state, default_config, cache, circuit_breaker, metrics
        )

        # Verify OHLCV was fetched (early exit path evaluated)
        assert mock_fetch_indicators.called

    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.bot.health_check')
    def test_health_check_called_periodically(self, mock_health_check, default_config):
        """Health check should be called at configured intervals."""
        from bingx_rl_trading_bot.scripts.production.pattern_5m.bot import _run_health_check
        from bingx_rl_trading_bot.scripts.production.pattern_5m.models import APICache, CircuitBreaker

        mock_exchange = Mock()
        cache = APICache()
        circuit_breaker = CircuitBreaker()
        metrics = PerformanceMetrics(session_start=datetime.now().isoformat())

        mock_health_check.return_value = {
            'timestamp': datetime.now().isoformat(),
            'api_ok': True,
            'circuit_breaker_ok': True,
        }

        _run_health_check(mock_exchange, default_config, cache, circuit_breaker, metrics, 'state_path')

        assert mock_health_check.called


# ── Position Sync Tests ───────────────────────────────────────

class TestPositionSync:
    """Test periodic position synchronization logic."""

    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.bot.datetime')
    def test_position_sync_interval_timing(self, mock_datetime):
        """Position sync should only trigger after sync_interval seconds."""
        from bingx_rl_trading_bot.scripts.production.pattern_5m.bot import _maybe_sync_position
        from bingx_rl_trading_bot.scripts.production.pattern_5m.models import APICache, CircuitBreaker
        import time

        # Force current time to be at a 5-minute boundary (minute=10)
        fake_now = datetime(2026, 2, 5, 12, 10, 0)
        mock_datetime.now.return_value = fake_now
        mock_datetime.fromtimestamp.return_value = datetime(2026, 2, 5, 11, 0, 0)
        mock_datetime.side_effect = lambda *a, **kw: datetime(*a, **kw)

        mock_exchange = Mock()
        state = {'position': None}
        config = {'symbol': 'BTC-USDT'}
        cache = APICache()
        circuit_breaker = CircuitBreaker()
        metrics = PerformanceMetrics(session_start=datetime.now().isoformat())

        # Last sync was 1 hour ago
        last_sync_time = time.time() - 3600

        # Should sync (interval passed + at 5-min boundary)
        new_sync_time = _maybe_sync_position(
            mock_exchange, state, config, cache, circuit_breaker, metrics, last_sync_time
        )

        # sync_time should be updated
        assert new_sync_time > last_sync_time


# ── Daily Loss Limit Tests ────────────────────────────────────

class TestDailyLossLimit:
    """Test daily loss limit checking."""

    def test_daily_loss_limit_not_reached(self):
        """Daily loss below limit → should return False."""
        from bingx_rl_trading_bot.scripts.production.pattern_5m.signals import check_daily_loss_limit
        from datetime import datetime

        state = {
            'daily_pnl': -2.0,
            'last_trade_date': datetime.now().strftime('%Y-%m-%d')
        }
        config = {'risk': {'max_daily_loss_pct': 5.0}}

        limit_reached = check_daily_loss_limit(state, config)

        assert limit_reached is False

    def test_daily_loss_limit_reached(self):
        """Daily loss >= limit → should return True."""
        from bingx_rl_trading_bot.scripts.production.pattern_5m.signals import check_daily_loss_limit
        from datetime import datetime

        state = {
            'daily_pnl': -5.5,
            'last_trade_date': datetime.now().strftime('%Y-%m-%d')
        }
        config = {'risk': {'max_daily_loss_pct': 5.0}}

        limit_reached = check_daily_loss_limit(state, config)

        assert limit_reached is True

    def test_daily_loss_limit_exactly_at_limit(self):
        """Daily loss exactly at limit → should return True."""
        from bingx_rl_trading_bot.scripts.production.pattern_5m.signals import check_daily_loss_limit
        from datetime import datetime

        state = {
            'daily_pnl': -5.0,
            'last_trade_date': datetime.now().strftime('%Y-%m-%d')
        }
        config = {'risk': {'max_daily_loss_pct': 5.0}}

        limit_reached = check_daily_loss_limit(state, config)

        assert limit_reached is True


# ── Candle-Aligned Timing Tests ──────────────────────────────

class TestCandleAlignedTiming:
    """Test candle-aligned timing helpers (v1.25.1)."""

    def test_get_candle_timing_returns_namedtuple(self):
        """_get_candle_timing should return CandleTiming with all fields."""
        timing = _get_candle_timing()

        assert isinstance(timing, CandleTiming)
        assert hasattr(timing, 'seconds_into_candle')
        assert hasattr(timing, 'seconds_until_close')
        assert hasattr(timing, 'candle_id')
        assert hasattr(timing, 'in_trading_window')

    def test_get_candle_timing_values_consistent(self):
        """seconds_into_candle + seconds_until_close should equal candle duration."""
        timing = _get_candle_timing()
        expected_seconds = CANDLE_DURATION_MS / 1000

        total = timing.seconds_into_candle + timing.seconds_until_close
        assert abs(total - expected_seconds) < 1.0  # within 1s tolerance

    def test_get_candle_timing_trading_window_flag(self):
        """in_trading_window should be True when seconds_into_candle < TRADING_WINDOW_SECONDS."""
        timing = _get_candle_timing()

        if timing.seconds_into_candle < TRADING_WINDOW_SECONDS:
            assert timing.in_trading_window is True
        else:
            assert timing.in_trading_window is False

    def test_get_candle_timing_candle_id_is_int(self):
        """candle_id should be a positive integer."""
        timing = _get_candle_timing()

        assert isinstance(timing.candle_id, int)
        assert timing.candle_id > 0

    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.bot.time')
    def test_get_candle_timing_at_candle_start(self, mock_time):
        """At exact candle boundary: seconds_into=0, in_trading_window=True."""
        candle_sec = CANDLE_DURATION_MS / 1000
        # Pick a timestamp that's exactly on a candle boundary for any timeframe
        boundary = 1700000000.0 - (1700000000.0 % candle_sec)
        mock_time.time.return_value = boundary
        timing = _get_candle_timing()

        assert timing.seconds_into_candle == 0.0
        assert timing.seconds_until_close == candle_sec
        assert timing.in_trading_window is True

    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.bot.time')
    def test_get_candle_timing_mid_candle(self, mock_time):
        """At 150s into candle: in_trading_window=False."""
        candle_sec = CANDLE_DURATION_MS / 1000
        boundary = 1700000000.0 - (1700000000.0 % candle_sec)
        mock_time.time.return_value = boundary + 150.0
        timing = _get_candle_timing()

        assert abs(timing.seconds_into_candle - 150.0) < 1.0
        assert timing.in_trading_window is False

    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.bot.shutdown_requested', False)
    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.bot.time')
    def test_interruptible_sleep_completes_normally(self, mock_time):
        """_interruptible_sleep should complete after duration when not interrupted."""
        # Make time.time() advance on each call
        start = 1699999800.0
        call_count = [0]

        def advancing_time():
            call_count[0] += 1
            return start + call_count[0] * 0.5

        mock_time.time.side_effect = advancing_time
        mock_time.sleep = Mock()

        _interruptible_sleep(2.0)

        # Should have called time.sleep at least once
        assert mock_time.sleep.called

    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.bot.time')
    def test_interruptible_sleep_respects_shutdown(self, mock_time):
        """_interruptible_sleep should exit early when shutdown_requested."""
        import bingx_rl_trading_bot.scripts.production.pattern_5m.bot as bot_module

        original = bot_module.shutdown_requested
        try:
            bot_module.shutdown_requested = True
            mock_time.time.return_value = 1699999800.0
            mock_time.sleep = Mock()

            _interruptible_sleep(10.0)

            # Should NOT have slept because shutdown was already requested
            assert not mock_time.sleep.called
        finally:
            bot_module.shutdown_requested = original

    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.bot._get_candle_timing')
    def test_calculate_sleep_no_position_maintenance(self, mock_timing):
        """No position + outside trading window → sleep up to MAX_MAINTENANCE_SLEEP."""
        mock_timing.return_value = CandleTiming(
            seconds_into_candle=100.0,
            seconds_until_close=200.0,
            candle_id=1000,
            in_trading_window=False,
        )

        duration = _calculate_sleep_duration(has_position=False, last_processed_candle_id=1000)

        # Should be min(MAX_MAINTENANCE_SLEEP, seconds_until_close + settle)
        expected = min(MAX_MAINTENANCE_SLEEP, 200.0 + CANDLE_SETTLE_SECONDS)
        assert duration == expected

    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.bot._get_candle_timing')
    def test_calculate_sleep_with_position_monitoring(self, mock_timing):
        """Has position + outside trading window → sleep POSITION_MONITOR_INTERVAL."""
        mock_timing.return_value = CandleTiming(
            seconds_into_candle=100.0,
            seconds_until_close=200.0,
            candle_id=1000,
            in_trading_window=False,
        )

        duration = _calculate_sleep_duration(has_position=True, last_processed_candle_id=1000)

        expected = min(POSITION_MONITOR_INTERVAL, 200.0 + CANDLE_SETTLE_SECONDS)
        assert duration == expected

    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.bot._get_candle_timing')
    def test_calculate_sleep_unprocessed_candle_in_window(self, mock_timing):
        """Unprocessed candle + in trading window → return 0 (process immediately)."""
        mock_timing.return_value = CandleTiming(
            seconds_into_candle=10.0,
            seconds_until_close=290.0,
            candle_id=1001,
            in_trading_window=True,
        )

        # last_processed is 1000, current candle is 1001 → unprocessed
        duration = _calculate_sleep_duration(has_position=False, last_processed_candle_id=1000)

        assert duration == 0

    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.bot._get_candle_timing')
    def test_calculate_sleep_already_processed_candle(self, mock_timing):
        """Already processed candle + in trading window → sleep until next candle."""
        mock_timing.return_value = CandleTiming(
            seconds_into_candle=10.0,
            seconds_until_close=290.0,
            candle_id=1000,
            in_trading_window=True,
        )

        # Same candle_id → already processed, should sleep
        duration = _calculate_sleep_duration(has_position=False, last_processed_candle_id=1000)

        assert duration > 0


# ── _should_sync_now Tests ───────────────────────────────────


class TestShouldSyncNow:
    """Test _should_sync_now() clock-aligned interval logic."""

    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.bot.datetime')
    def test_first_run_always_syncs(self, mock_dt):
        """last_sync_time=0 → always returns True."""
        mock_dt.now.return_value = datetime(2026, 2, 18, 12, 3, 0)
        assert _should_sync_now(0, 5) is True

    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.bot.datetime')
    def test_at_interval_boundary(self, mock_dt):
        """Current minute at 5-min boundary + enough time elapsed → True."""
        mock_dt.now.return_value = datetime(2026, 2, 18, 12, 10, 0)
        mock_dt.fromtimestamp.return_value = datetime(2026, 2, 18, 12, 5, 0)
        assert _should_sync_now(time.time() - 300, 5) is True

    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.bot.datetime')
    def test_not_at_interval_boundary(self, mock_dt):
        """Current minute NOT at 5-min boundary → False."""
        mock_dt.now.return_value = datetime(2026, 2, 18, 12, 7, 0)
        assert _should_sync_now(time.time() - 300, 5) is False

    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.bot.datetime')
    def test_duplicate_prevention_same_minute(self, mock_dt):
        """Same hour and minute as last sync → False (prevent duplicate)."""
        mock_dt.now.return_value = datetime(2026, 2, 18, 12, 10, 30)
        mock_dt.fromtimestamp.return_value = datetime(2026, 2, 18, 12, 10, 5)
        assert _should_sync_now(time.time() - 25, 5) is False

    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.bot.datetime')
    def test_30min_interval(self, mock_dt):
        """30-minute interval at minute 0 or 30 → True."""
        mock_dt.now.return_value = datetime(2026, 2, 18, 13, 0, 0)
        mock_dt.fromtimestamp.return_value = datetime(2026, 2, 18, 12, 30, 0)
        assert _should_sync_now(time.time() - 1800, 30) is True


# ── _verify_exchange_settings Tests ──────────────────────────


class TestVerifyExchangeSettings:
    """Test _verify_exchange_settings() exchange setup."""

    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.bot.set_margin_mode')
    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.bot.verify_position_mode')
    def test_success(self, mock_verify, mock_margin):
        """Both checks pass → no exception."""
        mock_verify.return_value = True
        exchange = MagicMock()
        config = {'symbol': 'BTC/USDT:USDT'}
        _verify_exchange_settings(exchange, config)
        mock_verify.assert_called_once()
        mock_margin.assert_called_once()

    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.bot.set_margin_mode')
    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.bot.verify_position_mode')
    def test_position_mode_failure_raises(self, mock_verify, mock_margin):
        """Position mode check fails → raises RuntimeError."""
        mock_verify.return_value = False
        exchange = MagicMock()
        config = {'symbol': 'BTC/USDT:USDT'}
        with pytest.raises(RuntimeError, match='Position mode'):
            _verify_exchange_settings(exchange, config)


# ── _fetch_and_calculate_indicators Tests ────────────────────


class TestFetchAndCalculateIndicators:
    """Test _fetch_and_calculate_indicators() data fetching."""

    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.bot.calculate_indicators')
    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.bot.fetch_ohlcv')
    def test_successful_fetch(self, mock_ohlcv, mock_calc):
        """Normal OHLCV data → returns DataFrame."""
        ohlcv_data = [[i * 300000, 50000 + i, 50050 + i, 49950 + i, 50010 + i, 100]
                       for i in range(50)]
        mock_ohlcv.return_value = ohlcv_data
        mock_calc.side_effect = lambda df, cfg: df
        exchange = MagicMock()
        config = {'symbol': 'BTC/USDT:USDT', 'timeframe': '5m'}
        result = _fetch_and_calculate_indicators(exchange, config)
        assert result is not None
        assert len(result) == 50

    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.bot.fetch_ohlcv')
    def test_insufficient_data_returns_none(self, mock_ohlcv):
        """Fewer than 20 candles → returns None."""
        mock_ohlcv.return_value = [[i, 50000, 50050, 49950, 50010, 100] for i in range(10)]
        exchange = MagicMock()
        config = {'symbol': 'BTC/USDT:USDT', 'timeframe': '5m'}
        result = _fetch_and_calculate_indicators(exchange, config)
        assert result is None

    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.bot.fetch_ohlcv')
    def test_empty_ohlcv_returns_none(self, mock_ohlcv):
        """Empty OHLCV list → returns None."""
        mock_ohlcv.return_value = []
        exchange = MagicMock()
        config = {'symbol': 'BTC/USDT:USDT', 'timeframe': '5m'}
        result = _fetch_and_calculate_indicators(exchange, config)
        assert result is None

    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.bot.fetch_ohlcv')
    def test_network_error_returns_none(self, mock_ohlcv):
        """Network error during fetch → returns None."""
        import ccxt
        mock_ohlcv.side_effect = ccxt.NetworkError('timeout')
        exchange = MagicMock()
        config = {'symbol': 'BTC/USDT:USDT', 'timeframe': '5m'}
        result = _fetch_and_calculate_indicators(exchange, config)
        assert result is None

    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.bot.fetch_ohlcv')
    def test_exchange_error_returns_none(self, mock_ohlcv):
        """Exchange error during fetch → returns None."""
        import ccxt
        mock_ohlcv.side_effect = ccxt.ExchangeError('invalid')
        exchange = MagicMock()
        config = {'symbol': 'BTC/USDT:USDT', 'timeframe': '5m'}
        result = _fetch_and_calculate_indicators(exchange, config)
        assert result is None


# ── _process_entry_signal Tests ───────────────────────────────


class TestProcessNoPosition:
    """Test _process_entry_signal() entry signal checking."""

    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.bot.open_position')
    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.bot.check_entry_signal')
    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.bot._fetch_and_calculate_indicators')
    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.bot.check_cooldown')
    def test_signal_detected_opens_position(self, mock_cool, mock_fetch, mock_signal, mock_open):
        """Signal detected → calls open_position, returns True on success."""
        mock_cool.return_value = True
        mock_fetch.return_value = pd.DataFrame({'close': [50000]})
        mock_signal.return_value = ('LONG', 'Pattern: BD-BD-BU (LONG)')
        mock_open.return_value = True

        exchange = MagicMock()
        state = {}
        config = {'symbol': 'BTC/USDT:USDT'}
        cache = APICache()
        cb = CircuitBreaker()
        metrics = PerformanceMetrics()

        result = _process_entry_signal(exchange, state, config, cache, cb, metrics)
        assert result is True
        mock_open.assert_called_once()

    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.bot._fetch_and_calculate_indicators')
    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.bot.check_cooldown')
    def test_cooldown_blocks(self, mock_cool, mock_fetch):
        """Cooldown active → returns False without checking signal."""
        mock_cool.return_value = False

        exchange = MagicMock()
        state = {}
        config = {'symbol': 'BTC/USDT:USDT'}
        cache = APICache()
        cb = CircuitBreaker()
        metrics = PerformanceMetrics()

        result = _process_entry_signal(exchange, state, config, cache, cb, metrics)
        assert result is False
        mock_fetch.assert_not_called()

    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.bot.check_entry_signal')
    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.bot._fetch_and_calculate_indicators')
    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.bot.check_cooldown')
    def test_no_signal_returns_false(self, mock_cool, mock_fetch, mock_signal):
        """No signal detected → returns False."""
        mock_cool.return_value = True
        mock_fetch.return_value = pd.DataFrame({'close': [50000]})
        mock_signal.return_value = (None, None)

        exchange = MagicMock()
        state = {}
        config = {'symbol': 'BTC/USDT:USDT'}
        cache = APICache()
        cb = CircuitBreaker()
        metrics = PerformanceMetrics()

        result = _process_entry_signal(exchange, state, config, cache, cb, metrics)
        assert result is False

    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.bot._fetch_and_calculate_indicators')
    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.bot.check_cooldown')
    def test_fetch_failure_returns_false(self, mock_cool, mock_fetch):
        """Indicator fetch fails → returns False."""
        mock_cool.return_value = True
        mock_fetch.return_value = None

        exchange = MagicMock()
        state = {}
        config = {'symbol': 'BTC/USDT:USDT'}
        cache = APICache()
        cb = CircuitBreaker()
        metrics = PerformanceMetrics()

        result = _process_entry_signal(exchange, state, config, cache, cb, metrics)
        assert result is False

    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.bot.open_position')
    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.bot.check_entry_signal')
    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.bot._fetch_and_calculate_indicators')
    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.bot.check_cooldown')
    def test_open_position_failure_returns_false(self, mock_cool, mock_fetch, mock_signal, mock_open):
        """Signal detected but open_position fails → returns False."""
        mock_cool.return_value = True
        mock_fetch.return_value = pd.DataFrame({'close': [50000]})
        mock_signal.return_value = ('SHORT', 'Pattern: DN-BU-BU (SHORT)')
        mock_open.return_value = False

        exchange = MagicMock()
        state = {}
        config = {'symbol': 'BTC/USDT:USDT'}
        cache = APICache()
        cb = CircuitBreaker()
        metrics = PerformanceMetrics()

        result = _process_entry_signal(exchange, state, config, cache, cb, metrics)
        assert result is False


# ── _process_existing_positions Tests (extended) ──────────────


class TestProcessExistingPositionExtended:
    """Extended tests for _process_existing_positions() early exit flow."""

    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.bot.save_state')
    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.bot.close_position_market')
    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.bot.check_early_exit_signal')
    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.bot.fetch_ticker_cached')
    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.bot._fetch_and_calculate_indicators')
    def test_early_exit_triggers_close(
        self, mock_fetch, mock_ticker, mock_early, mock_close, mock_save
    ):
        """Early exit triggered → close_position_market called."""
        mock_fetch.return_value = pd.DataFrame({'close': [50000]})
        mock_ticker.return_value = {'last': 50200.0}
        mock_early.return_value = (True, 3, '3-BD reversal', 1234567890000)
        mock_close.return_value = True

        pos = {'slot_id': 's1', 'direction': 'LONG', 'entry_price': 50000, 'reversal_count': 2}
        state = {'positions': {'s1': pos}, 'active_direction': 'LONG'}
        exchange = MagicMock()
        config = {'symbol': 'BTC/USDT:USDT', 'leverage': 3}
        cache = APICache()
        cb = CircuitBreaker()
        metrics = PerformanceMetrics()

        result = _process_existing_positions(exchange, state, config, cache, cb, metrics)
        assert result is True
        mock_close.assert_called_once()

    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.bot.check_early_exit_signal')
    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.bot.fetch_ticker_cached')
    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.bot._fetch_and_calculate_indicators')
    def test_no_early_exit_returns_false(self, mock_fetch, mock_ticker, mock_early):
        """No early exit signal → returns False."""
        mock_fetch.return_value = pd.DataFrame({'close': [50000]})
        mock_ticker.return_value = {'last': 50100.0}
        mock_early.return_value = (False, 0, None, None)

        pos = {'slot_id': 's1', 'direction': 'LONG', 'entry_price': 50000, 'reversal_count': 0}
        state = {'positions': {'s1': pos}, 'active_direction': 'LONG'}
        exchange = MagicMock()
        config = {'symbol': 'BTC/USDT:USDT', 'leverage': 3}
        cache = APICache()
        cb = CircuitBreaker()
        metrics = PerformanceMetrics()

        result = _process_existing_positions(exchange, state, config, cache, cb, metrics)
        assert result is False

    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.bot._fetch_and_calculate_indicators')
    def test_fetch_failure_returns_false(self, mock_fetch):
        """Indicator fetch fails → returns False without error."""
        mock_fetch.return_value = None

        pos = {'slot_id': 's1', 'direction': 'LONG', 'entry_price': 50000}
        state = {'positions': {'s1': pos}, 'active_direction': 'LONG'}
        exchange = MagicMock()
        config = {'symbol': 'BTC/USDT:USDT', 'leverage': 3}
        cache = APICache()
        cb = CircuitBreaker()
        metrics = PerformanceMetrics()

        result = _process_existing_positions(exchange, state, config, cache, cb, metrics)
        assert result is False


# ── _log_position_status Tests ───────────────────────────────


class TestLogPositionStatus:
    """Test _log_position_status() logging helper."""

    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.bot.fetch_ticker_cached')
    def test_normal_long_position(self, mock_ticker):
        """Logs PnL and TP/SL progress for LONG position."""
        mock_ticker.return_value = {'last': 51000.0}
        position = {
            'direction': 'LONG',
            'entry_price': 50000.0,
            'tp_price': 52000.0,
            'sl_price': 49000.0,
            'reason': 'Pattern: BD-BD-BU (LONG)',
        }
        config = {'symbol': 'BTC/USDT:USDT', 'leverage': 3}
        cache = APICache()
        exchange = MagicMock()
        # Should not raise
        _log_position_status(position, config, cache, exchange)

    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.bot.fetch_ticker_cached')
    def test_network_error_handled(self, mock_ticker):
        """Network error → does not raise."""
        import ccxt
        mock_ticker.side_effect = ccxt.NetworkError('timeout')
        position = {'direction': 'LONG', 'entry_price': 50000.0}
        config = {'symbol': 'BTC/USDT:USDT', 'leverage': 3}
        cache = APICache()
        exchange = MagicMock()
        _log_position_status(position, config, cache, exchange)  # Should not raise


# ── _log_waiting_status Tests ────────────────────────────────


class TestLogWaitingStatus:
    """Test _log_waiting_status() logging helper."""

    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.bot.fetch_ticker_cached')
    def test_normal_waiting(self, mock_ticker):
        """Logs waiting status with daily stats."""
        mock_ticker.return_value = {'last': 50000.0}
        state = {'daily_pnl': -1.5, 'daily_trades': 3}
        metrics = PerformanceMetrics()
        metrics.total_trades = 40
        metrics.actual_win_rate = 72.5
        cache = APICache()
        exchange = MagicMock()
        config = {'symbol': 'BTC/USDT:USDT'}
        _log_waiting_status(state, metrics, cache, exchange, config)  # Should not raise

    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.bot.fetch_ticker_cached')
    def test_network_error_handled(self, mock_ticker):
        """Network error → does not raise."""
        import ccxt
        mock_ticker.side_effect = ccxt.NetworkError('timeout')
        state = {}
        metrics = PerformanceMetrics()
        cache = APICache()
        exchange = MagicMock()
        config = {'symbol': 'BTC/USDT:USDT'}
        _log_waiting_status(state, metrics, cache, exchange, config)  # Should not raise


# ── _maybe_run_health_check Tests ────────────────────────────


class TestMaybeRunHealthCheck:
    """Test _maybe_run_health_check() clock-aligned wrapper."""

    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.bot._run_health_check')
    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.bot._should_sync_now')
    def test_runs_when_due(self, mock_sync, mock_hc):
        """When _should_sync_now=True → runs health check, returns new time."""
        mock_sync.return_value = True
        exchange = MagicMock()
        config = {'symbol': 'BTC/USDT:USDT'}
        cache = APICache()
        cb = CircuitBreaker()
        metrics = PerformanceMetrics()

        result = _maybe_run_health_check(
            exchange, config, cache, cb, metrics, 'state.json', 0, 30
        )
        assert result > 0
        mock_hc.assert_called_once()

    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.bot._run_health_check')
    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.bot._should_sync_now')
    def test_skips_when_not_due(self, mock_sync, mock_hc):
        """When _should_sync_now=False → skips, returns same time."""
        mock_sync.return_value = False
        exchange = MagicMock()
        config = {'symbol': 'BTC/USDT:USDT'}
        cache = APICache()
        cb = CircuitBreaker()
        metrics = PerformanceMetrics()

        old_time = 1234567890.0
        result = _maybe_run_health_check(
            exchange, config, cache, cb, metrics, 'state.json', old_time, 30
        )
        assert result == old_time
        mock_hc.assert_not_called()


# ── _wait_for_candle_settle ────────────────────────────────


class TestWaitForCandleSettle:
    """Test _wait_for_candle_settle() — pause until candle data stabilizes."""

    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.bot._interruptible_sleep')
    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.bot.time.time')
    def test_waits_if_too_early(self, mock_time, mock_sleep):
        """If within settle seconds → sleeps the remaining time."""
        # Pick a time exactly at a candle boundary + 1ms (works for any timeframe)
        candle_sec = CANDLE_DURATION_MS / 1000
        boundary = 1700000000.0 - (1700000000.0 % candle_sec)
        mock_time.return_value = boundary + 0.001  # 1ms into candle
        config = {'symbol': 'BTC/USDT:USDT'}
        _wait_for_candle_settle(config)
        mock_sleep.assert_called_once()
        wait_time = mock_sleep.call_args[0][0]
        assert wait_time > 0

    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.bot._interruptible_sleep')
    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.bot.time.time')
    def test_no_wait_if_settled(self, mock_time, mock_sleep):
        """If already past settle seconds → no sleep."""
        # Set time well past settle point (20s into candle, works for any timeframe)
        candle_sec = CANDLE_DURATION_MS / 1000
        boundary = 1700000000.0 - (1700000000.0 % candle_sec)
        mock_time.return_value = boundary + 20.0  # 20s into candle
        config = {'symbol': 'BTC/USDT:USDT'}
        _wait_for_candle_settle(config)
        mock_sleep.assert_not_called()


# ── _log_pattern_summary ────────────────────────────────────


class TestLogPatternSummary:
    """Test _log_pattern_summary() startup logging."""

    def test_static_patterns(self):
        """Static config → logs 'static' source."""
        config = {
            'strategy': {
                'long_patterns': ['A', 'B'],
                'short_patterns': ['C'],
            }
        }
        _log_pattern_summary(config)  # should not raise

    def test_dynamic_per_pattern(self):
        """Dynamic per-pattern config → logs TP/SL range."""
        config = {
            'strategy': {
                'long_patterns': ['A'],
                'short_patterns': ['B', 'C'],
            },
            '_dynamic_tpsl_per_pattern': True,
            '_dynamic_patterns_tpsl': {
                'A': (2.0, 3.0),
                'B': (1.5, 4.0),
                'C': (2.5, 2.0),
            },
        }
        _log_pattern_summary(config)  # should not raise

    def test_dynamic_universal(self):
        """Dynamic universal config → logs universal TP/SL."""
        config = {
            'strategy': {
                'long_patterns': ['A'],
                'short_patterns': ['B'],
            },
            '_dynamic_tpsl_universal': True,
            '_dynamic_tp': 2.0,
            '_dynamic_sl': 3.0,
        }
        _log_pattern_summary(config)  # should not raise

    def test_empty_per_pattern_tpsl(self):
        """Per-pattern mode with empty tpsl dict → no crash."""
        config = {
            'strategy': {'long_patterns': [], 'short_patterns': []},
            '_dynamic_tpsl_per_pattern': True,
            '_dynamic_patterns_tpsl': {},
        }
        _log_pattern_summary(config)  # should not raise


# ── signal_handler ─────────────────────────────────────────


class TestSignalHandler:
    """Test signal_handler() sets shutdown flag."""

    def test_sets_shutdown_flag(self):
        """Signal handler sets shutdown_requested = True."""
        import bingx_rl_trading_bot.scripts.production.pattern_5m.bot as bot_module
        original = bot_module.shutdown_requested
        try:
            bot_module.shutdown_requested = False
            signal_handler(2, None)  # SIGINT
            assert bot_module.shutdown_requested is True
        finally:
            bot_module.shutdown_requested = original


# ── _run_health_check ──────────────────────────────────────


class TestRunHealthCheck:
    """Test _run_health_check() warning paths."""

    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.bot.health_check')
    def test_healthy_status_no_warning(self, mock_hc):
        """Healthy status → debug log only, no warning."""
        mock_hc.return_value = {'status': 'healthy', 'checks': {}}
        _run_health_check(MagicMock(), {}, APICache(), CircuitBreaker(), PerformanceMetrics(), 's.json')

    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.bot.health_check')
    def test_degraded_status_logs_warning(self, mock_hc):
        """Degraded status → logs warning per failing check."""
        mock_hc.return_value = {
            'status': 'degraded',
            'checks': {
                'api': {'status': 'error', 'message': 'timeout'},
                'cb': {'status': 'ok'},
            }
        }
        _run_health_check(MagicMock(), {}, APICache(), CircuitBreaker(), PerformanceMetrics(), 's.json')

    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.bot.health_check')
    def test_exception_handled(self, mock_hc):
        """Exception during health check → caught, no crash."""
        mock_hc.side_effect = RuntimeError('fail')
        _run_health_check(MagicMock(), {}, APICache(), CircuitBreaker(), PerformanceMetrics(), 's.json')


# ── _maybe_sync_position no-sync path ────────────────────────


class TestMaybeSyncPositionNoSync:
    """Test _maybe_sync_position when sync is NOT needed (line 441)."""

    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.bot._should_sync_now',
           return_value=False)
    def test_no_sync_returns_same_time(self, mock_sync):
        """Sync not needed → returns original last_sync_time (line 441)."""
        exchange = MagicMock()
        state = {'position': None}
        config = {'symbol': 'BTC/USDT:USDT'}
        cache = APICache()
        cb = CircuitBreaker()
        metrics = PerformanceMetrics()
        original_time = 1000.0
        result = _maybe_sync_position(exchange, state, config, cache, cb, metrics, original_time)
        assert result == original_time


# ── _process_existing_positions — early exit failure + exception ────


class TestProcessExistingPositionFailPaths:
    """Test early exit failure and exception paths (lines 496-499)."""

    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.bot.save_state')
    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.bot.close_position_market',
           return_value=False)
    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.bot.check_early_exit_signal',
           return_value=(True, 3, '3-BD reversal', 1234567890000))
    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.bot.fetch_ticker_cached',
           return_value={'last': 50200.0})
    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.bot._fetch_and_calculate_indicators')
    def test_early_exit_close_fails(self, mock_fetch, mock_tick, mock_early, mock_close, mock_save):
        """Early exit triggered but close_position_market fails → return False (line 496)."""
        mock_fetch.return_value = pd.DataFrame({'close': [50000]})
        pos = {'slot_id': 's1', 'direction': 'LONG', 'entry_price': 50000, 'reversal_count': 2}
        state = {'positions': {'s1': pos}, 'active_direction': 'LONG'}
        result = _process_existing_positions(
            MagicMock(), state, {'symbol': 'BTC/USDT:USDT', 'leverage': 3},
            APICache(), CircuitBreaker(), PerformanceMetrics()
        )
        assert result is False

    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.bot.check_early_exit_signal',
           side_effect=RuntimeError('unexpected'))
    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.bot.fetch_ticker_cached',
           return_value={'last': 50200.0})
    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.bot._fetch_and_calculate_indicators')
    def test_early_exit_exception(self, mock_fetch, mock_tick, mock_early):
        """Exception during early exit check → caught, returns False (lines 498-499)."""
        mock_fetch.return_value = pd.DataFrame({'close': [50000]})
        pos = {'slot_id': 's1', 'direction': 'LONG', 'entry_price': 50000}
        state = {'positions': {'s1': pos}, 'active_direction': 'LONG'}
        result = _process_existing_positions(
            MagicMock(), state, {'symbol': 'BTC/USDT:USDT', 'leverage': 3},
            APICache(), CircuitBreaker(), PerformanceMetrics()
        )
        assert result is False


# ── _fetch_and_calculate_indicators — generic exception ────


class TestFetchIndicatorsGenericException:
    """Test _fetch_and_calculate_indicators generic exception (lines 592-594)."""

    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.bot.calculate_indicators')
    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.bot.fetch_ohlcv')
    def test_generic_exception_returns_none(self, mock_fetch, mock_calc):
        """Generic exception → returns None (lines 592-594)."""
        mock_fetch.return_value = [[1, 2, 3, 4, 5, 6]] * 30  # 30 candles
        mock_calc.side_effect = ValueError('bad data')
        result = _fetch_and_calculate_indicators(
            MagicMock(), {'symbol': 'BTC/USDT:USDT', 'timeframe': '5m'}
        )
        assert result is None


# ── _log_position_status / _log_waiting_status — generic exceptions ────


class TestLogPositionStatusGenericException:
    """Test _log_position_status generic exception (lines 636-637)."""

    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.bot.fetch_ticker_cached',
           side_effect=ValueError('bad ticker'))
    def test_generic_exception_no_crash(self, mock_ticker):
        """Generic exception → caught, no crash (lines 636-637)."""
        position = {'direction': 'LONG', 'entry_price': 50000.0}
        config = {'symbol': 'BTC/USDT:USDT', 'leverage': 3}
        _log_position_status(position, config, APICache(), MagicMock())


class TestLogWaitingStatusGenericException:
    """Test _log_waiting_status generic exception (lines 662-663)."""

    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.bot.fetch_ticker_cached',
           side_effect=ValueError('bad ticker'))
    def test_generic_exception_no_crash(self, mock_ticker):
        """Generic exception → caught, no crash (lines 662-663)."""
        state = {}
        metrics = PerformanceMetrics()
        _log_waiting_status(state, metrics, APICache(), MagicMock(), {'symbol': 'BTC/USDT:USDT'})


# ── _run_bot_main integration tests ────────────────────────────


def _shutdown_after_one_iter(*args, **kwargs):
    """Side effect for _interruptible_sleep that sets shutdown_requested=True."""
    import bingx_rl_trading_bot.scripts.production.pattern_5m.bot as bot_mod
    bot_mod.shutdown_requested = True


BOT = 'bingx_rl_trading_bot.scripts.production.pattern_5m.bot'


class TestRunBotMainNoPosition:
    """Test _run_bot_main loop with no position (maintenance window path)."""

    @patch(f'{BOT}.save_metrics')
    @patch(f'{BOT}.save_state')
    @patch(f'{BOT}._interruptible_sleep', side_effect=_shutdown_after_one_iter)
    @patch(f'{BOT}._calculate_sleep_duration', return_value=1.0)
    @patch(f'{BOT}._log_waiting_status')
    @patch(f'{BOT}._maybe_run_health_check', return_value=0.0)
    @patch(f'{BOT}._maybe_sync_position', return_value=0.0)
    @patch(f'{BOT}.check_daily_loss_limit', return_value=False)
    @patch(f'{BOT}._get_candle_timing')
    @patch(f'{BOT}.adjust_tpsl_to_config')
    @patch(f'{BOT}.recover_from_crash')
    @patch(f'{BOT}._verify_exchange_settings')
    @patch(f'{BOT}.sync_metrics_with_state', return_value=PerformanceMetrics())
    @patch(f'{BOT}.load_state', return_value={'position': None, 'daily_pnl': 0, 'daily_trades': 0})
    @patch(f'{BOT}.load_metrics', return_value=PerformanceMetrics())
    @patch(f'{BOT}.create_exchange')
    def test_maintenance_window_no_position(
        self, mock_exch, mock_lm, mock_ls, mock_sync_m, mock_ve, mock_rc, mock_adj,
        mock_timing, mock_dll, mock_msync, mock_mhc, mock_lws,
        mock_csd, mock_sleep, mock_ss, mock_sm
    ):
        """Loop runs once in maintenance window (no position) then shuts down."""
        import bingx_rl_trading_bot.scripts.production.pattern_5m.bot as bot_mod
        original = bot_mod.shutdown_requested
        bot_mod.shutdown_requested = False

        # Maintenance window (not in trading window)
        mock_timing.return_value = CandleTiming(
            in_trading_window=False,
            candle_id=100,
            seconds_into_candle=60.0,
            seconds_until_close=240.0,
        )
        mock_exch.return_value = MagicMock()

        try:
            _run_bot_main(
                config={'symbol': 'BTC/USDT:USDT', 'timeframe': '5m', 'leverage': 3},
                state_path='state.json',
                metrics_path='metrics.json',
            )
            # Verify graceful shutdown saves
            mock_ss.assert_called()
            mock_sm.assert_called()
        finally:
            bot_mod.shutdown_requested = original


class TestRunBotMainTradingWindowNoPosition:
    """Test _run_bot_main loop in trading window with no position."""

    @patch(f'{BOT}.save_metrics')
    @patch(f'{BOT}.save_state')
    @patch(f'{BOT}._interruptible_sleep', side_effect=_shutdown_after_one_iter)
    @patch(f'{BOT}._calculate_sleep_duration', return_value=1.0)
    @patch(f'{BOT}._process_entry_signal')
    @patch(f'{BOT}._wait_for_candle_settle')
    @patch(f'{BOT}.check_daily_loss_limit', return_value=False)
    @patch(f'{BOT}._get_candle_timing')
    @patch(f'{BOT}.adjust_tpsl_to_config')
    @patch(f'{BOT}.recover_from_crash')
    @patch(f'{BOT}._verify_exchange_settings')
    @patch(f'{BOT}.sync_metrics_with_state', return_value=PerformanceMetrics())
    @patch(f'{BOT}.load_state', return_value={'position': None, 'daily_pnl': 0, 'daily_trades': 0})
    @patch(f'{BOT}.load_metrics', return_value=PerformanceMetrics())
    @patch(f'{BOT}.create_exchange')
    def test_trading_window_processes_no_position(
        self, mock_exch, mock_lm, mock_ls, mock_sync_m, mock_ve, mock_rc, mock_adj,
        mock_timing, mock_dll, mock_settle, mock_pnp,
        mock_csd, mock_sleep, mock_ss, mock_sm
    ):
        """Trading window + no position → _process_entry_signal called."""
        import bingx_rl_trading_bot.scripts.production.pattern_5m.bot as bot_mod
        original = bot_mod.shutdown_requested
        bot_mod.shutdown_requested = False

        mock_timing.return_value = CandleTiming(
            in_trading_window=True,
            candle_id=100,
            seconds_into_candle=5.0,
            seconds_until_close=295.0,
        )
        mock_exch.return_value = MagicMock()

        try:
            _run_bot_main(
                config={'symbol': 'BTC/USDT:USDT', 'timeframe': '5m', 'leverage': 3},
                state_path='state.json',
                metrics_path='metrics.json',
            )
            mock_pnp.assert_called_once()
        finally:
            bot_mod.shutdown_requested = original


class TestRunBotMainTradingWindowWithPosition:
    """Test _run_bot_main in trading window with existing position."""

    @patch(f'{BOT}.save_metrics')
    @patch(f'{BOT}.save_state')
    @patch(f'{BOT}._interruptible_sleep', side_effect=_shutdown_after_one_iter)
    @patch(f'{BOT}._calculate_sleep_duration', return_value=1.0)
    @patch(f'{BOT}._process_existing_positions')
    @patch(f'{BOT}.check_position_status', return_value=False)
    @patch(f'{BOT}._wait_for_candle_settle')
    @patch(f'{BOT}.check_daily_loss_limit', return_value=False)
    @patch(f'{BOT}._get_candle_timing')
    @patch(f'{BOT}.adjust_tpsl_to_config')
    @patch(f'{BOT}.recover_from_crash')
    @patch(f'{BOT}._verify_exchange_settings')
    @patch(f'{BOT}.sync_metrics_with_state', return_value=PerformanceMetrics())
    @patch(f'{BOT}.load_state', return_value={
        'positions': {'s1': {'slot_id': 's1', 'direction': 'LONG', 'entry_price': 50000}},
        'active_direction': 'LONG',
        'daily_pnl': 0, 'daily_trades': 0,
    })
    @patch(f'{BOT}.load_metrics', return_value=PerformanceMetrics())
    @patch(f'{BOT}.create_exchange')
    def test_trading_window_processes_existing_position(
        self, mock_exch, mock_lm, mock_ls, mock_sync_m, mock_ve, mock_rc, mock_adj,
        mock_timing, mock_dll, mock_settle, mock_cps, mock_pep,
        mock_csd, mock_sleep, mock_ss, mock_sm
    ):
        """Trading window + has position + not closed → _process_existing_positions called."""
        import bingx_rl_trading_bot.scripts.production.pattern_5m.bot as bot_mod
        original = bot_mod.shutdown_requested
        bot_mod.shutdown_requested = False

        mock_timing.return_value = CandleTiming(
            in_trading_window=True,
            candle_id=100,
            seconds_into_candle=5.0,
            seconds_until_close=295.0,
        )
        mock_exch.return_value = MagicMock()

        try:
            _run_bot_main(
                config={'symbol': 'BTC/USDT:USDT', 'timeframe': '5m', 'leverage': 3},
                state_path='state.json',
                metrics_path='metrics.json',
            )
            mock_pep.assert_called_once()
        finally:
            bot_mod.shutdown_requested = original


class TestRunBotMainPositionClosedInTradingWindow:
    """Test position closed during trading window → tries new entry."""

    @patch(f'{BOT}.save_metrics')
    @patch(f'{BOT}.save_state')
    @patch(f'{BOT}._interruptible_sleep', side_effect=_shutdown_after_one_iter)
    @patch(f'{BOT}._calculate_sleep_duration', return_value=1.0)
    @patch(f'{BOT}._process_entry_signal')
    @patch(f'{BOT}.check_position_status', return_value=True)
    @patch(f'{BOT}._wait_for_candle_settle')
    @patch(f'{BOT}.check_daily_loss_limit', return_value=False)
    @patch(f'{BOT}._get_candle_timing')
    @patch(f'{BOT}.adjust_tpsl_to_config')
    @patch(f'{BOT}.recover_from_crash')
    @patch(f'{BOT}._verify_exchange_settings')
    @patch(f'{BOT}.sync_metrics_with_state', return_value=PerformanceMetrics())
    @patch(f'{BOT}.load_state', return_value={
        'positions': {'s1': {'slot_id': 's1', 'direction': 'LONG', 'entry_price': 50000}},
        'active_direction': 'LONG',
        'daily_pnl': 0, 'daily_trades': 0,
    })
    @patch(f'{BOT}.load_metrics', return_value=PerformanceMetrics())
    @patch(f'{BOT}.create_exchange')
    def test_position_closed_then_new_entry_attempted(
        self, mock_exch, mock_lm, mock_ls, mock_sync_m, mock_ve, mock_rc, mock_adj,
        mock_timing, mock_dll, mock_settle, mock_cps, mock_pnp,
        mock_csd, mock_sleep, mock_ss, mock_sm
    ):
        """Position closed in trading window → _process_entry_signal for new entry."""
        import bingx_rl_trading_bot.scripts.production.pattern_5m.bot as bot_mod
        original = bot_mod.shutdown_requested
        bot_mod.shutdown_requested = False

        mock_timing.return_value = CandleTiming(
            in_trading_window=True,
            candle_id=100,
            seconds_into_candle=5.0,
            seconds_until_close=295.0,
        )
        mock_exch.return_value = MagicMock()

        try:
            _run_bot_main(
                config={'symbol': 'BTC/USDT:USDT', 'timeframe': '5m', 'leverage': 3},
                state_path='state.json',
                metrics_path='metrics.json',
            )
            mock_pnp.assert_called_once()
        finally:
            bot_mod.shutdown_requested = original


class TestRunBotMainDailyLossLimit:
    """Test daily loss limit triggers pause."""

    @patch(f'{BOT}.save_metrics')
    @patch(f'{BOT}.save_state')
    @patch(f'{BOT}._interruptible_sleep', side_effect=_shutdown_after_one_iter)
    @patch(f'{BOT}.check_daily_loss_limit', return_value=True)
    @patch(f'{BOT}.adjust_tpsl_to_config')
    @patch(f'{BOT}.recover_from_crash')
    @patch(f'{BOT}._verify_exchange_settings')
    @patch(f'{BOT}.sync_metrics_with_state', return_value=PerformanceMetrics())
    @patch(f'{BOT}.load_state', return_value={'position': None, 'daily_pnl': -15, 'daily_trades': 5})
    @patch(f'{BOT}.load_metrics', return_value=PerformanceMetrics())
    @patch(f'{BOT}.create_exchange')
    def test_daily_loss_pauses_then_continues(
        self, mock_exch, mock_lm, mock_ls, mock_sync_m, mock_ve, mock_rc, mock_adj,
        mock_dll, mock_sleep, mock_ss, mock_sm
    ):
        """Daily loss limit hit → pause sleep called, then shutdown."""
        import bingx_rl_trading_bot.scripts.production.pattern_5m.bot as bot_mod
        original = bot_mod.shutdown_requested
        bot_mod.shutdown_requested = False
        mock_exch.return_value = MagicMock()

        try:
            _run_bot_main(
                config={'symbol': 'BTC/USDT:USDT', 'timeframe': '5m', 'leverage': 3},
                state_path='state.json',
                metrics_path='metrics.json',
            )
            mock_dll.assert_called()
            mock_sleep.assert_called()
        finally:
            bot_mod.shutdown_requested = original


class TestRunBotMainNetworkError:
    """Test network error in main loop → caught, retries."""

    @patch(f'{BOT}.save_metrics')
    @patch(f'{BOT}.save_state')
    @patch(f'{BOT}._interruptible_sleep', side_effect=_shutdown_after_one_iter)
    @patch(f'{BOT}.check_daily_loss_limit', side_effect=__import__('ccxt').NetworkError('timeout'))
    @patch(f'{BOT}.adjust_tpsl_to_config')
    @patch(f'{BOT}.recover_from_crash')
    @patch(f'{BOT}._verify_exchange_settings')
    @patch(f'{BOT}.sync_metrics_with_state', return_value=PerformanceMetrics())
    @patch(f'{BOT}.load_state', return_value={'position': None, 'daily_pnl': 0, 'daily_trades': 0})
    @patch(f'{BOT}.load_metrics', return_value=PerformanceMetrics())
    @patch(f'{BOT}.create_exchange')
    def test_network_error_caught(
        self, mock_exch, mock_lm, mock_ls, mock_sync_m, mock_ve, mock_rc, mock_adj,
        mock_dll, mock_sleep, mock_ss, mock_sm
    ):
        """NetworkError in loop body → caught, sleep, shutdown."""
        import bingx_rl_trading_bot.scripts.production.pattern_5m.bot as bot_mod
        original = bot_mod.shutdown_requested
        bot_mod.shutdown_requested = False
        mock_exch.return_value = MagicMock()

        try:
            _run_bot_main(
                config={'symbol': 'BTC/USDT:USDT', 'timeframe': '5m', 'leverage': 3},
                state_path='state.json',
                metrics_path='metrics.json',
            )
            mock_sleep.assert_called()
        finally:
            bot_mod.shutdown_requested = original


class TestRunBotMainExchangeError:
    """Test ExchangeError in main loop."""

    @patch(f'{BOT}.save_metrics')
    @patch(f'{BOT}.save_state')
    @patch(f'{BOT}._interruptible_sleep', side_effect=_shutdown_after_one_iter)
    @patch(f'{BOT}.check_daily_loss_limit', side_effect=__import__('ccxt').ExchangeError('invalid'))
    @patch(f'{BOT}.adjust_tpsl_to_config')
    @patch(f'{BOT}.recover_from_crash')
    @patch(f'{BOT}._verify_exchange_settings')
    @patch(f'{BOT}.sync_metrics_with_state', return_value=PerformanceMetrics())
    @patch(f'{BOT}.load_state', return_value={'position': None, 'daily_pnl': 0, 'daily_trades': 0})
    @patch(f'{BOT}.load_metrics', return_value=PerformanceMetrics())
    @patch(f'{BOT}.create_exchange')
    def test_exchange_error_caught(
        self, mock_exch, mock_lm, mock_ls, mock_sync_m, mock_ve, mock_rc, mock_adj,
        mock_dll, mock_sleep, mock_ss, mock_sm
    ):
        """ExchangeError → caught, sleep, shutdown."""
        import bingx_rl_trading_bot.scripts.production.pattern_5m.bot as bot_mod
        original = bot_mod.shutdown_requested
        bot_mod.shutdown_requested = False
        mock_exch.return_value = MagicMock()

        try:
            _run_bot_main(
                config={'symbol': 'BTC/USDT:USDT', 'timeframe': '5m', 'leverage': 3},
                state_path='state.json',
                metrics_path='metrics.json',
            )
            mock_sleep.assert_called()
        finally:
            bot_mod.shutdown_requested = original


class TestRunBotMainGenericException:
    """Test generic Exception in main loop."""

    @patch(f'{BOT}.save_metrics')
    @patch(f'{BOT}.save_state')
    @patch(f'{BOT}._interruptible_sleep', side_effect=_shutdown_after_one_iter)
    @patch(f'{BOT}.check_daily_loss_limit', side_effect=RuntimeError('unexpected'))
    @patch(f'{BOT}.adjust_tpsl_to_config')
    @patch(f'{BOT}.recover_from_crash')
    @patch(f'{BOT}._verify_exchange_settings')
    @patch(f'{BOT}.sync_metrics_with_state', return_value=PerformanceMetrics())
    @patch(f'{BOT}.load_state', return_value={'position': None, 'daily_pnl': 0, 'daily_trades': 0})
    @patch(f'{BOT}.load_metrics', return_value=PerformanceMetrics())
    @patch(f'{BOT}.create_exchange')
    def test_generic_exception_caught(
        self, mock_exch, mock_lm, mock_ls, mock_sync_m, mock_ve, mock_rc, mock_adj,
        mock_dll, mock_sleep, mock_ss, mock_sm
    ):
        """Generic Exception → caught, sleep, shutdown."""
        import bingx_rl_trading_bot.scripts.production.pattern_5m.bot as bot_mod
        original = bot_mod.shutdown_requested
        bot_mod.shutdown_requested = False
        mock_exch.return_value = MagicMock()

        try:
            _run_bot_main(
                config={'symbol': 'BTC/USDT:USDT', 'timeframe': '5m', 'leverage': 3},
                state_path='state.json',
                metrics_path='metrics.json',
            )
            mock_sleep.assert_called()
        finally:
            bot_mod.shutdown_requested = original


class TestRunBotMainNewMetrics:
    """Test _run_bot_main when load_metrics returns None (new metrics created)."""

    @patch(f'{BOT}.save_metrics')
    @patch(f'{BOT}.save_state')
    @patch(f'{BOT}._interruptible_sleep', side_effect=_shutdown_after_one_iter)
    @patch(f'{BOT}._calculate_sleep_duration', return_value=1.0)
    @patch(f'{BOT}._log_waiting_status')
    @patch(f'{BOT}._maybe_run_health_check', return_value=0.0)
    @patch(f'{BOT}._maybe_sync_position', return_value=0.0)
    @patch(f'{BOT}._get_candle_timing')
    @patch(f'{BOT}.check_daily_loss_limit', return_value=False)
    @patch(f'{BOT}.adjust_tpsl_to_config')
    @patch(f'{BOT}.recover_from_crash')
    @patch(f'{BOT}._verify_exchange_settings')
    @patch(f'{BOT}.sync_metrics_with_state')
    @patch(f'{BOT}.load_state', return_value={'position': None, 'daily_pnl': 0, 'daily_trades': 0})
    @patch(f'{BOT}.load_metrics', return_value=None)
    @patch(f'{BOT}.create_exchange')
    def test_none_metrics_creates_new(
        self, mock_exch, mock_lm, mock_ls, mock_sync_m, mock_ve, mock_rc, mock_adj,
        mock_dll, mock_timing, mock_msync, mock_mhc, mock_lws,
        mock_csd, mock_sleep, mock_ss, mock_sm
    ):
        """load_metrics returns None → new PerformanceMetrics created (lines 243-245)."""
        import bingx_rl_trading_bot.scripts.production.pattern_5m.bot as bot_mod
        original = bot_mod.shutdown_requested
        bot_mod.shutdown_requested = False
        mock_exch.return_value = MagicMock()
        mock_sync_m.side_effect = lambda m, s: m  # pass through

        mock_timing.return_value = CandleTiming(
            in_trading_window=False, candle_id=1,
            seconds_into_candle=60.0, seconds_until_close=240.0,
        )

        try:
            _run_bot_main(
                config={'symbol': 'BTC/USDT:USDT', 'timeframe': '5m', 'leverage': 3},
                state_path='state.json',
                metrics_path='metrics.json',
            )
            # sync_metrics_with_state was called with a PerformanceMetrics instance
            mock_sync_m.assert_called_once()
            args = mock_sync_m.call_args[0]
            assert isinstance(args[0], PerformanceMetrics)
        finally:
            bot_mod.shutdown_requested = original


class TestRunBotMainMaintenanceWithPosition:
    """Test maintenance window with position — TP/SL verify + log status."""

    @patch(f'{BOT}.save_metrics')
    @patch(f'{BOT}.save_state')
    @patch(f'{BOT}._interruptible_sleep', side_effect=_shutdown_after_one_iter)
    @patch(f'{BOT}._calculate_sleep_duration', return_value=1.0)
    @patch(f'{BOT}._log_position_status')
    @patch(f'{BOT}.verify_tp_sl_orders')
    @patch(f'{BOT}._maybe_run_health_check', return_value=0.0)
    @patch(f'{BOT}._maybe_sync_position', return_value=0.0)
    @patch(f'{BOT}.check_position_status', return_value=False)
    @patch(f'{BOT}.check_daily_loss_limit', return_value=False)
    @patch(f'{BOT}._get_candle_timing')
    @patch(f'{BOT}.adjust_tpsl_to_config')
    @patch(f'{BOT}.recover_from_crash')
    @patch(f'{BOT}._verify_exchange_settings')
    @patch(f'{BOT}.sync_metrics_with_state', return_value=PerformanceMetrics())
    @patch(f'{BOT}.load_state', return_value={
        'positions': {'s1': {'slot_id': 's1', 'direction': 'LONG', 'entry_price': 50000}},
        'active_direction': 'LONG',
        'daily_pnl': 0, 'daily_trades': 0,
    })
    @patch(f'{BOT}.load_metrics', return_value=PerformanceMetrics())
    @patch(f'{BOT}.create_exchange')
    def test_maintenance_with_position_verifies_tpsl(
        self, mock_exch, mock_lm, mock_ls, mock_sync_m, mock_ve, mock_rc, mock_adj,
        mock_timing, mock_dll, mock_cps, mock_msync, mock_mhc,
        mock_vtpsl, mock_lps, mock_csd, mock_sleep, mock_ss, mock_sm
    ):
        """Maintenance + position → verify_tp_sl_orders + _log_position_status called."""
        import bingx_rl_trading_bot.scripts.production.pattern_5m.bot as bot_mod
        original = bot_mod.shutdown_requested
        bot_mod.shutdown_requested = False
        mock_exch.return_value = MagicMock()

        mock_timing.return_value = CandleTiming(
            in_trading_window=False, candle_id=100,
            seconds_into_candle=60.0, seconds_until_close=240.0,
        )

        try:
            _run_bot_main(
                config={'symbol': 'BTC/USDT:USDT', 'timeframe': '5m', 'leverage': 3},
                state_path='state.json',
                metrics_path='metrics.json',
            )
            mock_vtpsl.assert_called_once()
            mock_lps.assert_called_once()
        finally:
            bot_mod.shutdown_requested = original


# ── run_bot entry point ────────────────────────────


class TestRunBotMainMaintenancePositionClosed:
    """Test maintenance window where position gets closed (line 326)."""

    @patch(f'{BOT}.save_metrics')
    @patch(f'{BOT}.save_state')
    @patch(f'{BOT}._interruptible_sleep', side_effect=_shutdown_after_one_iter)
    @patch(f'{BOT}._calculate_sleep_duration', return_value=1.0)
    @patch(f'{BOT}._log_waiting_status')
    @patch(f'{BOT}._maybe_run_health_check', return_value=0.0)
    @patch(f'{BOT}._maybe_sync_position', return_value=0.0)
    @patch(f'{BOT}.check_position_status', return_value=True)
    @patch(f'{BOT}.check_daily_loss_limit', return_value=False)
    @patch(f'{BOT}._get_candle_timing')
    @patch(f'{BOT}.adjust_tpsl_to_config')
    @patch(f'{BOT}.recover_from_crash')
    @patch(f'{BOT}._verify_exchange_settings')
    @patch(f'{BOT}.sync_metrics_with_state', return_value=PerformanceMetrics())
    @patch(f'{BOT}.load_state')
    @patch(f'{BOT}.load_metrics', return_value=PerformanceMetrics())
    @patch(f'{BOT}.create_exchange')
    def test_maintenance_position_closed_updates_has_position(
        self, mock_exch, mock_lm, mock_ls, mock_sync_m, mock_ve, mock_rc, mock_adj,
        mock_timing, mock_dll, mock_cps, mock_msync, mock_mhc,
        mock_lws, mock_csd, mock_sleep, mock_ss, mock_sm
    ):
        """Maintenance + position closed → has_position updated (line 326)."""
        import bingx_rl_trading_bot.scripts.production.pattern_5m.bot as bot_mod
        original = bot_mod.shutdown_requested
        bot_mod.shutdown_requested = False
        mock_exch.return_value = MagicMock()

        # State starts with positions, then check_position_status returns True (closed),
        # and state['positions'] is cleared by the mock side effect
        state_dict = {
            'positions': {'s1': {'slot_id': 's1', 'direction': 'LONG', 'entry_price': 50000}},
            'active_direction': 'LONG',
            'daily_pnl': 0, 'daily_trades': 0,
        }

        def cps_side_effect(*args, **kwargs):
            # Simulate position being closed — clear positions from state
            state_dict['positions'] = {}
            state_dict['active_direction'] = None
            return True

        mock_ls.return_value = state_dict
        mock_cps.side_effect = cps_side_effect

        mock_timing.return_value = CandleTiming(
            in_trading_window=False, candle_id=100,
            seconds_into_candle=60.0, seconds_until_close=240.0,
        )

        try:
            _run_bot_main(
                config={'symbol': 'BTC/USDT:USDT', 'timeframe': '5m', 'leverage': 3},
                state_path='state.json',
                metrics_path='metrics.json',
            )
            mock_cps.assert_called_once()
            # _log_waiting_status called (not _log_position_status) because position was cleared
            mock_lws.assert_called()
        finally:
            bot_mod.shutdown_requested = original


class TestRunBotEntryPoint:
    """Test run_bot() entry point (lines 169-217)."""

    @patch(f'{BOT}._run_bot_main')
    @patch(f'{BOT}.set_shutdown_checker')
    @patch(f'{BOT}.signal.signal')
    @patch(f'{BOT}.acquire_lock', return_value=True)
    @patch(f'{BOT}.release_lock')
    @patch(f'{BOT}._log_pattern_summary')
    @patch(f'{BOT}.setup_logging')
    @patch(f'{BOT}.validate_config')
    @patch(f'{BOT}.load_dynamic_patterns', side_effect=lambda c: c)
    @patch(f'{BOT}.load_config')
    @patch(f'{BOT}.os.chdir')
    def test_normal_run(
        self, mock_chdir, mock_lc, mock_ldp, mock_vc, mock_sl, mock_lps,
        mock_rl, mock_al, mock_sig, mock_ssc, mock_rbm
    ):
        """Normal run_bot flow: config → lock → run → release."""
        mock_lc.return_value = {
            'symbol': 'BTC/USDT:USDT', 'timeframe': '5m', 'leverage': 3,
            'debug_mode': False, 'json_logging': False,
        }
        run_bot('test_config.yaml')
        mock_al.assert_called_once()
        mock_rbm.assert_called_once()
        mock_rl.assert_called_once()

    @patch(f'{BOT}.sys.exit', side_effect=SystemExit(1))
    @patch(f'{BOT}.acquire_lock', return_value=False)
    @patch(f'{BOT}._log_pattern_summary')
    @patch(f'{BOT}.setup_logging')
    @patch(f'{BOT}.validate_config')
    @patch(f'{BOT}.load_dynamic_patterns', side_effect=lambda c: c)
    @patch(f'{BOT}.load_config')
    @patch(f'{BOT}.os.chdir')
    def test_lock_failure_exits(
        self, mock_chdir, mock_lc, mock_ldp, mock_vc, mock_sl, mock_lps,
        mock_al, mock_exit
    ):
        """Failed lock acquisition → sys.exit(1) (line 199)."""
        mock_lc.return_value = {
            'symbol': 'BTC/USDT:USDT', 'timeframe': '5m', 'leverage': 3,
            'debug_mode': False, 'json_logging': False,
        }
        with pytest.raises(SystemExit):
            run_bot('test_config.yaml')
        mock_exit.assert_called_once_with(1)

    @patch(f'{BOT}.release_lock')
    @patch(f'{BOT}._run_bot_main', side_effect=KeyboardInterrupt())
    @patch(f'{BOT}.set_shutdown_checker')
    @patch(f'{BOT}.signal.signal')
    @patch(f'{BOT}.acquire_lock', return_value=True)
    @patch(f'{BOT}._log_pattern_summary')
    @patch(f'{BOT}.setup_logging')
    @patch(f'{BOT}.validate_config')
    @patch(f'{BOT}.load_dynamic_patterns', side_effect=lambda c: c)
    @patch(f'{BOT}.load_config')
    @patch(f'{BOT}.os.chdir')
    def test_keyboard_interrupt_releases_lock(
        self, mock_chdir, mock_lc, mock_ldp, mock_vc, mock_sl, mock_lps,
        mock_al, mock_sig, mock_ssc, mock_rbm, mock_rl
    ):
        """KeyboardInterrupt → caught, lock released (lines 210-217)."""
        mock_lc.return_value = {
            'symbol': 'BTC/USDT:USDT', 'timeframe': '5m', 'leverage': 3,
            'debug_mode': False, 'json_logging': False,
        }
        run_bot('test_config.yaml')
        mock_rl.assert_called_once()

    @patch(f'{BOT}.release_lock')
    @patch(f'{BOT}._run_bot_main', side_effect=RuntimeError('fatal'))
    @patch(f'{BOT}.set_shutdown_checker')
    @patch(f'{BOT}.signal.signal')
    @patch(f'{BOT}.acquire_lock', return_value=True)
    @patch(f'{BOT}._log_pattern_summary')
    @patch(f'{BOT}.setup_logging')
    @patch(f'{BOT}.validate_config')
    @patch(f'{BOT}.load_dynamic_patterns', side_effect=lambda c: c)
    @patch(f'{BOT}.load_config')
    @patch(f'{BOT}.os.chdir')
    def test_fatal_error_releases_lock_and_reraises(
        self, mock_chdir, mock_lc, mock_ldp, mock_vc, mock_sl, mock_lps,
        mock_al, mock_sig, mock_ssc, mock_rbm, mock_rl
    ):
        """Fatal error → lock released, exception re-raised (lines 212-214)."""
        mock_lc.return_value = {
            'symbol': 'BTC/USDT:USDT', 'timeframe': '5m', 'leverage': 3,
            'debug_mode': False, 'json_logging': False,
        }
        with pytest.raises(RuntimeError, match='fatal'):
            run_bot('test_config.yaml')
        mock_rl.assert_called_once()
