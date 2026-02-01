"""Tests for bot.py — Early Exit triggers, main loop logic with exchange mock."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import Mock, MagicMock, patch
from datetime import datetime

from bingx_rl_trading_bot.scripts.production.pattern_5m.signals import check_early_exit_signal
from bingx_rl_trading_bot.scripts.production.pattern_5m.constants import CandleType
from bingx_rl_trading_bot.scripts.production.pattern_5m.models import PerformanceMetrics


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
    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.bot.fetch_ohlcv')
    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.bot.check_position_status')
    def test_process_position_calls_early_exit(
        self,
        mock_check_position,
        mock_fetch_ohlcv,
        mock_create_exchange,
        make_df_with_types,
        long_position,
        default_config
    ):
        """Main loop should check early exit when position exists."""
        from bingx_rl_trading_bot.scripts.production.pattern_5m.bot import _process_existing_position
        from bingx_rl_trading_bot.scripts.production.pattern_5m.models import APICache, CircuitBreaker

        # Setup mocks
        mock_exchange = Mock()
        mock_create_exchange.return_value = mock_exchange
        mock_check_position.return_value = False  # position still open

        # Mock OHLCV to return 3 BIG_DOWN candles
        df = make_df_with_types([CandleType.BIG_DOWN, CandleType.BIG_DOWN, CandleType.BIG_DOWN], n_bars=50)
        mock_fetch_ohlcv.return_value = df

        # Mock ticker for current price
        mock_exchange.fetch_ticker.return_value = {'last': 50200.0}

        state = {'position': long_position}
        cache = APICache()
        circuit_breaker = CircuitBreaker()
        metrics = PerformanceMetrics(session_start=datetime.now().isoformat())

        # Run the function (should trigger early exit logic)
        _process_existing_position(
            mock_exchange, state, default_config, cache, circuit_breaker, metrics, iteration=1
        )

        # Verify early exit was evaluated (check_position_status called)
        assert mock_check_position.called

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

    def test_position_sync_interval_timing(self):
        """Position sync should only trigger after sync_interval seconds."""
        from bingx_rl_trading_bot.scripts.production.pattern_5m.bot import _maybe_sync_position
        from bingx_rl_trading_bot.scripts.production.pattern_5m.models import APICache, CircuitBreaker
        import time

        mock_exchange = Mock()
        state = {'position': None}
        config = {'symbol': 'BTC-USDT'}
        cache = APICache()
        circuit_breaker = CircuitBreaker()
        metrics = PerformanceMetrics(session_start=datetime.now().isoformat())

        # First sync
        last_sync_time = time.time() - 3600  # 1 hour ago

        # Should sync (interval passed)
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
