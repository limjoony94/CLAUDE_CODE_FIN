"""Tests for models.py — PerformanceMetrics, APICache, CircuitBreaker."""

import time
import pytest
from collections import deque

from bingx_rl_trading_bot.scripts.production.pattern_5m.models import (
    PerformanceMetrics,
    APICache,
    CircuitBreaker,
    BOT_STATE_REQUIRED_KEYS,
)
from bingx_rl_trading_bot.scripts.production.pattern_5m.constants import (
    METRICS_WINDOW_SIZE,
    MIN_TRADES_FOR_COMPARISON,
)


# ── PerformanceMetrics ────────────────────────────────────────


class TestPerformanceMetricsInit:
    """Test PerformanceMetrics default initialization."""

    def test_defaults(self):
        m = PerformanceMetrics()
        assert m.total_trades == 0
        assert m.winning_trades == 0
        assert m.losing_trades == 0
        assert m.total_pnl_pct == 0.0
        assert m.actual_win_rate == 0.0
        assert m.actual_edge == 0.0
        assert isinstance(m.recent_wins, deque)
        assert isinstance(m.recent_losses, deque)
        assert len(m.recent_wins) == 0
        assert len(m.recent_losses) == 0

    def test_deque_maxlen(self):
        m = PerformanceMetrics()
        assert m.recent_wins.maxlen == METRICS_WINDOW_SIZE
        assert m.recent_losses.maxlen == METRICS_WINDOW_SIZE


class TestUpdateTrade:
    """Test update_trade() method."""

    def test_single_win(self):
        m = PerformanceMetrics()
        m.update_trade(5.0)
        assert m.total_trades == 1
        assert m.winning_trades == 1
        assert m.losing_trades == 0
        assert m.total_pnl_pct == 5.0
        assert m.actual_win_rate == 100.0
        assert m.actual_avg_win == 5.0
        assert list(m.recent_wins) == [5.0]
        assert list(m.recent_losses) == []

    def test_single_loss(self):
        m = PerformanceMetrics()
        m.update_trade(-3.0)
        assert m.total_trades == 1
        assert m.winning_trades == 0
        assert m.losing_trades == 1
        assert m.total_pnl_pct == -3.0
        assert m.actual_win_rate == 0.0
        assert m.actual_avg_loss == 3.0  # stored as abs
        assert list(m.recent_losses) == [3.0]

    def test_zero_pnl_counts_as_win(self):
        m = PerformanceMetrics()
        m.update_trade(0.0)
        assert m.winning_trades == 1
        assert m.losing_trades == 0

    def test_mixed_trades(self):
        m = PerformanceMetrics()
        m.update_trade(6.0)
        m.update_trade(-4.0)
        m.update_trade(8.0)
        assert m.total_trades == 3
        assert m.winning_trades == 2
        assert m.losing_trades == 1
        assert m.total_pnl_pct == pytest.approx(10.0)
        assert m.actual_win_rate == pytest.approx(200 / 3)  # 66.67%

    def test_deque_overflow(self):
        """Deque should evict oldest when exceeding METRICS_WINDOW_SIZE."""
        m = PerformanceMetrics()
        for i in range(METRICS_WINDOW_SIZE + 5):
            m.update_trade(float(i + 1))
        assert len(m.recent_wins) == METRICS_WINDOW_SIZE
        # oldest 5 (1..5) evicted, newest is METRICS_WINDOW_SIZE+5
        assert m.recent_wins[0] == 6.0
        assert m.recent_wins[-1] == float(METRICS_WINDOW_SIZE + 5)


class TestEdgeCalculation:
    """Test _recalculate() edge formula: total_pnl_pct / total_trades."""

    def test_edge_below_min_trades(self):
        m = PerformanceMetrics()
        for _ in range(MIN_TRADES_FOR_COMPARISON - 1):
            m.update_trade(2.0)
        assert m.actual_edge == 0.0  # not enough trades

    def test_edge_at_min_trades(self):
        m = PerformanceMetrics()
        for _ in range(MIN_TRADES_FOR_COMPARISON):
            m.update_trade(2.0)
        expected_edge = (2.0 * MIN_TRADES_FOR_COMPARISON) / MIN_TRADES_FOR_COMPARISON
        assert m.actual_edge == pytest.approx(expected_edge)

    def test_edge_mixed_pnl(self):
        m = PerformanceMetrics()
        pnls = [5.0, -3.0, 4.0, -2.0, 6.0, -1.0]
        for p in pnls:
            m.update_trade(p)
        total = sum(pnls)
        assert m.actual_edge == pytest.approx(total / len(pnls))

    def test_edge_all_losses(self):
        m = PerformanceMetrics()
        for _ in range(MIN_TRADES_FOR_COMPARISON + 2):
            m.update_trade(-5.0)
        assert m.actual_edge < 0


class TestSerializationRoundtrip:
    """Test to_dict() / from_dict() roundtrip."""

    def test_empty_roundtrip(self):
        m = PerformanceMetrics()
        d = m.to_dict()
        m2 = PerformanceMetrics.from_dict(d)
        assert m2.total_trades == 0
        assert m2.total_pnl_pct == 0.0

    def test_populated_roundtrip(self):
        m = PerformanceMetrics()
        m.update_trade(5.0)
        m.update_trade(-3.0)
        m.update_trade(7.0)
        m.update_api_latency(150.0)
        m.session_start = "2026-01-01T00:00:00"

        d = m.to_dict()
        m2 = PerformanceMetrics.from_dict(d)

        assert m2.total_trades == m.total_trades
        assert m2.winning_trades == m.winning_trades
        assert m2.losing_trades == m.losing_trades
        assert m2.total_pnl_pct == pytest.approx(m.total_pnl_pct)
        assert m2.actual_win_rate == pytest.approx(m.actual_win_rate)
        assert m2.actual_avg_win == pytest.approx(m.actual_avg_win)
        assert m2.actual_avg_loss == pytest.approx(m.actual_avg_loss)
        assert m2.actual_edge == pytest.approx(m.actual_edge)
        assert m2.avg_api_latency_ms == pytest.approx(m.avg_api_latency_ms)
        assert m2.api_call_count == m.api_call_count
        assert list(m2.recent_wins) == list(m.recent_wins)
        assert list(m2.recent_losses) == list(m.recent_losses)
        assert m2.session_start == m.session_start

    def test_from_dict_missing_keys(self):
        """from_dict should handle missing keys with defaults."""
        m = PerformanceMetrics.from_dict({})
        assert m.total_trades == 0
        assert m.total_pnl_pct == 0.0
        assert m.session_start == ''

    def test_deque_maxlen_preserved(self):
        """from_dict should restore deque with correct maxlen."""
        m = PerformanceMetrics()
        m.update_trade(1.0)
        d = m.to_dict()
        m2 = PerformanceMetrics.from_dict(d)
        assert m2.recent_wins.maxlen == METRICS_WINDOW_SIZE


class TestComparisonReport:
    """Test get_comparison_report()."""

    def test_insufficient_trades(self):
        m = PerformanceMetrics()
        report = m.get_comparison_report()
        assert "Insufficient" in report

    def test_report_with_enough_trades(self):
        m = PerformanceMetrics()
        for _ in range(MIN_TRADES_FOR_COMPARISON + 1):
            m.update_trade(2.0)
        report = m.get_comparison_report()
        assert "BACKTEST vs ACTUAL" in report
        assert "Win Rate" in report
        assert "Edge/Trade" in report


class TestApiLatency:
    """Test update_api_latency() EMA."""

    def test_first_call(self):
        m = PerformanceMetrics()
        m.update_api_latency(200.0)
        assert m.api_call_count == 1
        # EMA: 0.1*200 + 0.9*0 = 20.0
        assert m.avg_api_latency_ms == pytest.approx(20.0)

    def test_convergence(self):
        m = PerformanceMetrics()
        for _ in range(100):
            m.update_api_latency(100.0)
        # Should converge toward 100
        assert m.avg_api_latency_ms == pytest.approx(100.0, abs=1.0)


# ── APICache ──────────────────────────────────────────────────


class TestAPICache:
    """Test APICache TTL and invalidation."""

    def test_set_get_ticker(self):
        c = APICache()
        c.set_ticker({"last": 50000})
        result = c.get_ticker(ttl=10.0)
        assert result == {"last": 50000}

    def test_ticker_expired(self):
        c = APICache()
        c.set_ticker({"last": 50000})
        c.ticker_time = time.time() - 100  # force expired
        result = c.get_ticker(ttl=10.0)
        assert result is None

    def test_set_get_balance(self):
        c = APICache()
        c.set_balance({"usdt": 1000})
        result = c.get_balance(ttl=10.0)
        assert result == {"usdt": 1000}

    def test_set_get_positions_empty(self):
        c = APICache()
        c.set_positions([])
        result = c.get_positions(ttl=10.0)
        assert result == []

    def test_positions_none_vs_empty(self):
        """positions=None (not set) should return None; positions=[] (empty) should return []."""
        c = APICache()
        assert c.get_positions(ttl=10.0) is None
        c.set_positions([])
        assert c.get_positions(ttl=10.0) == []

    def test_invalidate_all(self):
        c = APICache()
        c.set_ticker({"last": 50000})
        c.set_balance({"usdt": 1000})
        c.set_positions([{"symbol": "BTC"}])
        c.invalidate_all()
        assert c.get_ticker(ttl=10.0) is None
        assert c.get_balance(ttl=10.0) is None
        assert c.get_positions(ttl=10.0) is None


# ── CircuitBreaker ────────────────────────────────────────────


class TestCircuitBreaker:
    """Test CircuitBreaker state transitions and backoff."""

    def test_initial_state(self):
        cb = CircuitBreaker()
        assert cb.can_execute() is True
        assert cb.is_open is False
        assert cb.get_wait_time() == 0

    def test_opens_after_threshold(self):
        cb = CircuitBreaker()
        for _ in range(cb.failure_threshold):
            cb.record_failure()
        assert cb.is_open is True

    def test_below_threshold_stays_closed(self):
        cb = CircuitBreaker()
        for _ in range(cb.failure_threshold - 1):
            cb.record_failure()
        assert cb.is_open is False
        assert cb.can_execute() is True

    def test_success_resets(self):
        cb = CircuitBreaker()
        for _ in range(cb.failure_threshold):
            cb.record_failure()
        assert cb.is_open is True
        cb.record_success()
        assert cb.is_open is False
        assert cb.failure_count == 0
        assert cb.backoff_level == 0

    def test_backoff_level_increments(self):
        cb = CircuitBreaker()
        # First round: opens circuit
        for _ in range(cb.failure_threshold):
            cb.record_failure()
        assert cb.backoff_level == 0
        # Additional failure while open: increments backoff
        cb.record_failure()
        assert cb.backoff_level == 1

    def test_can_execute_after_timeout(self):
        cb = CircuitBreaker()
        for _ in range(cb.failure_threshold):
            cb.record_failure()
        # Simulate time passing beyond timeout
        cb.last_failure_time = time.time() - 1000
        assert cb.can_execute() is True
        assert cb.is_open is False

    def test_wait_time_positive_when_open(self):
        cb = CircuitBreaker()
        for _ in range(cb.failure_threshold):
            cb.record_failure()
        wait = cb.get_wait_time()
        assert wait > 0


# ── BotState Schema ───────────────────────────────────────────


class TestBotStateSchema:
    """Test BOT_STATE_REQUIRED_KEYS correctness."""

    def test_required_keys_set(self):
        expected = {
            'position', 'daily_pnl', 'daily_trades', 'consecutive_losses',
            'total_trades', 'total_pnl', 'winning_trades', 'last_trade_date',
        }
        assert BOT_STATE_REQUIRED_KEYS == expected
