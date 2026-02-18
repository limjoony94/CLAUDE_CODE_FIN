"""Tests for position_close.py — trade recording and state management.

Uses mock objects to avoid real exchange calls.
"""

import pytest
from unittest.mock import MagicMock, patch

from bingx_rl_trading_bot.scripts.production.pattern_5m.position_close import (
    record_closed_position,
)
from bingx_rl_trading_bot.scripts.production.pattern_5m.models import (
    APICache,
    PerformanceMetrics,
)


@patch('bingx_rl_trading_bot.scripts.production.pattern_5m.position_close._update_confidence_log_outcome')
@patch('bingx_rl_trading_bot.scripts.production.pattern_5m.position_close.save_metrics')
@patch('bingx_rl_trading_bot.scripts.production.pattern_5m.position_close.save_state')
@patch('bingx_rl_trading_bot.scripts.production.pattern_5m.position_close.cancel_remaining_orders')
class TestRecordClosedPosition:
    """Test record_closed_position() state update logic."""

    @pytest.fixture
    def base_state(self):
        return {
            'position': {
                'direction': 'LONG',
                'entry_price': 50000.0,
                'quantity': 0.01,
                'tp_price': 51000.0,
                'sl_price': 49000.0,
                'reason': 'pattern: BU-BU-DN SHORT',
                'entry_time': '2026-02-18T10:00:00',
            },
            'total_trades': 5,
            'winning_trades': 3,
            'total_pnl': 10.0,
            'daily_trades': 2,
            'daily_pnl': 5.0,
            'consecutive_losses': 0,
            'last_trade': None,
            'last_signal_time': None,
        }

    @pytest.fixture
    def config(self):
        return {
            'symbol': 'BTC/USDT:USDT',
            'leverage': 3,
        }

    def test_no_position_returns_early(
        self, mock_cancel, mock_save_state, mock_save_metrics, mock_conf_log
    ):
        """No position in state → returns immediately."""
        state = {'position': None, 'total_trades': 0}
        record_closed_position(
            MagicMock(), state, {'leverage': 3}, 50000.0, 'TP', APICache()
        )
        mock_save_state.assert_not_called()

    def test_winning_trade_updates_state(
        self, mock_cancel, mock_save_state, mock_save_metrics, mock_conf_log,
        base_state, config
    ):
        """Winning trade → total_trades+1, winning_trades+1, pnl positive."""
        exit_price = 51000.0  # above entry for LONG = profit
        record_closed_position(
            MagicMock(), base_state, config, exit_price, 'TP', APICache()
        )
        assert base_state['total_trades'] == 6
        assert base_state['winning_trades'] == 4
        assert base_state['total_pnl'] > 10.0
        assert base_state['consecutive_losses'] == 0
        assert base_state['position'] is None
        mock_save_state.assert_called_once()

    def test_losing_trade_increments_consecutive_losses(
        self, mock_cancel, mock_save_state, mock_save_metrics, mock_conf_log,
        base_state, config
    ):
        """Losing trade → consecutive_losses incremented."""
        exit_price = 49000.0  # below entry for LONG = loss
        record_closed_position(
            MagicMock(), base_state, config, exit_price, 'SL', APICache()
        )
        assert base_state['total_trades'] == 6
        assert base_state['winning_trades'] == 3  # unchanged
        assert base_state['consecutive_losses'] == 1
        assert base_state['total_pnl'] < 10.0

    def test_consecutive_losses_accumulate(
        self, mock_cancel, mock_save_state, mock_save_metrics, mock_conf_log,
        base_state, config
    ):
        """Multiple losses → consecutive_losses keeps incrementing."""
        base_state['consecutive_losses'] = 2
        exit_price = 49000.0
        record_closed_position(
            MagicMock(), base_state, config, exit_price, 'SL', APICache()
        )
        assert base_state['consecutive_losses'] == 3

    def test_win_resets_consecutive_losses(
        self, mock_cancel, mock_save_state, mock_save_metrics, mock_conf_log,
        base_state, config
    ):
        """Win after losses → consecutive_losses resets to 0."""
        base_state['consecutive_losses'] = 3
        exit_price = 51000.0
        record_closed_position(
            MagicMock(), base_state, config, exit_price, 'TP', APICache()
        )
        assert base_state['consecutive_losses'] == 0

    def test_invalid_entry_price_pnl_zero(
        self, mock_cancel, mock_save_state, mock_save_metrics, mock_conf_log,
        base_state, config
    ):
        """Invalid entry_price (0) → pnl recorded as 0%."""
        base_state['position']['entry_price'] = 0
        record_closed_position(
            MagicMock(), base_state, config, 50000.0, 'MARKET', APICache()
        )
        # PnL should be 0, total_pnl unchanged
        assert base_state['total_pnl'] == 10.0
        assert base_state['total_trades'] == 6

    def test_position_cleared_after_recording(
        self, mock_cancel, mock_save_state, mock_save_metrics, mock_conf_log,
        base_state, config
    ):
        """Position set to None after recording."""
        record_closed_position(
            MagicMock(), base_state, config, 51000.0, 'TP', APICache()
        )
        assert base_state['position'] is None

    def test_last_trade_recorded(
        self, mock_cancel, mock_save_state, mock_save_metrics, mock_conf_log,
        base_state, config
    ):
        """last_trade dict populated with trade details."""
        record_closed_position(
            MagicMock(), base_state, config, 51000.0, 'TP', APICache()
        )
        lt = base_state['last_trade']
        assert lt['direction'] == 'LONG'
        assert lt['entry_price'] == 50000.0
        assert lt['exit_price'] == 51000.0
        assert lt['exit_reason'] == 'TP'
        assert 'pnl_pct' in lt
        assert 'closed_at' in lt

    def test_metrics_updated_when_provided(
        self, mock_cancel, mock_save_state, mock_save_metrics, mock_conf_log,
        base_state, config
    ):
        """PerformanceMetrics.update_trade() called + save_metrics called."""
        metrics = PerformanceMetrics()
        record_closed_position(
            MagicMock(), base_state, config, 51000.0, 'TP', APICache(),
            metrics=metrics,
        )
        assert metrics.total_trades == 1
        mock_save_metrics.assert_called_once_with(metrics)

    def test_cancel_orders_called_with_exchange(
        self, mock_cancel, mock_save_state, mock_save_metrics, mock_conf_log,
        base_state, config
    ):
        """cancel_remaining_orders called when exchange is provided."""
        exchange = MagicMock()
        record_closed_position(
            exchange, base_state, config, 51000.0, 'TP', APICache()
        )
        mock_cancel.assert_called_once()

    def test_no_cancel_when_exchange_none(
        self, mock_cancel, mock_save_state, mock_save_metrics, mock_conf_log,
        base_state, config
    ):
        """exchange=None → cancel_remaining_orders not called."""
        record_closed_position(
            None, base_state, config, 51000.0, 'TP', APICache()
        )
        mock_cancel.assert_not_called()

    def test_short_winning_trade(
        self, mock_cancel, mock_save_state, mock_save_metrics, mock_conf_log,
        base_state, config
    ):
        """SHORT winning trade (exit < entry) → positive PnL."""
        base_state['position']['direction'] = 'SHORT'
        base_state['position']['entry_price'] = 50000.0
        exit_price = 49000.0  # below entry for SHORT = profit
        record_closed_position(
            MagicMock(), base_state, config, exit_price, 'TP', APICache()
        )
        assert base_state['total_pnl'] > 10.0
        assert base_state['winning_trades'] == 4

    def test_daily_stats_updated(
        self, mock_cancel, mock_save_state, mock_save_metrics, mock_conf_log,
        base_state, config
    ):
        """daily_trades and daily_pnl updated."""
        record_closed_position(
            MagicMock(), base_state, config, 51000.0, 'TP', APICache()
        )
        assert base_state['daily_trades'] == 3
        assert base_state['daily_pnl'] > 5.0
