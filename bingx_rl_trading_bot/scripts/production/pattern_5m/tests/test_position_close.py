"""Tests for position_close.py — trade recording and state management.

Uses mock objects to avoid real exchange calls.
"""

import pytest
import ccxt
from unittest.mock import MagicMock, patch

from bingx_rl_trading_bot.scripts.production.pattern_5m.position_close import (
    record_closed_position,
    close_position_market,
    recover_position_to_state,
    recover_from_crash,
    _read_tpsl_from_exchange_orders,
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


# ── _read_tpsl_from_exchange_orders ────────────────────────


class TestReadTpslFromExchangeOrders:
    """Test _read_tpsl_from_exchange_orders() parsing logic."""

    def test_no_exchange_returns_none(self):
        """exchange=None → (None, None)."""
        tp, sl = _read_tpsl_from_exchange_orders(None, 'BTC/USDT:USDT', 'LONG')
        assert tp is None and sl is None

    def test_both_orders_found(self):
        """TP + SL orders on exchange → both prices returned."""
        exchange = MagicMock()
        exchange.fetch_open_orders.return_value = [
            {'type': 'TAKE_PROFIT_MARKET', 'stopPrice': 51000.0, 'info': {}},
            {'type': 'STOP_MARKET', 'stopPrice': 49000.0, 'info': {}},
        ]
        tp, sl = _read_tpsl_from_exchange_orders(exchange, 'BTC/USDT:USDT', 'LONG')
        assert tp == 51000.0
        assert sl == 49000.0

    def test_only_tp_found(self):
        """Only TP order → tp returned, sl is None."""
        exchange = MagicMock()
        exchange.fetch_open_orders.return_value = [
            {'type': 'TAKE_PROFIT_MARKET', 'stopPrice': 51000.0, 'info': {}},
        ]
        tp, sl = _read_tpsl_from_exchange_orders(exchange, 'BTC/USDT:USDT', 'LONG')
        assert tp == 51000.0
        assert sl is None

    def test_stop_price_from_info(self):
        """stopPrice missing at top level → reads from info dict."""
        exchange = MagicMock()
        exchange.fetch_open_orders.return_value = [
            {'type': 'STOP_MARKET', 'stopPrice': None, 'info': {'stopPrice': '49500.0'}},
        ]
        tp, sl = _read_tpsl_from_exchange_orders(exchange, 'BTC/USDT:USDT', 'LONG')
        assert tp is None
        assert sl == 49500.0

    def test_network_error_returns_none(self):
        """Network error → (None, None), no crash."""
        exchange = MagicMock()
        exchange.fetch_open_orders.side_effect = ccxt.NetworkError('timeout')
        tp, sl = _read_tpsl_from_exchange_orders(exchange, 'BTC/USDT:USDT', 'LONG')
        assert tp is None and sl is None

    def test_zero_stop_price_ignored(self):
        """stopPrice=0 → skipped."""
        exchange = MagicMock()
        exchange.fetch_open_orders.return_value = [
            {'type': 'STOP_MARKET', 'stopPrice': 0, 'info': {}},
        ]
        tp, sl = _read_tpsl_from_exchange_orders(exchange, 'BTC/USDT:USDT', 'LONG')
        assert tp is None and sl is None


# ── close_position_market ──────────────────────────────────


@patch('bingx_rl_trading_bot.scripts.production.pattern_5m.position_close.place_tp_sl_orders')
@patch('bingx_rl_trading_bot.scripts.production.pattern_5m.position_close.record_closed_position')
@patch('bingx_rl_trading_bot.scripts.production.pattern_5m.position_close.cancel_remaining_orders')
@patch('bingx_rl_trading_bot.scripts.production.pattern_5m.position_close.fetch_ticker_cached')
class TestClosePositionMarket:
    """Test close_position_market() market exit logic."""

    def _make_state(self):
        return {
            'position': {
                'direction': 'LONG',
                'quantity': 0.01,
                'remaining_quantity': 0.01,
            }
        }

    def test_no_position_returns_false(
        self, mock_ticker, mock_cancel, mock_record, mock_place
    ):
        """No position → False."""
        result = close_position_market(
            MagicMock(), {'position': None}, {'symbol': 'BTC/USDT:USDT'}, APICache()
        )
        assert result is False
        mock_cancel.assert_not_called()

    def test_successful_market_close(
        self, mock_ticker, mock_cancel, mock_record, mock_place
    ):
        """Normal close → cancel TP/SL, create market order, record."""
        exchange = MagicMock()
        exchange.create_order.return_value = {'average': 51000.0, 'id': 'mkt_1'}
        state = self._make_state()
        config = {'symbol': 'BTC/USDT:USDT'}
        result = close_position_market(exchange, state, config, APICache())
        assert result is True
        mock_cancel.assert_called_once()
        exchange.create_order.assert_called_once()
        mock_record.assert_called_once()

    def test_fallback_to_ticker_on_zero_fill(
        self, mock_ticker, mock_cancel, mock_record, mock_place
    ):
        """Fill price=0 → uses ticker fallback."""
        exchange = MagicMock()
        exchange.create_order.return_value = {'average': 0, 'price': 0, 'id': 'mkt_1'}
        mock_ticker.return_value = {'last': 50500.0}
        state = self._make_state()
        config = {'symbol': 'BTC/USDT:USDT'}
        result = close_position_market(exchange, state, config, APICache())
        assert result is True
        mock_ticker.assert_called_once()
        # record should be called with ticker price
        call_args = mock_record.call_args
        assert call_args[0][3] == 50500.0

    def test_network_error_replaces_tpsl(
        self, mock_ticker, mock_cancel, mock_record, mock_place
    ):
        """Market order fails → re-places TP/SL, returns False."""
        exchange = MagicMock()
        exchange.create_order.side_effect = ccxt.NetworkError('timeout')
        state = self._make_state()
        config = {'symbol': 'BTC/USDT:USDT'}
        result = close_position_market(exchange, state, config, APICache())
        assert result is False
        mock_place.assert_called_once()  # TP/SL re-placed
        mock_record.assert_not_called()

    def test_zero_quantity_returns_false(
        self, mock_ticker, mock_cancel, mock_record, mock_place
    ):
        """Quantity=0 → False."""
        state = {
            'position': {
                'direction': 'LONG',
                'quantity': 0,
                'remaining_quantity': 0,
            }
        }
        result = close_position_market(
            MagicMock(), state, {'symbol': 'BTC/USDT:USDT'}, APICache()
        )
        assert result is False


# ── recover_position_to_state ──────────────────────────────


@patch('bingx_rl_trading_bot.scripts.production.pattern_5m.position_close.place_tp_sl_orders')
@patch('bingx_rl_trading_bot.scripts.production.pattern_5m.position_close.save_state')
@patch('bingx_rl_trading_bot.scripts.production.pattern_5m.position_close._read_tpsl_from_exchange_orders')
@patch('bingx_rl_trading_bot.scripts.production.pattern_5m.position_close.calculate_tp_sl')
@patch('bingx_rl_trading_bot.scripts.production.pattern_5m.position_close.setup_scale_out')
class TestRecoverPositionToState:
    """Test recover_position_to_state() exchange → state recovery."""

    def test_basic_recovery_with_exchange_tpsl(
        self, mock_scale, mock_calc, mock_read, mock_save, mock_place
    ):
        """Exchange has TP/SL → uses them directly."""
        mock_read.return_value = (51000.0, 49000.0)
        mock_scale.return_value = []
        state = {'position': None}
        config = {'strategy': {}, 'symbol': 'BTC/USDT:USDT'}
        exchange_pos = {'entryPrice': 50000.0, 'contracts': 0.01}
        exchange = MagicMock()

        recover_position_to_state(state, config, exchange_pos, 'LONG', exchange, APICache())

        pos = state['position']
        assert pos['direction'] == 'LONG'
        assert pos['entry_price'] == 50000.0
        assert pos['tp_price'] == 51000.0
        assert pos['sl_price'] == 49000.0
        assert pos['recovered'] is True
        mock_calc.assert_not_called()  # Should not calculate — read from exchange
        mock_save.assert_called()

    def test_fallback_to_config_when_no_exchange_tpsl(
        self, mock_scale, mock_calc, mock_read, mock_save, mock_place
    ):
        """No TP/SL on exchange → calculate from config."""
        mock_read.return_value = (None, None)
        mock_calc.return_value = (51500.0, 48500.0, 3.0, 3.0)
        mock_scale.return_value = []
        state = {'position': None}
        config = {'strategy': {}, 'symbol': 'BTC/USDT:USDT'}
        exchange_pos = {'entryPrice': 50000.0, 'contracts': 0.01}

        recover_position_to_state(state, config, exchange_pos, 'SHORT')

        pos = state['position']
        assert pos['direction'] == 'SHORT'
        assert pos['tp_price'] == 51500.0
        mock_calc.assert_called_once()
        mock_place.assert_not_called()  # No exchange provided

    def test_places_orders_when_exchange_provided(
        self, mock_scale, mock_calc, mock_read, mock_save, mock_place
    ):
        """exchange provided → place_tp_sl_orders called."""
        mock_read.return_value = (51000.0, 49000.0)
        mock_scale.return_value = []
        state = {'position': None}
        config = {'strategy': {}, 'symbol': 'BTC/USDT:USDT'}
        exchange_pos = {'entryPrice': 50000.0, 'contracts': 0.01}
        exchange = MagicMock()

        recover_position_to_state(state, config, exchange_pos, 'LONG', exchange)

        mock_place.assert_called_once()

    def test_preserves_pattern_from_old_state(
        self, mock_scale, mock_calc, mock_read, mock_save, mock_place
    ):
        """Old position has pattern → preserved in reason."""
        mock_read.return_value = (51000.0, 49000.0)
        mock_scale.return_value = []
        state = {'position': {'reason': 'Pattern: BD-BD-U (LONG)'}}
        config = {'strategy': {}, 'symbol': 'BTC/USDT:USDT'}
        exchange_pos = {'entryPrice': 50000.0, 'contracts': 0.01}

        recover_position_to_state(state, config, exchange_pos, 'LONG')

        pos = state['position']
        assert 'BD-BD-U' in pos['reason']
        assert pos['pattern_name'] == 'BD-BD-U'


# ── recover_from_crash ─────────────────────────────────────


@patch('bingx_rl_trading_bot.scripts.production.pattern_5m.position_close.recalculate_position_orders')
@patch('bingx_rl_trading_bot.scripts.production.pattern_5m.position_close.recover_position_to_state')
@patch('bingx_rl_trading_bot.scripts.production.pattern_5m.position_close.record_closed_position')
@patch('bingx_rl_trading_bot.scripts.production.pattern_5m.position_close.detect_ghost_positions')
@patch('bingx_rl_trading_bot.scripts.production.pattern_5m.position_close.fetch_positions_cached')
@patch('bingx_rl_trading_bot.scripts.production.pattern_5m.position_close.fetch_ticker_cached')
class TestRecoverFromCrash:
    """Test recover_from_crash() 4-case recovery."""

    def test_case1_orphan_exchange_position(
        self, mock_ticker, mock_fetch_pos, mock_ghost,
        mock_record, mock_recover, mock_recalc
    ):
        """Exchange has position, local doesn't → recover to state."""
        mock_fetch_pos.return_value = [
            {'side': 'long', 'contracts': 0.01, 'entryPrice': 50000.0},
        ]
        state = {'position': None}
        config = {'symbol': 'BTC/USDT:USDT', 'strategy': {}}
        exchange = MagicMock()

        result = recover_from_crash(exchange, state, config, APICache())
        assert result is True
        mock_recover.assert_called_once()
        mock_record.assert_not_called()

    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.position_monitor.get_actual_exit_price')
    def test_case2_local_only_with_trade_history(
        self, mock_exit, mock_ticker, mock_fetch_pos, mock_ghost,
        mock_record, mock_recover, mock_recalc
    ):
        """Local has position, exchange doesn't, trade history found → record closed."""
        mock_fetch_pos.return_value = []  # no exchange position
        mock_exit.return_value = {'price': 51000.0, 'reason': 'TP'}
        state = {
            'position': {
                'direction': 'LONG', 'entry_price': 50000.0,
                'quantity': 0.01, 'entry_time': '2026-02-18T10:00:00',
            }
        }
        config = {'symbol': 'BTC/USDT:USDT'}
        result = recover_from_crash(MagicMock(), state, config, APICache())
        assert result is True
        mock_record.assert_called_once()
        # Check exit price used
        assert mock_record.call_args[0][3] == 51000.0

    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.position_monitor.get_actual_exit_price')
    def test_case2_local_only_ticker_fallback(
        self, mock_exit, mock_ticker, mock_fetch_pos, mock_ghost,
        mock_record, mock_recover, mock_recalc
    ):
        """Local only, no trade history → uses ticker fallback."""
        mock_fetch_pos.return_value = []
        mock_exit.return_value = None
        mock_ticker.return_value = {'last': 50200.0}
        state = {
            'position': {
                'direction': 'LONG', 'entry_price': 50000.0,
                'quantity': 0.01, 'entry_time': '2026-02-18T10:00:00',
            }
        }
        config = {'symbol': 'BTC/USDT:USDT'}
        result = recover_from_crash(MagicMock(), state, config, APICache())
        assert result is True
        assert mock_record.call_args[0][3] == 50200.0

    def test_case3_direction_mismatch(
        self, mock_ticker, mock_fetch_pos, mock_ghost,
        mock_record, mock_recover, mock_recalc
    ):
        """Both exist, directions mismatch → close local + recover exchange."""
        mock_fetch_pos.return_value = [
            {'side': 'short', 'contracts': 0.01, 'entryPrice': 50000.0},
        ]
        state = {
            'position': {
                'direction': 'LONG', 'entry_price': 50000.0,
                'quantity': 0.01, 'entry_time': '2026-02-18T10:00:00',
            }
        }
        config = {'symbol': 'BTC/USDT:USDT', 'strategy': {}}
        exchange = MagicMock()

        with patch(
            'bingx_rl_trading_bot.scripts.production.pattern_5m.position_monitor.get_actual_exit_price',
            return_value={'price': 49500.0, 'reason': 'SL'},
        ):
            result = recover_from_crash(exchange, state, config, APICache())

        assert result is True
        mock_record.assert_called_once()  # close old
        mock_recover.assert_called_once()  # recover new

    def test_case4_quantity_mismatch(
        self, mock_ticker, mock_fetch_pos, mock_ghost,
        mock_record, mock_recover, mock_recalc
    ):
        """Both exist, same direction, qty differs → recalculate."""
        mock_fetch_pos.return_value = [
            {'side': 'long', 'contracts': 0.02, 'entryPrice': 50000.0},
        ]
        state = {
            'position': {
                'direction': 'LONG', 'entry_price': 50000.0,
                'quantity': 0.01,
            }
        }
        config = {'symbol': 'BTC/USDT:USDT'}
        result = recover_from_crash(MagicMock(), state, config, APICache())
        assert result is True
        mock_recalc.assert_called_once()

    def test_consistent_state_returns_false(
        self, mock_ticker, mock_fetch_pos, mock_ghost,
        mock_record, mock_recover, mock_recalc
    ):
        """Both exist, same direction, same qty → no action."""
        mock_fetch_pos.return_value = [
            {'side': 'long', 'contracts': 0.01, 'entryPrice': 50000.0},
        ]
        state = {
            'position': {
                'direction': 'LONG', 'entry_price': 50000.0,
                'quantity': 0.01,
                'scale_out_stages': [],
            }
        }
        config = {'symbol': 'BTC/USDT:USDT'}
        result = recover_from_crash(MagicMock(), state, config, APICache())
        assert result is False
        mock_record.assert_not_called()
        mock_recover.assert_not_called()

    def test_network_error_returns_false(
        self, mock_ticker, mock_fetch_pos, mock_ghost,
        mock_record, mock_recover, mock_recalc
    ):
        """Network error during fetch → False."""
        mock_fetch_pos.side_effect = ccxt.NetworkError('timeout')
        state = {'position': None}
        config = {'symbol': 'BTC/USDT:USDT'}
        result = recover_from_crash(MagicMock(), state, config, APICache())
        assert result is False

    def test_no_positions_anywhere_returns_false(
        self, mock_ticker, mock_fetch_pos, mock_ghost,
        mock_record, mock_recover, mock_recalc
    ):
        """No exchange position, no local → no action."""
        mock_fetch_pos.return_value = []
        state = {'position': None}
        config = {'symbol': 'BTC/USDT:USDT'}
        result = recover_from_crash(MagicMock(), state, config, APICache())
        assert result is False
