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
    detect_ghost_positions,
    recalculate_position_orders,
)
from bingx_rl_trading_bot.scripts.production.pattern_5m.models import (
    APICache,
    PerformanceMetrics,
)


@patch('bingx_rl_trading_bot.scripts.production.pattern_5m.position_close._update_confidence_log_outcome')
@patch('bingx_rl_trading_bot.scripts.production.pattern_5m.position_close.save_metrics')
@patch('bingx_rl_trading_bot.scripts.production.pattern_5m.position_close.save_state')
@patch('bingx_rl_trading_bot.scripts.production.pattern_5m.position_close.cancel_remaining_orders')
@patch('bingx_rl_trading_bot.scripts.production.pattern_5m.position_close.cancel_emergency_sl')
@patch('bingx_rl_trading_bot.scripts.production.pattern_5m.position_close.update_emergency_sl')
class TestRecordClosedPosition:
    """Test record_closed_position() state update logic."""

    @pytest.fixture
    def base_state(self):
        return {
            'positions': {
                's1': {
                    'slot_id': 's1',
                    'direction': 'LONG',
                    'entry_price': 50000.0,
                    'quantity': 0.01,
                    'tp_price': 51000.0,
                    'sl_price': 49000.0,
                    'reason': 'pattern: BU-BU-DN SHORT',
                    'entry_time': '2026-02-18T10:00:00',
                },
            },
            'active_direction': 'LONG',
            'has_position': True,
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
        self, mock_update_esl, mock_cancel_esl,
        mock_cancel, mock_save_state, mock_save_metrics, mock_conf_log
    ):
        """No position in state -> returns immediately."""
        state = {'positions': {}, 'active_direction': None, 'total_trades': 0}
        record_closed_position(
            MagicMock(), state, {'leverage': 3}, 50000.0, 'TP', APICache()
        )
        mock_save_state.assert_not_called()

    def test_winning_trade_updates_state(
        self, mock_update_esl, mock_cancel_esl,
        mock_cancel, mock_save_state, mock_save_metrics, mock_conf_log,
        base_state, config
    ):
        """Winning trade -> total_trades+1, winning_trades+1, pnl positive."""
        exit_price = 51000.0  # above entry for LONG = profit
        record_closed_position(
            MagicMock(), base_state, config, exit_price, 'TP', APICache()
        )
        assert base_state['total_trades'] == 6
        assert base_state['winning_trades'] == 4
        assert base_state['total_pnl'] > 10.0
        assert base_state['consecutive_losses'] == 0
        assert len(base_state['positions']) == 0
        assert base_state['active_direction'] is None
        mock_save_state.assert_called_once()

    def test_losing_trade_increments_consecutive_losses(
        self, mock_update_esl, mock_cancel_esl,
        mock_cancel, mock_save_state, mock_save_metrics, mock_conf_log,
        base_state, config
    ):
        """Losing trade -> consecutive_losses incremented."""
        exit_price = 49000.0  # below entry for LONG = loss
        record_closed_position(
            MagicMock(), base_state, config, exit_price, 'SL', APICache()
        )
        assert base_state['total_trades'] == 6
        assert base_state['winning_trades'] == 3  # unchanged
        assert base_state['consecutive_losses'] == 1
        assert base_state['total_pnl'] < 10.0

    def test_consecutive_losses_accumulate(
        self, mock_update_esl, mock_cancel_esl,
        mock_cancel, mock_save_state, mock_save_metrics, mock_conf_log,
        base_state, config
    ):
        """Multiple losses -> consecutive_losses keeps incrementing."""
        base_state['consecutive_losses'] = 2
        exit_price = 49000.0
        record_closed_position(
            MagicMock(), base_state, config, exit_price, 'SL', APICache()
        )
        assert base_state['consecutive_losses'] == 3

    def test_win_resets_consecutive_losses(
        self, mock_update_esl, mock_cancel_esl,
        mock_cancel, mock_save_state, mock_save_metrics, mock_conf_log,
        base_state, config
    ):
        """Win after losses -> consecutive_losses resets to 0."""
        base_state['consecutive_losses'] = 3
        exit_price = 51000.0
        record_closed_position(
            MagicMock(), base_state, config, exit_price, 'TP', APICache()
        )
        assert base_state['consecutive_losses'] == 0

    def test_invalid_entry_price_pnl_zero(
        self, mock_update_esl, mock_cancel_esl,
        mock_cancel, mock_save_state, mock_save_metrics, mock_conf_log,
        base_state, config
    ):
        """Invalid entry_price (0) -> pnl recorded as 0%."""
        base_state['positions']['s1']['entry_price'] = 0
        record_closed_position(
            MagicMock(), base_state, config, 50000.0, 'MARKET', APICache()
        )
        # PnL should be 0, total_pnl unchanged
        assert base_state['total_pnl'] == 10.0
        assert base_state['total_trades'] == 6

    def test_position_cleared_after_recording(
        self, mock_update_esl, mock_cancel_esl,
        mock_cancel, mock_save_state, mock_save_metrics, mock_conf_log,
        base_state, config
    ):
        """Position removed from positions dict after recording."""
        record_closed_position(
            MagicMock(), base_state, config, 51000.0, 'TP', APICache()
        )
        assert len(base_state['positions']) == 0
        assert base_state['active_direction'] is None

    def test_last_trade_recorded(
        self, mock_update_esl, mock_cancel_esl,
        mock_cancel, mock_save_state, mock_save_metrics, mock_conf_log,
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
        self, mock_update_esl, mock_cancel_esl,
        mock_cancel, mock_save_state, mock_save_metrics, mock_conf_log,
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
        self, mock_update_esl, mock_cancel_esl,
        mock_cancel, mock_save_state, mock_save_metrics, mock_conf_log,
        base_state, config
    ):
        """cancel_remaining_orders called when exchange is provided."""
        exchange = MagicMock()
        record_closed_position(
            exchange, base_state, config, 51000.0, 'TP', APICache()
        )
        mock_cancel.assert_called_once()

    def test_no_cancel_when_exchange_none(
        self, mock_update_esl, mock_cancel_esl,
        mock_cancel, mock_save_state, mock_save_metrics, mock_conf_log,
        base_state, config
    ):
        """exchange=None -> cancel_remaining_orders not called."""
        record_closed_position(
            None, base_state, config, 51000.0, 'TP', APICache()
        )
        mock_cancel.assert_not_called()

    def test_short_winning_trade(
        self, mock_update_esl, mock_cancel_esl,
        mock_cancel, mock_save_state, mock_save_metrics, mock_conf_log,
        base_state, config
    ):
        """SHORT winning trade (exit < entry) -> positive PnL."""
        base_state['positions']['s1']['direction'] = 'SHORT'
        base_state['positions']['s1']['entry_price'] = 50000.0
        base_state['active_direction'] = 'SHORT'
        exit_price = 49000.0  # below entry for SHORT = profit
        record_closed_position(
            MagicMock(), base_state, config, exit_price, 'TP', APICache()
        )
        assert base_state['total_pnl'] > 10.0
        assert base_state['winning_trades'] == 4

    def test_daily_stats_updated(
        self, mock_update_esl, mock_cancel_esl,
        mock_cancel, mock_save_state, mock_save_metrics, mock_conf_log,
        base_state, config
    ):
        """daily_trades and daily_pnl updated."""
        record_closed_position(
            MagicMock(), base_state, config, 51000.0, 'TP', APICache()
        )
        assert base_state['daily_trades'] == 3
        assert base_state['daily_pnl'] > 5.0


# -- _read_tpsl_from_exchange_orders ----------------------------------------


class TestReadTpslFromExchangeOrders:
    """Test _read_tpsl_from_exchange_orders() parsing logic."""

    def test_no_exchange_returns_none(self):
        """exchange=None -> (None, None)."""
        tp, sl = _read_tpsl_from_exchange_orders(None, 'BTC/USDT:USDT', 'LONG')
        assert tp is None and sl is None

    def test_both_orders_found(self):
        """TP + SL orders on exchange -> both prices returned."""
        exchange = MagicMock()
        exchange.fetch_open_orders.return_value = [
            {'type': 'TAKE_PROFIT_MARKET', 'stopPrice': 51000.0, 'info': {}},
            {'type': 'STOP_MARKET', 'stopPrice': 49000.0, 'info': {}},
        ]
        tp, sl = _read_tpsl_from_exchange_orders(exchange, 'BTC/USDT:USDT', 'LONG')
        assert tp == 51000.0
        assert sl == 49000.0

    def test_only_tp_found(self):
        """Only TP order -> tp returned, sl is None."""
        exchange = MagicMock()
        exchange.fetch_open_orders.return_value = [
            {'type': 'TAKE_PROFIT_MARKET', 'stopPrice': 51000.0, 'info': {}},
        ]
        tp, sl = _read_tpsl_from_exchange_orders(exchange, 'BTC/USDT:USDT', 'LONG')
        assert tp == 51000.0
        assert sl is None

    def test_stop_price_from_info(self):
        """stopPrice missing at top level -> reads from info dict."""
        exchange = MagicMock()
        exchange.fetch_open_orders.return_value = [
            {'type': 'STOP_MARKET', 'stopPrice': None, 'info': {'stopPrice': '49500.0'}},
        ]
        tp, sl = _read_tpsl_from_exchange_orders(exchange, 'BTC/USDT:USDT', 'LONG')
        assert tp is None
        assert sl == 49500.0

    def test_network_error_returns_none(self):
        """Network error -> (None, None), no crash."""
        exchange = MagicMock()
        exchange.fetch_open_orders.side_effect = ccxt.NetworkError('timeout')
        tp, sl = _read_tpsl_from_exchange_orders(exchange, 'BTC/USDT:USDT', 'LONG')
        assert tp is None and sl is None

    def test_zero_stop_price_ignored(self):
        """stopPrice=0 -> skipped."""
        exchange = MagicMock()
        exchange.fetch_open_orders.return_value = [
            {'type': 'STOP_MARKET', 'stopPrice': 0, 'info': {}},
        ]
        tp, sl = _read_tpsl_from_exchange_orders(exchange, 'BTC/USDT:USDT', 'LONG')
        assert tp is None and sl is None


# -- close_position_market --------------------------------------------------


@patch('bingx_rl_trading_bot.scripts.production.pattern_5m.position_close.place_tp_sl_orders')
@patch('bingx_rl_trading_bot.scripts.production.pattern_5m.position_close.record_closed_position')
@patch('bingx_rl_trading_bot.scripts.production.pattern_5m.position_close.cancel_remaining_orders')
@patch('bingx_rl_trading_bot.scripts.production.pattern_5m.position_close.fetch_ticker_cached')
class TestClosePositionMarket:
    """Test close_position_market() market exit logic."""

    def _make_state(self):
        return {
            'positions': {
                's1': {
                    'slot_id': 's1',
                    'direction': 'LONG',
                    'quantity': 0.01,
                    'remaining_quantity': 0.01,
                },
            },
            'active_direction': 'LONG',
            'has_position': True,
        }

    def test_no_position_returns_false(
        self, mock_ticker, mock_cancel, mock_record, mock_place
    ):
        """No position -> False."""
        result = close_position_market(
            MagicMock(),
            {'positions': {}, 'active_direction': None, 'has_position': False},
            {'symbol': 'BTC/USDT:USDT'},
            APICache()
        )
        assert result is False
        mock_cancel.assert_not_called()

    def test_successful_market_close(
        self, mock_ticker, mock_cancel, mock_record, mock_place
    ):
        """Normal close -> cancel TP/SL, create market order, record."""
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
        """Fill price=0 -> uses ticker fallback."""
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
        """Market order fails -> re-places TP/SL, returns False."""
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
        """Quantity=0 -> False."""
        state = {
            'positions': {
                's1': {
                    'slot_id': 's1',
                    'direction': 'LONG',
                    'quantity': 0,
                    'remaining_quantity': 0,
                },
            },
            'active_direction': 'LONG',
            'has_position': True,
        }
        result = close_position_market(
            MagicMock(), state, {'symbol': 'BTC/USDT:USDT'}, APICache()
        )
        assert result is False


# -- recover_position_to_state ----------------------------------------------


@patch('bingx_rl_trading_bot.scripts.production.pattern_5m.position_close.place_tp_sl_orders')
@patch('bingx_rl_trading_bot.scripts.production.pattern_5m.position_close.save_state')
@patch('bingx_rl_trading_bot.scripts.production.pattern_5m.position_close._read_tpsl_from_exchange_orders')
@patch('bingx_rl_trading_bot.scripts.production.pattern_5m.position_close.calculate_tp_sl')
@patch('bingx_rl_trading_bot.scripts.production.pattern_5m.position_close.setup_scale_out')
class TestRecoverPositionToState:
    """Test recover_position_to_state() exchange -> state recovery."""

    def test_basic_recovery_with_exchange_tpsl(
        self, mock_scale, mock_calc, mock_read, mock_save, mock_place
    ):
        """Exchange has TP/SL -> uses them directly."""
        mock_read.return_value = (51000.0, 49000.0)
        mock_scale.return_value = []
        state = {'positions': {}, 'active_direction': None}
        config = {'strategy': {}, 'symbol': 'BTC/USDT:USDT'}
        exchange_pos = {'entryPrice': 50000.0, 'contracts': 0.01}
        exchange = MagicMock()

        recover_position_to_state(state, config, exchange_pos, 'LONG', exchange, APICache())

        positions = state['positions']
        assert len(positions) == 1
        pos = next(iter(positions.values()))
        assert pos['direction'] == 'LONG'
        assert pos['entry_price'] == 50000.0
        assert pos['tp_price'] == 51000.0
        assert pos['sl_price'] == 49000.0
        assert pos['recovered'] is True
        mock_calc.assert_not_called()  # Should not calculate -- read from exchange
        mock_save.assert_called()

    def test_fallback_to_config_when_no_exchange_tpsl(
        self, mock_scale, mock_calc, mock_read, mock_save, mock_place
    ):
        """No TP/SL on exchange -> calculate from config."""
        mock_read.return_value = (None, None)
        mock_calc.return_value = (51500.0, 48500.0, 3.0, 3.0)
        mock_scale.return_value = []
        state = {'positions': {}, 'active_direction': None}
        config = {'strategy': {}, 'symbol': 'BTC/USDT:USDT'}
        exchange_pos = {'entryPrice': 50000.0, 'contracts': 0.01}

        recover_position_to_state(state, config, exchange_pos, 'SHORT')

        positions = state['positions']
        assert len(positions) == 1
        pos = next(iter(positions.values()))
        assert pos['direction'] == 'SHORT'
        assert pos['tp_price'] == 51500.0
        mock_calc.assert_called_once()
        mock_place.assert_not_called()  # No exchange provided

    def test_places_orders_when_exchange_provided(
        self, mock_scale, mock_calc, mock_read, mock_save, mock_place
    ):
        """exchange provided -> place_tp_sl_orders called."""
        mock_read.return_value = (51000.0, 49000.0)
        mock_scale.return_value = []
        state = {'positions': {}, 'active_direction': None}
        config = {'strategy': {}, 'symbol': 'BTC/USDT:USDT'}
        exchange_pos = {'entryPrice': 50000.0, 'contracts': 0.01}
        exchange = MagicMock()

        recover_position_to_state(state, config, exchange_pos, 'LONG', exchange)

        mock_place.assert_called_once()

    def test_preserves_pattern_from_old_state(
        self, mock_scale, mock_calc, mock_read, mock_save, mock_place
    ):
        """Old position has pattern -> preserved in reason."""
        mock_read.return_value = (51000.0, 49000.0)
        mock_scale.return_value = []
        state = {
            'positions': {
                's_old': {
                    'slot_id': 's_old',
                    'direction': 'LONG',
                    'reason': 'Pattern: BD-BD-U (LONG)',
                },
            },
            'active_direction': 'LONG',
        }
        config = {'strategy': {}, 'symbol': 'BTC/USDT:USDT'}
        exchange_pos = {'entryPrice': 50000.0, 'contracts': 0.01}

        recover_position_to_state(state, config, exchange_pos, 'LONG')

        # The function creates a new slot; find it (not the old one)
        positions = state['positions']
        # There should be at least the new recovered slot
        new_slots = {sid: s for sid, s in positions.items() if s.get('recovered')}
        assert len(new_slots) >= 1
        pos = next(iter(new_slots.values()))
        assert 'BD-BD-U' in pos['reason']
        assert pos['pattern_name'] == 'BD-BD-U'


# -- recover_from_crash -----------------------------------------------------


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
        """Exchange has position, local doesn't -> recover to state."""
        mock_fetch_pos.return_value = [
            {'side': 'long', 'contracts': 0.01, 'entryPrice': 50000.0},
        ]
        state = {'positions': {}, 'active_direction': None}
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
        """Local has position, exchange doesn't, trade history found -> record closed."""
        mock_fetch_pos.return_value = []  # no exchange position
        mock_exit.return_value = {'price': 51000.0, 'reason': 'TP'}
        state = {
            'positions': {
                's1': {
                    'slot_id': 's1',
                    'direction': 'LONG', 'entry_price': 50000.0,
                    'quantity': 0.01, 'entry_time': '2026-02-18T10:00:00',
                },
            },
            'active_direction': 'LONG',
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
        """Local only, no trade history -> uses ticker fallback."""
        mock_fetch_pos.return_value = []
        mock_exit.return_value = None
        mock_ticker.return_value = {'last': 50200.0}
        state = {
            'positions': {
                's1': {
                    'slot_id': 's1',
                    'direction': 'LONG', 'entry_price': 50000.0,
                    'quantity': 0.01, 'entry_time': '2026-02-18T10:00:00',
                },
            },
            'active_direction': 'LONG',
        }
        config = {'symbol': 'BTC/USDT:USDT'}
        result = recover_from_crash(MagicMock(), state, config, APICache())
        assert result is True
        assert mock_record.call_args[0][3] == 50200.0

    def test_case3_direction_mismatch(
        self, mock_ticker, mock_fetch_pos, mock_ghost,
        mock_record, mock_recover, mock_recalc
    ):
        """Both exist, directions mismatch -> close local + recover exchange."""
        mock_fetch_pos.return_value = [
            {'side': 'short', 'contracts': 0.01, 'entryPrice': 50000.0},
        ]
        state = {
            'positions': {
                's1': {
                    'slot_id': 's1',
                    'direction': 'LONG', 'entry_price': 50000.0,
                    'quantity': 0.01, 'entry_time': '2026-02-18T10:00:00',
                },
            },
            'active_direction': 'LONG',
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
        """Both exist, same direction, qty differs -> recalculate."""
        mock_fetch_pos.return_value = [
            {'side': 'long', 'contracts': 0.02, 'entryPrice': 50000.0},
        ]
        state = {
            'positions': {
                's1': {
                    'slot_id': 's1',
                    'direction': 'LONG', 'entry_price': 50000.0,
                    'quantity': 0.01,
                },
            },
            'active_direction': 'LONG',
        }
        config = {'symbol': 'BTC/USDT:USDT'}
        result = recover_from_crash(MagicMock(), state, config, APICache())
        assert result is True
        mock_recalc.assert_called_once()

    def test_consistent_state_returns_false(
        self, mock_ticker, mock_fetch_pos, mock_ghost,
        mock_record, mock_recover, mock_recalc
    ):
        """Both exist, same direction, same qty -> no action."""
        mock_fetch_pos.return_value = [
            {'side': 'long', 'contracts': 0.01, 'entryPrice': 50000.0},
        ]
        state = {
            'positions': {
                's1': {
                    'slot_id': 's1',
                    'direction': 'LONG', 'entry_price': 50000.0,
                    'quantity': 0.01,
                    'scale_out_stages': [],
                },
            },
            'active_direction': 'LONG',
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
        """Network error during fetch -> False."""
        mock_fetch_pos.side_effect = ccxt.NetworkError('timeout')
        state = {'positions': {}, 'active_direction': None}
        config = {'symbol': 'BTC/USDT:USDT'}
        result = recover_from_crash(MagicMock(), state, config, APICache())
        assert result is False

    def test_no_positions_anywhere_returns_false(
        self, mock_ticker, mock_fetch_pos, mock_ghost,
        mock_record, mock_recover, mock_recalc
    ):
        """No exchange position, no local -> no action."""
        mock_fetch_pos.return_value = []
        state = {'positions': {}, 'active_direction': None}
        config = {'symbol': 'BTC/USDT:USDT'}
        result = recover_from_crash(MagicMock(), state, config, APICache())
        assert result is False


# -- detect_ghost_positions -------------------------------------------------


@patch('bingx_rl_trading_bot.scripts.production.pattern_5m.position_close.fetch_positions_cached')
class TestDetectGhostPositions:
    """Test detect_ghost_positions() ghost detection logic."""

    def test_no_exchange_position_no_warning(self, mock_fetch):
        """No exchange position -> no warning logged."""
        mock_fetch.return_value = []
        state = {'positions': {}, 'active_direction': None}
        config = {'symbol': 'BTC/USDT:USDT'}
        detect_ghost_positions(MagicMock(), state, config, APICache())

    def test_exchange_and_local_exist_no_ghost(self, mock_fetch):
        """Both exchange and local position -> no ghost."""
        mock_fetch.return_value = [
            {'side': 'long', 'contracts': 0.01, 'entryPrice': 50000.0},
        ]
        state = {
            'positions': {
                's1': {'slot_id': 's1', 'direction': 'LONG'},
            },
            'active_direction': 'LONG',
        }
        config = {'symbol': 'BTC/USDT:USDT'}
        detect_ghost_positions(MagicMock(), state, config, APICache())

    def test_ghost_detected_exchange_only(self, mock_fetch):
        """Exchange has position but local doesn't -> ghost detected."""
        mock_fetch.return_value = [
            {'side': 'short', 'contracts': 0.02, 'entryPrice': 48000.0},
        ]
        state = {'positions': {}, 'active_direction': None}
        config = {'symbol': 'BTC/USDT:USDT'}
        detect_ghost_positions(MagicMock(), state, config, APICache())

    def test_network_error_handled(self, mock_fetch):
        """Network error -> no crash."""
        mock_fetch.side_effect = ccxt.NetworkError('timeout')
        state = {'positions': {}, 'active_direction': None}
        config = {'symbol': 'BTC/USDT:USDT'}
        detect_ghost_positions(MagicMock(), state, config, APICache())

    def test_zero_contracts_ignored(self, mock_fetch):
        """Position with 0 contracts -> not treated as exchange position."""
        mock_fetch.return_value = [
            {'side': 'long', 'contracts': 0, 'entryPrice': 50000.0},
        ]
        state = {'positions': {}, 'active_direction': None}
        config = {'symbol': 'BTC/USDT:USDT'}
        detect_ghost_positions(MagicMock(), state, config, APICache())


# -- recalculate_position_orders --------------------------------------------


@patch('bingx_rl_trading_bot.scripts.production.pattern_5m.position_close.save_state')
@patch('bingx_rl_trading_bot.scripts.production.pattern_5m.position_close.place_tp_sl_orders')
@patch('bingx_rl_trading_bot.scripts.production.pattern_5m.position_close.cancel_remaining_orders')
@patch('bingx_rl_trading_bot.scripts.production.pattern_5m.position_close.calculate_tp_sl')
@patch('bingx_rl_trading_bot.scripts.production.pattern_5m.position_close.setup_scale_out')
class TestRecalculatePositionOrders:
    """Test recalculate_position_orders() partial-fill rebalancing."""

    def test_no_position_returns_false(
        self, mock_scale, mock_calc, mock_cancel, mock_place, mock_save
    ):
        """No position -> False."""
        result = recalculate_position_orders(
            MagicMock(),
            {'positions': {}, 'active_direction': None},
            {}, 0.02
        )
        assert result is False

    def test_successful_recalculation(
        self, mock_scale, mock_calc, mock_cancel, mock_place, mock_save
    ):
        """Normal recalc -> cancels old, updates qty, places new, returns True."""
        mock_calc.return_value = (51000.0, 49000.0, 2.0, 2.0)
        mock_scale.return_value = []

        position_data = {
            'slot_id': 's1',
            'direction': 'LONG',
            'entry_price': 50000.0,
            'quantity': 0.01,
            'remaining_quantity': 0.01,
            'vol_mult': 1.0,
            'reason': 'Pattern: BD-BD-U (LONG)',
            'sl_order_id': 'sl_new',
        }
        state = {
            'positions': {'s1': position_data},
            'active_direction': 'LONG',
            'has_position': True,
        }
        config = {'symbol': 'BTC/USDT:USDT', 'strategy': {}}

        def set_sl_id(exchange, state, config, position=None):
            pos = position if position is not None else next(iter(state['positions'].values()))
            pos['sl_order_id'] = 'sl_recalc'
        mock_place.side_effect = set_sl_id

        result = recalculate_position_orders(MagicMock(), state, config, 0.02)
        assert result is True
        pos = state['positions']['s1']
        assert pos['quantity'] == 0.02
        assert pos['remaining_quantity'] == 0.02
        mock_cancel.assert_called_once()
        mock_calc.assert_called_once()
        mock_place.assert_called_once()

    def test_cancel_failure_continues(
        self, mock_scale, mock_calc, mock_cancel, mock_place, mock_save
    ):
        """Cancel fails -> continues with recalculation."""
        mock_cancel.side_effect = ccxt.NetworkError('timeout')
        mock_calc.return_value = (51000.0, 49000.0, 2.0, 2.0)
        mock_scale.return_value = []

        position_data = {
            'slot_id': 's1',
            'direction': 'LONG', 'entry_price': 50000.0,
            'quantity': 0.01, 'remaining_quantity': 0.01,
            'vol_mult': 1.0, 'reason': '',
        }
        state = {
            'positions': {'s1': position_data},
            'active_direction': 'LONG',
            'has_position': True,
        }
        config = {'symbol': 'BTC/USDT:USDT', 'strategy': {}}

        def set_sl_id(exchange, state, config, position=None):
            pos = position if position is not None else next(iter(state['positions'].values()))
            pos['sl_order_id'] = 'sl_1'
        mock_place.side_effect = set_sl_id

        result = recalculate_position_orders(MagicMock(), state, config, 0.02)
        assert result is True
        assert state['positions']['s1']['quantity'] == 0.02

    def test_place_failure_returns_false(
        self, mock_scale, mock_calc, mock_cancel, mock_place, mock_save
    ):
        """Place orders fails -> returns False, state saved."""
        mock_calc.return_value = (51000.0, 49000.0, 2.0, 2.0)
        mock_scale.return_value = []
        mock_place.side_effect = Exception('order failed')

        position_data = {
            'slot_id': 's1',
            'direction': 'LONG', 'entry_price': 50000.0,
            'quantity': 0.01, 'remaining_quantity': 0.01,
            'vol_mult': 1.0, 'reason': '',
        }
        state = {
            'positions': {'s1': position_data},
            'active_direction': 'LONG',
            'has_position': True,
        }
        config = {'symbol': 'BTC/USDT:USDT', 'strategy': {}}
        result = recalculate_position_orders(MagicMock(), state, config, 0.02)
        assert result is False
        mock_save.assert_called()  # State saved even on failure

    def test_sl_not_confirmed_returns_false(
        self, mock_scale, mock_calc, mock_cancel, mock_place, mock_save
    ):
        """SL not confirmed after place -> returns False with warning."""
        mock_calc.return_value = (51000.0, 49000.0, 2.0, 2.0)
        mock_scale.return_value = []
        # place_tp_sl_orders doesn't set sl_order_id
        mock_place.return_value = None

        position_data = {
            'slot_id': 's1',
            'direction': 'LONG', 'entry_price': 50000.0,
            'quantity': 0.01, 'remaining_quantity': 0.01,
            'vol_mult': 1.0, 'reason': '',
            'sl_order_id': None,
        }
        state = {
            'positions': {'s1': position_data},
            'active_direction': 'LONG',
            'has_position': True,
        }
        config = {'symbol': 'BTC/USDT:USDT', 'strategy': {}}
        result = recalculate_position_orders(MagicMock(), state, config, 0.02)
        assert result is False


# -- close_position_market error paths --------------------------------------


@patch('bingx_rl_trading_bot.scripts.production.pattern_5m.position_close.record_closed_position')
@patch('bingx_rl_trading_bot.scripts.production.pattern_5m.position_close.place_tp_sl_orders')
@patch('bingx_rl_trading_bot.scripts.production.pattern_5m.position_close.cancel_remaining_orders')
class TestClosePositionMarketErrors:
    """Test close_position_market error handling paths."""

    def _make_state(self):
        return {
            'positions': {
                's1': {
                    'slot_id': 's1',
                    'direction': 'LONG', 'entry_price': 50000.0,
                    'quantity': 0.01, 'remaining_quantity': 0.01,
                    'tp_price': 51000.0, 'sl_price': 49000.0,
                },
            },
            'active_direction': 'LONG',
            'has_position': True,
        }

    def test_insufficient_funds(self, mock_cancel, mock_place, mock_record):
        """InsufficientFunds -> False, TP/SL re-placed."""
        exchange = MagicMock()
        exchange.create_order.side_effect = ccxt.InsufficientFunds('no funds')
        result = close_position_market(
            exchange, self._make_state(), {'symbol': 'BTC/USDT:USDT'},
            APICache(), 'early_exit', PerformanceMetrics()
        )
        assert result is False
        mock_place.assert_called_once()

    def test_invalid_order(self, mock_cancel, mock_place, mock_record):
        """InvalidOrder -> False, TP/SL re-placed."""
        exchange = MagicMock()
        exchange.create_order.side_effect = ccxt.InvalidOrder('invalid')
        result = close_position_market(
            exchange, self._make_state(), {'symbol': 'BTC/USDT:USDT'},
            APICache(), 'early_exit', PerformanceMetrics()
        )
        assert result is False

    def test_network_error(self, mock_cancel, mock_place, mock_record):
        """NetworkError -> False, TP/SL re-placed."""
        exchange = MagicMock()
        exchange.create_order.side_effect = ccxt.NetworkError('timeout')
        result = close_position_market(
            exchange, self._make_state(), {'symbol': 'BTC/USDT:USDT'},
            APICache(), 'early_exit', PerformanceMetrics()
        )
        assert result is False

    def test_exchange_error(self, mock_cancel, mock_place, mock_record):
        """ExchangeError -> False, TP/SL re-placed."""
        exchange = MagicMock()
        exchange.create_order.side_effect = ccxt.ExchangeError('rejected')
        result = close_position_market(
            exchange, self._make_state(), {'symbol': 'BTC/USDT:USDT'},
            APICache(), 'early_exit', PerformanceMetrics()
        )
        assert result is False

    def test_generic_exception(self, mock_cancel, mock_place, mock_record):
        """Generic exception -> False, TP/SL re-placed."""
        exchange = MagicMock()
        exchange.create_order.side_effect = RuntimeError('unexpected')
        result = close_position_market(
            exchange, self._make_state(), {'symbol': 'BTC/USDT:USDT'},
            APICache(), 'early_exit', PerformanceMetrics()
        )
        assert result is False

    def test_restore_tpsl_fails(self, mock_cancel, mock_place, mock_record):
        """TP/SL re-placement also fails -> still returns False."""
        exchange = MagicMock()
        exchange.create_order.side_effect = ccxt.InsufficientFunds('no funds')
        mock_place.side_effect = Exception('restore failed')
        result = close_position_market(
            exchange, self._make_state(), {'symbol': 'BTC/USDT:USDT'},
            APICache(), 'early_exit', PerformanceMetrics()
        )
        assert result is False


# -- recover_from_crash additional paths ------------------------------------


@patch('bingx_rl_trading_bot.scripts.production.pattern_5m.position_close.recalculate_position_orders')
@patch('bingx_rl_trading_bot.scripts.production.pattern_5m.position_close.recover_position_to_state')
@patch('bingx_rl_trading_bot.scripts.production.pattern_5m.position_close.record_closed_position')
@patch('bingx_rl_trading_bot.scripts.production.pattern_5m.position_close.detect_ghost_positions')
@patch('bingx_rl_trading_bot.scripts.production.pattern_5m.position_close.fetch_positions_cached')
@patch('bingx_rl_trading_bot.scripts.production.pattern_5m.position_close.fetch_ticker_cached')
class TestRecoverFromCrashExtended:
    """Test recover_from_crash additional error paths."""

    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.position_monitor.get_actual_exit_price')
    def test_case2_ticker_exception_fallback(
        self, mock_exit, mock_ticker, mock_fetch_pos, mock_ghost,
        mock_record, mock_recover, mock_recalc
    ):
        """Case 2: ticker fails -> fallback to entry_price."""
        mock_fetch_pos.return_value = []
        mock_exit.return_value = None
        mock_ticker.side_effect = Exception('no ticker')
        state = {
            'positions': {
                's1': {
                    'slot_id': 's1',
                    'direction': 'LONG', 'entry_price': 50000.0,
                    'quantity': 0.01, 'entry_time': '2026-02-18T10:00:00',
                },
            },
            'active_direction': 'LONG',
        }
        config = {'symbol': 'BTC/USDT:USDT'}
        result = recover_from_crash(MagicMock(), state, config, APICache())
        assert result is True
        assert mock_record.call_args[0][3] == 50000.0

    def test_exchange_error_returns_false(
        self, mock_ticker, mock_fetch_pos, mock_ghost,
        mock_record, mock_recover, mock_recalc
    ):
        """ExchangeError during fetch -> returns False."""
        mock_fetch_pos.side_effect = ccxt.ExchangeError('invalid')
        state = {'positions': {}, 'active_direction': None}
        config = {'symbol': 'BTC/USDT:USDT'}
        result = recover_from_crash(MagicMock(), state, config, APICache())
        assert result is False

    def test_generic_exception_returns_false(
        self, mock_ticker, mock_fetch_pos, mock_ghost,
        mock_record, mock_recover, mock_recalc
    ):
        """Generic exception -> returns False."""
        mock_fetch_pos.side_effect = RuntimeError('unexpected')
        state = {'positions': {}, 'active_direction': None}
        config = {'symbol': 'BTC/USDT:USDT'}
        result = recover_from_crash(MagicMock(), state, config, APICache())
        assert result is False

    def test_case4_quantity_mismatch_with_scale_out(
        self, mock_ticker, mock_fetch_pos, mock_ghost,
        mock_record, mock_recover, mock_recalc
    ):
        """Exchange qty != local qty sum -> recalculate (e.g. partial fill changed qty)."""
        mock_fetch_pos.return_value = [
            {'side': 'long', 'contracts': 0.015, 'entryPrice': 50000.0},
        ]
        state = {
            'positions': {
                's1': {
                    'slot_id': 's1',
                    'direction': 'LONG', 'entry_price': 50000.0,
                    'quantity': 0.01,
                    'scale_out_stages': [
                        {'quantity': 0.003, 'filled': False},
                        {'quantity': 0.003, 'filled': False},
                    ],
                },
            },
            'active_direction': 'LONG',
        }
        config = {'symbol': 'BTC/USDT:USDT'}
        result = recover_from_crash(MagicMock(), state, config, APICache())
        assert result is True
        mock_recalc.assert_called_once()


# -- _update_confidence_log_outcome -----------------------------------------


class TestUpdateConfidenceLogOutcome:
    """Test _update_confidence_log_outcome() CSV update logic."""

    def test_nonexistent_file_no_crash(self, tmp_path):
        """Non-existent CSV -> no-op."""
        from bingx_rl_trading_bot.scripts.production.pattern_5m.position_close import (
            _update_confidence_log_outcome,
        )
        with patch(
            'bingx_rl_trading_bot.scripts.production.pattern_5m.position_close.CONFIDENCE_LOG_FILE',
            str(tmp_path / 'nonexistent.csv'),
        ):
            _update_confidence_log_outcome('2026-02-18T10:00:00', 'WIN', 2.5)

    def test_updates_trailing_comma_row(self, tmp_path):
        """Row ending with comma -> appends outcome."""
        from bingx_rl_trading_bot.scripts.production.pattern_5m.position_close import (
            _update_confidence_log_outcome,
        )
        csv_file = tmp_path / 'confidence.csv'
        csv_file.write_text('time,pattern,confidence,outcome\n2026-02-18T10:00:00,BD-BD-BU,0.85,\n')
        with patch(
            'bingx_rl_trading_bot.scripts.production.pattern_5m.position_close.CONFIDENCE_LOG_FILE',
            str(csv_file),
        ):
            _update_confidence_log_outcome('2026-02-18T10:00:00', 'WIN', 2.5)
        content = csv_file.read_text()
        assert 'WIN:+2.50%' in content

    def test_no_trailing_comma_no_update(self, tmp_path):
        """No row with trailing comma -> no update."""
        from bingx_rl_trading_bot.scripts.production.pattern_5m.position_close import (
            _update_confidence_log_outcome,
        )
        csv_file = tmp_path / 'confidence.csv'
        csv_file.write_text('time,pattern,confidence,outcome\n2026-02-18,BD-BD-BU,0.85,WIN:+1.00%\n')
        with patch(
            'bingx_rl_trading_bot.scripts.production.pattern_5m.position_close.CONFIDENCE_LOG_FILE',
            str(csv_file),
        ):
            _update_confidence_log_outcome('2026-02-18T10:00:00', 'LOSS', -1.5)
        content = csv_file.read_text()
        assert 'LOSS' not in content

    def test_exception_no_crash(self, tmp_path):
        """Exception during file read -> no crash (lines 695-696)."""
        from bingx_rl_trading_bot.scripts.production.pattern_5m.position_close import (
            _update_confidence_log_outcome,
        )
        csv_file = tmp_path / 'confidence.csv'
        csv_file.write_text('header\ndata,\n')  # file exists
        with patch(
            'bingx_rl_trading_bot.scripts.production.pattern_5m.position_close.CONFIDENCE_LOG_FILE',
            str(csv_file),
        ):
            with patch('os.path.getsize', side_effect=OSError('disk error')):
                _update_confidence_log_outcome('2026-02-18T10:00:00', 'WIN', 2.5)

    def test_large_file_seek_branch(self, tmp_path):
        """File > 4096 bytes -> seeks to tail, skips partial line (lines 671-674)."""
        from bingx_rl_trading_bot.scripts.production.pattern_5m.position_close import (
            _update_confidence_log_outcome,
        )
        csv_file = tmp_path / 'confidence.csv'
        header = 'time,pattern,confidence,outcome\n'
        # Fill with >4096 bytes of completed rows
        filled_row = '2026-02-18T10:00:00,BD-BD-BU,0.85,WIN:+1.00%\n'
        rows_needed = (4096 // len(filled_row)) + 10
        filler = filled_row * rows_needed
        # Last row has trailing comma (empty outcome)
        filler += '2026-02-18T12:00:00,MU-U-H,0.90,\n'
        csv_file.write_text(header + filler)
        assert csv_file.stat().st_size > 4096

        with patch(
            'bingx_rl_trading_bot.scripts.production.pattern_5m.position_close.CONFIDENCE_LOG_FILE',
            str(csv_file),
        ):
            _update_confidence_log_outcome('2026-02-18T12:00:00', 'LOSS', -3.5)
        content = csv_file.read_text()
        assert 'LOSS:-3.50%' in content

    def test_empty_file_no_crash(self, tmp_path):
        """Empty file (0 tail lines) -> early return (line 682)."""
        from bingx_rl_trading_bot.scripts.production.pattern_5m.position_close import (
            _update_confidence_log_outcome,
        )
        csv_file = tmp_path / 'confidence.csv'
        csv_file.write_text('')
        with patch(
            'bingx_rl_trading_bot.scripts.production.pattern_5m.position_close.CONFIDENCE_LOG_FILE',
            str(csv_file),
        ):
            _update_confidence_log_outcome('2026-02-18T10:00:00', 'WIN', 2.5)
        assert csv_file.read_text() == ''


# -- record_closed_position hold time error (lines 162-163) -----------------


class TestRecordClosedPositionHoldTime:
    """Test hold time ValueError/TypeError in record_closed_position."""

    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.position_close._update_confidence_log_outcome')
    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.position_close.save_metrics')
    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.position_close.save_state')
    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.position_close.cancel_remaining_orders')
    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.position_close.cancel_emergency_sl')
    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.position_close.update_emergency_sl')
    def test_invalid_entry_time_no_crash(self, mock_update_esl, mock_cancel_esl,
                                          mock_cancel, mock_save, mock_metrics, mock_log):
        """Invalid entry_time string -> ValueError caught (lines 162-163)."""
        exchange = MagicMock()
        state = {
            'positions': {
                's1': {
                    'slot_id': 's1',
                    'direction': 'LONG', 'entry_price': 50000.0,
                    'quantity': 0.01, 'remaining_quantity': 0.01,
                    'reason': 'Pattern: BD-BD-BU (LONG)',
                    'entry_time': 'not-a-date',  # triggers ValueError in fromisoformat
                    'tp_price': 51000.0, 'sl_price': 49000.0,
                },
            },
            'active_direction': 'LONG',
            'has_position': True,
            'total_trades': 0, 'winning_trades': 0, 'total_pnl': 0.0,
            'daily_trades': 0, 'daily_pnl': 0.0, 'consecutive_losses': 0,
            'last_trade': None, 'last_signal_time': None,
        }
        config = {'symbol': 'BTC/USDT:USDT', 'leverage': 3}
        cache = APICache()
        metrics = PerformanceMetrics()
        record_closed_position(exchange, state, config, 51000.0, 'TP', cache, metrics)
        mock_save.assert_called()


# -- recover_position_to_state needs_tpsl (lines 293-294) ------------------


class TestRecoverPositionNeedsTpsl:
    """Test recover_position_to_state sets needs_tpsl=False after order placement."""

    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.position_close.save_state')
    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.position_close.place_tp_sl_orders')
    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.position_close.setup_scale_out',
           return_value=[])
    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.position_close._read_tpsl_from_exchange_orders',
           return_value=(None, None))
    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.position_close.calculate_tp_sl',
           return_value=(51000.0, 49000.0, 2.0, 2.0))
    def test_tp_order_placed_clears_needs_tpsl(self, mock_calc, mock_read, mock_setup,
                                                mock_place, mock_save):
        """After successful TP/SL placement -> needs_tpsl=False (lines 293-294)."""
        exchange = MagicMock()
        config = {'symbol': 'BTC/USDT:USDT', 'strategy': {
            'tp_pct': 2.0, 'sl_pct': 2.0,
        }}
        state = {'positions': {}, 'active_direction': None}
        exchange_pos = {'entryPrice': 50000.0, 'contracts': 0.01}

        def set_tp_order_id(exch, st, cfg, position=None):
            # The position dict is passed directly; set tp_order_id on it
            if position is not None:
                position['tp_order_id'] = 'tp_123'
            else:
                pos = next(iter(st['positions'].values()))
                pos['tp_order_id'] = 'tp_123'
        mock_place.side_effect = set_tp_order_id

        recover_position_to_state(state, config, exchange_pos, 'LONG', exchange)
        # save_state called twice: once for initial recovery, once for needs_tpsl=False
        assert mock_save.call_count == 2
        pos = next(iter(state['positions'].values()))
        assert pos['needs_tpsl'] is False
