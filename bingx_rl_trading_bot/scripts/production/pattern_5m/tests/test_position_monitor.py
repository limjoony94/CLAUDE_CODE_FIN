"""Tests for position_monitor.py — exit inference, scale-out fills,
position sync, and status checking."""

import pytest
import ccxt
from unittest.mock import MagicMock, patch, call

from bingx_rl_trading_bot.scripts.production.pattern_5m.position_monitor import (
    _infer_exit_reason,
    _infer_exit_from_price,
    _check_scale_out_fills,
    _handle_position_closed,
    get_actual_exit_price,
    check_position_status,
    sync_position_with_exchange,
)
from bingx_rl_trading_bot.scripts.production.pattern_5m.constants import (
    PRICE_TOLERANCE_PCT,
    TP_LOWER_MULT,
    TP_UPPER_MULT,
    SL_LOWER_MULT,
    SL_UPPER_MULT,
)
from bingx_rl_trading_bot.scripts.production.pattern_5m.models import (
    APICache,
)
from bingx_rl_trading_bot.scripts.production.pattern_5m.orders import _EXCHANGE_MANAGED


# ── Helper: build multi-position state ───────────────────────

def _make_position_state(position_dict, slot_id='s1'):
    """Build a v1.29+ multi-position state from a single position dict."""
    pos = dict(position_dict)
    pos['slot_id'] = slot_id
    direction = pos.get('direction')
    return {
        'positions': {slot_id: pos},
        'active_direction': direction,
    }


def _make_empty_position_state():
    """Build a v1.29+ state with no active positions."""
    return {
        'positions': {},
        'active_direction': None,
    }


# ── _infer_exit_reason (trade history based) ─────────────────


class TestInferExitReason:
    """Test _infer_exit_reason() proximity to TP/SL."""

    def test_tp_hit_within_tolerance(self):
        """Price very close to TP → 'TP'."""
        position = {'tp_price': 51000.0, 'sl_price': 49000.0}
        result = _infer_exit_reason(51000.0, position)
        assert result == 'TP'

    def test_sl_hit_within_tolerance(self):
        """Price very close to SL → 'SL'."""
        position = {'tp_price': 51000.0, 'sl_price': 49000.0}
        result = _infer_exit_reason(49000.0, position)
        assert result == 'SL'

    def test_market_exit(self):
        """Price far from both TP and SL → 'MARKET'."""
        position = {'tp_price': 51000.0, 'sl_price': 49000.0}
        result = _infer_exit_reason(50000.0, position)
        assert result == 'MARKET'

    def test_near_tp_within_tolerance(self):
        """Price within PRICE_TOLERANCE_PCT of TP → 'TP'."""
        position = {'tp_price': 51000.0, 'sl_price': 49000.0}
        # Within 0.3% of TP
        near_tp = 51000.0 * (1 - PRICE_TOLERANCE_PCT * 0.5)
        result = _infer_exit_reason(near_tp, position)
        assert result == 'TP'

    def test_zero_tp_sl(self):
        """TP/SL = 0 → 'MARKET'."""
        position = {'tp_price': 0, 'sl_price': 0}
        result = _infer_exit_reason(50000.0, position)
        assert result == 'MARKET'

    def test_missing_tp_sl_keys(self):
        """Missing TP/SL keys → 'MARKET'."""
        result = _infer_exit_reason(50000.0, {})
        assert result == 'MARKET'


# ── _infer_exit_from_price (current price based) ────────────


class TestInferExitFromPrice:
    """Test _infer_exit_from_price() directional proximity."""

    def test_long_tp_hit(self):
        """LONG: exit >= TP * TP_LOWER_MULT → 'TP'."""
        position = {
            'direction': 'LONG',
            'tp_price': 51000.0,
            'sl_price': 49000.0,
        }
        # Price at or above TP zone
        result = _infer_exit_from_price(51000.0, position)
        assert result == 'TP'

    def test_long_sl_hit(self):
        """LONG: exit <= SL * SL_UPPER_MULT → 'SL'."""
        position = {
            'direction': 'LONG',
            'tp_price': 51000.0,
            'sl_price': 49000.0,
        }
        result = _infer_exit_from_price(49000.0, position)
        assert result == 'SL'

    def test_long_unknown(self):
        """LONG: price between TP and SL → 'UNKNOWN'."""
        position = {
            'direction': 'LONG',
            'tp_price': 51000.0,
            'sl_price': 49000.0,
        }
        result = _infer_exit_from_price(50000.0, position)
        assert result == 'UNKNOWN'

    def test_short_tp_hit(self):
        """SHORT: exit <= TP * TP_UPPER_MULT → 'TP'."""
        position = {
            'direction': 'SHORT',
            'tp_price': 49000.0,
            'sl_price': 51000.0,
        }
        result = _infer_exit_from_price(49000.0, position)
        assert result == 'TP'

    def test_short_sl_hit(self):
        """SHORT: exit >= SL * SL_LOWER_MULT → 'SL'."""
        position = {
            'direction': 'SHORT',
            'tp_price': 49000.0,
            'sl_price': 51000.0,
        }
        result = _infer_exit_from_price(51000.0, position)
        assert result == 'SL'

    def test_short_unknown(self):
        """SHORT: price between TP and SL → 'UNKNOWN'."""
        position = {
            'direction': 'SHORT',
            'tp_price': 49000.0,
            'sl_price': 51000.0,
        }
        result = _infer_exit_from_price(50000.0, position)
        assert result == 'UNKNOWN'

    def test_zero_tp_returns_unknown(self):
        """TP = 0 → cannot match TP, returns 'UNKNOWN' or 'SL'."""
        position = {
            'direction': 'LONG',
            'tp_price': 0,
            'sl_price': 49000.0,
        }
        result = _infer_exit_from_price(50000.0, position)
        assert result in ('UNKNOWN', 'SL')

    def test_zero_sl_returns_tp_or_unknown(self):
        """SL = 0 → cannot match SL."""
        position = {
            'direction': 'LONG',
            'tp_price': 51000.0,
            'sl_price': 0,
        }
        result = _infer_exit_from_price(51000.0, position)
        assert result == 'TP'


# ── _check_scale_out_fills ───────────────────────────────────


@patch('bingx_rl_trading_bot.scripts.production.pattern_5m.position_monitor.save_state')
class TestCheckScaleOutFills:
    """Test _check_scale_out_fills() stage fill detection."""

    def _make_position(self, qty=0.01, stages=None):
        return {
            'slot_id': 's1',
            'quantity': qty,
            'remaining_quantity': qty,
            'scale_out_stages': stages or [],
            'rotation_enabled': False,
        }

    def test_no_reduction_no_fill(self, mock_save):
        """Exchange qty unchanged → no fills detected."""
        position = self._make_position(
            qty=0.01,
            stages=[{'stage': 1, 'quantity': 0.005, 'filled': False}],
        )
        state = {
            'positions': {'s1': position},
            'active_direction': None,
        }
        current_pos = {'contracts': 0.01}

        _check_scale_out_fills(state, position, current_pos)

        mock_save.assert_not_called()

    def test_stage_filled_on_qty_reduction(self, mock_save):
        """Exchange qty drops below threshold → stage marked filled."""
        position = self._make_position(
            qty=0.01,
            stages=[{'stage': 1, 'quantity': 0.005, 'filled': False}],
        )
        state = {
            'positions': {'s1': position},
            'active_direction': None,
        }
        # Exchange shows reduced qty (stage 1 filled: 0.01 - 0.005 = 0.005)
        current_pos = {'contracts': 0.005}

        _check_scale_out_fills(state, position, current_pos)

        assert position['scale_out_stages'][0]['filled'] is True
        assert position['remaining_quantity'] == 0.005
        mock_save.assert_called_once()

    def test_rotation_sets_partial(self, mock_save):
        """Stage 1 filled + rotation_enabled → is_partial=True."""
        position = self._make_position(
            qty=0.01,
            stages=[{'stage': 1, 'quantity': 0.005, 'filled': False}],
        )
        position['rotation_enabled'] = True
        state = {
            'positions': {'s1': position},
            'active_direction': None,
        }
        current_pos = {'contracts': 0.005}

        _check_scale_out_fills(state, position, current_pos)

        assert position['is_partial'] is True

    def test_already_filled_not_refilled(self, mock_save):
        """Already filled stage → not re-processed."""
        position = self._make_position(
            qty=0.01,
            stages=[{'stage': 1, 'quantity': 0.005, 'filled': True}],
        )
        position['remaining_quantity'] = 0.005  # Already reduced from prior fill
        state = {
            'positions': {'s1': position},
            'active_direction': None,
        }
        current_pos = {'contracts': 0.005}

        _check_scale_out_fills(state, position, current_pos)

        # Should not re-save since no new fills and qty already synced
        mock_save.assert_not_called()


# ── get_actual_exit_price ────────────────────────────────────


class TestGetActualExitPrice:
    """Test get_actual_exit_price() trade history lookup."""

    def test_no_position_returns_none(self):
        """No position in state → returns None."""
        state = _make_empty_position_state()
        result = get_actual_exit_price(MagicMock(), state, {'symbol': 'BTC/USDT:USDT'})
        assert result is None

    def test_trade_found(self):
        """Close-side trade after entry → returns price + reason."""
        exchange = MagicMock()
        exchange.fetch_my_trades.return_value = [
            {
                'side': 'sell',
                'price': 51000.0,
                'timestamp': 1700000000000,  # 2023-11 — well after entry
            },
        ]
        pos_dict = {
            'direction': 'LONG',
            'entry_time': '2020-01-01T00:00:00',
            'tp_price': 51000.0,
            'sl_price': 49000.0,
        }
        state = _make_position_state(pos_dict)
        result = get_actual_exit_price(exchange, state, {'symbol': 'BTC/USDT:USDT'})
        assert result is not None
        assert result['price'] == 51000.0
        assert result['reason'] == 'TP'

    def test_no_matching_trade(self):
        """No trade matching close side → returns None."""
        exchange = MagicMock()
        exchange.fetch_my_trades.return_value = [
            {'side': 'buy', 'price': 50000.0, 'timestamp': 2000000},
        ]
        pos_dict = {
            'direction': 'LONG',
            'entry_time': '2020-01-01T00:00:00',
            'tp_price': 51000.0,
            'sl_price': 49000.0,
        }
        state = _make_position_state(pos_dict)
        result = get_actual_exit_price(exchange, state, {'symbol': 'BTC/USDT:USDT'})
        assert result is None

    def test_network_error_returns_none(self):
        """Network error → returns None."""
        exchange = MagicMock()
        exchange.fetch_my_trades.side_effect = ccxt.NetworkError('timeout')
        pos_dict = {
            'direction': 'LONG',
            'entry_time': '2020-01-01T00:00:00',
        }
        state = _make_position_state(pos_dict)
        result = get_actual_exit_price(exchange, state, {'symbol': 'BTC/USDT:USDT'})
        assert result is None

    def test_short_exit_buy_side(self):
        """SHORT position → looks for 'buy' trades."""
        exchange = MagicMock()
        exchange.fetch_my_trades.return_value = [
            {'side': 'buy', 'price': 49000.0, 'timestamp': 1700000000000},
        ]
        pos_dict = {
            'direction': 'SHORT',
            'entry_time': '2020-01-01T00:00:00',
            'tp_price': 49000.0,
            'sl_price': 51000.0,
        }
        state = _make_position_state(pos_dict, slot_id='s1')
        # Use position kwarg to pass the specific slot
        result = get_actual_exit_price(exchange, state, {'symbol': 'BTC/USDT:USDT'})
        assert result is not None
        assert result['price'] == 49000.0

    def test_trade_before_entry_ignored(self):
        """Trade before entry_time → ignored."""
        exchange = MagicMock()
        exchange.fetch_my_trades.return_value = [
            {
                'side': 'sell',
                'price': 51000.0,
                'timestamp': 100,  # Very old timestamp
            },
        ]
        pos_dict = {
            'direction': 'LONG',
            'entry_time': '2026-01-01T00:00:00',
            'tp_price': 51000.0,
            'sl_price': 49000.0,
        }
        state = _make_position_state(pos_dict)
        result = get_actual_exit_price(exchange, state, {'symbol': 'BTC/USDT:USDT'})
        assert result is None


# ── check_position_status ────────────────────────────────────


@patch('bingx_rl_trading_bot.scripts.production.pattern_5m.position_monitor.time.sleep')
@patch('bingx_rl_trading_bot.scripts.production.pattern_5m.position_monitor.save_state')
class TestCheckPositionStatus:
    """Test check_position_status() — closed position detection."""

    def test_no_position_returns_false(self, mock_save, mock_sleep):
        """No position in state → returns False."""
        state = _make_empty_position_state()
        result = check_position_status(
            MagicMock(), state, {'symbol': 'BTC/USDT:USDT'},
            APICache()
        )
        assert result is False

    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.position_close.record_closed_position')
    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.exchange._interruptible_api_sleep',
           new=MagicMock())
    def test_position_still_open(self, mock_record, mock_save, mock_sleep):
        """Position still on exchange → returns False."""
        exchange = MagicMock()
        exchange.fetch_positions.return_value = [
            {'side': 'long', 'contracts': 0.01},
        ]
        pos_dict = {
            'direction': 'LONG',
            'entry_price': 50000.0,
            'quantity': 0.01,
            'remaining_quantity': 0.01,
            'scale_out_enabled': False,
        }
        state = _make_position_state(pos_dict)
        state.update({
            'total_trades': 0, 'winning_trades': 0, 'total_pnl': 0.0,
            'daily_trades': 0, 'daily_pnl': 0.0, 'consecutive_losses': 0,
            'last_trade': None, 'last_signal_time': None,
        })
        cache = APICache()
        result = check_position_status(exchange, state, {'symbol': 'BTC/USDT:USDT'}, cache)
        assert result is False
        mock_record.assert_not_called()

    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.position_close.record_closed_position')
    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.exchange._interruptible_api_sleep',
           new=MagicMock())
    def test_position_closed_detected(self, mock_record, mock_save, mock_sleep):
        """Position gone from exchange → detected as closed."""
        exchange = MagicMock()
        exchange.fetch_positions.return_value = []  # No positions
        exchange.fetch_my_trades.return_value = [
            {'side': 'sell', 'price': 51000.0, 'timestamp': 2000000},
        ]
        exchange.fetch_ticker.return_value = {'last': 51000.0}
        pos_dict = {
            'direction': 'LONG',
            'entry_price': 50000.0,
            'entry_time': '2020-01-01T00:00:00',
            'quantity': 0.01,
            'remaining_quantity': 0.01,
            'tp_price': 51000.0,
            'sl_price': 49000.0,
            'scale_out_enabled': False,
        }
        state = _make_position_state(pos_dict)
        state.update({
            'total_trades': 0, 'winning_trades': 0, 'total_pnl': 0.0,
            'daily_trades': 0, 'daily_pnl': 0.0, 'consecutive_losses': 0,
            'last_trade': None, 'last_signal_time': None,
        })
        cache = APICache()
        result = check_position_status(exchange, state, {'symbol': 'BTC/USDT:USDT'}, cache)
        assert result is True

    def test_network_error_returns_false(self, mock_save, mock_sleep):
        """Network error → returns False (safe default)."""
        exchange = MagicMock()
        exchange.fetch_positions.side_effect = ccxt.NetworkError('timeout')
        pos_dict = {
            'direction': 'LONG',
            'entry_price': 50000.0,
        }
        state = _make_position_state(pos_dict)
        cache = APICache()
        result = check_position_status(exchange, state, {'symbol': 'BTC/USDT:USDT'}, cache)
        assert result is False


# ── sync_position_with_exchange ──────────────────────────────


@patch('bingx_rl_trading_bot.scripts.production.pattern_5m.exchange._interruptible_api_sleep',
       new=MagicMock())
class TestSyncPositionWithExchange:
    """Test sync_position_with_exchange() — per-direction reconciliation."""

    def _make_state_with_position(self):
        pos_dict = {
            'slot_id': 's1',
            'direction': 'LONG',
            'entry_price': 50000.0,
            'entry_time': '2026-01-01T00:00:00',
            'quantity': 0.01,
            'tp_price': 51000.0,
            'sl_price': 49000.0,
        }
        return {
            'positions': {'s1': pos_dict},
            'active_direction': 'LONG',
            'total_trades': 5,
            'winning_trades': 3,
            'total_pnl': 10.0,
            'daily_trades': 1,
            'daily_pnl': 2.0,
            'consecutive_losses': 0,
            'last_trade': None,
            'last_signal_time': None,
        }

    def test_both_match_no_sync(self):
        """State and exchange agree → returns False."""
        exchange = MagicMock()
        exchange.fetch_positions.return_value = [
            {'side': 'long', 'contracts': 0.01},
        ]
        state = self._make_state_with_position()
        cache = APICache()
        config = {'symbol': 'BTC/USDT:USDT'}
        result = sync_position_with_exchange(exchange, state, config, cache)
        assert result is False

    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.position_close.record_closed_position')
    def test_state_has_exchange_doesnt(self, mock_record):
        """State has position, exchange doesn't → records close."""
        exchange = MagicMock()
        exchange.fetch_positions.return_value = []  # Empty
        exchange.fetch_my_trades.return_value = [
            {'side': 'sell', 'price': 51000.0, 'timestamp': 2000000},
        ]
        state = self._make_state_with_position()
        cache = APICache()
        config = {'symbol': 'BTC/USDT:USDT'}
        result = sync_position_with_exchange(exchange, state, config, cache)
        assert result is True
        mock_record.assert_called_once()

    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.position_close.recover_position_to_state')
    def test_exchange_has_state_doesnt(self, mock_recover):
        """Exchange has position, state doesn't → recovers."""
        exchange = MagicMock()
        exchange.fetch_positions.return_value = [
            {'side': 'long', 'contracts': 0.01, 'entryPrice': 50000.0},
        ]
        state = _make_empty_position_state()
        state.update({
            'total_trades': 0, 'winning_trades': 0, 'total_pnl': 0.0,
            'daily_trades': 0, 'daily_pnl': 0.0, 'consecutive_losses': 0,
            'last_trade': None, 'last_signal_time': None,
        })
        cache = APICache()
        config = {'symbol': 'BTC/USDT:USDT'}
        result = sync_position_with_exchange(exchange, state, config, cache)
        assert result is True
        mock_recover.assert_called_once()

    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.position_close.recover_position_to_state')
    def test_exchange_has_short_state_doesnt(self, mock_recover):
        """Exchange has SHORT, state empty → recovers as SHORT."""
        exchange = MagicMock()
        exchange.fetch_positions.return_value = [
            {'side': 'short', 'contracts': 0.01, 'entryPrice': 50000.0},
        ]
        state = _make_empty_position_state()
        state.update({
            'total_trades': 0, 'winning_trades': 0, 'total_pnl': 0.0,
            'daily_trades': 0, 'daily_pnl': 0.0, 'consecutive_losses': 0,
            'last_trade': None, 'last_signal_time': None,
        })
        cache = APICache()
        config = {'symbol': 'BTC/USDT:USDT'}
        result = sync_position_with_exchange(exchange, state, config, cache)
        assert result is True
        mock_recover.assert_called_once()
        # Check that direction 'SHORT' was passed
        assert mock_recover.call_args[0][3] == 'SHORT'

    def test_network_error_returns_false(self):
        """Network error during sync → returns False."""
        exchange = MagicMock()
        exchange.fetch_positions.side_effect = ccxt.NetworkError('timeout')
        state = self._make_state_with_position()
        cache = APICache()
        config = {'symbol': 'BTC/USDT:USDT'}
        result = sync_position_with_exchange(exchange, state, config, cache)
        assert result is False

    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.position_close.record_closed_position')
    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.position_close.recover_position_to_state')
    def test_direction_mismatch(self, mock_recover, mock_record):
        """State=LONG but exchange has SHORT → close LONG slots + recover SHORT."""
        exchange = MagicMock()
        exchange.fetch_positions.return_value = [
            {'side': 'short', 'contracts': 0.01, 'entryPrice': 49000.0},
        ]
        exchange.fetch_my_trades.return_value = []
        exchange.fetch_ticker.return_value = {'last': 50000.0}
        state = self._make_state_with_position()
        cache = APICache()
        config = {'symbol': 'BTC/USDT:USDT'}
        result = sync_position_with_exchange(exchange, state, config, cache)
        assert result is True
        # LONG slot closed (exchange has no LONG position)
        mock_record.assert_called_once()
        # SHORT orphan recovered (exchange has SHORT, no bot SHORT slots)
        mock_recover.assert_called_once()

    def test_no_position_either_side(self):
        """No position in state or exchange → returns False."""
        exchange = MagicMock()
        exchange.fetch_positions.return_value = []
        state = _make_empty_position_state()
        cache = APICache()
        config = {'symbol': 'BTC/USDT:USDT'}
        result = sync_position_with_exchange(exchange, state, config, cache)
        assert result is False

    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.position_close.record_closed_position')
    def test_state_closed_ticker_fallback(self, mock_record):
        """State has pos, exchange doesn't, no trade → ticker fallback."""
        exchange = MagicMock()
        exchange.fetch_positions.return_value = []
        exchange.fetch_my_trades.return_value = []
        exchange.fetch_ticker.return_value = {'last': 50500.0}
        state = self._make_state_with_position()
        cache = APICache()
        config = {'symbol': 'BTC/USDT:USDT'}
        result = sync_position_with_exchange(exchange, state, config, cache)
        assert result is True

    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.position_close.record_closed_position')
    def test_state_closed_ticker_fails_entry_fallback(self, mock_record):
        """State has pos, exchange doesn't, ticker fails → entry_price fallback."""
        exchange = MagicMock()
        exchange.fetch_positions.return_value = []
        exchange.fetch_my_trades.return_value = []
        exchange.fetch_ticker.side_effect = Exception('no ticker')
        state = self._make_state_with_position()
        cache = APICache()
        config = {'symbol': 'BTC/USDT:USDT'}
        result = sync_position_with_exchange(exchange, state, config, cache)
        assert result is True
        # Fallback to entry_price
        assert mock_record.call_args[0][3] == 50000.0

    def test_exchange_error_returns_false(self):
        """ExchangeError during sync → returns False."""
        exchange = MagicMock()
        exchange.fetch_positions.side_effect = ccxt.ExchangeError('invalid')
        state = self._make_state_with_position()
        cache = APICache()
        config = {'symbol': 'BTC/USDT:USDT'}
        result = sync_position_with_exchange(exchange, state, config, cache)
        assert result is False

    def test_generic_exception_returns_false(self):
        """Generic exception during sync → returns False."""
        exchange = MagicMock()
        exchange.fetch_positions.side_effect = RuntimeError('unexpected')
        state = self._make_state_with_position()
        cache = APICache()
        config = {'symbol': 'BTC/USDT:USDT'}
        result = sync_position_with_exchange(exchange, state, config, cache)
        assert result is False


# ── get_actual_exit_price additional paths ────────────────────


class TestGetActualExitPriceExtended:
    """Test get_actual_exit_price error paths."""

    def test_invalid_entry_time_still_works(self):
        """Invalid entry_time → entry_ts=0, still finds trades."""
        exchange = MagicMock()
        exchange.fetch_my_trades.return_value = [
            {'side': 'sell', 'price': 51000.0, 'timestamp': 100},
        ]
        pos_dict = {
            'direction': 'LONG',
            'entry_time': 'invalid-datetime',
            'tp_price': 51000.0,
            'sl_price': 49000.0,
        }
        state = _make_position_state(pos_dict)
        result = get_actual_exit_price(exchange, state, {'symbol': 'BTC/USDT:USDT'})
        assert result is not None

    def test_network_error_returns_none(self):
        """NetworkError → returns None."""
        exchange = MagicMock()
        exchange.fetch_my_trades.side_effect = ccxt.NetworkError('timeout')
        pos_dict = {'direction': 'LONG', 'entry_time': '2026-01-01T00:00:00'}
        state = _make_position_state(pos_dict)
        result = get_actual_exit_price(exchange, state, {'symbol': 'BTC/USDT:USDT'})
        assert result is None

    def test_exchange_error_returns_none(self):
        """ExchangeError → returns None."""
        exchange = MagicMock()
        exchange.fetch_my_trades.side_effect = ccxt.ExchangeError('invalid')
        pos_dict = {'direction': 'LONG', 'entry_time': '2026-01-01T00:00:00'}
        state = _make_position_state(pos_dict)
        result = get_actual_exit_price(exchange, state, {'symbol': 'BTC/USDT:USDT'})
        assert result is None

    def test_generic_exception_returns_none(self):
        """Generic exception → returns None."""
        exchange = MagicMock()
        exchange.fetch_my_trades.side_effect = RuntimeError('unexpected')
        pos_dict = {'direction': 'LONG', 'entry_time': '2026-01-01T00:00:00'}
        state = _make_position_state(pos_dict)
        result = get_actual_exit_price(exchange, state, {'symbol': 'BTC/USDT:USDT'})
        assert result is None


# ── check_position_status additional paths ────────────────────


@patch('bingx_rl_trading_bot.scripts.production.pattern_5m.position_monitor.time.sleep')
@patch('bingx_rl_trading_bot.scripts.production.pattern_5m.position_monitor.save_state')
class TestCheckPositionStatusExtended:
    """Test check_position_status direction mismatch and scale-out paths."""

    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.position_monitor._handle_position_closed')
    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.exchange._interruptible_api_sleep',
           new=MagicMock())
    def test_direction_mismatch_recovers(self, mock_handle, mock_save, mock_sleep):
        """State=LONG but exchange=SHORT → LONG slot closed (no LONG on exchange)."""
        exchange = MagicMock()
        exchange.fetch_positions.return_value = [
            {'side': 'short', 'contracts': 0.01, 'entryPrice': 49000.0},
        ]
        pos_dict = {
            'direction': 'LONG', 'entry_price': 50000.0,
            'quantity': 0.01, 'remaining_quantity': 0.01,
            'entry_time': '2026-01-01T00:00:00',
            'tp_price': 51000.0, 'sl_price': 49000.0,
            'scale_out_enabled': False,
        }
        state = _make_position_state(pos_dict)
        cache = APICache()
        result = check_position_status(exchange, state, {'symbol': 'BTC/USDT:USDT'}, cache)
        assert result is True
        # LONG slot detected as closed (exchange has no LONG position)
        mock_handle.assert_called_once()

    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.exchange._interruptible_api_sleep',
           new=MagicMock())
    def test_exchange_error_returns_false(self, mock_save, mock_sleep):
        """ExchangeError → returns False."""
        exchange = MagicMock()
        exchange.fetch_positions.side_effect = ccxt.ExchangeError('invalid')
        pos_dict = {'direction': 'LONG', 'entry_price': 50000.0}
        state = _make_position_state(pos_dict)
        cache = APICache()
        result = check_position_status(exchange, state, {'symbol': 'BTC/USDT:USDT'}, cache)
        assert result is False

    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.exchange._interruptible_api_sleep',
           new=MagicMock())
    def test_generic_exception_returns_false(self, mock_save, mock_sleep):
        """Generic exception → returns False."""
        exchange = MagicMock()
        exchange.fetch_positions.side_effect = RuntimeError('unexpected')
        pos_dict = {'direction': 'LONG', 'entry_price': 50000.0}
        state = _make_position_state(pos_dict)
        cache = APICache()
        result = check_position_status(exchange, state, {'symbol': 'BTC/USDT:USDT'}, cache)
        assert result is False

    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.position_close.record_closed_position')
    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.exchange._interruptible_api_sleep',
           new=MagicMock())
    def test_scale_out_enabled_checks_fills(self, mock_record, mock_save, mock_sleep):
        """Scale-out enabled + position open → checks scale-out fills."""
        exchange = MagicMock()
        exchange.fetch_positions.return_value = [
            {'side': 'long', 'contracts': 0.005},  # Partial fill
        ]
        pos_dict = {
            'direction': 'LONG', 'entry_price': 50000.0,
            'quantity': 0.01, 'remaining_quantity': 0.01,
            'scale_out_enabled': True,
            'scale_out_stages': [
                {'stage': 1, 'quantity': 0.005, 'filled': False, 'order_id': 'so1'},
            ],
        }
        state = _make_position_state(pos_dict)
        cache = APICache()
        result = check_position_status(exchange, state, {'symbol': 'BTC/USDT:USDT'}, cache)
        assert result is False  # Position still open


# ── _handle_position_closed (lines 343, 348-349, 358-363) ────


class TestHandlePositionClosed:
    """Test _handle_position_closed exit detection and scale-out reason adjustment."""

    @pytest.fixture
    def base_position(self):
        return {
            'slot_id': 's1',
            'direction': 'LONG',
            'entry_price': 50000.0,
            'tp_price': 51000.0,
            'sl_price': 49000.0,
            'scale_out_enabled': False,
        }

    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.position_monitor.time.sleep')
    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.position_close.record_closed_position')
    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.position_monitor.get_actual_exit_price',
           return_value={'price': 51000.0, 'reason': 'TP'})
    def test_actual_exit_found_on_first_try(self, mock_exit, mock_record, mock_sleep,
                                             base_position):
        """get_actual_exit_price succeeds → uses actual exit (lines 343, 348-349)."""
        exchange = MagicMock()
        state = _make_position_state(base_position, slot_id='s1')
        config = {'symbol': 'BTC/USDT:USDT'}
        cache = APICache()

        result = _handle_position_closed(exchange, state, config, base_position, cache, None)
        assert result is True
        mock_record.assert_called_once()
        # Check that it used TP as exit reason
        call_args = mock_record.call_args
        assert call_args[0][3] == 51000.0  # exit_price
        assert call_args[0][4] == 'TP'  # exit_reason

    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.position_monitor.time.sleep')
    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.position_close.record_closed_position')
    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.position_monitor.get_actual_exit_price',
           return_value={'price': 51000.0, 'reason': 'TP'})
    def test_scaleout_all_stages_filled(self, mock_exit, mock_record, mock_sleep):
        """scale_out_enabled + all stages filled → TP_SCALEOUT (lines 358-361)."""
        position = {
            'slot_id': 's1',
            'direction': 'SHORT', 'entry_price': 50000.0,
            'tp_price': 49000.0, 'sl_price': 51000.0,
            'scale_out_enabled': True,
            'scale_out_stages': [
                {'stage': 1, 'quantity': 0.005, 'filled': True},
                {'stage': 2, 'quantity': 0.005, 'filled': True},
            ],
        }
        exchange = MagicMock()
        state = _make_position_state(position, slot_id='s1')
        config = {'symbol': 'BTC/USDT:USDT'}
        cache = APICache()

        result = _handle_position_closed(exchange, state, config, position, cache, None)
        assert result is True
        call_args = mock_record.call_args
        assert call_args[0][4] == 'TP_SCALEOUT'

    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.position_monitor.time.sleep')
    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.position_close.record_closed_position')
    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.position_monitor.get_actual_exit_price',
           return_value={'price': 51000.0, 'reason': 'SL'})
    def test_scaleout_partial_sl(self, mock_exit, mock_record, mock_sleep):
        """scale_out_enabled + partial fills + SL → SL_AFTER_1_STAGES (lines 362-363)."""
        position = {
            'slot_id': 's1',
            'direction': 'LONG', 'entry_price': 50000.0,
            'tp_price': 51000.0, 'sl_price': 49000.0,
            'scale_out_enabled': True,
            'scale_out_stages': [
                {'stage': 1, 'quantity': 0.005, 'filled': True},
                {'stage': 2, 'quantity': 0.005, 'filled': False},
            ],
        }
        exchange = MagicMock()
        state = _make_position_state(position, slot_id='s1')
        config = {'symbol': 'BTC/USDT:USDT'}
        cache = APICache()

        result = _handle_position_closed(exchange, state, config, position, cache, None)
        assert result is True
        call_args = mock_record.call_args
        assert call_args[0][4] == 'SL_AFTER_1_STAGES'


# ── sync_position_with_exchange — actual_exit truthy ────────────────


@patch('bingx_rl_trading_bot.scripts.production.pattern_5m.exchange._interruptible_api_sleep',
       new=MagicMock())
class TestSyncPositionActualExitTruthy:
    """Cover line 86: actual_exit is truthy → record_closed_position with actual price."""

    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.position_close.record_closed_position')
    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.position_monitor.get_actual_exit_price',
           return_value={'price': 51500.0, 'reason': 'TP'})
    def test_actual_exit_truthy_calls_record(self, mock_exit, mock_record):
        """get_actual_exit_price returns truthy → uses actual price (line 86)."""
        exchange = MagicMock()
        exchange.fetch_positions.return_value = []  # no exchange position
        pos_dict = {
            'slot_id': 's1',
            'direction': 'LONG',
            'entry_price': 50000.0,
            'entry_time': '2026-01-01T00:00:00',
            'quantity': 0.01,
            'tp_price': 51000.0,
            'sl_price': 49000.0,
        }
        state = {
            'positions': {'s1': pos_dict},
            'active_direction': 'LONG',
            'total_trades': 5,
            'winning_trades': 3,
            'total_pnl': 10.0,
            'daily_trades': 1,
            'daily_pnl': 2.0,
            'consecutive_losses': 0,
            'last_trade': None,
            'last_signal_time': None,
        }
        cache = APICache()
        config = {'symbol': 'BTC/USDT:USDT'}
        result = sync_position_with_exchange(exchange, state, config, cache)
        assert result is True
        mock_record.assert_called_once()
        # Verify actual exit price was used
        call_args = mock_record.call_args
        assert call_args[0][3] == 51500.0  # exit_price
        assert call_args[0][4] == 'TP'  # exit_reason


# ── Per-slot TP/SL fill detection (v1.29.1) ─────────────────


def _make_two_slot_state(qty_each=0.01):
    """Build state with 2 LONG slots for per-slot fill detection tests."""
    s1 = {
        'slot_id': 's1',
        'direction': 'LONG',
        'entry_price': 50000.0,
        'entry_time': '2026-01-01T00:00:00',
        'quantity': qty_each,
        'remaining_quantity': qty_each,
        'tp_price': 51000.0,
        'sl_price': 49000.0,
        'tp_order_id': 'tp_s1',
        'sl_order_id': 'sl_s1',
        'scale_out_enabled': False,
    }
    s2 = {
        'slot_id': 's2',
        'direction': 'LONG',
        'entry_price': 50100.0,
        'entry_time': '2026-01-01T00:05:00',
        'quantity': qty_each,
        'remaining_quantity': qty_each,
        'tp_price': 51100.0,
        'sl_price': 49100.0,
        'tp_order_id': 'tp_s2',
        'sl_order_id': 'sl_s2',
        'scale_out_enabled': False,
    }
    return {
        'positions': {'s1': s1, 's2': s2},
        'active_direction': 'LONG',
        'total_trades': 0,
        'winning_trades': 0,
        'total_pnl': 0.0,
        'daily_trades': 0,
        'daily_pnl': 0.0,
        'consecutive_losses': 0,
        'last_trade': None,
        'last_signal_time': None,
    }


@patch('bingx_rl_trading_bot.scripts.production.pattern_5m.position_monitor.time.sleep')
@patch('bingx_rl_trading_bot.scripts.production.pattern_5m.position_monitor.save_state')
class TestPerSlotFillDetection:
    """Test per-slot TP/SL fill detection in check_position_status()."""

    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.position_close.record_closed_position')
    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.exchange._interruptible_api_sleep',
           new=MagicMock())
    def test_per_slot_tp_fill_detection(self, mock_record, mock_save, mock_sleep):
        """Slot s1 TP filled (TP order gone, SL still active) → detected and closed."""
        exchange = MagicMock()
        # Exchange still has LONG position but qty = s2 only (s1's TP filled)
        exchange.fetch_positions.return_value = [
            {'side': 'long', 'contracts': 0.01},  # only s2 remains
        ]
        # s1's SL still active, s2's TP+SL still active. s1's TP is GONE.
        exchange.fetch_open_orders.return_value = [
            {'id': 'sl_s1'},
            {'id': 'tp_s2'},
            {'id': 'sl_s2'},
        ]
        exchange.fetch_my_trades.return_value = [
            {'side': 'sell', 'price': 51000.0, 'timestamp': 2000000},
        ]
        exchange.fetch_ticker.return_value = {'last': 51000.0}

        state = _make_two_slot_state(qty_each=0.01)
        cache = APICache()
        result = check_position_status(exchange, state, {'symbol': 'BTC/USDT:USDT'}, cache)
        assert result is True
        # record_closed_position called for s1
        mock_record.assert_called_once()
        closed_slot = mock_record.call_args[1].get('position') or mock_record.call_args[0][7]
        assert closed_slot['slot_id'] == 's1'

    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.position_close.record_closed_position')
    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.exchange._interruptible_api_sleep',
           new=MagicMock())
    def test_per_slot_sl_fill_detection(self, mock_record, mock_save, mock_sleep):
        """Slot s1 SL filled (SL order gone, TP still active) → detected and closed."""
        exchange = MagicMock()
        exchange.fetch_positions.return_value = [
            {'side': 'long', 'contracts': 0.01},
        ]
        # s1's TP still active but SL is GONE
        exchange.fetch_open_orders.return_value = [
            {'id': 'tp_s1'},
            {'id': 'tp_s2'},
            {'id': 'sl_s2'},
        ]
        exchange.fetch_my_trades.return_value = [
            {'side': 'sell', 'price': 49000.0, 'timestamp': 2000000},
        ]
        exchange.fetch_ticker.return_value = {'last': 49000.0}

        state = _make_two_slot_state(qty_each=0.01)
        cache = APICache()
        result = check_position_status(exchange, state, {'symbol': 'BTC/USDT:USDT'}, cache)
        assert result is True
        mock_record.assert_called_once()
        closed_slot = mock_record.call_args[1].get('position') or mock_record.call_args[0][7]
        assert closed_slot['slot_id'] == 's1'

    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.position_close.record_closed_position')
    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.exchange._interruptible_api_sleep',
           new=MagicMock())
    def test_per_slot_both_orders_gone(self, mock_record, mock_save, mock_sleep):
        """Slot s1 both TP and SL gone → detected and closed."""
        exchange = MagicMock()
        exchange.fetch_positions.return_value = [
            {'side': 'long', 'contracts': 0.01},
        ]
        # Only s2's orders remain — s1's TP and SL both gone
        exchange.fetch_open_orders.return_value = [
            {'id': 'tp_s2'},
            {'id': 'sl_s2'},
        ]
        exchange.fetch_my_trades.return_value = [
            {'side': 'sell', 'price': 50500.0, 'timestamp': 2000000},
        ]
        exchange.fetch_ticker.return_value = {'last': 50500.0}

        state = _make_two_slot_state(qty_each=0.01)
        cache = APICache()
        result = check_position_status(exchange, state, {'symbol': 'BTC/USDT:USDT'}, cache)
        assert result is True
        mock_record.assert_called_once()
        closed_slot = mock_record.call_args[1].get('position') or mock_record.call_args[0][7]
        assert closed_slot['slot_id'] == 's1'

    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.position_close.record_closed_position')
    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.exchange._interruptible_api_sleep',
           new=MagicMock())
    def test_per_slot_exchange_managed_skip(self, mock_record, mock_save, mock_sleep):
        """EXCHANGE_MANAGED slot → skipped in per-slot detection."""
        exchange = MagicMock()
        exchange.fetch_positions.return_value = [
            {'side': 'long', 'contracts': 0.01},
        ]
        exchange.fetch_open_orders.return_value = [
            {'id': 'tp_s2'},
            {'id': 'sl_s2'},
        ]

        state = _make_two_slot_state(qty_each=0.01)
        # s1 has EXCHANGE_MANAGED TP → should be skipped
        state['positions']['s1']['tp_order_id'] = _EXCHANGE_MANAGED
        cache = APICache()
        result = check_position_status(exchange, state, {'symbol': 'BTC/USDT:USDT'}, cache)
        assert result is False
        mock_record.assert_not_called()

    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.position_close.record_closed_position')
    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.exchange._interruptible_api_sleep',
           new=MagicMock())
    def test_per_slot_no_mismatch_no_detection(self, mock_record, mock_save, mock_sleep):
        """Exchange qty matches local sum → no per-slot detection triggered."""
        exchange = MagicMock()
        exchange.fetch_positions.return_value = [
            {'side': 'long', 'contracts': 0.02},  # matches s1 + s2
        ]

        state = _make_two_slot_state(qty_each=0.01)
        cache = APICache()
        result = check_position_status(exchange, state, {'symbol': 'BTC/USDT:USDT'}, cache)
        assert result is False
        mock_record.assert_not_called()
        # fetch_open_orders should NOT be called (no mismatch)
        exchange.fetch_open_orders.assert_not_called()
