"""F v3 (f_v3_limit_close) unit tests.

Validates:
  enabled=false → MARKET behavior (legacy, F v2 path)
  enabled=true + TRAIL_TP → LIMIT placement, poll for fill, MARKET fallback on timeout
  enabled=true + non-TRAIL_TP (SL/Emergency/Timeout) → MARKET path (LIMIT only for trail)
  exit_method 필드 trade_history에 기록

Critical angles:
  A. Edge: timeout → MARKET fallback
  B. Parity: enabled=false matches F v2 exactly (no LIMIT call)
  C. Reason: SL/Emergency/Timeout는 LIMIT path 우회 (LIMIT은 trail에만)
  D. Rollback: missing config key → default false, MARKET behavior
  E. Filled fast: status='closed' immediate → return LIMIT method
"""
from unittest.mock import MagicMock, patch
import pytest


def _setup_position(mock_bot, direction='LONG', entry=70000.0):
    mock_bot.positions = [{
        'direction': direction, 'entry_price': entry, 'sl_price': entry * 0.99,
        'best_price': entry * 1.01, 'entry_time': '2026-04-27T12:00:00',
        'bars_held': 5, 'size_pct': 100.0, 'sl_order_id': 's1',
    }]


def _live_pos(direction='LONG', qty=0.001):
    return {direction: {'contracts': qty, 'side': direction.lower(),
                        'entry': 70000.0}}


class TestExitMethodGate:
    """F v3 enabled=false → MARKET path; enabled=true + TRAIL_TP → LIMIT path."""

    def test_fv3_disabled_uses_market(self, mock_bot):
        """B. Parity: F v3 disabled → _exchange_close (MARKET)."""
        mock_bot.config['strategy']['f_v3_limit_close'] = {'enabled': False}
        _setup_position(mock_bot)
        mock_bot._exchange_close = MagicMock(return_value=(70100.0, 1234567))

        mock_bot._do_close(0, {'reason': 'TRAIL_TP', 'exit_price': 70200.0})

        mock_bot._exchange_close.assert_called_once_with('LONG')
        assert mock_bot.trade_history[0]['exit_method'] == 'MARKET'

    def test_fv3_missing_config_uses_market(self, mock_bot):
        """D. Rollback: missing f_v3_limit_close key → default false → MARKET."""
        # No f_v3_limit_close key set
        _setup_position(mock_bot)
        mock_bot._exchange_close = MagicMock(return_value=(70100.0, 1234567))

        mock_bot._do_close(0, {'reason': 'TRAIL_TP', 'exit_price': 70200.0})

        mock_bot._exchange_close.assert_called_once_with('LONG')
        assert mock_bot.trade_history[0]['exit_method'] == 'MARKET'

    def test_fv3_enabled_sl_uses_market(self, mock_bot):
        """C. Reason: SL is NOT LIMIT-eligible — uses MARKET even with F v3 enabled."""
        mock_bot.config['strategy']['f_v3_limit_close'] = {'enabled': True}
        _setup_position(mock_bot)
        mock_bot._exchange_close = MagicMock(return_value=(70000.0, 1234567))

        mock_bot._do_close(0, {'reason': 'SL', 'exit_price': 69300.0})

        mock_bot._exchange_close.assert_called_once_with('LONG')
        assert mock_bot.trade_history[0]['exit_method'] == 'MARKET'

    def test_fv3_enabled_emergency_uses_market(self, mock_bot):
        """C. Emergency = MARKET (crash safety, no LIMIT)."""
        mock_bot.config['strategy']['f_v3_limit_close'] = {'enabled': True}
        _setup_position(mock_bot)
        mock_bot._exchange_close = MagicMock(return_value=(67900.0, 1234567))

        mock_bot._do_close(0, {'reason': 'EMERGENCY', 'exit_price': 67900.0})

        mock_bot._exchange_close.assert_called_once_with('LONG')

    def test_fv3_enabled_timeout_uses_market(self, mock_bot):
        """C. Timeout = MARKET (no LIMIT for forced close)."""
        mock_bot.config['strategy']['f_v3_limit_close'] = {'enabled': True}
        _setup_position(mock_bot)
        mock_bot._exchange_close = MagicMock(return_value=(69500.0, 1234567))

        mock_bot._do_close(0, {'reason': 'TIMEOUT', 'exit_price': 69500.0})

        mock_bot._exchange_close.assert_called_once_with('LONG')


class TestLimitClose:
    """F v3 LIMIT placement + polling logic."""

    def test_no_live_position_falls_back_market(self, mock_bot):
        """A. Edge: no live position visible → MARKET fallback."""
        mock_bot._get_live_positions = MagicMock(return_value={})
        mock_bot._exchange_close = MagicMock(return_value=(70100.0, 1234567))

        fill, ts, method = mock_bot._exchange_close_limit('LONG', 70200.0, timeout_s=60)

        assert method == 'MARKET_FALLBACK'
        mock_bot._exchange_close.assert_called_once()

    def test_limit_filled_immediately(self, mock_bot):
        """E. Filled fast: status='closed' on first poll → method='LIMIT'."""
        mock_bot._get_live_positions = MagicMock(return_value=_live_pos('LONG', 0.001))
        mock_bot.exchange.create_order.return_value = {'id': 'lim1'}
        mock_bot.exchange.fetch_order.return_value = {
            'status': 'closed', 'average': 70195.0, 'price': 70200.0,
            'timestamp': 1234567890,
        }

        with patch('time.sleep'):  # speed up test
            fill, ts, method = mock_bot._exchange_close_limit('LONG', 70200.0, timeout_s=60)

        assert method == 'LIMIT'
        assert fill == 70195.0
        mock_bot.exchange.create_order.assert_called_once()
        # LIMIT params: side, qty, price, reduceOnly
        call_args = mock_bot.exchange.create_order.call_args
        assert call_args[0][1] == 'limit'  # type
        assert call_args[0][2] == 'sell'   # side (LONG → sell to close)
        assert call_args[1]['params']['reduceOnly'] is True

    def test_limit_short_uses_buy_side(self, mock_bot):
        """A. SHORT close = BUY side."""
        mock_bot._get_live_positions = MagicMock(return_value=_live_pos('SHORT', 0.001))
        mock_bot.exchange.create_order.return_value = {'id': 'lim2'}
        mock_bot.exchange.fetch_order.return_value = {
            'status': 'closed', 'average': 69900.0, 'timestamp': 1234567890,
        }

        with patch('time.sleep'):
            fill, ts, method = mock_bot._exchange_close_limit('SHORT', 69900.0, timeout_s=60)

        assert method == 'LIMIT'
        call_args = mock_bot.exchange.create_order.call_args
        assert call_args[0][2] == 'buy'  # SHORT → buy to close

    def test_limit_placement_error_falls_back(self, mock_bot):
        """A. Edge: LIMIT placement raises → MARKET_ERROR_FALLBACK."""
        mock_bot._get_live_positions = MagicMock(return_value=_live_pos('LONG', 0.001))
        mock_bot.exchange.create_order.side_effect = Exception("API error")
        mock_bot._exchange_close = MagicMock(return_value=(70300.0, 1234567))

        fill, ts, method = mock_bot._exchange_close_limit('LONG', 70200.0, timeout_s=60)

        assert method == 'MARKET_ERROR_FALLBACK'
        mock_bot._exchange_close.assert_called_once()

    def test_limit_timeout_falls_back(self, mock_bot):
        """A. Timeout: status='open' throughout → cancel + MARKET_FALLBACK."""
        mock_bot._get_live_positions = MagicMock(return_value=_live_pos('LONG', 0.001))
        mock_bot.exchange.create_order.return_value = {'id': 'lim3'}
        mock_bot.exchange.fetch_order.return_value = {'status': 'open'}
        mock_bot._exchange_close = MagicMock(return_value=(70300.0, 1234567))

        # Patch time.time to fast-forward past timeout
        time_seq = [0.0, 0.5, 1.0, 1.5, 100.0, 100.5]  # 마지막 값으로 timeout 트리거
        with patch('time.sleep'), patch('time.time', side_effect=lambda: time_seq.pop(0) if time_seq else 100.0):
            fill, ts, method = mock_bot._exchange_close_limit('LONG', 70200.0, timeout_s=10)

        assert method == 'MARKET_FALLBACK'
        mock_bot.exchange.cancel_order.assert_called_once()
        mock_bot._exchange_close.assert_called_once()

    def test_limit_canceled_externally(self, mock_bot):
        """E. Edge: external cancel → break poll → MARKET_FALLBACK."""
        mock_bot._get_live_positions = MagicMock(return_value=_live_pos('LONG', 0.001))
        mock_bot.exchange.create_order.return_value = {'id': 'lim4'}
        mock_bot.exchange.fetch_order.return_value = {'status': 'canceled'}
        mock_bot._exchange_close = MagicMock(return_value=(70250.0, 1234567))

        with patch('time.sleep'):
            fill, ts, method = mock_bot._exchange_close_limit('LONG', 70200.0, timeout_s=60)

        assert method == 'MARKET_FALLBACK'

    def test_filled_returns_target_when_avg_missing(self, mock_bot):
        """A. Edge: order has no 'average' → use target_price as fill."""
        mock_bot._get_live_positions = MagicMock(return_value=_live_pos('LONG', 0.001))
        mock_bot.exchange.create_order.return_value = {'id': 'lim5'}
        mock_bot.exchange.fetch_order.return_value = {
            'status': 'closed', 'timestamp': 1234567890,
        }

        with patch('time.sleep'):
            fill, ts, method = mock_bot._exchange_close_limit('LONG', 70200.0, timeout_s=60)

        assert method == 'LIMIT'
        assert fill == 70200.0  # fell back to target


class TestTradeHistoryRecord:
    """exit_method 필드 정확 기록."""

    def test_market_records_method(self, mock_bot):
        """B. trade_history exit_method = 'MARKET' for non-LIMIT path."""
        mock_bot.config['strategy']['f_v3_limit_close'] = {'enabled': False}
        _setup_position(mock_bot)
        mock_bot._exchange_close = MagicMock(return_value=(70100.0, 1234567))

        mock_bot._do_close(0, {'reason': 'TRAIL_TP', 'exit_price': 70200.0})

        assert mock_bot.trade_history[0]['exit_method'] == 'MARKET'

    def test_limit_records_method(self, mock_bot):
        """B. trade_history exit_method = 'LIMIT' for filled LIMIT."""
        mock_bot.config['strategy']['f_v3_limit_close'] = {'enabled': True}
        _setup_position(mock_bot)
        # Bypass _exchange_close_limit, simulate it returning LIMIT
        mock_bot._exchange_close_limit = MagicMock(return_value=(70195.0, 1234567, 'LIMIT'))

        mock_bot._do_close(0, {'reason': 'TRAIL_TP', 'exit_price': 70200.0})

        assert mock_bot.trade_history[0]['exit_method'] == 'LIMIT'
        assert mock_bot.trade_history[0]['exit_price'] == 70195.0

    def test_market_fallback_records_method(self, mock_bot):
        """B. exit_method = 'MARKET_FALLBACK' on timeout."""
        mock_bot.config['strategy']['f_v3_limit_close'] = {'enabled': True}
        _setup_position(mock_bot)
        mock_bot._exchange_close_limit = MagicMock(return_value=(70300.0, 1234567, 'MARKET_FALLBACK'))

        mock_bot._do_close(0, {'reason': 'TRAIL_TP', 'exit_price': 70200.0})

        assert mock_bot.trade_history[0]['exit_method'] == 'MARKET_FALLBACK'
