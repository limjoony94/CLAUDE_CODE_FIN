"""Tests for TP/SL calculation, double exit (scale-out) logic,
and EXCHANGE_MANAGED sentinel behavior (v1.28.22)."""

import pytest
import ccxt
from unittest.mock import MagicMock, patch

from bingx_rl_trading_bot.scripts.production.pattern_5m.constants import (
    PATTERN_OPTIMAL_TPSL,
    SLIPPAGE_BUFFER_PCT,
    PRICE_ROUND_DECIMALS,
    DEFAULT_TP_PCT,
    DEFAULT_SL_PCT,
    TP1_RATIO,
    TP1_QTY_PCT,
)
from bingx_rl_trading_bot.scripts.production.pattern_5m.orders import (
    _EXCHANGE_MANAGED,
    _place_single_tp_order,
    _place_sl_order,
    _place_scale_out_orders,
    _verify_single_tp_order,
    _verify_sl_order,
    _verify_scale_out_orders,
    _cancel_existing_tpsl_orders,
    adjust_tpsl_to_config,
    cancel_remaining_orders,
    place_tp_sl_orders,
    verify_tp_sl_orders,
)


def _calculate_tp_sl_simple(
    entry_price: float,
    direction: int,  # +1 LONG, -1 SHORT
    tp_pct: float,
    sl_pct: float,
    vol_mult: float = 1.0,
):
    """Simplified TP/SL calc matching position_open._calculate_tp_sl."""
    tp_pct_adj = (tp_pct * vol_mult) + SLIPPAGE_BUFFER_PCT
    sl_pct_adj = (sl_pct * vol_mult) - SLIPPAGE_BUFFER_PCT
    tp_price = round(entry_price * (1 + direction * tp_pct_adj / 100), PRICE_ROUND_DECIMALS)
    sl_price = round(entry_price * (1 - direction * sl_pct_adj / 100), PRICE_ROUND_DECIMALS)
    return tp_price, sl_price, tp_pct_adj, sl_pct_adj


class TestTPSLCalculation:
    """Test TP/SL price calculation for each pattern."""

    @pytest.mark.parametrize("pattern", list(PATTERN_OPTIMAL_TPSL.keys()))
    def test_per_pattern_tpsl_long(self, pattern):
        """LONG TP should be above entry, SL below entry."""
        tp_pct, sl_pct = PATTERN_OPTIMAL_TPSL[pattern]
        entry = 100000.0
        tp, sl, _, _ = _calculate_tp_sl_simple(entry, +1, tp_pct, sl_pct)
        assert tp > entry, f"{pattern} LONG TP ({tp}) not above entry ({entry})"
        assert sl < entry, f"{pattern} LONG SL ({sl}) not below entry ({entry})"

    @pytest.mark.parametrize("pattern", list(PATTERN_OPTIMAL_TPSL.keys()))
    def test_per_pattern_tpsl_short(self, pattern):
        """SHORT TP should be below entry, SL above entry."""
        tp_pct, sl_pct = PATTERN_OPTIMAL_TPSL[pattern]
        entry = 100000.0
        tp, sl, _, _ = _calculate_tp_sl_simple(entry, -1, tp_pct, sl_pct)
        assert tp < entry, f"{pattern} SHORT TP ({tp}) not below entry ({entry})"
        assert sl > entry, f"{pattern} SHORT SL ({sl}) not above entry ({entry})"

    def test_slippage_buffer_applied(self):
        """TP gets slippage added (further), SL gets slippage subtracted (tighter)."""
        entry = 100000.0
        tp_pct, sl_pct = 1.0, 1.0
        _, _, tp_adj, sl_adj = _calculate_tp_sl_simple(entry, +1, tp_pct, sl_pct)
        assert tp_adj == tp_pct + SLIPPAGE_BUFFER_PCT
        assert sl_adj == sl_pct - SLIPPAGE_BUFFER_PCT

    def test_vol_mult_scales(self):
        """Volatility multiplier should scale TP/SL proportionally."""
        entry = 100000.0
        tp_pct, sl_pct = 1.0, 1.0
        _, _, tp1, sl1 = _calculate_tp_sl_simple(entry, +1, tp_pct, sl_pct, vol_mult=1.0)
        _, _, tp2, sl2 = _calculate_tp_sl_simple(entry, +1, tp_pct, sl_pct, vol_mult=1.5)
        assert tp2 > tp1
        # sl_adj = (sl_pct * vol_mult) - slippage → higher vol_mult → larger sl_adj → tighter stop
        assert sl2 > sl1


class TestDoubleExit:
    """Test scale-out (double exit) calculation."""

    def test_tp1_ratio(self):
        """TP1 should be at TP1_RATIO of full TP distance."""
        entry = 100000.0
        full_tp_pct = 1.5
        tp1_pct = full_tp_pct * TP1_RATIO
        tp1_price = entry * (1 + tp1_pct / 100)
        full_tp_price = entry * (1 + full_tp_pct / 100)
        assert tp1_price < full_tp_price
        assert tp1_price > entry

    def test_tp1_quantity_split(self):
        """TP1 should close TP1_QTY_PCT of total quantity."""
        total_qty = 0.01
        tp1_qty = total_qty * TP1_QTY_PCT / 100
        tp2_qty = total_qty - tp1_qty
        assert tp1_qty + tp2_qty == pytest.approx(total_qty)
        assert tp1_qty == pytest.approx(total_qty * 0.5)  # 50%

    def test_default_tpsl_fallback(self):
        """When pattern not in PATTERN_OPTIMAL_TPSL, defaults should be used."""
        entry = 100000.0
        tp, sl, _, _ = _calculate_tp_sl_simple(entry, +1, DEFAULT_TP_PCT, DEFAULT_SL_PCT)
        assert tp > entry
        assert sl < entry


# ── EXCHANGE_MANAGED Sentinel (v1.28.22) ─────────────────────


class TestExchangeManagedSentinel:
    """Test _EXCHANGE_MANAGED sentinel for TP/SL order management.

    v1.28.22 fix: crash recovery → tp/sl_order_id=None → verify retries
    → exchange "already exists" → infinite loop.  Sentinel breaks the loop.
    """

    @pytest.fixture
    def long_position(self):
        return {
            'direction': 'LONG',
            'quantity': 0.01,
            'remaining_quantity': 0.01,
            'tp_price': 51000.0,
            'sl_price': 49000.0,
            'tp_order_id': None,
            'sl_order_id': None,
        }

    # ── _place_single_tp_order sentinel ──────────────────────

    def test_tp_110407_sets_exchange_managed(self, long_position):
        """Error 110407 (TP already exists) → marks tp_order_id as EXCHANGE_MANAGED."""
        exchange = MagicMock()
        exchange.create_order.side_effect = ccxt.ExchangeError(
            'bingx {"code":110407,"msg":"TP order already exists"}'
        )
        _place_single_tp_order(
            exchange, long_position, 'BTC/USDT:USDT', 'sell', 0.01, 51000.0
        )
        assert long_position['tp_order_id'] == _EXCHANGE_MANAGED

    def test_tp_110413_sets_exchange_managed(self, long_position):
        """Error 110413 (TP exceeded) → marks tp_order_id as EXCHANGE_MANAGED."""
        exchange = MagicMock()
        exchange.create_order.side_effect = ccxt.ExchangeError(
            'bingx {"code":110413,"msg":"TP price already exceeded"}'
        )
        _place_single_tp_order(
            exchange, long_position, 'BTC/USDT:USDT', 'sell', 0.01, 51000.0
        )
        assert long_position['tp_order_id'] == _EXCHANGE_MANAGED

    def test_tp_other_error_no_sentinel(self, long_position):
        """Other ExchangeError → tp_order_id stays None."""
        exchange = MagicMock()
        exchange.create_order.side_effect = ccxt.ExchangeError('unknown error')
        _place_single_tp_order(
            exchange, long_position, 'BTC/USDT:USDT', 'sell', 0.01, 51000.0
        )
        assert long_position['tp_order_id'] is None

    def test_tp_success_sets_order_id(self, long_position):
        """Successful TP placement → tp_order_id set to real ID."""
        exchange = MagicMock()
        exchange.create_order.return_value = {'id': 'tp_123'}
        _place_single_tp_order(
            exchange, long_position, 'BTC/USDT:USDT', 'sell', 0.01, 51000.0
        )
        assert long_position['tp_order_id'] == 'tp_123'

    # ── _place_sl_order sentinel ─────────────────────────────

    def test_sl_110406_sets_exchange_managed(self, long_position):
        """Error 110406 (SL already exists) → marks sl_order_id as EXCHANGE_MANAGED."""
        exchange = MagicMock()
        exchange.create_order.side_effect = ccxt.ExchangeError(
            'bingx {"code":110406,"msg":"SL order already exists"}'
        )
        _place_sl_order(
            exchange, long_position, 'BTC/USDT:USDT', 'sell', 0.01, 49000.0
        )
        assert long_position['sl_order_id'] == _EXCHANGE_MANAGED

    def test_sl_other_error_no_sentinel(self, long_position):
        """Other ExchangeError → sl_order_id stays None."""
        exchange = MagicMock()
        exchange.create_order.side_effect = ccxt.ExchangeError('unknown error')
        _place_sl_order(
            exchange, long_position, 'BTC/USDT:USDT', 'sell', 0.01, 49000.0
        )
        assert long_position['sl_order_id'] is None

    def test_sl_success_sets_order_id(self, long_position):
        """Successful SL placement → sl_order_id set to real ID."""
        exchange = MagicMock()
        exchange.create_order.return_value = {'id': 'sl_456'}
        _place_sl_order(
            exchange, long_position, 'BTC/USDT:USDT', 'sell', 0.01, 49000.0
        )
        assert long_position['sl_order_id'] == 'sl_456'

    # ── _verify_single_tp_order sentinel ─────────────────────

    def test_verify_tp_skips_exchange_managed(self, long_position):
        """EXCHANGE_MANAGED → verify does nothing (no API call)."""
        long_position['tp_order_id'] = _EXCHANGE_MANAGED
        exchange = MagicMock()
        changed = _verify_single_tp_order(
            exchange, long_position, 'BTC/USDT:USDT', {}
        )
        assert changed is False
        exchange.create_order.assert_not_called()

    def test_verify_tp_missing_replaces(self, long_position):
        """tp_order_id=None → places new TP order."""
        exchange = MagicMock()
        exchange.create_order.return_value = {'id': 'tp_new'}
        changed = _verify_single_tp_order(
            exchange, long_position, 'BTC/USDT:USDT', {}
        )
        assert changed is True
        assert long_position['tp_order_id'] == 'tp_new'

    def test_verify_tp_110407_marks_managed(self, long_position):
        """Verify attempt → 110407 → marks as EXCHANGE_MANAGED."""
        exchange = MagicMock()
        exchange.create_order.side_effect = ccxt.ExchangeError(
            'bingx {"code":110407,"msg":"TP already exists"}'
        )
        changed = _verify_single_tp_order(
            exchange, long_position, 'BTC/USDT:USDT', {}
        )
        assert changed is True
        assert long_position['tp_order_id'] == _EXCHANGE_MANAGED

    def test_verify_tp_invalid_price_skips(self, long_position):
        """tp_price=0 → returns False without attempting."""
        long_position['tp_price'] = 0
        exchange = MagicMock()
        changed = _verify_single_tp_order(
            exchange, long_position, 'BTC/USDT:USDT', {}
        )
        assert changed is False
        exchange.create_order.assert_not_called()

    # ── _verify_sl_order sentinel ────────────────────────────

    def test_verify_sl_skips_exchange_managed(self, long_position):
        """EXCHANGE_MANAGED → verify does nothing."""
        long_position['sl_order_id'] = _EXCHANGE_MANAGED
        exchange = MagicMock()
        changed = _verify_sl_order(
            exchange, long_position, 'BTC/USDT:USDT', {}
        )
        assert changed is False
        exchange.create_order.assert_not_called()

    def test_verify_sl_missing_replaces(self, long_position):
        """sl_order_id=None → places new SL order."""
        exchange = MagicMock()
        exchange.create_order.return_value = {'id': 'sl_new'}
        changed = _verify_sl_order(
            exchange, long_position, 'BTC/USDT:USDT', {}
        )
        assert changed is True
        assert long_position['sl_order_id'] == 'sl_new'

    def test_verify_sl_110406_marks_managed(self, long_position):
        """Verify attempt → 110406 → marks as EXCHANGE_MANAGED."""
        exchange = MagicMock()
        exchange.create_order.side_effect = ccxt.ExchangeError(
            'bingx {"code":110406,"msg":"SL already exists"}'
        )
        changed = _verify_sl_order(
            exchange, long_position, 'BTC/USDT:USDT', {}
        )
        assert changed is True
        assert long_position['sl_order_id'] == _EXCHANGE_MANAGED

    def test_verify_sl_invalid_price_skips(self, long_position):
        """sl_price=0 → returns False without attempting."""
        long_position['sl_price'] = 0
        exchange = MagicMock()
        changed = _verify_sl_order(
            exchange, long_position, 'BTC/USDT:USDT', {}
        )
        assert changed is False
        exchange.create_order.assert_not_called()

    # ── _cancel_existing_tpsl_orders sentinel ────────────────

    def test_cancel_skips_exchange_managed_tp(self, long_position):
        """EXCHANGE_MANAGED tp_order_id → not cancelled."""
        long_position['tp_order_id'] = _EXCHANGE_MANAGED
        long_position['sl_order_id'] = 'sl_real'
        exchange = MagicMock()
        exchange.fetch_open_orders.return_value = [
            {'id': 'sl_real'},
        ]
        _cancel_existing_tpsl_orders(exchange, long_position, 'BTC/USDT:USDT')
        # Should only cancel SL, not TP
        exchange.cancel_order.assert_called_once_with('sl_real', 'BTC/USDT:USDT')

    def test_cancel_skips_exchange_managed_sl(self, long_position):
        """EXCHANGE_MANAGED sl_order_id → not cancelled."""
        long_position['tp_order_id'] = 'tp_real'
        long_position['sl_order_id'] = _EXCHANGE_MANAGED
        exchange = MagicMock()
        exchange.fetch_open_orders.return_value = [
            {'id': 'tp_real'},
        ]
        _cancel_existing_tpsl_orders(exchange, long_position, 'BTC/USDT:USDT')
        exchange.cancel_order.assert_called_once_with('tp_real', 'BTC/USDT:USDT')


# ── place_tp_sl_orders (orchestrator) ──────────────────────


@patch('bingx_rl_trading_bot.scripts.production.pattern_5m.orders.save_state')
class TestPlaceTpSlOrders:
    """Test place_tp_sl_orders() orchestration logic."""

    def _make_state(self, scale_out=False):
        position = {
            'slot_id': 's1',
            'direction': 'LONG',
            'quantity': 0.01,
            'remaining_quantity': 0.01,
            'tp_price': 51000.0,
            'sl_price': 49000.0,
            'tp_order_id': None,
            'sl_order_id': None,
            'scale_out_enabled': scale_out,
            'scale_out_stages': [
                {'stage': 1, 'quantity': 0.005, 'tp_price': 50500.0, 'filled': False, 'order_id': None},
            ] if scale_out else [],
        }
        return {'positions': {'s1': position}, 'active_direction': 'LONG'}

    def test_no_position_returns_early(self, mock_save):
        """No position → no orders placed."""
        exchange = MagicMock()
        place_tp_sl_orders(exchange, {'positions': {}}, {'symbol': 'BTC/USDT:USDT'})
        exchange.create_order.assert_not_called()
        mock_save.assert_not_called()

    def test_single_tp_sl_placed(self, mock_save):
        """Normal path → TP + SL orders placed + state saved."""
        exchange = MagicMock()
        exchange.create_order.return_value = {'id': 'order_123'}
        state = self._make_state()
        config = {'symbol': 'BTC/USDT:USDT'}

        place_tp_sl_orders(exchange, state, config)

        # 2 calls: one TP, one SL
        assert exchange.create_order.call_count == 2
        mock_save.assert_called_once()

    def test_scale_out_path(self, mock_save):
        """Scale-out enabled → stage TP orders + SL."""
        exchange = MagicMock()
        exchange.create_order.return_value = {'id': 'order_123'}
        state = self._make_state(scale_out=True)
        config = {'symbol': 'BTC/USDT:USDT'}

        place_tp_sl_orders(exchange, state, config)

        # 1 scale-out TP + 1 SL = 2 calls
        assert exchange.create_order.call_count == 2
        mock_save.assert_called_once()

    def test_network_error_handled(self, mock_save):
        """NetworkError → no crash, no save."""
        exchange = MagicMock()
        exchange.create_order.side_effect = ccxt.NetworkError('timeout')
        state = self._make_state()
        config = {'symbol': 'BTC/USDT:USDT'}

        place_tp_sl_orders(exchange, state, config)  # should not raise


# ── verify_tp_sl_orders ────────────────────────────────────


@patch('bingx_rl_trading_bot.scripts.production.pattern_5m.orders.save_state')
class TestVerifyTpSlOrders:
    """Test verify_tp_sl_orders() periodic health check."""

    def _make_state(self, tp_id='tp_1', sl_id='sl_1'):
        pos = {
            'slot_id': 's1',
            'direction': 'LONG',
            'quantity': 0.01,
            'remaining_quantity': 0.01,
            'tp_price': 51000.0,
            'sl_price': 49000.0,
            'tp_order_id': tp_id,
            'sl_order_id': sl_id,
            'scale_out_enabled': False,
            'scale_out_stages': [],
        }
        return {'positions': {'s1': pos}, 'active_direction': 'LONG'}

    def test_no_position_returns_early(self, mock_save):
        """No position → nothing to verify."""
        exchange = MagicMock()
        verify_tp_sl_orders(exchange, {'positions': {}}, {'symbol': 'BTC/USDT:USDT'})
        exchange.fetch_open_orders.assert_not_called()

    def test_orders_present_no_change(self, mock_save):
        """Both orders on exchange → no re-placement, no save."""
        exchange = MagicMock()
        exchange.fetch_open_orders.return_value = [
            {'id': 'tp_1'}, {'id': 'sl_1'},
        ]
        state = self._make_state()
        verify_tp_sl_orders(exchange, state, {'symbol': 'BTC/USDT:USDT'})
        # No create_order calls since both exist
        exchange.create_order.assert_not_called()
        mock_save.assert_not_called()

    def test_missing_tp_replaces_and_saves(self, mock_save):
        """TP missing from exchange → re-places, saves."""
        exchange = MagicMock()
        exchange.fetch_open_orders.return_value = [
            {'id': 'sl_1'},  # Only SL present
        ]
        exchange.fetch_positions.return_value = [
            {'contracts': 0.01, 'side': 'long'},
        ]
        exchange.create_order.return_value = {'id': 'tp_new'}
        state = self._make_state()
        verify_tp_sl_orders(exchange, state, {'symbol': 'BTC/USDT:USDT'})
        exchange.create_order.assert_called_once()
        mock_save.assert_called_once()
        assert state['positions']['s1']['tp_order_id'] == 'tp_new'

    def test_missing_sl_replaces_and_saves(self, mock_save):
        """SL missing from exchange → re-places, saves."""
        exchange = MagicMock()
        exchange.fetch_open_orders.return_value = [
            {'id': 'tp_1'},  # Only TP present
        ]
        exchange.fetch_positions.return_value = [
            {'contracts': 0.01, 'side': 'long'},
        ]
        exchange.create_order.return_value = {'id': 'sl_new'}
        state = self._make_state()
        verify_tp_sl_orders(exchange, state, {'symbol': 'BTC/USDT:USDT'})
        exchange.create_order.assert_called_once()
        mock_save.assert_called_once()
        assert state['positions']['s1']['sl_order_id'] == 'sl_new'

    def test_network_error_handled(self, mock_save):
        """Network error fetching orders → no crash."""
        exchange = MagicMock()
        exchange.fetch_open_orders.side_effect = ccxt.NetworkError('timeout')
        state = self._make_state()
        verify_tp_sl_orders(exchange, state, {'symbol': 'BTC/USDT:USDT'})
        mock_save.assert_not_called()


# ── adjust_tpsl_to_config ──────────────────────────────────


@patch('bingx_rl_trading_bot.scripts.production.pattern_5m.orders.save_state')
class TestAdjustTpslToConfig:
    """Test adjust_tpsl_to_config() startup reconciliation."""

    def test_no_position_returns_false(self, mock_save):
        """No position → returns False."""
        result = adjust_tpsl_to_config(MagicMock(), {'positions': {}}, {})
        assert result is False

    def test_dynamic_universal_no_pattern_needed(self, mock_save):
        """Dynamic universal mode → no pattern_name needed, skips if prices match."""
        entry = 50000.0
        tp_pct, sl_pct = 2.0, 3.0
        state = {'positions': {'s1': {'slot_id': 's1', 'direction': 'SHORT',
            'entry_price': entry,
            'tp_price': round(entry * (1 - tp_pct / 100), 1),
            'sl_price': round(entry * (1 + sl_pct / 100), 1),
        }}, 'active_direction': 'SHORT'}
        config = {'_dynamic_tpsl_universal': True, '_dynamic_tp': tp_pct, '_dynamic_sl': sl_pct,
                  'symbol': 'BTC/USDT:USDT'}
        result = adjust_tpsl_to_config(MagicMock(), state, config)
        assert result is False

    def test_dynamic_per_pattern_no_pattern_skips(self, mock_save):
        """Dynamic per-pattern mode, no pattern_name → skip."""
        state = {'positions': {'s1': {'slot_id': 's1', 'direction': 'LONG',
            'entry_price': 50000.0, 'tp_price': 0, 'sl_price': 0,
            'pattern_name': None, 'reason': 'Recovered from exchange',
        }}, 'active_direction': 'LONG'}
        config = {'_dynamic_tpsl_per_pattern': True, '_dynamic_patterns_tpsl': {},
                  'symbol': 'BTC/USDT:USDT'}
        result = adjust_tpsl_to_config(MagicMock(), state, config)
        assert result is False

    def test_no_pattern_in_reason_skips(self, mock_save):
        """No extractable pattern → skip."""
        state = {'positions': {'s1': {'slot_id': 's1', 'reason': 'manual entry', 'direction': 'LONG'}}, 'active_direction': 'LONG'}
        result = adjust_tpsl_to_config(MagicMock(), state, {'symbol': 'BTC/USDT:USDT'})
        assert result is False

    def test_prices_within_tolerance_skips(self, mock_save):
        """TP/SL within $1 tolerance → no adjustment."""
        if not PATTERN_OPTIMAL_TPSL:
            pytest.skip("No patterns configured")
        pat = next(iter(PATTERN_OPTIMAL_TPSL))
        tp_pct, sl_pct = PATTERN_OPTIMAL_TPSL[pat]
        entry = 100000.0
        state = {
            'positions': {'s1': {'slot_id': 's1',
                'direction': 'LONG',
                'entry_price': entry,
                'reason': f'Pattern: {pat} (LONG)',
                'tp_price': round(entry * (1 + tp_pct / 100), 1),
                'sl_price': round(entry * (1 - sl_pct / 100), 1),
            }},
            'active_direction': 'LONG',
        }
        result = adjust_tpsl_to_config(MagicMock(), state, {'symbol': 'BTC/USDT:USDT'})
        assert result is False
        mock_save.assert_not_called()

    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.orders._place_sl_order')
    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.orders._place_single_tp_order')
    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.orders._cancel_existing_tpsl_orders')
    def test_adjustment_done_when_prices_differ(
        self, mock_cancel, mock_tp, mock_sl, mock_save
    ):
        """TP/SL differ > $1 → cancels old, places new."""
        if not PATTERN_OPTIMAL_TPSL:
            pytest.skip("No patterns configured")
        pat = next(iter(PATTERN_OPTIMAL_TPSL))
        state = {
            'positions': {'s1': {'slot_id': 's1',
                'direction': 'LONG',
                'entry_price': 100000.0,
                'quantity': 0.01,
                'remaining_quantity': 0.01,
                'reason': f'Pattern: {pat} (LONG)',
                'tp_price': 999999.0,  # Way off
                'sl_price': 1.0,       # Way off
                'tp_order_id': 'old_tp',
                'sl_order_id': 'old_sl',
            }},
            'active_direction': 'LONG',
        }
        exchange = MagicMock()
        result = adjust_tpsl_to_config(exchange, state, {'symbol': 'BTC/USDT:USDT'})
        assert result is True
        mock_cancel.assert_called_once()
        mock_tp.assert_called_once()
        mock_sl.assert_called_once()
        mock_save.assert_called()


# ── cancel_remaining_orders ────────────────────────────────


class TestCancelRemainingOrders:
    """Test cancel_remaining_orders() full cancellation."""

    def test_no_position_returns_early(self):
        """No position → no API calls."""
        exchange = MagicMock()
        cancel_remaining_orders(exchange, {}, {'symbol': 'BTC/USDT:USDT'})
        exchange.fetch_open_orders.assert_not_called()

    def test_no_order_ids_returns_early(self):
        """Position but no order IDs → no API calls."""
        exchange = MagicMock()
        state = {'positions': {'s1': {'slot_id': 's1', 'tp_order_id': None, 'sl_order_id': None}}, 'active_direction': 'LONG'}
        cancel_remaining_orders(exchange, state, {'symbol': 'BTC/USDT:USDT'})
        exchange.fetch_open_orders.assert_not_called()

    def test_tp_and_sl_cancelled(self):
        """Both TP/SL on exchange → both cancelled."""
        exchange = MagicMock()
        exchange.fetch_open_orders.return_value = [
            {'id': 'tp_1'}, {'id': 'sl_1'},
        ]
        state = {
            'positions': {'s1': {'slot_id': 's1',
                'tp_order_id': 'tp_1',
                'sl_order_id': 'sl_1',
            }},
            'active_direction': 'LONG',
        }
        cancel_remaining_orders(exchange, state, {'symbol': 'BTC/USDT:USDT'})
        assert exchange.cancel_order.call_count == 2

    def test_sentinel_tp_skipped(self):
        """EXCHANGE_MANAGED TP → not cancelled."""
        exchange = MagicMock()
        exchange.fetch_open_orders.return_value = [{'id': 'sl_1'}]
        state = {
            'positions': {'s1': {'slot_id': 's1',
                'tp_order_id': _EXCHANGE_MANAGED,
                'sl_order_id': 'sl_1',
            }},
            'active_direction': 'LONG',
        }
        cancel_remaining_orders(exchange, state, {'symbol': 'BTC/USDT:USDT'})
        exchange.cancel_order.assert_called_once_with('sl_1', 'BTC/USDT:USDT')

    def test_sentinel_sl_skipped(self):
        """EXCHANGE_MANAGED SL → not cancelled."""
        exchange = MagicMock()
        exchange.fetch_open_orders.return_value = [{'id': 'tp_1'}]
        state = {
            'positions': {'s1': {'slot_id': 's1',
                'tp_order_id': 'tp_1',
                'sl_order_id': _EXCHANGE_MANAGED,
            }},
            'active_direction': 'LONG',
        }
        cancel_remaining_orders(exchange, state, {'symbol': 'BTC/USDT:USDT'})
        exchange.cancel_order.assert_called_once_with('tp_1', 'BTC/USDT:USDT')

    def test_scale_out_orders_cancelled(self):
        """Scale-out stage orders → cancelled."""
        exchange = MagicMock()
        exchange.fetch_open_orders.return_value = [
            {'id': 'so_1'}, {'id': 'so_2'},
        ]
        state = {
            'positions': {'s1': {'slot_id': 's1',
                'tp_order_id': None,
                'sl_order_id': None,
                'scale_out_stages': [
                    {'order_id': 'so_1', 'filled': False},
                    {'order_id': 'so_2', 'filled': False},
                ],
            }},
            'active_direction': 'LONG',
        }
        cancel_remaining_orders(exchange, state, {'symbol': 'BTC/USDT:USDT'})
        assert exchange.cancel_order.call_count == 2

    def test_filled_scale_out_not_cancelled(self):
        """Already filled scale-out stage → not cancelled."""
        exchange = MagicMock()
        exchange.fetch_open_orders.return_value = [{'id': 'so_2'}]
        state = {
            'positions': {'s1': {'slot_id': 's1',
                'tp_order_id': None,
                'sl_order_id': None,
                'scale_out_stages': [
                    {'order_id': 'so_1', 'filled': True},
                    {'order_id': 'so_2', 'filled': False},
                ],
            }},
            'active_direction': 'LONG',
        }
        cancel_remaining_orders(exchange, state, {'symbol': 'BTC/USDT:USDT'})
        exchange.cancel_order.assert_called_once_with('so_2', 'BTC/USDT:USDT')

    def test_network_error_handled(self):
        """Network error → no crash."""
        exchange = MagicMock()
        exchange.fetch_open_orders.side_effect = ccxt.NetworkError('timeout')
        state = {
            'positions': {'s1': {'slot_id': 's1', 'tp_order_id': 'tp_1', 'sl_order_id': 'sl_1'}}, 'active_direction': 'LONG',
        }
        cancel_remaining_orders(exchange, state, {'symbol': 'BTC/USDT:USDT'})  # no raise


# ── _place_scale_out_orders Tests ────────────────────────────


class TestPlaceScaleOutOrders:
    """Test _place_scale_out_orders() staged TP order placement."""

    def test_single_stage_success(self):
        """Single stage → places one TP order."""
        exchange = MagicMock()
        exchange.create_order.return_value = {'id': 'so_tp_1'}
        position = {'direction': 'LONG'}
        stages = [{'stage': 1, 'quantity': 0.005, 'tp_price': 51000.0}]
        _place_scale_out_orders(exchange, position, 'BTC/USDT:USDT', 'sell', stages)
        exchange.create_order.assert_called_once()
        assert stages[0]['order_id'] == 'so_tp_1'
        call_params = exchange.create_order.call_args[1]['params']
        assert call_params.get('positionSide') == 'BOTH'

    def test_two_stages_success(self):
        """Two stages → both placed with amount-based orders."""
        exchange = MagicMock()
        exchange.create_order.side_effect = [
            {'id': 'so_tp_1'},
            {'id': 'so_tp_2'},
        ]
        position = {'direction': 'LONG'}
        stages = [
            {'stage': 1, 'quantity': 0.003, 'tp_price': 51000.0},
            {'stage': 2, 'quantity': 0.007, 'tp_price': 52000.0},
        ]
        _place_scale_out_orders(exchange, position, 'BTC/USDT:USDT', 'sell', stages)
        assert exchange.create_order.call_count == 2
        assert stages[0]['order_id'] == 'so_tp_1'
        assert stages[1]['order_id'] == 'so_tp_2'

    def test_insufficient_funds_continues(self):
        """InsufficientFunds on first stage → continues to second."""
        exchange = MagicMock()
        exchange.create_order.side_effect = [
            ccxt.InsufficientFunds('no funds'),
            {'id': 'so_tp_2'},
        ]
        position = {'direction': 'SHORT'}
        stages = [
            {'stage': 1, 'quantity': 0.003, 'tp_price': 49000.0},
            {'stage': 2, 'quantity': 0.007, 'tp_price': 48000.0},
        ]
        _place_scale_out_orders(exchange, position, 'BTC/USDT:USDT', 'buy', stages)
        assert stages[1]['order_id'] == 'so_tp_2'
        assert 'order_id' not in stages[0]

    def test_exchange_error_continues(self):
        """ExchangeError on a stage → continues to next."""
        exchange = MagicMock()
        exchange.create_order.side_effect = ccxt.ExchangeError('invalid')
        position = {'direction': 'LONG'}
        stages = [{'stage': 1, 'quantity': 0.01, 'tp_price': 51000.0}]
        _place_scale_out_orders(exchange, position, 'BTC/USDT:USDT', 'sell', stages)
        assert 'order_id' not in stages[0]

    def test_generic_exception_continues(self):
        """Generic exception on a stage → continues to next."""
        exchange = MagicMock()
        exchange.create_order.side_effect = RuntimeError('weird')
        position = {'direction': 'LONG'}
        stages = [{'stage': 1, 'quantity': 0.01, 'tp_price': 51000.0}]
        _place_scale_out_orders(exchange, position, 'BTC/USDT:USDT', 'sell', stages)
        assert 'order_id' not in stages[0]

    def test_insufficient_funds_continues(self):
        """InsufficientFunds on a stage → continues."""
        exchange = MagicMock()
        exchange.create_order.side_effect = ccxt.InsufficientFunds('no funds')
        position = {'direction': 'LONG'}
        stages = [{'stage': 1, 'quantity': 0.01, 'tp_price': 51000.0}]
        _place_scale_out_orders(exchange, position, 'BTC/USDT:USDT', 'sell', stages)
        assert 'order_id' not in stages[0]


# ── _verify_scale_out_orders Tests ───────────────────────────


class TestVerifyScaleOutOrders:
    """Test _verify_scale_out_orders() staged TP order verification."""

    def test_all_orders_present(self):
        """All stage orders exist → no re-placement, returns False."""
        exchange = MagicMock()
        position = {'direction': 'LONG', 'quantity': 0.01}
        stages = [
            {'stage': 1, 'order_id': 'so_1', 'quantity': 0.005, 'tp_price': 51000.0},
            {'stage': 2, 'order_id': 'so_2', 'quantity': 0.005, 'tp_price': 52000.0},
        ]
        open_orders = {'so_1': {}, 'so_2': {}}
        result = _verify_scale_out_orders(exchange, position, 'BTC/USDT:USDT', stages, open_orders)
        assert result is False
        exchange.create_order.assert_not_called()

    def test_missing_order_re_placed(self):
        """Stage order missing from exchange → re-placed."""
        exchange = MagicMock()
        exchange.create_order.return_value = {'id': 'so_1_new'}
        position = {'direction': 'LONG', 'quantity': 0.01}
        stages = [
            {'stage': 1, 'order_id': 'so_1', 'quantity': 0.005, 'tp_price': 51000.0},
        ]
        open_orders = {}  # order missing
        result = _verify_scale_out_orders(exchange, position, 'BTC/USDT:USDT', stages, open_orders)
        assert result is True
        assert stages[0]['order_id'] == 'so_1_new'

    def test_never_placed_order_created(self):
        """Stage with order_id=None → creates new order."""
        exchange = MagicMock()
        exchange.create_order.return_value = {'id': 'so_new'}
        position = {'direction': 'SHORT', 'quantity': 0.01}
        stages = [
            {'stage': 1, 'order_id': None, 'quantity': 0.005, 'tp_price': 49000.0},
        ]
        result = _verify_scale_out_orders(exchange, position, 'BTC/USDT:USDT', stages, {})
        assert result is True
        assert stages[0]['order_id'] == 'so_new'

    def test_filled_stage_skipped(self):
        """Filled stage → not re-placed even if order_id missing."""
        exchange = MagicMock()
        position = {'direction': 'LONG', 'quantity': 0.01}
        stages = [
            {'stage': 1, 'order_id': None, 'filled': True, 'quantity': 0.005, 'tp_price': 51000.0},
        ]
        result = _verify_scale_out_orders(exchange, position, 'BTC/USDT:USDT', stages, {})
        assert result is False
        exchange.create_order.assert_not_called()

    def test_exchange_error_handled(self):
        """ExchangeError on re-placement → no crash, returns False for that stage."""
        exchange = MagicMock()
        exchange.create_order.side_effect = ccxt.ExchangeError('error')
        position = {'direction': 'LONG', 'quantity': 0.01}
        stages = [
            {'stage': 1, 'order_id': None, 'quantity': 0.005, 'tp_price': 51000.0},
        ]
        result = _verify_scale_out_orders(exchange, position, 'BTC/USDT:USDT', stages, {})
        assert result is False

    def test_last_stage_replaces_order(self):
        """Last stage with no order_id → re-places with amount-based order."""
        exchange = MagicMock()
        exchange.create_order.return_value = {'id': 'so_new'}
        position = {'direction': 'LONG', 'quantity': 0.01}
        stages = [
            {'stage': 1, 'order_id': None, 'quantity': 0.005, 'tp_price': 51000.0},
        ]
        _verify_scale_out_orders(exchange, position, 'BTC/USDT:USDT', stages, {})
        call_params = exchange.create_order.call_args[1]['params']
        assert call_params.get('positionSide') == 'BOTH'
        assert stages[0]['order_id'] == 'so_new'


# ── Error Path Tests for place_tp_sl_orders ──────────────────


class TestPlaceTpSlOrdersErrors:
    """Test error handling paths in place_tp_sl_orders."""

    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.orders.save_state')
    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.orders._place_sl_order')
    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.orders._place_single_tp_order')
    def test_network_error_no_crash(self, mock_tp, mock_sl, mock_save):
        """NetworkError during placement → no crash."""
        mock_tp.side_effect = ccxt.NetworkError('timeout')
        exchange = MagicMock()
        state = {'positions': {'s1': {'slot_id': 's1', 
            'direction': 'LONG', 'quantity': 0.01, 'tp_price': 51000.0,
            'sl_price': 49000.0,
        }}, 'active_direction': 'LONG'}
        config = {'symbol': 'BTC/USDT:USDT'}
        # place_tp_sl_orders catches NetworkError internally
        place_tp_sl_orders(exchange, state, config)

    def test_tp_110407_marks_exchange_managed(self):
        """TP order 110407 error → marks as EXCHANGE_MANAGED."""
        exchange = MagicMock()
        exchange.create_order.side_effect = ccxt.ExchangeError('110407 TP already exists')
        position = {'direction': 'LONG', 'quantity': 0.01}
        _place_single_tp_order(exchange, position, 'BTC/USDT:USDT', 'sell', 0.01, 51000.0)
        assert position['tp_order_id'] == _EXCHANGE_MANAGED

    def test_tp_110413_marks_exchange_managed(self):
        """TP order 110413 error → marks as EXCHANGE_MANAGED."""
        exchange = MagicMock()
        exchange.create_order.side_effect = ccxt.ExchangeError('110413 TP price exceeded')
        position = {'direction': 'LONG', 'quantity': 0.01}
        _place_single_tp_order(exchange, position, 'BTC/USDT:USDT', 'sell', 0.01, 51000.0)
        assert position['tp_order_id'] == _EXCHANGE_MANAGED

    def test_sl_110406_marks_exchange_managed(self):
        """SL order 110406 error → marks as EXCHANGE_MANAGED."""
        exchange = MagicMock()
        exchange.create_order.side_effect = ccxt.ExchangeError('110406 SL already exists')
        position = {'direction': 'LONG', 'quantity': 0.01}
        _place_sl_order(exchange, position, 'BTC/USDT:USDT', 'sell', 0.01, 49000.0)
        assert position['sl_order_id'] == _EXCHANGE_MANAGED

    def test_tp_insufficient_funds_no_crash(self):
        """InsufficientFunds on TP → no crash, no order_id."""
        exchange = MagicMock()
        exchange.create_order.side_effect = ccxt.InsufficientFunds('no funds')
        position = {'direction': 'LONG'}
        _place_single_tp_order(exchange, position, 'BTC/USDT:USDT', 'sell', 0.01, 51000.0)
        assert 'tp_order_id' not in position

    def test_sl_insufficient_funds_no_crash(self):
        """InsufficientFunds on SL → no crash, no order_id."""
        exchange = MagicMock()
        exchange.create_order.side_effect = ccxt.InsufficientFunds('no funds')
        position = {'direction': 'LONG'}
        _place_sl_order(exchange, position, 'BTC/USDT:USDT', 'sell', 0.01, 49000.0)
        assert 'sl_order_id' not in position


# ── cancel_remaining_orders Error Path Tests ─────────────────


class TestCancelRemainingOrdersErrors:
    """Test error handling paths in cancel_remaining_orders."""

    def test_order_not_found_handled(self):
        """OrderNotFound on cancel → continues silently."""
        exchange = MagicMock()
        exchange.fetch_open_orders.return_value = [{'id': 'tp_1'}]
        exchange.cancel_order.side_effect = ccxt.OrderNotFound('already gone')
        state = {
            'positions': {'s1': {'slot_id': 's1', 'tp_order_id': 'tp_1', 'sl_order_id': None}}, 'active_direction': 'LONG',
        }
        cancel_remaining_orders(exchange, state, {'symbol': 'BTC/USDT:USDT'})

    def test_exchange_error_on_cancel_handled(self):
        """ExchangeError on individual cancel → continues to next."""
        exchange = MagicMock()
        exchange.fetch_open_orders.return_value = [
            {'id': 'tp_1'}, {'id': 'sl_1'}
        ]
        exchange.cancel_order.side_effect = ccxt.ExchangeError('cannot cancel')
        state = {
            'positions': {'s1': {'slot_id': 's1', 'tp_order_id': 'tp_1', 'sl_order_id': 'sl_1'}}, 'active_direction': 'LONG',
        }
        cancel_remaining_orders(exchange, state, {'symbol': 'BTC/USDT:USDT'})
        assert exchange.cancel_order.call_count == 2

    def test_scale_out_order_not_found(self):
        """OrderNotFound on scale-out cancel → continues."""
        exchange = MagicMock()
        exchange.fetch_open_orders.return_value = [{'id': 'so_1'}]
        exchange.cancel_order.side_effect = ccxt.OrderNotFound('gone')
        state = {
            'positions': {'s1': {'slot_id': 's1',
                'tp_order_id': None, 'sl_order_id': None,
                'scale_out_stages': [{'order_id': 'so_1', 'filled': False}],
            }},
            'active_direction': 'LONG',
        }
        cancel_remaining_orders(exchange, state, {'symbol': 'BTC/USDT:USDT'})

    def test_exchange_error_on_fetch_handled(self):
        """ExchangeError on fetch_open_orders → no crash."""
        exchange = MagicMock()
        exchange.fetch_open_orders.side_effect = ccxt.ExchangeError('invalid')
        state = {
            'positions': {'s1': {'slot_id': 's1', 'tp_order_id': 'tp_1', 'sl_order_id': 'sl_1'}}, 'active_direction': 'LONG',
        }
        cancel_remaining_orders(exchange, state, {'symbol': 'BTC/USDT:USDT'})

    def test_generic_exception_on_fetch_handled(self):
        """Generic exception on fetch_open_orders → no crash."""
        exchange = MagicMock()
        exchange.fetch_open_orders.side_effect = RuntimeError('weird')
        state = {
            'positions': {'s1': {'slot_id': 's1', 'tp_order_id': 'tp_1', 'sl_order_id': 'sl_1'}}, 'active_direction': 'LONG',
        }
        cancel_remaining_orders(exchange, state, {'symbol': 'BTC/USDT:USDT'})

    def test_tp_generic_exception_on_cancel(self):
        """Generic exception on TP cancel → continues."""
        exchange = MagicMock()
        exchange.fetch_open_orders.return_value = [{'id': 'tp_1'}]
        exchange.cancel_order.side_effect = RuntimeError('weird')
        state = {
            'positions': {'s1': {'slot_id': 's1', 'tp_order_id': 'tp_1', 'sl_order_id': None}}, 'active_direction': 'LONG',
        }
        cancel_remaining_orders(exchange, state, {'symbol': 'BTC/USDT:USDT'})

    def test_sl_order_not_found(self):
        """SL OrderNotFound → silent."""
        exchange = MagicMock()
        exchange.fetch_open_orders.return_value = [{'id': 'sl_1'}]
        exchange.cancel_order.side_effect = ccxt.OrderNotFound('gone')
        state = {
            'positions': {'s1': {'slot_id': 's1', 'tp_order_id': None, 'sl_order_id': 'sl_1'}}, 'active_direction': 'LONG',
        }
        cancel_remaining_orders(exchange, state, {'symbol': 'BTC/USDT:USDT'})

    def test_sl_exchange_error_on_cancel(self):
        """SL ExchangeError → continues."""
        exchange = MagicMock()
        exchange.fetch_open_orders.return_value = [{'id': 'sl_1'}]
        exchange.cancel_order.side_effect = ccxt.ExchangeError('err')
        state = {
            'positions': {'s1': {'slot_id': 's1', 'tp_order_id': None, 'sl_order_id': 'sl_1'}}, 'active_direction': 'LONG',
        }
        cancel_remaining_orders(exchange, state, {'symbol': 'BTC/USDT:USDT'})

    def test_sl_generic_exception_on_cancel(self):
        """SL generic exception → continues."""
        exchange = MagicMock()
        exchange.fetch_open_orders.return_value = [{'id': 'sl_1'}]
        exchange.cancel_order.side_effect = RuntimeError('weird')
        state = {
            'positions': {'s1': {'slot_id': 's1', 'tp_order_id': None, 'sl_order_id': 'sl_1'}}, 'active_direction': 'LONG',
        }
        cancel_remaining_orders(exchange, state, {'symbol': 'BTC/USDT:USDT'})

    def test_scale_out_exchange_error_on_cancel(self):
        """Scale-out ExchangeError → continues."""
        exchange = MagicMock()
        exchange.fetch_open_orders.return_value = [{'id': 'so_1'}]
        exchange.cancel_order.side_effect = ccxt.ExchangeError('err')
        state = {
            'positions': {'s1': {'slot_id': 's1',
                'tp_order_id': None, 'sl_order_id': None,
                'scale_out_stages': [{'order_id': 'so_1', 'filled': False}],
            }},
            'active_direction': 'LONG',
        }
        cancel_remaining_orders(exchange, state, {'symbol': 'BTC/USDT:USDT'})

    def test_scale_out_generic_exception_on_cancel(self):
        """Scale-out generic exception → continues."""
        exchange = MagicMock()
        exchange.fetch_open_orders.return_value = [{'id': 'so_1'}]
        exchange.cancel_order.side_effect = RuntimeError('weird')
        state = {
            'positions': {'s1': {'slot_id': 's1',
                'tp_order_id': None, 'sl_order_id': None,
                'scale_out_stages': [{'order_id': 'so_1', 'filled': False}],
            }},
            'active_direction': 'LONG',
        }
        cancel_remaining_orders(exchange, state, {'symbol': 'BTC/USDT:USDT'})


# ── place_tp_sl_orders additional error paths ────────────────


class TestPlaceTpSlOrdersExchangeErrors:
    """Test ExchangeError and generic Exception in place_tp_sl_orders."""

    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.orders.save_state')
    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.orders._place_sl_order')
    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.orders._place_single_tp_order')
    def test_exchange_error_no_crash(self, mock_tp, mock_sl, mock_save):
        """ExchangeError → caught, no crash."""
        mock_tp.side_effect = ccxt.ExchangeError('order rejected')
        exchange = MagicMock()
        state = {'positions': {'s1': {'slot_id': 's1', 
            'direction': 'LONG', 'quantity': 0.01,
            'tp_price': 51000.0, 'sl_price': 49000.0,
        }}, 'active_direction': 'LONG'}
        place_tp_sl_orders(exchange, state, {'symbol': 'BTC/USDT:USDT'})

    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.orders.save_state')
    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.orders._place_sl_order')
    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.orders._place_single_tp_order')
    def test_generic_exception_no_crash(self, mock_tp, mock_sl, mock_save):
        """Generic exception → caught, no crash."""
        mock_tp.side_effect = RuntimeError('unexpected')
        exchange = MagicMock()
        state = {'positions': {'s1': {'slot_id': 's1', 
            'direction': 'LONG', 'quantity': 0.01,
            'tp_price': 51000.0, 'sl_price': 49000.0,
        }}, 'active_direction': 'LONG'}
        place_tp_sl_orders(exchange, state, {'symbol': 'BTC/USDT:USDT'})


# ── adjust_tpsl_to_config additional paths ────────────────────


class TestAdjustTpslToConfigExtended:
    """Test adjust_tpsl_to_config SHORT direction and error paths."""

    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.orders.save_state')
    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.orders._place_sl_order')
    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.orders._place_single_tp_order')
    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.orders._cancel_existing_tpsl_orders')
    def test_short_direction_calculates_correctly(self, mock_cancel, mock_tp, mock_sl, mock_save):
        """SHORT direction → TP below, SL above entry."""
        # Pick a known pattern
        pattern = list(PATTERN_OPTIMAL_TPSL.keys())[0]
        tp_pct, sl_pct = PATTERN_OPTIMAL_TPSL[pattern]
        exchange = MagicMock()
        exchange.fetch_open_orders.return_value = []
        state = {'positions': {'s1': {'slot_id': 's1',
            'direction': 'SHORT', 'entry_price': 50000.0,
            'quantity': 0.01, 'remaining_quantity': 0.01,
            'tp_price': 0, 'sl_price': 0,
            'reason': f'Pattern: {pattern} (SHORT)',
        }},
            'active_direction': 'SHORT',
        }
        config = {'symbol': 'BTC/USDT:USDT'}
        result = adjust_tpsl_to_config(exchange, state, config)
        assert result is True

    def test_pattern_not_in_optimal_returns_false(self):
        """Unknown pattern → returns False."""
        exchange = MagicMock()
        state = {'positions': {'s1': {'slot_id': 's1', 
            'direction': 'LONG', 'entry_price': 50000.0,
            'quantity': 0.01,
            'reason': 'Pattern: ZZ-ZZ-ZZ (LONG)',
        }}, 'active_direction': 'LONG'}
        config = {'symbol': 'BTC/USDT:USDT'}
        result = adjust_tpsl_to_config(exchange, state, config)
        assert result is False

    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.orders._cancel_existing_tpsl_orders')
    def test_generic_exception_returns_false(self, mock_cancel):
        """Generic exception during adjust → returns False."""
        mock_cancel.side_effect = RuntimeError('unexpected')
        pattern = list(PATTERN_OPTIMAL_TPSL.keys())[0]
        exchange = MagicMock()
        exchange.fetch_open_orders.return_value = []
        state = {'positions': {'s1': {'slot_id': 's1',
            'direction': 'LONG', 'entry_price': 50000.0,
            'quantity': 0.01, 'remaining_quantity': 0.01,
            'tp_price': 0, 'sl_price': 0,
            'reason': f'Pattern: {pattern} (LONG)',
        }},
            'active_direction': 'LONG',
        }
        config = {'symbol': 'BTC/USDT:USDT'}
        result = adjust_tpsl_to_config(exchange, state, config)
        assert result is False


# ── Dynamic mode TP/SL adjustment tests ───────────────────


class TestAdjustTpslDynamicMode:
    """Test adjust_tpsl_to_config with dynamic per-pattern and universal modes."""

    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.orders.save_state')
    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.orders._place_sl_order')
    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.orders._place_single_tp_order')
    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.orders._cancel_existing_tpsl_orders')
    def test_dynamic_per_pattern_adjusts_when_prices_differ(
        self, mock_cancel, mock_tp, mock_sl, mock_save
    ):
        """Dynamic per-pattern mode → adjusts TP/SL when prices are wrong."""
        entry = 67744.9
        state = {'positions': {'s1': {'slot_id': 's1',
            'direction': 'SHORT', 'entry_price': entry,
            'quantity': 0.0203, 'remaining_quantity': 0.0203,
            'tp_price': 67053.9,  # Wrong (too tight)
            'sl_price': 68408.8,  # Wrong (too tight)
            'pattern_name': 'ST-ST-U',
            'reason': 'Recovered from exchange',
            'tp_order_id': 'old_tp', 'sl_order_id': 'old_sl',
        }}, 'active_direction': 'SHORT'}
        config = {
            'symbol': 'BTC/USDT:USDT',
            '_dynamic_tpsl_per_pattern': True,
            '_dynamic_patterns_tpsl': {
                'ST-ST-U': [2.24, 3.94],
            },
        }
        result = adjust_tpsl_to_config(MagicMock(), state, config)
        assert result is True
        mock_cancel.assert_called_once()
        mock_tp.assert_called_once()
        mock_sl.assert_called_once()
        # Verify the new prices are based on per-pattern TP/SL + slippage buffer
        pos = state['positions']['s1']
        tp_pct_adj = 2.24 + SLIPPAGE_BUFFER_PCT  # TP: add slippage (target further)
        sl_pct_adj = 3.94 - SLIPPAGE_BUFFER_PCT  # SL: subtract slippage (tighter)
        expected_tp = round(entry * (1 - tp_pct_adj / 100), PRICE_ROUND_DECIMALS)
        expected_sl = round(entry * (1 + sl_pct_adj / 100), PRICE_ROUND_DECIMALS)
        assert pos['tp_price'] == expected_tp
        assert pos['sl_price'] == expected_sl

    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.orders.save_state')
    def test_dynamic_per_pattern_skips_when_prices_match(self, mock_save):
        """Dynamic per-pattern mode → skips if TP/SL already correct (with slippage)."""
        entry = 67744.9
        tp_pct, sl_pct = 2.24, 3.94
        # Prices must include slippage buffer to match calculate_tp_sl() output
        tp_pct_adj = tp_pct + SLIPPAGE_BUFFER_PCT
        sl_pct_adj = sl_pct - SLIPPAGE_BUFFER_PCT
        state = {'positions': {'s1': {'slot_id': 's1',
            'direction': 'SHORT', 'entry_price': entry,
            'tp_price': round(entry * (1 - tp_pct_adj / 100), PRICE_ROUND_DECIMALS),
            'sl_price': round(entry * (1 + sl_pct_adj / 100), PRICE_ROUND_DECIMALS),
            'pattern_name': 'ST-ST-U',
        }}, 'active_direction': 'SHORT'}
        config = {
            'symbol': 'BTC/USDT:USDT',
            '_dynamic_tpsl_per_pattern': True,
            '_dynamic_patterns_tpsl': {'ST-ST-U': [tp_pct, sl_pct]},
        }
        result = adjust_tpsl_to_config(MagicMock(), state, config)
        assert result is False
        mock_save.assert_not_called()

    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.orders.save_state')
    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.orders._place_sl_order')
    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.orders._place_single_tp_order')
    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.orders._cancel_existing_tpsl_orders')
    def test_dynamic_universal_adjusts_when_prices_differ(
        self, mock_cancel, mock_tp, mock_sl, mock_save
    ):
        """Dynamic universal mode → adjusts TP/SL when prices are wrong."""
        entry = 50000.0
        state = {'positions': {'s1': {'slot_id': 's1',
            'direction': 'LONG', 'entry_price': entry,
            'quantity': 0.01, 'remaining_quantity': 0.01,
            'tp_price': 50100.0,  # Wrong
            'sl_price': 49900.0,  # Wrong
        }}, 'active_direction': 'LONG'}
        config = {
            'symbol': 'BTC/USDT:USDT',
            '_dynamic_tpsl_universal': True,
            '_dynamic_tp': 2.0,
            '_dynamic_sl': 3.0,
        }
        result = adjust_tpsl_to_config(MagicMock(), state, config)
        assert result is True
        pos = state['positions']['s1']
        # calculate_tp_sl applies slippage buffer: TP + buffer, SL - buffer
        tp_pct_adj = 2.0 + SLIPPAGE_BUFFER_PCT
        sl_pct_adj = 3.0 - SLIPPAGE_BUFFER_PCT
        assert pos['tp_price'] == round(entry * (1 + tp_pct_adj / 100), PRICE_ROUND_DECIMALS)
        assert pos['sl_price'] == round(entry * (1 - sl_pct_adj / 100), PRICE_ROUND_DECIMALS)

    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.orders.save_state')
    def test_dynamic_per_pattern_unknown_pattern_skips(self, mock_save):
        """Dynamic per-pattern mode, pattern not in dict → skip."""
        state = {'positions': {'s1': {'slot_id': 's1',
            'direction': 'SHORT', 'entry_price': 50000.0,
            'tp_price': 0, 'sl_price': 0,
            'pattern_name': 'ZZ-ZZ-ZZ',
        }}, 'active_direction': 'SHORT'}
        config = {
            'symbol': 'BTC/USDT:USDT',
            '_dynamic_tpsl_per_pattern': True,
            '_dynamic_patterns_tpsl': {'AA-BB-CC': [2.0, 3.0]},
        }
        result = adjust_tpsl_to_config(MagicMock(), state, config)
        assert result is False

    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.orders.save_state')
    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.orders._place_sl_order')
    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.orders._place_single_tp_order')
    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.orders._cancel_existing_tpsl_orders')
    def test_dynamic_per_pattern_multiple_slots(
        self, mock_cancel, mock_tp, mock_sl, mock_save
    ):
        """Multiple slots with different patterns → each adjusted independently."""
        entry = 67744.9
        state = {'positions': {
            's1': {'slot_id': 's1', 'direction': 'SHORT', 'entry_price': entry,
                   'quantity': 0.02, 'remaining_quantity': 0.02,
                   'tp_price': 67053.9, 'sl_price': 68408.8,
                   'pattern_name': 'ST-ST-U',
                   'tp_order_id': 'tp1', 'sl_order_id': 'sl1'},
            's2': {'slot_id': 's2', 'direction': 'SHORT', 'entry_price': entry,
                   'quantity': 0.02, 'remaining_quantity': 0.02,
                   'tp_price': 67053.9, 'sl_price': 68408.8,
                   'pattern_name': 'H-DN-U',
                   'tp_order_id': 'tp2', 'sl_order_id': 'sl2'},
        }, 'active_direction': 'SHORT'}
        config = {
            'symbol': 'BTC/USDT:USDT',
            '_dynamic_tpsl_per_pattern': True,
            '_dynamic_patterns_tpsl': {
                'ST-ST-U': [2.24, 3.94],
                'H-DN-U': [3.0, 3.88],
            },
        }
        result = adjust_tpsl_to_config(MagicMock(), state, config)
        assert result is True
        assert mock_cancel.call_count == 2
        # Each slot should have different TP/SL
        s1 = state['positions']['s1']
        s2 = state['positions']['s2']
        assert s1['tp_price'] != s2['tp_price']
        assert s1['sl_price'] != s2['sl_price']

    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.orders._verify_emergency_sl')
    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.orders.save_state')
    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.orders._place_sl_order')
    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.orders._place_single_tp_order')
    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.orders._cancel_existing_tpsl_orders')
    def test_emergency_sl_verified_after_adjustment(
        self, mock_cancel, mock_tp, mock_sl, mock_save, mock_verify_emg
    ):
        """Emergency SL is verified after TP/SL adjustment."""
        entry = 67744.9
        exchange = MagicMock()
        state = {'positions': {'s1': {'slot_id': 's1',
            'direction': 'SHORT', 'entry_price': entry,
            'quantity': 0.02, 'remaining_quantity': 0.02,
            'tp_price': 67053.9, 'sl_price': 68408.8,
            'pattern_name': 'ST-ST-U',
            'tp_order_id': 'tp1', 'sl_order_id': 'sl1',
        }}, 'active_direction': 'SHORT', 'emergency_sl_order_id': 'old_emg'}
        config = {
            'symbol': 'BTC/USDT:USDT',
            '_dynamic_tpsl_per_pattern': True,
            '_dynamic_patterns_tpsl': {'ST-ST-U': [2.24, 3.94]},
        }
        result = adjust_tpsl_to_config(exchange, state, config)
        assert result is True
        mock_verify_emg.assert_called_once_with(exchange, state, config)

    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.orders._verify_emergency_sl')
    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.orders.save_state')
    def test_emergency_sl_verified_even_without_adjustment(self, mock_save, mock_verify_emg):
        """Emergency SL verified even when no TP/SL adjustment needed."""
        entry = 67744.9
        tp_pct, sl_pct = 2.24, 3.94
        state = {'positions': {'s1': {'slot_id': 's1',
            'direction': 'SHORT', 'entry_price': entry,
            'tp_price': round(entry * (1 - tp_pct / 100), 1),
            'sl_price': round(entry * (1 + sl_pct / 100), 1),
            'pattern_name': 'ST-ST-U',
        }}, 'active_direction': 'SHORT'}
        config = {
            'symbol': 'BTC/USDT:USDT',
            '_dynamic_tpsl_per_pattern': True,
            '_dynamic_patterns_tpsl': {'ST-ST-U': [tp_pct, sl_pct]},
        }
        result = adjust_tpsl_to_config(MagicMock(), state, config)
        assert result is False
        mock_verify_emg.assert_called_once()


# ── _cancel_existing_tpsl_orders edge cases ───────────────────


class TestCancelExistingTpslOrdersExtended:
    """Test _cancel_existing_tpsl_orders scale-out and error paths."""

    def test_scale_out_orders_cancelled(self):
        """Scale-out unfilled orders → cancelled."""
        exchange = MagicMock()
        exchange.fetch_open_orders.return_value = [{'id': 'so_1'}, {'id': 'so_2'}]
        position = {
            'tp_order_id': None, 'sl_order_id': None,
            'scale_out_stages': [
                {'order_id': 'so_1', 'filled': False},
                {'order_id': 'so_2', 'filled': True},  # Filled → not cancelled
            ],
        }
        _cancel_existing_tpsl_orders(exchange, position, 'BTC/USDT:USDT')
        exchange.cancel_order.assert_called_once_with('so_1', 'BTC/USDT:USDT')

    def test_exception_during_cancel(self):
        """Exception during cancel → caught, no crash."""
        exchange = MagicMock()
        exchange.fetch_open_orders.side_effect = Exception('fail')
        position = {'tp_order_id': 'tp_1', 'sl_order_id': None}
        _cancel_existing_tpsl_orders(exchange, position, 'BTC/USDT:USDT')


# ── _verify_single_tp_order exchange error codes ──────────────


class TestVerifyTpOrderExchangeErrorCodes:
    """Test _verify_single_tp_order with specific BingX error codes."""

    def test_110407_marks_exchange_managed(self):
        """110407 (TP exists) → marks tp_order_id as EXCHANGE_MANAGED."""
        exchange = MagicMock()
        exchange.create_order.side_effect = ccxt.ExchangeError('110407 TP already exists')
        position = {'direction': 'LONG', 'quantity': 0.01,
                     'tp_order_id': None, 'tp_price': 51000.0}
        result = _verify_single_tp_order(exchange, position, 'BTC/USDT:USDT', {})
        assert result is True
        assert position['tp_order_id'] == _EXCHANGE_MANAGED

    def test_110413_marks_exchange_managed(self):
        """110413 (TP exceeded) → marks tp_order_id as EXCHANGE_MANAGED."""
        exchange = MagicMock()
        exchange.create_order.side_effect = ccxt.ExchangeError('110413 TP price exceeded')
        position = {'direction': 'LONG', 'quantity': 0.01,
                     'tp_order_id': None, 'tp_price': 51000.0}
        result = _verify_single_tp_order(exchange, position, 'BTC/USDT:USDT', {})
        assert result is True
        assert position['tp_order_id'] == _EXCHANGE_MANAGED

    def test_other_exchange_error_no_change(self):
        """Other ExchangeError → logs, no state change."""
        exchange = MagicMock()
        exchange.create_order.side_effect = ccxt.ExchangeError('999 unknown')
        position = {'direction': 'LONG', 'quantity': 0.01,
                     'tp_order_id': None, 'tp_price': 51000.0}
        result = _verify_single_tp_order(exchange, position, 'BTC/USDT:USDT', {})
        assert result is False

    def test_generic_exception_no_crash(self):
        """Generic exception → no crash."""
        exchange = MagicMock()
        exchange.create_order.side_effect = RuntimeError('weird')
        position = {'direction': 'LONG', 'quantity': 0.01,
                     'tp_order_id': None, 'tp_price': 51000.0}
        result = _verify_single_tp_order(exchange, position, 'BTC/USDT:USDT', {})
        assert result is False

    def test_tp_price_missing_returns_false(self):
        """tp_price missing → returns False immediately."""
        exchange = MagicMock()
        position = {'direction': 'LONG', 'quantity': 0.01, 'tp_order_id': None}
        result = _verify_single_tp_order(exchange, position, 'BTC/USDT:USDT', {})
        assert result is False
        exchange.create_order.assert_not_called()

    def test_exchange_managed_skips(self):
        """tp_order_id == EXCHANGE_MANAGED → skips verification."""
        exchange = MagicMock()
        position = {'direction': 'LONG', 'quantity': 0.01,
                     'tp_order_id': _EXCHANGE_MANAGED, 'tp_price': 51000.0}
        result = _verify_single_tp_order(exchange, position, 'BTC/USDT:USDT', {})
        assert result is False
        exchange.create_order.assert_not_called()


# ── _verify_sl_order error paths ──────────────────────────────


class TestVerifySlOrderErrors:
    """Test _verify_sl_order exchange error and generic exception."""

    def test_other_exchange_error(self):
        """Non-110406 ExchangeError → logs error, no state change."""
        exchange = MagicMock()
        exchange.create_order.side_effect = ccxt.ExchangeError('999 unknown')
        position = {'direction': 'LONG', 'quantity': 0.01,
                     'sl_order_id': None, 'sl_price': 49000.0}
        result = _verify_sl_order(exchange, position, 'BTC/USDT:USDT', {})
        assert result is False

    def test_generic_exception(self):
        """Generic exception → no crash, returns False."""
        exchange = MagicMock()
        exchange.create_order.side_effect = RuntimeError('weird')
        position = {'direction': 'LONG', 'quantity': 0.01,
                     'sl_order_id': None, 'sl_price': 49000.0}
        result = _verify_sl_order(exchange, position, 'BTC/USDT:USDT', {})
        assert result is False

    def test_sl_price_missing_returns_false(self):
        """sl_price missing → returns False immediately."""
        exchange = MagicMock()
        position = {'direction': 'LONG', 'quantity': 0.01, 'sl_order_id': None}
        result = _verify_sl_order(exchange, position, 'BTC/USDT:USDT', {})
        assert result is False
        exchange.create_order.assert_not_called()

    def test_exchange_managed_skips(self):
        """sl_order_id == EXCHANGE_MANAGED → skips verification."""
        exchange = MagicMock()
        position = {'direction': 'LONG', 'quantity': 0.01,
                     'sl_order_id': _EXCHANGE_MANAGED, 'sl_price': 49000.0}
        result = _verify_sl_order(exchange, position, 'BTC/USDT:USDT', {})
        assert result is False
        exchange.create_order.assert_not_called()


# ── verify_tp_sl_orders error paths (lines 350, 368-371) ────


class TestVerifyTpSlOrdersErrorPaths:
    """Test verify_tp_sl_orders scale-out branch and exception handling."""

    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.orders.save_state')
    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.orders._verify_sl_order',
           return_value=False)
    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.orders._verify_scale_out_orders',
           return_value=True)
    def test_scale_out_branch_called(self, mock_vso, mock_vsl, mock_save):
        """scale_out_enabled + stages → calls _verify_scale_out_orders (line 350)."""
        exchange = MagicMock()
        exchange.fetch_open_orders.return_value = []
        state = {'positions': {'s1': {'slot_id': 's1',
            'direction': 'LONG',
            'scale_out_enabled': True,
            'scale_out_stages': [{'stage': 1, 'order_id': 'so_1'}],
            'tp_order_id': None, 'sl_order_id': None,
        }}, 'active_direction': 'LONG'}
        verify_tp_sl_orders(exchange, state, {'symbol': 'BTC/USDT:USDT'})
        mock_vso.assert_called_once()
        mock_save.assert_called_once()

    def test_exchange_error_caught(self):
        """ExchangeError on fetch_open_orders → caught (lines 368-369)."""
        exchange = MagicMock()
        exchange.fetch_open_orders.side_effect = ccxt.ExchangeError('invalid')
        state = {'positions': {'s1': {'slot_id': 's1', 'tp_order_id': 'tp_1', 'sl_order_id': 'sl_1'}}, 'active_direction': 'LONG'}
        verify_tp_sl_orders(exchange, state, {'symbol': 'BTC/USDT:USDT'})

    def test_generic_exception_caught(self):
        """Generic exception on fetch_open_orders → caught (lines 370-371)."""
        exchange = MagicMock()
        exchange.fetch_open_orders.side_effect = RuntimeError('unexpected')
        state = {'positions': {'s1': {'slot_id': 's1', 'tp_order_id': 'tp_1', 'sl_order_id': 'sl_1'}}, 'active_direction': 'LONG'}
        verify_tp_sl_orders(exchange, state, {'symbol': 'BTC/USDT:USDT'})


# ── _verify_scale_out_orders generic exception (lines 484-485) ──


class TestVerifyScaleOutOrdersGenericException:
    """Test _verify_scale_out_orders generic exception on re-placement."""

    def test_generic_exception_on_replace(self):
        """RuntimeError on create_order → logger.exception, returns False (lines 484-485)."""
        exchange = MagicMock()
        exchange.create_order.side_effect = RuntimeError('unexpected')
        position = {'direction': 'LONG', 'quantity': 0.01}
        stages = [
            {'stage': 1, 'order_id': None, 'quantity': 0.005, 'tp_price': 51000.0},
        ]
        result = _verify_scale_out_orders(exchange, position, 'BTC/USDT:USDT', stages, {})
        assert result is False


# ── Verify TP/SL qty mismatch guard ────────────────────────


class TestVerifyQtyMismatchGuard:
    """Test verify_tp_sl_orders skips directions where exchange qty < local sum."""

    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.orders.save_state')
    def test_verify_skips_direction_with_qty_mismatch(self, mock_save):
        """When exchange qty < local sum for LONG, LONG slots are skipped (no re-placement)."""
        exchange = MagicMock()

        # Two LONG slots, 0.01 each → local sum = 0.02
        state = {
            'positions': {
                's1': {
                    'slot_id': 's1', 'direction': 'LONG', 'quantity': 0.01,
                    'tp_order_id': 'tp1', 'sl_order_id': 'sl1',
                    'tp_price': 51000.0, 'sl_price': 49000.0,
                },
                's2': {
                    'slot_id': 's2', 'direction': 'LONG', 'quantity': 0.01,
                    'tp_order_id': 'tp2', 'sl_order_id': 'sl2',
                    'tp_price': 51000.0, 'sl_price': 49000.0,
                },
            },
            'active_direction': 'LONG',
        }
        config = {'symbol': 'BTC/USDT:USDT'}

        # fetch_open_orders: tp1 gone (would normally trigger re-placement)
        exchange.fetch_open_orders.return_value = [
            {'id': 'sl1'}, {'id': 'tp2'}, {'id': 'sl2'},
        ]

        # fetch_positions: exchange has only 0.01 (one slot filled) < 0.02 local
        exchange.fetch_positions.return_value = [
            {'contracts': 0.01, 'side': 'long'},
        ]

        verify_tp_sl_orders(exchange, state, config)

        # create_order should NOT be called (skipped due to qty mismatch)
        exchange.create_order.assert_not_called()

    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.orders.save_state')
    def test_verify_proceeds_when_qty_matches(self, mock_save):
        """When exchange qty matches local sum, verify proceeds normally."""
        exchange = MagicMock()
        exchange.create_order.return_value = {'id': 'tp1_new'}

        state = {
            'positions': {
                's1': {
                    'slot_id': 's1', 'direction': 'LONG', 'quantity': 0.01,
                    'tp_order_id': 'tp1', 'sl_order_id': 'sl1',
                    'tp_price': 51000.0, 'sl_price': 49000.0,
                },
            },
            'active_direction': 'LONG',
        }
        config = {'symbol': 'BTC/USDT:USDT'}

        # tp1 missing from open orders
        exchange.fetch_open_orders.return_value = [{'id': 'sl1'}]

        # exchange qty matches local
        exchange.fetch_positions.return_value = [
            {'contracts': 0.01, 'side': 'long'},
        ]

        verify_tp_sl_orders(exchange, state, config)

        # TP should be re-placed since qty matches
        assert exchange.create_order.called

    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.orders.save_state')
    def test_verify_fetch_positions_failure_no_skip(self, mock_save):
        """When fetch_positions fails, no directions are skipped (fallback)."""
        exchange = MagicMock()
        exchange.create_order.return_value = {'id': 'tp1_new'}

        state = {
            'positions': {
                's1': {
                    'slot_id': 's1', 'direction': 'LONG', 'quantity': 0.01,
                    'tp_order_id': 'tp1', 'sl_order_id': 'sl1',
                    'tp_price': 51000.0, 'sl_price': 49000.0,
                },
            },
            'active_direction': 'LONG',
        }
        config = {'symbol': 'BTC/USDT:USDT'}

        exchange.fetch_open_orders.return_value = [{'id': 'sl1'}]  # tp1 missing
        exchange.fetch_positions.side_effect = ccxt.NetworkError('timeout')

        verify_tp_sl_orders(exchange, state, config)

        # Should proceed normally (no skip) → re-place TP
        assert exchange.create_order.called
