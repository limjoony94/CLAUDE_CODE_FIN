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
    _verify_single_tp_order,
    _verify_sl_order,
    _cancel_existing_tpsl_orders,
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
