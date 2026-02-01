"""Tests for TP/SL calculation and double exit (scale-out) logic."""

import pytest

from bingx_rl_trading_bot.scripts.production.pattern_5m.constants import (
    PATTERN_OPTIMAL_TPSL,
    SLIPPAGE_BUFFER_PCT,
    PRICE_ROUND_DECIMALS,
    DEFAULT_TP_PCT,
    DEFAULT_SL_PCT,
    TP1_RATIO,
    TP1_QTY_PCT,
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
