"""Tests for pure position logic — calculate_tp_sl, exit inference."""

import pytest

from bingx_rl_trading_bot.scripts.production.pattern_5m.position_open import calculate_tp_sl
from bingx_rl_trading_bot.scripts.production.pattern_5m.position_monitor import (
    _infer_exit_reason,
    _infer_exit_from_price,
)
from bingx_rl_trading_bot.scripts.production.pattern_5m.constants import (
    SLIPPAGE_BUFFER_PCT,
    PRICE_TOLERANCE_PCT,
    TP_LOWER_MULT,
    TP_UPPER_MULT,
    SL_LOWER_MULT,
    SL_UPPER_MULT,
)


# ── calculate_tp_sl ──────────────────────────────────────────


class TestCalculateTpSl:
    """Test calculate_tp_sl() price calculation."""

    @pytest.fixture
    def base_strategy(self):
        return {'tp_pct': 2.0, 'sl_pct': 3.0}

    def test_long_tp_above_entry(self, base_strategy):
        """LONG: TP should be above entry price."""
        tp, sl, _, _ = calculate_tp_sl(50000.0, 1, base_strategy, 1.0)
        assert tp > 50000.0

    def test_long_sl_below_entry(self, base_strategy):
        """LONG: SL should be below entry price."""
        tp, sl, _, _ = calculate_tp_sl(50000.0, 1, base_strategy, 1.0)
        assert sl < 50000.0

    def test_short_tp_below_entry(self, base_strategy):
        """SHORT: TP should be below entry price."""
        tp, sl, _, _ = calculate_tp_sl(50000.0, -1, base_strategy, 1.0)
        assert tp < 50000.0

    def test_short_sl_above_entry(self, base_strategy):
        """SHORT: SL should be above entry price."""
        tp, sl, _, _ = calculate_tp_sl(50000.0, -1, base_strategy, 1.0)
        assert sl > 50000.0

    def test_slippage_buffer_applied(self, base_strategy):
        """TP pct includes slippage buffer; SL pct subtracts it."""
        _, _, tp_adj, sl_adj = calculate_tp_sl(50000.0, 1, base_strategy, 1.0)
        assert tp_adj == pytest.approx(2.0 + SLIPPAGE_BUFFER_PCT)
        assert sl_adj == pytest.approx(3.0 - SLIPPAGE_BUFFER_PCT)

    def test_vol_mult_scaling(self, base_strategy):
        """vol_mult > 1 should widen both TP and SL."""
        tp1, sl1, _, _ = calculate_tp_sl(50000.0, 1, base_strategy, 1.0)
        tp2, sl2, _, _ = calculate_tp_sl(50000.0, 1, base_strategy, 1.5)
        # LONG: tp2 > tp1 (wider TP), sl2 < sl1 (wider SL)
        assert tp2 > tp1
        assert sl2 < sl1

    def test_sl_floor_guard(self):
        """SL pct should not go below 0.1% even with slippage buffer."""
        strategy = {'tp_pct': 1.0, 'sl_pct': 0.02}  # very small SL
        _, _, _, sl_adj = calculate_tp_sl(50000.0, 1, strategy, 1.0)
        assert sl_adj >= 0.1

    def test_returns_four_values(self, base_strategy):
        """Should return (tp_price, sl_price, tp_pct_adjusted, sl_pct_adjusted)."""
        result = calculate_tp_sl(50000.0, 1, base_strategy, 1.0)
        assert len(result) == 4

    def test_dynamic_per_pattern_priority(self, base_strategy):
        """Dynamic per-pattern TP/SL takes priority over strategy defaults."""
        config = {
            '_dynamic_tpsl_per_pattern': True,
            '_dynamic_patterns_tpsl': {'U-U-U': (1.5, 2.5)},
        }
        tp, sl, tp_adj, sl_adj = calculate_tp_sl(
            50000.0, 1, base_strategy, 1.0, pattern='U-U-U', config=config
        )
        # Should use 1.5/2.5 not strategy's 2.0/3.0
        assert tp_adj == pytest.approx(1.5 + SLIPPAGE_BUFFER_PCT)
        assert sl_adj == pytest.approx(2.5 - SLIPPAGE_BUFFER_PCT)

    def test_dynamic_per_pattern_fallback(self, base_strategy):
        """Pattern not in dynamic dict → falls back to strategy defaults."""
        config = {
            '_dynamic_tpsl_per_pattern': True,
            '_dynamic_patterns_tpsl': {},  # no patterns
        }
        _, _, tp_adj, sl_adj = calculate_tp_sl(
            50000.0, 1, base_strategy, 1.0, pattern='MISSING', config=config
        )
        assert tp_adj == pytest.approx(2.0 + SLIPPAGE_BUFFER_PCT)
        assert sl_adj == pytest.approx(3.0 - SLIPPAGE_BUFFER_PCT)

    def test_dynamic_universal_mode(self, base_strategy):
        """Dynamic universal TP/SL overrides strategy defaults."""
        config = {
            '_dynamic_tpsl_universal': True,
            '_dynamic_tp': 1.8,
            '_dynamic_sl': 2.8,
        }
        _, _, tp_adj, sl_adj = calculate_tp_sl(
            50000.0, 1, base_strategy, 1.0, config=config
        )
        assert tp_adj == pytest.approx(1.8 + SLIPPAGE_BUFFER_PCT)
        assert sl_adj == pytest.approx(2.8 - SLIPPAGE_BUFFER_PCT)

    def test_regime_tp_sl_priority(self, base_strategy):
        """Regime TP/SL overrides strategy defaults but not dynamic."""
        _, _, tp_adj, sl_adj = calculate_tp_sl(
            50000.0, 1, base_strategy, 1.0, regime_tp_sl=(1.2, 1.8)
        )
        assert tp_adj == pytest.approx(1.2 + SLIPPAGE_BUFFER_PCT)
        assert sl_adj == pytest.approx(1.8 - SLIPPAGE_BUFFER_PCT)

    def test_price_rounding(self, base_strategy):
        """TP/SL prices should be rounded to 1 decimal."""
        tp, sl, _, _ = calculate_tp_sl(50000.123, 1, base_strategy, 1.0)
        # Check rounded to 1 decimal
        assert tp == round(tp, 1)
        assert sl == round(sl, 1)

    def test_symmetric_long_short(self, base_strategy):
        """LONG and SHORT at same entry should produce symmetric TP/SL distances."""
        tp_l, sl_l, tp_adj_l, sl_adj_l = calculate_tp_sl(50000.0, 1, base_strategy, 1.0)
        tp_s, sl_s, tp_adj_s, sl_adj_s = calculate_tp_sl(50000.0, -1, base_strategy, 1.0)
        # Adjusted percentages should be the same
        assert tp_adj_l == pytest.approx(tp_adj_s)
        assert sl_adj_l == pytest.approx(sl_adj_s)
        # TP/SL distance from entry should be approximately symmetric
        assert abs(tp_l - 50000.0) == pytest.approx(abs(tp_s - 50000.0), abs=1.0)
        assert abs(sl_l - 50000.0) == pytest.approx(abs(sl_s - 50000.0), abs=1.0)


# ── _infer_exit_reason ───────────────────────────────────────


class TestInferExitReason:
    """Test _infer_exit_reason() from trade history filled price."""

    def test_tp_hit(self):
        """Filled price near TP → 'TP'."""
        position = {'tp_price': 51000.0, 'sl_price': 49000.0}
        assert _infer_exit_reason(51000.0, position) == 'TP'

    def test_sl_hit(self):
        """Filled price near SL → 'SL'."""
        position = {'tp_price': 51000.0, 'sl_price': 49000.0}
        assert _infer_exit_reason(49000.0, position) == 'SL'

    def test_market_exit(self):
        """Filled price far from both TP/SL → 'MARKET'."""
        position = {'tp_price': 51000.0, 'sl_price': 49000.0}
        assert _infer_exit_reason(50000.0, position) == 'MARKET'

    def test_no_tp_sl(self):
        """Missing TP/SL → 'MARKET'."""
        position = {}
        assert _infer_exit_reason(50000.0, position) == 'MARKET'

    def test_tp_within_tolerance(self):
        """Filled price within PRICE_TOLERANCE_PCT of TP → 'TP'."""
        tp = 51000.0
        # Just within tolerance
        filled = tp * (1 + PRICE_TOLERANCE_PCT * 0.9)
        position = {'tp_price': tp, 'sl_price': 49000.0}
        assert _infer_exit_reason(filled, position) == 'TP'

    def test_tp_outside_tolerance(self):
        """Filled price outside PRICE_TOLERANCE_PCT of TP → not 'TP'."""
        tp = 51000.0
        filled = tp * (1 + PRICE_TOLERANCE_PCT * 1.5)
        position = {'tp_price': tp, 'sl_price': 49000.0}
        result = _infer_exit_reason(filled, position)
        # Could be MARKET or SL, but not TP
        assert result != 'TP'


# ── _infer_exit_from_price ───────────────────────────────────


class TestInferExitFromPrice:
    """Test _infer_exit_from_price() from current price vs TP/SL levels."""

    def test_long_tp_hit(self):
        """LONG: price above TP range → 'TP'."""
        position = {
            'direction': 'LONG',
            'tp_price': 51000.0,
            'sl_price': 49000.0,
        }
        assert _infer_exit_from_price(51050.0, position) == 'TP'

    def test_long_sl_hit(self):
        """LONG: price below SL range → 'SL'."""
        position = {
            'direction': 'LONG',
            'tp_price': 51000.0,
            'sl_price': 49000.0,
        }
        assert _infer_exit_from_price(48950.0, position) == 'SL'

    def test_long_unknown(self):
        """LONG: price between TP and SL → 'UNKNOWN'."""
        position = {
            'direction': 'LONG',
            'tp_price': 51000.0,
            'sl_price': 49000.0,
        }
        assert _infer_exit_from_price(50000.0, position) == 'UNKNOWN'

    def test_short_tp_hit(self):
        """SHORT: price below TP range → 'TP'."""
        position = {
            'direction': 'SHORT',
            'tp_price': 49000.0,
            'sl_price': 51000.0,
        }
        assert _infer_exit_from_price(48950.0, position) == 'TP'

    def test_short_sl_hit(self):
        """SHORT: price above SL range → 'SL'."""
        position = {
            'direction': 'SHORT',
            'tp_price': 49000.0,
            'sl_price': 51000.0,
        }
        assert _infer_exit_from_price(51050.0, position) == 'SL'

    def test_short_unknown(self):
        """SHORT: price between TP and SL → 'UNKNOWN'."""
        position = {
            'direction': 'SHORT',
            'tp_price': 49000.0,
            'sl_price': 51000.0,
        }
        assert _infer_exit_from_price(50000.0, position) == 'UNKNOWN'

    def test_no_tp_sl_prices(self):
        """Missing TP/SL prices → 'UNKNOWN'."""
        position = {'direction': 'LONG'}
        assert _infer_exit_from_price(50000.0, position) == 'UNKNOWN'

    def test_zero_tp_sl(self):
        """TP/SL = 0 → 'UNKNOWN' (zero guard)."""
        position = {
            'direction': 'LONG',
            'tp_price': 0,
            'sl_price': 0,
        }
        assert _infer_exit_from_price(50000.0, position) == 'UNKNOWN'

    def test_long_exact_tp(self):
        """LONG: price exactly at TP → 'TP'."""
        position = {
            'direction': 'LONG',
            'tp_price': 51000.0,
            'sl_price': 49000.0,
        }
        # Exactly at TP * TP_LOWER_MULT
        assert _infer_exit_from_price(51000.0, position) == 'TP'

    def test_short_exact_tp(self):
        """SHORT: price exactly at TP → 'TP'."""
        position = {
            'direction': 'SHORT',
            'tp_price': 49000.0,
            'sl_price': 51000.0,
        }
        assert _infer_exit_from_price(49000.0, position) == 'TP'
