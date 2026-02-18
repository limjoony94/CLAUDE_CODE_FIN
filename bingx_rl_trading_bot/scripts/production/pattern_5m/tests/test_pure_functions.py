"""Tests for pure functions — calculate_pnl, extract_pattern_name, setup_scale_out."""

import pytest

from bingx_rl_trading_bot.scripts.production.pattern_5m.position_close import calculate_pnl
from bingx_rl_trading_bot.scripts.production.pattern_5m.position_open import setup_scale_out
from bingx_rl_trading_bot.scripts.production.pattern_5m.utils import extract_pattern_name
from bingx_rl_trading_bot.scripts.production.pattern_5m.constants import FEE_PCT


# ── calculate_pnl ────────────────────────────────────────────


class TestCalculatePnl:
    """Test calculate_pnl() — leveraged and price-basis PnL."""

    def test_long_profit(self):
        """LONG: exit above entry → positive PnL."""
        pnl, price_pnl = calculate_pnl(50000.0, 51000.0, direction=1, leverage=3)
        assert pnl > 0
        assert price_pnl > 0

    def test_long_loss(self):
        """LONG: exit below entry → negative PnL."""
        pnl, price_pnl = calculate_pnl(50000.0, 49000.0, direction=1, leverage=3)
        assert pnl < 0
        assert price_pnl < 0

    def test_short_profit(self):
        """SHORT: exit below entry → positive PnL."""
        pnl, price_pnl = calculate_pnl(50000.0, 49000.0, direction=-1, leverage=3)
        assert pnl > 0
        assert price_pnl > 0

    def test_short_loss(self):
        """SHORT: exit above entry → negative PnL."""
        pnl, price_pnl = calculate_pnl(50000.0, 51000.0, direction=-1, leverage=3)
        assert pnl < 0
        assert price_pnl < 0

    def test_leverage_multiplier(self):
        """Leveraged PnL should be leverage × price PnL (approximately, fees differ)."""
        pnl_3x, price_pnl = calculate_pnl(50000.0, 51000.0, direction=1, leverage=3)
        pnl_1x, _ = calculate_pnl(50000.0, 51000.0, direction=1, leverage=1)
        # Price move is 2%, fee is 0.05% × 2 = 0.10%
        # pnl_3x ≈ (2% - 0.10%) × 3 = 5.7%, pnl_1x ≈ 2% - 0.10% = 1.9%
        # Ratio should be approximately 3x
        assert abs(pnl_3x / pnl_1x - 3.0) < 0.1

    def test_fee_deduction(self):
        """PnL should be less than raw price movement (fees deducted)."""
        pnl, _ = calculate_pnl(50000.0, 51000.0, direction=1, leverage=1)
        raw_pnl = (51000.0 / 50000.0 - 1) * 100  # 2.0%
        assert pnl < raw_pnl  # Fees reduce PnL

    def test_exact_values(self):
        """Verify exact PnL calculation with known values."""
        # LONG: entry=50000, exit=51000, 3x leverage
        # Price move = 1 × (51000/50000 - 1) × 100 = 2.0%
        # pnl = 2.0 × 3 - 2 × 0.05 × 3 = 6.0 - 0.30 = 5.70%
        # price_pnl = 2.0 - 2 × 0.05 = 2.0 - 0.10 = 1.90%
        pnl, price_pnl = calculate_pnl(50000.0, 51000.0, direction=1, leverage=3)
        assert pnl == pytest.approx(5.70, abs=0.01)
        assert price_pnl == pytest.approx(1.90, abs=0.01)

    def test_breakeven_exit(self):
        """Exit at entry → PnL should be slightly negative (fees only)."""
        pnl, _ = calculate_pnl(50000.0, 50000.0, direction=1, leverage=3)
        expected = -2 * FEE_PCT * 3  # Only fees
        assert pnl == pytest.approx(expected, abs=0.001)

    def test_symmetric_long_short(self):
        """Same move distance: LONG profit ≈ SHORT profit (symmetric)."""
        pnl_long, _ = calculate_pnl(50000.0, 51000.0, direction=1, leverage=3)
        pnl_short, _ = calculate_pnl(50000.0, 49000.0, direction=-1, leverage=3)
        assert pnl_long == pytest.approx(pnl_short, abs=0.1)


# ── extract_pattern_name ─────────────────────────────────────


class TestExtractPatternName:
    """Test extract_pattern_name() — regex pattern extraction."""

    def test_standard_format(self):
        """Standard reason string → extracts pattern."""
        assert extract_pattern_name('Pattern: BD-BD-U (LONG)') == 'BD-BD-U'

    def test_long_suffix(self):
        """Pattern with LONG direction suffix."""
        assert extract_pattern_name('Pattern: U-MU-H (LONG)') == 'U-MU-H'

    def test_short_suffix(self):
        """Pattern with SHORT direction suffix."""
        assert extract_pattern_name('Pattern: DN-BU-BU (SHORT)') == 'DN-BU-BU'

    def test_recovery_reason(self):
        """Recovered position reason."""
        assert extract_pattern_name('Recovered from exchange (BD-BD-BU)') == ''

    def test_pattern_colon_format(self):
        """Pattern: NAME format without parentheses."""
        assert extract_pattern_name('Pattern: MU-BD-ST') == 'MU-BD-ST'

    def test_no_pattern(self):
        """No pattern in reason → empty string."""
        assert extract_pattern_name('Manual trade') == ''

    def test_empty_string(self):
        """Empty string → empty string."""
        assert extract_pattern_name('') == ''

    def test_none_safe(self):
        """Should handle edge cases gracefully."""
        # extract_pattern_name expects str, but check empty
        assert extract_pattern_name('Pattern: ') == ''


# ── setup_scale_out ──────────────────────────────────────────


class TestSetupScaleOut:
    """Test setup_scale_out() — scale-out stage calculation."""

    @pytest.fixture
    def so_strategy(self):
        """Strategy with scale-out enabled."""
        return {
            'scale_out': {
                'enabled': True,
                'stages': [(0.5, 0.5), (0.5, 1.0)],  # 50% at 50% TP, 50% at 100% TP
            }
        }

    @pytest.fixture
    def no_so_strategy(self):
        """Strategy without scale-out."""
        return {'scale_out': {'enabled': False}}

    def test_disabled_returns_empty(self, no_so_strategy):
        """Scale-out disabled → empty list."""
        stages = setup_scale_out(no_so_strategy, 50000.0, 0.01, 1, 2.0)
        assert stages == []

    def test_missing_scale_out_key(self):
        """No scale_out key → empty list."""
        stages = setup_scale_out({}, 50000.0, 0.01, 1, 2.0)
        assert stages == []

    def test_enabled_creates_stages(self, so_strategy):
        """Enabled scale-out → correct number of stages."""
        stages = setup_scale_out(so_strategy, 50000.0, 0.01, 1, 2.0)
        assert len(stages) == 2

    def test_stage_quantities_sum(self, so_strategy):
        """Stage quantities should sum to total position quantity."""
        quantity = 0.01
        stages = setup_scale_out(so_strategy, 50000.0, quantity, 1, 2.0)
        total = sum(s['quantity'] for s in stages)
        assert total == pytest.approx(quantity, abs=0.0001)

    def test_stage_tp_prices_ascending_long(self, so_strategy):
        """LONG: stage TP prices should be ascending."""
        stages = setup_scale_out(so_strategy, 50000.0, 0.01, 1, 2.0)
        assert stages[0]['tp_price'] < stages[1]['tp_price']

    def test_stage_tp_prices_descending_short(self, so_strategy):
        """SHORT: stage TP prices should be descending."""
        stages = setup_scale_out(so_strategy, 50000.0, 0.01, -1, 2.0)
        assert stages[0]['tp_price'] > stages[1]['tp_price']

    def test_stage_fields(self, so_strategy):
        """Each stage should have all required fields."""
        stages = setup_scale_out(so_strategy, 50000.0, 0.01, 1, 2.0)
        for stage in stages:
            assert 'stage' in stage
            assert 'pct' in stage
            assert 'tp_mult' in stage
            assert 'tp_price' in stage
            assert 'quantity' in stage
            assert 'filled' in stage
            assert stage['filled'] is False

    def test_long_tp_above_entry(self, so_strategy):
        """LONG: all TP prices should be above entry."""
        stages = setup_scale_out(so_strategy, 50000.0, 0.01, 1, 2.0)
        for stage in stages:
            assert stage['tp_price'] > 50000.0

    def test_short_tp_below_entry(self, so_strategy):
        """SHORT: all TP prices should be below entry."""
        stages = setup_scale_out(so_strategy, 50000.0, 0.01, -1, 2.0)
        for stage in stages:
            assert stage['tp_price'] < 50000.0
