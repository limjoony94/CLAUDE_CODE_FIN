"""P0.3 validator tests — friction model + 6-criteria bootstrap.

Run: pytest tests/test_p03_validators.py -v
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# Make scripts/validators importable
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from validators.friction_model import (
    FrictionParams,
    get_friction,
    round_trip_cost_pct,
    round_trip_cost_usd,
    per_leg_taker_fee_usd,
    adjust_fill_price,
    funding_accrual_usd,
    apply_friction_to_returns,
    list_scenarios,
)
from validators.bootstrap_six_criteria import (
    DEFAULT_THRESHOLDS,
    T_SEAL_START_MS,
    compute_max_drawdown,
    compute_annualized_sharpe,
    cbb_resample,
    bootstrap_six_criteria,
    assert_no_sealed_data,
)


# ==================== Friction Model Tests ====================

class TestFrictionScenarios:
    def test_realistic_round_trip_pct(self):
        fr = get_friction("realistic")
        assert abs(fr.round_trip_pct - 0.16) < 1e-9, fr.round_trip_pct
        assert abs(round_trip_cost_pct(fr) - 0.16) < 1e-9

    def test_optimistic_round_trip_pct(self):
        fr = get_friction("optimistic")
        assert abs(fr.round_trip_pct - 0.10) < 1e-9

    def test_stress_round_trip_pct(self):
        fr = get_friction("stress")
        assert abs(fr.round_trip_pct - 0.20) < 1e-9

    def test_unknown_scenario_raises(self):
        with pytest.raises(ValueError, match="Unknown friction scenario"):
            get_friction("does_not_exist")

    def test_maker_stub_raises(self):
        with pytest.raises(NotImplementedError, match="Maker scenario"):
            get_friction("maker_realistic")

    def test_list_scenarios(self):
        scenarios = list_scenarios()
        assert set(scenarios) == {"optimistic", "realistic", "stress"}


class TestFrictionCosts:
    def test_round_trip_cost_usd_realistic(self):
        # 1500 USD * 0.16% / 100 = $2.40
        fr = get_friction("realistic")
        cost = round_trip_cost_usd(1500.0, fr)
        assert abs(cost - 2.40) < 1e-9, cost

    def test_per_leg_taker_fee(self):
        # 1500 * 0.045% / 100 = $0.675
        fr = get_friction("realistic")
        fee = per_leg_taker_fee_usd(1500.0, fr)
        assert abs(fee - 0.675) < 1e-9, fee

    def test_negative_notional_raises(self):
        fr = get_friction("realistic")
        with pytest.raises(ValueError):
            round_trip_cost_usd(-100.0, fr)


class TestFillPriceAdjustment:
    def test_buy_slip_higher(self):
        fr = get_friction("realistic")
        # 0.035% slippage on buy
        fill = adjust_fill_price(50000.0, "buy", fr)
        expected = 50000.0 * 1.00035
        assert abs(fill - expected) < 1e-6

    def test_sell_slip_lower(self):
        fr = get_friction("realistic")
        fill = adjust_fill_price(50000.0, "sell", fr)
        expected = 50000.0 * 0.99965
        assert abs(fill - expected) < 1e-6

    def test_invalid_side(self):
        fr = get_friction("realistic")
        with pytest.raises(ValueError, match="side"):
            adjust_fill_price(50000.0, "hold", fr)


class TestFundingAccrual:
    def test_long_pos_funding_long_pays(self):
        # Long $1000, single 8h tick of +0.01% (0.0001)
        # Long PAYS short → received = -1000 * 0.0001 = -$0.10
        result = funding_accrual_usd(1000.0, [0.0001])
        assert abs(result - (-0.10)) < 1e-9, result

    def test_long_neg_funding_long_receives(self):
        # Long $1000, single 8h tick of -0.01%
        # Long RECEIVES → +$0.10
        result = funding_accrual_usd(1000.0, [-0.0001])
        assert abs(result - 0.10) < 1e-9, result

    def test_short_pos_funding_short_receives(self):
        # Short $1000 (signed -1000), tick +0.0001
        # Short receives → result positive
        result = funding_accrual_usd(-1000.0, [0.0001])
        assert abs(result - 0.10) < 1e-9, result

    def test_long_24h_pos_funding(self):
        # Long $1000, 3 ticks (24h) of +0.0001 each
        # Total rate = 0.0003 → paid 1000 * 0.0003 = $0.30
        result = funding_accrual_usd(1000.0, [0.0001, 0.0001, 0.0001])
        assert abs(result - (-0.30)) < 1e-9, result

    def test_empty_funding_zero(self):
        assert funding_accrual_usd(1000.0, []) == 0.0


class TestApplyFrictionToReturns:
    def test_zero_trades_no_friction(self):
        rets = np.array([0.005, 0.01, -0.002])
        fr = get_friction("realistic")
        result = apply_friction_to_returns(rets, n_trades_per_period=0.0, params=fr)
        np.testing.assert_array_almost_equal(result, rets)

    def test_one_trade_per_period(self):
        # 0.16% RT subtracted from each period
        rets = np.array([0.01, 0.02])
        fr = get_friction("realistic")
        result = apply_friction_to_returns(rets, n_trades_per_period=1.0, params=fr)
        expected = np.array([0.01 - 0.0016, 0.02 - 0.0016])
        np.testing.assert_array_almost_equal(result, expected)


# ==================== Bootstrap & 6-Criteria Tests ====================

class TestComputeMetrics:
    def test_max_drawdown_basic(self):
        # +1, -3, +1 → cum [1, -2, -1] → peak [1, 1, 1] → dd [0, -3, -2] → min -3
        pnl = np.array([1.0, -3.0, 1.0])
        assert compute_max_drawdown(pnl) == -3.0

    def test_sharpe_constant_positive(self):
        # All same → std=0 → sharpe=0
        pnl = np.array([0.005] * 100)
        assert compute_annualized_sharpe(pnl) == 0.0

    def test_sharpe_positive(self):
        rng = np.random.default_rng(1)
        pnl = rng.normal(0.005, 0.01, 365)
        s = compute_annualized_sharpe(pnl)
        # Roughly mean/std * sqrt(365) ≈ 0.5 * sqrt(365) ≈ 9.5
        assert 5.0 < s < 15.0


class TestCbbResample:
    def test_shape(self):
        arr = np.arange(100, dtype=float)
        rng = np.random.default_rng(42)
        samples = cbb_resample(arr, block_size=3, n_samples=100, n_resamples=50, rng=rng)
        assert samples.shape == (50, 100)

    def test_values_in_original(self):
        arr = np.arange(50, dtype=float)
        rng = np.random.default_rng(42)
        samples = cbb_resample(arr, block_size=3, n_resamples=10, rng=rng)
        # All sampled values should be from original
        assert set(samples.flatten()).issubset(set(arr))

    def test_block_too_large_raises(self):
        arr = np.arange(5, dtype=float)
        with pytest.raises(ValueError):
            cbb_resample(arr, block_size=10, n_resamples=1)


class TestBootstrapSixCriteria:
    """Advisor's 5 unit tests: (a)-(e) coverage."""

    def test_a_constant_positive_pnl_p0_baseline_pass(self):
        """Constant +0.5%/day → all 6 PASS at P0_BASELINE thresholds."""
        pnl = pd.Series([0.005] * 365)
        # Sharpe with std=0 will be 0. P0_BASELINE has min_sharpe=0 so OK.
        result = bootstrap_six_criteria(pnl, baseline_pnl=None, priority="P0_BASELINE",
                                          B=500, block_size=3, seed=42)
        assert result["criteria"]["mean_pass"], result["point_estimates"]["mean"]
        assert result["criteria"]["p5_pass"]
        assert result["criteria"]["pos_rate_pass"]
        assert result["criteria"]["max_dd_pass"]
        # all_pass at P0_BASELINE
        assert result["all_pass"], result["criteria"]

    def test_b_random_walk_p2_fail_mean_or_p5(self):
        """Random walk centered at 0 → P2 mean threshold (0.10%) fail."""
        rng = np.random.default_rng(1)
        pnl = pd.Series(rng.normal(0.0, 0.01, 365))
        result = bootstrap_six_criteria(pnl, baseline_pnl=None, priority="P2",
                                          B=500, block_size=3, seed=42)
        # mean ≈ 0 < 0.001 (P2 target) → mean_pass False
        assert not result["criteria"]["mean_pass"], result["point_estimates"]["mean"]
        assert not result["all_pass"]

    def test_c_high_mean_extreme_drawdown_fail_dd(self):
        """High mean but extreme one-day drawdown → MaxDD fail."""
        # Mean +1%/day except one day at -10%, baseline P2 (max_dd_floor = -3%)
        pnl_list = [0.01] * 100 + [-0.10] + [0.01] * 100
        pnl = pd.Series(pnl_list)
        result = bootstrap_six_criteria(pnl, baseline_pnl=None, priority="P2",
                                          B=500, block_size=3, seed=42)
        assert not result["criteria"]["max_dd_pass"], result["point_estimates"]["max_dd"]

    def test_d_high_sharpe_negative_mean_fail(self):
        """Negative mean (consistently small loss) → mean fail at P2."""
        pnl = pd.Series([-0.0005] * 365)  # low std → high abs sharpe but negative
        result = bootstrap_six_criteria(pnl, baseline_pnl=None, priority="P2",
                                          B=500, block_size=3, seed=42)
        assert not result["criteria"]["mean_pass"], result["point_estimates"]["mean"]
        assert not result["all_pass"]

    def test_e_sealed_data_assertion_fires(self):
        """assert_no_sealed_data raises AssertionError on leak."""
        # Create df with a row inside sealed window
        df = pd.DataFrame({"t_close_ms": [T_SEAL_START_MS + 1, T_SEAL_START_MS - 1]})
        with pytest.raises(AssertionError, match="Sealed-data leak"):
            assert_no_sealed_data(df)

    def test_e_sealed_data_passes_when_clean(self):
        df = pd.DataFrame({"t_close_ms": [T_SEAL_START_MS - 1000, T_SEAL_START_MS - 500]})
        # Should not raise
        assert_no_sealed_data(df)

    def test_insufficient_samples_raises(self):
        pnl = pd.Series([0.001] * 10)
        with pytest.raises(ValueError, match="insufficient samples"):
            bootstrap_six_criteria(pnl, priority="P2", B=100)

    def test_unknown_priority_raises(self):
        pnl = pd.Series([0.001] * 100)
        with pytest.raises(ValueError, match="Unknown priority"):
            bootstrap_six_criteria(pnl, priority="P99", B=100)


class TestPBeatsBaseline:
    def test_strategy_beats_baseline(self):
        """Strategy mean > baseline mean → p_beats high."""
        rng = np.random.default_rng(42)
        strategy = pd.Series(rng.normal(0.005, 0.01, 365))
        baseline = pd.Series(rng.normal(0.001, 0.01, 365))
        result = bootstrap_six_criteria(strategy, baseline_pnl=baseline, priority="P0_BASELINE",
                                          B=500, block_size=3, seed=42)
        assert result["point_estimates"]["p_beats_baseline"] > 0.7

    def test_strategy_loses_to_baseline(self):
        rng = np.random.default_rng(42)
        strategy = pd.Series(rng.normal(0.0, 0.01, 365))
        baseline = pd.Series(rng.normal(0.005, 0.01, 365))
        result = bootstrap_six_criteria(strategy, baseline_pnl=baseline, priority="P0_BASELINE",
                                          B=500, block_size=3, seed=42)
        assert result["point_estimates"]["p_beats_baseline"] < 0.3
