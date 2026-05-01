"""P2 feature tests — lookahead-free + correctness.

Critical: every feature must be observable at t_close, NEVER use future bar data.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from analysis.p2_features import (
    z_score_rolling,
    compute_atr,
    add_z_funding_neg,
    add_z_volume_surge,
    add_price_quantile_30d,
    add_proxy_score_long,
    add_basis_pct,
    add_basis_z_30d,
    forward_return,
    HOUR_MS,
)


def make_synthetic_1h(n_bars=2000, base_price=50000.0, noise_seed=42):
    """Synthetic 1h OHLCV. n_bars hours."""
    rng = np.random.default_rng(noise_seed)
    starts_ms = np.arange(n_bars) * HOUR_MS + 1000000000000
    rets = rng.normal(0.0, 0.005, n_bars)
    close = base_price * np.exp(np.cumsum(rets))
    open_ = np.concatenate([[base_price], close[:-1]])
    high = np.maximum(open_, close) * (1 + rng.uniform(0, 0.003, n_bars))
    low = np.minimum(open_, close) * (1 - rng.uniform(0, 0.003, n_bars))
    volume = rng.uniform(100, 1000, n_bars)
    return pd.DataFrame({
        "t_open_ms": starts_ms,
        "open": open_, "high": high, "low": low, "close": close,
        "volume": volume,
        "t_close_ms": starts_ms + HOUR_MS,
    })


def make_synthetic_funding(n_ticks=300):
    """Synthetic 8h funding: same start as 1h frame."""
    rng = np.random.default_rng(7)
    starts = np.arange(n_ticks) * (8 * HOUR_MS) + 1000000000000
    rates = rng.normal(0.0001, 0.00005, n_ticks)
    return pd.DataFrame({"timestamp_ms": starts, "funding_rate": rates})


# ==================== Lookahead Tests ====================

class TestLookaheadFree:
    def test_z_score_rolling_uses_only_past_data(self):
        """z_score at row i must NOT depend on values at row > i-1."""
        s_a = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0])
        s_b = s_a.copy()
        s_b.iloc[-1] = 9999.0  # change FUTURE value
        # z at row 5 should be identical (it only uses rows 0-4)
        z_a = z_score_rolling(s_a, window=3)
        z_b = z_score_rolling(s_b, window=3)
        for i in range(8):  # rows before the modified future
            if not pd.isna(z_a.iloc[i]):
                assert abs(z_a.iloc[i] - z_b.iloc[i]) < 1e-9, \
                    f"row {i} differs: {z_a.iloc[i]} vs {z_b.iloc[i]}"

    def test_atr_uses_only_past_data(self):
        df_a = make_synthetic_1h(100)
        df_b = df_a.copy()
        df_b.loc[df_b.index[-1], "high"] = 99999.0
        df_b.loc[df_b.index[-1], "low"] = 0.001
        atr_a = compute_atr(df_a, lookback=14)
        atr_b = compute_atr(df_b, lookback=14)
        for i in range(50):
            if not pd.isna(atr_a.iloc[i]):
                assert abs(atr_a.iloc[i] - atr_b.iloc[i]) < 1e-9, \
                    f"ATR row {i} leaks: {atr_a.iloc[i]} vs {atr_b.iloc[i]}"

    def test_proxy_score_long_no_future_leak(self):
        """Modifying last row's funding/price/volume must not change earlier proxy_score."""
        df_perp_a = make_synthetic_1h(2000)
        df_funding = make_synthetic_funding(300)
        df_perp_b = df_perp_a.copy()
        # Corrupt last bar
        df_perp_b.loc[df_perp_b.index[-1], "close"] = 99999.0
        df_perp_b.loc[df_perp_b.index[-1], "volume"] = 999999.0
        ps_a = add_proxy_score_long(df_perp_a, df_funding)
        ps_b = add_proxy_score_long(df_perp_b, df_funding)
        # Only row 1999 (last) and possibly its neighbor should differ
        for i in range(800, 1995):  # well within rolling window range
            if not (pd.isna(ps_a.iloc[i]) or pd.isna(ps_b.iloc[i])):
                assert abs(ps_a.iloc[i] - ps_b.iloc[i]) < 1e-6, \
                    f"row {i} leak: {ps_a.iloc[i]} vs {ps_b.iloc[i]}"

    def test_price_quantile_30d_no_future_leak(self):
        df_a = make_synthetic_1h(1500)
        df_b = df_a.copy()
        df_b.loc[df_b.index[-1], "close"] = 99999.0
        q_a = add_price_quantile_30d(df_a)
        q_b = add_price_quantile_30d(df_b)
        for i in range(800, 1490):
            if not (pd.isna(q_a.iloc[i]) or pd.isna(q_b.iloc[i])):
                assert abs(q_a.iloc[i] - q_b.iloc[i]) < 1e-9, \
                    f"row {i}: {q_a.iloc[i]} vs {q_b.iloc[i]}"


# ==================== Correctness Tests ====================

class TestCorrectness:
    def test_z_score_constant_is_nan_or_zero(self):
        s = pd.Series([5.0] * 50)
        z = z_score_rolling(s, window=10)
        # std = 0 → division → nan or inf
        # Our impl returns nan for std=0
        valid = z.dropna()
        # Either all-nan (correct for constant) or all-zero
        if len(valid) > 0:
            assert valid.abs().max() < 1e-9 or valid.isna().all()

    def test_atr_positive(self):
        df = make_synthetic_1h(200)
        atr = compute_atr(df, lookback=14)
        non_nan = atr.dropna()
        assert (non_nan > 0).all()

    def test_proxy_score_finite(self):
        df_perp = make_synthetic_1h(2000)
        df_funding = make_synthetic_funding(300)
        ps = add_proxy_score_long(df_perp, df_funding)
        non_nan = ps.dropna()
        assert len(non_nan) > 100, f"too few non-nan: {len(non_nan)}"
        assert non_nan.abs().mean() < 100  # should be O(1) not blow up

    def test_basis_pct_basic(self):
        df_perp = make_synthetic_1h(100)
        df_spot = df_perp.copy()
        # Make perp 1% premium
        df_spot["close"] = df_spot["close"] / 1.01
        basis = add_basis_pct(df_perp, df_spot)
        valid = basis.dropna()
        # Should be ≈ 1% on average
        assert 0.5 < valid.mean() < 1.5

    def test_forward_return_shifts_correctly(self):
        df = make_synthetic_1h(100)
        fwd = forward_return(df, forward_hours=4)
        # fwd[i] = log(close[i+4] / close[i])
        for i in range(0, 90):
            expected = np.log(df["close"].iloc[i + 4] / df["close"].iloc[i])
            assert abs(fwd.iloc[i] - expected) < 1e-9


# ==================== Sealed Boundary Tests ====================

class TestSealedSafety:
    def test_features_on_real_free_window(self):
        """Smoke test: features compute on real 540d free window without errors."""
        df_perp = pd.read_parquet(PROJECT_ROOT / "data" / "btc_perp_1h_720d.parquet")
        # Apply free window
        T_SEAL = 1762128000000
        df_free = df_perp[df_perp["t_close_ms"] < T_SEAL].reset_index(drop=True)
        df_funding = pd.read_parquet(PROJECT_ROOT / "data" / "btc_funding_binance_720d.parquet")
        df_funding_free = df_funding[df_funding["timestamp_ms"] < T_SEAL].reset_index(drop=True)

        ps = add_proxy_score_long(df_free, df_funding_free)
        non_nan = ps.dropna()
        assert len(non_nan) > 5000, f"only {len(non_nan)} valid proxy scores"
        # Should have some extreme values
        assert non_nan.max() > 3.0
        assert non_nan.min() < -3.0
