"""P2 Features — lookahead-aware z-scores + price quantile + basis.

ALL features computed with t-1 shift on raw data; observation time = t_close.
Validator unit test enforces no leak.

Public API:
    add_z_funding_neg(df_funding_1h)  -> Series indexed by 1h grid t_open_ms
    add_z_velocity(df_perp_1h, sign='neg'|'pos') -> Series
    add_z_volume_surge(df_perp_1h) -> Series
    add_proxy_score_long(df_funding, df_perp) -> Series  (sum of zs, neg-side)
    add_proxy_score_short(...) -> Series (pos-side mirror)
    add_price_quantile_30d(df_perp_1h) -> Series
    add_basis_pct(df_perp, df_spot) -> Series
    add_basis_z_30d(df_perp, df_spot) -> Series

All Series indexed by `t_open_ms` so they can be merged into base 1h frame.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
import pandas as pd

# Constants
HOUR_MS = 3_600_000
DAY_MS = 86_400_000
ROLLING_30D_HOURS = 30 * 24  # 720 bars at 1h
ATR_LOOKBACK = 14


def _ensure_1h_indexed(df: pd.DataFrame) -> pd.DataFrame:
    """Sort by t_open_ms ascending; assert monotonic + interval == 1h."""
    df = df.sort_values("t_open_ms").reset_index(drop=True)
    diffs = df["t_open_ms"].diff().dropna()
    if len(diffs) == 0:
        return df
    common = diffs.value_counts().index[0]
    assert int(common) == HOUR_MS, f"Expected 1h interval, got mode {common}"
    return df


def add_funding_to_1h_grid(df_funding_8h: pd.DataFrame, df_perp_1h: pd.DataFrame) -> pd.Series:
    """Forward-fill 8h funding rate to 1h grid.

    Returns Series indexed positionally with df_perp_1h, value = funding_rate at last 8h tick ≤ t_open.
    """
    df_perp = _ensure_1h_indexed(df_perp_1h)
    df_funding = df_funding_8h.sort_values("timestamp_ms").reset_index(drop=True)
    out = np.full(len(df_perp), np.nan, dtype=np.float64)
    f_ts = df_funding["timestamp_ms"].values
    f_rates = df_funding["funding_rate"].values
    j = 0
    for i, t_open in enumerate(df_perp["t_open_ms"].values):
        while j < len(f_ts) - 1 and f_ts[j + 1] <= t_open:
            j += 1
        if j < len(f_ts) and f_ts[j] <= t_open:
            out[i] = f_rates[j]
    return pd.Series(out, index=df_perp.index, name="funding_rate_1h_ffilled")


def z_score_rolling(s: pd.Series, window: int) -> pd.Series:
    """Rolling z-score: (x - rolling_mean) / rolling_std with t-1 shift.

    Use shift(1) to ensure value at row i uses only data through row i-1.
    """
    rolled_mean = s.shift(1).rolling(window=window, min_periods=window).mean()
    rolled_std = s.shift(1).rolling(window=window, min_periods=window).std()
    return (s.shift(1) - rolled_mean) / rolled_std.replace(0, np.nan)


def add_z_funding_neg(df_perp_1h: pd.DataFrame, df_funding_8h: pd.DataFrame) -> pd.Series:
    """z-score of NEGATIVE funding (longs being paid). Higher = stronger negative funding."""
    df_perp = _ensure_1h_indexed(df_perp_1h)
    funding_1h = add_funding_to_1h_grid(df_funding_8h, df_perp)
    z = z_score_rolling(funding_1h, ROLLING_30D_HOURS)
    return -z  # invert so high = negative funding stronger


def add_z_funding_pos(df_perp_1h: pd.DataFrame, df_funding_8h: pd.DataFrame) -> pd.Series:
    """z-score of POSITIVE funding (longs paying shorts). Higher = stronger positive funding."""
    df_perp = _ensure_1h_indexed(df_perp_1h)
    funding_1h = add_funding_to_1h_grid(df_funding_8h, df_perp)
    return z_score_rolling(funding_1h, ROLLING_30D_HOURS)


def compute_atr(df_perp_1h: pd.DataFrame, lookback: int = ATR_LOOKBACK) -> pd.Series:
    """ATR with t-1 shift (uses high/low/close through previous bar)."""
    df = _ensure_1h_indexed(df_perp_1h)
    h = df["high"].shift(1)
    l = df["low"].shift(1)
    c_prev = df["close"].shift(2)
    tr = pd.concat([(h - l).abs(),
                     (h - c_prev).abs(),
                     (l - c_prev).abs()], axis=1).max(axis=1)
    atr = tr.rolling(window=lookback, min_periods=lookback).mean()
    return atr


def add_z_velocity_neg(df_perp_1h: pd.DataFrame, lookback_hours: int = 12) -> pd.Series:
    """z_velocity_neg: -(close[t-1] - close[t-1-lookback]) / atr_14[t-1].

    Higher = sharper drop relative to ATR (long-side cascade signal).
    """
    df = _ensure_1h_indexed(df_perp_1h)
    close_t_1 = df["close"].shift(1)
    close_lookback = df["close"].shift(1 + lookback_hours)
    atr = compute_atr(df, ATR_LOOKBACK)
    velocity = -(close_t_1 - close_lookback) / atr.replace(0, np.nan)
    return velocity


def add_z_velocity_pos(df_perp_1h: pd.DataFrame, lookback_hours: int = 12) -> pd.Series:
    """z_velocity_pos: +(close[t-1] - close[t-1-lookback]) / atr_14[t-1]."""
    df = _ensure_1h_indexed(df_perp_1h)
    close_t_1 = df["close"].shift(1)
    close_lookback = df["close"].shift(1 + lookback_hours)
    atr = compute_atr(df, ATR_LOOKBACK)
    return (close_t_1 - close_lookback) / atr.replace(0, np.nan)


def add_z_volume_surge(df_perp_1h: pd.DataFrame) -> pd.Series:
    """z-score of 1h volume vs 30d rolling distribution (t-1 shift)."""
    df = _ensure_1h_indexed(df_perp_1h)
    return z_score_rolling(df["volume"], ROLLING_30D_HOURS)


def add_price_quantile_30d(df_perp_1h: pd.DataFrame) -> pd.Series:
    """Rolling 30d trailing close-quantile rank of close[t-1]."""
    df = _ensure_1h_indexed(df_perp_1h)
    close_t_1 = df["close"].shift(1)
    out = np.full(len(df), np.nan, dtype=np.float64)
    window = ROLLING_30D_HOURS
    for i in range(window, len(df)):
        window_data = close_t_1.iloc[i - window + 1:i + 1].dropna().values
        if len(window_data) > 0:
            out[i] = (window_data < close_t_1.iloc[i]).mean()
    return pd.Series(out, index=df.index, name="price_quantile_30d")


def add_proxy_score_long(df_perp_1h: pd.DataFrame,
                          df_funding_8h: pd.DataFrame,
                          velocity_lookback_hours: int = 12) -> pd.Series:
    """proxy_score_long = z_funding_neg + z_velocity_neg + z_volume_surge."""
    z_fn = add_z_funding_neg(df_perp_1h, df_funding_8h)
    z_vn = add_z_velocity_neg(df_perp_1h, velocity_lookback_hours)
    z_vs = add_z_volume_surge(df_perp_1h)
    return z_fn + z_vn + z_vs


def add_proxy_score_short(df_perp_1h: pd.DataFrame,
                            df_funding_8h: pd.DataFrame,
                            velocity_lookback_hours: int = 12) -> pd.Series:
    """proxy_score_short = z_funding_pos + z_velocity_pos + z_volume_surge."""
    z_fp = add_z_funding_pos(df_perp_1h, df_funding_8h)
    z_vp = add_z_velocity_pos(df_perp_1h, velocity_lookback_hours)
    z_vs = add_z_volume_surge(df_perp_1h)
    return z_fp + z_vp + z_vs


def add_basis_pct(df_perp_1h: pd.DataFrame, df_spot_1h: pd.DataFrame) -> pd.Series:
    """basis_pct = (perp - spot) / spot × 100, observable at t (close to close).

    For lookahead-free use, take basis at t-1 in downstream features.
    """
    df_perp = _ensure_1h_indexed(df_perp_1h)
    df_spot = _ensure_1h_indexed(df_spot_1h)
    merged = df_perp[["t_open_ms", "close"]].rename(columns={"close": "perp_close"}).merge(
        df_spot[["t_open_ms", "close"]].rename(columns={"close": "spot_close"}),
        on="t_open_ms", how="left"
    )
    return ((merged["perp_close"] - merged["spot_close"]) / merged["spot_close"] * 100).reset_index(drop=True)


def add_basis_z_30d(df_perp_1h: pd.DataFrame, df_spot_1h: pd.DataFrame) -> pd.Series:
    """z-score of basis_pct vs 30d rolling distribution, with t-1 shift."""
    basis = add_basis_pct(df_perp_1h, df_spot_1h)
    return z_score_rolling(basis, ROLLING_30D_HOURS)


def forward_return(df_perp_1h: pd.DataFrame, forward_hours: int) -> pd.Series:
    """log(close[t + forward_hours] / close[t]) — for label."""
    df = _ensure_1h_indexed(df_perp_1h)
    return np.log(df["close"].shift(-forward_hours) / df["close"])


__all__ = [
    "add_funding_to_1h_grid",
    "z_score_rolling",
    "compute_atr",
    "add_z_funding_neg",
    "add_z_funding_pos",
    "add_z_velocity_neg",
    "add_z_velocity_pos",
    "add_z_volume_surge",
    "add_price_quantile_30d",
    "add_proxy_score_long",
    "add_proxy_score_short",
    "add_basis_pct",
    "add_basis_z_30d",
    "forward_return",
]
