"""6-criteria gate via stationary block bootstrap (advisor #2 spec: block size = 3 days).

Mandate § 0.5 — bootstrap (B=10000, 3-day random window) on daily P&L:
    1. mean ≥ target_daily
    2. p5 ≥ 0
    3. pos_rate ≥ 0.5
    4. p_beats_baseline ≥ 0.55
    5. MaxDD ≥ -X% (additive cumsum drawdown)
    6. Sharpe ≥ Y (annualized)

Bootstrap method: Circular Block Bootstrap (CBB) with block_size=3 (advisor pick).
Vectorized via NumPy advanced indexing for B=10000 efficiency.

Public API:
    bootstrap_six_criteria(daily_pnl, baseline_pnl, priority, B, block_size, seed)
    compute_max_drawdown(daily_pnl)
    compute_annualized_sharpe(daily_pnl)
    cbb_resample(arr, block_size, n_samples, n_resamples, rng)
    assert_no_sealed_data(df, time_col, T_seal_start_ms)

Sealed boundary (committed in holdout_seal.md):
    T_SEAL_START_MS = 1762128000000  (2025-11-02T14:00:00 UTC)
"""
from __future__ import annotations

from typing import Optional, Union

import numpy as np
import pandas as pd

# Per-priority threshold table (six_criteria_thresholds.md)
DEFAULT_THRESHOLDS = {
    "P0_BASELINE": {"target_daily": 0.0, "max_dd_floor": -1.0, "min_pos_rate": 0.5,
                     "min_p_beats": 0.0, "min_sharpe": 0.0},
    "P2": {"target_daily": 0.001, "max_dd_floor": -0.03, "min_pos_rate": 0.5,
            "min_p_beats": 0.55, "min_sharpe": 1.5},
    "P3_CELL": {"target_daily": 0.001, "max_dd_floor": -0.05, "min_pos_rate": 0.5,
                 "min_p_beats": 0.55, "min_sharpe": 1.5},
    "P3_AGGREGATED": {"target_daily": 0.001, "max_dd_floor": -0.05, "min_pos_rate": 0.5,
                       "min_p_beats": 0.55, "min_sharpe": 2.0},
    "P6_PORTFOLIO": {"target_daily": 0.00073, "max_dd_floor": -0.05, "min_pos_rate": 0.5,
                      "min_p_beats": 0.55, "min_sharpe": 2.0},
}

# Sealed boundary (holdout_seal.md amendment, 2026-05-01 P0.2 closure)
T_SEAL_START_MS = 1762128000000  # 2025-11-02T14:00:00 UTC


def compute_max_drawdown(daily_pnl: Union[pd.Series, np.ndarray]) -> float:
    """Max drawdown (additive cumsum, returns negative fraction)."""
    arr = np.asarray(daily_pnl)
    if len(arr) == 0:
        return 0.0
    cum = arr.cumsum()
    peak = np.maximum.accumulate(cum)
    dd = cum - peak
    return float(dd.min())


def compute_annualized_sharpe(daily_pnl: Union[pd.Series, np.ndarray],
                                periods_per_year: int = 365) -> float:
    """Annualized Sharpe (assumes daily_pnl is fractional).

    Returns 0 if std is below numerical epsilon (constant series).
    """
    arr = np.asarray(daily_pnl, dtype=np.float64)
    if len(arr) < 2:
        return 0.0
    if (arr.max() - arr.min()) < 1e-15:
        return 0.0
    s = arr.std(ddof=1)
    if s < 1e-15:
        return 0.0
    return float((arr.mean() / s) * np.sqrt(periods_per_year))


def cbb_resample(arr: np.ndarray, block_size: int = 3,
                  n_samples: Optional[int] = None,
                  n_resamples: int = 10000,
                  rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Vectorized Circular Block Bootstrap.

    Args:
        arr: 1D array of returns
        block_size: block length (default 3 = "3-day random window")
        n_samples: per-resample length (default = len(arr))
        n_resamples: number of bootstrap samples (B)
        rng: numpy Generator

    Returns:
        2D array shape (n_resamples, n_samples), each row a bootstrap sample.
    """
    if rng is None:
        rng = np.random.default_rng()
    n = len(arr)
    if n_samples is None:
        n_samples = n
    if n < block_size:
        raise ValueError(f"len(arr)={n} < block_size={block_size}")
    n_blocks = (n_samples + block_size - 1) // block_size
    starts = rng.integers(0, n, size=(n_resamples, n_blocks))  # (B, n_blocks)
    offsets = np.arange(block_size)  # (block_size,)
    indices = (starts[:, :, None] + offsets[None, None, :]) % n  # (B, n_blocks, block_size)
    indices = indices.reshape(n_resamples, -1)[:, :n_samples]    # (B, n_samples)
    return arr[indices]


def _vectorized_max_drawdown(samples_2d: np.ndarray) -> np.ndarray:
    """Per-row max drawdown for (B, T) array."""
    cum = samples_2d.cumsum(axis=1)
    peak = np.maximum.accumulate(cum, axis=1)
    dd = cum - peak
    return dd.min(axis=1)


def _vectorized_sharpe(samples_2d: np.ndarray, periods_per_year: int = 365) -> np.ndarray:
    """Per-row annualized Sharpe (zeroed for sub-epsilon std)."""
    means = samples_2d.mean(axis=1)
    stds = samples_2d.std(axis=1, ddof=1)
    safe = np.where(stds > 1e-15, stds, 1.0)
    sharpes = (means / safe) * np.sqrt(periods_per_year)
    sharpes[stds < 1e-15] = 0.0
    return sharpes


def bootstrap_six_criteria(daily_pnl: Union[pd.Series, np.ndarray],
                             baseline_pnl: Optional[Union[pd.Series, np.ndarray]] = None,
                             priority: str = "P2",
                             B: int = 10000,
                             block_size: int = 3,
                             seed: int = 42) -> dict:
    """Run 6-criteria gate via CBB.

    Args:
        daily_pnl: candidate strategy daily returns (fractions)
        baseline_pnl: baseline returns (e.g., buy-and-hold). If None, p_beats=1.0.
        priority: key into DEFAULT_THRESHOLDS
        B: bootstrap iterations
        block_size: block size in days (default 3)
        seed: RNG seed

    Returns:
        dict with point_estimates, criteria, all_pass, bootstrap_dist.

    Raises:
        ValueError if priority unknown or insufficient samples.
    """
    if priority not in DEFAULT_THRESHOLDS:
        raise ValueError(f"Unknown priority: {priority}. Valid: {list(DEFAULT_THRESHOLDS)}")
    thr = DEFAULT_THRESHOLDS[priority]

    arr = np.asarray(pd.Series(daily_pnl).dropna().values, dtype=np.float64)
    n = len(arr)
    if n < 30:
        raise ValueError(f"insufficient samples: {n} (need ≥30 for bootstrap)")
    if n < block_size:
        raise ValueError(f"len={n} < block_size={block_size}")

    rng = np.random.default_rng(seed)

    # Point estimates
    point_mean = float(arr.mean())
    point_p5 = float(np.percentile(arr, 5))
    point_pos_rate = float((arr > 0).mean())
    point_max_dd = compute_max_drawdown(arr)
    point_sharpe = compute_annualized_sharpe(arr)

    # Bootstrap (vectorized)
    samples = cbb_resample(arr, block_size=block_size, n_resamples=B, rng=rng)
    bs_means = samples.mean(axis=1)
    bs_p5 = np.percentile(samples, 5, axis=1)
    bs_pos = (samples > 0).mean(axis=1)
    bs_max_dd = _vectorized_max_drawdown(samples)
    bs_sharpe = _vectorized_sharpe(samples)

    # p_beats_baseline (paired bootstrap of difference of means)
    if baseline_pnl is not None:
        bl = np.asarray(pd.Series(baseline_pnl).dropna().values, dtype=np.float64)
        if len(bl) < block_size:
            p_beats = 1.0
        else:
            rng2 = np.random.default_rng(seed + 1)
            bl_samples = cbb_resample(bl, block_size=block_size, n_resamples=B, rng=rng2)
            # If lengths differ, take min for fair comparison
            min_n = min(samples.shape[1], bl_samples.shape[1])
            p_beats = float((samples[:, :min_n].mean(axis=1) > bl_samples[:, :min_n].mean(axis=1)).mean())
    else:
        p_beats = 1.0

    # Criteria
    crit = {
        "mean_pass": bool(point_mean >= thr["target_daily"]),
        "p5_pass": bool(point_p5 >= 0),
        "pos_rate_pass": bool(point_pos_rate >= thr["min_pos_rate"]),
        "p_beats_pass": bool(p_beats >= thr["min_p_beats"]),
        "max_dd_pass": bool(point_max_dd >= thr["max_dd_floor"]),
        "sharpe_pass": bool(point_sharpe >= thr["min_sharpe"]),
    }
    all_pass = all(crit.values())

    return {
        "priority": priority,
        "thresholds": thr,
        "point_estimates": {
            "mean": point_mean,
            "p5": point_p5,
            "pos_rate": point_pos_rate,
            "p_beats_baseline": float(p_beats),
            "max_dd": point_max_dd,
            "sharpe": point_sharpe,
        },
        "criteria": crit,
        "all_pass": bool(all_pass),
        "bootstrap_dist": {
            "mean": {"q05": float(np.percentile(bs_means, 5)),
                     "median": float(np.median(bs_means)),
                     "q95": float(np.percentile(bs_means, 95))},
            "p5": {"q05": float(np.percentile(bs_p5, 5)),
                   "median": float(np.median(bs_p5)),
                   "q95": float(np.percentile(bs_p5, 95))},
            "pos_rate": {"q05": float(np.percentile(bs_pos, 5)),
                         "median": float(np.median(bs_pos)),
                         "q95": float(np.percentile(bs_pos, 95))},
            "max_dd": {"q05": float(np.percentile(bs_max_dd, 5)),
                       "median": float(np.median(bs_max_dd)),
                       "q95": float(np.percentile(bs_max_dd, 95))},
            "sharpe": {"q05": float(np.percentile(bs_sharpe, 5)),
                       "median": float(np.median(bs_sharpe)),
                       "q95": float(np.percentile(bs_sharpe, 95))},
        },
        "n_samples": n,
        "B": B,
        "block_size": block_size,
        "seed": seed,
    }


def assert_no_sealed_data(df: pd.DataFrame, time_col: str = "t_close_ms",
                           T_seal_start_ms: int = T_SEAL_START_MS) -> None:
    """Guard: input data must not include sealed window. Raises AssertionError on leak.

    Use at start of any P0.5-P5 fitting/searching code.
    """
    if time_col not in df.columns:
        raise ValueError(f"DataFrame missing column {time_col!r}")
    leaks = df[df[time_col] >= T_seal_start_ms]
    if len(leaks) > 0:
        raise AssertionError(
            f"Sealed-data leak: {len(leaks)} rows have {time_col} >= {T_seal_start_ms} "
            f"(T_seal_start = 2025-11-02T14:00:00 UTC). "
            "P0.5-P5 fitting MUST exclude sealed window. "
            f"Filter: df = df[df['{time_col}'] < {T_seal_start_ms}]"
        )


__all__ = [
    "DEFAULT_THRESHOLDS",
    "T_SEAL_START_MS",
    "compute_max_drawdown",
    "compute_annualized_sharpe",
    "cbb_resample",
    "bootstrap_six_criteria",
    "assert_no_sealed_data",
]
