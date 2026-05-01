"""P0.4 Baselines — buy-and-hold + 1× constant long + random entry.

Validator end-to-end demonstration (advisor process check):
- Buy-and-hold should PASS at P0_BASELINE (loose) + serve as p_beats reference
- 1× long with funding should be borderline (drift vs funding cost)
- Random entry should FAIL (taker friction every day)

Sealed boundary enforced: only first 540d (T < T_seal_start) used.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from validators.friction_model import (
    get_friction,
    round_trip_cost_pct,
    apply_friction_to_returns,
)
from validators.bootstrap_six_criteria import (
    bootstrap_six_criteria,
    T_SEAL_START_MS,
    assert_no_sealed_data,
    compute_max_drawdown,
    compute_annualized_sharpe,
)

DATA_DIR = PROJECT_ROOT / "data"
EXP_DIR = PROJECT_ROOT / "experiments" / "p0"


def load_perp_1h_free() -> pd.DataFrame:
    df = pd.read_parquet(DATA_DIR / "btc_perp_1h_720d.parquet")
    df_free = df[df["t_close_ms"] < T_SEAL_START_MS].reset_index(drop=True)
    # Sanity: assert no leak
    assert_no_sealed_data(df_free, time_col="t_close_ms", T_seal_start_ms=T_SEAL_START_MS)
    return df_free


def load_funding_free() -> pd.DataFrame:
    df = pd.read_parquet(DATA_DIR / "btc_funding_binance_720d.parquet")
    df_free = df[df["timestamp_ms"] < T_SEAL_START_MS].reset_index(drop=True)
    return df_free


def aggregate_daily_close(df_1h: pd.DataFrame) -> pd.DataFrame:
    """Reduce 1h bars to daily close + computed log_ret."""
    df = df_1h.copy()
    df["day"] = pd.to_datetime(df["t_open_ms"], unit="ms", utc=True).dt.floor("D")
    daily = df.groupby("day").agg({"open": "first", "close": "last"}).reset_index()
    daily["log_ret"] = np.log(daily["close"] / daily["close"].shift(1))
    daily = daily.dropna(subset=["log_ret"]).reset_index(drop=True)
    return daily


def baseline_buy_and_hold(daily: pd.DataFrame, friction_scenario: str = "realistic") -> np.ndarray:
    """Buy-and-hold daily returns. Single entry start, single exit end.

    Per-day friction is negligible (RT 0.16% spread over 500+d ≈ 0.0003%/day).
    For pure baseline reference, return raw log_ret.
    """
    return daily["log_ret"].values


def baseline_1x_constant_long_with_funding(daily: pd.DataFrame,
                                              df_funding: pd.DataFrame,
                                              friction_scenario: str = "realistic") -> np.ndarray:
    """1× constant long: log_ret - funding_paid_per_day.

    Funding rate convention: positive funding rate → longs pay (subtract).
    """
    fr_df = df_funding.copy()
    fr_df["day"] = pd.to_datetime(fr_df["timestamp_ms"], unit="ms", utc=True).dt.floor("D")
    funding_daily = fr_df.groupby("day")["funding_rate"].sum().reset_index()
    merged = daily.merge(funding_daily, on="day", how="left")
    merged["funding_rate"] = merged["funding_rate"].fillna(0)
    # Long pays positive funding → subtract from net return
    merged["net_with_funding"] = merged["log_ret"] - merged["funding_rate"]
    return merged["net_with_funding"].values


def baseline_random_entry(daily: pd.DataFrame, friction_scenario: str = "realistic",
                           seed: int = 42) -> np.ndarray:
    """Random direction (±1) per day, pays full RT friction every day."""
    fr = get_friction(friction_scenario)
    rt_pct = round_trip_cost_pct(fr) / 100.0  # fraction
    rng = np.random.default_rng(seed)
    direction = rng.choice([-1, 1], size=len(daily))
    raw = daily["log_ret"].values
    return direction * raw - rt_pct


def evaluate_baseline(name: str, returns: np.ndarray, baseline_returns: np.ndarray = None,
                        priority: str = "P0_BASELINE", B: int = 10000) -> dict:
    """Run 6-criteria evaluation."""
    s = pd.Series(returns)
    bs = pd.Series(baseline_returns) if baseline_returns is not None else None
    return bootstrap_six_criteria(s, baseline_pnl=bs, priority=priority,
                                    B=B, block_size=3, seed=42)


def main():
    print(f"=== P0.4 Baselines (T_seal_start={T_SEAL_START_MS}) ===")
    df_1h = load_perp_1h_free()
    df_funding = load_funding_free()
    daily = aggregate_daily_close(df_1h)
    print(f"Free window: {len(daily)} daily samples")
    print(f"Span: {daily['day'].min()} to {daily['day'].max()}\n")

    # Baselines
    bh_realistic = baseline_buy_and_hold(daily, "realistic")
    long_funding_realistic = baseline_1x_constant_long_with_funding(daily, df_funding, "realistic")
    random_realistic = baseline_random_entry(daily, "realistic", seed=42)
    random_stress = baseline_random_entry(daily, "stress", seed=42)

    # Evaluate
    results = {
        "buy_and_hold": evaluate_baseline("buy_and_hold", bh_realistic, priority="P0_BASELINE"),
        "1x_long_with_funding_realistic": evaluate_baseline(
            "1x_long_with_funding_realistic", long_funding_realistic,
            baseline_returns=bh_realistic, priority="P0_BASELINE"),
        "random_entry_realistic": evaluate_baseline(
            "random_entry_realistic", random_realistic,
            baseline_returns=bh_realistic, priority="P0_BASELINE"),
        "random_entry_stress": evaluate_baseline(
            "random_entry_stress", random_stress,
            baseline_returns=bh_realistic, priority="P0_BASELINE"),
    }

    # Save raw
    out_path = EXP_DIR / "baselines_raw.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)

    # Console summary
    print("=== Results ===")
    for name, r in results.items():
        if "error" in r:
            print(f"\n[{name}] ERROR: {r['error']}")
            continue
        pe = r["point_estimates"]
        crit = r["criteria"]
        print(f"\n[{name}]")
        print(f"  n_samples={r['n_samples']}")
        print(f"  mean: {pe['mean']*100:.5f}%/day  ({'✅' if crit['mean_pass'] else '❌'})")
        print(f"  p5: {pe['p5']*100:.4f}%  ({'✅' if crit['p5_pass'] else '❌'})")
        print(f"  pos_rate: {pe['pos_rate']:.3f}  ({'✅' if crit['pos_rate_pass'] else '❌'})")
        print(f"  p_beats_baseline: {pe['p_beats_baseline']:.3f}  ({'✅' if crit['p_beats_pass'] else '❌'})")
        print(f"  max_dd: {pe['max_dd']*100:.2f}%  ({'✅' if crit['max_dd_pass'] else '❌'})")
        print(f"  sharpe: {pe['sharpe']:.2f}  ({'✅' if crit['sharpe_pass'] else '❌'})")
        print(f"  ALL_PASS (P0_BASELINE): {r['all_pass']}")

    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
