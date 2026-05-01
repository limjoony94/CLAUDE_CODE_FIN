"""P2 Run — force-flow reversal hypothesis testing.

Frequency scan + 8 evaluations (4 mechanisms × 2 friction scenarios).
Locked configs per `experiments/p2/precommit.md` (Option α — no sweep).

Output:
- experiments/p2/frequency_scan.md
- experiments/p2/results_raw.json
- experiments/p2/result.md (closure)

Run: python scripts/analysis/p2_run.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from analysis.p2_features import (
    add_proxy_score_long,
    add_proxy_score_short,
    add_price_quantile_30d,
    add_basis_z_30d,
)
from validators.friction_model import get_friction
from validators.bootstrap_six_criteria import (
    bootstrap_six_criteria,
    T_SEAL_START_MS,
    assert_no_sealed_data,
)

DATA_DIR = PROJECT_ROOT / "data"
EXP_DIR = PROJECT_ROOT / "experiments" / "p2"
EXP_DIR.mkdir(exist_ok=True)

# Locked configs from experiments/p2/precommit.md (Option α single-config)
MECHANISMS = {
    "M001": {
        "name": "force_flow_long_proxy",
        "hypothesis": "H3",
        "signal_fn": "force_flow_long",
        "params": {"threshold_high": 5.5, "low_q": 0.30},
        "forward_minutes": 240,  # 4h
        "scale": "1h",
    },
    "M002": {
        "name": "force_flow_short_proxy",
        "hypothesis": "H5",
        "signal_fn": "force_flow_short",
        "params": {"threshold_high": 5.5, "high_q": 0.70},
        "forward_minutes": 60,  # 1h
        "scale": "1h",
    },
    "M003": {
        "name": "cascade_window_5m_long",
        "hypothesis": "H4",
        "signal_fn": "force_flow_long",
        "params": {"threshold_high": 5.5, "low_q": 0.30},
        "forward_minutes": 15,
        "scale": "1m",
    },
    "M004": {
        "name": "basis_fade_perp",
        "hypothesis": "H7_basis",
        "signal_fn": "basis_fade",
        "params": {"basis_z_threshold": 2.0},
        "forward_minutes": 240,  # 4h
        "scale": "1h",
    },
}


# ==================== Signal generators ====================

def signal_force_flow_long(df_perp_1h, df_funding, threshold_high=5.5, low_q=0.30) -> pd.Series:
    """H3 long: proxy_score > th AND price_quantile_30d ≤ low_q. Returns int Series {0,1}."""
    proxy = add_proxy_score_long(df_perp_1h, df_funding)
    qpct = add_price_quantile_30d(df_perp_1h)
    sig = ((proxy > threshold_high) & (qpct <= low_q)).fillna(False).astype(int)
    return sig


def signal_force_flow_short(df_perp_1h, df_funding, threshold_high=5.5, high_q=0.70) -> pd.Series:
    """H5 short: proxy_score_short > th AND price_quantile_30d ≥ high_q. Returns int Series {-1, 0}."""
    proxy = add_proxy_score_short(df_perp_1h, df_funding)
    qpct = add_price_quantile_30d(df_perp_1h)
    sig = ((proxy > threshold_high) & (qpct >= high_q)).fillna(False)
    return -sig.astype(int)


def signal_basis_fade(df_perp_1h, df_spot_1h, basis_z_threshold=2.0) -> pd.Series:
    """H7-basis: bz < -threshold → long perp; bz > +threshold → short perp. Returns Series {-1,0,1}."""
    bz = add_basis_z_30d(df_perp_1h, df_spot_1h)
    sig = pd.Series(0, index=df_perp_1h.index, dtype=int)
    sig.loc[bz < -basis_z_threshold] = 1
    sig.loc[bz > basis_z_threshold] = -1
    return sig


# ==================== Backtest ====================

def evaluate_mechanism(mech_id, signals, df_perp_1h, df_perp_1m, forward_minutes, scenario):
    """Returns (daily_pnl: pd.Series indexed by day, n_signal_events: int)."""
    fr = get_friction(scenario)
    rt_pct = fr.round_trip_pct / 100.0

    if forward_minutes >= 60:
        # 1h forward
        forward_hours = forward_minutes // 60
        fwd_ret = np.log(df_perp_1h["close"].shift(-forward_hours) / df_perp_1h["close"])
        fwd_ret = fwd_ret.values
    else:
        # 1m forward (M003)
        df_perp_1m_idx = df_perp_1m.set_index("t_open_ms")["close"]
        forward_ms = forward_minutes * 60_000
        # Entry at bar close = t_open + 1h; forward to entry + forward_ms
        entry_ts = (df_perp_1h["t_open_ms"] + 3_600_000).values
        forward_ts = entry_ts + forward_ms
        # Vectorized lookup: pandas indexing
        entry_close = pd.Series(entry_ts).map(lambda t: df_perp_1m_idx.get(t, np.nan))
        forward_close = pd.Series(forward_ts).map(lambda t: df_perp_1m_idx.get(t, np.nan))
        fwd_ret = np.log(forward_close.values / entry_close.values)

    signal_arr = signals.values if isinstance(signals, pd.Series) else np.asarray(signals)
    is_active = (signal_arr != 0).astype(int)
    bar_pnl = np.where(np.isnan(fwd_ret), 0.0, signal_arr * fwd_ret) - is_active * rt_pct

    # Daily aggregate — preserve UTC tz to match baseline index
    df_local = df_perp_1h[["t_open_ms"]].copy()
    df_local["bar_pnl"] = bar_pnl
    df_local["day"] = pd.to_datetime(df_local["t_open_ms"], unit="ms", utc=True).dt.floor("D")
    daily = df_local.groupby("day")["bar_pnl"].sum()
    return daily, int(is_active.sum())


def buy_and_hold_daily(df_perp_1h: pd.DataFrame) -> pd.Series:
    """Daily log-return aggregation from 1h close."""
    df = df_perp_1h.copy()
    df["log_ret"] = np.log(df["close"]) - np.log(df["close"].shift(1))
    df["day"] = pd.to_datetime(df["t_open_ms"], unit="ms", utc=True).dt.floor("D")
    daily = df.groupby("day")["log_ret"].sum().dropna()
    return daily


# ==================== Orchestrator ====================

def main():
    print("=== P2 Run — Force-Flow Reversal Hypothesis Testing ===")

    # Load free window data
    df_perp_1h = pd.read_parquet(DATA_DIR / "btc_perp_1h_720d.parquet")
    df_perp_1h = df_perp_1h[df_perp_1h["t_close_ms"] < T_SEAL_START_MS].reset_index(drop=True)
    assert_no_sealed_data(df_perp_1h, "t_close_ms", T_SEAL_START_MS)

    df_funding = pd.read_parquet(DATA_DIR / "btc_funding_binance_720d.parquet")
    df_funding = df_funding[df_funding["timestamp_ms"] < T_SEAL_START_MS].reset_index(drop=True)

    df_spot_1h = pd.read_parquet(DATA_DIR / "btc_spot_1h_720d.parquet")
    df_spot_1h = df_spot_1h[df_spot_1h["t_close_ms"] < T_SEAL_START_MS].reset_index(drop=True)

    df_perp_1m = pd.read_parquet(DATA_DIR / "btc_perp_1m_365d.parquet")
    df_perp_1m = df_perp_1m[df_perp_1m["t_close_ms"] < T_SEAL_START_MS].reset_index(drop=True)

    print(f"Free window: 1h={len(df_perp_1h)}, 1m={len(df_perp_1m)}, "
          f"funding={len(df_funding)}, spot1h={len(df_spot_1h)}")

    # Generate signals
    sig_m001 = signal_force_flow_long(df_perp_1h, df_funding, **MECHANISMS["M001"]["params"])
    sig_m002 = signal_force_flow_short(df_perp_1h, df_funding, **MECHANISMS["M002"]["params"])
    sig_m003 = signal_force_flow_long(df_perp_1h, df_funding, **MECHANISMS["M003"]["params"])
    sig_m004 = signal_basis_fade(df_perp_1h, df_spot_1h, **MECHANISMS["M004"]["params"])
    signals = {"M001": sig_m001, "M002": sig_m002, "M003": sig_m003, "M004": sig_m004}

    # Frequency scan
    n_days = (df_perp_1h["t_open_ms"].max() - df_perp_1h["t_open_ms"].min()) / (86_400 * 1000)
    print(f"\n=== Frequency Scan ===")
    print(f"Free window span: {n_days:.1f} days")
    freq_summary = {}
    for mid, sig in signals.items():
        n_active = int((sig != 0).sum())
        freq = n_active / n_days if n_days > 0 else 0
        vacuous = freq < 0.5
        freq_summary[mid] = {
            "n_signals": n_active,
            "freq_per_day": round(float(freq), 3),
            "vacuous": vacuous,
            "min_freq_gate": 0.5,
        }
        flag = " ⚠️ VACUOUS" if vacuous else ""
        print(f"  {mid}: {n_active} signals over {n_days:.1f}d = {freq:.3f}/day{flag}")

    # Buy-and-hold baseline (same window)
    baseline_daily = buy_and_hold_daily(df_perp_1h)

    # Evaluate 8 (4 mech × 2 scenario)
    print(f"\n=== 6-Criteria Evaluations ===")
    results = {
        "timestamp_utc": "2026-05-01",
        "free_window_days": float(n_days),
        "n_baseline_days": int(len(baseline_daily)),
        "frequency_scan": freq_summary,
        "evaluations": {},
    }

    for mid, mech_cfg in MECHANISMS.items():
        for scenario in ["realistic", "stress"]:
            key = f"{mid}_{scenario}"
            if freq_summary[mid]["vacuous"]:
                results["evaluations"][key] = {
                    "status": "VACUOUS_FAIL",
                    "reason": f"freq {freq_summary[mid]['freq_per_day']:.3f}/d < gate 0.5/d",
                    "all_pass": False,
                    "n_signals": freq_summary[mid]["n_signals"],
                }
                print(f"  {key}: VACUOUS_FAIL (freq < 0.5/d)")
                continue
            try:
                daily, n_signals = evaluate_mechanism(
                    mid, signals[mid], df_perp_1h, df_perp_1m,
                    mech_cfg["forward_minutes"], scenario,
                )
                daily_aligned = daily.reindex(baseline_daily.index, fill_value=0.0)
                eval_result = bootstrap_six_criteria(
                    daily_pnl=daily_aligned,
                    baseline_pnl=baseline_daily,
                    priority="P2",
                    B=10000,
                    block_size=3,
                    seed=42,
                )
                eval_result["status"] = "PASS" if eval_result["all_pass"] else "FAIL"
                eval_result["n_signals_actually_evaluated"] = int(n_signals)
                results["evaluations"][key] = eval_result
                pe = eval_result["point_estimates"]
                cr = eval_result["criteria"]
                pass_count = sum(cr.values())
                print(f"\n  {key} → {eval_result['status']} ({pass_count}/6) signals={n_signals}")
                print(f"    mean: {pe['mean']*100:+.4f}%/d ({'✅' if cr['mean_pass'] else '❌'})")
                print(f"    p5(bs): {pe['p5']*100:+.4f}% ({'✅' if cr['p5_pass'] else '❌'})")
                print(f"    pos_rate: {pe['pos_rate']:.3f} ({'✅' if cr['pos_rate_pass'] else '❌'})")
                print(f"    p_beats: {pe['p_beats_baseline']:.3f} ({'✅' if cr['p_beats_pass'] else '❌'})")
                print(f"    max_dd: {pe['max_dd']*100:+.2f}% ({'✅' if cr['max_dd_pass'] else '❌'})")
                print(f"    sharpe: {pe['sharpe']:+.2f} ({'✅' if cr['sharpe_pass'] else '❌'})")
            except Exception as e:
                results["evaluations"][key] = {
                    "status": "ERROR",
                    "error": f"{type(e).__name__}: {str(e)[:300]}",
                    "all_pass": False,
                }
                print(f"\n  {key} → ERROR {type(e).__name__}: {e}")

    # Verdict
    pass_count = sum(1 for r in results["evaluations"].values() if r.get("all_pass"))
    fail_count = sum(1 for r in results["evaluations"].values() if r.get("status") in ("FAIL", "VACUOUS_FAIL"))
    print(f"\n=== P2 Verdict ===")
    print(f"PASS: {pass_count}/8, FAIL: {fail_count}/8")

    out_path = EXP_DIR / "results_raw.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nSaved {out_path}")
    return results


if __name__ == "__main__":
    main()
