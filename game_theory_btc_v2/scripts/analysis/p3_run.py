"""P3 Run — MAP-Elites Aggregate-Only (Option γ).

14 mechanisms × 2 friction scenarios = 28 evaluations.
Locked configs per mechanism_catalog.yaml + p2/precommit.md (Option α single-config).
NO regime cell breakdown (P3a). P3b conditional on aggregate borderline.

Mechanisms (Phase A active 12 + control_null 2):
- M001 force_flow_long_proxy (H3) — already P2 vacuous, re-confirm
- M002 force_flow_short_proxy (H5) — re-confirm
- M003 cascade_window_5m_long (H4) — re-confirm
- M004 basis_fade_perp (H7-basis) — re-confirm
- M005 funding_momentum_long
- M006 xs_momentum_30d (multi-asset 1d)
- M007 channel_breakout (control_null, expected FAIL)
- M008 compression_breakout (control_null)
- M009 wyckoff_spring (1h proxy of 5m)
- M010 distribution_topping (1h proxy of 15m)
- M011 high_vol_long_dip
- M012 low_vol_carry
- M013 eth_btc_outperform (multi-asset)
- M014 dispersion_low (multi-asset)

Run: python scripts/analysis/p3_run.py
"""
from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from analysis.p2_features import (
    HOUR_MS,
    add_proxy_score_long,
    add_proxy_score_short,
    add_price_quantile_30d,
    add_basis_z_30d,
    add_z_funding_pos,
    add_z_volume_surge,
    add_z_velocity_neg,
    add_z_velocity_pos,
    add_funding_to_1h_grid,
    compute_atr,
)
from validators.friction_model import get_friction
from validators.bootstrap_six_criteria import (
    bootstrap_six_criteria,
    T_SEAL_START_MS,
    assert_no_sealed_data,
)

DATA_DIR = PROJECT_ROOT / "data"
EXP_DIR = PROJECT_ROOT / "experiments" / "p3"
EXP_DIR.mkdir(exist_ok=True)


# ==================== Data Loading ====================

def load_free_data():
    df_perp_1h = pd.read_parquet(DATA_DIR / "btc_perp_1h_720d.parquet")
    df_perp_1h = df_perp_1h[df_perp_1h["t_close_ms"] < T_SEAL_START_MS].reset_index(drop=True)
    assert_no_sealed_data(df_perp_1h, "t_close_ms", T_SEAL_START_MS)

    df_funding = pd.read_parquet(DATA_DIR / "btc_funding_binance_720d.parquet")
    df_funding = df_funding[df_funding["timestamp_ms"] < T_SEAL_START_MS].reset_index(drop=True)

    df_spot_1h = pd.read_parquet(DATA_DIR / "btc_spot_1h_720d.parquet")
    df_spot_1h = df_spot_1h[df_spot_1h["t_close_ms"] < T_SEAL_START_MS].reset_index(drop=True)

    df_perp_1m = pd.read_parquet(DATA_DIR / "btc_perp_1m_365d.parquet")
    df_perp_1m = df_perp_1m[df_perp_1m["t_close_ms"] < T_SEAL_START_MS].reset_index(drop=True)

    df_perp_1d = pd.read_parquet(DATA_DIR / "btc_perp_1d_1500d.parquet")
    df_perp_1d = df_perp_1d[df_perp_1d["t_close_ms"] < T_SEAL_START_MS].reset_index(drop=True)

    df_multi = pd.read_parquet(DATA_DIR / "multi_asset_1d_800d.parquet")
    df_multi = df_multi[df_multi["t_close_ms"] < T_SEAL_START_MS].reset_index(drop=True)

    return df_perp_1h, df_funding, df_spot_1h, df_perp_1m, df_perp_1d, df_multi


# ==================== Helpers ====================

def daily_aggregate(df_perp_1h: pd.DataFrame, bar_pnl: np.ndarray) -> pd.Series:
    """Aggregate per-bar PnL to UTC daily."""
    df_local = df_perp_1h[["t_open_ms"]].copy()
    df_local["bar_pnl"] = bar_pnl
    df_local["day"] = pd.to_datetime(df_local["t_open_ms"], unit="ms", utc=True).dt.floor("D")
    return df_local.groupby("day")["bar_pnl"].sum()


def fwd_return_1h(df_perp_1h, forward_hours):
    return np.log(df_perp_1h["close"].shift(-forward_hours) / df_perp_1h["close"]).values


def fwd_return_1m_via_close(df_perp_1h, df_perp_1m, forward_minutes):
    df_1m_idx = df_perp_1m.set_index("t_open_ms")["close"]
    forward_ms = forward_minutes * 60_000
    entry_ts = (df_perp_1h["t_open_ms"] + 3_600_000).values
    forward_ts = entry_ts + forward_ms
    entry_close = pd.Series(entry_ts).map(lambda t: df_1m_idx.get(t, np.nan))
    forward_close = pd.Series(forward_ts).map(lambda t: df_1m_idx.get(t, np.nan))
    return np.log(forward_close.values / entry_close.values)


def buy_and_hold_daily(df_perp_1h: pd.DataFrame) -> pd.Series:
    df = df_perp_1h.copy()
    df["log_ret"] = np.log(df["close"]) - np.log(df["close"].shift(1))
    df["day"] = pd.to_datetime(df["t_open_ms"], unit="ms", utc=True).dt.floor("D")
    return df.groupby("day")["log_ret"].sum().dropna()


# ==================== 1h-Grid Mechanism Signal Generators ====================

def sig_M001(df_perp_1h, df_funding_8h, _):
    proxy = add_proxy_score_long(df_perp_1h, df_funding_8h)
    qpct = add_price_quantile_30d(df_perp_1h)
    return ((proxy > 5.5) & (qpct <= 0.30)).fillna(False).astype(int)


def sig_M002(df_perp_1h, df_funding_8h, _):
    proxy = add_proxy_score_short(df_perp_1h, df_funding_8h)
    qpct = add_price_quantile_30d(df_perp_1h)
    return -((proxy > 5.5) & (qpct >= 0.70)).fillna(False).astype(int)


def sig_M003(df_perp_1h, df_funding_8h, _):
    return sig_M001(df_perp_1h, df_funding_8h, _)  # same signal, different forward


def sig_M004(df_perp_1h, df_spot_1h, _):
    bz = add_basis_z_30d(df_perp_1h, df_spot_1h)
    s = pd.Series(0, index=df_perp_1h.index, dtype=int)
    s.loc[bz < -2.0] = 1
    s.loc[bz > 2.0] = -1
    return s


def sig_M005(df_perp_1h, df_funding_8h, _):
    """funding spike + price up + volume surge → long continuation."""
    z_fund_pos = add_z_funding_pos(df_perp_1h, df_funding_8h)
    qpct = add_price_quantile_30d(df_perp_1h)
    z_vol = add_z_volume_surge(df_perp_1h)
    return ((z_fund_pos > 2.0) & (qpct >= 0.60) & (z_vol > 1.0)).fillna(False).astype(int)


def sig_M007(df_perp_1h, _, __):
    """Channel breakout on 1h grid (proxy for 15m). EXPECTED FAIL (control_null)."""
    df = df_perp_1h.copy()
    high_15bar = df["high"].rolling(15, min_periods=15).max().shift(1)
    body_ratio = (df["close"] - df["open"]).abs() / (df["high"] - df["low"]).replace(0, np.nan)
    s = pd.Series(0, index=df.index, dtype=int)
    long_break = (df["close"] > high_15bar) & (body_ratio > 0.40)
    low_15bar = df["low"].rolling(15, min_periods=15).min().shift(1)
    short_break = (df["close"] < low_15bar) & (body_ratio > 0.40)
    s.loc[long_break.fillna(False)] = 1
    s.loc[short_break.fillna(False) & ~long_break.fillna(False)] = -1
    return s


def sig_M008(df_perp_1h, _, __):
    """Compression breakout. EXPECTED FAIL (control_null)."""
    df = df_perp_1h.copy()
    atr = compute_atr(df, lookback=7)
    atr_q = atr.rolling(720).quantile(0.30).shift(1)
    sma20 = df["close"].rolling(20).mean()
    bb_std = df["close"].rolling(20).std()
    bb_width = (4 * bb_std / sma20).rolling(720).quantile(0.20).shift(1)
    is_compressed = (atr.shift(1) < atr_q) & ((4 * bb_std / sma20).shift(1) < bb_width)
    high_5 = df["high"].rolling(5, min_periods=5).max().shift(1)
    low_5 = df["low"].rolling(5, min_periods=5).min().shift(1)
    s = pd.Series(0, index=df.index, dtype=int)
    s.loc[(is_compressed & (df["close"] > high_5)).fillna(False)] = 1
    s.loc[(is_compressed & (df["close"] < low_5)).fillna(False)] = -1
    return s


def sig_M009(df_perp_1h, _, __):
    """Wyckoff spring proxy on 1h (false breakdown + recovery). long_only."""
    df = df_perp_1h.copy()
    atr = compute_atr(df, 14)
    low_30 = df["low"].rolling(30, min_periods=30).min().shift(1)
    # Spring: previous 30 bars' min, current close > that min by >= 0.5 ATR, volume > avg*1.5
    vol_avg = df["volume"].rolling(30, min_periods=30).mean().shift(1)
    spring = (df["close"] > low_30 + 0.5 * atr) & (df["close"] > df["open"]) & (df["volume"] > vol_avg * 1.5)
    return spring.fillna(False).astype(int)


def sig_M010(df_perp_1h, _, __):
    """Distribution topping on 1h proxy. short_only."""
    df = df_perp_1h.copy()
    atr = compute_atr(df, 14)
    high_30 = df["high"].rolling(30, min_periods=30).max().shift(1)
    vol_avg = df["volume"].rolling(30, min_periods=30).mean().shift(1)
    topping = (df["close"] < high_30 - 0.5 * atr) & (df["close"] < df["open"]) & (df["volume"] > vol_avg * 1.5)
    return -(topping.fillna(False).astype(int))


def sig_M011(df_perp_1h, _, __):
    """High-vol long dip: atr_q > 70% AND close < rolling_low_5d."""
    df = df_perp_1h.copy()
    atr = compute_atr(df, 14)
    atr_pct = atr / df["close"].shift(1) * 100
    atr_q = atr_pct.rolling(720).quantile(0.70).shift(1)
    is_high_vol = atr_pct.shift(1) > atr_q
    low_5d = df["low"].rolling(120, min_periods=120).min().shift(1)
    is_dip = df["close"].shift(1) < low_5d * 1.001
    return (is_high_vol & is_dip).fillna(False).astype(int)


def sig_M012(df_perp_1h, df_funding_8h, _):
    """Low-vol carry: atr_q < 30% AND |funding| > 0.005% → direction = -sign(funding)."""
    df = df_perp_1h.copy()
    atr = compute_atr(df, 14)
    atr_pct = atr / df["close"].shift(1) * 100
    atr_q = atr_pct.rolling(720).quantile(0.30).shift(1)
    is_low_vol = atr_pct.shift(1) < atr_q
    funding_1h = add_funding_to_1h_grid(df_funding_8h, df_perp_1h)
    funding_t1 = funding_1h.shift(1)
    high_funding = funding_t1.abs() > 0.00005  # 0.005%
    sig = pd.Series(0, index=df.index, dtype=int)
    is_active = is_low_vol & high_funding
    sig.loc[(is_active & (funding_t1 > 0)).fillna(False)] = -1  # short when positive funding
    sig.loc[(is_active & (funding_t1 < 0)).fillna(False)] = 1
    return sig


# ==================== Multi-Asset (1d) Mechanism Daily PnL ====================

def daily_pnl_M006(df_multi_1d, lookback=30, scenario="realistic"):
    """XS momentum 30d: long top-ranked coin daily."""
    fr = get_friction(scenario)
    rt_pct = fr.round_trip_pct / 100.0
    panel = df_multi_1d.copy()
    panel["day"] = pd.to_datetime(panel["t_open_ms"], unit="ms", utc=True).dt.floor("D")
    panel = panel.sort_values(["symbol", "day"]).reset_index(drop=True)
    panel["log_ret_1d"] = panel.groupby("symbol")["close"].transform(lambda x: np.log(x / x.shift(1)))
    panel["log_ret_lookback"] = panel.groupby("symbol")["close"].transform(lambda x: np.log(x / x.shift(lookback)))
    panel["log_ret_lookback_lag"] = panel.groupby("symbol")["log_ret_lookback"].shift(1)
    panel["rank"] = panel.groupby("day")["log_ret_lookback_lag"].rank(method="first", ascending=False)
    top = panel[panel["rank"] == 1].copy()
    top["pnl"] = top["log_ret_1d"] - rt_pct
    daily = top.set_index("day")["pnl"].sort_index()
    return daily.dropna()


def daily_pnl_M013(df_multi_1d, lookback=30, threshold_pct=5.0, scenario="realistic"):
    """ETH outperforms BTC by >5% over 30d → long ETH."""
    fr = get_friction(scenario)
    rt_pct = fr.round_trip_pct / 100.0
    panel = df_multi_1d.copy()
    panel["day"] = pd.to_datetime(panel["t_open_ms"], unit="ms", utc=True).dt.floor("D")
    panel = panel.sort_values(["symbol", "day"]).reset_index(drop=True)
    panel["log_ret_1d"] = panel.groupby("symbol")["close"].transform(lambda x: np.log(x / x.shift(1)))
    panel["log_ret_lookback"] = panel.groupby("symbol")["close"].transform(lambda x: np.log(x / x.shift(lookback)))
    panel["log_ret_lookback_lag"] = panel.groupby("symbol")["log_ret_lookback"].shift(1)
    eth = panel[panel["symbol"].str.startswith("ETH")][["day", "log_ret_lookback_lag", "log_ret_1d"]].rename(
        columns={"log_ret_lookback_lag": "eth_30d_lag", "log_ret_1d": "eth_1d"})
    btc_full = pd.read_parquet(DATA_DIR / "btc_perp_1d_1500d.parquet")
    btc_full = btc_full[btc_full["t_close_ms"] < T_SEAL_START_MS].copy()
    btc_full["day"] = pd.to_datetime(btc_full["t_open_ms"], unit="ms", utc=True).dt.floor("D")
    btc_full["btc_30d_lag"] = np.log(btc_full["close"] / btc_full["close"].shift(lookback)).shift(1)
    btc_d = btc_full[["day", "btc_30d_lag"]]
    merged = eth.merge(btc_d, on="day", how="left")
    merged["spread"] = (merged["eth_30d_lag"] - merged["btc_30d_lag"]) * 100
    merged["signal"] = (merged["spread"] > threshold_pct).astype(int)
    merged["pnl"] = np.where(merged["signal"] == 1, merged["eth_1d"] - rt_pct, 0.0)
    daily = merged.set_index("day")["pnl"].sort_index()
    return daily.dropna()


def daily_pnl_M014(df_multi_1d, lookback=30, dispersion_threshold=0.015, scenario="realistic"):
    """Low dispersion regime: equal-weight 5-coin basket long."""
    fr = get_friction(scenario)
    rt_pct = fr.round_trip_pct / 100.0
    panel = df_multi_1d.copy()
    panel["day"] = pd.to_datetime(panel["t_open_ms"], unit="ms", utc=True).dt.floor("D")
    panel = panel.sort_values(["symbol", "day"]).reset_index(drop=True)
    panel["log_ret_1d"] = panel.groupby("symbol")["close"].transform(lambda x: np.log(x / x.shift(1)))
    # Rolling 30d std of daily log returns per symbol
    panel["std_30d"] = panel.groupby("symbol")["log_ret_1d"].transform(
        lambda x: x.rolling(lookback).std().shift(1))
    # Per-day mean of per-symbol stds (lower = more cohesive market)
    daily_dispersion = panel.groupby("day")["std_30d"].mean()
    # Equal-weight basket return
    daily_basket = panel.groupby("day")["log_ret_1d"].mean()
    df_eval = pd.DataFrame({"dispersion": daily_dispersion, "basket_ret": daily_basket})
    df_eval["signal"] = (df_eval["dispersion"] < dispersion_threshold).astype(int)
    df_eval["pnl"] = np.where(df_eval["signal"] == 1, df_eval["basket_ret"] - rt_pct, 0.0)
    return df_eval["pnl"].sort_index().dropna()


# ==================== Spec Configuration ====================

@dataclass
class Spec:
    mid: str
    name: str
    family: str
    grid: str  # '1h', '1m_fwd', '1d_multi', '1h_2signal'
    forward_min: int  # in minutes (only for 1h-grid; ignored for daily)
    signal_fn: Optional[Callable] = None  # for 1h grid
    daily_pnl_fn: Optional[Callable] = None  # for multi-asset
    expected_freq_per_day: float = 1.0
    expected_status: str = "active"  # 'active' or 'control_null_expected_fail'


SPECS: List[Spec] = [
    Spec("M001", "force_flow_long_proxy", "reversion", "1h", 240, signal_fn=sig_M001, expected_freq_per_day=0.4),
    Spec("M002", "force_flow_short_proxy", "reversion", "1h", 60, signal_fn=sig_M002, expected_freq_per_day=0.5),
    Spec("M003", "cascade_window_5m_long", "reversion", "1m_fwd", 15, signal_fn=sig_M003, expected_freq_per_day=0.4),
    Spec("M004", "basis_fade_perp", "reversion", "1h", 240, signal_fn=sig_M004, expected_freq_per_day=1.5),
    Spec("M005", "funding_momentum_long", "momentum", "1h", 240, signal_fn=sig_M005, expected_freq_per_day=0.5),
    Spec("M006", "xs_momentum_30d", "momentum", "1d_multi", 0, daily_pnl_fn=daily_pnl_M006, expected_freq_per_day=1.0),
    Spec("M007", "channel_breakout_atr", "breakout", "1h", 240, signal_fn=sig_M007, expected_freq_per_day=2.0,
         expected_status="control_null_expected_fail"),
    Spec("M008", "compression_breakout", "breakout", "1h", 240, signal_fn=sig_M008, expected_freq_per_day=0.5,
         expected_status="control_null_expected_fail"),
    Spec("M009", "wyckoff_spring_proxy", "pattern", "1h", 1440, signal_fn=sig_M009, expected_freq_per_day=0.5),
    Spec("M010", "distribution_topping_proxy", "pattern", "1h", 1440, signal_fn=sig_M010, expected_freq_per_day=0.4),
    Spec("M011", "high_vol_long_dip", "regime", "1h", 1440, signal_fn=sig_M011, expected_freq_per_day=0.7),
    Spec("M012", "low_vol_carry", "regime", "1h", 480, signal_fn=sig_M012, expected_freq_per_day=0.5),
    Spec("M013", "eth_btc_outperform", "cross_section", "1d_multi", 0, daily_pnl_fn=daily_pnl_M013,
         expected_freq_per_day=1.0),
    Spec("M014", "dispersion_low_diversify", "cross_section", "1d_multi", 0, daily_pnl_fn=daily_pnl_M014,
         expected_freq_per_day=0.5),
]


# ==================== Evaluator ====================

def evaluate_spec_1h(spec: Spec, df_perp_1h, df_funding, df_spot_1h, df_perp_1m, scenario):
    """Returns daily_pnl Series + n_signals."""
    fr = get_friction(scenario)
    rt_pct = fr.round_trip_pct / 100.0
    # Generate signal
    if spec.mid == "M004":
        signal = spec.signal_fn(df_perp_1h, df_spot_1h, None)
    elif spec.mid in ("M007", "M008", "M009", "M010", "M011"):
        signal = spec.signal_fn(df_perp_1h, None, None)
    else:
        signal = spec.signal_fn(df_perp_1h, df_funding, None)
    signal_arr = signal.values
    is_active = (signal_arr != 0).astype(int)

    # Forward return
    if spec.grid == "1m_fwd":
        fwd_ret = fwd_return_1m_via_close(df_perp_1h, df_perp_1m, spec.forward_min)
    else:
        fwd_hours = max(1, spec.forward_min // 60)
        fwd_ret = fwd_return_1h(df_perp_1h, fwd_hours)

    bar_pnl = np.where(np.isnan(fwd_ret), 0.0, signal_arr * fwd_ret) - is_active * rt_pct
    daily = daily_aggregate(df_perp_1h, bar_pnl)
    return daily, int(is_active.sum())


def main():
    print("=== P3 Run — MAP-Elites Aggregate-Only (Option γ) ===")
    df_perp_1h, df_funding, df_spot_1h, df_perp_1m, df_perp_1d, df_multi = load_free_data()
    n_days = (df_perp_1h["t_open_ms"].max() - df_perp_1h["t_open_ms"].min()) / (86_400 * 1000)
    print(f"Free window: 1h={len(df_perp_1h)}, multi={len(df_multi)}, days={n_days:.1f}")
    baseline_daily = buy_and_hold_daily(df_perp_1h)
    print(f"Baseline B&H: {len(baseline_daily)} daily samples\n")

    results = {
        "timestamp_utc": "2026-05-02",
        "free_window_days": float(n_days),
        "baseline_n": len(baseline_daily),
        "frequency_scan": {},
        "evaluations": {},
    }

    # ---- Run 14 mechanisms × 2 scenarios ----
    for spec in SPECS:
        for scenario in ["realistic", "stress"]:
            key = f"{spec.mid}_{scenario}"
            try:
                if spec.grid == "1d_multi":
                    daily_pnl = spec.daily_pnl_fn(df_multi, scenario=scenario)
                    n_signals = int((daily_pnl != 0).sum())
                else:
                    daily_pnl, n_signals = evaluate_spec_1h(
                        spec, df_perp_1h, df_funding, df_spot_1h, df_perp_1m, scenario
                    )

                # Frequency check (only on first scenario to dedupe)
                if scenario == "realistic":
                    freq = n_signals / n_days
                    results["frequency_scan"][spec.mid] = {
                        "n_signals": n_signals,
                        "freq_per_day": round(float(freq), 3),
                        "vacuous": freq < 0.5,
                        "expected_status": spec.expected_status,
                    }
                vacuous = results["frequency_scan"].get(spec.mid, {}).get("vacuous", False)
                if vacuous:
                    results["evaluations"][key] = {
                        "status": "VACUOUS_FAIL",
                        "all_pass": False,
                        "n_signals": n_signals,
                        "freq_per_day": results["frequency_scan"][spec.mid]["freq_per_day"],
                    }
                    continue

                # Align to baseline grid
                if isinstance(daily_pnl, pd.Series):
                    daily_aligned = daily_pnl.reindex(baseline_daily.index, fill_value=0.0)
                else:
                    daily_aligned = pd.Series(daily_pnl).reindex(baseline_daily.index, fill_value=0.0)

                eval_result = bootstrap_six_criteria(
                    daily_pnl=daily_aligned,
                    baseline_pnl=baseline_daily,
                    priority="P3_AGGREGATED",
                    B=10000, block_size=3, seed=42,
                )
                eval_result["status"] = "PASS" if eval_result["all_pass"] else "FAIL"
                eval_result["n_signals"] = n_signals
                results["evaluations"][key] = eval_result
            except Exception as e:
                results["evaluations"][key] = {
                    "status": "ERROR",
                    "error": f"{type(e).__name__}: {str(e)[:300]}",
                    "all_pass": False,
                }
                print(f"  {key}: ERROR {type(e).__name__}: {e}")

        # Print per-mechanism summary
        r_real = results["evaluations"].get(f"{spec.mid}_realistic", {})
        r_stress = results["evaluations"].get(f"{spec.mid}_stress", {})
        if r_real.get("status") == "VACUOUS_FAIL":
            print(f"  {spec.mid} ({spec.name}): VACUOUS ({r_real.get('freq_per_day', 0):.3f}/d)")
        elif r_real.get("status") == "ERROR":
            print(f"  {spec.mid} ({spec.name}): ERROR")
        else:
            pe_r = r_real.get("point_estimates", {})
            cr_r = r_real.get("criteria", {})
            pe_s = r_stress.get("point_estimates", {})
            cr_s = r_stress.get("criteria", {})
            if pe_r:
                print(f"  {spec.mid} ({spec.name}): n={r_real.get('n_signals', '?')}")
                print(f"    realistic: mean {pe_r.get('mean', 0)*100:+.4f}%/d, "
                      f"p_beats {pe_r.get('p_beats_baseline', 0):.3f}, "
                      f"max_dd {pe_r.get('max_dd', 0)*100:+.2f}%, "
                      f"sharpe {pe_r.get('sharpe', 0):+.2f} → {sum(cr_r.values())}/6 ({r_real.get('status')})")
                print(f"    stress: mean {pe_s.get('mean', 0)*100:+.4f}%/d, "
                      f"p_beats {pe_s.get('p_beats_baseline', 0):.3f}, "
                      f"max_dd {pe_s.get('max_dd', 0)*100:+.2f}% → {sum(cr_s.values())}/6 ({r_stress.get('status')})")

    # Verdict
    pass_count = sum(1 for r in results["evaluations"].values() if r.get("all_pass"))
    n_evals = len(results["evaluations"])
    print(f"\n=== P3 Verdict ===")
    print(f"PASS: {pass_count}/{n_evals}")
    borderline = [
        (k, v) for k, v in results["evaluations"].items()
        if v.get("status") == "FAIL"
        and v.get("point_estimates", {}).get("mean", -1) > 0
        and v.get("point_estimates", {}).get("p_beats_baseline", 0) > 0.55
    ]
    print(f"Borderline candidates (mean>0 AND p_beats>0.55): {len(borderline)}")
    if borderline:
        for k, v in borderline:
            pe = v["point_estimates"]
            print(f"  {k}: mean={pe['mean']*100:+.4f}%/d, p_beats={pe['p_beats_baseline']:.3f}")

    out_path = EXP_DIR / "results_raw.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nSaved {out_path}")
    return results


if __name__ == "__main__":
    main()
