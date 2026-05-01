"""P0.2 Quality Audit — verify all fetched data + compute holdout boundary.

Outputs:
- console summary
- experiments/p0/audit_results.json
- T_anchor + T_seal_start computed for holdout_seal.md amendment
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
EXP_DIR = Path(__file__).resolve().parents[2] / "experiments" / "p0"

FILES = {
    "btc_perp_1d_1500d.parquet": ("ohlcv", "1d", 86_400_000),
    "btc_perp_1h_720d.parquet": ("ohlcv", "1h", 3_600_000),
    "btc_perp_5m_720d.parquet": ("ohlcv", "5m", 300_000),
    "btc_perp_1m_365d.parquet": ("ohlcv", "1m", 60_000),
    "btc_spot_1h_720d.parquet": ("ohlcv", "1h", 3_600_000),
    "btc_funding_binance_720d.parquet": ("funding", None, None),
    "btc_funding_bybit_620d.parquet": ("funding", None, None),
    "multi_asset_1d_800d.parquet": ("ohlcv_multi", "1d", 86_400_000),
    "oi_forward.parquet": ("forward_oi", None, 3_600_000),
    "ls_account_forward.parquet": ("forward_ls", None, 3_600_000),
    "ls_position_forward.parquet": ("forward_ls", None, 3_600_000),
    "ls_global_forward.parquet": ("forward_ls", None, 3_600_000),
    "taker_volume_forward.parquet": ("forward_ls", None, 3_600_000),
}


def fmt_ts(ms):
    return pd.to_datetime(ms, unit="ms", utc=True).isoformat()


def audit_ohlcv(df: pd.DataFrame, interval_ms: int):
    res = {
        "rows": len(df),
        "span_open_min": fmt_ts(df["t_open_ms"].min()),
        "span_open_max": fmt_ts(df["t_open_ms"].max()),
        "span_close_max": fmt_ts(df["t_close_ms"].max()),
        "nan_count_ohlcv": int(df[["open", "high", "low", "close", "volume"]].isna().sum().sum()),
    }
    sdf = df.sort_values("t_open_ms")
    diffs = sdf["t_open_ms"].diff().dropna()
    if len(diffs) > 0:
        gap_count = int((diffs != interval_ms).sum())
        res["gap_count"] = gap_count
        res["gap_rate"] = round(gap_count / len(diffs), 4)
        # gap distribution
        unique_diffs = diffs.value_counts().head(5).to_dict()
        res["top_diffs_ms"] = {str(k): int(v) for k, v in unique_diffs.items()}
    # OHLC invariant
    invalid = ((df["high"] < df["low"])
               | (df["high"] < df["open"]) | (df["high"] < df["close"])
               | (df["low"] > df["open"]) | (df["low"] > df["close"])).sum()
    res["ohlc_invalid"] = int(invalid)
    # t_close - t_open assertion
    diff_close_open = (df["t_close_ms"] - df["t_open_ms"]).unique()
    res["t_close_minus_open_unique"] = list(map(int, diff_close_open))
    res["t_close_assertion_ok"] = len(diff_close_open) == 1 and int(diff_close_open[0]) == interval_ms
    # monotonic
    res["t_open_monotonic"] = bool(df["t_open_ms"].is_monotonic_increasing)
    res["t_open_unique"] = bool(df["t_open_ms"].nunique() == len(df))
    return res


def audit_funding(df: pd.DataFrame):
    res = {"rows": len(df)}
    if "timestamp_ms" not in df.columns:
        res["error"] = "no timestamp_ms column"
        return res
    res["span_min"] = fmt_ts(df["timestamp_ms"].min())
    res["span_max"] = fmt_ts(df["timestamp_ms"].max())
    res["fr_min"] = float(df["funding_rate"].min())
    res["fr_max"] = float(df["funding_rate"].max())
    res["fr_mean"] = float(df["funding_rate"].mean())
    res["fr_std"] = float(df["funding_rate"].std())
    sdf = df.sort_values("timestamp_ms")
    diffs = sdf["timestamp_ms"].diff().dropna()
    if len(diffs) > 0:
        # 8h native = 28800000 ms
        std_8h = 8 * 3600 * 1000
        gap_count = int((diffs != std_8h).sum())
        res["gap_count_vs_8h"] = gap_count
        res["gap_rate_vs_8h"] = round(gap_count / len(diffs), 4)
        unique_diffs = diffs.value_counts().head(5).to_dict()
        res["top_diffs_ms"] = {str(k): int(v) for k, v in unique_diffs.items()}
    return res


def audit_forward(df: pd.DataFrame):
    res = {"rows": len(df)}
    if "timestamp_ms" not in df.columns:
        res["error"] = "no timestamp_ms"
        return res
    res["span_min"] = fmt_ts(df["timestamp_ms"].min())
    res["span_max"] = fmt_ts(df["timestamp_ms"].max())
    actual_age_d = (df["timestamp_ms"].max() - df["timestamp_ms"].min()) / 1000 / 86400
    res["span_days"] = round(actual_age_d, 2)
    return res


def audit_ohlcv_multi(df: pd.DataFrame, interval_ms: int):
    res = {"total_rows": len(df), "symbols_in_file": list(df["symbol"].unique())}
    per_sym = {}
    for sym, sdf in df.groupby("symbol"):
        per_sym[sym] = audit_ohlcv(sdf, interval_ms)
    res["per_symbol"] = per_sym
    return res


def main():
    audit = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "files": {}}
    for fname, (kind, tf, interval_ms) in FILES.items():
        fp = DATA_DIR / fname
        if not fp.exists():
            audit["files"][fname] = {"status": "MISSING"}
            continue
        df = pd.read_parquet(fp)
        size_bytes = fp.stat().st_size
        if kind == "ohlcv":
            entry = {"size_bytes": size_bytes, **audit_ohlcv(df, interval_ms)}
        elif kind == "ohlcv_multi":
            entry = {"size_bytes": size_bytes, **audit_ohlcv_multi(df, interval_ms)}
        elif kind == "funding":
            entry = {"size_bytes": size_bytes, **audit_funding(df)}
        elif kind in ("forward_oi", "forward_ls"):
            entry = {"size_bytes": size_bytes, **audit_forward(df)}
        else:
            entry = {"size_bytes": size_bytes, "rows": len(df)}
        audit["files"][fname] = entry

    # Compute T_anchor + holdout boundary from primary perp 1h
    perp1h = pd.read_parquet(DATA_DIR / "btc_perp_1h_720d.parquet")
    T_anchor_ms = int(perp1h["t_close_ms"].max())
    T_seal_start_ms = T_anchor_ms - 180 * 86_400_000
    T_p3_holdout_start_ms = T_anchor_ms - 72 * 86_400_000  # P3 extra 10%

    audit["holdout"] = {
        "T_anchor_ms": T_anchor_ms,
        "T_anchor_iso": fmt_ts(T_anchor_ms),
        "T_seal_start_ms": T_seal_start_ms,
        "T_seal_start_iso": fmt_ts(T_seal_start_ms),
        "T_p3_holdout_start_ms": T_p3_holdout_start_ms,
        "T_p3_holdout_start_iso": fmt_ts(T_p3_holdout_start_ms),
        "sealed_window_days": 180,
        "p3_extra_holdout_days": 72,
        "rule": "Anchor = max(t_close_ms) of btc_perp_1h_720d.parquet at P0.2 closure. Immutable post-fetch.",
    }

    # Sealed slice counts per file
    sealed_counts = {}
    for fname in ["btc_perp_1h_720d.parquet", "btc_perp_5m_720d.parquet",
                  "btc_perp_1m_365d.parquet", "btc_spot_1h_720d.parquet",
                  "btc_perp_1d_1500d.parquet"]:
        fp = DATA_DIR / fname
        if not fp.exists():
            continue
        df = pd.read_parquet(fp, columns=["t_open_ms", "t_close_ms"])
        sealed = df[df["t_close_ms"] >= T_seal_start_ms]
        sealed_counts[fname] = {"total": len(df), "sealed_rows": len(sealed),
                                "sealed_pct": round(len(sealed) / len(df) * 100, 2) if len(df) else 0}
    audit["holdout"]["sealed_per_file"] = sealed_counts

    # Save
    out_path = EXP_DIR / "audit_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(audit, f, indent=2, ensure_ascii=False, default=str)

    # Console summary
    print("=== Audit Summary ===")
    for fname, info in audit["files"].items():
        if info.get("status") == "MISSING":
            print(f"  {fname}: MISSING")
            continue
        if "rows" in info:
            extra = ""
            if "gap_rate" in info:
                extra += f" gap={info['gap_rate']}"
            if "ohlc_invalid" in info:
                extra += f" ohlc_invalid={info['ohlc_invalid']}"
            if "t_close_assertion_ok" in info:
                extra += f" assertion={info['t_close_assertion_ok']}"
            if "fr_min" in info:
                extra += f" fr=[{info['fr_min']:.5f}, {info['fr_max']:.5f}]"
            if "span_open_min" in info:
                extra += f" span={info['span_open_min']}~{info['span_open_max']}"
            elif "span_min" in info:
                extra += f" span={info['span_min']}~{info['span_max']}"
            if "span_days" in info:
                extra += f" days={info['span_days']}"
            if "total_rows" in info:
                extra += f" syms={len(info.get('symbols_in_file', []))}"
            print(f"  {fname}: rows={info.get('rows', info.get('total_rows', 'n/a'))}, size={info['size_bytes']}b{extra}")
    print(f"\n=== Holdout ===")
    print(f"  T_anchor: {audit['holdout']['T_anchor_iso']}")
    print(f"  T_seal_start: {audit['holdout']['T_seal_start_iso']}")
    print(f"  T_p3_holdout_start: {audit['holdout']['T_p3_holdout_start_iso']}")
    print(f"  Sealed slice counts:")
    for fname, info in audit["holdout"]["sealed_per_file"].items():
        print(f"    {fname}: {info['sealed_rows']}/{info['total']} = {info['sealed_pct']}%")
    print(f"\nSaved {out_path}")


if __name__ == "__main__":
    main()
