"""P0.2 Main Fetcher — Binance OHLCV (perp + spot) + funding + multi-asset.

Fetches 720d (or 1500d for daily, 90d for 1m) of public market data, dedupe-merging
into local parquet. Resumable from disk, incremental writes.

Built-in assertions enforce lookahead-free timestamp policy:
- All OHLCV stored with t_open_ms (open time, ms UTC) and t_close_ms (open + interval).
- Feature observation time MUST be t_close_ms in downstream code.
- Validator unit test: artificial leak series → assertion catches.

Storage layout (under data/):
- btc_perp_1d_1500d.parquet
- btc_perp_1h_720d.parquet
- btc_perp_5m_720d.parquet
- btc_perp_1m_90d.parquet
- btc_spot_1h_720d.parquet
- btc_funding_binance_720d.parquet
- btc_funding_bybit_620d.parquet  (cross-check)
- multi_asset_1d_800d.parquet     (ETH/SOL/BNB/XRP/DOGE)

Run: python scripts/data/fetch_binance.py [--symbol BTC] [--only PERP_1H,SPOT_1H]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import ccxt
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"
DATA_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)
LOG_FILE = LOGS_DIR / "fetch_binance.log"

# Fetch specifications. Order matters for prioritization (run smallest first to validate).
FETCH_SPECS = {
    # daily (smallest, smoke test)
    "BTC_PERP_1D_1500D": {
        "kind": "ohlcv_perp", "symbol": "BTC/USDT:USDT",
        "timeframe": "1d", "lookback_days": 1500,
        "filename": "btc_perp_1d_1500d.parquet",
    },
    "BTC_SPOT_1H_720D": {
        "kind": "ohlcv_spot", "symbol": "BTC/USDT",
        "timeframe": "1h", "lookback_days": 720,
        "filename": "btc_spot_1h_720d.parquet",
    },
    "BTC_PERP_1H_720D": {
        "kind": "ohlcv_perp", "symbol": "BTC/USDT:USDT",
        "timeframe": "1h", "lookback_days": 720,
        "filename": "btc_perp_1h_720d.parquet",
    },
    # funding
    "BTC_FUNDING_BINANCE_720D": {
        "kind": "funding", "exchange": "binance", "symbol": "BTC/USDT:USDT",
        "lookback_days": 720,
        "filename": "btc_funding_binance_720d.parquet",
    },
    "BTC_FUNDING_BYBIT_620D": {
        "kind": "funding", "exchange": "bybit", "symbol": "BTC/USDT:USDT",
        "lookback_days": 620,
        "filename": "btc_funding_bybit_620d.parquet",
    },
    # multi-asset daily
    "MULTI_ASSET_1D_800D": {
        "kind": "multi_asset",
        "symbols": ["ETH/USDT:USDT", "SOL/USDT:USDT", "BNB/USDT:USDT",
                    "XRP/USDT:USDT", "DOGE/USDT:USDT"],
        "timeframe": "1d", "lookback_days": 800,
        "filename": "multi_asset_1d_800d.parquet",
    },
    # 5m large
    "BTC_PERP_5M_720D": {
        "kind": "ohlcv_perp", "symbol": "BTC/USDT:USDT",
        "timeframe": "5m", "lookback_days": 720,
        "filename": "btc_perp_5m_720d.parquet",
    },
    # 1m extended window (365d so first 185d is outside sealed-180d holdout)
    "BTC_PERP_1M_365D": {
        "kind": "ohlcv_perp", "symbol": "BTC/USDT:USDT",
        "timeframe": "1m", "lookback_days": 365,
        "filename": "btc_perp_1m_365d.parquet",
    },
}

TIMEFRAME_MS = {"1m": 60_000, "5m": 300_000, "15m": 900_000, "1h": 3_600_000,
                "4h": 14_400_000, "1d": 86_400_000}


def log(msg: str, level: str = "INFO"):
    ts = datetime.now(timezone.utc).isoformat()
    line = f"[{ts}][{level}] {msg}"
    print(line)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def make_exchange(exchange_name: str = "binance", market_type: str = "future") -> ccxt.Exchange:
    if exchange_name == "binance":
        opts = {"defaultType": "future"} if market_type == "future" else {}
        return ccxt.binance({"enableRateLimit": True, "options": opts})
    if exchange_name == "bybit":
        return ccxt.bybit({"enableRateLimit": True})
    if exchange_name == "okx":
        return ccxt.okx({"enableRateLimit": True})
    raise ValueError(f"Unknown exchange: {exchange_name}")


def assert_ohlcv_invariants(df: pd.DataFrame, timeframe: str, label: str):
    """Lookahead-free timestamp assertions."""
    if df.empty:
        return
    interval_ms = TIMEFRAME_MS[timeframe]
    # required columns
    required = {"t_open_ms", "t_close_ms", "open", "high", "low", "close", "volume"}
    missing = required - set(df.columns)
    assert not missing, f"[{label}] missing columns: {missing}"
    # t_close = t_open + interval
    diff = (df["t_close_ms"] - df["t_open_ms"]).unique()
    assert len(diff) == 1 and diff[0] == interval_ms, \
        f"[{label}] t_close - t_open != {interval_ms}; got unique diffs: {diff[:5]}"
    # monotonic + dedupe
    assert df["t_open_ms"].is_monotonic_increasing, f"[{label}] t_open not monotonic"
    assert df["t_open_ms"].nunique() == len(df), f"[{label}] duplicate t_open"
    # gap check (allow some gaps for downtime, but flag if >5%)
    expected_diffs = df["t_open_ms"].diff().dropna()
    expected_count_per_diff = (expected_diffs == interval_ms).sum()
    gap_rate = 1 - expected_count_per_diff / len(expected_diffs) if len(expected_diffs) else 0
    if gap_rate > 0.05:
        log(f"[{label}] gap rate {gap_rate:.2%} exceeds 5%", "WARN")


def fetch_ohlcv_paginated(ex: ccxt.Exchange, symbol: str, timeframe: str,
                          since_ms: int, until_ms: int, batch_limit: int = 1500) -> pd.DataFrame:
    """Paginated OHLCV fetch with retries + incremental progress."""
    interval_ms = TIMEFRAME_MS[timeframe]
    all_rows = []
    cursor = since_ms
    n_calls = 0
    while cursor < until_ms:
        for attempt in range(3):
            try:
                batch = ex.fetch_ohlcv(symbol, timeframe, since=cursor, limit=batch_limit)
                break
            except ccxt.NetworkError as e:
                wait = 2 ** attempt
                log(f"  network error (attempt {attempt+1}, wait {wait}s): {e}", "WARN")
                time.sleep(wait)
            except ccxt.BadRequest as e:
                # often "Invalid limit" or similar — try smaller limit
                if batch_limit > 100 and attempt == 0:
                    batch_limit = 500
                    log(f"  reducing batch_limit to {batch_limit}", "WARN")
                    continue
                log(f"  bad request: {e}", "ERROR")
                raise
        else:
            raise RuntimeError(f"  failed after 3 retries at cursor={cursor}")
        n_calls += 1
        if not batch:
            break
        all_rows.extend(batch)
        new_cursor = batch[-1][0] + interval_ms
        if new_cursor <= cursor:  # no progress
            break
        cursor = new_cursor
        if len(batch) < batch_limit:
            # last partial batch
            if cursor >= until_ms:
                break
        time.sleep(0.3)  # rate limit cushion
    log(f"  {symbol} {timeframe}: {n_calls} API calls, {len(all_rows)} raw rows")
    if not all_rows:
        return pd.DataFrame()
    df = pd.DataFrame(all_rows, columns=["t_open_ms", "open", "high", "low", "close", "volume"])
    df["t_open_ms"] = df["t_open_ms"].astype("int64")
    df["t_close_ms"] = df["t_open_ms"] + interval_ms
    df = df.drop_duplicates(subset=["t_open_ms"]).sort_values("t_open_ms").reset_index(drop=True)
    df = df[(df["t_open_ms"] >= since_ms) & (df["t_open_ms"] < until_ms)].reset_index(drop=True)
    return df


def merge_parquet(path: Path, new_df: pd.DataFrame, key: str = "t_open_ms") -> tuple[int, int]:
    """Dedupe-merge into existing parquet. Returns (added, total_after)."""
    if new_df.empty:
        total = len(pd.read_parquet(path)) if path.exists() else 0
        return 0, total
    if path.exists():
        existing = pd.read_parquet(path)
        before = len(existing)
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=[key]).sort_values(key).reset_index(drop=True)
    else:
        combined = new_df.drop_duplicates(subset=[key]).sort_values(key).reset_index(drop=True)
        before = 0
    combined.to_parquet(path, index=False)
    return len(combined) - before, len(combined)


def fetch_funding_paginated(ex: ccxt.Exchange, symbol: str,
                            since_ms: int, until_ms: int) -> pd.DataFrame:
    """Funding rate history pagination."""
    all_rows = []
    cursor = since_ms
    n_calls = 0
    while cursor < until_ms:
        for attempt in range(3):
            try:
                batch = ex.fetch_funding_rate_history(symbol, since=cursor, limit=1000)
                break
            except ccxt.NetworkError as e:
                time.sleep(2 ** attempt)
        else:
            raise RuntimeError(f"funding fetch failed at cursor={cursor}")
        n_calls += 1
        if not batch:
            break
        all_rows.extend(batch)
        last_ts = int(batch[-1]["timestamp"])
        if last_ts <= cursor:
            break
        cursor = last_ts + 1
        time.sleep(0.3)
    log(f"  {symbol} funding: {n_calls} calls, {len(all_rows)} records")
    if not all_rows:
        return pd.DataFrame()
    rows = []
    for r in all_rows:
        rows.append({
            "timestamp_ms": int(r["timestamp"]),
            "datetime_utc": r.get("datetime"),
            "symbol": r.get("symbol"),
            "funding_rate": float(r.get("fundingRate") or 0),
        })
    df = pd.DataFrame(rows).drop_duplicates(subset=["timestamp_ms"])
    df = df.sort_values("timestamp_ms").reset_index(drop=True)
    return df


def resolve_resume_since(path: Path, default_since_ms: int, key: str = "t_open_ms") -> int:
    """If parquet exists, resume from max(key) + 1; else use default."""
    if path.exists():
        try:
            df = pd.read_parquet(path, columns=[key])
            if len(df) > 0:
                last = int(df[key].max())
                resumed = last + 1
                log(f"  resume from {datetime.fromtimestamp(resumed/1000, tz=timezone.utc).isoformat()} (existing rows: {len(df)})")
                return resumed
        except Exception as e:
            log(f"  resume failed: {e}, using default", "WARN")
    return default_since_ms


def run_spec(spec_name: str, spec: dict, now_ms: int):
    log(f"=== {spec_name} ===")
    out_path = DATA_DIR / spec["filename"]
    lookback_days = spec["lookback_days"]
    target_since = now_ms - lookback_days * 86400 * 1000

    if spec["kind"] in ("ohlcv_perp", "ohlcv_spot"):
        market_type = "future" if spec["kind"] == "ohlcv_perp" else "spot"
        ex = make_exchange("binance", market_type=market_type)
        symbol = spec["symbol"]
        timeframe = spec["timeframe"]
        since_ms = resolve_resume_since(out_path, target_since, "t_open_ms")
        if since_ms >= now_ms:
            log(f"  already up-to-date, skipping")
            return
        new_df = fetch_ohlcv_paginated(ex, symbol, timeframe, since_ms, now_ms)
        if not new_df.empty:
            assert_ohlcv_invariants(new_df, timeframe, spec_name)
        added, total = merge_parquet(out_path, new_df, key="t_open_ms")
        log(f"  added {added} rows, total {total}")
    elif spec["kind"] == "funding":
        ex = make_exchange(spec["exchange"], market_type="future")
        ex.load_markets()
        symbol = spec["symbol"]
        since_ms = resolve_resume_since(out_path, target_since, "timestamp_ms")
        if since_ms >= now_ms:
            log(f"  already up-to-date, skipping")
            return
        new_df = fetch_funding_paginated(ex, symbol, since_ms, now_ms)
        added, total = merge_parquet(out_path, new_df, key="timestamp_ms")
        log(f"  added {added} rows, total {total}")
    elif spec["kind"] == "multi_asset":
        ex = make_exchange("binance", market_type="future")
        timeframe = spec["timeframe"]
        all_dfs = []
        for sym in spec["symbols"]:
            log(f"  -- {sym} --")
            tmp_path = DATA_DIR / f"_multi_asset_tmp_{sym.replace('/', '_').replace(':', '_')}.parquet"
            since_ms = resolve_resume_since(tmp_path, target_since, "t_open_ms")
            if since_ms >= now_ms:
                df = pd.read_parquet(tmp_path)
            else:
                df = fetch_ohlcv_paginated(ex, sym, timeframe, since_ms, now_ms)
                if not df.empty:
                    assert_ohlcv_invariants(df, timeframe, f"{spec_name}_{sym}")
                    df["symbol"] = sym
                    if tmp_path.exists():
                        prev = pd.read_parquet(tmp_path)
                        df = pd.concat([prev, df], ignore_index=True).drop_duplicates(
                            subset=["t_open_ms", "symbol"]).sort_values("t_open_ms").reset_index(drop=True)
                    df.to_parquet(tmp_path, index=False)
            df["symbol"] = sym
            all_dfs.append(df)
        if all_dfs:
            merged = pd.concat(all_dfs, ignore_index=True)
            merged = merged.drop_duplicates(subset=["t_open_ms", "symbol"]).reset_index(drop=True)
            merged.to_parquet(out_path, index=False)
            log(f"  multi-asset combined: {len(merged)} rows across {len(spec['symbols'])} symbols")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", default=None,
                        help="Comma-separated spec names to run (default: all)")
    parser.add_argument("--list", action="store_true", help="List available specs")
    args = parser.parse_args()
    if args.list:
        for k, v in FETCH_SPECS.items():
            print(f"  {k}: kind={v['kind']}, file={v['filename']}, lookback={v['lookback_days']}d")
        return
    selected = list(FETCH_SPECS.keys())
    if args.only:
        wanted = [s.strip() for s in args.only.split(",")]
        unknown = set(wanted) - set(FETCH_SPECS.keys())
        assert not unknown, f"Unknown specs: {unknown}"
        selected = wanted
    now_ms = int(time.time() * 1000)
    log(f"=== Main fetcher start (utc={datetime.now(timezone.utc).isoformat()}, specs={selected}) ===")
    summary = {}
    for name in selected:
        try:
            run_spec(name, FETCH_SPECS[name], now_ms)
            summary[name] = "ok"
        except Exception as e:
            log(f"FAILED {name}: {type(e).__name__}: {e}", "ERROR")
            summary[name] = f"FAIL: {type(e).__name__}: {str(e)[:200]}"
    log(f"=== Main fetcher end. summary={json.dumps(summary)} ===")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("Interrupted")
        sys.exit(0)
