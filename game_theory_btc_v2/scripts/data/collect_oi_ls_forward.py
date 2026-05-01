"""Forward collector — OI / L-S ratio / taker volume ratio.

Runs every hour (cron / scheduled task / manual loop). Each run:
1. Fetch last 28d of OI history + L-S ratio + taker volume from Binance free
2. Dedupe-merge into local parquet (no overwrite of existing rows)
3. Append summary to `logs/forward_collector.log`

Goal: Accumulate 60-90d real OI/L-S history for Phase B hypothesis testing.

Idempotent: 매 1h 실행해도 dedupe되어 안전. 누락 hour 발생 시 다음 run에서 backfill.

Storage:
- `data/oi_forward.parquet` (BTC perp 1h OI)
- `data/ls_account_forward.parquet`
- `data/ls_position_forward.parquet`
- `data/taker_volume_forward.parquet`

Lookahead-aware: timestamps are observation/snapshot times. No transformation.

Run: python scripts/data/collect_oi_ls_forward.py
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import ccxt
import pandas as pd
import requests

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
LOGS_DIR = PROJECT_ROOT / "logs"
DATA_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)
LOG_FILE = LOGS_DIR / "forward_collector.log"

SYMBOL_BINANCE = "BTC/USDT:USDT"
SYMBOL_BINANCE_REST = "BTCUSDT"


def log(msg: str):
    ts = datetime.now(timezone.utc).isoformat()
    line = f"[{ts}] {msg}"
    print(line)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def merge_parquet(path: Path, new_df: pd.DataFrame, key: str = "timestamp_ms") -> int:
    """Merge new_df into existing parquet, dedupe on key. Returns rows added."""
    if new_df.empty:
        return 0
    if path.exists():
        existing = pd.read_parquet(path)
        before = len(existing)
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=[key]).sort_values(key).reset_index(drop=True)
        added = len(combined) - before
    else:
        combined = new_df.drop_duplicates(subset=[key]).sort_values(key).reset_index(drop=True)
        added = len(combined)
    combined.to_parquet(path, index=False)
    return added


def fetch_oi_via_ccxt(now_ms: int) -> pd.DataFrame:
    """Fetch last ~28d OI via CCXT (Binance USDM)."""
    ex = ccxt.binance({"enableRateLimit": True, "options": {"defaultType": "future"}})
    since_ms = now_ms - 28 * 86400 * 1000
    try:
        data = ex.fetch_open_interest_history(SYMBOL_BINANCE, "1h", since=since_ms, limit=30)
    except Exception as e:
        log(f"OI fetch error (1st batch): {e}")
        return pd.DataFrame()
    all_records = list(data) if data else []
    # Iterate forward by 30 records (~30h) until reaching now
    if all_records:
        cursor = all_records[-1]["timestamp"] + 3600 * 1000
        while cursor < now_ms:
            try:
                batch = ex.fetch_open_interest_history(SYMBOL_BINANCE, "1h", since=cursor, limit=30)
            except Exception as e:
                log(f"OI fetch error during pagination at cursor={cursor}: {e}")
                break
            if not batch:
                break
            all_records.extend(batch)
            new_cursor = batch[-1]["timestamp"] + 3600 * 1000
            if new_cursor <= cursor:
                break
            cursor = new_cursor
            time.sleep(0.4)
    if not all_records:
        return pd.DataFrame()
    rows = []
    for r in all_records:
        rows.append({
            "timestamp_ms": int(r["timestamp"]),
            "datetime_utc": r.get("datetime"),
            "open_interest_amount": float(r.get("openInterestAmount") or 0),
            "open_interest_value_usd": float(r.get("openInterestValue") or 0),
            "symbol": r.get("symbol"),
        })
    df = pd.DataFrame(rows).drop_duplicates(subset=["timestamp_ms"])
    return df


def fetch_ls_ratio(endpoint_path: str, now_ms: int) -> pd.DataFrame:
    """Fetch last 28d of L/S ratio or taker volume via Binance public REST."""
    url = f"https://fapi.binance.com{endpoint_path}"
    since_ms = now_ms - 28 * 86400 * 1000
    params = {"symbol": SYMBOL_BINANCE_REST, "period": "1h", "startTime": since_ms, "limit": 500}
    try:
        r = requests.get(url, params=params, timeout=15)
        j = r.json()
    except Exception as e:
        log(f"REST fetch error {endpoint_path}: {e}")
        return pd.DataFrame()
    if not isinstance(j, list) or not j:
        log(f"Empty/invalid {endpoint_path}: {str(j)[:200]}")
        return pd.DataFrame()
    df = pd.DataFrame(j)
    df["timestamp_ms"] = df["timestamp"].astype(int)
    return df


def main():
    now_ms = int(time.time() * 1000)
    log(f"=== Forward collector run start (utc={datetime.now(timezone.utc).isoformat()}) ===")
    summary = {"start_utc": datetime.now(timezone.utc).isoformat()}

    # 1. OI
    oi_df = fetch_oi_via_ccxt(now_ms)
    if not oi_df.empty:
        added = merge_parquet(DATA_DIR / "oi_forward.parquet", oi_df, key="timestamp_ms")
        log(f"OI: fetched {len(oi_df)} records, added {added} new")
        summary["oi_added"] = added
        summary["oi_total_after"] = len(pd.read_parquet(DATA_DIR / "oi_forward.parquet"))
    else:
        summary["oi_added"] = 0

    # 2. Top L/S Account Ratio
    df = fetch_ls_ratio("/futures/data/topLongShortAccountRatio", now_ms)
    if not df.empty:
        added = merge_parquet(DATA_DIR / "ls_account_forward.parquet", df, key="timestamp_ms")
        log(f"LS account: fetched {len(df)}, added {added}")
        summary["ls_account_added"] = added

    # 3. Top L/S Position Ratio
    df = fetch_ls_ratio("/futures/data/topLongShortPositionRatio", now_ms)
    if not df.empty:
        added = merge_parquet(DATA_DIR / "ls_position_forward.parquet", df, key="timestamp_ms")
        log(f"LS position: fetched {len(df)}, added {added}")
        summary["ls_position_added"] = added

    # 4. Global L/S Account Ratio
    df = fetch_ls_ratio("/futures/data/globalLongShortAccountRatio", now_ms)
    if not df.empty:
        added = merge_parquet(DATA_DIR / "ls_global_forward.parquet", df, key="timestamp_ms")
        log(f"LS global: fetched {len(df)}, added {added}")
        summary["ls_global_added"] = added

    # 5. Taker Long/Short Volume Ratio
    df = fetch_ls_ratio("/futures/data/takerlongshortRatio", now_ms)
    if not df.empty:
        added = merge_parquet(DATA_DIR / "taker_volume_forward.parquet", df, key="timestamp_ms")
        log(f"Taker volume: fetched {len(df)}, added {added}")
        summary["taker_volume_added"] = added

    summary["end_utc"] = datetime.now(timezone.utc).isoformat()
    log(f"Run summary: {json.dumps(summary)}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        log("Interrupted by user")
        sys.exit(0)
    except Exception as e:
        log(f"FATAL: {type(e).__name__}: {e}")
        raise
