"""Probe Binance OI history depth + liquidation endpoints (advisor critical #1, #3).

Tests:
1. OI history actual lookback (mandate assumes 720d, advisor warns 30d limit)
2. Funding history depth
3. forceOrders endpoint variants (auth requirement, market-wide vs user-only)

Output: console + JSON to experiments/p0/binance_depth_probe.json
"""
import json
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import ccxt
import requests


def fmt_ts(ms):
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).isoformat()


def probe_oi_depth(ex, symbol, timeframe):
    """Test how far back OI history goes by asking for 720d, 365d, 60d, 30d."""
    now_ms = int(time.time() * 1000)
    out = {}
    for label, days in [("720d", 720), ("365d", 365), ("60d", 60), ("30d", 30)]:
        since_ms = now_ms - days * 86400 * 1000
        try:
            data = ex.fetch_open_interest_history(symbol, timeframe, since=since_ms, limit=1500)
            if data:
                earliest_ts = data[0]["timestamp"]
                latest_ts = data[-1]["timestamp"]
                actual_age_d = (now_ms - earliest_ts) / 1000 / 86400
                out[label] = {
                    "asked_days": days,
                    "count": len(data),
                    "earliest_iso": fmt_ts(earliest_ts),
                    "latest_iso": fmt_ts(latest_ts),
                    "actual_age_days": round(actual_age_d, 1),
                }
            else:
                out[label] = {"asked_days": days, "count": 0}
        except Exception as e:
            out[label] = {"asked_days": days, "error": f"{type(e).__name__}: {e}"}
        time.sleep(0.5)
    return out


def probe_funding_depth(ex, symbol):
    """Test funding history lookback."""
    now_ms = int(time.time() * 1000)
    out = {}
    for label, days in [("720d", 720), ("365d", 365)]:
        since_ms = now_ms - days * 86400 * 1000
        try:
            data = ex.fetch_funding_rate_history(symbol, since=since_ms, limit=1000)
            if data:
                earliest_ts = data[0]["timestamp"]
                actual_age_d = (now_ms - earliest_ts) / 1000 / 86400
                out[label] = {
                    "asked_days": days,
                    "count": len(data),
                    "earliest_iso": fmt_ts(earliest_ts),
                    "actual_age_days": round(actual_age_d, 1),
                }
            else:
                out[label] = {"asked_days": days, "count": 0}
        except Exception as e:
            out[label] = {"asked_days": days, "error": f"{type(e).__name__}: {e}"}
        time.sleep(0.5)
    return out


def probe_force_orders():
    """Test Binance liquidation endpoint variants for free public access."""
    out = []
    # Symbol parameter 다양화
    candidates = [
        ("/fapi/v1/forceOrders unauth (USER-private?)",
         "https://fapi.binance.com/fapi/v1/forceOrders",
         {"symbol": "BTCUSDT", "limit": 10}),
        ("/fapi/v1/allForceOrders unauth",
         "https://fapi.binance.com/fapi/v1/allForceOrders",
         {"symbol": "BTCUSDT", "limit": 10}),
        ("/fapi/v1/allForceOrdersInfo",
         "https://fapi.binance.com/fapi/v1/allForceOrdersInfo",
         {"symbol": "BTCUSDT", "limit": 10}),
        ("/futures/data/forceOrderInfo",
         "https://fapi.binance.com/futures/data/forceOrderInfo",
         {"symbol": "BTCUSDT"}),
        # OI hist endpoint (mirrored as REST, may have different lookback)
        ("/futures/data/openInterestHist",
         "https://fapi.binance.com/futures/data/openInterestHist",
         {"symbol": "BTCUSDT", "period": "1h", "limit": 5}),
        # Long/short ratio (H7 dependency)
        ("/futures/data/topLongShortAccountRatio",
         "https://fapi.binance.com/futures/data/topLongShortAccountRatio",
         {"symbol": "BTCUSDT", "period": "1h", "limit": 5}),
        ("/futures/data/topLongShortPositionRatio",
         "https://fapi.binance.com/futures/data/topLongShortPositionRatio",
         {"symbol": "BTCUSDT", "period": "1h", "limit": 5}),
        ("/futures/data/takerlongshortRatio",
         "https://fapi.binance.com/futures/data/takerlongshortRatio",
         {"symbol": "BTCUSDT", "period": "1h", "limit": 5}),
    ]
    for name, url, params in candidates:
        try:
            r = requests.get(url, params=params, timeout=10)
            try:
                preview = json.dumps(r.json(), ensure_ascii=False)[:400]
            except Exception:
                preview = r.text[:400]
            out.append({"name": name, "status": r.status_code, "preview": preview})
        except Exception as e:
            out.append({"name": name, "error": f"{type(e).__name__}: {e}"})
        time.sleep(0.5)
    return out


def probe_oi_hist_via_rest():
    """Try Binance REST openInterestHist with various lookback to check actual depth."""
    out = []
    now_ms = int(time.time() * 1000)
    url = "https://fapi.binance.com/futures/data/openInterestHist"
    for label, days in [("720d", 720), ("365d", 365), ("90d", 90), ("30d", 30)]:
        start = now_ms - days * 86400 * 1000
        params = {
            "symbol": "BTCUSDT", "period": "1h",
            "startTime": start, "endTime": now_ms,
            "limit": 500,
        }
        try:
            r = requests.get(url, params=params, timeout=10)
            j = r.json()
            if isinstance(j, list) and j:
                earliest = int(j[0]["timestamp"])
                age_d = (now_ms - earliest) / 1000 / 86400
                out.append({
                    "label": label, "asked_days": days,
                    "status": r.status_code, "count": len(j),
                    "earliest_iso": fmt_ts(earliest),
                    "actual_age_days": round(age_d, 1),
                })
            else:
                out.append({
                    "label": label, "asked_days": days,
                    "status": r.status_code,
                    "preview": json.dumps(j, ensure_ascii=False)[:200],
                })
        except Exception as e:
            out.append({"label": label, "error": f"{type(e).__name__}: {e}"})
        time.sleep(0.5)
    return out


def main():
    ts = datetime.now(timezone.utc).isoformat()
    print(f"=== Binance Depth Probe ({ts}) ===\n")

    ex = ccxt.binance({"enableRateLimit": True, "options": {"defaultType": "future"}})
    ex.load_markets()
    sym = "BTC/USDT:USDT"

    print("--- 1. OI history depth via CCXT ---")
    oi_ccxt = probe_oi_depth(ex, sym, "1h")
    for k, v in oi_ccxt.items():
        print(f"  {k}: {v}")

    print("\n--- 2. OI history depth via raw REST /futures/data/openInterestHist ---")
    oi_rest = probe_oi_hist_via_rest()
    for entry in oi_rest:
        print(f"  {entry}")

    print("\n--- 3. Funding history depth via CCXT ---")
    fund = probe_funding_depth(ex, sym)
    for k, v in fund.items():
        print(f"  {k}: {v}")

    print("\n--- 4. Liquidation / L/S ratio endpoints (unauthenticated) ---")
    fo = probe_force_orders()
    for entry in fo:
        print(f"  [{entry.get('name')}]")
        if "error" in entry:
            print(f"    ERROR: {entry['error']}")
        else:
            print(f"    status={entry['status']}, preview={entry.get('preview', '')[:200]}")

    summary = {
        "timestamp_utc": ts,
        "oi_ccxt": oi_ccxt,
        "oi_rest": oi_rest,
        "funding": fund,
        "force_orders": fo,
    }
    out_path = Path(__file__).resolve().parents[2] / "experiments" / "p0" / "binance_depth_probe.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
