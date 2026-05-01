"""Probe alternative OI sources after confirming Binance OI 30d limit (advisor #1 critical).

Tests:
1. Binance OI hard depth confirmation (proper limits)
2. Bybit OI history depth (alternative)
3. OKX OI history depth (alternative)
4. Binance public L/S ratio depth (free, partial substitute for OI delta signal)
5. Binance public taker long/short volume ratio depth

If Bybit/OKX has 720d OI → use as cross-exchange reference for H1/H6.
If only Binance L/S ratio 30d → re-design H1/H6 with shorter window or alt features.
"""
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import ccxt
import requests


def fmt_age(now_ms, ts_ms):
    return round((now_ms - ts_ms) / 1000 / 86400, 1)


def probe_binance_oi(now_ms):
    """Confirm Binance OI 30d limit with proper params (limit=30 max)."""
    ex = ccxt.binance({"enableRateLimit": True, "options": {"defaultType": "future"}})
    out = {}
    for days in [25, 28, 31, 33, 60, 180]:
        since_ms = now_ms - days * 86400 * 1000
        try:
            data = ex.fetch_open_interest_history("BTC/USDT:USDT", "1h", since=since_ms, limit=30)
            if data:
                age_d = fmt_age(now_ms, data[0]["timestamp"])
                out[f"{days}d_back"] = {"count": len(data), "earliest_age_days": age_d, "ok": True}
            else:
                out[f"{days}d_back"] = {"count": 0, "ok": False}
        except Exception as e:
            out[f"{days}d_back"] = {"error": f"{type(e).__name__}: {str(e)[:200]}"}
        time.sleep(0.4)
    return out


def probe_bybit_oi(now_ms):
    """Bybit OI history (potential 720d alternative)."""
    out = {}
    try:
        ex = ccxt.bybit({"enableRateLimit": True})
        ex.load_markets()
    except Exception as e:
        return {"init_error": f"{type(e).__name__}: {str(e)[:200]}"}
    # Multiple symbol formats for Bybit
    symbols_to_try = ["BTC/USDT:USDT", "BTCUSDT"]
    chosen = None
    for s in symbols_to_try:
        if s in ex.markets:
            chosen = s
            break
    out["symbol_chosen"] = chosen
    if not chosen:
        return out
    for days in [720, 365, 180, 90, 30]:
        since_ms = now_ms - days * 86400 * 1000
        try:
            data = ex.fetch_open_interest_history(chosen, "1h", since=since_ms, limit=200)
            if data:
                age_d = fmt_age(now_ms, data[0]["timestamp"])
                out[f"{days}d_back"] = {"count": len(data), "earliest_age_days": age_d, "ok": True}
            else:
                out[f"{days}d_back"] = {"count": 0, "ok": False}
        except Exception as e:
            out[f"{days}d_back"] = {"error": f"{type(e).__name__}: {str(e)[:200]}"}
        time.sleep(0.4)
    return out


def probe_okx_oi(now_ms):
    """OKX OI history."""
    out = {}
    try:
        ex = ccxt.okx({"enableRateLimit": True})
        ex.load_markets()
    except Exception as e:
        return {"init_error": f"{type(e).__name__}: {str(e)[:200]}"}
    for sym in ["BTC-USDT-SWAP", "BTC/USDT:USDT"]:
        if sym in ex.markets:
            chosen = sym
            break
    else:
        return {"no_btc_swap_symbol": True}
    out["symbol_chosen"] = chosen
    for days in [720, 365, 90, 30, 7]:
        since_ms = now_ms - days * 86400 * 1000
        try:
            data = ex.fetch_open_interest_history(chosen, "1H", since=since_ms, limit=100)
            if data:
                age_d = fmt_age(now_ms, data[0]["timestamp"])
                out[f"{days}d_back"] = {"count": len(data), "earliest_age_days": age_d, "ok": True}
            else:
                out[f"{days}d_back"] = {"count": 0, "ok": False}
        except Exception as e:
            out[f"{days}d_back"] = {"error": f"{type(e).__name__}: {str(e)[:200]}"}
        time.sleep(0.4)
    return out


def probe_ls_ratio_depth(now_ms):
    """Binance public L/S ratio + taker volume ratio depth."""
    out = {}
    endpoints = [
        ("topLongShortAccountRatio", "https://fapi.binance.com/futures/data/topLongShortAccountRatio"),
        ("topLongShortPositionRatio", "https://fapi.binance.com/futures/data/topLongShortPositionRatio"),
        ("globalLongShortAccountRatio", "https://fapi.binance.com/futures/data/globalLongShortAccountRatio"),
        ("takerlongshortRatio", "https://fapi.binance.com/futures/data/takerlongshortRatio"),
    ]
    for name, url in endpoints:
        out[name] = {}
        for days in [60, 31, 30, 28]:
            start = now_ms - days * 86400 * 1000
            try:
                params = {"symbol": "BTCUSDT", "period": "1h", "startTime": start, "limit": 500}
                r = requests.get(url, params=params, timeout=10)
                j = r.json()
                if isinstance(j, list) and j:
                    age_d = fmt_age(now_ms, int(j[0]["timestamp"]))
                    out[name][f"{days}d"] = {"status": r.status_code, "count": len(j), "earliest_age_days": age_d}
                else:
                    out[name][f"{days}d"] = {"status": r.status_code, "preview": json.dumps(j)[:150]}
            except Exception as e:
                out[name][f"{days}d"] = {"error": f"{type(e).__name__}: {str(e)[:200]}"}
            time.sleep(0.4)
    return out


def main():
    ts = datetime.now(timezone.utc).isoformat()
    now_ms = int(time.time() * 1000)
    print(f"=== OI Alternatives Probe ({ts}) ===\n")

    print("--- 1. Binance OI hard depth (30d limit confirmation) ---")
    bn = probe_binance_oi(now_ms)
    for k, v in bn.items():
        print(f"  {k}: {v}")

    print("\n--- 2. Bybit OI history (potential 720d alt) ---")
    bb = probe_bybit_oi(now_ms)
    for k, v in bb.items():
        print(f"  {k}: {v}")

    print("\n--- 3. OKX OI history ---")
    okx = probe_okx_oi(now_ms)
    for k, v in okx.items():
        print(f"  {k}: {v}")

    print("\n--- 4. Binance L/S ratio + taker volume depth ---")
    lsr = probe_ls_ratio_depth(now_ms)
    for endpoint, depths in lsr.items():
        print(f"  [{endpoint}]")
        for d, info in depths.items():
            print(f"    {d}: {info}")

    summary = {
        "timestamp_utc": ts,
        "binance_oi_confirm": bn,
        "bybit_oi": bb,
        "okx_oi": okx,
        "binance_ls_ratio": lsr,
    }
    out_path = Path(__file__).resolve().parents[2] / "experiments" / "p0" / "oi_alternatives_probe.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"\nSaved to {out_path}")


if __name__ == "__main__":
    main()
