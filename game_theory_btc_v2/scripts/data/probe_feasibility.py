"""P0.2 Feasibility Probe — single-call sanity check before full fetch.

Tests data source availability:
1. Binance OHLCV + OI history + funding history
2. BingX OHLCV + OI history + funding history
3. Coinglass free-tier liquidation endpoints
4. Spot price availability (for H6/H7 spot-perp basis)

Output: prints to console + JSON summary.
Run: python scripts/data/probe_feasibility.py
"""
import json
import sys
import time
from datetime import datetime, timezone

import ccxt
import requests


def _safe_call(fn, *args, **kwargs):
    """Call fn, return (ok, result_or_error_str, sample_repr)."""
    try:
        result = fn(*args, **kwargs)
        sample = result[:3] if isinstance(result, list) else result
        return True, None, repr(sample)[:600]
    except Exception as e:
        return False, f"{type(e).__name__}: {e}", None


def probe_exchange(exchange_name: str) -> dict:
    """Probe a CCXT exchange for OHLCV/OI/funding capability."""
    out = {"exchange": exchange_name, "tests": {}}
    try:
        ex_cls = getattr(ccxt, exchange_name)
        ex = ex_cls({"enableRateLimit": True})
        ex.load_markets()
        out["loaded"] = True
        out["has"] = {
            "fetchOHLCV": ex.has.get("fetchOHLCV", False),
            "fetchOpenInterestHistory": ex.has.get("fetchOpenInterestHistory", False),
            "fetchFundingRateHistory": ex.has.get("fetchFundingRateHistory", False),
        }
    except Exception as e:
        out["loaded"] = False
        out["init_error"] = f"{type(e).__name__}: {e}"
        return out

    # Try multiple symbol formats — exchanges differ
    candidate_symbols = ["BTC/USDT:USDT", "BTC/USDT", "BTCUSDT"]
    perp_symbol = None
    for s in candidate_symbols:
        if s in ex.markets:
            m = ex.markets[s]
            if m.get("contract") or m.get("swap") or m.get("future"):
                perp_symbol = s
                break
    if perp_symbol is None:
        # try first contract market
        for s, m in ex.markets.items():
            if (m.get("base") == "BTC" and m.get("quote") == "USDT" and
                    (m.get("swap") or m.get("contract"))):
                perp_symbol = s
                break
    out["perp_symbol_chosen"] = perp_symbol
    spot_symbol = "BTC/USDT" if "BTC/USDT" in ex.markets else None
    out["spot_symbol_chosen"] = spot_symbol

    # OHLCV perp 1h
    if perp_symbol:
        ok, err, sample = _safe_call(ex.fetch_ohlcv, perp_symbol, "1h", limit=5)
        out["tests"]["ohlcv_perp_1h"] = {"ok": ok, "err": err, "sample": sample}
        time.sleep(0.5)
    # OHLCV perp 5m
    if perp_symbol:
        ok, err, sample = _safe_call(ex.fetch_ohlcv, perp_symbol, "5m", limit=5)
        out["tests"]["ohlcv_perp_5m"] = {"ok": ok, "err": err, "sample": sample}
        time.sleep(0.5)
    # OHLCV spot 1h (for H7 basis)
    if spot_symbol:
        ok, err, sample = _safe_call(ex.fetch_ohlcv, spot_symbol, "1h", limit=5)
        out["tests"]["ohlcv_spot_1h"] = {"ok": ok, "err": err, "sample": sample}
        time.sleep(0.5)
    # OI history
    if perp_symbol and ex.has.get("fetchOpenInterestHistory"):
        ok, err, sample = _safe_call(ex.fetch_open_interest_history, perp_symbol, "1h", limit=5)
        out["tests"]["oi_history"] = {"ok": ok, "err": err, "sample": sample}
        time.sleep(0.5)
    else:
        out["tests"]["oi_history"] = {"ok": False, "err": "not_supported_by_ccxt", "sample": None}
    # Funding history
    if perp_symbol and ex.has.get("fetchFundingRateHistory"):
        ok, err, sample = _safe_call(ex.fetch_funding_rate_history, perp_symbol, limit=5)
        out["tests"]["funding_history"] = {"ok": ok, "err": err, "sample": sample}
        time.sleep(0.5)
    else:
        out["tests"]["funding_history"] = {"ok": False, "err": "not_supported_by_ccxt", "sample": None}
    return out


def probe_coinglass() -> dict:
    """Probe Coinglass public/free endpoints for liquidation data."""
    out = {"endpoints": []}
    candidates = [
        ("v2 liquidation_history h1",
         "https://open-api.coinglass.com/public/v2/liquidation_history",
         {"symbol": "BTC", "time_type": "h1"}),
        ("v2 liquidation",
         "https://open-api.coinglass.com/public/v2/liquidation",
         {"symbol": "BTC", "time_type": "h1"}),
        ("v3 liquidation aggregated history",
         "https://open-api-v3.coinglass.com/api/futures/liquidation/aggregated-history",
         {"symbol": "BTCUSDT", "interval": "h1"}),
        ("v3 liquidation history",
         "https://open-api-v3.coinglass.com/api/futures/liquidation/history",
         {"symbol": "BTCUSDT", "interval": "h1"}),
    ]
    for name, url, params in candidates:
        try:
            r = requests.get(url, params=params, timeout=10,
                             headers={"User-Agent": "Mozilla/5.0 (probe)"})
            entry = {"name": name, "url": url, "status": r.status_code,
                     "content_preview": r.text[:400]}
        except Exception as e:
            entry = {"name": name, "url": url, "error": f"{type(e).__name__}: {e}"}
        out["endpoints"].append(entry)
        time.sleep(0.5)
    return out


def main():
    print(f"=== P0.2 Feasibility Probe ({datetime.now(timezone.utc).isoformat()}) ===")
    summary = {"timestamp_utc": datetime.now(timezone.utc).isoformat(), "results": {}}

    for exch in ["binance", "bingx"]:
        print(f"\n--- Probing {exch} ---")
        result = probe_exchange(exch)
        summary["results"][exch] = result
        # console pretty print
        print(f"  loaded: {result.get('loaded')}")
        print(f"  perp_symbol: {result.get('perp_symbol_chosen')}")
        print(f"  spot_symbol: {result.get('spot_symbol_chosen')}")
        print(f"  has: {result.get('has')}")
        for tname, tres in result.get("tests", {}).items():
            print(f"  test[{tname}]: ok={tres['ok']}")
            if not tres['ok']:
                print(f"    err: {tres['err']}")
            else:
                print(f"    sample: {(tres['sample'] or '')[:200]}")

    print(f"\n--- Probing Coinglass ---")
    cg_result = probe_coinglass()
    summary["results"]["coinglass"] = cg_result
    for entry in cg_result["endpoints"]:
        print(f"  [{entry['name']}]")
        if "error" in entry:
            print(f"    error: {entry['error']}")
        else:
            print(f"    status={entry['status']}")
            print(f"    preview: {entry['content_preview'][:200]}")

    # Save JSON summary
    out_path = "experiments/p0/feasibility_probe_raw.json"
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False, default=str)
        print(f"\nSaved raw JSON to {out_path}")
    except Exception as e:
        print(f"\nFailed to save JSON: {e}")
    print("Done.")


if __name__ == "__main__":
    main()
