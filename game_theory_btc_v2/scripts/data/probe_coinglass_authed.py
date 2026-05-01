"""Probe Coinglass API with the registered free-tier key.

Tests:
1. Authentication header convention (CG-API-KEY vs coinglassSecret)
2. Free-tier endpoints availability for liquidation history
3. Rate limit / quota visibility

Run: python scripts/data/probe_coinglass_authed.py
"""
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import requests

from keys import get_coinglass


def probe(name: str, url: str, headers: dict, params: dict | None = None) -> dict:
    """Single GET, return result dict."""
    try:
        r = requests.get(url, headers=headers, params=params or {}, timeout=15)
        # try parse json
        try:
            j = r.json()
            preview = json.dumps(j, ensure_ascii=False)[:500]
        except Exception:
            preview = r.text[:500]
        return {"name": name, "status": r.status_code, "preview": preview}
    except Exception as e:
        return {"name": name, "error": f"{type(e).__name__}: {e}"}


def main():
    api_key = get_coinglass()
    print(f"Coinglass key loaded (masked): {api_key[:6]}...{api_key[-4:]} (len={len(api_key)})")
    print(f"Probe time: {datetime.now(timezone.utc).isoformat()}")

    # Two header conventions to test
    headers_v3 = {"CG-API-KEY": api_key, "Accept": "application/json"}
    headers_v2 = {"coinglassSecret": api_key, "Accept": "application/json"}

    targets = [
        # v3 (current generation)
        ("v3 liq aggregated-history (CG-API-KEY)",
         "https://open-api-v3.coinglass.com/api/futures/liquidation/aggregated-history",
         headers_v3, {"symbol": "BTCUSDT", "interval": "h1"}),
        ("v3 liq history (CG-API-KEY)",
         "https://open-api-v3.coinglass.com/api/futures/liquidation/history",
         headers_v3, {"exchange": "Binance", "symbol": "BTCUSDT", "interval": "h1"}),
        ("v3 liq aggregated-coin-history",
         "https://open-api-v3.coinglass.com/api/futures/liquidation/aggregated-coin-history",
         headers_v3, {"symbol": "BTC", "interval": "h1"}),
        ("v3 funding rate history",
         "https://open-api-v3.coinglass.com/api/futures/fundingRate/ohlc-history",
         headers_v3, {"exchange": "Binance", "symbol": "BTCUSDT", "interval": "h8"}),
        ("v3 open interest aggregated history",
         "https://open-api-v3.coinglass.com/api/futures/openInterest/ohlc-aggregated-history",
         headers_v3, {"symbol": "BTC", "interval": "h1"}),
        # v2 legacy (fallback)
        ("v2 liquidation_history h1 (coinglassSecret)",
         "https://open-api.coinglass.com/public/v2/liquidation_history",
         headers_v2, {"symbol": "BTC", "time_type": "h1"}),
        ("v2 liquidation_history h1 (CG-API-KEY)",
         "https://open-api.coinglass.com/public/v2/liquidation_history",
         headers_v3, {"symbol": "BTC", "time_type": "h1"}),
    ]

    results = []
    for name, url, hdrs, params in targets:
        print(f"\n--- {name} ---")
        res = probe(name, url, hdrs, params)
        results.append(res)
        if "error" in res:
            print(f"  ERROR: {res['error']}")
        else:
            print(f"  status={res['status']}")
            print(f"  preview: {res['preview'][:300]}")
        time.sleep(1.0)  # respect free tier

    # Save full output
    out_path = Path(__file__).resolve().parents[2] / "experiments" / "p0" / "coinglass_authed_probe_raw.json"
    summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "key_masked": f"{api_key[:6]}...{api_key[-4:]}",
        "results": results,
    }
    try:
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, ensure_ascii=False)
        print(f"\nSaved to {out_path}")
    except Exception as e:
        print(f"\nFailed to save: {e}")


if __name__ == "__main__":
    main()
