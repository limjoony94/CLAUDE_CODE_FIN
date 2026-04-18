#!/usr/bin/env python3
"""
Collect 5m OHLCV data for multiple assets from Binance (public, no API key).
Resample to 15m and save both.
Target: 365 days for vol-spike multi-asset strategy.
"""
import sys, time
from pathlib import Path
from datetime import datetime, timedelta
import ccxt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"

ASSETS = [
    "XRP", "DOGE", "AVAX", "LINK", "ADA", "BNB", "DOT",
    "NEAR", "LTC", "BCH", "ATOM", "UNI", "FIL",
    "APT", "ARB", "OP", "TRX",
]

DAYS = 365
TIMEFRAME = "5m"
BARS_PER_DAY_5M = 288

def collect_asset(exchange, symbol_base, days=365):
    """Collect 5m data for a single asset."""
    symbol = f"{symbol_base}/USDT"
    print(f"  Collecting {symbol} {days}d...", end=" ", flush=True)

    end_ts = int(datetime.now().timestamp() * 1000)
    start_ts = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
    since = start_ts

    all_candles = []
    batch = 0
    max_batches = 500  # safety limit

    while since < end_ts and batch < max_batches:
        try:
            candles = exchange.fetch_ohlcv(symbol, TIMEFRAME, since=since, limit=1000)
            if not candles:
                break
            all_candles.extend(candles)
            since = candles[-1][0] + 1
            batch += 1
            time.sleep(0.1)  # rate limit
        except Exception as e:
            print(f"ERROR: {e}")
            break

    if not all_candles:
        print("NO DATA")
        return None

    df = pd.DataFrame(all_candles, columns=["timestamp", "open", "high", "low", "close", "volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    print(f"{len(df)} bars ({len(df)/BARS_PER_DAY_5M:.0f}d)")
    return df

def main():
    print("=" * 60)
    print("Multi-Asset 5m Data Collection (Binance Public)")
    print("=" * 60)

    exchange = ccxt.binance({
        "enableRateLimit": True,
        "options": {"defaultType": "future"},
    })

    # Check which assets already have sufficient 5m data
    existing = {}
    for asset in ASSETS:
        for pattern in [f"{asset.lower()}_5m_*.csv", f"{asset.lower()}_binance_5m.csv"]:
            import glob
            matches = list(DATA_DIR.glob(pattern))
            for m in matches:
                try:
                    df = pd.read_csv(m)
                    if len(df) > BARS_PER_DAY_5M * 200:  # at least 200 days
                        existing[asset] = (m, len(df))
                except:
                    pass

    print(f"\nExisting sufficient data: {list(existing.keys())}")
    to_collect = [a for a in ASSETS if a not in existing]
    print(f"To collect: {to_collect} ({len(to_collect)} assets)")

    collected = {}
    for asset in to_collect:
        df = collect_asset(exchange, asset, DAYS)
        if df is not None and len(df) > BARS_PER_DAY_5M * 100:
            # Save 5m
            path_5m = DATA_DIR / f"{asset.lower()}_binance_5m.csv"
            df.to_csv(path_5m, index=False)

            # Resample to 15m
            df_idx = df.set_index("timestamp")
            df_15m = df_idx.resample("15min").agg({
                "open": "first", "high": "max", "low": "min",
                "close": "last", "volume": "sum"
            }).dropna(subset=["open"]).reset_index()
            path_15m = DATA_DIR / f"{asset.lower()}_15m_binance.csv"
            df_15m.to_csv(path_15m, index=False)

            collected[asset] = {"5m": len(df), "15m": len(df_15m), "days": len(df_15m)/96}
            print(f"    Saved: {path_5m.name} ({len(df)} bars) + {path_15m.name} ({len(df_15m)} bars)")
        time.sleep(0.5)

    # Also resample existing ETH and SOL if 15m doesn't exist
    for asset, (path, n) in existing.items():
        path_15m = DATA_DIR / f"{asset.lower()}_15m_binance.csv"
        if not path_15m.exists():
            df = pd.read_csv(path, parse_dates=["timestamp"])
            df_idx = df.set_index("timestamp")
            df_15m = df_idx.resample("15min").agg({
                "open": "first", "high": "max", "low": "min",
                "close": "last", "volume": "sum"
            }).dropna(subset=["open"]).reset_index()
            df_15m.to_csv(path_15m, index=False)
            print(f"  Resampled {asset}: {len(df_15m)} 15m bars")

    print(f"\nCollection complete. Collected: {len(collected)}, Existing: {len(existing)}")
    print(f"Total assets with 15m data: {len(collected) + len(existing) + 3}")  # +3 for BTC/ETH/SOL

    # Summary
    print("\n" + "=" * 60)
    print("Available 15m Data Summary")
    print("=" * 60)
    all_15m = list(DATA_DIR.glob("*_15m_*.csv")) + list(DATA_DIR.glob("*_15m_max.csv"))
    for f in sorted(all_15m):
        try:
            df = pd.read_csv(f)
            print(f"  {f.name}: {len(df)} bars ({len(df)/96:.0f} days)")
        except:
            print(f"  {f.name}: ERROR reading")

if __name__ == "__main__":
    main()
