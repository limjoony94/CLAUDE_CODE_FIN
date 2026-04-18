#!/usr/bin/env python3
"""Collect 5m data for additional altcoins from Binance."""
import time
from pathlib import Path
from datetime import datetime, timedelta
import ccxt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data"

# New assets not yet collected
NEW_ASSETS = [
    "PEPE","WLD","SEI","SUI","INJ","TIA","JUP",
    "MATIC","FTM","ALGO","SAND","MANA","AXS","CRV",
    "AAVE","MKR","COMP","SNX","DYDX","GMX",
    "RUNE","PENDLE","WIF","BONK","FLOKI","ENA",
    "STRK","MANTA","ONDO","PYTH"
]

def collect(exchange, symbol_base, days=365):
    symbol = f"{symbol_base}/USDT"
    print(f"  {symbol}...", end=" ", flush=True)
    since = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
    end = int(datetime.now().timestamp() * 1000)
    candles = []
    for _ in range(500):
        try:
            batch = exchange.fetch_ohlcv(symbol, "5m", since=since, limit=1000)
            if not batch: break
            candles.extend(batch)
            since = batch[-1][0] + 1
            if since >= end: break
            time.sleep(0.05)
        except Exception as e:
            print(f"ERR({e})")
            return None
    if not candles:
        print("NO DATA")
        return None
    df = pd.DataFrame(candles, columns=["timestamp","open","high","low","close","volume"])
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    days_actual = len(df) / 288
    print(f"{len(df)} bars ({days_actual:.0f}d)")
    return df

def main():
    print("Collecting additional altcoins 5m data...")
    ex = ccxt.binance({"enableRateLimit": True, "options": {"defaultType": "future"}})

    collected = 0
    for asset in NEW_ASSETS:
        # Skip if already exists
        if (DATA / f"{asset.lower()}_binance_5m.csv").exists():
            print(f"  {asset}: already exists, skip")
            continue
        if (DATA / f"{asset.lower()}_15m_binance.csv").exists():
            print(f"  {asset}: 15m exists, skip")
            continue

        df = collect(ex, asset)
        if df is not None and len(df) > 288 * 100:
            df.to_csv(DATA / f"{asset.lower()}_binance_5m.csv", index=False)
            # Resample to 15m
            df15 = df.set_index("timestamp").resample("15min").agg(
                {"open":"first","high":"max","low":"min","close":"last","volume":"sum"}
            ).dropna(subset=["open"]).reset_index()
            df15.to_csv(DATA / f"{asset.lower()}_15m_binance.csv", index=False)
            collected += 1
        time.sleep(0.3)

    print(f"\nCollected {collected} new assets")
    # Count total 15m files
    all_15m = list(DATA.glob("*_15m_binance.csv"))
    print(f"Total 15m Binance files: {len(all_15m)}")

if __name__ == "__main__":
    main()
