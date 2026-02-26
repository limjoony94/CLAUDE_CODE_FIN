"""Collect older 15m BTC data from Binance public API (no keys needed)."""

import sys
from pathlib import Path
from datetime import datetime

import ccxt
import pandas as pd
import time

project_root = Path(__file__).parent.parent.parent
OUTPUT_FILE = project_root / 'data' / 'btc_15m_binance_old.csv'

# Fetch from 2020-01-01 to 2024-02-23 (where BingX data starts)
START_DATE = '2020-01-01'
END_DATE = '2024-02-23 05:00:00'


def main():
    print("=" * 60)
    print("COLLECT OLD 15m BTC DATA FROM BINANCE")
    print(f"Target: {START_DATE} → {END_DATE}")
    print("=" * 60)

    exchange = ccxt.binance({
        'enableRateLimit': True,
        'options': {'defaultType': 'future'},
    })

    symbol = 'BTC/USDT:USDT'
    end_ms = int(pd.Timestamp(END_DATE).timestamp() * 1000)
    since_ms = int(pd.Timestamp(START_DATE).timestamp() * 1000)

    all_candles = []
    batch = 0
    max_batches = 200  # 200 * 1000 * 15min = 2083 days

    while batch < max_batches and since_ms < end_ms:
        batch += 1

        if batch % 10 == 1:
            print(f"  Batch {batch}: {datetime.fromtimestamp(since_ms/1000)} | Total: {len(all_candles)}")

        try:
            candles = exchange.fetch_ohlcv(
                symbol=symbol,
                timeframe='15m',
                since=since_ms,
                limit=1000,
            )
        except Exception as e:
            print(f"  Error: {e}")
            time.sleep(3)
            continue

        if not candles:
            print("  No more data")
            break

        # Filter candles before end date
        candles = [c for c in candles if c[0] < end_ms]
        all_candles.extend(candles)

        last_ts = candles[-1][0] if candles else since_ms
        since_ms = last_ts + 1

        if len(candles) < 1000:
            break

        time.sleep(0.3)

    print(f"\nFetched {len(all_candles)} candles total")

    if not all_candles:
        print("No data collected!")
        return

    df = pd.DataFrame(all_candles, columns=['ts_ms', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['ts_ms'], unit='ms').dt.strftime('%Y-%m-%d %H:%M:%S')
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]
    df.drop_duplicates(subset='timestamp', keep='first', inplace=True)
    df.sort_values('timestamp', inplace=True)
    df.reset_index(drop=True, inplace=True)

    print(f"Result: {df['timestamp'].iloc[0]} → {df['timestamp'].iloc[-1]} ({len(df)} candles)")
    days = (pd.Timestamp(df['timestamp'].iloc[-1]) - pd.Timestamp(df['timestamp'].iloc[0])).days
    print(f"Coverage: {days} days")

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved: {OUTPUT_FILE}")


if __name__ == '__main__':
    main()
