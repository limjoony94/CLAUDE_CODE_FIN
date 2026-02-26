"""Extend 15m BTC data to latest available from BingX."""

import sys
from pathlib import Path
from datetime import datetime

import ccxt
import pandas as pd
import time

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

DATA_FILE = project_root / 'data' / 'btc_15m_720days.csv'
OUTPUT_FILE = project_root / 'data' / 'btc_15m_extended.csv'


def main():
    print("=" * 60)
    print("EXTEND 15m BTC DATA TO LATEST")
    print("=" * 60)

    # Load existing data
    df_existing = pd.read_csv(DATA_FILE)
    # Keep only OHLCV columns
    ohlcv_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    available = [c for c in ohlcv_cols if c in df_existing.columns]
    df_existing = df_existing[available].copy()

    last_ts_str = df_existing['timestamp'].iloc[-1]
    last_ts = pd.Timestamp(last_ts_str)
    print(f"Existing data: {df_existing['timestamp'].iloc[0]} → {last_ts_str} ({len(df_existing)} candles)")

    # Setup exchange
    exchange = ccxt.bingx({
        'enableRateLimit': True,
        'options': {'defaultType': 'swap'},
    })

    symbol = 'BTC/USDT:USDT'
    timeframe = '15m'

    # Fetch from last timestamp forward
    since_ms = int(last_ts.timestamp() * 1000) + 1  # +1ms to avoid overlap
    all_new = []
    batch = 0
    max_batches = 30

    while batch < max_batches:
        batch += 1
        print(f"  Batch {batch}: since {datetime.fromtimestamp(since_ms/1000)}")

        try:
            candles = exchange.fetch_ohlcv(
                symbol=symbol,
                timeframe=timeframe,
                since=since_ms,
                limit=1000,
            )
        except Exception as e:
            print(f"  Error: {e}")
            time.sleep(2)
            continue

        if not candles:
            print("  No more data")
            break

        all_new.extend(candles)
        last_fetched = candles[-1][0]
        print(f"  Got {len(candles)} candles, last: {datetime.fromtimestamp(last_fetched/1000)}")

        if len(candles) < 1000:
            print("  Reached end of available data")
            break

        since_ms = last_fetched + 1
        time.sleep(0.5)

    if not all_new:
        print("No new data to add. Data is already up to date.")
        # Still save cleaned version
        df_existing.to_csv(OUTPUT_FILE, index=False)
        print(f"Saved: {OUTPUT_FILE} ({len(df_existing)} candles)")
        return

    # Convert new data to DataFrame
    df_new = pd.DataFrame(all_new, columns=['ts_ms', 'open', 'high', 'low', 'close', 'volume'])
    df_new['timestamp'] = pd.to_datetime(df_new['ts_ms'], unit='ms').dt.strftime('%Y-%m-%d %H:%M:%S')
    df_new = df_new[['timestamp', 'open', 'high', 'low', 'close', 'volume']]

    # Combine and deduplicate
    df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    df_combined.drop_duplicates(subset='timestamp', keep='first', inplace=True)
    df_combined.sort_values('timestamp', inplace=True)
    df_combined.reset_index(drop=True, inplace=True)

    print(f"\nResult: {df_combined['timestamp'].iloc[0]} → {df_combined['timestamp'].iloc[-1]}")
    print(f"Total candles: {len(df_combined)} (added {len(df_combined) - len(df_existing)} new)")

    df_combined.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved: {OUTPUT_FILE}")


if __name__ == '__main__':
    main()
