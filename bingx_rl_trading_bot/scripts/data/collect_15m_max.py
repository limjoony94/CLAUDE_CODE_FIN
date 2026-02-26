"""Collect MAXIMUM 15m BTC data from BingX — both backward and forward."""

import sys
from pathlib import Path
from datetime import datetime

import ccxt
import pandas as pd
import time

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

EXISTING_FILE = project_root / 'data' / 'btc_15m_extended.csv'
OUTPUT_FILE = project_root / 'data' / 'btc_15m_max.csv'


def fetch_backward(exchange, symbol, earliest_ts_ms, max_batches=100):
    """Fetch data backward from earliest existing timestamp."""
    all_candles = []
    batch = 0

    # Go backward: each batch = 1000 candles * 15min = 10.4 days
    until_ms = earliest_ts_ms - 1

    while batch < max_batches:
        batch += 1
        since_ms = until_ms - (1000 * 15 * 60 * 1000)

        print(f"  Backward batch {batch}: {datetime.fromtimestamp(since_ms/1000)} → {datetime.fromtimestamp(until_ms/1000)}")

        try:
            candles = exchange.fetch_ohlcv(
                symbol=symbol,
                timeframe='15m',
                since=since_ms,
                limit=1000,
            )
        except Exception as e:
            print(f"  Error: {e}")
            time.sleep(2)
            continue

        if not candles:
            print("  No more historical data available")
            break

        # Filter candles before our earliest
        candles = [c for c in candles if c[0] < earliest_ts_ms]
        if not candles:
            print("  No new earlier candles")
            break

        all_candles.extend(candles)
        oldest = candles[0][0]
        print(f"  Got {len(candles)} candles, oldest: {datetime.fromtimestamp(oldest/1000)}")

        if len(candles) < 500:  # Getting sparse
            print("  Approaching data boundary")
            # Try one more batch further back
            until_ms = oldest - 1
            time.sleep(0.5)
            continue

        until_ms = oldest - 1
        time.sleep(0.5)

    return all_candles


def fetch_forward(exchange, symbol, latest_ts_ms, max_batches=30):
    """Fetch data forward from latest existing timestamp."""
    all_candles = []
    batch = 0
    since_ms = latest_ts_ms + 1

    while batch < max_batches:
        batch += 1
        print(f"  Forward batch {batch}: since {datetime.fromtimestamp(since_ms/1000)}")

        try:
            candles = exchange.fetch_ohlcv(
                symbol=symbol,
                timeframe='15m',
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

        all_candles.extend(candles)
        last = candles[-1][0]
        print(f"  Got {len(candles)} candles, last: {datetime.fromtimestamp(last/1000)}")

        if len(candles) < 1000:
            break

        since_ms = last + 1
        time.sleep(0.5)

    return all_candles


def main():
    print("=" * 60)
    print("COLLECT MAXIMUM 15m BTC DATA")
    print("=" * 60)

    # Load existing
    df_existing = pd.read_csv(EXISTING_FILE)
    ohlcv_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    available = [c for c in ohlcv_cols if c in df_existing.columns]
    df_existing = df_existing[available].copy()

    earliest_str = df_existing['timestamp'].iloc[0]
    latest_str = df_existing['timestamp'].iloc[-1]
    earliest_ms = int(pd.Timestamp(earliest_str).timestamp() * 1000)
    latest_ms = int(pd.Timestamp(latest_str).timestamp() * 1000)
    print(f"Existing: {earliest_str} → {latest_str} ({len(df_existing)} candles)")

    # Exchange setup
    exchange = ccxt.bingx({
        'enableRateLimit': True,
        'options': {'defaultType': 'swap'},
    })
    symbol = 'BTC/USDT:USDT'

    # 1. Fetch backward
    print(f"\n--- BACKWARD EXTENSION ---")
    backward = fetch_backward(exchange, symbol, earliest_ms)
    print(f"Fetched {len(backward)} candles backward")

    # 2. Fetch forward
    print(f"\n--- FORWARD EXTENSION ---")
    forward = fetch_forward(exchange, symbol, latest_ms)
    print(f"Fetched {len(forward)} candles forward")

    # Combine all
    all_new = backward + forward
    if all_new:
        df_new = pd.DataFrame(all_new, columns=['ts_ms', 'open', 'high', 'low', 'close', 'volume'])
        df_new['timestamp'] = pd.to_datetime(df_new['ts_ms'], unit='ms').dt.strftime('%Y-%m-%d %H:%M:%S')
        df_new = df_new[['timestamp', 'open', 'high', 'low', 'close', 'volume']]

        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_combined = df_existing.copy()

    df_combined.drop_duplicates(subset='timestamp', keep='first', inplace=True)
    df_combined.sort_values('timestamp', inplace=True)
    df_combined.reset_index(drop=True, inplace=True)

    print(f"\n{'=' * 60}")
    print(f"RESULT: {df_combined['timestamp'].iloc[0]} → {df_combined['timestamp'].iloc[-1]}")
    days = (pd.Timestamp(df_combined['timestamp'].iloc[-1]) - pd.Timestamp(df_combined['timestamp'].iloc[0])).days
    print(f"Total: {len(df_combined)} candles ({days} days)")
    print(f"Added: {len(df_combined) - len(df_existing)} new candles")

    df_combined.to_csv(OUTPUT_FILE, index=False)
    print(f"Saved: {OUTPUT_FILE}")


if __name__ == '__main__':
    main()
