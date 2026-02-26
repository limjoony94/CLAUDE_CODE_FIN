"""Extend btc_5m_270days_reclassified.csv with latest BingX data.

Fetches BTC/USDT 5m candles from BingX public API (no API key needed),
reclassifies using production classify_candle, and appends to existing CSV.
"""
import sys
import time
from pathlib import Path
from datetime import datetime

import ccxt
import pandas as pd
import numpy as np

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from scripts.production.pattern_5m.indicators import classify_candle

AVG_BODY_WINDOW = 20
DATA_FILE = project_root / 'data' / 'btc_5m_270days_reclassified.csv'


def fetch_new_candles(since_ts_ms: int, max_batches: int = 100) -> pd.DataFrame:
    """Fetch 5m candles from BingX starting from since_ts_ms."""
    exchange = ccxt.bingx({
        'enableRateLimit': True,
        'options': {'defaultType': 'swap'},
    })

    all_candles = []
    batch = 0
    current_since = since_ts_ms

    while batch < max_batches:
        batch += 1
        print(f"  Batch {batch}: since={datetime.utcfromtimestamp(current_since/1000)}")

        try:
            candles = exchange.fetch_ohlcv(
                symbol='BTC/USDT:USDT',
                timeframe='5m',
                since=current_since,
                limit=1000,
            )
        except Exception as e:
            print(f"  API error: {e}")
            break

        if not candles:
            print("  No more data")
            break

        # Filter out duplicates with existing fetched data
        if all_candles:
            existing_ts = {c[0] for c in all_candles}
            new = [c for c in candles if c[0] not in existing_ts]
        else:
            new = candles

        if not new:
            print("  All duplicates, done")
            break

        all_candles.extend(new)
        print(f"  Got {len(new)} new candles (total: {len(all_candles)})")

        # Move forward
        latest_ts = max(c[0] for c in new)
        current_since = latest_ts + 1  # next ms after latest

        # If we got fewer than limit, we've reached the end
        if len(candles) < 1000:
            print("  Reached end of available data")
            break

        time.sleep(exchange.rateLimit / 1000)

    if not all_candles:
        return pd.DataFrame()

    df = pd.DataFrame(all_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df.drop_duplicates(subset=['timestamp'], keep='last')
    df = df.sort_values('timestamp').reset_index(drop=True)
    return df


def classify_and_build_patterns(df: pd.DataFrame, prev_types: list = None) -> pd.DataFrame:
    """Classify candles and build 3-candle patterns.

    prev_types: last 2 type_codes from existing data (for pattern continuity).
    """
    # Body calculations
    df['body'] = df['close'] - df['open']
    df['body_abs'] = abs(df['body'])

    # avg_body_20: rolling mean of body_abs
    df['avg_body_20'] = df['body_abs'].rolling(AVG_BODY_WINDOW).mean()

    # Classify each candle
    candle_types = []
    type_codes = []
    for i, row in df.iterrows():
        avg_b = df.at[i, 'avg_body_20']
        if pd.isna(avg_b):
            avg_b = 1.0
        ct = classify_candle(row, avg_b)
        candle_types.append(str(ct))
        type_codes.append(ct.value)

    df['candle_type'] = candle_types
    df['type_code'] = type_codes

    # Build pattern_3 with continuity from previous data
    all_codes = (prev_types or []) + type_codes
    patterns = []
    offset = len(prev_types) if prev_types else 0

    for i in range(len(type_codes)):
        abs_i = i + offset
        if abs_i >= 2:
            pat = f"{all_codes[abs_i-2]}-{all_codes[abs_i-1]}-{all_codes[abs_i]}"
            patterns.append(pat)
        else:
            patterns.append('')

    df['pattern_3'] = patterns
    return df


def main():
    print("=" * 60)
    print("EXTEND 5m DATA")
    print("=" * 60)

    # Load existing data
    print(f"Loading existing data: {DATA_FILE}")
    existing = pd.read_csv(DATA_FILE)
    existing['timestamp'] = pd.to_datetime(existing['timestamp'])

    last_ts = existing['timestamp'].max()
    print(f"Existing: {len(existing)} rows, {existing['timestamp'].min()} → {last_ts}")

    # Get last 2 type codes for pattern continuity
    prev_types = existing['type_code'].iloc[-2:].tolist()
    print(f"Last 2 types for continuity: {prev_types}")

    # Fetch new data starting from next candle
    since_ms = int(last_ts.timestamp() * 1000) + (5 * 60 * 1000)  # +5min
    print(f"\nFetching new data from {datetime.utcfromtimestamp(since_ms/1000)}...")

    new_df = fetch_new_candles(since_ms)

    if new_df.empty:
        print("No new data fetched!")
        return

    print(f"\nFetched: {len(new_df)} new candles")
    print(f"  Range: {new_df['timestamp'].min()} → {new_df['timestamp'].max()}")

    # Need avg_body_20 warm-up: use last 20 rows from existing for continuity
    warmup = existing[['timestamp', 'open', 'high', 'low', 'close', 'volume',
                        'body', 'body_abs']].tail(AVG_BODY_WINDOW).copy()

    # Combine warmup + new for rolling calculation
    combined = pd.concat([warmup, new_df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]],
                         ignore_index=True)
    combined['body'] = combined['close'] - combined['open']
    combined['body_abs'] = abs(combined['body'])
    # Use existing body_abs for warmup rows
    combined.loc[:len(warmup)-1, 'body'] = warmup['body'].values
    combined.loc[:len(warmup)-1, 'body_abs'] = warmup['body_abs'].values
    combined['avg_body_20'] = combined['body_abs'].rolling(AVG_BODY_WINDOW).mean()

    # Extract only new rows
    new_with_avg = combined.iloc[len(warmup):].copy().reset_index(drop=True)
    new_with_avg['timestamp'] = new_df['timestamp'].values

    # Classify
    print("Classifying candles...")
    new_classified = classify_and_build_patterns(new_with_avg, prev_types)

    # Match existing CSV format
    output_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume',
                   'body', 'body_abs', 'avg_body_20', 'candle_type', 'type_code', 'pattern_3']
    new_classified = new_classified[output_cols]

    # Format timestamp to match existing
    new_classified['timestamp'] = new_classified['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')
    existing['timestamp'] = existing['timestamp'].dt.strftime('%Y-%m-%d %H:%M:%S')

    # Remove any overlap
    existing_ts = set(existing['timestamp'])
    new_classified = new_classified[~new_classified['timestamp'].isin(existing_ts)]

    if len(new_classified) == 0:
        print("No new non-overlapping candles!")
        return

    # Append
    final = pd.concat([existing, new_classified], ignore_index=True)

    # Save
    final.to_csv(DATA_FILE, index=False)

    total_days = (pd.to_datetime(final['timestamp'].iloc[-1]) -
                  pd.to_datetime(final['timestamp'].iloc[0])).total_seconds() / 86400

    print(f"\n{'='*60}")
    print(f"DATA EXTENSION COMPLETE")
    print(f"{'='*60}")
    print(f"Previous: {len(existing)} rows")
    print(f"Added:    {len(new_classified)} rows")
    print(f"Total:    {len(final)} rows ({total_days:.1f} days)")
    print(f"Range:    {final['timestamp'].iloc[0]} → {final['timestamp'].iloc[-1]}")
    print(f"Saved:    {DATA_FILE}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
