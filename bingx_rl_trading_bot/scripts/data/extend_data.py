#!/usr/bin/env python3
"""
Extend existing CSV with latest BingX OHLCV data.
Appends new 5m candles from last CSV timestamp to now.
Reclassifies candle types using production classify_candle.
"""
import os, sys, time
import pandas as pd
import numpy as np
import yaml
import ccxt

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, PROJECT_ROOT)

from scripts.production.pattern_5m.indicators import classify_candle

CSV_PATH = os.path.join(PROJECT_ROOT, 'data', 'btc_5m_270days_reclassified.csv')
CONFIG_PATH = os.path.join(PROJECT_ROOT, 'config', 'api_keys.yaml')
SYMBOL = 'BTC/USDT:USDT'
TF = '5m'
LIMIT = 1000  # BingX max per request


def load_existing():
    df = pd.read_csv(CSV_PATH)
    last_ts = pd.to_datetime(df.iloc[-1]['timestamp'])
    print(f"Existing data: {len(df)} bars, last: {last_ts}")
    return df, last_ts


def create_exchange():
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)
    keys = cfg['bingx']['mainnet']
    ex = ccxt.bingx({
        'apiKey': keys['api_key'],
        'secret': keys['secret_key'],
        'options': {'defaultType': 'swap'},
    })
    return ex


def fetch_new_candles(exchange, since_ts):
    """Fetch all 5m candles from since_ts to now."""
    since_ms = int(since_ts.timestamp() * 1000) + 300000  # +1 bar to avoid duplicate
    all_candles = []
    while True:
        candles = exchange.fetch_ohlcv(SYMBOL, TF, since=since_ms, limit=LIMIT)
        if not candles:
            break
        all_candles.extend(candles)
        last_ms = candles[-1][0]
        print(f"  Fetched {len(candles)} candles, last: {pd.to_datetime(last_ms, unit='ms')}")
        if len(candles) < LIMIT:
            break
        since_ms = last_ms + 300000
        time.sleep(0.5)
    return all_candles


def candles_to_df(candles):
    """Convert OHLCV list to DataFrame matching existing CSV format."""
    rows = []
    for c in candles:
        rows.append({
            'timestamp': pd.to_datetime(c[0], unit='ms').strftime('%Y-%m-%d %H:%M:%S'),
            'open': c[1], 'high': c[2], 'low': c[3], 'close': c[4], 'volume': c[5],
        })
    df = pd.DataFrame(rows)
    return df


def classify_df(df):
    """Add candle classification columns."""
    df['body'] = df['close'] - df['open']
    df['body_abs'] = df['body'].abs()
    df['avg_body_20'] = df['body_abs'].rolling(20, min_periods=1).mean()

    types = []
    for i, row in df.iterrows():
        ct = classify_candle(row, row['avg_body_20'])
        types.append(ct.value if hasattr(ct, 'value') else str(ct))
    df['candle_type'] = types

    # type_code: 2-char abbreviation
    code_map = {
        'DOJI': 'D', 'DRAGONFLY': 'DF', 'GRAVESTONE': 'GS',
        'HAMMER': 'H', 'INV_HAMMER': 'IH', 'SPINNING_TOP': 'ST',
        'MARUBOZU_UP': 'MU', 'MARUBOZU_DOWN': 'MD',
        'BIG_UP': 'BU', 'BIG_DOWN': 'BD',
        'MED_UP': 'U', 'MED_DOWN': 'DN',
    }
    df['type_code'] = df['candle_type'].map(code_map).fillna('?')

    # pattern_3: rolling 3-candle pattern
    codes = df['type_code'].values
    patterns = [''] * len(codes)
    for i in range(2, len(codes)):
        patterns[i] = f"{codes[i-2]}-{codes[i-1]}-{codes[i]}"
    df['pattern_3'] = patterns

    return df


def main():
    print("=" * 60)
    print("EXTEND DATA — Append latest BingX OHLCV to CSV")
    print("=" * 60)

    existing_df, last_ts = load_existing()
    exchange = create_exchange()

    print(f"\nFetching candles since {last_ts}...")
    candles = fetch_new_candles(exchange, last_ts)
    print(f"Total new candles: {len(candles)}")

    if not candles:
        print("No new data to append.")
        return

    new_df = candles_to_df(candles)

    # Remove last (possibly incomplete) candle
    now = pd.Timestamp.now()
    current_bar_start = now.floor('5min')
    new_df = new_df[pd.to_datetime(new_df['timestamp']) < current_bar_start]
    print(f"After removing current bar: {len(new_df)} candles")

    if len(new_df) == 0:
        print("No complete candles to append.")
        return

    # Remove duplicates with existing data
    existing_last = existing_df.iloc[-1]['timestamp']
    new_df = new_df[pd.to_datetime(new_df['timestamp']) > pd.to_datetime(existing_last)]
    print(f"After dedup: {len(new_df)} new candles")

    if len(new_df) == 0:
        print("No new non-duplicate candles.")
        return

    # Classify new data (need rolling avg_body_20 from tail of existing)
    # Combine tail of existing + new for rolling calculation
    tail_n = 25
    combined = pd.concat([existing_df.tail(tail_n)[['timestamp', 'open', 'high', 'low', 'close', 'volume']],
                          new_df], ignore_index=True)
    combined = classify_df(combined)

    # Keep only the truly new rows
    new_classified = combined.iloc[tail_n:].reset_index(drop=True)

    # Recalculate avg_body_20 using full history for the first 20 new rows
    # (the rolling window may not be accurate for the first few rows with just tail_n)

    # Append to existing
    final_df = pd.concat([existing_df, new_classified], ignore_index=True)

    # Backup original
    backup = CSV_PATH + '.bak'
    if not os.path.exists(backup):
        existing_df.to_csv(backup, index=False)
        print(f"Backup saved: {backup}")

    final_df.to_csv(CSV_PATH, index=False)
    new_last = final_df.iloc[-1]['timestamp']
    new_days = (pd.to_datetime(new_last) - pd.to_datetime(final_df.iloc[0]['timestamp'])).days

    print(f"\n✅ Data extended: {len(existing_df)} → {len(final_df)} bars ({new_days} days)")
    print(f"   New range: {final_df.iloc[0]['timestamp']} → {new_last}")
    print(f"   Added: {len(new_classified)} bars ({len(new_classified)/288:.1f} days)")


if __name__ == '__main__':
    main()
