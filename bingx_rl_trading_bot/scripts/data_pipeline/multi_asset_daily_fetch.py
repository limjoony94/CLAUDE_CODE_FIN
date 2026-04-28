"""Fetch daily OHLCV for top-10 crypto via Binance API (CCXT).

Universe: top 10 by liquidity. USDT spot pairs.
Output: data/multi_asset_daily.parquet (combined long-format)

720 days target = ~2 years.
"""
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import ccxt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
OUT_FILE = ROOT / 'data' / 'multi_asset_daily.parquet'

UNIVERSE = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT', 'XRP/USDT',
            'ADA/USDT', 'DOGE/USDT', 'AVAX/USDT', 'TRX/USDT', 'LINK/USDT']
DAYS = 800  # over-fetch slightly for buffer


def fetch_daily(ex: ccxt.Exchange, symbol: str, days: int) -> pd.DataFrame:
    """Fetch last N days of daily OHLCV via paginated calls."""
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - days * 86_400_000
    all_rows = []
    cursor = start_ms
    while True:
        try:
            chunk = ex.fetch_ohlcv(symbol, '1d', since=cursor, limit=1000)
        except Exception as e:
            print(f'  fetch error {symbol} cursor={cursor}: {type(e).__name__}: {e}')
            time.sleep(2)
            continue
        if not chunk:
            break
        all_rows.extend(chunk)
        last_ts = chunk[-1][0]
        if last_ts >= end_ms - 86_400_000:
            break
        cursor = last_ts + 86_400_000
        time.sleep(0.2)  # be kind to API
    if not all_rows:
        return pd.DataFrame()
    df = pd.DataFrame(all_rows, columns=['ts_ms', 'open', 'high', 'low', 'close', 'volume'])
    df = df.drop_duplicates(subset='ts_ms').sort_values('ts_ms').reset_index(drop=True)
    df['symbol'] = symbol
    df['date'] = pd.to_datetime(df['ts_ms'], unit='ms', utc=True).dt.date
    return df[['date', 'symbol', 'open', 'high', 'low', 'close', 'volume', 'ts_ms']]


def main():
    ex = ccxt.binance({'options': {'defaultType': 'spot'}})
    ex.load_markets()
    all_dfs = []
    for sym in UNIVERSE:
        if sym not in ex.markets:
            print(f'! market not found: {sym}')
            continue
        print(f'Fetching {sym}...', end=' ', flush=True)
        df = fetch_daily(ex, sym, DAYS)
        if df.empty:
            print('FAILED')
            continue
        print(f'{len(df)} bars, {df.date.min()} → {df.date.max()}')
        all_dfs.append(df)

    combined = pd.concat(all_dfs, ignore_index=True)
    combined.to_parquet(OUT_FILE, index=False)
    print(f'\n=== summary ===')
    print(f'rows: {len(combined):,}')
    print(f'symbols: {sorted(combined.symbol.unique().tolist())}')
    pivot = combined.groupby('symbol')['date'].agg(['min', 'max', 'count'])
    print(pivot)
    print(f'saved: {OUT_FILE}')


if __name__ == '__main__':
    main()
