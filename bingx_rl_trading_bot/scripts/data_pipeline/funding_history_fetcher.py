"""Fetch funding rate history for 10 crypto perp pairs (Binance USDT-M).

Pre-reg: claudedocs/path_b_r3_funding_carry_prereg.md (commit 4435c76)
Universe LOCKED (must match Path B R1/R2 for fair comparison):
  BTC/ETH/SOL/BNB/XRP/ADA/DOGE/AVAX/TRX/LINK / USDT

Funding rate paid every 8h on Binance perpetuals. Fetched via CCXT
fetch_funding_rate_history. Free, no auth.

Output: data/funding_history.parquet (long-format)
"""
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import ccxt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
OUT_FILE = ROOT / 'data' / 'funding_history.parquet'

UNIVERSE = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT', 'XRP/USDT',
            'ADA/USDT', 'DOGE/USDT', 'AVAX/USDT', 'TRX/USDT', 'LINK/USDT']

DAYS = 800  # match multi_asset_daily.parquet horizon


def fetch_funding(ex: ccxt.Exchange, symbol: str, days: int) -> pd.DataFrame:
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - days * 86_400_000
    all_rows = []
    cursor = start_ms
    while True:
        try:
            chunk = ex.fetch_funding_rate_history(symbol, since=cursor, limit=1000)
        except Exception as e:
            print(f'  fetch error {symbol} cursor={cursor}: {type(e).__name__}: {e}')
            time.sleep(2)
            continue
        if not chunk:
            break
        all_rows.extend(chunk)
        last_ts = chunk[-1]['timestamp']
        if last_ts >= end_ms - 8 * 3_600_000:
            break
        cursor = last_ts + 8 * 3_600_000
        time.sleep(0.2)
    if not all_rows:
        return pd.DataFrame()
    rows = []
    for r in all_rows:
        rows.append({
            'ts_ms': r['timestamp'],
            'symbol': r['symbol'],
            'funding_rate': float(r['fundingRate']) if r.get('fundingRate') is not None else None,
        })
    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset='ts_ms').sort_values('ts_ms').reset_index(drop=True)
    df['datetime'] = pd.to_datetime(df['ts_ms'], unit='ms', utc=True)
    df['date'] = df['datetime'].dt.date
    return df[['datetime', 'date', 'symbol', 'funding_rate', 'ts_ms']]


def main():
    ex = ccxt.binance({'options': {'defaultType': 'future'}})
    ex.load_markets()
    all_dfs = []
    for sym in UNIVERSE:
        perp_sym = sym + ':USDT'
        if perp_sym not in ex.markets and sym not in ex.markets:
            print(f'! market not found: {sym} or {perp_sym}')
            continue
        use_sym = perp_sym if perp_sym in ex.markets else sym
        print(f'Fetching funding {use_sym}...', end=' ', flush=True)
        df = fetch_funding(ex, use_sym, DAYS)
        if df.empty:
            print('FAILED')
            continue
        # Normalize symbol back to BTC/USDT format for join with daily price panel
        df['symbol'] = sym
        print(f'{len(df)} periods, {df.date.min()} → {df.date.max()}, '
              f'mean={df.funding_rate.mean()*100:.4f}%/8h, '
              f'std={df.funding_rate.std()*100:.4f}%')
        all_dfs.append(df)

    if not all_dfs:
        print('!! no data fetched')
        return 1

    combined = pd.concat(all_dfs, ignore_index=True)
    combined.to_parquet(OUT_FILE, index=False)
    print(f'\n=== summary ===')
    print(f'rows: {len(combined):,}')
    print(f'symbols: {sorted(combined.symbol.unique().tolist())}')
    pivot = combined.groupby('symbol').agg(
        first=('date', 'min'),
        last=('date', 'max'),
        n=('date', 'count'),
        mean_rate_per_8h=('funding_rate', 'mean'),
        std_rate_per_8h=('funding_rate', 'std'),
    )
    pivot['annualized_mean_pct'] = pivot['mean_rate_per_8h'] * 3 * 365 * 100
    print(pivot.round(6))
    print(f'\nsaved: {OUT_FILE}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
