"""Fetch funding history + daily price for 30-coin universe via Bybit linear perp.

Pre-reg: claudedocs/path_b_r4_funding_carry_30coin_prereg.md (commit dac672f)

Universe LOCKED (committed pre-data-fetch):
  10 original:   BTC ETH SOL BNB XRP ADA DOGE AVAX TRX LINK
  10 mid-cap:    DOT MATIC LTC SHIB BCH ATOM UNI NEAR ICP FIL
  10 large-alt:  APT AAVE ARB OP INJ SUI TIA FTM ALGO SAND

Filter: ≥600d Bybit funding history. Drop list reported.

Outputs:
  data/bybit_funding_history.parquet  (long-format funding rates)
  data/bybit_daily_prices.parquet     (long-format OHLCV)
  data/bybit_universe_drop_list.json  (coins dropped + reason)
"""
import json
import sys
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import ccxt
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
DATA = ROOT / 'data'
DATA.mkdir(exist_ok=True)

FUND_OUT = DATA / 'bybit_funding_history.parquet'
PRICE_OUT = DATA / 'bybit_daily_prices.parquet'
DROP_LIST = DATA / 'bybit_universe_drop_list.json'

UNIVERSE = [
    'BTC', 'ETH', 'SOL', 'BNB', 'XRP', 'ADA', 'DOGE', 'AVAX', 'TRX', 'LINK',
    'DOT', 'MATIC', 'LTC', 'SHIB', 'BCH', 'ATOM', 'UNI', 'NEAR', 'ICP', 'FIL',
    'APT', 'AAVE', 'ARB', 'OP', 'INJ', 'SUI', 'TIA', 'FTM', 'ALGO', 'SAND',
]

DAYS = 800
MIN_HISTORY_DAYS = 600


def fetch_funding(ex: ccxt.Exchange, symbol: str, days: int) -> pd.DataFrame:
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - days * 86_400_000
    all_rows = []
    cursor = start_ms
    while True:
        try:
            chunk = ex.fetch_funding_rate_history(symbol, since=cursor, limit=200)
        except Exception as e:
            print(f'    fetch error cursor={cursor}: {type(e).__name__}: {e}')
            time.sleep(2)
            return pd.DataFrame()
        if not chunk:
            break
        all_rows.extend(chunk)
        last_ts = chunk[-1]['timestamp']
        if last_ts >= end_ms - 8 * 3_600_000:
            break
        cursor = last_ts + 8 * 3_600_000
        time.sleep(0.15)
    if not all_rows:
        return pd.DataFrame()
    rows = []
    for r in all_rows:
        rows.append({
            'ts_ms': r['timestamp'],
            'symbol': r['symbol'],
            'funding_rate': float(r['fundingRate']) if r.get('fundingRate') is not None else None,
        })
    df = pd.DataFrame(rows).drop_duplicates(subset='ts_ms').sort_values('ts_ms').reset_index(drop=True)
    df['datetime'] = pd.to_datetime(df['ts_ms'], unit='ms', utc=True)
    df['date'] = df['datetime'].dt.date
    return df


def fetch_daily(ex: ccxt.Exchange, symbol: str, days: int) -> pd.DataFrame:
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - days * 86_400_000
    all_rows = []
    cursor = start_ms
    while True:
        try:
            chunk = ex.fetch_ohlcv(symbol, '1d', since=cursor, limit=1000)
        except Exception as e:
            print(f'    fetch error cursor={cursor}: {type(e).__name__}: {e}')
            time.sleep(2)
            return pd.DataFrame()
        if not chunk:
            break
        all_rows.extend(chunk)
        last_ts = chunk[-1][0]
        if last_ts >= end_ms - 86_400_000:
            break
        cursor = last_ts + 86_400_000
        time.sleep(0.1)
    if not all_rows:
        return pd.DataFrame()
    df = pd.DataFrame(all_rows, columns=['ts_ms', 'open', 'high', 'low', 'close', 'volume'])
    df = df.drop_duplicates(subset='ts_ms').sort_values('ts_ms').reset_index(drop=True)
    df['date'] = pd.to_datetime(df['ts_ms'], unit='ms', utc=True).dt.date
    return df


def main():
    ex = ccxt.bybit({'options': {'defaultType': 'swap'}})
    ex.load_markets()

    fund_dfs = []
    price_dfs = []
    drop_list = {}

    for base in UNIVERSE:
        sym = f'{base}/USDT:USDT'
        if sym not in ex.markets:
            print(f'! {base}: market {sym} not on Bybit')
            drop_list[base] = 'market_not_found'
            continue

        print(f'Fetching {base}...')
        # Funding
        f = fetch_funding(ex, sym, DAYS)
        if f.empty or len(f) < MIN_HISTORY_DAYS * 3:
            n_periods = len(f) if not f.empty else 0
            n_days = n_periods / 3 if n_periods > 0 else 0
            print(f'  funding insufficient: {n_periods} periods = {n_days:.0f}d (gate ≥{MIN_HISTORY_DAYS}d)')
            drop_list[base] = f'funding_insufficient_{n_days:.0f}d'
            continue
        f['symbol'] = base + '/USDT'
        n_funding_days = len(f) / 3
        # Price
        p = fetch_daily(ex, sym, DAYS)
        if p.empty or len(p) < MIN_HISTORY_DAYS:
            n_p = len(p) if not p.empty else 0
            print(f'  price insufficient: {n_p}d (gate ≥{MIN_HISTORY_DAYS}d)')
            drop_list[base] = f'price_insufficient_{n_p}d'
            continue
        p['symbol'] = base + '/USDT'

        fund_dfs.append(f)
        price_dfs.append(p)
        print(f'  ✓ {n_funding_days:.0f}d funding × {len(p)}d price, '
              f'fund mean={f.funding_rate.mean()*100:.4f}%/8h')

    if not fund_dfs:
        print('!! no coins survived filter')
        return 1

    funding = pd.concat(fund_dfs, ignore_index=True)
    funding.to_parquet(FUND_OUT, index=False)
    prices = pd.concat(price_dfs, ignore_index=True)
    prices.to_parquet(PRICE_OUT, index=False)

    with open(DROP_LIST, 'w') as fp:
        json.dump({
            'universe_target': UNIVERSE,
            'min_history_days': MIN_HISTORY_DAYS,
            'kept': sorted(set([b for b in UNIVERSE if b not in drop_list])),
            'dropped': drop_list,
            'final_count': len(UNIVERSE) - len(drop_list),
        }, fp, indent=2, default=str)

    print(f'\n=== summary ===')
    print(f'kept {len(UNIVERSE) - len(drop_list)}/{len(UNIVERSE)} coins')
    print(f'dropped: {drop_list}')
    print(f'\nfunding rows: {len(funding):,}')
    print(f'price rows:   {len(prices):,}')
    print(f'\nfunding date range: {funding.date.min()} → {funding.date.max()}')
    print(f'price date range:   {prices.date.min()} → {prices.date.max()}')
    print(f'\nsaved: {FUND_OUT}')
    print(f'saved: {PRICE_OUT}')
    print(f'saved: {DROP_LIST}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
