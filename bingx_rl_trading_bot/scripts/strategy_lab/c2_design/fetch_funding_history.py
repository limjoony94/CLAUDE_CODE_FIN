"""C2 Design — 8-coin funding rate history fetcher.

Fetch BingX funding rate history for: BTC, ETH, SOL, BNB, XRP, DOGE, AVAX, LINK
Goal: 1+ year of 8h funding rates per coin → ~1095 data points per coin.

BingX API: ccxt fetch_funding_rate_history (paginated, 1000 records per call).

Output: data/c2_funding_history.parquet (or csv) — long format
  columns: timestamp_utc, symbol, funding_rate
"""
import time
import yaml
from pathlib import Path

import pandas as pd
import ccxt


COINS = ['BTC', 'ETH', 'SOL', 'BNB', 'XRP', 'DOGE', 'AVAX', 'LINK']


def get_exchange():
    project_root = Path(__file__).resolve().parents[3]
    with open(project_root / 'config' / 'api_keys.yaml') as f:
        keys = yaml.safe_load(f)
    return ccxt.bingx({
        'apiKey': keys['bingx']['mainnet']['api_key'],
        'secret': keys['bingx']['mainnet']['secret_key'],
        'enableRateLimit': True,
        'options': {'defaultType': 'swap'},
    })


def fetch_funding_history(ex, symbol: str, days_back: int = 400) -> pd.DataFrame:
    """Fetch funding rate history paginated."""
    since = int((pd.Timestamp.utcnow() - pd.Timedelta(days=days_back)).timestamp() * 1000)
    all_records = []
    fail_count = 0
    while True:
        try:
            records = ex.fetch_funding_rate_history(symbol, since=since, limit=1000)
        except Exception as e:
            print(f'  [{symbol}] API error: {e}')
            fail_count += 1
            if fail_count > 3:
                break
            time.sleep(2)
            continue
        if not records:
            break
        all_records.extend(records)
        if len(records) < 1000:
            break
        # advance since
        since = records[-1]['timestamp'] + 1
        time.sleep(0.3)

    if not all_records:
        return pd.DataFrame()
    df = pd.DataFrame(all_records)
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df['symbol'] = symbol
    df = df[['timestamp', 'symbol', 'fundingRate']].rename(columns={'fundingRate': 'funding_rate'})
    df = df.drop_duplicates('timestamp').sort_values('timestamp').reset_index(drop=True)
    return df


def main():
    ex = get_exchange()
    ex.load_markets()
    project_root = Path(__file__).resolve().parents[3]
    out_path = project_root / 'data' / 'c2_funding_history.csv'
    out_path.parent.mkdir(parents=True, exist_ok=True)

    all_dfs = []
    for coin in COINS:
        symbol = f'{coin}/USDT:USDT'
        if symbol not in ex.markets:
            print(f'  [{coin}] symbol {symbol} not in markets, skipping')
            continue
        print(f'Fetching {coin} funding history...')
        df = fetch_funding_history(ex, symbol, days_back=400)
        if df.empty:
            print(f'  [{coin}] empty, skipping')
            continue
        print(f'  [{coin}] {len(df):,} records, {df.timestamp.min()} → {df.timestamp.max()}')
        all_dfs.append(df)
        time.sleep(1)

    combined = pd.concat(all_dfs, ignore_index=True)
    combined.to_csv(out_path, index=False)
    print(f'\nSaved: {out_path}')

    # Summary
    print('\n=== Summary ===')
    summ = combined.groupby('symbol').agg(
        n=('funding_rate', 'count'),
        first=('timestamp', 'min'),
        last=('timestamp', 'max'),
        mean_funding_pct=('funding_rate', lambda x: x.mean() * 100),
        std_funding_pct=('funding_rate', lambda x: x.std() * 100),
    )
    print(summ.to_string())


if __name__ == '__main__':
    main()
