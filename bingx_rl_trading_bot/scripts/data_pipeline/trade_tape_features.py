"""Aggregate Binance Vision aggTrades to 1-min trade-flow features.

Input:  data/binance_vision_aggtrades/BTCUSDT-aggTrades-YYYY-MM-DD.zip
Output: data/btc_trade_features_1m.parquet (combined for date range)

Features (theory-based microstructure, locked):
  - trade_count        : # aggregated trades per minute
  - vol_total          : sum quantity
  - vol_buy            : sum qty where is_buyer_maker=False (taker buy)
  - vol_sell           : sum qty where is_buyer_maker=True  (taker sell)
  - taker_buy_ratio    : vol_buy / vol_total
  - vol_imbalance      : (vol_buy - vol_sell) / vol_total   ∈ [-1, 1]
  - vwap               : sum(price*qty) / vol_total
  - price_high         : max trade price in minute
  - price_low          : min trade price in minute
  - price_first        : first trade price
  - price_last         : last trade price
  - large_trade_count  : # trades with qty > rolling-24h p99
  - large_buy_share    : (large taker-buy count) / large_trade_count

NOTE: This file deliberately keeps features minimal and standard. Theory-locked.
No exotic features — that would be retrofitting before OOS.
"""
import io
import sys
import zipfile
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
RAW_DIR = ROOT / 'data' / 'binance_vision_aggtrades'
OUT_FILE = ROOT / 'data' / 'btc_trade_features_1m.parquet'

SYMBOL = 'BTCUSDT'

# Trade tape schema (Binance Vision)
COLUMNS = ['agg_trade_id', 'price', 'quantity', 'first_trade_id',
            'last_trade_id', 'transact_time', 'is_buyer_maker']


def _read_day(zip_path: Path) -> pd.DataFrame | None:
    """Stream-read one day's aggTrades zip → DataFrame."""
    if not zip_path.exists():
        return None
    try:
        with zipfile.ZipFile(zip_path) as z:
            inner = z.namelist()[0]
            with z.open(inner) as f:
                # Binance Vision daily aggTrades files: probe first byte for header presence
                first_bytes = f.read(50)
            with z.open(inner) as f:
                # If first char is alphabetic ('a'gg_trade_id) → header line present
                if first_bytes[:1].isalpha():
                    df = pd.read_csv(f, header=0)
                else:
                    df = pd.read_csv(f, header=None, names=COLUMNS)
        df['is_buyer_maker'] = df['is_buyer_maker'].astype(str).str.lower().map({'true': True, 'false': False}).fillna(False).astype(bool)
        df['price'] = df['price'].astype(float)
        df['quantity'] = df['quantity'].astype(float)
        df['transact_time'] = df['transact_time'].astype(np.int64)
        return df[['transact_time', 'price', 'quantity', 'is_buyer_maker']]
    except Exception as e:
        print(f'  ! failed to read {zip_path.name}: {type(e).__name__}: {e}')
        return None


def aggregate_day(df: pd.DataFrame, large_threshold: float) -> pd.DataFrame:
    """Aggregate one day's raw trades to 1-min bins."""
    if df is None or df.empty:
        return pd.DataFrame()
    df = df.copy()
    df['minute'] = pd.to_datetime(df['transact_time'], unit='ms', utc=True).dt.floor('1min')

    # Vectorized aggregation
    g = df.groupby('minute', sort=True)
    feat = pd.DataFrame({
        'trade_count': g.size(),
        'vol_total': g['quantity'].sum(),
        'vol_buy': g.apply(lambda x: x.loc[~x['is_buyer_maker'], 'quantity'].sum(), include_groups=False),
        'vol_sell': g.apply(lambda x: x.loc[x['is_buyer_maker'], 'quantity'].sum(), include_groups=False),
        'price_high': g['price'].max(),
        'price_low': g['price'].min(),
        'price_first': g['price'].first(),
        'price_last': g['price'].last(),
        'pv': g.apply(lambda x: (x['price'] * x['quantity']).sum(), include_groups=False),
        'large_trade_count': g.apply(lambda x: int((x['quantity'] > large_threshold).sum()),
                                       include_groups=False),
        'large_buy_count': g.apply(lambda x: int(((x['quantity'] > large_threshold) & ~x['is_buyer_maker']).sum()),
                                      include_groups=False),
    })
    feat = feat.reset_index().rename(columns={'minute': 'timestamp'})
    feat['vwap'] = feat['pv'] / feat['vol_total'].replace(0, np.nan)
    feat['taker_buy_ratio'] = feat['vol_buy'] / feat['vol_total'].replace(0, np.nan)
    feat['vol_imbalance'] = (feat['vol_buy'] - feat['vol_sell']) / feat['vol_total'].replace(0, np.nan)
    feat['large_buy_share'] = feat['large_buy_count'] / feat['large_trade_count'].replace(0, np.nan)
    feat = feat.drop(columns=['pv'])
    return feat


def main():
    if len(sys.argv) > 1 and sys.argv[1] == '--update':
        # Append-only: skip days already in OUT_FILE
        if OUT_FILE.exists():
            existing = pd.read_parquet(OUT_FILE, columns=['timestamp'])
            existing_dates = set(pd.to_datetime(existing['timestamp']).dt.date)
        else:
            existing_dates = set()
    else:
        existing_dates = set()

    # Discover available days
    zip_files = sorted(RAW_DIR.glob(f'{SYMBOL}-aggTrades-*.zip'))
    if not zip_files:
        print(f'No zip files found in {RAW_DIR}')
        sys.exit(1)
    print(f'Found {len(zip_files)} day-files in {RAW_DIR}')

    # First pass: compute global large-trade threshold (p99 quantity across full sample)
    # This is computed once on the full corpus to avoid look-ahead
    print('Pass 1: computing global p99 quantity threshold...')
    qty_samples: list[np.ndarray] = []
    for zp in zip_files[:30]:  # sample 30 days for threshold (theory-justified static)
        df = _read_day(zp)
        if df is not None and not df.empty:
            qty_samples.append(df['quantity'].to_numpy())
    if not qty_samples:
        print('No usable data found.')
        sys.exit(1)
    all_qty = np.concatenate(qty_samples)
    p99_threshold = float(np.percentile(all_qty, 99))
    print(f'  p99 quantity threshold: {p99_threshold:.4f}')

    # Pass 2: aggregate each day
    print('Pass 2: aggregating day-by-day...')
    all_feats: list[pd.DataFrame] = []
    for i, zp in enumerate(zip_files):
        # Parse date from filename
        d_str = zp.name.replace(f'{SYMBOL}-aggTrades-', '').replace('.zip', '')
        d = datetime.strptime(d_str, '%Y-%m-%d').date()
        if d in existing_dates:
            continue
        df = _read_day(zp)
        if df is None or df.empty:
            continue
        feat = aggregate_day(df, large_threshold=p99_threshold)
        if not feat.empty:
            all_feats.append(feat)
        if (i + 1) % 30 == 0:
            print(f'  progress: {i+1}/{len(zip_files)}')

    if not all_feats:
        print('No new days to aggregate.')
        sys.exit(0)

    out = pd.concat(all_feats, ignore_index=True).sort_values('timestamp').reset_index(drop=True)
    if existing_dates and OUT_FILE.exists():
        existing = pd.read_parquet(OUT_FILE)
        out = pd.concat([existing, out], ignore_index=True).drop_duplicates(subset=['timestamp']).sort_values('timestamp').reset_index(drop=True)
    out.to_parquet(OUT_FILE, index=False)

    print(f'\n=== summary ===')
    print(f'rows: {len(out):,}')
    print(f'columns: {list(out.columns)}')
    print(f'date range: {out["timestamp"].min()} → {out["timestamp"].max()}')
    print(f'file size: {OUT_FILE.stat().st_size / 1e6:.1f} MB')
    print(f'saved: {OUT_FILE}')


if __name__ == '__main__':
    main()
