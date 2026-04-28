"""Fetch historical APY panel from DefiLlama (free public API).

Per advisor delegation 2026-04-29 (DeFi-Track Week 1):
- Major lending/LP protocols: Aave (V2/V3), Compound, Curve, Convex, Pendle
- Build panel: pool × date × apy × tvlUsd, last 12-24 months
- Cohort analysis support: track first_seen / last_seen / zero-month rates

API endpoints:
- GET https://yields.llama.fi/pools  → current snapshot of all pools
- GET https://yields.llama.fi/chart/{pool_id} → historical apy/tvl for one pool

Output:
- data/defi_yields_pools.parquet  (one row per pool, current state)
- data/defi_yields_panel.parquet  (long-format: pool_id × date × apy × tvlUsd)
- data/defi_yields_cohort.parquet (per-pool first/last/min/max apy + lifetime)

No paid API. No auth. Standard urllib + json (no requests dependency added).
"""
import json
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT / 'data'
DATA_DIR.mkdir(exist_ok=True)

POOLS_FILE = DATA_DIR / 'defi_yields_pools.parquet'
PANEL_FILE = DATA_DIR / 'defi_yields_panel.parquet'
COHORT_FILE = DATA_DIR / 'defi_yields_cohort.parquet'

# Locked pre-reg targets (advisor: major lending/LP)
TARGET_PROJECTS = {
    'aave-v2', 'aave-v3',
    'compound', 'compound-v3',
    'curve-dex',
    'convex-finance',
    'pendle',
}

# Filter to USD-denominated stablecoin/major-asset pools (avoid exotic tail)
TARGET_SYMBOLS_KEEP_PREFIXES = (
    'USDC', 'USDT', 'DAI', 'FRAX', 'LUSD', 'GUSD', 'TUSD', 'USDD',
    'WETH', 'ETH', 'STETH', 'WSTETH', 'CBETH', 'RETH',
    'WBTC', 'BTC',
)

# Min TVL filter (avoid tiny/dead pools where APY is meaningless)
MIN_TVL_USD = 1_000_000  # $1M

POOLS_URL = 'https://yields.llama.fi/pools'
CHART_URL = 'https://yields.llama.fi/chart/{pool_id}'

USER_AGENT = 'CLAUDE_CODE_FIN/defi-track-research (research-only, low-rate)'


def http_get_json(url: str, retries: int = 5) -> dict | list:
    last_err = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={'User-Agent': USER_AGENT})
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode('utf-8'))
        except (urllib.error.URLError, urllib.error.HTTPError, json.JSONDecodeError) as e:
            last_err = e
            wait = 2 ** attempt
            print(f'  retry {attempt+1}/{retries} after {wait}s: {type(e).__name__}: {e}', flush=True)
            time.sleep(wait)
    raise RuntimeError(f'HTTP failed after {retries}: {last_err}')


def fetch_all_pools() -> pd.DataFrame:
    print(f'Fetching pools snapshot from {POOLS_URL} ...', flush=True)
    payload = http_get_json(POOLS_URL)
    if isinstance(payload, dict) and 'data' in payload:
        rows = payload['data']
    else:
        rows = payload
    df = pd.DataFrame(rows)
    print(f'  total pools: {len(df):,}', flush=True)
    return df


def filter_pools(pools: pd.DataFrame) -> pd.DataFrame:
    if 'project' not in pools.columns:
        raise RuntimeError(f'unexpected schema: {pools.columns.tolist()}')

    f = pools[pools['project'].isin(TARGET_PROJECTS)].copy()
    print(f'  after project filter: {len(f):,}', flush=True)

    if 'tvlUsd' in f.columns:
        f = f[f['tvlUsd'].fillna(0) >= MIN_TVL_USD]
        print(f'  after TVL >= ${MIN_TVL_USD:,}: {len(f):,}', flush=True)

    def keep_symbol(sym):
        if not isinstance(sym, str):
            return False
        return any(sym.upper().startswith(p) for p in TARGET_SYMBOLS_KEEP_PREFIXES)

    if 'symbol' in f.columns:
        f = f[f['symbol'].apply(keep_symbol)]
        print(f'  after symbol prefix filter: {len(f):,}', flush=True)

    return f.reset_index(drop=True)


def fetch_pool_history(pool_id: str) -> pd.DataFrame:
    url = CHART_URL.format(pool_id=pool_id)
    payload = http_get_json(url)
    if isinstance(payload, dict) and 'data' in payload:
        rows = payload['data']
    else:
        rows = payload
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if 'timestamp' not in df.columns:
        return pd.DataFrame()
    df['date'] = pd.to_datetime(df['timestamp'], utc=True).dt.date
    df['pool_id'] = pool_id
    keep = ['pool_id', 'date', 'tvlUsd', 'apy', 'apyBase', 'apyReward']
    keep = [c for c in keep if c in df.columns]
    return df[keep].drop_duplicates(subset=['pool_id', 'date'], keep='last')


def build_cohort(panel: pd.DataFrame, pools: pd.DataFrame) -> pd.DataFrame:
    g = panel.groupby('pool_id')
    cohort = g.agg(
        first_date=('date', 'min'),
        last_date=('date', 'max'),
        n_days=('date', 'nunique'),
        apy_mean=('apy', 'mean'),
        apy_median=('apy', 'median'),
        apy_min=('apy', 'min'),
        apy_max=('apy', 'max'),
        apy_std=('apy', 'std'),
        tvl_mean=('tvlUsd', 'mean'),
        tvl_min=('tvlUsd', 'min'),
        zero_apy_days=('apy', lambda s: int((s.fillna(0) <= 0.01).sum())),
    ).reset_index()
    meta_cols = [c for c in ['pool', 'project', 'chain', 'symbol', 'stablecoin', 'ilRisk', 'exposure']
                 if c in pools.columns]
    meta = pools[['pool'] + [c for c in meta_cols if c != 'pool']].rename(columns={'pool': 'pool_id'})
    cohort = cohort.merge(meta, on='pool_id', how='left')
    cohort['lifetime_days'] = (
        pd.to_datetime(cohort['last_date']) - pd.to_datetime(cohort['first_date'])
    ).dt.days
    cohort['zero_apy_rate'] = cohort['zero_apy_days'] / cohort['n_days'].clip(lower=1)
    return cohort


def main():
    print(f'=== DefiLlama yield panel fetch — {datetime.now(timezone.utc).isoformat()} ===', flush=True)
    pools = fetch_all_pools()
    pools.to_parquet(POOLS_FILE, index=False)
    print(f'saved snapshot: {POOLS_FILE}', flush=True)

    filtered = filter_pools(pools)
    if filtered.empty:
        print('!! no pools after filter — abort')
        return 1

    print(f'\nFetching history for {len(filtered)} pools ...', flush=True)
    panel_dfs = []
    fail = 0
    for i, row in filtered.iterrows():
        pid = row['pool']
        proj = row.get('project', '?')
        sym = row.get('symbol', '?')
        chain = row.get('chain', '?')
        try:
            h = fetch_pool_history(pid)
        except Exception as e:
            fail += 1
            print(f'  [{i+1}/{len(filtered)}] FAIL {proj}/{chain}/{sym} ({pid}): {e}', flush=True)
            continue
        if h.empty:
            fail += 1
            print(f'  [{i+1}/{len(filtered)}] EMPTY {proj}/{chain}/{sym} ({pid})', flush=True)
            continue
        panel_dfs.append(h)
        if (i + 1) % 25 == 0 or (i + 1) == len(filtered):
            print(f'  [{i+1}/{len(filtered)}] {proj}/{chain}/{sym} → {len(h)} rows', flush=True)
        time.sleep(0.15)  # rate-limit kindness

    if not panel_dfs:
        print('!! no historical data fetched')
        return 1

    panel = pd.concat(panel_dfs, ignore_index=True)
    panel = panel.sort_values(['pool_id', 'date']).reset_index(drop=True)
    panel.to_parquet(PANEL_FILE, index=False)
    print(f'\nsaved panel: {PANEL_FILE}  ({len(panel):,} rows, {panel.pool_id.nunique()} pools)', flush=True)

    cohort = build_cohort(panel, filtered)
    cohort.to_parquet(COHORT_FILE, index=False)
    print(f'saved cohort: {COHORT_FILE}', flush=True)

    print('\n=== summary ===')
    print(f'pools targeted     : {len(filtered)}')
    print(f'pools with history : {panel.pool_id.nunique()}')
    print(f'fetch failures     : {fail}')
    print(f'panel date range   : {panel.date.min()} → {panel.date.max()}')
    print(f'panel rows         : {len(panel):,}')
    print(f'\nper-project:')
    print(cohort.groupby('project').agg(
        pools=('pool_id', 'count'),
        median_apy=('apy_median', 'median'),
        median_lifetime_days=('lifetime_days', 'median'),
        median_zero_apy_rate=('zero_apy_rate', 'median'),
    ).to_string())
    return 0


if __name__ == '__main__':
    sys.exit(main())
