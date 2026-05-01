"""R1b — Path B R1 Cross-Sectional Momentum SWEEP (10 coins, daily).

Path B R1 surface: +0.13%/wk net, 'first round with edge > friction' borderline.
Sweep retry — most promising mechanism per master plan.

Pre-registered grid:
  lookback_days:   [14, 30, 60, 90]
  long_top_n:      [2, 3, 4]
  short_bottom_n:  [0, 2, 3]      # 0 = long-only
  rebalance_days:  [3, 7, 14]
= 4×3×3×3 = 108 configs

Friction LOCKED at 0.07% per transaction.
Multi-stage 50/25/25.

Trade-like format: each day's portfolio return = 1 "trade" for bootstrap framework.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts' / 'strategy_lab'))
from mechanism_sweep_standard import MechanismSweep


DATA = ROOT / 'data'
RESULTS = ROOT / 'results'
FRICTION_PCT = 0.07


UNIVERSE = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT', 'XRP/USDT',
            'ADA/USDT', 'DOGE/USDT', 'AVAX/USDT', 'TRX/USDT', 'LINK/USDT']

_PRICES_CACHE = None


def load_pivot():
    global _PRICES_CACHE
    if _PRICES_CACHE is None:
        df = pd.read_parquet(DATA / 'multi_asset_daily.parquet')
        df['date'] = pd.to_datetime(df['date'])
        pivot = df.pivot(index='date', columns='symbol', values='close').sort_index()
        pivot = pivot[[c for c in UNIVERSE if c in pivot.columns]]
        pivot = pivot.dropna(how='any')
        _PRICES_CACHE = pivot
    return _PRICES_CACHE


def run_xs_momentum(prices, params):
    look = params['lookback_days']
    rebal = params['rebalance_days']
    n_long = params['long_top_n']
    n_short = params['short_bottom_n']

    dates = prices.index
    n_dates = len(dates)
    if n_dates < look + 7:
        return pd.DataFrame()

    daily_ret = prices.pct_change().fillna(0)
    trail_ret = (prices / prices.shift(look) - 1) * 100

    pos = pd.DataFrame(0.0, index=dates, columns=prices.columns)

    last_rebal = -1
    cur_long = []
    cur_short = []
    for i in range(look, n_dates):
        if i - last_rebal >= rebal:
            ranks = trail_ret.iloc[i].dropna().sort_values(ascending=False)
            if len(ranks) < n_long + max(n_short, 1):
                continue
            cur_long = ranks.head(n_long).index.tolist()
            cur_short = ranks.tail(n_short).index.tolist() if n_short > 0 else []
            last_rebal = i
        for s in cur_long:
            pos.iat[i, prices.columns.get_loc(s)] = 1.0 / max(n_long, 1)
        if n_short > 0:
            for s in cur_short:
                pos.iat[i, prices.columns.get_loc(s)] = -1.0 / max(n_short, 1)

    pos_lag = pos.shift(1).fillna(0)
    port_ret_pct = (pos_lag * daily_ret).sum(axis=1) * 100

    turnover = (pos - pos_lag).abs().sum(axis=1)
    fric_pct = turnover * FRICTION_PCT

    out = pd.DataFrame({
        'close_ts': dates,
        'gross_pct': port_ret_pct.values,
        'net_pnl_pct': (port_ret_pct - fric_pct).values,
        'turnover': turnover.values,
    })
    # Drop warmup zero days
    out = out[out['turnover'] > 0].reset_index(drop=True)
    return out[['close_ts', 'gross_pct', 'net_pnl_pct']]


class R1bSweep(MechanismSweep):
    label = 'r1b_xs_momentum'
    mechanism_description = 'R1b — Path B R1 XS Momentum (10 coins, daily)'
    TS_COL = 'date'

    PARAM_GRID = {
        'lookback_days':   [14, 30, 60, 90],
        'long_top_n':      [2, 3, 4],
        'short_bottom_n':  [0, 2, 3],
        'rebalance_days':  [3, 7, 14],
    }

    def build_trades(self, df_segment, config):
        prices = load_pivot()
        ts_min = df_segment['date'].min()
        ts_max = df_segment['date'].max()
        prices_seg = prices[(prices.index >= ts_min) & (prices.index <= ts_max)]
        if len(prices_seg) < config['lookback_days'] + 7:
            return pd.DataFrame()
        return run_xs_momentum(prices_seg, config)


def main():
    prices = load_pivot()
    print(f'Daily prices: {prices.shape[0]} days × {prices.shape[1]} coins')
    print(f'Range: {prices.index.min().date()} → {prices.index.max().date()}')

    # Build df with date column for splitter
    df = pd.DataFrame({'date': prices.index})
    sweep = R1bSweep()
    result = sweep.run_sweep(df, RESULTS)
    if not result.deployable:
        print('\n→ R1b sweep: 0 OOS-passing configs.')
    else:
        print(f'\n→ R1b sweep: {result.oos_pass_count} OOS-passing configs.')


if __name__ == '__main__':
    main()
