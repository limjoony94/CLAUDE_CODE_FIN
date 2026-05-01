"""R2b — Path B R2 XS Reversal SWEEP (10 coins, daily, mirror of R1b).

Path B R2 surface: vacuous (4.64% dispersion < 5% gate).
Sweep retry — wider param + non-vacuous lookbacks.

Pre-registered grid:
  lookback_days:   [3, 5, 7, 14]    # short-term reversal
  long_top_n:      [2, 3]           # long the LOSERS (bottom)
  short_bottom_n:  [0, 2, 3]        # short the WINNERS (top)
  rebalance_days:  [1, 3, 7]
= 4×2×3×3 = 72 configs

Friction LOCKED 0.07%/transaction.
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts' / 'strategy_lab'))
sys.path.insert(0, str(ROOT / 'scripts' / 'analysis'))
from mechanism_sweep_standard import MechanismSweep
from r1b_xs_momentum_sweep import load_pivot, FRICTION_PCT

DATA = ROOT / 'data'
RESULTS = ROOT / 'results'


def run_xs_reversal(prices, params):
    """REVERSE signal: long the bottom, short the top."""
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
            # REVERSAL: long bottom, short top
            cur_long = ranks.tail(n_long).index.tolist()
            cur_short = ranks.head(n_short).index.tolist() if n_short > 0 else []
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
    out = out[out['turnover'] > 0].reset_index(drop=True)
    return out[['close_ts', 'gross_pct', 'net_pnl_pct']]


class R2bSweep(MechanismSweep):
    label = 'r2b_xs_reversal'
    mechanism_description = 'R2b — Path B R2 XS Reversal (10 coins, daily, mirror)'
    TS_COL = 'date'
    PARAM_GRID = {
        'lookback_days':   [3, 5, 7, 14],
        'long_top_n':      [2, 3],
        'short_bottom_n':  [0, 2, 3],
        'rebalance_days':  [1, 3, 7],
    }

    def build_trades(self, df_segment, config):
        prices = load_pivot()
        ts_min = df_segment[self.TS_COL].min()
        ts_max = df_segment[self.TS_COL].max()
        prices_seg = prices[(prices.index >= ts_min) & (prices.index <= ts_max)]
        if len(prices_seg) < config['lookback_days'] + 7:
            return pd.DataFrame()
        return run_xs_reversal(prices_seg, config)


def main():
    prices = load_pivot()
    print(f'Daily prices: {prices.shape[0]} days × {prices.shape[1]} coins')
    df = pd.DataFrame({'date': prices.index})
    sweep = R2bSweep()
    result = sweep.run_sweep(df, RESULTS)
    if not result.deployable:
        print('\n→ R2b sweep: 0 OOS-passing configs.')


if __name__ == '__main__':
    main()
