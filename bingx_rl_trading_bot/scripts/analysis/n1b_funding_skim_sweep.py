"""N1b — Funding Skim SWEEP (wider grid retry).

N1 surface: 0/7 sweep FAIL with narrow params.
Wider sweep (entry/exit thresholds, max_hold, friction sensitivity).

Pre-registered grid:
  entry_threshold_pct:  [0.01, 0.02, 0.04, 0.08]
  exit_threshold_pct:   [0.005, 0.01, 0.02]
  max_hold_periods:     [9, 21, 42]   # 3d, 7d, 14d
  capital_usd:          [1000]        # FIXED
= 4×3×3 = 36 configs

Multi-stage 50/25/25.
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

CAPITAL = 1000
UTIL = 1.0
MAKER_FRIC = 0.04
SLIP = 0.02
LEG = 2
SIDE = 2

_FUND_CACHE = None


def load_funding():
    global _FUND_CACHE
    if _FUND_CACHE is None:
        df = pd.read_csv(DATA / 'c2_funding_history.csv', parse_dates=['timestamp'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
        df['fund_pct'] = df['funding_rate'] * 100
        _FUND_CACHE = df.sort_values(['symbol', 'timestamp']).reset_index(drop=True)
    return _FUND_CACHE


def simulate(df, params):
    leg_notional = CAPITAL * UTIL / LEG
    fric_per_side = MAKER_FRIC + SLIP
    fric_per_trade = LEG * SIDE * fric_per_side / 100 * leg_notional

    entry_thr = params['entry_threshold_pct']
    exit_thr = params['exit_threshold_pct']
    max_hold = params['max_hold_periods']

    trades = []
    for symbol, group in df.groupby('symbol'):
        group = group.sort_values('timestamp').reset_index(drop=True)
        active = None
        for i, row in group.iterrows():
            rate_pct = row['fund_pct']
            ts = row['timestamp']
            if active is None:
                if abs(rate_pct) > entry_thr:
                    side = 'short_perp' if rate_pct > 0 else 'long_perp'
                    active = {'side': side, 'enter_ts': ts, 'enter_idx': i, 'cum': 0.0}
            else:
                if active['side'] == 'short_perp':
                    period_pnl = rate_pct / 100 * leg_notional
                else:
                    period_pnl = -rate_pct / 100 * leg_notional
                active['cum'] += period_pnl
                held = i - active['enter_idx']
                exit_now = (held >= max_hold) or (abs(rate_pct) < exit_thr)
                if exit_now:
                    net = active['cum'] - fric_per_trade
                    trades.append({
                        'close_ts': ts,
                        'gross_pct': active['cum'] / CAPITAL * 100,
                        'net_pnl_pct': net / CAPITAL * 100,
                    })
                    active = None
    return pd.DataFrame(trades)


class N1bSweep(MechanismSweep):
    label = 'n1b_funding_skim'
    mechanism_description = 'N1b — Funding Skim wider sweep'
    TS_COL = 'timestamp'
    PARAM_GRID = {
        'entry_threshold_pct':  [0.01, 0.02, 0.04, 0.08],
        'exit_threshold_pct':   [0.005, 0.01, 0.02],
        'max_hold_periods':     [9, 21, 42],
    }

    def build_trades(self, df_segment, config):
        full = load_funding()
        ts_min = df_segment[self.TS_COL].min()
        ts_max = df_segment[self.TS_COL].max()
        seg = full[(full['timestamp'] >= ts_min) & (full['timestamp'] <= ts_max)]
        return simulate(seg, config)


def main():
    df = load_funding()
    print(f'Funding records: {len(df):,}, {df.timestamp.min()} → {df.timestamp.max()}')
    df_seg = pd.DataFrame({'timestamp': df['timestamp'].unique()}).sort_values('timestamp').reset_index(drop=True)
    sweep = N1bSweep()
    result = sweep.run_sweep(df_seg, RESULTS)
    if not result.deployable:
        print('\n→ N1b sweep: 0 OOS-passing configs.')


if __name__ == '__main__':
    main()
