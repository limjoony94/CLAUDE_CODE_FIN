"""R40b — Volume Absorption SWEEP (BTC 1h, R40 essence).

R40 surface: avg_gross +0.034%/trade < friction 0.07% at n=2,760.

Mechanism essence: high volume + small body = institutional absorption proxy
                   (Wyckoff theory).

Pre-registered grid:
  vol_lookback:        [10, 20, 50]
  vol_mult:            [1.5, 2.0, 3.0]
  body_ratio_max:      [0.20, 0.30, 0.40]   # absorption (small body)
  conf_body_min:       [0.30, 0.50]         # confirmation bar body
  sl_atr_mult:         [1.0, 2.0]
  tp_atr_mult:         [2.0, 3.0]
  max_hold_bars:       [12, 24]
= 3×3×3×2×2×2×2 = 432 configs
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
FRICTION_RT_PCT = 0.14


def compute_atr(df, n=14):
    high, low, close = df['high'].values, df['low'].values, df['close'].values
    tr = np.zeros(len(df))
    for i in range(1, len(df)):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    return pd.Series(tr).rolling(n).mean().values


def simulate_trades(df, params):
    df = df.reset_index(drop=True).copy()
    op = df['open'].values
    hi = df['high'].values
    lo = df['low'].values
    cl = df['close'].values
    vol = df['volume'].values
    ts = df['timestamp'].values

    vol_sma = pd.Series(vol).rolling(params['vol_lookback']).mean().values
    atr = compute_atr(df, 14)

    vol_mult = params['vol_mult']
    body_max = params['body_ratio_max']
    conf_body_min = params['conf_body_min']
    sl_mult = params['sl_atr_mult']
    tp_mult = params['tp_atr_mult']
    max_hold = params['max_hold_bars']

    n = len(df)
    trades = []
    in_pos = False
    pos = None
    for i in range(50, n - 2):
        if in_pos:
            held = i - pos['entry_idx']
            exit_price = None
            if pos['side'] == 'LONG':
                if hi[i] >= pos['tp']:
                    exit_price = pos['tp']
                elif lo[i] <= pos['sl']:
                    exit_price = pos['sl']
                elif held >= max_hold:
                    exit_price = cl[i]
            else:
                if lo[i] <= pos['tp']:
                    exit_price = pos['tp']
                elif hi[i] >= pos['sl']:
                    exit_price = pos['sl']
                elif held >= max_hold:
                    exit_price = cl[i]
            if exit_price is not None:
                gross = ((exit_price / pos['entry'] - 1) * 100) if pos['side'] == 'LONG' else ((1 - exit_price / pos['entry']) * 100)
                net = gross - FRICTION_RT_PCT
                trades.append({'close_ts': ts[i], 'gross_pct': gross, 'net_pnl_pct': net})
                in_pos = False
                pos = None
                continue

        if not in_pos:
            # Bar i: absorption check
            if pd.isna(vol_sma[i]) or pd.isna(atr[i]) or atr[i] <= 0:
                continue
            if vol[i] < vol_mult * vol_sma[i]:
                continue
            rng_i = hi[i] - lo[i]
            if rng_i <= 0:
                continue
            body_i = abs(cl[i] - op[i]) / rng_i
            if body_i > body_max:
                continue
            # Bar i+1: confirmation
            j = i + 1
            if j >= n:
                continue
            rng_j = hi[j] - lo[j]
            if rng_j <= 0:
                continue
            body_j = cl[j] - op[j]
            if abs(body_j) / rng_j < conf_body_min:
                continue
            # Wick balance for direction
            lower_wick_i = min(op[i], cl[i]) - lo[i]
            upper_wick_i = hi[i] - max(op[i], cl[i])

            new_dir = 0
            if lower_wick_i > upper_wick_i and body_j > 0 and cl[j] > cl[i]:
                new_dir = 1
            elif upper_wick_i > lower_wick_i and body_j < 0 and cl[j] < cl[i]:
                new_dir = -1
            if new_dir == 0:
                continue

            entry_idx = j + 1
            if entry_idx >= n:
                continue
            entry = op[entry_idx]
            entry_atr = atr[i]
            if new_dir == 1:
                pos = {'side': 'LONG', 'entry_idx': entry_idx, 'entry': entry,
                       'sl': entry - sl_mult * entry_atr, 'tp': entry + tp_mult * entry_atr}
            else:
                pos = {'side': 'SHORT', 'entry_idx': entry_idx, 'entry': entry,
                       'sl': entry + sl_mult * entry_atr, 'tp': entry - tp_mult * entry_atr}
            in_pos = True
    return pd.DataFrame(trades)


class R40bSweep(MechanismSweep):
    label = 'r40b_volume_absorption'
    mechanism_description = 'R40b — Volume Absorption Wyckoff (BTC 1h)'
    PARAM_GRID = {
        'vol_lookback':    [10, 20, 50],
        'vol_mult':        [1.5, 2.0, 3.0],
        'body_ratio_max':  [0.20, 0.30, 0.40],
        'conf_body_min':   [0.30, 0.50],
        'sl_atr_mult':     [1.0, 2.0],
        'tp_atr_mult':     [2.0, 3.0],
        'max_hold_bars':   [12, 24],
    }

    def build_trades(self, df_segment, config):
        return simulate_trades(df_segment, config)


def main():
    df = pd.read_csv(DATA / 'btc_1h_720days.csv', parse_dates=['timestamp'])
    print(f'Data: {len(df):,} 1h bars')
    sweep = R40bSweep()
    result = sweep.run_sweep(df, RESULTS)
    if not result.deployable:
        print('\n→ R40b sweep: 0 OOS-passing configs.')


if __name__ == '__main__':
    main()
