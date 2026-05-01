"""R37b — NR7 + Bollinger Squeeze Compression Breakout SWEEP (BTC 1h).

R37 surface: 6th OOS NEG (single locked params, vacuous-borderline).
Self-contained 1h version for sweep retry.

Pre-registered grid (FROZEN):
  compression_lookback:  [5, 7, 10]      # NR-N
  bandwidth_lookback:    [20, 50]
  bandwidth_pctile_max:  [0.10, 0.20, 0.30]
  bb_period:             [20]
  bb_std:                [1.5, 2.0, 2.5]
  body_min_ratio:        [0.30, 0.40]
  sl_atr_mult:           [1.0, 2.0]
  tp_atr_mult:           [2.0, 3.0]
  max_hold_bars:         [24, 48]
= 3×2×3×3×2×2×2×2 = 864 configs

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
FRICTION_RT_PCT = 0.14  # taker RT


def compute_atr(df, n=14):
    high, low, close = df['high'].values, df['low'].values, df['close'].values
    tr = np.zeros(len(df))
    for i in range(1, len(df)):
        tr[i] = max(high[i] - low[i],
                    abs(high[i] - close[i-1]),
                    abs(low[i] - close[i-1]))
    return pd.Series(tr).rolling(n).mean().values


def add_compression(df, params):
    df = df.copy()
    n = len(df)
    hi = df['high'].values
    lo = df['low'].values
    cl = df['close'].values
    rng = hi - lo

    # NR-N
    cl_lookback = params['compression_lookback']
    nr = np.zeros(n, dtype=bool)
    for i in range(cl_lookback - 1, n):
        window = rng[i - cl_lookback + 1: i + 1]
        nr[i] = (rng[i] == window.min()) and (rng[i] > 0)
    df['nr'] = nr

    # Bollinger Bandwidth
    bb_p = params['bb_period']
    bb_std = params['bb_std']
    cl_ser = pd.Series(cl)
    sma = cl_ser.rolling(bb_p, min_periods=bb_p).mean()
    sd = cl_ser.rolling(bb_p, min_periods=bb_p).std(ddof=0)
    bw = ((sma + bb_std * sd) - (sma - bb_std * sd)) / sma
    df['bb_bw'] = bw.values
    df['bb_upper'] = (sma + bb_std * sd).values
    df['bb_lower'] = (sma - bb_std * sd).values

    bw_lb = params['bandwidth_lookback']
    bw_ser = pd.Series(bw.values)
    df['bb_pctile'] = bw_ser.rolling(bw_lb, min_periods=bw_lb).rank(pct=True).values
    return df


def simulate_trades(df, params):
    df = add_compression(df, params)
    df = df.reset_index(drop=True)
    op = df['open'].values
    hi = df['high'].values
    lo = df['low'].values
    cl = df['close'].values
    ts = df['timestamp'].values
    nr = df['nr'].values
    bb_pctile = df['bb_pctile'].values
    bb_upper = df['bb_upper'].values
    bb_lower = df['bb_lower'].values
    atr = compute_atr(df, 14)

    bw_max = params['bandwidth_pctile_max']
    body_min = params['body_min_ratio']
    sl_mult = params['sl_atr_mult']
    tp_mult = params['tp_atr_mult']
    max_hold = params['max_hold_bars']

    n = len(df)
    trades = []
    in_pos = False
    pos = None
    for i in range(50, n - 1):
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
            if any(pd.isna(x) for x in (bb_pctile[i], atr[i])):
                continue
            if atr[i] <= 0:
                continue
            # Compression check
            if not nr[i]:
                continue
            if bb_pctile[i] > bw_max:
                continue
            # Breakout direction (next bar relative to bb_upper/lower)
            if i + 1 >= n:
                break
            rng_i = hi[i] - lo[i]
            if rng_i <= 0:
                continue
            body = cl[i] - op[i]
            if abs(body) / rng_i < body_min:
                continue
            entry_idx = i + 1
            entry = op[entry_idx]
            entry_atr = atr[i]

            # Direction: above mid (>= bb mid) → LONG, else SHORT
            mid = (bb_upper[i] + bb_lower[i]) / 2
            if pd.isna(mid):
                continue
            if cl[i] >= mid:
                pos = {'side': 'LONG', 'entry_idx': entry_idx, 'entry': entry,
                       'sl': entry - sl_mult * entry_atr, 'tp': entry + tp_mult * entry_atr}
            else:
                pos = {'side': 'SHORT', 'entry_idx': entry_idx, 'entry': entry,
                       'sl': entry + sl_mult * entry_atr, 'tp': entry - tp_mult * entry_atr}
            in_pos = True
    return pd.DataFrame(trades)


class R37bSweep(MechanismSweep):
    label = 'r37b_compression_breakout'
    mechanism_description = 'R37b — NR-N + BB Squeeze Compression Breakout (BTC 1h)'

    PARAM_GRID = {
        'compression_lookback':  [5, 7, 10],
        'bandwidth_lookback':    [20, 50],
        'bandwidth_pctile_max':  [0.10, 0.20, 0.30],
        'bb_period':             [20],
        'bb_std':                [1.5, 2.0, 2.5],
        'body_min_ratio':        [0.30, 0.40],
        'sl_atr_mult':           [1.0, 2.0],
        'tp_atr_mult':           [2.0, 3.0],
        'max_hold_bars':         [24, 48],
    }

    def build_trades(self, df_segment, config):
        return simulate_trades(df_segment, config)


def main():
    df = pd.read_csv(DATA / 'btc_1h_720days.csv', parse_dates=['timestamp'])
    print(f'Data: {len(df):,} 1h bars')
    sweep = R37bSweep()
    result = sweep.run_sweep(df, RESULTS)
    if not result.deployable:
        print('\n→ R37b sweep: 0 OOS-passing configs.')


if __name__ == '__main__':
    main()
