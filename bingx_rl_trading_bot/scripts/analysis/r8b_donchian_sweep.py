"""R8b — 1h Donchian Breakout SWEEP retry.

R8 surface result: taker +0.04% gross/trade (양수, < friction floor 0.07%).
User critique 후 sweep retry.

Pre-registered grid (FROZEN):
  channel_lookback:   [12, 24, 48, 96]
  body_min_ratio:     [0.0 (off), 0.30, 0.40, 0.50]
  sl_atr_mult:        [0.5, 1.0, 1.5]
  tp_atr_mult:        [2.0, 3.0, 5.0]
  max_hold_bars:      [24, 48, 96]
  cooldown_bars:      [0, 1, 4]
= 4×4×3×3×3×3 = 1296 configs (큼, 대표 sample 1296 그대로)

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
FRICTION_RT_PCT = 0.14  # taker RT (0.07 × 2)


def compute_features(df, channel_lookback, atr_period=14):
    df = df.copy()
    df['channel_high'] = df['high'].shift(1).rolling(channel_lookback).max()
    df['channel_low'] = df['low'].shift(1).rolling(channel_lookback).min()
    df['body'] = (df['close'] - df['open']).abs()
    df['range'] = df['high'] - df['low']
    df['body_ratio'] = df['body'] / df['range'].replace(0, np.nan)
    prev_close = df['close'].shift(1)
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - prev_close).abs(),
        (df['low'] - prev_close).abs(),
    ], axis=1).max(axis=1)
    df['atr'] = tr.rolling(atr_period).mean()
    return df


def simulate_trades(df, params):
    df = compute_features(df, params['channel_lookback'])
    df = df.reset_index(drop=True)
    n = len(df)

    body_min = params['body_min_ratio']
    sl_mult = params['sl_atr_mult']
    tp_mult = params['tp_atr_mult']
    max_hold = params['max_hold_bars']
    cooldown = params['cooldown_bars']

    trades = []
    in_pos = False
    entry_idx = None
    entry_price = None
    direction = 0
    sl = None
    tp = None
    bars_held = 0
    last_exit_idx = -cooldown - 1

    for i in range(n - 1):
        row = df.iloc[i]
        if in_pos:
            bars_held += 1
            high = row['high']; low = row['low']
            exit_price = None
            if direction == 1:
                if high >= tp:
                    exit_price = tp
                elif low <= sl:
                    exit_price = sl
                elif bars_held >= max_hold:
                    exit_price = row['close']
            else:
                if low <= tp:
                    exit_price = tp
                elif high >= sl:
                    exit_price = sl
                elif bars_held >= max_hold:
                    exit_price = row['close']
            if exit_price is not None:
                gross = (exit_price - entry_price) / entry_price * 100 * direction
                net = gross - FRICTION_RT_PCT
                trades.append({
                    'close_ts': row['timestamp'],
                    'gross_pct': gross,
                    'net_pnl_pct': net,
                })
                in_pos = False
                last_exit_idx = i
                continue

        if not in_pos and (i - last_exit_idx) > cooldown:
            ch = row['channel_high']
            cl = row['channel_low']
            br = row['body_ratio']
            atr = row['atr']
            if pd.isna(ch) or pd.isna(cl) or pd.isna(atr) or atr <= 0:
                continue
            if not pd.isna(br) and br < body_min:
                continue
            close = row['close']
            new_dir = 0
            if close > ch:
                new_dir = 1
            elif close < cl:
                new_dir = -1
            if new_dir != 0 and i + 1 < n:
                next_row = df.iloc[i + 1]
                entry_idx = i + 1
                entry_price = next_row['open']
                direction = new_dir
                if direction == 1:
                    sl = entry_price - sl_mult * atr
                    tp = entry_price + tp_mult * atr
                else:
                    sl = entry_price + sl_mult * atr
                    tp = entry_price - tp_mult * atr
                in_pos = True
                bars_held = 0
    return pd.DataFrame(trades)


class R8bSweep(MechanismSweep):
    label = 'r8b_donchian'
    mechanism_description = 'R8b — 1h Donchian Breakout (parameter sweep)'

    PARAM_GRID = {
        'channel_lookback':  [12, 24, 48, 96],
        'body_min_ratio':    [0.0, 0.30, 0.40, 0.50],
        'sl_atr_mult':       [0.5, 1.0, 1.5],
        'tp_atr_mult':       [2.0, 3.0, 5.0],
        'max_hold_bars':     [24, 48, 96],
        'cooldown_bars':     [0, 1, 4],
    }

    def build_trades(self, df_segment, config):
        return simulate_trades(df_segment, config)


def main():
    df = pd.read_csv(DATA / 'btc_1h_720days.csv', parse_dates=['timestamp'])
    print(f'Data: {len(df):,} 1h bars, {df.timestamp.min()} → {df.timestamp.max()}')
    sweep = R8bSweep()
    result = sweep.run_sweep(df, RESULTS)
    if not result.deployable:
        print('\n→ R8b sweep: 0 OOS-passing configs.')
    else:
        print(f'\n→ R8b sweep: {result.oos_pass_count} OOS-passing configs.')


if __name__ == '__main__':
    main()
