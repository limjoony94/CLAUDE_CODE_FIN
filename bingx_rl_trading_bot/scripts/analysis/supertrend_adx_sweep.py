"""SuperTrend + ADX SWEEP (BTC 1h) — 32 mechanism 도달.

2 final mechanisms:
  1. SuperTrend trend-following entry
  2. ADX trend-strength filtered breakout
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'scripts' / 'strategy_lab'))
from mechanism_sweep_standard import MechanismSweep
from multi_indicator_batch_sweep import simulate_with_signals, compute_atr

DATA = ROOT / 'data'
RESULTS = ROOT / 'results'


def supertrend_signals(df, params):
    df = df.reset_index(drop=True)
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    n = len(df)
    atr_p = params['atr_period']
    mult = params['atr_mult']
    atr = compute_atr(df, atr_p)
    hl2 = (high + low) / 2
    upperband = hl2 + mult * atr
    lowerband = hl2 - mult * atr
    final_upper = np.zeros(n)
    final_lower = np.zeros(n)
    supertrend = np.zeros(n)
    direction = np.zeros(n, dtype=int)
    final_upper[0] = upperband[0] if not np.isnan(upperband[0]) else high[0]
    final_lower[0] = lowerband[0] if not np.isnan(lowerband[0]) else low[0]
    for i in range(1, n):
        if pd.isna(upperband[i]) or pd.isna(lowerband[i]):
            final_upper[i] = final_upper[i-1]
            final_lower[i] = final_lower[i-1]
            continue
        if upperband[i] < final_upper[i-1] or close[i-1] > final_upper[i-1]:
            final_upper[i] = upperband[i]
        else:
            final_upper[i] = final_upper[i-1]
        if lowerband[i] > final_lower[i-1] or close[i-1] < final_lower[i-1]:
            final_lower[i] = lowerband[i]
        else:
            final_lower[i] = final_lower[i-1]
        if close[i] > final_upper[i-1]:
            direction[i] = 1
        elif close[i] < final_lower[i-1]:
            direction[i] = -1
        else:
            direction[i] = direction[i-1]
    sig = np.zeros(n, dtype=int)
    for i in range(2, n):
        if direction[i] == 1 and direction[i-1] != 1:
            sig[i] = 1
        elif direction[i] == -1 and direction[i-1] != -1:
            sig[i] = -1
    return sig


def adx_breakout_signals(df, params):
    df = df.reset_index(drop=True)
    high = df['high'].values
    low = df['low'].values
    close = df['close'].values
    n = len(df)
    period = params['adx_period']
    # +DM, -DM
    plus_dm = np.zeros(n)
    minus_dm = np.zeros(n)
    for i in range(1, n):
        up = high[i] - high[i-1]
        dn = low[i-1] - low[i]
        if up > dn and up > 0:
            plus_dm[i] = up
        if dn > up and dn > 0:
            minus_dm[i] = dn
    atr = compute_atr(df, period)
    plus_di = 100 * pd.Series(plus_dm).ewm(alpha=1/period, adjust=False).mean().values / np.where(atr > 0, atr, 1e-10)
    minus_di = 100 * pd.Series(minus_dm).ewm(alpha=1/period, adjust=False).mean().values / np.where(atr > 0, atr, 1e-10)
    dx = 100 * np.abs(plus_di - minus_di) / np.where(plus_di + minus_di > 0, plus_di + minus_di, 1e-10)
    adx = pd.Series(dx).ewm(alpha=1/period, adjust=False).mean().values

    look = params['breakout_lookback']
    ch_high = pd.Series(high).shift(1).rolling(look).max().values
    ch_low = pd.Series(low).shift(1).rolling(look).min().values
    sig = np.zeros(n, dtype=int)
    adx_min = params['adx_min']
    for i in range(period * 2 + look + 5, n):
        if pd.isna(adx[i]) or adx[i] < adx_min:
            continue
        if pd.isna(ch_high[i]) or pd.isna(ch_low[i]):
            continue
        if close[i] > ch_high[i] and plus_di[i] > minus_di[i]:
            sig[i] = 1
        elif close[i] < ch_low[i] and plus_di[i] < minus_di[i]:
            sig[i] = -1
    return sig


class SuperTrendSweep(MechanismSweep):
    label = 'supertrend'
    mechanism_description = 'SuperTrend trend-following (1h)'
    PARAM_GRID = {
        'atr_period':    [10, 14, 21],
        'atr_mult':      [2.0, 3.0, 4.0],
        'sl_atr_mult':   [1.0, 2.0],
        'tp_atr_mult':   [2.0, 3.0],
        'max_hold_bars': [24, 48, 96],
    }
    def build_trades(self, df_segment, config):
        return simulate_with_signals(df_segment, supertrend_signals(df_segment, config), config)


class ADXBreakoutSweep(MechanismSweep):
    label = 'adx_breakout'
    mechanism_description = 'ADX-filtered Donchian breakout (1h)'
    PARAM_GRID = {
        'adx_period':         [14, 21],
        'adx_min':            [20, 25, 30],
        'breakout_lookback':  [12, 24, 48],
        'sl_atr_mult':        [1.0, 2.0],
        'tp_atr_mult':        [2.0, 3.0],
        'max_hold_bars':      [24, 48],
    }
    def build_trades(self, df_segment, config):
        return simulate_with_signals(df_segment, adx_breakout_signals(df_segment, config), config)


def main():
    df = pd.read_csv(DATA / 'btc_1h_720days.csv', parse_dates=['timestamp'])
    print(f'Data: {len(df):,} 1h bars')
    for sweep_class in [SuperTrendSweep, ADXBreakoutSweep]:
        print('\n' + '=' * 100)
        sweep = sweep_class()
        sweep.run_sweep(df, RESULTS)


if __name__ == '__main__':
    main()
