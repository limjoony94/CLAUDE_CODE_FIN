"""Multi-Indicator Batch SWEEP (BTC 1h).

4 distinct indicator families in one file (efficient batch):
  1. Stochastic %K %D cross
  2. Hour-of-day filter (TOD seasonality)
  3. Volume spike directional
  4. Range expansion (high-low expansion as breakout)

Each is a different mechanism class than 15 prior swept.
Common framework, ATR exit, 50/25/25 split.
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


def compute_stoch(high, low, close, k_period, d_period):
    high_n = pd.Series(high).rolling(k_period).max().values
    low_n = pd.Series(low).rolling(k_period).min().values
    k = (close - low_n) / np.where(high_n - low_n > 0, high_n - low_n, 1e-10) * 100
    d = pd.Series(k).rolling(d_period).mean().values
    return k, d


def simulate_with_signals(df, signals, params):
    """Generic simulator: signals (numpy array of -1/0/+1), ATR SL/TP."""
    df = df.reset_index(drop=True)
    op = df['open'].values
    hi = df['high'].values
    lo = df['low'].values
    cl = df['close'].values
    ts = df['timestamp'].values
    atr = compute_atr(df, 14)
    sl_mult = params['sl_atr_mult']
    tp_mult = params['tp_atr_mult']
    max_hold = params['max_hold_bars']

    n = len(df)
    trades = []
    in_pos = False
    pos = None
    for i in range(n - 1):
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
                trades.append({'close_ts': ts[i], 'gross_pct': gross, 'net_pnl_pct': gross - FRICTION_RT_PCT})
                in_pos = False
                pos = None
                continue
        if not in_pos and i + 1 < n and signals[i] != 0:
            if pd.isna(atr[i]) or atr[i] <= 0:
                continue
            entry = op[i + 1]
            entry_atr = atr[i]
            if signals[i] == 1:
                pos = {'side': 'LONG', 'entry_idx': i + 1, 'entry': entry,
                       'sl': entry - sl_mult * entry_atr, 'tp': entry + tp_mult * entry_atr}
            else:
                pos = {'side': 'SHORT', 'entry_idx': i + 1, 'entry': entry,
                       'sl': entry + sl_mult * entry_atr, 'tp': entry - tp_mult * entry_atr}
            in_pos = True
    return pd.DataFrame(trades)


def stoch_signals(df, params):
    k, d = compute_stoch(df['high'].values, df['low'].values, df['close'].values,
                          params['k_period'], params['d_period'])
    n = len(df)
    sig = np.zeros(n, dtype=int)
    for i in range(max(params['k_period'], params['d_period']) + 2, n):
        if pd.isna(k[i]) or pd.isna(d[i]) or pd.isna(k[i-1]) or pd.isna(d[i-1]):
            continue
        # Bull cross from oversold
        if k[i-1] <= params['oversold'] and d[i-1] <= params['oversold'] and k[i] > d[i]:
            sig[i] = 1
        elif k[i-1] >= params['overbought'] and d[i-1] >= params['overbought'] and k[i] < d[i]:
            sig[i] = -1
    return sig


def tod_signals(df, params):
    """TOD: trade only at specific hour-of-day (long bias for hour < pivot, short else)."""
    df = df.copy()
    df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
    cl = df['close'].values
    hour = df['hour'].values
    n = len(df)
    sig = np.zeros(n, dtype=int)
    target_hour = params['target_hour']
    direction = params['direction']  # 1=LONG, -1=SHORT
    last_hour = -1
    for i in range(50, n):
        if hour[i] == target_hour and last_hour != target_hour:
            sig[i] = direction
        last_hour = hour[i]
    return sig


def volume_spike_signals(df, params):
    vol = df['volume'].values
    cl = df['close'].values
    op = df['open'].values
    vol_sma = pd.Series(vol).rolling(params['vol_lookback']).mean().values
    n = len(df)
    sig = np.zeros(n, dtype=int)
    mult = params['vol_mult']
    for i in range(params['vol_lookback'] + 2, n):
        if pd.isna(vol_sma[i]) or vol[i] < mult * vol_sma[i]:
            continue
        body = cl[i] - op[i]
        if body > 0:
            sig[i] = 1
        elif body < 0:
            sig[i] = -1
    return sig


def range_expansion_signals(df, params):
    hi = df['high'].values
    lo = df['low'].values
    cl = df['close'].values
    rng = hi - lo
    rng_sma = pd.Series(rng).rolling(params['range_lookback']).mean().values
    n = len(df)
    sig = np.zeros(n, dtype=int)
    mult = params['range_mult']
    for i in range(params['range_lookback'] + 2, n):
        if pd.isna(rng_sma[i]) or rng[i] < mult * rng_sma[i]:
            continue
        if cl[i] > cl[i-1]:
            sig[i] = 1
        elif cl[i] < cl[i-1]:
            sig[i] = -1
    return sig


class StochSweep(MechanismSweep):
    label = 'stoch_cross'
    mechanism_description = 'Stochastic %K%D cross from extremes (1h)'
    PARAM_GRID = {
        'k_period':      [9, 14, 21],
        'd_period':      [3, 5],
        'oversold':      [20, 25, 30],
        'overbought':    [70, 75, 80],
        'sl_atr_mult':   [1.0, 2.0],
        'tp_atr_mult':   [2.0, 3.0],
        'max_hold_bars': [24, 48],
    }
    def build_trades(self, df_segment, config):
        sig = stoch_signals(df_segment, config)
        return simulate_with_signals(df_segment, sig, config)


class TODSweep(MechanismSweep):
    label = 'tod_filter'
    mechanism_description = 'Time-of-day filter (UTC hour, 1h)'
    PARAM_GRID = {
        'target_hour':   [0, 4, 8, 12, 16, 20],
        'direction':     [1, -1],
        'sl_atr_mult':   [1.0, 2.0],
        'tp_atr_mult':   [2.0, 3.0],
        'max_hold_bars': [12, 24, 48],
    }
    def build_trades(self, df_segment, config):
        sig = tod_signals(df_segment, config)
        return simulate_with_signals(df_segment, sig, config)


class VolumeSpikeSweep(MechanismSweep):
    label = 'volume_spike'
    mechanism_description = 'Volume spike directional (1h)'
    PARAM_GRID = {
        'vol_lookback':  [10, 20, 50],
        'vol_mult':      [1.5, 2.0, 3.0],
        'sl_atr_mult':   [1.0, 2.0],
        'tp_atr_mult':   [2.0, 3.0],
        'max_hold_bars': [12, 24, 48],
    }
    def build_trades(self, df_segment, config):
        sig = volume_spike_signals(df_segment, config)
        return simulate_with_signals(df_segment, sig, config)


class RangeExpansionSweep(MechanismSweep):
    label = 'range_expansion'
    mechanism_description = 'Range expansion breakout (1h)'
    PARAM_GRID = {
        'range_lookback': [10, 20, 50],
        'range_mult':     [1.5, 2.0, 3.0],
        'sl_atr_mult':    [1.0, 2.0],
        'tp_atr_mult':    [2.0, 3.0],
        'max_hold_bars':  [12, 24],
    }
    def build_trades(self, df_segment, config):
        sig = range_expansion_signals(df_segment, config)
        return simulate_with_signals(df_segment, sig, config)


def main():
    df = pd.read_csv(DATA / 'btc_1h_720days.csv', parse_dates=['timestamp'])
    print(f'Data: {len(df):,} 1h bars')

    for sweep_class in [StochSweep, TODSweep, VolumeSpikeSweep, RangeExpansionSweep]:
        print('\n' + '=' * 100)
        sweep = sweep_class()
        sweep.run_sweep(df, RESULTS)


if __name__ == '__main__':
    main()
