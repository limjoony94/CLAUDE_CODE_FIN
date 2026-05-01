"""RSI + BB Reversion SWEEP (BTC 1h).

Two classic mean-reversion mechanisms in one file (different param grids).
Both surface-tested implicitly through 28 round (M3 R10-R12 mean reversion family).

Pre-registered:
  RSI grid: rsi_period × rsi_lower × rsi_upper × tp_atr × sl_atr
  BB grid:  bb_period × bb_std × tp_atr × sl_atr

50/25/25 split, multi-stage promotion.
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


def compute_rsi(close, period):
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_gain = pd.Series(gain).ewm(alpha=1/period, adjust=False).mean().values
    avg_loss = pd.Series(loss).ewm(alpha=1/period, adjust=False).mean().values
    rs = avg_gain / np.where(avg_loss == 0, 1e-10, avg_loss)
    return 100 - 100 / (1 + rs)


def simulate_rsi_reversion(df, params):
    df = df.reset_index(drop=True).copy()
    op = df['open'].values
    hi = df['high'].values
    lo = df['low'].values
    cl = df['close'].values
    ts = df['timestamp'].values
    rsi = compute_rsi(cl, params['rsi_period'])
    atr = compute_atr(df, 14)
    rsi_lower = params['rsi_lower']
    rsi_upper = params['rsi_upper']
    sl_mult = params['sl_atr_mult']
    tp_mult = params['tp_atr_mult']
    max_hold = params['max_hold_bars']

    n = len(df)
    trades = []
    in_pos = False
    pos = None
    for i in range(params['rsi_period'] + 5, n - 1):
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
        if not in_pos:
            if pd.isna(atr[i]) or atr[i] <= 0 or pd.isna(rsi[i]) or pd.isna(rsi[i-1]):
                continue
            new_dir = 0
            # LONG: RSI crosses up from below lower
            if rsi[i-1] <= rsi_lower and rsi[i] > rsi_lower:
                new_dir = 1
            elif rsi[i-1] >= rsi_upper and rsi[i] < rsi_upper:
                new_dir = -1
            if new_dir != 0 and i + 1 < n:
                entry = op[i + 1]
                entry_atr = atr[i]
                if new_dir == 1:
                    pos = {'side': 'LONG', 'entry_idx': i + 1, 'entry': entry,
                           'sl': entry - sl_mult * entry_atr, 'tp': entry + tp_mult * entry_atr}
                else:
                    pos = {'side': 'SHORT', 'entry_idx': i + 1, 'entry': entry,
                           'sl': entry + sl_mult * entry_atr, 'tp': entry - tp_mult * entry_atr}
                in_pos = True
    return pd.DataFrame(trades)


def simulate_bb_reversion(df, params):
    df = df.reset_index(drop=True).copy()
    op = df['open'].values
    hi = df['high'].values
    lo = df['low'].values
    cl = df['close'].values
    ts = df['timestamp'].values
    bb_p = params['bb_period']
    bb_std = params['bb_std']
    cl_ser = pd.Series(cl)
    sma = cl_ser.rolling(bb_p, min_periods=bb_p).mean().values
    sd = cl_ser.rolling(bb_p, min_periods=bb_p).std(ddof=0).values
    upper = sma + bb_std * sd
    lower = sma - bb_std * sd
    atr = compute_atr(df, 14)
    sl_mult = params['sl_atr_mult']
    tp_mult = params['tp_atr_mult']
    max_hold = params['max_hold_bars']

    n = len(df)
    trades = []
    in_pos = False
    pos = None
    for i in range(bb_p + 5, n - 1):
        if in_pos:
            held = i - pos['entry_idx']
            exit_price = None
            # Mean revert exit: cross BB middle
            if pos['side'] == 'LONG':
                if not pd.isna(sma[i]) and cl[i] >= sma[i]:
                    exit_price = cl[i]
                elif lo[i] <= pos['sl']:
                    exit_price = pos['sl']
                elif held >= max_hold:
                    exit_price = cl[i]
            else:
                if not pd.isna(sma[i]) and cl[i] <= sma[i]:
                    exit_price = cl[i]
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
        if not in_pos:
            if pd.isna(upper[i]) or pd.isna(lower[i]) or pd.isna(atr[i]) or atr[i] <= 0:
                continue
            new_dir = 0
            if cl[i] < lower[i]:
                new_dir = 1
            elif cl[i] > upper[i]:
                new_dir = -1
            if new_dir != 0 and i + 1 < n:
                entry = op[i + 1]
                entry_atr = atr[i]
                if new_dir == 1:
                    pos = {'side': 'LONG', 'entry_idx': i + 1, 'entry': entry,
                           'sl': entry - sl_mult * entry_atr, 'tp': sma[i] if not pd.isna(sma[i]) else entry * 1.01}
                else:
                    pos = {'side': 'SHORT', 'entry_idx': i + 1, 'entry': entry,
                           'sl': entry + sl_mult * entry_atr, 'tp': sma[i] if not pd.isna(sma[i]) else entry * 0.99}
                in_pos = True
    return pd.DataFrame(trades)


class RSIReversionSweep(MechanismSweep):
    label = 'rsi_reversion'
    mechanism_description = 'RSI Cross Reversion (BTC 1h)'
    PARAM_GRID = {
        'rsi_period':     [7, 14, 21],
        'rsi_lower':      [20, 30, 35],
        'rsi_upper':      [65, 70, 80],
        'sl_atr_mult':    [1.0, 2.0],
        'tp_atr_mult':    [2.0, 3.0],
        'max_hold_bars':  [24, 48],
    }

    def build_trades(self, df_segment, config):
        return simulate_rsi_reversion(df_segment, config)


class BBReversionSweep(MechanismSweep):
    label = 'bb_reversion'
    mechanism_description = 'Bollinger Band Reversion (BTC 1h)'
    PARAM_GRID = {
        'bb_period':      [10, 20, 50],
        'bb_std':         [1.5, 2.0, 2.5],
        'sl_atr_mult':    [1.0, 2.0, 3.0],
        'tp_atr_mult':    [2.0, 3.0],         # ignored — exit on mid cross
        'max_hold_bars':  [24, 48],
    }

    def build_trades(self, df_segment, config):
        return simulate_bb_reversion(df_segment, config)


def main():
    df = pd.read_csv(DATA / 'btc_1h_720days.csv', parse_dates=['timestamp'])
    print(f'Data: {len(df):,} 1h bars\n')

    print('=' * 100)
    print('RSI Cross Reversion sweep')
    print('=' * 100)
    rsi_sweep = RSIReversionSweep()
    rsi_result = rsi_sweep.run_sweep(df, RESULTS)

    print('\n\n')
    print('=' * 100)
    print('BB Reversion sweep')
    print('=' * 100)
    bb_sweep = BBReversionSweep()
    bb_result = bb_sweep.run_sweep(df, RESULTS)


if __name__ == '__main__':
    main()
