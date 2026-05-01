"""R41b — MACD Cross + body filter SWEEP (BTC 1h, R41 essence).

R41 surface: avg_gross +0.034%/trade < friction 0.07% (arithmetic falsified at n=2,760).

Self-contained version on BTC 1h (simpler structure than original 5m+15m MTF).
Mechanism essence: MACD cross + body direction agreement + ATR exit.

Pre-registered grid (FROZEN):
  macd_fast:        [8, 12, 20]
  macd_slow:        [21, 26, 50]
  macd_signal:      [7, 9, 12]
  body_min_ratio:   [0.0, 0.30, 0.50]
  sl_atr_mult:      [1.0, 2.0]
  tp_atr_mult:      [2.0, 3.0]
  max_hold_bars:    [24, 48]
= 3×3×3×3×2×2×2 = 432 configs

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


def compute_macd(close, fast, slow, signal):
    cl_ser = pd.Series(close)
    ema_fast = cl_ser.ewm(span=fast, adjust=False).mean()
    ema_slow = cl_ser.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    sig = macd.ewm(span=signal, adjust=False).mean()
    return macd.values, sig.values


def compute_atr(df, n=14):
    high, low, close = df['high'].values, df['low'].values, df['close'].values
    tr = np.zeros(len(df))
    for i in range(1, len(df)):
        tr[i] = max(high[i] - low[i],
                    abs(high[i] - close[i-1]),
                    abs(low[i] - close[i-1]))
    return pd.Series(tr).rolling(n).mean().values


def simulate_trades(df, params):
    df = df.reset_index(drop=True).copy()
    op = df['open'].values
    hi = df['high'].values
    lo = df['low'].values
    cl = df['close'].values
    ts = df['timestamp'].values
    macd, sig = compute_macd(cl, params['macd_fast'], params['macd_slow'], params['macd_signal'])
    atr = compute_atr(df, 14)
    body_min = params['body_min_ratio']
    sl_mult = params['sl_atr_mult']
    tp_mult = params['tp_atr_mult']
    max_hold = params['max_hold_bars']

    n = len(df)
    trades = []
    in_pos = False
    pos = None
    for i in range(params['macd_slow'] + 5, n - 1):
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
                trades.append({
                    'close_ts': ts[i],
                    'gross_pct': gross,
                    'net_pnl_pct': net,
                })
                in_pos = False
                pos = None
                continue

        if not in_pos:
            if any(pd.isna(x) for x in (op[i], hi[i], lo[i], cl[i], macd[i], sig[i], macd[i-1], sig[i-1], atr[i])):
                continue
            if atr[i] <= 0:
                continue
            rng = hi[i] - lo[i]
            if rng <= 0:
                continue
            body = cl[i] - op[i]
            if abs(body) / rng < body_min:
                continue
            bull_cross = (macd[i-1] <= sig[i-1]) and (macd[i] > sig[i])
            bear_cross = (macd[i-1] >= sig[i-1]) and (macd[i] < sig[i])
            if not (bull_cross or bear_cross):
                continue
            if bull_cross and body <= 0:
                continue
            if bear_cross and body >= 0:
                continue
            entry_idx = i + 1
            if entry_idx >= n:
                break
            entry = op[entry_idx]
            entry_atr = atr[i]
            if bull_cross:
                pos = {'side': 'LONG', 'entry_idx': entry_idx, 'entry': entry,
                       'sl': entry - sl_mult * entry_atr, 'tp': entry + tp_mult * entry_atr}
            else:
                pos = {'side': 'SHORT', 'entry_idx': entry_idx, 'entry': entry,
                       'sl': entry + sl_mult * entry_atr, 'tp': entry - tp_mult * entry_atr}
            in_pos = True
    return pd.DataFrame(trades)


class R41bSweep(MechanismSweep):
    label = 'r41b_macd_cross'
    mechanism_description = 'R41b — MACD Cross + body filter (BTC 1h)'

    PARAM_GRID = {
        'macd_fast':       [8, 12, 20],
        'macd_slow':       [21, 26, 50],
        'macd_signal':     [7, 9, 12],
        'body_min_ratio':  [0.0, 0.30, 0.50],
        'sl_atr_mult':     [1.0, 2.0],
        'tp_atr_mult':     [2.0, 3.0],
        'max_hold_bars':   [24, 48],
    }

    def build_trades(self, df_segment, config):
        return simulate_trades(df_segment, config)


def main():
    df = pd.read_csv(DATA / 'btc_1h_720days.csv', parse_dates=['timestamp'])
    print(f'Data: {len(df):,} 1h bars')
    sweep = R41bSweep()
    result = sweep.run_sweep(df, RESULTS)
    if not result.deployable:
        print('\n→ R41b sweep: 0 OOS-passing configs.')


if __name__ == '__main__':
    main()
