"""R36b — EMA Pullback in Trend SWEEP (BTC 1h, R36 A pullback essence).

R36 surface: retracted false positive (5/5 anti-pattern instance).
Self-contained 1h sweep retry.

Mechanism essence: pullback to EMA in established trend, fade-or-retracement entry.

Pre-registered grid:
  ema_fast:         [9, 20]
  ema_slow:         [50, 100, 200]
  pullback_pct:     [0.30, 0.50]    # close vs EMA distance
  body_min_ratio:   [0.0, 0.40]
  sl_atr_mult:      [1.0, 2.0]
  tp_atr_mult:      [2.0, 3.0]
  max_hold_bars:    [24, 48]
= 2×3×2×2×2×2×2 = 192 configs
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
    ts = df['timestamp'].values
    cl_ser = pd.Series(cl)
    ema_fast = cl_ser.ewm(span=params['ema_fast'], adjust=False).mean().values
    ema_slow = cl_ser.ewm(span=params['ema_slow'], adjust=False).mean().values
    atr = compute_atr(df, 14)

    pullback_pct = params['pullback_pct'] / 100  # convert pct to fraction
    body_min = params['body_min_ratio']
    sl_mult = params['sl_atr_mult']
    tp_mult = params['tp_atr_mult']
    max_hold = params['max_hold_bars']

    n = len(df)
    trades = []
    in_pos = False
    pos = None

    for i in range(params['ema_slow'] + 5, n - 1):
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
            if any(pd.isna(x) for x in (ema_fast[i], ema_slow[i], atr[i])):
                continue
            if atr[i] <= 0:
                continue
            rng = hi[i] - lo[i]
            if rng <= 0:
                continue
            body = cl[i] - op[i]
            if abs(body) / rng < body_min:
                continue

            uptrend = ema_fast[i] > ema_slow[i]
            downtrend = ema_fast[i] < ema_slow[i]
            # Pullback to fast EMA
            close_to_ema = abs(cl[i] - ema_fast[i]) / cl[i]

            new_dir = 0
            if uptrend and close_to_ema <= pullback_pct and cl[i] > ema_fast[i] * (1 - pullback_pct):
                # Bullish pullback
                if body > 0:
                    new_dir = 1
            elif downtrend and close_to_ema <= pullback_pct and cl[i] < ema_fast[i] * (1 + pullback_pct):
                if body < 0:
                    new_dir = -1

            if new_dir != 0:
                entry_idx = i + 1
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


class R36bSweep(MechanismSweep):
    label = 'r36b_a_pullback'
    mechanism_description = 'R36b — EMA Pullback in Trend (BTC 1h)'
    PARAM_GRID = {
        'ema_fast':       [9, 20],
        'ema_slow':       [50, 100, 200],
        'pullback_pct':   [0.30, 0.50],
        'body_min_ratio': [0.0, 0.40],
        'sl_atr_mult':    [1.0, 2.0],
        'tp_atr_mult':    [2.0, 3.0],
        'max_hold_bars':  [24, 48],
    }

    def build_trades(self, df_segment, config):
        return simulate_trades(df_segment, config)


def main():
    df = pd.read_csv(DATA / 'btc_1h_720days.csv', parse_dates=['timestamp'])
    print(f'Data: {len(df):,} 1h bars')
    sweep = R36bSweep()
    result = sweep.run_sweep(df, RESULTS)
    if not result.deployable:
        print('\n→ R36b sweep: 0 OOS-passing configs.')


if __name__ == '__main__':
    main()
