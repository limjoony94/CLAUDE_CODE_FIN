"""C1b — C1 Breakout v2 Channel Breakout SWEEP (BTC 15m).

C1 production: BT +169.5% / 333d additive 1x, but LIVE -12.86%/14d (BT-LIVE divergence).
This sweep tests BT performance only (not LIVE). Mechanism essence:
  Channel breakout (15-bar high) + body filter + fractal SL + ATR trail TP.

Pre-registered grid:
  channel_lookback:    [10, 15, 24]
  body_min_ratio:      [0.30, 0.40, 0.50]
  swing_lookback:      [3, 5, 7]
  atr_trail_mult:      [2.0, 2.5, 3.0]
  emergency_pct:       [3.0]                 # FIXED
  max_hold_bars:       [96, 192, 288]        # 24h, 48h, 72h
= 3×3×3×3×3 = 243 configs

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
FRICTION_RT_PCT = 0.10  # taker RT 0.05% × 2


def compute_atr(df, n=14):
    high, low, close = df['high'].values, df['low'].values, df['close'].values
    tr = np.zeros(len(df))
    for i in range(1, len(df)):
        tr[i] = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
    return pd.Series(tr).rolling(n).mean().values


def find_swing_low(lows, idx, lookback=5):
    start = max(0, idx - lookback)
    return float(np.min(lows[start:idx + 1]))


def find_swing_high(highs, idx, lookback=5):
    start = max(0, idx - lookback)
    return float(np.max(highs[start:idx + 1]))


def simulate_trades(df, params):
    df = df.reset_index(drop=True).copy()
    op = df['open'].values
    hi = df['high'].values
    lo = df['low'].values
    cl = df['close'].values
    ts = df['timestamp'].values
    atr = compute_atr(df, 14)

    look = params['channel_lookback']
    body_min = params['body_min_ratio']
    swing_lb = params['swing_lookback']
    trail_mult = params['atr_trail_mult']
    emerg_pct = params['emergency_pct']
    max_hold = params['max_hold_bars']

    df['ch_high'] = pd.Series(hi).shift(1).rolling(look).max().values
    df['ch_low'] = pd.Series(lo).shift(1).rolling(look).min().values
    ch_high = df['ch_high'].values
    ch_low = df['ch_low'].values

    n = len(df)
    trades = []
    in_pos = False
    pos = None

    for i in range(look + 5, n - 1):
        if in_pos:
            held = i - pos['entry_idx']
            # Update trail
            if pos['side'] == 'LONG':
                pos['best'] = max(pos['best'], hi[i])
                trail_stop = pos['best'] - trail_mult * atr[i]
                pos['trail'] = max(pos['trail'], trail_stop) if pos['trail'] is not None else trail_stop
            else:
                pos['best'] = min(pos['best'], lo[i])
                trail_stop = pos['best'] + trail_mult * atr[i]
                pos['trail'] = min(pos['trail'], trail_stop) if pos['trail'] is not None else trail_stop

            exit_price = None
            # Emergency
            if pos['side'] == 'LONG' and lo[i] <= pos['emerg']:
                exit_price = pos['emerg']
            elif pos['side'] == 'SHORT' and hi[i] >= pos['emerg']:
                exit_price = pos['emerg']
            # Fractal SL
            if exit_price is None:
                if pos['side'] == 'LONG' and lo[i] <= pos['sl']:
                    exit_price = pos['sl']
                elif pos['side'] == 'SHORT' and hi[i] >= pos['sl']:
                    exit_price = pos['sl']
            # Trail TP
            if exit_price is None and pos['trail'] is not None:
                if pos['side'] == 'LONG' and lo[i] <= pos['trail']:
                    exit_price = pos['trail']
                elif pos['side'] == 'SHORT' and hi[i] >= pos['trail']:
                    exit_price = pos['trail']
            # Timeout
            if exit_price is None and held >= max_hold:
                exit_price = cl[i]

            if exit_price is not None:
                gross = ((exit_price / pos['entry'] - 1) * 100) if pos['side'] == 'LONG' else ((1 - exit_price / pos['entry']) * 100)
                net = gross - FRICTION_RT_PCT
                trades.append({'close_ts': ts[i], 'gross_pct': gross, 'net_pnl_pct': net})
                in_pos = False
                pos = None
                continue

        if not in_pos:
            if pd.isna(ch_high[i]) or pd.isna(ch_low[i]) or pd.isna(atr[i]) or atr[i] <= 0:
                continue
            rng = hi[i] - lo[i]
            if rng <= 0:
                continue
            body = cl[i] - op[i]
            if abs(body) / rng < body_min:
                continue
            new_dir = 0
            if cl[i] > ch_high[i] and body > 0:
                new_dir = 1
            elif cl[i] < ch_low[i] and body < 0:
                new_dir = -1
            if new_dir != 0 and i + 1 < n:
                entry_idx = i + 1
                entry = op[entry_idx]
                if new_dir == 1:
                    swing = find_swing_low(lo, i, swing_lb)
                    sl = swing - 0.0005 * entry
                    emerg = entry * (1 - emerg_pct / 100)
                    pos = {'side': 'LONG', 'entry_idx': entry_idx, 'entry': entry,
                           'sl': sl, 'emerg': emerg, 'best': entry, 'trail': None}
                else:
                    swing = find_swing_high(hi, i, swing_lb)
                    sl = swing + 0.0005 * entry
                    emerg = entry * (1 + emerg_pct / 100)
                    pos = {'side': 'SHORT', 'entry_idx': entry_idx, 'entry': entry,
                           'sl': sl, 'emerg': emerg, 'best': entry, 'trail': None}
                in_pos = True
    return pd.DataFrame(trades)


class C1bSweep(MechanismSweep):
    label = 'c1b_breakout'
    mechanism_description = 'C1b — Channel Breakout 15m (BT only)'
    PARAM_GRID = {
        'channel_lookback':    [10, 15, 24],
        'body_min_ratio':      [0.30, 0.40, 0.50],
        'swing_lookback':      [3, 5, 7],
        'atr_trail_mult':      [2.0, 2.5, 3.0],
        'emergency_pct':       [3.0],
        'max_hold_bars':       [96, 192, 288],
    }

    def build_trades(self, df_segment, config):
        return simulate_trades(df_segment, config)


def main():
    df = pd.read_csv(DATA / 'btc_15m_720days.csv', parse_dates=['timestamp'])
    print(f'Data: {len(df):,} 15m bars')
    sweep = C1bSweep()
    result = sweep.run_sweep(df, RESULTS)
    if not result.deployable:
        print('\n→ C1b sweep: 0 OOS-passing configs.')


if __name__ == '__main__':
    main()
