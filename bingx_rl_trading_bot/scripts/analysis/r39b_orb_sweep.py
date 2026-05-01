"""R39b — Daily Opening Range Breakout SWEEP (BTC 1h).

R39 surface: OOS FAIL.
Self-contained 1h with parameter sweep.

Pre-registered grid:
  range_hours:        [1, 2, 4]            # opening range definition
  range_session_hour: [0, 4, 8]            # session start (UTC hour)
  body_min_ratio:     [0.0, 0.30, 0.50]
  sl_atr_mult:        [1.0, 2.0]
  tp_atr_mult:        [2.0, 3.0]
  max_hold_bars:      [12, 24]
= 3×3×3×2×2×2 = 216 configs
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
    df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
    df['date'] = pd.to_datetime(df['timestamp']).dt.date
    op = df['open'].values
    hi = df['high'].values
    lo = df['low'].values
    cl = df['close'].values
    ts = df['timestamp'].values
    hour = df['hour'].values
    date = df['date'].values
    atr = compute_atr(df, 14)

    range_hours = params['range_hours']
    sess_start = params['range_session_hour']
    body_min = params['body_min_ratio']
    sl_mult = params['sl_atr_mult']
    tp_mult = params['tp_atr_mult']
    max_hold = params['max_hold_bars']

    # Compute opening range per day
    n = len(df)
    or_high = np.full(n, np.nan)
    or_low = np.full(n, np.nan)
    cur_date = None
    range_start_idx = None
    range_end_idx = None
    daily_range_done = False

    for i in range(n):
        d = date[i]
        h = hour[i]
        if d != cur_date:
            cur_date = d
            range_start_idx = None
            range_end_idx = None
            daily_range_done = False
        if not daily_range_done:
            if range_start_idx is None and h == sess_start:
                range_start_idx = i
                range_end_idx = i + range_hours - 1
            if range_start_idx is not None and i >= range_end_idx:
                or_h = max(hi[range_start_idx:range_end_idx + 1])
                or_l = min(lo[range_start_idx:range_end_idx + 1])
                # Apply to remaining bars of this day
                for j in range(range_end_idx + 1, n):
                    if date[j] != d:
                        break
                    or_high[j] = or_h
                    or_low[j] = or_l
                daily_range_done = True

    trades = []
    in_pos = False
    pos = None
    has_entered_today = {}
    for i in range(n - 1):
        d_today = date[i]
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

        if not in_pos and not has_entered_today.get(d_today, False):
            if pd.isna(or_high[i]) or pd.isna(or_low[i]) or pd.isna(atr[i]) or atr[i] <= 0:
                continue
            rng = hi[i] - lo[i]
            if rng <= 0:
                continue
            body = cl[i] - op[i]
            if abs(body) / rng < body_min:
                continue
            new_dir = 0
            if cl[i] > or_high[i]:
                new_dir = 1
            elif cl[i] < or_low[i]:
                new_dir = -1
            if new_dir != 0 and i + 1 < n:
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
                has_entered_today[d_today] = True
    return pd.DataFrame(trades)


class R39bSweep(MechanismSweep):
    label = 'r39b_orb'
    mechanism_description = 'R39b — Daily Opening Range Breakout (BTC 1h)'
    PARAM_GRID = {
        'range_hours':        [1, 2, 4],
        'range_session_hour': [0, 4, 8],
        'body_min_ratio':     [0.0, 0.30, 0.50],
        'sl_atr_mult':        [1.0, 2.0],
        'tp_atr_mult':        [2.0, 3.0],
        'max_hold_bars':      [12, 24],
    }

    def build_trades(self, df_segment, config):
        return simulate_trades(df_segment, config)


def main():
    df = pd.read_csv(DATA / 'btc_1h_720days.csv', parse_dates=['timestamp'])
    print(f'Data: {len(df):,} 1h bars')
    sweep = R39bSweep()
    result = sweep.run_sweep(df, RESULTS)
    if not result.deployable:
        print('\n→ R39b sweep: 0 OOS-passing configs.')


if __name__ == '__main__':
    main()
