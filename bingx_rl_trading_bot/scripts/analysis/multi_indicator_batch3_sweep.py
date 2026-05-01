"""Multi-Indicator Batch 3 SWEEP (BTC 1h) — 32 mechanism 도달용.

Final 6 mechanisms:
  1. R22 Stop hunt (wick reversal at extreme)
  2. R34 Mean reversion deep (RSI extreme + close-to-mean target)
  3. Path B R10 MTF confluence (1h alignment with 4h+daily)
  4. Path B R11 reversal (short-term mean revert)
  5. Path B R12 calendar session (Asia/EU/US session)
  6. R27 Ensemble vote (multi-indicator AND)

Common ATR exit framework, 50/25/25 split.
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


# Stop hunt: long lower-wick at recent low / upper-wick at recent high
def stop_hunt_signals(df, params):
    op = df['open'].values
    hi = df['high'].values
    lo = df['low'].values
    cl = df['close'].values
    n = len(df)
    sig = np.zeros(n, dtype=int)
    look = params['extreme_lookback']
    wick_min = params['wick_to_body_min']
    for i in range(look + 2, n):
        rng = hi[i] - lo[i]
        if rng <= 0:
            continue
        body = abs(cl[i] - op[i])
        if body <= 0:
            continue
        recent_low = np.min(lo[i - look:i])
        recent_high = np.max(hi[i - look:i])
        lower_wick = min(op[i], cl[i]) - lo[i]
        upper_wick = hi[i] - max(op[i], cl[i])
        if lo[i] <= recent_low and lower_wick / body >= wick_min and cl[i] > op[i]:
            sig[i] = 1
        elif hi[i] >= recent_high and upper_wick / body >= wick_min and cl[i] < op[i]:
            sig[i] = -1
    return sig


# Mean reversion deep: RSI extreme + price away from EMA20
def meanrev_deep_signals(df, params):
    cl = df['close'].values
    n = len(df)
    delta = np.diff(cl, prepend=cl[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    rsi_p = params['rsi_period']
    avg_g = pd.Series(gain).ewm(alpha=1/rsi_p, adjust=False).mean().values
    avg_l = pd.Series(loss).ewm(alpha=1/rsi_p, adjust=False).mean().values
    rsi = 100 - 100 / (1 + avg_g / np.where(avg_l == 0, 1e-10, avg_l))
    ema = pd.Series(cl).ewm(span=params['ema_period'], adjust=False).mean().values
    sig = np.zeros(n, dtype=int)
    rsi_l = params['rsi_extreme_low']
    rsi_h = params['rsi_extreme_high']
    pct_min = params['ema_dist_pct_min'] / 100
    for i in range(max(rsi_p, params['ema_period']) + 5, n):
        if pd.isna(rsi[i]) or pd.isna(ema[i]):
            continue
        dist_pct = (cl[i] - ema[i]) / ema[i]
        if rsi[i] < rsi_l and dist_pct < -pct_min:
            sig[i] = 1
        elif rsi[i] > rsi_h and dist_pct > pct_min:
            sig[i] = -1
    return sig


# MTF confluence: 1h trend (EMA20>50) + 4h trend + 1h close > prev day high (LONG)
def mtf_confluence_signals(df, params):
    df = df.copy()
    df['ts'] = pd.to_datetime(df['timestamp'])
    cl = df['close'].values
    cl_ser = pd.Series(cl)
    e20 = cl_ser.ewm(span=params['ema_fast'], adjust=False).mean().values
    e50 = cl_ser.ewm(span=params['ema_slow'], adjust=False).mean().values

    # 4h aggregation: every 4 bars
    e_4h_fast = cl_ser.ewm(span=params['ema_fast'] * 4, adjust=False).mean().values
    e_4h_slow = cl_ser.ewm(span=params['ema_slow'] * 4, adjust=False).mean().values

    n = len(df)
    sig = np.zeros(n, dtype=int)
    for i in range(params['ema_slow'] * 4 + 5, n):
        bull_1h = e20[i] > e50[i]
        bear_1h = e20[i] < e50[i]
        bull_4h = e_4h_fast[i] > e_4h_slow[i]
        bear_4h = e_4h_fast[i] < e_4h_slow[i]
        # Cross from non-aligned to aligned
        prev_align_long = (e20[i-1] > e50[i-1]) and (e_4h_fast[i-1] > e_4h_slow[i-1])
        prev_align_short = (e20[i-1] < e50[i-1]) and (e_4h_fast[i-1] < e_4h_slow[i-1])
        if bull_1h and bull_4h and not prev_align_long:
            sig[i] = 1
        elif bear_1h and bear_4h and not prev_align_short:
            sig[i] = -1
    return sig


# Short-term reversal: 5-bar streak revert
def reversal_signals(df, params):
    cl = df['close'].values
    n = len(df)
    sig = np.zeros(n, dtype=int)
    streak = params['streak_length']
    for i in range(streak + 1, n):
        # All up streak → short
        all_up = all(cl[i-j] > cl[i-j-1] for j in range(streak))
        all_dn = all(cl[i-j] < cl[i-j-1] for j in range(streak))
        if all_up:
            sig[i] = -1
        elif all_dn:
            sig[i] = 1
    return sig


# Calendar session: Asia (0-8 UTC), EU (8-16), US (16-24)
def calendar_session_signals(df, params):
    df = df.copy()
    df['hour'] = pd.to_datetime(df['timestamp']).dt.hour
    cl = df['close'].values
    hour = df['hour'].values
    n = len(df)
    sig = np.zeros(n, dtype=int)
    sess = params['session']  # 'asia', 'eu', 'us'
    direction = params['direction']
    if sess == 'asia':
        in_sess = lambda h: 0 <= h < 8
    elif sess == 'eu':
        in_sess = lambda h: 8 <= h < 16
    else:
        in_sess = lambda h: 16 <= h < 24
    for i in range(50, n):
        if in_sess(hour[i]) and not in_sess(hour[i-1]):
            sig[i] = direction
    return sig


# Ensemble vote: 3 indicators (RSI/EMA/Donchian) — all agree direction
def ensemble_signals(df, params):
    cl = df['close'].values
    hi = df['high'].values
    lo = df['low'].values
    n = len(df)
    delta = np.diff(cl, prepend=cl[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    avg_g = pd.Series(gain).ewm(alpha=1/14, adjust=False).mean().values
    avg_l = pd.Series(loss).ewm(alpha=1/14, adjust=False).mean().values
    rsi = 100 - 100 / (1 + avg_g / np.where(avg_l == 0, 1e-10, avg_l))
    e20 = pd.Series(cl).ewm(span=20, adjust=False).mean().values
    e50 = pd.Series(cl).ewm(span=50, adjust=False).mean().values
    look = params['don_lookback']
    ch_high = pd.Series(hi).shift(1).rolling(look).max().values
    ch_low = pd.Series(lo).shift(1).rolling(look).min().values
    sig = np.zeros(n, dtype=int)
    rsi_thr = params['rsi_threshold']
    for i in range(look + 50 + 5, n):
        if any(pd.isna(x) for x in (rsi[i], e20[i], e50[i], ch_high[i], ch_low[i])):
            continue
        long_votes = sum([rsi[i] < rsi_thr, e20[i] > e50[i], cl[i] > ch_high[i]])
        short_votes = sum([rsi[i] > 100 - rsi_thr, e20[i] < e50[i], cl[i] < ch_low[i]])
        if long_votes >= params['min_votes']:
            sig[i] = 1
        elif short_votes >= params['min_votes']:
            sig[i] = -1
    return sig


class StopHuntSweep(MechanismSweep):
    label = 'stop_hunt'
    mechanism_description = 'Stop hunt wick reversal (1h)'
    PARAM_GRID = {
        'extreme_lookback':   [10, 20, 50],
        'wick_to_body_min':   [1.0, 1.5, 2.0],
        'sl_atr_mult':        [1.0, 2.0],
        'tp_atr_mult':        [2.0, 3.0],
        'max_hold_bars':      [12, 24, 48],
    }
    def build_trades(self, df_segment, config):
        return simulate_with_signals(df_segment, stop_hunt_signals(df_segment, config), config)


class MeanRevDeepSweep(MechanismSweep):
    label = 'meanrev_deep'
    mechanism_description = 'Mean Reversion deep (RSI + EMA distance, 1h)'
    PARAM_GRID = {
        'rsi_period':          [14, 21],
        'rsi_extreme_low':     [20, 30],
        'rsi_extreme_high':    [70, 80],
        'ema_period':          [20, 50],
        'ema_dist_pct_min':    [1.0, 2.0],
        'sl_atr_mult':         [1.0, 2.0],
        'tp_atr_mult':         [2.0, 3.0],
        'max_hold_bars':       [24, 48],
    }
    def build_trades(self, df_segment, config):
        return simulate_with_signals(df_segment, meanrev_deep_signals(df_segment, config), config)


class MTFConfluenceSweep(MechanismSweep):
    label = 'mtf_confluence'
    mechanism_description = 'MTF EMA confluence (1h+4h, 1h)'
    PARAM_GRID = {
        'ema_fast':       [9, 20],
        'ema_slow':       [50, 100],
        'sl_atr_mult':    [1.0, 2.0],
        'tp_atr_mult':    [2.0, 3.0],
        'max_hold_bars':  [24, 48, 96],
    }
    def build_trades(self, df_segment, config):
        return simulate_with_signals(df_segment, mtf_confluence_signals(df_segment, config), config)


class ReversalSweep(MechanismSweep):
    label = 'short_term_reversal'
    mechanism_description = 'N-bar streak reversal (1h)'
    PARAM_GRID = {
        'streak_length':  [3, 4, 5, 6],
        'sl_atr_mult':    [1.0, 2.0],
        'tp_atr_mult':    [2.0, 3.0],
        'max_hold_bars':  [12, 24],
    }
    def build_trades(self, df_segment, config):
        return simulate_with_signals(df_segment, reversal_signals(df_segment, config), config)


class CalendarSessionSweep(MechanismSweep):
    label = 'calendar_session'
    mechanism_description = 'Calendar session entry (Asia/EU/US, 1h)'
    PARAM_GRID = {
        'session':       ['asia', 'eu', 'us'],
        'direction':     [1, -1],
        'sl_atr_mult':   [1.0, 2.0],
        'tp_atr_mult':   [2.0, 3.0],
        'max_hold_bars': [12, 24, 48],
    }
    def build_trades(self, df_segment, config):
        return simulate_with_signals(df_segment, calendar_session_signals(df_segment, config), config)


class EnsembleSweep(MechanismSweep):
    label = 'ensemble_vote'
    mechanism_description = 'Multi-indicator ensemble (RSI+EMA+Donchian, 1h)'
    PARAM_GRID = {
        'don_lookback':  [20, 50],
        'rsi_threshold': [30, 40],
        'min_votes':     [2, 3],
        'sl_atr_mult':   [1.0, 2.0],
        'tp_atr_mult':   [2.0, 3.0],
        'max_hold_bars': [24, 48],
    }
    def build_trades(self, df_segment, config):
        return simulate_with_signals(df_segment, ensemble_signals(df_segment, config), config)


def main():
    df = pd.read_csv(DATA / 'btc_1h_720days.csv', parse_dates=['timestamp'])
    print(f'Data: {len(df):,} 1h bars')
    for sweep_class in [StopHuntSweep, MeanRevDeepSweep, MTFConfluenceSweep,
                          ReversalSweep, CalendarSessionSweep, EnsembleSweep]:
        print('\n' + '=' * 100)
        sweep = sweep_class()
        sweep.run_sweep(df, RESULTS)


if __name__ == '__main__':
    main()
