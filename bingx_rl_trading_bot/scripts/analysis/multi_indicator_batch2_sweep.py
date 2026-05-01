"""Multi-Indicator Batch 2 SWEEP (BTC 1h).

6 additional mechanisms:
  1. Triple EMA alignment (5/20/50)
  2. Donchian + RSI combo
  3. Day-of-week filter
  4. Heikin-Ashi candle direction
  5. VWAP cross (anchored to weekly start)
  6. Volatility z-score reversion
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


def triple_ema_signals(df, params):
    cl = df['close'].values
    cl_ser = pd.Series(cl)
    e1 = cl_ser.ewm(span=params['ema_fast'], adjust=False).mean().values
    e2 = cl_ser.ewm(span=params['ema_mid'], adjust=False).mean().values
    e3 = cl_ser.ewm(span=params['ema_slow'], adjust=False).mean().values
    n = len(df)
    sig = np.zeros(n, dtype=int)
    for i in range(params['ema_slow'] + 5, n):
        long_align = e1[i] > e2[i] > e3[i]
        long_align_prev = e1[i-1] > e2[i-1] > e3[i-1]
        short_align = e1[i] < e2[i] < e3[i]
        short_align_prev = e1[i-1] < e2[i-1] < e3[i-1]
        if long_align and not long_align_prev:
            sig[i] = 1
        elif short_align and not short_align_prev:
            sig[i] = -1
    return sig


def donchian_rsi_signals(df, params):
    hi = df['high'].values
    lo = df['low'].values
    cl = df['close'].values
    look = params['don_lookback']
    ch_high = pd.Series(hi).shift(1).rolling(look).max().values
    ch_low = pd.Series(lo).shift(1).rolling(look).min().values

    delta = np.diff(cl, prepend=cl[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    rsi_p = params['rsi_period']
    avg_g = pd.Series(gain).ewm(alpha=1/rsi_p, adjust=False).mean().values
    avg_l = pd.Series(loss).ewm(alpha=1/rsi_p, adjust=False).mean().values
    rsi = 100 - 100 / (1 + avg_g / np.where(avg_l == 0, 1e-10, avg_l))

    n = len(df)
    sig = np.zeros(n, dtype=int)
    for i in range(look + rsi_p + 2, n):
        if pd.isna(ch_high[i]) or pd.isna(ch_low[i]):
            continue
        if cl[i] > ch_high[i] and rsi[i] < params['rsi_max_for_long']:
            sig[i] = 1
        elif cl[i] < ch_low[i] and rsi[i] > params['rsi_min_for_short']:
            sig[i] = -1
    return sig


def dow_signals(df, params):
    df = df.copy()
    df['dow'] = pd.to_datetime(df['timestamp']).dt.dayofweek
    n = len(df)
    sig = np.zeros(n, dtype=int)
    target_dow = params['target_dow']
    direction = params['direction']
    last_dow = -1
    for i in range(n):
        d = df['dow'].iloc[i]
        if d == target_dow and last_dow != target_dow:
            sig[i] = direction
        last_dow = d
    return sig


def heikin_ashi_signals(df, params):
    op = df['open'].values
    hi = df['high'].values
    lo = df['low'].values
    cl = df['close'].values
    n = len(df)
    ha_close = (op + hi + lo + cl) / 4
    ha_open = np.zeros(n)
    ha_open[0] = (op[0] + cl[0]) / 2
    for i in range(1, n):
        ha_open[i] = (ha_open[i-1] + ha_close[i-1]) / 2
    sig = np.zeros(n, dtype=int)
    streak = params['streak_required']
    for i in range(streak, n):
        bull_streak = all(ha_close[i-j] > ha_open[i-j] for j in range(streak))
        bear_streak = all(ha_close[i-j] < ha_open[i-j] for j in range(streak))
        # Cross from opposite
        prev_bull = ha_close[i-streak-1] > ha_open[i-streak-1] if i > streak else True
        prev_bear = ha_close[i-streak-1] < ha_open[i-streak-1] if i > streak else True
        if bull_streak and not prev_bull:
            sig[i] = 1
        elif bear_streak and not prev_bear:
            sig[i] = -1
    return sig


def vwap_cross_signals(df, params):
    df = df.copy()
    df['ts'] = pd.to_datetime(df['timestamp'])
    df['week'] = df['ts'].dt.isocalendar().week.astype(int)
    df['year'] = df['ts'].dt.year
    df['week_id'] = df['year'] * 100 + df['week']
    cl = df['close'].values
    vol = df['volume'].values
    n = len(df)
    # Anchored VWAP per week
    vwap = np.zeros(n)
    cur_week = -1
    cum_pv = 0.0
    cum_v = 0.0
    for i in range(n):
        w = df['week_id'].iloc[i]
        if w != cur_week:
            cum_pv = 0.0
            cum_v = 0.0
            cur_week = w
        cum_pv += cl[i] * vol[i]
        cum_v += vol[i]
        vwap[i] = cum_pv / cum_v if cum_v > 0 else cl[i]
    sig = np.zeros(n, dtype=int)
    pull_pct = params['pullback_pct'] / 100
    for i in range(20, n):
        diff = (cl[i] - vwap[i]) / vwap[i]
        if abs(diff) <= pull_pct and cl[i-1] < vwap[i-1] and cl[i] > vwap[i]:
            sig[i] = 1
        elif abs(diff) <= pull_pct and cl[i-1] > vwap[i-1] and cl[i] < vwap[i]:
            sig[i] = -1
    return sig


def vol_zscore_signals(df, params):
    cl = df['close'].values
    ret = np.diff(cl, prepend=cl[0]) / cl
    rolling_vol = pd.Series(ret).rolling(params['vol_lookback']).std(ddof=0).values
    long_vol = pd.Series(rolling_vol).rolling(params['baseline_lookback']).mean().values
    z = (rolling_vol - long_vol) / pd.Series(rolling_vol).rolling(params['baseline_lookback']).std(ddof=0).values
    n = len(df)
    sig = np.zeros(n, dtype=int)
    z_thr = params['z_threshold']
    for i in range(params['baseline_lookback'] + 5, n):
        if pd.isna(z[i]):
            continue
        # High vol = mean revert (counter to recent move)
        if z[i] > z_thr:
            if ret[i] > 0:
                sig[i] = -1  # short up move
            elif ret[i] < 0:
                sig[i] = 1   # long down move
    return sig


# Sweep classes
class TripleEMASweep(MechanismSweep):
    label = 'triple_ema'
    mechanism_description = 'Triple EMA alignment (1h)'
    PARAM_GRID = {
        'ema_fast':      [5, 9],
        'ema_mid':       [20, 21],
        'ema_slow':      [50, 100, 200],
        'sl_atr_mult':   [1.0, 2.0],
        'tp_atr_mult':   [2.0, 3.0],
        'max_hold_bars': [24, 48],
    }
    def build_trades(self, df_segment, config):
        return simulate_with_signals(df_segment, triple_ema_signals(df_segment, config), config)


class DonchianRSISweep(MechanismSweep):
    label = 'donchian_rsi'
    mechanism_description = 'Donchian + RSI combo (1h)'
    PARAM_GRID = {
        'don_lookback':         [12, 24],
        'rsi_period':           [9, 14],
        'rsi_max_for_long':     [60, 70, 80],
        'rsi_min_for_short':    [20, 30, 40],
        'sl_atr_mult':          [1.0, 2.0],
        'tp_atr_mult':          [2.0, 3.0],
        'max_hold_bars':        [24, 48],
    }
    def build_trades(self, df_segment, config):
        return simulate_with_signals(df_segment, donchian_rsi_signals(df_segment, config), config)


class DOWSweep(MechanismSweep):
    label = 'dow_filter'
    mechanism_description = 'Day-of-week filter (1h)'
    PARAM_GRID = {
        'target_dow':    [0, 1, 2, 3, 4, 5, 6],
        'direction':     [1, -1],
        'sl_atr_mult':   [1.0, 2.0],
        'tp_atr_mult':   [2.0, 3.0],
        'max_hold_bars': [24, 48],
    }
    def build_trades(self, df_segment, config):
        return simulate_with_signals(df_segment, dow_signals(df_segment, config), config)


class HeikinAshiSweep(MechanismSweep):
    label = 'heikin_ashi'
    mechanism_description = 'Heikin-Ashi streak (1h)'
    PARAM_GRID = {
        'streak_required': [2, 3, 4],
        'sl_atr_mult':     [1.0, 2.0],
        'tp_atr_mult':     [2.0, 3.0],
        'max_hold_bars':   [12, 24, 48],
    }
    def build_trades(self, df_segment, config):
        return simulate_with_signals(df_segment, heikin_ashi_signals(df_segment, config), config)


class WeeklyVWAPSweep(MechanismSweep):
    label = 'weekly_vwap_cross'
    mechanism_description = 'Weekly anchored VWAP cross (1h)'
    PARAM_GRID = {
        'pullback_pct':  [0.5, 1.0, 2.0],
        'sl_atr_mult':   [1.0, 2.0],
        'tp_atr_mult':   [2.0, 3.0],
        'max_hold_bars': [24, 48, 96],
    }
    def build_trades(self, df_segment, config):
        return simulate_with_signals(df_segment, vwap_cross_signals(df_segment, config), config)


class VolZSweep(MechanismSweep):
    label = 'vol_zscore_reversion'
    mechanism_description = 'Volatility z-score reversion (1h)'
    PARAM_GRID = {
        'vol_lookback':       [10, 20],
        'baseline_lookback':  [50, 100],
        'z_threshold':        [1.5, 2.0, 2.5],
        'sl_atr_mult':        [1.0, 2.0],
        'tp_atr_mult':        [2.0, 3.0],
        'max_hold_bars':      [12, 24],
    }
    def build_trades(self, df_segment, config):
        return simulate_with_signals(df_segment, vol_zscore_signals(df_segment, config), config)


def main():
    df = pd.read_csv(DATA / 'btc_1h_720days.csv', parse_dates=['timestamp'])
    print(f'Data: {len(df):,} 1h bars')
    for sweep_class in [TripleEMASweep, DonchianRSISweep, DOWSweep,
                          HeikinAshiSweep, WeeklyVWAPSweep, VolZSweep]:
        print('\n' + '=' * 100)
        sweep = sweep_class()
        sweep.run_sweep(df, RESULTS)


if __name__ == '__main__':
    main()
