"""R42b — Ehlers Dominant Cycle Mean Reversion SWEEP.

User critique 반영: R42 single-config (cycle_thr=0.7, sma=50) vacuous 후
parameter sweep으로 mechanism potential 측정.

Pre-registered grid (FROZEN, 코드 작성 시점에 LOCK):
  cycle_threshold:    [0.3, 0.5, 0.7]
  sma_trend_window:   [0 (OFF), 50, 200]
  atr_stop_mult:      [1.0, 2.0]
  timeout_mult:       [0.5, 1.0]
  smooth_window:      [4, 8]
  detrend_window:     [20, 40]
= 3×3×2×2×2×2 = 72 configs

Multi-stage validation:
  50% IS / 25% VAL / 25% fresh OOS (anti-fishing)
  IS sweep → top-5 by daily_net → VAL → OOS only val-PASS

Pre-reg: claudedocs/r42b_ehlers_sweep_prereg.md
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import hilbert

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'strategy_lab'))
from mechanism_sweep_standard import MechanismSweep


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'data'
RESULTS = ROOT / 'results'

FRICTION_RT_PCT = 0.10  # taker round-trip


def compute_atr(df, n=14):
    high, low, close = df['high'].values, df['low'].values, df['close'].values
    tr = np.zeros(len(df))
    for i in range(1, len(df)):
        tr[i] = max(high[i] - low[i],
                    abs(high[i] - close[i-1]),
                    abs(low[i] - close[i-1]))
    atr = pd.Series(tr).rolling(n).mean().values
    return atr


def compute_cycle_wave(close, smooth_window, detrend_window, cycle_norm_window=100, min_period=6, max_period=50):
    """Hilbert transform → cycle wave [-1, +1] + instantaneous period."""
    smoothed = pd.Series(close).rolling(smooth_window).mean().values
    detrended = smoothed - pd.Series(smoothed).rolling(detrend_window).mean().values

    valid_start = detrend_window + smooth_window
    n = len(close)
    cycle_wave = np.full(n, np.nan)
    inst_period = np.full(n, np.nan)

    if valid_start >= n:
        return cycle_wave, inst_period

    detrended_valid = detrended[valid_start:]
    detrended_valid = np.nan_to_num(detrended_valid, nan=0.0)

    if len(detrended_valid) < 10:
        return cycle_wave, inst_period

    analytic = hilbert(detrended_valid)
    phase = np.unwrap(np.angle(analytic))
    inst_freq = np.diff(phase) / (2 * np.pi)
    inst_freq = np.concatenate([[inst_freq[0]], inst_freq])
    period = np.where(np.abs(inst_freq) > 1e-10, 1 / np.abs(inst_freq), max_period)
    period = np.clip(period, min_period, max_period)

    real_part = np.real(analytic)
    abs_analytic = np.abs(analytic)
    abs_max = pd.Series(abs_analytic).rolling(cycle_norm_window).max().values
    abs_max = np.where(abs_max > 1e-10, abs_max, 1e-10)
    wave = real_part / abs_max
    wave = np.clip(wave, -1.0, 1.0)

    cycle_wave[valid_start:] = wave
    inst_period[valid_start:] = period
    return cycle_wave, inst_period


def simulate_trades(df, params):
    """Simulate trades for one parameter set. Returns DataFrame with close_ts, gross_pct, net_pnl_pct."""
    df = df.reset_index(drop=True).copy()
    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    op = df['open'].values
    ts = df['timestamp'].values

    cycle_wave, period = compute_cycle_wave(
        close,
        smooth_window=params['smooth_window'],
        detrend_window=params['detrend_window'],
    )
    atr = compute_atr(df, 14)

    sma_w = params['sma_trend_window']
    if sma_w > 0:
        sma = pd.Series(close).rolling(sma_w).mean().values
    else:
        sma = None  # filter OFF

    thr = params['cycle_threshold']
    atr_stop_mult = params['atr_stop_mult']
    timeout_mult = params['timeout_mult']

    # Generate signals
    n = len(df)
    signals = np.zeros(n, dtype=int)
    for i in range(2, n - 1):
        cw = cycle_wave[i]
        cw_p = cycle_wave[i-1]
        c = close[i]
        if np.isnan(cw) or np.isnan(cw_p):
            continue
        if sma is not None:
            s = sma[i]
            if np.isnan(s):
                continue
            trend_up = c > s
            trend_dn = c < s
        else:
            trend_up = True
            trend_dn = True

        if cw < -thr and cw > cw_p and trend_up:
            signals[i] = +1
        elif cw > +thr and cw < cw_p and trend_dn:
            signals[i] = -1

    # Simulate
    trades = []
    in_pos = False
    pos = None
    for i in range(n - 1):
        if not in_pos and signals[i] != 0:
            entry_idx = i + 1
            if entry_idx >= n:
                break
            side = 'LONG' if signals[i] == +1 else 'SHORT'
            entry_price = op[entry_idx]
            entry_atr = atr[i]
            if np.isnan(entry_atr) or entry_atr <= 0:
                continue
            entry_period = period[i] if not np.isnan(period[i]) else 25
            timeout_bars = int(entry_period * timeout_mult)
            timeout_idx = min(entry_idx + timeout_bars, n - 1)
            stop_price = (entry_price - atr_stop_mult * entry_atr
                          if side == 'LONG' else
                          entry_price + atr_stop_mult * entry_atr)

            pos = {
                'side': side, 'entry_idx': entry_idx, 'entry_price': entry_price,
                'stop_price': stop_price, 'timeout_idx': timeout_idx,
                'entry_ts': ts[entry_idx],
            }
            in_pos = True
            continue

        if in_pos:
            if i <= pos['entry_idx']:
                continue
            cw = cycle_wave[i]
            exit_price = None

            if pos['side'] == 'LONG' and low[i] <= pos['stop_price']:
                exit_price = pos['stop_price']
            elif pos['side'] == 'SHORT' and high[i] >= pos['stop_price']:
                exit_price = pos['stop_price']

            if exit_price is None and not np.isnan(cw):
                if pos['side'] == 'LONG' and cw > 0:
                    exit_price = close[i]
                elif pos['side'] == 'SHORT' and cw < 0:
                    exit_price = close[i]

            if exit_price is None and i >= pos['timeout_idx']:
                exit_price = close[i]

            if exit_price is not None:
                gross = ((exit_price - pos['entry_price']) / pos['entry_price'] * 100
                         if pos['side'] == 'LONG' else
                         (pos['entry_price'] - exit_price) / pos['entry_price'] * 100)
                net = gross - FRICTION_RT_PCT
                trades.append({
                    'close_ts': ts[i],
                    'gross_pct': gross,
                    'net_pnl_pct': net,
                })
                in_pos = False
                pos = None

    return pd.DataFrame(trades)


class R42bSweep(MechanismSweep):
    label = 'r42b_ehlers_cycle'
    mechanism_description = 'R42b — Ehlers Dominant Cycle Mean Reversion (parameter sweep)'

    PARAM_GRID = {
        'cycle_threshold':  [0.3, 0.5, 0.7],
        'sma_trend_window': [0, 50, 200],     # 0 = filter OFF
        'atr_stop_mult':    [1.0, 2.0],
        'timeout_mult':     [0.5, 1.0],
        'smooth_window':    [4, 8],
        'detrend_window':   [20, 40],
    }

    def build_trades(self, df_segment, config):
        return simulate_trades(df_segment, config)


def main():
    df = pd.read_csv(DATA / 'btc_1h_720days.csv', parse_dates=['timestamp'])
    print(f'Data: {len(df):,} bars, {df.timestamp.min()} → {df.timestamp.max()}')

    sweep = R42bSweep()
    result = sweep.run_sweep(df, RESULTS)

    if not result.deployable:
        print('\n→ R42b sweep: 0 OOS-passing configs. Mechanism falsified across grid.')
    else:
        print(f'\n→ R42b sweep: {result.oos_pass_count} OOS-passing configs. PROMISING — fresh OOS (last 25%) was untouched.')


if __name__ == '__main__':
    main()
