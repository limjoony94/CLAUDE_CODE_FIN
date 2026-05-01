"""R42 — Ehlers Dominant Cycle Mean Reversion BT.

Pre-registered: claudedocs/r42_ehlers_cycle_prereg.md (FROZEN).

Mechanism (NO modification allowed without new pre-reg):
  1. Hilbert transform of detrended smoothed close → instantaneous phase
  2. Cycle wave normalized to [-1, +1]
  3. Entry: cycle_wave extreme + turning + SMA50 trend filter
  4. Exit: cycle wave 0-cross opposite | 1×ATR stop | period×0.75 timeout
  5. Friction: 0.10% RT taker

Data: btc_1h_720days.csv (540d in-sample / 180d fresh OOS).
Output: F1-F6 PASS/FAIL per pre-reg.
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.signal import hilbert

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from strategy_lab.bootstrap_validator import bootstrap_validate, report as bootstrap_report


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'data'
RESULTS = ROOT / 'results'

# FROZEN parameters per pre-reg
PARAMS = {
    'smooth_window': 4,
    'detrend_window': 20,
    'cycle_norm_window': 100,
    'min_period_bars': 6,
    'max_period_bars': 50,
    'sma_trend_window': 50,
    'cycle_threshold': 0.70,
    'atr_window': 14,
    'atr_stop_mult': 1.0,
    'timeout_mult': 0.75,
    'friction_rt_pct': 0.10,
}


def compute_atr(df, n=14):
    high, low, close = df['high'].values, df['low'].values, df['close'].values
    tr = np.zeros(len(df))
    for i in range(1, len(df)):
        tr[i] = max(high[i] - low[i],
                    abs(high[i] - close[i-1]),
                    abs(low[i] - close[i-1]))
    atr = pd.Series(tr).rolling(n).mean().values
    return atr


def compute_cycle_wave(close):
    """Hilbert transform → instantaneous phase + normalized cycle wave."""
    smoothed = pd.Series(close).rolling(PARAMS['smooth_window']).mean().values
    detrended = smoothed - pd.Series(smoothed).rolling(PARAMS['detrend_window']).mean().values

    # Mask NaN before Hilbert
    valid_start = PARAMS['detrend_window'] + PARAMS['smooth_window']
    cycle_wave = np.full(len(close), np.nan)
    inst_period = np.full(len(close), np.nan)

    # Compute Hilbert on valid segment
    detrended_valid = detrended[valid_start:]
    detrended_valid = np.nan_to_num(detrended_valid, nan=0.0)

    analytic = hilbert(detrended_valid)
    phase = np.unwrap(np.angle(analytic))
    inst_freq = np.diff(phase) / (2 * np.pi)
    inst_freq = np.concatenate([[inst_freq[0]], inst_freq])
    period = np.where(np.abs(inst_freq) > 1e-10, 1 / np.abs(inst_freq), PARAMS['max_period_bars'])
    period = np.clip(period, PARAMS['min_period_bars'], PARAMS['max_period_bars'])

    # Cycle wave normalization
    real_part = np.real(analytic)
    abs_analytic = np.abs(analytic)
    abs_max = pd.Series(abs_analytic).rolling(PARAMS['cycle_norm_window']).max().values
    abs_max = np.where(abs_max > 1e-10, abs_max, 1e-10)
    wave = real_part / abs_max
    wave = np.clip(wave, -1.0, 1.0)

    cycle_wave[valid_start:] = wave
    inst_period[valid_start:] = period
    return cycle_wave, inst_period


def generate_signals(df):
    close = df['close'].values
    cycle_wave, period = compute_cycle_wave(close)
    sma50 = pd.Series(close).rolling(PARAMS['sma_trend_window']).mean().values

    df['cycle_wave'] = cycle_wave
    df['period'] = period
    df['sma50'] = sma50

    n = len(df)
    signals = np.zeros(n, dtype=int)  # +1 long, -1 short, 0 none
    for i in range(2, n - 1):
        cw = cycle_wave[i]
        cw_prev = cycle_wave[i-1]
        c = close[i]
        s = sma50[i]
        if np.isnan(cw) or np.isnan(cw_prev) or np.isnan(s):
            continue

        thr = PARAMS['cycle_threshold']
        if cw < -thr and cw > cw_prev and c > s:
            signals[i] = +1
        elif cw > +thr and cw < cw_prev and c < s:
            signals[i] = -1
    df['signal'] = signals
    return df


def simulate(df):
    """Simulate trades. Entry next bar open. Exit: cycle 0-cross opp / ATR stop / timeout."""
    df = df.reset_index(drop=True).copy()
    atr = compute_atr(df, PARAMS['atr_window'])
    df['atr'] = atr

    close = df['close'].values
    high = df['high'].values
    low = df['low'].values
    op = df['open'].values
    sig = df['signal'].values
    period = df['period'].values
    cycle_wave = df['cycle_wave'].values
    ts = df['timestamp'].values

    trades = []
    in_pos = False
    pos = None
    n = len(df)
    for i in range(n - 1):
        if not in_pos and sig[i] != 0:
            entry_idx = i + 1
            if entry_idx >= n:
                break
            side = 'LONG' if sig[i] == +1 else 'SHORT'
            entry_price = op[entry_idx]
            entry_atr = atr[i]
            if np.isnan(entry_atr):
                continue
            entry_period = period[i] if not np.isnan(period[i]) else 25
            timeout_bars = int(entry_period * PARAMS['timeout_mult'])
            timeout_idx = min(entry_idx + timeout_bars, n - 1)
            stop_price = (entry_price - PARAMS['atr_stop_mult'] * entry_atr
                          if side == 'LONG' else
                          entry_price + PARAMS['atr_stop_mult'] * entry_atr)

            pos = {
                'side': side, 'entry_idx': entry_idx, 'entry_price': entry_price,
                'stop_price': stop_price, 'timeout_idx': timeout_idx,
                'entry_ts': ts[entry_idx],
            }
            in_pos = True
            continue

        if in_pos:
            # Check exit at bar i (after entry)
            if i <= pos['entry_idx']:
                continue
            cw = cycle_wave[i]
            exit_price = None
            exit_reason = None

            # ATR stop (intrabar)
            if pos['side'] == 'LONG' and low[i] <= pos['stop_price']:
                exit_price = pos['stop_price']
                exit_reason = 'ATR_STOP'
            elif pos['side'] == 'SHORT' and high[i] >= pos['stop_price']:
                exit_price = pos['stop_price']
                exit_reason = 'ATR_STOP'

            # Cycle 0-cross opposite
            if exit_price is None and not np.isnan(cw):
                if pos['side'] == 'LONG' and cw > 0:
                    exit_price = close[i]
                    exit_reason = 'CYCLE_CROSS_UP'
                elif pos['side'] == 'SHORT' and cw < 0:
                    exit_price = close[i]
                    exit_reason = 'CYCLE_CROSS_DOWN'

            # Timeout
            if exit_price is None and i >= pos['timeout_idx']:
                exit_price = close[i]
                exit_reason = 'TIMEOUT'

            if exit_price is not None:
                gross_pct = ((exit_price - pos['entry_price']) / pos['entry_price'] * 100
                             if pos['side'] == 'LONG' else
                             (pos['entry_price'] - exit_price) / pos['entry_price'] * 100)
                net_pct = gross_pct - PARAMS['friction_rt_pct']
                trades.append({
                    'side': pos['side'],
                    'enter_ts': str(pos['entry_ts']),
                    'close_ts': str(ts[i]),
                    'entry_price': pos['entry_price'],
                    'exit_price': exit_price,
                    'gross_pct': gross_pct,
                    'net_pnl_pct': net_pct,
                    'duration_bars': i - pos['entry_idx'],
                    'exit_reason': exit_reason,
                })
                in_pos = False
                pos = None

    return pd.DataFrame(trades)


def evaluate_gates(df_t, span_days, label):
    print(f'\n=== {label} ===')
    if len(df_t) == 0:
        print('  No trades.')
        return {'overall_pass': False, 'reason': 'no_trades'}

    avg_gross = df_t['gross_pct'].mean()
    cum_net = df_t['net_pnl_pct'].sum()
    daily_net = cum_net / span_days
    n_trades = len(df_t)
    wr = (df_t['net_pnl_pct'] > 0).mean()

    print(f'  span_days: {span_days}')
    print(f'  n_trades: {n_trades}')
    print(f'  avg_gross/trade: {avg_gross:+.4f}%')
    print(f'  avg_net/trade: {df_t["net_pnl_pct"].mean():+.4f}%')
    print(f'  cum_net: {cum_net:+.2f}%')
    print(f'  daily_net: {daily_net:+.4f}%')
    print(f'  WR: {wr:.3f}')

    # Bootstrap
    df_t_bt = df_t.copy()
    df_t_bt['close_ts'] = pd.to_datetime(df_t_bt['close_ts'])
    span_min = df_t_bt['close_ts'].min()
    span_max = df_t_bt['close_ts'].max()
    res = bootstrap_validate(df_t_bt, span_min, span_max)
    bootstrap_report(res, f'{label} bootstrap')

    # F1-F6
    f1 = avg_gross > 0.07
    f2 = res.mean_daily_pct >= 0.20
    f3 = res.pos_rate >= 0.50
    f4 = res.p5_daily_pct >= 0.0
    min_n_window = res.min_n_trades_per_window if hasattr(res, 'min_n_trades_per_window') else 0
    f5 = min_n_window >= 3
    f6 = n_trades >= 50

    print(f'\n  Pre-registered gates:')
    print(f'  F1 avg_gross > 0.07%:        {avg_gross:+.4f}% — {"PASS" if f1 else "FAIL"}')
    print(f'  F2 bootstrap daily ≥ 0.20%:  {res.mean_daily_pct:+.4f}% — {"PASS" if f2 else "FAIL"}')
    print(f'  F3 bootstrap pos_rate ≥ 0.5: {res.pos_rate:.3f} — {"PASS" if f3 else "FAIL"}')
    print(f'  F4 bootstrap p5_daily ≥ 0:   {res.p5_daily_pct:+.4f}% — {"PASS" if f4 else "FAIL"}')
    print(f'  F5 min n/window ≥ 3:         {min_n_window} — {"PASS" if f5 else "FAIL"}')
    print(f'  F6 full n_trades ≥ 50:       {n_trades} — {"PASS" if f6 else "FAIL"}')

    overall = all([f1, f2, f3, f4, f5, f6])
    print(f'\n  OVERALL: {"✅ PASS" if overall else "🔴 FAIL"}')

    return {
        'span_days': span_days,
        'n_trades': int(n_trades),
        'avg_gross_pct': float(avg_gross),
        'avg_net_pct': float(df_t['net_pnl_pct'].mean()),
        'cum_net_pct': float(cum_net),
        'daily_net_pct': float(daily_net),
        'wr': float(wr),
        'bootstrap_mean_daily': float(res.mean_daily_pct),
        'bootstrap_pos_rate': float(res.pos_rate),
        'bootstrap_p5_daily': float(res.p5_daily_pct),
        'min_n_trades_per_window': int(min_n_window),
        'F1_avg_gross_pass': bool(f1),
        'F2_bootstrap_daily_pass': bool(f2),
        'F3_pos_rate_pass': bool(f3),
        'F4_p5_daily_pass': bool(f4),
        'F5_min_n_window_pass': bool(f5),
        'F6_full_n_pass': bool(f6),
        'overall_pass': bool(overall),
    }


def main():
    print('=' * 100)
    print('R42 — Ehlers Dominant Cycle Mean Reversion BT')
    print('=' * 100)
    print(f'Params (FROZEN per pre-reg): {PARAMS}')

    df = pd.read_csv(DATA / 'btc_1h_720days.csv', parse_dates=['timestamp'])
    print(f'Data: {len(df):,} bars, {df.timestamp.min()} → {df.timestamp.max()}')

    df = generate_signals(df)
    n_long = (df['signal'] == +1).sum()
    n_short = (df['signal'] == -1).sum()
    print(f'Signals: LONG {n_long}, SHORT {n_short}')

    # In-sample (540d) / Fresh OOS (180d) split
    span = (df['timestamp'].max() - df['timestamp'].min()).days
    cutoff = df['timestamp'].min() + pd.Timedelta(days=540)
    df_is = df[df['timestamp'] < cutoff].reset_index(drop=True)
    df_oos = df[df['timestamp'] >= cutoff].reset_index(drop=True)
    print(f'\nIS span: {(df_is.timestamp.max() - df_is.timestamp.min()).days}d')
    print(f'OOS span: {(df_oos.timestamp.max() - df_oos.timestamp.min()).days}d')

    df_is = generate_signals(df_is)
    df_oos = generate_signals(df_oos)

    # In-sample BT
    print('\n' + '=' * 100)
    print('IN-SAMPLE (540d) — develop window')
    print('=' * 100)
    trades_is = simulate(df_is)
    is_span = (df_is.timestamp.max() - df_is.timestamp.min()).days
    is_eval = evaluate_gates(trades_is, is_span, 'In-sample')

    # If IS PASSes, run OOS once
    if is_eval.get('overall_pass'):
        print('\n' + '=' * 100)
        print('FRESH OOS (180d) — final test, single attempt')
        print('=' * 100)
        trades_oos = simulate(df_oos)
        oos_span = (df_oos.timestamp.max() - df_oos.timestamp.min()).days
        oos_eval = evaluate_gates(trades_oos, oos_span, 'Fresh OOS')
    else:
        print('\n[IS FAIL → fresh OOS skipped per pre-reg (no data peek for failed mechanism)]')
        oos_eval = {'skipped': True, 'reason': 'IS failed pre-reg gates'}

    # Save
    out = {
        'date': datetime.now(timezone.utc).isoformat(),
        'mechanism': 'R42 — Ehlers Dominant Cycle Mean Reversion',
        'params': PARAMS,
        'data_span_days': span,
        'in_sample': is_eval,
        'fresh_oos': oos_eval,
    }
    out_path = RESULTS / f'r42_ehlers_cycle_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2, default=str)
    print(f'\nSaved: {out_path}')


if __name__ == '__main__':
    main()
