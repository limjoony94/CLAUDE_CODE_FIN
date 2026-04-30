"""R26 Grid — multi-window variance check.

Use 720d 5m data to extract 6 non-overlapping 30d windows and run the LIVE-parity
+ halt + funding BT on each. Output: ranging_fraction vs cum_total / daily PnL
distribution to assess where the past-month result sits.
"""
import json
from datetime import datetime
from pathlib import Path

import pandas as pd
import numpy as np

from r26_grid_5m_pastweek import (
    CFG, compute_ranging_1h, map_ranging_to_5m, simulate, summarize,
)

ROOT = Path(__file__).resolve().parent.parent.parent
DATA = ROOT / 'data'
RESULTS = ROOT / 'results'
RESULTS.mkdir(exist_ok=True)

# 720d 5m data (Binance) — used for variance check
DATA_5M = DATA / 'btc_5m_720days_binance.csv'
DATA_1H = DATA / 'btc_1h_720days.csv'


def main():
    print('Loading 720d data...')
    df5 = pd.read_csv(DATA_5M, parse_dates=['timestamp'])
    df1 = pd.read_csv(DATA_1H, parse_dates=['timestamp'])
    df5['timestamp'] = pd.to_datetime(df5['timestamp'], utc=True)
    df1['timestamp'] = pd.to_datetime(df1['timestamp'], utc=True)
    df5 = df5.sort_values('timestamp').reset_index(drop=True)
    df1 = df1.sort_values('timestamp').reset_index(drop=True)
    print(f'5m: {len(df5):,} bars, {df5.timestamp.min()} → {df5.timestamp.max()}')
    print(f'1h: {len(df1):,} bars, {df1.timestamp.min()} → {df1.timestamp.max()}\n')

    # Pre-compute 1h ranging filter once (uses full series)
    ranging_1h = compute_ranging_1h(df1)

    # 30d non-overlapping windows; need ≥30d lookback before each window
    # Total ~720d data, lookback 30d → start at day 30, 6 windows × 30d = 180d
    earliest = df1['timestamp'].iloc[0] + pd.Timedelta(days=31)  # need 30d ranging lookback + buffer
    latest = df5['timestamp'].iloc[-1] - pd.Timedelta(days=1)
    span = (latest - earliest).total_seconds() / 86400
    print(f'Available BT span: {span:.1f} days')

    # Pick 20 random non-overlapping 30d starts (n=6 sign test was borderline)
    n_windows = 20
    window_days = 30
    rng = np.random.default_rng(42)
    starts_offsets = rng.choice(
        int(span - window_days),
        size=n_windows,
        replace=False,
    )
    starts_offsets = sorted(starts_offsets)
    # Ensure non-overlap by stepping
    selected = []
    last_end = None
    for off in starts_offsets:
        s = earliest + pd.Timedelta(days=int(off))
        e = s + pd.Timedelta(days=window_days)
        if last_end is not None and s < last_end:
            continue
        selected.append((s, e))
        last_end = e
        if len(selected) >= n_windows:
            break
    if len(selected) < n_windows:
        # Fallback: evenly spaced (allows slight overlap if can't fit 20 non-overlap)
        selected = []
        step = (span - window_days) / n_windows
        for i in range(n_windows):
            s = earliest + pd.Timedelta(days=i * step)
            e = s + pd.Timedelta(days=window_days)
            selected.append((s, e))

    print(f'\nRunning {len(selected)} 30d windows (LIVE-parity + halt + funding)...\n')
    results = []
    for idx, (s, e) in enumerate(selected):
        mask = (df5['timestamp'] >= s) & (df5['timestamp'] <= e)
        df5w = df5.loc[mask].reset_index(drop=True)
        if len(df5w) < 1000:
            continue
        ranging_5m = map_ranging_to_5m(df5w, ranging_1h)
        n_days = (df5w['timestamp'].max() - df5w['timestamp'].min()).total_seconds() / 86400
        rf = ranging_1h[(ranging_1h.index >= s) & (ranging_1h.index <= e)].mean()

        for mode, label in [('legacy_no_rearm', 'A'), ('live_parity_with_halt', 'C')]:
            res = simulate(df5w, ranging_5m, mode=mode)
            summ = summarize(res, n_days)
            results.append({
                'window_idx': idx,
                'start': str(s), 'end': str(e), 'n_days': n_days,
                'ranging_fraction': float(rf),
                'mode': mode,
                'cum_total_pct': summ['cum_total_pct'],
                'daily_total_pct': summ['daily_total_pct'],
                'n_trades': summ['n_trades'],
                'wr': summ['wr'],
                'halt_triggered': summ['halt_triggered'],
                'halt_info': summ['halt_info'],
            })

    df_res = pd.DataFrame(results)
    print('Per-window results:')
    print(df_res[['window_idx', 'start', 'ranging_fraction', 'mode',
                   'cum_total_pct', 'daily_total_pct', 'n_trades', 'wr',
                   'halt_triggered']].to_string(index=False))
    print()

    print('=== Distribution by mode ===')
    for mode in ['legacy_no_rearm', 'live_parity_with_halt']:
        sub = df_res[df_res['mode'] == mode]
        print(f'\n{mode}:')
        print(f'  daily_total_pct: mean={sub["daily_total_pct"].mean():+.4f}%, '
              f'median={sub["daily_total_pct"].median():+.4f}%, '
              f'min={sub["daily_total_pct"].min():+.4f}%, '
              f'max={sub["daily_total_pct"].max():+.4f}%')
        print(f'  cum_total_pct (30d): mean={sub["cum_total_pct"].mean():+.4f}%, '
              f'median={sub["cum_total_pct"].median():+.4f}%, '
              f'min={sub["cum_total_pct"].min():+.4f}%, '
              f'max={sub["cum_total_pct"].max():+.4f}%')
        print(f'  ranging_fraction range: '
              f'[{sub["ranging_fraction"].min():.3f}, {sub["ranging_fraction"].max():.3f}], '
              f'mean={sub["ranging_fraction"].mean():.3f}')
        n_halts = int(sub['halt_triggered'].sum())
        print(f'  halts triggered: {n_halts}/{len(sub)}')

    print('\n=== Reference: Past month result for comparison ===')
    print('  legacy: +3.52% / +0.114%/day  (ranging 0.742)')
    print('  with_halt: -2.12% / -0.069%/day (ranging 0.742, halt at day 2)')

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path = RESULTS / f'r26_grid_multi_window_{ts}.json'
    df_res.to_json(out_path, orient='records', indent=2)
    print(f'\nSaved: {out_path}')


if __name__ == '__main__':
    main()
