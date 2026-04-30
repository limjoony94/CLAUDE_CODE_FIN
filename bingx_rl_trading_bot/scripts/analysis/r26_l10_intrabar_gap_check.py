"""R26 L=10× — Pure Intrabar Gap Risk Check.

Correct methodology: check if ANY 15m bar in 720d has single-bar adverse move
exceeding L=10× liquidation threshold (9.95%).

If max adverse < 9.95% anywhere → R26 1h grid at L=10× is liquidation-safe.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
DATA = ROOT / 'data'
RESULTS = ROOT / 'results'

DATA_15M = DATA / 'btc_15m_720days.csv'
DATA_1H = DATA / 'btc_1h_720days.csv'


def main():
    print('=' * 100)
    print('R26 L=10× Intrabar Gap Risk — Single-Bar Adverse Move Distribution')
    print('=' * 100)

    df15 = pd.read_csv(DATA_15M, parse_dates=['timestamp'])
    df15 = df15[['timestamp', 'open', 'high', 'low', 'close']].copy()
    df15 = df15.sort_values('timestamp').reset_index(drop=True)

    df1h = pd.read_csv(DATA_1H, parse_dates=['timestamp'])
    df1h = df1h[['timestamp', 'open', 'high', 'low', 'close']].copy()
    df1h = df1h.sort_values('timestamp').reset_index(drop=True)

    # Compute per-15m-bar adverse moves
    df15['prev_close'] = df15['close'].shift(1)
    df15['gap_down_pct'] = (df15['prev_close'] - df15['low']) / df15['prev_close'] * 100
    df15['gap_up_pct'] = (df15['high'] - df15['prev_close']) / df15['prev_close'] * 100
    df15['max_adverse_pct'] = df15[['gap_down_pct', 'gap_up_pct']].max(axis=1)
    df15['range_pct'] = (df15['high'] - df15['low']) / df15['prev_close'] * 100

    # Compute per-1h-bar adverse
    df1h['prev_close'] = df1h['close'].shift(1)
    df1h['gap_down_pct'] = (df1h['prev_close'] - df1h['low']) / df1h['prev_close'] * 100
    df1h['gap_up_pct'] = (df1h['high'] - df1h['prev_close']) / df1h['prev_close'] * 100
    df1h['max_adverse_pct'] = df1h[['gap_down_pct', 'gap_up_pct']].max(axis=1)

    # Liquidation thresholds per leverage
    leverages = [4, 5, 7, 10, 15, 20]
    print('Liquidation threshold = (1 - mm)/L = 99.5/L %\n')

    print('=== 15m bar adverse move distribution ===')
    print(f'  N bars: {len(df15)}')
    print(f'  Mean adverse: {df15["max_adverse_pct"].mean():.4f}%')
    print(f'  Median: {df15["max_adverse_pct"].median():.4f}%')
    print(f'  P95: {df15["max_adverse_pct"].quantile(0.95):.4f}%')
    print(f'  P99: {df15["max_adverse_pct"].quantile(0.99):.4f}%')
    print(f'  P99.9: {df15["max_adverse_pct"].quantile(0.999):.4f}%')
    print(f'  Max: {df15["max_adverse_pct"].max():.4f}%\n')

    print('=== 1h bar adverse move distribution ===')
    print(f'  Mean: {df1h["max_adverse_pct"].mean():.4f}%')
    print(f'  Median: {df1h["max_adverse_pct"].median():.4f}%')
    print(f'  P95: {df1h["max_adverse_pct"].quantile(0.95):.4f}%')
    print(f'  P99: {df1h["max_adverse_pct"].quantile(0.99):.4f}%')
    print(f'  P99.9: {df1h["max_adverse_pct"].quantile(0.999):.4f}%')
    print(f'  Max: {df1h["max_adverse_pct"].max():.4f}%\n')

    print('=== Liquidation-threshold-exceeding events (15m bars) ===')
    for L in leverages:
        threshold = 99.5 / L
        n_exceed = (df15['max_adverse_pct'] >= threshold).sum()
        per_yr = n_exceed / (720 / 365)
        print(f'  L={L}× (threshold {threshold:.2f}%): {n_exceed} 15m bars '
              f'({per_yr:.2f}/yr equivalent)')
    print()

    # Worst single bars
    print('=== Top 10 worst 15m bars (max adverse) ===')
    worst = df15.nlargest(10, 'max_adverse_pct')[['timestamp', 'max_adverse_pct',
                                                    'gap_down_pct', 'gap_up_pct',
                                                    'prev_close', 'low', 'high']]
    print(worst.to_string(index=False))
    print()

    # Liquidation risk verdict per leverage
    print('=== Liquidation risk per leverage (15m intrabar) ===')
    for L in leverages:
        threshold = 99.5 / L
        n_exceed = int((df15['max_adverse_pct'] >= threshold).sum())
        per_yr = n_exceed / (720 / 365)
        if n_exceed == 0:
            verdict = 'SAFE (no historical 15m bar reaches threshold)'
        elif per_yr <= 0.5:
            verdict = f'LOW RISK ({per_yr:.2f}/yr — rare event)'
        elif per_yr <= 2:
            verdict = f'MODERATE RISK ({per_yr:.2f}/yr)'
        else:
            verdict = f'HIGH RISK ({per_yr:.2f}/yr)'
        print(f'  L={L}×: {verdict}')

    out = {
        'date': datetime.now(timezone.utc).isoformat(),
        'analysis': 'intrabar gap risk per leverage',
        '15m_stats': {
            'mean': float(df15['max_adverse_pct'].mean()),
            'p95': float(df15['max_adverse_pct'].quantile(0.95)),
            'p99': float(df15['max_adverse_pct'].quantile(0.99)),
            'p999': float(df15['max_adverse_pct'].quantile(0.999)),
            'max': float(df15['max_adverse_pct'].max()),
        },
        'liquidation_events_per_leverage': {
            f'L{L}x': {
                'threshold_pct': 99.5/L,
                'n_15m_bars_exceed': int((df15['max_adverse_pct'] >= 99.5/L).sum()),
                'per_yr': float((df15['max_adverse_pct'] >= 99.5/L).sum() / (720/365)),
            }
            for L in leverages
        },
    }
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    p = RESULTS / f'r26_intrabar_gap_check_{ts}.json'
    with open(p, 'w') as fp:
        json.dump(out, fp, indent=2, default=str)
    print(f'\nSaved: {p}')


if __name__ == '__main__':
    main()
