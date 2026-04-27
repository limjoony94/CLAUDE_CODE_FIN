"""
Phase 0.1: M1 Scalping — Data Integrity Check
=============================================
5m / 15m / 1h / 4h(1h resample) timestamp alignment + missing bar 검증.

Inputs:
  - data/btc_5m_720days_binance.csv
  - data/btc_15m_720days.csv
  - data/btc_1h_720days.csv
  - 4h: 1h resample

Output:
  - results/m1_data_integrity_*.json
"""
import sys, json
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent


def load_ohlcv(path):
    df = pd.read_csv(path, parse_dates=['timestamp']).sort_values('timestamp').reset_index(drop=True)
    if df['timestamp'].dt.tz is None:
        df['timestamp'] = df['timestamp'].dt.tz_localize('UTC')
    return df


def check_alignment(df, expected_freq_minutes, label):
    """Check for missing bars and gaps."""
    n = len(df)
    if n < 2:
        return {'label': label, 'n': n, 'error': 'insufficient data'}

    diffs = df['timestamp'].diff().dt.total_seconds().dropna() / 60
    expected = float(expected_freq_minutes)

    n_expected_gap = int((diffs == expected).sum())
    n_short_gap = int((diffs < expected).sum())
    n_long_gap = int((diffs > expected).sum())
    long_gaps = diffs[diffs > expected]
    max_gap = float(long_gaps.max()) if len(long_gaps) > 0 else expected

    return {
        'label': label,
        'n_bars': n,
        'first_ts': df['timestamp'].iloc[0].isoformat(),
        'last_ts': df['timestamp'].iloc[-1].isoformat(),
        'days': round((df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]).total_seconds() / 86400, 2),
        'expected_freq_min': expected,
        'n_expected_gap': n_expected_gap,
        'n_short_gap': n_short_gap,
        'n_long_gap': n_long_gap,
        'max_gap_min': round(max_gap, 1),
        'pct_aligned': round(100 * n_expected_gap / max(1, n - 1), 4),
    }


def resample_to_4h(df_1h):
    """1h → 4h. 4h closes at 00:00, 04:00, 08:00, 12:00, 16:00, 20:00 UTC."""
    df = df_1h.set_index('timestamp')
    df4 = df.resample('4h', origin='epoch', label='right', closed='right').agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
    }).dropna(subset=['open']).reset_index()
    return df4


def check_cross_alignment(df_5m, df_15m, df_1h, df_4h):
    """For each 5m bar, verify the most recent CLOSED 15m/1h/4h bar exists."""
    issues = {'missing_15m': 0, 'missing_1h': 0, 'missing_4h': 0}
    sample_size = min(10000, len(df_5m))
    sample_idx = list(range(0, len(df_5m), max(1, len(df_5m) // sample_size)))[:sample_size]

    ts_15m = set(df_15m['timestamp'].values.astype('datetime64[ns]'))
    ts_1h = set(df_1h['timestamp'].values.astype('datetime64[ns]'))
    ts_4h = set(df_4h['timestamp'].values.astype('datetime64[ns]'))

    for i in sample_idx:
        t = df_5m['timestamp'].iloc[i].to_datetime64()
        floor_15m = pd.Timestamp(t).floor('15min').to_datetime64()
        floor_1h = pd.Timestamp(t).floor('1h').to_datetime64()
        floor_4h = pd.Timestamp(t).floor('4h').to_datetime64()
        if floor_15m not in ts_15m: issues['missing_15m'] += 1
        if floor_1h not in ts_1h: issues['missing_1h'] += 1
        if floor_4h not in ts_4h: issues['missing_4h'] += 1

    return {'sampled': len(sample_idx), **issues,
            'pct_missing_15m': round(100 * issues['missing_15m'] / max(1, len(sample_idx)), 4),
            'pct_missing_1h': round(100 * issues['missing_1h'] / max(1, len(sample_idx)), 4),
            'pct_missing_4h': round(100 * issues['missing_4h'] / max(1, len(sample_idx)), 4)}


def main():
    print("Loading data...")
    df_5m = load_ohlcv(ROOT / 'data' / 'btc_5m_720days_binance.csv')
    df_15m = load_ohlcv(ROOT / 'data' / 'btc_15m_720days.csv')
    df_1h = load_ohlcv(ROOT / 'data' / 'btc_1h_720days.csv')
    print(f"  5m: {len(df_5m):,} bars")
    print(f"  15m: {len(df_15m):,} bars")
    print(f"  1h: {len(df_1h):,} bars")

    print("Resampling 1h → 4h...")
    df_4h = resample_to_4h(df_1h)
    print(f"  4h: {len(df_4h):,} bars")

    print("\nAlignment check (within-TF):")
    a_5m = check_alignment(df_5m, 5, '5m')
    a_15m = check_alignment(df_15m, 15, '15m')
    a_1h = check_alignment(df_1h, 60, '1h')
    a_4h = check_alignment(df_4h, 240, '4h')

    for r in [a_5m, a_15m, a_1h, a_4h]:
        print(f"  {r['label']:>4}: n={r['n_bars']:>7,} days={r['days']:>6.1f} aligned={r['pct_aligned']:>6.2f}% gaps={r['n_long_gap']:>3} max_gap={r['max_gap_min']:>6.1f}m")

    print("\nCross-TF alignment (sample 10K 5m bars → check 15m/1h/4h floor exists):")
    cross = check_cross_alignment(df_5m, df_15m, df_1h, df_4h)
    print(f"  sampled: {cross['sampled']:,}")
    print(f"  missing 15m: {cross['missing_15m']} ({cross['pct_missing_15m']:.4f}%)")
    print(f"  missing 1h:  {cross['missing_1h']} ({cross['pct_missing_1h']:.4f}%)")
    print(f"  missing 4h:  {cross['missing_4h']} ({cross['pct_missing_4h']:.4f}%)")

    print("\n=== Verdict ===")
    all_pass = (a_5m['pct_aligned'] >= 99 and a_15m['pct_aligned'] >= 99
                and a_1h['pct_aligned'] >= 99 and a_4h['pct_aligned'] >= 99
                and cross['pct_missing_15m'] < 1 and cross['pct_missing_1h'] < 1
                and cross['pct_missing_4h'] < 1)
    if all_pass:
        print("PASS — 4-TF alignment 99%+ across all timeframes")
    else:
        print("WARN — alignment issues detected, see details")

    out = {
        'date': datetime.now(timezone.utc).isoformat(),
        'within_tf': {'5m': a_5m, '15m': a_15m, '1h': a_1h, '4h': a_4h},
        'cross_tf_alignment': cross,
        'verdict': 'PASS' if all_pass else 'WARN',
    }
    p = ROOT / 'results' / f'm1_data_integrity_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(p, 'w') as f: json.dump(out, f, indent=2, default=str)
    print(f"\nSaved: {p}")


if __name__ == '__main__':
    main()
