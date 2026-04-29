"""L2 Microstructure Feasibility EDA.

Pre-reg: claudedocs/l2_microstructure_feasibility_prereg.md (commit b35095e)

Substrate: BingX BTC-USDT L2 (top-20 depth @ 2 Hz, ~7 Hz trade prints).
Sample: 18.17h captured (2026-04-29 04:19 UTC start, ongoing 4-week run).

Arithmetic gate (R41 standard):
  PASS if (avg_gross × hit_rate − 0.07% taker RT) > 0 at N >= 500 events.

Features (4):
  F1 — Order Book Imbalance (OBI)
  F2 — Order Flow Imbalance (OFI, 1s window)
  F3 — Kyle's lambda × signed volume (5s window)
  F4 — Top-level queue depletion (snapshot vs 30s trailing)

Each feature:
  signal at t → predicted direction → hold horizon → realized mid % move
  if |signal| > threshold and hit_rate × avg_gross > 0.07%, PASS.
"""
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
STORAGE = ROOT / 'scripts' / 'data_pipeline' / 'storage'
RESULTS = ROOT / 'results'
RESULTS.mkdir(exist_ok=True)

DEPTH_FILE = STORAGE / 'btc_depth_20260429.parquet'
TRADES_FILE = STORAGE / 'btc_trades_20260429.parquet'

LOCKED = {
    'friction_rt_pct': 0.07,
    'min_events_per_feature': 500,
    'F1_obi_threshold': 0.10,        # |OBI - 0.5|
    'F1_horizon_ms': 5000,           # 5s ahead
    'F2_ofi_zscore_threshold': 1.0,  # 1σ event
    'F2_window_ms': 1000,            # 1s bucket
    'F2_horizon_ms': 5000,
    'F3_window_ms': 5000,
    'F3_lambda_lookback_ms': 300_000,  # 5min rolling fit
    'F3_top_decile_threshold': 1.0,    # absolute lambda*signed_vol top decile
    'F3_horizon_ms': 5000,
    'F4_short_qty_ratio': 0.3,
    'F4_long_qty_ratio': 0.7,
    'F4_lookback_ms': 30_000,
    'F4_horizon_ms': 30_000,
}


def load_depth() -> pd.DataFrame:
    df = pd.read_parquet(DEPTH_FILE)
    df = df.sort_values('event_ts_ms').reset_index(drop=True)
    df['mid'] = (df['bid_px_0'] + df['ask_px_0']) / 2
    return df


def load_trades() -> pd.DataFrame:
    df = pd.read_parquet(TRADES_FILE)
    df = df.sort_values('event_ts_ms').reset_index(drop=True)
    # signed quantity: is_buyer_maker=False means buyer is taker (aggressive buy)
    df['signed_qty'] = np.where(df['is_buyer_maker'], -df['qty'], df['qty'])
    return df


def future_mid(depth: pd.DataFrame, horizon_ms: int) -> pd.Series:
    """For each row, find mid price at t + horizon_ms (or NaN if past end)."""
    ts = depth['event_ts_ms'].values
    mids = depth['mid'].values
    n = len(depth)
    out = np.full(n, np.nan)
    j = 0
    for i in range(n):
        target = ts[i] + horizon_ms
        # advance j until ts[j] >= target
        while j < n and ts[j] < target:
            j += 1
        if j < n:
            out[i] = mids[j]
        # j cannot decrease — use local cursor
    return pd.Series(out, index=depth.index)


def feature_F1_obi(depth: pd.DataFrame) -> dict:
    """OBI = bid5 / (bid5 + ask5) − 0.5"""
    bid_qty = depth[[f'bid_qty_{i}' for i in range(5)]].sum(axis=1)
    ask_qty = depth[[f'ask_qty_{i}' for i in range(5)]].sum(axis=1)
    obi_raw = bid_qty / (bid_qty + ask_qty)
    signal = obi_raw - 0.5
    fmid = future_mid(depth, LOCKED['F1_horizon_ms'])
    realized = (fmid - depth['mid']) / depth['mid'] * 100  # in pct

    mask = (signal.abs() > LOCKED['F1_obi_threshold']) & realized.notna()
    n = int(mask.sum())
    if n < LOCKED['min_events_per_feature']:
        return {'feature': 'F1_OBI', 'n_events': n,
                'pass': False, 'reason': f'insufficient events: {n} < 500'}

    direction = np.sign(signal[mask])
    gross = realized[mask] * direction  # gross pct in predicted direction
    avg_gross = float(gross.mean())
    hit_rate = float((gross > 0).mean())
    edge = avg_gross  # per-event gross
    edge_after_friction = edge - LOCKED['friction_rt_pct']

    return {
        'feature': 'F1_OBI',
        'n_events': n,
        'avg_gross_pct': avg_gross,
        'hit_rate': hit_rate,
        'edge_after_friction_pct': edge_after_friction,
        'pass': edge_after_friction > 0,
    }


def feature_F2_ofi(depth: pd.DataFrame) -> dict:
    """OFI 1s windows: top-of-book inventory change."""
    df = depth[['event_ts_ms', 'bid_px_0', 'bid_qty_0', 'ask_px_0', 'ask_qty_0', 'mid']].copy()
    df['bucket'] = (df['event_ts_ms'] // LOCKED['F2_window_ms']).astype(int)

    # OFI per Cont-Kukanov-Stoikov (simplified)
    # within bucket: bid_added = sum(bid_qty[t]-bid_qty[t-1]) when bid_px unchanged
    df['bid_px_prev'] = df['bid_px_0'].shift(1)
    df['ask_px_prev'] = df['ask_px_0'].shift(1)
    df['bid_qty_diff'] = df['bid_qty_0'].diff()
    df['ask_qty_diff'] = df['ask_qty_0'].diff()
    df['bid_added'] = np.where(df['bid_px_0'] == df['bid_px_prev'], df['bid_qty_diff'], 0)
    df['ask_added'] = np.where(df['ask_px_0'] == df['ask_px_prev'], df['ask_qty_diff'], 0)

    bucket_agg = df.groupby('bucket').agg(
        ofi_raw=('bid_added', 'sum'),
        ask_added_sum=('ask_added', 'sum'),
        bid_qty_mean=('bid_qty_0', 'mean'),
        ask_qty_mean=('ask_qty_0', 'mean'),
        bucket_start_ts=('event_ts_ms', 'first'),
        bucket_end_ts=('event_ts_ms', 'last'),
        bucket_end_mid=('mid', 'last'),
    )
    bucket_agg['ofi'] = bucket_agg['ofi_raw'] - bucket_agg['ask_added_sum']
    bucket_agg['norm'] = (bucket_agg['bid_qty_mean'] + bucket_agg['ask_qty_mean']) / 2
    bucket_agg['ofi_normalized'] = bucket_agg['ofi'] / bucket_agg['norm'].replace(0, np.nan)

    # z-score over 5min rolling window (300 buckets)
    bucket_agg['ofi_z'] = (bucket_agg['ofi_normalized']
                           - bucket_agg['ofi_normalized'].rolling(300, min_periods=60).mean()
                           ) / bucket_agg['ofi_normalized'].rolling(300, min_periods=60).std()

    # For each bucket, find mid at bucket_end + horizon
    ts_arr = depth['event_ts_ms'].values
    mid_arr = depth['mid'].values

    targets = bucket_agg['bucket_end_ts'].values + LOCKED['F2_horizon_ms']
    fmids = np.full(len(targets), np.nan)
    j = 0
    for i in range(len(targets)):
        while j < len(ts_arr) and ts_arr[j] < targets[i]:
            j += 1
        if j < len(ts_arr):
            fmids[i] = mid_arr[j]
    bucket_agg['future_mid'] = fmids
    bucket_agg['realized_pct'] = (bucket_agg['future_mid'] - bucket_agg['bucket_end_mid']) / bucket_agg['bucket_end_mid'] * 100

    mask = (bucket_agg['ofi_z'].abs() > LOCKED['F2_ofi_zscore_threshold']) & bucket_agg['realized_pct'].notna()
    n = int(mask.sum())
    if n < LOCKED['min_events_per_feature']:
        return {'feature': 'F2_OFI', 'n_events': n,
                'pass': False, 'reason': f'insufficient events: {n} < 500'}

    direction = np.sign(bucket_agg.loc[mask, 'ofi_z'])
    gross = bucket_agg.loc[mask, 'realized_pct'] * direction
    avg_gross = float(gross.mean())
    hit_rate = float((gross > 0).mean())
    edge_after_friction = avg_gross - LOCKED['friction_rt_pct']
    return {
        'feature': 'F2_OFI',
        'n_events': n,
        'avg_gross_pct': avg_gross,
        'hit_rate': hit_rate,
        'edge_after_friction_pct': edge_after_friction,
        'pass': edge_after_friction > 0,
    }


def feature_F3_kyle_lambda(depth: pd.DataFrame, trades: pd.DataFrame) -> dict:
    """Kyle's lambda from signed-trade cumulative impact."""
    # Build 5s buckets of signed_vol from trades
    trades = trades.copy()
    trades['bucket'] = (trades['event_ts_ms'] // LOCKED['F3_window_ms']).astype(int)
    bucket_signed = trades.groupby('bucket')['signed_qty'].sum()

    # Mid at each bucket end (use last depth snapshot before bucket end)
    # Pick depth snapshot whose event_ts_ms <= bucket_end_ms (= bucket * window + window - 1)
    bucket_end_ts = bucket_signed.index * LOCKED['F3_window_ms'] + LOCKED['F3_window_ms'] - 1
    df_b = pd.DataFrame({'bucket': bucket_signed.index, 'signed_vol': bucket_signed.values,
                         'bucket_end_ts': bucket_end_ts})

    ts_arr = depth['event_ts_ms'].values
    mid_arr = depth['mid'].values

    # For each bucket end, find latest mid <= bucket_end_ts
    end_mids = np.full(len(df_b), np.nan)
    j = 0
    for i, target in enumerate(df_b['bucket_end_ts'].values):
        while j < len(ts_arr) - 1 and ts_arr[j + 1] <= target:
            j += 1
        if ts_arr[j] <= target:
            end_mids[i] = mid_arr[j]
    df_b['end_mid'] = end_mids
    df_b = df_b.dropna(subset=['end_mid'])

    df_b['delta_mid'] = df_b['end_mid'].diff()

    # Rolling lambda fit over 5min window (60 buckets of 5s = 5min)
    # Simple OLS: lambda = cov(signed_vol, delta_mid) / var(signed_vol)
    win = 60
    rolling_lambda = pd.Series(np.nan, index=df_b.index)
    sv = df_b['signed_vol'].values
    dm = df_b['delta_mid'].values
    for i in range(win, len(df_b)):
        x = sv[i-win:i]
        y = dm[i-win:i]
        x = x[~np.isnan(y)]
        y = y[~np.isnan(y)]
        if len(x) < 30:
            continue
        var_x = x.var()
        if var_x <= 0:
            continue
        rolling_lambda.iloc[i] = ((x - x.mean()) * (y - y.mean())).mean() / var_x

    df_b['lambda_hat'] = rolling_lambda.values
    df_b['signal'] = df_b['signed_vol'] * df_b['lambda_hat']

    # Future mid at bucket_end + horizon (5s)
    targets = df_b['bucket_end_ts'].values + LOCKED['F3_horizon_ms']
    fmids = np.full(len(targets), np.nan)
    j = 0
    for i in range(len(targets)):
        while j < len(ts_arr) and ts_arr[j] < targets[i]:
            j += 1
        if j < len(ts_arr):
            fmids[i] = mid_arr[j]
    df_b['future_mid'] = fmids
    df_b['realized_pct'] = (df_b['future_mid'] - df_b['end_mid']) / df_b['end_mid'] * 100

    valid = df_b.dropna(subset=['signal', 'realized_pct'])
    threshold = valid['signal'].abs().quantile(0.90)  # top decile
    mask = valid['signal'].abs() > threshold
    n = int(mask.sum())
    if n < LOCKED['min_events_per_feature']:
        return {'feature': 'F3_Kyle', 'n_events': n,
                'pass': False, 'reason': f'insufficient events: {n} < 500'}

    direction = np.sign(valid.loc[mask, 'signal'])
    gross = valid.loc[mask, 'realized_pct'] * direction
    avg_gross = float(gross.mean())
    hit_rate = float((gross > 0).mean())
    edge_after_friction = avg_gross - LOCKED['friction_rt_pct']
    return {
        'feature': 'F3_Kyle',
        'n_events': n,
        'avg_gross_pct': avg_gross,
        'hit_rate': hit_rate,
        'edge_after_friction_pct': edge_after_friction,
        'pass': edge_after_friction > 0,
    }


def feature_F4_queue_depletion(depth: pd.DataFrame) -> dict:
    """Top-level queue depletion vs 30s trailing mean."""
    df = depth[['event_ts_ms', 'bid_qty_0', 'ask_qty_0', 'mid']].copy()
    # Approximate 30s trailing mean: ~60 snapshots at 2 Hz
    win = 60
    df['bid_30s_mean'] = df['bid_qty_0'].rolling(win, min_periods=15).mean()
    df['ask_30s_mean'] = df['ask_qty_0'].rolling(win, min_periods=15).mean()

    df['bid_ratio'] = df['bid_qty_0'] / df['bid_30s_mean']
    df['ask_ratio'] = df['ask_qty_0'] / df['ask_30s_mean']

    # signal: bid empty + ask full → -1 (down break expected)
    #         ask empty + bid full → +1
    bid_empty = df['bid_ratio'] < LOCKED['F4_short_qty_ratio']
    ask_full = df['ask_ratio'] > LOCKED['F4_long_qty_ratio']
    ask_empty = df['ask_ratio'] < LOCKED['F4_short_qty_ratio']
    bid_full = df['bid_ratio'] > LOCKED['F4_long_qty_ratio']

    df['signal'] = 0
    df.loc[bid_empty & ask_full, 'signal'] = -1
    df.loc[ask_empty & bid_full, 'signal'] = +1

    fmid = future_mid(depth, LOCKED['F4_horizon_ms'])
    df['realized_pct'] = (fmid - df['mid']) / df['mid'] * 100

    mask = (df['signal'] != 0) & df['realized_pct'].notna()
    n = int(mask.sum())
    if n < LOCKED['min_events_per_feature']:
        return {'feature': 'F4_Queue', 'n_events': n,
                'pass': False, 'reason': f'insufficient events: {n} < 500'}

    gross = df.loc[mask, 'realized_pct'] * df.loc[mask, 'signal']
    avg_gross = float(gross.mean())
    hit_rate = float((gross > 0).mean())
    edge_after_friction = avg_gross - LOCKED['friction_rt_pct']
    return {
        'feature': 'F4_Queue',
        'n_events': n,
        'avg_gross_pct': avg_gross,
        'hit_rate': hit_rate,
        'edge_after_friction_pct': edge_after_friction,
        'pass': edge_after_friction > 0,
    }


def main():
    print('=' * 100)
    print('L2 Microstructure Feasibility EDA — 18h Sample Arithmetic Gate')
    print('=' * 100)
    print('Pre-reg: claudedocs/l2_microstructure_feasibility_prereg.md (b35095e)')
    print(f'Locked: friction={LOCKED["friction_rt_pct"]}% RT, min N={LOCKED["min_events_per_feature"]}\n')

    print('Loading depth...')
    depth = load_depth()
    print(f'  rows: {len(depth):,}')
    duration_s = (depth['event_ts_ms'].max() - depth['event_ts_ms'].min()) / 1000
    print(f'  duration: {duration_s/3600:.2f} h')
    print(f'  mid range: ${depth["mid"].min():.2f} → ${depth["mid"].max():.2f}\n')

    print('Loading trades...')
    trades = load_trades()
    print(f'  rows: {len(trades):,}')
    print(f'  signed qty std: {trades["signed_qty"].std():.4f}\n')

    print('=== F1 — Order Book Imbalance ===')
    r1 = feature_F1_obi(depth)
    for k, v in r1.items():
        print(f'  {k}: {v}')
    print()

    print('=== F2 — Order Flow Imbalance ===')
    r2 = feature_F2_ofi(depth)
    for k, v in r2.items():
        print(f'  {k}: {v}')
    print()

    print('=== F3 — Kyle\'s Lambda ===')
    r3 = feature_F3_kyle_lambda(depth, trades)
    for k, v in r3.items():
        print(f'  {k}: {v}')
    print()

    print('=== F4 — Top-Level Queue Depletion ===')
    r4 = feature_F4_queue_depletion(depth)
    for k, v in r4.items():
        print(f'  {k}: {v}')
    print()

    results = [r1, r2, r3, r4]
    n_pass = sum(1 for r in results if r.get('pass', False))
    print('=' * 100)
    print(f'VERDICT: {n_pass}/4 features clear arithmetic gate (avg_gross > {LOCKED["friction_rt_pct"]}% RT)')
    print('=' * 100)
    for r in results:
        ag = r.get('avg_gross_pct', 0)
        ef = r.get('edge_after_friction_pct', 0)
        hr = r.get('hit_rate', 0)
        n = r.get('n_events', 0)
        verdict = 'PASS' if r.get('pass') else 'FAIL'
        print(f'  {r["feature"]}: n={n}, avg_gross={ag:+.4f}%, hit={hr:.3f}, '
              f'edge_after_friction={ef:+.4f}%  → {verdict}')

    out = {
        'date': datetime.now(timezone.utc).isoformat(),
        'pre_reg_commit': 'b35095e',
        'locked': LOCKED,
        'sample_duration_h': duration_s / 3600,
        'depth_rows': len(depth),
        'trade_rows': len(trades),
        'features': results,
        'n_pass': n_pass,
    }
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    p = RESULTS / f'l2_microstructure_feasibility_{ts}.json'
    with open(p, 'w') as fp:
        json.dump(out, fp, indent=2, default=str)
    print(f'\nSaved: {p}')


if __name__ == '__main__':
    main()
