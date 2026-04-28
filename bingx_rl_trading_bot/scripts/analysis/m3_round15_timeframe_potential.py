"""M3-R15 — Timeframe axis potential assessment.

Apply user's 3-phase methodology to TIMEFRAME axis.
1h, 4h, 1d signal generation × broad param sweep × distribution metrics.
"""
import sys, json
from datetime import datetime, timezone
from pathlib import Path
from itertools import product
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / 'scripts' / 'analysis'))

from m1_bt_framework import compute_atr as compute_atr_list
from m2_round1_screening import compute_ema, compute_rsi, load_ohlcv, merge_htf
from m3_round14_potential import compute_potential


def compute_atr_arr(highs, lows, closes, period=14):
    return np.array(compute_atr_list(list(highs), list(lows), list(closes), period))


def rolling_pctile(arr, lookback, pct):
    s = pd.Series(arr)
    return s.rolling(lookback, min_periods=lookback).quantile(pct / 100).values


def prepare_tf_data(timeframe='1h'):
    """Build BTC + ETH + appropriate trend filters for given timeframe."""
    df_15m = load_ohlcv(ROOT / 'data' / 'btc_15m_720days.csv')
    df_15m_idx = df_15m.set_index('timestamp')

    if timeframe == '1h':
        df = df_15m_idx.resample('1H', label='left', closed='left').agg({
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
        }).dropna(subset=['open']).reset_index()
        bars_per_day = 24
        # Filter timeframes: 4h + 1d
        df_filter1 = df.set_index('timestamp').resample('4H', label='left', closed='left').agg({
            'close': 'last'}).dropna().reset_index()
        df_filter1['ema20'] = compute_ema(df_filter1['close'].values, 20)
        df_filter1['ema50'] = compute_ema(df_filter1['close'].values, 50)
        df_filter1['htf1_long'] = df_filter1['ema20'] > df_filter1['ema50']
        df_filter2 = df.set_index('timestamp').resample('1D', label='left', closed='left').agg({
            'close': 'last'}).dropna().reset_index()
        df_filter2['ema20'] = compute_ema(df_filter2['close'].values, 20)
        df_filter2['htf2_long'] = df_filter2['close'] > df_filter2['ema20']
        f1_minutes = 240; f2_minutes = 1440
    elif timeframe == '4h':
        df = df_15m_idx.resample('4H', label='left', closed='left').agg({
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
        }).dropna(subset=['open']).reset_index()
        bars_per_day = 6
        df_filter1 = df.set_index('timestamp').resample('1D', label='left', closed='left').agg({
            'close': 'last'}).dropna().reset_index()
        df_filter1['ema20'] = compute_ema(df_filter1['close'].values, 20)
        df_filter1['htf1_long'] = df_filter1['close'] > df_filter1['ema20']
        df_filter2 = None
        f1_minutes = 1440
    elif timeframe == '1d':
        df = df_15m_idx.resample('1D', label='left', closed='left').agg({
            'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
        }).dropna(subset=['open']).reset_index()
        bars_per_day = 1
        # 1d filter: 7-day SMA
        df['sma7'] = pd.Series(df['close'].values).rolling(7, min_periods=7).mean().values
        df['htf1_long'] = df['close'] > df['sma7']
        df_filter1 = None; df_filter2 = None
        f1_minutes = None

    # Indicators on main timeframe
    closes = df['close'].values
    highs = df['high'].values
    lows = df['low'].values
    df['rsi14'] = compute_rsi(closes, 14)
    df['atr14'] = compute_atr_arr(highs, lows, closes, 14)
    df['atr_pctile_70_200'] = rolling_pctile(df['atr14'].values, min(200, len(df)//4), 70)
    df['btc_return'] = df['close'].pct_change() * 100

    # Add close_time for merge_htf
    tf_minutes_map = {'1h': 60, '4h': 240, '1d': 1440}
    df['close_time'] = df['timestamp'] + pd.Timedelta(minutes=tf_minutes_map[timeframe])

    if df_filter1 is not None and timeframe != '1d':
        df = merge_htf(df, df_filter1.rename(columns={'htf1_long': 'h1f_long'}), f1_minutes, ['h1f_long'])
    elif timeframe == '1d':
        df['h1f_long'] = df['htf1_long']
    if df_filter2 is not None:
        df = merge_htf(df, df_filter2.rename(columns={'htf2_long': 'h2f_long'}), f2_minutes, ['h2f_long'])
    else:
        df['h2f_long'] = True  # placeholder, no second filter
    df = df.sort_values('timestamp').reset_index(drop=True)

    # ETH aggregation
    df_eth_5m = load_ohlcv(ROOT / 'data' / 'eth_binance_5m.csv')
    df_eth_idx = df_eth_5m.set_index('timestamp')
    eth_freq = {'1h': '1H', '4h': '4H', '1d': '1D'}[timeframe]
    df_eth = df_eth_idx.resample(eth_freq, label='left', closed='left').agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
    }).dropna(subset=['open']).reset_index()
    df_eth['eth_return'] = df_eth['close'].pct_change() * 100
    df_eth = df_eth.rename(columns={'close': 'eth_close'})
    tol = {'1h': pd.Timedelta(hours=1), '4h': pd.Timedelta(hours=4), '1d': pd.Timedelta(days=1)}[timeframe]
    df = pd.merge_asof(df.sort_values('timestamp'),
                        df_eth[['timestamp', 'eth_close', 'eth_return']].sort_values('timestamp'),
                        on='timestamp', direction='backward', tolerance=tol)
    df = df.sort_values('timestamp').reset_index(drop=True)

    h1f = df['h1f_long'].fillna(False).astype(bool).values
    h2f = df['h2f_long'].fillna(False).astype(bool).values

    valid = ((~pd.isna(df['rsi14'])) & (~pd.isna(df['atr14']))
              & (~pd.isna(df['atr_pctile_70_200']))
              & (~df['h1f_long'].isna()) & (~df['h2f_long'].isna())
              & (~pd.isna(df['eth_close'])) & (~pd.isna(df['eth_return']))
              & (~pd.isna(df['btc_return']))).values
    return df, h1f, h2f, valid, bars_per_day


def make_alpha_tf_entry(eth_thresh, btc_lag_thresh, atr_pctile, use_atr_filter=True):
    def fn(df, h1, h2, valid, params=None):
        n = len(df)
        btc_ret = df['btc_return'].values
        eth_ret = df['eth_return'].values
        atr = df['atr14'].values
        if use_atr_filter:
            if atr_pctile == 70:
                atr_pctile_col = df['atr_pctile_70_200'].values
            else:
                atr_pctile_col = rolling_pctile(atr, min(200, n//4), atr_pctile)
        sigs = []
        for i in range(2, n):
            if not valid[i]: continue
            if any(pd.isna(x) for x in (btc_ret[i - 1], eth_ret[i - 1])):
                continue
            if use_atr_filter:
                if pd.isna(atr[i]) or pd.isna(atr_pctile_col[i]): continue
                if not (atr[i] > atr_pctile_col[i]): continue
            if eth_ret[i - 1] > eth_thresh and btc_ret[i - 1] < btc_lag_thresh and h1[i] and h2[i]:
                sigs.append((i, 'LONG'))
            elif eth_ret[i - 1] < -eth_thresh and btc_ret[i - 1] > -btc_lag_thresh and (not h1[i]) and (not h2[i]):
                sigs.append((i, 'SHORT'))
        return sigs
    return fn


def run_bt_simple_tf(df, sigs, N_exit, friction):
    n = len(df)
    op = df['open'].values
    high = df['high'].values
    low = df['low'].values
    cl = df['close'].values
    timestamps = df['timestamp'].values
    sig_set = {idx: d for idx, d in sigs}
    in_pos = False
    pdir = None; pentry = None; pemerg = None; pstart = None
    cooldown = 0
    trades = []
    i = 0
    while i < n:
        if in_pos:
            ep = None
            if pdir == 'LONG' and low[i] <= pemerg: ep = pemerg
            elif pdir == 'SHORT' and high[i] >= pemerg: ep = pemerg
            held = i - pstart
            if ep is None and held >= N_exit:
                ep = cl[i]
            if ep is not None:
                gross = ((ep / pentry - 1) * 100) if pdir == 'LONG' else ((1 - ep / pentry) * 100)
                net = gross - friction
                trades.append({'entry_ts': str(timestamps[pstart]), 'exit_ts': str(timestamps[i]),
                                'gross': gross, 'net': net})
                in_pos = False
                cooldown = i + 2
        if not in_pos and i >= cooldown and i in sig_set:
            ni = i + 1
            if ni < n:
                pentry = op[ni]
                pdir = sig_set[i]
                pemerg = pentry * (0.985 if pdir == 'LONG' else 1.015)
                pstart = ni
                in_pos = True
                i = ni
                continue
        i += 1
    return trades


def trade_summary_simple(trades):
    if not trades: return None
    nets = [t['net'] for t in trades]
    days = (pd.to_datetime(trades[-1]['exit_ts']) - pd.to_datetime(trades[0]['entry_ts'])).days
    if days == 0: days = 1
    wins = sum(1 for x in nets if x > 0)
    n = len(nets)
    return {
        'n': n, 'days': days, 'daily_net': round(sum(nets)/days, 4),
        'wr_pct': round(100*wins/n, 2),
    }


def run_grid_tf(df, h1, h2, valid, factory, param_grid, N_grid, friction, train_frac=0.6, min_n=20):
    n_total = len(df)
    train_end = int(n_total * train_frac)
    df_tr = df.iloc[:train_end].reset_index(drop=True)
    df_te = df.iloc[train_end:].reset_index(drop=True)
    h1_tr, h1_te = h1[:train_end], h1[train_end:]
    h2_tr, h2_te = h2[:train_end], h2[train_end:]
    valid_tr = valid[:train_end]; valid_te = valid[train_end:]
    results = []
    for params in param_grid:
        for N in N_grid:
            entry_fn = factory(*params)
            sigs_tr = entry_fn(df_tr, h1_tr, h2_tr, valid_tr)
            sigs_te = entry_fn(df_te, h1_te, h2_te, valid_te)
            trades_tr = run_bt_simple_tf(df_tr, sigs_tr, N, friction)
            trades_te = run_bt_simple_tf(df_te, sigs_te, N, friction)
            s_tr = trade_summary_simple(trades_tr)
            s_te = trade_summary_simple(trades_te)
            if s_tr is None or s_te is None: continue
            if s_tr['n'] < min_n or s_te['n'] < min_n: continue
            results.append({
                'params': list(params), 'N': N,
                'train_daily': s_tr['daily_net'], 'train_n': s_tr['n'], 'train_wr': s_tr['wr_pct'],
                'test_daily': s_te['daily_net'], 'test_n': s_te['n'], 'test_wr': s_te['wr_pct'],
            })
    return results


def main():
    timeframe_results = {}

    # 1h - simpler mechanism (no ATR filter)
    print("=" * 80); print("1h TIMEFRAME — simple ETH-lag (no ATR filter)"); print("=" * 80)
    df, h1, h2, valid, bpd = prepare_tf_data('1h')
    print(f"  bars: {len(df):,} | days: {len(df)/bpd:.0f}")
    et_grid = (0.20, 0.40, 0.60, 0.80, 1.00)
    bl_grid = (0.05, 0.15, 0.25, 0.40)
    N_grid = (2, 4, 6, 8, 12)
    pgrid = list(product(et_grid, bl_grid))
    print(f"  configs: {len(pgrid)} × {len(N_grid)} = {len(pgrid)*len(N_grid)}")
    factory = lambda et, bl: make_alpha_tf_entry(et, bl, 70, use_atr_filter=False)
    res = run_grid_tf(df, h1, h2, valid, factory, pgrid, N_grid, friction=0.08, min_n=20)
    print(f"  valid configs: {len(res)}")
    timeframe_results['1h'] = res

    # 4h - simpler
    print("\n" + "=" * 80); print("4h TIMEFRAME — simple ETH-lag"); print("=" * 80)
    df, h1, h2, valid, bpd = prepare_tf_data('4h')
    print(f"  bars: {len(df):,} | days: {len(df)/bpd:.0f}")
    et_grid = (0.30, 0.60, 1.00, 1.50, 2.00)
    bl_grid = (0.10, 0.30, 0.50, 0.80)
    N_grid = (1, 2, 3, 4, 6)
    pgrid = list(product(et_grid, bl_grid))
    print(f"  configs: {len(pgrid)} × {len(N_grid)} = {len(pgrid)*len(N_grid)}")
    factory = lambda et, bl: make_alpha_tf_entry(et, bl, 70, use_atr_filter=False)
    res = run_grid_tf(df, h1, h2, valid, factory, pgrid, N_grid, friction=0.08, min_n=10)
    print(f"  valid configs: {len(res)}")
    timeframe_results['4h'] = res

    # 1d
    print("\n" + "=" * 80); print("1d TIMEFRAME — simple ETH-lag"); print("=" * 80)
    df, h1, h2, valid, bpd = prepare_tf_data('1d')
    print(f"  bars: {len(df):,} | days: {len(df)/bpd:.0f}")
    et_grid = (0.5, 1.0, 2.0, 3.0)
    bl_grid = (0.2, 0.5, 1.0)
    N_grid = (1, 2, 3, 5, 7)
    pgrid = list(product(et_grid, bl_grid))
    print(f"  configs: {len(pgrid)} × {len(N_grid)} = {len(pgrid)*len(N_grid)}")
    factory = lambda et, bl: make_alpha_tf_entry(et, bl, 70, use_atr_filter=False)
    res = run_grid_tf(df, h1, h2, valid, factory, pgrid, N_grid, friction=0.08, min_n=5)
    print(f"  valid configs: {len(res)}")
    timeframe_results['1d'] = res

    # Potential assessment
    print("\n" + "=" * 100); print("TIMEFRAME POTENTIAL ASSESSMENT"); print("=" * 100)
    print(f"{'TF':<5} {'n_configs':>10} {'p_train+':>10} {'p_test+':>10} {'p_both+':>10} {'corr':>8} "
          f"{'med_test':>10} {'max_test':>10} {'potential':>11} {'eligible':>10}")
    tf_potential = {}
    for tf, res in timeframe_results.items():
        pot = compute_potential(res, tf)
        tf_potential[tf] = pot
        if pot.get('n_configs', 0) == 0:
            print(f"{tf:<5} {'0':>10} {'N/A':>10}")
            continue
        print(f"{tf:<5} {pot['n_configs']:>10} {pot['p_train_pos']:>9.1f}% {pot['p_test_pos']:>9.1f}% "
              f"{pot['p_both_pos']:>9.1f}% {pot['corr_tt'] if pot['corr_tt'] is not None else 'N/A':>8} "
              f"{pot['median_test']:>+9.4f}% {pot['max_test']:>+9.4f}% {pot['potential_score']:>10.2f} "
              f"{'YES' if pot['phase2_eligible'] else 'no':>10}")

    eligible = [(tf, pot) for tf, pot in tf_potential.items() if pot.get('phase2_eligible', False)]
    eligible.sort(key=lambda kv: -kv[1]['potential_score'])
    print(f"\n{'=' * 80}\nPHASE-2 ELIGIBLE TIMEFRAMES\n{'=' * 80}")
    if eligible:
        for tf, pot in eligible:
            print(f"  {tf}: potential={pot['potential_score']:.2f}, p_both_pos={pot['p_both_pos']:.1f}%, corr={pot['corr_tt']}")
        winner = eligible[0]
        print(f"\n  → Highest-potential timeframe: {winner[0]}")
    else:
        print("  → 0 timeframes eligible.")
        print("  → R14 (15m) + R15 (1h, 4h, 1d) all below threshold")
        print("  → Cross-timeframe directional alpha 부재 distribution-level 확정")

    out = {'date': datetime.now(timezone.utc).isoformat(),
           'pre_reg': 'claudedocs/m3_round15_timeframe_potential.md',
           'timeframe_results': {k: [dict(r) for r in v] for k, v in timeframe_results.items()},
           'timeframe_potential': tf_potential,
           'eligible': [{'tf': t, 'pot': p} for t, p in eligible]}
    p = ROOT / 'results' / f'm3_r15_timeframe_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(p, 'w') as f: json.dump(out, f, indent=2, default=str)
    print(f"\nSaved: {p}")


if __name__ == '__main__':
    main()
