"""Path B R10 — Multi-TF Confluence Breakout LOCKED OOS.

Pre-reg: claudedocs/path_b_r10_mtf_confluence_prereg.md (commit 59e905c)

Stack 5m + 15m + 1h alignment. Entry only when all three confirm same direction.
"""
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
DATA = ROOT / 'data'
RESULTS = ROOT / 'results'

PRICE_FILE = DATA / 'btc_5m_720days_binance.csv'

LOCKED = {
    'asset': 'BTC/USDT',
    'breakout_5m_lookback': 12,
    'breakout_15m_lookback': 8,
    'trend_1h_sma_lookback': 24,
    'body_min_ratio': 0.40,
    'atr_period': 14,
    'sl_atr_mult': 1.0,
    'tp_atr_mult': 2.0,
    'max_hold_bars': 12,
    'cooldown_bars': 6,
    'friction_pct': 0.07,
}

GATES = {
    'min_confluence_events': 200,
    'body_filter_min_retention': 0.30,
    'random_pct': 0.95,
    'wf_min_pos': 3, 'wf_total': 5,
    'bs_n_iter': 1000, 'bs_window_days': 3, 'bs_min_pos_rate': 0.50,
    'tt_split': 0.60,
    'magnitude_min_daily_pct': 0.20,    # HARD
    'wr_min': 0.30,
    'rr_min': 1.5,
    'trades_per_day_min': 2.0,           # HARD
    'per_trade_gross_min_pct': 0.07,     # HARD
    'tail_max_5d_dd_pct': 15.0,
}


def load_data():
    df = pd.read_csv(PRICE_FILE, usecols=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    return df


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 5m breakout (12-bar = 1h equiv)
    look5 = LOCKED['breakout_5m_lookback']
    df['hh5'] = df['high'].shift(1).rolling(look5).max()
    df['ll5'] = df['low'].shift(1).rolling(look5).min()
    df['break5_up'] = df['close'] > df['hh5']
    df['break5_dn'] = df['close'] < df['ll5']

    # 15m breakout — aggregate to 15m, check
    df_15m = df.set_index('timestamp').resample('15min').agg({
        'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last', 'volume': 'sum'
    }).dropna()
    look15 = LOCKED['breakout_15m_lookback']
    df_15m['hh15'] = df_15m['high'].shift(1).rolling(look15).max()
    df_15m['ll15'] = df_15m['low'].shift(1).rolling(look15).min()
    df_15m['break15_up'] = df_15m['close'] > df_15m['hh15']
    df_15m['break15_dn'] = df_15m['close'] < df_15m['ll15']
    # forward-fill onto 5m grid
    df = df.merge(df_15m[['break15_up', 'break15_dn']].reset_index(),
                  on='timestamp', how='left')
    df[['break15_up', 'break15_dn']] = df[['break15_up', 'break15_dn']].ffill()

    # 1h trend SMA
    df_1h = df.set_index('timestamp').resample('1h').agg({
        'close': 'last'
    }).dropna()
    look1h = LOCKED['trend_1h_sma_lookback']
    df_1h['sma_1h'] = df_1h['close'].rolling(look1h).mean()
    df_1h['trend_up'] = df_1h['close'] > df_1h['sma_1h']
    df = df.merge(df_1h[['trend_up']].reset_index(), on='timestamp', how='left')
    df['trend_up'] = df['trend_up'].ffill()

    # body, ATR on 5m
    df['body'] = (df['close'] - df['open']).abs()
    df['range'] = df['high'] - df['low']
    df['body_ratio'] = df['body'] / df['range'].replace(0, np.nan)
    prev_close = df['close'].shift(1)
    tr = pd.concat([df['high'] - df['low'],
                    (df['high'] - prev_close).abs(),
                    (df['low'] - prev_close).abs()], axis=1).max(axis=1)
    df['atr'] = tr.rolling(LOCKED['atr_period']).mean()
    return df


def gate_A_events(df: pd.DataFrame) -> dict:
    long_cf = df['break5_up'] & df['break15_up'] & df['trend_up']
    short_cf = df['break5_dn'] & df['break15_dn'] & (~df['trend_up'].fillna(True))
    n_l = int(long_cf.sum())
    n_s = int(short_cf.sum())
    return {
        'long_confluence': n_l,
        'short_confluence': n_s,
        'total': n_l + n_s,
        'gate_min': GATES['min_confluence_events'],
        'pass': (n_l + n_s) >= GATES['min_confluence_events'],
    }


def gate_B_body(df: pd.DataFrame) -> dict:
    long_cf = df['break5_up'] & df['break15_up'] & df['trend_up']
    short_cf = df['break5_dn'] & df['break15_dn'] & (~df['trend_up'].fillna(True))
    cf = long_cf | short_cf
    after = cf & (df['body_ratio'] >= LOCKED['body_min_ratio'])
    n_pre = int(cf.sum())
    n_post = int(after.sum())
    retention = n_post / n_pre if n_pre > 0 else 0
    return {
        'pre': n_pre, 'post': n_post, 'retention': retention,
        'gate_min': GATES['body_filter_min_retention'],
        'pass': retention >= GATES['body_filter_min_retention'],
    }


def run_strategy(df: pd.DataFrame) -> pd.DataFrame:
    df = df.reset_index(drop=True)
    n = len(df)
    fric = LOCKED['friction_pct'] / 100.0
    sl_mult = LOCKED['sl_atr_mult']
    tp_mult = LOCKED['tp_atr_mult']
    max_hold = LOCKED['max_hold_bars']
    cooldown = LOCKED['cooldown_bars']

    trades = []
    in_pos = False; entry_idx = None; entry_price = None
    direction = 0; sl = None; tp = None; bars_held = 0
    last_exit_idx = -cooldown - 1

    for i in range(n - 1):
        row = df.iloc[i]
        next_row = df.iloc[i + 1]

        if in_pos:
            bars_held += 1
            high = row['high']; low = row['low']
            exit_reason = None; exit_price = None
            if direction == 1:
                if high >= tp: exit_price = tp; exit_reason = 'TP'
                elif low <= sl: exit_price = sl; exit_reason = 'SL'
                elif bars_held >= max_hold: exit_price = row['close']; exit_reason = 'TIMEOUT'
            else:
                if low <= tp: exit_price = tp; exit_reason = 'TP'
                elif high >= sl: exit_price = sl; exit_reason = 'SL'
                elif bars_held >= max_hold: exit_price = row['close']; exit_reason = 'TIMEOUT'

            if exit_reason:
                gross = (exit_price - entry_price) / entry_price * 100 * direction
                friction = 2 * fric * 100
                net = gross - friction
                trades.append({
                    'entry_time': df.iloc[entry_idx]['timestamp'],
                    'exit_time': row['timestamp'],
                    'direction': direction, 'entry_price': entry_price,
                    'exit_price': exit_price, 'gross_pct': gross,
                    'friction_pct': friction, 'net_pct': net,
                    'exit_reason': exit_reason, 'bars_held': bars_held,
                })
                in_pos = False
                last_exit_idx = i
                continue

        if not in_pos and (i - last_exit_idx) > cooldown:
            br5_u = row.get('break5_up', False)
            br5_d = row.get('break5_dn', False)
            br15_u = row.get('break15_up', False) if pd.notna(row.get('break15_up')) else False
            br15_d = row.get('break15_dn', False) if pd.notna(row.get('break15_dn')) else False
            tr_up = row.get('trend_up', False) if pd.notna(row.get('trend_up')) else False
            br_filter = row.get('body_ratio')
            atr = row.get('atr')

            if pd.isna(br_filter) or pd.isna(atr) or br_filter < LOCKED['body_min_ratio']:
                continue

            new_dir = 0
            if br5_u and br15_u and tr_up:
                new_dir = 1
            elif br5_d and br15_d and not tr_up:
                new_dir = -1
            if new_dir == 0:
                continue

            entry_price = next_row['open']
            entry_idx = i + 1
            direction = new_dir
            sl = entry_price - direction * sl_mult * atr
            tp = entry_price + direction * tp_mult * atr
            in_pos = True
            bars_held = 0

    return pd.DataFrame(trades)


def to_daily(trades, df):
    df_d = df.copy(); df_d['date'] = df_d['timestamp'].dt.date
    all_dates = pd.DataFrame({'date': sorted(df_d.date.unique())})
    if trades.empty:
        return all_dates.assign(daily_net_pct=0.0, n_trades=0)
    t = trades.copy()
    t['date'] = pd.to_datetime(t['exit_time']).dt.date
    daily = t.groupby('date').agg(daily_net_pct=('net_pct', 'sum'),
                                  n_trades=('net_pct', 'count')).reset_index()
    return all_dates.merge(daily, on='date', how='left').fillna(0)


def summarize(trades, daily):
    if trades.empty:
        return {'n_trades': 0, 'cum_net_pct': 0, 'avg_daily_net_pct': 0,
                'avg_trades_per_day': 0, 'win_rate': 0, 'rr_ratio': 0,
                'avg_gross_per_trade_pct': 0, 'worst_5d_pct': 0,
                'sharpe_annualized': 0, 'max_dd_pct': 0}
    n = len(trades)
    cum_net = float((1 + daily['daily_net_pct'] / 100).prod() - 1) * 100
    avg_daily_net = float(daily['daily_net_pct'].mean())
    freq = float(daily['n_trades'].mean())
    wr = float((trades['net_pct'] > 0).mean())
    avg_gross = float(trades['gross_pct'].mean())
    wins = trades[trades['net_pct'] > 0]
    losses = trades[trades['net_pct'] < 0]
    avg_win = float(wins['net_pct'].mean()) if len(wins) > 0 else 0
    avg_loss = float(losses['net_pct'].mean()) if len(losses) > 0 else 0
    rr = abs(avg_win / avg_loss) if avg_loss != 0 else 0
    nav = (1 + daily['daily_net_pct'].values / 100).cumprod()
    peak = np.maximum.accumulate(nav)
    dd = (peak - nav) / peak
    max_dd = float(dd.max()) * 100
    rolling_5d = pd.Series(daily['daily_net_pct'].values).rolling(5).apply(
        lambda x: (1 + x / 100).prod() - 1
    ) * 100
    worst_5d = float(rolling_5d.min())
    daily_std = float(daily['daily_net_pct'].std())
    sharpe = (avg_daily_net / daily_std * (365 ** 0.5)) if daily_std > 0 else 0.0
    return {
        'n_trades': int(n),
        'cum_net_pct': cum_net,
        'avg_daily_net_pct': avg_daily_net,
        'avg_trades_per_day': freq,
        'win_rate': wr,
        'avg_gross_per_trade_pct': avg_gross,
        'avg_win_pct': avg_win,
        'avg_loss_pct': avg_loss,
        'rr_ratio': rr,
        'max_dd_pct': max_dd,
        'worst_5d_pct': worst_5d,
        'sharpe_annualized': sharpe,
        'long_share': float((trades['direction'] == 1).mean()),
        'exit_distribution': trades['exit_reason'].value_counts().to_dict(),
    }


def main():
    print('=' * 100)
    print('Path B R10 — Multi-TF Confluence Breakout (round 20)')
    print('=' * 100)
    print('Pre-reg: 59e905c. EV pre-logged: P(T4 PASS) ~5-10%.\n')

    df = load_data()
    df = compute_features(df)
    print(f'BTC 5m: {len(df):,} bars, {df.timestamp.min()} → {df.timestamp.max()}')
    print(f'Days: {df.timestamp.dt.date.nunique()}\n')

    print('=== Gate A — Confluence events ===')
    gA = gate_A_events(df)
    print(f'  long: {gA["long_confluence"]}, short: {gA["short_confluence"]}, total: {gA["total"]}')
    print(f'  → {"PASS" if gA["pass"] else "FAIL"}\n')

    print('=== Gate B — Body filter retention ===')
    gB = gate_B_body(df)
    print(f'  pre: {gB["pre"]}, post: {gB["post"]}, retention: {gB["retention"]:.4f}')
    print(f'  → {"PASS" if gB["pass"] else "FAIL"}\n')

    if not gA['pass'] or not gB['pass']:
        print('EARLY EXIT: vacuous')
        out = {'pre_reg_commit': '59e905c', 'verdict': 'INCONCLUSIVE_VACUOUS',
               'gate_A': gA, 'gate_B': gB, 'locked': LOCKED, 'gates': GATES}
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        p = RESULTS / f'path_b_r10_mtf_oos_{ts}.json'
        with open(p, 'w') as fp: json.dump(out, fp, indent=2, default=str)
        print(f'Saved: {p}')
        return

    print('=== Run strategy ===')
    trades = run_strategy(df)
    daily = to_daily(trades, df)
    s = summarize(trades, daily)
    for k, v in s.items():
        if isinstance(v, dict):
            print(f'  {k}: {v}')
        elif isinstance(v, float):
            print(f'  {k}: {v:+.4f}')
        else:
            print(f'  {k}: {v}')
    print()

    if s['n_trades'] == 0:
        print('NO TRADES.')
        return

    print('=== Test 1 — WF 5-fold ===')
    folds = []
    n = len(df)
    fs = n // 6
    for i in range(5):
        ss = (i + 1) * fs
        ee = min(ss + fs, n)
        sub = compute_features(df.iloc[ss:ee][['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy())
        st = run_strategy(sub)
        sd = to_daily(st, sub)
        sf = summarize(st, sd)
        folds.append({'fold': i + 1, **sf})
        print(f'  fold {i+1}: trades={sf["n_trades"]}  cum={sf["cum_net_pct"]:+.4f}%  daily={sf["avg_daily_net_pct"]:+.4f}%  wr={sf["win_rate"]:.3f}')
    pos = sum(1 for f in folds if f['cum_net_pct'] > 0)
    t1 = pos >= GATES['wf_min_pos']
    print(f'  → {"PASS" if t1 else "FAIL"}  ({pos}/5)\n')

    print('=== Test 2 — Bootstrap 1000 × 3-day ===')
    nets = daily['daily_net_pct'].values
    win = GATES['bs_window_days']
    if len(nets) <= win:
        t2 = {'pass': False}
    else:
        random.seed(42)
        starts = random.sample(range(len(nets) - win), min(1000, len(nets) - win))
        cums = [(1 + nets[s:s+win] / 100).prod() - 1 for s in starts]
        arr = np.array(cums) * 100
        t2 = {'pos_rate': float((arr > 0).mean()), 'mean': float(arr.mean()),
              'p5': float(np.percentile(arr, 5)),
              'pass': float((arr > 0).mean()) >= GATES['bs_min_pos_rate']}
    print(f'  pos_rate: {t2.get("pos_rate", 0):.4f}  mean: {t2.get("mean", 0):+.4f}%  → {"PASS" if t2["pass"] else "FAIL"}\n')

    print('=== Test 3 — Train/Test 60/40 ===')
    split = int(n * GATES['tt_split'])
    df_tr = compute_features(df.iloc[:split][['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy())
    df_te = compute_features(df.iloc[split:][['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy())
    tr_t = run_strategy(df_tr); te_t = run_strategy(df_te)
    tr_d = to_daily(tr_t, df_tr); te_d = to_daily(te_t, df_te)
    s_tr = summarize(tr_t, tr_d); s_te = summarize(te_t, te_d)
    t3 = s_tr['cum_net_pct'] > 0 and s_te['cum_net_pct'] > 0
    print(f'  train: trades={s_tr["n_trades"]}  cum={s_tr["cum_net_pct"]:+.4f}%')
    print(f'  test:  trades={s_te["n_trades"]}  cum={s_te["cum_net_pct"]:+.4f}%')
    print(f'  → {"PASS" if t3 else "FAIL"}\n')

    t4 = s['avg_daily_net_pct'] >= GATES['magnitude_min_daily_pct']
    t5 = s['win_rate'] >= GATES['wr_min']
    t6 = s['rr_ratio'] >= GATES['rr_min']
    t7 = s['avg_trades_per_day'] >= GATES['trades_per_day_min']
    t8 = s['avg_gross_per_trade_pct'] > GATES['per_trade_gross_min_pct']
    t9 = s['worst_5d_pct'] >= -GATES['tail_max_5d_dd_pct']

    print(f'T4 daily≥0.2%:    {s["avg_daily_net_pct"]:+.4f}%  → {"PASS" if t4 else "FAIL"} (HARD)')
    print(f'T5 WR≥30%:        {s["win_rate"]:.4f}  → {"PASS" if t5 else "FAIL"}')
    print(f'T6 R:R≥1.5:       {s["rr_ratio"]:.4f}  → {"PASS" if t6 else "FAIL"}')
    print(f'T7 ≥2 trades/d:   {s["avg_trades_per_day"]:.4f}  → {"PASS" if t7 else "FAIL"} (HARD)')
    print(f'T8 gross>0.07%:   {s["avg_gross_per_trade_pct"]:+.4f}%  → {"PASS" if t8 else "FAIL"} (HARD)')
    print(f'T9 5d≥-15%:       {s["worst_5d_pct"]:+.4f}%  → {"PASS" if t9 else "FAIL"}\n')

    hard = t1 and t2['pass'] and t3 and t4 and t7 and t8

    print('=' * 100)
    print('FINAL VERDICT — R10')
    print('=' * 100)
    print(f'  HARD: {"PASS" if hard else "FAIL"}')
    print(f'  Sharpe: {s["sharpe_annualized"]:+.2f}, MDD: {s["max_dd_pct"]:+.2f}%')

    out = {
        'date': datetime.now(timezone.utc).isoformat(),
        'pre_reg_commit': '59e905c',
        'verdict': 'HARD_PASS' if hard else 'FAIL',
        'locked': LOCKED, 'gates': GATES,
        'gate_A': gA, 'gate_B': gB,
        'full_sample': s,
        'wf': {'folds': folds, 'pos': pos, 'pass': t1},
        'bootstrap_3d': t2,
        'train_test': {'train': s_tr, 'test': s_te, 'pass': t3},
        'tests': {
            'T4': {'value': s['avg_daily_net_pct'], 'pass': t4, 'hard': True},
            'T5': {'value': s['win_rate'], 'pass': t5},
            'T6': {'value': s['rr_ratio'], 'pass': t6},
            'T7': {'value': s['avg_trades_per_day'], 'pass': t7, 'hard': True},
            'T8': {'value': s['avg_gross_per_trade_pct'], 'pass': t8, 'hard': True},
            'T9': {'value': s['worst_5d_pct'], 'pass': t9},
        },
        'hard_pass': bool(hard),
    }
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    p = RESULTS / f'path_b_r10_mtf_oos_{ts}.json'
    with open(p, 'w') as fp: json.dump(out, fp, indent=2, default=str)
    print(f'\nSaved: {p}')


if __name__ == '__main__':
    main()
