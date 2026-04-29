"""Path B R12 — Calendar Session Momentum LOCKED OOS.

Pre-reg: claudedocs/path_b_r12_calendar_session_prereg.md (commit 6cdabee)

Mechanism: enter at 07:00 UTC and 13:00 UTC each day, direction by SMA trend filter.
Hold 4 bars (4h). 1× leverage.
"""
import json, random, sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
DATA = ROOT / 'data'
RESULTS = ROOT / 'results'

PRICE_FILE = DATA / 'btc_1h_720days.csv'

LOCKED = {
    'session_hours_utc': [7, 13],
    'hold_bars': 4,
    'trend_filter_sma_periods': 24,
    'friction_pct': 0.07,
    'capital_usd': 1500,
}

GATES = {
    'random_pct': 0.95,
    'wf_min_pos': 3, 'wf_total': 5,
    'bs_n_iter': 1000, 'bs_window_days': 3, 'bs_min_pos_rate': 0.50,
    'tt_split': 0.60,
    'magnitude_min_daily_pct': 0.20, 'wr_min': 0.30, 'rr_min': 1.0,
    'trades_per_day_min': 2.0, 'per_trade_gross_min_pct': 0.07,
    'tail_max_5d_dd_pct': 15.0,
}


def load_data():
    df = pd.read_csv(PRICE_FILE, usecols=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    df['hour'] = df['timestamp'].dt.hour
    df['date'] = df['timestamp'].dt.date
    return df


def compute_features(df):
    df = df.copy()
    df['sma'] = df['close'].rolling(LOCKED['trend_filter_sma_periods']).mean()
    df['trend_up'] = df['close'] > df['sma']
    return df


def run_strategy(df):
    df = df.reset_index(drop=True)
    n = len(df)
    fric = LOCKED['friction_pct'] / 100.0
    hold = LOCKED['hold_bars']
    sessions = LOCKED['session_hours_utc']

    trades = []
    in_pos = False
    entry_idx = None
    direction = 0
    bars_held = 0

    for i in range(n - 1):
        row = df.iloc[i]
        hour = row['hour']

        if in_pos:
            bars_held += 1
            if bars_held >= hold:
                exit_price = row['close']
                entry_price = df.iloc[entry_idx]['close']
                gross = (exit_price - entry_price) / entry_price * 100 * direction
                friction = 2 * fric * 100
                trades.append({
                    'entry_time': df.iloc[entry_idx]['timestamp'],
                    'exit_time': row['timestamp'],
                    'direction': direction,
                    'entry_price': entry_price, 'exit_price': exit_price,
                    'gross_pct': gross, 'net_pct': gross - friction,
                    'session_hour': df.iloc[entry_idx]['hour'],
                    'bars_held': bars_held,
                })
                in_pos = False
                continue

        if not in_pos and hour in sessions:
            tu = row.get('trend_up')
            if pd.isna(tu):
                continue
            new_dir = 1 if tu else -1
            entry_idx = i
            direction = new_dir
            in_pos = True
            bars_held = 0

    return pd.DataFrame(trades)


def to_daily(trades, df):
    df_d = df.copy()
    all_dates = pd.DataFrame({'date': sorted(df_d.date.unique())})
    if trades.empty:
        return all_dates.assign(daily_net_pct=0.0, n_trades=0)
    t = trades.copy(); t['date'] = pd.to_datetime(t['exit_time']).dt.date
    daily = t.groupby('date').agg(daily_net_pct=('net_pct', 'sum'), n_trades=('net_pct', 'count')).reset_index()
    return all_dates.merge(daily, on='date', how='left').fillna(0)


def summarize(trades, daily):
    if trades.empty:
        return {'n_trades': 0, 'cum_net_pct': 0, 'avg_daily_net_pct': 0,
                'avg_trades_per_day': 0, 'win_rate': 0, 'rr_ratio': 0,
                'avg_gross_per_trade_pct': 0, 'worst_5d_pct': 0,
                'sharpe_annualized': 0, 'max_dd_pct': 0}
    n = len(trades)
    cum_net = float((1 + daily['daily_net_pct']/100).prod() - 1) * 100
    avg_daily = float(daily['daily_net_pct'].mean())
    freq = float(daily['n_trades'].mean())
    wr = float((trades['net_pct'] > 0).mean())
    avg_g = float(trades['gross_pct'].mean())
    wins = trades[trades['net_pct'] > 0]; losses = trades[trades['net_pct'] < 0]
    aw = float(wins['net_pct'].mean()) if len(wins) else 0
    al = float(losses['net_pct'].mean()) if len(losses) else 0
    rr = abs(aw/al) if al != 0 else 0
    nav = (1 + daily['daily_net_pct'].values/100).cumprod()
    peak = np.maximum.accumulate(nav); dd = (peak-nav)/peak; mdd = float(dd.max())*100
    r5 = pd.Series(daily['daily_net_pct'].values).rolling(5).apply(lambda x: (1+x/100).prod()-1) * 100
    w5 = float(r5.min())
    ds = float(daily['daily_net_pct'].std())
    sh = (avg_daily/ds * (365**0.5)) if ds > 0 else 0
    return {
        'n_trades': int(n), 'cum_net_pct': cum_net, 'avg_daily_net_pct': avg_daily,
        'avg_trades_per_day': freq, 'win_rate': wr,
        'avg_gross_per_trade_pct': avg_g, 'avg_win_pct': aw, 'avg_loss_pct': al,
        'rr_ratio': rr, 'max_dd_pct': mdd, 'worst_5d_pct': w5, 'sharpe_annualized': sh,
        'long_share': float((trades['direction'] == 1).mean()),
        'session_distribution': trades.groupby('session_hour').size().to_dict(),
    }


def main():
    print('='*100)
    print('Path B R12 — Calendar Session Momentum (07:00 + 13:00 UTC)')
    print('='*100)
    print('Pre-reg: 6cdabee. Theory: Bouri-Lau-Lucey 2019.\n')

    df = load_data()
    df = compute_features(df)
    print(f'BTC 1h: {len(df):,} bars, days: {df.date.nunique()}\n')

    trades = run_strategy(df)
    daily = to_daily(trades, df)
    s = summarize(trades, daily)

    print('=== Full-sample BT ===')
    for k, v in s.items():
        if isinstance(v, dict): print(f'  {k}: {v}')
        elif isinstance(v, float): print(f'  {k}: {v:+.4f}')
        else: print(f'  {k}: {v}')
    print()

    if s['n_trades'] == 0:
        print('NO TRADES.')
        return

    print('=== Test 1 — WF 5-fold ===')
    folds = []; n = len(df); fs = n // 6
    for i in range(5):
        ss = (i+1)*fs; ee = min(ss+fs, n)
        sub = compute_features(df.iloc[ss:ee][['timestamp','open','high','low','close','volume','hour','date']].copy())
        st = run_strategy(sub); sd = to_daily(st, sub); sf = summarize(st, sd)
        folds.append({'fold': i+1, **sf})
        print(f'  fold {i+1}: trades={sf["n_trades"]}  cum={sf["cum_net_pct"]:+.4f}%  daily={sf["avg_daily_net_pct"]:+.4f}%')
    pos = sum(1 for f in folds if f['cum_net_pct'] > 0)
    t1 = pos >= GATES['wf_min_pos']
    print(f'  → {"PASS" if t1 else "FAIL"}  ({pos}/5)\n')

    print('=== Test 2 — Bootstrap 1000 × 3-day ===')
    nets = daily['daily_net_pct'].values; win = GATES['bs_window_days']
    if len(nets) <= win:
        t2 = {'pass': False}
    else:
        random.seed(42)
        starts = random.sample(range(len(nets) - win), min(1000, len(nets) - win))
        cums = [(1 + nets[s:s+win]/100).prod() - 1 for s in starts]
        arr = np.array(cums) * 100
        t2 = {'pos_rate': float((arr > 0).mean()), 'mean': float(arr.mean()),
              'p5': float(np.percentile(arr, 5)),
              'pass': float((arr > 0).mean()) >= GATES['bs_min_pos_rate']}
    print(f'  pos_rate: {t2.get("pos_rate", 0):.4f}  mean: {t2.get("mean", 0):+.4f}%  → {"PASS" if t2["pass"] else "FAIL"}\n')

    print('=== Test 3 — Train/Test 60/40 ===')
    split = int(n * GATES['tt_split'])
    df_tr = compute_features(df.iloc[:split][['timestamp','open','high','low','close','volume','hour','date']].copy())
    df_te = compute_features(df.iloc[split:][['timestamp','open','high','low','close','volume','hour','date']].copy())
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
    print(f'T6 R:R≥1.0:       {s["rr_ratio"]:.4f}  → {"PASS" if t6 else "FAIL"}')
    print(f'T7 ≥2 trades/d:   {s["avg_trades_per_day"]:.4f}  → {"PASS" if t7 else "FAIL"} (HARD)')
    print(f'T8 gross>0.07%:   {s["avg_gross_per_trade_pct"]:+.4f}%  → {"PASS" if t8 else "FAIL"} (HARD)')
    print(f'T9 5d≥-15%:       {s["worst_5d_pct"]:+.4f}%  → {"PASS" if t9 else "FAIL"}\n')

    hard = t1 and t2['pass'] and t3 and t4 and t7 and t8

    print('='*100)
    print('FINAL VERDICT — R12')
    print('='*100)
    print(f'  HARD: {"PASS" if hard else "FAIL"}')
    print(f'  Sharpe: {s["sharpe_annualized"]:+.2f}, MDD: {s["max_dd_pct"]:+.2f}%')

    out = {
        'date': datetime.now(timezone.utc).isoformat(), 'pre_reg_commit': '6cdabee',
        'verdict': 'HARD_PASS' if hard else 'FAIL',
        'full_sample': s,
        'wf': {'folds': folds, 'pos': pos, 'pass': t1}, 'bootstrap_3d': t2,
        'train_test': {'train': s_tr, 'test': s_te, 'pass': t3},
        'tests': {
            'T4': {'value': s['avg_daily_net_pct'], 'pass': t4, 'hard': True},
            'T5': {'value': s['win_rate'], 'pass': t5},
            'T6': {'value': s['rr_ratio'], 'pass': t6},
            'T7': {'value': s['avg_trades_per_day'], 'pass': t7, 'hard': True},
            'T8': {'value': s['avg_gross_per_trade_pct'], 'pass': t8, 'hard': True},
            'T9': {'value': s['worst_5d_pct'], 'pass': t9},
        },
        'hard_pass': bool(hard), 'locked': LOCKED, 'gates': GATES,
    }
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    p = RESULTS / f'path_b_r12_calendar_oos_{ts}.json'
    with open(p, 'w') as fp: json.dump(out, fp, indent=2, default=str)
    print(f'\nSaved: {p}')


if __name__ == '__main__':
    main()
