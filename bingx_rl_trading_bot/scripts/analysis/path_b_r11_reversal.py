"""Path B R11 — Lo-MacKinlay Extreme-Move Reversal LOCKED OOS.

Pre-reg: claudedocs/path_b_r11_lo_mackinlay_reversal_prereg.md (commit 6a76e83)

Mechanism: |5m return| ≥ 0.5% triggers reversal entry. Hold 6 bars.
ATR(14)-based exits.
"""
import json, random, sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
DATA = ROOT / 'data'
RESULTS = ROOT / 'results'

PRICE_FILE = DATA / 'btc_5m_720days_binance.csv'

LOCKED = {
    'asset': 'BTC/USDT', 'extreme_threshold_pct': 0.5,
    'atr_period': 14, 'sl_atr_mult': 0.5, 'tp_atr_mult': 1.0,
    'max_hold_bars': 6, 'cooldown_bars': 3,
    'friction_pct': 0.07,
}

GATES = {
    'min_events': 1000, 'random_pct': 0.95,
    'wf_min_pos': 3, 'wf_total': 5,
    'bs_n_iter': 1000, 'bs_window_days': 3, 'bs_min_pos_rate': 0.50,
    'tt_split': 0.60,
    'magnitude_min_daily_pct': 0.20, 'wr_min': 0.30, 'rr_min': 1.5,
    'trades_per_day_min': 2.0, 'per_trade_gross_min_pct': 0.07,
    'tail_max_5d_dd_pct': 15.0,
}


def load_data():
    df = pd.read_csv(PRICE_FILE, usecols=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df.sort_values('timestamp').reset_index(drop=True)


def compute_features(df):
    df = df.copy()
    df['ret_5m'] = df['close'].pct_change() * 100
    prev = df['close'].shift(1)
    tr = pd.concat([df['high']-df['low'], (df['high']-prev).abs(), (df['low']-prev).abs()], axis=1).max(axis=1)
    df['atr'] = tr.rolling(LOCKED['atr_period']).mean()
    return df


def gate_A(df):
    n = int((df['ret_5m'].abs() >= LOCKED['extreme_threshold_pct']).sum())
    return {'events': n, 'pass': n >= GATES['min_events'], 'gate': GATES['min_events']}


def gate_B_reversion_check(df):
    """Avg next-6-bar return vs trigger-bar return sign."""
    th = LOCKED['extreme_threshold_pct']
    df = df.copy()
    df['fwd_6bar_ret'] = (df['close'].shift(-6) / df['close'] - 1) * 100
    extreme = df[df['ret_5m'].abs() >= th].dropna(subset=['fwd_6bar_ret'])
    if len(extreme) == 0:
        return {'pass': False, 'note': 'no events'}
    # Reversion = sign(fwd_ret) opposite to sign(ret_5m). Compute avg fwd_ret * (-sign(ret_5m))
    aligned = extreme['fwd_6bar_ret'] * -np.sign(extreme['ret_5m'])
    avg_aligned = float(aligned.mean())
    return {
        'avg_aligned_fwd_return_pct': avg_aligned,
        'n_events': int(len(extreme)),
        'pass': avg_aligned > 0,  # positive means reversion direction works
        'note': 'positive = mean reversion observed in raw data',
    }


def run_strategy(df):
    df = df.reset_index(drop=True)
    n = len(df)
    fric = LOCKED['friction_pct'] / 100.0
    sl_m = LOCKED['sl_atr_mult']; tp_m = LOCKED['tp_atr_mult']
    max_hold = LOCKED['max_hold_bars']; cooldown = LOCKED['cooldown_bars']
    th = LOCKED['extreme_threshold_pct']

    trades = []
    in_pos = False; entry_idx = None; entry_price = None
    direction = 0; sl = None; tp = None; bars_held = 0
    last_exit_idx = -cooldown - 1

    for i in range(n - 1):
        row = df.iloc[i]; nxt = df.iloc[i + 1]

        if in_pos:
            bars_held += 1
            high = row['high']; low = row['low']
            xr = None; xp = None
            if direction == 1:
                if high >= tp: xp = tp; xr = 'TP'
                elif low <= sl: xp = sl; xr = 'SL'
                elif bars_held >= max_hold: xp = row['close']; xr = 'TIMEOUT'
            else:
                if low <= tp: xp = tp; xr = 'TP'
                elif high >= sl: xp = sl; xr = 'SL'
                elif bars_held >= max_hold: xp = row['close']; xr = 'TIMEOUT'
            if xr:
                gross = (xp - entry_price) / entry_price * 100 * direction
                friction = 2 * fric * 100
                trades.append({
                    'entry_time': df.iloc[entry_idx]['timestamp'],
                    'exit_time': row['timestamp'], 'direction': direction,
                    'entry_price': entry_price, 'exit_price': xp,
                    'gross_pct': gross, 'net_pct': gross - friction,
                    'exit_reason': xr, 'bars_held': bars_held,
                })
                in_pos = False; last_exit_idx = i
                continue

        if not in_pos and (i - last_exit_idx) > cooldown:
            r = row.get('ret_5m'); atr = row.get('atr')
            if pd.isna(r) or pd.isna(atr): continue
            if abs(r) < th: continue
            new_dir = -1 if r > 0 else 1  # REVERSAL
            entry_price = nxt['open']
            entry_idx = i + 1
            direction = new_dir
            sl = entry_price - direction * sl_m * atr
            tp = entry_price + direction * tp_m * atr
            in_pos = True; bars_held = 0

    return pd.DataFrame(trades)


def to_daily(trades, df):
    df_d = df.copy(); df_d['date'] = df_d['timestamp'].dt.date
    all_dates = pd.DataFrame({'date': sorted(df_d.date.unique())})
    if trades.empty: return all_dates.assign(daily_net_pct=0.0, n_trades=0)
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
        'exit_distribution': trades['exit_reason'].value_counts().to_dict(),
    }


def main():
    print('='*100)
    print('Path B R11 — Lo-MacKinlay Extreme-Move Reversal')
    print('='*100)
    print('Pre-reg: 6a76e83. Theory: Lo-MacKinlay 1988.')
    print('EV pre-logged: P(T4 PASS) ~8-15%.\n')

    df = load_data()
    df = compute_features(df)
    print(f'BTC 5m: {len(df):,} bars, days: {df.timestamp.dt.date.nunique()}\n')

    print('=== Gate A — Extreme move events ===')
    gA = gate_A(df)
    print(f'  events: {gA["events"]}  → {"PASS" if gA["pass"] else "FAIL"}\n')

    print('=== Gate B — Reversion sanity check (informational) ===')
    gB = gate_B_reversion_check(df)
    print(f'  avg aligned fwd 6-bar return: {gB.get("avg_aligned_fwd_return_pct", 0):+.5f}%')
    print(f'  → {"REVERSION OBSERVED" if gB["pass"] else "NO REVERSION (continuation)"}\n')

    if not gA['pass']:
        print('EARLY EXIT: vacuous')
        return

    print('=== Run strategy ===')
    trades = run_strategy(df)
    daily = to_daily(trades, df)
    s = summarize(trades, daily)
    for k, v in s.items():
        if isinstance(v, dict): print(f'  {k}: {v}')
        elif isinstance(v, float): print(f'  {k}: {v:+.4f}')
        else: print(f'  {k}: {v}')
    print()

    if s['n_trades'] == 0: return

    print('=== Test 1 — WF 5-fold ===')
    folds = []; n = len(df); fs = n // 6
    for i in range(5):
        ss = (i+1)*fs; ee = min(ss+fs, n)
        sub = compute_features(df.iloc[ss:ee][['timestamp','open','high','low','close','volume']].copy())
        st = run_strategy(sub); sd = to_daily(st, sub); sf = summarize(st, sd)
        folds.append({'fold': i+1, **sf})
        print(f'  fold {i+1}: trades={sf["n_trades"]}  cum={sf["cum_net_pct"]:+.4f}%  daily={sf["avg_daily_net_pct"]:+.4f}%  wr={sf["win_rate"]:.3f}')
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
    df_tr = compute_features(df.iloc[:split][['timestamp','open','high','low','close','volume']].copy())
    df_te = compute_features(df.iloc[split:][['timestamp','open','high','low','close','volume']].copy())
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

    print('='*100)
    print('FINAL VERDICT — R11')
    print('='*100)
    print(f'  HARD: {"PASS" if hard else "FAIL"}')
    print(f'  Sharpe: {s["sharpe_annualized"]:+.2f}, MDD: {s["max_dd_pct"]:+.2f}%')
    print(f'  Reversion observed in data: {gB.get("avg_aligned_fwd_return_pct", 0):+.5f}%')

    out = {
        'date': datetime.now(timezone.utc).isoformat(),
        'pre_reg_commit': '6a76e83',
        'verdict': 'HARD_PASS' if hard else 'FAIL',
        'gate_A': gA, 'gate_B_reversion_check': gB,
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
        'locked': LOCKED, 'gates': GATES,
    }
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    p = RESULTS / f'path_b_r11_reversal_oos_{ts}.json'
    with open(p, 'w') as fp: json.dump(out, fp, indent=2, default=str)
    print(f'\nSaved: {p}')


if __name__ == '__main__':
    main()
