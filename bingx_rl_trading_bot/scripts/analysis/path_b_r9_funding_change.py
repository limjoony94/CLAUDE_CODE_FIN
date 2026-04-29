"""Path B R9 — BTC Funding-Change Momentum LOCKED OOS.

Pre-reg: claudedocs/path_b_r9_funding_change_momentum_prereg.md (commit bc0ce10)

Mechanism: Δ(7d funding mean) signals positioning shift.
  Δ ≥ +0.005%/8h → LONG (positions accumulating bullish)
  Δ ≤ -0.005%/8h → SHORT (positions exiting / bearish)
  Hold 21 periods (7 days). Friction 0.07% × 2 = 0.14% RT.

DISCLOSURE: 0/1 LIVE-parity track record. T7 anticipated FAIL by design.
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

FUND_FILE = DATA / 'funding_history.parquet'
PRICE_FILE = DATA / 'btc_perp_daily_binance.parquet'

LOCKED = {
    'asset': 'BTC/USDT',
    'change_window_periods': 21,             # 7d × 3/day
    'entry_threshold_pct_per_8h': 0.005,
    'hold_periods': 21,
    'friction_pct': 0.07,
    'capital_usd': 1500,
}

GATES = {
    'min_events': 100,
    'random_pct': 0.95,
    'wf_min_pos': 3, 'wf_total': 5,
    'bs_n_iter': 1000, 'bs_window_days': 3, 'bs_min_pos_rate': 0.50,
    'tt_split': 0.60,
    'magnitude_min_daily_pct': 0.20,    # HARD
    'wr_min': 0.30,                      # RELAXED via A
    'rr_min': 1.0,
    'trades_per_day_min': 2.0,           # HARD
    'per_trade_gross_min_pct': 0.07,     # HARD
    'tail_max_5d_dd_pct': 15.0,          # RELAXED via E
}


def load_data():
    fund = pd.read_parquet(FUND_FILE)
    btc_fund = fund[fund['symbol'] == LOCKED['asset']].copy()
    btc_fund['datetime'] = pd.to_datetime(btc_fund['datetime'])
    btc_fund = btc_fund.sort_values('datetime').reset_index(drop=True)

    price = pd.read_parquet(PRICE_FILE)
    price['date'] = pd.to_datetime(price['date'])
    price = price[['date', 'close']].rename(columns={'close': 'btc_close'})

    # Map each 8h funding period to nearest day's close for return computation
    btc_fund['date'] = btc_fund['datetime'].dt.tz_localize(None).dt.floor('D')
    df = btc_fund.merge(price, on='date', how='left')
    return df


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    look = LOCKED['change_window_periods']
    df['curr_7d_mean'] = df['funding_rate'].rolling(look).mean()
    df['prior_7d_mean'] = df['curr_7d_mean'].shift(look)
    df['delta_7d'] = df['curr_7d_mean'] - df['prior_7d_mean']
    return df


def gate_A_events(df: pd.DataFrame) -> dict:
    th = LOCKED['entry_threshold_pct_per_8h'] / 100.0
    n = int((df['delta_7d'].abs() >= th).sum())
    return {
        'events': n,
        'gate_min': GATES['min_events'],
        'pass': n >= GATES['min_events'],
    }


def gate_B_predictive(df: pd.DataFrame) -> dict:
    """Check Δ(t) vs subsequent N-period BTC return."""
    if df['btc_close'].isna().any():
        df = df.dropna(subset=['btc_close'])
    n_fwd = LOCKED['hold_periods']
    df = df.copy().reset_index(drop=True)
    df['fwd_ret'] = df['btc_close'].shift(-n_fwd) / df['btc_close'] - 1
    valid = df.dropna(subset=['delta_7d', 'fwd_ret'])
    if len(valid) < 100:
        return {'pass': False, 'corr': None, 'note': 'insufficient'}
    corr = float(valid['delta_7d'].corr(valid['fwd_ret']))
    return {
        'corr_delta_vs_fwd_ret': corr,
        'n': int(len(valid)),
        'pass': corr > 0,
        'note': 'positive corr means continuation; negative means counter-trade direction'
    }


def run_strategy(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy().reset_index(drop=True)
    th = LOCKED['entry_threshold_pct_per_8h'] / 100.0
    hold = LOCKED['hold_periods']
    fric = LOCKED['friction_pct'] / 100.0

    trades = []
    in_pos = False
    entry_idx = None
    direction = 0
    bars_held = 0

    for i, row in df.iterrows():
        if pd.isna(row['delta_7d']) or pd.isna(row['btc_close']):
            continue

        if in_pos:
            bars_held += 1
            if bars_held >= hold:
                exit_price = row['btc_close']
                entry_price = df.iloc[entry_idx]['btc_close']
                gross = (exit_price - entry_price) / entry_price * 100 * direction
                friction = 2 * fric * 100
                net = gross - friction
                trades.append({
                    'entry_time': df.iloc[entry_idx]['datetime'],
                    'exit_time': row['datetime'],
                    'direction': direction,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'gross_pct': gross,
                    'friction_pct': friction,
                    'net_pct': net,
                    'bars_held': bars_held,
                    'delta_7d_at_entry': df.iloc[entry_idx]['delta_7d'],
                })
                in_pos = False
                continue

        if not in_pos:
            d = row['delta_7d']
            if abs(d) >= th:
                direction = 1 if d > 0 else -1
                entry_idx = i
                in_pos = True
                bars_held = 0

    return pd.DataFrame(trades)


def to_daily(trades: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    df_d = df.copy()
    df_d['date'] = df_d['datetime'].dt.date
    all_dates = pd.DataFrame({'date': sorted(df_d.date.unique())})
    if trades.empty:
        return all_dates.assign(daily_net_pct=0.0, n_trades=0)
    t = trades.copy()
    t['date'] = pd.to_datetime(t['exit_time']).dt.date
    daily = t.groupby('date').agg(daily_net_pct=('net_pct', 'sum'),
                                  n_trades=('net_pct', 'count')).reset_index()
    return all_dates.merge(daily, on='date', how='left').fillna(0)


def summarize(trades: pd.DataFrame, daily: pd.DataFrame) -> dict:
    if trades.empty:
        return {'n_trades': 0, 'cum_net_pct': 0, 'avg_daily_net_pct': 0,
                'avg_trades_per_day': 0, 'win_rate': 0, 'rr_ratio': 0,
                'avg_gross_per_trade_pct': 0, 'worst_5d_pct': 0,
                'sharpe_annualized': 0, 'max_dd_pct': 0}
    n = len(trades)
    cum_net = float((1 + daily['daily_net_pct'] / 100).prod() - 1) * 100
    avg_daily_net = float(daily['daily_net_pct'].mean())
    avg_freq = float(daily['n_trades'].mean())
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
        'avg_trades_per_day': avg_freq,
        'win_rate': wr,
        'avg_gross_per_trade_pct': avg_gross,
        'avg_win_pct': avg_win,
        'avg_loss_pct': avg_loss,
        'rr_ratio': rr,
        'max_dd_pct': max_dd,
        'worst_5d_pct': worst_5d,
        'sharpe_annualized': sharpe,
        'long_share': float((trades['direction'] == 1).mean()),
    }


def main():
    print('=' * 100)
    print('Path B R9 — BTC Funding-Change Momentum')
    print('=' * 100)
    print('Pre-reg: claudedocs/path_b_r9_funding_change_momentum_prereg.md (bc0ce10)')
    print('DISCLOSURE: 0/1 LIVE-parity. T7 anticipated FAIL (low-freq mechanism).\n')

    df = load_data()
    df = compute_features(df)
    print(f'BTC funding+price: {len(df):,} 8h periods, '
          f'{df.datetime.min().date()} → {df.datetime.max().date()}\n')

    print('=== Gate A — Sufficient events ===')
    gA = gate_A_events(df)
    print(f'  events: {gA["events"]}  → {"PASS" if gA["pass"] else "FAIL"}\n')

    print('=== Gate B — Predictive direction ===')
    gB = gate_B_predictive(df)
    print(f'  Corr(Δ_t, fwd_ret_t+21): {gB.get("corr_delta_vs_fwd_ret", 0):+.5f}  '
          f'n={gB.get("n", 0)}')
    print(f'  → {"PASS" if gB["pass"] else "FAIL"}  ({gB.get("note", "")})\n')

    if not gA['pass']:
        out = {'date': datetime.now(timezone.utc).isoformat(),
               'pre_reg_commit': 'bc0ce10', 'verdict': 'INCONCLUSIVE_VACUOUS',
               'gate_A': gA, 'gate_B': gB, 'locked': LOCKED, 'gates': GATES}
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        p = RESULTS / f'path_b_r9_funding_change_oos_{ts}.json'
        with open(p, 'w') as fp:
            json.dump(out, fp, indent=2, default=str)
        print(f'Saved: {p}')
        return

    if not gB['pass']:
        print('Gate B FAIL: continuation hypothesis wrong direction.')
        print('Per pre-reg: report this informational finding without flipping.')
        # We still run the strategy as designed (LONG on +Δ) and report negative result.

    print('=== Run strategy ===')
    trades = run_strategy(df)
    daily = to_daily(trades, df)
    s = summarize(trades, daily)
    for k, v in s.items():
        if isinstance(v, float):
            print(f'  {k}: {v:+.4f}')
        else:
            print(f'  {k}: {v}')
    print()

    if s['n_trades'] == 0:
        out = {'verdict': 'NO_TRADES', 'gate_A': gA, 'gate_B': gB, 'full_sample': s}
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        p = RESULTS / f'path_b_r9_funding_change_oos_{ts}.json'
        with open(p, 'w') as fp:
            json.dump(out, fp, indent=2, default=str)
        return

    print('=== Test 1 — WF 5-fold ===')
    folds = []
    n = len(df)
    fs = n // 6
    for i in range(5):
        ss = (i + 1) * fs
        ee = min(ss + fs, n)
        sub = compute_features(df.iloc[ss:ee][['datetime', 'date', 'funding_rate', 'btc_close']].copy())
        st = run_strategy(sub)
        sd = to_daily(st, sub)
        sf = summarize(st, sd)
        folds.append({'fold': i + 1, **sf})
        print(f'  fold {i+1}: trades={sf["n_trades"]}  cum={sf["cum_net_pct"]:+.4f}%  '
              f'daily={sf["avg_daily_net_pct"]:+.4f}%')
    pos = sum(1 for f in folds if f['cum_net_pct'] > 0)
    t1_pass = pos >= GATES['wf_min_pos']
    print(f'  → {"PASS" if t1_pass else "FAIL"}  ({pos}/5)\n')

    print('=== Test 2 — Bootstrap 1000 × 3-day windows ===')
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
    print(f'  pos_rate: {t2.get("pos_rate", 0):.4f}  mean: {t2.get("mean", 0):+.4f}%')
    print(f'  → {"PASS" if t2["pass"] else "FAIL"}\n')

    split = int(n * 0.6)
    df_tr = compute_features(df.iloc[:split].copy())
    df_te = compute_features(df.iloc[split:].copy())
    tr_t = run_strategy(df_tr); te_t = run_strategy(df_te)
    tr_d = to_daily(tr_t, df_tr); te_d = to_daily(te_t, df_te)
    s_tr = summarize(tr_t, tr_d); s_te = summarize(te_t, te_d)
    t3_pass = s_tr['cum_net_pct'] > 0 and s_te['cum_net_pct'] > 0
    print(f'=== Test 3 ===')
    print(f'  train: trades={s_tr["n_trades"]}  cum={s_tr["cum_net_pct"]:+.4f}%')
    print(f'  test:  trades={s_te["n_trades"]}  cum={s_te["cum_net_pct"]:+.4f}%')
    print(f'  → {"PASS" if t3_pass else "FAIL"}\n')

    t4_pass = s['avg_daily_net_pct'] >= GATES['magnitude_min_daily_pct']
    t5_pass = s['win_rate'] >= GATES['wr_min']
    t6_pass = s['rr_ratio'] >= GATES['rr_min']
    t7_pass = s['avg_trades_per_day'] >= GATES['trades_per_day_min']
    t8_pass = s['avg_gross_per_trade_pct'] > GATES['per_trade_gross_min_pct']
    t9_pass = s['worst_5d_pct'] >= -GATES['tail_max_5d_dd_pct']

    print(f'T4 daily≥0.2%:    {s["avg_daily_net_pct"]:+.4f}%  → {"PASS" if t4_pass else "FAIL"} (HARD)')
    print(f'T5 WR≥30%:        {s["win_rate"]:.4f}  → {"PASS" if t5_pass else "FAIL"}')
    print(f'T6 R:R≥1.0:       {s["rr_ratio"]:.4f}  → {"PASS" if t6_pass else "FAIL"}')
    print(f'T7 ≥2 trades/d:   {s["avg_trades_per_day"]:.4f}  → {"PASS" if t7_pass else "FAIL"} (HARD, anticipated FAIL)')
    print(f'T8 gross>0.07%:   {s["avg_gross_per_trade_pct"]:+.4f}%  → {"PASS" if t8_pass else "FAIL"} (HARD)')
    print(f'T9 5d≥-15%:       {s["worst_5d_pct"]:+.4f}%  → {"PASS" if t9_pass else "FAIL"}\n')

    hard_pass = t1_pass and t2['pass'] and t3_pass and t4_pass and t7_pass and t8_pass

    print('=' * 100)
    print('FINAL VERDICT — R9')
    print('=' * 100)
    print(f'  HARD: {"PASS" if hard_pass else "FAIL"}')
    print(f'  Sharpe: {s["sharpe_annualized"]:+.2f}')

    out = {
        'date': datetime.now(timezone.utc).isoformat(),
        'pre_reg_commit': 'bc0ce10',
        'live_parity_prior': '0/1',
        'verdict': 'HARD_PASS' if hard_pass else 'FAIL',
        'locked': LOCKED, 'gates': GATES,
        'gate_A': gA, 'gate_B': gB,
        'full_sample': s,
        'wf': {'folds': folds, 'pos': pos, 'pass': t1_pass},
        'bootstrap_3d': t2,
        'train_test': {'train': s_tr, 'test': s_te, 'pass': t3_pass},
        'tests': {
            'T4': {'value': s['avg_daily_net_pct'], 'pass': t4_pass, 'hard': True},
            'T5': {'value': s['win_rate'], 'pass': t5_pass},
            'T6': {'value': s['rr_ratio'], 'pass': t6_pass},
            'T7': {'value': s['avg_trades_per_day'], 'pass': t7_pass, 'hard': True},
            'T8': {'value': s['avg_gross_per_trade_pct'], 'pass': t8_pass, 'hard': True},
            'T9': {'value': s['worst_5d_pct'], 'pass': t9_pass},
        },
        'hard_pass': bool(hard_pass),
    }
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    p = RESULTS / f'path_b_r9_funding_change_oos_{ts}.json'
    with open(p, 'w') as fp:
        json.dump(out, fp, indent=2, default=str)
    print(f'\nSaved: {p}')


if __name__ == '__main__':
    main()
