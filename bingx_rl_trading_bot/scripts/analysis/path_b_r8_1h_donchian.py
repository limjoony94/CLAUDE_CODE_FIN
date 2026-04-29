"""Path B R8 — 1h Donchian Breakout with Static ATR TP/SL LOCKED OOS.

Pre-reg: claudedocs/path_b_r8_1h_donchian_breakout_prereg.md (commit 3e8f00e)

DISCLOSURE: BT methodology has 0/1 LIVE-parity (C1 failed -12.86%/14d).
R8 STATIC TP/SL specifically avoids C1's TRAILING_STOP_MARKET gap.

Logic:
  At each closed 1h bar t, check Donchian breakout (close > 24-bar high
  for long, close < 24-bar low for short). Body filter ≥ 0.40. Cooldown 1h.
  Entry at next bar open. SL = entry ∓ 1×ATR(14). TP = entry ± 3×ATR(14).
  Max hold 48h.
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

PRICE_FILE = DATA / 'btc_1h_720days.csv'

LOCKED = {
    'asset': 'BTC/USDT',
    'timeframe': '1h',
    'channel_lookback_bars': 24,
    'body_min_ratio': 0.40,
    'atr_period': 14,
    'sl_atr_mult': 1.0,
    'tp_atr_mult': 3.0,
    'max_hold_bars': 48,
    'friction_pct': 0.07,
    'cooldown_bars': 1,
}

GATES = {
    'min_breakout_events': 1000,
    'body_filter_min_retention': 0.40,
    'random_baseline_pct': 0.95,
    'wf_min_pos': 3, 'wf_total': 5,
    'bs_n_iter': 1000, 'bs_window_days': 3, 'bs_min_pos_rate': 0.50,
    'tt_split': 0.60,
    'magnitude_min_daily_pct': 0.20,    # HARD
    'wr_min': 0.30,                      # RELAXED via A
    'rr_min': 1.5,                       # RELAXED via A
    'trades_per_day_min': 2.0,           # HARD
    'per_trade_gross_min_pct': 0.07,     # HARD
    'tail_max_5d_dd_pct': 15.0,          # RELAXED via E
}


def load_data():
    df = pd.read_csv(PRICE_FILE, usecols=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    return df


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    look = LOCKED['channel_lookback_bars']
    df['channel_high'] = df['high'].shift(1).rolling(look).max()
    df['channel_low'] = df['low'].shift(1).rolling(look).min()
    df['body'] = (df['close'] - df['open']).abs()
    df['range'] = df['high'] - df['low']
    df['body_ratio'] = df['body'] / df['range'].replace(0, np.nan)

    prev_close = df['close'].shift(1)
    tr = pd.concat([
        df['high'] - df['low'],
        (df['high'] - prev_close).abs(),
        (df['low'] - prev_close).abs(),
    ], axis=1).max(axis=1)
    df['atr'] = tr.rolling(LOCKED['atr_period']).mean()
    return df


def gate_A_breakouts(df: pd.DataFrame) -> dict:
    long_break = df['close'] > df['channel_high']
    short_break = df['close'] < df['channel_low']
    n_total = int((long_break | short_break).sum())
    return {
        'breakouts_total': n_total,
        'long_breakouts': int(long_break.sum()),
        'short_breakouts': int(short_break.sum()),
        'gate_min': GATES['min_breakout_events'],
        'pass': n_total >= GATES['min_breakout_events'],
    }


def gate_B_filter(df: pd.DataFrame) -> dict:
    long_break = df['close'] > df['channel_high']
    short_break = df['close'] < df['channel_low']
    bo = long_break | short_break
    pass_body = df['body_ratio'] >= LOCKED['body_min_ratio']
    n_bo = int(bo.sum())
    n_after_body = int((bo & pass_body).sum())
    retention = n_after_body / n_bo if n_bo > 0 else 0
    return {
        'pre_filter': n_bo,
        'post_filter': n_after_body,
        'retention': retention,
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
    in_pos = False
    entry_idx = None
    entry_price = None
    direction = 0
    sl = None
    tp = None
    bars_held = 0
    last_exit_idx = -cooldown - 1

    for i in range(n - 1):
        row = df.iloc[i]
        next_row = df.iloc[i + 1]

        if in_pos:
            bars_held += 1
            high = row['high']; low = row['low']
            exit_reason = None; exit_price = None
            if direction == 1:
                if high >= tp:
                    exit_price = tp; exit_reason = 'TP'
                elif low <= sl:
                    exit_price = sl; exit_reason = 'SL'
                elif bars_held >= max_hold:
                    exit_price = row['close']; exit_reason = 'TIMEOUT'
            else:
                if low <= tp:
                    exit_price = tp; exit_reason = 'TP'
                elif high >= sl:
                    exit_price = sl; exit_reason = 'SL'
                elif bars_held >= max_hold:
                    exit_price = row['close']; exit_reason = 'TIMEOUT'

            if exit_reason:
                gross = (exit_price - entry_price) / entry_price * 100 * direction
                friction = 2 * fric * 100
                net = gross - friction
                trades.append({
                    'entry_time': df.iloc[entry_idx]['timestamp'],
                    'exit_time': row['timestamp'],
                    'direction': direction,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'gross_pct': gross,
                    'friction_pct': friction,
                    'net_pct': net,
                    'exit_reason': exit_reason,
                    'bars_held': bars_held,
                    'r_multiple': gross / (sl_mult * df.iloc[entry_idx]['atr'] / entry_price * 100) if df.iloc[entry_idx]['atr'] > 0 else 0,
                })
                in_pos = False
                last_exit_idx = i
                continue

        if not in_pos and (i - last_exit_idx) > cooldown:
            ch = row['channel_high']; cl = row['channel_low']
            br = row['body_ratio']; atr = row['atr']
            if pd.isna(ch) or pd.isna(cl) or pd.isna(br) or pd.isna(atr):
                continue
            if br < LOCKED['body_min_ratio']:
                continue
            new_dir = 0
            if row['close'] > ch:
                new_dir = 1
            elif row['close'] < cl:
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


def trades_to_daily(trades: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    df_d = df.copy(); df_d['date'] = df_d['timestamp'].dt.date
    all_dates = pd.DataFrame({'date': sorted(df_d.date.unique())})
    if trades.empty:
        return all_dates.assign(daily_net_pct=0.0, n_trades=0)
    t = trades.copy()
    t['date'] = pd.to_datetime(t['exit_time']).dt.date
    daily = t.groupby('date').agg(daily_net_pct=('net_pct', 'sum'), n_trades=('net_pct', 'count')).reset_index()
    return all_dates.merge(daily, on='date', how='left').fillna(0)


def summarize(trades: pd.DataFrame, daily: pd.DataFrame) -> dict:
    if trades.empty:
        return {'n_trades': 0, 'cum_net_pct': 0, 'avg_daily_net_pct': 0,
                'avg_trades_per_day': 0, 'win_rate': 0, 'rr_ratio': 0,
                'avg_gross_per_trade_pct': 0, 'worst_5d_pct': 0,
                'sharpe_annualized': 0, 'max_dd_pct': 0}
    n_trades = len(trades)
    n_days = len(daily)
    cum_net = float((1 + daily['daily_net_pct'] / 100).prod() - 1) * 100
    avg_daily_net = float(daily['daily_net_pct'].mean())
    avg_trades_per_day = float(daily['n_trades'].mean())
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

    exit_distribution = trades['exit_reason'].value_counts().to_dict()

    return {
        'n_trades': int(n_trades),
        'n_days': int(n_days),
        'cum_net_pct': cum_net,
        'avg_daily_net_pct': avg_daily_net,
        'avg_trades_per_day': avg_trades_per_day,
        'win_rate': wr,
        'avg_gross_per_trade_pct': avg_gross,
        'avg_win_pct': avg_win,
        'avg_loss_pct': avg_loss,
        'rr_ratio': rr,
        'max_dd_pct': max_dd,
        'worst_5d_pct': worst_5d,
        'sharpe_annualized': sharpe,
        'long_share': float((trades['direction'] == 1).mean()),
        'exit_distribution': exit_distribution,
    }


def main():
    print('=' * 100)
    print('Path B R8 — 1h Donchian Breakout (A+D+E user trade-offs accepted)')
    print('=' * 100)
    print('Pre-reg: claudedocs/path_b_r8_1h_donchian_breakout_prereg.md (3e8f00e)')
    print('DISCLOSURE: BT methodology 0/1 LIVE-parity (C1 -12.86%/14d).\n')

    df = load_data()
    df = compute_features(df)
    print(f'BTC 1h: {len(df):,} bars, {df.timestamp.min()} → {df.timestamp.max()}')
    print(f'Days: {df.timestamp.dt.date.nunique()}\n')

    print('=== Gate A — Breakout frequency ===')
    gA = gate_A_breakouts(df)
    print(f'  total breakouts: {gA["breakouts_total"]} '
          f'({gA["long_breakouts"]} long + {gA["short_breakouts"]} short)')
    print(f'  → {"PASS" if gA["pass"] else "FAIL"}\n')

    print('=== Gate B — Body filter retention ===')
    gB = gate_B_filter(df)
    print(f'  pre-filter: {gB["pre_filter"]}, post: {gB["post_filter"]}, '
          f'retention {gB["retention"]:.4f}')
    print(f'  → {"PASS" if gB["pass"] else "FAIL"}\n')

    if not gA['pass'] or not gB['pass']:
        verdict = 'INCONCLUSIVE_VACUOUS'
        print(f'EARLY EXIT: {verdict}')
        out = {'date': datetime.now(timezone.utc).isoformat(),
               'pre_reg_commit': '3e8f00e', 'verdict': verdict,
               'gate_A': gA, 'gate_B': gB, 'locked': LOCKED, 'gates': GATES}
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        p = RESULTS / f'path_b_r8_donchian_oos_{ts}.json'
        with open(p, 'w') as fp:
            json.dump(out, fp, indent=2, default=str)
        print(f'Saved: {p}')
        return

    print('=== Run full-sample strategy ===')
    trades = run_strategy(df)
    daily = trades_to_daily(trades, df)
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
        print('NO TRADES. Strategy did not enter — probably no break+body events.')
        out = {'date': datetime.now(timezone.utc).isoformat(),
               'pre_reg_commit': '3e8f00e', 'verdict': 'NO_TRADES',
               'gate_A': gA, 'gate_B': gB, 'full_sample': s,
               'locked': LOCKED, 'gates': GATES}
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        p = RESULTS / f'path_b_r8_donchian_oos_{ts}.json'
        with open(p, 'w') as fp:
            json.dump(out, fp, indent=2, default=str)
        print(f'Saved: {p}')
        return

    print('=== Test 1 — WF 5-fold ===')
    folds = []
    n = len(df)
    fold_size = n // (GATES['wf_total'] + 1)
    for i in range(GATES['wf_total']):
        ss = (i + 1) * fold_size
        ee = min(ss + fold_size, n)
        sub_df = df.iloc[ss:ee].reset_index(drop=True)
        sub_df = compute_features(sub_df)
        sub_t = run_strategy(sub_df)
        sub_d = trades_to_daily(sub_t, sub_df)
        sf = summarize(sub_t, sub_d)
        folds.append({'fold': i + 1, **sf})
        print(f'  fold {i+1}: trades={sf.get("n_trades", 0)}  '
              f'cum={sf.get("cum_net_pct", 0):+.4f}%  '
              f'daily={sf.get("avg_daily_net_pct", 0):+.4f}%  '
              f'wr={sf.get("win_rate", 0):.3f}')
    pos_count = sum(1 for f in folds if f.get('cum_net_pct', 0) > 0)
    t1_pass = pos_count >= GATES['wf_min_pos']
    print(f'  → {"PASS" if t1_pass else "FAIL"}  ({pos_count}/{GATES["wf_total"]})\n')

    print('=== Test 2 — Bootstrap 1000 × 3-day windows ===')
    nets = daily['daily_net_pct'].values
    win = GATES['bs_window_days']
    if len(nets) <= win:
        t2 = {'pass': False}
    else:
        random.seed(42)
        starts = random.sample(range(len(nets) - win), min(GATES['bs_n_iter'], len(nets) - win))
        cums = [(1 + nets[s:s+win] / 100).prod() - 1 for s in starts]
        arr = np.array(cums) * 100
        t2 = {'pos_rate': float((arr > 0).mean()),
              'mean': float(arr.mean()),
              'p5': float(np.percentile(arr, 5)),
              'p95': float(np.percentile(arr, 95)),
              'pass': float((arr > 0).mean()) >= GATES['bs_min_pos_rate']}
    print(f'  pos_rate: {t2.get("pos_rate", 0):.4f}  mean: {t2.get("mean", 0):+.4f}%')
    print(f'  → {"PASS" if t2["pass"] else "FAIL"}\n')

    print('=== Test 3 — Train/Test 60/40 ===')
    split = int(n * GATES['tt_split'])
    df_tr = compute_features(df.iloc[:split].reset_index(drop=True))
    df_te = compute_features(df.iloc[split:].reset_index(drop=True))
    tr_t = run_strategy(df_tr); te_t = run_strategy(df_te)
    tr_d = trades_to_daily(tr_t, df_tr); te_d = trades_to_daily(te_t, df_te)
    s_tr = summarize(tr_t, tr_d); s_te = summarize(te_t, te_d)
    t3_pass = s_tr['cum_net_pct'] > 0 and s_te['cum_net_pct'] > 0
    print(f'  train: trades={s_tr["n_trades"]}  cum={s_tr["cum_net_pct"]:+.4f}%  '
          f'wr={s_tr["win_rate"]:.3f}  rr={s_tr["rr_ratio"]:.2f}')
    print(f'  test:  trades={s_te["n_trades"]}  cum={s_te["cum_net_pct"]:+.4f}%  '
          f'wr={s_te["win_rate"]:.3f}  rr={s_te["rr_ratio"]:.2f}')
    print(f'  → {"PASS" if t3_pass else "FAIL"}\n')

    t4_pass = s['avg_daily_net_pct'] >= GATES['magnitude_min_daily_pct']
    t5_pass = s['win_rate'] >= GATES['wr_min']
    t6_pass = s['rr_ratio'] >= GATES['rr_min']
    t7_pass = s['avg_trades_per_day'] >= GATES['trades_per_day_min']
    t8_pass = s['avg_gross_per_trade_pct'] > GATES['per_trade_gross_min_pct']
    t9_pass = s['worst_5d_pct'] >= -GATES['tail_max_5d_dd_pct']

    print(f'T4 daily≥0.2%:    {s["avg_daily_net_pct"]:+.4f}%  → {"PASS" if t4_pass else "FAIL"} (HARD)')
    print(f'T5 WR≥30%:        {s["win_rate"]:.4f}  → {"PASS" if t5_pass else "FAIL"} (relaxed)')
    print(f'T6 R:R≥1.5:       {s["rr_ratio"]:.4f}  → {"PASS" if t6_pass else "FAIL"} (relaxed)')
    print(f'T7 ≥2 trades/d:   {s["avg_trades_per_day"]:.4f}  → {"PASS" if t7_pass else "FAIL"} (HARD)')
    print(f'T8 gross>0.07%:   {s["avg_gross_per_trade_pct"]:+.4f}%  → {"PASS" if t8_pass else "FAIL"} (HARD)')
    print(f'T9 5d≥-15%:       {s["worst_5d_pct"]:+.4f}%  → {"PASS" if t9_pass else "FAIL"} (relaxed)\n')

    hard_pass = t1_pass and t2['pass'] and t3_pass and t4_pass and t7_pass and t8_pass
    relaxed_pass = t5_pass and t6_pass and t9_pass
    all_pass = hard_pass and relaxed_pass

    print('=' * 100)
    print('FINAL VERDICT')
    print('=' * 100)
    print(f'  HARD gates (T1-4, T7, T8): {"ALL PASS" if hard_pass else "FAIL"}')
    print(f'  Relaxed gates (T5/T6/T9):  {"ALL PASS" if relaxed_pass else "PARTIAL"}')
    print(f'  Sharpe annualized: {s["sharpe_annualized"]:+.2f}')
    print(f'  Max DD: {s["max_dd_pct"]:+.4f}%\n')
    if all_pass:
        print('  ALL PASS — Round 18 candidate. LIVE-parity validation REQUIRED before deploy.')
    elif hard_pass:
        print('  HARD PASS, relaxed borderline. User review for surface decision.')
    else:
        print('  FAIL — Round 18 hardens 18-round ceiling pattern.')

    out = {
        'date': datetime.now(timezone.utc).isoformat(),
        'pre_reg_commit': '3e8f00e',
        'live_parity_prior': '0/1 (C1)',
        'verdict': 'PASS' if all_pass else ('HARD_PASS_RELAXED_PARTIAL' if hard_pass else 'FAIL'),
        'locked': LOCKED, 'gates': GATES,
        'gate_A': gA, 'gate_B': gB,
        'full_sample': s,
        'wf': {'folds': folds, 'pos_count': pos_count, 'pass': t1_pass},
        'bootstrap_3d': t2,
        'train_test': {'train': s_tr, 'test': s_te, 'pass': t3_pass},
        'tests': {
            'T4_magnitude': {'value': s['avg_daily_net_pct'], 'pass': t4_pass, 'hard': True},
            'T5_WR': {'value': s['win_rate'], 'pass': t5_pass, 'relaxed': True},
            'T6_RR': {'value': s['rr_ratio'], 'pass': t6_pass, 'relaxed': True},
            'T7_freq': {'value': s['avg_trades_per_day'], 'pass': t7_pass, 'hard': True},
            'T8_per_trade': {'value': s['avg_gross_per_trade_pct'], 'pass': t8_pass, 'hard': True},
            'T9_tail': {'value': s['worst_5d_pct'], 'pass': t9_pass, 'relaxed': True},
        },
        'hard_pass': bool(hard_pass), 'all_pass': bool(all_pass),
    }
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    p = RESULTS / f'path_b_r8_donchian_oos_{ts}.json'
    with open(p, 'w') as fp:
        json.dump(out, fp, indent=2, default=str)
    print(f'\nSaved: {p}')


if __name__ == '__main__':
    main()
