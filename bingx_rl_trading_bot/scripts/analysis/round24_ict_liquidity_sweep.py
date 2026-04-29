"""Round 24 — ICT/SMC Liquidity Sweep + Reversal (TradingView-inspired).

Pre-reg: claudedocs/round24_ict_liquidity_sweep_prereg.md (commit 51a12c1)

Mechanism:
  Detect pivot high/low (10 bars each side, 1h timeframe).
  Wait for sweep: bar's wick exceeds pivot by >= 0.05% AND closes back inside.
  Enter opposite direction at next bar open.
  SL = sweep_extreme + 0.1*ATR; TP = entry +/- 2*risk.
  Max hold 24h. Friction 0.07% RT.
"""
import json
import random
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
DATA = ROOT / 'data'
RESULTS = ROOT / 'results'
RESULTS.mkdir(exist_ok=True)

DATA_FILE = DATA / 'btc_1h_720days.csv'

LOCKED = {
    'asset': 'BTC/USDT',
    'tf': '1h',
    'pivot_lookback_bars': 10,
    'min_pivot_age_bars': 10,
    'max_pivot_age_bars': 72,
    'sweep_min_wick_pct': 0.05,
    'atr_period': 14,
    'sl_atr_buffer_mult': 0.1,
    'rr_target': 2.0,
    'max_hold_bars': 24,
    'friction_pct': 0.07,
    'capital_usd': 1500,
    'position_size_usd': 1500,
}

GATES = {
    'gate_A_min_setups': 100,
    'wf_min_pos_folds': 3,
    'wf_total_folds': 5,
    'bs_n_iter': 1000,
    'bs_window_days': 3,
    'bs_min_pos_rate': 0.50,
    'tt_split': 0.60,
    't4_min_daily_pct': 0.20,
    't7_min_trades_per_day': 2.0,
    't8_min_per_trade_gross_pct': 0.07,
    't9_max_5d_dd_pct': 15.0,
    't5_min_wr': 0.30,
}


def load_data() -> pd.DataFrame:
    df = pd.read_csv(DATA_FILE, parse_dates=['timestamp'])
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']].copy()
    df = df.sort_values('timestamp').reset_index(drop=True)
    return df


def compute_atr(df: pd.DataFrame, period: int) -> pd.Series:
    h, l, c = df['high'], df['low'], df['close']
    pc = c.shift(1)
    tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()


def find_pivots(df: pd.DataFrame, lookback: int) -> pd.DataFrame:
    """Identify confirmed pivots: bar i is pivot high if high[i] > high[i-N..i-1] AND high[i] > high[i+1..i+N]."""
    n = len(df)
    high = df['high'].values
    low = df['low'].values
    pivot_high = np.zeros(n, dtype=bool)
    pivot_low = np.zeros(n, dtype=bool)
    for i in range(lookback, n - lookback):
        h = high[i]
        l = low[i]
        if h == high[i - lookback:i + lookback + 1].max() and h > high[i - lookback:i].max() and h > high[i + 1:i + lookback + 1].max():
            pivot_high[i] = True
        if l == low[i - lookback:i + lookback + 1].min() and l < low[i - lookback:i].min() and l < low[i + 1:i + lookback + 1].min():
            pivot_low[i] = True
    out = df.copy()
    out['pivot_high'] = pivot_high
    out['pivot_low'] = pivot_low
    return out


def detect_sweeps_and_trade(df: pd.DataFrame) -> pd.DataFrame:
    """Walk forward, detect sweeps of active pivots, simulate trades."""
    df = df.copy()
    df['atr'] = compute_atr(df, LOCKED['atr_period'])

    n = len(df)
    high = df['high'].values
    low = df['low'].values
    open_ = df['open'].values
    close = df['close'].values
    pivot_high = df['pivot_high'].values
    pivot_low = df['pivot_low'].values
    atr = df['atr'].values
    ts = df['timestamp'].values

    # Active pivots: list of (idx, level, type), validated when i = pivot_idx + lookback
    active_pivots: list = []  # each: {'idx', 'level', 'type', 'expiry'}
    confirm_offset = LOCKED['pivot_lookback_bars']
    max_age = LOCKED['max_pivot_age_bars']
    sweep_min_pct = LOCKED['sweep_min_wick_pct']
    sl_buffer = LOCKED['sl_atr_buffer_mult']
    rr = LOCKED['rr_target']
    max_hold = LOCKED['max_hold_bars']
    fric = LOCKED['friction_pct']

    trades = []
    in_pos = False
    pos = None  # dict

    start = LOCKED['atr_period'] + LOCKED['pivot_lookback_bars'] * 2

    for i in range(start, n):
        # Confirm pivot from confirm_offset bars ago
        confirm_idx = i - confirm_offset
        if confirm_idx >= 0 and pivot_high[confirm_idx]:
            active_pivots.append({'idx': confirm_idx, 'level': high[confirm_idx],
                                  'type': 'high', 'expiry': confirm_idx + max_age})
        if confirm_idx >= 0 and pivot_low[confirm_idx]:
            active_pivots.append({'idx': confirm_idx, 'level': low[confirm_idx],
                                  'type': 'low', 'expiry': confirm_idx + max_age})

        # Drop expired
        active_pivots = [p for p in active_pivots if p['expiry'] >= i]

        # Manage open position first (intrabar SL/TP)
        if in_pos:
            # Check SL/TP touched intrabar
            if pos['side'] == 'LONG':
                if low[i] <= pos['sl']:
                    exit_price = pos['sl']
                    exit_reason = 'SL'
                elif high[i] >= pos['tp']:
                    exit_price = pos['tp']
                    exit_reason = 'TP'
                elif (i - pos['entry_idx']) >= max_hold:
                    exit_price = close[i]
                    exit_reason = 'TIMEOUT'
                else:
                    exit_price = None
                    exit_reason = None
            else:  # SHORT
                if high[i] >= pos['sl']:
                    exit_price = pos['sl']
                    exit_reason = 'SL'
                elif low[i] <= pos['tp']:
                    exit_price = pos['tp']
                    exit_reason = 'TP'
                elif (i - pos['entry_idx']) >= max_hold:
                    exit_price = close[i]
                    exit_reason = 'TIMEOUT'
                else:
                    exit_price = None
                    exit_reason = None

            if exit_price is not None:
                gross_pct = (exit_price - pos['entry_price']) / pos['entry_price'] * 100
                if pos['side'] == 'SHORT':
                    gross_pct = -gross_pct
                net_pct = gross_pct - fric
                trades.append({
                    'entry_ts': pos['entry_ts'],
                    'exit_ts': ts[i],
                    'side': pos['side'],
                    'entry_price': pos['entry_price'],
                    'exit_price': exit_price,
                    'gross_pct': gross_pct,
                    'net_pct': net_pct,
                    'exit_reason': exit_reason,
                    'hold_bars': i - pos['entry_idx'],
                    'pivot_idx': pos['pivot_idx'],
                })
                in_pos = False
                pos = None

        # Look for new sweep (only if not in position)
        if not in_pos and atr[i] > 0 and not np.isnan(atr[i]):
            for p in active_pivots[:]:
                if p['type'] == 'high':
                    sweep_threshold = p['level'] * (1 + sweep_min_pct / 100)
                    if high[i] > sweep_threshold and close[i] < p['level']:
                        # BEARISH sweep — signal SHORT at next bar
                        if i + 1 < n:
                            entry_price = open_[i + 1]
                            sl = high[i] + sl_buffer * atr[i]
                            risk = sl - entry_price
                            tp = entry_price - rr * risk
                            if risk > 0 and tp > 0:
                                pos = {'side': 'SHORT', 'entry_idx': i + 1,
                                       'entry_ts': ts[i + 1], 'entry_price': entry_price,
                                       'sl': sl, 'tp': tp, 'pivot_idx': p['idx']}
                                in_pos = True
                                active_pivots.remove(p)
                                break
                else:
                    sweep_threshold = p['level'] * (1 - sweep_min_pct / 100)
                    if low[i] < sweep_threshold and close[i] > p['level']:
                        # BULLISH sweep — signal LONG at next bar
                        if i + 1 < n:
                            entry_price = open_[i + 1]
                            sl = low[i] - sl_buffer * atr[i]
                            risk = entry_price - sl
                            tp = entry_price + rr * risk
                            if risk > 0:
                                pos = {'side': 'LONG', 'entry_idx': i + 1,
                                       'entry_ts': ts[i + 1], 'entry_price': entry_price,
                                       'sl': sl, 'tp': tp, 'pivot_idx': p['idx']}
                                in_pos = True
                                active_pivots.remove(p)
                                break

    return pd.DataFrame(trades)


def summarize(trades: pd.DataFrame, n_days: float) -> dict:
    if trades.empty:
        return {'n_trades': 0, 'cum_net_pct': 0, 'pass_t7': False, 'pass_t8': False}
    cum_net = float((1 + trades['net_pct'] / 100).prod() - 1) * 100
    cum_gross = float(trades['gross_pct'].sum())
    n = len(trades)
    avg_gross_per_trade = float(trades['gross_pct'].mean())
    avg_net_per_trade = float(trades['net_pct'].mean())
    wr = float((trades['net_pct'] > 0).mean())
    avg_win = float(trades.loc[trades['net_pct'] > 0, 'net_pct'].mean()) if (trades['net_pct'] > 0).any() else 0
    avg_loss = float(trades.loc[trades['net_pct'] < 0, 'net_pct'].mean()) if (trades['net_pct'] < 0).any() else 0
    rr_realized = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
    daily_pct = float(trades['net_pct'].sum() / n_days)
    trades_per_day = n / n_days

    # Daily aggregate for tail
    trades_copy = trades.copy()
    trades_copy['exit_date'] = pd.to_datetime(trades_copy['exit_ts']).dt.floor('D')
    daily_returns = trades_copy.groupby('exit_date')['net_pct'].sum()
    if len(daily_returns) >= 5:
        rolling_5d = daily_returns.rolling(5).sum()
        worst_5d = float(rolling_5d.min())
    else:
        worst_5d = float(daily_returns.min())

    return {
        'n_trades': n,
        'cum_net_pct': cum_net,
        'cum_gross_pct': cum_gross,
        'avg_gross_per_trade_pct': avg_gross_per_trade,
        'avg_net_per_trade_pct': avg_net_per_trade,
        'wr': wr,
        'avg_win_pct': avg_win,
        'avg_loss_pct': avg_loss,
        'rr_realized': rr_realized,
        'daily_pct': daily_pct,
        'trades_per_day': trades_per_day,
        'worst_5d_pct': worst_5d,
        'n_days': n_days,
        'pass_t7': trades_per_day >= GATES['t7_min_trades_per_day'],
        'pass_t8': avg_gross_per_trade >= GATES['t8_min_per_trade_gross_pct'],
    }


def gate_B_random_baseline(df: pd.DataFrame, actual_trades: pd.DataFrame,
                            n_iter: int = 1000) -> dict:
    """Randomly entry on bars matching the same trade count + side ratio."""
    if actual_trades.empty:
        return {'pass': False, 'reason': 'no actual trades'}
    rng = np.random.default_rng(42)
    n = len(df)
    actual_count = len(actual_trades)
    long_ratio = (actual_trades['side'] == 'LONG').mean()
    actual_cum = float((1 + actual_trades['net_pct'] / 100).prod() - 1) * 100

    cum_arr = []
    for _ in range(n_iter):
        starts = rng.integers(50, n - LOCKED['max_hold_bars'] - 1, size=actual_count)
        sides = np.where(rng.random(actual_count) < long_ratio, 'LONG', 'SHORT')
        sim_nets = []
        atr = compute_atr(df, LOCKED['atr_period']).values
        opens = df['open'].values
        highs = df['high'].values
        lows = df['low'].values
        closes = df['close'].values
        for s, side in zip(starts, sides):
            if np.isnan(atr[s]):
                continue
            entry = opens[s + 1] if s + 1 < n else opens[s]
            risk_distance = LOCKED['sl_atr_buffer_mult'] * atr[s]
            if side == 'LONG':
                sl = entry - risk_distance
                tp = entry + LOCKED['rr_target'] * risk_distance
            else:
                sl = entry + risk_distance
                tp = entry - LOCKED['rr_target'] * risk_distance
            # Simulate hold up to max_hold
            exit_price = None
            for j in range(1, LOCKED['max_hold_bars'] + 1):
                if s + j >= n:
                    break
                if side == 'LONG':
                    if lows[s + j] <= sl:
                        exit_price = sl
                        break
                    if highs[s + j] >= tp:
                        exit_price = tp
                        break
                else:
                    if highs[s + j] >= sl:
                        exit_price = sl
                        break
                    if lows[s + j] <= tp:
                        exit_price = tp
                        break
            if exit_price is None:
                exit_price = closes[min(s + LOCKED['max_hold_bars'], n - 1)]
            gross = (exit_price - entry) / entry * 100
            if side == 'SHORT':
                gross = -gross
            sim_nets.append(gross - LOCKED['friction_pct'])
        if sim_nets:
            cum = float((1 + np.array(sim_nets) / 100).prod() - 1) * 100
            cum_arr.append(cum)
    arr = np.array(cum_arr)
    p95 = float(np.percentile(arr, 95))
    return {
        'actual_cum_pct': actual_cum,
        'random_p95_pct': p95,
        'random_mean_pct': float(arr.mean()),
        'pass': actual_cum > p95,
    }


def test_1_walk_forward(df: pd.DataFrame, n_days: float) -> dict:
    folds = GATES['wf_total_folds']
    n = len(df)
    fold_size = n // (folds + 1)
    results = []
    for i in range(folds):
        s = (i + 1) * fold_size
        e = min(s + fold_size, n)
        sub = df.iloc[s:e].reset_index(drop=True)
        sub_p = find_pivots(sub, LOCKED['pivot_lookback_bars'])
        trades = detect_sweeps_and_trade(sub_p)
        sub_days = (sub['timestamp'].max() - sub['timestamp'].min()).total_seconds() / 86400
        summ = summarize(trades, sub_days)
        results.append({'fold': i + 1, **summ})
    pos_count = sum(1 for r in results if r.get('cum_net_pct', 0) > 0)
    return {'folds': results, 'pos_count': pos_count,
            'pass': pos_count >= GATES['wf_min_pos_folds']}


def test_2_bootstrap(trades: pd.DataFrame, n_days: float) -> dict:
    if trades.empty or n_days < 7:
        return {'pass': False, 'reason': 'insufficient'}
    trades_copy = trades.copy()
    trades_copy['exit_date'] = pd.to_datetime(trades_copy['exit_ts']).dt.floor('D')
    daily_returns = trades_copy.groupby('exit_date')['net_pct'].sum()
    daily_returns = daily_returns.reindex(
        pd.date_range(daily_returns.index.min(), daily_returns.index.max(), freq='D'),
        fill_value=0
    )
    nets = daily_returns.values
    n = len(nets)
    win = GATES['bs_window_days']
    if n <= win:
        return {'pass': False, 'reason': 'panel too short'}
    random.seed(42)
    starts = random.sample(range(n - win), min(GATES['bs_n_iter'], n - win))
    cums = [(1 + nets[s:s + win] / 100).prod() - 1 for s in starts]
    arr = np.array(cums) * 100
    pos_rate = float((arr > 0).mean())
    return {'n_iter': len(arr), 'mean_cum_pct': float(arr.mean()),
            'pos_rate': pos_rate, 'p5': float(np.percentile(arr, 5)),
            'p95': float(np.percentile(arr, 95)),
            'pass': pos_rate >= GATES['bs_min_pos_rate']}


def test_3_train_test(df: pd.DataFrame) -> dict:
    n = len(df)
    split = int(n * GATES['tt_split'])
    df_tr = df.iloc[:split].reset_index(drop=True)
    df_te = df.iloc[split:].reset_index(drop=True)
    tr_days = (df_tr['timestamp'].max() - df_tr['timestamp'].min()).total_seconds() / 86400
    te_days = (df_te['timestamp'].max() - df_te['timestamp'].min()).total_seconds() / 86400
    tr_p = find_pivots(df_tr, LOCKED['pivot_lookback_bars'])
    te_p = find_pivots(df_te, LOCKED['pivot_lookback_bars'])
    tr_trades = detect_sweeps_and_trade(tr_p)
    te_trades = detect_sweeps_and_trade(te_p)
    s_tr = summarize(tr_trades, tr_days)
    s_te = summarize(te_trades, te_days)
    return {'train': s_tr, 'test': s_te,
            'pass': (s_tr.get('cum_net_pct', 0) > 0) and (s_te.get('cum_net_pct', 0) > 0)}


def main():
    print('=' * 100)
    print('Round 24 — ICT/SMC Liquidity Sweep + Reversal (TradingView-inspired)')
    print('=' * 100)
    print('Pre-reg: claudedocs/round24_ict_liquidity_sweep_prereg.md (51a12c1)')
    print(f'Locked: {LOCKED}\n')

    df = load_data()
    n_days = (df['timestamp'].max() - df['timestamp'].min()).total_seconds() / 86400
    print(f'Data: {len(df):,} bars, {n_days:.1f} days, '
          f'{df["timestamp"].min()} → {df["timestamp"].max()}\n')

    print('Detecting pivots...')
    df_p = find_pivots(df, LOCKED['pivot_lookback_bars'])
    n_ph = int(df_p['pivot_high'].sum())
    n_pl = int(df_p['pivot_low'].sum())
    print(f'  pivot highs: {n_ph}, pivot lows: {n_pl}, total: {n_ph + n_pl}\n')

    print('Simulating sweeps + trades...')
    trades = detect_sweeps_and_trade(df_p)
    print(f'  trades executed: {len(trades)}\n')

    print('=== Gate A — Sufficient setups ===')
    gA = {'n_setups': len(trades), 'gate': GATES['gate_A_min_setups'],
          'pass': len(trades) >= GATES['gate_A_min_setups']}
    print(f'  setups: {gA["n_setups"]}, gate: ≥ {gA["gate"]} → '
          f'{"PASS" if gA["pass"] else "FAIL"}\n')

    if not gA['pass']:
        print('EARLY EXIT — vacuous sample.')
        out = {'date': datetime.now(timezone.utc).isoformat(),
               'pre_reg_commit': '51a12c1', 'verdict': 'INCONCLUSIVE_VACUOUS',
               'gate_A': gA, 'locked': LOCKED}
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        p = RESULTS / f'round24_ict_sweep_{ts}.json'
        with open(p, 'w') as fp:
            json.dump(out, fp, indent=2, default=str)
        return

    print('=== Full-sample summary ===')
    summ = summarize(trades, n_days)
    for k, v in summ.items():
        if isinstance(v, float):
            print(f'  {k}: {v:+.4f}')
        else:
            print(f'  {k}: {v}')
    print()

    print('=== Gate B — Random-baseline (1000 sims) ===')
    gB = gate_B_random_baseline(df_p, trades, n_iter=1000)
    print(f'  actual cum: {gB.get("actual_cum_pct", 0):+.4f}%')
    print(f'  random p95: {gB.get("random_p95_pct", 0):+.4f}%')
    print(f'  → {"PASS" if gB.get("pass") else "FAIL"}\n')

    print('=== T1 — WF 5-fold ===')
    t1 = test_1_walk_forward(df, n_days)
    for f_ in t1['folds']:
        print(f'  fold {f_["fold"]}: cum_net={f_.get("cum_net_pct", 0):+.4f}%  '
              f'trades={f_.get("n_trades", 0)} '
              f'daily={f_.get("daily_pct", 0):+.4f}%')
    print(f'  → {"PASS" if t1["pass"] else "FAIL"}  ({t1["pos_count"]}/5)\n')

    print('=== T2 — Bootstrap 1000 × 3-day ===')
    t2 = test_2_bootstrap(trades, n_days)
    print(f'  pos_rate: {t2.get("pos_rate", 0):.4f}, mean: {t2.get("mean_cum_pct", 0):+.4f}%')
    print(f'  → {"PASS" if t2.get("pass") else "FAIL"}\n')

    print('=== T3 — Train/Test 60/40 ===')
    t3 = test_3_train_test(df)
    print(f'  train cum: {t3["train"].get("cum_net_pct", 0):+.4f}%, '
          f'test cum: {t3["test"].get("cum_net_pct", 0):+.4f}%')
    print(f'  → {"PASS" if t3["pass"] else "FAIL"}\n')

    daily = summ.get('daily_pct', 0)
    t4_pass = daily >= GATES['t4_min_daily_pct']
    print(f'=== T4 (HARD) Daily ≥ {GATES["t4_min_daily_pct"]}% ===')
    print(f'  daily: {daily:+.4f}%  → {"PASS" if t4_pass else "FAIL"}\n')

    wr = summ.get('wr', 0)
    t5_pass = wr >= GATES['t5_min_wr']
    print(f'=== T5 WR ≥ {GATES["t5_min_wr"]} ===')
    print(f'  WR: {wr:.4f}  → {"PASS" if t5_pass else "FAIL"}\n')

    rr = summ.get('rr_realized', 0)
    t6_pass = rr >= 1.0
    print(f'=== T6 R:R ≥ 1.0 (realized) ===')
    print(f'  R:R: {rr:.4f}  → {"PASS" if t6_pass else "FAIL"}\n')

    tpd = summ.get('trades_per_day', 0)
    t7_pass = tpd >= GATES['t7_min_trades_per_day']
    print(f'=== T7 (HARD) Trades/day ≥ {GATES["t7_min_trades_per_day"]} ===')
    print(f'  trades/day: {tpd:.4f}  → {"PASS" if t7_pass else "FAIL"}\n')

    pt_gross = summ.get('avg_gross_per_trade_pct', 0)
    t8_pass = pt_gross >= GATES['t8_min_per_trade_gross_pct']
    print(f'=== T8 (HARD) Per-trade gross ≥ {GATES["t8_min_per_trade_gross_pct"]}% ===')
    print(f'  per-trade gross: {pt_gross:+.4f}%  → {"PASS" if t8_pass else "FAIL"}\n')

    worst5 = summ.get('worst_5d_pct', 0)
    t9_pass = worst5 >= -GATES['t9_max_5d_dd_pct']
    print(f'=== T9 Worst 5d ≥ -{GATES["t9_max_5d_dd_pct"]}% ===')
    print(f'  worst 5d: {worst5:+.4f}%  → {"PASS" if t9_pass else "FAIL"}\n')

    hard = [t4_pass, t7_pass, t8_pass]
    all_g = [gA['pass'], gB.get('pass', False), t1['pass'], t2.get('pass', False),
             t3['pass'], t4_pass, t5_pass, t6_pass, t7_pass, t8_pass, t9_pass]
    n_hard = sum(hard)
    n_all = sum(all_g)

    print('=' * 100)
    print(f'VERDICT: {n_all}/11 PASS  |  HARD: {n_hard}/3 (T4 daily, T7 freq, T8 gross)')
    print('=' * 100)

    out = {
        'date': datetime.now(timezone.utc).isoformat(),
        'pre_reg_commit': '51a12c1',
        'locked': LOCKED, 'gates': GATES,
        'full_summary': summ,
        'gate_A': gA, 'gate_B': gB,
        't1_wf': t1, 't2_bs': t2, 't3_tt': t3,
        't4': {'daily_pct': daily, 'pass': bool(t4_pass)},
        't5': {'wr': wr, 'pass': bool(t5_pass)},
        't6': {'rr': rr, 'pass': bool(t6_pass)},
        't7': {'trades_per_day': tpd, 'pass': bool(t7_pass)},
        't8': {'per_trade_gross_pct': pt_gross, 'pass': bool(t8_pass)},
        't9': {'worst_5d_pct': worst5, 'pass': bool(t9_pass)},
        'verdict_total_pass': n_all,
        'verdict_hard_pass': n_hard,
    }
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    p = RESULTS / f'round24_ict_sweep_{ts}.json'
    with open(p, 'w') as fp:
        json.dump(out, fp, indent=2, default=str)
    print(f'Saved: {p}')


if __name__ == '__main__':
    main()
