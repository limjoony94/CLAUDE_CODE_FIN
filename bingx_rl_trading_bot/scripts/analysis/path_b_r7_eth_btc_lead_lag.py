"""Path B R7 — ETH→BTC Lead-Lag Scalping LOCKED OOS.

Pre-reg: claudedocs/path_b_r7_eth_btc_lead_lag_prereg.md (commit 654a015)

DISCLOSURE: Methodology that found C1 produced C1's LIVE -12.86%/14d
failure (postmortem 20260427). BT result here is research artifact,
NOT deploy candidate.

Mechanism:
  At 5m bar t, signal = ETH 15min log return (3-bar lookback).
  If |signal| ≥ 0.30% AND BTC 1h trend filter aligned → enter at t+1.
  TP: 1.5×ATR. SL: 1.0×ATR. Max hold: 6 bars (30 min).

Tests: 9 gates per pre-reg. Locked, no retuning.
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
RESULTS.mkdir(exist_ok=True)

BTC_5M_FILE = DATA / 'btc_5m_720days_binance.csv'
ETH_5M_FILE = DATA / 'eth_binance_5m.csv'

LOCKED = {
    'asset_trade': 'BTC/USDT',
    'asset_signal': 'ETH/USDT',
    'signal_lookback_bars': 3,            # 15 min
    'signal_threshold_pct': 0.30,         # |ETH 15m return|
    'sma_short_periods': 240,
    'sma_long_periods': 600,
    'atr_period': 14,
    'tp_atr_mult': 1.5,
    'sl_atr_mult': 1.0,
    'max_hold_bars': 6,
    'friction_pct': 0.07,
    'capital_usd': 1500,
}

GATES = {
    'corr_min_lag1to6': 0.05,
    'min_signal_events': 2000,
    'random_baseline_pct': 0.95,
    'wf_min_pos': 3,
    'wf_total': 5,
    'bs_n_iter': 1000,
    'bs_window_days': 3,
    'bs_min_pos_rate': 0.50,
    'tt_split': 0.60,
    'magnitude_min_daily_pct': 0.20,      # HARD goal
    'wr_min': 0.40,
    'rr_min': 1.0,
    'trades_per_day_min': 2.0,
    'per_trade_gross_min_pct': 0.07,      # > taker round-trip
    'tail_max_5d_dd_pct': 10.0,
}


def load_data() -> pd.DataFrame:
    btc = pd.read_csv(BTC_5M_FILE, usecols=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    btc['timestamp'] = pd.to_datetime(btc['timestamp'])
    btc = btc.sort_values('timestamp').reset_index(drop=True)

    eth = pd.read_csv(ETH_5M_FILE, usecols=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    eth['timestamp'] = pd.to_datetime(eth['timestamp'])
    eth = eth.sort_values('timestamp').reset_index(drop=True)

    # Inner-join on timestamp
    btc = btc.rename(columns={c: 'btc_' + c for c in ['open', 'high', 'low', 'close', 'volume']})
    eth = eth.rename(columns={c: 'eth_' + c for c in ['open', 'high', 'low', 'close', 'volume']})
    df = btc.merge(eth, on='timestamp', how='inner').reset_index(drop=True)
    return df


def compute_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df['eth_15m_ret'] = np.log(df['eth_close'] / df['eth_close'].shift(LOCKED['signal_lookback_bars'])) * 100
    df['btc_sma_short'] = df['btc_close'].rolling(LOCKED['sma_short_periods']).mean()
    df['btc_sma_long'] = df['btc_close'].rolling(LOCKED['sma_long_periods']).mean()
    df['trend_up'] = df['btc_sma_short'] > df['btc_sma_long']
    # ATR using TR = max(high-low, |high-prev_close|, |low-prev_close|)
    prev_close = df['btc_close'].shift(1)
    tr = pd.concat([
        df['btc_high'] - df['btc_low'],
        (df['btc_high'] - prev_close).abs(),
        (df['btc_low'] - prev_close).abs(),
    ], axis=1).max(axis=1)
    df['atr'] = tr.rolling(LOCKED['atr_period']).mean()
    return df


def gate_A_lead_lag(df: pd.DataFrame) -> dict:
    """Lagged correlation Corr(ETH_ret_t, BTC_ret_{t+lag}) for lag 1..6."""
    eth_ret = df['eth_close'].pct_change()
    btc_ret = df['btc_close'].pct_change()
    corrs = {}
    for lag in range(1, 7):
        c = eth_ret.shift(lag).corr(btc_ret)
        corrs[f'lag_{lag}'] = float(c) if not np.isnan(c) else 0
    max_corr = max(corrs.values())
    return {
        'corrs': corrs,
        'max_corr': max_corr,
        'gate_min': GATES['corr_min_lag1to6'],
        'pass': max_corr >= GATES['corr_min_lag1to6'],
    }


def gate_B_events(df: pd.DataFrame) -> dict:
    valid = df['eth_15m_ret'].notna() & df['trend_up'].notna() & df['atr'].notna()
    cands = df[valid & (df['eth_15m_ret'].abs() >= LOCKED['signal_threshold_pct'])]
    return {
        'n_candidates': int(len(cands)),
        'gate_min': GATES['min_signal_events'],
        'pass': len(cands) >= GATES['min_signal_events'],
    }


def run_strategy(df: pd.DataFrame) -> pd.DataFrame:
    """Iterate bars, generate trades, compute P&L."""
    df = df.reset_index(drop=True)
    fric = LOCKED['friction_pct'] / 100.0
    th = LOCKED['signal_threshold_pct']
    max_hold = LOCKED['max_hold_bars']
    tp_mult = LOCKED['tp_atr_mult']
    sl_mult = LOCKED['sl_atr_mult']

    trades = []
    n = len(df)
    in_pos = False
    entry_idx = None
    entry_price = None
    direction = 0
    tp_price = None
    sl_price = None
    bars_held = 0

    for i in range(n - 1):
        row = df.iloc[i]
        next_row = df.iloc[i + 1]

        if in_pos:
            bars_held += 1
            high = row['btc_high']
            low = row['btc_low']
            exit_reason = None
            exit_price = None
            if direction == 1:
                if high >= tp_price:
                    exit_price = tp_price; exit_reason = 'TP'
                elif low <= sl_price:
                    exit_price = sl_price; exit_reason = 'SL'
                elif bars_held >= max_hold:
                    exit_price = row['btc_close']; exit_reason = 'TIMEOUT'
            else:
                if low <= tp_price:
                    exit_price = tp_price; exit_reason = 'TP'
                elif high >= sl_price:
                    exit_price = sl_price; exit_reason = 'SL'
                elif bars_held >= max_hold:
                    exit_price = row['btc_close']; exit_reason = 'TIMEOUT'

            if exit_reason:
                gross = (exit_price - entry_price) / entry_price * 100 * direction
                friction_pct = 2 * fric * 100
                net = gross - friction_pct
                trades.append({
                    'entry_time': df.iloc[entry_idx]['timestamp'],
                    'exit_time': row['timestamp'],
                    'direction': direction,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'gross_pct': gross,
                    'friction_pct': friction_pct,
                    'net_pct': net,
                    'exit_reason': exit_reason,
                    'bars_held': bars_held,
                })
                in_pos = False
                entry_idx = None
                continue

        if not in_pos:
            sig = row['eth_15m_ret']
            tu = row['trend_up']
            atr = row['atr']
            if pd.isna(sig) or pd.isna(tu) or pd.isna(atr):
                continue
            if abs(sig) < th:
                continue
            if sig > 0 and tu:
                direction = 1
            elif sig < 0 and not tu:
                direction = -1
            else:
                continue
            entry_price = next_row['btc_open']
            entry_idx = i + 1
            tp_price = entry_price + direction * tp_mult * atr
            sl_price = entry_price - direction * sl_mult * atr
            in_pos = True
            bars_held = 0

    return pd.DataFrame(trades)


def trades_to_daily(trades: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate trade net_pct into daily series."""
    if trades.empty:
        df_d = df.copy()
        df_d['date'] = df_d['timestamp'].dt.date
        return pd.DataFrame({'date': sorted(df_d.date.unique()), 'daily_net_pct': 0.0,
                             'n_trades': 0})
    trades = trades.copy()
    trades['date'] = pd.to_datetime(trades['exit_time']).dt.date
    daily = trades.groupby('date').agg(
        daily_net_pct=('net_pct', 'sum'),
        n_trades=('net_pct', 'count'),
    ).reset_index()

    df_d = df.copy()
    df_d['date'] = df_d['timestamp'].dt.date
    all_dates = pd.DataFrame({'date': sorted(df_d.date.unique())})
    daily = all_dates.merge(daily, on='date', how='left').fillna(0)
    return daily


def summarize(trades: pd.DataFrame, daily: pd.DataFrame) -> dict:
    if trades.empty:
        return {'n_trades': 0}
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
    }


def main():
    print('=' * 100)
    print('Path B R7 — ETH→BTC Lead-Lag Scalping')
    print('=' * 100)
    print('Pre-reg: claudedocs/path_b_r7_eth_btc_lead_lag_prereg.md (654a015)')
    print('DISCLOSURE: Methodology has 0/1 LIVE-parity track record (C1 failed).')
    print(f'Locked: {LOCKED}\n')

    df = load_data()
    print(f'Joined panel: {len(df):,} 5m bars, {df.timestamp.min()} → {df.timestamp.max()}')
    print(f'Days covered: {df.timestamp.dt.date.nunique()}\n')

    df = compute_features(df)

    print('=== Gate A — Lead-lag correlation ===')
    gA = gate_A_lead_lag(df)
    for lag, c in gA['corrs'].items():
        print(f'  Corr(ETH_ret_t, BTC_ret_t+{lag.split("_")[1]}): {c:+.5f}')
    print(f'  max corr: {gA["max_corr"]:+.5f}  gate ≥ {gA["gate_min"]}')
    print(f'  → {"PASS" if gA["pass"] else "FAIL"}\n')

    print('=== Gate B — Sufficient signal events ===')
    gB = gate_B_events(df)
    print(f'  candidate events: {gB["n_candidates"]}  gate ≥ {gB["gate_min"]}')
    print(f'  → {"PASS" if gB["pass"] else "FAIL"}\n')

    if not gA['pass'] or not gB['pass']:
        verdict = 'INCONCLUSIVE_VACUOUS'
        print(f'EARLY EXIT: {verdict}')
        out = {'date': datetime.now(timezone.utc).isoformat(),
               'pre_reg_commit': '654a015', 'verdict': verdict,
               'gate_A': gA, 'gate_B': gB, 'locked': LOCKED, 'gates': GATES}
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        p = RESULTS / f'path_b_r7_leadlag_oos_{ts}.json'
        with open(p, 'w') as fp:
            json.dump(out, fp, indent=2, default=str)
        print(f'Saved: {p}')
        return

    print('=== Run full-sample strategy ===')
    trades = run_strategy(df)
    daily = trades_to_daily(trades, df)
    s = summarize(trades, daily)
    for k, v in s.items():
        if isinstance(v, float):
            print(f'  {k}: {v:+.4f}')
        else:
            print(f'  {k}: {v}')
    print()

    if s['n_trades'] == 0:
        print('No trades produced. INCONCLUSIVE.')
        return

    # Test 1 — WF
    print('=== Test 1 — WF 5-fold ===')
    folds = []
    n = len(df)
    fold_size = n // (GATES['wf_total'] + 1)
    for i in range(GATES['wf_total']):
        ss = (i + 1) * fold_size
        ee = min(ss + fold_size, n)
        sub = df.iloc[ss:ee].copy()
        sub_trades = run_strategy(sub)
        sub_daily = trades_to_daily(sub_trades, sub)
        sf = summarize(sub_trades, sub_daily) if not sub_trades.empty else {'cum_net_pct': 0}
        folds.append({'fold': i + 1, **sf})
        cn = sf.get('cum_net_pct', 0)
        nt = sf.get('n_trades', 0)
        dn = sf.get('avg_daily_net_pct', 0)
        print(f'  fold {i+1}: trades={nt}  cum={cn:+.4f}%  daily={dn:+.4f}%')
    pos_count = sum(1 for f in folds if f.get('cum_net_pct', 0) > 0)
    t1_pass = pos_count >= GATES['wf_min_pos']
    print(f'  → {"PASS" if t1_pass else "FAIL"}  ({pos_count}/{GATES["wf_total"]})\n')

    # Test 2 — Bootstrap 1000 × 3-day windows (USER REQUIREMENT)
    print('=== Test 2 — Bootstrap 1000 × 3-day windows (USER) ===')
    nets = daily['daily_net_pct'].values
    win = GATES['bs_window_days']
    if len(nets) <= win:
        t2 = {'pass': False, 'reason': 'short'}
    else:
        random.seed(42)
        starts = random.sample(range(len(nets) - win), min(GATES['bs_n_iter'], len(nets) - win))
        cums = [(1 + nets[s:s+win] / 100).prod() - 1 for s in starts]
        arr = np.array(cums) * 100
        pos_rate = float((arr > 0).mean())
        t2 = {'n': len(arr), 'mean': float(arr.mean()), 'pos_rate': pos_rate,
              'p5': float(np.percentile(arr, 5)), 'p95': float(np.percentile(arr, 95)),
              'pass': pos_rate >= GATES['bs_min_pos_rate']}
    print(f'  3d window pos_rate: {t2.get("pos_rate", 0):.4f}  '
          f'mean: {t2.get("mean", 0):+.4f}%')
    print(f'  → {"PASS" if t2["pass"] else "FAIL"}\n')

    # Test 3 — Train/Test 60/40
    print('=== Test 3 — Train/Test 60/40 ===')
    split = int(n * GATES['tt_split'])
    df_tr = df.iloc[:split].copy()
    df_te = df.iloc[split:].copy()
    tr_trades = run_strategy(df_tr)
    te_trades = run_strategy(df_te)
    tr_daily = trades_to_daily(tr_trades, df_tr)
    te_daily = trades_to_daily(te_trades, df_te)
    s_tr = summarize(tr_trades, tr_daily) if not tr_trades.empty else {'cum_net_pct': 0}
    s_te = summarize(te_trades, te_daily) if not te_trades.empty else {'cum_net_pct': 0}
    t3_pass = s_tr.get('cum_net_pct', 0) > 0 and s_te.get('cum_net_pct', 0) > 0
    print(f'  train: trades={s_tr.get("n_trades", 0)}  cum={s_tr.get("cum_net_pct", 0):+.4f}%')
    print(f'  test:  trades={s_te.get("n_trades", 0)}  cum={s_te.get("cum_net_pct", 0):+.4f}%')
    print(f'  → {"PASS" if t3_pass else "FAIL"}\n')

    # T4: daily ≥ 0.2%
    t4_pass = s['avg_daily_net_pct'] >= GATES['magnitude_min_daily_pct']
    print(f'T4 daily≥0.2%: {s["avg_daily_net_pct"]:+.4f}%  → {"PASS" if t4_pass else "FAIL"}')

    # T5: WR ≥ 40%
    t5_pass = s['win_rate'] >= GATES['wr_min']
    print(f'T5 WR≥40%:     {s["win_rate"]:.4f}  → {"PASS" if t5_pass else "FAIL"}')

    # T6: R:R ≥ 1
    t6_pass = s['rr_ratio'] >= GATES['rr_min']
    print(f'T6 R:R≥1:      {s["rr_ratio"]:.4f}  → {"PASS" if t6_pass else "FAIL"}')

    # T7: trades/day ≥ 2
    t7_pass = s['avg_trades_per_day'] >= GATES['trades_per_day_min']
    print(f'T7 ≥2 trades/d:{s["avg_trades_per_day"]:.4f}  → {"PASS" if t7_pass else "FAIL"}')

    # T8: per-trade gross > taker RT
    t8_pass = s['avg_gross_per_trade_pct'] > GATES['per_trade_gross_min_pct']
    print(f'T8 gross>0.07%:{s["avg_gross_per_trade_pct"]:+.4f}%  → {"PASS" if t8_pass else "FAIL"}')

    # T9: tail
    t9_pass = s['worst_5d_pct'] >= -GATES['tail_max_5d_dd_pct']
    print(f'T9 5d≥-10%:    {s["worst_5d_pct"]:+.4f}%  → {"PASS" if t9_pass else "FAIL"}\n')

    all_hard_pass = t1_pass and t2['pass'] and t3_pass and t4_pass and t7_pass and t8_pass and t9_pass

    print('=' * 100)
    print('FINAL VERDICT')
    print('=' * 100)
    print(f'  Gate A lead-lag:    PASS  max_corr={gA["max_corr"]:+.4f}')
    print(f'  Gate B events:      PASS  n={gB["n_candidates"]}')
    print(f'  T1 WF:              {"PASS" if t1_pass else "FAIL"}  ({pos_count}/5)')
    print(f'  T2 BS 3d:           {"PASS" if t2["pass"] else "FAIL"}  pos={t2.get("pos_rate", 0):.4f}')
    print(f'  T3 TT 60/40:        {"PASS" if t3_pass else "FAIL"}')
    print(f'  T4 daily ≥ 0.2%:    {"PASS" if t4_pass else "FAIL"}  {s["avg_daily_net_pct"]:+.4f}%')
    print(f'  T5 WR ≥ 40%:        {"PASS" if t5_pass else "FAIL"}  {s["win_rate"]:.4f}')
    print(f'  T6 R:R ≥ 1.0:       {"PASS" if t6_pass else "FAIL"}  {s["rr_ratio"]:.4f}')
    print(f'  T7 trades/day ≥ 2:  {"PASS" if t7_pass else "FAIL"}  {s["avg_trades_per_day"]:.4f}')
    print(f'  T8 gross > 0.07%:   {"PASS" if t8_pass else "FAIL"}  {s["avg_gross_per_trade_pct"]:+.4f}%')
    print(f'  T9 5d ≥ -10%:       {"PASS" if t9_pass else "FAIL"}  {s["worst_5d_pct"]:+.4f}%')
    print(f'  Sharpe annualized:  {s["sharpe_annualized"]:+.2f}')
    print(f'\n  HARD CRITERIA (T1-4, 7-9): {"ALL PASS" if all_hard_pass else "FAIL"}')
    print(f'  T5/T6 (relaxable per user): {"PASS" if (t5_pass and t6_pass) else "PARTIAL"}')

    out = {
        'date': datetime.now(timezone.utc).isoformat(),
        'pre_reg_commit': '654a015',
        'live_parity_prior': '0/1 (C1 failed live)',
        'verdict': 'PASS' if all_hard_pass else 'FAIL',
        'locked': LOCKED, 'gates': GATES,
        'gate_A': gA, 'gate_B': gB,
        'full_sample': s,
        'wf': {'folds': folds, 'pos_count': pos_count, 'pass': t1_pass},
        'bootstrap_3d': t2,
        'train_test': {'train': s_tr, 'test': s_te, 'pass': t3_pass},
        'tests': {
            'T4_magnitude': {'value': s['avg_daily_net_pct'], 'pass': t4_pass},
            'T5_WR': {'value': s['win_rate'], 'pass': t5_pass},
            'T6_RR': {'value': s['rr_ratio'], 'pass': t6_pass},
            'T7_freq': {'value': s['avg_trades_per_day'], 'pass': t7_pass},
            'T8_per_trade': {'value': s['avg_gross_per_trade_pct'], 'pass': t8_pass},
            'T9_tail': {'value': s['worst_5d_pct'], 'pass': t9_pass},
        },
        'all_hard_pass': bool(all_hard_pass),
    }
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    p = RESULTS / f'path_b_r7_leadlag_oos_{ts}.json'
    with open(p, 'w') as fp:
        json.dump(out, fp, indent=2, default=str)
    print(f'\nSaved: {p}')


if __name__ == '__main__':
    main()
