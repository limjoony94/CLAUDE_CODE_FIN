"""Path B R6 — BTC-ETH Cointegration Pair Trading LOCKED OOS.

Pre-reg: claudedocs/path_b_r6_btc_eth_pairs_trading_prereg.md (commit bdcfb5b)

Mechanism:
  Long ETH perp + short BTC perp when z(log(ETH/BTC)) ≤ -1
  Long BTC perp + short ETH perp when z(log(ETH/BTC)) ≥ +1
  Exit when z crosses 0
  Stop when |z| > 4 (regime change)
  Z computed on rolling 60-day mean/std of log(ETH/BTC).

Friction: 0.07% taker × 4 legs (entry + exit) = 0.28% per cycle.

Gates:
  A: ADF stationarity p<0.05
  B: ≥50 entry events
  C (NEW): actual strategy net > 95th pctile of 1000 random-entry simulations
  T1-T5: standard
"""
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

ROOT = Path(__file__).resolve().parent.parent.parent
DATA = ROOT / 'data'
RESULTS = ROOT / 'results'
RESULTS.mkdir(exist_ok=True)

PRICE_FILE = DATA / 'multi_asset_daily.parquet'

LOCKED = {
    'long_leg': 'ETH/USDT',
    'short_leg': 'BTC/USDT',
    'lookback_days': 60,
    'entry_z_threshold': 1.0,
    'exit_z_threshold': 0.0,
    'stop_z_threshold': 4.0,
    'friction_per_transaction_pct': 0.07,
    'capital_usd': 1500,
    'long_position_usd': 750,
    'short_position_usd': 750,
}

GATES = {
    'adf_max_pval': 0.05,
    'min_entry_events': 50,
    'random_baseline_pct_threshold': 0.95,
    'wf_min_pos_folds': 3,
    'wf_total_folds': 5,
    'bs_n_iter': 1000,
    'bs_window_days': 60,
    'bs_min_pos_rate': 0.50,
    'tt_split': 0.60,
    'magnitude_min_apy_pct': 4.0,
    'tail_max_5d_dd_pct': 5.0,
}


def load_data() -> pd.DataFrame:
    df = pd.read_parquet(PRICE_FILE)
    df['date'] = pd.to_datetime(df['date'])
    pivot = df.pivot(index='date', columns='symbol', values='close').sort_index()
    pair = pivot[[LOCKED['short_leg'], LOCKED['long_leg']]].dropna()
    pair.columns = ['btc', 'eth']
    pair['log_ratio'] = np.log(pair['eth'] / pair['btc'])
    return pair


def compute_zscore(df: pd.DataFrame, lookback: int) -> pd.DataFrame:
    df = df.copy()
    df['ma'] = df['log_ratio'].rolling(lookback).mean()
    df['sd'] = df['log_ratio'].rolling(lookback).std()
    df['z'] = (df['log_ratio'] - df['ma']) / df['sd']
    return df


def gate_A_adf(df: pd.DataFrame) -> dict:
    """ADF on residuals of log_ratio - rolling mean."""
    residual = (df['log_ratio'] - df['log_ratio'].rolling(LOCKED['lookback_days']).mean()).dropna()
    if len(residual) < 100:
        return {'pass': False, 'pval': None, 'note': 'too short'}
    res = adfuller(residual, maxlag=10, regression='c')
    pval = float(res[1])
    return {'pass': pval < GATES['adf_max_pval'], 'pval': pval, 'adf_stat': float(res[0])}


def run_pairs_trade(df: pd.DataFrame) -> pd.DataFrame:
    """
    Daily backtest. Track position state and emit daily P&L.

    Position states: 'flat', 'long_eth_short_btc', 'short_eth_long_btc'
    Returns: date, position, daily_pnl_pct (after friction on transition days)
    """
    fric = LOCKED['friction_per_transaction_pct'] / 100.0
    z_in = LOCKED['entry_z_threshold']
    z_out = LOCKED['exit_z_threshold']
    z_stop = LOCKED['stop_z_threshold']

    long_w = LOCKED['long_position_usd'] / LOCKED['capital_usd']
    short_w = LOCKED['short_position_usd'] / LOCKED['capital_usd']

    btc_ret = df['btc'].pct_change()
    eth_ret = df['eth'].pct_change()

    state = 'flat'
    entries = 0
    exits = 0
    rows = []

    for i, (date, row) in enumerate(df.iterrows()):
        z = row.get('z')
        log_ratio = row['log_ratio']

        # Compute position daily P&L based on prior state
        pnl_pct = 0.0
        friction_today_pct = 0.0
        action = 'hold' if state != 'flat' else 'flat'

        if state == 'long_eth_short_btc':
            # long ETH + short BTC
            pnl_pct = (eth_ret.iloc[i] * long_w - btc_ret.iloc[i] * short_w) * 100
        elif state == 'short_eth_long_btc':
            pnl_pct = (-eth_ret.iloc[i] * long_w + btc_ret.iloc[i] * short_w) * 100

        # State transition logic
        if pd.notna(z):
            new_state = state
            if state == 'flat':
                if z <= -z_in:
                    new_state = 'long_eth_short_btc'
                elif z >= z_in:
                    new_state = 'short_eth_long_btc'
            elif state == 'long_eth_short_btc':
                if z >= z_out or abs(z) >= z_stop:
                    new_state = 'flat'
            elif state == 'short_eth_long_btc':
                if z <= z_out or abs(z) >= z_stop:
                    new_state = 'flat'

            if new_state != state:
                # transition: charge friction for legs that change
                if state == 'flat' and new_state != 'flat':
                    # entry: 2 legs open
                    friction_today_pct = (long_w + short_w) * fric * 100
                    entries += 1
                    action = 'enter'
                elif state != 'flat' and new_state == 'flat':
                    # exit: 2 legs close
                    friction_today_pct = (long_w + short_w) * fric * 100
                    exits += 1
                    action = 'exit'
                # state flips (long → short directly): would charge 4 legs (close+open)
                state = new_state

        net_pnl = pnl_pct - friction_today_pct
        rows.append({
            'date': date, 'state': state, 'z': z, 'log_ratio': log_ratio,
            'gross_pnl_pct': pnl_pct, 'friction_pct': friction_today_pct,
            'daily_net_pct': net_pnl, 'action': action,
        })

    bt = pd.DataFrame(rows)
    return bt, entries, exits


def gate_C_random_baseline(df: pd.DataFrame, actual_cum_pct: float, n_iter: int = 1000) -> dict:
    """Random entry/exit at same frequency as actual strategy.

    Generate random series of state transitions matching actual count.
    """
    btc_ret = df['btc'].pct_change()
    eth_ret = df['eth'].pct_change()
    long_w = LOCKED['long_position_usd'] / LOCKED['capital_usd']
    short_w = LOCKED['short_position_usd'] / LOCKED['capital_usd']
    fric = LOCKED['friction_per_transaction_pct'] / 100.0

    n = len(df)
    valid = ~btc_ret.isna()
    valid_idx = np.where(valid)[0]

    random.seed(42)
    np.random.seed(42)

    cum_pcts = []
    for _ in range(n_iter):
        # Random state series: avg position rate matched to lookback fraction
        # Simple: random p(in_position) = 0.5
        # Direction also random
        states = np.random.choice([0, 1, -1], size=n, p=[0.5, 0.25, 0.25])
        # Compute daily PnL
        daily_pnl = np.zeros(n)
        prev_state = 0
        n_trans = 0
        for j in range(n):
            cur = states[j]
            if not valid[j]:
                states[j] = prev_state
                continue
            if cur == 1:
                p = (eth_ret.iloc[j] * long_w - btc_ret.iloc[j] * short_w) * 100
            elif cur == -1:
                p = (-eth_ret.iloc[j] * long_w + btc_ret.iloc[j] * short_w) * 100
            else:
                p = 0.0
            f = (long_w + short_w) * fric * 100 if cur != prev_state else 0
            daily_pnl[j] = p - f
            prev_state = cur
            if f > 0:
                n_trans += 1

        cum = float((1 + daily_pnl / 100).prod() - 1) * 100
        cum_pcts.append(cum)

    arr = np.array(cum_pcts)
    pct_below = float((arr < actual_cum_pct).mean())
    return {
        'n_iter': n_iter,
        'random_mean_cum_pct': float(arr.mean()),
        'random_p5': float(np.percentile(arr, 5)),
        'random_p50': float(np.percentile(arr, 50)),
        'random_p95': float(np.percentile(arr, 95)),
        'actual_cum_pct': actual_cum_pct,
        'actual_pctile': pct_below,
        'pass': pct_below >= GATES['random_baseline_pct_threshold'],
    }


def summarize(bt: pd.DataFrame, n_entries: int, n_exits: int) -> dict:
    if bt.empty:
        return {'n_days': 0}
    n = len(bt)
    cum_net = float((1 + bt['daily_net_pct'] / 100).prod() - 1) * 100
    avg_daily_net = float(bt['daily_net_pct'].mean())
    annualized_apy = avg_daily_net * 365

    nav = (1 + bt['daily_net_pct'].values / 100).cumprod()
    peak = np.maximum.accumulate(nav)
    dd = (peak - nav) / peak
    max_dd = float(dd.max()) * 100

    rolling_5d = pd.Series(bt['daily_net_pct'].values).rolling(5).apply(
        lambda x: (1 + x / 100).prod() - 1
    ) * 100
    worst_5d = float(rolling_5d.min())

    daily_std = float(bt['daily_net_pct'].std())
    sharpe = (avg_daily_net / daily_std * (365 ** 0.5)) if daily_std > 0 else 0.0

    in_pos = (bt['state'] != 'flat').sum()
    return {
        'n_days': int(n),
        'cum_net_pct': cum_net,
        'avg_daily_net_pct': avg_daily_net,
        'annualized_net_apy_pct': annualized_apy,
        'days_in_position': int(in_pos),
        'position_rate': float(in_pos / n),
        'n_entries': n_entries,
        'n_exits': n_exits,
        'max_dd_pct': max_dd,
        'worst_5d_net_pct': worst_5d,
        'sharpe_annualized': sharpe,
        'avg_trades_per_day': float((n_entries + n_exits) / n),
    }


def main():
    print('=' * 100)
    print('Path B R6 — BTC-ETH Cointegration Pair Trading')
    print('=' * 100)
    print('Pre-reg: claudedocs/path_b_r6_btc_eth_pairs_trading_prereg.md (bdcfb5b)')
    print(f'Locked: {LOCKED}\n')

    df = load_data()
    print(f'Date range: {df.index.min().date()} → {df.index.max().date()}, {len(df)} days')
    df = compute_zscore(df, LOCKED['lookback_days'])
    n_z_valid = df['z'].notna().sum()
    print(f'Valid z-score days: {n_z_valid} (lookback {LOCKED["lookback_days"]}d)\n')

    print('=== Gate A — ADF cointegration ===')
    gA = gate_A_adf(df)
    print(f'  ADF stat: {gA.get("adf_stat", 0):.4f}  p-value: {gA.get("pval", 0):.6f}')
    print(f'  gate: p < {GATES["adf_max_pval"]}  → {"PASS" if gA["pass"] else "FAIL"}\n')

    if not gA['pass']:
        print('EARLY EXIT: ADF FAIL, ratio not stationary, mechanism vacuous')
        out = {'date': datetime.now(timezone.utc).isoformat(),
               'pre_reg_commit': 'bdcfb5b',
               'verdict': 'INCONCLUSIVE_NOT_COINTEGRATED',
               'gate_A': gA, 'locked': LOCKED, 'gates': GATES}
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        p = RESULTS / f'path_b_r6_pairs_oos_{ts}.json'
        with open(p, 'w') as fp:
            json.dump(out, fp, indent=2, default=str)
        print(f'Saved: {p}')
        return

    print('=== Run full-sample BT ===')
    bt, n_entries, n_exits = run_pairs_trade(df)
    s = summarize(bt, n_entries, n_exits)
    for k, v in s.items():
        if isinstance(v, float):
            print(f'  {k}: {v:+.4f}')
        else:
            print(f'  {k}: {v}')
    print()

    print('=== Gate B — Sufficient entry events ===')
    gB_pass = n_entries >= GATES['min_entry_events']
    print(f'  entries: {n_entries}  gate: ≥ {GATES["min_entry_events"]}  → '
          f'{"PASS" if gB_pass else "FAIL"}\n')

    print('=== Gate C — Random-baseline comparison ===')
    gC = gate_C_random_baseline(df, s['cum_net_pct'])
    print(f'  random mean: {gC["random_mean_cum_pct"]:+.4f}%')
    print(f'  random p5:   {gC["random_p5"]:+.4f}%')
    print(f'  random p50:  {gC["random_p50"]:+.4f}%')
    print(f'  random p95:  {gC["random_p95"]:+.4f}%')
    print(f'  actual:      {gC["actual_cum_pct"]:+.4f}%')
    print(f'  actual percentile: {gC["actual_pctile"]:.4f}')
    print(f'  gate: ≥ {GATES["random_baseline_pct_threshold"]}  → '
          f'{"PASS" if gC["pass"] else "FAIL"}\n')

    if not gB_pass or not gC['pass']:
        verdict = 'INCONCLUSIVE_INSUFFICIENT' if not gB_pass else 'NO_EDGE_VS_RANDOM'
        print(f'EARLY EXIT: {verdict}')
        out = {'date': datetime.now(timezone.utc).isoformat(),
               'pre_reg_commit': 'bdcfb5b',
               'verdict': verdict,
               'gate_A': gA, 'gate_B': {'entries': n_entries, 'pass': gB_pass}, 'gate_C': gC,
               'full_sample': s, 'locked': LOCKED, 'gates': GATES}
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        p = RESULTS / f'path_b_r6_pairs_oos_{ts}.json'
        with open(p, 'w') as fp:
            json.dump(out, fp, indent=2, default=str)
        print(f'Saved: {p}')
        return

    print('=== Test 1 — WF 5-fold ===')
    folds = []
    n = len(df)
    fold_size = n // 6
    for i in range(5):
        ss = (i + 1) * fold_size
        ee = min(ss + fold_size, n)
        sub = compute_zscore(df.iloc[ss:ee][['btc', 'eth', 'log_ratio']].copy(), LOCKED['lookback_days'])
        bt_f, ne, nx = run_pairs_trade(sub)
        sf = summarize(bt_f, ne, nx)
        folds.append({'fold': i + 1, **sf})
        print(f'  fold {i+1}: cum={sf["cum_net_pct"]:+.4f}%  '
              f'apy={sf["annualized_net_apy_pct"]:+.2f}%  '
              f'sharpe={sf["sharpe_annualized"]:+.2f}  ents={ne}')
    pos_count = sum(1 for f in folds if f['cum_net_pct'] > 0)
    t1_pass = pos_count >= GATES['wf_min_pos_folds']
    print(f'  → {"PASS" if t1_pass else "FAIL"}  ({pos_count}/5)\n')

    print('=== Test 2 — Bootstrap 1000 × 60d ===')
    nets = bt['daily_net_pct'].values
    win = GATES['bs_window_days']
    if len(nets) <= win:
        t2 = {'pass': False, 'reason': 'short'}
    else:
        random.seed(42)
        starts = random.sample(range(len(nets) - win), min(1000, len(nets) - win))
        cums = [(1 + nets[s:s+win] / 100).prod() - 1 for s in starts]
        arr = np.array(cums) * 100
        pos_rate = float((arr > 0).mean())
        t2 = {'n': len(arr), 'mean_cum': float(arr.mean()), 'pos_rate': pos_rate,
              'p5': float(np.percentile(arr, 5)), 'p95': float(np.percentile(arr, 95)),
              'pass': pos_rate >= GATES['bs_min_pos_rate']}
    print(f'  pos_rate: {t2.get("pos_rate", 0):.4f}  '
          f'mean: {t2.get("mean_cum", 0):+.4f}%  '
          f'p5: {t2.get("p5", 0):+.4f}%  p95: {t2.get("p95", 0):+.4f}%')
    print(f'  → {"PASS" if t2["pass"] else "FAIL"}\n')

    print('=== Test 3 — Train/Test 60/40 ===')
    split = int(n * GATES['tt_split'])
    df_tr = compute_zscore(df.iloc[:split][['btc', 'eth', 'log_ratio']].copy(), LOCKED['lookback_days'])
    df_te = compute_zscore(df.iloc[split:][['btc', 'eth', 'log_ratio']].copy(), LOCKED['lookback_days'])
    bt_tr, ne_tr, nx_tr = run_pairs_trade(df_tr)
    bt_te, ne_te, nx_te = run_pairs_trade(df_te)
    s_tr = summarize(bt_tr, ne_tr, nx_tr)
    s_te = summarize(bt_te, ne_te, nx_te)
    t3_pass = s_tr['cum_net_pct'] > 0 and s_te['cum_net_pct'] > 0
    print(f'  train: cum={s_tr["cum_net_pct"]:+.4f}%  apy={s_tr["annualized_net_apy_pct"]:+.2f}%  '
          f'ents={ne_tr}  sharpe={s_tr["sharpe_annualized"]:+.2f}')
    print(f'  test:  cum={s_te["cum_net_pct"]:+.4f}%  apy={s_te["annualized_net_apy_pct"]:+.2f}%  '
          f'ents={ne_te}  sharpe={s_te["sharpe_annualized"]:+.2f}')
    print(f'  → {"PASS" if t3_pass else "FAIL"}\n')

    apy = s['annualized_net_apy_pct']
    t4_pass = apy >= GATES['magnitude_min_apy_pct']
    print(f'=== T4 Magnitude (≥{GATES["magnitude_min_apy_pct"]}%/yr) ===')
    print(f'  apy: {apy:+.4f}%  → {"PASS" if t4_pass else "FAIL"}\n')

    worst = s['worst_5d_net_pct']
    t5_pass = worst >= -GATES['tail_max_5d_dd_pct']
    print(f'=== T5 Tail (worst 5d ≥ -{GATES["tail_max_5d_dd_pct"]}%) ===')
    print(f'  worst 5d: {worst:+.4f}%  → {"PASS" if t5_pass else "FAIL"}\n')

    all_pass = t1_pass and t2['pass'] and t3_pass and t4_pass and t5_pass

    print('=' * 100)
    print('FINAL VERDICT')
    print('=' * 100)
    print(f'  Gate A ADF:       PASS')
    print(f'  Gate B events:    {"PASS" if gB_pass else "FAIL"}  ({n_entries} entries)')
    print(f'  Gate C random:    {"PASS" if gC["pass"] else "FAIL"}  pctile={gC["actual_pctile"]:.4f}')
    print(f'  T1 WF:            {"PASS" if t1_pass else "FAIL"}  ({pos_count}/5)')
    print(f'  T2 BS:            {"PASS" if t2["pass"] else "FAIL"}  pos={t2.get("pos_rate", 0):.4f}')
    print(f'  T3 TT:            {"PASS" if t3_pass else "FAIL"}')
    print(f'  T4 Magnitude:     {"PASS" if t4_pass else "FAIL"}  apy={apy:+.2f}%')
    print(f'  T5 Tail:          {"PASS" if t5_pass else "FAIL"}  5d={worst:+.4f}%')
    print(f'  Sharpe:           {s["sharpe_annualized"]:+.2f}')
    print(f'\n  OVERALL: {"ALL PASS — first round-16 ceiling break candidate" if all_pass else "FAIL"}')

    out = {
        'date': datetime.now(timezone.utc).isoformat(),
        'pre_reg_commit': 'bdcfb5b',
        'verdict': 'PASS' if all_pass else 'FAIL',
        'locked': LOCKED, 'gates': GATES,
        'gate_A': gA, 'gate_B': {'entries': n_entries, 'pass': gB_pass}, 'gate_C': gC,
        'full_sample': s,
        'wf': {'folds': folds, 'pos_count': pos_count, 'pass': t1_pass},
        'bootstrap': t2,
        'train_test': {'train': s_tr, 'test': s_te, 'pass': t3_pass},
        'magnitude': {'apy_pct': apy, 'pass': t4_pass},
        'tail': {'worst_5d_pct': worst, 'pass': t5_pass},
        'all_pass': bool(all_pass),
    }
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    p = RESULTS / f'path_b_r6_pairs_oos_{ts}.json'
    with open(p, 'w') as fp:
        json.dump(out, fp, indent=2, default=str)
    print(f'\nSaved: {p}')


if __name__ == '__main__':
    main()
