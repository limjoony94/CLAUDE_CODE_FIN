"""Path B R5 — BTC Cash-and-Carry with full MTM (basis-aware re-run).

Per advisor 2026-04-29: original R5 BT only modeled funding accrual,
producing artifact Sharpe 11+ and worst-5d -0.088%. Real position MTM
includes spot-perp basis variation which can be ±0.1-0.5%/day.

Per R5 pre-reg caveat #1: "spot-perp basis risk... can widen 0.5-2% during
BTC sell-offs. Not modeled here." This addresses that disclosed limitation.

Method:
  Daily MTM_pct = funding_accrual_pct + Δbasis_pct × (perp_notional / capital)
  where basis = (perp_close - spot_close) / spot_close

Friction unchanged. Gates unchanged. Same locked params.
"""
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / 'scripts' / 'analysis'))
import path_b_r5_btc_cash_and_carry as r5mod

DATA = ROOT / 'data'
RESULTS = ROOT / 'results'

SPOT_FILE = DATA / 'multi_asset_daily.parquet'
PERP_FILE = DATA / 'btc_perp_daily_binance.parquet'


def load_data() -> pd.DataFrame:
    btc = r5mod.load_btc_funding()
    daily_fund = r5mod.daily_funding(btc)

    spot = pd.read_parquet(SPOT_FILE)
    spot = spot[spot['symbol'] == 'BTC/USDT'].copy()
    spot['date'] = pd.to_datetime(spot['date'])
    spot = spot[['date', 'close']].rename(columns={'close': 'spot_close'})

    perp = pd.read_parquet(PERP_FILE)
    perp['date'] = pd.to_datetime(perp['date'])
    perp = perp[['date', 'close']].rename(columns={'close': 'perp_close'})

    df = pd.DataFrame({
        'date': pd.to_datetime(daily_fund.index),
        'daily_funding': daily_fund.values,
    })
    df = df.merge(spot, on='date', how='inner').merge(perp, on='date', how='inner')
    df = df.sort_values('date').reset_index(drop=True)
    df['basis_pct'] = (df['perp_close'] - df['spot_close']) / df['spot_close'] * 100
    df['delta_basis_pct'] = df['basis_pct'].diff()
    return df


def run_carry_with_mtm(df: pd.DataFrame) -> pd.DataFrame:
    """Same as r5mod.run_carry but adds Δbasis × perp_notional/capital to net P&L."""
    LOCKED = r5mod.LOCKED
    df = df.copy()
    df['rolling_7d_apy'] = df['daily_funding'].rolling(7).mean() * 365 * 100

    in_pos = False
    rows = []
    perp_weight = LOCKED['perp_position_usd'] / LOCKED['capital_usd']
    spot_weight = LOCKED['spot_position_usd'] / LOCKED['capital_usd']

    spot_ret = df['spot_close'].pct_change() * 100
    perp_ret_proxy = df['perp_close'].pct_change() * 100

    for i, row in df.iterrows():
        action = 'hold' if in_pos else 'flat'
        friction_today_pct = 0.0
        funding_pnl_pct = 0.0
        basis_mtm_pct = 0.0
        rolling_apy = row['rolling_7d_apy']

        if not in_pos:
            if pd.notna(rolling_apy) and rolling_apy >= LOCKED['entry_threshold_apy_pct']:
                in_pos = True
                action = 'enter'
                friction_today_pct = (
                    (LOCKED['spot_friction_per_side_pct'] * spot_weight) +
                    (LOCKED['perp_friction_per_side_pct'] * perp_weight)
                )
        else:
            funding_pnl_pct = (row['daily_funding'] * perp_weight) * 100
            if i > 0 and pd.notna(spot_ret.iloc[i]) and pd.notna(perp_ret_proxy.iloc[i]):
                basis_mtm_pct = (spot_ret.iloc[i] * spot_weight) - (perp_ret_proxy.iloc[i] * perp_weight)
            if pd.notna(rolling_apy) and rolling_apy <= LOCKED['exit_threshold_apy_pct']:
                in_pos = False
                action = 'exit'
                friction_today_pct = (
                    (LOCKED['spot_friction_per_side_pct'] * spot_weight) +
                    (LOCKED['perp_friction_per_side_pct'] * perp_weight)
                )

        net_pct = funding_pnl_pct + basis_mtm_pct - friction_today_pct
        rows.append({
            'date': row['date'],
            'daily_funding': row['daily_funding'],
            'rolling_7d_apy': rolling_apy,
            'in_position': in_pos,
            'action': action,
            'daily_funding_pnl_pct': funding_pnl_pct,
            'daily_basis_mtm_pct': basis_mtm_pct,
            'daily_friction_pct': friction_today_pct,
            'daily_net_pct': net_pct,
            'spot_close': row['spot_close'],
            'perp_close': row['perp_close'],
            'basis_pct': row['basis_pct'],
        })
    return pd.DataFrame(rows)


def summarize(bt: pd.DataFrame) -> dict:
    if bt.empty:
        return {'n_days': 0}
    n = len(bt)
    cum_net = float((1 + bt['daily_net_pct'] / 100).prod() - 1) * 100
    cum_funding = float(bt['daily_funding_pnl_pct'].sum())
    cum_basis = float(bt['daily_basis_mtm_pct'].sum())
    cum_fric = float(bt['daily_friction_pct'].sum())
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

    n_in = int(bt['in_position'].sum())
    return {
        'n_days': int(n),
        'cum_net_pct': cum_net,
        'cum_funding_pct': cum_funding,
        'cum_basis_mtm_pct': cum_basis,
        'cum_friction_pct': cum_fric,
        'avg_daily_net_pct': avg_daily_net,
        'annualized_net_apy_pct': annualized_apy,
        'days_in_position': n_in,
        'position_rate': n_in / n,
        'n_entries': int((bt['action'] == 'enter').sum()),
        'n_exits': int((bt['action'] == 'exit').sum()),
        'max_dd_pct': max_dd,
        'worst_5d_net_pct': worst_5d,
        'sharpe_annualized': sharpe,
        'daily_basis_std_pct': float(bt['daily_basis_mtm_pct'].std()),
    }


def main():
    print('=' * 100)
    print('Path B R5 (MTM-aware re-run) — BTC Cash-and-Carry with basis P&L')
    print('=' * 100)
    print('Per advisor: original R5 modeled funding-only, producing artifact Sharpe 11+')
    print('This re-run adds Δbasis × notional MTM per R5 pre-reg caveat #1.\n')

    df = load_data()
    print(f'Date range: {df.date.min().date()} → {df.date.max().date()}')
    print(f'BTC daily basis: mean={df.basis_pct.mean():+.4f}%  '
          f'std={df.basis_pct.std():.4f}%  '
          f'min={df.basis_pct.min():+.4f}%  '
          f'max={df.basis_pct.max():+.4f}%')
    print(f'Δbasis daily std: {df.delta_basis_pct.std():.4f}% (this is the missing volatility source)\n')

    print('=== Full-sample BT with MTM ===')
    bt = run_carry_with_mtm(df)
    s = summarize(bt)
    for k, v in s.items():
        if isinstance(v, float):
            print(f'  {k}: {v:+.4f}')
        else:
            print(f'  {k}: {v}')
    print()

    print('Decomposition (cumulative):')
    print(f'  funding gain:    {s["cum_funding_pct"]:+.4f}%')
    print(f'  basis MTM drift: {s["cum_basis_mtm_pct"]:+.4f}%')
    print(f'  friction cost:   {s["cum_friction_pct"]:+.4f}%')
    print(f'  net (compounded): {s["cum_net_pct"]:+.4f}%\n')

    print('=== Test 1 — WF 5-fold ===')
    folds = []
    n = len(df)
    fold_size = n // 6
    for i in range(5):
        ss = (i + 1) * fold_size
        ee = min(ss + fold_size, n)
        sub = df.iloc[ss:ee].reset_index(drop=True)
        bt_f = run_carry_with_mtm(sub)
        sf = summarize(bt_f)
        folds.append({'fold': i + 1, **sf})
        print(f'  fold {i+1}: cum_net={sf["cum_net_pct"]:+.4f}%  '
              f'apy={sf["annualized_net_apy_pct"]:+.2f}%  '
              f'sharpe={sf["sharpe_annualized"]:+.2f}')
    pos_count = sum(1 for f in folds if f['cum_net_pct'] > 0)
    t1_pass = pos_count >= 3
    print(f'  → {"PASS" if t1_pass else "FAIL"}  ({pos_count}/5)\n')

    print('=== Test 2 — Bootstrap 1000 × 30d ===')
    nets = bt['daily_net_pct'].values
    win = 30
    random.seed(42)
    starts = random.sample(range(len(nets) - win), min(1000, len(nets) - win))
    cums = [(1 + nets[s:s+win] / 100).prod() - 1 for s in starts]
    arr = np.array(cums) * 100
    pos_rate = float((arr > 0).mean())
    t2_pass = pos_rate >= 0.5
    print(f'  pos_rate: {pos_rate:.4f}  mean: {arr.mean():+.4f}%  '
          f'p5: {np.percentile(arr, 5):+.4f}%  p95: {np.percentile(arr, 95):+.4f}%')
    print(f'  → {"PASS" if t2_pass else "FAIL"}\n')

    print('=== Test 3 — Train/Test 60/40 ===')
    split = int(n * 0.6)
    bt_tr = run_carry_with_mtm(df.iloc[:split].reset_index(drop=True))
    bt_te = run_carry_with_mtm(df.iloc[split:].reset_index(drop=True))
    s_tr = summarize(bt_tr)
    s_te = summarize(bt_te)
    t3_pass = (s_tr['cum_net_pct'] > 0) and (s_te['cum_net_pct'] > 0)
    print(f'  train: cum_net={s_tr["cum_net_pct"]:+.4f}%  apy={s_tr["annualized_net_apy_pct"]:+.2f}%')
    print(f'  test:  cum_net={s_te["cum_net_pct"]:+.4f}%  apy={s_te["annualized_net_apy_pct"]:+.2f}%')
    print(f'  → {"PASS" if t3_pass else "FAIL"}\n')

    apy = s['annualized_net_apy_pct']
    t4_pass = apy >= 4.0
    print(f'=== T4 Magnitude (≥4%/yr bank interest) ===')
    print(f'  apy: {apy:+.4f}%  → {"PASS" if t4_pass else "FAIL"}\n')

    worst = s['worst_5d_net_pct']
    t5_pass = worst >= -3.0
    print(f'=== T5 Tail (worst 5d ≥ -3%) ===')
    print(f'  worst 5d: {worst:+.4f}%  → {"PASS" if t5_pass else "FAIL"}\n')

    all_pass = t1_pass and t2_pass and t3_pass and t4_pass and t5_pass

    print('=' * 100)
    print('FINAL VERDICT (MTM-aware)')
    print('=' * 100)
    print(f'  T1 WF:        {"PASS" if t1_pass else "FAIL"}  ({pos_count}/5)')
    print(f'  T2 BS30d:     {"PASS" if t2_pass else "FAIL"}  pos={pos_rate:.4f}')
    print(f'  T3 TT60/40:   {"PASS" if t3_pass else "FAIL"}')
    print(f'  T4 Magnitude: {"PASS" if t4_pass else "FAIL"}  apy={apy:+.2f}%')
    print(f'  T5 Tail:      {"PASS" if t5_pass else "FAIL"}  5d={worst:+.4f}%')
    print(f'  Sharpe:       {s["sharpe_annualized"]:+.2f}  (was 11+ in funding-only model)')
    print(f'  Max DD:       {s["max_dd_pct"]:+.4f}%  (was 0.16% in funding-only model)')
    print(f'\n  OVERALL: {"ALL 5 PASS" if all_pass else "FAIL"}')

    out = {
        'date': datetime.now(timezone.utc).isoformat(),
        'pre_reg_commit': '0874707',
        'mtm_aware': True,
        'verdict': 'PASS' if all_pass else 'FAIL',
        'locked': r5mod.LOCKED,
        'full_sample': s,
        'wf': {'folds': folds, 'pos_count': pos_count, 'pass': t1_pass},
        'bootstrap': {'pos_rate': pos_rate, 'mean': float(arr.mean()),
                      'p5': float(np.percentile(arr, 5)),
                      'p95': float(np.percentile(arr, 95)), 'pass': t2_pass},
        'train_test': {'train': s_tr, 'test': s_te, 'pass': t3_pass},
        'magnitude': {'apy_pct': apy, 'pass': t4_pass},
        'tail': {'worst_5d_pct': worst, 'pass': t5_pass},
        'all_pass': bool(all_pass),
    }
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    p = RESULTS / f'path_b_r5_carry_mtm_oos_{ts}.json'
    with open(p, 'w') as fp:
        json.dump(out, fp, indent=2, default=str)
    print(f'\nSaved: {p}')


if __name__ == '__main__':
    main()
