"""Path B R5 — BTC Single-Coin Cash-and-Carry Basis Harvest LOCKED OOS.

Pre-reg: claudedocs/path_b_r5_btc_cash_and_carry_prereg.md (commit 0874707)

Mechanism:
  Long BTC spot ($750) + short BTC perp ($750), delta-neutral.
  Regime filter: enter when 7d trailing funding APY ≥ 3%, exit when ≤ 0%.
  Net daily P&L = funding payments collected (perp short receives positive funding)
                  - friction (entry/exit setups)
                  - basis drift (assumed 0 for backtest, real-world risk)

Data: data/funding_history.parquet (BTC funding from R3 fetch, 2024-02-19 → 2026-04-29).

Gates:
  Gate A: positive funding regime ≥ 200 days (≥25% of panel)
  Gate B: 7d windows with mean fund > 0 ≥ 70%
  T1 WF, T2 BS, T3 TT
  T4 magnitude ≥ 4%/yr net APY (RELAXED to bank-interest bar)
  T5 tail worst 5d ≥ -3%
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

FUNDING_FILE = DATA / 'funding_history.parquet'

LOCKED = {
    'asset': 'BTC/USDT',
    'entry_threshold_apy_pct': 3.0,
    'exit_threshold_apy_pct': 0.0,
    'spot_friction_per_side_pct': 0.10,    # taker
    'perp_friction_per_side_pct': 0.04,    # maker
    'capital_usd': 1500,
    'spot_position_usd': 750,
    'perp_position_usd': 750,
    'lookback_funding_periods_for_regime': 21,  # 7d × 3/day
}

GATES = {
    'gate_A_min_positive_regime_days': 200,
    'gate_B_min_pos_window_rate': 0.70,
    'wf_min_pos_folds': 3,
    'wf_total_folds': 5,
    'bs_n_iter': 1000,
    'bs_window_days': 30,
    'bs_min_pos_rate': 0.50,
    'tt_split': 0.60,
    'magnitude_min_apy_pct': 4.0,         # bank interest baseline
    'tail_max_5d_dd_pct': 3.0,
}


def load_btc_funding() -> pd.DataFrame:
    df = pd.read_parquet(FUNDING_FILE)
    btc = df[df['symbol'] == LOCKED['asset']].copy()
    btc['datetime'] = pd.to_datetime(btc['datetime'])
    btc = btc.sort_values('datetime').reset_index(drop=True)
    return btc


def daily_funding(btc: pd.DataFrame) -> pd.Series:
    """Aggregate 8h funding payments into daily mean rate."""
    btc = btc.copy()
    btc['date'] = btc['datetime'].dt.date
    daily = btc.groupby('date')['funding_rate'].sum()  # 3 periods/day → daily rate
    daily.index = pd.to_datetime(daily.index)
    return daily


def gate_A(daily_fund_df: pd.DataFrame) -> dict:
    """Days where 7d trailing funding APY ≥ 3%."""
    rolling_apy = daily_fund_df['daily_funding'].rolling(7).mean() * 365 * 100
    n_pos = int((rolling_apy >= LOCKED['entry_threshold_apy_pct']).sum())
    return {
        'positive_regime_days': n_pos,
        'gate': GATES['gate_A_min_positive_regime_days'],
        'pass': n_pos >= GATES['gate_A_min_positive_regime_days'],
        'total_days': int(len(daily_fund_df)),
    }


def gate_B(daily_fund_df: pd.DataFrame) -> dict:
    """7d windows where mean funding > 0."""
    rolling_mean = daily_fund_df['daily_funding'].rolling(7).mean()
    valid = rolling_mean.dropna()
    n_pos = int((valid > 0).sum())
    n_total = int(len(valid))
    rate = n_pos / n_total if n_total > 0 else 0
    return {
        'pos_windows': n_pos,
        'total_windows': n_total,
        'pos_rate': rate,
        'gate': GATES['gate_B_min_pos_window_rate'],
        'pass': rate >= GATES['gate_B_min_pos_window_rate'],
    }


def run_carry(daily_fund_df: pd.DataFrame) -> pd.DataFrame:
    """
    Daily backtest. Carry collected each day position is held.

    Returns DataFrame: date, daily_funding, in_position, daily_funding_pnl_pct,
                       daily_friction_pct, daily_net_pct, action
    """
    df = daily_fund_df.copy()
    df['rolling_7d_apy'] = df['daily_funding'].rolling(7).mean() * 365 * 100

    in_pos = False
    rows = []
    spot_fric_rt = LOCKED['spot_friction_per_side_pct'] * 2 / 100.0   # round-trip on spot
    perp_fric_rt = LOCKED['perp_friction_per_side_pct'] * 2 / 100.0   # round-trip on perp

    for _, row in df.iterrows():
        action = 'hold' if in_pos else 'flat'
        friction_today_pct = 0.0
        funding_pnl_pct = 0.0
        rolling_apy = row['rolling_7d_apy']

        if not in_pos:
            if pd.notna(rolling_apy) and rolling_apy >= LOCKED['entry_threshold_apy_pct']:
                in_pos = True
                action = 'enter'
                # Friction: setup costs both legs, half charged at entry
                # Spot side: 0.10% × $750 / $1500 = 0.05% portfolio drag
                # Perp side: 0.04% × $750 / $1500 = 0.02% portfolio drag
                # entry-only side: 0.07% drag (will exit later for matching exit-side)
                friction_today_pct = (
                    (LOCKED['spot_friction_per_side_pct'] * LOCKED['spot_position_usd'] / LOCKED['capital_usd']) +
                    (LOCKED['perp_friction_per_side_pct'] * LOCKED['perp_position_usd'] / LOCKED['capital_usd'])
                )
        else:
            # Funding accrues on perp short side notional
            # daily_funding (rate) × perp_notional / capital × 100 = pct of capital
            funding_pnl_pct = (row['daily_funding'] * LOCKED['perp_position_usd'] / LOCKED['capital_usd']) * 100
            if pd.notna(rolling_apy) and rolling_apy <= LOCKED['exit_threshold_apy_pct']:
                in_pos = False
                action = 'exit'
                friction_today_pct = (
                    (LOCKED['spot_friction_per_side_pct'] * LOCKED['spot_position_usd'] / LOCKED['capital_usd']) +
                    (LOCKED['perp_friction_per_side_pct'] * LOCKED['perp_position_usd'] / LOCKED['capital_usd'])
                )

        net_pct = funding_pnl_pct - friction_today_pct
        rows.append({
            'date': row['date'],
            'daily_funding': row['daily_funding'],
            'rolling_7d_apy': rolling_apy,
            'in_position': in_pos,
            'action': action,
            'daily_funding_pnl_pct': funding_pnl_pct,
            'daily_friction_pct': friction_today_pct,
            'daily_net_pct': net_pct,
        })

    return pd.DataFrame(rows)


def summarize(bt: pd.DataFrame) -> dict:
    if bt.empty:
        return {'n_days': 0}
    n = len(bt)
    cum_net = float((1 + bt['daily_net_pct'] / 100).prod() - 1) * 100
    cum_funding = float(bt['daily_funding_pnl_pct'].sum())
    cum_friction = float(bt['daily_friction_pct'].sum())

    avg_daily_net = float(bt['daily_net_pct'].mean())
    annualized_net_apy = avg_daily_net * 365

    n_in_pos = int(bt['in_position'].sum())
    pos_rate = n_in_pos / n
    n_entries = int((bt['action'] == 'enter').sum())
    n_exits = int((bt['action'] == 'exit').sum())

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

    avg_trades_per_day = (n_entries + n_exits) / n

    return {
        'n_days': int(n),
        'cum_net_pct': cum_net,
        'cum_funding_pct': cum_funding,
        'cum_friction_pct': cum_friction,
        'avg_daily_net_pct': avg_daily_net,
        'annualized_net_apy_pct': annualized_net_apy,
        'days_in_position': n_in_pos,
        'position_rate': pos_rate,
        'n_entries': n_entries,
        'n_exits': n_exits,
        'avg_trades_per_day': avg_trades_per_day,
        'max_dd_pct': max_dd,
        'worst_5d_net_pct': worst_5d,
        'sharpe_annualized': sharpe,
    }


def test_1_walk_forward(daily_fund_df: pd.DataFrame) -> dict:
    folds = GATES['wf_total_folds']
    n = len(daily_fund_df)
    fold_size = n // (folds + 1)
    results = []
    for i in range(folds):
        s = (i + 1) * fold_size
        e = min(s + fold_size, n)
        sub = daily_fund_df.iloc[s:e].reset_index(drop=True)
        bt = run_carry(sub)
        summ = summarize(bt)
        results.append({'fold': i + 1, **summ})
    pos_count = sum(1 for r in results if r.get('cum_net_pct', 0) > 0)
    return {'folds': results, 'pos_count': pos_count, 'pass': pos_count >= GATES['wf_min_pos_folds']}


def test_2_bootstrap(bt: pd.DataFrame) -> dict:
    nets = bt['daily_net_pct'].values
    n = len(nets)
    win = GATES['bs_window_days']
    if n <= win:
        return {'pass': False, 'reason': 'panel too short'}
    random.seed(42)
    starts = random.sample(range(n - win), min(GATES['bs_n_iter'], n - win))
    cums = [(1 + nets[s:s+win] / 100).prod() - 1 for s in starts]
    arr = np.array(cums) * 100
    pos_rate = float((arr > 0).mean())
    return {
        'n_iter': len(arr),
        'mean_cum_pct': float(arr.mean()),
        'pos_rate': pos_rate,
        'p5': float(np.percentile(arr, 5)),
        'p95': float(np.percentile(arr, 95)),
        'pass': pos_rate >= GATES['bs_min_pos_rate'],
    }


def test_3_train_test(daily_fund_df: pd.DataFrame) -> dict:
    n = len(daily_fund_df)
    split = int(n * GATES['tt_split'])
    bt_tr = run_carry(daily_fund_df.iloc[:split].reset_index(drop=True))
    bt_te = run_carry(daily_fund_df.iloc[split:].reset_index(drop=True))
    s_tr = summarize(bt_tr)
    s_te = summarize(bt_te)
    return {
        'train': s_tr, 'test': s_te,
        'pass': (s_tr.get('cum_net_pct', 0) > 0) and (s_te.get('cum_net_pct', 0) > 0),
    }


def main():
    print('=' * 100)
    print('Path B R5 — BTC Single-Coin Cash-and-Carry')
    print('=' * 100)
    print('Pre-reg: claudedocs/path_b_r5_btc_cash_and_carry_prereg.md (0874707)')
    print(f'Locked: {LOCKED}\n')

    btc = load_btc_funding()
    print(f'BTC funding: {len(btc):,} 8h periods, '
          f'{btc.datetime.min().date()} → {btc.datetime.max().date()}')

    daily_fund = daily_funding(btc)
    df = pd.DataFrame({'date': daily_fund.index, 'daily_funding': daily_fund.values})
    print(f'Daily aggregated: {len(df)} days, '
          f'mean={df.daily_funding.mean()*100:.4f}%/day = '
          f'{df.daily_funding.mean()*365*100:+.2f}%/yr\n')

    print('=== Gate A — Positive funding regime existence ===')
    gA = gate_A(df)
    print(f'  days w/ 7d trail APY ≥ {LOCKED["entry_threshold_apy_pct"]}%: {gA["positive_regime_days"]}')
    print(f'  gate: ≥ {gA["gate"]} days  → {"PASS" if gA["pass"] else "FAIL"}\n')

    print('=== Gate B — Funding sign stability ===')
    gB = gate_B(df)
    print(f'  positive 7d windows: {gB["pos_windows"]}/{gB["total_windows"]} = {gB["pos_rate"]:.4f}')
    print(f'  gate: ≥ {gB["gate"]}  → {"PASS" if gB["pass"] else "FAIL"}\n')

    if not gA['pass'] or not gB['pass']:
        verdict = 'INCONCLUSIVE_VACUOUS'
        print(f'EARLY EXIT: {verdict}')
        out = {'date': datetime.now(timezone.utc).isoformat(), 'pre_reg_commit': '0874707',
               'verdict': verdict, 'gate_A': gA, 'gate_B': gB,
               'locked': LOCKED, 'gates': GATES}
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        p = RESULTS / f'path_b_r5_carry_oos_{ts}.json'
        with open(p, 'w') as fp:
            json.dump(out, fp, indent=2, default=str)
        print(f'Saved: {p}')
        return

    print('=== Full-sample backtest ===')
    bt_full = run_carry(df)
    s_full = summarize(bt_full)
    for k, v in s_full.items():
        if isinstance(v, float):
            print(f'  {k}: {v:+.4f}')
        else:
            print(f'  {k}: {v}')
    print()

    print('=== Test 1 — WF 5-fold ===')
    t1 = test_1_walk_forward(df)
    for f_ in t1['folds']:
        print(f'  fold {f_["fold"]}: cum_net={f_.get("cum_net_pct", 0):+.4f}%  '
              f'apy={f_.get("annualized_net_apy_pct", 0):+.2f}%  '
              f'pos_rate={f_.get("position_rate", 0):.2f}')
    print(f'  → {"PASS" if t1["pass"] else "FAIL"}  ({t1["pos_count"]}/5)\n')

    print('=== Test 2 — Bootstrap 1000 × 30d ===')
    t2 = test_2_bootstrap(bt_full)
    print(f'  pos_rate: {t2.get("pos_rate", 0):.4f}  '
          f'mean: {t2.get("mean_cum_pct", 0):+.4f}%  '
          f'p5: {t2.get("p5", 0):+.4f}%  p95: {t2.get("p95", 0):+.4f}%')
    print(f'  → {"PASS" if t2["pass"] else "FAIL"}\n')

    print('=== Test 3 — Train/Test 60/40 ===')
    t3 = test_3_train_test(df)
    print(f'  train: cum_net={t3["train"].get("cum_net_pct", 0):+.4f}%  '
          f'apy={t3["train"].get("annualized_net_apy_pct", 0):+.2f}%')
    print(f'  test:  cum_net={t3["test"].get("cum_net_pct", 0):+.4f}%  '
          f'apy={t3["test"].get("annualized_net_apy_pct", 0):+.2f}%')
    print(f'  → {"PASS" if t3["pass"] else "FAIL"}\n')

    apy = s_full['annualized_net_apy_pct']
    t4_pass = apy >= GATES['magnitude_min_apy_pct']
    print(f'=== T4 Magnitude (≥{GATES["magnitude_min_apy_pct"]}%/yr = bank interest) ===')
    print(f'  net APY: {apy:+.4f}%/yr  → {"PASS" if t4_pass else "FAIL"}\n')

    worst = s_full['worst_5d_net_pct']
    t5_pass = worst >= -GATES['tail_max_5d_dd_pct']
    print(f'=== T5 Tail (worst 5d ≥ -{GATES["tail_max_5d_dd_pct"]}%) ===')
    print(f'  worst 5d: {worst:+.4f}%  → {"PASS" if t5_pass else "FAIL"}\n')

    all_pass = t1['pass'] and t2['pass'] and t3['pass'] and t4_pass and t5_pass

    print('=' * 100)
    print('FINAL VERDICT')
    print('=' * 100)
    print(f'  Gate A:         PASS')
    print(f'  Gate B:         PASS')
    print(f'  T1 WF:          {"PASS" if t1["pass"] else "FAIL"}  ({t1["pos_count"]}/5)')
    print(f'  T2 BS30d:       {"PASS" if t2["pass"] else "FAIL"}  pos={t2.get("pos_rate", 0):.4f}')
    print(f'  T3 TT60/40:     {"PASS" if t3["pass"] else "FAIL"}')
    print(f'  T4 Magnitude:   {"PASS" if t4_pass else "FAIL"}  apy={apy:+.2f}%')
    print(f'  T5 Tail:        {"PASS" if t5_pass else "FAIL"}  5d={worst:+.4f}%')
    print(f'\n  OVERALL: {"ALL 5 PASS — first deployable beat-bank-interest strategy" if all_pass else "FAIL"}')

    out = {
        'date': datetime.now(timezone.utc).isoformat(),
        'pre_reg_commit': '0874707',
        'verdict': 'PASS' if all_pass else 'FAIL',
        'locked': LOCKED, 'gates': GATES,
        'gate_A': gA, 'gate_B': gB,
        'full_sample': s_full,
        'test_1_wf': t1, 'test_2_bs': t2, 'test_3_tt': t3,
        'test_4_magnitude': {'apy_pct': apy, 'pass': t4_pass},
        'test_5_tail': {'worst_5d_pct': worst, 'pass': t5_pass},
        'all_pass': bool(all_pass),
    }
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    p = RESULTS / f'path_b_r5_carry_oos_{ts}.json'
    with open(p, 'w') as fp:
        json.dump(out, fp, indent=2, default=str)
    print(f'\nSaved: {p}')


if __name__ == '__main__':
    main()
