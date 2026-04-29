"""Path B R3 — Funding-Rate Carry Harvest LOCKED OOS.

Pre-reg: claudedocs/path_b_r3_funding_carry_prereg.md (commit 4435c76)
Theory: Koijen-Moskowitz-Pedersen 2018 + Hu-Lu-Zhang-Zhuang 2024.

Mechanism:
  Daily, compute trailing 7-day mean funding rate per coin (21 funding periods).
  Long bottom-3 (lowest funding = longs over-paid least, may even receive).
  Short top-3 (highest funding = longs over-paid most, hedge collects).
  Equal-weight, weekly Monday rebalance.

Friction: 0.07% taker round-trip per leg (consistent with PB-R1).

Returns decomposition:
  net_pct = price_return_pct + funding_return_pct - friction_pct
  - price_return: PnL from price moves of held legs (long benefit, short benefit when price falls)
  - funding_return: cumulative funding payments collected (long pays funding > 0, short collects)
  - friction: 0.07% per swap on each leg's notional

5 gates:
  Gate A (orthogonality): Spearman ρ(funding_rank, momentum_rank) < 0.7
  Gate B (vacuity): median 7d cross-sectional funding-std ≥ 0.05%/8h
  T1 WF, T2 BS, T3 TT, T4 magnitude ≥ 0.02%/day, T5 tail ≥ -10%
"""
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent.parent
DATA = ROOT / 'data'
RESULTS = ROOT / 'results'
RESULTS.mkdir(exist_ok=True)

PRICE_FILE = DATA / 'multi_asset_daily.parquet'
FUNDING_FILE = DATA / 'funding_history.parquet'

LOCKED = {
    'universe': ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT', 'XRP/USDT',
                 'ADA/USDT', 'DOGE/USDT', 'AVAX/USDT', 'TRX/USDT', 'LINK/USDT'],
    'lookback_funding_periods': 21,           # 7 days × 3/day
    'long_bottom_n': 3,
    'short_top_n': 3,
    'rebalance_frequency_days': 7,
    'friction_per_transaction': 0.07,         # %
    'momentum_lookback_days_for_corr': 30,    # for orthogonality check
}

GATES = {
    'orth_max_rho': 0.7,
    'vacuity_min_funding_std_per_8h': 0.0005,  # 0.05%
    'wf_min_pos': 3,
    'wf_total': 5,
    'bs_n_iter': 1000,
    'bs_window_days': 30,
    'bs_min_pos_rate': 0.50,
    'tt_split': 0.60,
    'magnitude_min_daily_net_pct': 0.02,
    'tail_max_5d_dd_pct': 10.0,
}


def load_data():
    price = pd.read_parquet(PRICE_FILE)
    price['date'] = pd.to_datetime(price['date'])
    price_pivot = price.pivot(index='date', columns='symbol', values='close').sort_index()
    price_pivot = price_pivot[[c for c in LOCKED['universe'] if c in price_pivot.columns]]
    price_pivot = price_pivot.dropna(how='any')

    fund = pd.read_parquet(FUNDING_FILE)
    fund['date'] = pd.to_datetime(fund['date'])
    fund = fund.sort_values(['symbol', 'datetime'])
    daily_fund = fund.groupby(['date', 'symbol'])['funding_rate'].mean().reset_index()
    fund_pivot = daily_fund.pivot(index='date', columns='symbol', values='funding_rate').sort_index()
    fund_pivot = fund_pivot[[c for c in LOCKED['universe'] if c in fund_pivot.columns]]

    common = price_pivot.index.intersection(fund_pivot.index)
    price_pivot = price_pivot.loc[common].dropna(how='any')
    fund_pivot = fund_pivot.loc[common].reindex(price_pivot.index)
    return price_pivot, fund_pivot


def gate_A_orthogonality(price: pd.DataFrame, fund: pd.DataFrame) -> dict:
    """Compute Spearman rho between 7d funding rank and 30d momentum rank, panel-wide."""
    look_fund = LOCKED['lookback_funding_periods'] / 3  # in days
    look_mom = LOCKED['momentum_lookback_days_for_corr']
    look = max(int(look_fund), look_mom)

    funding_signal = fund.rolling(int(look_fund), min_periods=int(look_fund)).mean()
    momentum_signal = (price / price.shift(look_mom) - 1) * 100

    rhos = []
    for date in price.index[look:]:
        f_row = funding_signal.loc[date]
        m_row = momentum_signal.loc[date]
        valid = f_row.notna() & m_row.notna()
        if valid.sum() < 5:
            continue
        rho, _ = spearmanr(f_row[valid].rank(), m_row[valid].rank())
        if not np.isnan(rho):
            rhos.append(rho)
    if not rhos:
        return {'pass': False, 'mean_rho': None, 'note': 'insufficient overlap'}
    mean_rho = float(np.mean(rhos))
    median_rho = float(np.median(rhos))
    return {
        'pass': abs(mean_rho) < GATES['orth_max_rho'],
        'mean_rho': mean_rho,
        'median_rho': median_rho,
        'gate_max_abs_rho': GATES['orth_max_rho'],
        'n_dates': len(rhos),
    }


def gate_B_vacuity(fund: pd.DataFrame) -> dict:
    look = LOCKED['lookback_funding_periods'] / 3
    funding_signal = fund.rolling(int(look), min_periods=int(look)).mean()
    cs_std = funding_signal.std(axis=1).dropna()
    median_std = float(cs_std.median())
    return {
        'pass': median_std >= GATES['vacuity_min_funding_std_per_8h'],
        'median_funding_std_per_8h': median_std,
        'gate_min': GATES['vacuity_min_funding_std_per_8h'],
        'note': 'cross-sectional std of trailing 7d mean funding',
    }


def run_carry(price: pd.DataFrame, fund: pd.DataFrame,
              friction_pct: float = None) -> pd.DataFrame:
    """
    Daily backtest. Hold longs/shorts between weekly rebalances.

    Returns DataFrame with columns:
      date, longs, shorts, daily_price_pct, daily_funding_pct, daily_friction_pct, daily_net_pct
    """
    if friction_pct is None:
        friction_pct = LOCKED['friction_per_transaction']

    look_fund_days = int(LOCKED['lookback_funding_periods'] / 3)
    rebal = LOCKED['rebalance_frequency_days']
    long_n = LOCKED['long_bottom_n']
    short_n = LOCKED['short_top_n']
    fric = friction_pct / 100.0

    funding_signal = fund.rolling(look_fund_days, min_periods=look_fund_days).mean()
    daily_returns = price.pct_change()

    rows = []
    longs = set()
    shorts = set()
    rebal_dates = list(price.index[look_fund_days::rebal])

    for i, date in enumerate(price.index):
        if i < look_fund_days:
            rows.append({'date': date, 'longs': frozenset(), 'shorts': frozenset(),
                         'daily_price_pct': 0.0, 'daily_funding_pct': 0.0,
                         'daily_friction_pct': 0.0, 'daily_net_pct': 0.0,
                         'is_rebal': False, 'n_swaps': 0})
            continue

        is_rebal = (date in rebal_dates)
        n_swaps = 0
        friction_today = 0.0
        if is_rebal:
            sig = funding_signal.loc[date]
            valid = sig.dropna()
            if len(valid) >= long_n + short_n:
                ranked = valid.sort_values()
                new_longs = frozenset(ranked.head(long_n).index)
                new_shorts = frozenset(ranked.tail(short_n).index)
                long_swaps = (len(new_longs - longs) + len(longs - new_longs))
                short_swaps = (len(new_shorts - shorts) + len(shorts - new_shorts))
                n_swaps = long_swaps + short_swaps
                friction_today = n_swaps * fric / (long_n + short_n) * 100.0
                longs = new_longs
                shorts = new_shorts

        if longs or shorts:
            ret_today = daily_returns.loc[date]
            long_ret = float(ret_today[list(longs)].mean()) if longs else 0.0
            short_ret = -float(ret_today[list(shorts)].mean()) if shorts else 0.0
            price_pct = ((long_ret + short_ret) / 2) * 100

            f_today = fund.loc[date]
            long_fund = float(f_today[list(longs)].mean()) if longs else 0.0
            short_fund = float(f_today[list(shorts)].mean()) if shorts else 0.0
            funding_pct = (-long_fund + short_fund) / 2 * 100
        else:
            price_pct = 0.0
            funding_pct = 0.0

        net_pct = price_pct + funding_pct - friction_today
        rows.append({
            'date': date, 'longs': frozenset(longs), 'shorts': frozenset(shorts),
            'daily_price_pct': price_pct, 'daily_funding_pct': funding_pct,
            'daily_friction_pct': friction_today, 'daily_net_pct': net_pct,
            'is_rebal': is_rebal, 'n_swaps': n_swaps,
        })
    return pd.DataFrame(rows)


def summarize(bt: pd.DataFrame) -> dict:
    if bt.empty:
        return {'n_days': 0}
    active = bt[(bt['daily_price_pct'] != 0) | (bt['daily_funding_pct'] != 0)].copy()
    if active.empty:
        return {'n_days': 0}
    n = len(active)
    cum_net = float((1 + active['daily_net_pct'] / 100).prod() - 1) * 100
    cum_price = float(active['daily_price_pct'].sum())
    cum_funding = float(active['daily_funding_pct'].sum())
    cum_fric = float(active['daily_friction_pct'].sum())

    avg_daily_net = float(active['daily_net_pct'].mean())
    avg_daily_price = float(active['daily_price_pct'].mean())
    avg_daily_funding = float(active['daily_funding_pct'].mean())
    avg_daily_fric = float(active['daily_friction_pct'].mean())

    nav = (1 + active['daily_net_pct'].values / 100).cumprod()
    peak = np.maximum.accumulate(nav)
    dd = (peak - nav) / peak
    max_dd = float(dd.max()) * 100

    rolling_5d = pd.Series(active['daily_net_pct'].values).rolling(5).apply(
        lambda x: (1 + x / 100).prod() - 1
    ) * 100
    worst_5d = float(rolling_5d.min())

    daily_std = float(active['daily_net_pct'].std())
    sharpe = (avg_daily_net / daily_std * (365 ** 0.5)) if daily_std > 0 else 0.0

    weekly_buckets = active.groupby(pd.Grouper(key='date', freq='W'))[
        ['daily_price_pct', 'daily_funding_pct', 'daily_fric_pct'.replace('fric_pct', 'friction_pct'),
         'daily_net_pct']
    ].sum()
    avg_weekly_net = float(weekly_buckets['daily_net_pct'].mean()) if not weekly_buckets.empty else 0.0

    decomposition = {
        'price_share_of_net': float(cum_price / cum_net) if cum_net != 0 else None,
        'funding_share_of_net': float(cum_funding / cum_net) if cum_net != 0 else None,
        'friction_share_of_gross': float(cum_fric / (cum_price + cum_funding))
            if (cum_price + cum_funding) != 0 else None,
    }

    return {
        'n_days': int(n),
        'cum_net_pct': cum_net,
        'cum_price_pct': cum_price,
        'cum_funding_pct': cum_funding,
        'cum_friction_pct': cum_fric,
        'avg_daily_net_pct': avg_daily_net,
        'avg_daily_price_pct': avg_daily_price,
        'avg_daily_funding_pct': avg_daily_funding,
        'avg_daily_fric_pct': avg_daily_fric,
        'avg_weekly_net_pct': avg_weekly_net,
        'annualized_net_pct': avg_daily_net * 365,
        'max_dd_pct': max_dd,
        'worst_5d_net_pct': worst_5d,
        'sharpe_annualized': sharpe,
        'decomposition': decomposition,
    }


def test_1_walk_forward(price: pd.DataFrame, fund: pd.DataFrame) -> dict:
    folds = GATES['wf_total']
    n = len(price)
    fold_size = n // (folds + 1)
    results = []
    for i in range(folds):
        s = (i + 1) * fold_size
        e = min(s + fold_size, n)
        sub_price = price.iloc[s:e]
        sub_fund = fund.iloc[s:e]
        bt = run_carry(sub_price, sub_fund)
        summ = summarize(bt)
        results.append({'fold': i + 1, **summ})
    pos_count = sum(1 for r in results if r.get('avg_weekly_net_pct', 0) > 0)
    return {'folds': results, 'pos_count': pos_count, 'pass': pos_count >= GATES['wf_min_pos']}


def test_2_bootstrap(bt: pd.DataFrame) -> dict:
    active = bt[(bt['daily_price_pct'] != 0) | (bt['daily_funding_pct'] != 0)]
    nets = active['daily_net_pct'].values
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


def test_3_train_test(price: pd.DataFrame, fund: pd.DataFrame) -> dict:
    n = len(price)
    split = int(n * GATES['tt_split'])
    bt_tr = run_carry(price.iloc[:split], fund.iloc[:split])
    bt_te = run_carry(price.iloc[split:], fund.iloc[split:])
    s_tr = summarize(bt_tr)
    s_te = summarize(bt_te)
    return {
        'train': s_tr, 'test': s_te,
        'pass': (s_tr.get('avg_weekly_net_pct', 0) > 0) and
                (s_te.get('avg_weekly_net_pct', 0) > 0),
    }


def main():
    print('=' * 100)
    print('Path B R3 — Funding-Rate Carry Harvest OOS')
    print('=' * 100)
    print(f'Pre-reg: claudedocs/path_b_r3_funding_carry_prereg.md (4435c76)')
    print(f'Locked: {LOCKED}\n')

    price, fund = load_data()
    print(f'Price pivot:   {price.shape[0]} days × {price.shape[1]} coins')
    print(f'Funding pivot: {fund.shape[0]} days × {fund.shape[1]} coins')
    print(f'Date range: {price.index.min().date()} → {price.index.max().date()}\n')

    print('=== Gate A — Orthogonality (ρ(funding, momentum) < 0.7) ===')
    gA = gate_A_orthogonality(price, fund)
    print(f'  mean ρ:   {gA.get("mean_rho", 0):+.4f}')
    print(f'  median ρ: {gA.get("median_rho", 0):+.4f}')
    print(f'  gate: |ρ| < {gA.get("gate_max_abs_rho", 0)}')
    print(f'  → {"PASS (distinct from R1)" if gA["pass"] else "FAIL — R3 NOT DISTINCT, abort"}\n')

    print('=== Gate B — Vacuity (funding cross-sectional dispersion) ===')
    gB = gate_B_vacuity(fund)
    print(f'  median 7d funding std: {gB.get("median_funding_std_per_8h", 0)*100:.4f}%/8h')
    print(f'  gate: ≥ {gB.get("gate_min", 0)*100:.4f}%/8h')
    print(f'  → {"PASS" if gB["pass"] else "FAIL (vacuous)"}\n')

    if not gA['pass'] or not gB['pass']:
        verdict = 'NOT_DISTINCT' if not gA['pass'] else 'INCONCLUSIVE_VACUOUS'
        print(f'EARLY EXIT: {verdict}')
        out = {'date': datetime.now(timezone.utc).isoformat(),
               'pre_reg_commit': '4435c76',
               'verdict': verdict,
               'gate_A': gA, 'gate_B': gB,
               'locked': LOCKED, 'gates': GATES}
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        p = RESULTS / f'path_b_r3_funding_oos_{ts}.json'
        with open(p, 'w') as fp:
            json.dump(out, fp, indent=2, default=str)
        print(f'Saved: {p}')
        return

    print('=== Full-sample backtest ===')
    bt_full = run_carry(price, fund)
    s_full = summarize(bt_full)
    for k, v in s_full.items():
        if isinstance(v, dict):
            print(f'  {k}:')
            for kk, vv in v.items():
                print(f'    {kk}: {vv}')
        elif isinstance(v, float):
            print(f'  {k}: {v:+.4f}')
        else:
            print(f'  {k}: {v}')
    print()

    print('=== Test 1 — WF 5-fold expanding ===')
    t1 = test_1_walk_forward(price, fund)
    for f_ in t1['folds']:
        wk = f_.get('avg_weekly_net_pct')
        d = f_.get('avg_daily_net_pct')
        n = f_.get('n_days', 0)
        print(f'  fold {f_["fold"]}: n={n:4d}  weekly_net={wk:+.4f}%  daily_net={d:+.4f}%')
    print(f'  → {"PASS" if t1["pass"] else "FAIL"}  ({t1["pos_count"]}/5)\n')

    print('=== Test 2 — Bootstrap 1000 × 30d ===')
    t2 = test_2_bootstrap(bt_full)
    print(f'  pos_rate: {t2.get("pos_rate", 0):.4f}  '
          f'mean: {t2.get("mean_cum_pct", 0):+.4f}%  '
          f'p5: {t2.get("p5", 0):+.4f}%  p95: {t2.get("p95", 0):+.4f}%')
    print(f'  → {"PASS" if t2["pass"] else "FAIL"}\n')

    print('=== Test 3 — Train/Test 60/40 ===')
    t3 = test_3_train_test(price, fund)
    print(f'  train: weekly_net={t3["train"].get("avg_weekly_net_pct", 0):+.4f}%  '
          f'daily_net={t3["train"].get("avg_daily_net_pct", 0):+.4f}%')
    print(f'  test:  weekly_net={t3["test"].get("avg_weekly_net_pct", 0):+.4f}%  '
          f'daily_net={t3["test"].get("avg_daily_net_pct", 0):+.4f}%')
    print(f'  → {"PASS" if t3["pass"] else "FAIL"}\n')

    print('=== Test 4 — Magnitude (≥0.02%/day = 7.3%/yr) ===')
    daily_net = s_full['avg_daily_net_pct']
    t4_pass = daily_net >= GATES['magnitude_min_daily_net_pct']
    print(f'  avg_daily_net: {daily_net:+.4f}%  (gate ≥{GATES["magnitude_min_daily_net_pct"]:.2f}%)')
    print(f'  → {"PASS" if t4_pass else "FAIL"}\n')

    print('=== Test 5 — Tail (worst 5d ≥ -10%) ===')
    worst = s_full['worst_5d_net_pct']
    t5_pass = worst >= -GATES['tail_max_5d_dd_pct']
    print(f'  worst 5d: {worst:+.4f}%  (gate ≥{-GATES["tail_max_5d_dd_pct"]:+.2f}%)')
    print(f'  → {"PASS" if t5_pass else "FAIL"}\n')

    print('=== Decomposition gate (price < 70% of net) ===')
    decomp = s_full['decomposition']
    price_share = decomp.get('price_share_of_net') or 0
    funding_share = decomp.get('funding_share_of_net') or 0
    print(f'  price_share of net: {price_share:+.2%}')
    print(f'  funding_share of net: {funding_share:+.2%}')
    decomp_pass = abs(price_share) < 0.7 if s_full['cum_net_pct'] > 0 else True
    print(f'  → {"PASS (carry dominates)" if decomp_pass else "FAIL (price-momentum in disguise)"}\n')

    all_pass = t1['pass'] and t2['pass'] and t3['pass'] and t4_pass and t5_pass and decomp_pass

    print('=' * 100)
    print('FINAL VERDICT')
    print('=' * 100)
    print(f'  Gate A orth:    PASS')
    print(f'  Gate B vacuity: PASS')
    print(f'  T1 WF:          {"PASS" if t1["pass"] else "FAIL"}  ({t1["pos_count"]}/5)')
    print(f'  T2 BS30d:       {"PASS" if t2["pass"] else "FAIL"}  pos={t2.get("pos_rate", 0):.4f}')
    print(f'  T3 TT60/40:     {"PASS" if t3["pass"] else "FAIL"}')
    print(f'  T4 Magnitude:   {"PASS" if t4_pass else "FAIL"}  daily={daily_net:+.4f}%')
    print(f'  T5 Tail:        {"PASS" if t5_pass else "FAIL"}  5d={worst:+.4f}%')
    print(f'  Decomp carry:   {"PASS" if decomp_pass else "FAIL"}  price={price_share:.0%}')
    print(f'\n  OVERALL: {"ALL PASS — 16th-round candidate breaks ceiling" if all_pass else "FAIL — round 16 hardens ceiling"}')

    out = {
        'date': datetime.now(timezone.utc).isoformat(),
        'pre_reg_commit': '4435c76',
        'verdict': 'PASS' if all_pass else 'FAIL',
        'locked': LOCKED, 'gates': GATES,
        'gate_A': gA, 'gate_B': gB,
        'full_sample': s_full,
        'test_1_wf': t1, 'test_2_bs': t2, 'test_3_tt': t3,
        'test_4_magnitude': {'daily_net_pct': daily_net, 'pass': t4_pass},
        'test_5_tail': {'worst_5d_pct': worst, 'pass': t5_pass},
        'decomposition_gate': {'price_share': price_share, 'pass': decomp_pass},
        'all_pass': bool(all_pass),
    }
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    p = RESULTS / f'path_b_r3_funding_oos_{ts}.json'
    with open(p, 'w') as fp:
        json.dump(out, fp, indent=2, default=str)
    print(f'\nSaved: {p}')


if __name__ == '__main__':
    main()
