"""DeFi-Track R1 — L2 Yield Rotation Top-3 OOS verification.

Pre-reg: claudedocs/defi_track_r1_yield_rotation_prereg.md (commit 266d808)
LOCKED: top-3 trailing 30d APY median, L2-only universe, monthly rebalance,
        0.4%/swap friction, equal-weight, $500 per position on $1,500 capital.

5 gates:
  T1. WF 5-fold expanding   → ≥3/5 folds with positive cumulative net
  T2. Bootstrap 1000 × 90d  → pos_rate ≥ 50%
  T3. Train/Test 60/40      → both train and test cumulative net > 0
  T4. Magnitude (NEW)       → full-sample net APY ≥ 7.3%/yr (=0.02%/day)
  T5. Tail-risk (NEW)       → worst 5-day net return ≥ -10%

Vacuity gate: median monthly eligible pool count ≥ 5.

Result: results/defi_track_r1_oos_{ts}.json
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

PANEL_FILE = DATA / 'defi_yields_panel.parquet'
COHORT_FILE = DATA / 'defi_yields_cohort.parquet'

LOCKED = {
    'universe_chains': ['Arbitrum', 'OP Mainnet', 'Base', 'Polygon'],
    'lookback_days': 30,
    'top_n': 3,
    'rebalance_frequency_days': 30,
    'friction_per_swap_pct': 0.4,
    'min_pool_history_days': 30,
    'capital_usd': 1500,
    'position_size_usd': 500,
}

GATES = {
    'wf_min_positive_folds': 3,
    'wf_total_folds': 5,
    'bs_n_iter': 1000,
    'bs_window_days': 90,
    'bs_min_pos_rate': 0.50,
    'tt_split': 0.60,
    'magnitude_min_net_apy_pct': 7.3,
    'tail_max_5d_dd_pct': 10.0,
}

VAC_MIN_MEDIAN_ELIGIBLE = 5
VAC_MIN_MEAN_HISTORY = 8


def load_panel():
    panel = pd.read_parquet(PANEL_FILE)
    cohort = pd.read_parquet(COHORT_FILE)
    panel['date'] = pd.to_datetime(panel['date'])
    panel = panel.sort_values(['pool_id', 'date']).reset_index(drop=True)

    chain_map = cohort.set_index('pool_id')['chain']
    proj_map = cohort.set_index('pool_id')['project']
    panel['chain'] = panel['pool_id'].map(chain_map)
    panel['project'] = panel['pool_id'].map(proj_map)
    return panel


def filter_universe(panel: pd.DataFrame) -> pd.DataFrame:
    f = panel[panel['chain'].isin(LOCKED['universe_chains'])].copy()
    return f.reset_index(drop=True)


def vacuity_check(panel: pd.DataFrame) -> dict:
    """median monthly eligible pool count ≥ VAC_MIN_MEDIAN_ELIGIBLE."""
    p = panel.copy()
    p['ym'] = p['date'].dt.to_period('M').astype(str)
    monthly_counts = p.groupby('ym')['pool_id'].nunique()
    eligible_for_strategy = []
    for ym, n in monthly_counts.items():
        if n >= LOCKED['top_n']:
            eligible_for_strategy.append(n)
    if not eligible_for_strategy:
        return {'pass': False, 'median': 0, 'mean': 0, 'reason': 'no months with ≥top_n pools'}
    median_eligible = float(np.median(eligible_for_strategy))
    mean_eligible = float(np.mean(eligible_for_strategy))
    return {
        'pass': (median_eligible >= VAC_MIN_MEDIAN_ELIGIBLE) and (mean_eligible >= VAC_MIN_MEAN_HISTORY),
        'median_eligible_per_month': median_eligible,
        'mean_eligible_per_month': mean_eligible,
        'gate_median': VAC_MIN_MEDIAN_ELIGIBLE,
        'gate_mean': VAC_MIN_MEAN_HISTORY,
        'months_with_top_n_pools': len(eligible_for_strategy),
        'total_months': len(monthly_counts),
    }


def build_apy_pivot(panel: pd.DataFrame) -> pd.DataFrame:
    """date × pool_id matrix of APY (% annualized)."""
    pv = panel.pivot_table(index='date', columns='pool_id', values='apy', aggfunc='last')
    pv = pv.sort_index()
    full_idx = pd.date_range(pv.index.min(), pv.index.max(), freq='D')
    pv = pv.reindex(full_idx)
    return pv


def run_rotation(apy_pivot: pd.DataFrame) -> pd.DataFrame:
    """
    Run monthly-rebalanced top-3 rotation.

    Returns DataFrame with columns:
      date, holdings (set), gross_daily_pct, friction_daily_pct, net_daily_pct
    Friction is charged on the rebalance day only (lump-sum), but reported as
    that day's friction_daily_pct (not amortized).
    """
    look = LOCKED['lookback_days']
    rebal = LOCKED['rebalance_frequency_days']
    top_n = LOCKED['top_n']
    fric_per_swap = LOCKED['friction_per_swap_pct'] / 100.0

    dates = apy_pivot.index
    rows = []
    holdings = set()
    rebal_dates = list(dates[look::rebal])

    for i, date in enumerate(dates):
        if i < look:
            rows.append({'date': date, 'holdings': frozenset(), 'gross_daily_pct': 0.0,
                         'friction_daily_pct': 0.0, 'net_daily_pct': 0.0,
                         'is_rebal': False, 'n_swaps': 0})
            continue

        is_rebal = (date in rebal_dates)
        if is_rebal:
            window = apy_pivot.iloc[i - look:i]
            trailing_med = window.median(axis=0)
            valid_count = window.notna().sum(axis=0)
            eligible = trailing_med[valid_count >= look].dropna()
            if len(eligible) < top_n:
                new_holdings = holdings
            else:
                new_holdings = frozenset(eligible.nlargest(top_n).index)

            entries = new_holdings - holdings
            exits = holdings - new_holdings
            n_swaps = len(entries) + len(exits)
            holdings = new_holdings

            friction_pct = (n_swaps * fric_per_swap * LOCKED['position_size_usd']) / LOCKED['capital_usd'] * 100.0
        else:
            n_swaps = 0
            friction_pct = 0.0

        if holdings:
            apys_today = apy_pivot.loc[date, list(holdings)].dropna()
            if len(apys_today) > 0:
                avg_apy = float(apys_today.mean())
            else:
                avg_apy = 0.0
            position_weight = len(holdings) / top_n
            daily_factor = (1 + avg_apy / 100.0) ** (1 / 365.0) - 1
            gross_daily_pct = daily_factor * 100.0 * position_weight
        else:
            gross_daily_pct = 0.0

        net_daily_pct = gross_daily_pct - friction_pct
        rows.append({
            'date': date,
            'holdings': frozenset(holdings),
            'gross_daily_pct': gross_daily_pct,
            'friction_daily_pct': friction_pct,
            'net_daily_pct': net_daily_pct,
            'is_rebal': is_rebal,
            'n_swaps': n_swaps,
        })
    return pd.DataFrame(rows)


def summarize(bt: pd.DataFrame) -> dict:
    if bt.empty:
        return {'n_days': 0}
    active = bt[bt['gross_daily_pct'] != 0].copy()
    if active.empty:
        return {'n_days': 0}
    n = len(active)
    cum_gross = float((1 + active['gross_daily_pct'] / 100).prod() - 1) * 100
    cum_net = float((1 + active['net_daily_pct'] / 100).prod() - 1) * 100
    avg_daily_net = float(active['net_daily_pct'].mean())
    avg_daily_gross = float(active['gross_daily_pct'].mean())
    avg_daily_fric = float(active['friction_daily_pct'].mean())
    annualized_net_apy = avg_daily_net * 365
    annualized_gross_apy = avg_daily_gross * 365

    nav = (1 + active['net_daily_pct'].values / 100).cumprod()
    peak = np.maximum.accumulate(nav)
    dd = (peak - nav) / peak
    max_dd = float(dd.max()) * 100

    rolling_5d = pd.Series(active['net_daily_pct'].values).rolling(5).apply(
        lambda x: (1 + x / 100).prod() - 1
    ) * 100
    worst_5d = float(rolling_5d.min())

    daily_std = float(active['net_daily_pct'].std())
    sharpe = (avg_daily_net / daily_std * (365 ** 0.5)) if daily_std > 0 else 0.0

    return {
        'n_days': int(n),
        'cum_gross_pct': cum_gross,
        'cum_net_pct': cum_net,
        'avg_daily_net_pct': avg_daily_net,
        'avg_daily_gross_pct': avg_daily_gross,
        'avg_daily_fric_pct': avg_daily_fric,
        'annualized_net_apy_pct': annualized_net_apy,
        'annualized_gross_apy_pct': annualized_gross_apy,
        'max_dd_pct': max_dd,
        'worst_5d_net_pct': worst_5d,
        'sharpe_annualized': sharpe,
        'n_rebalances': int(active['is_rebal'].sum()),
        'avg_swaps_per_rebal': float(active.loc[active['is_rebal'], 'n_swaps'].mean()) if active['is_rebal'].any() else 0.0,
    }


def test_1_walk_forward(apy_pivot: pd.DataFrame) -> dict:
    folds = GATES['wf_total_folds']
    n_dates = len(apy_pivot)
    fold_size = n_dates // (folds + 1)
    results = []
    for fold_i in range(folds):
        te_s = (fold_i + 1) * fold_size
        te_e = min(te_s + fold_size, n_dates)
        sub = apy_pivot.iloc[te_s:te_e]
        bt = run_rotation(sub)
        s = summarize(bt)
        results.append({'fold': fold_i + 1, **s})
    pos_count = sum(1 for r in results if r.get('cum_net_pct', 0) > 0)
    return {
        'folds': results,
        'pos_count': pos_count,
        'pass': pos_count >= GATES['wf_min_positive_folds'],
    }


def test_2_bootstrap(bt: pd.DataFrame) -> dict:
    active = bt[bt['gross_daily_pct'] != 0].copy()
    nets = active['net_daily_pct'].values
    n = len(nets)
    win = GATES['bs_window_days']
    if n <= win:
        return {'pass': False, 'reason': f'panel too short for {win}d windows'}
    random.seed(42)
    starts = random.sample(range(n - win), min(GATES['bs_n_iter'], n - win))
    cums = []
    for s in starts:
        slice_ = nets[s:s + win]
        c = float((1 + slice_ / 100).prod() - 1) * 100
        cums.append(c)
    arr = np.array(cums)
    pos_rate = float((arr > 0).mean())
    return {
        'n_iter': len(arr),
        'window_days': win,
        'mean_cum_pct': float(arr.mean()),
        'pos_rate': pos_rate,
        'p5_pct': float(np.percentile(arr, 5)),
        'p95_pct': float(np.percentile(arr, 95)),
        'pass': pos_rate >= GATES['bs_min_pos_rate'],
    }


def test_3_train_test(apy_pivot: pd.DataFrame) -> dict:
    n = len(apy_pivot)
    split = int(n * GATES['tt_split'])
    bt_tr = run_rotation(apy_pivot.iloc[:split])
    bt_te = run_rotation(apy_pivot.iloc[split:])
    s_tr = summarize(bt_tr)
    s_te = summarize(bt_te)
    return {
        'train': s_tr,
        'test': s_te,
        'pass': (s_tr.get('cum_net_pct', 0) > 0) and (s_te.get('cum_net_pct', 0) > 0),
    }


def test_4_magnitude(full_summary: dict) -> dict:
    apy = full_summary.get('annualized_net_apy_pct', 0)
    return {
        'annualized_net_apy_pct': apy,
        'gate': GATES['magnitude_min_net_apy_pct'],
        'pass': apy >= GATES['magnitude_min_net_apy_pct'],
    }


def test_5_tail(full_summary: dict) -> dict:
    worst = full_summary.get('worst_5d_net_pct', -100)
    return {
        'worst_5d_net_pct': worst,
        'gate': -GATES['tail_max_5d_dd_pct'],
        'pass': worst >= -GATES['tail_max_5d_dd_pct'],
    }


def main():
    print('=' * 100)
    print('DeFi-Track R1 — L2 Yield Rotation Top-3 OOS Verification')
    print('=' * 100)
    print(f'Pre-reg: claudedocs/defi_track_r1_yield_rotation_prereg.md (commit 266d808)')
    print(f'Locked params: {LOCKED}')
    print(f'Gates: {GATES}\n')

    panel = load_panel()
    f = filter_universe(panel)
    print(f'Full panel: {len(panel):,} rows × {panel.pool_id.nunique()} pools')
    print(f'L2 universe: {len(f):,} rows × {f.pool_id.nunique()} pools across '
          f'{f.chain.nunique()} chains')
    print(f'  pools per chain: {f.groupby("chain")["pool_id"].nunique().to_dict()}')
    print(f'  date range: {f.date.min().date()} → {f.date.max().date()}\n')

    print('=== Vacuity gate ===')
    vac = vacuity_check(f)
    print(f'  median eligible/mo: {vac.get("median_eligible_per_month", 0):.1f}  '
          f'(gate ≥{vac.get("gate_median", 0)})')
    print(f'  mean eligible/mo:   {vac.get("mean_eligible_per_month", 0):.1f}  '
          f'(gate ≥{vac.get("gate_mean", 0)})')
    print(f'  → {"PASS" if vac["pass"] else "FAIL (vacuous)"}\n')
    if not vac['pass']:
        out = {'date': datetime.now(timezone.utc).isoformat(),
               'pre_reg_commit': '266d808',
               'verdict': 'INCONCLUSIVE_VACUOUS',
               'vacuity': vac, 'locked': LOCKED, 'gates': GATES}
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        p = RESULTS / f'defi_track_r1_oos_{ts}.json'
        with open(p, 'w') as fp:
            json.dump(out, fp, indent=2, default=str)
        print(f'Saved: {p}')
        return

    apy_pivot = build_apy_pivot(f)
    print(f'APY pivot: {apy_pivot.shape[0]} days × {apy_pivot.shape[1]} pools\n')

    print('=== Full-sample backtest ===')
    bt_full = run_rotation(apy_pivot)
    s_full = summarize(bt_full)
    for k, v in s_full.items():
        if isinstance(v, float):
            print(f'  {k}: {v:+.4f}')
        else:
            print(f'  {k}: {v}')
    print()

    print('=== Test 1 — WF 5-fold expanding ===')
    t1 = test_1_walk_forward(apy_pivot)
    for f_ in t1['folds']:
        cum = f_.get('cum_net_pct')
        n = f_.get('n_days', 0)
        cum_str = f'{cum:+.2f}%' if cum is not None else 'NA'
        print(f'  fold {f_["fold"]}: n={n:4d}  cum_net={cum_str}')
    print(f'  positive folds: {t1["pos_count"]}/{GATES["wf_total_folds"]} → '
          f'{"PASS" if t1["pass"] else "FAIL"}\n')

    print('=== Test 2 — Bootstrap 1000 × 90d ===')
    t2 = test_2_bootstrap(bt_full)
    print(f'  mean cum: {t2.get("mean_cum_pct", 0):+.4f}%  '
          f'pos_rate: {t2.get("pos_rate", 0):.4f}  '
          f'p5: {t2.get("p5_pct", 0):+.4f}%  p95: {t2.get("p95_pct", 0):+.4f}%')
    print(f'  → {"PASS" if t2["pass"] else "FAIL"}\n')

    print('=== Test 3 — Train/Test 60/40 ===')
    t3 = test_3_train_test(apy_pivot)
    print(f'  train: n_days={t3["train"].get("n_days", 0)}  '
          f'cum_net={t3["train"].get("cum_net_pct", 0):+.4f}%  '
          f'apy={t3["train"].get("annualized_net_apy_pct", 0):+.2f}%/yr')
    print(f'  test:  n_days={t3["test"].get("n_days", 0)}  '
          f'cum_net={t3["test"].get("cum_net_pct", 0):+.4f}%  '
          f'apy={t3["test"].get("annualized_net_apy_pct", 0):+.2f}%/yr')
    print(f'  → {"PASS" if t3["pass"] else "FAIL"}\n')

    print('=== Test 4 — Magnitude (full-sample net APY ≥ 7.3%/yr) ===')
    t4 = test_4_magnitude(s_full)
    print(f'  full sample net APY: {t4["annualized_net_apy_pct"]:+.4f}%/yr  '
          f'(gate ≥{t4["gate"]:+.2f}%)  → {"PASS" if t4["pass"] else "FAIL"}\n')

    print('=== Test 5 — Tail-risk (worst 5d ≥ -10%) ===')
    t5 = test_5_tail(s_full)
    print(f'  worst 5d net: {t5["worst_5d_net_pct"]:+.4f}%  '
          f'(gate ≥{t5["gate"]:+.2f}%)  → {"PASS" if t5["pass"] else "FAIL"}\n')

    all_pass = t1['pass'] and t2['pass'] and t3['pass'] and t4['pass'] and t5['pass']
    print('=' * 100)
    print('FINAL VERDICT')
    print('=' * 100)
    print(f'  Vacuity:     PASS')
    print(f'  T1 WF:       {"PASS" if t1["pass"] else "FAIL"}  ({t1["pos_count"]}/{GATES["wf_total_folds"]})')
    print(f'  T2 BS90d:    {"PASS" if t2["pass"] else "FAIL"}  pos_rate={t2.get("pos_rate", 0):.4f}')
    print(f'  T3 TT60/40:  {"PASS" if t3["pass"] else "FAIL"}')
    print(f'  T4 Magnitude:{"PASS" if t4["pass"] else "FAIL"}  apy={t4["annualized_net_apy_pct"]:+.2f}%/yr')
    print(f'  T5 Tail:     {"PASS" if t5["pass"] else "FAIL"}  worst5d={t5["worst_5d_net_pct"]:+.2f}%')
    print(f'\n  OVERALL: {"ALL 5 PASS — paper-deploy candidate" if all_pass else "FAIL — escalate to advisor"}')

    out = {
        'date': datetime.now(timezone.utc).isoformat(),
        'pre_reg_commit': '266d808',
        'verdict': 'PASS' if all_pass else 'FAIL',
        'locked': LOCKED,
        'gates': GATES,
        'vacuity': vac,
        'full_sample': s_full,
        'test_1_wf': t1,
        'test_2_bs': t2,
        'test_3_tt': t3,
        'test_4_magnitude': t4,
        'test_5_tail': t5,
        'all_pass': bool(all_pass),
    }
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    p = RESULTS / f'defi_track_r1_oos_{ts}.json'
    with open(p, 'w') as fp:
        json.dump(out, fp, indent=2, default=str)
    print(f'\nSaved: {p}')


if __name__ == '__main__':
    main()
