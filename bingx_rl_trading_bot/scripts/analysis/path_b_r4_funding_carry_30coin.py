"""Path B R4 — Funding-Rate Carry on Bybit 27-coin universe (3 dropped pre-reg).

Pre-reg: claudedocs/path_b_r4_funding_carry_30coin_prereg.md (commit dac672f)

Reuses R3 mechanism unchanged. Only data source/universe changes.
"""
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / 'scripts' / 'analysis'))
import path_b_r3_funding_carry as r3mod

DATA = ROOT / 'data'
RESULTS = ROOT / 'results'

PRICE_FILE = DATA / 'bybit_daily_prices.parquet'
FUNDING_FILE = DATA / 'bybit_funding_history.parquet'


def load_data():
    price = pd.read_parquet(PRICE_FILE)
    price['date'] = pd.to_datetime(price['date'])
    price_pivot = price.pivot(index='date', columns='symbol', values='close').sort_index()
    coins = sorted(price_pivot.columns.tolist())
    price_pivot = price_pivot.dropna(how='any')

    fund = pd.read_parquet(FUNDING_FILE)
    fund['date'] = pd.to_datetime(fund['date'])
    fund = fund.sort_values(['symbol', 'datetime'])
    daily_fund = fund.groupby(['date', 'symbol'])['funding_rate'].mean().reset_index()
    fund_pivot = daily_fund.pivot(index='date', columns='symbol', values='funding_rate').sort_index()
    fund_pivot = fund_pivot[coins]
    common = price_pivot.index.intersection(fund_pivot.index)
    price_pivot = price_pivot.loc[common].dropna(how='any')
    fund_pivot = fund_pivot.loc[common].reindex(price_pivot.index)
    coins_aligned = price_pivot.columns.intersection(fund_pivot.columns)
    return price_pivot[coins_aligned], fund_pivot[coins_aligned]


def main():
    print('=' * 100)
    print('Path B R4 — Funding-Rate Carry, Bybit 27-coin universe')
    print('=' * 100)
    print('Pre-reg: claudedocs/path_b_r4_funding_carry_30coin_prereg.md (dac672f)')

    drop_meta_path = DATA / 'bybit_universe_drop_list.json'
    if drop_meta_path.exists():
        meta = json.loads(drop_meta_path.read_text())
        print(f"\nDrop list: {meta['dropped']}")
        print(f"Final coins ({meta['final_count']}/{len(meta['universe_target'])}):")
        print(f"  {meta['kept']}")

    # Override R3 module's universe to whatever's in the Bybit data
    price, fund = load_data()
    coins = price.columns.tolist()
    r3mod.LOCKED['universe'] = coins
    print(f'\nLoaded universe: {len(coins)} coins')
    print(f'Date range: {price.index.min().date()} → {price.index.max().date()}\n')

    print('=== Gate A — Orthogonality ===')
    gA = r3mod.gate_A_orthogonality(price, fund)
    print(f'  mean ρ:   {gA.get("mean_rho", 0):+.4f}')
    print(f'  median ρ: {gA.get("median_rho", 0):+.4f}')
    print(f'  → {"PASS" if gA["pass"] else "FAIL — abort"}\n')

    print('=== Gate B — Vacuity ===')
    gB = r3mod.gate_B_vacuity(fund)
    print(f'  median 7d funding std: {gB.get("median_funding_std_per_8h", 0)*100:.4f}%/8h')
    print(f'  R3 result was 0.0038% on 10 coins. Gate floor 0.0500%.')
    print(f'  → {"PASS" if gB["pass"] else "FAIL (vacuous)"}\n')

    if not gA['pass'] or not gB['pass']:
        verdict = 'NOT_DISTINCT' if not gA['pass'] else 'INCONCLUSIVE_VACUOUS'
        print(f'EARLY EXIT: {verdict}')
        out = {'date': datetime.now(timezone.utc).isoformat(),
               'pre_reg_commit': 'dac672f',
               'verdict': verdict,
               'universe': coins,
               'gate_A': gA, 'gate_B': gB,
               'locked': r3mod.LOCKED, 'gates': r3mod.GATES}
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        p = RESULTS / f'path_b_r4_funding_oos_{ts}.json'
        with open(p, 'w') as fp:
            json.dump(out, fp, indent=2, default=str)
        print(f'Saved: {p}')
        return

    print('=== Full-sample backtest ===')
    bt_full = r3mod.run_carry(price, fund)
    s_full = r3mod.summarize(bt_full)
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

    print('=== Test 1 — WF 5-fold ===')
    t1 = r3mod.test_1_walk_forward(price, fund)
    for f_ in t1['folds']:
        wk = f_.get('avg_weekly_net_pct', 0)
        d = f_.get('avg_daily_net_pct', 0)
        print(f'  fold {f_["fold"]}: weekly={wk:+.4f}%  daily={d:+.4f}%')
    print(f'  → {"PASS" if t1["pass"] else "FAIL"}  ({t1["pos_count"]}/5)\n')

    print('=== Test 2 — Bootstrap 1000 × 30d ===')
    t2 = r3mod.test_2_bootstrap(bt_full)
    print(f'  pos_rate: {t2.get("pos_rate", 0):.4f}  '
          f'mean: {t2.get("mean_cum_pct", 0):+.4f}%  '
          f'p5: {t2.get("p5", 0):+.4f}%  p95: {t2.get("p95", 0):+.4f}%')
    print(f'  → {"PASS" if t2["pass"] else "FAIL"}\n')

    print('=== Test 3 — Train/Test 60/40 ===')
    t3 = r3mod.test_3_train_test(price, fund)
    print(f'  train: weekly={t3["train"].get("avg_weekly_net_pct", 0):+.4f}%')
    print(f'  test:  weekly={t3["test"].get("avg_weekly_net_pct", 0):+.4f}%')
    print(f'  → {"PASS" if t3["pass"] else "FAIL"}\n')

    daily_net = s_full['avg_daily_net_pct']
    t4_pass = daily_net >= r3mod.GATES['magnitude_min_daily_net_pct']
    print(f'=== T4 Magnitude (≥{r3mod.GATES["magnitude_min_daily_net_pct"]}%/day) ===')
    print(f'  daily_net: {daily_net:+.4f}%  → {"PASS" if t4_pass else "FAIL"}\n')

    worst = s_full['worst_5d_net_pct']
    t5_pass = worst >= -r3mod.GATES['tail_max_5d_dd_pct']
    print(f'=== T5 Tail (worst 5d ≥ -{r3mod.GATES["tail_max_5d_dd_pct"]}%) ===')
    print(f'  worst 5d: {worst:+.4f}%  → {"PASS" if t5_pass else "FAIL"}\n')

    decomp = s_full['decomposition']
    price_share = decomp.get('price_share_of_net') or 0
    funding_share = decomp.get('funding_share_of_net') or 0
    print(f'=== Decomposition ===')
    print(f'  price_share:   {price_share:+.2%}')
    print(f'  funding_share: {funding_share:+.2%}')
    decomp_pass = (abs(price_share) < 0.7) if s_full['cum_net_pct'] > 0 else True
    print(f'  → {"PASS" if decomp_pass else "FAIL (price-momentum in disguise)"}\n')

    all_pass = t1['pass'] and t2['pass'] and t3['pass'] and t4_pass and t5_pass and decomp_pass

    print('=' * 100)
    print('FINAL VERDICT')
    print('=' * 100)
    print(f'  Universe:       27 coins (vs R3 10 coins)')
    print(f'  Gate A orth:    PASS')
    print(f'  Gate B vacuity: PASS  ({gB.get("median_funding_std_per_8h", 0)*100:.4f}%/8h)')
    print(f'  T1 WF:          {"PASS" if t1["pass"] else "FAIL"}  ({t1["pos_count"]}/5)')
    print(f'  T2 BS30d:       {"PASS" if t2["pass"] else "FAIL"}  pos={t2.get("pos_rate", 0):.4f}')
    print(f'  T3 TT60/40:     {"PASS" if t3["pass"] else "FAIL"}')
    print(f'  T4 Magnitude:   {"PASS" if t4_pass else "FAIL"}  daily={daily_net:+.4f}%')
    print(f'  T5 Tail:        {"PASS" if t5_pass else "FAIL"}  5d={worst:+.4f}%')
    print(f'  Decomp:         {"PASS" if decomp_pass else "FAIL"}  price_share={price_share:.0%}')
    print(f'\n  OVERALL: {"ALL PASS — first ceiling break in 16 rounds" if all_pass else "FAIL — 16th data point hardens ceiling"}')

    out = {
        'date': datetime.now(timezone.utc).isoformat(),
        'pre_reg_commit': 'dac672f',
        'universe': coins,
        'verdict': 'PASS' if all_pass else 'FAIL',
        'locked': r3mod.LOCKED, 'gates': r3mod.GATES,
        'gate_A': gA, 'gate_B': gB,
        'full_sample': s_full,
        'test_1_wf': t1, 'test_2_bs': t2, 'test_3_tt': t3,
        'test_4_magnitude': {'daily_net_pct': daily_net, 'pass': t4_pass},
        'test_5_tail': {'worst_5d_pct': worst, 'pass': t5_pass},
        'decomposition_gate': {'price_share': price_share, 'pass': decomp_pass},
        'all_pass': bool(all_pass),
    }
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    p = RESULTS / f'path_b_r4_funding_oos_{ts}.json'
    with open(p, 'w') as fp:
        json.dump(out, fp, indent=2, default=str)
    print(f'\nSaved: {p}')


if __name__ == '__main__':
    main()
