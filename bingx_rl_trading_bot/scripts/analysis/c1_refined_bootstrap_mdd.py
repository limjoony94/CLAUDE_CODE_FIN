"""
P2: Bootstrap MDD Verification for Refined Variants
====================================================

Advisor의 요구사항: Refined 변종의 MDD가 표본 특성인지 vs 일반 특성인지 확인.

방법:
  1. 각 변종의 trade 시퀀스를 원본 순서로 추출
  2. 1000회 resampling (with replacement, 시퀀스 교란)
  3. 각 resample마다 MDD 계산 → 95% CI 산출
  4. 관측 MDD가 CI 내부에 있는지, 꼬리가 얼마나 두꺼운지 확인

핵심 비교:
  - BASELINE MDD 5.38%
  - REFINED pure CD=0 MDD 8.21%
  - A variant CD=2 MDD 10.42% ← advisor 의심 포인트
  - C variant MDD 6.91%
  - D variant MDD 6.90%

질문: A의 MDD 10.42%가 통계적으로 유의미하게 BASELINE 5.38%보다 나쁜가?
     C/D의 MDD 7% 근처가 BASELINE과 구별 가능한가?
"""
import sys, os, json, random
from datetime import datetime
from pathlib import Path
from statistics import mean, median, stdev

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
from scripts.analysis.c1_refined_validation import (
    BASE_CFG, entry_baseline, entry_refined,
    run_bt, summarize, precompute,
)
from scripts.analysis.c1_refined_variants import (
    entry_refined_A, entry_refined_C, entry_refined_D,
)


def compute_mdd_from_trades(trades):
    """Additive MDD from trade sequence."""
    eq, peak, mdd = 0.0, 0.0, 0.0
    for t in trades:
        eq += t['pnl_pct']
        peak = max(peak, eq)
        mdd = max(mdd, peak - eq)
    return mdd, eq


def _stationary_bootstrap_indices(n, mean_block_len, rng):
    """Politis & Romano (1994) stationary bootstrap — geometric block lengths.

    Preserves short-range time dependence (drawdown clustering).
    p = 1/mean_block_len is the probability of starting a new block.
    """
    p = 1.0 / mean_block_len
    idx = [0] * n
    cur = rng.randrange(n)
    for i in range(n):
        idx[i] = cur
        if rng.random() < p:
            cur = rng.randrange(n)   # restart
        else:
            cur = (cur + 1) % n       # continue block
    return idx


def bootstrap_mdd(trades, n_boot=1000, seed=42, mean_block_len=20):
    """Stationary block bootstrap (preserves path dependence)."""
    rng = random.Random(seed)
    n = len(trades)
    if n == 0:
        return {'obs_mdd': 0, 'mean': 0, 'median': 0,
                'ci_low': 0, 'ci_high': 0, 'p99': 0}
    obs_mdd, _ = compute_mdd_from_trades(trades)
    mdds = []
    for _ in range(n_boot):
        idx = _stationary_bootstrap_indices(n, mean_block_len, rng)
        sample = [trades[j] for j in idx]
        m, _ = compute_mdd_from_trades(sample)
        mdds.append(m)
    mdds.sort()
    return {
        'obs_mdd': round(obs_mdd, 3),
        'mean': round(mean(mdds), 3),
        'median': round(median(mdds), 3),
        'stdev': round(stdev(mdds), 3),
        'ci_low':  round(mdds[int(0.025 * n_boot)], 3),
        'ci_high': round(mdds[int(0.975 * n_boot)], 3),
        'p99':     round(mdds[int(0.99  * n_boot)], 3),
        'p01':     round(mdds[int(0.01  * n_boot)], 3),
        'method':  f'stationary_bootstrap(mean_block={mean_block_len})',
    }


def main():
    csv = ROOT / 'data' / 'btc_5m_270days_reclassified.csv'
    df = pd.read_csv(csv, parse_dates=['timestamp']).sort_values('timestamp').reset_index(drop=True)
    df15 = df.set_index('timestamp').resample('15min').agg({
        'open': 'first', 'high': 'max', 'low': 'min',
        'close': 'last', 'volume': 'sum',
    }).dropna(subset=['open']).reset_index()
    print(f"Bars: {len(df15)}")

    CD0 = {**BASE_CFG, 'min_bars_between': 0}
    CD2 = {**BASE_CFG, 'min_bars_between': 2}

    variants = [
        ('BASELINE',            entry_baseline,  CD2),
        ('REFINED pure (CD=0)', entry_refined,   CD0),
        ('A: CD=2',             entry_refined_A, CD2),
        ('C: Channel soft',     entry_refined_C, CD0),
        ('D: Body 50%',         entry_refined_D, CD0),
    ]

    print(f"\n{'='*105}")
    print(f"  P2: Bootstrap MDD verification (1000 resamples)")
    print(f"{'='*105}")
    print(f"{'Variant':25s} {'Trades':>7s} {'obsMDD':>7s} {'meanB':>7s} "
          f"{'medB':>7s} {'σ':>6s} {'CI95Low':>8s} {'CI95Hi':>8s} {'P99':>7s}")
    print('-'*105)

    results = []
    baseline_mdds = None
    for name, fn, cfg in variants:
        pc = precompute(df15, cfg)
        trades = run_bt(df15, cfg, fn, 50, len(df15) - 1, **pc)
        boot = bootstrap_mdd(trades, n_boot=1000, seed=42)
        if name == 'BASELINE':
            baseline_mdds = boot  # for comparison
        print(f"{name:25s} {len(trades):>7} "
              f"{boot['obs_mdd']:>7.2f} {boot['mean']:>7.2f} "
              f"{boot['median']:>7.2f} {boot['stdev']:>6.2f} "
              f"{boot['ci_low']:>8.2f} {boot['ci_high']:>8.2f} "
              f"{boot['p99']:>7.2f}")
        results.append({'name': name, 'n_trades': len(trades), 'boot': boot})

    # Paired comparison: does variant's CI overlap baseline's CI?
    print(f"\n{'='*105}")
    print(f"  Baseline CI95 [{baseline_mdds['ci_low']:.2f}, {baseline_mdds['ci_high']:.2f}]")
    print(f"  변종 MDD 분포가 Baseline 분포와 구별되는지")
    print(f"{'='*105}")
    print(f"{'Variant':25s} {'CI Overlap':>14s} {'obs vs BASE_CI_HI':>22s} {'Verdict':>20s}")
    print('-'*105)
    for r in results:
        if r['name'] == 'BASELINE':
            continue
        b = r['boot']
        overlaps = not (b['ci_high'] < baseline_mdds['ci_low']
                        or b['ci_low'] > baseline_mdds['ci_high'])
        worse_than_base_hi = b['obs_mdd'] > baseline_mdds['ci_high']
        if b['obs_mdd'] > baseline_mdds['ci_high']:
            verdict = "⚠ WORSE than BASE CI"
        elif b['ci_high'] < baseline_mdds['ci_low']:
            verdict = "✓ BETTER than BASE"
        else:
            verdict = "~ INDISTINGUISHABLE"
        print(f"{r['name']:25s} {'YES' if overlaps else 'NO':>14s} "
              f"{b['obs_mdd']:>+8.2f} vs {baseline_mdds['ci_high']:>+6.2f}  "
              f"{verdict:>20s}")

    # Save
    out = {
        'date_run': datetime.now().isoformat(),
        'baseline_ci': baseline_mdds,
        'results': results,
    }
    out_path = ROOT / 'results' / 'c1_refined_bootstrap_mdd.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved: {out_path}")


if __name__ == '__main__':
    main()
