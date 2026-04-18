"""
Data-Mining Defense Check for Refined Variants
===============================================

Advisor의 비판에 대한 정면 검증:

  P1: 3-way split 단일 분할점(60/20/20)이 Test 구간 운 의존 가능성
      → 4개 다른 분할점(40/20/40, 50/20/30, 60/20/20, 70/15/15)에서
        C와 D 변종 전부 positive Test 유지하는지 확인

  P3: REFINED pure (CD=0) 단독 데이터포인트로 retrigger 루프 위험 판단 어려움
      → CD=1 중간값 테스트하여 CD 민감도 곡선 보강
        (CD=0 pure → CD=1 mid → CD=2 variant A)

각 분할점별로 Train/Val/Test 모두 양수여야 합격.
Test 구간에서 하나라도 실패하면 해당 변종은 data-mining 의심.
"""
import sys, os, json, math
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))

import pandas as pd
from scripts.analysis.c1_refined_validation import (
    BASE_CFG, entry_baseline, entry_refined,
    run_bt, summarize, precompute,
)
from scripts.analysis.c1_refined_variants import (
    entry_refined_C, entry_refined_D,
)


def split_three_way(df15, cfg, entry_fn, pc, train_frac, val_frac):
    """Arbitrary 3-way split."""
    n = len(df15)
    warmup = 50
    eff = n - warmup
    t1 = warmup + int(eff * train_frac)
    t2 = warmup + int(eff * (train_frac + val_frac))
    train_t = run_bt(df15, cfg, entry_fn, warmup, t1, **pc)
    val_t = run_bt(df15, cfg, entry_fn, t1 + 1, t2, **pc)
    test_t = run_bt(df15, cfg, entry_fn, t2 + 1, n - 1, **pc)
    return {
        'train': summarize(train_t),
        'val': summarize(val_t),
        'test': summarize(test_t),
        'split': f"{int(train_frac*100)}/{int(val_frac*100)}/{int((1-train_frac-val_frac)*100)}",
    }


def main():
    csv = ROOT / 'data' / 'btc_5m_270days_reclassified.csv'
    df = pd.read_csv(csv, parse_dates=['timestamp']).sort_values('timestamp').reset_index(drop=True)
    df15 = df.set_index('timestamp').resample('15min').agg({
        'open': 'first', 'high': 'max', 'low': 'min',
        'close': 'last', 'volume': 'sum',
    }).dropna(subset=['open']).reset_index()
    print(f"Bars: {len(df15)} | {df15['timestamp'].iloc[0]} ~ {df15['timestamp'].iloc[-1]}")

    CD0 = {**BASE_CFG, 'min_bars_between': 0}
    CD1 = {**BASE_CFG, 'min_bars_between': 1}
    CD2 = {**BASE_CFG, 'min_bars_between': 2}

    # Split points: (train_frac, val_frac) — test = 1 - train - val
    splits = [
        (0.40, 0.20),  # 40/20/40
        (0.50, 0.20),  # 50/20/30
        (0.60, 0.20),  # 60/20/20 (original)
        (0.70, 0.15),  # 70/15/15
    ]

    # P1: C and D variants under 4 split points
    print(f"\n{'='*90}")
    print(f"  P1: 3-way split sensitivity (C & D variants)")
    print(f"{'='*90}")
    print(f"{'Variant':25s} {'Split':>10s} {'Train':>10s} {'Val':>10s} {'Test':>10s} {'ALL+':>6s}")
    print('-'*90)

    p1_results = []
    variants_p1 = [
        ('BASELINE',        entry_baseline,  CD2),
        ('C: Channel soft', entry_refined_C, CD0),
        ('D: Body 50%',     entry_refined_D, CD0),
    ]
    for name, fn, cfg in variants_p1:
        pc = precompute(df15, cfg)
        for tf, vf in splits:
            r = split_three_way(df15, cfg, fn, pc, tf, vf)
            all_pos = all(r[k]['PnL'] > 0 for k in ('train', 'val', 'test'))
            flag = 'Y' if all_pos else 'N'
            print(f"{name:25s} {r['split']:>10s} "
                  f"{r['train']['PnL']:>+10.2f} "
                  f"{r['val']['PnL']:>+10.2f} "
                  f"{r['test']['PnL']:>+10.2f} "
                  f"{flag:>6s}")
            p1_results.append({'variant': name, **r, 'all_pos': all_pos})
        print()

    # P3: REFINED CD=0,1,2 sensitivity
    print(f"\n{'='*90}")
    print(f"  P3: REFINED pure (Body+ATR SL) — CD sensitivity curve")
    print(f"{'='*90}")
    print(f"{'Config':20s} {'Trades':>8s} {'WR%':>6s} {'FullPnL':>10s} "
          f"{'MDD':>6s} {'Test':>10s} {'Calmar':>8s}")
    print('-'*90)

    p3_results = []
    cd_configs = [
        ('REFINED CD=0', CD0),
        ('REFINED CD=1', CD1),
        ('REFINED CD=2', CD2),
    ]
    for name, cfg in cd_configs:
        pc = precompute(df15, cfg)
        # Full backtest
        full = run_bt(df15, cfg, entry_refined, 50, len(df15) - 1, **pc)
        fs = summarize(full)
        # Original 60/20/20 test
        r60 = split_three_way(df15, cfg, entry_refined, pc, 0.60, 0.20)
        calmar = fs['PnL'] / fs['MDD'] if fs['MDD'] > 0 else 0
        print(f"{name:20s} {fs['trades']:>8} {fs['WR']:>6} "
              f"{fs['PnL']:>+10.2f} {fs['MDD']:>6.2f} "
              f"{r60['test']['PnL']:>+10.2f} {calmar:>8.2f}")
        p3_results.append({'name': name, 'full': fs, '60_20_20': r60,
                           'calmar': round(calmar, 2)})

    # Summary: C/D variants pass rate
    print(f"\n{'='*90}")
    print(f"  SUMMARY")
    print(f"{'='*90}")
    for name, _, _ in variants_p1:
        passes = sum(1 for r in p1_results
                     if r['variant'] == name and r['all_pos'])
        total = sum(1 for r in p1_results if r['variant'] == name)
        test_pnls = [r['test']['PnL'] for r in p1_results if r['variant'] == name]
        min_test = min(test_pnls)
        max_test = max(test_pnls)
        flag = 'PASS' if passes == total else 'FAIL'
        print(f"  {name:25s}: {passes}/{total} splits ALL-positive  "
              f"| Test range [{min_test:+.2f}, {max_test:+.2f}]  [{flag}]")

    # Save
    out = {
        'date_run': datetime.now().isoformat(),
        'p1_split_sensitivity': p1_results,
        'p3_cd_sensitivity': p3_results,
    }
    out_path = ROOT / 'results' / 'c1_refined_dmining_check.json'
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2, default=str)
    print(f"\nSaved: {out_path}")


if __name__ == '__main__':
    main()
