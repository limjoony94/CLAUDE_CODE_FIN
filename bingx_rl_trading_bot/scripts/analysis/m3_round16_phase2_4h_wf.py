"""M3-R16 — Phase 2: 4h optimal parameter search via WF 5-fold.

Pre-reg: claudedocs/m3_round16_phase2_4h.md
- Refined grid (eth × btc × N): 144 combos
- WF 5-fold expanding window
- Robust criterion: 5/5 folds positive + mean > +0.02%/day
"""
import sys, json
from datetime import datetime, timezone
from pathlib import Path
from itertools import product
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / 'scripts' / 'analysis'))

from m3_round15_timeframe_potential import (prepare_tf_data, make_alpha_tf_entry,
                                              run_bt_simple_tf, trade_summary_simple)


def main():
    print("Loading 4h data...")
    df, h1, h2, valid, bpd = prepare_tf_data('4h')
    n_total = len(df)
    print(f"  bars: {n_total:,} | days: {n_total/bpd:.0f}\n")

    # Refined grid
    et_grid = (0.40, 0.50, 0.60, 0.70, 0.80, 1.00)
    bl_grid = (0.30, 0.40, 0.50, 0.60, 0.80, 1.00)
    N_grid = (1, 2, 3, 4)
    pgrid = list(product(et_grid, bl_grid))
    total_combos = len(pgrid) * len(N_grid)
    print(f"Grid: {len(pgrid)} param combos × {len(N_grid)} N = {total_combos} configs\n")

    # WF 5-fold expanding
    n_folds = 5
    fold_size = n_total // (n_folds + 1)
    fold_results = {}

    for combo_idx, (et, bl) in enumerate(pgrid):
        for N in N_grid:
            entry_fn = make_alpha_tf_entry(et, bl, 70, use_atr_filter=False)
            fold_dailies = []
            fold_ns = []
            for fold_i in range(n_folds):
                train_end = (fold_i + 1) * fold_size
                test_start = train_end
                test_end = min(test_start + fold_size, n_total)
                df_te = df.iloc[test_start:test_end].reset_index(drop=True)
                h1_te = h1[test_start:test_end]; h2_te = h2[test_start:test_end]; v_te = valid[test_start:test_end]
                sigs = entry_fn(df_te, h1_te, h2_te, v_te)
                trades = run_bt_simple_tf(df_te, sigs, N, friction=0.08)
                s = trade_summary_simple(trades)
                fold_dailies.append(s['daily_net'] if s else None)
                fold_ns.append(s['n'] if s else 0)
            valid_dailies = [d for d in fold_dailies if d is not None]
            if len(valid_dailies) == 0: continue
            pos_count = sum(1 for d in valid_dailies if d > 0)
            mean_daily = np.mean(valid_dailies)
            min_n = min(fold_ns) if fold_ns else 0
            min_daily = min(valid_dailies)
            key = (et, bl, N)
            fold_results[key] = {
                'eth_thresh': et, 'btc_lag': bl, 'N': N,
                'fold_dailies': fold_dailies, 'fold_ns': fold_ns,
                'pos_count': pos_count, 'mean_daily': float(mean_daily),
                'min_n': min_n, 'min_daily': min_daily,
            }
        if (combo_idx + 1) % 6 == 0:
            print(f"  progress {(combo_idx+1)*len(N_grid)}/{total_combos}")

    # Categorize
    all_5_5 = []  # 5/5 folds positive
    all_4_5 = []  # 4/5 folds positive
    all_3_5 = []
    for key, r in fold_results.items():
        if r['pos_count'] == 5:
            all_5_5.append(r)
        elif r['pos_count'] == 4:
            all_4_5.append(r)
        elif r['pos_count'] == 3:
            all_3_5.append(r)

    print(f"\n{'=' * 80}\nWF 5-fold ROBUST RESULTS\n{'=' * 80}")
    print(f"  5/5 folds positive: {len(all_5_5)}")
    print(f"  4/5 folds positive: {len(all_4_5)}")
    print(f"  3/5 folds positive: {len(all_3_5)}")
    print(f"  Total tested: {len(fold_results)}")

    # Bonferroni / chance check
    expected_5_5_chance = total_combos * 0.5 ** 5
    expected_4_5_chance = total_combos * 5 * 0.5 ** 5
    print(f"\n  Expected by chance (random walk):")
    print(f"    5/5: ~{expected_5_5_chance:.1f} | observed: {len(all_5_5)}")
    print(f"    4/5: ~{expected_4_5_chance:.1f} | observed: {len(all_4_5)}")

    # Top 5/5 robust configs
    all_5_5.sort(key=lambda r: -r['mean_daily'])
    print(f"\n{'=' * 80}\nTOP 5/5 ROBUST CONFIGS\n{'=' * 80}")
    if all_5_5:
        print(f"{'eth':>5} {'btc':>5} {'N':>3} {'mean':>10} {'min_d':>10} {'min_n':>6} {'fold dailies':>50}")
        for r in all_5_5[:15]:
            fd = ', '.join(f"{x:+.4f}" for x in r['fold_dailies'])
            print(f"{r['eth_thresh']:>5.2f} {r['btc_lag']:>5.2f} {r['N']:>3} "
                  f"{r['mean_daily']:>+9.4f}% {r['min_daily']:>+9.4f}% {r['min_n']:>5}  [{fd}]")

    # Pre-reg check on best 5/5
    print(f"\n{'=' * 80}\nPRE-REG CHECK (Phase 2 → Phase 3 gate)\n{'=' * 80}")
    if all_5_5:
        best = all_5_5[0]
        cond = {
            '5/5_folds_positive': best['pos_count'] == 5,
            'mean_gt_0.02': best['mean_daily'] > 0.02,
            'min_n_per_fold_ge_5': best['min_n'] >= 5,
            'no_fold_lt_minus_0.1': best['min_daily'] > -0.1,
        }
        all_pass = all(cond.values())
        print(f"  Best config: eth={best['eth_thresh']}, btc={best['btc_lag']}, N={best['N']}")
        print(f"  Mean daily: {best['mean_daily']:+.4f}%")
        print(f"  Min daily: {best['min_daily']:+.4f}%")
        print(f"  Min n: {best['min_n']}")
        print(f"  Conditions: {cond}")
        print(f"  Phase 2 → Phase 3 gate: {'PASS' if all_pass else 'FAIL'}")
        if all_pass:
            print(f"\n  → Eligible for Phase 3 (paper trade ramp-up)")
    else:
        print(f"  → 0 configs 5/5 robust. Phase 2 FAIL.")
        print(f"  → 4h R15 finding likely sample noise (small train n=10-18)")
        print(f"  → 사용자 결정 영역")
        all_pass = False

    out = {'date': datetime.now(timezone.utc).isoformat(),
           'pre_reg': 'claudedocs/m3_round16_phase2_4h.md',
           'total_configs_tested': len(fold_results),
           'robust_5_5': all_5_5,
           'robust_4_5': all_4_5,
           'robust_3_5_count': len(all_3_5),
           'expected_by_chance': {'5_5': expected_5_5_chance, '4_5': expected_4_5_chance},
           'best_5_5': all_5_5[0] if all_5_5 else None,
           'phase2_pass': all_pass}
    p = ROOT / 'results' / f'm3_r16_phase2_4h_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(p, 'w') as f: json.dump(out, f, indent=2, default=str)
    print(f"\nSaved: {p}")


if __name__ == '__main__':
    main()
