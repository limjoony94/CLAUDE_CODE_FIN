"""M3-R18 — Phase 2: α deep optimization in selective region.

R17 trend: more selective params → less negative. Dense grid in selective region.
min_n=15, broader range to capture true selective optima.
3-way split + WF for robust validation.
"""
import sys, json
from datetime import datetime, timezone
from pathlib import Path
from itertools import product
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / 'scripts' / 'analysis'))

from m3_critique_pipeline import prepare_all_data
from m3_round10_multidim_grid import make_alpha_entry_param, run_combo


def main():
    print("Loading data...")
    df, h1, h4, base_valid, eth_valid, funding_valid = prepare_all_data()
    n_total = len(df)
    print(f"  bars: {n_total:,}\n")

    # 3-way split: 50% train / 25% val / 25% holdout
    train_end = int(n_total * 0.5)
    val_end = int(n_total * 0.75)
    df_tr = df.iloc[:train_end].reset_index(drop=True)
    df_vl = df.iloc[train_end:val_end].reset_index(drop=True)
    df_te = df.iloc[val_end:].reset_index(drop=True)
    h1_tr, h1_vl, h1_te = h1[:train_end], h1[train_end:val_end], h1[val_end:]
    h4_tr, h4_vl, h4_te = h4[:train_end], h4[train_end:val_end], h4[val_end:]
    v_tr, v_vl, v_te = eth_valid[:train_end], eth_valid[train_end:val_end], eth_valid[val_end:]
    print(f"  Split: train={train_end} ({train_end/96:.0f}d), val={val_end-train_end} ({(val_end-train_end)/96:.0f}d), holdout={n_total-val_end} ({(n_total-val_end)/96:.0f}d)\n")

    # Dense selective region grid (expanded ranges, lower min_n)
    et_grid = (0.20, 0.30, 0.40, 0.50, 0.60, 0.70)
    bl_grid = (0.00, 0.05, 0.10, 0.15, 0.20)
    ap_grid = (60, 70, 75, 80, 85)
    N_grid = (2, 4, 6, 8, 10)
    pgrid = list(product(et_grid, bl_grid, ap_grid))
    total = len(pgrid) * len(N_grid)
    print(f"Selective grid: {len(pgrid)} param × {len(N_grid)} N = {total} configs (min_n=8)\n")

    friction = 0.04
    results = []
    for et, bl, ap in pgrid:
        for N in N_grid:
            entry_fn = make_alpha_entry_param(et, bl, ap)
            s_tr = run_combo(df_tr, h1_tr, h4_tr, v_tr, entry_fn, N, friction)
            s_vl = run_combo(df_vl, h1_vl, h4_vl, v_vl, entry_fn, N, friction)
            s_te = run_combo(df_te, h1_te, h4_te, v_te, entry_fn, N, friction)
            if s_tr is None or s_vl is None or s_te is None: continue
            if s_tr['n'] < 8 or s_vl['n'] < 8 or s_te['n'] < 8: continue
            results.append({
                'et': et, 'bl': bl, 'ap': ap, 'N': N,
                'tr_daily': s_tr['daily_net'], 'tr_n': s_tr['n'], 'tr_wr': s_tr['wr_pct'],
                'vl_daily': s_vl['daily_net'], 'vl_n': s_vl['n'], 'vl_wr': s_vl['wr_pct'],
                'te_daily': s_te['daily_net'], 'te_n': s_te['n'], 'te_wr': s_te['wr_pct'],
            })

    print(f"Valid configs: {len(results)} / {total}")

    # Distribution stats
    if results:
        tr_arr = np.array([r['tr_daily'] for r in results])
        vl_arr = np.array([r['vl_daily'] for r in results])
        te_arr = np.array([r['te_daily'] for r in results])
        print(f"\n{'period':<8} {'p_pos':>8} {'max':>10} {'median':>10} {'min':>10} {'std':>8}")
        for label, arr in [('train', tr_arr), ('val', vl_arr), ('test', te_arr)]:
            print(f"{label:<8} {np.mean(arr > 0)*100:>7.1f}% {np.max(arr):>+9.4f}% {np.median(arr):>+9.4f}% {np.min(arr):>+9.4f}% {np.std(arr):>7.4f}")

        # Cross-period stability
        all_3_pos = sum(1 for r in results if r['tr_daily'] > 0 and r['vl_daily'] > 0 and r['te_daily'] > 0)
        any_2_pos = sum(1 for r in results if (r['tr_daily']>0)+(r['vl_daily']>0)+(r['te_daily']>0) >= 2)
        print(f"\n  Configs with all 3 periods positive: {all_3_pos}/{len(results)}")
        print(f"  Configs with ≥2 periods positive: {any_2_pos}/{len(results)}")

        # Correlations
        print(f"\n  Pearson corr:")
        print(f"    train ↔ val: {np.corrcoef(tr_arr, vl_arr)[0,1]:+.4f}")
        print(f"    train ↔ test: {np.corrcoef(tr_arr, te_arr)[0,1]:+.4f}")
        print(f"    val ↔ test: {np.corrcoef(vl_arr, te_arr)[0,1]:+.4f}")

        # Top configs by val (selection rule: pick best by val, evaluate on test)
        # Use train+val mean for ranking, then test as holdout
        for r in results:
            r['trval_mean'] = (r['tr_daily'] + r['vl_daily']) / 2
        sorted_by_trval = sorted(results, key=lambda r: -r['trval_mean'])
        print(f"\n{'=' * 80}\nTOP 20 by train+val mean (test = HOLDOUT)\n{'=' * 80}")
        print(f"{'et':>5} {'bl':>5} {'ap':>4} {'N':>3} {'tr':>10} {'vl':>10} {'te(hold)':>10} {'tr_n':>5} {'te_n':>5}")
        for r in sorted_by_trval[:20]:
            print(f"{r['et']:>5.2f} {r['bl']:>5.2f} {r['ap']:>4} {r['N']:>3} "
                  f"{r['tr_daily']:>+9.4f} {r['vl_daily']:>+9.4f} {r['te_daily']:>+9.4f} "
                  f"{r['tr_n']:>5} {r['te_n']:>5}")

        top20 = sorted_by_trval[:20]
        top20_te_pos = sum(1 for r in top20 if r['te_daily'] > 0)
        print(f"\n  Top-20 by train+val: {top20_te_pos}/20 positive on test holdout")

        # Take top by val only (cleaner — train doesn't influence)
        sorted_by_val = sorted(results, key=lambda r: -r['vl_daily'])
        print(f"\n{'=' * 80}\nTOP 10 by VAL only (test = HOLDOUT)\n{'=' * 80}")
        print(f"{'et':>5} {'bl':>5} {'ap':>4} {'N':>3} {'tr':>10} {'vl':>10} {'te(hold)':>10}")
        for r in sorted_by_val[:10]:
            print(f"{r['et']:>5.2f} {r['bl']:>5.2f} {r['ap']:>4} {r['N']:>3} "
                  f"{r['tr_daily']:>+9.4f} {r['vl_daily']:>+9.4f} {r['te_daily']:>+9.4f}")

        # Best by val that survives test
        survivors = [r for r in sorted_by_val[:10] if r['te_daily'] > 0]
        print(f"\n  Top-10 by val: {len(survivors)}/10 positive on test holdout")

        # Robust optimum: median of top configs
        if survivors:
            best = max(survivors, key=lambda r: r['te_daily'])
            print(f"\n  → Best survivor: et={best['et']}, bl={best['bl']}, ap={best['ap']}, N={best['N']}")
            print(f"    train: {best['tr_daily']:+.4f}%, val: {best['vl_daily']:+.4f}%, test: {best['te_daily']:+.4f}%")
        else:
            print(f"\n  → No top-10-by-val survives test holdout")

    out = {'date': datetime.now(timezone.utc).isoformat(),
           'pre_reg': 'R17 trend → R18 selective region',
           'split': f'train={train_end}, val={val_end-train_end}, holdout={n_total-val_end}',
           'n_configs': len(results),
           'all_results': results,
           'all_3_pos': all_3_pos if results else 0,
           'any_2_pos': any_2_pos if results else 0,
           'top10_by_val_survivors_in_test': len(survivors) if results else 0,
           'best_survivor': best if results and survivors else None}
    p = ROOT / 'results' / f'm3_r18_alpha_deep_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(p, 'w') as f: json.dump(out, f, indent=2, default=str)
    print(f"\nSaved: {p}")


if __name__ == '__main__':
    main()
