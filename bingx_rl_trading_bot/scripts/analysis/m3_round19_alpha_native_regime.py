"""M3-R19 — α native regime (last 360 days) optimization.

R18 finding: α fires only in 2025-02+. Skip dead pre-period.
3-way split + WF within native regime for fair Phase 2 assessment.
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
    print(f"  Full data: {n_total:,} bars ({n_total/96:.0f} days)")

    # Filter to native regime (last 50% = 360 days, where α fires)
    native_start = n_total // 2
    df_n = df.iloc[native_start:].reset_index(drop=True)
    h1_n = h1[native_start:]; h4_n = h4[native_start:]; v_n = eth_valid[native_start:]
    n_native = len(df_n)
    print(f"  Native regime: {n_native:,} bars ({n_native/96:.0f} days)")
    print(f"    {df.iloc[native_start]['timestamp']} ~ {df.iloc[-1]['timestamp']}\n")

    # ============================================================
    # APPROACH A: 3-way split within native regime
    # ============================================================
    print("=" * 80); print("APPROACH A: 3-way split (50/25/25) within native regime"); print("=" * 80)
    tr_end = int(n_native * 0.5)
    vl_end = int(n_native * 0.75)
    df_tr = df_n.iloc[:tr_end].reset_index(drop=True)
    df_vl = df_n.iloc[tr_end:vl_end].reset_index(drop=True)
    df_te = df_n.iloc[vl_end:].reset_index(drop=True)
    h1_tr, h1_vl, h1_te = h1_n[:tr_end], h1_n[tr_end:vl_end], h1_n[vl_end:]
    h4_tr, h4_vl, h4_te = h4_n[:tr_end], h4_n[tr_end:vl_end], h4_n[vl_end:]
    vt, vv, vte = v_n[:tr_end], v_n[tr_end:vl_end], v_n[vl_end:]
    print(f"  train={tr_end} ({tr_end/96:.0f}d), val={vl_end-tr_end} ({(vl_end-tr_end)/96:.0f}d), test={n_native-vl_end} ({(n_native-vl_end)/96:.0f}d)\n")

    # Diagnostic: signal density in each split
    base_fn = make_alpha_entry_param(0.3, 0.1, 70)
    sigs_tr = base_fn(df_tr, h1_tr, h4_tr, vt)
    sigs_vl = base_fn(df_vl, h1_vl, h4_vl, vv)
    sigs_te = base_fn(df_te, h1_te, h4_te, vte)
    print(f"  Base config α(0.3, 0.1, 70) signals: train={len(sigs_tr)}, val={len(sigs_vl)}, test={len(sigs_te)}\n")

    # Grid sweep
    et_grid = (0.20, 0.30, 0.40, 0.50)
    bl_grid = (0.05, 0.10, 0.15, 0.20)
    ap_grid = (60, 70, 80)
    N_grid = (4, 6, 8, 12)
    pgrid = list(product(et_grid, bl_grid, ap_grid))
    total = len(pgrid) * len(N_grid)
    print(f"  Grid: {total} configs\n")

    friction = 0.04
    A_results = []
    for et, bl, ap in pgrid:
        for N in N_grid:
            entry_fn = make_alpha_entry_param(et, bl, ap)
            s_tr = run_combo(df_tr, h1_tr, h4_tr, vt, entry_fn, N, friction)
            s_vl = run_combo(df_vl, h1_vl, h4_vl, vv, entry_fn, N, friction)
            s_te = run_combo(df_te, h1_te, h4_te, vte, entry_fn, N, friction)
            if s_tr is None or s_vl is None or s_te is None: continue
            if s_tr['n'] < 8 or s_vl['n'] < 8 or s_te['n'] < 8: continue
            A_results.append({
                'et': et, 'bl': bl, 'ap': ap, 'N': N,
                'tr_daily': s_tr['daily_net'], 'tr_n': s_tr['n'],
                'vl_daily': s_vl['daily_net'], 'vl_n': s_vl['n'],
                'te_daily': s_te['daily_net'], 'te_n': s_te['n'],
            })
    print(f"  Valid configs: {len(A_results)}/{total}")

    if A_results:
        tr_arr = np.array([r['tr_daily'] for r in A_results])
        vl_arr = np.array([r['vl_daily'] for r in A_results])
        te_arr = np.array([r['te_daily'] for r in A_results])
        print(f"\n  Distribution (n={len(A_results)}):")
        print(f"    {'period':<8} {'p_pos':>7} {'max':>10} {'median':>10} {'min':>10}")
        for label, arr in [('train', tr_arr), ('val', vl_arr), ('test', te_arr)]:
            print(f"    {label:<8} {np.mean(arr > 0)*100:>6.1f}% {np.max(arr):>+9.4f}% {np.median(arr):>+9.4f}% {np.min(arr):>+9.4f}%")

        all_3_pos = sum(1 for r in A_results if r['tr_daily'] > 0 and r['vl_daily'] > 0 and r['te_daily'] > 0)
        print(f"\n    All-3 positive: {all_3_pos}/{len(A_results)}")

        # Sort by val (selection rule), check test
        sorted_by_val = sorted(A_results, key=lambda r: -r['vl_daily'])
        top10 = sorted_by_val[:10]
        top10_te_pos = sum(1 for r in top10 if r['te_daily'] > 0)
        print(f"\n  Top-10 by VAL → test holdout positive: {top10_te_pos}/10")
        print(f"\n  Top 10 by val:")
        print(f"  {'et':>5} {'bl':>5} {'ap':>4} {'N':>3} {'tr':>10} {'vl':>10} {'te(hold)':>10}")
        for r in top10:
            print(f"  {r['et']:>5.2f} {r['bl']:>5.2f} {r['ap']:>4} {r['N']:>3} "
                  f"{r['tr_daily']:>+9.4f} {r['vl_daily']:>+9.4f} {r['te_daily']:>+9.4f}")

    # ============================================================
    # APPROACH B: WF 4-fold within native regime
    # ============================================================
    print("\n" + "=" * 80); print("APPROACH B: WF 4-fold within native regime"); print("=" * 80)
    n_folds = 4
    fold_size = n_native // (n_folds + 1)
    print(f"  fold_size={fold_size} ({fold_size/96:.0f}d), {n_folds} test folds\n")

    B_results = []
    for et, bl, ap in pgrid:
        for N in N_grid:
            entry_fn = make_alpha_entry_param(et, bl, ap)
            fold_dailies = []; fold_ns = []
            for fold_i in range(n_folds):
                tr_e = (fold_i + 1) * fold_size
                te_s = tr_e
                te_e = min(te_s + fold_size, n_native)
                df_t = df_n.iloc[te_s:te_e].reset_index(drop=True)
                h1_t = h1_n[te_s:te_e]; h4_t = h4_n[te_s:te_e]; v_t = v_n[te_s:te_e]
                s = run_combo(df_t, h1_t, h4_t, v_t, entry_fn, N, friction)
                fold_dailies.append(s['daily_net'] if s else None)
                fold_ns.append(s['n'] if s else 0)
            valid_d = [d for d in fold_dailies if d is not None]
            if len(valid_d) < 3: continue
            pos_count = sum(1 for d in valid_d if d > 0)
            mean_d = float(np.mean(valid_d))
            B_results.append({
                'et': et, 'bl': bl, 'ap': ap, 'N': N,
                'fold_dailies': fold_dailies, 'fold_ns': fold_ns,
                'pos_count': pos_count, 'mean_daily': mean_d,
                'valid_folds': len(valid_d),
            })

    cat_4_4 = [r for r in B_results if r['pos_count'] == 4 and r['valid_folds'] == 4]
    cat_3_4 = [r for r in B_results if r['pos_count'] == 3 and r['valid_folds'] >= 3]
    print(f"  WF 4-fold results:")
    print(f"    4/4 folds positive: {len(cat_4_4)}")
    print(f"    3+/4 folds positive: {len(cat_3_4)}")
    exp_4_4 = total * 0.5 ** 4
    print(f"    Expected 4/4 by chance: ~{exp_4_4:.1f}")

    if cat_4_4:
        cat_4_4.sort(key=lambda r: -r['mean_daily'])
        print(f"\n  TOP 4/4 robust:")
        print(f"  {'et':>5} {'bl':>5} {'ap':>4} {'N':>3} {'mean':>10} {'folds':>40}")
        for r in cat_4_4[:10]:
            fd = ' '.join(f"{x:+.3f}" for x in r['fold_dailies'])
            print(f"  {r['et']:>5.2f} {r['bl']:>5.2f} {r['ap']:>4} {r['N']:>3} "
                  f"{r['mean_daily']:>+9.4f}% [{fd}]")

    # Phase 3 readiness verdict
    print(f"\n{'=' * 80}\nPHASE 3 READINESS (native regime)\n{'=' * 80}")
    cond_a = A_results and any(r['te_daily'] > 0 for r in sorted(A_results, key=lambda r: -r['vl_daily'])[:10])
    cond_b = cat_4_4 and any(r['mean_daily'] > 0 for r in cat_4_4)
    print(f"  Approach A (top-10 by val survives test): {'YES' if cond_a else 'NO'}")
    print(f"  Approach B (4/4 WF + mean > 0): {'YES' if cond_b else 'NO'}")
    print(f"  Combined readiness: {'YES' if cond_a and cond_b else 'NO'}")

    out = {'date': datetime.now(timezone.utc).isoformat(),
           'native_regime_days': n_native / 96,
           'approach_a_results': A_results,
           'approach_b_results': B_results,
           'cat_4_4': cat_4_4,
           'cond_a': bool(cond_a),
           'cond_b': bool(cond_b),
           'phase3_ready': bool(cond_a and cond_b)}
    p = ROOT / 'results' / f'm3_r19_alpha_native_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(p, 'w') as f: json.dump(out, f, indent=2, default=str)
    print(f"\nSaved: {p}")


if __name__ == '__main__':
    main()
