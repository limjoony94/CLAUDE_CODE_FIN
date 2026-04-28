"""M3-R18 — Phase 2 redesign: α deep optimization via WF 5-fold.

Train period (early ~50%) has 0 α-eligible signals — regime non-stationarity.
Switch from fixed train/val/test split to walk-forward.

WF 5-fold expanding window. For each config: count folds with daily_net > 0.
Robust optimum = config with most folds positive AND meaningful samples.
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
    print(f"  bars: {n_total:,} ({n_total/96:.0f} days)\n")

    # Diagnostic: signals per 6-month period at base config
    print("Diagnostic — α (et=0.3, bl=0.1, ap=70) signal density by period:")
    chunk_size = n_total // 6  # ~120 days each
    entry_fn_base = make_alpha_entry_param(0.3, 0.1, 70)
    for k in range(6):
        c_start = k * chunk_size
        c_end = (k + 1) * chunk_size
        df_c = df.iloc[c_start:c_end].reset_index(drop=True)
        h1_c, h4_c, v_c = h1[c_start:c_end], h4[c_start:c_end], eth_valid[c_start:c_end]
        sigs = entry_fn_base(df_c, h1_c, h4_c, v_c)
        ts_start = df.iloc[c_start]['timestamp']
        ts_end = df.iloc[c_end-1]['timestamp']
        print(f"  Chunk {k+1} ({ts_start} ~ {ts_end}): {len(sigs)} signals")
    print()

    # WF 5-fold expanding
    n_folds = 5
    fold_size = n_total // (n_folds + 1)

    # Selective + medium parameter grid
    et_grid = (0.20, 0.30, 0.40, 0.50)
    bl_grid = (0.05, 0.10, 0.15, 0.20)
    ap_grid = (60, 70, 80)
    N_grid = (4, 6, 8, 12)
    pgrid = list(product(et_grid, bl_grid, ap_grid))
    total_combos = len(pgrid) * len(N_grid)
    print(f"WF 5-fold on {len(pgrid)} param × {len(N_grid)} N = {total_combos} configs\n")

    friction = 0.04
    fold_results = []
    for et, bl, ap in pgrid:
        for N in N_grid:
            entry_fn = make_alpha_entry_param(et, bl, ap)
            fold_dailies = []
            fold_ns = []
            for fold_i in range(n_folds):
                tr_end = (fold_i + 1) * fold_size
                te_start = tr_end
                te_end = min(te_start + fold_size, n_total)
                df_t = df.iloc[te_start:te_end].reset_index(drop=True)
                h1_t = h1[te_start:te_end]; h4_t = h4[te_start:te_end]; v_t = eth_valid[te_start:te_end]
                s = run_combo(df_t, h1_t, h4_t, v_t, entry_fn, N, friction)
                fold_dailies.append(s['daily_net'] if s else None)
                fold_ns.append(s['n'] if s else 0)
            valid_dailies = [d for d in fold_dailies if d is not None]
            if not valid_dailies: continue
            pos_count = sum(1 for d in valid_dailies if d > 0)
            mean_daily = float(np.mean(valid_dailies))
            min_n = min(fold_ns)
            min_daily = min(valid_dailies)
            fold_results.append({
                'et': et, 'bl': bl, 'ap': ap, 'N': N,
                'fold_dailies': fold_dailies, 'fold_ns': fold_ns,
                'pos_count': pos_count, 'mean_daily': mean_daily,
                'min_n': min_n, 'min_daily': min_daily,
                'valid_folds': len(valid_dailies),
            })

    print(f"Total fold-tested configs: {len(fold_results)}\n")

    # Categorize
    cat_5_5 = [r for r in fold_results if r['pos_count'] == 5 and r['valid_folds'] == 5]
    cat_4_5 = [r for r in fold_results if r['pos_count'] == 4 and r['valid_folds'] == 5]
    cat_3_5 = [r for r in fold_results if r['pos_count'] == 3 and r['valid_folds'] == 5]
    cat_4_4 = [r for r in fold_results if r['pos_count'] == 4 and r['valid_folds'] == 4]
    print(f"WF results:")
    print(f"  5/5 folds positive: {len(cat_5_5)}")
    print(f"  4/5 folds positive: {len(cat_4_5)}")
    print(f"  3/5 folds positive: {len(cat_3_5)}")
    print(f"  4/4 (1 fold no-trade): {len(cat_4_4)}")

    # Expected by chance for each
    exp_5_5 = total_combos * 0.5 ** 5
    exp_4_5 = total_combos * 5 * 0.5 ** 5
    print(f"\n  Expected by chance: 5/5 ~{exp_5_5:.1f}, 4/5 ~{exp_4_5:.1f}")

    # Top by mean_daily within 4+/5 (ignoring 5/5 if empty)
    candidates = sorted(cat_5_5 + cat_4_5, key=lambda r: -r['mean_daily'])
    print(f"\n{'=' * 80}\nTOP CONFIGS (4+/5 folds positive)\n{'=' * 80}")
    if candidates:
        print(f"{'et':>5} {'bl':>5} {'ap':>4} {'N':>3} {'pos':>4} {'mean':>10} {'min_d':>10} {'min_n':>5} folds")
        for r in candidates[:15]:
            fd = ' '.join(f"{x:+.3f}" if x is not None else "  N/A" for x in r['fold_dailies'])
            print(f"{r['et']:>5.2f} {r['bl']:>5.2f} {r['ap']:>4} {r['N']:>3} {r['pos_count']:>4} "
                  f"{r['mean_daily']:>+9.4f}% {r['min_daily']:>+9.4f}% {r['min_n']:>5}  [{fd}]")
    else:
        print("  None.")

    # Pre-reg verdict: ≥1 config with 5/5 + mean > 0 → Phase 3 ready
    # Or ≥3 configs with 4/5 + mean > 0.005 → robust enough for Phase 3
    phase3_ready = (
        any(r['mean_daily'] > 0 for r in cat_5_5) or
        sum(1 for r in cat_4_5 if r['mean_daily'] > 0.005) >= 3
    )
    print(f"\n{'=' * 80}")
    print(f"PHASE 3 READINESS: {'YES' if phase3_ready else 'NO'}")
    print(f"{'=' * 80}")
    if phase3_ready:
        print(f"  → Robust optimum found. Proceed to Phase 3 (paper trade ramp-up).")
    else:
        print(f"  → No robust optimum. WF evidence consistent with R10/R16.")
        print(f"  → Even relative TOP-1 (α) cannot extract optimal params at robustness threshold.")

    out = {'date': datetime.now(timezone.utc).isoformat(),
           'pre_reg': 'R17 → R18 WF (regime-aware)',
           'total_configs_tested': len(fold_results),
           'cat_5_5': cat_5_5,
           'cat_4_5': cat_4_5,
           'cat_3_5_count': len(cat_3_5),
           'expected_by_chance': {'5_5': exp_5_5, '4_5': exp_4_5},
           'top_candidates_4plus_5': candidates[:15],
           'phase3_ready': phase3_ready}
    p = ROOT / 'results' / f'm3_r18_alpha_wf_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(p, 'w') as f: json.dump(out, f, indent=2, default=str)
    print(f"\nSaved: {p}")


if __name__ == '__main__':
    main()
