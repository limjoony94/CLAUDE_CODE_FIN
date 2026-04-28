"""M3-R17 — Comprehensive strategy potential across 8 families.

User-directed: comprehensive sweep + relative ranking + TOP-1 to deep optimization.
"""
import sys, json
from datetime import datetime, timezone
from pathlib import Path
from itertools import product
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / 'scripts' / 'analysis'))

from m3_critique_pipeline import (prepare_all_data, run_bt_with_spec, trade_summary,
                                    rolling_pctile)
from m3_round2_critique import prepare_data_with_eth_break
from m3_round8_critique import prepare_data_r8
from m3_round6_critique import prepare_data_r6
from m3_round10_multidim_grid import (make_alpha_entry_param, make_iota_entry_param,
                                        make_fixed_exit, run_combo)
from m3_round14_potential import (make_kappa_entry, make_sigma_entry, make_upsilon_entry, make_zeta_entry)


# Beta entry (BTC-ETH spread mean-rev)
def make_beta_entry(z_thresh, corr_thresh):
    def fn(df, h1, h4, valid, params=None):
        n = len(df)
        z = df['ratio_z'].values
        corr = df['corr50'].values
        sigs = []
        for i in range(1, n):
            if not valid[i]: continue
            if pd.isna(z[i]) or pd.isna(corr[i]): continue
            if not (corr[i] < corr_thresh): continue
            if z[i] < -z_thresh and h1[i] and h4[i]:
                sigs.append((i, 'LONG'))
            elif z[i] > z_thresh and (not h1[i]) and (not h4[i]):
                sigs.append((i, 'SHORT'))
        return sigs
    return fn


# Gamma entry (funding × cross-asset)
def make_gamma_entry(funding_thresh, rsi_thresh):
    def fn(df, h1, h4, valid, params=None):
        n = len(df)
        fsum = df['funding_8sum'].values
        rsi = df['rsi14'].values
        eth_ret = df['eth_return'].values
        sigs = []
        for i in range(1, n):
            if not valid[i]: continue
            if any(pd.isna(x) for x in (fsum[i], rsi[i], eth_ret[i - 1])): continue
            if fsum[i] >= funding_thresh and rsi[i] >= rsi_thresh and eth_ret[i - 1] < 0:
                sigs.append((i, 'SHORT'))
            elif fsum[i] <= -funding_thresh and rsi[i] <= (100 - rsi_thresh) and eth_ret[i - 1] > 0:
                sigs.append((i, 'LONG'))
        return sigs
    return fn


def run_family_grid(df, h1, h4, valid_mask, factory, param_grid, N_grid, friction=0.04, train_frac=0.6, min_n=30):
    n_total = len(df)
    train_end = int(n_total * train_frac)
    df_tr = df.iloc[:train_end].reset_index(drop=True)
    df_te = df.iloc[train_end:].reset_index(drop=True)
    h1_tr, h1_te = h1[:train_end], h1[train_end:]
    h4_tr, h4_te = h4[:train_end], h4[train_end:]
    valid_tr = valid_mask[:train_end]; valid_te = valid_mask[train_end:]

    results = []
    for params in param_grid:
        for N in N_grid:
            entry_fn = factory(*params)
            s_tr = run_combo(df_tr, h1_tr, h4_tr, valid_tr, entry_fn, N, friction)
            s_te = run_combo(df_te, h1_te, h4_te, valid_te, entry_fn, N, friction)
            if s_tr is None or s_te is None: continue
            if s_tr['n'] < min_n or s_te['n'] < min_n: continue
            results.append({
                'params': list(params), 'N': N,
                'train_daily': s_tr['daily_net'], 'train_n': s_tr['n'], 'train_wr': s_tr['wr_pct'],
                'test_daily': s_te['daily_net'], 'test_n': s_te['n'], 'test_wr': s_te['wr_pct'],
            })
    return results


def compute_potential_v2(results, name):
    """Composite potential score (relative ranking)."""
    if not results:
        return {'name': name, 'n_configs': 0, 'composite': -999}
    test_arr = np.array([r['test_daily'] for r in results])
    train_arr = np.array([r['train_daily'] for r in results])

    max_test = float(np.max(test_arr))
    median_test = float(np.median(test_arr))
    min_test = float(np.min(test_arr))
    std_test = float(np.std(test_arr))
    p_test_pos = float(np.mean(test_arr > 0) * 100)
    p_both_pos = float(np.mean((train_arr > 0) & (test_arr > 0)) * 100)
    corr_tt = float(np.corrcoef(train_arr, test_arr)[0, 1]) if len(train_arr) > 2 else 0

    composite = (max_test * 100 + median_test * 50 + p_both_pos * 0.5
                  + corr_tt * 30 - std_test * 30)

    best = max(results, key=lambda r: r['test_daily'])

    return {
        'name': name, 'n_configs': len(results),
        'max_test': round(max_test, 4), 'median_test': round(median_test, 4),
        'min_test': round(min_test, 4), 'std_test': round(std_test, 4),
        'p_test_pos': round(p_test_pos, 2), 'p_both_pos': round(p_both_pos, 2),
        'corr_tt': round(corr_tt, 4),
        'composite': round(composite, 2),
        'best_config': best,
    }


def main():
    print("Loading data...")
    df_r8, h1, h4, base_valid, eth_valid_ext, funding_valid, _, _ = prepare_data_r8()
    df_r6, _, _, _, _, _, upsilon_valid, _ = prepare_data_r6()
    df_r8['volume_sma'] = df_r6['volume_sma'].values
    df = df_r8
    df['eth_accel'] = pd.Series(df['eth_return'].values).diff(8).values

    eligible_with_filter = (h1 & h4 | (~h1) & (~h4))
    n_total = len(df)
    print(f"  bars: {n_total:,} | days: {n_total/96:.0f}\n")

    family_results = {}

    # α: ETH-lag steady-state
    print("=" * 80); print("[1/8] α — ETH-lag steady-state"); print("=" * 80)
    et_grid = (0.10, 0.20, 0.30, 0.40, 0.50)
    bl_grid = (0.05, 0.10, 0.15, 0.20, 0.25)
    ap_grid = (50, 60, 70, 80)
    N_grid = (2, 4, 6, 8, 12)
    pgrid = list(product(et_grid, bl_grid, ap_grid))
    print(f"  {len(pgrid)} param × {len(N_grid)} N = {len(pgrid)*len(N_grid)} configs")
    factory = lambda et, bl, ap: make_alpha_entry_param(et, bl, ap)
    res = run_family_grid(df, h1, h4, eth_valid_ext, factory, pgrid, N_grid, friction=0.04, min_n=30)
    print(f"  valid: {len(res)}")
    family_results['α'] = res

    # ι: α + ETH 24-bar break
    print("\n" + "=" * 80); print("[2/8] ι — α + ETH break filter"); print("=" * 80)
    et_grid = (0.20, 0.30, 0.40, 0.50)
    bl_grid = (0.05, 0.10, 0.15, 0.20)
    ap_grid = (60, 70, 80)
    lb_grid = (12, 18, 24, 30)
    N_grid_iota = (4, 6, 8, 12, 16)
    pgrid = list(product(et_grid, bl_grid, ap_grid, lb_grid))
    print(f"  {len(pgrid)} param × {len(N_grid_iota)} N = {len(pgrid)*len(N_grid_iota)} configs")
    factory = lambda et, bl, ap, lb: make_iota_entry_param(et, bl, ap, lb)
    res = run_family_grid(df, h1, h4, eth_valid_ext, factory, pgrid, N_grid_iota, friction=0.04, min_n=30)
    print(f"  valid: {len(res)}")
    family_results['ι'] = res

    # κ: α + mid-vol regime
    print("\n" + "=" * 80); print("[3/8] κ — α + mid-vol regime"); print("=" * 80)
    et_grid = (0.20, 0.30, 0.40, 0.50)
    bl_grid = (0.05, 0.10, 0.15, 0.20)
    lb_grid = (18, 24, 30, 36)
    N_grid_k = (4, 6, 8, 12, 16)
    pgrid = list(product(et_grid, bl_grid, lb_grid))
    print(f"  {len(pgrid)} param × {len(N_grid_k)} N = {len(pgrid)*len(N_grid_k)} configs")
    factory = lambda et, bl, lb: make_kappa_entry(et, bl, lb)
    res = run_family_grid(df, h1, h4, eth_valid_ext, factory, pgrid, N_grid_k, friction=0.04, min_n=30)
    print(f"  valid: {len(res)}")
    family_results['κ'] = res

    # σ: counter-trend at break
    print("\n" + "=" * 80); print("[4/8] σ — counter-trend at break"); print("=" * 80)
    rsi_grid = (60, 65, 70, 75, 80)
    lb_grid = (12, 18, 24, 30, 36)
    N_grid_s = (4, 6, 8, 12, 16)
    pgrid = list(product(rsi_grid, lb_grid))
    print(f"  {len(pgrid)} param × {len(N_grid_s)} N = {len(pgrid)*len(N_grid_s)} configs")
    factory = lambda rt, lb: make_sigma_entry(rt, lb)
    res = run_family_grid(df, h1, h4, eth_valid_ext, factory, pgrid, N_grid_s, friction=0.04, min_n=30)
    print(f"  valid: {len(res)}")
    family_results['σ'] = res

    # υ: volume × cross-asset
    print("\n" + "=" * 80); print("[5/8] υ — volume × cross-asset"); print("=" * 80)
    vol_grid = (1.5, 2.0, 2.5, 3.0, 4.0)
    et_grid = (0.10, 0.20, 0.30, 0.40, 0.50)
    N_grid_u = (2, 4, 6, 8, 12)
    pgrid = list(product(vol_grid, et_grid))
    print(f"  {len(pgrid)} param × {len(N_grid_u)} N = {len(pgrid)*len(N_grid_u)} configs")
    factory = lambda vm, et: make_upsilon_entry(vm, et)
    res = run_family_grid(df, h1, h4, upsilon_valid, factory, pgrid, N_grid_u, friction=0.04, min_n=30)
    print(f"  valid: {len(res)}")
    family_results['υ'] = res

    # ζ: ETH acceleration
    print("\n" + "=" * 80); print("[6/8] ζ — ETH return acceleration"); print("=" * 80)
    accel_grid = (0.2, 0.3, 0.5, 0.8, 1.0)
    N_grid_z = (2, 4, 6, 8, 12)
    pgrid = list(product(accel_grid,))
    print(f"  {len(pgrid)} param × {len(N_grid_z)} N = {len(pgrid)*len(N_grid_z)} configs")
    zeta_valid = eth_valid_ext & (~pd.isna(df['eth_accel'])).values
    factory = lambda at: make_zeta_entry(at)
    res = run_family_grid(df, h1, h4, zeta_valid, factory, pgrid, N_grid_z, friction=0.04, min_n=30)
    print(f"  valid: {len(res)}")
    family_results['ζ'] = res

    # β: BTC-ETH spread mean-rev
    print("\n" + "=" * 80); print("[7/8] β — spread mean-rev"); print("=" * 80)
    z_grid = (1.5, 2.0, 2.5, 3.0)
    corr_grid = (0.3, 0.5, 0.7)
    N_grid_b = (4, 8, 12, 16, 24)
    pgrid = list(product(z_grid, corr_grid))
    print(f"  {len(pgrid)} param × {len(N_grid_b)} N = {len(pgrid)*len(N_grid_b)} configs")
    factory = lambda z, c: make_beta_entry(z, c)
    res = run_family_grid(df, h1, h4, eth_valid_ext, factory, pgrid, N_grid_b, friction=0.04, min_n=30)
    print(f"  valid: {len(res)}")
    family_results['β'] = res

    # γ: funding × cross-asset
    print("\n" + "=" * 80); print("[8/8] γ — funding × cross-asset"); print("=" * 80)
    fsum_grid = (0.18, 0.24, 0.30, 0.36)
    rsi_grid = (65, 70, 75, 80)
    N_grid_g = (4, 8, 12, 16)
    pgrid = list(product(fsum_grid, rsi_grid))
    print(f"  {len(pgrid)} param × {len(N_grid_g)} N = {len(pgrid)*len(N_grid_g)} configs")
    factory = lambda f, r: make_gamma_entry(f, r)
    res = run_family_grid(df, h1, h4, funding_valid, factory, pgrid, N_grid_g, friction=0.04, min_n=20)
    print(f"  valid: {len(res)}")
    family_results['γ'] = res

    # Compute composite potential per family
    print("\n" + "=" * 100); print("STRATEGY POTENTIAL RANKING (Composite)"); print("=" * 100)
    family_potential = {}
    for name, res in family_results.items():
        family_potential[name] = compute_potential_v2(res, name)

    print(f"{'family':<6} {'n':>5} {'max_te':>9} {'med_te':>9} {'min_te':>9} {'std_te':>8} "
          f"{'p_te+':>7} {'p_both+':>9} {'corr':>7} {'composite':>11}")
    sorted_families = sorted(family_potential.items(), key=lambda kv: -kv[1].get('composite', -999))
    for name, p in sorted_families:
        if p.get('n_configs', 0) == 0:
            print(f"{name:<6} {'0':>5}")
            continue
        print(f"{name:<6} {p['n_configs']:>5} {p['max_test']:>+8.4f} {p['median_test']:>+8.4f} "
              f"{p['min_test']:>+8.4f} {p['std_test']:>7.4f} {p['p_test_pos']:>6.1f}% "
              f"{p['p_both_pos']:>8.1f}% {p['corr_tt']:>+6.3f} {p['composite']:>10.2f}")

    # TOP-1 selection (highest composite)
    valid_potentials = [(n, p) for n, p in sorted_families if p.get('n_configs', 0) > 0]
    if valid_potentials:
        top1_name, top1_pot = valid_potentials[0]
        print(f"\n{'=' * 80}\nTOP-1 STRATEGY: {top1_name}\n{'=' * 80}")
        print(f"  Composite: {top1_pot['composite']}")
        print(f"  Best config: params={top1_pot['best_config']['params']}, N={top1_pot['best_config']['N']}")
        print(f"    train daily: {top1_pot['best_config']['train_daily']:+.4f}%, n={top1_pot['best_config']['train_n']}")
        print(f"    test daily:  {top1_pot['best_config']['test_daily']:+.4f}%, n={top1_pot['best_config']['test_n']}")
        print(f"  Family stats: max_test={top1_pot['max_test']:+.4f}, median={top1_pot['median_test']:+.4f}")
        print(f"  Surface profitability: {'POSITIVE' if top1_pot['max_test'] > 0 else 'NEGATIVE'}")

        # Trend analysis: marginal effect of each param
        print(f"\n  Trend analysis (TOP-1 family marginal effects):")
        results = family_results[top1_name]
        # For each parameter axis, compute mean test_daily at each value
        param_axes = list(range(len(results[0]['params'])))
        for axis_idx in param_axes:
            unique_vals = sorted(set(r['params'][axis_idx] for r in results))
            print(f"    param[{axis_idx}]:")
            for v in unique_vals:
                subset = [r for r in results if r['params'][axis_idx] == v]
                mean_te = np.mean([r['test_daily'] for r in subset])
                print(f"      {v}: mean_test={mean_te:+.4f}% (n={len(subset)})")
        # N axis
        unique_N = sorted(set(r['N'] for r in results))
        print(f"    N_exit:")
        for nv in unique_N:
            subset = [r for r in results if r['N'] == nv]
            mean_te = np.mean([r['test_daily'] for r in subset])
            print(f"      N={nv}: mean_test={mean_te:+.4f}% (n={len(subset)})")

    out = {'date': datetime.now(timezone.utc).isoformat(),
           'pre_reg': 'claudedocs/m3_round17_comprehensive_potential.md',
           'family_potential': family_potential,
           'top1': sorted_families[0] if sorted_families else None,
           'family_results_keys': list(family_results.keys()),
           # Save sample of results per family (limited size)
           'family_results_top10': {n: sorted(r, key=lambda x: -x['test_daily'])[:10]
                                       for n, r in family_results.items()}}
    p = ROOT / 'results' / f'm3_r17_comprehensive_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(p, 'w') as f: json.dump(out, f, indent=2, default=str)
    print(f"\nSaved: {p}")

    # Also save raw all results separately for R18 trend analysis
    all_results_path = ROOT / 'results' / f'm3_r17_all_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(all_results_path, 'w') as f:
        json.dump({'family_results': family_results}, f, indent=2, default=str)
    print(f"All results: {all_results_path}")


if __name__ == '__main__':
    main()
