"""M3-R14 — Strategy Potential Assessment.

User-directed 3-phase methodology: assess POTENTIAL across 6 strategy families with
multi-dim parameter sweeps. Distribution-based metrics, not single-config conclusions.
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


# ---------- Generic combo runner ----------

def run_family_grid(df, h1, h4, valid_mask, eligible_mask,
                     entry_factory, param_grid, N_grid, friction=0.04,
                     train_frac=0.6, min_n=30):
    """Run grid sweep, return list of dicts with train/test metrics for each combo."""
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
            entry_fn = entry_factory(*params)
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


def compute_potential(results, name):
    """Compute potential metrics from grid sweep results."""
    if not results:
        return {'name': name, 'n_configs': 0, 'note': 'no valid configs', 'potential_score': -999}
    train_arr = np.array([r['train_daily'] for r in results])
    test_arr = np.array([r['test_daily'] for r in results])

    p_train_pos = np.mean(train_arr > 0) * 100
    p_test_pos = np.mean(test_arr > 0) * 100
    p_both_pos = np.mean((train_arr > 0) & (test_arr > 0)) * 100
    if len(train_arr) > 2:
        corr_tt = float(np.corrcoef(train_arr, test_arr)[0, 1])
    else:
        corr_tt = None

    median_test = float(np.median(test_arr))
    max_test = float(np.max(test_arr))
    min_test = float(np.min(test_arr))

    # Composite
    potential = p_both_pos * 1.0 + (corr_tt or 0) * 50 + median_test * 100

    # Find best config (max test_daily)
    best = max(results, key=lambda r: r['test_daily'])

    return {
        'name': name,
        'n_configs': len(results),
        'p_train_pos': round(p_train_pos, 2),
        'p_test_pos': round(p_test_pos, 2),
        'p_both_pos': round(p_both_pos, 2),
        'corr_tt': round(corr_tt, 4) if corr_tt is not None else None,
        'median_test': round(median_test, 4),
        'max_test': round(max_test, 4),
        'min_test': round(min_test, 4),
        'best_config': best,
        'potential_score': round(potential, 2),
        'phase2_eligible': p_both_pos >= 5 and (corr_tt or 0) > 0 and potential >= 5,
    }


# ---------- Family-specific entry factories ----------

def factory_alpha():
    return lambda et, bl, ap: make_alpha_entry_param(et, bl, ap)


def factory_iota():
    return lambda et, bl, ap, lb: make_iota_entry_param(et, bl, ap, lb)


def make_kappa_entry(eth_thresh, btc_lag, eth_break_lb):
    """ι entry + mid-vol (30-70 pctile)."""
    def fn(df, h1, h4, valid, params=None):
        n = len(df)
        btc_ret = df['btc_return'].values
        eth_ret = df['eth_return'].values
        eth_close = df['eth_close'].values
        atr = df['atr14'].values
        atr_lo = df['atr_pctile_30'].values if 'atr_pctile_30' in df.columns else rolling_pctile(atr, 200, 30)
        atr_hi = df['atr_pctile_70'].values if 'atr_pctile_70' in df.columns else rolling_pctile(atr, 200, 70)
        if eth_break_lb == 24 and 'eth_high_24_prev' in df.columns:
            eth_high_prev = df['eth_high_24_prev'].values
            eth_low_prev = df['eth_low_24_prev'].values
        else:
            eth_high_prev = pd.Series(eth_close).rolling(eth_break_lb, min_periods=eth_break_lb).max().shift(1).values
            eth_low_prev = pd.Series(eth_close).rolling(eth_break_lb, min_periods=eth_break_lb).min().shift(1).values
        sigs = []
        for i in range(2, n):
            if not valid[i]: continue
            if any(pd.isna(x) for x in (btc_ret[i - 1], eth_ret[i - 1], atr[i], atr_lo[i], atr_hi[i],
                                          eth_close[i], eth_high_prev[i], eth_low_prev[i])): continue
            if not (atr_lo[i] <= atr[i] <= atr_hi[i]): continue
            eth_up = eth_ret[i - 1] > eth_thresh
            btc_lag_up = btc_ret[i - 1] < btc_lag
            eth_break_up = eth_close[i] > eth_high_prev[i]
            eth_down = eth_ret[i - 1] < -eth_thresh
            btc_lag_down = btc_ret[i - 1] > -btc_lag
            eth_break_down = eth_close[i] < eth_low_prev[i]
            if eth_up and btc_lag_up and eth_break_up and h1[i] and h4[i]:
                sigs.append((i, 'LONG'))
            elif eth_down and btc_lag_down and eth_break_down and (not h1[i]) and (not h4[i]):
                sigs.append((i, 'SHORT'))
        return sigs
    return fn


def make_sigma_entry(rsi_thresh, eth_break_lb):
    """Counter-trend mean-rev at structural break."""
    def fn(df, h1, h4, valid, params=None):
        n = len(df)
        rsi = df['rsi14'].values
        btc_ret = df['btc_return'].values
        eth_close = df['eth_close'].values
        if eth_break_lb == 24 and 'eth_high_24_prev' in df.columns:
            eth_high_prev = df['eth_high_24_prev'].values
            eth_low_prev = df['eth_low_24_prev'].values
        else:
            eth_high_prev = pd.Series(eth_close).rolling(eth_break_lb, min_periods=eth_break_lb).max().shift(1).values
            eth_low_prev = pd.Series(eth_close).rolling(eth_break_lb, min_periods=eth_break_lb).min().shift(1).values
        sigs = []
        rsi_hi = rsi_thresh
        rsi_lo = 100 - rsi_thresh
        for i in range(2, n):
            if not valid[i]: continue
            if any(pd.isna(x) for x in (rsi[i], btc_ret[i - 1], eth_close[i], eth_high_prev[i], eth_low_prev[i])):
                continue
            if eth_close[i] > eth_high_prev[i] and rsi[i] >= rsi_hi and btc_ret[i - 1] > 0:
                sigs.append((i, 'SHORT'))
            elif eth_close[i] < eth_low_prev[i] and rsi[i] <= rsi_lo and btc_ret[i - 1] < 0:
                sigs.append((i, 'LONG'))
        return sigs
    return fn


def make_upsilon_entry(vol_mult, eth_thresh):
    """Volume spike + ETH×BTC align."""
    def fn(df, h1, h4, valid, params=None):
        n = len(df)
        vol = df['volume'].values
        vol_sma = df['volume_sma'].values if 'volume_sma' in df.columns else pd.Series(vol).rolling(20, min_periods=20).mean().values
        eth_ret = df['eth_return'].values
        btc_ret = df['btc_return'].values
        sigs = []
        for i in range(2, n):
            if not valid[i]: continue
            if any(pd.isna(x) for x in (vol_sma[i], eth_ret[i - 1], btc_ret[i - 1])): continue
            if vol[i] < vol_mult * vol_sma[i]: continue
            if eth_ret[i - 1] > eth_thresh and btc_ret[i - 1] > 0 and h1[i] and h4[i]:
                sigs.append((i, 'LONG'))
            elif eth_ret[i - 1] < -eth_thresh and btc_ret[i - 1] < 0 and (not h1[i]) and (not h4[i]):
                sigs.append((i, 'SHORT'))
        return sigs
    return fn


def make_zeta_entry(accel_thresh, accel_window=8):
    """ETH return acceleration."""
    def fn(df, h1, h4, valid, params=None):
        n = len(df)
        eth_ret = df['eth_return'].values
        btc_ret = df['btc_return'].values
        accel = pd.Series(eth_ret).diff(accel_window).values
        sigs = []
        for i in range(2, n):
            if not valid[i]: continue
            if any(pd.isna(x) for x in (accel[i], btc_ret[i - 1])): continue
            if accel[i] > accel_thresh and btc_ret[i - 1] > 0 and h1[i] and h4[i]:
                sigs.append((i, 'LONG'))
            elif accel[i] < -accel_thresh and btc_ret[i - 1] < 0 and (not h1[i]) and (not h4[i]):
                sigs.append((i, 'SHORT'))
        return sigs
    return fn


def main():
    print("Loading data (extended for all 6 families)...")
    # Use R8 prep which has mid-vol pctiles + ETH break columns
    df_r8, h1, h4, base_valid, eth_valid_ext, funding_valid, kappa_valid_r8, _ = prepare_data_r8()
    # R6 prep for volume_sma
    df_r6, _, _, _, _, _, upsilon_valid, _ = prepare_data_r6()
    # Merge volume_sma into df_r8
    for col in ['volume_sma']:
        df_r8[col] = df_r6[col].values
    df = df_r8
    eligible_with_filter = (h1 & h4 | (~h1) & (~h4))
    n_total = len(df)
    print(f"  bars: {n_total:,} | days: {n_total/96:.0f}\n")

    family_results = {}

    # ---------- α ----------
    print("=" * 80); print("α — broad sweep"); print("=" * 80)
    et_grid = (0.10, 0.20, 0.30, 0.40, 0.50)
    bl_grid = (0.05, 0.10, 0.15, 0.20)
    ap_grid = (50, 60, 70, 80)
    N_grid = (2, 4, 6, 8, 12)
    param_grid = list(product(et_grid, bl_grid, ap_grid))
    print(f"  {len(param_grid)} param combos × {len(N_grid)} N = {len(param_grid)*len(N_grid)} configs")
    factory = factory_alpha()
    res = run_family_grid(df, h1, h4, eth_valid_ext, eligible_with_filter & eth_valid_ext,
                          factory, param_grid, N_grid, friction=0.04, min_n=30)
    print(f"  valid configs: {len(res)}")
    family_results['α'] = res

    # ---------- ι ----------
    print("\n" + "=" * 80); print("ι — broad sweep (with ETH break lookback)"); print("=" * 80)
    et_grid = (0.20, 0.30, 0.40)
    bl_grid = (0.05, 0.10, 0.15)
    ap_grid = (60, 70, 80)
    lb_grid = (12, 24, 36)
    N_grid_iota = (4, 6, 8, 12, 16)
    param_grid = list(product(et_grid, bl_grid, ap_grid, lb_grid))
    print(f"  {len(param_grid)} param combos × {len(N_grid_iota)} N = {len(param_grid)*len(N_grid_iota)} configs")
    factory = factory_iota()
    res = run_family_grid(df, h1, h4, eth_valid_ext, eligible_with_filter & eth_valid_ext,
                          factory, param_grid, N_grid_iota, friction=0.04, min_n=30)
    print(f"  valid configs: {len(res)}")
    family_results['ι'] = res

    # ---------- κ ----------
    print("\n" + "=" * 80); print("κ — broad sweep (mid-vol regime)"); print("=" * 80)
    et_grid = (0.20, 0.30, 0.40)
    bl_grid = (0.05, 0.10, 0.15)
    lb_grid = (18, 24, 30)
    N_grid_k = (4, 6, 8, 12, 16)
    param_grid = list(product(et_grid, bl_grid, lb_grid))
    print(f"  {len(param_grid)} param combos × {len(N_grid_k)} N = {len(param_grid)*len(N_grid_k)} configs")
    factory = lambda et, bl, lb: make_kappa_entry(et, bl, lb)
    res = run_family_grid(df, h1, h4, eth_valid_ext, eligible_with_filter & eth_valid_ext,
                          factory, param_grid, N_grid_k, friction=0.04, min_n=30)
    print(f"  valid configs: {len(res)}")
    family_results['κ'] = res

    # ---------- σ ----------
    print("\n" + "=" * 80); print("σ — broad sweep (counter-trend at break)"); print("=" * 80)
    rsi_grid = (60, 65, 70, 75)
    lb_grid = (12, 18, 24, 30)
    N_grid_s = (4, 6, 8, 12)
    param_grid = list(product(rsi_grid, lb_grid))
    print(f"  {len(param_grid)} param combos × {len(N_grid_s)} N = {len(param_grid)*len(N_grid_s)} configs")
    factory = lambda rt, lb: make_sigma_entry(rt, lb)
    res = run_family_grid(df, h1, h4, eth_valid_ext, eth_valid_ext,  # counter-trend, no trend filter
                          factory, param_grid, N_grid_s, friction=0.04, min_n=30)
    print(f"  valid configs: {len(res)}")
    family_results['σ'] = res

    # ---------- υ ----------
    print("\n" + "=" * 80); print("υ — broad sweep (volume × cross-asset)"); print("=" * 80)
    vol_grid = (1.5, 2.0, 2.5, 3.0)
    et_grid = (0.10, 0.20, 0.30)
    N_grid_u = (2, 4, 6, 8, 12)
    param_grid = list(product(vol_grid, et_grid))
    print(f"  {len(param_grid)} param combos × {len(N_grid_u)} N = {len(param_grid)*len(N_grid_u)} configs")
    factory = lambda vm, et: make_upsilon_entry(vm, et)
    res = run_family_grid(df, h1, h4, upsilon_valid, eligible_with_filter & upsilon_valid,
                          factory, param_grid, N_grid_u, friction=0.04, min_n=30)
    print(f"  valid configs: {len(res)}")
    family_results['υ'] = res

    # ---------- ζ ----------
    print("\n" + "=" * 80); print("ζ — broad sweep (ETH return acceleration)"); print("=" * 80)
    accel_grid = (0.3, 0.5, 0.7, 1.0)
    N_grid_z = (2, 4, 6, 8, 12)
    param_grid = list(product(accel_grid,))
    print(f"  {len(param_grid)} param combos × {len(N_grid_z)} N = {len(param_grid)*len(N_grid_z)} configs")
    factory = lambda at: make_zeta_entry(at)
    # Compute zeta_valid via R8 prep (eth_accel)
    df['eth_accel'] = pd.Series(df['eth_return'].values).diff(8).values
    zeta_valid = eth_valid_ext & (~pd.isna(df['eth_accel'])).values
    res = run_family_grid(df, h1, h4, zeta_valid, eligible_with_filter & zeta_valid,
                          factory, param_grid, N_grid_z, friction=0.04, min_n=30)
    print(f"  valid configs: {len(res)}")
    family_results['ζ'] = res

    # ---------- POTENTIAL ASSESSMENT ----------
    print("\n" + "=" * 100); print("POTENTIAL ASSESSMENT"); print("=" * 100)
    print(f"{'family':<6} {'n_configs':>10} {'p_train+':>10} {'p_test+':>10} {'p_both+':>10} {'corr':>8} "
          f"{'med_test':>10} {'max_test':>10} {'potential':>11} {'eligible':>10}")
    family_potential = {}
    for fname, res in family_results.items():
        pot = compute_potential(res, fname)
        family_potential[fname] = pot
        if pot.get('n_configs', 0) == 0:
            print(f"{fname:<6} {'0':>10} {'N/A':>10}")
            continue
        print(f"{fname:<6} {pot['n_configs']:>10} {pot['p_train_pos']:>9.1f}% {pot['p_test_pos']:>9.1f}% "
              f"{pot['p_both_pos']:>9.1f}% {pot['corr_tt'] if pot['corr_tt'] is not None else 'N/A':>8} "
              f"{pot['median_test']:>+9.4f}% {pot['max_test']:>+9.4f}% {pot['potential_score']:>10.2f} "
              f"{'YES' if pot['phase2_eligible'] else 'no':>10}")

    # Rank
    eligible_families = [(fname, pot) for fname, pot in family_potential.items() if pot.get('phase2_eligible', False)]
    eligible_families.sort(key=lambda kv: -kv[1]['potential_score'])
    print(f"\n{'=' * 80}\nPHASE-2 ELIGIBLE FAMILIES (sorted by potential)\n{'=' * 80}")
    if eligible_families:
        for fname, pot in eligible_families:
            print(f"  {fname}: potential={pot['potential_score']:.2f}, p_both_pos={pot['p_both_pos']:.1f}%, corr={pot['corr_tt']}")
        winner = eligible_families[0]
        print(f"\n  → Highest-potential: {winner[0]} (proceed to Phase 2 optimization)")
    else:
        print("  → 0 families eligible for Phase 2.")
        print("  → Distribution analysis confirms: 6 families × broad sweep all below potential threshold.")
        print("  → Reinforces R10 finding (selection-from-grid noise)")

    out = {'date': datetime.now(timezone.utc).isoformat(),
           'pre_reg': 'claudedocs/m3_round14_potential_assessment.md',
           'family_results': {k: [{'params': r['params'], 'N': r['N'],
                                     'train_daily': r['train_daily'], 'train_n': r['train_n'],
                                     'test_daily': r['test_daily'], 'test_n': r['test_n']}
                                    for r in v]
                                for k, v in family_results.items()},
           'family_potential': family_potential,
           'eligible_families': [{'name': f, 'pot': p} for f, p in eligible_families]}
    p = ROOT / 'results' / f'm3_r14_potential_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(p, 'w') as f: json.dump(out, f, indent=2, default=str)
    print(f"\nSaved: {p}")


if __name__ == '__main__':
    main()
