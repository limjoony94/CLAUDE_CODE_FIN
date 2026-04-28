"""
M3-R3 — ν (volatility transition) + ξ (funding × ETH break compound)
=====================================================================
2 specs × 5 critiques = 2×5 matrix.
Pipeline 재활용 (m3_critique_pipeline.py).
"""
import sys, json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / 'scripts' / 'analysis'))

from m3_critique_pipeline import (prepare_all_data, EXIT_PARAMS,
                                    run_bt_with_spec, trade_summary,
                                    critique_random_baseline, critique_lookahead_audit,
                                    critique_friction_stress, critique_overfitting_probe,
                                    critique_bootstrap_3day)


# ---------- Extended data prep: ATR SMA50 + ETH break columns ----------

def prepare_data_r3(atr_sma_period=50, eth_break_lookback=24):
    df, h1, h4, base_valid, eth_valid, funding_valid = prepare_all_data()
    atr = df['atr14'].values
    df['atr_sma'] = pd.Series(atr).rolling(atr_sma_period, min_periods=atr_sma_period).mean().values
    # ν transition cross detection requires atr_sma_prev (i-1) — handled inline

    eth_close = df['eth_close'].values
    df['eth_high_24_prev'] = pd.Series(eth_close).rolling(eth_break_lookback, min_periods=eth_break_lookback).max().shift(1).values
    df['eth_low_24_prev'] = pd.Series(eth_close).rolling(eth_break_lookback, min_periods=eth_break_lookback).min().shift(1).values

    nu_valid = base_valid & (~pd.isna(df['atr_sma'])).values
    xi_valid = funding_valid & (~pd.isna(df['eth_high_24_prev'])).values & (~pd.isna(df['eth_low_24_prev'])).values & (~pd.isna(df['eth_close'])).values
    return df, h1, h4, base_valid, eth_valid, funding_valid, nu_valid, xi_valid


# ---------- ν entry (volatility regime transition) ----------

def entry_nu(df, h1, h4, valid, params=None):
    p = {'atr_sma_period': 50, 'btc_lag_thresh': 0.0} if params is None else params
    n = len(df)
    atr = df['atr14'].values
    btc_ret = df['btc_return'].values
    # Recompute SMA if param changed
    sma_period = p.get('atr_sma_period', 50)
    if sma_period != 50:
        atr_sma = pd.Series(atr).rolling(sma_period, min_periods=sma_period).mean().values
    else:
        atr_sma = df['atr_sma'].values

    sigs = []
    for i in range(2, n):
        if not valid[i]: continue
        if any(pd.isna(x) for x in (atr[i], atr[i - 1], atr_sma[i], atr_sma[i - 1], btc_ret[i - 1])):
            continue
        # Transition: ATR crosses ABOVE SMA
        cross_up = (atr[i] > atr_sma[i]) and (atr[i - 1] <= atr_sma[i - 1])
        if not cross_up:
            continue
        # Direction by BTC return + trend
        thr = p.get('btc_lag_thresh', 0.0)
        if btc_ret[i - 1] > thr and h1[i] and h4[i]:
            sigs.append((i, 'LONG'))
        elif btc_ret[i - 1] < -thr and (not h1[i]) and (not h4[i]):
            sigs.append((i, 'SHORT'))
    return sigs


# ---------- ξ entry (funding × ETH break compound) ----------

def entry_xi(df, h1, h4, valid, params=None):
    p = {'funding_sum_thresh': 0.24, 'eth_break_lookback': 24} if params is None else params
    n = len(df)
    fsum = df['funding_8sum'].values
    eth_close = df['eth_close'].values
    eth_high_prev = df['eth_high_24_prev'].values
    eth_low_prev = df['eth_low_24_prev'].values
    if p.get('eth_break_lookback', 24) != 24:
        lb = p['eth_break_lookback']
        eth_high_prev = pd.Series(eth_close).rolling(lb, min_periods=lb).max().shift(1).values
        eth_low_prev = pd.Series(eth_close).rolling(lb, min_periods=lb).min().shift(1).values

    sigs = []
    for i in range(1, n):
        if not valid[i]: continue
        if any(pd.isna(x) for x in (fsum[i], eth_close[i], eth_high_prev[i], eth_low_prev[i])):
            continue
        thr = p['funding_sum_thresh']
        # LONG: shorts crowded + ETH break up + trend long
        if fsum[i] <= -thr and eth_close[i] > eth_high_prev[i] and h1[i] and h4[i]:
            sigs.append((i, 'LONG'))
        elif fsum[i] >= thr and eth_close[i] < eth_low_prev[i] and (not h1[i]) and (not h4[i]):
            sigs.append((i, 'SHORT'))
    return sigs


# ---------- Spec definitions ----------

SPECS_R3 = {
    'nu': {
        'name': 'ν (vol transition + return + trend)',
        'entry_fn': entry_nu,
        'parameters': {'atr_sma_period': 50, 'btc_lag_thresh': 0.0},
        'sensitivity_params': {
            'atr_sma_period': [40, 60],
            'btc_lag_thresh': [-0.05, 0.05],
        },
        'valid_mask_key': 'nu_valid',
        'eligible_universe_with_filter': True,
        'direction_by_trend': True,
    },
    'xi': {
        'name': 'ξ (funding extreme × ETH break compound)',
        'entry_fn': entry_xi,
        'parameters': {'funding_sum_thresh': 0.24, 'eth_break_lookback': 24},
        'sensitivity_params': {
            'funding_sum_thresh': [0.19, 0.29],
            'eth_break_lookback': [19, 29],
        },
        'valid_mask_key': 'xi_valid',
        'eligible_universe_with_filter': True,
        'direction_by_trend': True,
    },
}


def main():
    print("Loading data + R3 columns (ATR SMA50 + ETH break 24)...")
    df, h1, h4, base_valid, eth_valid, funding_valid, nu_valid, xi_valid = prepare_data_r3()
    print(f"  bars: {len(df):,} | nu_valid: {int(nu_valid.sum()):,} | xi_valid: {int(xi_valid.sum()):,}\n")

    valid_map = {'nu_valid': nu_valid, 'xi_valid': xi_valid}
    eligible_with_filter_nu = (h1 & h4 | (~h1) & (~h4)) & nu_valid
    eligible_with_filter_xi = (h1 & h4 | (~h1) & (~h4)) & xi_valid
    eligible_map = {'nu_valid': eligible_with_filter_nu, 'xi_valid': eligible_with_filter_xi}

    matrix = {}
    for spec_id, spec in SPECS_R3.items():
        print("=" * 80); print(f"MECHANISM {spec_id}: {spec['name']}"); print("=" * 80)
        valid = valid_map[spec['valid_mask_key']]
        eligible = eligible_map[spec['valid_mask_key']]
        results = {}

        # C1
        print("  C1 random baseline...")
        c1 = critique_random_baseline(df, h1, h4, spec, valid, eligible)
        results['C1'] = c1
        print(f"     pass={c1['pass']} metrics: {c1['metrics']}")
        if not c1['pass']:
            results['skipped'] = ['C2', 'C3', 'C4', 'C5']
            matrix[spec_id] = results
            continue

        # C2
        print("  C2 look-ahead audit...")
        c2 = critique_lookahead_audit(df, h1, h4, spec, valid, eligible)
        results['C2'] = c2
        print(f"     pass={c2['pass']} metrics: {c2['metrics']}")
        if not c2['pass']:
            results['skipped'] = ['C3', 'C4', 'C5']
            matrix[spec_id] = results
            continue

        # C3
        print("  C3 friction stress...")
        c3 = critique_friction_stress(df, h1, h4, spec, valid)
        results['C3'] = c3
        print(f"     pass={c3['pass']} metrics: {c3['metrics']}")
        if not c3['pass']:
            results['skipped'] = ['C4', 'C5']
            matrix[spec_id] = results
            continue

        # C4
        print("  C4 overfitting probe...")
        c4 = critique_overfitting_probe(df, h1, h4, spec, valid)
        results['C4'] = c4
        print(f"     pass={c4['pass']} metrics: {c4['metrics']}")
        if not c4['pass']:
            results['skipped'] = ['C5']
            matrix[spec_id] = results
            continue

        # C5
        print("  C5 bootstrap 3-day (200 windows)...")
        c5 = critique_bootstrap_3day(df, h1, h4, spec, valid, n_bootstrap=200)
        results['C5'] = c5
        print(f"     pass={c5['pass']} metrics: {c5['metrics']}")
        matrix[spec_id] = results
        print()

    # 2×5 matrix
    print("=" * 100)
    print("M3-R3 — 2×5 MATRIX")
    print("=" * 100)
    print(f"{'mechanism':<48} {'C1':>10} {'C2':>10} {'C3':>10} {'C4':>10} {'C5':>10} {'died_at':>10}")
    for spec_id, res in matrix.items():
        spec_name = SPECS_R3[spec_id]['name']
        cells = []; died_at = '-'
        for ck in ['C1', 'C2', 'C3', 'C4', 'C5']:
            if ck not in res:
                cells.append('skip')
            else:
                cells.append('PASS' if res[ck]['pass'] else 'FAIL')
                if not res[ck]['pass'] and died_at == '-':
                    died_at = ck
        print(f"{spec_name:<48} " + " ".join(f"{c:>10}" for c in cells) + f" {died_at:>10}")

    out = {
        'date': datetime.now(timezone.utc).isoformat(),
        'spec_doc': 'claudedocs/m3_round3_specs.md',
        'matrix': {k: {ck: (v.get('pass') if isinstance(v, dict) else v) for ck, v in res.items()}
                    for k, res in matrix.items()},
        'full_results': matrix,
    }
    p = ROOT / 'results' / f'm3_r3_matrix_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(p, 'w') as f: json.dump(out, f, indent=2, default=str)
    print(f"\nSaved: {p}")


if __name__ == '__main__':
    main()
