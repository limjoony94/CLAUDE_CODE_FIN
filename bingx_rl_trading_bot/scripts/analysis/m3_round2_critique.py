"""
M3-R2 — α′ (Path A: data-grounded exit change) + ι (Path B: ETH break filter)
==============================================================================
Pipeline 재활용 (m3_critique_pipeline.py).
2 specs × 5 critiques = 2×5 matrix.
"""
import sys, json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / 'scripts' / 'analysis'))

from m3_critique_pipeline import (prepare_all_data, SPECS, EXIT_PARAMS,
                                    entry_alpha, run_bt_with_spec, trade_summary,
                                    critique_random_baseline, critique_lookahead_audit,
                                    critique_friction_stress, critique_overfitting_probe,
                                    critique_bootstrap_3day)
from m2_round2_screening import rolling_max, rolling_min_arr


# ---------- Extended data prep: ETH high/low rolling ----------

def prepare_data_with_eth_break(lookback=24):
    """Same as prepare_all_data but adds ETH 24-bar high/low for ι."""
    df, h1, h4, base_valid, eth_valid, funding_valid = prepare_all_data()
    eth_close = df['eth_close'].values
    # ETH high/low approximated by close (since we resampled ETH 5m → 15m last close only)
    df['eth_high_24_prev'] = pd.Series(eth_close).rolling(lookback, min_periods=lookback).max().shift(1).values
    df['eth_low_24_prev'] = pd.Series(eth_close).rolling(lookback, min_periods=lookback).min().shift(1).values

    # Update eth_valid to include new columns
    eth_valid_ext = eth_valid & (~pd.isna(df['eth_high_24_prev'])).values & (~pd.isna(df['eth_low_24_prev'])).values
    return df, h1, h4, base_valid, eth_valid_ext, funding_valid


# ---------- Spec ι entry function ----------

def entry_iota(df, h1, h4, valid, params=None):
    """ι: α 조건 + ETH 24-bar high(LONG)/low(SHORT) break filter."""
    p = {'eth_thresh': 0.3, 'btc_lag_thresh': 0.1, 'atr_pctile': 70.0, 'eth_break_lookback': 24} if params is None else params
    n = len(df)
    btc_ret = df['btc_return'].values
    eth_ret = df['eth_return'].values
    eth_close = df['eth_close'].values
    atr = df['atr14'].values
    atr_pctile_col = df['atr_pctile_70_200'].values
    eth_high_prev = df['eth_high_24_prev'].values
    eth_low_prev = df['eth_low_24_prev'].values

    if params and params.get('atr_pctile', 70) != 70:
        from m3_critique_pipeline import rolling_pctile
        atr_pctile_col = rolling_pctile(atr, 200, params['atr_pctile'])
    if params and params.get('eth_break_lookback', 24) != 24:
        lb = params['eth_break_lookback']
        eth_high_prev = pd.Series(eth_close).rolling(lb, min_periods=lb).max().shift(1).values
        eth_low_prev = pd.Series(eth_close).rolling(lb, min_periods=lb).min().shift(1).values

    sigs = []
    for i in range(2, n):
        if not valid[i]: continue
        if any(pd.isna(x) for x in (btc_ret[i - 1], eth_ret[i - 1], atr[i], atr_pctile_col[i],
                                      eth_high_prev[i], eth_low_prev[i])):
            continue
        # Regime gate
        if not (atr[i] > atr_pctile_col[i]): continue

        eth_up = eth_ret[i - 1] > p['eth_thresh']
        btc_lag_up = btc_ret[i - 1] < p['btc_lag_thresh']
        eth_break_up = eth_close[i] > eth_high_prev[i]  # ETH 자체 24-bar high break

        eth_down = eth_ret[i - 1] < -p['eth_thresh']
        btc_lag_down = btc_ret[i - 1] > -p['btc_lag_thresh']
        eth_break_down = eth_close[i] < eth_low_prev[i]

        if eth_up and btc_lag_up and eth_break_up and h1[i] and h4[i]:
            sigs.append((i, 'LONG'))
        elif eth_down and btc_lag_down and eth_break_down and (not h1[i]) and (not h4[i]):
            sigs.append((i, 'SHORT'))
    return sigs


# ---------- Spec definitions ----------

# α′ exit params: NO trail, NO SL, only emergency + N=16 timeout
ALPHA_PRIME_EXIT_PARAMS = {
    'use_sl': False,
    'use_trail': False,
    'sl_atr_mult': 0.0,  # ignored
    'trail_k': 0.0,  # ignored
    'emergency_pct': 1.5,
    'timeout_bars': 16,
    'min_bars_between': 2,
}


SPECS_R2 = {
    'alpha_prime': {
        'name': 'α′ (alpha entry + N=16 fixed exit)',
        'entry_fn': entry_alpha,  # 동일
        'parameters': {'eth_thresh': 0.3, 'btc_lag_thresh': 0.1, 'atr_pctile': 70.0},
        'sensitivity_params': {
            'eth_thresh': [0.24, 0.36],
            'btc_lag_thresh': [0.08, 0.12],
            'atr_pctile': [60.0, 80.0],
        },
        'valid_mask_key': 'eth_valid',
        'eligible_universe_with_filter': True,
        'direction_by_trend': True,
        'exit_params': ALPHA_PRIME_EXIT_PARAMS,
    },
    'iota': {
        'name': 'ι (alpha entry + ETH 24-bar break filter)',
        'entry_fn': entry_iota,
        'parameters': {'eth_thresh': 0.3, 'btc_lag_thresh': 0.1, 'atr_pctile': 70.0, 'eth_break_lookback': 24},
        'sensitivity_params': {
            'eth_thresh': [0.24, 0.36],
            'btc_lag_thresh': [0.08, 0.12],
            'atr_pctile': [60.0, 80.0],
            'eth_break_lookback': [19, 29],
        },
        'valid_mask_key': 'eth_valid',
        'eligible_universe_with_filter': True,
        'direction_by_trend': True,
        # ι uses standard exit (test if entry filter alone helps)
    },
}


def main():
    print("Loading data + ETH 24-bar break columns...")
    df, h1, h4, base_valid, eth_valid, funding_valid = prepare_data_with_eth_break(lookback=24)
    print(f"  bars: {len(df):,} | eth_valid_ext: {int(eth_valid.sum()):,}\n")

    valid_map = {'eth_valid': eth_valid, 'funding_valid': funding_valid}
    eligible_with_filter_eth = (h1 & h4 | (~h1) & (~h4)) & eth_valid

    matrix = {}
    for spec_id, spec in SPECS_R2.items():
        print("=" * 80); print(f"MECHANISM {spec_id}: {spec['name']}"); print("=" * 80)
        valid = valid_map[spec['valid_mask_key']]
        eligible = eligible_with_filter_eth if spec['valid_mask_key'] == 'eth_valid' else valid

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
        print("  C5 bootstrap 3-day (200 windows for compute, scaled from 1000)...")
        c5 = critique_bootstrap_3day(df, h1, h4, spec, valid, n_bootstrap=200)
        results['C5'] = c5
        print(f"     pass={c5['pass']} metrics: {c5['metrics']}")

        matrix[spec_id] = results
        print()

    # 2×5 matrix
    print("=" * 100)
    print("M3-R2 — 2×5 MATRIX (per-spec fail-fast)")
    print("=" * 100)
    print(f"{'mechanism':<48} {'C1':>10} {'C2':>10} {'C3':>10} {'C4':>10} {'C5':>10} {'died_at':>10}")
    for spec_id, res in matrix.items():
        spec_name = SPECS_R2[spec_id]['name']
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
        'spec_doc': 'claudedocs/m3_round2_specs.md',
        'matrix': {k: {ck: (v.get('pass') if isinstance(v, dict) else v) for ck, v in res.items()}
                    for k, res in matrix.items()},
        'full_results': matrix,
    }
    p = ROOT / 'results' / f'm3_r2_matrix_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(p, 'w') as f: json.dump(out, f, indent=2, default=str)
    print(f"\nSaved: {p}")


if __name__ == '__main__':
    main()
