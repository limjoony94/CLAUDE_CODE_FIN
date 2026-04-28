"""M3-R8 — κ mid-vol regime × ι + ζ ETH return acceleration."""
import sys, json
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / 'scripts' / 'analysis'))

from m3_critique_pipeline import (prepare_all_data, run_bt_with_spec, trade_summary,
                                    critique_random_baseline, critique_lookahead_audit,
                                    critique_friction_stress, critique_overfitting_probe,
                                    critique_bootstrap_3day, rolling_pctile)
from m3_round2_critique import prepare_data_with_eth_break


def prepare_data_r8(eth_break_lookback=24, eth_accel_window=8):
    df, h1, h4, base_valid, eth_valid_ext, funding_valid = prepare_data_with_eth_break(lookback=eth_break_lookback)
    atr = df['atr14'].values
    # Mid-vol band: 30-70th percentile (NOT in α's >70th, NOT below)
    df['atr_pctile_30'] = rolling_pctile(atr, 200, 30)
    df['atr_pctile_70'] = rolling_pctile(atr, 200, 70)

    # ETH return acceleration = ETH return now − ETH return eth_accel_window bars ago
    eth_ret = df['eth_return'].values
    df['eth_accel'] = pd.Series(eth_ret).diff(eth_accel_window).values

    kappa_valid = eth_valid_ext & (~pd.isna(df['atr_pctile_30'])).values & (~pd.isna(df['atr_pctile_70'])).values
    zeta_valid = eth_valid_ext & (~pd.isna(df['eth_accel'])).values
    return df, h1, h4, base_valid, eth_valid_ext, funding_valid, kappa_valid, zeta_valid


# κ: ι entry + mid-vol regime instead of high-vol (α uses >70th)
def entry_kappa(df, h1, h4, valid, params=None):
    p = {'eth_thresh': 0.3, 'btc_lag_thresh': 0.1, 'eth_break_lookback': 24} if params is None else params
    n = len(df)
    btc_ret = df['btc_return'].values
    eth_ret = df['eth_return'].values
    eth_close = df['eth_close'].values
    atr = df['atr14'].values
    atr_lo = df['atr_pctile_30'].values
    atr_hi = df['atr_pctile_70'].values
    eth_high_prev = df['eth_high_24_prev'].values
    eth_low_prev = df['eth_low_24_prev'].values
    if p.get('eth_break_lookback', 24) != 24:
        lb = p['eth_break_lookback']
        eth_high_prev = pd.Series(eth_close).rolling(lb, min_periods=lb).max().shift(1).values
        eth_low_prev = pd.Series(eth_close).rolling(lb, min_periods=lb).min().shift(1).values

    sigs = []
    for i in range(2, n):
        if not valid[i]: continue
        if any(pd.isna(x) for x in (btc_ret[i - 1], eth_ret[i - 1], atr[i], atr_lo[i], atr_hi[i],
                                      eth_close[i], eth_high_prev[i], eth_low_prev[i])):
            continue
        # MID-VOL REGIME: 30 ≤ atr ≤ 70 percentile
        if not (atr_lo[i] <= atr[i] <= atr_hi[i]): continue
        eth_up = eth_ret[i - 1] > p['eth_thresh']
        btc_lag_up = btc_ret[i - 1] < p['btc_lag_thresh']
        eth_break_up = eth_close[i] > eth_high_prev[i]
        eth_down = eth_ret[i - 1] < -p['eth_thresh']
        btc_lag_down = btc_ret[i - 1] > -p['btc_lag_thresh']
        eth_break_down = eth_close[i] < eth_low_prev[i]
        if eth_up and btc_lag_up and eth_break_up and h1[i] and h4[i]:
            sigs.append((i, 'LONG'))
        elif eth_down and btc_lag_down and eth_break_down and (not h1[i]) and (not h4[i]):
            sigs.append((i, 'SHORT'))
    return sigs


# ζ: ETH return acceleration alone + cross-asset trend
def entry_zeta(df, h1, h4, valid, params=None):
    p = {'eth_accel_thresh': 0.5} if params is None else params
    n = len(df)
    accel = df['eth_accel'].values
    btc_ret = df['btc_return'].values
    sigs = []
    for i in range(2, n):
        if not valid[i]: continue
        if any(pd.isna(x) for x in (accel[i], btc_ret[i - 1])): continue
        # LONG: ETH accelerating up + BTC also up + trend long
        if accel[i] > p['eth_accel_thresh'] and btc_ret[i - 1] > 0 and h1[i] and h4[i]:
            sigs.append((i, 'LONG'))
        elif accel[i] < -p['eth_accel_thresh'] and btc_ret[i - 1] < 0 and (not h1[i]) and (not h4[i]):
            sigs.append((i, 'SHORT'))
    return sigs


SPECS_R8 = {
    'kappa': {
        'name': 'κ (ι entry + MID-vol regime 30-70 pctile)',
        'entry_fn': entry_kappa,
        'parameters': {'eth_thresh': 0.3, 'btc_lag_thresh': 0.1, 'eth_break_lookback': 24},
        'sensitivity_params': {
            'eth_thresh': [0.24, 0.36],
            'btc_lag_thresh': [0.08, 0.12],
        },
        'valid_mask_key': 'kappa_valid',
        'eligible_universe_with_filter': True,
        'direction_by_trend': True,
    },
    'zeta': {
        'name': 'ζ (ETH return acceleration + BTC align)',
        'entry_fn': entry_zeta,
        'parameters': {'eth_accel_thresh': 0.5},
        'sensitivity_params': {'eth_accel_thresh': [0.4, 0.6]},
        'valid_mask_key': 'zeta_valid',
        'eligible_universe_with_filter': True,
        'direction_by_trend': True,
    },
}


def main():
    print("Loading data + R8 columns...")
    df, h1, h4, base_valid, eth_valid_ext, funding_valid, kappa_valid, zeta_valid = prepare_data_r8()
    print(f"  bars: {len(df):,} | kappa_valid: {int(kappa_valid.sum()):,} | zeta_valid: {int(zeta_valid.sum()):,}\n")

    valid_map = {'kappa_valid': kappa_valid, 'zeta_valid': zeta_valid}
    eligible_with_filter = (h1 & h4 | (~h1) & (~h4))
    eligible_map = {
        'kappa_valid': eligible_with_filter & kappa_valid,
        'zeta_valid': eligible_with_filter & zeta_valid,
    }

    matrix = {}
    for spec_id, spec in SPECS_R8.items():
        print("=" * 80); print(f"MECHANISM {spec_id}: {spec['name']}"); print("=" * 80)
        valid = valid_map[spec['valid_mask_key']]
        eligible = eligible_map[spec['valid_mask_key']]
        results = {}

        print("  C1 random baseline...")
        c1 = critique_random_baseline(df, h1, h4, spec, valid, eligible)
        results['C1'] = c1
        print(f"     pass={c1['pass']} metrics: {c1['metrics']}")
        if not c1['pass']:
            results['skipped'] = ['C2', 'C3', 'C4', 'C5']
            matrix[spec_id] = results; continue

        print("  C2 look-ahead audit...")
        c2 = critique_lookahead_audit(df, h1, h4, spec, valid, eligible)
        results['C2'] = c2
        print(f"     pass={c2['pass']} metrics: {c2['metrics']}")
        if not c2['pass']:
            results['skipped'] = ['C3', 'C4', 'C5']
            matrix[spec_id] = results; continue

        print("  C3 friction stress...")
        c3 = critique_friction_stress(df, h1, h4, spec, valid)
        results['C3'] = c3
        print(f"     pass={c3['pass']} metrics: {c3['metrics']}")
        if not c3['pass']:
            results['skipped'] = ['C4', 'C5']
            matrix[spec_id] = results; continue

        print("  C4 overfitting probe...")
        c4 = critique_overfitting_probe(df, h1, h4, spec, valid)
        results['C4'] = c4
        print(f"     pass={c4['pass']} metrics: {c4['metrics']}")
        if not c4['pass']:
            results['skipped'] = ['C5']
            matrix[spec_id] = results; continue

        print("  C5 bootstrap 3-day...")
        c5 = critique_bootstrap_3day(df, h1, h4, spec, valid, n_bootstrap=200)
        results['C5'] = c5
        print(f"     pass={c5['pass']} metrics: {c5['metrics']}")
        matrix[spec_id] = results

    print("\n" + "=" * 100)
    print("M3-R8 — 2×5 MATRIX")
    print("=" * 100)
    print(f"{'mechanism':<48} {'C1':>10} {'C2':>10} {'C3':>10} {'C4':>10} {'C5':>10} {'died_at':>10}")
    for spec_id, res in matrix.items():
        spec_name = SPECS_R8[spec_id]['name']
        cells = []; died_at = '-'
        for ck in ['C1', 'C2', 'C3', 'C4', 'C5']:
            if ck not in res:
                cells.append('skip')
            else:
                cells.append('PASS' if res[ck]['pass'] else 'FAIL')
                if not res[ck]['pass'] and died_at == '-':
                    died_at = ck
        print(f"{spec_name:<48} " + " ".join(f"{c:>10}" for c in cells) + f" {died_at:>10}")

    out = {'date': datetime.now(timezone.utc).isoformat(),
           'spec_doc': 'claudedocs/m3_round8_specs (inline)',
           'matrix': {k: {ck: (v.get('pass') if isinstance(v, dict) else v) for ck, v in res.items()}
                       for k, res in matrix.items()},
           'full_results': matrix}
    p = ROOT / 'results' / f'm3_r8_matrix_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(p, 'w') as f: json.dump(out, f, indent=2, default=str)
    print(f"\nSaved: {p}")


if __name__ == '__main__':
    main()
