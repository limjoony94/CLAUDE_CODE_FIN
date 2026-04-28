"""
M3-R4 — μ (funding momentum) + π (ETH/BTC ratio trend break)
=============================================================
2 specs × 5 critiques = 2×5 matrix.
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


# ---------- Extended data prep ----------

def prepare_data_r4(funding_accel_window=32, ratio_sma_period=20):
    df, h1, h4, base_valid, eth_valid, funding_valid = prepare_all_data()
    fsum = df['funding_8sum'].values
    df['funding_accel'] = pd.Series(fsum).diff(funding_accel_window).values  # f[i]-f[i-32]

    log_ratio = df['log_ratio'].values
    df['ratio_sma'] = pd.Series(log_ratio).rolling(ratio_sma_period, min_periods=ratio_sma_period).mean().values

    mu_valid = funding_valid & (~pd.isna(df['funding_accel'])).values & (~pd.isna(df['eth_return'])).values
    pi_valid = eth_valid & (~pd.isna(df['ratio_sma'])).values
    return df, h1, h4, base_valid, eth_valid, funding_valid, mu_valid, pi_valid


# ---------- μ entry (funding momentum / acceleration) ----------

def entry_mu(df, h1, h4, valid, params=None):
    p = {'funding_accel_window': 32, 'funding_accel_thresh': 0.10} if params is None else params
    n = len(df)
    fsum = df['funding_8sum'].values
    eth_ret = df['eth_return'].values
    fwin = p.get('funding_accel_window', 32)
    if fwin != 32:
        accel = pd.Series(fsum).diff(fwin).values
    else:
        accel = df['funding_accel'].values
    thr = p['funding_accel_thresh']

    sigs = []
    for i in range(2, n):
        if not valid[i]: continue
        if any(pd.isna(x) for x in (accel[i], eth_ret[i - 1])):
            continue
        # LONG: funding 가파르게 하락 (shorts build-up 가속) + ETH up
        if accel[i] < -thr and eth_ret[i - 1] > 0 and h1[i] and h4[i]:
            sigs.append((i, 'LONG'))
        elif accel[i] > thr and eth_ret[i - 1] < 0 and (not h1[i]) and (not h4[i]):
            sigs.append((i, 'SHORT'))
    return sigs


# ---------- π entry (ETH/BTC ratio SMA cross) ----------

def entry_pi(df, h1, h4, valid, params=None):
    p = {'ratio_sma_period': 20} if params is None else params
    n = len(df)
    log_ratio = df['log_ratio'].values
    btc_ret = df['btc_return'].values
    sma_p = p.get('ratio_sma_period', 20)
    if sma_p != 20:
        ratio_sma = pd.Series(log_ratio).rolling(sma_p, min_periods=sma_p).mean().values
    else:
        ratio_sma = df['ratio_sma'].values

    sigs = []
    for i in range(2, n):
        if not valid[i]: continue
        if any(pd.isna(x) for x in (log_ratio[i], log_ratio[i - 1], ratio_sma[i], ratio_sma[i - 1], btc_ret[i - 1])):
            continue
        # LONG: ratio crosses ABOVE SMA (BTC strengthening vs ETH)
        cross_up = (log_ratio[i] > ratio_sma[i]) and (log_ratio[i - 1] <= ratio_sma[i - 1])
        cross_down = (log_ratio[i] < ratio_sma[i]) and (log_ratio[i - 1] >= ratio_sma[i - 1])
        if cross_up and btc_ret[i - 1] > 0 and h1[i] and h4[i]:
            sigs.append((i, 'LONG'))
        elif cross_down and btc_ret[i - 1] < 0 and (not h1[i]) and (not h4[i]):
            sigs.append((i, 'SHORT'))
    return sigs


SPECS_R4 = {
    'mu': {
        'name': 'μ (funding accel + ETH align)',
        'entry_fn': entry_mu,
        'parameters': {'funding_accel_window': 32, 'funding_accel_thresh': 0.10},
        'sensitivity_params': {
            'funding_accel_window': [26, 38],
            'funding_accel_thresh': [0.08, 0.12],
        },
        'valid_mask_key': 'mu_valid',
        'eligible_universe_with_filter': True,
        'direction_by_trend': True,
    },
    'pi': {
        'name': 'π (BTC/ETH ratio SMA cross)',
        'entry_fn': entry_pi,
        'parameters': {'ratio_sma_period': 20},
        'sensitivity_params': {
            'ratio_sma_period': [16, 24],
        },
        'valid_mask_key': 'pi_valid',
        'eligible_universe_with_filter': True,
        'direction_by_trend': True,
    },
}


def main():
    print("Loading data + R4 columns (funding accel + ratio SMA)...")
    df, h1, h4, base_valid, eth_valid, funding_valid, mu_valid, pi_valid = prepare_data_r4()
    print(f"  bars: {len(df):,} | mu_valid: {int(mu_valid.sum()):,} | pi_valid: {int(pi_valid.sum()):,}\n")

    valid_map = {'mu_valid': mu_valid, 'pi_valid': pi_valid}
    eligible_map = {
        'mu_valid': (h1 & h4 | (~h1) & (~h4)) & mu_valid,
        'pi_valid': (h1 & h4 | (~h1) & (~h4)) & pi_valid,
    }

    matrix = {}
    for spec_id, spec in SPECS_R4.items():
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
            matrix[spec_id] = results
            continue

        print("  C2 look-ahead audit...")
        c2 = critique_lookahead_audit(df, h1, h4, spec, valid, eligible)
        results['C2'] = c2
        print(f"     pass={c2['pass']} metrics: {c2['metrics']}")
        if not c2['pass']:
            results['skipped'] = ['C3', 'C4', 'C5']
            matrix[spec_id] = results
            continue

        print("  C3 friction stress...")
        c3 = critique_friction_stress(df, h1, h4, spec, valid)
        results['C3'] = c3
        print(f"     pass={c3['pass']} metrics: {c3['metrics']}")
        if not c3['pass']:
            results['skipped'] = ['C4', 'C5']
            matrix[spec_id] = results
            continue

        print("  C4 overfitting probe...")
        c4 = critique_overfitting_probe(df, h1, h4, spec, valid)
        results['C4'] = c4
        print(f"     pass={c4['pass']} metrics: {c4['metrics']}")
        if not c4['pass']:
            results['skipped'] = ['C5']
            matrix[spec_id] = results
            continue

        print("  C5 bootstrap 3-day (200 windows)...")
        c5 = critique_bootstrap_3day(df, h1, h4, spec, valid, n_bootstrap=200)
        results['C5'] = c5
        print(f"     pass={c5['pass']} metrics: {c5['metrics']}")
        matrix[spec_id] = results

    print("\n" + "=" * 100)
    print("M3-R4 — 2×5 MATRIX")
    print("=" * 100)
    print(f"{'mechanism':<48} {'C1':>10} {'C2':>10} {'C3':>10} {'C4':>10} {'C5':>10} {'died_at':>10}")
    for spec_id, res in matrix.items():
        spec_name = SPECS_R4[spec_id]['name']
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
        'spec_doc': 'claudedocs/m3_round4_specs.md',
        'matrix': {k: {ck: (v.get('pass') if isinstance(v, dict) else v) for ck, v in res.items()}
                    for k, res in matrix.items()},
        'full_results': matrix,
    }
    p = ROOT / 'results' / f'm3_r4_matrix_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(p, 'w') as f: json.dump(out, f, indent=2, default=str)
    print(f"\nSaved: {p}")


if __name__ == '__main__':
    main()
