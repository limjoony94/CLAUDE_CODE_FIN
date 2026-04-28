"""M3-R7 — ψ funding settlement timing + τ 3-bar reversal."""
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
                                    critique_bootstrap_3day)


def prepare_data_r7():
    df, h1, h4, base_valid, eth_valid, funding_valid = prepare_all_data()
    df['hour_utc'] = pd.to_datetime(df['timestamp']).dt.hour.values
    psi_valid = funding_valid
    tau_valid = eth_valid
    return df, h1, h4, base_valid, eth_valid, funding_valid, psi_valid, tau_valid


def entry_psi(df, h1, h4, valid, params=None):
    p = {'funding_sum_thresh': 0.24} if params is None else params
    settlement_hours = {7, 15, 23}
    n = len(df)
    fsum = df['funding_8sum'].values
    btc_ret = df['btc_return'].values
    hour = df['hour_utc'].values
    sigs = []
    for i in range(2, n):
        if not valid[i]: continue
        if hour[i] not in settlement_hours: continue
        if any(pd.isna(x) for x in (fsum[i], btc_ret[i - 1])): continue
        if fsum[i] <= -p['funding_sum_thresh'] and btc_ret[i - 1] > 0 and h1[i] and h4[i]:
            sigs.append((i, 'LONG'))
        elif fsum[i] >= p['funding_sum_thresh'] and btc_ret[i - 1] < 0 and (not h1[i]) and (not h4[i]):
            sigs.append((i, 'SHORT'))
    return sigs


def entry_tau(df, h1, h4, valid, params=None):
    p = {'eth_thresh': 0.0} if params is None else params
    n = len(df)
    op = df['open'].values; cl = df['close'].values
    eth_ret = df['eth_return'].values
    sigs = []
    for i in range(4, n):
        if not valid[i]: continue
        # 3 prior bars all down
        prev_down = all(cl[i - k] < op[i - k] for k in (1, 2, 3))
        prev_up = all(cl[i - k] > op[i - k] for k in (1, 2, 3))
        if pd.isna(eth_ret[i - 1]): continue
        # Current bar reversal up (close > open AND close > prev 3 closes max)
        max_prev_close = max(cl[i - 1], cl[i - 2], cl[i - 3])
        min_prev_close = min(cl[i - 1], cl[i - 2], cl[i - 3])
        if prev_down and cl[i] > op[i] and cl[i] > max_prev_close and eth_ret[i - 1] > p['eth_thresh']:
            sigs.append((i, 'LONG'))
        elif prev_up and cl[i] < op[i] and cl[i] < min_prev_close and eth_ret[i - 1] < -p['eth_thresh']:
            sigs.append((i, 'SHORT'))
    return sigs


SPECS_R7 = {
    'psi': {
        'name': 'ψ (funding extreme × pre-settlement window)',
        'entry_fn': entry_psi,
        'parameters': {'funding_sum_thresh': 0.24},
        'sensitivity_params': {'funding_sum_thresh': [0.19, 0.29]},
        'valid_mask_key': 'psi_valid',
        'eligible_universe_with_filter': True,
        'direction_by_trend': True,
    },
    'tau': {
        'name': 'τ (3-bar reversal + ETH align)',
        'entry_fn': entry_tau,
        'parameters': {'eth_thresh': 0.0},
        'sensitivity_params': {'eth_thresh': [-0.05, 0.05]},
        'valid_mask_key': 'tau_valid',
        'eligible_universe_with_filter': False,
        'direction_by_trend': False,
    },
}


def main():
    print("Loading data + R7...")
    df, h1, h4, base_valid, eth_valid, funding_valid, psi_valid, tau_valid = prepare_data_r7()
    print(f"  bars: {len(df):,} | psi_valid: {int(psi_valid.sum()):,} | tau_valid: {int(tau_valid.sum()):,}\n")

    valid_map = {'psi_valid': psi_valid, 'tau_valid': tau_valid}
    eligible_with_filter = (h1 & h4 | (~h1) & (~h4))
    eligible_map = {
        'psi_valid': eligible_with_filter & psi_valid,
        'tau_valid': tau_valid,
    }

    matrix = {}
    for spec_id, spec in SPECS_R7.items():
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
    print("M3-R7 — 2×5 MATRIX")
    print("=" * 100)
    print(f"{'mechanism':<48} {'C1':>10} {'C2':>10} {'C3':>10} {'C4':>10} {'C5':>10} {'died_at':>10}")
    for spec_id, res in matrix.items():
        spec_name = SPECS_R7[spec_id]['name']
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
           'spec_doc': 'claudedocs/m3_round7_specs.md',
           'matrix': {k: {ck: (v.get('pass') if isinstance(v, dict) else v) for ck, v in res.items()}
                       for k, res in matrix.items()},
           'full_results': matrix}
    p = ROOT / 'results' / f'm3_r7_matrix_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(p, 'w') as f: json.dump(out, f, indent=2, default=str)
    print(f"\nSaved: {p}")


if __name__ == '__main__':
    main()
