"""
M3-R6 — υ (volume×cross-asset) + χ (wick rejection×RSI)
========================================================
2 specs × 5 critiques. User override of advisor "no R6".
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


def prepare_data_r6(volume_sma_period=20):
    df, h1, h4, base_valid, eth_valid, funding_valid = prepare_all_data()
    vol = df['volume'].values
    df['volume_sma'] = pd.Series(vol).rolling(volume_sma_period, min_periods=volume_sma_period).mean().values

    op = df['open'].values; hi = df['high'].values; lo = df['low'].values; cl = df['close'].values
    rng = hi - lo
    rng_safe = np.where(rng > 0, rng, np.nan)
    body_lo = np.minimum(op, cl)
    body_hi = np.maximum(op, cl)
    df['low_wick_ratio'] = (body_lo - lo) / rng_safe
    df['up_wick_ratio'] = (hi - body_hi) / rng_safe
    df['close_pos'] = (cl - lo) / rng_safe  # 0=at low, 1=at high

    upsilon_valid = eth_valid & (~pd.isna(df['volume_sma'])).values
    chi_valid = base_valid & (~pd.isna(df['low_wick_ratio'])).values & (~pd.isna(df['close_pos'])).values
    return df, h1, h4, base_valid, eth_valid, funding_valid, upsilon_valid, chi_valid


# ---------- υ entry ----------

def entry_upsilon(df, h1, h4, valid, params=None):
    p = {'vol_mult': 2.0, 'eth_thresh': 0.2} if params is None else params
    n = len(df)
    vol = df['volume'].values
    vol_sma = df['volume_sma'].values
    eth_ret = df['eth_return'].values
    btc_ret = df['btc_return'].values
    sigs = []
    for i in range(2, n):
        if not valid[i]: continue
        if any(pd.isna(x) for x in (vol_sma[i], eth_ret[i - 1], btc_ret[i - 1])): continue
        if vol[i] < p['vol_mult'] * vol_sma[i]: continue
        # LONG: ETH up + BTC up + trend long
        if eth_ret[i - 1] > p['eth_thresh'] and btc_ret[i - 1] > 0 and h1[i] and h4[i]:
            sigs.append((i, 'LONG'))
        elif eth_ret[i - 1] < -p['eth_thresh'] and btc_ret[i - 1] < 0 and (not h1[i]) and (not h4[i]):
            sigs.append((i, 'SHORT'))
    return sigs


# ---------- χ entry ----------

def entry_chi(df, h1, h4, valid, params=None):
    p = {'wick_thresh': 0.40, 'close_pos_thresh': 0.30, 'rsi_thresh': 35} if params is None else params
    n = len(df)
    low_wick = df['low_wick_ratio'].values
    up_wick = df['up_wick_ratio'].values
    close_pos = df['close_pos'].values
    rsi = df['rsi14'].values
    btc_ret = df['btc_return'].values
    sigs = []
    rsi_lo = p['rsi_thresh']
    rsi_hi = 100 - p['rsi_thresh']
    for i in range(2, n):
        if not valid[i]: continue
        if any(pd.isna(x) for x in (low_wick[i], up_wick[i], close_pos[i], rsi[i], btc_ret[i - 1])): continue
        # LONG: long lower wick + close in upper 30% + RSI oversold + recent down
        if (low_wick[i] >= p['wick_thresh'] and close_pos[i] >= (1 - p['close_pos_thresh'])
                and rsi[i] <= rsi_lo and btc_ret[i - 1] < 0):
            sigs.append((i, 'LONG'))
        # SHORT: long upper wick + close in lower 30% + RSI overbought + recent up
        elif (up_wick[i] >= p['wick_thresh'] and close_pos[i] <= p['close_pos_thresh']
                and rsi[i] >= rsi_hi and btc_ret[i - 1] > 0):
            sigs.append((i, 'SHORT'))
    return sigs


SPECS_R6 = {
    'upsilon': {
        'name': 'υ (volume spike + ETH×BTC align)',
        'entry_fn': entry_upsilon,
        'parameters': {'vol_mult': 2.0, 'eth_thresh': 0.2},
        'sensitivity_params': {
            'vol_mult': [1.6, 2.4],
            'eth_thresh': [0.16, 0.24],
        },
        'valid_mask_key': 'upsilon_valid',
        'eligible_universe_with_filter': True,
        'direction_by_trend': True,
    },
    'chi': {
        'name': 'χ (wick rejection + RSI extreme)',
        'entry_fn': entry_chi,
        'parameters': {'wick_thresh': 0.40, 'close_pos_thresh': 0.30, 'rsi_thresh': 35},
        'sensitivity_params': {
            'wick_thresh': [0.32, 0.48],
            'rsi_thresh': [30, 40],
        },
        'valid_mask_key': 'chi_valid',
        'eligible_universe_with_filter': False,  # counter-trend
        'direction_by_trend': False,
    },
}


def main():
    print("Loading data + R6 columns (volume SMA + wick ratios)...")
    df, h1, h4, base_valid, eth_valid, funding_valid, upsilon_valid, chi_valid = prepare_data_r6()
    print(f"  bars: {len(df):,} | upsilon_valid: {int(upsilon_valid.sum()):,} | chi_valid: {int(chi_valid.sum()):,}\n")

    valid_map = {'upsilon_valid': upsilon_valid, 'chi_valid': chi_valid}
    eligible_with_filter = (h1 & h4 | (~h1) & (~h4))
    eligible_map = {
        'upsilon_valid': eligible_with_filter & upsilon_valid,
        'chi_valid': chi_valid,  # counter-trend, no trend filter on universe
    }

    matrix = {}
    for spec_id, spec in SPECS_R6.items():
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
    print("M3-R6 — 2×5 MATRIX")
    print("=" * 100)
    print(f"{'mechanism':<48} {'C1':>10} {'C2':>10} {'C3':>10} {'C4':>10} {'C5':>10} {'died_at':>10}")
    for spec_id, res in matrix.items():
        spec_name = SPECS_R6[spec_id]['name']
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
        'spec_doc': 'claudedocs/m3_round6_specs.md',
        'matrix': {k: {ck: (v.get('pass') if isinstance(v, dict) else v) for ck, v in res.items()}
                    for k, res in matrix.items()},
        'full_results': matrix,
    }
    p = ROOT / 'results' / f'm3_r6_matrix_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(p, 'w') as f: json.dump(out, f, indent=2, default=str)
    print(f"\nSaved: {p}")


if __name__ == '__main__':
    main()
