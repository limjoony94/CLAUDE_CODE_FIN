"""
M3-R5 — ρ×ι (session-filtered ι) + σ (mean-rev at structural break)
====================================================================
LAST round per advisor. 2 specs × 5 critiques.
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
from m2_round1_screening import measure_mfe_for_signals, stats_mfe
from m2_round2_screening import measure_mfe_random_universe
from m3_round2_critique import prepare_data_with_eth_break, entry_iota


# ---------- Extended data prep ----------

def prepare_data_r5(eth_break_lookback=24):
    df, h1, h4, base_valid, eth_valid_ext, funding_valid = prepare_data_with_eth_break(lookback=eth_break_lookback)
    # Add hour-of-day
    df['hour_utc'] = pd.to_datetime(df['timestamp']).dt.hour.values
    return df, h1, h4, base_valid, eth_valid_ext, funding_valid


# ---------- ρ×ι: session-filtered ι ----------

def make_entry_iota_session(session_hours):
    """Factory: returns entry_fn restricted to session_hours (set/list)."""
    hour_set = set(session_hours)
    def entry_fn(df, h1, h4, valid, params=None):
        sigs_full = entry_iota(df, h1, h4, valid, params=params)
        hour = df['hour_utc'].values
        return [(i, d) for i, d in sigs_full if hour[i] in hour_set]
    return entry_fn


# ---------- σ: counter-trend mean-rev at structural break ----------

def entry_sigma(df, h1, h4, valid, params=None):
    p = {'rsi_thresh': 70, 'eth_break_lookback': 24} if params is None else params
    n = len(df)
    rsi = df['rsi14'].values
    btc_ret = df['btc_return'].values
    eth_close = df['eth_close'].values
    eth_high_prev = df['eth_high_24_prev'].values
    eth_low_prev = df['eth_low_24_prev'].values
    if p.get('eth_break_lookback', 24) != 24:
        lb = p['eth_break_lookback']
        eth_high_prev = pd.Series(eth_close).rolling(lb, min_periods=lb).max().shift(1).values
        eth_low_prev = pd.Series(eth_close).rolling(lb, min_periods=lb).min().shift(1).values

    sigs = []
    rsi_hi = p['rsi_thresh']
    rsi_lo = 100 - p['rsi_thresh']
    for i in range(2, n):
        if not valid[i]: continue
        if any(pd.isna(x) for x in (rsi[i], btc_ret[i - 1], eth_close[i], eth_high_prev[i], eth_low_prev[i])):
            continue
        # SHORT: ETH break up + BTC RSI overbought + recent up momentum (exhaustion)
        if eth_close[i] > eth_high_prev[i] and rsi[i] >= rsi_hi and btc_ret[i - 1] > 0:
            sigs.append((i, 'SHORT'))
        # LONG: ETH break down + BTC RSI oversold + recent down (exhaustion)
        elif eth_close[i] < eth_low_prev[i] and rsi[i] <= rsi_lo and btc_ret[i - 1] < 0:
            sigs.append((i, 'LONG'))
    return sigs


# ---------- Session sweep helper ----------

def sweep_iota_sessions(df, h1, h4, valid_mask, eligible_mask):
    """Test ι across 4 session windows. Return best by Δp50."""
    sessions = {
        'US (13-20 UTC)': set(range(13, 21)),
        'Asia (00-07 UTC)': set(range(0, 8)),
        'EU (08-12 UTC)': set(range(8, 13)),
        '24h (= ι base)': set(range(0, 24)),
    }
    base_params = {'eth_thresh': 0.3, 'btc_lag_thresh': 0.1, 'atr_pctile': 70.0, 'eth_break_lookback': 24}
    results = {}
    for name, hours in sessions.items():
        entry_fn = make_entry_iota_session(hours)
        signals = entry_fn(df, h1, h4, valid_mask, params=base_params)
        if not signals:
            results[name] = {'n': 0, 'diff_p50': None}
            continue
        cand_mfe = measure_mfe_for_signals(df, signals, max_bars=8)
        cand_stats = stats_mfe(cand_mfe, 0.20)
        if cand_stats is None:
            results[name] = {'n': len(signals), 'diff_p50': None}
            continue
        # Random baseline
        rnd_p50_list = []; rnd_pct_list = []
        for seed in (42, 123, 456):
            rnd = measure_mfe_random_universe(df, eligible_mask, h1, h4,
                                                target_n=cand_stats['n'], max_bars=8, seed=seed,
                                                direction_by_trend=True)
            rs = stats_mfe(rnd, 0.20)
            if rs:
                rnd_p50_list.append(rs['mfe_p50']); rnd_pct_list.append(rs['pct_mfe_gt_friction'])
        if not rnd_p50_list:
            results[name] = {'n': cand_stats['n'], 'diff_p50': None}
            continue
        rnd_p50 = sum(rnd_p50_list) / len(rnd_p50_list)
        rnd_pct = sum(rnd_pct_list) / len(rnd_pct_list)
        results[name] = {
            'n': cand_stats['n'],
            'cand_p50': cand_stats['mfe_p50'],
            'cand_pct': cand_stats['pct_mfe_gt_friction'],
            'random_p50': round(rnd_p50, 4),
            'random_pct': round(rnd_pct, 4),
            'diff_p50': round(cand_stats['mfe_p50'] - rnd_p50, 4),
            'diff_pct': round(cand_stats['pct_mfe_gt_friction'] - rnd_pct, 2),
        }
    return results


SPECS_R5 = {
    'sigma': {
        'name': 'σ (mean-rev at structural break)',
        'entry_fn': entry_sigma,
        'parameters': {'rsi_thresh': 70, 'eth_break_lookback': 24},
        'sensitivity_params': {
            'rsi_thresh': [65, 75],
            'eth_break_lookback': [19, 29],
        },
        'valid_mask_key': 'eth_valid',
        'eligible_universe_with_filter': False,  # counter-trend
        'direction_by_trend': False,
    },
}


def main():
    print("Loading data + R5 columns...")
    df, h1, h4, base_valid, eth_valid_ext, funding_valid = prepare_data_r5()
    print(f"  bars: {len(df):,} | eth_valid_ext: {int(eth_valid_ext.sum()):,}\n")

    eligible_with_filter_eth = (h1 & h4 | (~h1) & (~h4)) & eth_valid_ext
    eligible_no_filter_eth = eth_valid_ext  # for σ (counter-trend)

    matrix = {}

    # ---------- ρ×ι session sweep ----------
    print("=" * 80); print("MECHANISM ρ×ι: Session-filtered ι"); print("=" * 80)
    print("  Session sweep (4 windows)...")
    sweep = sweep_iota_sessions(df, h1, h4, eth_valid_ext, eligible_with_filter_eth)
    print(f"  {'session':<22} {'n':>6} {'Δp50':>10} {'Δ%>fr':>10}")
    for name, r in sweep.items():
        if r['diff_p50'] is None:
            print(f"  {name:<22} {r['n']:>6} {'N/A':>10} {'N/A':>10}")
        else:
            print(f"  {name:<22} {r['n']:>6} {r['diff_p50']:>+10.4f} {r['diff_pct']:>+10.2f}")

    # Pick best session by Δp50 (selection bias declared)
    valid_sweep = {k: v for k, v in sweep.items() if v['diff_p50'] is not None and k != '24h (= ι base)'}
    if valid_sweep:
        best_session = max(valid_sweep, key=lambda k: valid_sweep[k]['diff_p50'])
        print(f"\n  Best session (selection-biased): {best_session}")
        best_sw = valid_sweep[best_session]
        # Strict criterion check on best
        best_pass = (best_sw['diff_p50'] >= 0.05) and (best_sw['diff_pct'] >= 5.0)
        print(f"  Best session pass C1 relaxed: {best_pass}")

        rho_iota_results = {'sweep': sweep, 'best_session': best_session, 'best_metrics': best_sw,
                             'C1_pass': best_pass}
        if best_pass:
            # Run BT on best session
            print("\n  Running C3 friction stress on best session...")
            session_hours = {
                'US (13-20 UTC)': set(range(13, 21)),
                'Asia (00-07 UTC)': set(range(0, 8)),
                'EU (08-12 UTC)': set(range(8, 13)),
            }[best_session]
            spec_rho_iota = {
                'name': f'ρ×ι ({best_session})',
                'entry_fn': make_entry_iota_session(session_hours),
                'parameters': {'eth_thresh': 0.3, 'btc_lag_thresh': 0.1, 'atr_pctile': 70.0, 'eth_break_lookback': 24},
                'sensitivity_params': {'eth_thresh': [0.24, 0.36]},
                'valid_mask_key': 'eth_valid',
                'eligible_universe_with_filter': True,
                'direction_by_trend': True,
            }
            c3_results = {}
            for friction in (0.20, 0.30, 0.50, 0.80):
                trades = run_bt_with_spec(df, h1, h4, eth_valid_ext, spec_rho_iota, friction=friction)
                if not trades:
                    c3_results[friction] = None
                    continue
                s = trade_summary(trades, friction=friction)
                c3_results[friction] = s
            base = c3_results.get(0.20); med = c3_results.get(0.30)
            c3_pass = (base and base['daily_net'] > 0) and (med and med['daily_net'] > 0)
            rho_iota_results['C3_pass'] = c3_pass
            rho_iota_results['C3_metrics'] = {f'friction_{k}': (v['daily_net'] if v else None) for k, v in c3_results.items()}
            print(f"  C3 pass: {c3_pass} | metrics: {rho_iota_results['C3_metrics']}")
        else:
            print("  Skipping C3 — C1 fail on best session (selection-biased)")
        matrix['rho_iota'] = rho_iota_results

    # ---------- σ standalone ----------
    print("\n" + "=" * 80); print("MECHANISM σ: mean-rev at structural break"); print("=" * 80)
    spec = SPECS_R5['sigma']
    valid = eth_valid_ext
    eligible = eligible_no_filter_eth
    results = {}

    print("  C1 random baseline...")
    c1 = critique_random_baseline(df, h1, h4, spec, valid, eligible)
    results['C1'] = c1
    print(f"     pass={c1['pass']} metrics: {c1['metrics']}")
    if c1['pass']:
        print("  C2 look-ahead audit...")
        c2 = critique_lookahead_audit(df, h1, h4, spec, valid, eligible)
        results['C2'] = c2
        print(f"     pass={c2['pass']} metrics: {c2['metrics']}")
        if c2['pass']:
            print("  C3 friction stress...")
            c3 = critique_friction_stress(df, h1, h4, spec, valid)
            results['C3'] = c3
            print(f"     pass={c3['pass']} metrics: {c3['metrics']}")
            if c3['pass']:
                print("  C4 overfitting probe...")
                c4 = critique_overfitting_probe(df, h1, h4, spec, valid)
                results['C4'] = c4
                print(f"     pass={c4['pass']} metrics: {c4['metrics']}")
                if c4['pass']:
                    print("  C5 bootstrap 3-day...")
                    c5 = critique_bootstrap_3day(df, h1, h4, spec, valid, n_bootstrap=200)
                    results['C5'] = c5
                    print(f"     pass={c5['pass']} metrics: {c5['metrics']}")
    matrix['sigma'] = results

    # Final summary
    print("\n" + "=" * 100)
    print("M3-R5 — FINAL ROUND")
    print("=" * 100)

    out = {
        'date': datetime.now(timezone.utc).isoformat(),
        'spec_doc': 'claudedocs/m3_round5_specs.md',
        'rho_iota': matrix.get('rho_iota'),
        'sigma': {ck: {'pass': v.get('pass') if isinstance(v, dict) else v,
                        'metrics': v.get('metrics') if isinstance(v, dict) else None}
                   for ck, v in matrix.get('sigma', {}).items()},
        'full_results': matrix,
    }
    p = ROOT / 'results' / f'm3_r5_matrix_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(p, 'w') as f: json.dump(out, f, indent=2, default=str)
    print(f"\nSaved: {p}")


if __name__ == '__main__':
    main()
