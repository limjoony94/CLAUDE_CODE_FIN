"""M3-R9 — Deep verify on top 5 candidates with adjusted criterion.

Top 5: κ, ι, α, υ, σ.
Per candidate: friction breakdown, 10-seed strict, per-horizon, WF 5-fold, 3-way split, bootstrap.
Adjusted criterion: net daily > 0 @ 0.10% friction, WR ≥40%, R:R ≥1.5, WF 3/5+, bootstrap pass.
"""
import sys, json, random
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / 'scripts' / 'analysis'))

from m3_critique_pipeline import (prepare_all_data, run_bt_with_spec, trade_summary,
                                    rolling_pctile, EXIT_PARAMS,
                                    entry_alpha)
from m3_round2_critique import (prepare_data_with_eth_break, entry_iota,
                                  ALPHA_PRIME_EXIT_PARAMS)
from m3_round5_critique import entry_sigma
from m3_round6_critique import entry_upsilon, prepare_data_r6
from m3_round8_critique import entry_kappa, prepare_data_r8
from m2_round1_screening import measure_mfe_for_signals, stats_mfe, isolation_test
from m2_round2_screening import measure_mfe_random_universe


def deep_verify_one(df, h1, h4, valid, eligible, spec_name, entry_fn, params,
                    direction_by_trend, exit_params=None):
    """Run all deep tests for one candidate. Returns dict."""
    result = {'name': spec_name, 'params': params}
    spec = {
        'name': spec_name,
        'entry_fn': entry_fn,
        'parameters': params,
        'direction_by_trend': direction_by_trend,
    }
    if exit_params is not None:
        spec['exit_params'] = exit_params

    # Test 1: friction breakdown (fine-grained)
    print(f"  [{spec_name}] Test 1: friction grid...")
    friction_results = {}
    for f in (0.04, 0.06, 0.08, 0.10, 0.12, 0.15, 0.20):
        trades = run_bt_with_spec(df, h1, h4, valid, spec, friction=f)
        if not trades:
            friction_results[f] = None; continue
        s = trade_summary(trades, friction=f)
        friction_results[f] = {'n': s['n'], 'daily_net': s['daily_net'],
                                'wr_pct': s['wr_pct'], 'rr': s['rr'],
                                'sum_gross': s['sum_gross'], 'avg_gross': s['avg_gross']}
    result['friction'] = friction_results
    pos_frictions = [f for f, r in friction_results.items() if r and r['daily_net'] > 0]
    result['breakeven_friction'] = max(pos_frictions) if pos_frictions else None

    # Test 2: 10-seed strict
    print(f"  [{spec_name}] Test 2: 10-seed strict...")
    signals = entry_fn(df, h1, h4, valid, params=params)
    cand_mfe = measure_mfe_for_signals(df, signals, max_bars=8)
    cand_stats = stats_mfe(cand_mfe, friction=0.20)
    if cand_stats:
        seeds = list(range(42, 42 + 10 * 100, 100))
        per_seed = []
        for seed in seeds:
            rnd = measure_mfe_random_universe(df, eligible, h1, h4,
                                                target_n=cand_stats['n'], max_bars=8, seed=seed,
                                                direction_by_trend=direction_by_trend)
            rs = stats_mfe(rnd, 0.20)
            if rs:
                per_seed.append({'seed': seed,
                                  'diff_p50': cand_stats['mfe_p50'] - rs['mfe_p50'],
                                  'diff_pct': cand_stats['pct_mfe_gt_friction'] - rs['pct_mfe_gt_friction']})
        strict_count = sum(1 for r in per_seed if r['diff_p50'] >= 0.10 and r['diff_pct'] >= 10.0)
        relaxed_count = sum(1 for r in per_seed if r['diff_p50'] >= 0.05 and r['diff_pct'] >= 5.0)
        avg_dp50 = sum(r['diff_p50'] for r in per_seed) / len(per_seed) if per_seed else 0
        result['10seed'] = {'strict_pass': strict_count, 'relaxed_pass': relaxed_count,
                             'avg_diff_p50': round(avg_dp50, 4),
                             'std_diff_p50': float(np.std([r['diff_p50'] for r in per_seed])) if per_seed else 0,
                             'n_signals_after_seq': cand_stats['n']}
    else:
        result['10seed'] = {'note': 'no signals'}

    # Test 3: per-horizon fixed exit (gross only)
    print(f"  [{spec_name}] Test 3: per-horizon...")
    horizons = {}
    for h_bars in (4, 8, 12, 16, 24):
        iso = isolation_test(df, signals, h_bars, friction=0.20)
        if iso:
            horizons[h_bars] = {'n': iso['n_trades'], 'gross_sum': iso['gross_sum'],
                                 'gross_avg': iso['gross_avg'], 'gross_wr': iso['gross_wr_pct']}
    result['horizons'] = horizons

    # Test 4: WF 5-fold expanding
    print(f"  [{spec_name}] Test 4: WF 5-fold...")
    n = len(df)
    fold_size = n // 6
    wf = []
    for fold_i in range(5):
        train_end = (fold_i + 1) * fold_size
        test_start = train_end
        test_end = min(test_start + fold_size, n)
        df_test = df.iloc[test_start:test_end].reset_index(drop=True)
        h1_t = h1[test_start:test_end]; h4_t = h4[test_start:test_end]
        valid_t = valid[test_start:test_end]
        # Recompute df_test indicator dependencies — already have static columns from full data
        trades = run_bt_with_spec(df_test, h1_t, h4_t, valid_t, spec, friction=0.10)
        s = trade_summary(trades, friction=0.10) if trades else None
        wf.append({'fold': fold_i + 1, 'daily_net@0.10': s['daily_net'] if s else None,
                    'n': s['n'] if s else 0, 'wr': s['wr_pct'] if s else None})
    wf_pos = sum(1 for r in wf if r['daily_net@0.10'] is not None and r['daily_net@0.10'] > 0)
    result['wf'] = {'folds': wf, 'positive_count': wf_pos, 'pass': wf_pos >= 3}

    # Test 5: 3-way split (train 0-1/3, val 1/3-2/3, test 2/3-end)
    print(f"  [{spec_name}] Test 5: 3-way split...")
    third = n // 3
    splits = {}
    for label, (s_start, s_end) in [('train', (0, third)), ('val', (third, 2 * third)),
                                       ('test', (2 * third, n))]:
        df_s = df.iloc[s_start:s_end].reset_index(drop=True)
        h1_s = h1[s_start:s_end]; h4_s = h4[s_start:s_end]; valid_s = valid[s_start:s_end]
        trades = run_bt_with_spec(df_s, h1_s, h4_s, valid_s, spec, friction=0.10)
        ss = trade_summary(trades, friction=0.10) if trades else None
        splits[label] = {'daily_net@0.10': ss['daily_net'] if ss else None,
                          'n': ss['n'] if ss else 0,
                          'wr': ss['wr_pct'] if ss else None,
                          'rr': ss['rr'] if ss else None}
    result['three_way'] = splits

    # Test 6: 3-day bootstrap (200 windows for compute, scale 1000 mentally)
    print(f"  [{spec_name}] Test 6: bootstrap...")
    random.seed(42)
    bars_per_3day = 3 * 24 * 4
    max_start = n - bars_per_3day - 1
    if max_start > 0:
        starts = random.sample(range(max_start), min(200, max_start))
        bootstrap_pnls = []
        for start in starts:
            end = start + bars_per_3day
            df_w = df.iloc[start:end].reset_index(drop=True)
            h1_w = h1[start:end]; h4_w = h4[start:end]; valid_w = valid[start:end]
            trades = run_bt_with_spec(df_w, h1_w, h4_w, valid_w, spec, friction=0.10)
            cand_pnl = sum(t['net_pct'] for t in trades) if trades else 0
            bootstrap_pnls.append(cand_pnl)
        if bootstrap_pnls:
            mean_p = sum(bootstrap_pnls) / len(bootstrap_pnls)
            pos_rate = sum(1 for p in bootstrap_pnls if p > 0) / len(bootstrap_pnls)
            sorted_p = sorted(bootstrap_pnls)
            p5 = sorted_p[int(0.05 * len(sorted_p))]
            result['bootstrap'] = {'n': len(bootstrap_pnls), 'mean': round(mean_p, 4),
                                    'pos_rate': round(pos_rate, 4), 'p5': round(p5, 4),
                                    'pass': mean_p > 0 and pos_rate >= 0.5}

    # Adjusted criterion check
    f10 = friction_results.get(0.10)
    crit_check = {
        'daily_net_at_0.10_positive': f10 and f10['daily_net'] > 0,
        'wr_ge_40': f10 and f10['wr_pct'] >= 40,
        'rr_ge_1.5': f10 and f10['rr'] >= 1.5,
        'wf_3of5': result.get('wf', {}).get('pass', False),
        'three_way_test_positive': splits.get('test', {}).get('daily_net@0.10') and splits['test']['daily_net@0.10'] > 0,
        'bootstrap_pass': result.get('bootstrap', {}).get('pass', False),
    }
    crit_check['ALL_PASS'] = all(crit_check.values())
    result['adjusted_criterion'] = crit_check

    return result


def main():
    print("Loading data + R6 (volume) + R8 (mid-vol pctiles)...")
    # Use R8 prep which has both mid-vol pctiles AND ETH break columns (extends r6 indirectly)
    df_r8, h1, h4, base_valid, eth_valid_ext, funding_valid, kappa_valid, zeta_valid = prepare_data_r8()
    # Also need volume_sma (R6 column) — load r6 then merge
    df, _, _, _, _, _, upsilon_valid, _ = prepare_data_r6()
    # Use df (has volume_sma + wick) + ETH break columns from R8
    eth_break_cols = ['eth_high_24_prev', 'eth_low_24_prev', 'atr_pctile_30', 'atr_pctile_70']
    for col in eth_break_cols:
        df[col] = df_r8[col].values
    # Also need eth_close (already in df from prepare_all_data)

    print(f"  bars: {len(df):,}\n")

    eligible_with_filter = (h1 & h4 | (~h1) & (~h4))

    # 5 candidates
    candidates = [
        {
            'name': 'κ (ι + MID-vol regime)',
            'entry_fn': entry_kappa,
            'params': {'eth_thresh': 0.3, 'btc_lag_thresh': 0.1, 'eth_break_lookback': 24},
            'valid': eth_valid_ext & (~pd.isna(df['atr_pctile_30'])).values & (~pd.isna(df['atr_pctile_70'])).values,
            'eligible': eligible_with_filter & eth_valid_ext,
            'direction_by_trend': True,
        },
        {
            'name': 'ι (α + ETH 24-bar break)',
            'entry_fn': entry_iota,
            'params': {'eth_thresh': 0.3, 'btc_lag_thresh': 0.1, 'atr_pctile': 70.0, 'eth_break_lookback': 24},
            'valid': eth_valid_ext,
            'eligible': eligible_with_filter & eth_valid_ext,
            'direction_by_trend': True,
        },
        {
            'name': 'α (ETH-lag + 고변동성)',
            'entry_fn': entry_alpha,
            'params': {'eth_thresh': 0.3, 'btc_lag_thresh': 0.1, 'atr_pctile': 70.0},
            'valid': eth_valid_ext,
            'eligible': eligible_with_filter & eth_valid_ext,
            'direction_by_trend': True,
        },
        {
            'name': 'υ (volume × cross-asset)',
            'entry_fn': entry_upsilon,
            'params': {'vol_mult': 2.0, 'eth_thresh': 0.2},
            'valid': upsilon_valid,
            'eligible': eligible_with_filter & upsilon_valid,
            'direction_by_trend': True,
        },
        {
            'name': 'σ (mean-rev at structural break)',
            'entry_fn': entry_sigma,
            'params': {'rsi_thresh': 70, 'eth_break_lookback': 24},
            'valid': eth_valid_ext,
            'eligible': eth_valid_ext,  # counter-trend
            'direction_by_trend': False,
        },
    ]

    deep_results = {}
    for cand in candidates:
        print(f"\n{'=' * 80}")
        print(f"DEEP VERIFY: {cand['name']}")
        print('=' * 80)
        try:
            r = deep_verify_one(df, h1, h4, cand['valid'], cand['eligible'],
                                 cand['name'], cand['entry_fn'], cand['params'],
                                 cand['direction_by_trend'])
        except Exception as e:
            r = {'name': cand['name'], 'error': str(e)}
        deep_results[cand['name']] = r
        if 'adjusted_criterion' in r:
            print(f"  Adjusted criterion: {r['adjusted_criterion']}")

    # Summary table
    print("\n" + "=" * 100)
    print("M3-R9 — DEEP VERIFY SUMMARY (5 candidates × 6 tests)")
    print("=" * 100)
    print(f"{'mechanism':<38} {'BE_friction':>12} {'WF':>6} {'3way_test':>12} {'BS_mean':>10} {'PASS':>6}")
    for name, r in deep_results.items():
        if 'error' in r:
            print(f"{name:<38} ERROR: {r['error']}")
            continue
        be = r.get('breakeven_friction')
        be_s = f"{be:.2f}%" if be else "none"
        wf = r.get('wf', {}).get('positive_count', 0)
        tw = r.get('three_way', {}).get('test', {}).get('daily_net@0.10')
        tw_s = f"{tw:+.4f}" if tw is not None else "N/A"
        bs = r.get('bootstrap', {}).get('mean')
        bs_s = f"{bs:+.3f}" if bs is not None else "N/A"
        all_pass = r.get('adjusted_criterion', {}).get('ALL_PASS', False)
        print(f"{name:<38} {be_s:>12} {wf:>6} {tw_s:>12} {bs_s:>10} {'PASS' if all_pass else 'FAIL':>6}")

    out = {'date': datetime.now(timezone.utc).isoformat(),
           'spec_doc': 'claudedocs/m3_round9_adjusted_criterion.md',
           'adjusted_criterion': {
               'daily_net_positive_at_friction_0.10': True,
               'wr_min': 40, 'rr_min': 1.5, 'wf_min_folds': 3,
               'three_way_test_positive': True,
               'bootstrap_mean_positive_pos_rate_50': True,
           },
           'deep_results': deep_results}
    p = ROOT / 'results' / f'm3_r9_deep_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(p, 'w') as f: json.dump(out, f, indent=2, default=str)
    print(f"\nSaved: {p}")


if __name__ == '__main__':
    main()
