#!/usr/bin/env python3
"""Candidate_C (max_sl_atr=4.0, trail_K=2.5, max_hold_bars=192) validation.

9-flag GO protocol per docs/02-design/features/candidate_c_validation.design.md.

Compares baseline (3.3, 2.5, 192) vs candidate_C (4.0, 2.5, 192) under:
  - Clean bar_close BT
  - 5m + slippage (3 scenarios: low/med/high)
  - WF time-partition on adjusted trades
  - 3-way train/val/test split
  - Bootstrap PnL CI, MC direction p-value
  - Neighborhood ±1 step robustness

Output: results/candidate_c_validation_{timestamp}.json
"""
import sys
import os
import json
import math
import copy
import random
from pathlib import Path
from datetime import datetime, timezone
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import pandas as pd

# Reuse engines
import scripts.analysis.intrabar_trail_impact as ibt
import scripts.analysis.c1_intrabar_parity as cip
from scripts.analysis.c1_intrabar_parity import (
    run_bt_with_slippage, apply_slippage, wf_on_adjusted_trades,
    compute_mdd_additive, set_combo, reset_combo,
    train_boundary,
)
from scripts.analysis.c1_refined_bootstrap_mdd import (
    _stationary_bootstrap_indices,
)


# ─── Config ────────────────────────────────────────────────────────

COMBOS = {
    'baseline':    {'max_sl_atr': 3.3, 'trail_K': 2.5, 'max_hold_bars': 192},
    'candidate_C': {'max_sl_atr': 4.0, 'trail_K': 2.5, 'max_hold_bars': 192},
}

SLIP_SCENARIOS = {
    'low':  dict(entry_pct=0.025, exit_sl_pct=0.075, exit_trail_pct=0.025,
                 exit_emergency_pct=0.15, exit_timeout_pct=0.025),
    'med':  dict(entry_pct=0.05,  exit_sl_pct=0.15,  exit_trail_pct=0.05,
                 exit_emergency_pct=0.30, exit_timeout_pct=0.05),
    'high': dict(entry_pct=0.10,  exit_sl_pct=0.30,  exit_trail_pct=0.10,
                 exit_emergency_pct=0.60, exit_timeout_pct=0.10),
}

NBR_AXES = {
    'max_sl_atr':    [2.8, 3.0, 3.3, 3.6, 4.0, 4.5],
    'trail_K':       [2.0, 2.2, 2.5, 2.8, 3.0],
    'max_hold_bars': [96, 144, 192, 288],
}


# ─── Helpers ───────────────────────────────────────────────────────

def run_clean(combo_cfg):
    """Clean bar_close run, return trades with entry_bar, net, reason."""
    set_combo(**combo_cfg)
    trades = ibt.run_backtest(mode='bar_close')
    reset_combo()
    return trades


def run_slip(combo_cfg, slip_scenario):
    """5m + slippage run — pass slippage explicitly (default-arg binding bug)."""
    set_combo(**combo_cfg)
    trades = run_bt_with_slippage(mode='5m',
                                  slippage=SLIP_SCENARIOS[slip_scenario])
    reset_combo()
    return trades


def summarize(trades):
    if not trades:
        return {'count': 0, 'PnL': 0.0, 'MDD': 0.0, 'WR': 0.0}
    pnl = sum(t['net'] for t in trades)
    mdd = compute_mdd_additive(trades)
    wr = sum(1 for t in trades if t['net'] > 0) / len(trades) * 100
    return {'count': len(trades), 'PnL': round(pnl, 2),
            'MDD': round(mdd, 2), 'WR': round(wr, 1)}


def summarize_slice(trades):
    if not trades:
        return {'PnL': 0.0, 'MDD': 0.0, 'count': 0}
    pnl = sum(t['net'] for t in trades)
    mdd = compute_mdd_additive(trades)
    return {'PnL': round(pnl, 2), 'MDD': round(mdd, 2), 'count': len(trades)}


def three_way_split(trades):
    """60/20/20 time-partition by entry_bar."""
    n15 = ibt.n15
    warmup = 26
    t1 = warmup + int((n15 - warmup) * 0.6)
    t2 = warmup + int((n15 - warmup) * 0.8)
    train = [t for t in trades if t['entry_bar'] <= t1]
    val   = [t for t in trades if t1 < t['entry_bar'] <= t2]
    test  = [t for t in trades if t['entry_bar'] > t2]
    return {
        'train': summarize_slice(train),
        'val':   summarize_slice(val),
        'test':  summarize_slice(test),
        'boundary_train': t1,
        'boundary_val':   t2,
    }


def wf_folds(trades, n_folds=5):
    """Time-partition WF. Returns (positive_count, fold_pnls)."""
    if not trades:
        return 0, []
    ts = sorted(trades, key=lambda t: t['entry_bar'])
    first = ts[0]['entry_bar']
    last = ts[-1]['entry_bar']
    span = last - first
    if span < n_folds:
        return 0, []
    fold_size = span // n_folds
    pnls = []
    pos = 0
    for k in range(n_folds):
        lo = first + k * fold_size
        hi = first + (k + 1) * fold_size if k < n_folds - 1 else last + 1
        pnl = sum(t['net'] for t in ts if lo <= t['entry_bar'] < hi)
        pnls.append(round(pnl, 2))
        if pnl > 0:
            pos += 1
    return pos, pnls


def mc_direction_pvalue(trades, n_sims=999, seed=42):
    if not trades:
        return 1.0
    actual = sum(t['net'] for t in trades)
    rng = random.Random(seed)
    pnls = [t['net'] for t in trades]
    cnt = 0
    for _ in range(n_sims):
        s = sum((p if rng.random() < 0.5 else -p) for p in pnls)
        if s >= actual:
            cnt += 1
    return (cnt + 1) / (n_sims + 1)


def bootstrap_pnl(trades, n_boot=1000, seed=42, mean_block_len=20):
    if not trades:
        return {'obs_pnl': 0.0, 'pnl_ci_lo': 0.0, 'pnl_ci_hi': 0.0}
    rng = random.Random(seed)
    n = len(trades)
    obs = sum(t['net'] for t in trades)
    vals = []
    for _ in range(n_boot):
        idx = _stationary_bootstrap_indices(n, mean_block_len, rng)
        vals.append(sum(trades[j]['net'] for j in idx))
    vals.sort()
    return {
        'obs_pnl': round(obs, 3),
        'pnl_ci_lo': round(vals[int(0.025 * n_boot)], 3),
        'pnl_ci_hi': round(vals[int(0.975 * n_boot)], 3),
    }


def neighborhood_eval(center_combo, slip_scenario='med'):
    """6-neighbor axis-adjacent evaluation under slippage scenario."""
    nbs = []
    for ax, vals in NBR_AXES.items():
        cur = vals.index(center_combo[ax])
        for di in (-1, +1):
            j = cur + di
            if 0 <= j < len(vals):
                nb = dict(center_combo)
                nb[ax] = vals[j]
                nbs.append(nb)
    results = []
    pos = 0
    for nb in nbs:
        trades = run_slip(nb, slip_scenario)
        s = summarize(trades)
        is_pos = s['PnL'] > 0
        if is_pos:
            pos += 1
        results.append({'combo': nb, 'PnL': s['PnL'], 'MDD': s['MDD'],
                        'count': s['count'], 'positive': is_pos})
    return {'neighbors': results, 'positive_count': pos, 'total': len(nbs)}


# ─── GO Evaluation ─────────────────────────────────────────────────

def evaluate_go_9flags(base, cand):
    """Evaluate 9 GO flags. base/cand are dicts with full validation data."""
    f = {}

    # 1. WF clean 5/5
    f['wf_clean_pass'] = cand['wf_clean'][0] == 5

    # 2. WF slip (med) 5/5
    f['wf_slip_pass'] = cand['wf_slip_med'][0] == 5

    # 3. Three-way slip all positive
    tw = cand['three_way_slip']
    f['tw_pass'] = all(tw[s]['PnL'] > 0 for s in ('train', 'val', 'test'))

    # 4. Test not worse (clean AND slip)
    f['test_not_worse'] = (
        cand['three_way_clean']['test']['PnL']
        >= base['three_way_clean']['test']['PnL'] - 5.0
        and cand['three_way_slip']['test']['PnL']
        >= base['three_way_slip']['test']['PnL'] - 5.0
    )

    # 5. Neighborhood ≥ 5/6
    f['nbr_pass'] = cand['neighborhood']['positive_count'] >= 5

    # 6. MC p < 0.01
    f['mc_pass'] = cand['mc_p'] < 0.01

    # 7. Bootstrap CI lower > 0
    f['ci_pass'] = cand['bootstrap']['pnl_ci_lo'] > 0

    # 8. Train not degraded (slip)
    f['train_not_degraded'] = (
        cand['three_way_slip']['train']['PnL']
        >= base['three_way_slip']['train']['PnL'] - 2.0
    )

    # 9. Slippage sensitivity (all 3 scenarios PnL/MDD 우위)
    wins = 0
    details = {}
    for s in ('low', 'med', 'high'):
        b = base[f'slip_{s}']
        c = cand[f'slip_{s}']
        br = b['PnL'] / b['MDD'] if b['MDD'] > 0 else 0
        cr = c['PnL'] / c['MDD'] if c['MDD'] > 0 else 0
        details[s] = {'baseline_ratio': round(br, 2),
                      'candidate_ratio': round(cr, 2),
                      'winner': 'candidate' if cr > br else 'baseline'}
        if cr > br:
            wins += 1
    f['slip_sensitivity'] = wins == 3
    f['_slip_details'] = details

    return f


CORE_FLAGS = ['wf_clean_pass', 'wf_slip_pass', 'tw_pass',
              'train_not_degraded', 'slip_sensitivity']


def verdict(flags):
    for c in CORE_FLAGS:
        if not flags[c]:
            return 'STOP', f'core flag {c} failed'
    true_cnt = sum(1 for k, v in flags.items()
                   if not k.startswith('_') and v is True)
    if true_cnt == 9:
        return 'GO', '9/9 flags pass'
    missing = [k for k, v in flags.items()
               if not k.startswith('_') and v is not True]
    return 'STOP', f'{true_cnt}/9 pass; missing: {missing}'


# ─── Main ──────────────────────────────────────────────────────────

def main():
    t0 = datetime.now()
    print(f'{"="*70}')
    print(f'  Candidate_C Validation — 9-flag GO protocol')
    print(f'{"="*70}')
    print(f'Data: {ibt.n15} bars (15m), {ibt.n5} bars (5m)')
    print()

    results = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'combos': COMBOS,
        'slippage_scenarios': SLIP_SCENARIOS,
        'data': {'bars_15m': ibt.n15, 'bars_5m': ibt.n5},
    }

    # ─── 8 Primary Runs ─────────────────────────────────────────
    print('--- Primary runs (8) ---')
    trades_store = {}
    for combo_name, combo_cfg in COMBOS.items():
        print(f'\n{combo_name} {combo_cfg}')
        # Clean bar_close
        tr_clean = run_clean(combo_cfg)
        s_clean = summarize(tr_clean)
        trades_store[f'{combo_name}_clean'] = tr_clean
        print(f'  clean  : PnL={s_clean["PnL"]:+7.2f}% MDD={s_clean["MDD"]:.2f} '
              f'WR={s_clean["WR"]:5.1f}% N={s_clean["count"]}')
        results[f'{combo_name}_clean'] = s_clean

        for sc in ('low', 'med', 'high'):
            tr = run_slip(combo_cfg, sc)
            s = summarize(tr)
            trades_store[f'{combo_name}_slip_{sc}'] = tr
            print(f'  slip_{sc:4s}: PnL={s["PnL"]:+7.2f}% MDD={s["MDD"]:.2f} '
                  f'WR={s["WR"]:5.1f}% N={s["count"]}')
            results[f'{combo_name}_slip_{sc}'] = s

    # ─── Baseline full validation (for comparison test) ─────────
    print('\n--- Baseline full validation (for comparison) ---')
    base_data = {
        'three_way_clean': three_way_split(trades_store['baseline_clean']),
        'three_way_slip':  three_way_split(trades_store['baseline_slip_med']),
        'slip_low':  results['baseline_slip_low'],
        'slip_med':  results['baseline_slip_med'],
        'slip_high': results['baseline_slip_high'],
    }
    print(f'  3way clean: train={base_data["three_way_clean"]["train"]["PnL"]:+.2f} '
          f'val={base_data["three_way_clean"]["val"]["PnL"]:+.2f} '
          f'test={base_data["three_way_clean"]["test"]["PnL"]:+.2f}')
    print(f'  3way slip : train={base_data["three_way_slip"]["train"]["PnL"]:+.2f} '
          f'val={base_data["three_way_slip"]["val"]["PnL"]:+.2f} '
          f'test={base_data["three_way_slip"]["test"]["PnL"]:+.2f}')
    results['baseline_validation'] = base_data

    # ─── Candidate_C full validation ───────────────────────────
    print('\n--- Candidate_C full validation ---')
    c_clean_trades = trades_store['candidate_C_clean']
    c_med_trades   = trades_store['candidate_C_slip_med']

    cand_data = {
        'wf_clean':     wf_folds(c_clean_trades),
        'wf_slip_med':  wf_folds(c_med_trades),
        'three_way_clean': three_way_split(c_clean_trades),
        'three_way_slip':  three_way_split(c_med_trades),
        'mc_p': mc_direction_pvalue(c_med_trades),
        'bootstrap': bootstrap_pnl(c_med_trades),
        'slip_low':  results['candidate_C_slip_low'],
        'slip_med':  results['candidate_C_slip_med'],
        'slip_high': results['candidate_C_slip_high'],
    }
    print(f'  WF clean  : {cand_data["wf_clean"][0]}/5 folds={cand_data["wf_clean"][1]}')
    print(f'  WF slip   : {cand_data["wf_slip_med"][0]}/5 folds={cand_data["wf_slip_med"][1]}')
    print(f'  3way clean: train={cand_data["three_way_clean"]["train"]["PnL"]:+.2f} '
          f'val={cand_data["three_way_clean"]["val"]["PnL"]:+.2f} '
          f'test={cand_data["three_way_clean"]["test"]["PnL"]:+.2f}')
    print(f'  3way slip : train={cand_data["three_way_slip"]["train"]["PnL"]:+.2f} '
          f'val={cand_data["three_way_slip"]["val"]["PnL"]:+.2f} '
          f'test={cand_data["three_way_slip"]["test"]["PnL"]:+.2f}')
    print(f'  MC p      : {cand_data["mc_p"]:.4f}')
    print(f'  Bootstrap : obs={cand_data["bootstrap"]["obs_pnl"]} '
          f'CI=[{cand_data["bootstrap"]["pnl_ci_lo"]}, {cand_data["bootstrap"]["pnl_ci_hi"]}]')

    print('\n--- Neighborhood (candidate_C ±1, slip_med) ---')
    nbr = neighborhood_eval(COMBOS['candidate_C'], 'med')
    cand_data['neighborhood'] = nbr
    print(f'  {nbr["positive_count"]}/{nbr["total"]} positive')
    for n in nbr['neighbors']:
        mark = '✓' if n['positive'] else '✗'
        print(f'   {mark} {n["combo"]}: PnL={n["PnL"]:+.2f}% MDD={n["MDD"]:.2f}')

    results['candidate_C_validation'] = cand_data

    # ─── GO Evaluation ─────────────────────────────────────────
    print('\n--- 9-flag GO Evaluation ---')
    flags = evaluate_go_9flags(base_data, cand_data)
    for k, v in flags.items():
        if not k.startswith('_'):
            print(f'  {k}: {v}')
    print(f'  _slip_details: {flags["_slip_details"]}')
    results['go_flags'] = flags

    vdict, reason = verdict(flags)
    results['verdict'] = {'outcome': vdict, 'reason': reason}
    print(f'\n=== VERDICT: {vdict} — {reason} ===')

    # Save
    elapsed = (datetime.now() - t0).total_seconds()
    results['elapsed_sec'] = round(elapsed, 1)
    print(f'\nElapsed: {elapsed:.1f}s')

    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out = ROOT / 'results' / f'candidate_c_validation_{stamp}.json'
    out.write_text(json.dumps(results, indent=2, default=str))
    print(f'Results: {out}')


if __name__ == '__main__':
    main()
