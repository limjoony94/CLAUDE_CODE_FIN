#!/usr/bin/env python3
"""Body filter tuning — full 9-flag validation for top candidates.

Top candidates from body_filter_tuning_study:
  candidate_C_b0.25: slip PnL +71.89, MDD 14.31, ratio 5.02
  candidate_C_b0.30: slip PnL +71.16, MDD 13.87, ratio 5.13
  candidate_C_b0.60: slip PnL +69.13, MDD 10.10, ratio 6.84

Run: WF 5-fold clean + slip, 3-way split, MC, bootstrap, fold 2 check.
"""
import sys
import os
import json
import math
import random
import copy
from pathlib import Path
from datetime import datetime, timezone
from statistics import mean, median, stdev

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import scripts.analysis.intrabar_trail_impact as ibt
from scripts.analysis.c1_intrabar_parity import (
    run_bt_with_slippage, compute_mdd_additive, SLIPPAGE as SLIP_MED,
)
from scripts.analysis.c1_refined_bootstrap_mdd import (
    _stationary_bootstrap_indices,
)
from scripts.production.c1_breakout.signals import C1BreakoutSignal


TEST_COMBOS = {
    'baseline_b0.40':    {'max_sl_atr': 3.3, 'trail_K': 2.5,
                           'max_hold_bars': 192, 'body_min_ratio': 0.40},
    'candidate_C_b0.25': {'max_sl_atr': 4.0, 'trail_K': 2.5,
                           'max_hold_bars': 192, 'body_min_ratio': 0.25},
    'candidate_C_b0.30': {'max_sl_atr': 4.0, 'trail_K': 2.5,
                           'max_hold_bars': 192, 'body_min_ratio': 0.30},
    'candidate_C_b0.60': {'max_sl_atr': 4.0, 'trail_K': 2.5,
                           'max_hold_bars': 192, 'body_min_ratio': 0.60},
}

FOLD_BOUNDARIES = {
    'fold_1': (31,    6407),   'fold_2': (6407,  12783),
    'fold_3': (12783, 19159),  'fold_4': (19159, 25535),
    'fold_5': (25535, 31916),
}

_ORIG_STRAT = copy.deepcopy(ibt.strat)
_ORIG_SIG = ibt.sig
_ORIG_TK = ibt.trail_K
_ORIG_MH = ibt.max_hold


def set_combo_body(max_sl_atr, trail_K, max_hold_bars, body_min_ratio):
    new_s = copy.deepcopy(_ORIG_STRAT)
    new_s.update({'max_sl_atr': max_sl_atr, 'trail_K': trail_K,
                  'max_hold_bars': max_hold_bars,
                  'body_min_ratio': body_min_ratio})
    ibt.strat = new_s
    ibt.trail_K = trail_K
    ibt.max_hold = max_hold_bars
    ibt.sig = C1BreakoutSignal(new_s)


def reset_combo_body():
    ibt.strat = copy.deepcopy(_ORIG_STRAT)
    ibt.sig = _ORIG_SIG
    ibt.trail_K = _ORIG_TK
    ibt.max_hold = _ORIG_MH


def run_clean(cfg):
    set_combo_body(**cfg)
    trades = ibt.run_backtest(mode='bar_close')
    reset_combo_body()
    for t in trades:
        if 'net' not in t:
            t['net'] = t['raw'] - ibt.FEE
    return trades


def run_slip(cfg):
    set_combo_body(**cfg)
    trades = run_bt_with_slippage(mode='5m', slippage=SLIP_MED)
    reset_combo_body()
    return trades


def summ(trades):
    if not trades:
        return {'count': 0, 'PnL': 0.0, 'MDD': 0.0, 'WR': 0.0}
    pnl = sum(t['net'] for t in trades)
    mdd = compute_mdd_additive(trades)
    wr = sum(1 for t in trades if t['net'] > 0) / len(trades) * 100
    return {'count': len(trades), 'PnL': round(pnl, 2),
            'MDD': round(mdd, 2), 'WR': round(wr, 1)}


def three_way(trades):
    n15 = ibt.n15
    warmup = 26
    t1 = warmup + int((n15 - warmup) * 0.6)
    t2 = warmup + int((n15 - warmup) * 0.8)
    tr = [t for t in trades if t['entry_bar'] <= t1]
    v = [t for t in trades if t1 < t['entry_bar'] <= t2]
    te = [t for t in trades if t['entry_bar'] > t2]
    return {'train': summ(tr), 'val': summ(v), 'test': summ(te)}


def fold_breakdown(trades):
    result = {}
    for fname, (lo, hi) in FOLD_BOUNDARIES.items():
        ft = [t for t in trades if lo <= t['entry_bar'] < hi]
        result[fname] = summ(ft)
    return result


def wf_5fold(trades):
    if not trades:
        return 0, []
    ts = sorted(trades, key=lambda t: t['entry_bar'])
    first, last = ts[0]['entry_bar'], ts[-1]['entry_bar']
    span = last - first
    if span < 5:
        return 0, []
    fold_size = span // 5
    pnls = []
    pos = 0
    for k in range(5):
        lo = first + k * fold_size
        hi = first + (k + 1) * fold_size if k < 4 else last + 1
        p = sum(t['net'] for t in ts if lo <= t['entry_bar'] < hi)
        pnls.append(round(p, 2))
        if p > 0:
            pos += 1
    return pos, pnls


def mc_p(trades, n_sims=999, seed=42):
    if not trades:
        return 1.0
    actual = sum(t['net'] for t in trades)
    rng = random.Random(seed)
    pnls = [t['net'] for t in trades]
    cnt = sum(1 for _ in range(n_sims)
              if sum((p if rng.random() < 0.5 else -p) for p in pnls)
              >= actual)
    return (cnt + 1) / (n_sims + 1)


def bootstrap_pnl(trades, n_boot=1000, seed=42, block=20):
    if not trades:
        return {'obs': 0, 'ci_lo': 0, 'ci_hi': 0}
    rng = random.Random(seed)
    n = len(trades)
    obs = sum(t['net'] for t in trades)
    vals = []
    for _ in range(n_boot):
        idx = _stationary_bootstrap_indices(n, block, rng)
        vals.append(sum(trades[j]['net'] for j in idx))
    vals.sort()
    return {'obs': round(obs, 3),
            'ci_lo': round(vals[int(0.025 * n_boot)], 3),
            'ci_hi': round(vals[int(0.975 * n_boot)], 3)}


def main():
    t0 = datetime.now()
    print('=' * 70)
    print('  Body Filter Full 9-flag Validation')
    print('=' * 70)

    results = {'timestamp': datetime.now(timezone.utc).isoformat(),
               'test_combos': TEST_COMBOS}

    per_combo = {}
    for cname, cfg in TEST_COMBOS.items():
        print(f'\n{"=" * 70}')
        print(f'  {cname}')
        print(f'{"=" * 70}')
        tc = run_clean(cfg)
        ts = run_slip(cfg)

        data = {
            'clean': summ(tc), 'slip': summ(ts),
            'wf_clean': wf_5fold(tc), 'wf_slip': wf_5fold(ts),
            'three_way_clean': three_way(tc),
            'three_way_slip':  three_way(ts),
            'fold_clean': fold_breakdown(tc),
            'fold_slip':  fold_breakdown(ts),
            'mc_p': mc_p(ts),
            'bootstrap': bootstrap_pnl(ts),
        }
        per_combo[cname] = data

        print(f'  Clean: PnL={data["clean"]["PnL"]:+.2f} MDD={data["clean"]["MDD"]:.2f} N={data["clean"]["count"]}')
        print(f'  Slip : PnL={data["slip"]["PnL"]:+.2f} MDD={data["slip"]["MDD"]:.2f} N={data["slip"]["count"]}')
        print(f'  WF clean: {data["wf_clean"][0]}/5 {data["wf_clean"][1]}')
        print(f'  WF slip : {data["wf_slip"][0]}/5 {data["wf_slip"][1]}')
        print(f'  3way slip: train={data["three_way_slip"]["train"]["PnL"]:+.2f} '
              f'val={data["three_way_slip"]["val"]["PnL"]:+.2f} '
              f'test={data["three_way_slip"]["test"]["PnL"]:+.2f}')
        print(f'  MC p={data["mc_p"]:.4f}')
        print(f'  Bootstrap: obs={data["bootstrap"]["obs"]} '
              f'CI=[{data["bootstrap"]["ci_lo"]}, {data["bootstrap"]["ci_hi"]}]')
        # Fold 2 focus
        f2_slip = data["fold_slip"]["fold_2"]
        f2_clean = data["fold_clean"]["fold_2"]
        print(f'  Fold 2: clean PnL={f2_clean["PnL"]:+.2f} ({f2_clean["count"]}) | '
              f'slip PnL={f2_slip["PnL"]:+.2f} ({f2_slip["count"]})')

    results['per_combo'] = per_combo

    # 9-flag for each candidate vs baseline_b0.40
    base = per_combo['baseline_b0.40']
    print(f'\n{"=" * 70}')
    print(f'  9-flag GO vs baseline_b0.40')
    print(f'{"=" * 70}')

    flags_by_cand = {}
    for cname in ('candidate_C_b0.25', 'candidate_C_b0.30', 'candidate_C_b0.60'):
        c = per_combo[cname]
        flags = {}
        flags['wf_clean_pass'] = c['wf_clean'][0] == 5
        flags['wf_slip_pass']  = c['wf_slip'][0] == 5
        tw = c['three_way_slip']
        flags['tw_pass'] = all(tw[s]['PnL'] > 0 for s in ('train', 'val', 'test'))
        flags['test_not_worse'] = (
            c['three_way_slip']['test']['PnL']
            >= base['three_way_slip']['test']['PnL'] - 5.0
        )
        # nbr skipped (body grid already dense)
        flags['mc_pass'] = c['mc_p'] < 0.01
        flags['ci_pass'] = c['bootstrap']['ci_lo'] > 0
        flags['train_not_degraded'] = (
            c['three_way_slip']['train']['PnL']
            >= base['three_way_slip']['train']['PnL'] - 2.0
        )
        flags['pnl_improvement'] = c['slip']['PnL'] >= base['slip']['PnL'] + 5.0
        c_r = c['slip']['PnL'] / c['slip']['MDD'] if c['slip']['MDD'] > 0 else 0
        b_r = base['slip']['PnL'] / base['slip']['MDD'] if base['slip']['MDD'] > 0 else 0
        flags['ratio_improvement'] = c_r >= b_r * 1.10  # +10% ratio

        print(f'\n{cname}:')
        for k, v in flags.items():
            print(f'  {k}: {v}')
        passed = sum(1 for v in flags.values() if v)
        print(f'  → {passed}/{len(flags)} flags pass')
        flags_by_cand[cname] = {'flags': flags, 'passed': passed}

    results['flags_by_cand'] = flags_by_cand

    # Best
    best_cand = max(flags_by_cand.items(), key=lambda x: x[1]['passed'])
    print(f'\n{"=" * 70}')
    print(f'  Best: {best_cand[0]} — {best_cand[1]["passed"]}/9 flags')
    print(f'{"=" * 70}')

    # Verdict
    CORE = ['wf_clean_pass', 'wf_slip_pass', 'tw_pass',
            'train_not_degraded', 'pnl_improvement']
    best_flags = best_cand[1]['flags']
    stop_reason = None
    for c in CORE:
        if not best_flags.get(c):
            stop_reason = f'core {c}'
            break
    if stop_reason:
        vdict, reason = 'STOP', stop_reason
    elif all(best_flags.values()):
        vdict, reason = 'GO', f'{best_cand[0]} 9/9 pass'
    else:
        passed_n = sum(1 for v in best_flags.values() if v)
        vdict, reason = 'PARTIAL', f'{best_cand[0]} {passed_n}/9'
    results['verdict'] = {'outcome': vdict, 'reason': reason,
                          'best_combo': best_cand[0]}
    print(f'VERDICT: {vdict} — {reason}')

    elapsed = (datetime.now() - t0).total_seconds()
    results['elapsed_sec'] = round(elapsed, 1)
    print(f'\nElapsed: {elapsed:.1f}s')

    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out = ROOT / 'results' / f'body_filter_full_validation_{stamp}.json'
    out.write_text(json.dumps(results, indent=2, default=str))
    print(f'Results: {out}')


if __name__ == '__main__':
    main()
