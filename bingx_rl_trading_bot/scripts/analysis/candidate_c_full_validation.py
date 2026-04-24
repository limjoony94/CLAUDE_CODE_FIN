#!/usr/bin/env python3
"""Candidate_C Full Validation (2026-04-24)

Pure candidate_C = (max_sl_atr=4.0, trail_K=2.5, max_hold_bars=192) — NO body override,
NO trend filter. Follow-up to 2026-04-19 9-flag STOP (fold 2 slip = -9.03 core fail).

Today's standards applied:
  - Baseline-relative GO criteria (Δ, not absolute > 0)
  - 3-day random-window bootstrap (memory: research_protocol_3day_bootstrap)
  - Strict expanding WF + fold-Δ (memory: regime_filter_trend_20260419)
  - 5×5 sharp-peak neighborhood
  - 2-half parameter consistency
  - OOS split boundary sensitivity (5 ratios)

Baseline: (3.3, 2.5, 192) — production default body=0.4, NO trend filter.
SLIP_MED primary, LOW/HIGH reported for sensitivity.

Output: results/candidate_c_full_validation_{timestamp}.json
"""
import sys
import os
import json
import copy
import random
from pathlib import Path
from datetime import datetime, timezone
from statistics import mean, stdev, median

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import scripts.analysis.intrabar_trail_impact as ibt
from scripts.analysis.c1_intrabar_parity import (
    SLIPPAGE, run_bt_with_slippage, set_combo, reset_combo,
    compute_mdd_additive,
)
from scripts.analysis.c1_refined_bootstrap_mdd import _stationary_bootstrap_indices

FEE = ibt.FEE
WARMUP = 26
DATA_DAYS = ibt.n15 / 96
SEED = 42

SLIP_LOW  = {k: v * 0.5 for k, v in SLIPPAGE.items()}
SLIP_MED  = SLIPPAGE
SLIP_HIGH = {k: v * 2.0 for k, v in SLIPPAGE.items()}

BASELINE = {'max_sl_atr': 3.3, 'trail_K': 2.5, 'max_hold_bars': 192}
CANDIDATE = {'max_sl_atr': 4.0, 'trail_K': 2.5, 'max_hold_bars': 192}

BOOT_N = 1000
BOOT_WINDOW_BARS = 288    # 3 days * 96 bars/day
BOOT_MIN_START = WARMUP + 10  # leave warmup buffer

# Neighborhood 5×5 axes (max_sl_atr × trail_K)
NBR_SL = [3.3, 3.6, 4.0, 4.5, 5.0]
NBR_TK = [2.0, 2.2, 2.5, 2.8, 3.0]


# ─── BT runner helpers ────────────────────────────────────────────

def run_clean(combo):
    """Clean bar_close backtest — returns trades (net already net of FEE)."""
    set_combo(**combo)
    trades = ibt.run_backtest(mode='bar_close')
    reset_combo()
    return trades


def run_slip(combo, slippage):
    """5m + slippage backtest — returns trades (net reflects slip+fee)."""
    set_combo(**combo)
    trades = run_bt_with_slippage(mode='5m', slippage=slippage)
    reset_combo()
    return trades


def pnl_of(trades):
    return sum(t['net'] for t in trades)


def compute_metrics(trades):
    if not trades:
        return {'n': 0, 'pnl': 0, 'daily': 0, 'wr': 0, 'mdd': 0, 'ex_top5': 0}
    total = sum(t['net'] for t in trades)
    wins = [t for t in trades if t['net'] > 0]
    eq = pk = md = 0
    for t in trades:
        eq += t['net']; pk = max(pk, eq); md = max(md, pk - eq)
    n_top = max(1, int(len(trades) * 0.05))
    ex_top = sorted(trades, key=lambda t: t['net'], reverse=True)[n_top:]
    return {
        'n': len(trades),
        'pnl': round(total, 2),
        'daily': round(total / DATA_DAYS, 3),
        'wr': round(len(wins) / len(trades) * 100, 1),
        'mdd': round(md, 2),
        'ex_top5': round(sum(t['net'] for t in ex_top), 2),
    }


# ─── Test 1: Full Period ──────────────────────────────────────────

def test_full_period():
    print('\n' + '=' * 110)
    print('  TEST 1: Full Period (Clean + Slip3)')
    print('=' * 110)
    results = {}
    for vname, v in [('baseline', BASELINE), ('candidate_C', CANDIDATE)]:
        results[vname] = {}
        clean_trades = run_clean(v)
        results[vname]['clean'] = compute_metrics(clean_trades)
        results[vname]['_clean_trades'] = clean_trades
        for sname, slip in [('slip_low', SLIP_LOW),
                            ('slip_med', SLIP_MED),
                            ('slip_high', SLIP_HIGH)]:
            t = run_slip(v, slip)
            results[vname][sname] = compute_metrics(t)
            results[vname][f'_{sname}_trades'] = t
        for sname in ('clean', 'slip_low', 'slip_med', 'slip_high'):
            m = results[vname][sname]
            print(f'  {vname:<12} {sname:<10}: n={m["n"]:>4}, PnL={m["pnl"]:>+7.2f}, '
                  f'MDD={m["mdd"]:>5.2f}, daily={m["daily"]:>+6.3f}, ex5={m["ex_top5"]:>+7.2f}')

    print('\n  Deltas (cand - base):')
    for sname in ('clean', 'slip_low', 'slip_med', 'slip_high'):
        d_pnl = results['candidate_C'][sname]['pnl'] - results['baseline'][sname]['pnl']
        d_mdd = results['candidate_C'][sname]['mdd'] - results['baseline'][sname]['mdd']
        d_daily = results['candidate_C'][sname]['daily'] - results['baseline'][sname]['daily']
        print(f'    {sname:<10}: ΔPnL={d_pnl:>+7.2f}, ΔMDD={d_mdd:>+5.2f}, '
              f'Δdaily={d_daily:>+6.3f}')
    return results


# ─── Test 2: Strict Expanding WF 5-fold ───────────────────────────

def strict_expanding_wf(trades, n=5):
    """5-fold strict expanding — each fold's OOS is bars [WARMUP + i*w, +w)."""
    bars = ibt.n15
    oos_w = (bars - WARMUP) // n
    fold_pnls = []
    for i in range(n):
        s = WARMUP + i * oos_w
        e = s + oos_w
        p = sum(t['net'] for t in trades if s <= t['entry_bar'] < e)
        fold_pnls.append(round(p, 2))
    return fold_pnls


def test_strict_wf(full_results):
    print('\n' + '=' * 110)
    print('  TEST 2: Strict Expanding WF 5-fold (slip_med)')
    print('=' * 110)
    base_trades = full_results['baseline']['_slip_med_trades']
    cand_trades = full_results['candidate_C']['_slip_med_trades']
    base_folds = strict_expanding_wf(base_trades)
    cand_folds = strict_expanding_wf(cand_trades)
    base_pos = sum(1 for p in base_folds if p > 0)
    cand_pos = sum(1 for p in cand_folds if p > 0)
    deltas = [round(c - b, 2) for c, b in zip(cand_folds, base_folds)]
    d_nonneg = sum(1 for d in deltas if d >= 0)

    print(f'  {"Variant":<10} {"f1":>7} {"f2":>7} {"f3":>7} {"f4":>7} {"f5":>7} {"pos":>4}')
    for name, folds, pos in [('baseline', base_folds, base_pos),
                              ('candidate', cand_folds, cand_pos)]:
        fstr = ' '.join(f'{p:>+7.2f}' for p in folds)
        print(f'  {name:<10} {fstr}  {pos}/5')
    dstr = ' '.join(f'{d:>+7.2f}' for d in deltas)
    print(f'  {"Δ(c-b)":<10} {dstr}  Δ≥0: {d_nonneg}/5')

    return {
        'baseline_folds': base_folds,
        'candidate_folds': cand_folds,
        'baseline_pos': base_pos,
        'candidate_pos': cand_pos,
        'deltas': deltas,
        'delta_nonneg': d_nonneg,
        'fold2_delta': deltas[1],
        'fold2_cand': cand_folds[1],
    }


# ─── Test 3: OOS Split 5 Boundaries ───────────────────────────────

def three_way_split(trades, ratios):
    bars = ibt.n15
    tr_end = int(bars * ratios[0])
    vl_end = tr_end + int(bars * ratios[1])
    train = [t for t in trades if t['entry_bar'] < tr_end]
    val = [t for t in trades if tr_end <= t['entry_bar'] < vl_end]
    test = [t for t in trades if t['entry_bar'] >= vl_end]
    return {
        'train': round(sum(t['net'] for t in train), 2),
        'val': round(sum(t['net'] for t in val), 2),
        'test': round(sum(t['net'] for t in test), 2),
    }


def test_oos_split_boundaries(full_results):
    print('\n' + '=' * 110)
    print('  TEST 3: OOS Split 5 Boundaries (slip_med)')
    print('=' * 110)
    bounds = [(0.60, 0.20, 0.20), (0.50, 0.25, 0.25), (0.70, 0.15, 0.15),
              (0.55, 0.15, 0.30), (0.60, 0.15, 0.25)]
    cand_trades = full_results['candidate_C']['_slip_med_trades']
    base_trades = full_results['baseline']['_slip_med_trades']
    results = []
    print(f'  {"Split":<16} {"b_tr":>8} {"b_vl":>8} {"b_te":>8}  '
          f'{"c_tr":>8} {"c_vl":>8} {"c_te":>8}  {"cval&test>0":>12}')
    for b in bounds:
        bs = three_way_split(base_trades, b)
        cs = three_way_split(cand_trades, b)
        ok = cs['val'] > 0 and cs['test'] > 0
        results.append({'split': b, 'base': bs, 'candidate': cs, 'pass': ok})
        label = '/'.join(str(int(x * 100)) for x in b)
        print(f'  {label:<16} {bs["train"]:>+8.2f} {bs["val"]:>+8.2f} {bs["test"]:>+8.2f}  '
              f'{cs["train"]:>+8.2f} {cs["val"]:>+8.2f} {cs["test"]:>+8.2f}  '
              f'{"PASS" if ok else "FAIL":>12}')
    passed = sum(1 for r in results if r['pass'])
    print(f'\n  Summary: {passed}/5 splits val&test both positive (candidate)')
    return {'results': results, 'pass_count': passed}


# ─── Test 4: 3-Way Split 60/20/20 ─────────────────────────────────

def test_3way_split(full_results):
    print('\n' + '=' * 110)
    print('  TEST 4: 3-Way Split (60/20/20) — slip_med + clean')
    print('=' * 110)
    out = {}
    for scenario in ('clean', 'slip_med'):
        base_trades = full_results['baseline'][f'_{scenario}_trades']
        cand_trades = full_results['candidate_C'][f'_{scenario}_trades']
        bs = three_way_split(base_trades, (0.6, 0.2, 0.2))
        cs = three_way_split(cand_trades, (0.6, 0.2, 0.2))
        all_pos = cs['train'] > 0 and cs['val'] > 0 and cs['test'] > 0
        train_not_degraded = cs['train'] >= bs['train'] - 2.0  # -2pp tolerance
        out[scenario] = {
            'baseline': bs, 'candidate': cs,
            'all_positive': all_pos,
            'train_not_degraded': train_not_degraded,
            'deltas': {k: round(cs[k] - bs[k], 2) for k in ('train', 'val', 'test')},
        }
        print(f'\n  Scenario: {scenario}')
        print(f'    baseline : train={bs["train"]:+7.2f}, val={bs["val"]:+7.2f}, test={bs["test"]:+7.2f}')
        print(f'    candidate: train={cs["train"]:+7.2f}, val={cs["val"]:+7.2f}, test={cs["test"]:+7.2f}')
        d = out[scenario]['deltas']
        print(f'    Δ(c-b)   : train={d["train"]:+7.2f}, val={d["val"]:+7.2f}, test={d["test"]:+7.2f}')
        print(f'    all_pos={all_pos}, train_not_degraded={train_not_degraded} (tol=-2.0pp)')
    return out


# ─── Test 5: 3-Day Bootstrap + Relative ───────────────────────────

def bootstrap_3day(trades):
    rng = random.Random(SEED)
    pnls = []
    max_start = ibt.n15 - BOOT_WINDOW_BARS - 1
    for _ in range(BOOT_N):
        s = rng.randint(BOOT_MIN_START, max_start)
        e = s + BOOT_WINDOW_BARS
        pnls.append(sum(t['net'] for t in trades if s <= t['entry_bar'] < e))
    ps = sorted(pnls)
    std = stdev(pnls) if len(pnls) > 1 else 0
    return {
        'mean': round(mean(pnls), 3),
        'median': round(median(pnls), 3),
        'std': round(std, 3),
        'pos_pct': round(sum(1 for p in pnls if p > 0) / BOOT_N * 100, 1),
        'sharpe': round(mean(pnls) / std if std > 0 else 0, 3),
        'p5': round(ps[int(0.05 * BOOT_N)], 2),
        'p95': round(ps[int(0.95 * BOOT_N)], 2),
        'p_loss_2pp': round(sum(1 for p in pnls if p < -2) / BOOT_N * 100, 1),
    }


def bootstrap_relative(cand_trades, base_trades):
    """P(cand PnL > base PnL) in same 3-day random window."""
    rng = random.Random(SEED)
    cw = 0
    ties = 0
    diffs = []
    max_start = ibt.n15 - BOOT_WINDOW_BARS - 1
    for _ in range(BOOT_N):
        s = rng.randint(BOOT_MIN_START, max_start)
        e = s + BOOT_WINDOW_BARS
        cp = sum(t['net'] for t in cand_trades if s <= t['entry_bar'] < e)
        bp = sum(t['net'] for t in base_trades if s <= t['entry_bar'] < e)
        diff = cp - bp
        diffs.append(diff)
        if cp > bp:
            cw += 1
        elif cp == bp:
            ties += 1
    return {
        'p_cand_beats_base': round(cw / BOOT_N * 100, 1),
        'ties_pct': round(ties / BOOT_N * 100, 1),
        'mean_diff': round(mean(diffs), 3),
        'median_diff': round(median(diffs), 3),
    }


def test_bootstrap(full_results):
    print('\n' + '=' * 110)
    print('  TEST 5: 3-Day Random Window Bootstrap (N=1000, slip_med)')
    print('=' * 110)
    base_trades = full_results['baseline']['_slip_med_trades']
    cand_trades = full_results['candidate_C']['_slip_med_trades']
    bs_base = bootstrap_3day(base_trades)
    bs_cand = bootstrap_3day(cand_trades)
    print(f'  {"Variant":<12} {"mean":>7} {"median":>7} {"pos%":>6} {"sharpe":>7} '
          f'{"p5":>6} {"p95":>7} {"p_loss2":>8}')
    for name, b in [('baseline', bs_base), ('candidate', bs_cand)]:
        print(f'  {name:<12} {b["mean"]:>+7.3f} {b["median"]:>+7.3f} {b["pos_pct"]:>5.1f}% '
              f'{b["sharpe"]:>+7.3f} {b["p5"]:>+6.2f} {b["p95"]:>+7.2f} '
              f'{b["p_loss_2pp"]:>7.1f}%')
    rel = bootstrap_relative(cand_trades, base_trades)
    print(f'\n  TEST 5b: P(candidate > baseline per-window) = {rel["p_cand_beats_base"]}% '
          f'(ties {rel["ties_pct"]}%)')
    print(f'  Mean diff per window = {rel["mean_diff"]:+.3f}')
    return {'baseline': bs_base, 'candidate': bs_cand, 'relative': rel}


# ─── Test 6: Neighborhood 5×5 (sharp peak check) ──────────────────

def test_neighborhood(baseline_slip_pnl):
    print('\n' + '=' * 110)
    print('  TEST 6: Neighborhood 5×5 max_sl_atr × trail_K (sharp peak check, slip_med)')
    print('=' * 110)
    print(f'  Baseline (3.3, 2.5) PnL = {baseline_slip_pnl:+.2f}')
    threshold = baseline_slip_pnl + 10
    print(f'  GO criterion: combo PnL >= baseline + 10pp = {threshold:+.2f}\n')
    print(f'  {"max_sl":>7} {"tk":>5} {"PnL":>8} {"MDD":>6} {"daily":>7} {"GO":>4}')
    results = []
    go_count = 0
    for sl in NBR_SL:
        for tk in NBR_TK:
            combo = {'max_sl_atr': sl, 'trail_K': tk, 'max_hold_bars': 192}
            t = run_slip(combo, SLIP_MED)
            m = compute_metrics(t)
            go = m['pnl'] >= threshold
            if go:
                go_count += 1
            results.append({'sl': sl, 'tk': tk, 'pnl': m['pnl'],
                            'mdd': m['mdd'], 'daily': m['daily'], 'go': go})
            mark = 'GO' if go else '-'
            print(f'  {sl:>7.1f} {tk:>5.2f} {m["pnl"]:>+7.2f} {m["mdd"]:>5.2f} '
                  f'{m["daily"]:>+6.3f} {mark:>4}')
    print(f'\n  GO count: {go_count}/25')
    return {'results': results, 'go_count': go_count,
            'threshold': round(threshold, 2)}


# ─── Test 7: Parameter Consistency 2-Half ─────────────────────────

def test_consistency(full_results):
    print('\n' + '=' * 110)
    print('  TEST 7: Parameter Consistency 2-Half (slip_med)')
    print('=' * 110)
    mid = ibt.n15 // 2
    cand_trades = full_results['candidate_C']['_slip_med_trades']
    base_trades = full_results['baseline']['_slip_med_trades']
    h1c = sum(t['net'] for t in cand_trades if t['entry_bar'] < mid)
    h2c = sum(t['net'] for t in cand_trades if t['entry_bar'] >= mid)
    h1b = sum(t['net'] for t in base_trades if t['entry_bar'] < mid)
    h2b = sum(t['net'] for t in base_trades if t['entry_bar'] >= mid)
    d1 = h1c - h1b
    d2 = h2c - h2b
    print(f'  1st half [bar<{mid}] : base={h1b:>+7.2f}, cand={h1c:>+7.2f}, Δ={d1:>+7.2f}')
    print(f'  2nd half [bar≥{mid}] : base={h2b:>+7.2f}, cand={h2c:>+7.2f}, Δ={d2:>+7.2f}')
    pass_flag = d1 >= 3 and d2 >= 3
    print(f'  Both halves Δ ≥ +3pp: {"PASS" if pass_flag else "FAIL"}')
    return {
        'h1_cand': round(h1c, 2), 'h2_cand': round(h2c, 2),
        'h1_base': round(h1b, 2), 'h2_base': round(h2b, 2),
        'h1_delta': round(d1, 2), 'h2_delta': round(d2, 2),
        'pass': pass_flag,
    }


# ─── Test 8: MC Direction ─────────────────────────────────────────

def test_mc_direction(full_results):
    print('\n' + '=' * 110)
    print('  TEST 8: MC Direction 999 sign-randomization sims (slip_med)')
    print('=' * 110)
    cand_trades = full_results['candidate_C']['_slip_med_trades']
    real = sum(t['net'] for t in cand_trades)
    nets = [t['net'] for t in cand_trades]
    rng = random.Random(SEED)
    beat = 0
    N = 999
    for _ in range(N):
        shuffled = sum(abs(n) * rng.choice([-1, 1]) for n in nets)
        if real > shuffled:
            beat += 1
    p = 1 - beat / N
    print(f'  Real PnL = {real:+.2f}')
    print(f'  Beat {beat}/{N} sims → p-value = {p:.4f}')
    passed = p < 0.01
    print(f'  p < 0.01: {"PASS" if passed else "FAIL (warning only)"}')
    return {'p_value': round(p, 4), 'real_pnl': round(real, 2), 'pass': passed}


# ─── Test 9: Slippage Sensitivity (ratio comparison) ──────────────

def test_slip_sensitivity(full_results):
    print('\n' + '=' * 110)
    print('  TEST 9: Slippage Sensitivity (ratio = PnL/MDD)')
    print('=' * 110)
    results = {}
    wins = 0
    print(f'  {"Scenario":<10} {"base PnL":>9} {"base MDD":>9} {"base rat":>9}  '
          f'{"c PnL":>9} {"c MDD":>9} {"c rat":>9} {"winner":>8}')
    for sname in ('slip_low', 'slip_med', 'slip_high'):
        b = full_results['baseline'][sname]
        c = full_results['candidate_C'][sname]
        br = b['pnl'] / b['mdd'] if b['mdd'] > 0 else 0
        cr = c['pnl'] / c['mdd'] if c['mdd'] > 0 else 0
        winner = 'candidate' if cr > br else 'baseline'
        if cr > br:
            wins += 1
        results[sname] = {
            'baseline_pnl': b['pnl'], 'baseline_mdd': b['mdd'],
            'baseline_ratio': round(br, 2),
            'candidate_pnl': c['pnl'], 'candidate_mdd': c['mdd'],
            'candidate_ratio': round(cr, 2),
            'winner': winner,
        }
        print(f'  {sname:<10} {b["pnl"]:>+8.2f} {b["mdd"]:>8.2f} {br:>+8.2f}  '
              f'{c["pnl"]:>+8.2f} {c["mdd"]:>8.2f} {cr:>+8.2f} {winner:>8}')
    print(f'\n  Candidate wins: {wins}/3')
    return {'results': results, 'wins': wins}


# ─── Test 10: Bootstrap PnL CI (stationary block) ─────────────────

def test_bootstrap_ci(full_results):
    print('\n' + '=' * 110)
    print('  TEST 10: Stationary-Block Bootstrap PnL CI (slip_med, 1000 iter)')
    print('=' * 110)
    cand_trades = full_results['candidate_C']['_slip_med_trades']
    if not cand_trades:
        return {'obs_pnl': 0, 'pnl_ci_lo': 0, 'pnl_ci_hi': 0}
    rng = random.Random(SEED)
    n = len(cand_trades)
    obs = sum(t['net'] for t in cand_trades)
    vals = []
    for _ in range(1000):
        idx = _stationary_bootstrap_indices(n, 20, rng)
        vals.append(sum(cand_trades[j]['net'] for j in idx))
    vals.sort()
    lo = round(vals[int(0.025 * 1000)], 2)
    hi = round(vals[int(0.975 * 1000)], 2)
    print(f'  obs_pnl = {obs:+.2f}')
    print(f'  95% CI = [{lo:+.2f}, {hi:+.2f}]')
    lo_pos = lo > 0
    print(f'  CI lower > 0: {"PASS" if lo_pos else "FAIL"}')
    return {'obs_pnl': round(obs, 2), 'pnl_ci_lo': lo, 'pnl_ci_hi': hi,
            'lo_positive': lo_pos}


# ─── GO Protocol (baseline-relative) ──────────────────────────────

def evaluate_go(results):
    """10-flag GO protocol — baseline-relative (2026-04-24 standard)."""
    f1 = results['full']
    f2 = results['wf']
    f3 = results['oos_split']
    f4 = results['three_way']
    f5 = results['boot']
    f6 = results['neighbor']
    f7 = results['consistency']
    f8 = results['mc']
    f9 = results['slip_sens']
    f10 = results['bootstrap_ci']

    cand_slip_med = f1['candidate_C']['slip_med']
    base_slip_med = f1['baseline']['slip_med']
    cand_clean = f1['candidate_C']['clean']
    base_clean = f1['baseline']['clean']

    delta_pnl_slip = cand_slip_med['pnl'] - base_slip_med['pnl']
    delta_pnl_clean = cand_clean['pnl'] - base_clean['pnl']
    delta_daily_slip = cand_slip_med['daily'] - base_slip_med['daily']
    delta_mdd_slip = cand_slip_med['mdd'] - base_slip_med['mdd']

    flags = {}
    # Core 9
    flags['c1_slip_pnl_delta_10pp'] = delta_pnl_slip >= 10
    flags['c2_clean_pnl_delta_5pp'] = delta_pnl_clean >= 5
    flags['c3_wf_slip_4of5_fold2_nonneg'] = (f2['candidate_pos'] >= 4
                                              and f2['fold2_delta'] >= 0)
    flags['c4_3way_slip_all_pos_train_ok'] = (
        f4['slip_med']['all_positive']
        and f4['slip_med']['train_not_degraded']
    )
    boot_c = f5['candidate']
    flags['c5_boot_core_3of3'] = (boot_c['mean'] > 0.3
                                   and boot_c['pos_pct'] >= 55
                                   and boot_c['p5'] >= -3.5)
    flags['c6_boot_relative_55'] = f5['relative']['p_cand_beats_base'] >= 55
    flags['c7_consistency_halves'] = f7['pass']
    flags['c8_oos_split_4of5'] = f3['pass_count'] >= 4
    flags['c9_slip_sensitivity_3of3'] = f9['wins'] == 3

    # Sharp-peak (10th — required per 2026-04-21 protocol)
    flags['c10_neighborhood_8plus'] = f6['go_count'] >= 8

    warnings = {
        'w1_mc_p_under_0.01': f8['pass'],
        'w2_ci_lower_above_0': f10['lo_positive'],
        'w3_sharpe_0.35': boot_c['sharpe'] >= 0.35,
        'w4_mdd_improved_1pp': (base_slip_med['mdd'] - cand_slip_med['mdd']) >= 1.0,
    }

    deltas = {
        'delta_pnl_slip': round(delta_pnl_slip, 2),
        'delta_pnl_clean': round(delta_pnl_clean, 2),
        'delta_daily_slip': round(delta_daily_slip, 3),
        'delta_mdd_slip': round(delta_mdd_slip, 2),
    }
    return flags, warnings, deltas


# ─── Main ─────────────────────────────────────────────────────────

def main():
    t0 = datetime.now()
    print('=' * 110)
    print('  CANDIDATE_C FULL VALIDATION (2026-04-24)')
    print(f'  Data: {ibt.n15} bars = {DATA_DAYS:.1f} days (15m resampled from 5m)')
    print(f'  Baseline : {BASELINE}  (body=0.4, no trend filter)')
    print(f'  Candidate: {CANDIDATE}  (body=0.4, no trend filter)')
    print(f'  Slippage : SLIP_MED = {SLIP_MED}')
    print('=' * 110)

    results = {}
    results['full'] = test_full_period()
    results['wf'] = test_strict_wf(results['full'])
    results['oos_split'] = test_oos_split_boundaries(results['full'])
    results['three_way'] = test_3way_split(results['full'])
    results['boot'] = test_bootstrap(results['full'])
    results['neighbor'] = test_neighborhood(
        results['full']['baseline']['slip_med']['pnl']
    )
    results['consistency'] = test_consistency(results['full'])
    results['mc'] = test_mc_direction(results['full'])
    results['slip_sens'] = test_slip_sensitivity(results['full'])
    results['bootstrap_ci'] = test_bootstrap_ci(results['full'])

    # Strip trade arrays for JSON
    for v in ('baseline', 'candidate_C'):
        for k in list(results['full'][v].keys()):
            if k.startswith('_'):
                del results['full'][v][k]

    # GO
    print('\n' + '=' * 110)
    print('  GO PROTOCOL (10 Core Flags + 4 Warnings, baseline-relative)')
    print('=' * 110)
    flags, warnings, deltas = evaluate_go(results)

    print('\n  Key Deltas:')
    for k, v in deltas.items():
        print(f'    {k}: {v:+.3f}')

    print('\n  Core Flags:')
    for k, v in flags.items():
        print(f'    {"PASS" if v else "FAIL"} {k}')
    print('\n  Warnings (log only):')
    for k, v in warnings.items():
        print(f'    {"OK" if v else "WARN"} {k}')

    core_pass = sum(1 for v in flags.values() if v)
    warn_pass = sum(1 for v in warnings.values() if v)
    verdict = ('GO' if core_pass == 10 else
               'STOP' if core_pass < 8 else 'CONDITIONAL')
    reason = f'{core_pass}/10 core passes, {warn_pass}/4 warnings pass'
    print(f'\n  CORE: {core_pass}/10  WARNINGS: {warn_pass}/4')
    print(f'  VERDICT: {verdict} — {reason}')

    # Save
    out_json = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'script': 'candidate_c_full_validation.py',
        'baseline': BASELINE,
        'candidate': CANDIDATE,
        'slippage': {'med': SLIP_MED, 'low': SLIP_LOW, 'high': SLIP_HIGH},
        'data': {'bars_15m': ibt.n15, 'data_days': round(DATA_DAYS, 1)},
        'results': results,
        'deltas': deltas,
        'core_flags': flags,
        'warnings': warnings,
        'core_pass': core_pass,
        'warn_pass': warn_pass,
        'verdict': verdict,
        'reason': reason,
        'elapsed_sec': round((datetime.now() - t0).total_seconds(), 1),
    }
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out = ROOT / 'results' / f'candidate_c_full_validation_{stamp}.json'
    out.write_text(json.dumps(out_json, indent=2, default=str))
    print(f'\n  Saved: {out.relative_to(ROOT)}')
    print(f'  Elapsed: {out_json["elapsed_sec"]}s')


if __name__ == '__main__':
    main()
