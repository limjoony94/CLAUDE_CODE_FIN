#!/usr/bin/env python3
"""Progressive Trail Full Validation — 9-Test Battery

Design: docs/02-design/features/progressive_trail.design.md
Plan:   docs/01-plan/features/progressive_trail.plan.md

Tests:
  1. Full Period (Clean + Slip3)
  2. Strict Expanding WF 5-fold
  3. OOS Split 5 Boundaries
  4. 3-Way Split (60/20/20)
  5. Bootstrap 3-day × 1000
  5b. Bootstrap Relative (vs baseline per-window)
  6. Neighborhood 5×5 (sharp peak check)
  7. Parameter Consistency 2-half
  8. MC Direction 999 sims
  9. Fold 2 Regime Re-check
"""
import sys, copy, math, random, json
from datetime import datetime
from pathlib import Path
from statistics import mean, stdev, median

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
import os; os.chdir(ROOT)

import scripts.analysis.intrabar_trail_impact as ibt
from scripts.production.c1_breakout.signals import C1BreakoutSignal
from scripts.analysis.c1_intrabar_parity import SLIPPAGE, apply_slippage
from scripts.analysis.regime_filter_lowvol_study import run_bt_with_regime
from scripts.analysis.regime_filter_trend_study import precompute_trend_pass
from scripts.analysis.progressive_trail_extended import make_check_exit_profit

DATA_DAYS = ibt.n15 / 96
WARMUP = 26
BOOT_N = 1000
BOOT_W = 288
BOOT_MIN = 220
SEED = 42
FEE = 0.10

SLIP_LOW = {k: v*0.5 for k,v in SLIPPAGE.items()}
SLIP_MED = SLIPPAGE
SLIP_HIGH = {k: v*2.0 for k,v in SLIPPAGE.items()}

# TARGET variant (from extended study)
TARGET = {'tk_base': 2.5, 'tk_post': 0.5, 'threshold': 0.9}
BASELINE = {'tk_base': 2.5, 'tk_post': 2.5, 'threshold': 99.0}


# ─── BT runner ────────────────────────────────────────────────────
def run_variant(tk_base, tk_post, threshold, slippage, regime_passes):
    """Run BT with progressive exit, return trade list (net-adjusted)."""
    orig = ibt._check_exit_bar_close
    ibt._check_exit_bar_close = make_check_exit_profit(tk_base, tk_post, threshold)
    try:
        return run_bt_with_regime(mode='bar_close', regime_passes=regime_passes,
                                   slippage=slippage)
    finally:
        ibt._check_exit_bar_close = orig


def compute_metrics(trades):
    if not trades:
        return {'n':0,'pnl':0,'daily':0,'wr':0,'rr':0,'mdd':0,
                'mean_net':0,'tpd':0,'ex_top5':0}
    total = sum(t['net'] for t in trades)
    wins = [t for t in trades if t['net']>0]
    losses = [t for t in trades if t['net']<0]
    aw = mean(t['net'] for t in wins) if wins else 0
    al = abs(mean(t['net'] for t in losses)) if losses else 0
    eq=0; pk=0; md=0
    for t in trades:
        eq+=t['net']; pk=max(pk,eq); md=max(md,pk-eq)
    n_top = max(1,int(len(trades)*0.05))
    ex_top = sorted(trades, key=lambda t:t['net'],reverse=True)[n_top:]
    return {
        'n': len(trades),'pnl': round(total,2),
        'daily': round(total/DATA_DAYS,3),
        'wr': round(len(wins)/len(trades)*100,1),
        'rr': round(aw/al if al>0 else 0,2),
        'mdd': round(md,2),
        'mean_net': round(total/len(trades),3),
        'tpd': round(len(trades)/DATA_DAYS,2),
        'ex_top5': round(sum(t['net'] for t in ex_top),2),
    }


# ─── Test 1: Full period ─────────────────────────────────────────
def test_full_period(passes):
    print('\n' + '='*110)
    print('  TEST 1: Full Period (Clean + Slip3)')
    print('='*110)
    variants = {'baseline': BASELINE, 'target': TARGET}
    scenarios = {'clean': None, 'slip_low': SLIP_LOW, 'slip_med': SLIP_MED, 'slip_high': SLIP_HIGH}
    results = {}
    for vname, v in variants.items():
        results[vname] = {}
        for sname, slip in scenarios.items():
            trades = run_variant(v['tk_base'], v['tk_post'], v['threshold'], slip, passes)
            m = compute_metrics(trades)
            results[vname][sname] = m
            print(f'  {vname:<10} {sname:<10}: n={m["n"]:>4}, PnL={m["pnl"]:>+7.2f}, '
                  f'MDD={m["mdd"]:>5.2f}, daily={m["daily"]:>+6.3f}, ex5={m["ex_top5"]:>+7.2f}')
    # Deltas
    print('\n  Deltas (target − baseline):')
    for sname in scenarios:
        d_pnl = results['target'][sname]['pnl'] - results['baseline'][sname]['pnl']
        d_mdd = results['target'][sname]['mdd'] - results['baseline'][sname]['mdd']
        d_ex5 = results['target'][sname]['ex_top5'] - results['baseline'][sname]['ex_top5']
        print(f'    {sname:<10}: ΔPnL={d_pnl:>+7.2f}, ΔMDD={d_mdd:>+5.2f}, Δex5={d_ex5:>+7.2f}')
    return results


# ─── Test 2: Strict Expanding WF ─────────────────────────────────
def strict_expanding_wf(trades, n=5):
    """5-fold strict expanding: each fold OOS is chunks of (n15-WARMUP)/n."""
    bars = ibt.n15
    oos_w = (bars - WARMUP) // n
    fold_pnls = []
    for i in range(n):
        s = WARMUP + i*oos_w
        e = s + oos_w
        p = sum(t['net'] for t in trades if s <= t['entry_bar'] < e)
        fold_pnls.append(round(p,2))
    return fold_pnls


def test_strict_wf(passes):
    print('\n' + '='*110)
    print('  TEST 2: Strict Expanding WF 5-fold')
    print('='*110)
    base_trades = run_variant(**BASELINE, slippage=SLIP_MED, regime_passes=passes)
    target_trades = run_variant(**TARGET, slippage=SLIP_MED, regime_passes=passes)
    base_folds = strict_expanding_wf(base_trades)
    target_folds = strict_expanding_wf(target_trades)
    print(f'  {"Variant":<10} {"f1":>7} {"f2":>7} {"f3":>7} {"f4":>7} {"f5":>7} {"pos":>4}')
    for name, folds in [('baseline', base_folds),('target', target_folds)]:
        pos = sum(1 for p in folds if p>0)
        fstr = ' '.join(f'{p:>+7.2f}' for p in folds)
        print(f'  {name:<10} {fstr}  {pos}/5')
    return {
        'baseline_folds': base_folds, 'baseline_pos': sum(1 for p in base_folds if p>0),
        'target_folds': target_folds, 'target_pos': sum(1 for p in target_folds if p>0),
    }


# ─── Test 3: OOS Split 5 Boundaries ──────────────────────────────
def three_way_split(trades, ratios):
    bars = ibt.n15
    tr_end = int(bars*ratios[0])
    vl_end = tr_end + int(bars*ratios[1])
    train = [t for t in trades if t['entry_bar'] < tr_end]
    val = [t for t in trades if tr_end <= t['entry_bar'] < vl_end]
    test = [t for t in trades if t['entry_bar'] >= vl_end]
    return {'train': round(sum(t['net'] for t in train),2),
            'val':   round(sum(t['net'] for t in val),2),
            'test':  round(sum(t['net'] for t in test),2)}


def test_oos_split_boundaries(passes):
    print('\n' + '='*110)
    print('  TEST 3: OOS Split 5 Boundaries')
    print('='*110)
    bounds = [(0.60,0.20,0.20),(0.50,0.25,0.25),(0.70,0.15,0.15),
              (0.55,0.15,0.30),(0.60,0.15,0.25)]
    target_trades = run_variant(**TARGET, slippage=SLIP_MED, regime_passes=passes)
    base_trades = run_variant(**BASELINE, slippage=SLIP_MED, regime_passes=passes)
    results = []
    print(f'  {"Split":<16} {"base_tr":>8} {"base_vl":>8} {"base_te":>8}  '
          f'{"tgt_tr":>8} {"tgt_vl":>8} {"tgt_te":>8}  {"vl&te>0":>8}')
    for b in bounds:
        bs = three_way_split(base_trades, b)
        ts = three_way_split(target_trades, b)
        ok = ts['val'] > 0 and ts['test'] > 0
        results.append({'split': b, 'base': bs, 'target': ts, 'pass': ok})
        label = '/'.join(str(int(x*100)) for x in b)
        print(f'  {label:<16} {bs["train"]:>+8.2f} {bs["val"]:>+8.2f} {bs["test"]:>+8.2f}  '
              f'{ts["train"]:>+8.2f} {ts["val"]:>+8.2f} {ts["test"]:>+8.2f}  {"PASS" if ok else "FAIL":>8}')
    passed = sum(1 for r in results if r['pass'])
    print(f'\n  Summary: {passed}/5 splits val&test both positive (target)')
    return {'results': results, 'pass_count': passed}


# ─── Test 5+5b: Bootstrap 3-day ───────────────────────────────────
def bootstrap_3day(trades):
    rng = random.Random(SEED)
    pnls = []
    for _ in range(BOOT_N):
        s = rng.randint(BOOT_MIN, ibt.n15 - BOOT_W - 1)
        e = s + BOOT_W
        pnls.append(sum(t['net'] for t in trades if s <= t['entry_bar'] < e))
    ps = sorted(pnls)
    return {
        'mean': round(mean(pnls),3),
        'median': round(median(pnls),3),
        'std': round(stdev(pnls),3),
        'pos_pct': round(sum(1 for p in pnls if p>0)/BOOT_N*100,1),
        'sharpe': round(mean(pnls)/stdev(pnls) if stdev(pnls)>0 else 0,3),
        'p5': round(ps[50],2),
        'p_loss_2pp': round(sum(1 for p in pnls if p<-2)/BOOT_N*100,1),
    }


def bootstrap_relative(cand_trades, base_trades):
    rng = random.Random(SEED)
    cw = 0
    for _ in range(BOOT_N):
        s = rng.randint(BOOT_MIN, ibt.n15 - BOOT_W - 1)
        e = s + BOOT_W
        cp = sum(t['net'] for t in cand_trades if s <= t['entry_bar'] < e)
        bp = sum(t['net'] for t in base_trades if s <= t['entry_bar'] < e)
        if cp > bp: cw += 1
    return round(cw/BOOT_N*100,1)


def test_bootstrap(passes):
    print('\n' + '='*110)
    print('  TEST 5: Bootstrap 3-day × 1000')
    print('='*110)
    target_trades = run_variant(**TARGET, slippage=SLIP_MED, regime_passes=passes)
    base_trades = run_variant(**BASELINE, slippage=SLIP_MED, regime_passes=passes)
    bs_tgt = bootstrap_3day(target_trades)
    bs_base = bootstrap_3day(base_trades)
    print(f'  {"Variant":<10} {"mean":>7} {"median":>7} {"pos%":>6} {"sharpe":>7} '
          f'{"p5":>6} {"p_loss_2":>8}')
    for name, b in [('baseline', bs_base),('target', bs_tgt)]:
        print(f'  {name:<10} {b["mean"]:>+7.3f} {b["median"]:>+7.3f} {b["pos_pct"]:>5.1f}% '
              f'{b["sharpe"]:>+7.3f} {b["p5"]:>+6.2f} {b["p_loss_2pp"]:>7.1f}%')
    rel = bootstrap_relative(target_trades, base_trades)
    print(f'\n  TEST 5b: P(target > baseline per-window) = {rel}%')
    return {'baseline': bs_base, 'target': bs_tgt, 'relative': rel}


# ─── Test 4: 3-Way Split ──────────────────────────────────────────
def test_3way_split(passes):
    print('\n' + '='*110)
    print('  TEST 4: 3-Way Split (60/20/20)')
    print('='*110)
    target_trades = run_variant(**TARGET, slippage=SLIP_MED, regime_passes=passes)
    base_trades = run_variant(**BASELINE, slippage=SLIP_MED, regime_passes=passes)
    bs = three_way_split(base_trades, (0.6,0.2,0.2))
    ts = three_way_split(target_trades, (0.6,0.2,0.2))
    print(f'  baseline: train={bs["train"]:+.2f}, val={bs["val"]:+.2f}, test={bs["test"]:+.2f}')
    print(f'  target:   train={ts["train"]:+.2f}, val={ts["val"]:+.2f}, test={ts["test"]:+.2f}')
    all_pos = ts['train']>0 and ts['val']>0 and ts['test']>0
    print(f'  target all positive: {"PASS" if all_pos else "FAIL"}')
    return {'baseline': bs, 'target': ts, 'target_all_pos': all_pos}


# ─── Test 6: Neighborhood 5×5 ─────────────────────────────────────
def test_neighborhood(passes, baseline_pnl):
    print('\n' + '='*110)
    print('  TEST 6: Neighborhood 5×5 (sharp peak check)')
    print('='*110)
    thrs = [0.7, 0.8, 0.9, 1.0, 1.1]
    tks = [0.3, 0.4, 0.5, 0.6, 0.7]
    results = []
    print(f'  Baseline PnL = {baseline_pnl:+.2f} (SLIP_MED)')
    print(f'  GO criterion: combo PnL >= baseline + 10pp = {baseline_pnl+10:+.2f}')
    print(f'\n  {"thr":>5} {"tkT":>5} {"PnL":>8} {"ex5":>8} {"GO":>5}')
    go_count = 0
    for thr in thrs:
        for tk in tks:
            trades = run_variant(2.5, tk, thr, SLIP_MED, passes)
            m = compute_metrics(trades)
            go = m['pnl'] >= baseline_pnl + 10
            if go: go_count += 1
            results.append({'thr':thr, 'tk':tk, 'pnl':m['pnl'],
                           'ex5':m['ex_top5'], 'go':go})
            print(f'  {thr:>5.2f} {tk:>5.2f} {m["pnl"]:>+7.2f} {m["ex_top5"]:>+7.2f} '
                  f'{"GO" if go else "-":>5}')
    print(f'\n  GO count: {go_count}/25 (need >= 8)')
    return {'results': results, 'go_count': go_count}


# ─── Test 7: Parameter Consistency 2-half ─────────────────────────
def test_consistency(passes):
    print('\n' + '='*110)
    print('  TEST 7: Parameter Consistency 2-half')
    print('='*110)
    mid = ibt.n15 // 2
    target_trades = run_variant(**TARGET, slippage=SLIP_MED, regime_passes=passes)
    base_trades = run_variant(**BASELINE, slippage=SLIP_MED, regime_passes=passes)
    h1c = sum(t['net'] for t in target_trades if t['entry_bar'] < mid)
    h2c = sum(t['net'] for t in target_trades if t['entry_bar'] >= mid)
    h1b = sum(t['net'] for t in base_trades if t['entry_bar'] < mid)
    h2b = sum(t['net'] for t in base_trades if t['entry_bar'] >= mid)
    d1 = h1c - h1b
    d2 = h2c - h2b
    print(f'  1st half: base={h1b:+.2f}, target={h1c:+.2f}, Δ={d1:+.2f}')
    print(f'  2nd half: base={h2b:+.2f}, target={h2c:+.2f}, Δ={d2:+.2f}')
    ok = d1 >= 5 and d2 >= 5
    print(f'  Structural delta both halves >= +5pp: {"PASS" if ok else "FAIL"}')
    return {'h1_cand':round(h1c,2), 'h2_cand':round(h2c,2),
            'h1_base':round(h1b,2), 'h2_base':round(h2b,2),
            'h1_delta':round(d1,2), 'h2_delta':round(d2,2), 'pass':ok}


# ─── Test 8: MC Direction ─────────────────────────────────────────
def test_mc_direction(passes):
    print('\n' + '='*110)
    print('  TEST 8: MC Direction 999 sims')
    print('='*110)
    target_trades = run_variant(**TARGET, slippage=SLIP_MED, regime_passes=passes)
    real = sum(t['net'] for t in target_trades)
    nets = [t['net'] for t in target_trades]
    rng = random.Random(SEED)
    beat = 0
    N = 999
    for _ in range(N):
        shuffled = sum(abs(n) * rng.choice([-1, 1]) for n in nets)
        if real > shuffled: beat += 1
    p = 1 - beat/N
    print(f'  Real PnL = {real:+.2f}')
    print(f'  Beat {beat}/{N} sims → p-value = {p:.4f}')
    print(f'  p < 0.01: {"PASS" if p < 0.01 else "FAIL (warning)"}')
    return {'p_value': round(p,4), 'real_pnl': round(real,2)}


# ─── Test 9: Fold 2 Regime Re-check ───────────────────────────────
def test_fold2(passes):
    print('\n' + '='*110)
    print('  TEST 9: Fold 2 Regime Re-check')
    print('='*110)
    target_trades = run_variant(**TARGET, slippage=SLIP_MED, regime_passes=passes)
    base_trades = run_variant(**BASELINE, slippage=SLIP_MED, regime_passes=passes)
    oos_w = (ibt.n15 - WARMUP) // 5
    f2s = WARMUP + 1*oos_w
    f2e = f2s + oos_w
    cf2 = sum(t['net'] for t in target_trades if f2s <= t['entry_bar'] < f2e)
    bf2 = sum(t['net'] for t in base_trades if f2s <= t['entry_bar'] < f2e)
    d = cf2 - bf2
    print(f'  Fold 2 bars [{f2s}, {f2e})')
    print(f'  baseline fold2: {bf2:+.2f}')
    print(f'  target fold2:   {cf2:+.2f}')
    print(f'  Δ = {d:+.2f}')
    ok = cf2 > 0 and d >= 3
    print(f'  cand>0 AND Δ>=3pp: {"PASS" if ok else "FAIL (warning)"}')
    return {'baseline_fold2':round(bf2,2), 'target_fold2':round(cf2,2),
            'delta':round(d,2), 'pass':ok}


# ─── GO Protocol ──────────────────────────────────────────────────
def evaluate_go(results):
    r = results
    tgt_full_clean = r['full']['target']['clean']
    base_full_clean = r['full']['baseline']['clean']
    tgt_full_med = r['full']['target']['slip_med']
    base_full_med = r['full']['baseline']['slip_med']

    flags = {
        'f1_full_clean_gain':      tgt_full_clean['pnl'] - base_full_clean['pnl'] >= 10,
        'f2_full_slip_med_gain':   tgt_full_med['pnl'] - base_full_med['pnl'] >= 10,
        'f3_strict_wf_5of5':       r['wf']['target_pos'] == 5,
        'f4_oos_split_5of5':       r['oos_split']['pass_count'] == 5,
        'f5_3way_all_positive':    r['3way']['target_all_pos'],
        'f6_bootstrap_core':       (r['boot']['target']['mean']>0
                                    and r['boot']['target']['pos_pct']>=55
                                    and r['boot']['target']['p5']>=-3.5),
        'f7_bootstrap_relative':   r['boot']['relative'] >= 55,
        'f8_neighborhood_8plus':   r['neighbor']['go_count'] >= 8,
        'f9_structural_halves':    r['consistency']['pass'],
    }
    warnings = {
        'w1_mc_p_under_0.01':      r['mc']['p_value'] < 0.01,
        'w2_f6_ex_top5_pass':      tgt_full_med['ex_top5'] > 0,
        'w3_mdd_improved':         base_full_med['mdd'] - tgt_full_med['mdd'] >= 1.0,
        'w4_sharpe_0.45':          r['boot']['target']['sharpe'] >= 0.45,
        'w5_fold2_pass':           r['fold2']['pass'],
    }
    return flags, warnings


# ─── Main ─────────────────────────────────────────────────────────
def main():
    # Base strategy setup (cand_C_b0.60 + trend)
    base_strat = {'max_sl_atr':4.0,'trail_K':2.5,'max_hold_bars':192,'body_min_ratio':0.60}
    ibt.strat = copy.deepcopy(ibt.strat); ibt.strat.update(base_strat)
    ibt.trail_K = 2.5; ibt.max_hold = 192
    ibt.sig = C1BreakoutSignal(ibt.strat)
    passes = precompute_trend_pass(192, 1.0)

    print('='*110)
    print('  PROGRESSIVE TRAIL — FULL VALIDATION (9 Tests)')
    print(f'  Data: {ibt.n15} bars = {DATA_DAYS:.1f} days')
    print(f'  Base strategy: cand_C_b0.60 + trend_filter(1.0%/192)')
    print(f'  Target: tk_base={TARGET["tk_base"]}, tk_post={TARGET["tk_post"]}, threshold={TARGET["threshold"]}%')
    print('='*110)

    results = {}
    results['full']       = test_full_period(passes)
    results['wf']         = test_strict_wf(passes)
    results['oos_split']  = test_oos_split_boundaries(passes)
    results['3way']       = test_3way_split(passes)
    results['boot']       = test_bootstrap(passes)
    results['neighbor']   = test_neighborhood(passes, results['full']['baseline']['slip_med']['pnl'])
    results['consistency']= test_consistency(passes)
    results['mc']         = test_mc_direction(passes)
    results['fold2']      = test_fold2(passes)

    # GO
    print('\n' + '='*110)
    print('  GO PROTOCOL — 9 Core + 5 Warnings')
    print('='*110)
    flags, warnings = evaluate_go(results)
    for k,v in flags.items():
        print(f'  {"✅" if v else "❌"} {k}')
    print()
    for k,v in warnings.items():
        print(f'  {"✓" if v else "⚠"} {k} (warning)')
    core_pass = sum(1 for v in flags.values() if v)
    warn_pass = sum(1 for v in warnings.values() if v)
    print(f'\n  CORE: {core_pass}/9, WARNINGS: {warn_pass}/5')
    verdict = 'FULL GO — deploy-ready (enabled=false)' if core_pass == 9 else f'STOP — {9-core_pass} core fail'
    print(f'  VERDICT: {verdict}')

    # Save
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out = ROOT / 'results' / f'progressive_trail_validation_{stamp}.json'
    out.write_text(json.dumps({
        'timestamp': stamp,
        'target': TARGET, 'baseline': BASELINE,
        'results': results, 'flags': flags, 'warnings': warnings,
        'core_pass': core_pass, 'warning_pass': warn_pass, 'verdict': verdict,
    }, indent=2, default=str))
    print(f'\n  Saved: {out.relative_to(ROOT)}')


if __name__ == '__main__':
    main()
