#!/usr/bin/env python3
"""True Breakeven SL Move study.

Core idea: Trail은 그대로 유지. best_pnl > ACTIVATION_PCT 시 SL을 entry로 tighten.
→ Fractal SL tail risk 제거 + trail upside 보존.

이는 breakeven_trail(BUFFER 방식, 이미 기각)과 **완전히 다른 메커니즘**.

ACTIVATION sweep: 0.10, 0.20, 0.30, 0.50, 1.00 (%)
Combos: baseline (3.3, 2.5, 192), candidate_C (4.0, 2.5, 192)
Modes: clean bar_close, 5m + slip_med

Output: results/true_breakeven_sl_move_{timestamp}.json
"""
import sys
import os
import json
import math
import random
from pathlib import Path
from datetime import datetime, timezone
from statistics import mean

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import pandas as pd

import scripts.analysis.intrabar_trail_impact as ibt
import scripts.analysis.c1_intrabar_parity as cip
from scripts.analysis.c1_intrabar_parity import (
    run_bt_with_slippage, set_combo, reset_combo,
    compute_mdd_additive, SLIPPAGE as SLIP_MED,
)
from scripts.analysis.c1_refined_bootstrap_mdd import (
    _stationary_bootstrap_indices,
)

# ─── Monkey Patch: BE SL tighten ───────────────────────────────────

BE_ACTIVATION_PCT = 0.0
_ORIG_CHECK_EXIT_5M = ibt._check_exit_5m
_ORIG_CHECK_EXIT_BC = ibt._check_exit_bar_close


def _check_exit_5m_be_sl(pos, bar15, tk):
    """Fork of _check_exit_5m with BE SL tighten before exit checks."""
    d, ep = pos['d'], pos['ep']
    bh = pos['bh']
    start_5m = bar15 * 3
    end_5m = min(start_5m + 3, ibt.n5)
    atr_val = ibt.atr14[bar15]

    for i5 in range(start_5m, end_5m):
        if d == 'LONG':
            pos['bp'] = max(pos['bp'], ibt.h5[i5])
        else:
            pos['bp'] = min(pos['bp'], ibt.l5[i5])
        bp = pos['bp']

        # NEW: BE SL activation (one-time tighten)
        if BE_ACTIVATION_PCT > 0 and not pos.get('be_activated', False):
            if d == 'LONG':
                best_pnl_sofar = (bp / ep - 1) * 100
            else:
                best_pnl_sofar = (1 - bp / ep) * 100
            if best_pnl_sofar > BE_ACTIVATION_PCT:
                if d == 'LONG':
                    pos['sl'] = max(pos['sl'], ep)
                else:
                    pos['sl'] = min(pos['sl'], ep)
                pos['be_activated'] = True

        sl = pos['sl']

        # 1. SL
        if d == 'LONG' and ibt.l5[i5] <= sl:
            return {'reason': 'SL', 'exit_price': sl}
        elif d == 'SHORT' and ibt.h5[i5] >= sl:
            return {'reason': 'SL', 'exit_price': sl}

        # 2. Emergency
        if d == 'LONG':
            worst = (ibt.l5[i5] / ep - 1) * 100
        else:
            worst = (1 - ibt.h5[i5] / ep) * 100
        if worst <= -ibt.emergency_sl:
            ex_p = (ep * (1 - ibt.emergency_sl / 100) if d == 'LONG'
                    else ep * (1 + ibt.emergency_sl / 100))
            return {'reason': 'EMERGENCY', 'exit_price': ex_p}

        # 3. Timeout
        if bh >= ibt.max_hold:
            return {'reason': 'TIMEOUT', 'exit_price': ibt.c5[i5]}

        # 4. Trail TP (기존 그대로)
        if d == 'LONG':
            best_pnl = (bp / ep - 1) * 100
            cur_pnl = (ibt.c5[i5] / ep - 1) * 100
        else:
            best_pnl = (1 - bp / ep) * 100
            cur_pnl = (1 - ibt.c5[i5] / ep) * 100

        if (best_pnl > ibt.trail_act and not math.isnan(atr_val)
                and atr_val > 0):
            trail_dist_pct = tk * atr_val / ibt.c5[i5] * 100
            drawdown = best_pnl - cur_pnl
            if drawdown >= trail_dist_pct:
                realized = max(0, best_pnl - trail_dist_pct)
                ex_p = (ep * (1 + realized / 100) if d == 'LONG'
                        else ep * (1 - realized / 100))
                return {'reason': 'TRAIL_TP', 'exit_price': ex_p}

    return None


def _check_exit_bc_be_sl(pos, bar, tk):
    """Fork of _check_exit_bar_close with BE SL tighten."""
    d, ep, bp = pos['d'], pos['ep'], pos['bp']
    bh = pos['bh']

    # NEW: BE SL activation
    if BE_ACTIVATION_PCT > 0 and not pos.get('be_activated', False):
        if d == 'LONG':
            best_pnl_sofar = (bp / ep - 1) * 100
        else:
            best_pnl_sofar = (1 - bp / ep) * 100
        if best_pnl_sofar > BE_ACTIVATION_PCT:
            if d == 'LONG':
                pos['sl'] = max(pos['sl'], ep)
            else:
                pos['sl'] = min(pos['sl'], ep)
            pos['be_activated'] = True

    sl = pos['sl']

    # 1. SL
    if d == 'LONG' and ibt.l15[bar] <= sl:
        return {'reason': 'SL', 'exit_price': sl}
    elif d == 'SHORT' and ibt.h15[bar] >= sl:
        return {'reason': 'SL', 'exit_price': sl}

    # 2. Emergency
    if d == 'LONG':
        worst = (ibt.l15[bar] / ep - 1) * 100
    else:
        worst = (1 - ibt.h15[bar] / ep) * 100
    if worst <= -ibt.emergency_sl:
        ex_p = (ep * (1 - ibt.emergency_sl / 100) if d == 'LONG'
                else ep * (1 + ibt.emergency_sl / 100))
        return {'reason': 'EMERGENCY', 'exit_price': ex_p}

    # 3. Timeout
    if bh >= ibt.max_hold:
        return {'reason': 'TIMEOUT', 'exit_price': ibt.c15[bar]}

    # 4. Trail (기존)
    if d == 'LONG':
        best_pnl = (bp / ep - 1) * 100
        cur_pnl = (ibt.c15[bar] / ep - 1) * 100
    else:
        best_pnl = (1 - bp / ep) * 100
        cur_pnl = (1 - ibt.c15[bar] / ep) * 100

    atr_val = ibt.atr14[bar]
    if (best_pnl > ibt.trail_act and not math.isnan(atr_val) and atr_val > 0):
        trail_dist_pct = tk * atr_val / ibt.c15[bar] * 100
        drawdown = best_pnl - cur_pnl
        if drawdown >= trail_dist_pct:
            realized = max(0, best_pnl - trail_dist_pct)
            ex_p = (ep * (1 + realized / 100) if d == 'LONG'
                    else ep * (1 - realized / 100))
            return {'reason': 'TRAIL_TP', 'exit_price': ex_p}

    return None


def install_be_sl(activation_pct):
    global BE_ACTIVATION_PCT
    BE_ACTIVATION_PCT = activation_pct
    ibt._check_exit_5m = _check_exit_5m_be_sl
    ibt._check_exit_bar_close = _check_exit_bc_be_sl


def uninstall_be_sl():
    global BE_ACTIVATION_PCT
    BE_ACTIVATION_PCT = 0.0
    ibt._check_exit_5m = _ORIG_CHECK_EXIT_5M
    ibt._check_exit_bar_close = _ORIG_CHECK_EXIT_BC


# ─── Config ────────────────────────────────────────────────────────

COMBOS = {
    'baseline':    {'max_sl_atr': 3.3, 'trail_K': 2.5, 'max_hold_bars': 192},
    'candidate_C': {'max_sl_atr': 4.0, 'trail_K': 2.5, 'max_hold_bars': 192},
}

ACTIVATION_VALUES = [0.00, 0.10, 0.20, 0.30, 0.50, 1.00]


# ─── Helpers (same as breakeven_trail_study) ──────────────────────

def summarize(trades):
    if not trades:
        return {'count': 0, 'PnL': 0.0, 'MDD': 0.0, 'WR': 0.0}
    pnl = sum(t['net'] for t in trades)
    mdd = compute_mdd_additive(trades)
    wr = sum(1 for t in trades if t['net'] > 0) / len(trades) * 100
    return {'count': len(trades), 'PnL': round(pnl, 2),
            'MDD': round(mdd, 2), 'WR': round(wr, 1)}


def run_clean(combo_cfg, activation):
    install_be_sl(activation)
    set_combo(**combo_cfg)
    trades = ibt.run_backtest(mode='bar_close')
    reset_combo()
    uninstall_be_sl()
    for t in trades:
        if 'net' not in t:
            t['net'] = t['raw'] - ibt.FEE
    return trades


def run_slip(combo_cfg, activation):
    install_be_sl(activation)
    set_combo(**combo_cfg)
    trades = run_bt_with_slippage(mode='5m', slippage=SLIP_MED)
    reset_combo()
    uninstall_be_sl()
    return trades


def three_way_split(trades):
    n15 = ibt.n15
    warmup = 26
    t1 = warmup + int((n15 - warmup) * 0.6)
    t2 = warmup + int((n15 - warmup) * 0.8)
    tr = [t for t in trades if t['entry_bar'] <= t1]
    v = [t for t in trades if t1 < t['entry_bar'] <= t2]
    te = [t for t in trades if t['entry_bar'] > t2]
    def s(ts):
        if not ts: return {'PnL': 0.0, 'MDD': 0.0, 'count': 0}
        return {'PnL': round(sum(t['net'] for t in ts), 2),
                'MDD': round(compute_mdd_additive(ts), 2),
                'count': len(ts)}
    return {'train': s(tr), 'val': s(v), 'test': s(te)}


def wf_folds(trades, n_folds=5):
    if not trades: return 0, []
    ts = sorted(trades, key=lambda t: t['entry_bar'])
    first, last = ts[0]['entry_bar'], ts[-1]['entry_bar']
    span = last - first
    if span < n_folds: return 0, []
    fold_size = span // n_folds
    pnls = []
    pos = 0
    for k in range(n_folds):
        lo = first + k * fold_size
        hi = first + (k + 1) * fold_size if k < n_folds - 1 else last + 1
        p = sum(t['net'] for t in ts if lo <= t['entry_bar'] < hi)
        pnls.append(round(p, 2))
        if p > 0:
            pos += 1
    return pos, pnls


def exit_reason_breakdown(trades):
    reasons = {}
    for t in trades:
        r = t.get('reason_effective', t.get('reason', 'UNK'))
        reasons[r] = reasons.get(r, 0) + 1
    return reasons


# ─── Main ──────────────────────────────────────────────────────────

def main():
    t0 = datetime.now()
    print('=' * 70)
    print('  True Breakeven SL Move Study')
    print('=' * 70)
    print(f'Slippage: {SLIP_MED}')
    print()

    results = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'combos': COMBOS,
        'activation_values': ACTIVATION_VALUES,
        'slippage': SLIP_MED,
    }

    runs = {}
    print(f'{"combo":15s} {"act":>5s} {"clean PnL":>11s} {"clean MDD":>11s} '
          f'{"slip PnL":>10s} {"slip MDD":>10s} {"slip N":>8s} '
          f'{"slip WR":>7s} {"SL%":>6s} {"TRAIL%":>7s}')

    for cname, ccfg in COMBOS.items():
        for act in ACTIVATION_VALUES:
            tc = run_clean(ccfg, act)
            ts = run_slip(ccfg, act)
            sc = summarize(tc)
            ss = summarize(ts)

            rb = exit_reason_breakdown(ts)
            sl_pct = rb.get('SL', 0) / len(ts) * 100 if ts else 0
            tr_pct = rb.get('TRAIL_TP', 0) / len(ts) * 100 if ts else 0

            key = f'{cname}_a{act:.2f}'
            runs[key] = {'clean': sc, 'slip': ss, 'reasons': rb}

            print(f'{cname:15s} {act:>5.2f} '
                  f'{sc["PnL"]:>+10.2f}% {sc["MDD"]:>10.2f} '
                  f'{ss["PnL"]:>+9.2f}% {ss["MDD"]:>9.2f} {ss["count"]:>8d} '
                  f'{ss["WR"]:>6.1f}% {sl_pct:>5.1f}% {tr_pct:>6.1f}%')

    results['runs'] = runs

    # Regression check
    base_a0 = runs['baseline_a0.00']
    regression_ok = (abs(base_a0['clean']['PnL'] - 169.55) < 1.0
                     and abs(base_a0['slip']['PnL'] - 46.09) < 1.0)
    print(f'\nRegression: {"OK" if regression_ok else "FAIL"}')
    print(f'  baseline a=0 clean={base_a0["clean"]["PnL"]} '
          f'(expected 169.55), slip={base_a0["slip"]["PnL"]} (expected 46.09)')
    results['regression_check'] = {'ok': regression_ok}

    # Top performer by slip PnL
    sorted_runs = sorted(runs.items(), key=lambda x: x[1]['slip']['PnL'],
                         reverse=True)
    print('\n--- Top 5 by slip PnL ---')
    for key, r in sorted_runs[:5]:
        print(f'  {key}: slip PnL={r["slip"]["PnL"]:+.2f}% '
              f'MDD={r["slip"]["MDD"]:.2f} N={r["slip"]["count"]}')

    top_key = sorted_runs[0][0]
    top_cname, top_act_str = top_key.rsplit('_a', 1)
    top_act = float(top_act_str)
    top_cfg = COMBOS[top_cname]

    # Full validation on top
    print(f'\n--- Full validation: {top_key} ---')
    tc = run_clean(top_cfg, top_act)
    ts = run_slip(top_cfg, top_act)
    top_val = {
        'combo': top_cname, 'activation': top_act,
        'clean': summarize(tc), 'slip': summarize(ts),
        'wf_clean': wf_folds(tc), 'wf_slip': wf_folds(ts),
        'three_way_clean': three_way_split(tc),
        'three_way_slip':  three_way_split(ts),
    }
    print(f'  WF clean: {top_val["wf_clean"][0]}/5 {top_val["wf_clean"][1]}')
    print(f'  WF slip : {top_val["wf_slip"][0]}/5 {top_val["wf_slip"][1]}')
    print(f'  3way clean: {top_val["three_way_clean"]}')
    print(f'  3way slip : {top_val["three_way_slip"]}')
    results['top_validation'] = top_val

    # Baseline ref (same combo, act=0)
    base_tc = run_clean(top_cfg, 0.0)
    base_ts = run_slip(top_cfg, 0.0)
    base_ref = {
        'clean': summarize(base_tc), 'slip': summarize(base_ts),
        'three_way_slip': three_way_split(base_ts),
    }
    results['base_ref'] = base_ref

    # activation_stable (≥2 of act ∈ {0.2, 0.3, 0.5} better than base_ref)
    act_better = 0
    act_details = {}
    for a in (0.20, 0.30, 0.50):
        key = f'{top_cname}_a{a:.2f}'
        slip_pnl = runs[key]['slip']['PnL']
        is_better = slip_pnl > base_ref['slip']['PnL']
        act_details[str(a)] = {'slip_pnl': slip_pnl, 'better': is_better}
        if is_better:
            act_better += 1
    activation_stable = act_better >= 2
    results['activation_stable_check'] = {'better_count': act_better,
                                           'details': act_details,
                                           'passed': activation_stable}

    # 8-flag GO
    print('\n--- 8-flag GO Evaluation ---')
    flags = {}
    flags['wf_clean_pass'] = top_val['wf_clean'][0] == 5
    flags['wf_slip_pass'] = top_val['wf_slip'][0] == 5
    tw = top_val['three_way_slip']
    flags['tw_pass'] = all(tw[s]['PnL'] > 0 for s in ('train', 'val', 'test'))
    flags['train_not_degraded'] = (
        tw['train']['PnL'] >= base_ref['three_way_slip']['train']['PnL'] - 2.0
    )
    flags['pnl_improvement'] = (
        top_val['slip']['PnL'] >= base_ref['slip']['PnL'] + 3.0
        or top_val['slip']['MDD'] <= base_ref['slip']['MDD'] * 0.80
    )
    c_r = (top_val['slip']['PnL'] / top_val['slip']['MDD']
           if top_val['slip']['MDD'] > 0 else 0)
    b_r = (base_ref['slip']['PnL'] / base_ref['slip']['MDD']
           if base_ref['slip']['MDD'] > 0 else 0)
    flags['ratio_ok'] = c_r >= b_r * 1.05
    flags['activation_stable'] = activation_stable
    flags['rollback_ready'] = True

    for k, v in flags.items():
        print(f'  {k}: {v}')
    results['go_flags'] = flags

    CORE = ['wf_clean_pass', 'wf_slip_pass', 'tw_pass',
            'train_not_degraded', 'pnl_improvement']
    stop_reason = None
    for c in CORE:
        if not flags[c]:
            stop_reason = f'core flag {c} failed'
            break
    true_cnt = sum(1 for v in flags.values() if v is True)
    if stop_reason:
        vdict = 'STOP'; reason = stop_reason
    elif true_cnt == 8:
        vdict = 'GO'; reason = '8/8 flags pass'
    else:
        vdict = 'STOP'; reason = f'{true_cnt}/8'
    results['verdict'] = {'outcome': vdict, 'reason': reason,
                          'top_combo': top_cname, 'top_activation': top_act}
    print(f'\n=== VERDICT: {vdict} — {reason} ===')

    elapsed = (datetime.now() - t0).total_seconds()
    results['elapsed_sec'] = round(elapsed, 1)
    print(f'\nElapsed: {elapsed:.1f}s')

    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out = ROOT / 'results' / f'true_breakeven_sl_move_{stamp}.json'
    out.write_text(json.dumps(results, indent=2, default=str))
    print(f'Results: {out}')


if __name__ == '__main__':
    main()
