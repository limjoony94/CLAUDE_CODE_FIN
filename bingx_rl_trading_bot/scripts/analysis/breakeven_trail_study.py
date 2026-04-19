#!/usr/bin/env python3
"""Breakeven Trail study — net-loss trail 회피.

Core idea: trail이 fee+slip 차감 후 net-loss로 귀결되는 영역에서는
trail 발동하지 않고 hold. Fractal SL / Emergency가 downside 담당.

BUFFER sweep: 0.0, 0.10, 0.20, 0.30, 0.40 (%)
Combos: baseline (3.3, 2.5, 192), candidate_C (4.0, 2.5, 192)
Modes: clean bar_close, 5m + slip_med

Output: results/breakeven_trail_{timestamp}.json
"""
import sys
import os
import json
import math
import random
from pathlib import Path
from datetime import datetime, timezone
from statistics import mean, stdev

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

# ─── Monkey Patch for breakeven trail ──────────────────────────────

BREAKEVEN_BUFFER = 0.0
_ORIG_CHECK_EXIT_5M = ibt._check_exit_5m
_ORIG_CHECK_EXIT_BC = ibt._check_exit_bar_close


def _check_exit_5m_be(pos, bar15, tk):
    """Fork of ibt._check_exit_5m with breakeven guard on trail."""
    d, ep, sl = pos['d'], pos['ep'], pos['sl']
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

        # 4. Trail TP with breakeven guard
        if d == 'LONG':
            best_pnl = (bp / ep - 1) * 100
            cur_pnl = (ibt.c5[i5] / ep - 1) * 100
        else:
            best_pnl = (1 - bp / ep) * 100
            cur_pnl = (1 - ibt.c5[i5] / ep) * 100

        if (best_pnl > ibt.trail_act and not math.isnan(atr_val)
                and atr_val > 0):
            trail_dist_pct = tk * atr_val / ibt.c5[i5] * 100
            projected = best_pnl - trail_dist_pct

            # Breakeven guard (BUFFER=0: original max(0,.) behavior; BUFFER>0: strict hold)
            if BREAKEVEN_BUFFER > 0 and projected < BREAKEVEN_BUFFER:
                continue  # skip trail

            drawdown = best_pnl - cur_pnl
            if drawdown >= trail_dist_pct:
                realized = max(0, projected)  # original clamp preserved
                ex_p = (ep * (1 + realized / 100) if d == 'LONG'
                        else ep * (1 - realized / 100))
                return {'reason': 'TRAIL_TP', 'exit_price': ex_p}

    return None


def _check_exit_bc_be(pos, bar, tk):
    """Fork of ibt._check_exit_bar_close with breakeven guard."""
    d, ep, bp, sl = pos['d'], pos['ep'], pos['bp'], pos['sl']
    bh = pos['bh']

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

    # 4. Trail at bar close with breakeven guard
    if d == 'LONG':
        best_pnl = (bp / ep - 1) * 100
        cur_pnl = (ibt.c15[bar] / ep - 1) * 100
    else:
        best_pnl = (1 - bp / ep) * 100
        cur_pnl = (1 - ibt.c15[bar] / ep) * 100

    atr_val = ibt.atr14[bar]
    if (best_pnl > ibt.trail_act and not math.isnan(atr_val) and atr_val > 0):
        trail_dist_pct = tk * atr_val / ibt.c15[bar] * 100
        projected = best_pnl - trail_dist_pct

        if BREAKEVEN_BUFFER > 0 and projected < BREAKEVEN_BUFFER:
            return None  # hold

        drawdown = best_pnl - cur_pnl
        if drawdown >= trail_dist_pct:
            realized = max(0, projected)  # original clamp
            ex_p = (ep * (1 + realized / 100) if d == 'LONG'
                    else ep * (1 - realized / 100))
            return {'reason': 'TRAIL_TP', 'exit_price': ex_p}

    return None


def install_breakeven(buffer_pct):
    global BREAKEVEN_BUFFER
    BREAKEVEN_BUFFER = buffer_pct
    ibt._check_exit_5m = _check_exit_5m_be
    ibt._check_exit_bar_close = _check_exit_bc_be


def uninstall_breakeven():
    global BREAKEVEN_BUFFER
    BREAKEVEN_BUFFER = 0.0
    ibt._check_exit_5m = _ORIG_CHECK_EXIT_5M
    ibt._check_exit_bar_close = _ORIG_CHECK_EXIT_BC


# ─── Config ────────────────────────────────────────────────────────

COMBOS = {
    'baseline':    {'max_sl_atr': 3.3, 'trail_K': 2.5, 'max_hold_bars': 192},
    'candidate_C': {'max_sl_atr': 4.0, 'trail_K': 2.5, 'max_hold_bars': 192},
}

BUFFER_VALUES = [0.00, 0.10, 0.20, 0.30, 0.40]


# ─── Helpers ───────────────────────────────────────────────────────

def summarize_trades(trades):
    if not trades:
        return {'count': 0, 'PnL': 0.0, 'MDD': 0.0, 'WR': 0.0}
    pnl = sum(t['net'] for t in trades)
    mdd = compute_mdd_additive(trades)
    wr = sum(1 for t in trades if t['net'] > 0) / len(trades) * 100
    return {'count': len(trades), 'PnL': round(pnl, 2),
            'MDD': round(mdd, 2), 'WR': round(wr, 1)}


def run_clean(combo_cfg, buffer):
    install_breakeven(buffer)
    set_combo(**combo_cfg)
    trades = ibt.run_backtest(mode='bar_close')
    reset_combo()
    uninstall_breakeven()
    # Clean trades don't have 'net' — compute from raw - FEE
    for t in trades:
        if 'net' not in t:
            t['net'] = t['raw'] - ibt.FEE
    return trades


def run_slip(combo_cfg, buffer):
    install_breakeven(buffer)
    set_combo(**combo_cfg)
    trades = run_bt_with_slippage(mode='5m', slippage=SLIP_MED)
    reset_combo()
    uninstall_breakeven()
    return trades


def trail_breakdown(trades):
    """Analyze trail exits: loss vs profit."""
    trail = [t for t in trades if t.get('reason_effective',
                                          t.get('reason')) == 'TRAIL_TP']
    other = [t for t in trades if t.get('reason_effective',
                                          t.get('reason')) != 'TRAIL_TP']
    loss = [t for t in trail if t['net'] < 0]
    prof = [t for t in trail if t['net'] > 0]
    return {
        'trail_total': len(trail),
        'trail_loss_count': len(loss),
        'trail_profit_count': len(prof),
        'trail_loss_sum': round(sum(t['net'] for t in loss), 2),
        'trail_profit_sum': round(sum(t['net'] for t in prof), 2),
        'non_trail_count': len(other),
        'non_trail_pnl': round(sum(t['net'] for t in other), 2),
        'non_trail_reasons': {r: sum(1 for t in other
                                      if t.get('reason_effective',
                                                t.get('reason')) == r)
                               for r in ('SL', 'EMERGENCY', 'TIMEOUT')},
    }


def three_way_split(trades):
    n15 = ibt.n15
    warmup = 26
    t1 = warmup + int((n15 - warmup) * 0.6)
    t2 = warmup + int((n15 - warmup) * 0.8)
    tr = [t for t in trades if t['entry_bar'] <= t1]
    v = [t for t in trades if t1 < t['entry_bar'] <= t2]
    te = [t for t in trades if t['entry_bar'] > t2]

    def summ(ts):
        if not ts:
            return {'PnL': 0.0, 'MDD': 0.0, 'count': 0}
        return {'PnL': round(sum(t['net'] for t in ts), 2),
                'MDD': round(compute_mdd_additive(ts), 2),
                'count': len(ts)}
    return {'train': summ(tr), 'val': summ(v), 'test': summ(te)}


def wf_folds(trades, n_folds=5):
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
        p = sum(t['net'] for t in ts if lo <= t['entry_bar'] < hi)
        pnls.append(round(p, 2))
        if p > 0:
            pos += 1
    return pos, pnls


# ─── GO Evaluation ─────────────────────────────────────────────────

def evaluate_go_flags(base_ref, cand_result, buffer_stable):
    f = {}
    f['wf_clean_pass'] = cand_result['wf_clean'][0] == 5
    f['wf_slip_pass'] = cand_result['wf_slip'][0] == 5
    tw = cand_result['three_way_slip']
    f['tw_pass'] = all(tw[s]['PnL'] > 0 for s in ('train', 'val', 'test'))
    f['train_not_degraded'] = (
        cand_result['three_way_slip']['train']['PnL']
        >= base_ref['three_way_slip']['train']['PnL'] - 2.0
    )
    f['pnl_improvement'] = (
        cand_result['slip']['PnL'] >= base_ref['slip']['PnL'] + 5.0
    )
    c_r = (cand_result['slip']['PnL'] / cand_result['slip']['MDD']
           if cand_result['slip']['MDD'] > 0 else 0)
    b_r = (base_ref['slip']['PnL'] / base_ref['slip']['MDD']
           if base_ref['slip']['MDD'] > 0 else 0)
    f['ratio_ok'] = c_r >= b_r * 1.0
    f['buffer_stable'] = buffer_stable
    f['rollback_ready'] = True
    return f


CORE = ['wf_clean_pass', 'wf_slip_pass', 'tw_pass',
        'train_not_degraded', 'pnl_improvement']


def verdict(flags):
    for c in CORE:
        if not flags.get(c):
            return 'STOP', f'core flag {c} failed'
    n_true = sum(1 for k, v in flags.items() if v is True)
    if n_true == 8:
        return 'GO', '8/8 flags pass'
    return 'STOP', f'{n_true}/8 (need 8/8)'


# ─── Main ──────────────────────────────────────────────────────────

def main():
    t0 = datetime.now()
    print('=' * 70)
    print('  Breakeven Trail Study — net-loss trail 회피')
    print('=' * 70)
    print(f'Slippage: {SLIP_MED}')
    print()

    results = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'combos': COMBOS,
        'buffer_values': BUFFER_VALUES,
        'slippage': SLIP_MED,
        'data': {'bars_15m': ibt.n15, 'bars_5m': ibt.n5},
    }

    # ─── 20 Primary Runs ────────────────────────────────────────
    runs = {}
    breakdowns = {}
    print('--- 20 Primary Runs ---')
    print(f'{"combo":15s} {"buf":>5s} {"clean PnL":>11s} {"clean MDD":>11s} '
          f'{"slip PnL":>10s} {"slip MDD":>10s} {"slip N":>8s} '
          f'{"trail_loss":>10s} {"trail_prof":>10s}')

    for cname, ccfg in COMBOS.items():
        for buf in BUFFER_VALUES:
            tc = run_clean(ccfg, buf)
            ts = run_slip(ccfg, buf)
            sc = summarize_trades(tc)
            ss = summarize_trades(ts)
            bd = trail_breakdown(ts)

            key = f'{cname}_b{buf:.2f}'
            runs[key] = {'clean': sc, 'slip': ss}
            breakdowns[key] = bd

            print(f'{cname:15s} {buf:>5.2f} '
                  f'{sc["PnL"]:>+10.2f}% {sc["MDD"]:>10.2f} '
                  f'{ss["PnL"]:>+9.2f}% {ss["MDD"]:>9.2f} {ss["count"]:>8d} '
                  f'{bd["trail_loss_count"]:>10d}({bd["trail_loss_sum"]:>+6.1f}pp) '
                  f'{bd["trail_profit_count"]:>10d}')

    results['runs'] = runs
    results['trail_breakdown'] = breakdowns

    # Regression check
    base_b0 = runs['baseline_b0.00']
    print(f'\n--- Regression check ---')
    print(f'  baseline buf=0 clean PnL={base_b0["clean"]["PnL"]} '
          f'(expected ~169.55)')
    print(f'  baseline buf=0 slip PnL={base_b0["slip"]["PnL"]} '
          f'(expected ~46.09)')
    regression_ok = (abs(base_b0['clean']['PnL'] - 169.55) < 1.0
                     and abs(base_b0['slip']['PnL'] - 46.09) < 1.0)
    print(f'  Regression: {"OK" if regression_ok else "FAIL"}')
    results['regression_check'] = {'ok': regression_ok,
                                    'baseline_b0_clean': base_b0['clean']['PnL'],
                                    'baseline_b0_slip': base_b0['slip']['PnL']}

    # ─── Top performer 선별 ─────────────────────────────────────
    print('\n--- Top Performers (slip PnL) ---')
    sorted_runs = sorted(runs.items(), key=lambda x: x[1]['slip']['PnL'],
                         reverse=True)
    for key, r in sorted_runs[:5]:
        print(f'  {key}: slip PnL={r["slip"]["PnL"]:+.2f}% '
              f'MDD={r["slip"]["MDD"]:.2f} N={r["slip"]["count"]}')
    top_key = sorted_runs[0][0]
    top_cname, top_buf_str = top_key.rsplit('_b', 1)
    top_buf = float(top_buf_str)
    top_cfg = COMBOS[top_cname]

    # ─── Full validation on top performer ───────────────────────
    print(f'\n--- Full validation: {top_key} ---')
    tc = run_clean(top_cfg, top_buf)
    ts = run_slip(top_cfg, top_buf)

    top_val = {
        'combo': top_cname, 'buffer': top_buf,
        'clean': summarize_trades(tc),
        'slip':  summarize_trades(ts),
        'wf_clean': wf_folds(tc),
        'wf_slip':  wf_folds(ts),
        'three_way_clean': three_way_split(tc),
        'three_way_slip':  three_way_split(ts),
    }
    print(f'  WF clean: {top_val["wf_clean"][0]}/5 {top_val["wf_clean"][1]}')
    print(f'  WF slip : {top_val["wf_slip"][0]}/5 {top_val["wf_slip"][1]}')
    print(f'  3way clean: train={top_val["three_way_clean"]["train"]["PnL"]:+.2f} '
          f'val={top_val["three_way_clean"]["val"]["PnL"]:+.2f} '
          f'test={top_val["three_way_clean"]["test"]["PnL"]:+.2f}')
    print(f'  3way slip : train={top_val["three_way_slip"]["train"]["PnL"]:+.2f} '
          f'val={top_val["three_way_slip"]["val"]["PnL"]:+.2f} '
          f'test={top_val["three_way_slip"]["test"]["PnL"]:+.2f}')
    results['top_validation'] = top_val

    # Baseline reference (same combo, buf=0, full validated)
    base_tc = run_clean(top_cfg, 0.0)
    base_ts = run_slip(top_cfg, 0.0)
    base_ref = {
        'clean': summarize_trades(base_tc),
        'slip':  summarize_trades(base_ts),
        'three_way_clean': three_way_split(base_tc),
        'three_way_slip':  three_way_split(base_ts),
        'wf_clean': wf_folds(base_tc),
        'wf_slip':  wf_folds(base_ts),
    }
    results['base_reference_same_combo_b0'] = base_ref
    print(f'\n  Reference {top_cname} b0.00 slip PnL={base_ref["slip"]["PnL"]:+.2f}%')

    # ─── buffer_stable check ────────────────────────────────────
    # For top combo, count how many buffers > 0 gave slip PnL > base_ref slip
    buf_better = 0
    buf_details = {}
    for buf in BUFFER_VALUES:
        if buf == 0.0:
            continue
        key = f'{top_cname}_b{buf:.2f}'
        slip_pnl = runs[key]['slip']['PnL']
        is_better = slip_pnl > base_ref['slip']['PnL']
        buf_details[str(buf)] = {'slip_pnl': slip_pnl, 'better': is_better}
        if is_better:
            buf_better += 1
    buffer_stable = buf_better >= 2
    results['buffer_stable_check'] = {'better_count': buf_better,
                                       'details': buf_details,
                                       'passed': buffer_stable}
    print(f'\n  buffer_stable: {buf_better}/4 better → {"PASS" if buffer_stable else "FAIL"}')

    # ─── 8-flag GO Evaluation ───────────────────────────────────
    print('\n--- 8-flag GO Evaluation ---')
    flags = evaluate_go_flags(base_ref, top_val, buffer_stable)
    for k, v in flags.items():
        print(f'  {k}: {v}')
    results['go_flags'] = flags

    vdict, reason = verdict(flags)
    results['verdict'] = {'outcome': vdict, 'reason': reason,
                          'top_combo': top_cname, 'top_buffer': top_buf}
    print(f'\n=== VERDICT: {vdict} — {reason} ===')

    # ─── Save ───────────────────────────────────────────────────
    elapsed = (datetime.now() - t0).total_seconds()
    results['elapsed_sec'] = round(elapsed, 1)
    print(f'\nElapsed: {elapsed:.1f}s')

    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out = ROOT / 'results' / f'breakeven_trail_{stamp}.json'
    out.write_text(json.dumps(results, indent=2, default=str))
    print(f'Results: {out}')


if __name__ == '__main__':
    main()
