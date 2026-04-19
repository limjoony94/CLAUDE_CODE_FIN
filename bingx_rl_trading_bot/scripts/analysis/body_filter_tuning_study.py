#!/usr/bin/env python3
"""Body filter tuning study — entry selectivity.

body_min_ratio 스윕: 0.25~0.60
Combos: baseline, candidate_C
Modes: clean bar_close, 5m + slip_med
"""
import sys
import os
import json
import copy
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from scripts.production.c1_breakout.signals import C1BreakoutSignal
import scripts.analysis.intrabar_trail_impact as ibt
from scripts.analysis.c1_intrabar_parity import (
    run_bt_with_slippage, compute_mdd_additive, SLIPPAGE as SLIP_MED,
)

COMBOS = {
    'baseline':    {'max_sl_atr': 3.3, 'trail_K': 2.5, 'max_hold_bars': 192},
    'candidate_C': {'max_sl_atr': 4.0, 'trail_K': 2.5, 'max_hold_bars': 192},
}
BODY_VALUES = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.60]


_ORIG_STRAT = None
_ORIG_SIG = None
_ORIG_TRAIL_K = ibt.trail_K
_ORIG_MAX_HOLD = ibt.max_hold


def set_combo_body(max_sl_atr, trail_K, max_hold_bars, body_min_ratio):
    global _ORIG_STRAT, _ORIG_SIG
    if _ORIG_STRAT is None:
        _ORIG_STRAT = copy.deepcopy(ibt.strat)
        _ORIG_SIG = ibt.sig

    new_strat = copy.deepcopy(ibt.strat)
    new_strat['max_sl_atr'] = max_sl_atr
    new_strat['trail_K'] = trail_K
    new_strat['max_hold_bars'] = max_hold_bars
    new_strat['body_min_ratio'] = body_min_ratio

    ibt.strat = new_strat
    ibt.trail_K = trail_K
    ibt.max_hold = max_hold_bars
    ibt.sig = C1BreakoutSignal(new_strat)


def reset_combo_body():
    global _ORIG_STRAT, _ORIG_SIG
    if _ORIG_STRAT is None:
        return
    ibt.strat = _ORIG_STRAT
    ibt.sig = _ORIG_SIG
    ibt.trail_K = _ORIG_TRAIL_K
    ibt.max_hold = _ORIG_MAX_HOLD


def summarize(trades):
    if not trades:
        return {'count': 0, 'PnL': 0.0, 'MDD': 0.0, 'WR': 0.0}
    pnl = sum(t['net'] for t in trades)
    mdd = compute_mdd_additive(trades)
    wr = sum(1 for t in trades if t['net'] > 0) / len(trades) * 100
    return {'count': len(trades), 'PnL': round(pnl, 2),
            'MDD': round(mdd, 2), 'WR': round(wr, 1)}


def run_clean(combo_cfg, body):
    set_combo_body(**combo_cfg, body_min_ratio=body)
    trades = ibt.run_backtest(mode='bar_close')
    reset_combo_body()
    for t in trades:
        if 'net' not in t:
            t['net'] = t['raw'] - ibt.FEE
    return trades


def run_slip(combo_cfg, body):
    set_combo_body(**combo_cfg, body_min_ratio=body)
    trades = run_bt_with_slippage(mode='5m', slippage=SLIP_MED)
    reset_combo_body()
    return trades


def main():
    t0 = datetime.now()
    print('=' * 70)
    print('  Body Filter Tuning Study (Entry Selectivity)')
    print('=' * 70)

    results = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'combos': COMBOS, 'body_values': BODY_VALUES,
    }

    runs = {}
    print(f'{"combo":15s} {"body":>6s} {"clean PnL":>11s} {"clean N":>8s} '
          f'{"slip PnL":>10s} {"slip MDD":>10s} {"slip N":>8s} '
          f'{"slip WR":>8s} {"ratio":>7s}')

    for cname, ccfg in COMBOS.items():
        for body in BODY_VALUES:
            tc = run_clean(ccfg, body)
            ts = run_slip(ccfg, body)
            sc = summarize(tc)
            ss = summarize(ts)
            ratio = ss['PnL'] / ss['MDD'] if ss['MDD'] > 0 else 0

            key = f'{cname}_b{body:.2f}'
            runs[key] = {'clean': sc, 'slip': ss, 'ratio': round(ratio, 2)}

            print(f'{cname:15s} {body:>6.2f} '
                  f'{sc["PnL"]:>+10.2f}% {sc["count"]:>8d} '
                  f'{ss["PnL"]:>+9.2f}% {ss["MDD"]:>9.2f} {ss["count"]:>8d} '
                  f'{ss["WR"]:>7.1f}% {ratio:>7.2f}')

    results['runs'] = runs

    # Regression (baseline body=0.4)
    base = runs['baseline_b0.40']
    reg_ok = abs(base['slip']['PnL'] - 46.09) < 1.0
    print(f'\nRegression (baseline body=0.40 slip): {base["slip"]["PnL"]} '
          f'(expected ~46.09) → {"OK" if reg_ok else "FAIL"}')

    # Top performers
    print('\n--- Top 5 by slip PnL ---')
    sorted_ = sorted(runs.items(), key=lambda x: x[1]['slip']['PnL'],
                     reverse=True)
    for key, r in sorted_[:5]:
        print(f'  {key}: slip PnL={r["slip"]["PnL"]:+.2f} '
              f'MDD={r["slip"]["MDD"]:.2f} N={r["slip"]["count"]} '
              f'ratio={r["ratio"]:.2f}')

    print('\n--- Top 5 by ratio (PnL/MDD) ---')
    sorted_r = sorted(runs.items(), key=lambda x: x[1]['ratio'], reverse=True)
    for key, r in sorted_r[:5]:
        print(f'  {key}: ratio={r["ratio"]:.2f} PnL={r["slip"]["PnL"]:+.2f} '
              f'MDD={r["slip"]["MDD"]:.2f} N={r["slip"]["count"]}')

    # 4-flag GO for top
    top = sorted_[0]
    top_key = top[0]
    top_combo = top_key.rsplit('_b', 1)[0]
    base_same = runs[f'{top_combo}_b0.40']
    flags = {
        'pnl_improvement': top[1]['slip']['PnL'] >= base_same['slip']['PnL'] + 5.0,
        'trade_count_ok': top[1]['slip']['count'] >= base_same['slip']['count'] * 0.70,
        'ratio_ok': top[1]['ratio'] >= base_same['ratio'],
        'rollback_ready': True,
    }
    print(f'\n--- 4-flag GO (top: {top_key}) ---')
    for k, v in flags.items():
        print(f'  {k}: {v}')
    results['go_flags'] = flags
    results['top_candidate'] = top_key

    passed = sum(1 for v in flags.values() if v)
    vdict = 'GO' if passed == 4 else 'STOP'
    print(f'\n=== VERDICT: {vdict} ({passed}/4) ===')
    results['verdict'] = {'outcome': vdict, 'flags_passed': passed}

    elapsed = (datetime.now() - t0).total_seconds()
    results['elapsed_sec'] = round(elapsed, 1)
    print(f'\nElapsed: {elapsed:.1f}s')

    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out = ROOT / 'results' / f'body_filter_tuning_{stamp}.json'
    out.write_text(json.dumps(results, indent=2, default=str))
    print(f'Results: {out}')


if __name__ == '__main__':
    main()
