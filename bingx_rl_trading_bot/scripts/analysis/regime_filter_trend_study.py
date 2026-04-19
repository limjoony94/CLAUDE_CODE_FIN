#!/usr/bin/env python3
"""Trend filter study — fold 1/2 구분 가능성.

Rolling trend % > THR 시 진입 허용. Fold 2(-2.6%)만 차단, Fold 1(+24%)은 통과 기대.

Grid: 5 THR × 2 LOOKBACK = 10 runs.
"""
import sys
import os
import json
import math
import copy
from pathlib import Path
from datetime import datetime, timezone
from statistics import mean

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from scripts.production.c1_breakout.signals import C1BreakoutSignal
import scripts.analysis.intrabar_trail_impact as ibt
from scripts.analysis.c1_intrabar_parity import (
    compute_mdd_additive, SLIPPAGE as SLIP_MED, apply_slippage,
)
from scripts.analysis.regime_filter_lowvol_study import (
    run_bt_with_regime, summarize, fold_breakdown, wf_5fold,
)

CANDIDATE = {'max_sl_atr': 4.0, 'trail_K': 2.5, 'max_hold_bars': 192,
             'body_min_ratio': 0.60}

# Trend filter:
#   abs(trend_pct) > THR 시 진입 허용 (trending market)
#   abs(trend_pct) <= THR 시 skip (choppy/low-trend)
THR_VALUES = [1.0, 2.0, 3.0, 5.0, 8.0]  # % (absolute trend over lookback)
LOOKBACK_VALUES = [192, 384]  # 2-day, 4-day

FOLD_BOUNDARIES = {
    'fold_1': (31,    6407),   'fold_2': (6407,  12783),
    'fold_3': (12783, 19159),  'fold_4': (19159, 25535),
    'fold_5': (25535, 31916),
}

_ORIG_STRAT = copy.deepcopy(ibt.strat)
_ORIG_SIG = ibt.sig


def set_combo_body(cfg):
    new_s = copy.deepcopy(_ORIG_STRAT)
    new_s.update(cfg)
    ibt.strat = new_s
    ibt.trail_K = cfg['trail_K']
    ibt.max_hold = cfg['max_hold_bars']
    ibt.sig = C1BreakoutSignal(new_s)


def reset_combo_body():
    ibt.strat = copy.deepcopy(_ORIG_STRAT)
    ibt.sig = _ORIG_SIG


def precompute_trend_pass(lookback, thr_pct):
    """bar i에서 abs(trend) > thr (trending) 시 pass, else skip."""
    n = ibt.n15
    passes = [True] * n  # warmup default pass (but we'll fail if any NaN)
    for i in range(lookback, n):
        c0 = ibt.c15[i - lookback]
        ci = ibt.c15[i]
        if c0 <= 0:
            continue
        trend = (ci / c0 - 1) * 100
        passes[i] = abs(trend) > thr_pct
    return passes


def main():
    t0 = datetime.now()
    print('=' * 70)
    print('  Trend Filter — fold 1 vs fold 2 구분 시도')
    print(f'  Base: candidate_C_b0.60 = {CANDIDATE}')
    print('=' * 70)

    results = {'timestamp': datetime.now(timezone.utc).isoformat(),
               'candidate': CANDIDATE,
               'thr_values': THR_VALUES, 'lookback_values': LOOKBACK_VALUES}

    set_combo_body(CANDIDATE)

    # Baseline (no filter)
    all_pass = [True] * ibt.n15
    trades_slip_base = run_bt_with_regime(mode='5m', regime_passes=all_pass,
                                           slippage=SLIP_MED)
    base_slip = summarize(trades_slip_base)
    base_fold = fold_breakdown(trades_slip_base)
    base_wf = wf_5fold(trades_slip_base)
    print(f'\nBase (no filter): slip PnL={base_slip["PnL"]:+.2f} '
          f'MDD={base_slip["MDD"]:.2f} N={base_slip["count"]} '
          f'WF={base_wf[0]}/5')
    folds_str = [f'{base_fold["fold_" + str(i)]["PnL"]:+.2f}' for i in range(1, 6)]
    print(f'  Folds: {folds_str}')
    results['base'] = {'slip': base_slip, 'fold': base_fold, 'wf': base_wf}

    print('\n--- Grid ---')
    print(f'{"THR":>5s} {"LB":>5s} {"slip PnL":>10s} {"MDD":>7s} {"N":>6s} '
          f'{"WF":>5s} {"fold1":>8s} {"fold2":>8s} {"fold3":>8s} '
          f'{"fold4":>8s} {"fold5":>8s}')

    grid = {}
    for thr in THR_VALUES:
        for lb in LOOKBACK_VALUES:
            passes = precompute_trend_pass(lb, thr)
            trades = run_bt_with_regime(mode='5m', regime_passes=passes,
                                         slippage=SLIP_MED)
            s = summarize(trades)
            f = fold_breakdown(trades)
            w = wf_5fold(trades)
            key = f'thr{thr:.1f}_lb{lb}'
            grid[key] = {'slip': s, 'fold': f, 'wf': w,
                          'thr': thr, 'lookback': lb}
            print(f'{thr:>5.1f} {lb:>5d} {s["PnL"]:>+9.2f}% {s["MDD"]:>6.2f} '
                  f'{s["count"]:>6d} {w[0]}/{5} '
                  f'{f["fold_1"]["PnL"]:>+7.2f} {f["fold_2"]["PnL"]:>+7.2f} '
                  f'{f["fold_3"]["PnL"]:>+7.2f} {f["fold_4"]["PnL"]:>+7.2f} '
                  f'{f["fold_5"]["PnL"]:>+7.2f}')

    results['grid'] = grid
    reset_combo_body()

    # Best: fold 2 양수 AND fold 1 양수 유지 AND 전체 개선
    print('\n--- Best combos (fold 2 양수 + fold 1 양수) ---')
    both_positive = [(k, r) for k, r in grid.items()
                     if r['fold']['fold_2']['PnL'] > 0
                     and r['fold']['fold_1']['PnL'] > 0]
    if both_positive:
        both_positive.sort(key=lambda x: x[1]['slip']['PnL'], reverse=True)
        for k, r in both_positive[:5]:
            print(f'  {k}: slip PnL={r["slip"]["PnL"]:+.2f} '
                  f'fold1={r["fold"]["fold_1"]["PnL"]:+.2f} '
                  f'fold2={r["fold"]["fold_2"]["PnL"]:+.2f} '
                  f'WF={r["wf"][0]}/5')
        top_key, top_r = both_positive[0]
    else:
        print('No combo with both folds positive.')
        # fallback: best fold 2
        sorted_f2 = sorted(grid.items(),
                            key=lambda x: (x[1]['fold']['fold_2']['PnL'],
                                           x[1]['slip']['PnL']),
                            reverse=True)
        for k, r in sorted_f2[:3]:
            print(f'  {k}: slip PnL={r["slip"]["PnL"]:+.2f} '
                  f'fold1={r["fold"]["fold_1"]["PnL"]:+.2f} '
                  f'fold2={r["fold"]["fold_2"]["PnL"]:+.2f} '
                  f'WF={r["wf"][0]}/5')
        top_key, top_r = sorted_f2[0]

    # 3-flag GO
    flags = {
        'fold2_positive': top_r['fold']['fold_2']['PnL'] > 0,
        'fold1_preserved': top_r['fold']['fold_1']['PnL'] > 0,
        'wf_5of5': top_r['wf'][0] == 5,
    }
    print(f'\n--- 3-flag GO (top: {top_key}) ---')
    for k, v in flags.items():
        print(f'  {k}: {v}')
    passed = sum(1 for v in flags.values() if v)
    verdict = 'GO' if passed == 3 else 'STOP'
    print(f'\n=== VERDICT: {verdict} ({passed}/3) ===')

    results['top'] = top_key
    results['go_flags'] = flags
    results['verdict'] = {'outcome': verdict, 'passed': passed}

    elapsed = (datetime.now() - t0).total_seconds()
    results['elapsed_sec'] = round(elapsed, 1)
    print(f'\nElapsed: {elapsed:.1f}s')

    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out = ROOT / 'results' / f'regime_filter_trend_{stamp}.json'
    out.write_text(json.dumps(results, indent=2, default=str))
    print(f'Results: {out}')


if __name__ == '__main__':
    main()
