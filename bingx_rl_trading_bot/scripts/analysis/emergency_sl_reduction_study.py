#!/usr/bin/env python3
"""Emergency SL reduction study.

Emergency SL: fractal SL의 2차 보호. 현재 3.0%. 축소 시 tail 효과 측정.

Sweep: 1.5%, 2.0%, 2.5%, 3.0% × baseline/candidate_C × clean/slip
"""
import sys
import os
import json
import math
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import scripts.analysis.intrabar_trail_impact as ibt
from scripts.analysis.c1_intrabar_parity import (
    run_bt_with_slippage, set_combo, reset_combo,
    compute_mdd_additive, SLIPPAGE as SLIP_MED,
)

COMBOS = {
    'baseline':    {'max_sl_atr': 3.3, 'trail_K': 2.5, 'max_hold_bars': 192},
    'candidate_C': {'max_sl_atr': 4.0, 'trail_K': 2.5, 'max_hold_bars': 192},
}
ESL_VALUES = [1.5, 2.0, 2.5, 3.0]

_ORIG_ESL = ibt.emergency_sl


def set_esl(val):
    ibt.emergency_sl = val


def reset_esl():
    ibt.emergency_sl = _ORIG_ESL


def summarize(trades):
    if not trades:
        return {'count': 0, 'PnL': 0.0, 'MDD': 0.0, 'WR': 0.0}
    pnl = sum(t['net'] for t in trades)
    mdd = compute_mdd_additive(trades)
    wr = sum(1 for t in trades if t['net'] > 0) / len(trades) * 100
    return {'count': len(trades), 'PnL': round(pnl, 2),
            'MDD': round(mdd, 2), 'WR': round(wr, 1)}


def run_clean(combo_cfg, esl):
    set_esl(esl)
    set_combo(**combo_cfg)
    trades = ibt.run_backtest(mode='bar_close')
    reset_combo()
    reset_esl()
    for t in trades:
        if 'net' not in t:
            t['net'] = t['raw'] - ibt.FEE
    return trades


def run_slip(combo_cfg, esl):
    set_esl(esl)
    set_combo(**combo_cfg)
    trades = run_bt_with_slippage(mode='5m', slippage=SLIP_MED)
    reset_combo()
    reset_esl()
    return trades


def reason_breakdown(trades):
    r = {}
    for t in trades:
        k = t.get('reason_effective', t.get('reason', 'UNK'))
        r[k] = r.get(k, 0) + 1
    return r


def main():
    t0 = datetime.now()
    print('=' * 70)
    print('  Emergency SL Reduction Study')
    print('=' * 70)
    results = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'combos': COMBOS, 'esl_values': ESL_VALUES,
    }

    runs = {}
    print(f'{"combo":15s} {"esl":>5s} {"clean PnL":>10s} {"slip PnL":>10s} '
          f'{"slip MDD":>10s} {"slip N":>7s} {"SL":>5s} {"EMG":>5s} '
          f'{"TR":>5s} {"TO":>5s}')

    for cname, ccfg in COMBOS.items():
        for esl in ESL_VALUES:
            tc = run_clean(ccfg, esl)
            ts = run_slip(ccfg, esl)
            sc = summarize(tc)
            ss = summarize(ts)
            rb_slip = reason_breakdown(ts)
            rb_clean = reason_breakdown(tc)

            key = f'{cname}_esl{esl:.1f}'
            runs[key] = {
                'clean': sc, 'slip': ss,
                'reasons_slip': rb_slip, 'reasons_clean': rb_clean,
            }

            print(f'{cname:15s} {esl:>5.1f} '
                  f'{sc["PnL"]:>+9.2f}% {ss["PnL"]:>+9.2f}% '
                  f'{ss["MDD"]:>10.2f} {ss["count"]:>7d} '
                  f'{rb_slip.get("SL",0):>5d} {rb_slip.get("EMERGENCY",0):>5d} '
                  f'{rb_slip.get("TRAIL_TP",0):>5d} {rb_slip.get("TIMEOUT",0):>5d}')

    results['runs'] = runs

    # Regression check
    base3 = runs['baseline_esl3.0']
    reg_ok = abs(base3['slip']['PnL'] - 46.09) < 1.0
    print(f'\nRegression (baseline esl=3.0): slip PnL={base3["slip"]["PnL"]} '
          f'(expected ~46.09) → {"OK" if reg_ok else "FAIL"}')
    results['regression'] = {'ok': reg_ok}

    # Top by slip PnL
    sorted_ = sorted(runs.items(), key=lambda x: x[1]['slip']['PnL'],
                     reverse=True)
    print('\n--- Top 5 by slip PnL ---')
    for key, r in sorted_[:5]:
        print(f'  {key}: slip PnL={r["slip"]["PnL"]:+.2f} MDD={r["slip"]["MDD"]:.2f}')

    # 간단 verdict
    base_slip = base3['slip']['PnL']
    base_mdd = base3['slip']['MDD']

    improvements = []
    for key, r in runs.items():
        if 'baseline' not in key:
            continue
        pnl_delta = r['slip']['PnL'] - base_slip
        mdd_pct = (r['slip']['MDD'] - base_mdd) / base_mdd * 100 if base_mdd > 0 else 0
        improvements.append({
            'key': key, 'pnl_delta': round(pnl_delta, 2),
            'mdd_pct_change': round(mdd_pct, 1),
        })
    results['improvements_vs_base'] = improvements

    for imp in improvements:
        print(f'  {imp["key"]}: PnL Δ={imp["pnl_delta"]:+.2f}pp, '
              f'MDD change={imp["mdd_pct_change"]:+.1f}%')

    elapsed = (datetime.now() - t0).total_seconds()
    results['elapsed_sec'] = round(elapsed, 1)
    print(f'\nElapsed: {elapsed:.1f}s')

    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out = ROOT / 'results' / f'emergency_sl_reduction_{stamp}.json'
    out.write_text(json.dumps(results, indent=2, default=str))
    print(f'Results: {out}')


if __name__ == '__main__':
    main()
