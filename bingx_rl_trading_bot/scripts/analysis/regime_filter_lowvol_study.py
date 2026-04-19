#!/usr/bin/env python3
"""Low-vol regime filter study — fold 2 근본 해결 도전.

Rolling ATR% < THR 시 진입 skip. candidate_C_b0.60 base.

Grid: 5 THR × 3 LOOKBACK = 15 runs.
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

CANDIDATE = {'max_sl_atr': 4.0, 'trail_K': 2.5, 'max_hold_bars': 192,
             'body_min_ratio': 0.60}

THR_VALUES = [0.22, 0.24, 0.26, 0.28, 0.30]
LOOKBACK_VALUES = [96, 192, 288]

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


def precompute_regime_pass(lookback, thr_pct):
    """bar i에서 regime 통과 가능 여부 (과거 lookback bars ATR% 평균)."""
    n = ibt.n15
    passes = [True] * n  # default pass (warmup)
    for i in range(lookback, n):
        # exclusive past window
        atr_window = [x for x in ibt.atr14[i - lookback:i]
                      if not math.isnan(x)]
        if not atr_window:
            passes[i] = True
            continue
        close_mean = mean(ibt.c15[i - lookback:i])
        if close_mean <= 0:
            passes[i] = True
            continue
        atr_pct = mean(atr_window) / close_mean * 100
        passes[i] = atr_pct >= thr_pct
    return passes


def run_bt_with_regime(mode, regime_passes, slippage=None):
    """Fork of ibt.run_backtest + regime gate at entry."""
    WARMUP = 26
    FEE = ibt.FEE
    trades = []
    pos = None
    last_exit_bar = -3
    n15 = ibt.n15

    for bar in range(WARMUP, n15 - 1):
        # Exit check
        if pos is not None:
            if mode != '5m':
                if pos['d'] == 'LONG':
                    pos['bp'] = max(pos['bp'], ibt.h15[bar])
                else:
                    pos['bp'] = min(pos['bp'], ibt.l15[bar])
            pos['bh'] += 1

            if mode == '5m':
                exit_r = ibt._check_exit_5m(pos, bar, ibt.trail_K)
            elif mode == 'intrabar':
                exit_r = ibt._check_exit_intrabar(pos, bar, ibt.trail_K)
            else:
                exit_r = ibt._check_exit_bar_close(pos, bar, ibt.trail_K)

            if exit_r:
                xp = exit_r['exit_price']
                if pos['d'] == 'LONG':
                    raw = (xp / pos['ep'] - 1) * 100
                else:
                    raw = (1 - xp / pos['ep']) * 100
                trades.append({
                    'raw': raw, 'net': raw - FEE,
                    'reason': exit_r['reason'], 'bh': pos['bh'], 'd': pos['d'],
                    'entry_bar': pos['entry_bar'], 'exit_bar': bar,
                    'entry_price': pos['ep'], 'exit_price': xp,
                })
                pos = None
                last_exit_bar = bar
                continue

        # Entry check
        if pos is None and bar - last_exit_bar >= ibt.min_bars_between:
            # NEW: regime gate
            if not regime_passes[bar]:
                continue

            e = ibt.sig.check_entry(
                ibt.o15[bar], ibt.h15[bar], ibt.l15[bar], ibt.c15[bar],
                ibt.ch_h[bar] if hasattr(ibt, 'ch_h') else None,
                ibt.ch_l[bar] if hasattr(ibt, 'ch_l') else None,
                ibt.atr14[bar],
                ibt.sw_l[bar] if hasattr(ibt, 'sw_l') else None,
                ibt.sw_h[bar] if hasattr(ibt, 'sw_h') else None,
            )
            # Fall back to ibt module variables
            if e is None:
                e = ibt.sig.check_entry(
                    ibt.o15[bar], ibt.h15[bar], ibt.l15[bar], ibt.c15[bar],
                    ibt.ch_h[bar], ibt.ch_l[bar], ibt.atr14[bar],
                    ibt.sw_l[bar], ibt.sw_h[bar],
                )
            if e and bar + 1 < n15:
                pos = {
                    'd': e['direction'], 'ep': ibt.o15[bar + 1],
                    'sl': e['sl_price'], 'bp': ibt.o15[bar + 1],
                    'bh': 0, 'entry_bar': bar + 1,
                }

    # Apply slippage if needed
    if slippage is not None:
        adjusted = []
        for t in trades:
            entry_adv = slippage['entry_pct'] / 100
            if t['d'] == 'LONG':
                eff_entry = t['entry_price'] * (1 + entry_adv)
                raw_new = (t['exit_price'] / eff_entry - 1) * 100
            else:
                eff_entry = t['entry_price'] * (1 - entry_adv)
                raw_new = (1 - t['exit_price'] / eff_entry) * 100
            t_adj = dict(t)
            t_adj['raw'] = raw_new
            t_adj['net'] = apply_slippage(t_adj, slippage, ibt.emergency_sl)
            adjusted.append(t_adj)
        return adjusted
    return trades


def summarize(trades):
    if not trades:
        return {'count': 0, 'PnL': 0.0, 'MDD': 0.0, 'WR': 0.0}
    pnl = sum(t['net'] for t in trades)
    mdd = compute_mdd_additive(trades)
    wr = sum(1 for t in trades if t['net'] > 0) / len(trades) * 100
    return {'count': len(trades), 'PnL': round(pnl, 2),
            'MDD': round(mdd, 2), 'WR': round(wr, 1)}


def fold_breakdown(trades):
    out = {}
    for fname, (lo, hi) in FOLD_BOUNDARIES.items():
        ft = [t for t in trades if lo <= t['entry_bar'] < hi]
        out[fname] = summarize(ft)
    return out


def wf_5fold(trades):
    if not trades:
        return 0, []
    ts = sorted(trades, key=lambda t: t['entry_bar'])
    first, last = ts[0]['entry_bar'], ts[-1]['entry_bar']
    span = last - first
    if span < 5:
        return 0, []
    fs = span // 5
    pnls = []
    pos = 0
    for k in range(5):
        lo = first + k * fs
        hi = first + (k + 1) * fs if k < 4 else last + 1
        p = sum(t['net'] for t in ts if lo <= t['entry_bar'] < hi)
        pnls.append(round(p, 2))
        if p > 0:
            pos += 1
    return pos, pnls


def main():
    t0 = datetime.now()
    print('=' * 70)
    print('  Regime Filter (Low-Vol) — fold 2 근본 해결 도전')
    print(f'  Base: candidate_C_b0.60 = {CANDIDATE}')
    print('=' * 70)

    results = {'timestamp': datetime.now(timezone.utc).isoformat(),
               'candidate': CANDIDATE,
               'thr_values': THR_VALUES, 'lookback_values': LOOKBACK_VALUES}

    # Set combo once, indicators/signal ready
    set_combo_body(CANDIDATE)

    # Baseline (no regime filter)
    print('\n--- Baseline (no regime filter, candidate_C_b0.60) ---')
    all_pass = [True] * ibt.n15
    trades_clean_base = run_bt_with_regime(mode='bar_close',
                                            regime_passes=all_pass)
    trades_slip_base = run_bt_with_regime(mode='5m',
                                           regime_passes=all_pass,
                                           slippage=SLIP_MED)
    base_clean = summarize(trades_clean_base)
    base_slip = summarize(trades_slip_base)
    base_fold_slip = fold_breakdown(trades_slip_base)
    base_wf_slip = wf_5fold(trades_slip_base)
    print(f'  Clean: PnL={base_clean["PnL"]:+.2f} MDD={base_clean["MDD"]:.2f} N={base_clean["count"]}')
    print(f'  Slip : PnL={base_slip["PnL"]:+.2f} MDD={base_slip["MDD"]:.2f} N={base_slip["count"]}')
    print(f'  WF slip: {base_wf_slip[0]}/5 {base_wf_slip[1]}')
    print(f'  Fold 2 slip: {base_fold_slip["fold_2"]["PnL"]:+.2f}')

    results['base'] = {
        'clean': base_clean, 'slip': base_slip,
        'wf_slip': base_wf_slip, 'fold_slip': base_fold_slip,
    }

    # Grid
    print('\n--- Grid (THR × LOOKBACK) ---')
    print(f'{"THR":>5s} {"LB":>5s} {"slip PnL":>10s} {"MDD":>8s} {"N":>6s} '
          f'{"WF":>5s} {"fold1":>8s} {"fold2":>8s} {"fold3":>8s} '
          f'{"fold4":>8s} {"fold5":>8s}')

    grid_results = {}
    for thr in THR_VALUES:
        for lb in LOOKBACK_VALUES:
            passes = precompute_regime_pass(lb, thr)
            skipped_count = sum(1 for p in passes if not p)

            trades_slip = run_bt_with_regime(mode='5m', regime_passes=passes,
                                              slippage=SLIP_MED)
            s = summarize(trades_slip)
            fold = fold_breakdown(trades_slip)
            wf = wf_5fold(trades_slip)

            key = f'thr{thr:.2f}_lb{lb}'
            grid_results[key] = {
                'thr': thr, 'lookback': lb,
                'skipped_bars': skipped_count,
                'slip': s, 'fold': fold, 'wf_slip': wf,
            }

            print(f'{thr:>5.2f} {lb:>5d} {s["PnL"]:>+9.2f}% {s["MDD"]:>7.2f} '
                  f'{s["count"]:>6d} {wf[0]}/{5} '
                  f'{fold["fold_1"]["PnL"]:>+7.2f} {fold["fold_2"]["PnL"]:>+7.2f} '
                  f'{fold["fold_3"]["PnL"]:>+7.2f} {fold["fold_4"]["PnL"]:>+7.2f} '
                  f'{fold["fold_5"]["PnL"]:>+7.2f}')

    results['grid'] = grid_results

    reset_combo_body()

    # Best by: fold 2 양수 + full PnL 유지
    print('\n--- Best combos (fold 2 양수 우선) ---')
    f2_positive = [(k, r) for k, r in grid_results.items()
                   if r['fold']['fold_2']['PnL'] > 0]
    if f2_positive:
        print(f'{len(f2_positive)} combos have fold_2 > 0:')
        f2_positive.sort(key=lambda x: x[1]['slip']['PnL'], reverse=True)
        for k, r in f2_positive[:5]:
            print(f'  {k}: slip PnL={r["slip"]["PnL"]:+.2f} '
                  f'fold2={r["fold"]["fold_2"]["PnL"]:+.2f} '
                  f'WF={r["wf_slip"][0]}/5')
        top = f2_positive[0]
    else:
        print('No combo with fold_2 > 0.')
        # fallback: fold 2 least negative
        sorted_f2 = sorted(grid_results.items(),
                            key=lambda x: x[1]['fold']['fold_2']['PnL'],
                            reverse=True)
        for k, r in sorted_f2[:3]:
            print(f'  {k}: slip PnL={r["slip"]["PnL"]:+.2f} '
                  f'fold2={r["fold"]["fold_2"]["PnL"]:+.2f} '
                  f'WF={r["wf_slip"][0]}/5')
        top = sorted_f2[0]

    # 3-flag GO evaluation
    top_r = top[1]
    flags = {
        'fold2_slip_positive': top_r['fold']['fold_2']['PnL'] > 0,
        'overall_not_degraded': top_r['slip']['PnL'] >= base_slip['PnL'] - 5.0,
        'wf_slip_5of5': top_r['wf_slip'][0] == 5,
    }
    print(f'\n--- 3-flag GO (top: {top[0]}) ---')
    for k, v in flags.items():
        print(f'  {k}: {v}')
    passed = sum(1 for v in flags.values() if v)
    verdict = 'GO' if passed == 3 else 'STOP'
    print(f'\n=== VERDICT: {verdict} ({passed}/3) ===')
    results['top_candidate'] = top[0]
    results['go_flags'] = flags
    results['verdict'] = {'outcome': verdict, 'passed': passed}

    elapsed = (datetime.now() - t0).total_seconds()
    results['elapsed_sec'] = round(elapsed, 1)
    print(f'\nElapsed: {elapsed:.1f}s')

    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out = ROOT / 'results' / f'regime_filter_lowvol_{stamp}.json'
    out.write_text(json.dumps(results, indent=2, default=str))
    print(f'Results: {out}')


if __name__ == '__main__':
    main()
