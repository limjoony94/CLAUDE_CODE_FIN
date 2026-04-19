#!/usr/bin/env python3
"""Fold 2 regime analysis — candidate_C weakness diagnostic.

Investigates 2025-07-11 ~ 2025-09-15 period where candidate_C (4.0, 2.5, 192)
shows -9.03pp slip-WF fold fail while clean WF is +15.95pp.

Tests hypotheses H1-H7 from docs/01-plan/features/fold2_regime_analysis.plan.md:
  H1: Low volatility regime
  H2: Low breakout frequency
  H3: High SL exit ratio (whipsaw)
  H4: Poor R:R (avg_loss > avg_win)
  H5: Baseline vs candidate divergence (widening SL amplifies damage)
  H6: Concentrated sub-window loss
  H7: Regime classifier viable (threshold sweep)

Output: results/fold2_regime_analysis_{timestamp}.json
"""
import sys
import os
import json
import math
from pathlib import Path
from datetime import datetime, timezone
from statistics import mean, stdev, median

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import pandas as pd

import scripts.analysis.intrabar_trail_impact as ibt
import scripts.analysis.c1_intrabar_parity as cip
from scripts.analysis.c1_intrabar_parity import (
    run_bt_with_slippage, set_combo, reset_combo, SLIPPAGE as SLIP_MED,
)

COMBOS = {
    'baseline':    {'max_sl_atr': 3.3, 'trail_K': 2.5, 'max_hold_bars': 192},
    'candidate_C': {'max_sl_atr': 4.0, 'trail_K': 2.5, 'max_hold_bars': 192},
}

# Fold boundaries from candidate_c_validation run
FOLD_BOUNDARIES = {
    'fold_1': {'lo': 31,    'hi': 6407,  'label': '2025-05-05~2025-07-11'},
    'fold_2': {'lo': 6407,  'hi': 12783, 'label': '2025-07-11~2025-09-15'},
    'fold_3': {'lo': 12783, 'hi': 19159, 'label': '2025-09-15~2025-11-21'},
    'fold_4': {'lo': 19159, 'hi': 25535, 'label': '2025-11-21~2026-01-26'},
    'fold_5': {'lo': 25535, 'hi': 31916, 'label': '2026-01-26~2026-04-03'},
}


def compute_regime_metrics(fold_lo, fold_hi):
    h = ibt.h15[fold_lo:fold_hi]
    l = ibt.l15[fold_lo:fold_hi]
    c = ibt.c15[fold_lo:fold_hi]
    atr_slice = [x for x in ibt.atr14[fold_lo:fold_hi]
                 if not math.isnan(x)]

    atr_avg = mean(atr_slice) if atr_slice else 0
    close_mean = mean(c)
    atr_pct_avg = atr_avg / close_mean * 100 if close_mean > 0 else 0

    returns = [(c[i] / c[i-1] - 1) for i in range(1, len(c)) if c[i-1] > 0]
    ret_std = stdev(returns) * 100 if len(returns) > 1 else 0

    range_pcts = [(h[i] - l[i]) / c[i] * 100 for i in range(len(c)) if c[i] > 0]
    range_pct_avg = mean(range_pcts) if range_pcts else 0

    half = len(c) // 2
    trend_pct = (c[-1] / c[half] - 1) * 100 if half > 0 and c[half] > 0 else 0
    full_trend_pct = (c[-1] / c[0] - 1) * 100 if c[0] > 0 else 0

    sideways_idx = (max(h) - min(l)) / atr_avg if atr_avg > 0 else 0

    return {
        'atr_avg':              round(atr_avg, 2),
        'atr_pct_avg':          round(atr_pct_avg, 3),
        'returns_std_pct':      round(ret_std, 4),
        'range_pct_avg':        round(range_pct_avg, 3),
        'trend_pct_half':       round(trend_pct, 2),
        'trend_pct_full':       round(full_trend_pct, 2),
        'sideways_index':       round(sideways_idx, 2),
        'price_first':          round(c[0], 1),
        'price_last':           round(c[-1], 1),
        'price_max':            round(max(h), 1),
        'price_min':            round(min(l), 1),
        'bars':                 fold_hi - fold_lo,
    }


def compute_strategy_metrics(fold_trades, fold_lo, fold_hi):
    if not fold_trades:
        return {'count': 0}

    wins = [t for t in fold_trades if t['net'] > 0]
    losses = [t for t in fold_trades if t['net'] <= 0]

    reasons = {}
    for t in fold_trades:
        r = t.get('reason_effective', t.get('reason', 'UNK'))
        reasons[r] = reasons.get(r, 0) + 1
    reason_pct = {k: round(v / len(fold_trades) * 100, 1)
                  for k, v in reasons.items()}

    streak = max_streak = 0
    for t in fold_trades:
        if t['net'] <= 0:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0

    days = (fold_hi - fold_lo) / 96
    avg_win = sum(t['net'] for t in wins) / len(wins) if wins else 0
    avg_loss = sum(t['net'] for t in losses) / len(losses) if losses else 0
    rr = abs(avg_win / avg_loss) if avg_loss != 0 else 0
    bh_list = sorted(t.get('bh', 0) for t in fold_trades)

    return {
        'count':            len(fold_trades),
        'trades_per_day':   round(len(fold_trades) / days, 2) if days > 0 else 0,
        'wr_pct':           round(len(wins) / len(fold_trades) * 100, 1),
        'pnl_sum':          round(sum(t['net'] for t in fold_trades), 2),
        'avg_win':          round(avg_win, 3),
        'avg_loss':         round(avg_loss, 3),
        'rr':               round(rr, 2),
        'exit_reason_pct':  reason_pct,
        'max_consec_loss':  max_streak,
        'median_bars_held': bh_list[len(bh_list) // 2],
    }


def run_combo_once(combo_name, combo_cfg):
    """Execute once, return full-period trades + per-fold partitions."""
    set_combo(**combo_cfg)
    all_trades = run_bt_with_slippage(mode='5m', slippage=SLIP_MED)
    reset_combo()

    by_fold = {}
    for fname, fb in FOLD_BOUNDARIES.items():
        ft = [t for t in all_trades if fb['lo'] <= t['entry_bar'] < fb['hi']]
        by_fold[fname] = ft
    return all_trades, by_fold


def sub_window_microscopy(fold_trades, fold_lo, fold_hi, window_days=5):
    bars_per_window = window_days * 96
    stride = bars_per_window // 2
    ts_col = ibt.agg15['ts']

    windows = []
    cur = fold_lo
    while cur + bars_per_window <= fold_hi:
        wt = [t for t in fold_trades
              if cur <= t['entry_bar'] < cur + bars_per_window]
        pnl = sum(t['net'] for t in wt)
        wr = (sum(1 for t in wt if t['net'] > 0) / len(wt) * 100
              if wt else 0)
        windows.append({
            'start':  str(ts_col.iloc[cur])[:10],
            'end':    str(ts_col.iloc[min(cur + bars_per_window - 1,
                                          len(ts_col) - 1)])[:10],
            'trades': len(wt),
            'pnl':    round(pnl, 2),
            'wr':     round(wr, 1),
        })
        cur += stride

    sorted_w = sorted(windows, key=lambda w: w['pnl'])
    return {
        'all_windows': windows,
        'worst_3':     sorted_w[:3],
        'best_3':      sorted_w[-3:],
    }


def evaluate_regime_filter(regime_by_fold):
    candidates = [
        {'name': 'low_atr_pct', 'metric': 'atr_pct_avg', 'op': '<',
         'values': [0.3, 0.4, 0.5, 0.55, 0.6, 0.7]},
        {'name': 'low_returns_std', 'metric': 'returns_std_pct', 'op': '<',
         'values': [0.05, 0.08, 0.10, 0.12, 0.15, 0.20]},
        {'name': 'high_sideways', 'metric': 'sideways_index', 'op': '>',
         'values': [30, 40, 50, 60, 80, 100]},
        {'name': 'low_range_pct', 'metric': 'range_pct_avg', 'op': '<',
         'values': [0.2, 0.3, 0.4, 0.5, 0.6]},
    ]
    out = []
    for c in candidates:
        for th in c['values']:
            flagged = []
            vals = {}
            for fn, fm in regime_by_fold.items():
                v = fm[c['metric']]
                vals[fn] = v
                trig = (v < th) if c['op'] == '<' else (v > th)
                if trig:
                    flagged.append(fn)
            out.append({
                'rule':            f"{c['metric']} {c['op']} {th}",
                'flagged_folds':   flagged,
                'flags_fold_2':    'fold_2' in flagged,
                'also_flags':      [f for f in flagged if f != 'fold_2'],
                'fold_values':     vals,
            })
    return out


def summarize_hypotheses(regime, strat_by_combo, sub_win, reg_filters):
    r2 = regime['fold_2']
    r_others = [regime[f'fold_{i}'] for i in (1, 3, 4, 5)]
    s2c = strat_by_combo['candidate_C']['fold_2']
    s2b = strat_by_combo['baseline']['fold_2']
    others_c = [strat_by_combo['candidate_C'][f'fold_{i}'] for i in (1, 3, 4, 5)]

    def avg_others(key):
        vals = [o.get(key, 0) for o in others_c if isinstance(o.get(key), (int, float))]
        return sum(vals) / len(vals) if vals else 0

    def avg_reg(key):
        return sum(r[key] for r in r_others) / len(r_others)

    h = {}

    # H1 low vol
    h1_f2 = r2['atr_pct_avg']
    h1_others = avg_reg('atr_pct_avg')
    h['H1_low_vol'] = {
        'fold_2': h1_f2, 'others_avg': round(h1_others, 3),
        'ratio': round(h1_f2 / h1_others, 2) if h1_others > 0 else 0,
        'verdict': h1_f2 < h1_others * 0.9,
    }

    # H2 low breakout freq
    h2_f2 = s2c['trades_per_day']
    h2_others = avg_others('trades_per_day')
    h['H2_low_breakout'] = {
        'fold_2': h2_f2, 'others_avg': round(h2_others, 2),
        'ratio': round(h2_f2 / h2_others, 2) if h2_others > 0 else 0,
        'verdict': h2_f2 < h2_others * 0.9,
    }

    # H3 high SL exit
    sl_f2 = s2c.get('exit_reason_pct', {}).get('SL', 0)
    sl_others = [strat_by_combo['candidate_C'][f'fold_{i}']
                 .get('exit_reason_pct', {}).get('SL', 0)
                 for i in (1, 3, 4, 5)]
    sl_avg = sum(sl_others) / len(sl_others)
    h['H3_high_sl_exit'] = {
        'fold_2': sl_f2, 'others_avg': round(sl_avg, 1),
        'verdict': sl_f2 > sl_avg * 1.1,
    }

    # H4 poor R:R
    h4_f2 = s2c['rr']
    h4_others = avg_others('rr')
    h['H4_poor_rr'] = {
        'fold_2': h4_f2, 'others_avg': round(h4_others, 2),
        'verdict': h4_f2 < h4_others * 0.8,
    }

    # H5 baseline better in fold 2
    h['H5_widening_sl_amplifies'] = {
        'baseline_pnl': s2b['pnl_sum'],
        'candidate_pnl': s2c['pnl_sum'],
        'diff_cand_minus_base': round(s2c['pnl_sum'] - s2b['pnl_sum'], 2),
        'verdict_baseline_better': s2b['pnl_sum'] > s2c['pnl_sum'],
    }

    # H6 concentrated sub-window
    worst3_sum = sum(w['pnl'] for w in sub_win['worst_3'])
    h['H6_concentrated_loss'] = {
        'worst_3_pnl_sum': round(worst3_sum, 2),
        'fold_2_total': s2c['pnl_sum'],
        'worst_1_window': sub_win['worst_3'][0] if sub_win['worst_3'] else None,
        'verdict_concentrated':
            abs(worst3_sum) >= abs(s2c['pnl_sum']) * 0.7
            if s2c['pnl_sum'] != 0 else False,
    }

    # H7 regime filter viable
    clean = [f for f in reg_filters
             if f['flags_fold_2'] and len(f['also_flags']) <= 1]
    h['H7_regime_filter_viable'] = {
        'clean_filter_count': len(clean),
        'top_3_candidates':   clean[:3],
        'verdict':            len(clean) > 0,
    }

    return h


def main():
    t0 = datetime.now()
    print('=' * 70)
    print('  Fold 2 Regime Analysis — candidate_C weakness diagnostic')
    print('=' * 70)

    # 1. Regime metrics per fold
    print('\n--- Regime Metrics Per Fold ---')
    regime = {}
    for fname, fb in FOLD_BOUNDARIES.items():
        regime[fname] = compute_regime_metrics(fb['lo'], fb['hi'])
        r = regime[fname]
        print(f'{fname} ({fb["label"]})')
        print(f'  ATR%={r["atr_pct_avg"]:.3f} | ret_std={r["returns_std_pct"]:.4f} '
              f'| range%={r["range_pct_avg"]:.3f} | '
              f'trend_full={r["trend_pct_full"]:+.1f}% | sideways={r["sideways_index"]:.1f}')

    # 2. Strategy metrics per fold per combo
    print('\n--- Strategy Metrics Per Fold/Combo ---')
    strat = {}
    for cname, ccfg in COMBOS.items():
        _, by_fold = run_combo_once(cname, ccfg)
        strat[cname] = {}
        print(f'\n{cname}:')
        for fname, trades in by_fold.items():
            fb = FOLD_BOUNDARIES[fname]
            strat[cname][fname] = compute_strategy_metrics(trades, fb['lo'], fb['hi'])
            s = strat[cname][fname]
            if s['count'] > 0:
                print(f'  {fname}: N={s["count"]:3d} tpd={s["trades_per_day"]:4.2f} '
                      f'WR={s["wr_pct"]:5.1f}% PnL={s["pnl_sum"]:+7.2f}% '
                      f'R:R={s["rr"]:4.2f} SL%={s["exit_reason_pct"].get("SL",0):4.1f} '
                      f'streak={s["max_consec_loss"]:2d}')

    # 3. Sub-window microscopy (fold 2 candidate_C)
    print('\n--- Sub-window Microscopy (Fold 2, candidate_C, 5-day stride) ---')
    set_combo(**COMBOS['candidate_C'])
    cand_trades_full = run_bt_with_slippage(mode='5m', slippage=SLIP_MED)
    reset_combo()
    fb2 = FOLD_BOUNDARIES['fold_2']
    fold2_trades = [t for t in cand_trades_full
                    if fb2['lo'] <= t['entry_bar'] < fb2['hi']]
    sub_win = sub_window_microscopy(fold2_trades, fb2['lo'], fb2['hi'])
    print('Worst 3 windows:')
    for w in sub_win['worst_3']:
        print(f'  {w["start"]} ~ {w["end"]}: N={w["trades"]:3d} PnL={w["pnl"]:+6.2f}% WR={w["wr"]:5.1f}%')
    print('Best 3 windows:')
    for w in sub_win['best_3']:
        print(f'  {w["start"]} ~ {w["end"]}: N={w["trades"]:3d} PnL={w["pnl"]:+6.2f}% WR={w["wr"]:5.1f}%')

    # 4. Regime filter threshold sweep
    print('\n--- Regime Filter Calibration ---')
    reg_filters = evaluate_regime_filter(regime)
    clean_filters = [f for f in reg_filters
                     if f['flags_fold_2'] and len(f['also_flags']) <= 1]
    print(f'Clean filters (flag fold_2, ≤1 other): {len(clean_filters)}')
    for cf in clean_filters[:5]:
        print(f'  {cf["rule"]:40s} flags={cf["flagged_folds"]}')

    # 5. Hypothesis summary
    print('\n--- Hypothesis Evaluation (H1-H7) ---')
    hyp = summarize_hypotheses(regime, strat, sub_win, reg_filters)
    for k, v in hyp.items():
        verdict = v.get('verdict', v.get('verdict_baseline_better',
                                         v.get('verdict_concentrated')))
        print(f'  {k}: verdict={verdict}')
        details = {kk: vv for kk, vv in v.items()
                   if kk != 'verdict' and not isinstance(vv, list)}
        print(f'    {details}')

    # 6. Save
    out = {
        'timestamp': datetime.now(timezone.utc).isoformat(),
        'combos': COMBOS,
        'fold_boundaries': {k: {'bars': (v['lo'], v['hi']), 'label': v['label']}
                            for k, v in FOLD_BOUNDARIES.items()},
        'slippage_used': SLIP_MED,
        'regime_by_fold': regime,
        'strategy_by_fold_combo': strat,
        'fold_2_sub_windows': sub_win,
        'regime_filter_candidates': reg_filters,
        'hypothesis_summary': hyp,
    }
    elapsed = (datetime.now() - t0).total_seconds()
    out['elapsed_sec'] = round(elapsed, 1)
    print(f'\nElapsed: {elapsed:.1f}s')

    stamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    path = ROOT / 'results' / f'fold2_regime_analysis_{stamp}.json'
    path.write_text(json.dumps(out, indent=2, default=str))
    print(f'Results: {path}')


if __name__ == '__main__':
    main()
