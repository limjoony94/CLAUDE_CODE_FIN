#!/usr/bin/env python3
"""Progressive Trail Study — 진입 초기 vs TP 진행 후 trail_K 변화

가설: best_pnl(TP 진행률)이 THRESHOLD 이하일 땐 base_K(=2.5) 유지,
      THRESHOLD 초과 시 tight_K(<2.5)로 전환 → 수익 폭 보존.

Base strategy: cand_C_b0.60 + trend_filter(1.0/192)  (현재 최고 후보)
"""
import sys, copy, math, json
from pathlib import Path
from statistics import mean, stdev, median
import random

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
import os; os.chdir(ROOT)

import scripts.analysis.intrabar_trail_impact as ibt
from scripts.production.c1_breakout.signals import C1BreakoutSignal
from scripts.analysis.c1_intrabar_parity import SLIPPAGE as SLIP_MED, apply_slippage
from scripts.analysis.regime_filter_lowvol_study import run_bt_with_regime
from scripts.analysis.regime_filter_trend_study import precompute_trend_pass

DATA_DAYS = ibt.n15 / 96


def check_exit_bar_close_progressive(pos, bar, tk_base, tk_tight, threshold_pct):
    """Progressive trail: best_pnl < threshold → tk_base, else tk_tight.

    기존 intrabar_trail_impact._check_exit_bar_close와 동일하되 tk만 conditional.
    """
    d, ep, bp, sl = pos['d'], pos['ep'], pos['bp'], pos['sl']
    bh = pos['bh']
    c15 = ibt.c15; l15 = ibt.l15; h15 = ibt.h15; atr14 = ibt.atr14

    # 1. Fractal SL
    if d == 'LONG' and l15[bar] <= sl:
        return {'reason': 'SL', 'exit_price': sl}
    elif d == 'SHORT' and h15[bar] >= sl:
        return {'reason': 'SL', 'exit_price': sl}

    # 2. Emergency SL
    if d == 'LONG':
        worst_pnl = (l15[bar] / ep - 1) * 100
    else:
        worst_pnl = (1 - h15[bar] / ep) * 100
    if worst_pnl <= -ibt.emergency_sl:
        if d == 'LONG':
            return {'reason': 'EMERGENCY', 'exit_price': ep * (1 - ibt.emergency_sl / 100)}
        else:
            return {'reason': 'EMERGENCY', 'exit_price': ep * (1 + ibt.emergency_sl / 100)}

    # 3. Timeout
    if bh >= ibt.max_hold:
        return {'reason': 'TIMEOUT', 'exit_price': c15[bar]}

    # 4. Trail TP with progressive K
    if d == 'LONG':
        best_pnl = (bp / ep - 1) * 100
        cur_pnl = (c15[bar] / ep - 1) * 100
    else:
        best_pnl = (1 - bp / ep) * 100
        cur_pnl = (1 - c15[bar] / ep) * 100

    atr_val = atr14[bar]
    if best_pnl > ibt.trail_act and not math.isnan(atr_val) and atr_val > 0:
        # PROGRESSIVE: choose tk based on best_pnl threshold
        tk = tk_tight if best_pnl >= threshold_pct else tk_base
        trail_dist_pct = tk * atr_val / c15[bar] * 100
        drawdown = best_pnl - cur_pnl
        if drawdown >= trail_dist_pct:
            realized = max(0, best_pnl - trail_dist_pct)
            if d == 'LONG':
                return {'reason': 'TRAIL_TP', 'exit_price': ep * (1 + realized / 100)}
            else:
                return {'reason': 'TRAIL_TP', 'exit_price': ep * (1 - realized / 100)}

    return None


def run_bt_progressive(tk_base, tk_tight, threshold_pct, regime_passes,
                       slippage=SLIP_MED):
    """Monkey-patch ibt._check_exit_bar_close then call run_bt_with_regime."""
    orig_check = ibt._check_exit_bar_close

    def patched(pos, bar, tk):
        return check_exit_bar_close_progressive(pos, bar, tk_base, tk_tight, threshold_pct)

    ibt._check_exit_bar_close = patched
    try:
        trades = run_bt_with_regime(mode='bar_close', regime_passes=regime_passes,
                                     slippage=slippage)
    finally:
        ibt._check_exit_bar_close = orig_check
    return trades


def stats(trades):
    if not trades:
        return {'n': 0, 'pnl': 0, 'daily': 0, 'wr': 0, 'rr': 0, 'mdd': 0,
                'mean_net': 0, 'tpd': 0, 'boot': {}}
    total = sum(t['net'] for t in trades)
    daily = total / DATA_DAYS
    wins = [t for t in trades if t['net'] > 0]
    losses = [t for t in trades if t['net'] < 0]
    wr = len(wins) / len(trades) * 100
    avg_win = mean(t['net'] for t in wins) if wins else 0
    avg_loss = abs(mean(t['net'] for t in losses)) if losses else 0
    rr = avg_win / avg_loss if avg_loss > 0 else 0

    # MDD
    equity = 0; peak = 0; mdd = 0
    for t in trades:
        equity += t['net']
        peak = max(peak, equity)
        mdd = max(mdd, peak - equity)

    # Top-5% exclusion
    n_top = max(1, int(len(trades) * 0.05))
    sorted_desc = sorted(trades, key=lambda t: t['net'], reverse=True)
    ex_top = sorted_desc[n_top:]
    pnl_ex_top = sum(t['net'] for t in ex_top)

    # Bootstrap 3-day
    START_MIN = 220
    WINDOW = 288
    START_MAX = ibt.n15 - WINDOW - 1
    rng = random.Random(42)
    pnls = []
    for _ in range(1000):
        s = rng.randint(START_MIN, START_MAX)
        e = s + WINDOW
        pnls.append(sum(t['net'] for t in trades if s <= t['entry_bar'] < e))
    ps = sorted(pnls)
    boot = {
        'mean': mean(pnls),
        'pos_pct': sum(1 for p in pnls if p>0) / 1000 * 100,
        'sharpe': mean(pnls)/stdev(pnls) if stdev(pnls)>0 else 0,
        'p5': ps[50],
    }

    return {
        'n': len(trades), 'pnl': round(total,2), 'daily': round(daily,3),
        'wr': round(wr,1), 'rr': round(rr,2), 'mdd': round(mdd,2),
        'mean_net': round(total/len(trades),3), 'tpd': round(len(trades)/DATA_DAYS,2),
        'ex_top5': round(pnl_ex_top,2),
        'boot_pos': round(boot['pos_pct'],1),
        'boot_sharpe': round(boot['sharpe'],3),
        'boot_p5': round(boot['p5'],2),
        'boot_mean': round(boot['mean'],3),
    }


def main():
    # Setup base strategy: cand_C_b0.60 + trend_filter
    base_strat = {
        'max_sl_atr': 4.0, 'trail_K': 2.5, 'max_hold_bars': 192,
        'body_min_ratio': 0.60,
    }
    ibt.strat = copy.deepcopy(ibt.strat); ibt.strat.update(base_strat)
    ibt.trail_K = 2.5
    ibt.max_hold = 192
    ibt.sig = C1BreakoutSignal(ibt.strat)
    passes = precompute_trend_pass(192, 1.0)

    # Baseline: K=2.5 (base) fixed (threshold=inf effectively)
    base_trades = run_bt_progressive(2.5, 2.5, 99.0, passes)
    s = stats(base_trades)
    print('=' * 110)
    print('  Progressive Trail Study — cand_C_b0.60 + trend_filter base')
    print(f'  Data: {ibt.n15} bars = {DATA_DAYS:.1f} days, slippage=MED')
    print('=' * 110)
    print(f'\nBASELINE (K=2.5 fixed):')
    print(f'  n={s["n"]}, PnL={s["pnl"]:+.2f}, daily={s["daily"]:+.3f}, WR={s["wr"]}%, RR={s["rr"]}')
    print(f'  MDD={s["mdd"]}, mean_net={s["mean_net"]}, tpd={s["tpd"]}')
    print(f'  ex_top5={s["ex_top5"]:+.2f}, boot_pos={s["boot_pos"]}%, sharpe={s["boot_sharpe"]}, p5={s["boot_p5"]}')
    baseline = s

    # Grid: threshold × tk_post (tighten AND loosen)
    # tk_post < 2.5 → TIGHTEN (lock profits)
    # tk_post > 2.5 → LOOSEN (let profits run)
    thresholds = [0.3, 0.5, 0.8, 1.0, 1.5, 2.0]
    tk_tights = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0]  # 2.5 = baseline, <2.5 tighten, >2.5 loosen

    print(f'\nGrid: threshold × tk_post (tk_base=2.5 fixed)')
    print(f'  tk_post < 2.5 = TIGHTEN, > 2.5 = LOOSEN, = 2.5 = no change')
    print(f'{"thr":>5} {"tkP":>5} {"n":>5} {"PnL":>8} {"daily":>7} {"WR":>5} {"RR":>5} {"MDD":>6} {"mean":>7} {"tpd":>5} {"ex5":>8} {"pos":>6} {"shr":>7} {"p5":>7}  vs_base_pnl')

    results = []
    for thr in thresholds:
        for tkT in tk_tights:
            trades = run_bt_progressive(2.5, tkT, thr, passes)
            s = stats(trades)
            vs = s['pnl'] - baseline['pnl']
            print(f'{thr:>5.2f} {tkT:>5.1f} {s["n"]:>5} {s["pnl"]:>+7.2f} {s["daily"]:>+6.3f} '
                  f'{s["wr"]:>5.1f} {s["rr"]:>5.2f} {s["mdd"]:>6.2f} {s["mean_net"]:>+6.2f} '
                  f'{s["tpd"]:>5.2f} {s["ex_top5"]:>+7.2f} {s["boot_pos"]:>5.1f}% {s["boot_sharpe"]:>+6.3f} '
                  f'{s["boot_p5"]:>+6.2f}  {vs:>+7.2f}')
            results.append({'thr': thr, 'tkT': tkT, **s, 'vs_base': vs})

    # Sort & Rank
    print(f'\n=== Ranking by PnL (top 10) ===')
    results.sort(key=lambda r: r['pnl'], reverse=True)
    for r in results[:10]:
        print(f'  thr={r["thr"]}, tkT={r["tkT"]}: PnL={r["pnl"]:+.2f} '
              f'(vs base {r["vs_base"]:+.2f}), MDD={r["mdd"]}, ex5={r["ex_top5"]:+.2f}, '
              f'pos={r["boot_pos"]}%, sharpe={r["boot_sharpe"]}')

    print(f'\n=== Ranking by F6 (ex_top5) ===')
    results_f6 = sorted(results, key=lambda r: r['ex_top5'], reverse=True)
    for r in results_f6[:10]:
        print(f'  thr={r["thr"]}, tkT={r["tkT"]}: ex5={r["ex_top5"]:+.2f}, '
              f'PnL={r["pnl"]:+.2f}, pos={r["boot_pos"]}%, sharpe={r["boot_sharpe"]}')

    print(f'\n=== Ranking by Sharpe ===')
    results_sh = sorted(results, key=lambda r: r['boot_sharpe'], reverse=True)
    for r in results_sh[:10]:
        print(f'  thr={r["thr"]}, tkT={r["tkT"]}: sharpe={r["boot_sharpe"]}, '
              f'PnL={r["pnl"]:+.2f}, ex5={r["ex_top5"]:+.2f}, pos={r["boot_pos"]}%')


if __name__ == '__main__':
    main()
