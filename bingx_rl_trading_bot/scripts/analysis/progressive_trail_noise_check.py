#!/usr/bin/env python3
"""Progressive Trail NOISE sensitivity check.

bar_close vs intrabar (worst-case) vs 5m_resolution 비교.
tkT 0.1/0.3/0.5/1.0/2.5 별로 노이즈 충격 측정.

BT bar_close는 봉 종가에만 trail 체크 → 노이즈 가려짐.
LIVE는 intrabar 즉시 STOP_MARKET 트리거 → tkT 작을수록 노이즈에 취약.
"""
import sys, copy, math, random
from pathlib import Path
from statistics import mean, stdev

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
import os; os.chdir(ROOT)

import scripts.analysis.intrabar_trail_impact as ibt
from scripts.production.c1_breakout.signals import C1BreakoutSignal
from scripts.analysis.c1_intrabar_parity import SLIPPAGE as SLIP_MED, apply_slippage
from scripts.analysis.regime_filter_lowvol_study import run_bt_with_regime
from scripts.analysis.regime_filter_trend_study import precompute_trend_pass

DATA_DAYS = ibt.n15 / 96


# ─── Progressive exit factories for bar_close AND intrabar modes ───
def make_exit_bar_close(tk_base, tk_post, thr):
    def check(pos, bar, tk):
        d, ep, bp, sl = pos['d'], pos['ep'], pos['bp'], pos['sl']
        c15, l15, h15, atr = ibt.c15, ibt.l15, ibt.h15, ibt.atr14
        if d == 'LONG' and l15[bar] <= sl: return {'reason':'SL','exit_price':sl}
        elif d == 'SHORT' and h15[bar] >= sl: return {'reason':'SL','exit_price':sl}
        worst = (l15[bar]/ep-1)*100 if d=='LONG' else (1-h15[bar]/ep)*100
        if worst <= -ibt.emergency_sl:
            px = ep*(1-ibt.emergency_sl/100) if d=='LONG' else ep*(1+ibt.emergency_sl/100)
            return {'reason':'EMERGENCY','exit_price':px}
        if pos['bh'] >= ibt.max_hold: return {'reason':'TIMEOUT','exit_price':c15[bar]}
        bpl = (bp/ep-1)*100 if d=='LONG' else (1-bp/ep)*100
        cpl = (c15[bar]/ep-1)*100 if d=='LONG' else (1-c15[bar]/ep)*100
        a = atr[bar]
        if bpl > ibt.trail_act and not math.isnan(a) and a>0:
            k = tk_post if bpl >= thr else tk_base
            td = k*a/c15[bar]*100
            if bpl - cpl >= td:
                r = max(0, bpl - td)
                px = ep*(1+r/100) if d=='LONG' else ep*(1-r/100)
                return {'reason':'TRAIL_TP','exit_price':px}
        return None
    return check


def make_exit_intrabar(tk_base, tk_post, thr):
    """Intrabar: bar LOW (LONG) / HIGH (SHORT)로 worst-case trail 체크."""
    def check(pos, bar, tk):
        d, ep, bp, sl = pos['d'], pos['ep'], pos['bp'], pos['sl']
        c15, l15, h15, atr = ibt.c15, ibt.l15, ibt.h15, ibt.atr14
        if d == 'LONG' and l15[bar] <= sl: return {'reason':'SL','exit_price':sl}
        elif d == 'SHORT' and h15[bar] >= sl: return {'reason':'SL','exit_price':sl}
        worst = (l15[bar]/ep-1)*100 if d=='LONG' else (1-h15[bar]/ep)*100
        if worst <= -ibt.emergency_sl:
            px = ep*(1-ibt.emergency_sl/100) if d=='LONG' else ep*(1+ibt.emergency_sl/100)
            return {'reason':'EMERGENCY','exit_price':px}
        if pos['bh'] >= ibt.max_hold: return {'reason':'TIMEOUT','exit_price':c15[bar]}
        bpl = (bp/ep-1)*100 if d=='LONG' else (1-bp/ep)*100
        # INTRABAR: worst price for trail check
        if d == 'LONG':
            worst_pl = (l15[bar]/ep-1)*100
        else:
            worst_pl = (1-h15[bar]/ep)*100
        a = atr[bar]
        if bpl > ibt.trail_act and not math.isnan(a) and a>0:
            k = tk_post if bpl >= thr else tk_base
            td = k*a/c15[bar]*100
            # 드로우다운 계산: best_pnl - worst_pnl_in_bar
            dd_worst = bpl - worst_pl
            if dd_worst >= td:
                # Exit at trail boundary price (not worst)
                r = max(0, bpl - td)
                px = ep*(1+r/100) if d=='LONG' else ep*(1-r/100)
                return {'reason':'TRAIL_TP','exit_price':px}
        return None
    return check


def run_bt(check_fn, passes):
    orig = ibt._check_exit_bar_close
    ibt._check_exit_bar_close = check_fn
    try:
        return run_bt_with_regime(mode='bar_close', regime_passes=passes, slippage=SLIP_MED)
    finally:
        ibt._check_exit_bar_close = orig


def stats(trades):
    if not trades: return {'n':0,'pnl':0,'mdd':0,'wr':0,'ex5':0}
    total = sum(t['net'] for t in trades)
    wins = [t for t in trades if t['net']>0]
    eq=0; pk=0; md=0
    for t in trades:
        eq+=t['net']; pk=max(pk,eq); md=max(md,pk-eq)
    n_top = max(1,int(len(trades)*0.05))
    ex_top = sorted(trades, key=lambda t:t['net'],reverse=True)[n_top:]
    return {
        'n': len(trades), 'pnl': round(total,2),
        'mdd': round(md,2),
        'wr': round(len(wins)/len(trades)*100,1),
        'ex5': round(sum(t['net'] for t in ex_top),2),
        'daily': round(total/DATA_DAYS,3),
    }


def main():
    bs = {'max_sl_atr':4.0,'trail_K':2.5,'max_hold_bars':192,'body_min_ratio':0.60}
    ibt.strat = copy.deepcopy(ibt.strat); ibt.strat.update(bs)
    ibt.trail_K = 2.5; ibt.max_hold = 192
    ibt.sig = C1BreakoutSignal(ibt.strat)
    passes = precompute_trend_pass(192, 1.0)

    print('='*120)
    print('  Progressive Trail NOISE Sensitivity Check — bar_close vs intrabar')
    print(f'  Data: {ibt.n15} bars = {DATA_DAYS:.1f} days, slippage=MED, FEE=0.10 (RT)')
    print('='*120)

    configs = [
        ('baseline tk=2.5', 2.5, 2.5, 99.0),
        ('v4.8.0 thr=0.9 tkT=0.5', 2.5, 0.5, 0.9),
        ('thr=0.9 tkT=0.3', 2.5, 0.3, 0.9),
        ('thr=0.9 tkT=0.1', 2.5, 0.1, 0.9),
        ('thr=0.9 tkT=1.0', 2.5, 1.0, 0.9),
        ('thr=0.7 tkT=0.3', 2.5, 0.3, 0.7),
        ('thr=1.2 tkT=0.1', 2.5, 0.1, 1.2),
    ]

    print(f'\n{"Config":<28} {"Mode":<10} {"n":>4} {"PnL":>8} {"daily":>7} {"MDD":>6} {"WR":>5} {"ex5":>8}')
    print('-'*100)

    results = {}
    for name, tb, tp, thr in configs:
        for mode_name, factory in [('bar_close', make_exit_bar_close),
                                    ('intrabar', make_exit_intrabar)]:
            trades = run_bt(factory(tb, tp, thr), passes)
            s = stats(trades)
            results[(name, mode_name)] = s
            print(f'{name:<28} {mode_name:<10} {s["n"]:>4} {s["pnl"]:>+7.2f} {s["daily"]:>+6.3f} '
                  f'{s["mdd"]:>5.2f} {s["wr"]:>4.1f}% {s["ex5"]:>+7.2f}')
        # Gap
        bc = results[(name, 'bar_close')]
        ib = results[(name, 'intrabar')]
        gap_pnl = ib['pnl'] - bc['pnl']
        gap_ratio = (gap_pnl / bc['pnl'] * 100) if bc['pnl'] > 0 else float('nan')
        print(f'{name:<28} {"GAP":<10} {"":>4} {gap_pnl:>+7.2f} ({gap_ratio:+.1f}%)  '
              f'MDD Δ={ib["mdd"]-bc["mdd"]:+.2f}, ex5 Δ={ib["ex5"]-bc["ex5"]:+.2f}')
        print()

    print('='*120)
    print('  NOISE 충격 요약')
    print('='*120)
    print(f'{"Config":<28} {"bar_close":>10} {"intrabar":>10} {"Δ":>8} {"Δ%":>7}')
    for name, _, _, _ in configs:
        bc = results[(name,'bar_close')]['pnl']
        ib = results[(name,'intrabar')]['pnl']
        d = ib - bc
        dp = (d/bc*100) if bc > 0 else float('nan')
        print(f'{name:<28} {bc:>+9.2f} {ib:>+9.2f} {d:>+7.2f} {dp:>+6.1f}%')


if __name__ == '__main__':
    main()
