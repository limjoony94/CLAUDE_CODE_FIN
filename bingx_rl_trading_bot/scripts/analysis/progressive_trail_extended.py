#!/usr/bin/env python3
"""Progressive Trail Extended Study — 다양한 조합 점검

Phase 1: Fine grid near sweet spot (0.6~1.3% × K 0.5~1.5)
Phase 2: Vary tk_base (early K 1.5~3.5)
Phase 3: Two-step (3-tier trail K)
Phase 4: Time-based (hold_bars threshold instead of profit)
Phase 5: ATR-unit threshold
Phase 6: WF 5-fold validation on top candidates
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


# ─── Exit variants (monkey-patch targets) ─────────────────────────

def make_check_exit_profit(tk_base, tk_post, threshold_pct):
    """1-step: tk changes at profit threshold."""
    def check(pos, bar, tk_ignored):
        d, ep, bp, sl = pos['d'], pos['ep'], pos['bp'], pos['sl']
        c15 = ibt.c15; l15 = ibt.l15; h15 = ibt.h15; atr14 = ibt.atr14
        if d == 'LONG' and l15[bar] <= sl:
            return {'reason': 'SL', 'exit_price': sl}
        elif d == 'SHORT' and h15[bar] >= sl:
            return {'reason': 'SL', 'exit_price': sl}
        if d == 'LONG':
            worst = (l15[bar]/ep-1)*100
        else:
            worst = (1-h15[bar]/ep)*100
        if worst <= -ibt.emergency_sl:
            px = ep*(1 - ibt.emergency_sl/100) if d=='LONG' else ep*(1 + ibt.emergency_sl/100)
            return {'reason':'EMERGENCY', 'exit_price': px}
        if pos['bh'] >= ibt.max_hold:
            return {'reason':'TIMEOUT', 'exit_price': c15[bar]}
        if d == 'LONG':
            best_pnl = (bp/ep-1)*100; cur_pnl = (c15[bar]/ep-1)*100
        else:
            best_pnl = (1-bp/ep)*100; cur_pnl = (1-c15[bar]/ep)*100
        a = atr14[bar]
        if best_pnl > ibt.trail_act and not math.isnan(a) and a > 0:
            tk = tk_post if best_pnl >= threshold_pct else tk_base
            tdp = tk * a / c15[bar] * 100
            dd = best_pnl - cur_pnl
            if dd >= tdp:
                realized = max(0, best_pnl - tdp)
                px = ep*(1 + realized/100) if d=='LONG' else ep*(1 - realized/100)
                return {'reason':'TRAIL_TP', 'exit_price': px}
        return None
    return check


def make_check_exit_3tier(tk1, tk2, tk3, thr1, thr2):
    """3-tier: K1 (<thr1), K2 (thr1~thr2), K3 (>thr2)."""
    def check(pos, bar, tk_ignored):
        d, ep, bp, sl = pos['d'], pos['ep'], pos['bp'], pos['sl']
        c15 = ibt.c15; l15 = ibt.l15; h15 = ibt.h15; atr14 = ibt.atr14
        if d == 'LONG' and l15[bar] <= sl:
            return {'reason':'SL','exit_price':sl}
        elif d == 'SHORT' and h15[bar] >= sl:
            return {'reason':'SL','exit_price':sl}
        worst = (l15[bar]/ep-1)*100 if d=='LONG' else (1-h15[bar]/ep)*100
        if worst <= -ibt.emergency_sl:
            px = ep*(1-ibt.emergency_sl/100) if d=='LONG' else ep*(1+ibt.emergency_sl/100)
            return {'reason':'EMERGENCY','exit_price':px}
        if pos['bh'] >= ibt.max_hold:
            return {'reason':'TIMEOUT','exit_price':c15[bar]}
        if d == 'LONG':
            best_pnl=(bp/ep-1)*100; cur_pnl=(c15[bar]/ep-1)*100
        else:
            best_pnl=(1-bp/ep)*100; cur_pnl=(1-c15[bar]/ep)*100
        a=atr14[bar]
        if best_pnl > ibt.trail_act and not math.isnan(a) and a>0:
            if best_pnl < thr1: tk = tk1
            elif best_pnl < thr2: tk = tk2
            else: tk = tk3
            tdp = tk*a/c15[bar]*100
            dd = best_pnl - cur_pnl
            if dd >= tdp:
                realized = max(0, best_pnl - tdp)
                px = ep*(1+realized/100) if d=='LONG' else ep*(1-realized/100)
                return {'reason':'TRAIL_TP','exit_price':px}
        return None
    return check


def make_check_exit_time(tk_base, tk_post, hold_threshold):
    """Time-based: tk changes after hold_threshold bars."""
    def check(pos, bar, tk_ignored):
        d, ep, bp, sl = pos['d'], pos['ep'], pos['bp'], pos['sl']
        c15 = ibt.c15; l15 = ibt.l15; h15 = ibt.h15; atr14 = ibt.atr14
        if d == 'LONG' and l15[bar] <= sl:
            return {'reason':'SL','exit_price':sl}
        elif d == 'SHORT' and h15[bar] >= sl:
            return {'reason':'SL','exit_price':sl}
        worst = (l15[bar]/ep-1)*100 if d=='LONG' else (1-h15[bar]/ep)*100
        if worst <= -ibt.emergency_sl:
            px = ep*(1-ibt.emergency_sl/100) if d=='LONG' else ep*(1+ibt.emergency_sl/100)
            return {'reason':'EMERGENCY','exit_price':px}
        if pos['bh'] >= ibt.max_hold:
            return {'reason':'TIMEOUT','exit_price':c15[bar]}
        if d == 'LONG':
            best_pnl=(bp/ep-1)*100; cur_pnl=(c15[bar]/ep-1)*100
        else:
            best_pnl=(1-bp/ep)*100; cur_pnl=(1-c15[bar]/ep)*100
        a=atr14[bar]
        if best_pnl > ibt.trail_act and not math.isnan(a) and a>0:
            tk = tk_post if pos['bh'] >= hold_threshold else tk_base
            tdp = tk*a/c15[bar]*100
            dd = best_pnl-cur_pnl
            if dd >= tdp:
                realized = max(0, best_pnl - tdp)
                px = ep*(1+realized/100) if d=='LONG' else ep*(1-realized/100)
                return {'reason':'TRAIL_TP','exit_price':px}
        return None
    return check


def make_check_exit_atr(tk_base, tk_post, thr_atr_units):
    """ATR-unit threshold: switch when best_pnl >= thr_atr_units * ATR%."""
    def check(pos, bar, tk_ignored):
        d, ep, bp, sl = pos['d'], pos['ep'], pos['bp'], pos['sl']
        c15 = ibt.c15; l15 = ibt.l15; h15 = ibt.h15; atr14 = ibt.atr14
        if d == 'LONG' and l15[bar] <= sl:
            return {'reason':'SL','exit_price':sl}
        elif d == 'SHORT' and h15[bar] >= sl:
            return {'reason':'SL','exit_price':sl}
        worst = (l15[bar]/ep-1)*100 if d=='LONG' else (1-h15[bar]/ep)*100
        if worst <= -ibt.emergency_sl:
            px = ep*(1-ibt.emergency_sl/100) if d=='LONG' else ep*(1+ibt.emergency_sl/100)
            return {'reason':'EMERGENCY','exit_price':px}
        if pos['bh'] >= ibt.max_hold:
            return {'reason':'TIMEOUT','exit_price':c15[bar]}
        if d == 'LONG':
            best_pnl=(bp/ep-1)*100; cur_pnl=(c15[bar]/ep-1)*100
        else:
            best_pnl=(1-bp/ep)*100; cur_pnl=(1-c15[bar]/ep)*100
        a=atr14[bar]
        if best_pnl > ibt.trail_act and not math.isnan(a) and a>0:
            atr_pct = a / c15[bar] * 100
            threshold = thr_atr_units * atr_pct
            tk = tk_post if best_pnl >= threshold else tk_base
            tdp = tk*a/c15[bar]*100
            dd = best_pnl-cur_pnl
            if dd >= tdp:
                realized = max(0, best_pnl - tdp)
                px = ep*(1+realized/100) if d=='LONG' else ep*(1-realized/100)
                return {'reason':'TRAIL_TP','exit_price':px}
        return None
    return check


# ─── BT runner ────────────────────────────────────────────────────

def run_bt_patched(check_fn, regime_passes, slippage=SLIP_MED):
    orig = ibt._check_exit_bar_close
    ibt._check_exit_bar_close = check_fn
    try:
        return run_bt_with_regime(mode='bar_close', regime_passes=regime_passes, slippage=slippage)
    finally:
        ibt._check_exit_bar_close = orig


# ─── Stats ────────────────────────────────────────────────────────

def stats(trades):
    if not trades:
        return {'n':0,'pnl':0,'daily':0,'wr':0,'rr':0,'mdd':0,'mean_net':0,'tpd':0,
                'ex5':0,'boot_pos':0,'boot_sharpe':0,'boot_p5':0}
    total = sum(t['net'] for t in trades)
    daily = total / DATA_DAYS
    wins = [t for t in trades if t['net']>0]
    losses = [t for t in trades if t['net']<0]
    wr = len(wins)/len(trades)*100
    aw = mean(t['net'] for t in wins) if wins else 0
    al = abs(mean(t['net'] for t in losses)) if losses else 0
    rr = aw/al if al>0 else 0
    eq=0; pk=0; md=0
    for t in trades:
        eq+=t['net']; pk=max(pk,eq); md=max(md,pk-eq)
    n_top = max(1,int(len(trades)*0.05))
    ex_top = sorted(trades, key=lambda t:t['net'],reverse=True)[n_top:]
    pnl_ex = sum(t['net'] for t in ex_top)
    # bootstrap
    rng = random.Random(42)
    pnls = []
    for _ in range(1000):
        s = rng.randint(220, ibt.n15 - 289)
        e = s + 288
        pnls.append(sum(t['net'] for t in trades if s <= t['entry_bar'] < e))
    ps = sorted(pnls)
    pos = sum(1 for p in pnls if p>0)/1000*100
    sh = mean(pnls)/stdev(pnls) if stdev(pnls)>0 else 0
    return {'n':len(trades),'pnl':round(total,2),'daily':round(daily,3),
            'wr':round(wr,1),'rr':round(rr,2),'mdd':round(md,2),
            'mean_net':round(total/len(trades),3),'tpd':round(len(trades)/DATA_DAYS,2),
            'ex5':round(pnl_ex,2),'boot_pos':round(pos,1),
            'boot_sharpe':round(sh,3),'boot_p5':round(ps[50],2)}


def wf_breakdown(trades, n=5):
    """5-fold partition by entry_bar. Returns (fold_pnls, positive_count)."""
    if not trades: return [], 0
    bars = ibt.n15
    fw = bars // n
    fpnls = []
    for i in range(n):
        s = i * fw
        e = (i+1)*fw if i < n-1 else bars
        p = sum(t['net'] for t in trades if s <= t['entry_bar'] < e)
        fpnls.append(round(p,2))
    return fpnls, sum(1 for p in fpnls if p>0)


def row_print(label, s, base_pnl):
    vs = s['pnl'] - base_pnl
    print(f'{label:<28} {s["n"]:>4} {s["pnl"]:>+7.2f} {s["daily"]:>+6.3f} '
          f'{s["wr"]:>4.1f} {s["rr"]:>4.2f} {s["mdd"]:>5.2f} {s["mean_net"]:>+5.2f} '
          f'{s["tpd"]:>4.1f} {s["ex5"]:>+7.2f} {s["boot_pos"]:>5.1f}% {s["boot_sharpe"]:>+6.3f} '
          f'{s["boot_p5"]:>+6.2f}  {vs:>+7.2f}')


def hdr():
    print(f'{"Variant":<28} {"n":>4} {"PnL":>7} {"daily":>7} {"WR":>4} {"RR":>4} '
          f'{"MDD":>5} {"mean":>5} {"tpd":>4} {"ex5":>7} {"pos":>6} {"shr":>7} '
          f'{"p5":>6}  vs_base')


# ─── Main ─────────────────────────────────────────────────────────

def main():
    # Setup base strategy
    base_strat = {'max_sl_atr':4.0,'trail_K':2.5,'max_hold_bars':192,'body_min_ratio':0.60}
    ibt.strat = copy.deepcopy(ibt.strat); ibt.strat.update(base_strat)
    ibt.trail_K = 2.5; ibt.max_hold = 192
    ibt.sig = C1BreakoutSignal(ibt.strat)
    passes = precompute_trend_pass(192, 1.0)

    print('='*130)
    print(f'  Progressive Trail EXTENDED — base strategy: cand_C_b0.60 + trend(1.0/192)')
    print(f'  Data: {ibt.n15} bars = {DATA_DAYS:.1f} days, slippage=MED')
    print('='*130)

    # Baseline
    base = run_bt_patched(make_check_exit_profit(2.5, 2.5, 99), passes)
    bs = stats(base)
    print(f'\nBASELINE (tk=2.5 fixed): n={bs["n"]}, PnL={bs["pnl"]:+.2f}, daily={bs["daily"]:+.3f}, '
          f'ex5={bs["ex5"]:+.2f}, pos={bs["boot_pos"]}%, sharpe={bs["boot_sharpe"]}')
    BP = bs['pnl']
    all_results = []  # collect for ranking

    # ═══ PHASE 1: Fine grid near sweet spot ═══
    print('\n' + '='*130)
    print('  PHASE 1: Fine grid (thr 0.6~1.3, tkT 0.5~1.5)')
    print('='*130)
    hdr()
    fine_thrs = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3]
    fine_tks  = [0.5, 0.7, 0.9, 1.0, 1.2, 1.5]
    for thr in fine_thrs:
        for tkT in fine_tks:
            trades = run_bt_patched(make_check_exit_profit(2.5, tkT, thr), passes)
            s = stats(trades)
            row_print(f'P1 thr={thr} tkT={tkT}', s, BP)
            all_results.append((f'P1 thr={thr} tkT={tkT}', s, trades))

    # ═══ PHASE 2: Vary tk_base ═══
    print('\n' + '='*130)
    print('  PHASE 2: Vary tk_base (early K 1.5~3.5) × tkT {1.0, 1.5, 2.0}, thr=1.0 fixed')
    print('='*130)
    hdr()
    tk_bases = [1.5, 2.0, 2.5, 3.0, 3.5]
    tkT_set = [1.0, 1.5, 2.0, 2.5]
    for tkB in tk_bases:
        for tkT in tkT_set:
            trades = run_bt_patched(make_check_exit_profit(tkB, tkT, 1.0), passes)
            s = stats(trades)
            row_print(f'P2 B={tkB} T={tkT}', s, BP)
            all_results.append((f'P2 B={tkB} T={tkT}', s, trades))

    # ═══ PHASE 3: 3-tier ═══
    print('\n' + '='*130)
    print('  PHASE 3: 3-tier (K1 early, K2 mid, K3 late) — thr1=0.5, thr2=1.5')
    print('='*130)
    hdr()
    tier_grid = [
        # (K1, K2, K3) — various combinations
        (2.5, 2.0, 1.0),   # gradual tighten
        (2.5, 1.5, 1.0),
        (2.5, 1.5, 0.5),
        (3.0, 2.0, 1.0),
        (3.0, 1.5, 0.5),
        (3.5, 2.0, 1.0),
        (2.0, 2.5, 3.0),   # gradual loosen (inverse test)
        (2.5, 2.0, 1.5),
        (2.5, 1.0, 1.5),   # tighten then loosen (inverse U)
    ]
    for tk1, tk2, tk3 in tier_grid:
        trades = run_bt_patched(make_check_exit_3tier(tk1, tk2, tk3, 0.5, 1.5), passes)
        s = stats(trades)
        row_print(f'P3 {tk1}/{tk2}/{tk3}', s, BP)
        all_results.append((f'P3 {tk1}/{tk2}/{tk3}', s, trades))

    # ═══ PHASE 4: Time-based ═══
    print('\n' + '='*130)
    print('  PHASE 4: Time-based (hold_bars threshold, tk_base=2.5)')
    print('='*130)
    hdr()
    time_grid = [(8, 1.0), (8, 1.5), (16, 1.0), (16, 1.5), (16, 2.0),
                 (32, 1.0), (32, 1.5), (48, 1.0), (48, 1.5), (64, 1.0)]
    for ht, tkT in time_grid:
        trades = run_bt_patched(make_check_exit_time(2.5, tkT, ht), passes)
        s = stats(trades)
        row_print(f'P4 hold>={ht} T={tkT}', s, BP)
        all_results.append((f'P4 hold>={ht} T={tkT}', s, trades))

    # ═══ PHASE 5: ATR-unit threshold ═══
    print('\n' + '='*130)
    print('  PHASE 5: ATR-unit threshold (thr_in_ATR × tkT)')
    print('='*130)
    hdr()
    atr_grid = [(0.5, 1.0), (1.0, 1.0), (1.5, 1.0), (2.0, 1.0),
                (1.0, 1.5), (1.5, 1.5), (1.0, 0.5), (1.5, 0.7)]
    for thr_a, tkT in atr_grid:
        trades = run_bt_patched(make_check_exit_atr(2.5, tkT, thr_a), passes)
        s = stats(trades)
        row_print(f'P5 ATRthr={thr_a} T={tkT}', s, BP)
        all_results.append((f'P5 ATRthr={thr_a} T={tkT}', s, trades))

    # ═══ Summary Rankings ═══
    print('\n' + '='*130)
    print('  SUMMARY — Top 10 by F6 (ex_top5)')
    print('='*130)
    hdr()
    sorted_f6 = sorted(all_results, key=lambda x: x[1]['ex5'], reverse=True)
    for name, s, _ in sorted_f6[:10]:
        row_print(name, s, BP)

    print('\n' + '='*130)
    print('  SUMMARY — Top 10 by Sharpe')
    print('='*130)
    hdr()
    sorted_sh = sorted(all_results, key=lambda x: x[1]['boot_sharpe'], reverse=True)
    for name, s, _ in sorted_sh[:10]:
        row_print(name, s, BP)

    print('\n' + '='*130)
    print('  SUMMARY — Top 10 by PnL (must also beat baseline on F6)')
    print('='*130)
    hdr()
    f6_passers = [r for r in all_results if r[1]['ex5'] > bs['ex5']]
    sorted_pnl = sorted(f6_passers, key=lambda x: x[1]['pnl'], reverse=True)
    for name, s, _ in sorted_pnl[:10]:
        row_print(name, s, BP)

    # ═══ PHASE 6: WF 5-fold on top candidates ═══
    print('\n' + '='*130)
    print('  PHASE 6: WF 5-fold on top candidates (need fold stability)')
    print('='*130)
    # union: top-5 by F6 + top-5 by Sharpe
    top_candidates = {}
    for name, s, tr in sorted_f6[:5] + sorted_sh[:5]:
        top_candidates[name] = (s, tr)

    print(f'\n{"Candidate":<28} {"fold1":>7} {"fold2":>7} {"fold3":>7} {"fold4":>7} {"fold5":>7} {"pos":>5}')
    for name, (s, tr) in top_candidates.items():
        fpnls, pos = wf_breakdown(tr)
        fstr = ' '.join(f'{p:>+7.2f}' for p in fpnls)
        print(f'{name:<28} {fstr}  {pos}/5')

    # WF baseline
    fpnls_base, pos_base = wf_breakdown(base)
    print(f'{"BASELINE(tk=2.5)":<28} ' + ' '.join(f'{p:>+7.2f}' for p in fpnls_base) + f'  {pos_base}/5')


if __name__ == '__main__':
    main()
