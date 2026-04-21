#!/usr/bin/env python3
"""Trail Variants Extended Research (2026-04-21).

User-requested exploration:
  A. 광범위 threshold × K_post grid (tighten + loosen)
  C. Entry-anchor: 수익 X% 확보 시 trail을 entry_price로 고정
  D. Signal-bar-open anchor: 수익이 |entry - signal_open| × M 이상 진행 시 trail = signal_open
  E. Combined phased (baseline → entry-anchor → tight)

Base: cand_C_b0.60 + trend_filter(1.0%/192)
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


# ─── Variant A: progressive (threshold × K_post) ─────────────────
def make_exit_progressive(tk_base, tk_post, threshold_pct):
    def check(pos, bar, tk_ignored):
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
            k = tk_post if bpl >= threshold_pct else tk_base
            td = k*a/c15[bar]*100
            if bpl - cpl >= td:
                r = max(0, bpl - td)
                px = ep*(1+r/100) if d=='LONG' else ep*(1-r/100)
                return {'reason':'TRAIL_TP','exit_price':px}
        return None
    return check


# ─── Variant C: entry-anchor ─────────────────────────────────────
def make_exit_entry_anchor(anchor_profit_pct):
    """수익 anchor_profit_pct 달성 시 trail = entry_price 고정 (baseline K=2.5 이전엔 사용)."""
    def check(pos, bar, tk_ignored):
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

        # Entry-anchor mode: once best_pnl reaches anchor_profit_pct, trail = entry_price
        if bpl >= anchor_profit_pct:
            if d == 'LONG' and l15[bar] <= ep:
                return {'reason':'TRAIL_TP','exit_price':ep}
            elif d == 'SHORT' and h15[bar] >= ep:
                return {'reason':'TRAIL_TP','exit_price':ep}
            return None  # anchored, no ATR trail
        # Fallback: baseline K=2.5 trail
        cpl = (c15[bar]/ep-1)*100 if d=='LONG' else (1-c15[bar]/ep)*100
        a = atr[bar]
        if bpl > ibt.trail_act and not math.isnan(a) and a>0:
            td = 2.5*a/c15[bar]*100
            if bpl - cpl >= td:
                r = max(0, bpl - td)
                px = ep*(1+r/100) if d=='LONG' else ep*(1-r/100)
                return {'reason':'TRAIL_TP','exit_price':px}
        return None
    return check


# ─── Variant D: signal-bar-open anchor ────────────────────────────
def make_exit_signal_anchor(multiplier):
    """수익이 |entry - signal_open| × M 이상 진행 시 trail = signal_open (stop @ o[signal_bar])."""
    def check(pos, bar, tk_ignored):
        d, ep, bp, sl = pos['d'], pos['ep'], pos['bp'], pos['sl']
        c15, l15, h15, atr = ibt.c15, ibt.l15, ibt.h15, ibt.atr14
        if d == 'LONG' and l15[bar] <= sl: return {'reason':'SL','exit_price':sl}
        elif d == 'SHORT' and h15[bar] >= sl: return {'reason':'SL','exit_price':sl}
        worst = (l15[bar]/ep-1)*100 if d=='LONG' else (1-h15[bar]/ep)*100
        if worst <= -ibt.emergency_sl:
            px = ep*(1-ibt.emergency_sl/100) if d=='LONG' else ep*(1+ibt.emergency_sl/100)
            return {'reason':'EMERGENCY','exit_price':px}
        if pos['bh'] >= ibt.max_hold: return {'reason':'TIMEOUT','exit_price':c15[bar]}

        # Signal bar open (bar before entry_bar)
        sbar = pos['entry_bar'] - 1
        if sbar < 0: sbar = 0
        so = ibt.o15[sbar]
        delta = abs(ep - so)
        # 1:1 symmetric trigger for LONG: ep + M*delta (higher than entry by M×delta)
        if d == 'LONG':
            trigger_profit_price = ep + multiplier * delta
            if bp >= trigger_profit_price:
                # Trail pinned at signal_open
                if l15[bar] <= so:
                    return {'reason':'TRAIL_TP','exit_price':so}
                return None  # anchored
        else:  # SHORT
            trigger_profit_price = ep - multiplier * delta
            if bp <= trigger_profit_price:
                if h15[bar] >= so:
                    return {'reason':'TRAIL_TP','exit_price':so}
                return None

        # Fallback: baseline K=2.5 trail
        bpl = (bp/ep-1)*100 if d=='LONG' else (1-bp/ep)*100
        cpl = (c15[bar]/ep-1)*100 if d=='LONG' else (1-c15[bar]/ep)*100
        a = atr[bar]
        if bpl > ibt.trail_act and not math.isnan(a) and a>0:
            td = 2.5*a/c15[bar]*100
            if bpl - cpl >= td:
                r = max(0, bpl - td)
                px = ep*(1+r/100) if d=='LONG' else ep*(1-r/100)
                return {'reason':'TRAIL_TP','exit_price':px}
        return None
    return check


# ─── Variant E: Combined phased (baseline → entry-anchor → tight) ─
def make_exit_combined(anchor_thr, tight_thr, tk_tight):
    """3-phase:
      phase 1 (bpl < anchor_thr): baseline K=2.5
      phase 2 (anchor_thr <= bpl < tight_thr): trail = entry_price (anchor)
      phase 3 (bpl >= tight_thr): tight K=tk_tight trail from best
    """
    def check(pos, bar, tk_ignored):
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

        if bpl >= tight_thr:  # phase 3
            if bpl > ibt.trail_act and not math.isnan(a) and a>0:
                td = tk_tight*a/c15[bar]*100
                if bpl - cpl >= td:
                    r = max(0, bpl - td)
                    px = ep*(1+r/100) if d=='LONG' else ep*(1-r/100)
                    return {'reason':'TRAIL_TP','exit_price':px}
        elif bpl >= anchor_thr:  # phase 2 anchor at entry
            if d == 'LONG' and l15[bar] <= ep:
                return {'reason':'TRAIL_TP','exit_price':ep}
            elif d == 'SHORT' and h15[bar] >= ep:
                return {'reason':'TRAIL_TP','exit_price':ep}
        else:  # phase 1 baseline
            if bpl > ibt.trail_act and not math.isnan(a) and a>0:
                td = 2.5*a/c15[bar]*100
                if bpl - cpl >= td:
                    r = max(0, bpl - td)
                    px = ep*(1+r/100) if d=='LONG' else ep*(1-r/100)
                    return {'reason':'TRAIL_TP','exit_price':px}
        return None
    return check


# ─── BT runner ────────────────────────────────────────────────────
def run_bt(check_fn, passes):
    orig = ibt._check_exit_bar_close
    ibt._check_exit_bar_close = check_fn
    try:
        return run_bt_with_regime(mode='bar_close', regime_passes=passes, slippage=SLIP_MED)
    finally:
        ibt._check_exit_bar_close = orig


def stats(trades):
    if not trades: return {'n':0,'pnl':0,'daily':0,'mdd':0,'wr':0,'rr':0,'ex5':0,
                           'boot_pos':0,'boot_sh':0,'tpd':0}
    total = sum(t['net'] for t in trades)
    wins = [t for t in trades if t['net']>0]
    losses = [t for t in trades if t['net']<0]
    aw = mean(t['net'] for t in wins) if wins else 0
    al = abs(mean(t['net'] for t in losses)) if losses else 0
    eq=0; pk=0; md=0
    for t in trades:
        eq+=t['net']; pk=max(pk,eq); md=max(md,pk-eq)
    n_top = max(1,int(len(trades)*0.05))
    ex_top = sorted(trades, key=lambda t:t['net'],reverse=True)[n_top:]
    # bootstrap
    rng = random.Random(42)
    pnls = []
    for _ in range(1000):
        s = rng.randint(220, ibt.n15 - 289)
        e = s + 288
        pnls.append(sum(t['net'] for t in trades if s <= t['entry_bar'] < e))
    pos_pct = sum(1 for p in pnls if p>0)/1000*100
    sh = mean(pnls)/stdev(pnls) if stdev(pnls)>0 else 0
    return {
        'n': len(trades), 'pnl': round(total,2),
        'daily': round(total/DATA_DAYS,3),
        'mdd': round(md,2),
        'wr': round(len(wins)/len(trades)*100,1),
        'rr': round(aw/al if al>0 else 0,2),
        'ex5': round(sum(t['net'] for t in ex_top),2),
        'boot_pos': round(pos_pct,1),
        'boot_sh': round(sh,3),
        'tpd': round(len(trades)/DATA_DAYS,2),
    }


def wf5(trades):
    oos_w = (ibt.n15 - 26) // 5
    folds = []
    for i in range(5):
        s = 26 + i*oos_w
        e = s + oos_w
        folds.append(sum(t['net'] for t in trades if s <= t['entry_bar'] < e))
    return folds, sum(1 for p in folds if p>0)


def row(name, s, base_pnl):
    vs = s['pnl'] - base_pnl
    print(f'{name:<30} {s["n"]:>4} {s["pnl"]:>+7.2f} {s["daily"]:>+6.3f} '
          f'{s["wr"]:>4.1f} {s["rr"]:>4.2f} {s["mdd"]:>5.2f} '
          f'{s["ex5"]:>+7.2f} {s["boot_pos"]:>5.1f}% {s["boot_sh"]:>+6.3f}  {vs:>+7.2f}')


def hdr():
    print(f'{"Variant":<30} {"n":>4} {"PnL":>7} {"daily":>7} {"WR":>4} {"RR":>4} '
          f'{"MDD":>5} {"ex5":>7} {"pos":>6} {"shr":>7}  vs_base')


def main():
    bs = {'max_sl_atr':4.0,'trail_K':2.5,'max_hold_bars':192,'body_min_ratio':0.60}
    ibt.strat = copy.deepcopy(ibt.strat); ibt.strat.update(bs)
    ibt.trail_K = 2.5; ibt.max_hold = 192
    ibt.sig = C1BreakoutSignal(ibt.strat)
    passes = precompute_trend_pass(192, 1.0)

    print('='*130)
    print('  Trail Variants Extended Research')
    print(f'  Data: {ibt.n15} bars = {DATA_DAYS:.1f} days, base: cand_C+trend(1.0/192)')
    print('='*130)

    # Baseline reference
    base_trades = run_bt(make_exit_progressive(2.5, 2.5, 99), passes)
    b = stats(base_trades)
    BP = b['pnl']
    print(f'\nBASELINE (tk=2.5 fixed): n={b["n"]}, PnL={BP:+.2f}, MDD={b["mdd"]}, '
          f'ex5={b["ex5"]}, pos={b["boot_pos"]}%, sh={b["boot_sh"]}')

    # v4.8.0 current active config for reference
    curr = run_bt(make_exit_progressive(2.5, 0.5, 0.9), passes)
    c = stats(curr)
    print(f'CURRENT v4.8.0 (thr=0.9, tkT=0.5): n={c["n"]}, PnL={c["pnl"]:+.2f}, MDD={c["mdd"]}, '
          f'ex5={c["ex5"]}, pos={c["boot_pos"]}%, sh={c["boot_sh"]}')

    results = [('baseline_ref', b), ('v4.8.0_current', c)]

    # ═══ A: Wider progressive grid ═══
    print('\n' + '='*130)
    print('  VARIANT A: Wider threshold × K_post (tighten + loosen + early trigger)')
    print('='*130)
    hdr()
    thr_grid = [0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.2, 1.5, 2.0, 2.5]
    tk_grid = [0.1, 0.3, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0]
    for thr in thr_grid:
        for tk in tk_grid:
            if thr == 0.9 and tk == 0.5: continue  # already shown
            t = run_bt(make_exit_progressive(2.5, tk, thr), passes)
            s = stats(t)
            label = f'A thr={thr} tkT={tk}'
            row(label, s, BP)
            results.append((label, s, t))

    # ═══ C: Entry-anchor ═══
    print('\n' + '='*130)
    print('  VARIANT C: Entry-anchor (profit X% 이상 → trail = entry_price)')
    print('='*130)
    hdr()
    anchor_grid = [0.2, 0.3, 0.5, 0.7, 0.9, 1.0, 1.5, 2.0]
    for a in anchor_grid:
        t = run_bt(make_exit_entry_anchor(a), passes)
        s = stats(t)
        label = f'C anchor={a}%'
        row(label, s, BP)
        results.append((label, s, t))

    # ═══ D: Signal-bar-open anchor ═══
    print('\n' + '='*130)
    print('  VARIANT D: Signal-bar-open anchor (수익 >= M × |entry-signal_open|)')
    print('='*130)
    hdr()
    mult_grid = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
    for m in mult_grid:
        t = run_bt(make_exit_signal_anchor(m), passes)
        s = stats(t)
        label = f'D signal_x{m}'
        row(label, s, BP)
        results.append((label, s, t))

    # ═══ E: Combined 3-phase ═══
    print('\n' + '='*130)
    print('  VARIANT E: Combined 3-phase (baseline → entry-anchor → tight)')
    print('='*130)
    hdr()
    combined = [
        (0.5, 1.0, 0.5),  # anchor 0.5, tight 1.0 with K=0.5
        (0.5, 1.5, 0.5),
        (0.3, 1.0, 0.5),
        (0.5, 1.2, 0.3),
        (0.7, 1.5, 0.5),
        (0.5, 1.0, 1.0),
    ]
    for ac, tc, ktk in combined:
        t = run_bt(make_exit_combined(ac, tc, ktk), passes)
        s = stats(t)
        label = f'E anc={ac} tgt={tc} K={ktk}'
        row(label, s, BP)
        results.append((label, s, t))

    # ═══ Rankings ═══
    print('\n' + '='*130)
    print('  SUMMARY — Top 15 by PnL (F6 ex5 must > 0 to qualify)')
    print('='*130)
    hdr()
    qualified = [r for r in results if len(r) == 3 and r[1]['ex5'] > 0 and r[1]['pnl'] > BP]
    qualified.sort(key=lambda x: x[1]['pnl'], reverse=True)
    for name, s, _ in qualified[:15]:
        row(name, s, BP)

    print('\n' + '='*130)
    print('  SUMMARY — Top 15 by Sharpe (PnL > baseline)')
    print('='*130)
    hdr()
    pnl_beats = [r for r in results if len(r) == 3 and r[1]['pnl'] > BP]
    pnl_beats.sort(key=lambda x: x[1]['boot_sh'], reverse=True)
    for name, s, _ in pnl_beats[:15]:
        row(name, s, BP)

    print('\n' + '='*130)
    print('  SUMMARY — Top 10 by F6 ex_top5 (PnL > baseline)')
    print('='*130)
    hdr()
    f6_sorted = sorted(pnl_beats, key=lambda x: x[1]['ex5'], reverse=True)
    for name, s, _ in f6_sorted[:10]:
        row(name, s, BP)

    # ═══ WF on top 5 winners ═══
    print('\n' + '='*130)
    print('  WF 5-fold on top 5 by PnL')
    print('='*130)
    print(f'{"Candidate":<30} {"f1":>7} {"f2":>7} {"f3":>7} {"f4":>7} {"f5":>7} {"pos":>4}')
    # Baseline WF
    bfolds, bpos = wf5(base_trades)
    fs = ' '.join(f'{p:>+7.2f}' for p in bfolds)
    print(f'{"baseline_ref":<30} {fs}  {bpos}/5')
    for name, s, tr in qualified[:5]:
        folds, pos = wf5(tr)
        fs = ' '.join(f'{p:>+7.2f}' for p in folds)
        print(f'{name:<30} {fs}  {pos}/5')


if __name__ == '__main__':
    main()
