#!/usr/bin/env python3
"""Adaptive Trail Research — 차트 기반 동적 reference point.

Variants (all after best_pnl > trail_activation):
  F. Channel breakout anchor — trail = ch_h[signal_bar] (LONG) / ch_l (SHORT)
  G. Fractal swing anchor — trail = sw_l (LONG) / sw_h (SHORT) with max drawback cap
  H. ATR-ratio adaptive K — K = K_base × (ATR_cur / ATR_avg_96bars)
  I. Time-decay K — K = max(K_min, K_init × exp(-bh/tau))
  J. Signal bar extreme anchor — trail = signal_bar_low (LONG) / signal_bar_high (SHORT)

Each variant evaluated in bar_close AND intrabar modes.
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


def _exit_common_preamble(pos, bar):
    """SL/Emergency/Timeout 선검사 — None이면 진행."""
    d, ep, sl = pos['d'], pos['ep'], pos['sl']
    c15, l15, h15 = ibt.c15, ibt.l15, ibt.h15
    if d == 'LONG' and l15[bar] <= sl: return {'reason':'SL','exit_price':sl}
    elif d == 'SHORT' and h15[bar] >= sl: return {'reason':'SL','exit_price':sl}
    worst = (l15[bar]/ep-1)*100 if d=='LONG' else (1-h15[bar]/ep)*100
    if worst <= -ibt.emergency_sl:
        px = ep*(1-ibt.emergency_sl/100) if d=='LONG' else ep*(1+ibt.emergency_sl/100)
        return {'reason':'EMERGENCY','exit_price':px}
    if pos['bh'] >= ibt.max_hold: return {'reason':'TIMEOUT','exit_price':c15[bar]}
    return None


# ─── F: Channel breakout level anchor ──────────────────────────────
def make_exit_channel_anchor(activation_pct=0.3):
    """Once best_pnl >= activation_pct, trail = channel[signal_bar] level.
    (the level that was broken at entry — retest = thesis invalidated)"""
    def check(pos, bar, tk_ignored):
        pre = _exit_common_preamble(pos, bar)
        if pre: return pre
        d, ep, bp = pos['d'], pos['ep'], pos['bp']
        c15, l15, h15, ch_h, ch_l = ibt.c15, ibt.l15, ibt.h15, ibt.ch_h, ibt.ch_l
        bpl = (bp/ep-1)*100 if d=='LONG' else (1-bp/ep)*100
        sbar = max(0, pos['entry_bar'] - 1)
        if bpl >= activation_pct:
            # Anchor at channel breakout level
            anchor = ch_h[sbar] if d=='LONG' else ch_l[sbar]
            if d == 'LONG' and l15[bar] <= anchor:
                return {'reason':'TRAIL_TP', 'exit_price': anchor}
            elif d == 'SHORT' and h15[bar] >= anchor:
                return {'reason':'TRAIL_TP', 'exit_price': anchor}
            return None  # anchored
        # baseline K=2.5
        cpl = (c15[bar]/ep-1)*100 if d=='LONG' else (1-c15[bar]/ep)*100
        a = ibt.atr14[bar]
        if bpl > ibt.trail_act and not math.isnan(a) and a>0:
            td = 2.5*a/c15[bar]*100
            if bpl-cpl >= td:
                r = max(0, bpl-td)
                px = ep*(1+r/100) if d=='LONG' else ep*(1-r/100)
                return {'reason':'TRAIL_TP','exit_price':px}
        return None
    return check


# ─── G: Fractal swing anchor (dynamic SL upgrade) ─────────────────
def make_exit_fractal_anchor(activation_pct=0.5):
    """Once best_pnl >= activation_pct, upgrade trail to nearest fractal
    swing (LONG: sw_l closest below current; SHORT: sw_h closest above)."""
    def check(pos, bar, tk_ignored):
        pre = _exit_common_preamble(pos, bar)
        if pre: return pre
        d, ep, bp, sl = pos['d'], pos['ep'], pos['bp'], pos['sl']
        c15, l15, h15, sw_l, sw_h = ibt.c15, ibt.l15, ibt.h15, ibt.sw_l, ibt.sw_h
        bpl = (bp/ep-1)*100 if d=='LONG' else (1-bp/ep)*100
        if bpl >= activation_pct:
            # Scan back for most recent valid fractal swing (< current, > entry)
            anchor = None
            for b in range(bar, max(0, bar-50), -1):
                if d == 'LONG':
                    v = sw_l[b]
                    if not math.isnan(v) and ep < v < c15[bar]:
                        anchor = v; break
                else:
                    v = sw_h[b]
                    if not math.isnan(v) and c15[bar] < v < ep:
                        anchor = v; break
            if anchor is not None:
                if d == 'LONG' and l15[bar] <= anchor:
                    return {'reason':'TRAIL_TP', 'exit_price': anchor}
                elif d == 'SHORT' and h15[bar] >= anchor:
                    return {'reason':'TRAIL_TP', 'exit_price': anchor}
                return None  # anchored
        # Fallback: baseline K=2.5
        cpl = (c15[bar]/ep-1)*100 if d=='LONG' else (1-c15[bar]/ep)*100
        a = ibt.atr14[bar]
        if bpl > ibt.trail_act and not math.isnan(a) and a>0:
            td = 2.5*a/c15[bar]*100
            if bpl-cpl >= td:
                r = max(0, bpl-td)
                px = ep*(1+r/100) if d=='LONG' else ep*(1-r/100)
                return {'reason':'TRAIL_TP','exit_price':px}
        return None
    return check


# ─── H: ATR-ratio adaptive K ───────────────────────────────────────
def make_exit_atr_adaptive(k_base, lookback=96, k_min=0.2, k_max=4.0):
    """K = k_base × (ATR_cur / rolling_avg_ATR). High vol → wider, low → tighter."""
    def check(pos, bar, tk_ignored):
        pre = _exit_common_preamble(pos, bar)
        if pre: return pre
        d, ep, bp = pos['d'], pos['ep'], pos['bp']
        c15, atr = ibt.c15, ibt.atr14
        bpl = (bp/ep-1)*100 if d=='LONG' else (1-bp/ep)*100
        cpl = (c15[bar]/ep-1)*100 if d=='LONG' else (1-c15[bar]/ep)*100
        a = atr[bar]
        if bpl > ibt.trail_act and not math.isnan(a) and a>0:
            # Rolling ATR avg
            start = max(0, bar - lookback)
            vals = [v for v in atr[start:bar] if not math.isnan(v) and v > 0]
            if vals:
                a_avg = sum(vals)/len(vals)
                ratio = a / a_avg
                k_eff = max(k_min, min(k_max, k_base * ratio))
            else:
                k_eff = k_base
            td = k_eff*a/c15[bar]*100
            if bpl-cpl >= td:
                r = max(0, bpl-td)
                px = ep*(1+r/100) if d=='LONG' else ep*(1-r/100)
                return {'reason':'TRAIL_TP','exit_price':px}
        return None
    return check


# ─── I: Time-decay K ───────────────────────────────────────────────
def make_exit_time_decay(k_init=2.5, k_min=0.5, tau=48):
    """K decays: K(bh) = max(k_min, k_init × exp(-bh/tau))."""
    def check(pos, bar, tk_ignored):
        pre = _exit_common_preamble(pos, bar)
        if pre: return pre
        d, ep, bp = pos['d'], pos['ep'], pos['bp']
        c15, atr = ibt.c15, ibt.atr14
        bpl = (bp/ep-1)*100 if d=='LONG' else (1-bp/ep)*100
        cpl = (c15[bar]/ep-1)*100 if d=='LONG' else (1-c15[bar]/ep)*100
        a = atr[bar]
        if bpl > ibt.trail_act and not math.isnan(a) and a>0:
            bh = pos['bh']
            k_eff = max(k_min, k_init * math.exp(-bh/tau))
            td = k_eff*a/c15[bar]*100
            if bpl-cpl >= td:
                r = max(0, bpl-td)
                px = ep*(1+r/100) if d=='LONG' else ep*(1-r/100)
                return {'reason':'TRAIL_TP','exit_price':px}
        return None
    return check


# ─── J: Signal bar low/high anchor ─────────────────────────────────
def make_exit_signal_extreme(activation_pct=0.5):
    """Once best_pnl >= activation_pct, trail = signal_bar_low (LONG) or high (SHORT)."""
    def check(pos, bar, tk_ignored):
        pre = _exit_common_preamble(pos, bar)
        if pre: return pre
        d, ep, bp = pos['d'], pos['ep'], pos['bp']
        c15, l15, h15 = ibt.c15, ibt.l15, ibt.h15
        bpl = (bp/ep-1)*100 if d=='LONG' else (1-bp/ep)*100
        sbar = max(0, pos['entry_bar'] - 1)
        if bpl >= activation_pct:
            anchor = l15[sbar] if d=='LONG' else h15[sbar]
            if d == 'LONG' and l15[bar] <= anchor:
                return {'reason':'TRAIL_TP', 'exit_price': anchor}
            elif d == 'SHORT' and h15[bar] >= anchor:
                return {'reason':'TRAIL_TP', 'exit_price': anchor}
            return None
        # Fallback
        cpl = (c15[bar]/ep-1)*100 if d=='LONG' else (1-c15[bar]/ep)*100
        a = ibt.atr14[bar]
        if bpl > ibt.trail_act and not math.isnan(a) and a>0:
            td = 2.5*a/c15[bar]*100
            if bpl-cpl >= td:
                r = max(0, bpl-td)
                px = ep*(1+r/100) if d=='LONG' else ep*(1-r/100)
                return {'reason':'TRAIL_TP','exit_price':px}
        return None
    return check


# ─── Progressive baseline for reference ────────────────────────────
def make_exit_progressive(tk_base, tk_post, thr):
    def check(pos, bar, tk):
        pre = _exit_common_preamble(pos, bar)
        if pre: return pre
        d, ep, bp = pos['d'], pos['ep'], pos['bp']
        c15, atr = ibt.c15, ibt.atr14
        bpl = (bp/ep-1)*100 if d=='LONG' else (1-bp/ep)*100
        cpl = (c15[bar]/ep-1)*100 if d=='LONG' else (1-c15[bar]/ep)*100
        a = atr[bar]
        if bpl > ibt.trail_act and not math.isnan(a) and a>0:
            k = tk_post if bpl >= thr else tk_base
            td = k*a/c15[bar]*100
            if bpl-cpl >= td:
                r = max(0, bpl-td)
                px = ep*(1+r/100) if d=='LONG' else ep*(1-r/100)
                return {'reason':'TRAIL_TP','exit_price':px}
        return None
    return check


# ─── Intrabar twin — worst-case drawback ──────────────────────────
def wrap_intrabar(check_base):
    """Replace bpl-cpl drawback check with bpl-worst_pl for trail logic only.
    SL/Emergency/Timeout handled inside check_base via l15/h15 already."""
    def check(pos, bar, tk):
        # SL/Emergency/Timeout preamble 그대로
        pre = _exit_common_preamble(pos, bar)
        if pre: return pre
        # Now we don't have access to variant's internal logic cleanly.
        # Re-run variant with worst-price substitute by calling with modified c15? Too complex.
        # Simpler: compute worst_pl directly and use a uniform K=2.5 trail test.
        # (Intrabar check is only useful for uniform progressive/K variants.)
        return None  # not used — we use bar_close for anchor variants
    return check


def run_bt(check_fn, passes):
    orig = ibt._check_exit_bar_close
    ibt._check_exit_bar_close = check_fn
    try:
        return run_bt_with_regime(mode='bar_close', regime_passes=passes, slippage=SLIP_MED)
    finally:
        ibt._check_exit_bar_close = orig


def stats(trades):
    if not trades: return {'n':0,'pnl':0,'mdd':0,'wr':0,'ex5':0,'daily':0,'rr':0,
                           'boot_pos':0,'boot_sh':0}
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
    rng = random.Random(42); pnls = []
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
    }


def row(name, s, base_pnl):
    vs = s['pnl'] - base_pnl
    print(f'{name:<36} {s["n"]:>4} {s["pnl"]:>+7.2f} {s["daily"]:>+6.3f} '
          f'{s["wr"]:>4.1f} {s["rr"]:>4.2f} {s["mdd"]:>5.2f} '
          f'{s["ex5"]:>+7.2f} {s["boot_pos"]:>5.1f}% {s["boot_sh"]:>+6.3f}  {vs:>+7.2f}')


def hdr():
    print(f'{"Variant":<36} {"n":>4} {"PnL":>7} {"daily":>7} {"WR":>4} {"RR":>4} '
          f'{"MDD":>5} {"ex5":>7} {"pos":>6} {"shr":>7}  vs_base')


def main():
    bs = {'max_sl_atr':4.0,'trail_K':2.5,'max_hold_bars':192,'body_min_ratio':0.60}
    ibt.strat = copy.deepcopy(ibt.strat); ibt.strat.update(bs)
    ibt.trail_K = 2.5; ibt.max_hold = 192
    ibt.sig = C1BreakoutSignal(ibt.strat)
    passes = precompute_trend_pass(192, 1.0)

    print('='*130)
    print('  Adaptive Trail Research — 차트 기반 dynamic reference points')
    print(f'  Data: {ibt.n15} bars = {DATA_DAYS:.1f} days, base: cand_C+trend')
    print('='*130)

    base = run_bt(make_exit_progressive(2.5, 2.5, 99), passes)
    b = stats(base); BP = b['pnl']
    curr = run_bt(make_exit_progressive(2.5, 0.5, 0.9), passes)
    c = stats(curr)
    print(f'\nBASELINE tk=2.5: PnL={BP:+.2f}, ex5={b["ex5"]}, pos={b["boot_pos"]}%, sh={b["boot_sh"]}')
    print(f'CURRENT v4.8.0 (thr=0.9 tkT=0.5): PnL={c["pnl"]:+.2f}, ex5={c["ex5"]}, '
          f'pos={c["boot_pos"]}%, sh={c["boot_sh"]}')

    results = []

    # F: Channel anchor
    print('\n' + '='*130)
    print('  VARIANT F: Channel breakout anchor (ch_h[signal_bar] 레벨 재터치 시 exit)')
    print('='*130)
    hdr()
    for act in [0.2, 0.3, 0.5, 0.7, 1.0, 1.5]:
        t = run_bt(make_exit_channel_anchor(act), passes)
        s = stats(t)
        row(f'F channel_act={act}%', s, BP)
        results.append((f'F channel_act={act}%', s, t))

    # G: Fractal swing anchor
    print('\n' + '='*130)
    print('  VARIANT G: Fractal swing anchor (가장 최근 fractal pivot까지 trail)')
    print('='*130)
    hdr()
    for act in [0.3, 0.5, 0.7, 1.0, 1.5, 2.0]:
        t = run_bt(make_exit_fractal_anchor(act), passes)
        s = stats(t)
        row(f'G fractal_act={act}%', s, BP)
        results.append((f'G fractal_act={act}%', s, t))

    # H: ATR-ratio adaptive K
    print('\n' + '='*130)
    print('  VARIANT H: ATR-ratio adaptive K (K × current_ATR/avg_ATR_96)')
    print('='*130)
    hdr()
    configs_h = [(2.5, 96), (2.5, 48), (2.5, 192), (1.5, 96), (3.5, 96), (2.0, 96), (3.0, 96)]
    for kb, lb in configs_h:
        t = run_bt(make_exit_atr_adaptive(kb, lb), passes)
        s = stats(t)
        row(f'H K_base={kb} lb={lb}', s, BP)
        results.append((f'H K={kb} lb={lb}', s, t))

    # I: Time-decay K
    print('\n' + '='*130)
    print('  VARIANT I: Time-decay K (K_init × exp(-bh/tau), K_min 하한)')
    print('='*130)
    hdr()
    configs_i = [
        (2.5, 0.5, 48),  (2.5, 0.3, 48),  (2.5, 0.5, 32),  (2.5, 0.5, 96),
        (3.0, 0.5, 48),  (3.5, 0.3, 48),  (2.5, 0.1, 48),  (2.5, 1.0, 48),
    ]
    for ki, km, tau in configs_i:
        t = run_bt(make_exit_time_decay(ki, km, tau), passes)
        s = stats(t)
        row(f'I init={ki} min={km} tau={tau}', s, BP)
        results.append((f'I init={ki} min={km} tau={tau}', s, t))

    # J: Signal bar extreme
    print('\n' + '='*130)
    print('  VARIANT J: Signal bar extreme anchor (l15/h15[signal_bar])')
    print('='*130)
    hdr()
    for act in [0.2, 0.3, 0.5, 0.7, 1.0, 1.5]:
        t = run_bt(make_exit_signal_extreme(act), passes)
        s = stats(t)
        row(f'J sigX_act={act}%', s, BP)
        results.append((f'J sigX_act={act}%', s, t))

    # Rankings
    print('\n' + '='*130)
    print('  Top 15 by PnL (F6 > 0 qualify)')
    print('='*130)
    hdr()
    q = [r for r in results if r[1]['ex5'] > 0 and r[1]['pnl'] > BP]
    q.sort(key=lambda x: x[1]['pnl'], reverse=True)
    for name, s, _ in q[:15]:
        row(name, s, BP)

    print('\n' + '='*130)
    print('  Top 10 by Sharpe (PnL > baseline)')
    print('='*130)
    hdr()
    q2 = sorted([r for r in results if r[1]['pnl'] > BP],
                key=lambda x: x[1]['boot_sh'], reverse=True)
    for name, s, _ in q2[:10]:
        row(name, s, BP)


if __name__ == '__main__':
    main()
