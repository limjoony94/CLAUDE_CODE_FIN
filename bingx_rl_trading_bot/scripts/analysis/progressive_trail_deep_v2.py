#!/usr/bin/env python3
"""Progressive Trail DEEP Overfit Stress Tests V2 — T2/T6 corrected + slip calibration.

Fixes from V1:
  T2 redone: progressive-aware 5m and intrabar exit functions (monkey-patch all 3 modes)
  T6 redone: c1_intrabar_parity.FEE directly overridden
  Slippage calibration: parse LIVE state.json + log for actual slip observations
"""
import sys, copy, math, random, json, re
from pathlib import Path
from statistics import mean, stdev, median

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
import os; os.chdir(ROOT)

import scripts.analysis.intrabar_trail_impact as ibt
import scripts.analysis.c1_intrabar_parity as cip  # for FEE override
from scripts.production.c1_breakout.signals import C1BreakoutSignal
from scripts.analysis.c1_intrabar_parity import SLIPPAGE as SLIP_MED
from scripts.analysis.regime_filter_lowvol_study import run_bt_with_regime
from scripts.analysis.regime_filter_trend_study import precompute_trend_pass

DATA_DAYS = ibt.n15 / 96


# ─── Progressive exit functions (bar_close / intrabar / 5m) ─────
def make_exit_bc(tb, tp, thr):
    def ck(pos, bar, tk):
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
            k = tp if bpl >= thr else tb
            td = k*a/c15[bar]*100
            if bpl-cpl >= td:
                r = max(0, bpl-td)
                return {'reason':'TRAIL_TP',
                        'exit_price': ep*(1+r/100) if d=='LONG' else ep*(1-r/100)}
        return None
    return ck


def make_exit_intrabar(tb, tp, thr):
    """Progressive-aware intrabar check (uses bar low/high for worst-case drawdown)."""
    def ck(pos, bar, tk):
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
        # INTRABAR: worst price within bar for drawdown
        worst_pl = (l15[bar]/ep-1)*100 if d=='LONG' else (1-h15[bar]/ep)*100
        a = atr[bar]
        if bpl > ibt.trail_act and not math.isnan(a) and a>0:
            k = tp if bpl >= thr else tb
            td = k*a/c15[bar]*100
            if bpl - worst_pl >= td:
                r = max(0, bpl-td)
                return {'reason':'TRAIL_TP',
                        'exit_price': ep*(1+r/100) if d=='LONG' else ep*(1-r/100)}
        return None
    return ck


def make_exit_5m(tb, tp, thr):
    """Progressive-aware 5m sub-bar check. Updates bp through sub-bars, switches K by bpl."""
    def ck(pos, bar15, tk):
        d, ep, sl = pos['d'], pos['ep'], pos['sl']
        atr = ibt.atr14[bar15]
        start_5m = bar15 * 3
        end_5m = min(start_5m + 3, ibt.n5)
        c5, h5, l5 = ibt.c5, ibt.h5, ibt.l5
        for i5 in range(start_5m, end_5m):
            if d == 'LONG': pos['bp'] = max(pos['bp'], h5[i5])
            else: pos['bp'] = min(pos['bp'], l5[i5])
            bp = pos['bp']
            # SL
            if d == 'LONG' and l5[i5] <= sl: return {'reason':'SL','exit_price':sl}
            elif d == 'SHORT' and h5[i5] >= sl: return {'reason':'SL','exit_price':sl}
            # Emergency
            worst = (l5[i5]/ep-1)*100 if d=='LONG' else (1-h5[i5]/ep)*100
            if worst <= -ibt.emergency_sl:
                px = ep*(1-ibt.emergency_sl/100) if d=='LONG' else ep*(1+ibt.emergency_sl/100)
                return {'reason':'EMERGENCY','exit_price':px}
            # Timeout
            if pos['bh'] >= ibt.max_hold: return {'reason':'TIMEOUT','exit_price':c5[i5]}
            # Progressive trail
            bpl = (bp/ep-1)*100 if d=='LONG' else (1-bp/ep)*100
            cpl = (c5[i5]/ep-1)*100 if d=='LONG' else (1-c5[i5]/ep)*100
            if bpl > ibt.trail_act and not math.isnan(atr) and atr>0:
                k = tp if bpl >= thr else tb
                td = k*atr/c5[i5]*100
                if bpl-cpl >= td:
                    r = max(0, bpl-td)
                    return {'reason':'TRAIL_TP',
                            'exit_price': ep*(1+r/100) if d=='LONG' else ep*(1-r/100)}
        return None
    return ck


def run_bt(mode, tb, tp, thr, passes, slippage=SLIP_MED):
    """Full monkey-patch for the chosen mode."""
    orig_bc = ibt._check_exit_bar_close
    orig_ib = ibt._check_exit_intrabar
    orig_5m = ibt._check_exit_5m
    ibt._check_exit_bar_close = make_exit_bc(tb, tp, thr)
    ibt._check_exit_intrabar = make_exit_intrabar(tb, tp, thr)
    ibt._check_exit_5m = make_exit_5m(tb, tp, thr)
    try:
        return run_bt_with_regime(mode=mode, regime_passes=passes, slippage=slippage)
    finally:
        ibt._check_exit_bar_close = orig_bc
        ibt._check_exit_intrabar = orig_ib
        ibt._check_exit_5m = orig_5m


def stats(trades):
    if not trades: return {'n':0,'pnl':0,'mdd':0,'wr':0}
    total = sum(t['net'] for t in trades)
    eq=0; pk=0; md=0
    for t in trades:
        eq+=t['net']; pk=max(pk,eq); md=max(md,pk-eq)
    w = sum(1 for t in trades if t['net']>0)
    return {'n': len(trades), 'pnl': round(total,2), 'mdd': round(md,2),
            'wr': round(w/len(trades)*100,1)}


# ═══ T2 CORRECTED: progressive 5m and intrabar ═══
def test_t2_corrected(passes):
    print('\n' + '='*110)
    print('  T2-C: CORRECTED — progressive-aware bar_close / 5m / intrabar')
    print('='*110)
    print(f'  {"Config":<22} {"Mode":<10} {"n":>5} {"PnL":>9} {"MDD":>6} {"WR":>5}')
    # Track gaps for analysis
    results = {}
    for name, tb, tp, thr in [('baseline (tk=2.5)', 2.5, 2.5, 99),
                              ('v4.8.0 (thr=0.9 tkT=0.5)', 2.5, 0.5, 0.9),
                              ('tkT=0.3', 2.5, 0.3, 0.9),
                              ('tkT=0.1', 2.5, 0.1, 0.9)]:
        for mode in ['bar_close', '5m', 'intrabar']:
            t = run_bt(mode, tb, tp, thr, passes)
            s = stats(t)
            results[(name, mode)] = s
            print(f'  {name:<22} {mode:<10} {s["n"]:>5} {s["pnl"]:>+8.2f} {s["mdd"]:>+5.2f} {s["wr"]:>4.1f}%')
        # Gap summary
        bc = results[(name, 'bar_close')]['pnl']
        m5 = results[(name, '5m')]['pnl']
        ib = results[(name, 'intrabar')]['pnl']
        print(f'  {name:<22} Δ_5m vs bar_close: {m5-bc:+.2f} ({(m5-bc)/abs(bc)*100 if bc else 0:+.1f}%)  '
              f'Δ_intra: {ib-bc:+.2f}')
        print()
    return results


# ═══ T6 CORRECTED: FEE override ═══
def test_t6_corrected(passes):
    print('\n' + '='*110)
    print('  T6-C: CORRECTED Fee sensitivity (cip.FEE properly overridden)')
    print('='*110)
    orig_ibt_fee = ibt.FEE
    orig_cip_fee = cip.FEE
    print(f'  {"FEE%":>6} {"base PnL":>10} {"v4.8.0 PnL":>12} {"Δ":>8} {"base/trade":>12}')
    for fee in [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]:
        ibt.FEE = fee
        cip.FEE = fee
        t_b = run_bt('bar_close', 2.5, 2.5, 99, passes)
        t_v = run_bt('bar_close', 2.5, 0.5, 0.9, passes)
        sb = sum(t['net'] for t in t_b)
        sv = sum(t['net'] for t in t_v)
        per_trade_b = sb / max(len(t_b), 1)
        print(f'  {fee:>5.2f}% {sb:>+9.2f} {sv:>+11.2f} {sv-sb:>+7.2f} {per_trade_b:>+11.3f}')
    ibt.FEE = orig_ibt_fee
    cip.FEE = orig_cip_fee


# ═══ Slippage calibration from LIVE data ═══
def calibrate_live_slippage():
    print('\n' + '='*110)
    print('  LIVE Slippage Calibration — parse state.json + log')
    print('='*110)
    state_path = ROOT / 'bingx_rl_trading_bot' / 'results' / 'c1_breakout_state.json'
    if not state_path.exists():
        state_path = ROOT / 'results' / 'c1_breakout_state.json'
    log_path = ROOT / 'bingx_rl_trading_bot' / 'logs' / 'c1_breakout.log'
    if not log_path.exists():
        log_path = ROOT / 'logs' / 'c1_breakout.log'

    observed_slips = []
    # From state.json — exit_slippage_pct field (BUG#65 tracked)
    if state_path.exists():
        try:
            s = json.loads(state_path.read_text())
            for t in s.get('trade_history', []):
                if 'exit_slippage_pct' in t:
                    observed_slips.append(('exit', abs(t['exit_slippage_pct']), t.get('reason','')))
        except Exception as e:
            print(f'  state parse err: {e}')

    # From log — "Slippage: +X.XXX%" entry slippages
    entry_slips = []
    if log_path.exists():
        try:
            for line in log_path.read_text(encoding='utf-8', errors='ignore').splitlines():
                m = re.search(r'Slippage:\s*([+\-]?[\d.]+)%', line)
                if m:
                    entry_slips.append(abs(float(m.group(1))))
        except Exception as e:
            print(f'  log parse err: {e}')

    print(f'  Entry slippage samples (log): n={len(entry_slips)}')
    if entry_slips:
        print(f'    mean={mean(entry_slips):.3f}%  median={median(entry_slips):.3f}%  max={max(entry_slips):.3f}%')
        entry_mult = mean(entry_slips) / 0.05  # vs SLIP_MED 'entry_pct'
        print(f'    vs SLIP_MED entry(0.05%): mean ratio = ×{entry_mult:.2f}')

    print(f'  Exit slippage samples (state): n={len(observed_slips)}')
    if observed_slips:
        exit_vals = [v for _, v, _ in observed_slips]
        print(f'    mean={mean(exit_vals):.3f}%  median={median(exit_vals):.3f}%  max={max(exit_vals):.3f}%')
        # Compare by reason
        by_reason = {}
        for _, v, r in observed_slips:
            by_reason.setdefault(r, []).append(v)
        for r, vs in by_reason.items():
            print(f'    {r}: n={len(vs)}, mean={mean(vs):.3f}%')

    # Infer effective slip multiplier
    if entry_slips:
        return mean(entry_slips) / 0.05
    return None


def main():
    bs = {'max_sl_atr':4.0, 'trail_K':2.5, 'max_hold_bars':192, 'body_min_ratio':0.60}
    ibt.strat = copy.deepcopy(ibt.strat); ibt.strat.update(bs)
    ibt.trail_K = 2.5; ibt.max_hold = 192
    ibt.sig = C1BreakoutSignal(ibt.strat)
    passes = precompute_trend_pass(192, 1.0)

    print('='*110)
    print('  DEEP OVERFIT V2 — T2/T6 corrected + LIVE slip calibration')
    print(f'  Data: {ibt.n15} bars = {DATA_DAYS:.1f} days')
    print('='*110)

    results_t2 = test_t2_corrected(passes)
    test_t6_corrected(passes)
    live_mult = calibrate_live_slippage()

    # Summary
    print('\n' + '='*110)
    print('  종합 평가')
    print('='*110)
    print('\n  [T2] LIVE realism gap (bar_close → 5m → intrabar):')
    for cfg in ['baseline (tk=2.5)', 'v4.8.0 (thr=0.9 tkT=0.5)', 'tkT=0.3', 'tkT=0.1']:
        bc = results_t2[(cfg,'bar_close')]['pnl']
        m5 = results_t2[(cfg,'5m')]['pnl']
        ib = results_t2[(cfg,'intrabar')]['pnl']
        if bc != 0:
            print(f'    {cfg:<25}: bar_close {bc:+.2f} → 5m {m5:+.2f} ({(m5-bc)/abs(bc)*100:+.1f}%) '
                  f'→ intrabar {ib:+.2f} ({(ib-bc)/abs(bc)*100:+.1f}%)')
        else:
            print(f'    {cfg:<25}: bar_close {bc:+.2f} → 5m {m5:+.2f} → intrabar {ib:+.2f}')

    if live_mult is not None:
        print(f'\n  [LIVE SLIP] Observed entry slip mean ratio ×{live_mult:.2f} vs SLIP_MED')
        if live_mult > 3.0:
            print('    ⚠ LIVE slip HIGH — progressive edge may be eroded')
        elif live_mult > 1.5:
            print('    ⚠ LIVE slip moderately elevated — edge reduced but positive')
        else:
            print('    ✅ LIVE slip within SLIP_MED range — edge intact')


if __name__ == '__main__':
    main()
