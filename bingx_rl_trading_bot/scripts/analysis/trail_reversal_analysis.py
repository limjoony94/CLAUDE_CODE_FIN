#!/usr/bin/env python3
"""Trail Reversal Analysis — "손익분기 직전 exit" 패턴 심층 탐구.

User 관찰: LIVE trade가 계속 profit 확보 직전에 반전되어 net 근처 exit.
검증 목표:
  1. BT의 trade PnL 분포는 어떠한가? (breakeven cluster 존재?)
  2. best_pnl vs exit_pnl 격차 분포 (reversal gap)
  3. v4.8.0 progressive에서 threshold 0.9% 돌파 trade 비율
  4. Trail activation 이후 시간 경과 통계
  5. Baseline vs v4.8.0 "breakeven cluster" 차이
  6. LIVE 관찰 샘플과 BT 분포 비교
"""
import sys, copy, math, json
from pathlib import Path
from statistics import mean, median, stdev

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
import os; os.chdir(ROOT)

import scripts.analysis.intrabar_trail_impact as ibt
from scripts.production.c1_breakout.signals import C1BreakoutSignal
from scripts.analysis.c1_intrabar_parity import SLIPPAGE as SLIP_MED
from scripts.analysis.regime_filter_lowvol_study import run_bt_with_regime
from scripts.analysis.regime_filter_trend_study import precompute_trend_pass


# ─── Instrumented exit functions ─────────────────────────────────
def make_exit_instrumented(tb, tp, thr, trade_log):
    """Exit function that records per-trade diagnostics into trade_log."""
    def ck(pos, bar, tk):
        d, ep, bp, sl = pos['d'], pos['ep'], pos['bp'], pos['sl']
        c15, l15, h15, atr = ibt.c15, ibt.l15, ibt.h15, ibt.atr14

        # Track: per-trade best_pnl history
        bpl = (bp/ep-1)*100 if d=='LONG' else (1-bp/ep)*100
        cpl = (c15[bar]/ep-1)*100 if d=='LONG' else (1-c15[bar]/ep)*100

        # SL
        if d == 'LONG' and l15[bar] <= sl:
            pos['_reason'] = 'SL'; pos['_best_pnl'] = bpl
            return {'reason':'SL','exit_price':sl}
        elif d == 'SHORT' and h15[bar] >= sl:
            pos['_reason'] = 'SL'; pos['_best_pnl'] = bpl
            return {'reason':'SL','exit_price':sl}
        # Emergency
        worst = (l15[bar]/ep-1)*100 if d=='LONG' else (1-h15[bar]/ep)*100
        if worst <= -ibt.emergency_sl:
            pos['_reason'] = 'EMERGENCY'; pos['_best_pnl'] = bpl
            px = ep*(1-ibt.emergency_sl/100) if d=='LONG' else ep*(1+ibt.emergency_sl/100)
            return {'reason':'EMERGENCY','exit_price':px}
        # Timeout
        if pos['bh'] >= ibt.max_hold:
            pos['_reason'] = 'TIMEOUT'; pos['_best_pnl'] = bpl
            return {'reason':'TIMEOUT','exit_price':c15[bar]}
        # Trail
        a = atr[bar]
        if bpl > ibt.trail_act and not math.isnan(a) and a>0:
            # Track first threshold crossing
            if bpl >= thr and not pos.get('_crossed_thr', False):
                pos['_crossed_thr'] = True
                pos['_thr_bar'] = bar
                pos['_thr_bh'] = pos['bh']
            k = tp if bpl >= thr else tb
            td = k*a/c15[bar]*100
            if bpl - cpl >= td:
                r = max(0, bpl - td)
                pos['_reason'] = 'TRAIL_TP'
                pos['_best_pnl'] = bpl
                pos['_realized'] = r
                pos['_trail_k'] = k
                return {'reason':'TRAIL_TP',
                        'exit_price': ep*(1+r/100) if d=='LONG' else ep*(1-r/100)}
        return None
    return ck


def run_bt_instrumented(tb, tp, thr, passes):
    """Run BT and extract best_pnl/crossed_thr per trade via monkey-patch of run loop."""
    # Approach: re-run ibt-like loop manually to retain pos state.
    # Simpler: monkey-patch exit, but also instrument run_bt_with_regime.
    # For simplicity, we call run_bt_with_regime then for each trade compute best_pnl
    # from the price series using trade[entry_bar:exit_bar].
    orig = ibt._check_exit_bar_close
    ibt._check_exit_bar_close = make_exit_instrumented(tb, tp, thr, [])
    try:
        trades = run_bt_with_regime(mode='bar_close', regime_passes=passes, slippage=SLIP_MED)
    finally:
        ibt._check_exit_bar_close = orig
    # Compute best_pnl_observed for each trade from price series
    for t in trades:
        eb = t['entry_bar']; xb = t['exit_bar']
        ep = t['entry_price']
        if t['d'] == 'LONG':
            hi = max(ibt.h15[eb:xb+1]) if xb >= eb else ep
            best_pnl = (hi/ep - 1) * 100
        else:
            lo = min(ibt.l15[eb:xb+1]) if xb >= eb else ep
            best_pnl = (1 - lo/ep) * 100
        t['best_pnl_observed'] = best_pnl
        t['crossed_thr'] = best_pnl >= thr
        t['reversal_gap'] = best_pnl - (t['raw'] if 'raw' in t else 0)  # raw = pre-fee-slip PnL
    return trades


def distribution_table(trades, label):
    """Categorize PnL and print distribution."""
    cats = {
        'big_loss (<-2%)':    lambda n: n < -2,
        'med_loss (-2,-1%)':  lambda n: -2 <= n < -1,
        'small_loss (-1,-0.3%)': lambda n: -1 <= n < -0.3,
        'breakeven (-0.3,+0.3%)': lambda n: -0.3 <= n <= 0.3,
        'small_win (+0.3,+1%)': lambda n: 0.3 < n <= 1,
        'med_win (+1,+2%)':   lambda n: 1 < n <= 2,
        'big_win (+2,+5%)':   lambda n: 2 < n <= 5,
        'xl_win (>+5%)':      lambda n: n > 5,
    }
    print(f'\n  [{label}] PnL distribution (net after fee+slip):')
    tot = len(trades)
    total_pnl = sum(t['net'] for t in trades)
    for name, pred in cats.items():
        matches = [t for t in trades if pred(t['net'])]
        pct = len(matches)/tot*100 if tot else 0
        cat_pnl = sum(t['net'] for t in matches)
        cat_share = cat_pnl/total_pnl*100 if total_pnl else 0
        print(f'    {name:<28}: {len(matches):>4} ({pct:>5.1f}%), sum PnL: {cat_pnl:>+7.2f} ({cat_share:>+5.1f}% of total)')
    print(f'    {"TOTAL":<28}: {tot:>4}          sum PnL: {total_pnl:>+7.2f}')


def reversal_gap_analysis(trades, label):
    """Analyze how far price went into profit (best_pnl) vs exit PnL."""
    tp_trades = [t for t in trades if t.get('reason') == 'TRAIL_TP']
    if not tp_trades:
        print(f'\n  [{label}] No TRAIL_TP trades')
        return
    best_pnls = [t['best_pnl_observed'] for t in tp_trades]
    nets = [t['net'] for t in tp_trades]
    gaps = [t['best_pnl_observed'] - t['net'] for t in tp_trades]

    print(f'\n  [{label}] TRAIL_TP reversal gap analysis (n={len(tp_trades)}):')
    print(f'    best_pnl observed:  mean={mean(best_pnls):+.3f}%, median={median(best_pnls):+.3f}%, max={max(best_pnls):+.3f}%')
    print(f'    net exit PnL:       mean={mean(nets):+.3f}%, median={median(nets):+.3f}%')
    print(f'    reversal gap:       mean={mean(gaps):+.3f}%, median={median(gaps):+.3f}%')
    # Distribution of best_pnl_observed
    bins = [0.0, 0.3, 0.5, 0.7, 0.9, 1.2, 1.5, 2.0, 3.0, 5.0, 99.0]
    print(f'\n    best_pnl distribution (max profit reached):')
    for i in range(len(bins)-1):
        lo, hi = bins[i], bins[i+1]
        count = sum(1 for b in best_pnls if lo <= b < hi)
        avg_net_bin = mean([n for b,n in zip(best_pnls,nets) if lo<=b<hi]) if count else 0
        print(f'      [{lo:>4.1f}, {hi:>4.1f}]%: {count:>4} trades ({count/len(best_pnls)*100:>5.1f}%), '
              f'avg exit net={avg_net_bin:+.3f}%')


def threshold_crossing_analysis(trades, thr, label):
    """For progressive: how many trades reach threshold?"""
    tp_trades = [t for t in trades if t.get('reason') == 'TRAIL_TP']
    if not tp_trades: return
    crossed = [t for t in tp_trades if t.get('crossed_thr', False)]
    not_crossed = [t for t in tp_trades if not t.get('crossed_thr', False)]
    print(f'\n  [{label}] Threshold {thr}% crossing analysis:')
    print(f'    Crossed (progressive tight active): {len(crossed)} ({len(crossed)/len(tp_trades)*100:.1f}%)')
    print(f'    Not crossed (baseline K=2.5):       {len(not_crossed)} ({len(not_crossed)/len(tp_trades)*100:.1f}%)')
    if crossed:
        print(f'    crossed mean net: {mean(t["net"] for t in crossed):+.3f}%')
    if not_crossed:
        print(f'    not-crossed mean net: {mean(t["net"] for t in not_crossed):+.3f}%')


def compare_live_bt(bt_trades, live_trades_raw):
    """Compare BT distribution with LIVE trades (1x equivalent)."""
    # LIVE pnl_pct is 3x leveraged → divide by 3
    live_nets = [t['pnl_pct']/3 for t in live_trades_raw if 'pnl_pct' in t]
    print(f'\n  [LIVE vs BT distribution comparison]')
    print(f'    LIVE n={len(live_nets)}, BT n={len(bt_trades)}')
    if not live_nets: return
    print(f'    LIVE mean net: {mean(live_nets):+.3f}%, BT mean: {mean(t["net"] for t in bt_trades):+.3f}%')
    cats = [
        ('big_loss(<-2%)', lambda n: n < -2),
        ('breakeven(-0.3,+0.3)', lambda n: -0.3 <= n <= 0.3),
        ('big_win(>+2%)', lambda n: n > 2),
    ]
    print(f'    {"Category":<22} {"LIVE %":>10} {"BT %":>10}')
    for name, pred in cats:
        lp = sum(1 for n in live_nets if pred(n))/len(live_nets)*100
        bp = sum(1 for t in bt_trades if pred(t['net']))/len(bt_trades)*100
        print(f'    {name:<22} {lp:>9.1f}% {bp:>9.1f}%')


def main():
    bs = {'max_sl_atr':4.0, 'trail_K':2.5, 'max_hold_bars':192, 'body_min_ratio':0.60}
    ibt.strat = copy.deepcopy(ibt.strat); ibt.strat.update(bs)
    ibt.trail_K = 2.5; ibt.max_hold = 192
    ibt.sig = C1BreakoutSignal(ibt.strat)
    passes = precompute_trend_pass(192, 1.0)

    print('='*110)
    print('  Trail Reversal Analysis — "손익분기 직전 exit" 패턴 탐구')
    print(f'  Data: {ibt.n15} bars = {ibt.n15/96:.1f} days')
    print('='*110)

    # ═══ Baseline ═══
    print('\n\n━━━ BASELINE (tk=2.5 fixed) ━━━')
    base_trades = run_bt_instrumented(2.5, 2.5, 99.0, passes)
    distribution_table(base_trades, 'baseline')
    reversal_gap_analysis(base_trades, 'baseline')
    threshold_crossing_analysis(base_trades, 0.9, 'baseline')  # just for info

    # ═══ v4.8.0 Progressive ═══
    print('\n\n━━━ v4.8.0 PROGRESSIVE (thr=0.9, tkT=0.5) ━━━')
    v480_trades = run_bt_instrumented(2.5, 0.5, 0.9, passes)
    distribution_table(v480_trades, 'v4.8.0')
    reversal_gap_analysis(v480_trades, 'v4.8.0')
    threshold_crossing_analysis(v480_trades, 0.9, 'v4.8.0')

    # ═══ LIVE comparison ═══
    print('\n\n━━━ LIVE (state.json) ━━━')
    state_path = ROOT / 'bingx_rl_trading_bot' / 'results' / 'c1_breakout_state.json'
    if not state_path.exists():
        state_path = ROOT / 'results' / 'c1_breakout_state.json'
    try:
        state = json.loads(state_path.read_text())
        live = state.get('trade_history', [])
        print(f'\n  LIVE trade count: {len(live)}')
        if live:
            live_nets_1x = [t['pnl_pct']/3 for t in live]  # 3x → 1x equiv
            print(f'  LIVE mean net (1x): {mean(live_nets_1x):+.3f}%')
            print(f'  LIVE trade reasons:')
            reasons = {}
            for t in live:
                r = t.get('reason','?')
                reasons.setdefault(r, []).append(t['pnl_pct']/3)
            for r, nets in sorted(reasons.items()):
                print(f'    {r:<22}: n={len(nets)}, mean={mean(nets):+.3f}%')
        # Compare distribution
        compare_live_bt(v480_trades, live)
    except Exception as e:
        print(f'  Failed to parse state: {e}')


if __name__ == '__main__':
    main()
