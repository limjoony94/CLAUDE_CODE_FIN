#!/usr/bin/env python3
"""Progressive Trail DEEP Overfit Stress Tests.

8 tests designed specifically to challenge the v4.8.0 (thr=0.9, tkT=0.5) result:
  T1. True temporal train-test (no peek)
  T2. 5m resolution BT
  T3. Slippage stress (x1~x5)
  T4. Regime decomposition (vol/trend breakdown)
  T5. Block-shuffle Monte Carlo
  T6. Cost sensitivity (fee 0.05~0.30%)
  T7. Random trade subsampling
  T8. Body filter baseline robustness

Goal: Determine if progressive_trail is genuinely robust or overfit
      to specific dataset/slippage/fee/structure assumptions.
"""
import sys, copy, math, random
from pathlib import Path
from statistics import mean, stdev, median

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
import os; os.chdir(ROOT)

import scripts.analysis.intrabar_trail_impact as ibt
from scripts.production.c1_breakout.signals import C1BreakoutSignal
from scripts.analysis.c1_intrabar_parity import SLIPPAGE as SLIP_MED, apply_slippage
from scripts.analysis.regime_filter_lowvol_study import run_bt_with_regime
from scripts.analysis.regime_filter_trend_study import precompute_trend_pass

DATA_DAYS = ibt.n15 / 96


def make_exit(tk_base, tk_post, thr):
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
            if bpl-cpl >= td:
                r = max(0, bpl-td)
                px = ep*(1+r/100) if d=='LONG' else ep*(1-r/100)
                return {'reason':'TRAIL_TP','exit_price':px}
        return None
    return check


def run_bt(check_fn, passes, slippage=SLIP_MED, mode='bar_close'):
    orig = ibt._check_exit_bar_close
    ibt._check_exit_bar_close = check_fn
    try:
        return run_bt_with_regime(mode=mode, regime_passes=passes, slippage=slippage)
    finally:
        ibt._check_exit_bar_close = orig


def simple_stats(trades, start_bar=None, end_bar=None):
    if start_bar is not None or end_bar is not None:
        trades = [t for t in trades
                  if (start_bar is None or t['entry_bar'] >= start_bar)
                  and (end_bar is None or t['entry_bar'] < end_bar)]
    if not trades: return {'n':0, 'pnl':0, 'mdd':0, 'wr':0}
    total = sum(t['net'] for t in trades)
    eq=0; pk=0; md=0
    for t in trades:
        eq+=t['net']; pk=max(pk,eq); md=max(md,pk-eq)
    wins = sum(1 for t in trades if t['net']>0)
    return {'n': len(trades), 'pnl': round(total,2),
            'mdd': round(md,2), 'wr': round(wins/len(trades)*100,1)}


def setup(body=0.60):
    bs = {'max_sl_atr':4.0, 'trail_K':2.5, 'max_hold_bars':192, 'body_min_ratio':body}
    ibt.strat = copy.deepcopy(ibt.strat); ibt.strat.update(bs)
    ibt.trail_K = 2.5; ibt.max_hold = 192
    ibt.sig = C1BreakoutSignal(ibt.strat)


# ═══ T1: True temporal train-test ═══
def test_temporal_split(passes):
    print('\n' + '='*110)
    print('  T1: TRUE temporal train-test — optimize on 1st half, evaluate on 2nd half')
    print('='*110)
    mid = ibt.n15 // 2
    # Grid for 1st half optimization
    thr_grid = [0.3, 0.5, 0.7, 0.9, 1.2, 1.5]
    tk_grid = [0.3, 0.5, 1.0, 1.5, 2.5]

    print('  Step 1: train on 1st half (bars 0..{})'.format(mid))
    train_results = []
    for thr in thr_grid:
        for tk in tk_grid:
            t = run_bt(make_exit(2.5, tk, thr), passes)
            s = simple_stats(t, end_bar=mid)
            train_results.append((thr, tk, s['pnl'], s['mdd']))
    train_results.sort(key=lambda x: x[2], reverse=True)
    print(f'  Top 3 on 1st half:')
    for thr, tk, pnl, mdd in train_results[:3]:
        print(f'    thr={thr} tkT={tk}: 1st half PnL={pnl:+.2f}, MDD={mdd}')
    best_thr, best_tk, _, _ = train_results[0]
    print(f'  → Selected: thr={best_thr}, tkT={best_tk}')

    # Step 2: evaluate on 2nd half with selected
    baseline_trades = run_bt(make_exit(2.5, 2.5, 99), passes)
    selected_trades = run_bt(make_exit(2.5, best_tk, best_thr), passes)
    v480_trades = run_bt(make_exit(2.5, 0.5, 0.9), passes)

    print('  Step 2: evaluate on 2nd half (bars {}..{})'.format(mid, ibt.n15))
    for name, tr in [('baseline', baseline_trades),
                     ('selected_from_1st', selected_trades),
                     ('v4.8.0 (thr=0.9 tkT=0.5)', v480_trades)]:
        s = simple_stats(tr, start_bar=mid)
        print(f'    {name:<28}: 2nd half PnL={s["pnl"]:+.2f}, MDD={s["mdd"]}, n={s["n"]}, WR={s["wr"]}%')
    return best_thr, best_tk


# ═══ T2: 5m resolution BT ═══
def test_5m_resolution(passes):
    print('\n' + '='*110)
    print('  T2: 5m resolution BT (between bar_close and intrabar)')
    print('='*110)
    # 5m mode uses intrabar_trail_impact._check_exit_5m which handles 5m sub-bars
    # For progressive, we need custom 5m-aware exit. Too complex — approximate with intrabar.
    # Use existing _check_exit_5m from ibt with monkey-patched trail_K
    # NB: ibt._check_exit_5m uses module-level trail_K directly, so we must tweak trail_K to
    # an "effective" value. Instead, run bar_close and intrabar for contrast.
    for name, tb, tp, thr in [('baseline', 2.5, 2.5, 99),
                              ('v4.8.0', 2.5, 0.5, 0.9)]:
        for mode in ['bar_close', '5m', 'intrabar']:
            # run_bt_with_regime passes mode to ibt._check_exit_* via orchestrator
            # But our monkey-patch only intercepts _check_exit_bar_close. For 5m/intrabar, we
            # leave native ibt functions which use static trail_K.
            # → Simpler: set ibt.trail_K accordingly before run.
            # For progressive, 5m/intrabar native don't support it, so approximate:
            ibt.trail_K = tp if thr < 50 else tb  # for v4.8.0 approximate with tp=0.5
            t = run_bt_with_regime(mode=mode, regime_passes=passes, slippage=SLIP_MED)
            s = simple_stats(t)
            print(f'  {name:<10} {mode:<10} PnL={s["pnl"]:+.2f}, n={s["n"]}, MDD={s["mdd"]}')
        print()
    ibt.trail_K = 2.5


# ═══ T3: Slippage stress ═══
def test_slippage_stress(passes):
    print('\n' + '='*110)
    print('  T3: Slippage stress test (×0.5 ~ ×5.0 of SLIP_MED)')
    print('='*110)
    print(f'  {"Mult":>5} {"base PnL":>10} {"v4.8.0 PnL":>12} {"Δ":>8} {"base MDD":>10} {"v4.8.0 MDD":>12}')
    for mult in [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0]:
        slip = {k: v*mult for k,v in SLIP_MED.items()}
        t_base = run_bt(make_exit(2.5, 2.5, 99), passes, slippage=slip)
        t_v480 = run_bt(make_exit(2.5, 0.5, 0.9), passes, slippage=slip)
        sb = simple_stats(t_base); sv = simple_stats(t_v480)
        d = sv['pnl'] - sb['pnl']
        print(f'  x{mult:<4} {sb["pnl"]:>+9.2f} {sv["pnl"]:>+11.2f} {d:>+7.2f} {sb["mdd"]:>+9.2f} {sv["mdd"]:>+11.2f}')


# ═══ T4: Regime decomposition ═══
def test_regime_decomposition(passes):
    print('\n' + '='*110)
    print('  T4: Regime decomposition — per-regime PnL breakdown')
    print('='*110)
    # Classify each entry_bar by regime (vol + trend)
    c15, atr = ibt.c15, ibt.atr14
    atr_pct = [atr[i]/c15[i]*100 if c15[i]>0 else 0 for i in range(ibt.n15)]
    valid_atr = [a for a in atr_pct if not math.isnan(a) and a > 0]
    med_atr = median(valid_atr) if valid_atr else 0.3
    # Trend: 192-bar rolling trend %
    trend = [0.0] * ibt.n15
    for i in range(192, ibt.n15):
        if c15[i-192] > 0:
            trend[i] = (c15[i]/c15[i-192] - 1) * 100
    med_trend = abs(median([abs(t) for t in trend[192:]]))

    t_base = run_bt(make_exit(2.5, 2.5, 99), passes)
    t_v480 = run_bt(make_exit(2.5, 0.5, 0.9), passes)

    regimes = [
        ('high_vol_trending', lambda i: atr_pct[i]>=med_atr and abs(trend[i])>=med_trend),
        ('high_vol_choppy',   lambda i: atr_pct[i]>=med_atr and abs(trend[i])<med_trend),
        ('low_vol_trending',  lambda i: atr_pct[i]<med_atr and abs(trend[i])>=med_trend),
        ('low_vol_choppy',    lambda i: atr_pct[i]<med_atr and abs(trend[i])<med_trend),
    ]
    print(f'  Median ATR%={med_atr:.3f}, Median |trend|%={med_trend:.3f}')
    print(f'  {"Regime":<22} {"n":>4} {"base PnL":>10} {"v4.8.0 PnL":>12} {"Δ":>8}')
    for rname, f in regimes:
        b_sum = sum(t['net'] for t in t_base if f(t['entry_bar']))
        b_n = sum(1 for t in t_base if f(t['entry_bar']))
        v_sum = sum(t['net'] for t in t_v480 if f(t['entry_bar']))
        v_n = sum(1 for t in t_v480 if f(t['entry_bar']))
        d = v_sum - b_sum
        print(f'  {rname:<22} {v_n:>4} {b_sum:>+9.2f} {v_sum:>+11.2f} {d:>+7.2f}')


# ═══ T5: Block-shuffle MC ═══
def test_block_shuffle_mc(passes):
    print('\n' + '='*110)
    print('  T5: Block-shuffle Monte Carlo (preserves autocorrelation)')
    print('='*110)
    t_base = run_bt(make_exit(2.5, 2.5, 99), passes)
    t_v480 = run_bt(make_exit(2.5, 0.5, 0.9), passes)
    real_delta = sum(t['net'] for t in t_v480) - sum(t['net'] for t in t_base)
    # Block shuffle: take nets of v4.8.0, group into blocks of 20 trades, shuffle blocks
    nets_v = [t['net'] for t in t_v480]
    nets_b = [t['net'] for t in t_base]
    rng = random.Random(42)
    N_SIMS = 999
    BLOCK = 20
    beat = 0
    for _ in range(N_SIMS):
        # Shuffle v480 blocks
        vb = [nets_v[i:i+BLOCK] for i in range(0, len(nets_v), BLOCK)]
        rng.shuffle(vb)
        v_shuf = [x for blk in vb for x in blk]
        # Same for baseline
        bb = [nets_b[i:i+BLOCK] for i in range(0, len(nets_b), BLOCK)]
        rng.shuffle(bb)
        b_shuf = [x for blk in bb for x in blk]
        # Delta after shuffle — samples random realization
        simulated_delta = sum(v_shuf) - sum(b_shuf)
        if real_delta >= simulated_delta: beat += 1
    p = 1 - beat/N_SIMS
    print(f'  Real delta v4.8.0 - baseline: {real_delta:+.2f}')
    print(f'  Beat {beat}/{N_SIMS} block-shuffled sims → p = {p:.4f}')
    print(f'  {"✅ Robust" if p < 0.05 else "⚠ Not significant"}')


# ═══ T6: Cost sensitivity ═══
def test_cost_sensitivity(passes):
    print('\n' + '='*110)
    print('  T6: Fee sensitivity (FEE 0.05 ~ 0.30%)')
    print('='*110)
    orig_fee = ibt.FEE
    print(f'  {"FEE%":>5} {"base":>9} {"v4.8.0":>9} {"Δ":>8} {"Δ/trade":>10}')
    for fee in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
        ibt.FEE = fee
        t_base = run_bt(make_exit(2.5, 2.5, 99), passes)
        t_v480 = run_bt(make_exit(2.5, 0.5, 0.9), passes)
        sb = sum(t['net'] for t in t_base)
        sv = sum(t['net'] for t in t_v480)
        d = sv - sb
        d_per = d / max(len(t_v480), 1)
        print(f'  {fee:>5.2f} {sb:>+9.2f} {sv:>+9.2f} {d:>+7.2f} {d_per:>+9.3f}')
    ibt.FEE = orig_fee


# ═══ T7: Random trade subsampling ═══
def test_random_subsampling(passes):
    print('\n' + '='*110)
    print('  T7: Random trade subsampling — edge retention at lower trade counts')
    print('='*110)
    t_base = run_bt(make_exit(2.5, 2.5, 99), passes)
    t_v480 = run_bt(make_exit(2.5, 0.5, 0.9), passes)
    rng = random.Random(42)
    print(f'  {"Frac":>5} {"base n":>7} {"base PnL":>10} {"v PnL":>10} {"Δ":>8} {"Δ%":>7}')
    for frac in [0.3, 0.5, 0.7, 0.9]:
        deltas = []
        for trial in range(30):
            rng_t = random.Random(trial*42)
            b_sub = rng_t.sample(t_base, int(len(t_base)*frac))
            v_sub = rng_t.sample(t_v480, int(len(t_v480)*frac))
            deltas.append(sum(t['net'] for t in v_sub) - sum(t['net'] for t in b_sub))
        mean_d = mean(deltas); std_d = stdev(deltas) if len(deltas)>1 else 0
        print(f'  {frac:>5.1f} {int(len(t_base)*frac):>7} — mean Δ={mean_d:+.2f} ± {std_d:.2f} '
              f'(positive: {sum(1 for d in deltas if d>0)}/30)')


# ═══ T8: Body filter baseline robustness ═══
def test_body_robustness():
    print('\n' + '='*110)
    print('  T8: Progressive robustness across body_min_ratio baselines')
    print('='*110)
    print(f'  {"body":>5} {"base PnL":>10} {"v4.8.0 PnL":>12} {"Δ":>8} {"Δ% vs base":>12}')
    for body in [0.40, 0.50, 0.60, 0.70]:
        setup(body=body)
        passes = precompute_trend_pass(192, 1.0)
        t_base = run_bt(make_exit(2.5, 2.5, 99), passes)
        t_v480 = run_bt(make_exit(2.5, 0.5, 0.9), passes)
        sb = sum(t['net'] for t in t_base)
        sv = sum(t['net'] for t in t_v480)
        d = sv - sb
        dp = d/abs(sb)*100 if sb != 0 else float('nan')
        print(f'  {body:>5.2f} {sb:>+9.2f} {sv:>+11.2f} {d:>+7.2f} {dp:>+10.1f}%')
    # Reset to 0.60
    setup(body=0.60)


def main():
    setup(body=0.60)
    passes = precompute_trend_pass(192, 1.0)
    print('='*110)
    print('  PROGRESSIVE TRAIL — DEEP OVERFIT STRESS TESTS')
    print(f'  Data: {ibt.n15} bars = {DATA_DAYS:.1f} days')
    print('='*110)

    test_temporal_split(passes)
    test_5m_resolution(passes)
    test_slippage_stress(passes)
    test_regime_decomposition(passes)
    test_block_shuffle_mc(passes)
    test_cost_sensitivity(passes)
    test_random_subsampling(passes)
    test_body_robustness()

    print('\n' + '='*110)
    print('  DEEP OVERFIT STRESS — COMPLETE')
    print('='*110)


if __name__ == '__main__':
    main()
