#!/usr/bin/env python3
"""
C1 Breakout Refinement — Systematic Improvement
=================================================

Baseline: ch=15 trail=2.5 sl_atr=3.0 → PnL +317%, WR 35.8%, R:R 3.06

Refinements tested INDIVIDUALLY then combined:
  R1: Optimal base params (sl_atr=3.3 from sensitivity)
  R2: Body confirmation (breakout bar body > 50% of range, no wicks)
  R3: Volume confirmation (breakout bar volume > 1.3× avg)
  R4: ATR expansion filter (ATR rising = real breakout)
  R5: 2-bar confirmation (next bar also closes in breakout direction)
  R6: Progressive trail (starts wide, tightens after profit threshold)
  R7: Higher-TF trend filter (only breakout WITH 4h trend)
  R8: Time-of-day filter (avoid low-liquidity hours)
"""

import os, sys, json, numpy as np, pandas as pd
from collections import defaultdict

sys.stdout.reconfigure(line_buffering=True)

FEE_PCT = 0.10; LEVERAGE = 1; EMERGENCY_SL = 3.0
DATA_PATH = os.path.join(os.path.dirname(__file__), '..', '..',
                         'data', 'btc_5m_270days_reclassified.csv')
RESULTS_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'results')

def load_15m():
    df = pd.read_csv(DATA_PATH)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df['group'] = df.index // 3
    agg = df.groupby('group').agg(
        timestamp=('timestamp','first'), open=('open','first'),
        high=('high','max'), low=('low','min'),
        close=('close','last'), volume=('volume','sum')).reset_index(drop=True)
    return (agg['open'].values.astype(float), agg['high'].values.astype(float),
            agg['low'].values.astype(float), agg['close'].values.astype(float),
            agg['volume'].values.astype(float), agg['timestamp'].values, len(agg))

def atr_f(h,l,c,p=14):
    n=len(c);tr=np.empty(n);tr[0]=h[0]-l[0]
    for i in range(1,n):tr[i]=max(h[i]-l[i],abs(h[i]-c[i-1]),abs(l[i]-c[i-1]))
    a=np.full(n,np.nan)
    if n>=p:
        a[p-1]=tr[:p].mean()
        for i in range(p,n):a[i]=(a[i-1]*(p-1)+tr[i])/p
    return a

def ema_f(d,p):
    n=len(d);e=np.full(n,np.nan);k=2/(p+1)
    if n>p-1:
        e[p-1]=np.mean(d[:p])
        for i in range(p,n):e[i]=d[i]*k+e[i-1]*(1-k)
    return e

def calc_stats(trades,nb):
    if not trades:
        return dict(trades=0,wr=0,pnl=0,mdd=0,avg_pnl=0,avg_rr=0,
                    trades_per_day=0,daily_pnl=0,avg_bars_held=0,
                    win_avg=0,loss_avg=0,pnl_mdd=0,n_days=0)
    pnls=[t['pnl'] for t in trades]
    wins=[p for p in pnls if p>0];losses=[p for p in pnls if p<=0]
    eq=100.0;pk=100.0;mdd=0.0
    for t in sorted(trades,key=lambda x:x['exit_bar']):
        eq+=t['pnl']*(eq/100);pk=max(pk,eq)
        dd=(pk-eq)/pk*100 if pk>0 else 0;mdd=max(mdd,dd)
    tp=eq-100;nd=nb*15/1440
    wr=len(wins)/len(trades)*100
    wa=np.mean(wins) if wins else 0;la=np.mean(losses) if losses else 0
    rr=abs(wa/la) if la!=0 else float('inf')
    return dict(trades=len(trades),wr=round(wr,1),pnl=round(tp,2),
                mdd=round(mdd,2),pnl_mdd=round(tp/mdd,2) if mdd>0 else 0,
                avg_pnl=round(np.mean(pnls),4),win_avg=round(wa,4),
                loss_avg=round(la,4),avg_rr=round(rr,2),
                trades_per_day=round(len(trades)/nd,1) if nd>0 else 0,
                daily_pnl=round(tp/nd,4) if nd>0 else 0,
                avg_bars_held=round(np.mean([t['bars_held'] for t in trades]),1),
                n_days=round(nd,1))


def c1_refined(o, h, l, c, vol, a14, ts, n,
               # Base params
               channel=15, trail_K=2.5, max_sl_atr=3.0,
               # Refinement toggles
               body_confirm=False, body_min_ratio=0.5,
               vol_confirm=False, vol_mult=1.3,
               atr_expand=False, atr_expand_lookback=5,
               two_bar_confirm=False,
               progressive_trail=False, trail_tighten_after=0.5, trail_tight_K=1.5,
               trend_filter=False, trend_ema_period=200,
               time_filter=False,  # Avoid 00:00-04:00 UTC
               max_hold=192):
    """C1 Breakout with modular refinements."""

    # Precompute
    ch_h = np.full(n, np.nan); ch_l = np.full(n, np.nan)
    for i in range(channel, n):
        ch_h[i] = np.max(h[i-channel:i]); ch_l[i] = np.min(l[i-channel:i])

    sw_l = np.full(n, np.nan); sw_h = np.full(n, np.nan)
    for i in range(5, n-5):
        if l[i] == np.min(l[max(0,i-5):i+6]): sw_l[i] = l[i]
        if h[i] == np.max(h[max(0,i-5):i+6]): sw_h[i] = h[i]
    lsl = np.full(n, np.nan); lsh = np.full(n, np.nan)
    cs = np.nan; ch2 = np.nan
    for i in range(n):
        if not np.isnan(sw_l[i]): cs = sw_l[i]
        if not np.isnan(sw_h[i]): ch2 = sw_h[i]
        lsl[i] = cs; lsh[i] = ch2

    vol_ma = ema_f(vol, 20)

    # Trend EMA for filter
    trend_ema = ema_f(c, trend_ema_period) if trend_filter else None

    trades = []; pos = None; last_exit = -3

    for bar in range(max(channel+10, trend_ema_period+1 if trend_filter else 0), n):
        # ── Exit logic ──
        if pos is not None:
            ep = pos['ep']; d = pos['dir']; bh = bar - pos['eb']; slp = pos['sl']
            if d == 'LONG':
                pos['best'] = max(pos['best'], h[bar])
                cur = (c[bar]/ep-1)*100; bp = (pos['best']/ep-1)*100
            else:
                pos['best'] = min(pos['best'], l[bar])
                cur = (1-c[bar]/ep)*100; bp = (1-pos['best']/ep)*100

            worst = (l[bar]/ep-1)*100 if d == 'LONG' else (1-h[bar]/ep)*100
            if worst <= -EMERGENCY_SL:
                trades.append(dict(entry_bar=pos['eb'], exit_bar=bar, direction=d,
                    pnl=-EMERGENCY_SL*LEVERAGE-FEE_PCT, reason='EM', bars_held=bh))
                pos = None; last_exit = bar; continue

            # Fractal SL
            if d == 'LONG' and l[bar] <= slp:
                sp = (slp/ep-1)*100
                trades.append(dict(entry_bar=pos['eb'], exit_bar=bar, direction=d,
                    pnl=sp*LEVERAGE-FEE_PCT, reason='SL', bars_held=bh))
                pos = None; last_exit = bar; continue
            elif d == 'SHORT' and h[bar] >= slp:
                sp = (1-slp/ep)*100
                trades.append(dict(entry_bar=pos['eb'], exit_bar=bar, direction=d,
                    pnl=sp*LEVERAGE-FEE_PCT, reason='SL', bars_held=bh))
                pos = None; last_exit = bar; continue

            if bh >= max_hold:
                trades.append(dict(entry_bar=pos['eb'], exit_bar=bar, direction=d,
                    pnl=cur*LEVERAGE-FEE_PCT, reason='TO', bars_held=bh))
                pos = None; last_exit = bar; continue

            # Trail TP (progressive or standard)
            if bp > 0.05 and not np.isnan(a14[bar]) and a14[bar] > 0:
                if progressive_trail and bp > trail_tighten_after:
                    # Tighten trail after profit exceeds threshold
                    t_k = trail_tight_K
                else:
                    t_k = trail_K
                td = t_k * a14[bar] / c[bar] * 100
                if bp - cur >= td:
                    realized = max(0, bp - td)
                    trades.append(dict(entry_bar=pos['eb'], exit_bar=bar, direction=d,
                        pnl=realized*LEVERAGE-FEE_PCT, reason='TP', bars_held=bh))
                    pos = None; last_exit = bar; continue

        # ── Entry logic ──
        if pos is None and bar - last_exit >= 2:
            if np.isnan(ch_h[bar]) or np.isnan(a14[bar]):
                continue

            direction = None
            if c[bar] > ch_h[bar]: direction = 'LONG'
            elif c[bar] < ch_l[bar]: direction = 'SHORT'
            if not direction: continue

            # R2: Body confirmation
            if body_confirm:
                rng = h[bar] - l[bar]
                body = abs(c[bar] - o[bar])
                if rng <= 0 or body / rng < body_min_ratio:
                    continue
                # Body direction must match breakout
                if direction == 'LONG' and c[bar] < o[bar]: continue
                if direction == 'SHORT' and c[bar] > o[bar]: continue

            # R3: Volume confirmation
            if vol_confirm and not np.isnan(vol_ma[bar]) and vol_ma[bar] > 0:
                if vol[bar] < vol_mult * vol_ma[bar]:
                    continue

            # R4: ATR expansion
            if atr_expand:
                if bar < atr_expand_lookback: continue
                atr_prev = a14[bar - atr_expand_lookback]
                if np.isnan(atr_prev) or a14[bar] <= atr_prev:
                    continue  # ATR not expanding

            # R5: 2-bar confirmation (check previous bar already broke out,
            # this bar confirms)
            if two_bar_confirm:
                if bar < 1: continue
                if direction == 'LONG' and c[bar] <= c[bar-1]: continue
                if direction == 'SHORT' and c[bar] >= c[bar-1]: continue

            # R7: Trend filter
            if trend_filter and trend_ema is not None:
                if np.isnan(trend_ema[bar]): continue
                if direction == 'LONG' and c[bar] < trend_ema[bar]: continue
                if direction == 'SHORT' and c[bar] > trend_ema[bar]: continue

            # R8: Time filter (avoid 00:00-04:00 UTC)
            if time_filter:
                hour = pd.Timestamp(ts[bar]).hour
                if 0 <= hour < 4: continue

            # Entry
            if bar + 1 >= n: continue
            ep = o[bar+1]
            if direction == 'LONG':
                sw = lsl[bar]
                slp = max(sw, ep - max_sl_atr*a14[bar]) if not np.isnan(sw) else ep - max_sl_atr*a14[bar]
            else:
                sw = lsh[bar]
                slp = min(sw, ep + max_sl_atr*a14[bar]) if not np.isnan(sw) else ep + max_sl_atr*a14[bar]

            sd = abs(ep - slp) / ep * 100
            if sd < 0.15 or sd > 3.0: continue

            pos = dict(eb=bar+1, ep=ep, dir=direction, best=ep, sl=slp)

    return trades


def pr(name, s):
    checks = [s['wr']>50, s['avg_rr']>1.0, s['daily_pnl']>0.2,
              s['avg_pnl']>0.10, s['trades_per_day']>=2.0]
    marks = ''.join(['v' if x else 'x' for x in checks])
    print(f"  {name:<50} T={s['trades']:>5} WR={s['wr']:>4.1f}% R:R={s['avg_rr']:>4.2f} "
          f"PnL={s['pnl']:>+8.1f}% MDD={s['mdd']:>5.1f}% "
          f"Daily={s['daily_pnl']:>+7.4f}% [{marks}]")


def main():
    print("="*70)
    print("  C1 Breakout Refinement")
    print("="*70)
    o, h, l, c, vol, ts, n = load_15m()
    a14 = atr_f(h, l, c, 14)
    print(f"  {n} bars, {n*15/1440:.0f}d\n")

    # ═══════════════════════════════════
    #  Baseline
    # ═══════════════════════════════════
    print("── BASELINE ──")
    base = c1_refined(o,h,l,c,vol,a14,ts,n)
    bs = calc_stats(base, n)
    pr("Baseline (ch=15 tk=2.5 sl=3.0)", bs)

    # Optimized base params
    base_opt = c1_refined(o,h,l,c,vol,a14,ts,n, max_sl_atr=3.3)
    pr("R0: sl_atr=3.3 (from sensitivity)", calc_stats(base_opt, n))

    # ═══════════════════════════════════
    #  Individual Refinements
    # ═══════════════════════════════════
    print("\n── INDIVIDUAL REFINEMENTS (each added to baseline) ──")

    refinements = {
        'R2a: Body confirm 40%': dict(body_confirm=True, body_min_ratio=0.4),
        'R2b: Body confirm 50%': dict(body_confirm=True, body_min_ratio=0.5),
        'R2c: Body confirm 60%': dict(body_confirm=True, body_min_ratio=0.6),
        'R3a: Volume >1.2x': dict(vol_confirm=True, vol_mult=1.2),
        'R3b: Volume >1.5x': dict(vol_confirm=True, vol_mult=1.5),
        'R3c: Volume >2.0x': dict(vol_confirm=True, vol_mult=2.0),
        'R4a: ATR expand (lb=3)': dict(atr_expand=True, atr_expand_lookback=3),
        'R4b: ATR expand (lb=5)': dict(atr_expand=True, atr_expand_lookback=5),
        'R5: 2-bar confirmation': dict(two_bar_confirm=True),
        'R6a: Prog trail 0.5%/1.5K': dict(progressive_trail=True, trail_tighten_after=0.5, trail_tight_K=1.5),
        'R6b: Prog trail 1.0%/1.5K': dict(progressive_trail=True, trail_tighten_after=1.0, trail_tight_K=1.5),
        'R6c: Prog trail 0.5%/2.0K': dict(progressive_trail=True, trail_tighten_after=0.5, trail_tight_K=2.0),
        'R7a: Trend EMA(100)': dict(trend_filter=True, trend_ema_period=100),
        'R7b: Trend EMA(200)': dict(trend_filter=True, trend_ema_period=200),
        'R8: No 00-04 UTC': dict(time_filter=True),
    }

    individual_results = {}
    for rname, params in refinements.items():
        t = c1_refined(o,h,l,c,vol,a14,ts,n, **params)
        s = calc_stats(t, n)
        pr(rname, s)
        individual_results[rname] = dict(params=params, stats=s,
                                         pnl_delta=s['pnl']-bs['pnl'],
                                         wr_delta=s['wr']-bs['wr'])

    # ═══════════════════════════════════
    #  Best individual improvements
    # ═══════════════════════════════════
    print("\n── RANKING (by PnL improvement) ──")
    ranked = sorted(individual_results.items(), key=lambda x: -x[1]['stats']['pnl'])
    for rname, res in ranked:
        s = res['stats']
        delta = res['pnl_delta']
        wr_d = res['wr_delta']
        print(f"  {rname:<40} PnL={s['pnl']:>+8.1f}% (Δ{delta:>+7.1f}%) "
              f"WR={s['wr']:>4.1f}% (Δ{wr_d:>+4.1f}pp) T={s['trades']}")

    # ═══════════════════════════════════
    #  Combine top improvements
    # ═══════════════════════════════════
    print("\n── COMBINATIONS ──")

    # Pick top positive-delta refinements and combine
    combos = [
        ('Best2: sl=3.3 + body50%',
         dict(max_sl_atr=3.3, body_confirm=True, body_min_ratio=0.5)),
        ('Best3: sl=3.3 + body50% + vol1.5x',
         dict(max_sl_atr=3.3, body_confirm=True, body_min_ratio=0.5,
              vol_confirm=True, vol_mult=1.5)),
        ('Best4: sl=3.3 + body50% + progTrail',
         dict(max_sl_atr=3.3, body_confirm=True, body_min_ratio=0.5,
              progressive_trail=True, trail_tighten_after=0.5, trail_tight_K=1.5)),
        ('Best5: sl=3.3 + body40% + 2bar',
         dict(max_sl_atr=3.3, body_confirm=True, body_min_ratio=0.4,
              two_bar_confirm=True)),
        ('Best6: sl=3.3 + body50% + noNight',
         dict(max_sl_atr=3.3, body_confirm=True, body_min_ratio=0.5,
              time_filter=True)),
        ('AllIn: sl=3.3+body50%+vol1.2+prog+noNight',
         dict(max_sl_atr=3.3, body_confirm=True, body_min_ratio=0.5,
              vol_confirm=True, vol_mult=1.2,
              progressive_trail=True, trail_tighten_after=0.5, trail_tight_K=1.5,
              time_filter=True)),
    ]

    best_combo = None; best_pnl = bs['pnl']
    for cname, params in combos:
        t = c1_refined(o,h,l,c,vol,a14,ts,n, **params)
        s = calc_stats(t, n)
        pr(cname, s)
        if s['pnl'] > best_pnl:
            best_pnl = s['pnl']
            best_combo = (cname, params, s)

    # ═══════════════════════════════════
    #  Final validation of best
    # ═══════════════════════════════════
    print(f"\n{'='*70}")
    winner = best_combo if best_combo else ("Baseline", {}, bs)
    wname, wparams, ws = winner
    print(f"  BEST: {wname}")
    print(f"  Params: {wparams}")
    print(f"  T={ws['trades']} WR={ws['wr']}% R:R={ws['avg_rr']} "
          f"PnL={ws['pnl']:+.1f}% MDD={ws['mdd']:.1f}% PnL/MDD={ws['pnl_mdd']:.1f}")

    # WF
    print(f"\n  Walk-Forward:")
    t_best = c1_refined(o,h,l,c,vol,a14,ts,n, **wparams)
    oos_results = []
    for fi in range(5):
        ie=int(n*(fi+1)/6);oe=int(n*(fi+2)/6)
        ist = c1_refined(o[:ie],h[:ie],l[:ie],c[:ie],vol[:ie],a14[:ie],ts[:ie],ie, **wparams)
        oost = c1_refined(o[ie:oe],h[ie:oe],l[ie:oe],c[ie:oe],vol[ie:oe],
                          a14[ie:oe],ts[ie:oe],oe-ie, **wparams)
        iss = calc_stats(ist,ie); ooss = calc_stats(oost,oe-ie)
        oos_results.append(ooss)
        p = 'P' if ooss['pnl'] > 0 else 'F'
        print(f"    F{fi+1}: IS WR={iss['wr']}% PnL={iss['pnl']:+.1f}% | "
              f"OOS WR={ooss['wr']}% R:R={ooss['avg_rr']} PnL={ooss['pnl']:+.1f}% [{p}]")
    oos_pos = sum(1 for r in oos_results if r['pnl'] > 0)
    print(f"    OOS: {sum(r['pnl'] for r in oos_results):+.1f}%, {oos_pos}/5 PASS")

    # 3-way
    print(f"\n  3-Way Split:")
    s1=int(n*0.5);s2=int(n*0.75)
    for label,start,end in [("Train",0,s1),("Valid",s1,s2),("Test",s2,n)]:
        t = c1_refined(o[start:end],h[start:end],l[start:end],c[start:end],
                       vol[start:end],a14[start:end],ts[start:end],end-start, **wparams)
        s = calc_stats(t, end-start)
        print(f"    {label}: T={s['trades']} WR={s['wr']}% PnL={s['pnl']:+.1f}% "
              f"Daily={s['daily_pnl']:+.4f}%")

    # MC
    print(f"\n  MC (999 sims):")
    real_pnl = ws['pnl']
    better = 0; sp = []
    for seed in [42,123,7]:
        rng = np.random.RandomState(seed)
        for _ in range(333):
            shuffled = []
            for t2 in t_best:
                if rng.random() > 0.5:
                    shuffled.append(-t2['pnl'] - 2*FEE_PCT)
                else:
                    shuffled.append(t2['pnl'])
            eq = 100.0
            for p2 in shuffled:
                eq += p2*(eq/100)
            s_pnl = eq - 100
            sp.append(s_pnl)
            if s_pnl >= real_pnl: better += 1
    mc_p = better/len(sp)
    print(f"    Real={real_pnl:+.1f}% Shuf={np.mean(sp):+.1f}% p={mc_p:.4f} "
          f"{'DISC' if mc_p<0.05 else 'N-DISC'}")

    # Exit breakdown
    print(f"\n  Exit Breakdown:")
    rs = defaultdict(int)
    for t2 in t_best: rs[t2['reason']] += 1
    for reason in sorted(rs.keys()):
        subset = [t2 for t2 in t_best if t2['reason'] == reason]
        pnls = [t2['pnl'] for t2 in subset]
        wr = sum(1 for p in pnls if p > 0) / len(pnls) * 100
        print(f"    {reason:>4}: {len(subset):>5} ({len(subset)/len(t_best)*100:.0f}%) "
              f"WR={wr:.0f}% avg={np.mean(pnls):+.4f}%")

    # Improvement vs baseline
    print(f"\n  vs Baseline:")
    print(f"    PnL:  {bs['pnl']:+.1f}% → {ws['pnl']:+.1f}% (Δ{ws['pnl']-bs['pnl']:+.1f}%)")
    print(f"    WR:   {bs['wr']}% → {ws['wr']}% (Δ{ws['wr']-bs['wr']:+.1f}pp)")
    print(f"    R:R:  {bs['avg_rr']} → {ws['avg_rr']}")
    print(f"    MDD:  {bs['mdd']}% → {ws['mdd']}%")
    print(f"    T/d:  {bs['trades_per_day']} → {ws['trades_per_day']}")

    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, 'c1_breakout_refinement.json'), 'w') as f:
        json.dump(dict(best=wname, params={k:v for k,v in wparams.items()},
                       stats=ws), f, indent=2, default=str)
    print(f"\n  Saved.")


if __name__ == '__main__':
    main()
