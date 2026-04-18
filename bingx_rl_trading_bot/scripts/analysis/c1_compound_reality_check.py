#!/usr/bin/env python3
"""Compound return reality check — is +8,338% at 3x real?"""
import os,sys,logging,math
import pandas as pd, numpy as np
os.chdir(os.path.join(os.path.dirname(__file__),'..','..'))
sys.path.insert(0,'.')
logging.basicConfig(level=logging.WARNING)
from scripts.production.c1_breakout.indicators import compute_atr, compute_channel, compute_fractal_swings
from scripts.production.c1_breakout.signals import C1BreakoutSignal
from scripts.production.c1_breakout.config import load_config

df=pd.read_csv('data/btc_5m_270days_reclassified.csv')
df['group']=df.index//3
agg=df.groupby('group').agg(open=('open','first'),high=('high','max'),
    low=('low','min'),close=('close','last')).reset_index(drop=True)
o=agg['open'].tolist();h=agg['high'].tolist();l=agg['low'].tolist();c=agg['close'].tolist()
n=len(c);nd=n*15/1440;FEE=0.10
cfg=load_config();sig=C1BreakoutSignal(cfg['strategy'])
a14=compute_atr(h,l,c,14);ch_h,ch_l=compute_channel(h,l,15)
sw_l,sw_h=compute_fractal_swings(h,l)

# Collect all trades
trades=[];pos=None;last_exit=-3
for bar in range(26,n-1):
    if pos is not None:
        if pos['d']=='LONG':pos['bp']=max(pos['bp'],h[bar])
        else:pos['bp']=min(pos['bp'],l[bar])
        pos['bh']+=1
        ex=sig.check_exit(pos['d'],pos['ep'],pos['bp'],h[bar],l[bar],c[bar],pos['sl'],a14[bar],pos['bh'])
        if ex:
            xp=ex['exit_price']
            raw=((xp/pos['ep']-1) if pos['d']=='LONG' else (1-xp/pos['ep']))*100
            trades.append(raw)
            pos=None;last_exit=bar;continue
    if pos is None and bar-last_exit>=2:
        s=sig.check_entry(o[bar],h[bar],l[bar],c[bar],ch_h[bar],ch_l[bar],a14[bar],sw_l[bar],sw_h[bar])
        if s and bar+1<n:
            pos=dict(d=s['direction'],ep=o[bar+1],sl=s['sl_price'],bp=o[bar+1],bh=0)

raws = trades  # list of raw % per trade
print("="*70)
print("  Compound Return Reality Check")
print("="*70)
print(f"  Total trades: {len(raws)}")
print(f"  Days: {nd:.0f}")
print()

# ═══════════════════════════════════
# 1. Basic stats
# ═══════════════════════════════════
print("  1. 기본 통계")
wins = [r for r in raws if r > FEE]
losses = [r for r in raws if r <= FEE]
print(f"     WR: {len(wins)/len(raws)*100:.1f}%")
print(f"     Avg raw: {np.mean(raws):+.4f}%")
print(f"     Win avg raw: {np.mean(wins):+.4f}%")
print(f"     Loss avg raw: {np.mean(losses):+.4f}%")

# ═══════════════════════════════════
# 2. Compound at each leverage
# ═══════════════════════════════════
print(f"\n  2. 레버리지별 Compound (실제 거래 순서)")
for lev in [1, 2, 3, 5]:
    eq = 100.0
    for r in raws:
        pnl = (r - FEE) * lev
        eq += pnl * (eq / 100)
    print(f"     {lev}x: $100 → ${eq:,.0f} ({eq-100:+,.0f}%)")

# ═══════════════════════════════════
# 3. Additive (NO compound) — 가장 보수적
# ═══════════════════════════════════
print(f"\n  3. Additive (compound 없이, 항상 원금 $100 기준)")
for lev in [1, 2, 3, 5]:
    total = sum((r - FEE) * lev for r in raws)
    print(f"     {lev}x: 총 {total:+.1f}%, daily {total/nd:+.2f}%")

# ═══════════════════════════════════
# 4. Geometric mean per trade
# ═══════════════════════════════════
print(f"\n  4. 기하 평균 (Geometric Mean) per trade")
for lev in [1, 2, 3]:
    growth_factors = [1 + (r - FEE) * lev / 100 for r in raws]
    # Filter out negatives (can't take log of <=0)
    valid = [g for g in growth_factors if g > 0]
    if len(valid) < len(growth_factors):
        print(f"     {lev}x: {len(growth_factors)-len(valid)} trades went below 0 — LIQUIDATION!")
        continue
    geo_mean = np.exp(np.mean(np.log(valid)))
    total = geo_mean ** len(raws)
    print(f"     {lev}x: geo_mean = {geo_mean:.6f} per trade, "
          f"final = {total*100:.0f}% ({(total-1)*100:+.0f}%)")

# ═══════════════════════════════════
# 5. Path dependency: 순서 셔플 (1000회)
# ═══════════════════════════════════
print(f"\n  5. 거래 순서 셔플 (같은 거래, 다른 순서, 1000회)")
for lev in [1, 3]:
    results = []
    for seed in range(1000):
        rng = np.random.RandomState(seed)
        shuffled = rng.permutation(raws)
        eq = 100.0
        bust = False
        for r in shuffled:
            pnl = (r - FEE) * lev
            eq += pnl * (eq / 100)
            if eq <= 0:
                bust = True; break
        results.append(eq - 100 if not bust else -100)

    busts = sum(1 for r in results if r <= -99)
    valid = [r for r in results if r > -99]
    print(f"     {lev}x ({len(raws)} trades shuffled):")
    print(f"       중앙값: {np.median(valid):+,.0f}%")
    print(f"       평균:   {np.mean(valid):+,.0f}%")
    print(f"       5th:    {np.percentile(valid,5):+,.0f}%")
    print(f"       25th:   {np.percentile(valid,25):+,.0f}%")
    print(f"       75th:   {np.percentile(valid,75):+,.0f}%")
    print(f"       95th:   {np.percentile(valid,95):+,.0f}%")
    print(f"       파산:   {busts}/{len(results)}")
    print(f"       원래:   {8338 if lev==3 else 392:+,.0f}% (실제 순서)")

# ═══════════════════════════════════
# 6. 월별 수익률 (compound가 아닌 단순 %)
# ═══════════════════════════════════
print(f"\n  6. 월별 단순 수익률 (Additive, 3x)")
from collections import defaultdict
import pandas as pd
monthly = defaultdict(list)
all_trades_detail = []
pos=None;last_exit=-3
for bar in range(26,n-1):
    if pos is not None:
        if pos['d']=='LONG':pos['bp']=max(pos['bp'],h[bar])
        else:pos['bp']=min(pos['bp'],l[bar])
        pos['bh']+=1
        ex=sig.check_exit(pos['d'],pos['ep'],pos['bp'],h[bar],l[bar],c[bar],pos['sl'],a14[bar],pos['bh'])
        if ex:
            xp=ex['exit_price']
            raw=((xp/pos['ep']-1) if pos['d']=='LONG' else (1-xp/pos['ep']))*100
            ts_str = str(agg['timestamp'].iloc[pos['ebar']])[:7]
            pnl3 = (raw - FEE) * 3
            monthly[ts_str].append(pnl3)
            pos=None;last_exit=bar;continue
    if pos is None and bar-last_exit>=2:
        s=sig.check_entry(o[bar],h[bar],l[bar],c[bar],ch_h[bar],ch_l[bar],a14[bar],sw_l[bar],sw_h[bar])
        if s and bar+1<n:
            pos=dict(d=s['direction'],ep=o[bar+1],sl=s['sl_price'],bp=o[bar+1],bh=0,ebar=bar+1)

agg['timestamp'] = pd.to_datetime(agg.get('timestamp', range(n)))

print(f"     월    | 거래 | 합산PnL |  평균  | $1000 기준")
for m in sorted(monthly.keys()):
    pnls = monthly[m]
    total = sum(pnls)
    avg = np.mean(pnls)
    usd = total * 10  # $1000 * total% / 100 = total * 10
    print(f"     {m} | {len(pnls):>4} | {total:>+6.1f}% | {avg:>+5.2f}% | ${usd:>+7.0f}")

total_add = sum(sum(v) for v in monthly.values())
print(f"\n     총 합산 (additive 3x): {total_add:+.1f}%")
print(f"     $1,000 기준: ${total_add*10:+,.0f}")
print(f"     $10,000 기준: ${total_add*100:+,.0f}")
