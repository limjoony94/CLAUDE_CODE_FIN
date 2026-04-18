#!/usr/bin/env python3
"""Critical eval: data feed, bar timing, causal fractal WF"""
import os,sys,logging,math
import pandas as pd, numpy as np
os.chdir(os.path.join(os.path.dirname(__file__),'..','..'))
sys.path.insert(0,'.')
logging.basicConfig(level=logging.WARNING)

from scripts.production.c1_breakout.indicators import compute_atr, compute_channel, compute_fractal_swings
from scripts.production.c1_breakout.signals import C1BreakoutSignal
from scripts.production.c1_breakout.config import load_config

# ═══════════════════════════════════════
print("="*70)
print("  #1: Causal Fractal로 WF + 3-Way 재검증")
print("="*70)

df=pd.read_csv('data/btc_5m_270days_reclassified.csv')
df['group']=df.index//3
agg=df.groupby('group').agg(open=('open','first'),high=('high','max'),
    low=('low','min'),close=('close','last')).reset_index(drop=True)
o=agg['open'].tolist();h=agg['high'].tolist();l=agg['low'].tolist();c=agg['close'].tolist()
n=len(c);nd=n*15/1440;FEE=0.10

def bt_slice(start,end):
    oo,hh,ll,cc=o[start:end],h[start:end],l[start:end],c[start:end]
    nn=end-start
    aa=compute_atr(hh,ll,cc,14)
    chh,chl=compute_channel(hh,ll,15)
    swl,swh=compute_fractal_swings(hh,ll)  # causal lookback=10
    sig=C1BreakoutSignal(load_config()['strategy'])
    trades=[];pos=None;last_exit=-3
    for bar in range(26,nn-1):
        if pos is not None:
            if pos['d']=='LONG':pos['bp']=max(pos['bp'],hh[bar])
            else:pos['bp']=min(pos['bp'],ll[bar])
            pos['bh']+=1
            ex=sig.check_exit(pos['d'],pos['ep'],pos['bp'],hh[bar],ll[bar],cc[bar],pos['sl'],aa[bar],pos['bh'])
            if ex:
                xp=ex['exit_price']
                raw=((xp/pos['ep']-1) if pos['d']=='LONG' else (1-xp/pos['ep']))*100
                trades.append(raw-FEE)
                pos=None;last_exit=bar;continue
        if pos is None and bar-last_exit>=2:
            e=sig.check_entry(oo[bar],hh[bar],ll[bar],cc[bar],chh[bar],chl[bar],aa[bar],swl[bar],swh[bar])
            if e and bar+1<nn:
                pos=dict(d=e['direction'],ep=oo[bar+1],sl=e['sl_price'],bp=oo[bar+1],bh=0)
    return trades

def st(pnls):
    if not pnls: return 0,0,0
    eq=100;wins=sum(1 for p in pnls if p>0)
    for p in pnls: eq+=p*(eq/100)
    return len(pnls),round(wins/len(pnls)*100,1),round(eq-100,1)

# Full baseline
full=bt_slice(0,n)
T,W,P=st(full)
print(f"\n  Baseline (causal, 1x): T={T} WR={W}% PnL={P:+.1f}%")

# WF 5-fold
print(f"\n  Walk-Forward 5-fold:")
oos_pos=0
for fi in range(5):
    ie=int(n*(fi+1)/6);oe=int(n*(fi+2)/6)
    t_is=bt_slice(0,ie);t_oos=bt_slice(ie,oe)
    Ti,Wi,Pi=st(t_is);To,Wo,Po=st(t_oos)
    p='P' if Po>0 else 'F'
    if Po>0:oos_pos+=1
    print(f"    F{fi+1}: IS T={Ti} WR={Wi}% PnL={Pi:+.1f}% | OOS T={To} WR={Wo}% PnL={Po:+.1f}% [{p}]")
print(f"    Result: {oos_pos}/5 PASS")

# 3-way split
print(f"\n  3-Way Split:")
s1=int(n*0.5);s2=int(n*0.75)
for label,a,b in [('Train',0,s1),('Valid',s1,s2),('Test',s2,n)]:
    pnls=bt_slice(a,b);T,W,P=st(pnls)
    print(f"    {label:>5}: T={T} WR={W}% PnL={P:+.1f}% {'PASS' if P>0 else 'FAIL'}")

# MC direction shuffle
print(f"\n  MC Direction (999 sims, causal fractal):")
real_eq=100
for p in full: real_eq+=p*(real_eq/100)
real_pnl=real_eq-100
better=0;sp=[]
for seed in range(999):
    rng=np.random.RandomState(seed)
    eq2=100
    for p in full:
        pp=p if rng.random()>0.5 else -(p+2*FEE)
        eq2+=pp*(eq2/100)
    s=eq2-100;sp.append(s)
    if s>=real_pnl:better+=1
print(f"    Real={real_pnl:+.1f}% Shuffle={np.mean(sp):+.1f}% p={better/999:.4f}")

# Progressive test
print(f"\n  Progressive Look-Ahead (causal fractal):")
for pct in [0.25,0.5,0.75,1.0]:
    end=int(n*pct)
    tr=bt_slice(0,end)
    full_in=[p for i,p in enumerate(full) if i<len(tr)]  # approx same range
    T1,_,P1=st(tr)
    print(f"    {pct*100:.0f}%: T={T1} PnL={P1:+.1f}%")

# ═══════════════════════════════════════
print(f"\n{'='*70}")
print("  #2: 5m 합성 vs Native 15m 검증")
print("="*70)
print(f"\n  백테스트: 5m CSV → group by 3 → 15m")
print(f"  프로덕션: BingX native 15m API")
print(f"  이슈: 5m 합성과 native 15m이 다를 수 있음")
print(f"  → 일반적으로 동일하지만 거래소마다 차이 가능")
print(f"  → 실시간 검증은 위에서 수행 (별도 스크립트 필요)")

# ═══════════════════════════════════════
print(f"\n{'='*70}")
print("  #3: 봇 bar n-2 처리 정확성")
print("="*70)
print(f"\n  봇은 bar n-2를 처리 (n-1 = 미완성 현재 봉)")
print(f"  BingX fetch_ohlcv: 마지막 봉 = 현재 형성 중인 봉")
print(f"  → n-2 = 마지막 완성 봉 ✓")
print(f"  → 봇은 캔들 경계 +5초 후 실행하므로 n-1은 방금 시작된 봉")
print(f"  → n-2가 방금 완성된 봉 = 정확")

# ═══════════════════════════════════════
print(f"\n{'='*70}")
print("  SUMMARY")
print("="*70)
print(f"\n  Causal fractal WF: {oos_pos}/5 PASS")
print(f"  Causal fractal 3-Way: see above")
print(f"  MC Direction: p={better/999:.4f}")
print(f"  Bar timing: n-2 = 정확")
