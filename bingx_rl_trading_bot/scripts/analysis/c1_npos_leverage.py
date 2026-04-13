#!/usr/bin/env python3
"""C1 Breakout: N-pos + Leverage analysis"""
import os, sys, numpy as np, pandas as pd
sys.stdout.reconfigure(line_buffering=True)
os.chdir(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, os.getcwd())
from scripts.production.c1_breakout.signals import C1BreakoutSignal
from scripts.production.c1_breakout.indicators import compute_atr, compute_channel, compute_fractal_swings
from scripts.production.c1_breakout.config import load_config

df = pd.read_csv("data/btc_5m_270days_reclassified.csv")
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['group'] = df.index // 3
agg = df.groupby('group').agg(open=('open','first'),high=('high','max'),
    low=('low','min'),close=('close','last')).reset_index(drop=True)
o=agg['open'].tolist();h=agg['high'].tolist()
l=agg['low'].tolist();c=agg['close'].tolist()
n=len(c);nd=n*15/1440;FEE=0.10

cfg=load_config();sig=C1BreakoutSignal(cfg['strategy'])
at=compute_atr(h,l,c,14);ch_h,ch_l=compute_channel(h,l,15)
sw_l,sw_h=compute_fractal_swings(h,l,10)

# Count all signals
all_sigs=0
for bar in range(26,n-1):
    es=sig.check_entry(o[bar],h[bar],l[bar],c[bar],ch_h[bar],ch_l[bar],at[bar],sw_l[bar],sw_h[bar])
    if es: all_sigs+=1
print(f"Total entry signals: {all_sigs} ({all_sigs/nd:.1f}/d)")

print(f"\n{'='*75}")
header = f"  {'N':>2} {'Lev':>3} {'Size':>5} {'Trades':>6} {'T/d':>5} {'WR':>5} {'Daily%':>8} {'MDD%':>6} {'PnL/MDD':>8} {'Ann%':>7}"
print(header)
print(f"  {'─'*2} {'─'*3} {'─'*5} {'─'*6} {'─'*5} {'─'*5} {'─'*8} {'─'*6} {'─'*8} {'─'*7}")

for N in [1,2,3,5,7]:
    for lev in [1,2,3,5]:
        sp = 100.0/N
        positions=[];trades=[]
        for bar in range(26,n-1):
            closed=[]
            for idx,pos in enumerate(positions):
                if pos['d']=='LONG':pos['bp']=max(pos['bp'],h[bar])
                else:pos['bp']=min(pos['bp'],l[bar])
                pos['bh']+=1
                ex=sig.check_exit(pos['d'],pos['ep'],pos['bp'],
                    h[bar],l[bar],c[bar],pos['sl'],at[bar],pos['bh'])
                if ex:
                    xp=ex['exit_price']
                    pnl=((xp/pos['ep']-1) if pos['d']=='LONG' else (1-xp/pos['ep']))*100*lev-FEE
                    trades.append(dict(pnl=pnl,sp=sp,xb=bar))
                    closed.append(idx)
            for idx in sorted(closed,reverse=True):positions.pop(idx)
            if len(positions)<N:
                es=sig.check_entry(o[bar],h[bar],l[bar],c[bar],
                    ch_h[bar],ch_l[bar],at[bar],sw_l[bar],sw_h[bar])
                if es and bar+1<n:
                    positions.append(dict(d=es['direction'],ep=o[bar+1],
                        sl=es['sl_price'],bp=o[bar+1],bh=0))
        if not trades:continue
        eq=100.0;pk=100.0;mdd=0.0
        for t in sorted(trades,key=lambda x:x['xb']):
            eq+=t['pnl']*(t['sp']/100)*(eq/100)
            pk=max(pk,eq)
            dd=(pk-eq)/pk*100 if pk>0 else 0
            mdd=max(mdd,dd)
        pnl=eq-100;daily=pnl/nd
        pnls=[t['pnl'] for t in trades]
        wins=[p for p in pnls if p>0]
        wr=len(wins)/len(trades)*100
        pm=pnl/mdd if mdd>0 else 0
        ann=daily*365
        mark=""
        if 0.8<=daily<=1.2: mark=" <<< TARGET"
        print(f"  {N:>2} {lev:>2}x {sp:>4.0f}% {len(trades):>6} {len(trades)/nd:>4.1f} "
              f"{wr:>4.1f}% {daily:>+7.3f}% {mdd:>5.1f}% {pm:>7.1f} {ann:>+6.0f}%{mark}")
    print()

# Margin distribution explanation
print(f"\n{'='*75}")
print("  마진 분배 방식")
print("="*75)
print("""
  N-pos 마진 분배:
    N=1: 전체 자본의 100%를 1 포지션에 투입
    N=2: 전체 자본의 50%씩 2 포지션 동시 운영
    N=3: 전체 자본의 33%씩 3 포지션 동시 운영
    N=5: 전체 자본의 20%씩 5 포지션 동시 운영

  BingX Hedge Mode에서:
    - LONG/SHORT 독립 포지션 가능
    - 각 포지션은 격리 마진 (Isolated Margin) 사용 권장
    - 포지션별 마진 = 총 자본 / N
    - 레버리지는 포지션별 동일 적용

  예: 자본 $10,000, N=3, 2x leverage
    - 포지션당 마진: $3,333
    - 포지션당 노출: $6,666 (마진 × 레버리지)
    - 총 노출: $20,000 (자본의 200%)
    - 개별 SL 3.3×ATR ≈ 1% → 마진의 2% 손실/포지션
    - 최악(3 동시 SL): 마진의 6% 손실
""")
