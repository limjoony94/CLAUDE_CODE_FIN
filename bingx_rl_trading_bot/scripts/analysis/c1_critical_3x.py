#!/usr/bin/env python3
"""C1 Breakout critical evaluation at 3x leverage"""
import os,sys,logging,math
import pandas as pd, numpy as np
os.chdir(os.path.join(os.path.dirname(__file__),'..','..'))
sys.path.insert(0,'.')
logging.basicConfig(level=logging.WARNING)
from scripts.production.c1_breakout.indicators import compute_atr, compute_channel, compute_fractal_swings
from scripts.production.c1_breakout.signals import C1BreakoutSignal
from scripts.production.c1_breakout.config import load_config

df=pd.read_csv('data/btc_5m_270days_reclassified.csv')
df['timestamp']=pd.to_datetime(df['timestamp'])
df['group']=df.index//3
agg=df.groupby('group').agg(timestamp=('timestamp','first'),open=('open','first'),
    high=('high','max'),low=('low','min'),close=('close','last')).reset_index(drop=True)
o=agg['open'].tolist();h=agg['high'].tolist();l=agg['low'].tolist();c=agg['close'].tolist()
ts=agg['timestamp'].values;n=len(c);nd=n*15/1440;FEE=0.10
cfg=load_config();sig=C1BreakoutSignal(cfg['strategy'])
a14=compute_atr(h,l,c,14);ch_h,ch_l=compute_channel(h,l,15)
sw_l,sw_h=compute_fractal_swings(h,l,10)

def bt_slice(start,end,oo,hh,ll,cc):
    nn=end-start
    aa=compute_atr(hh,ll,cc,14);chh,chl=compute_channel(hh,ll,15)
    swl,swh=compute_fractal_swings(hh,ll,10)
    s=C1BreakoutSignal(cfg['strategy'])
    trades=[];pos=None;last_exit=-3
    for bar in range(26,nn-1):
        if pos is not None:
            if pos['d']=='LONG':pos['bp']=max(pos['bp'],hh[bar])
            else:pos['bp']=min(pos['bp'],ll[bar])
            pos['bh']+=1
            ex=s.check_exit(pos['d'],pos['ep'],pos['bp'],hh[bar],ll[bar],cc[bar],pos['sl'],aa[bar],pos['bh'])
            if ex:
                xp=ex['exit_price']
                raw=((xp/pos['ep']-1) if pos['d']=='LONG' else (1-xp/pos['ep']))*100
                trades.append(dict(raw=raw,r=ex['reason'],xb=bar+start,eb=pos['ebar']+start,
                    d=pos['d'],ep=pos['ep'],xp=xp,bh=pos['bh'],bp=pos['bp']))
                pos=None;last_exit=bar;continue
        if pos is None and bar-last_exit>=2:
            e=s.check_entry(oo[bar],hh[bar],ll[bar],cc[bar],chh[bar],chl[bar],aa[bar],swl[bar],swh[bar])
            if e and bar+1<nn:
                pos=dict(d=e['direction'],ep=oo[bar+1],sl=e['sl_price'],bp=oo[bar+1],bh=0,ebar=bar+1)
    return trades

def st(trades,lev=3):
    if not trades:return dict(T=0,WR=0,RR=0,PnL=0,MDD=0)
    pnls=[(t['raw']-FEE)*lev for t in trades]
    wins=[p for p in pnls if p>0]
    eq=100.0;pk=100.0;mdd=0.0
    for p in pnls:eq+=p*(eq/100);pk=max(pk,eq);dd=(pk-eq)/pk*100;mdd=max(mdd,dd)
    wr=len(wins)/len(pnls)*100
    wa=np.mean(wins) if wins else 0;la=np.mean([p for p in pnls if p<=0]) or 0
    rr=abs(wa/la) if la else 0
    return dict(T=len(pnls),WR=round(wr,1),RR=round(rr,2),PnL=round(eq-100,1),MDD=round(mdd,1))

all_t=bt_slice(0,n,o,h,l,c)
print(f"Baseline 3x: {st(all_t)}")

# 1. Look-ahead progressive
print(f"\n{'='*70}")
print("  1. Progressive Look-Ahead Test")
print('='*70)
for pct in [0.25,0.5,0.75,1.0]:
    end=int(n*pct)
    tr=bt_slice(0,end,o[:end],h[:end],l[:end],c[:end])
    full_in=[t for t in all_t if t['xb']<end]
    print(f"  {pct*100:.0f}%: trunc={len(tr)} full_in_range={len(full_in)} "
          f"diff={abs(len(tr)-len(full_in))} {'OK' if abs(len(tr)-len(full_in))<=1 else 'MISMATCH'}")

# 2. 3-Way Split
print(f"\n{'='*70}")
print("  2. 3-Way Split (3x)")
print('='*70)
s1=int(n*0.5);s2=int(n*0.75)
for label,a,b in [('Train',0,s1),('Valid',s1,s2),('Test',s2,n)]:
    tr=bt_slice(a,b,o[a:b],h[a:b],l[a:b],c[a:b])
    s=st(tr)
    days=(b-a)*15/1440
    print(f"  {label:>5} ({days:.0f}d): T={s['T']} WR={s['WR']}% R:R={s['RR']} "
          f"PnL={s['PnL']:+.1f}% MDD={s['MDD']:.1f}% {'PASS' if s['PnL']>0 else 'FAIL'}")

# 3. WF 5-fold
print(f"\n{'='*70}")
print("  3. Walk-Forward 5-fold (3x)")
print('='*70)
oos_pos=0
for fi in range(5):
    ie=int(n*(fi+1)/6);oe=int(n*(fi+2)/6)
    t_is=bt_slice(0,ie,o[:ie],h[:ie],l[:ie],c[:ie])
    t_oos=bt_slice(ie,oe,o[ie:oe],h[ie:oe],l[ie:oe],c[ie:oe])
    si=st(t_is);so=st(t_oos)
    p='P' if so['PnL']>0 else 'F'
    if so['PnL']>0:oos_pos+=1
    print(f"  F{fi+1}: IS T={si['T']} PnL={si['PnL']:+.1f}% | "
          f"OOS T={so['T']} WR={so['WR']}% R:R={so['RR']} PnL={so['PnL']:+.1f}% [{p}]")
print(f"  Result: {oos_pos}/5 PASS")

# 4. MC (999 sims)
print(f"\n{'='*70}")
print("  4. MC Direction Shuffle (3x, 999 sims)")
print('='*70)
real_eq=100.0
for t in all_t:real_eq+=((t['raw']-FEE)*3)*(real_eq/100)
real_pnl=real_eq-100
better=0;sp=[]
for seed in [42,123,7]:
    rng=np.random.RandomState(seed)
    for _ in range(333):
        eq2=100.0
        for t in all_t:
            raw=t['raw'] if rng.random()>0.5 else -t['raw']
            eq2+=((raw-FEE)*3)*(eq2/100)
        s2=eq2-100;sp.append(s2)
        if s2>=real_pnl:better+=1
mc_p=better/len(sp)
print(f"  Real={real_pnl:+.1f}% Shuffle={np.mean(sp):+.1f}% p={mc_p:.4f}")

# 5. Top trade removal sensitivity
print(f"\n{'='*70}")
print("  5. Top Trade Removal (3x)")
print('='*70)
sorted_t=sorted(all_t,key=lambda x:-x['raw'])
for rm in [0,1,3,5,10,20]:
    remaining=sorted_t[rm:]
    s=st(remaining)
    removed=[f"{t['raw']:+.1f}%" for t in sorted_t[:rm]]
    print(f"  -{rm:>2} trades: T={s['T']} PnL={s['PnL']:+.1f}% MDD={s['MDD']:.1f}%"
          f"{' removed:'+str(removed[:5]) if rm<=5 else ''}")

# 6. Param sensitivity (3x)
print(f"\n{'='*70}")
print("  6. Parameter Sensitivity (3x)")
print('='*70)
for ch in [12,14,15,16,18]:
    chh2,chl2=compute_channel(h,l,ch)
    t2=[];pos=None;last_exit=-3
    for bar in range(26,n-1):
        if pos is not None:
            if pos['d']=='LONG':pos['bp']=max(pos['bp'],h[bar])
            else:pos['bp']=min(pos['bp'],l[bar])
            pos['bh']+=1
            ex=sig.check_exit(pos['d'],pos['ep'],pos['bp'],h[bar],l[bar],c[bar],pos['sl'],a14[bar],pos['bh'])
            if ex:
                xp=ex['exit_price']
                raw=((xp/pos['ep']-1) if pos['d']=='LONG' else (1-xp/pos['ep']))*100
                t2.append(dict(raw=raw,xb=bar));pos=None;last_exit=bar;continue
        if pos is None and bar-last_exit>=2:
            e=sig.check_entry(o[bar],h[bar],l[bar],c[bar],chh2[bar],chl2[bar],a14[bar],sw_l[bar],sw_h[bar])
            if e and bar+1<n:pos=dict(d=e['direction'],ep=o[bar+1],sl=e['sl_price'],bp=o[bar+1],bh=0)
    s=st(t2)
    m='<<<' if ch==15 else ''
    print(f"  ch={ch}: T={s['T']} PnL={s['PnL']:+.1f}% MDD={s['MDD']:.1f}% {m}")

# 7. Monthly (3x)
print(f"\n{'='*70}")
print("  7. Monthly Breakdown (3x)")
print('='*70)
monthly={};eq=100.0
for t in all_t:
    pnl3=(t['raw']-FEE)*3;eq_before=eq;eq+=pnl3*(eq/100)
    m=str(ts[t['eb']])[:7]
    if m not in monthly:monthly[m]={'start':eq_before,'pnl_usd':0,'trades':0}
    monthly[m]['end']=eq;monthly[m]['pnl_usd']+=eq-eq_before
    monthly[m]['trades']+=1
pm=0
for m in sorted(monthly):
    d=monthly[m];ret=(d['end']/d['start']-1)*100
    if ret>0:pm+=1
    print(f"  {m}: {d['trades']:>3}t {ret:>+7.1f}% {'PASS' if ret>0 else 'FAIL'}")
print(f"  Positive: {pm}/{len(monthly)}")
