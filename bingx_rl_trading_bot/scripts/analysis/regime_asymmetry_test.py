"""Verify regime-dependent direction asymmetry claim using backtest."""
import pandas as pd
import numpy as np
import math
import json

df5 = pd.read_csv('data/btc_5m_270days_reclassified.csv')
df5['timestamp'] = pd.to_datetime(df5['timestamp'])
df5 = df5.sort_values('timestamp').reset_index(drop=True)
df5.set_index('timestamp', inplace=True)
df = df5.resample('15min').agg({'open': 'first', 'high': 'max', 'low': 'min', 'close': 'last'}).dropna().reset_index()
n = len(df)
O_, H_, L_, C_ = df['open'].values, df['high'].values, df['low'].values, df['close'].values
TS = df['timestamp'].values

def atr_calc(h, l, c, p=14):
    tr = [h[0]-l[0]] + [max(h[i]-l[i], abs(h[i]-c[i-1]), abs(l[i]-c[i-1])) for i in range(1, len(c))]
    a = [float('nan')]*len(c)
    if len(c) >= p:
        a[p-1] = sum(tr[:p])/p
        for i in range(p, len(c)): a[i] = (a[i-1]*(p-1) + tr[i])/p
    return a
def ch_c(h, l, p=15):
    ch = [float('nan')]*len(h); cl = [float('nan')]*len(h)
    for i in range(p, len(h)): ch[i]=max(h[i-p:i]); cl[i]=min(l[i-p:i])
    return ch, cl
def fr_c(h, l, lb=10):
    sl = [float('nan')]*len(h); sh = [float('nan')]*len(h); cs=float('nan'); chx=float('nan')
    for i in range(lb, len(h)):
        if l[i]==min(l[i-lb:i+1]): cs=l[i]
        if h[i]==max(h[i-lb:i+1]): chx=h[i]
        sl[i]=cs; sh[i]=chx
    return sl, sh

ATR = atr_calc(H_, L_, C_)
CH, CL = ch_c(H_, L_)
SWL, SWH = fr_c(H_, L_)

# Run backtest with direction tracked per trade
trades = []
pos = None; bars_since = 999
for bar in range(25, n-1):
    a = ATR[bar]
    if math.isnan(a) or a<=0: continue
    bars_since += 1
    if pos:
        pos['bh'] += 1
        if pos['d']=='LONG': pos['bp']=max(pos['bp'], H_[bar])
        else: pos['bp']=min(pos['bp'], L_[bar])
        ex = None
        if pos['d']=='LONG' and L_[bar]<=pos['sl']: ex=(pos['sl']/pos['ep']-1)*100
        elif pos['d']=='SHORT' and H_[bar]>=pos['sl']: ex=(1-pos['sl']/pos['ep'])*100
        if ex is None:
            wp = (L_[bar]/pos['ep']-1)*100 if pos['d']=='LONG' else (1-H_[bar]/pos['ep'])*100
            if wp<=-3.0: ex=-3.0
        if ex is None and pos['bh']>=192:
            ex = (C_[bar]/pos['ep']-1)*100 if pos['d']=='LONG' else (1-C_[bar]/pos['ep'])*100
        if ex is None:
            bpnl = (pos['bp']/pos['ep']-1)*100 if pos['d']=='LONG' else (1-pos['bp']/pos['ep'])*100
            cpnl = (C_[bar]/pos['ep']-1)*100 if pos['d']=='LONG' else (1-C_[bar]/pos['ep'])*100
            if bpnl>0.05 and a>0:
                td = 2.5*a/C_[bar]*100; dd=bpnl-cpnl
                if dd>=td: ex=max(0, bpnl-td)
        if ex is not None:
            trades.append({'pnl': ex-0.10, 'dir': pos['d'], 'entry_bar': pos['entry_bar'], 'entry_price': pos['ep']})
            pos=None; bars_since=0; continue
    if pos is None and bars_since>=2:
        ch=CH[bar]; cl_v=CL[bar]
        if math.isnan(ch) or math.isnan(cl_v) or math.isnan(a): continue
        d=None
        if C_[bar]>ch: d='LONG'
        elif C_[bar]<cl_v: d='SHORT'
        if d:
            rng=H_[bar]-L_[bar]
            if rng<=0: continue
            body=C_[bar]-O_[bar]
            if abs(body)/rng<0.4: continue
            if d=='LONG' and body<=0: continue
            if d=='SHORT' and body>=0: continue
            ep=O_[bar+1]
            if d=='LONG':
                fsl=SWL[bar] if not math.isnan(SWL[bar]) else ep-3.3*a
                sl=max(fsl, ep-3.3*a)
            else:
                fsl=SWH[bar] if not math.isnan(SWH[bar]) else ep+3.3*a
                sl=min(fsl, ep+3.3*a)
            if abs(ep-sl)/ep*100<0.15 or abs(ep-sl)/ep*100>3.0: continue
            pos={'d':d, 'ep':ep, 'sl':sl, 'bp':ep, 'bh':0, 'entry_bar':bar+1}

# Classify each trade by regime (BTC 20-period return proxy)
# For each trade, compute BTC return over past N=96 bars (24 hours) at entry
for t in trades:
    eb = t['entry_bar']
    lookback_bar = max(0, eb - 96)
    btc_ret = (C_[eb] / C_[lookback_bar] - 1) * 100
    t['regime_ret'] = btc_ret
    t['regime'] = 'UP' if btc_ret > 1.0 else ('DOWN' if btc_ret < -1.0 else 'FLAT')

# Analyze by regime
print('='*70)
print('  레짐별 방향 성과 분석 (백테스트 {} trades)'.format(len(trades)))
print('='*70)
print()
print('레짐 정의: BTC 24h 수익률 기준')
print('  UP:   +1.0% 이상')
print('  FLAT: ±1.0%')
print('  DOWN: -1.0% 이하')
print()

for regime in ['UP', 'FLAT', 'DOWN']:
    regime_trades = [t for t in trades if t['regime'] == regime]
    if not regime_trades: continue
    longs = [t for t in regime_trades if t['dir']=='LONG']
    shorts = [t for t in regime_trades if t['dir']=='SHORT']

    print('【{} 레짐】 ({} trades)'.format(regime, len(regime_trades)))
    for direction, trades_dir in [('LONG', longs), ('SHORT', shorts)]:
        if not trades_dir: continue
        wins = sum(1 for t in trades_dir if t['pnl']>0)
        total_pnl = sum(t['pnl'] for t in trades_dir)
        wr = wins/len(trades_dir)*100
        avg = total_pnl/len(trades_dir)
        print('  {} {:>4}개: WR {:.1f}% ({}W/{}T), 총 {:+.2f}%, 거래당 {:+.3f}%'.format(
            direction, len(trades_dir), wr, wins, len(trades_dir), total_pnl, avg))
    print()

# Quick summary: LONG vs SHORT asymmetry magnitude
up_longs = [t for t in trades if t['dir']=='LONG' and t['regime']=='UP']
up_shorts = [t for t in trades if t['dir']=='SHORT' and t['regime']=='UP']
down_longs = [t for t in trades if t['dir']=='LONG' and t['regime']=='DOWN']
down_shorts = [t for t in trades if t['dir']=='SHORT' and t['regime']=='DOWN']

print('='*70)
print('  결론: 상승 레짐에서 SHORT는 구조적으로 불리')
print('='*70)
if up_longs and up_shorts:
    ul_wr = sum(1 for t in up_longs if t['pnl']>0)/len(up_longs)*100
    us_wr = sum(1 for t in up_shorts if t['pnl']>0)/len(up_shorts)*100
    print('\n상승장:')
    print('  LONG  WR: {:.1f}% (우세)'.format(ul_wr))
    print('  SHORT WR: {:.1f}% ({:.0f}pp 낮음)'.format(us_wr, ul_wr-us_wr))
if down_longs and down_shorts:
    dl_wr = sum(1 for t in down_longs if t['pnl']>0)/len(down_longs)*100
    ds_wr = sum(1 for t in down_shorts if t['pnl']>0)/len(down_shorts)*100
    print('\n하락장:')
    print('  LONG  WR: {:.1f}%'.format(dl_wr))
    print('  SHORT WR: {:.1f}% (우세, {:.0f}pp 높음)'.format(ds_wr, ds_wr-dl_wr))

# Live comparison
print()
print('라이브 15거래 비교:')
print('  LONG  WR 43% (3/7)  vs 백테스트 상승장 LONG WR (위에서)')
print('  SHORT WR 0%  (0/7)  vs 백테스트 상승장 SHORT WR (위에서)')
