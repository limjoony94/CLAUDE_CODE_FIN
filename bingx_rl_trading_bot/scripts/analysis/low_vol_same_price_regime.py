"""Check backtest performance in same-price, low-volatility regime as live."""
import pandas as pd
import numpy as np
import math

df5 = pd.read_csv('data/btc_5m_270days_reclassified.csv')
df5['timestamp'] = pd.to_datetime(df5['timestamp'])
df5 = df5.sort_values('timestamp').reset_index(drop=True)
df5.set_index('timestamp', inplace=True)
df = df5.resample('15min').agg({'open':'first','high':'max','low':'min','close':'last'}).dropna().reset_index()
n = len(df)
O_, H_, L_, C_ = df['open'].values, df['high'].values, df['low'].values, df['close'].values

def atr_calc(h, l, c, p=14):
    tr = [h[0]-l[0]] + [max(h[i]-l[i], abs(h[i]-c[i-1]), abs(l[i]-c[i-1])) for i in range(1, len(c))]
    a = [float('nan')]*len(c)
    if len(c) >= p:
        a[p-1] = sum(tr[:p])/p
        for i in range(p, len(c)): a[i] = (a[i-1]*(p-1)+tr[i])/p
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

# Run backtest, record each trade with entry ATR% and entry price
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
            trades.append({'pnl':(ex-0.10)*3, 'dir':pos['d'], 'entry_price':pos['ep'],
                          'atr_pct':pos['atr_pct'], 'bt_bar':pos['entry_bar']})
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
            atr_pct = a / ep * 100
            pos={'d':d, 'ep':ep, 'sl':sl, 'bp':ep, 'bh':0, 'atr_pct':atr_pct, 'entry_bar':bar+1}

print('백테스트 총 거래: {} (3x PnL)'.format(len(trades)))
print()

# Live regime: price $73,749~$75,299, ATR% ~0.29%
LIVE_PRICE_MIN, LIVE_PRICE_MAX = 73000, 76000
LIVE_ATR_MIN, LIVE_ATR_MAX = 0.21, 0.44  # from observed live range
LIVE_ATR_PCT_MEAN = 0.288

# Filter: same price range
same_price = [t for t in trades if LIVE_PRICE_MIN <= t['entry_price'] <= LIVE_PRICE_MAX]
print('같은 가격대 거래 ({}~{}): {}개'.format(LIVE_PRICE_MIN, LIVE_PRICE_MAX, len(same_price)))

# Filter: same price AND low volatility (ATR% <= live max)
low_vol = [t for t in trades if LIVE_PRICE_MIN <= t['entry_price'] <= LIVE_PRICE_MAX and t['atr_pct'] <= LIVE_ATR_MAX]
print('같은 가격대 + 저변동성 (ATR ≤ {:.2f}%): {}개'.format(LIVE_ATR_MAX, len(low_vol)))
print()

def analyze(trades, name):
    if not trades:
        print('  {}: no trades'.format(name))
        return
    n = len(trades)
    wins = sum(1 for t in trades if t['pnl']>0)
    total = sum(t['pnl'] for t in trades)
    avg = total/n
    longs = [t for t in trades if t['dir']=='LONG']
    shorts = [t for t in trades if t['dir']=='SHORT']
    long_w = sum(1 for t in longs if t['pnl']>0)
    short_w = sum(1 for t in shorts if t['pnl']>0)

    print('【{}】 n={}'.format(name, n))
    print('  WR: {:.1f}% ({}W/{}T)'.format(wins/n*100, wins, n))
    print('  총 PnL (3x): {:+.2f}%  | 거래당: {:+.3f}%'.format(total, avg))
    if longs:
        print('  LONG  WR {:.1f}% ({}/{}), PnL {:+.2f}%'.format(long_w/len(longs)*100, long_w, len(longs), sum(t['pnl'] for t in longs)))
    if shorts:
        print('  SHORT WR {:.1f}% ({}/{}), PnL {:+.2f}%'.format(short_w/len(shorts)*100, short_w, len(shorts), sum(t['pnl'] for t in shorts)))
    print()

print('백테스트 전체:')
analyze(trades, 'ALL')

analyze(same_price, '같은 가격대 ($73K~$76K)')

analyze(low_vol, '같은 가격대 + 저변동성')

# 라이브 상황 with 13거래
print('='*60)
print('라이브 (13거래, 3x):')
print('  WR: 23.1% (3/13), PnL: -2.46% (실제 기록)')
print('  LONG  WR: 42.9% (3/7), PnL: +6.28%')
print('  SHORT WR: 0.0% (0/6 or 0/7), PnL: -9.XX%')
print('='*60)
