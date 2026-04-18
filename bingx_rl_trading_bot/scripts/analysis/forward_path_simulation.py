"""Forward Monte Carlo path simulation from current live state."""
import json
import numpy as np
import random
import pandas as pd
import math

# Current state
with open('results/c1_breakout_state.json') as f:
    state = json.load(f)

v26 = state['trade_history'][1:]
current_pnl = sum(t['pnl_pct'] for t in v26)
n_done = len(v26)
wins = [t for t in v26 if t['pnl_pct'] > 0]
losses = [t for t in v26 if t['pnl_pct'] <= 0]

# Build equity curve to find DD
eq = [1.0]
for t in v26:
    eq.append(eq[-1] * (1 + t['pnl_pct']/100))
peak = [eq[0]]
for e in eq[1:]:
    peak.append(max(peak[-1], e))
dd_curve = [(p-e)/p*100 for e, p in zip(eq, peak)]
current_dd = dd_curve[-1]
current_equity = eq[-1]
cur_peak = peak[-1]

print('=== 현재 상태 ===')
print('거래: {} | PnL: {:+.2f}% | Equity: {:.4f} | DD: {:.2f}% | Peak: {:.4f}'.format(
    n_done, current_pnl, current_equity, current_dd, cur_peak))
print()

# Load backtest trades for forward simulation
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

bt_trades = []
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
            bt_trades.append((ex-0.10)*3)  # 3x leveraged PnL
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
            pos={'d':d, 'ep':ep, 'sl':sl, 'bp':ep, 'bh':0}

bt_pnls = np.array(bt_trades)
print('백테스트 거래 기준 (3x):')
print('  N: {} | 평균: {:+.3f}% | 중위: {:+.3f}% | std: {:.2f}%'.format(len(bt_pnls), bt_pnls.mean(), np.median(bt_pnls), bt_pnls.std()))
print()

# Forward Monte Carlo: sample from backtest distribution
random.seed(42)
np.random.seed(42)
SIMS = 5000
HORIZONS = [10, 20, 35, 50, 100]  # additional trades from current

print('='*70)
print('  포워드 시뮬레이션 (현재 상태 기준, {} simulations)'.format(SIMS))
print('='*70)
print()

for h in HORIZONS:
    # Sample h trades from backtest distribution
    final_pnls = []
    recovery_count = 0
    max_dd_count = 0  # how many sims exceed current DD of 9.78% significantly

    for _ in range(SIMS):
        # Simulate from current state
        sample = np.random.choice(bt_pnls, size=h, replace=True)
        sim_eq = current_equity
        sim_peak = cur_peak
        sim_max_dd = current_dd
        recovered = False
        for p in sample:
            sim_eq = sim_eq * (1 + p/100)
            if sim_eq > sim_peak:
                sim_peak = sim_eq
                if not recovered:
                    recovered = True  # First time making new equity high
            dd_now = (sim_peak - sim_eq) / sim_peak * 100
            if dd_now > sim_max_dd:
                sim_max_dd = dd_now
        final_pnl = (sim_eq - 1) * 100  # ROI from initial
        final_pnls.append(final_pnl)
        if recovered:
            recovery_count += 1
        if sim_max_dd > 15.5:
            max_dd_count += 1

    final_pnls = np.array(final_pnls)
    p5 = np.percentile(final_pnls, 5)
    p25 = np.percentile(final_pnls, 25)
    p50 = np.percentile(final_pnls, 50)
    p75 = np.percentile(final_pnls, 75)
    p95 = np.percentile(final_pnls, 95)
    p_positive = (final_pnls > 0).sum() / SIMS * 100
    p_recover = recovery_count / SIMS * 100
    p_critical_dd = max_dd_count / SIMS * 100

    print('+{} trades (total {}):'.format(h, n_done + h))
    print('  ROI 분포: P5 {:+.1f}%  P25 {:+.1f}%  P50 {:+.1f}%  P75 {:+.1f}%  P95 {:+.1f}%'.format(p5, p25, p50, p75, p95))
    print('  P(ROI > 0) = {:.1f}%'.format(p_positive))
    print('  P(자본 회복 — 새 고점) = {:.1f}%'.format(p_recover))
    print('  P(DD > 15.5% 경고) = {:.1f}%'.format(p_critical_dd))
    print()

# Also: probability of hitting DD > 15.5% in next N trades (regardless of final PnL)
print('='*70)
print('  DD 경고 발생 확률 (DD > 15.5% 돌파 시점까지)')
print('='*70)

for h in HORIZONS:
    hit_count = 0
    for _ in range(SIMS):
        sample = np.random.choice(bt_pnls, size=h, replace=True)
        sim_eq = current_equity
        sim_peak = cur_peak
        sim_max_dd = current_dd
        for p in sample:
            sim_eq = sim_eq * (1 + p/100)
            if sim_eq > sim_peak: sim_peak = sim_eq
            dd_now = (sim_peak - sim_eq) / sim_peak * 100
            if dd_now > sim_max_dd: sim_max_dd = dd_now
        if sim_max_dd > 15.5:
            hit_count += 1
    p = hit_count / SIMS * 100
    print('  앞으로 {:>3} trade 내 DD > 15.5% 확률: {:.1f}%'.format(h, p))

# Bonus: expected time to reach statistical significance
print()
print('='*70)
print('  통계 유의성 도달 예상 시간')
print('='*70)
# Need ~200 trades for statistical significance
# 3.1 trades/day
# Remaining: 200 - 15 = 185 trades
trades_per_day = 3.1
remaining = 200 - n_done
days_needed = remaining / trades_per_day
print('  200거래 도달까지: 약 {:.0f}일 ({} trades 남음)'.format(days_needed, remaining))
print('  50거래 도달까지: 약 {:.0f}일'.format((50-n_done)/trades_per_day))

# Save
result = {
    'current': {'n_trades': n_done, 'pnl': current_pnl, 'equity': current_equity, 'dd': current_dd},
    'note': 'Forward MC from current state using backtest 3x distribution'
}
with open('results/forward_path_simulation.json', 'w') as f:
    json.dump(result, f, indent=2)
