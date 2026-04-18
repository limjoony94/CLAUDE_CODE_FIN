"""Compare live 13-trade window against all 13-trade windows in backtest."""
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

def atr_calc(h, l, c, p=14):
    tr = [h[0]-l[0]] + [max(h[i]-l[i], abs(h[i]-c[i-1]), abs(l[i]-c[i-1])) for i in range(1, len(c))]
    a = [float('nan')]*len(c)
    if len(c) >= p:
        a[p-1] = sum(tr[:p])/p
        for i in range(p, len(c)): a[i] = (a[i-1]*(p-1) + tr[i])/p
    return a

def ch_c(h, l, p=15):
    ch = [float('nan')]*len(h); cl = [float('nan')]*len(h)
    for i in range(p, len(h)):
        ch[i] = max(h[i-p:i]); cl[i] = min(l[i-p:i])
    return ch, cl

def fr_c(h, l, lb=10):
    sl = [float('nan')]*len(h); sh = [float('nan')]*len(h)
    cs = float('nan'); chx = float('nan')
    for i in range(lb, len(h)):
        if l[i] == min(l[i-lb:i+1]): cs = l[i]
        if h[i] == max(h[i-lb:i+1]): chx = h[i]
        sl[i] = cs; sh[i] = chx
    return sl, sh

ATR = atr_calc(H_, L_, C_)
CH, CL = ch_c(H_, L_)
SWL, SWH = fr_c(H_, L_)

trades = []
pos = None; bars_since = 999
for bar in range(25, n-1):
    a = ATR[bar]
    if math.isnan(a) or a <= 0: continue
    bars_since += 1
    if pos:
        pos['bh'] += 1
        if pos['d'] == 'LONG': pos['bp'] = max(pos['bp'], H_[bar])
        else: pos['bp'] = min(pos['bp'], L_[bar])
        ex = None
        if pos['d'] == 'LONG' and L_[bar] <= pos['sl']: ex = (pos['sl']/pos['ep']-1)*100
        elif pos['d'] == 'SHORT' and H_[bar] >= pos['sl']: ex = (1-pos['sl']/pos['ep'])*100
        if ex is None:
            wp = (L_[bar]/pos['ep']-1)*100 if pos['d']=='LONG' else (1-H_[bar]/pos['ep'])*100
            if wp <= -3.0: ex = -3.0
        if ex is None and pos['bh'] >= 192:
            ex = (C_[bar]/pos['ep']-1)*100 if pos['d']=='LONG' else (1-C_[bar]/pos['ep'])*100
        if ex is None:
            bpnl = (pos['bp']/pos['ep']-1)*100 if pos['d']=='LONG' else (1-pos['bp']/pos['ep'])*100
            cpnl = (C_[bar]/pos['ep']-1)*100 if pos['d']=='LONG' else (1-C_[bar]/pos['ep'])*100
            if bpnl > 0.05 and a > 0:
                td = 2.5*a/C_[bar]*100; dd = bpnl - cpnl
                if dd >= td: ex = max(0, bpnl-td)
        if ex is not None:
            trades.append(ex - 0.10)  # 1x additive per trade
            pos = None; bars_since = 0; continue
    if pos is None and bars_since >= 2:
        ch = CH[bar]; cl_v = CL[bar]
        if math.isnan(ch) or math.isnan(cl_v) or math.isnan(a): continue
        d = None
        if C_[bar] > ch: d = 'LONG'
        elif C_[bar] < cl_v: d = 'SHORT'
        if d:
            rng = H_[bar] - L_[bar]
            if rng <= 0: continue
            body = C_[bar] - O_[bar]
            if abs(body)/rng < 0.4: continue
            if d == 'LONG' and body <= 0: continue
            if d == 'SHORT' and body >= 0: continue
            ep = O_[bar+1]
            if d == 'LONG':
                fsl = SWL[bar] if not math.isnan(SWL[bar]) else ep - 3.3*a
                sl = max(fsl, ep - 3.3*a)
            else:
                fsl = SWH[bar] if not math.isnan(SWH[bar]) else ep + 3.3*a
                sl = min(fsl, ep + 3.3*a)
            if abs(ep-sl)/ep*100 < 0.15 or abs(ep-sl)/ep*100 > 3.0: continue
            pos = {'d': d, 'ep': ep, 'sl': sl, 'bp': ep, 'bh': 0}

# 1x → 3x PnL for equivalence with live
pnls_3x = [p*3 for p in trades]

print('Backtest: {} trades (1x avg: {:.3f}%, 3x avg: {:.3f}%)'.format(len(trades), np.mean(trades), np.mean(pnls_3x)))
print()

# Rolling 13-trade windows (compound equity)
W = 13
n_win = len(pnls_3x) - W + 1
window_pnls = []
window_wrs = []
window_max_dds = []
window_eq_final = []

for i in range(n_win):
    wnd = pnls_3x[i:i+W]
    wins = sum(1 for p in wnd if p > 0)
    wr = wins / W * 100
    # Compound equity curve
    eq = [1.0]
    for p in wnd:
        eq.append(eq[-1] * (1 + p/100))
    peak = [eq[0]]
    for e in eq[1:]:
        peak.append(max(peak[-1], e))
    dd = max((p - e) / p * 100 for e, p in zip(eq, peak))
    window_pnls.append((eq[-1] - 1) * 100)
    window_wrs.append(wr)
    window_max_dds.append(dd)
    window_eq_final.append(eq[-1])

window_pnls = np.array(window_pnls)
window_wrs = np.array(window_wrs)
window_max_dds = np.array(window_max_dds)

live_pnl = -3.63
live_wr = 23.1
live_dd = 9.78

print('=== 13-Trade 윈도우 분포 (백테스트 {}개 윈도우) ==='.format(n_win))
print()
print('ROI %:')
print('  최악: {:+.1f}%  평균: {:+.1f}%  최고: {:+.1f}%'.format(window_pnls.min(), window_pnls.mean(), window_pnls.max()))
print('  P5:   {:+.1f}%  P25: {:+.1f}%  P50: {:+.1f}%  P75: {:+.1f}%  P95: {:+.1f}%'.format(
    np.percentile(window_pnls, 5), np.percentile(window_pnls, 25), np.percentile(window_pnls, 50),
    np.percentile(window_pnls, 75), np.percentile(window_pnls, 95)))
pct = (window_pnls <= live_pnl).sum() / n_win * 100
print('  라이브 PnL {:+.2f}%는 {:.0f}th percentile (백테스트 {}% 거래가 더 나쁨)'.format(live_pnl, pct, pct))
print()

print('Win Rate %:')
print('  최악: {:.1f}%  평균: {:.1f}%  최고: {:.1f}%'.format(window_wrs.min(), window_wrs.mean(), window_wrs.max()))
print('  P5:   {:.1f}%  P25: {:.1f}%  P50: {:.1f}%  P75: {:.1f}%  P95: {:.1f}%'.format(
    np.percentile(window_wrs, 5), np.percentile(window_wrs, 25), np.percentile(window_wrs, 50),
    np.percentile(window_wrs, 75), np.percentile(window_wrs, 95)))
pct_wr = (window_wrs <= live_wr).sum() / n_win * 100
print('  라이브 WR {:.1f}%는 {:.0f}th percentile'.format(live_wr, pct_wr))
print()

print('Max Drawdown %:')
print('  최소: {:.1f}%  평균: {:.1f}%  최대: {:.1f}%'.format(window_max_dds.min(), window_max_dds.mean(), window_max_dds.max()))
print('  P5:   {:.1f}%  P25: {:.1f}%  P50: {:.1f}%  P75: {:.1f}%  P95: {:.1f}%'.format(
    np.percentile(window_max_dds, 5), np.percentile(window_max_dds, 25), np.percentile(window_max_dds, 50),
    np.percentile(window_max_dds, 75), np.percentile(window_max_dds, 95)))
pct_dd = (window_max_dds >= live_dd).sum() / n_win * 100
print('  라이브 DD {:.1f}%는 상위 {:.0f}% (백테스트 {}% 윈도우가 더 깊은 DD)'.format(live_dd, pct_dd, pct_dd))
print()

# Negative windows
neg_windows = (window_pnls < 0).sum()
print('음수 PnL 윈도우: {}/{} ({:.1f}%)'.format(neg_windows, n_win, neg_windows/n_win*100))
loss_windows = (window_pnls < -3).sum()
print('-3% 이하 윈도우: {}/{} ({:.1f}%)'.format(loss_windows, n_win, loss_windows/n_win*100))
deep_loss = (window_pnls < -10).sum()
print('-10% 이하 윈도우: {}/{} ({:.1f}%)'.format(deep_loss, n_win, deep_loss/n_win*100))

result = {
    'live': {'pnl': live_pnl, 'wr': live_wr, 'max_dd': live_dd, 'n_trades': 13},
    'backtest_windows': {
        'n_windows': n_win,
        'pnl': {'min': float(window_pnls.min()), 'max': float(window_pnls.max()),
                'mean': float(window_pnls.mean()), 'std': float(window_pnls.std()),
                'p5': float(np.percentile(window_pnls, 5)), 'p50': float(np.percentile(window_pnls, 50)),
                'p95': float(np.percentile(window_pnls, 95))},
        'wr': {'min': float(window_wrs.min()), 'max': float(window_wrs.max()),
               'mean': float(window_wrs.mean())},
        'max_dd': {'min': float(window_max_dds.min()), 'max': float(window_max_dds.max()),
                   'mean': float(window_max_dds.mean())},
    },
    'live_vs_backtest': {
        'pnl_percentile': float(pct),
        'wr_percentile': float(pct_wr),
        'dd_percentile_top': float(pct_dd),
    }
}
with open('results/live_window_analysis.json', 'w') as f:
    json.dump(result, f, indent=2)
