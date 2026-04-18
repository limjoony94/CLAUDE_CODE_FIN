"""Liquidation risk modeling for C1 Breakout v2 at 3x/10x leverage."""
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
O, H, L, C = df['open'].values, df['high'].values, df['low'].values, df['close'].values


def atr_calc(h, l, c, p=14):
    tr = [h[0] - l[0]] + [max(h[i] - l[i], abs(h[i] - c[i-1]), abs(l[i] - c[i-1])) for i in range(1, len(c))]
    a = [float('nan')] * len(c)
    if len(c) >= p:
        a[p-1] = sum(tr[:p]) / p
        for i in range(p, len(c)):
            a[i] = (a[i-1] * (p-1) + tr[i]) / p
    return a


def ch_c(h, l, p=15):
    ch = [float('nan')] * len(h)
    cl = [float('nan')] * len(h)
    for i in range(p, len(h)):
        ch[i] = max(h[i-p:i])
        cl[i] = min(l[i-p:i])
    return ch, cl


def fr_c(h, l, lb=10):
    sl = [float('nan')] * len(h)
    sh = [float('nan')] * len(h)
    cs = float('nan')
    ch_ = float('nan')
    for i in range(lb, len(h)):
        if l[i] == min(l[i-lb:i+1]):
            cs = l[i]
        if h[i] == max(h[i-lb:i+1]):
            ch_ = h[i]
        sl[i] = cs
        sh[i] = ch_
    return sl, sh


ATR = atr_calc(H, L, C)
CH, CL = ch_c(H, L)
SWL, SWH = fr_c(H, L)

LIQ_THRESHOLD = 0.095

trades = []
pos = None
bars_since = 999

for bar in range(25, n-1):
    a = ATR[bar]
    if math.isnan(a) or a <= 0:
        continue
    bars_since += 1
    if pos:
        pos['bh'] += 1
        if pos['d'] == 'LONG':
            pos['bp'] = max(pos['bp'], H[bar])
            pos['worst'] = min(pos['worst'], L[bar])
        else:
            pos['bp'] = min(pos['bp'], L[bar])
            pos['worst'] = max(pos['worst'], H[bar])

        ex = None
        reason = None
        if pos['d'] == 'LONG' and L[bar] <= pos['sl']:
            ex = (pos['sl']/pos['ep']-1) * 100
            reason = 'SL'
        elif pos['d'] == 'SHORT' and H[bar] >= pos['sl']:
            ex = (1 - pos['sl']/pos['ep']) * 100
            reason = 'SL'

        if ex is None:
            wp = (L[bar]/pos['ep']-1) * 100 if pos['d'] == 'LONG' else (1 - H[bar]/pos['ep']) * 100
            if wp <= -3.0:
                ex = -3.0
                reason = 'EMERG'

        if ex is None and pos['bh'] >= 192:
            ex = (C[bar]/pos['ep']-1) * 100 if pos['d'] == 'LONG' else (1 - C[bar]/pos['ep']) * 100
            reason = 'TIMEOUT'

        if ex is None:
            bpnl = (pos['bp']/pos['ep']-1) * 100 if pos['d'] == 'LONG' else (1 - pos['bp']/pos['ep']) * 100
            cpnl = (C[bar]/pos['ep']-1) * 100 if pos['d'] == 'LONG' else (1 - C[bar]/pos['ep']) * 100
            if bpnl > 0.05 and a > 0:
                td = 2.5 * a / C[bar] * 100
                dd = bpnl - cpnl
                if dd >= td:
                    ex = max(0, bpnl - td)
                    reason = 'TRAIL'

        if ex is not None:
            if pos['d'] == 'LONG':
                max_adverse = (pos['ep'] - pos['worst']) / pos['ep']
            else:
                max_adverse = (pos['worst'] - pos['ep']) / pos['ep']

            trades.append({
                'pnl': ex - 0.10,
                'reason': reason,
                'dir': pos['d'],
                'bars': pos['bh'],
                'max_adverse': max_adverse,
                'entry': pos['ep'],
                'worst': pos['worst'],
                'sl_pct': abs(pos['ep']-pos['sl'])/pos['ep'],
            })
            pos = None
            bars_since = 0
            continue

    if pos is None and bars_since >= 2:
        ch = CH[bar]
        cl_ = CL[bar]
        if math.isnan(ch) or math.isnan(cl_) or math.isnan(a):
            continue
        d = None
        if C[bar] > ch:
            d = 'LONG'
        elif C[bar] < cl_:
            d = 'SHORT'
        if d:
            rng = H[bar] - L[bar]
            if rng <= 0:
                continue
            body = C[bar] - O[bar]
            if abs(body) / rng < 0.4:
                continue
            if d == 'LONG' and body <= 0:
                continue
            if d == 'SHORT' and body >= 0:
                continue
            ep = O[bar+1]
            if d == 'LONG':
                fsl = SWL[bar] if not math.isnan(SWL[bar]) else ep - 3.3 * a
                sl = max(fsl, ep - 3.3 * a)
            else:
                fsl = SWH[bar] if not math.isnan(SWH[bar]) else ep + 3.3 * a
                sl = min(fsl, ep + 3.3 * a)
            if abs(ep-sl)/ep*100 < 0.15 or abs(ep-sl)/ep*100 > 3.0:
                continue
            pos = {'d': d, 'ep': ep, 'sl': sl, 'bp': ep, 'worst': ep, 'bh': 0}

n_trades = len(trades)
total_pnl = sum(t['pnl'] for t in trades)
maes = [t['max_adverse']*100 for t in trades]
liq_at_risk = [t for t in trades if t['max_adverse'] >= LIQ_THRESHOLD]
winners = [t for t in trades if t['pnl'] > 0]
losers = [t for t in trades if t['pnl'] <= 0]

print('=== Backtest Summary ({} trades, additive 1x) ==='.format(n_trades))
print('Total PnL: {:+.1f}%'.format(total_pnl))
print()
print('=== MAE Distribution ===')
print('  Mean:   {:.2f}%'.format(np.mean(maes)))
print('  Median: {:.2f}%'.format(np.median(maes)))
print('  P90:    {:.2f}%'.format(np.percentile(maes, 90)))
print('  P95:    {:.2f}%'.format(np.percentile(maes, 95)))
print('  P99:    {:.2f}%'.format(np.percentile(maes, 99)))
print('  Max:    {:.2f}%'.format(max(maes)))
print()
print('=== Liquidation Risk (3x/10x, ~9.5% threshold) ===')
print('At risk: {}/{} ({:.2f}%)'.format(len(liq_at_risk), n_trades, len(liq_at_risk)/n_trades*100))

for t in liq_at_risk[:5]:
    print('  {} entry=${:.0f} MAE={:.2f}% SL={:.2f}% reason={}'.format(
        t['dir'], t['entry'], t['max_adverse']*100, t['sl_pct']*100, t['reason']))
if len(liq_at_risk) > 5:
    print('  +{} more'.format(len(liq_at_risk)-5))

print()
print('=== MAE by Outcome ===')
print('  Winners median MAE: {:.2f}%'.format(np.median([t['max_adverse']*100 for t in winners])))
print('  Losers median MAE:  {:.2f}%'.format(np.median([t['max_adverse']*100 for t in losers])))
print()

# SL distribution stats
sl_pcts = [t['sl_pct']*100 for t in trades]
print('=== SL Distance Distribution ===')
print('  Mean:   {:.2f}%'.format(np.mean(sl_pcts)))
print('  Max:    {:.2f}%'.format(max(sl_pcts)))
print()

# Exit reason breakdown
from collections import Counter
reasons = Counter(t['reason'] for t in trades)
print('=== Exit Reasons ===')
for r, c in reasons.most_common():
    print('  {}: {} ({:.1f}%)'.format(r, c, c/n_trades*100))

# Save to JSON
result = {
    'n_trades': n_trades,
    'total_pnl': total_pnl,
    'mae_stats': {
        'mean': float(np.mean(maes)),
        'median': float(np.median(maes)),
        'p90': float(np.percentile(maes, 90)),
        'p95': float(np.percentile(maes, 95)),
        'p99': float(np.percentile(maes, 99)),
        'max': float(max(maes)),
    },
    'liquidation_risk': {
        'threshold_pct': LIQ_THRESHOLD*100,
        'at_risk_count': len(liq_at_risk),
        'at_risk_pct': len(liq_at_risk)/n_trades*100,
    },
    'exit_reasons': dict(reasons),
}
with open('results/liquidation_risk_check.json', 'w') as f:
    json.dump(result, f, indent=2)

print('\nResults saved to results/liquidation_risk_check.json')
