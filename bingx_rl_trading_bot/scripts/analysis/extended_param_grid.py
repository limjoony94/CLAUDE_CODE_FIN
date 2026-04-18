"""Extended parameter grid test (+/-50%) for C1 Breakout v2."""
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
    a = [float('nan')] * len(c)
    if len(c) >= p:
        a[p-1] = sum(tr[:p])/p
        for i in range(p, len(c)):
            a[i] = (a[i-1]*(p-1) + tr[i])/p
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


def run_backtest(channel_p, trail_k, max_sl_atr, body_min, atr_p):
    ATR = atr_calc(H_, L_, C_, atr_p)
    CH, CL = ch_c(H_, L_, channel_p)
    SWL, SWH = fr_c(H_, L_, 10)

    trades = []
    pos = None; bars_since = 999

    for bar in range(max(25, channel_p+1, atr_p+1), n-1):
        a = ATR[bar]
        if math.isnan(a) or a <= 0: continue
        bars_since += 1

        if pos:
            pos['bh'] += 1
            if pos['d'] == 'LONG':
                pos['bp'] = max(pos['bp'], H_[bar])
            else:
                pos['bp'] = min(pos['bp'], L_[bar])

            ex = None
            if pos['d'] == 'LONG' and L_[bar] <= pos['sl']:
                ex = (pos['sl']/pos['ep']-1)*100
            elif pos['d'] == 'SHORT' and H_[bar] >= pos['sl']:
                ex = (1-pos['sl']/pos['ep'])*100

            if ex is None:
                wp = (L_[bar]/pos['ep']-1)*100 if pos['d']=='LONG' else (1-H_[bar]/pos['ep'])*100
                if wp <= -3.0: ex = -3.0

            if ex is None and pos['bh'] >= 192:
                ex = (C_[bar]/pos['ep']-1)*100 if pos['d']=='LONG' else (1-C_[bar]/pos['ep'])*100

            if ex is None:
                bpnl = (pos['bp']/pos['ep']-1)*100 if pos['d']=='LONG' else (1-pos['bp']/pos['ep'])*100
                cpnl = (C_[bar]/pos['ep']-1)*100 if pos['d']=='LONG' else (1-C_[bar]/pos['ep'])*100
                if bpnl > 0.05 and a > 0:
                    td = trail_k * a / C_[bar] * 100
                    dd = bpnl - cpnl
                    if dd >= td: ex = max(0, bpnl-td)

            if ex is not None:
                trades.append(ex - 0.10)
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
                if abs(body)/rng < body_min: continue
                if d == 'LONG' and body <= 0: continue
                if d == 'SHORT' and body >= 0: continue
                ep = O_[bar+1]
                if d == 'LONG':
                    fsl = SWL[bar] if not math.isnan(SWL[bar]) else ep - max_sl_atr*a
                    sl = max(fsl, ep - max_sl_atr*a)
                else:
                    fsl = SWH[bar] if not math.isnan(SWH[bar]) else ep + max_sl_atr*a
                    sl = min(fsl, ep + max_sl_atr*a)
                if abs(ep-sl)/ep*100 < 0.15 or abs(ep-sl)/ep*100 > 3.0: continue
                pos = {'d': d, 'ep': ep, 'sl': sl, 'bp': ep, 'bh': 0}

    return sum(trades), len(trades)


# Baseline
baseline_pnl, baseline_n = run_backtest(15, 2.5, 3.3, 0.4, 14)
print('Baseline: PnL={:+.1f}%, Trades={}'.format(baseline_pnl, baseline_n))
print()

# Extended grid (~+/-50%)
grids = {
    'channel_period': [8, 10, 12, 15, 18, 20, 23],
    'trail_K':        [1.3, 1.8, 2.2, 2.5, 2.8, 3.3, 3.8],
    'max_sl_atr':     [2.0, 2.5, 3.0, 3.3, 3.6, 4.0, 5.0],
    'body_min_ratio': [0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55],
    'atr_period':     [8, 10, 12, 14, 16, 18, 20],
}

results = {}
for param, values in grids.items():
    print('=== {} ==='.format(param))
    pnls = []
    for v in values:
        args = {'channel_p': 15, 'trail_k': 2.5, 'max_sl_atr': 3.3, 'body_min': 0.4, 'atr_p': 14}
        key_map = {'channel_period': 'channel_p', 'trail_K': 'trail_k', 'max_sl_atr': 'max_sl_atr',
                   'body_min_ratio': 'body_min', 'atr_period': 'atr_p'}
        args[key_map[param]] = v
        pnl, ntr = run_backtest(**args)
        pnls.append((v, pnl, ntr))
        mark = '*' if v == {'channel_period': 15, 'trail_K': 2.5, 'max_sl_atr': 3.3, 'body_min_ratio': 0.4, 'atr_period': 14}[param] else ' '
        print('  {}{:>6}: PnL={:>+7.1f}%, Trades={}'.format(mark, v, pnl, ntr))

    pos_count = sum(1 for _, p, _ in pnls if p > 0)
    results[param] = {
        'values': [v for v, _, _ in pnls],
        'pnls': [p for _, p, _ in pnls],
        'trades': [t for _, _, t in pnls],
        'positive_count': pos_count,
        'total_count': len(pnls),
        'min_pnl': min(p for _, p, _ in pnls),
        'max_pnl': max(p for _, p, _ in pnls),
    }
    print('  → {}/{} 양수, 범위 [{:.1f}%, {:.1f}%]'.format(pos_count, len(pnls), results[param]['min_pnl'], results[param]['max_pnl']))
    print()

# Overall verdict
total_tests = sum(r['total_count'] for r in results.values())
total_pos = sum(r['positive_count'] for r in results.values())
print('=== Overall (+/-50% extended grid) ===')
print('{}/{} configurations positive ({:.1f}%)'.format(total_pos, total_tests, total_pos/total_tests*100))
min_overall = min(r['min_pnl'] for r in results.values())
print('Worst case: {:+.1f}%'.format(min_overall))

if total_pos == total_tests:
    verdict = 'ROBUST'
elif total_pos / total_tests >= 0.9:
    verdict = 'STRONG'
elif total_pos / total_tests >= 0.7:
    verdict = 'MARGINAL'
else:
    verdict = 'FRAGILE'
print('Verdict: {}'.format(verdict))

results['baseline'] = {'pnl': baseline_pnl, 'trades': baseline_n}
results['summary'] = {
    'total_configs': total_tests,
    'positive_configs': total_pos,
    'positive_pct': total_pos/total_tests*100,
    'min_pnl': min_overall,
    'verdict': verdict,
}
with open('results/extended_param_grid.json', 'w') as f:
    json.dump(results, f, indent=2)
