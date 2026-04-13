#!/usr/bin/env python3
"""v2.5 verification: SL-first priority + lookback=10 unified."""
import os, sys, math
import pandas as pd, numpy as np
os.chdir(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, '.')
from scripts.production.c1_breakout.indicators import compute_atr, compute_channel, compute_fractal_swings
from scripts.production.c1_breakout.signals import C1BreakoutSignal
from scripts.production.c1_breakout.config import load_config

df = pd.read_csv('data/btc_5m_270days_reclassified.csv')
df['timestamp'] = pd.to_datetime(df['timestamp'])
df['group'] = df.index // 3
agg = df.groupby('group').agg(open=('open','first'), high=('high','max'),
    low=('low','min'), close=('close','last')).reset_index(drop=True)
o = agg['open'].tolist(); h = agg['high'].tolist()
l = agg['low'].tolist(); c = agg['close'].tolist()
n = len(c); FEE = 0.10

cfg = load_config()
sig = C1BreakoutSignal(cfg['strategy'])
a14 = compute_atr(h, l, c, 14)
ch_h, ch_l = compute_channel(h, l, 15)
sw_l, sw_h = compute_fractal_swings(h, l, 10)

# Backtest with v2.5 (SL-first priority, lookback=10)
trades = []; pos = None; last_exit = -3
for bar in range(26, n - 1):
    if pos is not None:
        if pos['d'] == 'LONG': pos['bp'] = max(pos['bp'], h[bar])
        else: pos['bp'] = min(pos['bp'], l[bar])
        pos['bh'] += 1
        ex = sig.check_exit(pos['d'], pos['ep'], pos['bp'],
                            h[bar], l[bar], c[bar], pos['sl'], a14[bar], pos['bh'])
        if ex:
            xp = ex['exit_price']
            if pos['d'] == 'LONG': raw = (xp / pos['ep'] - 1) * 100
            else: raw = (1 - xp / pos['ep']) * 100
            trades.append(dict(raw=raw, net=raw - FEE, reason=ex['reason'],
                               bh=pos['bh'], d=pos['d']))
            pos = None; last_exit = bar; continue
    if pos is None and bar - last_exit >= 2:
        e = sig.check_entry(o[bar], h[bar], l[bar], c[bar],
                            ch_h[bar], ch_l[bar], a14[bar], sw_l[bar], sw_h[bar])
        if e and bar + 1 < n:
            pos = dict(d=e['direction'], ep=o[bar + 1], sl=e['sl_price'],
                       bp=o[bar + 1], bh=0)

# Stats
pnls = [t['net'] for t in trades]
wins = [p for p in pnls if p > 0]
losses = [p for p in pnls if p <= 0]
total = sum(pnls)
eq = 0; pk = 0; mdd = 0
for p in pnls:
    eq += p; pk = max(pk, eq); mdd = max(mdd, pk - eq)
wr = len(wins) / len(pnls) * 100
wa = np.mean(wins) if wins else 0
la = np.mean(losses) if losses else -1
rr = abs(wa / la) if la else 0
nd = n * 15 / 1440

print('=' * 70)
print('  C1 Breakout v2.5 — Verification')
print('  SL-first priority + lookback=10 + production signals.py')
print('=' * 70)

print(f'\n  Baseline (1x additive, {nd:.0f}d):')
print(f'    Trades:    {len(trades)}')
print(f'    WR:        {wr:.1f}%')
print(f'    R:R:       {rr:.2f}')
print(f'    PnL:       {total:+.1f}%')
print(f'    MDD:       {mdd:.1f}%')
print(f'    Daily:     {total/nd:+.4f}%')

# Exit breakdown
reasons = {}
for t in trades:
    r = t['reason']
    if r not in reasons: reasons[r] = []
    reasons[r].append(t['net'])

print(f'\n  Exit breakdown:')
for r in sorted(reasons.keys()):
    plist = reasons[r]
    w = sum(1 for p in plist if p > 0)
    wr2 = w / len(plist) * 100
    print(f'    {r:>10}: {len(plist):>4}건 WR={wr2:.0f}% avg={np.mean(plist):+.4f}% sum={sum(plist):+.1f}%')

# Comparison vs previous (v2.4 lookback=10)
# v2.4 had Emergency-first: T=1028, WR=36.6%, PnL=+169.5%, MDD=5.5%
print(f'\n  vs v2.4 (Emergency-first, same lookback=10):')
print(f'    {"":>15} {"v2.4":>10} {"v2.5":>10} {"delta":>10}')
print(f'    {"Trades":>15} {"1028":>10} {len(trades):>10} {len(trades)-1028:>+10}')
print(f'    {"WR":>15} {"36.6%":>10} {wr:.1f}%{"":<4} {wr-36.6:>+9.1f}pp')
print(f'    {"PnL":>15} {"+169.5%":>10} {total:>+9.1f}% {total-169.5:>+9.1f}pp')
print(f'    {"MDD":>15} {"5.5%":>10} {mdd:>9.1f}% {mdd-5.5:>+9.1f}pp')

# Quick WF check
print(f'\n  Walk-Forward 5-fold (quick check):')
oos_pass = 0
for fi in range(5):
    ie = int(n * (fi + 1) / 6); oe = int(n * (fi + 2) / 6)
    a2 = compute_atr(h[:oe], l[:oe], c[:oe], 14)
    ch2, cl2 = compute_channel(h[:oe], l[:oe], 15)
    sl2, sh2 = compute_fractal_swings(h[:oe], l[:oe], 10)
    sig2 = C1BreakoutSignal(cfg['strategy'])
    oos_t = []; pos2 = None; le2 = -3
    for bar in range(max(26, ie), oe - 1):
        if pos2 is not None:
            if pos2['d'] == 'LONG': pos2['bp'] = max(pos2['bp'], h[bar])
            else: pos2['bp'] = min(pos2['bp'], l[bar])
            pos2['bh'] += 1
            ex2 = sig2.check_exit(pos2['d'], pos2['ep'], pos2['bp'],
                                  h[bar], l[bar], c[bar], pos2['sl'], a2[bar], pos2['bh'])
            if ex2:
                xp2 = ex2['exit_price']
                if pos2['d'] == 'LONG': raw2 = (xp2 / pos2['ep'] - 1) * 100
                else: raw2 = (1 - xp2 / pos2['ep']) * 100
                oos_t.append(raw2 - FEE)
                pos2 = None; le2 = bar; continue
        if pos2 is None and bar - le2 >= 2:
            e2 = sig2.check_entry(o[bar], h[bar], l[bar], c[bar],
                                  ch2[bar], cl2[bar], a2[bar], sl2[bar], sh2[bar])
            if e2 and bar + 1 < oe:
                pos2 = dict(d=e2['direction'], ep=o[bar + 1], sl=e2['sl_price'],
                            bp=o[bar + 1], bh=0)
    oos_pnl = sum(oos_t)
    p = 'P' if oos_pnl > 0 else 'F'
    if oos_pnl > 0: oos_pass += 1
    print(f'    F{fi+1}: OOS T={len(oos_t):>3} PnL={oos_pnl:>+6.1f}% [{p}]')
print(f'    Result: {oos_pass}/5')

# MC
print(f'\n  MC Direction Shuffle (999 sims):')
real_pnl = total
better = 0
rng = np.random.RandomState(42)
for _ in range(999):
    sp = sum((t['raw'] if rng.random() > 0.5 else -t['raw']) - FEE for t in trades)
    if sp >= real_pnl: better += 1
mc_p = better / 999
print(f'    p={mc_p:.4f} {"DISC" if mc_p < 0.05 else "N-DISC"}')
