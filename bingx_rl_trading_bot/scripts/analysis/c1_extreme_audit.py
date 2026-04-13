#!/usr/bin/env python3
"""Verify extreme PnL trades — how are they possible with trailing stop?"""
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
agg = df.groupby('group').agg(timestamp=('timestamp','first'), open=('open','first'),
    high=('high','max'), low=('low','min'), close=('close','last')).reset_index(drop=True)
o = agg['open'].tolist(); h = agg['high'].tolist()
l = agg['low'].tolist(); c = agg['close'].tolist()
ts = agg['timestamp'].values; n = len(c); FEE = 0.10
cfg = load_config(); sig = C1BreakoutSignal(cfg['strategy'])
a14 = compute_atr(h, l, c, 14)
ch_h, ch_l = compute_channel(h, l, 15)
sw_l, sw_h = compute_fractal_swings(h, l, 10)

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
            td = sig.trail_K * a14[bar] / c[bar] * 100 if not math.isnan(a14[bar]) and a14[bar] > 0 else 0
            if pos['d'] == 'LONG': best_pnl = (pos['bp'] / pos['ep'] - 1) * 100
            else: best_pnl = (1 - pos['bp'] / pos['ep']) * 100
            trades.append(dict(d=pos['d'], ep=pos['ep'], xp=xp, raw=raw, net=raw - FEE,
                reason=ex['reason'], bh=pos['bh'], bar=bar, ebar=pos['ebar'],
                bp=pos['bp'], sl=pos['sl'], best_pnl=best_pnl, trail_dist=td,
                atr=a14[bar], entry_ts=str(ts[pos['ebar']])[:16],
                exit_ts=str(ts[bar])[:16]))
            pos = None; last_exit = bar; continue
    if pos is None and bar - last_exit >= 2:
        e = sig.check_entry(o[bar], h[bar], l[bar], c[bar],
                            ch_h[bar], ch_l[bar], a14[bar], sw_l[bar], sw_h[bar])
        if e and bar + 1 < n:
            pos = dict(d=e['direction'], ep=o[bar + 1], sl=e['sl_price'],
                       bp=o[bar + 1], bh=0, ebar=bar + 1)

print('=' * 80)
print('  극단 분포 거래 검증')
print('=' * 80)

# ── 1. 극단 수익 (net > +3%) ──
print('\n  [1] 큰 수익 거래 (net > +3%):')
big_wins = sorted([t for t in trades if t['net'] > 3], key=lambda x: -x['net'])
print(f'  {"#":>3} {"dir":>5} {"entry":>10} {"best":>10} {"exit":>10} '
      f'{"best%":>7} {"trail%":>7} {"raw%":>7} {"reason":>8} {"bars":>5} {"ATR":>6}')
for i, t in enumerate(big_wins):
    print(f'  {i+1:>3} {t["d"]:>5} {t["ep"]:>10.1f} {t["bp"]:>10.1f} {t["xp"]:>10.1f} '
          f'{t["best_pnl"]:>+6.2f}% {t["trail_dist"]:>6.2f}% {t["raw"]:>+6.2f}% '
          f'{t["reason"]:>8} {t["bh"]:>4}b ${t["atr"]:>5.0f}')

print(f'\n  수학적 검증 (realized = max(0, best_pnl - trail_dist)):')
all_match = True
for i, t in enumerate(big_wins):
    realized = max(0, t['best_pnl'] - t['trail_dist'])
    if t['d'] == 'LONG':
        expected_xp = t['ep'] * (1 + realized / 100)
    else:
        expected_xp = t['ep'] * (1 - realized / 100)
    match = abs(t['xp'] - expected_xp) < 1.0
    if not match: all_match = False
    print(f'    #{i+1}: best={t["best_pnl"]:+.2f}% - trail={t["trail_dist"]:.2f}% '
          f'= realized={realized:.2f}% | raw={t["raw"]:+.2f}% | '
          f'xp_exp={expected_xp:.1f} xp_act={t["xp"]:.1f} {"OK" if match else "**MISMATCH**"}')
print(f'  전체 일치: {"PASS" if all_match else "FAIL"}')

# ── 2. 극단 손실 (net < -2%) ──
print(f'\n  [2] 큰 손실 거래 (net < -2%):')
big_losses = sorted([t for t in trades if t['net'] < -2], key=lambda x: x['net'])
print(f'  {"#":>3} {"dir":>5} {"entry":>10} {"SL":>10} {"exit":>10} '
      f'{"raw%":>7} {"net%":>7} {"reason":>10} {"bars":>5} {"SL dist%":>8}')
for i, t in enumerate(big_losses):
    sl_dist = abs(t['ep'] - t['sl']) / t['ep'] * 100
    print(f'  {i+1:>3} {t["d"]:>5} {t["ep"]:>10.1f} {t["sl"]:>10.1f} {t["xp"]:>10.1f} '
          f'{t["raw"]:>+6.2f}% {t["net"]:>+6.2f}% {t["reason"]:>10} {t["bh"]:>4}b '
          f'{sl_dist:>7.2f}%')

# Emergency 상세
emg = [t for t in trades if t['reason'] == 'EMERGENCY']
print(f'\n  Emergency 거래: {len(emg)}건')
for t in emg:
    sl_dist = abs(t['ep'] - t['sl']) / t['ep'] * 100
    print(f'    {t["d"]} ep={t["ep"]:.1f} sl={t["sl"]:.1f} (SL dist={sl_dist:.2f}%)')
    print(f'    xp={t["xp"]:.1f} raw={t["raw"]:+.4f}%')
    print(f'    SL이 {sl_dist:.2f}% > Emergency 3.0% → Emergency가 먼저 발동')

# ── 3. 이론적 한계 ──
print(f'\n  [3] 이론적 한계 vs 실제')
all_nets = [t['net'] for t in trades]
all_raws = [t['raw'] for t in trades]
sl_raws = [t['raw'] for t in trades if t['reason'] == 'SL']
tp_raws = [t['raw'] for t in trades if t['reason'] == 'TRAIL_TP']

print(f'  최대 단일 손실 (raw):  {min(all_raws):+.4f}% (이론한계: -3.00% Emergency)')
print(f'  최대 SL 손실 (raw):    {min(sl_raws):+.4f}% (이론한계: -3.00% sl_max_pct)')
print(f'  최대 단일 수익 (raw):  {max(all_raws):+.4f}% (이론한계: 무한 - trail은 상한이 아님)')
print(f'  최대 단일 수익 (net):  {max(all_nets):+.4f}%')

# ── 4. Trail이 수익을 제한하지 않는 원리 설명 ──
print(f'\n  [4] Trail 메커니즘 원리')
print(f'  =' * 35)
t_best = max(trades, key=lambda x: x['raw'])
print(f'  최대 수익 거래: {t_best["d"]}')
print(f'    진입:     ${t_best["ep"]:,.1f} ({t_best["entry_ts"]})')
print(f'    best:     ${t_best["bp"]:,.1f} (best_pnl = {t_best["best_pnl"]:+.2f}%)')
print(f'    청산:     ${t_best["xp"]:,.1f} (raw = {t_best["raw"]:+.2f}%)')
print(f'    trail:    {t_best["trail_dist"]:.2f}% (청산 시점)')
print(f'    보유:     {t_best["bh"]}봉 ({t_best["bh"]*15/60:.1f}시간)')
print(f'')
print(f'    동작:')
print(f'    1. 진입 ${t_best["ep"]:,.0f}')
print(f'    2. 추세 → best 갱신 반복 → trail이 best를 따라 올라감')
print(f'    3. best ${t_best["bp"]:,.0f} 도달 ({t_best["best_pnl"]:+.1f}%)')
print(f'    4. 되돌림 >= {t_best["trail_dist"]:.1f}% → trail 발동')
print(f'    5. 실현 = {t_best["best_pnl"]:.1f}% - {t_best["trail_dist"]:.1f}% = {t_best["raw"]:.1f}%')
print(f'')
print(f'    Trail = "수익 상한"이 아니라 "되돌림 허용폭"')
print(f'    추세가 강하면 best가 무한 갱신 → 수익도 무한 확장 가능')

# ── 5. 수익 상위 20건의 best_pnl vs trail_dist ──
print(f'\n  [5] 수익 상위 20건: best_pnl vs trail_dist')
top20 = sorted(trades, key=lambda x: -x['raw'])[:20]
print(f'  {"#":>3} {"best%":>8} {"trail%":>8} {"raw%":>8} {"bars":>5} {"ratio":>6}')
for i, t in enumerate(top20):
    ratio = t['best_pnl'] / t['trail_dist'] if t['trail_dist'] > 0 else 0
    print(f'  {i+1:>3} {t["best_pnl"]:>+7.2f}% {t["trail_dist"]:>7.2f}% '
          f'{t["raw"]:>+7.2f}% {t["bh"]:>4}b {ratio:>5.1f}x')
print(f'\n  ratio = best_pnl / trail_dist')
print(f'  ratio가 클수록 추세가 trail 대비 길게 지속된 것')
print(f'  평균 ratio: {np.mean([t["best_pnl"]/t["trail_dist"] for t in top20 if t["trail_dist"]>0]):.1f}x')
