#!/usr/bin/env python3
"""Verify loss handling, position sizing, drawdowns in C1 backtest."""
import os, sys
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
            trades.append(dict(d=pos['d'], ep=pos['ep'], xp=xp, raw=raw, net=raw - FEE,
                reason=ex['reason'], bh=pos['bh'], bar=bar, ebar=pos['ebar']))
            pos = None; last_exit = bar; continue
    if pos is None and bar - last_exit >= 2:
        e = sig.check_entry(o[bar], h[bar], l[bar], c[bar],
                            ch_h[bar], ch_l[bar], a14[bar], sw_l[bar], sw_h[bar])
        if e and bar + 1 < n:
            pos = dict(d=e['direction'], ep=o[bar + 1], sl=e['sl_price'],
                       bp=o[bar + 1], bh=0, ebar=bar + 1)

# ═══ 검증 5 후속: 전반/후반 raw% 차이 원인 ═══
print('=' * 70)
print('  검증 5 후속: 전반/후반 평균 |raw| 차이 원인')
print('=' * 70)
first_half = trades[:len(trades) // 2]
second_half = trades[len(trades) // 2:]

for label, subset in [('전반 514건', first_half), ('후반 514건', second_half)]:
    sl_t = [t for t in subset if t['reason'] == 'SL']
    tp_t = [t for t in subset if t['reason'] == 'TRAIL_TP']
    be_t = [t for t in tp_t if abs(t['raw']) < 0.01]
    win_t = [t for t in tp_t if t['net'] > 0]
    avg_sl = np.mean([t['raw'] for t in sl_t]) if sl_t else 0
    avg_win = np.mean([t['raw'] for t in win_t]) if win_t else 0
    print(f'\n  {label}:')
    print(f'    SL: {len(sl_t)}건, 평균 raw={avg_sl:+.4f}%')
    print(f'    Trail 수익: {len(win_t)}건, 평균 raw={avg_win:+.4f}%')
    print(f'    Trail BE:   {len(be_t)}건 (raw=0)')

mid_bar = trades[len(trades) // 2]['ebar']
print(f'\n  BTC 가격 범위:')
print(f'    전반: ${min(c[:mid_bar]):,.0f} ~ ${max(c[:mid_bar]):,.0f}')
print(f'    후반: ${min(c[mid_bar:]):,.0f} ~ ${max(c[mid_bar:]):,.0f}')

a14_1 = [a14[t['ebar']] for t in first_half if not np.isnan(a14[t['ebar']])]
a14_2 = [a14[t['ebar']] for t in second_half if not np.isnan(a14[t['ebar']])]
print(f'    전반 평균 ATR: ${np.mean(a14_1):,.0f}')
print(f'    후반 평균 ATR: ${np.mean(a14_2):,.0f}')
print(f'    ATR 변화가 |raw%| 차이의 원인 (마진 누적 아님)')

# ═══ 검증 9: SL raw% 분포 ═══
print(f'\n{"=" * 70}')
print('  검증 9: 모든 SL 거래의 raw% 분포')
print('=' * 70)
sl_raws = [t['raw'] for t in trades if t['reason'] == 'SL']
print(f'  SL 거래: {len(sl_raws)}건')
all_neg = all(r < 0 for r in sl_raws)
print(f'  모두 음수? {"YES" if all_neg else "NO"}')
print(f'  최소: {min(sl_raws):+.4f}%')
print(f'  최대: {max(sl_raws):+.4f}%')
print(f'  평균: {np.mean(sl_raws):+.4f}%')

# ═══ 검증 10: 연속 손실 & Drawdown ═══
print(f'\n{"=" * 70}')
print('  검증 10: 연속 손실 구간 & Drawdown')
print('=' * 70)
max_consec = 0; cur_consec = 0; max_dd_run = 0; dd_run = 0
for t in trades:
    if t['net'] <= 0:
        cur_consec += 1; dd_run += t['net']
        max_consec = max(max_consec, cur_consec)
        max_dd_run = min(max_dd_run, dd_run)
    else:
        cur_consec = 0; dd_run = 0
print(f'  최대 연속 손실: {max_consec}건')
print(f'  연속 손실 최대 누적: {max_dd_run:+.2f}%')

eq = 0; pk = 0; mdd = 0; dd_list = []
for t in trades:
    eq += t['net']; pk = max(pk, eq)
    dd = pk - eq; dd_list.append(dd); mdd = max(mdd, dd)
print(f'  최대 Drawdown (additive): {mdd:.2f}%')

print(f'\n  큰 Drawdown 구간 (>3%):')
in_dd = False; dd_start = 0
for i in range(len(trades)):
    if dd_list[i] > 3 and not in_dd:
        in_dd = True; dd_start = i
    elif dd_list[i] < 1 and in_dd:
        in_dd = False
        pk_dd = max(dd_list[dd_start:i + 1])
        print(f'    거래 #{dd_start + 1}~#{i}: peak DD={pk_dd:.2f}%, {i - dd_start}건')

# ═══ 검증 11: 거래별 수익/손실 완전한 분포 히스토그램 ═══
print(f'\n{"=" * 70}')
print('  검증 11: 거래 net% 분포 (텍스트 히스토그램)')
print('=' * 70)
nets = [t['net'] for t in trades]
bins = [(-10,-3), (-3,-2), (-2,-1), (-1,-0.5), (-0.5,-0.1), (-0.1,0),
        (0,0.5), (0.5,1), (1,2), (2,5), (5,10), (10,50)]
for lo, hi in bins:
    cnt = sum(1 for n2 in nets if lo <= n2 < hi)
    bar = '#' * (cnt // 5) if cnt > 0 else ''
    print(f'  [{lo:>+5.1f}, {hi:>+5.1f}): {cnt:>4} {bar}')
