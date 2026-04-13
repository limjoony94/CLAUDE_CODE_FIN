#!/usr/bin/env python3
"""Emergency vs SL priority analysis — does loss exceed SL?"""
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
ts = agg['timestamp'].values; n = len(c)
cfg = load_config(); sig = C1BreakoutSignal(cfg['strategy'])
a14 = compute_atr(h, l, c, 14)
ch_h2, ch_l2 = compute_channel(h, l, 15)
sw_l2, sw_h2 = compute_fractal_swings(h, l, 10)

# 모든 거래 + 동시 발동 체크
all_trades = []; pos = None; last_exit = -3
for bar in range(26, n - 1):
    if pos is not None:
        if pos['d'] == 'LONG': pos['bp'] = max(pos['bp'], h[bar])
        else: pos['bp'] = min(pos['bp'], l[bar])
        pos['bh'] += 1

        # 각 조건 개별 체크
        d = pos['d']; ep = pos['ep']; sl = pos['sl']
        if d == 'LONG': worst = (l[bar] / ep - 1) * 100
        else: worst = (1 - h[bar] / ep) * 100
        emg_hit = worst <= -3.0
        if d == 'LONG': sl_hit = l[bar] <= sl
        else: sl_hit = h[bar] >= sl

        ex = sig.check_exit(d, ep, pos['bp'], h[bar], l[bar], c[bar], sl, a14[bar], pos['bh'])
        if ex:
            xp = ex['exit_price']
            if d == 'LONG': raw = (xp / ep - 1) * 100
            else: raw = (1 - xp / ep) * 100
            sl_dist = abs(ep - sl) / ep * 100
            all_trades.append(dict(
                d=d, ep=ep, xp=xp, raw=raw, net=raw - 0.10,
                reason=ex['reason'], bh=pos['bh'], bar=bar,
                sl=sl, sl_dist=sl_dist,
                bar_h=h[bar], bar_l=l[bar],
                emg_hit=emg_hit, sl_hit=sl_hit,
                both_hit=emg_hit and sl_hit,
                entry_ts=str(ts[pos['ebar']])[:19],
                exit_ts=str(ts[bar])[:19],
            ))
            pos = None; last_exit = bar; continue
    if pos is None and bar - last_exit >= 2:
        e = sig.check_entry(o[bar], h[bar], l[bar], c[bar],
                            ch_h2[bar], ch_l2[bar], a14[bar], sw_l2[bar], sw_h2[bar])
        if e and bar + 1 < n:
            pos = dict(d=e['direction'], ep=o[bar + 1], sl=e['sl_price'],
                       bp=o[bar + 1], bh=0, ebar=bar + 1)

print('=' * 70)
print('  Emergency vs SL 우선순위 충돌 분석')
print('=' * 70)

# 1. 같은 봉에서 Emergency + SL 동시 발동한 거래
both = [t for t in all_trades if t['both_hit']]
emg_only = [t for t in all_trades if t['emg_hit'] and not t['sl_hit']]
sl_only = [t for t in all_trades if t['sl_hit'] and not t['emg_hit']]

print(f'\n  동시 발동 통계:')
print(f'    Emergency만 발동:   {len(emg_only)}건')
print(f'    SL만 발동:          {len(sl_only)}건')
print(f'    Emergency+SL 동시:  {len(both)}건')
print(f'    총 SL 관련:         {len([t for t in all_trades if t["reason"] in ("SL","EMERGENCY")])}건')

if both:
    print(f'\n  동시 발동 거래 상세:')
    for i, t in enumerate(both):
        # SL이 먼저 체결되었다면의 가격
        sl_raw = -t['sl_dist']  # SL에서 청산 시 raw%
        emg_raw = -3.0  # Emergency raw%
        diff = emg_raw - sl_raw  # 차이 (Emergency가 더 나쁨)

        print(f'\n    --- 거래 #{i+1} ---')
        print(f'    {t["d"]} {t["entry_ts"]} ~ {t["exit_ts"]}')
        print(f'    진입가:     ${t["ep"]:,.1f}')

        if t['d'] == 'SHORT':
            emg_price = t['ep'] * (1 + 3.0 / 100)
            print(f'    SL 가격:    ${t["sl"]:,.1f}  (거리 {t["sl_dist"]:.2f}%)')
            print(f'    EMG 가격:   ${emg_price:,.1f}  (거리 3.00%)')
            print(f'    봉 high:    ${t["bar_h"]:,.1f}')
            print(f'')
            print(f'    가격 순서 (SHORT, 아래→위 = 나쁨):')
            print(f'      ${t["ep"]:>12,.1f}  진입')
            print(f'      ${t["sl"]:>12,.1f}  SL       ← 여기서 먼저 체결됨')
            print(f'      ${emg_price:>12,.1f}  Emergency ← 코드는 여기서 기록')
            print(f'      ${t["bar_h"]:>12,.1f}  봉 최고가')
        else:
            emg_price = t['ep'] * (1 - 3.0 / 100)
            print(f'    SL 가격:    ${t["sl"]:,.1f}  (거리 {t["sl_dist"]:.2f}%)')
            print(f'    EMG 가격:   ${emg_price:,.1f}  (거리 3.00%)')
            print(f'    봉 low:     ${t["bar_l"]:,.1f}')
            print(f'')
            print(f'    가격 순서 (LONG, 위→아래 = 나쁨):')
            print(f'      ${t["ep"]:>12,.1f}  진입')
            print(f'      ${t["sl"]:>12,.1f}  SL       ← 여기서 먼저 체결됨')
            print(f'      ${emg_price:>12,.1f}  Emergency ← 코드는 여기서 기록')
            print(f'      ${t["bar_l"]:>12,.1f}  봉 최저가')

        print(f'')
        print(f'    실제 기록: {t["reason"]} raw={t["raw"]:+.2f}%')
        print(f'    SL였다면:  SL raw={sl_raw:+.2f}%')
        print(f'    차이:      {abs(diff):.2f}%p 더 나쁘게 기록 (비관적)')

print(f'\n{"=" * 70}')
print('  결론')
print('=' * 70)
print(f'  Emergency+SL 동시 발동: {len(both)}건')
if both:
    total_diff = sum(3.0 - t['sl_dist'] for t in both)
    print(f'  비관적 PnL 차이 합계: {total_diff:+.2f}%p')
    print(f'  (실제 시장에서는 SL이 먼저 체결 → 이 차이만큼 실제가 더 좋음)')
    print(f'')
    print(f'  원인: check_exit 우선순위')
    print(f'    1순위 Emergency: worst_pnl 체크 (봉 내 최악 가격 기준)')
    print(f'    2순위 SL:        high/low vs sl_price 비교')
    print(f'')
    print(f'    같은 봉에서 SL($-2.9%)과 Emergency($-3.0%)가 동시 도달 시')
    print(f'    코드는 Emergency(-3.0%)를 선택 → 실제보다 나쁜 가격으로 기록')
    print(f'')
    print(f'  영향: 백테스트가 {total_diff:.2f}%p 비관적 (전체 PnL 169%에서 무시 가능)')
    print(f'  프로덕션: Exchange STOP_MARKET이 SL 가격에서 먼저 체결 → 문제 없음')
else:
    print(f'  동시 발동 없음 — Emergency가 SL을 넘어서는 문제 없음')

# 추가: SL 거래 중 SL 가격을 넘어선 것이 있는가?
print(f'\n{"=" * 70}')
print('  SL 거래 — 청산가가 SL 가격과 정확히 일치하는가?')
print('=' * 70)
sl_trades = [t for t in all_trades if t['reason'] == 'SL']
exact = 0; off = 0
for t in sl_trades:
    if abs(t['xp'] - t['sl']) < 0.01:
        exact += 1
    else:
        off += 1
        print(f'  MISMATCH: {t["d"]} xp={t["xp"]:.1f} sl={t["sl"]:.1f} diff=${t["xp"]-t["sl"]:.1f}')
print(f'  SL 가격 정확 일치: {exact}/{len(sl_trades)}')
print(f'  불일치: {off}/{len(sl_trades)}')
if exact == len(sl_trades):
    print(f'  → 모든 SL이 정확한 SL 가격에서 청산 (슬리피지 없음 = 백테스트 가정)')
