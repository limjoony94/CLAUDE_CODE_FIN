#!/usr/bin/env python3
"""Compare backtest trail vs Exchange TRAILING_STOP_MARKET mechanics."""
import math

entry = 80000
best = 88000
atr = 400
trail_K = 2.5

print('=' * 70)
print('  Backtest Trail vs Exchange TRAILING_STOP_MARKET')
print('=' * 70)

print(f'\n  entry=${entry:,}, best=${best:,}, ATR=${atr}, trail_K={trail_K}')

# Backtest formula
print(f'\n  [Backtest (signals.py check_exit)]')
print(f'    best_pnl = (best/entry - 1) * 100')
print(f'    cur_pnl  = (current/entry - 1) * 100')
print(f'    drawdown = best_pnl - cur_pnl = (best - current) / entry * 100')
print(f'    trail_dist = trail_K * ATR / current * 100')
print(f'    trigger: drawdown >= trail_dist')

# Exchange formula
print(f'\n  [Exchange (BingX TRAILING_STOP_MARKET)]')
print(f'    callbackRate = round(trail_K * ATR / price * 100, 0.1)')
print(f'    drawdown = (best - current) / best * 100  ← best 기준!')
print(f'    trigger: drawdown >= callbackRate')

# Key difference
print(f'\n  [핵심 차이]')
print(f'    Backtest: (best-cur) / entry   ← 분모 = entry')
print(f'    Exchange: (best-cur) / best    ← 분모 = best')
print(f'    best > entry이므로 Exchange drawdown < Backtest drawdown')
print(f'    → Exchange가 더 늦게 발동 (더 많이 내려가야 함)')

# Numerical comparison
print(f'\n  [수치 비교 — current 가격별]')
header = f'  {"current":>10} | {"BT dd":>8} {"BT trail":>9} {"BT":>7} | {"EX dd":>8} {"EX cb":>8} {"EX":>7} | {"gap":>6}'
print(header)
print('  ' + '-' * len(header))

callback_updated = round(trail_K * atr / 87000 * 100, 1)  # 갱신 시 current 기준

for cur in [87500, 87300, 87100, 87000, 86800, 86500, 86000, 85500, 85000]:
    # Backtest
    bt_dd = (best - cur) / entry * 100
    bt_trail = trail_K * atr / cur * 100
    bt_trig = bt_dd >= bt_trail

    # Exchange (callbackRate = 갱신된 값, drawdown = best 기준)
    ex_cb = round(trail_K * atr / cur * 100, 1)
    ex_dd = (best - cur) / best * 100
    ex_trig = ex_dd >= ex_cb

    gap = 'same' if bt_trig == ex_trig else ('BT 1st' if bt_trig else 'EX 1st')
    bt_label = 'TRIG' if bt_trig else 'hold'
    ex_label = 'TRIG' if ex_trig else 'hold'
    print(f'  {cur:>10,} | {bt_dd:>7.3f}% {bt_trail:>8.3f}% {bt_label:>7} | '
          f'{ex_dd:>7.3f}% {ex_cb:>7.1f}% {ex_label:>7} | {gap:>6}')

# Find exact trigger prices
print(f'\n  [발동 가격 계산]')

# Backtest: (best-cur)/entry >= trail_K*atr/cur
# (best-cur)*cur >= trail_K*atr*entry
# -cur^2 + best*cur - trail_K*atr*entry >= 0
# cur^2 - best*cur + trail_K*atr*entry <= 0
a_coeff = 1
b_coeff = -best
c_coeff = trail_K * atr * entry
disc = b_coeff**2 - 4*a_coeff*c_coeff
if disc >= 0:
    bt_cur = (-b_coeff - math.sqrt(disc)) / (2*a_coeff)
    print(f'  Backtest 발동가: ${bt_cur:,.1f}')
    bt_raw = (bt_cur/entry - 1) * 100
    print(f'    → raw PnL at trigger: {bt_raw:+.3f}%')

# Exchange: (best-cur)/best >= trail_K*atr/cur
# (best-cur)*cur >= trail_K*atr*best
# -cur^2 + best*cur - trail_K*atr*best >= 0
c_coeff2 = trail_K * atr * best
disc2 = best**2 - 4*c_coeff2
if disc2 >= 0:
    ex_cur = (best - math.sqrt(disc2)) / 2
    print(f'  Exchange 발동가: ${ex_cur:,.1f}')
    ex_raw = (ex_cur/entry - 1) * 100
    print(f'    → raw PnL at trigger: {ex_raw:+.3f}%')

if disc >= 0 and disc2 >= 0:
    price_diff = bt_cur - ex_cur
    pnl_diff = (bt_cur - ex_cur) / entry * 100
    print(f'\n  차이:')
    print(f'    가격: ${price_diff:,.1f} (Exchange가 ${abs(price_diff):,.0f} 더 내려가야)')
    print(f'    PnL:  {pnl_diff:+.4f}%p (거래당 이만큼 Exchange가 덜 실현)')

# Impact on the actual biggest trade
print(f'\n  [최대 수익 거래 (SHORT $120K→$101K)에서의 차이 시뮬레이션]')
e2 = 120436; b2 = 101517; atr2 = 1887

# Backtest trigger
c2_coeff = trail_K * atr2 * e2
disc3 = b2**2 - 4*c2_coeff
if disc3 >= 0:
    # SHORT: best is low, current goes up
    # Backtest: (cur-best)/entry*100 >= trail_K*atr/cur*100 (reversed for SHORT)
    # Actually for SHORT: best_pnl = (1-best/entry)*100, cur_pnl = (1-cur/entry)*100
    # drawdown = best_pnl - cur_pnl = (cur-best)/entry*100
    # trail_dist = trail_K*atr/cur*100
    # (cur-best)*cur >= trail_K*atr*entry
    # cur^2 - best*cur - trail_K*atr*entry >= 0... wait let me redo

    # SHORT backtest: (cur-best)/entry >= trail_K*atr/cur
    # (cur-best)*cur >= trail_K*atr*entry
    # cur^2 - best*cur - trail_K*atr*entry >= 0
    disc_s = b2**2 + 4*trail_K*atr2*e2
    bt_s = (b2 + math.sqrt(disc_s)) / 2
    bt_s_raw = (1 - bt_s/e2) * 100
    print(f'  Backtest SHORT 발동가: ${bt_s:,.1f} (raw = {bt_s_raw:+.2f}%)')

    # Exchange: (cur-best)/best >= callback
    # (cur-best) >= best * trail_K * atr / cur
    # cur^2 - best*cur - trail_K*atr*best >= 0
    disc_e = b2**2 + 4*trail_K*atr2*b2
    ex_s = (b2 + math.sqrt(disc_e)) / 2
    ex_s_raw = (1 - ex_s/e2) * 100
    print(f'  Exchange SHORT 발동가: ${ex_s:,.1f} (raw = {ex_s_raw:+.2f}%)')

    diff_s = ex_s - bt_s
    diff_pnl = (bt_s - ex_s) / e2 * 100  # SHORT이므로 higher exit = less profit
    print(f'  차이: ${diff_s:,.0f} (Exchange가 ${abs(diff_s):,.0f} 더 올라가야)')
    print(f'  PnL 차이: {diff_pnl:+.3f}%p/거래')

print(f'\n  [결론]')
print(f'  1. Exchange Trail은 best 기준 되돌림, 백테스트는 entry 기준 되돌림')
print(f'  2. 수학적으로 Exchange가 약간 늦게 발동 (더 많이 되돌려야 함)')
print(f'  3. 봇 내부 check_exit(=백테스트)가 먼저 발동하여 MARKET close')
print(f'  4. Exchange Trail은 봇 다운 시의 백업. 봇 정상 시 봇이 먼저 청산')
print(f'  5. 봇이 먼저 청산 → 결과는 백테스트와 동일')
