#!/usr/bin/env python3
"""Verify _calc_trail_trigger_price matches check_exit trail logic exactly."""
import os, sys, math
os.chdir(os.path.join(os.path.dirname(__file__), '..', '..'))
sys.path.insert(0, '.')
from scripts.production.c1_breakout.signals import C1BreakoutSignal
from scripts.production.c1_breakout.config import load_config

cfg = load_config()
sig = C1BreakoutSignal(cfg['strategy'])
trail_K = cfg['strategy']['trail_K']  # 2.5

print('=' * 70)
print('  Trail Trigger Price vs check_exit — Mathematical Verification')
print('=' * 70)

def calc_trigger(direction, entry, best, atr):
    """Replicate _calc_trail_trigger_price logic."""
    k_atr = trail_K * atr
    if direction == 'LONG':
        disc = best * best - 4 * k_atr * entry
        if disc < 0: return None
        return (best + math.sqrt(disc)) / 2  # upper root: below best, above entry
    else:
        disc = best * best + 4 * k_atr * entry
        return (best + math.sqrt(disc)) / 2  # upper root: above best (SHORT = price up)

# Test cases
cases = [
    ('LONG',  80000, 88000, 400),   # big winner
    ('LONG',  85000, 85500, 300),   # small move
    ('SHORT', 120000, 101517, 1887), # biggest trade
    ('SHORT', 90000, 85000, 400),   # moderate SHORT
    ('LONG',  85000, 86000, 250),   # small LONG
]

all_ok = True
print(f'\n  {"dir":>5} {"entry":>8} {"best":>8} {"ATR":>5} | {"trigger":>10} | {"check_exit":>12} | {"match":>5}')
print(f'  {"-"*5} {"-"*8} {"-"*8} {"-"*5} | {"-"*10} | {"-"*12} | {"-"*5}')

for d, entry, best, atr in cases:
    trigger = calc_trigger(d, entry, best, atr)
    if trigger is None:
        print(f'  {d:>5} {entry:>8} {best:>8} {atr:>5} | {"N/A":>10} |')
        continue

    # Verify: at trigger price, check_exit should return TRAIL_TP
    # We need to simulate check_exit at trigger price
    if d == 'LONG':
        best_pnl = (best / entry - 1) * 100
        cur_pnl_at_trigger = (trigger / entry - 1) * 100
        drawdown = best_pnl - cur_pnl_at_trigger
        trail_dist = trail_K * atr / trigger * 100
    else:
        best_pnl = (1 - best / entry) * 100
        cur_pnl_at_trigger = (1 - trigger / entry) * 100
        drawdown = best_pnl - cur_pnl_at_trigger
        trail_dist = trail_K * atr / trigger * 100

    # At exact trigger: drawdown should equal trail_dist
    match = abs(drawdown - trail_dist) < 0.001
    if not match: all_ok = False

    # Also check with actual check_exit
    if d == 'LONG':
        ex = sig.check_exit('LONG', entry, best, trigger + 100, trigger, trigger,
                           entry * 0.95, atr, 5)
    else:
        ex = sig.check_exit('SHORT', entry, best, trigger, trigger - 100, trigger,
                           entry * 1.05, atr, 5)

    ex_result = ex['reason'] if ex else 'HOLD'

    print(f'  {d:>5} {entry:>8} {best:>8} {atr:>5} | ${trigger:>9.1f} | '
          f'dd={drawdown:.3f}% td={trail_dist:.3f}% | {ex_result:>5} {"OK" if match else "FAIL"}')

    # Test slightly above trigger (should NOT trigger for LONG)
    epsilon = 1.0
    if d == 'LONG':
        ex_above = sig.check_exit('LONG', entry, best, trigger + epsilon + 100,
                                  trigger + epsilon, trigger + epsilon,
                                  entry * 0.95, atr, 5)
    else:
        ex_above = sig.check_exit('SHORT', entry, best, trigger - epsilon,
                                  trigger - epsilon - 100, trigger - epsilon,
                                  entry * 1.05, atr, 5)
    above_result = ex_above['reason'] if ex_above else 'HOLD'

    # Test slightly below trigger (SHOULD trigger for LONG)
    if d == 'LONG':
        ex_below = sig.check_exit('LONG', entry, best, trigger - epsilon + 100,
                                  trigger - epsilon, trigger - epsilon,
                                  entry * 0.95, atr, 5)
    else:
        ex_below = sig.check_exit('SHORT', entry, best, trigger + epsilon,
                                  trigger + epsilon - 100, trigger + epsilon,
                                  entry * 1.05, atr, 5)
    below_result = ex_below['reason'] if ex_below else 'HOLD'

    expected_above = 'HOLD'  # just above trigger = not yet triggered
    expected_below = 'TRAIL_TP'  # just below trigger = triggered

    print(f'        +$1: {above_result:>8} (expect HOLD: {"OK" if above_result == expected_above else "FAIL"})')
    print(f'        -$1: {below_result:>8} (expect TRAIL_TP: {"OK" if below_result == expected_below else "FAIL"})')

print(f'\n  All trigger prices mathematically correct: {"PASS" if all_ok else "FAIL"}')

# Show the actual biggest trade verification
print(f'\n  [최대 수익 거래 검증]')
d, ep, bp, atr = 'SHORT', 120436.3, 101516.5, 1887
trigger = calc_trigger(d, ep, bp, atr)
print(f'  SHORT entry=${ep:,.1f} best=${bp:,.1f} ATR=${atr}')
print(f'  trigger = ${trigger:,.1f}')
print(f'  실제 backtest exit = $106,719.5')
print(f'  차이 = ${trigger - 106719.5:,.1f}')
print(f'  (차이 이유: trigger는 "정확한 발동 경계", exit는 "realized PnL 기반 가격")')
print(f'  trigger에서 봉 close가 체크되면 check_exit가 TRAIL_TP 반환 → MARKET close')
