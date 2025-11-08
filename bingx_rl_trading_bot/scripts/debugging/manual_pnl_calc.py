"""
Manual P&L Calculation & Emergency Exit Check
==============================================
"""

from datetime import datetime

# Position Info
ENTRY_PRICE = 106377.4
CURRENT_PRICE = 106400.2
ENTRY_TIME = "2025-10-17 07:05:00"
QUANTITY = 0.010388048365712166
POSITION_VALUE = 276.26
LEVERAGE = 4

# Calculate
print("="*80)
print("MANUAL P&L CALCULATION")
print("="*80)

# 1. Price change
price_change = CURRENT_PRICE - ENTRY_PRICE
price_change_pct = price_change / ENTRY_PRICE

print(f"\n[1] Price Movement")
print(f"   Entry Price: ${ENTRY_PRICE:,.2f}")
print(f"   Current Price: ${CURRENT_PRICE:,.2f}")
print(f"   Change: ${price_change:+,.2f} ({price_change_pct*100:+.4f}%)")

# 2. Leveraged P&L
leveraged_pnl_pct = price_change_pct * LEVERAGE
pnl_usd = POSITION_VALUE * leveraged_pnl_pct

print(f"\n[2] P&L (LONG position)")
print(f"   Unleveraged: {price_change_pct*100:+.4f}%")
print(f"   Leveraged (4x): {leveraged_pnl_pct*100:+.4f}%")
print(f"   P&L USD: ${pnl_usd:+,.2f}")

# 3. Holding time
entry_dt = datetime.fromisoformat(ENTRY_TIME)
current_dt = datetime.now()
duration = (current_dt - entry_dt).total_seconds() / 3600

print(f"\n[3] Holding Time")
print(f"   Entry: {ENTRY_TIME}")
print(f"   Current: {current_dt.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"   Duration: {duration:.2f} hours")

# 4. Exit checks
print(f"\n[4] Emergency Exit Checks")

EMERGENCY_STOP_LOSS = -0.05  # -5%
EMERGENCY_MAX_HOLD_HOURS = 8  # 8 hours

print(f"   Emergency Stop Loss: {EMERGENCY_STOP_LOSS*100}%")
print(f"   Current P&L: {leveraged_pnl_pct*100:+.4f}%")
if leveraged_pnl_pct <= EMERGENCY_STOP_LOSS:
    print(f"   ⚠️ SHOULD TRIGGER: Emergency Stop Loss!")
else:
    print(f"   ✅ Stop Loss OK (P&L > {EMERGENCY_STOP_LOSS*100}%)")

print(f"\n   Emergency Max Hold: {EMERGENCY_MAX_HOLD_HOURS}h")
print(f"   Current Duration: {duration:.2f}h")
if duration >= EMERGENCY_MAX_HOLD_HOURS:
    print(f"   🚨 SHOULD TRIGGER: Emergency Max Hold! (초과: {duration - EMERGENCY_MAX_HOLD_HOURS:.2f}h)")
else:
    print(f"   ✅ Max Hold OK ({EMERGENCY_MAX_HOLD_HOURS - duration:.2f}h remaining)")

# 5. Why not exiting?
print(f"\n[5] Exit Logic Analysis")
print(f"   ML Exit Signal: 0.003 (threshold: 0.70)")
print(f"   → ML Exit: NOT triggered (signal too low)")
print(f"")
print(f"   Emergency Stop Loss: {leveraged_pnl_pct*100:+.4f}% > -5%")
print(f"   → Stop Loss: NOT triggered (P&L above threshold)")
print(f"")
print(f"   Holding Duration: {duration:.2f}h >= 8.0h")
print(f"   → Max Hold: SHOULD TRIGGER!")

print(f"\n[6] CONCLUSION")
print(f"   🚨 Emergency Max Hold이 작동해야 하는데 작동하지 않음!")
print(f"   원인 조사 필요:")
print(f"   1. check_exit_signal() 함수가 제대로 호출되는가?")
print(f"   2. hours_held 계산이 올바른가?")
print(f"   3. entry_time parsing이 올바른가?")

print("="*80)
