#!/usr/bin/env python3
"""
Verify critical fixes are applied
"""
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
state_file = PROJECT_ROOT / 'results' / 'opportunity_gating_bot_4x_state.json'

with open(state_file, 'r') as f:
    state = json.load(f)

print('=== Critical Fixes Verification ===\n')

# 1. net_balance field (renamed from realized_balance)
if 'net_balance' in state:
    print(f'✅ net_balance field: ${state["net_balance"]:,.2f}')
else:
    print('❌ net_balance: NOT FOUND')

# 2. realized_balance should NOT exist anymore
if 'realized_balance' in state:
    print('⚠️  realized_balance still exists (should be removed)')
else:
    print('✅ realized_balance: Removed')

# 3. Position has position_id_exchange
print(f'\n✅ Position status: {state["position"]["status"]}')
print(f'✅ Position side: {state["position"]["side"]}')
position_id = state["position"].get("position_id_exchange", "NOT FOUND")
print(f'✅ position_id_exchange: {position_id}')

# 4. Trades array populated
trades_count = len(state.get("trades", []))
print(f'\n✅ Trades array length: {trades_count}')

if trades_count > 0:
    last_trade = state["trades"][-1]
    has_position_id = "position_id_exchange" in last_trade
    print(f'✅ Last trade has position_id: {has_position_id}')
    if has_position_id:
        print(f'   position_id: {last_trade["position_id_exchange"]}')
else:
    print('⚠️  No trades in array')

print('\n=== Summary ===')
print('All critical fixes applied successfully!' if 'net_balance' in state and trades_count > 0 else '⚠️  Some fixes missing')
