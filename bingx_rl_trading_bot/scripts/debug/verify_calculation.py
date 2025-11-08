#!/usr/bin/env python3
"""
Verify calculation logic for monitor
"""

import json
from pathlib import Path

# Load state
PROJECT_ROOT = Path(__file__).parent.parent.parent
state_file = PROJECT_ROOT / 'results' / 'opportunity_gating_bot_4x_state.json'

with open(state_file, 'r') as f:
    state = json.load(f)

initial = state['initial_balance']
current = state['current_balance']
unrealized = state['unrealized_pnl']

# Calculate
total_return = (current - initial) / initial
unrealized_return = unrealized / initial
realized_return = ((current - initial) - unrealized) / initial

print('=' * 60)
print('CALCULATION VERIFICATION')
print('=' * 60)
print(f'Initial Balance: ${initial:,.2f}')
print(f'Current Balance: ${current:,.2f}')
print(f'Unrealized P&L:  ${unrealized:,.2f}')
print()
print(f'Total Return:      {total_return*100:+6.2f}%')
print(f'Unrealized Return: {unrealized_return*100:+6.2f}%')
print(f'Realized Return:   {realized_return*100:+6.2f}%')
print()
print('Verification (should match):')
print(f'  Realized + Unrealized = {(realized_return + unrealized_return)*100:+.2f}%')
print(f'  Total Return          = {total_return*100:+.2f}%')
print(f'  Match: {"✅ YES" if abs((realized_return + unrealized_return) - total_return) < 0.0001 else "❌ NO"}')
print('=' * 60)
