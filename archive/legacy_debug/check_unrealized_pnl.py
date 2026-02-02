#!/usr/bin/env python3
"""Debug unrealized P&L calculation."""

import json
import os

# Load state
state_path = os.path.join(os.path.dirname(__file__), '..', '..', 'results', 'opportunity_gating_bot_4x_state.json')
with open(state_path, 'r') as f:
    state = json.load(f)

initial_balance = state.get('initial_balance', 1000.0)
current_balance = state.get('current_balance', 0)
realized_balance = state.get('realized_balance')
unrealized_pnl = state.get('unrealized_pnl')

print(f'Initial Balance: ${initial_balance:,.2f}')
print(f'Current Balance: ${current_balance:,.2f}')
print(f'Realized Balance: ${realized_balance:,.2f}' if realized_balance is not None else 'Realized Balance: None')
print(f'Unrealized P&L: ${unrealized_pnl:,.2f}' if unrealized_pnl is not None else 'Unrealized P&L: None')
print()

# Calculate returns
total_return = (current_balance - initial_balance) / initial_balance if initial_balance > 0 else 0
realized_return = (realized_balance - initial_balance) / initial_balance if realized_balance is not None and initial_balance > 0 else 0

print(f'Total Return: {total_return*100:.2f}%')
print(f'Realized Return: {realized_return*100:.2f}%')
print()

# Calculate unrealized return (as monitor does)
if unrealized_pnl is not None:
    unrealized_return = unrealized_pnl / initial_balance
    print(f'Unrealized Return (from state): {unrealized_return*100:.2f}%')
else:
    unrealized_return = total_return - realized_return
    print(f'Unrealized Return (fallback): {unrealized_return*100:.2f}%')
print()

print(f'Check: unrealized_pnl is not None = {unrealized_pnl is not None}')
print(f'Check: type(unrealized_pnl) = {type(unrealized_pnl)}')
print(f'Check: unrealized_pnl = {unrealized_pnl}')
