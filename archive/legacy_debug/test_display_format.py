#!/usr/bin/env python3
"""Test display formatting."""

import sys
import os
import json

# Load state
state_path = os.path.join(os.path.dirname(__file__), '..', '..', 'results', 'opportunity_gating_bot_4x_state.json')
with open(state_path, 'r') as f:
    state = json.load(f)

trades = state.get('trades', [])
closed_trades = [t for t in trades if t.get('status') == 'CLOSED']

print(f'Total closed trades: {len(closed_trades)}')
print()

# Get initial balance for account-based return calculation
initial_balance = state.get('initial_balance', 1000.0)
print(f'Initial balance: ${initial_balance:,.2f}')
print()

# Test display logic
recent = closed_trades[-5:]
for i, trade in enumerate(reversed(recent), 1):
    side = trade.get('side', 'N/A')
    entry_price = trade.get('entry_price', 0)
    exit_price = trade.get('exit_price', 0)
    pnl_usd = trade.get('pnl_usd_net', 0)
    exit_reason = trade.get('exit_reason', 'N/A')

    # Account-based return: P&L / Initial Balance
    # Shows impact of each trade on total account
    account_return_pct = (pnl_usd / initial_balance) * 100 if initial_balance > 0 else 0

    # Shorten exit reason
    if 'ML Exit' in exit_reason:
        exit_reason = 'ML Exit'
    elif 'Max Hold' in exit_reason:
        exit_reason = 'Max Hold'
    elif 'Manual' in exit_reason:
        exit_reason = 'Manual'

    print(f"│ #{len(closed_trades)-len(recent)+i:>3d}  {side:>5s}  │  ${entry_price:>10,.2f} → ${exit_price:>10,.2f}  │  "
          f"{account_return_pct:>+6.2f}% (${pnl_usd:>+8.2f})  │  {exit_reason:<10s}  │")
