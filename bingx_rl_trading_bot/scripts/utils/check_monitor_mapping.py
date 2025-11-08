#!/usr/bin/env python3
import json
import os

state_file = os.path.join(os.path.dirname(__file__), '..', '..', 'results', 'opportunity_gating_bot_4x_state.json')

with open(state_file, 'r') as f:
    state = json.load(f)

closed = [t for t in state['trades'] if t.get('status') == 'CLOSED']
print(f'Total closed trades: {len(closed)}\n')

# Monitor displays last 5 in reversed order
for i, t in enumerate(reversed(closed[-5:]), 1):
    print(f'Monitor #{i}:')
    print(f'  Entry: ${t.get("entry_price"):,.2f} → Exit: ${t.get("exit_price"):,.2f}')
    print(f'  Exit Reason: {t.get("exit_reason", "N/A")[:40]}')
    print(f'  pnl_usd (gross): ${t.get("pnl_usd", 0):.2f}')
    print(f'  pnl_usd_net: ${t.get("pnl_usd_net", 0):.2f}')
    print(f'  total_fee: ${t.get("total_fee", 0):.2f}')
    print(f'  fees_reconciled: {t.get("fees_reconciled", False)}')
    print()
