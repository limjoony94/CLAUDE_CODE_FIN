#!/usr/bin/env python3
import json
import os

state_file = os.path.join(os.path.dirname(__file__), '..', '..', 'results', 'opportunity_gating_bot_4x_state.json')

with open(state_file, 'r') as f:
    state = json.load(f)

closed_trades = [t for t in state['trades'] if t.get('status') == 'CLOSED']

print('=== CLOSED TRADES IN STATE FILE ===\n')
for i, trade in enumerate(closed_trades, 1):
    print(f"Trade {i}:")
    print(f"  Order ID: {trade.get('order_id')}")
    print(f"  Exit Reason: {trade.get('exit_reason', 'N/A')}")
    print(f"  pnl_usd (gross): ${trade.get('pnl_usd', 0):.2f}")
    print(f"  pnl_usd_net: ${trade.get('pnl_usd_net', 0):.2f}")
    print(f"  total_fee: ${trade.get('total_fee', 0):.2f}")
    print(f"  fees_reconciled: {trade.get('fees_reconciled', False)}")
    print()
