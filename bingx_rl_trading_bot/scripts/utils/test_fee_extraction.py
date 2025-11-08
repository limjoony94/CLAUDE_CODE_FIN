#!/usr/bin/env python3
"""
Test fee extraction from state file
"""
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import json

def extract_fees_from_state(state):
    """Extract fee data from state file (V2 reconciliation compatible)"""
    if not state:
        return {'total_fees': 0, 'entry_fees': 0, 'exit_fees': 0, 'order_details': []}

    try:
        trades = state.get('trades', [])
        total_fees = 0.0
        entry_fees = 0.0
        exit_fees = 0.0
        order_details = []

        for trade in trades:
            # Get fee from state file (populated by V2 reconciliation)
            fee = trade.get('total_fee', 0)
            fee = float(fee) if fee else 0.0

            if fee > 0:
                total_fees += fee

                # Approximate 50/50 split between entry and exit
                entry_fees += fee / 2
                exit_fees += fee / 2

                order_details.append({
                    'order_id': trade.get('order_id', 'N/A'),
                    'type': 'position',
                    'side': trade.get('side', 'UNKNOWN'),
                    'fee': fee,
                    'currency': 'USDT',
                    'source': 'v2_reconciliation'
                })

        return {
            'total_fees': total_fees,
            'entry_fees': entry_fees,
            'exit_fees': exit_fees,
            'order_details': order_details
        }

    except Exception as e:
        return {'total_fees': 0, 'entry_fees': 0, 'exit_fees': 0, 'order_details': []}


# Load state
state_file = 'results/opportunity_gating_bot_4x_state.json'
with open(state_file) as f:
    state = json.load(f)

print('='*80)
print('FEE EXTRACTION TEST (V2 Reconciliation Compatible)')
print('='*80)

# Extract fees
fee_data = extract_fees_from_state(state)

print(f'\n✅ Fee Data Extracted:')
print(f'   Total Fees:  ${fee_data["total_fees"]:.2f}')
print(f'   Entry Fees:  ${fee_data["entry_fees"]:.2f}')
print(f'   Exit Fees:   ${fee_data["exit_fees"]:.2f}')
print(f'   Order Count: {len(fee_data["order_details"])}')

if fee_data['order_details']:
    print(f'\n📋 Order Details:')
    for order in fee_data['order_details']:
        print(f'   - Order: {order["order_id"]}, Fee: ${order["fee"]:.2f}, Source: {order["source"]}')

print('\n' + '='*80)
