#!/usr/bin/env python3
"""
Fix Duplicate Trades in State File
Keep only RECONCILED trades, remove duplicates and non-reconciled versions
"""

import json
from datetime import datetime
from collections import defaultdict

state_file = 'results/opportunity_gating_bot_4x_state.json'

print('='*100)
print('FIXING DUPLICATE TRADES IN STATE FILE')
print('='*100)

# Backup first
import shutil
backup_file = state_file + f'.backup_before_dedup_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
shutil.copy(state_file, backup_file)
print(f'\n💾 Backup created: {backup_file}')

# Load state
with open(state_file, 'r') as f:
    state = json.load(f)

all_trades = state.get('trades', [])
print(f'\n📊 Total trades in state: {len(all_trades)}')

# Separate by status
closed_trades = [t for t in all_trades if t.get('status') == 'CLOSED']
open_trades = [t for t in all_trades if t.get('status') != 'CLOSED']

print(f'   Closed: {len(closed_trades)}')
print(f'   Open/Other: {len(open_trades)}')

# Group closed trades by position
# Use position_id_exchange if available, otherwise order_id
positions = defaultdict(list)

for trade in closed_trades:
    pos_id_ex = trade.get('position_id_exchange')
    order_id = trade.get('order_id')

    # Key: use position_id_exchange if available, otherwise order_id
    if pos_id_ex:
        key = str(pos_id_ex)
    else:
        key = str(order_id)

    positions[key].append(trade)

print(f'\n📋 GROUPED CLOSED TRADES:')
print(f'   Unique positions: {len(positions)}')

# For each position, keep only the best trade(s)
cleaned_trades = []

for position_id, trades in positions.items():
    print(f'\nPosition {position_id}:')
    print(f'   {len(trades)} trades found')

    # Prefer RECONCILED trades
    reconciled = [t for t in trades if t.get('exchange_reconciled', False)]
    non_reconciled = [t for t in trades if not t.get('exchange_reconciled', False)]

    print(f'   Reconciled: {len(reconciled)}, Non-reconciled: {len(non_reconciled)}')

    if reconciled:
        # If we have reconciled trades, use those
        if len(reconciled) > 1:
            # Multiple reconciled trades - keep only the first one (or merge?)
            print(f'   ⚠️  Multiple reconciled trades! Keeping first one.')
            # Keep the one that's NOT manual (if available)
            non_manual_reconciled = [t for t in reconciled if not t.get('manual_trade', False)]
            if non_manual_reconciled:
                cleaned_trades.append(non_manual_reconciled[0])
            else:
                cleaned_trades.append(reconciled[0])
        else:
            # One reconciled trade - perfect
            cleaned_trades.append(reconciled[0])
    else:
        # No reconciled trades - keep the non-reconciled ones
        # But if multiple, only keep unique ones
        if len(non_reconciled) > 1:
            print(f'   ⚠️  Multiple non-reconciled trades! Keeping first one.')
        cleaned_trades.append(non_reconciled[0])

print(f'\n📊 CLEANING RESULTS:')
print(f'   Original closed trades: {len(closed_trades)}')
print(f'   Cleaned closed trades: {len(cleaned_trades)}')
print(f'   Removed duplicates: {len(closed_trades) - len(cleaned_trades)}')

# Verify no OPEN positions are in cleaned trades
print(f'\n🔍 CHECKING FOR MISPLACED OPEN POSITIONS:')
current_position = state.get('position', {})
current_position_order_id = current_position.get('order_id')

if current_position_order_id:
    print(f'   Current OPEN position: Order {current_position_order_id}')

    # Check if this order is in cleaned_trades
    misplaced = [t for t in cleaned_trades if str(t.get('order_id')) == str(current_position_order_id)]

    if misplaced:
        print(f'   ⚠️  Found {len(misplaced)} OPEN position(s) in CLOSED trades!')
        for t in misplaced:
            print(f'      Removing Order {t.get("order_id")} from closed trades')
            cleaned_trades.remove(t)

# Update state
state['trades'] = cleaned_trades

# Save
with open(state_file, 'w') as f:
    json.dump(state, f, indent=2)

print(f'\n✅ State file updated!')
print(f'   Final closed trades: {len(cleaned_trades)}')
print(f'   Backup: {backup_file}')
print('='*100)
