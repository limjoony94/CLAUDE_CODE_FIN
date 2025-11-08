#!/usr/bin/env python3
"""
Keep Only Reconciled Trades
Remove all non-reconciled trades since they're duplicates of reconciled ones
"""

import json
from datetime import datetime

state_file = 'results/opportunity_gating_bot_4x_state.json'

print('='*100)
print('KEEPING ONLY RECONCILED TRADES (Exchange Ground Truth)')
print('='*100)

# Backup
import shutil
backup_file = state_file + f'.backup_reconciled_only_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
shutil.copy(state_file, backup_file)
print(f'\n💾 Backup created: {backup_file}')

# Load state
with open(state_file, 'r') as f:
    state = json.load(f)

all_trades = state.get('trades', [])
print(f'\n📊 Total trades: {len(all_trades)}')

# Separate
closed_trades = [t for t in all_trades if t.get('status') == 'CLOSED']
open_trades = [t for t in all_trades if t.get('status') != 'CLOSED']

print(f'   Closed: {len(closed_trades)}')
print(f'   Open/Other: {len(open_trades)}')

# Separate reconciled vs non-reconciled
reconciled = [t for t in closed_trades if t.get('exchange_reconciled', False)]
non_reconciled = [t for t in closed_trades if not t.get('exchange_reconciled', False)]

print(f'\n📋 CLOSED TRADES:')
print(f'   Reconciled (from exchange): {len(reconciled)} ✅')
print(f'   Non-reconciled (bot-created): {len(non_reconciled)}')

print(f'\n🗑️  REMOVING NON-RECONCILED CLOSED TRADES:')
for i, t in enumerate(non_reconciled, 1):
    order_id = t.get('order_id', 'N/A')
    entry_time = t.get('entry_time', 'N/A')
    pos_id = t.get('position_id_exchange', 'N/A')

    try:
        if isinstance(entry_time, str) and 'T' in entry_time:
            dt = datetime.fromisoformat(entry_time)
            entry_str = dt.strftime('%Y-%m-%d %H:%M:%S')
        else:
            entry_str = str(entry_time)
    except:
        entry_str = str(entry_time)

    print(f'   {i}. Order {order_id}: {entry_str} (Position: {pos_id})')
    print(f'      Reason: Duplicate of reconciled trade')

# Keep only reconciled trades
cleaned_closed_trades = reconciled

print(f'\n✅ FINAL RESULT:')
print(f'   Keeping {len(cleaned_closed_trades)} reconciled closed trades')
print(f'   Removing {len(non_reconciled)} non-reconciled duplicates')

# Update state (keep open trades unchanged)
state['trades'] = cleaned_closed_trades

# Save
with open(state_file, 'w') as f:
    json.dump(state, f, indent=2)

print(f'\n💾 State file updated!')
print(f'   Total trades: {len(cleaned_closed_trades)}')
print(f'   Backup: {backup_file}')
print('='*100)
