#!/usr/bin/env python3
"""
Clean State File - Remove Old Reconciled Trades
"""

import json
from datetime import datetime

state_file = 'results/opportunity_gating_bot_4x_state.json'

print('='*100)
print('CLEANING STATE FILE')
print('='*100)

# Load state
with open(state_file, 'r') as f:
    state = json.load(f)

# Get session_start
session_start = state.get('session_start')
if not session_start:
    print('❌ No session_start found in state file!')
    exit(1)

session_start_dt = datetime.fromisoformat(session_start)
session_start_ts = session_start_dt.timestamp()

print(f'\n📅 Bot Session Start: {session_start_dt.strftime("%Y-%m-%d %H:%M:%S")}')

# Get trades
all_trades = state.get('trades', [])
print(f'📊 Total trades in state: {len(all_trades)}')

# Separate trades
trades_to_keep = []
trades_to_remove = []

for trade in all_trades:
    entry_time = trade.get('entry_time')

    # Parse entry time
    try:
        if isinstance(entry_time, str) and 'T' in entry_time:
            entry_dt = datetime.fromisoformat(entry_time)
            entry_ts = entry_dt.timestamp()

            # Keep if after session_start
            if entry_ts >= session_start_ts:
                trades_to_keep.append(trade)
            else:
                trades_to_remove.append(trade)
        else:
            # Keep if can't parse (safety)
            print(f'⚠️  Warning: Could not parse entry_time for order {trade.get("order_id")}: {entry_time}')
            trades_to_keep.append(trade)
    except Exception as e:
        # Keep if error (safety)
        print(f'⚠️  Warning: Error parsing entry_time for order {trade.get("order_id")}: {e}')
        trades_to_keep.append(trade)

print(f'\n📋 SUMMARY:')
print(f'   Trades to KEEP: {len(trades_to_keep)} (after session_start)')
print(f'   Trades to REMOVE: {len(trades_to_remove)} (before session_start)')

if trades_to_remove:
    print(f'\n🗑️  REMOVING {len(trades_to_remove)} OLD TRADES:')
    for i, trade in enumerate(trades_to_remove, 1):
        entry_time = trade.get('entry_time', 'N/A')
        order_id = trade.get('order_id', 'N/A')
        try:
            if isinstance(entry_time, str) and 'T' in entry_time:
                dt = datetime.fromisoformat(entry_time)
                entry_time_str = dt.strftime('%Y-%m-%d %H:%M:%S')
            else:
                entry_time_str = str(entry_time)
        except:
            entry_time_str = str(entry_time)

        print(f'   {i:2d}. Order {order_id}: {entry_time_str}')

# Update state
state['trades'] = trades_to_keep

# Save
with open(state_file, 'w') as f:
    json.dump(state, f, indent=2)

print(f'\n✅ State file updated!')
print(f'   New total trades: {len(trades_to_keep)}')
print(f'\n💾 Backup saved as: opportunity_gating_bot_4x_state.json.backup_*')
print('='*100)
