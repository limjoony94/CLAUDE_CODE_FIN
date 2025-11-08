#!/usr/bin/env python3
"""List all trades in state file"""

import json
from datetime import datetime

state_file = 'results/opportunity_gating_bot_4x_state.json'

with open(state_file, 'r') as f:
    state = json.load(f)

closed_trades = [t for t in state.get('trades', []) if t.get('status') == 'CLOSED']

print(f'\nTotal closed trades: {len(closed_trades)}\n')

for i, trade in enumerate(closed_trades, 1):
    entry_time = trade.get('entry_time', 'N/A')
    order_id = trade.get('order_id', 'N/A')
    reconciled = trade.get('exchange_reconciled', False)
    manual = trade.get('manual_trade', False)

    # Parse entry time if it's a datetime string
    try:
        if isinstance(entry_time, str) and 'T' in entry_time:
            dt = datetime.fromisoformat(entry_time)
            entry_time_str = dt.strftime('%Y-%m-%d %H:%M:%S')
        else:
            entry_time_str = entry_time
    except:
        entry_time_str = str(entry_time)

    flags = []
    if reconciled:
        flags.append('RECONCILED')
    if manual:
        flags.append('MANUAL')

    flag_str = f" [{', '.join(flags)}]" if flags else ""

    print(f'{i:2d}. Order {order_id}: {entry_time_str}{flag_str}')
