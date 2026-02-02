#!/usr/bin/env python3
"""Check reconciliation results"""
import json
from pathlib import Path

state_file = Path(__file__).parent.parent.parent / 'results' / 'opportunity_gating_bot_4x_state.json'

with open(state_file, 'r') as f:
    state = json.load(f)

trades = state.get('trades', [])
closed_trades = [t for t in trades if t.get('status') == 'CLOSED']
reconciled_trades = [t for t in trades if t.get('exchange_reconciled')]
bot_trades = [t for t in closed_trades if not t.get('manual_trade')]
manual_trades = [t for t in closed_trades if t.get('manual_trade')]

print('='*80)
print('RECONCILIATION RESULTS')
print('='*80)
print(f'\nTotal trades in state: {len(trades)}')
print(f'Closed trades: {len(closed_trades)}')
print(f'  - Bot trades: {len(bot_trades)}')
print(f'  - Manual trades: {len(manual_trades)}')
print(f'Exchange reconciled: {len(reconciled_trades)}')

total_net_pnl = sum(t.get('pnl_usd_net', 0) for t in closed_trades)
bot_net_pnl = sum(t.get('pnl_usd_net', 0) for t in bot_trades)
manual_net_pnl = sum(t.get('pnl_usd_net', 0) for t in manual_trades)

print(f'\nTotal Net P&L: ${total_net_pnl:.2f}')
print(f'  - Bot P&L: ${bot_net_pnl:.2f}')
print(f'  - Manual P&L: ${manual_net_pnl:.2f}')

print('\n' + '='*80)
print('DETAILED TRADES (Exchange Reconciled)')
print('='*80)

for i, trade in enumerate(reconciled_trades, 1):
    print(f"\nTrade {i}:")
    print(f"  Position ID: {trade.get('position_id_exchange')}")
    print(f"  Side: {trade.get('side')}")
    print(f"  Entry: ${trade.get('entry_price', 0):,.2f} @ {trade.get('entry_time', 'N/A')[:19]}")
    print(f"  Exit:  ${trade.get('exit_price', 0):,.2f} @ {trade.get('close_time', 'N/A')[:19]}")
    print(f"  Quantity: {trade.get('quantity', 0)}")
    print(f"  Realized P&L: ${trade.get('pnl_usd', 0):.2f}")
    print(f"  Total Fees: ${trade.get('total_fee', 0):.2f}")
    print(f"  Net P&L: ${trade.get('pnl_usd_net', 0):.2f}")
    print(f"  Type: {'MANUAL' if trade.get('manual_trade') else 'BOT'}")

print('\n' + '='*80)
