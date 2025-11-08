#!/usr/bin/env python3
"""Test monitor trade display."""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import yaml
from api.bingx_client import BingXClient

# Load API keys
config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'config', 'api_keys.yaml')
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

api_key = config['bingx']['mainnet']['api_key']
secret_key = config['bingx']['mainnet']['secret_key']
client = BingXClient(api_key, secret_key, testnet=False)

# Fetch recent trades from API
api_trades = client.exchange.fetch_my_trades('BTC/USDT:USDT', limit=50)

print(f"Total API trades: {len(api_trades)}")

# Reconstruct positions from trades
positions = []
current_position = None

# Sort trades by timestamp (oldest first)
sorted_trades = sorted(api_trades, key=lambda x: x['timestamp'])

print(f"\nProcessing {len(sorted_trades)} trades...")

for trade in sorted_trades:
    side = trade['side']  # 'buy' or 'sell'
    price = float(trade['price'])
    amount = float(trade['amount'])
    cost = float(trade['cost'])
    fee = float(trade.get('fee', {}).get('cost', 0))
    timestamp = trade['timestamp']

    print(f"\nTrade: {side.upper():4s} | {amount:.4f} BTC @ ${price:,.2f} | Fee: ${fee:.4f}")

    if side == 'buy':
        # Opening LONG position
        if current_position is None or current_position.get('status') == 'CLOSED':
            print(f"  → Starting new LONG position")
            current_position = {
                'side': 'LONG',
                'entry_orders': [],
                'exit_orders': [],
                'total_entry_qty': 0,
                'total_entry_cost': 0,
                'total_entry_fee': 0,
                'total_exit_qty': 0,
                'total_exit_revenue': 0,
                'total_exit_fee': 0,
            }

        current_position['entry_orders'].append({'price': price, 'amount': amount, 'cost': cost, 'fee': fee, 'timestamp': timestamp})
        current_position['total_entry_qty'] += amount
        current_position['total_entry_cost'] += cost
        current_position['total_entry_fee'] += fee
        print(f"  → Added to position: Total qty={current_position['total_entry_qty']:.4f}")

    elif side == 'sell':
        # Closing LONG position
        if current_position:
            current_position['exit_orders'].append({'price': price, 'amount': amount, 'revenue': cost, 'fee': fee, 'timestamp': timestamp})
            current_position['total_exit_qty'] += amount
            current_position['total_exit_revenue'] += cost
            current_position['total_exit_fee'] += fee
            print(f"  → Exit from position: Total exit qty={current_position['total_exit_qty']:.4f}, Entry qty={current_position['total_entry_qty']:.4f}")

            # Check if position fully closed
            qty_diff = abs(current_position['total_exit_qty'] - current_position['total_entry_qty'])
            print(f"  → Qty difference: {qty_diff:.6f}")

            if qty_diff < 0.0001:
                print(f"  → Position CLOSED!")
                current_position['status'] = 'CLOSED'

                # Calculate P&L
                gross_pnl = current_position['total_exit_revenue'] - current_position['total_entry_cost']
                total_fees = current_position['total_entry_fee'] + current_position['total_exit_fee']
                net_pnl = gross_pnl - total_fees

                # Average prices
                avg_entry = current_position['total_entry_cost'] / current_position['total_entry_qty']
                avg_exit = current_position['total_exit_revenue'] / current_position['total_exit_qty']

                # Margin (4x leverage)
                margin = current_position['total_entry_cost'] / 4
                pnl_pct = (net_pnl / margin) * 100

                current_position['avg_entry_price'] = avg_entry
                current_position['avg_exit_price'] = avg_exit
                current_position['net_pnl'] = net_pnl
                current_position['margin'] = margin
                current_position['pnl_pct'] = pnl_pct
                current_position['exit_reason'] = 'Closed'

                print(f"  → P&L: ${net_pnl:+.2f} ({pnl_pct:+.2f}%)")
                print(f"  → Entry: ${avg_entry:,.2f}, Exit: ${avg_exit:,.2f}")

                positions.append(current_position)
                current_position = None
        else:
            print(f"  ⚠️  SELL without open position!")

print(f"\n\nTotal closed positions reconstructed: {len(positions)}")

for i, pos in enumerate(positions, 1):
    print(f"\nPosition #{i}:")
    print(f"  Entry: ${pos['avg_entry_price']:,.2f} → Exit: ${pos['avg_exit_price']:,.2f}")
    print(f"  P&L: ${pos['net_pnl']:+.2f} ({pos['pnl_pct']:+.2f}%)")
    print(f"  Margin: ${pos['margin']:,.2f}")
    print(f"  Entry orders: {len(pos['entry_orders'])}, Exit orders: {len(pos['exit_orders'])}")
