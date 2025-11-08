#!/usr/bin/env python3
"""Analyze positions from API trades and calculate P&L with all fees."""

import sys
import os
import json
from datetime import datetime
from collections import defaultdict

def analyze_positions(trades):
    """Analyze trades and reconstruct positions with P&L."""

    # Sort trades by timestamp (oldest first)
    sorted_trades = sorted(trades, key=lambda x: x['timestamp'])

    positions = []
    current_position = None

    for trade in sorted_trades:
        side = trade['side']  # 'buy' or 'sell'
        timestamp = datetime.fromtimestamp(trade['timestamp'] / 1000)
        price = float(trade['price'])
        amount = float(trade['amount'])
        cost = float(trade['cost'])
        fee = float(trade.get('fee', {}).get('cost', 0))
        order_id = trade['order']

        if side == 'buy':
            # Opening a LONG position or adding to it
            if current_position is None or current_position['side'] == 'CLOSED':
                # Start new position
                current_position = {
                    'side': 'LONG',
                    'entry_time': timestamp,
                    'entry_orders': [],
                    'exit_orders': [],
                    'total_entry_qty': 0,
                    'total_entry_cost': 0,
                    'total_entry_fee': 0,
                    'total_exit_qty': 0,
                    'total_exit_revenue': 0,
                    'total_exit_fee': 0,
                }

            # Add to entry
            current_position['entry_orders'].append({
                'order_id': order_id,
                'time': timestamp,
                'price': price,
                'amount': amount,
                'cost': cost,
                'fee': fee
            })
            current_position['total_entry_qty'] += amount
            current_position['total_entry_cost'] += cost
            current_position['total_entry_fee'] += fee

        elif side == 'sell':
            # Closing a LONG position
            if current_position is None:
                print(f"WARNING: SELL order {order_id} without open position")
                continue

            # Add to exit
            current_position['exit_orders'].append({
                'order_id': order_id,
                'time': timestamp,
                'price': price,
                'amount': amount,
                'revenue': cost,  # For SELL, cost is revenue
                'fee': fee
            })
            current_position['total_exit_qty'] += amount
            current_position['total_exit_revenue'] += cost
            current_position['total_exit_fee'] += fee

            # Check if position is fully closed
            if abs(current_position['total_exit_qty'] - current_position['total_entry_qty']) < 0.0001:
                # Position closed
                current_position['exit_time'] = timestamp
                current_position['side'] = 'CLOSED'

                # Calculate P&L
                gross_pnl = current_position['total_exit_revenue'] - current_position['total_entry_cost']
                total_fees = current_position['total_entry_fee'] + current_position['total_exit_fee']
                net_pnl = gross_pnl - total_fees

                # Calculate average prices
                avg_entry = current_position['total_entry_cost'] / current_position['total_entry_qty']
                avg_exit = current_position['total_exit_revenue'] / current_position['total_exit_qty']

                # Calculate margin (with 4x leverage)
                margin = current_position['total_entry_cost'] / 4
                pnl_pct = (net_pnl / margin) * 100

                current_position['avg_entry_price'] = avg_entry
                current_position['avg_exit_price'] = avg_exit
                current_position['gross_pnl'] = gross_pnl
                current_position['total_fees'] = total_fees
                current_position['net_pnl'] = net_pnl
                current_position['margin'] = margin
                current_position['pnl_pct'] = pnl_pct

                positions.append(current_position)
                current_position = None

    # If there's an open position, add it
    if current_position and current_position['side'] != 'CLOSED':
        current_position['status'] = 'OPEN'
        positions.append(current_position)

    return positions

def main():
    # Load API trades
    trades_path = os.path.join(os.path.dirname(__file__), 'api_trades.json')
    with open(trades_path, 'r') as f:
        trades = json.load(f)

    print('=' * 100)
    print('POSITION ANALYSIS FROM API TRADES')
    print('=' * 100)

    positions = analyze_positions(trades)

    print(f"\nFound {len(positions)} positions\n")

    for i, pos in enumerate(reversed(positions[-5:]), 1):
        print(f"{'=' * 100}")
        print(f"Position #{i}")
        print(f"{'=' * 100}")
        print(f"Side:            {pos.get('side', pos.get('status', 'UNKNOWN'))}")
        print(f"Entry Time:      {pos['entry_time'].strftime('%Y-%m-%d %H:%M:%S')}")

        if 'exit_time' in pos:
            print(f"Exit Time:       {pos['exit_time'].strftime('%Y-%m-%d %H:%M:%S')}")

        print(f"\nEntry Orders ({len(pos['entry_orders'])}):")
        for order in pos['entry_orders']:
            print(f"  {order['time'].strftime('%Y-%m-%d %H:%M:%S')} | {order['amount']:.4f} BTC @ ${order['price']:,.2f} | Fee: ${order['fee']:.4f}")

        if pos['exit_orders']:
            print(f"\nExit Orders ({len(pos['exit_orders'])}):")
            for order in pos['exit_orders']:
                print(f"  {order['time'].strftime('%Y-%m-%d %H:%M:%S')} | {order['amount']:.4f} BTC @ ${order['price']:,.2f} | Fee: ${order['fee']:.4f}")

        print(f"\nSummary:")
        print(f"  Total Entry:     {pos['total_entry_qty']:.4f} BTC @ avg ${pos.get('avg_entry_price', 0):,.2f}")
        print(f"  Entry Cost:      ${pos['total_entry_cost']:,.2f}")
        print(f"  Entry Fees:      ${pos['total_entry_fee']:.4f}")

        if 'avg_exit_price' in pos:
            print(f"  Total Exit:      {pos['total_exit_qty']:.4f} BTC @ avg ${pos['avg_exit_price']:,.2f}")
            print(f"  Exit Revenue:    ${pos['total_exit_revenue']:,.2f}")
            print(f"  Exit Fees:       ${pos['total_exit_fee']:.4f}")
            print(f"\nP&L:")
            print(f"  Gross P&L:       ${pos['gross_pnl']:+,.2f}")
            print(f"  Total Fees:      ${pos['total_fees']:,.4f}")
            print(f"  Net P&L:         ${pos['net_pnl']:+,.2f}")
            print(f"  Margin (4x):     ${pos['margin']:,.2f}")
            print(f"  Return %:        {pos['pnl_pct']:+.2f}%")
        else:
            print(f"  Status:          OPEN (not closed yet)")

        print()

    # Now compare with state file
    print(f"{'=' * 100}")
    print('STATE FILE COMPARISON')
    print(f"{'=' * 100}")

    state_path = os.path.join(os.path.dirname(__file__), '..', '..', 'results', 'opportunity_gating_bot_4x_state.json')
    with open(state_path, 'r') as f:
        state = json.load(f)

    print(f"\nState file trades (last 5):")
    for i, trade in enumerate(reversed(state.get('trades', [])[-5:]), 1):
        side = trade.get('side', 'UNKNOWN')
        entry_price = trade.get('entry_price', 0)
        exit_price = trade.get('exit_price', 0)
        pnl_net = trade.get('pnl_usd_net', trade.get('pnl_usd', 0))
        pnl_pct = trade.get('pnl_pct', 0) * 100
        entry_fee = trade.get('entry_fee', 0)
        exit_fee = trade.get('exit_fee', 0)
        total_fee = trade.get('total_fee', 0)

        print(f"\n#{i} {side:8s} | ${entry_price:,.2f} → ${exit_price:,.2f} | {pnl_pct:+.2f}% (${pnl_net:+.2f})")
        print(f"   Entry Fee: ${entry_fee:.4f} | Exit Fee: ${exit_fee:.4f} | Total: ${total_fee:.4f}")
        if trade.get('manual_trade'):
            print(f"   *** MANUAL TRADE ***")

if __name__ == '__main__':
    main()
