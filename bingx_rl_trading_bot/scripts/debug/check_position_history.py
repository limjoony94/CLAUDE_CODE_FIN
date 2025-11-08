#!/usr/bin/env python3
"""Check position history from BingX API and compare with state file."""

import sys
import os
import json
from datetime import datetime
import yaml

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from api.bingx_client import BingXClient

def main():
    # Load API keys
    config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'config', 'api_keys.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Use mainnet to get real trade history
    api_key = config['bingx']['mainnet']['api_key']
    secret_key = config['bingx']['mainnet']['secret_key']

    # Initialize client (mainnet=False means use mainnet)
    client = BingXClient(api_key, secret_key, testnet=False)

    print('=' * 100)
    print('CLOSED ORDERS/TRADES (Last 10 from BingX API)')
    print('=' * 100)

    # Try to fetch closed orders using CCXT's fetch_closed_orders
    try:
        # Fetch closed orders (this includes filled orders)
        closed_orders = client.exchange.fetch_closed_orders('BTC/USDT:USDT', limit=20)

        print(f"\nFound {len(closed_orders)} closed orders")

        for i, order in enumerate(closed_orders[:10], 1):
            print(f'\n{"=" * 100}')
            print(f'Order #{i}')
            print(f'{"=" * 100}')
            print(f"Order ID:        {order.get('id')}")
            print(f"Timestamp:       {datetime.fromtimestamp(order.get('timestamp', 0) / 1000).strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Side:            {order.get('side')}")  # buy or sell
            print(f"Type:            {order.get('type')}")  # market, limit, etc
            print(f"Price:           ${order.get('price', 0):,.2f}")
            print(f"Amount:          {order.get('amount', 0):.6f} BTC")
            print(f"Filled:          {order.get('filled', 0):.6f} BTC")
            print(f"Cost:            ${order.get('cost', 0):,.2f}")
            print(f"Fee:             ${order.get('fee', {}).get('cost', 0):.4f} {order.get('fee', {}).get('currency', '')}")
            print(f"Status:          {order.get('status')}")

            print(f"\nRaw Order Data:")
            print(json.dumps(order, indent=2, default=str))

    except Exception as e:
        print(f"Error fetching closed orders: {e}")

    # Try to get my trades (filled orders with PnL information)
    print(f'\n{"=" * 100}')
    print('MY TRADES (Filled Orders with Details)')
    print('=' * 100)

    try:
        # Fetch my trades (this shows actual executed trades)
        my_trades = client.exchange.fetch_my_trades('BTC/USDT:USDT', limit=20)

        print(f"\nFound {len(my_trades)} trades")

        for i, trade in enumerate(my_trades[:10], 1):
            print(f'\n{"=" * 100}')
            print(f'Trade #{i}')
            print(f'{"=" * 100}')
            print(f"Trade ID:        {trade.get('id')}")
            print(f"Order ID:        {trade.get('order')}")
            print(f"Timestamp:       {datetime.fromtimestamp(trade.get('timestamp', 0) / 1000).strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"Side:            {trade.get('side')}")  # buy or sell
            print(f"Price:           ${trade.get('price', 0):,.2f}")
            print(f"Amount:          {trade.get('amount', 0):.6f} BTC")
            print(f"Cost:            ${trade.get('cost', 0):,.2f}")
            print(f"Fee:             ${trade.get('fee', {}).get('cost', 0):.4f} {trade.get('fee', {}).get('currency', '')}")

            # Check if there's PnL info
            if 'info' in trade:
                print(f"\nAdditional Info:")
                info = trade['info']
                if 'profit' in info:
                    print(f"  Profit:        ${info.get('profit', 0)}")
                if 'realizedProfit' in info:
                    print(f"  Realized PnL:  ${info.get('realizedProfit', 0)}")

            print(f"\nRaw Trade Data:")
            print(json.dumps(trade, indent=2, default=str))

    except Exception as e:
        print(f"Error fetching my trades: {e}")

    # Load and compare with state file
    print('\n' + '=' * 100)
    print('STATE FILE COMPARISON')
    print('=' * 100)

    state_path = os.path.join(os.path.dirname(__file__), '..', '..', 'results', 'opportunity_gating_bot_4x_state.json')
    with open(state_path, 'r') as f:
        state = json.load(f)

    print(f"\nState file trades (last 5):")
    for i, trade in enumerate(reversed(state.get('trades', [])[-5:]), 1):
        entry_price = trade.get('entry_price', 0)
        exit_price = trade.get('exit_price', 0)
        pnl_usd = trade.get('pnl_usd_net', trade.get('pnl_usd', 0))
        pnl_pct = trade.get('pnl_pct', 0) * 100
        side = trade.get('side', 'UNKNOWN')

        print(f"\n#{i} {side:8s} | ${entry_price:,.2f} → ${exit_price:,.2f} | {pnl_pct:+.2f}% (${pnl_usd:+.2f})")
        print(f"   Position Value: ${trade.get('position_value', 0):,.2f}")
        print(f"   Total Fee: ${trade.get('total_fee', 0):.2f}")
        print(f"   Entry Fee: ${trade.get('entry_fee', 0):.2f}")
        print(f"   Exit Fee: ${trade.get('exit_fee', 0):.2f}")
        if trade.get('manual_trade'):
            print(f"   *** MANUAL TRADE ***")
            if 'orders' in trade:
                print(f"   Orders: {len(trade['orders'])} orders")

if __name__ == '__main__':
    main()
