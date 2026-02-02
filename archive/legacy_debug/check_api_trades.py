#!/usr/bin/env python3
"""Fetch all trades from BingX API and save to file for analysis."""

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

    # Initialize client (mainnet)
    client = BingXClient(api_key, secret_key, testnet=False)

    print('Fetching trades from BingX API...')

    # Fetch my trades
    my_trades = client.exchange.fetch_my_trades('BTC/USDT:USDT', limit=50)

    print(f"Found {len(my_trades)} trades")

    # Save to file
    output_path = os.path.join(os.path.dirname(__file__), 'api_trades.json')
    with open(output_path, 'w') as f:
        json.dump(my_trades, f, indent=2, default=str)

    print(f"Saved to: {output_path}")

    # Print summary
    print(f"\n{'=' * 100}")
    print(f"TRADE SUMMARY (Last 10)")
    print(f"{'=' * 100}")

    for i, trade in enumerate(my_trades[:10], 1):
        timestamp = datetime.fromtimestamp(trade.get('timestamp', 0) / 1000).strftime('%Y-%m-%d %H:%M:%S')
        side = trade.get('side', 'unknown')
        price = trade.get('price', 0)
        amount = trade.get('amount', 0)
        cost = trade.get('cost', 0)
        fee = trade.get('fee', {}).get('cost', 0)
        order_id = trade.get('order', 'unknown')

        print(f"\n#{i} {timestamp} | {side.upper():4s} | {amount:.4f} BTC @ ${price:,.2f} | Cost: ${cost:,.2f} | Fee: ${fee:.4f}")
        print(f"   Order ID: {order_id}")

if __name__ == '__main__':
    main()
