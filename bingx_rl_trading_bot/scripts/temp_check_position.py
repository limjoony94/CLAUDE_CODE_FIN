#!/usr/bin/env python
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.api.bingx_client import BingXClient
import yaml

# Load API keys
with open('config/api_keys.yaml', 'r') as f:
    config = yaml.safe_load(f)

api_key = config['bingx']['mainnet']['api_key']
api_secret = config['bingx']['mainnet']['secret_key']

# Initialize client
client = BingXClient(api_key, api_secret, testnet=False)

# Get positions
positions = client.get_positions('BTC-USDT')

if positions:
    for pos in positions:
        print('=== EXCHANGE POSITION ===')
        print(f"Side: {pos.get('positionSide', 'N/A')}")
        print(f"Quantity: {pos.get('positionAmt', 0)} BTC")
        print(f"Entry Price: ${float(pos.get('avgPrice', 0)):,.2f}")
        print(f"Leverage: {pos.get('leverage', 0)}x")
        notional = abs(float(pos.get('positionAmt', 0)) * float(pos.get('avgPrice', 0)))
        print(f"Notional: ${notional:,.2f}")
        print(f"Unrealized P&L: ${float(pos.get('unrealizedProfit', 0)):,.2f}")
else:
    print('No positions on exchange')
