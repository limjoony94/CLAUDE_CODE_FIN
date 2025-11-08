#!/usr/bin/env python3
"""
Test various BingX APIs to fetch closed position history
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import yaml
from api.bingx_client import BingXClient
from datetime import datetime, timedelta

def test_closed_positions():
    """Test different methods to fetch closed position data"""
    
    # Load API credentials
    with open('config/api_keys.yaml', 'r') as f:
        config = yaml.safe_load(f)
    
    api_key = config['bingx']['mainnet']['api_key']
    api_secret = config['bingx']['mainnet']['secret_key']
    
    client = BingXClient(api_key, api_secret, testnet=False)
    
    print('='*80)
    print('Testing BingX Closed Position APIs')
    print('='*80)
    
    # Method 1: CCXT fetchPositionsHistory
    print('\n1. Testing CCXT fetchPositionsHistory():')
    try:
        if hasattr(client.exchange, 'fetchPositionsHistory'):
            positions = client.exchange.fetchPositionsHistory(symbol='BTC/USDT:USDT')
            print(f'   ✅ Success! Found {len(positions)} positions')
            if positions:
                print(f'   First position: {positions[0]}')
        else:
            print('   ❌ Method not supported')
    except Exception as e:
        print(f'   ❌ Error: {e}')
    
    # Method 2: Direct API - Trade History
    print('\n2. Testing swapV2PrivateGetTradeAllOrders():')
    try:
        params = {
            'symbol': 'BTC-USDT',
            'startTime': int((datetime.now() - timedelta(days=7)).timestamp() * 1000),
            'endTime': int(datetime.now().timestamp() * 1000)
        }
        result = client.exchange.swapV2PrivateGetTradeAllOrders(params)
        print(f'   ✅ Success! Response: {result}')
    except Exception as e:
        print(f'   ❌ Error: {e}')
    
    # Method 3: Position History
    print('\n3. Testing swapV2PrivateGetPositionsHistory():')
    try:
        params = {
            'symbol': 'BTC-USDT'
        }
        result = client.exchange.swapV2PrivateGetPositionsHistory(params)
        print(f'   ✅ Success! Response: {result}')
    except Exception as e:
        print(f'   ❌ Error: {e}')
    
    # Method 4: User Historical Orders
    print('\n4. Testing swapV2PrivateGetUserHistoricalOrders():')
    try:
        params = {
            'symbol': 'BTC-USDT'
        }
        result = client.exchange.swapV2PrivateGetUserHistoricalOrders(params)
        print(f'   ✅ Success! Response: {result}')
    except Exception as e:
        print(f'   ❌ Error: {e}')
    
    # Method 5: Fetch closed orders (might include position info)
    print('\n5. Testing fetch_closed_orders():')
    try:
        orders = client.exchange.fetch_closed_orders('BTC/USDT:USDT', limit=10)
        print(f'   ✅ Success! Found {len(orders)} closed orders')
        if orders:
            print(f'   First order: {orders[0]}')
    except Exception as e:
        print(f'   ❌ Error: {e}')
    
    # Method 6: Fetch my trades
    print('\n6. Testing fetch_my_trades():')
    try:
        since = int((datetime.now() - timedelta(days=7)).timestamp() * 1000)
        trades = client.exchange.fetch_my_trades('BTC/USDT:USDT', since=since, limit=50)
        print(f'   ✅ Success! Found {len(trades)} trades')
        if trades:
            print(f'\n   Sample trade:')
            for key, value in trades[0].items():
                print(f'   {key}: {value}')
    except Exception as e:
        print(f'   ❌ Error: {e}')
    
    print('\n' + '='*80)

if __name__ == '__main__':
    test_closed_positions()
