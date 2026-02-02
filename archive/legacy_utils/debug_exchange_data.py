#!/usr/bin/env python3
"""
Debug Exchange API Data
Shows raw order data from exchange with timestamps to verify reconciliation
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import yaml
import json
from datetime import datetime, timedelta
from api.bingx_client import BingXClient

def debug_exchange_orders():
    """Fetch and display raw exchange orders with full details"""

    # Load API credentials
    with open('config/api_keys.yaml', 'r') as f:
        config = yaml.safe_load(f)

    api_key = config['bingx']['mainnet']['api_key']
    api_secret = config['bingx']['mainnet']['secret_key']

    client = BingXClient(api_key, api_secret, testnet=False)

    print('='*100)
    print('DEBUG: Exchange API Raw Data')
    print('='*100)

    # Load state file to check bot_start_time
    state_file = 'results/opportunity_gating_bot_4x_state.json'
    with open(state_file, 'r') as f:
        state = json.load(f)

    # Check various time fields
    print('\n📋 STATE FILE TIME FIELDS:')
    print(f"   session_start: {state.get('session_start', 'NOT FOUND')}")
    print(f"   start_time: {state.get('start_time', 'NOT FOUND')}")
    print(f"   timestamp: {state.get('timestamp', 'NOT FOUND')}")

    # Convert session_start to timestamp
    if 'session_start' in state:
        session_start_dt = datetime.fromisoformat(state['session_start'])
        session_start_ts = session_start_dt.timestamp()
        print(f"\n   session_start as timestamp: {session_start_ts}")
        print(f"   session_start readable: {session_start_dt.strftime('%Y-%m-%d %H:%M:%S')}")

    # Fetch orders from last 7 days
    print('\n\n📊 FETCHING ORDERS FROM EXCHANGE (Last 7 days):')
    try:
        params = {
            'symbol': 'BTC-USDT',
            'startTime': int((datetime.now() - timedelta(days=7)).timestamp() * 1000),
            'endTime': int(datetime.now().timestamp() * 1000)
        }
        result = client.exchange.swapV2PrivateGetTradeAllOrders(params)

        if result.get('code') == '0':
            orders = result.get('data', {}).get('orders', [])
            filled_orders = [o for o in orders if o.get('status') == 'FILLED']

            print(f'   Total orders: {len(orders)}')
            print(f'   Filled orders: {len(filled_orders)}')

            # Display each filled order with details
            print('\n' + '='*100)
            print('FILLED ORDERS (Most Recent First):')
            print('='*100)

            for i, order in enumerate(sorted(filled_orders, key=lambda x: int(x.get('time', 0)), reverse=True), 1):
                order_time_ms = int(order.get('time', 0))
                order_time_dt = datetime.fromtimestamp(order_time_ms / 1000)

                print(f'\n{"─"*100}')
                print(f'Order #{i}:')
                print(f'  Time: {order_time_dt.strftime("%Y-%m-%d %H:%M:%S")} (timestamp: {order_time_ms / 1000})')
                print(f'  Order ID: {order.get("orderId")}')
                print(f'  Position ID: {order.get("positionID")}')
                print(f'  Side: {order.get("side")}')
                print(f'  Type: {order.get("type")}')
                print(f'  Price: ${float(order.get("price", 0)):,.2f}')
                print(f'  Quantity: {float(order.get("executedQty", 0)):.4f}')
                print(f'  Value: ${float(order.get("cumQuote", 0)):,.2f}')
                print(f'  Profit: ${float(order.get("profit", 0)):,.2f}')
                print(f'  Commission: ${abs(float(order.get("commission", 0))):,.2f}')
                print(f'  Status: {order.get("status")}')

                # Check if this order would be included by reconciliation
                if 'session_start' in state:
                    if order_time_ms / 1000 >= session_start_ts:
                        print(f'  ✅ WOULD BE INCLUDED (after session_start: {session_start_dt.strftime("%Y-%m-%d %H:%M:%S")})')
                    else:
                        print(f'  ❌ WOULD BE EXCLUDED (before session_start: {session_start_dt.strftime("%Y-%m-%d %H:%M:%S")})')

            # Summary
            print('\n' + '='*100)
            print('SUMMARY:')
            print('='*100)

            if 'session_start' in state:
                included_orders = [o for o in filled_orders if int(o.get('time', 0)) / 1000 >= session_start_ts]
                excluded_orders = [o for o in filled_orders if int(o.get('time', 0)) / 1000 < session_start_ts]

                print(f'  Orders AFTER session_start ({session_start_dt.strftime("%Y-%m-%d %H:%M:%S")}): {len(included_orders)}')
                print(f'  Orders BEFORE session_start: {len(excluded_orders)}')

                if len(excluded_orders) > 0:
                    print(f'\n  ⚠️  WARNING: {len(excluded_orders)} old orders would be excluded with proper filtering!')
                    print(f'  ⚠️  These are likely the "trades you never made" that appeared in the monitor.')
            else:
                print('  ⚠️  WARNING: No session_start found in state file!')
                print('  ⚠️  This means ALL historical orders are included (no time filter)!')
        else:
            print(f"❌ API Error: {result.get('msg')}")

    except Exception as e:
        print(f"❌ Failed to fetch orders: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    debug_exchange_orders()
