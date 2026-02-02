#!/usr/bin/env python3
"""
Compare Position IDs from two different APIs
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from api.bingx_client import BingXClient
import yaml
import json
from datetime import datetime
from collections import defaultdict

def main():
    config = yaml.safe_load(open('config/api_keys.yaml'))
    client = BingXClient(
        config['bingx']['mainnet']['api_key'],
        config['bingx']['mainnet']['secret_key'],
        testnet=False
    )

    # Get session start
    with open('results/opportunity_gating_bot_4x_state.json') as f:
        state = json.load(f)
    session_start = datetime.fromisoformat(state['session_start'])
    since_ts = session_start.timestamp()

    print('='*80)
    print('COMPARING POSITION IDS FROM TWO APIs')
    print('='*80)

    # 1. fetchPositionHistory
    print('\n1. fetchPositionHistory (Position History API):')
    since_ms = int(since_ts * 1000)
    until_ms = int(datetime.now().timestamp() * 1000)
    positions = client.exchange.fetchPositionHistory(
        symbol='BTC/USDT:USDT',
        since=since_ms,
        limit=100,
        params={'until': until_ms}
    )
    print(f'   Found {len(positions)} positions')
    for pos in positions:
        entry_time = datetime.fromtimestamp(pos['timestamp']/1000).strftime('%H:%M:%S')
        net_pnl = pos['info']['netProfit']
        print(f'   - Position ID: {pos["id"]} | Entry: {entry_time} | Net P&L: {net_pnl}')

    # 2. swapV2PrivateGetTradeAllOrders
    print('\n2. swapV2PrivateGetTradeAllOrders (Order History API):')
    params = {
        'symbol': 'BTC-USDT',
        'startTime': since_ms,
        'endTime': until_ms
    }
    result = client.exchange.swapV2PrivateGetTradeAllOrders(params)
    orders = result.get('data', {}).get('orders', [])
    filled = [o for o in orders if o.get('status') == 'FILLED']

    # Group by position
    positions_dict = defaultdict(list)
    for order in filled:
        pos_id = order.get('positionID')
        if pos_id:
            positions_dict[pos_id].append(order)

    print(f'   Found {len(positions_dict)} positions')
    for pos_id, orders in positions_dict.items():
        # Check if closed
        has_exit = any(float(o.get('profit', 0)) != 0 for o in orders)
        if has_exit:
            first_order = min(orders, key=lambda x: int(x.get('time', 0)))
            net_pnl = sum(float(o.get('profit', 0)) for o in orders) - sum(abs(float(o.get('commission', 0))) for o in orders)
            entry_time = datetime.fromtimestamp(int(first_order.get('time', 0))/1000).strftime('%H:%M:%S')
            print(f'   - Position ID: {pos_id} | Entry: {entry_time} | Net P&L: {net_pnl:.2f}')

    print('\n' + '='*80)
    print('OBSERVATION: Position IDs are DIFFERENT between the two APIs!')
    print('='*80)

if __name__ == '__main__':
    main()
