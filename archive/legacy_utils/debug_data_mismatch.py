#!/usr/bin/env python3
"""
Debug Data Mismatch: Compare State File vs Exchange API
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import yaml
import json
from datetime import datetime
from api.bingx_client import BingXClient

def compare_data(state_file_path, api_client):
    """Compare state file data with exchange API data"""

    # Read state file
    with open(state_file_path, 'r') as f:
        state = json.load(f)

    print('='*100)
    print('  🔍 STATE FILE vs EXCHANGE API COMPARISON')
    print('='*100)

    # 1. Balance Comparison
    print('\n📊 BALANCE COMPARISON')
    print('-'*100)

    state_balance = state.get('current_balance', 0)
    state_initial = state.get('initial_balance', 0)

    balance_info = api_client.get_balance()
    exchange_balance = float(balance_info['balance']['balance'])
    available_margin = float(balance_info['balance']['availableMargin'])

    print(f'State File:')
    print(f'  Initial Balance: ${state_initial:.2f}')
    print(f'  Current Balance: ${state_balance:.2f}')
    print(f'  Change: ${state_balance - state_initial:+.2f}')

    print(f'\nExchange API:')
    print(f'  Total Balance: ${exchange_balance:.2f}')
    print(f'  Available Margin: ${available_margin:.2f}')

    balance_diff = exchange_balance - state_balance
    print(f'\n⚠️  Difference: ${balance_diff:+.2f} ({balance_diff/state_balance*100:+.2f}%)')

    # 2. Position Comparison
    print('\n\n🎯 POSITION COMPARISON')
    print('-'*100)

    state_position = state.get('position')
    print(f'State File Position:')
    if state_position and state_position.get('status') == 'OPEN':
        print(f'  Status: {state_position.get("status")}')
        print(f'  Side: {state_position.get("side")}')
        print(f'  Entry Price: ${state_position.get("entry_price"):,.2f}')
        print(f'  Quantity: {state_position.get("quantity")}')
        print(f'  Entry Time: {state_position.get("entry_time")}')
        print(f'  Order ID: {state_position.get("order_id")}')
    else:
        print(f'  No open position')

    print(f'\nExchange API Position:')
    positions = api_client.exchange.fetch_positions(['BTC/USDT:USDT'])
    open_positions = [p for p in positions if float(p.get('contracts', 0)) > 0]

    if open_positions:
        for pos in open_positions:
            print(f'  Position ID: {pos["info"]["positionId"]}')
            print(f'  Side: {pos.get("side").upper()}')
            print(f'  Entry Price: ${pos.get("entryPrice"):,.2f}')
            print(f'  Quantity: {pos.get("contracts")}')
            print(f'  Unrealized P&L: ${pos.get("unrealizedPnl"):.2f}')
            print(f'  Realized P&L: ${pos.get("realizedPnl"):.2f}')
            print(f'  Leverage: {pos.get("leverage")}x')
    else:
        print(f'  No open positions')

    # 3. Recent Orders from Exchange
    print('\n\n📋 RECENT ORDERS FROM EXCHANGE (Last 20)')
    print('-'*100)

    orders = api_client.exchange.fetch_orders('BTC/USDT:USDT', limit=20)

    # Filter current session
    session_start_str = state.get('session_start', '')
    if session_start_str:
        session_start = datetime.fromisoformat(session_start_str)
    else:
        session_start = datetime(2025, 10, 22, 2, 8, 57)  # Default

    print(f'Session Start: {session_start}')
    print()

    current_session_orders = []
    for order in orders:
        order_time_str = order.get('datetime', '')
        if order_time_str:
            try:
                order_time = datetime.fromisoformat(order_time_str.replace('Z', '+00:00'))
                if order_time.replace(tzinfo=None) > session_start:
                    current_session_orders.append(order)
            except:
                pass

    print(f'Found {len(current_session_orders)} orders in current session:\n')

    for i, order in enumerate(current_session_orders, 1):
        fee_info = order.get('fee', {})
        fee_cost = float(fee_info.get('cost', 0)) if fee_info.get('cost') else 0.0

        print(f'{i:2d}. {order["id"]} | {order.get("datetime", "N/A")[:19]}')
        print(f'    {order["type"]:6s} {order["side"]:4s} | Price: ${order.get("price", 0):>10,.2f} | '
              f'Amount: {order.get("amount", 0):>8.4f} | Fee: ${fee_cost:.4f}')
        print(f'    Status: {order["status"]}')
        print()

    # 4. Trades Comparison
    print('\n📈 TRADES COMPARISON')
    print('-'*100)

    state_trades = state.get('trades', [])
    state_closed = [t for t in state_trades if t.get('status') == 'CLOSED']

    print(f'State File Closed Trades: {len(state_closed)}')
    for i, trade in enumerate(state_closed, 1):
        print(f'\n{i}. Order ID: {trade.get("order_id")}')
        print(f'   Side: {trade.get("side")} | Entry: ${trade.get("entry_price"):,.2f} → '
              f'Exit: ${trade.get("exit_price"):,.2f}')
        print(f'   Entry Fee: ${trade.get("entry_fee", 0):.2f} | Exit Fee: ${trade.get("exit_fee", 0):.2f} | '
              f'Total: ${trade.get("total_fee", 0):.2f}')
        print(f'   P&L (gross): ${trade.get("pnl_usd", 0):.2f} | P&L (net): ${trade.get("pnl_usd_net", 0):.2f}')
        print(f'   Exit Reason: {trade.get("exit_reason", "N/A")}')
        print(f'   Fees Reconciled: {trade.get("fees_reconciled", False)}')

    # 5. Cross-check orders vs trades
    print('\n\n🔄 CROSS-CHECK: Orders vs Trades')
    print('-'*100)

    state_order_ids = set()
    for trade in state_trades:
        if trade.get('order_id'):
            state_order_ids.add(trade.get('order_id'))
        if trade.get('close_order_id'):
            state_order_ids.add(trade.get('close_order_id'))

    exchange_order_ids = set(order['id'] for order in current_session_orders)

    print(f'State File Order IDs: {len(state_order_ids)}')
    print(f'Exchange Order IDs (session): {len(exchange_order_ids)}')

    missing_in_state = exchange_order_ids - state_order_ids
    missing_in_exchange = state_order_ids - exchange_order_ids

    if missing_in_state:
        print(f'\n⚠️  Orders in Exchange but NOT in State ({len(missing_in_state)}):')
        for oid in missing_in_state:
            matching_order = next((o for o in current_session_orders if o['id'] == oid), None)
            if matching_order:
                print(f'   - {oid} ({matching_order["side"]} {matching_order["type"]} @ ${matching_order.get("price", 0):.2f})')

    if missing_in_exchange:
        print(f'\n⚠️  Orders in State but NOT in Exchange ({len(missing_in_exchange)}):')
        for oid in missing_in_exchange:
            if oid != 'N/A' and oid != 'EXISTING_FROM_EXCHANGE':
                print(f'   - {oid}')

    print('\n' + '='*100)

if __name__ == '__main__':
    # Load API keys
    config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'config', 'api_keys.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Initialize mainnet client
    api_key = config['bingx']['mainnet']['api_key']
    secret_key = config['bingx']['mainnet']['secret_key']
    client = BingXClient(api_key, secret_key, testnet=False)

    # State file path
    state_file = os.path.join(os.path.dirname(__file__), '..', '..', 'results',
                              'opportunity_gating_bot_4x_state.json')

    compare_data(state_file, client)
