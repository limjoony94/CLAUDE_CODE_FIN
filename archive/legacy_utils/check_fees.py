#!/usr/bin/env python3
"""Check actual fees and orders from exchange"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from api.bingx_client import BingXClient
import yaml
from datetime import datetime

# Load API keys
config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'config', 'api_keys.yaml')
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

api_key = config['bingx']['mainnet']['api_key']
secret_key = config['bingx']['mainnet']['secret_key']

client = BingXClient(api_key, secret_key, testnet=False)

# Get account balance
balance = client.get_balance()
print('=== ACCOUNT BALANCE ===')
print(f'Total Balance: ${float(balance["balance"]["balance"]):.2f}')
print(f'Available Margin: ${float(balance["balance"]["availableMargin"]):.2f}')
print()

# Get recent orders
print('=== RECENT ORDERS (Last 20) ===')
orders = client.exchange.fetch_orders('BTC/USDT:USDT', limit=20)

# Filter for current session (after 2025-10-22T02:08:57 KST = 2025-10-21T17:08:57 UTC)
session_start = datetime(2025, 10, 21, 17, 8, 57)
current_session_orders = []

for order in orders:
    order_time_str = order.get('datetime', '')
    if order_time_str:
        try:
            order_time = datetime.fromisoformat(order_time_str.replace('Z', '+00:00'))
            if order_time > session_start:
                current_session_orders.append(order)
        except:
            pass

print(f'\n=== CURRENT SESSION ORDERS (after {session_start}) ===')
print(f'Found {len(current_session_orders)} orders\n')

total_fees = 0.0
for i, order in enumerate(current_session_orders, 1):
    fee_info = order.get('fee', {})
    fee_cost = fee_info.get('cost', 0)
    fee_cost = float(fee_cost) if fee_cost else 0.0
    total_fees += fee_cost

    print(f'{i}. Order ID: {order["id"]}')
    print(f'   Time: {order.get("datetime", "N/A")}')
    print(f'   Type: {order["type"]} {order["side"]}')
    print(f'   Price: ${order.get("price", 0):.2f}')
    print(f'   Amount: {order.get("amount", 0):.8f}')
    print(f'   Cost: ${order.get("cost", 0):.2f}')
    print(f'   Fee: ${fee_cost:.4f} {fee_info.get("currency", "")}')
    print(f'   Status: {order["status"]}')
    print()

print(f'Total Fees (current session): ${total_fees:.2f}')
print()

# Check specific orders from state file
print('=== CHECKING ORDERS FROM STATE FILE ===')
order_ids = [
    '1980731025916116992',  # Trade #1 entry
    '1980765013355479040',  # Trade #2 entry
    '1980816546583494656',  # Trade #2 exit
    '1980836712356724736',  # Trade #3 entry (current)
    '1980836713140420608',  # Trade #3 stop loss
]

state_order_fees = 0.0
for order_id in order_ids:
    try:
        order = client.exchange.fetch_order(order_id, 'BTC/USDT:USDT')
        fee_info = order.get('fee', {})
        fee_cost = fee_info.get('cost', 0)
        fee_cost = float(fee_cost) if fee_cost else 0.0
        state_order_fees += fee_cost

        print(f'Order {order_id}:')
        print(f'  Time: {order.get("datetime", "N/A")}')
        print(f'  Type: {order.get("type", "?")} {order.get("side", "?")}')
        print(f'  Fee: ${fee_cost:.4f} {fee_info.get("currency", "")}')
        print()
    except Exception as e:
        print(f'Order {order_id}: Could not fetch - {e}')
        print()

print(f'Total Fees (from state file orders): ${state_order_fees:.2f}')
print()

# Get income history (funding fees, commission)
print('=== INCOME HISTORY (Fees & Funding) ===')
try:
    # BingX specific API call for income history
    income = client.exchange.fapiPrivateGetSwapV2UserIncome({
        'symbol': 'BTC-USDT',
        'limit': 20
    })

    if income and 'data' in income:
        total_commission = 0.0
        total_funding = 0.0

        for record in income['data']:
            income_type = record.get('incomeType', '')
            amount = float(record.get('income', 0))
            timestamp = int(record.get('time', 0)) / 1000
            dt = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')

            print(f'{dt} | {income_type}: ${amount:.4f}')

            if income_type == 'COMMISSION':
                total_commission += abs(amount)
            elif income_type == 'FUNDING_FEE':
                total_funding += amount

        print()
        print(f'Total Commission (fees): ${total_commission:.2f}')
        print(f'Total Funding Fees: ${total_funding:.2f}')
        print(f'Total Costs: ${total_commission + abs(total_funding):.2f}')
except Exception as e:
    print(f'Could not fetch income history: {e}')
