"""
Check Position History via API
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.api.bingx_client import BingXClient
import yaml
from datetime import datetime

# Load API keys
PROJECT_ROOT = Path(__file__).parent.parent.parent
with open(PROJECT_ROOT / 'config' / 'api_keys.yaml', 'r') as f:
    config = yaml.safe_load(f)
    api_config = config['bingx']['mainnet']

client = BingXClient(api_config['api_key'], api_config['secret_key'], testnet=False)

# Get all orders for the position
position_id = '1979572874504257536'
print(f'=== All Orders for Position {position_id} ===\n')

# Get orders
orders = client.exchange.fetch_orders('BTC/USDT:USDT', limit=50)

position_orders = [o for o in orders if o.get('info', {}).get('positionID') == position_id]

print(f'Found {len(position_orders)} orders for this position:\n')

total_buy_qty = 0
total_buy_value = 0
total_sell_qty = 0
total_sell_value = 0
total_buy_fee = 0
total_sell_fee = 0

for order in sorted(position_orders, key=lambda x: x['timestamp']):
    dt = datetime.fromtimestamp(order['timestamp']/1000)
    print(f'Order ID: {order["id"]}')
    print(f'  Time: {dt} KST')
    print(f'  Side: {order["side"].upper()}')
    print(f'  Price: {order.get("average", order.get("price"))}')
    print(f'  Amount: {order["filled"]} BTC')
    print(f'  Cost: ${order.get("cost", 0):.2f}')

    fee_info = order.get('fee', {})
    fee = fee_info.get('cost', 0) if fee_info else 0
    fee = float(fee) if fee else 0
    print(f'  Fee: ${fee:.6f}')
    print(f'  Status: {order["status"]}')

    # Calculate totals
    if order['side'] == 'buy':
        total_buy_qty += order['filled']
        total_buy_value += order.get('cost', 0)
        total_buy_fee += fee
    else:
        total_sell_qty += order['filled']
        total_sell_value += order.get('cost', 0)
        total_sell_fee += fee

    # Check for profit in order info
    if 'profit' in order.get('info', {}):
        print(f'  Realized Profit (from order): ${order["info"]["profit"]}')

    print()

print('='*70)
print('SUMMARY:')
print('='*70)
print(f'Total BUY: {total_buy_qty:.4f} BTC @ avg ${total_buy_value/total_buy_qty if total_buy_qty > 0 else 0:.2f}')
print(f'  Cost: ${total_buy_value:.2f}')
print(f'  Fees: ${total_buy_fee:.6f}')
print()
print(f'Total SELL: {total_sell_qty:.4f} BTC @ avg ${total_sell_value/total_sell_qty if total_sell_qty > 0 else 0:.2f}')
print(f'  Value: ${total_sell_value:.2f}')
print(f'  Fees: ${total_sell_fee:.6f}')
print()
print(f'Net Position: {total_buy_qty - total_sell_qty:.4f} BTC')
print()

# Calculate P&L
if total_sell_qty > 0:
    gross_pnl = total_sell_value - (total_buy_value * (total_sell_qty / total_buy_qty))
    total_fees = total_buy_fee + total_sell_fee
    net_pnl = gross_pnl - total_fees

    print(f'Gross P&L: ${gross_pnl:.2f}')
    print(f'Total Fees: ${total_fees:.6f}')
    print(f'Net P&L: ${net_pnl:.2f}')

# Also check account balance changes
print('\n' + '='*70)
print('ACCOUNT BALANCE CHECK:')
print('='*70)

try:
    balance = client.exchange.fetch_balance()
    usdt_balance = balance.get('USDT', {})
    print(f"Current USDT Balance: ${usdt_balance.get('total', 'N/A')}")
    print(f"  Free: ${usdt_balance.get('free', 'N/A')}")
    print(f"  Used: ${usdt_balance.get('used', 'N/A')}")
except Exception as e:
    print(f"Error fetching balance: {e}")
