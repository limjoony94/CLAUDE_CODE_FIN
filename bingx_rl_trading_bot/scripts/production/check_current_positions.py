"""
Check Current Positions on BingX
================================

Quick script to check current open positions and recent orders.
"""

import sys
from pathlib import Path
import yaml
import os

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.api.bingx_client import BingXClient

CONFIG_DIR = PROJECT_ROOT / "config"

print("="*80)
print("CHECKING CURRENT POSITIONS")
print("="*80)

# Load API keys
def load_api_keys():
    api_keys_file = CONFIG_DIR / "api_keys.yaml"
    if api_keys_file.exists():
        with open(api_keys_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            return config.get('bingx', {}).get('testnet', {})
    return {}

_api_config = load_api_keys()
API_KEY = _api_config.get('api_key', os.getenv("BINGX_API_KEY", ""))
API_SECRET = _api_config.get('secret_key', os.getenv("BINGX_API_SECRET", ""))
USE_TESTNET = False  # Mainnet

# Initialize client
print("\n1. Initializing BingX client...")
client = BingXClient(
    api_key=API_KEY,
    secret_key=API_SECRET,
    testnet=USE_TESTNET
)
print(f"   ✅ Connected to {'Testnet' if USE_TESTNET else 'Mainnet'}")

# Get balance
print("\n2. Checking balance...")
balance_info = client.get_balance()
if isinstance(balance_info, dict):
    balance = float(balance_info.get('USDT', {}).get('free', 0))
else:
    balance = float(balance_info)
print(f"   Balance: ${balance:,.2f}")

# Get current positions
print("\n3. Checking open positions...")
try:
    positions = client.exchange.fetch_positions(['BTC/USDT:USDT'])

    open_positions = [p for p in positions if float(p.get('contracts', 0)) != 0]

    if open_positions:
        print(f"   ⚠️  WARNING: {len(open_positions)} OPEN POSITION(S) FOUND!")
        for pos in open_positions:
            side = pos.get('side', 'unknown').upper()
            contracts = float(pos.get('contracts', 0))
            entry_price = float(pos.get('entryPrice', 0))
            notional = float(pos.get('notional', 0))
            unrealized_pnl = float(pos.get('unrealizedPnl', 0))
            leverage = float(pos.get('leverage', 1))

            print(f"\n   Position {open_positions.index(pos) + 1}:")
            print(f"     Side: {side}")
            print(f"     Contracts: {contracts} BTC")
            print(f"     Entry Price: ${entry_price:,.2f}")
            print(f"     Notional: ${notional:,.2f}")
            print(f"     Leverage: {leverage}x")
            print(f"     Unrealized P&L: ${unrealized_pnl:+,.2f}")
    else:
        print("   ✅ No open positions")

except Exception as e:
    print(f"   ❌ Error fetching positions: {e}")

# Get recent orders
print("\n4. Checking recent orders...")
try:
    # Get last 10 orders
    orders = client.exchange.fetch_orders('BTC/USDT:USDT', limit=10)

    print(f"   Found {len(orders)} recent orders:")
    for i, order in enumerate(orders[-5:], 1):  # Last 5 orders
        order_id = order.get('id', 'N/A')
        side = order.get('side', 'unknown').upper()
        order_type = order.get('type', 'unknown').upper()
        amount = float(order.get('amount', 0))
        filled = float(order.get('filled', 0))
        status = order.get('status', 'unknown')
        timestamp = order.get('datetime', 'N/A')

        print(f"\n   Order {i}:")
        print(f"     ID: {order_id}")
        print(f"     Time: {timestamp}")
        print(f"     Side: {side} {order_type}")
        print(f"     Amount: {amount:.6f} BTC")
        print(f"     Filled: {filled:.6f} BTC")
        print(f"     Status: {status}")

except Exception as e:
    print(f"   ❌ Error fetching orders: {e}")

print("\n" + "="*80)
print("POSITION CHECK COMPLETE")
print("="*80)
