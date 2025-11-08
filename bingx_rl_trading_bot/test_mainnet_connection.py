"""
Test Mainnet Connection
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.api.bingx_client import BingXClient

# API Keys (same as testnet)
API_KEY = "NyXnyvNWKyIAbRrsFs5CLlZevncCBueUAZLWvGEqbZBnpb69zUXTzTIP0cqEyg5zXAsPvajwrggEiIamlg"
API_SECRET = "XLi5Q6ljhHs5WDpVZivadiBTNkaub0VerM4GR166oEuMpF8FE6xZXq0K3RN9VEiFMZG1mAJcUL1EktQTgceA"

print("="*80)
print("MAINNET CONNECTION TEST")
print("="*80)
print()

# Initialize mainnet client
print("Initializing BingX Mainnet client...")
client = BingXClient(api_key=API_KEY, secret_key=API_SECRET, testnet=False)
print("✅ Client initialized")
print()

# Test 1: Get balance
print("Test 1: Get Balance")
print("-"*80)
try:
    balance_info = client.get_balance()
    balance = float(balance_info.get('balance', {}).get('balance', 0))
    available = float(balance_info.get('balance', {}).get('availableMargin', 0))

    print(f"✅ Balance retrieved successfully")
    print(f"   Total Balance: ${balance:,.2f} USDT")
    print(f"   Available Margin: ${available:,.2f} USDT")
    print()
except Exception as e:
    print(f"❌ Balance retrieval failed: {e}")
    print()

# Test 2: Get current positions
print("Test 2: Get Current Positions")
print("-"*80)
try:
    positions = client.get_positions("BTC-USDT")
    print(f"✅ Positions retrieved successfully")
    if positions:
        for pos in positions:
            print(f"   Position: {pos['positionSide']} {pos['positionAmt']} BTC")
            print(f"   Entry Price: ${pos['entryPrice']}")
            print(f"   Unrealized P&L: ${pos['unrealizedProfit']}")
    else:
        print(f"   No open positions")
    print()
except Exception as e:
    print(f"❌ Position retrieval failed: {e}")
    print()

# Test 3: Get market data
print("Test 3: Get Market Data")
print("-"*80)
try:
    ticker = client.get_ticker("BTC-USDT")
    print(f"✅ Market data retrieved successfully")
    print(f"   BTC-USDT Price: ${ticker['lastPrice']}")
    print()
except Exception as e:
    print(f"❌ Market data retrieval failed: {e}")
    print()

print("="*80)
print("MAINNET CONNECTION TEST COMPLETE")
print("="*80)
print()
print("⚠️ WARNING: You are connected to MAINNET")
print("   All trades will use REAL MONEY")
print("   Make sure you understand the risks")
print()
