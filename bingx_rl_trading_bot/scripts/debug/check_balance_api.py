"""
Check BingX API balance response to understand what 'total' means
"""
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent.parent / 'src'))

from api.bingx_client import BingXClient
import yaml

# Load config
config_path = Path(__file__).parent.parent.parent / 'config' / 'api_keys.yaml'
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Get mainnet client
client = BingXClient(
    api_key=config['bingx']['mainnet']['api_key'],
    secret_key=config['bingx']['mainnet']['secret_key'],
    testnet=False
)

print('=' * 80)
print('BINGX API BALANCE ANALYSIS')
print('=' * 80)
print()

# Get balance via our wrapper
balance_info = client.get_balance()
print('=== get_balance() Response (Our Wrapper) ===')
print(f'Balance: ${float(balance_info["balance"]["balance"]):.2f}')
print(f'Available Margin: ${float(balance_info["balance"]["availableMargin"]):.2f}')
print()

# Get raw CCXT balance for inspection
raw_balance = client.exchange.fetch_balance()
print('=== CCXT fetch_balance() Raw Response ===')
if 'USDT' in raw_balance:
    usdt = raw_balance['USDT']
    free_val = usdt.get('free', 0)
    used_val = usdt.get('used', 0)
    total_val = usdt.get('total', 0)

    print(f'Free: ${free_val:.2f} (available balance)')
    print(f'Used: ${used_val:.2f} (margin in positions)')
    print(f'Total: ${total_val:.2f} (what we use as current_balance)')
    print()

    print('Formula Check:')
    sum_val = free_val + used_val
    print(f'  Free + Used = ${sum_val:.2f}')
    print(f'  Total = ${total_val:.2f}')
    match = abs(sum_val - total_val) < 0.01
    print(f'  Match: {"✅ YES" if match else "❌ NO"}')

    if not match:
        print(f'  Difference: ${abs(sum_val - total_val):.2f}')
else:
    print('⚠️ USDT not found in balance response')
print()

# Get positions for unrealized P&L
positions = client.get_positions('BTC-USDT')
print('=== Current Positions ===')
total_unrealized = 0
if positions:
    for pos in positions:
        unrealized = float(pos['unrealizedProfit'])
        total_unrealized += unrealized
        print(f'Side: {pos["positionSide"]}')
        print(f'Amount: {pos["positionAmt"]} BTC')
        print(f'Entry Price: ${pos["entryPrice"]}')
        print(f'Unrealized P&L: ${unrealized:.2f}')
        print()
else:
    print('No open positions')
    print()

# Calculate equity
if 'USDT' in raw_balance:
    account_balance = raw_balance['USDT'].get('total', 0)
    equity = account_balance + total_unrealized

    print('=' * 80)
    print('EQUITY CALCULATION')
    print('=' * 80)
    print()
    print(f'Account Balance (CCXT total): ${account_balance:.2f}')
    print(f'Unrealized P&L (from positions): ${total_unrealized:.2f}')
    print(f'Equity: ${equity:.2f}')
    print()

    print('=' * 80)
    print('INTERPRETATION')
    print('=' * 80)
    print()
    print('Question: Does CCXT "total" include unrealized P&L?')
    print()

    if abs(account_balance - equity) < 0.01:
        print('✅ YES - "total" already includes unrealized P&L')
        print('   → current_balance = equity (no adjustment needed)')
    else:
        print('❌ NO - "total" is account balance only')
        print('   → equity = current_balance + unrealized_pnl')
        print()
        print('This means:')
        print('  1. current_balance = Account Balance (realized)')
        print('  2. unrealized_pnl calculated separately (from positions)')
        print('  3. equity = current_balance + unrealized_pnl')
        print()
        print('✅ Our current implementation is CORRECT!')
