#!/usr/bin/env python3
"""
Verify if API balance is equity or realized-only balance
"""

import sys
from pathlib import Path
import yaml

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.api.bingx_client import BingXClient

def verify_equity():
    """Check if API balance includes unrealized P&L"""

    # Load API keys
    with open(project_root / 'config' / 'api_keys.yaml', 'r') as f:
        config = yaml.safe_load(f)
        api_key = config['bingx']['mainnet']['api_key']
        secret_key = config['bingx']['mainnet']['secret_key']

    client = BingXClient(api_key, secret_key, testnet=False)

    print('=' * 80)
    print('🔍 EQUITY CALCULATION VERIFICATION')
    print('=' * 80)
    print()

    # Get raw CCXT balance
    print('📊 CCXT Raw Balance Data:')
    raw_balance = client.exchange.fetch_balance()
    usdt_data = raw_balance.get('USDT', {})

    print(f"   total: ${usdt_data.get('total', 0):.2f}")
    print(f"   free: ${usdt_data.get('free', 0):.2f}")
    print(f"   used: ${usdt_data.get('used', 0):.2f}")
    print()

    # Get BingXClient balance (wrapper)
    print('📊 BingXClient Balance:')
    balance_data = client.get_balance()
    balance = float(balance_data.get('balance', {}).get('balance', 0))
    available = float(balance_data.get('balance', {}).get('availableMargin', 0))

    print(f"   balance: ${balance:.2f} (CCXT total)")
    print(f"   availableMargin: ${available:.2f} (CCXT free)")
    print()

    # Get position
    print('📊 Position Data:')
    positions = client.get_positions('BTC-USDT')
    if positions and len(positions) > 0:
        pos = positions[0]
        unrealized = float(pos.get('unrealizedProfit', 0))
        entry = float(pos.get('entryPrice', 0))
        qty = float(pos.get('positionAmt', 0))
        side = pos.get('positionSide', 'NONE')

        print(f"   Side: {side}")
        print(f"   Quantity: {qty:.4f} BTC")
        print(f"   Entry Price: ${entry:.2f}")
        print(f"   Unrealized P&L: ${unrealized:.2f}")
        print()

        # Calculate equity both ways
        print('🧮 EQUITY CALCULATION TEST:')
        print()

        method1_equity = balance + unrealized
        method2_equity = balance  # If balance is already equity

        print(f"   Method 1 (balance + unrealized):")
        print(f"   ${balance:.2f} + ${unrealized:.2f} = ${method1_equity:.2f}")
        print()

        print(f"   Method 2 (balance is equity):")
        print(f"   Equity = ${method2_equity:.2f}")
        print()

        # Check account info for total equity
        print('📊 CCXT Account Info (info field):')
        if 'info' in raw_balance:
            info = raw_balance['info']
            if isinstance(info, dict):
                print(f"   Raw info keys: {list(info.keys())}")

                # Check data field (BingX response)
                if 'data' in info:
                    data = info['data']
                    if isinstance(data, dict):
                        print(f"\n   BingX data fields:")
                        for key, value in data.items():
                            print(f"     {key}: {value}")
                    elif isinstance(data, list) and len(data) > 0:
                        print(f"\n   BingX data (first item):")
                        item = data[0]
                        for key, value in item.items():
                            print(f"     {key}: {value}")
        print()

        print('=' * 80)
        print('📝 CONCLUSION:')
        print('=' * 80)
        print()
        print('If balance INCLUDES unrealized P&L:')
        print(f'  → Equity should be: ${balance:.2f} (Method 2)')
        print(f'  → Calculation: equity = balance')
        print()
        print('If balance EXCLUDES unrealized P&L:')
        print(f'  → Equity should be: ${method1_equity:.2f} (Method 1)')
        print(f'  → Calculation: equity = balance + unrealized_pnl')
        print()

        # Hint: check if total - free = used matches margin requirement
        used = usdt_data.get('used', 0)
        print(f'Verification:')
        print(f'  used (margin): ${used:.2f}')
        print(f'  If unrealized IS included: used should = margin only')
        print(f'  If unrealized NOT included: used = margin (unrealized separate)')

    else:
        print('   No open position')

if __name__ == "__main__":
    verify_equity()
