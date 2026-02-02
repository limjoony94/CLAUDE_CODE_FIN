#!/usr/bin/env python3
"""
Check balance history from BingX API to identify cause of balance decrease
"""

import sys
from pathlib import Path
from datetime import datetime
import yaml

# Add project root to path
project_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(project_root))

from src.api.bingx_client import BingXClient

def check_balance_history():
    """Query balance history and income records from exchange"""

    # Load API keys
    with open(project_root / 'config' / 'api_keys.yaml', 'r') as f:
        config = yaml.safe_load(f)
        api_key = config['bingx']['mainnet']['api_key']
        secret_key = config['bingx']['mainnet']['secret_key']

    client = BingXClient(api_key, secret_key, testnet=False)

    # Reset time: 2025-10-25 03:46:44
    reset_time_ms = int(datetime(2025, 10, 25, 3, 46, 0).timestamp() * 1000)

    print('=' * 80)
    print('💰 Balance History Investigation')
    print('   Reset Time: 2025-10-25 03:46:44 KST')
    print('   Expected: $4,561.00')
    print('   Current: $4,539.64')
    print('   Decrease: $21.36')
    print('=' * 80)
    print()

    try:
        # Method 1: Check current balance
        print('📊 Current Balance from API:')
        balance = client.get_balance()
        print(f"   Balance: {balance}")
        print()

        # Method 2: Check closed position history (manual closures during bot shutdown)
        print('📋 Checking Closed Positions since reset...')
        print('   (User reported manually closing position during bot shutdown)')
        print()

        try:
            # Fetch my trades to see closed positions
            trades = client.exchange.fetch_my_trades(
                symbol='BTC/USDT:USDT',
                since=reset_time_ms,
                limit=100
            )

            if len(trades) == 0:
                print('   ⚠️  No trades found since reset')
                print()
            else:
                print(f'   ✅ Found {len(trades)} trades since reset')
                print()

                total_realized_pnl = 0
                total_fees = 0

                for trade in trades:
                    ts = datetime.fromtimestamp(trade['timestamp'] / 1000)
                    side = trade['side'].upper()
                    price = trade['price']
                    amount = trade['amount']
                    cost = trade['cost']

                    # Fee info
                    fee = trade.get('fee', {})
                    if isinstance(fee, dict):
                        fee_cost = float(fee.get('cost', 0))
                        fee_currency = fee.get('currency', 'USDT')
                    else:
                        fee_cost = 0
                        fee_currency = 'USDT'

                    if fee_currency == 'USDT':
                        total_fees += fee_cost

                    # Check if this is a closing trade (has realized PnL)
                    info = trade.get('info', {})
                    realized_pnl = float(info.get('realizedProfit', 0))
                    if realized_pnl != 0:
                        total_realized_pnl += realized_pnl

                    time_str = ts.strftime('%m-%d %H:%M:%S')
                    price_str = f'${price:>10,.2f}'
                    amount_str = f'{amount:.4f} BTC'
                    cost_str = f'${cost:>8,.2f}'
                    fee_str = f'${fee_cost:>6.2f}'

                    pnl_str = f'${realized_pnl:>8.2f}' if realized_pnl != 0 else '    -    '

                    print(f'   {time_str} | {side:4s} | {price_str} | {amount_str} | Fee: {fee_str} | P&L: {pnl_str}')

                print()
                print('=' * 80)
                print(f'   💵 Total Realized P&L: ${total_realized_pnl:.2f}')
                print(f'   💸 Total Fees: ${total_fees:.2f}')
                print(f'   📊 Net Impact: ${total_realized_pnl - total_fees:.2f}')
                print('=' * 80)
                print()

        except Exception as e:
            print(f'   ⚠️  Trade history not available: {e}')
            import traceback
            traceback.print_exc()
            print()

        # Method 3: Check funding fees separately
        print('💸 Checking Funding Fee Records...')
        try:
            # Try to get funding fee history
            funding_fees = client.exchange.fetch_funding_history(
                symbol='BTC/USDT:USDT',
                since=reset_time_ms,
                limit=100
            )

            if len(funding_fees) == 0:
                print('   ℹ️  No funding fees since reset')
                print()
            else:
                print(f'   ✅ Found {len(funding_fees)} funding fee records')
                print()

                total_funding = 0
                for fee in funding_fees:
                    ts = datetime.fromtimestamp(fee['timestamp'] / 1000)
                    amount = float(fee['amount'])
                    total_funding += amount

                    time_str = ts.strftime('%m-%d %H:%M:%S')
                    amount_str = f"${amount:>8.4f}"

                    print(f'   {time_str} | Funding Fee | {amount_str}')

                print()
                print(f'   💵 Total Funding Fees: ${total_funding:.2f}')

        except Exception as e:
            print(f'   ℹ️  Funding fee history not available: {e}')
            print()

        # Method 4: Check position history for any adjustments
        print('📈 Checking Position Changes...')
        try:
            positions = client.exchange.fetch_positions(['BTC/USDT:USDT'])

            for pos in positions:
                if float(pos.get('contracts', 0)) != 0:
                    side = pos.get('side', 'NONE')
                    size = float(pos.get('contracts', 0))
                    entry_price = float(pos.get('entryPrice', 0))
                    unrealized_pnl = float(pos.get('unrealizedPnl', 0))

                    print(f'   Position: {side} {size} BTC @ ${entry_price:.2f}')
                    print(f'   Unrealized P&L: ${unrealized_pnl:.2f}')
                    print()

        except Exception as e:
            print(f'   ⚠️  Could not fetch positions: {e}')
            print()

        # Summary
        print('=' * 80)
        print('📊 Summary')
        print('=' * 80)
        print()
        print('✅ Balance at Reset: $4,561.00')
        print('✅ Current Balance: $4,539.64')
        print('❓ Decrease: $21.36')
        print()
        print('Possible Causes:')
        print('1. Check income history above for funding fees')
        print('2. Check if position was partially closed/adjusted')
        print('3. Check for any commission or system fees')
        print()

    except Exception as e:
        print(f'❌ Error: {e}')
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_balance_history()
