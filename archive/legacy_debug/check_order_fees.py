#!/usr/bin/env python3
"""Check order fees from API and match with state file trades."""

import sys
import os
import json
from datetime import datetime
import yaml

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from api.bingx_client import BingXClient

def main():
    # Load API keys
    config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'config', 'api_keys.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Use mainnet
    api_key = config['bingx']['mainnet']['api_key']
    secret_key = config['bingx']['mainnet']['secret_key']

    # Initialize client
    client = BingXClient(api_key, secret_key, testnet=False)

    # Load state file
    state_path = os.path.join(os.path.dirname(__file__), '..', '..', 'results', 'opportunity_gating_bot_4x_state.json')
    with open(state_path, 'r') as f:
        state = json.load(f)

    # Load API trades
    trades_path = os.path.join(os.path.dirname(__file__), 'api_trades.json')
    with open(trades_path, 'r') as f:
        api_trades = json.load(f)

    # Create lookup dict for API trades by order ID
    api_trades_by_order = {}
    for trade in api_trades:
        order_id = trade['order']
        if order_id not in api_trades_by_order:
            api_trades_by_order[order_id] = []
        api_trades_by_order[order_id].append(trade)

    print('=' * 100)
    print('FEE COMPARISON: STATE FILE vs API')
    print('=' * 100)

    # Check last 5 trades from state file
    recent_trades = list(reversed(state.get('trades', [])[-5:]))

    for i, state_trade in enumerate(recent_trades, 1):
        print(f'\n{"=" * 100}')
        print(f'State Trade #{i}')
        print(f'{"=" * 100}')

        side = state_trade.get('side', 'UNKNOWN')
        entry_price = state_trade.get('entry_price', 0)
        exit_price = state_trade.get('exit_price', 0)
        quantity = state_trade.get('quantity', 0)

        # State file fees
        entry_fee_state = state_trade.get('entry_fee', 0)
        exit_fee_state = state_trade.get('exit_fee', 0)
        total_fee_state = state_trade.get('total_fee', 0)
        pnl_net_state = state_trade.get('pnl_usd_net', state_trade.get('pnl_usd', 0))

        print(f"Side:            {side}")
        print(f"Entry:           ${entry_price:,.2f}")
        print(f"Exit:            ${exit_price:,.2f}")
        print(f"Quantity:        {quantity:.4f} BTC")

        print(f"\nState File Fees:")
        print(f"  Entry Fee:     ${entry_fee_state:.4f}")
        print(f"  Exit Fee:      ${exit_fee_state:.4f}")
        print(f"  Total Fee:     ${total_fee_state:.4f}")
        print(f"  Net P&L:       ${pnl_net_state:+.2f}")

        # Try to find matching API trade(s)
        entry_order_id = state_trade.get('order_id', '')

        # For manual trades, order_id might be multiple IDs separated by comma
        if state_trade.get('manual_trade') and 'orders' in state_trade:
            print(f"\n*** MANUAL TRADE with {len(state_trade['orders'])} orders ***")

            total_entry_fee = 0
            total_exit_fee = 0

            for order in state_trade['orders']:
                order_side = order.get('side', '')
                order_id = str(order.get('order_id', ''))
                order_fee = order.get('fee', 0)

                if order_id in api_trades_by_order:
                    api_trade = api_trades_by_order[order_id][0]
                    api_fee = float(api_trade.get('fee', {}).get('cost', 0))
                    print(f"  {order_side:4s} Order {order_id}: State ${order_fee:.4f} | API ${api_fee:.4f}")

                    if order_side.upper() == 'BUY':
                        total_entry_fee += api_fee
                    elif order_side.upper() == 'SELL':
                        total_exit_fee += api_fee
                else:
                    print(f"  {order_side:4s} Order {order_id}: NOT FOUND in API")

            print(f"\nRecalculated Fees from API:")
            print(f"  Entry Fee:     ${total_entry_fee:.4f}")
            print(f"  Exit Fee:      ${total_exit_fee:.4f}")
            print(f"  Total Fee:     ${total_entry_fee + total_exit_fee:.4f}")

            # Calculate fee discrepancy
            fee_diff = abs((total_entry_fee + total_exit_fee) - total_fee_state)
            if fee_diff > 0.01:
                print(f"  ⚠️  DISCREPANCY: ${fee_diff:.4f} difference")

        else:
            # Regular bot trade - check order_id
            if entry_order_id in api_trades_by_order:
                entry_api_trade = api_trades_by_order[entry_order_id][0]
                entry_fee_api = entry_api_trade.get('fee', {}).get('cost', 0)
                print(f"\nAPI Entry Fee:   ${entry_fee_api:.4f} (order {entry_order_id})")

                # Check if there's a known exit order (need to match by time/price)
                # For now, show if fees are missing
                if abs(entry_fee_state - entry_fee_api) > 0.01:
                    print(f"  ⚠️  MISMATCH: State ${entry_fee_state:.4f} vs API ${entry_fee_api:.4f}")
            else:
                print(f"\n⚠️  Entry order {entry_order_id} NOT FOUND in API")

    # Check if we need to fetch funding fees
    print(f'\n{"=" * 100}')
    print('FUNDING FEES (if available)')
    print(f'{"=" * 100}')
    print("\nNote: Funding fees may need to be fetched separately from income history")
    print("This requires additional API call to get_income_history()")

if __name__ == '__main__':
    main()
