#!/usr/bin/env python3
"""
Reconcile Historical Trade Fees from Exchange
Fetches actual fees from exchange orders and updates state file.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import yaml
import json
from datetime import datetime
from api.bingx_client import BingXClient

def get_order_fee(client, order_id, symbol='BTC/USDT:USDT'):
    """Get actual fee from exchange order"""
    try:
        order = client.exchange.fetch_order(order_id, symbol)
        fee_info = order.get('fee', {})
        fee_cost = fee_info.get('cost', 0)
        return float(fee_cost) if fee_cost else 0.0
    except Exception as e:
        print(f'⚠️  Could not fetch fee for order {order_id}: {e}')
        return None

def reconcile_fees(state_file_path, api_client):
    """Reconcile fees from exchange for all closed trades"""

    # Read state file
    with open(state_file_path, 'r') as f:
        state = json.load(f)

    trades = state.get('trades', [])
    closed_trades = [t for t in trades if t.get('status') == 'CLOSED']

    print(f'\n📊 Found {len(closed_trades)} closed trades')
    print(f'='*80)

    reconciled_count = 0
    failed_count = 0
    total_fees = 0.0

    for i, trade in enumerate(closed_trades, 1):
        trade_id = trade.get('order_id', 'unknown')

        # Skip if already reconciled
        if trade.get('fees_reconciled', False):
            print(f'{i}. Trade {trade_id}: Already reconciled (skipping)')
            continue

        print(f'\n{i}. Trade {trade_id}:')

        fees_updated = False
        entry_fee_val = 0.0
        exit_fee_val = 0.0

        # Get entry fee
        entry_order_id = trade.get('order_id')
        if entry_order_id and entry_order_id != 'N/A':
            entry_fee = get_order_fee(api_client, entry_order_id)
            if entry_fee is not None:
                trade['entry_fee'] = entry_fee
                entry_fee_val = entry_fee
                fees_updated = True
                print(f'   Entry fee: ${entry_fee:.2f}')
            else:
                failed_count += 1

        # Get exit fee
        exit_order_id = trade.get('close_order_id')
        if exit_order_id and exit_order_id != 'N/A':
            exit_fee = get_order_fee(api_client, exit_order_id)
            if exit_fee is not None:
                trade['exit_fee'] = exit_fee
                exit_fee_val = exit_fee
                fees_updated = True
                print(f'   Exit fee: ${exit_fee:.2f}')
            else:
                # Exit order might not exist for stop loss
                print(f'   Exit fee: $0.00 (no order found)')
                trade['exit_fee'] = 0.0

        # Calculate total fee and net P&L
        if fees_updated:
            total_fee = entry_fee_val + exit_fee_val
            pnl_usd = trade.get('pnl_usd', 0.0)

            trade['total_fee'] = total_fee
            trade['pnl_usd_net'] = pnl_usd - total_fee
            trade['fees_reconciled'] = True

            total_fees += total_fee
            reconciled_count += 1

            print(f'   Total fee: ${total_fee:.2f}')
            print(f'   P&L (before fees): ${pnl_usd:.2f}')
            print(f'   P&L (after fees): ${pnl_usd - total_fee:.2f}')
            print(f'   ✅ Reconciled')

    print(f'\n{'='*80}')
    print(f'\n📈 Reconciliation Summary:')
    print(f'   Trades processed: {len(closed_trades)}')
    print(f'   Successfully reconciled: {reconciled_count}')
    print(f'   Failed: {failed_count}')
    print(f'   Total fees: ${total_fees:.2f}')

    if reconciled_count > 0:
        # Save updated state
        print(f'\n💾 Saving updated state file...')
        with open(state_file_path, 'w') as f:
            json.dump(state, f, indent=2)
        print(f'✅ State file updated!')
    else:
        print(f'\nℹ️  No changes needed (all trades already reconciled or no fees found)')

    return reconciled_count, failed_count, total_fees

if __name__ == '__main__':
    print('='*80)
    print('  🔄 HISTORICAL FEE RECONCILIATION')
    print('='*80)

    # Load API keys
    config_path = os.path.join(os.path.dirname(__file__), '..', '..', 'config', 'api_keys.yaml')
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Initialize mainnet client
    api_key = config['bingx']['mainnet']['api_key']
    secret_key = config['bingx']['mainnet']['secret_key']
    client = BingXClient(api_key, secret_key, testnet=False)

    print(f'✅ Connected to BingX mainnet')

    # State file path
    state_file = os.path.join(os.path.dirname(__file__), '..', '..', 'results',
                              'opportunity_gating_bot_4x_state.json')

    print(f'📁 State file: {state_file}')

    # Reconcile fees
    reconciled, failed, total = reconcile_fees(state_file, client)

    print(f'\n{'='*80}')
    print(f'✅ Fee reconciliation complete!')
    print(f'='*80)
