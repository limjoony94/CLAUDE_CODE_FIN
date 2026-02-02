#!/usr/bin/env python3
"""
Reconcile State File from Exchange Ground Truth
Fetches closed position data from exchange and updates state file.
No complex calculations - uses exchange data as-is.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import yaml
import json
from datetime import datetime, timedelta
from api.bingx_client import BingXClient
from collections import defaultdict

def get_closed_orders_from_exchange(client, days=7):
    """Get all closed orders from exchange (ground truth)"""
    try:
        # Use BingX direct API for complete data
        params = {
            'symbol': 'BTC-USDT',
            'startTime': int((datetime.now() - timedelta(days=days)).timestamp() * 1000),
            'endTime': int(datetime.now().timestamp() * 1000)
        }
        result = client.exchange.swapV2PrivateGetTradeAllOrders(params)

        if result.get('code') == '0':
            orders = result.get('data', {}).get('orders', [])
            # Filter only FILLED orders (exclude CANCELLED)
            filled_orders = [o for o in orders if o.get('status') == 'FILLED']
            return filled_orders
        else:
            print(f"❌ API Error: {result.get('msg')}")
            return []
    except Exception as e:
        print(f"❌ Failed to fetch orders: {e}")
        return []

def group_orders_by_position(orders):
    """Group orders by positionID to reconstruct closed positions"""
    positions = defaultdict(list)

    for order in orders:
        position_id = order.get('positionID')
        if position_id:
            positions[position_id].append(order)

    return positions

def identify_closed_positions(positions):
    """Identify fully closed positions (have both entry and exit)"""
    closed_positions = []

    for position_id, orders in positions.items():
        # Sort by time
        orders = sorted(orders, key=lambda x: int(x.get('time', 0)))

        # Check if position is closed (has exit orders)
        has_exit = any(
            float(order.get('profit', 0)) != 0 or
            order.get('type') == 'STOP_MARKET' and order.get('status') == 'FILLED'
            for order in orders
        )

        if has_exit:
            # Find entry and exit orders
            entry_orders = []
            exit_orders = []

            for order in orders:
                profit = float(order.get('profit', 0))
                order_type = order.get('type', '')

                # Exit order: has profit != 0 OR is filled STOP order
                if profit != 0 or (order_type == 'STOP_MARKET' and order.get('status') == 'FILLED'):
                    exit_orders.append(order)
                else:
                    entry_orders.append(order)

            if entry_orders and exit_orders:
                closed_positions.append({
                    'position_id': position_id,
                    'entry_orders': entry_orders,
                    'exit_orders': exit_orders,
                    'all_orders': orders
                })

    return closed_positions

def calculate_position_pnl(position):
    """Calculate position P&L from exchange data (ground truth)"""
    entry_orders = position['entry_orders']
    exit_orders = position['exit_orders']

    # Calculate total entry
    total_entry_qty = sum(float(o.get('executedQty', 0)) for o in entry_orders)
    total_entry_value = sum(float(o.get('cumQuote', 0)) for o in entry_orders)
    avg_entry_price = total_entry_value / total_entry_qty if total_entry_qty > 0 else 0

    # Calculate total exit
    total_exit_qty = sum(float(o.get('executedQty', 0)) for o in exit_orders)
    total_exit_value = sum(float(o.get('cumQuote', 0)) for o in exit_orders)
    avg_exit_price = total_exit_value / total_exit_qty if total_exit_qty > 0 else 0

    # Get realized P&L from exit orders (GROUND TRUTH - includes slippage!)
    realized_pnl = sum(float(o.get('profit', 0)) for o in exit_orders)

    # Get fees from ALL orders (entry + exit)
    total_fees = sum(abs(float(o.get('commission', 0))) for o in entry_orders + exit_orders)

    # Net P&L = Realized P&L - Fees
    net_pnl = realized_pnl - total_fees

    # Determine side
    first_order = position['all_orders'][0]
    side = first_order.get('side')  # BUY or SELL

    # Get timestamps
    entry_time = min(int(o.get('time', 0)) for o in entry_orders)
    exit_time = max(int(o.get('time', 0)) for o in exit_orders)

    return {
        'position_id': position['position_id'],
        'side': side,
        'entry_price': avg_entry_price,
        'exit_price': avg_exit_price,
        'quantity': total_entry_qty,
        'entry_time': entry_time,
        'exit_time': exit_time,
        'realized_pnl': realized_pnl,  # Ground truth from exchange
        'total_fees': total_fees,
        'net_pnl': net_pnl,
        'entry_order_ids': [o.get('orderId') for o in entry_orders],
        'exit_order_ids': [o.get('orderId') for o in exit_orders]
    }

def reconcile_state_file(state_file_path, api_client, bot_start_time=None, days=7):
    """
    Reconcile state file from exchange ground truth.
    Only updates positions that closed AFTER bot start time.
    """
    print('='*80)
    print('Reconciling State File from Exchange Ground Truth')
    print('='*80)

    # Load state file
    with open(state_file_path, 'r') as f:
        state = json.load(f)

    # Get bot start time (if provided)
    if bot_start_time is None:
        # Try start_time first, then session_start, default to 0
        bot_start_time = state.get('start_time')
        if bot_start_time is None:
            session_start = state.get('session_start')
            if session_start:
                bot_start_time = datetime.fromisoformat(session_start).timestamp()
                print(f'ℹ️  Using session_start as bot_start_time')
            else:
                bot_start_time = 0
                print(f'⚠️  No start_time or session_start found, using 0 (ALL historical data)')
        elif isinstance(bot_start_time, str):
            bot_start_time = datetime.fromisoformat(bot_start_time).timestamp()

    print(f'\n📅 Bot Start Time: {datetime.fromtimestamp(bot_start_time).strftime("%Y-%m-%d %H:%M:%S")}')
    print(f'📅 Fetching orders from last {days} days...')

    # Fetch all closed orders from exchange
    orders = get_closed_orders_from_exchange(api_client, days=days)
    print(f'✅ Fetched {len(orders)} filled orders from exchange')

    # Group by position
    positions_dict = group_orders_by_position(orders)
    print(f'📊 Found {len(positions_dict)} unique positions')

    # Identify closed positions
    closed_positions = identify_closed_positions(positions_dict)
    print(f'✅ Identified {len(closed_positions)} closed positions')

    # Calculate P&L for each closed position
    print(f'\n{"="*80}')
    print('Closed Positions (Ground Truth from Exchange):')
    print(f'{"="*80}\n')

    reconciled_trades = []
    total_net_pnl = 0

    for i, position in enumerate(closed_positions, 1):
        pnl_data = calculate_position_pnl(position)

        # Only include positions closed AFTER bot start
        if pnl_data['exit_time'] / 1000 >= bot_start_time:
            print(f"Position {i}:")
            print(f"  Position ID: {pnl_data['position_id']}")
            print(f"  Side: {pnl_data['side']}")
            print(f"  Entry: ${pnl_data['entry_price']:,.2f} @ {datetime.fromtimestamp(pnl_data['entry_time']/1000).strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"  Exit:  ${pnl_data['exit_price']:,.2f} @ {datetime.fromtimestamp(pnl_data['exit_time']/1000).strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"  Quantity: {pnl_data['quantity']}")
            print(f"  Realized P&L: ${pnl_data['realized_pnl']:.2f} (ground truth)")
            print(f"  Total Fees: ${pnl_data['total_fees']:.2f}")
            print(f"  Net P&L: ${pnl_data['net_pnl']:.2f}")
            print()

            reconciled_trades.append(pnl_data)
            total_net_pnl += pnl_data['net_pnl']

    print(f'{"="*80}')
    print(f'\n📈 Summary:')
    print(f'   Total Closed Positions (after bot start): {len(reconciled_trades)}')
    print(f'   Total Net P&L: ${total_net_pnl:.2f}')

    # Update state file with ground truth data
    print(f'\n💾 Updating state file with ground truth data...')

    state_trades = state.get('trades', [])

    # Remove old reconciled trades (from previous reconciliation runs)
    old_reconciled_trades = [t for t in state_trades if t.get('exchange_reconciled', False)]
    if old_reconciled_trades:
        print(f'🗑️  Removing {len(old_reconciled_trades)} old reconciled trades...')
        state_trades = [t for t in state_trades if not t.get('exchange_reconciled', False)]
        state['trades'] = state_trades

    updated_count = 0
    new_count = 0

    for pnl_data in reconciled_trades:
        # Try to find matching trade in state file by entry order ID
        entry_order_id = pnl_data['entry_order_ids'][0] if pnl_data['entry_order_ids'] else None

        matching_trade = None
        if entry_order_id:
            matching_trade = next(
                (t for t in state_trades if str(t.get('order_id')) == str(entry_order_id)),
                None
            )

        if matching_trade:
            # Update existing trade with ground truth
            matching_trade['pnl_usd'] = pnl_data['realized_pnl']
            matching_trade['total_fee'] = pnl_data['total_fees']
            matching_trade['pnl_usd_net'] = pnl_data['net_pnl']
            matching_trade['entry_fee'] = sum(
                abs(float(o.get('commission', 0))) for o in
                next(p for p in closed_positions if p['position_id'] == pnl_data['position_id'])['entry_orders']
            )
            matching_trade['exit_fee'] = sum(
                abs(float(o.get('commission', 0))) for o in
                next(p for p in closed_positions if p['position_id'] == pnl_data['position_id'])['exit_orders']
            )
            matching_trade['exchange_reconciled'] = True
            matching_trade['position_id_exchange'] = pnl_data['position_id']
            updated_count += 1
            print(f'   ✅ Updated trade {entry_order_id} with ground truth')
        else:
            # This is a new trade not in state file (e.g., manual trade)
            # Create new trade entry from exchange data
            new_trade = {
                'order_id': entry_order_id,
                'position_id_exchange': pnl_data['position_id'],
                'side': pnl_data['side'],
                'entry_price': pnl_data['entry_price'],
                'exit_price': pnl_data['exit_price'],
                'quantity': pnl_data['quantity'],
                'entry_time': datetime.fromtimestamp(pnl_data['entry_time'] / 1000).isoformat(),
                'close_time': datetime.fromtimestamp(pnl_data['exit_time'] / 1000).isoformat(),
                'pnl_usd': pnl_data['realized_pnl'],
                'total_fee': pnl_data['total_fees'],
                'pnl_usd_net': pnl_data['net_pnl'],
                'entry_fee': sum(
                    abs(float(o.get('commission', 0))) for o in
                    next(p for p in closed_positions if p['position_id'] == pnl_data['position_id'])['entry_orders']
                ),
                'exit_fee': sum(
                    abs(float(o.get('commission', 0))) for o in
                    next(p for p in closed_positions if p['position_id'] == pnl_data['position_id'])['exit_orders']
                ),
                'status': 'CLOSED',
                'exit_reason': 'Reconciled from exchange',
                'exchange_reconciled': True,
                'manual_trade': True  # Mark as manual since not in bot state
            }
            state_trades.append(new_trade)
            new_count += 1
            print(f'   ➕ Added new trade {entry_order_id} from exchange')

    # Save updated state
    with open(state_file_path, 'w') as f:
        json.dump(state, f, indent=2)

    print(f'\n✅ State file updated!')
    print(f'   Updated: {updated_count} trades')
    print(f'   Added: {new_count} new trades')
    print(f'\n{"="*80}')

    return reconciled_trades

if __name__ == '__main__':
    # Load API credentials
    with open('config/api_keys.yaml', 'r') as f:
        config = yaml.safe_load(f)

    api_key = config['bingx']['mainnet']['api_key']
    api_secret = config['bingx']['mainnet']['secret_key']

    client = BingXClient(api_key, api_secret, testnet=False)

    state_file = os.path.join(
        os.path.dirname(__file__), '..', '..',
        'results', 'opportunity_gating_bot_4x_state.json'
    )

    # Reconcile state file from exchange
    reconcile_state_file(state_file, client, days=7)
