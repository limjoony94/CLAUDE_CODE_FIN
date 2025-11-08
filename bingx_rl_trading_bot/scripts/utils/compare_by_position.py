#!/usr/bin/env python3
"""
Compare Exchange vs State - By POSITION (not by order)
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

import yaml
import json
from datetime import datetime, timedelta
from api.bingx_client import BingXClient
from collections import defaultdict

def main():
    print('='*100)
    print('EXCHANGE vs STATE COMPARISON - BY CLOSED POSITION')
    print('='*100)

    # Load API credentials
    with open('config/api_keys.yaml', 'r') as f:
        config = yaml.safe_load(f)

    api_key = config['bingx']['mainnet']['api_key']
    api_secret = config['bingx']['mainnet']['secret_key']

    client = BingXClient(api_key, api_secret, testnet=False)

    # Load state file
    state_file = 'results/opportunity_gating_bot_4x_state.json'
    with open(state_file, 'r') as f:
        state = json.load(f)

    # Get session start time
    session_start = state.get('session_start')
    if session_start:
        session_start_dt = datetime.fromisoformat(session_start)
        session_start_ts = session_start_dt.timestamp()
        print(f'\n📅 Bot Session Start: {session_start_dt.strftime("%Y-%m-%d %H:%M:%S")}')
    else:
        print('\n⚠️  No session_start found in state file!')
        return

    # Fetch orders from exchange
    print(f'\n📊 FETCHING FROM EXCHANGE (Last 7 days):')
    try:
        params = {
            'symbol': 'BTC-USDT',
            'startTime': int((datetime.now() - timedelta(days=7)).timestamp() * 1000),
            'endTime': int(datetime.now().timestamp() * 1000)
        }
        result = client.exchange.swapV2PrivateGetTradeAllOrders(params)

        if result.get('code') == '0':
            all_orders = result.get('data', {}).get('orders', [])
            filled_orders = [o for o in all_orders if o.get('status') == 'FILLED']

            # Filter by session_start
            recent_orders = [o for o in filled_orders if int(o.get('time', 0)) / 1000 >= session_start_ts]

            print(f'   Orders AFTER session_start: {len(recent_orders)} ✅')

            # Group by position
            positions = defaultdict(list)
            for order in recent_orders:
                position_id = order.get('positionID')
                if position_id:
                    positions[position_id].append(order)

            print(f'   Unique positions: {len(positions)}')

            # Identify closed positions
            exchange_closed_positions = []
            for position_id, orders in positions.items():
                orders = sorted(orders, key=lambda x: int(x.get('time', 0)))

                # Check if closed
                has_exit = any(
                    float(order.get('profit', 0)) != 0 or
                    order.get('type') == 'STOP_MARKET'
                    for order in orders
                )

                if has_exit:
                    entry_orders = [o for o in orders if float(o.get('profit', 0)) == 0 and o.get('type') != 'STOP_MARKET']
                    exit_orders = [o for o in orders if float(o.get('profit', 0)) != 0 or o.get('type') == 'STOP_MARKET']

                    if entry_orders and exit_orders:
                        first_entry = entry_orders[0]
                        last_exit = exit_orders[-1]

                        entry_time = datetime.fromtimestamp(int(first_entry.get('time', 0)) / 1000)
                        exit_time = datetime.fromtimestamp(int(last_exit.get('time', 0)) / 1000)

                        total_profit = sum(float(o.get('profit', 0)) for o in exit_orders)
                        total_fees = sum(abs(float(o.get('commission', 0))) for o in orders)
                        net_pnl = total_profit - total_fees

                        exchange_closed_positions.append({
                            'position_id': position_id,
                            'entry_order_id': first_entry.get('orderId'),
                            'entry_time': entry_time,
                            'exit_time': exit_time,
                            'net_pnl': net_pnl,
                            'side': first_entry.get('side'),
                            'num_orders': len(orders)
                        })

            print(f'\n📊 EXCHANGE - Closed Positions (after session_start):')
            print('='*100)

            for i, pos in enumerate(sorted(exchange_closed_positions, key=lambda x: x['entry_time']), 1):
                print(f"{i:2d}. Position {pos['position_id']}")
                print(f"    Entry Order: {pos['entry_order_id']}")
                print(f"    Side: {pos['side']} | Orders: {pos['num_orders']}")
                print(f"    Entry: {pos['entry_time'].strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"    Exit:  {pos['exit_time'].strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"    Net P&L: ${pos['net_pnl']:,.2f}")
                print()

            print(f'Total Exchange Closed Positions: {len(exchange_closed_positions)} ✅')

        else:
            print(f"❌ API Error: {result.get('msg')}")
            return

    except Exception as e:
        print(f"❌ Failed to fetch orders: {e}")
        import traceback
        traceback.print_exc()
        return

    # Now check state file - GROUP BY POSITION
    print(f'\n📄 STATE FILE - Closed Positions (Grouped):')
    print('='*100)

    state_trades = state.get('trades', [])
    closed_trades = [t for t in state_trades if t.get('status') == 'CLOSED']

    # Group by position_id_exchange or order_id
    state_positions = defaultdict(list)
    for trade in closed_trades:
        position_id = trade.get('position_id_exchange', trade.get('order_id'))
        state_positions[position_id].append(trade)

    print(f'Total closed trades in state: {len(closed_trades)}')
    print(f'Grouped into positions: {len(state_positions)}\n')

    state_closed_positions = []
    for position_id, trades in state_positions.items():
        # Calculate aggregated data
        total_pnl = sum(t.get('pnl_usd_net', 0) for t in trades)
        entry_times = [t.get('entry_time') for t in trades if t.get('entry_time')]

        if entry_times:
            try:
                entry_dt = datetime.fromisoformat(entry_times[0])
                entry_time_str = entry_dt.strftime('%Y-%m-%d %H:%M:%S')
            except:
                entry_time_str = str(entry_times[0])
        else:
            entry_time_str = 'N/A'

        state_closed_positions.append({
            'position_id': position_id,
            'num_trades': len(trades),
            'entry_time': entry_time_str,
            'total_pnl': total_pnl,
            'reconciled': any(t.get('exchange_reconciled', False) for t in trades)
        })

    for i, pos in enumerate(state_closed_positions, 1):
        reconciled_flag = " [RECONCILED]" if pos['reconciled'] else ""
        print(f"{i:2d}. Position {pos['position_id']}")
        print(f"    Entry: {pos['entry_time']}")
        print(f"    Trades: {pos['num_trades']} | Net P&L: ${pos['total_pnl']:,.2f}{reconciled_flag}")
        print()

    # COMPARISON
    print(f'='*100)
    print('COMPARISON RESULTS (BY CLOSED POSITION):')
    print('='*100)
    print(f'Exchange Closed Positions (after session_start): {len(exchange_closed_positions)} ✅ GROUND TRUTH')
    print(f'State File Closed Positions: {len(state_closed_positions)}')

    if len(state_closed_positions) > len(exchange_closed_positions):
        print(f'\n⚠️  WARNING: State has {len(state_closed_positions) - len(exchange_closed_positions)} MORE positions!')
    elif len(state_closed_positions) < len(exchange_closed_positions):
        print(f'\n⚠️  WARNING: State has {len(exchange_closed_positions) - len(state_closed_positions)} FEWER positions!')
    else:
        print(f'\n✅ MATCH: Same number of closed positions!')

    # Match by entry_order_id
    exchange_entry_ids = set(pos['entry_order_id'] for pos in exchange_closed_positions)
    state_position_ids = set(str(pos['position_id']) for pos in state_closed_positions)

    # For state positions, get the entry order IDs
    state_entry_ids = set()
    for position_id, trades in state_positions.items():
        # Try to find the entry order (first order or non-reconciled)
        for trade in trades:
            if not trade.get('exchange_reconciled', False):
                state_entry_ids.add(str(trade.get('order_id')))
                break
        else:
            # All reconciled, use the position_id_exchange
            if trades[0].get('position_id_exchange'):
                # Find the entry order id from the position
                pass
            state_entry_ids.add(str(position_id))

    print(f'\n📋 DETAILED MATCHING:')
    for i, ex_pos in enumerate(exchange_closed_positions, 1):
        ex_pos_id = ex_pos['position_id']
        matched = False

        # Check if this position_id is in state
        for st_pos in state_closed_positions:
            if str(st_pos['position_id']) == str(ex_pos_id):
                matched = True
                pnl_diff = abs(st_pos['total_pnl'] - ex_pos['net_pnl'])
                if pnl_diff < 10:  # Within $10
                    print(f'   ✅ Exchange Position {i} (ID {ex_pos_id}): MATCHED in state')
                else:
                    print(f'   ⚠️  Exchange Position {i} (ID {ex_pos_id}): MATCHED but P&L differs by ${pnl_diff:.2f}')
                    print(f'       Exchange: ${ex_pos["net_pnl"]:.2f} | State: ${st_pos["total_pnl"]:.2f}')
                break

        if not matched:
            print(f'   ❌ Exchange Position {i} (ID {ex_pos_id}): NOT FOUND in state!')

    # Check state positions not in exchange
    for st_pos in state_closed_positions:
        found = False
        for ex_pos in exchange_closed_positions:
            if str(st_pos['position_id']) == str(ex_pos['position_id']):
                found = True
                break

        if not found:
            print(f'   ⚠️  State Position {st_pos["position_id"]}: NOT FOUND in exchange (might be OPEN or duplicate)')

if __name__ == '__main__':
    main()
