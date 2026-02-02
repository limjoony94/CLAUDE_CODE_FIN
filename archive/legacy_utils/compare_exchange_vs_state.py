#!/usr/bin/env python3
"""
Compare Exchange API Data vs State File
Shows exactly what the exchange has vs what the state file shows
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
    print('EXCHANGE vs STATE FILE COMPARISON')
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

            print(f'   Total orders: {len(all_orders)}')
            print(f'   Filled orders: {len(filled_orders)}')

            # Filter by session_start
            recent_orders = [o for o in filled_orders if int(o.get('time', 0)) / 1000 >= session_start_ts]
            old_orders = [o for o in filled_orders if int(o.get('time', 0)) / 1000 < session_start_ts]

            print(f'   Orders AFTER session_start: {len(recent_orders)} ✅')
            print(f'   Orders BEFORE session_start: {len(old_orders)} ❌')

            # Group recent orders by position
            positions = defaultdict(list)
            for order in recent_orders:
                position_id = order.get('positionID')
                if position_id:
                    positions[position_id].append(order)

            print(f'\n📊 EXCHANGE - Recent Closed Positions (after {session_start_dt.strftime("%Y-%m-%d %H:%M:%S")}):')
            print('='*100)

            closed_positions = []
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

                        closed_positions.append({
                            'position_id': position_id,
                            'entry_order_id': first_entry.get('orderId'),
                            'entry_time': entry_time,
                            'exit_time': exit_time,
                            'net_pnl': net_pnl,
                            'profit': total_profit,
                            'fees': total_fees,
                            'side': first_entry.get('side')
                        })

            for i, pos in enumerate(sorted(closed_positions, key=lambda x: x['entry_time']), 1):
                print(f"{i:2d}. Position {pos['position_id']}")
                print(f"    Entry Order: {pos['entry_order_id']}")
                print(f"    Side: {pos['side']}")
                print(f"    Entry: {pos['entry_time'].strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"    Exit:  {pos['exit_time'].strftime('%Y-%m-%d %H:%M:%S')}")
                print(f"    Profit: ${pos['profit']:,.2f} | Fees: ${pos['fees']:,.2f} | Net: ${pos['net_pnl']:,.2f}")
                print()

            print(f'Total Exchange Positions (after session_start): {len(closed_positions)}')

        else:
            print(f"❌ API Error: {result.get('msg')}")
            return

    except Exception as e:
        print(f"❌ Failed to fetch orders: {e}")
        import traceback
        traceback.print_exc()
        return

    # Now check state file
    print(f'\n📄 STATE FILE - Closed Trades:')
    print('='*100)

    state_trades = state.get('trades', [])
    closed_trades = [t for t in state_trades if t.get('status') == 'CLOSED']

    print(f'Total closed trades in state: {len(closed_trades)}\n')

    for i, trade in enumerate(closed_trades, 1):
        entry_time = trade.get('entry_time', 'N/A')
        order_id = trade.get('order_id', 'N/A')
        pnl_net = trade.get('pnl_usd_net', 0)
        reconciled = trade.get('exchange_reconciled', False)
        manual = trade.get('manual_trade', False)

        flags = []
        if reconciled:
            flags.append('RECONCILED')
        if manual:
            flags.append('MANUAL')
        flag_str = f" [{', '.join(flags)}]" if flags else ""

        try:
            if isinstance(entry_time, str) and 'T' in entry_time:
                dt = datetime.fromisoformat(entry_time)
                entry_time_str = dt.strftime('%Y-%m-%d %H:%M:%S')
            else:
                entry_time_str = str(entry_time)
        except:
            entry_time_str = str(entry_time)

        print(f'{i:2d}. Order {order_id}: {entry_time_str} | Net P&L: ${pnl_net:,.2f}{flag_str}')

    # COMPARISON
    print(f'\n' + '='*100)
    print('COMPARISON RESULTS:')
    print('='*100)
    print(f'Exchange Positions (after session_start): {len(closed_positions)} ✅ GROUND TRUTH')
    print(f'State File Trades: {len(closed_trades)}')

    if len(closed_trades) > len(closed_positions):
        print(f'\n⚠️  WARNING: State file has {len(closed_trades) - len(closed_positions)} MORE trades than exchange!')
        print(f'⚠️  This suggests old historical data is still in the state file.')
    elif len(closed_trades) < len(closed_positions):
        print(f'\n⚠️  WARNING: State file has {len(closed_positions) - len(closed_trades)} FEWER trades than exchange!')
        print(f'⚠️  Some trades from exchange are not in state file.')
    else:
        print(f'\n✅ MATCH: State file and exchange have same number of positions!')

    # Match by order_id
    print(f'\n📋 MATCHING BY ORDER ID:')
    exchange_order_ids = set(pos['entry_order_id'] for pos in closed_positions)
    state_order_ids = set(str(t.get('order_id')) for t in closed_trades)

    in_both = exchange_order_ids & state_order_ids
    only_exchange = exchange_order_ids - state_order_ids
    only_state = state_order_ids - exchange_order_ids

    print(f'   In BOTH: {len(in_both)}')
    print(f'   Only in EXCHANGE: {len(only_exchange)}')
    print(f'   Only in STATE: {len(only_state)}')

    if only_state:
        print(f'\n⚠️  TRADES IN STATE BUT NOT IN EXCHANGE (after session_start):')
        for order_id in sorted(only_state):
            trade = next(t for t in closed_trades if str(t.get('order_id')) == order_id)
            entry_time = trade.get('entry_time', 'N/A')
            try:
                if isinstance(entry_time, str) and 'T' in entry_time:
                    dt = datetime.fromisoformat(entry_time)
                    entry_time_str = dt.strftime('%Y-%m-%d %H:%M:%S')

                    # Check if before session_start
                    if dt.timestamp() < session_start_ts:
                        print(f'   ❌ Order {order_id}: {entry_time_str} (BEFORE session_start - should be removed!)')
                    else:
                        print(f'   ⚠️  Order {order_id}: {entry_time_str} (after session_start but not in exchange)')
                else:
                    print(f'   ⚠️  Order {order_id}: {entry_time}')
            except:
                print(f'   ⚠️  Order {order_id}: {entry_time}')

if __name__ == '__main__':
    main()
