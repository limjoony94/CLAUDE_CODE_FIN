"""
Exchange Reconciliation - Ground Truth from Exchange API
========================================================

Simple approach: Fetch closed position data from exchange and update state file.
No complex calculations - uses exchange data as-is.
"""

import json
from datetime import datetime, timedelta
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


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
            logger.error(f"❌ API Error: {result.get('msg')}")
            return []
    except Exception as e:
        logger.error(f"❌ Failed to fetch orders: {e}")
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
    entry_fees = sum(abs(float(o.get('commission', 0))) for o in entry_orders)
    exit_fees = sum(abs(float(o.get('commission', 0))) for o in exit_orders)

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
        'entry_fees': entry_fees,
        'exit_fees': exit_fees,
        'net_pnl': net_pnl,
        'entry_order_ids': [o.get('orderId') for o in entry_orders],
        'exit_order_ids': [o.get('orderId') for o in exit_orders]
    }


def reconcile_state_from_exchange(state, api_client, bot_start_time=None, days=7):
    """
    Reconcile state from exchange ground truth.
    Returns: (updated_count, new_count)
    """
    logger.info("="*80)
    logger.info("🔄 Reconciling State from Exchange Ground Truth")
    logger.info("="*80)

    # Get bot start time
    if bot_start_time is None:
        # Try start_time first, then session_start, default to 0
        bot_start_time = state.get('start_time')
        if bot_start_time is None:
            session_start = state.get('session_start')
            if session_start:
                bot_start_time = datetime.fromisoformat(session_start).timestamp()
                logger.info(f"ℹ️  Using session_start as bot_start_time")
            else:
                bot_start_time = 0
                logger.warning(f"⚠️  No start_time or session_start found, using 0 (ALL historical data)")
        elif isinstance(bot_start_time, str):
            bot_start_time = datetime.fromisoformat(bot_start_time).timestamp()

    logger.info(f"📅 Bot Start Time: {datetime.fromtimestamp(bot_start_time).strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"📅 Fetching orders from last {days} days...")

    # Fetch all closed orders from exchange
    orders = get_closed_orders_from_exchange(api_client, days=days)
    logger.info(f"✅ Fetched {len(orders)} filled orders from exchange")

    # Group by position
    positions_dict = group_orders_by_position(orders)
    logger.info(f"📊 Found {len(positions_dict)} unique positions")

    # Identify closed positions
    closed_positions = identify_closed_positions(positions_dict)
    logger.info(f"✅ Identified {len(closed_positions)} closed positions")

    # Calculate P&L for each closed position
    reconciled_trades = []
    total_net_pnl = 0

    for position in closed_positions:
        pnl_data = calculate_position_pnl(position)

        # Only include positions closed AFTER bot start
        if pnl_data['exit_time'] / 1000 >= bot_start_time:
            reconciled_trades.append(pnl_data)
            total_net_pnl += pnl_data['net_pnl']

    logger.info(f"📈 Total Closed Positions (after bot start): {len(reconciled_trades)}")
    logger.info(f"📈 Total Net P&L: ${total_net_pnl:.2f}")

    # Update state file with ground truth data
    logger.info(f"💾 Updating state with ground truth data...")

    state_trades = state.get('trades', [])

    # Remove old reconciled trades (from previous reconciliation runs)
    old_reconciled_trades = [t for t in state_trades if t.get('exchange_reconciled', False)]
    if old_reconciled_trades:
        logger.info(f"🗑️  Removing {len(old_reconciled_trades)} old reconciled trades...")
        state_trades = [t for t in state_trades if not t.get('exchange_reconciled', False)]
        state['trades'] = state_trades

    updated_count = 0
    new_count = 0

    for pnl_data in reconciled_trades:
        # Try to find matching trade in state by entry order ID
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
            matching_trade['entry_fee'] = pnl_data['entry_fees']
            matching_trade['exit_fee'] = pnl_data['exit_fees']
            matching_trade['exchange_reconciled'] = True
            matching_trade['position_id_exchange'] = pnl_data['position_id']
            updated_count += 1
            logger.info(f"   ✅ Updated trade {entry_order_id} with ground truth")
        else:
            # This is a new trade not in state (e.g., manual trade)
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
                'entry_fee': pnl_data['entry_fees'],
                'exit_fee': pnl_data['exit_fees'],
                'status': 'CLOSED',
                'exit_reason': 'Reconciled from exchange',
                'exchange_reconciled': True,
                'manual_trade': True  # Mark as manual since not in bot state
            }
            state_trades.append(new_trade)
            new_count += 1
            logger.info(f"   ➕ Added new trade {entry_order_id} from exchange")

    logger.info(f"✅ Reconciliation complete: {updated_count} updated, {new_count} added")
    logger.info("="*80)

    return (updated_count, new_count)
