"""
Exchange Reconciliation V2 - Using Position History API
=======================================================

Refactored to use fetchPositionHistory() instead of manual order grouping.
Much simpler, faster, and more accurate.

Improvements:
- 90% less code (266 lines → ~100 lines)
- Direct position data from exchange (no manual grouping)
- Pre-calculated P&L from exchange (no manual calculation)
- Single API call (instead of order fetching + grouping)
"""

import json
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


def fetch_closed_positions_from_exchange(client, since_timestamp=None, days=7):
    """
    Fetch closed positions directly from exchange using Position History API.

    Args:
        client: BingX API client
        since_timestamp: Unix timestamp (seconds) to fetch positions after
        days: Number of days to look back (if since_timestamp not provided)

    Returns:
        List of closed position dictionaries with P&L data
    """
    try:
        # Calculate time range
        if since_timestamp is None:
            since = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
        else:
            since = int(since_timestamp * 1000)  # Convert to milliseconds

        until = int(datetime.now().timestamp() * 1000)

        logger.info(f"📊 Fetching closed positions from exchange...")
        logger.info(f"   Time range: {datetime.fromtimestamp(since/1000).strftime('%Y-%m-%d %H:%M:%S')} to {datetime.fromtimestamp(until/1000).strftime('%Y-%m-%d %H:%M:%S')}")

        # Fetch position history (already grouped and with P&L calculated!)
        positions = client.exchange.fetchPositionHistory(
            symbol='BTC/USDT:USDT',
            since=since,
            limit=1000,  # Max positions to fetch
            params={'until': until}
        )

        logger.info(f"✅ Fetched {len(positions)} closed positions from exchange")

        return positions

    except Exception as e:
        logger.error(f"❌ Failed to fetch position history: {e}")
        import traceback
        traceback.print_exc()
        return []


def convert_position_to_trade_format(position):
    """
    Convert exchange position format to our trade format.

    Exchange provides:
    - positionId, entryPrice, avgClosePrice (exitPrice)
    - realizedPnl, netProfit (includes fees)
    - positionCommission (total fees)
    - openTime, lastUpdateTimestamp (close time)
    - side (long/short), contracts (quantity)
    """
    info = position.get('info', {})

    # Extract data
    position_id = position.get('id')  # position['info']['positionId']
    entry_price = position.get('entryPrice', 0)
    exit_price = info.get('avgClosePrice', 0)  # Not in standard fields
    quantity = position.get('contracts', 0)

    # P&L data (GROUND TRUTH from exchange!)
    realized_pnl = position.get('realizedPnl', 0)  # Profit excluding fees
    net_pnl = float(info.get('netProfit', 0))  # Profit including fees
    total_fees = abs(float(info.get('positionCommission', 0)))

    # Timestamps
    entry_time = position.get('timestamp', 0) / 1000  # Convert to seconds
    exit_time = position.get('lastUpdateTimestamp', 0) / 1000

    # Side (long/short → BUY/SELL)
    side = position.get('side', '').upper()  # 'long' → 'LONG'
    if side == 'LONG':
        side = 'BUY'
    elif side == 'SHORT':
        side = 'SELL'

    return {
        'position_id': position_id,
        'entry_price': float(entry_price),
        'exit_price': float(exit_price),
        'quantity': float(quantity),
        'side': side,
        'entry_time': entry_time,
        'exit_time': exit_time,
        'realized_pnl': float(realized_pnl),
        'net_pnl': net_pnl,
        'total_fees': total_fees,
        'raw_position': position  # Keep raw data for debugging
    }


def reconcile_state_from_exchange_v2(state, api_client, bot_start_time=None, days=7):
    """
    Reconcile state from exchange using Position History API.

    Much simpler than V1:
    1. Fetch closed positions directly (no order grouping needed)
    2. Match with state trades
    3. Update or add trades

    Args:
        state: Bot state dictionary
        api_client: BingX API client
        bot_start_time: Timestamp to filter positions (None = use session_start)
        days: Days to look back

    Returns:
        (updated_count, new_count)
    """
    logger.info("="*80)
    logger.info("🔄 Reconciling State from Exchange (V2 - Position History API)")
    logger.info("="*80)

    # ✅ NEW (2025-10-25): Check for recent manual trading history reset
    # If reset happened recently (within 1 hour), skip trade reconciliation
    # This respects manual resets and prevents auto-re-population of cleared trades
    reconciliation_log = state.get('reconciliation_log', [])
    if reconciliation_log:
        # Check last 5 log entries for recent trading_history_reset
        recent_logs = reconciliation_log[-5:]
        for log_entry in reversed(recent_logs):
            if log_entry.get('event') == 'trading_history_reset':
                reset_time_str = log_entry.get('timestamp', '')
                try:
                    reset_time = datetime.fromisoformat(reset_time_str.replace('Z', '+00:00'))
                    # Make now timezone-aware if reset_time is
                    now = datetime.now(reset_time.tzinfo) if reset_time.tzinfo else datetime.now()
                    time_since_reset = (now - reset_time).total_seconds()

                    # If reset was within 1 hour (3600 seconds), skip trade reconciliation
                    if time_since_reset < 3600:
                        logger.info(f"📌 Recent trading history reset detected ({time_since_reset/60:.1f} min ago)")
                        logger.info(f"   Skipping trade reconciliation to preserve manual reset")
                        logger.info(f"   Trade reconciliation will resume after 1 hour window")
                        logger.info("="*80)
                        return (0, 0)  # Skip reconciliation
                except Exception as e:
                    logger.debug(f"   Error parsing reset timestamp: {e}")
                    continue

    # Get bot start time
    if bot_start_time is None:
        bot_start_time = state.get('start_time')
        if bot_start_time is None:
            session_start = state.get('session_start')
            if session_start:
                bot_start_time = datetime.fromisoformat(session_start).timestamp()
                logger.info(f"ℹ️  Using session_start as bot_start_time")
            else:
                bot_start_time = 0
                logger.warning(f"⚠️  No start_time or session_start found")

    # ✅ FIX 2025-10-27: Handle string bot_start_time outside the if block
    # This handles case when bot_start_time is passed as a string (e.g., session_start)
    if isinstance(bot_start_time, str):
        bot_start_time = datetime.fromisoformat(bot_start_time).timestamp()

    logger.info(f"📅 Bot Start Time: {datetime.fromtimestamp(bot_start_time).strftime('%Y-%m-%d %H:%M:%S')}")

    # Fetch closed positions from exchange
    positions = fetch_closed_positions_from_exchange(
        api_client,
        since_timestamp=bot_start_time,
        days=days
    )

    if not positions:
        logger.warning("⚠️  No positions fetched from exchange")
        return (0, 0)

    # Convert to our trade format
    reconciled_trades = []
    total_net_pnl = 0

    for position in positions:
        trade_data = convert_position_to_trade_format(position)

        # Filter by bot start time (positions closed after bot start)
        if trade_data['exit_time'] >= bot_start_time:
            reconciled_trades.append(trade_data)
            total_net_pnl += trade_data['net_pnl']

    logger.info(f"📈 Closed Positions (after bot start): {len(reconciled_trades)}")
    logger.info(f"📈 Total Net P&L: ${total_net_pnl:.2f}")

    # Update state file
    logger.info(f"💾 Updating state with ground truth...")

    state_trades = state.get('trades', [])

    # Remove old reconciled trades
    old_reconciled = [t for t in state_trades if t.get('exchange_reconciled', False)]
    if old_reconciled:
        logger.info(f"🗑️  Removing {len(old_reconciled)} old reconciled trades...")
        state_trades = [t for t in state_trades if not t.get('exchange_reconciled', False)]
        state['trades'] = state_trades

    updated_count = 0
    new_count = 0

    for trade_data in reconciled_trades:
        position_id = trade_data['position_id']
        entry_time = datetime.fromtimestamp(trade_data['entry_time'])
        entry_price = trade_data['entry_price']

        # Match by entry time + entry price (more reliable than position ID)
        # Position IDs are different between fetchPositionHistory and swapV2PrivateGetTradeAllOrders
        matching_trade = None
        for t in state_trades:
            if t.get('entry_time') and t.get('entry_price'):
                try:
                    t_entry_time = datetime.fromisoformat(t['entry_time'])
                    t_entry_price = float(t['entry_price'])

                    # Match if entry time within 5 seconds AND entry price within 0.1%
                    time_diff = abs((entry_time - t_entry_time).total_seconds())
                    price_diff = abs((entry_price - t_entry_price) / t_entry_price)

                    if time_diff < 5 and price_diff < 0.001:  # 5 seconds, 0.1%
                        matching_trade = t
                        break
                except:
                    continue

        # If not found, this is a new trade (manual or external)
        if matching_trade:
            # Update existing trade with ground truth
            matching_trade['pnl_usd'] = trade_data['realized_pnl']
            matching_trade['total_fee'] = trade_data['total_fees']
            matching_trade['pnl_usd_net'] = trade_data['net_pnl']
            matching_trade['exchange_reconciled'] = True
            matching_trade['position_history_id'] = position_id  # History record ID
            # Keep original position_id_exchange if it exists (from orders API)
            if not matching_trade.get('position_id_exchange'):
                matching_trade['position_id_exchange'] = position_id
            updated_count += 1
            logger.info(f"   ✅ Updated trade (order {matching_trade.get('order_id')} → history {position_id})")
        else:
            # New trade from exchange (manual trade)
            new_trade = {
                'order_id': position_id,  # Use position ID as order ID
                'position_id_exchange': position_id,
                'side': trade_data['side'],
                'entry_price': trade_data['entry_price'],
                'exit_price': trade_data['exit_price'],
                'quantity': trade_data['quantity'],
                'entry_time': datetime.fromtimestamp(trade_data['entry_time']).isoformat(),
                'close_time': datetime.fromtimestamp(trade_data['exit_time']).isoformat(),
                'pnl_usd': trade_data['realized_pnl'],
                'total_fee': trade_data['total_fees'],
                'pnl_usd_net': trade_data['net_pnl'],
                'status': 'CLOSED',
                'exit_reason': 'Exchange Reconciled',
                'exchange_reconciled': True,
                'manual_trade': True
            }
            state_trades.append(new_trade)
            new_count += 1
            logger.info(f"   ➕ Added new trade (position {position_id})")

    logger.info(f"✅ Reconciliation complete: {updated_count} updated, {new_count} added")
    logger.info("="*80)

    return (updated_count, new_count)
