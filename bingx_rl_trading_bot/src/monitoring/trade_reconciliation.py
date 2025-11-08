"""
Trade Reconciliation System
===========================

Reconciles bot state with actual exchange order history via API.
Detects manual trades and ensures P&L accuracy.

Author: Claude Code
Date: 2025-10-19
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class TradeReconciliation:
    """
    Reconciles bot-managed trades with actual exchange order history.

    Features:
    - Detects manual trades (orders not in bot state)
    - Verifies bot-managed trade execution
    - Calculates accurate P&L from exchange data
    - Maintains reconciliation audit log
    """

    def __init__(self, exchange_client, state_file_path: Path):
        """
        Initialize trade reconciliation system.

        Args:
            exchange_client: BingXClient instance with API access
            state_file_path: Path to bot state JSON file
        """
        self.exchange = exchange_client
        self.state_file = state_file_path

    def load_state(self) -> Dict:
        """Load current bot state from file."""
        if not self.state_file.exists():
            logger.error(f"State file not found: {self.state_file}")
            return {}

        with open(self.state_file, 'r') as f:
            return json.load(f)

    def save_state(self, state: Dict):
        """Save updated state to file."""
        with open(self.state_file, 'w') as f:
            json.dump(state, f, indent=2)
        logger.info(f"✅ State saved to {self.state_file}")

    def get_position_history(self, symbol: str = 'BTC-USDT',
                            since: Optional[datetime] = None,
                            limit: int = 100) -> List[Dict]:
        """
        Fetch position history from exchange API.

        Args:
            symbol: Trading pair symbol (BTC-USDT format)
            since: Only fetch positions after this time (default: 7 days ago)
            limit: Max number of positions to fetch

        Returns:
            List of position dictionaries from exchange
        """
        if since is None:
            since = datetime.now() - timedelta(days=7)

        start_ts = int(since.timestamp() * 1000)
        end_ts = int(datetime.now().timestamp() * 1000)

        try:
            response = self.exchange.exchange.swap_v1_private_get_trade_positionhistory({
                'symbol': symbol,
                'startTs': start_ts,
                'endTs': end_ts,
                'limit': limit
            })

            if response.get('code') == '0' and 'data' in response:
                positions = response['data'].get('positionHistory', [])
                logger.info(f"📥 Fetched {len(positions)} positions from exchange since {since}")
                return positions
            else:
                logger.error(f"❌ Unexpected response format: {response}")
                return []

        except Exception as e:
            logger.error(f"❌ Error fetching position history from exchange: {e}")
            return []

    def get_bot_position_ids(self, state: Dict) -> set:
        """Extract all position IDs from bot state."""
        position_ids = set()

        # Current position
        position = state.get('position')
        if position and position.get('position_id_exchange'):
            position_ids.add(position['position_id_exchange'])

        # Historical trades
        for trade in state.get('trades', []):
            if trade.get('position_id_exchange'):
                position_ids.add(trade['position_id_exchange'])
            # Also check order_id for backwards compatibility
            elif trade.get('order_id'):
                position_ids.add(trade['order_id'])

        return position_ids

    def detect_manual_positions(self, state: Dict,
                                positions: List[Dict]) -> List[Dict]:
        """
        Detect manual positions (positions not created by bot).

        Args:
            state: Current bot state
            positions: Position history from exchange API

        Returns:
            List of detected manual position dictionaries
        """
        bot_position_ids = self.get_bot_position_ids(state)
        manual_positions = []

        for position in positions:
            position_id = position.get('positionId')

            # Check if this position is not in bot state
            if position_id and position_id not in bot_position_ids:
                manual_positions.append(position)

        if manual_positions:
            logger.warning(f"⚠️  Detected {len(manual_positions)} manual/untracked positions")

        return manual_positions

    def create_trade_record_from_position(self, position: Dict) -> Dict:
        """
        Create a trade record from position history data.

        Args:
            position: Position data from exchange API

        Returns:
            Trade record dictionary
        """
        # Extract position data
        position_id = position.get('positionId')
        symbol = position.get('symbol', 'BTC-USDT')
        side = position.get('positionSide', 'LONG')

        # Times
        open_time = int(position.get('openTime', 0))
        update_time = int(position.get('updateTime', 0))
        entry_time = datetime.fromtimestamp(open_time / 1000).isoformat()
        exit_time = datetime.fromtimestamp(update_time / 1000).isoformat()

        # Prices and amounts
        entry_price = float(position.get('avgPrice', 0))
        exit_price = float(position.get('avgClosePrice', 0))
        quantity = float(position.get('positionAmt', 0))
        close_qty = float(position.get('closePositionAmt', 0))

        # P&L (from exchange calculations)
        realized_profit = float(position.get('realisedProfit', 0))  # Gross P&L
        net_profit = float(position.get('netProfit', 0))  # Net P&L (after fees & funding)
        commission = abs(float(position.get('positionCommission', 0)))
        funding = float(position.get('totalFunding', 0))

        # Leverage
        leverage = int(position.get('leverage', 4))

        # Calculate price change percentage
        price_change = (exit_price - entry_price) / entry_price if entry_price > 0 else 0

        # Check if position is closed
        is_closed = abs(quantity - close_qty) < 0.0001

        # Build trade record
        trade_record = {
            'position_id_exchange': position_id,
            'order_id': position_id,  # Use position ID as order ID for compatibility
            'side': side,
            'manual_trade': True,
            'status': 'CLOSED' if is_closed else 'OPEN',
            'entry_time': entry_time,
            'entry_price': entry_price,
            'quantity': quantity,
            'leverage': leverage,
            'note': f"Position reconciled from exchange API - {datetime.now().strftime('%Y-%m-%d')}"
        }

        # Add exit info if closed
        if is_closed:
            trade_record.update({
                'close_time': exit_time,
                'exit_price': exit_price,
                'exit_time': exit_time,
                'price_change_pct': price_change,
                'pnl_usd': realized_profit,  # Gross P&L
                'pnl_usd_net': net_profit,   # Net P&L (includes fees & funding)
                'total_fee': commission,
                'funding_fee': funding,
                'exit_reason': 'Exchange Reconciled',
                'exchange_reconciled': True
            })

        return trade_record

    def reconcile(self, lookback_days: int = 7) -> Dict:
        """
        Perform full reconciliation of bot state with exchange.

        Args:
            lookback_days: How many days of history to check

        Returns:
            Reconciliation report dictionary
        """
        logger.info("="*70)
        logger.info("TRADE RECONCILIATION STARTING (Position History)")
        logger.info("="*70)

        # Load current state
        state = self.load_state()
        if not state:
            return {'error': 'Failed to load state'}

        # Get position history from exchange
        since = datetime.now() - timedelta(days=lookback_days)
        positions = self.get_position_history(symbol='BTC-USDT', since=since, limit=100)

        if not positions:
            logger.warning("⚠️  No positions found in exchange history")
            return {'status': 'no_positions'}

        # Detect manual positions
        manual_positions = self.detect_manual_positions(state, positions)

        if not manual_positions:
            logger.info("✅ No manual/untracked positions detected")
            return {'status': 'clean', 'manual_trades': 0}

        logger.info(f"\n📊 Found {len(manual_positions)} manual position(s)")

        # Process each manual position
        reconciliation_log = []
        added_trades = []

        for position in manual_positions:
            pos_id = position.get('positionId')
            logger.info(f"\n📍 Processing Position: {pos_id}")

            # Extract position data
            entry_price = float(position.get('avgPrice', 0))
            exit_price = float(position.get('avgClosePrice', 0))
            quantity = float(position.get('positionAmt', 0))
            net_profit = float(position.get('netProfit', 0))
            side = position.get('positionSide', 'LONG')

            logger.info(f"   Side: {side}")
            logger.info(f"   Entry: {quantity:.4f} @ ${entry_price:.2f}")
            logger.info(f"   Exit: {quantity:.4f} @ ${exit_price:.2f}")
            logger.info(f"   Net P&L: ${net_profit:.2f}")

            # Create trade record
            trade_record = self.create_trade_record_from_position(position)
            added_trades.append(trade_record)

            # Log reconciliation
            reconciliation_log.append({
                'timestamp': datetime.now().isoformat(),
                'event': 'manual_position_detected',
                'position_id': pos_id,
                'side': side,
                'pnl': net_profit,
                'status': 'closed'
            })

        # Update state with manual trades
        if added_trades:
            # Add to trades list
            for trade in added_trades:
                state['trades'].append(trade)

            # Update stats
            manual_pnl = sum(t.get('pnl_usd_net', 0) for t in added_trades if t['status'] == 'CLOSED')

            state['stats']['manual_trades'] = state['stats'].get('manual_trades', 0) + len([t for t in added_trades if t['status'] == 'CLOSED'])
            state['stats']['total_pnl_usd'] = state['stats'].get('total_pnl_usd', 0) + manual_pnl
            state['stats']['manual_trade_pnl'] = state['stats'].get('manual_trade_pnl', 0) + manual_pnl

            # Add to reconciliation log
            state['reconciliation_log'].extend(reconciliation_log)

            # Save updated state
            self.save_state(state)

            logger.info("\n" + "="*70)
            logger.info(f"✅ RECONCILIATION COMPLETE")
            logger.info(f"   Manual trades added: {len(added_trades)}")
            logger.info(f"   Total manual P&L: ${manual_pnl:.2f}")
            logger.info("="*70)

        return {
            'status': 'success',
            'manual_trades_found': len(added_trades),
            'total_manual_pnl': sum(t.get('pnl_usd_net', 0) for t in added_trades if t['status'] == 'CLOSED'),
            'trades_added': added_trades,
            'reconciliation_log': reconciliation_log
        }
