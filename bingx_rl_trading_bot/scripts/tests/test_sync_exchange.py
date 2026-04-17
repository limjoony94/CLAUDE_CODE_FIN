"""_sync_exchange tests — ghost detection + orphan adoption integration.

Covers:
  BUG#48 orphan adoption with SL restoration (end-to-end)
  BUG#50 ghost exit reason via trade.info.orderType
  BUG#36 ghost resolution filters trades before entry_time
  BUG#45 ghost exit_time uses exchange timestamp
  API error does not corrupt local state
"""
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock
import pytest


class TestSyncAPIError:
    """D. Rollback: API error → skip sync, do NOT modify local state."""

    def test_api_error_preserves_local_positions(self, mock_bot):
        mock_bot.positions = [{
            'direction': 'LONG', 'entry_price': 70000, 'sl_price': 69500,
            'best_price': 70000, 'entry_time': '2026-04-17T12:00:00',
            'bars_held': 5, 'size_pct': 100.0,
        }]
        mock_bot.exchange.fetch_positions.side_effect = Exception('network')
        mock_bot._sync_exchange()
        # Position should still be there (not wrongly ghosted)
        assert len(mock_bot.positions) == 1


class TestOrphanAdoption:
    """BUG#48: exchange has position, local doesn't → adopt + restore SL."""

    def test_orphan_adopted_with_exchange_sl(self, mock_bot):
        """Happy path: SL exists on exchange → restore actual sl_price."""
        mock_bot.positions = []
        mock_bot.exchange.fetch_positions.return_value = [{
            'side': 'LONG', 'contracts': 0.01, 'entryPrice': 70000.0,
        }]
        # Exchange has SL at 69500 (tighter than default 3% emergency = 67900)
        mock_bot.exchange.fetch_open_orders.return_value = [{
            'id': 'existing_sl', 'side': 'sell', 'type': 'STOP_MARKET',
            'reduceOnly': True,
            'info': {'type': 'STOP_MARKET', 'stopPrice': '69500'},
        }]
        mock_bot._sync_exchange()
        assert len(mock_bot.positions) == 1
        pos = mock_bot.positions[0]
        assert pos['direction'] == 'LONG'
        assert pos['entry_price'] == 70000.0
        assert pos['sl_price'] == 69500.0  # restored, NOT 3% fallback
        assert pos['sl_order_id'] == 'existing_sl'

    def test_orphan_adopted_with_3pct_fallback(self, mock_bot):
        """D. Rollback: no SL on exchange (true orphan) → 3% emergency."""
        mock_bot.positions = []
        mock_bot.exchange.fetch_positions.return_value = [{
            'side': 'LONG', 'contracts': 0.01, 'entryPrice': 70000.0,
        }]
        mock_bot.exchange.fetch_open_orders.return_value = []  # no SL
        mock_bot._sync_exchange()
        pos = mock_bot.positions[0]
        # emergency_sl_pct=3.0 from default config → 70000 × 0.97 = 67900
        assert abs(pos['sl_price'] - 67900.0) < 0.01
        assert pos['sl_order_id'] == ''

    def test_orphan_short_direction(self, mock_bot):
        """C. Bug interaction: SHORT orphan → SL above entry."""
        mock_bot.positions = []
        mock_bot.exchange.fetch_positions.return_value = [{
            'side': 'SHORT', 'contracts': 0.01, 'entryPrice': 70000.0,
        }]
        mock_bot.exchange.fetch_open_orders.return_value = [{
            'id': 'short_sl', 'side': 'buy', 'type': 'STOP_MARKET',
            'reduceOnly': True,
            'info': {'type': 'STOP_MARKET', 'stopPrice': '70500'},
        }]
        mock_bot._sync_exchange()
        pos = mock_bot.positions[0]
        assert pos['direction'] == 'SHORT'
        assert pos['sl_price'] == 70500.0  # above entry = correct SHORT SL


class TestGhostDetection:
    """Local has position, exchange doesn't → ghost exit."""

    def test_ghost_with_stop_order_type_classified_as_exchange_sl(self, mock_bot):
        """BUG#50: trade.info.orderType = STOP_MARKET → EXCHANGE_SL."""
        entry_time = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        mock_bot.positions = [{
            'direction': 'LONG', 'entry_price': 70000, 'sl_price': 69500,
            'best_price': 70200, 'entry_time': entry_time,
            'bars_held': 4, 'size_pct': 100.0,
            'last_callback': 0.6,
        }]
        mock_bot.exchange.fetch_positions.return_value = []  # gone
        # fetch_my_trades returns a closing trade with explicit STOP_MARKET type
        mock_bot.exchange.fetch_my_trades.return_value = [{
            'id': 't1', 'side': 'sell', 'amount': 0.01, 'price': 69500.0,
            'timestamp': int(datetime.now(timezone.utc).timestamp() * 1000),
            'info': {'orderType': 'STOP_MARKET'},
        }]
        mock_bot._sync_exchange()
        assert len(mock_bot.positions) == 0
        assert len(mock_bot.trade_history) == 1
        assert mock_bot.trade_history[-1]['reason'] == 'EXCHANGE_SL'

    def test_ghost_with_trailing_type_classified_as_trail(self, mock_bot):
        """BUG#50: TRAILING_STOP_MARKET → EXCHANGE_TRAIL."""
        entry_time = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        mock_bot.positions = [{
            'direction': 'LONG', 'entry_price': 70000, 'sl_price': 69500,
            'best_price': 71000, 'entry_time': entry_time,
            'bars_held': 4, 'size_pct': 100.0,
            'last_callback': 0.6,
        }]
        mock_bot.exchange.fetch_positions.return_value = []
        mock_bot.exchange.fetch_my_trades.return_value = [{
            'id': 't1', 'side': 'sell', 'amount': 0.01, 'price': 70500.0,
            'timestamp': int(datetime.now(timezone.utc).timestamp() * 1000),
            'info': {'orderType': 'TRAILING_STOP_MARKET'},
        }]
        mock_bot._sync_exchange()
        assert mock_bot.trade_history[-1]['reason'] == 'EXCHANGE_TRAIL'

    def test_ghost_filters_trades_before_entry(self, mock_bot):
        """BUG#36: closing trade must be AFTER entry_time.

        Prevents using a prior position's close as current position's exit.
        """
        now = datetime.now(timezone.utc)
        entry_time = now.isoformat()
        mock_bot.positions = [{
            'direction': 'LONG', 'entry_price': 70000, 'sl_price': 69500,
            'best_price': 70000, 'entry_time': entry_time,
            'bars_held': 1, 'size_pct': 100.0,
        }]
        mock_bot.exchange.fetch_positions.return_value = []
        # Stale trade from 2 hours ago — BEFORE entry
        stale_ts = int((now - timedelta(hours=2)).timestamp() * 1000)
        mock_bot.exchange.fetch_my_trades.return_value = [{
            'id': 'stale', 'side': 'sell', 'amount': 0.01, 'price': 68000.0,
            'timestamp': stale_ts,
            'info': {'orderType': 'STOP_MARKET'},
        }]
        mock_bot._sync_exchange()
        # Ghost fallback (no valid trade) → use sl_price as exit
        assert len(mock_bot.trade_history) == 1
        assert mock_bot.trade_history[-1]['exit_price'] == 69500.0  # sl_price fallback

    def test_ghost_uses_exchange_timestamp(self, mock_bot):
        """BUG#45: exit_time from trade timestamp, not detection time."""
        entry_time = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
        actual_exit_ts = int((datetime.now(timezone.utc) - timedelta(minutes=30))
                             .timestamp() * 1000)
        mock_bot.positions = [{
            'direction': 'LONG', 'entry_price': 70000, 'sl_price': 69500,
            'best_price': 70000, 'entry_time': entry_time,
            'bars_held': 8, 'size_pct': 100.0,
        }]
        mock_bot.exchange.fetch_positions.return_value = []
        mock_bot.exchange.fetch_my_trades.return_value = [{
            'id': 't1', 'side': 'sell', 'amount': 0.01, 'price': 69500.0,
            'timestamp': actual_exit_ts,
            'info': {'orderType': 'STOP_MARKET'},
        }]
        mock_bot._sync_exchange()
        # exit_time should reflect the actual trade timestamp (30 min ago),
        # not "now" (detection time)
        recorded = mock_bot.trade_history[-1]['exit_time']
        actual_dt = datetime.fromtimestamp(actual_exit_ts / 1000, tz=timezone.utc)
        expected_prefix = actual_dt.replace(tzinfo=None).isoformat()[:19]
        assert recorded.startswith(expected_prefix)
