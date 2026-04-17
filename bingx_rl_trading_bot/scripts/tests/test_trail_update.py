"""_update_exchange_trail tests.

Covers:
  BUG#35 _force_trail_reset on startup (legacy priceRate cleanup)
  BUG#46 asymmetric LOOSEN-only policy (ATR↑ re-place, ATR↓ skip)
  BUG#59 _trail_update_fail_streak counter (3+ consecutive failures)
"""
import pytest
from unittest.mock import MagicMock


def _make_pos(direction='LONG', last_callback=0.5):
    return {
        'direction': direction,
        'entry_price': 70000,
        'sl_price': 69500 if direction == 'LONG' else 70500,
        'best_price': 70200 if direction == 'LONG' else 69800,
        'last_callback': last_callback,
        'sl_order_id': 'existing_sl',
        'bars_held': 3,
        'size_pct': 100.0,
    }


class TestForceResetFlag:
    """BUG#35: on first cycle after restart with position, cancel+re-place trail."""

    def test_force_reset_cancels_existing_trail(self, mock_bot):
        pos = _make_pos()
        mock_bot._force_trail_reset = True
        mock_bot.exchange.fetch_open_orders.return_value = [
            {'id': 'existing_sl', 'type': 'STOP_MARKET',
             'info': {'type': 'STOP_MARKET', 'stopPrice': '69500'}},
            {'id': 'old_trail', 'type': 'TRAILING_STOP_MARKET',
             'info': {'type': 'TRAILING_STOP_MARKET'}},
        ]
        mock_bot.exchange.fetch_positions.return_value = [{
            'side': 'LONG', 'contracts': 0.01, 'entryPrice': 70000,
        }]
        mock_bot.exchange.create_order.return_value = {'id': 'new_trail'}
        mock_bot._update_exchange_trail(pos, 70100, 100.0)
        # Should have cancelled old trail and created new one
        mock_bot.exchange.cancel_order.assert_called()
        mock_bot.exchange.create_order.assert_called()
        # Flag consumed
        assert mock_bot._force_trail_reset is False

    def test_force_reset_cleared_after_run(self, mock_bot):
        """C. Bug interaction: flag is one-shot, not persistent."""
        pos = _make_pos()
        mock_bot._force_trail_reset = True
        mock_bot.exchange.fetch_open_orders.return_value = []
        mock_bot.exchange.fetch_positions.return_value = [{
            'side': 'LONG', 'contracts': 0.01, 'entryPrice': 70000,
        }]
        mock_bot.exchange.create_order.return_value = {'id': 'x'}
        mock_bot._update_exchange_trail(pos, 70100, 100.0)
        assert mock_bot._force_trail_reset is False
        # Call again — flag should stay False, no unnecessary cancels
        mock_bot.exchange.cancel_order.reset_mock()
        mock_bot._update_exchange_trail(pos, 70100, 100.0)


class TestLoosenOnlyPolicy:
    """BUG#46: only re-place trail when new_callback > old + 0.1 (widen).

    Rationale: BingX TRAILING_STOP_MARKET cancel+re-place resets best_price
    tracking. LOOSEN is safe (wider trail far from best), TIGHTEN is handled
    by bot's check_exit to preserve exchange tracking.
    """

    def test_loosen_replaces_trail(self, mock_bot):
        """ATR rose → new_callback > old + 0.1 → replace."""
        pos = _make_pos(last_callback=0.5)
        mock_bot._force_trail_reset = False
        mock_bot.exchange.fetch_open_orders.return_value = [
            {'id': 'existing_sl', 'type': 'STOP_MARKET',
             'info': {'type': 'STOP_MARKET', 'stopPrice': '69500'}},
            {'id': 'old_trail', 'type': 'TRAILING_STOP_MARKET',
             'info': {'type': 'TRAILING_STOP_MARKET'}},
        ]
        mock_bot.exchange.fetch_positions.return_value = [{
            'side': 'LONG', 'contracts': 0.01, 'entryPrice': 70000,
        }]
        mock_bot.exchange.create_order.return_value = {'id': 'new_trail'}
        # ATR high → trail_K * atr / price * 100 = 2.5 * 700 / 70100 * 100 = 2.5%
        # old_callback = 0.5, new = 2.5 → diff > 0.1 → LOOSEN
        mock_bot._update_exchange_trail(pos, 70100, 700.0)
        # Cancel happened (loosen path)
        mock_bot.exchange.cancel_order.assert_called()
        # New trail placed
        assert pos['last_callback'] > 0.5

    def test_no_replace_when_atr_steady(self, mock_bot):
        """B. Parity: ATR unchanged → callback unchanged → no replace.

        Preserves BingX best_price tracking.
        """
        pos = _make_pos(last_callback=0.5)
        mock_bot._force_trail_reset = False
        mock_bot.exchange.fetch_open_orders.return_value = [
            {'id': 'existing_sl', 'type': 'STOP_MARKET',
             'info': {'type': 'STOP_MARKET', 'stopPrice': '69500'}},
            {'id': 'trail', 'type': 'TRAILING_STOP_MARKET',
             'info': {'type': 'TRAILING_STOP_MARKET'}},
        ]
        # ATR produces same callback = 0.5
        # trail_K * atr / price * 100 = 2.5 * 140.2 / 70100 * 100 = ~0.5
        mock_bot._update_exchange_trail(pos, 70100, 140.2)
        # No cancel should happen
        mock_bot.exchange.cancel_order.assert_not_called()

    def test_no_replace_when_atr_falls_tighten(self, mock_bot):
        """BUG#46 core: ATR↓ should NOT trigger replace (tighten preserved for check_exit)."""
        pos = _make_pos(last_callback=2.0)
        mock_bot._force_trail_reset = False
        mock_bot.exchange.fetch_open_orders.return_value = [
            {'id': 'existing_sl', 'type': 'STOP_MARKET',
             'info': {'type': 'STOP_MARKET', 'stopPrice': '69500'}},
            {'id': 'trail', 'type': 'TRAILING_STOP_MARKET',
             'info': {'type': 'TRAILING_STOP_MARKET'}},
        ]
        # Low ATR → new_callback much smaller than 2.0
        mock_bot._update_exchange_trail(pos, 70100, 100.0)
        mock_bot.exchange.cancel_order.assert_not_called()


class TestFailureStreak:
    """BUG#59: _trail_update_fail_streak counter for observability."""

    def test_success_resets_streak(self, mock_bot):
        pos = _make_pos()
        mock_bot._force_trail_reset = False
        mock_bot.exchange.fetch_open_orders.return_value = [
            {'id': 'existing_sl', 'type': 'STOP_MARKET',
             'info': {'type': 'STOP_MARKET', 'stopPrice': '69500'}},
            {'id': 'trail', 'type': 'TRAILING_STOP_MARKET',
             'info': {'type': 'TRAILING_STOP_MARKET'}},
        ]
        mock_bot._trail_update_fail_streak = 5
        mock_bot._update_exchange_trail(pos, 70100, 100.0)
        assert mock_bot._trail_update_fail_streak == 0

    def test_failure_increments_streak(self, mock_bot):
        pos = _make_pos()
        mock_bot._force_trail_reset = False
        mock_bot.exchange.fetch_open_orders.side_effect = Exception('rate limit')
        mock_bot._update_exchange_trail(pos, 70100, 100.0)
        assert mock_bot._trail_update_fail_streak == 1
        mock_bot._update_exchange_trail(pos, 70100, 100.0)
        assert mock_bot._trail_update_fail_streak == 2
        mock_bot._update_exchange_trail(pos, 70100, 100.0)
        assert mock_bot._trail_update_fail_streak == 3

    def test_elevated_warning_at_streak_3(self, mock_bot, caplog):
        """A. Edge: ≥3 consecutive failures → elevated warning message."""
        import logging
        caplog.set_level(logging.WARNING, logger='c1_breakout')
        pos = _make_pos()
        mock_bot._force_trail_reset = False
        mock_bot.exchange.fetch_open_orders.side_effect = Exception('E')
        for _ in range(3):
            mock_bot._update_exchange_trail(pos, 70100, 100.0)
        # Last message at streak=3 should mention SL verification NOT running
        streak_warnings = [r for r in caplog.records if 'NOT running' in r.message]
        assert len(streak_warnings) >= 1
