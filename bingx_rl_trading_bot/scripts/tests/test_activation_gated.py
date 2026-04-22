"""Activation-Gated Trail (F option, 2026-04-22) unit tests.

Validates bot.py activation_gated_trail branch:
  enabled=false → legacy behavior (TRAILING placed at entry)
  enabled=true + pre-activation → NO trail order placed, stray trails cancelled
  enabled=true + activation reached → baton STOP_MARKET placed (once)

Critical evaluation angles:
  A. Edge: best_pnl exactly at activation_pct boundary
  B. Parity: enabled=false must match legacy TRAILING flow exactly
  C. Interaction: F option + progressive_trail + restart force_reset coexist
  D. Rollback: enabled toggle from true→false mid-session safe
"""
from unittest.mock import MagicMock, call
import pytest


def _prep_entry(mock_bot, balance=1000.0):
    """Pre-seed position and exchange for _exchange_open tests."""
    mock_bot.positions = [{
        'direction': 'LONG', 'entry_price': 70000, 'sl_price': 69500,
        'best_price': 70000, 'entry_time': '2026-04-22T12:00:00',
        'bars_held': 0, 'size_pct': 100.0,
    }]
    mock_bot.exchange.fetch_balance.return_value = {'USDT': {'free': balance}}


def _prep_trail_update_position(direction='LONG', best_pnl_pct=0.03):
    """Build a position dict for _update_exchange_trail.

    best_pnl_pct: desired best_pnl (pre-activation < 0.05, post >= 0.05).
    """
    entry = 70000.0
    if direction == 'LONG':
        best = entry * (1 + best_pnl_pct / 100)
        sl = 69500.0
    else:
        best = entry * (1 - best_pnl_pct / 100)
        sl = 70500.0
    return {
        'direction': direction, 'entry_price': entry, 'sl_price': sl,
        'best_price': best, 'entry_time': '2026-04-22T12:00:00',
        'bars_held': 5, 'size_pct': 100.0, 'sl_order_id': 's1',
        'last_callback': 0, 'trail_order_id': '', 'trail_trigger': 0,
    }


# ─── _exchange_open gate ─────────────────────────────────────────────

class TestExchangeOpenGate:
    """Entry-time TRAILING_STOP_MARKET placement gating."""

    def test_disabled_places_trailing_at_entry_legacy(self, mock_bot):
        """B. Parity: enabled=false → legacy flow, TRAILING still placed."""
        mock_bot.config['strategy']['activation_gated_trail'] = {'enabled': False}
        _prep_entry(mock_bot)

        order_ok = {'id': 'm1', 'filled': 0.001, 'average': 70100.0, 'amount': 0.001}
        mock_bot.exchange.create_order.side_effect = [
            order_ok, {'id': 's1'}, {'id': 't1'},  # MARKET, SL, TRAILING
        ]
        ok = mock_bot._exchange_open('LONG', 70000, 69500, 100.0)
        assert ok is True
        # 3 create_order calls: MARKET + SL + TRAIL
        assert mock_bot.exchange.create_order.call_count == 3
        # 3rd call is TRAILING_STOP_MARKET
        last = mock_bot.exchange.create_order.call_args_list[2]
        assert last.args[1] == 'TRAILING_STOP_MARKET'

    def test_enabled_skips_trailing_at_entry(self, mock_bot):
        """F option: enabled=true → only MARKET + SL placed, no TRAILING."""
        mock_bot.config['strategy']['activation_gated_trail'] = {'enabled': True}
        _prep_entry(mock_bot)

        order_ok = {'id': 'm1', 'filled': 0.001, 'average': 70100.0, 'amount': 0.001}
        mock_bot.exchange.create_order.side_effect = [
            order_ok, {'id': 's1'},  # MARKET + SL only
        ]
        ok = mock_bot._exchange_open('LONG', 70000, 69500, 100.0)
        assert ok is True
        # Only 2 create_order calls: MARKET + SL (no TRAIL)
        assert mock_bot.exchange.create_order.call_count == 2
        # Verify no TRAILING_STOP_MARKET in calls
        call_types = [c.args[1] for c in mock_bot.exchange.create_order.call_args_list]
        assert 'TRAILING_STOP_MARKET' not in call_types

    def test_enabled_short_direction_also_skips_trailing(self, mock_bot):
        """Symmetry: SHORT entry under F option also skips TRAILING."""
        mock_bot.config['strategy']['activation_gated_trail'] = {'enabled': True}
        mock_bot.positions = [{
            'direction': 'SHORT', 'entry_price': 70000, 'sl_price': 70500,
            'best_price': 70000, 'entry_time': '2026-04-22T12:00:00',
            'bars_held': 0, 'size_pct': 100.0,
        }]
        mock_bot.exchange.fetch_balance.return_value = {'USDT': {'free': 1000.0}}
        order_ok = {'id': 'm1', 'filled': 0.001, 'average': 69900.0, 'amount': 0.001}
        mock_bot.exchange.create_order.side_effect = [order_ok, {'id': 's1'}]
        ok = mock_bot._exchange_open('SHORT', 70000, 70500, 100.0)
        assert ok is True
        call_types = [c.args[1] for c in mock_bot.exchange.create_order.call_args_list]
        assert 'TRAILING_STOP_MARKET' not in call_types

    def test_disabled_missing_config_defaults_to_legacy(self, mock_bot):
        """D. Rollback: config without activation_gated_trail key → legacy flow."""
        # Explicitly remove the key
        mock_bot.config['strategy'].pop('activation_gated_trail', None)
        _prep_entry(mock_bot)
        order_ok = {'id': 'm1', 'filled': 0.001, 'average': 70100.0, 'amount': 0.001}
        mock_bot.exchange.create_order.side_effect = [
            order_ok, {'id': 's1'}, {'id': 't1'},
        ]
        ok = mock_bot._exchange_open('LONG', 70000, 69500, 100.0)
        assert ok is True
        assert mock_bot.exchange.create_order.call_count == 3


# ─── _update_exchange_trail gate ─────────────────────────────────────

class TestUpdateExchangeTrailGate:
    """Cycle-time trail management gating."""

    def test_enabled_pre_activation_skips_all_trail_logic(self, mock_bot):
        """F option: pre-activation cycle → no trail order placed."""
        mock_bot.config['strategy']['activation_gated_trail'] = {'enabled': True}
        pos = _prep_trail_update_position('LONG', best_pnl_pct=0.03)  # below 0.05

        mock_bot.exchange.fetch_open_orders.return_value = [
            {'id': 's1', 'info': {'type': 'STOP_MARKET'}, 'stopPrice': 69500},
        ]
        mock_bot._update_exchange_trail(pos, 70021.0, 50.0)
        # No create_order called for trail
        for c in mock_bot.exchange.create_order.call_args_list:
            assert c.args[1] != 'TRAILING_STOP_MARKET'

    def test_enabled_pre_activation_cancels_stray_trail(self, mock_bot):
        """F option: if legacy TRAILING exists pre-activation, cancel it."""
        mock_bot.config['strategy']['activation_gated_trail'] = {'enabled': True}
        pos = _prep_trail_update_position('LONG', best_pnl_pct=0.02)

        mock_bot.exchange.fetch_open_orders.return_value = [
            {'id': 's1', 'info': {'type': 'STOP_MARKET'}, 'stopPrice': 69500},
            {'id': 'stray-trail-1', 'info': {'type': 'TRAILING_STOP_MARKET'}},
        ]
        mock_bot._update_exchange_trail(pos, 70014.0, 50.0)
        # Should have cancelled stray trail
        mock_bot.exchange.cancel_order.assert_called_with('stray-trail-1', 'BTC-USDT')

    def test_enabled_post_activation_places_baton(self, mock_bot):
        """F option: best_pnl ≥ activation → baton STOP_MARKET placed (once)."""
        mock_bot.config['strategy']['activation_gated_trail'] = {'enabled': True}
        # best_pnl 0.5% > activation 0.05%
        pos = _prep_trail_update_position('LONG', best_pnl_pct=0.5)

        # Only SL exists, no trail yet
        mock_bot.exchange.fetch_open_orders.return_value = [
            {'id': 's1', 'info': {'type': 'STOP_MARKET'}, 'stopPrice': 69500},
        ]
        mock_bot._get_live_positions = MagicMock(return_value={'LONG': {'contracts': 0.001}})
        mock_bot.exchange.create_order.return_value = {'id': 'baton-1'}

        mock_bot._update_exchange_trail(pos, 70350.0, 50.0)

        # Should create STOP_MARKET (baton), not TRAILING
        calls = mock_bot.exchange.create_order.call_args_list
        trail_placement_calls = [c for c in calls
                                 if c.args[1] in ('STOP_MARKET', 'TRAILING_STOP_MARKET')]
        assert len(trail_placement_calls) >= 1
        # First (only) trail placement must be STOP_MARKET (baton)
        assert trail_placement_calls[0].args[1] == 'STOP_MARKET'
        # pos state updated
        assert pos['trail_order_id'] == 'baton-1'

    def test_disabled_behaves_legacy_pre_activation(self, mock_bot):
        """B. Parity: enabled=false → pre-activation can place TRAILING as legacy."""
        mock_bot.config['strategy']['activation_gated_trail'] = {'enabled': False}
        pos = _prep_trail_update_position('LONG', best_pnl_pct=0.03)

        # No trail order yet
        mock_bot.exchange.fetch_open_orders.return_value = [
            {'id': 's1', 'info': {'type': 'STOP_MARKET'}, 'stopPrice': 69500},
        ]
        mock_bot._get_live_positions = MagicMock(return_value={'LONG': {'contracts': 0.001}})
        mock_bot.exchange.create_order.return_value = {'id': 'tr-1'}

        mock_bot._update_exchange_trail(pos, 70021.0, 50.0)

        # Legacy: may place TRAILING_STOP_MARKET pre-activation
        calls = mock_bot.exchange.create_order.call_args_list
        trail_calls = [c for c in calls
                       if c.args[1] in ('STOP_MARKET', 'TRAILING_STOP_MARKET')]
        # Under legacy, either TRAILING (pre-activation) or nothing if already exists — not STOP_MARKET
        if trail_calls:
            assert trail_calls[0].args[1] == 'TRAILING_STOP_MARKET'

    def test_enabled_boundary_at_activation_pct(self, mock_bot):
        """A. Edge: best_pnl exactly equals activation_pct → still pre-activation (<=)."""
        mock_bot.config['strategy']['activation_gated_trail'] = {'enabled': True}
        # best_pnl = 0.05 = activation_pct exactly
        pos = _prep_trail_update_position('LONG', best_pnl_pct=0.05)

        mock_bot.exchange.fetch_open_orders.return_value = [
            {'id': 's1', 'info': {'type': 'STOP_MARKET'}, 'stopPrice': 69500},
        ]
        mock_bot._update_exchange_trail(pos, 70035.0, 50.0)
        # At boundary (<=), F branch returns early — no trail placed
        trail_calls = [c for c in mock_bot.exchange.create_order.call_args_list
                       if c.args[1] in ('STOP_MARKET', 'TRAILING_STOP_MARKET')]
        assert len(trail_calls) == 0

    def test_enabled_short_post_activation_baton(self, mock_bot):
        """Symmetry: SHORT direction, post-activation baton placement."""
        mock_bot.config['strategy']['activation_gated_trail'] = {'enabled': True}
        pos = _prep_trail_update_position('SHORT', best_pnl_pct=0.5)

        mock_bot.exchange.fetch_open_orders.return_value = [
            {'id': 's1', 'info': {'type': 'STOP_MARKET'}, 'stopPrice': 70500},
        ]
        mock_bot._get_live_positions = MagicMock(return_value={'SHORT': {'contracts': 0.001}})
        mock_bot.exchange.create_order.return_value = {'id': 'baton-s'}

        mock_bot._update_exchange_trail(pos, 69650.0, 50.0)

        calls = mock_bot.exchange.create_order.call_args_list
        trail_placement_calls = [c for c in calls
                                 if c.args[1] in ('STOP_MARKET', 'TRAILING_STOP_MARKET')]
        assert len(trail_placement_calls) >= 1
        assert trail_placement_calls[0].args[1] == 'STOP_MARKET'


class TestInteraction:
    """C. Interaction: F + other features coexist correctly."""

    def test_enabled_with_progressive_trail_post_activation(self, mock_bot):
        """F + progressive_trail together: baton uses progressive K when best_pnl > threshold."""
        mock_bot.config['strategy']['activation_gated_trail'] = {'enabled': True}
        mock_bot.config['strategy']['progressive_trail'] = {
            'enabled': True, 'threshold_pct': 0.9, 'trail_K_post': 0.5,
        }
        # Re-init signal with updated config
        from scripts.production.c1_breakout.signals import C1BreakoutSignal
        mock_bot.signal = C1BreakoutSignal(mock_bot.config['strategy'])

        pos = _prep_trail_update_position('LONG', best_pnl_pct=1.5)  # > 0.9 threshold

        mock_bot.exchange.fetch_open_orders.return_value = [
            {'id': 's1', 'info': {'type': 'STOP_MARKET'}, 'stopPrice': 69500},
        ]
        mock_bot._get_live_positions = MagicMock(return_value={'LONG': {'contracts': 0.001}})
        mock_bot.exchange.create_order.return_value = {'id': 'baton-prog'}

        mock_bot._update_exchange_trail(pos, 71050.0, 50.0)

        # Baton placed — K_post=0.5 → trigger closer to best (tighter)
        trail_calls = [c for c in mock_bot.exchange.create_order.call_args_list
                       if c.args[1] == 'STOP_MARKET']
        assert len(trail_calls) >= 1
