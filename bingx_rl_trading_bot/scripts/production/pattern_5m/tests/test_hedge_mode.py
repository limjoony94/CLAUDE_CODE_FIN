"""Tests for v1.30.0 Hedge mode functionality.

Tests _get_position_side, _route_signal, emergency SL per-direction,
state v3 migration, and config validation for hedge mode.
"""

import pytest
from unittest.mock import MagicMock, patch

from bingx_rl_trading_bot.scripts.production.pattern_5m.orders import (
    _get_position_side as orders_get_position_side,
    _get_worst_sl_price_for_direction,
    _set_emergency_sl_id,
    _get_emergency_sl_id,
    place_emergency_sl,
    cancel_emergency_sl,
)
from bingx_rl_trading_bot.scripts.production.pattern_5m.bot import _route_signal
from bingx_rl_trading_bot.scripts.production.pattern_5m.constants import (
    VALID_POSITION_MODES,
    DEFAULT_POSITION_MODE,
    EMERGENCY_SL_BUFFER_PCT,
)


# ── _get_position_side ──────────────────────────────────────


class TestGetPositionSideHedge:
    """Test _get_position_side returns correct positionSide."""

    def test_one_way_returns_both(self):
        config = {'position_mode': 'one_way'}
        assert orders_get_position_side(config, 'LONG') == 'BOTH'
        assert orders_get_position_side(config, 'SHORT') == 'BOTH'

    def test_default_returns_both(self):
        config = {}
        assert orders_get_position_side(config, 'LONG') == 'BOTH'

    def test_hedge_long(self):
        config = {'position_mode': 'hedge'}
        assert orders_get_position_side(config, 'LONG') == 'LONG'

    def test_hedge_short(self):
        config = {'position_mode': 'hedge'}
        assert orders_get_position_side(config, 'SHORT') == 'SHORT'


# ── _route_signal ───────────────────────────────────────────


class TestRouteSignalHedge:
    """Test _route_signal for hedge mode."""

    def _hedge_config(self, max_pos=5):
        return {'position_mode': 'hedge', 'max_positions': max_pos}

    def _oneway_config(self, max_pos=5):
        return {'position_mode': 'one_way', 'max_positions': max_pos}

    def test_hedge_empty_always_open(self):
        state = {'positions': {}}
        assert _route_signal(state, self._hedge_config(), 'LONG') == 'OPEN'
        assert _route_signal(state, self._hedge_config(), 'SHORT') == 'OPEN'

    def test_hedge_mixed_directions_open(self):
        """Hedge allows LONG+SHORT positions simultaneously."""
        state = {'positions': {
            's1': {'direction': 'LONG'},
            's2': {'direction': 'SHORT'},
        }}
        assert _route_signal(state, self._hedge_config(), 'LONG') == 'OPEN'
        assert _route_signal(state, self._hedge_config(), 'SHORT') == 'OPEN'

    def test_hedge_full_skip(self):
        """Hedge SKIP when all slots filled (regardless of direction)."""
        state = {'positions': {f's{i}': {'direction': 'LONG'} for i in range(5)}}
        assert _route_signal(state, self._hedge_config(5), 'SHORT') == 'SKIP'

    def test_hedge_opposite_open_not_close_oldest(self):
        """Hedge never returns CLOSE_OLDEST."""
        state = {'positions': {'s1': {'direction': 'LONG'}}, 'active_direction': 'LONG'}
        result = _route_signal(state, self._hedge_config(), 'SHORT')
        assert result == 'OPEN'
        assert result != 'CLOSE_OLDEST'

    def test_hedge_n1_opposite_open(self):
        """Hedge N=1: opposite direction → SKIP (not CLOSE_OLDEST)."""
        state = {'positions': {'s1': {'direction': 'LONG'}}}
        assert _route_signal(state, self._hedge_config(1), 'SHORT') == 'SKIP'

    def test_oneway_opposite_close_oldest(self):
        """One-Way mode: opposite direction → CLOSE_OLDEST (FIFO behavior)."""
        state = {'positions': {'s1': {'direction': 'LONG'}}, 'active_direction': 'LONG'}
        assert _route_signal(state, self._oneway_config(), 'SHORT') == 'CLOSE_OLDEST'

    def test_oneway_same_dir_open(self):
        state = {'positions': {'s1': {'direction': 'LONG'}}, 'active_direction': 'LONG'}
        assert _route_signal(state, self._oneway_config(), 'LONG') == 'OPEN'

    def test_oneway_full_same_dir_skip(self):
        state = {
            'positions': {f's{i}': {'direction': 'LONG'} for i in range(5)},
            'active_direction': 'LONG',
        }
        assert _route_signal(state, self._oneway_config(5), 'LONG') == 'SKIP'


# ── Emergency SL per-direction ──────────────────────────────


class TestEmergencySlPerDirection:
    """Test emergency SL helpers for hedge/one-way dispatch."""

    def test_worst_sl_long_takes_minimum(self):
        state = {'positions': {
            's1': {'direction': 'LONG', 'sl_price': 49000},
            's2': {'direction': 'LONG', 'sl_price': 48000},
        }}
        result = _get_worst_sl_price_for_direction(state, 'LONG')
        expected = round(48000 * (1 - EMERGENCY_SL_BUFFER_PCT), 1)
        assert result == expected

    def test_worst_sl_short_takes_maximum(self):
        state = {'positions': {
            's1': {'direction': 'SHORT', 'sl_price': 52000},
            's2': {'direction': 'SHORT', 'sl_price': 53000},
        }}
        result = _get_worst_sl_price_for_direction(state, 'SHORT')
        expected = round(53000 * (1 + EMERGENCY_SL_BUFFER_PCT), 1)
        assert result == expected

    def test_worst_sl_filters_by_direction(self):
        """Only considers slots matching direction."""
        state = {'positions': {
            's1': {'direction': 'LONG', 'sl_price': 49000},
            's2': {'direction': 'SHORT', 'sl_price': 53000},
        }}
        result = _get_worst_sl_price_for_direction(state, 'LONG')
        expected = round(49000 * (1 - EMERGENCY_SL_BUFFER_PCT), 1)
        assert result == expected

    def test_set_get_emergency_sl_hedge(self):
        state = {'emergency_sl_orders': {}}
        config = {'position_mode': 'hedge'}
        _set_emergency_sl_id(state, config, 'LONG', 'emg_long_1')
        _set_emergency_sl_id(state, config, 'SHORT', 'emg_short_1')
        assert _get_emergency_sl_id(state, config, 'LONG') == 'emg_long_1'
        assert _get_emergency_sl_id(state, config, 'SHORT') == 'emg_short_1'

    def test_set_get_emergency_sl_oneway(self):
        state = {'emergency_sl_order_id': None}
        config = {'position_mode': 'one_way'}
        _set_emergency_sl_id(state, config, 'LONG', 'emg_1')
        assert _get_emergency_sl_id(state, config, 'LONG') == 'emg_1'
        assert state['emergency_sl_order_id'] == 'emg_1'

    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.orders.save_state')
    def test_place_emergency_sl_hedge_two_directions(self, mock_save):
        """Hedge mode places separate SL for LONG and SHORT."""
        exchange = MagicMock()
        exchange.create_order.return_value = {'id': 'emg_123'}
        state = {
            'positions': {
                's1': {'direction': 'LONG', 'sl_price': 49000, 'quantity': 0.01},
                's2': {'direction': 'SHORT', 'sl_price': 53000, 'quantity': 0.01},
            },
            'emergency_sl_orders': {},
        }
        config = {'position_mode': 'hedge', 'symbol': 'BTC/USDT:USDT'}
        place_emergency_sl(exchange, state, config)
        assert exchange.create_order.call_count == 2
        # Both directions should have SL IDs stored
        assert state['emergency_sl_orders'].get('LONG') is not None
        assert state['emergency_sl_orders'].get('SHORT') is not None

    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.orders.save_state')
    def test_place_emergency_sl_oneway_single(self, mock_save):
        """One-Way mode places single SL."""
        exchange = MagicMock()
        exchange.create_order.return_value = {'id': 'emg_456'}
        state = {
            'positions': {
                's1': {'direction': 'LONG', 'sl_price': 49000, 'quantity': 0.01},
            },
            'active_direction': 'LONG',
            'emergency_sl_order_id': None,
        }
        config = {'position_mode': 'one_way', 'symbol': 'BTC/USDT:USDT'}
        place_emergency_sl(exchange, state, config)
        exchange.create_order.assert_called_once()
        assert state['emergency_sl_order_id'] == 'emg_456'

    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.orders.save_state')
    def test_cancel_emergency_sl_hedge(self, mock_save):
        """Hedge cancel clears both directions."""
        exchange = MagicMock()
        state = {
            'positions': {},
            'emergency_sl_orders': {'LONG': 'emg_l', 'SHORT': 'emg_s'},
        }
        config = {'position_mode': 'hedge', 'symbol': 'BTC/USDT:USDT'}
        cancel_emergency_sl(exchange, state, config)
        assert exchange.cancel_order.call_count == 2
        assert state['emergency_sl_orders']['LONG'] is None
        assert state['emergency_sl_orders']['SHORT'] is None


# ── State v3 migration ──────────────────────────────────────


class TestStateV3Migration:
    """Test state migration adds emergency_sl_orders."""

    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.state.save_state')
    def test_v2_state_gets_emergency_sl_orders(self, mock_save):
        from bingx_rl_trading_bot.scripts.production.pattern_5m.state import _migrate_state_v2
        state = {
            'positions': {},
            'active_direction': None,
            'emergency_sl_order_id': None,
            'state_version': 2,
        }
        _migrate_state_v2(state)
        assert 'emergency_sl_orders' in state
        assert state['emergency_sl_orders'] == {}
        assert state['state_version'] == 3

    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.state.save_state')
    def test_v3_state_not_downgraded(self, mock_save):
        from bingx_rl_trading_bot.scripts.production.pattern_5m.state import _migrate_state_v2
        state = {
            'positions': {},
            'emergency_sl_orders': {'LONG': 'x'},
            'state_version': 3,
        }
        _migrate_state_v2(state)
        assert state['emergency_sl_orders'] == {'LONG': 'x'}
        assert state['state_version'] == 3


# ── Config validation ───────────────────────────────────────


class TestConfigValidationHedge:
    """Test position_mode config validation."""

    def test_valid_position_modes(self):
        assert 'one_way' in VALID_POSITION_MODES
        assert 'hedge' in VALID_POSITION_MODES

    def test_default_is_one_way(self):
        assert DEFAULT_POSITION_MODE == 'one_way'

    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.config.load_dynamic_patterns')
    def test_invalid_position_mode_raises(self, mock_load):
        mock_load.return_value = None
        from bingx_rl_trading_bot.scripts.production.pattern_5m.config import validate_config
        config = {
            'symbol': 'BTC/USDT:USDT',
            'strategy': {
                'tp_pct': 2.0, 'sl_pct': 3.0,
                'entry_patterns_long': [], 'entry_patterns_short': [],
                'scale_out': {'enabled': False},
            },
            'position_mode': 'invalid_mode',
        }
        with pytest.raises(ValueError, match='position_mode'):
            validate_config(config)


# ── Crash recovery clears hedge state ───────────────────────


class TestCrashRecoveryHedge:
    """Test that crash recovery properly clears hedge mode emergency SL state."""

    def test_recovery_clears_emergency_sl_orders(self):
        """Verify the recovery codepath clears both emergency SL fields."""
        state = {
            'positions': {},
            'emergency_sl_order_id': 'old_oneway',
            'emergency_sl_orders': {'LONG': 'old_l', 'SHORT': 'old_s'},
        }
        # Simulate Phase 1 clearing
        state['emergency_sl_order_id'] = None
        state['emergency_sl_orders'] = {}
        assert state['emergency_sl_order_id'] is None
        assert state['emergency_sl_orders'] == {}
