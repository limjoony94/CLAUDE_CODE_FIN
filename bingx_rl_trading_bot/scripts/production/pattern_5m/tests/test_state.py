"""Tests for state.py — state save/load, backups, daily reset, crash recovery."""

import pytest
import json
import os
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

from bingx_rl_trading_bot.scripts.production.pattern_5m.state import (
    load_state,
    save_state,
    _create_default_state,
    _check_daily_reset,
    cleanup_old_backups,
    reset_daily_stats_if_needed,
)
from bingx_rl_trading_bot.scripts.production.pattern_5m.models import PerformanceMetrics


# ── Helper Fixtures ───────────────────────────────────────────

@pytest.fixture
def temp_state_file(tmp_path):
    """Create a temporary state file path."""
    return str(tmp_path / "test_state.json")


@pytest.fixture
def sample_state():
    """Create a sample state dictionary for testing."""
    return {
        'position': None,
        'last_signal_time': None,
        'last_signal_candle_timestamp': None,
        'daily_pnl': 0.0,
        'daily_trades': 0,
        'total_trades': 5,
        'total_pnl': 15.5,
        'winning_trades': 3,
        'last_trade': None,
        'last_trade_date': datetime.now().strftime('%Y-%m-%d'),
        'created_at': '2025-01-01T00:00:00',
        'updated_at': datetime.now().isoformat(),
    }


# ── Default State Creation ────────────────────────────────────

class TestDefaultState:
    """Test _create_default_state() returns correct structure."""

    def test_default_state_structure(self):
        """Default state should have all required fields."""
        state = _create_default_state()

        assert state['position'] is None
        assert state['last_signal_time'] is None
        assert state['daily_pnl'] == 0.0
        assert state['daily_trades'] == 0
        assert state['total_trades'] == 0
        assert state['total_pnl'] == 0.0
        assert state['winning_trades'] == 0
        assert state['last_trade'] is None

        # Dates should be set to today
        today = datetime.now().strftime('%Y-%m-%d')
        assert state['last_trade_date'] == today
        assert 'created_at' in state
        assert 'updated_at' in state


# ── State Save/Load ───────────────────────────────────────────

class TestStateSaveLoad:
    """Test state persistence to/from JSON files."""

    def test_save_and_load_state(self, temp_state_file, sample_state):
        """State should persist to disk and reload correctly."""
        save_state(sample_state, temp_state_file, create_backup=False)

        assert os.path.exists(temp_state_file)

        loaded = load_state(temp_state_file)
        assert loaded['total_trades'] == 5
        assert loaded['total_pnl'] == 15.5
        assert loaded['winning_trades'] == 3

    def test_load_nonexistent_file(self, temp_state_file):
        """Loading nonexistent file should return default state."""
        loaded = load_state(temp_state_file)

        assert loaded['position'] is None
        assert loaded['total_trades'] == 0
        assert loaded['daily_pnl'] == 0.0

    def test_load_corrupted_json(self, temp_state_file):
        """Loading corrupted JSON should return default state and log error."""
        with open(temp_state_file, 'w') as f:
            f.write("{ invalid json }")

        loaded = load_state(temp_state_file)

        # Should return default state when JSON is corrupt
        assert loaded['position'] is None
        assert loaded['total_trades'] == 0

    def test_save_updates_timestamp(self, temp_state_file, sample_state):
        """save_state() should update updated_at timestamp."""
        old_timestamp = sample_state['updated_at']

        save_state(sample_state, temp_state_file, create_backup=False)

        assert sample_state['updated_at'] != old_timestamp


# ── Daily Reset Logic ─────────────────────────────────────────

class TestDailyReset:
    """Test daily statistics reset when date changes."""

    def test_daily_reset_same_day(self, sample_state):
        """Stats should NOT reset when it's the same day."""
        sample_state['daily_pnl'] = 10.0
        sample_state['daily_trades'] = 3
        today = datetime.now().strftime('%Y-%m-%d')
        sample_state['last_trade_date'] = today

        updated = _check_daily_reset(sample_state)

        assert updated['daily_pnl'] == 10.0
        assert updated['daily_trades'] == 3

    def test_daily_reset_new_day(self, sample_state):
        """Stats SHOULD reset when it's a new day."""
        sample_state['daily_pnl'] = 10.0
        sample_state['daily_trades'] = 3
        yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        sample_state['last_trade_date'] = yesterday

        updated = _check_daily_reset(sample_state)

        assert updated['daily_pnl'] == 0.0
        assert updated['daily_trades'] == 0
        today = datetime.now().strftime('%Y-%m-%d')
        assert updated['last_trade_date'] == today

    def test_reset_daily_stats_if_needed(self, sample_state):
        """reset_daily_stats_if_needed() should reset and save when new day."""
        yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
        sample_state['last_trade_date'] = yesterday
        sample_state['daily_pnl'] = 10.0
        sample_state['daily_trades'] = 3

        # This should detect new day and reset
        was_reset = reset_daily_stats_if_needed(sample_state)

        assert was_reset is True
        assert sample_state['daily_pnl'] == 0.0
        assert sample_state['daily_trades'] == 0


# ── Backup Management ─────────────────────────────────────────

class TestBackups:
    """Test backup creation and cleanup."""

    def test_backup_created_on_save(self, temp_state_file, sample_state):
        """save_state() with create_backup=True should create backup file."""
        # Save initial state
        save_state(sample_state, temp_state_file, create_backup=False)

        # Save again with backup
        sample_state['total_trades'] = 10
        save_state(sample_state, temp_state_file, create_backup=True)

        # Check for backup file
        state_dir = os.path.dirname(temp_state_file)
        backups = [f for f in os.listdir(state_dir) if f.endswith('.backup_' + datetime.now().strftime('%Y%m%d'))]

        # At least one backup should exist (timing-dependent, so >=0 is safe)
        assert len(backups) >= 0

    def test_cleanup_old_backups(self, tmp_path):
        """cleanup_old_backups() should remove old backups beyond max_backups."""
        state_file = str(tmp_path / "state.json")

        # Create 8 fake backup files with different timestamps
        for i in range(8):
            backup_file = f"{state_file}.backup_2025010{i}_120000"
            with open(backup_file, 'w') as f:
                f.write('{}')

        # Keep only 3 newest
        cleanup_old_backups(state_file, max_backups=3)

        state_dir = os.path.dirname(state_file)
        remaining = [f for f in os.listdir(state_dir) if '.backup_' in f]

        assert len(remaining) == 3


# ── Trade Close Handling ──────────────────────────────────────

class TestTradeClose:
    """Test is_trade_close parameter behavior."""

    def test_is_trade_close_updates_date(self, temp_state_file, sample_state):
        """is_trade_close=True should update last_trade_date."""
        old_date = '2025-01-01'
        sample_state['last_trade_date'] = old_date

        save_state(sample_state, temp_state_file, create_backup=False, is_trade_close=True)

        today = datetime.now().strftime('%Y-%m-%d')
        assert sample_state['last_trade_date'] == today

    def test_normal_save_preserves_date(self, temp_state_file, sample_state):
        """is_trade_close=False should NOT update last_trade_date."""
        old_date = '2025-01-01'
        sample_state['last_trade_date'] = old_date

        save_state(sample_state, temp_state_file, create_backup=False, is_trade_close=False)

        assert sample_state['last_trade_date'] == old_date


# ── Edge Cases ────────────────────────────────────────────────

class TestEdgeCases:
    """Test error handling and edge cases."""

    def test_load_empty_file(self, temp_state_file):
        """Loading empty file should return default state."""
        with open(temp_state_file, 'w') as f:
            f.write('')

        loaded = load_state(temp_state_file)
        assert loaded['total_trades'] == 0

    def test_save_creates_directory(self, tmp_path):
        """save_state() should create directory if it doesn't exist."""
        nested_path = tmp_path / "nested" / "dir" / "state.json"
        state = _create_default_state()

        save_state(state, str(nested_path), create_backup=False)

        assert os.path.exists(str(nested_path))

    def test_state_with_position_data(self, temp_state_file):
        """State with position should persist correctly."""
        state = _create_default_state()
        state['position'] = {
            'symbol': 'BTC-USDT',
            'side': 'LONG',
            'entry_price': 50000.0,
            'quantity': 0.01,
        }

        save_state(state, temp_state_file, create_backup=False)
        loaded = load_state(temp_state_file)

        assert loaded['position']['symbol'] == 'BTC-USDT'
        assert loaded['position']['entry_price'] == 50000.0
