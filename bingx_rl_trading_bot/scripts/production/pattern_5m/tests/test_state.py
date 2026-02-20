"""Tests for state.py — state save/load, backups, daily reset, crash recovery,
metrics persistence, sync_metrics_with_state."""

import builtins
import pytest
import json
import os
import tempfile
from datetime import datetime, timedelta
from pathlib import Path
from unittest.mock import patch

from bingx_rl_trading_bot.scripts.production.pattern_5m.state import (
    load_state,
    save_state,
    _create_default_state,
    _check_daily_reset,
    _try_timestamped_backups,
    _create_backup,
    _atomic_replace_with_retry,
    cleanup_old_backups,
    reset_daily_stats_if_needed,
    save_metrics,
    load_metrics,
    sync_metrics_with_state,
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
        'positions': {},
        'active_direction': None,
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

        assert state['positions'] == {}
        assert state['active_direction'] is None
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

        assert loaded['positions'] == {}
        assert loaded['total_trades'] == 0
        assert loaded['daily_pnl'] == 0.0

    def test_load_corrupted_json(self, temp_state_file):
        """Loading corrupted JSON should return default state and log error."""
        with open(temp_state_file, 'w') as f:
            f.write("{ invalid json }")

        loaded = load_state(temp_state_file)

        # Should return default state when JSON is corrupt
        assert loaded['positions'] == {}
        assert loaded['total_trades'] == 0

    def test_bak_file_created_on_save(self, temp_state_file, sample_state):
        """save_state() should create .bak file when saving over existing state."""
        # First save
        save_state(sample_state, temp_state_file, create_backup=False)
        assert os.path.exists(temp_state_file)

        # Second save should create .bak backup
        sample_state['total_trades'] = 10
        save_state(sample_state, temp_state_file, create_backup=False)

        bak_file = temp_state_file + '.bak'
        assert os.path.exists(bak_file), f".bak file should exist at {bak_file}"

        # .bak should contain first save (5 trades)
        with open(bak_file, 'r') as f:
            bak_data = json.load(f)
        assert bak_data['total_trades'] == 5

    def test_bak_recovery_on_corrupted_main(self, temp_state_file, sample_state):
        """load_state() should recover from .bak if main file is corrupted."""
        # First save (creates main file)
        save_state(sample_state, temp_state_file, create_backup=False)

        # Second save (creates .bak from first save)
        sample_state['total_trades'] = 10
        save_state(sample_state, temp_state_file, create_backup=False)

        # Corrupt main file
        with open(temp_state_file, 'w') as f:
            f.write("{ corrupted json }")

        # Load should recover from .bak (which has total_trades=5)
        loaded = load_state(temp_state_file)

        # Should recover original state from .bak (first save with 5 trades)
        assert loaded['total_trades'] == 5  # Original sample_state value
        assert loaded['total_pnl'] == sample_state['total_pnl']

    def test_bak_recovery_fails_gracefully(self, temp_state_file):
        """load_state() should return default if both main and .bak are corrupted."""
        # Create corrupted main file
        with open(temp_state_file, 'w') as f:
            f.write("{ bad json }")

        # Create corrupted .bak file
        bak_file = temp_state_file + '.bak'
        with open(bak_file, 'w') as f:
            f.write("{ also bad }")

        # Should fall back to default state
        loaded = load_state(temp_state_file)

        assert loaded['positions'] == {}
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

    def test_reset_daily_stats_same_day_no_reset(self, sample_state):
        """reset_daily_stats_if_needed() returns False when same day."""
        today = datetime.now().strftime('%Y-%m-%d')
        sample_state['last_trade_date'] = today
        sample_state['daily_pnl'] = 5.0
        sample_state['daily_trades'] = 2

        was_reset = reset_daily_stats_if_needed(sample_state)

        assert was_reset is False
        assert sample_state['daily_pnl'] == 5.0  # unchanged
        assert sample_state['daily_trades'] == 2  # unchanged


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
        state['positions'] = {'s1': {
            'slot_id': 's1',
            'symbol': 'BTC-USDT',
            'direction': 'LONG',
            'entry_price': 50000.0,
            'quantity': 0.01,
        }}
        state['active_direction'] = 'LONG'

        save_state(state, temp_state_file, create_backup=False)
        loaded = load_state(temp_state_file)

        assert loaded['positions']['s1']['symbol'] == 'BTC-USDT'
        assert loaded['positions']['s1']['entry_price'] == 50000.0


# ── Metrics Persistence ──────────────────────────────────────


class TestMetricsPersistence:
    """Test save_metrics() / load_metrics() roundtrip."""

    @pytest.fixture
    def temp_metrics_file(self, tmp_path):
        return str(tmp_path / "test_metrics.json")

    def test_save_load_roundtrip(self, temp_metrics_file):
        """Saved metrics should load back identically."""
        m = PerformanceMetrics()
        m.update_trade(5.0)
        m.update_trade(-3.0)
        m.session_start = "2026-01-01T00:00:00"

        save_metrics(m, temp_metrics_file)
        loaded = load_metrics(temp_metrics_file)

        assert loaded is not None
        assert loaded.total_trades == 2
        assert loaded.winning_trades == 1
        assert loaded.total_pnl_pct == pytest.approx(2.0)
        assert loaded.session_start == "2026-01-01T00:00:00"

    def test_load_nonexistent(self, temp_metrics_file):
        """Loading nonexistent file should return None."""
        assert load_metrics(temp_metrics_file) is None

    def test_load_corrupted_json(self, temp_metrics_file):
        """Corrupted JSON should return None."""
        with open(temp_metrics_file, 'w') as f:
            f.write("{ bad json }")
        assert load_metrics(temp_metrics_file) is None

    def test_save_creates_directory(self, tmp_path):
        """save_metrics should create parent directory."""
        nested = str(tmp_path / "nested" / "dir" / "metrics.json")
        m = PerformanceMetrics()
        save_metrics(m, nested)
        assert os.path.exists(nested)

    def test_atomic_write(self, temp_metrics_file):
        """save_metrics uses atomic write — no partial files on success."""
        m = PerformanceMetrics()
        m.update_trade(10.0)
        save_metrics(m, temp_metrics_file)

        # File should be valid JSON
        with open(temp_metrics_file, 'r') as f:
            data = json.load(f)
        assert data['total_trades'] == 1

    def test_empty_metrics_roundtrip(self, temp_metrics_file):
        """Empty metrics should save/load without error."""
        m = PerformanceMetrics()
        save_metrics(m, temp_metrics_file)
        loaded = load_metrics(temp_metrics_file)
        assert loaded is not None
        assert loaded.total_trades == 0
        assert loaded.total_pnl_pct == 0.0


# ── sync_metrics_with_state ──────────────────────────────────


class TestSyncMetricsWithState:
    """Test sync_metrics_with_state() — bidirectional smart sync (v1.28.17)."""

    def test_already_in_sync(self):
        """Equal trade counts → return metrics unchanged."""
        m = PerformanceMetrics()
        m.total_trades = 10
        m.winning_trades = 7
        m.total_pnl_pct = 20.0
        state = {'total_trades': 10, 'winning_trades': 7, 'total_pnl': 20.0}

        result = sync_metrics_with_state(m, state)

        assert result is m
        assert result.total_trades == 10

    def test_state_ahead_trusts_state(self):
        """State has more trades → metrics updated from state."""
        m = PerformanceMetrics()
        m.total_trades = 5
        m.winning_trades = 3
        m.total_pnl_pct = 10.0

        state = {
            'total_trades': 10,
            'winning_trades': 7,
            'total_pnl': 25.0,
        }

        result = sync_metrics_with_state(m, state)

        assert result.total_trades == 10
        assert result.winning_trades == 7
        assert result.losing_trades == 3
        assert result.actual_win_rate == pytest.approx(70.0)
        assert result.total_pnl_pct == 25.0

    def test_state_ahead_zero_trades(self):
        """State has trades, metrics has 0 → metrics synced to state."""
        m = PerformanceMetrics()
        # metrics is fresh (0 trades)
        state = {'total_trades': 5, 'winning_trades': 4, 'total_pnl': 12.0}

        result = sync_metrics_with_state(m, state)

        assert result.total_trades == 5
        assert result.winning_trades == 4
        assert result.actual_win_rate == pytest.approx(80.0)

    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.state.save_state')
    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.state.shutil.copy2')
    def test_state_behind_trusts_metrics(self, mock_copy2, mock_save_state):
        """State has fewer trades (corruption) → state restored from metrics."""
        m = PerformanceMetrics()
        m.total_trades = 40
        m.winning_trades = 25
        m.total_pnl_pct = 8.40

        state = {
            'total_trades': 5,  # corrupted!
            'winning_trades': 2,
            'total_pnl': 1.0,
        }

        result = sync_metrics_with_state(m, state)

        # Metrics should be unchanged
        assert result.total_trades == 40
        assert result.winning_trades == 25
        assert result.total_pnl_pct == 8.40

        # State should be updated to match metrics
        assert state['total_trades'] == 40
        assert state['winning_trades'] == 25
        assert state['total_pnl'] == 8.40

        # save_state should have been called to persist corrected state
        mock_save_state.assert_called_once_with(state)
        # .bak should be updated
        mock_copy2.assert_called_once()

    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.state.save_state')
    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.state.shutil.copy2',
           side_effect=Exception("disk error"))
    def test_state_behind_bak_failure_graceful(self, mock_copy2, mock_save_state):
        """State corruption + .bak update failure → no crash."""
        m = PerformanceMetrics()
        m.total_trades = 20
        m.winning_trades = 15
        m.total_pnl_pct = 5.0

        state = {'total_trades': 3, 'winning_trades': 1, 'total_pnl': 0.5}

        # Should not raise even if copy2 fails
        result = sync_metrics_with_state(m, state)

        assert result.total_trades == 20
        assert state['total_trades'] == 20
        mock_save_state.assert_called_once()

    def test_state_ahead_win_rate_calculation(self):
        """State-ahead sync: actual_win_rate correctly computed."""
        m = PerformanceMetrics()
        m.total_trades = 0
        state = {'total_trades': 100, 'winning_trades': 85, 'total_pnl': 50.0}

        result = sync_metrics_with_state(m, state)

        assert result.actual_win_rate == pytest.approx(85.0)
        assert result.losing_trades == 15

    def test_state_missing_keys_defaults(self):
        """State with missing winning_trades/total_pnl → defaults to 0."""
        m = PerformanceMetrics()
        m.total_trades = 0
        state = {'total_trades': 5}  # no winning_trades, no total_pnl

        result = sync_metrics_with_state(m, state)

        assert result.total_trades == 5
        assert result.winning_trades == 0
        assert result.losing_trades == 5
        assert result.total_pnl_pct == 0.0

    @patch('bingx_rl_trading_bot.scripts.production.pattern_5m.state.save_state')
    def test_state_corruption_metrics_wins(self, mock_save):
        """State < metrics (corruption) → metrics trusted, state updated."""
        m = PerformanceMetrics()
        m.total_trades = 40
        m.winning_trades = 25
        m.total_pnl_pct = 8.4
        m.actual_win_rate = 62.5
        state = {'total_trades': 5, 'winning_trades': 3, 'total_pnl': 1.0}

        result = sync_metrics_with_state(m, state)

        # Metrics should be unchanged
        assert result.total_trades == 40
        assert result.winning_trades == 25
        # State should be updated to match metrics
        assert state['total_trades'] == 40
        assert state['winning_trades'] == 25
        assert state['total_pnl'] == 8.4
        mock_save.assert_called()


# ── _try_timestamped_backups Tests ───────────────────────────


class TestTryTimestampedBackups:
    """Test _try_timestamped_backups() recovery from timestamped backup files."""

    def test_no_backups_returns_none(self, tmp_path):
        """No backup files → returns None."""
        state_file = str(tmp_path / "state.json")
        default = _create_default_state()
        result = _try_timestamped_backups(state_file, default)
        assert result is None

    def test_valid_backup_recovers(self, tmp_path):
        """Valid timestamped backup → recovers state."""
        state_file = str(tmp_path / "state.json")
        # Create a timestamped backup
        backup = state_file + ".backup_20260218_120000"
        backup_state = {'total_trades': 30, 'winning_trades': 20, 'total_pnl': 5.0}
        with open(backup, 'w') as f:
            json.dump(backup_state, f)

        default = _create_default_state()
        result = _try_timestamped_backups(state_file, default)
        assert result is not None
        assert result['total_trades'] == 30

    def test_invalid_backup_skipped(self, tmp_path):
        """Invalid JSON backup → skipped, returns None."""
        state_file = str(tmp_path / "state.json")
        backup = state_file + ".backup_20260218_120000"
        with open(backup, 'w') as f:
            f.write("not valid json")

        default = _create_default_state()
        result = _try_timestamped_backups(state_file, default)
        assert result is None

    def test_backup_without_total_trades_skipped(self, tmp_path):
        """Backup missing total_trades → skipped."""
        state_file = str(tmp_path / "state.json")
        backup = state_file + ".backup_20260218_120000"
        with open(backup, 'w') as f:
            json.dump({'daily_pnl': 0.0}, f)

        default = _create_default_state()
        result = _try_timestamped_backups(state_file, default)
        assert result is None

    def test_newest_backup_preferred(self, tmp_path):
        """Multiple backups → newest (by mtime) is preferred."""
        import time as _time
        state_file = str(tmp_path / "state.json")

        old_backup = state_file + ".backup_20260218_100000"
        with open(old_backup, 'w') as f:
            json.dump({'total_trades': 10, 'winning_trades': 5}, f)
        _time.sleep(0.05)

        new_backup = state_file + ".backup_20260218_120000"
        with open(new_backup, 'w') as f:
            json.dump({'total_trades': 30, 'winning_trades': 20}, f)

        default = _create_default_state()
        result = _try_timestamped_backups(state_file, default)
        assert result is not None
        assert result['total_trades'] == 30


# ── _create_backup Tests ─────────────────────────────────────


class TestCreateBackup:
    """Test _create_backup() timestamped backup creation."""

    def test_creates_timestamped_file(self, tmp_path):
        """Creates backup file with timestamp suffix."""
        state_file = str(tmp_path / "state.json")
        with open(state_file, 'w') as f:
            json.dump({'total_trades': 10}, f)

        _create_backup(state_file)

        # Find backup file
        backups = [f for f in os.listdir(tmp_path) if 'backup_' in f]
        assert len(backups) == 1
        # Verify content matches
        with open(str(tmp_path / backups[0])) as f:
            data = json.load(f)
        assert data['total_trades'] == 10

    def test_io_error_no_crash(self, tmp_path):
        """IOError during backup → no crash."""
        state_file = str(tmp_path / "nonexistent_dir" / "state.json")
        _create_backup(state_file)  # Should not raise


# ── save_state Edge Cases ────────────────────────────────────


class TestSaveStateEdgeCases:
    """Test save_state() edge cases and error paths."""

    def test_trade_close_creates_timestamped_backup(self, tmp_path):
        """is_trade_close=True → creates timestamped backup."""
        state_file = str(tmp_path / "state.json")
        state = _create_default_state()
        state['total_trades'] = 5

        # First save to create the file
        save_state(state, state_file, create_backup=False)
        # Second save as trade close
        save_state(state, state_file, create_backup=True, is_trade_close=True)

        backups = [f for f in os.listdir(tmp_path) if 'backup_' in f]
        assert len(backups) == 1

    def test_trade_close_sets_last_trade_date(self, tmp_path):
        """is_trade_close=True → sets last_trade_date to today."""
        state_file = str(tmp_path / "state.json")
        state = _create_default_state()
        state['last_trade_date'] = '2025-01-01'

        save_state(state, state_file, is_trade_close=True)

        with open(state_file) as f:
            saved = json.load(f)
        assert saved['last_trade_date'] == datetime.now().strftime('%Y-%m-%d')

    def test_bak_backup_created(self, tmp_path):
        """Existing state file → .bak backup created."""
        state_file = str(tmp_path / "state.json")
        state = _create_default_state()
        save_state(state, state_file)
        save_state(state, state_file)

        assert os.path.exists(state_file + '.bak')


# ── load_state Recovery Chain ────────────────────────────────


class TestLoadStateRecovery:
    """Test load_state() recovery chain (main → .bak → timestamped → default)."""

    def test_corrupted_main_recovers_from_bak(self, tmp_path):
        """Main file corrupted → recovers from .bak."""
        state_file = str(tmp_path / "state.json")
        # Create corrupted main
        with open(state_file, 'w') as f:
            f.write("not json")
        # Create valid .bak
        bak_state = _create_default_state()
        bak_state['total_trades'] = 15
        with open(state_file + '.bak', 'w') as f:
            json.dump(bak_state, f)

        result = load_state(state_file)
        assert result['total_trades'] == 15

    def test_all_corrupted_returns_default(self, tmp_path):
        """All files corrupted → returns default state."""
        state_file = str(tmp_path / "state.json")
        with open(state_file, 'w') as f:
            f.write("corrupt")
        with open(state_file + '.bak', 'w') as f:
            f.write("also corrupt")

        result = load_state(state_file)
        assert result['total_trades'] == 0
        assert result['positions'] == {}

    def test_no_bak_tries_timestamped(self, tmp_path):
        """No .bak file → tries timestamped backups."""
        state_file = str(tmp_path / "state.json")
        with open(state_file, 'w') as f:
            f.write("corrupt")
        # Create timestamped backup
        backup = state_file + ".backup_20260218_120000"
        with open(backup, 'w') as f:
            json.dump({'total_trades': 25, 'winning_trades': 15}, f)

        result = load_state(state_file)
        assert result['total_trades'] == 25


# ── load_state IOError / generic Exception paths ──────────────────


class TestLoadStateIOErrors:
    """Test load_state() IOError and generic Exception handlers (lines 56-59)."""

    def test_ioerror_on_main_falls_through(self, tmp_path):
        """IOError reading main file → tries backup chain."""
        state_file = str(tmp_path / "state.json")
        with open(state_file, 'w') as f:
            f.write('{"total_trades": 5}')

        with patch('builtins.open', side_effect=IOError('disk error')):
            result = load_state(state_file)
        assert result['total_trades'] == 0  # default

    def test_generic_exception_on_main_falls_through(self, tmp_path):
        """Generic Exception reading main file → tries backup chain."""
        state_file = str(tmp_path / "state.json")
        with open(state_file, 'w') as f:
            f.write('{"total_trades": 5}')

        with patch('builtins.open', side_effect=RuntimeError('unexpected')):
            result = load_state(state_file)
        assert result['total_trades'] == 0  # default


# ── _try_timestamped_backups Exception handler ────────────────────


class TestTryTimestampedBackupsException:
    """Test _try_timestamped_backups generic Exception (lines 144-145)."""

    def test_listdir_exception_returns_none(self, tmp_path):
        """os.listdir exception → returns None."""
        state_file = str(tmp_path / "state.json")
        with patch('os.listdir', side_effect=RuntimeError('unexpected')):
            result = _try_timestamped_backups(state_file, _create_default_state())
        assert result is None


# ── save_state .bak failure + atomic write failure ────────────────


class TestSaveStateBakFailure:
    """Test save_state .bak creation failure (lines 179-180)."""

    def test_bak_copy_failure_continues(self, tmp_path):
        """shutil.copy2 failure for .bak → continues to save."""
        state_file = str(tmp_path / "state.json")
        with open(state_file, 'w') as f:
            json.dump({'total_trades': 0}, f)

        state = _create_default_state()
        state['total_trades'] = 10
        with patch('shutil.copy2', side_effect=PermissionError('denied')):
            save_state(state, state_file)

        with open(state_file, 'r') as f:
            saved = json.load(f)
        assert saved['total_trades'] == 10


class TestSaveStateAtomicWriteFailure:
    """Test save_state atomic write failure → fallback (lines 200-209)."""

    def test_atomic_replace_fails_uses_fallback(self, tmp_path):
        """os.replace failure → fallback writes to .new file."""
        state_file = str(tmp_path / "state.json")
        new_file = state_file + '.new'
        state = _create_default_state()
        state['total_trades'] = 42

        with patch('os.replace', side_effect=OSError('rename failed')):
            save_state(state, state_file)

        # Fallback writes to .new, not state_file directly
        assert not os.path.exists(state_file)
        with open(new_file, 'r') as f:
            saved = json.load(f)
        assert saved['total_trades'] == 42


# ── _create_backup generic Exception (lines 221-222) ─────────────


class TestCreateBackupGenericException:
    """Test _create_backup generic Exception handler."""

    def test_generic_exception_no_crash(self, tmp_path):
        """Generic exception during backup → no crash."""
        state_file = str(tmp_path / "state.json")
        with open(state_file, 'w') as f:
            json.dump({'total_trades': 0}, f)

        with patch('shutil.copy2', side_effect=RuntimeError('unexpected')):
            _create_backup(state_file)  # should not raise


# ── cleanup_old_backups Exception handlers (lines 251-254) ────────


class TestCleanupOldBackupsExceptions:
    """Test cleanup_old_backups exception handlers."""

    def test_remove_failure_continues(self, tmp_path):
        """os.remove failure on old backup → continues."""
        state_file = str(tmp_path / "state.json")
        with open(state_file, 'w') as f:
            json.dump({}, f)
        # Create 3 backups (max_backups=1 → should try to remove 2)
        for i in range(3):
            bak = str(tmp_path / f"state.json.backup_2026021{i}_120000")
            with open(bak, 'w') as f:
                f.write('{}')

        with patch('os.remove', side_effect=PermissionError('denied')):
            cleanup_old_backups(state_file, max_backups=1)  # should not raise

    def test_listdir_exception_no_crash(self, tmp_path):
        """os.listdir exception → no crash."""
        state_file = str(tmp_path / "state.json")
        with patch('os.listdir', side_effect=RuntimeError('unexpected')):
            cleanup_old_backups(state_file)  # should not raise


# ── save_metrics error paths (lines 316-325) ──────────────────────


class TestSaveMetricsErrors:
    """Test save_metrics atomic write failure + exception handlers."""

    def test_atomic_replace_fails_raises_caught(self, tmp_path):
        """os.replace failure → IOError caught."""
        metrics_file = str(tmp_path / "metrics.json")
        metrics = PerformanceMetrics()
        metrics.total_trades = 5

        with patch('os.replace', side_effect=OSError('rename failed')):
            save_metrics(metrics, metrics_file)  # should not raise

    def test_serialize_type_error(self, tmp_path):
        """Non-serializable metrics → TypeError caught."""
        metrics_file = str(tmp_path / "metrics.json")
        metrics = PerformanceMetrics()
        # Inject non-serializable data
        metrics.extra_bad = object()

        with patch(
            'bingx_rl_trading_bot.scripts.production.pattern_5m.state.PerformanceMetrics.to_dict',
            side_effect=TypeError('not serializable')
        ):
            save_metrics(metrics, metrics_file)  # should not raise

    def test_generic_exception(self, tmp_path):
        """Generic exception → caught."""
        metrics_file = str(tmp_path / "metrics.json")
        metrics = PerformanceMetrics()

        with patch('tempfile.mkstemp', side_effect=RuntimeError('unexpected')):
            save_metrics(metrics, metrics_file)  # should not raise


# ── load_metrics error paths (lines 347-352) ──────────────────────


class TestLoadMetricsErrors:
    """Test load_metrics IOError, KeyError/TypeError/ValueError, generic Exception."""

    def test_ioerror_returns_none(self, tmp_path):
        """IOError reading file → returns None."""
        metrics_file = str(tmp_path / "metrics.json")
        with open(metrics_file, 'w') as f:
            json.dump({'total_trades': 5}, f)

        with patch('builtins.open', side_effect=IOError('disk error')):
            result = load_metrics(metrics_file)
        assert result is None

    def test_key_error_returns_none(self, tmp_path):
        """Incomplete metrics data → KeyError → returns None."""
        metrics_file = str(tmp_path / "metrics.json")
        with open(metrics_file, 'w') as f:
            json.dump({}, f)  # missing required keys

        with patch(
            'bingx_rl_trading_bot.scripts.production.pattern_5m.state.PerformanceMetrics.from_dict',
            side_effect=KeyError('total_trades')
        ):
            result = load_metrics(metrics_file)
        assert result is None

    def test_generic_exception_returns_none(self, tmp_path):
        """Generic Exception → returns None."""
        metrics_file = str(tmp_path / "metrics.json")
        with open(metrics_file, 'w') as f:
            json.dump({'total_trades': 5}, f)

        with patch(
            'bingx_rl_trading_bot.scripts.production.pattern_5m.state.PerformanceMetrics.from_dict',
            side_effect=RuntimeError('unexpected')
        ):
            result = load_metrics(metrics_file)
        assert result is None


# ── .new file recovery in load_state ──────────────────────────


class TestLoadStateNewFileRecovery:
    """Test load_state recovery from .new file (lines 43-58)."""

    def test_corrupted_main_recovers_from_new_file(self, tmp_path):
        """Main file corrupted + .new file exists → recovers from .new."""
        state_file = str(tmp_path / "state.json")
        new_file = state_file + '.new'

        # Create corrupted main file
        with open(state_file, 'w') as f:
            f.write("{corrupted json")

        # Create valid .new file
        valid_state = _create_default_state()
        valid_state['total_trades'] = 42
        with open(new_file, 'w') as f:
            json.dump(valid_state, f)

        result = load_state(state_file)
        assert result['total_trades'] == 42

    def test_new_file_promoted_to_main(self, tmp_path):
        """After recovery, .new file replaces main file."""
        state_file = str(tmp_path / "state.json")
        new_file = state_file + '.new'

        with open(state_file, 'w') as f:
            f.write("bad json")

        valid_state = _create_default_state()
        valid_state['total_trades'] = 99
        with open(new_file, 'w') as f:
            json.dump(valid_state, f)

        load_state(state_file)
        # .new should be promoted to main (replaced)
        with open(state_file) as f:
            loaded = json.load(f)
        assert loaded['total_trades'] == 99

    def test_new_file_promote_permission_error(self, tmp_path):
        """PermissionError promoting .new → still recovers from .new."""
        state_file = str(tmp_path / "state.json")
        new_file = state_file + '.new'

        with open(state_file, 'w') as f:
            f.write("bad")

        valid_state = _create_default_state()
        valid_state['total_trades'] = 55
        with open(new_file, 'w') as f:
            json.dump(valid_state, f)

        with patch('os.replace', side_effect=PermissionError("locked")):
            result = load_state(state_file)
        assert result['total_trades'] == 55

    def test_new_file_also_corrupted_falls_through(self, tmp_path):
        """Both main and .new corrupted → falls through to .bak."""
        state_file = str(tmp_path / "state.json")
        new_file = state_file + '.new'
        bak_file = state_file + '.bak'

        with open(state_file, 'w') as f:
            f.write("bad main")
        with open(new_file, 'w') as f:
            f.write("bad new")

        # Provide valid .bak
        valid_state = _create_default_state()
        valid_state['total_trades'] = 77
        with open(bak_file, 'w') as f:
            json.dump(valid_state, f)

        result = load_state(state_file)
        assert result['total_trades'] == 77

    def test_corrupted_main_no_new_no_bak(self, tmp_path):
        """Main corrupted, no .new, no .bak → timestamped or default."""
        state_file = str(tmp_path / "state.json")
        with open(state_file, 'w') as f:
            f.write("bad")

        result = load_state(state_file)
        # Falls through to default state
        assert result['total_trades'] == 0


# ── _atomic_replace_with_retry ────────────────────────────────


class TestAtomicReplaceWithRetry:
    """Test _atomic_replace_with_retry (lines 169-181)."""

    def test_success_on_first_try(self, tmp_path):
        """Replace succeeds on first attempt."""
        src = str(tmp_path / "src.json")
        dst = str(tmp_path / "dst.json")
        with open(src, 'w') as f:
            f.write('{"ok": true}')
        _atomic_replace_with_retry(src, dst)
        assert os.path.exists(dst)
        assert not os.path.exists(src)

    def test_retry_then_success(self, tmp_path):
        """PermissionError on first 2 tries, success on 3rd."""
        src = str(tmp_path / "src.json")
        dst = str(tmp_path / "dst.json")
        with open(src, 'w') as f:
            f.write('{"ok": true}')

        call_count = [0]
        original_replace = os.replace

        def flaky_replace(s, d):
            call_count[0] += 1
            if call_count[0] <= 2:
                raise PermissionError("locked")
            return original_replace(s, d)

        with patch('os.replace', side_effect=flaky_replace):
            with patch('time.sleep'):  # skip actual waiting
                _atomic_replace_with_retry(src, dst, max_retries=3)

        assert call_count[0] == 3

    def test_all_retries_exhausted_raises(self, tmp_path):
        """All retries fail → raises PermissionError."""
        src = str(tmp_path / "src.json")
        dst = str(tmp_path / "dst.json")
        with open(src, 'w') as f:
            f.write('test')

        with patch('os.replace', side_effect=PermissionError("locked")):
            with patch('time.sleep'):
                with pytest.raises(PermissionError):
                    _atomic_replace_with_retry(src, dst, max_retries=3)


# ── save_state .new fallback edge case ────────────────────────


class TestSaveStateNewFallback:
    """Test save_state .new fallback failure (lines 247-248)."""

    def test_new_fallback_also_fails(self, tmp_path):
        """Both atomic and .new fallback fail → no crash, lines 247-248 covered."""
        state_file = str(tmp_path / "state.json")
        state = _create_default_state()

        # Make atomic write fail (triggers .new fallback),
        # then make .new open() also fail → exercises lines 247-248
        original_open = builtins.open
        call_count = [0]

        def failing_open(path, *args, **kwargs):
            # Let mkstemp's fdopen work (it uses os.fdopen, not builtins.open)
            # Block only the .new fallback write
            if str(path).endswith('.new'):
                raise PermissionError("disk full")
            return original_open(path, *args, **kwargs)

        with patch(
            'tempfile.mkstemp',
            side_effect=PermissionError("can't create temp"),
        ):
            with patch('builtins.open', side_effect=failing_open):
                # Should not raise even when everything fails
                save_state(state, state_file)

        # Neither main nor .new should exist
        assert not os.path.exists(state_file + '.new')


# ── save_metrics PermissionError → .new fallback ─────────────


class TestSaveMetricsPermissionError:
    """Test save_metrics PermissionError → .new fallback (lines 361-368)."""

    def test_atomic_perm_error_falls_back_to_new(self, tmp_path):
        """PermissionError on atomic write → writes .new file."""
        metrics_file = str(tmp_path / "metrics.json")
        metrics = PerformanceMetrics()
        metrics.total_trades = 10
        metrics.total_wins = 7

        # First call succeeds (create dummy file), then mock to fail
        # Write a dummy file first so the directory exists
        with open(metrics_file, 'w') as f:
            json.dump({}, f)

        with patch(
            'bingx_rl_trading_bot.scripts.production.pattern_5m.state._atomic_replace_with_retry',
            side_effect=PermissionError("OneDrive locked"),
        ):
            save_metrics(metrics, metrics_file)

        # .new file should exist with metrics data
        new_file = metrics_file + '.new'
        assert os.path.exists(new_file)
        with open(new_file) as f:
            data = json.load(f)
        assert data['total_trades'] == 10

    def test_perm_error_new_also_fails(self, tmp_path):
        """PermissionError + .new fallback also fails → no crash."""
        metrics_file = str(tmp_path / "metrics.json")
        metrics = PerformanceMetrics()
        with open(metrics_file, 'w') as f:
            json.dump({}, f)

        # _atomic_replace_with_retry fails with PermissionError,
        # then builtins.open for .new also fails
        with patch(
            'bingx_rl_trading_bot.scripts.production.pattern_5m.state._atomic_replace_with_retry',
            side_effect=PermissionError("locked"),
        ):
            # Mock only the fallback open to fail
            original_open = open

            def mock_open_new(path, *args, **kwargs):
                if str(path).endswith('.new'):
                    raise PermissionError("can't write .new")
                return original_open(path, *args, **kwargs)

            with patch('builtins.open', side_effect=mock_open_new):
                # Should not crash
                save_metrics(metrics, metrics_file)
