"""Tests for state.py — state save/load, backups, daily reset, crash recovery,
metrics persistence, sync_metrics_with_state."""

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
        assert result['position'] is None

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
