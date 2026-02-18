"""Tests for utils/logging_config.py — JSON formatter, flushing handler, setup, cleanup."""

import os
import json
import time
import logging
import pytest
from unittest.mock import patch, MagicMock

from bingx_rl_trading_bot.scripts.production.pattern_5m.utils.logging_config import (
    JSONFormatter,
    FlushingFileHandler,
    SignalConditionFilter,
    setup_logging,
    _cleanup_old_logs,
)


# ── JSONFormatter ─────────────────────────────────────────


class TestJSONFormatter:
    """Test JSONFormatter structured output."""

    def test_basic_format(self):
        """format() produces valid JSON with required fields."""
        fmt = JSONFormatter()
        record = logging.LogRecord(
            name='test', level=logging.INFO, pathname='test.py',
            lineno=10, msg='hello world', args=(), exc_info=None,
        )
        output = fmt.format(record)
        data = json.loads(output)
        assert data['level'] == 'INFO'
        assert data['message'] == 'hello world'
        assert 'timestamp' in data
        assert 'module' in data
        assert 'function' in data
        assert data['line'] == 10

    def test_extra_data_included(self):
        """format() includes extra_data attribute when present."""
        fmt = JSONFormatter()
        record = logging.LogRecord(
            name='test', level=logging.WARNING, pathname='test.py',
            lineno=5, msg='with data', args=(), exc_info=None,
        )
        record.extra_data = {'key': 'value'}
        output = fmt.format(record)
        data = json.loads(output)
        assert data['data'] == {'key': 'value'}

    def test_no_extra_data(self):
        """format() omits 'data' key when no extra_data."""
        fmt = JSONFormatter()
        record = logging.LogRecord(
            name='test', level=logging.DEBUG, pathname='test.py',
            lineno=1, msg='plain', args=(), exc_info=None,
        )
        output = fmt.format(record)
        data = json.loads(output)
        assert 'data' not in data


# ── FlushingFileHandler ───────────────────────────────────


class TestFlushingFileHandler:
    """Test FlushingFileHandler flush behavior."""

    def test_warning_flushes_immediately(self, tmp_path):
        """WARNING+ records trigger immediate flush."""
        logfile = str(tmp_path / 'test.log')
        handler = FlushingFileHandler(logfile, flush_interval=999.0)
        handler.setFormatter(logging.Formatter('%(message)s'))
        record = logging.LogRecord(
            name='test', level=logging.WARNING, pathname='test.py',
            lineno=1, msg='warn msg', args=(), exc_info=None,
        )
        with patch.object(handler, 'flush', wraps=handler.flush) as mock_flush:
            handler.emit(record)
            assert mock_flush.call_count >= 1
        handler.close()

    def test_info_flushes_on_interval(self, tmp_path):
        """INFO records flush when interval elapsed."""
        logfile = str(tmp_path / 'test.log')
        handler = FlushingFileHandler(logfile, flush_interval=0.0)
        handler.setFormatter(logging.Formatter('%(message)s'))
        handler._last_flush = 0  # force interval elapsed
        record = logging.LogRecord(
            name='test', level=logging.INFO, pathname='test.py',
            lineno=1, msg='info msg', args=(), exc_info=None,
        )
        with patch.object(handler, 'flush', wraps=handler.flush) as mock_flush:
            handler.emit(record)
            assert mock_flush.call_count >= 1
        handler.close()

    def test_info_no_flush_within_interval(self, tmp_path):
        """INFO records skip flush within interval."""
        logfile = str(tmp_path / 'test.log')
        handler = FlushingFileHandler(logfile, flush_interval=999.0)
        handler.setFormatter(logging.Formatter('%(message)s'))
        handler._last_flush = time.time()  # just flushed
        record = logging.LogRecord(
            name='test', level=logging.INFO, pathname='test.py',
            lineno=1, msg='info msg', args=(), exc_info=None,
        )
        # Patch flush at the handler level to track explicit calls
        original_last_flush = handler._last_flush
        handler.emit(record)
        # _last_flush should not have been updated (no flush triggered)
        assert handler._last_flush == original_last_flush


# ── SignalConditionFilter ─────────────────────────────────


class TestSignalConditionFilter:
    """Test SignalConditionFilter debug gating."""

    def test_info_always_passes(self):
        """INFO+ always passes regardless of debug_mode."""
        f = SignalConditionFilter(debug_mode=False)
        record = logging.LogRecord(
            name='test', level=logging.INFO, pathname='', lineno=0,
            msg='info', args=(), exc_info=None,
        )
        assert f.filter(record) is True

    def test_debug_blocked_when_not_debug(self):
        """DEBUG blocked when debug_mode=False."""
        f = SignalConditionFilter(debug_mode=False)
        record = logging.LogRecord(
            name='test', level=logging.DEBUG, pathname='', lineno=0,
            msg='debug', args=(), exc_info=None,
        )
        assert f.filter(record) is False

    def test_debug_allowed_when_debug(self):
        """DEBUG allowed when debug_mode=True."""
        f = SignalConditionFilter(debug_mode=True)
        record = logging.LogRecord(
            name='test', level=logging.DEBUG, pathname='', lineno=0,
            msg='debug', args=(), exc_info=None,
        )
        assert f.filter(record) is True


# ── setup_logging ─────────────────────────────────────────


class TestSetupLogging:
    """Test setup_logging() configuration."""

    def test_returns_logger(self, tmp_path):
        """Returns a configured logger."""
        logger = setup_logging(log_dir=str(tmp_path), bot_name='test_bot')
        assert isinstance(logger, logging.Logger)
        assert logger.name == 'pattern_5m'
        assert len(logger.handlers) == 2  # file + console
        logger.handlers.clear()

    def test_debug_mode_sets_debug_level(self, tmp_path):
        """debug_mode=True sets logger level to DEBUG."""
        logger = setup_logging(debug_mode=True, log_dir=str(tmp_path), bot_name='test_bot')
        assert logger.level == logging.DEBUG
        logger.handlers.clear()

    def test_normal_mode_sets_info_level(self, tmp_path):
        """debug_mode=False sets logger level to INFO."""
        logger = setup_logging(debug_mode=False, log_dir=str(tmp_path), bot_name='test_bot')
        assert logger.level == logging.INFO
        logger.handlers.clear()

    def test_json_format(self, tmp_path):
        """json_format=True uses JSONFormatter."""
        logger = setup_logging(json_format=True, log_dir=str(tmp_path), bot_name='test_bot')
        file_handler = logger.handlers[0]
        assert isinstance(file_handler.formatter, JSONFormatter)
        logger.handlers.clear()

    def test_standard_format(self, tmp_path):
        """json_format=False uses standard Formatter."""
        logger = setup_logging(json_format=False, log_dir=str(tmp_path), bot_name='test_bot')
        file_handler = logger.handlers[0]
        assert not isinstance(file_handler.formatter, JSONFormatter)
        logger.handlers.clear()

    def test_creates_log_dir(self, tmp_path):
        """Creates log directory if it doesn't exist."""
        new_dir = str(tmp_path / 'subdir' / 'logs')
        logger = setup_logging(log_dir=new_dir, bot_name='test_bot')
        assert os.path.isdir(new_dir)
        logger.handlers.clear()

    def test_handlers_have_filter(self, tmp_path):
        """Handlers have SignalConditionFilter."""
        logger = setup_logging(log_dir=str(tmp_path), bot_name='test_bot')
        for handler in logger.handlers:
            filters = handler.filters
            assert any(isinstance(f, SignalConditionFilter) for f in filters)
        logger.handlers.clear()

    def test_clears_existing_handlers(self, tmp_path):
        """Clears existing handlers before adding new ones."""
        logger = setup_logging(log_dir=str(tmp_path), bot_name='test_bot')
        assert len(logger.handlers) == 2
        # Call again — should still be 2 (cleared old ones)
        logger = setup_logging(log_dir=str(tmp_path), bot_name='test_bot')
        assert len(logger.handlers) == 2
        logger.handlers.clear()


# ── _cleanup_old_logs ─────────────────────────────────────


class TestCleanupOldLogs:
    """Test _cleanup_old_logs() retention policy."""

    def test_removes_old_logs(self, tmp_path):
        """Removes logs older than retention_days."""
        old_file = tmp_path / 'bot_20250101.log'
        old_file.write_text('old log')
        # Set mtime to 30 days ago
        old_mtime = time.time() - (30 * 86400)
        os.utime(str(old_file), (old_mtime, old_mtime))

        _cleanup_old_logs(str(tmp_path), 'bot', retention_days=7)
        assert not old_file.exists()

    def test_keeps_recent_logs(self, tmp_path):
        """Keeps logs newer than retention_days."""
        recent_file = tmp_path / 'bot_20260218.log'
        recent_file.write_text('recent log')

        _cleanup_old_logs(str(tmp_path), 'bot', retention_days=7)
        assert recent_file.exists()

    def test_oserror_ignored(self, tmp_path):
        """OSError during removal is silently handled."""
        old_file = tmp_path / 'bot_20250101.log'
        old_file.write_text('data')
        old_mtime = time.time() - (30 * 86400)
        os.utime(str(old_file), (old_mtime, old_mtime))

        with patch('bingx_rl_trading_bot.scripts.production.pattern_5m.utils.logging_config.os.remove',
                   side_effect=OSError('perm denied')):
            _cleanup_old_logs(str(tmp_path), 'bot', retention_days=7)
            # Should not raise

    def test_no_matching_files(self, tmp_path):
        """No matching files → no-op."""
        _cleanup_old_logs(str(tmp_path), 'bot', retention_days=7)  # should not raise
