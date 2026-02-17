"""
Pattern 5m Bot - Logging Configuration
Enhanced logging setup with JSON format support and debug mode filtering.
"""

import os
import glob
import json
import logging
from datetime import datetime
from typing import Optional

from ..constants import BOT_NAME, LOG_DIR

# Log retention: delete files older than this many days
LOG_RETENTION_DAYS = 7


class JSONFormatter(logging.Formatter):
    """JSON structured logging formatter."""

    def format(self, record: logging.LogRecord) -> str:
        log_data = {
            'timestamp': datetime.now().isoformat(),
            'level': record.levelname,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno,
        }
        if hasattr(record, 'extra_data'):
            log_data['data'] = record.extra_data
        return json.dumps(log_data)


class FlushingFileHandler(logging.FileHandler):
    """FileHandler that flushes WARNING+ immediately, buffers INFO/DEBUG every 5s."""

    def __init__(self, *args, flush_interval: float = 5.0, **kwargs):
        super().__init__(*args, **kwargs)
        import time as _time
        self._last_flush = _time.time()
        self._flush_interval = flush_interval
        self._time = _time

    def emit(self, record: logging.LogRecord) -> None:
        super().emit(record)
        now = self._time.time()
        if record.levelno >= logging.WARNING or (now - self._last_flush) >= self._flush_interval:
            self.flush()
            self._last_flush = now


class SignalConditionFilter(logging.Filter):
    """Filter for detailed signal condition logging."""

    def __init__(self, debug_mode: bool = False):
        super().__init__()
        self.debug_mode = debug_mode

    def filter(self, record: logging.LogRecord) -> bool:
        # Always allow non-debug messages
        if record.levelno >= logging.INFO:
            return True
        # Only allow debug messages if debug_mode is enabled
        return self.debug_mode


def setup_logging(
    debug_mode: bool = False,
    json_format: bool = False,
    log_dir: Optional[str] = None,
    bot_name: Optional[str] = None
) -> logging.Logger:
    """
    Enhanced logging setup with debug mode and JSON format support.

    Args:
        debug_mode: Enable debug level logging
        json_format: Use JSON structured logging format
        log_dir: Custom log directory (default: LOG_DIR constant)
        bot_name: Custom bot name for log file (default: BOT_NAME constant)

    Returns:
        Configured logger instance
    """
    _log_dir = log_dir or LOG_DIR
    _bot_name = bot_name or BOT_NAME

    os.makedirs(_log_dir, exist_ok=True)

    # Create logger
    logger = logging.getLogger('pattern_5m')
    logger.setLevel(logging.DEBUG if debug_mode else logging.INFO)
    logger.handlers.clear()

    # File handler (always detailed, with immediate flush for real-time logging)
    log_file = os.path.join(_log_dir, f"{_bot_name}_{datetime.now().strftime('%Y%m%d')}.log")
    file_handler = FlushingFileHandler(log_file, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG if debug_mode else logging.INFO)

    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG if debug_mode else logging.INFO)

    # Set formatters
    if json_format:
        formatter = JSONFormatter()
    else:
        formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s')

    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    # Add filter for debug mode
    signal_filter = SignalConditionFilter(debug_mode)
    file_handler.addFilter(signal_filter)
    console_handler.addFilter(signal_filter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    # Clean up old log files
    _cleanup_old_logs(_log_dir, _bot_name, LOG_RETENTION_DAYS)

    return logger


def _cleanup_old_logs(log_dir: str, bot_name: str, retention_days: int) -> None:
    """Remove log files older than retention_days."""
    import time as _time
    cutoff = _time.time() - (retention_days * 86400)
    pattern = os.path.join(log_dir, f"{bot_name}_*.log")
    for filepath in glob.glob(pattern):
        try:
            if os.path.getmtime(filepath) < cutoff:
                os.remove(filepath)
                logging.getLogger('pattern_5m').info(f"Removed old log: {os.path.basename(filepath)}")
        except OSError:
            pass


