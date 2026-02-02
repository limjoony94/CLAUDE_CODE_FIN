"""
Pattern 5m Bot - Logging Configuration
Enhanced logging setup with JSON format support and debug mode filtering.
"""

import os
import json
import logging
from datetime import datetime
from typing import Optional

from ..constants import BOT_NAME, LOG_DIR


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
    """FileHandler that flushes after every log entry for real-time logging."""

    def emit(self, record: logging.LogRecord) -> None:
        super().emit(record)
        self.flush()


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

    return logger


def log_signal_conditions(logger: logging.Logger, df, config: dict, level: str = 'DEBUG') -> None:
    """
    Log detailed signal conditions for debugging.

    Args:
        logger: Logger instance
        df: DataFrame with OHLCV and indicator data
        config: Bot configuration dictionary
        level: Log level (DEBUG, INFO, etc.)
    """
    current = df.iloc[-2]  # Last complete candle
    strategy = config.get('strategy', {})

    conditions = {
        'candle_time': str(df.iloc[-2]['datetime']),
        'body': float(current['body']),
        'body_pct': float(current.get('body_pct', 0)),
        'prev_body': float(current['prev_body']),
        'volume_ratio': float(current.get('volume_ratio', 0)),
        'prev_body_ratio': float(current.get('prev_body_ratio', 0)),
        'bullish_engulf': bool(current.get('bullish_engulf', False)),
        'bearish_engulf': bool(current.get('bearish_engulf', False)),
    }

    # Check individual conditions
    checks = {
        'is_bullish': current['body'] > 0,
        'is_bearish': current['body'] < 0,
        'prev_is_bearish': current['prev_body'] < 0,
        'prev_is_bullish': current['prev_body'] > 0,
        'close_gt_prev_open': current['close'] > df.iloc[-3]['open'] if len(df) > 2 else False,
        'volume_filter_pass': current.get('volume_ratio', 0) >= strategy.get('min_volume_ratio', 1.0),
        'prev_body_filter_pass': current.get('prev_body_ratio', 0) >= strategy.get('min_prev_body_ratio', 0.3),
        'body_pct_filter_pass': current.get('body_pct', 0) >= strategy.get('min_body_pct', 0),
    }

    log_func = getattr(logger, level.lower(), logger.debug)
    log_func(f"Signal conditions: {json.dumps(conditions)}")
    log_func(f"Condition checks: {json.dumps(checks)}")
