"""
Pattern 5m Bot - Utilities Package
"""
import re

from .lock import FileLock, acquire_lock, release_lock
from .logging_config import setup_logging, JSONFormatter, SignalConditionFilter

_PATTERN_RE = re.compile(r'Pattern:\s*(\S+)')


def extract_pattern_name(reason: str) -> str:
    """Extract pattern name from position reason string, e.g. 'Pattern: BD-BD-U (LONG)' -> 'BD-BD-U'."""
    m = _PATTERN_RE.search(reason)
    return m.group(1) if m else ''


__all__ = [
    'FileLock', 'acquire_lock', 'release_lock',
    'setup_logging', 'JSONFormatter', 'SignalConditionFilter',
    'extract_pattern_name',
]
