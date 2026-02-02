"""
Engulf 5m Bot - Position Management
Open, close, and manage trading positions.

This module re-exports functions from the split modules for backward compatibility.
See: position_open.py, position_monitor.py, position_close.py
"""

# Re-export from position_open
from .position_open import (
    get_position_size,
    open_position,
)

# Re-export from position_monitor
from .position_monitor import (
    sync_position_with_exchange,
    check_position_status,
    get_actual_exit_price,
)

# Re-export from position_close
from .position_close import (
    record_closed_position,
    recover_position_to_state,
    recover_from_crash,
)

__all__ = [
    # position_open
    'get_position_size',
    'open_position',
    # position_monitor
    'sync_position_with_exchange',
    'check_position_status',
    'get_actual_exit_price',
    # position_close
    'record_closed_position',
    'recover_position_to_state',
    'recover_from_crash',
]
