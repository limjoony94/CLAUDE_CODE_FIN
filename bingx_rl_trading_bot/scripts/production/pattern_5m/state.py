"""
Pattern 5m Bot - State Management
Load, save, and manage bot state with backup functionality.
"""

import os
import json
import logging
import shutil
import tempfile
import time
from datetime import datetime
from typing import Dict, Any, Optional

from .constants import STATE_FILE, METRICS_FILE, MAX_STATE_BACKUPS
from .models import PerformanceMetrics, BOT_STATE_REQUIRED_KEYS

logger = logging.getLogger('pattern_5m')


def load_state(state_file: str = STATE_FILE) -> Dict[str, Any]:
    """
    Load bot state from JSON file with .bak recovery on corruption.

    Args:
        state_file: Path to state JSON file

    Returns:
        State dictionary (new or loaded)
    """
    default_state = _create_default_state()

    if os.path.exists(state_file):
        try:
            with open(state_file, 'r') as f:
                state = json.load(f)
                state = _migrate_state_v2(state)
                state = _ensure_required_keys(state, default_state)
                state = _check_daily_reset(state)
                return state
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse state JSON: {e}")

            # Try .new file first (written by save_state fallback)
            new_file = state_file + '.new'
            if os.path.exists(new_file):
                try:
                    with open(new_file, 'r') as f:
                        state = json.load(f)
                    logger.warning(f"Recovered state from .new file")
                    try:
                        os.replace(new_file, state_file)
                    except PermissionError:
                        logger.debug("Could not promote .new to main (PermissionError)")
                    state = _migrate_state_v2(state)
                    state = _ensure_required_keys(state, default_state)
                    state = _check_daily_reset(state)
                    return state
                except Exception:
                    logger.debug("Failed to recover from .new file")

            # Try to recover from .bak file
            bak_file = state_file + '.bak'
            if os.path.exists(bak_file):
                logger.warning(f"⚠️ Attempting recovery from {bak_file}")
                try:
                    with open(bak_file, 'r') as f:
                        state = json.load(f)
                        logger.info(f"Successfully recovered state from backup")
                        state = _migrate_state_v2(state)
                        state = _ensure_required_keys(state, default_state)
                        state = _check_daily_reset(state)
                        return state
                except Exception as bak_error:
                    logger.error(f"❌ Backup recovery failed: {bak_error}")
            else:
                logger.warning(f"⚠️ No backup file found at {bak_file}")
        except (IOError, OSError) as e:
            logger.error(f"Failed to read state file: {e}")
        except Exception as e:
            logger.exception(f"Failed to load state: {e}")

    # Try timestamped backups as last resort
    backup_state = _try_timestamped_backups(state_file, default_state)
    if backup_state is not None:
        return backup_state

    logger.warning(f"⚠️ Returning default state (no valid state file found)")
    return default_state


def _create_default_state() -> Dict[str, Any]:
    """Create a new default state dictionary (v3: hedge-capable)."""
    return {
        'positions': {},  # v1.29.0: {slot_id: position_dict}
        'active_direction': None,  # One-Way mode: None / 'LONG' / 'SHORT'
        'emergency_sl_order_id': None,  # v1.29.0: One-Way emergency SL (kept for compat)
        'emergency_sl_orders': {},  # v1.30.0: Hedge per-direction {'LONG': id, 'SHORT': id}
        'state_version': 3,
        'last_signal_time': None,
        'last_signal_candle_timestamp': None,
        'daily_pnl': 0.0,
        'daily_trades': 0,
        'consecutive_losses': 0,
        'total_trades': 0,
        'total_pnl': 0.0,
        'winning_trades': 0,
        'last_trade': None,
        'last_trade_date': datetime.now().strftime('%Y-%m-%d'),
        'created_at': datetime.now().isoformat(),
        'updated_at': datetime.now().isoformat(),
        'peak_equity': 0.0,  # v1.34.0: MDD sizing high watermark
    }


def _migrate_state_v2(state: Dict[str, Any]) -> Dict[str, Any]:
    """Migrate to v2 dict format. Handles:
      - v1 (single position): state['position'] → dict
      - v2-list (partial impl): state['positions'] = list → dict
      - v2-dict (current): no-op

    v1 format: state['position'] = dict | None
    v2 format: state['positions'] = {slot_id: dict, ...}
    """
    # v1 → v2: single position to dict
    if 'position' in state and 'positions' not in state:
        old_pos = state.pop('position')
        if old_pos:
            slot_id = old_pos.get('slot_id', _generate_slot_id())
            old_pos['slot_id'] = slot_id
            state['positions'] = {slot_id: old_pos}
            state['active_direction'] = old_pos.get('direction')
        else:
            state['positions'] = {}
            state['active_direction'] = None
        state['emergency_sl_order_id'] = state.pop('sl_order_id', None)
        state['state_version'] = 2
        logger.info(f"State migrated v1→v2: {len(state['positions'])} active positions")

    # v2-list → v2-dict: convert list format to dict format
    elif 'positions' in state and isinstance(state['positions'], list):
        old_list = state['positions']
        new_dict = {}
        for slot in old_list:
            if isinstance(slot, dict):
                slot_id = slot.get('slot_id', _generate_slot_id())
                slot['slot_id'] = slot_id
                new_dict[slot_id] = slot
        state['positions'] = new_dict
        if new_dict:
            first_slot = next(iter(new_dict.values()))
            state.setdefault('active_direction', first_slot.get('direction'))
        else:
            state.setdefault('active_direction', None)
        state.setdefault('emergency_sl_order_id', None)
        state['state_version'] = 2
        logger.info(f"State migrated v2-list→v2-dict: {len(new_dict)} active positions")

    # Ensure v2 fields exist even on v2-dict states
    state.setdefault('active_direction', None)
    state.setdefault('emergency_sl_order_id', None)

    # v2→v3: add hedge emergency SL dict
    state.setdefault('emergency_sl_orders', {})
    if state.get('state_version', 2) < 3:
        state['state_version'] = 3
        logger.debug("State upgraded to v3 (hedge emergency_sl_orders)")

    # Keep has_position in sync
    state['has_position'] = len(state.get('positions') or {}) > 0
    return state


def _generate_slot_id() -> str:
    """Generate a short unique slot ID."""
    import uuid
    return uuid.uuid4().hex[:8]


def _downgrade_state_v1(state: Dict[str, Any]) -> Dict[str, Any]:
    """Downgrade v2 (multi position dict) → v1 (single position) for rollback.

    Takes the first active position (if any) and puts it back as scalar.
    """
    if 'positions' in state and 'position' not in state:
        positions = state.pop('positions')
        if isinstance(positions, dict):
            values = list(positions.values())
            state['position'] = values[0] if values else None
        elif isinstance(positions, list):
            state['position'] = positions[0] if positions else None
        else:
            state['position'] = None
        state.pop('state_version', None)
        state.pop('active_direction', None)
        state.pop('emergency_sl_order_id', None)
        logger.info(f"State downgraded v2→v1: position={'active' if state['position'] else 'None'}")
    return state


def _ensure_required_keys(state: Dict[str, Any], defaults: Dict[str, Any]) -> Dict[str, Any]:
    """Fill missing required keys from defaults. Logs a warning for each missing key."""
    missing = BOT_STATE_REQUIRED_KEYS - state.keys()
    if missing:
        logger.warning(f"State missing required keys, filling defaults: {missing}")
        for key in missing:
            state[key] = defaults[key]
    return state


def update_peak_equity(state: Dict[str, Any], current_equity: float) -> Dict[str, Any]:
    """Update peak equity high watermark for MDD sizing (v1.34.0)."""
    if current_equity > state.get('peak_equity', 0):
        state['peak_equity'] = current_equity
    return state


def _check_daily_reset(state: Dict[str, Any]) -> Dict[str, Any]:
    """Check if daily stats should be reset (new trading day). Used at load time (no save)."""
    today = datetime.now().strftime('%Y-%m-%d')
    last_date = state.get('last_trade_date', '')

    if last_date and last_date != today:
        logger.info(f"New day detected ({last_date} -> {today}), resetting daily stats")
        _reset_daily_fields(state, today)

    return state


def _try_timestamped_backups(state_file: str, default_state: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """Try to recover state from timestamped backup files (newest first)."""
    try:
        state_dir = os.path.dirname(state_file)
        state_name = os.path.basename(state_file)
        backup_pattern = f"{state_name}.backup_"

        backups = []
        for filename in os.listdir(state_dir):
            if filename.startswith(backup_pattern):
                filepath = os.path.join(state_dir, filename)
                backups.append((filepath, os.path.getmtime(filepath)))

        if not backups:
            return None

        # Sort newest first
        backups.sort(key=lambda x: x[1], reverse=True)
        logger.warning(f"⚠️ Trying {len(backups)} timestamped backups for recovery...")

        for filepath, _ in backups:
            try:
                with open(filepath, 'r') as f:
                    state = json.load(f)
                # Validate: must have total_trades key
                if 'total_trades' not in state:
                    continue
                logger.info(f"✅ Recovered state from backup: {os.path.basename(filepath)}")
                state = _migrate_state_v2(state)
                state = _ensure_required_keys(state, default_state)
                state = _check_daily_reset(state)
                return state
            except (json.JSONDecodeError, IOError, OSError):
                continue
    except Exception as e:
        logger.debug(f"Failed to search timestamped backups: {e}")

    return None


def _atomic_replace_with_retry(src: str, dst: str, max_retries: int = 3) -> None:
    """os.replace() with exponential backoff retry for OneDrive file locking."""
    for attempt in range(max_retries):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if attempt < max_retries - 1:
                wait = 0.1 * (2 ** attempt)  # 0.1s, 0.2s, 0.4s
                logger.debug(f"os.replace retry {attempt+1}/{max_retries}, waiting {wait:.1f}s")
                time.sleep(wait)
            else:
                raise


def save_state(
    state: Dict[str, Any],
    state_file: str = STATE_FILE,
    create_backup: bool = True,
    is_trade_close: bool = False
) -> None:
    """
    Save bot state to JSON file with .bak backup and atomic write.

    Args:
        state: State dictionary to save
        state_file: Path to state JSON file
        create_backup: Whether to create a backup before saving
        is_trade_close: Whether this save is due to a trade closing
    """
    state['updated_at'] = datetime.now().isoformat()

    # Only update last_trade_date when a trade actually closes
    if is_trade_close:
        state['last_trade_date'] = datetime.now().strftime('%Y-%m-%d')

    state_dir = os.path.dirname(state_file)
    if state_dir:
        os.makedirs(state_dir, exist_ok=True)

    # 1. Create .bak backup of existing state
    if os.path.exists(state_file):
        try:
            shutil.copy2(state_file, state_file + '.bak')
        except Exception as e:
            logger.warning(f"Failed to create .bak backup: {e}")

    # 2. Create timestamped backup only on trade close (avoid excessive I/O)
    if create_backup and is_trade_close and os.path.exists(state_file):
        _create_backup(state_file)

    # 3. Atomic write: write to temp file, then rename
    try:
        # Write to temporary file in same directory (for atomic rename on same filesystem)
        tmp_dir = os.path.dirname(state_file) or '.'
        fd, tmp_path = tempfile.mkstemp(
            dir=tmp_dir,
            prefix='.tmp_state_',
            suffix='.json'
        )
        try:
            with os.fdopen(fd, 'w') as f:
                json.dump(state, f, indent=2, default=str)
            # fd is now closed by os.fdopen — do NOT close again
            _atomic_replace_with_retry(tmp_path, state_file)
        except Exception:
            # fd already closed by os.fdopen's with block
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise
    except Exception as e:
        logger.exception(f"Failed to save state atomically: {e}")
        # Fallback: write to .new file (load_state can recover from it)
        new_file = state_file + '.new'
        try:
            with open(new_file, 'w') as f:
                json.dump(state, f, indent=2, default=str)
            logger.warning(f"State saved to {new_file} (recovery pending)")
        except Exception as e2:
            logger.error(f"Failed to save state even to .new: {e2}")


def _create_backup(state_file: str) -> None:
    """Create a timestamped backup of the state file."""
    try:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_file = f"{state_file}.backup_{timestamp}"
        shutil.copy2(state_file, backup_file)
        cleanup_old_backups(state_file)
    except (IOError, OSError) as e:
        logger.warning(f"Backup failed (I/O error): {e}")
    except Exception as e:
        logger.warning(f"Backup failed: {e}")


def cleanup_old_backups(state_file: str, max_backups: int = MAX_STATE_BACKUPS) -> None:
    """
    Remove old backup files, keeping only the most recent ones.

    Args:
        state_file: Path to main state file
        max_backups: Maximum number of backups to keep
    """
    try:
        state_dir = os.path.dirname(state_file)
        state_name = os.path.basename(state_file)
        backup_pattern = f"{state_name}.backup_"

        backups = []
        for filename in os.listdir(state_dir):
            if filename.startswith(backup_pattern):
                filepath = os.path.join(state_dir, filename)
                backups.append((filepath, os.path.getmtime(filepath)))

        # Sort by modification time (newest first)
        backups.sort(key=lambda x: x[1], reverse=True)

        # Remove old backups
        for filepath, _ in backups[max_backups:]:
            try:
                os.remove(filepath)
            except Exception as e:
                logger.debug(f"Failed to remove old backup {filepath}: {e}")
    except Exception as e:
        logger.debug(f"Failed to cleanup old backups: {e}")


def _reset_daily_fields(state: Dict[str, Any], today: str) -> None:
    """Reset daily stat fields (shared by load-time and runtime reset)."""
    state['daily_pnl'] = 0.0
    state['daily_trades'] = 0
    state['consecutive_losses'] = 0
    state['last_trade_date'] = today


def reset_daily_stats_if_needed(state: Dict[str, Any]) -> bool:
    """
    Reset daily statistics if it's a new trading day.

    Args:
        state: State dictionary to check/update

    Returns:
        True if stats were reset, False otherwise
    """
    today = datetime.now().strftime('%Y-%m-%d')
    last_date = state.get('last_trade_date', '')

    if last_date != today:
        if last_date:
            logger.info(f"📅 New day detected ({last_date} → {today}), resetting daily stats")
        _reset_daily_fields(state, today)
        save_state(state)
        return True

    return False


# ============================================================
# METRICS PERSISTENCE
# ============================================================

def save_metrics(metrics: PerformanceMetrics, metrics_file: str = METRICS_FILE) -> None:
    """
    Save performance metrics to file.

    Args:
        metrics: PerformanceMetrics instance to save
        metrics_file: Path to metrics JSON file
    """
    try:
        metrics_data = metrics.to_dict()
        metrics_dir = os.path.dirname(metrics_file)
        if metrics_dir:
            os.makedirs(metrics_dir, exist_ok=True)
        # Atomic write: write to temp file, then rename
        tmp_dir = os.path.dirname(metrics_file) or '.'
        fd, tmp_path = tempfile.mkstemp(
            dir=tmp_dir,
            prefix='.tmp_metrics_',
            suffix='.json'
        )
        try:
            with os.fdopen(fd, 'w') as f:
                json.dump(metrics_data, f, indent=2)
            _atomic_replace_with_retry(tmp_path, metrics_file)
        except Exception:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise
    except PermissionError as e:
        # Fallback: write to .new file for OneDrive locking
        logger.warning(f"Failed to save metrics atomically (PermissionError): {e}")
        new_file = metrics_file + '.new'
        try:
            with open(new_file, 'w') as f:
                json.dump(metrics_data, f, indent=2)
            logger.warning(f"Metrics saved to {new_file} (recovery pending)")
        except Exception as e2:
            logger.error(f"Failed to save metrics even to .new: {e2}")
    except (IOError, OSError) as e:
        logger.warning(f"Failed to save metrics (I/O error): {e}")
    except (TypeError, ValueError) as e:
        logger.warning(f"Failed to serialize metrics: {e}")
    except Exception as e:
        logger.warning(f"Failed to save metrics: {e}")


def load_metrics(metrics_file: str = METRICS_FILE) -> Optional[PerformanceMetrics]:
    """
    Load performance metrics from file.

    Args:
        metrics_file: Path to metrics JSON file

    Returns:
        PerformanceMetrics instance or None if loading fails
    """
    try:
        if os.path.exists(metrics_file):
            with open(metrics_file, 'r') as f:
                data = json.load(f)
            metrics = PerformanceMetrics.from_dict(data)
            logger.info(f"Loaded metrics: {metrics.total_trades} trades, {metrics.actual_win_rate:.1f}% WR")
            return metrics
    except json.JSONDecodeError as e:
        logger.warning(f"Failed to parse metrics JSON: {e}")
    except (IOError, OSError) as e:
        logger.warning(f"Failed to read metrics file: {e}")
    except (KeyError, TypeError, ValueError) as e:
        logger.warning(f"Failed to deserialize metrics: {e}")
    except Exception as e:
        logger.warning(f"Failed to load metrics: {e}")

    return None


def sync_metrics_with_state(metrics: PerformanceMetrics, state: Dict[str, Any]) -> PerformanceMetrics:
    """
    Synchronize metrics with state if they're out of sync.

    When state has MORE trades than metrics, trust state (normal: state updated, metrics lagged).
    When state has FEWER trades than metrics, trust metrics (state likely corrupted by crash).

    Args:
        metrics: PerformanceMetrics instance
        state: State dictionary

    Returns:
        Synchronized PerformanceMetrics instance
    """
    state_trades = state.get('total_trades', 0)

    if state_trades == metrics.total_trades:
        return metrics

    logger.warning(f"⚠️ Metrics out of sync: state={state_trades}, metrics={metrics.total_trades}")

    if state_trades >= metrics.total_trades:
        # State has more trades → trust state (normal case)
        metrics.total_trades = state_trades
        metrics.winning_trades = state.get('winning_trades', 0)
        metrics.losing_trades = metrics.total_trades - metrics.winning_trades

        if metrics.total_trades > 0:
            metrics.actual_win_rate = (metrics.winning_trades / metrics.total_trades) * 100

        metrics.total_pnl_pct = state.get('total_pnl', 0.0)
        metrics._recalculate()

        logger.info(f"✅ Metrics synced with state: {metrics.total_trades} trades, {metrics.actual_win_rate:.1f}% WR")
    else:
        # State has FEWER trades → state likely corrupted!
        logger.critical(
            f"🚨 State corruption detected: state={state_trades} < metrics={metrics.total_trades}. "
            f"Trusting metrics and updating state."
        )
        # Update state to match metrics (prevent data loss)
        state['total_trades'] = metrics.total_trades
        state['winning_trades'] = metrics.winning_trades
        state['total_pnl'] = metrics.total_pnl_pct
        save_state(state)

        # Also update .bak with corrected data (save_state's _create_backup
        # copies the OLD corrupted main → .bak BEFORE writing, so .bak still has bad data)
        try:
            shutil.copy2(STATE_FILE, STATE_FILE + '.bak')
        except Exception:
            logger.debug("Could not update .bak after state restoration")

        logger.info(
            f"✅ State restored from metrics: {metrics.total_trades} trades, {metrics.actual_win_rate:.1f}% WR"
        )

    return metrics
