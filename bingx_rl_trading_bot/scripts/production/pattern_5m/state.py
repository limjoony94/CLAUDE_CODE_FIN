"""
Pattern 5m Bot - State Management
Load, save, and manage bot state with backup functionality.
"""

import os
import json
import logging
import shutil
import tempfile
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
                state = _ensure_required_keys(state, default_state)
                state = _check_daily_reset(state)
                return state
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse state JSON: {e}")
            # Try to recover from .bak file
            bak_file = state_file + '.bak'
            if os.path.exists(bak_file):
                logger.warning(f"⚠️ Attempting recovery from {bak_file}")
                try:
                    with open(bak_file, 'r') as f:
                        state = json.load(f)
                        logger.info(f"Successfully recovered state from backup")
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
    """Create a new default state dictionary."""
    return {
        'position': None,
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
    }


def _ensure_required_keys(state: Dict[str, Any], defaults: Dict[str, Any]) -> Dict[str, Any]:
    """Fill missing required keys from defaults. Logs a warning for each missing key."""
    missing = BOT_STATE_REQUIRED_KEYS - state.keys()
    if missing:
        logger.warning(f"State missing required keys, filling defaults: {missing}")
        for key in missing:
            state[key] = defaults[key]
    return state


def _check_daily_reset(state: Dict[str, Any]) -> Dict[str, Any]:
    """Check if daily stats should be reset (new trading day)."""
    today = datetime.now().strftime('%Y-%m-%d')
    last_date = state.get('last_trade_date', '')

    if last_date and last_date != today:
        logger.info(f"New day detected ({last_date} -> {today}), resetting daily stats")
        state['daily_pnl'] = 0.0
        state['daily_trades'] = 0
        state['consecutive_losses'] = 0
        state['last_trade_date'] = today

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
                state = _ensure_required_keys(state, default_state)
                state = _check_daily_reset(state)
                return state
            except (json.JSONDecodeError, IOError, OSError):
                continue
    except Exception:
        pass

    return None


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
            os.replace(tmp_path, state_file)
        except Exception:
            # fd already closed by os.fdopen's with block
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise
    except Exception as e:
        logger.exception(f"Failed to save state atomically: {e}")
        # Fallback to non-atomic write
        with open(state_file, 'w') as f:
            json.dump(state, f, indent=2, default=str)


def _create_backup(state_file: str) -> None:
    """Create a timestamped backup of the state file."""
    try:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        backup_file = f"{state_file}.backup_{timestamp}"
        with open(state_file, 'r') as f:
            with open(backup_file, 'w') as bf:
                bf.write(f.read())
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
            except Exception:
                pass
    except Exception:
        pass


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
        state['daily_pnl'] = 0.0
        state['daily_trades'] = 0
        state['consecutive_losses'] = 0
        state['last_trade_date'] = today
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
            os.replace(tmp_path, metrics_file)
        except Exception:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise
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

        logger.info(
            f"✅ State restored from metrics: {metrics.total_trades} trades, {metrics.actual_win_rate:.1f}% WR"
        )

    return metrics
