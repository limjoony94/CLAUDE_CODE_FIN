"""
Engulf 5m Bot - File Lock Utilities
Cross-platform file locking for single instance enforcement.
"""

import os
import json
import logging
import platform
import subprocess
from datetime import datetime
from typing import Optional, List
from abc import ABC, abstractmethod

from ..constants import BOT_NAME, BOT_VERSION, LOCK_FILE

logger = logging.getLogger(__name__)

# Global lock handle
_lock_file_handle = None


class FileLock(ABC):
    """Abstract base class for file locking."""

    @abstractmethod
    def acquire(self, filepath: str) -> bool:
        """Acquire the lock. Returns True on success."""
        pass

    @abstractmethod
    def release(self, filepath: str) -> None:
        """Release the lock."""
        pass


class WindowsFileLock(FileLock):
    """Windows-specific file locking using msvcrt."""

    def __init__(self):
        self._handle = None

    def acquire(self, filepath: str) -> bool:
        import msvcrt
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            self._handle = open(filepath, 'w')
            try:
                msvcrt.locking(self._handle.fileno(), msvcrt.LK_NBLCK, 1)
            except IOError:
                self._handle.close()
                self._handle = None
                return False

            self._write_lock_info(filepath)
            return True
        except (IOError, OSError) as e:
            logger.error(f"Failed to acquire lock (I/O error): {e}")
            if self._handle:
                self._handle.close()
                self._handle = None
            return False
        except Exception as e:
            logger.error(f"Failed to acquire lock: {e}")
            if self._handle:
                self._handle.close()
                self._handle = None
            return False

    def release(self, filepath: str) -> None:
        import msvcrt
        if self._handle:
            try:
                msvcrt.locking(self._handle.fileno(), msvcrt.LK_UNLCK, 1)
            except Exception:
                pass
            self._handle.close()
            self._handle = None

        self._cleanup_file(filepath)

    def _write_lock_info(self, filepath: str) -> None:
        if self._handle:
            lock_data = {
                'pid': os.getpid(),
                'timestamp': datetime.now().isoformat(),
                'bot_version': BOT_VERSION,
            }
            self._handle.write(json.dumps(lock_data))
            self._handle.flush()

    def _cleanup_file(self, filepath: str) -> None:
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
        except (IOError, OSError):
            pass
        except Exception:
            pass


class UnixFileLock(FileLock):
    """Unix-specific file locking using fcntl."""

    def __init__(self):
        self._handle = None

    def acquire(self, filepath: str) -> bool:
        import fcntl
        try:
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            self._handle = open(filepath, 'w')
            fcntl.flock(self._handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

            self._write_lock_info(filepath)
            return True
        except (IOError, OSError):
            if self._handle:
                self._handle.close()
                self._handle = None
            return False

    def release(self, filepath: str) -> None:
        import fcntl
        if self._handle:
            try:
                fcntl.flock(self._handle.fileno(), fcntl.LOCK_UN)
            except Exception:
                pass
            self._handle.close()
            self._handle = None

        self._cleanup_file(filepath)

    def _write_lock_info(self, filepath: str) -> None:
        if self._handle:
            lock_data = {
                'pid': os.getpid(),
                'timestamp': datetime.now().isoformat(),
                'bot_version': BOT_VERSION,
            }
            self._handle.write(json.dumps(lock_data))
            self._handle.flush()

    def _cleanup_file(self, filepath: str) -> None:
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
        except (IOError, OSError):
            pass
        except Exception:
            pass


def get_file_lock() -> FileLock:
    """Factory function to get the appropriate file lock for the platform."""
    if platform.system() == 'Windows':
        return WindowsFileLock()
    return UnixFileLock()


def is_bot_process_running(pid: Optional[int]) -> bool:
    """Check if a process with the given PID is running."""
    if pid is None:
        return False

    try:
        if platform.system() == 'Windows':
            return _check_windows_process(pid)
        else:
            os.kill(pid, 0)
            return True
    except (OSError, ProcessLookupError):
        return False
    except Exception:
        return True


def _check_windows_process(pid: int) -> bool:
    """Check if a Windows process is running."""
    try:
        result = subprocess.run(
            ['tasklist', '/FI', f'PID eq {pid}', '/FO', 'CSV', '/NH'],
            capture_output=True, text=True, timeout=5,
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0)
        )
        output = result.stdout.strip().lower()
        if output and ('python.exe' in output or 'pythonw.exe' in output):
            return True
        return False
    except Exception:
        # Fallback to ctypes
        try:
            import ctypes
            kernel32 = ctypes.windll.kernel32
            PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
            handle = kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, pid)
            if handle:
                kernel32.CloseHandle(handle)
                return True
            return False
        except Exception:
            return False


def check_duplicate_instances() -> List[int]:
    """Check for other running instances of this bot."""
    other_pids = []
    current_pid = os.getpid()

    if platform.system() != 'Windows':
        return other_pids

    try:
        for process_name in ['python.exe', 'pythonw.exe']:
            try:
                result = subprocess.run(
                    ['wmic', 'process', 'where', f"name='{process_name}'",
                     'get', 'processid,commandline', '/format:csv'],
                    capture_output=True, text=True, timeout=10,
                    creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0)
                )
                for line in result.stdout.strip().split('\n'):
                    if BOT_NAME in line.lower() or 'pattern_5m' in line.lower():
                        parts = line.strip().split(',')
                        if len(parts) >= 2:
                            try:
                                pid = int(parts[-1])
                                if pid != current_pid and pid not in other_pids:
                                    other_pids.append(pid)
                            except ValueError:
                                continue
            except Exception:
                continue
    except Exception as e:
        logger.debug(f"Could not check duplicate instances: {e}")

    return other_pids


# Global lock instance
_lock_instance: Optional[FileLock] = None


def acquire_lock() -> bool:
    """Acquire the bot lock file."""
    global _lock_instance

    os.makedirs(os.path.dirname(LOCK_FILE), exist_ok=True)

    # Check for duplicate instances first
    other_instances = check_duplicate_instances()
    if other_instances:
        logger.error("🔴 DUPLICATE INSTANCE DETECTED!")
        logger.error(f"Other bot PIDs: {other_instances}")
        return False

    _lock_instance = get_file_lock()
    if _lock_instance.acquire(LOCK_FILE):
        logger.info(f"🔒 Lock acquired (PID: {os.getpid()})")
        return True

    logger.error("🔴 LOCK ACQUISITION FAILED!")
    return False


def release_lock() -> None:
    """Release the bot lock file."""
    global _lock_instance

    if _lock_instance:
        _lock_instance.release(LOCK_FILE)
        _lock_instance = None
        logger.info("🔓 Lock released")