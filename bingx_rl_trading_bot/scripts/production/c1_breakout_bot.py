"""
C1 Breakout v2 Bot — Entry Point
==================================
15m Channel Breakout + Fractal SL + ATR Trailing TP

Usage:
  python scripts/production/c1_breakout_bot.py
  tmux new-session -d -s c1 "python scripts/production/c1_breakout_bot.py"
"""
import os
import logging
import logging.handlers
import sys
from pathlib import Path
from datetime import datetime

project_root = Path(__file__).resolve().parent.parent.parent
os.chdir(project_root)
sys.path.insert(0, str(project_root))

from scripts.production.c1_breakout.bot import C1BreakoutBot

Path("logs").mkdir(exist_ok=True)

# Duplicate instance guard — prevent two bots trading same account
_LOCK_PATH = Path(os.environ.get('TEMP', '/tmp')) / 'c1_breakout.lock'

def _acquire_lock():
    """Prevent duplicate bot instances. Uses temp dir to avoid OneDrive conflicts."""
    if _LOCK_PATH.exists():
        try:
            old_pid = int(_LOCK_PATH.read_text().strip())
            # Check if process still alive (Windows-compatible)
            import ctypes
            kernel32 = ctypes.windll.kernel32
            handle = kernel32.OpenProcess(0x100000, False, old_pid)  # SYNCHRONIZE
            if handle:
                kernel32.CloseHandle(handle)
                print(f"ERROR: Another C1 Breakout bot is running (PID {old_pid}). Aborting.")
                sys.exit(1)
        except (ValueError, OSError, AttributeError):
            pass  # Stale lock or non-Windows — proceed
    _LOCK_PATH.write_text(str(os.getpid()))

def _release_lock():
    try:
        _LOCK_PATH.unlink(missing_ok=True)
    except Exception:
        pass

# Date-based log file (append on restart, rotate daily at midnight)
log_file = "logs/c1_breakout.log"
file_handler = logging.handlers.TimedRotatingFileHandler(
    log_file, when='midnight', backupCount=30, utc=True, encoding='utf-8')
file_handler.suffix = "%Y-%m-%d"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
    handlers=[file_handler, logging.StreamHandler()]
)

logger = logging.getLogger('c1_breakout')

if __name__ == '__main__':
    _acquire_lock()
    try:
        logger.info("=" * 60)
        logger.info(f"BOT START — {datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')} UTC")
        logger.info("=" * 60)
        bot = C1BreakoutBot()
        bot.run()
    finally:
        _release_lock()
