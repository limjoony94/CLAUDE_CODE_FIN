"""
R26 Grid Bot — Entry Point
============================
1h volatility-harvest grid trading on BTC/USDT BingX perp.
Pre-reg: claudedocs/round26_grid_ranging_prereg.md (commit bd9a233)
Verification: claudedocs/round34_r26_leverage_prereg.md (eb1a271)

Usage:
  python scripts/production/r26_grid_bot.py
  tmux new-session -d -s r26 "python scripts/production/r26_grid_bot.py"

NOTE — User manually adjusts:
  - config/r26_grid_config.yaml → risk.per_level_notional_usd (capital per level)
  - config/r26_grid_config.yaml → exchange.trading_leverage (≤ exchange_leverage)
"""
import os
import logging
import logging.handlers
import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent.parent
os.chdir(project_root)
sys.path.insert(0, str(project_root))

from scripts.production.r26_grid.bot import R26GridBot

Path("logs").mkdir(exist_ok=True)

# Duplicate-instance guard (prevents two bots trading same account)
_LOCK_PATH = Path(os.environ.get('TEMP', '/tmp')) / 'r26_grid.lock'


def _acquire_lock():
    """Prevent duplicate bot instances using PID file in temp dir."""
    if _LOCK_PATH.exists():
        try:
            with open(_LOCK_PATH, 'r') as f:
                pid = int(f.read().strip())
            # Check if process alive
            try:
                os.kill(pid, 0)
                logging.error(f"Another R26 bot is running (PID {pid}). Aborting.")
                sys.exit(1)
            except (ProcessLookupError, PermissionError):
                # Stale lock, override
                pass
        except Exception:
            pass

    with open(_LOCK_PATH, 'w') as f:
        f.write(str(os.getpid()))


def _release_lock():
    if _LOCK_PATH.exists():
        try:
            _LOCK_PATH.unlink()
        except Exception:
            pass


def _setup_logging():
    """Daily-rotating file handler + stderr."""
    log_path = Path('logs/r26_grid.log')
    log_path.parent.mkdir(parents=True, exist_ok=True)

    formatter = logging.Formatter(
        '%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    file_handler = logging.handlers.TimedRotatingFileHandler(
        log_path, when='midnight', interval=1, backupCount=30, utc=True
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)

    stream_handler = logging.StreamHandler(sys.stderr)
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(logging.INFO)

    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.addHandler(file_handler)
    root.addHandler(stream_handler)


def main():
    _setup_logging()
    _acquire_lock()
    try:
        bot = R26GridBot()
        bot.run()
    finally:
        _release_lock()


if __name__ == '__main__':
    main()
