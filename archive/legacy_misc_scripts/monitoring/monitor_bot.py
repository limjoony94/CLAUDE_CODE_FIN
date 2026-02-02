"""
Automated Bot Monitoring Script

Monitors sweet2_paper_trading bot and reports key events:
- First trade
- XGBoost Prob > 0.7 (entry signal)
- Errors or issues
"""

import time
from pathlib import Path
from datetime import datetime
from loguru import logger

PROJECT_ROOT = Path(__file__).parent.parent.parent
LOG_FILE = PROJECT_ROOT / "logs" / "sweet2_paper_trading_20251010.log"
MONITOR_LOG = PROJECT_ROOT / "logs" / "monitoring_20251010.log"

# Configure logger
logger.remove()
logger.add(MONITOR_LOG, rotation="100 MB", level="INFO")
logger.add(lambda msg: print(msg, end=""), level="INFO")

logger.info("=" * 80)
logger.info("Bot Monitoring Started")
logger.info("=" * 80)

last_position = 0
last_check_time = datetime.now()
check_interval = 60  # Check every 60 seconds

# Track state
trades_seen = 0
max_prob_seen = 0.0
entry_signals_seen = 0

def parse_log_line(line):
    """Parse log line for key information"""
    if "XGBoost Prob:" in line:
        try:
            prob = float(line.split("XGBoost Prob:")[1].strip().split()[0])
            return "prob", prob
        except:
            pass
    elif "ENTRY" in line and "BTC @" in line:
        return "entry", line
    elif "EXIT" in line and "Reason:" in line:
        return "exit", line
    elif "ERROR" in line or "WARNING" in line:
        return "alert", line
    return None, None

logger.info(f"Monitoring log file: {LOG_FILE}")
logger.info(f"Check interval: {check_interval} seconds")
logger.info(f"Waiting for events...")
logger.info("")

try:
    while True:
        # Check if log file exists
        if not LOG_FILE.exists():
            logger.warning(f"Log file not found: {LOG_FILE}")
            time.sleep(check_interval)
            continue

        # Read new lines
        with open(LOG_FILE, 'r', encoding='utf-8') as f:
            f.seek(last_position)
            new_lines = f.readlines()
            last_position = f.tell()

        # Process new lines
        for line in new_lines:
            event_type, data = parse_log_line(line)

            if event_type == "prob":
                if data > max_prob_seen:
                    max_prob_seen = data
                    if data > 0.5:
                        logger.info(f"🔔 High Probability Detected: {data:.3f}")

                if data > 0.7:
                    entry_signals_seen += 1
                    logger.success(f"🎯 ENTRY SIGNAL! XGBoost Prob: {data:.3f}")

            elif event_type == "entry":
                trades_seen += 1
                logger.success(f"🚀 TRADE #{trades_seen} ENTRY!")
                logger.success(f"   {data.strip()}")

            elif event_type == "exit":
                logger.info(f"📊 TRADE EXIT:")
                logger.info(f"   {data.strip()}")

            elif event_type == "alert":
                logger.warning(f"⚠️ ALERT: {data.strip()}")

        # Periodic status update
        current_time = datetime.now()
        elapsed = (current_time - last_check_time).total_seconds()

        if elapsed >= 300:  # Every 5 minutes
            logger.info("")
            logger.info(f"⏰ Status Update ({current_time.strftime('%H:%M:%S')})")
            logger.info(f"   Trades: {trades_seen}")
            logger.info(f"   Entry Signals (>0.7): {entry_signals_seen}")
            logger.info(f"   Max Prob Seen: {max_prob_seen:.3f}")
            logger.info(f"   Monitoring... ✅")
            logger.info("")
            last_check_time = current_time

        time.sleep(check_interval)

except KeyboardInterrupt:
    logger.info("")
    logger.info("=" * 80)
    logger.info("Monitoring Stopped")
    logger.info("=" * 80)
    logger.info(f"Total Trades: {trades_seen}")
    logger.info(f"Total Entry Signals (>0.7): {entry_signals_seen}")
    logger.info(f"Max Probability Seen: {max_prob_seen:.3f}")
