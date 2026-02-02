"""
Monitor Grid Search Progress
=============================

Monitor the lookback period grid search progress in real-time.
"""

import time
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
log_dir = PROJECT_ROOT / "logs"

# Find latest lookback optimization log
log_files = sorted(log_dir.glob("lookback_optimization_*.log"), key=lambda x: x.stat().st_mtime, reverse=True)

if len(log_files) == 0:
    print("❌ No lookback optimization log files found")
    sys.exit(1)

latest_log = log_files[0]
print(f"📊 Monitoring: {latest_log.name}")
print(f"=" * 80)

# Monitor log file
last_size = 0
while True:
    current_size = latest_log.stat().st_size

    if current_size > last_size:
        with open(latest_log, 'r') as f:
            f.seek(last_size)
            new_content = f.read()
            print(new_content, end='')
            last_size = current_size

    time.sleep(2)
