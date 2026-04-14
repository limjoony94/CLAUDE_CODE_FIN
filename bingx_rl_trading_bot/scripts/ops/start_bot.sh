#!/bin/bash
# start_bot.sh — C1 Breakout v2 봇 시작
set -euo pipefail

BOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"
LOG_DIR="$BOT_DIR/logs"
LOG_FILE="$LOG_DIR/c1_breakout_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "$LOG_DIR"

if pgrep -f "c1_breakout_bot.py" > /dev/null 2>&1; then
    echo "C1 Breakout already running"
    pgrep -af "c1_breakout_bot.py"
    exit 1
fi

cd "$BOT_DIR"
nohup python scripts/production/c1_breakout_bot.py > "$LOG_FILE" 2>&1 &
PID=$!

echo "C1 Breakout v2 started (PID: $PID)"
echo "Log: $LOG_FILE"
echo "Stop: kill $PID"
