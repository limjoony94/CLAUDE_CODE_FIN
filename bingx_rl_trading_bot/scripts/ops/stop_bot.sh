#!/bin/bash
# stop_bot.sh — C1 Breakout v2 봇 중지
set -euo pipefail

echo "Stopping C1 Breakout bot..."

# Find and kill c1_breakout processes
PIDS=$(pgrep -f "c1_breakout_bot.py" || true)
if [ -z "$PIDS" ]; then
    echo "No C1 Breakout bot process found"
    exit 0
fi

echo "Found PIDs: $PIDS"

# Send SIGINT for graceful shutdown (saves state)
kill -INT $PIDS 2>/dev/null || true
echo "Sent SIGINT — waiting for graceful shutdown..."

# Wait up to 15s
for i in $(seq 1 15); do
    if ! pgrep -f "c1_breakout_bot.py" > /dev/null 2>&1; then
        echo "Bot stopped gracefully after ${i}s"
        exit 0
    fi
    sleep 1
done

# Force kill if still running
echo "Graceful stop timed out — force killing..."
kill -9 $PIDS 2>/dev/null || true
echo "Bot force killed"
