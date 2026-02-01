#!/bin/bash
# stop_bot.sh — graceful 봇 중지
set -euo pipefail

SESSION="pattern_5m"

if ! tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "ℹ️  Session '$SESSION' not running"
    # Check for orphan processes
    PIDS=$(pgrep -f "pattern_5m_bot.py" || true)
    if [ -n "$PIDS" ]; then
        echo "⚠️  Found orphan processes: $PIDS"
        echo "Kill with: kill $PIDS"
    fi
    exit 0
fi

# Send SIGINT (Ctrl+C) for graceful shutdown
tmux send-keys -t "$SESSION" C-c
echo "📤 Sent SIGINT to session '$SESSION'"

# Wait up to 15s for graceful exit
for i in $(seq 1 15); do
    if ! tmux has-session -t "$SESSION" 2>/dev/null; then
        echo "✅ Bot stopped gracefully after ${i}s"
        exit 0
    fi
    sleep 1
done

# Force kill if still running
echo "⚠️  Graceful stop timed out, force killing..."
tmux kill-session -t "$SESSION" 2>/dev/null || true
echo "🛑 Session force killed"
