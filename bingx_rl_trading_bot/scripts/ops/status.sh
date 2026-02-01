#!/bin/bash
# status.sh — 프로세스 상태 확인
set -euo pipefail

SESSION="pattern_5m"
BOT_DIR="/home/sp/.openclaw/workspace/CLAUDE_CODE_FIN/bingx_rl_trading_bot"

echo "=== 🤖 Bot Process Status ==="

# tmux session
if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "✅ tmux session '$SESSION': RUNNING"
    tmux list-panes -t "$SESSION" -F '  PID: #{pane_pid}  CMD: #{pane_current_command}'
else
    echo "❌ tmux session '$SESSION': NOT RUNNING"
fi

# Python processes
echo ""
echo "=== Python Processes ==="
PROCS=$(pgrep -af "pattern_5m_bot.py" || true)
if [ -n "$PROCS" ]; then
    echo "$PROCS"
else
    echo "No pattern_5m_bot.py processes found"
fi

# Bot state file
echo ""
echo "=== Bot State ==="
STATE_FILE="$BOT_DIR/results/pattern_5m_bot_state.json"
if [ -f "$STATE_FILE" ]; then
    echo "Last modified: $(stat -c '%y' "$STATE_FILE" 2>/dev/null || stat -f '%Sm' "$STATE_FILE" 2>/dev/null)"
    python3 -c "
import json, sys
with open('$STATE_FILE') as f:
    s = json.load(f)
for k in ['status','uptime','last_signal','open_positions','total_trades']:
    if k in s: print(f'  {k}: {s[k]}')
" 2>/dev/null || echo "  (could not parse state file)"
else
    echo "  No state file found"
fi

# Latest log
echo ""
echo "=== Latest Log ==="
LATEST_LOG=$(ls -t "$BOT_DIR/logs/pattern_5m_bot_"*.log 2>/dev/null | head -1)
if [ -n "$LATEST_LOG" ]; then
    echo "File: $LATEST_LOG"
    echo "Last 3 lines:"
    tail -3 "$LATEST_LOG"
else
    echo "No log files found"
fi
