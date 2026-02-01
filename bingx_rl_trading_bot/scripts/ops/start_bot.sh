#!/bin/bash
# start_bot.sh — tmux 세션으로 pattern_5m 봇 시작
set -euo pipefail

BOT_DIR="/home/sp/.openclaw/workspace/CLAUDE_CODE_FIN/bingx_rl_trading_bot"
SESSION="pattern_5m"
LOG_DIR="$BOT_DIR/logs"
LOG_FILE="$LOG_DIR/pattern_5m_bot_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "$LOG_DIR"

if tmux has-session -t "$SESSION" 2>/dev/null; then
    echo "⚠️  Session '$SESSION' already running"
    tmux list-panes -t "$SESSION" -F '#{pane_pid} #{pane_current_command}'
    exit 1
fi

tmux new-session -d -s "$SESSION" \
    "cd $BOT_DIR && python3 scripts/production/pattern_5m_bot.py 2>&1 | tee $LOG_FILE"

echo "✅ Bot started in tmux session '$SESSION'"
echo "📄 Log: $LOG_FILE"
echo "💡 Attach: tmux attach -t $SESSION"
