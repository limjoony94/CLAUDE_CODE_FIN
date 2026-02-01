#!/bin/bash
# log_tail.sh — 최근 로그 N줄
set -euo pipefail

BOT_DIR="/home/sp/.openclaw/workspace/CLAUDE_CODE_FIN/bingx_rl_trading_bot"
LINES="${1:-50}"

LATEST_LOG=$(ls -t "$BOT_DIR/logs/pattern_5m_bot_"*.log 2>/dev/null | head -1)

if [ -z "$LATEST_LOG" ]; then
    # Try structured logs
    LATEST_LOG=$(ls -t "$BOT_DIR/logs/"*.jsonl "$BOT_DIR/logs/"*.log 2>/dev/null | head -1)
fi

if [ -z "$LATEST_LOG" ]; then
    echo "❌ No log files found in $BOT_DIR/logs/"
    exit 1
fi

echo "📄 Log: $LATEST_LOG"
echo "📏 Last $LINES lines:"
echo "---"
tail -n "$LINES" "$LATEST_LOG"
