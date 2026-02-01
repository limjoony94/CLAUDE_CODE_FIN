#!/bin/bash
# restart_bot.sh — 봇 재시작
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

echo "🔄 Restarting bot..."
"$SCRIPT_DIR/stop_bot.sh"
sleep 2
"$SCRIPT_DIR/start_bot.sh"
