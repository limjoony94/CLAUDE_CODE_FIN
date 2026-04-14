#!/bin/bash
# status.sh — C1 Breakout v2 상태 확인
set -euo pipefail

BOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"

echo "=== C1 Breakout v2 Status ==="

# Python processes
PROCS=$(pgrep -af "c1_breakout_bot.py" || true)
if [ -n "$PROCS" ]; then
    echo "RUNNING: $PROCS"
else
    echo "NOT RUNNING"
fi

# Bot state file
echo ""
echo "=== State ==="
STATE_FILE="$BOT_DIR/results/c1_breakout_state.json"
if [ -f "$STATE_FILE" ]; then
    echo "Last modified: $(stat -c '%y' "$STATE_FILE" 2>/dev/null || stat -f '%Sm' "$STATE_FILE" 2>/dev/null || echo '?')"
    python3 -c "
import json
with open('$STATE_FILE') as f:
    s = json.load(f)
print(f'  Positions: {len(s.get(\"positions\",[]))}')
print(f'  Trades: {len(s.get(\"trade_history\",[]))}')
print(f'  Updated: {s.get(\"updated\",\"?\")}')
" 2>/dev/null || echo "  (could not parse state file)"
else
    echo "  No state file found"
fi

# Latest log
echo ""
echo "=== Latest Log ==="
LOG_FILE="$BOT_DIR/logs/c1_breakout.log"
if [ -f "$LOG_FILE" ]; then
    echo "Last 5 lines:"
    tail -5 "$LOG_FILE"
else
    echo "No log file found"
fi
