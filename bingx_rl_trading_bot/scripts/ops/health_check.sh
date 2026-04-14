#!/bin/bash
# health_check.sh — C1 Breakout v2 API 연결 + 상태 요약
set -euo pipefail

BOT_DIR="$(cd "$(dirname "$0")/../.." && pwd)"

echo "=== C1 Breakout v2 Health Check ==="
echo "Time: $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo ""

# 1. Process check
echo "--- Process ---"
if pgrep -f "c1_breakout_bot.py" > /dev/null 2>&1; then
    echo "Bot: RUNNING (PID $(pgrep -f c1_breakout_bot.py))"
else
    echo "Bot: STOPPED"
fi

# 2. API connectivity
echo ""
echo "--- API Connectivity ---"
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" --connect-timeout 5 "https://open-api.bingx.com/openApi/swap/v2/server/time" 2>/dev/null || echo "000")
if [ "$HTTP_CODE" = "200" ]; then
    echo "BingX API: OK (HTTP $HTTP_CODE)"
else
    echo "BingX API: FAILED (HTTP $HTTP_CODE)"
fi

# 3. State summary
echo ""
echo "--- State ---"
STATE_FILE="$BOT_DIR/results/c1_breakout_state.json"
if [ -f "$STATE_FILE" ]; then
    python3 -c "
import json
with open('$STATE_FILE') as f:
    s = json.load(f)
print(f'  Positions: {len(s.get(\"positions\",[]))}')
print(f'  Trades: {len(s.get(\"trade_history\",[]))}')
print(f'  Updated: {s.get(\"updated\",\"?\")}')
" 2>/dev/null || echo "  (could not parse state)"
else
    echo "  No state file found"
fi

# 4. Disk usage
echo ""
echo "--- Disk ---"
du -sh "$BOT_DIR/logs/" 2>/dev/null | awk '{print "  Logs: "$1}' || true
du -sh "$BOT_DIR/results/" 2>/dev/null | awk '{print "  Results: "$1}' || true
