#!/bin/bash
# health_check.sh — API 연결 + 메트릭 요약
set -euo pipefail

BOT_DIR="/home/sp/.openclaw/workspace/CLAUDE_CODE_FIN/bingx_rl_trading_bot"

echo "=== 🏥 Health Check ==="
echo "Time: $(date '+%Y-%m-%d %H:%M:%S %Z')"
echo ""

# 1. Process check
echo "--- Process ---"
if tmux has-session -t pattern_5m 2>/dev/null; then
    echo "✅ Bot process: RUNNING"
else
    echo "❌ Bot process: STOPPED"
fi

# 2. API connectivity
echo ""
echo "--- API Connectivity ---"
HTTP_CODE=$(curl -s -o /dev/null -w "%{http_code}" --connect-timeout 5 "https://open-api.bingx.com/openApi/swap/v2/server/time" 2>/dev/null || echo "000")
if [ "$HTTP_CODE" = "200" ]; then
    echo "✅ BingX API: OK (HTTP $HTTP_CODE)"
else
    echo "❌ BingX API: FAILED (HTTP $HTTP_CODE)"
fi

# 3. Metrics summary
echo ""
echo "--- Metrics ---"
METRICS_FILE="$BOT_DIR/results/pattern_5m_metrics.json"
if [ -f "$METRICS_FILE" ]; then
    python3 -c "
import json
with open('$METRICS_FILE') as f:
    m = json.load(f)
if isinstance(m, dict):
    for k, v in list(m.items())[:10]:
        print(f'  {k}: {v}')
elif isinstance(m, list) and len(m) > 0:
    last = m[-1] if isinstance(m[-1], dict) else {}
    for k, v in list(last.items())[:10]:
        print(f'  {k}: {v}')
" 2>/dev/null || echo "  (could not parse metrics)"
else
    echo "  No metrics file found"
fi

# 4. Disk usage
echo ""
echo "--- Disk ---"
du -sh "$BOT_DIR/logs/" 2>/dev/null | awk '{print "  Logs: "$1}'
du -sh "$BOT_DIR/results/" 2>/dev/null | awk '{print "  Results: "$1}'
du -sh "$BOT_DIR/models/" 2>/dev/null | awk '{print "  Models: "$1}'
