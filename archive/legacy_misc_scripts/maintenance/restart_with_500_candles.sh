#!/bin/bash
echo "🔄 Restarting Bot with 500 candles..."
cd "$(dirname "$0")"

# Start new bot (it will use updated code with 500 candles)
nohup python scripts/production/sweet2_paper_trading.py > logs/restart_$(date +%Y%m%d_%H%M%S).log 2>&1 &
NEW_PID=$!

echo "✅ New bot started with PID: $NEW_PID"
echo "📊 Lookback: 500 candles (was 300)"
echo "⏳ Wait 10 seconds for initialization..."
sleep 10

echo "📝 Checking logs for confirmation..."
tail -30 logs/sweet2_paper_trading_*.log | grep -E "500|Phase 4|37 features"

echo ""
echo "✅ Bot restart complete!"
echo "📊 Verify: tail -f logs/sweet2_paper_trading_*.log"
