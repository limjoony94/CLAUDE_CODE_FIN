#!/bin/bash
echo "🛑 Stopping ALL bot processes..."
pkill -f "sweet2_paper_trading" 2>/dev/null || echo "pkill not available, using taskkill..."
taskkill //F //FI "WINDOWTITLE eq *sweet2*" 2>/dev/null || echo "No Windows processes"
taskkill //F //IM python.exe 2>/dev/null || echo "Attempting Linux kill..."
sleep 3

echo "✅ All bots stopped"
echo "🔄 Starting single instance with 500 candles..."

cd "$(dirname "$0")"
nohup python scripts/production/sweet2_paper_trading.py > /dev/null 2>&1 &
NEW_PID=$!

echo "✅ Bot started with PID: $NEW_PID"
sleep 5

echo "📝 Checking log..."
tail -30 logs/sweet2_paper_trading_*.log | grep -E "500 candles|Phase 4 Base|37 features"

echo ""
echo "✅ Single bot instance now running!"
