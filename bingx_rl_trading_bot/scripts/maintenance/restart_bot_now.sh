#!/bin/bash
echo "Stopping old bot..."
pkill -f sweet2_paper_trading
sleep 3
echo "Starting Phase 4 Base bot..."
cd "$(dirname "$0")"
nohup python scripts/production/sweet2_paper_trading.py > /dev/null 2>&1 &
echo "Bot restarted. Check logs/sweet2_paper_trading_*.log for Phase 4 Base confirmation"
