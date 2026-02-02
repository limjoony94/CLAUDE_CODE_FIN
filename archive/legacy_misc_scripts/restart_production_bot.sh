#!/bin/bash

# Production Bot Restart Script
# Upgrades from Phase 2 (33 features) to Phase 4 Base (37 features)
# Expected improvement: +920% (0.75% → 7.68%)

set -e  # Exit on error

PROJECT_DIR="C:\Users\J\OneDrive\CLAUDE_CODE_FIN\bingx_rl_trading_bot"
LOG_DIR="$PROJECT_DIR/logs"
SCRIPT="$PROJECT_DIR/scripts/production/sweet2_paper_trading.py"

echo "=========================================="
echo "Production Bot Restart"
echo "=========================================="
echo "Upgrade: Phase 2 → Phase 4 Base"
echo "Expected: 0.75% → 7.68% (+920%)"
echo ""

# Step 1: Stop current bot
echo "[1/5] Stopping current bot (Phase 2)..."
pkill -f sweet2_paper_trading || echo "No running bot found"
sleep 2

# Verify stopped
if pgrep -f sweet2_paper_trading > /dev/null; then
    echo "ERROR: Bot still running. Force killing..."
    pkill -9 -f sweet2_paper_trading
    sleep 2
fi

echo "✅ Bot stopped"

# Step 2: Verify Phase 4 Base model exists
echo ""
echo "[2/5] Verifying Phase 4 Base model..."
MODEL_FILE="$PROJECT_DIR/models/xgboost_v4_phase4_advanced_lookahead3_thresh0.pkl"
FEATURES_FILE="$PROJECT_DIR/models/xgboost_v4_phase4_advanced_lookahead3_thresh0_features.txt"

if [ ! -f "$MODEL_FILE" ]; then
    echo "ERROR: Phase 4 Base model not found: $MODEL_FILE"
    exit 1
fi

if [ ! -f "$FEATURES_FILE" ]; then
    echo "ERROR: Features file not found: $FEATURES_FILE"
    exit 1
fi

echo "✅ Phase 4 Base model verified"

# Step 3: Backup old logs
echo ""
echo "[3/5] Backing up logs..."
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
if ls $LOG_DIR/sweet2_paper_trading_*.log 1> /dev/null 2>&1; then
    mkdir -p "$LOG_DIR/backup"
    cp $LOG_DIR/sweet2_paper_trading_*.log "$LOG_DIR/backup/"
    echo "✅ Logs backed up to $LOG_DIR/backup/"
else
    echo "No logs to backup"
fi

# Step 4: Start new bot with Phase 4 Base
echo ""
echo "[4/5] Starting bot with Phase 4 Base model..."
cd "$PROJECT_DIR"

# Start in background
nohup python "$SCRIPT" > "$LOG_DIR/sweet2_restart_$TIMESTAMP.log" 2>&1 &
BOT_PID=$!

echo "✅ Bot started (PID: $BOT_PID)"

# Step 5: Verify startup
echo ""
echo "[5/5] Verifying startup..."
sleep 5

# Check if process is running
if ps -p $BOT_PID > /dev/null; then
    echo "✅ Bot is running"

    # Check logs for Phase 4 Base confirmation
    sleep 3
    LATEST_LOG=$(ls -t $LOG_DIR/sweet2_paper_trading_*.log 2>/dev/null | head -1)

    if [ -f "$LATEST_LOG" ]; then
        echo ""
        echo "=== Latest Log (last 20 lines) ==="
        tail -20 "$LATEST_LOG"

        echo ""
        echo "=== Checking for Phase 4 Base confirmation ==="
        if grep -q "Phase 4 Base" "$LATEST_LOG"; then
            echo "✅ CONFIRMED: Phase 4 Base model loaded!"
        elif grep -q "37 features" "$LATEST_LOG"; then
            echo "✅ CONFIRMED: 37 features loaded (Phase 4 Base)"
        else
            echo "⚠️  WARNING: Could not confirm Phase 4 Base in logs"
            echo "    Check manually: tail -f $LATEST_LOG"
        fi
    fi
else
    echo "❌ ERROR: Bot process died. Check logs:"
    echo "    tail -100 $LOG_DIR/sweet2_restart_$TIMESTAMP.log"
    exit 1
fi

echo ""
echo "=========================================="
echo "✅ RESTART COMPLETE"
echo "=========================================="
echo ""
echo "Next Steps:"
echo "1. Monitor logs: tail -f $LATEST_LOG"
echo "2. Check first trades (expect 2-3 per day)"
echo "3. Verify XGBoost probabilities > 0.7 for entries"
echo "4. Monitor returns vs 7.68% expected"
echo ""
echo "Expected Performance (Phase 4 Base):"
echo "  - Returns: 7.68% per 5 days (~1.5% per day)"
echo "  - Win Rate: 69.1%"
echo "  - Trades: ~15 per 5 days (3 per day)"
echo "  - Max Drawdown: <1%"
echo ""
echo "Monitor with: tail -f $LATEST_LOG | grep -E 'ENTRY|EXIT|Phase 4|XGBoost Prob'"
echo "=========================================="
