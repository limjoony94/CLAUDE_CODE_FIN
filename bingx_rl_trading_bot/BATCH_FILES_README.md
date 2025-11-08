# Batch Files Guide

**Opportunity Gating Trading Bot (4x Leverage)**

Quick reference for all batch file commands to control and monitor the trading bot.

---

## Quick Start

```
1. START_BOT.bat      - Start the trading bot
2. QUANT_MONITOR.bat  - Monitor performance in real-time
3. STOP_BOT.bat       - Stop the bot when needed
```

---

## Available Commands

### 🚀 START_BOT.bat
**Purpose**: Start the Opportunity Gating trading bot

**What it does**:
- Checks if bot is already running
- Verifies all model files exist
- Starts `opportunity_gating_bot_4x.py`
- Displays bot configuration

**Usage**:
```cmd
START_BOT.bat
```

**Expected Output**:
```
✅ No conflicting bot process found
✅ All model files found (LONG Entry, SHORT Entry, LONG Exit, SHORT Exit)
Starting: python scripts/production/opportunity_gating_bot_4x.py

Bot configuration:
  LONG Threshold: 0.65
  SHORT Threshold: 0.70 (with Opportunity Gating)
  Gate Threshold: 0.001
  Position Sizing: Dynamic (20-95%)
  Max Hold: 240 candles (4 hours)
  Take Profit: 3.0%
  Stop Loss: -1.5%
```

---

### 🛑 STOP_BOT.bat
**Purpose**: Safely stop the trading bot

**What it does**:
- Checks if bot is running
- Shows lock file status
- Terminates Python processes
- Removes lock files

**Usage**:
```cmd
STOP_BOT.bat
```

**Warning**: This stops **ALL** Python processes. If you have other Python programs running, they will also be stopped.

---

### 🔄 RESTART_BOT.bat
**Purpose**: Restart the bot (stop + start)

**What it does**:
- Stops currently running bot
- Waits for clean shutdown (3 seconds)
- Checks state files
- Calls START_BOT.bat to restart

**Usage**:
```cmd
RESTART_BOT.bat
```

**Use cases**:
- After updating model files
- After changing configuration
- To reset a stuck bot
- After system updates

---

### 📊 QUANT_MONITOR.bat
**Purpose**: Real-time monitoring dashboard

**What it does**:
- Displays live performance metrics
- Shows risk analytics (Sharpe, Sortino, VaR)
- Tracks signal quality
- ASCII visualization
- Automated alerts

**Usage**:
```cmd
QUANT_MONITOR.bat
```

**Features**:
- Real-time P&L tracking
- Win rate and trade statistics
- Current position status
- Latest signals (LONG/SHORT probabilities)
- Session performance summary

**Requirements**:
- Bot must be running
- State file must exist: `results\opportunity_gating_bot_4x_state.json`
- Log files: `logs\opportunity_gating_bot_4x_*.log`

---

### ℹ️ STATUS_BOT.bat
**Purpose**: Quick status check

**What it does**:
- Checks if bot is running
- Verifies state file exists
- Shows latest log entries
- Verifies model files (4 files expected)
- Lists available commands

**Usage**:
```cmd
STATUS_BOT.bat
```

**Output sections**:
1. Bot Process Status (running/stopped)
2. State File Status (exists/missing)
3. Log Files Status (latest entries)
4. Model Files Status (4/4 required)
5. Quick Actions (available commands)

---

## Bot Configuration

**Strategy**: Opportunity Gating with 4x Leverage
```yaml
Entry Signals:
  LONG Entry: probability >= 0.65
  SHORT Entry: probability >= 0.70 + Opportunity Gating
    Gate Check: EV(SHORT) > EV(LONG) + 0.001

Position Sizing: Dynamic (20-95%)
  - Signal strength
  - Volatility adjustment
  - Regime detection
  - Win/loss streak

Exit Conditions:
  Take Profit: +3.0% (on leveraged P&L)
  Stop Loss: -1.5% (on leveraged P&L)
  Max Hold: 240 candles (4 hours)

Leverage: 4x (applied to all P&L calculations)
```

---

## File Locations

### Models (required for bot operation)
```
models/
  ├── xgboost_v4_phase4_advanced_lookahead3_thresh0.pkl        (LONG Entry)
  ├── xgboost_v4_phase4_advanced_lookahead3_thresh0_scaler.pkl (LONG Scaler)
  ├── xgboost_short_redesigned_20251016_233322.pkl             (SHORT Entry)
  └── xgboost_short_redesigned_20251016_233322_scaler.pkl      (SHORT Scaler)
```

### State & Logs
```
results/
  └── opportunity_gating_bot_4x_state.json  (Bot state, trades, stats)

logs/
  └── opportunity_gating_bot_4x_YYYYMMDD.log  (Daily log files)
```

### Configuration
```
config/
  └── api_keys.yaml  (BingX API credentials)
```

---

## Expected Performance (Backtest)

Based on 105-day backtest (2025-07-02 to 2025-10-14):

```
Total Return: 18.13% per window
Win Rate: 63.9%
Trades per Week: ~26 trades
Average Trade Duration: 2.3 hours
Sharpe Ratio: ~11.88
```

**LONG vs SHORT Distribution**:
- LONG trades: 80%
- SHORT trades: 20% (gated for quality)

---

## Troubleshooting

### Bot won't start
1. Run `STATUS_BOT.bat` to check model files
2. Verify all 4 model files exist
3. Check if bot is already running (`STOP_BOT.bat` first)
4. Review latest log file for errors

### Monitor shows no data
1. Ensure bot is running (`STATUS_BOT.bat`)
2. Check state file exists: `results\opportunity_gating_bot_4x_state.json`
3. Wait 1-2 minutes for first data collection
4. Verify log files are being created

### Bot stopped unexpectedly
1. Check log files for errors: `logs\opportunity_gating_bot_4x_*.log`
2. Common issues:
   - API connection lost (BingX testnet down)
   - Feature calculation error (data quality)
   - JSON serialization error (state file corruption)
3. Use `RESTART_BOT.bat` to restart

### Corrupted state file
```cmd
# Delete corrupted state and restart fresh
cd bingx_rl_trading_bot
del /F /Q results\opportunity_gating_bot_4x_state.json
START_BOT.bat
```

---

## Safety Notes

1. **Testnet Environment**: Bot runs on BingX testnet (virtual money)
2. **No Real Risk**: All trading is simulated
3. **Week 1 Validation**: Monitor for 7 days before considering real money
4. **Expected Metrics**: Aim for >60% win rate, ~26 trades/week
5. **Stop Anytime**: Use `STOP_BOT.bat` to stop immediately

---

## Workflow Examples

### Daily Monitoring Routine
```cmd
1. STATUS_BOT.bat         (Quick health check)
2. QUANT_MONITOR.bat      (Review performance)
3. Check for trades       (Win rate, P&L)
```

### After Model Update
```cmd
1. STOP_BOT.bat           (Stop current bot)
2. Replace model files    (Copy new .pkl files)
3. RESTART_BOT.bat        (Start with new models)
4. QUANT_MONITOR.bat      (Verify working)
```

### Weekly Review
```cmd
1. QUANT_MONITOR.bat      (Full performance review)
2. Check metrics:
   - Total trades (expect ~26/week)
   - Win rate (expect >60%)
   - Total P&L (expect positive)
   - Sharpe ratio (expect >2.0)
```

---

## Support

For issues or questions:
1. Check log files: `logs\opportunity_gating_bot_4x_*.log`
2. Review state file: `results\opportunity_gating_bot_4x_state.json`
3. Run `STATUS_BOT.bat` for diagnostic info
4. Refer to project documentation in `claudedocs/`

---

**Last Updated**: 2025-10-17
**Bot Version**: Opportunity Gating Bot (4x Leverage)
**Environment**: BingX Testnet
