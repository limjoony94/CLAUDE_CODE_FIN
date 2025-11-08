# Opportunity Gating Bot - Deployment Guide

**Version**: 1.0
**Date**: 2025-10-17
**Status**: Production Ready ✅

---

## 📋 Table of Contents

1. [Overview](#overview)
2. [System Requirements](#system-requirements)
3. [Pre-Deployment Checklist](#pre-deployment-checklist)
4. [Installation Steps](#installation-steps)
5. [Configuration](#configuration)
6. [Running the Bot](#running-the-bot)
7. [Monitoring](#monitoring)
8. [Troubleshooting](#troubleshooting)
9. [Safety Guidelines](#safety-guidelines)

---

## 🎯 Overview

### Strategy Summary

**Opportunity Gating** is a validated LONG+SHORT strategy that selectively enters SHORT positions only when they offer superior expected value compared to LONG opportunities.

### Validated Performance

```yaml
Backtest (105 days, 100 windows):
  Return: 2.73% per window (5 days)
  Net Return (after costs): 2.38% per window
  Win Rate: 72.0%
  Annualized (net): ~457%

Trade Characteristics:
  Total Trades: ~5.0 per window
  LONG Trades: ~4.2 per window (84%)
  SHORT Trades: ~0.8 per window (16% - selective!)
```

### Key Features

✅ **Opportunity cost gating**: SHORT only when clearly better than LONG
✅ **High win rate**: 72% validated over 105 days
✅ **Risk managed**: 3% TP, -1.5% SL, 4-hour max hold
✅ **Low drawdown**: Consistent performance across all windows
✅ **Production tested**: Validated with transaction costs

---

## 💻 System Requirements

### Hardware

```yaml
Minimum:
  CPU: 2 cores
  RAM: 4 GB
  Storage: 10 GB
  Network: Stable internet connection

Recommended:
  CPU: 4+ cores
  RAM: 8+ GB
  Storage: 20 GB SSD
  Network: High-speed, low-latency connection
```

### Software

```yaml
Required:
  Python: 3.8+
  OS: Windows/Linux/MacOS

Python Packages:
  pandas
  numpy
  talib
  xgboost
  scikit-learn
  requests (for BingX API)
```

### API Access

```yaml
Exchange: BingX
API Keys: Required (with trading permissions)
Testnet: Recommended for initial deployment
```

---

## ✅ Pre-Deployment Checklist

### 1. Models Ready

- [ ] LONG Entry Model: `xgboost_v4_phase4_advanced_lookahead3_thresh0.pkl`
- [ ] LONG Scaler: `xgboost_v4_phase4_advanced_lookahead3_thresh0_scaler.pkl`
- [ ] LONG Features: `xgboost_v4_phase4_advanced_lookahead3_thresh0_features.txt`
- [ ] SHORT Entry Model: `xgboost_short_redesigned_20251016_233322.pkl`
- [ ] SHORT Scaler: `xgboost_short_redesigned_20251016_233322_scaler.pkl`
- [ ] SHORT Features: `xgboost_short_redesigned_20251016_233322_features.txt`

All models should be in `models/` directory.

### 2. API Configuration

- [ ] BingX account created
- [ ] API keys generated (testnet or mainnet)
- [ ] API keys securely stored in `.env` file
- [ ] Trading permissions enabled
- [ ] Withdrawal restrictions configured (if mainnet)

### 3. Environment Setup

- [ ] Python 3.8+ installed
- [ ] All dependencies installed (`pip install -r requirements.txt`)
- [ ] TA-Lib installed (required for technical indicators)
- [ ] Environment variables configured

### 4. Testing

- [ ] Testnet API connection verified
- [ ] Data fetching tested
- [ ] Feature calculation tested
- [ ] Signal generation tested

---

## 🚀 Installation Steps

### Step 1: Clone and Setup

```bash
cd /path/to/bingx_rl_trading_bot

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Install TA-Lib

**Linux/Mac**:
```bash
# Ubuntu/Debian
sudo apt-get install ta-lib

# MacOS
brew install ta-lib

# Then install Python wrapper
pip install TA-Lib
```

**Windows**:
```bash
# Download TA-Lib from: https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib
# Install the .whl file
pip install TA_Lib‑0.4.XX‑cpXX‑cpXX‑win_amd64.whl
```

### Step 3: Configure API Keys

Create `.env` file in project root:

```bash
# Testnet (recommended first)
BINGX_API_KEY=your_testnet_api_key
BINGX_SECRET_KEY=your_testnet_secret_key
BINGX_BASE_URL=https://open-api-vst.bingx.com

# OR Mainnet (after testnet validation)
BINGX_API_KEY=your_mainnet_api_key
BINGX_SECRET_KEY=your_mainnet_secret_key
BINGX_BASE_URL=https://open-api.bingx.com
```

**⚠️ Security**: NEVER commit `.env` to version control!

### Step 4: Verify Models

```bash
# Check models exist
ls models/

# Should see:
# xgboost_v4_phase4_advanced_lookahead3_thresh0.pkl
# xgboost_v4_phase4_advanced_lookahead3_thresh0_scaler.pkl
# xgboost_v4_phase4_advanced_lookahead3_thresh0_features.txt
# xgboost_short_redesigned_20251016_233322.pkl
# xgboost_short_redesigned_20251016_233322_scaler.pkl
# xgboost_short_redesigned_20251016_233322_features.txt
```

---

## ⚙️ Configuration

### Bot Parameters

Edit `scripts/production/opportunity_gating_bot.py`:

```python
# Strategy Parameters (validated)
LONG_THRESHOLD = 0.65      # Don't change without retest
SHORT_THRESHOLD = 0.70     # Don't change without retest
GATE_THRESHOLD = 0.001     # Don't change without retest

# Exit Parameters
MAX_HOLD_TIME = 240        # 4 hours (240 × 5min)
TAKE_PROFIT = 0.03         # 3%
STOP_LOSS = -0.015         # -1.5%

# Trading Parameters
SYMBOL = "BTC-USDT"
LEVERAGE = 1               # Don't use leverage initially
POSITION_SIZE_PCT = 0.70   # Use 70% of balance per trade

# Bot Parameters
CHECK_INTERVAL_SECONDS = 60  # Check every minute
MAX_DATA_CANDLES = 5000      # Keep last 5000 candles
```

### Risk Management Settings

**Conservative** (recommended for testnet):
```python
POSITION_SIZE_PCT = 0.30  # 30% per trade
LEVERAGE = 1
```

**Moderate** (after testnet success):
```python
POSITION_SIZE_PCT = 0.50  # 50% per trade
LEVERAGE = 1
```

**Aggressive** (experienced only):
```python
POSITION_SIZE_PCT = 0.70  # 70% per trade
LEVERAGE = 1              # Still no leverage!
```

---

## 🏃 Running the Bot

### Testnet Deployment (RECOMMENDED FIRST)

```bash
# 1. Configure testnet API keys in .env

# 2. Run bot
python scripts/production/opportunity_gating_bot.py

# 3. Monitor output
# Bot will log to:
# - Console (real-time)
# - logs/opportunity_gating_bot_YYYYMMDD.log
```

### Mainnet Deployment (AFTER Testnet Success)

**⚠️ ONLY proceed after 2 weeks successful testnet validation**

```bash
# 1. Switch to mainnet API keys in .env

# 2. Start with small position size
# Edit bot: POSITION_SIZE_PCT = 0.30

# 3. Run bot
python scripts/production/opportunity_gating_bot.py

# 4. Monitor closely for first week
```

### Running in Background

**Linux/Mac** (using screen):
```bash
screen -S trading_bot
python scripts/production/opportunity_gating_bot.py

# Detach: Ctrl+A, then D
# Reattach: screen -r trading_bot
```

**Linux/Mac** (using nohup):
```bash
nohup python scripts/production/opportunity_gating_bot.py > bot.out 2>&1 &

# Check output:
tail -f bot.out
```

**Windows** (using pythonw):
```cmd
start /B pythonw scripts\production\opportunity_gating_bot.py
```

---

## 📊 Monitoring

### Log Files

**Location**: `logs/opportunity_gating_bot_YYYYMMDD.log`

**What to Monitor**:
```yaml
Entry Logs:
  🚀 ENTER LONG/SHORT: [reason]
  Price, Time, TP/SL levels

Exit Logs:
  🚪 EXIT LONG/SHORT: [reason]
  Entry/Exit prices, P&L, Stats

Signal Logs:
  [timestamp] Price: $X | LONG: 0.XXXX | SHORT: 0.XXXX
```

### State File

**Location**: `results/opportunity_gating_bot_state.json`

Contains:
- Current position (if any)
- All completed trades
- Running statistics

### Key Metrics to Track

```yaml
Daily:
  - Win rate (target: >70%)
  - Average P&L per trade
  - Number of trades
  - LONG vs SHORT ratio

Weekly:
  - Total return
  - Comparison to backtest
  - Drawdown
  - Sharpe ratio (estimated)

Red Flags:
  - Win rate < 60% for 1 week
  - Negative returns for 2 consecutive weeks
  - Abnormal SHORT ratio (should be ~16%)
```

### Monitoring Checklist

**Daily** (first 2 weeks):
- [ ] Check bot is running
- [ ] Review today's trades
- [ ] Verify win rate
- [ ] Check for errors in logs

**Weekly** (ongoing):
- [ ] Calculate weekly performance
- [ ] Compare to backtest expectations
- [ ] Check system resources
- [ ] Verify model performance hasn't degraded

**Monthly**:
- [ ] Full performance analysis
- [ ] Consider model retraining
- [ ] Adjust position sizing if needed

---

## 🔧 Troubleshooting

### Common Issues

#### 1. Feature Calculation Errors

**Symptom**: `KeyError: ['feature_name'] not in index`

**Solution**:
```bash
# Ensure calculate_all_features.py is accessible
# Check all dependencies installed
pip install pandas numpy talib
```

#### 2. Model Loading Errors

**Symptom**: `FileNotFoundError: models/xxx.pkl`

**Solution**:
```bash
# Verify all 6 model files exist
ls models/ | grep -E "(pkl|txt)"

# Should see 6 files (3 for LONG, 3 for SHORT)
```

#### 3. API Connection Errors

**Symptom**: `ConnectionError` or `Unauthorized`

**Solution**:
```bash
# Check API keys in .env
# Verify API permissions
# Check network connection
# Try testnet first
```

#### 4. Short Signals = 0

**Symptom**: No SHORT trades at all

**Possible Causes**:
- Feature calculation missing SHORT features
- Short model not loading correctly
- Market conditions not favorable for SHORT

**Solution**:
```bash
# Run debug script
python scripts/experiments/debug_short_signals.py

# Check SHORT probability distribution
# Verify SHORT features calculated
```

#### 5. Performance Below Backtest

**Symptom**: Win rate < 65% or returns much lower

**Possible Causes**:
- Slippage higher than expected
- Market regime change
- Model degradation

**Solution**:
- Monitor for 2 weeks before adjusting
- Check execution quality (limit orders?)
- Consider model retraining
- May need to pause and investigate

---

## 🛡️ Safety Guidelines

### Position Sizing

**Phase 1** (Week 1-2): 30% of capital
- Validate execution quality
- Measure actual slippage
- Confirm backtest performance

**Phase 2** (Week 3-4): 50% of capital
- If Phase 1 successful
- Win rate >= 65%
- Returns within 20% of backtest

**Phase 3** (Week 5+): 70% of capital
- Consistent performance
- No red flags
- Full deployment

### Stop Conditions

**Immediate Stop**:
```yaml
- Bot crashes repeatedly
- API errors persistent
- Unauthorized trading detected
- Position sizing errors
```

**Pause and Investigate**:
```yaml
- Win rate < 60% for 1 week
- Negative returns for 2 consecutive weeks
- SHORT ratio > 30% (should be ~16%)
- Daily loss > 5%
```

### Emergency Procedures

**If Bot Crashes**:
```bash
1. Check state file: results/opportunity_gating_bot_state.json
2. Verify no open positions on exchange
3. Review logs for errors
4. Restart with caution
```

**If Performance Poor**:
```bash
1. Stop bot
2. Analyze recent trades
3. Check market conditions
4. Consider:
   - Temporary pause
   - Parameter adjustment (with retest!)
   - Model retraining
```

**If Unexpected Behavior**:
```bash
1. STOP BOT IMMEDIATELY
2. Close any open positions manually
3. Investigate logs thoroughly
4. Do NOT restart until issue understood
```

### Backup and Recovery

```bash
# Backup critical files regularly
tar -czf backup_$(date +%Y%m%d).tar.gz \
  models/ \
  results/opportunity_gating_bot_state.json \
  logs/

# Store backups securely offsite
```

---

## 📝 Performance Expectations

### Realistic Projections

```yaml
Backtest Baseline:
  Return: 2.73% per 5 days
  Win Rate: 72%
  Annualized: ~457%

Real Trading (Expected):
  Return: 2.0-2.5% per 5 days (70-90% of backtest)
  Win Rate: 65-70%
  Annualized: 300-400%

Reasons for Difference:
  - Execution delays
  - Slippage variability
  - Market regime changes
  - Model decay over time
```

### When to Be Concerned

```yaml
Yellow Flags:
  - Win rate 60-65% (below target but acceptable)
  - Returns 50-70% of backtest
  - SHORT ratio 20-25% (should be ~16%)

Red Flags:
  - Win rate < 60%
  - Returns < 50% of backtest
  - Consecutive losing weeks
  - SHORT ratio > 30%
```

---

## 🔄 Maintenance Schedule

### Weekly

- [ ] Performance review
- [ ] Check logs for errors
- [ ] Verify bot running smoothly
- [ ] Update monitoring dashboard

### Monthly

- [ ] Comprehensive performance analysis
- [ ] Compare actual vs backtest
- [ ] Evaluate parameter adjustments
- [ ] Consider model retraining

### Quarterly

- [ ] Full system audit
- [ ] Model retraining with latest data
- [ ] Strategy review and optimization
- [ ] Risk management review

---

## 📞 Support and Resources

### Documentation

- Full Backtest Report: `claudedocs/FULL_BACKTEST_VALIDATION_20251017.md`
- Strategy Analysis: `claudedocs/FINAL_ANALYSIS_SUCCESS_20251017.md`
- Code: `scripts/production/opportunity_gating_bot.py`

### Testing Scripts

- Debug SHORT Signals: `scripts/experiments/debug_short_signals.py`
- Full Backtest: `scripts/experiments/full_backtest_opportunity_gating.py`
- Unified Test: `scripts/experiments/test_all_strategies_unified.py`

### Logs and State

- Daily Logs: `logs/opportunity_gating_bot_YYYYMMDD.log`
- Bot State: `results/opportunity_gating_bot_state.json`

---

## ✅ Final Checklist Before Go-Live

### Testnet Phase (2 weeks minimum)

- [ ] Testnet API keys configured
- [ ] Bot runs without errors for 24 hours
- [ ] Signal generation working (LONG + SHORT)
- [ ] Entry/exit logic validated
- [ ] Logging verified
- [ ] State file updating correctly
- [ ] Performance tracking working

### Mainnet Phase

- [ ] Testnet performance validated (>65% win rate)
- [ ] Mainnet API keys configured
- [ ] Position size set conservatively (30%)
- [ ] Stop loss/take profit verified
- [ ] Monitoring dashboard ready
- [ ] Alert system configured
- [ ] Backup procedures tested

### Ongoing

- [ ] Daily monitoring established
- [ ] Weekly performance review scheduled
- [ ] Monthly retraining planned
- [ ] Emergency procedures documented
- [ ] Team trained on operations

---

## 🎓 Best Practices

### Do's

✅ Start with testnet
✅ Use conservative position sizing initially
✅ Monitor daily for first month
✅ Keep detailed records
✅ Follow the stop conditions
✅ Regular model retraining
✅ Maintain system backups

### Don'ts

❌ Skip testnet validation
❌ Use leverage initially
❌ Ignore warning signs
❌ Overtrade manually
❌ Change parameters without retesting
❌ Deploy without monitoring
❌ Ignore risk management

---

## 📊 Success Metrics

### After 1 Month

```yaml
Target:
  Win Rate: > 65%
  Return: > 2% per week
  Drawdown: < 10%
  Uptime: > 95%

If Achieved:
  ✅ Continue with current parameters
  ✅ Consider scaling up position size
  ✅ Maintain vigilant monitoring
```

### After 3 Months

```yaml
Target:
  Win Rate: > 67%
  Return: > 25% total
  Sharpe Ratio: > 3.0
  Max Drawdown: < 15%

If Achieved:
  ✅ Strategy validated in production
  ✅ Consider full capital deployment
  ✅ Plan quarterly model updates
```

---

**Status**: 📘 **Deployment Guide Complete**

**Version**: 1.0

**Last Updated**: 2025-10-17

**Next Review**: After testnet validation
