# Pre-Flight Checklist - Opportunity Gating Bot 4x
**Date**: 2025-10-17 04:00 KST
**Bot**: `opportunity_gating_bot_4x.py`
**Status**: ✅ **READY FOR TESTNET**

---

## ✅ Pre-Flight Checks Complete

### 1. Model Files ✅
```
✅ xgboost_v4_phase4_advanced_lookahead3_thresh0.pkl (1.1M)
✅ xgboost_v4_phase4_advanced_lookahead3_thresh0_scaler.pkl (2.3K)
✅ xgboost_v4_phase4_advanced_lookahead3_thresh0_features.txt (847 bytes)
✅ xgboost_short_redesigned_20251016_233322.pkl (546K)
✅ xgboost_short_redesigned_20251016_233322_scaler.pkl (1.4K)
✅ xgboost_short_redesigned_20251016_233322_features.txt (677 bytes)
```

### 2. Parameters Validation ✅
```yaml
Strategy:
  LONG_THRESHOLD: 0.65
  SHORT_THRESHOLD: 0.70
  GATE_THRESHOLD: 0.001

Leverage & Sizing:
  LEVERAGE: 4x
  Position Sizing: Dynamic (20-95%)
  Base: 50%, Max: 95%, Min: 20%

Exit:
  MAX_HOLD_TIME: 240 candles (4 hours)
  TAKE_PROFIT: 0.03 (3%)
  STOP_LOSS: -0.015 (-1.5%)

Models:
  LONG: xgboost_v4_phase4_advanced_lookahead3_thresh0.pkl
  SHORT: xgboost_short_redesigned_20251016_233322.pkl
```

### 3. API Configuration ✅
```yaml
Location: config/api_keys.yaml
API Key: NyXnyvNWK...Iamlg (configured)
Secret Key: XLi5Q6ljh...QTgceA (configured)
Mode: Testnet (USE_TESTNET = True)
```

### 4. Code Integration ✅
```python
✅ Imports: yaml, os added
✅ Config loading: load_api_keys() function
✅ BingXClient: Proper initialization with api_key, secret_key, testnet
✅ Balance handling: Fixed for BingXClient format
✅ Leverage setting: Both LONG and SHORT leverage set
✅ Error handling: API key validation
```

### 5. File Structure ✅
```
bingx_rl_trading_bot/
├── config/
│   └── api_keys.yaml ✅ (testnet keys configured)
├── models/ ✅ (all 6 files present)
├── scripts/production/
│   ├── opportunity_gating_bot_4x.py ✅ (integrated)
│   └── dynamic_position_sizing.py ✅ (existing)
├── scripts/experiments/
│   └── calculate_all_features.py ✅ (existing)
├── src/api/
│   └── bingx_client.py ✅ (existing)
└── logs/ ✅ (ready)
```

---

## 🎯 Expected Performance (from backtest)

```yaml
Performance:
  Return: 18.13% per window (5 days)
  Win Rate: 63.9%
  Trades: 18.5 per window
    - LONG: 15.7 (85%)
    - SHORT: 2.8 (15%)

Capital Growth:
  Initial: $10,000
  Final: $19,762 (105 days)
  Total Return: +97.6%

Position Sizing:
  Average: 51.4%
  Range: 20-95% (dynamic)
```

---

## 🚀 Testnet Deployment Plan

### Phase 1: Initial Run (1 Hour)
```bash
# Start bot
cd C:\Users\J\OneDrive\CLAUDE_CODE_FIN\bingx_rl_trading_bot
python scripts/production/opportunity_gating_bot_4x.py

# Monitor:
- Check connection successful
- Verify leverage set correctly
- Monitor signal generation (LONG/SHORT probs)
- Watch for first entry signal
```

### Phase 2: First Trade Validation (4-24 Hours)
```yaml
Validate:
  - Entry signal triggers correctly
  - Position size calculated properly (20-95%)
  - Leveraged position created (4x)
  - Exit signals work (TP/SL/Max Hold)

Monitor:
  - Trade execution
  - P&L calculation
  - State persistence
  - Error handling
```

### Phase 3: Week 1 Validation (7 Days)
```yaml
Success Criteria:
  - Win Rate: > 60%
  - Return: > 12% per window
  - No system errors
  - Leverage working correctly
  - Both LONG and SHORT trades executed

Expected Trades:
  - ~3.7 trades per day (18.5 / 5 days)
  - ~26 trades in 7 days
  - LONG: ~22 trades (85%)
  - SHORT: ~4 trades (15%)
```

---

## ⚠️ Risk Management

### Position Level
- Stop Loss: -1.5% (hard limit)
- Take Profit: +3% (lock profits)
- Max Hold: 4 hours (prevent long drawdowns)
- Dynamic sizing: 20-95% (adaptive risk)

### Account Level
- Max leverage: 4x (validated safe level)
- Position monitoring: Real-time
- Daily P&L tracking
- Alert if win rate < 55%

### System Level
- State persistence (no data loss)
- Error handling (robust)
- Logging (comprehensive)
- Manual override (emergency stop)

---

## 📋 Final Checklist Before Run

- [x] All 6 model files present
- [x] Parameters match backtest configuration
- [x] API keys configured in config/api_keys.yaml
- [x] Testnet mode enabled (USE_TESTNET = True)
- [x] BingXClient integrated correctly
- [x] Balance handling fixed
- [x] Leverage setting for both LONG/SHORT
- [x] State file location configured
- [x] Logging configured
- [ ] **User approval to start testnet run**

---

## 🎬 Ready to Launch

**Command**:
```bash
cd C:\Users\J\OneDrive\CLAUDE_CODE_FIN\bingx_rl_trading_bot
python scripts/production/opportunity_gating_bot_4x.py
```

**Log Location**:
```
logs/opportunity_gating_bot_4x_20251017.log
```

**State File**:
```
results/opportunity_gating_bot_4x_state.json
```

---

**Status**: 🟢 **ALL CHECKS PASSED - READY FOR TESTNET**

**Next Action**: User confirmation to start testnet execution
