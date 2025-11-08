# Quick Start Guide

## ⚡ CURRENT SYSTEM: Opportunity Gating + 4x Leverage

**Status**: ✅ Running on BingX Testnet (Week 1 Validation)
**Deployed**: 2025-10-17 04:08:36
**Strategy**: SHORT only when EV(SHORT) > EV(LONG) + 0.001 gate
**Expected**: 18.13% per 5-day window, 63.9% win rate

---

## 📊 For Complete Current Information

**👉 PRIMARY GUIDE**: [`SYSTEM_STATUS.md`](SYSTEM_STATUS.md)
- Current bot status and configuration
- Latest signal check and performance
- Monitoring commands and troubleshooting
- Expected performance and validation goals

**👉 DEPLOYMENT DETAILS**: [`claudedocs/OPPORTUNITY_GATING_DEPLOYMENT_20251017.md`](claudedocs/OPPORTUNITY_GATING_DEPLOYMENT_20251017.md)
- Complete deployment journey (04:02-04:10)
- Strategy explanation and innovation
- Backtest validation results
- Week 1 validation plan

**👉 MONITORING GUIDE**: [`claudedocs/QUANT_MONITOR_GUIDE.md`](claudedocs/QUANT_MONITOR_GUIDE.md)
- Professional monitoring dashboard
- Gate effectiveness tracking
- Alert thresholds and metrics

---

## 🚀 Quick Commands

### Check Bot Status
```bash
# View live log
tail -f logs/opportunity_gating_bot_4x_20251017.log

# Check if running (Windows)
tasklist | findstr python

# Check latest signals
grep "LONG: .* | SHORT:" logs/opportunity_gating_bot_4x_20251017.log | tail -10
```

### Monitor Trades
```bash
# Entry signals
grep "ENTER" logs/opportunity_gating_bot_4x_20251017.log

# Exit signals
grep "EXIT" logs/opportunity_gating_bot_4x_20251017.log

# Gate blocks (SHORT prevented by gate)
grep "blocked by gate" logs/opportunity_gating_bot_4x_20251017.log
```

---

## 📜 Historical Documentation (Phase 4 System)

**Note**: The information below describes the previous Phase 4 system (October 10-16, 2025).
This has been superseded by the **Opportunity Gating + 4x Leverage** system deployed on October 17, 2025.

<details>
<summary>Click to view Phase 4 historical guide</summary>

### Phase 4 Base Production Deployment (Historical)

**Previous Problem:** Production bot was using **Phase 2 model** (33 features, 0.75% expected)
**Solution:** Upgraded to **Phase 4 Base model** (37 features, 7.68% expected)
**Impact:** **+920% performance improvement!**

**Phase 2 실제 성능:**
- XGBoost probability: 0.2-0.5 (threshold 0.7 이하!)
- Result: **No trades** (signal too weak)
- Expected: 0.75% per 5 days

**Phase 4 Base 예상 성능:**
- XGBoost probability: 0.7+ (더 강한 signal)
- Expected: 7.68% per 5 days
- Win rate: 69.1%
- Trades: ~15 per 5 days

</details>

---

## ⚡ 즉시 실행 (3 Steps)

### Step 1: Bot Restart (Choose One)

**Windows:**
```cmd
cd C:\Users\J\OneDrive\CLAUDE_CODE_FIN\bingx_rl_trading_bot
scripts\restart_production_bot.bat
```

**Linux/Mac:**
```bash
cd C:\Users\J\OneDrive\CLAUDE_CODE_FIN\bingx_rl_trading_bot
bash scripts/restart_production_bot.sh
```

**Manual (if scripts fail):**
```bash
# 1. Stop current bot
pkill -f sweet2_paper_trading

# 2. Start new bot
cd C:\Users\J\OneDrive\CLAUDE_CODE_FIN\bingx_rl_trading_bot
python scripts/production/sweet2_paper_trading.py
```

### Step 2: Verify Phase 4 Base Loaded

**Check logs for:**
```
✅ XGBoost Phase 4 Base model loaded: 37 features
✅ Advanced Technical Features initialized
```

**Command:**
```bash
tail -20 logs/sweet2_paper_trading_*.log | grep "Phase 4 Base"
```

### Step 3: Monitor First Trades

**Monitor log (real-time):**
```bash
tail -f logs/sweet2_paper_trading_*.log | grep -E "ENTRY|EXIT|XGBoost Prob"
```

**Expected:**
- XGBoost Prob > 0.7 for entries
- 2-3 trades per day
- Win rate > 60%

---

## 📊 Performance Comparison

### Phase 2 (Current - Old Model)
```yaml
Model: xgboost_v3_lookahead3_thresh1_phase2.pkl
Features: 33
Training F1: 0.054

Expected Performance:
  - Returns: 0.75% per 5 days
  - Win rate: 54.3%
  - Trades: ~5 per 5 days

Actual Performance (Today):
  - XGBoost Prob: 0.2-0.5 (< 0.7 threshold)
  - Trades: 0 (No trades!)
  - Reason: Signal too weak
```

### Phase 4 Base (New - Best Model)
```yaml
Model: xgboost_v4_phase4_advanced_lookahead3_thresh0.pkl
Features: 37 (10 baseline + 27 advanced)
Training F1: 0.089 (+65% vs Phase 2)

Expected Performance:
  - Returns: 7.68% per 5 days (+920% vs Phase 2!)
  - Win rate: 69.1%
  - Trades: ~15 per 5 days
  - Sharpe: 11.88
  - Max DD: 0.90%

Statistical Validation:
  - n=29 windows (2-day)
  - Bootstrap 95% CI: [0.67%, 1.84%]
  - Effect size: d=0.606 (large)
  - Power: 88.3%
  - Verdict: CONFIDENT ✅
```

**Improvement: +920% returns, +15 F1 points, +14.8% win rate**

---

## 📈 24-Hour Monitoring Plan

### Hour 0-4: First Check
```yaml
Check:
  - ✅ Phase 4 Base loaded (37 features)
  - ✅ XGBoost probabilities calculated
  - ✅ Advanced features working
  - ⏳ Wait for first trade

Expected:
  - First trade within 4-8 hours
  - XGBoost Prob > 0.7 for entry
```

### Hour 4-12: Early Trades
```yaml
Check:
  - 1-2 trades executed
  - Win rate tracking starts
  - No errors in logs

Red Flags:
  - No trades after 12 hours → Check threshold (try 0.6)
  - Multiple losses → Review strategy
  - Errors in feature calculation → Check logs
```

### Hour 12-24: First Day Complete
```yaml
Expected (Day 1):
  - Trades: 2-4
  - Win rate: >60%
  - Returns: ~0.25% (7.68% / 30 days)
  - Max DD: <1%

Success Criteria:
  - At least 2 trades
  - Win rate >55%
  - Positive returns
  - No drawdown >1.5%
```

---

## 🔍 Troubleshooting

### Issue 1: Bot Not Starting
```yaml
Symptoms:
  - Script exits immediately
  - No log file created

Solutions:
  1. Check model file exists:
     ls -lh models/xgboost_v4_phase4_advanced_lookahead3_thresh0.pkl

  2. Check Python environment:
     python --version  # Should be 3.9+

  3. Check dependencies:
     pip install -r requirements.txt

  4. Run with debug:
     python scripts/production/sweet2_paper_trading.py 2>&1 | tee debug.log
```

### Issue 2: Still Using Phase 2
```yaml
Symptoms:
  - Log shows "Phase 2 model loaded"
  - Only 33 features loaded

Solutions:
  1. Check code was updated:
     grep "Phase 4 Base" scripts/production/sweet2_paper_trading.py

  2. Hard restart:
     pkill -9 -f sweet2_paper_trading
     sleep 5
     python scripts/production/sweet2_paper_trading.py

  3. Verify model path in code (line 123-125):
     Should be: xgboost_v4_phase4_advanced_lookahead3_thresh0.pkl
```

### Issue 3: No Trades After 12 Hours
```yaml
Symptoms:
  - XGBoost Prob always < 0.7
  - "No trades yet" in logs

Diagnosis:
  1. Check probabilities:
     grep "XGBoost Prob" logs/sweet2_paper_trading_*.log | tail -20

  2. If all < 0.7 → Market not suitable OR threshold too high

Solutions:
  1. Lower threshold to 0.6 (edit sweet2_paper_trading.py line 66):
     XGB_THRESHOLD_STRONG = 0.6  # was 0.7

  2. Restart bot

  3. Monitor: Should see entries with Prob 0.6-0.7
```

### Issue 4: High Drawdown
```yaml
Symptoms:
  - Multiple consecutive losses
  - Drawdown > 2%

Actions:
  1. STOP bot immediately:
     pkill -f sweet2_paper_trading

  2. Review trades:
     cat results/sweet2_paper_trading_trades_*.csv

  3. Check market regime:
     grep "Market Regime" logs/sweet2_paper_trading_*.log | tail -20

  4. If Bear market → Increase threshold to 0.8
```

---

## 📊 Daily Monitoring Commands

### Quick Status Check
```bash
# Check if running
ps aux | grep sweet2_paper_trading

# Check recent activity
tail -50 logs/sweet2_paper_trading_*.log

# Check probabilities
grep "XGBoost Prob" logs/sweet2_paper_trading_*.log | tail -10

# Check for trades
grep -E "ENTRY|EXIT" logs/sweet2_paper_trading_*.log
```

### Performance Metrics
```bash
# Check latest performance summary
grep -A 20 "📊 SWEET-2 PERFORMANCE" logs/sweet2_paper_trading_*.log | tail -25

# Check trade count
grep "Total Trades" logs/sweet2_paper_trading_*.log | tail -1

# Check win rate
grep "Win Rate" logs/sweet2_paper_trading_*.log | tail -1

# Check returns
grep "vs B&H" logs/sweet2_paper_trading_*.log | tail -1
```

### Detailed Analysis
```bash
# View all trades (CSV)
cat results/sweet2_paper_trading_trades_*.csv

# View market regime history
cat results/sweet2_market_regime_history_*.csv

# View bot state
cat results/sweet2_paper_trading_state.json
```

---

## 🎯 Success Criteria

### Week 1 Targets
```yaml
Minimum Success (Continue):
  - Trades: ≥14 (2 per day × 7)
  - Win rate: ≥60%
  - Returns: ≥1.2% (70% of expected 1.75%)
  - Max DD: <2%

Good Performance (Confident):
  - Trades: ≥21 (3 per day × 7)
  - Win rate: ≥65%
  - Returns: ≥1.5% (85% of expected)
  - Max DD: <1.5%

Excellent (Beat Expectations):
  - Trades: ≥28 (4 per day × 7)
  - Win rate: ≥68%
  - Returns: ≥1.75% (100% of expected)
  - Max DD: <1%
```

### Decision Matrix (End of Week 1)
```yaml
If Excellent or Good:
  ✅ Continue with Phase 4 Base
  ✅ Start LSTM development planning
  ✅ Collect data for 6 months (LSTM training)

If Minimum Success:
  ⚠️ Continue but investigate
  - Analyze losing trades
  - Check market regimes
  - Consider threshold adjustment (0.6-0.8)

If Below Minimum (<60% win rate OR <1.2% returns):
  🔴 Stop and deep dive
  - Full trade analysis
  - Model validation on new data
  - Check for model drift
  - Consider retraining
```

---

## 📋 Files Reference

### Models
- **Production:** `models/xgboost_v4_phase4_advanced_lookahead3_thresh0.pkl` (37 features) ✅
- **Old:** `models/xgboost_v3_lookahead3_thresh1_phase2.pkl` (33 features) ❌

### Scripts
- **Restart (Windows):** `scripts/restart_production_bot.bat`
- **Restart (Linux/Mac):** `scripts/restart_production_bot.sh`
- **Bot:** `scripts/production/sweet2_paper_trading.py`

### Logs & Results
- **Logs:** `logs/sweet2_paper_trading_*.log`
- **Trades:** `results/sweet2_paper_trading_trades_*.csv`
- **State:** `results/sweet2_paper_trading_state.json`

### Documentation
1. **This guide:** `QUICK_START_GUIDE.md`
2. **Full analysis:** `claudedocs/FINAL_SUMMARY_AND_NEXT_STEPS.md`
3. **Lag features:** `claudedocs/LAG_FEATURES_ROOT_CAUSE_ANALYSIS.md`
4. **Production plan:** `claudedocs/PRODUCTION_DEPLOYMENT_PLAN.md`

---

## 🚀 Next Steps After Week 1

### If Performance Good (≥70% of expected)
```yaml
Month 1:
  - Continue Phase 4 Base production
  - Collect 30 days data
  - Monthly retraining

Month 2-4:
  - LSTM development
  - Collect 6 months data (50K+ candles)
  - Train LSTM model
  - Expected: 8-10% (LSTM alone)

Month 5-6:
  - Ensemble (XGBoost + LSTM)
  - Expected: 10-12%+
  - Production deployment
```

### If Performance Needs Improvement (50-70%)
```yaml
Immediate:
  - Threshold optimization (test 0.6, 0.65, 0.7, 0.75, 0.8)
  - Regime-specific thresholds
  - Feature importance analysis

Week 2-3:
  - Retrain with new data
  - Validate on holdout
  - A/B test old vs new model

Week 4:
  - Deploy best performing configuration
```

---

## 🎯 비판적 사고 최종 정리

**발견한 Critical Issue:**
1. ❌ Production bot이 Phase 2 사용 중 (0.75% expected)
2. ❌ XGBoost Prob 0.2-0.5 (< 0.7 threshold) → **No trades!**
3. ✅ Phase 4 Base 준비 완료 (7.68% expected, +920%)

**즉시 실행 필요:**
1. 🚨 **Bot 재시작** (Phase 2 → Phase 4 Base)
2. 📊 **24시간 모니터링** 시작
3. ✅ **Week 1 validation** 진행

**Long-term Plan:**
- Month 1-2: Phase 4 Base production
- Month 2-4: LSTM development (10-12% expected)
- Month 5-6: Ensemble deployment

**Confidence: HIGH** ✅
**Action Required: Bot Restart** 🚨
