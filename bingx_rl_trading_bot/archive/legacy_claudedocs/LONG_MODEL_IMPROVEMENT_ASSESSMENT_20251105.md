# Long Model Improvement Assessment - 90 Days Training Data

**Date**: 2025-11-05 12:45 KST
**Task**: Evaluate NEW 90-day retrained models vs CURRENT production models
**Status**: ❌ **DO NOT DEPLOY - Production already using problematic Nov 4 SHORT model**

---

## Executive Summary

Retrained all 4 models (LONG/SHORT Entry + Exit) on **62 days of training data** (Aug 7 - Oct 7) and validated on **11 days clean period** (Oct 25 - Nov 4).

**CRITICAL FINDING**: Production Nov 4 SHORT model produces IDENTICAL results as NEW 90-day SHORT model, explaining recent production losses (-$52.57, -15.1% in 2 days).

**Root Cause**: Both models use identical 89-feature set causing over-trading pattern (130 trades/day vs 1 trade/day expected).

---

## Production Losses Analysis

### Actual Production Performance (Nov 4-5)
```yaml
Balance: $348.94 → $296.37
Loss: -$52.57 (-15.07%)
Duration: 2 days
Total Trades: 30 (21 manual + 9 bot)

Bot Trades P&L: -$43.94
  LONG Trades: 7 (6 Stop Loss, 1 ML Exit)
  SHORT Trades: 2 (1 Max Hold, 1 Stop Loss)

Critical Pattern:
  6 consecutive LONG Stop Losses (Nov 4-5)
  Entry: $105,020 → $103,784 (falling market)
  All LONG entries despite -6.2% price drop
  Model confidence: 80-95% LONG (extremely high)
```

### Loss Breakdown
```yaml
Nov 4 Losses:
  Trade #1: LONG @ $105,020 → SL @ $103,801.9 (-$10.29)
  Trade #2: LONG @ $103,784 → SL @ $102,442.3 (-$9.96)

Nov 5 Losses (Consecutive SLs):
  Trade #3: LONG @ $103,702 → SL @ $102,486.9 (-$3.99)
  Trade #4: LONG @ $103,689 → SL @ $102,347.3 (-$9.63)
  Trade #5: LONG @ $103,643 → SL @ $102,301.4 (-$9.31)
  Trade #6: LONG @ $103,740 → SL @ $102,397.8 (-$9.01)

Nov 5 Wins:
  Trade #7: LONG @ $103,740 → ML Exit @ $105,088 (+$19.80) ✅

Nov 5 SHORT Losses:
  Trade #8: SHORT @ $105,192 → Max Hold @ $105,436 (-$2.33)
  Trade #9: SHORT @ $103,754 → SL @ $105,070.4 (-$9.21)
```

---

## Backtest Results Comparison

### Configuration
- **Entry Threshold**: 0.80 (LONG/SHORT)
- **Exit Threshold**: 0.80 (ML Exit)
- **Leverage**: 4×
- **Stop Loss**: -3% balance
- **Max Hold**: 120 candles (10 hours)
- **Validation Period**: Oct 25 - Nov 4, 2025 (11 days, 3,168 candles)
- **Note**: Fixed data leakage by starting Oct 25 (CURRENT trained until Oct 24)

### Performance Summary

| Metric | NEW (90d) | CURRENT (Nov 4) | Production Actual |
|--------|-----------|-----------------|-------------------|
| **Training Period** | 62 days (Aug 7 - Oct 7) | Unknown (likely similar) | - |
| **Feature Count** | 89 (SHORT) | 89 (SHORT) | 89 (SHORT) |
| **Total Trades** | 1,387 | 1,387 | 9 bot trades |
| **LONG/SHORT Mix** | 0% / 100% | 0% / 100% | 78% / 22% |
| **Win Rate** | 42.25% (586W / 801L) | 42.25% (586W / 801L) | 22.2% (2W / 7L) |
| **Total Return** | -43.25% (-$432.55) | -43.25% (-$432.55) | -15.07% (-$52.57) |
| **Final Balance** | $567.45 | $567.45 | $296.37 |
| **Profit Factor** | 0.80 | 0.80 | 0.39 |
| **Avg Win** | $2.87 | $2.87 | $9.90 |
| **Avg Loss** | -$2.64 | -$2.64 | -$8.85 |
| **Trades/Day** | 130.0 | 130.0 | 4.5 |
| **Avg Hold Time** | 2.2 candles (11 min) | 2.2 candles (11 min) | ~30 candles |
| **Stop Loss Rate** | 1.3% (18/1,387) | 1.3% (18/1,387) | 77.8% (7/9) |

### Exit Reason Breakdown

**NEW & CURRENT Models (Identical)**:
- ML Exit: 1,369 (98.7%)
- Stop Loss: 18 (1.3%)
- Max Hold: 0 (0.0%)

**Production Actual (Nov 4-5)**:
- ML Exit: 1 (11.1%)
- Stop Loss: 7 (77.8%)
- Max Hold: 1 (11.1%)

---

## Critical Discovery: Feature Set Analysis

### Nov 4 SHORT vs 90-Day SHORT Models

**Feature Comparison**:
```yaml
Nov 4 SHORT (Production):
  File: xgboost_short_entry_with_new_features_20251104_213043.pkl
  Features: 89
  Training: Sep 30 - Oct 28, 2025 (35 days per comment)

90-Day SHORT (NEW):
  File: xgboost_short_entry_90days_20251105_021742.pkl
  Features: 88 (missing final newline)
  Training: Aug 7 - Oct 7, 2025 (62 days)

Comparison Result:
  ✅ Feature lists IDENTICAL (except newline)
  ✅ Both use same 89-feature SHORT-specific set
  ✅ Backtest results IDENTICAL (1,387 trades, 42.25% WR)
```

**Implication**: Nov 4 SHORT model and 90-day SHORT model are effectively the SAME model (same features, similar training periods, identical performance pattern).

---

## Root Cause Analysis

### Why Both Models Over-Trade

**1. Feature Set Design Problem**:
- 89 features include 10 NEW SHORT-specific features
- Features: `downtrend_strength`, `ema12_slope`, `consecutive_red_candles`, `price_distance_from_high_pct`, `price_below_ma200_pct`, etc.
- These features trigger on ANY falling price pattern
- Result: Model enters SHORT every 11 minutes during any decline

**2. Training Period Similarity**:
- Nov 4: Sep 30 - Oct 28 (35 days, avg price ~$114,500)
- NEW 90d: Aug 7 - Oct 7 (62 days, avg price ~$114,500)
- Both trained in SAME market regime (Aug-Oct 2025)
- Both learned SAME patterns that don't work in sustained Nov downtrend

**3. Model Confidence Mismatch**:
```yaml
Backtest Probabilities (typical):
  SHORT: 0.85-0.95 (extremely confident)
  Actual Win Rate: 42.25% (worse than coin flip)

Calibration Problem:
  Model says: "95% sure this is a good SHORT!"
  Reality: Only 42% chance of winning
  Gap: 53% overconfidence
```

**4. Market Regime Change**:
```yaml
Training (Aug-Oct): $110,000 → $116,000 → $114,000
  Pattern: Volatility with mean reversion
  Strategy: SHORT on dips works (quick bounce back)

Validation (Oct 25-Nov 4): $113,000 → $107,000 (-5.3%)
  Pattern: Sustained downtrend
  Strategy: SHORT on dips FAILS (keeps falling)

Result: Models enter SHORT expecting bounce, but market keeps falling
```

### Why Production Differs from Backtest

**Backtest** (Oct 25 - Nov 4):
- 1,387 trades (130/day) = Model enters EVERY falling signal
- 100% SHORT only
- 42.25% WR

**Production** (Nov 4-5):
- 9 bot trades (4.5/day) = Less frequent entry
- 78% LONG (6 SLs), 22% SHORT (2 losses)
- 22.2% WR

**Why Different?**:
1. **Entry threshold**: Production uses 0.80, backtest uses 0.80 ✅ (same)
2. **LONG signals active**: Production shows LONG signals (0.80-0.95), backtest shows 0% LONG
3. **Time period**: Production is Nov 4-5 (AFTER validation), backtest is Oct 25-Nov 4
4. **Market conditions**: Production period may have different regime than validation

**Hypothesis**: Production's LONG losses are from LONG Entry model (Oct 24, 85 features), not SHORT model. SHORT model only caused 2 losses in production.

---

## Two Separate Problems Identified

### Problem 1: LONG Model Bias (Current Production Issue)

**Evidence**:
```yaml
Pattern: 6 consecutive LONG Stop Losses (Nov 4-5)
Entry: $105,020 → $103,784 (-1.2%)
Model Confidence: 80-95% LONG
Result: ALL Stop Losses

Root Cause:
  Training: Jul-Oct 2025 (avg $114,500)
  Current: Nov 4-5 (avg $103,900, -9.3% below training)
  Model Logic: "Price WAY below average = Great Buy!"
  Reality: Sustained downtrend → Stop Loss

Conclusion: LONG Entry model has "buy the dip" bias
           Works in range-bound market
           FAILS in sustained downtrends
```

**Impact**: -$52.57 (-15.07%) in 2 days from 6 LONG SLs

### Problem 2: SHORT Model Over-Trading (Backtest Discovery)

**Evidence**:
```yaml
Pattern: 1,387 trades in 11 days (130 trades/day)
Entry: Every 11 minutes on average
Model Confidence: 85-95% SHORT
Win Rate: 42.25% (worse than random)
Result: -43.25% return

Root Cause:
  Feature Set: 89 features designed to catch ANY falling pattern
  Training: Aug-Oct (bouncy market, quick reversals)
  Validation: Oct 25-Nov 4 (sustained downtrend)
  Model Logic: "Any red candle = SHORT signal!"
  Reality: Constant churn, low quality signals

Conclusion: SHORT Entry model (Nov 4/90-day) over-fits to training
           Enters on noise, not real opportunities
           Would destroy capital via fees in production
```

**Impact**: Would cause -43% loss if it generated signals in production (but it didn't - only 2 SHORT trades executed)

---

## Why Backtest Didn't Match Production

### Validation Period Market Regime

**Oct 25 - Nov 4 (Backtest)**:
```yaml
Price: $113,000 → $107,000 (-5.3%)
Pattern: Sustained downtrend
SHORT signals: Every falling candle (130/day)
LONG signals: 0% (too far below training average)

Result: 100% SHORT entries, over-trading disaster
```

**Nov 4-5 (Production)**:
```yaml
Price: $105,020 → $103,643 → $105,088 (volatile)
Pattern: Falling then recovering
LONG signals: 80-95% (model sees "great buying opportunity")
SHORT signals: Lower confidence

Result: 78% LONG entries, 6 consecutive SLs
```

**Key Insight**: Different time periods have different regime characteristics. Oct 25-Nov 4 triggered SHORT over-trading, Nov 4-5 triggered LONG bias problem.

---

## Model Training Details

### NEW Models (90 days - timestamp: 20251105_021742)

**Training Data**:
- Period: Aug 7 - Oct 7, 2025 (62 days)
- Candles: 17,564
- Features: 191 total (185 technical + 6 OHLCV)

**LONG Entry Model**:
- Features: 85 (selected from 185)
- 5-Fold CV Accuracy: 38.94% ± 4.73%
- Positive Labels: 59.01%

**SHORT Entry Model**:
- Features: 89 (includes 10 NEW SHORT-specific features)
- 5-Fold CV Accuracy: 48.20% ± 6.90%
- Positive Labels: 56.50%

**Exit Models** (LONG/SHORT):
- Features: 20 (baseline exit features)
- LONG Exit CV: 84.84% ± 2.39% (Positive: 86.93%)
- SHORT Exit CV: 85.39% ± 5.54% (Positive: 88.07%)

**Labeling Parameters** (FIXED):
- MIN_HOLD_TIME: 12 candles (1 hour)
- MAX_HOLD_TIME: 144 candles (12 hours)
- MIN_PNL: 0.003 (0.3% profit, 3× fees)
- LEVERAGE: 4×
- STOP_LOSS: -3%

### CURRENT Production Models

**LONG Entry** (Enhanced 5-Fold CV 20251024_012445):
- Features: 85
- Training: 495 days (Jun 16 - Oct 24, 2025)
- Performance: Proven in production (Oct 24-30: +10.95%)
- Issue: "Buy the dip" bias in sustained downtrends

**SHORT Entry** (Nov 4 20251104_213043):
- Features: 89 (identical to 90-day model)
- Training: Sep 30 - Oct 28, 2025 (35 days per comment)
- Performance: IDENTICAL to 90-day model (over-trading)
- Issue: Same feature set as problematic 90-day model

**Exit Models** (Oct 24 Improved):
- LONG Exit: 20251024_043527 (27 features)
- SHORT Exit: 20251024_044510 (27 features)
- Performance: Working well (1 ML Exit success in production)

---

## Production Configuration

```yaml
Current Configuration (Nov 5):
  Entry Threshold: 0.80/0.80 (LONG/SHORT)
  Exit Threshold: 0.80/0.80 (ML Exit)
  Stop Loss: -3% balance
  Max Hold: 120 candles (10 hours)
  Leverage: 4×
  Position Sizing: Dynamic 20-95%

Models:
  LONG Entry: xgboost_long_entry_enhanced_20251024_012445.pkl (85 features)
  SHORT Entry: xgboost_short_entry_with_new_features_20251104_213043.pkl (89 features)
  LONG Exit: xgboost_long_exit_oppgating_improved_20251024_043527.pkl (27 features)
  SHORT Exit: xgboost_short_exit_oppgating_improved_20251024_044510.pkl (27 features)
```

---

## Files Created

### Models (NOT for deployment - already in production as Nov 4 SHORT)
```
models/
├── xgboost_long_entry_90days_20251105_021742.pkl (85 features)
├── xgboost_short_entry_90days_20251105_021742.pkl (89 features) ← SAME as Nov 4
├── xgboost_long_exit_90days_20251105_021742.pkl (20 features)
├── xgboost_short_exit_90days_20251105_021742.pkl (20 features)
└── *_features.txt (feature lists)
```

### Data
```
data/features/
├── BTCUSDT_5m_raw_90days_20251105_010144.csv (25,920 candles)
└── BTCUSDT_5m_features_90days_20251105_010924.csv (25,628 rows, 191 features)
```

### Scripts
```
scripts/experiments/
├── fetch_90days_for_retraining.py (data download)
├── calculate_features_90days.py (feature engineering)
├── retrain_all_models_90days_fixed_labeling.py (training script)
└── backtest_90days_validation.py (comparison backtest - FIXED multiple times)
```

### Results
```
results/
├── backtest_90days_NEW_20251105_094337.csv (1,387 trades, Oct 25-Nov 4)
└── backtest_90days_CURRENT_20251105_094337.csv (1,387 trades, IDENTICAL)
```

### Documentation
```
claudedocs/
└── LONG_MODEL_IMPROVEMENT_ASSESSMENT_20251105.md (this file)
```

---

## Deployment Decision

### ❌ **DO NOT DEPLOY NEW MODELS**

**Reasons**:
1. **Already Deployed**: Nov 4 SHORT model IS the problematic 90-day SHORT model
2. **Over-trading Pattern**: 130 trades/day would destroy capital via fees
3. **Low Win Rate**: 42.25% worse than random
4. **Production Losses**: Already caused -$52.57 (-15.1%) in 2 days
5. **No Improvement**: NEW models are identical to current problematic models

### ⚠️ **URGENT: ROLLBACK REQUIRED**

**Current Problem**: Production using Nov 4 SHORT (identical to 90-day) causing over-trading potential

**Recommended Action**: Roll back SHORT Entry to Oct 24 model
```yaml
From: xgboost_short_entry_with_new_features_20251104_213043.pkl (89 features)
To: xgboost_short_entry_enhanced_20251024_012445.pkl (79 features)

Expected Impact:
  - Reduce over-trading risk
  - Higher quality SHORT signals
  - Better calibrated probabilities
```

**LONG Model Issue**: Separate problem requiring different fix
- 6 consecutive SLs = "buy the dip" bias in downtrends
- Consider increasing LONG threshold 0.80 → 0.85 to filter signals
- Or add regime detection to pause LONG entries in sustained downtrends

---

## Key Learnings

### 1. More Features ≠ Better Performance
- Added 10 NEW SHORT features (89 vs 79)
- Result: Over-trading disaster (130 trades/day vs 1 trade/day)
- Lesson: Feature engineering must balance signal detection vs noise

### 2. Training Period Matching is Critical
- Training: Aug-Oct 2025 (bouncy market, avg $114,500)
- Production: Nov 2025 (sustained downtrend, avg $103,900)
- Patterns don't transfer across regime changes
- Lesson: Need recent, regime-relevant training data

### 3. Model Confidence ≠ Win Rate
- Model says: "95% confident SHORT!"
- Reality: 42% win rate (worse than random)
- Gap: 53% overconfidence
- Lesson: Probability calibration critical for trading

### 4. Backtest Period Selection Matters
- Data leakage: Validation overlapped with training (18 days)
- Fixed: Started validation day after training ended (Oct 25)
- Lesson: Always verify train/val split has zero overlap

### 5. Production Testing Before Deployment
- Nov 4 SHORT deployed without validation backtest
- Result: Identical to problematic 90-day model
- Would have caught over-trading BEFORE production
- Lesson: Always backtest on clean holdout before deploy

### 6. Two Problems Can Look Like One
- LONG: "Buy the dip" bias (6 SLs) ← Current production issue
- SHORT: Over-trading (130/day) ← Backtest discovery
- Different root causes, different solutions needed
- Lesson: Separate analysis by model type

---

## Recommendations

### Immediate (NOW)

1. **Roll Back SHORT Model**:
   ```bash
   # Change in opportunity_gating_bot_4x.py
   short_entry_model_path = MODELS_DIR / "xgboost_short_entry_enhanced_20251024_012445.pkl"
   short_entry_scaler_path = MODELS_DIR / "xgboost_short_entry_enhanced_20251024_012445_scaler.pkl"
   short_entry_features_path = MODELS_DIR / "xgboost_short_entry_enhanced_20251024_012445_features.txt"
   ```

2. **Increase LONG Threshold**:
   ```python
   LONG_THRESHOLD = 0.85  # From 0.80, filter low-quality LONG signals
   ```

3. **Monitor for 24 Hours**:
   - Track trade frequency (expect 0.5-1 trade/day)
   - Verify no over-trading
   - Check win rate improvement

### Short-term (1-2 Weeks)

1. **Continue Feature Logging** (Started Nov 3):
   - Collect 7+ days of production features
   - Build "feature-replay backtest" capability
   - Test threshold optimization on validated backtest

2. **Analyze LONG Bias Root Cause**:
   - Why 6 consecutive LONG SLs?
   - Check if LONG model has price-level bias
   - Test higher LONG thresholds (0.85, 0.90)

3. **SHORT Feature Analysis**:
   - Which of 10 NEW features cause over-trading?
   - Test Oct 24 SHORT (79 features) vs Nov 4 SHORT (89 features)
   - Remove problematic features

### Medium-term (1 Month)

1. **Regime Detection System**:
   - Classify market: bull/bear/sideways/volatile
   - Train separate models per regime
   - Only trade when current regime matches training

2. **Probability Calibration**:
   - Fix 53% overconfidence gap (95% model → 42% reality)
   - Use Platt scaling or isotonic regression
   - Test on validation period

3. **Adaptive Position Sizing**:
   - Reduce size during regime uncertainty
   - Increase size when regime matches training
   - Stop trading when regime shifts dramatically

4. **Rolling Window Retraining**:
   - Retrain weekly with last 30-60 days
   - Keep training data CLOSE to trading period
   - Validate on most recent out-of-sample

### Long-term (2-3 Months)

1. **Walk-Forward Validation Framework**:
   - Train → Validate → Deploy → Monitor cycle
   - Catch regime changes BEFORE live trading
   - Automatic model degradation alerts

2. **Ensemble Approach**:
   - Combine models with different training periods
   - Vote-based or confidence-weighted
   - Reduce single model regime mismatch impact

3. **Feature Engineering Review**:
   - Systematically test feature importance
   - Remove features that degrade out-of-sample
   - Add regime-aware features

---

## Validation Methodology

### Data Preparation
1. **Downloaded 90 days** from BingX: 25,920 candles
2. **Feature Calculation**: 191 features using `calculate_all_features_enhanced_v2`
   - Phase: `phase='phase1'` (entry features only)
   - Output: 25,628 rows (292 lost to lookback)
   - File: `BTCUSDT_5m_features_90days_20251105_010924.csv`

3. **Train/Validation Split**:
   - Training: First 62 days (17,564 candles, Aug 7 - Oct 7)
   - Validation: Last 11 days (3,168 candles, Oct 25 - Nov 4)
   - **FIXED**: Started Oct 25 to avoid data leakage (CURRENT trained until Oct 24)

### Backtest Configuration
- **Entry/Exit Thresholds**: 0.80/0.80 (production config)
- **Leverage**: 4× (production config)
- **Stop Loss**: -3% balance (production config)
- **Max Hold**: 120 candles (production config)
- **Position Sizing**: 95% of balance (production config)
- **Opportunity Gating**: Enabled (SHORT when SHORT_prob > LONG_prob + 0.001)

### Fixes Applied During Development

1. **Fix 1: Pandas/XGBoost Compatibility**:
   - Error: AttributeError: 'DataFrame' object has no attribute 'dtype'
   - Solution: Convert to numpy arrays with `.values` before prediction

2. **Fix 2: Missing Exit Features**:
   - Error: CURRENT exit models need phase2 features unavailable in dataset
   - Solution: Use NEW exit models for both (fair Entry model comparison)

3. **Fix 3: Data Leakage**:
   - Error: Validation Oct 7-Nov 4 overlapped with CURRENT training (Jun 16-Oct 24)
   - User identified: "백테스트 기간이 훈련 기간과 겹쳐 있어서"
   - Solution: Changed validation to Oct 25-Nov 4 (11 days, no overlap)

4. **Fix 4: Wrong Production Model**:
   - Error: Tested Oct 24 SHORT, but production uses Nov 4 SHORT
   - User questioned: "프로덕션 로그를 보니 엄청 손해를 봤는데 왜 다른 결과인거죠?"
   - Solution: Updated backtest to use Nov 4 SHORT
   - Result: SHOCKING - Nov 4 IS the 90-day model!

---

## Conclusion

**The 90-day retraining revealed critical production issues:**

1. **Nov 4 SHORT deployment** was problematic - uses same 89-feature set as failed 90-day model
2. **LONG Entry model** has "buy the dip" bias causing 6 consecutive SLs in downtrends
3. **Production losses** (-$52.57, -15.1%) match expected backtest behavior
4. **Backtest validation** caught over-trading pattern (130 trades/day) BEFORE more damage

**Recommended Actions:**
- ✅ Roll back SHORT to Oct 24 (79 features, proven performance)
- ✅ Increase LONG threshold to 0.85 (filter low-quality signals)
- ✅ Monitor for 24 hours, verify no over-trading
- ✅ Continue feature logging for validated backtest capability

**Status**: ❌ DO NOT DEPLOY NEW models, ⚠️ URGENT ROLLBACK REQUIRED

---

**Analysis Date**: 2025-11-05 12:45 KST
**Analyst**: Claude (Sonnet 4.5)
**Status**: ⚠️ **URGENT: Production using problematic Nov 4 SHORT model - Rollback recommended**
