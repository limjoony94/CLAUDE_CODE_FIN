# LONG Entry Model Backtest Data Leakage Analysis

**Date**: 2025-11-05 13:00 KST
**Issue**: User identified data leakage in LONG Entry model backtests
**Status**: ⚠️ **CRITICAL - Reported backtest performance likely overfitted**

---

## Executive Summary

User correctly identified that the LONG Entry model (Enhanced 5-Fold CV 20251024_012445) backtest includes training period data, causing significant data leakage and overfitting.

**Key Finding**: Reported +1,209.26% backtest return (104 days) includes **76 days of training data**, making it an unreliable performance estimate.

**Actual Clean Validation**: Only Oct 24-30 production period (+10.95% in 6 days) is truly out-of-sample.

---

## Data Leakage Evidence

### LONG Entry Model Training (Enhanced 5-Fold CV)

**Training Script**: `retrain_models_phase2.py`

```yaml
Model: xgboost_long_entry_enhanced_20251024_012445.pkl
Training Period: Jul 14 - Sep 28, 2025 (76 days, 21,940 candles)
Validation Period: Sep 28 - Oct 26, 2025 (28 days, 8,064 candles)
Total Data: 104 days (30,004 candles)
```

### Reported Backtest Performance

**Document**: `DEPLOYMENT_CONFIRMED_ENHANCED_BASELINE_20251030.md`

```yaml
Backtest Period: July 14, 2025 - October 26, 2025 (104 days)
Performance: +1,209.26% return (13.1x capital)
Total Trades: 4,135
Win Rate: 56.41%
```

### Data Leakage Calculation

```yaml
Backtest Period: Jul 14 - Oct 26 (104 days)
Training Period: Jul 14 - Sep 28 (76 days)
Overlap: Jul 14 - Sep 28 (76 days = 73.1% of backtest period)

Clean Validation: Sep 28 - Oct 26 (28 days = 26.9% of backtest period only)
```

**Conclusion**: **73.1% of backtest data was seen during training** → Severe data leakage

---

## Why This is a Problem

### 1. Overfitted Performance Metrics

**Reported Performance** (+1,209.26% return):
- Based on 76 days of training data + 28 days validation
- Model already learned patterns from 76 days
- Performance on training data is always better than unseen data
- **Not representative of true out-of-sample performance**

**True Out-of-Sample Performance** (Clean):
- Oct 24-30 production: +10.95% in 6 days
- Only this period is completely unseen by model
- Much lower than reported 104-day average (+11.6% per day)

### 2. Inflated Win Rate

```yaml
Reported (104 days): 56.41% WR
  - Includes 73.1% training data where model "knows the answers"
  - Model trained to maximize accuracy on this exact data

Expected (Clean): Unknown (need proper validation)
  - Oct 24-30 production: 100% WR (1/1 bot trade, small sample)
  - Recent Nov 4-5: 22.2% WR (2/9, falling market)
  - True WR likely between 40-60% depending on regime
```

### 3. False Confidence in Deployment

**Impact of Data Leakage**:
- Team deployed model believing +1,209% performance was achievable
- Reality: Performance highly dependent on market regime
- Nov 4-5 losses (-15.1%) show model struggles in downtrends
- "Buy the dip" bias not visible in training-heavy backtest

---

## Correct Validation Methodology

### What Should Have Been Done

**Option A: Holdout Validation Only**
```yaml
Training: Jul 14 - Sep 28, 2025 (76 days)
Validation: Sep 28 - Oct 26, 2025 (28 days) ← ONLY THIS for backtest
Backtest Report: Performance on 28 days, NOT 104 days

Expected Impact:
  - Lower reported return (not +1,209%)
  - More realistic win rate
  - Visible regime sensitivity
```

**Option B: Walk-Forward Validation**
```yaml
Window 1: Train on Jul 14-Aug 14 → Validate Aug 14-28
Window 2: Train on Jul 14-Aug 28 → Validate Aug 28-Sep 14
Window 3: Train on Jul 14-Sep 14 → Validate Sep 14-28
...
Final: Aggregate all validation periods (never train on validation data)

Benefits:
  - Multiple out-of-sample tests
  - Detect regime changes early
  - More robust performance estimate
```

**Option C: Time-Series Split**
```yaml
Split 1: Train Jul 14-Aug 31 → Validate Sep 1-15
Split 2: Train Jul 14-Sep 15 → Validate Sep 16-30
Split 3: Train Jul 14-Sep 30 → Validate Oct 1-15
Final: Average performance across all validation splits

Benefits:
  - Every datapoint used exactly once for validation
  - No data leakage
  - Realistic performance estimate
```

### What Was Actually Done (WRONG)

```yaml
Training: Jul 14 - Sep 28, 2025 (76 days)
Validation: Sep 28 - Oct 26, 2025 (28 days)
Backtest Report: Jul 14 - Oct 26, 2025 (104 days) ← INCLUDES TRAINING DATA

Result: 73.1% data leakage → Overfitted +1,209% return
```

---

## Production Reality Check

### Clean Out-of-Sample Performance

**Oct 24-30 (6 days)**: ✅ Clean validation
```yaml
Period: Oct 24 - Oct 30, 2025
Balance: $348.94 → $387.14
Return: +10.95%
Trades: 1 bot trade (SHORT, 100% WR, ML Exit success)
Status: UNSEEN by model (trained until Oct 24)
```

**Nov 4-5 (2 days)**: ✅ Clean validation, ❌ Poor performance
```yaml
Period: Nov 4 - Nov 5, 2025
Balance: $348.94 → $296.37
Return: -15.07%
Trades: 9 bot trades (22.2% WR, 6 consecutive LONG SLs)
Status: UNSEEN by model
Issue: "Buy the dip" bias in sustained downtrend
```

### Why Production Differs from Backtest

**Backtest** (73.1% training data):
- Model "knows" patterns from Jul-Sep training
- High win rate (56.41%) because it's predicting seen data
- +1,209% return inflated by training period performance

**Production** (100% unseen data):
- Oct 24-30: +10.95% (market regime similar to training)
- Nov 4-5: -15.07% (market regime different, LONG bias exposed)
- True performance varies dramatically with regime

---

## Impact Assessment

### 1. Trust in Model Performance

**Before Discovery**:
- Team believed model achieves +1,209% return
- Confidence in 56.41% win rate
- Expected consistent profitability

**After Discovery**:
- Reported +1,209% is unreliable (includes training data)
- True out-of-sample: +10.95% (Oct 24-30) to -15.07% (Nov 4-5)
- Win rate varies: 100% (Oct) to 22% (Nov)
- Performance highly regime-dependent

### 2. Deployment Decisions

**Affected Decisions**:
1. Model selection: Chose Enhanced 5-Fold CV based on inflated backtest
2. Threshold optimization: Used leaky backtest for threshold tuning
3. Risk management: Position sizing based on overfitted metrics
4. Production expectations: Expected +1,209% level returns

**Reality**:
- Model works in similar regimes (Oct 24-30: +10.95%)
- Model fails in different regimes (Nov 4-5: -15.07%)
- Need regime detection or adaptive thresholds

### 3. Future Model Validation

**Lesson Learned**:
- Always validate on 100% out-of-sample data
- Never include training period in backtest reports
- Use walk-forward or time-series split validation
- Report multiple regime performances separately

---

## Corrected Performance Estimates

### Reliable Out-of-Sample Results

**Oct 24-30 (Similar Regime)**:
```yaml
Period: 6 days (1,728 candles)
Return: +10.95%
Trades: 1 (SHORT)
Win Rate: 100% (1/1, small sample)
Regime: Post-training, similar volatility
Status: ✅ Model works as expected
```

**Nov 4-5 (Different Regime)**:
```yaml
Period: 2 days (576 candles)
Return: -15.07%
Trades: 9 (7 LONG, 2 SHORT)
Win Rate: 22.2% (2/9)
Regime: Sustained downtrend, -6.2% price drop
Status: ❌ LONG bias exposed
```

**Combined (8 days clean validation)**:
```yaml
Period: Oct 24 - Nov 5, 2025 (8 days)
Return: -5.83% (net of both periods)
Trades: 10 (8 bot trades total)
Win Rate: 30.0% (3/10)
Status: ⚠️ Regime-dependent performance
```

### Unreliable Backtest Results

**Reported 104-day Backtest** (❌ INVALID):
```yaml
Period: Jul 14 - Oct 26, 2025
Return: +1,209.26%
Win Rate: 56.41%
Status: ❌ INVALID (73.1% data leakage)
Reason: Includes 76 days of training data
```

**Sep 28 - Oct 26 Only** (✅ VALID but MISSING):
```yaml
Period: 28 days (validation period only)
Return: NOT REPORTED (should be calculated)
Win Rate: NOT REPORTED
Status: ✅ Would be valid, but not isolated in backtest
```

---

## Recommendations

### Immediate Actions

1. **Recalculate Clean Backtest** (Sep 28 - Oct 26 only):
   - Remove Jul 14 - Sep 28 training period from backtest
   - Report only validation period performance
   - Compare to production results (Oct 24-30, Nov 4-5)

2. **Update Documentation**:
   - Mark +1,209% backtest as INVALID (data leakage)
   - Document correct validation methodology
   - Report only clean out-of-sample results

3. **Adjust Expectations**:
   - Base decisions on Oct 24-Nov 5 production (8 days, -5.83%)
   - Recognize regime-dependent performance
   - Don't expect +1,209% level returns

### Short-term (1-2 Weeks)

1. **Proper Validation Framework**:
   - Implement walk-forward validation
   - Test on multiple unseen periods
   - Report performance by regime type

2. **Regime Analysis**:
   - Classify market regimes in training data
   - Test model performance per regime
   - Identify which regimes model works best

3. **Feature Logging Analysis**:
   - Week 2 of production feature logging
   - Build feature-replay backtest
   - Validate threshold optimization properly

### Long-term (1 Month+)

1. **Systematic Validation Process**:
   - Standard: Train/Validate split with ZERO overlap
   - Required: Multiple out-of-sample periods
   - Best practice: Walk-forward or time-series CV
   - Documentation: Always report split methodology

2. **Regime-Aware Modeling**:
   - Train separate models per regime
   - Add regime detection system
   - Only trade when regime matches training

3. **Continuous Monitoring**:
   - Track production vs backtest accuracy
   - Alert on performance degradation
   - Retrain when regime shifts detected

---

## Key Learnings

### 1. Data Leakage is Easy to Miss

```yaml
Training: Jul 14 - Sep 28 (76 days)
Validation: Sep 28 - Oct 26 (28 days)
Backtest Report: Jul 14 - Oct 26 (104 days) ← WRONG

Why Easy to Miss:
  - Training script correctly split data (76d train, 28d val)
  - But backtest script used entire 104-day dataset
  - No one caught the mismatch between train split and backtest period
```

### 2. User Review is Valuable

```yaml
User Comment: "해당 모델의 백테스트는 모델이 훈련된 기간의 데이터와 겹쳐서 과적합 가능성이 높음"
Translation: "The model's backtest overlaps with training period data, high overfitting risk"

Impact:
  - User immediately spotted what team missed
  - Saved from continuing with invalid validation
  - Highlighted need for rigorous validation review
```

### 3. Production is the Ultimate Test

```yaml
Backtest (with leakage): +1,209.26% return, 56.41% WR
Production (clean): +10.95% (Oct) to -15.07% (Nov)

Reality Check:
  - Backtest inflated expectations
  - Production reveals true performance
  - Regime sensitivity not visible in leaky backtest
```

### 4. Trust but Verify

```yaml
Before: "Model achieved +1,209% in backtest"
After: "Backtest had 73.1% data leakage"

Lesson:
  - Always verify train/test split
  - Always report validation methodology
  - Always test on multiple unseen periods
  - Always document data leakage checks
```

---

## Action Items

### For Current Deployment

**Status**: ✅ LONG threshold increased to 0.85 (Nov 5)
**Reason**: Production showed LONG bias (6 consecutive SLs Nov 4-5)
**Expected**: Fewer but higher quality LONG signals

**Monitoring** (Next 24-48 hours):
- Track trade frequency (expect 0.5-1 trade/day)
- Verify no LONG bias in downtrends
- Check win rate improvement
- Compare to leaky backtest expectations

### For Future Modeling

**Required Changes**:
1. ✅ Always use 100% out-of-sample validation
2. ✅ Never report backtest that includes training data
3. ✅ Document train/val split in every backtest
4. ✅ Verify split before reporting performance
5. ✅ Test on multiple unseen periods (walk-forward)
6. ✅ Report performance by regime separately

---

## Conclusion

User correctly identified that the LONG Entry model (Enhanced 5-Fold CV 20251024_012445) backtest suffers from **73.1% data leakage**, making the reported +1,209.26% return unreliable.

**Corrected Understanding**:
- Reported +1,209%: ❌ INVALID (includes training data)
- Oct 24-30 production: ✅ VALID (+10.95%, similar regime)
- Nov 4-5 production: ✅ VALID (-15.07%, different regime)
- True performance: Regime-dependent, -5.83% over 8 clean days

**Immediate Actions Taken**:
- ✅ LONG threshold increased to 0.85 (filter low-quality signals)
- ✅ SHORT model rolled back to Oct 24 (remove over-trading risk)
- ✅ Bot restarted with corrected configuration

**Next Steps**:
- Monitor production for 24-48 hours
- Recalculate clean backtest (Sep 28-Oct 26 only)
- Implement walk-forward validation framework
- Add regime detection system

---

**Analysis Date**: 2025-11-05 13:00 KST
**Reported by**: User (data leakage identified)
**Status**: ⚠️ **CRITICAL - Backtest performance unreliable due to 73.1% data leakage**
