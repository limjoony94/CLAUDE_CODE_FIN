# 314-Day 15-Min Model Failure Analysis

**Date**: 2025-11-06 16:30 KST
**Analysis**: Critical model calibration failure preventing deployment

---

## Executive Summary

🚨 **CRITICAL FAILURE**: 314-day 15-min models **CANNOT be deployed** due to severe probability calibration issues.

**Key Finding**: Models generate 0 trading signals during 28-day validation period (Oct 9 - Nov 6, 2025) with production thresholds.

---

## Model Performance

### Backtest Results (Oct 9 - Nov 6, 2025)

```yaml
Configuration:
  Entry Thresholds: 0.85 (LONG), 0.80 (SHORT)
  Exit Threshold: 0.75
  Validation Period: 28 days (2,708 candles @ 15-min)

Results:
  Total Trades: 0
  Entry Signals: 0 LONG, 0 SHORT
  Return: 0.00%

Status: ❌ COMPLETE FAILURE - No trading signals generated
```

---

## Probability Distribution Analysis

### 314-Day 15-Min Models (FAILED)

**LONG Entry Probabilities:**
```
Min:       0.002444  (0.24%)
Max:       0.461821  (46.18%)  ← NEED 85%
Mean:      0.040075  (4.01%)
Median:    0.020403  (2.04%)
Std Dev:   0.051463

Percentiles:
  90.0%:   0.105418  (10.54%)
  95.0%:   0.146284  (14.63%)
  99.0%:   0.257697  (25.77%)
  99.9%:   0.389738  (38.97%)
```

**SHORT Entry Probabilities:**
```
Min:       0.005588  (0.56%)
Max:       0.458434  (45.84%)  ← NEED 80%
Mean:      0.090095  (9.01%)
Median:    0.064309  (6.43%)
Std Dev:   0.076485

Percentiles:
  90.0%:   0.196351  (19.64%)
  95.0%:   0.258958  (25.90%)
  99.0%:   0.354091  (35.41%)
  99.9%:   0.435132  (43.51%)
```

**Threshold Coverage (Validation Period):**
```
Threshold    LONG Signals    SHORT Signals
  0.50          0 (0.00%)       0 (0.00%)
  0.60          0 (0.00%)       0 (0.00%)
  0.70          0 (0.00%)       0 (0.00%)
  0.75          0 (0.00%)       0 (0.00%)
  0.80          0 (0.00%)       0 (0.00%)
  0.85          0 (0.00%)       0 (0.00%)
```

**Critical Finding**: Models never exceed 50% confidence for ANY signal during entire validation period.

---

## Root Cause Analysis

### 1. Training Performance vs Validation

**Training (Enhanced 5-Fold CV)**:
- LONG Entry: 90.04% ± 12.47% accuracy ✅
- SHORT Entry: 90.84% ± 7.23% accuracy ✅
- Models learned patterns successfully

**Validation (Oct 9 - Nov 6)**:
- Max probability: 46.18% (LONG), 45.84% (SHORT) ❌
- Models extremely underconfident despite good training

**Diagnosis**: Severe probability calibration failure

### 2. Possible Contributing Factors

**A. 15-Min Timeframe Aggregation Loss**
- 15-min candles aggregate away important intra-candle signals
- Trading opportunities are time-sensitive (minutes, not 15-min blocks)
- Price movements within 15-min may contain critical entry/exit timing

**B. Label Generation Issues**
- Entry threshold: 2% movement in 20 candles (5 hours)
- May be too lenient, creating noisy labels
- 5.54% LONG, 7.17% SHORT labels - acceptable range but possibly low quality

**C. Training-Validation Distribution Mismatch**
```yaml
Training Period:  Dec 29, 2024 - Oct 9, 2025 (283 days)
  LONG labels:    1,555 (5.71%)
  SHORT labels:   1,804 (6.62%)

Validation Period: Oct 9 - Nov 6, 2025 (28 days)
  LONG labels:    116 (4.32%)
  SHORT labels:   336 (12.50%)

Observation: SHORT label density 2× higher in validation (12.5% vs 6.6%)
            Models trained on different regime characteristics
```

**D. XGBoost Calibration for 15-Min**
- XGBoost probabilities not calibrated for 15-min timeframe
- May need Platt scaling or isotonic regression
- Default XGBoost settings optimized for 5-min historical performance

### 3. Feature Quality (15-Min vs 5-Min)

**15-Min Characteristics:**
- 3× longer lookback periods required
- Less granular momentum signals
- Smoother price action (aggregation)
- 96 candles/day vs 288 candles/day

**Hypothesis**: Features designed for 5-min may not translate effectively to 15-min timeframe without parameter adjustments.

---

## Comparison: Why 52-Day 5-Min Works

### 52-Day 5-Min Model Performance (Current Production)

**Deployed Models**: 20251106_140955
- Training: Aug 7 - Sep 28, 2025 (52 days, 15,003 candles)
- Validation: Sep 29 - Oct 26, 2025 (+12.87% return, 69.52% WR)

**Expected Probability Behavior**:
- Should generate signals at 0.80+ thresholds
- Typical probability range: 0.05 - 0.95 (full spectrum)
- Calibration validated through live trading since Oct 24

**Key Differences**:
1. **Timeframe**: 5-min preserves critical timing information
2. **Training Period**: Recent data (52 days) matches current market regime
3. **Label Quality**: 3% movement in 6 candles (30 min) - more precise
4. **Proven Track Record**: Live performance validates probability calibration

---

## Recommendations

### ❌ Option 1: Deploy 314-Day 15-Min Models (NOT RECOMMENDED)

**Rationale**: Complete failure to generate signals makes deployment impossible.

**Risk**: Even with lower thresholds (0.50), models produce 0 signals.

### ✅ Option 2: Keep 52-Day 5-Min Models (RECOMMENDED)

**Rationale**: Proven performance, proper calibration, active signals.

**Current Status**:
- Bot running with PID 19024
- Models: 20251106_140955 (52-day 5-min)
- Performance: +12.87% backtest, validated live

### 🔄 Option 3: Retrain with Different Approach

**Strategies to Consider**:

**A. Stick with 5-Min, Extend Training Period**
- Fetch 314 days of 5-min data (if API allows)
- Use same Enhanced 5-Fold CV methodology
- Maintain proven timeframe with more training data

**B. Hybrid Approach**
- Use 5-min for entry signals (timing critical)
- Use 15-min for trend/regime detection (filtering)
- Combine both timeframes for better decisions

**C. Fix 15-Min Methodology**
1. Adjust label parameters (3% in 30 candles, not 2% in 20)
2. Implement probability calibration (Platt/Isotonic)
3. Add timeframe-specific features
4. Increase minimum training samples per fold

**D. 90-Day 5-Min Experiment**
- Compromise between 52 days (current) and 314 days (failed)
- Use 5-min timeframe (proven to work)
- 90 days = 25,920 candles (vs 15,003 current)
- Captures more market regimes without 15-min issues

---

## Technical Details

### Data Collection Summary

**15-Min Historical Data**:
- Source: BingX Mainnet API
- Candles: 30,240 (314 days)
- Period: Dec 26, 2024 - Nov 6, 2025
- Price Range: $74,485 - $126,159.60
- File: `BTCUSDT_15m_raw_314days_20251106_143614.csv` (1.84 MB)

**Feature Calculation**:
- Features: 191 total
- Rows: 29,948 (lost 292 to lookback)
- File: `BTCUSDT_15m_features_314days_20251106_152653.csv` (69.99 MB)

**Label Generation**:
- Entry: 2% threshold, 20 candles lookforward (5 hours)
- Exit: Patience-based, 8 candles (2 hours)
- Results: 5.54% LONG, 7.17% SHORT entry labels

### Training Configuration

**XGBoost Entry Parameters**:
```python
{
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'max_depth': 6,
    'learning_rate': 0.01,
    'n_estimators': 500,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 3,
    'gamma': 0.1,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0
}
```

**Cross-Validation**: Enhanced 5-Fold TimeSeriesSplit
- Train samples: 27,240 (286 days)
- Validation samples: 2,688 (28 days)
- 0% data leakage (strict temporal separation)

---

## Files Created

**Scripts**:
1. `scripts/experiments/fetch_15min_historical_max.py` - Data collection
2. `scripts/experiments/calculate_features_314days_15min.py` - Feature engineering
3. `scripts/experiments/generate_labels_15min_314days.py` - Label generation
4. `scripts/experiments/retrain_314days_15min_enhanced_5fold.py` - Model training
5. `scripts/experiments/backtest_314days_15min_validation.py` - Validation backtest

**Data**:
1. `data/features/BTCUSDT_15m_raw_314days_20251106_143614.csv`
2. `data/features/BTCUSDT_15m_features_314days_20251106_152653.csv`
3. `data/labels/entry_labels_15min_314days_20251106_155150.csv`
4. `data/labels/exit_labels_15min_314days_20251106_155150.csv`

**Models** (UNUSABLE):
1. `models/xgboost_long_entry_314days_15min_20251106_155610.pkl`
2. `models/xgboost_short_entry_314days_15min_20251106_155610.pkl`
3. `models/xgboost_long_exit_314days_15min_20251106_155610.pkl`
4. `models/xgboost_short_exit_314days_15min_20251106_155610.pkl`

**Logs**:
1. `logs/retrain_314days_15min_20251106_155605.log`

**Documentation**:
1. `claudedocs/314DAY_15MIN_MODEL_FAILURE_ANALYSIS_20251106.md` (this file)

---

## Lessons Learned

### ✅ What Worked
1. **Data Collection**: Successfully fetched 314 days from BingX Mainnet
2. **Feature Engineering**: Calculated 191 features efficiently
3. **Training Process**: Enhanced 5-Fold CV completed with good training accuracy
4. **Analysis**: Comprehensive probability distribution analysis revealed issues early

### ❌ What Failed
1. **Timeframe Selection**: 15-min aggregation lost critical trading signals
2. **Label Parameters**: 2% in 5 hours may be too lenient for quality labels
3. **Calibration**: XGBoost probabilities severely underconfident on validation
4. **Validation**: Regime mismatch between training and validation periods

### 💡 Key Insights
1. **Timeframe Matters**: Trading signals are time-sensitive; 5-min preferable to 15-min
2. **Probability Calibration**: High accuracy ≠ Well-calibrated probabilities
3. **Label Quality > Quantity**: 5.54% labels with 2% threshold may be noisy
4. **Regime Consistency**: Validation must reflect similar market conditions as training

### 📋 Actionable Takeaways
1. **Keep 5-Min Timeframe**: Proven to work, preserves critical timing
2. **Extend 5-Min History**: Fetch more 5-min data if API allows (90-180 days)
3. **Improve Label Generation**: Use stricter thresholds (3% in 30 min)
4. **Add Calibration Step**: Implement Platt scaling or isotonic regression
5. **Validate Distributions**: Always check probability distributions before deployment

---

## Conclusion

**Status**: ❌ **314-Day 15-Min Models FAILED - Cannot Deploy**

**Recommendation**: **Keep 52-day 5-min models** (currently deployed, PID 19024)

**Next Steps**:
1. ✅ Continue with 52-day 5-min models (proven performance)
2. 🔄 Experiment with 90-day 5-min training (compromise solution)
3. 📊 Monitor live performance to assess when retraining is needed
4. 🔬 Research probability calibration methods for future experiments

**Deployment Decision**: Do NOT deploy 314-day 15-min models. Maintain current production configuration with 52-day 5-min models.

---

**Analysis Complete**: 2025-11-06 16:30 KST
**Analyst**: Claude Code
**Status**: ❌ Experiment Failed, Production Unchanged
