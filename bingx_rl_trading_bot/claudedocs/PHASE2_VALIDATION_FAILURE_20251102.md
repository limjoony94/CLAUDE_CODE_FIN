# Phase 2 Model Retraining - Validation Failure Report

**Date**: 2025-11-02 18:15 KST
**Status**: ❌ **FAILED VALIDATION - DO NOT DEPLOY**
**Models**: Phase 2 (20251102_181218)

---

## Executive Summary

Phase 2 model retraining **completely failed validation** on the 28-day holdout period, showing severe overfitting to training data.

**Critical Results:**
- ❌ Return: **-15.53%** (expected: +30-40%)
- ❌ Win Rate: **28.1%** (expected: 70%+)
- ❌ Sharpe Ratio: **-5.819** (expected: positive)
- ❌ Performance vs Current: **-53.57% degradation**

**Recommendation**: **KEEP CURRENT MODELS** (Walk-Forward Decoupled 20251027_194313)

---

## Validation Results (28-Day Holdout)

### Test Configuration
```yaml
Test Period: Sep 28 - Oct 26, 2025
Candles: 8,064 (28 days)
Initial Capital: $10,000
Leverage: 4x
Fees: 0.05% taker
Thresholds: Entry 0.75/0.75, Exit 0.75/0.75
```

### Phase 2 Model Performance
```yaml
Performance:
  Final Balance: $8,446.93
  Total Return: -15.53% ← MASSIVE LOSS
  Sharpe Ratio: -5.819 ← NEGATIVE RISK-ADJUSTED RETURN

Trading:
  Total Trades: 160
  Wins: 45 (28.1%) ← TERRIBLE WIN RATE
  Losses: 115 (71.9%)

P&L:
  Average Win: $11.33
  Average Loss: -$17.94 ← LOSSES > WINS
  Total Fees: $2,139.87 ← HIGH FEE BURDEN

Exit Distribution:
  ML Exit: 159 (99.4%) ← MODEL EXITS, BUT POORLY
  Stop Loss: 1 (0.6%)
  Max Hold: 0 (0.0%)

Hold Time:
  Average: 5.3 candles (0.4 hours) ← VERY SHORT HOLDS
```

### Comparison to Current Production
```yaml
Current Models (Walk-Forward Decoupled 20251027):
  Return: +38.04% per 5-day window
  Win Rate: 73.86%
  Sharpe: 6.610
  Trades: ~23 per 5-day window
  ML Exit: 77.0%

Phase 2 Models (20251102):
  Return: -15.53% (28 days)
  Win Rate: 28.1%
  Sharpe: -5.819
  Trades: 160 (28 days)
  ML Exit: 99.4%

Degradation: -53.57% performance drop ❌
```

---

## Root Cause Analysis

### 1. Training Data Issues

**Problem**: Training data not representative of validation period

```yaml
Training Data:
  Period: Jul 14 - Sep 28, 2025 (76 days, 21,940 candles)
  LONG Labels: 2,248 (10.25%)
  SHORT Labels: 2,679 (12.21%)

Validation Data:
  Period: Sep 28 - Oct 26, 2025 (28 days, 8,064 candles)

Issue:
  - Market conditions changed between training and validation
  - Model learned patterns specific to Jul-Sep period
  - Failed to generalize to Oct period
```

### 2. Feature Engineering Problems

**Problem**: Feature mismatches and substitutions degraded model quality

```yaml
Original Exit Features (Current Production):
  - volatility_20
  - ema_26
  - sma_50

Phase 2 Substitutions:
  - volatility_20 → volatility_10 ← DIFFERENT SIGNAL
  - ema_26 → ema_10 ← DIFFERENT SIGNAL
  - sma_50 → sma_10 ← DIFFERENT SIGNAL

Impact:
  - Models trained on different features than intended
  - Feature substitutions changed signal characteristics
  - Model behavior diverged from design
```

### 3. Label Quality Issues

**Problem**: Entry/Exit labels may not reflect actual trading performance

```yaml
Entry Labels:
  Source: trade_outcome_labels_20251031_145044.csv
  Method: Trade outcome analysis (signal_long, signal_short)

Exit Labels:
  Source: exit_labels_patience_20251030_051002.csv
  Method: Patience-based labeling (long_exit_patience, short_exit_patience)

Potential Issues:
  - Entry labels may be overfitted to historical outcomes
  - Exit labels based on patience, not actual profit optimization
  - Label methods not validated on out-of-sample data
```

### 4. Training Methodology

**Problem**: Simple time-split validation insufficient

```yaml
Phase 2 Approach:
  - Single 76/28 day split
  - Train on Jul-Sep, validate on Sep-Oct
  - No cross-validation on multiple time periods
  - No walk-forward validation

Current Production Approach:
  - Walk-Forward Decoupled with TimeSeriesSplit
  - 5-fold cross-validation
  - Filtered simulation (84-85% efficiency)
  - Per-window model selection (no look-ahead bias)

Advantage: Walk-Forward >> Simple Time Split
```

---

## Failure Indicators

### 1. Severe Overfitting
- Training accuracy: 94%+
- Validation accuracy: 28.1%
- **Gap: 66% degradation** → Classic overfitting

### 2. Loss Distribution Problem
- Average loss ($-17.94) > Average win ($11.33)
- Loss rate (71.9%) > Win rate (28.1%)
- Risk/reward ratio: 1.58 (UNFAVORABLE)

### 3. Excessive Trading
- 160 trades in 28 days = 5.7 trades/day
- Average hold: 0.4 hours (24 minutes)
- High frequency → High fees ($2,139.87)
- Churning capital without profit

### 4. Negative Risk-Adjusted Returns
- Sharpe Ratio: -5.819
- Sortino Ratio: (likely negative)
- Max Drawdown: (not calculated, but likely >20%)

---

## What Went Wrong

### Comparison: Phase 2 vs Walk-Forward Decoupled

| Aspect | Walk-Forward Decoupled (WORKS) | Phase 2 (FAILED) |
|--------|--------------------------------|------------------|
| Training | 5-fold TimeSeriesSplit | Single 76/28 split |
| Validation | Multiple unseen windows | Single 28-day period |
| Features | Consistent across train/test | Feature mismatches |
| Labels | Direct from simulation | External label files |
| Overfitting Prevention | Walk-Forward + Filtering | None (simple split) |
| Result | +38.04% return, 73.86% WR | -15.53% return, 28.1% WR |

**Key Insight**: Walk-Forward Decoupled methodology is CRITICAL for preventing overfitting. Phase 2's simple time-split approach failed completely.

---

## Lessons Learned

### 1. Walk-Forward Validation is Essential
- Simple train/test splits insufficient for time series
- Must validate on multiple unseen time periods
- Walk-Forward Decoupled prevents look-ahead bias

### 2. Feature Consistency Matters
- Feature substitutions change model behavior
- Must use exact same features in train/test
- Feature engineering must be validated

### 3. Label Quality Critical
- External labels may not align with trading performance
- Preference: Labels from actual trade simulation
- Must validate label methodology

### 4. Don't Retrain Without Improvement Hypothesis
- Phase 2 goal: "Retrain on more data"
- Missing: Clear hypothesis for improvement
- Result: Wasted effort, negative results

---

## Recommendations

### Immediate Action
✅ **KEEP CURRENT MODELS** (Walk-Forward Decoupled 20251027_194313)
- Performance: +38.04% return, 73.86% WR
- Methodology: Proven Walk-Forward validation
- Status: Production-ready

❌ **DO NOT DEPLOY Phase 2 Models** (20251102_181218)
- Performance: -15.53% return, 28.1% WR
- Issue: Severe overfitting
- Status: Failed validation

### Future Retraining (If Needed)

If retraining is required, use this methodology:

1. **Walk-Forward Decoupled Approach** (MANDATORY)
   ```python
   - TimeSeriesSplit with 5+ folds
   - Train on past N days, test on next 5 days
   - Select best fold, train on full period
   - Prevents look-ahead bias
   ```

2. **Consistent Feature Engineering**
   ```python
   - Same features in train/test
   - No feature substitutions
   - Validate features exist in dataset
   ```

3. **Label from Simulation**
   ```python
   - Generate labels from trade outcome simulation
   - Don't use external label files
   - Validate label distribution
   ```

4. **Multiple Validation Windows**
   ```python
   - Test on 4+ separate time periods
   - Calculate mean/std performance
   - Require consistent performance
   ```

5. **Improvement Hypothesis**
   ```python
   - Clear hypothesis: "Recent data improves X by Y%"
   - Baseline: Current model performance
   - Target: X% improvement over baseline
   - Abort if hypothesis not validated
   ```

### Alternative: Incremental Improvements

Instead of full retraining, consider incremental improvements:

1. **Hyperparameter Tuning**
   - Grid search on current Walk-Forward Decoupled
   - Test on recent validation data
   - Deploy if improvement >5%

2. **Feature Engineering**
   - Add new features to current models
   - Validate with Walk-Forward approach
   - Deploy if improvement >5%

3. **Threshold Optimization**
   - Optimize entry/exit thresholds on recent data
   - Quick wins with existing models
   - Already proven successful (Phase 1: EXIT 0.70)

---

## Phase 2 Training Details

### Models Created
```yaml
Timestamp: 20251102_181218

LONG Entry:
  - File: xgboost_long_entry_phase2_20251102_181218.pkl
  - Features: 171
  - Best Fold: 5/5
  - Val Accuracy: 94.26%

SHORT Entry:
  - File: xgboost_short_entry_phase2_20251102_181218.pkl
  - Features: 171
  - Best Fold: 4/5
  - Val Accuracy: 93.90%

LONG Exit:
  - File: xgboost_long_exit_phase2_20251102_181218.pkl
  - Features: 27
  - Accuracy: 87.36%
  - Prediction Rate: 35.43%

SHORT Exit:
  - File: xgboost_short_exit_phase2_20251102_181218.pkl
  - Features: 27
  - Accuracy: 86.33%
  - Prediction Rate: 36.50%
```

### Training Configuration
```yaml
Entry Models:
  max_depth: 6
  learning_rate: 0.01
  n_estimators: 500
  subsample: 0.8
  colsample_bytree: 0.8

Exit Models:
  max_depth: 4
  learning_rate: 0.05
  n_estimators: 300
```

---

## Conclusion

**Phase 2 model retraining FAILED validation** due to severe overfitting and methodological flaws.

**Key Findings:**
1. ❌ -15.53% return on 28-day holdout (expected: +30-40%)
2. ❌ 28.1% win rate (expected: 70%+)
3. ❌ -53.57% performance degradation vs current models
4. ❌ Simple time-split insufficient (Walk-Forward required)
5. ❌ Feature mismatches degraded model quality

**Recommendation**:
- ✅ KEEP Walk-Forward Decoupled models (20251027_194313)
- ❌ DO NOT DEPLOY Phase 2 models (20251102_181218)
- 📋 Future retraining MUST use Walk-Forward Decoupled methodology

**Status**: Phase 2 closed as FAILED. Continue with current production models.

---

**Files Created:**
- Training script: `scripts/experiments/retrain_models_phase2.py`
- Validation script: `scripts/experiments/validate_phase2_models.py`
- Models: `models/xgboost_*_phase2_20251102_181218.pkl`
- This report: `claudedocs/PHASE2_VALIDATION_FAILURE_20251102.md`

**Next Action**: Continue monitoring current production models. Consider Phase 1-style threshold optimization instead of full retraining.
