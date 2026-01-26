# Full Dataset Entry Model Results

**Date**: 2025-10-31 18:51 KST
**Status**: ✅ **LONG TRADES FIXED** but ⚠️ **PERFORMANCE STILL BELOW TARGET**
**Next Step**: Retrain Exit models with full dataset methodology

---

## Executive Summary

After retraining Entry models with full dataset (no pre-filtering), LONG trades now trigger successfully but overall performance remains below Production targets.

### Key Results

| Metric | Filtered Training | Full Dataset | Production Target | Gap to Target |
|--------|------------------|--------------|------------------|---------------|
| **Return** | -2.18% | **+0.71%** | +38.04% | -37.33pp |
| **Win Rate** | 45.1% | **44.2%** | 73.86% | -29.66pp |
| **LONG Trades** | 0 (0%) | **212 (72.6%)** | ~50% | FIXED ✅ |
| **SHORT Trades** | 215 (100%) | **80 (27.4%)** | ~50% | Improved |
| **ML Exit** | 0.0% | **0.0%** | 77.0% | Still broken |
| **Stop Loss** | 12.6% | **33.9%** | ~8% | Too high |

**Progress**: LONG trigger problem solved ✅, but performance improvement minimal

---

## Training Methodology

### What We Did

**Full Dataset Training** (NO Pre-Filtering):
```python
# Previous (filtered - 0 LONG trades):
long_candidates = filter_entry_candidates(df, 'LONG')  # 4,256 samples (14.2%)
long_labels = simulate_trade_outcomes(long_candidates, 'LONG')

# New (full dataset - 212 LONG trades):
long_labels = simulate_trade_outcomes(df, 'LONG')  # 30,004 samples (100%)
short_labels = simulate_trade_outcomes(df, 'SHORT')  # 30,004 samples (100%)
```

**Training Results**:
```yaml
LONG Entry:
  Samples: 30,004 (vs 4,256 filtered, +605% increase)
  Positive Labels: 14,910 (49.7%)
  Best Fold: 1/5 (F1: 0.6346)
  Avg Prediction Rate: 44.65%

SHORT Entry:
  Samples: 30,004 (vs 2,594 filtered, +1057% increase)
  Positive Labels: 14,669 (48.9%)
  Best Fold: 1/5 (F1: 0.3959)
  Avg Prediction Rate: 22.70%
```

---

## Backtest Results (19 Windows, ~95 Days)

### Overall Performance

```yaml
Configuration:
  Entry: full_dataset_20251031_184949 (85/79 features, NO filtering)
  Exit: threshold_075_20251024 (27 features, filtered training)
  Threshold: LONG 0.75, SHORT 0.75
  Leverage: 4x, Initial Capital: $10,000

Performance:
  Return: +0.71% per 5-day window
  Win Rate: 44.2% (129W / 163L)
  Total Trades: 292 (15.4 per window, ~3.1/day)

Exit Distribution:
  ML Exit: 0.0% (0 trades) ← STILL BROKEN
  Stop Loss: 33.9% (99 trades) ← HIGH
  Max Hold: 59.6% (174 trades) ← PRIMARY EXIT

LONG/SHORT Distribution:
  LONG: 72.6% (212 trades) ← FIXED! (was 0%)
  SHORT: 27.4% (80 trades)
```

---

## Improvement vs Filtered Training

### Filtered Training (Previous - 45.1% WR)
```yaml
Training: 4,256 LONG, 2,594 SHORT (filtered candidates)
Result: Models failed on full dataset

Performance:
  Return: -2.18% per window
  Win Rate: 45.1% (97W / 118L)
  ML Exit: 0.0%
  Trades: 11.3 per window

LONG/SHORT:
  LONG: 0.0% (0 trades) ← PROBLEM
  SHORT: 100.0% (215 trades)
```

### Full Dataset Training (NEW - 44.2% WR)
```yaml
Training: 30,004 LONG, 30,004 SHORT (full dataset)
Result: LONG trades trigger, but low win rate

Performance:
  Return: +0.71% per window (+2.89pp improvement)
  Win Rate: 44.2% (129W / 163L) (-0.9pp)
  ML Exit: 0.0% (still broken)
  Trades: 15.4 per window

LONG/SHORT:
  LONG: 72.6% (212 trades) ← FIXED!
  SHORT: 27.4% (80 trades)
```

**Improvement**: LONG trigger problem solved (+72.6pp), but performance still far from target

---

## Root Cause Analysis

### Issue 1: Entry-Exit Model Mismatch

**Entry Models**:
- Trained on full 30,004 candles (NO filtering)
- See complete data distribution
- Models trigger correctly (212 LONG, 80 SHORT)

**Exit Models**:
- Trained on **filtered candidates** (4,256 LONG, 2,594 SHORT)
- Incompatible with full dataset Entry predictions
- Result: 0% ML Exit rate (Exit models never trigger)

**Evidence**:
```yaml
Filtered Entry (45.1% WR):
  ML Exit: 0.0% (filtered Entry + filtered Exit)

Full Dataset Entry (44.2% WR):
  ML Exit: 0.0% (full Entry + filtered Exit)

Conclusion: Exit models are the problem, not Entry models
```

### Issue 2: High Stop Loss Rate (33.9%)

**Symptoms**:
- 99 out of 292 trades hit Stop Loss (33.9%)
- Production target: ~8%

**Possible Causes**:
1. Entry models picking too many losing trades
2. 0.75 threshold too low for full dataset models
3. Exit models not working → positions hemorrhage until SL

**Evidence**:
```yaml
Windows 6-11 (Poor Performance):
  Win Rate: 35-43%
  Stop Loss Rate: High
  Return: Negative (-9.21% to -11.17%)

Windows 1-5 (Good Performance):
  Win Rate: 60-75%
  Stop Loss Rate: Lower
  Return: Positive (+3.61% to +37.91%)
```

### Issue 3: Max Hold Dominance (59.6%)

**Symptoms**:
- 174 out of 292 trades hit Max Hold (59.6%)
- Production target: ~15%

**Root Cause**:
- ML Exit 0% → positions never exit cleanly
- Positions wait full 120 candles before forced exit
- Many winners turn into losers while waiting

**Example**:
```
Trade enters profitable (+2% leveraged)
ML Exit threshold: 0.75 (never triggers)
Position waits 120 candles
Price reverses
Exit at Max Hold: -1% (loss)
```

---

## What Worked

1. ✅ **Full Dataset Training**: LONG trades now trigger (0 → 212 trades)
2. ✅ **Production Features**: 85/79 features enable predictions
3. ✅ **LONG/SHORT Balance**: 72.6% / 27.4% (vs 0% / 100%)
4. ✅ **Trade Frequency**: 15.4 per window (good activity)

---

## What Didn't Work

1. ❌ **Exit Models**: 0% ML Exit (still broken with filtered training)
2. ❌ **Win Rate**: 44.2% (far from 73.86% target)
3. ❌ **Stop Loss Rate**: 33.9% (4× higher than target)
4. ❌ **Return**: +0.71% (far from +38.04% target)

---

## Comparison: Full Dataset Entry vs Production

### Production Methodology (73.86% WR)

```yaml
Training:
  Entry: Full dataset (30,004 candles, NO filtering)
  Exit: Full dataset (30,004 candles, NO filtering)
  Integration: Both trained on same data distribution

Exit Models:
  ML Exit: 77.0% (primary mechanism)
  Training: Compatible with Entry models

Result:
  Win Rate: 73.86%
  Return: +38.04% per 5-day window
  Stop Loss: ~8%
```

### Our Approach (44.2% WR)

```yaml
Training:
  Entry: Full dataset (30,004 candles, NO filtering) ✅
  Exit: Filtered dataset (4,256/2,594 candidates) ❌
  Integration: MISMATCH - different data distributions

Exit Models:
  ML Exit: 0.0% (incompatible with Entry models)
  Training: Trained on filtered subset

Result:
  Win Rate: 44.2%
  Return: +0.71% per 5-day window
  Stop Loss: 33.9%
```

**Key Difference**: Production trains Exit models on full dataset too (we didn't)

---

## Solution: Retrain Exit Models with Full Dataset

### Approach

**Method**:
1. Retrain Exit models on full 30,004 candles (NO filtering)
2. Use same methodology as Entry models:
   - Walk-Forward 5-fold validation
   - Rule-based labels (leveraged_pnl > 0.02)
   - Production's 27 Exit features
3. Ensure Entry-Exit data distribution compatibility

**Expected Result**:
```yaml
ML Exit: 0% → 77% (Exit models trigger)
Win Rate: 44.2% → 70%+ (better exits)
Stop Loss: 33.9% → 8% (fewer losses)
Return: +0.71% → +35%+ per window
```

**Trade-offs**:
- Pro: Solves Entry-Exit mismatch completely
- Pro: Expected to reach Production performance
- Con: Requires Exit model retraining (~30 minutes)

---

## Alternative: Adjust Entry Thresholds

### Quick Test (No Retraining)

**Method**:
1. Raise Entry thresholds: 0.75 → 0.85
2. Test with existing models
3. Evaluate performance

**Expected Result**:
```yaml
Trade Frequency: 15.4 → 8-10 per window (more selective)
Win Rate: 44.2% → 55-60% (higher quality)
Stop Loss: 33.9% → 20-25% (fewer poor entries)
ML Exit: Still 0% (doesn't fix Exit models)
```

**Trade-offs**:
- Pro: Fast to test (no retraining, 5 minutes)
- Pro: May reduce Stop Loss rate
- Con: Doesn't fix ML Exit problem
- Con: Won't reach 73.86% WR target
- Con: Band-aid fix, not root cause solution

---

## Recommendation

### Primary: **Retrain Exit Models with Full Dataset**

**Rationale**:
1. Root cause: Entry-Exit data distribution mismatch
2. Production trains Exit on full dataset
3. Current filtered Exit incompatible with full dataset Entry
4. LONG trigger problem solved proves full dataset training works

**Implementation**:
```python
# Retrain Exit models (same as Entry approach)
# NO filtering - train on all 30,004 candles

def simulate_exit_outcomes(df, side):
    """
    Generate Exit labels from rule-based logic
    Train on ALL candles (no pre-filtering)
    """
    labels = []

    for idx in range(len(df)):
        # Calculate leveraged P&L from current candle
        # Label: 1 if should exit (profit target met), 0 otherwise

    return np.array(labels)

# Train with full dataset
long_exit_labels = simulate_exit_outcomes(df, 'LONG')
short_exit_labels = simulate_exit_outcomes(df, 'SHORT')

# Walk-Forward validation (same as Entry)
long_exit_model, long_exit_scaler = train_walk_forward(
    df, long_exit_labels, long_exit_features, 'LONG_EXIT'
)
```

**Expected Outcome**:
- ML Exit: 0% → 77%
- Win Rate: 44.2% → 70%+
- Return: +0.71% → +35%+ per window
- Stop Loss: 33.9% → 8%

**Time**: 30-45 minutes (Exit models training)

---

## Alternative: Quick Threshold Test

If user wants fast validation before full retraining:

```python
# Test with higher thresholds (no retraining)
LONG_THRESHOLD = 0.85  # Higher from 0.75
SHORT_THRESHOLD = 0.85  # Higher from 0.75

# Expected: Win Rate 44.2% → 55-60%, but ML Exit still 0%
```

**Time**: 5 minutes (just backtest)

**Limitation**: Doesn't fix Exit models, won't reach 73.86% target

---

## Files Created

### Training
- `scripts/experiments/retrain_entry_full_dataset.py` - Full dataset Entry training script
- `models/xgboost_long_entry_full_dataset_20251031_184949.pkl` - LONG model (85 features)
- `models/xgboost_short_entry_full_dataset_20251031_184949.pkl` - SHORT model (79 features)

### Backtest
- `scripts/experiments/backtest_full_dataset_models.py` - Validation script
- `results/backtest_full_dataset_20251031_185146.csv` - Results (292 trades)

### Documentation
- `claudedocs/FULL_DATASET_RESULTS_20251031.md` - This document

---

## Next Steps

### Immediate (User Decision)

**Option A**: Retrain Exit models with full dataset (30-45 mins)
  - Most likely to reach Production targets (73.86% WR)
  - Fixes root cause (Entry-Exit mismatch)
  - Recommended approach

**Option B**: Quick threshold test (5 mins)
  - Validates if higher thresholds help
  - Won't fix Exit models
  - Good for diagnostic

**Option C**: Deploy current models (not recommended)
  - LONG trades working
  - But performance still poor (44.2% WR vs 73.86% target)
  - ML Exit 0% means no clean exits

### Future Work

After reaching 70%+ Win Rate with full dataset Exit models:

1. **Full 108-Window Backtest** - Validate on complete period
2. **Compare vs Production** - Head-to-head with 73.86% WR baseline
3. **Deploy to Testnet** - Live testing with real market conditions
4. **Monitor Week 1** - Track performance vs backtest expectations

---

## Key Learnings

### What We Discovered

1. **Full Dataset Training Works**: LONG trigger problem solved (0 → 212 trades)
2. **Entry-Exit Must Match**: Exit models must be trained on same data distribution
3. **Filtered Training Fails**: Models trained on subset can't generalize to full data
4. **Root Cause Matters**: Fixing Entry models alone isn't enough (Exit models broken too)

### What Production Does Right

1. **Consistent Data Distribution**: Entry and Exit trained on same full dataset
2. **Walk-Forward Validation**: Prevents overfitting to single fold
3. **Decoupled Labels**: Independent training enables iteration
4. **Complete Integration**: Both Entry and Exit models see same data

---

## Conclusion

**Progress**: LONG trigger problem solved ✅ (0 → 212 trades)

**Status**: Performance still below target (44.2% vs 73.86% WR)

**Root Cause**: Entry-Exit data distribution mismatch (full vs filtered training)

**Solution**: Retrain Exit models with full dataset methodology

**Expected Result**: Win Rate 44.2% → 70%+, ML Exit 0% → 77%

**Confidence**: 🟡 **MODERATE** - Entry models work, Exit models are the remaining blocker

---

**Report Generated**: 2025-10-31 18:51 KST
**Author**: Claude Code (Comprehensive Analysis)
**Status**: ✅ **LONG TRADES FIXED** | ⚠️ **NEEDS EXIT MODEL RETRAINING**
