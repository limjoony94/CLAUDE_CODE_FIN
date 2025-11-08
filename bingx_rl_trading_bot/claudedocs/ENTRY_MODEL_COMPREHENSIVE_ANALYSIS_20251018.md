# Entry Model Improvement: Comprehensive Analysis
**Date**: 2025-10-18
**Status**: ❌ FAILED - Training improvements don't translate to trading performance

---

## Executive Summary

**Objective**: Improve Entry model training precision to reduce false positives and overtrading.

**Approaches Tested**:
1. **Baseline** - Original labeling (LONG: 13.7% precision)
2. **2-of-3 Scoring** - Entry-quality criteria (+151% precision → 34.4%)
3. **Trade-Outcome** - Simulated trade results (+168% precision → 36.69%)

**Result**: ❌ **BOTH improvement attempts FAILED in backtest despite training success**

---

## Problem Statement

### Original Issues (Baseline)
- LONG Entry Precision: **13.7%** (86.3% false positives)
- Overtrading: 50-101 trades per window (vs target ~35)
- Win Rate: ~60% (acceptable but precision too low)
- Returns: 13.93% per window (good performance despite low precision)

### User Direction
> "threshold 최적화 이전에 훈련 최적화를 진행하면 안되나요?"
> (Shouldn't we optimize training before threshold optimization?)

**Insight**: Focus on improving label quality, not just adjusting thresholds.

---

## Approach 1: 2-of-3 Entry Quality Scoring

### Methodology
Borrowed from successful Exit labeling approach:

**Criteria** (2 of 3 required):
1. **Profit Target**: Max profit ≥ 0.4% within 4 hours
2. **Early Entry Advantage**: Current entry ≥ 95% of delayed entry profit
3. **Relative Performance**: Entry within 0.2% of optimal price

### Training Results ✅
```
LONG Entry Model:
  Precision: 13.7% → 34.4% (+151% improvement)
  Recall: 19.8%
  Positive Label Rate: 25.3%

SHORT Entry Model:
  Precision: 30.3%
  Positive Label Rate: 19.1%
```

**Observation**: Significant training metric improvement achieved.

### Backtest Results ❌
```
Performance Comparison (105 days, 100 windows):
  Metric              Baseline    2-of-3      Change
  ------------------------------------------------
  Win Rate            60.0%       49.8%       -17.0%
  Avg Return          13.93%      8.40%       -39.7%
  Total Trades        35.3        49.2        +39.1%
  SHORT Trades        5.2         16.2        +209.6%

  Problematic Windows:
    WR < 40%          18          34          +88.9%
    >50 trades        24          38          +58.3%
```

**Verdict**: ❌ Worse trading performance despite better training metrics

### Root Cause Analysis

**Why Training Success ≠ Trading Success?**

1. **Labeling Philosophy Mismatch**
   - 2-of-3 criteria optimized for "predictable entries"
   - NOT optimized for "profitable trades"
   - Entry quality ≠ trade profitability

2. **Comparison with Exit Success**
   - Exit labeling: criteria = actual trade outcomes ✅
   - Entry labeling: criteria ≠ trade profitability ❌

3. **SHORT Overtrading**
   - +209.6% increase in SHORT trades
   - Models learned to predict "good SHORT entries"
   - But these entries don't lead to profitable trades

---

## Approach 2: Trade-Outcome Based Labeling

### Methodology
**Philosophy**: Label entries based on simulated complete trades (entry → hold → exit)

**Implementation**:
- Trade Simulator: Simulates trades using actual Exit models
- Labeling: 2-of-3 scoring on trade outcomes (not entries)

**Criteria** (2 of 3 required):
1. **Profitable Trade**: Leveraged P&L ≥ 2%
2. **Good Risk-Reward**: MAE < 2%, MFE > 4% (2:1 ratio)
3. **Efficient Exit**: ML Exit model triggered (not emergency)

### Sample Training Results ✅
```
Data: 5,000 candles (sample validation)

LONG Entry Model:
  Precision: 13.7% → 36.69% (+168% improvement)
  Recall: 39.84%
  F1: 38.20%
  Positive Label Rate: 20.9%

SHORT Entry Model:
  Precision: 20.78%
  Recall: 30.05%
  F1: 24.57%
  Positive Label Rate: 16.3%
```

**Observation**: Even better training improvement than 2-of-3.

### Critical Discovery ⚠️
```
Risk-Reward Criterion Pass Rates:
  LONG:  3 / 4,894 (0.1%)
  SHORT: 7 / 4,894 (0.1%)
```

**Issue**: Risk-Reward criterion (MAE<2%, MFE>4%) is too strict!
- Only 0.1% of trades meet this criterion
- Effectively becomes 2-of-2 system (Profitable + Efficient Exit)

### Backtest Results ❌
```
Test: 20 windows (500 candles each), OUTSIDE training data

Performance Comparison:
  Metric              Baseline    Sample      Change
  ------------------------------------------------
  Win Rate            87.39%      62.23%      -28.8%
  Avg Return          1.36%       0.45%       -67.0%
  Total Return        9.83%       2.48%       -74.7%
  Avg Trades          7.2         6.9         -4.2%
  LONG Trades         5.5         4.5         -17.3%
  SHORT Trades        1.7         2.4         +38.2%
  ML Exit Rate        97.0%       62.8%       -35.2%
```

**Verdict**: ❌ **WORSE performance than Baseline AND 2-of-3**

### Root Cause Analysis

**Why Trade-Outcome Labeling FAILED?**

1. **ML Exit Rate Dropped Dramatically**
   - Baseline: 97.0% ML exits
   - Sample: 62.8% ML exits (-35.2%)
   - Models not reaching ML exits → emergency exits instead

2. **Same Pattern as 2-of-3**
   - Training metrics improved ✅
   - Backtest performance degraded ❌
   - Suggests fundamental misalignment

3. **Potential Issues**:
   - **Simulation Mismatch**: Simulated trades ≠ real trades
   - **Overfitting**: 5,000 sample not representative
   - **Risk-Reward Too Strict**: 0.1% pass rate creates bad labels
   - **Exit Model Dependency**: Entry labels depend on Exit model behavior

---

## Comprehensive Comparison

### Training Metrics
```
Approach          LONG Precision   SHORT Precision   Improvement
----------------------------------------------------------------
Baseline          13.7%            Unknown           -
2-of-3            34.4%            30.3%             +151%
Trade-Outcome     36.69%           20.78%            +168%
```

**All improvements showed significant training metric gains.**

### Backtest Performance (Avg Return per Window)
```
Approach          Avg Return   Win Rate   Trades   Verdict
-----------------------------------------------------------
Baseline          13.93%       60.0%      35.3     ✅ Best
2-of-3            8.40%        49.8%      49.2     ❌ Failed
Trade-Outcome     0.45%        62.23%     6.9      ❌ Failed
```

**Both improvements FAILED to translate training gains to trading performance.**

---

## Key Insights

### 1. Training Precision ≠ Trading Performance

**Evidence**:
- 2-of-3: +151% precision → -39.7% returns
- Trade-Outcome: +168% precision → -67.0% returns

**Lesson**: Models can learn to predict labels accurately while making worse trading decisions.

### 2. Exit Labeling Success Doesn't Transfer to Entry

**Exit Success**:
- Exit criteria = actual trade outcomes
- Direct alignment between training and trading goals

**Entry Failure**:
- Entry criteria ≠ trade profitability
- Indirect relationship between "good entry" and "profitable trade"

### 3. Fundamental Challenge

**Problem**: Entry quality is only ONE factor in trade profitability.

**Other Factors**:
- Market conditions during hold period
- Exit timing (handled by Exit models)
- Volatility and momentum
- Risk management (stop loss, max hold)

**Implication**: Improving Entry labels alone may not improve overall trading performance if labels don't account for these factors.

---

## Root Cause: Why Both Approaches Failed

### Hypothesis 1: Labeling Philosophy Mismatch ⚠️

**2-of-3 Entry Quality**:
- Optimized for: "Is this entry price good relative to future prices?"
- Missing: "Will a trade from here be profitable given Exit strategy?"

**Trade-Outcome**:
- Optimized for: "Does simulated trade from here profit?"
- Missing: "Are simulations representative of real trades?"

### Hypothesis 2: Exit Model Dependency 🔗

**Trade-Outcome Issue**:
- Entry labels depend on Exit model predictions
- If Exit models change, Entry labels become invalid
- Creates circular dependency: Entry → Exit → Entry

**Evidence**:
- ML Exit Rate dropped from 97% to 62.8%
- Suggests models learned entries that don't lead to ML exits

### Hypothesis 3: Overfitting to Noise 📊

**Risk-Reward Criterion** (0.1% pass rate):
- Only 3-7 out of 4,894 candles meet criterion
- Extremely rare events may be noise, not signal
- Models may learn to avoid these "good" trades

### Hypothesis 4: Sample Size Limitations 📏

**Trade-Outcome Sample**:
- Only 5,000 candles (vs 30,517 full dataset)
- May not be representative of full market conditions
- Backtest on different period shows poor generalization

---

## Lessons Learned

### 1. Evidence > Assumptions ✅

User guidance to "optimize training first" was correct direction, but:
- Assumption: Better training labels → better trading
- Evidence: Training improvement ≠ backtest improvement

**Lesson**: Always validate with backtest, not just training metrics.

### 2. Exit Success ≠ Entry Blueprint ⚠️

Exit labeling 2-of-3 approach worked because:
- Exit decision is immediate (exit now vs hold)
- Exit criteria = actual trade outcomes

Entry labeling failed because:
- Entry decision is forward-looking (will trade profit?)
- Entry criteria ≠ actual trade profitability

**Lesson**: Different problems require different solutions.

### 3. Simulation Limitations 🎭

Trade-Outcome approach:
- Clever idea: simulate trades to create labels
- Implementation challenge: simulations ≠ reality
- Risk: circular dependency with Exit models

**Lesson**: Simulations can introduce artifacts that hurt generalization.

---

## Recommendations

### Option 1: Keep Baseline Models ✅ RECOMMENDED

**Rationale**:
- Baseline performs best in backtest (13.93% vs 8.40% vs 0.45%)
- 60% win rate is acceptable
- 13.7% precision is low but trading performance is good

**Action**: Accept that training precision ≠ trading performance

### Option 2: Threshold Optimization 🎯

**Approach**: User's original suggestion (which we deferred)
- Increase LONG threshold: 0.65 → 0.70 or 0.75
- Adjust SHORT threshold: 0.70 → 0.72 or 0.75

**Rationale**:
- Reduce false positives by being more selective
- May improve precision without changing labels

**Risk**: May reduce trade count too much

### Option 3: Hybrid Approach 🔀

**Combine**:
- Baseline labels (proven to work)
- Threshold optimization (reduce false positives)
- Add confidence-based position sizing

**Example**:
```python
if long_prob >= LONG_THRESHOLD:
    confidence = (long_prob - LONG_THRESHOLD) / (1 - LONG_THRESHOLD)
    position_size = min_size + (max_size - min_size) * confidence
```

### Option 4: Feature Engineering Instead of Labeling 🔧

**Hypothesis**: Problem isn't labels, it's features.

**Approach**:
- Keep baseline labels
- Add features that capture:
  - Recent win rate
  - Market regime (trending vs ranging)
  - Volatility state
  - Exit model confidence

**Rationale**: Let model learn when to be selective, not force it via labels.

---

## Next Steps

### Immediate: Choose Direction

**User Decision Required**:
1. Accept baseline performance (13.93% return, 60% WR)
2. Try threshold optimization on baseline models
3. Explore hybrid approach (baseline + confidence sizing)
4. Investigate feature engineering approach

### If Continuing Improvement:

**Recommended**:
1. Start with threshold optimization (simplest, lowest risk)
2. Backtest multiple threshold combinations
3. If successful, deploy to testnet
4. Monitor real trading performance

**Not Recommended**:
- Full dataset Trade-Outcome labeling (failed in sample)
- Other 2-of-3 variants (fundamental issue proven)

---

## Conclusion

**Summary**:
- Tested two labeling improvement approaches
- Both achieved +150% training precision improvement
- Both FAILED in backtest (-40% to -67% returns)

**Key Finding**: Training precision ≠ trading performance

**Recommendation**: **Keep baseline models** or try threshold optimization, not label engineering.

**Lesson**: Sometimes the "worse" model is the better trader.

---

## Files Created

### Scripts
1. `improved_entry_labeling.py` - 2-of-3 entry quality labeling (FAILED)
2. `retrain_entry_models_improved_labeling.py` - Retrain with 2-of-3
3. `backtest_improved_entry_models.py` - Backtest 2-of-3 models
4. `compare_entry_models.py` - Baseline vs 2-of-3 comparison
5. `trade_simulator.py` - Trade simulation for labeling
6. `trade_outcome_labeling.py` - Trade-outcome based labeling
7. `retrain_entry_models_trade_outcome_sample.py` - Sample training
8. `backtest_trade_outcome_sample.py` - Sample backtest

### Documentation
1. `ENTRY_MODEL_IMPROVEMENT_20251018.md` - Initial improvement report
2. `ENTRY_MODEL_BACKTEST_ANALYSIS_20251018.md` - 2-of-3 failure analysis
3. `ENTRY_MODEL_COMPREHENSIVE_ANALYSIS_20251018.md` - This report

### Models (FAILED - not recommended for use)
1. `xgboost_long_improved_labeling_20251018_051817.pkl` (2-of-3)
2. `xgboost_short_improved_labeling_20251018_051822.pkl` (2-of-3)
3. `xgboost_long_trade_outcome_sample_20251018_171324.pkl` (Trade-Outcome)
4. `xgboost_short_trade_outcome_sample_20251018_171324.pkl` (Trade-Outcome)

---

**Status**: Analysis Complete - Awaiting User Decision on Next Direction
