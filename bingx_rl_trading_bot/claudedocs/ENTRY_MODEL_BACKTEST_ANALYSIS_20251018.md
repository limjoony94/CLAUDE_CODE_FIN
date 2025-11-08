# Entry Model Backtest Analysis - Improved vs Baseline

**Date**: 2025-10-18 05:40 KST
**Status**: ❌ **IMPROVED MODELS UNDERPERFORM BASELINE**

---

## Executive Summary

Successfully retrained LONG/SHORT Entry models with improved 2-of-3 labeling system, achieving **+151% precision improvement** in training. However, **backtest results show significant underperformance** compared to baseline models.

### Critical Findings

**Training Success**:
- LONG Precision: 13.7% → 34.4% (+151.3%)
- SHORT Precision: 30.3%, Recall: 58.1%

**Backtest Failure**:
- Win Rate: 60.0% → 49.8% (-17.0% WORSE)
- Returns: 13.93% → 8.40% (-39.7% WORSE)
- Trades: 35.3 → 49.2 (+39.1% MORE)

**Verdict**: Higher training precision does NOT translate to better trading performance.

---

## Detailed Comparison

### Performance Metrics

```yaml
Total Trades (per window):
  Baseline: 35.3 trades
  Improved: 49.2 trades
  Change: +13.8 (+39.1%) ⚠️ MORE OVERTRADING

LONG Trades:
  Baseline: 30.1 trades
  Improved: 33.0 trades
  Change: +2.9 (+9.5%)

SHORT Trades:
  Baseline: 5.2 trades
  Improved: 16.2 trades
  Change: +11.0 (+209.6%) ⚠️ MASSIVE INCREASE

Win Rate:
  Baseline: 60.0%
  Improved: 49.8%
  Change: -10.2% (-17.0%) ❌ WORSE

Avg Window Return:
  Baseline: 13.93%
  Improved: 8.40%
  Change: -5.52% (-39.7%) ❌ WORSE

Avg Position Size:
  Baseline: 50.8%
  Improved: 41.2%
  Change: -9.6% (-18.8%)
```

### Problematic Windows Analysis

```yaml
Windows with Win Rate < 40%:
  Baseline: 18 windows
  Improved: 34 windows
  Change: +16 windows ❌ WORSE

Windows with >50 Trades (Overtrading):
  Baseline: 24 windows
  Improved: 38 windows
  Change: +14 windows ❌ WORSE

Maximum Trades per Window:
  Baseline: 101 trades
  Improved: 131 trades
  Change: +30 trades ❌ WORSE
```

### Capital Distribution

```yaml
Final Capital per Window (Starting $10,000):
  Baseline:
    Mean: $11,392.51
    Median: $11,060.98
    Std: $1,550.40
    Min: $9,933.38
    Max: $19,990.73

  Improved:
    Mean: $10,840.38
    Median: $10,601.61
    Std: $896.98
    Min: $9,386.43
    Max: $13,860.29
```

### Overall Assessment

```yaml
Windows where Improved Model is Better:
  Win Rate: 23/100 (23.0%) ❌
  Returns: 33/100 (33.0%) ❌
  Fewer Trades: 22/100 (22.0%) ❌

Conclusion:
  Improved models only outperform in ~25-33% of windows
  Baseline models clearly superior in 67-77% of windows
```

---

## Root Cause Analysis

### 1. Training Precision vs Trading Performance

**Problem**: +151% precision improvement in training did NOT translate to better backtest performance.

**Why**:
- Training precision measures "% of positive predictions that are correct"
- Trading performance measures "% of trades that are profitable"
- These are NOT the same thing!

**Example**:
```python
Training Precision 34.4%: Of 100 predicted entries, 34.4 are labeled positive
Trading Win Rate 49.8%: Of 100 actual trades, 49.8 are profitable

# The disconnect:
# - Training labels = "Meets 2-of-3 criteria"
# - Trading profit = "Actual P&L after exit"
# - These measure different things!
```

### 2. Massive SHORT Overtrading

**Problem**: SHORT trades increased by **+209.6%** (5.2 → 16.2 trades/window)

**Impact**:
- More SHORT trades = More capital lock opportunities
- Opportunity Gating should prevent this, but didn't
- Suggests improved SHORT model has lower threshold or wrong calibration

**Analysis**:
```yaml
Baseline SHORT Entry: 5.2 trades/window (15% of all trades)
Improved SHORT Entry: 16.2 trades/window (33% of all trades)

Result:
  - More SHORT signals above 0.70 threshold
  - Suggests improved model is TOO sensitive
  - May need higher SHORT threshold (0.75-0.80?)
```

### 3. Lower Win Rate in Problematic Windows

**Problem**: Windows with WR < 40% increased from 18 to 34 (+16 windows)

**Analysis**:
- Baseline problematic windows: 18/100 (18%)
- Improved problematic windows: 34/100 (34%)
- **Nearly DOUBLED the failure rate**

**Why**:
- Improved models may be overfitting to training data
- 2-of-3 labeling might not capture real-world entry quality
- Feature engineering differences between training and backtest

### 4. Feature or Threshold Mismatch

**Possible Issues**:
1. **Feature Mismatch**: Improved models use 43 LONG features, baseline uses ?
2. **Threshold Mismatch**: Thresholds (0.65 LONG, 0.70 SHORT) were optimized for baseline
3. **Calibration Issues**: Improved model probabilities may not be well-calibrated

---

## Hypothesis: Why Did This Happen?

### Theory 1: Labeling Philosophy Mismatch

**2-of-3 Criteria**:
1. Profit Target (0.4% within 4h)
2. Early Entry Advantage (beats delayed entry)
3. Relative Performance (within 0.2% of best entry)

**Issue**: These criteria optimize for "good prediction targets" but NOT "profitable trades"

**Example Scenario**:
```
Entry at $100:
- Profit Target: +0.4% reached ✅
- Early Entry: Beats delayed entry ✅
- Relative Performance: Within 0.2% of best ✅
→ Label = 1 (2-of-3 criteria met)

But in actual trading:
- Entry at $100
- Exit at $100.30 (+0.3%)
- After fees: +0.2%
- ML Exit triggers at wrong time
→ Actual trade: Small profit or loss

The label said "good entry" but trade was mediocre
```

### Theory 2: Overfitting to Historical Patterns

**2-of-3 scoring** might be overfitting to specific historical patterns:
- Trained on 2025-07-01 to 2025-10-15 (same data as backtest)
- Labels might capture patterns specific to this period
- Doesn't generalize to different market conditions

### Theory 3: SHORT Model Calibration

**SHORT precision 30.3%** is actually LOWER than LONG precision 34.4%:
- Yet SHORT trades increased by +209%
- Suggests SHORT model is outputting higher probabilities
- May need recalibration or higher threshold

---

## Lessons Learned

### 1. Training Metrics ≠ Trading Performance

**Key Insight**: Optimizing training precision does not guarantee better trading results.

**Why**: Training measures "label correctness", trading measures "profit".

**Implication**: Need to validate ALL model improvements through backtesting before assuming they're better.

### 2. Labeling Quality Matters MORE Than Expected

**Discovery**: 2-of-3 scoring looked promising but failed in practice.

**Reason**: The criteria optimized for "predictable entries" not "profitable trades".

**Learning**: Exit labeling worked because exit criteria = actual trade outcomes. Entry labeling failed because entry criteria ≠ trade profitability.

### 3. Feature Engineering Requires Alignment

**Issue**: Improved models use different feature calculation pipeline.

**Risk**: Feature mismatch between training and inference can cause model degradation.

**Solution**: Ensure feature engineering is IDENTICAL between training and production.

### 4. User Feedback Was Right (Initially)

**User said**: "Shouldn't we optimize training before threshold optimization?"

**We did**: Improved training through better labeling → **Failed in backtest**

**Conclusion**: User was right to prioritize training, but the labeling approach needs rethinking.

---

## Next Steps

### Option 1: Threshold Optimization (Improved Models)

**Approach**: Try to salvage improved models by optimizing entry thresholds

**Rationale**:
- Current thresholds (0.65 LONG, 0.70 SHORT) optimized for baseline
- Improved models may need different thresholds
- SHORT threshold too low (causing +209% overtrading)

**Action**:
- Test LONG threshold range: 0.70-0.85
- Test SHORT threshold range: 0.75-0.90
- Find optimal balance

**Expected Outcome**: Might reduce overtrading, but unlikely to match baseline performance

**Risk**: Wasting time on fundamentally flawed models

### Option 2: Revert to Baseline Models

**Approach**: Keep using baseline entry models (current production)

**Rationale**:
- Baseline models achieve 60% win rate, 13.93% returns
- Already deployed and validated
- Improved models show no benefit

**Action**:
- Document improved model failure
- Continue with baseline models
- Focus on other optimizations (exit, sizing, etc.)

**Expected Outcome**: Maintain current performance

**Risk**: Missing potential improvements if threshold optimization would work

### Option 3: Rethink Entry Labeling

**Approach**: Design new entry labeling system based on actual trading outcomes

**Rationale**:
- Current 2-of-3 system doesn't capture trade profitability
- Need labels that align with actual trading performance
- Learn from Exit labeling success (exit criteria = trade outcomes)

**New Labeling Ideas**:
```yaml
Criteria:
  1. Trade Profitability: Entry + Exit yields >2% profit
  2. Risk-Reward: Max adverse move < 2%, max favorable > 4%
  3. Execution Quality: Entry near local optimal (within 0.5%)

Philosophy:
  - Label based on TRADE OUTCOME, not entry quality alone
  - Consider both entry AND exit in labeling
  - Optimize for actual trading metrics
```

**Action**:
- Design trade-outcome-based labeling
- Retrain models with new labels
- Backtest again

**Expected Outcome**: Better alignment between training and trading

**Risk**: Another round of experimentation (time investment)

### Option 4: Feature Engineering Review

**Approach**: Investigate feature differences between baseline and improved models

**Rationale**:
- Improved models use `calculate_all_features()`
- Baseline models use different feature pipeline
- Feature mismatch might be the issue

**Action**:
- Compare feature sets between baseline and improved
- Identify missing or extra features
- Re-train improved models with EXACT baseline features

**Expected Outcome**: Improved models with better alignment

**Risk**: If features aren't the issue, this won't help

---

## Recommendation

### Recommended Path: **Option 2 (Revert to Baseline) + Option 3 (Rethink Labeling)**

**Phase 1: Immediate** (Revert to Baseline)
- Keep using baseline entry models in production
- Document improved model failure
- Maintain current 60% WR, 13.93% returns

**Phase 2: Research** (Rethink Entry Labeling)
- Design trade-outcome-based labeling system
- Learn from Exit labeling success
- Consider full trade path (entry → hold → exit) in labels

**Phase 3: Validation** (If New Labeling Promising)
- Retrain models with trade-outcome labels
- Backtest thoroughly
- Only deploy if significant improvement (>15% return increase)

**Rationale**:
1. Don't deploy models that underperform baseline
2. Spend time on research, not on fixing fundamentally flawed approach
3. Learn from this failure to design better labeling
4. User's intuition was right: training matters, but we need the RIGHT training

---

## Files and Scripts

### Created Files:
```
src/labeling/improved_entry_labeling.py
scripts/experiments/test_improved_entry_labeling_simple.py
scripts/experiments/retrain_entry_models_improved_labeling.py
scripts/experiments/backtest_improved_entry_models.py
scripts/experiments/compare_entry_models.py
claudedocs/ENTRY_MODEL_IMPROVEMENT_20251018.md (initial report)
claudedocs/ENTRY_MODEL_BACKTEST_ANALYSIS_20251018.md (this report)
```

### Model Files:
```
models/xgboost_long_improved_labeling_20251018_051817.pkl
models/xgboost_long_improved_labeling_20251018_051817_scaler.pkl
models/xgboost_long_improved_labeling_20251018_051817_features.txt
models/xgboost_long_improved_labeling_20251018_051817_metadata.json

models/xgboost_short_improved_labeling_20251018_051822.pkl
models/xgboost_short_improved_labeling_20251018_051822_scaler.pkl
models/xgboost_short_improved_labeling_20251018_051822_features.txt
models/xgboost_short_improved_labeling_20251018_051822_metadata.json
```

### Result Files:
```
results/full_backtest_opportunity_gating_4x_20251018_042912.csv (BASELINE)
results/full_backtest_opportunity_gating_4x_20251018_053404.csv (IMPROVED)
```

---

## Conclusion

**Success**: Improved training precision by +151% (13.7% → 34.4%)

**Failure**: Backtest performance decreased by -39.7% (13.93% → 8.40% returns)

**Learning**: Training metrics don't guarantee trading performance. Need alignment between training labels and actual trading outcomes.

**Action**: Keep baseline models, rethink entry labeling approach.

**Quote**:
> "In theory, theory and practice are the same. In practice, they're not." - Yogi Berra
>
> **Applied**: Better training precision (theory) didn't improve trading (practice).

---

**Report Generated**: 2025-10-18 05:40 KST
**Analysis By**: Entry Model Improvement Project
**Status**: Baseline models remain superior - DO NOT deploy improved models
