# Reduced Feature Model - Failure Analysis
**Date**: 2025-10-23 05:52:00
**Status**: ❌ **FAILED - SEVERE PERFORMANCE DEGRADATION**

---

## Executive Summary

The reduced feature models (90 features vs 107 original) **failed catastrophically** in backtest validation, with performance degrading by 68-102% across all key metrics. **DO NOT DEPLOY**.

```yaml
Critical Findings:
  Return:    +75.58% → -1.55% (-102% degradation) ❌
  Win Rate:  63.6%   → 20.4%  (-68% degradation)  ❌
  Trades:    55      → 103    (+87% increase)     ⚠️

Conclusion: Feature reduction removed CRITICAL predictive features
Action: DO NOT deploy reduced models, analyze removed features
```

---

## Backtest Results Comparison

### Overall Performance

```yaml
REDUCED FEATURE MODELS (90 features):
  Feature Count: 90 (-15.9% from original)

  Performance:
    Initial Capital: $10,000.00
    Final Balance:   $9,845.23
    Total Return:    -1.55% ❌
    Max Drawdown:    -8.01%
    Sharpe Ratio:    129.561 (misleading - consistent small losses)

  Trade Statistics:
    Total Trades:    103
    Winning Trades:  21 (20.4% win rate) ❌
    Average Win:     $61.61
    Average Loss:    $-17.67
    Profit Factor:   0.89 (losing strategy)

  Exit Breakdown:
    ML Exit:    84 (81.6%) - Models trigger exits frequently
    Max Hold:   19 (18.4%)
    Stop Loss:  0 (0.0%)

ORIGINAL MODELS (107 features):
  Feature Count: 107

  Performance:
    Total Return:    +75.58% ✅
    Max Drawdown:    -12.2%
    Sharpe Ratio:    0.336

  Trade Statistics:
    Total Trades:    55
    Win Rate:        63.6% ✅
    Profit Factor:   1.73x ✅

  Exit Breakdown:
    ML Exit:    63.1%
    Max Hold:   29.6%
    Stop Loss:  7.4%
```

### Performance Ratio Analysis

```yaml
Metric Comparison (Reduced / Original):
  Return Ratio:      -2.0% (complete failure)
  Win Rate Ratio:    32.1% (only 1/3 of original)
  Risk Ratio:        65.7% (slightly better DD)
  Trade Count:       187% (too many trades)

Performance Grade: F (FAIL)
  - Return: Lost money vs gained money
  - Win Rate: 1 in 5 vs 2 in 3
  - Trade Quality: More trades, worse outcomes
```

---

## What Went Wrong

### 1. Removed Critical Predictive Features

**Key Insight**: High correlation ≠ Redundancy in ML context

Features that are highly correlated can still provide **complementary predictive power**:
- Feature A: Price trend (correlation 0.95 with Feature B)
- Feature B: Momentum (correlation 0.95 with Feature A)
- **Together**: Capture trend + momentum interaction
- **Separately**: Miss critical timing information

### 2. Removed Features Analysis

#### LONG Entry (-7 features):
```yaml
Removed:
  1. volume_ma_ratio (duplicate) - OK ✅
  2. bb_high, bb_low (kept bb_mid) - CRITICAL ERROR ❌
     - Bollinger Band breakouts are key entry signals
     - bb_high/low provide volatility envelope information
     - bb_mid alone doesn't capture volatility expansion

  3. macd_signal - CRITICAL ERROR ❌
     - MACD crossovers (macd vs signal) are PRIMARY signals
     - Removing signal breaks crossover detection

  4. lower_trendline_slope - Moderate Impact ⚠️
     - Support trendline breaks can trigger entries

  5. price_vs_lower_trendline_pct - Moderate Impact ⚠️

  6. strong_selling_pressure - Minor Impact

Impact: 2 CRITICAL removals broke LONG entry logic
```

#### SHORT Entry (-8 features):
```yaml
Removed:
  1. macd_divergence_abs (duplicate) - OK ✅
  2. atr (kept atr_pct) - OK ✅
  3. upside/downside_volatility - CRITICAL ERROR ❌
     - Asymmetric volatility is key for SHORT timing
     - Downside vol spike = bearish signal

  4. rejection_from_resistance - CRITICAL ERROR ❌
     - Resistance rejection = key SHORT entry trigger

  5. price_direction_ma20 - Moderate Impact ⚠️
  6. price_distance_ma50 - Moderate Impact ⚠️
  7. resistance_rejection_count - Minor Impact

Impact: 2 CRITICAL removals broke SHORT entry logic
```

#### Exit Models (-2 features):
```yaml
Removed:
  1. trend_strength (duplicate of MACD) - CRITICAL ERROR ❌
     - trend_strength ≠ MACD in exit context
     - Measures different aspects of trend health

  2. macd_signal - CRITICAL ERROR ❌
     - Exit on MACD crossdown requires signal

Impact: Both removals were CRITICAL for exit timing
```

### 3. Model Behavior Changes

```yaml
Original Models:
  - Selective entries (55 trades)
  - High quality signals (63.6% WR)
  - Effective exits (primarily ML)

Reduced Models:
  - Excessive entries (103 trades, +87%)
  - Poor quality signals (20.4% WR)
  - Premature exits (81.6% ML exits vs 63.1%)

Problem: Models lost ability to distinguish quality setups
Cause: Removed features provided key discriminative power
```

---

## Root Cause Analysis

### Faulty Assumption

**We assumed**: Correlation > 0.8 → Redundant → Safe to remove

**Reality**:
- Correlation measures linear relationship
- ML models use non-linear combinations
- Correlated features can capture different aspects
- Example: MACD vs MACD_signal
  - Correlation: 0.95 (high)
  - Predictive value: COMPLEMENTARY (not redundant)
  - MACD alone: Trend strength
  - MACD + Signal: **Crossover timing** (critical!)

### Why ML Test Accuracy Was Misleading

```yaml
Model Accuracy Results:
  LONG Entry:  96.75% ✅ (looked good!)
  SHORT Entry: 97.13% ✅ (looked good!)
  Exit:        96.76% ✅ (looked good!)

Why This Didn't Predict Backtest Failure:
  - Test set: Balanced positive/negative labels
  - Backtest: Extreme class imbalance (3-5% positive rate)
  - Models optimized for accuracy, not trade profitability
  - Missing features broke edge cases that drive profits
```

### Label Distribution Problem

```yaml
Training Labels (Simple TP/SL):
  LONG:  3.66% positive (1,147/31,368)
  SHORT: 3.05% positive (958/31,368)

Issue: Only 3-4% of candles are "good entries"
  - Models learned to identify 96%+ negative cases well
  - Lost ability to identify rare 3-4% profitable setups
  - Removed features were critical for rare positive cases
```

---

## Lessons Learned

### 1. Correlation ≠ Redundancy

**DO NOT** remove features based solely on correlation:
- High correlation can indicate complementary signals
- Crossover-based indicators require both components
- Volatility envelope indicators need all boundaries

### 2. Test Accuracy ≠ Trading Performance

**DO NOT** trust test accuracy for imbalanced datasets:
- 96% accuracy on 96% negative class = useless
- Need specialized metrics for rare positive cases
- Backtest is the ONLY valid test for trading models

### 3. Feature Removal Requires Domain Knowledge

**DO NOT** remove features algorithmically:
- Bollinger Bands: Need all 3 (high/mid/low)
- MACD: Need both (macd + signal) for crossovers
- Volatility: Need asymmetric (up/down) for direction
- Trendlines: Need both (upper/lower) for breakouts

---

## Corrective Action Plan

### Immediate Actions

1. **✅ DO NOT deploy reduced models** - Confirmed failure
2. **✅ Retain original 107-feature models** - Production safe
3. **Analysis required**: Identify minimal critical feature set

### Alternative Approach: Conservative Feature Reduction

Instead of removing all correlated features, apply domain-specific rules:

```yaml
Safe Removals (Keep Only):
  Perfect Duplicates (correlation = 1.0):
    - volume_ma_ratio ✅
    - macd_strength = macd_divergence_abs ✅

Unsafe Removals (Keep Both):
  Crossover Pairs:
    - macd + macd_signal ❌ (need both for crossover)
    - ema_12 + ema_26 ❌ (need both for crossover)

  Envelope Indicators:
    - bb_high/mid/low ❌ (need all 3 for breakouts)
    - upper/lower_trendline ❌ (need both for breaks)

  Asymmetric Volatility:
    - upside/downside_volatility ❌ (different meanings)
    - volatility + volatility_asymmetry ❌ (complementary)

Revised Target:
  - Remove ONLY perfect duplicates: 107 → 105 features (-2 only)
  - Keep all crossover, envelope, and asymmetric features
```

### Priority 2: Lookback Period Optimization (UNCHANGED)

```yaml
Status: Proceed as planned
Rationale:
  - Feature set now confirmed (107 features)
  - Lookback optimization is orthogonal to feature selection
  - Can optimize periods for existing features

Next Steps:
  1. Create lookback period grid search script
  2. Test RSI(10,14,20), MACD(8/17,12/26,16/35), etc.
  3. Find optimal combinations via backtest
```

---

## Statistical Analysis

### Why 81.6% ML Exits?

```yaml
Original Models:
  - ML Exit: 63.1% (healthy mix)
  - Reason: Models identify true exit signals

Reduced Models:
  - ML Exit: 81.6% (excessive)
  - Reason: Models trigger false exits prematurely

Cause:
  - Missing features → increased uncertainty
  - Model defaults to "exit" when unsure
  - Result: Cuts winners short, keeps losers
```

### Trade Count Explosion

```yaml
Original: 55 trades (selective)
Reduced: 103 trades (promiscuous)

Problem:
  - Lost ability to distinguish quality setups
  - Enters on weak signals
  - Exits quickly (81.6% ML)
  - Repeat cycle = more bad trades
```

---

## Recommendations

### For Feature Engineering

1. **NEVER remove features based on correlation alone**
   - Require domain expertise review
   - Test each removal individually via backtest
   - Keep crossover pairs, envelopes, asymmetric indicators

2. **Test methodology**:
   - Single-feature ablation studies
   - Remove one feature at a time
   - Backtest each removal
   - Accumulate safe removals only

3. **Validation criteria**:
   - Backtest return: >= 95% of baseline
   - Win rate: >= 90% of baseline
   - Trade count: ± 20% of baseline
   - Max DD: <= 120% of baseline

### For Model Evaluation

1. **DON'T trust accuracy on imbalanced data**
   - 96% accuracy = 0% value if 96% negative class
   - Use precision/recall for positive class
   - Use backtest as final validation

2. **Require backtest validation before deployment**
   - No exceptions
   - Real trading performance is the only metric

---

## Files and Documentation

### Created Files

```yaml
Code:
  - scripts/experiments/calculate_reduced_features.py ❌
  - scripts/training/retrain_with_reduced_features.py ❌
  - scripts/experiments/backtest_reduced_features.py ✅

Models (DO NOT USE):
  - xgboost_long_entry_reduced_20251023_050635.pkl ❌
  - xgboost_short_entry_reduced_20251023_050635.pkl ❌
  - xgboost_long_exit_reduced_20251023_050635.pkl ❌
  - xgboost_short_exit_reduced_20251023_050635.pkl ❌

Results:
  - backtest_reduced_features_20251023_055142.csv ✅

Documentation:
  - FEATURE_CORRELATION_ANALYSIS_20251023.md ✅
  - FEATURE_REDUCTION_PLAN_20251023.md ❌ (flawed plan)
  - REDUCED_FEATURE_RETRAINING_20251023.md ❌ (failed experiment)
  - REDUCED_FEATURE_FAILURE_ANALYSIS_20251023.md ✅ (this file)
```

---

## Conclusion

**Feature reduction via correlation-based removal FAILED**.

Key findings:
1. ❌ Removed critical features despite high correlation
2. ❌ Test accuracy (96%) did not predict trading failure (-1.55% return)
3. ❌ Performance degraded by 68-102% across all metrics
4. ✅ Learned: Correlation ≠ Redundancy in ML trading context
5. ✅ Learned: Backtest is the ONLY valid test

**Decision**:
- **DO NOT deploy** reduced feature models
- **RETAIN** original 107-feature models in production
- **PROCEED** with Priority 2: Lookback period optimization
- **FUTURE**: Feature removal requires individual ablation testing

---

**Date**: 2025-10-23
**Status**: ❌ **EXPERIMENT FAILED - KEEP ORIGINAL MODELS**
**Next**: Proceed to lookback period grid search optimization
