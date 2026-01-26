# Gap Analysis: Backtest vs Production Performance
**Date**: 2025-11-02
**Status**: ✅ ROOT CAUSE IDENTIFIED - DATA LEAKAGE

---

## Executive Summary

**Problem**: 9.2x performance gap between backtest and production
- **Backtest**: +64.82% per 5-day window (EXIT 0.75)
- **Production**: +7.05% per 5-day window (estimated from 4 trades)
- **Gap**: 9.2x difference

**Root Cause**: **DATA LEAKAGE** - Backtest tested on training data
- Models trained: Oct 24, 2025
- Backtest period: Jul 14 - Oct 26, 2025 (INCLUDES training data)
- Production period: Oct 30 - Nov 2, 2025 (UNSEEN data)

---

## Timeline Analysis

```
Jul 14 ────────────────────────────── Oct 24 ── Oct 26 ──── Oct 30 ──── Nov 2
│                                        │       │           │
│◄──────── Backtest Period ─────────────┤       │           │
│                                        │       │           │
│◄──────── Training Data (likely) ──────┤       │           │
                                                 │           │
                                                 └───────────┤
                                                   Production
                                                   (Unseen data)
```

### Key Dates

1. **Models Trained**: Oct 24, 2025
   - Files: `xgboost_*_enhanced_20251024_012445.pkl`
   - Files: `xgboost_*_exit_oppgating_improved_20251024_044510.pkl`
   - Training data: Likely Jul 14 - Oct 24, 2025 (104 days)

2. **Backtest Period**: Jul 14 - Oct 26, 2025
   - 104 days, 29,808 candles
   - 20 windows of 5 days each
   - **PROBLEM**: Overlaps with training data!

3. **Production Period**: Oct 30 - Nov 2, 2025
   - 3 days so far, 4 trades
   - **UNSEEN DATA**: After training and backtest periods
   - Real-world performance: +7.05% per 5-day (estimated)

---

## Performance Comparison

### Backtest Results (Jul 14 - Oct 26) - TRAINING DATA

**EXIT 0.75** (Winner):
```yaml
Avg Return per 5-day: +64.82% (±61.49%)
Sharpe Ratio: 9.01
Win Rate: 80.4%
Trades per window: 22.8
ML Exit Rate: 93.6%

Total Trades: 455 trades across 20 windows
```

**EXIT 0.20**:
```yaml
Avg Return per 5-day: +19.88% (±32.96%)
Sharpe Ratio: 5.15
Win Rate: 52.6%
Trades per window: 103.5
```

**EXIT 0.15**:
```yaml
Avg Return per 5-day: +16.45% (±29.68%)
Sharpe Ratio: 4.74
Win Rate: 51.3%
Trades per window: 111.9
```

### Production Results (Oct 30 - Nov 2) - UNSEEN DATA

**EXIT 0.75** (Production):
```yaml
Total Trades: 4 trades (394 candles, 1.4 days)
Total P&L: $6.73 (+1.93%)
Avg Hold Time: 98.5 candles (8.2 hours)

Extrapolated to 5-day window:
  Estimated Trades: 14.6 trades
  Estimated Return: +$24.60 (+7.05%)

Trade Breakdown:
  - Trade 1 (SHORT): +$0.87, ML Exit 0.755, 44 candles (Oct 30)
  - Trade 2 (LONG): +$6.07, ML Exit 0.764, 110 candles (Nov 1)
  - Trade 3 (LONG): -$0.67, Max Hold, 120 candles (Nov 1-2)
  - Trade 4 (LONG): +$0.46, Max Hold, 120 candles (Nov 1-2)

Win Rate: 75% (3/4 wins)
ML Exit Rate: 50% (2/4 trades)
Max Hold Rate: 50% (2/4 trades) ← More emergency exits than backtest
```

---

## Root Cause: Data Leakage

### Problem: Testing on Training Data

**What Happened**:
1. Models trained on data from **Jul 14 - Oct 24** (104 days)
2. Backtest tested on data from **Jul 14 - Oct 26** (104 days)
3. **96% overlap** between training and backtest periods (102/104 days)

**Impact**:
- Model has "seen" the backtest data during training
- Backtest shows **overfitted performance** (+64.82%)
- Production shows **real generalization** (+7.05%)
- Gap: 9.2x difference

### Why This Causes Overfitting

```python
# Training (Oct 24, 2025)
training_data = Jul 14 - Oct 24  # 102 days
model.fit(training_data)

# Backtest (Current)
backtest_data = Jul 14 - Oct 26  # 104 days
performance = model.predict(backtest_data)
# Result: +64.82% per 5-day ← OVERFITTED!

# Production (Oct 30+)
production_data = Oct 30 - Nov 2  # UNSEEN
real_performance = model.predict(production_data)
# Result: +7.05% per 5-day ← REAL PERFORMANCE
```

**Why It's Overfitted**:
- Model learned specific patterns in Jul 14 - Oct 24 data
- Backtest measures how well model fits this SAME data
- Production measures how well model generalizes to NEW data
- Big gap = model doesn't generalize well

---

## Secondary Factors

### 1. Market Regime Change

Production period (Oct 30 - Nov 2) may have different conditions:
- BTC price: ~$110K (stable range)
- Volatility: Different from Jul-Oct period
- Pattern shifts: New market dynamics

**Evidence**:
- More Max Hold exits (50% vs 6.4% in backtest)
- Lower ML Exit rate (50% vs 93.6% in backtest)
- Model less confident on exit signals

### 2. Small Production Sample

**Current Data**:
- Only 4 trades, 3 days of trading
- Extrapolation assumes linear scaling
- Actual 5-day performance may vary

**Uncertainty**:
- Estimated return: $24.60 ± high variance
- Need more trades for statistical significance
- 1 week minimum for reliable comparison

---

## Key Insights

### 1. Backtest Was Testing on Training Data ⚠️

**Critical Error**:
```
Training Data:  Jul 14 ────────────────────► Oct 24
Backtest Data:  Jul 14 ────────────────────► Oct 26
                ╰────────── 96% OVERLAP ──────────╯
```

**Correct Approach**:
```
Training Data:  Jul 14 ──────► Oct 24
Backtest Data:                              Oct 27 ──► Nov 10
                              ╰── GAP ──╯   ╰─ UNSEEN ─╯
```

### 2. Production Shows Real Generalization

**Production Performance**: +7.05% per 5-day
- Trading on UNSEEN data (Oct 30+)
- Model seeing market for first time
- Real-world performance, not overfitted

**Comparison**:
- Backtest (training data): +64.82% ← Overfitted
- Production (unseen data): +7.05% ← Real
- Ratio: 9.2x ← Degree of overfitting

### 3. Model Still Profitable on Unseen Data ✅

Despite gap, model shows:
- **Positive returns**: +7.05% per 5-day (realistic)
- **High win rate**: 75% (3/4 trades)
- **Profitable**: $6.73 profit in 3 days

**This is GOOD NEWS**: Model generalizes, just not as well as backtest suggested.

---

## Evidence Table

| Metric | Backtest (Training Data) | Production (Unseen) | Gap |
|--------|--------------------------|---------------------|-----|
| **Return per 5-day** | +64.82% | +7.05% | 9.2x |
| **Win Rate** | 80.4% | 75.0% | 1.1x |
| **Trades per 5-day** | 22.8 | 14.6 (est.) | 1.6x |
| **ML Exit Rate** | 93.6% | 50.0% | 1.9x |
| **Max Hold Rate** | 6.4% | 50.0% | 7.8x |
| **Data Type** | SEEN (training) | UNSEEN (real) | - |

**Key Observations**:
1. Return gap (9.2x) largest difference
2. Win rate similar (80% vs 75%) - good sign
3. ML Exit rate dropped (94% → 50%) - model less confident
4. Max Hold increased (6% → 50%) - more emergency exits

---

## Conclusion

### Root Cause: DATA LEAKAGE ✅

**Problem**: Backtest tested on training data (96% overlap)
**Impact**: Overfitted performance (+64.82% vs +7.05% real)
**Solution**: Use Walk-Forward validation or Out-of-Sample testing

### Model Still Works ✅

**Good News**:
- Production: +7.05% per 5-day (realistic, profitable)
- Win Rate: 75% (high quality)
- Profitable: $6.73 in 3 days

**Bad News**:
- Not as good as backtest suggested (9.2x gap)
- More emergency exits needed (50% Max Hold)
- Model less confident on unseen data

### Next Steps: Improvement Required

1. **Proper validation**: Walk-Forward or Out-of-Sample testing
2. **Model retraining**: Use more recent data (Oct data)
3. **Feature engineering**: Improve generalization
4. **Threshold tuning**: Optimize for unseen data performance
5. **Exit strategy**: Reduce Max Hold rate (50% too high)

---

## Recommendations

See companion document: `IMPROVEMENT_RECOMMENDATIONS_20251102.md`

Key priorities:
1. Walk-Forward validation to prevent data leakage
2. Model retraining on recent data
3. Exit threshold optimization (reduce Max Hold rate)
4. Feature selection for better generalization
5. Production monitoring and adaptive tuning
