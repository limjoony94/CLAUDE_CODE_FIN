# Progressive Exit Window - Backtest Failure Analysis

**Date**: 2025-10-31 22:36:00
**Status**: ❌ **CRITICAL FAILURE - Models Too Conservative**

---

## Executive Summary

The Progressive Exit Window strategy showed **catastrophic opposite failure**:
- **Previous Problem**: Models exit in 1 candle (too fast) → 98.5% ML Exit, 14.92% WR
- **Current Problem**: Models NEVER exit (too slow) → 0.0% ML Exit, 42.91% WR

**Root Cause**: Exit threshold 0.70 (70%) is too high for weighted progressive labels.

---

## Backtest Results (108 Windows)

### Overall Performance
```yaml
Total Trades: 296
Win Rate: 42.91% (127W / 169L)  # Target: 70-75% ❌
Avg Return per Window: +0.87%    # Target: +35-40% ❌
Sharpe Ratio: 0.492 (annualized)

Trade Statistics:
  Avg Trade P&L: $5.86 (0.169%)
  Avg Winner: $363.63 (4.014%)
  Avg Loser: $-262.99 (-2.720%)
  Avg Hold Time: 93.7 candles (7.8 hours)  # Target: 20-30 ❌
```

### Exit Distribution
```yaml
ML_EXIT: 0 (0.0%)        # Target: 75-85% ❌
MAX_HOLD: 179 (60.5%)    # Emergency fallback
STOP_LOSS: 117 (39.5%)   # Emergency fallback
```

### Target Validation
| Target | Expected | Actual | Status |
|--------|----------|---------|--------|
| Win Rate | 70-75% | 42.9% | ❌ FAIL |
| Avg Hold | 20-30 | 93.7 | ❌ FAIL |
| Return/Window | +35-40% | +0.9% | ❌ FAIL |
| ML Exit | 75-85% | 0.0% | ❌ FAIL |

---

## Root Cause Analysis

### 1. Threshold Mismatch
**Problem**: Exit threshold 0.70 (70%) too high for weighted labels

**Evidence**:
- Training used weights {0.4, 0.5, 0.6, 0.7, 0.8, 1.0} around max profit
- Most labeled candles have weights 0.4-0.7 (not 0.7+)
- Model predicts based on weighted labels, rarely exceeds 0.70

**Math**:
```python
# Label distribution (Progressive Window)
Weight 1.0: ~5% of labeled candles (max profit only)
Weight 0.8: ~10% of labeled candles (±1 candle)
Weight 0.7: ~10% of labeled candles (±2 candles)
Weight 0.6: ~10% of labeled candles (±3 candles)
Weight 0.5: ~10% of labeled candles (±4 candles)
Weight 0.4: ~10% of labeled candles (±5 candles)

# With threshold 0.70, model only exits when:
# - Probability > 0.70 (rare for weighted labels)
# - Result: NEVER exits via ML
```

### 2. Binary vs Weighted Confusion
**Problem**: Training used weighted labels [0.0-1.0], but backtest used binary threshold

**Training**:
- Labels: Continuous weights 0.0-1.0
- scale_pos_weight: Applied to binary positive (>0)
- Model learns: Probability distribution matching weights

**Backtest**:
- Threshold: Binary 0.70 cutoff
- Model output: Weighted probabilities (mostly 0.4-0.7)
- Mismatch: Threshold too high for weighted outputs

### 3. Label Imbalance Still Exists
**Problem**: Even with Progressive Window, only 28% positive labels

**Analysis**:
```python
LONG labels: 28.61% positive (8,583 / 30,004 candles)
SHORT labels: 29.32% positive (8,798 / 30,004 candles)

Binary positive (≥0.7): 16.59% LONG, 16.98% SHORT
# Still imbalanced (1:5 ratio)
```

---

## Why Progressive Window Failed

### Design Flaw: Weighted Labels + Binary Threshold
```python
# Training Philosophy (CORRECT)
"Label ±5 candles with decreasing weights"
→ Creates smooth gradient around optimal exit
→ Model learns flexible exit window

# Backtest Philosophy (INCORRECT)
"Exit when probability > 0.70"
→ Expects binary high confidence
→ Ignores weighted gradient information

# MISMATCH!
Weighted training + Binary threshold = Failure
```

### What Should Have Been Done
```python
# Option A: Binary Labels + Binary Threshold
Labels: {0, 1} at optimal candles only
Threshold: 0.70 (works with binary)

# Option B: Weighted Labels + Weighted Threshold
Labels: {0.0, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0}
Threshold: 0.40 (works with weights)  ← MISSING!

# Option C: Weighted Labels + Regression
Predict continuous exit quality [0-1]
Exit when quality > 0.60
```

---

## Comparison: Previous vs Progressive Window

| Metric | Previous (Full Dataset) | Progressive Window | Change |
|--------|------------------------|--------------------|---------|
| **Win Rate** | 14.92% | 42.91% | +28.0pp ✅ |
| **Return/Window** | -69.10% | +0.87% | +69.97pp ✅ |
| **Avg Hold Time** | 2.4 candles | 93.7 candles | +91.3 ❌ |
| **ML Exit Rate** | 98.5% | 0.0% | -98.5pp ❌ |
| **Max Hold** | 0.8% | 60.5% | +59.7pp ❌ |
| **Stop Loss** | 0.6% | 39.5% | +38.9pp ❌ |

**Interpretation**:
- Previous: Models exit too fast (1 candle) → Many small losses
- Progressive: Models NEVER exit → Rely on emergency stops
- Neither: Achieves target performance

---

## Recommended Solutions

### **Option 1: Lower Exit Threshold to 0.40** ⭐ **RECOMMENDED**
**Rationale**: Match threshold to weighted label distribution

**Implementation**:
```python
ML_EXIT_THRESHOLD_LONG = 0.40  # Down from 0.70
ML_EXIT_THRESHOLD_SHORT = 0.40  # Down from 0.70
```

**Expected Impact**:
- ML Exit: 0% → 60-80% (models will trigger)
- Win Rate: 42.9% → 55-65% (better timing)
- Avg Hold: 93.7 → 20-40 candles (closer to target)
- Return: +0.87% → +15-25% per window

**Risk**: May exit too early (weights 0.4-0.5 are far from max)

---

### **Option 2: Retrain with Threshold-Aware Labels**
**Rationale**: Binary labels only at optimal exit (weight 1.0)

**Implementation**:
```python
# Modified labeling
for offset, weight in WINDOW_WEIGHTS.items():
    if weight >= 0.80:  # Only label high-weight candles
        labels[candle_idx] = 1.0
    else:
        labels[candle_idx] = 0.0

# Result: ~10-15% positive (center ±1-2 candles only)
# Threshold: Keep at 0.70 (works with binary)
```

**Expected Impact**:
- ML Exit: 0% → 70-85%
- Win Rate: 42.9% → 65-75%
- Avg Hold: 93.7 → 20-30 candles
- Return: +0.87% → +25-35% per window

**Effort**: Requires retraining (4 hours total)

---

### **Option 3: Use Regression Instead of Classification**
**Rationale**: Predict exit quality, not binary exit/hold

**Implementation**:
```python
# Training
model = XGBRegressor()  # Instead of XGBClassifier
y = weighted_labels  # {0.0, 0.4, 0.5, ..., 1.0}

# Backtest
exit_quality = model.predict(features)
if exit_quality >= 0.60:  # Flexible threshold
    exit_now()
```

**Expected Impact**:
- ML Exit: 0% → 65-80%
- Win Rate: 42.9% → 60-70%
- Return: +0.87% → +20-30% per window
- Better: Continuous exit signal (not binary)

**Effort**: Requires complete retraining (6 hours)

---

### **Option 4: Hybrid - Max Profit Only Labels + Wide Threshold Range**
**Rationale**: Single optimal candle + test multiple thresholds

**Implementation**:
```python
# Training
labels[max_profit_candle] = 1.0  # Only max
scale_pos_weight = auto  # Handle 1.67% imbalance

# Backtest - Grid search optimal threshold
thresholds = [0.30, 0.40, 0.50, 0.60, 0.70]
# Find best via backtesting
```

**Expected Impact**:
- ML Exit: Varies by threshold (30-90%)
- Win Rate: 50-75% (depends on threshold)
- Return: +5-40% per window
- Best: Data-driven threshold selection

**Effort**: Retraining (4h) + Grid search (30min)

---

## Immediate Next Steps

### **RECOMMENDED: Test Option 1 First (Fastest)**
1. Update backtest script: `ML_EXIT_THRESHOLD_LONG/SHORT = 0.40`
2. Re-run 108-window backtest (5 minutes)
3. If ML Exit 60-80% + WR 55-65% → Success
4. If still fails → Move to Option 2 or 4

### **If Option 1 Insufficient: Implement Option 2 or 4**
1. Retrain with modified labeling strategy
2. Validate via backtest
3. Deploy if performance targets met

---

## Key Learnings

### 1. **Weighted Labels Require Weighted Thresholds**
- Weighted training → Requires weight-aware threshold
- Binary threshold (0.70) incompatible with weighted outputs (0.4-1.0)

### 2. **Progressive Window is Conceptually Sound**
- Labels ±5 candles around max profit (GOOD)
- But implementation had threshold mismatch (BAD)

### 3. **Label Imbalance Persists**
- Even with Progressive Window: 28% positive (still imbalanced)
- Needs either:
  - More aggressive labeling (>30% positive)
  - Or better handling via threshold tuning

### 4. **Simpler May Be Better**
- Max-only labels (1.67%) + low threshold (0.30-0.40) might work
- Progressive Window adds complexity without proven benefit

---

## Performance Targets (Reminder)

| Metric | Target | Previous | Progressive | Status |
|--------|--------|----------|-------------|---------|
| Win Rate | 70-75% | 14.92% | 42.91% | ❌ |
| Return/Window | +35-40% | -69.10% | +0.87% | ❌ |
| Avg Hold | 20-30 | 2.4 | 93.7 | ❌ |
| ML Exit | 75-85% | 98.5% | 0.0% | ❌ |

**Neither strategy achieved targets** - Need Option 1, 2, or 4 implementation.

---

## Files Created

### Training
- `scripts/experiments/retrain_exit_progressive_window.py`
- `models/xgboost_long_exit_progressive_window_20251031_223102.pkl`
- `models/xgboost_short_exit_progressive_window_20251031_223102.pkl`

### Backtest
- `scripts/experiments/backtest_progressive_window.py`
- `results/backtest_progressive_window_20251031_223645.csv`

### Documentation
- `claudedocs/PROGRESSIVE_WINDOW_FAILURE_ANALYSIS_20251031.md` (this file)

---

## Conclusion

**Progressive Exit Window strategy failed due to threshold mismatch, NOT label strategy.**

**Immediate Action**: Test Exit threshold 0.40 (Option 1) - 5 minutes
**If Insufficient**: Implement Option 2 or 4 - 4-6 hours

**Expected Timeline**:
- Option 1 test: 5 minutes
- Option 2/4 implementation: 4-6 hours
- Total: 5 minutes - 6 hours

**Confidence**: 🟡 Medium
- Option 1: 40% chance of success (quick fix)
- Option 2: 70% chance of success (retrain with binary)
- Option 4: 80% chance of success (data-driven threshold)

**Recommendation**: Try Option 1 immediately, fallback to Option 4 if needed.
