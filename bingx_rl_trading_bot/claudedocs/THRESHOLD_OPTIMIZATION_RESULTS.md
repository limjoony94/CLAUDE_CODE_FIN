# Threshold Optimization Results

**Date**: 2025-10-15
**Strategy**: Clean Slate - Optimize Phase 4 model thresholds only
**Status**: ✅ **EXECUTION COMPLETE** (no more analysis)

---

## Executive Summary

**Critical Discovery**: SHORT model barely trading at threshold 0.70!

**Current State**:
- LONG: 25.88 trades/week (99% of all trades)
- SHORT: 0.32 trades/week (1 trade per 3 weeks!)
- **Problem**: System is 99% LONG-only despite having SHORT capability

**Optimal Configuration**:
- LONG threshold: 0.80 (increase from 0.70)
- SHORT threshold: 0.50 (decrease from 0.70)

**Expected Impact**:
- LONG: 17.70 trades/week (-31.6%, more selective)
- SHORT: 9.37 trades/week (+2850%, actually trading)
- **Better balance**: 65% LONG / 35% SHORT (was 99% / 1%)

---

## Detailed Results

### LONG Entry Model

| Threshold | Trades/Week | Precision | Recall | F1 | Signals |
|-----------|-------------|-----------|--------|-----|---------|
| 0.50 | 45.72 | 17.01% | 70.00% | 27.37% | 576 |
| 0.55 | 37.78 | 19.12% | 65.00% | 29.55% | 476 |
| 0.60 | 32.78 | 20.82% | 61.43% | 31.10% | 413 |
| 0.65 | 29.29 | 23.04% | 60.71% | 33.40% | 369 |
| **0.70 (current)** | **25.88** | **25.15%** | **58.57%** | **35.19%** | **326** |
| 0.75 | 22.23 | 26.79% | 53.57% | 35.71% | 280 |
| **0.80 (optimal)** | **17.70** | **27.80%** | **44.29%** | **34.16%** | **223** |

**Probability Distribution**:
- Mean: 0.2190
- Std: 0.2213
- Range: [0.0016, 0.9893]

**Recommendation**: **Increase to 0.80**
- Trade-off: -31.6% trades, +10.5% precision
- More selective, higher quality LONG entries

### SHORT Entry Model

| Threshold | Trades/Week | Precision | Recall | F1 | Signals |
|-----------|-------------|-----------|--------|-----|---------|
| **0.50 (optimal)** | **9.37** | **16.10%** | **13.97%** | **14.96%** | **118** |
| 0.55 | 5.64 | 14.08% | 7.35% | 9.66% | 71 |
| 0.60 | 2.62 | 27.27% | 6.62% | 10.65% | 33 |
| 0.65 | 1.27 | 25.00% | 2.94% | 5.26% | 16 |
| **0.70 (current)** | **0.32** | **50.00%** | **1.47%** | **2.86%** | **4** |
| 0.75 | 0.24 | 66.67% | 1.47% | 2.88% | 3 |
| 0.80 | 0.00 | 0.00% | 0.00% | 0.00% | 0 |

**Probability Distribution**:
- Mean: 0.1757
- Std: 0.1251
- Range: [0.0022, 0.7962]

**Recommendation**: **Decrease to 0.50**
- **Critical**: Current 0.70 generates only 0.32 trades/week (1 every 3 weeks!)
- At 0.50: 9.37 trades/week (+2850% increase)
- Trade-off: Lower precision (16.10% vs 50.00%), but actually generates trades

---

## Key Insights

### 1. SHORT Model Severely Underfired
```
Current: 0.70 threshold → 0.32 trades/week → 99% LONG system
Optimal: 0.50 threshold → 9.37 trades/week → Balanced LONG/SHORT
```

**Why?**
- SHORT model has lower probability distribution (mean 0.1757 vs LONG 0.2190)
- At 0.70, SHORT model generates only 4 signals in 2.5 weeks (basically nothing)
- The 50% precision at 0.70 is misleading (only 2 true positives out of 4)

### 2. LONG/SHORT Balance
```
Current configuration (0.70 / 0.70):
  - 25.88 LONG + 0.32 SHORT = 26.2 total
  - 99% LONG, 1% SHORT (unbalanced)

Optimal configuration (0.80 / 0.50):
  - 17.70 LONG + 9.37 SHORT = 27.1 total
  - 65% LONG, 35% SHORT (balanced)
```

**Impact**:
- Similar total trade frequency (26.2 → 27.1)
- Much better directional balance
- Can capture both bull and bear moves

### 3. Precision vs Recall Trade-off

**LONG (0.70 → 0.80)**:
- Precision: 25.15% → 27.80% (+10.5%)
- Recall: 58.57% → 44.29% (-24.4%)
- **Strategy**: More selective, miss some moves but higher win rate

**SHORT (0.70 → 0.50)**:
- Precision: 50.00% → 16.10% (-67.8%)
- Recall: 1.47% → 13.97% (+850%)
- **Strategy**: Much less selective, catch more moves despite lower precision

---

## Implementation Recommendation

### Option A: Full Optimal (Recommended)

**Change**:
```python
# In phase4_dynamic_testnet_trading.py
LONG_ENTRY_THRESHOLD = 0.80  # was 0.70
SHORT_ENTRY_THRESHOLD = 0.50  # was 0.70
```

**Expected Outcome**:
- Total trades: 27.1/week (similar to current)
- LONG/SHORT balance: 65% / 35% (much better)
- LONG quality: +10.5% precision
- SHORT activation: Actually trades SHORT

**Risk**: LOW
- Total trade frequency unchanged
- Only rebalancing between directions
- SHORT precision drops but at least it trades

### Option B: Conservative (Lower Risk)

**Change**:
```python
# In phase4_dynamic_testnet_trading.py
LONG_ENTRY_THRESHOLD = 0.75  # was 0.70 (moderate increase)
SHORT_ENTRY_THRESHOLD = 0.55  # was 0.70 (moderate decrease)
```

**Expected Outcome**:
- LONG: 22.23 trades/week, 26.79% precision
- SHORT: 5.64 trades/week, 14.08% precision
- Total: 27.87/week
- Balance: 80% LONG / 20% SHORT

**Risk**: LOWER
- Smaller changes from current
- Still improves SHORT activation
- Less precision drop for SHORT

### Option C: Test Separately (Lowest Risk)

**Phase 1**: Increase LONG threshold only
```python
LONG_ENTRY_THRESHOLD = 0.80
SHORT_ENTRY_THRESHOLD = 0.70  # unchanged
```
Run 1 week, observe LONG quality improvement

**Phase 2**: Decrease SHORT threshold
```python
LONG_ENTRY_THRESHOLD = 0.80
SHORT_ENTRY_THRESHOLD = 0.50  # or 0.55
```
Run 1 week, observe SHORT activation

**Risk**: LOWEST
- A/B test each change
- Can rollback independently
- More data to validate

---

## Decision Matrix

| Option | LONG Change | SHORT Change | Total Risk | Expected Benefit | Time to Validate |
|--------|-------------|--------------|------------|------------------|------------------|
| A (Full) | +0.10 | -0.20 | Low | High (balanced) | 2-4 weeks |
| B (Conservative) | +0.05 | -0.15 | Lower | Medium | 2-4 weeks |
| C (Phased) | Sequential | Sequential | Lowest | Validated | 4-6 weeks |

---

## My Recommendation: **Option A (Full Optimal)**

**Rationale**:
1. **SHORT activation is critical**: Current 0.32 trades/week is dysfunctional
2. **Total frequency maintained**: 26.2 → 27.1 (minimal change)
3. **Low risk**: Just rebalancing LONG/SHORT, not changing total exposure
4. **Better diversification**: 65/35 balance captures both directions
5. **LONG improvement**: +10.5% precision for LONG is valuable

**Implementation**:
```python
# phase4_dynamic_testnet_trading.py
LONG_ENTRY_THRESHOLD = 0.80  # was 0.70
SHORT_ENTRY_THRESHOLD = 0.50  # was 0.70
```

**Expected 4-week results**:
- ~108 total trades (71 LONG, 37 SHORT)
- Improved LONG win rate (~28% precision)
- SHORT actually trading (was barely trading)
- Better market regime coverage

---

## Validation Plan

### Week 1-2: Initial Validation
- Monitor trade frequency (target: 20-30/week)
- Track LONG vs SHORT distribution (target: 60-70% LONG)
- Measure precision for each direction

### Week 3-4: Performance Validation
- Calculate realized win rates
- Compare to backtest expectations
- Assess if SHORT trades are profitable

### Adjustment Criteria

**If after 2 weeks**:
- LONG trades < 15/week → Lower to 0.75
- SHORT trades < 5/week → Lower to 0.45
- SHORT trades > 15/week → Raise to 0.55
- Combined WR < 60% → Revert to 0.70/0.70

---

## Files Generated

**Results CSV**: `results/threshold_optimization_phase4_results.csv`
- All 14 threshold tests (7 LONG × 7 SHORT)
- Full metrics for each threshold

**Script**: `scripts/production/optimize_threshold_phase4.py`
- Reusable for future threshold optimization
- Works with Phase 4 production models

---

## Conclusion

**Status**: ✅ **EXECUTION COMPLETE**

**Key Finding**: SHORT model severely undertrades at current threshold

**Action**: Implement Option A (LONG 0.80, SHORT 0.50)

**Expected Outcome**: Balanced LONG/SHORT system with maintained trade frequency

**Next**: Update bot configuration and validate over 4 weeks

---

**End of Analysis**
**End of Execution**
**Results Obtained**
**Implementation Ready**
