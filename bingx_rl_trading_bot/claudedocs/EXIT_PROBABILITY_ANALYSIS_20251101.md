# Exit Probability Distribution Analysis
**Date**: 2025-11-01 03:49 KST
**Status**: 🔴 **CRITICAL MODEL CALIBRATION ISSUE IDENTIFIED**

---

## Executive Summary

**Root Cause Found**: Exit models output probabilities FAR BELOW the production threshold

**Impact**: Threshold 0.75 is **18.5× higher** than maximum model output (LONG), **2.7× higher** (SHORT)

**Result**: ML Exit system completely non-functional at current threshold

---

## Exit Model Probability Outputs (Recent 4 Weeks - 8,065 Candles)

### LONG Exit Model
```
Maximum:        0.0405 (4.05%)
99th Percentile: 0.0320 (3.20%)
Mean:           0.0074 (0.74%)
Median:         0.0059 (0.59%)

Range: 0.18% to 4.05%
```

### SHORT Exit Model
```
Maximum:        0.2772 (27.72%)
99th Percentile: 0.2698 (26.98%)
Mean:           0.1058 (10.58%)
Median:         0.1074 (10.74%)

Range: 3.22% to 27.72%
```

---

## Threshold Reachability Analysis

| Threshold | LONG Coverage | SHORT Coverage | Status |
|-----------|--------------|----------------|---------|
| 0.15 (15%) | 0.00% | 18.15% | ⚠️ Only SHORT works |
| 0.20 (20%) | 0.00% | 3.73% | ⚠️ Barely reachable |
| 0.25 (25%) | 0.00% | 2.54% | ⚠️ Almost unreachable |
| **0.30 (30%)** | **0.00%** | **0.00%** | **🔴 UNREACHABLE** |
| 0.75 (75%) | 0.00% | 0.00% | 🔴 **COMPLETELY UNREACHABLE** |

**Key Finding**: Threshold 0.30+ is **NEVER** reached by either model

---

## Impact on Current Production (Exit 0.75)

```yaml
Current Threshold: 0.75 (75%)

LONG Exit:
  Max Output: 0.0405 (4.05%)
  Gap: 0.75 - 0.0405 = 0.7095 (71 percentage points!)
  Result: NEVER triggers ML Exit

SHORT Exit:
  Max Output: 0.2772 (27.72%)
  Gap: 0.75 - 0.2772 = 0.4728 (47 percentage points!)
  Result: NEVER triggers ML Exit

ML Exit Rate: 0.0% (100% emergency exits via Max Hold)
```

**Current Strategy**: Not using ML Exit system. All exits via 120-candle Max Hold emergency rule (10-hour time stop).

---

## Performance vs Threshold (Recent 4 Weeks)

| Exit Threshold | Return | Win Rate | ML Exit | Status |
|---------------|--------|----------|---------|---------|
| 0.15 | +1.78% | 65.4% | 73.2% | ML works, -54% return |
| 0.20 | +2.99% | 77.3% | 52.0% | ML works, -22% return |
| 0.25 | +3.51% | 79.2% | 43.9% | ML works, -8% return |
| **0.75** | **+3.83%** | **79.2%** | **0.0%** | **🔵 BASELINE** |

**User Constraint**: "ML exit 0%일 때를 기준으로 삼고 해당 결과보다 더 잘 나와야겠죠?"
(Must beat +3.83% baseline)

**Finding**: NO threshold beats baseline while making ML Exit work

---

## Root Cause Analysis

### Why Models Output Low Probabilities

**Hypothesis 1: Training Label Imbalance**
- Exit labels may be extremely rare (e.g., < 10% positive class)
- Model learns to be conservative, rarely predicts high probability
- XGBoost outputs reflect training data distribution

**Hypothesis 2: Feature Limitations**
- Current 27 features may not capture strong exit signals
- Model lacks information to confidently predict exits
- Conservative predictions due to uncertainty

**Hypothesis 3: Training Methodology**
- Models may have been trained with different threshold in mind
- Calibration optimized for lower thresholds
- No post-training probability calibration applied

---

## Three Path Options

### Option A: Keep Exit 0.75 (CURRENT - BEST PERFORMANCE) ✅

**Pros**:
- ✅ Highest return: +3.83% per 5-day window
- ✅ Highest win rate: 79.2%
- ✅ Consistent behavior (emergency Max Hold works well)
- ✅ No code changes needed

**Cons**:
- ❌ ML Exit system completely unused (0%)
- ❌ Strategy is fixed 10-hour hold, not adaptive
- ❌ Risk if market regime changes

**Recommendation**: Keep if market remains stable

---

### Option B: Lower Threshold to 0.25 (MAKE ML EXIT WORK) ⚠️

**Pros**:
- ✅ ML Exit functional: 43.9%
- ✅ System working as designed
- ✅ Adaptive to market conditions

**Cons**:
- ❌ Return drops: +3.51% (-8% vs baseline)
- ❌ Does NOT beat user's performance standard
- ❌ User explicitly requires: "더 잘 나와야겠죠" (must be better)

**Recommendation**: Not recommended (fails user requirement)

---

### Option C: Retrain Exit Models (LONG-TERM FIX) 🔧

**Approach**:
1. Re-label Exit training data with more balanced labels
2. Add probability calibration (Platt scaling, isotonic regression)
3. Train models to output probabilities in 0.3-0.9 range
4. Validate on historical data before deployment

**Pros**:
- ✅ Can use threshold 0.75 with ML Exit working
- ✅ Fixes fundamental calibration issue
- ✅ Future-proof solution

**Cons**:
- ❌ Time-intensive (1-2 days work)
- ❌ No guarantee of improved performance
- ❌ Risk of worse results than current

**Recommendation**: Consider if Exit 0.75 performance degrades

---

## Practical Threshold Ranges (If Retraining)

Based on 99th percentile outputs:

**LONG Exit**:
- Current Max: 0.0405 (4.05%)
- Target Range: 0.60 - 0.90 (for threshold 0.75 to work)
- **Gap**: 14.8× to 22.2× improvement needed

**SHORT Exit**:
- Current Max: 0.2772 (27.72%)
- Target Range: 0.60 - 0.90 (for threshold 0.75 to work)
- **Gap**: 2.2× to 3.2× improvement needed

---

## Recommendation

**MAINTAIN EXIT 0.75 (Current Production)** ✅

**Rationale**:
1. ✅ **Highest profitability** - Meets user's performance standard
2. ✅ **User requirement** - "ML exit 0%일 때를 기준으로 삼고 해당 결과보다 더 잘 나와야겠죠"
3. ✅ **No alternatives** - No reachable threshold beats +3.83% baseline
4. ✅ **Working strategy** - Emergency Max Hold is effective in current market
5. ⚠️ **Known limitation** - ML Exit system unused, but not hurting performance

**Monitoring Plan**:
- Track if market regime changes reduce Exit 0.75 effectiveness
- Monitor if emergency Max Hold stops working well
- If performance degrades, revisit Option C (model retraining)

---

## Conclusion

**User's decision to keep Exit 0.75 was CORRECT based on evidence:**

1. Exit models physically cannot reach threshold 0.75 (max output: 4% LONG, 28% SHORT)
2. All reachable thresholds (0.15-0.25) underperform the baseline
3. Current strategy (emergency Max Hold) works well in recent market
4. User explicitly prioritized profitability over ML Exit usage

**ML Exit 0% is a model calibration issue, NOT a strategy failure.**

The system is working as well as it can given current model limitations. Retraining would be needed to make threshold 0.75 work with ML Exit, but there's no guarantee it would improve profitability.

---

**Status**: ✅ **ANALYSIS COMPLETE - EXIT 0.75 VALIDATED**
**Action**: **MAINTAIN CURRENT PRODUCTION SETTINGS**
**Next Steps**: Monitor performance, consider retraining if market changes
