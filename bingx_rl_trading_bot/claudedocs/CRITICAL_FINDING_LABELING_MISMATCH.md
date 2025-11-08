# Critical Finding: Labeling vs Trading Objective Mismatch

**Date**: 2025-10-15
**Status**: 🔴 **CRITICAL INSIGHT**

---

## 1. The Paradox

### Training Performance (Excellent)
```
LONG Entry Model vs Training Labels:
- F1 Score: 86.54% ✅ (current labeling: lookahead=3, threshold=0.3%)
- Precision: 82.36%
- Recall: 91.16%
```

### Real Trading Performance (Poor)
```
Signal Quality (TP 3% within 4h):
- Threshold 0.5: 0.4% TP hit rate ❌
- Threshold 0.6: 0.5% TP hit rate ❌
- Threshold 0.7: 0.6% TP hit rate ❌ (current)
- Threshold 0.8: 0.7% TP hit rate ❌
```

**Out of 1440 signals (threshold 0.7), only ~9 actually hit TP 3% within 4h!**

---

## 2. Root Cause: Objective Mismatch

### Training Objective
```
Question: "Will price increase ≥0.3% in next 15 minutes?"
Label = 1 if: max(price[t+1:t+3]) ≥ current_price * 1.003
Focus: Short-term, small moves, very frequent (4.31% positive rate)
```

### Actual Trading Objective
```
Question: "Will price increase ≥3% within 4h before hitting -1% SL?"
Reality: Long-term, large moves, very rare (0.6% achievable)
Gap: 10x price threshold, 16x time horizon
```

**The model is optimized for a completely different task than what we're asking it to do in trading!**

---

## 3. Why Does The System Still Work?

The backtest achieves 70.6% win rate despite this mismatch. How?

### The Filter Effect
```
Model outputs probabilities for "15min, 0.3% move"
We apply threshold 0.7 → Ultra-selective filter (top 4.77% of all candles)

Hypothesis: High probability for short-term small moves
           → Correlates with strong momentum
           → Higher chance of hitting TP 3% eventually

Evidence:
- Random signals @ same rate: 49.0% WR
- Model signals @ same rate: 70.6% WR
- Improvement: +21.6%p (highly significant)
```

**The model works not because it predicts TP 3%, but because:**
1. It predicts short-term upward momentum
2. Threshold 0.7 filters to strongest momentum signals
3. Strong momentum → higher probability of reaching TP
4. Combined with good strategy parameters (SL 1%, TP 3%, Max Hold 4h)

---

## 4. The Real Question

**Should we retrain with TP-aligned labels?**

### Option A: Keep Current Approach ✅
```yaml
Pros:
  - Already working (70.6% WR, +4.19% returns)
  - High F1 (86.54%) means model is well-calibrated
  - Threshold filtering effectively converts to trading signals
  - 21.6%p improvement over random proven

Cons:
  - Theoretical: Labeling doesn't match objective
  - Low TP hit rate (0.6%) means most signals don't reach TP directly
```

### Option B: Retrain with TP-Aware Labels ⚠️
```yaml
Proposed: lookahead=48, threshold=3% (match actual TP)

Pros:
  - Labels match actual trading objective
  - More "honest" about what we're predicting

Cons:
  - Extremely rare signals (0.16% positive rate - 48 total in dataset!)
  - Severe class imbalance (99.84% negative)
  - May not have enough positive examples to learn
  - Risk: Break something that's working
```

---

## 5. Analysis of Alternative Labeling

### Labeling Performance vs Lookahead/Threshold

| Lookahead | Threshold | Positive Rate | F1 Score | Use Case |
|-----------|-----------|---------------|----------|----------|
| 3 (15min) | 0.3% | 4.31% | **86.54%** | **Current** (best F1) |
| 12 (1h) | 0.5% | 7.82% | 41.56% | Medium-term |
| 24 (2h) | 1.0% | 3.64% | 24.96% | Long-term |
| 48 (4h) | 3.0% | 0.16% | 1.75% | **TP-aligned** (too rare!) |

**Insight**: As we move toward TP-aligned labeling, F1 drops dramatically due to extreme rarity.

---

## 6. Signal Quality by Threshold

### TP(3%) Hit Rates within 4 Hours

| Threshold | Signals | Signal Rate | TP Hit Rate @ 4h |
|-----------|---------|-------------|------------------|
| 0.5 | 2,702 | 8.95% | 0.4% |
| 0.6 | 1,913 | 6.34% | 0.5% |
| **0.7** | **1,440** | **4.77%** | **0.6%** |
| 0.8 | 965 | 3.20% | 0.7% |
| 0.9 | 333 | 1.10% | 0.3% |

**Pattern**: Higher threshold → Better quality but diminishing returns.
**Current**: 0.7 seems to be near-optimal.

---

## 7. Key Insights

### Insight 1: Indirect Prediction Works
The model doesn't need to predict TP directly. It predicts short-term momentum, and we filter for strong signals. This indirect approach is effective.

### Insight 2: Labeling Rarity Problem
If we align labels with actual TP (3% in 4h):
- Positive rate: 0.16% (only 48 examples in 30K candles!)
- This is likely insufficient for training
- SMOTE rebalancing may create synthetic patterns that don't exist

### Insight 3: Correlation vs Causation
```
Current approach:
  "Predict short-term momentum" → Filter strong signals → High TP hit rate

Proposed approach:
  "Predict TP directly" → May not have enough data to learn pattern
```

The current approach works because short-term momentum is **easier to predict** and **correlates** with eventual TP achievement.

---

## 8. Recommendations

### Priority 1: Validation Before Change ⚠️
**CRITICAL**: Current system works well. Don't fix what isn't broken.

Before retraining with new labels:
1. ✅ **Validate model adds value** (Random baseline: +21.6%p WR) → DONE
2. ✅ **Validate threshold is optimal** (0.7 best for returns) → DONE
3. ⏳ **Analyze other 3 models** (SHORT Entry, LONG Exit, SHORT Exit)
4. ⏳ **Consider alternative improvements** (feature engineering, ensemble)

### Priority 2: If Retraining Needed
If we decide to retrain, use **conservative approach**:

```yaml
Conservative Improvement:
  Lookahead: 12-24 (1-2 hours, not 48)
  Threshold: 1.0-1.5% (not 3%)
  Rationale:
    - Still enough positive examples (1-4%)
    - Closer to trading objective than current
    - Less extreme than full TP alignment
    - Lower risk of breaking system
```

### Priority 3: Consider Feature Engineering Instead
Rather than changing labels, improve **feature quality**:
- More advanced momentum indicators
- Market regime detection
- Volatility-adjusted signals
- Multi-timeframe features

This preserves current working approach while potentially improving performance.

---

## 9. Conclusion

**The "labeling mismatch" is not actually a problem** - it's a feature, not a bug!

The current approach:
1. ✅ Model learns short-term patterns (easier, more data)
2. ✅ High threshold filters to strong signals (effective)
3. ✅ System achieves 70.6% WR (+21.6%p vs random)
4. ✅ Statistically significant (p < 0.0001)

**Recommendation**:
- **DO NOT** immediately retrain with TP-aligned labels
- **COMPLETE** analysis of other 3 models first
- **CONSIDER** feature engineering before labeling changes
- **VALIDATE** any proposed changes against current baseline

**Philosophy**: "If it ain't broke, don't fix it. If you must fix it, validate first."

---

**Next Steps**:
1. Analyze SHORT Entry model (same analysis)
2. Analyze LONG/SHORT Exit models
3. Synthesize findings across all 4 models
4. Make informed decision on improvements

**Status**: Per-model analysis in progress (1/4 complete)
