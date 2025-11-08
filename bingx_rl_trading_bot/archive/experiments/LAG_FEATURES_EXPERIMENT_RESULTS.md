# Lag Features Experiment - Critical Analysis

## Executive Summary

**Hypothesis:** Adding lag features (t-1, t-2) would enable XGBoost to learn temporal patterns and improve trading performance.

**Result:** ❌ **FAILED** - Performance degraded by 69%

**Decision:** **Keep Phase 4 Base Model** (37 features, no lags)

---

## Experiment Overview

### Motivation

User question: "거래 판단 분석에 사용되는 캔들은 몇개인거야?" (How many candles are used for trading decisions?)

**Discovery:** XGBoost only uses **1 candle** (current timepoint's aggregated features)
- Features calculated from up to 50 candles (support/resistance lookback)
- But model receives only current values: `[RSI_now, MACD_now, ...]`
- Cannot learn sequences like "RSI rising for 3 consecutive candles"

### Hypothesis

Adding lag features would:
- Enable temporal pattern learning (trends, momentum)
- Improve win rate: 69.1% → 75-80%
- Improve returns: 7.68% → 9-10% per 5 days

### Implementation

**Lag Feature Generation:**
- Base features: 37 (10 baseline + 27 advanced)
- Lag features: 37 × 2 lags (t-1, t-2) = 74
- Momentum features: 37 × 2 = 74 (rate of change)
- **Total: 185 temporal features**

**Training:**
- Model: XGBoost (300 trees, depth=6, lr=0.05)
- Dataset: 17,228 samples
- Positive samples: 642 (3.7%)
- Cross-validation: 5-fold time series split

---

## Results Comparison

### Training Metrics

| Metric | Phase 4 Base | Phase 4 Lag | Change |
|--------|--------------|-------------|--------|
| **F1 Score** | 0.089 | 0.046 | -48% ❌ |
| **Precision** | 0.120 | 0.058 | -52% ❌ |
| **Recall** | 0.096 | 0.038 | -60% ❌ |
| **Accuracy** | 0.938 | 0.949 | +1% ⚠️ |

**Observation:** Lag model has higher accuracy but much worse F1/precision/recall → Predicting mostly negative class

### Backtest Performance

| Metric | Phase 4 Base | Phase 4 Lag | Change |
|--------|--------------|-------------|--------|
| **Returns vs B&H** | +7.68% | +2.38% | **-69%** ❌ |
| **Win Rate** | 69.1% | 75.3% | +9% ✅ |
| **Sharpe Ratio** | 11.88 | 20.14 | +70% ⚠️ |
| **Max Drawdown** | 0.90% | 1.16% | +29% ❌ |
| **Avg Trades** | ~15 | 8.7 | -42% ❌ |
| **Statistical Sig** | p=0.0003 | p=0.024 | Both significant |

**Key Finding:** Lag model is more conservative (higher win rate, fewer trades) but **MUCH LESS PROFITABLE**

### Feature Importance Analysis

**Top 30 features in Lag Model:**
- **22/30 are lag/momentum features** (73%)
- Top feature: `num_resistance_touches_momentum_2` (18.8%)

**Conclusion:** Model IS using temporal patterns, but they're not helpful for trading!

---

## Root Cause Analysis

### Why Lag Features Failed

#### 1. **Feature Dilution (Too Many Features)**

```
Positive samples: 642
Features: 185
Ratio: 3.5 samples per feature
```

**Problem:** Not enough data to learn meaningful patterns from 185 features
- XGBoost overfits to training-specific correlations
- Signal spread across many correlated lag features → noise amplification

#### 2. **Correlated Features**

Lag features are highly correlated:
- `RSI_now`, `RSI_lag1`, `RSI_lag2` → ~0.95+ correlation
- `MACD_now`, `MACD_lag1`, `MACD_lag2` → ~0.90+ correlation

**Problem:** Redundant information confuses the model
- Multiple features carry same signal → feature competition
- Model picks arbitrary splits instead of meaningful patterns

#### 3. **Training-Specific Patterns**

Temporal patterns learned during training don't generalize:
- "3 consecutive RSI increases" may be common in training period
- But doesn't repeat in validation/backtest periods
- Overfitting to training data temporal structure

#### 4. **Conservative Predictions**

Lower precision/recall (0.058/0.038) means:
- Model predicts positive class very rarely
- Higher threshold (0.7) filters out most signals
- Fewer trades (8.7 vs 15) → less profitable despite higher win rate

---

## Statistical Analysis

### Performance by Market Regime

**Threshold 0.7 Results:**

| Regime | Base Returns | Lag Returns | Difference |
|--------|--------------|-------------|------------|
| **Bull** | +13.43% | +5.84% | -57% ❌ |
| **Bear** | +3.93% | +1.96% | -50% ❌ |
| **Sideways** | +9.08% | +2.19% | -76% ❌ |

**Finding:** Lag features HURT performance in ALL market regimes

### Statistical Significance

**Base Model:**
- t-statistic: 3.78
- p-value: 0.0003
- **Highly significant** improvement over B&H ✅

**Lag Model:**
- t-statistic: 2.77
- p-value: 0.024
- Significant improvement over B&H, but **MUCH WORSE than base** ❌

---

## Alternative Approaches Considered

### ❌ Failed Approach: XGBoost with Lag Features
- Too many features → overfitting
- Correlated features → noise
- Temporal patterns don't generalize

### ✅ Better Alternatives:

#### 1. **Feature Selection (Select Best Lag Features)**
```python
# Instead of all 185 features, select top 20-30 lag features
# Based on importance or correlation analysis
top_lag_features = [
    'num_resistance_touches_momentum_2',
    'price_vs_lower_trendline_pct_lag1',
    'close_change_1_momentum_1',
    # ... top 17 more
]
# Total: 37 base + 20 selected lags = 57 features
```

**Expected:** Less overfitting, better generalization
**Risk:** May still not improve if temporal patterns not useful

#### 2. **LSTM/RNN for True Temporal Learning**
```python
# Feed last N candles as sequence
# LSTM learns temporal dependencies directly
input_shape = (lookback_candles, n_features)  # e.g., (10, 37)
```

**Expected:** Better temporal pattern learning
**Risk:** More complex, harder to train, needs more data

#### 3. **Ensemble: XGBoost + LSTM**
```python
# XGBoost: Feature-based predictions
# LSTM: Sequence-based predictions
# Combine: Weighted average or stacking
```

**Expected:** Best of both worlds
**Risk:** Complexity, harder to maintain

#### 4. **Keep Base Model, Improve Features**
```python
# Focus on better feature engineering instead of lag features
# - Market microstructure features
# - Order flow imbalance
# - Volume profile analysis
# - Higher timeframe alignment
```

**Expected:** More robust features → better performance
**Risk:** Requires domain expertise

---

## Recommendations

### Immediate Action
✅ **Keep Phase 4 Base Model (37 features, 7.68% returns)**
- Proven performance in backtest
- Currently running on testnet
- No reason to change

### Future Exploration

**Priority 1: Feature Selection**
- Try selecting top 20 lag features based on importance
- Test with 37 base + 20 selected lags = 57 features
- Validate with backtest before deployment

**Priority 2: Different Temporal Approach**
- Research LSTM/RNN for crypto trading
- Compare LSTM vs XGBoost on same dataset
- Consider ensemble if both show promise

**Priority 3: Better Features**
- Add market microstructure features
- Volume profile analysis
- Higher timeframe confirmation (15m, 1h)

### NOT Recommended
❌ **Do NOT use all 185 lag features** - Proven to hurt performance
❌ **Do NOT add more lag periods** (t-3, t-4, ...) - Will make overfitting worse

---

## Lessons Learned

### Critical Thinking Validation

✅ **Hypothesis tested systematically:**
1. Identified limitation (1 candle input)
2. Proposed solution (lag features)
3. Implemented properly (185 temporal features)
4. Validated rigorously (backtest comparison)
5. **Accepted negative result** → Kept base model

**Key Insight:** Not all "logical" improvements actually work. Evidence > assumptions.

### Feature Engineering Principles

1. **More features ≠ Better performance**
   - Need sufficient samples per feature (rule of thumb: 10-50×)
   - Lag features: 642 samples ÷ 185 features = 3.5× (too low!)

2. **Correlated features dilute signal**
   - Lag features are inherently correlated with base features
   - Model wastes splits on redundant information

3. **Temporal patterns need careful validation**
   - Patterns in training may not persist in validation
   - XGBoost not ideal for temporal learning (use LSTM/RNN instead)

### Model Selection

- **XGBoost strength:** Feature-based learning, cross-sectional patterns
- **XGBoost weakness:** Temporal patterns, sequential dependencies
- **Right tool for job:** Use LSTM/RNN for temporal, XGBoost for features

---

## Conclusion

**Lag features experiment was valuable** - We learned:
1. XGBoost with 185 lag features **reduces performance by 69%**
2. Temporal patterns from lag features **do not generalize**
3. **Base model (37 features) is the best** we have right now

**Next steps:**
- ✅ Keep base model running on testnet
- 🔍 Explore feature selection (top 20 lags only)
- 🧪 Research LSTM/RNN for true temporal learning

**비판적 사고 결과:** 가설이 틀렸다. Lag features는 성과를 악화시켰다. Base model을 유지한다.
