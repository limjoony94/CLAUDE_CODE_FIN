# Lag Features Complete Analysis - Lessons Learned

## Executive Summary

**Question:** "거래 판단 분석에 사용되는 캔들은 몇개인거야?" (How many candles are used for trading decisions?)

**Discovery:** XGBoost only uses 1 candle → Attempted solution with lag features → Failed even with proper hyperparameter tuning

**Final Decision:** ✅ **Keep Phase 4 Base Model (37 features, 7.68% returns)**

---

## Experiment Timeline

### Phase 1: Initial Lag Features (Untuned)
**Hypothesis:** Add lag features to enable temporal pattern learning

**Implementation:**
- 37 base features → 185 temporal features (lags + momentum)
- **Critical Error:** Used same hyperparameters designed for 37 features!
- Parameters: max_depth=6, colsample_bytree=0.8, lr=0.05

**Results:**
- F1: 0.089 → 0.046 (-48%)
- Returns: 7.68% → 2.38% (-69%)
- Win rate: 69.1% → 75.3% (higher but fewer trades)

**Conclusion:** ❌ Lag features made performance much worse

---

### Phase 2: User Correction & Hyperparameter Tuning
**User Insight:** "지표들이 추가가 되었는데 파라미터 조정을 하지 않았다?" ✅

**Critical Finding:** I failed to adjust hyperparameters for 185 features!

**Hyperparameter Tuning Results:**
- Tested 30 parameter combinations via RandomizedSearchCV
- **Best F1: 0.075** (up from 0.046 untuned)

**Optimal Parameters for 185 Features:**
```python
# Critical changes for 185 features:
colsample_bytree: 0.8 → 0.5    # Sample 50% features (prevent overfitting)
min_child_weight: 1 → 5        # Stronger regularization
gamma: 0 → 0.3                 # Tree pruning
learning_rate: 0.05 → 0.03     # Slower, more stable learning
colsample_bylevel: 0.7         # NEW: per-level sampling
```

**Key Insight:** Hyperparameters matter! Tuning for 185 features required:
1. **Reduced feature sampling** (0.5 instead of 0.8)
2. **Increased regularization** (min_child_weight=5, gamma=0.3)
3. **Slower learning** (lr=0.03)

---

### Phase 3: Tuned Lag Features Backtest

**Results:**
- F1: 0.046 → 0.075 (+63% improvement!)
- Returns: 2.38% → 3.56% (+50% improvement!)
- Win rate: 75.3% → 71.5%

**But still worse than Base:**
- Tuned Lag: 3.56% per 5 days
- Base: 7.68% per 5 days
- **-54% worse than base**

---

## Final Comparison

| Model | F1 Score | Returns (vs B&H) | Win Rate | Trades | Sharpe | Max DD |
|-------|----------|------------------|----------|--------|--------|--------|
| **Base (37 features)** | **0.089** | **+7.68%** | **69.1%** | ~15 | 11.88 | 0.90% |
| Lag Untuned (185 features, wrong params) | 0.046 | +2.38% | 75.3% | 8.7 | 20.14 | 1.16% |
| **Lag TUNED (185 features, optimized)** | **0.075** | **+3.56%** | 71.5% | 12.3 | 9.00 | 1.47% |

**Statistical Significance:**
- Tuned Lag vs B&H: t=4.18, **p=0.003** (significant)
- But still 54% worse than Base model

---

## Root Cause Analysis

### Why Lag Features Still Failed (Even With Tuning)

#### 1. **Overfitting to Training Patterns**
```
Positive samples: 642
Features: 185
Ratio: 3.5 samples per feature
```

**Problem:** Not enough data even with regularization
- Model learns training-specific temporal correlations
- These patterns don't generalize to new data

#### 2. **Correlated Features**
Lag features highly correlated with base features:
- `RSI`, `RSI_lag1`, `RSI_lag2` → ~0.95 correlation
- Creates redundant, noisy signal

#### 3. **Temporal Patterns May Not Be Useful**
XGBoost learned temporal patterns BUT they don't help trading:
- 22/30 top features are lag/momentum features
- Model IS using them, but they're not predictive

#### 4. **Conservative Predictions**
Lower precision/recall = more conservative:
- Fewer trades (12.3 vs 15)
- Higher win rate (71.5% vs 69.1%)
- But less profitable overall

---

## Lessons Learned

### ✅ What Worked

**1. User's Critical Thinking**
- Correctly identified missing hyperparameter tuning
- "지표들이 추가가 되었는데 파라미터 조정을 하지 않았다?"
- This was a critical oversight that needed fixing

**2. Systematic Hyperparameter Tuning**
- 30 combinations tested
- F1 improved 63% (0.046 → 0.075)
- Returns improved 50% (2.38% → 3.56%)
- Proved that hyperparameters matter for high-dimensional data

**3. Evidence-Based Decision Making**
- Tested hypothesis thoroughly
- Tuning helped but not enough
- Accepted negative result → kept best model

### ❌ What Didn't Work

**1. Lag Features for XGBoost**
- Adding temporal context through lag features
- Even with proper hyperparameters: 3.56% vs 7.68% (base)
- Fundamental issue: XGBoost not ideal for temporal patterns

**2. Initial Assumption**
- Assumed XGBoost could learn temporal patterns with lag features
- Reality: Creates overfitting and noise, not useful signals

---

## Alternative Approaches

### Option 1: Feature Selection (Not All Lag Features)
```python
# Select top 20 lag features by importance
# Total: 37 base + 20 selected lags = 57 features
```
**Pros:** Less overfitting, cleaner signal
**Cons:** Still may not capture useful patterns

### Option 2: LSTM/RNN for Temporal Learning
```python
# Feed sequences directly
input_shape = (10 candles, 37 features)
# LSTM learns temporal dependencies
```
**Pros:** Designed for sequences
**Cons:** More complex, needs more data

### Option 3: Ensemble (XGBoost + LSTM)
```python
# Combine both approaches
xgb_pred + lstm_pred → final_prediction
```
**Pros:** Best of both worlds
**Cons:** Complexity, maintenance

### Option 4: Better Base Features (Recommended)
```python
# Focus on feature engineering instead of lag features
# - Market microstructure
# - Order flow
# - Volume profile
# - Higher timeframe alignment (15m, 1h)
```
**Pros:** More robust, proven approach
**Cons:** Requires domain expertise

---

## Key Insights

### 1. **Hyperparameters Matter for High-Dimensional Data**

When features increase from 37 → 185:
- **Feature sampling must decrease** (colsample_bytree: 0.8 → 0.5)
- **Regularization must increase** (min_child_weight, gamma)
- **Learning rate may need adjustment** (lr: 0.05 → 0.03)

**Rule of Thumb:** >100 features with <1000 positive samples → aggressive regularization

### 2. **More Features ≠ Better Performance**

185 temporal features performed worse than 37 base features:
- Signal dilution across correlated features
- Overfitting to training-specific patterns
- Computational cost increase without benefit

**Rule of Thumb:** 10-50 samples per feature for robust learning

### 3. **Tool Selection Matters**

- **XGBoost strength:** Feature-based, cross-sectional patterns
- **XGBoost weakness:** Temporal sequences, time dependencies
- **Better for temporal:** LSTM, RNN, Transformer

**Right Tool:** Use LSTM for temporal, XGBoost for features

### 4. **Critical Thinking Process**

1. Identify problem (1 candle limitation)
2. Propose solution (lag features)
3. Implement properly (with hyperparameter tuning!)
4. Validate rigorously (backtest)
5. **Accept negative results** (keep base model)

**Key:** Evidence > assumptions. Don't force solutions that don't work.

---

## Recommendations

### Immediate (Next Steps)

**✅ Keep Base Model Running**
- Phase 4 Advanced: 37 features
- Returns: 7.68% per 5 days
- Proven performance, stable

**✅ Document Learnings**
- Hyperparameter tuning process
- Why lag features failed
- What to try next

### Short-Term (1-2 weeks)

**Option 1: Better Feature Engineering**
- Add market microstructure features
- Volume profile analysis
- Higher timeframe confirmation (15m, 1h signals)
- Expected: +10-20% improvement over base

**Option 2: Feature Selection Approach**
- Select top 20 lag features by importance
- Test 57-feature model (37 base + 20 lags)
- May reduce overfitting while keeping useful patterns

### Medium-Term (1-2 months)

**Option 1: LSTM Exploration**
- Research LSTM for crypto trading
- Compare LSTM vs XGBoost on same features
- Consider ensemble if both show promise

**Option 2: Multi-Timeframe Model**
- 5m signals (current)
- 15m trend confirmation
- 1h regime classification
- Ensemble predictions

### Long-Term (3+ months)

**Advanced ML Pipeline:**
1. Feature selection (reduce from 37 to 20-25)
2. LSTM for temporal patterns
3. XGBoost for feature patterns
4. Ensemble with confidence weighting
5. Continuous retraining pipeline

---

## Conclusion

**Summary:**
- ✅ User correctly identified missing hyperparameter tuning
- ✅ Tuning improved lag features by 63% (F1) and 50% (returns)
- ❌ But lag features still underperform base by 54%
- ✅ Keep base model, try different approach

**Critical Thinking Validated:**
1. Test hypotheses systematically
2. Accept negative results
3. Learn from failures
4. Iterate to better solutions

**비판적 사고 결론:**
- 파라미터 튜닝은 중요하다 (user was right!)
- 하지만 lag features는 근본적으로 효과가 없다
- Base model (7.68%)을 유지한다
- 다른 접근 방법을 시도한다 (feature engineering, LSTM, etc.)

**Final Verdict:** Evidence-based decision making > assumptions. Base model is best.
