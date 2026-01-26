# Model Redesign Plan - 2025-10-21

## Executive Summary

**Current Status**: New models (2025-10-21) fail on Test Set
- Best performance: -3.77% return, 57.6% win rate
- All 16 threshold combinations result in losses
- SHORT model shows severe calibration issues (+37% probability shift)

**Decision**: Full model redesign required
**Timeline**: 2-3 days
**Goal**: Achieve >=60% win rate, positive returns on out-of-sample data

---

## Problem Analysis

### 1. SHORT Model Calibration Failure ⚠️ CRITICAL

**Symptoms**:
```yaml
Training Set:
  Mean Probability: 0.2845
  >= 0.70 threshold: 14.92%

Test Set:
  Mean Probability: 0.3908 (+37% ⬆️)
  >= 0.70 threshold: 23.34%

Impact:
  - Expected SHORT: 15% of trades
  - Actual SHORT: 75% of trades (5x more)
  - Severely degraded trade quality
```

**Root Causes**:
1. **Feature Distribution Shift**: SHORT features (ATR, volatility, trend strength) behave differently in Test period
2. **Overfitting**: Model memorized training patterns, doesn't generalize
3. **Market Regime Change**: Test period (Sep 27 - Oct 18) has different characteristics than training

---

### 2. LONG Model Performance

**Symptoms**:
```yaml
Training vs Test Shift: Minimal (0.2425 → 0.2237, -8%)
Signal Frequency: Stable (14.33% → 12.04%)
High Confidence: Degraded (10.74% → 5.83%)

Status: ✅ Relatively stable, but loses confidence on test set
```

**Issues**:
- Lower precision in Test Set
- Fewer high-confidence opportunities
- Still contributes to overall losses when combined with SHORT

---

### 3. Win Rate Below Target

**Current**: 57.6% (best case)
**Target**: >= 60%
**Gap**: -2.4%

**Contributing Factors**:
- Low model precision (19-21%)
- Poor risk/reward ratio (losses > wins on average)
- Exit strategy may be suboptimal

---

### 4. Trade Frequency Issues

**Expected**: ~3.5 trades/day (LONG 85%, SHORT 15%)
**Actual**: 4.2-6.1 trades/day (LONG 25%, SHORT 75%)

**Problems**:
- Too many SHORT signals
- Opportunity gating not working effectively
- Capital locked in poor-quality SHORT trades

---

## Redesign Strategy

### Phase 1: Diagnostic Deep Dive (Day 1 Morning)

#### 1.1 Market Regime Analysis
**Goal**: Understand Test Set characteristics

```python
Tasks:
1. Volatility regime classification
   - High/Medium/Low volatility periods
   - Compare Train vs Test distributions

2. Trend regime classification
   - Strong Bull/Bull/Neutral/Bear/Strong Bear
   - Identify regime shifts

3. Volume profile analysis
   - Compare Train vs Test volume patterns

4. Price action analysis
   - Support/resistance behavior
   - Breakout/breakdown patterns
```

**Output**: Market regime features for modeling

---

#### 1.2 Feature Distribution Analysis
**Goal**: Identify features with distribution shift

```python
Tasks:
1. Statistical tests for each feature
   - Kolmogorov-Smirnov test (Train vs Test)
   - Identify features with p < 0.05

2. Feature stability ranking
   - Rank by distribution similarity
   - Flag unstable features for removal/redesign

3. SHORT-specific feature analysis
   - Focus on top 15 SHORT features
   - Understand why they shift on Test Set
```

**Output**: Feature stability report, removal candidates

---

#### 1.3 Label Quality Analysis
**Goal**: Validate Trade-Outcome labeling

```python
Tasks:
1. Label distribution by regime
   - Are positive labels uniform across regimes?
   - Or concentrated in specific periods?

2. Label quality metrics
   - Precision/Recall by regime
   - Identify problematic regimes

3. Alternative labeling exploration
   - Simple profit threshold
   - Risk-adjusted returns
   - Sharpe ratio-based
```

**Output**: Labeling strategy recommendation

---

### Phase 2: Feature Engineering (Day 1 Afternoon)

#### 2.1 Market Regime Features
**Implementation**:

```python
New Features:
1. volatility_regime (categorical: low/medium/high)
   - Based on 20-period rolling ATR percentile

2. trend_regime (categorical: strong_bear/-1, bear/-0.5, neutral/0, bull/0.5, strong_bull/1)
   - Based on multi-timeframe MA alignment

3. volume_regime (categorical: low/medium/high)
   - Based on volume percentile

4. market_phase (categorical: accumulation/markup/distribution/markdown)
   - Wyckoff market cycle identification

5. regime_stability (continuous: 0-1)
   - How long in current regime
```

---

#### 2.2 Cross-Timeframe Features
**Implementation**:

```python
New Features:
1. Higher timeframe alignment
   - 15min, 1h, 4h trend direction
   - Multi-timeframe confluence score

2. Price action context
   - Distance from key levels (%, multiple timeframes)
   - Recent swing high/low penetration

3. Momentum consistency
   - RSI alignment across timeframes
   - MACD alignment score
```

---

#### 2.3 Risk-Adjusted Features
**Implementation**:

```python
New Features:
1. sharpe_ratio_5 (5-period rolling Sharpe)
2. sharpe_ratio_20 (20-period rolling Sharpe)
3. sortino_ratio (downside deviation adjusted)
4. max_drawdown_pct (rolling max drawdown)
5. recovery_factor (gain/max drawdown)
6. profit_factor (gross_profit/gross_loss)
```

---

#### 2.4 Order Flow Features (if data available)
**Implementation**:

```python
New Features (if BingX provides):
1. buy_sell_ratio (aggressive buy vs sell volume)
2. large_order_flow (volume from large orders)
3. bid_ask_imbalance
4. cumulative_delta
```

---

### Phase 3: Model Architecture Improvements (Day 2 Morning)

#### 3.1 Calibration Strategy
**Method**: Isotonic Regression + Platt Scaling

```python
Implementation:
1. Train base XGBoost model
2. Hold out calibration set (10% of training)
3. Fit Isotonic Regression on calibration set
4. Apply calibration to all predictions

Validation:
- Expected calibration error (ECE)
- Reliability diagrams
- Brier score
```

---

#### 3.2 Ensemble Approach
**Method**: Multi-model voting

```python
Models:
1. XGBoost (current)
2. LightGBM (faster, different regularization)
3. Random Forest (lower variance)

Combination:
- Soft voting (average probabilities)
- Weighted by calibration score
- Threshold per model independently
```

---

#### 3.3 Separate Models by Regime
**Method**: Regime-specific specialists

```python
Approach:
1. Train separate models for each volatility regime
   - Low volatility LONG/SHORT models
   - High volatility LONG/SHORT models

2. Router model
   - Classifies current regime
   - Routes to appropriate specialist

Benefits:
- Models specialized for specific conditions
- Better generalization
- Reduced distribution shift impact
```

---

### Phase 4: Training Strategy Redesign (Day 2 Afternoon)

#### 4.1 Cross-Validation Approach
**Method**: Time-series walk-forward validation

```python
Strategy:
1. Split data into 5 folds (chronological)
2. Train on fold 1, validate on fold 2
3. Train on folds 1-2, validate on fold 3
4. Train on folds 1-3, validate on fold 4
5. Train on folds 1-4, validate on fold 5
6. Average metrics across folds

Benefits:
- More robust performance estimate
- Detects temporal instability
- Avoids overfitting to single test set
```

---

#### 4.2 Hyperparameter Optimization
**Method**: Optuna with walk-forward CV

```python
Parameters to Optimize:
XGBoost:
- n_estimators: [100, 200, 300, 400, 500]
- max_depth: [3, 4, 5, 6, 7]
- learning_rate: [0.01, 0.03, 0.05, 0.1]
- subsample: [0.6, 0.7, 0.8, 0.9]
- colsample_bytree: [0.6, 0.7, 0.8, 0.9]
- min_child_weight: [1, 3, 5, 7]
- gamma: [0, 0.1, 0.2, 0.5]

Objective: Maximize F1-score on validation folds
Trials: 100 per model
```

---

#### 4.3 Label Engineering
**Method**: Multi-criteria with regime awareness

```python
New Labeling Strategy:
1. Regime-adjusted thresholds
   - Higher profit requirements in low volatility
   - Lower thresholds in high volatility

2. Risk-adjusted criteria
   - Must meet Sharpe ratio threshold
   - Maximum adverse excursion limit

3. 3-of-4 criteria (more strict):
   a. Profitable (>= 2% leveraged)
   b. Good risk/reward (MAE/MFE ratio)
   c. ML exit (not emergency)
   d. Sharpe ratio >= 1.0
```

---

### Phase 5: Implementation & Testing (Day 3)

#### 5.1 Model Retraining
**Process**:

```yaml
Morning:
1. Retrain LONG entry model
   - New features + calibration
   - Walk-forward CV
   - Hyperparameter optimization

2. Retrain SHORT entry model
   - New features + calibration
   - Walk-forward CV
   - Hyperparameter optimization

Afternoon:
3. Retrain Exit models (if needed)
4. Integrate regime classification
5. Build ensemble system
```

---

#### 5.2 Validation & Testing
**Process**:

```yaml
Test Set Validation:
1. Predictions on held-out test set
2. Calibration assessment
3. Win rate, return metrics
4. Trade frequency analysis
5. LONG/SHORT distribution

Acceptance Criteria:
- Win Rate >= 60%
- Positive return
- Trade frequency: 3-5/day
- LONG/SHORT ratio: 70-90% / 10-30%
- Sharpe ratio > 1.0
```

---

#### 5.3 72H Recent Backtest
**Process**:

```yaml
Final Validation:
1. Run on most recent 72 hours
2. Compare vs current production
3. Risk assessment
4. Deployment decision
```

---

## Risk Management

### Technical Risks

**Risk 1: Overfitting to validation folds**
- Mitigation: Multiple CV folds, ensemble approach
- Monitoring: Performance on final test set

**Risk 2: Feature engineering increases complexity**
- Mitigation: Feature selection, importance ranking
- Monitoring: Model interpretability, feature stability

**Risk 3: Longer training time**
- Mitigation: Parallel processing, efficient code
- Expectation: 6-8 hours total training

---

### Business Risks

**Risk 1: Model still fails on test set**
- Mitigation: Incremental validation, early stopping if not improving
- Backup: Return to 2025-10-18 models

**Risk 2: Overly conservative models**
- Mitigation: Balance win rate with trade frequency
- Monitoring: Opportunity cost analysis

**Risk 3: Time investment (2-3 days)**
- Mitigation: Clear milestones, go/no-go decisions
- Backup: Can abort and return to previous system

---

## Success Criteria

### Minimum Viable Performance
```yaml
Test Set (out-of-sample):
- Win Rate: >= 60%
- Return: > 0% (positive)
- Trade Frequency: 3-5/day
- LONG/SHORT: 70-90% / 10-30%
- Max Drawdown: < 15%
```

### Stretch Goals
```yaml
Test Set:
- Win Rate: >= 65%
- Return: >= +5%
- Sharpe Ratio: >= 1.5
- Trade Frequency: 4-6/day
- Calibration: ECE < 0.05
```

---

## Timeline & Milestones

### Day 1: Diagnostic & Feature Engineering
```yaml
Morning (4 hours):
  [x] Problem analysis (complete)
  [ ] Market regime analysis
  [ ] Feature distribution analysis
  [ ] Label quality analysis

Afternoon (4 hours):
  [ ] Implement regime features
  [ ] Implement cross-timeframe features
  [ ] Implement risk-adjusted features
  [ ] Feature validation

Deliverable: New feature set ready for training
Go/No-Go: If features show instability, reconsider approach
```

---

### Day 2: Model Architecture & Training
```yaml
Morning (4 hours):
  [ ] Implement calibration pipeline
  [ ] Setup ensemble framework
  [ ] Setup walk-forward CV
  [ ] Hyperparameter optimization

Afternoon (4 hours):
  [ ] Train LONG entry model
  [ ] Train SHORT entry model
  [ ] Train Exit models (if needed)
  [ ] Model validation

Deliverable: Trained models with improved architecture
Go/No-Go: If validation shows <55% win rate, abort
```

---

### Day 3: Integration & Testing
```yaml
Morning (4 hours):
  [ ] Integrate regime classification
  [ ] Build ensemble system
  [ ] Test set validation
  [ ] Performance analysis

Afternoon (4 hours):
  [ ] 72H recent backtest
  [ ] Compare vs current production
  [ ] Deployment preparation
  [ ] Documentation

Deliverable: Production-ready models or decision to abort
Final Decision: Deploy vs Return to 2025-10-18 models
```

---

## Abort Criteria

**Stop redesign if**:
1. Day 1 analysis shows fundamental data issues (e.g., data quality problems)
2. Day 2 validation shows win rate < 55% (worse than current)
3. Day 3 test shows negative returns and win rate < 60%

**Fallback Plan**:
- Return to 2025-10-18 models (currently in production)
- Continue operating with proven system
- Plan alternative strategy (e.g., different ML approach, rule-based system)

---

## Resource Requirements

### Computational
```yaml
Training Time: 6-8 hours total
  - Feature engineering: 1 hour
  - Hyperparameter optimization: 3-4 hours
  - Model training: 2-3 hours

Memory: ~4-8 GB RAM
Storage: ~5 GB for data and models
```

### Data
```yaml
Historical Data: 109 days (July 1 - Oct 18)
  - Training: 87.5 days (80%)
  - Validation: Cross-validation folds
  - Test: 21.9 days (20%)

Additional: Real-time data for regime classification
```

---

## Next Steps

### Immediate Actions (Today)
```yaml
Priority 1: Market Regime Analysis
  - Classify volatility regimes
  - Identify regime shifts
  - Compare train vs test

Priority 2: Feature Distribution Analysis
  - Statistical tests for all features
  - Identify problematic features
  - SHORT-specific investigation

Priority 3: Label Quality Assessment
  - Analyze label distribution
  - Test alternative labeling methods
```

### Tomorrow (Day 2)
```yaml
Priority 1: Feature Engineering
  - Implement regime features
  - Implement multi-timeframe features
  - Implement risk-adjusted features

Priority 2: Model Training
  - Setup calibration pipeline
  - Hyperparameter optimization
  - Train new models
```

### Day 3
```yaml
Priority 1: Validation
  - Test set validation
  - 72H backtest

Priority 2: Decision
  - Deploy or abort
  - Documentation
```

---

## Conclusion

This redesign addresses the root causes of model failure:
1. **SHORT model calibration** → Regime-aware features + calibration
2. **Feature distribution shift** → Stability analysis + regime classification
3. **Low win rate** → Better features + improved training
4. **Poor generalization** → Walk-forward CV + ensemble

**Expected Outcome**: Models that achieve >=60% win rate and positive returns on out-of-sample data.

**Risk**: 2-3 days investment with no guarantee of success.

**Mitigation**: Clear milestones, abort criteria, fallback to proven system.

---

**Approval Required**: Proceed with Phase 1 (Market Regime & Feature Analysis)?
