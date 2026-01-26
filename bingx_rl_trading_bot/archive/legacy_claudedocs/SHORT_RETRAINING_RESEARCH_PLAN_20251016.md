# SHORT Model Retraining Research Plan

**Date**: 2025-10-16 (Session 2)
**Objective**: Develop SHORT Entry model with >50% win rate
**Current Performance**: 20% win rate (backtest), <1% win rate (signal quality analysis)
**Target Performance**: >50% win rate, >0.5% signal rate

---

## 📋 Research Overview

### Problem Statement
Current SHORT model fundamentally fails to predict profitable SHORT trades due to:
1. Training-production objective mismatch
2. Insufficient bearish market data
3. LONG-optimized features used for SHORT

### Research Approaches (Parallel Execution)

**Approach 1: TP/SL-Aligned Labeling** (HIGHEST PRIORITY)
- Align training labels with actual trade outcomes
- Predict TP hit probability vs SL hit probability
- **Expected Impact**: HIGH (addresses root cause)

**Approach 2: Market Regime Analysis**
- Analyze current data coverage (Bull/Bear/Sideways)
- Identify data gaps for SHORT training
- **Expected Impact**: MEDIUM (improves training examples)

**Approach 3: SHORT-Specific Features**
- Engineer bearish pattern indicators
- Add contrarian/resistance features
- **Expected Impact**: MEDIUM-HIGH (better signal quality)

---

## 🔬 Approach 1: TP/SL-Aligned Labeling

### Current vs Proposed Methodology

**Current (FAILED)**:
```python
def create_short_labels(df, lookahead=3, threshold=0.003):
    """
    Label 1: Price decreases by 0.3% within 15 minutes
    Label 0: Otherwise
    """
    # Problem: This is NOT what we actually trade!
    # - We don't exit at 0.3% profit
    # - We don't exit at 15 minutes
    # - We use TP/SL system (3% TP, 1% SL, 4h max hold)
```

**Proposed (TP/SL-Aligned)**:
```python
def create_short_labels_tp_sl_aligned(df, tp_pct=0.03, sl_pct=0.01, max_hold_candles=48):
    """
    Label 1: Trade hits TP (-3%) before SL (+1%) within max_hold (4h)
    Label 0: Trade hits SL first, or max hold expires without TP

    This EXACTLY matches backtest evaluation criteria!
    """
    labels = []

    for i in range(len(df)):
        if i >= len(df) - max_hold_candles:
            labels.append(0)
            continue

        entry_price = df['close'].iloc[i]

        # Simulate actual SHORT trade
        tp_price = entry_price * (1 - tp_pct)  # -3% (price drops)
        sl_price = entry_price * (1 + sl_pct)  # +1% (price rises)

        tp_hit = False
        sl_hit = False

        for j in range(1, max_hold_candles + 1):
            if i + j >= len(df):
                break

            candle = df.iloc[i + j]

            # Check if TP hit (price dropped to target)
            if candle['low'] <= tp_price:
                tp_hit = True
                break

            # Check if SL hit (price rose to stop loss)
            if candle['high'] >= sl_price:
                sl_hit = True
                break

        # Label 1: TP hit before SL
        labels.append(1 if tp_hit and not sl_hit else 0)

    return np.array(labels)
```

### Why This Works

**Alignment**:
- Training objective = Backtest evaluation
- Model learns: "Will this trade be profitable?" (not "Will price drop 0.3%?")
- Direct optimization of actual trading outcome

**Expected Results**:
```
Current labeling:
  Positive rate: 4.00% (too many false positives)
  Model learns: noise patterns for 0.3% drops
  Real win rate: 0.6-0.8%

TP/SL-aligned labeling:
  Positive rate: ~1-2% (only true profitable trades)
  Model learns: actual profitable SHORT setups
  Expected win rate: 40-60% (realistic for SHORT)
```

### Implementation Plan

**Step 1: Create New Training Script**
- File: `scripts/experiments/train_short_tp_sl_aligned.py`
- Implement TP/SL simulation labeling
- Use same features as current model (44 features)
- Train XGBoost with new labels

**Step 2: Validation**
- Compare label distribution (old vs new)
- Analyze positive example characteristics
- Verify label quality with manual inspection

**Step 3: Training**
- Train on full dataset (30,467 candles)
- Use TimeSeriesSplit cross-validation
- Compare training metrics to current model

**Step 4: Backtesting**
- Run same backtest as current model
- Compare win rates, signal rates, profitability
- Success criteria: >50% win rate

---

## 🔬 Approach 2: Market Regime Analysis

### Data Coverage Assessment

**Current Dataset**:
- Period: Aug 7 - Oct 15, 2025 (~68 days)
- Total candles: 30,517 (5-minute)
- Regime distribution: Unknown (need analysis)

**Analysis Tasks**:

**Task 1: Regime Distribution**
```python
# Analyze regime coverage in training data
regimes = df['regime_trend'].value_counts()
print(f"Bull: {bull_pct}%")
print(f"Bear: {bear_pct}%")
print(f"Sideways: {sideways_pct}%")

# Expected issue:
# - BTC upward bias → mostly Bull/Sideways
# - Insufficient Bear data → poor SHORT learning
```

**Task 2: SHORT Opportunity Distribution**
```python
# Using TP/SL-aligned labels, analyze by regime
short_opps_by_regime = df.groupby('regime_trend')['short_label'].sum()

# Expected:
# Bull: Few SHORT opportunities
# Bear: Most SHORT opportunities
# Sideways: Moderate SHORT opportunities

# If Bear data < 20%, need more data collection
```

**Task 3: Data Collection Strategy**
```python
# If insufficient Bear data identified:
# 1. Download historical data from known Bear periods
#    - BTC crash periods: Nov 2021, May 2022, Nov 2022
# 2. Combine with current data
# 3. Retrain with expanded dataset

# Target: 30%+ Bear regime data for balanced learning
```

### Expected Outcome

**If Bear data sufficient**:
- Current data is adequate, proceed with Approach 1

**If Bear data insufficient**:
- Download 6-12 months additional historical data
- Focus on Bear market periods
- Retrain with expanded dataset
- Expected improvement: +10-20% win rate

---

## 🔬 Approach 3: SHORT-Specific Features

### Current Feature Analysis

**Existing 44 Features**:
```
Momentum: rsi, macd, stochrsi, willr, cci, cmo, roc, mfi, tsi, kst
Trend: ema_21, ema_50, adx, di_plus, di_minus, aroon_up, aroon_down
Volatility: atr, bb_width, true_range
Volume: volume_ratio, obv, cmf
```

**Problem**: These are LONG-biased features
- Momentum indicators: Detect upward momentum
- Trend followers: Identify uptrends
- No bearish-specific patterns

### Proposed SHORT Features

**Category 1: Resistance & Distribution**
```python
# Feature: Distance from major resistance
def calculate_resistance_proximity(df, lookback=100):
    """
    SHORT opportunity higher when price near resistance
    """
    resistance = df['high'].rolling(window=lookback).max()
    distance_to_resistance = (resistance - df['close']) / df['close']
    return distance_to_resistance

# Feature: Distribution phase detection
def detect_distribution(df, volume_threshold=1.5):
    """
    High volume + sideways price = distribution (bearish)
    """
    volume_spike = df['volume'] > df['volume'].rolling(20).mean() * volume_threshold
    price_flat = df['close'].rolling(10).std() / df['close'] < 0.01
    return (volume_spike & price_flat).astype(int)
```

**Category 2: Bearish Divergences**
```python
# Feature: RSI bearish divergence strength
def rsi_bearish_divergence_strength(df):
    """
    Price makes higher high, RSI makes lower high → bearish
    Quantify divergence strength (not just binary)
    """
    price_hh = df['close'] > df['close'].shift(14)
    rsi_lh = df['rsi'] < df['rsi'].shift(14)
    divergence = price_hh & rsi_lh

    # Strength = magnitude of divergence
    divergence_strength = (df['close'].pct_change(14) -
                          df['rsi'].pct_change(14) / 100)
    return divergence_strength * divergence

# Similar for MACD, volume divergences
```

**Category 3: Contrarian Indicators**
```python
# Feature: Funding rate extreme (crypto-specific)
def funding_rate_extreme(df):
    """
    High positive funding = too many longs = SHORT opportunity
    (Requires funding rate data from BingX API)
    """
    # TODO: Add funding rate data collection
    pass

# Feature: Fear & Greed extreme
def fear_greed_extreme(df):
    """
    Extreme greed (>80) = potential reversal = SHORT opportunity
    (Requires external API: Alternative.me Crypto Fear & Greed Index)
    """
    # TODO: Add API integration
    pass
```

**Category 4: Bearish Patterns**
```python
# Feature: Lower highs / lower lows
def downtrend_strength(df, window=20):
    """
    Quantify downtrend momentum
    """
    highs = df['high'].rolling(window).max()
    lows = df['low'].rolling(window).min()

    downtrend = (highs.diff() < 0) & (lows.diff() < 0)
    downtrend_strength = abs(df['close'].pct_change(window)) * downtrend
    return downtrend_strength

# Feature: Bearish candlestick patterns
def bearish_pattern_score(df):
    """
    Aggregate bearish patterns: shooting star, evening star, dark cloud
    """
    patterns = []

    # Shooting star
    upper_shadow = df['high'] - df[['open', 'close']].max(axis=1)
    body = abs(df['close'] - df['open'])
    shooting_star = (upper_shadow > 2 * body) & (df['close'] < df['open'])
    patterns.append(shooting_star)

    # Evening star (3-candle pattern)
    # Dark cloud cover
    # etc.

    return sum(patterns).astype(int)
```

### Implementation Plan

**Step 1: Feature Engineering Script**
- File: `src/features/short_specific_features.py`
- Implement 10-15 new SHORT features
- Test on historical data

**Step 2: Feature Selection**
- Combine with existing 44 features (total ~60 features)
- Use feature importance analysis
- Select top 50 features for training

**Step 3: Training with New Features**
- Train XGBoost with expanded feature set
- Compare to baseline (44 features)
- Measure improvement

**Expected Impact**: +15-25% win rate improvement

---

## 📊 Validation Framework

### Comprehensive Testing Protocol

**Level 1: Label Quality Check**
```python
def validate_labels(df, labels):
    """
    Verify labels match actual trade outcomes
    """
    # Sample 100 random positive labels
    # Manually verify: Does trade hit TP before SL?
    # Acceptance: >95% label accuracy
    pass
```

**Level 2: Cross-Validation**
```python
def time_series_cross_validation(X, y, n_splits=5):
    """
    Use TimeSeriesSplit to prevent future data leakage
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)

    fold_results = []
    for train_idx, val_idx in tscv.split(X):
        # Train on train_idx
        # Evaluate on val_idx
        # Store metrics
        pass

    # Report: Mean ± Std across folds
    # Success: All folds >40% win rate
```

**Level 3: Signal Quality Analysis**
```python
def analyze_signal_quality(model, df, thresholds=[0.5, 0.6, 0.7, 0.8, 0.9]):
    """
    Same analysis as before, but with new model
    """
    for threshold in thresholds:
        signals = model.predict_proba(X)[:, 1] >= threshold
        # Calculate actual win rate @ 4h
        # Measure signal rate
        # Compute avg profit
        pass

    # Success: At least one threshold shows >50% win rate
```

**Level 4: Full Backtest**
```python
def comprehensive_backtest(model):
    """
    Run same backtest as current model
    Compare side-by-side
    """
    # Use: backtest_dual_model_mainnet.py
    # Replace SHORT model with new model
    # Run 21 windows

    # Success criteria:
    # - SHORT win rate >50%
    # - SHORT signal rate >0.5%
    # - Overall strategy improvement
```

**Level 5: Regime-Specific Performance**
```python
def analyze_by_regime(results):
    """
    Ensure model works across all regimes
    """
    for regime in ['Bull', 'Bear', 'Sideways']:
        regime_trades = results[results['regime'] == regime]
        win_rate = regime_trades['win'].mean()

        # Acceptable:
        # Bear: >60% (best conditions for SHORT)
        # Sideways: >50%
        # Bull: >40% (hardest for SHORT)
```

### Success Criteria

**Minimum Acceptable Performance (MVP)**:
```yaml
Overall:
  win_rate: ">50%"
  signal_rate: ">0.5%"
  sharpe_ratio: ">0" (positive)

By Regime:
  Bear: ">60%"
  Sideways: ">50%"
  Bull: ">40%"

Validation:
  cross_val_consistency: "All folds within ±10% of mean"
  label_accuracy: ">95%"
  signal_quality: "At least one threshold >50% win rate"
```

**Target Performance (IDEAL)**:
```yaml
Overall:
  win_rate: ">60%"
  signal_rate: ">1.0%"
  sharpe_ratio: ">5.0"

By Regime:
  Bear: ">70%"
  Sideways: ">60%"
  Bull: ">50%"
```

---

## 🗓️ Implementation Timeline

### Week 1: Approach 1 (TP/SL-Aligned Labeling)

**Day 1-2**:
- ✅ Create research plan (DONE)
- ⏳ Implement TP/SL labeling function
- ⏳ Generate new labels on full dataset
- ⏳ Validate label quality (sample 100 trades)

**Day 3-4**:
- Train XGBoost with new labels
- Cross-validation (5 folds)
- Analyze training metrics vs current model

**Day 5-7**:
- Signal quality analysis
- Full backtest
- Performance comparison
- **Decision**: Deploy if >50% win rate, else continue to Approach 2/3

### Week 2: Approach 2 (Market Regime Analysis)

**Day 1-2**:
- Analyze current data regime distribution
- Identify Bear data coverage
- **If insufficient**: Download historical Bear market data

**Day 3-4**:
- Retrain with expanded dataset (if needed)
- Compare to Week 1 results
- Measure improvement

**Day 5-7**:
- Combine best of Approach 1 + 2
- Full validation
- **Decision**: Deploy if >60% win rate, else continue to Approach 3

### Week 3: Approach 3 (SHORT-Specific Features)

**Day 1-3**:
- Implement SHORT-specific features (10-15 new features)
- Feature importance analysis
- Feature selection

**Day 4-7**:
- Train with expanded feature set
- Combine with best labeling from Approach 1
- Combine with best data from Approach 2
- **Final validation and deployment decision**

### Week 4: Production Deployment & Monitoring

**Day 1-2**:
- Deploy best model to production
- Update configuration
- Restart bot

**Day 3-7**:
- Monitor first 50 SHORT trades
- Validate live performance matches backtest
- Adjust thresholds if needed

---

## 📁 File Structure

### New Files to Create

**Training Scripts**:
```
scripts/experiments/
├── train_short_tp_sl_aligned.py (Week 1)
├── analyze_regime_distribution.py (Week 2)
├── train_short_expanded_data.py (Week 2)
└── train_short_full_features.py (Week 3)
```

**Feature Engineering**:
```
src/features/
└── short_specific_features.py (Week 3)
```

**Validation**:
```
scripts/validation/
├── validate_short_labels.py
├── cross_validate_short_model.py
└── compare_short_models.py
```

**Documentation**:
```
claudedocs/
├── SHORT_RETRAINING_RESEARCH_PLAN_20251016.md (this file)
├── SHORT_APPROACH1_RESULTS.md (Week 1)
├── SHORT_APPROACH2_RESULTS.md (Week 2)
├── SHORT_APPROACH3_RESULTS.md (Week 3)
└── SHORT_FINAL_MODEL_REPORT.md (Week 4)
```

---

## 🎯 Expected Outcomes

### Scenario Analysis

**Best Case** (All approaches succeed):
- TP/SL alignment: +30% win rate → 50%
- Expanded data: +10% win rate → 60%
- New features: +10% win rate → 70%
- **Final: 70% SHORT win rate (matches LONG!)**

**Expected Case** (Approach 1 succeeds):
- TP/SL alignment: +30% win rate → 50%
- **Final: 50% SHORT win rate (profitable, deployable)**

**Worst Case** (All approaches fail):
- Win rate remains <30%
- **Conclusion: SHORT trading fundamentally unprofitable for this market**
- **Action: Disable SHORT permanently, LONG-only strategy**

---

## 📊 Progress Tracking

### Approach 1: TP/SL-Aligned Labeling
- [ ] Create labeling function
- [ ] Generate labels on full dataset
- [ ] Validate label quality
- [ ] Train model
- [ ] Cross-validation
- [ ] Signal quality analysis
- [ ] Full backtest
- [ ] Performance comparison

### Approach 2: Market Regime Analysis
- [ ] Analyze regime distribution
- [ ] Identify data gaps
- [ ] Download additional data (if needed)
- [ ] Retrain with expanded data
- [ ] Measure improvement

### Approach 3: SHORT-Specific Features
- [ ] Design features (10-15)
- [ ] Implement feature engineering
- [ ] Feature importance analysis
- [ ] Train with new features
- [ ] Measure improvement

### Deployment
- [ ] Final model selection
- [ ] Production deployment
- [ ] Monitor live performance
- [ ] Validate vs backtest

---

## 💡 Key Principles

1. **Systematic Approach**: Test each approach independently, then combine
2. **Evidence-Based**: Every decision backed by data and metrics
3. **Validation-First**: Comprehensive validation before deployment
4. **Iterative Improvement**: Each week builds on previous results
5. **Clear Success Criteria**: >50% win rate minimum for deployment

---

**Next Action**: Start Week 1 - Implement TP/SL-Aligned Labeling

**Status**: 📋 **RESEARCH PLAN COMPLETE** → 🔬 **STARTING IMPLEMENTATION**

---

**Created By**: Claude (SuperClaude Framework - Research Mode)
**Date**: 2025-10-16 (Session 2)
**Estimated Completion**: 3-4 weeks
**Confidence**: 🟡 MEDIUM-HIGH (Approach 1 very promising, others additive)
