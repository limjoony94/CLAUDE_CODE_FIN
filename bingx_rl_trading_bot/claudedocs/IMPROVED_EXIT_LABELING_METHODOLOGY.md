# Improved EXIT Labeling Methodology

**Date**: 2025-10-16
**Purpose**: Design proper EXIT labeling to fix model inversion problem
**Goal**: Train EXIT models that predict optimal exit timing (not inverted)

---

## Problem Statement

**Current Issue**: Peak/Trough labeling creates labels AFTER optimal exit timing
- Labels peaks as "exit now" → Too late, price already moved
- Model learns: High confidence in peak = bad exit timing
- Result: Inverted model (low prob = good exit, high prob = bad exit)

**Required Solution**: Labels that identify optimal exit timing BEFORE peak/trough occurs

---

## Design Principles

### 1. Lead-Time Principle
**Exit signals should have lead time for execution**
- Don't label AT peak/trough (too late)
- Label BEFORE peak/trough (actionable timing)
- Lead time: 6-12 candles (30-60 minutes)

### 2. Quality-Over-Quantity Principle
**Better to have fewer high-quality labels than many mediocre ones**
- Strict criteria for positive labels
- Accept imbalanced dataset (10-20% positive rate)
- Focus on exits that significantly beat holding

### 3. Relative Performance Principle
**Label based on comparison to alternatives**
- Exit now vs exit later (next N candles)
- Exit now vs holding to max holding time
- Exit now vs optimal exit in future

### 4. Profit-Threshold Principle
**Only label profitable exits above minimum threshold**
- Ignore small gains (<0.5%)
- Focus on exits with meaningful profit (>0.5%)
- Reduces noise from sideways markets

---

## Proposed Labeling Methodology

### Option A: Lead-Time Peak/Trough Labeling

**Concept**: Predict peak/trough BEFORE it occurs

```python
def create_leadtime_exit_labels(df, side='LONG'):
    """
    Label = 1 if peak/trough occurs in FUTURE (next 6-12 candles)
    This gives model lead time to exit before peak
    """
    labels = np.zeros(len(df))

    for i in range(len(df) - 48):  # Need lookahead window
        # Look forward 6-12 candles
        future_window = df['close'].iloc[i+6:i+13]
        current_price = df['close'].iloc[i]

        if side == 'LONG':
            # Will there be a peak soon?
            future_max = future_window.max()
            future_peak = future_window.idxmax()

            # Check if significant peak ahead
            peak_distance = (future_max - current_price) / current_price

            if peak_distance > 0.003:  # 0.3% rise ahead
                # Also check it IS a peak (falls after)
                post_peak = df['close'].iloc[future_peak+1:future_peak+7].mean()
                if post_peak < future_max * 0.995:  # Falls 0.5% after peak
                    labels[i] = 1

        else:  # SHORT
            # Will there be a trough soon?
            future_min = future_window.min()
            future_trough = future_window.idxmin()

            # Check if significant trough ahead
            trough_distance = (current_price - future_min) / current_price

            if trough_distance > 0.003:  # 0.3% fall ahead
                # Also check it IS a trough (rises after)
                post_trough = df['close'].iloc[future_trough+1:future_trough+7].mean()
                if post_trough > future_min * 1.005:  # Rises 0.5% after trough
                    labels[i] = 1

    return labels
```

**Pros**:
- ✅ Built on familiar peak/trough concept
- ✅ Provides lead time for execution
- ✅ Focuses on significant reversals only

**Cons**:
- ⚠️ Still sensitive to peak/trough window parameters
- ⚠️ May miss optimal exits between peaks

---

### Option B: Relative Performance Labeling

**Concept**: Label exits that beat future alternatives

```python
def create_relative_performance_labels(df, trades, side='LONG'):
    """
    Label = 1 if exiting NOW beats exiting in next 24-48 candles
    Compares current profit to all future possible exits
    """
    labels = np.zeros(len(df))

    # For each trade
    for trade in trades:
        entry_idx = trade['entry_idx']
        entry_price = trade['entry_price']

        # Evaluate each candle during holding period
        max_holding = min(entry_idx + 96, len(df))  # 8 hours max

        for i in range(entry_idx + 1, max_holding - 48):
            current_price = df['close'].iloc[i]

            # Calculate profit if exit now
            if side == 'LONG':
                profit_now = (current_price - entry_price) / entry_price
            else:
                profit_now = (entry_price - current_price) / entry_price

            # Calculate best profit if exit later (next 24-48 candles)
            future_prices = df['close'].iloc[i+1:i+49]

            if side == 'LONG':
                future_profits = (future_prices - entry_price) / entry_price
                best_future_profit = future_profits.max()
            else:
                future_profits = (entry_price - future_prices) / entry_price
                best_future_profit = future_profits.max()

            # Label = 1 if exiting now is better (or close enough)
            # Allow small tolerance (within 0.1% of best future)
            if profit_now >= best_future_profit - 0.001:
                labels[i] = 1
            else:
                labels[i] = 0

    return labels
```

**Pros**:
- ✅ Directly optimizes exit timing
- ✅ Learns "when is best time to exit"
- ✅ Adapts to market conditions

**Cons**:
- ⚠️ Requires simulated trades for training
- ⚠️ More complex implementation
- ⚠️ May overfit to historical patterns

---

### Option C: Profit-Threshold with Momentum

**Concept**: Label exits with good profit AND weakening momentum

```python
def create_profit_momentum_labels(df, trades, side='LONG'):
    """
    Label = 1 if:
    1. Current profit > threshold (0.5%)
    2. Momentum weakening (reversal imminent)
    3. Exit now beats holding to max time
    """
    labels = np.zeros(len(df))

    for trade in trades:
        entry_idx = trade['entry_idx']
        entry_price = trade['entry_price']

        max_holding = min(entry_idx + 96, len(df))

        for i in range(entry_idx + 6, max_holding - 12):  # Skip first hour
            current_price = df['close'].iloc[i]

            # 1. Check profit threshold
            if side == 'LONG':
                profit = (current_price - entry_price) / entry_price
            else:
                profit = (entry_price - current_price) / entry_price

            if profit < 0.005:  # Less than 0.5% profit
                continue

            # 2. Check momentum weakening
            rsi = df['rsi'].iloc[i]
            rsi_slope = df['rsi'].iloc[i] - df['rsi'].iloc[i-6]

            momentum_weakening = False
            if side == 'LONG':
                # For LONG: High RSI turning down = weakening
                if rsi > 60 and rsi_slope < -5:
                    momentum_weakening = True
            else:
                # For SHORT: Low RSI turning up = weakening
                if rsi < 40 and rsi_slope > 5:
                    momentum_weakening = True

            if not momentum_weakening:
                continue

            # 3. Check beats holding to max
            final_price = df['close'].iloc[max_holding-1]
            if side == 'LONG':
                final_profit = (final_price - entry_price) / entry_price
            else:
                final_profit = (entry_price - final_price) / entry_price

            if profit > final_profit:
                labels[i] = 1

    return labels
```

**Pros**:
- ✅ Multiple criteria ensure quality
- ✅ Uses momentum indicators (RSI)
- ✅ Focuses on profitable exits only

**Cons**:
- ⚠️ Requires feature calculation (RSI)
- ⚠️ More complex logic
- ⚠️ May be too restrictive (very low positive rate)

---

### Option D: Combined Multi-Criteria (RECOMMENDED)

**Concept**: Combine best aspects of all approaches

```python
def create_combined_exit_labels(df, trades, side='LONG'):
    """
    Multi-criteria labeling:
    1. Lead-time peak/trough detection (timing)
    2. Profit threshold (quality)
    3. Relative performance (optimality)
    4. Momentum confirmation (confidence)

    Label = 1 if ALL criteria met (strict)
    """
    labels = np.zeros(len(df))

    for trade in trades:
        entry_idx = trade['entry_idx']
        entry_price = trade['entry_price']

        max_holding = min(entry_idx + 96, len(df))

        for i in range(entry_idx + 6, max_holding - 24):
            current_price = df['close'].iloc[i]

            # Criterion 1: Profit Threshold (>0.5%)
            if side == 'LONG':
                profit = (current_price - entry_price) / entry_price
            else:
                profit = (entry_price - current_price) / entry_price

            if profit < 0.005:
                continue

            # Criterion 2: Lead-Time Peak/Trough Detection
            future_window = df['close'].iloc[i+1:i+13]  # Next 6-12 candles

            peak_ahead = False
            if side == 'LONG':
                future_max = future_window.max()
                if (future_max - current_price) / current_price > 0.002:  # 0.2% rise
                    peak_idx = i + future_window.idxmax()
                    post_peak = df['close'].iloc[peak_idx+1:peak_idx+7].mean()
                    if post_peak < future_max * 0.997:  # Falls after
                        peak_ahead = True
            else:
                future_min = future_window.min()
                if (current_price - future_min) / current_price > 0.002:  # 0.2% fall
                    trough_idx = i + future_window.idxmin()
                    post_trough = df['close'].iloc[trough_idx+1:trough_idx+7].mean()
                    if post_trough > future_min * 1.003:  # Rises after
                        peak_ahead = True

            if not peak_ahead:
                continue

            # Criterion 3: Relative Performance (beats next 24 candles)
            future_prices = df['close'].iloc[i+1:i+25]

            if side == 'LONG':
                future_profits = (future_prices - entry_price) / entry_price
                best_future = future_profits.max()
            else:
                future_profits = (entry_price - future_prices) / entry_price
                best_future = future_profits.max()

            beats_future = profit >= best_future - 0.0005  # Within 0.05%

            if not beats_future:
                continue

            # Criterion 4: Momentum Confirmation
            rsi = df['rsi'].iloc[i]
            rsi_slope = df['rsi'].iloc[i] - df['rsi'].iloc[i-3]

            momentum_ok = False
            if side == 'LONG':
                if rsi > 55 and rsi_slope < 0:  # High RSI declining
                    momentum_ok = True
            else:
                if rsi < 45 and rsi_slope > 0:  # Low RSI rising
                    momentum_ok = True

            if not momentum_ok:
                continue

            # All criteria met!
            labels[i] = 1

    return labels
```

**Expected Characteristics**:
- **Positive Rate**: 5-15% (very selective)
- **Quality**: High (multiple criteria)
- **Lead Time**: 6-12 candles (actionable)
- **Profit Focus**: Only exits with >0.5% profit
- **Optimality**: Beats future alternatives

**Pros**:
- ✅ Very high quality labels
- ✅ Multiple criteria reduce false positives
- ✅ Lead time for execution
- ✅ Profit-focused

**Cons**:
- ⚠️ Low positive rate (may need more data)
- ⚠️ Complex implementation
- ⚠️ May be too strict (miss some good exits)

---

## Implementation Plan

### Phase 1: Data Preparation
```python
# 1. Load historical data
df = pd.read_csv("BTCUSDT_5m_max.csv")
df = calculate_features(df)

# 2. Simulate trades to get entry points
trades_long = simulate_long_trades(df)
trades_short = simulate_short_trades(df)

# 3. Create labels using combined methodology
labels_long = create_combined_exit_labels(df, trades_long, side='LONG')
labels_short = create_combined_exit_labels(df, trades_short, side='SHORT')

# 4. Check positive rate
print(f"LONG Exit positive rate: {labels_long.sum() / len(labels_long) * 100:.2f}%")
print(f"SHORT Exit positive rate: {labels_short.sum() / len(labels_short) * 100:.2f}%")
```

### Phase 2: Feature Selection
```python
# Use same features as current EXIT models, but verify relevance
features = [
    'rsi', 'macd', 'bb_width',
    'current_pnl_pct', 'pnl_from_peak',
    'holding_hours', 'volume_ratio',
    'recent_volatility', 'support_distance',
    # ... other features
]

# Add new features if needed
new_features = [
    'rsi_slope_3',  # RSI change over 3 candles
    'price_to_recent_high',  # Distance to recent high
    'momentum_divergence',  # RSI vs price divergence
]
```

### Phase 3: Model Training
```python
# Use XGBoost with same hyperparameters
model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=class_weight_ratio
)

# Train with cross-validation
cv_scores = cross_val_score(model, X, y, cv=TimeSeriesSplit(n_splits=5))

# Expected metrics with improved labeling:
# - Precision: 60-70% (higher than 55.2%)
# - Recall: 40-60% (may be lower due to stricter labels)
# - F1: 50-65%
```

### Phase 4: Validation
```python
# 1. Probability distribution check
probs = model.predict_proba(X_test)[:, 1]
print(f"Mean: {np.mean(probs):.4f}")  # Should be 0.10-0.20 (not 0.50!)
print(f"Std: {np.std(probs):.4f}")

# 2. Signal quality by probability range
for low, high in [(0.0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 1.0)]:
    range_mask = (probs >= low) & (probs < high)
    if range_mask.sum() > 0:
        range_labels = y_test[range_mask]
        precision = range_labels.sum() / len(range_labels)
        print(f"{low:.1f}-{high:.1f}: Precision = {precision:.2%}")

# Expected: Higher probability = higher precision (NOT inverted!)
```

### Phase 5: Backtest Validation
```python
# Test on same 21 windows
thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]

for thresh in thresholds:
    results = backtest_with_exit_threshold(df, models, scalers, features, thresh)
    print(f"Threshold {thresh}: Return = {results['return']:.2f}%")

# Expected:
# - Higher threshold = better results (NOT inverted)
# - Threshold 0.7-0.8 should be best
# - Should beat inverted logic (+11.60%)
```

---

## Success Criteria

### Training Metrics
- [ ] Precision: ≥60% (up from 55.2%)
- [ ] Positive rate: 10-20% (down from 50%)
- [ ] Mean probability: 0.10-0.20 (NOT 0.50)
- [ ] High prob (>0.7) has high precision (NOT inverted)

### Backtest Performance
- [ ] Return: >+11.60% (beats inverted logic)
- [ ] Win rate: >75% (beats inverted logic)
- [ ] Sharpe: >9.82 (beats inverted logic)
- [ ] Trade frequency: 60-100 per window (reasonable)

### Logic Validation
- [ ] High probability (>0.7) = good exits (NOT inverted!)
- [ ] Threshold 0.7-0.8 performs best (NOT 0.5-0.6)
- [ ] Window-by-window consistency (NOT random)

---

## Risk Mitigation

### Risk 1: Too Few Positive Labels
**Mitigation**:
- Relax criteria slightly if positive rate <5%
- Use longer historical data (more trades)
- Consider ensemble with original peak/trough labels

### Risk 2: Overfitting to Historical Data
**Mitigation**:
- Use time series cross-validation
- Test on out-of-sample period (most recent 20%)
- Monitor live performance vs backtest

### Risk 3: Worse Than Inverted Logic
**Mitigation**:
- If new models don't beat inverted logic, keep using inverted
- Iterate on labeling methodology
- Consider hybrid: New labels + inverted fallback

---

## Timeline

### Week 1: Implementation (Current Week)
- Day 1: Implement combined labeling methodology ✅ In progress
- Day 2: Generate labels, verify quality
- Day 3: Train new EXIT models
- Day 4: Backtest validation
- Day 5: Deploy if successful

### Week 2: Monitoring
- Monitor testnet performance
- Compare to inverted logic baseline
- Refine if needed

### Month 1: Production
- Deploy to production if validated
- Weekly monitoring
- Monthly retraining with new data

---

## Conclusion

**Recommended Approach**: Combined Multi-Criteria Labeling (Option D)

**Key Improvements**:
1. Lead-time peak/trough → Exit BEFORE reversal
2. Profit threshold → Focus on quality exits
3. Relative performance → Optimize timing
4. Momentum confirmation → Reduce false signals

**Expected Outcome**:
- Properly calibrated EXIT models (NOT inverted)
- Performance > +11.60% (beats inverted logic)
- Win rate > 75%
- Confidence in model predictions

**User's Vision Realized**:
> "신뢰할 수 있는 모델을 구축하는게 중요합니다"
> "Building a reliable model is more important"

This methodology builds truly reliable EXIT models that learn correct exit timing.

---

**Document Version**: 1.0
**Date**: 2025-10-16
**Status**: Design Complete, Ready for Implementation
**Next Step**: Implement combined labeling code
