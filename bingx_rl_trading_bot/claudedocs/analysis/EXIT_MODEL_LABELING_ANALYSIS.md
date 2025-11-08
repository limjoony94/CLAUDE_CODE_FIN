# Exit Model Labeling Strategy - Analysis & Design

**Date**: 2025-10-14
**Context**: Current system uses Entry Dual Model (LONG/SHORT), Exit is Rule-based
**Goal**: Add Exit Dual Model (LONG exit / SHORT exit) with ML-learned timing

---

## 🎯 Current System Analysis

### Entry System (Already Dual Model)
```python
# Line 268-294: Load LONG + SHORT models
self.long_model = load("xgboost_v4_phase4_advanced_lookahead3_thresh0.pkl")
self.short_model = load("xgboost_short_model_lookahead3_thresh0.3.pkl")

# Line 869-886: Independent predictions
prob_long = self.long_model.predict_proba(features)[0][1]
prob_short = self.short_model.predict_proba(features)[0][1]

if prob_long >= 0.7:
    enter("LONG")
elif prob_short >= 0.7:
    enter("SHORT")
```

**Status**: ✅ Entry specialization implemented

### Exit System (Current - Rule-based)
```python
# Line 1066-1075: Fixed rules
if pnl_pct <= -0.01:          # -1%
    exit("Stop Loss")
elif pnl_pct >= 0.03:         # +3%
    exit("Take Profit")
elif hours_held >= 4:
    exit("Max Holding")
```

**Problems**:
- ❌ Fixed thresholds don't adapt to market conditions
- ❌ 91.6% of trades hit Max Hold (exiting too late?)
- ❌ No consideration of momentum, volatility, or pattern breaks
- ❌ Same rules for LONG and SHORT (asymmetric markets)

**Status**: ⚠️ Needs ML-learned exit timing

---

## 💡 Exit Labeling Challenge

### Core Question
**"What makes a good exit point?"**

Unlike entry (simple: P&L > 0), exit labeling is complex:
- Multiple candidate exit points per trade
- Trade-off: Early exit (safe) vs Late exit (maximize profit)
- Hindsight bias: We know future prices, but model won't

### Example Trade Analysis
```
Trade: LONG Entry @ $100

Candle 1: $101 (+1%) - Exit here? → Miss $3 gain
Candle 2: $102 (+2%) - Exit here? → Miss $2 gain
Candle 3: $103 (+3%) - Exit here? → ✅ TP hit, optimal!
Candle 4: $102 (+2%) - Too late, lost $1 profit
Candle 5: $100 (0%)  - Too late, broke even
```

**Question**: Which candles should be labeled as "good exit"?
- Only Candle 3 (peak)?
- Candles 2-3 (near peak)?
- Any candle with positive P&L?

---

## 🧪 Labeling Strategy Options

### Option 1: Peak Exit Labeling
**Definition**: Label only the best exit point (highest P&L) as positive

**Implementation**:
```python
for each historical_trade:
    all_pnl_during_trade = []
    for candle in trade.candles:
        pnl = calculate_pnl(candle.price, entry_price)
        all_pnl_during_trade.append((candle_idx, pnl))

    # Find peak
    best_exit_idx, best_pnl = max(all_pnl_during_trade, key=lambda x: x[1])

    # Label only peak as 1
    labels[best_exit_idx] = 1
    labels[all_other_indices] = 0
```

**Pros**:
- ✅ Clear objective: Exit at peak
- ✅ Model learns to recognize profit maximization points

**Cons**:
- ❌ Unrealistic: Can't know peak in real-time
- ❌ Very imbalanced: Only 1 positive per trade (99% negative)
- ❌ Ignores "good enough" exits near peak

**Expected Performance**: Poor (too idealistic)

---

### Option 2: Near-Peak Window Labeling
**Definition**: Label exits within X% of peak as positive

**Implementation**:
```python
for each historical_trade:
    # Find peak
    peak_pnl = max([calculate_pnl(c, entry) for c in trade.candles])

    # Label exits within 80% of peak as good
    threshold = peak_pnl * 0.8

    for candle in trade.candles:
        pnl = calculate_pnl(candle.price, entry_price)
        if pnl >= threshold:
            labels[candle_idx] = 1  # Good exit
        else:
            labels[candle_idx] = 0  # Bad exit
```

**Pros**:
- ✅ More realistic: Multiple "good" exits
- ✅ Better balance: 10-30% positive labels
- ✅ Learns "near-optimal" pattern recognition

**Cons**:
- ⚠️ Arbitrary threshold (80%? 90%?)
- ⚠️ Still uses hindsight (knows peak)

**Expected Performance**: Good (realistic + balanced)

---

### Option 3: Rule-Beating Labeling
**Definition**: Label exits that beat current rule-based system

**Implementation**:
```python
for each historical_trade:
    # Simulate what rule-based exit would do
    rule_exit_pnl = simulate_rule_exit(trade, SL=-0.01, TP=0.03, max_hold=4h)

    # For each candle, compare ML exit vs Rule exit
    for candle in trade.candles:
        ml_exit_pnl = calculate_pnl(candle.price, entry_price)

        if ml_exit_pnl > rule_exit_pnl:
            labels[candle_idx] = 1  # ML exit beats rules
        else:
            labels[candle_idx] = 0  # Rules are better
```

**Pros**:
- ✅ Practical: Beats existing system
- ✅ No hindsight bias: Compares to actual rule outcome
- ✅ Incremental improvement approach

**Cons**:
- ⚠️ Limited by baseline: Can't exceed rule performance by much
- ⚠️ May inherit rule biases

**Expected Performance**: Moderate (safe improvement)

---

### Option 4: Future P&L Prediction Labeling
**Definition**: Label exits where immediate exit beats holding

**Implementation**:
```python
for each candle in trade:
    pnl_if_exit_now = calculate_pnl(current_price, entry_price)

    # Look ahead N candles (e.g., 12 candles = 1 hour)
    pnl_if_hold_1h = calculate_pnl(price_in_1h, entry_price)

    if pnl_if_exit_now > pnl_if_hold_1h:
        labels[candle_idx] = 1  # Exit now better than holding
    else:
        labels[candle_idx] = 0  # Holding is better
```

**Pros**:
- ✅ Real-time applicable: Compares current vs near future
- ✅ No peak hunting: Just "is now a good time?"
- ✅ Balanced labels: ~50/50 split expected

**Cons**:
- ⚠️ Lookahead parameter arbitrary (1h? 2h?)
- ⚠️ May exit too early if lookahead too short

**Expected Performance**: Good (realistic comparison)

---

### Option 5: Hybrid Multi-Criteria Labeling
**Definition**: Combine multiple criteria for exit quality

**Implementation**:
```python
def score_exit_quality(candle, trade):
    score = 0

    # Criterion 1: Positive P&L (weight 0.3)
    if candle.pnl > 0:
        score += 0.3 * (candle.pnl / trade.peak_pnl)

    # Criterion 2: Near peak (weight 0.3)
    if candle.pnl >= trade.peak_pnl * 0.8:
        score += 0.3

    # Criterion 3: Momentum reversal (weight 0.2)
    if detect_momentum_shift(candle):
        score += 0.2

    # Criterion 4: Pattern break (weight 0.2)
    if detect_pattern_break(candle):
        score += 0.2

    return score

# Label based on score
for candle in trade.candles:
    score = score_exit_quality(candle, trade)
    labels[candle_idx] = 1 if score >= 0.6 else 0
```

**Pros**:
- ✅ Sophisticated: Multiple factors
- ✅ Captures complex exit logic
- ✅ Adaptive to different conditions

**Cons**:
- ❌ Complex: Many parameters to tune
- ❌ Computational cost: Many indicators
- ❌ Overfitting risk: Too many criteria

**Expected Performance**: Variable (high potential, high risk)

---

## 📊 Labeling Strategy Comparison

| Strategy | Realism | Balance | Complexity | Hindsight Bias | Expected F1 |
|----------|---------|---------|------------|----------------|-------------|
| **Peak Exit** | Low | Very Poor | Low | High | 0.1-0.2 |
| **Near-Peak** | Medium | Good | Medium | Medium | 0.4-0.5 |
| **Rule-Beating** | High | Good | Low | Low | 0.3-0.4 |
| **Future P&L** | High | Excellent | Medium | Low | 0.45-0.55 |
| **Hybrid** | High | Good | High | Medium | 0.5-0.6 |

---

## 🎯 Recommended Strategy

### Primary: **Option 4 (Future P&L Prediction)** + **Option 2 (Near-Peak Window)**

**Hybrid Approach**:
```python
def label_exit_point(candle, trade, entry_price):
    """
    Label exit as positive if BOTH conditions met:
    1. Within 80% of peak (near-optimal)
    2. Beats holding for next 1 hour

    This ensures exits are both:
    - Near-optimal (not too early)
    - Better than holding (not too late)
    """
    # Condition 1: Near peak (hindsight filter)
    peak_pnl = trade.peak_pnl
    current_pnl = calculate_pnl(candle.price, entry_price)
    near_peak = current_pnl >= (peak_pnl * 0.8)

    # Condition 2: Beats holding (real-time applicable)
    pnl_if_hold_1h = calculate_pnl(candle.future_price_1h, entry_price)
    beats_holding = current_pnl > pnl_if_hold_1h

    # Both conditions required
    return 1 if (near_peak and beats_holding) else 0
```

**Rationale**:
- ✅ Combines best of both: Near-optimal + Real-time applicable
- ✅ Balanced labels: ~30-40% positive
- ✅ Realistic: Model learns "good enough + timely" exits
- ✅ Moderate hindsight: Filters out obviously bad exits

**Expected Performance**: F1 0.45-0.55 (+400-500% vs baseline labeling)

---

## 🔬 Implementation Plan

### Phase 1: Data Preparation (2 hours)

**Task 1.1**: Generate historical trade dataset
```python
# Simulate all possible trades on historical data
for candle in historical_data:
    # LONG trades
    if should_enter_long(candle):
        trade = simulate_long_trade(
            entry=candle,
            exit_rules=(SL=-0.01, TP=0.03, max_hold=4h)
        )
        long_trades.append(trade)

    # SHORT trades
    if should_enter_short(candle):
        trade = simulate_short_trade(
            entry=candle,
            exit_rules=(SL=-0.01, TP=0.03, max_hold=4h)
        )
        short_trades.append(trade)
```

**Task 1.2**: Generate exit labels
```python
def generate_exit_labels(trades, direction):
    """Generate exit labels for LONG or SHORT trades"""
    exit_samples = []

    for trade in trades:
        # For each candle during the trade
        for candle_idx in range(len(trade.candles)):
            candle = trade.candles[candle_idx]

            # Calculate features (base + position)
            features = calculate_exit_features(
                candle=candle,
                trade=trade,
                base_features=37,
                position_features=8
            )

            # Calculate label
            label = label_exit_point(
                candle=candle,
                trade=trade,
                entry_price=trade.entry_price
            )

            exit_samples.append({
                'features': features,
                'label': label,
                'direction': direction
            })

    return exit_samples

long_exit_samples = generate_exit_labels(long_trades, "LONG")
short_exit_samples = generate_exit_labels(short_trades, "SHORT")
```

### Phase 2: Model Training (2 hours)

**Task 2.1**: Train LONG Exit Model
```python
from xgboost import XGBClassifier

# Features: Base (37) + Position (8) = 45 total
exit_features = base_features + [
    'time_held',           # Hours since entry
    'current_pnl_pct',     # Current P&L %
    'pnl_peak',            # Best P&L so far
    'pnl_trough',          # Worst P&L so far
    'volatility_since_entry',
    'volume_change',
    'momentum_shift',
    'pattern_health'       # Support/resistance status
]

long_exit_model = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    scale_pos_weight=(neg_count / pos_count)  # Handle imbalance
)

long_exit_model.fit(
    X=long_exit_samples['features'],
    y=long_exit_samples['labels']
)

# Save
save(long_exit_model, "xgboost_v4_long_exit.pkl")
```

**Task 2.2**: Train SHORT Exit Model
```python
# Same process for SHORT trades
short_exit_model = XGBClassifier(...)
short_exit_model.fit(
    X=short_exit_samples['features'],
    y=short_exit_samples['labels']
)

save(short_exit_model, "xgboost_v4_short_exit.pkl")
```

### Phase 3: Backtest Integration (2 hours)

**Task 3.1**: Create hybrid exit logic
```python
def check_exit(position, current_candle, models):
    """
    Hybrid exit logic: ML + Safety Rules

    Priority:
    1. Hard stops (safety)
    2. ML exit model (learned timing)
    3. Max holding (backup)
    """
    # Safety: Hard stops always respected
    if position.pnl_pct <= -0.015:  # -1.5% hard stop
        return True, "Hard Stop Loss"

    if position.pnl_pct >= 0.035:   # +3.5% hard take profit
        return True, "Hard Take Profit"

    # ML Exit Decision
    exit_features = calculate_exit_features(current_candle, position)

    if position.direction == "LONG":
        exit_prob = models['long_exit'].predict_proba(exit_features)[0][1]
    else:  # SHORT
        exit_prob = models['short_exit'].predict_proba(exit_features)[0][1]

    if exit_prob >= 0.75:  # 75% threshold
        return True, "ML Exit"

    # Backup: Max holding
    if position.hours_held >= 4:
        return True, "Max Holding"

    return False, None
```

**Task 3.2**: Compare performance
```
Test 1: Entry Dual + Exit Rules (current)
Test 2: Entry Dual + Exit Dual (ML)

Metrics:
- Returns per 2 days
- Win Rate
- Average hold time
- Exit reason distribution
```

### Phase 4: Analysis & Decision (1 hour)

**Success Criteria**:
1. ✅ Returns improvement: >10%
2. ✅ Win rate improvement: >5%
3. ✅ Average hold time: 2-3 hours (vs 4h max)
4. ✅ Exit distribution: <50% Max Hold (vs current 91.6%)

**Decision Logic**:
```python
if ml_exit_improvement > 10% and max_hold_pct < 50%:
    deploy("Entry Dual + Exit Dual")
elif ml_exit_improvement > 5%:
    optimize_exit_threshold()
else:
    keep("Entry Dual + Exit Rules")
```

---

## 📈 Expected Benefits

### From Exit Model Specialization

**Hypothesis**: ML-learned exits beat fixed rules

**Expected Improvements**:
1. **Hold Time Reduction**: 4h → 2-3h average
   - Faster capital rotation
   - More trades per day
   - +15-20% trade frequency

2. **Exit Quality**: 91.6% Max Hold → 30-40% Max Hold
   - Exit near optimal points
   - Fewer "too late" exits
   - +10-15% avg P&L per trade

3. **Direction Awareness**: LONG vs SHORT exits differ
   - LONG exit: Detect exhaustion, resistance
   - SHORT exit: Detect reversal, support
   - +5-10% from specialization

**Total Expected**: +25-40% performance improvement

---

## ⚠️ Potential Risks

### Risk 1: Exit Too Early
**Scenario**: Model exits at +1% when trade could reach +3%

**Mitigation**:
- Use Near-Peak labeling (80% of peak)
- Set minimum P&L threshold (don't exit <0.5%)
- Monitor "missed profit" metric

### Risk 2: Exit Too Late
**Scenario**: Model waits for better opportunity, trade reverses

**Mitigation**:
- Hard stops always respected (-1.5%, +3.5%)
- Max Hold backup (4h)
- Monitor "profit erosion" metric

### Risk 3: Data Imbalance
**Scenario**: Too many negative exit examples (don't exit)

**Mitigation**:
- Use `scale_pos_weight` in XGBoost
- Ensure 30-40% positive labels
- Stratified train/test split

### Risk 4: Overfitting
**Scenario**: Model memorizes specific trades

**Mitigation**:
- Cross-validation (5-fold)
- Regularization (max_depth=6, learning_rate=0.05)
- Test on out-of-sample data

---

## 🎯 Summary

**Current System**:
- Entry: LONG model + SHORT model ✅ (Dual, specialized)
- Exit: Fixed rules (SL/TP/Max Hold) ⚠️ (Not learned)

**Proposed System**:
- Entry: LONG model + SHORT model ✅ (Keep)
- Exit: LONG exit + SHORT exit ⚠️ (Add ML-learned)

**Labeling Strategy**:
- **Recommended**: Near-Peak + Future P&L hybrid
- **Expected F1**: 0.45-0.55 (+400-500% vs naive)
- **Expected Balance**: 30-40% positive labels

**Implementation**:
1. Generate historical trade dataset (LONG + SHORT)
2. Label exits: Near-peak (80%) AND beats-holding (1h)
3. Train 2 exit models (LONG exit, SHORT exit)
4. Integrate with hybrid logic (ML + safety stops)
5. Backtest and compare vs current system

**Expected Outcome**:
- +25-40% performance improvement from ML exits
- +10-15% from direction-specific exits
- **Total**: +35-55% improvement over current system

**Estimated Time**: 6-8 hours total

---

**Status**: 📋 Analysis Complete, Ready for Implementation
**Next**: Generate exit labels and train models

🚀 **Let's build ML-learned exit models!**
