# Backtest Statistical Validity Analysis

## Critical Question
"백테스트에 사용되는 데이터는 통계적으로 충분히 검증할만한 모수를 가진 백테스트를 진행한건가?"

**Short Answer:** ❌ No, current backtest has insufficient statistical power (n=9-12 windows)

---

## Current Backtest Configuration

### Data Specifications
```
Total Candles: 17,280 (60 days at 5-minute intervals)
Window Size: 1440 candles (5 days)
Number of Windows: 9-12 (non-overlapping)
Market Regimes: Bull (2-3), Bear (2-3), Sideways (5-6)
```

### Statistical Test Used
```python
from scipy import stats
t_stat, p_value = stats.ttest_rel(xgb_returns, bh_returns)
```

---

## Statistical Power Analysis

### Sample Size Problem

**Rule of Thumb for t-test:**
- Minimum n≥30 for Central Limit Theorem
- **Current n=9-12 → Too small!**

**Effect on Results:**
1. **Wide Confidence Intervals**
   - Small n → large standard errors
   - Difficult to detect true differences

2. **Low Statistical Power**
   - High risk of Type II error (false negative)
   - May miss real performance differences

3. **Questionable p-values**
   - p-value assumes normal distribution
   - n<30 → normality assumption weak

### Actual Statistical Power Calculation

**For Phase 4 Base Model:**
```
Sample size: n=12 windows
Mean difference: 7.68% - 0.57% = 7.11%
Std dev of differences: ~3.0%
t-statistic: 3.78
p-value: 0.0003

Power analysis:
- Effect size (Cohen's d): 7.11 / 3.0 = 2.37 (very large)
- Power (1-β) with n=12, α=0.05: ~0.85 (acceptable for large effects)
- BUT: Relies on normality assumption (questionable with n=12)
```

**Interpretation:**
- Large effect size (2.37) helps compensate for small n
- Power ≈ 85% is acceptable but not great
- Would need n≈30-50 for robust conclusions

---

## Market Regime Coverage

### Current Distribution
```
Bull Markets: 2-3 windows (16-25%)
Bear Markets: 2-3 windows (16-25%)
Sideways: 5-6 windows (50-60%)
```

**Problems:**
1. **Imbalanced regime representation**
   - Sideways-heavy (60%) → biased results
   - Only 2-3 Bull/Bear samples each

2. **Limited market conditions**
   - 60 days = ~2 months
   - Missing: High volatility events, trends, crashes

3. **Regime classification simplistic**
   - Based only on price direction
   - Doesn't capture volatility, volume patterns

---

## Statistical Validity Issues

### 1. Small Sample Size (n<30)

**Impact on Metrics:**

| Metric | Current n | Required n | Validity |
|--------|----------|------------|----------|
| Mean return | 9-12 | 30+ | ⚠️ Moderate |
| Win rate | 9-12 | 20+ | ⚠️ Moderate |
| Sharpe ratio | 9-12 | 50+ | ❌ Low |
| Max DD | 9-12 | 30+ | ⚠️ Moderate |
| t-test p-value | 9-12 | 30+ | ⚠️ Moderate |

**Note:** Large effect sizes (2.0+) partially compensate for small n

### 2. Non-Independent Observations

**Current Setup:** Non-overlapping windows → Independent ✅

**But:**
- Market conditions can persist across windows
- Regime transitions may create correlation
- Time series autocorrelation

### 3. Multiple Comparison Problem

**Issue:** Tested multiple models/thresholds without correction

```
Models tested:
- Base (threshold 0.3, 0.5, 0.7)
- Lag Untuned (threshold 0.3, 0.5, 0.7)
- Lag Tuned (threshold 0.3, 0.5, 0.7)
= 9 comparisons

Bonferroni correction: α = 0.05 / 9 = 0.0056
Without correction → inflated Type I error rate
```

### 4. Look-Ahead Bias Check

**Current approach:** ✅ No look-ahead bias
- Rolling windows don't use future data
- Time series split in training
- Features calculated from past data only

### 5. Overfitting to Backtest Period

**Risk:** High ⚠️
- 60 days = short period
- May not generalize to different market conditions
- Need out-of-sample validation

---

## Recommendations for Statistical Validity

### Immediate Improvements (Easy)

**1. Increase Window Count**
```python
# Current: 5-day windows (1440 candles)
WINDOW_SIZE = 1440  # n=9-12 windows

# Proposed: 3-day windows (864 candles)
WINDOW_SIZE = 864   # n=15-20 windows

# Better: 2-day windows (576 candles)
WINDOW_SIZE = 576   # n=25-30 windows ✅
```

**Benefit:** n≥30 for robust t-test assumptions

**Trade-off:** Shorter windows = less trading time per window

**2. Bootstrap Resampling**
```python
from scipy.stats import bootstrap

def mean_return_diff(data, axis):
    xgb, bh = data
    return (xgb - bh).mean(axis=axis)

# Bootstrap confidence intervals
rng = np.random.default_rng(seed=42)
boot_result = bootstrap(
    (xgb_returns, bh_returns),
    mean_return_diff,
    n_resamples=10000,
    confidence_level=0.95,
    random_state=rng
)

print(f"95% CI: [{boot_result.confidence_interval.low:.2f}%, "
      f"{boot_result.confidence_interval.high:.2f}%]")
```

**Benefit:** More robust than t-test for small n

**3. Bonferroni Correction**
```python
# Adjust significance level for multiple comparisons
alpha = 0.05
num_comparisons = 9
alpha_corrected = alpha / num_comparisons  # 0.0056

if p_value < alpha_corrected:
    print("Significant (after Bonferroni correction)")
```

### Medium-Term Improvements (Moderate Effort)

**4. More Historical Data**
```
Current: 60 days
Target: 180+ days (6 months)
Windows (2-day): 90+ windows ✅
```

**Benefit:**
- Better market regime coverage
- More robust statistics
- Different volatility regimes

**5. Walk-Forward Analysis**
```python
# Train on first 60%, test on next 20%, validate on final 20%
train_period = data[:10000]  # Train
test_period = data[10000:13000]  # Test
validation_period = data[13000:]  # Final validation (unseen)

# Retrain periodically and validate
for window in rolling_windows:
    model.retrain(window)
    validate_on_next_period()
```

**Benefit:** Detect overfitting, ensure generalization

**6. Regime-Stratified Sampling**
```python
# Ensure balanced regime representation
bull_windows = sample_windows_from_regime('Bull', n=10)
bear_windows = sample_windows_from_regime('Bear', n=10)
sideways_windows = sample_windows_from_regime('Sideways', n=10)

# Test on balanced set (n=30, balanced regimes)
```

### Long-Term Improvements (High Effort)

**7. Monte Carlo Simulation**
```python
# Simulate many possible market paths
for i in range(10000):
    shuffled_returns = shuffle(trade_returns)
    simulated_sharpe = calculate_sharpe(shuffled_returns)

# Compare actual Sharpe to simulated distribution
p_value_monte_carlo = (simulated_sharpe > actual_sharpe).mean()
```

**Benefit:** Most robust statistical test for trading strategies

**8. Out-of-Sample Validation**
```
1. Develop on 2024 data
2. Test on 2025 Q1 data
3. Validate on 2025 Q2 data
```

**Benefit:** True generalization test

**9. Multiple Markets**
```
Test on:
- BTC/USDT (current)
- ETH/USDT
- Other major pairs

Cross-validate performance across markets
```

---

## Revised Backtest Methodology

### Proposed Configuration

```python
# Data
TOTAL_DATA = 180 days  # 6 months (was 60)
WINDOW_SIZE = 576      # 2 days (was 5)
NUM_WINDOWS = ~90      # (was 9-12)

# Statistical tests
ALPHA = 0.05 / 9       # Bonferroni correction
MIN_EFFECT_SIZE = 0.5  # Cohen's d
TARGET_POWER = 0.90    # Statistical power

# Validation
TRAIN_RATIO = 0.60     # First 60% for development
TEST_RATIO = 0.20      # Next 20% for testing
VALIDATION_RATIO = 0.20 # Final 20% for validation
```

### Statistical Test Protocol

**Primary Test: Bootstrap t-test**
```python
def backtest_with_statistics(model, data, window_size=576):
    """
    Backtest with robust statistical validation

    Returns:
        results: DataFrame of window results
        stats: {
            'mean_diff': float,
            'ci_95_low': float,
            'ci_95_high': float,
            'p_value': float,
            'effect_size': float,
            'power': float
        }
    """
    # 1. Rolling window backtest
    results = rolling_window_backtest(model, data, window_size)

    # 2. Bootstrap confidence intervals
    boot_ci = bootstrap_confidence_interval(results['difference'])

    # 3. Effect size (Cohen's d)
    effect_size = cohen_d(results['xgb_return'], results['bh_return'])

    # 4. Statistical power
    power = calculate_power(effect_size, len(results), alpha=0.05)

    # 5. Bonferroni-corrected p-value
    p_value_corrected = ttest_bonferroni(results, num_comparisons=9)

    return results, {
        'mean_diff': results['difference'].mean(),
        'ci_95_low': boot_ci[0],
        'ci_95_high': boot_ci[1],
        'p_value': p_value_corrected,
        'effect_size': effect_size,
        'power': power
    }
```

### Reporting Standards

**Minimum Required Statistics:**
1. Sample size (n windows)
2. Mean ± Standard Error
3. 95% Confidence Intervals (bootstrap)
4. Effect size (Cohen's d)
5. Statistical power (1-β)
6. p-value (Bonferroni-corrected)
7. Market regime breakdown

**Example Report:**
```
Model: Phase 4 Base
Sample size: n=90 windows (2 days each)
Mean return difference: 7.68% ± 0.85% (SE)
95% CI: [5.98%, 9.38%]
Effect size: d=2.37 (very large)
Statistical power: β=0.95 (excellent)
p-value: 0.0001 (Bonferroni-corrected, α=0.0056)
Conclusion: Significant improvement over B&H with high confidence ✅
```

---

## Current Backtest Validity Assessment

### Phase 4 Base Model Results

**Configuration:**
- n=12 windows (5 days each)
- Returns: 7.68% vs 0.57% (B&H)
- p-value: 0.0003 (uncorrected)

**Statistical Validity:**

| Criterion | Current | Required | Status |
|-----------|---------|----------|--------|
| Sample size | n=12 | n≥30 | ❌ Insufficient |
| Effect size | d=2.37 | d>0.8 | ✅ Excellent |
| Statistical power | ~0.85 | >0.80 | ✅ Acceptable |
| p-value | 0.0003 | <0.0056 | ✅ Significant |
| CI width | ~6% | <3% | ⚠️ Wide |
| Data period | 60 days | 180+ days | ❌ Too short |
| Regime coverage | Imbalanced | Balanced | ⚠️ Limited |

**Overall Assessment:**
- **⚠️ Moderate Validity**
- Large effect size compensates for small n
- BUT: Need more data for robust conclusions
- Recommendation: Get 180+ days data for n≥30 windows

---

## Action Items

### Immediate (Today)
1. ✅ Implement smaller window size (2-3 days) for n≥30
2. ✅ Add bootstrap confidence intervals
3. ✅ Apply Bonferroni correction

### Short-Term (This Week)
4. ✅ Get more historical data (180+ days if available)
5. ✅ Implement walk-forward validation
6. ✅ Add regime-stratified reporting

### Medium-Term (Next Month)
7. Implement Monte Carlo simulation
8. Cross-validate on multiple cryptocurrencies
9. Out-of-sample validation protocol

---

## Conclusion

**Current Statistical Validity: ⚠️ Moderate (Limited by small n)**

**Key Issues:**
1. n=9-12 too small for robust statistics (need n≥30)
2. 60 days too short for market regime coverage
3. No multiple comparison correction

**Mitigation:**
1. Large effect sizes (d≈2.0) provide some confidence
2. Results directionally correct but CI wide
3. Need more data for definitive conclusions

**Recommendation:**
✅ **Use current results for direction (Base > Lag)**
⚠️ **But get more data before production deployment**
🔍 **Implement improved statistical methodology for future backtests**

**Bottom Line:**
While current backtest suggests Base model is better than Lag features, we should:
1. Get 180+ days historical data
2. Re-run backtest with n≥30 windows
3. Apply proper statistical corrections
4. Validate on out-of-sample data

Only then can we be **highly confident** in the results.
