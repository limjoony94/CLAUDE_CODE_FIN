# Feature Calculation Mismatch Analysis

**Investigation Date**: 2025-10-28
**Issue**: Walk-Forward Decoupled Entry models failing in production (0.0000 probability)
**Goal**: Identify differences between Training, Backtest, and Production feature calculation

---

## Executive Summary

### 🚨 CRITICAL MISMATCHES FOUND

| Critical Area | Training | Backtest | Production | Impact |
|--------------|----------|----------|------------|--------|
| **NaN Handling** | `ffill().bfill().fillna(0)` | `dropna()` | `ffill().bfill().fillna(0)` | 🔴 **CRITICAL** |
| **Data Loss** | 0 rows lost | Loses rows | 0 rows lost | 🔴 **CRITICAL** |
| **Row Alignment** | Training rows ≠ Backtest rows | Different | - | 🔴 **CRITICAL** |

### Root Cause Analysis

The **PRIMARY ISSUE** is that **training and backtest use DIFFERENT NaN handling**:

1. **Training** (`calculate_all_features.py` line 201):
   ```python
   df = df.ffill().bfill().fillna(0)  # ✅ Keeps ALL rows
   ```

2. **Backtest** (`backtest_walkforward_models_075.py` line 61):
   ```python
   df = prepare_exit_features(df)
   # Which calls calculate_all_features_enhanced_v2() line 247:
   df = df.dropna().reset_index(drop=True)  # ❌ DROPS ROWS!
   ```

3. **Production** (calls same pipeline as training):
   ```python
   df_features = calculate_all_features_enhanced_v2(df.copy(), phase='phase1')
   # Which internally does: df.dropna()  # ❌ DROPS ROWS!
   ```

**Result**: Models were trained on data with different row counts and indices than what's used in backtest/production!

---

## Detailed Comparison

### 1. Feature Calculation Pipeline

#### Training (calculate_all_features.py)

```python
def calculate_all_features(df):
    """Training pipeline - 4 steps"""

    # Step 1: LONG basic features
    df = calculate_features(df)

    # Step 2: LONG advanced features
    adv_features = AdvancedTechnicalFeatures(lookback_sr=200, lookback_trend=50)
    df = adv_features.calculate_all_features(df)

    # Step 3: SHORT features
    df = calculate_symmetric_features(df)
    df = calculate_inverse_features(df)
    df = calculate_opportunity_cost_features(df)

    # Step 4: ⚠️ CRITICAL - Clean NaN
    df = df.ffill().bfill().fillna(0)  # ✅ KEEPS ALL ROWS

    return df
```

#### Backtest (backtest_walkforward_models_075.py)

```python
# Line 55-62
df = pd.read_csv(DATA_DIR / "BTCUSDT_5m_features.csv")

# Prepare Exit features (adds 15 enhanced features for Exit models)
df = prepare_exit_features(df)
# ⚠️ THIS CALLS: calculate_all_features_enhanced_v2()
```

#### Production (opportunity_gating_bot_4x.py)

```python
# Line 1095-1096
df_features = calculate_all_features_enhanced_v2(df.copy(), phase='phase1')
df_features = prepare_exit_features(df_features)  # Add EXIT-specific features
```

**Analysis**:
- ✅ **IDENTICAL**: All 3 use same base feature functions
- 🚨 **CRITICAL DIFFERENCE**: Training uses `ffill().bfill().fillna(0)`, others use `dropna()`

---

### 2. NaN/Inf Handling - THE CRITICAL MISMATCH

#### Training: Keep ALL rows, fill NaN with 0

```python
# calculate_all_features.py Line 201
df = df.ffill().bfill().fillna(0)
```

**Effect**:
- ✅ No rows lost
- ✅ All candles included
- ✅ Index preserved
- ⚠️ Early candles (first 200) have 0-filled features

#### Backtest/Production: DROP rows with NaN

```python
# calculate_all_features_enhanced_v2.py Line 247
df = df.dropna().reset_index(drop=True)
```

**Effect**:
- ❌ First ~200-300 rows DROPPED (due to rolling windows)
- ❌ Index RESET (0, 1, 2, ...)
- ❌ Row count DIFFERENT from training
- 🚨 **MODELS TRAINED ON DIFFERENT DATA THAN WHAT'S USED IN PRODUCTION**

---

### 3. Feature Order & Selection

#### Training

```python
# calculate_all_features.py Line 209-230
SHORT_FEATURE_COLUMNS = [
    # Symmetric (13)
    'rsi_deviation', 'rsi_direction', 'rsi_extreme',
    'macd_strength', 'macd_direction', 'macd_divergence_abs',
    'price_distance_ma20', 'price_direction_ma20',
    'price_distance_ma50', 'price_direction_ma50',
    'volatility', 'atr_pct', 'atr',

    # Inverse (15)
    'negative_momentum', 'negative_acceleration',
    'down_candle_ratio', 'down_candle_body',
    'lower_low_streak', 'resistance_rejection_count',
    'bearish_divergence', 'volume_decline_ratio',
    'distribution_signal', 'down_candle',
    'lower_low', 'near_resistance', 'rejection_from_resistance',
    'volume_on_decline', 'volume_on_advance',

    # Opportunity Cost (10)
    'bear_market_strength', 'trend_strength', 'downtrend_confirmed',
    'volatility_asymmetry', 'below_support', 'support_breakdown',
    'panic_selling', 'downside_volatility', 'upside_volatility', 'ema_12'
]
# Total: 38 features
```

#### Backtest/Production (Same as Training)

```python
# Features loaded from model files:
# xgboost_long_entry_walkforward_decoupled_20251027_194313_features.txt
# xgboost_short_entry_walkforward_decoupled_20251027_194313_features.txt
```

**Analysis**:
- ✅ **IDENTICAL**: Feature names and order are same
- ✅ **IDENTICAL**: Feature selection follows model's feature list

---

### 4. Data Preprocessing & Scaling

#### Training

```python
# StandardScaler fitted during training
# Saved as: *_scaler.pkl
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

#### Backtest/Production

```python
# Load saved scaler
scaler = joblib.load(MODELS_DIR / "*_scaler.pkl")
X_scaled = scaler.transform(X)
```

**Analysis**:
- ✅ **IDENTICAL**: Same scaler used (fit during training, transform in production)
- ✅ **IDENTICAL**: StandardScaler parameters preserved

---

### 5. Edge Cases & Warmup Period

#### Training

```python
# First 200-300 rows have 0-filled features due to:
# - Rolling windows (200-period lookbacks)
# - ffill().bfill().fillna(0)

# But these rows are INCLUDED in training data!
```

#### Backtest/Production

```python
# First 200-300 rows DROPPED due to:
# - Rolling windows (200-period lookbacks)
# - df.dropna()

# These rows are EXCLUDED from inference!
```

**Analysis**:
- 🚨 **CRITICAL MISMATCH**: Training includes 0-filled early rows, production drops them
- 🚨 **Model never saw dropna() data during training!**

---

### 6. Timestamp Handling

#### Training

```python
# No explicit timestamp filtering
# All rows kept after feature calculation
```

#### Backtest

```python
# Line 232-233 (per window)
df_window = df.iloc[start:end].copy()
df_window = df_window.reset_index(drop=True)
```

#### Production

```python
# Lines 967-1052: filter_completed_candles()
# Filters out in-progress candles based on timestamp
df_completed = df[df['timestamp'] < current_candle_start].copy()
```

**Analysis**:
- ✅ **SAFE**: Timestamp handling is correct
- ✅ **SAFE**: Production correctly filters completed candles only

---

## Impact Assessment

### Why Models Return 0.0000 in Production

**Scenario**: Production receives live data with 1000 candles

1. **Feature Calculation** calls `calculate_all_features_enhanced_v2()`:
   ```python
   df_features = calculate_all_features_enhanced_v2(df.copy(), phase='phase1')
   ```

2. **Inside this function** (line 247):
   ```python
   df = df.dropna().reset_index(drop=True)  # ❌ DROPS ~200 rows!
   ```

3. **Result**:
   - Input: 1000 candles (indices 0-999)
   - After dropna: ~800 candles (indices 0-799, **BUT RESET**)
   - Latest candle: Now at index 799 (was 999)

4. **Production then does**:
   ```python
   latest = df_features.iloc[-1:].copy()  # Gets index 799
   ```

5. **But model was trained expecting**:
   - Index 999 data (NOT 799)
   - Features with 0-fills (NOT dropped rows)
   - Different statistical distribution

**Result**: Model encounters **distribution shift** and returns 0.0000 (no confidence)

---

## Recommended Fix

### Option 1: Align ALL to ffill().bfill().fillna(0) ✅ RECOMMENDED

**Changes Required**:

1. **Modify** `calculate_all_features_enhanced_v2.py` line 247:
   ```python
   # BEFORE
   df = df.dropna().reset_index(drop=True)

   # AFTER
   df = df.ffill().bfill().fillna(0)  # Align with training
   ```

2. **Test backtest** with this change - performance should improve

3. **Deploy to production** - models should work correctly

**Pros**:
- ✅ Minimal code changes
- ✅ Matches training exactly
- ✅ Preserves all candles
- ✅ Production already uses this indirectly (through calculate_all_features)

**Cons**:
- ⚠️ Early candles have 0-filled features (but model trained on this)

---

### Option 2: Retrain with dropna() ❌ NOT RECOMMENDED

**Changes Required**:

1. **Modify training** to use dropna() instead of fillna(0)
2. **Retrain ALL models** (Entry + Exit)
3. **Re-validate backtests**

**Pros**:
- Cleaner data (no 0-fills)

**Cons**:
- ❌ Requires full retraining (hours of work)
- ❌ Loses historical data
- ❌ Risk of introducing new bugs
- ❌ Backtests need re-validation

---

## Verification Steps

### After Applying Fix

1. **Unit Test**: Compare feature calculation outputs
   ```python
   # Test that training and production produce IDENTICAL features
   df_train = calculate_all_features(df_raw.copy())
   df_prod = calculate_all_features_enhanced_v2(df_raw.copy(), phase='phase1')

   assert len(df_train) == len(df_prod)
   assert df_train.index.equals(df_prod.index)
   ```

2. **Backtest Test**: Run backtest with fixed code
   ```bash
   python scripts/experiments/backtest_walkforward_models_075.py
   ```
   - Expected: Win Rate > 50%, ML Exit > 0%

3. **Production Test**: Deploy to testnet first
   - Monitor probabilities (should be > 0.0000)
   - Verify signals generated

---

## Conclusion

### Root Cause

**Training used `ffill().bfill().fillna(0)` while backtest/production used `dropna()`**

This caused:
1. Different row counts between training and inference
2. Different index ranges
3. Distribution shift (0-fills vs dropped rows)
4. Models returning 0.0000 (no confidence)

### Solution

**Change `calculate_all_features_enhanced_v2.py` line 247 from `dropna()` to `ffill().bfill().fillna(0)`**

This aligns all systems with training methodology.

---

## Files Affected

### To Modify

1. `scripts/experiments/calculate_all_features_enhanced_v2.py`
   - Line 247: Change `dropna()` to `ffill().bfill().fillna(0)`

### To Test

1. `scripts/experiments/backtest_walkforward_models_075.py`
2. `scripts/production/opportunity_gating_bot_4x.py`

### To Verify

1. Compare training vs production feature outputs
2. Run 108-window backtest
3. Test on testnet before mainnet

---

**Report Generated**: 2025-10-28
**Analysis By**: Root Cause Analyst Persona
**Confidence**: 🟢 **VERY HIGH** - Clear mismatch identified with concrete evidence
