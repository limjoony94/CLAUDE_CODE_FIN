# Feature Calculation Mismatch - Visual Code Comparison

**Date**: 2025-10-28
**Issue**: Training vs Production NaN handling mismatch causing 0.0000 predictions

---

## 🚨 THE CRITICAL DIFFERENCE

### Side-by-Side Comparison

```diff
┌─────────────────────────────────────────────────────────────────────┐
│ TRAINING: calculate_all_features.py (Line 201)                     │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   def calculate_all_features(df):                                  │
│       # ... calculate features ...                                 │
│                                                                     │
│       # Step 3: Clean NaN                                          │
│       print("  4/4 Cleaning NaN values...")                        │
+       df = df.ffill().bfill().fillna(0)  ✅ KEEPS ALL ROWS         │
│                                                                     │
│       print(f"  ✅ All features calculated ({len(df)} rows)")      │
│       return df                                                     │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────┐
│ PRODUCTION: calculate_all_features_enhanced_v2.py (Line 247)       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   def calculate_all_features_enhanced_v2(df, phase='phase1'):      │
│       # ... calculate features ...                                 │
│                                                                     │
│       # Clean NaN values                                           │
│       print("\nCleaning NaN values...")                            │
-       df = df.dropna().reset_index(drop=True)  ❌ DROPS ~200 ROWS  │
│       final_rows = len(df)                                         │
│                                                                     │
│       return df                                                     │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Impact Visualization

### Training Data Flow

```
┌──────────────────────────────────────────────────────────────────┐
│ RAW DATA (1000 candles)                                          │
│ Index: 0, 1, 2, ..., 997, 998, 999                               │
└───────────────────────┬──────────────────────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────────────────────┐
│ CALCULATE FEATURES (rolling windows 200)                         │
│ - First 200 rows: NaN in long-term features                      │
│ - Rows 200-999: Complete features                                │
└───────────────────────┬──────────────────────────────────────────┘
                        │
                        ▼ ffill().bfill().fillna(0)
                        │
┌──────────────────────────────────────────────────────────────────┐
│ TRAINING DATA (1000 candles) ✅ ALL KEPT                         │
│ Index: 0, 1, 2, ..., 997, 998, 999                               │
│ - Rows 0-199: Features filled with 0 (from bfill/fillna)         │
│ - Rows 200-999: Real calculated features                         │
└──────────────────────────────────────────────────────────────────┘
```

### Production Data Flow (BROKEN)

```
┌──────────────────────────────────────────────────────────────────┐
│ RAW DATA (1000 candles)                                          │
│ Index: 0, 1, 2, ..., 997, 998, 999                               │
└───────────────────────┬──────────────────────────────────────────┘
                        │
                        ▼
┌──────────────────────────────────────────────────────────────────┐
│ CALCULATE FEATURES (rolling windows 200)                         │
│ - First 200 rows: NaN in long-term features                      │
│ - Rows 200-999: Complete features                                │
└───────────────────────┬──────────────────────────────────────────┘
                        │
                        ▼ dropna().reset_index(drop=True)
                        │
┌──────────────────────────────────────────────────────────────────┐
│ PRODUCTION DATA (800 candles) ❌ FIRST 200 DROPPED               │
│ Index: 0, 1, 2, ..., 797, 798, 799  ⚠️ RESET!                    │
│ - Original row 200 → NEW index 0                                 │
│ - Original row 999 → NEW index 799                               │
└───────────────────────┬──────────────────────────────────────────┘
                        │
                        ▼ latest = df.iloc[-1:]
                        │
┌──────────────────────────────────────────────────────────────────┐
│ LATEST CANDLE: Index 799 (was 999)                               │
│ ⚠️ MODEL EXPECTS: Index 999 data with 0-fills                    │
│ 🚨 DISTRIBUTION SHIFT: Model returns 0.0000                       │
└──────────────────────────────────────────────────────────────────┘
```

---

## Exact Code Locations

### File 1: Training (CORRECT ✅)

**File**: `scripts/experiments/calculate_all_features.py`

```python
# Lines 199-205
def calculate_all_features(df):
    # ... feature calculation ...

    # Step 3: Clean NaN
    print("  4/4 Cleaning NaN values...")
    df = df.ffill().bfill().fillna(0)  # ✅ CORRECT: Keeps all rows

    print(f"  ✅ All features calculated ({len(df)} rows)")
    return df
```

**Behavior**:
- ✅ Forward fill NaN from valid values
- ✅ Backward fill remaining NaN
- ✅ Fill any remaining NaN with 0
- ✅ NO ROWS LOST
- ✅ Index preserved (0 to N-1)

---

### File 2: Production (BROKEN ❌)

**File**: `scripts/experiments/calculate_all_features_enhanced_v2.py`

```python
# Lines 245-249
def calculate_all_features_enhanced_v2(df, phase='phase1'):
    # ... feature calculation ...

    # Clean NaN values
    print("\nCleaning NaN values...")
    df = df.dropna().reset_index(drop=True)  # ❌ WRONG: Drops rows!
    final_rows = len(df)

    return df
```

**Behavior**:
- ❌ Drops ALL rows with ANY NaN value
- ❌ First ~200 rows LOST (due to rolling windows)
- ❌ Index RESET to 0, 1, 2, ... (no longer matches training)
- ❌ Row count DIFFERENT from training
- ❌ Latest candle at different index than training expected

---

### File 3: Backtest (Uses Broken Production Code)

**File**: `scripts/experiments/backtest_walkforward_models_075.py`

```python
# Lines 59-62
df = pd.read_csv(DATA_DIR / "BTCUSDT_5m_features.csv")

# Prepare Exit features (adds 15 enhanced features for Exit models)
print("\nPreparing Exit features...")
df = prepare_exit_features(df)  # ⚠️ Calls calculate_all_features_enhanced_v2()
```

**Flow**:
```
prepare_exit_features(df)
  └─> Adds 15 new features
      └─> Returns df (no dropna here)

BUT prepare_exit_features() does NOT call calculate_all_features_enhanced_v2()
So WHERE is the dropna() happening?

Answer: backtest_walkforward_models_075.py doesn't call
calculate_all_features_enhanced_v2() at all!
It loads PRE-CALCULATED features from CSV: "BTCUSDT_5m_features.csv"
```

**Wait, let me re-check this...**

Actually, looking at the backtest code more carefully:
- Line 55: `df = pd.read_csv(DATA_DIR / "BTCUSDT_5m_features.csv")`
- This CSV was pre-calculated using which method?

Let me trace backward...

---

### File 4: Production Bot (Uses Both)

**File**: `scripts/production/opportunity_gating_bot_4x.py`

```python
# Lines 1095-1096
df_features = calculate_all_features_enhanced_v2(df.copy(), phase='phase1')
df_features = prepare_exit_features(df_features)  # Add EXIT-specific features
```

**Flow**:
```
calculate_all_features_enhanced_v2(df)
  └─> Calls calculate_all_features(df)        # ✅ Uses ffill/bfill/fillna(0)
  └─> Calls calculate_long_term_features(df)
  └─> Calls calculate_all_advanced_indicators(df)
  └─> Calls calculate_advanced_ratio_features(df)
  └─> df.dropna().reset_index(drop=True)     # ❌ Then DROPS rows!

prepare_exit_features(df_features)
  └─> Adds 15 features (volume_surge, price_vs_ma20, etc.)
  └─> Returns df (no additional NaN handling)
```

---

## The Nested Call Problem

### Production Bot Execution Order

```
opportunity_gating_bot_4x.py: get_signals()
│
├─> calculate_all_features_enhanced_v2(df, phase='phase1')
│   │
│   ├─> calculate_all_features(df)  ✅ Uses ffill().bfill().fillna(0)
│   │   └─> Returns 1000 rows with 0-fills
│   │
│   ├─> calculate_long_term_features(df)
│   │   └─> Adds features, may create NaN
│   │
│   ├─> calculate_all_advanced_indicators(df)
│   │   └─> Adds features, may create NaN
│   │
│   ├─> calculate_advanced_ratio_features(df)
│   │   └─> Adds features, may create NaN
│   │
│   └─> df.dropna().reset_index(drop=True)  ❌ DROPS rows created by steps above!
│       └─> Returns ~800 rows (200 lost)
│
└─> prepare_exit_features(df_features)
    └─> Adds 15 features
    └─> Returns df (no dropna)
```

### Training Execution Order

```
Training Script
│
└─> calculate_all_features(df)  ✅ Complete pipeline
    │
    ├─> calculate_features(df)          # LONG basic
    ├─> AdvancedTechnicalFeatures(df)  # LONG advanced
    ├─> calculate_symmetric_features(df)  # SHORT symmetric
    ├─> calculate_inverse_features(df)    # SHORT inverse
    ├─> calculate_opportunity_cost_features(df)  # SHORT opportunity
    │
    └─> df.ffill().bfill().fillna(0)  ✅ KEEPS all rows
        └─> Returns 1000 rows with 0-fills
```

**KEY DIFFERENCE**: Training does NOT call `calculate_all_features_enhanced_v2()`!

---

## Evidence Summary

### What We Know

1. **Training** uses:
   - `calculate_all_features()` ONLY
   - Ends with: `df.ffill().bfill().fillna(0)`
   - Keeps ALL rows

2. **Production** uses:
   - `calculate_all_features_enhanced_v2()` which internally calls `calculate_all_features()`
   - Adds: Long-term features (23) + Advanced indicators (11) + Ratios (24)
   - Ends with: `df.dropna().reset_index(drop=True)`
   - **DROPS ~200 rows**

3. **Backtest** uses:
   - Pre-calculated CSV: `BTCUSDT_5m_features.csv`
   - Need to check: How was this CSV created?
   - Then adds: `prepare_exit_features()` (15 features)

---

## The Real Question

**Which feature set was used during Walk-Forward Decoupled training?**

Looking at training script name: `retrain_entry_walkforward_decoupled_075.py`

Need to check:
1. What features does it calculate?
2. Does it use `calculate_all_features()` or `calculate_all_features_enhanced_v2()`?

Let me check the training script...

---

## Next Investigation Step

**Check**: `scripts/experiments/retrain_entry_walkforward_decoupled_075.py`

Key questions:
1. Which feature calculation function does it use?
2. Does it match production's feature calculation?
3. Does it use ffill/bfill/fillna(0) or dropna()?

**Expected Finding**: Training likely uses DIFFERENT feature calculation than production!

---

**Report Status**: ⚠️ PARTIAL - Need to verify training script feature calculation
**Next Action**: Read `retrain_entry_walkforward_decoupled_075.py` feature calculation section
