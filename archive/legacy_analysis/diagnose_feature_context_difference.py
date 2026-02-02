"""
Feature Context Difference Diagnostic
======================================

Compares feature calculations between two contexts:
1. BACKTEST CONTEXT: Full 30,296-candle dataset (static)
2. PRODUCTION CONTEXT: Rolling 1000-candle window

Purpose: Identify EXACT numerical differences in feature values
         that cause signal strength degradation in production.

Root Cause Investigation for:
- Production signals: LONG 0.6418, SHORT 0.7280 (below 0.80)
- Backtest signals: Regularly > 0.80 (7.36 trades/day)

Created: 2025-10-28
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import pickle
import joblib
from datetime import datetime

from scripts.experiments.calculate_all_features_enhanced_v2 import calculate_all_features_enhanced_v2
from scripts.experiments.retrain_exit_models_opportunity_gating import prepare_exit_features

# Paths
DATA_DIR = PROJECT_ROOT / "data" / "historical"  # Changed to historical for raw OHLCV
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

print("="*80)
print("FEATURE CONTEXT DIFFERENCE DIAGNOSTIC")
print("="*80)
print()
print("Purpose: Compare feature calculations in backtest vs production contexts")
print("Method: Same 1000 candles, different calculation contexts")
print()

# ============================================================================
# STEP 1: Load Full Dataset (Backtest Context)
# ============================================================================

print("-"*80)
print("STEP 1: Loading Full Dataset (Backtest Context)")
print("-"*80)

# Load RAW OHLCV data (not pre-calculated features)
df_full = pd.read_csv(DATA_DIR / "BTCUSDT_5m_max.csv")
print(f"✅ Loaded {len(df_full):,} candles (RAW OHLCV)")
print(f"   Date range: {df_full['timestamp'].iloc[0]} to {df_full['timestamp'].iloc[-1]}")
print(f"   Columns: {list(df_full.columns)}")
print()

# ============================================================================
# STEP 2: Extract Last 1000 Candles for Production Simulation
# ============================================================================

print("-"*80)
print("STEP 2: Extracting Last 1000 Candles (Production Simulation)")
print("-"*80)

# Get the EXACT last 1000 candles that production would fetch
df_production_raw = df_full.iloc[-1000:].copy().reset_index(drop=True)
print(f"✅ Extracted {len(df_production_raw):,} candles")
print(f"   Date range: {df_production_raw['timestamp'].iloc[0]} to {df_production_raw['timestamp'].iloc[-1]}")
print()

# ============================================================================
# STEP 3: Calculate Features - BACKTEST CONTEXT
# ============================================================================

print("-"*80)
print("STEP 3: Calculating Features - BACKTEST CONTEXT (Full Dataset)")
print("-"*80)
print("This is how backtest calculates features on full 30,296 candles")
print()

# Calculate on FULL dataset (backtest way)
df_backtest = calculate_all_features_enhanced_v2(df_full.copy(), phase='phase1')
df_backtest = prepare_exit_features(df_backtest)

print(f"✅ Backtest features calculated: {len(df_backtest):,} complete rows")
print(f"   NaN removed: {len(df_full) - len(df_backtest):,} rows")
print()

# ============================================================================
# STEP 4: Calculate Features - PRODUCTION CONTEXT
# ============================================================================

print("-"*80)
print("STEP 4: Calculating Features - PRODUCTION CONTEXT (1000 Candles)")
print("-"*80)
print("This is how production calculates features on rolling 1000-candle window")
print()

# Calculate on 1000-candle window (production way)
df_production = calculate_all_features_enhanced_v2(df_production_raw.copy(), phase='phase1')
df_production = prepare_exit_features(df_production)

print(f"✅ Production features calculated: {len(df_production):,} complete rows")
print(f"   NaN removed: {len(df_production_raw) - len(df_production):,} rows")
print()

# ============================================================================
# STEP 5: Align and Compare LAST CANDLE
# ============================================================================

print("-"*80)
print("STEP 5: Comparing LAST CANDLE Features")
print("-"*80)
print()

# Get the LAST complete candle from both contexts
# Find the matching timestamp
production_last_timestamp = df_production['timestamp'].iloc[-1]
print(f"Production last timestamp: {production_last_timestamp}")

# Find this timestamp in backtest
backtest_match_idx = df_backtest[df_backtest['timestamp'] == production_last_timestamp].index
if len(backtest_match_idx) == 0:
    print("⚠️ WARNING: Could not find matching timestamp in backtest data!")
    print("   This should not happen - investigating...")
    sys.exit(1)

backtest_match_idx = backtest_match_idx[0]
print(f"Backtest matching index: {backtest_match_idx}")
print()

# Extract the matching rows
backtest_last = df_backtest.iloc[backtest_match_idx].copy()
production_last = df_production.iloc[-1].copy()

# ============================================================================
# STEP 6: Load Models and Feature Lists
# ============================================================================

print("-"*80)
print("STEP 6: Loading Models and Feature Lists")
print("-"*80)

# LONG Entry
with open(MODELS_DIR / "xgboost_long_entry_enhanced_20251024_012445_features.txt", 'r') as f:
    long_features = [line.strip() for line in f]

print(f"✅ LONG Entry: {len(long_features)} features")

# SHORT Entry
with open(MODELS_DIR / "xgboost_short_entry_enhanced_20251024_012445_features.txt", 'r') as f:
    short_features = [line.strip() for line in f]

print(f"✅ SHORT Entry: {len(short_features)} features")
print()

# ============================================================================
# STEP 7: Compare Feature Values
# ============================================================================

print("="*80)
print("FEATURE VALUE COMPARISON - ROOT CAUSE ANALYSIS")
print("="*80)
print()

# Compare LONG features
print("-"*80)
print("LONG ENTRY FEATURES (85 features)")
print("-"*80)
print()

long_diffs = []
for feature in long_features:
    backtest_val = backtest_last[feature]
    production_val = production_last[feature]

    # Calculate absolute and relative difference
    abs_diff = abs(backtest_val - production_val)

    if abs(backtest_val) > 1e-10:
        rel_diff_pct = (abs_diff / abs(backtest_val)) * 100
    else:
        rel_diff_pct = 0 if abs_diff < 1e-10 else 999.9

    long_diffs.append({
        'feature': feature,
        'backtest': backtest_val,
        'production': production_val,
        'abs_diff': abs_diff,
        'rel_diff_pct': rel_diff_pct
    })

df_long_diffs = pd.DataFrame(long_diffs)
df_long_diffs = df_long_diffs.sort_values('abs_diff', ascending=False)

# Show top 20 differences
print("Top 20 Features with Largest Absolute Differences:")
print()
for idx, row in df_long_diffs.head(20).iterrows():
    print(f"{row['feature']:40s} | Backtest: {row['backtest']:12.6f} | Production: {row['production']:12.6f} | "
          f"Diff: {row['abs_diff']:10.6f} | {row['rel_diff_pct']:7.2f}%")

print()

# Statistics
print("STATISTICS:")
print(f"  Mean absolute difference: {df_long_diffs['abs_diff'].mean():.6f}")
print(f"  Median absolute difference: {df_long_diffs['abs_diff'].median():.6f}")
print(f"  Max absolute difference: {df_long_diffs['abs_diff'].max():.6f}")
print(f"  Features with >1% relative difference: {(df_long_diffs['rel_diff_pct'] > 1.0).sum()}")
print(f"  Features with >5% relative difference: {(df_long_diffs['rel_diff_pct'] > 5.0).sum()}")
print(f"  Features with >10% relative difference: {(df_long_diffs['rel_diff_pct'] > 10.0).sum()}")
print()

# Compare SHORT features
print("-"*80)
print("SHORT ENTRY FEATURES (79 features)")
print("-"*80)
print()

short_diffs = []
for feature in short_features:
    backtest_val = backtest_last[feature]
    production_val = production_last[feature]

    abs_diff = abs(backtest_val - production_val)

    if abs(backtest_val) > 1e-10:
        rel_diff_pct = (abs_diff / abs(backtest_val)) * 100
    else:
        rel_diff_pct = 0 if abs_diff < 1e-10 else 999.9

    short_diffs.append({
        'feature': feature,
        'backtest': backtest_val,
        'production': production_val,
        'abs_diff': abs_diff,
        'rel_diff_pct': rel_diff_pct
    })

df_short_diffs = pd.DataFrame(short_diffs)
df_short_diffs = df_short_diffs.sort_values('abs_diff', ascending=False)

print("Top 20 Features with Largest Absolute Differences:")
print()
for idx, row in df_short_diffs.head(20).iterrows():
    print(f"{row['feature']:40s} | Backtest: {row['backtest']:12.6f} | Production: {row['production']:12.6f} | "
          f"Diff: {row['abs_diff']:10.6f} | {row['rel_diff_pct']:7.2f}%")

print()

print("STATISTICS:")
print(f"  Mean absolute difference: {df_short_diffs['abs_diff'].mean():.6f}")
print(f"  Median absolute difference: {df_short_diffs['abs_diff'].median():.6f}")
print(f"  Max absolute difference: {df_short_diffs['abs_diff'].max():.6f}")
print(f"  Features with >1% relative difference: {(df_short_diffs['rel_diff_pct'] > 1.0).sum()}")
print(f"  Features with >5% relative difference: {(df_short_diffs['rel_diff_pct'] > 5.0).sum()}")
print(f"  Features with >10% relative difference: {(df_short_diffs['rel_diff_pct'] > 10.0).sum()}")
print()

# ============================================================================
# STEP 8: Identify Critical Features (200-period and rolling features)
# ============================================================================

print("-"*80)
print("STEP 8: Critical Features Analysis")
print("-"*80)
print("Checking features most likely to differ due to context:")
print()

critical_features = [
    'ma_200', 'ema_200', 'volume_ma_200', 'atr_200', 'rsi_200',
    'bb_upper_200', 'bb_mid_200', 'bb_lower_200',
    'vwap', 'vp_poc', 'vp_value_area_high', 'vp_value_area_low'
]

print("200-Period and Rolling Window Features:")
for feature in critical_features:
    if feature in long_features:
        row = df_long_diffs[df_long_diffs['feature'] == feature].iloc[0]
        print(f"  {feature:30s} | Backtest: {row['backtest']:12.6f} | Production: {row['production']:12.6f} | "
              f"Diff: {row['abs_diff']:10.6f} | {row['rel_diff_pct']:7.2f}%")

print()

# ============================================================================
# STEP 9: Calculate Model Probabilities in Both Contexts
# ============================================================================

print("-"*80)
print("STEP 9: Model Probability Comparison")
print("-"*80)
print()

# Load models and scalers
print("Loading LONG Entry model...")
with open(MODELS_DIR / "xgboost_long_entry_enhanced_20251024_012445.pkl", 'rb') as f:
    long_model = pickle.load(f)
long_scaler = joblib.load(MODELS_DIR / "xgboost_long_entry_enhanced_20251024_012445_scaler.pkl")

print("Loading SHORT Entry model...")
with open(MODELS_DIR / "xgboost_short_entry_enhanced_20251024_012445.pkl", 'rb') as f:
    short_model = pickle.load(f)
short_scaler = joblib.load(MODELS_DIR / "xgboost_short_entry_enhanced_20251024_012445_scaler.pkl")
print()

# LONG predictions
backtest_long_feat = backtest_last[long_features].values.reshape(1, -1)
production_long_feat = production_last[long_features].values.reshape(1, -1)

backtest_long_scaled = long_scaler.transform(backtest_long_feat)
production_long_scaled = long_scaler.transform(production_long_feat)

backtest_long_prob = long_model.predict_proba(backtest_long_scaled)[0][1]
production_long_prob = long_model.predict_proba(production_long_scaled)[0][1]

print("LONG ENTRY SIGNAL:")
print(f"  Backtest context:   {backtest_long_prob:.4f}")
print(f"  Production context: {production_long_prob:.4f}")
print(f"  Difference:         {abs(backtest_long_prob - production_long_prob):.4f}")
print()

# SHORT predictions
backtest_short_feat = backtest_last[short_features].values.reshape(1, -1)
production_short_feat = production_last[short_features].values.reshape(1, -1)

backtest_short_scaled = short_scaler.transform(backtest_short_feat)
production_short_scaled = short_scaler.transform(production_short_feat)

backtest_short_prob = short_model.predict_proba(backtest_short_scaled)[0][1]
production_short_prob = short_model.predict_proba(production_short_scaled)[0][1]

print("SHORT ENTRY SIGNAL:")
print(f"  Backtest context:   {backtest_short_prob:.4f}")
print(f"  Production context: {production_short_prob:.4f}")
print(f"  Difference:         {abs(backtest_short_prob - production_short_prob):.4f}")
print()

# ============================================================================
# STEP 10: Save Results
# ============================================================================

print("-"*80)
print("STEP 10: Saving Results")
print("-"*80)

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

# Save LONG differences
long_output = RESULTS_DIR / f"feature_diff_long_{timestamp}.csv"
df_long_diffs.to_csv(long_output, index=False)
print(f"✅ LONG differences saved: {long_output.name}")

# Save SHORT differences
short_output = RESULTS_DIR / f"feature_diff_short_{timestamp}.csv"
df_short_diffs.to_csv(short_output, index=False)
print(f"✅ SHORT differences saved: {short_output.name}")

print()

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("="*80)
print("ROOT CAUSE ANALYSIS SUMMARY")
print("="*80)
print()

print("PROBABILITY DIFFERENCES:")
print(f"  LONG:  Backtest {backtest_long_prob:.4f} → Production {production_long_prob:.4f} "
      f"(Δ = {abs(backtest_long_prob - production_long_prob):.4f})")
print(f"  SHORT: Backtest {backtest_short_prob:.4f} → Production {production_short_prob:.4f} "
      f"(Δ = {abs(backtest_short_prob - production_short_prob):.4f})")
print()

print("FEATURE DIFFERENCES:")
print(f"  LONG:  {(df_long_diffs['rel_diff_pct'] > 1.0).sum()} features differ by >1%")
print(f"  SHORT: {(df_short_diffs['rel_diff_pct'] > 1.0).sum()} features differ by >1%")
print()

print("CRITICAL INSIGHT:")
if (df_long_diffs['rel_diff_pct'] > 1.0).sum() > 0 or (df_short_diffs['rel_diff_pct'] > 1.0).sum() > 0:
    print("  ⚠️ CONTEXT-DEPENDENT FEATURE CALCULATION CONFIRMED")
    print("  Features computed on 1000-candle window differ from full dataset context")
    print("  This explains why production signals are systematically lower than backtest")
    print()
    print("  Root Cause: Rolling window features (MA200, VWAP, Volume Profile, etc.)")
    print("              calculate differently with limited historical context")
else:
    print("  ✅ Features are IDENTICAL between contexts")
    print("  Problem must be elsewhere - checking model/scaler files or prediction logic")

print()
print("="*80)
print("DIAGNOSTIC COMPLETE")
print("="*80)
