"""
Feature Value Comparison at Target Timestamp
=============================================

Compare exact feature values between:
1. Live bot execution (simulated with 1000 candles ending at target time)
2. Backtest execution (historical CSV with different lookback window)

Goal: Identify which features differ and cause signal prediction mismatch.

Target: 2025-10-24 01:00 KST (2025-10-23 16:00 UTC)
Live Signal: SHORT 0.8122, LONG 0.3707
Backtest Signal: SHORT 0.6512, LONG 0.0846
"""

import pandas as pd
import numpy as np
import pickle
import joblib
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiments.calculate_all_features_enhanced_v2 import calculate_all_features_enhanced_v2

DATA_DIR = PROJECT_ROOT / "data" / "historical"
MODELS_DIR = PROJECT_ROOT / "models"

print("="*80)
print("FEATURE COMPARISON AT TARGET TIMESTAMP")
print("="*80)
print()

# Target timestamp
target_time_kst = pd.Timestamp('2025-10-24 01:00:00')
target_time_utc = target_time_kst - pd.Timedelta(hours=9)  # 2025-10-23 16:00:00 UTC

print(f"Target Time: {target_time_kst} KST ({target_time_utc} UTC)")
print()

# Load models and features
print("Loading models...")
long_model_path = MODELS_DIR / "xgboost_long_entry_enhanced_20251024_012445.pkl"
with open(long_model_path, 'rb') as f:
    long_model = pickle.load(f)

long_scaler_path = MODELS_DIR / "xgboost_long_entry_enhanced_20251024_012445_scaler.pkl"
long_scaler = joblib.load(long_scaler_path)

long_features_path = MODELS_DIR / "xgboost_long_entry_enhanced_20251024_012445_features.txt"
with open(long_features_path, 'r') as f:
    long_feature_columns = [line.strip() for line in f.readlines()]

short_model_path = MODELS_DIR / "xgboost_short_entry_enhanced_20251024_012445.pkl"
with open(short_model_path, 'rb') as f:
    short_model = pickle.load(f)

short_scaler_path = MODELS_DIR / "xgboost_short_entry_enhanced_20251024_012445_scaler.pkl"
short_scaler = joblib.load(short_scaler_path)

short_features_path = MODELS_DIR / "xgboost_short_entry_enhanced_20251024_012445_features.txt"
with open(short_features_path, 'r') as f:
    short_feature_columns = [line.strip() for line in f.readlines()]

print(f"✅ LONG model loaded ({len(long_feature_columns)} features)")
print(f"✅ SHORT model loaded ({len(short_feature_columns)} features)")
print()

# ============================================================================
# METHOD 1: Backtest Method (Historical CSV with large lookback)
# ============================================================================
print("="*80)
print("METHOD 1: BACKTEST METHOD (Historical CSV)")
print("="*80)
print()

df_path = DATA_DIR / "BTCUSDT_5m_updated.csv"
df_all = pd.read_csv(df_path)
df_all['timestamp'] = pd.to_datetime(df_all['timestamp'])
df_all = df_all.sort_values('timestamp').reset_index(drop=True)

# Large lookback (from 2025-10-22 00:00)
lookback_start = pd.Timestamp('2025-10-22 00:00:00')
df_backtest = df_all[(df_all['timestamp'] >= lookback_start) &
                      (df_all['timestamp'] <= target_time_utc)].copy()

print(f"Loaded {len(df_backtest)} candles (lookback from {lookback_start})")
print()

print("Calculating features (Backtest method)...")
df_backtest_features = calculate_all_features_enhanced_v2(df_backtest, phase='phase1')
print(f"✅ Features calculated: {len(df_backtest_features)} rows")
print()

# Find target timestamp
backtest_row = df_backtest_features[df_backtest_features['timestamp'] == target_time_utc]

if len(backtest_row) == 0:
    print(f"❌ Target timestamp NOT FOUND in backtest features")
    print(f"   Nearest: {df_backtest_features['timestamp'].iloc[-1]}")
    sys.exit(1)

backtest_row = backtest_row.iloc[0]
print(f"✅ Found target timestamp in backtest features")
print(f"   Index: {backtest_row.name}")
print(f"   Price: ${backtest_row['close']:,.2f}")
print()

# Extract LONG features
try:
    long_feat_backtest = backtest_row[long_feature_columns].values.reshape(1, -1)
    long_feat_scaled_backtest = long_scaler.transform(long_feat_backtest)
    long_prob_backtest = long_model.predict_proba(long_feat_scaled_backtest)[0, 1]
    print(f"LONG Probability (Backtest): {long_prob_backtest:.4f}")
except Exception as e:
    print(f"❌ LONG prediction error: {e}")
    long_prob_backtest = None

# Extract SHORT features
try:
    short_feat_backtest = backtest_row[short_feature_columns].values.reshape(1, -1)
    short_feat_scaled_backtest = short_scaler.transform(short_feat_backtest)
    short_prob_backtest = short_model.predict_proba(short_feat_scaled_backtest)[0, 1]
    print(f"SHORT Probability (Backtest): {short_prob_backtest:.4f}")
except Exception as e:
    print(f"❌ SHORT prediction error: {e}")
    short_prob_backtest = None

print()

# ============================================================================
# METHOD 2: Production Method (1000 candles ending at target time)
# ============================================================================
print("="*80)
print("METHOD 2: PRODUCTION METHOD (1000 candles ending at target)")
print("="*80)
print()

# Get 1000 candles ENDING at target time (simulating what production saw)
# Need to account for feature calculation loss (~292 rows)
# So we need 1000 + 292 = 1292 candles before target time to get 1000 after features

total_candles_needed = 1000 + 300  # 300 for VP lookback safety margin

# Find the starting point
df_sorted = df_all.sort_values('timestamp').reset_index(drop=True)
target_idx = df_sorted[df_sorted['timestamp'] == target_time_utc].index

if len(target_idx) == 0:
    print(f"❌ Target timestamp NOT FOUND in CSV")
    sys.exit(1)

target_idx = target_idx[0]
start_idx = max(0, target_idx - total_candles_needed + 1)

df_production = df_sorted.iloc[start_idx:target_idx+1].copy()

print(f"Loaded {len(df_production)} candles (simulating production window)")
print(f"   Range: {df_production['timestamp'].min()} to {df_production['timestamp'].max()}")
print()

print("Calculating features (Production method)...")
df_production_features = calculate_all_features_enhanced_v2(df_production, phase='phase1')
print(f"✅ Features calculated: {len(df_production_features)} rows")
print()

# Find target timestamp
production_row = df_production_features[df_production_features['timestamp'] == target_time_utc]

if len(production_row) == 0:
    print(f"❌ Target timestamp NOT FOUND in production features")
    print(f"   Data range after features: {df_production_features['timestamp'].min()} to {df_production_features['timestamp'].max()}")
    print(f"   Nearest: {df_production_features['timestamp'].iloc[-1]}")
    sys.exit(1)

production_row = production_row.iloc[0]
print(f"✅ Found target timestamp in production features")
print(f"   Index: {production_row.name}")
print(f"   Price: ${production_row['close']:,.2f}")
print()

# Extract LONG features
try:
    long_feat_production = production_row[long_feature_columns].values.reshape(1, -1)
    long_feat_scaled_production = long_scaler.transform(long_feat_production)
    long_prob_production = long_model.predict_proba(long_feat_scaled_production)[0, 1]
    print(f"LONG Probability (Production): {long_prob_production:.4f}")
except Exception as e:
    print(f"❌ LONG prediction error: {e}")
    long_prob_production = None

# Extract SHORT features
try:
    short_feat_production = production_row[short_feature_columns].values.reshape(1, -1)
    short_feat_scaled_production = short_scaler.transform(short_feat_production)
    short_prob_production = short_model.predict_proba(short_feat_scaled_production)[0, 1]
    print(f"SHORT Probability (Production): {short_prob_production:.4f}")
except Exception as e:
    print(f"❌ SHORT prediction error: {e}")
    short_prob_production = None

print()

# ============================================================================
# COMPARISON
# ============================================================================
print("="*80)
print("SIGNAL COMPARISON")
print("="*80)
print()

print(f"Live Bot (from logs):     LONG={0.3707:.4f}, SHORT={0.8122:.4f}")
print(f"Backtest Method:          LONG={long_prob_backtest:.4f}, SHORT={short_prob_backtest:.4f}")
print(f"Production Method:        LONG={long_prob_production:.4f}, SHORT={short_prob_production:.4f}")
print()

if long_prob_backtest is not None and long_prob_production is not None:
    print(f"LONG Difference (Backtest vs Production): {long_prob_backtest - long_prob_production:+.4f}")
if short_prob_backtest is not None and short_prob_production is not None:
    print(f"SHORT Difference (Backtest vs Production): {short_prob_backtest - short_prob_production:+.4f}")
print()

# ============================================================================
# FEATURE VALUE COMPARISON
# ============================================================================
print("="*80)
print("FEATURE VALUE COMPARISON")
print("="*80)
print()

# Compare LONG features
print("TOP 20 LONG FEATURE DIFFERENCES:")
print("-" * 80)
long_diffs = []
for feat in long_feature_columns:
    try:
        backtest_val = backtest_row[feat]
        production_val = production_row[feat]
        diff = abs(backtest_val - production_val)
        pct_diff = (diff / (abs(production_val) + 1e-10)) * 100
        long_diffs.append({
            'feature': feat,
            'backtest': backtest_val,
            'production': production_val,
            'abs_diff': diff,
            'pct_diff': pct_diff
        })
    except:
        pass

long_diffs_df = pd.DataFrame(long_diffs)
long_diffs_df = long_diffs_df.sort_values('abs_diff', ascending=False).head(20)

for idx, row in long_diffs_df.iterrows():
    print(f"{row['feature']:40s} | Backtest: {row['backtest']:12.6f} | Production: {row['production']:12.6f} | Diff: {row['abs_diff']:12.6f} ({row['pct_diff']:6.1f}%)")

print()
print()

# Compare SHORT features
print("TOP 20 SHORT FEATURE DIFFERENCES:")
print("-" * 80)
short_diffs = []
for feat in short_feature_columns:
    try:
        backtest_val = backtest_row[feat]
        production_val = production_row[feat]
        diff = abs(backtest_val - production_val)
        pct_diff = (diff / (abs(production_val) + 1e-10)) * 100
        short_diffs.append({
            'feature': feat,
            'backtest': backtest_val,
            'production': production_val,
            'abs_diff': diff,
            'pct_diff': pct_diff
        })
    except:
        pass

short_diffs_df = pd.DataFrame(short_diffs)
short_diffs_df = short_diffs_df.sort_values('abs_diff', ascending=False).head(20)

for idx, row in short_diffs_df.iterrows():
    print(f"{row['feature']:40s} | Backtest: {row['backtest']:12.6f} | Production: {row['production']:12.6f} | Diff: {row['abs_diff']:12.6f} ({row['pct_diff']:6.1f}%)")

print()
print()

# Summary
print("="*80)
print("SUMMARY")
print("="*80)
print()

print(f"Target: {target_time_kst} KST ({target_time_utc} UTC)")
print()
print("Signal Predictions:")
print(f"  Live Bot:      LONG=0.3707, SHORT=0.8122")
print(f"  Backtest:      LONG={long_prob_backtest:.4f}, SHORT={short_prob_backtest:.4f}")
print(f"  Production:    LONG={long_prob_production:.4f}, SHORT={short_prob_production:.4f}")
print()

# Check which method matches live better
if long_prob_production is not None and short_prob_production is not None:
    prod_long_diff = abs(long_prob_production - 0.3707)
    prod_short_diff = abs(short_prob_production - 0.8122)
    backtest_long_diff = abs(long_prob_backtest - 0.3707)
    backtest_short_diff = abs(short_prob_backtest - 0.8122)

    print("Match with Live Bot:")
    print(f"  Production Method:  LONG error={prod_long_diff:.4f}, SHORT error={prod_short_diff:.4f}")
    print(f"  Backtest Method:    LONG error={backtest_long_diff:.4f}, SHORT error={backtest_short_diff:.4f}")
    print()

    if (prod_long_diff + prod_short_diff) < (backtest_long_diff + backtest_short_diff):
        print("✅ Production Method matches live bot better")
    else:
        print("⚠️ Backtest Method matches live bot better")

print()
print(f"Top LONG feature differences: {len(long_diffs_df)} features analyzed")
print(f"Top SHORT feature differences: {len(short_diffs_df)} features analyzed")
print()
print("Feature comparison complete!")
