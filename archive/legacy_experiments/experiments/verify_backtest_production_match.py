"""
Verify that production method matches backtest method
Using the same models (reverted to backtest models)
"""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import pickle
import joblib

from scripts.experiments.calculate_all_features import calculate_all_features
from scripts.experiments.retrain_exit_models_opportunity_gating import prepare_exit_features

DATA_DIR = PROJECT_ROOT / "data" / "historical"
MODELS_DIR = PROJECT_ROOT / "models"

print("="*80)
print("VERIFY: BACKTEST vs PRODUCTION METHOD MATCH")
print("="*80)

# ============================================================================
# STEP 1: Load backtest models (same as production now)
# ============================================================================
print("\n📁 Loading backtest models...")

# LONG Entry
long_model_path = MODELS_DIR / "xgboost_v4_phase4_advanced_lookahead3_thresh0.pkl"
with open(long_model_path, 'rb') as f:
    long_model = pickle.load(f)

long_scaler_path = MODELS_DIR / "xgboost_v4_phase4_advanced_lookahead3_thresh0_scaler.pkl"
with open(long_scaler_path, 'rb') as f:
    long_scaler = pickle.load(f)

long_features_path = MODELS_DIR / "xgboost_v4_phase4_advanced_lookahead3_thresh0_features.txt"
with open(long_features_path, 'r') as f:
    long_feature_names = [line.strip() for line in f.readlines()]

print(f"  LONG model: {long_model_path.name}")
print(f"  LONG features: {len(long_feature_names)}")
print(f"  LONG scaler: {type(long_scaler).__name__}")

# SHORT Entry
short_model_path = MODELS_DIR / "xgboost_short_redesigned_20251016_233322.pkl"
with open(short_model_path, 'rb') as f:
    short_model = pickle.load(f)

short_scaler_path = MODELS_DIR / "xgboost_short_redesigned_20251016_233322_scaler.pkl"
with open(short_scaler_path, 'rb') as f:
    short_scaler = pickle.load(f)

short_features_path = MODELS_DIR / "xgboost_short_redesigned_20251016_233322_features.txt"
with open(short_features_path, 'r') as f:
    short_feature_names = [line.strip() for line in f.readlines()]

print(f"  SHORT model: {short_model_path.name}")
print(f"  SHORT features: {len(short_feature_names)}")
print(f"  SHORT scaler: {type(short_scaler).__name__}")

# ============================================================================
# STEP 2: Load data and calculate features
# ============================================================================
print("\n📊 Loading data...")
data_file = DATA_DIR / "BTCUSDT_5m_max.csv"
df = pd.read_csv(data_file)
print(f"  Total candles: {len(df):,}")
print(f"  Date range: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")

print("\n🔧 Calculating features (BACKTEST METHOD)...")
df_backtest = calculate_all_features(df.copy())
df_backtest = prepare_exit_features(df_backtest)
print(f"  ✅ Features calculated: {len(df_backtest):,} candles")

# Also simulate production method (last 1440 candles)
print("\n🔧 Calculating features (PRODUCTION METHOD - last 1440)...")
df_production = df.tail(1440).copy()
df_production = calculate_all_features(df_production)
df_production = prepare_exit_features(df_production)
print(f"  ✅ Features calculated: {len(df_production):,} candles")

# ============================================================================
# STEP 3: Test predictions on random sample of timestamps
# ============================================================================
print("\n🎯 Testing predictions on random sample...")

# Select 50 random timestamps from the dataset
np.random.seed(42)
sample_indices = np.random.choice(len(df_backtest), size=min(50, len(df_backtest)), replace=False)
sample_timestamps = df_backtest.iloc[sample_indices]['timestamp'].values

results = []
for ts in sample_timestamps:
    # Backtest method
    row_bt = df_backtest[df_backtest['timestamp'] == ts]
    if len(row_bt) == 0:
        continue

    # LONG prediction - backtest
    long_feat_bt = row_bt[long_feature_names].values
    long_scaled_bt = long_scaler.transform(long_feat_bt)
    long_pred_bt = long_model.predict_proba(long_scaled_bt)[0][1]

    # SHORT prediction - backtest
    short_feat_bt = row_bt[short_feature_names].values
    short_scaled_bt = short_scaler.transform(short_feat_bt)
    short_pred_bt = short_model.predict_proba(short_scaled_bt)[0][1]

    # Production method (if in last 1440)
    row_prod = df_production[df_production['timestamp'] == ts]
    if len(row_prod) > 0:
        # LONG prediction - production
        long_feat_prod = row_prod[long_feature_names].values
        long_scaled_prod = long_scaler.transform(long_feat_prod)
        long_pred_prod = long_model.predict_proba(long_scaled_prod)[0][1]

        # SHORT prediction - production
        short_feat_prod = row_prod[short_feature_names].values
        short_scaled_prod = short_scaler.transform(short_feat_prod)
        short_pred_prod = short_model.predict_proba(short_scaled_prod)[0][1]

        # Compare
        long_diff = abs(long_pred_bt - long_pred_prod)
        short_diff = abs(short_pred_bt - short_pred_prod)
    else:
        long_pred_prod = None
        short_pred_prod = None
        long_diff = None
        short_diff = None

    results.append({
        'timestamp': ts,
        'long_backtest': long_pred_bt,
        'long_production': long_pred_prod,
        'long_diff': long_diff,
        'short_backtest': short_pred_bt,
        'short_production': short_pred_prod,
        'short_diff': short_diff,
        'price': row_bt['close'].iloc[0]
    })

df_results = pd.DataFrame(results)

# Filter for timestamps in both datasets
df_both = df_results[df_results['long_production'].notna()].copy()

print(f"\n📊 RESULTS ({len(df_both)} timestamps in both datasets):")
print("="*80)

if len(df_both) == 0:
    print("⚠️  No overlapping timestamps found")
else:
    print(f"\n📈 LONG Probability:")
    print(f"  Backtest mean: {df_both['long_backtest'].mean():.4f} ({df_both['long_backtest'].mean()*100:.2f}%)")
    print(f"  Production mean: {df_both['long_production'].mean():.4f} ({df_both['long_production'].mean()*100:.2f}%)")
    print(f"  Mean difference: {df_both['long_diff'].mean():.6f} ({df_both['long_diff'].mean()*100:.4f}%)")
    print(f"  Max difference: {df_both['long_diff'].max():.6f} ({df_both['long_diff'].max()*100:.4f}%)")

    print(f"\n📉 SHORT Probability:")
    print(f"  Backtest mean: {df_both['short_backtest'].mean():.4f} ({df_both['short_backtest'].mean()*100:.2f}%)")
    print(f"  Production mean: {df_both['short_production'].mean():.4f} ({df_both['short_production'].mean()*100:.2f}%)")
    print(f"  Mean difference: {df_both['short_diff'].mean():.6f} ({df_both['short_diff'].mean()*100:.4f}%)")
    print(f"  Max difference: {df_both['short_diff'].max():.6f} ({df_both['short_diff'].max()*100:.4f}%)")

    print(f"\n📋 Sample (first 20):")
    print(f"{'Timestamp':<20} {'LONG BT%':<10} {'LONG PR%':<10} {'Diff%':<10} {'SHORT BT%':<10} {'SHORT PR%':<10} {'Diff%':<10}")
    print("-"*100)

    for idx, row in df_both.head(20).iterrows():
        print(f"{row['timestamp']:<20} {row['long_backtest']*100:<10.2f} {row['long_production']*100:<10.2f} "
              f"{row['long_diff']*100:<10.4f} {row['short_backtest']*100:<10.2f} {row['short_production']*100:<10.2f} "
              f"{row['short_diff']*100:<10.4f}")

    # Check if exact match
    exact_match_threshold = 1e-10
    long_exact = df_both[df_both['long_diff'] < exact_match_threshold]
    short_exact = df_both[df_both['short_diff'] < exact_match_threshold]

    print(f"\n✅ Exact matches (< {exact_match_threshold}):")
    print(f"  LONG: {len(long_exact)}/{len(df_both)} ({len(long_exact)/len(df_both)*100:.1f}%)")
    print(f"  SHORT: {len(short_exact)}/{len(df_both)} ({len(short_exact)/len(df_both)*100:.1f}%)")

    if len(long_exact) == len(df_both) and len(short_exact) == len(df_both):
        print(f"\n✅ PERFECT MATCH: Backtest and Production methods are IDENTICAL!")
        print(f"  → Production will reproduce backtest results")
    else:
        print(f"\n⚠️  Methods differ slightly")
        print(f"  → May be due to floating point precision or data differences")

# ============================================================================
# STEP 4: Test on full dataset - distribution check
# ============================================================================
print(f"\n📊 Full Dataset Distribution Check:")

# Calculate all LONG probabilities
print(f"\nCalculating LONG probabilities for all {len(df_backtest)} candles...")
all_long_probs = []
for idx in range(len(df_backtest)):
    row = df_backtest.iloc[idx:idx+1]
    long_feat = row[long_feature_names].values
    long_scaled = long_scaler.transform(long_feat)
    long_prob = long_model.predict_proba(long_scaled)[0][1]
    all_long_probs.append(long_prob)

all_long_probs = np.array(all_long_probs)

print(f"\n📈 LONG Probability Distribution (Full Dataset):")
print(f"  Mean: {all_long_probs.mean():.4f} ({all_long_probs.mean()*100:.2f}%)")
print(f"  Median: {np.median(all_long_probs):.4f} ({np.median(all_long_probs)*100:.2f}%)")
print(f"  Min: {all_long_probs.min():.4f} ({all_long_probs.min()*100:.2f}%)")
print(f"  Max: {all_long_probs.max():.4f} ({all_long_probs.max()*100:.2f}%)")

print(f"\n  Distribution:")
print(f"    ≥90%: {(all_long_probs >= 0.9).sum():5d} ({(all_long_probs >= 0.9).sum()/len(all_long_probs)*100:5.1f}%)")
print(f"    ≥80%: {(all_long_probs >= 0.8).sum():5d} ({(all_long_probs >= 0.8).sum()/len(all_long_probs)*100:5.1f}%)")
print(f"    ≥70%: {(all_long_probs >= 0.7).sum():5d} ({(all_long_probs >= 0.7).sum()/len(all_long_probs)*100:5.1f}%)")
print(f"    ≥65%: {(all_long_probs >= 0.65).sum():5d} ({(all_long_probs >= 0.65).sum()/len(all_long_probs)*100:5.1f}%)")
print(f"    <65%: {(all_long_probs < 0.65).sum():5d} ({(all_long_probs < 0.65).sum()/len(all_long_probs)*100:5.1f}%)")

print(f"\n🎯 Expected Entry Rate (≥65% threshold):")
print(f"  LONG entries: {(all_long_probs >= 0.65).sum()/len(all_long_probs)*100:.2f}%")
print(f"  Expected trades per 5-day window (1440 candles): {(all_long_probs >= 0.65).sum()/len(all_long_probs)*1440:.1f}")

print("\n" + "="*80)
print("VERIFICATION COMPLETE")
print("="*80)
