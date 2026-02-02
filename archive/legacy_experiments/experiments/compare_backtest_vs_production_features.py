"""
Compare feature values: Backtest vs Production
Find why production shows higher LONG probabilities
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
print("BACKTEST vs PRODUCTION FEATURE COMPARISON")
print("="*80)

# ============================================================================
# METHOD 1: BACKTEST WAY (Full dataset features)
# ============================================================================
print("\n" + "="*80)
print("METHOD 1: BACKTEST WAY")
print("="*80)

# Load full data
data_file = DATA_DIR / "BTCUSDT_5m_max.csv"
df_full = pd.read_csv(data_file)
print(f"\n✅ Loaded {len(df_full):,} candles")
print(f"   Date range: {df_full['timestamp'].iloc[0]} to {df_full['timestamp'].iloc[-1]}")

# Calculate features on FULL dataset
print("\nCalculating features on FULL dataset...")
df_backtest = calculate_all_features(df_full.copy())
df_backtest = prepare_exit_features(df_backtest)
print(f"✅ Features calculated ({len(df_backtest)} rows)")

# ============================================================================
# METHOD 2: PRODUCTION WAY (Last 1000 candles only)
# ============================================================================
print("\n" + "="*80)
print("METHOD 2: PRODUCTION WAY")
print("="*80)

# Take only last 1000 candles (simulating production)
df_last_1000 = df_full.tail(1000).copy()
df_last_1000 = df_last_1000.reset_index(drop=True)
print(f"\n✅ Extracted last 1000 candles")
print(f"   Date range: {df_last_1000['timestamp'].iloc[0]} to {df_last_1000['timestamp'].iloc[-1]}")

# Calculate features on ONLY these 1000 candles
print("\nCalculating features on LAST 1000 ONLY...")
df_production = calculate_all_features(df_last_1000.copy())
df_production = prepare_exit_features(df_production)
print(f"✅ Features calculated ({len(df_production)} rows)")

# ============================================================================
# LOAD MODEL AND COMPARE PREDICTIONS
# ============================================================================
print("\n" + "="*80)
print("LOADING MODEL")
print("="*80)

long_model_path = MODELS_DIR / "xgboost_long_trade_outcome_full_20251018_233146.pkl"
long_scaler_path = MODELS_DIR / "xgboost_long_trade_outcome_full_20251018_233146_scaler.pkl"
long_features_path = MODELS_DIR / "xgboost_long_trade_outcome_full_20251018_233146_features.txt"

with open(long_model_path, 'rb') as f:
    long_model = pickle.load(f)
long_scaler = joblib.load(long_scaler_path)
with open(long_features_path, 'r') as f:
    long_features = [line.strip() for line in f.readlines() if line.strip()]

print(f"✅ Model loaded: {len(long_features)} features")

# ============================================================================
# COMPARE: Same time points, different feature calculations
# ============================================================================
print("\n" + "="*80)
print("COMPARISON: LAST 100 POINTS")
print("="*80)

# Get last 100 points from backtest method
backtest_last_100 = df_backtest.tail(100)

# Get last 100 points from production method
production_last_100 = df_production.tail(100)

print(f"\nBacktest timestamps: {backtest_last_100['timestamp'].iloc[0]} to {backtest_last_100['timestamp'].iloc[-1]}")
print(f"Production timestamps: {production_last_100['timestamp'].iloc[0]} to {production_last_100['timestamp'].iloc[-1]}")

# Calculate probabilities for both
backtest_probs = []
production_probs = []

for i in range(len(backtest_last_100)):
    # Backtest
    try:
        feat = backtest_last_100[long_features].iloc[i:i+1].values
        scaled = long_scaler.transform(feat)
        prob = long_model.predict_proba(scaled)[0][1]
        backtest_probs.append(prob)
    except:
        backtest_probs.append(np.nan)

    # Production
    try:
        feat = production_last_100[long_features].iloc[i:i+1].values
        scaled = long_scaler.transform(feat)
        prob = long_model.predict_proba(scaled)[0][1]
        production_probs.append(prob)
    except:
        production_probs.append(np.nan)

print(f"\n📊 PROBABILITY COMPARISON (Last 100 points):")
print(f"\nBacktest Method:")
print(f"  Mean: {np.nanmean(backtest_probs):.4f} ({np.nanmean(backtest_probs)*100:.2f}%)")
print(f"  Median: {np.nanmedian(backtest_probs):.4f} ({np.nanmedian(backtest_probs)*100:.2f}%)")
print(f"  Min: {np.nanmin(backtest_probs):.4f} ({np.nanmin(backtest_probs)*100:.2f}%)")
print(f"  Max: {np.nanmax(backtest_probs):.4f} ({np.nanmax(backtest_probs)*100:.2f}%)")

print(f"\nProduction Method:")
print(f"  Mean: {np.nanmean(production_probs):.4f} ({np.nanmean(production_probs)*100:.2f}%)")
print(f"  Median: {np.nanmedian(production_probs):.4f} ({np.nanmedian(production_probs)*100:.2f}%)")
print(f"  Min: {np.nanmin(production_probs):.4f} ({np.nanmin(production_probs)*100:.2f}%)")
print(f"  Max: {np.nanmax(production_probs):.4f} ({np.nanmax(production_probs)*100:.2f}%)")

print(f"\n⚠️ DIFFERENCE:")
print(f"  Mean difference: {(np.nanmean(production_probs) - np.nanmean(backtest_probs))*100:+.2f}%")
print(f"  Median difference: {(np.nanmedian(production_probs) - np.nanmedian(backtest_probs))*100:+.2f}%")

# ============================================================================
# FEATURE VALUE COMPARISON
# ============================================================================
print("\n" + "="*80)
print("FEATURE VALUE COMPARISON (Last point)")
print("="*80)

# Compare feature values for the very last point
backtest_features = backtest_last_100[long_features].iloc[-1]
production_features = production_last_100[long_features].iloc[-1]

# Find features with biggest differences
differences = []
for feat in long_features:
    b_val = backtest_features[feat]
    p_val = production_features[feat]
    diff = p_val - b_val
    pct_diff = (diff / b_val * 100) if b_val != 0 else 0
    differences.append({
        'feature': feat,
        'backtest': b_val,
        'production': p_val,
        'diff': diff,
        'pct_diff': pct_diff
    })

diff_df = pd.DataFrame(differences)
diff_df['abs_pct_diff'] = diff_df['pct_diff'].abs()
diff_df = diff_df.sort_values('abs_pct_diff', ascending=False)

print("\n🔍 Top 15 Features with Biggest Differences:")
print(diff_df.head(15)[['feature', 'backtest', 'production', 'pct_diff']].to_string(index=False))

# ============================================================================
# CHECK SPECIFIC PRODUCTION TIMESTAMPS
# ============================================================================
print("\n" + "="*80)
print("CHECK: PRODUCTION TIMESTAMPS (Oct 19, 05:40-06:40)")
print("="*80)

# Find these timestamps in data
target_times = [
    "2025-10-18 20:40:00",  # 05:40 KST
    "2025-10-18 20:45:00",
    "2025-10-18 20:50:00",
    "2025-10-18 21:40:00",  # 06:40 KST
]

print("\nSearching for production timestamps in data...")
for target in target_times:
    # In backtest data
    backtest_row = df_backtest[df_backtest['timestamp'] == target]
    if len(backtest_row) > 0:
        try:
            feat = backtest_row[long_features].iloc[0:1].values
            scaled = long_scaler.transform(feat)
            prob = long_model.predict_proba(scaled)[0][1]
            print(f"\n{target} (Backtest):")
            print(f"  LONG prob: {prob:.4f} ({prob*100:.2f}%)")
        except Exception as e:
            print(f"\n{target} (Backtest): Error - {e}")

    # In production data
    production_row = df_production[df_production['timestamp'] == target]
    if len(production_row) > 0:
        try:
            feat = production_row[long_features].iloc[0:1].values
            scaled = long_scaler.transform(feat)
            prob = long_model.predict_proba(scaled)[0][1]
            print(f"{target} (Production):")
            print(f"  LONG prob: {prob:.4f} ({prob*100:.2f}%)")
        except Exception as e:
            print(f"{target} (Production): Error - {e}")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
