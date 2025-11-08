"""
Diagnose Entry Model Probability Distributions
==============================================

Purpose: Check what probabilities the Production Entry models output
to understand why no trades were executed at threshold 0.75
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import pickle

print("="*80)
print("ENTRY MODEL PROBABILITY DISTRIBUTION ANALYSIS")
print("="*80)
print()

# Load models
MODELS_DIR = PROJECT_ROOT / "models"

print("Loading Entry models...")
with open(MODELS_DIR / "xgboost_long_entry_walkforward_decoupled_20251027_194313.pkl", 'rb') as f:
    long_entry_model = pickle.load(f)
with open(MODELS_DIR / "xgboost_short_entry_walkforward_decoupled_20251027_194313.pkl", 'rb') as f:
    short_entry_model = pickle.load(f)

with open(MODELS_DIR / "xgboost_long_entry_walkforward_decoupled_20251027_194313_scaler.pkl", 'rb') as f:
    long_scaler = pickle.load(f)
with open(MODELS_DIR / "xgboost_short_entry_walkforward_decoupled_20251027_194313_scaler.pkl", 'rb') as f:
    short_scaler = pickle.load(f)

with open(MODELS_DIR / "xgboost_long_entry_walkforward_decoupled_20251027_194313_features.pkl", 'rb') as f:
    long_entry_features = pickle.load(f)
with open(MODELS_DIR / "xgboost_short_entry_walkforward_decoupled_20251027_194313_features.pkl", 'rb') as f:
    short_entry_features = pickle.load(f)

print(f"✅ LONG Entry: {len(long_entry_features)} features")
print(f"✅ SHORT Entry: {len(short_entry_features)} features")
print()

# Load data
DATA_DIR = PROJECT_ROOT / "data" / "features"
print("Loading data...")
features_file = DATA_DIR / "BTCUSDT_5m_features.csv"
df = pd.read_csv(features_file)
print(f"✅ Loaded {len(df)} candles")
print()

# Sample 10,000 candles for analysis
sample_size = min(10000, len(df))
sample_df = df.tail(sample_size).copy()

# Extract features
long_features_df = sample_df[long_entry_features].fillna(0)
short_features_df = sample_df[short_entry_features].fillna(0)

# Scale
long_features_scaled = long_scaler.transform(long_features_df.values)
short_features_scaled = short_scaler.transform(short_features_df.values)

# Get probabilities
long_probs = long_entry_model.predict_proba(long_features_scaled)[:, 1]
short_probs = short_entry_model.predict_proba(short_features_scaled)[:, 1]

print("="*80)
print("LONG ENTRY MODEL - Probability Distribution")
print("="*80)
print(f"Mean:   {long_probs.mean():.4f}")
print(f"Median: {np.median(long_probs):.4f}")
print(f"Std:    {long_probs.std():.4f}")
print(f"Min:    {long_probs.min():.4f}")
print(f"Max:    {long_probs.max():.4f}")
print()
print("Threshold Analysis:")
for threshold in [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.75, 0.80]:
    pct = (long_probs >= threshold).sum() / len(long_probs) * 100
    print(f"  ≥{threshold:.2f}: {pct:6.2f}% of candles")
print()

print("="*80)
print("SHORT ENTRY MODEL - Probability Distribution")
print("="*80)
print(f"Mean:   {short_probs.mean():.4f}")
print(f"Median: {np.median(short_probs):.4f}")
print(f"Std:    {short_probs.std():.4f}")
print(f"Min:    {short_probs.min():.4f}")
print(f"Max:    {short_probs.max():.4f}")
print()
print("Threshold Analysis:")
for threshold in [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.75, 0.80]:
    pct = (short_probs >= threshold).sum() / len(short_probs) * 100
    print(f"  ≥{threshold:.2f}: {pct:6.2f}% of candles")
print()

# Recommended thresholds
print("="*80)
print("RECOMMENDATIONS")
print("="*80)
print()

# Find thresholds that achieve 10-20% entry rate (reasonable trade frequency)
long_threshold_10pct = np.percentile(long_probs, 90)
long_threshold_15pct = np.percentile(long_probs, 85)
long_threshold_20pct = np.percentile(long_probs, 80)

short_threshold_10pct = np.percentile(short_probs, 90)
short_threshold_15pct = np.percentile(short_probs, 85)
short_threshold_20pct = np.percentile(short_probs, 80)

print("LONG Entry - Threshold for Target Entry Rate:")
print(f"  10% entry rate: threshold {long_threshold_10pct:.3f}")
print(f"  15% entry rate: threshold {long_threshold_15pct:.3f}")
print(f"  20% entry rate: threshold {long_threshold_20pct:.3f}")
print()
print("SHORT Entry - Threshold for Target Entry Rate:")
print(f"  10% entry rate: threshold {short_threshold_10pct:.3f}")
print(f"  15% entry rate: threshold {short_threshold_15pct:.3f}")
print(f"  20% entry rate: threshold {short_threshold_20pct:.3f}")
print()

# Compare current threshold
current_threshold = 0.75
long_entry_pct = (long_probs >= current_threshold).sum() / len(long_probs) * 100
short_entry_pct = (short_probs >= current_threshold).sum() / len(short_probs) * 100

print(f"🚨 ISSUE IDENTIFIED:")
print(f"  Current threshold: {current_threshold}")
print(f"  LONG entry rate: {long_entry_pct:.2f}% (too low!)")
print(f"  SHORT entry rate: {short_entry_pct:.2f}% (too low!)")
print()
print("💡 SOLUTION:")
if long_entry_pct < 5 and short_entry_pct < 5:
    print("  ❌ Threshold 0.75 is TOO HIGH for these models")
    print("  ✅ RECOMMENDED: Test thresholds 0.10-0.30 for reasonable trade frequency")
    print()
    print("  Suggested Entry Thresholds:")
    print(f"    LONG:  {long_threshold_15pct:.3f} (for 15% entry rate)")
    print(f"    SHORT: {short_threshold_15pct:.3f} (for 15% entry rate)")
