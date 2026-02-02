"""
Debug: Why are SHORT signals = 0?
==================================

Investigate why SHORT model produces no trades
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.production.train_xgboost_improved_v3_phase2 import calculate_features
from scripts.production.advanced_technical_features import AdvancedTechnicalFeatures

DATA_DIR = PROJECT_ROOT / "data" / "historical"
MODELS_DIR = PROJECT_ROOT / "models"

print("="*80)
print("DEBUG: SHORT SIGNAL ANALYSIS")
print("="*80 + "\n")

# Load models
print("Loading models...")
long_model_path = MODELS_DIR / "xgboost_v4_phase4_advanced_lookahead3_thresh0.pkl"
with open(long_model_path, 'rb') as f:
    long_model = pickle.load(f)

long_scaler_path = MODELS_DIR / "xgboost_v4_phase4_advanced_lookahead3_thresh0_scaler.pkl"
with open(long_scaler_path, 'rb') as f:
    long_scaler = pickle.load(f)

long_features_path = MODELS_DIR / "xgboost_v4_phase4_advanced_lookahead3_thresh0_features.txt"
with open(long_features_path, 'r') as f:
    long_feature_columns = [line.strip() for line in f.readlines()]

short_model_path = MODELS_DIR / "xgboost_short_redesigned_20251016_233322.pkl"
with open(short_model_path, 'rb') as f:
    short_model = pickle.load(f)

short_scaler_path = MODELS_DIR / "xgboost_short_redesigned_20251016_233322_scaler.pkl"
with open(short_scaler_path, 'rb') as f:
    short_scaler = pickle.load(f)

short_features_path = MODELS_DIR / "xgboost_short_redesigned_20251016_233322_features.txt"
with open(short_features_path, 'r') as f:
    short_feature_columns = [line.strip() for line in f.readlines()]

print(f"  ✅ Models loaded\n")

# Load data
print("Loading data...")
data_file = DATA_DIR / "BTCUSDT_5m_max.csv"
df_full = pd.read_csv(data_file)
df = df_full.tail(15000).reset_index(drop=True)
print(f"  ✅ Data loaded: {len(df)} candles\n")

print("Calculating features...")
df = calculate_features(df)
adv_features = AdvancedTechnicalFeatures(lookback_sr=50, lookback_trend=20)
df = adv_features.calculate_all_features(df)
df = df.ffill().bfill().fillna(0)
print(f"  ✅ Features calculated\n")

# Calculate signals
print("Calculating signals...")
long_probs = []
short_probs = []
missing_features = set()

for i in range(len(df)):
    try:
        long_feat = df[long_feature_columns].iloc[i:i+1].values
        long_feat_scaled = long_scaler.transform(long_feat)
        long_prob = long_model.predict_proba(long_feat_scaled)[0][1]
    except Exception as e:
        long_prob = 0.0
        if i == 0:
            print(f"  ⚠️  LONG model error: {e}")

    long_probs.append(long_prob)

    try:
        short_feat = df[short_feature_columns].iloc[i:i+1].values
        short_feat_scaled = short_scaler.transform(short_feat)
        short_prob = short_model.predict_proba(short_feat_scaled)[0][1]
    except KeyError as e:
        short_prob = 0.0
        # Track missing features
        missing_features.add(str(e))
        if i == 0:
            print(f"  ⚠️  SHORT model error: {e}")
    except Exception as e:
        short_prob = 0.0
        if i == 0:
            print(f"  ⚠️  SHORT model error: {e}")

    short_probs.append(short_prob)

df['long_prob'] = long_probs
df['short_prob'] = short_probs

if missing_features:
    print(f"\n❌ MISSING FEATURES DETECTED:")
    for feat in missing_features:
        print(f"  {feat}")
    print()

print(f"  ✅ Signals calculated\n")

# Analyze signal distribution
print("="*80)
print("SIGNAL DISTRIBUTION ANALYSIS")
print("="*80 + "\n")

print("LONG Signal Statistics:")
print(f"  Min: {df['long_prob'].min():.4f}")
print(f"  Max: {df['long_prob'].max():.4f}")
print(f"  Mean: {df['long_prob'].mean():.4f}")
print(f"  Median: {df['long_prob'].median():.4f}")
print(f"  Std: {df['long_prob'].std():.4f}")

print(f"\nLONG Signal Counts by Threshold:")
for thresh in [0.50, 0.60, 0.65, 0.70, 0.80]:
    count = (df['long_prob'] >= thresh).sum()
    pct = count / len(df) * 100
    print(f"  >= {thresh:.2f}: {count:,} ({pct:.2f}%)")

print(f"\nSHORT Signal Statistics:")
print(f"  Min: {df['short_prob'].min():.4f}")
print(f"  Max: {df['short_prob'].max():.4f}")
print(f"  Mean: {df['short_prob'].mean():.4f}")
print(f"  Median: {df['short_prob'].median():.4f}")
print(f"  Std: {df['short_prob'].std():.4f}")

print(f"\nSHORT Signal Counts by Threshold:")
for thresh in [0.50, 0.60, 0.65, 0.70, 0.80]:
    count = (df['short_prob'] >= thresh).sum()
    pct = count / len(df) * 100
    print(f"  >= {thresh:.2f}: {count:,} ({pct:.2f}%)")

# Check if SHORT is always 0
if df['short_prob'].max() == 0:
    print(f"\n🚨 CRITICAL: ALL SHORT probabilities are 0!")
    print(f"   This means SHORT model is NOT generating any signals.")
    print(f"   Possible causes:")
    print(f"     1. Missing features (check above)")
    print(f"     2. Model not working with current data")
    print(f"     3. Feature calculation error")

# Sample some high probability cases
print(f"\n" + "="*80)
print("SAMPLE HIGH PROBABILITY CASES")
print("="*80 + "\n")

print("Top 5 LONG signals:")
top_long = df.nlargest(5, 'long_prob')[['timestamp', 'long_prob', 'short_prob', 'close']]
print(top_long.to_string(index=False))

print(f"\nTop 5 SHORT signals:")
top_short = df.nlargest(5, 'short_prob')[['timestamp', 'long_prob', 'short_prob', 'close']]
print(top_short.to_string(index=False))

print(f"\n" + "="*80)
print("DEBUG COMPLETE")
print("="*80 + "\n")
