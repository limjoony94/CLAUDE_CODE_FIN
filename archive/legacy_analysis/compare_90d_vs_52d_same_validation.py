"""
Fair Comparison: 90-Day vs 52-Day Models on Same Validation Period
====================================================================

Purpose: Test both model sets on IDENTICAL validation period for fair comparison

Current Issue:
  - 52-day models validated on Sep 29 - Oct 26 → SUCCESS
  - 90-day models validated on Oct 9 - Nov 6 → PARTIAL (SHORT failed)
  - Different validation periods = unfair comparison

Solution:
  - Test 90-day models on Sep 29 - Oct 26 (same as 52-day)
  - Compare results on IDENTICAL period
  - Determine if 90-day truly better or validation period effect

Models:
  52-Day: xgboost_{long|short}_entry_52day_20251106_140955.pkl
  90-Day: xgboost_{long|short}_entry_90days_tradeoutcome_20251106_193900.pkl

Created: 2025-11-06 20:50 KST
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import pickle
import joblib

# Paths
DATA_DIR = PROJECT_ROOT / "data" / "features"
LABELS_DIR = PROJECT_ROOT / "data" / "labels"
MODELS_DIR = PROJECT_ROOT / "models"

# Input
FEATURES_90D = DATA_DIR / "BTCUSDT_5m_features_90days_complete_20251106_164542.csv"
LABELS_90D = LABELS_DIR / "trade_outcome_labels_90days_20251106_193715.csv"

# 52-Day Models
MODEL_52D_LONG = MODELS_DIR / "xgboost_long_entry_52day_20251106_140955.pkl"
MODEL_52D_SHORT = MODELS_DIR / "xgboost_short_entry_52day_20251106_140955.pkl"
SCALER_52D_LONG = MODELS_DIR / "xgboost_long_entry_52day_20251106_140955_scaler.pkl"
SCALER_52D_SHORT = MODELS_DIR / "xgboost_short_entry_52day_20251106_140955_scaler.pkl"
FEATURES_52D_LONG = MODELS_DIR / "xgboost_long_entry_52day_20251106_140955_features.txt"
FEATURES_52D_SHORT = MODELS_DIR / "xgboost_short_entry_52day_20251106_140955_features.txt"

# 90-Day Models
MODEL_90D_LONG = MODELS_DIR / "xgboost_long_entry_90days_tradeoutcome_20251106_193900.pkl"
MODEL_90D_SHORT = MODELS_DIR / "xgboost_short_entry_90days_tradeoutcome_20251106_193900.pkl"
SCALER_90D_LONG = MODELS_DIR / "xgboost_long_entry_90days_tradeoutcome_20251106_193900_scaler.pkl"
SCALER_90D_SHORT = MODELS_DIR / "xgboost_short_entry_90days_tradeoutcome_20251106_193900_scaler.pkl"
FEATURES_90D_LONG = MODELS_DIR / "xgboost_long_entry_90days_tradeoutcome_20251106_193900_features.txt"
FEATURES_90D_SHORT = MODELS_DIR / "xgboost_short_entry_90days_tradeoutcome_20251106_193900_features.txt"

# Validation period (SAME for both models)
VAL_START = "2025-09-29"
VAL_END = "2025-10-26"

print("=" * 80)
print("FAIR COMPARISON: 90-DAY vs 52-DAY MODELS")
print("=" * 80)
print()
print(f"🎯 Goal: Test both on IDENTICAL validation period")
print(f"📅 Validation: {VAL_START} to {VAL_END} (28 days)")
print()

# ============================================================================
# LOAD DATA
# ============================================================================

print("=" * 80)
print("LOADING DATA")
print("=" * 80)
print()

print(f"📖 Loading features: {FEATURES_90D.name}")
df_features = pd.read_csv(FEATURES_90D)
df_features['timestamp'] = pd.to_datetime(df_features['timestamp'])

print(f"📖 Loading labels: {LABELS_90D.name}")
df_labels = pd.read_csv(LABELS_90D)
df_labels['timestamp'] = pd.to_datetime(df_labels['timestamp'])

# Merge
df = df_features.merge(
    df_labels[['timestamp', 'signal_long', 'signal_short']],
    on='timestamp',
    how='inner'
)

# Filter to validation period
df_val = df[(df['timestamp'] >= VAL_START) & (df['timestamp'] <= VAL_END)].copy()

print(f"✅ Validation data loaded:")
print(f"   Period: {df_val['timestamp'].min()} to {df_val['timestamp'].max()}")
print(f"   Rows: {len(df_val):,}")
print(f"   Days: {(df_val['timestamp'].max() - df_val['timestamp'].min()).days}")
print(f"   LONG labels: {df_val['signal_long'].sum()} ({df_val['signal_long'].mean()*100:.2f}%)")
print(f"   SHORT labels: {df_val['signal_short'].sum()} ({df_val['signal_short'].mean()*100:.2f}%)")
print()

# ============================================================================
# LOAD 52-DAY MODELS
# ============================================================================

print("=" * 80)
print("LOADING 52-DAY MODELS")
print("=" * 80)
print()

with open(MODEL_52D_LONG, 'rb') as f:
    model_52d_long = pickle.load(f)
with open(MODEL_52D_SHORT, 'rb') as f:
    model_52d_short = pickle.load(f)
scaler_52d_long = joblib.load(SCALER_52D_LONG)
scaler_52d_short = joblib.load(SCALER_52D_SHORT)

with open(FEATURES_52D_LONG, 'r') as f:
    features_52d_long = [line.strip() for line in f]
with open(FEATURES_52D_SHORT, 'r') as f:
    features_52d_short = [line.strip() for line in f]

print(f"✅ 52-Day models loaded:")
print(f"   LONG Entry: {len(features_52d_long)} features")
print(f"   SHORT Entry: {len(features_52d_short)} features")
print()

# ============================================================================
# LOAD 90-DAY MODELS
# ============================================================================

print("=" * 80)
print("LOADING 90-DAY MODELS")
print("=" * 80)
print()

with open(MODEL_90D_LONG, 'rb') as f:
    model_90d_long = pickle.load(f)
with open(MODEL_90D_SHORT, 'rb') as f:
    model_90d_short = pickle.load(f)
scaler_90d_long = joblib.load(SCALER_90D_LONG)
scaler_90d_short = joblib.load(SCALER_90D_SHORT)

with open(FEATURES_90D_LONG, 'r') as f:
    features_90d_long = [line.strip() for line in f]
with open(FEATURES_90D_SHORT, 'r') as f:
    features_90d_short = [line.strip() for line in f]

print(f"✅ 90-Day models loaded:")
print(f"   LONG Entry: {len(features_90d_long)} features")
print(f"   SHORT Entry: {len(features_90d_short)} features")
print()

# ============================================================================
# VALIDATE 52-DAY MODELS
# ============================================================================

print("=" * 80)
print("VALIDATING 52-DAY MODELS (Sep 29 - Oct 26)")
print("=" * 80)
print()

# LONG
X_val_long_52d = df_val[features_52d_long].fillna(0)
X_val_long_52d_scaled = scaler_52d_long.transform(X_val_long_52d)
probs_52d_long = model_52d_long.predict_proba(X_val_long_52d_scaled)[:, 1]

print(f"🔵 52-Day LONG Entry:")
print(f"   Min: {probs_52d_long.min():.4f} ({probs_52d_long.min()*100:.2f}%)")
print(f"   Max: {probs_52d_long.max():.4f} ({probs_52d_long.max()*100:.2f}%)")
print(f"   Mean: {probs_52d_long.mean():.4f} ({probs_52d_long.mean()*100:.2f}%)")
print(f"   >= 0.85: {(probs_52d_long >= 0.85).sum()} ({(probs_52d_long >= 0.85).sum()/len(probs_52d_long)*100:.2f}%)")
print()

# SHORT
X_val_short_52d = df_val[features_52d_short].fillna(0)
X_val_short_52d_scaled = scaler_52d_short.transform(X_val_short_52d)
probs_52d_short = model_52d_short.predict_proba(X_val_short_52d_scaled)[:, 1]

print(f"🔴 52-Day SHORT Entry:")
print(f"   Min: {probs_52d_short.min():.4f} ({probs_52d_short.min()*100:.2f}%)")
print(f"   Max: {probs_52d_short.max():.4f} ({probs_52d_short.max()*100:.2f}%)")
print(f"   Mean: {probs_52d_short.mean():.4f} ({probs_52d_short.mean()*100:.2f}%)")
print(f"   >= 0.80: {(probs_52d_short >= 0.80).sum()} ({(probs_52d_short >= 0.80).sum()/len(probs_52d_short)*100:.2f}%)")
print()

# ============================================================================
# VALIDATE 90-DAY MODELS
# ============================================================================

print("=" * 80)
print("VALIDATING 90-DAY MODELS (Sep 29 - Oct 26)")
print("=" * 80)
print()

# LONG
X_val_long_90d = df_val[features_90d_long].fillna(0)
X_val_long_90d_scaled = scaler_90d_long.transform(X_val_long_90d)
probs_90d_long = model_90d_long.predict_proba(X_val_long_90d_scaled)[:, 1]

print(f"🔵 90-Day LONG Entry:")
print(f"   Min: {probs_90d_long.min():.4f} ({probs_90d_long.min()*100:.2f}%)")
print(f"   Max: {probs_90d_long.max():.4f} ({probs_90d_long.max()*100:.2f}%)")
print(f"   Mean: {probs_90d_long.mean():.4f} ({probs_90d_long.mean()*100:.2f}%)")
print(f"   >= 0.85: {(probs_90d_long >= 0.85).sum()} ({(probs_90d_long >= 0.85).sum()/len(probs_90d_long)*100:.2f}%)")
print()

# SHORT
X_val_short_90d = df_val[features_90d_short].fillna(0)
X_val_short_90d_scaled = scaler_90d_short.transform(X_val_short_90d)
probs_90d_short = model_90d_short.predict_proba(X_val_short_90d_scaled)[:, 1]

print(f"🔴 90-Day SHORT Entry:")
print(f"   Min: {probs_90d_short.min():.4f} ({probs_90d_short.min()*100:.2f}%)")
print(f"   Max: {probs_90d_short.max():.4f} ({probs_90d_short.max()*100:.2f}%)")
print(f"   Mean: {probs_90d_short.mean():.4f} ({probs_90d_short.mean()*100:.2f}%)")
print(f"   >= 0.80: {(probs_90d_short >= 0.80).sum()} ({(probs_90d_short >= 0.80).sum()/len(probs_90d_short)*100:.2f}%)")
print()

# ============================================================================
# COMPARISON
# ============================================================================

print("=" * 80)
print("COMPARISON: 52-DAY vs 90-DAY (IDENTICAL VALIDATION PERIOD)")
print("=" * 80)
print()

print(f"📊 Validation Period: Sep 29 - Oct 26, 2025 (28 days, {len(df_val):,} candles)")
print()

print(f"🔵 LONG Entry Maximum Probability:")
print(f"   52-Day: {probs_52d_long.max()*100:.2f}% (threshold: 85%)")
print(f"   90-Day: {probs_90d_long.max()*100:.2f}% (threshold: 85%)")
print(f"   Difference: {(probs_90d_long.max() - probs_52d_long.max())*100:+.2f}%")
if probs_90d_long.max() > probs_52d_long.max():
    print(f"   Winner: 🏆 90-Day (+{(probs_90d_long.max() - probs_52d_long.max())*100:.2f}%)")
else:
    print(f"   Winner: 🏆 52-Day (+{(probs_52d_long.max() - probs_90d_long.max())*100:.2f}%)")
print()

print(f"🔴 SHORT Entry Maximum Probability:")
print(f"   52-Day: {probs_52d_short.max()*100:.2f}% (threshold: 80%)")
print(f"   90-Day: {probs_90d_short.max()*100:.2f}% (threshold: 80%)")
print(f"   Difference: {(probs_90d_short.max() - probs_52d_short.max())*100:+.2f}%")
if probs_90d_short.max() > probs_52d_short.max():
    print(f"   Winner: 🏆 90-Day (+{(probs_90d_short.max() - probs_52d_short.max())*100:.2f}%)")
else:
    print(f"   Winner: 🏆 52-Day (+{(probs_52d_short.max() - probs_90d_short.max())*100:.2f}%)")
print()

# Threshold coverage
long_52d_signals = (probs_52d_long >= 0.85).sum()
long_90d_signals = (probs_90d_long >= 0.85).sum()
short_52d_signals = (probs_52d_short >= 0.80).sum()
short_90d_signals = (probs_90d_short >= 0.80).sum()

print(f"📈 Signal Generation (reaching production thresholds):")
print(f"   LONG (>= 0.85):")
print(f"     52-Day: {long_52d_signals} signals ({long_52d_signals/len(probs_52d_long)*100:.2f}%)")
print(f"     90-Day: {long_90d_signals} signals ({long_90d_signals/len(probs_90d_long)*100:.2f}%)")
if long_90d_signals > long_52d_signals:
    print(f"     Winner: 🏆 90-Day (+{long_90d_signals - long_52d_signals} signals)")
elif long_52d_signals > long_90d_signals:
    print(f"     Winner: 🏆 52-Day (+{long_52d_signals - long_90d_signals} signals)")
else:
    print(f"     Tie: Both generate {long_52d_signals} signals")
print()

print(f"   SHORT (>= 0.80):")
print(f"     52-Day: {short_52d_signals} signals ({short_52d_signals/len(probs_52d_short)*100:.2f}%)")
print(f"     90-Day: {short_90d_signals} signals ({short_90d_signals/len(probs_90d_short)*100:.2f}%)")
if short_90d_signals > short_52d_signals:
    print(f"     Winner: 🏆 90-Day (+{short_90d_signals - short_52d_signals} signals)")
elif short_52d_signals > short_90d_signals:
    print(f"     Winner: 🏆 52-Day (+{short_52d_signals - short_90d_signals} signals)")
else:
    print(f"     Tie: Both generate {short_52d_signals} signals")
print()

# ============================================================================
# OVERALL WINNER
# ============================================================================

print("=" * 80)
print("OVERALL ASSESSMENT")
print("=" * 80)
print()

long_winner = "90-Day" if probs_90d_long.max() > probs_52d_long.max() else "52-Day"
short_winner = "90-Day" if probs_90d_short.max() > probs_52d_short.max() else "52-Day"

print(f"🏆 Winners on Same Validation Period:")
print(f"   LONG Entry: {long_winner}")
print(f"   SHORT Entry: {short_winner}")
print()

if long_winner == "90-Day" and short_winner == "90-Day":
    print(f"✅ CONCLUSION: 90-Day models are BETTER")
    print(f"   More training data → better calibration on validation")
    print(f"   Previous SHORT failure was due to different validation period")
    print()
    print(f"📋 RECOMMENDATION: Use 90-Day models")
elif long_winner == "52-Day" and short_winner == "52-Day":
    print(f"✅ CONCLUSION: 52-Day models are BETTER")
    print(f"   Recency > data quantity for this market")
    print(f"   90-Day includes irrelevant historical regimes")
    print()
    print(f"📋 RECOMMENDATION: Keep 52-Day models (current production)")
else:
    print(f"⚠️  CONCLUSION: MIXED RESULTS")
    print(f"   Each model set has strengths")
    print(f"   May need ensemble or hybrid approach")
    print()
    print(f"📋 RECOMMENDATION: Further analysis needed")

print()

# ============================================================================
# VALIDATION PERIOD EFFECT ANALYSIS
# ============================================================================

print("=" * 80)
print("VALIDATION PERIOD EFFECT ANALYSIS")
print("=" * 80)
print()

print(f"📊 Label Distribution Comparison:")
print(f"   Sep 29 - Oct 26 (tested above):")
print(f"     LONG: {df_val['signal_long'].mean()*100:.2f}%")
print(f"     SHORT: {df_val['signal_short'].mean()*100:.2f}%")
print()

# Check Oct 9 - Nov 6 period
df_val_oct = df[(df['timestamp'] >= "2025-10-09") & (df['timestamp'] <= "2025-11-06")].copy()
if len(df_val_oct) > 0:
    print(f"   Oct 9 - Nov 6 (90-day previous test):")
    print(f"     LONG: {df_val_oct['signal_long'].mean()*100:.2f}%")
    print(f"     SHORT: {df_val_oct['signal_short'].mean()*100:.2f}%")
    print()

    print(f"   Difference:")
    print(f"     LONG: {(df_val_oct['signal_long'].mean() - df_val['signal_long'].mean())*100:+.2f}%")
    print(f"     SHORT: {(df_val_oct['signal_short'].mean() - df_val['signal_short'].mean())*100:+.2f}%")
    print()

print(f"✅ Fair comparison complete!")
print()
