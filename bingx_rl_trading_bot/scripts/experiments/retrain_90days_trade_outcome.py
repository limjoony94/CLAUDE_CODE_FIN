"""
Retrain 90-Day Models with Trade Outcome Labels
===============================================

Purpose: Apply successful 52-day methodology to 90-day dataset

Key Changes from Previous Failed Attempts:
1. Use trade outcome labels (13.73% / 15.09%) instead of relaxed labels (1.41% / 2.50%)
2. Load feature names from 52-day models (avoid feature mismatch)
3. Enhanced 5-Fold CV with TimeSeriesSplit
4. Train on Aug 9 - Oct 8 (60 days)
5. Validate on Oct 9 - Nov 6 (28 days, 100% out-of-sample)

Expected Outcome: Well-calibrated models reaching production thresholds (0.85/0.80)

Created: 2025-11-06 20:40 KST
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import joblib
import warnings
warnings.filterwarnings('ignore')

# Paths
DATA_DIR = PROJECT_ROOT / "data" / "features"
LABELS_DIR = PROJECT_ROOT / "data" / "labels"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# Input files
FEATURES_90D = DATA_DIR / "BTCUSDT_5m_features_90days_complete_20251106_164542.csv"
LABELS_90D = LABELS_DIR / "trade_outcome_labels_90days_20251106_193715.csv"

# Reference feature lists (from successful 52-day models)
REF_LONG_ENTRY_FEATURES = MODELS_DIR / "xgboost_long_entry_52day_20251106_140955_features.txt"
REF_SHORT_ENTRY_FEATURES = MODELS_DIR / "xgboost_short_entry_52day_20251106_140955_features.txt"
REF_LONG_EXIT_FEATURES = MODELS_DIR / "xgboost_long_exit_52day_20251106_140955_features.txt"
REF_SHORT_EXIT_FEATURES = MODELS_DIR / "xgboost_short_exit_52day_20251106_140955_features.txt"

# Training configuration
TRAIN_START = "2025-08-09"
TRAIN_END = "2025-10-08"
VAL_START = "2025-10-09"
VAL_END = "2025-11-06"

N_FOLDS = 5
RANDOM_STATE = 42

# XGBoost parameters (Enhanced 5-Fold CV standard)
ENTRY_PARAMS = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'max_depth': 6,
    'learning_rate': 0.01,
    'n_estimators': 500,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 3,
    'gamma': 0.1,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'random_state': RANDOM_STATE,
    'n_jobs': -1
}

EXIT_PARAMS = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'max_depth': 4,
    'learning_rate': 0.05,
    'n_estimators': 300,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 1,
    'gamma': 0,
    'reg_alpha': 0.01,
    'reg_lambda': 0.1,
    'random_state': RANDOM_STATE,
    'n_jobs': -1
}

print("=" * 80)
print("RETRAIN 90-DAY MODELS WITH TRADE OUTCOME LABELS")
print("=" * 80)
print()

# ============================================================================
# LOAD FEATURE LISTS FROM 52-DAY MODELS
# ============================================================================

print("=" * 80)
print("LOADING FEATURE LISTS FROM 52-DAY MODELS")
print("=" * 80)
print()

with open(REF_LONG_ENTRY_FEATURES, 'r') as f:
    long_entry_features = [line.strip() for line in f]

with open(REF_SHORT_ENTRY_FEATURES, 'r') as f:
    short_entry_features = [line.strip() for line in f]

with open(REF_LONG_EXIT_FEATURES, 'r') as f:
    long_exit_features = [line.strip() for line in f]

with open(REF_SHORT_EXIT_FEATURES, 'r') as f:
    short_exit_features = [line.strip() for line in f]

print(f"✅ Feature lists loaded from 52-day models:")
print(f"   LONG Entry: {len(long_entry_features)} features")
print(f"   SHORT Entry: {len(short_entry_features)} features")
print(f"   LONG Exit: {len(long_exit_features)} features")
print(f"   SHORT Exit: {len(short_exit_features)} features")
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

print(f"   Rows: {len(df_features):,}")
print(f"   Period: {df_features['timestamp'].min()} to {df_features['timestamp'].max()}")
print()

print(f"📖 Loading labels: {LABELS_90D.name}")
df_labels = pd.read_csv(LABELS_90D)
df_labels['timestamp'] = pd.to_datetime(df_labels['timestamp'])

print(f"   Rows: {len(df_labels):,}")
print(f"   LONG: {df_labels['signal_long'].sum()} ({df_labels['signal_long'].mean()*100:.2f}%)")
print(f"   SHORT: {df_labels['signal_short'].sum()} ({df_labels['signal_short'].mean()*100:.2f}%)")
print()

# Merge
print("🔗 Merging features with labels...")
df = df_features.merge(
    df_labels[['timestamp', 'signal_long', 'signal_short']],
    on='timestamp',
    how='inner'
)

print(f"   Merged rows: {len(df):,}")
print()

# ============================================================================
# SPLIT TRAIN/VAL
# ============================================================================

print("=" * 80)
print("SPLITTING TRAIN/VAL")
print("=" * 80)
print()

df_train = df[(df['timestamp'] >= TRAIN_START) & (df['timestamp'] <= TRAIN_END)].copy()
df_val = df[(df['timestamp'] >= VAL_START) & (df['timestamp'] <= VAL_END)].copy()

print(f"📊 Training Set:")
print(f"   Period: {df_train['timestamp'].min()} to {df_train['timestamp'].max()}")
print(f"   Rows: {len(df_train):,}")
print(f"   Days: {(df_train['timestamp'].max() - df_train['timestamp'].min()).days}")
print(f"   LONG labels: {df_train['signal_long'].sum()} ({df_train['signal_long'].mean()*100:.2f}%)")
print(f"   SHORT labels: {df_train['signal_short'].sum()} ({df_train['signal_short'].mean()*100:.2f}%)")
print()

print(f"📊 Validation Set:")
print(f"   Period: {df_val['timestamp'].min()} to {df_val['timestamp'].max()}")
print(f"   Rows: {len(df_val):,}")
print(f"   Days: {(df_val['timestamp'].max() - df_val['timestamp'].min()).days}")
print(f"   LONG labels: {df_val['signal_long'].sum()} ({df_val['signal_long'].mean()*100:.2f}%)")
print(f"   SHORT labels: {df_val['signal_short'].sum()} ({df_val['signal_short'].mean()*100:.2f}%)")
print()

# ============================================================================
# CHECK FEATURE AVAILABILITY
# ============================================================================

print("=" * 80)
print("CHECKING FEATURE AVAILABILITY")
print("=" * 80)
print()

available_features = set(df_train.columns)

long_entry_available = [f for f in long_entry_features if f in available_features]
short_entry_available = [f for f in short_entry_features if f in available_features]
long_exit_available = [f for f in long_exit_features if f in available_features]
short_exit_available = [f for f in short_exit_features if f in available_features]

print(f"✅ Feature Availability:")
print(f"   LONG Entry: {len(long_entry_available)}/{len(long_entry_features)} ({len(long_entry_available)/len(long_entry_features)*100:.1f}%)")
print(f"   SHORT Entry: {len(short_entry_available)}/{len(short_entry_features)} ({len(short_entry_available)/len(short_entry_features)*100:.1f}%)")
print(f"   LONG Exit: {len(long_exit_available)}/{len(long_exit_features)} ({len(long_exit_available)/len(long_exit_features)*100:.1f}%)")
print(f"   SHORT Exit: {len(short_exit_available)}/{len(short_exit_features)} ({len(short_exit_available)/len(short_exit_features)*100:.1f}%)")
print()

if len(long_entry_available) < len(long_entry_features) * 0.9:
    print(f"⚠️  WARNING: LONG Entry has <90% features")
if len(short_entry_available) < len(short_entry_features) * 0.9:
    print(f"⚠️  WARNING: SHORT Entry has <90% features")

# Use available features
long_entry_features_final = long_entry_available
short_entry_features_final = short_entry_available
long_exit_features_final = long_exit_available
short_exit_features_final = short_exit_available

print(f"📝 Using available features for training")
print()

# ============================================================================
# TRAIN LONG ENTRY
# ============================================================================

print("=" * 80)
print("TRAINING LONG ENTRY MODEL")
print("=" * 80)
print()

X_train = df_train[long_entry_features_final].fillna(0)
y_train = df_train['signal_long']

print(f"📊 Training data:")
print(f"   Features: {X_train.shape[1]}")
print(f"   Samples: {len(X_train):,}")
print(f"   Positive: {y_train.sum()} ({y_train.mean()*100:.2f}%)")
print()

# Scale
scaler_long_entry = StandardScaler()
X_train_scaled = scaler_long_entry.fit_transform(X_train)

# 5-Fold CV
tscv = TimeSeriesSplit(n_splits=N_FOLDS)
fold_scores = []

print(f"🔄 Enhanced 5-Fold CV:")
for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train_scaled), 1):
    X_fold_train = X_train_scaled[train_idx]
    y_fold_train = y_train.iloc[train_idx]
    X_fold_val = X_train_scaled[val_idx]
    y_fold_val = y_train.iloc[val_idx]

    model_fold = xgb.XGBClassifier(**ENTRY_PARAMS, early_stopping_rounds=50, verbosity=0)
    model_fold.fit(
        X_fold_train, y_fold_train,
        eval_set=[(X_fold_val, y_fold_val)],
        verbose=False
    )

    score = model_fold.score(X_fold_val, y_fold_val)
    fold_scores.append(score)
    print(f"   Fold {fold}: {score:.4f} accuracy")

print(f"   Mean CV accuracy: {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f}")
print()

# Train final model
model_long_entry = xgb.XGBClassifier(**ENTRY_PARAMS, verbosity=0)
model_long_entry.fit(X_train_scaled, y_train)

print(f"✅ LONG Entry model trained")
print()

# Save
timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
long_entry_model_path = MODELS_DIR / f"xgboost_long_entry_90days_tradeoutcome_{timestamp_str}.pkl"
long_entry_scaler_path = MODELS_DIR / f"xgboost_long_entry_90days_tradeoutcome_{timestamp_str}_scaler.pkl"
long_entry_features_path = MODELS_DIR / f"xgboost_long_entry_90days_tradeoutcome_{timestamp_str}_features.txt"

joblib.dump(model_long_entry, long_entry_model_path)
joblib.dump(scaler_long_entry, long_entry_scaler_path)
with open(long_entry_features_path, 'w') as f:
    f.write('\n'.join(long_entry_features_final))

print(f"💾 Saved:")
print(f"   Model: {long_entry_model_path.name}")
print(f"   Scaler: {long_entry_scaler_path.name}")
print(f"   Features: {long_entry_features_path.name}")
print()

# ============================================================================
# TRAIN SHORT ENTRY
# ============================================================================

print("=" * 80)
print("TRAINING SHORT ENTRY MODEL")
print("=" * 80)
print()

X_train = df_train[short_entry_features_final].fillna(0)
y_train = df_train['signal_short']

print(f"📊 Training data:")
print(f"   Features: {X_train.shape[1]}")
print(f"   Samples: {len(X_train):,}")
print(f"   Positive: {y_train.sum()} ({y_train.mean()*100:.2f}%)")
print()

# Scale
scaler_short_entry = StandardScaler()
X_train_scaled = scaler_short_entry.fit_transform(X_train)

# 5-Fold CV
fold_scores = []

print(f"🔄 Enhanced 5-Fold CV:")
for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train_scaled), 1):
    X_fold_train = X_train_scaled[train_idx]
    y_fold_train = y_train.iloc[train_idx]
    X_fold_val = X_train_scaled[val_idx]
    y_fold_val = y_train.iloc[val_idx]

    model_fold = xgb.XGBClassifier(**ENTRY_PARAMS, early_stopping_rounds=50, verbosity=0)
    model_fold.fit(
        X_fold_train, y_fold_train,
        eval_set=[(X_fold_val, y_fold_val)],
        verbose=False
    )

    score = model_fold.score(X_fold_val, y_fold_val)
    fold_scores.append(score)
    print(f"   Fold {fold}: {score:.4f} accuracy")

print(f"   Mean CV accuracy: {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f}")
print()

# Train final model
model_short_entry = xgb.XGBClassifier(**ENTRY_PARAMS, verbosity=0)
model_short_entry.fit(X_train_scaled, y_train)

print(f"✅ SHORT Entry model trained")
print()

# Save
short_entry_model_path = MODELS_DIR / f"xgboost_short_entry_90days_tradeoutcome_{timestamp_str}.pkl"
short_entry_scaler_path = MODELS_DIR / f"xgboost_short_entry_90days_tradeoutcome_{timestamp_str}_scaler.pkl"
short_entry_features_path = MODELS_DIR / f"xgboost_short_entry_90days_tradeoutcome_{timestamp_str}_features.txt"

joblib.dump(model_short_entry, short_entry_model_path)
joblib.dump(scaler_short_entry, short_entry_scaler_path)
with open(short_entry_features_path, 'w') as f:
    f.write('\n'.join(short_entry_features_final))

print(f"💾 Saved:")
print(f"   Model: {short_entry_model_path.name}")
print(f"   Scaler: {short_entry_scaler_path.name}")
print(f"   Features: {short_entry_features_path.name}")
print()

# ============================================================================
# VALIDATE ON OUT-OF-SAMPLE
# ============================================================================

print("=" * 80)
print("VALIDATION ON OUT-OF-SAMPLE PERIOD")
print("=" * 80)
print()

print(f"📊 Validation Period: {VAL_START} to {VAL_END} ({(df_val['timestamp'].max() - df_val['timestamp'].min()).days} days)")
print()

# LONG Entry validation
X_val_long = df_val[long_entry_features_final].fillna(0)
X_val_long_scaled = scaler_long_entry.transform(X_val_long)
long_probs = model_long_entry.predict_proba(X_val_long_scaled)[:, 1]

print(f"🔵 LONG Entry Probabilities:")
print(f"   Min: {long_probs.min():.4f} ({long_probs.min()*100:.2f}%)")
print(f"   Max: {long_probs.max():.4f} ({long_probs.max()*100:.2f}%)")
print(f"   Mean: {long_probs.mean():.4f} ({long_probs.mean()*100:.2f}%)")
print(f"   Median: {np.median(long_probs):.4f} ({np.median(long_probs)*100:.2f}%)")
print(f"   >= 0.85 (threshold): {(long_probs >= 0.85).sum()} ({(long_probs >= 0.85).sum()/len(long_probs)*100:.2f}%)")
print()

# SHORT Entry validation
X_val_short = df_val[short_entry_features_final].fillna(0)
X_val_short_scaled = scaler_short_entry.transform(X_val_short)
short_probs = model_short_entry.predict_proba(X_val_short_scaled)[:, 1]

print(f"🔴 SHORT Entry Probabilities:")
print(f"   Min: {short_probs.min():.4f} ({short_probs.min()*100:.2f}%)")
print(f"   Max: {short_probs.max():.4f} ({short_probs.max()*100:.2f}%)")
print(f"   Mean: {short_probs.mean():.4f} ({short_probs.mean()*100:.2f}%)")
print(f"   Median: {np.median(short_probs):.4f} ({np.median(short_probs)*100:.2f}%)")
print(f"   >= 0.80 (threshold): {(short_probs >= 0.80).sum()} ({(short_probs >= 0.80).sum()/len(short_probs)*100:.2f}%)")
print()

# ============================================================================
# COMPARISON VS 52-DAY AND PREVIOUS ATTEMPTS
# ============================================================================

print("=" * 80)
print("COMPARISON: 90-Day Trade Outcome vs 52-Day vs Previous Failures")
print("=" * 80)
print()

print(f"📊 Previous 90-Day Failures:")
print(f"   Relaxed Labels (1.5% in 120min):")
print(f"     LONG Max: 36.58% (need 85%) ❌")
print(f"     SHORT Max: 27.01% (need 80%) ❌")
print(f"     Feature Match: 100% (171/171)")
print(f"     Label Quality: LOW (too lenient)")
print()

print(f"📊 Current 90-Day (Trade Outcome Labels):")
print(f"   LONG Max: {long_probs.max()*100:.2f}% (need 85%)")
print(f"   SHORT Max: {short_probs.max()*100:.2f}% (need 80%)")
print(f"   Feature Match: {len(long_entry_features_final)}/{len(long_entry_features)} ({len(long_entry_features_final)/len(long_entry_features)*100:.1f}%)")
print(f"   Label Quality: HIGH (risk-aware, 13.73% / 15.09%)")
print()

print(f"📊 52-Day Success Baseline:")
print(f"   LONG Max: 87.42% ✅")
print(f"   SHORT Max: 92.60% ✅")
print(f"   Feature Match: 100%")
print(f"   Label Quality: HIGH (9.79% / 10.89%)")
print()

if long_probs.max() >= 0.85 and short_probs.max() >= 0.80:
    print(f"✅ SUCCESS: Models reach production thresholds!")
    print(f"   Can generate trading signals in validation period")
elif long_probs.max() >= 0.70 and short_probs.max() >= 0.70:
    print(f"⚠️  PARTIAL: Models show improvement but below threshold")
    print(f"   Better than previous attempts but not ready for production")
else:
    print(f"❌ FAILURE: Models still under-confident")
    print(f"   Similar calibration issues as previous attempts")

print()

# ============================================================================
# SUMMARY
# ============================================================================

print("=" * 80)
print("TRAINING COMPLETE")
print("=" * 80)
print()

print(f"📊 Models Trained:")
print(f"   LONG Entry: {long_entry_model_path.name}")
print(f"   SHORT Entry: {short_entry_model_path.name}")
print()

print(f"📈 Label Statistics:")
print(f"   Training: 60 days (Aug 9 - Oct 8)")
print(f"   LONG: {df_train['signal_long'].sum()} ({df_train['signal_long'].mean()*100:.2f}%)")
print(f"   SHORT: {df_train['signal_short'].sum()} ({df_train['signal_short'].mean()*100:.2f}%)")
print()

print(f"🎯 Validation Results:")
print(f"   Period: 28 days (Oct 9 - Nov 6)")
print(f"   LONG Max: {long_probs.max()*100:.2f}%")
print(f"   SHORT Max: {short_probs.max()*100:.2f}%")
print()

print(f"✅ Training complete!")
print()
