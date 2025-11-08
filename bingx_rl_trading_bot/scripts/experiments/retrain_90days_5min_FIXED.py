"""
Retrain All 4 Models on 90-Day 5-Min Dataset - FIXED VERSION
=============================================================

PURPOSE: Use 52-day model's ACTUAL feature lists (not hardcoded wrong names)

KEY FIX: Load feature names from 52-day model files instead of hardcoding

Dataset: 90 days of 5-min data (Aug 8 - Nov 6, 2025)
Training: 61 days (Aug 9 - Oct 8)
Validation: 28 days (Oct 9 - Nov 6)

Created: 2025-11-06 17:25 KST
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import pickle
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Paths
DATA_DIR = PROJECT_ROOT / "data" / "features"
LABELS_DIR = PROJECT_ROOT / "data" / "labels"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Input files
FEATURES_FILE = DATA_DIR / "BTCUSDT_5m_features_90days_complete_20251106_164542.csv"
ENTRY_LABELS_FILE = LABELS_DIR / "entry_labels_90days_5min_relaxed_20251106_170658.csv"
EXIT_LABELS_FILE = LABELS_DIR / "exit_labels_90days_5min_relaxed_20251106_170658.csv"

# 52-day model feature files (to get CORRECT feature names)
REF_LONG_ENTRY_FEATURES = MODELS_DIR / "xgboost_long_entry_52day_20251106_140955_features.txt"
REF_SHORT_ENTRY_FEATURES = MODELS_DIR / "xgboost_short_entry_52day_20251106_140955_features.txt"
REF_LONG_EXIT_FEATURES = MODELS_DIR / "xgboost_long_exit_52day_20251106_140955_features.txt"
REF_SHORT_EXIT_FEATURES = MODELS_DIR / "xgboost_short_exit_52day_20251106_140955_features.txt"

# Validation split (61 days training + 28 days validation)
VALIDATION_START = "2025-10-09"

# XGBoost parameters (Enhanced 5-Fold CV)
ENTRY_PARAMS = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'max_depth': 6,
    'learning_rate': 0.01,
    'n_estimators': 500,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 5,
    'gamma': 0.1,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'random_state': 42,
    'n_jobs': -1,
    'early_stopping_rounds': 50
}

EXIT_PARAMS = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'max_depth': 5,
    'learning_rate': 0.01,
    'n_estimators': 300,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 3,
    'gamma': 0.05,
    'reg_alpha': 0.05,
    'reg_lambda': 0.5,
    'random_state': 42,
    'n_jobs': -1,
    'early_stopping_rounds': 30
}

print("=" * 80)
print("RETRAINING 90-DAY MODELS - FIXED VERSION (CORRECT FEATURE NAMES)")
print("=" * 80)
print()
print(f"📂 Features: {FEATURES_FILE.name}")
print(f"📊 Entry Labels: {ENTRY_LABELS_FILE.name}")
print(f"🚪 Exit Labels: {EXIT_LABELS_FILE.name}")
print()
print(f"🔧 FIX: Using 52-day model's ACTUAL feature names")
print(f"   Reference: {REF_LONG_ENTRY_FEATURES.name}")
print()

# ============================================================================
# LOAD FEATURE NAMES FROM 52-DAY MODELS
# ============================================================================

print("=" * 80)
print("LOADING FEATURE NAMES FROM 52-DAY MODELS")
print("=" * 80)
print()

with open(REF_LONG_ENTRY_FEATURES, 'r') as f:
    long_entry_features = [line.strip() for line in f]
print(f"✅ LONG Entry features: {len(long_entry_features)}")

with open(REF_SHORT_ENTRY_FEATURES, 'r') as f:
    short_entry_features = [line.strip() for line in f]
print(f"✅ SHORT Entry features: {len(short_entry_features)}")

with open(REF_LONG_EXIT_FEATURES, 'r') as f:
    long_exit_features = [line.strip() for line in f]
print(f"✅ LONG Exit features: {len(long_exit_features)}")

with open(REF_SHORT_EXIT_FEATURES, 'r') as f:
    short_exit_features = [line.strip() for line in f]
print(f"✅ SHORT Exit features: {len(short_exit_features)}")
print()

# ============================================================================
# LOAD DATA
# ============================================================================

print("=" * 80)
print("LOADING DATA")
print("=" * 80)
print()

print("📖 Loading features...")
df_features = pd.read_csv(FEATURES_FILE)
df_features['timestamp'] = pd.to_datetime(df_features['timestamp'])
print(f"   Rows: {len(df_features):,}")
print(f"   Columns: {len(df_features.columns)}")
print()

print("📖 Loading entry labels...")
df_entry_labels = pd.read_csv(ENTRY_LABELS_FILE)
df_entry_labels['timestamp'] = pd.to_datetime(df_entry_labels['timestamp'])
print(f"   LONG entries: {df_entry_labels['long_entry_label'].sum():,} ({df_entry_labels['long_entry_label'].sum() / len(df_entry_labels) * 100:.2f}%)")
print(f"   SHORT entries: {df_entry_labels['short_entry_label'].sum():,} ({df_entry_labels['short_entry_label'].sum() / len(df_entry_labels) * 100:.2f}%)")
print()

print("📖 Loading exit labels...")
df_exit_labels = pd.read_csv(EXIT_LABELS_FILE)
df_exit_labels['timestamp'] = pd.to_datetime(df_exit_labels['timestamp'])
print(f"   LONG exits: {df_exit_labels['long_exit_label'].sum():,} ({df_exit_labels['long_exit_label'].sum() / len(df_exit_labels) * 100:.2f}%)")
print(f"   SHORT exits: {df_exit_labels['short_exit_label'].sum():,} ({df_exit_labels['short_exit_label'].sum() / len(df_exit_labels) * 100:.2f}%)")
print()

# Merge data
df = df_features.merge(df_entry_labels[['timestamp', 'long_entry_label', 'short_entry_label']], on='timestamp')
df = df.merge(df_exit_labels[['timestamp', 'long_exit_label', 'short_exit_label']], on='timestamp')

print(f"✅ Data merged: {len(df):,} rows")
print()

# Check feature availability
print("=" * 80)
print("CHECKING FEATURE AVAILABILITY")
print("=" * 80)
print()

available_cols = set(df.columns)

long_entry_available = [f for f in long_entry_features if f in available_cols]
long_entry_missing = [f for f in long_entry_features if f not in available_cols]

short_entry_available = [f for f in short_entry_features if f in available_cols]
short_entry_missing = [f for f in short_entry_features if f not in available_cols]

long_exit_available = [f for f in long_exit_features if f in available_cols]
long_exit_missing = [f for f in long_exit_features if f not in available_cols]

short_exit_available = [f for f in short_exit_features if f in available_cols]
short_exit_missing = [f for f in short_exit_features if f not in available_cols]

print(f"📊 LONG Entry: {len(long_entry_available)}/{len(long_entry_features)} available ({len(long_entry_available)/len(long_entry_features)*100:.1f}%)")
if long_entry_missing:
    print(f"   Missing: {len(long_entry_missing)} features")
    print(f"   First 5 missing: {long_entry_missing[:5]}")

print(f"📊 SHORT Entry: {len(short_entry_available)}/{len(short_entry_features)} available ({len(short_entry_available)/len(short_entry_features)*100:.1f}%)")
if short_entry_missing:
    print(f"   Missing: {len(short_entry_missing)} features")
    print(f"   First 5 missing: {short_entry_missing[:5]}")

print(f"📊 LONG Exit: {len(long_exit_available)}/{len(long_exit_features)} available ({len(long_exit_available)/len(long_exit_features)*100:.1f}%)")
if long_exit_missing:
    print(f"   Missing: {len(long_exit_missing)} features")

print(f"📊 SHORT Exit: {len(short_exit_available)}/{len(short_exit_features)} available ({len(short_exit_available)/len(short_exit_features)*100:.1f}%)")
if short_exit_missing:
    print(f"   Missing: {len(short_exit_missing)} features")
print()

# ============================================================================
# TRAIN/VALIDATION SPLIT
# ============================================================================

print("=" * 80)
print("TRAIN/VALIDATION SPLIT")
print("=" * 80)
print()

train_mask = df['timestamp'] < VALIDATION_START
train_df = df[train_mask].copy()
val_df = df[~train_mask].copy()

print(f"📚 Training Set:")
print(f"   Period: {train_df['timestamp'].min()} to {train_df['timestamp'].max()}")
print(f"   Rows: {len(train_df):,}")
print(f"   Days: {(train_df['timestamp'].max() - train_df['timestamp'].min()).days}")
print()

print(f"✅ Validation Set:")
print(f"   Period: {val_df['timestamp'].min()} to {val_df['timestamp'].max()}")
print(f"   Rows: {len(val_df):,}")
print(f"   Days: {(val_df['timestamp'].max() - val_df['timestamp'].min()).days}")
print()

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def train_model_with_cv(X_train, y_train, X_val, y_val, params, model_name):
    """Train model with Enhanced 5-Fold CV"""

    print(f"🔧 Training {model_name}...")
    print(f"   Training samples: {len(X_train):,}")
    print(f"   Positive labels: {y_train.sum():,} ({y_train.sum() / len(y_train) * 100:.2f}%)")
    print(f"   Validation samples: {len(X_val):,}")
    print()

    # TimeSeriesSplit for CV
    tscv = TimeSeriesSplit(n_splits=5)

    # Track best model
    best_model = None
    best_score = 0

    print("   Enhanced 5-Fold Cross-Validation:")
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train), 1):
        X_fold_train = X_train.iloc[train_idx]
        y_fold_train = y_train.iloc[train_idx]
        X_fold_val = X_train.iloc[val_idx]
        y_fold_val = y_train.iloc[val_idx]

        # Train fold model
        model = xgb.XGBClassifier(**params)
        model.fit(
            X_fold_train, y_fold_train,
            eval_set=[(X_fold_val, y_fold_val)],
            verbose=False
        )

        # Evaluate on fold validation
        val_preds = model.predict_proba(X_fold_val)[:, 1]
        val_score = ((val_preds > 0.5) == y_fold_val).mean()

        print(f"      Fold {fold}: Accuracy = {val_score:.4f}")

        # Track best
        if val_score > best_score:
            best_score = val_score
            best_model = model

    print(f"   ✅ Best CV Score: {best_score:.4f}")
    print()

    # Final evaluation on validation set
    val_preds = best_model.predict(X_val)
    val_proba = best_model.predict_proba(X_val)[:, 1]

    print(f"   📊 Validation Set Performance:")
    print(f"      Accuracy: {(val_preds == y_val).mean():.4f}")
    print(f"      Positive predictions: {val_preds.sum():,} ({val_preds.sum() / len(val_preds) * 100:.2f}%)")
    print(f"      Mean probability: {val_proba.mean():.4f}")
    print(f"      Max probability: {val_proba.max():.4f}")
    print(f"      Min probability: {val_proba.min():.4f}")
    print()

    # Probability distribution
    print(f"   📈 Probability Distribution:")
    print(f"      >0.90: {(val_proba > 0.90).sum():,} ({(val_proba > 0.90).sum() / len(val_proba) * 100:.2f}%)")
    print(f"      >0.80: {(val_proba > 0.80).sum():,} ({(val_proba > 0.80).sum() / len(val_proba) * 100:.2f}%)")
    print(f"      >0.70: {(val_proba > 0.70).sum():,} ({(val_proba > 0.70).sum() / len(val_proba) * 100:.2f}%)")
    print(f"      >0.50: {(val_proba > 0.50).sum():,} ({(val_proba > 0.50).sum() / len(val_proba) * 100:.2f}%)")
    print()

    # Confusion matrix
    cm = confusion_matrix(y_val, val_preds)
    print(f"   Confusion Matrix:")
    print(f"      TN: {cm[0,0]:,}  FP: {cm[0,1]:,}")
    print(f"      FN: {cm[1,0]:,}  TP: {cm[1,1]:,}")
    print()

    return best_model

timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')

# ============================================================================
# TRAIN ALL 4 MODELS
# ============================================================================

print("=" * 80)
print("TRAINING ALL 4 MODELS")
print("=" * 80)
print()

# 1. LONG Entry
print("1/4: LONG ENTRY MODEL")
print("-" * 80)
X_train_long = train_df[long_entry_available]
y_train_long = train_df['long_entry_label']
X_val_long = val_df[long_entry_available]
y_val_long = val_df['long_entry_label']

scaler_long_entry = StandardScaler()
X_train_long_scaled = pd.DataFrame(
    scaler_long_entry.fit_transform(X_train_long),
    columns=X_train_long.columns,
    index=X_train_long.index
)
X_val_long_scaled = pd.DataFrame(
    scaler_long_entry.transform(X_val_long),
    columns=X_val_long.columns,
    index=X_val_long.index
)

model_long_entry = train_model_with_cv(
    X_train_long_scaled, y_train_long,
    X_val_long_scaled, y_val_long,
    ENTRY_PARAMS,
    "LONG Entry"
)

# Save
with open(MODELS_DIR / f"xgboost_long_entry_90days_5min_FIXED_{timestamp_str}.pkl", 'wb') as f:
    pickle.dump(model_long_entry, f)
with open(MODELS_DIR / f"xgboost_long_entry_90days_5min_FIXED_{timestamp_str}_scaler.pkl", 'wb') as f:
    pickle.dump(scaler_long_entry, f)
with open(MODELS_DIR / f"xgboost_long_entry_90days_5min_FIXED_{timestamp_str}_features.txt", 'w') as f:
    f.write('\n'.join(long_entry_available))
print(f"💾 LONG Entry model saved")
print()

# 2. SHORT Entry
print("2/4: SHORT ENTRY MODEL")
print("-" * 80)
X_train_short = train_df[short_entry_available]
y_train_short = train_df['short_entry_label']
X_val_short = val_df[short_entry_available]
y_val_short = val_df['short_entry_label']

scaler_short_entry = StandardScaler()
X_train_short_scaled = pd.DataFrame(
    scaler_short_entry.fit_transform(X_train_short),
    columns=X_train_short.columns,
    index=X_train_short.index
)
X_val_short_scaled = pd.DataFrame(
    scaler_short_entry.transform(X_val_short),
    columns=X_val_short.columns,
    index=X_val_short.index
)

model_short_entry = train_model_with_cv(
    X_train_short_scaled, y_train_short,
    X_val_short_scaled, y_val_short,
    ENTRY_PARAMS,
    "SHORT Entry"
)

# Save
with open(MODELS_DIR / f"xgboost_short_entry_90days_5min_FIXED_{timestamp_str}.pkl", 'wb') as f:
    pickle.dump(model_short_entry, f)
with open(MODELS_DIR / f"xgboost_short_entry_90days_5min_FIXED_{timestamp_str}_scaler.pkl", 'wb') as f:
    pickle.dump(scaler_short_entry, f)
with open(MODELS_DIR / f"xgboost_short_entry_90days_5min_FIXED_{timestamp_str}_features.txt", 'w') as f:
    f.write('\n'.join(short_entry_available))
print(f"💾 SHORT Entry model saved")
print()

# 3. LONG Exit
print("3/4: LONG EXIT MODEL")
print("-" * 80)
X_train_long_exit = train_df[long_exit_available]
y_train_long_exit = train_df['long_exit_label']
X_val_long_exit = val_df[long_exit_available]
y_val_long_exit = val_df['long_exit_label']

scaler_long_exit = StandardScaler()
X_train_long_exit_scaled = pd.DataFrame(
    scaler_long_exit.fit_transform(X_train_long_exit),
    columns=X_train_long_exit.columns,
    index=X_train_long_exit.index
)
X_val_long_exit_scaled = pd.DataFrame(
    scaler_long_exit.transform(X_val_long_exit),
    columns=X_val_long_exit.columns,
    index=X_val_long_exit.index
)

model_long_exit = train_model_with_cv(
    X_train_long_exit_scaled, y_train_long_exit,
    X_val_long_exit_scaled, y_val_long_exit,
    EXIT_PARAMS,
    "LONG Exit"
)

# Save
with open(MODELS_DIR / f"xgboost_long_exit_90days_5min_FIXED_{timestamp_str}.pkl", 'wb') as f:
    pickle.dump(model_long_exit, f)
with open(MODELS_DIR / f"xgboost_long_exit_90days_5min_FIXED_{timestamp_str}_scaler.pkl", 'wb') as f:
    pickle.dump(scaler_long_exit, f)
with open(MODELS_DIR / f"xgboost_long_exit_90days_5min_FIXED_{timestamp_str}_features.txt", 'w') as f:
    f.write('\n'.join(long_exit_available))
print(f"💾 LONG Exit model saved")
print()

# 4. SHORT Exit
print("4/4: SHORT EXIT MODEL")
print("-" * 80)
X_train_short_exit = train_df[short_exit_available]
y_train_short_exit = train_df['short_exit_label']
X_val_short_exit = val_df[short_exit_available]
y_val_short_exit = val_df['short_exit_label']

scaler_short_exit = StandardScaler()
X_train_short_exit_scaled = pd.DataFrame(
    scaler_short_exit.fit_transform(X_train_short_exit),
    columns=X_train_short_exit.columns,
    index=X_train_short_exit.index
)
X_val_short_exit_scaled = pd.DataFrame(
    scaler_short_exit.transform(X_val_short_exit),
    columns=X_val_short_exit.columns,
    index=X_val_short_exit.index
)

model_short_exit = train_model_with_cv(
    X_train_short_exit_scaled, y_train_short_exit,
    X_val_short_exit_scaled, y_val_short_exit,
    EXIT_PARAMS,
    "SHORT Exit"
)

# Save
with open(MODELS_DIR / f"xgboost_short_exit_90days_5min_FIXED_{timestamp_str}.pkl", 'wb') as f:
    pickle.dump(model_short_exit, f)
with open(MODELS_DIR / f"xgboost_short_exit_90days_5min_FIXED_{timestamp_str}_scaler.pkl", 'wb') as f:
    pickle.dump(scaler_short_exit, f)
with open(MODELS_DIR / f"xgboost_short_exit_90days_5min_FIXED_{timestamp_str}_features.txt", 'w') as f:
    f.write('\n'.join(short_exit_available))
print(f"💾 SHORT Exit model saved")
print()

# ============================================================================
# SUMMARY
# ============================================================================

print("=" * 80)
print("TRAINING COMPLETE - FIXED VERSION")
print("=" * 80)
print()

print(f"✅ All 4 models trained with CORRECT feature names")
print(f"   Timestamp: {timestamp_str}")
print()

print(f"📊 Feature Usage:")
print(f"   LONG Entry: {len(long_entry_available)}/{len(long_entry_features)} features ({len(long_entry_available)/len(long_entry_features)*100:.1f}%)")
print(f"   SHORT Entry: {len(short_entry_available)}/{len(short_entry_features)} features ({len(short_entry_available)/len(short_entry_features)*100:.1f}%)")
print(f"   LONG Exit: {len(long_exit_available)}/{len(long_exit_features)} features ({len(long_exit_available)/len(long_exit_features)*100:.1f}%)")
print(f"   SHORT Exit: {len(short_exit_available)}/{len(short_exit_features)} features ({len(short_exit_available)/len(short_exit_features)*100:.1f}%)")
print()

print("=" * 80)
print("NEXT STEPS")
print("=" * 80)
print()
print("1. Check validation probability distributions")
print("2. If probabilities reach thresholds (>80-85%), run backtest")
print("3. Compare vs 52-day models")
print()
print("✅ Fixed training script complete!")
