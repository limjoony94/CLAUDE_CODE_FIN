"""
314-Day 15-Minute Model Retraining - Enhanced 5-Fold CV
========================================================

Training Strategy:
- Dataset: 314 days of 15-min candles (Dec 26, 2024 - Nov 6, 2025)
- Method: Enhanced 5-Fold CV with TimeSeriesSplit (Oct 24 methodology)
- Split: Last 28 days = Backtest ONLY, Remaining 286 days = 5-Fold CV training
- 0% data leakage (backtest period NEVER used in training)

Models to Train:
1. LONG Entry (286 days, 15-min)
2. SHORT Entry (286 days, 15-min)
3. LONG Exit (286 days, 15-min)
4. SHORT Exit (286 days, 15-min)

Date: 2025-11-06
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import joblib
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Paths
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data" / "features"
LABELS_DIR = BASE_DIR / "data" / "labels"
MODELS_DIR = BASE_DIR / "models"

# Create timestamp for this training run
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

# Training configuration
N_FOLDS = 5
RANDOM_STATE = 42

# XGBoost parameters (from Oct 24 Enhanced training)
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


def load_and_split_data():
    """Load 314-day 15-min data and split into training/backtest"""
    print("=" * 80)
    print("LOADING 314-DAY 15-MINUTE DATA")
    print("=" * 80)

    # Load 314-day features (15-min)
    features_path = DATA_DIR / "BTCUSDT_15m_features_314days_20251106_152653.csv"
    print(f"\nLoading features: {features_path.name}")
    df_features = pd.read_csv(features_path)
    df_features['timestamp'] = pd.to_datetime(df_features['timestamp'])
    print(f"  Period: {df_features['timestamp'].min()} to {df_features['timestamp'].max()}")
    print(f"  Rows: {len(df_features):,}")

    # Load Entry labels (15-min)
    entry_labels_path = LABELS_DIR / "entry_labels_15min_314days_20251106_155150.csv"
    print(f"\nLoading entry labels: {entry_labels_path.name}")
    df_entry_labels = pd.read_csv(entry_labels_path)
    df_entry_labels['timestamp'] = pd.to_datetime(df_entry_labels['timestamp'])
    print(f"  Period: {df_entry_labels['timestamp'].min()} to {df_entry_labels['timestamp'].max()}")
    print(f"  Rows: {len(df_entry_labels):,}")

    # Load Exit labels (15-min)
    exit_labels_path = LABELS_DIR / "exit_labels_15min_314days_20251106_155150.csv"
    print(f"\nLoading exit labels: {exit_labels_path.name}")
    df_exit_labels = pd.read_csv(exit_labels_path)
    df_exit_labels['timestamp'] = pd.to_datetime(df_exit_labels['timestamp'])
    print(f"  Period: {df_exit_labels['timestamp'].min()} to {df_exit_labels['timestamp'].max()}")
    print(f"  Rows: {len(df_exit_labels):,}")

    # Merge labels with features
    df = df_features.merge(
        df_entry_labels[['timestamp', 'signal_long', 'signal_short']],
        on='timestamp',
        how='inner'
    ).rename(columns={
        'signal_long': 'long_label',
        'signal_short': 'short_label'
    })

    df = df.merge(
        df_exit_labels[['timestamp', 'long_exit_patience', 'short_exit_patience']],
        on='timestamp',
        how='inner'
    ).rename(columns={
        'long_exit_patience': 'long_exit_label',
        'short_exit_patience': 'short_exit_label'
    })

    # Drop rows with missing labels
    df = df.dropna(subset=['long_label', 'short_label', 'long_exit_label', 'short_exit_label'])

    print(f"\n✅ Merged Dataset (with labels):")
    print(f"  Period: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"  Rows: {len(df):,}")
    print(f"  Days: {(df['timestamp'].max() - df['timestamp'].min()).days}")

    # CRITICAL: Split Last 28 Days for Backtest ONLY
    print(f"\n⚠️  CRITICAL: Preventing Data Leakage")
    print(f"   Last 28 days: BACKTEST ONLY (never used in training)")
    print(f"   Remaining days: TRAINING ONLY (5-Fold CV)")
    print(f"   NO OVERLAP between training and backtest periods")

    # 28 days in 15-min candles = 28 * 24 * 60 / 15 = 2,688 candles
    validation_candles = 28 * 24 * 60 // 15  # 2,688 candles
    train_end_idx = len(df) - validation_candles

    df_train = df.iloc[:train_end_idx].copy()
    df_backtest = df.iloc[train_end_idx:].copy()

    print(f"\n📚 Training Set (for 5-Fold CV ONLY):")
    print(f"  Period: {df_train['timestamp'].min()} to {df_train['timestamp'].max()}")
    print(f"  Rows: {len(df_train):,}")
    print(f"  Days: {(df_train['timestamp'].max() - df_train['timestamp'].min()).days}")
    print(f"  LONG labels: {df_train['long_label'].sum():,} ({df_train['long_label'].mean():.2%})")
    print(f"  SHORT labels: {df_train['short_label'].sum():,} ({df_train['short_label'].mean():.2%})")

    print(f"\n✅ Backtest Set (100% Out-of-Sample):")
    print(f"  Period: {df_backtest['timestamp'].min()} to {df_backtest['timestamp'].max()}")
    print(f"  Rows: {len(df_backtest):,}")
    print(f"  Days: {(df_backtest['timestamp'].max() - df_backtest['timestamp'].min()).days}")
    print(f"  LONG labels: {df_backtest['long_label'].sum():,} ({df_backtest['long_label'].mean():.2%})")
    print(f"  SHORT labels: {df_backtest['short_label'].sum():,} ({df_backtest['short_label'].mean():.2%})")

    return df_train, df_backtest


def train_entry_model_5fold(df_train, model_type='long'):
    """Train Entry model using Enhanced 5-Fold CV"""
    print("\n" + "=" * 80)
    print(f"TRAINING {model_type.upper()} ENTRY MODEL - Enhanced 5-Fold CV")
    print("=" * 80)

    # Use feature list from current 52-day models
    feature_file = MODELS_DIR / f"xgboost_{model_type}_entry_52day_20251106_140955_features.txt"
    with open(feature_file, 'r') as f:
        original_features = [line.strip() for line in f.readlines()]

    # Filter to only use features that exist in the dataset
    available_features = df_train.columns.tolist()
    feature_cols = [f for f in original_features if f in available_features]

    print(f"\n📋 Features: {len(feature_cols)} (filtered from {len(original_features)} original)")
    if len(feature_cols) < len(original_features):
        missing = set(original_features) - set(feature_cols)
        print(f"⚠️  Skipped {len(missing)} missing features")

    # Prepare data
    X = df_train[feature_cols].fillna(0).replace([np.inf, -np.inf], 0).values
    y = df_train[f'{model_type}_label'].values

    print(f"  X shape: {X.shape}")
    print(f"  y positive: {y.sum():,} ({y.mean():.2%})")

    # Enhanced 5-Fold CV with TimeSeriesSplit
    print(f"\n🔄 Enhanced 5-Fold Cross-Validation:")
    tscv = TimeSeriesSplit(n_splits=N_FOLDS)

    fold_models = []
    fold_scores = []

    for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X)):
        print(f"\n{'─' * 80}")
        print(f"Fold {fold_idx + 1}/{N_FOLDS}")
        print(f"{'─' * 80}")

        # Split fold data
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]

        print(f"  Train: {len(X_train_fold):,} samples")
        print(f"  Val: {len(X_val_fold):,} samples")

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_fold)
        X_val_scaled = scaler.transform(X_val_fold)

        # Train model
        model = xgb.XGBClassifier(**ENTRY_PARAMS)
        model.fit(
            X_train_scaled, y_train_fold,
            eval_set=[(X_val_scaled, y_val_fold)],
            verbose=False
        )

        # Evaluate
        val_score = model.score(X_val_scaled, y_val_fold)
        fold_scores.append(val_score)
        fold_models.append((model, scaler, val_score))

        print(f"  Validation Accuracy: {val_score:.4f}")

    # Select best fold
    best_fold_idx = np.argmax(fold_scores)
    best_model, best_scaler, best_score = fold_models[best_fold_idx]

    print(f"\n✅ Best Fold: {best_fold_idx + 1}")
    print(f"  Validation Accuracy: {best_score:.4f}")
    print(f"  Mean CV Accuracy: {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f}")

    # Save best model
    model_path = MODELS_DIR / f"xgboost_{model_type}_entry_314days_15min_{TIMESTAMP}.pkl"
    scaler_path = MODELS_DIR / f"xgboost_{model_type}_entry_314days_15min_{TIMESTAMP}_scaler.pkl"
    features_path = MODELS_DIR / f"xgboost_{model_type}_entry_314days_15min_{TIMESTAMP}_features.txt"

    joblib.dump(best_model, model_path)
    joblib.dump(best_scaler, scaler_path)
    with open(features_path, 'w') as f:
        f.write('\n'.join(feature_cols))

    print(f"\n💾 Saved:")
    print(f"  Model: {model_path.name}")
    print(f"  Scaler: {scaler_path.name}")
    print(f"  Features: {features_path.name}")

    return best_model, best_scaler, feature_cols


def train_exit_model_5fold(df_train, model_type='long'):
    """Train Exit model using Enhanced 5-Fold CV"""
    print("\n" + "=" * 80)
    print(f"TRAINING {model_type.upper()} EXIT MODEL - Enhanced 5-Fold CV")
    print("=" * 80)

    # Use feature list from current 52-day models
    feature_file = MODELS_DIR / f"xgboost_{model_type}_exit_52day_20251106_140955_features.txt"

    with open(feature_file, 'r') as f:
        original_features = [line.strip() for line in f.readlines()]

    # Filter to only use features that exist in the dataset
    available_features = df_train.columns.tolist()
    feature_cols = [f for f in original_features if f in available_features]

    print(f"\n📋 Features: {len(feature_cols)} (filtered from {len(original_features)} original)")
    if len(feature_cols) < len(original_features):
        missing = set(original_features) - set(feature_cols)
        print(f"⚠️  Skipped {len(missing)} missing features")

    # Prepare data
    X = df_train[feature_cols].fillna(0).replace([np.inf, -np.inf], 0).values
    y = df_train[f'{model_type}_exit_label'].values

    print(f"  X shape: {X.shape}")
    print(f"  y positive: {y.sum():,} ({y.mean():.2%})")

    # Enhanced 5-Fold CV with TimeSeriesSplit
    print(f"\n🔄 Enhanced 5-Fold Cross-Validation:")
    tscv = TimeSeriesSplit(n_splits=N_FOLDS)

    fold_models = []
    fold_scores = []

    for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X)):
        print(f"\n{'─' * 80}")
        print(f"Fold {fold_idx + 1}/{N_FOLDS}")
        print(f"{'─' * 80}")

        # Split fold data
        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]

        print(f"  Train: {len(X_train_fold):,} samples")
        print(f"  Val: {len(X_val_fold):,} samples")

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_fold)
        X_val_scaled = scaler.transform(X_val_fold)

        # Train model
        model = xgb.XGBClassifier(**EXIT_PARAMS)
        model.fit(
            X_train_scaled, y_train_fold,
            eval_set=[(X_val_scaled, y_val_fold)],
            verbose=False
        )

        # Evaluate
        val_score = model.score(X_val_scaled, y_val_fold)
        fold_scores.append(val_score)
        fold_models.append((model, scaler, val_score))

        print(f"  Validation Accuracy: {val_score:.4f}")

    # Select best fold
    best_fold_idx = np.argmax(fold_scores)
    best_model, best_scaler, best_score = fold_models[best_fold_idx]

    print(f"\n✅ Best Fold: {best_fold_idx + 1}")
    print(f"  Validation Accuracy: {best_score:.4f}")
    print(f"  Mean CV Accuracy: {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f}")

    # Save best model
    model_path = MODELS_DIR / f"xgboost_{model_type}_exit_314days_15min_{TIMESTAMP}.pkl"
    scaler_path = MODELS_DIR / f"xgboost_{model_type}_exit_314days_15min_{TIMESTAMP}_scaler.pkl"
    features_path = MODELS_DIR / f"xgboost_{model_type}_exit_314days_15min_{TIMESTAMP}_features.txt"

    joblib.dump(best_model, model_path)
    joblib.dump(best_scaler, scaler_path)
    with open(features_path, 'w') as f:
        f.write('\n'.join(feature_cols))

    print(f"\n💾 Saved:")
    print(f"  Model: {model_path.name}")
    print(f"  Scaler: {scaler_path.name}")
    print(f"  Features: {features_path.name}")

    return best_model, best_scaler, feature_cols


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("314-DAY 15-MINUTE MODEL TRAINING - ENHANCED 5-FOLD CV")
    print("=" * 80)
    print(f"\nTimestamp: {TIMESTAMP}")
    print(f"Training Strategy: Enhanced 5-Fold CV (Oct 24 methodology)")
    print(f"Dataset: 314 days @ 15-min candles")
    print(f"Split: 286 days train (5-Fold CV) + 28 days backtest")
    print()

    # Load and split data
    df_train, df_backtest = load_and_split_data()

    # Train all 4 models
    print("\n" + "=" * 80)
    print("TRAINING 4 MODELS")
    print("=" * 80)

    # 1. LONG Entry
    long_entry_model, long_entry_scaler, long_entry_features = train_entry_model_5fold(df_train, 'long')

    # 2. SHORT Entry
    short_entry_model, short_entry_scaler, short_entry_features = train_entry_model_5fold(df_train, 'short')

    # 3. LONG Exit
    long_exit_model, long_exit_scaler, long_exit_features = train_exit_model_5fold(df_train, 'long')

    # 4. SHORT Exit
    short_exit_model, short_exit_scaler, short_exit_features = train_exit_model_5fold(df_train, 'short')

    # Summary
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print()
    print(f"✅ All 4 models trained successfully!")
    print(f"   Training Period: {df_train['timestamp'].min()} to {df_train['timestamp'].max()}")
    print(f"   Training Days: {(df_train['timestamp'].max() - df_train['timestamp'].min()).days}")
    print(f"   Training Candles: {len(df_train):,}")
    print()
    print(f"📊 Models Saved:")
    print(f"   1. LONG Entry: {len(long_entry_features)} features")
    print(f"   2. SHORT Entry: {len(short_entry_features)} features")
    print(f"   3. LONG Exit: {len(long_exit_features)} features")
    print(f"   4. SHORT Exit: {len(short_exit_features)} features")
    print()
    print(f"📁 Model Identifier: 314days_15min_{TIMESTAMP}")
    print()
    print("=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print()
    print("1. Backtest on 28-day out-of-sample period")
    print("2. Compare vs current 52-day 5-min models")
    print("3. Decide deployment based on backtest performance")
    print()
