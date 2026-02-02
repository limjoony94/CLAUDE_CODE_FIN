#!/usr/bin/env python3
"""
Clean Model Comparison: 76-Day vs 52-Day Training
==================================================

Purpose: Fair comparison of training data length impact
Method: Exact same methodology, only training period length differs

Model A (76-Day):
  Training: Jul 14 - Sep 28, 2025 (76 days)
  Backtest: Sep 29 - Oct 26, 2025 (27 days)

Model B (52-Day):
  Training: Aug 7 - Sep 28, 2025 (52 days)
  Backtest: Sep 29 - Oct 26, 2025 (27 days)

Common:
  - Same methodology: Enhanced 5-Fold CV with TimeSeriesSplit
  - Same backtest period: Sep 29 - Oct 26
  - Same training end: Sep 28
  - 0% data leakage for both

Date: 2025-11-06
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Paths
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data" / "features"
LABELS_DIR = BASE_DIR / "data" / "labels"
MODELS_DIR = BASE_DIR / "models"

# Create timestamp
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

# Training configuration
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


def load_data_with_period(start_date, end_date):
    """Load data for specified period"""
    print(f"\n{'=' * 80}")
    print(f"LOADING DATA: {start_date} to {end_date}")
    print('=' * 80)

    # Load features
    features_path = DATA_DIR / "BTCUSDT_5m_features.csv"
    df_features = pd.read_csv(features_path)
    df_features['timestamp'] = pd.to_datetime(df_features['timestamp'])

    # Load Entry labels
    entry_labels_path = LABELS_DIR / "trade_outcome_labels_20251031_145044.csv"
    df_entry_labels = pd.read_csv(entry_labels_path)
    df_entry_labels['timestamp'] = pd.to_datetime(df_entry_labels['timestamp'])

    # Load Exit labels
    exit_labels_path = LABELS_DIR / "exit_labels_patience_20251030_051002.csv"
    df_exit_labels = pd.read_csv(exit_labels_path)
    df_exit_labels['timestamp'] = pd.to_datetime(df_exit_labels['timestamp'])

    # Merge labels with features
    df = df_features.merge(
        df_entry_labels[['timestamp', 'signal_long', 'signal_short']],
        on='timestamp', how='left'
    ).rename(columns={
        'signal_long': 'long_label',
        'signal_short': 'short_label'
    })

    df = df.merge(
        df_exit_labels[['timestamp', 'long_exit_patience', 'short_exit_patience']],
        on='timestamp', how='left'
    ).rename(columns={
        'long_exit_patience': 'long_exit_label',
        'short_exit_patience': 'short_exit_label'
    })

    # Drop rows with missing labels
    df = df.dropna(subset=['long_label', 'short_label', 'long_exit_label', 'short_exit_label'])

    # Filter to specified period
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    df = df[(df['timestamp'] >= start) & (df['timestamp'] <= end)].copy()
    df = df.reset_index(drop=True)

    print(f"\nDataset:")
    print(f"  Period: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"  Rows: {len(df):,}")
    print(f"  Days: {(df['timestamp'].max() - df['timestamp'].min()).days}")
    print(f"  LONG Entry: {df['long_label'].sum():,} ({df['long_label'].mean():.2%})")
    print(f"  SHORT Entry: {df['short_label'].sum():,} ({df['short_label'].mean():.2%})")
    print(f"  LONG Exit: {df['long_exit_label'].sum():,} ({df['long_exit_label'].mean():.2%})")
    print(f"  SHORT Exit: {df['short_exit_label'].sum():,} ({df['short_exit_label'].mean():.2%})")

    return df


def train_entry_model_5fold(df_train, model_type='long', model_suffix=''):
    """Train Entry model using Enhanced 5-Fold CV"""
    print(f"\n{'=' * 80}")
    print(f"TRAINING {model_type.upper()} ENTRY MODEL - {model_suffix}")
    print('=' * 80)

    # Prepare features
    feature_cols = [col for col in df_train.columns if col not in [
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'long_label', 'short_label', 'long_exit_label', 'short_exit_label'
    ]]

    label_col = f'{model_type}_label'

    X = df_train[feature_cols].fillna(0).replace([np.inf, -np.inf], 0).values
    y = df_train[label_col].values

    print(f"\nDataset:")
    print(f"  Samples: {len(X):,}")
    print(f"  Features: {len(feature_cols)}")
    print(f"  Positive Rate: {y.mean():.2%}")

    # Enhanced 5-Fold CV with TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=N_FOLDS)

    fold_models = []
    fold_scores = []

    for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X)):
        print(f"\n  Fold {fold_idx + 1}/{N_FOLDS}...")

        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_fold)
        X_val_scaled = scaler.transform(X_val_fold)

        # Train model
        model = xgb.XGBClassifier(**ENTRY_PARAMS)
        model.fit(X_train_scaled, y_train_fold,
                 eval_set=[(X_val_scaled, y_val_fold)],
                 verbose=False)

        val_score = model.score(X_val_scaled, y_val_fold)
        fold_scores.append(val_score)
        fold_models.append((model, scaler, val_score))

        print(f"    Validation Accuracy: {val_score:.4f}")

    # Select best fold
    best_fold_idx = np.argmax(fold_scores)
    best_model, best_scaler, best_score = fold_models[best_fold_idx]

    print(f"\n✅ Best Fold: {best_fold_idx + 1}/{N_FOLDS}")
    print(f"   Validation Accuracy: {best_score:.4f}")
    print(f"   Mean CV Accuracy: {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f}")

    # Save model
    model_name = f"xgboost_{model_type}_entry_{model_suffix}_{TIMESTAMP}"
    model_path = MODELS_DIR / f"{model_name}.pkl"
    scaler_path = MODELS_DIR / f"{model_name}_scaler.pkl"
    features_path = MODELS_DIR / f"{model_name}_features.txt"

    joblib.dump(best_model, model_path)
    joblib.dump(best_scaler, scaler_path)

    with open(features_path, 'w') as f:
        f.write('\n'.join(feature_cols))

    print(f"\n💾 Saved:")
    print(f"   {model_path.name}")
    print(f"   {scaler_path.name}")
    print(f"   {features_path.name} ({len(feature_cols)} features)")

    return best_model, best_scaler, feature_cols


def train_exit_model_5fold(df_train, model_type='long', model_suffix=''):
    """Train Exit model using Enhanced 5-Fold CV"""
    print(f"\n{'=' * 80}")
    print(f"TRAINING {model_type.upper()} EXIT MODEL - {model_suffix}")
    print('=' * 80)

    # Exit features (simplified set to match available features)
    exit_features = [
        'rsi', 'macd', 'macd_signal', 'bb_width', 'atr',
        'ema_12', 'vwap', 'trend_strength', 'volatility_regime',
        'volume_ratio', 'lower_low', 'bb_width'
    ]

    # Filter to available features
    available_features = df_train.columns.tolist()
    feature_cols = [f for f in exit_features if f in available_features]

    print(f"\nFeatures: {len(feature_cols)} (from {len(exit_features)} original)")

    label_col = f'{model_type}_exit_label'

    X = df_train[feature_cols].fillna(0).replace([np.inf, -np.inf], 0).values
    y = df_train[label_col].values

    print(f"Samples: {len(X):,}")
    print(f"Positive Rate: {y.mean():.2%}")

    # Enhanced 5-Fold CV
    tscv = TimeSeriesSplit(n_splits=N_FOLDS)

    fold_models = []
    fold_scores = []

    for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X)):
        print(f"\n  Fold {fold_idx + 1}/{N_FOLDS}...")

        X_train_fold, X_val_fold = X[train_idx], X[val_idx]
        y_train_fold, y_val_fold = y[train_idx], y[val_idx]

        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_fold)
        X_val_scaled = scaler.transform(X_val_fold)

        # Train model
        model = xgb.XGBClassifier(**EXIT_PARAMS)
        model.fit(X_train_scaled, y_train_fold,
                 eval_set=[(X_val_scaled, y_val_fold)],
                 verbose=False)

        val_score = model.score(X_val_scaled, y_val_fold)
        fold_scores.append(val_score)
        fold_models.append((model, scaler, val_score))

        print(f"    Validation Accuracy: {val_score:.4f}")

    # Select best fold
    best_fold_idx = np.argmax(fold_scores)
    best_model, best_scaler, best_score = fold_models[best_fold_idx]

    print(f"\n✅ Best Fold: {best_fold_idx + 1}/{N_FOLDS}")
    print(f"   Validation Accuracy: {best_score:.4f}")
    print(f"   Mean CV Accuracy: {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f}")

    # Save model
    model_name = f"xgboost_{model_type}_exit_{model_suffix}_{TIMESTAMP}"
    model_path = MODELS_DIR / f"{model_name}.pkl"
    scaler_path = MODELS_DIR / f"{model_name}_scaler.pkl"
    features_path = MODELS_DIR / f"{model_name}_features.txt"

    joblib.dump(best_model, model_path)
    joblib.dump(best_scaler, scaler_path)

    with open(features_path, 'w') as f:
        f.write('\n'.join(feature_cols))

    print(f"\n💾 Saved:")
    print(f"   {model_path.name}")
    print(f"   {scaler_path.name}")
    print(f"   {features_path.name} ({len(feature_cols)} features)")

    return best_model, best_scaler, feature_cols


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("CLEAN MODEL COMPARISON: 76-DAY vs 52-DAY TRAINING")
    print("=" * 80)
    print("\nObjective: Fair comparison of training data length impact")
    print("Method: Same methodology, only training period differs")
    print(f"Timestamp: {TIMESTAMP}")

    # ========================================================================
    # MODEL A: 76-DAY TRAINING
    # ========================================================================
    print("\n" + "=" * 80)
    print("MODEL A: 76-DAY TRAINING (Jul 14 - Sep 28)")
    print("=" * 80)

    df_train_76d = load_data_with_period(
        start_date="2025-07-14 00:00:00",
        end_date="2025-09-28 23:59:59"
    )

    print("\n🚀 Training 76-Day Models (4 total)...")

    # Train LONG Entry
    train_entry_model_5fold(df_train_76d, 'long', '76day')

    # Train SHORT Entry
    train_entry_model_5fold(df_train_76d, 'short', '76day')

    # Train LONG Exit
    train_exit_model_5fold(df_train_76d, 'long', '76day')

    # Train SHORT Exit
    train_exit_model_5fold(df_train_76d, 'short', '76day')

    print("\n✅ 76-Day Models Complete!")

    # ========================================================================
    # MODEL B: 52-DAY TRAINING
    # ========================================================================
    print("\n" + "=" * 80)
    print("MODEL B: 52-DAY TRAINING (Aug 7 - Sep 28)")
    print("=" * 80)

    df_train_52d = load_data_with_period(
        start_date="2025-08-07 00:00:00",
        end_date="2025-09-28 23:59:59"
    )

    print("\n🚀 Training 52-Day Models (4 total)...")

    # Train LONG Entry
    train_entry_model_5fold(df_train_52d, 'long', '52day')

    # Train SHORT Entry
    train_entry_model_5fold(df_train_52d, 'short', '52day')

    # Train LONG Exit
    train_exit_model_5fold(df_train_52d, 'long', '52day')

    # Train SHORT Exit
    train_exit_model_5fold(df_train_52d, 'short', '52day')

    print("\n✅ 52-Day Models Complete!")

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE - READY FOR COMPARISON")
    print("=" * 80)

    print(f"\n📊 Models Trained:")
    print(f"  76-Day Models: 4 models (LONG/SHORT Entry + Exit)")
    print(f"  52-Day Models: 4 models (LONG/SHORT Entry + Exit)")
    print(f"  Total: 8 models")

    print(f"\n📋 Training Periods:")
    print(f"  76-Day: Jul 14 - Sep 28, 2025")
    print(f"  52-Day: Aug 7 - Sep 28, 2025")

    print(f"\n✅ Backtest Period (Both Models):")
    print(f"  Sep 29 - Oct 26, 2025 (27 days)")
    print(f"  Status: 100% Out-of-Sample (0% overlap)")

    print(f"\n📝 Next Steps:")
    print(f"  1. Run backtest on Sep 29 - Oct 26 for both model sets")
    print(f"  2. Compare performance: 76d vs 52d")
    print(f"  3. Deploy best performer")

    print(f"\n💾 Model Files:")
    print(f"  Pattern: xgboost_*_{{76day|52day}}_{TIMESTAMP}.pkl")
    print(f"  Location: {MODELS_DIR}")

    print("\n" + "=" * 80)
    print("✅ ALL TRAINING COMPLETE")
    print("=" * 80)
