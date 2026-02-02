"""
Phase 2 Model Retraining - Maximum Data Utilization
====================================================

User Requirement:
"Phase 2 가장 긴 데이터로 재훈련, 가장 최근 데이터는 백테스트용 28일. 그 이전 데이터는 훈련."

Implementation:
- Training Period: Jul 14 - Sep 28, 2025 (21,940 candles, 76 days)
- Validation Period: Sep 28 - Oct 26, 2025 (8,064 candles, 28 days)
- Models to Retrain: LONG/SHORT Entry (Enhanced 5-Fold CV) + LONG/SHORT Exit (oppgating_improved)

Date: 2025-11-02
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
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

# Create timestamp for this training run
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

# Training configuration
N_FOLDS = 5
RANDOM_STATE = 42

# XGBoost parameters (from enhanced training)
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
    """Load data and split into training/validation"""
    print("=" * 80)
    print("LOADING AND SPLITTING DATA")
    print("=" * 80)

    # Load features
    features_path = DATA_DIR / "BTCUSDT_5m_features.csv"
    df_features = pd.read_csv(features_path)
    df_features['timestamp'] = pd.to_datetime(df_features['timestamp'])

    # Load Entry labels (trade outcome)
    entry_labels_path = BASE_DIR / "data" / "labels" / "trade_outcome_labels_20251031_145044.csv"
    df_entry_labels = pd.read_csv(entry_labels_path)
    df_entry_labels['timestamp'] = pd.to_datetime(df_entry_labels['timestamp'])

    # Load Exit labels (patience-based)
    exit_labels_path = BASE_DIR / "data" / "labels" / "exit_labels_patience_20251030_051002.csv"
    df_exit_labels = pd.read_csv(exit_labels_path)
    df_exit_labels['timestamp'] = pd.to_datetime(df_exit_labels['timestamp'])

    print(f"\nLoaded Files:")
    print(f"  Features: {len(df_features):,} candles")
    print(f"  Entry Labels: {len(df_entry_labels):,} candles")
    print(f"  Exit Labels: {len(df_exit_labels):,} candles")

    # Merge labels with features (on timestamp)
    # Entry labels
    df = df_features.merge(
        df_entry_labels[['timestamp', 'signal_long', 'signal_short']],
        on='timestamp',
        how='left'
    )

    # Rename entry label columns
    df = df.rename(columns={
        'signal_long': 'long_label',
        'signal_short': 'short_label'
    })

    # Exit labels (from exit_labels file)
    df = df.merge(
        df_exit_labels[['timestamp', 'long_exit_patience', 'short_exit_patience']],
        on='timestamp',
        how='left'
    )

    # Rename exit label columns
    df = df.rename(columns={
        'long_exit_patience': 'long_exit_label',
        'short_exit_patience': 'short_exit_label'
    })

    # Drop rows with missing labels
    df = df.dropna(subset=['long_label', 'short_label', 'long_exit_label', 'short_exit_label'])

    print(f"\nMerged Dataset:")
    print(f"  Period: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"  Candles: {len(df):,}")
    print(f"  Days: {(df['timestamp'].max() - df['timestamp'].min()).days}")

    # Calculate split (28 days = 8,064 candles)
    validation_candles = 28 * 24 * 60 // 5
    train_end_idx = len(df) - validation_candles

    # Split data
    df_train = df.iloc[:train_end_idx].copy()
    df_val = df.iloc[train_end_idx:].copy()

    print(f"\nTraining Data:")
    print(f"  Period: {df_train['timestamp'].min()} to {df_train['timestamp'].max()}")
    print(f"  Candles: {len(df_train):,}")
    print(f"  Days: {(df_train['timestamp'].max() - df_train['timestamp'].min()).days}")
    print(f"  LONG labels: {df_train['long_label'].sum():,} ({df_train['long_label'].mean():.2%})")
    print(f"  SHORT labels: {df_train['short_label'].sum():,} ({df_train['short_label'].mean():.2%})")

    print(f"\nValidation Data (28-day holdout):")
    print(f"  Period: {df_val['timestamp'].min()} to {df_val['timestamp'].max()}")
    print(f"  Candles: {len(df_val):,}")
    print(f"  Days: {(df_val['timestamp'].max() - df_val['timestamp'].min()).days}")
    print(f"  LONG labels: {df_val['long_label'].sum():,} ({df_val['long_label'].mean():.2%})")
    print(f"  SHORT labels: {df_val['short_label'].sum():,} ({df_val['short_label'].mean():.2%})")

    return df_train, df_val


def prepare_entry_features(df):
    """Prepare Entry model features"""
    feature_cols = [col for col in df.columns if col not in [
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'long_label', 'short_label', 'long_exit_label', 'short_exit_label'
    ]]
    return feature_cols


def prepare_exit_features(df):
    """Prepare Exit model features (27 features for oppgating_improved)"""
    # Enhanced Exit features (from oppgating_improved models)
    # Note: Using available features only (sma_50, ema_26, volatility_20 not available)
    exit_features = [
        'rsi', 'macd', 'macd_signal', 'bb_high', 'bb_mid', 'bb_low',
        'atr', 'sma_20', 'sma_10', 'ema_12', 'ema_10',  # Replaced: sma_50→sma_10, ema_26→ema_10
        'volume_sma', 'volatility_10',  # Replaced: volatility_20→volatility_10
        # Enhanced features
        'volume_surge', 'price_acceleration', 'price_vs_ma20', 'price_vs_ma50',
        'rsi_slope', 'rsi_overbought', 'rsi_oversold', 'rsi_divergence',
        'macd_histogram_slope', 'macd_crossover', 'macd_crossunder',
        'bb_position', 'higher_high', 'near_support'
    ]

    # Add enhanced features if not present
    if 'volume_surge' not in df.columns:
        df['volume_surge'] = (df['volume'] > df['volume'].rolling(20).mean() * 1.5).astype(float)

    if 'price_acceleration' not in df.columns:
        df['price_acceleration'] = df['close'].pct_change(5)

    if 'price_vs_ma20' not in df.columns:
        if 'sma_20' in df.columns:
            df['price_vs_ma20'] = (df['close'] - df['sma_20']) / df['sma_20']
        else:
            ma_20 = df['close'].rolling(20).mean()
            df['price_vs_ma20'] = (df['close'] - ma_20) / ma_20

    if 'price_vs_ma50' not in df.columns:
        if 'sma_50' in df.columns:
            df['price_vs_ma50'] = (df['close'] - df['sma_50']) / df['sma_50']
        else:
            ma_50 = df['close'].rolling(50).mean()
            df['price_vs_ma50'] = (df['close'] - ma_50) / ma_50

    if 'rsi_slope' not in df.columns:
        df['rsi_slope'] = df['rsi'].diff(3) / 3

    if 'rsi_overbought' not in df.columns:
        df['rsi_overbought'] = (df['rsi'] > 70).astype(float)

    if 'rsi_oversold' not in df.columns:
        df['rsi_oversold'] = (df['rsi'] < 30).astype(float)

    if 'rsi_divergence' not in df.columns:
        df['rsi_divergence'] = 0

    if 'macd_histogram_slope' not in df.columns:
        if 'macd_diff' in df.columns:
            df['macd_histogram_slope'] = df['macd_diff'].diff(3) / 3
        else:
            df['macd_histogram_slope'] = 0

    if 'macd_crossover' not in df.columns:
        df['macd_crossover'] = ((df['macd'] > df['macd_signal']) &
                                (df['macd'].shift(1) <= df['macd_signal'].shift(1))).astype(float)

    if 'macd_crossunder' not in df.columns:
        df['macd_crossunder'] = ((df['macd'] < df['macd_signal']) &
                                 (df['macd'].shift(1) >= df['macd_signal'].shift(1))).astype(float)

    if 'bb_position' not in df.columns:
        if 'bb_high' in df.columns and 'bb_low' in df.columns:
            df['bb_position'] = (df['close'] - df['bb_low']) / (df['bb_high'] - df['bb_low'])
        else:
            df['bb_position'] = 0.5

    if 'higher_high' not in df.columns:
        df['higher_high'] = (df['high'] > df['high'].shift(1)).astype(float)

    if 'near_support' not in df.columns:
        df['near_support'] = 0

    # Clean NaN
    df = df.ffill().bfill()

    return exit_features


def train_entry_model(df_train, side='LONG'):
    """Train Entry model with Enhanced 5-Fold CV"""
    print("\n" + "=" * 80)
    print(f"TRAINING {side} ENTRY MODEL (Enhanced 5-Fold CV)")
    print("=" * 80)

    # Prepare features and labels
    feature_cols = prepare_entry_features(df_train)
    label_col = 'long_label' if side == 'LONG' else 'short_label'

    X = df_train[feature_cols].values
    y = df_train[label_col].values

    print(f"\nDataset:")
    print(f"  Samples: {len(X):,}")
    print(f"  Features: {len(feature_cols)}")
    print(f"  Label Distribution: {np.bincount(y.astype(int))}")
    print(f"  Positive Rate: {y.mean():.2%}")

    # Time-based cross-validation
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

        # Train XGBoost
        model = xgb.XGBClassifier(**ENTRY_PARAMS)
        model.fit(X_train_scaled, y_train_fold, verbose=False)

        # Evaluate
        train_score = model.score(X_train_scaled, y_train_fold)
        val_score = model.score(X_val_scaled, y_val_fold)

        # Prediction rate
        val_preds = model.predict(X_val_scaled)
        pred_rate = val_preds.mean()

        print(f"  Train Accuracy: {train_score:.4f}")
        print(f"  Val Accuracy: {val_score:.4f}")
        print(f"  Prediction Rate: {pred_rate:.2%}")

        fold_models.append((model, scaler))
        fold_scores.append(val_score)

    # Select best fold
    best_fold_idx = np.argmax(fold_scores)
    best_model, best_scaler = fold_models[best_fold_idx]

    print(f"\n{'=' * 80}")
    print(f"BEST FOLD: {best_fold_idx + 1}/{N_FOLDS}")
    print(f"  Val Accuracy: {fold_scores[best_fold_idx]:.4f}")
    print(f"{'=' * 80}")

    # Train final model on full training data
    print(f"\nTraining final model on full training data...")
    scaler_final = StandardScaler()
    X_scaled = scaler_final.fit_transform(X)

    model_final = xgb.XGBClassifier(**ENTRY_PARAMS)
    model_final.fit(X_scaled, y, verbose=False)

    # Save model
    model_name = f"xgboost_{side.lower()}_entry_phase2_{TIMESTAMP}"
    model_path = MODELS_DIR / f"{model_name}.pkl"
    scaler_path = MODELS_DIR / f"{model_name}_scaler.pkl"
    features_path = MODELS_DIR / f"{model_name}_features.txt"

    joblib.dump(model_final, model_path)
    joblib.dump(scaler_final, scaler_path)

    with open(features_path, 'w') as f:
        f.write('\n'.join(feature_cols))

    print(f"\n✅ Model saved:")
    print(f"  Model: {model_path.name}")
    print(f"  Scaler: {scaler_path.name}")
    print(f"  Features: {features_path.name} ({len(feature_cols)} features)")

    return model_final, scaler_final, feature_cols


def train_exit_model(df_train, side='LONG'):
    """Train Exit model (oppgating_improved methodology)"""
    print("\n" + "=" * 80)
    print(f"TRAINING {side} EXIT MODEL (oppgating_improved)")
    print("=" * 80)

    # Prepare features and labels
    exit_features = prepare_exit_features(df_train)
    label_col = 'long_exit_label' if side == 'LONG' else 'short_exit_label'

    X = df_train[exit_features].values
    y = df_train[label_col].values

    print(f"\nDataset:")
    print(f"  Samples: {len(X):,}")
    print(f"  Features: {len(exit_features)}")
    print(f"  Label Distribution: {np.bincount(y.astype(int))}")
    print(f"  Positive Rate: {y.mean():.2%}")

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train XGBoost
    model = xgb.XGBClassifier(**EXIT_PARAMS)
    model.fit(X_scaled, y, verbose=False)

    # Evaluate
    train_score = model.score(X_scaled, y)
    train_preds = model.predict(X_scaled)
    pred_rate = train_preds.mean()

    print(f"\nTraining Results:")
    print(f"  Accuracy: {train_score:.4f}")
    print(f"  Prediction Rate: {pred_rate:.2%}")

    # Save model
    model_name = f"xgboost_{side.lower()}_exit_phase2_{TIMESTAMP}"
    model_path = MODELS_DIR / f"{model_name}.pkl"
    scaler_path = MODELS_DIR / f"{model_name}_scaler.pkl"
    features_path = MODELS_DIR / f"{model_name}_features.txt"

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)

    with open(features_path, 'w') as f:
        f.write('\n'.join(exit_features))

    print(f"\n✅ Model saved:")
    print(f"  Model: {model_path.name}")
    print(f"  Scaler: {scaler_path.name}")
    print(f"  Features: {features_path.name} ({len(exit_features)} features)")

    return model, scaler, exit_features

def main():
    print("\n" + "=" * 80)
    print("PHASE 2 MODEL RETRAINING")
    print("=" * 80)
    print(f"Timestamp: {TIMESTAMP}")
    print(f"User Requirement: Use longest data, 28-day validation holdout")
    print("=" * 80)

    # Load and split data
    df_train, df_val = load_and_split_data()

    # Train Entry models
    long_entry_model, long_entry_scaler, long_entry_features = train_entry_model(df_train, 'LONG')
    short_entry_model, short_entry_scaler, short_entry_features = train_entry_model(df_train, 'SHORT')

    # Train Exit models
    long_exit_model, long_exit_scaler, long_exit_features = train_exit_model(df_train, 'LONG')
    short_exit_model, short_exit_scaler, short_exit_features = train_exit_model(df_train, 'SHORT')

    # Summary
    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"\nModels Timestamp: {TIMESTAMP}")
    print(f"\nLONG Entry: {len(long_entry_features)} features")
    print(f"SHORT Entry: {len(short_entry_features)} features")
    print(f"LONG Exit: {len(long_exit_features)} features")
    print(f"SHORT Exit: {len(short_exit_features)} features")

    print(f"\nNext Step: Run validation backtest on 28-day holdout period")
    print(f"  Validation Period: {df_val['timestamp'].min()} to {df_val['timestamp'].max()}")
    print(f"  Validation Candles: {len(df_val):,}")


if __name__ == "__main__":
    main()