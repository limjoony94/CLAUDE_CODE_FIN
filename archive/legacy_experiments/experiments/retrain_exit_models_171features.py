"""
Retrain Exit Models with 171 Features (Optimal Triple Barrier Labels)

Improvements vs current Exit models:
1. Features: 171 vs 12 (14× more features)
2. Labels: Optimal Triple Barrier (15% exit rate) vs simple next-candle
3. Training: 90-day vs 52-day (more data)
4. Method: P&L-weighted scoring with 1:1 R/R ATR barriers
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import pickle
import joblib
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit

# Paths
BASE_DIR = Path(__file__).parent.parent.parent
DATA_DIR = BASE_DIR / "data"
FEATURES_DIR = DATA_DIR / "features"
LABELS_DIR = DATA_DIR / "labels"
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Input files
FEATURES_FILE = FEATURES_DIR / "BTCUSDT_5m_features_90days_complete_20251106_164542.csv"
OPTIMAL_EXIT_LABELS = LABELS_DIR / "exit_labels_optimal_triple_barrier_20251106_223351.csv"

# Reference: Use same 171 features as 90-day Entry models
LONG_ENTRY_FEATURES_REF = MODELS_DIR / "xgboost_long_entry_90days_tradeoutcome_20251106_193900_features.txt"

# Output timestamp
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

# Training parameters
TRAIN_END_DATE = "2025-10-08 23:59:59"  # Same as Entry models
VAL_START_DATE = "2025-10-09 00:00:00"


def load_feature_names():
    """Load 171 feature names from 90-day Entry model"""
    with open(LONG_ENTRY_FEATURES_REF, 'r') as f:
        features = [line.strip() for line in f.readlines() if line.strip()]
    print(f"✅ Reference features loaded: {len(features)}")
    return features


def train_model_with_cv(X_train, y_train, X_val, y_val, model_name):
    """
    Train XGBoost model with Enhanced 5-Fold Time Series Cross-Validation
    """
    print(f"\n🔧 Training {model_name}...")
    print(f"   Training samples: {len(X_train):,}")
    print(f"   Positive labels: {y_train.sum():,} ({y_train.mean()*100:.2f}%)")
    print(f"   Validation samples: {len(X_val):,}")

    # Enhanced 5-Fold Time Series Cross-Validation
    tscv = TimeSeriesSplit(n_splits=5)

    best_score = 0
    best_model = None

    print(f"\n   Enhanced 5-Fold Cross-Validation:")

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train), 1):
        X_fold_train = X_train.iloc[train_idx]
        y_fold_train = y_train.iloc[train_idx]
        X_fold_val = X_train.iloc[val_idx]
        y_fold_val = y_train.iloc[val_idx]

        # Train model
        model = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='logloss',
            max_depth=6,
            learning_rate=0.1,
            n_estimators=200,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            tree_method='hist'
        )

        model.fit(
            X_fold_train, y_fold_train,
            eval_set=[(X_fold_val, y_fold_val)],
            verbose=False
        )

        # Evaluate
        score = model.score(X_fold_val, y_fold_val)
        print(f"      Fold {fold}: Accuracy = {score:.4f}")

        if score > best_score:
            best_score = score
            best_model = model

    print(f"   ✅ Best CV Score: {best_score:.4f}")

    # Validation set performance
    val_preds = best_model.predict(X_val)
    val_proba = best_model.predict_proba(X_val)[:, 1]

    accuracy = (val_preds == y_val).mean()
    pos_preds = val_preds.sum()

    print(f"\n   📊 Validation Set Performance:")
    print(f"      Accuracy: {accuracy:.4f}")
    print(f"      Positive predictions: {pos_preds} ({pos_preds/len(y_val)*100:.2f}%)")
    print(f"      Mean probability: {val_proba.mean():.4f}")
    print(f"      Max probability: {val_proba.max():.4f}")

    # Probability distribution
    print(f"\n   📈 Probability Distribution:")
    for threshold in [0.90, 0.85, 0.80, 0.75, 0.70]:
        count = (val_proba >= threshold).sum()
        pct = count / len(val_proba) * 100
        print(f"      >{threshold:.2f}: {count} ({pct:.2f}%)")

    # Confusion matrix
    tp = ((val_preds == 1) & (y_val == 1)).sum()
    tn = ((val_preds == 0) & (y_val == 0)).sum()
    fp = ((val_preds == 1) & (y_val == 0)).sum()
    fn = ((val_preds == 0) & (y_val == 1)).sum()

    print(f"\n   Confusion Matrix:")
    print(f"      TN: {tn}  FP: {fp}")
    print(f"      FN: {fn}  TP: {tp}")

    return best_model


def main():
    print("="*80)
    print("RETRAINING EXIT MODELS WITH 171 FEATURES")
    print("="*80)

    print(f"\n📂 Features: {FEATURES_FILE.name}")
    print(f"📊 Exit Labels: {OPTIMAL_EXIT_LABELS.name}")

    # Load feature names
    print(f"\n{'='*80}")
    print("LOADING FEATURE NAMES")
    print(f"{'='*80}")
    feature_columns = load_feature_names()

    # Load data
    print(f"\n{'='*80}")
    print("LOADING DATA")
    print(f"{'='*80}")

    print(f"\n📖 Loading features...")
    df_features = pd.read_csv(FEATURES_FILE)
    df_features['timestamp'] = pd.to_datetime(df_features['timestamp'])
    print(f"   Rows: {len(df_features):,}")
    print(f"   Columns: {len(df_features.columns)}")

    print(f"\n📖 Loading optimal exit labels...")
    df_exit_labels = pd.read_csv(OPTIMAL_EXIT_LABELS)
    df_exit_labels['timestamp'] = pd.to_datetime(df_exit_labels['timestamp'])
    print(f"   LONG exit signals: {df_exit_labels['exit_label_long'].sum():,} ({df_exit_labels['exit_label_long'].mean()*100:.2f}%)")
    print(f"   SHORT exit signals: {df_exit_labels['exit_label_short'].sum():,} ({df_exit_labels['exit_label_short'].mean()*100:.2f}%)")

    # Merge
    df = df_features.copy()
    df = df.merge(df_exit_labels[['timestamp', 'exit_label_long', 'exit_label_short']], on='timestamp', how='left')
    df['long_exit'] = df['exit_label_long'].fillna(0).astype(int)
    df['short_exit'] = df['exit_label_short'].fillna(0).astype(int)

    print(f"\n✅ Data merged: {len(df):,} rows")

    # Check feature availability
    print(f"\n{'='*80}")
    print("CHECKING FEATURE AVAILABILITY")
    print(f"{'='*80}")

    missing_features = [f for f in feature_columns if f not in df.columns]
    available_features = [f for f in feature_columns if f in df.columns]

    print(f"\n📊 Exit models: {len(available_features)}/{len(feature_columns)} available ({len(available_features)/len(feature_columns)*100:.1f}%)")
    if missing_features:
        print(f"\n⚠️ Missing features ({len(missing_features)}):")
        for f in missing_features[:10]:
            print(f"   - {f}")
        if len(missing_features) > 10:
            print(f"   ... and {len(missing_features)-10} more")
        raise ValueError(f"Missing {len(missing_features)} features")

    # Train/Val split
    print(f"\n{'='*80}")
    print("TRAIN/VALIDATION SPLIT")
    print(f"{'='*80}")

    train_mask = df['timestamp'] <= TRAIN_END_DATE
    val_mask = df['timestamp'] >= VAL_START_DATE

    df_train = df[train_mask].copy()
    df_val = df[val_mask].copy()

    print(f"\n📚 Training Set:")
    print(f"   Period: {df_train['timestamp'].min()} to {df_train['timestamp'].max()}")
    print(f"   Rows: {len(df_train):,}")
    print(f"   Days: {(df_train['timestamp'].max() - df_train['timestamp'].min()).days}")

    print(f"\n✅ Validation Set:")
    print(f"   Period: {df_val['timestamp'].min()} to {df_val['timestamp'].max()}")
    print(f"   Rows: {len(df_val):,}")
    print(f"   Days: {(df_val['timestamp'].max() - df_val['timestamp'].min()).days}")

    # Prepare features
    X_train = df_train[available_features]
    X_val = df_val[available_features]
    y_train_long = df_train['long_exit']
    y_val_long = df_val['long_exit']
    y_train_short = df_train['short_exit']
    y_val_short = df_val['short_exit']

    # Train LONG Exit model
    print(f"\n{'='*80}")
    print("1/2: TRAINING LONG EXIT MODEL")
    print(f"{'='*80}")

    # Scale features
    scaler_long = StandardScaler()
    X_train_long_scaled = pd.DataFrame(
        scaler_long.fit_transform(X_train),
        columns=available_features,
        index=X_train.index
    )
    X_val_long_scaled = pd.DataFrame(
        scaler_long.transform(X_val),
        columns=available_features,
        index=X_val.index
    )

    model_long = train_model_with_cv(
        X_train_long_scaled, y_train_long,
        X_val_long_scaled, y_val_long,
        "LONG Exit"
    )

    # Save LONG Exit model
    long_exit_model_path = MODELS_DIR / f"xgboost_long_exit_optimal_{TIMESTAMP}.pkl"
    long_exit_scaler_path = MODELS_DIR / f"xgboost_long_exit_optimal_{TIMESTAMP}_scaler.pkl"
    long_exit_features_path = MODELS_DIR / f"xgboost_long_exit_optimal_{TIMESTAMP}_features.txt"

    with open(long_exit_model_path, 'wb') as f:
        pickle.dump(model_long, f)
    joblib.dump(scaler_long, long_exit_scaler_path)
    with open(long_exit_features_path, 'w') as f:
        f.write('\n'.join(available_features))

    print(f"\n💾 LONG Exit model saved: {long_exit_model_path.name}")

    # Train SHORT Exit model
    print(f"\n{'='*80}")
    print("2/2: TRAINING SHORT EXIT MODEL")
    print(f"{'='*80}")

    # Scale features
    scaler_short = StandardScaler()
    X_train_short_scaled = pd.DataFrame(
        scaler_short.fit_transform(X_train),
        columns=available_features,
        index=X_train.index
    )
    X_val_short_scaled = pd.DataFrame(
        scaler_short.transform(X_val),
        columns=available_features,
        index=X_val.index
    )

    model_short = train_model_with_cv(
        X_train_short_scaled, y_train_short,
        X_val_short_scaled, y_val_short,
        "SHORT Exit"
    )

    # Save SHORT Exit model
    short_exit_model_path = MODELS_DIR / f"xgboost_short_exit_optimal_{TIMESTAMP}.pkl"
    short_exit_scaler_path = MODELS_DIR / f"xgboost_short_exit_optimal_{TIMESTAMP}_scaler.pkl"
    short_exit_features_path = MODELS_DIR / f"xgboost_short_exit_optimal_{TIMESTAMP}_features.txt"

    with open(short_exit_model_path, 'wb') as f:
        pickle.dump(model_short, f)
    joblib.dump(scaler_short, short_exit_scaler_path)
    with open(short_exit_features_path, 'w') as f:
        f.write('\n'.join(available_features))

    print(f"\n💾 SHORT Exit model saved: {short_exit_model_path.name}")

    # Summary
    print(f"\n{'='*80}")
    print("TRAINING COMPLETE")
    print(f"{'='*80}")

    print(f"\n✅ Exit models trained with 171 features (14× more than current 12)")
    print(f"   Timestamp: {TIMESTAMP}")

    print(f"\n📦 Models Saved:")
    print(f"   - {long_exit_model_path.name}")
    print(f"   - {short_exit_model_path.name}")

    print(f"\n{'='*80}")
    print("NEXT STEPS")
    print(f"{'='*80}")
    print(f"1. Backtest 90-day Full Set with improved Exit models")
    print(f"2. Compare vs current 52-day Exit models (12 features)")
    print(f"3. Deploy if performance superior")


if __name__ == "__main__":
    main()
