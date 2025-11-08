"""
Retrain Exit Models - Peak/Trough Labels
==========================================

Retrains Exit models using peak/trough-based labels.

Strategy:
  - Data: Full dataset (495 days, BTCUSDT_5m_features_enhanced_exit.csv)
  - Labels: Peak/Trough exit labels (created by create_peak_trough_exit_labels.py)
  - Method: Walk-Forward 5-Fold Cross-Validation
  - Output: Best fold selected based on precision/recall balance

Expected Impact: More optimal exit timing, better profits per trade

Created: 2025-10-30
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import pickle
import joblib
from datetime import datetime
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import precision_score, recall_score, f1_score

# Configuration
LEVERAGE = 4
EMERGENCY_STOP_LOSS = -0.03
EMERGENCY_MAX_HOLD = 120

DATA_DIR = PROJECT_ROOT / "data" / "features"
LABELS_DIR = PROJECT_ROOT / "data" / "labels"
MODELS_DIR = PROJECT_ROOT / "models"

print("=" * 80)
print("EXIT MODEL RETRAINING - PEAK/TROUGH LABELS")
print("=" * 80)
print()
print("Strategy: Train on peak/trough-based labels for optimal exit timing")
print("  ✅ Using full dataset (104 days)")
print("  ✅ Walk-Forward 5-Fold Cross-Validation")
print("  ✅ Best fold selected by precision/recall balance")
print()

# Load Full Features Dataset
print("-" * 80)
print("STEP 1: Loading Full Dataset")
print("-" * 80)

df = pd.read_csv(DATA_DIR / "BTCUSDT_5m_features_enhanced_exit.csv")
df['timestamp'] = pd.to_datetime(df['timestamp'])

print(f"✅ Features loaded: {len(df):,} candles")
print(f"   Date range: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")
print(f"   Days: {(df['timestamp'].max() - df['timestamp'].min()).days}")
print()

# Load Peak/Trough Labels
print("-" * 80)
print("STEP 2: Loading Peak/Trough Exit Labels")
print("-" * 80)

# Find latest labels file
labels_files = sorted(LABELS_DIR.glob("exit_labels_peak_trough_*.csv"))
if not labels_files:
    raise FileNotFoundError("No peak/trough exit labels found! Run create_peak_trough_exit_labels.py first.")

latest_labels_file = labels_files[-1]
print(f"  Loading: {latest_labels_file.name}")

labels_df = pd.read_csv(latest_labels_file)
labels_df['timestamp'] = pd.to_datetime(labels_df['timestamp'])

print(f"  ✅ Labels loaded: {len(labels_df):,} candles")
print()

# Merge features with labels (on timestamp)
print("-" * 80)
print("STEP 3: Merging Features with Labels")
print("-" * 80)

# Keep only overlapping timestamps
df = df.merge(
    labels_df[['timestamp', 'long_exit_peak', 'short_exit_trough']],
    on='timestamp',
    how='inner'
)

print(f"  ✅ Merged dataset: {len(df):,} candles")
print(f"  LONG Exit signals: {df['long_exit_peak'].sum():,} ({df['long_exit_peak'].mean()*100:.2f}%)")
print(f"  SHORT Exit signals: {df['short_exit_trough'].sum():,} ({df['short_exit_trough'].mean()*100:.2f}%)")
print()

# Prepare Features
print("-" * 80)
print("STEP 4: Preparing Features")
print("-" * 80)

# Exit feature columns (27 features)
exit_feature_columns = [
    'rsi_14', 'rsi_9', 'rsi_25',
    'macd', 'macd_signal', 'macd_histogram',
    'bb_upper', 'bb_middle', 'bb_lower', 'bb_width',
    'ema_9', 'ema_21', 'ema_50',
    'volume_sma_20',
    'atr_14',
    'volume_surge', 'price_acceleration',
    'price_vs_ma20', 'price_vs_ma50',
    'volatility_20',
    'rsi_slope', 'rsi_overbought', 'rsi_oversold', 'rsi_divergence',
    'macd_histogram_slope', 'macd_crossover', 'macd_crossunder',
    'bb_position', 'higher_high', 'near_support'
]

# Verify all features exist
missing_features = [f for f in exit_feature_columns if f not in df.columns]
if missing_features:
    print(f"⚠️ Missing features: {missing_features}")
    print("  Using available features only...")
    exit_feature_columns = [f for f in exit_feature_columns if f in df.columns]

print(f"  ✅ Features: {len(exit_feature_columns)}")
print()

# Split features and labels
X = df[exit_feature_columns].values
y_long = df['long_exit_peak'].values
y_short = df['short_exit_trough'].values

print(f"  Features shape: {X.shape}")
print(f"  LONG labels shape: {y_long.shape} (signal rate: {y_long.mean()*100:.2f}%)")
print(f"  SHORT labels shape: {y_short.shape} (signal rate: {y_short.mean()*100:.2f}%)")
print()

# Train LONG Exit Model
print("=" * 80)
print("STEP 5: Training LONG Exit Model (Walk-Forward 5-Fold)")
print("=" * 80)
print()

tscv = TimeSeriesSplit(n_splits=5)
best_long_model = None
best_long_scaler = None
best_long_score = -1
best_long_fold = -1

for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
    print(f"Fold {fold_idx}/5:")

    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y_long[train_idx], y_long[val_idx]

    # Scale features
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Train XGBoost
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss',
        use_label_encoder=False
    )

    model.fit(X_train_scaled, y_train, verbose=False)

    # Evaluate
    y_pred = model.predict(X_val_scaled)
    precision = precision_score(y_val, y_pred, zero_division=0)
    recall = recall_score(y_val, y_pred, zero_division=0)
    f1 = f1_score(y_val, y_pred, zero_division=0)

    # Composite score (balance precision and recall)
    composite_score = (precision + recall) / 2

    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1: {f1:.4f}")
    print(f"  Composite: {composite_score:.4f}")
    print()

    # Track best fold
    if composite_score > best_long_score:
        best_long_score = composite_score
        best_long_model = model
        best_long_scaler = scaler
        best_long_fold = fold_idx

print(f"✅ Best LONG Exit model: Fold {best_long_fold} (Composite Score: {best_long_score:.4f})")
print()

# Train SHORT Exit Model
print("=" * 80)
print("STEP 6: Training SHORT Exit Model (Walk-Forward 5-Fold)")
print("=" * 80)
print()

best_short_model = None
best_short_scaler = None
best_short_score = -1
best_short_fold = -1

for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
    print(f"Fold {fold_idx}/5:")

    X_train, X_val = X[train_idx], X[val_idx]
    y_train, y_val = y_short[train_idx], y_short[val_idx]

    # Scale features
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Train XGBoost
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss',
        use_label_encoder=False
    )

    model.fit(X_train_scaled, y_train, verbose=False)

    # Evaluate
    y_pred = model.predict(X_val_scaled)
    precision = precision_score(y_val, y_pred, zero_division=0)
    recall = recall_score(y_val, y_pred, zero_division=0)
    f1 = f1_score(y_val, y_pred, zero_division=0)

    # Composite score (balance precision and recall)
    composite_score = (precision + recall) / 2

    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1: {f1:.4f}")
    print(f"  Composite: {composite_score:.4f}")
    print()

    # Track best fold
    if composite_score > best_short_score:
        best_short_score = composite_score
        best_short_model = model
        best_short_scaler = scaler
        best_short_fold = fold_idx

print(f"✅ Best SHORT Exit model: Fold {best_short_fold} (Composite Score: {best_short_score:.4f})")
print()

# Save Models
print("=" * 80)
print("STEP 7: Saving Models")
print("=" * 80)
print()

timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")

# Save LONG Exit Model
long_model_path = MODELS_DIR / f"xgboost_long_exit_peak_trough_{timestamp_str}.pkl"
long_scaler_path = MODELS_DIR / f"xgboost_long_exit_peak_trough_{timestamp_str}_scaler.pkl"
long_features_path = MODELS_DIR / f"xgboost_long_exit_peak_trough_{timestamp_str}_features.txt"

with open(long_model_path, 'wb') as f:
    pickle.dump(best_long_model, f)

joblib.dump(best_long_scaler, long_scaler_path)

with open(long_features_path, 'w') as f:
    for feat in exit_feature_columns:
        f.write(f"{feat}\n")

print(f"✅ LONG Exit model saved:")
print(f"   Model: {long_model_path.name}")
print(f"   Scaler: {long_scaler_path.name}")
print(f"   Features: {long_features_path.name}")
print(f"   Best Fold: {best_long_fold}/5")
print(f"   Composite Score: {best_long_score:.4f}")
print()

# Save SHORT Exit Model
short_model_path = MODELS_DIR / f"xgboost_short_exit_peak_trough_{timestamp_str}.pkl"
short_scaler_path = MODELS_DIR / f"xgboost_short_exit_peak_trough_{timestamp_str}_scaler.pkl"
short_features_path = MODELS_DIR / f"xgboost_short_exit_peak_trough_{timestamp_str}_features.txt"

with open(short_model_path, 'wb') as f:
    pickle.dump(best_short_model, f)

joblib.dump(best_short_scaler, short_scaler_path)

with open(short_features_path, 'w') as f:
    for feat in exit_feature_columns:
        f.write(f"{feat}\n")

print(f"✅ SHORT Exit model saved:")
print(f"   Model: {short_model_path.name}")
print(f"   Scaler: {short_scaler_path.name}")
print(f"   Features: {short_features_path.name}")
print(f"   Best Fold: {best_short_fold}/5")
print(f"   Composite Score: {best_short_score:.4f}")
print()

# Summary
print("=" * 80)
print("TRAINING COMPLETE")
print("=" * 80)
print()
print("Models Trained:")
print(f"  ✅ LONG Exit (Peak-based)")
print(f"     Timestamp: {timestamp_str}")
print(f"     Best Fold: {best_long_fold}/5")
print(f"     Composite Score: {best_long_score:.4f}")
print()
print(f"  ✅ SHORT Exit (Trough-based)")
print(f"     Timestamp: {timestamp_str}")
print(f"     Best Fold: {best_short_fold}/5")
print(f"     Composite Score: {best_short_score:.4f}")
print()
print("Next Steps:")
print("  1. Run backtest: python scripts/experiments/backtest_peak_trough_exits.py")
print("  2. Deploy to production if backtest results are good")
print()
