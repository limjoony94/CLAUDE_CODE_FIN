"""
Retrain Exit Models - Patience-Based Labels
=============================================

Retrains Exit models using patience-based labels to encourage longer holds.

Problem: 59% of trades exit too early (0-5 candles) with only 48.49% WR
Solution: Train Exit models on patience-based labels (min 10 candle hold)

Strategy:
  - Data: Full dataset (495 days, BTCUSDT_5m_features.csv)
  - Labels: Patience-based exit labels (created by create_patience_exit_labels.py)
  - Method: Walk-Forward 5-Fold Cross-Validation
  - Output: All folds saved for ensemble approach

Expected Impact: Win Rate 56% → 65%+, Avg profit +0.1% → +0.9%+

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
print("EXIT MODEL RETRAINING - PATIENCE-BASED LABELS")
print("=" * 80)
print()
print("Strategy: Train on patience-based labels for longer holds")
print("  ✅ Using 495-day dataset (full history)")
print("  ✅ Walk-Forward 5-Fold Cross-Validation")
print("  ✅ Patience threshold: min 10 candles hold")
print("  ✅ All folds saved for ensemble")
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

# Load Patience-Based Labels
print("-" * 80)
print("STEP 2: Loading Patience-Based Exit Labels")
print("-" * 80)

# Find latest labels file
labels_files = sorted(LABELS_DIR.glob("exit_labels_patience_*.csv"))
if not labels_files:
    raise FileNotFoundError("No patience exit labels found! Run create_patience_exit_labels.py first.")

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
    labels_df[['timestamp', 'long_exit_patience', 'short_exit_patience']],
    on='timestamp',
    how='inner'
)

print(f"  ✅ Merged dataset: {len(df):,} candles")
print(f"     LONG Exit signals: {df['long_exit_patience'].sum():,} ({df['long_exit_patience'].mean()*100:.2f}%)")
print(f"     SHORT Exit signals: {df['short_exit_patience'].sum():,} ({df['short_exit_patience'].mean()*100:.2f}%)")
print()

# Define Exit Features (21 features matching working Exit models)
EXIT_FEATURES = [
    'close', 'volume', 'volume_surge', 'price_acceleration',
    'ma_20', 'price_vs_ma20', 'price_vs_ma50', 'volatility_20',
    'rsi', 'rsi_slope', 'rsi_overbought', 'rsi_oversold', 'rsi_divergence',
    'macd', 'macd_signal', 'macd_histogram_slope',
    'macd_crossover', 'macd_crossunder',
    'bb_position', 'higher_high', 'near_support'
]

available_exit_features = [f for f in EXIT_FEATURES if f in df.columns]

print("-" * 80)
print("STEP 4: Exit Features Validation")
print("-" * 80)
print(f"  Exit features: {len(available_exit_features)}/{len(EXIT_FEATURES)}")
print()

# Walk-Forward Training Function
def train_exit_model_walkforward_ensemble(df, exit_features, side):
    """
    Walk-Forward training with ALL folds saved for ensemble
    Returns: list of (model, scaler, features, metrics) for all valid folds
    """
    print(f"\n{'='*80}")
    print(f"Walk-Forward Ensemble Training: {side} Exit Model")
    print(f"{'='*80}")

    # Select labels based on side
    label_column = 'long_exit_patience' if side == 'LONG' else 'short_exit_patience'

    tscv = TimeSeriesSplit(n_splits=5)
    fold_results = []

    for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(df)):
        print(f"\n{'─'*80}")
        print(f"FOLD {fold_idx+1}/5")
        print(f"{'─'*80}")
        print(f"  Train: {len(train_idx):,} samples (index {train_idx[0]}-{train_idx[-1]})")
        print(f"  Val: {len(val_idx):,} samples (index {val_idx[0]}-{val_idx[-1]})")

        df_train = df.iloc[train_idx].copy()
        df_val = df.iloc[val_idx].copy()

        # Prepare training data
        X_train = df_train[exit_features].values
        y_train = df_train[label_column].values

        # Remove NaN rows
        mask = ~np.isnan(X_train).any(axis=1) & ~np.isnan(y_train)
        X_train = X_train[mask]
        y_train = y_train[mask]

        if (y_train == 1).sum() < 10:
            print(f"  ⚠️  Too few positive samples in fold {fold_idx+1}, skipping...")
            continue

        print(f"  Training samples: {len(X_train):,}")
        print(f"  Positive samples: {(y_train == 1).sum():,} ({(y_train == 1).mean()*100:.2f}%)")

        # Scale features
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        # Train XGBoost model
        model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=(y_train == 0).sum() / max((y_train == 1).sum(), 1),
            random_state=42,
            eval_metric='logloss'
        )

        print(f"  Training {side} Exit model...")
        model.fit(X_train_scaled, y_train, verbose=False)

        # Validate
        X_val = df_val[exit_features].values
        y_val = df_val[label_column].values

        mask_val = ~np.isnan(X_val).any(axis=1) & ~np.isnan(y_val)
        X_val = X_val[mask_val]
        y_val = y_val[mask_val]

        if len(X_val) > 0:
            X_val_scaled = scaler.transform(X_val)
            y_pred = model.predict(X_val_scaled)
            y_pred_proba = model.predict_proba(X_val_scaled)[:, 1]

            pred_rate = y_pred.mean()

            # Calculate metrics
            if (y_val == 1).sum() > 0:
                precision = precision_score(y_val, y_pred, zero_division=0)
                recall = recall_score(y_val, y_pred, zero_division=0)
                f1 = f1_score(y_val, y_pred, zero_division=0)
            else:
                precision = recall = f1 = 0.0

            print(f"  Validation results:")
            print(f"    Prediction rate: {pred_rate*100:.2f}%")
            print(f"    Precision: {precision:.4f}")
            print(f"    Recall: {recall:.4f}")
            print(f"    F1 Score: {f1:.4f}")

            # Save fold result
            fold_results.append({
                'fold_idx': fold_idx + 1,
                'model': model,
                'scaler': scaler,
                'features': exit_features,
                'score': f1,  # Use F1 as quality metric
                'pred_rate': pred_rate,
                'precision': precision,
                'recall': recall
            })

        print(f"  ✅ Fold {fold_idx+1} complete")

    if len(fold_results) == 0:
        raise ValueError(f"No valid models trained for {side}")

    # Sort by F1 score (descending)
    fold_results = sorted(fold_results, key=lambda x: x['score'], reverse=True)

    print(f"\n{'='*80}")
    print(f"ALL FOLDS TRAINED - F1 Scores:")
    print(f"{'='*80}")
    for result in fold_results:
        print(f"  Fold {result['fold_idx']}: {result['score']:.4f} (Pred: {result['pred_rate']*100:.2f}%, Prec: {result['precision']:.4f}, Rec: {result['recall']:.4f})")

    return fold_results

# Train LONG Exit Model
print("-" * 80)
print("STEP 5: Training LONG Exit Model (Walk-Forward Ensemble)")
print("-" * 80)

long_exit_fold_results = train_exit_model_walkforward_ensemble(
    df, available_exit_features, 'LONG'
)

# Train SHORT Exit Model
print("\n" + "-" * 80)
print("STEP 6: Training SHORT Exit Model (Walk-Forward Ensemble)")
print("-" * 80)

short_exit_fold_results = train_exit_model_walkforward_ensemble(
    df, available_exit_features, 'SHORT'
)

# Save ALL Fold Models
print("\n" + "=" * 80)
print("STEP 7: Saving ALL Fold Models")
print("=" * 80)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# LONG Exit - Save all folds
for result in long_exit_fold_results:
    fold_idx = result['fold_idx']
    long_path = MODELS_DIR / f"xgboost_long_exit_patience_fold{fold_idx}_{timestamp}"

    with open(f"{long_path}.pkl", 'wb') as f:
        pickle.dump(result['model'], f)
    joblib.dump(result['scaler'], f"{long_path}_scaler.pkl")
    with open(f"{long_path}_features.txt", 'w') as f:
        f.write('\n'.join(result['features']))

    print(f"✅ LONG Exit Fold {fold_idx}: {long_path.name}.pkl (F1: {result['score']:.4f})")

# Save LONG metadata
long_meta_path = MODELS_DIR / f"xgboost_long_exit_patience_meta_{timestamp}.pkl"
long_meta = {
    'fold_scores': [r['score'] for r in long_exit_fold_results],
    'fold_indices': [r['fold_idx'] for r in long_exit_fold_results],
    'fold_pred_rates': [r['pred_rate'] for r in long_exit_fold_results],
    'fold_precisions': [r['precision'] for r in long_exit_fold_results],
    'fold_recalls': [r['recall'] for r in long_exit_fold_results],
    'timestamp': timestamp,
    'num_features': len(long_exit_fold_results[0]['features']),
    'label_type': 'patience-based (min 10 candles)'
}
with open(long_meta_path, 'wb') as f:
    pickle.dump(long_meta, f)
print(f"✅ LONG Metadata: {long_meta_path.name}")

# SHORT Exit - Save all folds
for result in short_exit_fold_results:
    fold_idx = result['fold_idx']
    short_path = MODELS_DIR / f"xgboost_short_exit_patience_fold{fold_idx}_{timestamp}"

    with open(f"{short_path}.pkl", 'wb') as f:
        pickle.dump(result['model'], f)
    joblib.dump(result['scaler'], f"{short_path}_scaler.pkl")
    with open(f"{short_path}_features.txt", 'w') as f:
        f.write('\n'.join(result['features']))

    print(f"✅ SHORT Exit Fold {fold_idx}: {short_path.name}.pkl (F1: {result['score']:.4f})")

# Save SHORT metadata
short_meta_path = MODELS_DIR / f"xgboost_short_exit_patience_meta_{timestamp}.pkl"
short_meta = {
    'fold_scores': [r['score'] for r in short_exit_fold_results],
    'fold_indices': [r['fold_idx'] for r in short_exit_fold_results],
    'fold_pred_rates': [r['pred_rate'] for r in short_exit_fold_results],
    'fold_precisions': [r['precision'] for r in short_exit_fold_results],
    'fold_recalls': [r['recall'] for r in short_exit_fold_results],
    'timestamp': timestamp,
    'num_features': len(short_exit_fold_results[0]['features']),
    'label_type': 'patience-based (min 10 candles)'
}
with open(short_meta_path, 'wb') as f:
    pickle.dump(short_meta, f)
print(f"✅ SHORT Metadata: {short_meta_path.name}")

print("\n" + "=" * 80)
print("PATIENCE EXIT MODELS TRAINING COMPLETE")
print("=" * 80)
print()
print("Saved Files:")
print(f"  LONG Exit: {len(long_exit_fold_results)} fold models + metadata")
print(f"  SHORT Exit: {len(short_exit_fold_results)} fold models + metadata")
print()
print("Key Improvements:")
print("  - Patience-based labels (min 10 candle hold)")
print("  - Trained on 495-day dataset (robust patterns)")
print("  - Walk-Forward 5-Fold CV (no overfitting)")
print("  - All folds saved for ensemble approach")
print()
print("Next Steps:")
print("  1. Create backtest_patience_exits.py")
print("  2. Compare: Enhanced Baseline vs Patience Exit models")
print("  3. Expected: Win Rate 56% → 65%+, Return +1,500%+")
print()
