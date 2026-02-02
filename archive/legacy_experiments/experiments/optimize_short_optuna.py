"""
Optuna Hyperparameter Optimization for SHORT Trading
목표: SHORT win rate 60% 달성

전략:
1. 3-class 모델 하이퍼파라미터 최적화
2. SHORT 신호 품질 향상
3. Class imbalance 보완
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from datetime import datetime
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.utils.class_weight import compute_sample_weight
import optuna
from optuna.samplers import TPESampler

import sys
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.production.train_xgboost_improved_v3_phase2 import calculate_features
from scripts.production.advanced_technical_features import AdvancedTechnicalFeatures

# Directories
DATA_DIR = PROJECT_ROOT / "data" / "historical"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"


def create_labels_3class(df, lookahead=3, threshold=0.003):
    """Create 3-class labels"""
    labels = []
    for i in range(len(df)):
        if i >= len(df) - lookahead:
            labels.append(0)
            continue

        current_price = df['close'].iloc[i]
        future_prices = df['close'].iloc[i+1:i+1+lookahead]

        max_future = future_prices.max()
        min_future = future_prices.min()

        increase_pct = (max_future - current_price) / current_price
        decrease_pct = (current_price - min_future) / current_price

        if increase_pct >= threshold and increase_pct > decrease_pct:
            labels.append(1)  # LONG
        elif decrease_pct >= threshold and decrease_pct > increase_pct:
            labels.append(2)  # SHORT
        else:
            labels.append(0)  # NEUTRAL

    return np.array(labels)


def load_data():
    """Load and prepare data"""
    print("Loading data...")
    df = pd.read_csv(DATA_DIR / "BTCUSDT_5m_max.csv")

    # Calculate features
    df = calculate_features(df)
    adv_features = AdvancedTechnicalFeatures()
    df = adv_features.calculate_all_features(df)

    df = df.ffill().dropna()

    # Get feature columns
    baseline_features = ['rsi', 'macd', 'macd_signal', 'bb_upper', 'bb_lower',
                        'ema_21', 'ema_50', 'atr', 'volume_ratio', 'price_distance_ema']
    advanced_features = adv_features.get_feature_names()
    feature_columns = [f for f in baseline_features + advanced_features if f in df.columns]

    # Create labels
    labels = create_labels_3class(df, lookahead=3, threshold=0.003)

    X = df[feature_columns].values
    y = labels

    print(f"Data loaded: {len(X)} samples, {len(feature_columns)} features")

    # Class distribution
    unique, counts = np.unique(y, return_counts=True)
    for cls, cnt in zip(unique, counts):
        cls_name = ['NEUTRAL', 'LONG', 'SHORT'][cls]
        print(f"  {cls_name}: {cnt} ({cnt/len(y)*100:.1f}%)")

    return X, y, feature_columns


def backtest_short_performance(model, X, y):
    """
    Backtest SHORT performance using time series split
    Returns: SHORT win rate
    """
    tscv = TimeSeriesSplit(n_splits=5)
    short_wins = []
    short_total = []

    for train_idx, val_idx in tscv.split(X):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Fit with balanced weights
        sample_weights = compute_sample_weight('balanced', y_train)
        model.fit(X_train, y_train, sample_weight=sample_weights, verbose=False)

        # Predict on validation
        y_pred = model.predict(X_val)

        # Calculate SHORT metrics
        short_mask = (y_val == 2)
        if short_mask.sum() > 0:
            short_pred = y_pred[short_mask]
            short_true = y_val[short_mask]
            short_correct = (short_pred == short_true).sum()
            short_wins.append(short_correct)
            short_total.append(len(short_true))

    if sum(short_total) == 0:
        return 0.0

    short_win_rate = sum(short_wins) / sum(short_total)
    return short_win_rate


def objective(trial):
    """Optuna objective function"""

    # Hyperparameters to optimize
    params = {
        'n_estimators': trial.suggest_int('n_estimators', 100, 500),
        'max_depth': trial.suggest_int('max_depth', 3, 8),
        'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 5),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 5),
        'random_state': 42,
        'use_label_encoder': False,
        'eval_metric': 'mlogloss',
        'num_class': 3
    }

    model = xgb.XGBClassifier(**params)

    # Backtest SHORT performance
    short_win_rate = backtest_short_performance(model, X_global, y_global)

    # Calculate F1 scores for all classes
    tscv = TimeSeriesSplit(n_splits=3)
    f1_scores = []

    for train_idx, val_idx in tscv.split(X_global):
        X_train, X_val = X_global[train_idx], X_global[val_idx]
        y_train, y_val = y_global[train_idx], y_global[val_idx]

        sample_weights = compute_sample_weight('balanced', y_train)
        model.fit(X_train, y_train, sample_weight=sample_weights, verbose=False)

        y_pred = model.predict(X_val)
        f1 = f1_score(y_val, y_pred, average=None, zero_division=0)
        f1_scores.append(f1)

    avg_f1 = np.mean(f1_scores, axis=0)
    f1_short = avg_f1[2] if len(avg_f1) > 2 else 0

    # Objective: Maximize SHORT win rate + SHORT F1
    # 60% weight on win rate, 40% on F1
    score = 0.6 * short_win_rate + 0.4 * f1_short

    return score


def main():
    """Main optimization pipeline"""
    global X_global, y_global

    print("="*80)
    print("SHORT Strategy Optimization with Optuna")
    print("="*80)

    # Load data
    X_global, y_global, feature_columns = load_data()

    print(f"\nStarting optimization...")
    print(f"Objective: Maximize SHORT win rate + F1 score")
    print(f"Trials: 100")
    print(f"Expected time: 1-2 hours\n")

    # Create study
    study = optuna.create_study(
        direction='maximize',
        sampler=TPESampler(seed=42),
        study_name='short_optimization'
    )

    # Optimize
    study.optimize(objective, n_trials=100, show_progress_bar=True)

    # Results
    print("\n" + "="*80)
    print("Optimization Results")
    print("="*80)

    print(f"\nBest score: {study.best_value:.4f}")
    print(f"\nBest hyperparameters:")
    for key, value in study.best_params.items():
        print(f"  {key}: {value}")

    # Train final model with best params
    print("\n" + "="*80)
    print("Training Final Model with Best Parameters")
    print("="*80)

    best_params = study.best_params.copy()
    best_params.update({
        'random_state': 42,
        'use_label_encoder': False,
        'eval_metric': 'mlogloss',
        'num_class': 3
    })

    final_model = xgb.XGBClassifier(**best_params)

    # Train on all data
    sample_weights = compute_sample_weight('balanced', y_global)
    final_model.fit(X_global, y_global, sample_weight=sample_weights, verbose=False)

    # Evaluate
    y_pred = final_model.predict(X_global)
    f1 = f1_score(y_global, y_pred, average=None, zero_division=0)
    precision = precision_score(y_global, y_pred, average=None, zero_division=0)
    recall = recall_score(y_global, y_pred, average=None, zero_division=0)

    print(f"\nFinal Model Performance:")
    print(f"  NEUTRAL - F1: {f1[0]:.3f}, Precision: {precision[0]:.3f}, Recall: {recall[0]:.3f}")
    print(f"  LONG    - F1: {f1[1]:.3f}, Precision: {precision[1]:.3f}, Recall: {recall[1]:.3f}")
    print(f"  SHORT   - F1: {f1[2]:.3f}, Precision: {precision[2]:.3f}, Recall: {recall[2]:.3f}")

    # Backtest SHORT
    short_win_rate = backtest_short_performance(final_model, X_global, y_global)
    print(f"\nSHORT Win Rate (Cross-validation): {short_win_rate*100:.1f}%")

    # Save model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"xgboost_v4_short_optimized_{timestamp}"
    model_path = MODELS_DIR / f"{model_name}.pkl"

    with open(model_path, 'wb') as f:
        pickle.dump(final_model, f)

    # Save features
    feature_path = MODELS_DIR / f"{model_name}_features.txt"
    with open(feature_path, 'w') as f:
        for feature in feature_columns:
            f.write(f"{feature}\n")

    # Save metadata
    import json
    metadata = {
        "model_name": model_name,
        "model_type": "3-class optimized",
        "optimization_method": "Optuna TPE",
        "n_trials": 100,
        "best_score": float(study.best_value),
        "best_params": study.best_params,
        "n_features": len(feature_columns),
        "short_win_rate": float(short_win_rate),
        "f1_neutral": float(f1[0]),
        "f1_long": float(f1[1]),
        "f1_short": float(f1[2]),
        "timestamp": datetime.now().isoformat()
    }

    metadata_path = MODELS_DIR / f"{model_name}_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "="*80)
    print("Model Saved")
    print("="*80)
    print(f"Model: {model_path}")
    print(f"Features: {feature_path}")
    print(f"Metadata: {metadata_path}")

    # Decision
    print("\n" + "="*80)
    print("Decision")
    print("="*80)

    if short_win_rate >= 0.60:
        print(f"✅ SUCCESS! SHORT win rate {short_win_rate*100:.1f}% >= 60%")
        print(f"✅ Proceed to backtest with optimized model")
    elif short_win_rate >= 0.55:
        print(f"⚠️ MARGINAL: SHORT win rate {short_win_rate*100:.1f}% (55-60%)")
        print(f"⚠️ Consider additional optimization or accept with caution")
    else:
        print(f"❌ FAILED: SHORT win rate {short_win_rate*100:.1f}% < 55%")
        print(f"❌ Consider feature engineering or different timeframe")

    return study, final_model, metadata


if __name__ == "__main__":
    study, model, metadata = main()
