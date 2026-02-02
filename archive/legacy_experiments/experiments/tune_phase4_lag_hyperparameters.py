"""
Hyperparameter Tuning for Phase 4 Lag Features

CRITICAL ISSUE IDENTIFIED:
- Added 185 features (up from 37)
- Did NOT adjust XGBoost hyperparameters
- Used same params designed for 37 features → underfitting!

Solution:
- Tune hyperparameters for 185 temporal features
- Test deeper trees, feature sampling, regularization
- Find optimal configuration for lag features

Expected Improvement:
- Proper params may unlock lag features' potential
- F1: 0.046 → 0.10+ (with tuning)
- Returns: 2.38% → 7-10% per 5 days
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import sys
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import f1_score, make_scorer
import xgboost as xgb
from loguru import logger

# Add parent directory to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.production.train_xgboost_improved_v3_phase2 import calculate_features
from scripts.production.advanced_technical_features import AdvancedTechnicalFeatures
from scripts.production.lag_features import LagFeatureGenerator

DATA_DIR = PROJECT_ROOT / "data" / "historical"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"


def load_and_prepare_data():
    """Load data and create temporal features"""
    logger.info("="*80)
    logger.info("Hyperparameter Tuning for Lag Features")
    logger.info("="*80)

    # Load data
    data_file = DATA_DIR / "BTCUSDT_5m_max.csv"
    df = pd.read_csv(data_file)
    logger.info(f"Raw data: {len(df)} candles")

    # Calculate baseline features
    logger.info("Calculating baseline features...")
    df = calculate_features(df)

    # Calculate advanced features
    logger.info("Calculating advanced technical features...")
    adv_features = AdvancedTechnicalFeatures(lookback_sr=50, lookback_trend=20)
    df = adv_features.calculate_all_features(df)

    # Create lag features
    logger.info("Creating lag features...")
    base_feature_columns = [
        'close_change_1', 'close_change_3', 'volume_ma_ratio',
        'rsi', 'macd', 'macd_signal', 'macd_diff',
        'bb_high', 'bb_mid', 'bb_low'
    ] + adv_features.get_feature_names()

    lag_gen = LagFeatureGenerator(lag_periods=[1, 2])
    df_temporal, temporal_columns = lag_gen.create_all_temporal_features(
        df, base_feature_columns, include_momentum=True
    )

    # Handle NaN
    df_temporal = df_temporal.ffill()
    df_temporal = df_temporal.dropna()

    logger.success(f"✅ Data prepared: {len(df_temporal)} samples, {len(temporal_columns)} features")

    return df_temporal, temporal_columns


def create_labels(df, lookahead=3, threshold=0.003):
    """Create labels"""
    labels = []

    for i in range(len(df)):
        if i >= len(df) - lookahead:
            labels.append(0)
            continue

        current_price = df['close'].iloc[i]
        future_prices = df['close'].iloc[i+1:i+1+lookahead]
        max_future_price = future_prices.max()

        price_increase_pct = (max_future_price - current_price) / current_price

        if price_increase_pct >= threshold:
            labels.append(1)
        else:
            labels.append(0)

    return np.array(labels)


def tune_hyperparameters(X, y):
    """
    Hyperparameter tuning for 185 features

    Focus areas:
    1. Tree depth (max_depth) - Need deeper trees for 185 features
    2. Feature sampling (colsample_bytree) - Prevent overfitting with many features
    3. Regularization (min_child_weight, gamma) - Control complexity
    4. Number of trees (n_estimators) - More iterations to learn patterns
    """

    logger.info("\n" + "="*80)
    logger.info("Hyperparameter Search Space")
    logger.info("="*80)

    # Calculate class weight
    scale_pos_weight = (len(y) - np.sum(y)) / np.sum(y)

    # Search space designed for 185 features
    param_distributions = {
        # Tree complexity (CRITICAL: deeper for 185 features)
        'max_depth': [6, 8, 10, 12],  # Was 6 → test 8-12

        # Feature sampling (CRITICAL: lower for 185 features to prevent overfitting)
        'colsample_bytree': [0.3, 0.5, 0.7],  # Was 0.8 → test 0.3-0.7
        'colsample_bylevel': [0.5, 0.7, 0.9],  # NEW: per-level sampling

        # Number of trees (more iterations for complex feature space)
        'n_estimators': [300, 500, 700],  # Was 300 → test 500-700

        # Learning rate (balance with n_estimators)
        'learning_rate': [0.03, 0.05, 0.1],  # Was 0.05 → test range

        # Regularization (prevent overfitting with many features)
        'min_child_weight': [1, 3, 5],  # Was implicit 1 → test 1-5
        'gamma': [0, 0.1, 0.3],  # NEW: minimum loss reduction

        # Sampling (prevent overfitting)
        'subsample': [0.7, 0.8, 0.9],  # Was 0.8 → test range
    }

    logger.info("\nParameter ranges:")
    for param, values in param_distributions.items():
        logger.info(f"  {param}: {values}")

    # Base model
    base_model = xgb.XGBClassifier(
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric='logloss',
        tree_method='hist'  # Faster for many features
    )

    # F1 scorer
    f1_scorer = make_scorer(f1_score, zero_division=0)

    # Time series cross-validation
    tscv = TimeSeriesSplit(n_splits=3)  # Reduced for speed

    logger.info(f"\nRunning RandomizedSearchCV:")
    logger.info(f"  Iterations: 30")
    logger.info(f"  CV folds: 3")
    logger.info(f"  Scoring: F1 Score")
    logger.info(f"  This may take 10-20 minutes...")

    # Randomized search (faster than GridSearch)
    random_search = RandomizedSearchCV(
        estimator=base_model,
        param_distributions=param_distributions,
        n_iter=30,  # Test 30 random combinations
        scoring=f1_scorer,
        cv=tscv,
        verbose=2,
        random_state=42,
        n_jobs=-1  # Use all CPUs
    )

    # Fit
    random_search.fit(X, y)

    # Results
    logger.info("\n" + "="*80)
    logger.info("Hyperparameter Tuning Results")
    logger.info("="*80)

    logger.success(f"\n✅ Best F1 Score: {random_search.best_score_:.4f}")
    logger.info(f"\nBest Parameters:")
    for param, value in random_search.best_params_.items():
        logger.info(f"  {param}: {value}")

    # Compare to default params
    logger.info(f"\nComparison to Default (37 features params):")
    logger.info(f"  max_depth: 6 → {random_search.best_params_.get('max_depth', 6)}")
    logger.info(f"  colsample_bytree: 0.8 → {random_search.best_params_.get('colsample_bytree', 0.8):.2f}")
    logger.info(f"  n_estimators: 300 → {random_search.best_params_.get('n_estimators', 300)}")
    logger.info(f"  learning_rate: 0.05 → {random_search.best_params_.get('learning_rate', 0.05):.3f}")

    # Top 10 configurations
    results_df = pd.DataFrame(random_search.cv_results_)
    results_df = results_df.sort_values('rank_test_score')

    logger.info(f"\nTop 10 Configurations:")
    for idx, row in results_df.head(10).iterrows():
        logger.info(f"\n  Rank {int(row['rank_test_score'])}:")
        logger.info(f"    F1: {row['mean_test_score']:.4f} ± {row['std_test_score']:.4f}")
        logger.info(f"    Params: {row['params']}")

    # Save results
    results_file = RESULTS_DIR / "hyperparameter_tuning_lag_features.csv"
    results_df.to_csv(results_file, index=False)
    logger.success(f"\n✅ Full results saved: {results_file}")

    return random_search.best_estimator_, random_search.best_params_, random_search.best_score_


def train_final_model_with_best_params(X, y, best_params):
    """Train final model with best parameters on all data"""

    logger.info("\n" + "="*80)
    logger.info("Training Final Model with Best Parameters")
    logger.info("="*80)

    # Calculate class weight
    scale_pos_weight = (len(y) - np.sum(y)) / np.sum(y)

    # Create model with best params
    final_model = xgb.XGBClassifier(
        **best_params,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric='logloss',
        tree_method='hist'
    )

    logger.info(f"Training on {len(X)} samples...")
    final_model.fit(X, y, verbose=False)

    # Evaluate
    y_pred = final_model.predict(X)
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    accuracy = accuracy_score(y, y_pred)
    precision = precision_score(y, y_pred, zero_division=0)
    recall = recall_score(y, y_pred, zero_division=0)
    f1 = f1_score(y, y_pred, zero_division=0)

    logger.info(f"\nFinal Model Performance (Training Set):")
    logger.info(f"  Accuracy: {accuracy:.3f}")
    logger.info(f"  Precision: {precision:.3f}")
    logger.info(f"  Recall: {recall:.3f}")
    logger.info(f"  F1 Score: {f1:.3f}")

    # Compare to original lag model
    logger.info(f"\nComparison to Original Lag Model:")
    logger.info(f"  Original F1: 0.046 (untuned)")
    logger.info(f"  Tuned F1: {f1:.3f}")
    logger.info(f"  Improvement: {((f1 - 0.046) / 0.046) * 100:+.1f}%")

    return final_model


def save_tuned_model(model, feature_columns, best_params, best_score):
    """Save tuned model"""

    model_name = "xgboost_v4_phase4_lag_tuned_lookahead3_thresh3"
    model_path = MODELS_DIR / f"{model_name}.pkl"
    feature_path = MODELS_DIR / f"{model_name}_features.txt"
    params_path = MODELS_DIR / f"{model_name}_params.json"

    # Save model
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    # Save features
    with open(feature_path, 'w') as f:
        for feature in feature_columns:
            f.write(f"{feature}\n")

    # Save params
    import json
    params_metadata = {
        "best_params": best_params,
        "cv_f1_score": float(best_score),
        "n_features": len(feature_columns),
        "has_lag_features": True,
        "tuned": True
    }

    with open(params_path, 'w') as f:
        json.dump(params_metadata, f, indent=2)

    logger.info("\n" + "="*80)
    logger.info("Model Saved")
    logger.info("="*80)
    logger.info(f"Model: {model_path}")
    logger.info(f"Features: {feature_path}")
    logger.info(f"Params: {params_path}")

    return model_path


def main():
    """Main tuning pipeline"""

    # Load data
    df, temporal_columns = load_and_prepare_data()

    # Create labels
    lookahead = 3
    threshold = 0.003

    logger.info(f"\nCreating labels (lookahead={lookahead}, threshold={threshold*100}%)...")
    labels = create_labels(df, lookahead=lookahead, threshold=threshold)

    # Prepare features
    X = df[temporal_columns].values
    y = labels

    logger.info(f"\nDataset:")
    logger.info(f"  Samples: {len(X)}")
    logger.info(f"  Features: {len(temporal_columns)}")
    logger.info(f"  Positive: {np.sum(y)} ({np.mean(y)*100:.1f}%)")

    # Hyperparameter tuning
    best_model, best_params, best_score = tune_hyperparameters(X, y)

    # Train final model with best params
    final_model = train_final_model_with_best_params(X, y, best_params)

    # Save
    model_path = save_tuned_model(final_model, temporal_columns, best_params, best_score)

    logger.success("\n" + "="*80)
    logger.success("✅ Hyperparameter Tuning Complete!")
    logger.success("="*80)
    logger.info(f"\nNext Steps:")
    logger.info(f"1. Backtest tuned model")
    logger.info(f"2. Compare: Base (7.68%) vs Untuned Lag (2.38%) vs Tuned Lag (?)")
    logger.info(f"3. If tuned lag > base → Deploy to testnet")
    logger.info(f"\n비판적 검증: 파라미터 튜닝이 lag features의 성능을 향상시켰는가?")


if __name__ == "__main__":
    main()
