"""
XGBoost Phase 4 Training with Lag Features for Temporal Pattern Learning

Problem Identified:
- Current XGBoost only sees 1 candle (current timepoint's aggregated features)
- Cannot learn temporal patterns like "RSI rising for 3 consecutive candles"

Solution:
- Add lag features (t-1, t-2) for all 37 base features
- Add momentum features (rate of change between timepoints)
- Enable temporal pattern learning in XGBoost

Expected Improvement:
- Win rate: 69.1% → 75-80%
- Returns: 7.68% → 9-10% per 5 days
- Better momentum and trend recognition

Feature Expansion:
- Base features: 37
- Lag features: 37 × 2 lags = 74
- Momentum features: 37 × 2 = 74
- Total: 185 temporal features
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from datetime import datetime
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from loguru import logger

# Import baseline features
import sys
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.production.train_xgboost_improved_v3_phase2 import calculate_features
from scripts.production.advanced_technical_features import AdvancedTechnicalFeatures
from scripts.production.lag_features import LagFeatureGenerator, validate_temporal_features

# Directories
DATA_DIR = PROJECT_ROOT / "data" / "historical"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_and_prepare_data():
    """Load data and calculate all features including lag features"""
    logger.info("="*80)
    logger.info("XGBoost Phase 4: Temporal Features with Lags")
    logger.info("="*80)
    logger.info("\nLoading data...")

    data_file = DATA_DIR / "BTCUSDT_5m_max.csv"
    df = pd.read_csv(data_file)

    logger.info(f"Raw data: {len(df)} candles")

    # Calculate baseline features (Phase 2)
    logger.info("\nCalculating baseline features...")
    df = calculate_features(df)

    # Calculate advanced features
    logger.info("Calculating advanced technical features...")
    adv_features = AdvancedTechnicalFeatures(lookback_sr=50, lookback_trend=20)
    df = adv_features.calculate_all_features(df)

    # Handle NaN from feature calculation
    rows_before = len(df)
    df = df.ffill()
    df = df.dropna()
    rows_after = len(df)

    logger.info(f"After feature calculation NaN handling: {rows_before} → {rows_after} rows")

    return df, adv_features


def get_base_feature_columns(df, adv_features):
    """Get base feature column names (before lag features)"""

    # Baseline features (10 from improved Phase 2)
    baseline_features = [
        'close_change_1', 'close_change_3', 'volume_ma_ratio',
        'rsi', 'macd', 'macd_signal', 'macd_diff',
        'bb_high', 'bb_mid', 'bb_low'
    ]

    # Advanced features (27)
    advanced_features = adv_features.get_feature_names()

    # Combine
    all_features = baseline_features + advanced_features

    # Filter to only existing columns
    available_features = [f for f in all_features if f in df.columns]

    logger.info(f"\nBase feature selection:")
    logger.info(f"  Baseline features: {len([f for f in baseline_features if f in df.columns])}")
    logger.info(f"  Advanced features: {len([f for f in advanced_features if f in df.columns])}")
    logger.info(f"  Total base features: {len(available_features)}")

    return available_features


def create_temporal_features(df, base_feature_columns):
    """
    Create lag and momentum features for temporal pattern learning

    Returns:
        df_temporal: DataFrame with temporal features
        temporal_feature_columns: All feature columns to use for training
    """
    logger.info("\n" + "="*80)
    logger.info("Creating Temporal Features")
    logger.info("="*80)

    # Initialize lag feature generator (t-1, t-2)
    lag_gen = LagFeatureGenerator(lag_periods=[1, 2])

    # Create all temporal features (lags + momentum)
    df_temporal, temporal_columns = lag_gen.create_all_temporal_features(
        df, base_feature_columns, include_momentum=True
    )

    # Validate temporal features
    is_valid = validate_temporal_features(df_temporal, base_feature_columns, temporal_columns)

    if not is_valid:
        logger.error("Temporal feature validation failed!")
        raise ValueError("Temporal features failed validation")

    logger.success("✅ Temporal feature validation passed")

    return df_temporal, temporal_columns


def create_labels(df, lookahead=3, threshold=0.003):
    """
    Create labels for classification

    Label = 1 if price increases > threshold% in next lookahead candles
    Label = 0 otherwise
    """
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


def train_xgboost_with_lag_features(df, feature_columns, labels, lookahead=3, threshold=0.003):
    """Train XGBoost with lag features"""

    logger.info("\n" + "="*80)
    logger.info(f"Training XGBoost Phase 4 with Lag Features")
    logger.info(f"Lookahead: {lookahead} candles, Threshold: {threshold*100}%")
    logger.info("="*80)

    # Prepare features and labels
    X = df[feature_columns].values
    y = labels

    logger.info(f"\nDataset:")
    logger.info(f"  Samples: {len(X)}")
    logger.info(f"  Features: {len(feature_columns)}")
    logger.info(f"  Positive samples: {np.sum(y)} ({np.mean(y)*100:.1f}%)")
    logger.info(f"  Negative samples: {len(y) - np.sum(y)} ({(1-np.mean(y))*100:.1f}%)")

    # Time series split
    tscv = TimeSeriesSplit(n_splits=5)

    best_model = None
    best_score = 0

    fold_scores = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Calculate class weight (ratio of negative to positive)
        scale_pos_weight = (len(y_train) - np.sum(y_train)) / np.sum(y_train)

        # Train model with class weighting
        model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            eval_metric='logloss'
        )

        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )

        # Evaluate
        y_pred = model.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred, zero_division=0)
        recall = recall_score(y_val, y_pred, zero_division=0)
        f1 = f1_score(y_val, y_pred, zero_division=0)

        fold_scores.append({
            'fold': fold,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        })

        logger.info(f"\nFold {fold}:")
        logger.info(f"  Accuracy: {accuracy:.3f}")
        logger.info(f"  Precision: {precision:.3f}")
        logger.info(f"  Recall: {recall:.3f}")
        logger.info(f"  F1 Score: {f1:.3f}")

        if f1 > best_score:
            best_score = f1
            best_model = model

    # Average scores
    avg_scores = pd.DataFrame(fold_scores).mean()

    logger.info("\n" + "="*80)
    logger.info("Cross-Validation Results (Average)")
    logger.info("="*80)
    logger.info(f"Accuracy: {avg_scores['accuracy']:.3f}")
    logger.info(f"Precision: {avg_scores['precision']:.3f}")
    logger.info(f"Recall: {avg_scores['recall']:.3f}")
    logger.info(f"F1 Score: {avg_scores['f1']:.3f}")

    # Train final model on all data
    logger.info("\n" + "="*80)
    logger.info("Training Final Model on All Data")
    logger.info("="*80)

    # Calculate class weight for final model
    scale_pos_weight = (len(y) - np.sum(y)) / np.sum(y)
    logger.info(f"Class weight (scale_pos_weight): {scale_pos_weight:.2f}")

    final_model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric='logloss'
    )

    final_model.fit(X, y, verbose=False)

    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': final_model.feature_importances_
    }).sort_values('importance', ascending=False)

    logger.info("\nTop 30 Most Important Features:")
    logger.info(feature_importance.head(30).to_string(index=False))

    # Analyze lag feature impact
    lag_features_in_top30 = feature_importance.head(30)['feature'].str.contains('lag|momentum').sum()
    logger.success(f"\n✅ Lag/Momentum features in Top 30: {lag_features_in_top30}/30")

    return final_model, feature_importance, avg_scores


def save_model(model, feature_columns, lookahead, threshold, scores):
    """Save model and metadata"""

    model_name = f"xgboost_v4_phase4_lag_lookahead{lookahead}_thresh{int(threshold*1000)}"
    model_path = MODELS_DIR / f"{model_name}.pkl"
    feature_path = MODELS_DIR / f"{model_name}_features.txt"

    # Save model
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    # Save features
    with open(feature_path, 'w') as f:
        for feature in feature_columns:
            f.write(f"{feature}\n")

    # Save metadata
    metadata = {
        "model_name": model_name,
        "n_features": len(feature_columns),
        "has_lag_features": True,
        "lag_periods": [1, 2],
        "has_momentum_features": True,
        "lookahead": lookahead,
        "threshold": threshold,
        "timestamp": datetime.now().isoformat(),
        "scores": {
            "accuracy": float(scores['accuracy']),
            "precision": float(scores['precision']),
            "recall": float(scores['recall']),
            "f1": float(scores['f1'])
        }
    }

    import json
    metadata_path = MODELS_DIR / f"{model_name}_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info("\n" + "="*80)
    logger.info("Model Saved")
    logger.info("="*80)
    logger.info(f"Model: {model_path}")
    logger.info(f"Features: {feature_path}")
    logger.info(f"Metadata: {metadata_path}")

    return model_path


def main():
    """Main training pipeline"""

    # Load and prepare data
    df, adv_features = load_and_prepare_data()

    # Get base feature columns
    base_feature_columns = get_base_feature_columns(df, adv_features)

    # Create temporal features (lags + momentum)
    df_temporal, temporal_feature_columns = create_temporal_features(df, base_feature_columns)

    # Create labels
    lookahead = 3
    threshold = 0.003  # 0.3% threshold

    logger.info(f"\nCreating labels (lookahead={lookahead}, threshold={threshold*100}%)...")
    labels = create_labels(df_temporal, lookahead=lookahead, threshold=threshold)

    # Train model with lag features
    model, feature_importance, scores = train_xgboost_with_lag_features(
        df_temporal, temporal_feature_columns, labels, lookahead=lookahead, threshold=threshold
    )

    # Save model
    model_path = save_model(model, temporal_feature_columns, lookahead, threshold, scores)

    logger.info("\n" + "="*80)
    logger.success("✅ XGBoost Phase 4 Lag Features Training Complete!")
    logger.info("="*80)
    logger.info(f"\nExpected Improvement vs Phase 4 Base:")
    logger.info(f"  Base model: F1={0.089:.3f}, Win Rate=69.1%, Returns=7.68%/5d")
    logger.info(f"  Lag model: F1={scores['f1']:.3f} (expected >0.10)")
    logger.info(f"  Expected: Win Rate 75-80%, Returns 9-10%/5d")
    logger.info(f"\nNext Steps:")
    logger.info(f"1. Backtest lag-feature model")
    logger.info(f"2. Compare: Phase 4 Base vs Phase 4 Lag")
    logger.info(f"3. If improvement confirmed → Deploy to testnet")
    logger.info(f"\n비판적 검증: Lag features가 실제로 temporal patterns를 학습했는가?")


if __name__ == "__main__":
    main()
