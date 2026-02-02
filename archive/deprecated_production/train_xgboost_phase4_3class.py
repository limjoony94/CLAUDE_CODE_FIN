"""
XGBoost Phase 4 - 3-Class Classification Training

CRITICAL UPDATE (2025-10-10):
- Implements 3-class classification for bidirectional trading
- Label 0: NEUTRAL (sideways, no trade)
- Label 1: LONG (upward movement)
- Label 2: SHORT (downward movement)

비판적 사고:
"명시적으로 LONG과 SHORT를 학습시키면,
inverse probability 방식의 근본적 결함을 해결할 수 있다"
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from datetime import datetime
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# Import baseline features
import sys
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.production.train_xgboost_improved_v3_phase2 import calculate_features
from scripts.production.advanced_technical_features import AdvancedTechnicalFeatures

# Directories
DATA_DIR = PROJECT_ROOT / "data" / "historical"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def create_labels_3class(df, lookahead=3, threshold=0.003):
    """
    Create 3-class labels for bidirectional trading

    Label 0 (NEUTRAL): No significant movement in either direction
    Label 1 (LONG): Price increases > threshold% in next lookahead candles
    Label 2 (SHORT): Price decreases > threshold% in next lookahead candles

    Args:
        df: DataFrame with OHLCV data
        lookahead: Number of candles to look ahead
        threshold: Minimum price change (as decimal, e.g., 0.003 = 0.3%)

    Returns:
        numpy array of labels (0, 1, or 2)
    """
    labels = []

    for i in range(len(df)):
        if i >= len(df) - lookahead:
            labels.append(0)  # Neutral for last candles (no future data)
            continue

        current_price = df['close'].iloc[i]
        future_prices = df['close'].iloc[i+1:i+1+lookahead]

        max_future_price = future_prices.max()
        min_future_price = future_prices.min()

        # Calculate maximum increase and decrease
        increase_pct = (max_future_price - current_price) / current_price
        decrease_pct = (current_price - min_future_price) / current_price

        # Classify based on which movement is stronger
        if increase_pct >= threshold and increase_pct > decrease_pct:
            labels.append(1)  # LONG (upward movement dominant)
        elif decrease_pct >= threshold and decrease_pct > increase_pct:
            labels.append(2)  # SHORT (downward movement dominant)
        else:
            labels.append(0)  # NEUTRAL (sideways or small movements)

    return np.array(labels)


def load_and_prepare_data():
    """Load data and calculate all features"""
    print("="*80)
    print("XGBoost Phase 4: 3-Class Classification Training")
    print("="*80)
    print("\nLoading data...")

    data_file = DATA_DIR / "BTCUSDT_5m_max.csv"
    df = pd.read_csv(data_file)

    print(f"Raw data: {len(df)} candles")

    # Calculate baseline features (Phase 2)
    print("\nCalculating baseline features (10 features)...")
    df = calculate_features(df)

    # Calculate advanced features (27 features)
    print("Calculating advanced technical features (27 features)...")
    adv_features = AdvancedTechnicalFeatures(lookback_sr=50, lookback_trend=20)
    df = adv_features.calculate_all_features(df)

    # Handle NaN
    rows_before = len(df)
    df = df.ffill()
    df = df.dropna()
    rows_after = len(df)

    print(f"After NaN handling: {rows_before} → {rows_after} rows")

    return df, adv_features


def get_feature_columns(df, adv_features):
    """Get all feature column names (37 features total)"""

    # Baseline features (10)
    baseline_features = [
        'rsi', 'macd', 'macd_signal', 'bb_upper', 'bb_lower',
        'ema_21', 'ema_50', 'atr', 'volume_ratio', 'price_distance_ema'
    ]

    # Advanced features (27)
    advanced_features = adv_features.get_feature_names()

    # Combine
    all_features = baseline_features + advanced_features

    # Filter to only existing columns
    available_features = [f for f in all_features if f in df.columns]

    print(f"\nFeature selection:")
    print(f"  Baseline features: {len([f for f in baseline_features if f in df.columns])}")
    print(f"  Advanced features: {len([f for f in advanced_features if f in df.columns])}")
    print(f"  Total features: {len(available_features)}")

    return available_features


def train_xgboost_3class(df, feature_columns, labels, lookahead=3, threshold=0.003):
    """Train XGBoost with 3-class classification"""

    print("\n" + "="*80)
    print(f"Training XGBoost 3-Class Model")
    print(f"Lookahead: {lookahead} candles, Threshold: {threshold*100}%")
    print("="*80)

    # Prepare features and labels
    X = df[feature_columns].values
    y = labels

    # Class distribution
    unique, counts = np.unique(y, return_counts=True)
    class_dist = dict(zip(unique, counts))

    print(f"\nDataset:")
    print(f"  Samples: {len(X)}")
    print(f"  Features: {len(feature_columns)}")
    print(f"\n  Class Distribution:")
    print(f"    Class 0 (NEUTRAL): {class_dist.get(0, 0):5d} ({class_dist.get(0, 0)/len(y)*100:5.1f}%)")
    print(f"    Class 1 (LONG):    {class_dist.get(1, 0):5d} ({class_dist.get(1, 0)/len(y)*100:5.1f}%)")
    print(f"    Class 2 (SHORT):   {class_dist.get(2, 0):5d} ({class_dist.get(2, 0)/len(y)*100:5.1f}%)")

    # Time series split
    tscv = TimeSeriesSplit(n_splits=5)

    best_model = None
    best_score = 0

    fold_scores = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Calculate class weights for balancing
        from sklearn.utils.class_weight import compute_sample_weight
        sample_weights = compute_sample_weight('balanced', y_train)

        # Train model with 3 classes and balanced weights
        model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.01,
            min_child_weight=5,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=1,
            reg_alpha=1,
            reg_lambda=2,
            random_state=42,
            use_label_encoder=False,
            eval_metric='mlogloss',
            num_class=3
        )

        model.fit(
            X_train, y_train,
            sample_weight=sample_weights,  # Apply balanced weights
            eval_set=[(X_val, y_val)],
            verbose=False
        )

        # Evaluate
        y_pred = model.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)

        # Per-class metrics
        precision = precision_score(y_val, y_pred, average=None, zero_division=0)
        recall = recall_score(y_val, y_pred, average=None, zero_division=0)
        f1 = f1_score(y_val, y_pred, average=None, zero_division=0)

        fold_scores.append({
            'fold': fold,
            'accuracy': accuracy,
            'precision_neutral': precision[0] if len(precision) > 0 else 0,
            'precision_long': precision[1] if len(precision) > 1 else 0,
            'precision_short': precision[2] if len(precision) > 2 else 0,
            'recall_neutral': recall[0] if len(recall) > 0 else 0,
            'recall_long': recall[1] if len(recall) > 1 else 0,
            'recall_short': recall[2] if len(recall) > 2 else 0,
            'f1_neutral': f1[0] if len(f1) > 0 else 0,
            'f1_long': f1[1] if len(f1) > 1 else 0,
            'f1_short': f1[2] if len(f1) > 2 else 0
        })

        print(f"\nFold {fold}:")
        print(f"  Accuracy: {accuracy:.3f}")
        print(f"  NEUTRAL - Precision: {precision[0]:.3f}, Recall: {recall[0]:.3f}, F1: {f1[0]:.3f}")
        print(f"  LONG    - Precision: {precision[1]:.3f}, Recall: {recall[1]:.3f}, F1: {f1[1]:.3f}")
        print(f"  SHORT   - Precision: {precision[2]:.3f}, Recall: {recall[2]:.3f}, F1: {f1[2]:.3f}")

        # Use average F1 for best model selection
        avg_f1 = np.mean(f1)
        if avg_f1 > best_score:
            best_score = avg_f1
            best_model = model

    # Average scores
    avg_scores = pd.DataFrame(fold_scores).mean()

    print("\n" + "="*80)
    print("Cross-Validation Results (Average)")
    print("="*80)
    print(f"Accuracy: {avg_scores['accuracy']:.3f}")
    print(f"\nNEUTRAL - Precision: {avg_scores['precision_neutral']:.3f}, Recall: {avg_scores['recall_neutral']:.3f}, F1: {avg_scores['f1_neutral']:.3f}")
    print(f"LONG    - Precision: {avg_scores['precision_long']:.3f}, Recall: {avg_scores['recall_long']:.3f}, F1: {avg_scores['f1_long']:.3f}")
    print(f"SHORT   - Precision: {avg_scores['precision_short']:.3f}, Recall: {avg_scores['recall_short']:.3f}, F1: {avg_scores['f1_short']:.3f}")

    # Train final model on all data
    print("\n" + "="*80)
    print("Training Final Model on All Data")
    print("="*80)

    # Calculate sample weights for final model
    from sklearn.utils.class_weight import compute_sample_weight
    final_sample_weights = compute_sample_weight('balanced', y)

    final_model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.01,
        min_child_weight=5,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=1,
        reg_alpha=1,
        reg_lambda=2,
        random_state=42,
        use_label_encoder=False,
        eval_metric='mlogloss',
        num_class=3
    )

    final_model.fit(X, y, sample_weight=final_sample_weights, verbose=False)

    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': final_model.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\nTop 20 Most Important Features:")
    print(feature_importance.head(20).to_string(index=False))

    return final_model, feature_importance, avg_scores


def save_model(model, feature_columns, lookahead, threshold, scores):
    """Save 3-class model and metadata"""

    model_name = f"xgboost_v4_phase4_3class_lookahead{lookahead}_thresh{int(threshold*1000)}"
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
        "model_type": "3-class classification",
        "classes": {
            "0": "NEUTRAL (sideways)",
            "1": "LONG (upward)",
            "2": "SHORT (downward)"
        },
        "n_features": len(feature_columns),
        "lookahead": lookahead,
        "threshold": threshold,
        "timestamp": datetime.now().isoformat(),
        "scores": {
            "accuracy": float(scores['accuracy']),
            "f1_neutral": float(scores['f1_neutral']),
            "f1_long": float(scores['f1_long']),
            "f1_short": float(scores['f1_short']),
            "precision_long": float(scores['precision_long']),
            "precision_short": float(scores['precision_short']),
            "recall_long": float(scores['recall_long']),
            "recall_short": float(scores['recall_short'])
        }
    }

    import json
    metadata_path = MODELS_DIR / f"{model_name}_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "="*80)
    print("Model Saved")
    print("="*80)
    print(f"Model: {model_path}")
    print(f"Features: {feature_path}")
    print(f"Metadata: {metadata_path}")

    return model_path


def main():
    """Main training pipeline"""

    # Load and prepare data
    df, adv_features = load_and_prepare_data()

    # Get feature columns
    feature_columns = get_feature_columns(df, adv_features)

    # Create 3-class labels
    lookahead = 3
    threshold = 0.003  # 0.3% minimum movement

    print(f"\nCreating 3-class labels (lookahead={lookahead}, threshold={threshold*100}%)...")
    labels = create_labels_3class(df, lookahead=lookahead, threshold=threshold)

    # Train model
    model, feature_importance, scores = train_xgboost_3class(
        df, feature_columns, labels, lookahead=lookahead, threshold=threshold
    )

    # Save model
    model_path = save_model(model, feature_columns, lookahead, threshold, scores)

    print("\n" + "="*80)
    print("✅ XGBoost 3-Class Training Complete!")
    print("="*80)
    print(f"\nNext Steps:")
    print(f"1. Backtest with 3-class model")
    print(f"2. Validate SHORT profitability (target: >60% win rate)")
    print(f"3. Compare: 3-class vs 2-class (inverse) vs LONG-only")
    print(f"4. Deploy if SHORT win rate >= 60%")
    print(f"\n비판적 검증: 3-class가 SHORT 수익성을 개선했는가?")


if __name__ == "__main__":
    main()
