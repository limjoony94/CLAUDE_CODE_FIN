"""
XGBoost Phase 4 Training with Advanced Technical Features

60 Features:
- 33 baseline features (Phase 2)
- 27 advanced features (지지/저항, 추세선, 패턴, 다이버전스)

목표: 레버리지 없이 0.3-0.5%/day 달성 → Leverage 2x 적용 시 0.6-1.0%/day

비판적 사고:
"전문 트레이더가 보는 여러 캔들 패턴을 XGBoost가 학습하면,
레버리지 없이도 더 높은 수익을 낼 수 있다"
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from datetime import datetime
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import MinMaxScaler

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


def load_and_prepare_data():
    """Load data and calculate all features"""
    print("="*80)
    print("XGBoost Phase 4: Advanced Technical Features")
    print("="*80)
    print("\nLoading data...")

    data_file = DATA_DIR / "BTCUSDT_5m_max.csv"
    df = pd.read_csv(data_file)

    print(f"Raw data: {len(df)} candles")

    # Calculate baseline features (Phase 2)
    print("\nCalculating baseline features (33 features)...")
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


def create_labels(df, lookahead=3, threshold=0.01):
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


def get_feature_columns(df, adv_features):
    """Get all feature column names"""

    # Baseline features (33 from Phase 2)
    baseline_features = [
        'returns', 'log_returns', 'close_change_1', 'close_change_3',
        'volume_change', 'volume_ma_ratio', 'rsi', 'rsi_ma', 'rsi_change',
        'macd', 'macd_signal', 'macd_diff', 'bb_high', 'bb_mid', 'bb_low',
        'bb_width', 'bb_position', 'atr', 'atr_pct', 'adx', 'ema_9', 'ema_21',
        'ema_diff', 'price_vs_ema9', 'price_vs_ema21', 'stoch_k', 'stoch_d',
        'stoch_diff', 'obv', 'obv_ema', 'obv_divergence', 'vwap', 'price_vs_vwap'
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


def train_xgboost_phase4(df, feature_columns, labels, lookahead=3, threshold=0.01):
    """Train XGBoost with advanced features and MinMaxScaler normalization"""

    print("\n" + "="*80)
    print(f"Training XGBoost Phase 4 with MinMaxScaler(-1, 1)")
    print(f"Lookahead: {lookahead} candles, Threshold: {threshold*100}%")
    print("="*80)

    # Prepare features and labels
    X = df[feature_columns].values
    y = labels

    print(f"\nDataset:")
    print(f"  Samples: {len(X)}")
    print(f"  Features: {len(feature_columns)}")
    print(f"  Positive samples: {np.sum(y)} ({np.mean(y)*100:.1f}%)")
    print(f"  Negative samples: {len(y) - np.sum(y)} ({(1-np.mean(y))*100:.1f}%)")

    # Time series split
    tscv = TimeSeriesSplit(n_splits=5)

    best_model = None
    best_scaler = None
    best_score = 0

    fold_scores = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # ✅ Apply MinMaxScaler normalization to [-1, 1] range
        scaler = MinMaxScaler(feature_range=(-1, 1))
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)

        # Calculate class weight (ratio of negative to positive)
        scale_pos_weight = (len(y_train) - np.sum(y_train)) / np.sum(y_train)

        # Train model with class weighting
        model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,  # Weight positive class higher
            random_state=42,
            eval_metric='logloss'
        )

        model.fit(
            X_train_scaled, y_train,
            eval_set=[(X_val_scaled, y_val)],
            verbose=False
        )

        # Evaluate
        y_pred = model.predict(X_val_scaled)
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

        print(f"\nFold {fold}:")
        print(f"  Accuracy: {accuracy:.3f}")
        print(f"  Precision: {precision:.3f}")
        print(f"  Recall: {recall:.3f}")
        print(f"  F1 Score: {f1:.3f}")

        if f1 > best_score:
            best_score = f1
            best_model = model
            best_scaler = scaler

    # Average scores
    avg_scores = pd.DataFrame(fold_scores).mean()

    print("\n" + "="*80)
    print("Cross-Validation Results (Average)")
    print("="*80)
    print(f"Accuracy: {avg_scores['accuracy']:.3f}")
    print(f"Precision: {avg_scores['precision']:.3f}")
    print(f"Recall: {avg_scores['recall']:.3f}")
    print(f"F1 Score: {avg_scores['f1']:.3f}")

    # Train final model on all data
    print("\n" + "="*80)
    print("Training Final Model on All Data with MinMaxScaler(-1, 1)")
    print("="*80)

    # ✅ Fit scaler on all data
    final_scaler = MinMaxScaler(feature_range=(-1, 1))
    X_scaled = final_scaler.fit_transform(X)

    # Calculate class weight for final model
    scale_pos_weight = (len(y) - np.sum(y)) / np.sum(y)
    print(f"Class weight (scale_pos_weight): {scale_pos_weight:.2f}")

    final_model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,  # Weight positive class higher
        random_state=42,
        eval_metric='logloss'
    )

    final_model.fit(X_scaled, y, verbose=False)

    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': final_model.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\nTop 20 Most Important Features:")
    print(feature_importance.head(20).to_string(index=False))

    # Check advanced features impact
    advanced_in_top20 = feature_importance.head(20)['feature'].isin(
        AdvancedTechnicalFeatures(50, 20).get_feature_names()
    ).sum()

    print(f"\n✅ Advanced features in Top 20: {advanced_in_top20}/20")
    print(f"✅ MinMaxScaler applied: All features normalized to [-1, 1] range")

    return final_model, final_scaler, feature_importance, avg_scores


def save_model(model, scaler, feature_columns, lookahead, threshold, scores):
    """Save model, scaler, and metadata"""

    model_name = f"xgboost_v4_phase4_advanced_lookahead{lookahead}_thresh{int(threshold*100)}"
    model_path = MODELS_DIR / f"{model_name}.pkl"
    scaler_path = MODELS_DIR / f"{model_name}_scaler.pkl"
    feature_path = MODELS_DIR / f"{model_name}_features.txt"

    # Save model
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    # ✅ Save scaler
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)

    # Save features
    with open(feature_path, 'w') as f:
        for feature in feature_columns:
            f.write(f"{feature}\n")

    # Save metadata
    metadata = {
        "model_name": model_name,
        "n_features": len(feature_columns),
        "lookahead": lookahead,
        "threshold": threshold,
        "normalized": True,
        "scaler": "MinMaxScaler",
        "scaler_range": [-1, 1],
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

    print("\n" + "="*80)
    print("Model Saved")
    print("="*80)
    print(f"Model: {model_path}")
    print(f"Scaler: {scaler_path}")
    print(f"Features: {feature_path}")
    print(f"Metadata: {metadata_path}")

    return model_path


def main():
    """Main training pipeline"""

    # Load and prepare data
    df, adv_features = load_and_prepare_data()

    # Get feature columns
    feature_columns = get_feature_columns(df, adv_features)

    # Create labels
    lookahead = 3
    threshold = 0.003  # Lowered to 0.3% to get ~5-10% positive samples

    print(f"\nCreating labels (lookahead={lookahead}, threshold={threshold*100}%)...")
    labels = create_labels(df, lookahead=lookahead, threshold=threshold)

    # Train model with StandardScaler
    model, scaler, feature_importance, scores = train_xgboost_phase4(
        df, feature_columns, labels, lookahead=lookahead, threshold=threshold
    )

    # Save model and scaler
    model_path = save_model(model, scaler, feature_columns, lookahead, threshold, scores)

    print("\n" + "="*80)
    print("✅ XGBoost Phase 4 Training Complete with MinMaxScaler!")
    print("="*80)
    print(f"\nKey Improvements:")
    print(f"1. ✅ MinMaxScaler normalization applied (range: [-1, 1])")
    print(f"2. All features now in consistent [-1, 1] range")
    print(f"3. Small-scale indicators (RSI, Divergence) now properly weighted")
    print(f"\nNext Steps:")
    print(f"1. Backtest with normalized features")
    print(f"2. Compare performance: MinMaxScaler vs StandardScaler vs None")
    print(f"3. Train SHORT model with same normalization")
    print(f"\n비판적 검증: MinMaxScaler(-1,1)이 실제로 성능 향상을 가져왔는가?")


if __name__ == "__main__":
    main()
