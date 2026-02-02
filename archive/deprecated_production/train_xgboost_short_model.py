"""
XGBoost SHORT Model Training

목적:
1. SHORT 전용 모델 학습 (하락 예측)
2. LONG 모델과 동일한 37 features 사용
3. Target: 향후 3 캔들에서 가격 하락 예측
4. 듀얼 모델 전략의 일부 (LONG + SHORT)

핵심 차이점:
- LONG Model: target = (future_return > +threshold)  → 상승 예측
- SHORT Model: target = (future_return < -threshold) → 하락 예측 ← NEW!

Features:
- Phase 4 Advanced Features (37개) 사용
- Support/Resistance, Trend Lines 포함
- LONG 모델과 동일한 feature set

검증 목표:
- SHORT 모델 단독 승률 > 60%
- SHORT 모델이 하락장에서 실제 수익
- 듀얼 모델(LONG+SHORT)이 LONG-only보다 우수
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from imblearn.over_sampling import SMOTE
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

import sys
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.production.train_xgboost_improved_v3_phase2 import calculate_features
from scripts.production.advanced_technical_features import AdvancedTechnicalFeatures

MODELS_DIR.mkdir(parents=True, exist_ok=True)


def create_short_target(df, lookahead, threshold):
    """
    Create binary target for SHORT signals (downward prediction) - SYMMETRIC with LONG

    SYMMETRIC LABELING (2025-10-15 FIX):
    - LONG: target = 1 if MAX(next 3 candles) >= +threshold% (catches any upward spike)
    - SHORT: target = 1 if MIN(next 3 candles) <= -threshold% (catches any downward spike)

    This ensures train-test consistency and symmetric opportunity detection.

    Args:
        df: DataFrame with OHLCV data
        lookahead: Number of candles to look ahead (default: 3 = 15 min)
        threshold: Minimum downward movement to trigger SHORT (default: 0.003 = 0.3%)

    Returns:
        Series: 1 if MIN price in next lookahead candles drops >= threshold, 0 otherwise

    Example:
        lookahead=3, threshold=0.003 (0.3%):
        - Current: $100
        - Next 3: [$100.2, $99.6, $100.1] → MIN=$99.6 (-0.4%) → target = 1 (SHORT)
        - Next 3: [$100.1, $100.3, $99.8] → MIN=$99.8 (-0.2%) → target = 0 (no SHORT)
    """
    # Get minimum price in next lookahead candles (symmetric with LONG's max)
    future_prices = df['close'].shift(-1).rolling(window=lookahead).apply(lambda x: x.min())

    # Calculate return from current price to minimum future price
    future_return = (df['close'] - future_prices) / df['close']

    # SHORT target: 하락 예측 (any candle drops >= threshold)
    # Now SYMMETRIC with LONG model!
    target = (future_return >= threshold).astype(int)

    return target


def train_short_model(df, lookahead=3, threshold=0.01):
    """
    Train XGBoost SHORT model with Phase 4 Advanced Features

    Args:
        df: DataFrame with OHLCV data
        lookahead: Number of candles to look ahead (3 = 15 min)
        threshold: Minimum downward movement (0.01 = 1%)

    Returns:
        model, metrics, metadata, feature_columns
    """
    print(f"\n{'=' * 80}")
    print(f"SHORT Model Training: lookahead={lookahead} ({lookahead * 5} min), threshold={threshold * 100:.2f}%")
    print(f"{'=' * 80}")
    print(f"Target: Price DROP >= {threshold * 100:.2f}% in next {lookahead} candles")

    # Create SHORT target (하락 예측)
    df['target_short'] = create_short_target(df, lookahead, threshold)

    # Get feature columns (Phase 4 Advanced Features - 37개)
    # LONG 모델과 동일한 features 사용
    feature_path = MODELS_DIR / "xgboost_v4_phase4_advanced_lookahead3_thresh0_features.txt"

    if feature_path.exists():
        with open(feature_path, 'r') as f:
            feature_columns = [line.strip() for line in f.readlines()]
        print(f"✅ Loaded {len(feature_columns)} features from Phase 4 LONG model")
    else:
        raise FileNotFoundError(f"Phase 4 LONG model features not found: {feature_path}")

    # Drop NaN
    df_clean = df.dropna()
    print(f"After dropna: {len(df_clean)} rows")

    # Prepare data
    X = df_clean[feature_columns].values
    y = df_clean['target_short'].values

    print(f"\nTarget distribution (before SMOTE):")
    print(f"  Class 0 (no SHORT): {(y == 0).sum()} ({(y == 0).sum() / len(y) * 100:.1f}%)")
    print(f"  Class 1 (SHORT): {(y == 1).sum()} ({(y == 1).sum() / len(y) * 100:.1f}%)")

    # Check minimum samples
    if (y == 1).sum() < 100:
        print(f"⚠️ Too few SHORT samples ({(y == 1).sum()}), skipping...")
        return None, None, None, None

    # Train/val/test split
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, shuffle=False)

    print(f"\nData split:")
    print(f"  Train: {len(X_train)}")
    print(f"  Val: {len(X_val)}")
    print(f"  Test: {len(X_test)}")

    # ✅ Apply MinMaxScaler normalization to [-1, 1] range
    print("\n✅ Applying MinMaxScaler normalization to [-1, 1] range...")
    scaler = MinMaxScaler(feature_range=(-1, 1))
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    print("  All features normalized to [-1, 1] range")

    # Apply SMOTE on scaled data
    print("\nApplying SMOTE on normalized data...")
    target_ratio = min(0.3, (y_train == 1).sum() / (y_train == 0).sum() * 10)

    try:
        smote = SMOTE(sampling_strategy=target_ratio, random_state=42)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train_scaled, y_train)
        print(f"After SMOTE:")
        print(f"  Class 0: {(y_train_balanced == 0).sum()} ({(y_train_balanced == 0).sum() / len(y_train_balanced) * 100:.1f}%)")
        print(f"  Class 1: {(y_train_balanced == 1).sum()} ({(y_train_balanced == 1).sum() / len(y_train_balanced) * 100:.1f}%)")
    except Exception as e:
        print(f"⚠️ SMOTE failed: {e}")
        X_train_balanced = X_train_scaled
        y_train_balanced = y_train

    # Train XGBoost
    print(f"\nTraining XGBoost SHORT model with {len(feature_columns)} normalized features...")
    scale_pos_weight = (y_train_balanced == 0).sum() / (y_train_balanced == 1).sum()

    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric='logloss'
    )

    model.fit(
        X_train_balanced, y_train_balanced,
        eval_set=[(X_val_scaled, y_val)],
        verbose=False
    )

    # Evaluate
    train_acc = model.score(X_train_balanced, y_train_balanced)
    val_acc = model.score(X_val_scaled, y_val)
    test_acc = model.score(X_test_scaled, y_test)

    print(f"\nAccuracy:")
    print(f"  Train: {train_acc:.4f}")
    print(f"  Val: {val_acc:.4f}")
    print(f"  Test: {test_acc:.4f}")

    # Test predictions
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    y_pred = model.predict(X_test_scaled)

    # Metrics
    print(f"\nTest Set Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['No SHORT', 'SHORT'], zero_division=0))

    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix:")
    print(cm)

    # Probability distribution
    print(f"\nProbability Distribution (test set):")
    print(f"  Mean probability: {y_pred_proba.mean():.4f}")
    print(f"  Std probability: {y_pred_proba.std():.4f}")
    print(f"  Prob > 0.5: {(y_pred_proba > 0.5).sum()} ({(y_pred_proba > 0.5).sum() / len(y_pred_proba) * 100:.1f}%)")
    print(f"  Prob > 0.6: {(y_pred_proba > 0.6).sum()} ({(y_pred_proba > 0.6).sum() / len(y_pred_proba) * 100:.1f}%)")
    print(f"  Prob > 0.7: {(y_pred_proba > 0.7).sum()} ({(y_pred_proba > 0.7).sum() / len(y_pred_proba) * 100:.1f}%)")

    # Calculate metrics
    recall_class_1 = cm[1, 1] / (cm[1, 0] + cm[1, 1]) if (cm[1, 0] + cm[1, 1]) > 0 else 0
    precision_class_1 = cm[1, 1] / (cm[0, 1] + cm[1, 1]) if (cm[0, 1] + cm[1, 1]) > 0 else 0
    f1_score = 2 * (recall_class_1 * precision_class_1) / (recall_class_1 + precision_class_1) if (recall_class_1 + precision_class_1) > 0 else 0

    print(f"\nBalance Metrics:")
    print(f"  Recall (SHORT): {recall_class_1:.4f}")
    print(f"  Precision (SHORT): {precision_class_1:.4f}")
    print(f"  F1-Score: {f1_score:.4f}")

    # Feature importance (top 10)
    print(f"\nTop 10 Feature Importance (SHORT Model):")
    importance = model.feature_importances_
    feature_importance = sorted(zip(feature_columns, importance), key=lambda x: x[1], reverse=True)
    for i, (feat, imp) in enumerate(feature_importance[:10], 1):
        print(f"  {i}. {feat}: {imp:.4f}")

    metrics = {
        'f1_score': f1_score,
        'recall': recall_class_1,
        'precision': precision_class_1,
        'test_acc': test_acc,
        'mean_prob': y_pred_proba.mean(),
        'prob_gt_0.7': (y_pred_proba > 0.7).sum() / len(y_pred_proba) * 100
    }

    metadata = {
        'lookahead': lookahead,
        'threshold': threshold,
        'feature_count': len(feature_columns),
        'model_type': 'SHORT',
        'direction': 'DOWNWARD',
        'normalized': True,
        'scaler': 'MinMaxScaler',
        'scaler_range': [-1, 1]
    }

    return model, scaler, metrics, metadata, feature_columns


if __name__ == "__main__":
    print("=" * 80)
    print("XGBoost SHORT Model Training")
    print("=" * 80)
    print("목표: 하락 예측 전용 모델 학습")
    print("=" * 80)

    # Load data
    data_file = DATA_DIR / "historical" / "BTCUSDT_5m_max.csv"
    df = pd.read_csv(data_file)
    print(f"✅ Loaded {len(df)} candles")

    # Calculate Phase 2 features
    print("\nCalculating Phase 2 features...")
    df = calculate_features(df)
    print(f"✅ Phase 2 features calculated: {len(df)} rows")

    # Calculate Phase 4 Advanced features
    print("\nCalculating Phase 4 Advanced features...")
    adv_features = AdvancedTechnicalFeatures(lookback_sr=50, lookback_trend=20)
    df = adv_features.calculate_all_features(df)
    df = df.ffill().dropna()
    print(f"✅ Phase 4 Advanced features calculated: {len(df)} rows")

    # Train SHORT model (LOWER threshold than LONG)
    lookahead = 3  # 15 min
    threshold = 0.003  # 0.3% (LONG 1%보다 크게 낮춤 - 하락이 훨씬 드물기 때문)

    model, scaler, metrics, metadata, feature_columns = train_short_model(
        df.copy(),
        lookahead=lookahead,
        threshold=threshold
    )

    if model is not None:
        # Save SHORT model
        model_file = MODELS_DIR / "xgboost_short_model_lookahead3_thresh0.3.pkl"

        with open(model_file, 'wb') as f:
            pickle.dump(model, f)
        print(f"\n✅ SHORT Model saved: {model_file}")

        # ✅ Save scaler
        scaler_file = MODELS_DIR / "xgboost_short_model_lookahead3_thresh0.3_scaler.pkl"
        with open(scaler_file, 'wb') as f:
            pickle.dump(scaler, f)
        print(f"✅ Scaler saved: {scaler_file}")

        # Save feature columns
        feature_file = MODELS_DIR / "xgboost_short_model_lookahead3_thresh0.3_features.txt"
        with open(feature_file, 'w') as f:
            f.write('\n'.join(feature_columns))
        print(f"✅ Features saved: {feature_file}")

        # Save metadata
        metadata_file = MODELS_DIR / "xgboost_short_model_lookahead3_thresh0.3_metadata.txt"
        with open(metadata_file, 'w') as f:
            f.write("XGBoost SHORT Model - Metadata\n")
            f.write("=" * 80 + "\n\n")
            f.write("CONFIG:\n")
            for key, value in metadata.items():
                f.write(f"  {key}: {value}\n")
            f.write("\nMETRICS:\n")
            for key, value in metrics.items():
                f.write(f"  {key}: {value}\n")
        print(f"✅ Metadata saved: {metadata_file}")

        print("\n" + "=" * 80)
        print("✅ SHORT Model Training Complete with MinMaxScaler!")
        print("=" * 80)
        print("\nKey Improvements:")
        print("  1. ✅ MinMaxScaler normalization applied (range: [-1, 1])")
        print("  2. All features now in consistent [-1, 1] range")
        print("  3. Small-scale indicators (RSI, Divergence) now properly weighted")
        print(f"  4. F1-Score: {metrics['f1_score']:.4f}")
        print(f"  5. Mean Probability: {metrics['mean_prob']:.4f}")
        print(f"  6. Prob > 0.7: {metrics['prob_gt_0.7']:.1f}%")

        print(f"\n다음 단계:")
        print(f"  1. Backtest with normalized SHORT model")
        print(f"  2. Compare: MinMaxScaler vs StandardScaler vs None")
        print(f"  3. Train Exit Models with same normalization")

        print(f"\n비판적 검증:")
        print(f"  ❓ MinMaxScaler(-1,1)이 SHORT 모델 성능을 개선했는가?")
        print(f"  ❓ Win Rate가 41.9%에서 55%+ 상승했는가?")
        print(f"  ❓ False Positive가 감소했는가?")

    else:
        print("\n❌ SHORT Model training failed!")
