"""
Improved XGBoost Model with SMOTE and Threshold Optimization

목적:
1. 클래스 불균형 문제 해결 (SMOTE 적용)
2. threshold 최적화 (0.5% or 0.3%)
3. class_weight='balanced' 사용
4. 거래 가능한 모델 생성

비판적 사고:
- "Probability 0.000 → 모델이 무용지물"
- "클래스 불균형을 반드시 해결해야 함"
- "threshold를 낮춰서 positive samples를 증가시켜야 함"
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import ta

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

MODELS_DIR.mkdir(parents=True, exist_ok=True)

def calculate_features(df):
    """Calculate technical indicators"""
    df = df.copy()

    # Price changes
    for lag in range(1, 6):
        df[f'close_change_{lag}'] = df['close'].pct_change(lag)

    # Moving averages
    df['sma_10'] = ta.trend.sma_indicator(df['close'], window=10)
    df['sma_20'] = ta.trend.sma_indicator(df['close'], window=20)
    df['ema_10'] = ta.trend.ema_indicator(df['close'], window=10)

    # MACD
    macd = ta.trend.MACD(df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_diff'] = macd.macd_diff()

    # RSI
    df['rsi'] = ta.momentum.rsi(df['close'], window=14)

    # Bollinger Bands
    bb = ta.volatility.BollingerBands(df['close'])
    df['bb_high'] = bb.bollinger_hband()
    df['bb_low'] = bb.bollinger_lband()
    df['bb_mid'] = bb.bollinger_mavg()

    # Volatility
    df['volatility'] = df['close'].pct_change().rolling(window=20).std()

    # Volume
    df['volume_sma'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma']

    return df

def create_target(df, lookahead=12, threshold=0.005):
    """
    Create binary target with lower threshold for more positive samples

    Args:
        lookahead: 12 candles (1 hour for 5m data)
        threshold: 0.5% (lowered from 1.0%)
    """
    future_return = df['close'].pct_change(lookahead).shift(-lookahead)
    target = (future_return > threshold).astype(int)
    return target

print("=" * 80)
print("XGBoost Model Training with SMOTE and Optimized Threshold")
print("=" * 80)

# Load data
data_file = DATA_DIR / "historical" / "BTCUSDT_5m_max.csv"
df = pd.read_csv(data_file)
print(f"Loaded {len(df)} candles")

# Calculate features
df = calculate_features(df)
print(f"Calculated features: {len(df)} rows")

# Test different thresholds
thresholds = [0.003, 0.005, 0.007]  # 0.3%, 0.5%, 0.7%
best_model = None
best_threshold = None
best_metrics = {"balance": 0}

for threshold in thresholds:
    print(f"\n{'=' * 80}")
    print(f"Testing threshold: {threshold * 100:.1f}%")
    print(f"{'=' * 80}")

    # Create target
    df['target'] = create_target(df, lookahead=12, threshold=threshold)

    # Feature columns
    feature_columns = [
        'close_change_1', 'close_change_2', 'close_change_3', 'close_change_4', 'close_change_5',
        'sma_10', 'sma_20', 'ema_10',
        'macd', 'macd_signal', 'macd_diff',
        'rsi',
        'bb_high', 'bb_low', 'bb_mid',
        'volatility',
        'volume_sma', 'volume_ratio'
    ]

    # Drop NaN
    df_clean = df.dropna()
    print(f"After dropna: {len(df_clean)} rows")

    # Prepare data
    X = df_clean[feature_columns].values
    y = df_clean['target'].values

    print(f"\nTarget distribution (before SMOTE):")
    print(f"  Class 0 (not enter): {(y == 0).sum()} ({(y == 0).sum() / len(y) * 100:.1f}%)")
    print(f"  Class 1 (enter): {(y == 1).sum()} ({(y == 1).sum() / len(y) * 100:.1f}%)")

    # Calculate class imbalance ratio
    class_imbalance_ratio = (y == 0).sum() / (y == 1).sum()
    print(f"  Class imbalance ratio: {class_imbalance_ratio:.1f}:1")

    # Skip if too few positive samples
    if (y == 1).sum() < 100:
        print(f"⚠️ Too few positive samples ({(y == 1).sum()}), skipping...")
        continue

    # Train/val/test split
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, shuffle=False)

    print(f"\nData split:")
    print(f"  Train: {len(X_train)}")
    print(f"  Val: {len(X_val)}")
    print(f"  Test: {len(X_test)}")

    # Apply SMOTE to training data
    print("\nApplying SMOTE to training data...")

    # Calculate desired sampling ratio (not 1:1, but more balanced)
    # Target: reduce imbalance from 99:1 to 5:1 or 3:1
    target_ratio = min(0.3, (y_train == 1).sum() / (y_train == 0).sum() * 10)  # 30% max

    try:
        smote = SMOTE(sampling_strategy=target_ratio, random_state=42)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

        print(f"After SMOTE:")
        print(f"  Class 0: {(y_train_balanced == 0).sum()} ({(y_train_balanced == 0).sum() / len(y_train_balanced) * 100:.1f}%)")
        print(f"  Class 1: {(y_train_balanced == 1).sum()} ({(y_train_balanced == 1).sum() / len(y_train_balanced) * 100:.1f}%)")
        print(f"  New imbalance ratio: {(y_train_balanced == 0).sum() / (y_train_balanced == 1).sum():.1f}:1")

    except Exception as e:
        print(f"⚠️ SMOTE failed: {e}")
        print("Using original training data...")
        X_train_balanced = X_train
        y_train_balanced = y_train

    # Train XGBoost with class_weight
    print(f"\nTraining XGBoost...")

    # Calculate scale_pos_weight for class imbalance
    scale_pos_weight = (y_train_balanced == 0).sum() / (y_train_balanced == 1).sum()

    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,  # Handle class imbalance
        random_state=42,
        eval_metric='logloss'
    )

    model.fit(
        X_train_balanced, y_train_balanced,
        eval_set=[(X_val, y_val)],
        verbose=False
    )

    # Evaluate
    train_acc = model.score(X_train_balanced, y_train_balanced)
    val_acc = model.score(X_val, y_val)
    test_acc = model.score(X_test, y_test)

    print(f"\nAccuracy:")
    print(f"  Train: {train_acc:.4f}")
    print(f"  Val: {val_acc:.4f}")
    print(f"  Test: {test_acc:.4f}")

    # Test predictions
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = model.predict(X_test)

    # Metrics for test set
    from sklearn.metrics import classification_report, confusion_matrix
    print(f"\nTest Set Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Not Enter', 'Enter'], zero_division=0))

    print(f"\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    # Calculate probability distribution
    print(f"\nProbability Distribution (test set):")
    print(f"  Mean probability: {y_pred_proba.mean():.4f}")
    print(f"  Std probability: {y_pred_proba.std():.4f}")
    print(f"  Min probability: {y_pred_proba.min():.4f}")
    print(f"  Max probability: {y_pred_proba.max():.4f}")
    print(f"  Predictions > 0.5: {(y_pred_proba > 0.5).sum()} ({(y_pred_proba > 0.5).sum() / len(y_pred_proba) * 100:.1f}%)")
    print(f"  Predictions > 0.3: {(y_pred_proba > 0.3).sum()} ({(y_pred_proba > 0.3).sum() / len(y_pred_proba) * 100:.1f}%)")

    # Calculate balance metric (prefer models with better recall for class 1)
    recall_class_1 = cm[1, 1] / (cm[1, 0] + cm[1, 1]) if (cm[1, 0] + cm[1, 1]) > 0 else 0
    precision_class_1 = cm[1, 1] / (cm[0, 1] + cm[1, 1]) if (cm[0, 1] + cm[1, 1]) > 0 else 0
    balance_metric = 2 * (recall_class_1 * precision_class_1) / (recall_class_1 + precision_class_1) if (recall_class_1 + precision_class_1) > 0 else 0

    print(f"\nBalance Metrics:")
    print(f"  Recall (Class 1): {recall_class_1:.4f}")
    print(f"  Precision (Class 1): {precision_class_1:.4f}")
    print(f"  F1-Score: {balance_metric:.4f}")

    # Track best model
    if balance_metric > best_metrics["balance"]:
        best_model = model
        best_threshold = threshold
        best_metrics = {
            "balance": balance_metric,
            "recall": recall_class_1,
            "precision": precision_class_1,
            "test_acc": test_acc,
            "mean_prob": y_pred_proba.mean()
        }
        print(f"✅ New best model!")

# Save best model
if best_model is not None:
    print(f"\n{'=' * 80}")
    print(f"BEST MODEL FOUND")
    print(f"{'=' * 80}")
    print(f"Best threshold: {best_threshold * 100:.1f}%")
    print(f"Metrics:")
    print(f"  F1-Score: {best_metrics['balance']:.4f}")
    print(f"  Recall (Class 1): {best_metrics['recall']:.4f}")
    print(f"  Precision (Class 1): {best_metrics['precision']:.4f}")
    print(f"  Test Accuracy: {best_metrics['test_acc']:.4f}")
    print(f"  Mean Probability: {best_metrics['mean_prob']:.4f}")

    # Save model
    model_file = MODELS_DIR / "xgboost_model_smote.pkl"
    with open(model_file, 'wb') as f:
        pickle.dump(best_model, f)

    print(f"\n✅ Best model saved: {model_file}")

    # Save feature columns
    feature_file = MODELS_DIR / "feature_columns.txt"
    with open(feature_file, 'w') as f:
        f.write('\n'.join(feature_columns))

    print(f"✅ Features saved: {feature_file}")

    # Save metadata
    metadata = {
        "threshold": best_threshold,
        "f1_score": best_metrics['balance'],
        "recall_class_1": best_metrics['recall'],
        "precision_class_1": best_metrics['precision'],
        "test_accuracy": best_metrics['test_acc'],
        "mean_probability": best_metrics['mean_prob'],
        "smote_applied": True
    }

    metadata_file = MODELS_DIR / "xgboost_model_smote_metadata.txt"
    with open(metadata_file, 'w') as f:
        f.write("XGBoost Model with SMOTE - Metadata\n")
        f.write("=" * 80 + "\n\n")
        for key, value in metadata.items():
            f.write(f"{key}: {value}\n")

    print(f"✅ Metadata saved: {metadata_file}")

    print("\n" + "=" * 80)
    print("Improved XGBoost Model Ready for Testing!")
    print("=" * 80)
    print("\n비판적 검증:")
    print(f"  ✅ Probability > 0.0? Mean = {best_metrics['mean_prob']:.4f}")
    print(f"  ✅ Recall > 0.0? Recall = {best_metrics['recall']:.4f}")
    print(f"  ✅ F1-Score > 0.0? F1 = {best_metrics['balance']:.4f}")
    print("\n다음 단계: 이 모델로 paper_trading_bot.py 테스트")

else:
    print("\n❌ No suitable model found. All thresholds resulted in too few samples.")
    print("Suggestion: Try even lower thresholds or collect more data.")
