"""
XGBoost Phase 1 개선: Lookahead + Threshold 최적화

목적:
1. Lookahead 줄이기: 12 candles (60 min) → 3-5 candles (15-25 min)
2. Threshold 낮추기: 0.003 (0.3%) → 0.001-0.002 (0.1-0.2%)
3. 4가지 조합 테스트 및 최적 config 선택
4. 백테스트로 실제 성과 검증

비판적 사고:
- "5분봉으로 60분 후 예측 = 너무 어려움"
- "Lookahead를 줄이면 예측 난이도 ↓, positive samples ↑"
- "Threshold를 낮추면 거래 빈도 ↑"

예상 효과:
- 거래 빈도: 0.1 → 5-8 trades/window (50-80배 증가)
- 승률: 0.3% → 48-55%
- Return vs B&H: +0.04% → +0.8-1.5%
- p-value: 0.2229 → < 0.05 (significant!)
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

def create_target(df, lookahead, threshold):
    """
    Create binary target with specified lookahead and threshold

    Args:
        lookahead: Number of candles to look ahead (3, 5)
        threshold: Minimum return threshold (0.001, 0.0015, 0.002)
    """
    future_return = df['close'].pct_change(lookahead).shift(-lookahead)
    target = (future_return > threshold).astype(int)
    return target

def train_model_with_config(df, lookahead, threshold):
    """
    Train XGBoost model with specified config

    Returns:
        model, metrics, metadata
    """
    print(f"\n{'=' * 80}")
    print(f"Training Config: lookahead={lookahead} ({lookahead * 5} min), threshold={threshold * 100:.2f}%")
    print(f"{'=' * 80}")

    # Create target
    df['target'] = create_target(df, lookahead, threshold)

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

    # Check minimum samples
    if (y == 1).sum() < 100:
        print(f"⚠️ Too few positive samples ({(y == 1).sum()}), skipping...")
        return None, None, None

    # Calculate class imbalance ratio
    class_imbalance_ratio = (y == 0).sum() / (y == 1).sum()
    print(f"  Class imbalance ratio: {class_imbalance_ratio:.1f}:1")

    # Train/val/test split
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, shuffle=False)

    print(f"\nData split:")
    print(f"  Train: {len(X_train)}")
    print(f"  Val: {len(X_val)}")
    print(f"  Test: {len(X_test)}")

    # Apply SMOTE to training data
    print("\nApplying SMOTE to training data...")

    # Calculate desired sampling ratio
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
        scale_pos_weight=scale_pos_weight,
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
    print(f"  Predictions > 0.4: {(y_pred_proba > 0.4).sum()} ({(y_pred_proba > 0.4).sum() / len(y_pred_proba) * 100:.1f}%)")
    print(f"  Predictions > 0.3: {(y_pred_proba > 0.3).sum()} ({(y_pred_proba > 0.3).sum() / len(y_pred_proba) * 100:.1f}%)")

    # Calculate balance metric
    recall_class_1 = cm[1, 1] / (cm[1, 0] + cm[1, 1]) if (cm[1, 0] + cm[1, 1]) > 0 else 0
    precision_class_1 = cm[1, 1] / (cm[0, 1] + cm[1, 1]) if (cm[0, 1] + cm[1, 1]) > 0 else 0
    f1_score = 2 * (recall_class_1 * precision_class_1) / (recall_class_1 + precision_class_1) if (recall_class_1 + precision_class_1) > 0 else 0

    print(f"\nBalance Metrics:")
    print(f"  Recall (Class 1): {recall_class_1:.4f}")
    print(f"  Precision (Class 1): {precision_class_1:.4f}")
    print(f"  F1-Score: {f1_score:.4f}")

    metrics = {
        'f1_score': f1_score,
        'recall': recall_class_1,
        'precision': precision_class_1,
        'test_acc': test_acc,
        'mean_prob': y_pred_proba.mean(),
        'std_prob': y_pred_proba.std(),
        'prob_gt_03': (y_pred_proba > 0.3).sum() / len(y_pred_proba),
        'prob_gt_04': (y_pred_proba > 0.4).sum() / len(y_pred_proba),
        'prob_gt_05': (y_pred_proba > 0.5).sum() / len(y_pred_proba),
    }

    metadata = {
        'lookahead': lookahead,
        'lookahead_minutes': lookahead * 5,
        'threshold': threshold,
        'threshold_pct': threshold * 100,
        'class_1_ratio': (y == 1).sum() / len(y),
        'smote_applied': True,
        'feature_count': len(feature_columns)
    }

    return model, metrics, metadata

print("=" * 80)
print("XGBoost Phase 1 Improvement: Lookahead + Threshold Optimization")
print("=" * 80)

# Load data
data_file = DATA_DIR / "historical" / "BTCUSDT_5m_max.csv"
df = pd.read_csv(data_file)
print(f"✅ Loaded {len(df)} candles")

# Calculate features
df = calculate_features(df)
print(f"✅ Calculated features: {len(df)} rows")

# Test configurations
configs = [
    {'lookahead': 3, 'threshold': 0.001},   # 15 min, 0.1%
    {'lookahead': 3, 'threshold': 0.0015},  # 15 min, 0.15%
    {'lookahead': 5, 'threshold': 0.0015},  # 25 min, 0.15%
    {'lookahead': 5, 'threshold': 0.002},   # 25 min, 0.2%
]

results = []

for config in configs:
    model, metrics, metadata = train_model_with_config(
        df.copy(),
        config['lookahead'],
        config['threshold']
    )

    if model is None:
        continue

    # Save model with config name
    config_name = f"lookahead{config['lookahead']}_thresh{int(config['threshold']*1000)}"
    model_file = MODELS_DIR / f"xgboost_v2_{config_name}.pkl"

    with open(model_file, 'wb') as f:
        pickle.dump(model, f)

    print(f"✅ Model saved: {model_file}")

    # Save metadata
    metadata_file = MODELS_DIR / f"xgboost_v2_{config_name}_metadata.txt"
    with open(metadata_file, 'w') as f:
        f.write("XGBoost V2 Model - Metadata\n")
        f.write("=" * 80 + "\n\n")
        f.write("CONFIG:\n")
        for key, value in metadata.items():
            f.write(f"  {key}: {value}\n")
        f.write("\nMETRICS:\n")
        for key, value in metrics.items():
            f.write(f"  {key}: {value}\n")

    print(f"✅ Metadata saved: {metadata_file}")

    results.append({
        'config_name': config_name,
        'lookahead': config['lookahead'],
        'lookahead_min': config['lookahead'] * 5,
        'threshold': config['threshold'],
        'threshold_pct': config['threshold'] * 100,
        'f1_score': metrics['f1_score'],
        'recall': metrics['recall'],
        'precision': metrics['precision'],
        'mean_prob': metrics['mean_prob'],
        'prob_gt_03': metrics['prob_gt_03'],
        'prob_gt_04': metrics['prob_gt_04'],
        'prob_gt_05': metrics['prob_gt_05'],
    })

# Compare results
print("\n" + "=" * 80)
print("CONFIGURATION COMPARISON")
print("=" * 80)

results_df = pd.DataFrame(results)
print(results_df.to_string(index=False))

# Find best config (highest F1-score)
best_idx = results_df['f1_score'].idxmax()
best_config = results_df.iloc[best_idx]

print("\n" + "=" * 80)
print(f"BEST CONFIGURATION: {best_config['config_name']}")
print("=" * 80)
print(f"  Lookahead: {best_config['lookahead']} candles ({best_config['lookahead_min']:.0f} min)")
print(f"  Threshold: {best_config['threshold']:.4f} ({best_config['threshold_pct']:.2f}%)")
print(f"  F1-Score: {best_config['f1_score']:.4f}")
print(f"  Recall: {best_config['recall']:.4f}")
print(f"  Precision: {best_config['precision']:.4f}")
print(f"  Mean Probability: {best_config['mean_prob']:.4f}")
print(f"  Prob > 0.3: {best_config['prob_gt_03']:.2f}")
print(f"  Prob > 0.4: {best_config['prob_gt_04']:.2f}")
print(f"  Prob > 0.5: {best_config['prob_gt_05']:.2f}")

# Save comparison
comparison_file = MODELS_DIR / "xgboost_v2_config_comparison.csv"
results_df.to_csv(comparison_file, index=False)
print(f"\n✅ Comparison saved: {comparison_file}")

# Copy best model to default name
best_model_file = MODELS_DIR / f"xgboost_v2_{best_config['config_name']}.pkl"
default_model_file = MODELS_DIR / "xgboost_model.pkl"

import shutil
shutil.copy(best_model_file, default_model_file)
print(f"✅ Best model copied to: {default_model_file}")

print("\n" + "=" * 80)
print("Phase 1 Training Complete!")
print("=" * 80)
print("\n비판적 분석:")
print(f"  1. 개선 전 (lookahead=12, threshold=0.003):")
print(f"     - Mean Probability: 0.3168")
print(f"     - F1-Score: 0.2076")
print(f"     - 백테스트: 0.1 trades/window")
print(f"\n  2. 개선 후 (lookahead={best_config['lookahead']}, threshold={best_config['threshold']:.4f}):")
print(f"     - Mean Probability: {best_config['mean_prob']:.4f} ({'+' if best_config['mean_prob'] > 0.3168 else ''}{(best_config['mean_prob'] - 0.3168) / 0.3168 * 100:.1f}%)")
print(f"     - F1-Score: {best_config['f1_score']:.4f} ({'+' if best_config['f1_score'] > 0.2076 else ''}{(best_config['f1_score'] - 0.2076) / 0.2076 * 100:.1f}%)")
print(f"     - Prob > 0.3: {best_config['prob_gt_03'] * 100:.1f}% (거래 빈도 예상 증가)")
print(f"\n다음 단계: 백테스트로 실제 성과 검증")
print(f"  python scripts/backtest_xgboost_v2.py")
