"""
Entry Model Training with Multi-Timeframe Features

목적:
- 현행 라벨링 (15min/0.3%) 유지
- Multi-timeframe features로 성능 향상
- LONG + SHORT Entry 모델 모두 재학습

전략:
- 기존 33 features + 새 36 multi-timeframe features = 69 features
- 학습 가능한 작업 유지하면서 정보 추가
- 단기 신호 + 중기 맥락 + 장기 추세

예상 효과:
- LONG F1: 15.8% → 20-25% (+26-58%)
- SHORT F1: 12.7% → 18-23% (+42-81%)
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import xgboost as xgb
from sklearn.metrics import classification_report, confusion_matrix
import ta
from datetime import datetime

# Import multi-timeframe features
import sys
sys.path.append(str(Path(__file__).parent))
from multi_timeframe_features import MultiTimeframeFeatures

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

MODELS_DIR.mkdir(parents=True, exist_ok=True)


def calculate_features(df):
    """Calculate ALL technical indicators (original + multi-timeframe)"""
    df = df.copy()

    # ========== ORIGINAL FEATURES (Phase 1 + Phase 2) ==========

    # Price changes
    for lag in range(1, 6):
        df[f'close_change_{lag}'] = df['close'].pct_change(lag)

    # Moving averages (original)
    df['sma_10'] = ta.trend.sma_indicator(df['close'], window=10)
    df['sma_20'] = ta.trend.sma_indicator(df['close'], window=20)
    df['ema_10'] = ta.trend.ema_indicator(df['close'], window=10)

    # MACD
    macd = ta.trend.MACD(df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_diff'] = macd.macd_diff()

    # RSI (original)
    df['rsi'] = ta.momentum.rsi(df['close'], window=14)

    # Bollinger Bands
    bb = ta.volatility.BollingerBands(df['close'])
    df['bb_high'] = bb.bollinger_hband()
    df['bb_low'] = bb.bollinger_lband()
    df['bb_mid'] = bb.bollinger_mavg()

    # Volatility (original)
    df['volatility'] = df['close'].pct_change().rolling(window=20).std()

    # Volume (original)
    df['volume_sma'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma']

    # Short-term features (Phase 2)
    df['ema_3'] = ta.trend.ema_indicator(df['close'], window=3)
    df['ema_5'] = ta.trend.ema_indicator(df['close'], window=5)
    df['price_mom_3'] = df['close'].pct_change(3)
    df['price_mom_5'] = df['close'].pct_change(5)
    df['rsi_5'] = ta.momentum.rsi(df['close'], window=5)
    df['rsi_7'] = ta.momentum.rsi(df['close'], window=7)
    df['volatility_5'] = df['close'].pct_change().rolling(window=5).std()
    df['volatility_10'] = df['close'].pct_change().rolling(window=10).std()
    df['volume_spike'] = df['volume'] / df['volume'].rolling(window=5).mean()
    df['volume_trend'] = df['volume'].rolling(window=3).mean() / df['volume'].rolling(window=10).mean()
    df['price_vs_ema3'] = (df['close'] - df['ema_3']) / df['ema_3']
    df['price_vs_ema5'] = (df['close'] - df['ema_5']) / df['ema_5']
    df['body_size'] = abs(df['close'] - df['open']) / df['open']
    df['upper_shadow'] = (df['high'] - df[['open', 'close']].max(axis=1)) / df['open']
    df['lower_shadow'] = (df[['open', 'close']].min(axis=1) - df['low']) / df['open']

    # ========== NEW MULTI-TIMEFRAME FEATURES ==========
    mtf_features = MultiTimeframeFeatures()
    df = mtf_features.calculate_all_features(df)

    return df


def create_target_long(df, lookahead=3, threshold=0.003):
    """Create LONG Entry target (upward movement)"""
    future_prices = df['close'].shift(-1).rolling(window=lookahead).apply(lambda x: x.max())
    future_return = (future_prices - df['close']) / df['close']
    target = (future_return > threshold).astype(int)
    return target


def create_target_short(df, lookahead=3, threshold=0.003):
    """Create SHORT Entry target (downward movement)"""
    future_prices = df['close'].shift(-1).rolling(window=lookahead).apply(lambda x: x.min())
    future_return = (df['close'] - future_prices) / df['close']
    target = (future_return > threshold).astype(int)
    return target


def train_model(df, direction='LONG', lookahead=3, threshold=0.003):
    """
    Train Entry model with multi-timeframe features

    Args:
        df: DataFrame with features
        direction: 'LONG' or 'SHORT'
        lookahead: candles to look ahead (3 = 15min)
        threshold: min return threshold (0.003 = 0.3%)

    Returns:
        model, metrics, metadata, feature_columns
    """
    print(f"\n{'=' * 80}")
    print(f"{direction} Entry Model Training (Multi-Timeframe Features)")
    print(f"Lookahead: {lookahead} ({lookahead * 5} min), Threshold: {threshold * 100:.1f}%")
    print(f"{'=' * 80}")

    # Create target
    if direction == 'LONG':
        df['target'] = create_target_long(df, lookahead, threshold)
    else:
        df['target'] = create_target_short(df, lookahead, threshold)

    # Original features (33)
    original_features = [
        # Phase 1 (18)
        'close_change_1', 'close_change_2', 'close_change_3', 'close_change_4', 'close_change_5',
        'sma_10', 'sma_20', 'ema_10',
        'macd', 'macd_signal', 'macd_diff',
        'rsi',
        'bb_high', 'bb_low', 'bb_mid',
        'volatility',
        'volume_sma', 'volume_ratio',

        # Phase 2 (15)
        'ema_3', 'ema_5',
        'price_mom_3', 'price_mom_5',
        'rsi_5', 'rsi_7',
        'volatility_5', 'volatility_10',
        'volume_spike', 'volume_trend',
        'price_vs_ema3', 'price_vs_ema5',
        'body_size', 'upper_shadow', 'lower_shadow'
    ]

    # Multi-timeframe features (36)
    mtf_features = MultiTimeframeFeatures()
    multi_features = mtf_features.get_feature_names()

    # Combine all features
    feature_columns = original_features + multi_features

    print(f"\nTotal Features: {len(feature_columns)}")
    print(f"  Original (Phase 1+2): {len(original_features)}")
    print(f"  Multi-Timeframe: {len(multi_features)}")

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
    if (y == 1).sum() < 500:
        print(f"⚠️ Too few positive samples ({(y == 1).sum()}), may overfit...")

    # Time-based train/val/test split
    train_size = int(len(X) * 0.6)
    val_size = int(len(X) * 0.2)

    X_train = X[:train_size]
    y_train = y[:train_size]
    X_val = X[train_size:train_size+val_size]
    y_val = y[train_size:train_size+val_size]
    X_test = X[train_size+val_size:]
    y_test = y[train_size+val_size:]

    print(f"\nData split (time-based):")
    print(f"  Train: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
    print(f"  Val: {len(X_val)} ({len(X_val)/len(X)*100:.1f}%)")
    print(f"  Test: {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")

    # Apply SMOTE
    print("\nApplying SMOTE...")
    pos_ratio = (y_train == 1).sum() / (y_train == 0).sum()
    target_ratio = min(0.3, pos_ratio * 5)

    try:
        smote = SMOTE(sampling_strategy=target_ratio, random_state=42)
        X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
        print(f"After SMOTE:")
        print(f"  Class 0: {(y_train_balanced == 0).sum()} ({(y_train_balanced == 0).sum() / len(y_train_balanced) * 100:.1f}%)")
        print(f"  Class 1: {(y_train_balanced == 1).sum()} ({(y_train_balanced == 1).sum() / len(y_train_balanced) * 100:.1f}%)")
    except Exception as e:
        print(f"⚠️ SMOTE failed: {e}")
        X_train_balanced = X_train
        y_train_balanced = y_train

    # Train XGBoost
    print(f"\nTraining XGBoost with {len(feature_columns)} features...")
    scale_pos_weight = (y_train_balanced == 0).sum() / (y_train_balanced == 1).sum()

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

    # Metrics
    print(f"\nTest Set Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Not Enter', 'Enter'], zero_division=0))

    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix:")
    print(cm)

    # Calculate F1
    recall = cm[1, 1] / (cm[1, 0] + cm[1, 1]) if (cm[1, 0] + cm[1, 1]) > 0 else 0
    precision = cm[1, 1] / (cm[0, 1] + cm[1, 1]) if (cm[0, 1] + cm[1, 1]) > 0 else 0
    f1 = 2 * (recall * precision) / (recall + precision) if (recall + precision) > 0 else 0

    print(f"\nKey Metrics:")
    print(f"  Recall: {recall:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  F1-Score: {f1:.4f}")

    # Probability distribution
    print(f"\nProbability Distribution (test set):")
    print(f"  Mean: {y_pred_proba.mean():.4f}")
    for threshold_check in [0.5, 0.6, 0.7, 0.8, 0.9]:
        count = (y_pred_proba > threshold_check).sum()
        pct = count / len(y_pred_proba) * 100
        print(f"  Prob > {threshold_check}: {count} ({pct:.2f}%)")

    # Feature importance (top 15)
    print(f"\nTop 15 Feature Importance:")
    importance = model.feature_importances_
    feature_importance = sorted(zip(feature_columns, importance), key=lambda x: x[1], reverse=True)
    for i, (feat, imp) in enumerate(feature_importance[:15], 1):
        print(f"  {i}. {feat}: {imp:.4f}")

    metrics = {
        'f1_score': f1,
        'recall': recall,
        'precision': precision,
        'test_acc': test_acc,
        'mean_prob': y_pred_proba.mean(),
    }

    metadata = {
        'direction': direction,
        'lookahead': lookahead,
        'threshold': threshold,
        'feature_count': len(feature_columns),
        'phase': 'multi_timeframe',
        'train_date': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }

    return model, metrics, metadata, feature_columns


if __name__ == "__main__":
    print("=" * 80)
    print("Entry Model Training with Multi-Timeframe Features")
    print("=" * 80)

    # Load data
    data_file = DATA_DIR / "historical" / "BTCUSDT_5m_max.csv"
    df = pd.read_csv(data_file)
    print(f"✅ Loaded {len(df)} candles")

    # Calculate features (all 69 features)
    print("\n계산 중: Original (33) + Multi-Timeframe (36) features...")
    df = calculate_features(df)
    print(f"✅ Features calculated: {len(df)} rows")

    # Best config (from previous validation)
    best_config = {'lookahead': 3, 'threshold': 0.003}  # 15 min, 0.3%

    # Train LONG Entry Model
    print("\n" + "=" * 80)
    print("1/2: LONG Entry Model")
    print("=" * 80)

    model_long, metrics_long, metadata_long, features_long = train_model(
        df.copy(),
        direction='LONG',
        lookahead=best_config['lookahead'],
        threshold=best_config['threshold']
    )

    # Save LONG model
    model_file_long = MODELS_DIR / "xgboost_long_entry_multitimeframe.pkl"
    with open(model_file_long, 'wb') as f:
        pickle.dump(model_long, f)
    print(f"\n✅ LONG model saved: {model_file_long}")

    # Save feature list
    feature_file_long = MODELS_DIR / "xgboost_long_entry_multitimeframe_features.txt"
    with open(feature_file_long, 'w') as f:
        f.write('\n'.join(features_long))
    print(f"✅ Features saved: {feature_file_long}")

    # Train SHORT Entry Model
    print("\n" + "=" * 80)
    print("2/2: SHORT Entry Model")
    print("=" * 80)

    model_short, metrics_short, metadata_short, features_short = train_model(
        df.copy(),
        direction='SHORT',
        lookahead=best_config['lookahead'],
        threshold=best_config['threshold']
    )

    # Save SHORT model
    model_file_short = MODELS_DIR / "xgboost_short_entry_multitimeframe.pkl"
    with open(model_file_short, 'wb') as f:
        pickle.dump(model_short, f)
    print(f"\n✅ SHORT model saved: {model_file_short}")

    # Save feature list
    feature_file_short = MODELS_DIR / "xgboost_short_entry_multitimeframe_features.txt"
    with open(feature_file_short, 'w') as f:
        f.write('\n'.join(features_short))
    print(f"✅ Features saved: {feature_file_short}")

    # Summary
    print("\n" + "=" * 80)
    print("Training Complete!")
    print("=" * 80)

    print(f"\n현행 모델 vs 신규 모델 비교:")
    print(f"\n{'Metric':<20} {'현행 LONG':<15} {'신규 LONG':<15} {'변화':<15}")
    print("-" * 65)
    current_long_f1 = 0.158
    new_long_f1 = metrics_long['f1_score']
    change_long = (new_long_f1 - current_long_f1) / current_long_f1 * 100
    print(f"{'F1 Score':<20} {current_long_f1:<15.4f} {new_long_f1:<15.4f} {change_long:+.1f}%")
    print(f"{'Features':<20} {33:<15} {len(features_long):<15} {'+36':<15}")

    print(f"\n{'Metric':<20} {'현행 SHORT':<15} {'신규 SHORT':<15} {'변화':<15}")
    print("-" * 65)
    current_short_f1 = 0.127
    new_short_f1 = metrics_short['f1_score']
    change_short = (new_short_f1 - current_short_f1) / current_short_f1 * 100
    print(f"{'F1 Score':<20} {current_short_f1:<15.4f} {new_short_f1:<15.4f} {change_short:+.1f}%")
    print(f"{'Features':<20} {33:<15} {len(features_short):<15} {'+36':<15}")

    print(f"\n다음 단계: 백테스트로 실제 성과 검증")
    print(f"예상:")
    print(f"  - Win Rate: 70.6% → 72-75% (+1-4%p)")
    print(f"  - Returns: +4.19% → +5-6.5% (+19-55%)")
