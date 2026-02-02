"""
XGBoost Phase 2: Short-term Features 추가

목적:
1. Phase 1 최적 config 사용 (lookahead=3, threshold=0.001)
2. Short-term features 15개 추가 (15분 예측에 최적화)
3. 거래 품질 향상 (평균 이익 > 거래 비용)
4. 승률 향상 (45.9% → 50-55%)

비판적 사고:
- "Phase 1: 거래는 하지만 수익 낮음"
- "원인: 거래 품질 낮음 (features가 15분 예측에 부적절)"
- "해결: Short-term features 추가"

추가 Features:
1. Fast EMAs: ema_3, ema_5 (15분, 25분)
2. Short-term momentum: price_mom_3, price_mom_5
3. Short-term RSI: rsi_5, rsi_7 (25분, 35분)
4. Short-term volatility: volatility_5, volatility_10
5. Volume patterns: volume_spike, volume_trend
6. Price position: price_vs_ema3, price_vs_ema5
7. Candlestick: body_size, upper_shadow, lower_shadow

예상 효과:
- 거래 품질 향상: 평균 이익 0.05% → 0.2-0.3%
- 승률 향상: 45.9% → 50-55%
- Return vs B&H: -1.86% → +0.5-1.5%
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import xgboost as xgb
import ta

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

MODELS_DIR.mkdir(parents=True, exist_ok=True)

def calculate_features(df):
    """Calculate technical indicators with SHORT-TERM features"""
    df = df.copy()

    # ========== ORIGINAL FEATURES (Phase 1) ==========
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

    # ========== NEW SHORT-TERM FEATURES (Phase 2) ==========

    # 1. Fast EMAs (15분, 25분)
    df['ema_3'] = ta.trend.ema_indicator(df['close'], window=3)  # 15 min
    df['ema_5'] = ta.trend.ema_indicator(df['close'], window=5)  # 25 min

    # 2. Short-term momentum
    df['price_mom_3'] = df['close'].pct_change(3)  # 15 min momentum
    df['price_mom_5'] = df['close'].pct_change(5)  # 25 min momentum

    # 3. Short-term RSI
    df['rsi_5'] = ta.momentum.rsi(df['close'], window=5)  # 25 min RSI
    df['rsi_7'] = ta.momentum.rsi(df['close'], window=7)  # 35 min RSI

    # 4. Short-term volatility
    df['volatility_5'] = df['close'].pct_change().rolling(window=5).std()
    df['volatility_10'] = df['close'].pct_change().rolling(window=10).std()

    # 5. Volume patterns
    df['volume_spike'] = df['volume'] / df['volume'].rolling(window=5).mean()
    df['volume_trend'] = df['volume'].rolling(window=3).mean() / df['volume'].rolling(window=10).mean()

    # 6. Price position (vs fast EMAs)
    df['price_vs_ema3'] = (df['close'] - df['ema_3']) / df['ema_3']
    df['price_vs_ema5'] = (df['close'] - df['ema_5']) / df['ema_5']

    # 7. Candlestick patterns
    df['body_size'] = abs(df['close'] - df['open']) / df['open']
    df['upper_shadow'] = (df['high'] - df[['open', 'close']].max(axis=1)) / df['open']
    df['lower_shadow'] = (df[['open', 'close']].min(axis=1) - df['low']) / df['open']

    return df

def create_target(df, lookahead, threshold):
    """
    Create binary target with specified lookahead and threshold
    """
    future_return = df['close'].pct_change(lookahead).shift(-lookahead)
    target = (future_return > threshold).astype(int)
    return target

def train_model_with_config(df, lookahead, threshold):
    """
    Train XGBoost model with Phase 2 features
    """
    print(f"\n{'=' * 80}")
    print(f"Phase 2 Training: lookahead={lookahead} ({lookahead * 5} min), threshold={threshold * 100:.2f}%")
    print(f"{'=' * 80}")

    # Create target
    df['target'] = create_target(df, lookahead, threshold)

    # Feature columns (Original + NEW Short-term)
    feature_columns = [
        # Original features (18)
        'close_change_1', 'close_change_2', 'close_change_3', 'close_change_4', 'close_change_5',
        'sma_10', 'sma_20', 'ema_10',
        'macd', 'macd_signal', 'macd_diff',
        'rsi',
        'bb_high', 'bb_low', 'bb_mid',
        'volatility',
        'volume_sma', 'volume_ratio',

        # NEW Short-term features (15)
        'ema_3', 'ema_5',  # Fast EMAs
        'price_mom_3', 'price_mom_5',  # Short-term momentum
        'rsi_5', 'rsi_7',  # Short-term RSI
        'volatility_5', 'volatility_10',  # Short-term volatility
        'volume_spike', 'volume_trend',  # Volume patterns
        'price_vs_ema3', 'price_vs_ema5',  # Price position
        'body_size', 'upper_shadow', 'lower_shadow'  # Candlestick
    ]

    print(f"Total Features: {len(feature_columns)} (Original: 18, NEW: 15)")

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

    # Train/val/test split
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, shuffle=False)

    print(f"\nData split:")
    print(f"  Train: {len(X_train)}")
    print(f"  Val: {len(X_val)}")
    print(f"  Test: {len(X_test)}")

    # Apply SMOTE
    print("\nApplying SMOTE...")
    target_ratio = min(0.3, (y_train == 1).sum() / (y_train == 0).sum() * 10)

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

    # Metrics
    from sklearn.metrics import classification_report, confusion_matrix
    print(f"\nTest Set Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['Not Enter', 'Enter'], zero_division=0))

    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix:")
    print(cm)

    # Probability distribution
    print(f"\nProbability Distribution (test set):")
    print(f"  Mean probability: {y_pred_proba.mean():.4f}")
    print(f"  Std probability: {y_pred_proba.std():.4f}")
    print(f"  Prob > 0.3: {(y_pred_proba > 0.3).sum()} ({(y_pred_proba > 0.3).sum() / len(y_pred_proba) * 100:.1f}%)")
    print(f"  Prob > 0.4: {(y_pred_proba > 0.4).sum()} ({(y_pred_proba > 0.4).sum() / len(y_pred_proba) * 100:.1f}%)")
    print(f"  Prob > 0.5: {(y_pred_proba > 0.5).sum()} ({(y_pred_proba > 0.5).sum() / len(y_pred_proba) * 100:.1f}%)")

    # Calculate metrics
    recall_class_1 = cm[1, 1] / (cm[1, 0] + cm[1, 1]) if (cm[1, 0] + cm[1, 1]) > 0 else 0
    precision_class_1 = cm[1, 1] / (cm[0, 1] + cm[1, 1]) if (cm[0, 1] + cm[1, 1]) > 0 else 0
    f1_score = 2 * (recall_class_1 * precision_class_1) / (recall_class_1 + precision_class_1) if (recall_class_1 + precision_class_1) > 0 else 0

    print(f"\nBalance Metrics:")
    print(f"  Recall (Class 1): {recall_class_1:.4f}")
    print(f"  Precision (Class 1): {precision_class_1:.4f}")
    print(f"  F1-Score: {f1_score:.4f}")

    # Feature importance (top 10)
    print(f"\nTop 10 Feature Importance:")
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
    }

    metadata = {
        'lookahead': lookahead,
        'threshold': threshold,
        'feature_count': len(feature_columns),
        'phase': 2
    }

    return model, metrics, metadata, feature_columns


if __name__ == "__main__":
    print("=" * 80)
    print("XGBoost Phase 2: Short-term Features")
    print("=" * 80)

    # Load data
    data_file = DATA_DIR / "historical" / "BTCUSDT_5m_max.csv"
    df = pd.read_csv(data_file)
    print(f"✅ Loaded {len(df)} candles")

    # Calculate features (with SHORT-TERM features)
    df = calculate_features(df)
    print(f"✅ Calculated features: {len(df)} rows")

    # Use Phase 1 best config
    best_config = {'lookahead': 3, 'threshold': 0.001}  # 15 min, 0.1%

    print(f"\n{'=' * 80}")
    print(f"Training with Phase 1 Best Config + Short-term Features")
    print(f"{'=' * 80}")

    model, metrics, metadata, feature_columns = train_model_with_config(
        df.copy(),
        best_config['lookahead'],
        best_config['threshold']
    )

    if model is not None:
        # Save model
        config_name = f"lookahead{best_config['lookahead']}_thresh{int(best_config['threshold']*1000)}_phase2"
        model_file = MODELS_DIR / f"xgboost_v3_{config_name}.pkl"

        with open(model_file, 'wb') as f:
            pickle.dump(model, f)
        print(f"\n✅ Model saved: {model_file}")

        # Save feature columns
        feature_file = MODELS_DIR / f"xgboost_v3_{config_name}_features.txt"
        with open(feature_file, 'w') as f:
            f.write('\n'.join(feature_columns))
        print(f"✅ Features saved: {feature_file}")

        # Save metadata
        metadata_file = MODELS_DIR / f"xgboost_v3_{config_name}_metadata.txt"
        with open(metadata_file, 'w') as f:
            f.write("XGBoost V3 (Phase 2) - Metadata\n")
            f.write("=" * 80 + "\n\n")
            f.write("CONFIG:\n")
            for key, value in metadata.items():
                f.write(f"  {key}: {value}\n")
            f.write("\nMETRICS:\n")
            for key, value in metrics.items():
                f.write(f"  {key}: {value}\n")
        print(f"✅ Metadata saved: {metadata_file}")

        # Copy to default name
        default_model_file = MODELS_DIR / "xgboost_model_phase2.pkl"
        import shutil
        shutil.copy(model_file, default_model_file)
        print(f"✅ Copied to: {default_model_file}")

        print("\n" + "=" * 80)
        print("Phase 2 Training Complete!")
        print("=" * 80)
        print("\n비판적 분석:")
        print(f"  Phase 1 (18 features):")
        print(f"    - F1-Score: 0.3321")
        print(f"    - Mean Probability: 0.4249")
        print(f"    - Backtest: -1.86% vs B&H")
        print(f"\n  Phase 2 (33 features, +15 short-term):")
        print(f"    - F1-Score: {metrics['f1_score']:.4f} ({'+' if metrics['f1_score'] > 0.3321 else ''}{(metrics['f1_score'] - 0.3321) / 0.3321 * 100:.1f}%)")
        print(f"    - Mean Probability: {metrics['mean_prob']:.4f} ({'+' if metrics['mean_prob'] > 0.4249 else ''}{(metrics['mean_prob'] - 0.4249) / 0.4249 * 100:.1f}%)")
        print(f"    - Backtest: 다음 단계")
        print(f"\n다음 단계: 백테스트로 실제 성과 검증")
        print(f"  python scripts/backtest_xgboost_v3_phase2.py")
    else:
        print("\n❌ Training failed!")
