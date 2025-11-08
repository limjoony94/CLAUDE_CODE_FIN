"""
Train XGBoost Phase 3: With 15-minute Features

목표: Bull market detection 개선
- 5분 기본 features (33개)
- 15분 long-term features (14개)
- Total: 47개 features
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
import pickle
from pathlib import Path


if __name__ == "__main__":
    import sys
    PROJECT_ROOT = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(PROJECT_ROOT))

    from scripts.production.train_xgboost_improved_v3_phase2 import calculate_features as calculate_features_with_15m
    DATA_DIR = PROJECT_ROOT / "data"
    MODELS_DIR = PROJECT_ROOT / "models"

    # Parameters
    LOOKAHEAD = 3  # 15 minutes (3 × 5 min)
    PROFIT_THRESHOLD = 0.001  # 0.1%

    print("=" * 80)
    print("XGBoost Phase 3: 5분 + 15분 Multi-timeframe Features")
    print("=" * 80)

    # Load enhanced data (5m with 15m features)
    data_file = DATA_DIR / "historical" / "BTCUSDT_5m_with_15m_features.csv"
    df = pd.read_csv(data_file)
    print(f"✅ Loaded enhanced data: {len(df)} candles")

    # Calculate 5m features
    df = calculate_features_with_15m(df)
    print(f"✅ Calculated 5m features")

    # Drop NaN
    df = df.dropna()
    print(f"After dropna: {len(df)} rows")

    # Create target
    df['future_return'] = df['close'].pct_change(LOOKAHEAD).shift(-LOOKAHEAD) * 100
    df['target'] = (df['future_return'] > PROFIT_THRESHOLD).astype(int)

    # Remove last LOOKAHEAD rows
    df = df[:-LOOKAHEAD].copy()

    # Features: 5m features + 15m features
    feature_cols_5m = [
        'returns', 'volatility', 'volume_change',
        'rsi', 'rsi_ma', 'rsi_std',
        'macd', 'macd_signal', 'macd_diff',
        'bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'bb_position',
        'ema_9', 'ema_21', 'ema_50',
        'atr', 'adx',
        'obv', 'obv_ema',
        'stoch_k', 'stoch_d',
        # SHORT-TERM features
        'returns_short', 'volatility_short',
        'rsi_short', 'macd_short',
        'ema_5', 'ema_10',
        'momentum_short', 'roc_short', 'cci_short',
        'stoch_rsi_k', 'stoch_rsi_d',
        'williams_r'
    ]

    feature_cols_15m = [
        'ema_50_15m', 'ema_200_15m', 'ema_cross_15m', 'ema_dist_15m',
        'rsi_15m', 'macd_15m', 'macd_signal_15m', 'macd_diff_15m',
        'adx_15m', 'volatility_15m', 'momentum_15m',
        'support_15m', 'resistance_15m', 'price_position_15m'
    ]

    feature_columns = feature_cols_5m + feature_cols_15m
    print(f"\nTotal features: {len(feature_columns)}")
    print(f"  - 5m features: {len(feature_cols_5m)}")
    print(f"  - 15m features: {len(feature_cols_15m)}")

    # Prepare data
    X = df[feature_columns].values
    y = df['target'].values

    print(f"\nTarget distribution (before SMOTE):")
    print(f"  Class 0 (not enter): {(y == 0).sum()} ({(y == 0).mean() * 100:.1f}%)")
    print(f"  Class 1 (enter): {(y == 1).sum()} ({(y == 1).mean() * 100:.1f}%)")

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # SMOTE
    smote = SMOTE(random_state=42)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

    print(f"\nAfter SMOTE:")
    print(f"  Class 0: {(y_train_balanced == 0).sum()}")
    print(f"  Class 1: {(y_train_balanced == 1).sum()}")

    # Train XGBoost
    print(f"\n{'=' * 80}")
    print("Training XGBoost with Multi-timeframe Features...")
    print(f"{'=' * 80}")

    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        tree_method='hist'
    )

    model.fit(
        X_train_balanced,
        y_train_balanced,
        eval_set=[(X_test, y_test)],
        verbose=50
    )

    # Evaluate
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    print(f"\n{'=' * 80}")
    print("Evaluation Results")
    print(f"{'=' * 80}\n")

    print(classification_report(y_test, y_pred, target_names=['Not Enter', 'Enter']))

    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_test, y_pred)
    print(cm)

    f1 = f1_score(y_test, y_pred)
    print(f"\nF1-Score: {f1:.4f}")

    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    print(f"\nTop 20 Most Important Features:")
    print(feature_importance.head(20).to_string(index=False))

    print(f"\n15m Features in Top 20:")
    top20_15m = feature_importance.head(20)[feature_importance.head(20)['feature'].str.contains('_15m')]
    if len(top20_15m) > 0:
        print(top20_15m.to_string(index=False))
        print(f"✅ {len(top20_15m)} out of 20 are 15m features!")
    else:
        print("No 15m features in top 20")

    # Save model
    model_file = MODELS_DIR / "xgboost_phase3_with_15m_features.pkl"
    with open(model_file, 'wb') as f:
        pickle.dump(model, f)
    print(f"\n✅ Model saved: {model_file}")

    # Save feature columns
    feature_file = MODELS_DIR / "xgboost_phase3_with_15m_features_columns.txt"
    with open(feature_file, 'w') as f:
        for col in feature_columns:
            f.write(f"{col}\n")
    print(f"✅ Feature columns saved: {feature_file}")

    print(f"\n{'=' * 80}")
    print("XGBoost Phase 3 Training Complete!")
    print(f"{'=' * 80}")
