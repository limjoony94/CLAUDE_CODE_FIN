"""
Train Regime-Specific XGBoost Models

비판적 사고:
"Bull/Bear/Sideways 시장에서 중요한 패턴이 다르다면,
각 regime에 특화된 모델을 훈련하는 것이 더 나을 것이다"

Hypothesis:
- Bull market: Momentum indicators 중요
- Bear market: Support/resistance levels 중요
- Sideways: Volatility breakout 중요

→ 각 regime별 모델이 general model보다 우수할 것
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import sys
import json
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import xgboost as xgb

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.production.train_xgboost_improved_v3_phase2 import calculate_features
from scripts.production.advanced_technical_features import AdvancedTechnicalFeatures
from scripts.production.regime_detector import RegimeDetector

DATA_DIR = PROJECT_ROOT / "data" / "historical"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# Training parameters
LOOKAHEAD = 3  # candles
THRESHOLD = 0.003  # 0.3%
N_FOLDS = 5

# Feature selection (same as Phase 4 Advanced)
BASELINE_FEATURES = [
    'rsi', 'macd', 'macd_signal',
    'bb_high', 'bb_mid', 'bb_low',
    'close_change_1', 'close_change_3',
    'volume_ratio', 'volatility'
]

def create_labels(df, lookahead=LOOKAHEAD, threshold=THRESHOLD):
    """Create binary labels for profitable trades"""
    labels = []

    for i in range(len(df)):
        if i + lookahead >= len(df):
            labels.append(0)
            continue

        future_prices = df['close'].iloc[i+1:i+1+lookahead]
        current_price = df['close'].iloc[i]
        future_returns = (future_prices / current_price) - 1

        max_return = future_returns.max()

        if max_return >= threshold:
            labels.append(1)
        else:
            labels.append(0)

    return labels

def train_regime_model(
    df_regime: pd.DataFrame,
    regime_name: str,
    feature_columns: list
) -> dict:
    """
    Train XGBoost model for specific regime

    Returns:
        dict with model, metrics, and metadata
    """
    print(f"\n{'=' * 80}")
    print(f"Training XGBoost for {regime_name} Market")
    print(f"{'=' * 80}")

    # Create labels
    df_regime['label'] = create_labels(df_regime, LOOKAHEAD, THRESHOLD)

    # Prepare data
    X = df_regime[feature_columns].values
    y = df_regime['label'].values

    # Remove any NaN
    mask = ~np.isnan(X).any(axis=1)
    X = X[mask]
    y = y[mask]

    print(f"\nDataset:")
    print(f"  Samples: {len(X)}")
    print(f"  Features: {len(feature_columns)}")
    print(f"  Positive samples: {y.sum()} ({(y.sum()/len(y)*100):.1f}%)")
    print(f"  Negative samples: {len(y)-y.sum()} ({((len(y)-y.sum())/len(y)*100):.1f}%)")

    # Cross-validation
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)

    fold_results = []

    for fold_idx, (train_idx, val_idx) in enumerate(cv.split(X, y), 1):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Calculate scale_pos_weight
        neg_count = (y_train == 0).sum()
        pos_count = (y_train == 1).sum()
        scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0

        # Train model
        model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=5,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'
        )

        model.fit(X_train, y_train, verbose=False)

        # Evaluate
        y_pred = model.predict(X_val)

        acc = accuracy_score(y_val, y_pred)
        prec = precision_score(y_val, y_pred, zero_division=0)
        rec = recall_score(y_val, y_pred, zero_division=0)
        f1 = f1_score(y_val, y_pred, zero_division=0)

        fold_results.append({
            'accuracy': acc,
            'precision': prec,
            'recall': rec,
            'f1': f1
        })

        print(f"\nFold {fold_idx}:")
        print(f"  Accuracy: {acc:.3f}")
        print(f"  Precision: {prec:.3f}")
        print(f"  Recall: {rec:.3f}")
        print(f"  F1 Score: {f1:.3f}")

    # Average metrics
    avg_metrics = {
        'accuracy': np.mean([r['accuracy'] for r in fold_results]),
        'precision': np.mean([r['precision'] for r in fold_results]),
        'recall': np.mean([r['recall'] for r in fold_results]),
        'f1': np.mean([r['f1'] for r in fold_results])
    }

    print(f"\n{'=' * 80}")
    print(f"Cross-Validation Results (Average)")
    print(f"{'=' * 80}")
    print(f"Accuracy: {avg_metrics['accuracy']:.3f}")
    print(f"Precision: {avg_metrics['precision']:.3f}")
    print(f"Recall: {avg_metrics['recall']:.3f}")
    print(f"F1 Score: {avg_metrics['f1']:.3f}")

    # Train final model on all data
    print(f"\n{'=' * 80}")
    print(f"Training Final Model on All {regime_name} Data")
    print(f"{'=' * 80}")

    neg_count = (y == 0).sum()
    pos_count = (y == 1).sum()
    scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0

    print(f"Class weight (scale_pos_weight): {scale_pos_weight:.2f}")

    final_model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )

    final_model.fit(X, y, verbose=False)

    # Feature importance
    importance_df = pd.DataFrame({
        'feature': feature_columns,
        'importance': final_model.feature_importances_
    }).sort_values('importance', ascending=False)

    print(f"\nTop 10 Most Important Features for {regime_name}:")
    print(importance_df.head(10).to_string(index=False))

    return {
        'model': final_model,
        'metrics': avg_metrics,
        'feature_importance': importance_df,
        'regime': regime_name,
        'n_samples': len(X),
        'positive_rate': (y.sum() / len(y))
    }


print("=" * 80)
print("Training Regime-Specific XGBoost Models")
print("=" * 80)

# Load data
data_file = DATA_DIR / "BTCUSDT_5m_max.csv"
df = pd.read_csv(data_file)
print(f"✅ Data loaded: {len(df)} candles")

# Calculate features
print("\nCalculating features...")
df = calculate_features(df)
adv_features = AdvancedTechnicalFeatures(lookback_sr=50, lookback_trend=20)
df = adv_features.calculate_all_features(df)
df = df.ffill().dropna()
print(f"✅ Features calculated: {len(df)} rows")

# Detect regimes
print("\nDetecting market regimes...")
detector = RegimeDetector(lookback_window=240)
df = detector.classify_dataset_regimes(df)

stats = detector.get_regime_statistics(df)
print(f"  Bull: {stats['bull_count']:,} ({stats['bull_pct']:.1f}%)")
print(f"  Bear: {stats['bear_count']:,} ({stats['bear_pct']:.1f}%)")
print(f"  Sideways: {stats['sideways_count']:,} ({stats['sideways_pct']:.1f}%)")

# Get advanced feature columns
advanced_features = adv_features.get_feature_names()
feature_columns = BASELINE_FEATURES + advanced_features

print(f"\nTotal features: {len(feature_columns)}")
print(f"  Baseline: {len(BASELINE_FEATURES)}")
print(f"  Advanced: {len(advanced_features)}")

# Train models for each regime
results = {}

for regime in ['Bull', 'Bear', 'Sideways']:
    df_regime = df[df['regime'] == regime].copy()

    if len(df_regime) < 100:
        print(f"\n⚠️ Skipping {regime} - insufficient data ({len(df_regime)} samples)")
        continue

    result = train_regime_model(df_regime, regime, feature_columns)
    results[regime] = result

    # Save model
    model_file = MODELS_DIR / f"xgboost_regime_{regime.lower()}.pkl"
    with open(model_file, 'wb') as f:
        pickle.dump(result['model'], f)
    print(f"✅ Model saved: {model_file}")

    # Save feature columns
    feature_file = MODELS_DIR / f"xgboost_regime_{regime.lower()}_features.txt"
    with open(feature_file, 'w') as f:
        for feat in feature_columns:
            f.write(f"{feat}\n")

    # Save metadata
    metadata = {
        'regime': regime,
        'n_samples': result['n_samples'],
        'positive_rate': float(result['positive_rate']),
        'metrics': {k: float(v) for k, v in result['metrics'].items()},
        'lookahead': LOOKAHEAD,
        'threshold': THRESHOLD,
        'n_features': len(feature_columns)
    }

    metadata_file = MODELS_DIR / f"xgboost_regime_{regime.lower()}_metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

# Save comparison results
print(f"\n{'=' * 80}")
print("Regime Model Comparison")
print(f"{'=' * 80}")

comparison_data = []
for regime in ['Bull', 'Bear', 'Sideways']:
    if regime in results:
        r = results[regime]
        comparison_data.append({
            'regime': regime,
            'n_samples': r['n_samples'],
            'positive_rate': r['positive_rate'] * 100,
            'accuracy': r['metrics']['accuracy'],
            'precision': r['metrics']['precision'],
            'recall': r['metrics']['recall'],
            'f1': r['metrics']['f1']
        })

comparison_df = pd.DataFrame(comparison_data)
print(comparison_df.to_string(index=False))

comparison_file = RESULTS_DIR / "regime_models_comparison.csv"
comparison_df.to_csv(comparison_file, index=False)
print(f"\n✅ Comparison saved: {comparison_file}")

print(f"\n{'=' * 80}")
print("✅ All Regime Models Trained!")
print(f"{'=' * 80}")

print(f"\n비판적 분석:")
print(f"  각 regime별로 서로 다른 feature importance를 보일 것으로 예상")
print(f"  - Bull: Momentum indicators 중요")
print(f"  - Bear: Support/resistance 중요")
print(f"  - Sideways: Volatility breakout 중요")
print(f"\n  Next: Ensemble backtest로 general model과 비교!")
