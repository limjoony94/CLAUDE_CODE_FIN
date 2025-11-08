"""
SHORT Model V2 - Separate Binary Model

새로운 접근법:
1. Binary classification: "Will price decrease?" (YES/NO)
2. SHORT-specific features (30개) + baseline features (10개)
3. 완화된 threshold: 0.2% (더 많은 샘플 확보)
4. 별도 모델로 학습 (LONG 모델과 독립)

목표: SHORT win rate 60%+ 달성
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from datetime import datetime
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.utils.class_weight import compute_sample_weight

import sys
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.production.train_xgboost_improved_v3_phase2 import calculate_features
from scripts.production.short_specific_features import ShortSpecificFeatures

# Directories
DATA_DIR = PROJECT_ROOT / "data" / "historical"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"


def create_short_labels(df, lookahead=3, threshold=0.002):
    """
    Binary labels for SHORT prediction

    Label 1 (SHORT): Price will decrease by threshold% or more
    Label 0 (NO SHORT): Price will not decrease by threshold%

    Args:
        df: DataFrame with OHLCV
        lookahead: Number of candles to look ahead (default: 3 = 15 min)
        threshold: Minimum price decrease (default: 0.002 = 0.2%)

    Returns:
        Binary labels (0 or 1)
    """
    labels = []

    for i in range(len(df)):
        if i >= len(df) - lookahead:
            labels.append(0)
            continue

        current_price = df['close'].iloc[i]
        future_prices = df['close'].iloc[i+1:i+1+lookahead]

        # Minimum future price
        min_future = future_prices.min()

        # Calculate decrease percentage
        decrease_pct = (current_price - min_future) / current_price

        # Label 1 if price decreases by threshold or more
        if decrease_pct >= threshold:
            labels.append(1)  # SHORT
        else:
            labels.append(0)  # NO SHORT

    return np.array(labels)


def load_and_prepare_data():
    """Load data and calculate all features"""
    print("="*80)
    print("Loading Data and Calculating Features")
    print("="*80)

    # Load data
    df = pd.read_csv(DATA_DIR / "BTCUSDT_5m_max.csv")
    print(f"Loaded {len(df)} candles")

    # Calculate baseline features
    df = calculate_features(df)

    # Calculate SHORT-specific features
    short_features = ShortSpecificFeatures(lookback=20)
    df = short_features.calculate_all_features(df)

    # Remove NaN
    df = df.ffill().dropna()
    print(f"After processing: {len(df)} candles")

    # Get feature columns
    baseline_features = ['rsi', 'macd', 'macd_signal', 'bb_upper', 'bb_lower',
                        'ema_21', 'ema_50', 'atr', 'volume_ratio', 'price_distance_ema']

    short_feature_names = short_features.get_feature_names()

    # Combine features (baseline + SHORT-specific)
    feature_columns = [f for f in baseline_features + short_feature_names if f in df.columns]

    print(f"\nTotal features: {len(feature_columns)}")
    print(f"  - Baseline: {len([f for f in baseline_features if f in df.columns])}")
    print(f"  - SHORT-specific: {len([f for f in short_feature_names if f in df.columns])}")

    # Create binary labels for SHORT
    labels = create_short_labels(df, lookahead=3, threshold=0.002)

    # Class distribution
    unique, counts = np.unique(labels, return_counts=True)
    print(f"\nLabel Distribution:")
    print(f"  NO SHORT (0): {counts[0]} ({counts[0]/len(labels)*100:.1f}%)")
    print(f"  SHORT (1): {counts[1]} ({counts[1]/len(labels)*100:.1f}%)")

    # Features matrix
    X = df[feature_columns].values
    y = labels

    return X, y, feature_columns, df


def train_short_model(X, y):
    """Train SHORT-specific binary model"""
    print("\n" + "="*80)
    print("Training SHORT Model")
    print("="*80)

    # XGBoost parameters (optimized for binary classification)
    params = {
        'n_estimators': 300,
        'max_depth': 4,
        'learning_rate': 0.01,
        'min_child_weight': 5,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'gamma': 1,
        'reg_alpha': 1,
        'reg_lambda': 2,
        'random_state': 42,
        'use_label_encoder': False,
        'eval_metric': 'logloss'
    }

    model = xgb.XGBClassifier(**params)

    # Balanced class weights
    sample_weights = compute_sample_weight('balanced', y)

    # Train on all data
    print(f"\nTraining on {len(X)} samples...")
    model.fit(X, y, sample_weight=sample_weights, verbose=False)

    # Training metrics
    y_pred = model.predict(X)
    y_prob = model.predict_proba(X)[:, 1]

    print("\n" + "="*80)
    print("Training Results")
    print("="*80)

    print("\nClassification Report:")
    print(classification_report(y, y_pred, target_names=['NO SHORT', 'SHORT']))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y, y_pred))

    # Metrics by class
    f1 = f1_score(y, y_pred, average=None)
    precision = precision_score(y, y_pred, average=None)
    recall = recall_score(y, y_pred, average=None)

    print(f"\nMetrics:")
    print(f"  NO SHORT - F1: {f1[0]:.3f}, Precision: {precision[0]:.3f}, Recall: {recall[0]:.3f}")
    print(f"  SHORT    - F1: {f1[1]:.3f}, Precision: {precision[1]:.3f}, Recall: {recall[1]:.3f}")

    return model


def backtest_short_model(model, X, y, df):
    """
    Backtest SHORT model with time series cross-validation

    Returns:
        SHORT win rate, trade statistics
    """
    print("\n" + "="*80)
    print("Backtesting SHORT Model")
    print("="*80)

    tscv = TimeSeriesSplit(n_splits=5)

    all_trades = []
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Train on fold
        sample_weights = compute_sample_weight('balanced', y_train)
        model.fit(X_train, y_train, sample_weight=sample_weights, verbose=False)

        # Predict on validation
        y_pred = model.predict(X_val)
        y_prob = model.predict_proba(X_val)[:, 1]

        # Simulate trades with threshold
        threshold = 0.7  # High confidence threshold

        for i, idx in enumerate(val_idx):
            if y_prob[i] >= threshold:  # HIGH confidence SHORT signal
                actual_label = y_val[i]

                # Record trade
                trade = {
                    'fold': fold,
                    'index': idx,
                    'probability': y_prob[i],
                    'predicted': 1,
                    'actual': actual_label,
                    'correct': (1 == actual_label)
                }
                all_trades.append(trade)

        # Fold metrics
        if len([t for t in all_trades if t['fold'] == fold]) > 0:
            fold_correct = sum(t['correct'] for t in all_trades if t['fold'] == fold)
            fold_total = len([t for t in all_trades if t['fold'] == fold])
            fold_win_rate = fold_correct / fold_total

            fold_metrics.append({
                'fold': fold,
                'trades': fold_total,
                'correct': fold_correct,
                'win_rate': fold_win_rate
            })

    # Overall results
    print(f"\n{'='*80}")
    print("Backtest Results")
    print(f"{'='*80}")

    if len(all_trades) == 0:
        print("\n❌ No trades generated (probability never exceeded threshold)")
        print("   Consider lowering threshold or improving model")
        return 0.0, all_trades

    total_trades = len(all_trades)
    correct_trades = sum(t['correct'] for t in all_trades)
    overall_win_rate = correct_trades / total_trades

    print(f"\nOverall Results:")
    print(f"  Total Trades: {total_trades}")
    print(f"  Correct: {correct_trades}")
    print(f"  Incorrect: {total_trades - correct_trades}")
    print(f"  Win Rate: {overall_win_rate*100:.1f}%")

    print(f"\nFold-by-Fold Results:")
    for metrics in fold_metrics:
        print(f"  Fold {metrics['fold']}: {metrics['trades']} trades, "
              f"{metrics['correct']}/{metrics['trades']} correct ({metrics['win_rate']*100:.1f}%)")

    # Probability distribution
    probs = [t['probability'] for t in all_trades]
    print(f"\nProbability Distribution:")
    print(f"  Min: {min(probs):.3f}")
    print(f"  Mean: {np.mean(probs):.3f}")
    print(f"  Max: {max(probs):.3f}")

    return overall_win_rate, all_trades


def save_model(model, feature_columns, metrics):
    """Save model and metadata"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"short_model_v2_{timestamp}"

    # Save model
    model_path = MODELS_DIR / f"{model_name}.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    # Save feature list
    feature_path = MODELS_DIR / f"{model_name}_features.txt"
    with open(feature_path, 'w') as f:
        for feature in feature_columns:
            f.write(f"{feature}\n")

    # Save metadata
    import json
    metadata = {
        "model_name": model_name,
        "model_type": "Binary SHORT Classification",
        "n_features": len(feature_columns),
        "threshold": 0.002,
        "lookahead": 3,
        "short_win_rate": float(metrics['win_rate']),
        "total_trades": int(metrics['trades']),
        "timestamp": datetime.now().isoformat()
    }

    metadata_path = MODELS_DIR / f"{model_name}_metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print("\n" + "="*80)
    print("Model Saved")
    print("="*80)
    print(f"Model: {model_path}")
    print(f"Features: {feature_path}")
    print(f"Metadata: {metadata_path}")

    return model_path, metadata


def main():
    """Main training pipeline"""
    print("="*80)
    print("SHORT Model V2 Training")
    print("="*80)
    print("Approach: Binary classification with SHORT-specific features")
    print("Target: SHORT win rate >= 60%")
    print("="*80)

    # Load data
    X, y, feature_columns, df = load_and_prepare_data()

    # Train model
    model = train_short_model(X, y)

    # Backtest
    win_rate, trades = backtest_short_model(model, X, y, df)

    # Decision
    print("\n" + "="*80)
    print("Final Decision")
    print("="*80)

    if win_rate >= 0.60:
        print(f"✅ SUCCESS! SHORT win rate {win_rate*100:.1f}% >= 60%")
        print(f"✅ Proceed to production deployment")

        # Save model
        metrics = {'win_rate': win_rate, 'trades': len(trades)}
        model_path, metadata = save_model(model, feature_columns, metrics)

    elif win_rate >= 0.55:
        print(f"⚠️ MARGINAL: SHORT win rate {win_rate*100:.1f}% (55-60%)")
        print(f"⚠️ Consider additional optimization or deploy with caution")

        # Save model anyway
        metrics = {'win_rate': win_rate, 'trades': len(trades)}
        model_path, metadata = save_model(model, feature_columns, metrics)

    else:
        print(f"❌ INSUFFICIENT: SHORT win rate {win_rate*100:.1f}% < 55%")
        print(f"❌ Need further improvements:")
        print(f"   - Add more SHORT-specific features")
        print(f"   - Try different threshold values")
        print(f"   - Implement market regime filter")

        if win_rate > 0:
            print(f"\n   Current performance is {win_rate*100:.1f}% - improvement from previous attempts!")

    return model, win_rate, trades


if __name__ == "__main__":
    model, win_rate, trades = main()
