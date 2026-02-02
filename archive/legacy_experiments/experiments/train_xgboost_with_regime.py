"""
XGBoost with Unsupervised Market Regime Classification

Approach:
- K-Means clustering on rolling window features
- Identify 4 market regimes automatically
- Add regime as feature to XGBoost (37 → 38 features)
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from datetime import datetime
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Import baseline
import sys
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.production.train_xgboost_improved_v3_phase2 import calculate_features
from scripts.production.advanced_technical_features import AdvancedTechnicalFeatures
from scripts.experiments.train_xgboost_realistic_labels import create_realistic_labels, get_feature_columns

# Directories
DATA_DIR = PROJECT_ROOT / "data" / "historical"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

MODELS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_and_prepare_data():
    """Load data and calculate features"""
    print("="*80)
    print("XGBoost with Market Regime Classification (Unsupervised)")
    print("="*80)
    print("\nLoading data...")

    data_file = DATA_DIR / "BTCUSDT_5m_max.csv"
    df = pd.read_csv(data_file)
    print(f"Raw data: {len(df)} candles")

    # Calculate features
    print("\nCalculating features...")
    df = calculate_features(df)
    adv_features = AdvancedTechnicalFeatures(lookback_sr=50, lookback_trend=20)
    df = adv_features.calculate_all_features(df)

    # Handle NaN
    rows_before = len(df)
    df = df.ffill()
    df = df.dropna()
    rows_after = len(df)
    print(f"After NaN handling: {rows_before} → {rows_after} rows")

    return df, adv_features


def classify_market_regime(df, lookback=20, n_clusters=4):
    """
    K-Means clustering for market regime classification

    Features for clustering:
    - Returns (20-candle window)
    - Volatility (ATR)
    - Volume change
    - Trend strength (ADX)

    Args:
        df: DataFrame with features
        lookback: Rolling window size
        n_clusters: Number of regimes

    Returns:
        regime_labels: Cluster assignments
        kmeans_model: Trained K-Means model
        regime_stats: Statistics for each regime
    """
    print(f"\nClassifying Market Regimes (Unsupervised K-Means):")
    print(f"  Lookback window: {lookback} candles")
    print(f"  Number of regimes: {n_clusters}")

    # Calculate regime features
    regime_features = pd.DataFrame()

    # Returns (rolling mean and std)
    returns = df['close'].pct_change()
    regime_features['returns_mean'] = returns.rolling(lookback).mean()
    regime_features['returns_std'] = returns.rolling(lookback).std()

    # Volatility
    if 'atr_pct' in df.columns:
        regime_features['volatility'] = df['atr_pct']
    else:
        regime_features['volatility'] = returns.rolling(lookback).std()

    # Volume
    if 'volume_change' in df.columns:
        regime_features['volume'] = df['volume_change']
    else:
        regime_features['volume'] = df['volume'].pct_change()

    # Trend strength
    if 'adx' in df.columns:
        regime_features['trend_strength'] = df['adx']
    else:
        regime_features['trend_strength'] = abs(regime_features['returns_mean'])

    # Handle NaN from rolling
    regime_features = regime_features.ffill().fillna(0)

    # Standardize features for K-Means
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(regime_features)

    # K-Means clustering
    print("\nRunning K-Means clustering...")
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    regime_labels = kmeans.fit_predict(features_scaled)

    # Analyze regimes
    regime_stats = {}
    for regime_id in range(n_clusters):
        mask = regime_labels == regime_id
        regime_data = regime_features[mask]

        stats = {
            'count': int(mask.sum()),
            'pct': float(mask.mean() * 100),
            'returns_mean': float(regime_data['returns_mean'].mean() * 100),
            'returns_std': float(regime_data['returns_std'].mean() * 100),
            'volatility': float(regime_data['volatility'].mean() * 100),
            'volume': float(regime_data['volume'].mean()),
            'trend_strength': float(regime_data['trend_strength'].mean())
        }

        # Interpret regime
        if stats['volatility'] > regime_features['volatility'].median():
            vol_label = "High Vol"
        else:
            vol_label = "Low Vol"

        if stats['returns_mean'] > 0.05:
            trend_label = "Bull"
        elif stats['returns_mean'] < -0.05:
            trend_label = "Bear"
        else:
            trend_label = "Sideways"

        stats['interpretation'] = f"{vol_label} + {trend_label}"
        regime_stats[f"Regime {regime_id}"] = stats

    print("\nRegime Statistics:")
    for regime_name, stats in regime_stats.items():
        print(f"\n{regime_name}: {stats['interpretation']}")
        print(f"  Samples: {stats['count']} ({stats['pct']:.1f}%)")
        print(f"  Avg Returns: {stats['returns_mean']:.3f}%")
        print(f"  Volatility: {stats['volatility']:.3f}%")
        print(f"  Trend Strength: {stats['trend_strength']:.2f}")

    return regime_labels, kmeans, scaler, regime_stats


def train_xgboost_with_regime(df, feature_columns, labels, regime_labels, regime_stats):
    """Train XGBoost with regime feature"""

    print("\n" + "="*80)
    print("Training XGBoost with Market Regime Feature")
    print("="*80)

    # Add regime as feature
    df_with_regime = df[feature_columns].copy()
    df_with_regime['market_regime'] = regime_labels

    # Get actual feature columns used
    feature_columns_extended = list(df_with_regime.columns)

    X = df_with_regime.values
    y = labels

    print(f"\nDataset:")
    print(f"  Samples: {len(X)}")
    print(f"  Features: {len(feature_columns_extended)} (37 + 1 regime)")
    print(f"  Positive samples: {np.sum(y)} ({np.mean(y)*100:.1f}%)")

    # Time series split
    tscv = TimeSeriesSplit(n_splits=5)

    best_model = None
    best_score = 0
    fold_scores = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        scale_pos_weight = (len(y_train) - np.sum(y_train)) / np.sum(y_train)

        model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=42,
            eval_metric='logloss'
        )

        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

        y_pred = model.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred, zero_division=0)
        recall = recall_score(y_val, y_pred, zero_division=0)
        f1 = f1_score(y_val, y_pred, zero_division=0)

        fold_scores.append({'fold': fold, 'accuracy': accuracy, 'precision': precision,
                          'recall': recall, 'f1': f1})

        print(f"\nFold {fold}: Acc={accuracy:.3f}, P={precision:.3f}, R={recall:.3f}, F1={f1:.3f}")

        if f1 > best_score:
            best_score = f1
            best_model = model

    avg_scores = pd.DataFrame(fold_scores).mean()

    print("\n" + "="*80)
    print("Cross-Validation Results (Average)")
    print("="*80)
    print(f"Accuracy: {avg_scores['accuracy']:.3f}")
    print(f"Precision: {avg_scores['precision']:.3f}")
    print(f"Recall: {avg_scores['recall']:.3f}")
    print(f"F1 Score: {avg_scores['f1']:.3f}")

    print("\nComparison to Baseline:")
    baseline_f1 = 0.089
    realistic_f1 = 0.513
    improvement_vs_baseline = ((avg_scores['f1'] - baseline_f1) / baseline_f1) * 100
    improvement_vs_realistic = ((avg_scores['f1'] - realistic_f1) / realistic_f1) * 100
    print(f"  Baseline F1: {baseline_f1:.3f}")
    print(f"  Realistic Labels F1: {realistic_f1:.3f}")
    print(f"  With Regime F1: {avg_scores['f1']:.3f}")
    print(f"  vs Baseline: {improvement_vs_baseline:+.1f}%")
    print(f"  vs Realistic: {improvement_vs_realistic:+.1f}%")

    # Train final model
    print("\n" + "="*80)
    print("Training Final Model")
    print("="*80)

    scale_pos_weight = (len(y) - np.sum(y)) / np.sum(y)

    final_model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric='logloss'
    )

    final_model.fit(X, y, verbose=False)

    # Feature importance
    # Ensure lengths match
    assert len(feature_columns_extended) == len(final_model.feature_importances_), \
        f"Feature mismatch: {len(feature_columns_extended)} vs {len(final_model.feature_importances_)}"

    feature_importance = pd.DataFrame({
        'feature': feature_columns_extended,
        'importance': final_model.feature_importances_
    }).sort_values('importance', ascending=False)

    print("\nTop 20 Most Important Features:")
    print(feature_importance.head(20).to_string(index=False))

    # Check regime importance
    regime_importance = feature_importance[feature_importance['feature'] == 'market_regime']['importance'].values
    regime_rank = (feature_importance['feature'] == 'market_regime').idxmax() + 1

    print(f"\nMarket Regime Feature:")
    print(f"  Importance: {regime_importance[0] if len(regime_importance) > 0 else 0:.4f}")
    print(f"  Rank: #{regime_rank} out of {len(feature_columns_extended)}")

    return final_model, feature_importance, avg_scores, feature_columns_extended


def save_model(model, feature_columns, kmeans, scaler, regime_stats, scores):
    """Save model and metadata"""

    model_name = "xgboost_v4_with_regime"
    model_path = MODELS_DIR / f"{model_name}.pkl"
    feature_path = MODELS_DIR / f"{model_name}_features.txt"
    kmeans_path = MODELS_DIR / f"{model_name}_kmeans.pkl"
    scaler_path = MODELS_DIR / f"{model_name}_scaler.pkl"

    # Save model
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    # Save K-Means and scaler
    with open(kmeans_path, 'wb') as f:
        pickle.dump(kmeans, f)

    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)

    # Save features
    with open(feature_path, 'w') as f:
        for feature in feature_columns:
            f.write(f"{feature}\n")

    # Metadata
    metadata = {
        "model_name": model_name,
        "n_features": len(feature_columns),
        "has_regime_feature": True,
        "n_regimes": 4,
        "regime_lookback": 20,
        "timestamp": datetime.now().isoformat(),
        "regime_stats": regime_stats,
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
    print(f"K-Means: {kmeans_path}")
    print(f"Scaler: {scaler_path}")
    print(f"Features: {feature_path}")
    print(f"Metadata: {metadata_path}")

    return model_path


def main():
    """Main training pipeline"""

    # Load data
    df, adv_features = load_and_prepare_data()

    # Get features
    feature_columns = get_feature_columns(df, adv_features)

    # Classify market regimes (Unsupervised!)
    regime_labels, kmeans, scaler, regime_stats = classify_market_regime(
        df, lookback=20, n_clusters=4
    )

    # Create realistic labels
    labels, label_stats = create_realistic_labels(df, max_hold=48, stop_loss=0.01,
                                                 take_profit=0.03, positive_threshold=0.0)

    # Train with regime feature
    model, feature_importance, scores, feature_columns_extended = train_xgboost_with_regime(
        df, feature_columns, labels, regime_labels, regime_stats
    )

    # Save model
    model_path = save_model(model, feature_columns_extended, kmeans, scaler,
                          regime_stats, scores)

    print("\n" + "="*80)
    print("✅ XGBoost with Market Regime Training Complete!")
    print("="*80)
    print(f"\nKey Insight:")
    print(f"  Unsupervised learning identified {len(regime_stats)} market regimes")
    print(f"  Regime feature added to XGBoost (37 → 38 features)")
    print(f"  Next: Backtest to see if regime-awareness improves returns!")


if __name__ == "__main__":
    main()
