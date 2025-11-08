"""
SHORT Entry - Peak/Trough Labeling (BREAKTHROUGH VERSION)

Evolution:
  V1: TP/SL labeling (3% TP, 4h) → 0.6% positive rate → <1% precision
  V2: TP/SL labeling (0.5% TP, 1h) → 10.99% positive rate → 15-30% precision
  V3: Peak/Trough labeling → ??% positive rate → ??% precision

Expected: 35-50% precision (matching LONG Exit model)
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
import xgboost as xgb
import sys

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.production.train_xgboost_improved_v3_phase2 import calculate_features
from scripts.production.advanced_technical_features import AdvancedTechnicalFeatures
from src.features.sell_signal_features import SellSignalFeatures
from src.labeling.peak_trough_labeling import PeakTroughLabeling

DATA_DIR = PROJECT_ROOT / "data" / "historical"
MODELS_DIR = PROJECT_ROOT / "models"


def load_and_prepare_data():
    """Load data with ALL features"""
    print("\n" + "="*80)
    print("Loading Data and Calculating Features")
    print("="*80)

    data_file = DATA_DIR / "BTCUSDT_5m_max.csv"
    df = pd.read_csv(data_file)
    print(f"✅ Loaded {len(df):,} candles")

    # Base + Advanced features
    print("Calculating base features...")
    df = calculate_features(df)

    print("Calculating advanced features...")
    adv = AdvancedTechnicalFeatures()
    df = adv.calculate_all_features(df)

    # SELL-SPECIFIC features
    print("Calculating SELL-SPECIFIC features...")
    sell_features = SellSignalFeatures()
    df = sell_features.calculate_all_features(df)

    df = df.ffill().dropna()
    print(f"✅ After cleaning: {len(df):,} candles")

    # Feature columns
    exclude_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'date',
                    'nearest_support', 'nearest_resistance']
    feature_columns = [c for c in df.columns if c not in exclude_cols]

    print(f"\n✅ Total Features: {len(feature_columns)}")

    return df, feature_columns


def train_short_model(df, feature_columns):
    """Train SHORT Entry with Peak/Trough labeling"""
    print("\n" + "="*80)
    print("Training SHORT Entry Model (Peak/Trough Labeling)")
    print("="*80)

    # Create labels using Peak/Trough method
    labeler = PeakTroughLabeling(
        lookforward=48,  # 4 hours
        peak_window=10,
        near_threshold=0.80,  # 80% of trough
        holding_hours=1
    )

    labels = labeler.create_short_entry_labels(df)

    # Prepare data
    X = df[feature_columns].values
    y = labels

    # Normalize
    scaler = MinMaxScaler(feature_range=(-1, 1))
    X_scaled = scaler.fit_transform(X)

    # Cross-validation
    print("\n" + "="*80)
    print("Cross-Validation (5 Folds)")
    print("="*80)

    tscv = TimeSeriesSplit(n_splits=5)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_scaled)):
        print(f"\nFold {fold + 1}/5")

        X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        pos_count = np.sum(y_train == 1)
        pos_rate = pos_count / len(y_train) * 100
        scale_pos_weight = np.sum(y_train == 0) / pos_count if pos_count > 0 else 1

        print(f"  Train: {len(train_idx):,} | Positive: {pos_count:,} ({pos_rate:.2f}%)")

        # Train
        model = xgb.XGBClassifier(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            gamma=0.1,
            min_child_weight=5,
            scale_pos_weight=scale_pos_weight,
            random_state=42
        )

        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

        # Evaluate at different thresholds
        y_pred_proba = model.predict_proba(X_val)[:, 1]

        for threshold in [0.5, 0.6, 0.7]:
            y_pred = (y_pred_proba >= threshold).astype(int)

            tp = np.sum((y_pred == 1) & (y_val == 1))
            fp = np.sum((y_pred == 1) & (y_val == 0))
            fn = np.sum((y_pred == 0) & (y_val == 1))

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0

            print(f"  Threshold {threshold}: Precision={precision*100:.1f}%, Recall={recall*100:.1f}%")

        fold_results.append({'precision_0.5': precision})

    # Train final model
    print("\n" + "="*80)
    print("Training Final Model")
    print("="*80)

    pos_count = np.sum(y == 1)
    scale_pos_weight = np.sum(y == 0) / pos_count

    final_model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0.1,
        min_child_weight=5,
        scale_pos_weight=scale_pos_weight,
        random_state=42
    )

    final_model.fit(X_scaled, y, verbose=False)
    print(f"✅ Trained on {len(X_scaled):,} samples")

    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': final_model.feature_importances_
    }).sort_values('importance', ascending=False)

    print(f"\n🏆 Top 20 Most Important Features:")
    for idx, row in feature_importance.head(20).iterrows():
        marker = "🎯" if any(row['feature'].startswith(prefix) for prefix in [
            'rsi_weakening', 'rsi_bearish', 'macd_weakening', 'overbought',
            'volume_declining', 'bearish', 'distribution', 'exhaustion', 'rejection'
        ]) else "  "
        print(f"  {marker} {row['feature']:45s}: {row['importance']:.4f}")

    return final_model, scaler, fold_results


def save_model(model, scaler, feature_columns):
    """Save model"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    model_path = MODELS_DIR / f"xgboost_short_peak_trough_{timestamp}.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    scaler_path = MODELS_DIR / f"xgboost_short_peak_trough_{timestamp}_scaler.pkl"
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)

    features_path = MODELS_DIR / f"xgboost_short_peak_trough_{timestamp}_features.txt"
    with open(features_path, 'w') as f:
        for feature in feature_columns:
            f.write(f"{feature}\n")

    metadata_path = MODELS_DIR / f"xgboost_short_peak_trough_{timestamp}_metadata.txt"
    with open(metadata_path, 'w') as f:
        f.write("XGBoost SHORT Entry - Peak/Trough Labeling\n")
        f.write("="*80 + "\n\n")
        f.write("LABELING STRATEGY:\n")
        f.write("  Method: near_trough_80pct_and_beats_holding_1h\n")
        f.write("  Inspired by LONG Exit success (35.2% precision)\n\n")
        f.write("CONCEPT:\n")
        f.write("  - Detect future troughs (low points)\n")
        f.write("  - Enter SHORT when price is within 80% of trough\n")
        f.write("  - Only if entering now > waiting 1 hour\n\n")
        f.write("FEATURES:\n")
        f.write(f"  Total: {len(feature_columns)}\n")
        f.write("  Base + Advanced: ~66\n")
        f.write("  SELL-SPECIFIC: ~42\n\n")
        f.write("PARAMETERS:\n")
        f.write("  Lookforward: 48 candles (4 hours)\n")
        f.write("  Peak window: 10\n")
        f.write("  Near threshold: 80%\n")
        f.write("  Holding comparison: 1 hour\n")

    print(f"\n✅ Model: {model_path.name}")
    print(f"✅ Scaler: {scaler_path.name}")

    return model_path


def main():
    print("="*80)
    print("SHORT Entry - Peak/Trough Labeling (BREAKTHROUGH)")
    print("="*80)

    print("\n🎯 Evolution:")
    print("  V1 (TP/SL 3%, 4h): 0.6% positive rate → <1% precision")
    print("  V2 (TP/SL 0.5%, 1h): 10.99% positive rate → 15-30% precision")
    print("  V3 (Peak/Trough): ??% positive rate → ??% precision")
    print("\n  Expected: 35-50% precision (matching LONG Exit)")

    # Load
    df, feature_columns = load_and_prepare_data()

    # Train
    model, scaler, fold_results = train_short_model(df, feature_columns)

    # Save
    model_path = save_model(model, scaler, feature_columns)

    # Results
    avg_precision = np.mean([r['precision_0.5'] for r in fold_results])

    print("\n" + "="*80)
    print("✅ TRAINING COMPLETE")
    print("="*80)

    print(f"\n📊 Results:")
    print(f"  Average Precision (threshold 0.5): {avg_precision*100:.1f}%")

    # Comparison
    print(f"\n📈 Comparison to Previous Versions:")
    print(f"  V1 (TP/SL 3%): <1% precision")
    print(f"  V2 (TP/SL 0.5%): 15-30% precision")
    print(f"  V3 (Peak/Trough): {avg_precision*100:.1f}% precision")

    if avg_precision >= 0.35:
        improvement = avg_precision / 0.20 - 1  # vs V2 average
        print(f"\n🎉 BREAKTHROUGH! +{improvement*100:.0f}% improvement!")
        print(f"   Peak/Trough labeling works!")
    elif avg_precision >= 0.25:
        print(f"\n✅ GOOD! Significant improvement over TP/SL")
    else:
        print(f"\n⚠️ Similar to V2, may need parameter tuning")

    print("\n📋 Next Steps:")
    print("  1. Backtest this model")
    print("  2. Compare to rule-based SHORT")
    print("  3. Retrain LONG Exit with same method")


if __name__ == "__main__":
    main()
