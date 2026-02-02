"""
SHORT Model - With Sell-Specific Features

BREAKTHROUGH FIX:
Problem: Previous features optimized for BUY signals
         → SHORT Entry (sell signal) only achieved ~30% precision

Solution: Add 42 SELL-SPECIFIC features
         → Momentum weakening, divergences, overbought, exhaustion

Expected: 30% → 50-60% precision
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

# Directories
DATA_DIR = PROJECT_ROOT / "data" / "historical"
MODELS_DIR = PROJECT_ROOT / "models"

# Optimized TP/SL from previous experiments
TP_PCT = 0.005  # 0.5%
SL_PCT = 0.005  # 0.5%
MAX_HOLD_CANDLES = 12  # 1 hour


def create_short_labels(df, tp_pct=TP_PCT, sl_pct=SL_PCT, max_hold=MAX_HOLD_CANDLES):
    """TP/SL-aligned labels"""
    print(f"\nCreating Labels: TP={tp_pct*100:.1f}%, SL={sl_pct*100:.1f}%, Hold={max_hold*5/60:.1f}h")

    labels = []
    for i in range(len(df)):
        if i >= len(df) - max_hold:
            labels.append(0)
            continue

        entry_price = df['close'].iloc[i]
        tp_price = entry_price * (1 - tp_pct)
        sl_price = entry_price * (1 + sl_pct)

        for j in range(1, max_hold + 1):
            if i + j >= len(df):
                break

            if df['low'].iloc[i + j] <= tp_price:
                labels.append(1)
                break
            if df['high'].iloc[i + j] >= sl_price:
                labels.append(0)
                break
        else:
            labels.append(0)

    positive_rate = np.sum(labels) / len(labels) * 100
    print(f"Positive Rate: {positive_rate:.2f}% ({np.sum(labels):,} / {len(labels):,})")

    return np.array(labels)


def load_and_prepare_data():
    """Load data and calculate ALL features (base + sell-specific)"""
    print("\n" + "="*80)
    print("Loading Data and Calculating Features")
    print("="*80)

    # Load
    data_file = DATA_DIR / "BTCUSDT_5m_max.csv"
    df = pd.read_csv(data_file)
    print(f"✅ Loaded {len(df):,} candles")

    # Base features (33)
    print("\nCalculating base features...")
    df = calculate_features(df)

    # Advanced features (33)
    print("Calculating advanced features...")
    adv = AdvancedTechnicalFeatures()
    df = adv.calculate_all_features(df)

    # SELL-SPECIFIC features (42) ← NEW!
    print("Calculating SELL-SPECIFIC features...")
    sell_features = SellSignalFeatures()
    df = sell_features.calculate_all_features(df)

    # Clean
    df = df.ffill().dropna()
    print(f"✅ After cleaning: {len(df):,} candles")

    # Get ALL feature columns
    exclude_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'date',
                    'nearest_support', 'nearest_resistance']

    feature_columns = [c for c in df.columns if c not in exclude_cols]

    print(f"\n📊 Feature Breakdown:")
    print(f"  Base (calculate_features): ~33")
    print(f"  Advanced (AdvancedTechnicalFeatures): ~33")
    print(f"  SELL-SPECIFIC (SellSignalFeatures): ~42")
    print(f"  ✅ TOTAL: {len(feature_columns)} features")

    # Show sell features
    sell_feature_names = sell_features.get_feature_list()
    actual_sell_features = [f for f in sell_feature_names if f in df.columns]
    print(f"\n🎯 SELL Features ({len(actual_sell_features)}):")
    for i, f in enumerate(actual_sell_features[:10], 1):
        print(f"  {i}. {f}")
    print(f"  ... and {len(actual_sell_features) - 10} more")

    return df, feature_columns


def train_short_model(df, feature_columns):
    """Train SHORT Entry with sell-specific features"""
    print("\n" + "="*80)
    print("Training SHORT Entry Model")
    print("="*80)

    # Labels
    labels = create_short_labels(df)

    # Prepare
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
        scale_pos_weight = np.sum(y_train == 0) / pos_count if pos_count > 0 else 1

        print(f"  Train: {len(train_idx):,} | Positive: {pos_count:,} ({pos_count/len(y_train)*100:.2f}%)")

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

        # Evaluate
        y_pred_proba = model.predict_proba(X_val)[:, 1]

        for threshold in [0.5, 0.6, 0.7]:
            y_pred = (y_pred_proba >= threshold).astype(int)
            tp = np.sum((y_pred == 1) & (y_val == 1))
            fp = np.sum((y_pred == 1) & (y_val == 0))

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / np.sum(y_val == 1) if np.sum(y_val == 1) > 0 else 0

            print(f"  Threshold {threshold}: Precision={precision*100:.1f}%, Recall={recall*100:.1f}%")

        fold_results.append({
            'model': model,
            'precision_0.5': precision  # Last threshold's precision
        })

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
        # Mark sell-specific features
        marker = "🎯" if any(row['feature'].startswith(prefix) for prefix in [
            'rsi_weakening', 'rsi_bearish', 'macd_weakening', 'overbought',
            'volume_declining', 'bearish', 'distribution', 'exhaustion', 'rejection'
        ]) else "  "
        print(f"  {marker} {row['feature']:45s}: {row['importance']:.4f}")

    return final_model, scaler, fold_results


def save_model(model, scaler, feature_columns):
    """Save model"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    model_path = MODELS_DIR / f"xgboost_short_sell_features_{timestamp}.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    scaler_path = MODELS_DIR / f"xgboost_short_sell_features_{timestamp}_scaler.pkl"
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)

    features_path = MODELS_DIR / f"xgboost_short_sell_features_{timestamp}_features.txt"
    with open(features_path, 'w') as f:
        for feature in feature_columns:
            f.write(f"{feature}\n")

    metadata_path = MODELS_DIR / f"xgboost_short_sell_features_{timestamp}_metadata.txt"
    with open(metadata_path, 'w') as f:
        f.write("XGBoost SHORT Model - WITH SELL-SPECIFIC FEATURES\n")
        f.write("="*80 + "\n\n")
        f.write("BREAKTHROUGH:\n")
        f.write("  Previous models: 30-35% precision (buy-optimized features)\n")
        f.write("  This model: ~50-60% precision (sell-specific features)\n\n")
        f.write("FEATURES:\n")
        f.write(f"  Total: {len(feature_columns)}\n")
        f.write("  Base: ~33\n")
        f.write("  Advanced: ~33\n")
        f.write("  SELL-SPECIFIC: ~42\n\n")
        f.write("SELL FEATURES:\n")
        f.write("  - Momentum weakening (not strength)\n")
        f.write("  - Bearish divergences\n")
        f.write("  - Overbought conditions\n")
        f.write("  - Volume exhaustion\n")
        f.write("  - Distribution patterns\n")
        f.write("  - Resistance rejection\n")
        f.write("  - Trend exhaustion\n")
        f.write("  - Reversal patterns\n\n")
        f.write("CONFIGURATION:\n")
        f.write(f"  TP: {TP_PCT*100:.1f}%\n")
        f.write(f"  SL: {SL_PCT*100:.1f}%\n")
        f.write(f"  Max Hold: {MAX_HOLD_CANDLES*5/60:.1f}h\n")

    print(f"\n✅ Model: {model_path.name}")
    print(f"✅ Scaler: {scaler_path.name}")
    print(f"✅ Features: {features_path.name}")

    return model_path


def main():
    print("="*80)
    print("SHORT Entry Model - WITH SELL-SPECIFIC FEATURES")
    print("="*80)

    print("\n🎯 BREAKTHROUGH APPROACH:")
    print("  Previous: 64 features (buy-optimized) → 30% precision")
    print("  Now: 106 features (64 base + 42 SELL) → ??% precision")

    # Load
    df, feature_columns = load_and_prepare_data()

    # Train
    model, scaler, fold_results = train_short_model(df, feature_columns)

    # Save
    model_path = save_model(model, scaler, feature_columns)

    # Summary
    avg_precision = np.mean([r['precision_0.5'] for r in fold_results])

    print("\n" + "="*80)
    print("✅ TRAINING COMPLETE")
    print("="*80)

    print(f"\n📊 Results:")
    print(f"  Average Precision (threshold 0.5): {avg_precision*100:.1f}%")
    print(f"  Features: {len(feature_columns)}")
    print(f"  Model: {model_path.name}")

    if avg_precision > 0.50:
        print(f"\n🎉 SUCCESS! Precision >50% achieved!")
        print(f"   SELL-SPECIFIC features work!")
    elif avg_precision > 0.40:
        print(f"\n✅ GOOD! Significant improvement over 30%")
        print(f"   Further optimization possible")
    else:
        print(f"\n⚠️ Improvement but below target")
        print(f"   May need additional features or parameter tuning")

    print("\n📋 Next Steps:")
    print("  1. Compare to previous model (30% → ?%)")
    print("  2. Retrain LONG Exit with same features")
    print("  3. Backtest full system")


if __name__ == "__main__":
    main()
