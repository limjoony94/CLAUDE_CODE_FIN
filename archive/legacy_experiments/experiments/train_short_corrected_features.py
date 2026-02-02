"""
SHORT Model - CORRECTED Version

CRITICAL FIXES:
1. Use ACTUAL feature names (not wrong names!)
2. Use ALL 66 available features
3. Use realistic TP (0.5-1.0%) instead of 3%
4. TP/SL-aligned labeling

Expected: >50% win rate
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

# Directories
DATA_DIR = PROJECT_ROOT / "data" / "historical"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# Realistic TP/SL for SHORT (OPTIMIZED V2)
TP_PCT = 0.005  # 0.5% take profit (very realistic!)
SL_PCT = 0.005  # 0.5% stop loss (symmetric)
MAX_HOLD_CANDLES = 12  # 1 hour (fast scalping)


def create_short_labels_tp_sl(df, tp_pct=TP_PCT, sl_pct=SL_PCT, max_hold_candles=MAX_HOLD_CANDLES):
    """
    Create TP/SL-aligned labels for SHORT

    Label 1: Trade hits TP before SL within max_hold
    Label 0: Otherwise
    """
    print("\n" + "="*80)
    print("Creating TP/SL-Aligned Labels (REALISTIC SHORT)")
    print("="*80)
    print(f"Take Profit: {tp_pct*100:.1f}% (price drops)")
    print(f"Stop Loss: {sl_pct*100:.1f}% (price rises)")
    print(f"Max Hold: {max_hold_candles} candles ({max_hold_candles*5/60:.1f} hours)")

    labels = []

    for i in range(len(df)):
        if i >= len(df) - max_hold_candles:
            labels.append(0)
            continue

        entry_price = df['close'].iloc[i]
        tp_price = entry_price * (1 - tp_pct)
        sl_price = entry_price * (1 + sl_pct)

        tp_hit = False
        sl_hit = False

        for j in range(1, max_hold_candles + 1):
            if i + j >= len(df):
                break

            low = df['low'].iloc[i + j]
            high = df['high'].iloc[i + j]

            if low <= tp_price:
                tp_hit = True
                break

            if high >= sl_price:
                sl_hit = True
                break

        labels.append(1 if tp_hit and not sl_hit else 0)

    labels = np.array(labels)

    # Analysis
    positive_count = np.sum(labels == 1)
    positive_rate = positive_count / len(labels) * 100

    print(f"\nLabel Distribution:")
    print(f"  TP WIN (1): {positive_count:,} ({positive_rate:.2f}%)")
    print(f"  SL/MAX_HOLD (0): {len(labels) - positive_count:,} ({100-positive_rate:.2f}%)")

    if positive_rate < 2:
        print(f"\n⚠️ WARNING: Positive rate {positive_rate:.2f}% is low")
        print(f"   Model may struggle to learn with few positive examples")
    elif positive_rate > 10:
        print(f"\n✅ GOOD: Positive rate {positive_rate:.2f}% is healthy")
        print(f"   Sufficient positive examples for training")

    return labels


def load_and_prepare_data():
    """Load data and calculate ALL available features"""
    print("\n" + "="*80)
    print("Loading Data and Calculating ALL Features")
    print("="*80)

    # Load
    data_file = DATA_DIR / "BTCUSDT_5m_max.csv"
    df = pd.read_csv(data_file)
    print(f"Loaded {len(df):,} candles")

    # Calculate baseline features (33 features)
    print("Calculating baseline features...")
    df = calculate_features(df)

    # Calculate advanced features (33 features)
    print("Calculating advanced features...")
    adv_features = AdvancedTechnicalFeatures(lookback_sr=50, lookback_trend=20)
    df = adv_features.calculate_all_features(df)

    # Clean
    df = df.ffill().dropna()
    print(f"After cleaning: {len(df):,} candles")

    # Get ALL feature columns (excluding OHLCV and metadata)
    exclude_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume', 'date',
                    'nearest_support', 'nearest_resistance',  # These are used for calcs, not features
                    'body_size', 'upper_shadow', 'lower_shadow', 'total_range'  # Duplicated
                    ]

    all_columns = df.columns.tolist()
    feature_columns = [c for c in all_columns if c not in exclude_cols]

    print(f"\n✅ Total Features: {len(feature_columns)}")
    print(f"\nFeature Breakdown:")
    print(f"  Baseline (calculate_features): ~33")
    print(f"  Advanced (AdvancedTechnicalFeatures): ~33")

    # Show first 20 features
    print(f"\nFirst 20 Features:")
    for i, f in enumerate(feature_columns[:20], 1):
        print(f"  {i}. {f}")

    return df, feature_columns


def train_short_model(df, feature_columns):
    """Train SHORT model with corrected features"""
    print("\n" + "="*80)
    print("Training SHORT Model (CORRECTED)")
    print("="*80)

    # Create labels
    labels = create_short_labels_tp_sl(df)

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

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_scaled)):
        print(f"\nFold {fold + 1}/5")

        X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        pos_count = np.sum(y_train == 1)
        neg_count = np.sum(y_train == 0)
        scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1

        print(f"  Train: {len(train_idx):,} | Positive: {pos_count:,} ({pos_count/len(y_train)*100:.2f}%)")
        print(f"  Val: {len(val_idx):,}")

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
            random_state=42,
            eval_metric='logloss'
        )

        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

        # Evaluate at different thresholds
        y_pred_proba = model.predict_proba(X_val)[:, 1]

        for threshold in [0.5, 0.6, 0.7]:
            y_pred = (y_pred_proba >= threshold).astype(int)

            tp = np.sum((y_pred == 1) & (y_val == 1))
            fp = np.sum((y_pred == 1) & (y_val == 0))
            tn = np.sum((y_pred == 0) & (y_val == 0))
            fn = np.sum((y_pred == 0) & (y_val == 1))

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0

            print(f"  Threshold {threshold}: Win Rate={precision*100:.1f}%, Recall={recall*100:.1f}%")

    # Train final model
    print("\n" + "="*80)
    print("Training Final Model (Full Dataset)")
    print("="*80)

    pos_count = np.sum(y == 1)
    neg_count = np.sum(y == 0)
    scale_pos_weight = neg_count / pos_count

    final_model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0.1,
        min_child_weight=5,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric='logloss'
    )

    final_model.fit(X_scaled, y, verbose=False)

    print(f"✅ Final model trained on {len(X_scaled):,} samples")

    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': final_model.feature_importances_
    }).sort_values('importance', ascending=False)

    print(f"\nTop 15 Most Important Features:")
    for idx, row in feature_importance.head(15).iterrows():
        print(f"  {row['feature']:40s}: {row['importance']:.4f}")

    return final_model, scaler


def save_model(model, scaler, feature_columns):
    """Save model with corrected metadata"""
    print("\n" + "="*80)
    print("Saving Model")
    print("="*80)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Model
    model_path = MODELS_DIR / f"xgboost_short_corrected_{timestamp}.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"✅ Model: {model_path.name}")

    # Scaler
    scaler_path = MODELS_DIR / f"xgboost_short_corrected_{timestamp}_scaler.pkl"
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"✅ Scaler: {scaler_path.name}")

    # Features
    features_path = MODELS_DIR / f"xgboost_short_corrected_{timestamp}_features.txt"
    with open(features_path, 'w') as f:
        for feature in feature_columns:
            f.write(f"{feature}\n")
    print(f"✅ Features: {features_path.name}")

    # Metadata
    metadata_path = MODELS_DIR / f"xgboost_short_corrected_{timestamp}_metadata.txt"
    with open(metadata_path, 'w') as f:
        f.write("XGBoost SHORT Model - CORRECTED VERSION\n")
        f.write("="*80 + "\n\n")
        f.write("CRITICAL FIXES:\n")
        f.write("  1. Used ACTUAL feature names (not wrong names!)\n")
        f.write(f"  2. Used ALL {len(feature_columns)} available features\n")
        f.write(f"  3. Used realistic TP ({TP_PCT*100:.1f}%) instead of 3%\n")
        f.write("  4. TP/SL-aligned labeling\n\n")
        f.write("CONFIGURATION:\n")
        f.write(f"  TP: {TP_PCT*100:.1f}% (price drops)\n")
        f.write(f"  SL: {SL_PCT*100:.1f}% (price rises)\n")
        f.write(f"  Max Hold: {MAX_HOLD_CANDLES} candles ({MAX_HOLD_CANDLES*5/60:.1f} hours)\n")
        f.write(f"  Features: {len(feature_columns)}\n")
        f.write(f"  Model: XGBoost\n")
        f.write(f"  Normalization: MinMaxScaler [-1, 1]\n")

    print(f"✅ Metadata: {metadata_path.name}")

    return model_path


def main():
    print("="*80)
    print("SHORT Model Training - CORRECTED VERSION")
    print("="*80)

    print("\n🎯 Critical Fixes Applied (V2 - 0.5% TP):")
    print("  1. Feature names CORRECTED (bb_high not bb_upper, etc.)")
    print("  2. Using ALL 64 features (not just 4!)")
    print("  3. OPTIMIZED TP 0.5% (market analysis: 34.7% win rate)")
    print("  4. Fast scalping 1h max hold (not 4h)")

    # Load data
    df, feature_columns = load_and_prepare_data()

    # Train
    model, scaler = train_short_model(df, feature_columns)

    # Save
    model_path = save_model(model, scaler, feature_columns)

    print("\n" + "="*80)
    print("✅ TRAINING COMPLETE")
    print("="*80)

    print(f"\n📋 Next Steps:")
    print(f"  1. Test signal quality:")
    print(f"     (Modify analyze_short_entry_model.py to use this model)")

    print(f"\n  2. Backtest:")
    print(f"     (Modify backtest_dual_model_mainnet.py to use this model)")

    print(f"\n  3. Expected Results:")
    print(f"     - Win Rate: >50% (vs current <1%)")
    print(f"     - Signal Rate: 2-5% (vs current 1%)")
    print(f"     - Features: 66 (vs previous 4!)")

    print(f"\n  4. If successful:")
    print(f"     - Deploy to production")
    print(f"     - Monitor first 50 trades")
    print(f"     - Adjust thresholds if needed")


if __name__ == "__main__":
    main()
