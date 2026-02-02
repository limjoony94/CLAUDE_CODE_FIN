"""
Retrain LONG/SHORT Entry Models with Improved Labeling
=======================================================

Replaces old peak/trough labeling with 2-of-3 scoring system

Expected Impact:
- Current LONG Precision: 13.7%
- Expected LONG Precision: >20%
- Fewer false positives → Better win rate
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from datetime import datetime
import joblib
import json

from src.labeling.improved_entry_labeling import ImprovedEntryLabeling

# Import feature calculator
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from experiments.calculate_all_features import calculate_all_features

def load_and_prepare_data():
    """Load BTC data and calculate all features"""
    print("\n" + "="*80)
    print("Loading BTC Data")
    print("="*80)

    # Load raw data
    data_path = "data/historical/BTCUSDT_5m_max.csv"
    if not os.path.exists(data_path):
        data_path = "C:/Users/J/OneDrive/CLAUDE_CODE_FIN/bingx_rl_trading_bot/data/historical/BTCUSDT_5m_max.csv"

    df = pd.read_csv(data_path)
    print(f"Loaded {len(df):,} raw candles")

    # Calculate ALL features (LONG + SHORT)
    print("\nCalculating all features...")
    df = calculate_all_features(df)
    print(f"Features calculated: {len(df.columns)} columns")

    return df

def retrain_long_entry_model(df):
    """Retrain LONG Entry model with improved labeling"""
    print("\n" + "="*80)
    print("RETRAINING LONG ENTRY MODEL")
    print("="*80)

    # ========================================================================
    # 1. Create Improved Labels
    # ========================================================================
    labeler = ImprovedEntryLabeling(
        profit_threshold=0.004,  # 0.4%
        lookforward_min=6,  # 30min
        lookforward_max=48,  # 4h
        lead_time_min=6,
        lead_time_max=24,
        relative_tolerance=0.002,  # 0.2%
        scoring_threshold=2  # 2 of 3
    )

    print("\nCreating improved LONG Entry labels...")
    labels = labeler.create_long_entry_labels(df)

    # ========================================================================
    # 2. Prepare Features (Use 44 features from current model)
    # ========================================================================
    long_features = [
        'close_change_1', 'close_change_3', 'volume_ma_ratio', 'rsi', 'macd',
        'macd_signal', 'macd_diff', 'bb_high', 'bb_mid', 'bb_low',
        'distance_to_support_pct', 'distance_to_resistance_pct',
        'num_support_touches', 'num_resistance_touches',
        'upper_trendline_slope', 'lower_trendline_slope',
        'price_vs_upper_trendline_pct', 'price_vs_lower_trendline_pct',
        'rsi_bullish_divergence', 'rsi_bearish_divergence',
        'macd_bullish_divergence', 'macd_bearish_divergence',
        'double_top', 'double_bottom', 'higher_highs_lows', 'lower_highs_lows',
        'volume_price_correlation', 'price_volume_trend',
        'body_to_range_ratio', 'upper_shadow_ratio', 'lower_shadow_ratio',
        'bullish_engulfing', 'bearish_engulfing', 'hammer', 'shooting_star', 'doji',
        'distance_from_recent_high_pct', 'bearish_candle_count',
        'red_candle_volume_ratio', 'strong_selling_pressure',
        'price_momentum_near_resistance', 'rsi_from_recent_peak',
        'consecutive_up_candles'
    ]

    print(f"\nUsing {len(long_features)} features")

    # Check available features
    available_features = [f for f in long_features if f in df.columns]
    missing_features = [f for f in long_features if f not in df.columns]

    if missing_features:
        print(f"\n⚠️ Missing {len(missing_features)} features:")
        for f in missing_features[:10]:
            print(f"   - {f}")
        print(f"\nUsing {len(available_features)} available features")
        long_features = available_features

    X = df[long_features].values
    y = labels

    print(f"\nDataset shape:")
    print(f"  X: {X.shape}")
    print(f"  y: {y.shape}")
    print(f"  Positive rate: {np.sum(y)/len(y)*100:.2f}%")

    # ========================================================================
    # 3. Train/Test Split
    # ========================================================================
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False, random_state=42
    )

    print(f"\nTrain/Test Split:")
    print(f"  Train: {len(X_train):,} samples")
    print(f"  Test:  {len(X_test):,} samples")

    # ========================================================================
    # 4. Feature Scaling
    # ========================================================================
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # ========================================================================
    # 5. Train XGBoost Model
    # ========================================================================
    print("\nTraining XGBoost model...")

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=len(y_train[y_train == 0]) / len(y_train[y_train == 1]),
        random_state=42,
        eval_metric='logloss'
    )

    model.fit(
        X_train_scaled, y_train,
        eval_set=[(X_test_scaled, y_test)],
        verbose=False
    )

    # ========================================================================
    # 6. Evaluate Model
    # ========================================================================
    from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print("\n" + "="*80)
    print("LONG ENTRY MODEL PERFORMANCE")
    print("="*80)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f} (Current: 0.137)")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")

    if precision > 0.137:
        improvement = (precision - 0.137) / 0.137 * 100
        print(f"\n✅ Precision improved by {improvement:+.1f}%!")
    else:
        print(f"\n⚠️ Precision did not improve")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    # ========================================================================
    # 7. Save Model
    # ========================================================================
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"xgboost_long_improved_labeling_{timestamp}"

    model_path = f"models/{model_name}.pkl"
    scaler_path = f"models/{model_name}_scaler.pkl"
    features_path = f"models/{model_name}_features.txt"
    metadata_path = f"models/{model_name}_metadata.json"

    # Save model
    joblib.dump(model, model_path)
    print(f"\n✅ Model saved: {model_path}")

    # Save scaler
    joblib.dump(scaler, scaler_path)
    print(f"✅ Scaler saved: {scaler_path}")

    # Save features
    with open(features_path, 'w') as f:
        for feature in long_features:
            f.write(f"{feature}\n")
    print(f"✅ Features saved: {features_path}")

    # Save metadata
    metadata = {
        'model_name': model_name,
        'training_date': timestamp,
        'labeling_method': 'improved_entry_labeling_2of3',
        'num_features': len(long_features),
        'num_train_samples': len(X_train),
        'num_test_samples': len(X_test),
        'positive_rate_train': float(np.sum(y_train) / len(y_train)),
        'positive_rate_test': float(np.sum(y_test) / len(y_test)),
        'scores': {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1)
        },
        'labeling_params': {
            'profit_threshold': 0.004,
            'lookforward_min': 6,
            'lookforward_max': 48,
            'lead_time_min': 6,
            'lead_time_max': 24,
            'relative_tolerance': 0.002,
            'scoring_threshold': 2
        }
    }

    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✅ Metadata saved: {metadata_path}")

    return model_name, precision

def retrain_short_entry_model(df):
    """Retrain SHORT Entry model with improved labeling"""
    print("\n" + "="*80)
    print("RETRAINING SHORT ENTRY MODEL")
    print("="*80)

    # ========================================================================
    # 1. Create Improved Labels
    # ========================================================================
    labeler = ImprovedEntryLabeling(
        profit_threshold=0.004,
        lookforward_min=6,
        lookforward_max=48,
        lead_time_min=6,
        lead_time_max=24,
        relative_tolerance=0.002,
        scoring_threshold=2
    )

    print("\nCreating improved SHORT Entry labels...")
    labels = labeler.create_short_entry_labels(df)

    # ========================================================================
    # 2. Prepare Features (Use 38 features from current model)
    # ========================================================================
    short_features = [
        'rsi_deviation', 'rsi_direction', 'rsi_extreme',
        'macd_strength', 'macd_direction', 'macd_divergence_abs',
        'price_distance_ma20', 'price_direction_ma20',
        'price_distance_ma50', 'price_direction_ma50',
        'volatility', 'atr_pct', 'atr',
        'negative_momentum', 'negative_acceleration',
        'down_candle_ratio', 'down_candle_body', 'lower_low_streak',
        'resistance_rejection_count', 'bearish_divergence',
        'volume_decline_ratio', 'distribution_signal',
        'down_candle', 'lower_low', 'near_resistance',
        'rejection_from_resistance', 'volume_on_decline', 'volume_on_advance',
        'bear_market_strength', 'trend_strength', 'downtrend_confirmed',
        'volatility_asymmetry', 'below_support', 'support_breakdown',
        'panic_selling', 'downside_volatility', 'upside_volatility', 'ema_12'
    ]

    print(f"\nUsing {len(short_features)} features")

    available_features = [f for f in short_features if f in df.columns]
    missing_features = [f for f in short_features if f not in df.columns]

    if missing_features:
        print(f"\n⚠️ Missing {len(missing_features)} features:")
        for f in missing_features[:10]:
            print(f"   - {f}")
        print(f"\nUsing {len(available_features)} available features")
        short_features = available_features

    X = df[short_features].values
    y = labels

    print(f"\nDataset shape:")
    print(f"  X: {X.shape}")
    print(f"  y: {y.shape}")
    print(f"  Positive rate: {np.sum(y)/len(y)*100:.2f}%")

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False, random_state=42
    )

    print(f"\nTrain/Test Split:")
    print(f"  Train: {len(X_train):,} samples")
    print(f"  Test:  {len(X_test):,} samples")

    # Feature Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train XGBoost Model
    print("\nTraining XGBoost model...")

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=len(y_train[y_train == 0]) / len(y_train[y_train == 1]),
        random_state=42,
        eval_metric='logloss'
    )

    model.fit(
        X_train_scaled, y_train,
        eval_set=[(X_test_scaled, y_test)],
        verbose=False
    )

    # Evaluate
    from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score

    y_pred = model.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print("\n" + "="*80)
    print("SHORT ENTRY MODEL PERFORMANCE")
    print("="*80)
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    # Save Model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"xgboost_short_improved_labeling_{timestamp}"

    model_path = f"models/{model_name}.pkl"
    scaler_path = f"models/{model_name}_scaler.pkl"
    features_path = f"models/{model_name}_features.txt"
    metadata_path = f"models/{model_name}_metadata.json"

    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)

    with open(features_path, 'w') as f:
        for feature in short_features:
            f.write(f"{feature}\n")

    metadata = {
        'model_name': model_name,
        'training_date': timestamp,
        'labeling_method': 'improved_entry_labeling_2of3',
        'num_features': len(short_features),
        'num_train_samples': len(X_train),
        'num_test_samples': len(X_test),
        'positive_rate_train': float(np.sum(y_train) / len(y_train)),
        'positive_rate_test': float(np.sum(y_test) / len(y_test)),
        'scores': {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1)
        },
        'labeling_params': {
            'profit_threshold': 0.004,
            'lookforward_min': 6,
            'lookforward_max': 48,
            'lead_time_min': 6,
            'lead_time_max': 24,
            'relative_tolerance': 0.002,
            'scoring_threshold': 2
        }
    }

    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\n✅ SHORT Model saved: {model_path}")
    print(f"✅ Scaler saved: {scaler_path}")
    print(f"✅ Features saved: {features_path}")
    print(f"✅ Metadata saved: {metadata_path}")

    return model_name, precision

def main():
    """Main retraining workflow"""
    print("="*80)
    print("RETRAIN ENTRY MODELS WITH IMPROVED LABELING")
    print("="*80)
    print("\nGoal: Improve entry model precision from 13.7% to >20%")
    print("Method: Replace old labeling with 2-of-3 scoring system")

    # Load data
    df = load_and_prepare_data()

    # Retrain LONG Entry model
    long_model_name, long_precision = retrain_long_entry_model(df)

    # Retrain SHORT Entry model
    short_model_name, short_precision = retrain_short_entry_model(df)

    # Summary
    print("\n" + "="*80)
    print("RETRAINING COMPLETE")
    print("="*80)
    print(f"\n✅ LONG Entry Model:  {long_model_name}")
    print(f"   Precision: {long_precision:.4f} (Baseline: 0.137)")
    print(f"\n✅ SHORT Entry Model: {short_model_name}")
    print(f"   Precision: {short_precision:.4f}")

    print("\nNext Steps:")
    print("1. Backtest improved models")
    print("2. Compare vs baseline (current models)")
    print("3. If successful, deploy to production")

if __name__ == "__main__":
    main()
