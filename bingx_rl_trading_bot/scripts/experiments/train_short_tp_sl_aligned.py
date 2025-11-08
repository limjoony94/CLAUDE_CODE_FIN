"""
SHORT Model Training - TP/SL-Aligned Labeling (Approach 1)

Problem: Current SHORT model trained to predict "0.3% decrease in 15min"
         But actual trading uses TP/SL system (3% TP, 1% SL, 4h max hold)

Solution: Train model to predict "Will trade hit TP before SL within max hold?"
         This EXACTLY matches backtest evaluation criteria!

Expected: >50% win rate (vs current <1%)
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
from scripts.production.market_regime_filter import MarketRegimeFilter

# Directories
DATA_DIR = PROJECT_ROOT / "data" / "historical"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# TP/SL Parameters (MUST match backtest!)
TP_PCT = 0.03  # 3% take profit
SL_PCT = 0.01  # 1% stop loss
MAX_HOLD_CANDLES = 48  # 4 hours (48 * 5min)


def create_short_labels_tp_sl_aligned(df, tp_pct=TP_PCT, sl_pct=SL_PCT, max_hold_candles=MAX_HOLD_CANDLES):
    """
    Create labels that EXACTLY match actual trading outcomes

    Label 1: Trade hits TP (-3%) before SL (+1%) within max_hold (4h)
    Label 0: Trade hits SL first, or max hold expires without TP

    This is what we ACTUALLY evaluate in backtest!
    """
    print("\n" + "="*80)
    print("Creating TP/SL-Aligned Labels for SHORT Trading")
    print("="*80)
    print(f"Take Profit: {tp_pct*100:.1f}% (price drops)")
    print(f"Stop Loss: {sl_pct*100:.1f}% (price rises)")
    print(f"Max Hold: {max_hold_candles} candles ({max_hold_candles*5/60:.1f} hours)")

    labels = []
    trade_outcomes = []  # For analysis

    for i in range(len(df)):
        if i >= len(df) - max_hold_candles:
            labels.append(0)
            trade_outcomes.append('INCOMPLETE')
            continue

        entry_price = df['close'].iloc[i]

        # Calculate TP/SL prices for SHORT trade
        tp_price = entry_price * (1 - tp_pct)  # Price drops 3%
        sl_price = entry_price * (1 + sl_pct)  # Price rises 1%

        tp_hit = False
        sl_hit = False
        exit_candle = None

        # Simulate trade: check each future candle
        for j in range(1, max_hold_candles + 1):
            if i + j >= len(df):
                break

            candle = df.iloc[i + j]

            # Check TP first (priority in real trading)
            if candle['low'] <= tp_price:
                tp_hit = True
                exit_candle = j
                break

            # Then check SL
            if candle['high'] >= sl_price:
                sl_hit = True
                exit_candle = j
                break

        # Determine label and outcome
        if tp_hit and not sl_hit:
            labels.append(1)  # PROFITABLE TRADE
            trade_outcomes.append(f'TP@{exit_candle}')
        elif sl_hit and not tp_hit:
            labels.append(0)  # LOSING TRADE
            trade_outcomes.append(f'SL@{exit_candle}')
        else:
            labels.append(0)  # MAX HOLD (neither TP nor SL)
            trade_outcomes.append('MAX_HOLD')

    labels = np.array(labels)

    # Analysis
    print(f"\nLabel Distribution:")
    unique, counts = np.unique(labels, return_counts=True)
    for label, count in zip(unique, counts):
        pct = count / len(labels) * 100
        label_name = "TP WIN" if label == 1 else "SL/MAX_HOLD LOSS"
        print(f"  {label_name} ({label}): {count:,} ({pct:.2f}%)")

    # Outcome analysis
    outcome_counts = pd.Series(trade_outcomes).value_counts()
    print(f"\nTrade Outcomes:")
    tp_outcomes = outcome_counts[outcome_counts.index.str.startswith('TP')]
    sl_outcomes = outcome_counts[outcome_counts.index.str.startswith('SL')]
    max_hold_count = outcome_counts.get('MAX_HOLD', 0)
    incomplete_count = outcome_counts.get('INCOMPLETE', 0)

    print(f"  TP Hits: {tp_outcomes.sum():,} ({tp_outcomes.sum()/len(labels)*100:.2f}%)")
    print(f"  SL Hits: {sl_outcomes.sum():,} ({sl_outcomes.sum()/len(labels)*100:.2f}%)")
    print(f"  Max Hold: {max_hold_count:,} ({max_hold_count/len(labels)*100:.2f}%)")
    print(f"  Incomplete: {incomplete_count:,} ({incomplete_count/len(labels)*100:.2f}%)")

    # Calculate Win Rate (if we traded every signal)
    tp_count = tp_outcomes.sum()
    total_completed = tp_outcomes.sum() + sl_outcomes.sum() + max_hold_count
    theoretical_win_rate = tp_count / total_completed * 100 if total_completed > 0 else 0
    print(f"\nTheoretical Win Rate (all trades): {theoretical_win_rate:.2f}%")
    print(f"  (This is best-case if model predicts perfectly)")

    return labels, trade_outcomes


def load_and_prepare_data():
    """Load data and calculate all 44 features (same as current model)"""
    print("\n" + "="*80)
    print("Loading Data and Calculating Features")
    print("="*80)

    # Load historical data
    data_file = DATA_DIR / "BTCUSDT_5m_max.csv"
    df = pd.read_csv(data_file)
    print(f"Loaded {len(df):,} candles")

    # Calculate baseline features
    print("Calculating baseline features...")
    df = calculate_features(df)

    # Calculate advanced features
    print("Calculating advanced features...")
    adv_features = AdvancedTechnicalFeatures(lookback_sr=50, lookback_trend=20)
    df = adv_features.calculate_all_features(df)

    # Add regime features
    print("Calculating market regime features...")
    regime_filter = MarketRegimeFilter()
    df = regime_filter.get_regime_features(df)

    # Clean data
    df = df.ffill().dropna()
    print(f"After cleaning: {len(df):,} candles")

    # Define feature columns (same 44 as current model)
    feature_columns = [
        # Baseline features
        'rsi', 'macd', 'macd_signal', 'bb_upper', 'bb_lower',
        'ema_21', 'ema_50', 'atr', 'volume_ratio', 'price_distance_ema',

        # Advanced Phase 4 features
        'atr_ratio', 'bb_width', 'true_range', 'high_low_range',
        'stochrsi', 'willr', 'cci', 'cmo', 'uo', 'roc', 'mfi', 'tsi', 'kst',
        'adx', 'di_plus', 'di_minus', 'aroon_up', 'aroon_down', 'vi',
        'obv', 'cmf',
        'macd_histogram', 'bb_position', 'price_momentum',

        # Pattern features (from AdvancedTechnicalFeatures)
        'distance_to_support_pct', 'distance_to_resistance_pct',
        'num_support_touches', 'num_resistance_touches',
        'upper_trendline_slope', 'lower_trendline_slope',
        'price_vs_upper_trendline_pct', 'price_vs_lower_trendline_pct',
        'rsi_bullish_divergence', 'rsi_bearish_divergence',
        'macd_bullish_divergence', 'macd_bearish_divergence'
    ]

    # Verify features exist
    missing_features = [f for f in feature_columns if f not in df.columns]
    if missing_features:
        print(f"\n⚠️ WARNING: Missing features: {missing_features}")
        feature_columns = [f for f in feature_columns if f in df.columns]

    print(f"\n✅ Features: {len(feature_columns)}")

    return df, feature_columns


def train_short_model_tp_sl(df, feature_columns):
    """Train SHORT model with TP/SL-aligned labels"""
    print("\n" + "="*80)
    print("Training SHORT Model (TP/SL-Aligned)")
    print("="*80)

    # Create TP/SL-aligned labels
    labels, outcomes = create_short_labels_tp_sl_aligned(df)

    # Prepare features
    X = df[feature_columns].values
    y = labels

    # Normalize features
    scaler = MinMaxScaler(feature_range=(-1, 1))
    X_scaled = scaler.fit_transform(X)

    # Time Series Cross-Validation
    print("\n" + "="*80)
    print("Cross-Validation (5 Folds)")
    print("="*80)

    tscv = TimeSeriesSplit(n_splits=5)

    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_scaled)):
        print(f"\nFold {fold + 1}/5")
        print(f"  Train: {len(train_idx):,} samples")
        print(f"  Val: {len(val_idx):,} samples")

        X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Calculate class weights (TP wins are rare)
        positive_count = np.sum(y_train == 1)
        negative_count = np.sum(y_train == 0)
        scale_pos_weight = negative_count / positive_count if positive_count > 0 else 1

        print(f"  Positive samples: {positive_count:,} ({positive_count/len(y_train)*100:.2f}%)")
        print(f"  Negative samples: {negative_count:,} ({negative_count/len(y_train)*100:.2f}%)")
        print(f"  Scale pos weight: {scale_pos_weight:.2f}")

        # Train XGBoost
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

        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False
        )

        # Evaluate
        y_pred_proba = model.predict_proba(X_val)[:, 1]

        # Test at different thresholds
        for threshold in [0.5, 0.6, 0.7, 0.8]:
            y_pred = (y_pred_proba >= threshold).astype(int)

            tp = np.sum((y_pred == 1) & (y_val == 1))
            fp = np.sum((y_pred == 1) & (y_val == 0))
            tn = np.sum((y_pred == 0) & (y_val == 0))
            fn = np.sum((y_pred == 0) & (y_val == 1))

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            # Win rate (among predicted positives)
            win_rate = precision  # For TP/SL labels, precision = win rate!

            print(f"  Threshold {threshold}: Precision={precision*100:.1f}% (Win Rate), Recall={recall*100:.1f}%, F1={f1*100:.1f}%")

        fold_results.append({
            'fold': fold,
            'model': model,
            'val_idx': val_idx,
            'y_val': y_val,
            'y_pred_proba': y_pred_proba
        })

    # Train final model on full dataset
    print("\n" + "="*80)
    print("Training Final Model (Full Dataset)")
    print("="*80)

    positive_count = np.sum(y == 1)
    negative_count = np.sum(y == 0)
    scale_pos_weight = negative_count / positive_count

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

    # Analyze feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': final_model.feature_importances_
    }).sort_values('importance', ascending=False)

    print(f"\nTop 10 Most Important Features:")
    for idx, row in feature_importance.head(10).iterrows():
        print(f"  {row['feature']:30s}: {row['importance']:.4f}")

    return final_model, scaler, fold_results


def analyze_predictions(model, scaler, df, feature_columns):
    """Analyze model predictions across probability thresholds"""
    print("\n" + "="*80)
    print("Prediction Distribution Analysis")
    print("="*80)

    X = df[feature_columns].values
    X_scaled = scaler.transform(X)
    y_pred_proba = model.predict_proba(X_scaled)[:, 1]

    print(f"\nProbability Statistics:")
    print(f"  Mean: {np.mean(y_pred_proba):.4f}")
    print(f"  Std: {np.std(y_pred_proba):.4f}")
    print(f"  Min: {np.min(y_pred_proba):.4f}")
    print(f"  Max: {np.max(y_pred_proba):.4f}")
    print(f"  Median: {np.median(y_pred_proba):.4f}")

    print(f"\nPredictions by Threshold:")
    for threshold in [0.5, 0.6, 0.7, 0.8, 0.9]:
        count = np.sum(y_pred_proba >= threshold)
        pct = count / len(y_pred_proba) * 100
        print(f"  >= {threshold}: {count:,} ({pct:.2f}%)")

    return y_pred_proba


def save_model(model, scaler, feature_columns):
    """Save model, scaler, and metadata"""
    print("\n" + "="*80)
    print("Saving Model")
    print("="*80)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save model
    model_path = MODELS_DIR / f"xgboost_short_tp_sl_aligned_{timestamp}.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"✅ Model saved: {model_path.name}")

    # Save scaler
    scaler_path = MODELS_DIR / f"xgboost_short_tp_sl_aligned_{timestamp}_scaler.pkl"
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"✅ Scaler saved: {scaler_path.name}")

    # Save features
    features_path = MODELS_DIR / f"xgboost_short_tp_sl_aligned_{timestamp}_features.txt"
    with open(features_path, 'w') as f:
        for feature in feature_columns:
            f.write(f"{feature}\n")
    print(f"✅ Features saved: {features_path.name}")

    # Save metadata
    metadata_path = MODELS_DIR / f"xgboost_short_tp_sl_aligned_{timestamp}_metadata.txt"
    with open(metadata_path, 'w') as f:
        f.write("XGBoost SHORT Model - TP/SL-Aligned Labeling\n")
        f.write("="*80 + "\n\n")
        f.write("LABELING METHOD:\n")
        f.write(f"  Type: TP/SL-Aligned (Approach 1)\n")
        f.write(f"  Take Profit: {TP_PCT*100:.1f}% (price drops)\n")
        f.write(f"  Stop Loss: {SL_PCT*100:.1f}% (price rises)\n")
        f.write(f"  Max Hold: {MAX_HOLD_CANDLES} candles ({MAX_HOLD_CANDLES*5/60:.1f} hours)\n\n")
        f.write("FEATURES:\n")
        f.write(f"  Count: {len(feature_columns)}\n")
        f.write(f"  Type: Same as current model (baseline + advanced)\n\n")
        f.write("TRAINING:\n")
        f.write(f"  Timestamp: {timestamp}\n")
        f.write(f"  Model: XGBoost\n")
        f.write(f"  Normalization: MinMaxScaler [-1, 1]\n")

    print(f"✅ Metadata saved: {metadata_path.name}")

    return model_path, scaler_path


def main():
    """Main training pipeline"""
    print("="*80)
    print("SHORT Model Training - TP/SL-Aligned Labeling (Approach 1)")
    print("="*80)
    print("\n🎯 Goal: Train model to predict PROFITABLE SHORT trades")
    print("       (TP hit before SL, matching actual backtest criteria)")
    print("\n⚠️ Current model predicts '0.3% drop in 15min' (WRONG objective!)")
    print("✅ New model predicts 'TP before SL in 4h' (CORRECT objective!)")

    # Load data
    df, feature_columns = load_and_prepare_data()

    # Train model
    model, scaler, fold_results = train_short_model_tp_sl(df, feature_columns)

    # Analyze predictions
    predictions = analyze_predictions(model, scaler, df, feature_columns)

    # Save model
    model_path, scaler_path = save_model(model, scaler, feature_columns)

    # Final summary
    print("\n" + "="*80)
    print("✅ TRAINING COMPLETE")
    print("="*80)
    print(f"\nModel: {model_path.name}")
    print(f"Scaler: {scaler_path.name}")

    print(f"\n📋 Next Steps:")
    print(f"  1. Run signal quality analysis:")
    print(f"     python scripts/experiments/analyze_short_entry_model.py")
    print(f"     (Update script to use new model)")

    print(f"\n  2. Run backtest:")
    print(f"     python scripts/experiments/backtest_dual_model_mainnet.py")
    print(f"     (Update script to use new SHORT model)")

    print(f"\n  3. Compare results:")
    print(f"     - Current SHORT: 20% win rate (backtest), <1% win rate (signal quality)")
    print(f"     - New SHORT: ??? win rate (TO BE TESTED)")

    print(f"\n  4. Deployment criteria:")
    print(f"     - Minimum: >50% win rate")
    print(f"     - Target: >60% win rate")
    print(f"     - If met: Replace current SHORT model in production")

    print("\n" + "="*80)


if __name__ == "__main__":
    main()
