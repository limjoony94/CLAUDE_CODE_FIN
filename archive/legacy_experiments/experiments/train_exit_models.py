"""
Train LONG Exit and SHORT Exit Models

Goal: Learn optimal exit timing for LONG and SHORT positions separately

Labeling Strategy:
- Near-Peak: Exit within 80% of peak P&L
- Future P&L: Exit beats holding for next 1 hour
- Hybrid: BOTH conditions required (AND logic)

Features:
- Base: 37 technical indicators (same as entry)
- Position: 8 position-specific features
- Total: 45 features

Models:
- xgboost_v4_long_exit.pkl
- xgboost_v4_short_exit.pkl
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import sys
from loguru import logger
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier

# Project setup
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.production.train_xgboost_improved_v3_phase2 import calculate_features
from scripts.production.advanced_technical_features import AdvancedTechnicalFeatures

DATA_DIR = PROJECT_ROOT / "data" / "historical"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# Trading parameters (same as production)
STOP_LOSS = 0.01
TAKE_PROFIT = 0.03
MAX_HOLDING_HOURS = 4
TRANSACTION_COST = 0.0002

# Exit labeling parameters
NEAR_PEAK_THRESHOLD = 0.80  # 80% of peak P&L
LOOKAHEAD_HOURS = 1         # 1 hour lookahead (12 candles)
LOOKAHEAD_CANDLES = 12      # 5min * 12 = 60 minutes


def simulate_trade_forward(entry_idx, df, direction="LONG", max_candles=48):
    """
    Simulate a trade forward from entry point

    Args:
        entry_idx: Entry candle index
        df: DataFrame with OHLCV data
        direction: "LONG" or "SHORT"
        max_candles: Max holding period (48 candles = 4 hours)

    Returns:
        dict with trade details
    """
    entry_price = df['close'].iloc[entry_idx]
    peak_pnl = -999999
    trough_pnl = 999999
    candle_data = []

    for offset in range(1, min(max_candles + 1, len(df) - entry_idx)):
        candle_idx = entry_idx + offset
        current_price = df['close'].iloc[candle_idx]

        # Calculate P&L based on direction
        if direction == "LONG":
            pnl_pct = (current_price - entry_price) / entry_price
        else:  # SHORT
            pnl_pct = (entry_price - current_price) / entry_price

        # Track peak and trough
        if pnl_pct > peak_pnl:
            peak_pnl = pnl_pct
        if pnl_pct < trough_pnl:
            trough_pnl = pnl_pct

        # Check exit conditions
        exit_reason = None
        if pnl_pct <= -STOP_LOSS:
            exit_reason = "SL"
        elif pnl_pct >= TAKE_PROFIT:
            exit_reason = "TP"

        candle_data.append({
            'candle_idx': candle_idx,
            'offset': offset,
            'price': current_price,
            'pnl_pct': pnl_pct,
            'peak_pnl': peak_pnl,
            'trough_pnl': trough_pnl,
            'exit_reason': exit_reason
        })

        # Exit if SL/TP hit
        if exit_reason:
            break

    return {
        'entry_idx': entry_idx,
        'entry_price': entry_price,
        'direction': direction,
        'candles': candle_data,
        'final_pnl': candle_data[-1]['pnl_pct'] if candle_data else 0,
        'peak_pnl': peak_pnl,
        'trough_pnl': trough_pnl,
        'duration': len(candle_data)
    }


def label_exit_point(candle, trade, lookahead_candles=LOOKAHEAD_CANDLES):
    """
    Label exit point using hybrid strategy:
    1. Near-Peak: Within 80% of peak P&L
    2. Future P&L: Beats holding for next N candles

    Returns:
        1 if good exit, 0 otherwise
    """
    current_pnl = candle['pnl_pct']
    peak_pnl = trade['peak_pnl']

    # Condition 1: Near peak (80% threshold)
    near_peak = current_pnl >= (peak_pnl * NEAR_PEAK_THRESHOLD)

    # Condition 2: Beats holding for lookahead period
    current_offset = candle['offset']
    future_offset = current_offset + lookahead_candles

    # Get future P&L (if available)
    future_candle = None
    for c in trade['candles']:
        if c['offset'] == future_offset:
            future_candle = c
            break

    if future_candle is None:
        # No future data (near end of trade)
        # Use final P&L as future
        future_pnl = trade['final_pnl']
    else:
        future_pnl = future_candle['pnl_pct']

    beats_holding = current_pnl > future_pnl

    # Hybrid: BOTH conditions required
    return 1 if (near_peak and beats_holding) else 0


def calculate_position_features(candle, trade, df):
    """
    Calculate position-specific features for exit model

    Returns:
        dict with 8 position features
    """
    entry_idx = trade['entry_idx']
    candle_idx = candle['candle_idx']
    offset = candle['offset']

    # Feature 1: Time held (hours)
    time_held = offset / 12  # 5min candles, 12 = 1 hour

    # Feature 2: Current P&L percentage
    current_pnl_pct = candle['pnl_pct']

    # Feature 3: Peak P&L so far
    pnl_peak = candle['peak_pnl']

    # Feature 4: Trough P&L so far
    pnl_trough = candle['trough_pnl']

    # Feature 5: P&L from peak (drawdown)
    pnl_from_peak = current_pnl_pct - pnl_peak

    # Feature 6: Volatility since entry
    entry_to_current = df['close'].iloc[entry_idx:candle_idx+1]
    returns = entry_to_current.pct_change().dropna()
    volatility_since_entry = returns.std() if len(returns) > 1 else 0

    # Feature 7: Volume change (current vs entry)
    entry_volume = df['volume'].iloc[entry_idx]
    current_volume = df['volume'].iloc[candle_idx]
    volume_change = (current_volume - entry_volume) / entry_volume if entry_volume > 0 else 0

    # Feature 8: Momentum shift (recent 3 candles)
    if candle_idx >= 3:
        recent_prices = df['close'].iloc[candle_idx-2:candle_idx+1]
        recent_returns = recent_prices.pct_change().dropna()
        momentum_shift = recent_returns.mean() if len(recent_returns) > 0 else 0
    else:
        momentum_shift = 0

    return {
        'time_held': time_held,
        'current_pnl_pct': current_pnl_pct,
        'pnl_peak': pnl_peak,
        'pnl_trough': pnl_trough,
        'pnl_from_peak': pnl_from_peak,
        'volatility_since_entry': volatility_since_entry,
        'volume_change': volume_change,
        'momentum_shift': momentum_shift
    }


def generate_exit_training_data(df, base_feature_columns, direction="LONG", sample_every=5):
    """
    Generate training data for exit model

    Args:
        df: DataFrame with features calculated
        base_feature_columns: List of 37 base feature names
        direction: "LONG" or "SHORT"
        sample_every: Sample every N candles to reduce data size

    Returns:
        DataFrame with features and labels
    """
    logger.info(f"Generating exit training data for {direction}...")

    samples = []
    trades_count = 0

    # Sample entry points (every N candles to speed up)
    for entry_idx in range(0, len(df) - 60, sample_every):  # Need 60+ candles ahead
        # Simulate trade
        trade = simulate_trade_forward(entry_idx, df, direction, max_candles=48)

        if len(trade['candles']) < 5:
            continue  # Skip very short trades

        trades_count += 1

        # For each candle during the trade
        for candle in trade['candles']:
            candle_idx = candle['candle_idx']

            # Get base features
            base_features = df[base_feature_columns].iloc[candle_idx].to_dict()

            # Get position features
            position_features = calculate_position_features(candle, trade, df)

            # Get label
            label = label_exit_point(candle, trade)

            # Combine features
            sample = {**base_features, **position_features, 'label': label}
            samples.append(sample)

    logger.info(f"  Simulated {trades_count} {direction} trades")
    logger.info(f"  Generated {len(samples)} exit samples")

    return pd.DataFrame(samples)


def train_exit_model(df_samples, direction="LONG"):
    """
    Train exit model (XGBoost Classifier)

    Args:
        df_samples: DataFrame with features and labels
        direction: "LONG" or "SHORT" (for naming)

    Returns:
        trained model, feature columns, scores
    """
    logger.info(f"\n{'=' * 80}")
    logger.info(f"Training {direction} Exit Model")
    logger.info(f"{'=' * 80}")

    # Separate features and labels
    X = df_samples.drop(columns=['label'])
    y = df_samples['label']

    feature_columns = list(X.columns)

    logger.info(f"Training data: {len(X)} samples")
    logger.info(f"Features: {len(feature_columns)}")
    logger.info(f"Label distribution: {y.value_counts().to_dict()}")

    pos_ratio = y.sum() / len(y)
    logger.info(f"Positive ratio: {pos_ratio:.2%}")

    # Split train/test - CRITICAL: Convert to numpy arrays to match entry model scalers
    X_train, X_test, y_train, y_test = train_test_split(
        X.values, y.values, test_size=0.2, random_state=42, stratify=y.values
    )

    # ✅ Apply MinMaxScaler normalization to [-1, 1] range (numpy arrays for consistency)
    logger.info("✅ Applying MinMaxScaler normalization to [-1, 1] range (numpy arrays)...")
    scaler = MinMaxScaler(feature_range=(-1, 1))
    X_train_scaled = scaler.fit_transform(X_train)  # Now numpy array, no feature names
    X_test_scaled = scaler.transform(X_test)
    logger.info("  All features normalized to [-1, 1] range")
    logger.info("  ✅ Scaler trained on numpy arrays (matches entry model format)")

    # Calculate scale_pos_weight
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0

    logger.info(f"Scale pos weight: {scale_pos_weight:.2f}")

    # Train XGBoost
    model = XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        min_child_weight=5,
        gamma=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric='logloss',
        use_label_encoder=False
    )

    model.fit(X_train_scaled, y_train)

    # Evaluate
    from sklearn.metrics import classification_report, confusion_matrix

    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]

    logger.info(f"\n📊 Test Set Performance:")
    logger.info("\n" + classification_report(y_test, y_pred, digits=3))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    logger.info(f"\nConfusion Matrix:")
    logger.info(f"TN: {cm[0][0]}, FP: {cm[0][1]}")
    logger.info(f"FN: {cm[1][0]}, TP: {cm[1][1]}")

    # Feature importance (top 10)
    feature_importance = pd.DataFrame({
        'feature': feature_columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    logger.info(f"\n🔝 Top 10 Features:")
    for idx, row in feature_importance.head(10).iterrows():
        logger.info(f"  {row['feature']}: {row['importance']:.4f}")

    # Check position features importance
    logger.info(f"\n📊 Position Features Importance:")
    position_features = [
        'time_held', 'current_pnl_pct', 'pnl_peak', 'pnl_trough',
        'pnl_from_peak', 'volatility_since_entry', 'volume_change', 'momentum_shift'
    ]
    for feat in position_features:
        importance = feature_importance[feature_importance['feature'] == feat]['importance'].values
        if len(importance) > 0:
            rank = feature_importance[feature_importance['feature'] == feat].index[0] + 1
            logger.info(f"  {feat}: {importance[0]:.4f} (rank #{rank})")

    # Calculate metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

    scores = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred)
    }

    logger.success(f"\n✅ {direction} Exit Model Training Complete with MinMaxScaler!")
    logger.info(f"   F1 Score: {scores['f1']:.3f}")
    logger.info(f"   Precision: {scores['precision']:.3f}")
    logger.info(f"   Recall: {scores['recall']:.3f}")

    return model, scaler, feature_columns, scores, feature_importance


def main():
    logger.info("=" * 80)
    logger.info("Train Exit Models (LONG + SHORT)")
    logger.info("=" * 80)

    # Load data
    logger.info("\nLoading historical data...")
    data_file = DATA_DIR / "BTCUSDT_5m_max.csv"
    df = pd.read_csv(data_file)
    logger.success(f"✅ Data loaded: {len(df)} candles")

    # Calculate features
    logger.info("\nCalculating features...")
    df = calculate_features(df)
    adv_features = AdvancedTechnicalFeatures(lookback_sr=50, lookback_trend=20)
    df = adv_features.calculate_all_features(df)
    df = df.ffill().dropna()
    logger.success(f"✅ Features calculated: {len(df)} rows")

    # Load existing entry model to get feature list
    entry_model_path = MODELS_DIR / "xgboost_v4_realistic_labels.pkl"
    features_path = MODELS_DIR / "xgboost_v4_realistic_labels_features.txt"

    with open(features_path, 'r') as f:
        base_feature_columns = [line.strip() for line in f.readlines()]

    logger.info(f"Base features: {len(base_feature_columns)}")

    # Generate exit training data for LONG
    logger.info(f"\n{'=' * 80}")
    logger.info("Phase 1: Generate LONG Exit Training Data")
    logger.info(f"{'=' * 80}")

    df_long_exit = generate_exit_training_data(
        df=df,
        base_feature_columns=base_feature_columns,
        direction="LONG",
        sample_every=10  # Sample every 10 candles for speed
    )

    logger.success(f"✅ LONG exit data: {len(df_long_exit)} samples")

    # Generate exit training data for SHORT
    logger.info(f"\n{'=' * 80}")
    logger.info("Phase 2: Generate SHORT Exit Training Data")
    logger.info(f"{'=' * 80}")

    df_short_exit = generate_exit_training_data(
        df=df,
        base_feature_columns=base_feature_columns,
        direction="SHORT",
        sample_every=10
    )

    logger.success(f"✅ SHORT exit data: {len(df_short_exit)} samples")

    # Train LONG Exit Model
    logger.info(f"\n{'=' * 80}")
    logger.info("Phase 3: Train LONG Exit Model")
    logger.info(f"{'=' * 80}")

    long_exit_model, long_scaler, long_exit_features, long_scores, long_importance = train_exit_model(
        df_samples=df_long_exit,
        direction="LONG"
    )

    # Save LONG Exit Model
    long_model_path = MODELS_DIR / "xgboost_v4_long_exit.pkl"
    long_scaler_path = MODELS_DIR / "xgboost_v4_long_exit_scaler.pkl"
    long_features_path = MODELS_DIR / "xgboost_v4_long_exit_features.txt"
    long_metadata_path = MODELS_DIR / "xgboost_v4_long_exit_metadata.json"

    with open(long_model_path, 'wb') as f:
        pickle.dump(long_exit_model, f)

    # ✅ Save scaler
    with open(long_scaler_path, 'wb') as f:
        pickle.dump(long_scaler, f)

    with open(long_features_path, 'w') as f:
        f.write('\n'.join(long_exit_features))

    import json
    with open(long_metadata_path, 'w') as f:
        json.dump({
            'model_name': 'xgboost_v4_long_exit',
            'n_features': len(long_exit_features),
            'base_features': len(base_feature_columns),
            'position_features': 8,
            'scores': long_scores,
            'normalized': True,
            'scaler': 'MinMaxScaler',
            'scaler_range': [-1, 1],
            'labeling_strategy': 'near_peak_80pct_and_beats_holding_1h',
            'timestamp': pd.Timestamp.now().isoformat()
        }, f, indent=2)

    logger.success(f"✅ LONG exit model saved: {long_model_path}")
    logger.success(f"✅ LONG scaler saved: {long_scaler_path}")

    # Train SHORT Exit Model
    logger.info(f"\n{'=' * 80}")
    logger.info("Phase 4: Train SHORT Exit Model")
    logger.info(f"{'=' * 80}")

    short_exit_model, short_scaler, short_exit_features, short_scores, short_importance = train_exit_model(
        df_samples=df_short_exit,
        direction="SHORT"
    )

    # Save SHORT Exit Model
    short_model_path = MODELS_DIR / "xgboost_v4_short_exit.pkl"
    short_scaler_path = MODELS_DIR / "xgboost_v4_short_exit_scaler.pkl"
    short_features_path = MODELS_DIR / "xgboost_v4_short_exit_features.txt"
    short_metadata_path = MODELS_DIR / "xgboost_v4_short_exit_metadata.json"

    with open(short_model_path, 'wb') as f:
        pickle.dump(short_exit_model, f)

    # ✅ Save scaler
    with open(short_scaler_path, 'wb') as f:
        pickle.dump(short_scaler, f)

    with open(short_features_path, 'w') as f:
        f.write('\n'.join(short_exit_features))

    with open(short_metadata_path, 'w') as f:
        json.dump({
            'model_name': 'xgboost_v4_short_exit',
            'n_features': len(short_exit_features),
            'base_features': len(base_feature_columns),
            'position_features': 8,
            'scores': short_scores,
            'normalized': True,
            'scaler': 'MinMaxScaler',
            'scaler_range': [-1, 1],
            'labeling_strategy': 'near_peak_80pct_and_beats_holding_1h',
            'timestamp': pd.Timestamp.now().isoformat()
        }, f, indent=2)

    logger.success(f"✅ SHORT exit model saved: {short_model_path}")
    logger.success(f"✅ SHORT scaler saved: {short_scaler_path}")

    # Final Summary
    logger.info(f"\n{'=' * 80}")
    logger.info("✅ ALL EXIT MODELS TRAINED!")
    logger.info(f"{'=' * 80}")

    logger.info(f"\nLONG Exit Model:")
    logger.info(f"  F1 Score: {long_scores['f1']:.3f}")
    logger.info(f"  Precision: {long_scores['precision']:.3f}")
    logger.info(f"  Recall: {long_scores['recall']:.3f}")
    logger.info(f"  Samples: {len(df_long_exit)}")

    logger.info(f"\nSHORT Exit Model:")
    logger.info(f"  F1 Score: {short_scores['f1']:.3f}")
    logger.info(f"  Precision: {short_scores['precision']:.3f}")
    logger.info(f"  Recall: {short_scores['recall']:.3f}")
    logger.info(f"  Samples: {len(df_short_exit)}")

    logger.info(f"\nModel Files:")
    logger.info(f"  {long_model_path}")
    logger.info(f"  {short_model_path}")

    logger.info(f"\n📊 Next Step: Run backtest to compare:")
    logger.info(f"  1. Entry Dual + Exit Rules (current)")
    logger.info(f"  2. Entry Dual + Exit Dual (ML)")

    logger.info(f"\n{'=' * 80}")
    logger.success("✅ Training Complete!")
    logger.info(f"{'=' * 80}")


if __name__ == "__main__":
    main()
