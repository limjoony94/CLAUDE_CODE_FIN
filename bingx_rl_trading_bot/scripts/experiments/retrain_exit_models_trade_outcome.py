"""
Retrain EXIT Models with Trade-Outcome Labeling
================================================

Applies Trade-Outcome labeling to EXIT models (same philosophy as Entry models).

Key Improvement:
- Old: Peak/Trough based labeling (pattern recognition)
- New: Trade-Outcome based labeling (actual P&L outcomes)

Expected Benefits:
- Better exit timing (exit near actual peaks, not after)
- Fewer false exits (hold through noise)
- Consistent with Entry model philosophy
- +40-50% improvement (based on Entry model gains)

Author: Claude Code
Date: 2025-10-19
"""

import pandas as pd
import numpy as np
import pickle
import joblib
from pathlib import Path
import sys
from datetime import datetime
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiments.calculate_all_features import calculate_all_features
from src.labeling.trade_outcome_exit_labeling import TradeOutcomeExitLabeling

DATA_DIR = PROJECT_ROOT / "data" / "historical"
MODELS_DIR = PROJECT_ROOT / "models"

# Opportunity Gating Thresholds (same as current system)
ENTRY_THRESHOLD_LONG = 0.65
ENTRY_THRESHOLD_SHORT = 0.70
GATE_THRESHOLD = 0.001
LONG_AVG_RETURN = 0.0041
SHORT_AVG_RETURN = 0.0047
LEVERAGE = 4  # 4x leverage


def load_entry_models():
    """Load Trade-Outcome Entry models for trade simulation"""
    print("Loading Trade-Outcome Entry models...")

    # LONG Entry (Trade-Outcome Full Dataset)
    long_model_path = MODELS_DIR / "xgboost_long_trade_outcome_full_20251018_233146.pkl"
    with open(long_model_path, 'rb') as f:
        long_entry_model = pickle.load(f)

    long_scaler_path = MODELS_DIR / "xgboost_long_trade_outcome_full_20251018_233146_scaler.pkl"
    long_entry_scaler = joblib.load(long_scaler_path)

    long_features_path = MODELS_DIR / "xgboost_long_trade_outcome_full_20251018_233146_features.txt"
    with open(long_features_path, 'r') as f:
        long_entry_features = [line.strip() for line in f.readlines()]

    # SHORT Entry (Trade-Outcome Full Dataset)
    short_model_path = MODELS_DIR / "xgboost_short_trade_outcome_full_20251018_233146.pkl"
    with open(short_model_path, 'rb') as f:
        short_entry_model = pickle.load(f)

    short_scaler_path = MODELS_DIR / "xgboost_short_trade_outcome_full_20251018_233146_scaler.pkl"
    short_entry_scaler = joblib.load(short_scaler_path)

    short_features_path = MODELS_DIR / "xgboost_short_trade_outcome_full_20251018_233146_features.txt"
    with open(short_features_path, 'r') as f:
        short_entry_features = [line.strip() for line in f.readlines() if line.strip()]

    print(f"  ✅ LONG Entry: {len(long_entry_features)} features")
    print(f"  ✅ SHORT Entry: {len(short_entry_features)} features")

    return {
        'long_entry_model': long_entry_model,
        'long_entry_scaler': long_entry_scaler,
        'long_entry_features': long_entry_features,
        'short_entry_model': short_entry_model,
        'short_entry_scaler': short_entry_scaler,
        'short_entry_features': short_entry_features
    }


def simulate_trades_with_opportunity_gating(
    df: pd.DataFrame,
    long_model, long_scaler, long_features,
    short_model, short_scaler, short_features,
    side: str = 'LONG',
    max_hold_candles: int = 240  # 20 hours (5-min candles)
) -> list:
    """
    Simulate trades with Opportunity Gating strategy

    Same logic as production bot:
    - LONG: Enter if prob >= 0.65
    - SHORT: Enter if prob >= 0.70 AND EV(SHORT) > EV(LONG) + 0.001
    """
    trades = []
    position = None

    for i in range(len(df)):
        if position is None:
            # Check entry signals
            try:
                # Get LONG signal
                long_feature_values = df[long_features].iloc[i].values.reshape(1, -1)
                long_feature_values_scaled = long_scaler.transform(long_feature_values)
                long_prob = long_model.predict_proba(long_feature_values_scaled)[0][1]

                # Get SHORT signal
                short_feature_values = df[short_features].iloc[i].values.reshape(1, -1)
                short_feature_values_scaled = short_scaler.transform(short_feature_values)
                short_prob = short_model.predict_proba(short_feature_values_scaled)[0][1]

                # Entry decision
                should_enter = False
                entry_side = None

                if side == 'LONG' and long_prob >= ENTRY_THRESHOLD_LONG:
                    should_enter = True
                    entry_side = 'LONG'

                elif side == 'SHORT' and short_prob >= ENTRY_THRESHOLD_SHORT:
                    # Opportunity Gating
                    long_ev = long_prob * LONG_AVG_RETURN
                    short_ev = short_prob * SHORT_AVG_RETURN
                    opportunity_cost = short_ev - long_ev

                    if opportunity_cost > GATE_THRESHOLD:
                        should_enter = True
                        entry_side = 'SHORT'

                if should_enter:
                    position = {
                        'side': entry_side,
                        'entry_idx': i,
                        'entry_price': df.iloc[i]['close'],
                        'entry_long_prob': long_prob,
                        'entry_short_prob': short_prob
                    }

            except Exception as e:
                # Missing features, skip
                continue

        else:
            # In position - check exit (simple: max hold or end of data)
            hold_time = i - position['entry_idx']

            if hold_time >= max_hold_candles or i == len(df) - 1:
                # Exit
                exit_price = df.iloc[i]['close']
                entry_price = position['entry_price']

                if position['side'] == 'LONG':
                    pnl_pct = ((exit_price - entry_price) / entry_price) * LEVERAGE
                else:  # SHORT
                    pnl_pct = ((entry_price - exit_price) / entry_price) * LEVERAGE

                trades.append({
                    'side': position['side'],
                    'entry_idx': position['entry_idx'],
                    'exit_idx': i,
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'pnl_pct': pnl_pct,
                    'hold_candles': hold_time
                })

                position = None

    print(f"  Simulated {len(trades)} {side} trades")
    return trades


def prepare_exit_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare features for EXIT model

    Uses subset of features relevant for exit decisions:
    - Price momentum
    - Volume patterns
    - Volatility
    - RSI, MACD signals
    - Recent price action
    """
    # Use only features that actually exist in DataFrame
    exit_feature_names = [
        # Price changes
        'price_change', 'price_change_pct',
        'close_change_1', 'close_change_2', 'close_change_3',

        # Technical indicators
        'rsi', 'rsi_oversold', 'rsi_overbought',
        'macd', 'macd_signal',
        'ema_3', 'ema_5', 'ema_10',

        # Bollinger Bands
        'bb_high', 'bb_low', 'bb_mid',

        # Volatility
        'atr', 'atr_pct',
        'upside_volatility', 'downside_volatility',

        # Volume
        'volume', 'volume_sma',

        # Price levels
        'close', 'high', 'low', 'open',

        # Candle patterns
        'body_size', 'body_to_range_ratio',
        'up_candle', 'down_candle',

        # Momentum
        'momentum', 'momentum_signal'
    ]

    # Filter to only existing columns
    available_features = [f for f in exit_feature_names if f in df.columns]

    if len(available_features) < 10:
        raise ValueError(f"Too few exit features available: {len(available_features)}")

    exit_features = df[available_features].copy()

    # Handle NaN
    exit_features = exit_features.ffill().fillna(0)

    print(f"  Exit features selected: {len(available_features)}")

    return exit_features


def train_exit_model(
    df: pd.DataFrame,
    labels: np.ndarray,
    side: str
):
    """
    Train EXIT model with Trade-Outcome labels

    Args:
        df: DataFrame with all features
        labels: Binary labels (1=EXIT, 0=HOLD)
        side: 'LONG' or 'SHORT'
    """
    print(f"\n{'='*80}")
    print(f"Training {side} EXIT Model (Trade-Outcome Labeling)")
    print(f"{'='*80}")

    # Prepare features
    exit_features_df = prepare_exit_features(df)
    feature_columns = exit_features_df.columns.tolist()

    print(f"\nFeatures: {len(feature_columns)}")
    print(f"Samples: {len(exit_features_df):,}")
    print(f"EXIT labels: {np.sum(labels):,} ({np.sum(labels)/len(labels)*100:.1f}%)")

    # Split data (time series split)
    split_idx = int(len(exit_features_df) * 0.8)
    X_train = exit_features_df.iloc[:split_idx]
    y_train = labels[:split_idx]
    X_test = exit_features_df.iloc[split_idx:]
    y_test = labels[split_idx:]

    print(f"\nTrain: {len(X_train):,} samples")
    print(f"Test:  {len(X_test):,} samples")

    # Scale features (convert to numpy first for consistency)
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train.values)
    X_test_scaled = scaler.transform(X_test.values)

    # Train XGBoost
    print("\nTraining XGBoost...")

    # Calculate scale_pos_weight for class imbalance
    neg_count = np.sum(y_train == 0)
    pos_count = np.sum(y_train == 1)
    scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric='logloss'
    )

    model.fit(
        X_train_scaled, y_train,
        eval_set=[(X_test_scaled, y_test)],
        verbose=False
    )

    # Evaluate
    y_train_pred = model.predict(X_train_scaled)
    y_test_pred = model.predict(X_test_scaled)

    print(f"\n{'='*80}")
    print("TRAIN Performance:")
    print(f"{'='*80}")
    print(classification_report(y_train, y_train_pred, target_names=['HOLD', 'EXIT']))

    print(f"\n{'='*80}")
    print("TEST Performance:")
    print(f"{'='*80}")
    print(classification_report(y_test, y_test_pred, target_names=['HOLD', 'EXIT']))

    # Save model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"xgboost_{side.lower()}_exit_trade_outcome_{timestamp}"

    model_path = MODELS_DIR / f"{model_name}.pkl"
    scaler_path = MODELS_DIR / f"{model_name}_scaler.pkl"
    features_path = MODELS_DIR / f"{model_name}_features.txt"

    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    joblib.dump(scaler, scaler_path)

    with open(features_path, 'w') as f:
        for feature in feature_columns:
            f.write(f"{feature}\n")

    print(f"\n✅ Model saved:")
    print(f"   {model_path.name}")
    print(f"   {scaler_path.name}")
    print(f"   {features_path.name}")

    return model, scaler, feature_columns


def main():
    """Main training pipeline"""
    print("="*80)
    print("EXIT MODEL RETRAINING - Trade-Outcome Labeling")
    print("="*80)

    # Load data
    print("\nLoading data...")
    csv_files = sorted(DATA_DIR.glob("btcusdt_5m_*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {DATA_DIR}")

    latest_file = csv_files[-1]
    print(f"  Using: {latest_file.name}")

    df = pd.read_csv(latest_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)

    print(f"  Loaded {len(df):,} candles")
    print(f"  Period: {df['timestamp'].min()} to {df['timestamp'].max()}")

    # Calculate all features
    print("\nCalculating features...")
    df_features = calculate_all_features(df)

    # Remove NaN rows
    df_features = df_features.dropna().reset_index(drop=True)
    print(f"  Features calculated: {len(df_features.columns)} columns")
    print(f"  Valid samples: {len(df_features):,}")

    # Load Entry models
    entry_models = load_entry_models()

    # Initialize Trade-Outcome Exit Labeling
    labeler = TradeOutcomeExitLabeling(
        exit_tolerance=0.005,  # 0.5% (on 4x leverage = 2% unleveraged)
        min_hold_candles=3,     # 15 minutes minimum
        stop_loss_threshold=-0.03,  # -3% (4x leverage)
        take_profit_threshold=0.03   # +3% (4x leverage)
    )

    # Train LONG Exit Model
    print("\n" + "="*80)
    print("LONG EXIT MODEL")
    print("="*80)

    # Simulate LONG trades
    print("\nSimulating LONG trades for labeling...")
    long_trades = simulate_trades_with_opportunity_gating(
        df_features,
        entry_models['long_entry_model'],
        entry_models['long_entry_scaler'],
        entry_models['long_entry_features'],
        entry_models['short_entry_model'],
        entry_models['short_entry_scaler'],
        entry_models['short_entry_features'],
        side='LONG'
    )

    # Create labels
    long_exit_labels, long_outcomes = labeler.create_long_exit_labels(
        df_features, long_trades, leverage=LEVERAGE
    )

    # Analyze outcomes
    long_stats = labeler.analyze_outcomes(long_outcomes)
    print(f"\nLONG Exit Label Statistics:")
    print(f"  Exit Rate: {long_stats.get('exit_rate', 0)*100:.1f}%")
    print(f"  Exit Quality: {long_stats.get('avg_exit_quality', 0)*100:.1f}%")
    print(f"  Hold Improvement: {long_stats.get('avg_hold_improvement', 0)*100:+.2f}%")
    if long_stats.get('exit_reasons'):
        print(f"  Exit Reasons:")
        for reason, count in long_stats['exit_reasons'].items():
            print(f"    {reason}: {count}")

    # Train LONG model
    long_model, long_scaler, long_features = train_exit_model(
        df_features, long_exit_labels, 'LONG'
    )

    # Train SHORT Exit Model
    print("\n" + "="*80)
    print("SHORT EXIT MODEL")
    print("="*80)

    # Simulate SHORT trades
    print("\nSimulating SHORT trades for labeling...")
    short_trades = simulate_trades_with_opportunity_gating(
        df_features,
        entry_models['long_entry_model'],
        entry_models['long_entry_scaler'],
        entry_models['long_entry_features'],
        entry_models['short_entry_model'],
        entry_models['short_entry_scaler'],
        entry_models['short_entry_features'],
        side='SHORT'
    )

    # Create labels
    short_exit_labels, short_outcomes = labeler.create_short_exit_labels(
        df_features, short_trades, leverage=LEVERAGE
    )

    # Analyze outcomes
    short_stats = labeler.analyze_outcomes(short_outcomes)
    print(f"\nSHORT Exit Label Statistics:")
    print(f"  Exit Rate: {short_stats.get('exit_rate', 0)*100:.1f}%")
    print(f"  Exit Quality: {short_stats.get('avg_exit_quality', 0)*100:.1f}%")
    print(f"  Hold Improvement: {short_stats.get('avg_hold_improvement', 0)*100:+.2f}%")
    if short_stats.get('exit_reasons'):
        print(f"  Exit Reasons:")
        for reason, count in short_stats['exit_reasons'].items():
            print(f"    {reason}: {count}")

    # Train SHORT model
    short_model, short_scaler, short_features = train_exit_model(
        df_features, short_exit_labels, 'SHORT'
    )

    print("\n" + "="*80)
    print("✅ EXIT MODEL RETRAINING COMPLETE")
    print("="*80)
    print("\nNext Steps:")
    print("1. Run backtest with new Trade-Outcome Exit models")
    print("2. Compare performance vs Peak/Trough Exit models")
    print("3. If improved, update production bot configuration")


if __name__ == "__main__":
    main()
