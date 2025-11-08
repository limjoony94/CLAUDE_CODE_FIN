"""
Retrain EXIT Models with Improved Labeling

Uses Combined Multi-Criteria labeling methodology:
1. Lead-time peak/trough detection
2. Profit threshold
3. Relative performance
4. Momentum confirmation

Expected improvements:
- Proper calibration (high prob = good exits, NOT inverted)
- Better quality labels (10-20% positive rate)
- Higher precision (>60% vs 55.2%)
- Better backtest performance (>+11.60% target)

Author: Claude Code
Date: 2025-10-16
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import sys
from datetime import datetime
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit, cross_val_score
from sklearn.metrics import precision_score, recall_score, f1_score, classification_report

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.production.train_xgboost_improved_v3_phase2 import calculate_features
from scripts.production.advanced_technical_features import AdvancedTechnicalFeatures
from src.features.sell_signal_features import SellSignalFeatures
from src.labeling.improved_exit_labeling import ImprovedExitLabeling, simulate_trades_for_labeling

DATA_DIR = PROJECT_ROOT / "data" / "historical"
MODELS_DIR = PROJECT_ROOT / "models"

ENTRY_THRESHOLD_LONG = 0.70
ENTRY_THRESHOLD_SHORT = 0.65


def load_entry_models():
    """Load trained ENTRY models for trade simulation"""
    print("Loading ENTRY models for trade simulation...")

    # LONG Entry
    with open(MODELS_DIR / "xgboost_v4_phase4_advanced_lookahead3_thresh0.pkl", 'rb') as f:
        long_entry_model = pickle.load(f)
    with open(MODELS_DIR / "xgboost_v4_phase4_advanced_lookahead3_thresh0_scaler.pkl", 'rb') as f:
        long_entry_scaler = pickle.load(f)
    with open(MODELS_DIR / "xgboost_v4_phase4_advanced_lookahead3_thresh0_features.txt", 'r') as f:
        long_entry_features = [line.strip() for line in f]

    # SHORT Entry
    with open(MODELS_DIR / "xgboost_short_peak_trough_20251016_131939.pkl", 'rb') as f:
        short_entry_model = pickle.load(f)
    with open(MODELS_DIR / "xgboost_short_peak_trough_20251016_131939_scaler.pkl", 'rb') as f:
        short_entry_scaler = pickle.load(f)
    with open(MODELS_DIR / "xgboost_short_peak_trough_20251016_131939_features.txt", 'r') as f:
        short_entry_features = [line.strip() for line in f]

    print("✅ ENTRY models loaded")

    return {
        'long_entry_model': long_entry_model,
        'long_entry_scaler': long_entry_scaler,
        'long_entry_features': long_entry_features,
        'short_entry_model': short_entry_model,
        'short_entry_scaler': short_entry_scaler,
        'short_entry_features': short_entry_features
    }


def prepare_exit_features(df):
    """
    Prepare EXIT features with enhanced market context

    2025-10-16 Enhancement: Added market context features
    - Volume analysis
    - Price momentum and trends
    - Volatility metrics
    - RSI dynamics

    These features help differentiate good vs bad exit timing
    without requiring position context.
    """
    print("\nCalculating enhanced market context features...")

    # Volume features
    df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
    df['volume_surge'] = (df['volume'] > df['volume'].rolling(20).mean() * 1.5).astype(float)

    # Price momentum features (check column availability)
    if 'sma_50' in df.columns:
        df['price_vs_ma20'] = (df['close'] - df['sma_50']) / df['sma_50']
    elif 'ma_50' in df.columns:
        df['price_vs_ma20'] = (df['close'] - df['ma_50']) / df['ma_50']
    else:
        # Calculate if not available
        ma_20 = df['close'].rolling(20).mean()
        df['price_vs_ma20'] = (df['close'] - ma_20) / ma_20

    if 'sma_200' in df.columns:
        df['price_vs_ma50'] = (df['close'] - df['sma_200']) / df['sma_200']
    elif 'ma_200' in df.columns:
        df['price_vs_ma50'] = (df['close'] - df['ma_200']) / df['ma_200']
    else:
        # Calculate if not available
        ma_50 = df['close'].rolling(50).mean()
        df['price_vs_ma50'] = (df['close'] - ma_50) / ma_50

    # Volatility features
    df['returns'] = df['close'].pct_change()
    df['volatility_20'] = df['returns'].rolling(20).std()
    df['volatility_regime'] = (df['volatility_20'] > df['volatility_20'].rolling(100).median()).astype(float)

    # RSI dynamics
    df['rsi_slope'] = df['rsi'].diff(3) / 3  # Rate of change over 3 candles (15 min)
    df['rsi_overbought'] = (df['rsi'] > 70).astype(float)
    df['rsi_oversold'] = (df['rsi'] < 30).astype(float)
    df['rsi_divergence'] = 0  # Placeholder for now

    # MACD dynamics
    df['macd_histogram_slope'] = df['macd_hist'].diff(3) / 3 if 'macd_hist' in df.columns else 0
    df['macd_crossover'] = ((df['macd'] > df['macd_signal']) &
                            (df['macd'].shift(1) <= df['macd_signal'].shift(1))).astype(float)
    df['macd_crossunder'] = ((df['macd'] < df['macd_signal']) &
                             (df['macd'].shift(1) >= df['macd_signal'].shift(1))).astype(float)

    # Price patterns
    df['higher_high'] = (df['high'] > df['high'].shift(1)).astype(float)
    df['lower_low'] = (df['low'] < df['low'].shift(1)).astype(float)
    df['price_acceleration'] = df['close'].diff().diff()  # Second derivative

    # Support/Resistance proximity
    if 'support_level' in df.columns and 'resistance_level' in df.columns:
        df['near_resistance'] = (df['close'] > df['resistance_level'] * 0.98).astype(float)
        df['near_support'] = (df['close'] < df['support_level'] * 1.02).astype(float)
    else:
        df['near_resistance'] = 0
        df['near_support'] = 0

    # Bollinger Band position
    if 'bb_upper' in df.columns and 'bb_lower' in df.columns:
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    else:
        df['bb_position'] = 0.5

    # Clean NaN
    df = df.ffill().bfill()

    print(f"✅ Enhanced features calculated")

    return df


def train_exit_model(df, labels, model_name, side):
    """
    Train EXIT model with improved labels

    Args:
        df: DataFrame with features
        labels: Improved labels (1 = should exit, 0 = hold)
        model_name: Name for saving model
        side: 'LONG' or 'SHORT'

    Returns:
        model, scaler, features, metrics
    """
    print(f"\n{'='*80}")
    print(f"Training {side} EXIT Model with Improved Labels")
    print(f"{'='*80}")

    # Define EXIT features (ENHANCED 2025-10-16)
    # Basic technical features
    technical_features = [
        'rsi', 'macd', 'macd_signal', 'macd_hist',
        'bb_upper', 'bb_middle', 'bb_lower', 'bb_width',
        'atr', 'volume_sma_ratio',
        'ema_12', 'ema_26', 'sma_50', 'sma_200',
        'stoch_k', 'stoch_d',
        'adx', 'plus_di', 'minus_di',
        'cci', 'roc', 'williams_r',
        'obv', 'cmf', 'mfi',
        'vwap', 'typical_price',
        'recent_high', 'recent_low',
        'support_level', 'resistance_level',
        'trend_strength', 'volatility_regime',
        'volume_surge', 'price_acceleration',
        'momentum_quality', 'breakout_signal',
        'sell_signal_strength', 'sell_momentum_score',
        # Enhanced market context features (2025-10-16)
        'volume_ratio', 'price_vs_ma20', 'price_vs_ma50',
        'volatility_20', 'rsi_slope', 'rsi_overbought', 'rsi_oversold',
        'rsi_divergence', 'macd_histogram_slope', 'macd_crossover',
        'macd_crossunder', 'higher_high', 'lower_low',
        'near_resistance', 'near_support', 'bb_position'
    ]

    # Position features (we'll simulate these)
    position_features = [
        'holding_minutes',
        'current_pnl_pct',
        'pnl_peak',
        'pnl_trough',
        'pnl_from_peak',
        'pnl_from_trough',
        'consecutive_candles_up',
        'consecutive_candles_down'
    ]

    all_features = technical_features + position_features

    # Filter features that exist in df
    available_features = [f for f in all_features if f in df.columns]

    print(f"Using {len(available_features)} features")
    print(f"  Technical: {len([f for f in available_features if f in technical_features])}")
    print(f"  Position: {len([f for f in available_features if f in position_features])}")

    # For now, use only technical features (position features need trade simulation)
    # TODO: Add position feature simulation
    features_to_use = [f for f in available_features if f in technical_features]

    # Prepare X, y
    X = df[features_to_use].values
    y = labels

    # Remove NaN rows
    valid_idx = ~np.isnan(X).any(axis=1)
    X = X[valid_idx]
    y = y[valid_idx]

    print(f"\nDataset:")
    print(f"  Total samples: {len(y):,}")
    print(f"  Positive (exit): {y.sum():,} ({y.sum()/len(y)*100:.2f}%)")
    print(f"  Negative (hold): {(len(y)-y.sum()):,} ({(len(y)-y.sum())/len(y)*100:.2f}%)")

    # Check if positive rate is reasonable
    positive_rate = y.sum() / len(y)
    if positive_rate < 0.05:
        print(f"⚠️ WARNING: Positive rate very low ({positive_rate*100:.2f}%)")
        print(f"   Labeling may be too strict")
    elif positive_rate > 0.30:
        print(f"⚠️ WARNING: Positive rate high ({positive_rate*100:.2f}%)")
        print(f"   Labeling may be too loose")
    else:
        print(f"✅ Positive rate reasonable ({positive_rate*100:.2f}%)")

    # Scale features
    scaler = MinMaxScaler(feature_range=(-1, 1))
    X_scaled = scaler.fit_transform(X)

    # Calculate class weight
    neg_count = len(y) - y.sum()
    pos_count = y.sum()
    scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0

    print(f"\nClass weight: {scale_pos_weight:.2f}")

    # Train XGBoost model
    print(f"\nTraining XGBoost model...")

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1
    )

    # Time series cross-validation
    print(f"Running 5-fold time series cross-validation...")
    tscv = TimeSeriesSplit(n_splits=5)
    cv_scores = cross_val_score(model, X_scaled, y, cv=tscv, scoring='precision', n_jobs=-1)

    print(f"Cross-validation precision: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"Folds: {', '.join([f'{s:.4f}' for s in cv_scores])}")

    # Train final model
    print(f"\nTraining final model on all data...")
    model.fit(X_scaled, y)

    # Evaluate on training data (for comparison with original models)
    y_pred = model.predict(X_scaled)
    y_proba = model.predict_proba(X_scaled)[:, 1]

    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)

    print(f"\n{'='*80}")
    print(f"Training Metrics:")
    print(f"{'='*80}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1 Score: {f1:.4f}")

    # Probability distribution analysis
    print(f"\nProbability Distribution:")
    print(f"  Mean: {y_proba.mean():.4f}")
    print(f"  Median: {np.median(y_proba):.4f}")
    print(f"  Std: {y_proba.std():.4f}")
    print(f"  Min: {y_proba.min():.4f}")
    print(f"  Max: {y_proba.max():.4f}")

    # Check if distribution is balanced (problem indicator)
    if 0.45 < y_proba.mean() < 0.55:
        print(f"⚠️ WARNING: Mean probability ~0.5 (balanced distribution)")
        print(f"   This may indicate same problem as original models")
    else:
        print(f"✅ Mean probability {y_proba.mean():.4f} (not balanced)")

    # Probability percentiles
    print(f"\n  Percentiles:")
    for p in [10, 25, 50, 75, 90, 95, 99]:
        val = np.percentile(y_proba, p)
        print(f"    {p}th: {val:.4f}")

    # Signal quality by probability range
    print(f"\nSignal Quality by Probability Range:")
    ranges = [(0.0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 1.0)]

    for low, high in ranges:
        mask = (y_proba >= low) & (y_proba < high)
        if mask.sum() > 0:
            range_precision = y[mask].sum() / mask.sum()
            print(f"  {low:.1f}-{high:.1f}: {mask.sum():6,} samples, Precision: {range_precision:.2%}")

    # Check for inversion
    print(f"\n🔍 Inversion Check:")
    low_prob_precision = y[y_proba < 0.5].sum() / (y_proba < 0.5).sum() if (y_proba < 0.5).sum() > 0 else 0
    high_prob_precision = y[y_proba >= 0.5].sum() / (y_proba >= 0.5).sum() if (y_proba >= 0.5).sum() > 0 else 0

    print(f"  Low prob (<0.5): Precision = {low_prob_precision:.2%}")
    print(f"  High prob (>=0.5): Precision = {high_prob_precision:.2%}")

    if low_prob_precision > high_prob_precision:
        print(f"  ❌ MODEL IS INVERTED! (low prob > high prob precision)")
    else:
        print(f"  ✅ Model NOT inverted (high prob >= low prob precision)")

    # Save model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = MODELS_DIR / f"{model_name}_improved_{timestamp}.pkl"
    scaler_path = MODELS_DIR / f"{model_name}_improved_{timestamp}_scaler.pkl"
    features_path = MODELS_DIR / f"{model_name}_improved_{timestamp}_features.txt"

    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    with open(features_path, 'w') as f:
        f.write('\n'.join(features_to_use))

    print(f"\n✅ Model saved:")
    print(f"   {model_path}")
    print(f"   {scaler_path}")
    print(f"   {features_path}")

    return model, scaler, features_to_use, {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'cv_precision_mean': cv_scores.mean(),
        'cv_precision_std': cv_scores.std(),
        'prob_mean': y_proba.mean(),
        'prob_median': np.median(y_proba),
        'positive_rate': positive_rate,
        'inverted': low_prob_precision > high_prob_precision
    }


def main():
    print("="*80)
    print("Retrain EXIT Models with Improved Labeling")
    print("="*80)

    # Load data
    print("\nLoading data...")
    df = pd.read_csv(DATA_DIR / "BTCUSDT_5m_max.csv")
    df = calculate_features(df)
    adv = AdvancedTechnicalFeatures(lookback_sr=50, lookback_trend=20)
    df = adv.calculate_all_features(df)
    sell = SellSignalFeatures()
    df = sell.calculate_all_features(df)
    df = df.ffill().dropna()
    print(f"✅ {len(df):,} candles loaded")

    # Calculate enhanced EXIT features (2025-10-16)
    df = prepare_exit_features(df)

    # Load ENTRY models
    entry_models = load_entry_models()

    # Initialize improved labeling (RELAXED parameters 2025-10-16)
    labeler = ImprovedExitLabeling(
        lead_time_min=3,         # RELAXED: 6 → 3 (15 min)
        lead_time_max=24,        # RELAXED: 12 → 24 (2 hours)
        profit_threshold=0.003,  # RELAXED: 0.5% → 0.3%
        peak_threshold=0.002,    # RELAXED: 0.3% → 0.2%
        momentum_rsi_high=55.0,
        momentum_rsi_low=45.0,
        relative_tolerance=0.001 # RELAXED: 0.05% → 0.1%
    )

    print(f"\nImproved Labeling Configuration (RELAXED):")
    print(f"  Lead time: {labeler.lead_time_min}-{labeler.lead_time_max} candles (15min - 2h)")
    print(f"  Profit threshold: {labeler.profit_threshold*100:.1f}%")
    print(f"  Peak threshold: {labeler.peak_threshold*100:.1f}%")
    print(f"  Relative tolerance: {labeler.relative_tolerance*100:.2f}%")
    print(f"  RSI thresholds: LONG>{labeler.momentum_rsi_high}, SHORT<{labeler.momentum_rsi_low}")

    # ========================================================================
    # LONG EXIT Model
    # ========================================================================

    print(f"\n" + "="*80)
    print("Phase 1: LONG EXIT Model")
    print("="*80)

    # Simulate LONG trades
    print(f"\nSimulating LONG trades for labeling...")
    long_trades = simulate_trades_for_labeling(
        df,
        entry_models['long_entry_model'],
        entry_models['long_entry_scaler'],
        entry_models['long_entry_features'],
        ENTRY_THRESHOLD_LONG,
        side='LONG'
    )

    print(f"✅ Simulated {len(long_trades):,} LONG trades")

    # Create improved labels
    print(f"\nCreating improved LONG EXIT labels...")
    long_exit_labels = labeler.create_long_exit_labels(df, long_trades)

    # Validate labels
    long_stats = labeler.validate_labels(long_exit_labels, df)
    print(f"\nLONG EXIT Label Statistics:")
    print(f"  Total candles: {long_stats['total_candles']:,}")
    print(f"  Positive labels: {long_stats['positive_labels']:,}")
    print(f"  Positive rate: {long_stats['positive_rate']*100:.2f}%")
    print(f"  Average spacing: {long_stats['avg_spacing']:.1f} candles")
    print(f"  Minimum spacing: {long_stats['min_spacing']} candles")

    # Train LONG EXIT model
    long_exit_model, long_exit_scaler, long_exit_features, long_metrics = train_exit_model(
        df, long_exit_labels, "xgboost_long_exit", "LONG"
    )

    # ========================================================================
    # SHORT EXIT Model
    # ========================================================================

    print(f"\n" + "="*80)
    print("Phase 2: SHORT EXIT Model")
    print("="*80)

    # Simulate SHORT trades
    print(f"\nSimulating SHORT trades for labeling...")
    short_trades = simulate_trades_for_labeling(
        df,
        entry_models['short_entry_model'],
        entry_models['short_entry_scaler'],
        entry_models['short_entry_features'],
        ENTRY_THRESHOLD_SHORT,
        side='SHORT'
    )

    print(f"✅ Simulated {len(short_trades):,} SHORT trades")

    # Create improved labels
    print(f"\nCreating improved SHORT EXIT labels...")
    short_exit_labels = labeler.create_short_exit_labels(df, short_trades)

    # Validate labels
    short_stats = labeler.validate_labels(short_exit_labels, df)
    print(f"\nSHORT EXIT Label Statistics:")
    print(f"  Total candles: {short_stats['total_candles']:,}")
    print(f"  Positive labels: {short_stats['positive_labels']:,}")
    print(f"  Positive rate: {short_stats['positive_rate']*100:.2f}%")
    print(f"  Average spacing: {short_stats['avg_spacing']:.1f} candles")
    print(f"  Minimum spacing: {short_stats['min_spacing']} candles")

    # Train SHORT EXIT model
    short_exit_model, short_exit_scaler, short_exit_features, short_metrics = train_exit_model(
        df, short_exit_labels, "xgboost_short_exit", "SHORT"
    )

    # ========================================================================
    # Summary
    # ========================================================================

    print(f"\n" + "="*80)
    print("RETRAINING COMPLETE")
    print("="*80)

    print(f"\nLONG EXIT Model:")
    print(f"  Precision: {long_metrics['precision']:.4f}")
    print(f"  CV Precision: {long_metrics['cv_precision_mean']:.4f} ± {long_metrics['cv_precision_std']:.4f}")
    print(f"  Prob Mean: {long_metrics['prob_mean']:.4f}")
    print(f"  Positive Rate: {long_metrics['positive_rate']*100:.2f}%")
    print(f"  Inverted: {'YES ❌' if long_metrics['inverted'] else 'NO ✅'}")

    print(f"\nSHORT EXIT Model:")
    print(f"  Precision: {short_metrics['precision']:.4f}")
    print(f"  CV Precision: {short_metrics['cv_precision_mean']:.4f} ± {short_metrics['cv_precision_std']:.4f}")
    print(f"  Prob Mean: {short_metrics['prob_mean']:.4f}")
    print(f"  Positive Rate: {short_metrics['positive_rate']*100:.2f}%")
    print(f"  Inverted: {'YES ❌' if short_metrics['inverted'] else 'NO ✅'}")

    print(f"\n" + "="*80)
    print("Next Steps:")
    print("="*80)
    print(f"1. Backtest retrained models")
    print(f"2. Compare performance to inverted logic (+11.60% target)")
    print(f"3. If better: Deploy retrained models")
    print(f"4. If not: Keep using inverted logic, iterate on labeling")

    print(f"\n" + "="*80)


if __name__ == "__main__":
    main()
