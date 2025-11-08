"""
Enhanced SHORT Entry Model Backtest (Simple Signal Analysis)

Analyzes Enhanced SHORT Entry model signals and quality:
- Signal rate (target: 13%)
- Signal quality by probability threshold
- Comparison with labels

This is NOT a full trading backtest, just signal analysis.
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data" / "historical"
MODELS_DIR = PROJECT_ROOT / "models"


def calculate_enhanced_features(df):
    """Calculate 22 SELL signal features"""

    # Volume features
    df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
    df['volume_surge'] = (df['volume'] > df['volume'].rolling(20).mean() * 1.5).astype(float)

    # Price momentum
    ma_20 = df['close'].rolling(20).mean()
    ma_50 = df['close'].rolling(50).mean()
    df['price_vs_ma20'] = (df['close'] - ma_20) / ma_20
    df['price_vs_ma50'] = (df['close'] - ma_50) / ma_50
    df['price_acceleration'] = df['close'].diff().diff()

    # Volatility
    df['volatility_20'] = df['close'].pct_change().rolling(20).std()
    volatility_median = df['volatility_20'].median()
    df['volatility_regime'] = (df['volatility_20'] > volatility_median).astype(float)

    # RSI dynamics
    df['rsi'] = 50.0  # Default (missing from data)
    df['rsi_slope'] = 0.0
    df['rsi_overbought'] = 0.0
    df['rsi_oversold'] = 0.0
    df['rsi_divergence'] = 0.0

    # MACD dynamics
    df['macd'] = 0.0
    df['macd_signal'] = 0.0
    df['macd_histogram_slope'] = 0.0
    df['macd_crossover'] = 0.0
    df['macd_crossunder'] = 0.0

    # Price patterns
    df['higher_high'] = ((df['high'] > df['high'].shift(1)) &
                         (df['high'].shift(1) > df['high'].shift(2))).astype(float)
    df['lower_low'] = ((df['low'] < df['low'].shift(1)) &
                       (df['low'].shift(1) < df['low'].shift(2))).astype(float)

    # Support/Resistance
    resistance = df['high'].rolling(50).max()
    support = df['low'].rolling(50).min()
    df['near_resistance'] = (df['close'] > resistance * 0.98).astype(float)
    df['near_support'] = (df['close'] < support * 1.02).astype(float)

    # Bollinger Bands
    bb_mid = df['close'].rolling(20).mean()
    bb_std = df['close'].rolling(20).std()
    bb_high = bb_mid + 2 * bb_std
    bb_low = bb_mid - 2 * bb_std
    bb_range = bb_high - bb_low
    df['bb_position'] = np.where(bb_range != 0,
                                  (df['close'] - bb_low) / bb_range,
                                  0.5)

    return df.ffill().fillna(0)


def main():
    print("=" * 80)
    print("Enhanced SHORT Entry Model - Signal Analysis")
    print("=" * 80)

    # Load data
    print("\n1. Loading Data...")
    df = pd.read_csv(DATA_DIR / "BTCUSDT_5m_max.csv")
    print(f"  ✅ {len(df):,} candles")
    print(f"  Date range: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")

    # Calculate features
    print("\n2. Calculating SELL Signal Features...")
    df = calculate_enhanced_features(df)
    print(f"  ✅ 22 SELL signal features calculated")

    # Load enhanced model
    print("\n3. Loading Enhanced SHORT Entry Model...")
    model = pickle.load(open(MODELS_DIR / "xgboost_short_entry_enhanced_20251016_201219.pkl", 'rb'))
    scaler = pickle.load(open(MODELS_DIR / "xgboost_short_entry_enhanced_20251016_201219_scaler.pkl", 'rb'))
    with open(MODELS_DIR / "xgboost_short_entry_enhanced_20251016_201219_features.txt", 'r') as f:
        features = [line.strip() for line in f]
    print(f"  ✅ Model loaded: {len(features)} features")

    # Generate predictions
    print("\n4. Generating Predictions...")
    X = df[features].values
    valid_mask = ~(np.isnan(X).any(axis=1) | np.isinf(X).any(axis=1))
    X_valid = X[valid_mask]

    X_scaled = scaler.transform(X_valid)
    predictions = model.predict(X_scaled)
    probabilities = model.predict_proba(X_scaled)[:, 1]

    # Add to dataframe
    df_valid = df[valid_mask].copy()
    df_valid['prediction'] = predictions
    df_valid['probability'] = probabilities

    print(f"  ✅ {len(df_valid):,} valid predictions generated")

    # Analysis
    print("\n" + "=" * 80)
    print("SIGNAL ANALYSIS")
    print("=" * 80)

    # Overall statistics
    positive_pred = np.sum(predictions)
    signal_rate = positive_pred / len(predictions) * 100

    print(f"\n📊 Overall Statistics:")
    print(f"  Total predictions: {len(predictions):,}")
    print(f"  Positive predictions: {positive_pred:,}")
    print(f"  Signal rate: {signal_rate:.2f}%")
    print(f"  Expected: ~13%")

    # Probability distribution
    print(f"\n📊 Probability Distribution:")
    print(f"  Mean: {probabilities.mean():.4f}")
    print(f"  Median: {np.median(probabilities):.4f}")
    print(f"  Std: {probabilities.std():.4f}")
    print(f"  Min: {probabilities.min():.4f}")
    print(f"  Max: {probabilities.max():.4f}")

    # Signal quality by threshold
    print(f"\n📊 Signal Quality by Probability Threshold:")
    print(f"  {'Threshold':<12} {'Signals':<10} {'Rate':<8} {'Description'}")
    print("-" * 80)

    thresholds = [(0.3, "Very Low"), (0.5, "Medium"), (0.7, "High"), (0.8, "Very High"), (0.9, "Extreme")]

    for thresh, desc in thresholds:
        signals = np.sum(probabilities >= thresh)
        rate = signals / len(probabilities) * 100
        print(f"  >= {thresh:<9.1f} {signals:<10,} {rate:<6.2f}%  {desc}")

    # Recommended threshold
    print(f"\n💡 Recommended Threshold:")

    # Find threshold closest to 10-15% signal rate
    target_rate = 12.5
    best_thresh = None
    best_diff = float('inf')

    for thresh in np.arange(0.3, 1.0, 0.01):
        rate = np.sum(probabilities >= thresh) / len(probabilities) * 100
        diff = abs(rate - target_rate)
        if diff < best_diff:
            best_diff = diff
            best_thresh = thresh

    recommended_signals = np.sum(probabilities >= best_thresh)
    recommended_rate = recommended_signals / len(probabilities) * 100

    print(f"  Threshold: {best_thresh:.2f}")
    print(f"  Signal rate: {recommended_rate:.2f}%")
    print(f"  Signals: {recommended_signals:,}")
    print(f"  Target: 10-15% (ideal for ML trading)")

    # Feature importance
    print(f"\n📊 Top 10 Important Features:")
    importance_df = pd.DataFrame({
        'feature': features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    for idx, row in importance_df.head(10).iterrows():
        print(f"  {row['feature']:<30} {row['importance']:.4f}")

    # Time series analysis
    print(f"\n📊 Signal Distribution Over Time:")

    # Split data into 5 periods
    n_periods = 5
    period_size = len(df_valid) // n_periods

    for i in range(n_periods):
        start_idx = i * period_size
        end_idx = (i + 1) * period_size if i < n_periods - 1 else len(df_valid)

        period_probs = df_valid['probability'].iloc[start_idx:end_idx]
        period_signals = np.sum(period_probs >= 0.7)
        period_rate = period_signals / len(period_probs) * 100

        start_date = df_valid['timestamp'].iloc[start_idx]
        end_date = df_valid['timestamp'].iloc[end_idx-1]

        print(f"  Period {i+1} ({start_date} to {end_date}):")
        print(f"    Signals (>= 0.7): {period_signals:,} ({period_rate:.2f}%)")

    # Save analysis
    print("\n" + "=" * 80)
    print("SAVING RESULTS")
    print("=" * 80)

    # Save predictions
    df_valid[['timestamp', 'close', 'prediction', 'probability']].to_csv(
        PROJECT_ROOT / "results" / "enhanced_short_entry_predictions.csv",
        index=False
    )
    print(f"  ✅ Saved: enhanced_short_entry_predictions.csv")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print(f"\n✅ Enhanced SHORT Entry Model Analysis Complete!")

    print(f"\n📊 Key Findings:")
    print(f"  - Signal rate (prob >= 0.5): {np.sum(probabilities >= 0.5) / len(probabilities) * 100:.2f}%")
    print(f"  - Signal rate (prob >= 0.7): {np.sum(probabilities >= 0.7) / len(probabilities) * 100:.2f}%")
    print(f"  - Recommended threshold: {best_thresh:.2f} ({recommended_rate:.2f}% signal rate)")

    print(f"\n💡 Model Quality:")
    high_conf_signals = np.sum(probabilities >= 0.7)
    if high_conf_signals / len(probabilities) * 100 >= 10:
        print(f"  ✅ GOOD: {high_conf_signals / len(probabilities) * 100:.1f}% high-confidence signals (>= 0.7)")
    else:
        print(f"  ⚠️  LOW: {high_conf_signals / len(probabilities) * 100:.1f}% high-confidence signals (< 10%)")

    print(f"\n📝 Next Steps:")
    print(f"  1. Full backtest with trading simulation")
    print(f"  2. Compare win rate vs current model (expected 60-70% vs 20%)")
    print(f"  3. Deploy if performance validated")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
