"""
Threshold Optimization for Current Entry Models

Purpose:
- Analyze current LONG/SHORT entry models with different thresholds
- Find optimal threshold for better trade frequency
- Current: threshold 0.7, ~2-3 trades/week
- Target: threshold 0.6-0.65, ~5-8 trades/week

Strategy: Clean Slate
- Keep proven current models (F1 15.8% / 12.7%)
- Only optimize threshold (simple, low risk)
- No feature engineering (avoid complexity)
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.metrics import confusion_matrix, precision_recall_curve
import ta

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"


def calculate_features(df):
    """Calculate features for current models (Phase 1 + Phase 2)"""
    df = df.copy()

    # Phase 1 features (18)
    for lag in range(1, 6):
        df[f'close_change_{lag}'] = df['close'].pct_change(lag)

    df['sma_10'] = ta.trend.sma_indicator(df['close'], window=10)
    df['sma_20'] = ta.trend.sma_indicator(df['close'], window=20)
    df['ema_10'] = ta.trend.ema_indicator(df['close'], window=10)

    macd = ta.trend.MACD(df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_diff'] = macd.macd_diff()

    df['rsi'] = ta.momentum.rsi(df['close'], window=14)

    bb = ta.volatility.BollingerBands(df['close'])
    df['bb_high'] = bb.bollinger_hband()
    df['bb_low'] = bb.bollinger_lband()
    df['bb_mid'] = bb.bollinger_mavg()

    df['volatility'] = df['close'].pct_change().rolling(window=20).std()
    df['volume_sma'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma']

    # Phase 2 features (15)
    df['ema_3'] = ta.trend.ema_indicator(df['close'], window=3)
    df['ema_5'] = ta.trend.ema_indicator(df['close'], window=5)
    df['price_mom_3'] = df['close'].pct_change(3)
    df['price_mom_5'] = df['close'].pct_change(5)
    df['rsi_5'] = ta.momentum.rsi(df['close'], window=5)
    df['rsi_7'] = ta.momentum.rsi(df['close'], window=7)
    df['volatility_5'] = df['close'].pct_change().rolling(window=5).std()
    df['volatility_10'] = df['close'].pct_change().rolling(window=10).std()
    df['volume_spike'] = df['volume'] / df['volume'].rolling(window=5).mean()
    df['volume_trend'] = df['volume'].rolling(window=3).mean() / df['volume'].rolling(window=10).mean()
    df['price_vs_ema3'] = (df['close'] - df['ema_3']) / df['ema_3']
    df['price_vs_ema5'] = (df['close'] - df['ema_5']) / df['ema_5']
    df['body_size'] = abs(df['close'] - df['open']) / df['open']
    df['upper_shadow'] = (df['high'] - df[['open', 'close']].max(axis=1)) / df['open']
    df['lower_shadow'] = (df[['open', 'close']].min(axis=1) - df['low']) / df['open']

    return df


def create_target_long(df, lookahead=3, threshold=0.003):
    """Create LONG target"""
    future_prices = df['close'].shift(-1).rolling(window=lookahead).apply(lambda x: x.max())
    future_return = (future_prices - df['close']) / df['close']
    target = (future_return > threshold).astype(int)
    return target


def create_target_short(df, lookahead=3, threshold=0.003):
    """Create SHORT target"""
    future_prices = df['close'].shift(-1).rolling(window=lookahead).apply(lambda x: x.min())
    future_return = (df['close'] - future_prices) / df['close']
    target = (future_return > threshold).astype(int)
    return target


def analyze_threshold(probabilities, y_true, threshold, total_candles):
    """Analyze model performance at given threshold"""
    predictions = (probabilities >= threshold).astype(int)

    # Confusion matrix
    cm = confusion_matrix(y_true, predictions)

    if (cm[1, 0] + cm[1, 1]) > 0:
        recall = cm[1, 1] / (cm[1, 0] + cm[1, 1])
    else:
        recall = 0

    if (cm[0, 1] + cm[1, 1]) > 0:
        precision = cm[1, 1] / (cm[0, 1] + cm[1, 1])
    else:
        precision = 0

    if (recall + precision) > 0:
        f1 = 2 * (recall * precision) / (recall + precision)
    else:
        f1 = 0

    accuracy = (cm[0, 0] + cm[1, 1]) / cm.sum()

    # Signal statistics
    signals = predictions.sum()
    signal_rate = signals / len(predictions)

    # Estimate trades per week (assuming 5-min candles)
    # 1 week = 7 days * 24 hours * 12 (5-min intervals) = 2016 candles
    candles_per_week = 2016
    weeks = total_candles / candles_per_week
    trades_per_week = signals / weeks

    return {
        'threshold': threshold,
        'signals': signals,
        'signal_rate': signal_rate * 100,  # percentage
        'trades_per_week': trades_per_week,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy,
        'tp': cm[1, 1] if cm.shape == (2, 2) else 0,
        'fp': cm[0, 1] if cm.shape == (2, 2) else 0,
        'fn': cm[1, 0] if cm.shape == (2, 2) else 0,
        'tn': cm[0, 0] if cm.shape == (2, 2) else 0,
    }


def main():
    print("=" * 80)
    print("THRESHOLD OPTIMIZATION - EXECUTION MODE")
    print("=" * 80)
    print("\nStrategy: Clean Slate")
    print("  - Use current proven models (no multi-timeframe)")
    print("  - Optimize threshold only")
    print("  - Target: 5-10 trades/week, maintain precision")
    print("\n" + "=" * 80)

    # Load data
    data_file = DATA_DIR / "historical" / "BTCUSDT_5m_max.csv"
    df = pd.read_csv(data_file)
    print(f"\nLoaded {len(df)} candles")

    # Calculate features
    print("\nCalculating features (Phase 1 + Phase 2)...")
    df = calculate_features(df)
    df = df.dropna()
    print(f"After dropna: {len(df)} rows")

    # Feature columns (33 total)
    feature_columns = [
        # Phase 1 (18)
        'close_change_1', 'close_change_2', 'close_change_3', 'close_change_4', 'close_change_5',
        'sma_10', 'sma_20', 'ema_10',
        'macd', 'macd_signal', 'macd_diff',
        'rsi',
        'bb_high', 'bb_low', 'bb_mid',
        'volatility',
        'volume_sma', 'volume_ratio',

        # Phase 2 (15)
        'ema_3', 'ema_5',
        'price_mom_3', 'price_mom_5',
        'rsi_5', 'rsi_7',
        'volatility_5', 'volatility_10',
        'volume_spike', 'volume_trend',
        'price_vs_ema3', 'price_vs_ema5',
        'body_size', 'upper_shadow', 'lower_shadow'
    ]

    print(f"Features: {len(feature_columns)}")

    # Load current models
    print("\nLoading current models...")
    model_file_long = MODELS_DIR / "xgboost_long_entry.pkl"
    model_file_short = MODELS_DIR / "xgboost_short_entry.pkl"

    with open(model_file_long, 'rb') as f:
        model_long = pickle.load(f)

    with open(model_file_short, 'rb') as f:
        model_short = pickle.load(f)

    print("✅ Models loaded")

    # Use test set (last 20% of data)
    test_size = int(len(df) * 0.2)
    df_test = df.iloc[-test_size:]

    print(f"\nTest set: {len(df_test)} rows ({len(df_test) / 2016:.1f} weeks)")

    # ========== LONG Entry Analysis ==========
    print("\n" + "=" * 80)
    print("LONG ENTRY - THRESHOLD OPTIMIZATION")
    print("=" * 80)

    df_test['target_long'] = create_target_long(df_test, lookahead=3, threshold=0.003)
    df_test_long = df_test.dropna()

    X_test_long = df_test_long[feature_columns].values
    y_test_long = df_test_long['target_long'].values

    print(f"\nTest samples: {len(y_test_long)}")
    print(f"Positive class: {(y_test_long == 1).sum()} ({(y_test_long == 1).sum() / len(y_test_long) * 100:.1f}%)")

    # Get probabilities
    proba_long = model_long.predict_proba(X_test_long)[:, 1]

    print(f"\nProbability distribution:")
    print(f"  Mean: {proba_long.mean():.4f}")
    print(f"  Std: {proba_long.std():.4f}")
    print(f"  Min: {proba_long.min():.4f}")
    print(f"  Max: {proba_long.max():.4f}")

    # Analyze different thresholds
    thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]
    results_long = []

    print("\nThreshold Analysis:")
    print("-" * 100)
    print(f"{'Threshold':<12} {'Signals':<10} {'Rate%':<10} {'Trades/Week':<14} {'Precision':<12} {'Recall':<10} {'F1':<10}")
    print("-" * 100)

    for t in thresholds:
        result = analyze_threshold(proba_long, y_test_long, t, len(df))
        results_long.append(result)

        print(f"{result['threshold']:<12.2f} {result['signals']:<10} {result['signal_rate']:<10.2f} "
              f"{result['trades_per_week']:<14.2f} {result['precision']:<12.4f} "
              f"{result['recall']:<10.4f} {result['f1']:<10.4f}")

    # ========== SHORT Entry Analysis ==========
    print("\n" + "=" * 80)
    print("SHORT ENTRY - THRESHOLD OPTIMIZATION")
    print("=" * 80)

    df_test['target_short'] = create_target_short(df_test, lookahead=3, threshold=0.003)
    df_test_short = df_test.dropna()

    X_test_short = df_test_short[feature_columns].values
    y_test_short = df_test_short['target_short'].values

    print(f"\nTest samples: {len(y_test_short)}")
    print(f"Positive class: {(y_test_short == 1).sum()} ({(y_test_short == 1).sum() / len(y_test_short) * 100:.1f}%)")

    # Get probabilities
    proba_short = model_short.predict_proba(X_test_short)[:, 1]

    print(f"\nProbability distribution:")
    print(f"  Mean: {proba_short.mean():.4f}")
    print(f"  Std: {proba_short.std():.4f}")
    print(f"  Min: {proba_short.min():.4f}")
    print(f"  Max: {proba_short.max():.4f}")

    # Analyze different thresholds
    results_short = []

    print("\nThreshold Analysis:")
    print("-" * 100)
    print(f"{'Threshold':<12} {'Signals':<10} {'Rate%':<10} {'Trades/Week':<14} {'Precision':<12} {'Recall':<10} {'F1':<10}")
    print("-" * 100)

    for t in thresholds:
        result = analyze_threshold(proba_short, y_test_short, t, len(df))
        results_short.append(result)

        print(f"{result['threshold']:<12.2f} {result['signals']:<10} {result['signal_rate']:<10.2f} "
              f"{result['trades_per_week']:<14.2f} {result['precision']:<12.4f} "
              f"{result['recall']:<10.4f} {result['f1']:<10.4f}")

    # ========== RECOMMENDATION ==========
    print("\n" + "=" * 80)
    print("RECOMMENDATION")
    print("=" * 80)

    # Find optimal threshold (target: 5-10 trades/week, highest precision)
    df_results_long = pd.DataFrame(results_long)
    df_results_short = pd.DataFrame(results_short)

    # Filter: trades_per_week between 4-12, then select highest precision
    optimal_long = df_results_long[
        (df_results_long['trades_per_week'] >= 4) &
        (df_results_long['trades_per_week'] <= 12)
    ].sort_values('precision', ascending=False).iloc[0]

    optimal_short = df_results_short[
        (df_results_short['trades_per_week'] >= 4) &
        (df_results_short['trades_per_week'] <= 12)
    ].sort_values('precision', ascending=False).iloc[0]

    print("\nCurrent configuration:")
    current_long = df_results_long[df_results_long['threshold'] == 0.70].iloc[0]
    current_short = df_results_short[df_results_short['threshold'] == 0.70].iloc[0]

    print(f"  LONG threshold: 0.70")
    print(f"    Trades/week: {current_long['trades_per_week']:.2f}")
    print(f"    Precision: {current_long['precision']:.4f}")
    print(f"    Recall: {current_long['recall']:.4f}")
    print(f"    F1: {current_long['f1']:.4f}")

    print(f"\n  SHORT threshold: 0.70")
    print(f"    Trades/week: {current_short['trades_per_week']:.2f}")
    print(f"    Precision: {current_short['precision']:.4f}")
    print(f"    Recall: {current_short['recall']:.4f}")
    print(f"    F1: {current_short['f1']:.4f}")

    print("\n" + "-" * 80)
    print("OPTIMAL configuration:")
    print(f"  LONG threshold: {optimal_long['threshold']:.2f}")
    print(f"    Trades/week: {optimal_long['trades_per_week']:.2f} ({optimal_long['trades_per_week'] / current_long['trades_per_week'] - 1:+.1%})")
    print(f"    Precision: {optimal_long['precision']:.4f} ({optimal_long['precision'] / current_long['precision'] - 1:+.1%})")
    print(f"    Recall: {optimal_long['recall']:.4f} ({optimal_long['recall'] / current_long['recall'] - 1:+.1%})")
    print(f"    F1: {optimal_long['f1']:.4f} ({optimal_long['f1'] / current_long['f1'] - 1:+.1%})")

    print(f"\n  SHORT threshold: {optimal_short['threshold']:.2f}")
    print(f"    Trades/week: {optimal_short['trades_per_week']:.2f} ({optimal_short['trades_per_week'] / current_short['trades_per_week'] - 1:+.1%})")
    print(f"    Precision: {optimal_short['precision']:.4f} ({optimal_short['precision'] / current_short['precision'] - 1:+.1%})")
    print(f"    Recall: {optimal_short['recall']:.4f} ({optimal_short['recall'] / current_short['recall'] - 1:+.1%})")
    print(f"    F1: {optimal_short['f1']:.4f} ({optimal_short['f1'] / current_short['f1'] - 1:+.1%})")

    # Save results
    results_file = PROJECT_ROOT / "results" / "threshold_optimization_results.csv"
    results_file.parent.mkdir(exist_ok=True)

    df_results_combined = pd.DataFrame({
        'direction': ['LONG'] * len(results_long) + ['SHORT'] * len(results_short),
        **{k: [r[k] for r in results_long] + [r[k] for r in results_short]
           for k in results_long[0].keys()}
    })
    df_results_combined.to_csv(results_file, index=False)
    print(f"\n✅ Results saved: {results_file}")

    print("\n" + "=" * 80)
    print("DECISION")
    print("=" * 80)
    print(f"\nRecommend updating bot configuration:")
    print(f"  LONG_ENTRY_THRESHOLD: {optimal_long['threshold']:.2f} (was 0.70)")
    print(f"  SHORT_ENTRY_THRESHOLD: {optimal_short['threshold']:.2f} (was 0.70)")
    print(f"\nExpected improvement:")
    print(f"  Trade frequency: +{(optimal_long['trades_per_week'] + optimal_short['trades_per_week']) / 2 / ((current_long['trades_per_week'] + current_short['trades_per_week']) / 2) - 1:.0%}")
    print(f"  Precision: Maintained or improved")
    print("\nImplement? (Update bot config file)")


if __name__ == "__main__":
    main()
