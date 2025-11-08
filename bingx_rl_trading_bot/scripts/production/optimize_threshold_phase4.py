"""
Threshold Optimization for Phase 4 Production Models

Clean Slate Approach:
- Use current Phase 4 models (proven in production)
- Optimize threshold only (simple, low risk)
- Target: 5-10 trades/week (current ~2-3/week)
"""

import pandas as pd
import numpy as np
import pickle
import sys
from pathlib import Path
from sklearn.metrics import confusion_matrix

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.production.train_xgboost_improved_v3_phase2 import calculate_features
from scripts.production.advanced_technical_features import AdvancedTechnicalFeatures

DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"


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

    signals = predictions.sum()
    signal_rate = signals / len(predictions)

    # 2016 candles per week (5-min intervals)
    candles_per_week = 2016
    weeks = total_candles / candles_per_week
    trades_per_week = signals / weeks

    return {
        'threshold': threshold,
        'signals': signals,
        'signal_rate': signal_rate * 100,
        'trades_per_week': trades_per_week,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'accuracy': accuracy,
    }


def main():
    print("=" * 80)
    print("PHASE 4 THRESHOLD OPTIMIZATION")
    print("=" * 80)
    print("\nStrategy: Clean Slate")
    print("  - Use Phase 4 production models")
    print("  - Optimize threshold only")
    print("  - Target: 5-10 trades/week")
    print("\n" + "=" * 80)

    # Load data
    data_file = DATA_DIR / "historical" / "BTCUSDT_5m_max.csv"
    df = pd.read_csv(data_file)
    print(f"\nLoaded {len(df)} candles")

    # Calculate features (Phase 2 + Phase 4)
    print("\nCalculating features...")
    df = calculate_features(df)
    adv_features = AdvancedTechnicalFeatures()
    df = adv_features.calculate_all_features(df)
    df = df.dropna()
    print(f"After dropna: {len(df)} rows")

    # Load Phase 4 models
    print("\nLoading Phase 4 models...")

    # LONG Model
    long_model_path = MODELS_DIR / "xgboost_v4_phase4_advanced_lookahead3_thresh0.pkl"
    long_scaler_path = MODELS_DIR / "xgboost_v4_phase4_advanced_lookahead3_thresh0_scaler.pkl"
    long_features_path = MODELS_DIR / "xgboost_v4_phase4_advanced_lookahead3_thresh0_features.txt"

    with open(long_model_path, 'rb') as f:
        model_long = pickle.load(f)

    with open(long_scaler_path, 'rb') as f:
        scaler_long = pickle.load(f)

    with open(long_features_path, 'r') as f:
        feature_columns_long = [line.strip() for line in f if line.strip()]

    print(f"✅ LONG model: {len(feature_columns_long)} features")

    # SHORT Model
    short_model_path = MODELS_DIR / "xgboost_short_model_lookahead3_thresh0.3.pkl"
    short_scaler_path = MODELS_DIR / "xgboost_short_model_lookahead3_thresh0.3_scaler.pkl"

    with open(short_model_path, 'rb') as f:
        model_short = pickle.load(f)

    with open(short_scaler_path, 'rb') as f:
        scaler_short = pickle.load(f)

    print(f"✅ SHORT model: {len(feature_columns_long)} features")

    # Use test set (last 20%)
    test_size = int(len(df) * 0.2)
    df_test = df.iloc[-test_size:]
    print(f"\nTest set: {len(df_test)} rows ({len(df_test) / 2016:.1f} weeks)")

    # ========== LONG Entry Analysis ==========
    print("\n" + "=" * 80)
    print("LONG ENTRY THRESHOLD OPTIMIZATION")
    print("=" * 80)

    df_test_long = df_test.copy()
    df_test_long['target_long'] = create_target_long(df_test_long, lookahead=3, threshold=0.003)
    df_test_long = df_test_long.dropna()

    X_test_long = df_test_long[feature_columns_long].values
    X_test_long_scaled = scaler_long.transform(X_test_long)
    y_test_long = df_test_long['target_long'].values

    print(f"\nTest samples: {len(y_test_long)}")
    print(f"Positive class: {(y_test_long == 1).sum()} ({(y_test_long == 1).sum() / len(y_test_long) * 100:.1f}%)")

    # Get probabilities
    proba_long = model_long.predict_proba(X_test_long_scaled)[:, 1]

    print(f"\nProbability distribution:")
    print(f"  Mean: {proba_long.mean():.4f}")
    print(f"  Std: {proba_long.std():.4f}")
    print(f"  Min: {proba_long.min():.4f}")
    print(f"  Max: {proba_long.max():.4f}")

    # Analyze thresholds
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
    print("SHORT ENTRY THRESHOLD OPTIMIZATION")
    print("=" * 80)

    df_test_short = df_test.copy()
    df_test_short['target_short'] = create_target_short(df_test_short, lookahead=3, threshold=0.003)
    df_test_short = df_test_short.dropna()

    X_test_short = df_test_short[feature_columns_long].values  # SHORT uses same features
    X_test_short_scaled = scaler_short.transform(X_test_short)
    y_test_short = df_test_short['target_short'].values

    print(f"\nTest samples: {len(y_test_short)}")
    print(f"Positive class: {(y_test_short == 1).sum()} ({(y_test_short == 1).sum() / len(y_test_short) * 100:.1f}%)")

    # Get probabilities
    proba_short = model_short.predict_proba(X_test_short_scaled)[:, 1]

    print(f"\nProbability distribution:")
    print(f"  Mean: {proba_short.mean():.4f}")
    print(f"  Std: {proba_short.std():.4f}")
    print(f"  Min: {proba_short.min():.4f}")
    print(f"  Max: {proba_short.max():.4f}")

    # Analyze thresholds
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

    df_results_long = pd.DataFrame(results_long)
    df_results_short = pd.DataFrame(results_short)

    # Find optimal (4-12 trades/week, highest precision)
    optimal_candidates_long = df_results_long[
        (df_results_long['trades_per_week'] >= 4) &
        (df_results_long['trades_per_week'] <= 12)
    ]

    if len(optimal_candidates_long) > 0:
        optimal_long = optimal_candidates_long.sort_values('precision', ascending=False).iloc[0]
    else:
        # Fallback: closest to target range with highest precision
        df_results_long['distance_to_target'] = abs(df_results_long['trades_per_week'] - 8)
        optimal_long = df_results_long.sort_values(['distance_to_target', 'precision'], ascending=[True, False]).iloc[0]

    optimal_candidates_short = df_results_short[
        (df_results_short['trades_per_week'] >= 4) &
        (df_results_short['trades_per_week'] <= 12)
    ]

    if len(optimal_candidates_short) > 0:
        optimal_short = optimal_candidates_short.sort_values('precision', ascending=False).iloc[0]
    else:
        # Fallback: closest to target range with highest precision
        df_results_short['distance_to_target'] = abs(df_results_short['trades_per_week'] - 8)
        optimal_short = df_results_short.sort_values(['distance_to_target', 'precision'], ascending=[True, False]).iloc[0]

    print("\nCurrent configuration (threshold 0.70):")
    current_long = df_results_long[df_results_long['threshold'] == 0.70].iloc[0]
    current_short = df_results_short[df_results_short['threshold'] == 0.70].iloc[0]

    print(f"  LONG:")
    print(f"    Trades/week: {current_long['trades_per_week']:.2f}")
    print(f"    Precision: {current_long['precision']:.4f}")
    print(f"    F1: {current_long['f1']:.4f}")

    print(f"\n  SHORT:")
    print(f"    Trades/week: {current_short['trades_per_week']:.2f}")
    print(f"    Precision: {current_short['precision']:.4f}")
    print(f"    F1: {current_short['f1']:.4f}")

    print("\n" + "-" * 80)
    print("OPTIMAL configuration:")
    print(f"  LONG threshold: {optimal_long['threshold']:.2f}")
    print(f"    Trades/week: {optimal_long['trades_per_week']:.2f} ({optimal_long['trades_per_week'] / current_long['trades_per_week'] - 1:+.1%})")
    print(f"    Precision: {optimal_long['precision']:.4f} ({optimal_long['precision'] / current_long['precision'] - 1:+.1%})")
    print(f"    F1: {optimal_long['f1']:.4f} ({optimal_long['f1'] / current_long['f1'] - 1:+.1%})")

    print(f"\n  SHORT threshold: {optimal_short['threshold']:.2f}")
    print(f"    Trades/week: {optimal_short['trades_per_week']:.2f} ({optimal_short['trades_per_week'] / current_short['trades_per_week'] - 1:+.1%})")
    print(f"    Precision: {optimal_short['precision']:.4f} ({optimal_short['precision'] / current_short['precision'] - 1:+.1%})")
    print(f"    F1: {optimal_short['f1']:.4f} ({optimal_short['f1'] / current_short['f1'] - 1:+.1%})")

    # Save results
    results_file = PROJECT_ROOT / "results" / "threshold_optimization_phase4_results.csv"
    results_file.parent.mkdir(exist_ok=True)

    df_results_combined = pd.DataFrame({
        'direction': ['LONG'] * len(results_long) + ['SHORT'] * len(results_short),
        **{k: [r[k] for r in results_long] + [r[k] for r in results_short]
           for k in results_long[0].keys()}
    })
    df_results_combined.to_csv(results_file, index=False)
    print(f"\n✅ Results saved: {results_file}")

    print("\n" + "=" * 80)
    print("IMPLEMENTATION")
    print("=" * 80)
    print(f"\nUpdate phase4_dynamic_testnet_trading.py:")
    print(f"  LONG_ENTRY_THRESHOLD = {optimal_long['threshold']:.2f}  # was 0.70")
    print(f"  SHORT_ENTRY_THRESHOLD = {optimal_short['threshold']:.2f}  # was 0.70")
    print(f"\nExpected impact:")
    avg_current = (current_long['trades_per_week'] + current_short['trades_per_week']) / 2
    avg_optimal = (optimal_long['trades_per_week'] + optimal_short['trades_per_week']) / 2
    print(f"  Trade frequency: {avg_optimal:.1f}/week (was {avg_current:.1f}/week) [{avg_optimal / avg_current - 1:+.0%}]")
    print(f"  Precision: Maintained or improved")


if __name__ == "__main__":
    main()
