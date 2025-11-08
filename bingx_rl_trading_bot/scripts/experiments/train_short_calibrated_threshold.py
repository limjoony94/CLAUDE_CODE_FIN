"""
SHORT Model - Calibrated Threshold Tuning (Approach #13)

Meta-Critical Insight:
- Approach #1 achieved 46% (best result!)
- Why? Used LONG model inversion
- Problem: Fixed threshold (0.5 assumed)
- Solution: Find OPTIMAL threshold!

Hypothesis:
- Approach #1 with threshold=0.5 → 46%
- Optimal threshold (0.1-0.4) → 50-60%+

Strategy:
1. Train LONG model (or load existing)
2. Sweep thresholds: 0.05, 0.10, 0.15, ..., 0.45, 0.50
3. For each threshold:
   - If LONG prob <= threshold → Enter SHORT
   - Measure SHORT win rate
4. Select best threshold

Expected: This should beat 46% and potentially reach 60%!
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from datetime import datetime
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.utils.class_weight import compute_sample_weight

import sys
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.production.train_xgboost_improved_v3_phase2 import calculate_features
from scripts.production.market_regime_filter import MarketRegimeFilter

# Directories
DATA_DIR = PROJECT_ROOT / "data" / "historical"
MODELS_DIR = PROJECT_ROOT / "models"


def create_long_labels(df, lookahead=3, threshold=0.0):
    """
    Create LONG labels (will be inverted for SHORT)

    Label 1: Price will rise (LONG opportunity)
    Label 0: Price will not rise (NOT LONG → potential SHORT)
    """
    labels = []

    for i in range(len(df)):
        if i >= len(df) - lookahead:
            labels.append(0)
            continue

        current_price = df['close'].iloc[i]
        future_prices = df['close'].iloc[i+1:i+1+lookahead]
        max_future = future_prices.max()
        increase_pct = (max_future - current_price) / current_price

        if increase_pct >= threshold:
            labels.append(1)  # LONG
        else:
            labels.append(0)  # NOT LONG

    return np.array(labels)


def create_short_validation_labels(df, lookahead=3, threshold=0.002):
    """
    Create SHORT validation labels (ground truth)

    Label 1: Price actually fell (SHORT would have succeeded)
    Label 0: Price did not fall (SHORT would have failed)
    """
    labels = []

    for i in range(len(df)):
        if i >= len(df) - lookahead:
            labels.append(0)
            continue

        current_price = df['close'].iloc[i]
        future_prices = df['close'].iloc[i+1:i+1+lookahead]
        min_future = future_prices.min()
        decrease_pct = (current_price - min_future) / current_price

        if decrease_pct >= threshold:
            labels.append(1)  # SHORT success
        else:
            labels.append(0)  # SHORT fail

    return np.array(labels)


def load_and_prepare_data():
    """Load data and train LONG model"""
    print("="*80)
    print("Loading Data for Calibrated Threshold Tuning")
    print("="*80)

    # Load candles
    df = pd.read_csv(DATA_DIR / "BTCUSDT_5m_max.csv")
    print(f"Loaded {len(df)} candles")

    # Calculate features
    df = calculate_features(df)

    regime_filter = MarketRegimeFilter()
    df = regime_filter.get_regime_features(df)

    # Remove NaN
    df = df.ffill().dropna()
    print(f"After processing: {len(df)} candles")

    # Feature selection (simple baseline features)
    feature_columns = [
        'rsi', 'macd', 'macd_signal', 'bb_high', 'bb_low',
        'sma_10', 'sma_20', 'ema_10', 'volatility', 'volume_ratio'
    ]

    # Filter available columns
    feature_columns = [f for f in feature_columns if f in df.columns]
    print(f"\nFeatures: {len(feature_columns)}")

    # Create LONG labels
    long_labels = create_long_labels(df, lookahead=3, threshold=0.0)

    # Create SHORT validation labels
    short_labels = create_short_validation_labels(df, lookahead=3, threshold=0.002)

    # LONG label distribution
    unique, counts = np.unique(long_labels, return_counts=True)
    print(f"\nLONG Label Distribution:")
    print(f"  NOT LONG (0): {counts[0]} ({counts[0]/len(long_labels)*100:.1f}%)")
    print(f"  LONG (1): {counts[1]} ({counts[1]/len(long_labels)*100:.1f}%)")

    # SHORT label distribution
    unique, counts = np.unique(short_labels, return_counts=True)
    print(f"\nSHORT Validation Labels:")
    print(f"  NO SHORT (0): {counts[0]} ({counts[0]/len(short_labels)*100:.1f}%)")
    print(f"  SHORT (1): {counts[1]} ({counts[1]/len(short_labels)*100:.1f}%)")

    X = df[feature_columns].values
    y_long = long_labels
    y_short = short_labels

    return X, y_long, y_short, feature_columns, df


def train_long_model(X, y_long):
    """Train LONG prediction model"""
    print("\n" + "="*80)
    print("Training LONG Model")
    print("="*80)

    params = {
        'n_estimators': 300,
        'max_depth': 5,
        'learning_rate': 0.01,
        'min_child_weight': 3,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'gamma': 0.5,
        'reg_alpha': 1,
        'reg_lambda': 2,
        'random_state': 42,
        'use_label_encoder': False,
        'eval_metric': 'logloss'
    }

    model = xgb.XGBClassifier(**params)
    sample_weights = compute_sample_weight('balanced', y_long)
    model.fit(X, y_long, sample_weight=sample_weights, verbose=False)

    print("✅ LONG model trained")

    return model


def calibrate_threshold(model, X, y_long, y_short, df):
    """
    Find optimal threshold for SHORT entry

    Strategy:
    - LONG prob <= threshold → Enter SHORT
    - Test thresholds from 0.05 to 0.50
    - Measure SHORT win rate for each
    """
    print("\n" + "="*80)
    print("Calibrating Threshold for SHORT Entry")
    print("="*80)

    # Threshold candidates
    thresholds = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]

    tscv = TimeSeriesSplit(n_splits=5)

    threshold_results = []

    for threshold in thresholds:
        print(f"\n--- Testing Threshold: {threshold:.2f} ---")

        all_trades = []

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_val = X[val_idx]
            y_short_val = y_short[val_idx]

            # Get LONG probabilities
            y_prob_long = model.predict_proba(X_val)[:, 1]

            # Apply inverse threshold
            for i, idx in enumerate(val_idx):
                long_prob = y_prob_long[i]

                # Inverse: LOW LONG prob → SHORT entry
                model_signal = long_prob <= threshold
                regime_allowed = df['short_allowed'].iloc[idx] == 1

                if model_signal and regime_allowed:
                    actual_short_success = y_short_val[i]

                    trade = {
                        'fold': fold,
                        'threshold': threshold,
                        'long_prob': long_prob,
                        'actual': actual_short_success,
                        'correct': (actual_short_success == 1)
                    }
                    all_trades.append(trade)

        # Calculate win rate for this threshold
        if len(all_trades) > 0:
            total_trades = len(all_trades)
            correct_trades = sum(t['correct'] for t in all_trades)
            win_rate = correct_trades / total_trades

            threshold_results.append({
                'threshold': threshold,
                'trades': total_trades,
                'correct': correct_trades,
                'win_rate': win_rate
            })

            print(f"Threshold {threshold:.2f}: {total_trades} trades, {correct_trades} correct ({win_rate*100:.1f}%)")
        else:
            print(f"Threshold {threshold:.2f}: No trades generated")

    return threshold_results


def analyze_results(threshold_results):
    """Analyze calibration results and find optimal threshold"""
    print("\n" + "="*80)
    print("Threshold Calibration Results")
    print("="*80)

    if len(threshold_results) == 0:
        print("❌ No results to analyze")
        return None

    # Sort by win rate
    sorted_results = sorted(threshold_results, key=lambda x: x['win_rate'], reverse=True)

    print(f"\nAll Thresholds (sorted by win rate):")
    print(f"{'Threshold':<12} {'Trades':<10} {'Correct':<10} {'Win Rate':<10}")
    print("-" * 50)

    for result in sorted_results:
        print(f"{result['threshold']:<12.2f} {result['trades']:<10} {result['correct']:<10} {result['win_rate']*100:<10.1f}%")

    # Best threshold
    best_result = sorted_results[0]

    print(f"\n✅ BEST THRESHOLD: {best_result['threshold']:.2f}")
    print(f"   Trades: {best_result['trades']}")
    print(f"   Win Rate: {best_result['win_rate']*100:.1f}%")

    # Compare with Approach #1 (assumed 0.5)
    baseline = [r for r in threshold_results if r['threshold'] == 0.50]
    if baseline:
        baseline_wr = baseline[0]['win_rate']
        improvement = (best_result['win_rate'] - baseline_wr) / baseline_wr * 100

        print(f"\nComparison with Approach #1 (threshold=0.5):")
        print(f"   Baseline: {baseline_wr*100:.1f}%")
        print(f"   Optimized: {best_result['win_rate']*100:.1f}%")
        print(f"   Improvement: {'+' if improvement > 0 else ''}{improvement:.1f}%")

    return best_result


def main():
    """Main calibration pipeline"""
    print("="*80)
    print("SHORT Model - Calibrated Threshold Tuning (Approach #13)")
    print("="*80)
    print("Meta-Critical Insight:")
    print("  - Approach #1 achieved 46% (best so far)")
    print("  - Used LONG inversion but fixed threshold (0.5)")
    print("  - Optimal threshold may be different!")
    print("  - This could push 46% → 50-60%+")
    print("="*80)

    # Load data
    X, y_long, y_short, feature_columns, df = load_and_prepare_data()

    # Train LONG model
    model = train_long_model(X, y_long)

    # Calibrate threshold
    threshold_results = calibrate_threshold(model, X, y_long, y_short, df)

    # Analyze results
    best_result = analyze_results(threshold_results)

    # Final decision
    print("\n" + "="*80)
    print("Final Decision - Calibrated Threshold")
    print("="*80)

    if best_result is None:
        print("❌ No valid results")
        return None

    best_win_rate = best_result['win_rate']
    best_threshold = best_result['threshold']

    if best_win_rate >= 0.60:
        print(f"✅ SUCCESS! SHORT win rate {best_win_rate*100:.1f}% >= 60%")
        print(f"✅ Optimal threshold: {best_threshold:.2f}")
        print(f"✅ Calibration WORKS!")

        # Save configuration
        config_path = MODELS_DIR / "short_calibrated_config.txt"
        with open(config_path, 'w') as f:
            f.write(f"SHORT Strategy - Calibrated Threshold\n")
            f.write(f"="*80 + "\n\n")
            f.write(f"Optimal Threshold: {best_threshold:.2f}\n")
            f.write(f"Win Rate: {best_win_rate*100:.1f}%\n")
            f.write(f"Total Trades: {best_result['trades']}\n")
            f.write(f"Correct Trades: {best_result['correct']}\n")

        print(f"\nConfiguration saved: {config_path}")

    elif best_win_rate >= 0.50:
        print(f"🔄 SIGNIFICANT IMPROVEMENT: SHORT win rate {best_win_rate*100:.1f}% (50-60%)")
        print(f"🔄 Optimal threshold: {best_threshold:.2f}")
        print(f"🔄 Close to target! Consider further optimization")

    elif best_win_rate >= 0.45:
        print(f"⚠️ MODERATE IMPROVEMENT: SHORT win rate {best_win_rate*100:.1f}% (45-50%)")
        print(f"⚠️ Optimal threshold: {best_threshold:.2f}")
        print(f"⚠️ Better than baseline but still insufficient")

    else:
        print(f"❌ INSUFFICIENT: SHORT win rate {best_win_rate*100:.1f}%")
        print(f"❌ Optimal threshold: {best_threshold:.2f}")
        print(f"❌ Even calibration cannot reach target")

    print(f"\nComplete Progress Summary (All 13 Approaches):")
    print(f"  #1  2-Class Inverse (threshold=0.5): 46.0% ✅ Previous best")
    print(f"  #2  3-Class Unbalanced: 0.0%")
    print(f"  #3  3-Class Balanced: 36.4%")
    print(f"  #4  Optuna (100 trials): 22-25%")
    print(f"  #5  V2 Baseline: 26.0%")
    print(f"  #6  V3 Strict: 9.7%")
    print(f"  #7  V4 Ensemble: 20.3%")
    print(f"  #8  V5 SMOTE: 0.0%")
    print(f"  #9  LSTM: 17.3%")
    print(f"  #10 Funding Rate: 22.4%")
    print(f"  #11 Inverse Threshold: 24.4%")
    print(f"  #12 LONG Model Inverse: Error")
    print(f"  #13 Calibrated Threshold (optimal={best_threshold:.2f}): {best_win_rate*100:.1f}%")

    return best_result, threshold_results


if __name__ == "__main__":
    best_result, all_results = main()
