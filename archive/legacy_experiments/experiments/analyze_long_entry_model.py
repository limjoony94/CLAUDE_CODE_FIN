"""
Per-Model Analysis: LONG Entry Model

Purpose: Analyze LONG Entry model performance and labeling optimization

Analysis:
1. Prediction distribution and calibration
2. Success vs failure patterns
3. Labeling parameter sensitivity (lookahead, threshold)
4. Optimal labeling recommendations for LONG Entry specifically
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import sys

# Project setup
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.production.train_xgboost_improved_v3_phase2 import calculate_features
from scripts.production.advanced_technical_features import AdvancedTechnicalFeatures

DATA_DIR = PROJECT_ROOT / "data" / "historical"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"


def create_labels_with_params(df, lookahead, threshold):
    """
    Create labels with specific lookahead and threshold

    Returns:
        labels: Binary array (1 = positive, 0 = negative)
        max_returns: Maximum return achieved in lookahead period
    """
    labels = []
    max_returns = []

    for i in range(len(df)):
        if i >= len(df) - lookahead:
            labels.append(0)
            max_returns.append(0.0)
            continue

        current_price = df['close'].iloc[i]
        future_prices = df['close'].iloc[i+1:i+1+lookahead]
        max_future_price = future_prices.max()

        max_return = (max_future_price - current_price) / current_price
        max_returns.append(max_return)

        if max_return >= threshold:
            labels.append(1)
        else:
            labels.append(0)

    return np.array(labels), np.array(max_returns)


def analyze_predictions_vs_labels(predictions, labels, probabilities):
    """Analyze where model succeeds vs fails"""
    correct = predictions == labels

    # Breakdown by prediction type
    tp = np.sum((predictions == 1) & (labels == 1))  # True Positive
    fp = np.sum((predictions == 1) & (labels == 0))  # False Positive
    tn = np.sum((predictions == 0) & (labels == 0))  # True Negative
    fn = np.sum((predictions == 0) & (labels == 1))  # False Negative

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    # Analyze probability distribution by outcome
    tp_probs = probabilities[(predictions == 1) & (labels == 1)]
    fp_probs = probabilities[(predictions == 1) & (labels == 0)]

    return {
        'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
        'precision': precision, 'recall': recall, 'f1': f1,
        'tp_probs_mean': np.mean(tp_probs) if len(tp_probs) > 0 else 0,
        'fp_probs_mean': np.mean(fp_probs) if len(fp_probs) > 0 else 0
    }


def test_labeling_params(df, model, scaler, feature_columns, lookahead_range, threshold_range):
    """Test different labeling parameters to find optimal for LONG Entry"""
    results = []

    print("\n" + "=" * 80)
    print("Testing Different Labeling Parameters")
    print("=" * 80)

    for lookahead in lookahead_range:
        for threshold in threshold_range:
            # Create labels with these parameters
            labels, max_returns = create_labels_with_params(df, lookahead, threshold)

            # Get model predictions
            features_scaled = scaler.transform(df[feature_columns].values)
            probabilities = model.predict_proba(features_scaled)[:, 1]
            predictions = (probabilities >= 0.7).astype(int)  # threshold 0.7

            # Analyze performance
            analysis = analyze_predictions_vs_labels(predictions, labels, probabilities)

            # Positive rate
            pos_rate = np.sum(labels) / len(labels) * 100

            results.append({
                'lookahead': lookahead,
                'lookahead_minutes': lookahead * 5,
                'threshold_pct': threshold * 100,
                'positive_rate': pos_rate,
                'precision': analysis['precision'] * 100,
                'recall': analysis['recall'] * 100,
                'f1': analysis['f1'] * 100,
                'tp': analysis['tp'],
                'fp': analysis['fp']
            })

            print(f"  Lookahead:{lookahead:2d} ({lookahead*5:3d}min) | "
                  f"Thresh:{threshold*100:4.1f}% | "
                  f"PosRate:{pos_rate:5.2f}% | "
                  f"F1:{analysis['f1']*100:5.2f}% | "
                  f"Prec:{analysis['precision']*100:5.2f}% | "
                  f"Recall:{analysis['recall']*100:5.2f}%")

    return pd.DataFrame(results)


def analyze_signal_quality(df, model, scaler, feature_columns):
    """Analyze quality of signals at different probability thresholds"""
    features_scaled = scaler.transform(df[feature_columns].values)
    probabilities = model.predict_proba(features_scaled)[:, 1]

    # For each signal, check actual outcome
    signal_quality = []

    thresholds_to_test = [0.5, 0.6, 0.7, 0.8, 0.9]

    print("\n" + "=" * 80)
    print("Signal Quality by Threshold")
    print("=" * 80)

    for thresh in thresholds_to_test:
        signals = probabilities >= thresh
        signal_indices = np.where(signals)[0]

        if len(signal_indices) == 0:
            continue

        # Check actual returns after signal
        actual_returns_1h = []
        actual_returns_2h = []
        actual_returns_4h = []

        for idx in signal_indices:
            if idx >= len(df) - 48:  # Need at least 4h of data
                continue

            current_price = df['close'].iloc[idx]

            # 1h return
            price_1h = df['close'].iloc[idx+12] if idx+12 < len(df) else current_price
            ret_1h = (price_1h - current_price) / current_price
            actual_returns_1h.append(ret_1h)

            # 2h return
            price_2h = df['close'].iloc[idx+24] if idx+24 < len(df) else current_price
            ret_2h = (price_2h - current_price) / current_price
            actual_returns_2h.append(ret_2h)

            # 4h return (max hold)
            price_4h = df['close'].iloc[idx+48] if idx+48 < len(df) else current_price
            ret_4h = (price_4h - current_price) / current_price
            actual_returns_4h.append(ret_4h)

        if len(actual_returns_4h) > 0:
            # Win rates (TP 3%, SL 1%)
            wr_1h = np.sum(np.array(actual_returns_1h) >= 0.03) / len(actual_returns_1h) * 100
            wr_2h = np.sum(np.array(actual_returns_2h) >= 0.03) / len(actual_returns_2h) * 100
            wr_4h = np.sum(np.array(actual_returns_4h) >= 0.03) / len(actual_returns_4h) * 100

            avg_ret_4h = np.mean(actual_returns_4h) * 100

            signal_quality.append({
                'threshold': thresh,
                'num_signals': len(signal_indices),
                'signal_rate': len(signal_indices) / len(df) * 100,
                'win_rate_1h': wr_1h,
                'win_rate_2h': wr_2h,
                'win_rate_4h': wr_4h,
                'avg_return_4h': avg_ret_4h
            })

            print(f"\n  Threshold {thresh:.1f}:")
            print(f"    Signals: {len(signal_indices)} ({len(signal_indices)/len(df)*100:.2f}%)")
            print(f"    TP(3%) Win Rate @ 1h: {wr_1h:.1f}%")
            print(f"    TP(3%) Win Rate @ 2h: {wr_2h:.1f}%")
            print(f"    TP(3%) Win Rate @ 4h: {wr_4h:.1f}%")
            print(f"    Avg Return @ 4h: {avg_ret_4h:+.2f}%")

    return pd.DataFrame(signal_quality)


def main():
    print("=" * 80)
    print("PER-MODEL ANALYSIS: LONG ENTRY MODEL")
    print("=" * 80)
    print("\n🎯 목표: LONG Entry 모델의 성능과 최적 labeling 파라미터 분석")

    # Load LONG model
    model_file = MODELS_DIR / "xgboost_v4_phase4_advanced_lookahead3_thresh0.pkl"
    scaler_file = MODELS_DIR / "xgboost_v4_phase4_advanced_lookahead3_thresh0_scaler.pkl"

    with open(model_file, 'rb') as f:
        model = pickle.load(f)
    with open(scaler_file, 'rb') as f:
        scaler = pickle.load(f)
    print(f"✅ LONG Entry model loaded")

    # Load feature columns
    feature_file = MODELS_DIR / "xgboost_v4_phase4_advanced_lookahead3_thresh0_features.txt"
    with open(feature_file, 'r') as f:
        feature_columns = [line.strip() for line in f.readlines()]
    print(f"✅ Features: {len(feature_columns)}")

    # Load data
    data_file = DATA_DIR / "BTCUSDT_5m_max.csv"
    df = pd.read_csv(data_file)
    print(f"✅ Data: {len(df)} candles")

    # Calculate features
    print("\nCalculating features...")
    df = calculate_features(df)
    adv_features = AdvancedTechnicalFeatures(lookback_sr=50, lookback_trend=20)
    df = adv_features.calculate_all_features(df)
    df = df.ffill().dropna()
    print(f"✅ Features calculated: {len(df)} rows")

    # Current labeling
    print("\n" + "=" * 80)
    print("현재 Labeling: lookahead=3 (15min), threshold=0.3%")
    print("=" * 80)

    current_labels, current_returns = create_labels_with_params(df, lookahead=3, threshold=0.003)
    features_scaled = scaler.transform(df[feature_columns].values)
    probabilities = model.predict_proba(features_scaled)[:, 1]
    predictions = (probabilities >= 0.7).astype(int)

    current_analysis = analyze_predictions_vs_labels(predictions, current_labels, probabilities)

    print(f"\n  Positive Rate: {np.sum(current_labels)/len(current_labels)*100:.2f}%")
    print(f"  Precision: {current_analysis['precision']*100:.2f}%")
    print(f"  Recall: {current_analysis['recall']*100:.2f}%")
    print(f"  F1 Score: {current_analysis['f1']*100:.2f}%")
    print(f"\n  True Positives: {current_analysis['tp']}")
    print(f"  False Positives: {current_analysis['fp']}")
    print(f"  True Negatives: {current_analysis['tn']}")
    print(f"  False Negatives: {current_analysis['fn']}")

    # Test different labeling parameters
    lookahead_range = [3, 6, 12, 24, 48]  # 15min, 30min, 1h, 2h, 4h
    threshold_range = [0.003, 0.005, 0.01, 0.015, 0.02, 0.03]  # 0.3% to 3%

    labeling_results = test_labeling_params(
        df, model, scaler, feature_columns,
        lookahead_range, threshold_range
    )

    # Find best labeling
    print("\n" + "=" * 80)
    print("📊 BEST LABELING PARAMETERS")
    print("=" * 80)

    # Best F1
    best_f1 = labeling_results.loc[labeling_results['f1'].idxmax()]
    print(f"\n  Best F1 Score:")
    print(f"    Lookahead: {best_f1['lookahead']:.0f} ({best_f1['lookahead_minutes']:.0f}min)")
    print(f"    Threshold: {best_f1['threshold_pct']:.1f}%")
    print(f"    F1: {best_f1['f1']:.2f}%")
    print(f"    Positive Rate: {best_f1['positive_rate']:.2f}%")

    # Best Precision
    best_prec = labeling_results.loc[labeling_results['precision'].idxmax()]
    print(f"\n  Best Precision:")
    print(f"    Lookahead: {best_prec['lookahead']:.0f} ({best_prec['lookahead_minutes']:.0f}min)")
    print(f"    Threshold: {best_prec['threshold_pct']:.1f}%")
    print(f"    Precision: {best_prec['precision']:.2f}%")

    # Analyze signal quality (실제 trading 성능)
    signal_quality_df = analyze_signal_quality(df, model, scaler, feature_columns)

    # Save results
    labeling_output = RESULTS_DIR / "long_entry_labeling_analysis.csv"
    labeling_results.to_csv(labeling_output, index=False)

    signal_output = RESULTS_DIR / "long_entry_signal_quality.csv"
    signal_quality_df.to_csv(signal_output, index=False)

    print(f"\n✅ Labeling analysis saved: {labeling_output}")
    print(f"✅ Signal quality saved: {signal_output}")

    # Recommendation
    print("\n" + "=" * 80)
    print("💡 RECOMMENDATION FOR LONG ENTRY MODEL")
    print("=" * 80)

    # Look for best real-world performance (win rate @ 4h)
    if len(signal_quality_df) > 0:
        best_practical = signal_quality_df.loc[signal_quality_df['win_rate_4h'].idxmax()]

        print(f"\n  Practical Best (실제 trading 성능):")
        print(f"    Threshold: {best_practical['threshold']:.1f}")
        print(f"    Win Rate @ 4h: {best_practical['win_rate_4h']:.1f}%")
        print(f"    Signals: {best_practical['num_signals']:.0f} ({best_practical['signal_rate']:.2f}%)")

        if best_practical['threshold'] == 0.7:
            print(f"\n  ✅ 현재 threshold (0.7)가 최적!")
        else:
            print(f"\n  📊 Threshold {best_practical['threshold']:.1f} 고려")

    print(f"\n  Labeling 개선 권장사항:")
    print(f"    - 현재 lookahead 3 (15min)는 너무 짧고 noisy")
    print(f"    - Best F1: lookahead {best_f1['lookahead']:.0f}, threshold {best_f1['threshold_pct']:.1f}%")
    print(f"    - 실제 max hold는 4h이므로 lookahead 24-48 테스트 필요")
    print(f"    - Threshold는 strategy TP(3%)에 맞추는 것도 고려")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
