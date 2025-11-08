"""
Out-of-Sample Validation for Multi-Timeframe Models

목적:
- Test set과 다른 시간대 데이터로 검증
- Overfitting 여부 확인
- 실제 성능 추정

Critical:
- 이 테스트가 pass되어야 다음 단계 진행
- F1 >= 20% (현행 15.8% + 4%p) 필요
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
import sys

# Add parent to path for imports
sys.path.append(str(Path(__file__).parent))
from multi_timeframe_features import MultiTimeframeFeatures

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"


def calculate_features(df):
    """Calculate ALL features (same as training)"""
    import ta

    df = df.copy()

    # Original features
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

    # Short-term features
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

    # Multi-timeframe features
    mtf_features = MultiTimeframeFeatures()
    df = mtf_features.calculate_all_features(df)

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


def validate_model(model, X, y, direction, dataset_name):
    """Validate model on dataset"""
    print(f"\n{'=' * 80}")
    print(f"{direction} Entry Model - {dataset_name}")
    print(f"{'=' * 80}")

    # Predictions
    y_pred_proba = model.predict_proba(X)[:, 1]
    y_pred = model.predict(X)

    # Metrics
    accuracy = model.score(X, y)
    print(f"\nAccuracy: {accuracy:.4f}")

    # Classification report
    print(f"\nClassification Report:")
    print(classification_report(y, y_pred, target_names=['Not Enter', 'Enter'], zero_division=0))

    # Confusion matrix
    cm = confusion_matrix(y, y_pred)
    print(f"\nConfusion Matrix:")
    print(cm)

    # Calculate F1
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

    print(f"\n{'=' * 80}")
    print(f"KEY METRICS:")
    print(f"  Recall: {recall:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    print(f"{'=' * 80}")

    # Probability distribution
    print(f"\nProbability Distribution:")
    print(f"  Mean: {y_pred_proba.mean():.4f}")
    print(f"  Std: {y_pred_proba.std():.4f}")
    for threshold_check in [0.5, 0.6, 0.7, 0.8, 0.9]:
        count = (y_pred_proba > threshold_check).sum()
        pct = count / len(y_pred_proba) * 100
        print(f"  Prob > {threshold_check}: {count} ({pct:.2f}%)")

    return {
        'accuracy': accuracy,
        'recall': recall,
        'precision': precision,
        'f1': f1,
        'mean_prob': y_pred_proba.mean()
    }


if __name__ == "__main__":
    print("=" * 80)
    print("Out-of-Sample Validation")
    print("=" * 80)

    # Load full data
    data_file = DATA_DIR / "historical" / "BTCUSDT_5m_max.csv"
    df_full = pd.read_csv(data_file)
    print(f"\nTotal candles: {len(df_full)}")

    # Calculate features
    print("\n계산 중: All 69 features...")
    df_full = calculate_features(df_full)
    df_full = df_full.dropna()
    print(f"After features & dropna: {len(df_full)} rows")

    # Split data
    # Training used: first 60% for train, next 20% for val, last 20% for test
    # Total: 29,686 rows after dropna
    # Train: 0-17,811 (60%)
    # Val: 17,812-23,748 (20%)
    # Test: 23,749-29,686 (20%)

    # Out-of-sample: LATEST data (completely unseen)
    # Use last 2 weeks (약 4,000 candles)
    oos_size = min(4000, int(len(df_full) * 0.15))

    # Original test set (used in training report)
    test_start = int(len(df_full) * 0.8)
    df_test = df_full.iloc[test_start:test_start + int(len(df_full) * 0.2)]

    # Out-of-sample set (newest data, never seen before)
    df_oos = df_full.iloc[-oos_size:]

    print(f"\nData splits:")
    print(f"  Original test set: {len(df_test)} rows (used in training)")
    print(f"  Out-of-sample: {len(df_oos)} rows (completely unseen)")

    # Load feature list
    feature_file = MODELS_DIR / "xgboost_long_entry_multitimeframe_features.txt"
    with open(feature_file, 'r') as f:
        feature_columns = [line.strip() for line in f.readlines()]

    print(f"\nFeatures: {len(feature_columns)}")

    # ========== LONG Entry Model ==========
    print("\n" + "=" * 80)
    print("LONG ENTRY MODEL VALIDATION")
    print("=" * 80)

    # Load model
    model_file = MODELS_DIR / "xgboost_long_entry_multitimeframe.pkl"
    with open(model_file, 'rb') as f:
        model_long = pickle.load(f)

    # Prepare test data
    df_test['target'] = create_target_long(df_test, lookahead=3, threshold=0.003)
    df_test_clean = df_test.dropna()
    X_test = df_test_clean[feature_columns].values
    y_test = df_test_clean['target'].values

    # Prepare OOS data
    df_oos['target'] = create_target_long(df_oos, lookahead=3, threshold=0.003)
    df_oos_clean = df_oos.dropna()
    X_oos = df_oos_clean[feature_columns].values
    y_oos = df_oos_clean['target'].values

    print(f"\nTarget distribution:")
    print(f"  Test set:")
    print(f"    Class 0: {(y_test == 0).sum()} ({(y_test == 0).sum()/len(y_test)*100:.1f}%)")
    print(f"    Class 1: {(y_test == 1).sum()} ({(y_test == 1).sum()/len(y_test)*100:.1f}%)")
    print(f"  Out-of-sample:")
    print(f"    Class 0: {(y_oos == 0).sum()} ({(y_oos == 0).sum()/len(y_oos)*100:.1f}%)")
    print(f"    Class 1: {(y_oos == 1).sum()} ({(y_oos == 1).sum()/len(y_oos)*100:.1f}%)")

    # Validate on test set (should match training report)
    metrics_test_long = validate_model(model_long, X_test, y_test, "LONG", "Test Set (Original)")

    # Validate on out-of-sample (THE CRITICAL TEST)
    metrics_oos_long = validate_model(model_long, X_oos, y_oos, "LONG", "Out-of-Sample (Unseen)")

    # ========== SHORT Entry Model ==========
    print("\n" + "=" * 80)
    print("SHORT ENTRY MODEL VALIDATION")
    print("=" * 80)

    # Load model
    model_file = MODELS_DIR / "xgboost_short_entry_multitimeframe.pkl"
    with open(model_file, 'rb') as f:
        model_short = pickle.load(f)

    # Prepare test data
    df_test['target'] = create_target_short(df_test, lookahead=3, threshold=0.003)
    df_test_clean = df_test.dropna()
    X_test = df_test_clean[feature_columns].values
    y_test = df_test_clean['target'].values

    # Prepare OOS data
    df_oos['target'] = create_target_short(df_oos, lookahead=3, threshold=0.003)
    df_oos_clean = df_oos.dropna()
    X_oos = df_oos_clean[feature_columns].values
    y_oos = df_oos_clean['target'].values

    print(f"\nTarget distribution:")
    print(f"  Test set:")
    print(f"    Class 0: {(y_test == 0).sum()} ({(y_test == 0).sum()/len(y_test)*100:.1f}%)")
    print(f"    Class 1: {(y_test == 1).sum()} ({(y_test == 1).sum()/len(y_test)*100:.1f}%)")
    print(f"  Out-of-sample:")
    print(f"    Class 0: {(y_oos == 0).sum()} ({(y_oos == 0).sum()/len(y_oos)*100:.1f}%)")
    print(f"    Class 1: {(y_oos == 1).sum()} ({(y_oos == 1).sum()/len(y_oos)*100:.1f}%)")

    # Validate on test set
    metrics_test_short = validate_model(model_short, X_test, y_test, "SHORT", "Test Set (Original)")

    # Validate on out-of-sample
    metrics_oos_short = validate_model(model_short, X_oos, y_oos, "SHORT", "Out-of-Sample (Unseen)")

    # ========== SUMMARY ==========
    print("\n" + "=" * 80)
    print("VALIDATION SUMMARY")
    print("=" * 80)

    print(f"\n{'Model':<15} {'Dataset':<20} {'F1 Score':<12} {'Change':<12} {'Verdict':<20}")
    print("-" * 80)

    # LONG
    test_long_f1 = metrics_test_long['f1']
    oos_long_f1 = metrics_oos_long['f1']
    change_long = oos_long_f1 - test_long_f1
    change_long_pct = (change_long / test_long_f1 * 100) if test_long_f1 > 0 else 0

    print(f"{'LONG Entry':<15} {'Test (Original)':<20} {test_long_f1:<12.4f} {'-':<12} {'Baseline':<20}")
    print(f"{'LONG Entry':<15} {'Out-of-Sample':<20} {oos_long_f1:<12.4f} {change_long_pct:>+11.1f}% ", end='')
    if oos_long_f1 >= 0.20:
        print("✅ PASS")
        verdict_long = "PASS"
    elif oos_long_f1 >= 0.158:
        print("⚠️ MARGINAL")
        verdict_long = "MARGINAL"
    else:
        print("❌ FAIL")
        verdict_long = "FAIL"

    # SHORT
    test_short_f1 = metrics_test_short['f1']
    oos_short_f1 = metrics_oos_short['f1']
    change_short = oos_short_f1 - test_short_f1
    change_short_pct = (change_short / test_short_f1 * 100) if test_short_f1 > 0 else 0

    print(f"{'SHORT Entry':<15} {'Test (Original)':<20} {test_short_f1:<12.4f} {'-':<12} {'Baseline':<20}")
    print(f"{'SHORT Entry':<15} {'Out-of-Sample':<20} {oos_short_f1:<12.4f} {change_short_pct:>+11.1f}% ", end='')
    if oos_short_f1 >= 0.20:
        print("✅ PASS")
        verdict_short = "PASS"
    elif oos_short_f1 >= 0.127:
        print("⚠️ MARGINAL")
        verdict_short = "MARGINAL"
    else:
        print("❌ FAIL")
        verdict_short = "FAIL"

    # Current model baseline
    print(f"\n{'Current Model':<15} {'Baseline':<20} {'0.1580 (LONG)':<12} {'-':<12} {'Proven (70.6% WR)':<20}")
    print(f"{'Current Model':<15} {'Baseline':<20} {'0.1270 (SHORT)':<12} {'-':<12} {'Proven (70.6% WR)':<20}")

    # Final verdict
    print("\n" + "=" * 80)
    print("FINAL VERDICT")
    print("=" * 80)

    if verdict_long == "PASS" and verdict_short == "PASS":
        print("\n✅ GATE 1 PASSED: Proceed to Cross-Validation")
        print(f"  LONG OOS F1: {oos_long_f1:.4f} >= 0.20 ✅")
        print(f"  SHORT OOS F1: {oos_short_f1:.4f} >= 0.20 ✅")
        print("\nNext step: python scripts/production/cross_validate_models.py")
        exit(0)

    elif verdict_long in ["PASS", "MARGINAL"] and verdict_short in ["PASS", "MARGINAL"]:
        print("\n⚠️ GATE 1 MARGINAL: Consider feature pruning")
        print(f"  LONG OOS F1: {oos_long_f1:.4f}")
        print(f"  SHORT OOS F1: {oos_short_f1:.4f}")
        print("\nRecommendation:")
        print("  1. Reduce features (69 → 30-40)")
        print("  2. Retrain with fewer features")
        print("  3. Re-run this validation")
        print("\nOr proceed with caution:")
        print("  python scripts/production/cross_validate_models.py")
        exit(1)

    else:
        print("\n❌ GATE 1 FAILED: Do NOT proceed")
        print(f"  LONG OOS F1: {oos_long_f1:.4f}")
        print(f"  SHORT OOS F1: {oos_short_f1:.4f}")
        print("\nAnalysis:")
        if oos_long_f1 < 0.158 or oos_short_f1 < 0.127:
            print("  ⚠️ Performance WORSE than current model")
            print("  → Severe overfitting detected")
            print("  → Current model is better")
        else:
            print("  ⚠️ Performance below target (20%)")
            print("  → Moderate overfitting")
            print("  → Feature reduction may help")

        print("\nRecommendation:")
        print("  Option 1: ABANDON this approach, keep current model")
        print("  Option 2: Drastically reduce features (69 → 20-30)")
        print("  Option 3: Try different feature selection strategy")
        print("\n⛔ DO NOT proceed to backtest without fixing this!")
        exit(2)
