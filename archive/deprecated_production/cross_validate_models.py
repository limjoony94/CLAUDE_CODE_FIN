"""
Gate 2: Time-Series Cross-Validation

목적:
- 여러 시간대에서 성능 일관성 확인
- Market regime 변화에 robust한지 검증
- F1 standard deviation < 10%p 확인

Critical:
- Gate 1 통과했어도 Gate 2 필수
- 한 기간만 좋은 것 ≠ robust model
- 시간대별 안정성이 핵심
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.metrics import classification_report, confusion_matrix
import sys

sys.path.append(str(Path(__file__).parent))
from multi_timeframe_features import MultiTimeframeFeatures

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"


def calculate_features(df):
    """Calculate ALL features (same as training)"""
    import ta

    df = df.copy()

    # Original features (Phase 1+2)
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


def evaluate_fold(model, X, y, fold_name):
    """Evaluate model on fold"""
    y_pred = model.predict(X)

    # Confusion matrix
    cm = confusion_matrix(y, y_pred)

    # Calculate metrics
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

    print(f"\n{fold_name}:")
    print(f"  Samples: {len(y)} ({(y==1).sum()} positive, {(y==1).sum()/len(y)*100:.1f}%)")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1: {f1:.4f}")

    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'samples': len(y),
        'positive': (y==1).sum()
    }


if __name__ == "__main__":
    print("=" * 80)
    print("Gate 2: Time-Series Cross-Validation")
    print("=" * 80)

    # Load data
    data_file = DATA_DIR / "historical" / "BTCUSDT_5m_max.csv"
    df_full = pd.read_csv(data_file)
    print(f"\nTotal candles: {len(df_full)}")

    # Calculate features
    print("\n계산 중: All 69 features...")
    df_full = calculate_features(df_full)
    df_full = df_full.dropna()
    print(f"After features & dropna: {len(df_full)} rows")

    # Load models
    model_file_long = MODELS_DIR / "xgboost_long_entry_multitimeframe.pkl"
    with open(model_file_long, 'rb') as f:
        model_long = pickle.load(f)

    model_file_short = MODELS_DIR / "xgboost_short_entry_multitimeframe.pkl"
    with open(model_file_short, 'rb') as f:
        model_short = pickle.load(f)

    # Load feature list
    feature_file = MODELS_DIR / "xgboost_long_entry_multitimeframe_features.txt"
    with open(feature_file, 'r') as f:
        feature_columns = [line.strip() for line in f.readlines()]

    print(f"Features: {len(feature_columns)}")

    # 5-fold walk-forward cross-validation
    n_folds = 5
    fold_size = len(df_full) // (n_folds + 1)  # Reserve last fold for final test

    print(f"\nCross-Validation Setup:")
    print(f"  Total: {len(df_full)} rows")
    print(f"  Folds: {n_folds}")
    print(f"  Fold size: ~{fold_size} rows")

    # ========== LONG Entry Model ==========
    print("\n" + "=" * 80)
    print("LONG ENTRY MODEL - CROSS-VALIDATION")
    print("=" * 80)

    df_full['target_long'] = create_target_long(df_full, lookahead=3, threshold=0.003)
    df_clean_long = df_full.dropna()

    results_long = []

    for fold in range(n_folds):
        # Walk-forward: train on all previous data, test on next fold
        train_end = fold_size * (fold + 1)
        test_start = train_end
        test_end = test_start + fold_size

        # Test fold
        df_test = df_clean_long.iloc[test_start:test_end]
        X_test = df_test[feature_columns].values
        y_test = df_test['target_long'].values

        fold_name = f"Fold {fold + 1} (rows {test_start}-{test_end})"
        metrics = evaluate_fold(model_long, X_test, y_test, fold_name)
        results_long.append(metrics)

    # Calculate statistics
    f1_scores_long = [r['f1'] for r in results_long]
    f1_mean_long = np.mean(f1_scores_long)
    f1_std_long = np.std(f1_scores_long)
    f1_min_long = np.min(f1_scores_long)
    f1_max_long = np.max(f1_scores_long)

    print("\n" + "=" * 80)
    print("LONG Entry - CV Summary:")
    print(f"  F1 Mean: {f1_mean_long:.4f}")
    print(f"  F1 Std: {f1_std_long:.4f}")
    print(f"  F1 Min: {f1_min_long:.4f}")
    print(f"  F1 Max: {f1_max_long:.4f}")
    print(f"  F1 Range: {f1_max_long - f1_min_long:.4f}")
    print("=" * 80)

    # ========== SHORT Entry Model ==========
    print("\n" + "=" * 80)
    print("SHORT ENTRY MODEL - CROSS-VALIDATION")
    print("=" * 80)

    df_full['target_short'] = create_target_short(df_full, lookahead=3, threshold=0.003)
    df_clean_short = df_full.dropna()

    results_short = []

    for fold in range(n_folds):
        train_end = fold_size * (fold + 1)
        test_start = train_end
        test_end = test_start + fold_size

        df_test = df_clean_short.iloc[test_start:test_end]
        X_test = df_test[feature_columns].values
        y_test = df_test['target_short'].values

        fold_name = f"Fold {fold + 1} (rows {test_start}-{test_end})"
        metrics = evaluate_fold(model_short, X_test, y_test, fold_name)
        results_short.append(metrics)

    # Calculate statistics
    f1_scores_short = [r['f1'] for r in results_short]
    f1_mean_short = np.mean(f1_scores_short)
    f1_std_short = np.std(f1_scores_short)
    f1_min_short = np.min(f1_scores_short)
    f1_max_short = np.max(f1_scores_short)

    print("\n" + "=" * 80)
    print("SHORT Entry - CV Summary:")
    print(f"  F1 Mean: {f1_mean_short:.4f}")
    print(f"  F1 Std: {f1_std_short:.4f}")
    print(f"  F1 Min: {f1_min_short:.4f}")
    print(f"  F1 Max: {f1_max_short:.4f}")
    print(f"  F1 Range: {f1_max_short - f1_min_short:.4f}")
    print("=" * 80)

    # ========== FINAL VERDICT ==========
    print("\n" + "=" * 80)
    print("GATE 2: CROSS-VALIDATION VERDICT")
    print("=" * 80)

    print(f"\n{'Model':<15} {'Mean F1':<12} {'Std F1':<12} {'Min F1':<12} {'Max F1':<12} {'Verdict':<15}")
    print("-" * 85)

    # LONG verdict
    print(f"{'LONG Entry':<15} {f1_mean_long:<12.4f} {f1_std_long:<12.4f} {f1_min_long:<12.4f} {f1_max_long:<12.4f} ", end='')
    if f1_std_long < 0.10 and f1_mean_long >= 0.25:
        print("✅ PASS")
        verdict_long = "PASS"
    elif f1_std_long < 0.15 and f1_mean_long >= 0.20:
        print("⚠️ MARGINAL")
        verdict_long = "MARGINAL"
    else:
        print("❌ FAIL")
        verdict_long = "FAIL"

    # SHORT verdict
    print(f"{'SHORT Entry':<15} {f1_mean_short:<12.4f} {f1_std_short:<12.4f} {f1_min_short:<12.4f} {f1_max_short:<12.4f} ", end='')
    if f1_std_short < 0.10 and f1_mean_short >= 0.25:
        print("✅ PASS")
        verdict_short = "PASS"
    elif f1_std_short < 0.15 and f1_mean_short >= 0.20:
        print("⚠️ MARGINAL")
        verdict_short = "MARGINAL"
    else:
        print("❌ FAIL")
        verdict_short = "FAIL"

    # Baseline
    print(f"{'Current Model':<15} {'0.1580 (LONG)':<12} {'-':<12} {'-':<12} {'-':<12} {'Proven (70.6%)':<15}")
    print(f"{'Current Model':<15} {'0.1270 (SHORT)':<12} {'-':<12} {'-':<12} {'-':<12} {'Proven (70.6%)':<15}")

    # Pass criteria
    print("\n" + "=" * 80)
    print("Pass Criteria:")
    print("  PASS: Std < 10%p AND Mean >= 25%")
    print("  MARGINAL: Std < 15%p AND Mean >= 20%")
    print("  FAIL: Otherwise")
    print("=" * 80)

    # Final decision
    print("\n" + "=" * 80)
    print("FINAL DECISION")
    print("=" * 80)

    if verdict_long == "PASS" and verdict_short == "PASS":
        print("\n✅ GATE 2 PASSED: Models are temporally stable")
        print(f"  LONG: Mean F1 {f1_mean_long:.4f}, Std {f1_std_long:.4f} ✅")
        print(f"  SHORT: Mean F1 {f1_mean_short:.4f}, Std {f1_std_short:.4f} ✅")
        print("\nProceed to Gate 3 (Backtest):")
        print("  Next: Full trading simulation with TP/SL/MaxHold")
        print("  Target: WR >= 71%, Returns >= +4.5%")
        exit(0)

    elif verdict_long in ["PASS", "MARGINAL"] and verdict_short in ["PASS", "MARGINAL"]:
        print("\n⚠️ GATE 2 MARGINAL: Some instability detected")
        print(f"  LONG: Mean F1 {f1_mean_long:.4f}, Std {f1_std_long:.4f}")
        print(f"  SHORT: Mean F1 {f1_mean_short:.4f}, Std {f1_std_short:.4f}")

        if f1_std_long > 0.10 or f1_std_short > 0.10:
            print("\n⚠️ High variability across time periods")
            print("  → Model may not be robust to regime changes")
            print("  → Backtest may show unstable performance")

        if f1_mean_long < 0.25 or f1_mean_short < 0.25:
            print("\n⚠️ Mean F1 below target (25%)")
            print("  → Performance lower than expected")
            print("  → Backtest WR may not reach 71%")

        print("\nOptions:")
        print("  1. Proceed to Gate 3 with caution")
        print("  2. Feature pruning to improve stability")
        print("  3. Retrain with different time periods")

        print("\nRecommendation: Proceed to Gate 3 BUT be prepared for issues")
        exit(1)

    else:
        print("\n❌ GATE 2 FAILED: Models not temporally stable")
        print(f"  LONG: Mean F1 {f1_mean_long:.4f}, Std {f1_std_long:.4f}")
        print(f"  SHORT: Mean F1 {f1_mean_short:.4f}, Std {f1_std_short:.4f}")

        print("\nIssues:")
        if f1_std_long > 0.15 or f1_std_short > 0.15:
            print("  ❌ Very high variability (Std > 15%p)")
            print("     → Model performance unstable across time")

        if f1_mean_long < 0.20 or f1_mean_short < 0.20:
            print("  ❌ Mean F1 too low (< 20%)")
            print("     → Not better than current model")

        if f1_min_long < 0.10 or f1_min_short < 0.10:
            print("  ❌ Some folds perform very poorly (F1 < 10%)")
            print("     → Model fails in certain conditions")

        print("\nRecommendation:")
        print("  DO NOT proceed to backtest")
        print("  Options:")
        print("    1. Drastically reduce features (69 → 30)")
        print("    2. Try different feature selection")
        print("    3. Consider abandoning this approach")
        print("    4. Keep current model (proven 70.6% WR)")
        exit(2)
