"""
SHORT Model - Inverse Threshold Strategy (Approach #11)

비판적 통찰:
- 10가지 접근법 실패 → 잘못된 질문을 하고 있었음!
- Wrong: "언제 가격이 하락하는가?" (8.7% minority 예측)
- Right: "언제 가격이 안정/상승하는가?" (91.3% majority 예측)

핵심 아이디어:
- Majority class (91.3%) 예측은 쉬움
- 모델이 "안정/상승 아님"을 확신할 때 = SHORT 신호!
- Threshold inversion: prob <= 0.3 → SHORT

예상 결과: 50-60% win rate (Approach #1 inverse: 46% 이미 입증)
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from datetime import datetime
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_sample_weight

import sys
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.production.train_xgboost_improved_v3_phase2 import calculate_features
from scripts.production.short_specific_features import ShortSpecificFeatures
from scripts.production.market_regime_filter import MarketRegimeFilter

# Directories
DATA_DIR = PROJECT_ROOT / "data" / "historical"
MODELS_DIR = PROJECT_ROOT / "models"


def create_inverse_labels(df, lookahead=3, threshold=0.002):
    """
    Inverse Labeling Strategy

    Label 1 (91.3%): 가격이 안정적이거나 상승 (NOT falling)
    Label 0 (8.7%): 가격이 하락 (falling)

    모델은 Label 1을 예측하도록 학습 (쉬움!)
    SHORT 진입: model_prob <= 0.3 (모델이 "안정 아님"을 확신)
    """
    labels = []

    for i in range(len(df)):
        if i >= len(df) - lookahead:
            labels.append(1)  # Default: stable (majority)
            continue

        current_price = df['close'].iloc[i]
        future_prices = df['close'].iloc[i+1:i+1+lookahead]
        min_future = future_prices.min()
        decrease_pct = (current_price - min_future) / current_price

        if decrease_pct >= threshold:
            labels.append(0)  # FALLING (minority class)
        else:
            labels.append(1)  # STABLE/RISING (majority class)

    return np.array(labels)


def load_and_prepare_data():
    """Load data with all features"""
    print("="*80)
    print("Loading Data for Inverse Threshold Strategy")
    print("="*80)

    # Load candles
    df = pd.read_csv(DATA_DIR / "BTCUSDT_5m_max.csv")
    print(f"Loaded {len(df)} candles")

    # Calculate features
    df = calculate_features(df)

    short_features = ShortSpecificFeatures(lookback=20)
    df = short_features.calculate_all_features(df)

    regime_filter = MarketRegimeFilter()
    df = regime_filter.get_regime_features(df)

    # Remove NaN
    df = df.ffill().dropna()
    print(f"After processing: {len(df)} candles")

    # Feature selection (use V2 baseline features)
    baseline_features = ['rsi', 'macd', 'macd_signal', 'bb_upper', 'bb_lower',
                        'ema_21', 'ema_50', 'atr', 'volume_ratio', 'price_distance_ema']

    short_feature_names = short_features.get_feature_names()

    all_features = baseline_features + short_feature_names
    feature_columns = [f for f in all_features if f in df.columns]

    print(f"\nTotal features: {len(feature_columns)}")
    print(f"  - Baseline: {len([f for f in baseline_features if f in df.columns])}")
    print(f"  - SHORT-specific: {len([f for f in short_feature_names if f in df.columns])}")

    # Create INVERSE labels
    labels = create_inverse_labels(df, lookahead=3, threshold=0.002)

    # Class distribution
    unique, counts = np.unique(labels, return_counts=True)
    print(f"\nInverse Label Distribution:")
    print(f"  FALLING (0): {counts[0]} ({counts[0]/len(labels)*100:.1f}%)")
    print(f"  STABLE/RISING (1): {counts[1]} ({counts[1]/len(labels)*100:.1f}%)")
    print(f"\n✅ Model will learn to predict MAJORITY class (easier!)")

    X = df[feature_columns].values
    y = labels

    return X, y, feature_columns, df


def train_and_backtest(X, y, df):
    """Train and backtest with INVERSE threshold"""
    print("\n" + "="*80)
    print("Training XGBoost with Inverse Threshold Strategy")
    print("="*80)

    tscv = TimeSeriesSplit(n_splits=5)

    all_trades = []
    fold_metrics = []

    # Model parameters (same as V2 baseline)
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

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Train model
        model = xgb.XGBClassifier(**params)
        sample_weights = compute_sample_weight('balanced', y_train)
        model.fit(X_train, y_train, sample_weight=sample_weights, verbose=False)

        # Predict probabilities for class 1 (STABLE/RISING)
        y_prob = model.predict_proba(X_val)[:, 1]

        # INVERSE threshold: Enter SHORT when prob <= 0.3
        # (모델이 "STABLE/RISING 아님"을 확신 = FALLING 예상)
        inverse_threshold = 0.3

        for i, idx in enumerate(val_idx):
            # Inverse logic: LOW probability = SHORT signal
            model_signal = y_prob[i] <= inverse_threshold
            regime_allowed = df['short_allowed'].iloc[idx] == 1

            if model_signal and regime_allowed:
                # Get actual label (0 = FALLING, 1 = STABLE/RISING)
                actual_label = y_val[i]

                # SUCCESS: We predicted FALLING (0) and it actually fell
                correct = (actual_label == 0)

                trade = {
                    'fold': fold,
                    'index': idx,
                    'probability': y_prob[i],  # Low prob = SHORT signal
                    'regime': df['regime_trend'].iloc[idx],
                    'predicted': 0,  # We predict FALLING
                    'actual': actual_label,
                    'correct': correct
                }
                all_trades.append(trade)

        # Fold metrics
        fold_trades = [t for t in all_trades if t['fold'] == fold]
        if len(fold_trades) > 0:
            fold_correct = sum(t['correct'] for t in fold_trades)
            fold_total = len(fold_trades)
            fold_win_rate = fold_correct / fold_total

            fold_metrics.append({
                'fold': fold,
                'trades': fold_total,
                'correct': fold_correct,
                'win_rate': fold_win_rate
            })

            print(f"Fold {fold}: {fold_total} trades, {fold_correct} correct ({fold_win_rate*100:.1f}%)")

    # Overall results
    print(f"\n{'='*80}")
    print("Inverse Threshold Strategy Results")
    print(f"{'='*80}")

    if len(all_trades) == 0:
        print("\n❌ No trades generated")
        return 0.0, all_trades, None

    total_trades = len(all_trades)
    correct_trades = sum(t['correct'] for t in all_trades)
    overall_win_rate = correct_trades / total_trades

    print(f"\nOverall Results:")
    print(f"  Total Trades: {total_trades}")
    print(f"  Correct: {correct_trades}")
    print(f"  Incorrect: {total_trades - correct_trades}")
    print(f"  Win Rate: {overall_win_rate*100:.1f}%")

    # Probability distribution analysis
    probs = [t['probability'] for t in all_trades]
    print(f"\nProbability Distribution (LOW = SHORT signal):")
    print(f"  Mean: {np.mean(probs):.3f}")
    print(f"  Std: {np.std(probs):.3f}")
    print(f"  Min: {np.min(probs):.3f}")
    print(f"  Max: {np.max(probs):.3f}")

    # Regime analysis
    trades_df = pd.DataFrame(all_trades)
    print(f"\nTrades by Regime:")
    for regime in trades_df['regime'].unique():
        regime_trades = trades_df[trades_df['regime'] == regime]
        regime_correct = regime_trades['correct'].sum()
        regime_total = len(regime_trades)
        regime_wr = regime_correct / regime_total * 100 if regime_total > 0 else 0
        print(f"  {regime}: {regime_total} trades, {regime_correct} correct ({regime_wr:.1f}%)")

    # Train final model
    final_model = xgb.XGBClassifier(**params)
    sample_weights = compute_sample_weight('balanced', y)
    final_model.fit(X, y, sample_weight=sample_weights, verbose=False)

    return overall_win_rate, all_trades, final_model


def main():
    """Main pipeline"""
    print("="*80)
    print("SHORT Model - Inverse Threshold Strategy (Approach #11)")
    print("="*80)
    print("Critical Insight:")
    print("  - 10 approaches failed by asking wrong question")
    print("  - Wrong: 'When will price FALL?' (8.7% minority)")
    print("  - Right: 'When will price be STABLE/RISING?' (91.3% majority)")
    print("  - SHORT signal: Model says 'NOT stable' (prob <= 0.3)")
    print("="*80)

    # Load data
    X, y, feature_columns, df = load_and_prepare_data()

    # Train and backtest
    win_rate, trades, model = train_and_backtest(X, y, df)

    # Decision
    print("\n" + "="*80)
    print("Final Decision - Inverse Threshold Strategy")
    print("="*80)

    if win_rate >= 0.60:
        print(f"✅ SUCCESS! SHORT win rate {win_rate*100:.1f}% >= 60%")
        print(f"✅ Inverse threshold strategy WORKS!")

        # Save model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"short_inverse_threshold_{timestamp}"
        model_path = MODELS_DIR / f"{model_name}.pkl"

        with open(model_path, 'wb') as f:
            pickle.dump(model, f)

        # Save features
        feature_path = MODELS_DIR / f"{model_name}_features.txt"
        with open(feature_path, 'w') as f:
            for feature in feature_columns:
                f.write(f"{feature}\n")

        print(f"\nModel saved: {model_path}")

    elif win_rate >= 0.45:
        print(f"🔄 MAJOR IMPROVEMENT: SHORT win rate {win_rate*100:.1f}% (45-60%)")
        print(f"🔄 Inverse threshold showing strong promise!")
        print(f"🔄 Consider threshold tuning (try 0.2, 0.25, 0.35)")

    elif win_rate >= 0.30:
        print(f"⚠️ SOME IMPROVEMENT: SHORT win rate {win_rate*100:.1f}% (30-45%)")
        print(f"⚠️ Better than baseline but still insufficient")

    else:
        print(f"❌ INSUFFICIENT: SHORT win rate {win_rate*100:.1f}%")
        print(f"❌ Inverse threshold also cannot solve the problem")

    print(f"\nComplete Progress Summary:")
    print(f"  V2 (baseline): 26.0%")
    print(f"  Approach #1 (label inverse): 46.0%")
    print(f"  LSTM: 17.3%")
    print(f"  Funding rate: 22.4%")
    print(f"  Inverse Threshold: {win_rate*100:.1f}%")

    return win_rate, trades, model


if __name__ == "__main__":
    win_rate, trades, model = main()
