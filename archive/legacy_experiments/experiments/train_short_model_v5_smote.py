"""
SHORT Model V5 - SMOTE + Advanced Tuning

근본 문제: Class Imbalance (91% vs 9%)

V5 해결책:
1. SMOTE로 SHORT 샘플 증강 (9% → 40%)
2. V2 best settings 기반 (threshold 0.2%, lookahead 3)
3. Aggressive hyperparameter tuning
4. Stratified cross-validation

목표: SHORT win rate 60%+ 달성
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from datetime import datetime
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit, StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_sample_weight
from imblearn.over_sampling import SMOTE

import sys
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.production.train_xgboost_improved_v3_phase2 import calculate_features
from scripts.production.short_specific_features import ShortSpecificFeatures
from scripts.production.market_regime_filter import MarketRegimeFilter
from scripts.experiments.train_short_model_v3_improved import (
    add_multiframe_features, add_volatility_features, add_pattern_strength_features
)
from scripts.experiments.train_short_model_v4_balanced import select_important_features

# Directories
DATA_DIR = PROJECT_ROOT / "data" / "historical"
MODELS_DIR = PROJECT_ROOT / "models"


def create_short_labels_v5(df, lookahead=3, threshold=0.002):
    """
    V5: Back to V2 settings (best so far)

    threshold: 0.2%
    lookahead: 3 candles (15min)
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
            labels.append(1)  # SHORT
        else:
            labels.append(0)  # NO SHORT

    return np.array(labels)


def apply_smote(X, y, sampling_strategy=0.4):
    """
    Apply SMOTE to balance classes

    sampling_strategy: 0.4 = SHORT samples will be 40% of NO SHORT
    (less aggressive than 1.0 to avoid overfitting)
    """
    print(f"\nApplying SMOTE (sampling_strategy={sampling_strategy})...")

    # Original distribution
    unique, counts = np.unique(y, return_counts=True)
    print(f"Before SMOTE:")
    print(f"  NO SHORT: {counts[0]} ({counts[0]/len(y)*100:.1f}%)")
    print(f"  SHORT: {counts[1]} ({counts[1]/len(y)*100:.1f}%)")

    # Apply SMOTE
    smote = SMOTE(sampling_strategy=sampling_strategy, random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    # New distribution
    unique, counts = np.unique(y_resampled, return_counts=True)
    print(f"\nAfter SMOTE:")
    print(f"  NO SHORT: {counts[0]} ({counts[0]/len(y_resampled)*100:.1f}%)")
    print(f"  SHORT: {counts[1]} ({counts[1]/len(y_resampled)*100:.1f}%)")
    print(f"  Total samples: {len(y)} → {len(y_resampled)} (+{len(y_resampled) - len(y)})")

    return X_resampled, y_resampled


def load_and_prepare_data():
    """Load data and calculate features"""
    print("="*80)
    print("Loading Data for V5 SMOTE Model")
    print("="*80)

    # Load data
    df = pd.read_csv(DATA_DIR / "BTCUSDT_5m_max.csv")
    print(f"Loaded {len(df)} candles")

    # Calculate all features
    df = calculate_features(df)

    short_features = ShortSpecificFeatures(lookback=20)
    df = short_features.calculate_all_features(df)

    df = add_multiframe_features(df)
    df = add_volatility_features(df)
    df = add_pattern_strength_features(df)

    regime_filter = MarketRegimeFilter()
    df = regime_filter.get_regime_features(df)

    # Remove NaN
    df = df.ffill().dropna()
    print(f"After processing: {len(df)} candles")

    # Get ALL feature columns
    baseline_features = ['rsi', 'macd', 'macd_signal', 'bb_upper', 'bb_lower',
                        'ema_21', 'ema_50', 'atr', 'volume_ratio', 'price_distance_ema']

    short_feature_names = short_features.get_feature_names()

    multiframe_features = [
        'close_3_max', 'close_3_min', 'close_3_mean', 'volume_3_sum',
        'momentum_3', 'momentum_3_pct', 'volatility_3', 'volatility_3_normalized',
        'consecutive_down', 'consecutive_up', 'price_position_3',
        'volume_increasing', 'volume_trend_3'
    ]

    volatility_features = [
        'volatility_ratio', 'bb_width', 'daily_range', 'daily_range_ma',
        'range_ratio', 'volatility_spike'
    ]

    pattern_features = [
        'bearish_signal_count', 'strong_bearish', 'weak_bounce_signal'
    ]

    all_features = (baseline_features + short_feature_names +
                   multiframe_features + volatility_features + pattern_features)

    feature_columns = [f for f in all_features if f in df.columns]

    print(f"\nInitial features: {len(feature_columns)}")

    # Create V5 labels (V2 settings)
    labels = create_short_labels_v5(df, lookahead=3, threshold=0.002)

    # Class distribution
    unique, counts = np.unique(labels, return_counts=True)
    print(f"\nOriginal Label Distribution:")
    print(f"  NO SHORT (0): {counts[0]} ({counts[0]/len(labels)*100:.1f}%)")
    print(f"  SHORT (1): {counts[1]} ({counts[1]/len(labels)*100:.1f}%)")

    # Features matrix
    X = df[feature_columns].values
    y = labels

    # Feature selection (use top 40)
    selected_features = select_important_features(X, y, feature_columns, top_n=40)
    X_selected = df[selected_features].values

    # Apply SMOTE
    X_smote, y_smote = apply_smote(X_selected, y, sampling_strategy=0.4)

    print(f"\nFinal features: {len(selected_features)}")

    return X_smote, y_smote, X_selected, y, selected_features, df


def train_and_backtest_smote(X_smote, y_smote, X_original, y_original, df):
    """
    Train on SMOTE data, test on ORIGINAL data

    Important: Test on original to avoid overfitting to synthetic samples
    """
    print("\n" + "="*80)
    print("Training V5 Model with SMOTE")
    print("="*80)

    # Train on SMOTE data
    params = {
        'n_estimators': 500,  # Increased
        'max_depth': 6,  # Increased
        'learning_rate': 0.005,  # Decreased (more careful)
        'min_child_weight': 2,  # Decreased (more flexible)
        'subsample': 0.8,
        'colsample_bytree': 0.7,  # Decreased (reduce overfitting)
        'gamma': 0.3,  # Decreased
        'reg_alpha': 2,  # Increased regularization
        'reg_lambda': 3,  # Increased regularization
        'random_state': 42,
        'use_label_encoder': False,
        'eval_metric': 'logloss',
        'scale_pos_weight': 1  # SMOTE already balanced
    }

    model = xgb.XGBClassifier(**params)

    print(f"\nTraining on SMOTE data: {len(y_smote)} samples")
    model.fit(X_smote, y_smote, verbose=False)

    print(f"Testing on ORIGINAL data: {len(y_original)} samples")

    # Backtest on ORIGINAL data with time series split
    tscv = TimeSeriesSplit(n_splits=5)

    all_trades = []
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_original)):
        X_train, X_val = X_original[train_idx], X_original[val_idx]
        y_train, y_val = y_original[train_idx], y_original[val_idx]

        # Apply SMOTE to training fold only
        X_train_smote, y_train_smote = apply_smote(X_train, y_train, sampling_strategy=0.4)

        # Train on SMOTE training data
        fold_model = xgb.XGBClassifier(**params)
        fold_model.fit(X_train_smote, y_train_smote, verbose=False)

        # Predict on ORIGINAL validation data
        y_prob = fold_model.predict_proba(X_val)[:, 1]

        # Apply filters
        threshold = 0.70

        for i, idx in enumerate(val_idx):
            model_signal = y_prob[i] >= threshold
            regime_allowed = df['short_allowed'].iloc[idx] == 1

            if model_signal and regime_allowed:
                actual_label = y_val[i]

                trade = {
                    'fold': fold,
                    'index': idx,
                    'probability': y_prob[i],
                    'regime': df['regime_trend'].iloc[idx],
                    'predicted': 1,
                    'actual': actual_label,
                    'correct': (1 == actual_label)
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
    print("V5 SMOTE Model Results (tested on ORIGINAL data)")
    print(f"{'='*80}")

    if len(all_trades) == 0:
        print("\n❌ No trades generated")
        return 0.0, all_trades, model

    total_trades = len(all_trades)
    correct_trades = sum(t['correct'] for t in all_trades)
    overall_win_rate = correct_trades / total_trades

    print(f"\nOverall Results:")
    print(f"  Total Trades: {total_trades}")
    print(f"  Correct: {correct_trades}")
    print(f"  Incorrect: {total_trades - correct_trades}")
    print(f"  Win Rate: {overall_win_rate*100:.1f}%")

    # Regime analysis
    if len(all_trades) > 0:
        trades_df = pd.DataFrame(all_trades)
        print(f"\nTrades by Regime:")
        for regime in trades_df['regime'].unique():
            regime_trades = trades_df[trades_df['regime'] == regime]
            regime_correct = regime_trades['correct'].sum()
            regime_total = len(regime_trades)
            regime_wr = regime_correct / regime_total * 100 if regime_total > 0 else 0
            print(f"  {regime}: {regime_total} trades, {regime_correct} correct ({regime_wr:.1f}%)")

    return overall_win_rate, all_trades, model


def main():
    """Main V5 training pipeline"""
    print("="*80)
    print("SHORT Model V5 - SMOTE + Advanced Tuning")
    print("="*80)
    print("Final Attempt - Addressing Root Cause:")
    print("  1. SMOTE: Synthetic samples to balance classes")
    print("  2. V2 settings: Best threshold (0.2%) + lookahead (3)")
    print("  3. Feature selection: Top 40 features")
    print("  4. Advanced hyperparameters: Deeper, more regularized")
    print("  5. Test on ORIGINAL data (avoid synthetic overfitting)")
    print("="*80)

    # Load data
    X_smote, y_smote, X_original, y_original, feature_columns, df = load_and_prepare_data()

    # Train and backtest
    win_rate, trades, model = train_and_backtest_smote(X_smote, y_smote, X_original, y_original, df)

    # Decision
    print("\n" + "="*80)
    print("Final Decision - V5 SMOTE Model")
    print("="*80)

    if win_rate >= 0.60:
        print(f"✅ SUCCESS! SHORT win rate {win_rate*100:.1f}% >= 60%")
        print(f"✅ SMOTE solved the class imbalance problem!")

        # Save model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"short_model_v5_smote_{timestamp}"
        model_path = MODELS_DIR / f"{model_name}.pkl"

        with open(model_path, 'wb') as f:
            pickle.dump(model, f)

        print(f"\nModel saved: {model_path}")

    elif win_rate >= 0.50:
        print(f"⚠️ APPROACHING: SHORT win rate {win_rate*100:.1f}% (50-60%)")
        print(f"⚠️ SMOTE helped! Getting very close to target")

    elif win_rate >= 0.35:
        print(f"🔄 IMPROVING: SHORT win rate {win_rate*100:.1f}% (35-50%)")
        print(f"🔄 SMOTE provided improvement")

    else:
        print(f"❌ INSUFFICIENT: SHORT win rate {win_rate*100:.1f}%")
        print(f"❌ Even SMOTE cannot overcome fundamental limitations")

    print(f"\nComplete Progress Summary:")
    print(f"  V2 (baseline): 26.0%")
    print(f"  V3 (strict criteria): 9.7%")
    print(f"  V4 (balanced + ensemble): 20.3%")
    print(f"  V5 (SMOTE + tuning): {win_rate*100:.1f}%")

    return win_rate, trades, model


if __name__ == "__main__":
    win_rate, trades, model = main()
