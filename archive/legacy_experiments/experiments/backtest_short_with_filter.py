"""
SHORT Strategy Backtest with Market Regime Filter

통합 전략:
1. SHORT Model V2 (binary classification, 40 features)
2. Market Regime Filter (context-aware filtering)
3. High confidence threshold (0.7)

목표: SHORT win rate 60%+ 달성
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
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


def create_short_labels(df, lookahead=3, threshold=0.002):
    """Create binary SHORT labels"""
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


def load_and_prepare_data():
    """Load data and calculate all features"""
    print("="*80)
    print("Loading Data and Calculating Features")
    print("="*80)

    # Load data
    df = pd.read_csv(DATA_DIR / "BTCUSDT_5m_max.csv")
    print(f"Loaded {len(df)} candles")

    # Calculate baseline features
    df = calculate_features(df)

    # Calculate SHORT-specific features
    short_features = ShortSpecificFeatures(lookback=20)
    df = short_features.calculate_all_features(df)

    # Calculate regime features
    regime_filter = MarketRegimeFilter()
    df = regime_filter.get_regime_features(df)

    # Remove NaN
    df = df.ffill().dropna()
    print(f"After processing: {len(df)} candles")

    # Get feature columns
    baseline_features = ['rsi', 'macd', 'macd_signal', 'bb_upper', 'bb_lower',
                        'ema_21', 'ema_50', 'atr', 'volume_ratio', 'price_distance_ema']

    short_feature_names = short_features.get_feature_names()

    # Combine features (baseline + SHORT-specific)
    feature_columns = [f for f in baseline_features + short_feature_names if f in df.columns]

    print(f"\nTotal features: {len(feature_columns)}")

    # Create binary labels
    labels = create_short_labels(df, lookahead=3, threshold=0.002)

    # Class distribution
    unique, counts = np.unique(labels, return_counts=True)
    print(f"\nLabel Distribution:")
    print(f"  NO SHORT (0): {counts[0]} ({counts[0]/len(labels)*100:.1f}%)")
    print(f"  SHORT (1): {counts[1]} ({counts[1]/len(labels)*100:.1f}%)")

    # Regime distribution
    short_allowed_pct = df['short_allowed'].sum() / len(df) * 100
    print(f"\nRegime Filter:")
    print(f"  SHORT Allowed: {df['short_allowed'].sum()} / {len(df)} ({short_allowed_pct:.1f}%)")

    # Features matrix
    X = df[feature_columns].values
    y = labels

    return X, y, feature_columns, df


def backtest_with_filter(X, y, df, model_threshold=0.7):
    """
    Backtest SHORT model with regime filter

    Strategy:
    1. Model predicts SHORT with high confidence (>= threshold)
    2. Regime filter allows SHORT
    3. Both conditions met → Enter SHORT trade

    Returns:
        Win rate, trade statistics
    """
    print("\n" + "="*80)
    print("Backtesting SHORT Model with Regime Filter")
    print("="*80)
    print(f"Model Threshold: {model_threshold}")
    print(f"Regime Filter: Enabled")

    import xgboost as xgb

    tscv = TimeSeriesSplit(n_splits=5)

    all_trades = []
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Train model on fold
        params = {
            'n_estimators': 300,
            'max_depth': 4,
            'learning_rate': 0.01,
            'min_child_weight': 5,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'gamma': 1,
            'reg_alpha': 1,
            'reg_lambda': 2,
            'random_state': 42,
            'use_label_encoder': False,
            'eval_metric': 'logloss'
        }

        model = xgb.XGBClassifier(**params)
        sample_weights = compute_sample_weight('balanced', y_train)
        model.fit(X_train, y_train, sample_weight=sample_weights, verbose=False)

        # Predict on validation
        y_pred = model.predict(X_val)
        y_prob = model.predict_proba(X_val)[:, 1]

        # Apply regime filter
        for i, idx in enumerate(val_idx):
            model_signal = y_prob[i] >= model_threshold
            regime_allowed = df['short_allowed'].iloc[idx] == 1

            # Both conditions must be met
            if model_signal and regime_allowed:
                actual_label = y_val[i]

                # Record trade
                trade = {
                    'fold': fold,
                    'index': idx,
                    'probability': y_prob[i],
                    'regime': df['regime_trend'].iloc[idx],
                    'regime_momentum': df['regime_momentum'].iloc[idx],
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

    # Overall results
    print(f"\n{'='*80}")
    print("Backtest Results")
    print(f"{'='*80}")

    if len(all_trades) == 0:
        print("\n❌ No trades generated (no signals passed both filters)")
        print("   Consider:")
        print("     - Lowering model threshold")
        print("     - Relaxing regime filter")
        return 0.0, all_trades

    total_trades = len(all_trades)
    correct_trades = sum(t['correct'] for t in all_trades)
    overall_win_rate = correct_trades / total_trades

    print(f"\nOverall Results:")
    print(f"  Total Trades: {total_trades}")
    print(f"  Correct: {correct_trades}")
    print(f"  Incorrect: {total_trades - correct_trades}")
    print(f"  Win Rate: {overall_win_rate*100:.1f}%")

    print(f"\nFold-by-Fold Results:")
    for metrics in fold_metrics:
        print(f"  Fold {metrics['fold']}: {metrics['trades']} trades, "
              f"{metrics['correct']}/{metrics['trades']} correct ({metrics['win_rate']*100:.1f}%)")

    # Regime analysis
    trades_df = pd.DataFrame(all_trades)
    print(f"\nTrades by Regime:")
    for regime in trades_df['regime'].unique():
        regime_trades = trades_df[trades_df['regime'] == regime]
        regime_correct = regime_trades['correct'].sum()
        regime_total = len(regime_trades)
        regime_wr = regime_correct / regime_total * 100 if regime_total > 0 else 0

        print(f"  {regime}: {regime_total} trades, {regime_correct} correct ({regime_wr:.1f}%)")

    # Probability distribution
    probs = [t['probability'] for t in all_trades]
    print(f"\nProbability Distribution:")
    print(f"  Min: {min(probs):.3f}")
    print(f"  Mean: {np.mean(probs):.3f}")
    print(f"  Max: {max(probs):.3f}")

    return overall_win_rate, all_trades


def try_different_thresholds(X, y, df):
    """Try different model thresholds to find optimal"""
    print("\n" + "="*80)
    print("Testing Different Thresholds")
    print("="*80)

    thresholds = [0.6, 0.65, 0.7, 0.75, 0.8]
    results = []

    for threshold in thresholds:
        print(f"\n--- Testing threshold: {threshold} ---")
        win_rate, trades = backtest_with_filter(X, y, df, model_threshold=threshold)

        results.append({
            'threshold': threshold,
            'win_rate': win_rate,
            'trades': len(trades)
        })

    # Summary
    print("\n" + "="*80)
    print("Threshold Optimization Results")
    print("="*80)

    for result in results:
        print(f"Threshold {result['threshold']}: Win Rate {result['win_rate']*100:.1f}%, Trades: {result['trades']}")

    # Best threshold
    best = max(results, key=lambda x: x['win_rate'])
    print(f"\nBest Threshold: {best['threshold']}")
    print(f"  Win Rate: {best['win_rate']*100:.1f}%")
    print(f"  Trades: {best['trades']}")

    return results, best


def main():
    """Main backtesting pipeline"""
    print("="*80)
    print("SHORT Strategy with Regime Filter - Final Backtest")
    print("="*80)

    # Load data
    X, y, feature_columns, df = load_and_prepare_data()

    # Test with default threshold (0.7)
    print("\n" + "="*80)
    print("Testing Default Threshold (0.7)")
    print("="*80)

    win_rate, trades = backtest_with_filter(X, y, df, model_threshold=0.7)

    # Try different thresholds
    results, best = try_different_thresholds(X, y, df)

    # Final decision
    print("\n" + "="*80)
    print("Final Decision")
    print("="*80)

    if best['win_rate'] >= 0.60:
        print(f"✅ SUCCESS! Best SHORT win rate {best['win_rate']*100:.1f}% >= 60%")
        print(f"✅ Optimal threshold: {best['threshold']}")
        print(f"✅ Number of trades: {best['trades']}")
        print(f"\nRecommendation: Deploy SHORT strategy with regime filter")

    elif best['win_rate'] >= 0.55:
        print(f"⚠️ MARGINAL: Best SHORT win rate {best['win_rate']*100:.1f}% (55-60%)")
        print(f"⚠️ Optimal threshold: {best['threshold']}")
        print(f"⚠️ Number of trades: {best['trades']}")
        print(f"\nRecommendation: Deploy with caution, monitor closely")

    else:
        print(f"❌ INSUFFICIENT: Best SHORT win rate {best['win_rate']*100:.1f}% < 55%")
        print(f"❌ Tested thresholds: {[r['threshold'] for r in results]}")
        print(f"\nRecommendation: Further improvements needed")
        print(f"  Options:")
        print(f"    1. Different timeframe (15m, 1H)")
        print(f"    2. Additional SHORT-specific features")
        print(f"    3. Stricter regime filter")
        print(f"    4. Accept LONG-only strategy")

    return best


if __name__ == "__main__":
    best = main()
