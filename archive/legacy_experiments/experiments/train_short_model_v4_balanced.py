"""
SHORT Model V4 - Balanced Approach with Feature Selection

V3 실패 원인:
- Too strict (4.2% SHORT labels)
- Too long lookahead (45min too hard)
- Too many features (61, overfitting)

V4 개선:
1. Threshold: 0.3% (중간값, ~6-7% labels 예상)
2. Lookahead: 6 candles (30min, 중간값)
3. Feature importance 기반 top features 선택
4. Ensemble: XGBoost + LightGBM voting

목표: SHORT win rate 60%+ 달성
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from datetime import datetime
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.ensemble import VotingClassifier

import sys
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.production.train_xgboost_improved_v3_phase2 import calculate_features
from scripts.production.short_specific_features import ShortSpecificFeatures
from scripts.production.market_regime_filter import MarketRegimeFilter
from scripts.experiments.train_short_model_v3_improved import (
    add_multiframe_features, add_volatility_features, add_pattern_strength_features
)

# Directories
DATA_DIR = PROJECT_ROOT / "data" / "historical"
MODELS_DIR = PROJECT_ROOT / "models"


def create_short_labels_v4(df, lookahead=6, threshold=0.003):
    """
    V4: Balanced labels

    threshold: 0.3% (between 0.2% and 0.5%)
    lookahead: 6 candles / 30min (between 15min and 45min)
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


def select_important_features(X, y, feature_columns, top_n=40):
    """
    Feature importance 기반 선택

    Train quick model and select top N features
    """
    print(f"\nSelecting top {top_n} features...")

    # Quick XGBoost to get feature importance
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.05,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )

    sample_weights = compute_sample_weight('balanced', y)
    model.fit(X, y, sample_weight=sample_weights, verbose=False)

    # Get feature importance
    importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'feature': feature_columns,
        'importance': importances
    }).sort_values('importance', ascending=False)

    # Select top N
    top_features = feature_importance_df.head(top_n)['feature'].tolist()

    print(f"\nTop {top_n} features selected:")
    for i, (feat, imp) in enumerate(zip(top_features[:10], feature_importance_df.head(10)['importance']), 1):
        print(f"  {i}. {feat}: {imp:.4f}")

    return top_features


def load_and_prepare_data():
    """Load data and calculate features"""
    print("="*80)
    print("Loading Data for V4 Balanced Model")
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

    # Create V4 labels (balanced)
    labels = create_short_labels_v4(df, lookahead=6, threshold=0.003)

    # Class distribution
    unique, counts = np.unique(labels, return_counts=True)
    print(f"\nLabel Distribution (V4 - Balanced):")
    print(f"  NO SHORT (0): {counts[0]} ({counts[0]/len(labels)*100:.1f}%)")
    print(f"  SHORT (1): {counts[1]} ({counts[1]/len(labels)*100:.1f}%)")

    # Features matrix
    X = df[feature_columns].values
    y = labels

    # Feature selection
    selected_features = select_important_features(X, y, feature_columns, top_n=40)

    # Recreate X with selected features only
    X_selected = df[selected_features].values

    print(f"\nFinal features: {len(selected_features)}")

    return X_selected, y, selected_features, df


def create_ensemble_model():
    """
    Ensemble: XGBoost + LightGBM

    두 모델의 투표로 최종 결정
    """
    # XGBoost
    xgb_model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.01,
        min_child_weight=3,
        subsample=0.8,
        colsample_bytree=0.8,
        gamma=0.5,
        reg_alpha=1,
        reg_lambda=2,
        random_state=42,
        use_label_encoder=False,
        eval_metric='logloss'
    )

    # LightGBM
    lgb_model = lgb.LGBMClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.01,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1,
        reg_lambda=2,
        random_state=42,
        verbose=-1
    )

    # Voting Classifier (soft voting - use probabilities)
    ensemble = VotingClassifier(
        estimators=[('xgb', xgb_model), ('lgb', lgb_model)],
        voting='soft'
    )

    return ensemble


def train_and_backtest(X, y, df):
    """Train and backtest V4 ensemble model"""
    print("\n" + "="*80)
    print("Training and Backtesting V4 Ensemble Model")
    print("="*80)
    print("Ensemble: XGBoost + LightGBM (Soft Voting)")

    tscv = TimeSeriesSplit(n_splits=5)

    all_trades = []
    fold_metrics = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Train ensemble on fold
        ensemble = create_ensemble_model()
        sample_weights = compute_sample_weight('balanced', y_train)
        ensemble.fit(X_train, y_train, sample_weight=sample_weights)

        # Predict on validation
        y_prob = ensemble.predict_proba(X_val)[:, 1]

        # Apply filters
        threshold = 0.65

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
    print("V4 Ensemble Results")
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

    # Regime analysis
    trades_df = pd.DataFrame(all_trades)
    print(f"\nTrades by Regime:")
    for regime in trades_df['regime'].unique():
        regime_trades = trades_df[trades_df['regime'] == regime]
        regime_correct = regime_trades['correct'].sum()
        regime_total = len(regime_trades)
        regime_wr = regime_correct / regime_total * 100 if regime_total > 0 else 0
        print(f"  {regime}: {regime_total} trades, {regime_correct} correct ({regime_wr:.1f}%)")

    # Train final model on all data
    final_ensemble = create_ensemble_model()
    sample_weights = compute_sample_weight('balanced', y)
    final_ensemble.fit(X, y, sample_weight=sample_weights)

    return overall_win_rate, all_trades, final_ensemble


def main():
    """Main V4 training pipeline"""
    print("="*80)
    print("SHORT Model V4 - Balanced Approach with Ensemble")
    print("="*80)
    print("Improvements:")
    print("  1. Threshold: 0.3% (balanced, ~6% labels)")
    print("  2. Lookahead: 6 candles (30min)")
    print("  3. Feature selection: Top 40 important features")
    print("  4. Ensemble: XGBoost + LightGBM (soft voting)")
    print("="*80)

    # Load data
    X, y, feature_columns, df = load_and_prepare_data()

    # Train and backtest
    win_rate, trades, model = train_and_backtest(X, y, df)

    # Decision
    print("\n" + "="*80)
    print("Final Decision")
    print("="*80)

    if win_rate >= 0.60:
        print(f"✅ SUCCESS! SHORT win rate {win_rate*100:.1f}% >= 60%")
        print(f"✅ V4 ensemble achieved target!")

        # Save model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"short_model_v4_ensemble_{timestamp}"
        model_path = MODELS_DIR / f"{model_name}.pkl"

        with open(model_path, 'wb') as f:
            pickle.dump(model, f)

        print(f"\nModel saved: {model_path}")

    elif win_rate >= 0.50:
        print(f"⚠️ APPROACHING: SHORT win rate {win_rate*100:.1f}% (50-60%)")
        print(f"⚠️ Getting closer! Continue fine-tuning")

    elif win_rate >= 0.35:
        print(f"🔄 IMPROVING: SHORT win rate {win_rate*100:.1f}% (35-50%)")
        print(f"🔄 Better than previous, continue improvements")

    else:
        print(f"❌ INSUFFICIENT: SHORT win rate {win_rate*100:.1f}%")

    print(f"\nProgress Summary:")
    print(f"  V2 (baseline): 26.0%")
    print(f"  V3 (strict): 9.7%")
    print(f"  V4 (balanced): {win_rate*100:.1f}%")

    return win_rate, trades, model


if __name__ == "__main__":
    win_rate, trades, model = main()
