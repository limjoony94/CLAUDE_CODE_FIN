"""
SHORT Model V3 - Improved with Stricter Criteria

개선사항:
1. Threshold 증가: 0.2% → 0.5% (더 명확한 하락만 학습)
2. Lookahead 증가: 3 → 9 candles (45분, 더 긴 예측)
3. Multi-timeframe features (5min + 15min aggregation)
4. Volatility-based features
5. Consecutive pattern features

목표: SHORT win rate 60%+ 달성
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from datetime import datetime
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import classification_report, confusion_matrix, f1_score, precision_score, recall_score
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


def create_short_labels_v3(df, lookahead=9, threshold=0.005):
    """
    V3: Stricter labels for clearer SHORT signals

    Changes from V2:
    - threshold: 0.002 → 0.005 (0.2% → 0.5%)
    - lookahead: 3 → 9 (15min → 45min)

    Goal: Learn only clear downward movements
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


def add_multiframe_features(df):
    """
    Multi-timeframe features

    Aggregate 5-min data to 15-min patterns
    """
    # Rolling 3-candle (15min) aggregations
    df['close_3_max'] = df['close'].rolling(3).max()
    df['close_3_min'] = df['close'].rolling(3).min()
    df['close_3_mean'] = df['close'].rolling(3).mean()
    df['volume_3_sum'] = df['volume'].rolling(3).sum()

    # 15-min momentum
    df['momentum_3'] = df['close'] - df['close'].shift(3)
    df['momentum_3_pct'] = df['momentum_3'] / df['close'].shift(3)

    # 15-min volatility
    df['volatility_3'] = df['close'].rolling(3).std()
    df['volatility_3_normalized'] = df['volatility_3'] / df['close']

    # Consecutive patterns
    df['consecutive_down'] = 0
    df['consecutive_up'] = 0

    for i in range(3, len(df)):
        # Count consecutive down candles
        down_count = 0
        for j in range(3):
            if df['close'].iloc[i-j] < df['close'].iloc[i-j-1]:
                down_count += 1
            else:
                break
        df.loc[df.index[i], 'consecutive_down'] = down_count

        # Count consecutive up candles
        up_count = 0
        for j in range(3):
            if df['close'].iloc[i-j] > df['close'].iloc[i-j-1]:
                up_count += 1
            else:
                break
        df.loc[df.index[i], 'consecutive_up'] = up_count

    # Price position in 15-min range
    df['price_position_3'] = (df['close'] - df['close_3_min']) / (df['close_3_max'] - df['close_3_min'])
    df['price_position_3'].fillna(0.5, inplace=True)

    # Volume trend
    df['volume_increasing'] = (df['volume'] > df['volume'].shift(1)).astype(int)
    df['volume_trend_3'] = df['volume_increasing'].rolling(3).sum()

    return df


def add_volatility_features(df):
    """
    Volatility-based features

    HIGH volatility = opportunity for SHORT
    """
    # ATR-based volatility
    df['volatility_ratio'] = df['atr'] / df['close']

    # Bollinger Band width
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['close']

    # Price range
    df['daily_range'] = (df['high'] - df['low']) / df['low']
    df['daily_range_ma'] = df['daily_range'].rolling(20).mean()
    df['range_ratio'] = df['daily_range'] / df['daily_range_ma']

    # Volatility spike (sudden increase)
    df['volatility_spike'] = (df['volatility_ratio'] > df['volatility_ratio'].rolling(10).mean() * 1.5).astype(int)

    return df


def add_pattern_strength_features(df):
    """
    Pattern strength indicators

    Combine multiple bearish signals
    """
    # Bearish signal count
    bearish_signals = [
        'bearish_div_rsi',
        'bearish_div_macd',
        'shooting_star',
        'evening_star',
        'bearish_engulfing',
        'dark_cloud_cover',
        'bb_upper_rejection',
        'overbought_reversal_signal'
    ]

    df['bearish_signal_count'] = 0
    for signal in bearish_signals:
        if signal in df.columns:
            df['bearish_signal_count'] += df[signal]

    # Strong bearish (2+ signals)
    df['strong_bearish'] = (df['bearish_signal_count'] >= 2).astype(int)

    # Weak bounce after drop
    df['weak_bounce_signal'] = 0
    for i in range(5, len(df)):
        recent_close = df['close'].iloc[i-5:i]
        if len(recent_close) > 0:
            drop = (recent_close.max() - recent_close.min()) / recent_close.max()
            bounce = (df['close'].iloc[i] - recent_close.min()) / recent_close.min()

            # Dropped >1% but bounced <0.3%
            if drop > 0.01 and bounce < 0.003:
                df.loc[df.index[i], 'weak_bounce_signal'] = 1

    return df


def load_and_prepare_data():
    """Load data and calculate ALL features"""
    print("="*80)
    print("Loading Data and Calculating V3 Improved Features")
    print("="*80)

    # Load data
    df = pd.read_csv(DATA_DIR / "BTCUSDT_5m_max.csv")
    print(f"Loaded {len(df)} candles")

    # Calculate baseline features
    df = calculate_features(df)

    # Calculate SHORT-specific features
    short_features = ShortSpecificFeatures(lookback=20)
    df = short_features.calculate_all_features(df)

    # NEW: Multi-timeframe features
    print("Adding multi-timeframe features...")
    df = add_multiframe_features(df)

    # NEW: Volatility features
    print("Adding volatility features...")
    df = add_volatility_features(df)

    # NEW: Pattern strength features
    print("Adding pattern strength features...")
    df = add_pattern_strength_features(df)

    # Calculate regime features
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

    # Combine ALL features
    all_features = (baseline_features + short_feature_names +
                   multiframe_features + volatility_features + pattern_features)

    feature_columns = [f for f in all_features if f in df.columns]

    print(f"\nTotal features: {len(feature_columns)}")
    print(f"  - Baseline: {len([f for f in baseline_features if f in df.columns])}")
    print(f"  - SHORT-specific: {len([f for f in short_feature_names if f in df.columns])}")
    print(f"  - Multi-timeframe: {len([f for f in multiframe_features if f in df.columns])}")
    print(f"  - Volatility: {len([f for f in volatility_features if f in df.columns])}")
    print(f"  - Pattern strength: {len([f for f in pattern_features if f in df.columns])}")

    # Create V3 labels (stricter)
    labels = create_short_labels_v3(df, lookahead=9, threshold=0.005)

    # Class distribution
    unique, counts = np.unique(labels, return_counts=True)
    print(f"\nLabel Distribution (V3 - Stricter):")
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


def train_and_backtest(X, y, df):
    """Train and backtest V3 model"""
    print("\n" + "="*80)
    print("Training and Backtesting SHORT Model V3")
    print("="*80)

    import xgboost as xgb

    tscv = TimeSeriesSplit(n_splits=5)

    all_trades = []
    fold_metrics = []

    # Model parameters
    params = {
        'n_estimators': 300,
        'max_depth': 5,  # Increased from 4
        'learning_rate': 0.01,
        'min_child_weight': 3,  # Decreased from 5 (allow more flexibility)
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'gamma': 0.5,  # Decreased from 1
        'reg_alpha': 1,
        'reg_lambda': 2,
        'random_state': 42,
        'use_label_encoder': False,
        'eval_metric': 'logloss'
    }

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Train model on fold
        model = xgb.XGBClassifier(**params)
        sample_weights = compute_sample_weight('balanced', y_train)
        model.fit(X_train, y_train, sample_weight=sample_weights, verbose=False)

        # Predict on validation
        y_pred = model.predict(X_val)
        y_prob = model.predict_proba(X_val)[:, 1]

        # Apply filters
        threshold = 0.65  # Lower threshold (more trades to evaluate)

        for i, idx in enumerate(val_idx):
            model_signal = y_prob[i] >= threshold
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
    print("V3 Model Results")
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
    final_model = xgb.XGBClassifier(**params)
    sample_weights = compute_sample_weight('balanced', y)
    final_model.fit(X, y, sample_weight=sample_weights, verbose=False)

    return overall_win_rate, all_trades, final_model


def main():
    """Main V3 training pipeline"""
    print("="*80)
    print("SHORT Model V3 - Improved Training")
    print("="*80)
    print("Improvements:")
    print("  1. Threshold: 0.2% → 0.5% (clearer signals)")
    print("  2. Lookahead: 3 → 9 candles (45min prediction)")
    print("  3. Multi-timeframe features (15min aggregation)")
    print("  4. Volatility features")
    print("  5. Pattern strength features")
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
        print(f"✅ V3 improvements achieved target!")

        # Save model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"short_model_v3_improved_{timestamp}"
        model_path = MODELS_DIR / f"{model_name}.pkl"

        with open(model_path, 'wb') as f:
            pickle.dump(model, f)

        print(f"\nModel saved: {model_path}")

    elif win_rate >= 0.55:
        print(f"⚠️ MARGINAL: SHORT win rate {win_rate*100:.1f}% (55-60%)")
        print(f"⚠️ Close to target, consider additional tuning")

    elif win_rate >= 0.40:
        print(f"🔄 IMPROVING: SHORT win rate {win_rate*100:.1f}% (40-55%)")
        print(f"🔄 Better than previous attempts (26%), continue improvements")

    else:
        print(f"❌ INSUFFICIENT: SHORT win rate {win_rate*100:.1f}%")
        print(f"❌ V3 improvements not enough")

    return win_rate, trades, model


if __name__ == "__main__":
    win_rate, trades, model = main()
