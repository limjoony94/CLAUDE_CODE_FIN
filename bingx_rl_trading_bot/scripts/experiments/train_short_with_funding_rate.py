"""
SHORT Model - WITH FUNDING RATE

비판적 통찰:
- 9가지 접근법 모두 실패 → 데이터 문제
- Missing: Market sentiment
- Solution: Funding Rate!

Funding Rate:
- Positive: Longs pay Shorts (bullish sentiment)
- Negative: Shorts pay Longs (bearish sentiment)
- HIGH positive funding + price rejection = SHORT opportunity!

목표: SHORT win rate 60%+ 달성
"""

import pandas as pd
import numpy as np
import pickle
import requests
from pathlib import Path
from datetime import datetime, timedelta
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


def get_funding_rate_history():
    """
    Fetch historical funding rate from BingX API

    Returns:
        DataFrame with timestamp and funding_rate
    """
    print("Fetching funding rate history from BingX...")

    url = 'https://open-api.bingx.com/openApi/swap/v2/quote/fundingRate'
    params = {
        'symbol': 'BTC-USDT',
        'limit': 1000  # Maximum available
    }

    try:
        response = requests.get(url, params=params)
        data = response.json()

        if data.get('code') == 0:
            funding_data = data.get('data', [])
            print(f"✅ Retrieved {len(funding_data)} funding rate records")

            # Convert to DataFrame
            df = pd.DataFrame(funding_data)
            df['timestamp'] = pd.to_datetime(df['fundingTime'], unit='ms')
            df['funding_rate'] = df['fundingRate'].astype(float)

            # Sort by time
            df = df.sort_values('timestamp').reset_index(drop=True)

            print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
            print(f"Funding rate range: {df['funding_rate'].min():.6f} to {df['funding_rate'].max():.6f}")

            return df[['timestamp', 'funding_rate']]
        else:
            print(f"❌ API error: {data}")
            return None

    except Exception as e:
        print(f"❌ Error fetching funding rate: {e}")
        return None


def merge_funding_rate_with_candles(candles_df, funding_df):
    """
    Merge funding rate with 5-min candles

    Strategy: Forward fill (funding rate applies until next update)
    """
    print("\nMerging funding rate with candle data...")

    # Ensure timestamp columns
    if 'timestamp' not in candles_df.columns:
        # Assume index or create from first column
        if 'open_time' in candles_df.columns:
            candles_df['timestamp'] = pd.to_datetime(candles_df['open_time'])
        else:
            print("⚠️ No timestamp column, skipping funding rate merge")
            return candles_df

    candles_df['timestamp'] = pd.to_datetime(candles_df['timestamp'])

    # Merge asof (forward fill)
    merged = pd.merge_asof(
        candles_df.sort_values('timestamp'),
        funding_df,
        on='timestamp',
        direction='backward'  # Use most recent funding rate
    )

    # Calculate funding rate features
    merged['funding_rate_abs'] = merged['funding_rate'].abs()
    merged['funding_rate_positive'] = (merged['funding_rate'] > 0).astype(int)
    merged['funding_rate_high'] = (merged['funding_rate'] > 0.0001).astype(int)  # >0.01%

    # Rolling funding rate features
    merged['funding_rate_ma_3'] = merged['funding_rate'].rolling(3, min_periods=1).mean()
    merged['funding_rate_trend'] = merged['funding_rate'] - merged['funding_rate'].shift(1)

    print(f"✅ Merged {len(merged)} candles with funding rate")
    print(f"Funding rate coverage: {merged['funding_rate'].notna().sum()} / {len(merged)} candles")

    return merged


def create_short_labels(df, lookahead=3, threshold=0.002):
    """Create SHORT labels"""
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
    """Load data with funding rate"""
    print("="*80)
    print("Loading Data WITH FUNDING RATE")
    print("="*80)

    # Load candles
    df = pd.read_csv(DATA_DIR / "BTCUSDT_5m_max.csv")
    print(f"Loaded {len(df)} candles")

    # Add timestamp if not present
    if 'timestamp' not in df.columns and 'open_time' not in df.columns:
        # Create dummy timestamps (5-min intervals, recent data)
        end_time = datetime.now()
        start_time = end_time - timedelta(minutes=5 * len(df))
        df['timestamp'] = pd.date_range(start=start_time, periods=len(df), freq='5min')

    # Get funding rate
    funding_df = get_funding_rate_history()

    if funding_df is not None:
        # Merge with candles
        df = merge_funding_rate_with_candles(df, funding_df)
    else:
        print("⚠️ Continuing without funding rate")
        df['funding_rate'] = 0
        df['funding_rate_abs'] = 0
        df['funding_rate_positive'] = 0
        df['funding_rate_high'] = 0
        df['funding_rate_ma_3'] = 0
        df['funding_rate_trend'] = 0

    # Calculate other features
    df = calculate_features(df)

    short_features = ShortSpecificFeatures(lookback=20)
    df = short_features.calculate_all_features(df)

    regime_filter = MarketRegimeFilter()
    df = regime_filter.get_regime_features(df)

    # Remove NaN
    df = df.ffill().dropna()
    print(f"After processing: {len(df)} candles")

    # Feature selection
    baseline_features = ['rsi', 'macd', 'macd_signal', 'bb_upper', 'bb_lower',
                        'ema_21', 'ema_50', 'atr', 'volume_ratio', 'price_distance_ema']

    short_feature_names = short_features.get_feature_names()

    # NEW: Funding rate features
    funding_features = [
        'funding_rate', 'funding_rate_abs', 'funding_rate_positive',
        'funding_rate_high', 'funding_rate_ma_3', 'funding_rate_trend'
    ]

    all_features = baseline_features + short_feature_names + funding_features
    feature_columns = [f for f in all_features if f in df.columns]

    print(f"\nTotal features: {len(feature_columns)}")
    print(f"  - Baseline: {len([f for f in baseline_features if f in df.columns])}")
    print(f"  - SHORT-specific: {len([f for f in short_feature_names if f in df.columns])}")
    print(f"  - Funding rate: {len([f for f in funding_features if f in df.columns])}")

    # Create labels
    labels = create_short_labels(df, lookahead=3, threshold=0.002)

    # Class distribution
    unique, counts = np.unique(labels, return_counts=True)
    print(f"\nLabel Distribution:")
    print(f"  NO SHORT (0): {counts[0]} ({counts[0]/len(labels)*100:.1f}%)")
    print(f"  SHORT (1): {counts[1]} ({counts[1]/len(labels)*100:.1f}%)")

    X = df[feature_columns].values
    y = labels

    return X, y, feature_columns, df


def train_and_backtest(X, y, df):
    """Train and backtest with funding rate"""
    print("\n" + "="*80)
    print("Training XGBoost WITH FUNDING RATE")
    print("="*80)

    tscv = TimeSeriesSplit(n_splits=5)

    all_trades = []
    fold_metrics = []

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

        # Train
        model = xgb.XGBClassifier(**params)
        sample_weights = compute_sample_weight('balanced', y_train)
        model.fit(X_train, y_train, sample_weight=sample_weights, verbose=False)

        # Predict
        y_prob = model.predict_proba(X_val)[:, 1]

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
                    'funding_rate': df['funding_rate'].iloc[idx] if 'funding_rate' in df.columns else 0,
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
    print("Results WITH FUNDING RATE")
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

    # Train final model
    final_model = xgb.XGBClassifier(**params)
    sample_weights = compute_sample_weight('balanced', y)
    final_model.fit(X, y, sample_weight=sample_weights, verbose=False)

    return overall_win_rate, all_trades, final_model


def main():
    """Main training pipeline"""
    print("="*80)
    print("SHORT Model WITH FUNDING RATE - Final Attempt")
    print("="*80)
    print("Game Changer:")
    print("  - Funding Rate: Market sentiment indicator")
    print("  - HIGH funding + resistance = SHORT opportunity!")
    print("  - This is the missing piece!")
    print("="*80)

    # Load data
    X, y, feature_columns, df = load_and_prepare_data()

    # Train and backtest
    win_rate, trades, model = train_and_backtest(X, y, df)

    # Decision
    print("\n" + "="*80)
    print("Final Decision - WITH FUNDING RATE")
    print("="*80)

    if win_rate >= 0.60:
        print(f"✅ SUCCESS! SHORT win rate {win_rate*100:.1f}% >= 60%")
        print(f"✅ FUNDING RATE was the missing piece!")

        # Save model
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_name = f"short_with_funding_rate_{timestamp}"
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
        print(f"🔄 Funding rate helped significantly!")

    elif win_rate >= 0.30:
        print(f"⚠️ SOME IMPROVEMENT: SHORT win rate {win_rate*100:.1f}% (30-45%)")
        print(f"⚠️ Funding rate provided modest improvement")

    else:
        print(f"❌ INSUFFICIENT: SHORT win rate {win_rate*100:.1f}%")
        print(f"❌ Even funding rate cannot solve the problem")

    print(f"\nComplete Progress Summary:")
    print(f"  XGBoost (no funding): 26.0%")
    print(f"  LSTM (no funding): 17.3%")
    print(f"  XGBoost WITH funding rate: {win_rate*100:.1f}%")

    return win_rate, trades, model


if __name__ == "__main__":
    win_rate, trades, model = main()
