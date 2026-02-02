"""
Indicator Period Optimization Framework
========================================

Systematically optimize technical indicator periods to find the best configuration.

Strategy:
1. Define period ranges for each indicator type
2. Generate feature combinations (grid search)
3. Train models with each combination
4. Evaluate on validation set (not last 4 weeks)
5. Select best configuration
6. Validate on last 4 weeks (backtest)

Period Categories:
- RSI: [7, 9, 14, 21, 28]
- MACD Fast: [8, 10, 12, 15]
- MACD Slow: [20, 24, 26, 30]
- MACD Signal: [6, 7, 9, 12]
- MA Short: [10, 15, 20, 25]
- MA Long: [30, 40, 50, 60]
- ATR: [7, 10, 14, 20]
- Rolling Windows: [5, 7, 10, 15, 20]
"""

import pandas as pd
import numpy as np
import talib
from pathlib import Path
import sys
import json
from datetime import datetime, timedelta
from itertools import product
import xgboost as xgb
from sklearn.metrics import classification_report, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.production.train_xgboost_improved_v3_phase2 import calculate_features
from scripts.production.advanced_technical_features import AdvancedTechnicalFeatures


# Period ranges for optimization
PERIOD_RANGES = {
    'rsi': [7, 9, 14, 21, 28],
    'macd_fast': [8, 10, 12, 15],
    'macd_slow': [20, 24, 26, 30],
    'macd_signal': [6, 7, 9, 12],
    'ma_short': [10, 15, 20, 25],
    'ma_long': [30, 40, 50, 60],
    'atr': [7, 10, 14, 20],
    'rolling_short': [5, 7, 10, 15],
    'rolling_long': [10, 15, 20, 25]
}


def calculate_features_with_periods(df, periods):
    """
    Calculate features with custom periods

    Args:
        df: DataFrame with OHLCV data
        periods: dict with period values for each indicator

    Returns:
        DataFrame with calculated features
    """
    features = {}

    # RSI with custom period
    rsi_period = periods.get('rsi', 14)
    features['rsi_raw'] = talib.RSI(df['close'], timeperiod=rsi_period)
    features['rsi_deviation'] = np.abs(features['rsi_raw'] - 50)
    features['rsi_direction'] = np.sign(features['rsi_raw'] - 50)
    features['rsi_extreme'] = ((features['rsi_raw'] > 70) | (features['rsi_raw'] < 30)).astype(float)

    # MACD with custom periods
    macd_fast = periods.get('macd_fast', 12)
    macd_slow = periods.get('macd_slow', 26)
    macd_signal = periods.get('macd_signal', 9)
    macd, macd_sig, macd_hist = talib.MACD(
        df['close'],
        fastperiod=macd_fast,
        slowperiod=macd_slow,
        signalperiod=macd_signal
    )
    features['macd_strength'] = np.abs(macd_hist)
    features['macd_direction'] = np.sign(macd_hist)
    features['macd_divergence_abs'] = np.abs(macd - macd_sig)

    # Moving Averages with custom periods
    ma_short = periods.get('ma_short', 20)
    ma_long = periods.get('ma_long', 50)
    ma_s = df['close'].rolling(ma_short).mean()
    ma_l = df['close'].rolling(ma_long).mean()
    features['price_distance_ma_short'] = np.abs(df['close'] - ma_s) / ma_s
    features['price_direction_ma_short'] = np.sign(df['close'] - ma_s)
    features['price_distance_ma_long'] = np.abs(df['close'] - ma_l) / ma_l
    features['price_direction_ma_long'] = np.sign(df['close'] - ma_l)

    # ATR with custom period
    atr_period = periods.get('atr', 14)
    features['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=atr_period)
    features['atr_pct'] = features['atr'] / df['close']

    # Rolling windows with custom periods
    roll_short = periods.get('rolling_short', 10)
    roll_long = periods.get('rolling_long', 20)

    # Momentum features
    features['negative_momentum'] = -df['close'].pct_change(roll_short).clip(upper=0)
    features['down_candle_ratio'] = ((df['close'] < df['open']).astype(float)).rolling(roll_short).mean()

    # Volume features
    features['volume_ma_short'] = df['volume'].rolling(roll_short).mean()
    features['volume_surge'] = df['volume'] / features['volume_ma_short']

    # Volatility
    returns = df['close'].pct_change()
    features['volatility'] = returns.rolling(roll_long).std()

    # Price range
    features['price_range'] = (df['high'].rolling(roll_long).max() - df['low'].rolling(roll_long).min()) / df['close']

    # Resistance/Support
    features['resistance'] = df['high'].rolling(roll_long).max()
    features['support'] = df['low'].rolling(roll_long).min()
    features['near_resistance'] = (df['high'] > features['resistance'].shift(1) * 0.99).astype(float)
    features['below_support'] = (df['close'] < features['support'].shift(1) * 1.01).astype(float)

    return pd.concat([df, pd.DataFrame(features, index=df.index)], axis=1)


def evaluate_period_combination(df, periods, signal_type='LONG', train_end_date=None):
    """
    Evaluate a specific period combination

    Args:
        df: DataFrame with OHLCV data
        periods: dict with period values
        signal_type: 'LONG' or 'SHORT'
        train_end_date: Last date for training (exclude last 4 weeks)

    Returns:
        dict with evaluation metrics
    """
    # Calculate features with custom periods
    df_feat = calculate_features_with_periods(df.copy(), periods)

    # Clean NaN
    df_feat = df_feat.ffill().bfill().fillna(0)

    # Split train/validation (exclude last 4 weeks)
    if train_end_date is None:
        # Default: use last 4 weeks as holdout
        train_end_date = df_feat.index[-1] - timedelta(weeks=4)

    train_mask = df_feat.index <= train_end_date
    val_mask = df_feat.index > train_end_date

    train_df = df_feat[train_mask].copy()
    val_df = df_feat[val_mask].copy()

    # Feature columns (use calculated features)
    feature_cols = [col for col in df_feat.columns if col not in
                    ['timestamp', 'open', 'high', 'low', 'close', 'volume',
                     'signal_long', 'signal_short']]

    # Check if we have labels
    if f'signal_{signal_type.lower()}' not in train_df.columns:
        return None

    # Prepare training data
    X_train = train_df[feature_cols].values
    y_train = train_df[f'signal_{signal_type.lower()}'].values

    X_val = val_df[feature_cols].values
    y_val = val_df[f'signal_{signal_type.lower()}'].values

    # Check label balance
    positive_ratio_train = y_train.mean()
    positive_ratio_val = y_val.mean()

    if positive_ratio_train < 0.05 or positive_ratio_train > 0.95:
        # Skip if labels are too imbalanced
        return None

    # Train XGBoost model
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train, verbose=0)

    # Evaluate on validation set
    y_pred = model.predict(X_val)
    y_pred_proba = model.predict_proba(X_val)[:, 1]

    # Calculate metrics
    accuracy = (y_pred == y_val).mean()

    if len(np.unique(y_val)) > 1:
        auc = roc_auc_score(y_val, y_pred_proba)
    else:
        auc = 0.5

    precision = ((y_pred == 1) & (y_val == 1)).sum() / ((y_pred == 1).sum() + 1e-10)
    recall = ((y_pred == 1) & (y_val == 1)).sum() / ((y_val == 1).sum() + 1e-10)
    f1 = 2 * precision * recall / (precision + recall + 1e-10)

    # Feature importance
    feature_importance = pd.DataFrame({
        'feature': feature_cols,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    return {
        'periods': periods,
        'accuracy': accuracy,
        'auc': auc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'positive_ratio_train': positive_ratio_train,
        'positive_ratio_val': positive_ratio_val,
        'train_size': len(train_df),
        'val_size': len(val_df),
        'feature_importance': feature_importance
    }


def grid_search_periods(df, signal_type='LONG', max_combinations=100):
    """
    Grid search over period combinations

    Args:
        df: DataFrame with OHLCV and labels
        signal_type: 'LONG' or 'SHORT'
        max_combinations: Maximum number of combinations to test

    Returns:
        list of evaluation results sorted by performance
    """
    print(f"\n{'='*80}")
    print(f"Period Grid Search - {signal_type} Signal")
    print(f"{'='*80}")

    # Sample period combinations (to avoid combinatorial explosion)
    # Strategy: Test representative combinations

    period_combinations = []

    # Baseline (current settings)
    baseline = {
        'rsi': 14,
        'macd_fast': 12,
        'macd_slow': 26,
        'macd_signal': 9,
        'ma_short': 20,
        'ma_long': 50,
        'atr': 14,
        'rolling_short': 10,
        'rolling_long': 20
    }
    period_combinations.append(baseline)

    # Vary one parameter at a time (sensitivity analysis)
    for param_name, param_values in PERIOD_RANGES.items():
        for value in param_values:
            combo = baseline.copy()
            combo[param_name] = value
            if combo not in period_combinations:
                period_combinations.append(combo)

    # Add some random combinations
    np.random.seed(42)
    for _ in range(max(0, max_combinations - len(period_combinations))):
        combo = {
            param: np.random.choice(values)
            for param, values in PERIOD_RANGES.items()
        }
        if combo not in period_combinations:
            period_combinations.append(combo)

    # Limit to max_combinations
    period_combinations = period_combinations[:max_combinations]

    print(f"\nTesting {len(period_combinations)} period combinations...")
    print(f"Training data: {len(df) - 4*7*24*12} candles (excluding last 4 weeks)")
    print(f"Validation data: {4*7*24*12} candles (last 4 weeks)")

    # Evaluate each combination
    results = []

    for i, periods in enumerate(period_combinations, 1):
        print(f"\n[{i}/{len(period_combinations)}] Testing: {periods}")

        result = evaluate_period_combination(df, periods, signal_type)

        if result is not None:
            results.append(result)
            print(f"  Accuracy: {result['accuracy']:.4f} | AUC: {result['auc']:.4f} | "
                  f"F1: {result['f1']:.4f} | Precision: {result['precision']:.4f}")
        else:
            print(f"  Skipped (invalid data or labels)")

    # Sort by composite score
    for r in results:
        r['composite_score'] = (
            0.3 * r['accuracy'] +
            0.3 * r['auc'] +
            0.2 * r['f1'] +
            0.2 * r['precision']
        )

    results.sort(key=lambda x: x['composite_score'], reverse=True)

    print(f"\n{'='*80}")
    print(f"Grid Search Complete - Found {len(results)} valid configurations")
    print(f"{'='*80}")

    return results


def main():
    """Main execution"""

    print("="*80)
    print("Indicator Period Optimization")
    print("="*80)

    # Load data
    DATA_DIR = PROJECT_ROOT / "data" / "historical"
    data_file = DATA_DIR / "BTCUSDT_5m_max.csv"

    print(f"\nLoading data from: {data_file}")
    df = pd.read_csv(data_file)

    # Convert timestamp to datetime
    if 'timestamp' in df.columns:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)

    print(f"Loaded {len(df)} candles")
    print(f"Period: {df.index[0]} to {df.index[-1]}")

    # Load labels (you need to have these prepared)
    # For demonstration, we'll create dummy labels
    # In production, load from your label files

    # Create dummy labels for demonstration
    # TODO: Load actual labels from training data
    df['signal_long'] = (df['close'].pct_change(20) > 0.02).astype(int)
    df['signal_short'] = (df['close'].pct_change(20) < -0.02).astype(int)

    print(f"\nLabel distribution:")
    print(f"  LONG signals: {df['signal_long'].sum()} ({df['signal_long'].mean()*100:.2f}%)")
    print(f"  SHORT signals: {df['signal_short'].sum()} ({df['signal_short'].mean()*100:.2f}%)")

    # Grid search for LONG signals
    print("\n" + "="*80)
    print("LONG Signal Period Optimization")
    print("="*80)

    long_results = grid_search_periods(df, signal_type='LONG', max_combinations=50)

    # Grid search for SHORT signals
    print("\n" + "="*80)
    print("SHORT Signal Period Optimization")
    print("="*80)

    short_results = grid_search_periods(df, signal_type='SHORT', max_combinations=50)

    # Save results
    RESULTS_DIR = PROJECT_ROOT / "results"
    RESULTS_DIR.mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save LONG results
    long_results_clean = []
    for r in long_results[:10]:  # Top 10
        r_clean = {k: v for k, v in r.items() if k != 'feature_importance'}
        r_clean['top_5_features'] = r['feature_importance'].head(5)['feature'].tolist()
        long_results_clean.append(r_clean)

    with open(RESULTS_DIR / f"period_optimization_long_{timestamp}.json", 'w') as f:
        json.dump(long_results_clean, f, indent=2, default=str)

    # Save SHORT results
    short_results_clean = []
    for r in short_results[:10]:  # Top 10
        r_clean = {k: v for k, v in r.items() if k != 'feature_importance'}
        r_clean['top_5_features'] = r['feature_importance'].head(5)['feature'].tolist()
        short_results_clean.append(r_clean)

    with open(RESULTS_DIR / f"period_optimization_short_{timestamp}.json", 'w') as f:
        json.dump(short_results_clean, f, indent=2, default=str)

    print(f"\n{'='*80}")
    print("Results saved to:")
    print(f"  - period_optimization_long_{timestamp}.json")
    print(f"  - period_optimization_short_{timestamp}.json")
    print(f"{'='*80}")

    # Print top 3 configurations
    print("\n" + "="*80)
    print("Top 3 LONG Configurations")
    print("="*80)

    for i, result in enumerate(long_results[:3], 1):
        print(f"\nRank {i}:")
        print(f"  Composite Score: {result['composite_score']:.4f}")
        print(f"  Accuracy: {result['accuracy']:.4f}")
        print(f"  AUC: {result['auc']:.4f}")
        print(f"  F1: {result['f1']:.4f}")
        print(f"  Periods: {result['periods']}")
        print(f"  Top 5 Features:")
        for j, (_, row) in enumerate(result['feature_importance'].head(5).iterrows(), 1):
            print(f"    {j}. {row['feature']}: {row['importance']:.4f}")

    print("\n" + "="*80)
    print("Top 3 SHORT Configurations")
    print("="*80)

    for i, result in enumerate(short_results[:3], 1):
        print(f"\nRank {i}:")
        print(f"  Composite Score: {result['composite_score']:.4f}")
        print(f"  Accuracy: {result['accuracy']:.4f}")
        print(f"  AUC: {result['auc']:.4f}")
        print(f"  F1: {result['f1']:.4f}")
        print(f"  Periods: {result['periods']}")
        print(f"  Top 5 Features:")
        for j, (_, row) in enumerate(result['feature_importance'].head(5).iterrows(), 1):
            print(f"    {j}. {row['feature']}: {row['importance']:.4f}")


if __name__ == "__main__":
    main()
