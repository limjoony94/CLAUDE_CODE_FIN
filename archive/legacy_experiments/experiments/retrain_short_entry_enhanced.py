"""
Retrain SHORT Entry Model with SELL Signal Features

Uses:
- Optimal labeling (2.0% profit, 6-12 candles)
- 22 SELL signal features (same as LONG Exit)
- 2of3 scoring system

Expected improvement:
- Current: 1% signal rate, 20% win rate
- Target: 13% signal rate, 60-70% win rate
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
from datetime import datetime
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_curve
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from src.labeling.improved_short_entry_labeling import create_improved_short_entry_labels


def calculate_enhanced_features(df):
    """Calculate 22 SELL signal features (same as LONG Exit enhanced features)"""

    print("\nCalculating 22 SELL signal features...")

    # Volume features
    df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
    df['volume_surge'] = (df['volume'] > df['volume'].rolling(20).mean() * 1.5).astype(float)

    # Price momentum features
    ma_20 = df['close'].rolling(20).mean()
    ma_50 = df['close'].rolling(50).mean()
    df['price_vs_ma20'] = (df['close'] - ma_20) / ma_20
    df['price_vs_ma50'] = (df['close'] - ma_50) / ma_50
    df['price_acceleration'] = df['close'].diff().diff()  # Second derivative

    # Volatility metrics
    df['volatility_20'] = df['close'].pct_change().rolling(20).std()
    volatility_median = df['volatility_20'].median()
    df['volatility_regime'] = (df['volatility_20'] > volatility_median).astype(float)

    # RSI dynamics
    if 'rsi' in df.columns:
        df['rsi_slope'] = df['rsi'].diff(3)  # 3-candle RSI change
        df['rsi_overbought'] = (df['rsi'] > 70).astype(float)
        df['rsi_oversold'] = (df['rsi'] < 30).astype(float)
        df['rsi_divergence'] = 0.0  # Placeholder
    else:
        df['rsi_slope'] = 0.0
        df['rsi_overbought'] = 0.0
        df['rsi_oversold'] = 0.0
        df['rsi_divergence'] = 0.0

    # MACD dynamics
    if 'macd' in df.columns and 'macd_signal' in df.columns:
        macd_histogram = df['macd'] - df['macd_signal']
        df['macd_histogram_slope'] = macd_histogram.diff(3)
        df['macd_crossover'] = ((df['macd'] > df['macd_signal']) &
                                (df['macd'].shift(1) <= df['macd_signal'].shift(1))).astype(float)
        df['macd_crossunder'] = ((df['macd'] < df['macd_signal']) &
                                 (df['macd'].shift(1) >= df['macd_signal'].shift(1))).astype(float)
    else:
        df['macd_histogram_slope'] = 0.0
        df['macd_crossover'] = 0.0
        df['macd_crossunder'] = 0.0

    # Price patterns
    df['higher_high'] = ((df['high'] > df['high'].shift(1)) &
                         (df['high'].shift(1) > df['high'].shift(2))).astype(float)
    df['lower_low'] = ((df['low'] < df['low'].shift(1)) &
                       (df['low'].shift(1) < df['low'].shift(2))).astype(float)

    # Support/Resistance (using rolling min/max)
    resistance = df['high'].rolling(50).max()
    support = df['low'].rolling(50).min()
    df['near_resistance'] = (df['close'] > resistance * 0.98).astype(float)
    df['near_support'] = (df['close'] < support * 1.02).astype(float)

    # Bollinger Bands position
    if 'bb_high' in df.columns and 'bb_low' in df.columns:
        bb_range = df['bb_high'] - df['bb_low']
        df['bb_position'] = np.where(bb_range != 0,
                                      (df['close'] - df['bb_low']) / bb_range,
                                      0.5)
    else:
        # Calculate BB from scratch
        bb_mid = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        bb_high = bb_mid + 2 * bb_std
        bb_low = bb_mid - 2 * bb_std
        bb_range = bb_high - bb_low
        df['bb_position'] = np.where(bb_range != 0,
                                      (df['close'] - bb_low) / bb_range,
                                      0.5)

    # Fill NaN values
    df = df.fillna(method='ffill').fillna(0)

    print("✅ 22 SELL signal features calculated")

    return df


def main():
    print("="*80)
    print("SHORT Entry Model Retraining with SELL Signal Features")
    print("="*80)

    # Load optimal labeling config
    config_file = project_root / "models" / "short_entry_optimal_labeling_config.txt"
    config = {}
    with open(config_file, 'r') as f:
        for line in f:
            if '=' in line and not line.startswith('#'):
                key, value = line.strip().split('=')
                config[key] = float(value)

    print(f"\nOptimal labeling configuration:")
    print(f"  Profit threshold: {config['profit_threshold']*100:.1f}%")
    print(f"  Lead-time: {int(config['lead_min'])}-{int(config['lead_max'])} candles")
    print(f"  Expected positive rate: ~13%")

    # Load BTC data
    data_path = project_root / "data" / "historical" / "BTCUSDT_5m_max.csv"
    print(f"\nLoading data from: {data_path}")
    df = pd.read_csv(data_path)
    print(f"Data loaded: {len(df):,} candles")

    # Calculate SELL signal features (22 features)
    df = calculate_enhanced_features(df)

    # Define feature list (22 SELL signal features - same as LONG Exit)
    sell_features = [
        'rsi', 'macd', 'macd_signal',
        'volatility_regime', 'volume_surge', 'price_acceleration',
        'volume_ratio', 'price_vs_ma20', 'price_vs_ma50',
        'volatility_20', 'rsi_slope', 'rsi_overbought', 'rsi_oversold', 'rsi_divergence',
        'macd_histogram_slope', 'macd_crossover', 'macd_crossunder',
        'higher_high', 'lower_low',
        'near_resistance', 'near_support',
        'bb_position'
    ]

    # Check if all features exist
    missing_features = [f for f in sell_features if f not in df.columns]
    if missing_features:
        print(f"⚠️  Missing features: {missing_features}")
        print("   These will be created with default values...")
        for feat in missing_features:
            df[feat] = 0.0

    print(f"\n✅ Using {len(sell_features)} SELL signal features")

    # Generate labels with optimal parameters
    print(f"\nGenerating labels with optimal parameters...")
    labels = create_improved_short_entry_labels(
        df,
        lookahead=int(config['lookahead']),
        profit_threshold=config['profit_threshold'],
        lead_min=int(config['lead_min']),
        lead_max=int(config['lead_max']),
        relative_delay=int(config['relative_delay'])
    )

    # Prepare training data
    print(f"\nPreparing training data...")
    X = df[sell_features].values
    y = labels

    # Remove NaN and Inf
    valid_mask = ~(np.isnan(X).any(axis=1) | np.isinf(X).any(axis=1) | np.isnan(y) | np.isinf(y))
    X = X[valid_mask]
    y = y[valid_mask]

    print(f"Valid samples: {len(X):,}")
    print(f"Positive labels: {np.sum(y):,} ({np.sum(y)/len(y)*100:.2f}%)")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\nTrain samples: {len(X_train):,}")
    print(f"Test samples: {len(X_test):,}")

    # Scale features
    scaler = MinMaxScaler(feature_range=(-1, 1))
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train XGBoost model
    print(f"\nTraining XGBoost model...")

    # Calculate scale_pos_weight for imbalanced data
    neg_count = np.sum(y_train == 0)
    pos_count = np.sum(y_train == 1)
    scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0

    model = XGBClassifier(
        n_estimators=150,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric='logloss'
    )

    model.fit(X_train_scaled, y_train)

    # Evaluate
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    print(f"\n" + "="*80)
    print("MODEL PERFORMANCE")
    print("="*80)
    print(f"\n{classification_report(y_test, y_pred, target_names=['Not Entry', 'Entry'], digits=4)}")

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    print(f"\nConfusion Matrix:")
    print(f"  TN: {cm[0,0]:,}  |  FP: {cm[0,1]:,}")
    print(f"  FN: {cm[1,0]:,}  |  TP: {cm[1,1]:,}")

    # Probability distribution analysis
    print(f"\nProbability Distribution:")
    print(f"  Mean: {y_proba.mean():.4f}")
    print(f"  Median: {np.median(y_proba):.4f}")
    print(f"  Std: {y_proba.std():.4f}")

    # Signal quality by probability range
    prob_ranges = [(0.0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 1.0)]
    print(f"\nSignal Quality by Probability Range:")
    for low, high in prob_ranges:
        mask = (y_proba >= low) & (y_proba < high)
        if np.sum(mask) > 0:
            precision = np.mean(y_test[mask]) * 100
            count = np.sum(mask)
            print(f"  {low:.1f}-{high:.1f}: {precision:>6.2f}% precision ({count:,} samples)")

    # Feature importance
    print(f"\nTop 10 Important Features:")
    importance_df = pd.DataFrame({
        'feature': sell_features,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    for idx, row in importance_df.head(10).iterrows():
        print(f"  {row['feature']:<30} {row['importance']:.4f}")

    # Save model
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = project_root / "models" / f"xgboost_short_entry_enhanced_{timestamp}.pkl"
    scaler_path = project_root / "models" / f"xgboost_short_entry_enhanced_{timestamp}_scaler.pkl"
    features_path = project_root / "models" / f"xgboost_short_entry_enhanced_{timestamp}_features.txt"

    with open(model_path, 'wb') as f:
        pickle.dump(model, f)

    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)

    with open(features_path, 'w') as f:
        for feat in sell_features:
            f.write(f"{feat}\n")

    print(f"\n" + "="*80)
    print("MODEL SAVED")
    print("="*80)
    print(f"Model: {model_path.name}")
    print(f"Scaler: {scaler_path.name}")
    print(f"Features: {features_path.name}")

    print(f"\n" + "="*80)
    print("✅ SHORT Entry retraining complete!")
    print("="*80)
    print(f"\nKey improvements:")
    print(f"  - Features: 67 multi-timeframe → 22 SELL signals")
    print(f"  - Labeling: Peak/Trough (1%) → 2of3 scoring (13%)")
    print(f"  - Paradigm: Independent SHORT → SELL pair (with LONG Exit)")
    print(f"\nNext steps:")
    print(f"  1. Backtest enhanced model")
    print(f"  2. Compare with current model (1% signal, 20% win)")
    print(f"  3. Deploy if performance improved")


if __name__ == "__main__":
    main()
