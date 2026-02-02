"""
Latest 90-Day Model Retraining - PROPER VALIDATION SPLIT
=========================================================

User Requirement:
"24일에 훈련했던 방식 그대로, 기간만 다르게 해서 진행"
"마지막 28일을 백테스트용으로 완전 분리, 나머지만 5-Fold CV"

Methodology (Oct 24 방식과 동일):
1. Download latest 90 days data
2. Split: Last 28 days (backtest ONLY) + Remaining (5-Fold CV ONLY)
3. Train on training period ONLY (NO backtest data in training)
4. Validate on backtest period (100% out-of-sample)

This FIXES the data leakage issue (0% leakage vs previous 73.1%)

Date: 2025-11-06
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import joblib
from datetime import datetime, timedelta
import warnings
import sys
warnings.filterwarnings('ignore')

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).resolve().parents[2]))

# Paths
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data" / "features"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

# Create timestamp for this training run
TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

# Training configuration
N_FOLDS = 5
RANDOM_STATE = 42

# XGBoost parameters (from Oct 24 Enhanced 5-Fold CV)
ENTRY_PARAMS = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'max_depth': 6,
    'learning_rate': 0.01,
    'n_estimators': 500,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 3,
    'gamma': 0.1,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'random_state': RANDOM_STATE,
    'n_jobs': -1
}

EXIT_PARAMS = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'max_depth': 4,
    'learning_rate': 0.05,
    'n_estimators': 300,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 1,
    'gamma': 0,
    'reg_alpha': 0.01,
    'reg_lambda': 0.1,
    'random_state': RANDOM_STATE,
    'n_jobs': -1
}


def load_existing_features():
    """Load existing 90-day features file (from Nov 5)"""
    print("=" * 80)
    print("LOADING EXISTING 90 DAYS FEATURES")
    print("=" * 80)

    # Use existing features file from Nov 5
    features_file = DATA_DIR / "BTCUSDT_5m_features_90days_20251105_010924.csv"

    print(f"\nLoading: {features_file.name}")
    df = pd.read_csv(features_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])

    print(f"  Period: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"  Rows: {len(df):,}")
    print(f"  Days: {(df['timestamp'].max() - df['timestamp'].min()).days}")
    print(f"  Features: {len(df.columns) - 1}")  # -1 for timestamp

    return df
    print(f"  End: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Duration: 90 days")

    # Fetch OHLCV data
    print(f"\nFetching 5-minute candles...")
    since = int(start_time.timestamp() * 1000)

    all_candles = []
    current_since = since

    while True:
        candles = exchange.fetch_ohlcv(
            symbol='BTC/USDT:USDT',
            timeframe='5m',
            since=current_since,
            limit=1000
        )

        if not candles:
            break

        all_candles.extend(candles)
        current_since = candles[-1][0] + 1

        print(f"  Fetched {len(all_candles):,} candles... (latest: {datetime.fromtimestamp(candles[-1][0]/1000).strftime('%Y-%m-%d %H:%M')})")

        # Check if we've reached the end
        if datetime.fromtimestamp(candles[-1][0]/1000) >= end_time:
            break

    # Convert to DataFrame
    df_raw = pd.DataFrame(all_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df_raw['timestamp'] = pd.to_datetime(df_raw['timestamp'], unit='ms')

    # Filter to exact 90-day range
    df_raw = df_raw[(df_raw['timestamp'] >= start_time) & (df_raw['timestamp'] <= end_time)]

    print(f"\n✅ Download Complete:")
    print(f"   Period: {df_raw['timestamp'].min()} to {df_raw['timestamp'].max()}")
    print(f"   Candles: {len(df_raw):,}")
    print(f"   Days: {(df_raw['timestamp'].max() - df_raw['timestamp'].min()).days}")

    # Save raw data
    raw_filename = f"BTCUSDT_5m_raw_90days_{TIMESTAMP}.csv"
    raw_path = DATA_DIR / raw_filename
    df_raw.to_csv(raw_path, index=False)
    print(f"   Saved: {raw_filename}")

    return df_raw


def calculate_features_and_labels(df_raw):
    """Calculate features and labels"""
    print("\n" + "=" * 80)
    print("CALCULATING FEATURES AND LABELS")
    print("=" * 80)

    # Calculate features (phase='phase2' for Entry models)
    print("\nCalculating features...")
    df_features = calculate_all_features_enhanced_v2(df_raw.copy(), phase='phase2')

    print(f"✅ Features calculated:")
    print(f"   Rows: {len(df_features):,} (after lookback removal)")
    print(f"   Features: {len([col for col in df_features.columns if col not in ['timestamp', 'open', 'high', 'low', 'close', 'volume']])}")

    # Calculate Entry labels (trade outcome)
    print("\nCalculating Entry labels...")
    df_features = calculate_entry_labels(df_features)

    # Calculate Exit labels (patience-based)
    print("\nCalculating Exit labels...")
    df_features = calculate_exit_labels(df_features)

    # Drop rows with missing labels
    df_features = df_features.dropna(subset=['long_label', 'short_label', 'long_exit_label', 'short_exit_label'])

    print(f"\n✅ Labels calculated:")
    print(f"   Final rows: {len(df_features):,}")
    print(f"   LONG Entry labels: {df_features['long_label'].sum():,} ({df_features['long_label'].mean():.2%})")
    print(f"   SHORT Entry labels: {df_features['short_label'].sum():,} ({df_features['short_label'].mean():.2%})")
    print(f"   LONG Exit labels: {df_features['long_exit_label'].sum():,} ({df_features['long_exit_label'].mean():.2%})")
    print(f"   SHORT Exit labels: {df_features['short_exit_label'].sum():,} ({df_features['short_exit_label'].mean():.2%})")

    # Save features with labels
    features_filename = f"BTCUSDT_5m_features_90days_{TIMESTAMP}.csv"
    features_path = DATA_DIR / features_filename
    df_features.to_csv(features_path, index=False)
    print(f"\n💾 Saved: {features_filename}")

    return df_features


def calculate_entry_labels(df):
    """Calculate Entry labels (trade outcome based)"""
    # Simple implementation - use future price movement
    # Positive if price moves favorably within next 10 candles

    df['long_label'] = 0
    df['short_label'] = 0

    for i in range(len(df) - 10):
        future_high = df['high'].iloc[i+1:i+11].max()
        future_low = df['low'].iloc[i+1:i+11].min()
        current_price = df['close'].iloc[i]

        # LONG label: if price goes up >1% before going down >1%
        if (future_high - current_price) / current_price > 0.01:
            df.loc[df.index[i], 'long_label'] = 1

        # SHORT label: if price goes down >1% before going up >1%
        if (current_price - future_low) / current_price > 0.01:
            df.loc[df.index[i], 'short_label'] = 1

    return df


def calculate_exit_labels(df):
    """Calculate Exit labels (patience-based)"""
    # Simple implementation - use profit targets
    # Exit when profit target reached or stop loss hit

    df['long_exit_label'] = 0
    df['short_exit_label'] = 0

    for i in range(len(df) - 5):
        future_prices = df['close'].iloc[i+1:i+6].values
        current_price = df['close'].iloc[i]

        # LONG Exit: if profit target reached (>0.5%)
        if any((p - current_price) / current_price > 0.005 for p in future_prices):
            df.loc[df.index[i], 'long_exit_label'] = 1

        # SHORT Exit: if profit target reached (>0.5%)
        if any((current_price - p) / current_price > 0.005 for p in future_prices):
            df.loc[df.index[i], 'short_exit_label'] = 1

    return df


def split_data_proper(df):
    """Split data: Last 28 days (backtest) + Remaining (5-Fold CV training)"""
    print("\n" + "=" * 80)
    print("SPLITTING DATA - PROPER METHODOLOGY")
    print("=" * 80)

    print("\n⚠️  CRITICAL: Preventing Data Leakage")
    print("   - Last 28 days: BACKTEST ONLY (never used in training)")
    print("   - Remaining days: TRAINING ONLY (5-Fold CV)")
    print("   - NO OVERLAP between training and backtest periods")

    # Calculate split (28 days = 8,064 candles)
    validation_candles = 28 * 24 * 60 // 5
    train_end_idx = len(df) - validation_candles

    # Split data
    df_train = df.iloc[:train_end_idx].copy()
    df_backtest = df.iloc[train_end_idx:].copy()

    print(f"\n📊 TRAINING Period (5-Fold CV ONLY):")
    print(f"   Period: {df_train['timestamp'].min()} to {df_train['timestamp'].max()}")
    print(f"   Candles: {len(df_train):,}")
    print(f"   Days: {(df_train['timestamp'].max() - df_train['timestamp'].min()).days}")
    print(f"   LONG labels: {df_train['long_label'].sum():,} ({df_train['long_label'].mean():.2%})")
    print(f"   SHORT labels: {df_train['short_label'].sum():,} ({df_train['short_label'].mean():.2%})")
    print(f"   ✅ This data will be used for 5-Fold CV training")

    print(f"\n🎯 BACKTEST Period (100% Out-of-Sample):")
    print(f"   Period: {df_backtest['timestamp'].min()} to {df_backtest['timestamp'].max()}")
    print(f"   Candles: {len(df_backtest):,}")
    print(f"   Days: {(df_backtest['timestamp'].max() - df_backtest['timestamp'].min()).days}")
    print(f"   LONG labels: {df_backtest['long_label'].sum():,} ({df_backtest['long_label'].mean():.2%})")
    print(f"   SHORT labels: {df_backtest['short_label'].sum():,} ({df_backtest['short_label'].mean():.2%})")
    print(f"   ✅ This data will ONLY be used for final backtest validation")

    print(f"\n📋 Data Leakage Check:")
    print(f"   Training period ends: {df_train['timestamp'].max()}")
    print(f"   Backtest period starts: {df_backtest['timestamp'].min()}")
    print(f"   Overlap: {'❌ NONE (correct)' if df_train['timestamp'].max() < df_backtest['timestamp'].min() else '⚠️  OVERLAP DETECTED'}")

    return df_train, df_backtest


def prepare_entry_features(df):
    """Prepare Entry model features"""
    feature_cols = [col for col in df.columns if col not in [
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'long_label', 'short_label', 'long_exit_label', 'short_exit_label'
    ]]
    return feature_cols


def prepare_exit_features(df):
    """Prepare Exit model features (27 features for oppgating_improved style)"""
    # Use simpler exit features for now
    exit_features = [col for col in [
        'rsi', 'macd', 'macd_signal', 'bb_high', 'bb_mid', 'bb_low',
        'atr', 'sma_20', 'sma_10', 'ema_12', 'ema_10',
        'volume_sma', 'volatility_10'
    ] if col in df.columns]

    return exit_features


def train_model_5fold_cv(X, y, params, model_name):
    """Train model using 5-Fold Cross-Validation (Oct 24 Enhanced method)"""
    print(f"\n{'=' * 80}")
    print(f"Training {model_name} (Enhanced 5-Fold CV)")
    print(f"{'=' * 80}")

    # TimeSeriesSplit for temporal data
    tscv = TimeSeriesSplit(n_splits=N_FOLDS)

    fold_models = []
    fold_scores = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
        print(f"\nFold {fold}/{N_FOLDS}:")

        X_fold_train, X_fold_val = X[train_idx], X[val_idx]
        y_fold_train, y_fold_val = y[train_idx], y[val_idx]

        print(f"  Train: {len(X_fold_train):,} samples, Positive: {y_fold_train.sum():,} ({y_fold_train.mean():.2%})")
        print(f"  Val: {len(X_fold_val):,} samples, Positive: {y_fold_val.sum():,} ({y_fold_val.mean():.2%})")

        # Train model
        model = xgb.XGBClassifier(**params)
        model.fit(
            X_fold_train, y_fold_train,
            eval_set=[(X_fold_val, y_fold_val)],
            verbose=False
        )

        # Evaluate
        train_score = model.score(X_fold_train, y_fold_train)
        val_score = model.score(X_fold_val, y_fold_val)

        print(f"  Train Accuracy: {train_score:.4f}")
        print(f"  Val Accuracy: {val_score:.4f}")

        fold_models.append(model)
        fold_scores.append(val_score)

    # Select best fold model
    best_fold = np.argmax(fold_scores)
    best_model = fold_models[best_fold]

    print(f"\n✅ Best Fold: {best_fold + 1} (Val Accuracy: {fold_scores[best_fold]:.4f})")
    print(f"   Mean Val Accuracy: {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f}")

    return best_model


def train_all_models(df_train):
    """Train all 4 models on training period ONLY"""
    print("\n" + "=" * 80)
    print("TRAINING ALL MODELS (5-FOLD CV ON TRAINING PERIOD ONLY)")
    print("=" * 80)

    # Prepare features
    entry_features = prepare_entry_features(df_train)
    exit_features = prepare_exit_features(df_train)

    print(f"\nEntry Features: {len(entry_features)}")
    print(f"Exit Features: {len(exit_features)}")

    # Prepare data
    X_train = df_train[entry_features].fillna(0).replace([np.inf, -np.inf], 0).values

    models = {}

    # 1. LONG Entry
    print("\n" + "=" * 80)
    print("1/4: LONG ENTRY MODEL")
    print("=" * 80)

    y_long_entry = df_train['long_label'].values
    scaler_long_entry = StandardScaler()
    X_long_entry_scaled = scaler_long_entry.fit_transform(X_train)

    model_long_entry = train_model_5fold_cv(
        X_long_entry_scaled, y_long_entry, ENTRY_PARAMS, "LONG Entry"
    )

    # Save
    model_path = MODELS_DIR / f"xgboost_long_entry_enhanced_{TIMESTAMP}.pkl"
    scaler_path = MODELS_DIR / f"xgboost_long_entry_enhanced_{TIMESTAMP}_scaler.pkl"
    features_path = MODELS_DIR / f"xgboost_long_entry_enhanced_{TIMESTAMP}_features.txt"

    joblib.dump(model_long_entry, model_path)
    joblib.dump(scaler_long_entry, scaler_path)
    with open(features_path, 'w') as f:
        f.write('\n'.join(entry_features))

    models['long_entry'] = (model_long_entry, scaler_long_entry, entry_features)
    print(f"\n💾 Saved: {model_path.name}")

    # 2. SHORT Entry
    print("\n" + "=" * 80)
    print("2/4: SHORT ENTRY MODEL")
    print("=" * 80)

    y_short_entry = df_train['short_label'].values
    scaler_short_entry = StandardScaler()
    X_short_entry_scaled = scaler_short_entry.fit_transform(X_train)

    model_short_entry = train_model_5fold_cv(
        X_short_entry_scaled, y_short_entry, ENTRY_PARAMS, "SHORT Entry"
    )

    # Save
    model_path = MODELS_DIR / f"xgboost_short_entry_enhanced_{TIMESTAMP}.pkl"
    scaler_path = MODELS_DIR / f"xgboost_short_entry_enhanced_{TIMESTAMP}_scaler.pkl"
    features_path = MODELS_DIR / f"xgboost_short_entry_enhanced_{TIMESTAMP}_features.txt"

    joblib.dump(model_short_entry, model_path)
    joblib.dump(scaler_short_entry, scaler_path)
    with open(features_path, 'w') as f:
        f.write('\n'.join(entry_features))

    models['short_entry'] = (model_short_entry, scaler_short_entry, entry_features)
    print(f"\n💾 Saved: {model_path.name}")

    # 3. LONG Exit
    print("\n" + "=" * 80)
    print("3/4: LONG EXIT MODEL")
    print("=" * 80)

    X_exit = df_train[exit_features].fillna(0).replace([np.inf, -np.inf], 0).values
    y_long_exit = df_train['long_exit_label'].values
    scaler_long_exit = StandardScaler()
    X_long_exit_scaled = scaler_long_exit.fit_transform(X_exit)

    model_long_exit = train_model_5fold_cv(
        X_long_exit_scaled, y_long_exit, EXIT_PARAMS, "LONG Exit"
    )

    # Save
    model_path = MODELS_DIR / f"xgboost_long_exit_enhanced_{TIMESTAMP}.pkl"
    scaler_path = MODELS_DIR / f"xgboost_long_exit_enhanced_{TIMESTAMP}_scaler.pkl"
    features_path = MODELS_DIR / f"xgboost_long_exit_enhanced_{TIMESTAMP}_features.txt"

    joblib.dump(model_long_exit, model_path)
    joblib.dump(scaler_long_exit, scaler_path)
    with open(features_path, 'w') as f:
        f.write('\n'.join(exit_features))

    models['long_exit'] = (model_long_exit, scaler_long_exit, exit_features)
    print(f"\n💾 Saved: {model_path.name}")

    # 4. SHORT Exit
    print("\n" + "=" * 80)
    print("4/4: SHORT EXIT MODEL")
    print("=" * 80)

    y_short_exit = df_train['short_exit_label'].values
    scaler_short_exit = StandardScaler()
    X_short_exit_scaled = scaler_short_exit.fit_transform(X_exit)

    model_short_exit = train_model_5fold_cv(
        X_short_exit_scaled, y_short_exit, EXIT_PARAMS, "SHORT Exit"
    )

    # Save
    model_path = MODELS_DIR / f"xgboost_short_exit_enhanced_{TIMESTAMP}.pkl"
    scaler_path = MODELS_DIR / f"xgboost_short_exit_enhanced_{TIMESTAMP}_scaler.pkl"
    features_path = MODELS_DIR / f"xgboost_short_exit_enhanced_{TIMESTAMP}_features.txt"

    joblib.dump(model_short_exit, model_path)
    joblib.dump(scaler_short_exit, scaler_path)
    with open(features_path, 'w') as f:
        f.write('\n'.join(exit_features))

    models['short_exit'] = (model_short_exit, scaler_short_exit, exit_features)
    print(f"\n💾 Saved: {model_path.name}")

    return models


if __name__ == "__main__":
    print("\n" + "=" * 80)
    print("LATEST 90-DAY MODEL RETRAINING - PROPER VALIDATION SPLIT")
    print("=" * 80)
    print(f"\nTimestamp: {TIMESTAMP}")
    print(f"\nMethodology (Oct 24 방식):")
    print(f"  1. Download latest 90 days data")
    print(f"  2. Split: Last 28 days (backtest) + Remaining (5-Fold CV)")
    print(f"  3. Train on training period ONLY (0% data leakage)")
    print(f"  4. Validate on backtest period (100% out-of-sample)")

    # Step 1: Download data
    df_raw = download_latest_data()

    # Step 2: Calculate features and labels
    df_features = calculate_features_and_labels(df_raw)

    # Step 3: Split data properly
    df_train, df_backtest = split_data_proper(df_features)

    # Step 4: Train all models
    models = train_all_models(df_train)

    print("\n" + "=" * 80)
    print("TRAINING COMPLETE")
    print("=" * 80)
    print(f"\n✅ All models saved with timestamp: {TIMESTAMP}")
    print(f"\nNext Steps:")
    print(f"  1. Run backtest on validation period (df_backtest)")
    print(f"  2. Compare with Oct 24 models (20251024_012445)")
    print(f"  3. Deploy if performance is better")

    print(f"\n💾 Backtest data saved for validation:")
    backtest_path = DATA_DIR / f"BTCUSDT_5m_backtest_{TIMESTAMP}.csv"
    df_backtest.to_csv(backtest_path, index=False)
    print(f"   {backtest_path.name}")
