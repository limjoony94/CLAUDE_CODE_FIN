#!/usr/bin/env python3
"""
Train Models with Long-Term Features
=====================================

Train 4 models with enhanced features (baseline 107 + long-term 23 = 140 total):
1. LONG Entry Model
2. SHORT Entry Model
3. LONG Exit Model
4. SHORT Exit Model

Features:
- Baseline: RSI, MACD, BB, ATR, EMA, Volume, etc. (107 features)
- Long-term: MA/EMA 200, Volume 200, ATR 200, RSI 200, BB 200, S/R 200 (23 features)
- Total: 140 features

Expected Improvements:
- Win Rate: 63.6% → 67-70% (+3-6%p)
- Sharpe: 0.336 → 0.37-0.40 (+10-20%)
- Max DD: -12.2% → -8-10% (-20-30% improvement)

Created: 2025-10-23
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import sys
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiments.calculate_features_with_longterm import (
    calculate_all_features_enhanced,
    get_long_term_feature_list
)

DATA_DIR = PROJECT_ROOT / "data" / "historical"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# Training parameters
LEVERAGE = 4
TEST_SIZE = 0.2
RANDOM_STATE = 42

# XGBoost hyperparameters (optimized from previous training)
XGBOOST_PARAMS = {
    'n_estimators': 300,
    'max_depth': 6,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 3,
    'gamma': 0.1,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'random_state': RANDOM_STATE,
    'n_jobs': -1,
    'scale_pos_weight': 1.0
}

def create_trade_outcome_labels(df, forward_periods=240, profit_threshold=0.03, loss_threshold=-0.015):
    """
    Create trade outcome labels for LONG and SHORT

    Args:
        df: DataFrame with OHLCV + features
        forward_periods: Max holding period (240 candles = 20 hours)
        profit_threshold: Take profit at +3% (leveraged P&L)
        loss_threshold: Stop loss at -1.5% (leveraged P&L)

    Returns:
        DataFrame with outcome labels
    """
    print(f"\nCreating trade outcome labels...")
    print(f"  Forward periods: {forward_periods} candles ({forward_periods/12:.1f} hours)")
    print(f"  Profit threshold: {profit_threshold*100:.1f}% (leveraged)")
    print(f"  Loss threshold: {loss_threshold*100:.1f}% (leveraged)")

    df = df.copy()

    # Initialize outcome columns
    df['long_outcome'] = 0  # 0 = loss, 1 = profit
    df['short_outcome'] = 0

    # Calculate forward returns (leveraged)
    for i in range(len(df) - forward_periods):
        entry_price = df.iloc[i]['close']

        # Look forward for exit conditions
        for j in range(1, forward_periods + 1):
            if i + j >= len(df):
                break

            current_price = df.iloc[i + j]['close']

            # LONG outcome (leveraged P&L)
            long_pnl = ((current_price - entry_price) / entry_price) * LEVERAGE
            if long_pnl >= profit_threshold:
                df.iloc[i, df.columns.get_loc('long_outcome')] = 1  # Profit
                break
            elif long_pnl <= loss_threshold:
                df.iloc[i, df.columns.get_loc('long_outcome')] = 0  # Loss
                break

            # SHORT outcome (leveraged P&L)
            short_pnl = ((entry_price - current_price) / entry_price) * LEVERAGE
            if short_pnl >= profit_threshold:
                df.iloc[i, df.columns.get_loc('short_outcome')] = 1  # Profit
                break
            elif short_pnl <= loss_threshold:
                df.iloc[i, df.columns.get_loc('short_outcome')] = 0  # Loss
                break

    # Statistics
    total_samples = len(df) - forward_periods
    long_profit_rate = df['long_outcome'][:total_samples].sum() / total_samples * 100
    short_profit_rate = df['short_outcome'][:total_samples].sum() / total_samples * 100

    print(f"\n  Label Statistics:")
    print(f"    Total samples: {total_samples:,}")
    print(f"    LONG profit rate: {long_profit_rate:.1f}%")
    print(f"    SHORT profit rate: {short_profit_rate:.1f}%")

    return df

def create_exit_labels(df, forward_periods=240):
    """
    Create exit labels for LONG and SHORT positions

    1 = Exit now (favorable), 0 = Hold (not favorable)

    Args:
        df: DataFrame with OHLCV + features
        forward_periods: Periods to look ahead for exit decision

    Returns:
        DataFrame with exit labels
    """
    print(f"\nCreating exit labels...")
    print(f"  Forward periods: {forward_periods} candles ({forward_periods/12:.1f} hours)")

    df = df.copy()

    # Initialize exit columns
    df['long_exit'] = 0  # 0 = hold, 1 = exit
    df['short_exit'] = 0

    # For each candle, check if price will decline (LONG exit) or rise (SHORT exit) soon
    for i in range(len(df) - forward_periods):
        current_price = df.iloc[i]['close']

        # Look forward to find peak/trough
        future_prices = df.iloc[i+1:i+forward_periods+1]['close'].values

        if len(future_prices) == 0:
            continue

        # LONG exit: Label as 1 if price will decline from current level
        max_future = future_prices.max()
        max_gain = ((max_future - current_price) / current_price) * 100

        # If max future gain is small (<1%), label as exit (price stalling/declining)
        if max_gain < 1.0:
            df.iloc[i, df.columns.get_loc('long_exit')] = 1

        # SHORT exit: Label as 1 if price will rise from current level
        min_future = future_prices.min()
        max_short_gain = ((current_price - min_future) / current_price) * 100

        # If max short gain is small (<1%), label as exit (price rising/stalling)
        if max_short_gain < 1.0:
            df.iloc[i, df.columns.get_loc('short_exit')] = 1

    # Statistics
    total_samples = len(df) - forward_periods
    long_exit_rate = df['long_exit'][:total_samples].sum() / total_samples * 100
    short_exit_rate = df['short_exit'][:total_samples].sum() / total_samples * 100

    print(f"\n  Label Statistics:")
    print(f"    Total samples: {total_samples:,}")
    print(f"    LONG exit rate: {long_exit_rate:.1f}%")
    print(f"    SHORT exit rate: {short_exit_rate:.1f}%")

    return df

def train_model(X_train, y_train, X_val, y_val, model_name, class_weight='balanced'):
    """
    Train XGBoost model

    Args:
        X_train, y_train: Training data
        X_val, y_val: Validation data
        model_name: Name for logging
        class_weight: Class weight strategy

    Returns:
        Trained model, scaler, feature names
    """
    print(f"\n{'='*80}")
    print(f"Training {model_name}")
    print(f"{'='*80}")

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # Calculate scale_pos_weight if using balanced
    if class_weight == 'balanced':
        neg_count = (y_train == 0).sum()
        pos_count = (y_train == 1).sum()
        scale_pos_weight = neg_count / pos_count if pos_count > 0 else 1.0
        print(f"\nClass distribution:")
        print(f"  Negative: {neg_count:,} ({neg_count/len(y_train)*100:.1f}%)")
        print(f"  Positive: {pos_count:,} ({pos_count/len(y_train)*100:.1f}%)")
        print(f"  Scale pos weight: {scale_pos_weight:.2f}")
    else:
        scale_pos_weight = 1.0

    # Update XGBoost params with scale_pos_weight
    params = XGBOOST_PARAMS.copy()
    params['scale_pos_weight'] = scale_pos_weight

    # Train model
    print(f"\nTraining XGBoost...")
    model = XGBClassifier(**params)
    model.fit(
        X_train_scaled, y_train,
        eval_set=[(X_val_scaled, y_val)],
        verbose=False
    )

    # Predictions
    y_train_pred = model.predict(X_train_scaled)
    y_val_pred = model.predict(X_val_scaled)

    y_train_proba = model.predict_proba(X_train_scaled)[:, 1]
    y_val_proba = model.predict_proba(X_val_scaled)[:, 1]

    # Metrics
    print(f"\n{'='*60}")
    print(f"TRAINING SET PERFORMANCE")
    print(f"{'='*60}")
    print(classification_report(y_train, y_train_pred, digits=3))
    train_auc = roc_auc_score(y_train, y_train_proba)
    print(f"ROC AUC: {train_auc:.4f}")

    print(f"\n{'='*60}")
    print(f"VALIDATION SET PERFORMANCE")
    print(f"{'='*60}")
    print(classification_report(y_val, y_val_pred, digits=3))
    val_auc = roc_auc_score(y_val, y_val_proba)
    print(f"ROC AUC: {val_auc:.4f}")

    # Feature importance (top 20)
    feature_importance = pd.DataFrame({
        'feature': X_train.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)

    print(f"\n{'='*60}")
    print(f"TOP 20 FEATURES")
    print(f"{'='*60}")
    for idx, row in feature_importance.head(20).iterrows():
        print(f"  {row['feature']:<40} {row['importance']:.4f}")

    # Check if long-term features are in top 20
    long_term_features = get_long_term_feature_list()
    long_term_in_top20 = [f for f in feature_importance.head(20)['feature'].values if f in long_term_features]
    print(f"\n  Long-term features in top 20: {len(long_term_in_top20)}")
    if long_term_in_top20:
        for f in long_term_in_top20:
            importance = feature_importance[feature_importance['feature'] == f]['importance'].values[0]
            print(f"    ✅ {f:<35} {importance:.4f}")

    return model, scaler, X_train.columns.tolist(), {
        'train_auc': train_auc,
        'val_auc': val_auc,
        'feature_importance': feature_importance
    }

def main():
    print("="*80)
    print("TRAIN MODELS WITH LONG-TERM FEATURES")
    print("="*80)
    print()

    # Load raw data
    print("="*80)
    print("Loading Raw Data")
    print("="*80)

    data_file = DATA_DIR / "BTCUSDT_5m_max.csv"
    df_raw = pd.read_csv(data_file)
    print(f"✅ Loaded {len(df_raw):,} candles")
    print()

    # Calculate enhanced features
    print("="*80)
    print("Calculating Enhanced Features")
    print("="*80)

    df_features = calculate_all_features_enhanced(df_raw)
    print(f"\n✅ Features calculated: {len(df_features.columns)} total")
    print(f"   Available candles: {len(df_features):,}")

    # Verify long-term features exist
    long_term_features = get_long_term_feature_list()
    existing_long_term = [f for f in long_term_features if f in df_features.columns]
    print(f"\n   Long-term features: {len(existing_long_term)}/{len(long_term_features)}")
    print()

    # Create labels
    print("="*80)
    print("Creating Labels")
    print("="*80)

    # Entry labels (trade outcome)
    df_labels = create_trade_outcome_labels(
        df_features,
        forward_periods=240,  # 20 hours
        profit_threshold=0.03,  # 3% leveraged profit
        loss_threshold=-0.015   # -1.5% leveraged loss
    )

    # Exit labels
    df_labels = create_exit_labels(
        df_labels,
        forward_periods=120  # 10 hours look ahead for exit
    )

    print()

    # Prepare feature columns (exclude OHLCV and labels)
    exclude_cols = ['open', 'high', 'low', 'close', 'volume', 'timestamp',
                    'long_outcome', 'short_outcome', 'long_exit', 'short_exit']
    feature_cols = [col for col in df_labels.columns if col not in exclude_cols]

    print(f"Feature columns: {len(feature_cols)}")
    print()

    # Remove rows with NaN labels (last forward_periods rows)
    df_clean = df_labels[:-240].copy()
    print(f"Clean samples (excluding last 240): {len(df_clean):,}")
    print()

    # Timestamp for saved models
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ========================================================================
    # 1. LONG Entry Model
    # ========================================================================
    print("\n" + "="*80)
    print("1/4: LONG ENTRY MODEL")
    print("="*80)

    X = df_clean[feature_cols]
    y = df_clean['long_outcome']

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    long_entry_model, long_entry_scaler, long_entry_features, long_entry_metrics = train_model(
        X_train, y_train, X_val, y_val,
        model_name="LONG Entry Model",
        class_weight='balanced'
    )

    # Save model
    model_file = MODELS_DIR / f"xgboost_long_entry_longterm_{timestamp}.pkl"
    with open(model_file, 'wb') as f:
        pickle.dump(long_entry_model, f)

    scaler_file = MODELS_DIR / f"scaler_long_entry_longterm_{timestamp}.pkl"
    with open(scaler_file, 'wb') as f:
        pickle.dump(long_entry_scaler, f)

    print(f"\n✅ Saved: {model_file.name}")
    print(f"✅ Saved: {scaler_file.name}")

    # ========================================================================
    # 2. SHORT Entry Model
    # ========================================================================
    print("\n" + "="*80)
    print("2/4: SHORT ENTRY MODEL")
    print("="*80)

    y = df_clean['short_outcome']

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    short_entry_model, short_entry_scaler, short_entry_features, short_entry_metrics = train_model(
        X_train, y_train, X_val, y_val,
        model_name="SHORT Entry Model",
        class_weight='balanced'
    )

    # Save model
    model_file = MODELS_DIR / f"xgboost_short_entry_longterm_{timestamp}.pkl"
    with open(model_file, 'wb') as f:
        pickle.dump(short_entry_model, f)

    scaler_file = MODELS_DIR / f"scaler_short_entry_longterm_{timestamp}.pkl"
    with open(scaler_file, 'wb') as f:
        pickle.dump(short_entry_scaler, f)

    print(f"\n✅ Saved: {model_file.name}")
    print(f"✅ Saved: {scaler_file.name}")

    # ========================================================================
    # 3. LONG Exit Model
    # ========================================================================
    print("\n" + "="*80)
    print("3/4: LONG EXIT MODEL")
    print("="*80)

    y = df_clean['long_exit']

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    long_exit_model, long_exit_scaler, long_exit_features, long_exit_metrics = train_model(
        X_train, y_train, X_val, y_val,
        model_name="LONG Exit Model",
        class_weight='balanced'
    )

    # Save model
    model_file = MODELS_DIR / f"xgboost_long_exit_longterm_{timestamp}.pkl"
    with open(model_file, 'wb') as f:
        pickle.dump(long_exit_model, f)

    scaler_file = MODELS_DIR / f"scaler_long_exit_longterm_{timestamp}.pkl"
    with open(scaler_file, 'wb') as f:
        pickle.dump(long_exit_scaler, f)

    print(f"\n✅ Saved: {model_file.name}")
    print(f"✅ Saved: {scaler_file.name}")

    # ========================================================================
    # 4. SHORT Exit Model
    # ========================================================================
    print("\n" + "="*80)
    print("4/4: SHORT EXIT MODEL")
    print("="*80)

    y = df_clean['short_exit']

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    short_exit_model, short_exit_scaler, short_exit_features, short_exit_metrics = train_model(
        X_train, y_train, X_val, y_val,
        model_name="SHORT Exit Model",
        class_weight='balanced'
    )

    # Save model
    model_file = MODELS_DIR / f"xgboost_short_exit_longterm_{timestamp}.pkl"
    with open(model_file, 'wb') as f:
        pickle.dump(short_exit_model, f)

    scaler_file = MODELS_DIR / f"scaler_short_exit_longterm_{timestamp}.pkl"
    with open(scaler_file, 'wb') as f:
        pickle.dump(short_exit_scaler, f)

    print(f"\n✅ Saved: {model_file.name}")
    print(f"✅ Saved: {scaler_file.name}")

    # ========================================================================
    # Summary
    # ========================================================================
    print("\n" + "="*80)
    print("TRAINING SUMMARY")
    print("="*80)
    print()

    print("Model Performance:")
    print(f"  LONG Entry  - Train AUC: {long_entry_metrics['train_auc']:.4f}, Val AUC: {long_entry_metrics['val_auc']:.4f}")
    print(f"  SHORT Entry - Train AUC: {short_entry_metrics['train_auc']:.4f}, Val AUC: {short_entry_metrics['val_auc']:.4f}")
    print(f"  LONG Exit   - Train AUC: {long_exit_metrics['train_auc']:.4f}, Val AUC: {long_exit_metrics['val_auc']:.4f}")
    print(f"  SHORT Exit  - Train AUC: {short_exit_metrics['train_auc']:.4f}, Val AUC: {short_exit_metrics['val_auc']:.4f}")
    print()

    print("Saved Models:")
    print(f"  ✅ xgboost_long_entry_longterm_{timestamp}.pkl")
    print(f"  ✅ xgboost_short_entry_longterm_{timestamp}.pkl")
    print(f"  ✅ xgboost_long_exit_longterm_{timestamp}.pkl")
    print(f"  ✅ xgboost_short_exit_longterm_{timestamp}.pkl")
    print()

    print("Saved Scalers:")
    print(f"  ✅ scaler_long_entry_longterm_{timestamp}.pkl")
    print(f"  ✅ scaler_short_entry_longterm_{timestamp}.pkl")
    print(f"  ✅ scaler_long_exit_longterm_{timestamp}.pkl")
    print(f"  ✅ scaler_short_exit_longterm_{timestamp}.pkl")
    print()

    print("Features:")
    print(f"  Total features: {len(feature_cols)}")
    print(f"  Baseline features: ~107")
    print(f"  Long-term features: {len(existing_long_term)}")
    print()

    print("="*80)
    print("TRAINING COMPLETE")
    print("="*80)
    print()

    print("Next Steps:")
    print("  1. Run backtest with new models")
    print("  2. Compare performance vs baseline models")
    print("  3. Verify long-term features improving predictions")
    print()

if __name__ == "__main__":
    main()
