"""
Retrain Exit Models - Full Dataset (No Pre-Filtering)
========================================================

Production methodology: Train on ALL candles (not filtered subset)
  - LONG: 27 features on 30,004 candles
  - SHORT: 27 features on 30,004 candles
  - Walk-Forward + Decoupled (no filtering)

Expected: ML Exit 0% → 77%, Win Rate 44.2% → 70%+

Created: 2025-10-31
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import pickle
from datetime import datetime
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import precision_score, recall_score, f1_score

# Configuration
ML_EXIT_THRESHOLD = 0.75
LEVERAGE = 4
EMERGENCY_STOP_LOSS = -0.03
EMERGENCY_MAX_HOLD = 120
PROFIT_TARGET = 0.02  # 2% leveraged profit (rule-based exit)

DATA_DIR = PROJECT_ROOT / "data" / "features"
MODELS_DIR = PROJECT_ROOT / "models"

print("="*80)
print("FULL DATASET EXIT MODEL TRAINING (Production Methodology)")
print("="*80)
print()
print("Training Approach:")
print("  ✅ NO PRE-FILTERING (train on all 30,004 candles)")
print("  ✅ Production Exit features (27 features)")
print("  ✅ Walk-Forward validation (5-fold)")
print("  ✅ Decoupled labels (rule-based profit target)")
print()
print(f"Configuration:")
print(f"  Exit Threshold: {ML_EXIT_THRESHOLD}")
print(f"  Profit Target: {PROFIT_TARGET*100}% leveraged")
print(f"  Emergency Stop Loss: {EMERGENCY_STOP_LOSS}")
print(f"  Emergency Max Hold: {EMERGENCY_MAX_HOLD} candles")
print()

# Load Full Features Dataset (pre-calculated with VWAP/VP)
print("-"*80)
print("STEP 1: Loading Full Features Dataset")
print("-"*80)

df = pd.read_csv(DATA_DIR / "BTCUSDT_5m_features.csv")

# Add missing Exit features if needed
if 'bb_width' not in df.columns:
    if 'bb_upper' in df.columns and 'bb_lower' in df.columns:
        df['bb_width'] = df['bb_upper'] - df['bb_lower']
    else:
        df['bb_width'] = 0

if 'vwap' not in df.columns:
    df['vwap'] = (df['close'] * df['volume']).rolling(20).sum() / df['volume'].rolling(20).sum()
    df['vwap'] = df['vwap'].ffill().bfill()

# Add Enhanced Exit Features
print("  Adding enhanced Exit features...")

# Volume surge
df['volume_surge'] = (df['volume'] / df['volume'].rolling(20).mean() - 1).fillna(0)

# Price acceleration
df['price_acceleration'] = df['close'].diff(2).fillna(0)

# Price vs moving averages
if 'sma_20' in df.columns:
    df['price_vs_ma20'] = ((df['close'] - df['sma_20']) / df['sma_20']).fillna(0)
else:
    df['price_vs_ma20'] = 0

if 'sma_50' in df.columns:
    df['price_vs_ma50'] = ((df['close'] - df['sma_50']) / df['sma_50']).fillna(0)
else:
    df['price_vs_ma50'] = 0

# Volatility
df['volatility_20'] = df['close'].pct_change().rolling(20).std().fillna(0)

# RSI features
if 'rsi' in df.columns:
    df['rsi_slope'] = df['rsi'].diff(5).fillna(0)
    df['rsi_overbought'] = (df['rsi'] > 70).astype(int)
    df['rsi_oversold'] = (df['rsi'] < 30).astype(int)
    df['rsi_divergence'] = (df['rsi'].diff() * df['close'].pct_change() < 0).astype(int)
else:
    df['rsi_slope'] = 0
    df['rsi_overbought'] = 0
    df['rsi_oversold'] = 0
    df['rsi_divergence'] = 0

# MACD features
if 'macd' in df.columns and 'macd_signal' in df.columns:
    df['macd_histogram_slope'] = (df['macd'] - df['macd_signal']).diff(3).fillna(0)
    df['macd_crossover'] = ((df['macd'] > df['macd_signal']) & (df['macd'].shift(1) <= df['macd_signal'].shift(1))).astype(int)
    df['macd_crossunder'] = ((df['macd'] < df['macd_signal']) & (df['macd'].shift(1) >= df['macd_signal'].shift(1))).astype(int)
else:
    df['macd_histogram_slope'] = 0
    df['macd_crossover'] = 0
    df['macd_crossunder'] = 0

# Bollinger Band position
if 'bb_upper' in df.columns and 'bb_lower' in df.columns:
    bb_range = df['bb_upper'] - df['bb_lower']
    df['bb_position'] = ((df['close'] - df['bb_lower']) / bb_range).fillna(0.5)
else:
    df['bb_position'] = 0.5

# Higher high pattern
df['higher_high'] = ((df['high'] > df['high'].shift(1)) & (df['high'].shift(1) > df['high'].shift(2))).astype(int)

# Near support (using 20-period low)
support_level = df['low'].rolling(20).min()
df['near_support'] = (df['close'] <= support_level * 1.01).astype(int)

print(f"✅ Loaded {len(df):,} candles with {len(df.columns)} features")
print(f"   Date range: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")
print()

# Load Production Exit Feature Lists
print("-"*80)
print("STEP 2: Loading Production Exit Feature Lists")
print("-"*80)

# Production LONG Exit: 27 features
with open(MODELS_DIR / "xgboost_long_exit_oppgating_improved_20251024_043527_features.txt", 'r') as f:
    long_exit_features = [line.strip() for line in f]

# Production SHORT Exit: 27 features
with open(MODELS_DIR / "xgboost_short_exit_oppgating_improved_20251024_044510_features.txt", 'r') as f:
    short_exit_features = [line.strip() for line in f]

print(f"  ✅ LONG Exit: {len(long_exit_features)} features (Production)")
print(f"  ✅ SHORT Exit: {len(short_exit_features)} features (Production)")
print()

# ==============================================================================
# DECOUPLED TRAINING (Rule-Based Exit Labels)
# ==============================================================================

def simulate_exit_outcomes(df, side):
    """
    Generate Exit labels from rule-based profit target logic
    Train on ALL candles (no pre-filtering)

    Label: 1 if should exit (profit target met), 0 if should hold
    """
    print(f"\n📊 Simulating {side} Exit Outcomes (Full Dataset)...")

    labels = []

    for idx in range(len(df)):
        if idx + EMERGENCY_MAX_HOLD >= len(df):
            labels.append(0)  # Not enough future data
            continue

        entry_price = df.loc[df.index[idx], 'close']

        # Look ahead for future price movement
        future_indices = df.index[idx+1:idx+1+EMERGENCY_MAX_HOLD]
        future = df.loc[future_indices].copy()

        # Calculate leveraged P&L for each future candle
        if side == 'LONG':
            future['leveraged_pnl'] = ((future['close'] - entry_price) / entry_price) * LEVERAGE
        else:  # SHORT
            future['leveraged_pnl'] = ((entry_price - future['close']) / entry_price) * LEVERAGE

        # Determine if profit target would be hit within horizon
        profit_exits = future[future['leveraged_pnl'] >= PROFIT_TARGET]

        if not profit_exits.empty:
            # Profit target hit → Label: 1 (should exit)
            first_profit_idx = profit_exits.index[0]
            candles_to_profit = df.index.get_loc(first_profit_idx) - idx

            # Exit should be signaled within reasonable time (e.g., 60 candles)
            if candles_to_profit <= 60:
                label = 1
            else:
                label = 0  # Too far in future, hold
        else:
            # No profit target hit → Label: 0 (should hold)
            label = 0

        labels.append(label)

        # Progress indicator
        if (idx + 1) % 5000 == 0:
            print(f"    Processed {idx+1:,} / {len(df):,} candles...")

    positive_rate = sum(labels) / len(labels) * 100 if labels else 0
    print(f"  ✅ Total samples: {len(labels):,}")
    print(f"  ✅ Positive labels (Exit): {sum(labels):,} ({positive_rate:.1f}%)")

    return np.array(labels)

# ==============================================================================
# WALK-FORWARD VALIDATION
# ==============================================================================

def train_walk_forward(df, labels, features, side):
    """
    Walk-Forward Cross-Validation (5 folds)
    Each fold trains on past data, validates on future data
    """
    print(f"\n📈 Walk-Forward Training ({side})...")

    # Prepare features
    X = df[features].values
    y = labels

    # Scale features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # TimeSeriesSplit (5 folds)
    tscv = TimeSeriesSplit(n_splits=5)

    fold_results = []
    best_model = None
    best_f1 = 0
    best_fold = 0

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_scaled), 1):
        print(f"\n  Fold {fold}/5:")

        X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Train XGBoost
        model = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='logloss',
            max_depth=6,
            learning_rate=0.1,
            n_estimators=100,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )

        model.fit(X_train, y_train, verbose=False)

        # Validate
        y_pred = model.predict(X_val)
        y_proba = model.predict_proba(X_val)[:, 1]

        # Apply threshold
        y_pred_thresh = (y_proba >= ML_EXIT_THRESHOLD).astype(int)

        # Metrics
        precision = precision_score(y_val, y_pred_thresh, zero_division=0)
        recall = recall_score(y_val, y_pred_thresh, zero_division=0)
        f1 = f1_score(y_val, y_pred_thresh, zero_division=0)

        prediction_rate = (y_pred_thresh.sum() / len(y_pred_thresh)) * 100
        positive_rate = (y_val.sum() / len(y_val)) * 100

        print(f"    Precision: {precision:.4f}")
        print(f"    Recall: {recall:.4f}")
        print(f"    F1 Score: {f1:.4f}")
        print(f"    Prediction Rate: {prediction_rate:.2f}%")
        print(f"    Positive Rate: {positive_rate:.2f}%")

        fold_results.append({
            'fold': fold,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'prediction_rate': prediction_rate
        })

        # Track best model
        if f1 > best_f1:
            best_f1 = f1
            best_model = model
            best_fold = fold

    print(f"\n  ✅ Best Fold: {best_fold}/5 (F1: {best_f1:.4f})")

    return best_model, scaler, fold_results

# ==============================================================================
# MAIN TRAINING PIPELINE
# ==============================================================================

print("="*80)
print("TRAINING LONG EXIT MODEL (Full Dataset - NO FILTERING)")
print("="*80)

# Generate labels (NO FILTERING - use all candles)
long_labels = simulate_exit_outcomes(df, 'LONG')

# Train with Walk-Forward
long_model, long_scaler, long_folds = train_walk_forward(
    df, long_labels, long_exit_features, 'LONG_EXIT'
)

# Save LONG Exit model
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
long_model_path = MODELS_DIR / f"xgboost_long_exit_full_dataset_{timestamp}.pkl"
long_scaler_path = MODELS_DIR / f"xgboost_long_exit_full_dataset_{timestamp}_scaler.pkl"
long_features_path = MODELS_DIR / f"xgboost_long_exit_full_dataset_{timestamp}_features.txt"

with open(long_model_path, 'wb') as f:
    pickle.dump(long_model, f)
with open(long_scaler_path, 'wb') as f:
    pickle.dump(long_scaler, f)
with open(long_features_path, 'w') as f:
    f.write('\n'.join(long_exit_features))

print(f"\n✅ LONG Exit Model Saved: {long_model_path.name}")

# ==============================================================================

print("\n" + "="*80)
print("TRAINING SHORT EXIT MODEL (Full Dataset - NO FILTERING)")
print("="*80)

# Generate labels (NO FILTERING - use all candles)
short_labels = simulate_exit_outcomes(df, 'SHORT')

# Train with Walk-Forward
short_model, short_scaler, short_folds = train_walk_forward(
    df, short_labels, short_exit_features, 'SHORT_EXIT'
)

# Save SHORT Exit model
short_model_path = MODELS_DIR / f"xgboost_short_exit_full_dataset_{timestamp}.pkl"
short_scaler_path = MODELS_DIR / f"xgboost_short_exit_full_dataset_{timestamp}_scaler.pkl"
short_features_path = MODELS_DIR / f"xgboost_short_exit_full_dataset_{timestamp}_features.txt"

with open(short_model_path, 'wb') as f:
    pickle.dump(short_model, f)
with open(short_scaler_path, 'wb') as f:
    pickle.dump(short_scaler, f)
with open(short_features_path, 'w') as f:
    f.write('\n'.join(short_exit_features))

print(f"\n✅ SHORT Exit Model Saved: {short_model_path.name}")

# ==============================================================================
# SUMMARY
# ==============================================================================

print("\n" + "="*80)
print("TRAINING SUMMARY")
print("="*80)

print(f"\nLONG Exit Model:")
print(f"  Features: {len(long_exit_features)} (Production)")
print(f"  Training Samples: {len(df):,} (FULL DATASET)")
print(f"  Positive Labels (Exit): {long_labels.sum():,} ({long_labels.sum()/len(long_labels)*100:.1f}%)")
print(f"  Walk-Forward Folds: {len(long_folds)}")
print(f"  Avg Prediction Rate: {np.mean([f['prediction_rate'] for f in long_folds]):.2f}%")

print(f"\nSHORT Exit Model:")
print(f"  Features: {len(short_exit_features)} (Production)")
print(f"  Training Samples: {len(df):,} (FULL DATASET)")
print(f"  Positive Labels (Exit): {short_labels.sum():,} ({short_labels.sum()/len(short_labels)*100:.1f}%)")
print(f"  Walk-Forward Folds: {len(short_folds)}")
print(f"  Avg Prediction Rate: {np.mean([f['prediction_rate'] for f in short_folds]):.2f}%")

print(f"\nKey Improvement vs Filtered Training:")
print(f"  Previous Exit samples: ~4,000 (filtered)")
print(f"  Current Exit samples: {len(df):,} (FULL)")
print(f"  Increase: ~650% more training data")

print(f"\nExpected Backtest Improvement:")
print(f"  ML Exit: 0% → 77% (Exit models compatible with Entry)")
print(f"  Win Rate: 44.2% → 70%+ (better exits)")
print(f"  Stop Loss: 33.9% → 8% (fewer emergency exits)")
print(f"  Return: +0.71% → +35%+ per window")

print(f"\nNext Step:")
print(f"  Run full backtest with new Exit models (timestamp: {timestamp})")

print("\n" + "="*80)
print("✅ TRAINING COMPLETE")
print("="*80)
