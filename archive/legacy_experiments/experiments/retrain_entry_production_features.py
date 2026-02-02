"""
Retrain Entry Models - Production Features (85 LONG, 79 SHORT)
================================================================

Uses Production's proven feature set to fix Phase 1 failures:
  - LONG: 85 features (includes VWAP/VP + Candlesticks)
  - SHORT: 79 features (includes VWAP/VP + Volatility)
  - Walk-Forward + Filtered + Decoupled methodology

Root Cause Fixed:
  Phase 1 removed 71% LONG / 99% SHORT features
  Missing: VWAP/VP (39), Candlesticks (5), Volatility filters
  Production features proven: 73.86% WR, +38.04% return

Created: 2025-10-31
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import pickle
import joblib
from datetime import datetime
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import precision_score, recall_score, f1_score

# Configuration
ENTRY_THRESHOLD = 0.75
ML_EXIT_THRESHOLD = 0.75
LEVERAGE = 4
EMERGENCY_STOP_LOSS = -0.03
EMERGENCY_MAX_HOLD = 120

DATA_DIR = PROJECT_ROOT / "data" / "features"
MODELS_DIR = PROJECT_ROOT / "models"

print("="*80)
print("PRODUCTION FEATURES ENTRY MODEL TRAINING")
print("="*80)
print()
print("Feature Set:")
print("  ✅ LONG: 85 Production features (VWAP/VP + Candlesticks)")
print("  ✅ SHORT: 79 Production features (VWAP/VP + Volatility)")
print()
print("Methodology:")
print("  ✅ Walk-Forward Validation (5-fold)")
print("  ✅ Filtered Simulation (heuristic pre-filtering)")
print("  ✅ Decoupled Training (rule-based Exit labels)")
print()
print(f"Configuration:")
print(f"  Entry Threshold: {ENTRY_THRESHOLD}")
print(f"  ML Exit Threshold: {ML_EXIT_THRESHOLD}")
print(f"  Emergency Stop Loss: {EMERGENCY_STOP_LOSS}")
print(f"  Emergency Max Hold: {EMERGENCY_MAX_HOLD} candles")
print()

# Load Full Features Dataset
print("-"*80)
print("STEP 1: Loading Full Features Dataset")
print("-"*80)

df = pd.read_csv(DATA_DIR / "BTCUSDT_5m_features.csv")
print(f"✅ Loaded {len(df):,} candles with {len(df.columns)} features")
print(f"   Date range: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")
print()

# Load Production Feature Lists
print("-"*80)
print("STEP 2: Loading Production Feature Lists")
print("-"*80)

# Production LONG Entry: 85 features (includes VWAP/VP + Candlesticks)
with open(MODELS_DIR / "xgboost_long_entry_walkforward_decoupled_20251027_194313_features.txt", 'r') as f:
    long_entry_features = [line.strip() for line in f]

# Production SHORT Entry: 79 features (includes VWAP/VP + Volatility)
with open(MODELS_DIR / "xgboost_short_entry_walkforward_decoupled_20251027_194313_features.txt", 'r') as f:
    short_entry_features = [line.strip() for line in f]

print(f"  ✅ LONG Entry: {len(long_entry_features)} features (Production)")
print(f"  ✅ SHORT Entry: {len(short_entry_features)} features (Production)")
print()

# ==============================================================================
# OPTION A: Filtered Simulation (Candidate Pre-Filtering)
# ==============================================================================

def filter_entry_candidates(df, side):
    """
    Pre-filter candidates using simple heuristics
    Reduces computational load by ~92% (Production methodology)
    """
    print(f"\n🔍 Filtering {side} Entry Candidates...")

    mask = pd.Series(True, index=df.index)

    # Volume filter
    if 'volume' in df.columns:
        volume_filter = df['volume'] > df['volume'].rolling(20).mean()
        mask = mask & volume_filter

    # RSI filter
    if 'rsi' in df.columns:
        if side == 'LONG':
            rsi_filter = (df['rsi'] > 30) & (df['rsi'] < 70)
        else:  # SHORT
            rsi_filter = (df['rsi'] > 40) & (df['rsi'] < 80)
        mask = mask & rsi_filter

    # MACD filter (if available)
    if 'macd' in df.columns and 'macd_signal' in df.columns:
        if side == 'LONG':
            macd_filter = df['macd'] > df['macd_signal']
        else:  # SHORT
            macd_filter = df['macd'] < df['macd_signal']
        mask = mask & macd_filter

    # Trend filter (if available)
    if 'sma_20' in df.columns and 'sma_50' in df.columns:
        if side == 'LONG':
            trend_filter = df['sma_20'] > df['sma_50']
        else:  # SHORT
            trend_filter = df['sma_20'] < df['sma_50']
        mask = mask & trend_filter

    candidates = df[mask].copy()
    reduction_pct = (1 - len(candidates) / len(df)) * 100

    print(f"  Before: {len(df):,} candles")
    print(f"  After:  {len(candidates):,} candidates")
    print(f"  Reduction: {reduction_pct:.1f}%")

    return candidates

# ==============================================================================
# OPTION C: Decoupled Training (Rule-Based Exit Labels)
# ==============================================================================

def simulate_trade_outcomes(df_candidates, side):
    """
    Generate Entry labels from rule-based Exit simulation
    Decoupled: No dependency on ML Exit models
    """
    print(f"\n📊 Simulating {side} Trade Outcomes...")

    labels = []
    df = df_candidates.reset_index(drop=True)

    for idx in range(len(df)):
        if idx + EMERGENCY_MAX_HOLD >= len(df):
            labels.append(0)  # Not enough future data
            continue

        entry_price = df.loc[idx, 'close']

        # Look ahead for exit conditions
        future = df.iloc[idx+1:idx+1+EMERGENCY_MAX_HOLD].copy()

        # Calculate leveraged P&L for each future candle
        if side == 'LONG':
            future['leveraged_pnl'] = ((future['close'] - entry_price) / entry_price) * LEVERAGE
        else:  # SHORT
            future['leveraged_pnl'] = ((entry_price - future['close']) / entry_price) * LEVERAGE

        # Find first exit condition
        exit_idx = None
        exit_reason = None

        # Check Stop Loss
        sl_hit = future[future['leveraged_pnl'] <= EMERGENCY_STOP_LOSS]
        if not sl_hit.empty:
            exit_idx = sl_hit.index[0] - idx - 1
            exit_reason = 'SL'

        # Check profitable exit (decoupled rule: > 2% leveraged profit)
        if exit_idx is None:
            profit_exit = future[future['leveraged_pnl'] >= 0.02]
            if not profit_exit.empty:
                exit_idx = profit_exit.index[0] - idx - 1
                exit_reason = 'PROFIT'

        # Max Hold
        if exit_idx is None:
            exit_idx = len(future) - 1
            exit_reason = 'MAX_HOLD'

        # Label: 1 if profitable exit, 0 otherwise
        final_pnl = future.iloc[exit_idx]['leveraged_pnl']
        label = 1 if final_pnl > 0 else 0

        labels.append(label)

    positive_rate = sum(labels) / len(labels) * 100 if labels else 0
    print(f"  Total candidates: {len(labels):,}")
    print(f"  Positive labels: {sum(labels):,} ({positive_rate:.1f}%)")

    return np.array(labels)

# ==============================================================================
# OPTION B: Walk-Forward Validation
# ==============================================================================

def train_walk_forward(df_candidates, labels, features, side):
    """
    Walk-Forward Cross-Validation (5 folds)
    Each fold trains on past data, validates on future data
    """
    print(f"\n📈 Walk-Forward Training ({side})...")

    # Prepare features
    X = df_candidates[features].values
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
        y_pred_thresh = (y_proba >= ENTRY_THRESHOLD).astype(int)

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
print("TRAINING LONG ENTRY MODEL (Production Features)")
print("="*80)

# Filter candidates
long_candidates = filter_entry_candidates(df, 'LONG')

# Generate labels (decoupled)
long_labels = simulate_trade_outcomes(long_candidates, 'LONG')

# Train with Walk-Forward
long_model, long_scaler, long_folds = train_walk_forward(
    long_candidates, long_labels, long_entry_features, 'LONG'
)

# Save LONG model
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
long_model_path = MODELS_DIR / f"xgboost_long_entry_production_features_{timestamp}.pkl"
long_scaler_path = MODELS_DIR / f"xgboost_long_entry_production_features_{timestamp}_scaler.pkl"
long_features_path = MODELS_DIR / f"xgboost_long_entry_production_features_{timestamp}_features.txt"

with open(long_model_path, 'wb') as f:
    pickle.dump(long_model, f)
with open(long_scaler_path, 'wb') as f:
    pickle.dump(long_scaler, f)
with open(long_features_path, 'w') as f:
    f.write('\n'.join(long_entry_features))

print(f"\n✅ LONG Model Saved: {long_model_path.name}")

# ==============================================================================

print("\n" + "="*80)
print("TRAINING SHORT ENTRY MODEL (Production Features)")
print("="*80)

# Filter candidates
short_candidates = filter_entry_candidates(df, 'SHORT')

# Generate labels (decoupled)
short_labels = simulate_trade_outcomes(short_candidates, 'SHORT')

# Train with Walk-Forward
short_model, short_scaler, short_folds = train_walk_forward(
    short_candidates, short_labels, short_entry_features, 'SHORT'
)

# Save SHORT model
short_model_path = MODELS_DIR / f"xgboost_short_entry_production_features_{timestamp}.pkl"
short_scaler_path = MODELS_DIR / f"xgboost_short_entry_production_features_{timestamp}_scaler.pkl"
short_features_path = MODELS_DIR / f"xgboost_short_entry_production_features_{timestamp}_features.txt"

with open(short_model_path, 'wb') as f:
    pickle.dump(short_model, f)
with open(short_scaler_path, 'wb') as f:
    pickle.dump(short_scaler, f)
with open(short_features_path, 'w') as f:
    f.write('\n'.join(short_entry_features))

print(f"\n✅ SHORT Model Saved: {short_model_path.name}")

# ==============================================================================
# SUMMARY
# ==============================================================================

print("\n" + "="*80)
print("TRAINING SUMMARY")
print("="*80)

print(f"\nLONG Entry Model:")
print(f"  Features: {len(long_entry_features)} (Production)")
print(f"  Candidates: {len(long_candidates):,}")
print(f"  Positive Labels: {long_labels.sum():,} ({long_labels.sum()/len(long_labels)*100:.1f}%)")
print(f"  Walk-Forward Folds: {len(long_folds)}")
print(f"  Avg Prediction Rate: {np.mean([f['prediction_rate'] for f in long_folds]):.2f}%")

print(f"\nSHORT Entry Model:")
print(f"  Features: {len(short_entry_features)} (Production)")
print(f"  Candidates: {len(short_candidates):,}")
print(f"  Positive Labels: {short_labels.sum():,} ({short_labels.sum()/len(short_labels)*100:.1f}%)")
print(f"  Walk-Forward Folds: {len(short_folds)}")
print(f"  Avg Prediction Rate: {np.mean([f['prediction_rate'] for f in short_folds]):.2f}%")

print(f"\nNext Step:")
print(f"  Run full 108-window backtest to validate performance")
print(f"  Expected: Win Rate should improve toward Production's 73.86%")

print("\n" + "="*80)
print("✅ TRAINING COMPLETE")
print("="*80)
