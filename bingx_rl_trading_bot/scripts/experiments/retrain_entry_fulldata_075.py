"""
Retrain Entry Models - FULL Dataset (Top-3 Weighted Ensemble)
==============================================================

Retrains Entry models using the COMPLETE dataset (all 104 days, 30,004 candles).

Strategy:
  - Data: Full dataset (Jul 14 - Oct 26, 2025) - 104 days
  - Method: Walk-Forward 5-Fold Cross-Validation
  - Output: Top-3 Weighted Ensemble

Rationale:
  - 30-day retraining produced non-functional models (0 trades)
  - Insufficient training data caused models to become overly conservative
  - Solution: Use complete dataset for robust model training

Created: 2025-10-30
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
print("ENTRY MODEL RETRAINING - FULL DATASET")
print("="*80)
print()
print("Strategy: Comprehensive training on complete dataset")
print("  ✅ Using ALL available data (104 days, ~30K candles)")
print("  ✅ Walk-Forward 5-Fold Cross-Validation")
print("  ✅ Top-3 Weighted Ensemble output")
print()
print(f"Configuration:")
print(f"  Entry Threshold: {ENTRY_THRESHOLD}")
print(f"  ML Exit Threshold: {ML_EXIT_THRESHOLD}")
print(f"  Emergency Stop Loss: {EMERGENCY_STOP_LOSS}")
print(f"  Emergency Max Hold: {EMERGENCY_MAX_HOLD} candles")
print()

# Load Full Features Dataset
print("-"*80)
print("STEP 1: Loading Complete Dataset")
print("-"*80)

df = pd.read_csv(DATA_DIR / "BTCUSDT_5m_features.csv")
df['timestamp'] = pd.to_datetime(df['timestamp'])

print(f"✅ Dataset loaded: {len(df):,} candles")
print(f"   Date range: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")
print(f"   Days: {(df['timestamp'].max() - df['timestamp'].min()).days}")
print()

# Load Entry Feature Lists
print("-"*80)
print("STEP 2: Loading Entry Feature Lists")
print("-"*80)

with open(MODELS_DIR / "xgboost_long_entry_enhanced_20251024_012445_features.txt", 'r') as f:
    long_entry_features = [line.strip() for line in f]

with open(MODELS_DIR / "xgboost_short_entry_enhanced_20251024_012445_features.txt", 'r') as f:
    short_entry_features = [line.strip() for line in f]

print(f"  ✅ LONG Entry: {len(long_entry_features)} features")
print(f"  ✅ SHORT Entry: {len(short_entry_features)} features")
print()

# ==============================================================================
# Helper Functions (same as before)
# ==============================================================================

def filter_entry_candidates(df, side):
    """Pre-filter candidates using simple heuristics"""
    mask = pd.Series(True, index=df.index)

    # Volume filter
    if 'volume' in df.columns:
        volume_filter = df['volume'] > df['volume'].rolling(20).mean()
        mask = mask & volume_filter

    # RSI filter
    if 'rsi' in df.columns:
        rsi_filter = (df['rsi'] > 30) & (df['rsi'] < 70)
        mask = mask & rsi_filter

    if side == 'LONG':
        if 'ma_50' in df.columns:
            trend_filter = df['close'] > df['ma_50']
            mask = mask & trend_filter
        if 'macd' in df.columns and 'macd_signal' in df.columns:
            macd_filter = df['macd'] > df['macd_signal']
            mask = mask & macd_filter
    else:
        if 'ma_50' in df.columns:
            trend_filter = df['close'] < df['ma_50']
            mask = mask & trend_filter
        if 'macd' in df.columns and 'macd_signal' in df.columns:
            macd_filter = df['macd'] < df['macd_signal']
            mask = mask & macd_filter

    valid_range = (df.index >= 100) & (df.index < len(df) - EMERGENCY_MAX_HOLD)
    mask = mask & valid_range

    candidates = df[mask].index.tolist()
    return candidates

def label_exits_rule_based(df, side):
    """Simple rule-based exit labels"""
    exit_labels = np.zeros(len(df))

    for i in range(100, len(df) - EMERGENCY_MAX_HOLD):
        entry_price = df['close'].iloc[i]

        for j in range(1, EMERGENCY_MAX_HOLD + 1):
            if i + j >= len(df):
                break

            current_price = df['close'].iloc[i + j]

            if side == 'LONG':
                pnl = (current_price - entry_price) / entry_price
            else:
                pnl = (entry_price - current_price) / entry_price

            leveraged_pnl = pnl * LEVERAGE

            if leveraged_pnl > 0.02 and j < 60:
                exit_labels[i + j] = 1
                break
            elif leveraged_pnl <= EMERGENCY_STOP_LOSS:
                break

    return exit_labels

def train_exit_model_fold(df, side):
    """Train Exit model using rule-based labels"""
    exit_labels = label_exits_rule_based(df, side)

    EXIT_FEATURES = [
        'close', 'volume', 'volume_surge', 'price_acceleration',
        'ma_20', 'ma_50', 'price_vs_ma20', 'price_vs_ma50',
        'volatility_20',
        'rsi', 'rsi_slope', 'rsi_overbought', 'rsi_oversold', 'rsi_divergence',
        'macd', 'macd_signal', 'macd_histogram', 'macd_histogram_slope',
        'macd_crossover', 'macd_crossunder',
        'bb_upper', 'bb_lower', 'bb_position',
        'higher_high', 'near_support'
    ]

    available_features = [f for f in EXIT_FEATURES if f in df.columns]

    X = df[available_features].values
    y = exit_labels

    mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
    X = X[mask]
    y = y[mask]

    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=(y == 0).sum() / max((y == 1).sum(), 1),
        random_state=42,
        eval_metric='logloss'
    )

    model.fit(X_scaled, y, verbose=False)

    return model, scaler, available_features

def simulate_single_trade(df, entry_idx, exit_model, exit_scaler, exit_features, side):
    """Simulate a single trade using the Exit model"""
    entry_price = df['close'].iloc[entry_idx]

    max_hold_end = min(entry_idx + EMERGENCY_MAX_HOLD + 1, len(df))
    prices = df['close'].iloc[entry_idx+1:max_hold_end].values

    if side == 'LONG':
        pnl_series = (prices - entry_price) / entry_price
    else:
        pnl_series = (entry_price - prices) / entry_price

    for i, pnl_pct in enumerate(pnl_series):
        current_idx = entry_idx + 1 + i
        hold_time = i + 1
        leveraged_pnl = pnl_pct * LEVERAGE

        if leveraged_pnl <= EMERGENCY_STOP_LOSS:
            return {
                'entry_idx': entry_idx,
                'pnl_pct': pnl_pct,
                'leveraged_pnl': leveraged_pnl,
                'hold_time': hold_time,
                'exit_reason': 'stop_loss'
            }

        try:
            exit_feat = df[exit_features].iloc[current_idx:current_idx+1].values
            if not np.isnan(exit_feat).any():
                exit_scaled = exit_scaler.transform(exit_feat)
                exit_prob = exit_model.predict_proba(exit_scaled)[0][1]

                if exit_prob >= ML_EXIT_THRESHOLD:
                    return {
                        'entry_idx': entry_idx,
                        'pnl_pct': pnl_pct,
                        'leveraged_pnl': leveraged_pnl,
                        'hold_time': hold_time,
                        'exit_reason': 'ml_exit'
                    }
        except:
            pass

        if hold_time >= EMERGENCY_MAX_HOLD:
            return {
                'entry_idx': entry_idx,
                'pnl_pct': pnl_pct,
                'leveraged_pnl': leveraged_pnl,
                'hold_time': hold_time,
                'exit_reason': 'max_hold'
            }

    final_pnl = pnl_series[-1] if len(pnl_series) > 0 else 0
    return {
        'entry_idx': entry_idx,
        'pnl_pct': final_pnl,
        'leveraged_pnl': final_pnl * LEVERAGE,
        'hold_time': len(pnl_series),
        'exit_reason': 'data_end'
    }

def label_entries_with_simulation(df, entry_candidates, exit_model, exit_scaler, exit_features, side):
    """Label entry points using trade simulation"""
    entry_labels = np.zeros(len(df))

    for entry_idx in entry_candidates:
        result = simulate_single_trade(df, entry_idx, exit_model, exit_scaler, exit_features, side)

        score = 0
        if result['leveraged_pnl'] > 0.02:
            score += 1
        if result['hold_time'] <= 60:
            score += 1
        if result['exit_reason'] == 'ml_exit':
            score += 1

        if score >= 2:
            entry_labels[entry_idx] = 1

    return entry_labels

# ==============================================================================
# MODIFIED: Save ALL Folds (not just best)
# ==============================================================================

def train_entry_model_walkforward_ensemble(df, entry_features, side):
    """
    Walk-Forward training with ALL folds saved for ensemble
    Returns: list of (model, scaler, features, score) for all valid folds
    """
    print(f"\n{'='*80}")
    print(f"Walk-Forward Ensemble Training: {side} Entry Model")
    print(f"{'='*80}")

    tscv = TimeSeriesSplit(n_splits=5)

    fold_results = []  # Store ALL folds

    for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(df)):
        print(f"\n{'─'*80}")
        print(f"FOLD {fold_idx+1}/5")
        print(f"{'─'*80}")
        print(f"  Train: {len(train_idx):,} samples (index {train_idx[0]}-{train_idx[-1]})")
        print(f"  Val: {len(val_idx):,} samples (index {val_idx[0]}-{val_idx[-1]})")

        df_train = df.iloc[train_idx].copy()
        df_val = df.iloc[val_idx].copy()

        # Train Exit model on THIS fold
        print(f"\n  Step 1: Training Exit model on Fold {fold_idx+1} data...")
        exit_model, exit_scaler, exit_features = train_exit_model_fold(df_train, side)

        # Filter entry candidates
        train_candidates = filter_entry_candidates(df_train, side)

        # Label entries
        entry_labels = label_entries_with_simulation(
            df_train, train_candidates, exit_model, exit_scaler, exit_features, side
        )

        # Train Entry model
        print(f"\n  Step 2: Training {side} Entry model on Fold {fold_idx+1}...")

        available_features = [f for f in entry_features if f in df_train.columns]
        X_train = df_train[available_features].values
        y_train = entry_labels

        mask = ~np.isnan(X_train).any(axis=1) & ~np.isnan(y_train)
        X_train = X_train[mask]
        y_train = y_train[mask]

        if (y_train == 1).sum() < 10:
            print(f"  ⚠️  Too few positive samples in fold {fold_idx+1}, skipping...")
            continue

        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)

        model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
            random_state=42,
            eval_metric='logloss'
        )

        model.fit(X_train_scaled, y_train, verbose=False)

        # Validate
        X_val = df_val[available_features].values
        mask_val = ~np.isnan(X_val).any(axis=1)
        X_val = X_val[mask_val]

        if len(X_val) > 0:
            X_val_scaled = scaler.transform(X_val)
            y_pred = model.predict(X_val_scaled)

            pred_rate = y_pred.mean()
            print(f"  Validation prediction rate: {pred_rate*100:.2f}%")

            # Save THIS fold result
            fold_results.append({
                'fold_idx': fold_idx + 1,
                'model': model,
                'scaler': scaler,
                'features': available_features,
                'score': pred_rate
            })

        print(f"  ✅ Fold {fold_idx+1} complete")

    if len(fold_results) == 0:
        raise ValueError(f"No valid models trained for {side}")

    # Sort by score (descending)
    fold_results = sorted(fold_results, key=lambda x: x['score'], reverse=True)

    print(f"\n{'='*80}")
    print(f"ALL FOLDS TRAINED - Scores:")
    print(f"{'='*80}")
    for result in fold_results:
        print(f"  Fold {result['fold_idx']}: {result['score']*100:.2f}%")

    return fold_results

# ==============================================================================
# Main Training Pipeline
# ==============================================================================

print("-"*80)
print("STEP 3: Training LONG Entry Model (Walk-Forward Ensemble)")
print("-"*80)

long_fold_results = train_entry_model_walkforward_ensemble(
    df, long_entry_features, 'LONG'
)

print("\n" + "-"*80)
print("STEP 4: Training SHORT Entry Model (Walk-Forward Ensemble)")
print("-"*80)

short_fold_results = train_entry_model_walkforward_ensemble(
    df, short_entry_features, 'SHORT'
)

# Save ALL Fold Models
print("\n" + "="*80)
print("STEP 5: Saving ALL Fold Models")
print("="*80)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# LONG Entry - Save all folds
for result in long_fold_results:
    fold_idx = result['fold_idx']
    long_path = MODELS_DIR / f"xgboost_long_entry_ensemble_fold{fold_idx}_{timestamp}"

    with open(f"{long_path}.pkl", 'wb') as f:
        pickle.dump(result['model'], f)
    joblib.dump(result['scaler'], f"{long_path}_scaler.pkl")
    with open(f"{long_path}_features.txt", 'w') as f:
        f.write('\n'.join(result['features']))

    print(f"✅ LONG Entry Fold {fold_idx}: {long_path.name}.pkl (score: {result['score']*100:.2f}%)")

# Save LONG metadata (fold scores for weighting)
long_meta_path = MODELS_DIR / f"xgboost_long_entry_ensemble_meta_{timestamp}.pkl"
long_meta = {
    'fold_scores': [r['score'] for r in long_fold_results],
    'fold_indices': [r['fold_idx'] for r in long_fold_results],
    'timestamp': timestamp,
    'num_features': len(long_fold_results[0]['features'])
}
with open(long_meta_path, 'wb') as f:
    pickle.dump(long_meta, f)
print(f"✅ LONG Metadata: {long_meta_path.name}")

# SHORT Entry - Save all folds
for result in short_fold_results:
    fold_idx = result['fold_idx']
    short_path = MODELS_DIR / f"xgboost_short_entry_ensemble_fold{fold_idx}_{timestamp}"

    with open(f"{short_path}.pkl", 'wb') as f:
        pickle.dump(result['model'], f)
    joblib.dump(result['scaler'], f"{short_path}_scaler.pkl")
    with open(f"{short_path}_features.txt", 'w') as f:
        f.write('\n'.join(result['features']))

    print(f"✅ SHORT Entry Fold {fold_idx}: {short_path.name}.pkl (score: {result['score']*100:.2f}%)")

# Save SHORT metadata
short_meta_path = MODELS_DIR / f"xgboost_short_entry_ensemble_meta_{timestamp}.pkl"
short_meta = {
    'fold_scores': [r['score'] for r in short_fold_results],
    'fold_indices': [r['fold_idx'] for r in short_fold_results],
    'timestamp': timestamp,
    'num_features': len(short_fold_results[0]['features'])
}
with open(short_meta_path, 'wb') as f:
    pickle.dump(short_meta, f)
print(f"✅ SHORT Metadata: {short_meta_path.name}")

print("\n" + "="*80)
print("ENSEMBLE TRAINING COMPLETE")
print("="*80)
print()
print("Saved Files:")
print(f"  LONG: {len(long_fold_results)} fold models + metadata")
print(f"  SHORT: {len(short_fold_results)} fold models + metadata")
print()
print("Next Steps:")
print("  1. Implement Top-3 Weighted Ensemble in production bot")
print("  2. Run backtest comparison (Best Fold vs Top-3 Ensemble)")
print("  3. Deploy if ensemble performance > Best Fold + 5%")
print()