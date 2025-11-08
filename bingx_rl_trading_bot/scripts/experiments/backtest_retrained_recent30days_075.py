"""
Backtest Retrained Entry Models - Recent 30 Days Validation
============================================================

Validates Entry models retrained on recent 30 days (timestamp 20251030_003126).

Comparison:
  - Option A: Best Fold only (Fold 5 for both LONG and SHORT)
  - Option B: Top-3 Weighted Ensemble (Folds 5, 3, 2 for LONG; Folds 5, 2, 3 for SHORT)

Period: Recent 30 days (Sep 26 - Oct 26, 2025)
Models: 20251030_003126

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
from datetime import datetime, timedelta

# Configuration
ENTRY_THRESHOLD = 0.75
ML_EXIT_THRESHOLD = 0.75
LEVERAGE = 4
EMERGENCY_STOP_LOSS = -0.03
EMERGENCY_MAX_HOLD = 120
INITIAL_BALANCE = 10000
MIN_POSITION_PCT = 0.20
MAX_POSITION_PCT = 0.95
TAKER_FEE = 0.0005

DATA_DIR = PROJECT_ROOT / "data" / "features"
CSV_FILE = DATA_DIR / "BTCUSDT_5m_features_exit_ready.csv"
MODELS_DIR = PROJECT_ROOT / "models"

# Model timestamp
MODEL_TIMESTAMP = "20251030_003126"

print("="*80)
print("BACKTEST: RETRAINED MODELS (RECENT 30 DAYS)")
print("="*80)
print()
print(f"Models: {MODEL_TIMESTAMP}")
print(f"Period: Recent 30 days (Sep 26 - Oct 26, 2025)")
print()

# Load data
print("Loading data...")
df = pd.read_csv(CSV_FILE)
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Filter to recent 30 days
cutoff_date = df['timestamp'].max() - pd.Timedelta(days=30)
df = df[df['timestamp'] >= cutoff_date].copy()
df = df.sort_values('timestamp').reset_index(drop=True)

print(f"✅ Loaded {len(df):,} candles")
print(f"   Period: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")
print(f"   Days: {(df['timestamp'].max() - df['timestamp'].min()).days}")
print()

# Load Exit models
print("Loading Exit models...")
long_exit_model_path = MODELS_DIR / "xgboost_long_exit_threshold_075_20251027_190512.pkl"
with open(long_exit_model_path, 'rb') as f:
    long_exit_model = pickle.load(f)

long_exit_scaler_path = MODELS_DIR / "xgboost_long_exit_threshold_075_20251027_190512_scaler.pkl"
long_exit_scaler = joblib.load(long_exit_scaler_path)

long_exit_features_path = MODELS_DIR / "xgboost_long_exit_threshold_075_20251027_190512_features.txt"
with open(long_exit_features_path, 'r') as f:
    long_exit_features = [line.strip() for line in f.readlines() if line.strip()]

short_exit_model_path = MODELS_DIR / "xgboost_short_exit_threshold_075_20251027_190512.pkl"
with open(short_exit_model_path, 'rb') as f:
    short_exit_model = pickle.load(f)

short_exit_scaler_path = MODELS_DIR / "xgboost_short_exit_threshold_075_20251027_190512_scaler.pkl"
short_exit_scaler = joblib.load(short_exit_scaler_path)

short_exit_features_path = MODELS_DIR / "xgboost_short_exit_threshold_075_20251027_190512_features.txt"
with open(short_exit_features_path, 'r') as f:
    short_exit_features = [line.strip() for line in f.readlines() if line.strip()]

print(f"✅ Exit models loaded")
print(f"   LONG Exit: {len(long_exit_features)} features")
print(f"   SHORT Exit: {len(short_exit_features)} features")
print()

# Load Entry models and metadata
print("Loading Entry models...")

# LONG Entry
long_entry_folds = {}
for fold in [1, 2, 3, 4, 5]:
    model_path = MODELS_DIR / f"xgboost_long_entry_ensemble_fold{fold}_{MODEL_TIMESTAMP}.pkl"
    with open(model_path, 'rb') as f:
        long_entry_folds[fold] = pickle.load(f)

long_meta_path = MODELS_DIR / f"xgboost_long_entry_ensemble_meta_{MODEL_TIMESTAMP}.pkl"
with open(long_meta_path, 'rb') as f:
    long_meta = pickle.load(f)

# Load features and scaler from Fold 1 files (all folds have identical features)
long_features_path = MODELS_DIR / f"xgboost_long_entry_ensemble_fold1_{MODEL_TIMESTAMP}_features.txt"
with open(long_features_path, 'r') as f:
    long_entry_features = [line.strip() for line in f.readlines() if line.strip()]

long_scaler_path = MODELS_DIR / f"xgboost_long_entry_ensemble_fold1_{MODEL_TIMESTAMP}_scaler.pkl"
long_entry_scaler = joblib.load(long_scaler_path)

# Map fold scores correctly using fold_indices
long_fold_scores = {
    fold_idx: score
    for fold_idx, score in zip(long_meta['fold_indices'], long_meta['fold_scores'])
}

print(f"✅ LONG Entry loaded: {len(long_entry_features)} features")
print(f"   Fold scores: {long_fold_scores}")

# SHORT Entry
short_entry_folds = {}
for fold in [1, 2, 3, 4, 5]:
    model_path = MODELS_DIR / f"xgboost_short_entry_ensemble_fold{fold}_{MODEL_TIMESTAMP}.pkl"
    with open(model_path, 'rb') as f:
        short_entry_folds[fold] = pickle.load(f)

short_meta_path = MODELS_DIR / f"xgboost_short_entry_ensemble_meta_{MODEL_TIMESTAMP}.pkl"
with open(short_meta_path, 'rb') as f:
    short_meta = pickle.load(f)

# Load features and scaler from Fold 1 files (all folds have identical features)
short_features_path = MODELS_DIR / f"xgboost_short_entry_ensemble_fold1_{MODEL_TIMESTAMP}_features.txt"
with open(short_features_path, 'r') as f:
    short_entry_features = [line.strip() for line in f.readlines() if line.strip()]

short_scaler_path = MODELS_DIR / f"xgboost_short_entry_ensemble_fold1_{MODEL_TIMESTAMP}_scaler.pkl"
short_entry_scaler = joblib.load(short_scaler_path)

# Map fold scores correctly using fold_indices
short_fold_scores = {
    fold_idx: score
    for fold_idx, score in zip(short_meta['fold_indices'], short_meta['fold_scores'])
}

print(f"✅ SHORT Entry loaded: {len(short_entry_features)} features")
print(f"   Fold scores: {short_fold_scores}")
print()

# Determine Top-3 folds
long_top3_folds = sorted(long_fold_scores.items(), key=lambda x: x[1], reverse=True)[:3]
short_top3_folds = sorted(short_fold_scores.items(), key=lambda x: x[1], reverse=True)[:3]

print("Top-3 Folds:")
print(f"  LONG: {[f'Fold {f} ({s:.2f}%)' for f, s in long_top3_folds]}")
print(f"  SHORT: {[f'Fold {f} ({s:.2f}%)' for f, s in short_top3_folds]}")
print()

# Calculate ensemble weights
def calculate_ensemble_weights(top3_folds):
    """Calculate score-based weights for Top-3 ensemble"""
    total_score = sum(score for fold, score in top3_folds)
    if total_score == 0:
        return {fold: 1.0/3 for fold, score in top3_folds}
    return {fold: score / total_score for fold, score in top3_folds}

long_weights = calculate_ensemble_weights(long_top3_folds)
short_weights = calculate_ensemble_weights(short_top3_folds)

print("Ensemble Weights:")
print(f"  LONG: {long_weights}")
print(f"  SHORT: {short_weights}")
print()

# Entry prediction functions
def predict_entry_bestfold(df_row, side):
    """Predict entry using best fold only"""
    if side == 'LONG':
        features = long_entry_features
        scaler = long_entry_scaler
        best_fold = long_top3_folds[0][0]
        model = long_entry_folds[best_fold]
    else:
        features = short_entry_features
        scaler = short_entry_scaler
        best_fold = short_top3_folds[0][0]
        model = short_entry_folds[best_fold]

    try:
        feat = df_row[features].values.reshape(1, -1)
        if np.isnan(feat).any():
            return 0.0
        feat_scaled = scaler.transform(feat)
        prob = model.predict_proba(feat_scaled)[0][1]
        return prob
    except:
        return 0.0

def predict_entry_ensemble(df_row, side):
    """Predict entry using Top-3 Weighted Ensemble"""
    if side == 'LONG':
        features = long_entry_features
        scaler = long_entry_scaler
        top3_folds = long_top3_folds
        models = long_entry_folds
        weights = long_weights
    else:
        features = short_entry_features
        scaler = short_entry_scaler
        top3_folds = short_top3_folds
        models = short_entry_folds
        weights = short_weights

    try:
        feat = df_row[features].values.reshape(1, -1)
        if np.isnan(feat).any():
            return 0.0
        feat_scaled = scaler.transform(feat)

        # Weighted average of Top-3 fold predictions
        probs = []
        for fold, score in top3_folds:
            prob = models[fold].predict_proba(feat_scaled)[0][1]
            probs.append(prob * weights[fold])

        ensemble_prob = sum(probs)
        return ensemble_prob
    except:
        return 0.0

# Dynamic Position Sizing
def calculate_position_size(prob, balance):
    """Calculate dynamic position size based on signal strength"""
    prob_normalized = max(0, min(1, (prob - ENTRY_THRESHOLD) / (1 - ENTRY_THRESHOLD)))
    size_pct = MIN_POSITION_PCT + (MAX_POSITION_PCT - MIN_POSITION_PCT) * (prob_normalized ** 1.5)
    return min(MAX_POSITION_PCT, max(MIN_POSITION_PCT, size_pct))

# Trade simulation
def simulate_trade(df, entry_idx, side, exit_model, exit_scaler, exit_features):
    """Simulate a single trade"""
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

        # Stop Loss
        if leveraged_pnl <= EMERGENCY_STOP_LOSS:
            fee_total = 2 * TAKER_FEE * LEVERAGE
            return leveraged_pnl - fee_total, hold_time, 'stop_loss'

        # ML Exit
        try:
            exit_feat = df[exit_features].iloc[current_idx:current_idx+1].values
            if not np.isnan(exit_feat).any():
                exit_scaled = exit_scaler.transform(exit_feat)
                exit_prob = exit_model.predict_proba(exit_scaled)[0][1]

                if exit_prob >= ML_EXIT_THRESHOLD:
                    fee_total = 2 * TAKER_FEE * LEVERAGE
                    return leveraged_pnl - fee_total, hold_time, 'ml_exit'
        except:
            pass

        # Max Hold
        if hold_time >= EMERGENCY_MAX_HOLD:
            fee_total = 2 * TAKER_FEE * LEVERAGE
            return leveraged_pnl - fee_total, hold_time, 'max_hold'

    # Fallback
    final_pnl = pnl_series[-1] if len(pnl_series) > 0 else 0
    fee_total = 2 * TAKER_FEE * LEVERAGE
    return final_pnl * LEVERAGE - fee_total, len(pnl_series), 'data_end'

# Backtest function
def run_backtest(predict_fn, option_name):
    """Run backtest with given prediction function"""
    print(f"Running backtest: {option_name}")
    print("-" * 80)

    balance = INITIAL_BALANCE
    trades = []
    position = None

    long_count = 0
    short_count = 0
    wins = 0
    losses = 0
    total_pnl = 0

    exit_reasons = {'ml_exit': 0, 'stop_loss': 0, 'max_hold': 0, 'data_end': 0}
    position_sizes = []

    for i in range(len(df) - EMERGENCY_MAX_HOLD):
        if i % 1000 == 0:
            print(f"  Progress: {i:,}/{len(df):,} ({i/len(df)*100:.1f}%)")

        if position is None:
            row = df.iloc[i]

            # Check LONG
            long_prob = predict_fn(row, 'LONG')
            if long_prob >= ENTRY_THRESHOLD:
                position_size_pct = calculate_position_size(long_prob, balance)
                position_value = balance * position_size_pct

                pnl_pct, hold_time, exit_reason = simulate_trade(
                    df, i, 'LONG', long_exit_model, long_exit_scaler, long_exit_features
                )

                pnl_dollars = position_value * pnl_pct
                balance += pnl_dollars

                trades.append({
                    'entry_idx': i,
                    'side': 'LONG',
                    'prob': long_prob,
                    'position_pct': position_size_pct,
                    'pnl_pct': pnl_pct,
                    'pnl_dollars': pnl_dollars,
                    'hold_time': hold_time,
                    'exit_reason': exit_reason
                })

                long_count += 1
                if pnl_dollars > 0:
                    wins += 1
                else:
                    losses += 1
                total_pnl += pnl_dollars
                exit_reasons[exit_reason] += 1
                position_sizes.append(position_size_pct)

                if balance <= 0:
                    print("  ⚠️  Balance depleted - stopping backtest")
                    break

                continue

            # Check SHORT (Opportunity Gating)
            short_prob = predict_fn(row, 'SHORT')
            if short_prob >= ENTRY_THRESHOLD:
                # Opportunity gating: SHORT only if EV(SHORT) > EV(LONG) + 0.001
                long_ev = long_prob * 0.0041
                short_ev = short_prob * 0.0047
                opportunity_cost = short_ev - long_ev

                if opportunity_cost > 0.001:
                    position_size_pct = calculate_position_size(short_prob, balance)
                    position_value = balance * position_size_pct

                    pnl_pct, hold_time, exit_reason = simulate_trade(
                        df, i, 'SHORT', short_exit_model, short_exit_scaler, short_exit_features
                    )

                    pnl_dollars = position_value * pnl_pct
                    balance += pnl_dollars

                    trades.append({
                        'entry_idx': i,
                        'side': 'SHORT',
                        'prob': short_prob,
                        'position_pct': position_size_pct,
                        'pnl_pct': pnl_pct,
                        'pnl_dollars': pnl_dollars,
                        'hold_time': hold_time,
                        'exit_reason': exit_reason
                    })

                    short_count += 1
                    if pnl_dollars > 0:
                        wins += 1
                    else:
                        losses += 1
                    total_pnl += pnl_dollars
                    exit_reasons[exit_reason] += 1
                    position_sizes.append(position_size_pct)

                    if balance <= 0:
                        print("  ⚠️  Balance depleted - stopping backtest")
                        break

    print(f"  ✅ Backtest complete")
    print()

    # Calculate metrics
    total_trades = long_count + short_count
    win_rate = wins / total_trades * 100 if total_trades > 0 else 0
    total_return = (balance - INITIAL_BALANCE) / INITIAL_BALANCE * 100
    avg_position_pct = np.mean(position_sizes) * 100 if position_sizes else 0

    return {
        'option': option_name,
        'total_trades': total_trades,
        'long_count': long_count,
        'short_count': short_count,
        'wins': wins,
        'losses': losses,
        'win_rate': win_rate,
        'total_pnl': total_pnl,
        'total_return': total_return,
        'final_balance': balance,
        'avg_position_pct': avg_position_pct,
        'exit_reasons': exit_reasons,
        'trades': trades
    }

# Run both backtests
print("="*80)
print("OPTION A: BEST FOLD ONLY")
print("="*80)
print()
result_a = run_backtest(predict_entry_bestfold, "Option A (Best Fold)")

print("="*80)
print("OPTION B: TOP-3 WEIGHTED ENSEMBLE")
print("="*80)
print()
result_b = run_backtest(predict_entry_ensemble, "Option B (Top-3 Ensemble)")

# Print comparison
print("="*80)
print("RESULTS COMPARISON")
print("="*80)
print()

def print_results(result):
    print(f"{result['option']}:")
    print(f"  Total Trades: {result['total_trades']:,}")
    print(f"    LONG: {result['long_count']:,} ({result['long_count']/result['total_trades']*100:.1f}%)" if result['total_trades'] > 0 else "")
    print(f"    SHORT: {result['short_count']:,} ({result['short_count']/result['total_trades']*100:.1f}%)" if result['total_trades'] > 0 else "")
    print(f"  Win Rate: {result['win_rate']:.2f}% ({result['wins']}W / {result['losses']}L)")
    print(f"  Total Return: {result['total_return']:+.2f}%")
    print(f"  Total P&L: ${result['total_pnl']:,.2f}")
    print(f"  Final Balance: ${result['final_balance']:,.2f}")
    print(f"  Avg Position Size: {result['avg_position_pct']:.1f}%")
    print(f"  Exit Reasons:")
    for reason, count in result['exit_reasons'].items():
        pct = count / result['total_trades'] * 100 if result['total_trades'] > 0 else 0
        print(f"    {reason}: {count} ({pct:.1f}%)")
    print()

print_results(result_a)
print_results(result_b)

# Determine winner
if result_b['total_return'] > result_a['total_return']:
    improvement = result_b['total_return'] - result_a['total_return']
    print(f"✅ Winner: Option B (Top-3 Ensemble)")
    print(f"   Improvement: +{improvement:.2f}%")
else:
    print(f"✅ Winner: Option A (Best Fold)")

print()
print("="*80)
print("BACKTEST COMPLETE")
print("="*80)
