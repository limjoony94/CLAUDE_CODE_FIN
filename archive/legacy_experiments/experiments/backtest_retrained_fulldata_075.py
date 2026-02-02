"""
Backtest Retrained Entry Models - FULL Dataset Validation
==========================================================

Validates Entry models retrained on complete dataset (timestamp 20251030_012702).

Comparison:
  - Option A: Best Fold only (Fold 2 LONG, Fold 4 SHORT)
  - Option B: Top-3 Weighted Ensemble (Folds 2,5,3 LONG; Folds 4,3,5 SHORT)

Period: Complete dataset (Jul 14 - Oct 26, 2025) - 104 days
Models: 20251030_012702

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

# Paths
DATA_DIR = PROJECT_ROOT / "data" / "features"
CSV_FILE = DATA_DIR / "BTCUSDT_5m_features_exit_ready.csv"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
MODEL_TIMESTAMP = "20251030_012702"

print("="*80)
print("BACKTEST - FULL DATASET RETRAINED ENTRY MODELS")
print("="*80)
print()
print(f"Models: {MODEL_TIMESTAMP}")
print(f"Entry Threshold: {ENTRY_THRESHOLD}")
print(f"Exit Threshold: {ML_EXIT_THRESHOLD}")
print(f"Period: Complete dataset (Jul 14 - Oct 26, 2025)")
print()

# Load data
print("Loading data...")
df = pd.read_csv(CSV_FILE)
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values('timestamp').reset_index(drop=True)
print(f"✅ Loaded {len(df):,} candles")
print(f"   Date range: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")
print()

# Load Entry models and metadata
print("Loading Entry models...")
print()

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

print(f"  LONG Entry: {len(long_entry_features)} features")
print(f"  Fold scores: {long_fold_scores}")
print()

# SHORT Entry
short_entry_folds = {}
for fold in [1, 2, 3, 4, 5]:
    model_path = MODELS_DIR / f"xgboost_short_entry_ensemble_fold{fold}_{MODEL_TIMESTAMP}.pkl"
    with open(model_path, 'rb') as f:
        short_entry_folds[fold] = pickle.load(f)

short_meta_path = MODELS_DIR / f"xgboost_short_entry_ensemble_meta_{MODEL_TIMESTAMP}.pkl"
with open(short_meta_path, 'rb') as f:
    short_meta = pickle.load(f)

# Load features and scaler from Fold 1 files
short_features_path = MODELS_DIR / f"xgboost_short_entry_ensemble_fold1_{MODEL_TIMESTAMP}_features.txt"
with open(short_features_path, 'r') as f:
    short_entry_features = [line.strip() for line in f.readlines() if line.strip()]

short_scaler_path = MODELS_DIR / f"xgboost_short_entry_ensemble_fold1_{MODEL_TIMESTAMP}_scaler.pkl"
short_entry_scaler = joblib.load(short_scaler_path)

# Map fold scores correctly
short_fold_scores = {
    fold_idx: score
    for fold_idx, score in zip(short_meta['fold_indices'], short_meta['fold_scores'])
}

print(f"  SHORT Entry: {len(short_entry_features)} features")
print(f"  Fold scores: {short_fold_scores}")
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

print(f"✅ LONG Exit: {len(long_exit_features)} features")
print(f"✅ SHORT Exit: {len(short_exit_features)} features")
print()

# Dynamic position sizing
def calculate_position_size(prob, threshold, min_pct=MIN_POSITION_PCT, max_pct=MAX_POSITION_PCT):
    """Calculate position size based on signal strength"""
    if prob < threshold:
        return 0.0

    # Exponential scaling from threshold to 1.0
    excess = prob - threshold
    max_excess = 1.0 - threshold

    if max_excess <= 0:
        return max_pct

    normalized = excess / max_excess
    scaled = normalized ** 1.5

    position_pct = min_pct + (max_pct - min_pct) * scaled
    return position_pct

# Generate predictions
def predict_ensemble(df_row, folds, scaler, features, fold_scores, top_n=3):
    """Top-N weighted ensemble prediction"""
    # Get top-N folds
    sorted_folds = sorted(fold_scores.items(), key=lambda x: x[1], reverse=True)
    top_folds = sorted_folds[:top_n]

    # Prepare features
    feat_values = df_row[features].values.reshape(1, -1)
    feat_scaled = scaler.transform(feat_values)

    # Get predictions from top folds
    predictions = []
    weights = []
    for fold_idx, score in top_folds:
        model = folds[fold_idx]
        prob = model.predict_proba(feat_scaled)[0][1]
        predictions.append(prob)
        weights.append(score)

    # Normalize weights
    total_weight = sum(weights)
    if total_weight > 0:
        weights = [w / total_weight for w in weights]
    else:
        weights = [1.0 / len(weights)] * len(weights)

    # Weighted average
    ensemble_prob = sum(p * w for p, w in zip(predictions, weights))
    return ensemble_prob

def predict_best_fold(df_row, folds, scaler, features, fold_scores):
    """Best fold only prediction"""
    # Get best fold
    best_fold = max(fold_scores.items(), key=lambda x: x[1])[0]

    # Prepare features
    feat_values = df_row[features].values.reshape(1, -1)
    feat_scaled = scaler.transform(feat_values)

    # Get prediction
    model = folds[best_fold]
    prob = model.predict_proba(feat_scaled)[0][1]
    return prob

# Simulate trade
def simulate_trade(df, entry_idx, side, exit_model, exit_scaler, exit_features, position_size_pct):
    """Simulate a single trade with dynamic position sizing"""
    entry_price = df['close'].iloc[entry_idx]
    entry_time = df['timestamp'].iloc[entry_idx]

    max_hold_end = min(entry_idx + EMERGENCY_MAX_HOLD + 1, len(df))
    prices = df['close'].iloc[entry_idx+1:max_hold_end].values

    for i, current_price in enumerate(prices):
        current_idx = entry_idx + 1 + i
        hold_time = i + 1

        # Calculate P&L
        if side == 'LONG':
            pnl_pct = (current_price - entry_price) / entry_price
        else:
            pnl_pct = (entry_price - current_price) / entry_price

        leveraged_pnl = pnl_pct * LEVERAGE

        # Check Stop Loss
        if leveraged_pnl <= EMERGENCY_STOP_LOSS:
            fee_total = 2 * TAKER_FEE * LEVERAGE
            final_pnl = leveraged_pnl - fee_total
            return final_pnl * position_size_pct, hold_time, 'stop_loss', current_idx

        # Check ML Exit
        try:
            exit_feat = df[exit_features].iloc[current_idx:current_idx+1].values
            if not np.isnan(exit_feat).any():
                exit_scaled = exit_scaler.transform(exit_feat)
                exit_prob = exit_model.predict_proba(exit_scaled)[0][1]

                if exit_prob >= ML_EXIT_THRESHOLD:
                    fee_total = 2 * TAKER_FEE * LEVERAGE
                    final_pnl = leveraged_pnl - fee_total
                    return final_pnl * position_size_pct, hold_time, 'ml_exit', current_idx
        except Exception:
            pass

        # Check Max Hold
        if hold_time >= EMERGENCY_MAX_HOLD:
            fee_total = 2 * TAKER_FEE * LEVERAGE
            final_pnl = leveraged_pnl - fee_total
            return final_pnl * position_size_pct, hold_time, 'max_hold', current_idx

    # Data end fallback
    final_pnl = leveraged_pnl if len(prices) > 0 else 0
    fee_total = 2 * TAKER_FEE * LEVERAGE
    return (final_pnl - fee_total) * position_size_pct, len(prices), 'data_end', entry_idx + len(prices)

# Backtest function
def run_backtest(option_name, use_ensemble=True):
    """Run backtest with specified configuration"""
    print(f"{'='*80}")
    print(f"OPTION {option_name}: {'Top-3 Weighted Ensemble' if use_ensemble else 'Best Fold Only'}")
    print(f"{'='*80}")
    print()

    balance = INITIAL_BALANCE
    trades = []
    current_position = None

    for i in range(len(df) - EMERGENCY_MAX_HOLD - 1):
        if i % 5000 == 0:
            print(f"  Processing candle {i:,}/{len(df):,} ({i/len(df)*100:.1f}%)")

        if current_position is not None:
            continue

        # Generate Entry predictions
        if use_ensemble:
            long_prob = predict_ensemble(
                df.iloc[i], long_entry_folds, long_entry_scaler,
                long_entry_features, long_fold_scores, top_n=3
            )
            short_prob = predict_ensemble(
                df.iloc[i], short_entry_folds, short_entry_scaler,
                short_entry_features, short_fold_scores, top_n=3
            )
        else:
            long_prob = predict_best_fold(
                df.iloc[i], long_entry_folds, long_entry_scaler,
                long_entry_features, long_fold_scores
            )
            short_prob = predict_best_fold(
                df.iloc[i], short_entry_folds, short_entry_scaler,
                short_entry_features, short_fold_scores
            )

        # Check entry signals
        long_size_pct = calculate_position_size(long_prob, ENTRY_THRESHOLD)
        short_size_pct = calculate_position_size(short_prob, ENTRY_THRESHOLD)

        # Opportunity gating
        if long_size_pct > 0:
            long_ev = long_prob * 0.0041
            short_ev = short_prob * 0.0047
            opportunity_cost = short_ev - long_ev

            if opportunity_cost <= 0.001:
                current_position = {
                    'side': 'LONG',
                    'entry_idx': i,
                    'entry_prob': long_prob,
                    'position_size_pct': long_size_pct,
                    'exit_model': long_exit_model,
                    'exit_scaler': long_exit_scaler,
                    'exit_features': long_exit_features
                }

        if current_position is None and short_size_pct > 0:
            short_ev = short_prob * 0.0047
            long_ev = long_prob * 0.0041
            opportunity_cost = short_ev - long_ev

            if opportunity_cost > 0.001:
                current_position = {
                    'side': 'SHORT',
                    'entry_idx': i,
                    'entry_prob': short_prob,
                    'position_size_pct': short_size_pct,
                    'exit_model': short_exit_model,
                    'exit_scaler': short_exit_scaler,
                    'exit_features': short_exit_features
                }

        # Execute trade
        if current_position is not None:
            pnl_pct, hold_time, exit_reason, exit_idx = simulate_trade(
                df, current_position['entry_idx'], current_position['side'],
                current_position['exit_model'], current_position['exit_scaler'],
                current_position['exit_features'], current_position['position_size_pct']
            )

            balance_change = balance * pnl_pct
            new_balance = balance + balance_change

            trades.append({
                'entry_idx': current_position['entry_idx'],
                'entry_time': df['timestamp'].iloc[current_position['entry_idx']],
                'entry_price': df['close'].iloc[current_position['entry_idx']],
                'exit_idx': exit_idx,
                'exit_time': df['timestamp'].iloc[exit_idx] if exit_idx < len(df) else df['timestamp'].iloc[-1],
                'exit_price': df['close'].iloc[exit_idx] if exit_idx < len(df) else df['close'].iloc[-1],
                'side': current_position['side'],
                'entry_prob': current_position['entry_prob'],
                'position_size_pct': current_position['position_size_pct'],
                'hold_time': hold_time,
                'exit_reason': exit_reason,
                'pnl_pct': pnl_pct,
                'balance_before': balance,
                'balance_after': new_balance
            })

            balance = new_balance
            current_position = None

    print(f"  Processing complete!")
    print()

    # Calculate stats
    if len(trades) == 0:
        print("⚠️  NO TRADES GENERATED")
        print()
        return None

    trades_df = pd.DataFrame(trades)
    wins = trades_df[trades_df['pnl_pct'] > 0]
    losses = trades_df[trades_df['pnl_pct'] <= 0]

    total_return = (balance - INITIAL_BALANCE) / INITIAL_BALANCE
    win_rate = len(wins) / len(trades_df)

    # Exit distribution
    exit_counts = trades_df['exit_reason'].value_counts()
    ml_exits = exit_counts.get('ml_exit', 0)
    sl_exits = exit_counts.get('stop_loss', 0)
    maxhold_exits = exit_counts.get('max_hold', 0)

    print(f"RESULTS:")
    print(f"  Final Balance: ${balance:,.2f}")
    print(f"  Total Return: {total_return*100:+.2f}%")
    print(f"  Total Trades: {len(trades_df)}")
    print(f"    LONG: {len(trades_df[trades_df['side']=='LONG'])} ({len(trades_df[trades_df['side']=='LONG'])/len(trades_df)*100:.1f}%)")
    print(f"    SHORT: {len(trades_df[trades_df['side']=='SHORT'])} ({len(trades_df[trades_df['side']=='SHORT'])/len(trades_df)*100:.1f}%)")
    print(f"  Win Rate: {win_rate*100:.2f}% ({len(wins)}W / {len(losses)}L)")
    print(f"  Average Trade: {trades_df['pnl_pct'].mean()*100:+.4f}%")
    print(f"  Average Win: {wins['pnl_pct'].mean()*100:+.4f}%" if len(wins) > 0 else "  Average Win: N/A")
    print(f"  Average Loss: {losses['pnl_pct'].mean()*100:.4f}%" if len(losses) > 0 else "  Average Loss: N/A")
    print(f"  Average Hold: {trades_df['hold_time'].mean():.1f} candles ({trades_df['hold_time'].mean()/12:.2f} hours)")
    print(f"  Average Position Size: {trades_df['position_size_pct'].mean()*100:.1f}%")
    print()
    print(f"EXIT DISTRIBUTION:")
    print(f"  ML Exit: {ml_exits} ({ml_exits/len(trades_df)*100:.1f}%)")
    print(f"  Stop Loss: {sl_exits} ({sl_exits/len(trades_df)*100:.1f}%)")
    print(f"  Max Hold: {maxhold_exits} ({maxhold_exits/len(trades_df)*100:.1f}%)")
    print()

    return {
        'option': option_name,
        'final_balance': balance,
        'total_return': total_return,
        'total_trades': len(trades_df),
        'win_rate': win_rate,
        'avg_trade': trades_df['pnl_pct'].mean(),
        'avg_position_size': trades_df['position_size_pct'].mean(),
        'ml_exit_rate': ml_exits / len(trades_df),
        'trades_df': trades_df
    }

# Run both options
print("="*80)
print("RUNNING BACKTESTS")
print("="*80)
print()

results_a = run_backtest("A", use_ensemble=False)
results_b = run_backtest("B", use_ensemble=True)

# Save results
if results_a is not None:
    results_a['trades_df'].to_csv(
        RESULTS_DIR / f"backtest_fulldata_OPTION_A_bestfold_{MODEL_TIMESTAMP}.csv",
        index=False
    )
    print(f"✅ Saved Option A trades to results/")

if results_b is not None:
    results_b['trades_df'].to_csv(
        RESULTS_DIR / f"backtest_fulldata_OPTION_B_ensemble_{MODEL_TIMESTAMP}.csv",
        index=False
    )
    print(f"✅ Saved Option B trades to results/")

print()
print("="*80)
print("BACKTEST COMPLETE")
print("="*80)
