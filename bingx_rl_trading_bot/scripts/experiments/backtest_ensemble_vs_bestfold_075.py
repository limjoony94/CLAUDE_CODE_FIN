"""
Backtest Comparison: Best Fold vs Top-3 Weighted Ensemble
===========================================================

Compares performance of:
  Option A: Best Fold (current approach)
  Option B: Top-3 Weighted Ensemble (new approach)

Both use same Exit models for fair comparison.

Created: 2025-10-29
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
# POSITION_SIZE_PCT = 0.95  # REMOVED: Now using dynamic sizing
MIN_POSITION_PCT = 0.20
MAX_POSITION_PCT = 0.95
TAKER_FEE = 0.0005

def calculate_position_size_from_signal(signal_prob, entry_threshold=0.75):
    """
    Calculate position size based on signal strength (signal-based dynamic sizing)

    Logic (from DynamicPositionSizer):
    - At threshold (0.75): 20% position
    - Linear scaling with exponential reward for strong signals
    - At 1.0: 95% position

    Args:
        signal_prob: XGBoost probability (0.75-1.0)
        entry_threshold: Minimum probability for entry

    Returns:
        float: Position size percentage (0.20-0.95)
    """
    if signal_prob < entry_threshold:
        return 0.0

    # Normalize from [threshold, 1.0] to [0.0, 1.0]
    normalized = (signal_prob - entry_threshold) / (1.0 - entry_threshold)

    # Apply exponential scaling (reward strong signals)
    factor = normalized ** 1.5

    # Scale to position size: 20% min, 95% max
    position_pct = MIN_POSITION_PCT + ((MAX_POSITION_PCT - MIN_POSITION_PCT) * factor)

    return min(position_pct, MAX_POSITION_PCT)

DATA_DIR = PROJECT_ROOT / "data" / "features"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

print("="*80)
print("BACKTEST COMPARISON: BEST FOLD vs TOP-3 WEIGHTED ENSEMBLE")
print("="*80)
print()
print(f"Configuration:")
print(f"  Entry Threshold: {ENTRY_THRESHOLD}")
print(f"  ML Exit Threshold: {ML_EXIT_THRESHOLD}")
print(f"  Emergency Stop Loss: {EMERGENCY_STOP_LOSS}")
print(f"  Emergency Max Hold: {EMERGENCY_MAX_HOLD} candles")
print(f"  Leverage: {LEVERAGE}x")
print(f"  Position Sizing: DYNAMIC (Signal-Based)")
print(f"    - Min: {MIN_POSITION_PCT*100:.0f}% (at threshold)")
print(f"    - Max: {MAX_POSITION_PCT*100:.0f}% (at 1.0 prob)")
print(f"    - Scaling: Exponential (^1.5)")
print(f"  Initial Balance: ${INITIAL_BALANCE:,}")
print()

# Load data
print("-"*80)
print("Loading Data")
print("-"*80)

# Use exit-ready CSV with all 21 Exit features present
df = pd.read_csv(DATA_DIR / "BTCUSDT_5m_features_exit_ready.csv")
print(f"✅ Loaded {len(df):,} candles")
print(f"   Date range: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")
print()

# ==============================================================================
# OPTION A: Best Fold Models
# ==============================================================================

print("-"*80)
print("OPTION A: Loading Best Fold Models")
print("-"*80)

BESTFOLD_TIMESTAMP = "20251027_194313"

# LONG Entry (Best Fold 2)
long_bestfold_path = MODELS_DIR / f"xgboost_long_entry_walkforward_decoupled_{BESTFOLD_TIMESTAMP}.pkl"
with open(long_bestfold_path, 'rb') as f:
    long_bestfold_model = pickle.load(f)

long_bestfold_scaler = joblib.load(MODELS_DIR / f"xgboost_long_entry_walkforward_decoupled_{BESTFOLD_TIMESTAMP}_scaler.pkl")

with open(MODELS_DIR / f"xgboost_long_entry_walkforward_decoupled_{BESTFOLD_TIMESTAMP}_features.txt", 'r') as f:
    long_features = [line.strip() for line in f]

print(f"✅ LONG Best Fold: {len(long_features)} features")

# SHORT Entry (Best Fold 4)
short_bestfold_path = MODELS_DIR / f"xgboost_short_entry_walkforward_decoupled_{BESTFOLD_TIMESTAMP}.pkl"
with open(short_bestfold_path, 'rb') as f:
    short_bestfold_model = pickle.load(f)

short_bestfold_scaler = joblib.load(MODELS_DIR / f"xgboost_short_entry_walkforward_decoupled_{BESTFOLD_TIMESTAMP}_scaler.pkl")

with open(MODELS_DIR / f"xgboost_short_entry_walkforward_decoupled_{BESTFOLD_TIMESTAMP}_features.txt", 'r') as f:
    short_features = [line.strip() for line in f]

print(f"✅ SHORT Best Fold: {len(short_features)} features")
print()

# ==============================================================================
# OPTION B: Top-3 Weighted Ensemble
# ==============================================================================

print("-"*80)
print("OPTION B: Loading Top-3 Weighted Ensemble")
print("-"*80)

ENSEMBLE_TIMESTAMP = "20251029_194432"

# Load LONG Ensemble metadata
long_meta_path = MODELS_DIR / f"xgboost_long_entry_ensemble_meta_{ENSEMBLE_TIMESTAMP}.pkl"
with open(long_meta_path, 'rb') as f:
    long_meta = pickle.load(f)

long_top3_folds = long_meta['fold_indices'][:3]
long_top3_scores = long_meta['fold_scores'][:3]

long_ensemble_models = []
long_ensemble_scalers = []

for fold_idx in long_top3_folds:
    model_path = MODELS_DIR / f"xgboost_long_entry_ensemble_fold{fold_idx}_{ENSEMBLE_TIMESTAMP}.pkl"
    scaler_path = MODELS_DIR / f"xgboost_long_entry_ensemble_fold{fold_idx}_{ENSEMBLE_TIMESTAMP}_scaler.pkl"

    with open(model_path, 'rb') as f:
        long_ensemble_models.append(pickle.load(f))

    long_ensemble_scalers.append(joblib.load(scaler_path))

long_weights = np.array(long_top3_scores) / sum(long_top3_scores)

print(f"✅ LONG Ensemble loaded (Top-3 folds)")
for fold_idx, score, weight in zip(long_top3_folds, long_top3_scores, long_weights):
    print(f"   Fold {fold_idx}: score {score:.4f}, weight {weight:.4f}")

# Load SHORT Ensemble metadata
short_meta_path = MODELS_DIR / f"xgboost_short_entry_ensemble_meta_{ENSEMBLE_TIMESTAMP}.pkl"
with open(short_meta_path, 'rb') as f:
    short_meta = pickle.load(f)

short_top3_folds = short_meta['fold_indices'][:3]
short_top3_scores = short_meta['fold_scores'][:3]

short_ensemble_models = []
short_ensemble_scalers = []

for fold_idx in short_top3_folds:
    model_path = MODELS_DIR / f"xgboost_short_entry_ensemble_fold{fold_idx}_{ENSEMBLE_TIMESTAMP}.pkl"
    scaler_path = MODELS_DIR / f"xgboost_short_entry_ensemble_fold{fold_idx}_{ENSEMBLE_TIMESTAMP}_scaler.pkl"

    with open(model_path, 'rb') as f:
        short_ensemble_models.append(pickle.load(f))

    short_ensemble_scalers.append(joblib.load(scaler_path))

short_weights = np.array(short_top3_scores) / sum(short_top3_scores)

print(f"✅ SHORT Ensemble loaded (Top-3 folds)")
for fold_idx, score, weight in zip(short_top3_folds, short_top3_scores, short_weights):
    print(f"   Fold {fold_idx}: score {score:.4f}, weight {weight:.4f}")
print()

# ==============================================================================
# Exit Models (Same for both options)
# ==============================================================================

print("-"*80)
print("Loading Exit Models (Shared)")
print("-"*80)

# LONG Exit (Reverted to threshold_075 for faster comparison - enhanced features not in CSV)
long_exit_path = MODELS_DIR / "xgboost_long_exit_threshold_075_20251027_190512.pkl"
with open(long_exit_path, 'rb') as f:
    long_exit_model = pickle.load(f)

long_exit_scaler = joblib.load(MODELS_DIR / "xgboost_long_exit_threshold_075_20251027_190512_scaler.pkl")

with open(MODELS_DIR / "xgboost_long_exit_threshold_075_20251027_190512_features.txt", 'r') as f:
    long_exit_features = [line.strip() for line in f]

# SHORT Exit (Reverted to threshold_075 for faster comparison - enhanced features not in CSV)
short_exit_path = MODELS_DIR / "xgboost_short_exit_threshold_075_20251027_190512.pkl"
with open(short_exit_path, 'rb') as f:
    short_exit_model = pickle.load(f)

short_exit_scaler = joblib.load(MODELS_DIR / "xgboost_short_exit_threshold_075_20251027_190512_scaler.pkl")

with open(MODELS_DIR / "xgboost_short_exit_threshold_075_20251027_190512_features.txt", 'r') as f:
    short_exit_features = [line.strip() for line in f]

print(f"✅ Exit models loaded")
print(f"   LONG Exit: {len(long_exit_features)} features")
print(f"   SHORT Exit: {len(short_exit_features)} features")
print()

# ==============================================================================
# Helper Functions
# ==============================================================================

def weighted_ensemble_predict(features, ensemble_models, ensemble_scalers, weights):
    """Weighted ensemble prediction"""
    predictions = []

    for model, scaler in zip(ensemble_models, ensemble_scalers):
        features_scaled = scaler.transform(features)
        prob_both = model.predict_proba(features_scaled)[0]
        prob = prob_both[1]
        predictions.append(prob)

    weighted_pred = np.average(predictions, weights=weights)
    return weighted_pred

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

def run_backtest(long_entry_predict_fn, short_entry_predict_fn, option_name):
    """Run backtest with specified entry models"""
    print(f"\n{'='*80}")
    print(f"Running Backtest: {option_name}")
    print(f"{'='*80}")

    balance = INITIAL_BALANCE
    trades = []

    for i in range(100, len(df) - EMERGENCY_MAX_HOLD):
        # Get Entry predictions
        long_feat = df[long_features].iloc[i:i+1].values
        short_feat = df[short_features].iloc[i:i+1].values

        if np.isnan(long_feat).any() or np.isnan(short_feat).any():
            continue

        long_prob = long_entry_predict_fn(long_feat)
        short_prob = short_entry_predict_fn(short_feat)

        # Entry logic (Opportunity Gating)
        entered = False

        if long_prob >= ENTRY_THRESHOLD:
            # Calculate dynamic position size based on signal strength
            position_size_pct = calculate_position_size_from_signal(long_prob, ENTRY_THRESHOLD)

            # Simulate LONG trade
            pnl_pct, hold_time, exit_reason = simulate_trade(
                df, i, 'LONG', long_exit_model, long_exit_scaler, long_exit_features
            )

            position_value = balance * position_size_pct
            pnl_dollars = position_value * pnl_pct
            balance += pnl_dollars

            trades.append({
                'entry_idx': i,
                'side': 'LONG',
                'signal_prob': long_prob,
                'position_size_pct': position_size_pct,
                'pnl_pct': pnl_pct,
                'pnl_dollars': pnl_dollars,
                'hold_time': hold_time,
                'exit_reason': exit_reason,
                'balance': balance
            })

            entered = True

        elif short_prob >= ENTRY_THRESHOLD:
            # Opportunity Gating check
            long_ev = long_prob * 0.0041
            short_ev = short_prob * 0.0047
            opportunity_cost = short_ev - long_ev

            if opportunity_cost > 0.001:
                # Calculate dynamic position size based on signal strength
                position_size_pct = calculate_position_size_from_signal(short_prob, ENTRY_THRESHOLD)

                # Simulate SHORT trade
                pnl_pct, hold_time, exit_reason = simulate_trade(
                    df, i, 'SHORT', short_exit_model, short_exit_scaler, short_exit_features
                )

                position_value = balance * position_size_pct
                pnl_dollars = position_value * pnl_pct
                balance += pnl_dollars

                trades.append({
                    'entry_idx': i,
                    'side': 'SHORT',
                    'signal_prob': short_prob,
                    'position_size_pct': position_size_pct,
                    'pnl_pct': pnl_pct,
                    'pnl_dollars': pnl_dollars,
                    'hold_time': hold_time,
                    'exit_reason': exit_reason,
                    'balance': balance
                })

                entered = True

    # Calculate metrics
    trades_df = pd.DataFrame(trades)

    if len(trades_df) == 0:
        print(f"⚠️  No trades for {option_name}")
        return None

    total_return = ((balance - INITIAL_BALANCE) / INITIAL_BALANCE) * 100
    win_trades = len(trades_df[trades_df['pnl_pct'] > 0])
    loss_trades = len(trades_df[trades_df['pnl_pct'] <= 0])
    win_rate = (win_trades / len(trades_df)) * 100 if len(trades_df) > 0 else 0

    print(f"\nResults:")
    print(f"  Total Trades: {len(trades_df)}")
    print(f"  Win Rate: {win_rate:.2f}% ({win_trades}W / {loss_trades}L)")
    print(f"  Total Return: {total_return:+.2f}%")
    print(f"  Final Balance: ${balance:,.2f}")
    print(f"  Avg Hold Time: {trades_df['hold_time'].mean():.1f} candles")
    print(f"  Avg Position Size: {trades_df['position_size_pct'].mean()*100:.1f}%")

    # Exit distribution
    exit_counts = trades_df['exit_reason'].value_counts()
    print(f"\nExit Distribution:")
    for reason, count in exit_counts.items():
        pct = (count / len(trades_df)) * 100
        print(f"  {reason}: {count} ({pct:.1f}%)")

    # LONG/SHORT distribution
    side_counts = trades_df['side'].value_counts()
    print(f"\nSide Distribution:")
    for side, count in side_counts.items():
        pct = (count / len(trades_df)) * 100
        print(f"  {side}: {count} ({pct:.1f}%)")

    return {
        'option': option_name,
        'total_trades': len(trades_df),
        'win_rate': win_rate,
        'total_return': total_return,
        'final_balance': balance,
        'avg_hold_time': trades_df['hold_time'].mean(),
        'trades_df': trades_df
    }

# ==============================================================================
# Run Backtests
# ==============================================================================

# Option A: Best Fold
def long_bestfold_predict(features):
    features_scaled = long_bestfold_scaler.transform(features)
    prob_both = long_bestfold_model.predict_proba(features_scaled)[0]
    return prob_both[1]

def short_bestfold_predict(features):
    features_scaled = short_bestfold_scaler.transform(features)
    prob_both = short_bestfold_model.predict_proba(features_scaled)[0]
    return prob_both[1]

result_bestfold = run_backtest(
    long_bestfold_predict,
    short_bestfold_predict,
    "Option A: Best Fold"
)

# Option B: Top-3 Weighted Ensemble
def long_ensemble_predict(features):
    return weighted_ensemble_predict(features, long_ensemble_models, long_ensemble_scalers, long_weights)

def short_ensemble_predict(features):
    return weighted_ensemble_predict(features, short_ensemble_models, short_ensemble_scalers, short_weights)

result_ensemble = run_backtest(
    long_ensemble_predict,
    short_ensemble_predict,
    "Option B: Top-3 Weighted Ensemble"
)

# ==============================================================================
# Comparison Summary
# ==============================================================================

if result_bestfold is not None and result_ensemble is not None:
    print(f"\n{'='*80}")
    print(f"COMPARISON SUMMARY")
    print(f"{'='*80}")
    print()
    print(f"{'Metric':<25} {'Best Fold':<20} {'Top-3 Ensemble':<20} {'Winner':<10}")
    print(f"{'-'*80}")

    metrics = [
        ('Total Trades', 'total_trades', False),
        ('Win Rate (%)', 'win_rate', True),
        ('Total Return (%)', 'total_return', True),
        ('Final Balance ($)', 'final_balance', True),
        ('Avg Hold Time', 'avg_hold_time', False)
    ]

    for label, key, higher_better in metrics:
        val_a = result_bestfold[key]
        val_b = result_ensemble[key]

        if key == 'final_balance':
            str_a = f"${val_a:,.2f}"
            str_b = f"${val_b:,.2f}"
        elif key in ['win_rate', 'total_return']:
            str_a = f"{val_a:.2f}%"
            str_b = f"{val_b:.2f}%"
        else:
            str_a = f"{val_a:.2f}"
            str_b = f"{val_b:.2f}"

        if higher_better:
            winner = "Ensemble ✅" if val_b > val_a else "Best Fold"
        else:
            winner = "Ensemble ✅" if val_b < val_a else "Best Fold"

        print(f"{label:<25} {str_a:<20} {str_b:<20} {winner:<10}")

    print()
    print("Improvement:")
    return_improvement = ((result_ensemble['total_return'] - result_bestfold['total_return']) / abs(result_bestfold['total_return'])) * 100
    wr_improvement = result_ensemble['win_rate'] - result_bestfold['win_rate']

    print(f"  Return: {return_improvement:+.2f}%")
    print(f"  Win Rate: {wr_improvement:+.2f}pp")
    print()

    if return_improvement >= 5.0:
        print("🚀 RECOMMENDATION: DEPLOY TOP-3 ENSEMBLE (+5% threshold met)")
    else:
        print("⚠️  RECOMMENDATION: KEEP BEST FOLD (improvement < 5%)")

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
print(f"\n✅ Comparison complete - {timestamp}")
