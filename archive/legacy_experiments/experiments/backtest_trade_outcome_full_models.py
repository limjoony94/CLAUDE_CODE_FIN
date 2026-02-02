"""
Backtest Trade-Outcome Full Dataset Models
==========================================

Validate Trade-Outcome labeling approach with full dataset models.
Compare vs Baseline and Sample models.
"""

import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import pickle
import joblib
from datetime import datetime

from scripts.experiments.calculate_all_features import calculate_all_features
from scripts.experiments.retrain_exit_models_opportunity_gating import prepare_exit_features

DATA_DIR = PROJECT_ROOT / "data" / "historical"
MODELS_DIR = PROJECT_ROOT / "models"

print("="*80)
print("BACKTEST: TRADE-OUTCOME FULL DATASET MODELS")
print("="*80)

# ============================================================================
# STEP 1: Load Data
# ============================================================================

print("\n" + "-"*80)
print("STEP 1: Loading Historical Data")
print("-"*80)

data_file = DATA_DIR / "BTCUSDT_5m_max.csv"
df = pd.read_csv(data_file)
print(f"\n✅ Loaded {len(df):,} candles")
print(f"   Date range: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")

# Calculate ALL features
print("\nCalculating features...")
df = calculate_all_features(df)
df = prepare_exit_features(df)
print(f"✅ All features calculated ({len(df.columns)} columns)")

# ============================================================================
# STEP 2: Load Full Dataset Models
# ============================================================================

print("\n" + "-"*80)
print("STEP 2: Loading Trade-Outcome Full Dataset Models")
print("-"*80)

# Find latest full models
long_models = sorted(MODELS_DIR.glob("xgboost_long_trade_outcome_full_*.pkl"))
short_models = sorted(MODELS_DIR.glob("xgboost_short_trade_outcome_full_*.pkl"))

# Filter out scaler/features files
long_models = [m for m in long_models if '_scaler' not in m.name and '_features' not in m.name]
short_models = [m for m in short_models if '_scaler' not in m.name and '_features' not in m.name]

if not long_models or not short_models:
    print("❌ ERROR: Full dataset models not found!")
    sys.exit(1)

long_model_path = long_models[-1]
short_model_path = short_models[-1]

# Extract timestamp from filename
timestamp = long_model_path.stem.replace('xgboost_long_trade_outcome_full_', '')

print(f"\nLONG Entry Model: {long_model_path.name}")
print(f"SHORT Entry Model: {short_model_path.name}")

# Load LONG model
with open(long_model_path, 'rb') as f:
    long_model = pickle.load(f)

long_scaler_path = MODELS_DIR / f"xgboost_long_trade_outcome_full_{timestamp}_scaler.pkl"
long_scaler = joblib.load(long_scaler_path)

long_features_path = MODELS_DIR / f"xgboost_long_trade_outcome_full_{timestamp}_features.txt"
with open(long_features_path, 'r') as f:
    long_features = [line.strip() for line in f.readlines() if line.strip()]

# Load SHORT model
with open(short_model_path, 'rb') as f:
    short_model = pickle.load(f)

short_scaler_path = MODELS_DIR / f"xgboost_short_trade_outcome_full_{timestamp}_scaler.pkl"
short_scaler = joblib.load(short_scaler_path)

short_features_path = MODELS_DIR / f"xgboost_short_trade_outcome_full_{timestamp}_features.txt"
with open(short_features_path, 'r') as f:
    short_features = [line.strip() for line in f.readlines() if line.strip()]

print(f"  ✅ LONG: {len(long_features)} features")
print(f"  ✅ SHORT: {len(short_features)} features")

# Load Exit models
print("\nLoading Exit models...")
long_exit_model_path = MODELS_DIR / "xgboost_long_exit_oppgating_improved_20251017_151624.pkl"
with open(long_exit_model_path, 'rb') as f:
    long_exit_model = pickle.load(f)

long_exit_scaler_path = MODELS_DIR / "xgboost_long_exit_oppgating_improved_20251017_151624_scaler.pkl"
long_exit_scaler = joblib.load(long_exit_scaler_path)

long_exit_features_path = MODELS_DIR / "xgboost_long_exit_oppgating_improved_20251017_151624_features.txt"
with open(long_exit_features_path, 'r') as f:
    long_exit_features = [line.strip() for line in f.readlines() if line.strip()]

short_exit_model_path = MODELS_DIR / "xgboost_short_exit_oppgating_improved_20251017_152440.pkl"
with open(short_exit_model_path, 'rb') as f:
    short_exit_model = pickle.load(f)

short_exit_scaler_path = MODELS_DIR / "xgboost_short_exit_oppgating_improved_20251017_152440_scaler.pkl"
short_exit_scaler = joblib.load(short_exit_scaler_path)

short_exit_features_path = MODELS_DIR / "xgboost_short_exit_oppgating_improved_20251017_152440_features.txt"
with open(short_exit_features_path, 'r') as f:
    short_exit_features = [line.strip() for line in f.readlines() if line.strip()]

print(f"  ✅ LONG Exit: {len(long_exit_features)} features")
print(f"  ✅ SHORT Exit: {len(short_exit_features)} features")

# ============================================================================
# STEP 3: Run Backtest
# ============================================================================

print("\n" + "-"*80)
print("STEP 3: Running Backtest (Sliding Window)")
print("-"*80)

# Backtest parameters
WINDOW_SIZE = 1440  # 5 days
STEP_SIZE = 72  # 6 hours
INITIAL_BALANCE = 10000
LEVERAGE = 4
FEE_RATE = 0.0005  # 0.05% BingX trading fee (maker/taker)

# Strategy parameters
LONG_THRESHOLD = 0.65
SHORT_THRESHOLD = 0.70
GATE_THRESHOLD = 0.001
ML_EXIT_THRESHOLD_LONG = 0.70
ML_EXIT_THRESHOLD_SHORT = 0.72
EMERGENCY_STOP_LOSS = -0.03  # Balance-Based: -3% total balance (optimized 2025-10-22, Return: +50.31%, WR: 57.1%, MDD: -15%)
EMERGENCY_MAX_HOLD = 96

LONG_AVG_RETURN = 0.0041
SHORT_AVG_RETURN = 0.0047

# Position sizing
MIN_POSITION_SIZE = 0.20
MAX_POSITION_SIZE = 0.95

def calculate_position_size(prob, avg_return):
    """Dynamic position sizing"""
    normalized = (prob - 0.5) / 0.5
    size = MIN_POSITION_SIZE + (MAX_POSITION_SIZE - MIN_POSITION_SIZE) * normalized
    return max(MIN_POSITION_SIZE, min(MAX_POSITION_SIZE, size))

def run_window_backtest(window_df, window_start_idx):
    """Run backtest on a single window"""
    balance = INITIAL_BALANCE
    position = None
    trades = []

    for i in range(len(window_df)):
        current_idx = window_start_idx + i

        # Skip if position open
        if position is not None:
            current_price = window_df['close'].iloc[i]
            hold_time = i - position['entry_idx']

            # Calculate P&L
            if position['side'] == 'LONG':
                pnl_pct = (current_price - position['entry_price']) / position['entry_price']
            else:
                pnl_pct = (position['entry_price'] - current_price) / position['entry_price']

            leveraged_pnl_pct = pnl_pct * LEVERAGE

            # Check exit conditions
            exit_triggered = False
            exit_reason = None

            # 1. ML Exit
            try:
                if position['side'] == 'LONG':
                    exit_feat = window_df[long_exit_features].iloc[i:i+1].values
                    exit_scaled = long_exit_scaler.transform(exit_feat)
                    exit_prob = long_exit_model.predict_proba(exit_scaled)[0][1]
                    threshold = ML_EXIT_THRESHOLD_LONG
                else:
                    exit_feat = window_df[short_exit_features].iloc[i:i+1].values
                    exit_scaled = short_exit_scaler.transform(exit_feat)
                    exit_prob = short_exit_model.predict_proba(exit_scaled)[0][1]
                    threshold = ML_EXIT_THRESHOLD_SHORT

                if exit_prob >= threshold:
                    exit_triggered = True
                    exit_reason = 'ml_exit'
            except:
                pass

            # 2. Emergency Stop Loss (Balance-Based)
            # Calculate balance loss: leveraged P&L × position size
            balance_loss_pct = leveraged_pnl_pct * position['position_size_pct']
            if balance_loss_pct <= -EMERGENCY_STOP_LOSS:  # -6% total balance
                exit_triggered = True
                exit_reason = 'emergency_stop_loss'

            # 3. Emergency Max Hold
            if hold_time >= EMERGENCY_MAX_HOLD:
                exit_triggered = True
                exit_reason = 'emergency_max_hold'

            if exit_triggered:
                # Close position
                pnl_usd = balance * position['position_size_pct'] * leveraged_pnl_pct

                # Calculate fees (entry + exit)
                position_value = balance * position['position_size_pct']
                entry_fee = position_value * FEE_RATE
                exit_fee = position_value * FEE_RATE
                total_fees = entry_fee + exit_fee

                # Subtract fees from P&L
                pnl_usd_after_fees = pnl_usd - total_fees
                balance += pnl_usd_after_fees

                trades.append({
                    'side': position['side'],
                    'entry_price': position['entry_price'],
                    'exit_price': current_price,
                    'pnl_pct': pnl_pct,
                    'leveraged_pnl_pct': leveraged_pnl_pct,
                    'pnl_usd': pnl_usd,
                    'fees': total_fees,
                    'pnl_after_fees': pnl_usd_after_fees,
                    'hold_time': hold_time,
                    'exit_reason': exit_reason,
                    'position_size_pct': position['position_size_pct']
                })

                position = None

            continue

        # Check entry signals
        try:
            # Get LONG probability
            long_feat = window_df[long_features].iloc[i:i+1].values
            long_scaled = long_scaler.transform(long_feat)
            long_prob = long_model.predict_proba(long_scaled)[0][1]

            # Get SHORT probability
            short_feat = window_df[short_features].iloc[i:i+1].values
            short_scaled = short_scaler.transform(short_feat)
            short_prob = short_model.predict_proba(short_scaled)[0][1]

            # Entry logic
            if long_prob >= LONG_THRESHOLD:
                position_size = calculate_position_size(long_prob, LONG_AVG_RETURN)
                position = {
                    'side': 'LONG',
                    'entry_idx': i,
                    'entry_price': window_df['close'].iloc[i],
                    'position_size_pct': position_size,
                    'entry_prob': long_prob
                }

            elif short_prob >= SHORT_THRESHOLD:
                long_ev = long_prob * LONG_AVG_RETURN
                short_ev = short_prob * SHORT_AVG_RETURN
                opportunity_cost = short_ev - long_ev

                if opportunity_cost > GATE_THRESHOLD:
                    position_size = calculate_position_size(short_prob, SHORT_AVG_RETURN)
                    position = {
                        'side': 'SHORT',
                        'entry_idx': i,
                        'entry_price': window_df['close'].iloc[i],
                        'position_size_pct': position_size,
                        'entry_prob': short_prob
                    }

        except Exception as e:
            continue

    # Close any remaining position
    if position is not None:
        current_price = window_df['close'].iloc[-1]
        hold_time = len(window_df) - 1 - position['entry_idx']

        if position['side'] == 'LONG':
            pnl_pct = (current_price - position['entry_price']) / position['entry_price']
        else:
            pnl_pct = (position['entry_price'] - current_price) / position['entry_price']

        leveraged_pnl_pct = pnl_pct * LEVERAGE
        pnl_usd = balance * position['position_size_pct'] * leveraged_pnl_pct

        # Calculate fees (entry + exit)
        position_value = balance * position['position_size_pct']
        entry_fee = position_value * FEE_RATE
        exit_fee = position_value * FEE_RATE
        total_fees = entry_fee + exit_fee

        # Subtract fees from P&L
        pnl_usd_after_fees = pnl_usd - total_fees
        balance += pnl_usd_after_fees

        trades.append({
            'side': position['side'],
            'entry_price': position['entry_price'],
            'exit_price': current_price,
            'pnl_pct': pnl_pct,
            'leveraged_pnl_pct': leveraged_pnl_pct,
            'pnl_usd': pnl_usd,
            'fees': total_fees,
            'pnl_after_fees': pnl_usd_after_fees,
            'hold_time': hold_time,
            'exit_reason': 'window_end',
            'position_size_pct': position['position_size_pct']
        })

    return balance, trades

# Run backtest
print(f"\nBacktest Configuration:")
print(f"  Window Size: {WINDOW_SIZE} candles (5 days)")
print(f"  Step Size: {STEP_SIZE} candles (6 hours)")
print(f"  Initial Balance: ${INITIAL_BALANCE:,.2f}")
print(f"  Leverage: {LEVERAGE}x")

window_results = []
total_windows = (len(df) - WINDOW_SIZE) // STEP_SIZE

print(f"\nProcessing {total_windows} windows...")

for window_num in range(total_windows):
    start_idx = window_num * STEP_SIZE
    end_idx = start_idx + WINDOW_SIZE

    if end_idx > len(df):
        break

    window_df = df.iloc[start_idx:end_idx].copy().reset_index(drop=True)
    final_balance, trades = run_window_backtest(window_df, start_idx)

    # Calculate metrics
    total_trades = len(trades)
    if total_trades > 0:
        wins = sum(1 for t in trades if t['pnl_after_fees'] > 0)  # Use after-fee P&L
        win_rate = wins / total_trades
        avg_return = (final_balance / INITIAL_BALANCE - 1)
        long_trades = sum(1 for t in trades if t['side'] == 'LONG')
        short_trades = sum(1 for t in trades if t['side'] == 'SHORT')
        total_fees = sum(t['fees'] for t in trades)
    else:
        win_rate = 0
        avg_return = 0
        long_trades = 0
        short_trades = 0
        total_fees = 0

    window_results.append({
        'window': window_num + 1,
        'total_trades': total_trades,
        'long_trades': long_trades,
        'short_trades': short_trades,
        'win_rate': win_rate,
        'return': avg_return,
        'final_balance': final_balance,
        'total_fees': total_fees
    })

    if (window_num + 1) % 20 == 0:
        print(f"  Processed {window_num + 1}/{total_windows} windows...")

print(f"✅ Backtest complete: {len(window_results)} windows")

# ============================================================================
# STEP 4: Analyze Results
# ============================================================================

print("\n" + "="*80)
print("BACKTEST RESULTS SUMMARY")
print("="*80)

results_df = pd.DataFrame(window_results)

# Overall metrics
avg_trades = results_df['total_trades'].mean()
avg_long_trades = results_df['long_trades'].mean()
avg_short_trades = results_df['short_trades'].mean()
avg_win_rate = results_df['win_rate'].mean()
avg_return = results_df['return'].mean()

avg_fees = results_df['total_fees'].mean()

print(f"\n📊 Average Per Window (5 days):")
print(f"  Total Trades: {avg_trades:.1f}")
print(f"    - LONG: {avg_long_trades:.1f} ({avg_long_trades/avg_trades*100:.1f}%)")
print(f"    - SHORT: {avg_short_trades:.1f} ({avg_short_trades/avg_trades*100:.1f}%)")
print(f"  Win Rate: {avg_win_rate*100:.1f}%")
print(f"  Average Return: {avg_return*100:.2f}%")
print(f"  Average Fees: ${avg_fees:.2f} ({avg_fees/INITIAL_BALANCE*100:.2f}% of capital)")

# Problematic windows
low_wr_windows = results_df[results_df['win_rate'] < 0.40]
high_trade_windows = results_df[results_df['total_trades'] > 50]

print(f"\n⚠️ Problematic Windows:")
print(f"  Win Rate < 40%: {len(low_wr_windows)} windows")
print(f"  Total Trades > 50: {len(high_trade_windows)} windows")

if len(low_wr_windows) > 0:
    print(f"    Min Win Rate: {results_df['win_rate'].min()*100:.1f}%")
if len(high_trade_windows) > 0:
    print(f"    Max Trades: {results_df['total_trades'].max():.0f}")

# Distribution
print(f"\n📈 Win Rate Distribution:")
print(f"  < 40%: {len(results_df[results_df['win_rate'] < 0.40])} windows")
print(f"  40-50%: {len(results_df[(results_df['win_rate'] >= 0.40) & (results_df['win_rate'] < 0.50)])} windows")
print(f"  50-60%: {len(results_df[(results_df['win_rate'] >= 0.50) & (results_df['win_rate'] < 0.60)])} windows")
print(f"  60-70%: {len(results_df[(results_df['win_rate'] >= 0.60) & (results_df['win_rate'] < 0.70)])} windows")
print(f"  >= 70%: {len(results_df[results_df['win_rate'] >= 0.70])} windows")

print("\n" + "="*80)
print(f"TRADE-OUTCOME FULL DATASET MODELS BACKTEST COMPLETE")
print("="*80)

print(f"\n📝 Results:")
print(f"   Total Windows: {len(results_df)}")
print(f"   Average Return per 5-day window: {avg_return*100:.2f}%")
print(f"   Average Win Rate: {avg_win_rate*100:.1f}%")
print(f"   Average Trades: {avg_trades:.1f}")

print("\n🎯 Next: Compare with Baseline and Sample models")
