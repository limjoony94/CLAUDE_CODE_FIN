"""
Backtest Full Dataset Entry Models (No Pre-Filtering)
======================================================

Validate performance improvement with models trained on full 30,004 candles

Expected:
  - LONG Trades: 0 → 50%+ (model should trigger)
  - Win Rate: 45.1% → 70%+
  - Return: -2.18% → +35%+ per window

Models:
  - xgboost_long_entry_full_dataset_20251031_184949.pkl (85 features)
  - xgboost_short_entry_full_dataset_20251031_184949.pkl (79 features)
  - xgboost_long_exit_oppgating_improved_20251024_043527.pkl (27 features)
  - xgboost_short_exit_oppgating_improved_20251024_044510.pkl (27 features)
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import pickle
from datetime import datetime

# Configuration
LONG_THRESHOLD = 0.75
SHORT_THRESHOLD = 0.75
ML_EXIT_THRESHOLD_LONG = 0.75
ML_EXIT_THRESHOLD_SHORT = 0.75
LEVERAGE = 4
EMERGENCY_STOP_LOSS = -0.03
EMERGENCY_MAX_HOLD = 120
INITIAL_BALANCE = 10000

DATA_DIR = PROJECT_ROOT / "data" / "features"
MODELS_DIR = PROJECT_ROOT / "models"

print("="*80)
print("FULL DATASET MODELS BACKTEST (No Pre-Filtering)")
print("="*80)
print()
print("Models:")
print("  Entry: xgboost_*_entry_full_dataset_20251031_184949.pkl")
print("  Exit: xgboost_*_exit_oppgating_improved_20251024_*.pkl")
print()
print("Configuration:")
print(f"  Entry Threshold (LONG): {LONG_THRESHOLD}")
print(f"  Entry Threshold (SHORT): {SHORT_THRESHOLD}")
print(f"  Exit Threshold (LONG): {ML_EXIT_THRESHOLD_LONG}")
print(f"  Exit Threshold (SHORT): {ML_EXIT_THRESHOLD_SHORT}")
print(f"  Leverage: {LEVERAGE}x")
print(f"  Stop Loss: {EMERGENCY_STOP_LOSS*100}% balance")
print(f"  Max Hold: {EMERGENCY_MAX_HOLD} candles")
print()

# Load data with all features (177 features including VWAP/VP)
print("-"*80)
print("Loading Features Dataset")
print("-"*80)

features_file = DATA_DIR / "BTCUSDT_5m_features.csv"
df = pd.read_csv(features_file)

# Add missing Exit features if needed
if 'bb_width' not in df.columns:
    if 'bb_upper' in df.columns and 'bb_lower' in df.columns:
        df['bb_width'] = df['bb_upper'] - df['bb_lower']
    else:
        df['bb_width'] = 0

if 'vwap' not in df.columns:
    df['vwap'] = (df['close'] * df['volume']).rolling(20).sum() / df['volume'].rolling(20).sum()
    df['vwap'] = df['vwap'].ffill().bfill()

print(f"✅ Loaded {len(df):,} candles with {len(df.columns)} features")
print()

# Load models
print("-"*80)
print("Loading Models")
print("-"*80)

# Entry models (full dataset)
long_entry_model_path = MODELS_DIR / "xgboost_long_entry_full_dataset_20251031_184949.pkl"
short_entry_model_path = MODELS_DIR / "xgboost_short_entry_full_dataset_20251031_184949.pkl"

with open(long_entry_model_path, 'rb') as f:
    long_entry_model = pickle.load(f)

with open(short_entry_model_path, 'rb') as f:
    short_entry_model = pickle.load(f)

# Entry scalers
long_entry_scaler_path = MODELS_DIR / "xgboost_long_entry_full_dataset_20251031_184949_scaler.pkl"
short_entry_scaler_path = MODELS_DIR / "xgboost_short_entry_full_dataset_20251031_184949_scaler.pkl"

with open(long_entry_scaler_path, 'rb') as f:
    long_entry_scaler = pickle.load(f)

with open(short_entry_scaler_path, 'rb') as f:
    short_entry_scaler = pickle.load(f)

# Entry features
long_entry_features_path = MODELS_DIR / "xgboost_long_entry_full_dataset_20251031_184949_features.txt"
short_entry_features_path = MODELS_DIR / "xgboost_short_entry_full_dataset_20251031_184949_features.txt"

with open(long_entry_features_path, 'r') as f:
    long_entry_features = [line.strip() for line in f]

with open(short_entry_features_path, 'r') as f:
    short_entry_features = [line.strip() for line in f]

# Exit models (unchanged)
long_exit_model_path = MODELS_DIR / "xgboost_long_exit_oppgating_improved_20251024_043527.pkl"
short_exit_model_path = MODELS_DIR / "xgboost_short_exit_oppgating_improved_20251024_044510.pkl"

with open(long_exit_model_path, 'rb') as f:
    long_exit_model = pickle.load(f)

with open(short_exit_model_path, 'rb') as f:
    short_exit_model = pickle.load(f)

# Exit scalers
long_exit_scaler_path = MODELS_DIR / "xgboost_long_exit_oppgating_improved_20251024_043527_scaler.pkl"
short_exit_scaler_path = MODELS_DIR / "xgboost_short_exit_oppgating_improved_20251024_044510_scaler.pkl"

with open(long_exit_scaler_path, 'rb') as f:
    long_exit_scaler = pickle.load(f)

with open(short_exit_scaler_path, 'rb') as f:
    short_exit_scaler = pickle.load(f)

# Exit features
long_exit_features_path = MODELS_DIR / "xgboost_long_exit_oppgating_improved_20251024_043527_features.txt"
short_exit_features_path = MODELS_DIR / "xgboost_short_exit_oppgating_improved_20251024_044510_features.txt"

with open(long_exit_features_path, 'r') as f:
    long_exit_features = [line.strip() for line in f]

with open(short_exit_features_path, 'r') as f:
    short_exit_features = [line.strip() for line in f]

print(f"✅ LONG Entry: {len(long_entry_features)} features")
print(f"✅ SHORT Entry: {len(short_entry_features)} features")
print(f"✅ LONG Exit: {len(long_exit_features)} features")
print(f"✅ SHORT Exit: {len(short_exit_features)} features")
print()

# Backtest logic
print("="*80)
print("RUNNING BACKTEST (19 Windows, 5-Day Windows)")
print("="*80)
print()

WINDOW_SIZE = 1440  # 5 days * 288 candles
all_trades = []
window_results = []

for window_idx in range(19):
    window_start = window_idx * WINDOW_SIZE
    window_end = window_start + WINDOW_SIZE

    if window_end > len(df):
        break

    window_df = df.iloc[window_start:window_end].copy()

    print(f"Window {window_idx+1}/19: {window_df['timestamp'].iloc[0]} to {window_df['timestamp'].iloc[-1]}")

    balance = INITIAL_BALANCE
    position = None
    trades = []

    for i in range(len(window_df)):
        current_price = window_df['close'].iloc[i]

        # Exit logic
        if position is not None:
            hold_time = i - position['entry_idx']

            # Calculate P&L
            if position['side'] == 'LONG':
                pnl_pct = (current_price - position['entry_price']) / position['entry_price']
            else:
                pnl_pct = (position['entry_price'] - current_price) / position['entry_price']

            leveraged_pnl = pnl_pct * LEVERAGE

            # Exit conditions
            exit_reason = None

            # 1. Stop Loss
            if leveraged_pnl <= EMERGENCY_STOP_LOSS:
                exit_reason = 'STOP_LOSS'

            # 2. Max Hold
            elif hold_time >= EMERGENCY_MAX_HOLD:
                exit_reason = 'MAX_HOLD'

            # 3. ML Exit
            else:
                try:
                    if position['side'] == 'LONG':
                        exit_features_values = window_df[long_exit_features].iloc[i:i+1].values
                        exit_features_scaled = long_exit_scaler.transform(exit_features_values)
                        exit_prob = long_exit_model.predict_proba(exit_features_scaled)[0, 1]

                        if exit_prob >= ML_EXIT_THRESHOLD_LONG:
                            exit_reason = 'ML_EXIT'
                    else:
                        exit_features_values = window_df[short_exit_features].iloc[i:i+1].values
                        exit_features_scaled = short_exit_scaler.transform(exit_features_values)
                        exit_prob = short_exit_model.predict_proba(exit_features_scaled)[0, 1]

                        if exit_prob >= ML_EXIT_THRESHOLD_SHORT:
                            exit_reason = 'ML_EXIT'
                except Exception as e:
                    pass

            # Execute exit
            if exit_reason:
                # Calculate final P&L
                leveraged_value = position['size'] * LEVERAGE
                pnl_usd = leveraged_value * pnl_pct

                # Fee (0.05% taker)
                entry_fee = position['size'] * LEVERAGE * 0.0005
                exit_fee = position['size'] * LEVERAGE * 0.0005
                total_fee = entry_fee + exit_fee

                net_pnl = pnl_usd - total_fee

                balance += net_pnl

                trades.append({
                    'window': window_idx + 1,
                    'side': position['side'],
                    'entry_price': position['entry_price'],
                    'exit_price': current_price,
                    'size': position['size'],
                    'pnl_pct': pnl_pct * 100,
                    'leveraged_pnl_pct': leveraged_pnl * 100,
                    'net_pnl': net_pnl,
                    'exit_reason': exit_reason,
                    'hold_time': hold_time
                })

                position = None

        # Entry logic (if no position)
        if position is None:
            try:
                # LONG Entry
                long_features_values = window_df[long_entry_features].iloc[i:i+1].values
                long_features_scaled = long_entry_scaler.transform(long_features_values)
                long_prob = long_entry_model.predict_proba(long_features_scaled)[0, 1]

                # SHORT Entry
                short_features_values = window_df[short_entry_features].iloc[i:i+1].values
                short_features_scaled = short_entry_scaler.transform(short_features_values)
                short_prob = short_entry_model.predict_proba(short_features_scaled)[0, 1]

                # Entry decision
                if long_prob >= LONG_THRESHOLD:
                    # Dynamic position sizing (20-95%)
                    signal_strength = (long_prob - LONG_THRESHOLD) / (1 - LONG_THRESHOLD)
                    position_pct = 0.20 + (signal_strength * 0.75)
                    position_size = balance * position_pct

                    position = {
                        'side': 'LONG',
                        'entry_price': current_price,
                        'entry_idx': i,
                        'size': position_size,
                        'prob': long_prob
                    }

                elif short_prob >= SHORT_THRESHOLD:
                    # Dynamic position sizing (20-95%)
                    signal_strength = (short_prob - SHORT_THRESHOLD) / (1 - SHORT_THRESHOLD)
                    position_pct = 0.20 + (signal_strength * 0.75)
                    position_size = balance * position_pct

                    position = {
                        'side': 'SHORT',
                        'entry_price': current_price,
                        'entry_idx': i,
                        'size': position_size,
                        'prob': short_prob
                    }
            except Exception as e:
                pass

    # Close any open position at window end
    if position is not None:
        current_price = window_df['close'].iloc[-1]

        if position['side'] == 'LONG':
            pnl_pct = (current_price - position['entry_price']) / position['entry_price']
        else:
            pnl_pct = (position['entry_price'] - current_price) / position['entry_price']

        leveraged_pnl = pnl_pct * LEVERAGE
        leveraged_value = position['size'] * LEVERAGE
        pnl_usd = leveraged_value * pnl_pct

        entry_fee = position['size'] * LEVERAGE * 0.0005
        exit_fee = position['size'] * LEVERAGE * 0.0005
        total_fee = entry_fee + exit_fee

        net_pnl = pnl_usd - total_fee
        balance += net_pnl

        trades.append({
            'window': window_idx + 1,
            'side': position['side'],
            'entry_price': position['entry_price'],
            'exit_price': current_price,
            'size': position['size'],
            'pnl_pct': pnl_pct * 100,
            'leveraged_pnl_pct': leveraged_pnl * 100,
            'net_pnl': net_pnl,
            'exit_reason': 'WINDOW_END',
            'hold_time': len(window_df) - position['entry_idx']
        })

    # Window summary
    window_pnl = sum([t['net_pnl'] for t in trades])
    window_return_pct = (balance - INITIAL_BALANCE) / INITIAL_BALANCE * 100
    window_win_rate = sum([1 for t in trades if t['net_pnl'] > 0]) / len(trades) * 100 if trades else 0

    window_results.append({
        'window': window_idx + 1,
        'trades': len(trades),
        'wins': sum([1 for t in trades if t['net_pnl'] > 0]),
        'losses': sum([1 for t in trades if t['net_pnl'] <= 0]),
        'pnl': window_pnl,
        'return_pct': window_return_pct,
        'win_rate': window_win_rate
    })

    all_trades.extend(trades)

    print(f"  Trades: {len(trades)}, Win Rate: {window_win_rate:.1f}%, Return: {window_return_pct:+.2f}%")

# Overall Results
print()
print("="*80)
print("BACKTEST RESULTS")
print("="*80)

trades_df = pd.DataFrame(all_trades) if all_trades else pd.DataFrame()

total_trades = len(all_trades)
if total_trades > 0:
    total_wins = trades_df[trades_df['net_pnl'] > 0].shape[0]
    total_losses = trades_df[trades_df['net_pnl'] <= 0].shape[0]
    overall_win_rate = (total_wins / total_trades * 100)
    avg_return_per_window = trades_df.groupby('window')['net_pnl'].sum().mean() / INITIAL_BALANCE * 100

    # Exit distribution
    ml_exits = trades_df[trades_df['exit_reason'] == 'ML_EXIT'].shape[0]
    sl_exits = trades_df[trades_df['exit_reason'] == 'STOP_LOSS'].shape[0]
    maxhold_exits = trades_df[trades_df['exit_reason'] == 'MAX_HOLD'].shape[0]

    # Side distribution
    long_trades = trades_df[trades_df['side'] == 'LONG'].shape[0]
    short_trades = trades_df[trades_df['side'] == 'SHORT'].shape[0]

    print()
    print(f"Total Trades: {total_trades}")
    print(f"Win Rate: {overall_win_rate:.1f}% ({total_wins}W / {total_losses}L)")
    print(f"Avg Return per Window: {avg_return_per_window:+.2f}%")
    print()
    print(f"Exit Distribution:")
    print(f"  ML Exit: {ml_exits/total_trades*100:.1f}% ({ml_exits} trades)")
    print(f"  Stop Loss: {sl_exits/total_trades*100:.1f}% ({sl_exits} trades)")
    print(f"  Max Hold: {maxhold_exits/total_trades*100:.1f}% ({maxhold_exits} trades)")
    print()
    print(f"LONG/SHORT Distribution:")
    print(f"  LONG: {long_trades/total_trades*100:.1f}% ({long_trades} trades)")
    print(f"  SHORT: {short_trades/total_trades*100:.1f}% ({short_trades} trades)")
else:
    total_wins = 0
    total_losses = 0
    overall_win_rate = 0
    avg_return_per_window = 0
    print()
    print("⚠️  WARNING: No trades executed!")

print()
print("="*80)
print("IMPROVEMENT vs PRODUCTION FEATURES (Filtered Training)")
print("="*80)
print()
print("Production Features (Filtered - 45.1% WR, -2.18% return):")
print("  - Trained on 4,256 LONG, 2,594 SHORT candidates (filtered)")
print("  - 0 LONG trades, 215 SHORT trades")
print()
print("Full Dataset (NO Filtering):")
print("  - Trained on 30,004 LONG, 30,004 SHORT samples (full)")
if total_trades > 0:
    print(f"  - {long_trades} LONG trades, {short_trades} SHORT trades")
    print(f"  - Win Rate: {overall_win_rate:.1f}% (target: 70%+)")
    print(f"  - Return: {avg_return_per_window:+.2f}% per window (target: +35%+)")
else:
    print("  - No trades executed (investigate model predictions)")

# Save results
if total_trades > 0:
    results_dir = PROJECT_ROOT / "results"
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = results_dir / f"backtest_full_dataset_{timestamp}.csv"
    trades_df.to_csv(output_file, index=False)
    print()
    print(f"✅ Results saved: {output_file.name}")

print()
print("="*80)
print("✅ BACKTEST COMPLETE")
print("="*80)
