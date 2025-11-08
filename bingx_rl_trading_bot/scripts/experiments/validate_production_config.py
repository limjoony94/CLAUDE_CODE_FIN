"""
Validate ACTUAL Production Configuration Performance
====================================================

Configuration (from production bot code):
  Entry: Enhanced 20251024_012445 @ 0.80/0.80
  Exit: OppGating Improved 20251024_043527 @ 0.75/0.75

Purpose: Measure current performance to identify specific gaps vs targets

Targets:
  - Win Rate: 70-75%
  - Return: +35-40% per window
  - ML Exit: 75-85%
  - Avg Hold: 20-30 candles
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

# EXACT Production Configuration
LEVERAGE = 4
EMERGENCY_STOP_LOSS = -0.03
EMERGENCY_MAX_HOLD = 120
INITIAL_BALANCE = 10000
WINDOW_SIZE = 5

LONG_ENTRY_THRESHOLD = 0.80
SHORT_ENTRY_THRESHOLD = 0.80
ML_EXIT_THRESHOLD_LONG = 0.75
ML_EXIT_THRESHOLD_SHORT = 0.75

DATA_DIR = PROJECT_ROOT / "data" / "features"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

print("="*80)
print("PRODUCTION CONFIGURATION VALIDATION")
print("="*80)
print()
print("Configuration:")
print(f"  Entry Thresholds: {LONG_ENTRY_THRESHOLD} / {SHORT_ENTRY_THRESHOLD}")
print(f"  Exit Thresholds: {ML_EXIT_THRESHOLD_LONG} / {ML_EXIT_THRESHOLD_SHORT}")
print(f"  Stop Loss: {EMERGENCY_STOP_LOSS * 100}%")
print(f"  Max Hold: {EMERGENCY_MAX_HOLD} candles")
print()

# Load data
print("Loading data...")
df = pd.read_csv(DATA_DIR / "BTCUSDT_5m_features.csv")

# Enhanced features for Exit models
df['volume_surge'] = (df['volume'] / df['volume'].rolling(20).mean() - 1).fillna(0)
df['price_acceleration'] = df['close'].diff(2).fillna(0)

if 'sma_20' in df.columns:
    df['price_vs_ma20'] = ((df['close'] - df['sma_20']) / df['sma_20']).fillna(0)
else:
    df['price_vs_ma20'] = 0

if 'sma_50' not in df.columns:
    df['sma_50'] = df['close'].rolling(50).mean()
df['price_vs_ma50'] = ((df['close'] - df['sma_50']) / df['sma_50']).fillna(0)

df['volatility_20'] = df['close'].pct_change().rolling(20).std().fillna(0)

if 'rsi' in df.columns:
    df['rsi_slope'] = df['rsi'].diff(5).fillna(0)
    df['rsi_overbought'] = (df['rsi'] > 70).astype(int)
    df['rsi_oversold'] = (df['rsi'] < 30).astype(int)
    df['rsi_divergence'] = (df['rsi'].diff() * df['close'].pct_change() < 0).astype(int)

if 'macd' in df.columns and 'macd_signal' in df.columns:
    df['macd_histogram_slope'] = (df['macd'] - df['macd_signal']).diff(3).fillna(0)
    df['macd_crossover'] = ((df['macd'] > df['macd_signal']) & (df['macd'].shift(1) <= df['macd_signal'].shift(1))).astype(int)
    df['macd_crossunder'] = ((df['macd'] < df['macd_signal']) & (df['macd'].shift(1) >= df['macd_signal'].shift(1))).astype(int)

if 'bb_high' in df.columns and 'bb_low' in df.columns:
    bb_range = df['bb_high'] - df['bb_low']
    bb_range = bb_range.replace(0, 1e-10)
    df['bb_position'] = ((df['close'] - df['bb_low']) / bb_range).fillna(0.5)

df['higher_high'] = (df['high'] > df['high'].shift(1)).astype(int)

support_level = df['low'].rolling(20).min()
df['near_support'] = (df['close'] <= support_level * 1.01).astype(int)

print(f"✅ Loaded {len(df)} candles")
print()

# Load Entry models
print("Loading Entry models...")
with open(MODELS_DIR / "xgboost_long_entry_enhanced_20251024_012445.pkl", 'rb') as f:
    long_entry_model = pickle.load(f)
long_entry_scaler = joblib.load(MODELS_DIR / "xgboost_long_entry_enhanced_20251024_012445_scaler.pkl")
with open(MODELS_DIR / "xgboost_long_entry_enhanced_20251024_012445_features.txt", 'r') as f:
    long_entry_features = [line.strip() for line in f.readlines()]

with open(MODELS_DIR / "xgboost_short_entry_enhanced_20251024_012445.pkl", 'rb') as f:
    short_entry_model = pickle.load(f)
short_entry_scaler = joblib.load(MODELS_DIR / "xgboost_short_entry_enhanced_20251024_012445_scaler.pkl")
with open(MODELS_DIR / "xgboost_short_entry_enhanced_20251024_012445_features.txt", 'r') as f:
    short_entry_features = [line.strip() for line in f.readlines() if line.strip()]

print(f"✅ Entry Models: LONG {len(long_entry_features)} features, SHORT {len(short_entry_features)} features")
print()

# Load Exit models
print("Loading Exit models...")
with open(MODELS_DIR / "xgboost_long_exit_oppgating_improved_20251024_043527.pkl", 'rb') as f:
    long_exit_model = pickle.load(f)
with open(MODELS_DIR / "xgboost_long_exit_oppgating_improved_20251024_043527_features.txt", 'r') as f:
    long_exit_features = [line.strip() for line in f.readlines() if line.strip()]

with open(MODELS_DIR / "xgboost_short_exit_oppgating_improved_20251024_044510.pkl", 'rb') as f:
    short_exit_model = pickle.load(f)
with open(MODELS_DIR / "xgboost_short_exit_oppgating_improved_20251024_044510_features.txt", 'r') as f:
    short_exit_features = [line.strip() for line in f.readlines() if line.strip()]

print(f"✅ Exit Models: LONG {len(long_exit_features)} features, SHORT {len(short_exit_features)} features")
print()

# Run backtest
candles_per_window = WINDOW_SIZE * 288
total_windows = (len(df) - 100) // candles_per_window
print(f"Backtesting {total_windows} windows (5-day each)")
print()

balance = INITIAL_BALANCE
position = None
all_trades = []

for window_num in range(total_windows):
    window_start = 100 + (window_num * candles_per_window)
    window_end = window_start + candles_per_window
    window_df = df.iloc[window_start:window_end].copy().reset_index(drop=True)

    if len(window_df) < 100:
        continue

    for idx in range(len(window_df)):
        # Monitor position
        if position is not None:
            current_price = window_df.iloc[idx]['close']
            side = position['side']
            entry_price = position['entry_price']
            hold_time = idx - position['entry_idx']

            if side == 'LONG':
                pnl_pct = ((current_price - entry_price) / entry_price) * LEVERAGE
            else:
                pnl_pct = ((entry_price - current_price) / entry_price) * LEVERAGE

            exit_reason = None

            # Stop Loss
            if pnl_pct <= EMERGENCY_STOP_LOSS:
                exit_reason = 'STOP_LOSS'
            # Max Hold
            elif hold_time >= EMERGENCY_MAX_HOLD:
                exit_reason = 'MAX_HOLD'
            # ML Exit
            else:
                if side == 'LONG':
                    exit_features_df = window_df.iloc[[idx]][long_exit_features]
                    exit_prob = long_exit_model.predict_proba(exit_features_df.values)[0, 1]
                    if exit_prob >= ML_EXIT_THRESHOLD_LONG:
                        exit_reason = 'ML_EXIT'
                else:
                    exit_features_df = window_df.iloc[[idx]][short_exit_features]
                    exit_prob = short_exit_model.predict_proba(exit_features_df.values)[0, 1]
                    if exit_prob >= ML_EXIT_THRESHOLD_SHORT:
                        exit_reason = 'ML_EXIT'

            # Execute exit
            if exit_reason is not None:
                gross_pnl = position['position_size'] * pnl_pct
                fees = position['position_size'] * 0.0005 * 2
                net_pnl = gross_pnl - fees

                trade = {
                    'window': window_num,
                    'side': side,
                    'exit_reason': exit_reason,
                    'hold_time': hold_time,
                    'net_pnl': net_pnl,
                    'leveraged_pnl_pct': pnl_pct
                }

                balance += net_pnl
                all_trades.append(trade)
                position = None

        # Check for entry
        if position is None and idx < len(window_df) - EMERGENCY_MAX_HOLD:
            long_features_df = window_df.iloc[[idx]][long_entry_features]
            short_features_df = window_df.iloc[[idx]][short_entry_features]

            long_features_scaled = long_entry_scaler.transform(long_features_df.values)
            short_features_scaled = short_entry_scaler.transform(short_features_df.values)

            long_prob = long_entry_model.predict_proba(long_features_scaled)[0, 1]
            short_prob = short_entry_model.predict_proba(short_features_scaled)[0, 1]

            enter_side = None

            if long_prob >= LONG_ENTRY_THRESHOLD:
                enter_side = 'LONG'
            elif short_prob >= SHORT_ENTRY_THRESHOLD:
                long_ev = long_prob * 0.0041
                short_ev = short_prob * 0.0047
                opportunity_cost = short_ev - long_ev
                if opportunity_cost > 0.001:
                    enter_side = 'SHORT'

            if enter_side is not None:
                current_price = window_df.iloc[idx]['close']
                position_size = balance * 0.95
                position = {
                    'side': enter_side,
                    'entry_idx': idx,
                    'entry_price': current_price,
                    'position_size': position_size
                }

    # Close position at window end
    if position is not None:
        idx = len(window_df) - 1
        current_price = window_df.iloc[idx]['close']
        side = position['side']
        entry_price = position['entry_price']

        if side == 'LONG':
            pnl_pct = ((current_price - entry_price) / entry_price) * LEVERAGE
        else:
            pnl_pct = ((entry_price - current_price) / entry_price) * LEVERAGE

        gross_pnl = position['position_size'] * pnl_pct
        fees = position['position_size'] * 0.0005 * 2
        net_pnl = gross_pnl - fees

        trade = {
            'window': window_num,
            'side': side,
            'exit_reason': 'WINDOW_END',
            'hold_time': idx - position['entry_idx'],
            'net_pnl': net_pnl,
            'leveraged_pnl_pct': pnl_pct
        }

        balance += net_pnl
        all_trades.append(trade)
        position = None

# Analysis
print("="*80)
print("BACKTEST RESULTS")
print("="*80)
print()

if len(all_trades) == 0:
    print("❌ NO TRADES EXECUTED")
    print()
    print("Diagnosis needed - likely Entry threshold too high")
else:
    trades_df = pd.DataFrame(all_trades)

    total_trades = len(trades_df)
    wins = len(trades_df[trades_df['net_pnl'] > 0])
    losses = len(trades_df[trades_df['net_pnl'] <= 0])
    win_rate = (wins / total_trades * 100)

    total_return = ((balance - INITIAL_BALANCE) / INITIAL_BALANCE) * 100
    avg_return = total_return / total_windows

    ml_exits = len(trades_df[trades_df['exit_reason'] == 'ML_EXIT'])
    sl_exits = len(trades_df[trades_df['exit_reason'] == 'STOP_LOSS'])
    mh_exits = len(trades_df[trades_df['exit_reason'] == 'MAX_HOLD'])
    we_exits = len(trades_df[trades_df['exit_reason'] == 'WINDOW_END'])

    ml_exit_pct = (ml_exits / total_trades * 100)
    avg_hold = trades_df['hold_time'].mean()

    long_trades = len(trades_df[trades_df['side'] == 'LONG'])
    short_trades = len(trades_df[trades_df['side'] == 'SHORT'])

    print(f"Total Trades: {total_trades} ({wins}W / {losses}L)")
    print(f"Win Rate: {win_rate:.1f}% | Target: 70-75%")
    print(f"Return: {avg_return:.2f}% per window | Target: 35-40%")
    print(f"ML Exit: {ml_exit_pct:.1f}% | Target: 75-85%")
    print(f"Avg Hold: {avg_hold:.1f} candles | Target: 20-30")
    print()
    print(f"Exit Distribution:")
    print(f"  ML Exit: {ml_exits} ({ml_exit_pct:.1f}%)")
    print(f"  Stop Loss: {sl_exits} ({sl_exits/total_trades*100:.1f}%)")
    print(f"  Max Hold: {mh_exits} ({mh_exits/total_trades*100:.1f}%)")
    print(f"  Window End: {we_exits} ({we_exits/total_trades*100:.1f}%)")
    print()
    print(f"LONG/SHORT: {long_trades} ({long_trades/total_trades*100:.1f}%) / {short_trades} ({short_trades/total_trades*100:.1f}%)")
    print()

    # Gap analysis
    print("="*80)
    print("GAP ANALYSIS")
    print("="*80)
    print()

    win_rate_gap = 72.5 - win_rate  # target midpoint
    return_gap = 37.5 - avg_return
    ml_exit_gap = 80 - ml_exit_pct
    hold_gap = avg_hold - 25

    print(f"Win Rate Gap: {win_rate_gap:+.1f}pp")
    print(f"Return Gap: {return_gap:+.2f}%")
    print(f"ML Exit Gap: {ml_exit_gap:+.1f}pp")
    print(f"Hold Time Gap: {hold_gap:+.1f} candles")
    print()

    # Priority recommendations
    print("="*80)
    print("PRIORITY IMPROVEMENTS")
    print("="*80)
    print()

    issues = []

    if win_rate < 70:
        issues.append(("Win Rate", win_rate_gap, "CRITICAL"))
    if avg_return < 35:
        issues.append(("Return", return_gap, "CRITICAL"))
    if ml_exit_pct < 75:
        issues.append(("ML Exit", ml_exit_gap, "HIGH"))
    if avg_hold < 20 or avg_hold > 30:
        issues.append(("Hold Time", hold_gap, "MEDIUM"))

    for i, (metric, gap, priority) in enumerate(issues, 1):
        print(f"{i}. [{priority}] {metric}: {gap:+.1f} gap from target")

    if len(issues) == 0:
        print("✅ ALL TARGETS MET!")

    print()

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = RESULTS_DIR / f"production_validation_{timestamp}.csv"
    trades_df.to_csv(results_file, index=False)
    print(f"Results saved: {results_file}")

print("="*80)
print("✅ VALIDATION COMPLETE")
print("="*80)
