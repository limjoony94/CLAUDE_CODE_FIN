"""
Realistic Backtest: CURRENT PRODUCTION MODELS (Enhanced 20251024)
=================================================================

Test current production models with REALISTIC constraints.
"""

import pandas as pd
import numpy as np
import pickle
import joblib
from pathlib import Path
import sys
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiments.retrain_exit_models_opportunity_gating import prepare_exit_features

DATA_DIR = PROJECT_ROOT / "data" / "features"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

print("="*80)
print("REALISTIC BACKTEST: CURRENT PRODUCTION MODELS")
print("="*80)
print()

# Configuration (TESTING PRODUCTION SETTINGS: 0.80 threshold)
LEVERAGE = 4
ENTRY_THRESHOLD = 0.80  # PRODUCTION setting (was 0.75 in previous test)
ML_EXIT_THRESHOLD_LONG = 0.80  # PRODUCTION setting (was 0.75)
ML_EXIT_THRESHOLD_SHORT = 0.80  # PRODUCTION setting (was 0.75)
EMERGENCY_STOP_LOSS = -0.03
EMERGENCY_MAX_HOLD = 120
GATE_THRESHOLD = 0.001

LONG_AVG_RETURN = 0.0041
SHORT_AVG_RETURN = 0.0047

INITIAL_CAPITAL = 10000.0
TAKER_FEE = 0.0005

print(f"Configuration (TESTING 0.80 THRESHOLD):")
print(f"  Entry Threshold: {ENTRY_THRESHOLD}")
print(f"  ML Exit Threshold: {ML_EXIT_THRESHOLD_LONG}")
print(f"  Stop Loss: {EMERGENCY_STOP_LOSS}")
print(f"  Max Hold: {EMERGENCY_MAX_HOLD} candles")
print()

# Load Data
print("-"*80)
print("STEP 1: Loading Data")
print("-"*80)

df = pd.read_csv(DATA_DIR / "BTCUSDT_5m_features.csv")
print(f"✅ Loaded {len(df):,} candles")

df = prepare_exit_features(df)
print(f"✅ Exit features added")
print()

# Load Models
print("-"*80)
print("STEP 2: Loading PRODUCTION Models")
print("-"*80)

with open(MODELS_DIR / "xgboost_long_entry_enhanced_20251024_012445.pkl", 'rb') as f:
    long_entry_model = pickle.load(f)
long_entry_scaler = joblib.load(MODELS_DIR / "xgboost_long_entry_enhanced_20251024_012445_scaler.pkl")
with open(MODELS_DIR / "xgboost_long_entry_enhanced_20251024_012445_features.txt", 'r') as f:
    long_entry_features = [line.strip() for line in f]

with open(MODELS_DIR / "xgboost_short_entry_enhanced_20251024_012445.pkl", 'rb') as f:
    short_entry_model = pickle.load(f)
short_entry_scaler = joblib.load(MODELS_DIR / "xgboost_short_entry_enhanced_20251024_012445_scaler.pkl")
with open(MODELS_DIR / "xgboost_short_entry_enhanced_20251024_012445_features.txt", 'r') as f:
    short_entry_features = [line.strip() for line in f]

print(f"  ✅ LONG Entry: {len(long_entry_features)} features")
print(f"  ✅ SHORT Entry: {len(short_entry_features)} features")
print()

print("-"*80)
print("STEP 3: Loading Exit Models")
print("-"*80)

with open(MODELS_DIR / "xgboost_long_exit_oppgating_improved_20251024_043527.pkl", 'rb') as f:
    long_exit_model = pickle.load(f)
long_exit_scaler = joblib.load(MODELS_DIR / "xgboost_long_exit_oppgating_improved_20251024_043527_scaler.pkl")
with open(MODELS_DIR / "xgboost_long_exit_oppgating_improved_20251024_043527_features.txt", 'r') as f:
    long_exit_features = [line.strip() for line in f]

with open(MODELS_DIR / "xgboost_short_exit_oppgating_improved_20251024_044510.pkl", 'rb') as f:
    short_exit_model = pickle.load(f)
short_exit_scaler = joblib.load(MODELS_DIR / "xgboost_short_exit_oppgating_improved_20251024_044510_scaler.pkl")
with open(MODELS_DIR / "xgboost_short_exit_oppgating_improved_20251024_044510_features.txt", 'r') as f:
    short_exit_features = [line.strip() for line in f]

print(f"  ✅ LONG Exit: {len(long_exit_features)} features")
print(f"  ✅ SHORT Exit: {len(short_exit_features)} features")
print()

def calculate_position_size(signal_prob):
    if signal_prob < 0.5:
        return 0.20
    elif signal_prob >= 0.95:
        return 0.95
    else:
        return 0.20 + (signal_prob - 0.5) / 0.5 * 0.75

def simulate_trade(df, entry_idx, side, capital, position_size_pct, exit_model, exit_scaler, exit_features, ml_exit_threshold):
    entry_price = df['close'].iloc[entry_idx]
    position_value = capital * position_size_pct
    leveraged_value = position_value * LEVERAGE
    entry_fee = leveraged_value * TAKER_FEE

    max_hold_end = min(entry_idx + EMERGENCY_MAX_HOLD + 1, len(df))

    for i in range(1, max_hold_end - entry_idx):
        current_idx = entry_idx + i
        current_price = df['close'].iloc[current_idx]

        if side == 'LONG':
            pnl = (current_price - entry_price) / entry_price
        else:
            pnl = (entry_price - current_price) / entry_price

        leveraged_pnl_pct = pnl * LEVERAGE
        balance_pnl_pct = leveraged_pnl_pct * position_size_pct

        if balance_pnl_pct <= EMERGENCY_STOP_LOSS:
            exit_fee = leveraged_value * TAKER_FEE
            net_pnl = (balance_pnl_pct * capital) - entry_fee - exit_fee
            return {
                'exit_idx': current_idx,
                'pnl_pct': balance_pnl_pct,
                'net_pnl': net_pnl,
                'hold_time': i,
                'exit_reason': 'stop_loss'
            }

        try:
            exit_feat = df[exit_features].iloc[current_idx:current_idx+1].values
            if not np.isnan(exit_feat).any():
                exit_scaled = exit_scaler.transform(exit_feat)
                exit_prob = exit_model.predict_proba(exit_scaled)[0][1]

                if exit_prob >= ml_exit_threshold:
                    exit_fee = leveraged_value * TAKER_FEE
                    net_pnl = (balance_pnl_pct * capital) - entry_fee - exit_fee
                    return {
                        'exit_idx': current_idx,
                        'pnl_pct': balance_pnl_pct,
                        'net_pnl': net_pnl,
                        'hold_time': i,
                        'exit_reason': 'ml_exit'
                    }
        except:
            pass

        if i >= EMERGENCY_MAX_HOLD:
            exit_fee = leveraged_value * TAKER_FEE
            net_pnl = (balance_pnl_pct * capital) - entry_fee - exit_fee
            return {
                'exit_idx': current_idx,
                'pnl_pct': balance_pnl_pct,
                'net_pnl': net_pnl,
                'hold_time': i,
                'exit_reason': 'max_hold'
            }

    final_pnl = 0.0
    exit_fee = leveraged_value * TAKER_FEE
    net_pnl = (final_pnl * capital) - entry_fee - exit_fee
    return {
        'exit_idx': max_hold_end - 1,
        'pnl_pct': final_pnl,
        'net_pnl': net_pnl,
        'hold_time': max_hold_end - entry_idx - 1,
        'exit_reason': 'data_end'
    }

def backtest_window(df_window, starting_capital):
    X_long = df_window[long_entry_features].values
    X_long_scaled = long_entry_scaler.transform(X_long)
    long_probs = long_entry_model.predict_proba(X_long_scaled)[:, 1]

    X_short = df_window[short_entry_features].values
    X_short_scaled = short_entry_scaler.transform(X_short)
    short_probs = short_entry_model.predict_proba(X_short_scaled)[:, 1]

    trades = []
    current_capital = starting_capital

    i = 0
    while i < len(df_window):
        long_prob = long_probs[i]
        short_prob = short_probs[i]

        side = None
        entry_prob = 0.0

        if long_prob >= ENTRY_THRESHOLD and short_prob >= ENTRY_THRESHOLD:
            long_ev = long_prob * LONG_AVG_RETURN
            short_ev = short_prob * SHORT_AVG_RETURN
            opportunity_cost = short_ev - long_ev

            if opportunity_cost > GATE_THRESHOLD:
                side = 'SHORT'
                entry_prob = short_prob
            elif long_prob >= short_prob:
                side = 'LONG'
                entry_prob = long_prob
        elif long_prob >= ENTRY_THRESHOLD:
            side = 'LONG'
            entry_prob = long_prob
        elif short_prob >= ENTRY_THRESHOLD:
            long_ev = long_prob * LONG_AVG_RETURN
            short_ev = short_prob * SHORT_AVG_RETURN
            if short_ev - long_ev > GATE_THRESHOLD:
                side = 'SHORT'
                entry_prob = short_prob

        if side is None or i >= len(df_window) - EMERGENCY_MAX_HOLD:
            i += 1
            continue

        position_size_pct = calculate_position_size(entry_prob)

        if side == 'LONG':
            result = simulate_trade(
                df_window, i, side, current_capital,
                position_size_pct, long_exit_model, long_exit_scaler, long_exit_features, ML_EXIT_THRESHOLD_LONG
            )
        else:
            result = simulate_trade(
                df_window, i, side, current_capital,
                position_size_pct, short_exit_model, short_exit_scaler, short_exit_features, ML_EXIT_THRESHOLD_SHORT
            )

        result['entry_idx'] = i
        result['side'] = side
        result['entry_prob'] = entry_prob
        result['position_size_pct'] = position_size_pct
        result['capital_before'] = current_capital

        current_capital += result['net_pnl']
        result['capital_after'] = current_capital

        trades.append(result)
        i = result['exit_idx'] + 1

    return trades, current_capital

# Run Backtest
print("-"*80)
print("STEP 4: Running Realistic Backtest")
print("-"*80)
print()

window_size = 720
all_windows = []

for i in range(0, len(df) - window_size, window_size):
    window_df = df.iloc[i:i+window_size].reset_index(drop=True)
    all_windows.append(window_df)

print(f"Total windows: {len(all_windows)}")
print()

current_capital = INITIAL_CAPITAL
window_results = []

for window_idx, window_df in enumerate(all_windows):
    trades, ending_capital = backtest_window(window_df, current_capital)

    if len(trades) == 0:
        continue

    window_return = ((ending_capital - current_capital) / current_capital) * 100
    wins = sum(1 for t in trades if t['net_pnl'] > 0)
    win_rate = (wins / len(trades)) * 100 if len(trades) > 0 else 0

    ml_exits = sum(1 for t in trades if t['exit_reason'] == 'ml_exit')
    sl_exits = sum(1 for t in trades if t['exit_reason'] == 'stop_loss')
    max_hold_exits = sum(1 for t in trades if t['exit_reason'] == 'max_hold')

    long_trades = sum(1 for t in trades if t['side'] == 'LONG')
    short_trades = sum(1 for t in trades if t['side'] == 'SHORT')

    window_results.append({
        'window': window_idx,
        'total_trades': len(trades),
        'long_trades': long_trades,
        'short_trades': short_trades,
        'win_rate': win_rate,
        'window_return': window_return,
        'starting_capital': current_capital,
        'ending_capital': ending_capital,
        'ml_exit_rate': (ml_exits / len(trades)) * 100,
        'sl_exits': sl_exits,
        'max_hold_exits': max_hold_exits
    })

    current_capital = ending_capital

    if (window_idx + 1) % 20 == 0:
        print(f"  Window {window_idx+1}/{len(all_windows)} complete...")

print()
print(f"✅ Backtest complete: {len(window_results)} windows")
print()

# Results
df_results = pd.DataFrame(window_results)

print("="*80)
print("BACKTEST RESULTS: CURRENT PRODUCTION MODELS")
print("="*80)
print()
print(f"Total Windows: {len(df_results)}")
print(f"Total Trades: {df_results['total_trades'].sum():,}")
print(f"  LONG: {df_results['long_trades'].sum():,} ({df_results['long_trades'].sum() / df_results['total_trades'].sum() * 100:.1f}%)")
print(f"  SHORT: {df_results['short_trades'].sum():,} ({df_results['short_trades'].sum() / df_results['total_trades'].sum() * 100:.1f}%)")
print()

print(f"Overall Win Rate: {df_results['win_rate'].mean():.2f}%")
print(f"Avg Trades per Window: {df_results['total_trades'].mean():.1f}")
print(f"Trades per Day: {df_results['total_trades'].mean() / 5:.1f}")
print()

print(f"Capital Growth:")
print(f"  Initial: ${INITIAL_CAPITAL:,.2f}")
print(f"  Final: ${current_capital:,.2f}")
print(f"  Total Return: {((current_capital - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100:.2f}%")
print()

print(f"Exit Distribution:")
print(f"  ML Exit: {df_results['ml_exit_rate'].mean():.1f}%")
print(f"  Stop Loss: {df_results['sl_exits'].sum()} ({df_results['sl_exits'].sum() / df_results['total_trades'].sum() * 100:.1f}%)")
print(f"  Max Hold: {df_results['max_hold_exits'].sum()} ({df_results['max_hold_exits'].sum() / df_results['total_trades'].sum() * 100:.1f}%)")
print()

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_file = RESULTS_DIR / f"backtest_production_realistic_{timestamp}.csv"
df_results.to_csv(results_file, index=False)
print(f"✅ Results saved: {results_file.name}")
print()
