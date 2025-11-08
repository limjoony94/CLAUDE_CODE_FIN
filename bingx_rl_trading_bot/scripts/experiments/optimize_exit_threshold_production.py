"""
Optimize Exit Threshold - Production Configuration
==================================================

Current Problem:
  - Exit threshold 0.75 → ML Exit 0% (too high)
  - All exits via Max Hold (91.1%)
  - Avg Hold 116 candles (target: 20-30)

Solution:
  - Grid search Exit thresholds [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]
  - Find threshold that achieves:
    * ML Exit: 75-85%
    * Avg Hold: 20-30 candles
    * Win Rate: 70-75%
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
LEVERAGE = 4
EMERGENCY_STOP_LOSS = -0.03
EMERGENCY_MAX_HOLD = 120
INITIAL_BALANCE = 10000
WINDOW_SIZE = 5

LONG_ENTRY_THRESHOLD = 0.80
SHORT_ENTRY_THRESHOLD = 0.80

# Exit threshold grid search
EXIT_THRESHOLDS = [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50]

DATA_DIR = PROJECT_ROOT / "data" / "features"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

print("="*80)
print("EXIT THRESHOLD OPTIMIZATION - PRODUCTION CONFIG")
print("="*80)
print()
print(f"Testing Exit Thresholds: {EXIT_THRESHOLDS}")
print()

# Load data
print("Loading data...")
df = pd.read_csv(DATA_DIR / "BTCUSDT_5m_features.csv")

# Enhanced features
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

# Load models
print("Loading models...")
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

with open(MODELS_DIR / "xgboost_long_exit_oppgating_improved_20251024_043527.pkl", 'rb') as f:
    long_exit_model = pickle.load(f)
with open(MODELS_DIR / "xgboost_long_exit_oppgating_improved_20251024_043527_features.txt", 'r') as f:
    long_exit_features = [line.strip() for line in f.readlines() if line.strip()]

with open(MODELS_DIR / "xgboost_short_exit_oppgating_improved_20251024_044510.pkl", 'rb') as f:
    short_exit_model = pickle.load(f)
with open(MODELS_DIR / "xgboost_short_exit_oppgating_improved_20251024_044510_features.txt", 'r') as f:
    short_exit_features = [line.strip() for line in f.readlines() if line.strip()]

print(f"✅ Models loaded")
print()

candles_per_window = WINDOW_SIZE * 288
total_windows = (len(df) - 100) // candles_per_window

def run_backtest(exit_threshold):
    """Run backtest with given exit threshold"""
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

                if pnl_pct <= EMERGENCY_STOP_LOSS:
                    exit_reason = 'STOP_LOSS'
                elif hold_time >= EMERGENCY_MAX_HOLD:
                    exit_reason = 'MAX_HOLD'
                else:
                    if side == 'LONG':
                        exit_features_df = window_df.iloc[[idx]][long_exit_features]
                        exit_prob = long_exit_model.predict_proba(exit_features_df.values)[0, 1]
                        if exit_prob >= exit_threshold:
                            exit_reason = 'ML_EXIT'
                    else:
                        exit_features_df = window_df.iloc[[idx]][short_exit_features]
                        exit_prob = short_exit_model.predict_proba(exit_features_df.values)[0, 1]
                        if exit_prob >= exit_threshold:
                            exit_reason = 'ML_EXIT'

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

    return all_trades, balance

# Grid search
print("="*80)
print("GRID SEARCH RESULTS")
print("="*80)
print()

results = []

for exit_threshold in EXIT_THRESHOLDS:
    all_trades, final_balance = run_backtest(exit_threshold)

    if len(all_trades) == 0:
        print(f"Exit {exit_threshold:.2f}: NO TRADES")
        continue

    trades_df = pd.DataFrame(all_trades)

    total_trades = len(trades_df)
    wins = len(trades_df[trades_df['net_pnl'] > 0])
    win_rate = (wins / total_trades * 100)

    total_return = ((final_balance - INITIAL_BALANCE) / INITIAL_BALANCE) * 100
    avg_return = total_return / total_windows

    ml_exits = len(trades_df[trades_df['exit_reason'] == 'ML_EXIT'])
    ml_exit_pct = (ml_exits / total_trades * 100)

    avg_hold = trades_df['hold_time'].mean()

    result = {
        'exit_threshold': exit_threshold,
        'total_trades': total_trades,
        'win_rate': win_rate,
        'avg_return': avg_return,
        'ml_exit_pct': ml_exit_pct,
        'avg_hold': avg_hold,
        'final_balance': final_balance
    }
    results.append(result)

    print(f"Exit {exit_threshold:.2f}: WR {win_rate:.1f}% | Ret {avg_return:.1f}% | ML {ml_exit_pct:.1f}% | Hold {avg_hold:.1f}")

df_results = pd.DataFrame(results)

# Score by target achievement
df_results['wr_score'] = np.clip(df_results['win_rate'] / 72.5, 0, 1.2)  # target 70-75%
df_results['ml_score'] = np.clip(df_results['ml_exit_pct'] / 80, 0, 1.2)  # target 75-85%
df_results['hold_score'] = 1 - np.abs(df_results['avg_hold'] - 25) / 100  # target 20-30
df_results['hold_score'] = np.clip(df_results['hold_score'], 0, 1)

df_results['composite_score'] = (
    df_results['wr_score'] * 0.3 +
    df_results['ml_score'] * 0.4 +  # ML Exit is critical
    df_results['hold_score'] * 0.3
)

df_results = df_results.sort_values('composite_score', ascending=False)

print()
print("="*80)
print("BEST CONFIGURATIONS")
print("="*80)
print()

print(df_results[['exit_threshold', 'win_rate', 'avg_return', 'ml_exit_pct', 'avg_hold', 'composite_score']].head(3).to_string(index=False))
print()

best = df_results.iloc[0]
print(f"✅ OPTIMAL EXIT THRESHOLD: {best['exit_threshold']:.2f}")
print(f"  Win Rate: {best['win_rate']:.1f}% (target: 70-75%)")
print(f"  Return: {best['avg_return']:.1f}% per window (target: 35-40%)")
print(f"  ML Exit: {best['ml_exit_pct']:.1f}% (target: 75-85%)")
print(f"  Avg Hold: {best['avg_hold']:.1f} candles (target: 20-30)")
print(f"  Composite Score: {best['composite_score']:.3f}")
print()

# Save
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_file = RESULTS_DIR / f"exit_threshold_optimization_{timestamp}.csv"
df_results.to_csv(results_file, index=False)
print(f"Results saved: {results_file}")

print("="*80)
print("✅ OPTIMIZATION COMPLETE")
print("="*80)
