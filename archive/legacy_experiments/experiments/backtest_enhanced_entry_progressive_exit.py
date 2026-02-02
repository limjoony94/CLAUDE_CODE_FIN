"""
Backtest: ACTUAL Production Entry + Progressive Window Exit
===========================================================

CORRECTED ANALYSIS:
  - Entry: Enhanced 20251024_012445 (ACTUAL production)
  - Entry Thresholds: 0.80/0.80 (ACTUAL production)
  - Exit: Progressive Window 20251031_223102 (NEW to test)
  - Exit Thresholds: Grid search [0.05-0.20]

Previous backtest FAILED because:
  - Used Walk-Forward Decoupled models (rolled back, not in production)
  - Used Entry threshold 0.75 (actual is 0.80)
  - Wrong model combination

Expected:
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
from datetime import datetime, timedelta

# Configuration
LEVERAGE = 4
EMERGENCY_STOP_LOSS = -0.03
EMERGENCY_MAX_HOLD = 120
INITIAL_BALANCE = 10000
WINDOW_SIZE = 5  # Days per window

# ACTUAL Production Entry Thresholds
LONG_ENTRY_THRESHOLD = 0.80  # ACTUAL production
SHORT_ENTRY_THRESHOLD = 0.80  # ACTUAL production

# Grid search for optimal Exit thresholds
EXIT_THRESHOLDS = [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20]

DATA_DIR = PROJECT_ROOT / "data" / "features"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

print("="*80)
print("ENHANCED ENTRY (PRODUCTION) + PROGRESSIVE EXIT - GRID SEARCH")
print("="*80)
print()
print("Entry Models (ACTUAL Production):")
print("  LONG: xgboost_long_entry_enhanced_20251024_012445.pkl")
print("  SHORT: xgboost_short_entry_enhanced_20251024_012445.pkl")
print("  Entry Thresholds: 0.80 / 0.80 (ACTUAL production)")
print()
print("Exit Models (NEW - Testing):")
print("  LONG: xgboost_long_exit_progressive_window_20251031_223102.pkl")
print("  SHORT: xgboost_short_exit_progressive_window_20251031_223102.pkl")
print()
print(f"Exit Threshold Grid Search: {EXIT_THRESHOLDS}")
print()

# Load data
print("-"*80)
print("Loading data...")
features_file = DATA_DIR / "BTCUSDT_5m_features.csv"
df = pd.read_csv(features_file)

# Add enhanced features
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

# Load Entry models (ACTUAL Production)
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

print("✅ Loaded Entry models")
print(f"  LONG Entry: {len(long_entry_features)} features")
print(f"  SHORT Entry: {len(short_entry_features)} features")
print()

# Load Exit models (Progressive Window)
print("Loading Exit models...")
with open(MODELS_DIR / "xgboost_long_exit_progressive_window_20251031_223102.pkl", 'rb') as f:
    long_exit_model = pickle.load(f)
with open(MODELS_DIR / "xgboost_long_exit_progressive_window_20251031_223102_features.txt", 'r') as f:
    long_exit_features = [line.strip() for line in f.readlines() if line.strip()]

with open(MODELS_DIR / "xgboost_short_exit_progressive_window_20251031_223102.pkl", 'rb') as f:
    short_exit_model = pickle.load(f)
with open(MODELS_DIR / "xgboost_short_exit_progressive_window_20251031_223102_features.txt", 'r') as f:
    short_exit_features = [line.strip() for line in f.readlines() if line.strip()]

print("✅ Loaded Exit models")
print(f"  LONG Exit: {len(long_exit_features)} features")
print(f"  SHORT Exit: {len(short_exit_features)} features")
print()

# Calculate windows
candles_per_day = 288
candles_per_window = WINDOW_SIZE * candles_per_day
total_windows = (len(df) - 100) // candles_per_window
print(f"Total windows: {total_windows}")
print()

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
                    else:
                        exit_features_df = window_df.iloc[[idx]][short_exit_features]
                        exit_prob = short_exit_model.predict_proba(exit_features_df.values)[0, 1]

                    if exit_prob >= exit_threshold:
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

            # Check for entry if no position
            if position is None and idx < len(window_df) - EMERGENCY_MAX_HOLD:
                long_features_df = window_df.iloc[[idx]][long_entry_features]
                short_features_df = window_df.iloc[[idx]][short_entry_features]

                # Scale features
                long_features_scaled = long_entry_scaler.transform(long_features_df.values)
                short_features_scaled = short_entry_scaler.transform(short_features_df.values)

                long_prob = long_entry_model.predict_proba(long_features_scaled)[0, 1]
                short_prob = short_entry_model.predict_proba(short_features_scaled)[0, 1]

                enter_side = None

                if long_prob >= LONG_ENTRY_THRESHOLD:
                    enter_side = 'LONG'
                elif short_prob >= SHORT_ENTRY_THRESHOLD:
                    # Opportunity gating
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

        # Close position at end of window
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
print("GRID SEARCH: EXIT THRESHOLDS")
print("="*80)
print()

results = []

for exit_threshold in EXIT_THRESHOLDS:
    print(f"Testing Exit Threshold: {exit_threshold}")
    print("-"*80)

    all_trades, final_balance = run_backtest(exit_threshold)

    if len(all_trades) == 0:
        print(f"  No trades executed")
        print()
        continue

    trades_df = pd.DataFrame(all_trades)

    # Calculate metrics
    total_trades = len(trades_df)
    wins = len(trades_df[trades_df['net_pnl'] > 0])
    losses = len(trades_df[trades_df['net_pnl'] <= 0])
    win_rate = (wins / total_trades * 100) if total_trades > 0 else 0

    total_return = ((final_balance - INITIAL_BALANCE) / INITIAL_BALANCE) * 100
    avg_return = total_return / total_windows if total_windows > 0 else 0

    ml_exits = len(trades_df[trades_df['exit_reason'] == 'ML_EXIT'])
    ml_exit_pct = (ml_exits / total_trades * 100) if total_trades > 0 else 0

    avg_hold = trades_df['hold_time'].mean()

    long_trades = len(trades_df[trades_df['side'] == 'LONG'])
    short_trades = len(trades_df[trades_df['side'] == 'SHORT'])
    long_pct = (long_trades / total_trades * 100) if total_trades > 0 else 0
    short_pct = (short_trades / total_trades * 100) if total_trades > 0 else 0

    result = {
        'exit_threshold': exit_threshold,
        'total_trades': total_trades,
        'win_rate': win_rate,
        'avg_return': avg_return,
        'ml_exit_pct': ml_exit_pct,
        'avg_hold': avg_hold,
        'long_pct': long_pct,
        'short_pct': short_pct,
        'final_balance': final_balance
    }
    results.append(result)

    print(f"  Trades: {total_trades} ({wins}W / {losses}L)")
    print(f"  Win Rate: {win_rate:.1f}%")
    print(f"  Return: {avg_return:.2f}% per window")
    print(f"  ML Exit: {ml_exit_pct:.1f}%")
    print(f"  Avg Hold: {avg_hold:.1f} candles")
    print(f"  LONG/SHORT: {long_pct:.1f}% / {short_pct:.1f}%")
    print()

print("="*80)
print("GRID SEARCH RESULTS SUMMARY")
print("="*80)
print()

if len(results) > 0:
    df_results = pd.DataFrame(results)

    # Composite score (target-based)
    df_results['score'] = (
        (df_results['win_rate'] / 75) * 0.3 +      # 30% weight on Win Rate (target 70-75%)
        (df_results['avg_return'] / 40) * 0.3 +    # 30% weight on Return (target 35-40%)
        (df_results['ml_exit_pct'] / 80) * 0.2 +   # 20% weight on ML Exit (target 75-85%)
        (1 - abs(df_results['avg_hold'] - 25) / 25) * 0.2  # 20% weight on Hold Time (target 20-30)
    )

    df_results = df_results.sort_values('score', ascending=False)

    print(df_results.to_string(index=False))
    print()

    best = df_results.iloc[0]
    print(f"✅ BEST CONFIGURATION:")
    print(f"  Exit Threshold: {best['exit_threshold']}")
    print(f"  Composite Score: {best['score']:.3f}")
    print(f"  Win Rate: {best['win_rate']:.1f}% (target: 70-75%)")
    print(f"  Return: {best['avg_return']:.2f}% per window (target: 35-40%)")
    print(f"  ML Exit: {best['ml_exit_pct']:.1f}% (target: 75-85%)")
    print(f"  Avg Hold: {best['avg_hold']:.1f} candles (target: 20-30)")
    print()

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = RESULTS_DIR / f"enhanced_entry_progressive_exit_grid_{timestamp}.csv"
    df_results.to_csv(results_file, index=False)
    print(f"Results saved: {results_file}")

print("="*80)
print("✅ GRID SEARCH COMPLETE")
print("="*80)
