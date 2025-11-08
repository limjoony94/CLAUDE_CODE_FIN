"""
Backtest: Production Entry + Progressive Window Exit
======================================================

CRITICAL INSIGHT FROM ANALYSIS:
  - Entry models are the primary problem
  - Production Entry (Walk-Forward Decoupled): 73.86% WR ✅
  - Full Dataset Entry: 14.92% WR ❌

SOLUTION:
  - Entry: Walk-Forward Decoupled (proven)
  - Exit: Progressive Window (new)
  - Threshold: Grid search [0.05-0.20] for optimal ML Exit rate

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
from datetime import datetime

# Configuration
LEVERAGE = 4
EMERGENCY_STOP_LOSS = -0.03
EMERGENCY_MAX_HOLD = 120
INITIAL_BALANCE = 10000
WINDOW_SIZE = 5  # Days per window

# Thresholds to test
LONG_ENTRY_THRESHOLD = 0.75
SHORT_ENTRY_THRESHOLD = 0.75

# Grid search for optimal Exit thresholds
EXIT_THRESHOLDS = [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20]

DATA_DIR = PROJECT_ROOT / "data" / "features"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

print("="*80)
print("PRODUCTION ENTRY + PROGRESSIVE EXIT - GRID SEARCH")
print("="*80)
print()
print("Entry Models:")
print("  LONG: xgboost_long_entry_walkforward_decoupled_20251027_194313.pkl")
print("  SHORT: xgboost_short_entry_walkforward_decoupled_20251027_194313.pkl")
print()
print("Exit Models:")
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
    df['bb_upper'] = df['bb_high']
    df['bb_lower'] = df['bb_low']
else:
    df['bb_position'] = 0.5

df['close_return'] = df['close'].pct_change().fillna(0)
df['volume_return'] = df['volume'].pct_change().fillna(0)
df['high_low_spread'] = ((df['high'] - df['low']) / df['close']).fillna(0)
df['candle_body_size'] = df['body_size']
df['adx'] = df.get('trend_strength', 0)
df['obv'] = df['volume'].cumsum()
df['mfi'] = df.get('rsi', 50)
df['higher_high'] = ((df['high'] > df['high'].shift(1)) & (df['high'].shift(1) > df['high'].shift(2))).astype(int)

support_level = df['low'].rolling(20).min()
df['near_support'] = (df['close'] <= support_level * 1.01).astype(int)

print(f"✅ Loaded {len(df):,} candles")
print()

# Load models
print("Loading models...")

# Production Entry models
with open(MODELS_DIR / 'xgboost_long_entry_walkforward_decoupled_20251027_194313.pkl', 'rb') as f:
    long_entry_model = pickle.load(f)

with open(MODELS_DIR / 'xgboost_short_entry_walkforward_decoupled_20251027_194313.pkl', 'rb') as f:
    short_entry_model = pickle.load(f)

# Progressive Window Exit models
with open(MODELS_DIR / 'xgboost_long_exit_progressive_window_20251031_223102.pkl', 'rb') as f:
    long_exit_model = pickle.load(f)

with open(MODELS_DIR / 'xgboost_short_exit_progressive_window_20251031_223102.pkl', 'rb') as f:
    short_exit_model = pickle.load(f)

# Load feature lists
with open(MODELS_DIR / 'xgboost_long_entry_walkforward_decoupled_20251027_194313_features.txt', 'r') as f:
    long_entry_features = [line.strip() for line in f.readlines()]

with open(MODELS_DIR / 'xgboost_short_entry_walkforward_decoupled_20251027_194313_features.txt', 'r') as f:
    short_entry_features = [line.strip() for line in f.readlines()]

with open(MODELS_DIR / 'xgboost_long_exit_progressive_window_20251031_223102_features.txt', 'r') as f:
    long_exit_features = [line.strip() for line in f.readlines()]

with open(MODELS_DIR / 'xgboost_short_exit_progressive_window_20251031_223102_features.txt', 'r') as f:
    short_exit_features = [line.strip() for line in f.readlines()]

print(f"✅ Loaded 4 models")
print(f"  LONG Entry: {len(long_entry_features)} features")
print(f"  SHORT Entry: {len(short_entry_features)} features")
print(f"  LONG Exit: {len(long_exit_features)} features")
print(f"  SHORT Exit: {len(short_exit_features)} features")
print()

# Calculate windows
CANDLES_PER_WINDOW = 1440
windows = []
for start_idx in range(0, len(df) - CANDLES_PER_WINDOW, CANDLES_PER_WINDOW):
    end_idx = start_idx + CANDLES_PER_WINDOW
    windows.append((start_idx, end_idx))

print(f"Total windows: {len(windows)}")
print()

# Grid search
print("="*80)
print("GRID SEARCH: EXIT THRESHOLDS")
print("="*80)
print()

grid_results = []

for exit_threshold in EXIT_THRESHOLDS:
    print(f"Testing Exit Threshold: {exit_threshold:.2f}")
    print("-"*80)

    all_trades = []

    for window_num, (start_idx, end_idx) in enumerate(windows, 1):
        window_df = df.iloc[start_idx:end_idx].copy().reset_index(drop=True)

        balance = INITIAL_BALANCE
        position = None

        for idx in range(len(window_df)):
            current_candle = window_df.iloc[idx]
            current_price = current_candle['close']

            # Check for exit if in position
            if position is not None:
                side = position['side']
                entry_price = position['entry_price']
                entry_idx = position['entry_idx']
                hold_time = idx - entry_idx

                # Calculate current P&L
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

                long_prob = long_entry_model.predict_proba(long_features_df.values)[0, 1]
                short_prob = short_entry_model.predict_proba(short_features_df.values)[0, 1]

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

    # Calculate metrics
    if len(all_trades) > 0:
        df_trades = pd.DataFrame(all_trades)

        total_trades = len(df_trades)
        winners = df_trades[df_trades['net_pnl'] > 0]
        win_rate = len(winners) / total_trades * 100 if total_trades > 0 else 0

        ml_exit_pct = (df_trades['exit_reason'] == 'ML_EXIT').sum() / total_trades * 100
        avg_hold = df_trades['hold_time'].mean()

        # Per-window return
        window_returns = []
        for window_num in range(1, len(windows) + 1):
            window_trades = df_trades[df_trades['window'] == window_num]
            window_return = window_trades['net_pnl'].sum() / INITIAL_BALANCE * 100
            window_returns.append(window_return)

        avg_return = np.mean(window_returns)

        # Side distribution
        long_pct = (df_trades['side'] == 'LONG').sum() / total_trades * 100
        short_pct = (df_trades['side'] == 'SHORT').sum() / total_trades * 100

        result = {
            'exit_threshold': exit_threshold,
            'total_trades': total_trades,
            'win_rate': win_rate,
            'avg_return': avg_return,
            'ml_exit_pct': ml_exit_pct,
            'avg_hold': avg_hold,
            'long_pct': long_pct,
            'short_pct': short_pct
        }

        grid_results.append(result)

        print(f"  Trades: {total_trades}")
        print(f"  Win Rate: {win_rate:.2f}%")
        print(f"  Return/Window: {avg_return:+.2f}%")
        print(f"  ML Exit: {ml_exit_pct:.1f}%")
        print(f"  Avg Hold: {avg_hold:.1f} candles")
        print(f"  LONG/SHORT: {long_pct:.1f}% / {short_pct:.1f}%")
        print()
    else:
        print("  No trades executed")
        print()

# Summary
print("="*80)
print("GRID SEARCH RESULTS SUMMARY")
print("="*80)
print()

if len(grid_results) > 0:
    df_results = pd.DataFrame(grid_results)

    print(f"{'Threshold':<12} {'Trades':<8} {'Win Rate':<10} {'Return':<10} {'ML Exit':<10} {'Hold':<8} {'Target':<8}")
    print("-"*80)

    for _, row in df_results.iterrows():
        # Check targets
        targets_met = 0
        if 70 <= row['win_rate'] <= 75:
            targets_met += 1
        if 35 <= row['avg_return'] <= 40:
            targets_met += 1
        if 75 <= row['ml_exit_pct'] <= 85:
            targets_met += 1
        if 20 <= row['avg_hold'] <= 30:
            targets_met += 1

        marker = "✅" if targets_met >= 3 else ""

        print(f"{row['exit_threshold']:<12.2f} {row['total_trades']:<8} {row['win_rate']:<10.2f} {row['avg_return']:<+10.2f} {row['ml_exit_pct']:<10.1f} {row['avg_hold']:<8.1f} {targets_met}/4 {marker}")

    print()

    # Best configuration
    df_results['score'] = (
        (df_results['win_rate'] / 75) * 0.3 +
        (df_results['avg_return'] / 40) * 0.3 +
        (df_results['ml_exit_pct'] / 80) * 0.2 +
        (1 - abs(df_results['avg_hold'] - 25) / 25) * 0.2
    )

    best_idx = df_results['score'].idxmax()
    best = df_results.loc[best_idx]

    print("BEST CONFIGURATION:")
    print("-"*80)
    print(f"  Exit Threshold: {best['exit_threshold']:.2f}")
    print(f"  Win Rate: {best['win_rate']:.2f}% (Target: 70-75%)")
    print(f"  Return/Window: {best['avg_return']:+.2f}% (Target: +35-40%)")
    print(f"  ML Exit: {best['ml_exit_pct']:.1f}% (Target: 75-85%)")
    print(f"  Avg Hold: {best['avg_hold']:.1f} candles (Target: 20-30)")
    print(f"  Composite Score: {best['score']:.4f}")
    print()

    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = RESULTS_DIR / f"grid_search_production_entry_progressive_exit_{timestamp}.csv"
    df_results.to_csv(results_file, index=False)
    print(f"✅ Results saved: {results_file}")

print()
print("="*80)
print("✅ GRID SEARCH COMPLETE")
print("="*80)
