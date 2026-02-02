"""
Backtest Progressive Exit Window Models
==========================================

Progressive Exit Window Strategy:
  - Labels: ±5 candles around max profit with weights
  - Weights: {0: 1.0, ±1: 0.8, ±2: 0.7, ±3: 0.6, ±4: 0.5, ±5: 0.4}
  - Exit Threshold: 0.7 (70% probability)

Expected Performance:
  - Win Rate: 70-75% (vs current 14.92%)
  - Avg Hold: 20-30 candles (vs current 2.4)
  - Return: +35-40% per window (vs current -69.10%)
  - ML Exit: 75-85% (vs current 98.5%)

Models:
  - LONG Entry: xgboost_long_entry_full_dataset_20251031_184949.pkl
  - SHORT Entry: xgboost_short_entry_full_dataset_20251031_184949.pkl
  - LONG Exit: xgboost_long_exit_progressive_window_20251031_223102.pkl
  - SHORT Exit: xgboost_short_exit_progressive_window_20251031_223102.pkl
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
ML_EXIT_THRESHOLD_LONG = 0.70  # Progressive Window threshold
ML_EXIT_THRESHOLD_SHORT = 0.70  # Progressive Window threshold
LEVERAGE = 4
EMERGENCY_STOP_LOSS = -0.03
EMERGENCY_MAX_HOLD = 120
INITIAL_BALANCE = 10000
WINDOW_SIZE = 5  # Days per window

DATA_DIR = PROJECT_ROOT / "data" / "features"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

print("="*80)
print("PROGRESSIVE EXIT WINDOW BACKTEST")
print("="*80)
print()
print("Models:")
print("  LONG Entry: xgboost_long_entry_full_dataset_20251031_184949.pkl")
print("  SHORT Entry: xgboost_short_entry_full_dataset_20251031_184949.pkl")
print("  LONG Exit: xgboost_long_exit_progressive_window_20251031_223102.pkl")
print("  SHORT Exit: xgboost_short_exit_progressive_window_20251031_223102.pkl")
print()
print("Configuration:")
print(f"  Entry Threshold (LONG): {LONG_THRESHOLD}")
print(f"  Entry Threshold (SHORT): {SHORT_THRESHOLD}")
print(f"  Exit Threshold (LONG): {ML_EXIT_THRESHOLD_LONG} (Progressive Window)")
print(f"  Exit Threshold (SHORT): {ML_EXIT_THRESHOLD_SHORT} (Progressive Window)")
print(f"  Leverage: {LEVERAGE}x")
print(f"  Stop Loss: {EMERGENCY_STOP_LOSS*100}% balance")
print(f"  Max Hold: {EMERGENCY_MAX_HOLD} candles")
print()

# Load data
print("-"*80)
print("Loading Features Dataset")
print("-"*80)

features_file = DATA_DIR / "BTCUSDT_5m_features.csv"
df = pd.read_csv(features_file)

# Add Enhanced Exit Features (same as training)
print("  Adding enhanced Exit features...")

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
else:
    df['rsi_slope'] = 0
    df['rsi_overbought'] = 0
    df['rsi_oversold'] = 0
    df['rsi_divergence'] = 0

if 'macd' in df.columns and 'macd_signal' in df.columns:
    df['macd_histogram_slope'] = (df['macd'] - df['macd_signal']).diff(3).fillna(0)
else:
    df['macd_histogram_slope'] = 0

if 'bb_high' in df.columns and 'bb_low' in df.columns:
    bb_range = df['bb_high'] - df['bb_low']
    bb_range = bb_range.replace(0, 1e-10)
    df['bb_position'] = ((df['close'] - df['bb_low']) / bb_range).fillna(0.5)
    df['bb_upper'] = df['bb_high']
    df['bb_lower'] = df['bb_low']
else:
    df['bb_position'] = 0.5
    df['bb_upper'] = df['close']
    df['bb_lower'] = df['close']

df['close_return'] = df['close'].pct_change().fillna(0)
df['volume_return'] = df['volume'].pct_change().fillna(0)
df['high_low_spread'] = ((df['high'] - df['low']) / df['close']).fillna(0)
df['candle_body_size'] = df['body_size']

df['adx'] = df.get('trend_strength', 0)
df['obv'] = df['volume'].cumsum()
df['mfi'] = df.get('rsi', 50)

df['higher_high'] = ((df['high'] > df['high'].shift(1)) & (df['high'].shift(1) > df['high'].shift(2))).astype(int)

print(f"✅ Loaded {len(df):,} candles with enhanced features")
print()

# Load models
print("-"*80)
print("Loading Models")
print("-"*80)

with open(MODELS_DIR / 'xgboost_long_entry_full_dataset_20251031_184949.pkl', 'rb') as f:
    long_entry_model = pickle.load(f)

with open(MODELS_DIR / 'xgboost_short_entry_full_dataset_20251031_184949.pkl', 'rb') as f:
    short_entry_model = pickle.load(f)

with open(MODELS_DIR / 'xgboost_long_exit_progressive_window_20251031_223102.pkl', 'rb') as f:
    long_exit_model = pickle.load(f)

with open(MODELS_DIR / 'xgboost_short_exit_progressive_window_20251031_223102.pkl', 'rb') as f:
    short_exit_model = pickle.load(f)

# Load feature columns
with open(MODELS_DIR / 'xgboost_long_entry_full_dataset_20251031_184949_features.txt', 'r') as f:
    long_entry_features = [line.strip() for line in f.readlines()]

with open(MODELS_DIR / 'xgboost_short_entry_full_dataset_20251031_184949_features.txt', 'r') as f:
    short_entry_features = [line.strip() for line in f.readlines()]

with open(MODELS_DIR / 'xgboost_long_exit_progressive_window_20251031_223102_features.txt', 'r') as f:
    long_exit_features = [line.strip() for line in f.readlines()]

with open(MODELS_DIR / 'xgboost_short_exit_progressive_window_20251031_223102_features.txt', 'r') as f:
    short_exit_features = [line.strip() for line in f.readlines()]

print(f"✅ Loaded 4 models:")
print(f"  LONG Entry ({len(long_entry_features)} features)")
print(f"  SHORT Entry ({len(short_entry_features)} features)")
print(f"  LONG Exit ({len(long_exit_features)} features)")
print(f"  SHORT Exit ({len(short_exit_features)} features)")
print()

# Backtest
print("-"*80)
print("Running 108-Window Backtest")
print("-"*80)

# Calculate windows (5 days = 1440 candles)
CANDLES_PER_WINDOW = 1440
windows = []
for start_idx in range(0, len(df) - CANDLES_PER_WINDOW, CANDLES_PER_WINDOW):
    end_idx = start_idx + CANDLES_PER_WINDOW
    windows.append((start_idx, end_idx))

print(f"Total windows: {len(windows)}")
print()

all_trades = []
window_results = []

for window_num, (start_idx, end_idx) in enumerate(windows, 1):
    window_df = df.iloc[start_idx:end_idx].copy().reset_index(drop=True)

    # Track window state
    balance = INITIAL_BALANCE
    position = None
    window_trades = []

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

            # Emergency exits
            exit_reason = None

            # Stop Loss
            if pnl_pct <= EMERGENCY_STOP_LOSS:
                exit_reason = 'STOP_LOSS'

            # Max Hold
            elif hold_time >= EMERGENCY_MAX_HOLD:
                exit_reason = 'MAX_HOLD'

            # ML Exit
            else:
                # Prepare features
                if side == 'LONG':
                    exit_features_df = window_df.iloc[[idx]][long_exit_features]
                    exit_prob = long_exit_model.predict_proba(exit_features_df.values)[0, 1]
                    exit_threshold = ML_EXIT_THRESHOLD_LONG
                else:
                    exit_features_df = window_df.iloc[[idx]][short_exit_features]
                    exit_prob = short_exit_model.predict_proba(exit_features_df.values)[0, 1]
                    exit_threshold = ML_EXIT_THRESHOLD_SHORT

                if exit_prob >= exit_threshold:
                    exit_reason = 'ML_EXIT'

            # Execute exit
            if exit_reason is not None:
                # Calculate final P&L
                gross_pnl = position['position_size'] * pnl_pct
                fees = position['position_size'] * 0.0005 * 2  # Entry + exit
                net_pnl = gross_pnl - fees

                trade = {
                    'window': window_num,
                    'side': side,
                    'entry_idx': entry_idx,
                    'entry_price': entry_price,
                    'exit_idx': idx,
                    'exit_price': current_price,
                    'exit_reason': exit_reason,
                    'hold_time': hold_time,
                    'position_size': position['position_size'],
                    'gross_pnl': gross_pnl,
                    'fees': fees,
                    'net_pnl': net_pnl,
                    'leveraged_pnl_pct': pnl_pct
                }

                balance += net_pnl
                window_trades.append(trade)
                all_trades.append(trade)
                position = None

        # Check for entry if no position
        if position is None and idx < len(window_df) - EMERGENCY_MAX_HOLD:
            # Prepare features
            long_features_df = window_df.iloc[[idx]][long_entry_features]
            short_features_df = window_df.iloc[[idx]][short_entry_features]

            long_prob = long_entry_model.predict_proba(long_features_df.values)[0, 1]
            short_prob = short_entry_model.predict_proba(short_features_df.values)[0, 1]

            # Entry logic
            enter_side = None

            if long_prob >= LONG_THRESHOLD:
                enter_side = 'LONG'
            elif short_prob >= SHORT_THRESHOLD:
                # Opportunity gating
                long_ev = long_prob * 0.0041
                short_ev = short_prob * 0.0047
                opportunity_cost = short_ev - long_ev

                if opportunity_cost > 0.001:
                    enter_side = 'SHORT'

            # Enter position
            if enter_side is not None:
                position_size = balance * 0.95

                position = {
                    'side': enter_side,
                    'entry_idx': idx,
                    'entry_price': current_price,
                    'position_size': position_size
                }

    # Close position at end of window if still open
    if position is not None:
        idx = len(window_df) - 1
        current_price = window_df.iloc[idx]['close']
        side = position['side']
        entry_price = position['entry_price']
        entry_idx = position['entry_idx']
        hold_time = idx - entry_idx

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
            'entry_idx': entry_idx,
            'entry_price': entry_price,
            'exit_idx': idx,
            'exit_price': current_price,
            'exit_reason': 'WINDOW_END',
            'hold_time': hold_time,
            'position_size': position['position_size'],
            'gross_pnl': gross_pnl,
            'fees': fees,
            'net_pnl': net_pnl,
            'leveraged_pnl_pct': pnl_pct
        }

        balance += net_pnl
        window_trades.append(trade)
        all_trades.append(trade)
        position = None

    # Window stats
    window_return = ((balance - INITIAL_BALANCE) / INITIAL_BALANCE) * 100
    window_results.append({
        'window': window_num,
        'trades': len(window_trades),
        'return_pct': window_return
    })

    print(f"Window {window_num:3d}/108: {len(window_trades):2d} trades, Return: {window_return:+7.2f}%")

# Results
print()
print("="*80)
print("BACKTEST RESULTS")
print("="*80)
print()

df_trades = pd.DataFrame(all_trades)
df_windows = pd.DataFrame(window_results)

# Overall stats
total_trades = len(df_trades)
winners = df_trades[df_trades['net_pnl'] > 0]
losers = df_trades[df_trades['net_pnl'] <= 0]
win_rate = len(winners) / total_trades * 100 if total_trades > 0 else 0

avg_return = df_windows['return_pct'].mean()
std_return = df_windows['return_pct'].std()
sharpe = (avg_return / std_return) * np.sqrt(73) if std_return > 0 else 0

print("Overall Performance:")
print(f"  Total Trades: {total_trades:,}")
print(f"  Win Rate: {win_rate:.2f}% ({len(winners)}W / {len(losers)}L)")
print(f"  Avg Return per Window: {avg_return:+.2f}%")
print(f"  Return StdDev: {std_return:.2f}%")
print(f"  Sharpe Ratio: {sharpe:.3f} (annualized)")
print()

# Trade stats
print("Trade Statistics:")
print(f"  Avg Trade P&L: ${df_trades['net_pnl'].mean():.2f} ({df_trades['leveraged_pnl_pct'].mean()*100:.3f}%)")
print(f"  Avg Winner: ${winners['net_pnl'].mean():.2f} ({winners['leveraged_pnl_pct'].mean()*100:.3f}%)")
print(f"  Avg Loser: ${losers['net_pnl'].mean():.2f} ({losers['leveraged_pnl_pct'].mean()*100:.3f}%)")
print(f"  Avg Hold Time: {df_trades['hold_time'].mean():.1f} candles ({df_trades['hold_time'].mean()/12:.1f} hours)")
print()

# Exit distribution
print("Exit Distribution:")
for reason in df_trades['exit_reason'].unique():
    count = (df_trades['exit_reason'] == reason).sum()
    pct = count / total_trades * 100
    print(f"  {reason}: {count:,} ({pct:.1f}%)")
print()

# Side distribution
print("Side Distribution:")
for side in df_trades['side'].unique():
    count = (df_trades['side'] == side).sum()
    pct = count / total_trades * 100
    print(f"  {side}: {count:,} ({pct:.1f}%)")
print()

# Save results
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
results_file = RESULTS_DIR / f"backtest_progressive_window_{timestamp}.csv"
df_trades.to_csv(results_file, index=False)
print(f"✅ Results saved: {results_file}")
print()

# Comparison to previous
print("="*80)
print("COMPARISON TO PREVIOUS")
print("="*80)
print()

print("Metric                 Previous     Progressive   Improvement")
print("-" * 65)
print(f"Win Rate               14.92%       {win_rate:.2f}%        {win_rate-14.92:+.2f}pp")
print(f"Return per Window      -69.10%      {avg_return:+.2f}%        {avg_return+69.10:+.2f}pp")
print(f"Avg Hold Time          2.4          {df_trades['hold_time'].mean():.1f}           {df_trades['hold_time'].mean()-2.4:+.1f}")

ml_exit_pct = (df_trades['exit_reason'] == 'ML_EXIT').sum() / total_trades * 100
print(f"ML Exit Rate           98.5%        {ml_exit_pct:.1f}%          {ml_exit_pct-98.5:+.1f}pp")
print()

# Validation
print("="*80)
print("TARGET VALIDATION")
print("="*80)
print()

print("Target             Expected      Actual        Status")
print("-" * 65)
print(f"Win Rate           70-75%        {win_rate:.1f}%         {'✅ PASS' if win_rate >= 70 else '❌ FAIL'}")
print(f"Avg Hold           20-30         {df_trades['hold_time'].mean():.1f}           {'✅ PASS' if 20 <= df_trades['hold_time'].mean() <= 30 else '❌ FAIL'}")
print(f"Return/Window      +35-40%       {avg_return:+.1f}%         {'✅ PASS' if avg_return >= 35 else '❌ FAIL'}")
print(f"ML Exit            75-85%        {ml_exit_pct:.1f}%          {'✅ PASS' if 75 <= ml_exit_pct <= 85 else '❌ FAIL'}")
print()

print("="*80)
print("✅ BACKTEST COMPLETE")
print("="*80)
