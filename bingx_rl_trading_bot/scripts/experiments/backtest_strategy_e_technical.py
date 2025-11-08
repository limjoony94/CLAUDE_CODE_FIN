"""
Backtest Strategy E: Pure Technical (EMA Crossover)
====================================================

Entry: EMA(9) × EMA(21) crossover + RSI + volume confirmation
Exit: Fixed 2:1 R:R + trailing stop
No ML models required

Created: 2025-10-30
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
from datetime import datetime

# Configuration
LEVERAGE = 4
INITIAL_BALANCE = 10000
TAKER_FEE = 0.0005

# Entry thresholds
RSI_LONG_MIN = 50
RSI_SHORT_MAX = 50

# Exit parameters
TAKE_PROFIT = 0.015  # 1.5% leveraged = 0.375% price
STOP_LOSS = 0.0075   # 0.75% leveraged = 0.1875% price
MAX_HOLD = 120       # 10 hours
TRAILING_ACTIVATION = 0.01  # 1.0% leveraged
TRAILING_DISTANCE = 0.004   # 0.4% leveraged

# Data
DATA_DIR = PROJECT_ROOT / "data" / "features"
CSV_FILE = DATA_DIR / "BTCUSDT_5m_features_exit_ready.csv"

print("=" * 80)
print("STRATEGY E: PURE TECHNICAL (EMA CROSSOVER)")
print("=" * 80)
print()

# Load data
print("Loading data...")
df = pd.read_csv(CSV_FILE)
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values('timestamp').reset_index(drop=True)
print(f"✅ Loaded {len(df):,} candles")
print(f"   Period: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")
print()

# Ensure required indicators exist
required = ['ema_9', 'ema_21', 'rsi', 'volume']
missing = [col for col in required if col not in df.columns]
if missing:
    print(f"⚠️ Missing indicators: {missing}")
    print("   Calculating now...")

    if 'ema_9' not in df.columns:
        df['ema_9'] = df['close'].ewm(span=9, adjust=False).mean()
    if 'ema_21' not in df.columns:
        df['ema_21'] = df['close'].ewm(span=21, adjust=False).mean()
    if 'rsi' not in df.columns:
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = -delta.where(delta < 0, 0).rolling(14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))

    df = df.fillna(method='bfill').fillna(0)
    print("✅ Indicators calculated")
    print()

# Entry signal detection
def check_entry_signal(df, idx):
    """EMA Crossover + RSI + Volume confirmation"""
    if idx < 1:
        return None

    ema_9 = df['ema_9'].iloc[idx]
    ema_21 = df['ema_21'].iloc[idx]
    ema_9_prev = df['ema_9'].iloc[idx - 1]
    ema_21_prev = df['ema_21'].iloc[idx - 1]

    rsi = df['rsi'].iloc[idx]
    volume = df['volume'].iloc[idx]
    volume_avg = df['volume'].iloc[max(0, idx-20):idx].mean()

    # LONG Signal: EMA crossover up + RSI bullish + volume
    if (ema_9 > ema_21 and ema_9_prev <= ema_21_prev and
        rsi > RSI_LONG_MIN and
        volume > volume_avg):
        return 'LONG'

    # SHORT Signal: EMA crossover down + RSI bearish + volume
    if (ema_9 < ema_21 and ema_9_prev >= ema_21_prev and
        rsi < RSI_SHORT_MAX and
        volume > volume_avg):
        return 'SHORT'

    return None

# Exit signal detection
def check_exit_signal(entry_price, current_price, side, hold_time, highest_pnl):
    """Fixed 2:1 R:R + Trailing Stop"""
    # Calculate P&L
    if side == 'LONG':
        pnl = (current_price - entry_price) / entry_price
    else:
        pnl = (entry_price - current_price) / entry_price

    leveraged_pnl = pnl * LEVERAGE

    # Take Profit (2:1 R:R)
    if leveraged_pnl >= TAKE_PROFIT:
        return 'take_profit', leveraged_pnl

    # Stop Loss (1:1 R:R for 2:1 system)
    if leveraged_pnl <= -STOP_LOSS:
        return 'stop_loss', leveraged_pnl

    # Trailing Stop (after profit reaches 1%)
    if highest_pnl >= TRAILING_ACTIVATION:
        trailing_threshold = highest_pnl - TRAILING_DISTANCE
        if leveraged_pnl < trailing_threshold:
            return 'trailing_stop', leveraged_pnl

    # Max Hold
    if hold_time >= MAX_HOLD:
        return 'max_hold', leveraged_pnl

    return None, leveraged_pnl

# Backtesting loop
print("Starting backtest...")
print()

balance = INITIAL_BALANCE
position = None
trades = []
peak_balance = INITIAL_BALANCE
max_drawdown = 0

for idx in range(100, len(df)):
    current_price = df['close'].iloc[idx]
    current_time = df['timestamp'].iloc[idx]

    # Check for entry if no position
    if position is None:
        signal = check_entry_signal(df, idx)

        if signal is not None:
            # Calculate position size (50% of balance)
            position_pct = 0.50
            position_value = balance * position_pct

            # Entry
            position = {
                'side': signal,
                'entry_idx': idx,
                'entry_price': current_price,
                'entry_time': current_time,
                'position_value': position_value,
                'hold_time': 0,
                'highest_pnl': 0
            }

    # Check for exit if in position
    elif position is not None:
        position['hold_time'] += 1

        # Calculate current P&L
        if position['side'] == 'LONG':
            pnl = (current_price - position['entry_price']) / position['entry_price']
        else:
            pnl = (position['entry_price'] - current_price) / position['entry_price']

        leveraged_pnl = pnl * LEVERAGE

        # Track highest P&L for trailing stop
        if leveraged_pnl > position['highest_pnl']:
            position['highest_pnl'] = leveraged_pnl

        # Check exit
        exit_reason, final_pnl = check_exit_signal(
            position['entry_price'],
            current_price,
            position['side'],
            position['hold_time'],
            position['highest_pnl']
        )

        if exit_reason is not None:
            # Calculate fees
            fee_total = 2 * TAKER_FEE * LEVERAGE  # Entry + Exit
            final_pnl_after_fees = final_pnl - fee_total

            # Calculate dollar P&L
            pnl_dollars = position['position_value'] * final_pnl_after_fees

            # Update balance
            balance += pnl_dollars

            # Track drawdown
            if balance > peak_balance:
                peak_balance = balance
            drawdown = (peak_balance - balance) / peak_balance
            if drawdown > max_drawdown:
                max_drawdown = drawdown

            # Record trade
            trade = {
                'entry_time': position['entry_time'],
                'exit_time': current_time,
                'side': position['side'],
                'entry_price': position['entry_price'],
                'exit_price': current_price,
                'hold_time': position['hold_time'],
                'pnl_pct': final_pnl_after_fees,
                'pnl_dollars': pnl_dollars,
                'balance': balance,
                'exit_reason': exit_reason
            }
            trades.append(trade)

            # Clear position
            position = None

# Results
print("=" * 80)
print("BACKTEST RESULTS - STRATEGY E")
print("=" * 80)
print()

if len(trades) == 0:
    print("⚠️ No trades executed")
    sys.exit(0)

# Convert to DataFrame
trades_df = pd.DataFrame(trades)

# Calculate metrics
total_return = (balance - INITIAL_BALANCE) / INITIAL_BALANCE
win_trades = trades_df[trades_df['pnl_pct'] > 0]
loss_trades = trades_df[trades_df['pnl_pct'] <= 0]
win_rate = len(win_trades) / len(trades_df)

avg_win = win_trades['pnl_pct'].mean() if len(win_trades) > 0 else 0
avg_loss = loss_trades['pnl_pct'].mean() if len(loss_trades) > 0 else 0
avg_trade = trades_df['pnl_pct'].mean()

profit_factor = abs(win_trades['pnl_pct'].sum() / loss_trades['pnl_pct'].sum()) if len(loss_trades) > 0 and loss_trades['pnl_pct'].sum() != 0 else float('inf')

# Exit distribution
exit_counts = trades_df['exit_reason'].value_counts()

# Performance metrics
print(f"Initial Balance: ${INITIAL_BALANCE:,.2f}")
print(f"Final Balance: ${balance:,.2f}")
print(f"Total Return: {total_return*100:+.2f}%")
print(f"Max Drawdown: {max_drawdown*100:.2f}%")
print()

print(f"Total Trades: {len(trades_df)}")
print(f"Winning Trades: {len(win_trades)} ({len(win_trades)/len(trades_df)*100:.1f}%)")
print(f"Losing Trades: {len(loss_trades)} ({len(loss_trades)/len(trades_df)*100:.1f}%)")
print(f"Win Rate: {win_rate*100:.2f}%")
print()

print(f"Average Trade: {avg_trade*100:+.4f}%")
print(f"Average Win: {avg_win*100:+.4f}%")
print(f"Average Loss: {avg_loss*100:.4f}%")
print(f"Profit Factor: {profit_factor:.2f}x")
print()

print("Exit Distribution:")
for reason, count in exit_counts.items():
    pct = count / len(trades_df) * 100
    print(f"  {reason}: {count} ({pct:.1f}%)")
print()

# LONG vs SHORT
long_trades = trades_df[trades_df['side'] == 'LONG']
short_trades = trades_df[trades_df['side'] == 'SHORT']
print(f"LONG Trades: {len(long_trades)} ({len(long_trades)/len(trades_df)*100:.1f}%)")
print(f"SHORT Trades: {len(short_trades)} ({len(short_trades)/len(trades_df)*100:.1f}%)")
print()

# Hold time analysis
print(f"Average Hold Time: {trades_df['hold_time'].mean():.1f} candles ({trades_df['hold_time'].mean()/12:.2f} hours)")
print(f"Max Hold Time: {trades_df['hold_time'].max()} candles")
print()

# Save results
results_dir = PROJECT_ROOT / "results"
results_dir.mkdir(exist_ok=True)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_file = results_dir / f"strategy_e_technical_{timestamp}.csv"
trades_df.to_csv(results_file, index=False)
print(f"✅ Results saved: {results_file}")
print()

print("=" * 80)
print("STRATEGY E BACKTEST COMPLETE")
print("=" * 80)
