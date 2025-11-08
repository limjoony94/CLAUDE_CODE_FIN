"""
Backtest Strategy F: Volatility Breakout
==========================================

Entry: Bollinger Band squeeze + ATR expansion + breakout
Exit: Volatility contraction + profit targets

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

# Exit parameters
TAKE_PROFIT = 0.02   # 2% leveraged
STOP_LOSS = 0.01     # 1% leveraged (tight for breakouts)
MAX_HOLD = 120       # 10 hours

# Data
DATA_DIR = PROJECT_ROOT / "data" / "features"
CSV_FILE = DATA_DIR / "BTCUSDT_5m_features_exit_ready.csv"

print("=" * 80)
print("STRATEGY F: VOLATILITY BREAKOUT")
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

# Calculate required indicators
print("Calculating indicators...")

# Bollinger Bands
bb_period = 20
bb_std = 2
df['bb_mid'] = df['close'].rolling(bb_period).mean()
df['bb_std'] = df['close'].rolling(bb_period).std()
df['bb_upper'] = df['bb_mid'] + (bb_std * df['bb_std'])
df['bb_lower'] = df['bb_mid'] - (bb_std * df['bb_std'])
df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_mid']

# ATR (Average True Range)
df['high_low'] = df['high'] - df['low']
df['high_close'] = abs(df['high'] - df['close'].shift(1))
df['low_close'] = abs(df['low'] - df['close'].shift(1))
df['true_range'] = df[['high_low', 'high_close', 'low_close']].max(axis=1)
df['atr'] = df['true_range'].rolling(14).mean()

df = df.fillna(method='bfill').fillna(0)
print("✅ Indicators calculated")
print()

# Entry signal detection
def check_entry_signal(df, idx):
    """Bollinger Band Squeeze + ATR Expansion + Breakout"""
    if idx < 20:
        return None

    # Current values
    bb_width = df['bb_width'].iloc[idx]
    bb_width_avg = df['bb_width'].iloc[max(0,idx-20):idx].mean()

    atr = df['atr'].iloc[idx]
    atr_avg = df['atr'].iloc[max(0,idx-20):idx].mean()

    close = df['close'].iloc[idx]
    close_prev = df['close'].iloc[idx - 1]
    bb_upper = df['bb_upper'].iloc[idx]
    bb_lower = df['bb_lower'].iloc[idx]

    # Squeeze detection
    is_squeeze = bb_width < bb_width_avg * 0.7

    # Volatility expansion
    atr_expanding = atr > atr_avg * 1.2

    # LONG: Squeeze release + breakout up
    if (is_squeeze and
        atr_expanding and
        close > bb_upper and
        close > close_prev):
        return 'LONG'

    # SHORT: Squeeze release + breakdown
    if (is_squeeze and
        atr_expanding and
        close < bb_lower and
        close < close_prev):
        return 'SHORT'

    return None

# Exit signal detection
def check_exit_signal(entry_price, entry_atr, current_price, current_atr, side, hold_time):
    """Profit target + volatility contraction + stop loss"""
    # Calculate P&L
    if side == 'LONG':
        pnl = (current_price - entry_price) / entry_price
    else:
        pnl = (entry_price - current_price) / entry_price

    leveraged_pnl = pnl * LEVERAGE

    # Take Profit (2% leveraged)
    if leveraged_pnl >= TAKE_PROFIT:
        return 'take_profit', leveraged_pnl

    # Stop Loss (tight 1%)
    if leveraged_pnl <= -STOP_LOSS:
        return 'stop_loss', leveraged_pnl

    # Volatility Contraction (move exhausted)
    if current_atr < entry_atr * 0.7:
        return 'volatility_exit', leveraged_pnl

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
    current_atr = df['atr'].iloc[idx]

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
                'entry_atr': current_atr,
                'position_value': position_value,
                'hold_time': 0
            }

    # Check for exit if in position
    elif position is not None:
        position['hold_time'] += 1

        # Check exit
        exit_reason, final_pnl = check_exit_signal(
            position['entry_price'],
            position['entry_atr'],
            current_price,
            current_atr,
            position['side'],
            position['hold_time']
        )

        if exit_reason is not None:
            # Calculate fees
            fee_total = 2 * TAKER_FEE * LEVERAGE
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
print("BACKTEST RESULTS - STRATEGY F")
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
results_file = results_dir / f"strategy_f_volatility_{timestamp}.csv"
trades_df.to_csv(results_file, index=False)
print(f"✅ Results saved: {results_file}")
print()

print("=" * 80)
print("STRATEGY F BACKTEST COMPLETE")
print("=" * 80)
