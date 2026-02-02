"""
Backtest Strategy A: Exit-Only (Rule Entry + ML Exit)
======================================================

Entry: Conservative rule-based (Price > EMA20, RSI bullish, MACD positive, Volume)
Exit: Proven ML Exit models (95% historical success rate)

Created: 2025-10-30
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
INITIAL_BALANCE = 10000
TAKER_FEE = 0.0005

# Exit parameters
EMERGENCY_STOP_LOSS = -0.03  # -3% balance
MAX_HOLD = 120              # 10 hours
ML_EXIT_THRESHOLD = 0.75    # 75% probability

# Data and models
DATA_DIR = PROJECT_ROOT / "data" / "features"
CSV_FILE = DATA_DIR / "BTCUSDT_5m_features_exit_ready.csv"
MODELS_DIR = PROJECT_ROOT / "models"

print("=" * 80)
print("STRATEGY A: EXIT-ONLY (RULE ENTRY + ML EXIT)")
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

# Load Exit models
print("Loading ML Exit models...")
long_exit_model_path = MODELS_DIR / "xgboost_long_exit_threshold_075_20251027_190512.pkl"
with open(long_exit_model_path, 'rb') as f:
    long_exit_model = pickle.load(f)

long_exit_scaler_path = MODELS_DIR / "xgboost_long_exit_threshold_075_20251027_190512_scaler.pkl"
long_exit_scaler = joblib.load(long_exit_scaler_path)

long_exit_features_path = MODELS_DIR / "xgboost_long_exit_threshold_075_20251027_190512_features.txt"
with open(long_exit_features_path, 'r') as f:
    long_exit_features = [line.strip() for line in f.readlines() if line.strip()]

short_exit_model_path = MODELS_DIR / "xgboost_short_exit_threshold_075_20251027_190512.pkl"
with open(short_exit_model_path, 'rb') as f:
    short_exit_model = pickle.load(f)

short_exit_scaler_path = MODELS_DIR / "xgboost_short_exit_threshold_075_20251027_190512_scaler.pkl"
short_exit_scaler = joblib.load(short_exit_scaler_path)

short_exit_features_path = MODELS_DIR / "xgboost_short_exit_threshold_075_20251027_190512_features.txt"
with open(short_exit_features_path, 'r') as f:
    short_exit_features = [line.strip() for line in f.readlines() if line.strip()]

print(f"✅ LONG Exit model loaded ({len(long_exit_features)} features)")
print(f"✅ SHORT Exit model loaded ({len(short_exit_features)} features)")
print()

# Ensure required indicators exist
required = ['ema_20', 'rsi', 'macd', 'macd_signal', 'volume']
missing = [col for col in required if col not in df.columns]
if missing:
    print(f"⚠️ Missing indicators: {missing}")
    print("   Calculating now...")

    if 'ema_20' not in df.columns:
        df['ema_20'] = df['close'].ewm(span=20, adjust=False).mean()
    if 'rsi' not in df.columns:
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = -delta.where(delta < 0, 0).rolling(14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
    if 'macd' not in df.columns or 'macd_signal' not in df.columns:
        ema_12 = df['close'].ewm(span=12, adjust=False).mean()
        ema_26 = df['close'].ewm(span=26, adjust=False).mean()
        df['macd'] = ema_12 - ema_26
        df['macd_signal'] = df['macd'].ewm(span=9, adjust=False).mean()

    df = df.fillna(method='bfill').fillna(0)
    print("✅ Indicators calculated")
    print()

# Entry signal detection
def check_entry_signal(df, idx):
    """Conservative rule-based entry"""
    close = df['close'].iloc[idx]
    ema_20 = df['ema_20'].iloc[idx]
    rsi = df['rsi'].iloc[idx]
    macd = df['macd'].iloc[idx]
    macd_signal = df['macd_signal'].iloc[idx]
    volume = df['volume'].iloc[idx]
    volume_avg = df['volume'].iloc[max(0, idx-20):idx].mean()

    # LONG: All conditions must be true
    if (close > ema_20 and                    # Uptrend
        rsi > 50 and rsi < 70 and             # Bullish but not overbought
        macd > macd_signal and                # Momentum positive
        volume > volume_avg):                 # Volume confirmation
        return 'LONG'

    # SHORT: Conservative (fewer trades)
    if (close < ema_20 and                    # Downtrend
        rsi < 50 and rsi > 30 and             # Bearish but not oversold
        macd < macd_signal and                # Momentum negative
        volume > volume_avg):                 # Volume confirmation
        return 'SHORT'

    return None

# Exit signal detection
def check_exit_signal(df, idx, position):
    """ML Exit + Emergency rules"""
    current_price = df['close'].iloc[idx]
    entry_price = position['entry_price']
    side = position['side']
    hold_time = position['hold_time']

    # Calculate P&L
    if side == 'LONG':
        pnl = (current_price - entry_price) / entry_price
    else:
        pnl = (entry_price - current_price) / entry_price

    leveraged_pnl = pnl * LEVERAGE

    # Emergency Stop Loss (balance-based)
    position_pct = position['position_value'] / position['balance_at_entry']
    price_sl_pct = abs(EMERGENCY_STOP_LOSS) / (position_pct * LEVERAGE)

    if leveraged_pnl <= EMERGENCY_STOP_LOSS:
        return 'stop_loss', leveraged_pnl

    # Max Hold
    if hold_time >= MAX_HOLD:
        return 'max_hold', leveraged_pnl

    # ML Exit
    try:
        if side == 'LONG':
            exit_feat = df[long_exit_features].iloc[idx:idx+1].values
            if not np.isnan(exit_feat).any():
                exit_scaled = long_exit_scaler.transform(exit_feat)
                exit_prob = long_exit_model.predict_proba(exit_scaled)[0][1]

                if exit_prob >= ML_EXIT_THRESHOLD:
                    return 'ml_exit', leveraged_pnl
        else:  # SHORT
            exit_feat = df[short_exit_features].iloc[idx:idx+1].values
            if not np.isnan(exit_feat).any():
                exit_scaled = short_exit_scaler.transform(exit_feat)
                exit_prob = short_exit_model.predict_proba(exit_scaled)[0][1]

                if exit_prob >= ML_EXIT_THRESHOLD:
                    return 'ml_exit', leveraged_pnl
    except Exception as e:
        pass  # Continue to emergency rules

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
                'balance_at_entry': balance,
                'hold_time': 0
            }

    # Check for exit if in position
    elif position is not None:
        position['hold_time'] += 1

        # Check exit
        exit_reason, final_pnl = check_exit_signal(df, idx, position)

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
print("BACKTEST RESULTS - STRATEGY A")
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
results_file = results_dir / f"strategy_a_exit_only_{timestamp}.csv"
trades_df.to_csv(results_file, index=False)
print(f"✅ Results saved: {results_file}")
print()

print("=" * 80)
print("STRATEGY A BACKTEST COMPLETE")
print("=" * 80)
