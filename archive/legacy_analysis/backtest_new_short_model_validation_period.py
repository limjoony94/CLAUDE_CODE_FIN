"""
Backtest NEW SHORT Model on Validation Period (Oct 28 - Nov 4)
================================================================

Tests new SHORT entry model on data NOT used for training.

Training Period: Sep 30 - Oct 28 (80%)
Validation Period: Oct 28 - Nov 4 (20%) ← BACKTEST ON THIS

Models:
  NEW SHORT Entry: xgboost_short_entry_with_new_features_20251104_213043
  LONG Entry: xgboost_long_entry_enhanced_20251024_012445 (existing)
  Exit: xgboost_*_exit_oppgating_improved_20251024_* (existing)
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# Import production feature calculation
from scripts.production.production_features_v1 import calculate_all_features_enhanced_v2
from scripts.production.production_exit_features_v1 import prepare_exit_features

# Configuration (same as production)
ENTRY_THRESHOLD_LONG = 0.80
ENTRY_THRESHOLD_SHORT = 0.80
ML_EXIT_THRESHOLD = 0.70
LEVERAGE = 4
STOP_LOSS = -0.03  # -3% balance-based
MAX_HOLD = 120  # 10 hours
FEE_RATE = 0.0005  # 0.05% per trade

DATA_DIR = PROJECT_ROOT / "data" / "features"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

print("="*80)
print("BACKTEST NEW SHORT MODEL - VALIDATION PERIOD")
print("="*80)
print()
print("📅 Test Period: Oct 28 - Nov 4, 2025")
print("   (20% validation set - NOT used in training)")
print()
print("Configuration:")
print(f"  Entry Threshold: LONG {ENTRY_THRESHOLD_LONG}, SHORT {ENTRY_THRESHOLD_SHORT}")
print(f"  ML Exit Threshold: {ML_EXIT_THRESHOLD}")
print(f"  Stop Loss: {STOP_LOSS*100}% (balance-based)")
print(f"  Max Hold: {MAX_HOLD} candles ({MAX_HOLD/12:.1f}h)")
print(f"  Leverage: {LEVERAGE}x")
print(f"  Fee Rate: {FEE_RATE*100}% per side")
print()

# ==============================================================================
# Load Data and Features
# ==============================================================================

print("-"*80)
print("STEP 1: Loading Data and Calculating Features")
print("-"*80)

csv_file = sorted(DATA_DIR.glob("BTCUSDT_5m_raw_35days_*.csv"), reverse=True)[0]
print(f"✅ Loading: {csv_file.name}")

df_raw = pd.read_csv(csv_file)
df_raw['timestamp'] = pd.to_datetime(df_raw['timestamp'])
for col in ['open', 'high', 'low', 'close', 'volume']:
    df_raw[col] = df_raw[col].astype(float)

print(f"   {len(df_raw):,} raw candles")
print()

print("⏳ Calculating features...")
df = calculate_all_features_enhanced_v2(df_raw.copy(), phase='phase1')
df = prepare_exit_features(df)
print(f"✅ {len(df):,} candles with features")
print()

# Extract validation period (Oct 28 - Nov 4)
split_date = '2025-10-28'
df_val = df[df['timestamp'] >= split_date].copy().reset_index(drop=True)

print(f"📊 Validation Period (Oct 28 - Nov 4):")
print(f"   Candles: {len(df_val):,}")
print(f"   Date range: {df_val['timestamp'].iloc[0]} to {df_val['timestamp'].iloc[-1]}")
print(f"   Price range: ${df_val['close'].min():,.1f} - ${df_val['close'].max():,.1f}")
print(f"   Price change: {(df_val['close'].iloc[-1] - df_val['close'].iloc[0]) / df_val['close'].iloc[0] * 100:+.2f}%")
print()

# ==============================================================================
# Load Models
# ==============================================================================

print("-"*80)
print("STEP 2: Loading Models")
print("-"*80)

# NEW SHORT Entry Model
short_entry_model = joblib.load(MODELS_DIR / "xgboost_short_entry_with_new_features_20251104_213043.pkl")
short_entry_scaler = joblib.load(MODELS_DIR / "xgboost_short_entry_with_new_features_20251104_213043_scaler.pkl")
with open(MODELS_DIR / "xgboost_short_entry_with_new_features_20251104_213043_features.txt", 'r') as f:
    short_entry_features = [line.strip() for line in f.readlines()]

print(f"✅ NEW SHORT Entry: {len(short_entry_features)} features")

# Existing LONG Entry Model
long_entry_model = joblib.load(MODELS_DIR / "xgboost_long_entry_enhanced_20251024_012445.pkl")
long_entry_scaler = joblib.load(MODELS_DIR / "xgboost_long_entry_enhanced_20251024_012445_scaler.pkl")
with open(MODELS_DIR / "xgboost_long_entry_enhanced_20251024_012445_features.txt", 'r') as f:
    long_entry_features = [line.strip() for line in f.readlines()]

print(f"✅ LONG Entry: {len(long_entry_features)} features")

# Exit Models
long_exit_model = joblib.load(MODELS_DIR / "xgboost_long_exit_oppgating_improved_20251024_043527.pkl")
long_exit_scaler = joblib.load(MODELS_DIR / "xgboost_long_exit_oppgating_improved_20251024_043527_scaler.pkl")
with open(MODELS_DIR / "xgboost_long_exit_oppgating_improved_20251024_043527_features.txt", 'r') as f:
    long_exit_features = [line.strip() for line in f.readlines()]

short_exit_model = joblib.load(MODELS_DIR / "xgboost_short_exit_oppgating_improved_20251024_044510.pkl")
short_exit_scaler = joblib.load(MODELS_DIR / "xgboost_short_exit_oppgating_improved_20251024_044510_scaler.pkl")
with open(MODELS_DIR / "xgboost_short_exit_oppgating_improved_20251024_044510_features.txt", 'r') as f:
    short_exit_features = [line.strip() for line in f.readlines()]

print(f"✅ Exit Models: LONG {len(long_exit_features)}, SHORT {len(short_exit_features)} features")
print()

# ==============================================================================
# Calculate Probabilities
# ==============================================================================

print("-"*80)
print("STEP 3: Calculating Entry/Exit Probabilities")
print("-"*80)

# LONG Entry
X_long_entry = df_val[long_entry_features].values
X_long_entry_scaled = long_entry_scaler.transform(X_long_entry)
long_entry_probs = long_entry_model.predict_proba(X_long_entry_scaled)[:, 1]

# SHORT Entry
X_short_entry = df_val[short_entry_features].values
X_short_entry_scaled = short_entry_scaler.transform(X_short_entry)
short_entry_probs = short_entry_model.predict_proba(X_short_entry_scaled)[:, 1]

# LONG Exit
X_long_exit = df_val[long_exit_features].values
X_long_exit_scaled = long_exit_scaler.transform(X_long_exit)
long_exit_probs = long_exit_model.predict_proba(X_long_exit_scaled)[:, 1]

# SHORT Exit
X_short_exit = df_val[short_exit_features].values
X_short_exit_scaled = short_exit_scaler.transform(X_short_exit)
short_exit_probs = short_exit_model.predict_proba(X_short_exit_scaled)[:, 1]

print(f"✅ Probabilities calculated for {len(df_val):,} candles")
print()

print("📊 Signal Statistics:")
print(f"   LONG Entry >0.80: {(long_entry_probs >= ENTRY_THRESHOLD_LONG).sum():,} ({(long_entry_probs >= ENTRY_THRESHOLD_LONG).sum()/len(df_val)*100:.1f}%)")
print(f"   SHORT Entry >0.80: {(short_entry_probs >= ENTRY_THRESHOLD_SHORT).sum():,} ({(short_entry_probs >= ENTRY_THRESHOLD_SHORT).sum()/len(df_val)*100:.1f}%)")
print(f"   LONG Exit >0.70: {(long_exit_probs >= ML_EXIT_THRESHOLD).sum():,} ({(long_exit_probs >= ML_EXIT_THRESHOLD).sum()/len(df_val)*100:.1f}%)")
print(f"   SHORT Exit >0.70: {(short_exit_probs >= ML_EXIT_THRESHOLD).sum():,} ({(short_exit_probs >= ML_EXIT_THRESHOLD).sum()/len(df_val)*100:.1f}%)")
print()

# ==============================================================================
# Run Backtest Simulation
# ==============================================================================

print("-"*80)
print("STEP 4: Running Backtest Simulation")
print("-"*80)

initial_balance = 100.0
balance = initial_balance
position = None  # {'side': 'LONG/SHORT', 'entry_idx': int, 'entry_price': float, 'entry_time': timestamp}
trades = []

for i in range(len(df_val)):
    current_time = df_val.iloc[i]['timestamp']
    current_price = df_val.iloc[i]['close']

    # Check if in position
    if position is not None:
        entry_idx = position['entry_idx']
        entry_price = position['entry_price']
        side = position['side']
        hold_time = i - entry_idx

        # Calculate current P&L
        if side == 'LONG':
            pnl_pct = (current_price - entry_price) / entry_price
            exit_prob = long_exit_probs[i]
        else:  # SHORT
            pnl_pct = (entry_price - current_price) / entry_price
            exit_prob = short_exit_probs[i]

        leveraged_pnl_pct = pnl_pct * LEVERAGE

        # Exit conditions
        exit_reason = None

        # 1. Stop Loss
        if leveraged_pnl_pct <= STOP_LOSS:
            exit_reason = f"Stop Loss ({STOP_LOSS*100:.1f}%)"

        # 2. ML Exit
        elif exit_prob >= ML_EXIT_THRESHOLD:
            exit_reason = f"ML Exit ({exit_prob:.4f})"

        # 3. Max Hold Time
        elif hold_time >= MAX_HOLD:
            exit_reason = f"Max Hold ({MAX_HOLD} candles)"

        # Execute exit if triggered
        if exit_reason:
            # Calculate final P&L
            pnl_usd = balance * leveraged_pnl_pct
            entry_fee = balance * FEE_RATE
            exit_fee = (balance + pnl_usd) * FEE_RATE
            total_fee = entry_fee + exit_fee
            pnl_net = pnl_usd - total_fee

            balance += pnl_net

            # Record trade
            trades.append({
                'entry_time': position['entry_time'],
                'exit_time': current_time,
                'side': side,
                'entry_price': entry_price,
                'exit_price': current_price,
                'hold_time': hold_time,
                'pnl_pct': pnl_pct,
                'leveraged_pnl_pct': leveraged_pnl_pct,
                'pnl_gross': pnl_usd,
                'entry_fee': entry_fee,
                'exit_fee': exit_fee,
                'total_fee': total_fee,
                'pnl_net': pnl_net,
                'balance_after': balance,
                'exit_reason': exit_reason
            })

            position = None

    # Check for new entry (only if no position)
    if position is None:
        long_signal = long_entry_probs[i] >= ENTRY_THRESHOLD_LONG
        short_signal = short_entry_probs[i] >= ENTRY_THRESHOLD_SHORT

        # Opportunity Gating: Choose side with higher EV
        if long_signal or short_signal:
            long_ev = long_entry_probs[i] if long_signal else 0
            short_ev = short_entry_probs[i] if short_signal else 0

            # SHORT only if EV(SHORT) > EV(LONG) + 0.001
            if short_ev > long_ev + 0.001:
                chosen_side = 'SHORT'
            elif long_signal:
                chosen_side = 'LONG'
            else:
                chosen_side = None

            if chosen_side:
                position = {
                    'side': chosen_side,
                    'entry_idx': i,
                    'entry_price': current_price,
                    'entry_time': current_time
                }

print(f"✅ Backtest complete: {len(trades)} trades executed")
print()

# ==============================================================================
# Analyze Results
# ==============================================================================

print("="*80)
print("BACKTEST RESULTS")
print("="*80)
print()

if len(trades) == 0:
    print("⚠️  No trades executed during validation period")
    print("   Check entry thresholds and signal generation")
    sys.exit(0)

df_trades = pd.DataFrame(trades)

# Overall Statistics
total_return = (balance - initial_balance) / initial_balance * 100
total_pnl = balance - initial_balance
num_trades = len(df_trades)

wins = df_trades[df_trades['pnl_net'] > 0]
losses = df_trades[df_trades['pnl_net'] <= 0]
win_rate = len(wins) / num_trades * 100 if num_trades > 0 else 0

avg_win = wins['pnl_net'].mean() if len(wins) > 0 else 0
avg_loss = losses['pnl_net'].mean() if len(losses) > 0 else 0
profit_factor = wins['pnl_net'].sum() / abs(losses['pnl_net'].sum()) if len(losses) > 0 and losses['pnl_net'].sum() != 0 else float('inf')

print("📊 Overall Performance:")
print(f"   Initial Balance: ${initial_balance:.2f}")
print(f"   Final Balance: ${balance:.2f}")
print(f"   Total Return: {total_return:+.2f}%")
print(f"   Total P&L: ${total_pnl:+.2f}")
print()

print(f"📈 Trade Statistics:")
print(f"   Total Trades: {num_trades}")
print(f"   Wins: {len(wins)} ({len(wins)/num_trades*100:.1f}%)")
print(f"   Losses: {len(losses)} ({len(losses)/num_trades*100:.1f}%)")
print(f"   Win Rate: {win_rate:.1f}%")
print()

print(f"💰 P&L Statistics:")
print(f"   Avg Win: ${avg_win:.2f}")
print(f"   Avg Loss: ${avg_loss:.2f}")
print(f"   Profit Factor: {profit_factor:.2f}")
print(f"   Total Fees: ${df_trades['total_fee'].sum():.2f}")
print(f"   Fee Ratio: {df_trades['total_fee'].sum() / abs(df_trades['pnl_gross'].sum()) * 100:.1f}%")
print()

# Side Distribution
long_trades = df_trades[df_trades['side'] == 'LONG']
short_trades = df_trades[df_trades['side'] == 'SHORT']

print(f"📊 Side Distribution:")
print(f"   LONG: {len(long_trades)} trades ({len(long_trades)/num_trades*100:.1f}%)")
if len(long_trades) > 0:
    long_win_rate = len(long_trades[long_trades['pnl_net'] > 0]) / len(long_trades) * 100
    long_pnl = long_trades['pnl_net'].sum()
    print(f"     Win Rate: {long_win_rate:.1f}%")
    print(f"     Total P&L: ${long_pnl:+.2f}")

print(f"   SHORT: {len(short_trades)} trades ({len(short_trades)/num_trades*100:.1f}%)")
if len(short_trades) > 0:
    short_win_rate = len(short_trades[short_trades['pnl_net'] > 0]) / len(short_trades) * 100
    short_pnl = short_trades['pnl_net'].sum()
    print(f"     Win Rate: {short_win_rate:.1f}%")
    print(f"     Total P&L: ${short_pnl:+.2f}")
print()

# Exit Reason Analysis
print(f"🚪 Exit Reasons:")
exit_reasons = df_trades['exit_reason'].value_counts()
for reason, count in exit_reasons.items():
    print(f"   {reason}: {count} ({count/num_trades*100:.1f}%)")
print()

# Hold Time Statistics
avg_hold = df_trades['hold_time'].mean()
print(f"⏱️  Hold Time:")
print(f"   Average: {avg_hold:.1f} candles ({avg_hold/12:.1f}h)")
print(f"   Min: {df_trades['hold_time'].min()} candles ({df_trades['hold_time'].min()/12:.1f}h)")
print(f"   Max: {df_trades['hold_time'].max()} candles ({df_trades['hold_time'].max()/12:.1f}h)")
print()

# ==============================================================================
# Focus on SHORT Performance
# ==============================================================================

if len(short_trades) > 0:
    print("-"*80)
    print("🎯 SHORT PERFORMANCE ANALYSIS (NEW MODEL)")
    print("-"*80)
    print()

    short_wins = short_trades[short_trades['pnl_net'] > 0]
    short_losses = short_trades[short_trades['pnl_net'] <= 0]

    print(f"📊 SHORT Trade Summary:")
    print(f"   Total SHORT trades: {len(short_trades)}")
    print(f"   Wins: {len(short_wins)} ({len(short_wins)/len(short_trades)*100:.1f}%)")
    print(f"   Losses: {len(short_losses)} ({len(short_losses)/len(short_trades)*100:.1f}%)")
    print(f"   Total P&L: ${short_trades['pnl_net'].sum():+.2f}")
    print(f"   Avg P&L: ${short_trades['pnl_net'].mean():+.2f}")
    print()

    print("🔝 Top 3 SHORT Trades:")
    top_short = short_trades.nlargest(3, 'pnl_net')
    for i, (idx, trade) in enumerate(top_short.iterrows(), 1):
        print(f"   {i}. {trade['entry_time']} → {trade['exit_time']}")
        print(f"      Entry: ${trade['entry_price']:,.1f} | Exit: ${trade['exit_price']:,.1f}")
        print(f"      P&L: ${trade['pnl_net']:+.2f} ({trade['leveraged_pnl_pct']*100:+.2f}%)")
        print(f"      Reason: {trade['exit_reason']}")
        print()

# ==============================================================================
# Save Results
# ==============================================================================

print("-"*80)
print("STEP 5: Saving Results")
print("-"*80)

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
csv_filename = f"backtest_new_short_validation_{timestamp}.csv"
csv_path = RESULTS_DIR / csv_filename

df_trades.to_csv(csv_path, index=False)
print(f"✅ Trades saved: {csv_path}")

# Save summary
summary_filename = f"backtest_new_short_validation_{timestamp}_summary.txt"
summary_path = RESULTS_DIR / summary_filename

with open(summary_path, 'w') as f:
    f.write("="*80 + "\n")
    f.write("BACKTEST NEW SHORT MODEL - VALIDATION PERIOD\n")
    f.write("="*80 + "\n\n")
    f.write(f"Test Period: Oct 28 - Nov 4, 2025\n")
    f.write(f"Total Candles: {len(df_val):,}\n")
    f.write(f"Price Range: ${df_val['close'].min():,.1f} - ${df_val['close'].max():,.1f}\n\n")
    f.write(f"Overall Performance:\n")
    f.write(f"  Total Return: {total_return:+.2f}%\n")
    f.write(f"  Total Trades: {num_trades}\n")
    f.write(f"  Win Rate: {win_rate:.1f}%\n")
    f.write(f"  Profit Factor: {profit_factor:.2f}\n\n")
    f.write(f"Side Distribution:\n")
    f.write(f"  LONG: {len(long_trades)} ({len(long_trades)/num_trades*100:.1f}%)\n")
    f.write(f"  SHORT: {len(short_trades)} ({len(short_trades)/num_trades*100:.1f}%)\n\n")
    if len(short_trades) > 0:
        f.write(f"SHORT Performance:\n")
        f.write(f"  Win Rate: {len(short_wins)/len(short_trades)*100:.1f}%\n")
        f.write(f"  Total P&L: ${short_trades['pnl_net'].sum():+.2f}\n")

print(f"✅ Summary saved: {summary_path}")
print()

print("="*80)
print("BACKTEST COMPLETE")
print("="*80)
print()
print(f"📁 Results:")
print(f"   - {csv_filename}")
print(f"   - {summary_filename}")
print()

if total_return > 0:
    print("✅ NEW Model shows POSITIVE return on validation data")
    print(f"   {len(short_trades)} SHORT trades generated")
    print(f"   Recommendation: Model ready for production deployment")
else:
    print("⚠️  NEW Model shows NEGATIVE return on validation data")
    print(f"   Review trade quality and adjust thresholds if needed")

print()
print("="*80)
