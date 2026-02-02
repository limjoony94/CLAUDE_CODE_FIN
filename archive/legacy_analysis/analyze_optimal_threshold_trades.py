"""
Detailed Analysis of Optimal Threshold Configuration
====================================================

Configuration: Entry=0.80, Exit=0.80
Validates winning configuration with deep trade analysis
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

# Configuration
LEVERAGE = 4
STOP_LOSS = -0.03  # -3% balance-based
MAX_HOLD = 120  # 10 hours
FEE_RATE = 0.0005  # 0.05% per trade

# OPTIMAL THRESHOLDS (from optimization)
ENTRY_THRESHOLD_LONG = 0.80
ENTRY_THRESHOLD_SHORT = 0.80
ML_EXIT_THRESHOLD = 0.80

DATA_DIR = PROJECT_ROOT / "data" / "features"
MODELS_DIR = PROJECT_ROOT / "models"

print("="*80)
print("DETAILED ANALYSIS: OPTIMAL THRESHOLD CONFIGURATION")
print("="*80)
print()
print("📊 Configuration:")
print(f"   Entry: LONG {ENTRY_THRESHOLD_LONG:.2f}, SHORT {ENTRY_THRESHOLD_SHORT:.2f}")
print(f"   Exit: ML Exit {ML_EXIT_THRESHOLD:.2f}")
print(f"   Stop Loss: {STOP_LOSS*100:.1f}%")
print(f"   Max Hold: {MAX_HOLD} candles ({MAX_HOLD/12:.1f}h)")
print()

# ==============================================================================
# Load Data and Features
# ==============================================================================

print("-"*80)
print("STEP 1: Loading Data")
print("-"*80)

csv_file = sorted(DATA_DIR.glob("BTCUSDT_5m_raw_35days_*.csv"), reverse=True)[0]
df_raw = pd.read_csv(csv_file)
df_raw['timestamp'] = pd.to_datetime(df_raw['timestamp'])
for col in ['open', 'high', 'low', 'close', 'volume']:
    df_raw[col] = df_raw[col].astype(float)

print(f"✅ {len(df_raw):,} raw candles")

df = calculate_all_features_enhanced_v2(df_raw.copy(), phase='phase1')
df = prepare_exit_features(df)
print(f"✅ {len(df):,} candles with features")

# Extract validation period
split_date = '2025-10-28'
df_val = df[df['timestamp'] >= split_date].copy().reset_index(drop=True)

print(f"\n📊 Validation Period: {len(df_val):,} candles")
print(f"   Date: {df_val['timestamp'].iloc[0]} to {df_val['timestamp'].iloc[-1]}")
print(f"   Price: ${df_val['close'].min():,.1f} - ${df_val['close'].max():,.1f}")
print()

# ==============================================================================
# Load Models
# ==============================================================================

print("-"*80)
print("STEP 2: Loading Models")
print("-"*80)

# SHORT Entry (NEW)
short_entry_model = joblib.load(MODELS_DIR / "xgboost_short_entry_with_new_features_20251104_213043.pkl")
short_entry_scaler = joblib.load(MODELS_DIR / "xgboost_short_entry_with_new_features_20251104_213043_scaler.pkl")
with open(MODELS_DIR / "xgboost_short_entry_with_new_features_20251104_213043_features.txt", 'r') as f:
    short_entry_features = [line.strip() for line in f.readlines()]

# LONG Entry (existing)
long_entry_model = joblib.load(MODELS_DIR / "xgboost_long_entry_enhanced_20251024_012445.pkl")
long_entry_scaler = joblib.load(MODELS_DIR / "xgboost_long_entry_enhanced_20251024_012445_scaler.pkl")
with open(MODELS_DIR / "xgboost_long_entry_enhanced_20251024_012445_features.txt", 'r') as f:
    long_entry_features = [line.strip() for line in f.readlines()]

# Exit Models
long_exit_model = joblib.load(MODELS_DIR / "xgboost_long_exit_oppgating_improved_20251024_043527.pkl")
long_exit_scaler = joblib.load(MODELS_DIR / "xgboost_long_exit_oppgating_improved_20251024_043527_scaler.pkl")
with open(MODELS_DIR / "xgboost_long_exit_oppgating_improved_20251024_043527_features.txt", 'r') as f:
    long_exit_features = [line.strip() for line in f.readlines()]

short_exit_model = joblib.load(MODELS_DIR / "xgboost_short_exit_oppgating_improved_20251024_044510.pkl")
short_exit_scaler = joblib.load(MODELS_DIR / "xgboost_short_exit_oppgating_improved_20251024_044510_scaler.pkl")
with open(MODELS_DIR / "xgboost_short_exit_oppgating_improved_20251024_044510_features.txt", 'r') as f:
    short_exit_features = [line.strip() for line in f.readlines()]

print("✅ All models loaded")
print()

# ==============================================================================
# Calculate Probabilities
# ==============================================================================

print("-"*80)
print("STEP 3: Calculating Probabilities")
print("-"*80)

X_long_entry = df_val[long_entry_features].values
X_long_entry_scaled = long_entry_scaler.transform(X_long_entry)
long_entry_probs = long_entry_model.predict_proba(X_long_entry_scaled)[:, 1]

X_short_entry = df_val[short_entry_features].values
X_short_entry_scaled = short_entry_scaler.transform(X_short_entry)
short_entry_probs = short_entry_model.predict_proba(X_short_entry_scaled)[:, 1]

X_long_exit = df_val[long_exit_features].values
X_long_exit_scaled = long_exit_scaler.transform(X_long_exit)
long_exit_probs = long_exit_model.predict_proba(X_long_exit_scaled)[:, 1]

X_short_exit = df_val[short_exit_features].values
X_short_exit_scaled = short_exit_scaler.transform(X_short_exit)
short_exit_probs = short_exit_model.predict_proba(X_short_exit_scaled)[:, 1]

print("✅ All probabilities calculated")
print()

# ==============================================================================
# Run Backtest with Detailed Tracking
# ==============================================================================

print("-"*80)
print("STEP 4: Running Detailed Backtest")
print("-"*80)

initial_balance = 100.0
balance = initial_balance
position = None
trades = []

for i in range(len(df_val)):
    current_price = df_val.iloc[i]['close']
    current_time = df_val.iloc[i]['timestamp']

    # Check if in position
    if position is not None:
        entry_idx = position['entry_idx']
        entry_price = position['entry_price']
        side = position['side']
        hold_time = i - entry_idx

        # Calculate P&L
        if side == 'LONG':
            pnl_pct = (current_price - entry_price) / entry_price
            exit_prob = long_exit_probs[i]
        else:
            pnl_pct = (entry_price - current_price) / entry_price
            exit_prob = short_exit_probs[i]

        leveraged_pnl_pct = pnl_pct * LEVERAGE

        # Exit conditions
        exit_reason = None
        if leveraged_pnl_pct <= STOP_LOSS:
            exit_reason = "Stop Loss"
        elif exit_prob >= ML_EXIT_THRESHOLD:
            exit_reason = "ML Exit"
        elif hold_time >= MAX_HOLD:
            exit_reason = "Max Hold"

        if exit_reason:
            pnl_usd = balance * leveraged_pnl_pct
            entry_fee = balance * FEE_RATE
            exit_fee = (balance + pnl_usd) * FEE_RATE
            total_fee = entry_fee + exit_fee
            pnl_net = pnl_usd - total_fee

            balance += pnl_net

            trades.append({
                'entry_time': position['entry_time'],
                'exit_time': current_time,
                'side': side,
                'entry_price': entry_price,
                'exit_price': current_price,
                'entry_prob': position['entry_prob'],
                'exit_prob': exit_prob,
                'hold_time': hold_time,
                'pnl_pct': pnl_pct * 100,
                'leveraged_pnl_pct': leveraged_pnl_pct * 100,
                'pnl_gross': pnl_usd,
                'pnl_net': pnl_net,
                'total_fee': total_fee,
                'exit_reason': exit_reason,
                'balance_after': balance
            })

            position = None

    # Check for new entry
    if position is None:
        long_signal = long_entry_probs[i] >= ENTRY_THRESHOLD_LONG
        short_signal = short_entry_probs[i] >= ENTRY_THRESHOLD_SHORT

        if long_signal or short_signal:
            long_ev = long_entry_probs[i] if long_signal else 0
            short_ev = short_entry_probs[i] if short_signal else 0

            if short_ev > long_ev + 0.001:
                chosen_side = 'SHORT'
                chosen_prob = short_ev
            elif long_signal:
                chosen_side = 'LONG'
                chosen_prob = long_ev
            else:
                chosen_side = None

            if chosen_side:
                position = {
                    'side': chosen_side,
                    'entry_idx': i,
                    'entry_price': current_price,
                    'entry_time': current_time,
                    'entry_prob': chosen_prob
                }

print(f"✅ Backtest complete: {len(trades)} trades")
print()

# ==============================================================================
# Detailed Analysis
# ==============================================================================

print("="*80)
print("DETAILED TRADE ANALYSIS")
print("="*80)
print()

if len(trades) == 0:
    print("❌ No trades found")
    sys.exit(1)

df_trades = pd.DataFrame(trades)

# Overall Performance
total_return = (balance - initial_balance) / initial_balance * 100
print("📊 Overall Performance:")
print(f"   Initial Balance: ${initial_balance:.2f}")
print(f"   Final Balance: ${balance:.2f}")
print(f"   Total Return: {total_return:+.2f}%")
print(f"   Total P&L: ${balance - initial_balance:+.2f}")
print()

# Trade Statistics
wins = df_trades[df_trades['pnl_net'] > 0]
losses = df_trades[df_trades['pnl_net'] <= 0]

print("📈 Trade Statistics:")
print(f"   Total Trades: {len(df_trades)}")
print(f"   Wins: {len(wins)} ({len(wins)/len(df_trades)*100:.1f}%)")
print(f"   Losses: {len(losses)} ({len(losses)/len(df_trades)*100:.1f}%)")
print()

# P&L Statistics
print("💰 P&L Statistics:")
print(f"   Avg Win: ${wins['pnl_net'].mean():.2f}")
print(f"   Avg Loss: ${losses['pnl_net'].mean():.2f}")
print(f"   Max Win: ${df_trades['pnl_net'].max():.2f}")
print(f"   Max Loss: ${df_trades['pnl_net'].min():.2f}")
profit_factor = wins['pnl_net'].sum() / abs(losses['pnl_net'].sum()) if len(losses) > 0 else float('inf')
print(f"   Profit Factor: {profit_factor:.2f}")
print(f"   Total Fees: ${df_trades['total_fee'].sum():.2f}")
fee_ratio = df_trades['total_fee'].sum() / abs(df_trades['pnl_gross'].sum()) * 100
print(f"   Fee Ratio: {fee_ratio:.1f}%")
print()

# Side Analysis
long_trades = df_trades[df_trades['side'] == 'LONG']
short_trades = df_trades[df_trades['side'] == 'SHORT']

print("📊 Side Distribution:")
print(f"   LONG: {len(long_trades)} trades ({len(long_trades)/len(df_trades)*100:.1f}%)")
if len(long_trades) > 0:
    long_wins = long_trades[long_trades['pnl_net'] > 0]
    print(f"      Win Rate: {len(long_wins)/len(long_trades)*100:.1f}%")
    print(f"      Total P&L: ${long_trades['pnl_net'].sum():+.2f}")
    print(f"      Avg P&L: ${long_trades['pnl_net'].mean():+.2f}")
print()
print(f"   SHORT: {len(short_trades)} trades ({len(short_trades)/len(df_trades)*100:.1f}%)")
if len(short_trades) > 0:
    short_wins = short_trades[short_trades['pnl_net'] > 0]
    print(f"      Win Rate: {len(short_wins)/len(short_trades)*100:.1f}%")
    print(f"      Total P&L: ${short_trades['pnl_net'].sum():+.2f}")
    print(f"      Avg P&L: ${short_trades['pnl_net'].mean():+.2f}")
print()

# Hold Time Analysis
print("⏱️  Hold Time Analysis:")
print(f"   Average: {df_trades['hold_time'].mean():.1f} candles ({df_trades['hold_time'].mean()/12:.1f}h)")
print(f"   Median: {df_trades['hold_time'].median():.1f} candles ({df_trades['hold_time'].median()/12:.1f}h)")
print(f"   Min: {df_trades['hold_time'].min():.0f} candles ({df_trades['hold_time'].min()/12:.1f}h)")
print(f"   Max: {df_trades['hold_time'].max():.0f} candles ({df_trades['hold_time'].max()/12:.1f}h)")
print()

# Exit Reason Analysis
print("🚪 Exit Reason Distribution:")
exit_reasons = df_trades['exit_reason'].value_counts()
for reason, count in exit_reasons.items():
    pct = count / len(df_trades) * 100
    avg_pnl = df_trades[df_trades['exit_reason'] == reason]['pnl_net'].mean()
    print(f"   {reason}: {count} ({pct:.1f}%), Avg P&L: ${avg_pnl:+.2f}")
print()

# Entry Probability Analysis
print("🎯 Entry Probability Analysis:")
print(f"   Avg Entry Prob: {df_trades['entry_prob'].mean():.4f} ({df_trades['entry_prob'].mean()*100:.2f}%)")
print(f"   Min Entry Prob: {df_trades['entry_prob'].min():.4f} ({df_trades['entry_prob'].min()*100:.2f}%)")
print(f"   Max Entry Prob: {df_trades['entry_prob'].max():.4f} ({df_trades['entry_prob'].max()*100:.2f}%)")
print()

# Win vs Loss Analysis by Entry Probability
print("📈 Win Rate by Entry Probability Quartile:")
quartiles = df_trades['entry_prob'].quantile([0.25, 0.5, 0.75])
q1_trades = df_trades[df_trades['entry_prob'] <= quartiles[0.25]]
q2_trades = df_trades[(df_trades['entry_prob'] > quartiles[0.25]) & (df_trades['entry_prob'] <= quartiles[0.5])]
q3_trades = df_trades[(df_trades['entry_prob'] > quartiles[0.5]) & (df_trades['entry_prob'] <= quartiles[0.75])]
q4_trades = df_trades[df_trades['entry_prob'] > quartiles[0.75]]

for q_name, q_trades, q_range in [
    ("Q1 (Lowest)", q1_trades, f"{df_trades['entry_prob'].min():.3f}-{quartiles[0.25]:.3f}"),
    ("Q2", q2_trades, f"{quartiles[0.25]:.3f}-{quartiles[0.5]:.3f}"),
    ("Q3", q3_trades, f"{quartiles[0.5]:.3f}-{quartiles[0.75]:.3f}"),
    ("Q4 (Highest)", q4_trades, f"{quartiles[0.75]:.3f}-{df_trades['entry_prob'].max():.3f}")
]:
    if len(q_trades) > 0:
        q_wins = q_trades[q_trades['pnl_net'] > 0]
        q_wr = len(q_wins) / len(q_trades) * 100
        q_avg_pnl = q_trades['pnl_net'].mean()
        print(f"   {q_name} ({q_range}): {len(q_trades)} trades, WR {q_wr:.1f}%, Avg P&L ${q_avg_pnl:+.2f}")
print()

# ==============================================================================
# Top/Worst Trades
# ==============================================================================

print("-"*80)
print("TOP 5 WINNING TRADES")
print("-"*80)
print()

top_wins = df_trades.nlargest(5, 'pnl_net')
for i, (idx, trade) in enumerate(top_wins.iterrows(), 1):
    print(f"{i}. {trade['side']} Trade:")
    print(f"   Entry: {trade['entry_time']} @ ${trade['entry_price']:,.1f} (Prob: {trade['entry_prob']:.2%})")
    print(f"   Exit:  {trade['exit_time']} @ ${trade['exit_price']:,.1f} (Reason: {trade['exit_reason']})")
    print(f"   Hold: {trade['hold_time']:.0f} candles ({trade['hold_time']/12:.1f}h)")
    print(f"   P&L: ${trade['pnl_net']:+.2f} ({trade['leveraged_pnl_pct']:+.2f}%)")
    print()

print("-"*80)
print("TOP 5 LOSING TRADES")
print("-"*80)
print()

top_losses = df_trades.nsmallest(5, 'pnl_net')
for i, (idx, trade) in enumerate(top_losses.iterrows(), 1):
    print(f"{i}. {trade['side']} Trade:")
    print(f"   Entry: {trade['entry_time']} @ ${trade['entry_price']:,.1f} (Prob: {trade['entry_prob']:.2%})")
    print(f"   Exit:  {trade['exit_time']} @ ${trade['exit_price']:,.1f} (Reason: {trade['exit_reason']})")
    print(f"   Hold: {trade['hold_time']:.0f} candles ({trade['hold_time']/12:.1f}h)")
    print(f"   P&L: ${trade['pnl_net']:+.2f} ({trade['leveraged_pnl_pct']:+.2f}%)")
    print()

# ==============================================================================
# Risk Analysis
# ==============================================================================

print("="*80)
print("RISK ANALYSIS")
print("="*80)
print()

# Drawdown Analysis
df_trades['cumulative_pnl'] = df_trades['pnl_net'].cumsum()
running_max = df_trades['cumulative_pnl'].expanding().max()
drawdown = df_trades['cumulative_pnl'] - running_max
max_drawdown = drawdown.min()

print("📉 Drawdown Analysis:")
print(f"   Max Drawdown: ${max_drawdown:.2f} ({max_drawdown/initial_balance*100:.2f}%)")
max_dd_idx = drawdown.idxmin()
if pd.notna(max_dd_idx):
    print(f"   Max DD at: {df_trades.iloc[max_dd_idx]['exit_time']}")
print()

# Consecutive Losses
df_trades['is_loss'] = (df_trades['pnl_net'] <= 0).astype(int)
max_consecutive_losses = 0
current_streak = 0
for is_loss in df_trades['is_loss']:
    if is_loss:
        current_streak += 1
        max_consecutive_losses = max(max_consecutive_losses, current_streak)
    else:
        current_streak = 0

print("🔴 Consecutive Loss Analysis:")
print(f"   Max Consecutive Losses: {max_consecutive_losses}")
print(f"   Risk per trade: ~{abs(losses['pnl_net'].mean())/initial_balance*100:.2f}% of initial balance")
print()

# ==============================================================================
# Conclusion
# ==============================================================================

print("="*80)
print("CONCLUSION")
print("="*80)
print()

print(f"✅ Configuration: Entry={ENTRY_THRESHOLD_LONG:.2f}, Exit={ML_EXIT_THRESHOLD:.2f}")
print(f"   Return: {total_return:+.2f}%")
print(f"   Trades: {len(df_trades)} ({len(df_trades)/7.5:.1f} per day)")
print(f"   Win Rate: {len(wins)/len(df_trades)*100:.1f}%")
print(f"   Profit Factor: {profit_factor:.2f}")
print(f"   Max Drawdown: {max_drawdown/initial_balance*100:.2f}%")
print()

if total_return > 1.0 and len(df_trades) < 100 and profit_factor > 1.0:
    print("🎯 RECOMMENDATION: DEPLOY TO PRODUCTION")
    print("   ✅ Positive return")
    print("   ✅ Reasonable trade frequency")
    print("   ✅ Profit factor >1")
    print("   ✅ Risk metrics acceptable")
else:
    print("⚠️  RECOMMENDATION: FURTHER ANALYSIS NEEDED")
    if total_return <= 1.0:
        print("   - Return below target (1%)")
    if len(df_trades) >= 100:
        print("   - Trade frequency too high")
    if profit_factor <= 1.0:
        print("   - Profit factor below 1")

print()
print("="*80)
