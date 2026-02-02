"""
Full 28-Day Backtest - Current Production Settings (CSV Data)
Created: 2025-11-04 01:42 KST
Purpose: Validate current production performance on full 28 days using downloaded CSV data

Configuration:
- Entry: 0.80/0.80 (LONG/SHORT)
- Exit: 0.70/0.70 (ML Exit)
- Models: Enhanced 5-Fold CV (20251024_012445)
- Stop Loss: -3% balance
- Max Hold: 120 candles (10 hours)
- Leverage: 4x

Data Source: CSV (8,064 candles, Oct 6 - Nov 3)
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import sys
from datetime import datetime
import yaml

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.production.production_features_v1 import calculate_all_features_enhanced_v2
from scripts.production.production_exit_features_v1 import prepare_exit_features

# Current production configuration (as of 2025-11-04 01:42 KST)
LONG_THRESHOLD = 0.80  # Current production setting
SHORT_THRESHOLD = 0.80  # Current production setting
ML_EXIT_THRESHOLD_LONG = 0.70  # Current production setting
ML_EXIT_THRESHOLD_SHORT = 0.70  # Current production setting
EMERGENCY_STOP_LOSS = 0.03  # -3% balance
EMERGENCY_MAX_HOLD_TIME = 120  # 10 hours (120 candles)
LEVERAGE = 4
INITIAL_CAPITAL = 10000
SYMBOL = "BTC-USDT"

print("="*80)
print("FULL 28-DAY BACKTEST - CURRENT PRODUCTION SETTINGS (CSV DATA)")
print("="*80)
print(f"⏰ Backtest Time: 2025-11-04 01:42 KST")
print(f"📊 Period: Full 28 days (8,064 candles)")
print(f"🎯 Entry: {LONG_THRESHOLD}/{SHORT_THRESHOLD} (LONG/SHORT)")
print(f"🚪 Exit: {ML_EXIT_THRESHOLD_LONG}/{ML_EXIT_THRESHOLD_SHORT} (ML Exit)")
print(f"🛡️ Stop Loss: {EMERGENCY_STOP_LOSS*100}% balance")
print(f"⏱️ Max Hold: {EMERGENCY_MAX_HOLD_TIME} candles")
print(f"⚡ Leverage: {LEVERAGE}x")

# Load CSV data
csv_file = PROJECT_ROOT / "data" / "features" / "BTCUSDT_5m_raw_latest4weeks_20251104_014102.csv"
print(f"\n📥 Loading CSV data from: {csv_file.name}")
df = pd.read_csv(csv_file)

# Convert columns
df['timestamp'] = pd.to_datetime(df['timestamp'])
for col in ['open', 'high', 'low', 'close', 'volume']:
    df[col] = df[col].astype(float)

print(f"✅ Data loaded: {len(df)} candles")
print(f"   Period: {df.iloc[0]['timestamp']} to {df.iloc[-1]['timestamp']}")
print(f"   Price range: ${df['close'].min():,.1f} - ${df['close'].max():,.1f}")

# Calculate features
print("\n🔧 Calculating features...")
df_features = calculate_all_features_enhanced_v2(df.copy(), phase='phase1')
df_features = prepare_exit_features(df_features)
print(f"✅ Features calculated: {len(df_features)} rows")
print(f"   Period: {df_features.iloc[0]['timestamp']} to {df_features.iloc[-1]['timestamp']}")
print(f"   Duration: {len(df_features)/12:.1f} hours ({len(df_features)/288:.1f} days)")

# Load models (Enhanced 5-Fold CV - Current production)
MODELS_DIR = PROJECT_ROOT / "models"

print("\n📦 Loading production models...")
long_entry_model = joblib.load(MODELS_DIR / "xgboost_long_entry_enhanced_20251024_012445.pkl")
long_entry_scaler = joblib.load(MODELS_DIR / "xgboost_long_entry_enhanced_20251024_012445_scaler.pkl")
with open(MODELS_DIR / "xgboost_long_entry_enhanced_20251024_012445_features.txt", 'r') as f:
    long_entry_features = [line.strip() for line in f.readlines()]

short_entry_model = joblib.load(MODELS_DIR / "xgboost_short_entry_enhanced_20251024_012445.pkl")
short_entry_scaler = joblib.load(MODELS_DIR / "xgboost_short_entry_enhanced_20251024_012445_scaler.pkl")
with open(MODELS_DIR / "xgboost_short_entry_enhanced_20251024_012445_features.txt", 'r') as f:
    short_entry_features = [line.strip() for line in f.readlines()]

long_exit_model = joblib.load(MODELS_DIR / "xgboost_long_exit_oppgating_improved_20251024_043527.pkl")
long_exit_scaler = joblib.load(MODELS_DIR / "xgboost_long_exit_oppgating_improved_20251024_043527_scaler.pkl")
with open(MODELS_DIR / "xgboost_long_exit_oppgating_improved_20251024_043527_features.txt", 'r') as f:
    long_exit_features = [line.strip() for line in f.readlines()]

short_exit_model = joblib.load(MODELS_DIR / "xgboost_short_exit_oppgating_improved_20251024_044510.pkl")
short_exit_scaler = joblib.load(MODELS_DIR / "xgboost_short_exit_oppgating_improved_20251024_044510_scaler.pkl")
with open(MODELS_DIR / "xgboost_short_exit_oppgating_improved_20251024_044510_features.txt", 'r') as f:
    short_exit_features = [line.strip() for line in f.readlines()]

print(f"✅ Models loaded:")
print(f"   LONG Entry: {len(long_entry_features)} features")
print(f"   SHORT Entry: {len(short_entry_features)} features")
print(f"   LONG Exit: {len(long_exit_features)} features")
print(f"   SHORT Exit: {len(short_exit_features)} features")

# Backtest simulation
print("\n🚀 Running backtest simulation on full 28-day dataset...")
print(f"   Total rows to simulate: {len(df_features):,}")

trades = []
balance = INITIAL_CAPITAL
position = None
equity_curve = []

for i in range(len(df_features)):
    row = df_features.iloc[i]
    current_price = row['close']
    timestamp = row['timestamp']
    equity = balance  # Default: equity = balance (no position)

    # Position management
    if position is not None:
        # Update unrealized P&L
        if position['side'] == 'LONG':
            price_change = (current_price - position['entry_price']) / position['entry_price']
        else:  # SHORT
            price_change = (position['entry_price'] - current_price) / position['entry_price']

        leveraged_pnl_pct = price_change * LEVERAGE
        unrealized_pnl = position['position_value'] * leveraged_pnl_pct
        equity = balance + unrealized_pnl

        # Emergency stop loss (balance-based)
        if leveraged_pnl_pct <= -EMERGENCY_STOP_LOSS:
            # Close position - Stop Loss
            fee_pct = 0.0005
            exit_fee = position['position_value'] * fee_pct
            pnl_net = unrealized_pnl - exit_fee
            balance += pnl_net

            trades.append({
                'entry_time': position['entry_time'],
                'exit_time': timestamp,
                'side': position['side'],
                'entry_price': position['entry_price'],
                'exit_price': current_price,
                'position_value': position['position_value'],
                'pnl_pct': leveraged_pnl_pct,
                'pnl_net': pnl_net,
                'exit_reason': f"Stop Loss ({leveraged_pnl_pct*100:.2f}%)",
                'hold_candles': position['hold_candles']
            })
            position = None
            equity = balance

        # Emergency max hold time
        elif position['hold_candles'] >= EMERGENCY_MAX_HOLD_TIME:
            # Close position - Max Hold
            fee_pct = 0.0005
            exit_fee = position['position_value'] * fee_pct
            pnl_net = unrealized_pnl - exit_fee
            balance += pnl_net

            trades.append({
                'entry_time': position['entry_time'],
                'exit_time': timestamp,
                'side': position['side'],
                'entry_price': position['entry_price'],
                'exit_price': current_price,
                'position_value': position['position_value'],
                'pnl_pct': leveraged_pnl_pct,
                'pnl_net': pnl_net,
                'exit_reason': f"Max Hold ({position['hold_candles']} candles)",
                'hold_candles': position['hold_candles']
            })
            position = None
            equity = balance

        # ML Exit signal
        else:
            exit_features_dict = row[long_exit_features if position['side'] == 'LONG' else short_exit_features].to_dict()
            exit_features_list = [exit_features_dict[f] for f in (long_exit_features if position['side'] == 'LONG' else short_exit_features)]
            exit_scaler = long_exit_scaler if position['side'] == 'LONG' else short_exit_scaler
            exit_model = long_exit_model if position['side'] == 'LONG' else short_exit_model
            exit_threshold = ML_EXIT_THRESHOLD_LONG if position['side'] == 'LONG' else ML_EXIT_THRESHOLD_SHORT

            exit_features_scaled = exit_scaler.transform([exit_features_list])
            exit_prob = exit_model.predict_proba(exit_features_scaled)[0][1]

            if exit_prob >= exit_threshold:
                # Close position - ML Exit
                fee_pct = 0.0005
                exit_fee = position['position_value'] * fee_pct
                pnl_net = unrealized_pnl - exit_fee
                balance += pnl_net

                trades.append({
                    'entry_time': position['entry_time'],
                    'exit_time': timestamp,
                    'side': position['side'],
                    'entry_price': position['entry_price'],
                    'exit_price': current_price,
                    'position_value': position['position_value'],
                    'pnl_pct': leveraged_pnl_pct,
                    'pnl_net': pnl_net,
                    'exit_reason': f"ML Exit ({exit_prob:.3f})",
                    'hold_candles': position['hold_candles']
                })
                position = None
                equity = balance
            else:
                position['hold_candles'] += 1

    # Entry signals (only if no position)
    if position is None:
        # Get entry features
        long_features_dict = row[long_entry_features].to_dict()
        short_features_dict = row[short_entry_features].to_dict()

        long_features_list = [long_features_dict[f] for f in long_entry_features]
        short_features_list = [short_features_dict[f] for f in short_entry_features]

        # Scale and predict
        long_features_scaled = long_entry_scaler.transform([long_features_list])
        short_features_scaled = short_entry_scaler.transform([short_features_list])

        long_prob = long_entry_model.predict_proba(long_features_scaled)[0][1]
        short_prob = short_entry_model.predict_proba(short_features_scaled)[0][1]

        # Opportunity gating logic
        long_ev = long_prob - (1 - long_prob)
        short_ev = short_prob - (1 - short_prob)

        # Entry decision
        if long_prob >= LONG_THRESHOLD and long_ev > short_ev + 0.001:
            # Enter LONG
            position_size_pct = min(0.95, max(0.20, long_prob))
            position_value = balance * position_size_pct
            fee_pct = 0.0005
            entry_fee = position_value * fee_pct
            balance -= entry_fee

            position = {
                'side': 'LONG',
                'entry_time': timestamp,
                'entry_price': current_price,
                'position_value': position_value,
                'hold_candles': 0
            }
            equity = balance

        elif short_prob >= SHORT_THRESHOLD and short_ev > long_ev + 0.001:
            # Enter SHORT
            position_size_pct = min(0.95, max(0.20, short_prob))
            position_value = balance * position_size_pct
            fee_pct = 0.0005
            entry_fee = position_value * fee_pct
            balance -= entry_fee

            position = {
                'side': 'SHORT',
                'entry_time': timestamp,
                'entry_price': current_price,
                'position_value': position_value,
                'hold_candles': 0
            }
            equity = balance

    equity_curve.append(equity)

    # Progress indicator
    if (i + 1) % 1000 == 0:
        print(f"   Progress: {i+1:,}/{len(df_features):,} ({(i+1)/len(df_features)*100:.1f}%)")

# Close remaining position (if any)
if position is not None:
    current_price = df_features.iloc[-1]['close']
    timestamp = df_features.iloc[-1]['timestamp']

    if position['side'] == 'LONG':
        price_change = (current_price - position['entry_price']) / position['entry_price']
    else:
        price_change = (position['entry_price'] - current_price) / position['entry_price']

    leveraged_pnl_pct = price_change * LEVERAGE
    unrealized_pnl = position['position_value'] * leveraged_pnl_pct
    fee_pct = 0.0005
    exit_fee = position['position_value'] * fee_pct
    pnl_net = unrealized_pnl - exit_fee
    balance += pnl_net

    trades.append({
        'entry_time': position['entry_time'],
        'exit_time': timestamp,
        'side': position['side'],
        'entry_price': position['entry_price'],
        'exit_price': current_price,
        'position_value': position['position_value'],
        'pnl_pct': leveraged_pnl_pct,
        'pnl_net': pnl_net,
        'exit_reason': 'End of Period',
        'hold_candles': position['hold_candles']
    })
    equity_curve[-1] = balance

# Results analysis
print("\n" + "="*80)
print("📊 FULL 28-DAY BACKTEST RESULTS")
print("="*80)

final_balance = balance
total_return = (final_balance - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100

print(f"\n💰 Performance:")
print(f"   Initial Capital: ${INITIAL_CAPITAL:,.2f}")
print(f"   Final Balance: ${final_balance:,.2f}")
print(f"   Total Return: {total_return:+.2f}%")
print(f"   Total P&L: ${final_balance - INITIAL_CAPITAL:+,.2f}")

if len(trades) > 0:
    df_trades = pd.DataFrame(trades)

    # Trade statistics
    total_trades = len(df_trades)
    long_trades = len(df_trades[df_trades['side'] == 'LONG'])
    short_trades = len(df_trades[df_trades['side'] == 'SHORT'])

    winning_trades = df_trades[df_trades['pnl_net'] > 0]
    losing_trades = df_trades[df_trades['pnl_net'] <= 0]

    win_rate = len(winning_trades) / total_trades * 100 if total_trades > 0 else 0

    avg_win = winning_trades['pnl_net'].mean() if len(winning_trades) > 0 else 0
    avg_loss = losing_trades['pnl_net'].mean() if len(losing_trades) > 0 else 0

    profit_factor = abs(winning_trades['pnl_net'].sum() / losing_trades['pnl_net'].sum()) if len(losing_trades) > 0 and losing_trades['pnl_net'].sum() != 0 else float('inf')

    avg_hold = df_trades['hold_candles'].mean()

    print(f"\n📈 Trade Statistics:")
    print(f"   Total Trades: {total_trades}")
    print(f"   LONG Trades: {long_trades} ({long_trades/total_trades*100:.1f}%)")
    print(f"   SHORT Trades: {short_trades} ({short_trades/total_trades*100:.1f}%)")
    print(f"   Winning Trades: {len(winning_trades)} ({win_rate:.1f}%)")
    print(f"   Losing Trades: {len(losing_trades)}")
    print(f"   Average Win: ${avg_win:,.2f}")
    print(f"   Average Loss: ${avg_loss:,.2f}")
    print(f"   Profit Factor: {profit_factor:.2f}")
    print(f"   Average Hold: {avg_hold:.1f} candles ({avg_hold/12:.1f} hours)")

    # Exit reason breakdown
    print(f"\n🚪 Exit Reasons:")
    exit_reasons = df_trades['exit_reason'].str.extract(r'^([^(]+)')[0].value_counts()
    for reason, count in exit_reasons.items():
        print(f"   {reason.strip()}: {count} ({count/total_trades*100:.1f}%)")

    # Save results
    results_dir = PROJECT_ROOT / "results"
    results_dir.mkdir(exist_ok=True)
    output_file = results_dir / "backtest_28days_full_20251104_0142.csv"
    df_trades.to_csv(output_file, index=False)
    print(f"\n💾 Results saved: {output_file}")

    # Save summary
    summary_file = results_dir / "backtest_28days_full_20251104_0142_summary.txt"
    with open(summary_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("FULL 28-DAY BACKTEST - CURRENT PRODUCTION SETTINGS\n")
        f.write("="*80 + "\n\n")
        f.write(f"Backtest Time: 2025-11-04 01:42 KST\n")
        f.write(f"Data Source: CSV (8,064 candles)\n")
        f.write(f"Period: {df_features.iloc[0]['timestamp']} to {df_features.iloc[-1]['timestamp']}\n")
        f.write(f"Duration: {len(df_features)/12:.1f} hours ({len(df_features)/288:.1f} days)\n\n")
        f.write(f"Configuration:\n")
        f.write(f"  Entry: {LONG_THRESHOLD}/{SHORT_THRESHOLD} (LONG/SHORT)\n")
        f.write(f"  Exit: {ML_EXIT_THRESHOLD_LONG}/{ML_EXIT_THRESHOLD_SHORT} (ML Exit)\n")
        f.write(f"  Stop Loss: {EMERGENCY_STOP_LOSS*100}%\n")
        f.write(f"  Max Hold: {EMERGENCY_MAX_HOLD_TIME} candles\n")
        f.write(f"  Leverage: {LEVERAGE}x\n\n")
        f.write(f"Performance:\n")
        f.write(f"  Initial Capital: ${INITIAL_CAPITAL:,.2f}\n")
        f.write(f"  Final Balance: ${final_balance:,.2f}\n")
        f.write(f"  Total Return: {total_return:+.2f}%\n")
        f.write(f"  Total P&L: ${final_balance - INITIAL_CAPITAL:+,.2f}\n\n")
        f.write(f"Trade Statistics:\n")
        f.write(f"  Total Trades: {total_trades}\n")
        f.write(f"  LONG Trades: {long_trades} ({long_trades/total_trades*100:.1f}%)\n")
        f.write(f"  SHORT Trades: {short_trades} ({short_trades/total_trades*100:.1f}%)\n")
        f.write(f"  Win Rate: {win_rate:.1f}%\n")
        f.write(f"  Average Win: ${avg_win:,.2f}\n")
        f.write(f"  Average Loss: ${avg_loss:,.2f}\n")
        f.write(f"  Profit Factor: {profit_factor:.2f}\n")
        f.write(f"  Average Hold: {avg_hold:.1f} candles ({avg_hold/12:.1f} hours)\n\n")
        f.write(f"Exit Reasons:\n")
        for reason, count in exit_reasons.items():
            f.write(f"  {reason.strip()}: {count} ({count/total_trades*100:.1f}%)\n")
    print(f"💾 Summary saved: {summary_file}")

    # Print all trades
    print(f"\n📋 All Trades ({len(df_trades)} total):")
    if len(df_trades) <= 20:
        print(df_trades.to_string(index=False))
    else:
        print("First 10 trades:")
        print(df_trades.head(10).to_string(index=False))
        print("\n...")
        print(f"\nLast 10 trades:")
        print(df_trades.tail(10).to_string(index=False))
else:
    print("\n⚠️  No trades executed during backtest period")
    print("   Possible reasons:")
    print("   - Entry thresholds too high (0.80/0.80)")
    print("   - Market conditions not favorable")
    print("   - All signals below threshold")

print("\n" + "="*80)
print("✅ Full 28-Day Backtest Complete!")
print("="*80)
