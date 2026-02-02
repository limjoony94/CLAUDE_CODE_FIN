#!/usr/bin/env python3
"""
5-Day Backtest with Current Production Models
Purpose: Validate production performance on recent 1440 candles (5 days)
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from datetime import datetime
import pandas as pd
import numpy as np
import joblib
import yaml
from pathlib import Path
from src.api.bingx_client import BingXClient
from scripts.experiments.calculate_all_features_enhanced_v2 import calculate_all_features_enhanced_v2
from scripts.production.production_exit_features_v1 import prepare_exit_features

# Load API keys
CONFIG_DIR = Path(__file__).resolve().parent.parent.parent / "config"

def load_api_keys():
    api_keys_file = CONFIG_DIR / "api_keys.yaml"
    if api_keys_file.exists():
        with open(api_keys_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            return config.get('bingx', {}).get('mainnet', {})
    return {}

_api_config = load_api_keys()
API_KEY = _api_config.get('api_key', "")
API_SECRET = _api_config.get('secret_key', "")

# Production Models (Enhanced 5-Fold CV - 20251024_012445)
LONG_ENTRY_MODEL = "models/xgboost_long_entry_enhanced_20251024_012445.pkl"
SHORT_ENTRY_MODEL = "models/xgboost_short_entry_enhanced_20251024_012445.pkl"
LONG_EXIT_MODEL = "models/xgboost_long_exit_oppgating_improved_20251024_043527.pkl"
SHORT_EXIT_MODEL = "models/xgboost_short_exit_oppgating_improved_20251024_044510.pkl"

# Production Configuration (CORRECTED to match actual production)
LONG_THRESHOLD = 0.80  # Actual production setting
SHORT_THRESHOLD = 0.80  # Actual production setting
ML_EXIT_THRESHOLD_LONG = 0.70  # Actual production setting
ML_EXIT_THRESHOLD_SHORT = 0.70  # Actual production setting
EMERGENCY_STOP_LOSS = 0.03  # -3% of balance
EMERGENCY_MAX_HOLD_TIME = 120  # 10 hours (120 candles)
LEVERAGE = 4
INITIAL_CAPITAL = 10000
OPPORTUNITY_GATE_THRESHOLD = 0.001

print("=" * 80)
print("5-DAY BACKTEST - PRODUCTION MODELS (1440 CANDLES)")
print("=" * 80)
print(f"Models: Enhanced 5-Fold CV (20251024_012445)")
print(f"Entry Thresholds: LONG {LONG_THRESHOLD}, SHORT {SHORT_THRESHOLD}")
print(f"Exit Thresholds: LONG {ML_EXIT_THRESHOLD_LONG}, SHORT {ML_EXIT_THRESHOLD_SHORT}")
print(f"Stop Loss: -{EMERGENCY_STOP_LOSS*100}% balance")
print(f"Max Hold: {EMERGENCY_MAX_HOLD_TIME} candles")
print(f"Leverage: {LEVERAGE}x")
print("=" * 80)
print()

# Fetch data
print("📡 Fetching data from BingX API...")
client = BingXClient(api_key=API_KEY, secret_key=API_SECRET, testnet=False)
klines = client.get_klines(symbol="BTC-USDT", interval="5m", limit=1440)

df = pd.DataFrame(klines)
df['timestamp'] = pd.to_datetime(df['time'], unit='ms', utc=True).dt.tz_localize(None)
for col in ['open', 'high', 'low', 'close', 'volume']:
    df[col] = df[col].astype(float)

print(f"✅ Fetched {len(df)} candles")
print(f"   From: {df.iloc[0]['timestamp']}")
print(f"   To:   {df.iloc[-1]['timestamp']}")
print()

# Calculate features
print("🔧 Calculating features...")
df_features = calculate_all_features_enhanced_v2(df.copy(), phase='phase1')
print(f"✅ Features calculated: {len(df_features)} rows (lost {len(df) - len(df_features)} due to lookback)")

# Add enhanced exit features (volume_surge, price_acceleration, etc.)
print("🔧 Adding enhanced exit features...")
df_features = prepare_exit_features(df_features)
print(f"✅ Enhanced exit features added")
print()

# Load models
print("🤖 Loading models...")
long_entry_model = joblib.load(LONG_ENTRY_MODEL)
short_entry_model = joblib.load(SHORT_ENTRY_MODEL)
long_exit_model = joblib.load(LONG_EXIT_MODEL)
short_exit_model = joblib.load(SHORT_EXIT_MODEL)

long_entry_scaler = joblib.load(LONG_ENTRY_MODEL.replace('.pkl', '_scaler.pkl'))
short_entry_scaler = joblib.load(SHORT_ENTRY_MODEL.replace('.pkl', '_scaler.pkl'))
long_exit_scaler = joblib.load(LONG_EXIT_MODEL.replace('.pkl', '_scaler.pkl'))
short_exit_scaler = joblib.load(SHORT_EXIT_MODEL.replace('.pkl', '_scaler.pkl'))

with open(LONG_ENTRY_MODEL.replace('.pkl', '_features.txt'), 'r') as f:
    long_entry_features = [line.strip() for line in f.readlines() if line.strip()]
with open(SHORT_ENTRY_MODEL.replace('.pkl', '_features.txt'), 'r') as f:
    short_entry_features = [line.strip() for line in f.readlines() if line.strip()]
with open(LONG_EXIT_MODEL.replace('.pkl', '_features.txt'), 'r') as f:
    long_exit_features = [line.strip() for line in f.readlines() if line.strip()]
with open(SHORT_EXIT_MODEL.replace('.pkl', '_features.txt'), 'r') as f:
    short_exit_features = [line.strip() for line in f.readlines() if line.strip()]

print(f"✅ Models loaded")
print(f"   LONG Entry: {len(long_entry_features)} features")
print(f"   SHORT Entry: {len(short_entry_features)} features")
print(f"   LONG Exit: {len(long_exit_features)} features")
print(f"   SHORT Exit: {len(short_exit_features)} features")
print()

# Verify exit features are in df_features
print("🔍 Checking exit features availability...")
missing_long_exit = [f for f in long_exit_features if f not in df_features.columns]
missing_short_exit = [f for f in short_exit_features if f not in df_features.columns]

if missing_long_exit:
    print(f"⚠️ LONG Exit missing {len(missing_long_exit)} features: {missing_long_exit}")
if missing_short_exit:
    print(f"⚠️ SHORT Exit missing {len(missing_short_exit)} features: {missing_short_exit}")

if not missing_long_exit and not missing_short_exit:
    print(f"✅ All exit features present in df_features")
print()

# Backtest
print("🔄 Running backtest...")
balance = INITIAL_CAPITAL
position = None
trades = []

for idx in range(len(df_features)):
    row = df_features.iloc[idx]
    current_price = row['close']
    current_time = row['timestamp']

    # Check for position exit first
    if position is not None:
        hold_time = idx - position['entry_idx']

        # Calculate unrealized P&L
        if position['side'] == 'LONG':
            price_change_pct = (current_price - position['entry_price']) / position['entry_price']
        else:  # SHORT
            price_change_pct = (position['entry_price'] - current_price) / position['entry_price']

        leveraged_pnl_pct = price_change_pct * LEVERAGE
        unrealized_pnl_usd = position['position_value'] * leveraged_pnl_pct

        # Track peak
        if 'peak_pnl' not in position:
            position['peak_pnl'] = unrealized_pnl_usd
        else:
            position['peak_pnl'] = max(position['peak_pnl'], unrealized_pnl_usd)

        drawdown_from_peak = (position['peak_pnl'] - unrealized_pnl_usd) / position['position_value']

        # Calculate balance-based stop loss
        position_size_pct = position['position_value'] / position['entry_balance']
        balance_sl_pct = EMERGENCY_STOP_LOSS / position_size_pct

        # Emergency Stop Loss check
        stop_loss_triggered = False
        if position['side'] == 'LONG':
            stop_price = position['entry_price'] * (1 - balance_sl_pct)
            if current_price <= stop_price:
                stop_loss_triggered = True
        else:  # SHORT
            stop_price = position['entry_price'] * (1 + balance_sl_pct)
            if current_price >= stop_price:
                stop_loss_triggered = True

        # Emergency Max Hold check
        max_hold_triggered = hold_time >= EMERGENCY_MAX_HOLD_TIME

        # ML Exit check (PRODUCTION STYLE - use features from df_features)
        ml_exit_triggered = False
        exit_prob = 0.0
        try:
            # Get features for current candle (PRODUCTION METHOD)
            current_features = df_features.iloc[[idx]]

            if position['side'] == 'LONG':
                # Extract and scale exit features
                exit_feat_values = current_features[long_exit_features].values
                exit_feat_scaled = long_exit_scaler.transform(exit_feat_values)
                exit_prob = long_exit_model.predict_proba(exit_feat_scaled)[0][1]

                if exit_prob >= ML_EXIT_THRESHOLD_LONG:
                    ml_exit_triggered = True

            else:  # SHORT
                # Extract and scale exit features
                exit_feat_values = current_features[short_exit_features].values
                exit_feat_scaled = short_exit_scaler.transform(exit_feat_values)
                exit_prob = short_exit_model.predict_proba(exit_feat_scaled)[0][1]

                if exit_prob >= ML_EXIT_THRESHOLD_SHORT:
                    ml_exit_triggered = True

        except Exception as e:
            print(f"⚠️ ML Exit error at idx {idx}: {e}")
            exit_prob = 0.0

        # Log exit probabilities for monitoring
        if hold_time % 20 == 0:  # Log every 20 candles
            print(f"   Hold {hold_time:3d} candles | {position['side']:5s} | P&L: {leveraged_pnl_pct*100:+.2f}% | Exit prob: {exit_prob:.3f} (threshold: {ML_EXIT_THRESHOLD_LONG if position['side']=='LONG' else ML_EXIT_THRESHOLD_SHORT})")

        # Exit decision
        if stop_loss_triggered or max_hold_triggered or ml_exit_triggered:
            # Close position
            exit_reason = 'ML Exit' if ml_exit_triggered else ('Stop Loss' if stop_loss_triggered else 'Max Hold')
            print(f"🚪 EXIT: {exit_reason} | Hold: {hold_time} candles | Exit prob: {exit_prob:.3f} | P&L: {leveraged_pnl_pct*100:+.2f}%")

            # Calculate final P&L
            net_pnl_usd = unrealized_pnl_usd
            balance += net_pnl_usd

            trades.append({
                'entry_time': position['entry_time'],
                'exit_time': current_time,
                'side': position['side'],
                'entry_price': position['entry_price'],
                'exit_price': current_price,
                'position_value': position['position_value'],
                'hold_time': hold_time,
                'pnl_pct': leveraged_pnl_pct * 100,
                'pnl_usd': net_pnl_usd,
                'exit_reason': exit_reason,
                'balance': balance
            })

            position = None

    # Check for new entry (if no position)
    if position is None:
        try:
            # LONG signal
            long_feat_df = df_features.loc[[idx], long_entry_features]
            long_feat_scaled = long_entry_scaler.transform(long_feat_df.values)
            long_prob = long_entry_model.predict_proba(long_feat_scaled)[0][1]

            # SHORT signal
            short_feat_df = df_features.loc[[idx], short_entry_features]
            short_feat_scaled = short_entry_scaler.transform(short_feat_df.values)
            short_prob = short_entry_model.predict_proba(short_feat_scaled)[0][1]
        except Exception as e:
            long_prob = 0.0
            short_prob = 0.0

        # Entry decision with Opportunity Gating
        if long_prob >= LONG_THRESHOLD:
            # Enter LONG
            position_value = balance * 0.95  # 95% of balance
            position = {
                'side': 'LONG',
                'entry_price': current_price,
                'entry_time': current_time,
                'entry_idx': idx,
                'position_value': position_value,
                'entry_balance': balance,
                'entry_prob': long_prob
            }
            print(f"📈 ENTER LONG @ ${current_price:,.1f} | Prob: {long_prob:.3f} | Position: ${position_value:,.0f}")

        elif short_prob >= SHORT_THRESHOLD:
            # Opportunity Gating check
            long_ev = long_prob * 0.0041
            short_ev = short_prob * 0.0047
            opportunity_cost = short_ev - long_ev

            if opportunity_cost > OPPORTUNITY_GATE_THRESHOLD:
                # Enter SHORT
                position_value = balance * 0.95
                position = {
                    'side': 'SHORT',
                    'entry_price': current_price,
                    'entry_time': current_time,
                    'entry_idx': idx,
                    'position_value': position_value,
                    'entry_balance': balance,
                    'entry_prob': short_prob
                }
                print(f"📉 ENTER SHORT @ ${current_price:,.1f} | Prob: {short_prob:.3f} | Position: ${position_value:,.0f}")

# Close final position if still open
if position is not None:
    final_row = df_features.iloc[-1]
    current_price = final_row['close']
    current_time = final_row['timestamp']
    hold_time = len(df_features) - 1 - position['entry_idx']

    if position['side'] == 'LONG':
        price_change_pct = (current_price - position['entry_price']) / position['entry_price']
    else:
        price_change_pct = (position['entry_price'] - current_price) / position['entry_price']

    leveraged_pnl_pct = price_change_pct * LEVERAGE
    net_pnl_usd = position['position_value'] * leveraged_pnl_pct
    balance += net_pnl_usd

    trades.append({
        'entry_time': position['entry_time'],
        'exit_time': current_time,
        'side': position['side'],
        'entry_price': position['entry_price'],
        'exit_price': current_price,
        'position_value': position['position_value'],
        'hold_time': hold_time,
        'pnl_pct': leveraged_pnl_pct * 100,
        'pnl_usd': net_pnl_usd,
        'exit_reason': 'Backtest End',
        'balance': balance
    })

# Results
print("=" * 80)
print("BACKTEST RESULTS")
print("=" * 80)
print()

if len(trades) > 0:
    df_trades = pd.DataFrame(trades)

    total_return_pct = ((balance - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100
    wins = df_trades[df_trades['pnl_usd'] > 0]
    losses = df_trades[df_trades['pnl_usd'] <= 0]
    win_rate = len(wins) / len(df_trades) * 100

    print(f"Initial Capital: ${INITIAL_CAPITAL:,.2f}")
    print(f"Final Balance: ${balance:,.2f}")
    print(f"Total Return: {total_return_pct:+.2f}%")
    print()

    print(f"Total Trades: {len(df_trades)}")
    print(f"  Wins: {len(wins)} ({len(wins)/len(df_trades)*100:.1f}%)")
    print(f"  Losses: {len(losses)} ({len(losses)/len(df_trades)*100:.1f}%)")
    print(f"Win Rate: {win_rate:.1f}%")
    print()

    print(f"Average Trade: {df_trades['pnl_pct'].mean():+.2f}% ({df_trades['pnl_usd'].mean():+.2f} USDT)")
    print(f"Average Win: {wins['pnl_pct'].mean():+.2f}% ({wins['pnl_usd'].mean():+.2f} USDT)")
    if len(losses) > 0:
        print(f"Average Loss: {losses['pnl_pct'].mean():+.2f}% ({losses['pnl_usd'].mean():+.2f} USDT)")
    print()

    print(f"Average Hold Time: {df_trades['hold_time'].mean():.1f} candles ({df_trades['hold_time'].mean()/12:.1f} hours)")
    print()

    # Exit distribution
    exit_reasons = df_trades['exit_reason'].value_counts()
    print("Exit Reasons:")
    for reason, count in exit_reasons.items():
        print(f"  {reason}: {count} ({count/len(df_trades)*100:.1f}%)")
    print()

    # Side distribution
    side_dist = df_trades['side'].value_counts()
    print("Side Distribution:")
    for side, count in side_dist.items():
        print(f"  {side}: {count} ({count/len(df_trades)*100:.1f}%)")
    print()

    # Best and worst trades
    print(f"Best Trade: {df_trades['pnl_pct'].max():+.2f}% ({df_trades['pnl_usd'].max():+.2f} USDT)")
    print(f"Worst Trade: {df_trades['pnl_pct'].min():+.2f}% ({df_trades['pnl_usd'].min():+.2f} USDT)")
    print()

    # Save results
    output_file = f"results/backtest_5days_production_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df_trades.to_csv(output_file, index=False)
    print(f"💾 Results saved to: {output_file}")
    print("=" * 80)
else:
    print("No trades executed during backtest period")
    print("=" * 80)
