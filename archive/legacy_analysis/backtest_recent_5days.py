"""
Recent 5-Day Backtest - Current Production Models
Created: 2025-11-03
Purpose: Validate current model performance on recent 5 days (1440 candles)
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

from src.api.bingx_client import BingXClient
from scripts.production.production_features_v1 import calculate_all_features_enhanced_v2
from scripts.production.production_exit_features_v1 import prepare_exit_features

# Load API keys
config_path = PROJECT_ROOT / "config" / "api_keys.yaml"
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

bingx_config = config['bingx']['mainnet']
client = BingXClient(
    api_key=bingx_config['api_key'],
    secret_key=bingx_config['secret_key']
)

# Production configuration
LONG_THRESHOLD = 0.70
SHORT_THRESHOLD = 0.70
ML_EXIT_THRESHOLD_LONG = 0.75
ML_EXIT_THRESHOLD_SHORT = 0.75
EMERGENCY_STOP_LOSS = 0.03  # -3% balance
EMERGENCY_MAX_HOLD_TIME = 120  # 10 hours
LEVERAGE = 4
INITIAL_CAPITAL = 10000
SYMBOL = "BTC-USDT"

print("="*80)
print("RECENT 5-DAY BACKTEST - CURRENT PRODUCTION MODELS")
print("="*80)

# Fetch recent 1440 candles (5 days)
print("\n📥 Fetching recent 1440 candles...")
klines = client.get_klines(SYMBOL, "5m", limit=1440)
df = pd.DataFrame(klines)

df['timestamp'] = pd.to_datetime(df['time'], unit='ms', utc=True).dt.tz_localize(None)
for col in ['open', 'high', 'low', 'close', 'volume']:
    df[col] = df[col].astype(float)

print(f"✅ Data fetched: {len(df)} candles")
print(f"   Period: {df.iloc[0]['timestamp']} to {df.iloc[-1]['timestamp']}")
print(f"   Price range: ${df['close'].min():,.1f} - ${df['close'].max():,.1f}")

# Calculate features
print("\n🔧 Calculating features...")
df_features = calculate_all_features_enhanced_v2(df.copy(), phase='phase1')
df_features = prepare_exit_features(df_features)
print(f"✅ Features calculated: {len(df_features)} rows")

# Load models
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

# Backtest simulation with PRELOADING (production-like)
print("\n🚀 Running backtest simulation...")
print("   ⚠️  Using production-style preloading: each candle uses ONLY past data")

balance = INITIAL_CAPITAL
position = None
trades = []
equity_curve = []

# Minimum candles needed for feature calculation (based on longest lookback)
MIN_PRELOAD_CANDLES = 300  # 200 for long-term + buffer

for idx in range(len(df_features)):
    candle = df_features.iloc[idx]
    timestamp = candle['timestamp']
    close_price = candle['close']

    # Get position size dynamically based on probability
    def calculate_position_size(probability):
        """Simple dynamic sizing: 20% to 95% based on probability"""
        min_size = 0.20
        max_size = 0.95
        threshold = 0.65  # Minimum threshold for sizing

        if probability < threshold:
            return min_size

        # Linear interpolation between threshold and 1.0
        normalized_prob = (probability - threshold) / (1.0 - threshold)
        size_pct = min_size + (max_size - min_size) * normalized_prob

        return min(max_size, max(min_size, size_pct))

    # Track equity
    if position is None:
        equity_curve.append({'timestamp': timestamp, 'equity': balance})
    else:
        # Calculate unrealized P&L
        if position['side'] == 'LONG':
            price_change_pct = (close_price / position['entry_price']) - 1
        else:  # SHORT
            price_change_pct = (position['entry_price'] / close_price) - 1

        leveraged_pnl_pct = price_change_pct * LEVERAGE
        unrealized_pnl = position['size_usd'] * leveraged_pnl_pct
        current_equity = balance + unrealized_pnl
        equity_curve.append({'timestamp': timestamp, 'equity': current_equity})

    # Check if we have a position
    if position is not None:
        # Exit logic
        exit_triggered = False
        exit_reason = None

        # 1. Emergency Stop Loss
        if position['side'] == 'LONG':
            price_change_pct = (close_price / position['entry_price']) - 1
        else:
            price_change_pct = (position['entry_price'] / close_price) - 1

        leveraged_pnl_pct = price_change_pct * LEVERAGE
        balance_pnl_pct = (position['size_usd'] / balance) * leveraged_pnl_pct

        if balance_pnl_pct <= -EMERGENCY_STOP_LOSS:
            exit_triggered = True
            exit_reason = 'Emergency Stop Loss'

        # 2. Emergency Max Hold
        hold_time = idx - position['entry_idx']
        if hold_time >= EMERGENCY_MAX_HOLD_TIME:
            exit_triggered = True
            exit_reason = 'Emergency Max Hold'

        # 3. ML Exit
        if not exit_triggered:
            candle_row = candle.to_frame().T

            if position['side'] == 'LONG':
                exit_feat_df = candle_row[long_exit_features]
                exit_feat_normalized = long_exit_scaler.transform(exit_feat_df.values)
                exit_prob = long_exit_model.predict_proba(exit_feat_normalized)[0][1]

                if exit_prob >= ML_EXIT_THRESHOLD_LONG:
                    exit_triggered = True
                    exit_reason = f'ML Exit ({exit_prob:.3f})'
            else:
                exit_feat_df = candle_row[short_exit_features]
                exit_feat_normalized = short_exit_scaler.transform(exit_feat_df.values)
                exit_prob = short_exit_model.predict_proba(exit_feat_normalized)[0][1]

                if exit_prob >= ML_EXIT_THRESHOLD_SHORT:
                    exit_triggered = True
                    exit_reason = f'ML Exit ({exit_prob:.3f})'

        # Exit position
        if exit_triggered:
            # Calculate final P&L
            if position['side'] == 'LONG':
                price_change_pct = (close_price / position['entry_price']) - 1
            else:
                price_change_pct = (position['entry_price'] / close_price) - 1

            leveraged_pnl_pct = price_change_pct * LEVERAGE
            pnl_usd = position['size_usd'] * leveraged_pnl_pct

            # Trading fees (0.04% entry + 0.04% exit = 0.08% total)
            total_fee = position['size_usd'] * 0.0008
            net_pnl = pnl_usd - total_fee

            balance += net_pnl

            # Record trade
            trades.append({
                'entry_time': position['entry_time'],
                'exit_time': timestamp,
                'side': position['side'],
                'entry_price': position['entry_price'],
                'exit_price': close_price,
                'size_usd': position['size_usd'],
                'entry_prob': position['entry_prob'],
                'hold_candles': hold_time,
                'price_change_pct': price_change_pct * 100,
                'leveraged_pnl_pct': leveraged_pnl_pct * 100,
                'pnl_usd': pnl_usd,
                'fee_usd': total_fee,
                'net_pnl': net_pnl,
                'exit_reason': exit_reason,
                'balance_after': balance
            })

            position = None

    # Entry logic (only if no position)
    if position is None:
        candle_row = candle.to_frame().T

        # LONG Entry
        long_feat_df = candle_row[long_entry_features]
        long_feat_normalized = long_entry_scaler.transform(long_feat_df.values)
        long_prob = long_entry_model.predict_proba(long_feat_normalized)[0][1]

        # SHORT Entry
        short_feat_df = candle_row[short_entry_features]
        short_feat_normalized = short_entry_scaler.transform(short_feat_df.values)
        short_prob = short_entry_model.predict_proba(short_feat_normalized)[0][1]

        # Opportunity gating
        if long_prob >= LONG_THRESHOLD:
            # Enter LONG
            size_pct = calculate_position_size(long_prob)
            size_usd = balance * size_pct

            position = {
                'side': 'LONG',
                'entry_time': timestamp,
                'entry_idx': idx,
                'entry_price': close_price,
                'size_usd': size_usd,
                'entry_prob': long_prob
            }
        elif short_prob >= SHORT_THRESHOLD:
            # Opportunity cost check
            long_ev = long_prob * 0.0041
            short_ev = short_prob * 0.0047
            opportunity_cost = short_ev - long_ev

            if opportunity_cost > 0.001:
                # Enter SHORT
                size_pct = calculate_position_size(short_prob)
                size_usd = balance * size_pct

                position = {
                    'side': 'SHORT',
                    'entry_time': timestamp,
                    'entry_idx': idx,
                    'entry_price': close_price,
                    'size_usd': size_usd,
                    'entry_prob': short_prob
                }

# Close any remaining position at end
if position is not None:
    close_price = df_features.iloc[-1]['close']
    timestamp = df_features.iloc[-1]['timestamp']

    if position['side'] == 'LONG':
        price_change_pct = (close_price / position['entry_price']) - 1
    else:
        price_change_pct = (position['entry_price'] / close_price) - 1

    leveraged_pnl_pct = price_change_pct * LEVERAGE
    pnl_usd = position['size_usd'] * leveraged_pnl_pct
    total_fee = position['size_usd'] * 0.0008
    net_pnl = pnl_usd - total_fee
    balance += net_pnl

    hold_time = len(df_features) - 1 - position['entry_idx']

    trades.append({
        'entry_time': position['entry_time'],
        'exit_time': timestamp,
        'side': position['side'],
        'entry_price': position['entry_price'],
        'exit_price': close_price,
        'size_usd': position['size_usd'],
        'entry_prob': position['entry_prob'],
        'hold_candles': hold_time,
        'price_change_pct': price_change_pct * 100,
        'leveraged_pnl_pct': leveraged_pnl_pct * 100,
        'pnl_usd': pnl_usd,
        'fee_usd': total_fee,
        'net_pnl': net_pnl,
        'exit_reason': 'End of Period',
        'balance_after': balance
    })

# Results
print("\n" + "="*80)
print("BACKTEST RESULTS - RECENT 5 DAYS")
print("="*80)

df_trades = pd.DataFrame(trades)
total_return = ((balance - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100

print(f"\n📊 Performance Summary:")
print(f"   Initial Capital: ${INITIAL_CAPITAL:,.2f}")
print(f"   Final Balance: ${balance:,.2f}")
print(f"   Total Return: {total_return:+.2f}%")
print(f"   Net P&L: ${balance - INITIAL_CAPITAL:+,.2f}")

if len(df_trades) > 0:
    wins = df_trades[df_trades['net_pnl'] > 0]
    losses = df_trades[df_trades['net_pnl'] <= 0]

    print(f"\n📈 Trade Statistics:")
    print(f"   Total Trades: {len(df_trades)}")
    print(f"   Wins: {len(wins)} ({len(wins)/len(df_trades)*100:.1f}%)")
    print(f"   Losses: {len(losses)} ({len(losses)/len(df_trades)*100:.1f}%)")
    print(f"   Trades per Day: {len(df_trades) / 5:.1f}")

    print(f"\n💰 P&L Breakdown:")
    print(f"   Average Trade: ${df_trades['net_pnl'].mean():+.2f}")
    print(f"   Average Win: ${wins['net_pnl'].mean():+.2f}" if len(wins) > 0 else "   Average Win: N/A")
    print(f"   Average Loss: ${losses['net_pnl'].mean():+.2f}" if len(losses) > 0 else "   Average Loss: N/A")
    print(f"   Best Trade: ${df_trades['net_pnl'].max():+.2f}")
    print(f"   Worst Trade: ${df_trades['net_pnl'].min():+.2f}")

    if len(wins) > 0 and len(losses) > 0:
        profit_factor = wins['net_pnl'].sum() / abs(losses['net_pnl'].sum())
        print(f"   Profit Factor: {profit_factor:.2f}x")

    print(f"\n⏱️  Hold Time:")
    print(f"   Average: {df_trades['hold_candles'].mean():.1f} candles ({df_trades['hold_candles'].mean()/12:.1f} hours)")
    print(f"   Median: {df_trades['hold_candles'].median():.0f} candles ({df_trades['hold_candles'].median()/12:.1f} hours)")
    print(f"   Max: {df_trades['hold_candles'].max():.0f} candles ({df_trades['hold_candles'].max()/12:.1f} hours)")

    print(f"\n🎯 Side Distribution:")
    long_trades = df_trades[df_trades['side'] == 'LONG']
    short_trades = df_trades[df_trades['side'] == 'SHORT']
    print(f"   LONG: {len(long_trades)} ({len(long_trades)/len(df_trades)*100:.1f}%)")
    print(f"   SHORT: {len(short_trades)} ({len(short_trades)/len(df_trades)*100:.1f}%)")

    print(f"\n🚪 Exit Distribution:")
    for reason in df_trades['exit_reason'].unique():
        count = len(df_trades[df_trades['exit_reason'].str.contains(reason.split('(')[0].strip())])
        pct = count / len(df_trades) * 100
        print(f"   {reason.split('(')[0].strip()}: {count} ({pct:.1f}%)")

    # Recent trades
    print(f"\n📋 Most Recent Trades (Last 5):")
    print("-" * 80)
    for _, trade in df_trades.tail(5).iterrows():
        hold_hours = trade['hold_candles'] / 12
        print(f"{trade['entry_time']} | {trade['side']:5s} | "
              f"Entry: ${trade['entry_price']:>10,.2f} | "
              f"Exit: ${trade['exit_price']:>10,.2f} | "
              f"Hold: {hold_hours:4.1f}h | "
              f"P&L: ${trade['net_pnl']:+7.2f} | "
              f"{trade['exit_reason']}")
else:
    print("\n⚠️ No trades executed during backtest period")

# Risk metrics
df_equity = pd.DataFrame(equity_curve)
returns = df_equity['equity'].pct_change().dropna()
max_drawdown = ((df_equity['equity'].cummax() - df_equity['equity']) / df_equity['equity'].cummax()).max() * 100

print(f"\n⚠️  Risk Metrics:")
print(f"   Max Drawdown: {max_drawdown:.2f}%")
if len(returns) > 0:
    sharpe_ratio = returns.mean() / returns.std() * np.sqrt(12 * 24 * 5) if returns.std() > 0 else 0
    print(f"   Sharpe Ratio: {sharpe_ratio:.3f}")

# Save results
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_file = PROJECT_ROOT / "results" / f"backtest_recent_5days_{timestamp}.csv"
df_trades.to_csv(results_file, index=False)
print(f"\n💾 Results saved: {results_file}")

print("\n" + "="*80)
