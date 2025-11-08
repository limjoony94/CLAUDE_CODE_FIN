"""
24-Hour Production Backtest
지난 24시간 데이터로 프로덕션 모델/설정 백테스트
"""

import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(project_root)
sys.path.append(os.path.join(project_root, 'scripts', 'experiments'))
sys.path.append(os.path.join(project_root, 'scripts', 'production'))

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pytz
import joblib
import yaml
from src.api.bingx_client import BingXClient
from calculate_all_features_enhanced_v2 import calculate_all_features_enhanced_v2
from retrain_exit_models_opportunity_gating import prepare_exit_features
from dynamic_position_sizing import DynamicPositionSizer

# =============================================================================
# PRODUCTION CONFIGURATION (프로덕션 봇과 동일)
# =============================================================================

# Entry Thresholds
LONG_THRESHOLD = 0.70
SHORT_THRESHOLD = 0.70
GATE_THRESHOLD = 0.001

# Exit Parameters
ML_EXIT_THRESHOLD_LONG = 0.80
ML_EXIT_THRESHOLD_SHORT = 0.80
EMERGENCY_STOP_LOSS = 0.03  # -3% balance
EMERGENCY_MAX_HOLD_TIME = 120  # 10 hours

# Leverage & Sizing
LEVERAGE = 4
INITIAL_BALANCE = 4672.5318  # Current production balance

# Expected Values
LONG_AVG_RETURN = 0.0041
SHORT_AVG_RETURN = 0.0047

# =============================================================================
# LOAD PRODUCTION MODELS
# =============================================================================

print("Loading production models...")

# LONG Entry
long_model = joblib.load('models/xgboost_long_entry_enhanced_20251024_012445.pkl')
long_scaler = joblib.load('models/xgboost_long_entry_enhanced_20251024_012445_scaler.pkl')
with open('models/xgboost_long_entry_enhanced_20251024_012445_features.txt', 'r') as f:
    long_features = [line.strip() for line in f if line.strip()]

# SHORT Entry
short_model = joblib.load('models/xgboost_short_entry_enhanced_20251024_012445.pkl')
short_scaler = joblib.load('models/xgboost_short_entry_enhanced_20251024_012445_scaler.pkl')
with open('models/xgboost_short_entry_enhanced_20251024_012445_features.txt', 'r') as f:
    short_features = [line.strip() for line in f if line.strip()]

# LONG Exit
long_exit_model = joblib.load('models/xgboost_long_exit_oppgating_improved_20251024_043527.pkl')
long_exit_scaler = joblib.load('models/xgboost_long_exit_oppgating_improved_20251024_043527_scaler.pkl')
with open('models/xgboost_long_exit_oppgating_improved_20251024_043527_features.txt', 'r') as f:
    long_exit_features = [line.strip() for line in f if line.strip()]

# SHORT Exit
short_exit_model = joblib.load('models/xgboost_short_exit_oppgating_improved_20251024_044510.pkl')
short_exit_scaler = joblib.load('models/xgboost_short_exit_oppgating_improved_20251024_044510_scaler.pkl')
with open('models/xgboost_short_exit_oppgating_improved_20251024_044510_features.txt', 'r') as f:
    short_exit_features = [line.strip() for line in f if line.strip()]

print(f"✅ Models loaded:")
print(f"   LONG Entry: {len(long_features)} features")
print(f"   SHORT Entry: {len(short_features)} features")
print(f"   LONG Exit: {len(long_exit_features)} features")
print(f"   SHORT Exit: {len(short_exit_features)} features")

# Initialize Position Sizer
sizer = DynamicPositionSizer(
    base_position_pct=0.50,
    max_position_pct=0.95,
    min_position_pct=0.20
)
print(f"✅ Position Sizer initialized")

# =============================================================================
# FETCH 24H DATA
# =============================================================================

print("\n" + "="*80)
print("FETCHING 24-HOUR DATA")
print("="*80)

# Load API keys
config_path = os.path.join(project_root, 'config', 'api_keys.yaml')
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

# Initialize client
client = BingXClient(
    api_key=config['bingx']['mainnet']['api_key'],
    secret_key=config['bingx']['mainnet']['secret_key']
)

# Calculate time range
kst = pytz.timezone('Asia/Seoul')
now_kst = datetime.now(kst)
start_time = now_kst - timedelta(hours=24)

print(f"Period: {start_time.strftime('%Y-%m-%d %H:%M KST')} ~ {now_kst.strftime('%Y-%m-%d %H:%M KST')}")

# Fetch data (need extra for feature calculation)
start_utc = start_time.astimezone(pytz.UTC)
end_utc = now_kst.astimezone(pytz.UTC)

# Get 1000 candles to ensure we have enough for feature calculation
klines = client.get_klines(
    symbol='BTC-USDT',
    interval='5m',
    limit=1000
)

# Convert to DataFrame
# Note: klines returns list of dicts with 'time' key (not 'timestamp')
df = pd.DataFrame(klines)
df = df.rename(columns={'time': 'timestamp'})

# Convert timestamp (milliseconds to datetime)
df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
df['timestamp'] = df['timestamp'].dt.tz_convert('Asia/Seoul').dt.tz_localize(None)

# Convert price columns
for col in ['open', 'high', 'low', 'close', 'volume']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

print(f"Fetched {len(df)} raw candles")
if len(df) > 0:
    print(f"Raw date range: {df['timestamp'].min()} ~ {df['timestamp'].max()}")

# Don't filter here - we need historical data for features
# We'll filter after feature calculation

# =============================================================================
# CALCULATE FEATURES
# =============================================================================

print("\nCalculating features...")

# Calculate all features (df already has 1000 candles for lookback)
df_features = calculate_all_features_enhanced_v2(df.copy(), phase='phase1')
df_features = prepare_exit_features(df_features)

print(f"Features calculated: {len(df_features)} total rows")
print(f"Total features: {len(df_features.columns)}")

# Filter to 24h period AFTER feature calculation
if len(df_features) > 0:
    df_features = df_features[df_features['timestamp'] >= start_time.replace(tzinfo=None)].copy()
    print(f"Filtered to 24h period: {len(df_features)} rows")
    print(f"Period: {df_features['timestamp'].min()} ~ {df_features['timestamp'].max()}")
else:
    print("⚠️  No features calculated")
    sys.exit(1)

# =============================================================================
# BACKTEST ENGINE
# =============================================================================

print("\n" + "="*80)
print("RUNNING BACKTEST")
print("="*80)

balance = INITIAL_BALANCE
position = None
trades = []
equity_curve = []

for idx in range(len(df_features)):
    row = df_features.iloc[idx]
    timestamp = row['timestamp']
    price = row['close']

    # Track equity
    unrealized_pnl = 0
    if position is not None:
        if position['side'] == 'LONG':
            unrealized_pnl = (price - position['entry_price']) / position['entry_price'] * position['size'] * LEVERAGE
        else:  # SHORT
            unrealized_pnl = (position['entry_price'] - price) / position['entry_price'] * position['size'] * LEVERAGE

    equity = balance + unrealized_pnl
    equity_curve.append({
        'timestamp': timestamp,
        'balance': balance,
        'equity': equity,
        'price': price
    })

    # === EXIT LOGIC ===
    if position is not None:
        should_exit = False
        exit_reason = None

        # Calculate P&L
        if position['side'] == 'LONG':
            pnl_pct = (price - position['entry_price']) / position['entry_price']
            pnl_usd = pnl_pct * position['size'] * LEVERAGE
        else:  # SHORT
            pnl_pct = (position['entry_price'] - price) / position['entry_price']
            pnl_usd = pnl_pct * position['size'] * LEVERAGE

        # Emergency Stop Loss (balance-based)
        balance_loss_pct = pnl_usd / balance
        if balance_loss_pct <= -EMERGENCY_STOP_LOSS:
            should_exit = True
            exit_reason = 'emergency_sl'

        # Emergency Max Hold
        hold_time = idx - position['entry_idx']
        if hold_time >= EMERGENCY_MAX_HOLD_TIME:
            should_exit = True
            exit_reason = 'max_hold'

        # ML Exit Signal
        if not should_exit:
            if position['side'] == 'LONG':
                exit_feat = row[long_exit_features].values.reshape(1, -1)
                exit_feat_scaled = long_exit_scaler.transform(exit_feat)
                exit_prob = long_exit_model.predict_proba(exit_feat_scaled)[0][1]

                if exit_prob >= ML_EXIT_THRESHOLD_LONG:
                    should_exit = True
                    exit_reason = 'ml_exit'
            else:  # SHORT
                exit_feat = row[short_exit_features].values.reshape(1, -1)
                exit_feat_scaled = short_exit_scaler.transform(exit_feat)
                exit_prob = short_exit_model.predict_proba(exit_feat_scaled)[0][1]

                if exit_prob >= ML_EXIT_THRESHOLD_SHORT:
                    should_exit = True
                    exit_reason = 'ml_exit'

        # Execute Exit
        if should_exit:
            balance += pnl_usd

            trades.append({
                'entry_time': position['entry_time'],
                'exit_time': timestamp,
                'side': position['side'],
                'entry_price': position['entry_price'],
                'exit_price': price,
                'size': position['size'],
                'pnl_usd': pnl_usd,
                'pnl_pct': pnl_pct,
                'hold_candles': hold_time,
                'exit_reason': exit_reason,
                'entry_long_prob': position.get('long_prob', 0),
                'entry_short_prob': position.get('short_prob', 0)
            })

            position = None

    # === ENTRY LOGIC ===
    if position is None:
        # Calculate entry signals
        try:
            long_feat = row[long_features].values.reshape(1, -1)
            long_feat_scaled = long_scaler.transform(long_feat)
            long_prob = long_model.predict_proba(long_feat_scaled)[0][1]
        except:
            long_prob = 0.0

        try:
            short_feat = row[short_features].values.reshape(1, -1)
            short_feat_scaled = short_scaler.transform(short_feat)
            short_prob = short_model.predict_proba(short_feat_scaled)[0][1]
        except:
            short_prob = 0.0

        # Check LONG entry
        if long_prob >= LONG_THRESHOLD:
            sizing_result = sizer.get_position_size_simple(
                capital=balance,
                signal_strength=long_prob,
                leverage=LEVERAGE
            )
            size_pct = sizing_result['position_size_pct']
            position = {
                'side': 'LONG',
                'entry_time': timestamp,
                'entry_price': price,
                'entry_idx': idx,
                'size': balance * size_pct,
                'long_prob': long_prob,
                'short_prob': short_prob
            }

        # Check SHORT entry (with Opportunity Gate)
        elif short_prob >= SHORT_THRESHOLD:
            long_ev = long_prob * LONG_AVG_RETURN
            short_ev = short_prob * SHORT_AVG_RETURN
            opportunity_cost = short_ev - long_ev

            if opportunity_cost > GATE_THRESHOLD:
                sizing_result = sizer.get_position_size_simple(
                    capital=balance,
                    signal_strength=short_prob,
                    leverage=LEVERAGE
                )
                size_pct = sizing_result['position_size_pct']
                position = {
                    'side': 'SHORT',
                    'entry_time': timestamp,
                    'entry_price': price,
                    'entry_idx': idx,
                    'size': balance * size_pct,
                    'long_prob': long_prob,
                    'short_prob': short_prob
                }

# Close any remaining position
if position is not None:
    price = df_features.iloc[-1]['close']
    timestamp = df_features.iloc[-1]['timestamp']

    if position['side'] == 'LONG':
        pnl_pct = (price - position['entry_price']) / position['entry_price']
    else:
        pnl_pct = (position['entry_price'] - price) / position['entry_price']

    pnl_usd = pnl_pct * position['size'] * LEVERAGE
    balance += pnl_usd

    trades.append({
        'entry_time': position['entry_time'],
        'exit_time': timestamp,
        'side': position['side'],
        'entry_price': position['entry_price'],
        'exit_price': price,
        'size': position['size'],
        'pnl_usd': pnl_usd,
        'pnl_pct': pnl_pct,
        'hold_candles': len(df_features) - 1 - position['entry_idx'],
        'exit_reason': 'end_of_period',
        'entry_long_prob': position.get('long_prob', 0),
        'entry_short_prob': position.get('short_prob', 0)
    })

# =============================================================================
# RESULTS
# =============================================================================

print("\n" + "="*80)
print("BACKTEST RESULTS - LAST 24 HOURS")
print("="*80)

df_trades = pd.DataFrame(trades)
df_equity = pd.DataFrame(equity_curve)

# Overall Performance
total_return = (balance - INITIAL_BALANCE) / INITIAL_BALANCE * 100
total_pnl = balance - INITIAL_BALANCE

print(f"\n📊 Overall Performance:")
print(f"   Initial Balance: ${INITIAL_BALANCE:,.2f}")
print(f"   Final Balance:   ${balance:,.2f}")
print(f"   Total P&L:       ${total_pnl:+,.2f}")
print(f"   Total Return:    {total_return:+.2f}%")

if len(df_trades) > 0:
    wins = df_trades[df_trades['pnl_usd'] > 0]
    losses = df_trades[df_trades['pnl_usd'] <= 0]
    win_rate = len(wins) / len(df_trades) * 100

    print(f"\n📈 Trade Statistics:")
    print(f"   Total Trades:    {len(df_trades)}")
    print(f"   Wins:            {len(wins)} ({win_rate:.1f}%)")
    print(f"   Losses:          {len(losses)} ({100-win_rate:.1f}%)")
    print(f"   Avg Win:         ${wins['pnl_usd'].mean():+,.2f}" if len(wins) > 0 else "   Avg Win:         N/A")
    print(f"   Avg Loss:        ${losses['pnl_usd'].mean():+,.2f}" if len(losses) > 0 else "   Avg Loss:        N/A")

    print(f"\n🎯 Trade Distribution:")
    long_trades = df_trades[df_trades['side'] == 'LONG']
    short_trades = df_trades[df_trades['side'] == 'SHORT']
    print(f"   LONG:            {len(long_trades)} ({len(long_trades)/len(df_trades)*100:.1f}%)")
    print(f"   SHORT:           {len(short_trades)} ({len(short_trades)/len(df_trades)*100:.1f}%)")

    print(f"\n⏱️ Hold Time:")
    print(f"   Avg Hold:        {df_trades['hold_candles'].mean():.1f} candles ({df_trades['hold_candles'].mean()/12:.1f}h)")
    print(f"   Max Hold:        {df_trades['hold_candles'].max()} candles ({df_trades['hold_candles'].max()/12:.1f}h)")

    print(f"\n🚪 Exit Reasons:")
    exit_counts = df_trades['exit_reason'].value_counts()
    for reason, count in exit_counts.items():
        pct = count / len(df_trades) * 100
        print(f"   {reason:15s}: {count} ({pct:.1f}%)")

    # Max Drawdown
    df_equity['cummax'] = df_equity['equity'].cummax()
    df_equity['drawdown'] = (df_equity['equity'] - df_equity['cummax']) / df_equity['cummax'] * 100
    max_dd = df_equity['drawdown'].min()

    print(f"\n📉 Risk Metrics:")
    print(f"   Max Drawdown:    {max_dd:.2f}%")

    # Trade Details
    print(f"\n" + "="*80)
    print("TRADE DETAILS")
    print("="*80)

    for i, trade in df_trades.iterrows():
        profit_symbol = "✅" if trade['pnl_usd'] > 0 else "❌"
        print(f"\n{profit_symbol} Trade #{i+1}: {trade['side']}")
        print(f"   Entry:  {trade['entry_time']} @ ${trade['entry_price']:,.1f}")
        print(f"   Exit:   {trade['exit_time']} @ ${trade['exit_price']:,.1f}")
        print(f"   P&L:    ${trade['pnl_usd']:+,.2f} ({trade['pnl_pct']*100:+.2f}%)")
        print(f"   Hold:   {trade['hold_candles']} candles ({trade['hold_candles']/12:.1f}h)")
        print(f"   Exit:   {trade['exit_reason']}")
        print(f"   Signal: LONG {trade['entry_long_prob']:.4f} | SHORT {trade['entry_short_prob']:.4f}")
else:
    print(f"\n⚠️  No trades executed in the 24-hour period")

print("\n" + "="*80)
print("BACKTEST COMPLETE")
print("="*80)

# Save results
output_file = 'results/backtest_24h_production.csv'
if len(df_trades) > 0:
    df_trades.to_csv(output_file, index=False)
    print(f"\n💾 Results saved to: {output_file}")
