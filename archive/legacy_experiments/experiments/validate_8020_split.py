"""
Validate 80/20 Split Models on 20% Holdout
===========================================

Test 80/20 Split models on unseen 20% holdout (6,001 candles, Oct 5-26)
Compare to Phase 2 Enhanced (-59.14%) and Current Production (-59.48%)
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime

# Paths
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = BASE_DIR / "data" / "features"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

# Configuration (same as training)
ENTRY_THRESHOLD = 0.75
ML_EXIT_THRESHOLD = 0.70  # Current production setting
LEVERAGE = 4
EMERGENCY_STOP_LOSS = -0.03  # -3% of balance
EMERGENCY_MAX_HOLD = 120  # 10 hours
INITIAL_BALANCE = 10000.0

print("=" * 80)
print("80/20 SPLIT MODELS VALIDATION")
print("=" * 80)
print()

# Load complete dataset
print("Loading complete dataset...")
df = pd.read_csv(DATA_DIR / "BTCUSDT_5m_features_complete.csv")
df['timestamp'] = pd.to_datetime(df['timestamp'])
print(f"✅ Loaded {len(df):,} candles")
print(f"   Date range: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")
print()

# Split into training (80%) and validation (20%)
train_size = int(len(df) * 0.8)
df_validation = df.iloc[train_size:].copy()

print("Validation Holdout (20%):")
print(f"  Candles: {len(df_validation):,}")
print(f"  Date range: {df_validation['timestamp'].iloc[0]} to {df_validation['timestamp'].iloc[-1]}")
print(f"  Period: {(df_validation['timestamp'].iloc[-1] - df_validation['timestamp'].iloc[0]).days} days")
print()

# Load 80/20 Split models (timestamp: 20251102_204412)
timestamp = "20251102_204412"
print(f"Loading 80/20 Split models (timestamp: {timestamp})...")

long_entry_model = joblib.load(MODELS_DIR / f"xgboost_long_entry_8020split_{timestamp}.pkl")
long_entry_scaler = joblib.load(MODELS_DIR / f"xgboost_long_entry_8020split_{timestamp}_scaler.pkl")
with open(MODELS_DIR / f"xgboost_long_entry_8020split_{timestamp}_features.txt", 'r') as f:
    long_entry_features = [line.strip() for line in f if line.strip()]

short_entry_model = joblib.load(MODELS_DIR / f"xgboost_short_entry_8020split_{timestamp}.pkl")
short_entry_scaler = joblib.load(MODELS_DIR / f"xgboost_short_entry_8020split_{timestamp}_scaler.pkl")
with open(MODELS_DIR / f"xgboost_short_entry_8020split_{timestamp}_features.txt", 'r') as f:
    short_entry_features = [line.strip() for line in f if line.strip()]

long_exit_model = joblib.load(MODELS_DIR / f"xgboost_long_exit_8020split_{timestamp}.pkl")
long_exit_scaler = joblib.load(MODELS_DIR / f"xgboost_long_exit_8020split_{timestamp}_scaler.pkl")
with open(MODELS_DIR / f"xgboost_long_exit_8020split_{timestamp}_features.txt", 'r') as f:
    long_exit_features = [line.strip() for line in f if line.strip()]

short_exit_model = joblib.load(MODELS_DIR / f"xgboost_short_exit_8020split_{timestamp}.pkl")
short_exit_scaler = joblib.load(MODELS_DIR / f"xgboost_short_exit_8020split_{timestamp}_scaler.pkl")
with open(MODELS_DIR / f"xgboost_short_exit_8020split_{timestamp}_features.txt", 'r') as f:
    short_exit_features = [line.strip() for line in f if line.strip()]

print(f"✅ Models loaded successfully")
print(f"   LONG Entry: {len(long_entry_features)} features")
print(f"   SHORT Entry: {len(short_entry_features)} features")
print(f"   LONG Exit: {len(long_exit_features)} features")
print(f"   SHORT Exit: {len(short_exit_features)} features")
print()

# Prepare enhanced exit features function
def prepare_exit_features(df):
    """Add 14 enhanced features for Exit models (exact match with training)"""
    df = df.copy()

    # Volume surge
    df['volume_surge'] = df['volume'] / df['volume'].rolling(20).mean()

    # Price acceleration
    df['price_acceleration'] = df['close'].diff(2) - df['close'].diff(1)

    # Price position relative to moving averages
    df['price_vs_ma20'] = (df['close'] - df['sma_20']) / df['sma_20']
    df['price_vs_ma50'] = (df['close'] - df['sma_50']) / df['sma_50']

    # Volatility
    df['volatility_20'] = df['close'].rolling(20).std()

    # RSI features
    df['rsi_slope'] = df['rsi'].diff(5)
    df['rsi_overbought'] = (df['rsi'] > 70).astype(int)
    df['rsi_oversold'] = (df['rsi'] < 30).astype(int)
    df['rsi_divergence'] = (df['rsi'].diff(10) * df['close'].diff(10)) < 0

    # MACD features
    df['macd_histogram_slope'] = df['macd_histogram'].diff(3)
    df['macd_crossover'] = ((df['macd'] > df['macd_signal']) &
                             (df['macd'].shift(1) <= df['macd_signal'].shift(1))).astype(int)
    df['macd_crossunder'] = ((df['macd'] < df['macd_signal']) &
                              (df['macd'].shift(1) >= df['macd_signal'].shift(1))).astype(int)

    # Bollinger Bands position (if not already present)
    if 'bb_position' not in df.columns:
        df['bb_position'] = (df['close'] - df['bb_low']) / (df['bb_high'] - df['bb_low'])

    # Price action
    df['higher_high'] = (df['high'] > df['high'].shift(1)).astype(int)

    return df

# Add enhanced exit features
df_validation = prepare_exit_features(df_validation)

# Drop rows with NaN after feature creation
initial_count = len(df_validation)
df_validation = df_validation.dropna()
dropped = initial_count - len(df_validation)
print(f"Dropped {dropped} rows with NaN (warmup period)")
print(f"Validation candles after cleanup: {len(df_validation):,}")
print()

# Backtest simulation
print("=" * 80)
print("RUNNING BACKTEST ON 20% HOLDOUT")
print("=" * 80)
print()

balance = INITIAL_BALANCE
position = None
trades = []

for idx in range(len(df_validation)):
    row = df_validation.iloc[idx]
    current_time = row['timestamp']
    current_price = row['close']

    # Check for existing position first
    if position:
        # Check exit conditions
        hold_time = idx - position['entry_idx']
        leveraged_pnl_pct = (current_price - position['entry_price']) / position['entry_price']
        if position['side'] == 'SHORT':
            leveraged_pnl_pct = -leveraged_pnl_pct
        leveraged_pnl_pct *= LEVERAGE

        # Emergency Stop Loss
        position_size_pct = position['position_value'] / position['balance_at_entry']
        balance_loss_pct = leveraged_pnl_pct * position_size_pct

        if balance_loss_pct <= EMERGENCY_STOP_LOSS:
            # Stop Loss triggered
            balance = position['balance_at_entry'] * (1 + balance_loss_pct)
            trades.append({
                'entry_time': position['entry_time'],
                'exit_time': current_time,
                'side': position['side'],
                'entry_price': position['entry_price'],
                'exit_price': current_price,
                'position_size': position['position_value'],
                'leveraged_pnl_pct': leveraged_pnl_pct * 100,
                'balance_change_pct': balance_loss_pct * 100,
                'hold_candles': hold_time,
                'exit_reason': 'Stop Loss'
            })
            position = None
            continue

        # Emergency Max Hold
        if hold_time >= EMERGENCY_MAX_HOLD:
            # Max hold triggered
            balance = position['balance_at_entry'] * (1 + balance_loss_pct)
            trades.append({
                'entry_time': position['entry_time'],
                'exit_time': current_time,
                'side': position['side'],
                'entry_price': position['entry_price'],
                'exit_price': current_price,
                'position_size': position['position_value'],
                'leveraged_pnl_pct': leveraged_pnl_pct * 100,
                'balance_change_pct': balance_loss_pct * 100,
                'hold_candles': hold_time,
                'exit_reason': 'Max Hold'
            })
            position = None
            continue

        # ML Exit signal
        exit_features = long_exit_features if position['side'] == 'LONG' else short_exit_features
        exit_scaler = long_exit_scaler if position['side'] == 'LONG' else short_exit_scaler
        exit_model = long_exit_model if position['side'] == 'LONG' else short_exit_model

        exit_feat_df = df_validation.iloc[[idx]][exit_features]
        if exit_feat_df.shape[1] == len(exit_features):
            exit_feat_scaled = exit_scaler.transform(exit_feat_df.values)
            exit_prob = exit_model.predict_proba(exit_feat_scaled)[0][1]

            if exit_prob >= ML_EXIT_THRESHOLD:
                # ML Exit triggered
                balance = position['balance_at_entry'] * (1 + balance_loss_pct)
                trades.append({
                    'entry_time': position['entry_time'],
                    'exit_time': current_time,
                    'side': position['side'],
                    'entry_price': position['entry_price'],
                    'exit_price': current_price,
                    'position_size': position['position_value'],
                    'leveraged_pnl_pct': leveraged_pnl_pct * 100,
                    'balance_change_pct': balance_loss_pct * 100,
                    'hold_candles': hold_time,
                    'exit_reason': 'ML Exit',
                    'exit_prob': exit_prob
                })
                position = None
                continue

    # Entry signals (only if no position)
    if not position:
        # LONG Entry signal
        long_feat_df = df_validation.iloc[[idx]][long_entry_features]
        if long_feat_df.shape[1] == len(long_entry_features):
            long_feat_scaled = long_entry_scaler.transform(long_feat_df.values)
            long_prob = long_entry_model.predict_proba(long_feat_scaled)[0][1]
        else:
            long_prob = 0.0

        # SHORT Entry signal
        short_feat_df = df_validation.iloc[[idx]][short_entry_features]
        if short_feat_df.shape[1] == len(short_entry_features):
            short_feat_scaled = short_entry_scaler.transform(short_feat_df.values)
            short_prob = short_entry_model.predict_proba(short_feat_scaled)[0][1]
        else:
            short_prob = 0.0

        # Enter LONG
        if long_prob >= ENTRY_THRESHOLD:
            # Dynamic position sizing (linear from 20% to 95%)
            position_pct = 0.20 + (long_prob - ENTRY_THRESHOLD) * (0.95 - 0.20) / (1.0 - ENTRY_THRESHOLD)
            position_value = balance * position_pct

            position = {
                'side': 'LONG',
                'entry_time': current_time,
                'entry_price': current_price,
                'entry_idx': idx,
                'position_value': position_value,
                'balance_at_entry': balance,
                'entry_prob': long_prob
            }

        # Enter SHORT (with opportunity gating)
        elif short_prob >= ENTRY_THRESHOLD:
            # Opportunity cost gating (simplified: just use threshold)
            # Dynamic position sizing
            position_pct = 0.20 + (short_prob - ENTRY_THRESHOLD) * (0.95 - 0.20) / (1.0 - ENTRY_THRESHOLD)
            position_value = balance * position_pct

            position = {
                'side': 'SHORT',
                'entry_time': current_time,
                'entry_price': current_price,
                'entry_idx': idx,
                'position_value': position_value,
                'balance_at_entry': balance,
                'entry_prob': short_prob
            }

# Close any remaining position at end
if position:
    current_price = df_validation.iloc[-1]['close']
    current_time = df_validation.iloc[-1]['timestamp']
    hold_time = len(df_validation) - 1 - position['entry_idx']

    leveraged_pnl_pct = (current_price - position['entry_price']) / position['entry_price']
    if position['side'] == 'SHORT':
        leveraged_pnl_pct = -leveraged_pnl_pct
    leveraged_pnl_pct *= LEVERAGE

    position_size_pct = position['position_value'] / position['balance_at_entry']
    balance_loss_pct = leveraged_pnl_pct * position_size_pct
    balance = position['balance_at_entry'] * (1 + balance_loss_pct)

    trades.append({
        'entry_time': position['entry_time'],
        'exit_time': current_time,
        'side': position['side'],
        'entry_price': position['entry_price'],
        'exit_price': current_price,
        'position_size': position['position_value'],
        'leveraged_pnl_pct': leveraged_pnl_pct * 100,
        'balance_change_pct': balance_loss_pct * 100,
        'hold_candles': hold_time,
        'exit_reason': 'End of Period'
    })

# Calculate performance metrics
print("=" * 80)
print("VALIDATION RESULTS")
print("=" * 80)
print()

total_return = ((balance - INITIAL_BALANCE) / INITIAL_BALANCE) * 100
wins = sum(1 for t in trades if t['balance_change_pct'] > 0)
losses = len(trades) - wins
win_rate = (wins / len(trades) * 100) if len(trades) > 0 else 0.0

print(f"Configuration:")
print(f"  Entry Threshold: {ENTRY_THRESHOLD}")
print(f"  ML Exit Threshold: {ML_EXIT_THRESHOLD}")
print(f"  Leverage: {LEVERAGE}x")
print(f"  Stop Loss: {EMERGENCY_STOP_LOSS * 100:.1f}% balance")
print(f"  Max Hold: {EMERGENCY_MAX_HOLD} candles ({EMERGENCY_MAX_HOLD / 12:.1f}h)")
print()

print(f"Overall Performance:")
print(f"  Initial Balance: ${INITIAL_BALANCE:,.2f}")
print(f"  Final Balance: ${balance:,.2f}")
print(f"  Total Return: {total_return:+.2f}%")
print(f"  Total Trades: {len(trades)}")
print(f"  Win Rate: {win_rate:.1f}% ({wins}W / {losses}L)")
print()

if len(trades) > 0:
    # Exit reason distribution
    exit_reasons = {}
    for t in trades:
        reason = t['exit_reason']
        exit_reasons[reason] = exit_reasons.get(reason, 0) + 1

    print("Exit Distribution:")
    for reason, count in sorted(exit_reasons.items(), key=lambda x: x[1], reverse=True):
        pct = count / len(trades) * 100
        print(f"  {reason}: {count} ({pct:.1f}%)")
    print()

    # Side distribution
    long_trades = sum(1 for t in trades if t['side'] == 'LONG')
    short_trades = len(trades) - long_trades
    print("Side Distribution:")
    print(f"  LONG: {long_trades} ({long_trades / len(trades) * 100:.1f}%)")
    print(f"  SHORT: {short_trades} ({short_trades / len(trades) * 100:.1f}%)")
    print()

    # Average metrics
    avg_hold = np.mean([t['hold_candles'] for t in trades])
    avg_pnl = np.mean([t['balance_change_pct'] for t in trades])

    wins_only = [t for t in trades if t['balance_change_pct'] > 0]
    losses_only = [t for t in trades if t['balance_change_pct'] <= 0]

    avg_win = np.mean([t['balance_change_pct'] for t in wins_only]) if wins_only else 0
    avg_loss = np.mean([t['balance_change_pct'] for t in losses_only]) if losses_only else 0

    print("Trade Characteristics:")
    print(f"  Avg Hold Time: {avg_hold:.1f} candles ({avg_hold / 12:.1f}h)")
    print(f"  Avg P&L: {avg_pnl:+.2f}%")
    print(f"  Avg Win: {avg_win:+.2f}%")
    print(f"  Avg Loss: {avg_loss:+.2f}%")
    print()

# Comparison to baselines
print("=" * 80)
print("COMPARISON TO BASELINES")
print("=" * 80)
print()

baseline_phase2 = -59.14
baseline_prod = -59.48

improvement_phase2 = total_return - baseline_phase2
improvement_prod = total_return - baseline_prod

print(f"Phase 2 Enhanced (Baseline 1):")
print(f"  Return: {baseline_phase2:+.2f}%")
print(f"  80/20 Split: {total_return:+.2f}%")
print(f"  Improvement: {improvement_phase2:+.2f}%")
print(f"  Status: {'✅ BEAT BASELINE' if improvement_phase2 > 5 else '❌ BELOW THRESHOLD'}")
print()

print(f"Current Production (Baseline 2):")
print(f"  Return: {baseline_prod:+.2f}%")
print(f"  80/20 Split: {total_return:+.2f}%")
print(f"  Improvement: {improvement_prod:+.2f}%")
print(f"  Status: {'✅ BEAT BASELINE' if improvement_prod > 5 else '❌ BELOW THRESHOLD'}")
print()

# Deployment decision
print("=" * 80)
print("DEPLOYMENT DECISION")
print("=" * 80)
print()

if improvement_phase2 > 5 and improvement_prod > 5:
    print("✅ DEPLOY RECOMMENDED")
    print(f"   Both baselines beaten by >5%")
    print(f"   Phase 2 improvement: +{improvement_phase2:.2f}%")
    print(f"   Production improvement: +{improvement_prod:.2f}%")
else:
    print("❌ DO NOT DEPLOY")
    print(f"   Improvement threshold not met (need >5%)")
    print(f"   Phase 2 improvement: +{improvement_phase2:.2f}%")
    print(f"   Production improvement: +{improvement_prod:.2f}%")
print()

# Save results
trades_df = pd.DataFrame(trades)
output_file = RESULTS_DIR / f"validate_8020split_20% holdout_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
trades_df.to_csv(output_file, index=False)
print(f"✅ Results saved: {output_file.name}")
