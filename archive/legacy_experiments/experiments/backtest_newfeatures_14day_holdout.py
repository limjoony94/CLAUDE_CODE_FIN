"""
Backtest New Feature Models - 14-Day Holdout Validation
========================================================

Validates NEW 120-feature Entry models on 14-day holdout period.

Target Performance:
  - Trade Frequency: 3-8 trades/day at threshold 0.75
  - Returns: Beat (market_change × 4x leverage) benchmark
  - Win Rate: 60-75%
  - LONG/SHORT Balance: ~50/50

Created: 2025-10-29
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# Configuration
ENTRY_THRESHOLD = 0.75
ML_EXIT_THRESHOLD = 0.75
LEVERAGE = 4
EMERGENCY_STOP_LOSS = -0.03
EMERGENCY_MAX_HOLD = 120
INITIAL_CAPITAL = 10000
HOLDOUT_DAYS = 14

DATA_DIR = PROJECT_ROOT / "data" / "features"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

print("="*80)
print("BACKTEST - NEW 120-FEATURE MODELS (14-DAY HOLDOUT)")
print("="*80)
print()
print(f"Configuration:")
print(f"  Entry Threshold: {ENTRY_THRESHOLD}")
print(f"  ML Exit Threshold: {ML_EXIT_THRESHOLD}")
print(f"  Emergency Stop Loss: {EMERGENCY_STOP_LOSS}")
print(f"  Emergency Max Hold: {EMERGENCY_MAX_HOLD} candles")
print(f"  Holdout Period: {HOLDOUT_DAYS} days")
print(f"  Initial Capital: ${INITIAL_CAPITAL:,}")
print()

# ==============================================================================
# STEP 1: Load Data (14-day holdout)
# ==============================================================================

print("-"*80)
print("STEP 1: Loading Holdout Data")
print("-"*80)

df = pd.read_csv(DATA_DIR / "BTCUSDT_5m_pattern_features.csv")
print(f"✅ Full dataset: {len(df):,} candles")

# Take last 14 days
holdout_candles = HOLDOUT_DAYS * 24 * 12  # 14 days × 24 hours × 12 (5min candles)
df = df.iloc[-holdout_candles:].copy().reset_index(drop=True)
print(f"✅ Holdout period: {len(df):,} candles ({HOLDOUT_DAYS} days)")
print(f"   Date range: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")
print()

# Prepare Exit Features
def prepare_exit_features(df):
    df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
    df['volume_surge'] = (df['volume'] > df['volume'].rolling(20).mean() * 1.5).astype(float)

    if 'ma_50' not in df.columns:
        df['ma_50'] = df['close'].rolling(50).mean()
    if 'ma_200' not in df.columns:
        df['ma_200'] = df['close'].rolling(200).mean()

    df['price_vs_ma20'] = (df['close'] - df['ma_50']) / df['ma_50']
    df['price_vs_ma50'] = (df['close'] - df['ma_200']) / df['ma_200']

    df['returns'] = df['close'].pct_change()
    df['volatility_20'] = df['returns'].rolling(20).std()
    df['volatility_regime'] = (df['volatility_20'] > df['volatility_20'].rolling(100).median()).astype(float)

    if 'rsi' in df.columns:
        df['rsi_slope'] = df['rsi'].diff(3) / 3
        df['rsi_overbought'] = (df['rsi'] > 70).astype(float)
        df['rsi_oversold'] = (df['rsi'] < 30).astype(float)
    else:
        df['rsi_slope'] = 0
        df['rsi_overbought'] = 0
        df['rsi_oversold'] = 0

    df['rsi_divergence'] = 0

    if 'macd_hist' in df.columns:
        df['macd_histogram_slope'] = df['macd_hist'].diff(3) / 3
    else:
        df['macd_histogram_slope'] = 0

    if 'macd' in df.columns and 'macd_signal' in df.columns:
        df['macd_crossover'] = ((df['macd'] > df['macd_signal']) &
                                (df['macd'].shift(1) <= df['macd_signal'].shift(1))).astype(float)
        df['macd_crossunder'] = ((df['macd'] < df['macd_signal']) &
                                 (df['macd'].shift(1) >= df['macd_signal'].shift(1))).astype(float)
    else:
        df['macd_crossover'] = 0
        df['macd_crossunder'] = 0

    df['higher_high'] = (df['high'] > df['high'].shift(1)).astype(float)
    df['lower_low'] = (df['low'] < df['low'].shift(1)).astype(float)
    df['price_acceleration'] = df['close'].diff().diff()
    df['near_resistance'] = 0
    df['near_support'] = 0

    if 'bb_upper' in df.columns and 'bb_lower' in df.columns:
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    else:
        df['bb_position'] = 0.5

    df = df.ffill().bfill()
    return df

df = prepare_exit_features(df)
print("✅ Exit features prepared")
print()

# ==============================================================================
# STEP 2: Load Models
# ==============================================================================

print("-"*80)
print("STEP 2: Loading Models")
print("-"*80)

# Load NEW Entry models
long_entry_model = joblib.load(MODELS_DIR / "xgboost_long_entry_newfeatures_20251029_191359.pkl")
long_entry_scaler = joblib.load(MODELS_DIR / "xgboost_long_entry_newfeatures_20251029_191359_scaler.pkl")
with open(MODELS_DIR / "xgboost_long_entry_newfeatures_20251029_191359_features.txt", 'r') as f:
    long_entry_features = [line.strip() for line in f if line.strip()]

short_entry_model = joblib.load(MODELS_DIR / "xgboost_short_entry_newfeatures_20251029_191359.pkl")
short_entry_scaler = joblib.load(MODELS_DIR / "xgboost_short_entry_newfeatures_20251029_191359_scaler.pkl")
with open(MODELS_DIR / "xgboost_short_entry_newfeatures_20251029_191359_features.txt", 'r') as f:
    short_entry_features = [line.strip() for line in f if line.strip()]

print(f"✅ NEW LONG Entry: {len(long_entry_features)} features")
print(f"✅ NEW SHORT Entry: {len(short_entry_features)} features")

# Load Exit models
long_exit_model = joblib.load(MODELS_DIR / "xgboost_long_exit_threshold_075_20251027_190512.pkl")
long_exit_scaler = joblib.load(MODELS_DIR / "xgboost_long_exit_threshold_075_20251027_190512_scaler.pkl")
with open(MODELS_DIR / "xgboost_long_exit_threshold_075_20251027_190512_features.txt", 'r') as f:
    long_exit_features = [line.strip() for line in f if line.strip()]

short_exit_model = joblib.load(MODELS_DIR / "xgboost_short_exit_threshold_075_20251027_190512.pkl")
short_exit_scaler = joblib.load(MODELS_DIR / "xgboost_short_exit_threshold_075_20251027_190512_scaler.pkl")
with open(MODELS_DIR / "xgboost_short_exit_threshold_075_20251027_190512_features.txt", 'r') as f:
    short_exit_features = [line.strip() for line in f if line.strip()]

print(f"✅ LONG Exit: {len(long_exit_features)} features")
print(f"✅ SHORT Exit: {len(short_exit_features)} features")
print()

# ==============================================================================
# STEP 3: Run Backtest
# ==============================================================================

print("-"*80)
print("STEP 3: Running Backtest")
print("-"*80)
print()

capital = INITIAL_CAPITAL
position = None
trades = []

for i in range(100, len(df) - EMERGENCY_MAX_HOLD):
    row = df.iloc[i]
    timestamp = row['timestamp']
    price = row['close']

    # Exit Logic
    if position is not None:
        side = position['side']
        entry_price = position['entry_price']
        position_size_pct = position['position_size_pct']

        if side == 'LONG':
            price_change_pct = (price - entry_price) / entry_price
        else:
            price_change_pct = (entry_price - price) / entry_price

        leveraged_pnl_pct = price_change_pct * LEVERAGE

        should_exit = False
        exit_reason = None

        # Emergency Stop Loss
        if leveraged_pnl_pct <= EMERGENCY_STOP_LOSS:
            should_exit = True
            exit_reason = "STOP_LOSS"

        # Emergency Max Hold
        elif i - position['entry_index'] >= EMERGENCY_MAX_HOLD:
            should_exit = True
            exit_reason = "MAX_HOLD"

        # ML Exit
        else:
            if side == 'LONG':
                X_exit = row[long_exit_features].values.reshape(1, -1)
                X_exit_scaled = long_exit_scaler.transform(X_exit)
                exit_prob = long_exit_model.predict_proba(X_exit_scaled)[0][1]
                if exit_prob >= ML_EXIT_THRESHOLD:
                    should_exit = True
                    exit_reason = "ML_EXIT"
            else:
                X_exit = row[short_exit_features].values.reshape(1, -1)
                X_exit_scaled = short_exit_scaler.transform(X_exit)
                exit_prob = short_exit_model.predict_proba(X_exit_scaled)[0][1]
                if exit_prob >= ML_EXIT_THRESHOLD:
                    should_exit = True
                    exit_reason = "ML_EXIT"

        if should_exit:
            pnl_usd = capital * position_size_pct * leveraged_pnl_pct
            capital += pnl_usd
            hold_time = i - position['entry_index']

            trades.append({
                'entry_time': position['entry_time'],
                'exit_time': timestamp,
                'side': side,
                'entry_price': entry_price,
                'exit_price': price,
                'position_size_pct': position_size_pct,
                'pnl_pct': leveraged_pnl_pct * 100,
                'pnl_usd': pnl_usd,
                'exit_reason': exit_reason,
                'hold_time': hold_time
            })
            position = None

    # Entry Logic
    if position is None:
        X_long = row[long_entry_features].values.reshape(1, -1)
        X_long_scaled = long_entry_scaler.transform(X_long)
        long_prob = long_entry_model.predict_proba(X_long_scaled)[0][1]

        X_short = row[short_entry_features].values.reshape(1, -1)
        X_short_scaled = short_entry_scaler.transform(X_short)
        short_prob = short_entry_model.predict_proba(X_short_scaled)[0][1]

        if long_prob >= ENTRY_THRESHOLD:
            side = 'LONG'
            entry_prob = long_prob
        elif short_prob >= ENTRY_THRESHOLD:
            side = 'SHORT'
            entry_prob = short_prob
        else:
            continue

        # Dynamic Position Sizing
        if entry_prob < 0.65:
            position_size_pct = 0.20
        elif entry_prob >= 0.85:
            position_size_pct = 0.95
        else:
            position_size_pct = 0.20 + (0.95 - 0.20) * ((entry_prob - 0.65) / (0.85 - 0.65))

        position = {
            'side': side,
            'entry_price': price,
            'entry_time': timestamp,
            'entry_index': i,
            'entry_prob': entry_prob,
            'position_size_pct': position_size_pct
        }

# ==============================================================================
# STEP 4: Calculate Results
# ==============================================================================

print("="*80)
print("BACKTEST RESULTS")
print("="*80)
print()

total_return = ((capital - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100
trade_freq = len(trades) / HOLDOUT_DAYS if trades else 0

print(f"Performance:")
print(f"  Initial Capital: ${INITIAL_CAPITAL:,}")
print(f"  Final Capital: ${capital:,.2f}")
print(f"  Total Return: {total_return:+.2f}%")
print()

print(f"Trading Activity:")
print(f"  Total Trades: {len(trades)}")
print(f"  Trade Frequency: {trade_freq:.2f} trades/day")
print(f"  Target: 3-8 trades/day {'✅' if 3 <= trade_freq <= 8 else '❌'}")
print()

if trades:
    df_trades = pd.DataFrame(trades)

    wins = df_trades[df_trades['pnl_usd'] > 0]
    losses = df_trades[df_trades['pnl_usd'] <= 0]

    print(f"Win/Loss Analysis:")
    print(f"  Wins: {len(wins)} ({len(wins)/len(df_trades)*100:.1f}%)")
    print(f"  Losses: {len(losses)} ({len(losses)/len(df_trades)*100:.1f}%)")
    print(f"  Win Rate: {len(wins)/len(df_trades)*100:.1f}%")
    print()

    print(f"P&L Analysis:")
    print(f"  Average Trade: ${df_trades['pnl_usd'].mean():+.2f} ({df_trades['pnl_pct'].mean():+.2f}%)")
    print(f"  Best Trade: ${df_trades['pnl_usd'].max():+.2f} ({df_trades['pnl_pct'].max():+.2f}%)")
    print(f"  Worst Trade: ${df_trades['pnl_usd'].min():+.2f} ({df_trades['pnl_pct'].min():+.2f}%)")
    print()

    print(f"Direction Split:")
    long_trades = df_trades[df_trades['side'] == 'LONG']
    short_trades = df_trades[df_trades['side'] == 'SHORT']
    print(f"  LONG: {len(long_trades)} ({len(long_trades)/len(df_trades)*100:.1f}%)")
    print(f"  SHORT: {len(short_trades)} ({len(short_trades)/len(df_trades)*100:.1f}%)")
    print()

    print(f"Exit Distribution:")
    for reason in ['ML_EXIT', 'STOP_LOSS', 'MAX_HOLD']:
        count = len(df_trades[df_trades['exit_reason'] == reason])
        pct = count / len(df_trades) * 100
        print(f"  {reason}: {count} ({pct:.1f}%)")
    print()

    print(f"Hold Time:")
    print(f"  Average: {df_trades['hold_time'].mean():.1f} candles ({df_trades['hold_time'].mean()/12:.1f} hours)")
    print(f"  Median: {df_trades['hold_time'].median():.1f} candles ({df_trades['hold_time'].median()/12:.1f} hours)")
    print()

# Calculate Benchmark
market_start = df.iloc[100]['close']
market_end = df.iloc[-EMERGENCY_MAX_HOLD]['close']
market_change = (market_end - market_start) / market_start
leveraged_benchmark = market_change * LEVERAGE * 100

print(f"Benchmark Comparison:")
print(f"  Market Change: {market_change*100:+.2f}%")
print(f"  Leveraged (4x): {leveraged_benchmark:+.2f}%")
print(f"  Strategy Return: {total_return:+.2f}%")
print(f"  Outperformance: {total_return - leveraged_benchmark:+.2f}%")
print()

# Save Results
if trades:
    output_file = RESULTS_DIR / f"backtest_newfeatures_14day_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df_trades.to_csv(output_file, index=False)
    print(f"✅ Results saved: {output_file.name}")
    print()

# Final Assessment
print("="*80)
print("TARGET ASSESSMENT")
print("="*80)
print()

targets_met = []
targets_failed = []

if 3 <= trade_freq <= 8:
    targets_met.append(f"✅ Trade Frequency: {trade_freq:.2f}/day (target: 3-8)")
else:
    targets_failed.append(f"❌ Trade Frequency: {trade_freq:.2f}/day (target: 3-8)")

if total_return > leveraged_benchmark:
    targets_met.append(f"✅ Beat Benchmark: {total_return:+.2f}% vs {leveraged_benchmark:+.2f}%")
else:
    targets_failed.append(f"❌ Beat Benchmark: {total_return:+.2f}% vs {leveraged_benchmark:+.2f}%")

if trades:
    win_rate = len(wins) / len(df_trades) * 100
    if 60 <= win_rate <= 75:
        targets_met.append(f"✅ Win Rate: {win_rate:.1f}% (target: 60-75%)")
    else:
        targets_failed.append(f"⚠️ Win Rate: {win_rate:.1f}% (target: 60-75%)")

    long_pct = len(long_trades) / len(df_trades) * 100
    if 40 <= long_pct <= 60:
        targets_met.append(f"✅ LONG/SHORT Balance: {long_pct:.1f}%/{100-long_pct:.1f}%")
    else:
        targets_failed.append(f"⚠️ LONG/SHORT Balance: {long_pct:.1f}%/{100-long_pct:.1f}%")

print("Targets Met:")
for target in targets_met:
    print(f"  {target}")
print()

if targets_failed:
    print("Targets Not Met:")
    for target in targets_failed:
        print(f"  {target}")
    print()

success_rate = len(targets_met) / (len(targets_met) + len(targets_failed)) * 100
print(f"Overall Success: {len(targets_met)}/{len(targets_met)+len(targets_failed)} ({success_rate:.0f}%)")
print()
