"""
Backtest RETRAINED Models with Threshold 0.70
==============================================

Models Under Test:
  Entry: RETRAINED with Latest Data (timestamp: 20251029_081454)
    - LONG: 80 features
    - SHORT: 79 features

Configuration:
  Entry Threshold: 0.70 (LOWERED from 0.75)
  Exit Threshold: 0.75
  Stop Loss: -3%
  Max Hold: 120 candles (10h)
  Leverage: 4x

Test Period: 14-day Holdout (Oct 13-26)

Expected Performance:
  Trade Frequency: 4.00/day (56 trades in 14 days)
  LONG: 51 samples (3.64/day)
  SHORT: 5 samples (0.36/day)

Success Criteria:
  - Trade frequency ≥ 4.0/day
  - Win rate 60-75%
  - Return > 0%
  - Reasonable LONG/SHORT balance

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
ENTRY_THRESHOLD = 0.70  # LOWERED from 0.75
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
print("BACKTEST: RETRAINED MODELS - THRESHOLD 0.70")
print("="*80)
print()
print("Models: RETRAINED (20251029_081454)")
print("Configuration:")
print(f"  Entry Threshold: {ENTRY_THRESHOLD} (LOWERED from 0.75)")
print(f"  Exit Threshold: {ML_EXIT_THRESHOLD}")
print(f"  Stop Loss: {EMERGENCY_STOP_LOSS*100}%")
print(f"  Max Hold: {EMERGENCY_MAX_HOLD} candles ({EMERGENCY_MAX_HOLD/12}h)")
print(f"  Leverage: {LEVERAGE}x")
print()
print("Expected Performance:")
print("  Trade Frequency: 4.00/day (56 trades)")
print("  LONG: 51 samples (3.64/day)")
print("  SHORT: 5 samples (0.36/day)")
print()

# Load data
print("-"*80)
print("Loading Data")
print("-"*80)

df = pd.read_csv(DATA_DIR / "BTCUSDT_5m_features.csv")
print(f"✅ Loaded {len(df):,} candles")

# Select holdout period
holdout_candles = HOLDOUT_DAYS * 24 * 12
holdout_start_idx = len(df) - holdout_candles
df = df.iloc[holdout_start_idx:].copy().reset_index(drop=True)

print(f"✅ Holdout period: {len(df):,} candles ({HOLDOUT_DAYS} days)")
print(f"   {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")
print()

# Prepare exit features
def prepare_exit_features(df):
    """Prepare EXIT features with enhanced market context"""
    df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
    df['volume_surge'] = (df['volume'] > df['volume'].rolling(20).mean() * 1.5).astype(float)

    if 'ma_50' in df.columns:
        df['price_vs_ma20'] = (df['close'] - df['ma_50']) / df['ma_50']
    else:
        ma_20 = df['close'].rolling(20).mean()
        df['price_vs_ma20'] = (df['close'] - ma_20) / ma_20

    if 'ma_200' in df.columns:
        df['price_vs_ma50'] = (df['close'] - df['ma_200']) / df['ma_200']
    else:
        ma_50 = df['close'].rolling(50).mean()
        df['price_vs_ma50'] = (df['close'] - ma_50) / ma_50

    df['returns'] = df['close'].pct_change()
    df['volatility_20'] = df['returns'].rolling(20).std()
    df['volatility_regime'] = (df['volatility_20'] > df['volatility_20'].rolling(100).median()).astype(float)
    df['rsi_slope'] = df['rsi'].diff(3) / 3
    df['rsi_overbought'] = (df['rsi'] > 70).astype(float)
    df['rsi_oversold'] = (df['rsi'] < 30).astype(float)
    df['rsi_divergence'] = 0
    df['macd_histogram_slope'] = df['macd_hist'].diff(3) / 3 if 'macd_hist' in df.columns else 0
    df['macd_crossover'] = ((df['macd'] > df['macd_signal']) &
                            (df['macd'].shift(1) <= df['macd_signal'].shift(1))).astype(float)
    df['macd_crossunder'] = ((df['macd'] < df['macd_signal']) &
                             (df['macd'].shift(1) >= df['macd_signal'].shift(1))).astype(float)
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
print(f"✅ Exit features prepared")
print()

# Load models
print("-"*80)
print("Loading Models")
print("-"*80)

# Entry Models (RETRAINED)
long_entry_model = joblib.load(MODELS_DIR / "xgboost_long_entry_retrained_latest_20251029_081454.pkl")
long_entry_scaler = joblib.load(MODELS_DIR / "xgboost_long_entry_retrained_latest_20251029_081454_scaler.pkl")
with open(MODELS_DIR / "xgboost_long_entry_retrained_latest_20251029_081454_features.txt", 'r') as f:
    long_entry_features = [line.strip() for line in f if line.strip()]

short_entry_model = joblib.load(MODELS_DIR / "xgboost_short_entry_retrained_latest_20251029_081454.pkl")
short_entry_scaler = joblib.load(MODELS_DIR / "xgboost_short_entry_retrained_latest_20251029_081454_scaler.pkl")
with open(MODELS_DIR / "xgboost_short_entry_retrained_latest_20251029_081454_features.txt", 'r') as f:
    short_entry_features = [line.strip() for line in f if line.strip()]

print(f"✅ Entry Models: LONG {len(long_entry_features)}, SHORT {len(short_entry_features)} features")

# Exit Models
long_exit_model = joblib.load(MODELS_DIR / "xgboost_long_exit_threshold_075_20251027_190512.pkl")
long_exit_scaler = joblib.load(MODELS_DIR / "xgboost_long_exit_threshold_075_20251027_190512_scaler.pkl")
with open(MODELS_DIR / "xgboost_long_exit_threshold_075_20251027_190512_features.txt", 'r') as f:
    long_exit_features = [line.strip() for line in f if line.strip()]

short_exit_model = joblib.load(MODELS_DIR / "xgboost_short_exit_threshold_075_20251027_190512.pkl")
short_exit_scaler = joblib.load(MODELS_DIR / "xgboost_short_exit_threshold_075_20251027_190512_scaler.pkl")
with open(MODELS_DIR / "xgboost_short_exit_threshold_075_20251027_190512_features.txt", 'r') as f:
    short_exit_features = [line.strip() for line in f if line.strip()]

print(f"✅ Exit Models: LONG {len(long_exit_features)}, SHORT {len(short_exit_features)} features")
print()

# Run backtest
print("="*80)
print("RUNNING BACKTEST")
print("="*80)

capital = INITIAL_CAPITAL
position = None
trades = []

for i in range(100, len(df) - EMERGENCY_MAX_HOLD):
    row = df.iloc[i]
    timestamp = row['timestamp']
    price = row['close']

    # Position management
    if position is not None:
        side = position['side']
        entry_price = position['entry_price']
        entry_time = position['entry_time']
        position_size_pct = position['position_size_pct']

        # Calculate P&L
        if side == 'LONG':
            price_change_pct = (price - entry_price) / entry_price
        else:
            price_change_pct = (entry_price - price) / entry_price

        leveraged_pnl_pct = price_change_pct * LEVERAGE

        # Exit conditions
        should_exit = False
        exit_reason = None

        # Emergency stop loss
        if leveraged_pnl_pct <= EMERGENCY_STOP_LOSS:
            should_exit = True
            exit_reason = "STOP_LOSS"

        # Emergency max hold
        elif i - position['entry_index'] >= EMERGENCY_MAX_HOLD:
            should_exit = True
            exit_reason = "MAX_HOLD"

        # ML exit
        else:
            if side == 'LONG':
                X_exit = row[long_exit_features].values.reshape(1, -1)
                X_exit_scaled = long_exit_scaler.transform(X_exit)
                exit_prob = long_exit_model.predict_proba(X_exit_scaled)[0][1]

                if exit_prob >= ML_EXIT_THRESHOLD:
                    should_exit = True
                    exit_reason = "ML_EXIT"

            else:  # SHORT
                X_exit = row[short_exit_features].values.reshape(1, -1)
                X_exit_scaled = short_exit_scaler.transform(X_exit)
                exit_prob = short_exit_model.predict_proba(X_exit_scaled)[0][1]

                if exit_prob >= ML_EXIT_THRESHOLD:
                    should_exit = True
                    exit_reason = "ML_EXIT"

        # Execute exit
        if should_exit:
            pnl_usd = capital * position_size_pct * leveraged_pnl_pct
            capital += pnl_usd

            hold_time = i - position['entry_index']

            trades.append({
                'entry_time': entry_time,
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

    # Entry signals
    if position is None:
        # LONG entry
        X_long = row[long_entry_features].values.reshape(1, -1)
        X_long_scaled = long_entry_scaler.transform(X_long)
        long_prob = long_entry_model.predict_proba(X_long_scaled)[0][1]

        # SHORT entry
        X_short = row[short_entry_features].values.reshape(1, -1)
        X_short_scaled = short_entry_scaler.transform(X_short)
        short_prob = short_entry_model.predict_proba(X_short_scaled)[0][1]

        # Entry decision
        if long_prob >= ENTRY_THRESHOLD:
            side = 'LONG'
            entry_prob = long_prob
        elif short_prob >= ENTRY_THRESHOLD:
            side = 'SHORT'
            entry_prob = short_prob
        else:
            continue

        # Dynamic position sizing
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

# Results
print("\n" + "="*80)
print("BACKTEST RESULTS - THRESHOLD 0.70")
print("="*80)

total_return = ((capital - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100
print(f"\nCapital: ${INITIAL_CAPITAL:,.2f} → ${capital:,.2f}")
print(f"Total Return: {total_return:+.2f}%")

if trades:
    df_trades = pd.DataFrame(trades)

    wins = df_trades[df_trades['pnl_usd'] > 0]
    losses = df_trades[df_trades['pnl_usd'] <= 0]

    print(f"\nTrades: {len(df_trades)}")
    print(f"  Trade Frequency: {len(df_trades)/HOLDOUT_DAYS:.2f} trades/day")
    print(f"  LONG: {len(df_trades[df_trades['side'] == 'LONG'])} ({len(df_trades[df_trades['side'] == 'LONG'])/len(df_trades)*100:.1f}%)")
    print(f"  SHORT: {len(df_trades[df_trades['side'] == 'SHORT'])} ({len(df_trades[df_trades['side'] == 'SHORT'])/len(df_trades)*100:.1f}%)")

    print(f"\nWin Rate: {len(wins)/len(df_trades)*100:.1f}% ({len(wins)}W / {len(losses)}L)")
    print(f"Avg Win: ${wins['pnl_usd'].mean():.2f}" if len(wins) > 0 else "Avg Win: N/A")
    print(f"Avg Loss: ${losses['pnl_usd'].mean():.2f}" if len(losses) > 0 else "Avg Loss: N/A")
    print(f"Avg Hold: {df_trades['hold_time'].mean():.1f} candles ({df_trades['hold_time'].mean()/12:.1f}h)")

    print(f"\nExit Reasons:")
    for reason, count in df_trades['exit_reason'].value_counts().items():
        print(f"  {reason}: {count} ({count/len(df_trades)*100:.1f}%)")

    # Save results
    output_file = RESULTS_DIR / f"backtest_retrained_threshold_070_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df_trades.to_csv(output_file, index=False)
    print(f"\n✅ Results saved: {output_file}")

    # Success criteria
    print("\n" + "="*80)
    print("SUCCESS CRITERIA EVALUATION")
    print("="*80)

    trade_freq = len(df_trades) / HOLDOUT_DAYS
    win_rate = len(wins) / len(df_trades) * 100

    print(f"\n1. Trade Frequency: {trade_freq:.2f}/day (target: ≥4.0)")
    if trade_freq >= 4.0:
        print("   ✅ PASS")
    else:
        print(f"   ❌ FAIL (-{4.0-trade_freq:.2f} trades/day short)")

    print(f"\n2. Win Rate: {win_rate:.1f}% (target: 60-75%)")
    if 60 <= win_rate <= 75:
        print("   ✅ PASS")
    elif win_rate > 50:
        print(f"   ⚠️  BORDERLINE (acceptable but outside optimal range)")
    else:
        print(f"   ❌ FAIL (< 50%)")

    print(f"\n3. Return: {total_return:+.2f}% (target: > 0%)")
    if total_return > 0:
        print("   ✅ PASS")
    else:
        print("   ❌ FAIL")

    print(f"\n4. ML Exit Usage: {(df_trades['exit_reason'] == 'ML_EXIT').sum()/len(df_trades)*100:.1f}%")
    if (df_trades['exit_reason'] == 'ML_EXIT').sum() / len(df_trades) >= 0.70:
        print("   ✅ PASS (primary exit mechanism)")
    else:
        print("   ⚠️  BORDERLINE (< 70% ML exit)")

    # Comparison with threshold 0.75
    print("\n" + "="*80)
    print("COMPARISON: THRESHOLD 0.70 vs 0.75")
    print("="*80)
    print()
    print("Threshold 0.75 (Previous):")
    print("  Trade Frequency: 0.43/day ❌")
    print("  Total Trades: 6")
    print("  Win Rate: 50.0% ⚠️")
    print("  Return: +2.58%")
    print()
    print(f"Threshold 0.70 (Current):")
    print(f"  Trade Frequency: {trade_freq:.2f}/day", "✅" if trade_freq >= 4.0 else "❌")
    print(f"  Total Trades: {len(df_trades)}")
    print(f"  Win Rate: {win_rate:.1f}%", "✅" if 60 <= win_rate <= 75 else "⚠️")
    print(f"  Return: {total_return:+.2f}%")
    print()
    improvement = (trade_freq / 0.43 - 1) * 100
    print(f"Improvement: {improvement:+.1f}% trade frequency")

else:
    print("\n⚠️  No trades executed")

print("\n" + "="*80)
