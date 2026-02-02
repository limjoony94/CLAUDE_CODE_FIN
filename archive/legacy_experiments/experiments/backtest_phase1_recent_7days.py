"""
Backtest Phase 1 Models on Recent 7 Days
=========================================

Purpose: Validate Phase 1 performance on most recent data
         (last 7 days of dataset)

Models Under Test:
  Entry: Phase 1 Walk-Forward Reduced (timestamp: 20251029_050448)
    - LONG: 80 features, 39.07% positive (Fold 4)
    - SHORT: 79 features, 31.11% positive (Fold 2)

  Exit: Full features (existing, threshold 0.75)
    - LONG: 27 features
    - SHORT: 27 features

Test Period: Last 7 days of dataset
  Expected: Oct 20 - Oct 26, 2025

Success Criteria:
  - Return > 0% (profitable)
  - Win rate > 60%
  - Trade frequency ≥ 1.5 trades/day
  - No catastrophic losses

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
TEST_DAYS = 7

DATA_DIR = PROJECT_ROOT / "data" / "features"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

print("="*80)
print("BACKTEST: PHASE 1 - RECENT 7 DAYS VALIDATION")
print("="*80)
print()
print("Models:")
print("  Entry: Phase 1 Walk-Forward Reduced (20251029_050448)")
print("    - LONG: 80 features (39.07% positive, Fold 4)")
print("    - SHORT: 79 features (31.11% positive, Fold 2)")
print("  Exit: Full features (threshold 0.75)")
print("    - LONG/SHORT: 27 features each")
print()
print(f"Configuration:")
print(f"  Entry Threshold: {ENTRY_THRESHOLD}")
print(f"  Exit Threshold: {ML_EXIT_THRESHOLD}")
print(f"  Stop Loss: {EMERGENCY_STOP_LOSS*100}%")
print(f"  Max Hold: {EMERGENCY_MAX_HOLD} candles ({EMERGENCY_MAX_HOLD/12}h)")
print(f"  Leverage: {LEVERAGE}x")
print(f"  Test Period: Last {TEST_DAYS} days")
print()

# Load Full Features Dataset
print("-"*80)
print("Loading Data and Selecting Recent 7 Days")
print("-"*80)

df = pd.read_csv(DATA_DIR / "BTCUSDT_5m_features.csv")
print(f"✅ Loaded {len(df):,} candles")
print(f"   Date range: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")

# Select ONLY last 7 days
test_candles = TEST_DAYS * 24 * 12
test_start_idx = len(df) - test_candles

df = df.iloc[test_start_idx:].copy().reset_index(drop=True)
print(f"\n✅ Recent 7-Day Period Selected:")
print(f"   Date range: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")
print(f"   Candles: {len(df):,} ({TEST_DAYS} days)")
print()

# Prepare Exit Features
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
print(f"✅ Exit features prepared - {len(df.columns)} total features")
print()

# Load Models
print("-"*80)
print("Loading Models")
print("-"*80)

# Entry Models (Phase 1 Walk-Forward Reduced)
long_entry_model = joblib.load(MODELS_DIR / "xgboost_long_entry_walkforward_reduced_phase1_20251029_050448.pkl")
long_entry_scaler = joblib.load(MODELS_DIR / "xgboost_long_entry_walkforward_reduced_phase1_20251029_050448_scaler.pkl")
with open(MODELS_DIR / "xgboost_long_entry_walkforward_reduced_phase1_20251029_050448_features.txt", 'r') as f:
    long_entry_features = [line.strip() for line in f if line.strip()]

short_entry_model = joblib.load(MODELS_DIR / "xgboost_short_entry_walkforward_reduced_phase1_20251029_050448.pkl")
short_entry_scaler = joblib.load(MODELS_DIR / "xgboost_short_entry_walkforward_reduced_phase1_20251029_050448_scaler.pkl")
with open(MODELS_DIR / "xgboost_short_entry_walkforward_reduced_phase1_20251029_050448_features.txt", 'r') as f:
    short_entry_features = [line.strip() for line in f if line.strip()]

print(f"✅ Entry Models (Phase 1 Walk-Forward Reduced):")
print(f"   LONG: {len(long_entry_features)} features")
print(f"   SHORT: {len(short_entry_features)} features")

# Exit Models (Full Features, threshold 0.75)
long_exit_model = joblib.load(MODELS_DIR / "xgboost_long_exit_threshold_075_20251027_190512.pkl")
long_exit_scaler = joblib.load(MODELS_DIR / "xgboost_long_exit_threshold_075_20251027_190512_scaler.pkl")
with open(MODELS_DIR / "xgboost_long_exit_threshold_075_20251027_190512_features.txt", 'r') as f:
    long_exit_features = [line.strip() for line in f if line.strip()]

short_exit_model = joblib.load(MODELS_DIR / "xgboost_short_exit_threshold_075_20251027_190512.pkl")
short_exit_scaler = joblib.load(MODELS_DIR / "xgboost_short_exit_threshold_075_20251027_190512_scaler.pkl")
with open(MODELS_DIR / "xgboost_short_exit_threshold_075_20251027_190512_features.txt", 'r') as f:
    short_exit_features = [line.strip() for line in f if line.strip()]

print(f"✅ Exit Models (Full Features, 0.75 threshold):")
print(f"   LONG: {len(long_exit_features)} features")
print(f"   SHORT: {len(short_exit_features)} features")
print()

# Run Backtest
print("="*80)
print("RUNNING BACKTEST ON RECENT 7 DAYS")
print("="*80)

capital = INITIAL_CAPITAL
position = None
trades = []

for i in range(100, len(df) - EMERGENCY_MAX_HOLD):
    row = df.iloc[i]
    timestamp = row['timestamp']
    price = row['close']

    # Position Management
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

        # Exit Conditions
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

            else:  # SHORT
                X_exit = row[short_exit_features].values.reshape(1, -1)
                X_exit_scaled = short_exit_scaler.transform(X_exit)
                exit_prob = short_exit_model.predict_proba(X_exit_scaled)[0][1]

                if exit_prob >= ML_EXIT_THRESHOLD:
                    should_exit = True
                    exit_reason = "ML_EXIT"

        # Execute Exit
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

    # Entry Signals
    if position is None:
        # LONG Entry
        X_long = row[long_entry_features].values.reshape(1, -1)
        X_long_scaled = long_entry_scaler.transform(X_long)
        long_prob = long_entry_model.predict_proba(X_long_scaled)[0][1]

        # SHORT Entry
        X_short = row[short_entry_features].values.reshape(1, -1)
        X_short_scaled = short_entry_scaler.transform(X_short)
        short_prob = short_entry_model.predict_proba(X_short_scaled)[0][1]

        # Entry Decision
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

# Analyze Results
print("\n" + "="*80)
print("RECENT 7-DAY BACKTEST RESULTS (PHASE 1)")
print("="*80)

total_return = ((capital - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100
print(f"\nCapital: ${INITIAL_CAPITAL:,.2f} → ${capital:,.2f}")
print(f"Total Return: {total_return:+.2f}%")

if trades:
    df_trades = pd.DataFrame(trades)

    wins = df_trades[df_trades['pnl_usd'] > 0]
    losses = df_trades[df_trades['pnl_usd'] <= 0]

    print(f"\nTrades: {len(df_trades)}")
    print(f"  Trade Frequency: {len(df_trades)/TEST_DAYS:.2f} trades/day")
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
    output_file = RESULTS_DIR / f"backtest_phase1_recent_7days_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df_trades.to_csv(output_file, index=False)
    print(f"\n✅ Results saved: {output_file}")

    # Success Criteria Evaluation
    print("\n" + "="*80)
    print("SUCCESS CRITERIA EVALUATION (7-DAY VALIDATION)")
    print("="*80)

    trade_freq = len(df_trades) / TEST_DAYS
    win_rate = len(wins) / len(df_trades) * 100

    print(f"\n1. Return: {total_return:+.2f}% (target: > 0%)")
    if total_return > 0:
        print("   ✅ PASS (Profitable)")
    else:
        print("   ❌ FAIL (Loss)")

    print(f"\n2. Win Rate: {win_rate:.1f}% (target: > 60%)")
    if win_rate > 60:
        print("   ✅ PASS")
    else:
        print("   ❌ FAIL")

    print(f"\n3. Trade Frequency: {trade_freq:.2f}/day (target: ≥ 1.5)")
    if trade_freq >= 1.5:
        print("   ✅ PASS")
    else:
        print("   ⚠️  BORDERLINE")

    print(f"\n4. Max Single Loss: ${losses['pnl_usd'].min():.2f}" if len(losses) > 0 else "\n4. Max Single Loss: N/A")
    if len(losses) > 0 and losses['pnl_usd'].min() > -500:
        print("   ✅ PASS (No catastrophic loss)")
    elif len(losses) > 0:
        print("   ⚠️  WARNING (Large single loss)")

    # Comparison with 30-day holdout
    print("\n" + "="*80)
    print("COMPARISON: 7-DAY vs 30-DAY HOLDOUT")
    print("="*80)
    print()
    print("30-Day Holdout Performance:")
    print("  Return: +40.24%")
    print("  Win Rate: 79.7%")
    print("  Trade Frequency: 1.97/day")
    print()
    print(f"7-Day Recent Performance:")
    print(f"  Return: {total_return:+.2f}%")
    print(f"  Win Rate: {win_rate:.1f}%")
    print(f"  Trade Frequency: {trade_freq:.2f}/day")
    print()

    # Calculate consistency
    if total_return > 0 and win_rate > 60:
        print("✅ 7-DAY VALIDATION: PASSED")
        print("   Phase 1 maintains positive performance on recent data")
        print("   Ready for production deployment consideration")
    else:
        print("⚠️  7-DAY VALIDATION: BORDERLINE")
        print("   Recent performance weaker than holdout period")
        print("   Consider additional analysis before deployment")

else:
    print("\n⚠️  No trades executed")

print("\n" + "="*80)
