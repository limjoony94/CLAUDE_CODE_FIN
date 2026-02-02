"""
Backtest with Production Settings - Full Data Period
====================================================

Uses exact production configuration:
- Entry Models: xgboost_long_entry_enhanced_20251024_012445.pkl
- Exit Models: xgboost_long_exit_oppgating_improved_20251024_043527.pkl
- LONG Threshold: 0.65
- SHORT Threshold: 0.70
- ML Exit Threshold: 0.80 (LONG/SHORT)
- Stop Loss: -3%
- Max Hold: 120 candles (10h)
- Leverage: 4x

Date Range: Full available data (2025-07-01 to 2025-10-23, 114 days)
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import pickle
import joblib

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiments.calculate_all_features_enhanced_v2 import calculate_all_features_enhanced_v2
from scripts.experiments.retrain_exit_models_opportunity_gating import prepare_exit_features

# =============================================================================
# PRODUCTION CONFIGURATION
# =============================================================================

# Entry Thresholds
LONG_THRESHOLD = 0.65
SHORT_THRESHOLD = 0.70
GATE_THRESHOLD = 0.001

# Exit Parameters
ML_EXIT_THRESHOLD_LONG = 0.80
ML_EXIT_THRESHOLD_SHORT = 0.80
EMERGENCY_STOP_LOSS = 0.03  # -3%
EMERGENCY_MAX_HOLD_TIME = 120  # 10 hours

# Expected Values
LONG_AVG_RETURN = 0.0041
SHORT_AVG_RETURN = 0.0047

# Trading
LEVERAGE = 4
INITIAL_CAPITAL = 10000

# Model Paths (PRODUCTION)
MODELS_DIR = PROJECT_ROOT / "models"
LONG_ENTRY_MODEL = MODELS_DIR / "xgboost_long_entry_enhanced_20251024_012445.pkl"
LONG_ENTRY_SCALER = MODELS_DIR / "xgboost_long_entry_enhanced_20251024_012445_scaler.pkl"
SHORT_ENTRY_MODEL = MODELS_DIR / "xgboost_short_entry_enhanced_20251024_012445.pkl"
SHORT_ENTRY_SCALER = MODELS_DIR / "xgboost_short_entry_enhanced_20251024_012445_scaler.pkl"
LONG_EXIT_MODEL = MODELS_DIR / "xgboost_long_exit_oppgating_improved_20251024_043527.pkl"
LONG_EXIT_SCALER = MODELS_DIR / "xgboost_long_exit_oppgating_improved_20251024_043527_scaler.pkl"
SHORT_EXIT_MODEL = MODELS_DIR / "xgboost_short_exit_oppgating_improved_20251024_044510.pkl"
SHORT_EXIT_SCALER = MODELS_DIR / "xgboost_short_exit_oppgating_improved_20251024_044510_scaler.pkl"

print("=" * 80)
print("BACKTEST: PRODUCTION SETTINGS - FULL DATA PERIOD")
print("=" * 80)
print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"\nConfiguration:")
print(f"  Entry: LONG={LONG_THRESHOLD}, SHORT={SHORT_THRESHOLD}, Gate={GATE_THRESHOLD}")
print(f"  Exit: ML={ML_EXIT_THRESHOLD_LONG}, SL=-{EMERGENCY_STOP_LOSS*100}%, MaxHold={EMERGENCY_MAX_HOLD_TIME/12}h")
print(f"  Leverage: {LEVERAGE}x")
print("=" * 80)

# =============================================================================
# LOAD MODELS
# =============================================================================

print("\nLoading models...")
long_entry_model = joblib.load(LONG_ENTRY_MODEL)
long_entry_scaler = joblib.load(LONG_ENTRY_SCALER)
short_entry_model = joblib.load(SHORT_ENTRY_MODEL)
short_entry_scaler = joblib.load(SHORT_ENTRY_SCALER)
long_exit_model = joblib.load(LONG_EXIT_MODEL)
long_exit_scaler = joblib.load(LONG_EXIT_SCALER)
short_exit_model = joblib.load(SHORT_EXIT_MODEL)
short_exit_scaler = joblib.load(SHORT_EXIT_SCALER)

# Load feature lists
long_entry_features_path = MODELS_DIR / "xgboost_long_entry_enhanced_20251024_012445_features.txt"
with open(long_entry_features_path, 'r') as f:
    long_entry_features = [line.strip() for line in f.readlines()]

short_entry_features_path = MODELS_DIR / "xgboost_short_entry_enhanced_20251024_012445_features.txt"
with open(short_entry_features_path, 'r') as f:
    short_entry_features = [line.strip() for line in f.readlines() if line.strip()]

long_exit_features_path = MODELS_DIR / "xgboost_long_exit_oppgating_improved_20251024_043527_features.txt"
with open(long_exit_features_path, 'r') as f:
    long_exit_features = [line.strip() for line in f.readlines()]

short_exit_features_path = MODELS_DIR / "xgboost_short_exit_oppgating_improved_20251024_044510_features.txt"
with open(short_exit_features_path, 'r') as f:
    short_exit_features = [line.strip() for line in f.readlines()]

print("✅ All models loaded")
print(f"  LONG Entry features: {len(long_entry_features)}")
print(f"  SHORT Entry features: {len(short_entry_features)}")
print(f"  LONG Exit features: {len(long_exit_features)}")
print(f"  SHORT Exit features: {len(short_exit_features)}")

# =============================================================================
# LOAD DATA (FULL PERIOD)
# =============================================================================

print("\nLoading data (full period)...")
data_path = PROJECT_ROOT / "data" / "historical" / "BTCUSDT_5m_updated.csv"
df = pd.read_csv(data_path)
df['timestamp'] = pd.to_datetime(df['timestamp'])

print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
print(f"  Total days: {(df['timestamp'].max() - df['timestamp'].min()).days} days")
print(f"  Total candles: {len(df):,}")

# =============================================================================
# CALCULATE FEATURES
# =============================================================================

print("\nCalculating features...")
df_features = calculate_all_features_enhanced_v2(df)
print(f"✅ Features calculated ({len(df_features)} rows)")

# Prepare exit features (add market context features)
print("\nPreparing exit features...")
df_features = prepare_exit_features(df_features)
print(f"✅ Exit features prepared ({len(df_features)} rows)")

# =============================================================================
# BACKTEST LOGIC
# =============================================================================

def dynamic_position_size(prob, min_size=0.20, max_size=0.95):
    """Dynamic position sizing based on signal strength"""
    if prob < 0.65:
        return min_size
    elif prob >= 0.85:
        return max_size
    else:
        # Linear interpolation
        return min_size + (max_size - min_size) * ((prob - 0.65) / (0.85 - 0.65))

def run_backtest():
    """Run backtest with production settings"""

    capital = INITIAL_CAPITAL
    position = None
    trades = []

    for i in range(len(df_features)):
        row = df_features.iloc[i]
        timestamp = row['timestamp']
        price = row['close']

        # === POSITION MANAGEMENT ===
        if position is not None:
            side = position['side']
            entry_price = position['entry_price']
            entry_time = position['entry_time']
            position_size_pct = position['position_size_pct']

            # Calculate P&L
            if side == 'LONG':
                price_change_pct = (price - entry_price) / entry_price
            else:  # SHORT
                price_change_pct = (entry_price - price) / entry_price

            leveraged_pnl_pct = price_change_pct * LEVERAGE

            # Check exit conditions
            should_exit = False
            exit_reason = None

            # 1. Emergency Stop Loss (-3% balance)
            if leveraged_pnl_pct <= -EMERGENCY_STOP_LOSS:
                should_exit = True
                exit_reason = "STOP_LOSS"

            # 2. Emergency Max Hold (10 hours = 120 candles)
            elif i - position['entry_index'] >= EMERGENCY_MAX_HOLD_TIME:
                should_exit = True
                exit_reason = "MAX_HOLD"

            # 3. ML Exit
            else:
                # Get current features for exit prediction
                if side == 'LONG':
                    X_exit = row[long_exit_features].values.reshape(1, -1)
                    X_exit_scaled = long_exit_scaler.transform(X_exit)
                    exit_prob = long_exit_model.predict_proba(X_exit_scaled)[0][1]

                    if exit_prob >= ML_EXIT_THRESHOLD_LONG:
                        should_exit = True
                        exit_reason = "ML_EXIT"

                else:  # SHORT
                    X_exit = row[short_exit_features].values.reshape(1, -1)
                    X_exit_scaled = short_exit_scaler.transform(X_exit)
                    exit_prob = short_exit_model.predict_proba(X_exit_scaled)[0][1]

                    if exit_prob >= ML_EXIT_THRESHOLD_SHORT:
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

        # === ENTRY SIGNALS ===
        if position is None:
            # Get entry probabilities
            X_long = row[long_entry_features].values.reshape(1, -1)
            X_long_scaled = long_entry_scaler.transform(X_long)
            long_prob = long_entry_model.predict_proba(X_long_scaled)[0][1]

            X_short = row[short_entry_features].values.reshape(1, -1)
            X_short_scaled = short_entry_scaler.transform(X_short)
            short_prob = short_entry_model.predict_proba(X_short_scaled)[0][1]

            # Opportunity cost gating
            long_ev = long_prob * LONG_AVG_RETURN
            short_ev = short_prob * SHORT_AVG_RETURN
            opportunity_cost = short_ev - long_ev

            # Entry logic
            if long_prob >= LONG_THRESHOLD:
                side = 'LONG'
                entry_prob = long_prob
            elif short_prob >= SHORT_THRESHOLD and opportunity_cost > GATE_THRESHOLD:
                side = 'SHORT'
                entry_prob = short_prob
            else:
                continue

            # Dynamic position sizing
            position_size_pct = dynamic_position_size(entry_prob)

            position = {
                'side': side,
                'entry_price': price,
                'entry_time': timestamp,
                'entry_index': i,
                'entry_prob': entry_prob,
                'position_size_pct': position_size_pct
            }

    return capital, trades

# =============================================================================
# RUN BACKTEST
# =============================================================================

print("\nRunning backtest...")
final_capital, trades = run_backtest()

# =============================================================================
# ANALYZE RESULTS
# =============================================================================

print("\n" + "=" * 80)
print("BACKTEST RESULTS")
print("=" * 80)

total_return = ((final_capital - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100
print(f"\nCapital: ${INITIAL_CAPITAL:,.2f} → ${final_capital:,.2f}")
print(f"Total Return: {total_return:+.2f}%")

# Calculate period metrics
start_date = df['timestamp'].min()
end_date = df['timestamp'].max()
total_days = (end_date - start_date).days
print(f"\nPeriod: {start_date.date()} to {end_date.date()} ({total_days} days)")
print(f"Daily Return: {total_return/total_days:.2f}%")
print(f"Monthly Return (30d): {total_return/total_days*30:.2f}%")

if trades:
    df_trades = pd.DataFrame(trades)

    wins = df_trades[df_trades['pnl_usd'] > 0]
    losses = df_trades[df_trades['pnl_usd'] <= 0]

    print(f"\nTrades: {len(df_trades)}")
    print(f"  Trades/Day: {len(df_trades)/total_days:.2f}")
    print(f"  LONG: {len(df_trades[df_trades['side'] == 'LONG'])} ({len(df_trades[df_trades['side'] == 'LONG'])/len(df_trades)*100:.1f}%)")
    print(f"  SHORT: {len(df_trades[df_trades['side'] == 'SHORT'])} ({len(df_trades[df_trades['side'] == 'SHORT'])/len(df_trades)*100:.1f}%)")

    print(f"\nWin Rate: {len(wins)/len(df_trades)*100:.1f}% ({len(wins)}W / {len(losses)}L)")
    print(f"Avg Win: ${wins['pnl_usd'].mean():.2f}" if len(wins) > 0 else "Avg Win: N/A")
    print(f"Avg Loss: ${losses['pnl_usd'].mean():.2f}" if len(losses) > 0 else "Avg Loss: N/A")
    print(f"Avg Hold: {df_trades['hold_time'].mean():.1f} candles ({df_trades['hold_time'].mean()/12:.1f}h)")

    # Profit factor
    if len(losses) > 0 and losses['pnl_usd'].sum() != 0:
        profit_factor = abs(wins['pnl_usd'].sum() / losses['pnl_usd'].sum())
        print(f"Profit Factor: {profit_factor:.2f}x")

    print(f"\nExit Reasons:")
    for reason, count in df_trades['exit_reason'].value_counts().items():
        print(f"  {reason}: {count} ({count/len(df_trades)*100:.1f}%)")

    # Risk metrics
    print(f"\nRisk Metrics:")
    cumulative = df_trades['pnl_usd'].cumsum()
    running_peak = cumulative.cummax()
    drawdown = cumulative - running_peak
    max_drawdown = (drawdown.min() / INITIAL_CAPITAL) * 100
    print(f"  Max Drawdown: {max_drawdown:.2f}%")
    print(f"  Best Trade: ${df_trades['pnl_usd'].max():.2f} ({df_trades['pnl_pct'].max():.2f}%)")
    print(f"  Worst Trade: ${df_trades['pnl_usd'].min():.2f} ({df_trades['pnl_pct'].min():.2f}%)")

    # Save results
    output_file = PROJECT_ROOT / "results" / f"backtest_production_full_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    df_trades.to_csv(output_file, index=False)
    print(f"\n✅ Results saved: {output_file}")

else:
    print("\n⚠️  No trades executed")

print("\n" + "=" * 80)
