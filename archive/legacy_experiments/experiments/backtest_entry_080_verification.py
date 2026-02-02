"""
Backtest Entry 0.80 Verification - Recent Data
===============================================

Verify Entry threshold 0.80 performance with current market data

Configuration:
- Entry: LONG=0.80, SHORT=0.80
- Exit: ML=0.80
- Stop Loss: -3%
- Max Hold: 120 candles (10h)
- Leverage: 4x
- Period: Last 3-5 days
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import joblib

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiments.calculate_all_features_enhanced_v2 import calculate_all_features_enhanced_v2
from scripts.experiments.retrain_exit_models_opportunity_gating import prepare_exit_features

# =============================================================================
# CONFIGURATION - ENTRY 0.80
# =============================================================================

LONG_THRESHOLD = 0.80  # ← TESTING THIS
SHORT_THRESHOLD = 0.80  # ← TESTING THIS
GATE_THRESHOLD = 0.001

ML_EXIT_THRESHOLD_LONG = 0.80
ML_EXIT_THRESHOLD_SHORT = 0.80
EMERGENCY_STOP_LOSS = 0.03
EMERGENCY_MAX_HOLD_TIME = 120

LONG_AVG_RETURN = 0.0041
SHORT_AVG_RETURN = 0.0047

LEVERAGE = 4
INITIAL_CAPITAL = 10000

# Model Paths
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
print("ENTRY 0.80 VERIFICATION BACKTEST - RECENT DATA")
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
    long_exit_features = [line.strip() for line in f.readlines() if line.strip()]

short_exit_features_path = MODELS_DIR / "xgboost_short_exit_oppgating_improved_20251024_044510_features.txt"
with open(short_exit_features_path, 'r') as f:
    short_exit_features = [line.strip() for line in f.readlines() if line.strip()]

print(f"✓ LONG Entry: {len(long_entry_features)} features")
print(f"✓ SHORT Entry: {len(short_entry_features)} features")
print(f"✓ LONG Exit: {len(long_exit_features)} features")
print(f"✓ SHORT Exit: {len(short_exit_features)} features")

# =============================================================================
# LOAD DATA - RECENT PERIOD
# =============================================================================

print("\nLoading recent market data...")
data_path = PROJECT_ROOT / "data" / "historical" / "BTCUSDT_5m_updated.csv"
df_full = pd.read_csv(data_path)
df_full['timestamp'] = pd.to_datetime(df_full['timestamp'])

# Get last 5 days only
end_date = df_full['timestamp'].max()
start_date = end_date - timedelta(days=5)
df_raw = df_full[df_full['timestamp'] >= start_date].copy()

# Calculate features
print("Calculating features...")
df = calculate_all_features_enhanced_v2(df_raw)
df = df.dropna().reset_index(drop=True)

print(f"\nData loaded:")
print(f"  Period: {df['timestamp'].min()} to {df['timestamp'].max()}")
print(f"  Candles: {len(df)}")
print(f"  Days: {(df['timestamp'].max() - df['timestamp'].min()).days} days")

# =============================================================================
# BACKTEST SIMULATION
# =============================================================================

class Trade:
    def __init__(self, entry_idx, entry_time, entry_price, direction, capital, leverage):
        self.entry_idx = entry_idx
        self.entry_time = entry_time
        self.entry_price = entry_price
        self.direction = direction
        self.capital = capital
        self.leverage = leverage
        self.position_size = capital * leverage
        self.hold_time = 0

    def update_pnl(self, current_price):
        if self.direction == 'LONG':
            price_change_pct = (current_price - self.entry_price) / self.entry_price
        else:  # SHORT
            price_change_pct = (self.entry_price - current_price) / self.entry_price

        leveraged_pnl_pct = price_change_pct * self.leverage
        pnl_dollar = self.capital * leveraged_pnl_pct
        return leveraged_pnl_pct, pnl_dollar

print("\n" + "=" * 80)
print("RUNNING BACKTEST...")
print("=" * 80)

capital = INITIAL_CAPITAL
position = None
trades = []
balance_history = []

for i in range(len(df)):
    current_time = df.loc[i, 'timestamp']
    current_price = df.loc[i, 'close']

    # Record balance
    if position:
        _, pnl = position.update_pnl(current_price)
        current_balance = capital + pnl
    else:
        current_balance = capital

    balance_history.append({
        'timestamp': current_time,
        'balance': current_balance
    })

    # If in position, check exit
    if position:
        position.hold_time += 1
        leveraged_pnl_pct, pnl_dollar = position.update_pnl(current_price)

        # Prepare exit features
        exit_features_dict = prepare_exit_features(
            df, i, position.entry_idx,
            position.direction, position.entry_price
        )

        should_exit = False
        exit_reason = None

        # Check emergency stops first
        if leveraged_pnl_pct <= -EMERGENCY_STOP_LOSS:
            should_exit = True
            exit_reason = "STOP_LOSS"
        elif position.hold_time >= EMERGENCY_MAX_HOLD_TIME:
            should_exit = True
            exit_reason = "MAX_HOLD"
        else:
            # Check ML exit
            if position.direction == 'LONG':
                exit_feat_df = pd.DataFrame([exit_features_dict])[long_exit_features]
                exit_feat_scaled = long_exit_scaler.transform(exit_feat_df)
                exit_prob = long_exit_model.predict_proba(exit_feat_scaled)[0, 1]

                if exit_prob >= ML_EXIT_THRESHOLD_LONG:
                    should_exit = True
                    exit_reason = "ML_EXIT"
            else:  # SHORT
                exit_feat_df = pd.DataFrame([exit_features_dict])[short_exit_features]
                exit_feat_scaled = short_exit_scaler.transform(exit_feat_df)
                exit_prob = short_exit_model.predict_proba(exit_feat_scaled)[0, 1]

                if exit_prob >= ML_EXIT_THRESHOLD_SHORT:
                    should_exit = True
                    exit_reason = "ML_EXIT"

        if should_exit:
            capital += pnl_dollar
            trades.append({
                'entry_time': position.entry_time,
                'exit_time': current_time,
                'direction': position.direction,
                'entry_price': position.entry_price,
                'exit_price': current_price,
                'pnl_pct': leveraged_pnl_pct,
                'pnl_dollar': pnl_dollar,
                'hold_time': position.hold_time,
                'exit_reason': exit_reason
            })
            position = None

    # If no position, check entry
    if not position and i >= 100:  # Need history for features
        # LONG signal
        try:
            long_feat_df = df.loc[[i], long_entry_features].copy()
            long_feat_scaled = long_entry_scaler.transform(long_feat_df)
            long_prob = long_entry_model.predict_proba(long_feat_scaled)[0, 1]
        except:
            long_prob = 0.0

        # SHORT signal
        try:
            short_feat_df = df.loc[[i], short_entry_features].copy()
            short_feat_scaled = short_entry_scaler.transform(short_feat_df)
            short_prob = short_entry_model.predict_proba(short_feat_scaled)[0, 1]
        except:
            short_prob = 0.0

        # Entry decision
        if long_prob >= LONG_THRESHOLD:
            position = Trade(i, current_time, current_price, 'LONG', capital, LEVERAGE)
        elif short_prob >= SHORT_THRESHOLD:
            long_ev = long_prob * LONG_AVG_RETURN
            short_ev = short_prob * SHORT_AVG_RETURN
            opportunity_cost = short_ev - long_ev

            if opportunity_cost > GATE_THRESHOLD:
                position = Trade(i, current_time, current_price, 'SHORT', capital, LEVERAGE)

# Close final position if any
if position:
    leveraged_pnl_pct, pnl_dollar = position.update_pnl(df.loc[len(df)-1, 'close'])
    capital += pnl_dollar
    trades.append({
        'entry_time': position.entry_time,
        'exit_time': df.loc[len(df)-1, 'timestamp'],
        'direction': position.direction,
        'entry_price': position.entry_price,
        'exit_price': df.loc[len(df)-1, 'close'],
        'pnl_pct': leveraged_pnl_pct,
        'pnl_dollar': pnl_dollar,
        'hold_time': position.hold_time,
        'exit_reason': 'BACKTEST_END'
    })

# =============================================================================
# RESULTS
# =============================================================================

df_trades = pd.DataFrame(trades)
df_balance = pd.DataFrame(balance_history)

print("\n" + "=" * 80)
print("BACKTEST RESULTS - ENTRY 0.80 VERIFICATION")
print("=" * 80)

if len(df_trades) > 0:
    total_return = (capital - INITIAL_CAPITAL) / INITIAL_CAPITAL * 100
    winners = df_trades[df_trades['pnl_pct'] > 0]
    losers = df_trades[df_trades['pnl_pct'] <= 0]
    win_rate = len(winners) / len(df_trades) * 100 if len(df_trades) > 0 else 0

    period_days = (df['timestamp'].max() - df['timestamp'].min()).days
    trades_per_day = len(df_trades) / max(period_days, 1)

    long_trades = df_trades[df_trades['direction'] == 'LONG']
    short_trades = df_trades[df_trades['direction'] == 'SHORT']

    ml_exits = df_trades[df_trades['exit_reason'] == 'ML_EXIT']
    sl_exits = df_trades[df_trades['exit_reason'] == 'STOP_LOSS']
    mh_exits = df_trades[df_trades['exit_reason'] == 'MAX_HOLD']

    print(f"\nPeriod: {df['timestamp'].min().strftime('%Y-%m-%d')} to {df['timestamp'].max().strftime('%Y-%m-%d')} ({period_days} days)")
    print(f"\nPerformance:")
    print(f"  Initial Capital: ${INITIAL_CAPITAL:,.2f}")
    print(f"  Final Capital: ${capital:,.2f}")
    print(f"  Total Return: {total_return:+.2f}%")
    print(f"\nTrades:")
    print(f"  Total: {len(df_trades)}")
    print(f"  Per Day: {trades_per_day:.1f}")
    print(f"  LONG: {len(long_trades)} ({len(long_trades)/len(df_trades)*100:.1f}%)")
    print(f"  SHORT: {len(short_trades)} ({len(short_trades)/len(df_trades)*100:.1f}%)")
    print(f"\nWin Rate:")
    print(f"  Overall: {win_rate:.1f}% ({len(winners)}/{len(df_trades)})")
    print(f"  Winners: {len(winners)} (avg: {winners['pnl_pct'].mean()*100:+.2f}%)")
    print(f"  Losers: {len(losers)} (avg: {losers['pnl_pct'].mean()*100:+.2f}%)")
    print(f"\nExit Reasons:")
    print(f"  ML Exit: {len(ml_exits)} ({len(ml_exits)/len(df_trades)*100:.1f}%)")
    print(f"  Stop Loss: {len(sl_exits)} ({len(sl_exits)/len(df_trades)*100:.1f}%)")
    print(f"  Max Hold: {len(mh_exits)} ({len(mh_exits)/len(df_trades)*100:.1f}%)")
    print(f"\n⚠️  CRITICAL: Trades per day = {trades_per_day:.1f}")
    print(f"    (Expected from grid search: ~5.1/day)")

    if trades_per_day < 2.0:
        print(f"\n🚨 WARNING: Very low trade frequency!")
        print(f"   Threshold 0.80 may be too high for current market conditions")

else:
    print("\n⚠️  NO TRADES EXECUTED!")
    print("   Entry threshold 0.80 did not trigger any entries")
    print("   Threshold is TOO HIGH for current market")

print("\n" + "=" * 80)

# Save results
results_file = PROJECT_ROOT / "results" / f"backtest_entry_080_verification_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
if len(df_trades) > 0:
    df_trades.to_csv(results_file, index=False)
    print(f"\n✓ Results saved: {results_file}")
else:
    print(f"\n⚠️  No results to save (0 trades)")
