"""
Validate 80/20 Split - MULTI-STAGE EXIT (SMART APPROACH)
=========================================================

Backtest on 20% holdout data (6,001 candles, Oct 5-26)
to validate MULTI-STAGE EXIT with Time Gates.

🧠 SMART SOLUTION: Time-Aware Exit Control

Problem: ASYMMETRIC 모델이 24-41분에 Exit (예상: 2-4시간)
Root Cause: ML은 TIME을 직접 이해 못함 - Feature pattern만 학습
Solution: Production 로직에서 TIME을 직접 통제

MULTI-STAGE EXIT LOGIC:
  Stage 1 (0-2h): EXIT BLOCKED
    - Exception: >2% profit (8× fees) → Early Exit OK
    - Purpose: 강제로 2시간 대기

  Stage 2 (2-6h): ML Exit Normal (0.75 threshold)
    - ML이 정상적으로 Exit 결정
    - Purpose: ML 지능 활용

  Stage 3 (6-10h): ML Exit Relaxed (0.60 threshold)
    - Exit 더 쉽게 (Max Hold 방지)
    - Purpose: 너무 오래 hold 방지

  Stage 4 (10h+): Emergency Force Exit
    - 무조건 Exit
    - Purpose: Capital 회전

Expected Results:
  - Avg Hold: 2-4 hours (vs 24-41 min before)
  - Trades/day: 5-10 (vs 17-27 before)
  - Fee Ratio: 40-60% (vs 128-195% before)
  - Return: Positive (vs -10% to -14% before)
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import pickle
import joblib
from datetime import datetime

# Configuration
ENTRY_THRESHOLD_LONG = 0.75
ENTRY_THRESHOLD_SHORT = 0.75
ML_EXIT_THRESHOLD_LONG = 0.75  # Stage 2 (2-6h)
ML_EXIT_THRESHOLD_SHORT = 0.75  # Stage 2 (2-6h)
LEVERAGE = 4
EMERGENCY_STOP_LOSS = -0.03
EMERGENCY_MAX_HOLD = 120
INITIAL_CAPITAL = 10000

# Multi-Stage Exit Configuration
STAGE_1_MAX_CANDLES = 24  # 2 hours (0-2h: EXIT BLOCKED)
STAGE_1_EARLY_EXIT_PROFIT = 0.02  # 2% profit allows early exit (8× fees)
STAGE_2_MAX_CANDLES = 72  # 6 hours (2-6h: Normal ML Exit 0.75)
STAGE_3_MAX_CANDLES = 120  # 10 hours (6-10h: Relaxed ML Exit 0.60)
STAGE_3_EXIT_THRESHOLD = 0.60  # Relaxed threshold for Stage 3
# Stage 4: 10h+ → Emergency Force Exit

# Trading fees (BingX taker fees)
ENTRY_FEE_RATE = 0.0005  # 0.05% entry
EXIT_FEE_RATE = 0.0005   # 0.05% exit
ROUND_TRIP_FEE_RATE = 0.001  # 0.10% total

DATA_DIR = PROJECT_ROOT / "data" / "features"
MODELS_DIR = PROJECT_ROOT / "models"

print("="*80)
print("VALIDATION: MULTI-STAGE EXIT (SMART APPROACH)")
print("="*80)
print()
print("📊 Validation Configuration:")
print(f"  Dataset: 20% holdout (6,001 candles, Oct 5-26)")
print(f"  Models: 8020split_fixed_20251103_102148 (ASYMMETRIC)")
print(f"  Entry Threshold: LONG {ENTRY_THRESHOLD_LONG}, SHORT {ENTRY_THRESHOLD_SHORT}")
print()
print("🧠 Multi-Stage Exit Logic:")
print(f"  Stage 1 (0-{STAGE_1_MAX_CANDLES} candles = 0-2h): EXIT BLOCKED")
print(f"    Exception: >>{STAGE_1_EARLY_EXIT_PROFIT*100}% profit → Early Exit OK")
print(f"  Stage 2 ({STAGE_1_MAX_CANDLES}-{STAGE_2_MAX_CANDLES} candles = 2-6h): ML Exit {ML_EXIT_THRESHOLD_LONG}")
print(f"  Stage 3 ({STAGE_2_MAX_CANDLES}-{STAGE_3_MAX_CANDLES} candles = 6-10h): ML Exit {STAGE_3_EXIT_THRESHOLD} (Relaxed)")
print(f"  Stage 4 (>{STAGE_3_MAX_CANDLES} candles = 10h+): Force Exit")
print()
print(f"  Emergency SL: {EMERGENCY_STOP_LOSS * 100}%")
print(f"  Leverage: {LEVERAGE}x")
print(f"  Trading Fees: {ROUND_TRIP_FEE_RATE * 100}% round-trip (0.05% entry + 0.05% exit)")
print(f"  Initial Capital: ${INITIAL_CAPITAL:,}")
print()

# ==============================================================================
# STEP 1: Load 20% Holdout Data
# ==============================================================================

print("-"*80)
print("STEP 1: Loading 20% Holdout Data")
print("-"*80)

df_full = pd.read_csv(DATA_DIR / "BTCUSDT_5m_features_complete.csv")
df_full['timestamp'] = pd.to_datetime(df_full['timestamp'])

# 80/20 split
split_idx = int(len(df_full) * 0.8)
df_val = df_full.iloc[split_idx:].copy().reset_index(drop=True)

print(f"✅ Loaded {len(df_val):,} candles (20% holdout)")
print(f"   Date range: {df_val['timestamp'].min()} to {df_val['timestamp'].max()}")
print(f"   Period: {(df_val['timestamp'].max() - df_val['timestamp'].min()).days} days")
print()

# ==============================================================================
# STEP 2: Load Models
# ==============================================================================

print("-"*80)
print("STEP 2: Loading Trained Models")
print("-"*80)

timestamp = "20251103_102148"  # ASYMMETRIC: Entry (1h, 0.3%), Exit (2h, 0.5%)

# Load LONG models
long_entry_model = joblib.load(MODELS_DIR / f"xgboost_long_entry_8020split_fixed_{timestamp}.pkl")
long_entry_scaler = joblib.load(MODELS_DIR / f"xgboost_long_entry_8020split_fixed_{timestamp}_scaler.pkl")
with open(MODELS_DIR / f"xgboost_long_entry_8020split_fixed_{timestamp}_features.txt", 'r') as f:
    long_entry_features = [line.strip() for line in f]

long_exit_model = joblib.load(MODELS_DIR / f"xgboost_long_exit_8020split_fixed_{timestamp}.pkl")
long_exit_scaler = joblib.load(MODELS_DIR / f"xgboost_long_exit_8020split_fixed_{timestamp}_scaler.pkl")

# Load SHORT models
short_entry_model = joblib.load(MODELS_DIR / f"xgboost_short_entry_8020split_fixed_{timestamp}.pkl")
short_entry_scaler = joblib.load(MODELS_DIR / f"xgboost_short_entry_8020split_fixed_{timestamp}_scaler.pkl")
with open(MODELS_DIR / f"xgboost_short_entry_8020split_fixed_{timestamp}_features.txt", 'r') as f:
    short_entry_features = [line.strip() for line in f]

short_exit_model = joblib.load(MODELS_DIR / f"xgboost_short_exit_8020split_fixed_{timestamp}.pkl")
short_exit_scaler = joblib.load(MODELS_DIR / f"xgboost_short_exit_8020split_fixed_{timestamp}_scaler.pkl")

print(f"✅ LONG Entry: {len(long_entry_features)} features")
print(f"✅ SHORT Entry: {len(short_entry_features)} features")
print(f"✅ Exit models loaded")
print()

# Load exit features from trained model
with open(MODELS_DIR / f"xgboost_long_exit_8020split_fixed_{timestamp}_features.txt", 'r') as f:
    EXIT_FEATURES = [line.strip() for line in f]

print(f"✅ Exit Features: {len(EXIT_FEATURES)} features")
print(f"   Features: {', '.join(EXIT_FEATURES[:5])}...")
print()

# ==============================================================================
# STEP 3: Backtest on 20% Holdout
# ==============================================================================

print("-"*80)
print("STEP 3: Running Validation Backtest (20% Holdout)")
print("-"*80)
print()

balance = INITIAL_CAPITAL
position = None
trades = []

for i in range(len(df_val)):
    current_candle = df_val.iloc[i]
    current_price = current_candle['close']

    # Check for exit if in position
    if position is not None:
        hold_time = i - position['entry_idx']

        # Calculate current P&L
        if position['side'] == 'LONG':
            pnl_pct = (current_price - position['entry_price']) / position['entry_price']
        else:  # SHORT
            pnl_pct = (position['entry_price'] - current_price) / position['entry_price']

        leveraged_pnl = pnl_pct * LEVERAGE

        # Check exit conditions - MULTI-STAGE EXIT
        exit_reason = None

        # Emergency Stop Loss (ALL STAGES)
        if leveraged_pnl <= EMERGENCY_STOP_LOSS:
            exit_reason = "Stop Loss"

        # Multi-Stage Exit Logic
        elif hold_time < STAGE_1_MAX_CANDLES:
            # Stage 1 (0-2h): EXIT BLOCKED
            # Exception: Early Profit Target (>2% = 8× fees)
            if leveraged_pnl > STAGE_1_EARLY_EXIT_PROFIT:
                exit_reason = f"Early Profit (>{STAGE_1_EARLY_EXIT_PROFIT*100}%)"
            # else: EXIT BLOCKED (no exit_reason)

        elif hold_time < STAGE_2_MAX_CANDLES:
            # Stage 2 (2-6h): Normal ML Exit (0.75)
            exit_features_df = df_val.iloc[[i]][EXIT_FEATURES].copy()

            if position['side'] == 'LONG':
                exit_features_scaled = long_exit_scaler.transform(exit_features_df)
                exit_prob = long_exit_model.predict_proba(exit_features_scaled)[0][1]
                threshold = ML_EXIT_THRESHOLD_LONG
            else:
                exit_features_scaled = short_exit_scaler.transform(exit_features_df)
                exit_prob = short_exit_model.predict_proba(exit_features_scaled)[0][1]
                threshold = ML_EXIT_THRESHOLD_SHORT

            if exit_prob >= threshold:
                exit_reason = f"ML Exit Stage2 ({exit_prob:.3f})"

        elif hold_time < STAGE_3_MAX_CANDLES:
            # Stage 3 (6-10h): Relaxed ML Exit (0.60)
            exit_features_df = df_val.iloc[[i]][EXIT_FEATURES].copy()

            if position['side'] == 'LONG':
                exit_features_scaled = long_exit_scaler.transform(exit_features_df)
                exit_prob = long_exit_model.predict_proba(exit_features_scaled)[0][1]
            else:
                exit_features_scaled = short_exit_scaler.transform(exit_features_df)
                exit_prob = short_exit_model.predict_proba(exit_features_scaled)[0][1]

            if exit_prob >= STAGE_3_EXIT_THRESHOLD:
                exit_reason = f"ML Exit Stage3 ({exit_prob:.3f})"

        else:
            # Stage 4 (10h+): Emergency Force Exit
            exit_reason = "Emergency Max Hold"

        # Exit if triggered
        if exit_reason:
            # Calculate P&L with fees
            position_size_usd = position['position_size']

            # Entry fee (already paid)
            entry_fee = position_size_usd * ENTRY_FEE_RATE

            # Exit fee
            exit_value = position_size_usd * (1 + leveraged_pnl)
            exit_fee = exit_value * EXIT_FEE_RATE

            # Total P&L = leveraged P&L - fees
            gross_pnl = position_size_usd * leveraged_pnl
            net_pnl = gross_pnl - entry_fee - exit_fee

            balance += net_pnl

            # Record trade
            trades.append({
                'entry_idx': position['entry_idx'],
                'exit_idx': i,
                'side': position['side'],
                'entry_price': position['entry_price'],
                'exit_price': current_price,
                'position_size': position_size_usd,
                'hold_time': hold_time,
                'pnl_pct': pnl_pct * 100,
                'leveraged_pnl_pct': leveraged_pnl * 100,
                'gross_pnl': gross_pnl,
                'entry_fee': entry_fee,
                'exit_fee': exit_fee,
                'net_pnl': net_pnl,
                'exit_reason': exit_reason,
                'balance': balance
            })

            position = None

    # Check for entry if no position
    if position is None:
        # Prepare entry features
        long_features_df = df_val.iloc[[i]][long_entry_features].copy()
        short_features_df = df_val.iloc[[i]][short_entry_features].copy()

        # Scale
        long_features_scaled = long_entry_scaler.transform(long_features_df)
        short_features_scaled = short_entry_scaler.transform(short_features_df)

        # Predict
        long_prob = long_entry_model.predict_proba(long_features_scaled)[0][1]
        short_prob = short_entry_model.predict_proba(short_features_scaled)[0][1]

        # Check entry conditions
        if long_prob >= ENTRY_THRESHOLD_LONG:
            # Enter LONG
            position_size = balance * 0.95  # 95% of balance
            position = {
                'side': 'LONG',
                'entry_idx': i,
                'entry_price': current_price,
                'position_size': position_size,
                'entry_prob': long_prob
            }

        elif short_prob >= ENTRY_THRESHOLD_SHORT:
            # Enter SHORT
            position_size = balance * 0.95  # 95% of balance
            position = {
                'side': 'SHORT',
                'entry_idx': i,
                'entry_price': current_price,
                'position_size': position_size,
                'entry_prob': short_prob
            }

# Convert to DataFrame
df_trades = pd.DataFrame(trades)

# ==============================================================================
# STEP 4: Calculate Metrics
# ==============================================================================

print("-"*80)
print("STEP 4: Validation Results")
print("-"*80)
print()

if len(df_trades) > 0:
    total_return = ((balance - INITIAL_CAPITAL) / INITIAL_CAPITAL) * 100

    wins = df_trades[df_trades['net_pnl'] > 0]
    losses = df_trades[df_trades['net_pnl'] <= 0]
    win_rate = (len(wins) / len(df_trades)) * 100

    avg_trade_pnl = df_trades['net_pnl'].mean()
    avg_win = wins['net_pnl'].mean() if len(wins) > 0 else 0
    avg_loss = losses['net_pnl'].mean() if len(losses) > 0 else 0

    avg_hold_candles = df_trades['hold_time'].mean()
    avg_hold_hours = avg_hold_candles / 12

    # Exit distribution
    ml_exits = df_trades[df_trades['exit_reason'].str.contains('ML Exit', na=False)]
    sl_exits = df_trades[df_trades['exit_reason'] == 'Stop Loss']
    mh_exits = df_trades[df_trades['exit_reason'] == 'Max Hold']

    ml_exit_rate = (len(ml_exits) / len(df_trades)) * 100
    sl_rate = (len(sl_exits) / len(df_trades)) * 100
    mh_rate = (len(mh_exits) / len(df_trades)) * 100

    # Side distribution
    long_trades = df_trades[df_trades['side'] == 'LONG']
    short_trades = df_trades[df_trades['side'] == 'SHORT']

    # Trades per day
    total_days = (df_val['timestamp'].max() - df_val['timestamp'].min()).days
    trades_per_day = len(df_trades) / total_days if total_days > 0 else 0

    # Fee impact
    total_fees = df_trades['entry_fee'].sum() + df_trades['exit_fee'].sum()
    total_gross_pnl = df_trades['gross_pnl'].sum()
    fee_impact_pct = (total_fees / abs(total_gross_pnl)) * 100 if total_gross_pnl != 0 else 0

    print(f"📊 Performance (20% Holdout - {len(df_val):,} candles, {total_days} days):")
    print(f"   Total Return: {total_return:+.2f}%")
    print(f"   Win Rate: {win_rate:.1f}% ({len(wins)}W / {len(losses)}L)")
    print(f"   Total Trades: {len(df_trades)}")
    print()

    print(f"💰 P&L Breakdown:")
    print(f"   Avg Trade: ${avg_trade_pnl:+.2f}")
    print(f"   Avg Win: ${avg_win:+.2f}")
    print(f"   Avg Loss: ${avg_loss:+.2f}")
    print(f"   Total Fees: ${total_fees:.2f} ({fee_impact_pct:.1f}% of gross P&L)")
    print()

    print(f"📈 Exit Distribution:")
    print(f"   ML Exit: {ml_exit_rate:.1f}% ({len(ml_exits)}/{len(df_trades)})")
    print(f"   Stop Loss: {sl_rate:.1f}% ({len(sl_exits)}/{len(df_trades)})")
    print(f"   Max Hold: {mh_rate:.1f}% ({len(mh_exits)}/{len(df_trades)})")
    print()

    print(f"⏱️  Trade Characteristics:")
    print(f"   Avg Hold: {avg_hold_candles:.1f} candles ({avg_hold_hours:.1f} hours)")
    print(f"   Trades/Day: {trades_per_day:.1f}")
    print()

    print(f"📊 Side Distribution:")
    print(f"   LONG: {len(long_trades)} trades ({len(long_trades)/len(df_trades)*100:.1f}%)")
    print(f"   SHORT: {len(short_trades)} trades ({len(short_trades)/len(df_trades)*100:.1f}%)")
    print()

    # Comparison to baselines
    print("-"*80)
    print("Baseline Comparison")
    print("-"*80)
    print()

    baseline_phase2 = -59.14
    baseline_production = -59.48

    improvement_phase2 = total_return - baseline_phase2
    improvement_production = total_return - baseline_production

    print(f"📊 Performance vs Baselines (20% Holdout):")
    print(f"   Phase 2 Enhanced: {baseline_phase2:+.2f}%")
    print(f"   Current Production: {baseline_production:+.2f}%")
    print(f"   Relaxed Labeling: {total_return:+.2f}%")
    print()

    print(f"✅ Improvement:")
    print(f"   vs Phase 2: {improvement_phase2:+.2f}pp")
    print(f"   vs Production: {improvement_production:+.2f}pp")
    print()

    # Deployment decision
    print("-"*80)
    print("Deployment Decision")
    print("-"*80)
    print()

    meets_improvement = improvement_phase2 > 5 and improvement_production > 5
    meets_trades = 5 <= trades_per_day <= 20
    meets_ml_exit = ml_exit_rate > 80

    print(f"🎯 Deployment Criteria:")
    print(f"   {'✅' if meets_improvement else '❌'} Return improvement > 5%: {min(improvement_phase2, improvement_production):+.2f}pp")
    print(f"   {'✅' if meets_trades else '❌'} Trades/day 5-20: {trades_per_day:.1f}")
    print(f"   {'✅' if meets_ml_exit else '❌'} ML Exit > 80%: {ml_exit_rate:.1f}%")
    print()

    if meets_improvement and meets_trades and meets_ml_exit:
        print("🎉 RECOMMENDATION: ✅ DEPLOY TO PRODUCTION")
        print("   All criteria met - significant improvement over baselines")
    else:
        print("⚠️  RECOMMENDATION: ❌ DO NOT DEPLOY")
        print("   Some criteria not met - further optimization needed")

else:
    print("❌ No trades executed during validation period")

# Save results
print()
print("-"*80)
print("STEP 5: Saving Results")
print("-"*80)

timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
results_path = PROJECT_ROOT / "results" / f"validate_8020split_relaxed_20%holdout_{timestamp_str}.csv"
df_trades.to_csv(results_path, index=False)

print(f"✅ Results saved: {results_path}")
print()
print("="*80)
print("VALIDATION COMPLETE")
print("="*80)
