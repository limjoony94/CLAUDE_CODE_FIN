"""
Exit Model Performance Analysis
================================

Analyze exit timing and performance to identify improvement opportunities
for LONG and SHORT exit models separately.

Author: Claude Code
Date: 2025-10-18
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import sys
import time
from datetime import datetime

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiments.calculate_all_features import calculate_all_features
from scripts.experiments.retrain_exit_models_opportunity_gating import prepare_exit_features
from scripts.production.dynamic_position_sizing import DynamicPositionSizer

DATA_DIR = PROJECT_ROOT / "data" / "historical"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# Strategy Parameters
LEVERAGE = 4
LONG_THRESHOLD = 0.65
SHORT_THRESHOLD = 0.70
GATE_THRESHOLD = 0.001
ML_EXIT_THRESHOLD = 0.70
EMERGENCY_STOP_LOSS = -0.03  # Balance-Based: -3% total balance (optimized 2025-10-22, Return: +50.31%, WR: 57.1%, MDD: -15%)
EMERGENCY_MAX_HOLD = 96
LONG_AVG_RETURN = 0.0041
SHORT_AVG_RETURN = 0.0047
INITIAL_CAPITAL = 10000.0
TAKER_FEE = 0.0005

print("="*80)
print("EXIT MODEL PERFORMANCE ANALYSIS")
print("="*80)

# Load models
print("\nLoading models...")
with open(MODELS_DIR / "xgboost_v4_phase4_advanced_lookahead3_thresh0.pkl", 'rb') as f:
    long_model = pickle.load(f)
with open(MODELS_DIR / "xgboost_v4_phase4_advanced_lookahead3_thresh0_scaler.pkl", 'rb') as f:
    long_scaler = pickle.load(f)
with open(MODELS_DIR / "xgboost_v4_phase4_advanced_lookahead3_thresh0_features.txt", 'r') as f:
    long_feature_columns = [line.strip() for line in f.readlines()]

with open(MODELS_DIR / "xgboost_short_redesigned_20251016_233322.pkl", 'rb') as f:
    short_model = pickle.load(f)
with open(MODELS_DIR / "xgboost_short_redesigned_20251016_233322_scaler.pkl", 'rb') as f:
    short_scaler = pickle.load(f)
with open(MODELS_DIR / "xgboost_short_redesigned_20251016_233322_features.txt", 'r') as f:
    short_feature_columns = [line.strip() for line in f.readlines()]

with open(MODELS_DIR / "xgboost_long_exit_oppgating_improved_20251017_151624.pkl", 'rb') as f:
    long_exit_model = pickle.load(f)
with open(MODELS_DIR / "xgboost_long_exit_oppgating_improved_20251017_151624_scaler.pkl", 'rb') as f:
    long_exit_scaler = pickle.load(f)
with open(MODELS_DIR / "xgboost_long_exit_oppgating_improved_20251017_151624_features.txt", 'r') as f:
    long_exit_feature_columns = [line.strip() for line in f.readlines() if line.strip()]

with open(MODELS_DIR / "xgboost_short_exit_oppgating_improved_20251017_152440.pkl", 'rb') as f:
    short_exit_model = pickle.load(f)
with open(MODELS_DIR / "xgboost_short_exit_oppgating_improved_20251017_152440_scaler.pkl", 'rb') as f:
    short_exit_scaler = pickle.load(f)
with open(MODELS_DIR / "xgboost_short_exit_oppgating_improved_20251017_152440_features.txt", 'r') as f:
    short_exit_feature_columns = [line.strip() for line in f.readlines() if line.strip()]

print("✅ Models loaded")

# Initialize sizer
sizer = DynamicPositionSizer(base_position_pct=0.50, max_position_pct=0.95, min_position_pct=0.20)

# Load and prepare data
print("\nLoading data...")
df_full = pd.read_csv(DATA_DIR / "BTCUSDT_5m_max.csv")
print(f"Calculating features...")
df = calculate_all_features(df_full)
df = prepare_exit_features(df)

# Pre-calculate signals
print("Pre-calculating signals...")
long_feat = df[long_feature_columns].values
long_feat_scaled = long_scaler.transform(long_feat)
df['long_prob'] = long_model.predict_proba(long_feat_scaled)[:, 1]

short_feat = df[short_feature_columns].values
short_feat_scaled = short_scaler.transform(short_feat)
df['short_prob'] = short_model.predict_proba(short_feat_scaled)[:, 1]

print("✅ Data prepared\n")


def analyze_exit_timing(window_df, side='LONG'):
    """
    Analyze exit timing for a specific side

    For each trade:
    - Track exit probability over time
    - Compare ML exit timing vs optimal timing
    - Measure opportunity cost of early/late exit
    """
    trades_analysis = []
    position = None

    for i in range(len(window_df) - 1):
        current_price = window_df['close'].iloc[i]

        # Entry logic
        if position is None:
            long_prob = window_df['long_prob'].iloc[i]
            short_prob = window_df['short_prob'].iloc[i]

            # Check for entry (matching our strategy)
            if side == 'LONG' and long_prob >= LONG_THRESHOLD:
                sizing_result = sizer.get_position_size_simple(
                    capital=INITIAL_CAPITAL, signal_strength=long_prob, leverage=LEVERAGE
                )
                position = {
                    'entry_idx': i,
                    'entry_price': current_price,
                    'quantity': sizing_result['leveraged_value'] / current_price,
                    'position_value': sizing_result['position_value'],
                    'exit_probs': [],  # Track exit probability over time
                    'pnl_history': [],  # Track P&L over time
                }
            elif side == 'SHORT' and short_prob >= SHORT_THRESHOLD:
                long_ev = long_prob * LONG_AVG_RETURN
                short_ev = short_prob * SHORT_AVG_RETURN
                if (short_ev - long_ev) > GATE_THRESHOLD:
                    sizing_result = sizer.get_position_size_simple(
                        capital=INITIAL_CAPITAL, signal_strength=short_prob, leverage=LEVERAGE
                    )
                    position = {
                        'entry_idx': i,
                        'entry_price': current_price,
                        'quantity': sizing_result['leveraged_value'] / current_price,
                        'position_value': sizing_result['position_value'],
                        'exit_probs': [],
                        'pnl_history': [],
                    }

        # Track position
        if position is not None:
            time_in_pos = i - position['entry_idx']

            # Calculate P&L
            entry_notional = position['quantity'] * position['entry_price']
            current_notional = position['quantity'] * current_price

            if side == 'LONG':
                pnl_usd = current_notional - entry_notional
                price_change_pct = (current_price - position['entry_price']) / position['entry_price']
            else:
                pnl_usd = entry_notional - current_notional
                price_change_pct = (position['entry_price'] - current_price) / position['entry_price']

            leveraged_pnl_pct = price_change_pct * LEVERAGE

            # Get exit probability
            try:
                if side == 'LONG':
                    exit_features_values = window_df[long_exit_feature_columns].iloc[i:i+1].values
                    exit_features_scaled = long_exit_scaler.transform(exit_features_values)
                    exit_prob = long_exit_model.predict_proba(exit_features_scaled)[0][1]
                else:
                    exit_features_values = window_df[short_exit_feature_columns].iloc[i:i+1].values
                    exit_features_scaled = short_exit_scaler.transform(exit_features_values)
                    exit_prob = short_exit_model.predict_proba(exit_features_scaled)[0][1]

                position['exit_probs'].append(exit_prob)
                position['pnl_history'].append(leveraged_pnl_pct)

                # Check exit
                should_exit = False
                exit_reason = None

                if exit_prob >= ML_EXIT_THRESHOLD:
                    should_exit = True
                    exit_reason = 'ml_exit'
                elif leveraged_pnl_pct <= EMERGENCY_STOP_LOSS:
                    should_exit = True
                    exit_reason = 'emergency_stop_loss'
                elif time_in_pos >= EMERGENCY_MAX_HOLD:
                    should_exit = True
                    exit_reason = 'emergency_max_hold'

                if should_exit:
                    # Calculate fees
                    entry_commission = position['position_value'] * LEVERAGE * TAKER_FEE
                    exit_commission = position['quantity'] * current_price * TAKER_FEE
                    total_commission = entry_commission + exit_commission
                    net_pnl_usd = pnl_usd - total_commission

                    # Find peak/trough P&L
                    pnl_hist = np.array(position['pnl_history'])
                    peak_pnl = pnl_hist.max() if len(pnl_hist) > 0 else leveraged_pnl_pct
                    trough_pnl = pnl_hist.min() if len(pnl_hist) > 0 else leveraged_pnl_pct

                    # Calculate "optimal" exit (peak for LONG, trough for SHORT)
                    if side == 'LONG':
                        optimal_pnl = peak_pnl
                        opportunity_cost = peak_pnl - leveraged_pnl_pct
                    else:
                        optimal_pnl = trough_pnl  # For SHORT, trough is best
                        opportunity_cost = trough_pnl - leveraged_pnl_pct

                    trades_analysis.append({
                        'side': side,
                        'hold_time': time_in_pos,
                        'exit_reason': exit_reason,
                        'exit_pnl_pct': leveraged_pnl_pct,
                        'peak_pnl_pct': peak_pnl,
                        'trough_pnl_pct': trough_pnl,
                        'optimal_pnl_pct': optimal_pnl,
                        'opportunity_cost_pct': opportunity_cost,
                        'exit_prob': exit_prob,
                        'avg_exit_prob': np.mean(position['exit_probs']),
                        'max_exit_prob': np.max(position['exit_probs']),
                        'exit_prob_at_peak': position['exit_probs'][np.argmax(pnl_hist)] if len(pnl_hist) > 0 else exit_prob,
                        'won': net_pnl_usd > 0
                    })

                    position = None

            except Exception as e:
                pass

    return trades_analysis


# Analyze LONG exits
print("="*80)
print("ANALYZING LONG EXIT PERFORMANCE")
print("="*80)

# Sample 10 windows for detailed analysis
window_size = 1440
step_size = 288
num_sample_windows = 10
sample_indices = np.linspace(0, (len(df) - window_size) // step_size - 1, num_sample_windows, dtype=int)

long_trades = []
for idx in sample_indices:
    start_idx = idx * step_size
    end_idx = start_idx + window_size
    if end_idx <= len(df):
        window_df = df.iloc[start_idx:end_idx].reset_index(drop=True)
        window_trades = analyze_exit_timing(window_df, side='LONG')
        long_trades.extend(window_trades)

if len(long_trades) > 0:
    long_df = pd.DataFrame(long_trades)

    print(f"\n📊 LONG Exit Analysis ({len(long_df)} trades):\n")

    # Overall metrics
    print(f"Win Rate: {long_df['won'].sum() / len(long_df) * 100:.1f}%")
    print(f"Avg Exit P&L: {long_df['exit_pnl_pct'].mean()*100:+.2f}%")
    print(f"Avg Peak P&L: {long_df['peak_pnl_pct'].mean()*100:+.2f}%")
    print(f"Avg Opportunity Cost: {long_df['opportunity_cost_pct'].mean()*100:.2f}%")

    # Exit reason breakdown
    print(f"\n🚪 Exit Reasons:")
    for reason in long_df['exit_reason'].unique():
        reason_df = long_df[long_df['exit_reason'] == reason]
        print(f"  {reason}:")
        print(f"    Count: {len(reason_df)} ({len(reason_df)/len(long_df)*100:.1f}%)")
        print(f"    Win Rate: {reason_df['won'].sum() / len(reason_df) * 100:.1f}%")
        print(f"    Avg Exit P&L: {reason_df['exit_pnl_pct'].mean()*100:+.2f}%")
        print(f"    Avg Opportunity Cost: {reason_df['opportunity_cost_pct'].mean()*100:.2f}%")

    # ML Exit timing analysis
    ml_exits = long_df[long_df['exit_reason'] == 'ml_exit']
    if len(ml_exits) > 0:
        print(f"\n🤖 ML Exit Timing Analysis:")
        print(f"  Avg Hold Time: {ml_exits['hold_time'].mean():.1f} candles ({ml_exits['hold_time'].mean()*5:.0f} min)")
        print(f"  Exit Probability at Exit: {ml_exits['exit_prob'].mean():.3f}")
        print(f"  Exit Probability at Peak: {ml_exits['exit_prob_at_peak'].mean():.3f}")
        print(f"  Opportunity Cost (Peak - Exit): {ml_exits['opportunity_cost_pct'].mean()*100:.2f}%")

        # Categorize timing
        early_exits = ml_exits[ml_exits['opportunity_cost_pct'] > 0.01]  # Left >1% on table
        good_exits = ml_exits[(ml_exits['opportunity_cost_pct'] >= -0.01) & (ml_exits['opportunity_cost_pct'] <= 0.01)]
        late_exits = ml_exits[ml_exits['opportunity_cost_pct'] < -0.01]  # Gave back >1%

        print(f"\n  Timing Quality:")
        print(f"    Early (>1% left): {len(early_exits)} ({len(early_exits)/len(ml_exits)*100:.1f}%)")
        print(f"    Good (±1%): {len(good_exits)} ({len(good_exits)/len(ml_exits)*100:.1f}%)")
        print(f"    Late (<-1% gave back): {len(late_exits)} ({len(late_exits)/len(ml_exits)*100:.1f}%)")


# Analyze SHORT exits
print("\n" + "="*80)
print("ANALYZING SHORT EXIT PERFORMANCE")
print("="*80)

short_trades = []
for idx in sample_indices:
    start_idx = idx * step_size
    end_idx = start_idx + window_size
    if end_idx <= len(df):
        window_df = df.iloc[start_idx:end_idx].reset_index(drop=True)
        window_trades = analyze_exit_timing(window_df, side='SHORT')
        short_trades.extend(window_trades)

if len(short_trades) > 0:
    short_df = pd.DataFrame(short_trades)

    print(f"\n📊 SHORT Exit Analysis ({len(short_df)} trades):\n")

    # Overall metrics
    print(f"Win Rate: {short_df['won'].sum() / len(short_df) * 100:.1f}%")
    print(f"Avg Exit P&L: {short_df['exit_pnl_pct'].mean()*100:+.2f}%")
    print(f"Avg Trough P&L: {short_df['trough_pnl_pct'].mean()*100:+.2f}%")  # For SHORT
    print(f"Avg Opportunity Cost: {short_df['opportunity_cost_pct'].mean()*100:.2f}%")

    # Exit reason breakdown
    print(f"\n🚪 Exit Reasons:")
    for reason in short_df['exit_reason'].unique():
        reason_df = short_df[short_df['exit_reason'] == reason]
        print(f"  {reason}:")
        print(f"    Count: {len(reason_df)} ({len(reason_df)/len(short_df)*100:.1f}%)")
        print(f"    Win Rate: {reason_df['won'].sum() / len(reason_df) * 100:.1f}%")
        print(f"    Avg Exit P&L: {reason_df['exit_pnl_pct'].mean()*100:+.2f}%")
        print(f"    Avg Opportunity Cost: {reason_df['opportunity_cost_pct'].mean()*100:.2f}%")

    # ML Exit timing analysis
    ml_exits = short_df[short_df['exit_reason'] == 'ml_exit']
    if len(ml_exits) > 0:
        print(f"\n🤖 ML Exit Timing Analysis:")
        print(f"  Avg Hold Time: {ml_exits['hold_time'].mean():.1f} candles ({ml_exits['hold_time'].mean()*5:.0f} min)")
        print(f"  Exit Probability at Exit: {ml_exits['exit_prob'].mean():.3f}")
        print(f"  Opportunity Cost (Trough - Exit): {ml_exits['opportunity_cost_pct'].mean()*100:.2f}%")

        # Categorize timing
        early_exits = ml_exits[ml_exits['opportunity_cost_pct'] > 0.01]
        good_exits = ml_exits[(ml_exits['opportunity_cost_pct'] >= -0.01) & (ml_exits['opportunity_cost_pct'] <= 0.01)]
        late_exits = ml_exits[ml_exits['opportunity_cost_pct'] < -0.01]

        print(f"\n  Timing Quality:")
        print(f"    Early (>1% left): {len(early_exits)} ({len(early_exits)/len(ml_exits)*100:.1f}%)")
        print(f"    Good (±1%): {len(good_exits)} ({len(good_exits)/len(ml_exits)*100:.1f}%)")
        print(f"    Late (<-1% gave back): {len(late_exits)} ({len(late_exits)/len(ml_exits)*100:.1f}%)")


# Summary
print("\n" + "="*80)
print("IMPROVEMENT OPPORTUNITIES")
print("="*80)

if len(long_trades) > 0:
    long_df = pd.DataFrame(long_trades)
    ml_long = long_df[long_df['exit_reason'] == 'ml_exit']
    if len(ml_long) > 0:
        avg_opp_cost = ml_long['opportunity_cost_pct'].mean()
        print(f"\n💡 LONG Exit:")
        print(f"  Average opportunity cost: {avg_opp_cost*100:.2f}%")
        if avg_opp_cost > 0.005:
            print(f"  ⚠️  Exiting TOO EARLY on average (leaving {avg_opp_cost*100:.2f}% on table)")
            print(f"  → Consider: Increase exit threshold or add peak-detection logic")
        elif avg_opp_cost < -0.005:
            print(f"  ⚠️  Exiting TOO LATE on average (giving back {-avg_opp_cost*100:.2f}%)")
            print(f"  → Consider: Decrease exit threshold or improve reversal detection")
        else:
            print(f"  ✅ Timing is reasonable (±0.5% of optimal)")

if len(short_trades) > 0:
    short_df = pd.DataFrame(short_trades)
    ml_short = short_df[short_df['exit_reason'] == 'ml_exit']
    if len(ml_short) > 0:
        avg_opp_cost = ml_short['opportunity_cost_pct'].mean()
        print(f"\n💡 SHORT Exit:")
        print(f"  Average opportunity cost: {avg_opp_cost*100:.2f}%")
        if avg_opp_cost > 0.005:
            print(f"  ⚠️  Exiting TOO EARLY on average (leaving {avg_opp_cost*100:.2f}% on table)")
            print(f"  → Consider: Increase exit threshold or add trough-detection logic")
        elif avg_opp_cost < -0.005:
            print(f"  ⚠️  Exiting TOO LATE on average (giving back {-avg_opp_cost*100:.2f}%)")
            print(f"  → Consider: Decrease exit threshold or improve reversal detection")
        else:
            print(f"  ✅ Timing is reasonable (±0.5% of optimal)")

print("\n" + "="*80)