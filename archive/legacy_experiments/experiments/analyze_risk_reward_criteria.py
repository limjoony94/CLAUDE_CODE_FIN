"""
Analyze Risk-Reward Criteria for Trade-Outcome Labeling
========================================================

Analyze actual MAE/MFE distributions to find realistic thresholds.
"""

import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import pickle
import joblib
from datetime import datetime

from scripts.experiments.calculate_all_features import calculate_all_features
from scripts.experiments.retrain_exit_models_opportunity_gating import prepare_exit_features
from scripts.experiments.trade_simulator import TradeSimulator

DATA_DIR = PROJECT_ROOT / "data" / "historical"
MODELS_DIR = PROJECT_ROOT / "models"

print("="*80)
print("RISK-REWARD CRITERIA ANALYSIS")
print("="*80)

# ============================================================================
# STEP 1: Load Data and Models
# ============================================================================

print("\n" + "-"*80)
print("STEP 1: Loading Data and Exit Models")
print("-"*80)

# Load data
data_file = DATA_DIR / "BTCUSDT_5m_max.csv"
df = pd.read_csv(data_file)
print(f"\n✅ Loaded {len(df):,} candles")

# Use sample for analysis (last 5,000)
df = df.tail(5000).copy().reset_index(drop=True)
print(f"   Using sample: {len(df):,} candles")

# Calculate features
print("\nCalculating features...")
df = calculate_all_features(df)
df = prepare_exit_features(df)
print(f"✅ Features calculated ({len(df.columns)} columns)")

# Load Exit models
print("\nLoading Exit models...")
long_exit_model_path = MODELS_DIR / "xgboost_long_exit_oppgating_improved_20251017_151624.pkl"
with open(long_exit_model_path, 'rb') as f:
    long_exit_model = pickle.load(f)

long_exit_scaler_path = MODELS_DIR / "xgboost_long_exit_oppgating_improved_20251017_151624_scaler.pkl"
long_exit_scaler = joblib.load(long_exit_scaler_path)

long_exit_features_path = MODELS_DIR / "xgboost_long_exit_oppgating_improved_20251017_151624_features.txt"
with open(long_exit_features_path, 'r') as f:
    long_exit_features = [line.strip() for line in f.readlines() if line.strip()]

short_exit_model_path = MODELS_DIR / "xgboost_short_exit_oppgating_improved_20251017_152440.pkl"
with open(short_exit_model_path, 'rb') as f:
    short_exit_model = pickle.load(f)

short_exit_scaler_path = MODELS_DIR / "xgboost_short_exit_oppgating_improved_20251017_152440_scaler.pkl"
short_exit_scaler = joblib.load(short_exit_scaler_path)

short_exit_features_path = MODELS_DIR / "xgboost_short_exit_oppgating_improved_20251017_152440_features.txt"
with open(short_exit_features_path, 'r') as f:
    short_exit_features = [line.strip() for line in f.readlines() if line.strip()]

print(f"  ✅ LONG Exit: {len(long_exit_features)} features")
print(f"  ✅ SHORT Exit: {len(short_exit_features)} features")

# Create simulators
long_simulator = TradeSimulator(
    exit_model=long_exit_model,
    exit_scaler=long_exit_scaler,
    exit_features=long_exit_features,
    leverage=4,
    ml_exit_threshold=0.70,
    emergency_stop_loss=-0.04,
    emergency_max_hold=96
)

short_simulator = TradeSimulator(
    exit_model=short_exit_model,
    exit_scaler=short_exit_scaler,
    exit_features=short_exit_features,
    leverage=4,
    ml_exit_threshold=0.72,
    emergency_stop_loss=-0.04,
    emergency_max_hold=96
)

print("✅ Trade simulators ready")

# ============================================================================
# STEP 2: Simulate All Trades and Collect MAE/MFE Data
# ============================================================================

print("\n" + "-"*80)
print("STEP 2: Simulating Trades and Collecting MAE/MFE Data")
print("-"*80)

# Simulate LONG trades
print("\nSimulating LONG trades...")
long_results = []
for i in range(100, len(df) - 96):  # Leave room for max hold
    result = long_simulator.simulate_trade(df, i, 'LONG')
    result['entry_idx'] = i
    long_results.append(result)

long_df = pd.DataFrame(long_results)
print(f"✅ Simulated {len(long_df):,} LONG trades")

# Simulate SHORT trades
print("\nSimulating SHORT trades...")
short_results = []
for i in range(100, len(df) - 96):
    result = short_simulator.simulate_trade(df, i, 'SHORT')
    result['entry_idx'] = i
    short_results.append(result)

short_df = pd.DataFrame(short_results)
print(f"✅ Simulated {len(short_df):,} SHORT trades")

# ============================================================================
# STEP 3: Analyze MAE/MFE Distributions
# ============================================================================

print("\n" + "="*80)
print("MAE/MFE DISTRIBUTION ANALYSIS")
print("="*80)

def analyze_mae_mfe(df, side):
    """Analyze MAE/MFE distribution"""
    print(f"\n{'='*80}")
    print(f"{side} TRADES ANALYSIS")
    print(f"{'='*80}")

    # Calculate leveraged values (4x)
    df['leveraged_pnl'] = df['pnl_pct'] * 4
    df['leveraged_mae'] = df['mae'] * 4
    df['leveraged_mfe'] = df['mfe'] * 4

    # Overall statistics
    print(f"\n📊 Overall Statistics (Leveraged 4x):")
    print(f"  Total Trades: {len(df):,}")
    print(f"  Profitable (>=2%): {len(df[df['leveraged_pnl'] >= 0.02]):,} ({len(df[df['leveraged_pnl'] >= 0.02])/len(df)*100:.1f}%)")
    print(f"  Average P&L: {df['leveraged_pnl'].mean()*100:.2f}%")
    print(f"  Average MAE: {df['leveraged_mae'].mean()*100:.2f}%")
    print(f"  Average MFE: {df['leveraged_mfe'].mean()*100:.2f}%")

    # MAE Distribution
    print(f"\n📉 MAE Distribution (Leveraged):")
    mae_thresholds = [-0.01, -0.02, -0.03, -0.04, -0.05, -0.06, -0.08, -0.10]
    for thresh in mae_thresholds:
        count = len(df[df['leveraged_mae'] >= thresh])
        pct = count / len(df) * 100
        print(f"  MAE >= {thresh*100:.0f}%: {count:,} ({pct:.1f}%)")

    # MFE Distribution
    print(f"\n📈 MFE Distribution (Leveraged):")
    mfe_thresholds = [0.01, 0.02, 0.03, 0.04, 0.05, 0.06, 0.08, 0.10]
    for thresh in mfe_thresholds:
        count = len(df[df['leveraged_mfe'] >= thresh])
        pct = count / len(df) * 100
        print(f"  MFE >= {thresh*100:.0f}%: {count:,} ({pct:.1f}%)")

    # Analyze profitable trades only
    profitable = df[df['leveraged_pnl'] >= 0.02].copy()
    if len(profitable) > 0:
        print(f"\n💰 Profitable Trades Analysis (P&L >= 2%):")
        print(f"  Count: {len(profitable):,}")
        print(f"  Average MAE: {profitable['leveraged_mae'].mean()*100:.2f}%")
        print(f"  Average MFE: {profitable['leveraged_mfe'].mean()*100:.2f}%")
        print(f"  Median MAE: {profitable['leveraged_mae'].median()*100:.2f}%")
        print(f"  Median MFE: {profitable['leveraged_mfe'].median()*100:.2f}%")

        # Risk-Reward combinations
        print(f"\n🎯 Risk-Reward Combinations (on Profitable Trades):")
        combinations = [
            (-0.02, 0.04),  # Current: Very strict
            (-0.03, 0.03),  # Balanced
            (-0.04, 0.02),  # More lenient
            (-0.05, 0.02),  # Very lenient
            (-0.03, 0.04),  # Strict upside, lenient downside
            (-0.04, 0.03),  # Medium
        ]

        for mae_thresh, mfe_thresh in combinations:
            count = len(profitable[(profitable['leveraged_mae'] >= mae_thresh) &
                                  (profitable['leveraged_mfe'] >= mfe_thresh)])
            pct = count / len(profitable) * 100
            total_pct = count / len(df) * 100
            print(f"  MAE >= {mae_thresh*100:.0f}% AND MFE >= {mfe_thresh*100:.0f}%:")
            print(f"    {count:,} ({pct:.1f}% of profitable, {total_pct:.1f}% of all)")

    # ML Exit effectiveness
    ml_exit_trades = df[df['exit_reason'] == 'ml_exit']
    print(f"\n🤖 ML Exit Effectiveness:")
    print(f"  ML Exit Trades: {len(ml_exit_trades):,} ({len(ml_exit_trades)/len(df)*100:.1f}%)")
    if len(ml_exit_trades) > 0:
        print(f"  Profitable: {len(ml_exit_trades[ml_exit_trades['leveraged_pnl'] >= 0.02]):,} ({len(ml_exit_trades[ml_exit_trades['leveraged_pnl'] >= 0.02])/len(ml_exit_trades)*100:.1f}%)")
        print(f"  Average P&L: {ml_exit_trades['leveraged_pnl'].mean()*100:.2f}%")

    return df

# Analyze both sides
long_analyzed = analyze_mae_mfe(long_df, "LONG")
short_analyzed = analyze_mae_mfe(short_df, "SHORT")

# ============================================================================
# STEP 4: Recommend Optimal Thresholds
# ============================================================================

print("\n" + "="*80)
print("RECOMMENDED THRESHOLDS")
print("="*80)

print(f"\n🎯 Current Thresholds (Too Strict):")
print(f"  MAE < -2%, MFE > 4%: Only 0.1% of trades qualify")
print(f"  Problem: Misses most good trades")

print(f"\n✅ Recommended Option 1: BALANCED")
print(f"  MAE >= -3%, MFE >= 3%")
print(f"  Rationale: Balanced risk-reward, captures more good trades")

print(f"\n✅ Recommended Option 2: LENIENT (Better for catching opportunities)")
print(f"  MAE >= -4%, MFE >= 2%")
print(f"  Rationale: More lenient, captures most profitable trades")

print(f"\n✅ Recommended Option 3: STRICT UPSIDE (Quality focus)")
print(f"  MAE >= -3%, MFE >= 4%")
print(f"  Rationale: Lenient on risk, strict on reward quality")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)

print(f"\n📝 Recommendation: Start with Option 2 (MAE >= -4%, MFE >= 2%)")
print(f"   This captures most profitable trades while maintaining quality")
print(f"   Can tighten later if needed based on backtest results")
