"""
EXIT MODEL FAILURE INVESTIGATION

Analyzes why full dataset Exit models failed catastrophically:
- Current: 15.2% WR, -69.10% return, 98.5% ML Exit
- Expected: 73.86% WR, +38.04% return, 77.0% ML Exit

Investigation Areas:
1. Exit label distribution and timing patterns
2. Exit model predictions vs actual outcomes
3. Comparison with Production Exit models
4. Alternative labeling strategies
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
from datetime import datetime

# Paths
BASE_DIR = Path("C:/Users/J/OneDrive/CLAUDE_CODE_FIN/bingx_rl_trading_bot")
RESULTS_DIR = BASE_DIR / "results"
MODELS_DIR = BASE_DIR / "models"

# Load backtest results
print("=" * 80)
print("EXIT MODEL FAILURE INVESTIGATION")
print("=" * 80)
print()

# Load full dataset backtest results
backtest_file = RESULTS_DIR / "backtest_complete_full_dataset_20251031_191006.csv"
df_backtest = pd.read_csv(backtest_file)

print(f"✅ Loaded {len(df_backtest)} trades from backtest")
print()

# ============================================================================
# PART 1: Exit Label Distribution Analysis
# ============================================================================
print("=" * 80)
print("PART 1: EXIT LABEL DISTRIBUTION ANALYSIS")
print("=" * 80)
print()

# Load features dataset to analyze labels
features_file = BASE_DIR / "data" / "features" / "BTCUSDT_5m_features.csv"
df_features = pd.read_csv(features_file)
df_features['timestamp'] = pd.to_datetime(df_features['timestamp'])

print(f"✅ Loaded {len(df_features)} candles from features dataset")
print()

# Constants from training
LEVERAGE = 4
PROFIT_TARGET = 0.02  # 2% leveraged
EMERGENCY_MAX_HOLD = 120  # 10 hours

def calculate_exit_labels(df, side):
    """
    Reproduce the Exit labeling logic to analyze distribution
    """
    labels = []
    label_reasons = []
    candles_to_exit = []

    for idx in range(len(df)):
        if idx + EMERGENCY_MAX_HOLD >= len(df):
            labels.append(0)
            label_reasons.append('insufficient_future')
            candles_to_exit.append(np.nan)
            continue

        entry_price = df.loc[df.index[idx], 'close']

        # Look ahead for future price movement
        future_indices = df.index[idx+1:idx+1+EMERGENCY_MAX_HOLD]
        future = df.loc[future_indices].copy()

        # Calculate leveraged P&L
        if side == 'LONG':
            future['leveraged_pnl'] = ((future['close'] - entry_price) / entry_price) * LEVERAGE
        else:
            future['leveraged_pnl'] = ((entry_price - future['close']) / entry_price) * LEVERAGE

        # Check if profit target hit
        profit_exits = future[future['leveraged_pnl'] >= PROFIT_TARGET]

        if not profit_exits.empty:
            first_profit_idx = profit_exits.index[0]
            candles_elapsed = df.index.get_loc(first_profit_idx) - idx

            # Label = 1 if profit hit within 60 candles
            if candles_elapsed <= 60:
                labels.append(1)
                label_reasons.append('profit_within_60')
                candles_to_exit.append(candles_elapsed)
            else:
                labels.append(0)
                label_reasons.append('profit_after_60')
                candles_to_exit.append(candles_elapsed)
        else:
            labels.append(0)
            label_reasons.append('no_profit')
            candles_to_exit.append(np.nan)

    return np.array(labels), label_reasons, candles_to_exit

print("Calculating LONG Exit labels...")
long_labels, long_reasons, long_candles = calculate_exit_labels(df_features, 'LONG')

print("Calculating SHORT Exit labels...")
short_labels, short_reasons, short_candles = calculate_exit_labels(df_features, 'SHORT')

print()
print("-" * 80)
print("LONG Exit Label Distribution:")
print("-" * 80)
print(f"Total candles: {len(long_labels):,}")
print(f"Label = 1 (Exit): {long_labels.sum():,} ({long_labels.mean()*100:.2f}%)")
print(f"Label = 0 (Hold): {(1-long_labels).sum():,} ({(1-long_labels.mean())*100:.2f}%)")
print()

from collections import Counter
long_reason_counts = Counter(long_reasons)
print("Label Reasons:")
for reason, count in long_reason_counts.most_common():
    print(f"  {reason}: {count:,} ({count/len(long_reasons)*100:.2f}%)")
print()

# Analyze timing for Exit labels
long_exit_candles = [c for c, l in zip(long_candles, long_labels) if l == 1 and not np.isnan(c)]
if long_exit_candles:
    print(f"Candles to profit (when Exit=1):")
    print(f"  Mean: {np.mean(long_exit_candles):.1f} candles")
    print(f"  Median: {np.median(long_exit_candles):.1f} candles")
    print(f"  Min: {np.min(long_exit_candles):.0f} candles")
    print(f"  Max: {np.max(long_exit_candles):.0f} candles")
print()

print("-" * 80)
print("SHORT Exit Label Distribution:")
print("-" * 80)
print(f"Total candles: {len(short_labels):,}")
print(f"Label = 1 (Exit): {short_labels.sum():,} ({short_labels.mean()*100:.2f}%)")
print(f"Label = 0 (Hold): {(1-short_labels).sum():,} ({(1-short_labels.mean())*100:.2f}%)")
print()

short_reason_counts = Counter(short_reasons)
print("Label Reasons:")
for reason, count in short_reason_counts.most_common():
    print(f"  {reason}: {count:,} ({count/len(short_reasons)*100:.2f}%)")
print()

short_exit_candles = [c for c, l in zip(short_candles, short_labels) if l == 1 and not np.isnan(c)]
if short_exit_candles:
    print(f"Candles to profit (when Exit=1):")
    print(f"  Mean: {np.mean(short_exit_candles):.1f} candles")
    print(f"  Median: {np.median(short_exit_candles):.1f} candles")
    print(f"  Min: {np.min(short_exit_candles):.0f} candles")
    print(f"  Max: {np.max(short_exit_candles):.0f} candles")
print()

# ============================================================================
# PART 2: Exit Model Predictions vs Actual Outcomes
# ============================================================================
print("=" * 80)
print("PART 2: EXIT MODEL PREDICTIONS VS ACTUAL OUTCOMES")
print("=" * 80)
print()

# Analyze backtest trades
df_backtest['profit_pct'] = (df_backtest['exit_price'] - df_backtest['entry_price']) / df_backtest['entry_price'] * 100
df_backtest['leveraged_pnl_pct'] = df_backtest['profit_pct'] * LEVERAGE

# Separate by exit reason
ml_exits = df_backtest[df_backtest['exit_reason'] == 'ML_EXIT']
sl_exits = df_backtest[df_backtest['exit_reason'] == 'STOP_LOSS']
max_hold_exits = df_backtest[df_backtest['exit_reason'] == 'MAX_HOLD']

print(f"Total Trades: {len(df_backtest):,}")
print(f"  ML Exits: {len(ml_exits):,} ({len(ml_exits)/len(df_backtest)*100:.1f}%)")
print(f"  Stop Loss: {len(sl_exits):,} ({len(sl_exits)/len(df_backtest)*100:.1f}%)")
print()

print("-" * 80)
print("ML Exit Performance:")
print("-" * 80)
wins_ml = ml_exits[ml_exits['net_pnl'] > 0]
losses_ml = ml_exits[ml_exits['net_pnl'] <= 0]

print(f"Win Rate: {len(wins_ml)/len(ml_exits)*100:.2f}% ({len(wins_ml)}/{len(ml_exits)})")
print(f"Avg Profit (winners): {wins_ml['net_pnl'].mean():.2f} USDT ({wins_ml['leveraged_pnl_pct'].mean():.2f}%)")
print(f"Avg Loss (losers): {losses_ml['net_pnl'].mean():.2f} USDT ({losses_ml['leveraged_pnl_pct'].mean():.2f}%)")
print(f"Overall Avg: {ml_exits['net_pnl'].mean():.2f} USDT ({ml_exits['leveraged_pnl_pct'].mean():.2f}%)")
print(f"Avg Hold Time: {ml_exits['hold_time'].mean():.1f} candles ({ml_exits['hold_time'].mean()/12:.1f} hours)")
print()

# Analyze by side
long_ml = ml_exits[ml_exits['side'] == 'LONG']
short_ml = ml_exits[ml_exits['side'] == 'SHORT']

print("LONG ML Exits:")
print(f"  Count: {len(long_ml):,}")
print(f"  Win Rate: {(long_ml['net_pnl'] > 0).sum()/len(long_ml)*100:.2f}%")
print(f"  Avg P&L: {long_ml['leveraged_pnl_pct'].mean():.2f}%")
print(f"  Avg Hold: {long_ml['hold_time'].mean():.1f} candles")
print()

print("SHORT ML Exits:")
print(f"  Count: {len(short_ml):,}")
print(f"  Win Rate: {(short_ml['net_pnl'] > 0).sum()/len(short_ml)*100:.2f}%")
print(f"  Avg P&L: {short_ml['leveraged_pnl_pct'].mean():.2f}%")
print(f"  Avg Hold: {short_ml['hold_time'].mean():.1f} candles")
print()

# ============================================================================
# PART 3: Timing Analysis - When are ML Exits triggering?
# ============================================================================
print("=" * 80)
print("PART 3: ML EXIT TIMING ANALYSIS")
print("=" * 80)
print()

# Analyze ML Exit timing relative to profit potential
print("Analyzing ML Exit timing vs profit potential...")
print()

# Sample analysis: For each ML Exit, what was the max profit achievable?
def analyze_exit_timing(trade_row, df_features):
    """
    For a given trade, analyze if ML Exit was optimal timing
    """
    entry_time = pd.to_datetime(trade_row['entry_time'])
    exit_time = pd.to_datetime(trade_row['exit_time'])
    entry_price = trade_row['entry_price']
    exit_price = trade_row['exit_price']
    side = trade_row['side']

    # Get price data during trade
    mask = (df_features['timestamp'] >= entry_time) & (df_features['timestamp'] <= exit_time)
    trade_data = df_features[mask].copy()

    if len(trade_data) < 2:
        return None

    # Calculate P&L at each candle
    if side == 'LONG':
        trade_data['pnl_pct'] = ((trade_data['close'] - entry_price) / entry_price) * LEVERAGE
    else:
        trade_data['pnl_pct'] = ((entry_price - trade_data['close']) / entry_price) * LEVERAGE

    # Find max profit during trade
    max_pnl = trade_data['pnl_pct'].max()
    min_pnl = trade_data['pnl_pct'].min()

    # Actual exit P&L
    if side == 'LONG':
        actual_pnl = ((exit_price - entry_price) / entry_price) * LEVERAGE
    else:
        actual_pnl = ((entry_price - exit_price) / entry_price) * LEVERAGE

    return {
        'max_pnl': max_pnl,
        'min_pnl': min_pnl,
        'actual_pnl': actual_pnl,
        'missed_profit': max_pnl - actual_pnl,
        'avoided_loss': actual_pnl - min_pnl
    }

# Sample 100 ML Exit trades for timing analysis
print("Analyzing sample of 100 ML Exit trades...")
sample_ml = ml_exits.sample(min(100, len(ml_exits)), random_state=42)

timing_analysis = []
for _, trade in sample_ml.iterrows():
    result = analyze_exit_timing(trade, df_features)
    if result:
        timing_analysis.append(result)

if timing_analysis:
    df_timing = pd.DataFrame(timing_analysis)

    print(f"Sample Size: {len(df_timing)}")
    print()
    print("Exit Timing Quality:")
    print(f"  Avg Max Profit Available: {df_timing['max_pnl'].mean()*100:.2f}%")
    print(f"  Avg Actual Exit P&L: {df_timing['actual_pnl'].mean()*100:.2f}%")
    print(f"  Avg Missed Profit: {df_timing['missed_profit'].mean()*100:.2f}%")
    print()
    print(f"  Trades exiting at loss: {(df_timing['actual_pnl'] < 0).sum()} ({(df_timing['actual_pnl'] < 0).sum()/len(df_timing)*100:.1f}%)")
    print(f"  Trades with profit available: {(df_timing['max_pnl'] > 0.02).sum()} ({(df_timing['max_pnl'] > 0.02).sum()/len(df_timing)*100:.1f}%)")
    print()

# ============================================================================
# PART 4: Alternative Exit Labeling Strategies
# ============================================================================
print("=" * 80)
print("PART 4: ALTERNATIVE EXIT LABELING STRATEGIES")
print("=" * 80)
print()

print("Current Strategy (FAILED):")
print("  Label = 1 if profit target (2%) hit within 60 candles")
print("  Result: 15.2% WR, -69.10% return")
print()
print("-" * 80)
print()

print("ALTERNATIVE STRATEGIES TO TEST:")
print()

print("1. OPTIMAL EXIT TIMING")
print("   Label = 1 at the candle with MAXIMUM profit")
print("   Rationale: Train model to recognize optimal exit point")
print("   Expected: Higher win rate, better profit capture")
print()

print("2. PROFIT THRESHOLD WITH DRAWDOWN PROTECTION")
print("   Label = 1 if:")
print("   - Currently profitable (>0.5% leveraged)")
print("   - AND max drawdown ahead < -1%")
print("   Rationale: Exit when profit at risk")
print("   Expected: Protect profits, avoid giving back gains")
print()

print("3. RELATIVE STRENGTH EXIT")
print("   Label = 1 if:")
print("   - Current profit in top 70% of future max profit")
print("   - OR price momentum reversing")
print("   Rationale: Exit when 70% of potential captured")
print("   Expected: Balance profit capture vs early exit")
print()

print("4. PRODUCTION-STYLE (OPPORTUNITY COST)")
print("   Label = 1 if:")
print("   - Holding longer has negative expected value")
print("   - Based on empirical hold time vs profit relationship")
print("   Rationale: Matches Production's approach")
print("   Expected: Closer to Production performance")
print()

print("5. MULTI-CONDITION EXIT")
print("   Label = 1 if ANY:")
print("   - Profit target hit (2%)")
print("   - Momentum reversal with profit >0.5%")
print("   - Hold time >30 candles with profit >1%")
print("   Rationale: Multiple valid exit conditions")
print("   Expected: More realistic exit scenarios")
print()

# ============================================================================
# PART 5: Comparison with Production Exit Models
# ============================================================================
print("=" * 80)
print("PART 5: COMPARISON WITH PRODUCTION EXIT MODELS")
print("=" * 80)
print()

# Load Production Exit models for comparison
prod_long_exit = MODELS_DIR / "xgboost_long_exit_oppgating_improved_20251024_043527.pkl"
prod_short_exit = MODELS_DIR / "xgboost_short_exit_oppgating_improved_20251024_044510.pkl"

print("Production Exit Models:")
print(f"  LONG: {prod_long_exit.name}")
print(f"  SHORT: {prod_short_exit.name}")
print()

print("Production Performance (108 windows):")
print("  Win Rate: 73.86%")
print("  Return: +38.04% per 5-day window")
print("  ML Exit: 77.0%")
print("  Stop Loss: ~8%")
print()

print("Full Dataset Performance (19 windows):")
print("  Win Rate: 15.2%")
print("  Return: -69.10% per 5-day window")
print("  ML Exit: 98.5%")
print("  Stop Loss: 0.6%")
print()

print("Performance Gap:")
print(f"  Win Rate: -58.66pp")
print(f"  Return: -107.14pp")
print(f"  ML Exit: +21.5pp (too aggressive)")
print()

# ============================================================================
# SUMMARY AND RECOMMENDATIONS
# ============================================================================
print("=" * 80)
print("SUMMARY AND RECOMMENDATIONS")
print("=" * 80)
print()

print("KEY FINDINGS:")
print()
print("1. Label Distribution:")
print(f"   - LONG: {long_labels.mean()*100:.1f}% Exit labels")
print(f"   - SHORT: {short_labels.mean()*100:.1f}% Exit labels")
print(f"   - Balance seems reasonable (30-40%)")
print()

print("2. Exit Timing Problem:")
print("   - ML Exits trigger 98.5% (vs 77% Production)")
print("   - But only 15.2% win rate (vs 73.86% Production)")
print("   - Models exit too frequently at wrong times")
print()

print("3. Current Labeling Flaw:")
print("   - 'Profit within 60 candles' is too simplistic")
print("   - Doesn't consider:")
print("     * Optimal exit timing (max profit point)")
print("     * Risk of future drawdown")
print("     * Momentum reversal signals")
print("     * Opportunity cost of holding")
print()

print("RECOMMENDED NEXT STEPS:")
print()
print("1. Test Alternative Strategy #1 (Optimal Exit Timing)")
print("   - Label at max profit point")
print("   - Most direct improvement")
print()

print("2. Test Alternative Strategy #3 (Relative Strength)")
print("   - 70% of max profit threshold")
print("   - Balances early/late exit")
print()

print("3. Compare both vs Production performance")
print("   - Target: 70%+ win rate")
print("   - Target: +30%+ return per window")
print()

print("=" * 80)
print("✅ INVESTIGATION COMPLETE")
print("=" * 80)
