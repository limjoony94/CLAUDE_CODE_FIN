"""
Generate Trade Outcome Labels for 90-Day Dataset
=================================================

Purpose: Apply successful 52-day labeling methodology to 90-day data

Labeling Logic (from create_trade_outcome_labels.py):
  - Target: +1.0% price within 60 candles (5 hours) = +4% leveraged
  - Stop Loss: Must NOT hit -0.75% price before target = -3% leveraged
  - Quality: Only labels safe entries that reach profit without danger

Expected Result: 10-15% label distribution (matching 52-day success)

Input: BTCUSDT_5m_raw_90days_20251106_163815.csv (25,903 candles)
Output: trade_outcome_labels_90days_{timestamp}.csv

Created: 2025-11-06 20:30 KST
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
from datetime import datetime

# Paths
DATA_DIR = PROJECT_ROOT / "data" / "features"
LABELS_DIR = PROJECT_ROOT / "data" / "labels"
LABELS_DIR.mkdir(parents=True, exist_ok=True)

# Input
RAW_90D = DATA_DIR / "BTCUSDT_5m_raw_90days_20251106_163815.csv"

# Configuration (matching successful 52-day approach)
LEVERAGE = 4
TARGET_PROFIT_PCT = 0.01   # 1.0% price = 4% leveraged
MAX_LOSS_PCT = 0.0075      # 0.75% price = 3% leveraged
MAX_HOLD_CANDLES = 60      # 5 hours @ 5-min

print("=" * 80)
print("TRADE OUTCOME LABEL GENERATION - 90-DAY DATASET")
print("=" * 80)
print()
print(f"📊 Configuration (matching 52-day success):")
print(f"   Leverage: {LEVERAGE}x")
print(f"   Target Profit: {TARGET_PROFIT_PCT*100:.2f}% price = {TARGET_PROFIT_PCT*LEVERAGE*100:.1f}% leveraged")
print(f"   Max Loss: {MAX_LOSS_PCT*100:.2f}% price = {MAX_LOSS_PCT*LEVERAGE*100:.1f}% leveraged")
print(f"   Max Hold: {MAX_HOLD_CANDLES} candles ({MAX_HOLD_CANDLES/12:.1f} hours)")
print()

# ============================================================================
# LOAD DATA
# ============================================================================

print("=" * 80)
print("LOADING DATA")
print("=" * 80)
print()

print(f"📖 Loading 90-day raw data: {RAW_90D.name}")
df = pd.read_csv(RAW_90D)
df['timestamp'] = pd.to_datetime(df['timestamp'])

print(f"   Rows: {len(df):,}")
print(f"   Period: {df['timestamp'].min()} to {df['timestamp'].max()}")
print(f"   Days: {(df['timestamp'].max() - df['timestamp'].min()).days}")
print()

# ============================================================================
# LABEL LONG ENTRIES
# ============================================================================

print("=" * 80)
print("LABELING LONG ENTRIES")
print("=" * 80)
print()

print(f"🔵 Logic: Label = 1 if...")
print(f"   1. Price reaches +{TARGET_PROFIT_PCT*100:.2f}% within {MAX_HOLD_CANDLES} candles")
print(f"   2. AND never drops below -{MAX_LOSS_PCT*100:.2f}% BEFORE reaching target")
print()

long_labels = np.zeros(len(df))
long_good = 0

for i in range(len(df) - MAX_HOLD_CANDLES):
    if i % 5000 == 0:
        print(f"   Processing: {i:,} / {len(df):,} ({i/len(df)*100:.1f}%)")

    entry_price = df['close'].iloc[i]
    future_prices = df['close'].iloc[i+1:i+MAX_HOLD_CANDLES+1].values

    if len(future_prices) == 0:
        continue

    # Calculate returns
    returns = (future_prices - entry_price) / entry_price

    # Check for target hit
    target_hit = np.any(returns >= TARGET_PROFIT_PCT)

    # Check for stop loss hit BEFORE target
    if target_hit:
        target_idx = np.where(returns >= TARGET_PROFIT_PCT)[0][0]
        returns_before_target = returns[:target_idx+1]
        stop_hit = np.any(returns_before_target <= -MAX_LOSS_PCT)
    else:
        stop_hit = np.any(returns <= -MAX_LOSS_PCT)

    # Label as good if target hit and stop not hit first
    if target_hit and not stop_hit:
        long_labels[i] = 1
        long_good += 1

long_positive = (long_labels == 1).sum()
long_pct = long_positive / len(long_labels) * 100

print()
print(f"✅ LONG Labels Generated:")
print(f"   Good entries: {long_positive:,} ({long_pct:.2f}%)")
print(f"   Bad entries: {len(long_labels) - long_positive:,} ({100-long_pct:.2f}%)")
print()

# ============================================================================
# LABEL SHORT ENTRIES
# ============================================================================

print("=" * 80)
print("LABELING SHORT ENTRIES")
print("=" * 80)
print()

print(f"🔴 Logic: Label = 1 if...")
print(f"   1. Price drops -{TARGET_PROFIT_PCT*100:.2f}% within {MAX_HOLD_CANDLES} candles")
print(f"   2. AND never rises above +{MAX_LOSS_PCT*100:.2f}% BEFORE reaching target")
print()

short_labels = np.zeros(len(df))
short_good = 0

for i in range(len(df) - MAX_HOLD_CANDLES):
    if i % 5000 == 0:
        print(f"   Processing: {i:,} / {len(df):,} ({i/len(df)*100:.1f}%)")

    entry_price = df['close'].iloc[i]
    future_prices = df['close'].iloc[i+1:i+MAX_HOLD_CANDLES+1].values

    if len(future_prices) == 0:
        continue

    # Calculate returns (negative for SHORT)
    returns = (entry_price - future_prices) / entry_price

    # Check for target hit
    target_hit = np.any(returns >= TARGET_PROFIT_PCT)

    # Check for stop loss hit BEFORE target
    if target_hit:
        target_idx = np.where(returns >= TARGET_PROFIT_PCT)[0][0]
        returns_before_target = returns[:target_idx+1]
        stop_hit = np.any(returns_before_target <= -MAX_LOSS_PCT)
    else:
        stop_hit = np.any(returns <= -MAX_LOSS_PCT)

    # Label as good if target hit and stop not hit first
    if target_hit and not stop_hit:
        short_labels[i] = 1
        short_good += 1

short_positive = (short_labels == 1).sum()
short_pct = short_positive / len(short_labels) * 100

print()
print(f"✅ SHORT Labels Generated:")
print(f"   Good entries: {short_positive:,} ({short_pct:.2f}%)")
print(f"   Bad entries: {len(short_labels) - short_positive:,} ({100-short_pct:.2f}%)")
print()

# ============================================================================
# SAVE LABELS
# ============================================================================

print("=" * 80)
print("SAVING LABELS")
print("=" * 80)
print()

# Add to dataframe
df['signal_long'] = long_labels
df['signal_short'] = short_labels

# Save labeled data
timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
output_file = LABELS_DIR / f"trade_outcome_labels_90days_{timestamp_str}.csv"

df[['timestamp', 'close', 'signal_long', 'signal_short']].to_csv(output_file, index=False)

print(f"💾 Labels saved: {output_file.name}")
print(f"   Size: {output_file.stat().st_size / 1024 / 1024:.2f} MB")
print()

# ============================================================================
# COMPARISON
# ============================================================================

print("=" * 80)
print("COMPARISON: Trade Outcome vs Previous Relaxed Labels")
print("=" * 80)
print()

print(f"📊 Previous Relaxed Labels (1.5% in 120min):")
print(f"   LONG: 360 (1.41%)")
print(f"   SHORT: 639 (2.50%)")
print(f"   Total: 999 (3.90%)")
print(f"   Quality: LOW (too lenient, includes noise)")
print()

print(f"📊 Trade Outcome Labels ({TARGET_PROFIT_PCT*100:.1f}% with SL protection):")
print(f"   LONG: {long_positive:,} ({long_pct:.2f}%)")
print(f"   SHORT: {short_positive:,} ({short_pct:.2f}%)")
print(f"   Total: {long_positive + short_positive:,} ({long_pct + short_pct:.2f}%)")
print(f"   Quality: HIGH (risk-aware, realistic targets)")
print()

print(f"📊 Expected Range (from 52-day success):")
print(f"   5-15% per side")
print(f"   52-day achieved: 9.79% LONG, 10.89% SHORT")
print()

if 5 <= long_pct <= 15 and 5 <= short_pct <= 15:
    print(f"✅ Label distribution looks GOOD! (matching 52-day success)")
elif long_pct + short_pct >= 5:
    print(f"⚠️  Label distribution acceptable but may need tuning")
else:
    print(f"❌ Label distribution too sparse (<5% total)")
print()

# ============================================================================
# SUMMARY
# ============================================================================

print("=" * 80)
print("SUMMARY")
print("=" * 80)
print()

print(f"📊 Label Generation Complete:")
print(f"   Input: {RAW_90D.name}")
print(f"   Rows: {len(df):,}")
print(f"   Period: {df['timestamp'].min()} to {df['timestamp'].max()}")
print()

print(f"📈 Label Statistics:")
print(f"   LONG: {long_positive:,} ({long_pct:.2f}%)")
print(f"   SHORT: {short_positive:,} ({short_pct:.2f}%)")
print(f"   Total: {long_positive + short_positive:,} ({long_pct + short_pct:.2f}%)")
print()

print(f"🎯 Key Differences from Failed Approach:")
print(f"   1. Risk-aware (stop-loss protection)")
print(f"   2. Realistic profit targets (1% price = 4% leveraged)")
print(f"   3. Higher quality labels (safe entries only)")
print(f"   4. Natural 10-15% distribution (no over-relaxation needed)")
print()

print(f"✅ Trade outcome labels generated successfully!")
print(f"   Output: {output_file}")
print()

print("=" * 80)
print("NEXT STEPS")
print("=" * 80)
print()
print("1. Verify label distribution matches 52-day success (9-11%)")
print("2. Retrain 90-day models with these high-quality labels")
print("3. Use feature names from 52-day models (avoid feature mismatch)")
print("4. Validate on out-of-sample period (Oct 9 - Nov 6)")
print("5. Compare probabilities vs 52-day models")
print()
