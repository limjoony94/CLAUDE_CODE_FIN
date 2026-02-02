"""
Breakthrough Results - Deep Analysis

체계적 분석:
1. Trade frequency vs precision trade-off
2. Threshold sensitivity analysis
3. Model confidence distribution
4. Exit reason breakdown
5. Performance consistency
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import matplotlib.pyplot as plt
import sys

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.production.train_xgboost_improved_v3_phase2 import calculate_features
from scripts.production.advanced_technical_features import AdvancedTechnicalFeatures
from src.features.sell_signal_features import SellSignalFeatures

DATA_DIR = PROJECT_ROOT / "data" / "historical"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

print("=" * 80)
print("Breakthrough Results - Deep Analysis")
print("=" * 80)

# Load models
print("\n1. Loading Models...")
long_entry_file = MODELS_DIR / "xgboost_v4_phase4_advanced_lookahead3_thresh0.pkl"
long_entry_scaler_file = MODELS_DIR / "xgboost_v4_phase4_advanced_lookahead3_thresh0_scaler.pkl"
with open(long_entry_file, 'rb') as f:
    long_entry_model = pickle.load(f)
with open(long_entry_scaler_file, 'rb') as f:
    long_entry_scaler = pickle.load(f)

short_entry_file = MODELS_DIR / "xgboost_short_peak_trough_20251016_131939.pkl"
short_entry_scaler_file = MODELS_DIR / "xgboost_short_peak_trough_20251016_131939_scaler.pkl"
with open(short_entry_file, 'rb') as f:
    short_entry_model = pickle.load(f)
with open(short_entry_scaler_file, 'rb') as f:
    short_entry_scaler = pickle.load(f)

# Load features
long_entry_feature_file = MODELS_DIR / "xgboost_v4_phase4_advanced_lookahead3_thresh0_features.txt"
with open(long_entry_feature_file, 'r') as f:
    long_entry_features = [line.strip() for line in f.readlines()]

short_entry_feature_file = MODELS_DIR / "xgboost_short_peak_trough_20251016_131939_features.txt"
with open(short_entry_feature_file, 'r') as f:
    short_entry_features = [line.strip() for line in f.readlines()]

print("✅ Models loaded")

# Load data
print("\n2. Loading Data...")
data_file = DATA_DIR / "BTCUSDT_5m_max.csv"
df = pd.read_csv(data_file)
print(f"✅ Loaded {len(df):,} candles")

# Calculate features
print("\n3. Calculating Features...")
df = calculate_features(df)
adv_features = AdvancedTechnicalFeatures(lookback_sr=50, lookback_trend=20)
df = adv_features.calculate_all_features(df)
sell_features = SellSignalFeatures()
df = sell_features.calculate_all_features(df)
df = df.ffill().dropna()
print(f"✅ {len(df):,} candles with features")

# Calculate probabilities
print("\n4. Calculating Model Probabilities...")

# LONG Entry
long_entry_row = df[long_entry_features].values
long_features_scaled = long_entry_scaler.transform(long_entry_row)
long_probs = long_entry_model.predict_proba(long_features_scaled)[:, 1]

# SHORT Entry
short_entry_row = df[short_entry_features].values
short_features_scaled = short_entry_scaler.transform(short_entry_row)
short_probs = short_entry_model.predict_proba(short_features_scaled)[:, 1]

print("✅ Probabilities calculated")

# Analysis 1: Signal Distribution
print("\n" + "=" * 80)
print("ANALYSIS 1: Signal Distribution")
print("=" * 80)

thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]

print("\nLONG Entry Model:")
for thresh in thresholds:
    count = np.sum(long_probs >= thresh)
    pct = count / len(long_probs) * 100
    print(f"  Prob >= {thresh:.1f}: {count:5,} ({pct:5.2f}%)")

print("\nSHORT Entry Model (Breakthrough):")
for thresh in thresholds:
    count = np.sum(short_probs >= thresh)
    pct = count / len(short_probs) * 100
    print(f"  Prob >= {thresh:.1f}: {count:5,} ({pct:5.2f}%)")

# Analysis 2: Trade Frequency Estimation
print("\n" + "=" * 80)
print("ANALYSIS 2: Expected Trade Frequency")
print("=" * 80)

# 5-day windows
WINDOW_SIZE = 1440
num_windows = len(df) // WINDOW_SIZE

print(f"\nTotal candles: {len(df):,}")
print(f"Window size: {WINDOW_SIZE} (5 days)")
print(f"Number of windows: {num_windows}")

for thresh in thresholds:
    long_signals = np.sum(long_probs >= thresh)
    short_signals = np.sum(short_probs >= thresh)
    total_signals = long_signals + short_signals

    signals_per_window = total_signals / num_windows
    signals_per_day = signals_per_window / 5

    print(f"\nThreshold {thresh:.1f}:")
    print(f"  LONG signals: {long_signals:,}")
    print(f"  SHORT signals: {short_signals:,}")
    print(f"  Total: {total_signals:,}")
    print(f"  Per window (5d): {signals_per_window:.1f}")
    print(f"  Per day: {signals_per_day:.1f}")

# Analysis 3: Threshold Sensitivity
print("\n" + "=" * 80)
print("ANALYSIS 3: Trade Frequency vs Precision Trade-off")
print("=" * 80)

# Historical precision from training
LONG_ENTRY_PRECISION = 0.702  # 70.2%
SHORT_ENTRY_PRECISION = 0.552  # 55.2%

print("\nExpected win rates at different thresholds:")
print("(Note: Higher threshold → fewer trades but higher precision)")

for thresh in thresholds:
    # Simplified assumption: precision increases with threshold
    # This is an approximation - actual precision would need validation
    long_signals = np.sum(long_probs >= thresh)
    short_signals = np.sum(short_probs >= thresh)
    total_signals = long_signals + short_signals

    signals_per_day = (total_signals / num_windows) / 5

    # Estimate: threshold 0.7 = base precision, higher threshold = higher precision
    precision_boost = (thresh - 0.7) * 0.2  # +20% per 0.1 threshold increase
    estimated_long_precision = min(0.95, LONG_ENTRY_PRECISION + precision_boost)
    estimated_short_precision = min(0.95, SHORT_ENTRY_PRECISION + precision_boost)

    print(f"\nThreshold {thresh:.1f}:")
    print(f"  Trades/day: {signals_per_day:.2f}")
    print(f"  Est. LONG precision: {estimated_long_precision*100:.1f}%")
    print(f"  Est. SHORT precision: {estimated_short_precision*100:.1f}%")

# Analysis 4: Probability Distribution
print("\n" + "=" * 80)
print("ANALYSIS 4: Probability Distribution Statistics")
print("=" * 80)

print("\nLONG Entry:")
print(f"  Mean: {np.mean(long_probs):.4f}")
print(f"  Median: {np.median(long_probs):.4f}")
print(f"  Std: {np.std(long_probs):.4f}")
print(f"  Min: {np.min(long_probs):.4f}")
print(f"  Max: {np.max(long_probs):.4f}")

print("\nSHORT Entry:")
print(f"  Mean: {np.mean(short_probs):.4f}")
print(f"  Median: {np.median(short_probs):.4f}")
print(f"  Std: {np.std(short_probs):.4f}")
print(f"  Min: {np.min(short_probs):.4f}")
print(f"  Max: {np.max(short_probs):.4f}")

# Analysis 5: Signal Conflicts
print("\n" + "=" * 80)
print("ANALYSIS 5: Signal Conflicts (LONG vs SHORT)")
print("=" * 80)

for thresh in [0.5, 0.6, 0.7]:
    long_signals = long_probs >= thresh
    short_signals = short_probs >= thresh

    conflicts = np.sum(long_signals & short_signals)
    only_long = np.sum(long_signals & ~short_signals)
    only_short = np.sum(~long_signals & short_signals)
    no_signal = np.sum(~long_signals & ~short_signals)

    total = len(long_probs)

    print(f"\nThreshold {thresh:.1f}:")
    print(f"  Only LONG: {only_long:,} ({only_long/total*100:.2f}%)")
    print(f"  Only SHORT: {only_short:,} ({only_short/total*100:.2f}%)")
    print(f"  Both (conflict): {conflicts:,} ({conflicts/total*100:.2f}%)")
    print(f"  No signal: {no_signal:,} ({no_signal/total*100:.2f}%)")

# Analysis 6: Comparison with Previous System
print("\n" + "=" * 80)
print("ANALYSIS 6: Comparison with Previous Findings")
print("=" * 80)

print("\nPrevious System (2025-10-14):")
print("  Expected: 21 trades/week")
print("  Actual: ~2.3 trades/week (89% below expected)")
print("  Issue: Trade frequency too low")

current_thresh_0_7 = (np.sum(long_probs >= 0.7) + np.sum(short_probs >= 0.7)) / num_windows / 5
trades_per_week = current_thresh_0_7 * 7

print(f"\nCurrent System (Breakthrough, threshold 0.7):")
print(f"  Expected: {trades_per_week:.1f} trades/week")
print(f"  Trade frequency: {current_thresh_0_7:.2f} trades/day")

if trades_per_week < 10:
    print("  ⚠️ WARNING: Trade frequency still low!")
    print("  Recommendation: Consider threshold 0.6 or 0.5")
elif trades_per_week < 21:
    print("  ✅ Acceptable: Trade frequency improved but still below target")
    print("  Recommendation: Fine-tune threshold between 0.6-0.7")
else:
    print("  ✅ GOOD: Trade frequency meets expectations")

# Analysis 7: Recommendations
print("\n" + "=" * 80)
print("RECOMMENDATIONS")
print("=" * 80)

# Calculate optimal threshold
optimal_thresh = None
for thresh in [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]:
    signals_per_day = (np.sum(long_probs >= thresh) + np.sum(short_probs >= thresh)) / num_windows / 5
    trades_per_week = signals_per_day * 7

    # Target: 15-25 trades/week (3 per day on average)
    if 15 <= trades_per_week <= 25:
        optimal_thresh = thresh
        break

if optimal_thresh:
    print(f"\n1. THRESHOLD OPTIMIZATION:")
    print(f"   Recommended threshold: {optimal_thresh:.2f}")
    print(f"   Expected: {trades_per_week:.1f} trades/week")
    print(f"   Rationale: Balance between frequency and precision")
else:
    print(f"\n1. THRESHOLD OPTIMIZATION:")
    print(f"   Current threshold 0.7 produces {trades_per_week:.1f} trades/week")
    if trades_per_week < 10:
        print(f"   ⚠️ Recommendation: Lower to 0.5-0.6 for more trades")
    else:
        print(f"   ✅ Current threshold acceptable")

print(f"\n2. SHORT EXIT MODEL:")
print(f"   Status: Still using old model (34.9% precision)")
print(f"   Action: Retrain with Peak/Trough labeling")
print(f"   Expected: 50-60% precision (like LONG Exit)")

print(f"\n3. VALIDATION:")
print(f"   Action: Run extended testnet period")
print(f"   Duration: 1 week minimum")
print(f"   Monitor: Win rates match backtest expectations")

print(f"\n4. PRODUCTION DEPLOYMENT:")
print(f"   Step 1: Deploy SHORT Entry + LONG Exit breakthrough models")
print(f"   Step 2: Monitor for 1 week")
print(f"   Step 3: Deploy retrained SHORT Exit")
print(f"   Step 4: Full 4-model system validation")

print("\n" + "=" * 80)
print("Analysis Complete!")
print("=" * 80)
