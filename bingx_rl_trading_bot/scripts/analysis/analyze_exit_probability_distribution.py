"""
Exit Probability Distribution Analysis
=======================================
Purpose: Understand why ML Exit stops working above threshold 0.30
Method: Analyze actual probability outputs from Exit models on recent data
"""

import pandas as pd
import numpy as np
import pickle
import sys
from pathlib import Path
from datetime import timedelta
import matplotlib.pyplot as plt

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
RESULTS_DIR = BASE_DIR / "results"

# Import prepare_exit_features from production
from scripts.experiments.retrain_exit_models_opportunity_gating import prepare_exit_features

print("=" * 80)
print("EXIT PROBABILITY DISTRIBUTION ANALYSIS")
print("=" * 80)
print()

# Load Exit models
print("Loading Exit models...")
with open(MODELS_DIR / "xgboost_long_exit_oppgating_improved_20251024_043527.pkl", 'rb') as f:
    long_exit_model = pickle.load(f)
with open(MODELS_DIR / "xgboost_long_exit_oppgating_improved_20251024_043527_features.txt", 'r') as f:
    long_exit_features = [line.strip() for line in f.readlines() if line.strip()]

with open(MODELS_DIR / "xgboost_short_exit_oppgating_improved_20251024_044510.pkl", 'rb') as f:
    short_exit_model = pickle.load(f)
with open(MODELS_DIR / "xgboost_short_exit_oppgating_improved_20251024_044510_features.txt", 'r') as f:
    short_exit_features = [line.strip() for line in f.readlines() if line.strip()]

print(f"✓ LONG Exit: {len(long_exit_features)} features")
print(f"✓ SHORT Exit: {len(short_exit_features)} features")
print()

# Load recent 4 weeks data
print("Loading dataset...")
df_master = pd.read_csv(DATA_DIR / "features" / "BTCUSDT_5m_features.csv")
df_master['timestamp'] = pd.to_datetime(df_master['timestamp'])

cutoff_date = df_master['timestamp'].max() - timedelta(days=28)
df_recent = df_master[df_master['timestamp'] >= cutoff_date].copy().reset_index(drop=True)

# Add enhanced exit features using production function
df_recent = prepare_exit_features(df_recent)

print(f"✓ Recent 4 weeks: {len(df_recent):,} candles")
print()

# Calculate Exit probabilities for all candles
print("Calculating Exit probabilities for all candles...")
print("This will take a moment...")
print()

long_exit_probs = []
short_exit_probs = []

for i in range(len(df_recent)):
    candle = df_recent.iloc[i]

    # LONG Exit probability
    try:
        long_feat = candle[long_exit_features].values.reshape(1, -1)
        long_prob = long_exit_model.predict_proba(long_feat)[0, 1]
        long_exit_probs.append(long_prob)
    except:
        long_exit_probs.append(np.nan)

    # SHORT Exit probability
    try:
        short_feat = candle[short_exit_features].values.reshape(1, -1)
        short_prob = short_exit_model.predict_proba(short_feat)[0, 1]
        short_exit_probs.append(short_prob)
    except:
        short_exit_probs.append(np.nan)

df_recent['long_exit_prob'] = long_exit_probs
df_recent['short_exit_prob'] = short_exit_probs

print("✓ Exit probabilities calculated")
print()

# Analyze distribution
print("=" * 80)
print("EXIT PROBABILITY DISTRIBUTION STATISTICS")
print("=" * 80)
print()

print("LONG Exit Model:")
long_stats = df_recent['long_exit_prob'].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
print(long_stats)
print()

print("SHORT Exit Model:")
short_stats = df_recent['short_exit_prob'].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
print(short_stats)
print()

# Threshold coverage analysis
print("=" * 80)
print("THRESHOLD COVERAGE ANALYSIS")
print("=" * 80)
print()

thresholds = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.70, 0.75]

print("Threshold   LONG >= threshold   SHORT >= threshold")
print("─" * 60)
for threshold in thresholds:
    long_coverage = (df_recent['long_exit_prob'] >= threshold).sum() / len(df_recent) * 100
    short_coverage = (df_recent['short_exit_prob'] >= threshold).sum() / len(df_recent) * 100
    print(f"{threshold:5.2f}       {long_coverage:5.2f}%             {short_coverage:5.2f}%")

print()

# Key findings
print("=" * 80)
print("KEY FINDINGS")
print("=" * 80)
print()

long_max = df_recent['long_exit_prob'].max()
short_max = df_recent['short_exit_prob'].max()
long_99 = df_recent['long_exit_prob'].quantile(0.99)
short_99 = df_recent['short_exit_prob'].quantile(0.99)

print(f"Maximum probabilities:")
print(f"  LONG Exit:  {long_max:.4f}")
print(f"  SHORT Exit: {short_max:.4f}")
print()

print(f"99th percentile (top 1%):")
print(f"  LONG Exit:  {long_99:.4f}")
print(f"  SHORT Exit: {short_99:.4f}")
print()

if long_max < 0.30 or short_max < 0.30:
    print("🔴 CRITICAL ISSUE:")
    print("   Exit models NEVER output probabilities >= 0.30")
    print("   → Threshold 0.30+ will NEVER trigger ML Exit")
    print("   → This explains why ML Exit = 0% for threshold >= 0.30")
    print()

if long_99 < 0.25 or short_99 < 0.25:
    print("⚠️ WARNING:")
    print("   99% of probabilities are below 0.25")
    print("   → Threshold 0.25+ will trigger ML Exit < 1% of time")
    print()

# Recommendation
print("=" * 80)
print("RECOMMENDATION")
print("=" * 80)
print()

if long_max < 0.75 or short_max < 0.75:
    print("📌 Threshold 0.75 is UNREACHABLE for this model")
    print("   → Model probabilities don't go that high")
    print("   → ML Exit will ALWAYS be 0% at this threshold")
    print()
    print("🎯 Practical threshold range:")
    long_practical = long_99
    short_practical = short_99
    print(f"   LONG Exit: 0.10 - {long_practical:.2f} (based on 99th percentile)")
    print(f"   SHORT Exit: 0.10 - {short_practical:.2f} (based on 99th percentile)")
    print()
    print("💡 To use threshold 0.75:")
    print("   → Need to RETRAIN Exit models")
    print("   → Adjust training labels/parameters")
    print("   → Calibrate model to output higher probabilities")

print("=" * 80)

# Save statistics
timestamp = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
output_file = RESULTS_DIR / f"exit_probability_distribution_{timestamp}.csv"

stats_df = pd.DataFrame({
    'threshold': thresholds,
    'long_coverage_pct': [(df_recent['long_exit_prob'] >= t).sum() / len(df_recent) * 100 for t in thresholds],
    'short_coverage_pct': [(df_recent['short_exit_prob'] >= t).sum() / len(df_recent) * 100 for t in thresholds]
})

stats_df.to_csv(output_file, index=False)
print(f"✓ Statistics saved: {output_file.name}")
print()
