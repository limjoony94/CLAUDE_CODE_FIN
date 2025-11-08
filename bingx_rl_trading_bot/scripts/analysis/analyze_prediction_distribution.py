"""
Prediction Probability Distribution Analysis

목적: LONG/SHORT 모델이 실제 데이터에서 어떤 확률 분포를 생성하는지 분석
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.production.train_xgboost_improved_v3_phase2 import calculate_features
from scripts.production.advanced_technical_features import AdvancedTechnicalFeatures

# Load data
DATA_DIR = PROJECT_ROOT / "data" / "historical"
MODELS_DIR = PROJECT_ROOT / "models"

df = pd.read_csv(DATA_DIR / "BTCUSDT_5m_max.csv")

# Calculate features
df = calculate_features(df)
adv_features = AdvancedTechnicalFeatures()
df = adv_features.calculate_all_features(df)
df = df.ffill().dropna()

print("=" * 80)
print("PREDICTION PROBABILITY DISTRIBUTION ANALYSIS")
print("=" * 80)
print(f"Data: {len(df)} candles\n")

# Load models and scalers
with open(MODELS_DIR / 'xgboost_v4_phase4_advanced_lookahead3_thresh0.pkl', 'rb') as f:
    long_model = pickle.load(f)

with open(MODELS_DIR / 'xgboost_v4_phase4_advanced_lookahead3_thresh0_scaler.pkl', 'rb') as f:
    long_scaler = pickle.load(f)

with open(MODELS_DIR / 'xgboost_short_model_lookahead3_thresh0.3.pkl', 'rb') as f:
    short_model = pickle.load(f)

with open(MODELS_DIR / 'xgboost_short_model_lookahead3_thresh0.3_scaler.pkl', 'rb') as f:
    short_scaler = pickle.load(f)

with open(MODELS_DIR / 'xgboost_v4_phase4_advanced_lookahead3_thresh0_features.txt', 'r') as f:
    feature_columns = [line.strip() for line in f]

# Get features
X = df[feature_columns].values

# Predict probabilities
X_long_scaled = long_scaler.transform(X)
X_short_scaled = short_scaler.transform(X)

prob_long = long_model.predict_proba(X_long_scaled)[:, 1]
prob_short = short_model.predict_proba(X_short_scaled)[:, 1]

print("=" * 80)
print("PROBABILITY STATISTICS")
print("=" * 80)
print(f"{'Metric':<25} {'LONG':>15} {'SHORT':>15} {'LONG/SHORT':>15}")
print("-" * 80)
print(f"{'Mean':<25} {prob_long.mean():>15.4f} {prob_short.mean():>15.4f} {prob_long.mean()/prob_short.mean():>15.2f}x")
print(f"{'Median':<25} {np.median(prob_long):>15.4f} {np.median(prob_short):>15.4f} {np.median(prob_long)/np.median(prob_short):>15.2f}x")
print(f"{'Std':<25} {prob_long.std():>15.4f} {prob_short.std():>15.4f} {prob_long.std()/prob_short.std():>15.2f}x")
print(f"{'Min':<25} {prob_long.min():>15.4f} {prob_short.min():>15.4f}")
print(f"{'Max':<25} {prob_long.max():>15.4f} {prob_short.max():>15.4f}")

print()
print("=" * 80)
print("THRESHOLD ANALYSIS (Current: LONG 0.80, SHORT 0.80)")
print("=" * 80)

thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]

print(f"{'Threshold':<12} {'LONG Count':>15} {'LONG %':>10} {'SHORT Count':>15} {'SHORT %':>10} {'Ratio':>10}")
print("-" * 80)

for thresh in thresholds:
    long_count = (prob_long >= thresh).sum()
    short_count = (prob_short >= thresh).sum()
    long_pct = long_count / len(prob_long) * 100
    short_pct = short_count / len(prob_short) * 100
    ratio = long_count / short_count if short_count > 0 else float('inf')

    marker = " ← CURRENT" if thresh == 0.80 else ""
    print(f"{thresh:<12.2f} {long_count:>15} {long_pct:>10.2f} {short_count:>15} {short_pct:>10.2f} {ratio:>10.2f}x{marker}")

print()
print("=" * 80)
print("PERCENTILE ANALYSIS")
print("=" * 80)

percentiles = [50, 75, 90, 95, 99]

print(f"{'Percentile':<12} {'LONG':>15} {'SHORT':>15}")
print("-" * 80)
for p in percentiles:
    long_p = np.percentile(prob_long, p)
    short_p = np.percentile(prob_short, p)
    print(f"{p:<12}th {long_p:>15.4f} {short_p:>15.4f}")

# Critical insight: What causes low SHORT probability?
print()
print("=" * 80)
print("CRITICAL INSIGHT: Why LOW SHORT Probability?")
print("=" * 80)

# Analyze feature values when SHORT probability is high vs low
df['prob_long'] = prob_long
df['prob_short'] = prob_short

# High SHORT prob samples
high_short = df[df['prob_short'] >= 0.70].copy()
low_short = df[df['prob_short'] < 0.10].copy()

if len(high_short) > 0 and len(low_short) > 0:
    print(f"\nHigh SHORT prob samples (>= 0.70): {len(high_short)} ({len(high_short)/len(df)*100:.2f}%)")
    print(f"Low SHORT prob samples (< 0.10): {len(low_short)} ({len(low_short)/len(df)*100:.2f}%)")
    print()

    # Top features for SHORT model
    top_short_features = ['close_change_1', 'price_vs_upper_trendline_pct', 'num_resistance_touches',
                          'price_vs_lower_trendline_pct', 'distance_to_support_pct']

    print("Feature values comparison (High SHORT vs Low SHORT):")
    print(f"{'Feature':<35} {'High SHORT':>15} {'Low SHORT':>15} {'Difference':>15}")
    print("-" * 85)

    for feat in top_short_features:
        if feat in df.columns:
            high_val = high_short[feat].mean()
            low_val = low_short[feat].mean()
            diff = high_val - low_val
            print(f"{feat:<35} {high_val:>15.4f} {low_val:>15.4f} {diff:>15.4f}")

print()
print("=" * 80)
print("CONCLUSION")
print("=" * 80)
print("If LONG probability is consistently higher than SHORT:")
print("  → Model learned from feature distributions")
print("  → Feature engineering may favor LONG detection")
print("  → Structural bias in technical indicators")
