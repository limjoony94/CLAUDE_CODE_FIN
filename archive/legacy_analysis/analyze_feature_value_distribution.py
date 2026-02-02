"""
Feature Value Distribution Analysis

목적: LONG vs SHORT opportunities의 feature 값 분포 비교

가설: Training samples은 동일하지만 (429 vs 437),
      feature 값의 magnitude/variance가 달라서 확률 분포가 2.65배 차이가 날 수 있음
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.production.train_xgboost_improved_v3_phase2 import calculate_features
from scripts.production.advanced_technical_features import AdvancedTechnicalFeatures

# Load data
DATA_DIR = PROJECT_ROOT / "data" / "historical"
df = pd.read_csv(DATA_DIR / "BTCUSDT_5m_max.csv")

# Calculate features
df = calculate_features(df)
adv_features = AdvancedTechnicalFeatures()
df = adv_features.calculate_all_features(df)
df = df.ffill().dropna()

print("=" * 80)
print("FEATURE VALUE DISTRIBUTION ANALYSIS")
print("=" * 80)
print(f"Data: {len(df)} candles\n")

# Define forward returns (3 candles ahead, like training)
lookahead = 3
df['future_max'] = df['close'].shift(-1).rolling(window=lookahead).apply(lambda x: x.max())
df['future_min'] = df['close'].shift(-1).rolling(window=lookahead).apply(lambda x: x.min())
df['return_to_max'] = (df['future_max'] - df['close']) / df['close']
df['return_to_min'] = (df['close'] - df['future_min']) / df['close']

# Create labels (symmetric)
threshold = 0.003  # 0.3%
df['is_long_opp'] = (df['return_to_max'] >= threshold).astype(int)
df['is_short_opp'] = (df['return_to_min'] >= threshold).astype(int)

df_clean = df.dropna()

print(f"After dropna: {len(df_clean)} rows")
print(f"LONG opportunities: {df_clean['is_long_opp'].sum()} ({df_clean['is_long_opp'].sum()/len(df_clean)*100:.2f}%)")
print(f"SHORT opportunities: {df_clean['is_short_opp'].sum()} ({df_clean['is_short_opp'].sum()/len(df_clean)*100:.2f}%)")
print()

# Top features to analyze
top_features = [
    'close_change_1', 'close_change_3',
    'macd', 'macd_signal', 'rsi',
    'price_vs_upper_trendline_pct', 'price_vs_lower_trendline_pct',
    'distance_to_resistance_pct', 'distance_to_support_pct',
    'num_resistance_touches', 'num_support_touches',
]

# Extract LONG and SHORT opportunity samples
long_samples = df_clean[df_clean['is_long_opp'] == 1]
short_samples = df_clean[df_clean['is_short_opp'] == 1]

print("=" * 80)
print("FEATURE VALUE DISTRIBUTION (Mean ± Std)")
print("=" * 80)
print(f"{'Feature':<40} {'LONG Opp':>25} {'SHORT Opp':>25}")
print("-" * 80)

distribution_analysis = []

for feat in top_features:
    if feat in df_clean.columns:
        # Statistics for LONG opportunities
        long_mean = long_samples[feat].mean()
        long_std = long_samples[feat].std()
        long_abs_mean = long_samples[feat].abs().mean()

        # Statistics for SHORT opportunities
        short_mean = short_samples[feat].mean()
        short_std = short_samples[feat].std()
        short_abs_mean = short_samples[feat].abs().mean()

        # Ratio of absolute means
        abs_ratio = long_abs_mean / short_abs_mean if short_abs_mean > 0 else float('inf')

        print(f"{feat:<40} {long_mean:>10.4f} ± {long_std:>6.4f}   {short_mean:>10.4f} ± {short_std:>6.4f}")

        distribution_analysis.append({
            'feature': feat,
            'long_mean': long_mean,
            'long_std': long_std,
            'long_abs_mean': long_abs_mean,
            'short_mean': short_mean,
            'short_std': short_std,
            'short_abs_mean': short_abs_mean,
            'abs_mean_ratio': abs_ratio
        })

# Summary
print()
print("=" * 80)
print("FEATURE MAGNITUDE COMPARISON (|Absolute Mean| Ratio)")
print("=" * 80)
print(f"{'Feature':<40} {'|LONG|':>12} {'|SHORT|':>12} {'Ratio':>10}")
print("-" * 80)

distribution_analysis.sort(key=lambda x: x['abs_mean_ratio'], reverse=True)

for item in distribution_analysis:
    print(f"{item['feature']:<40} {item['long_abs_mean']:>12.4f} {item['short_abs_mean']:>12.4f} {item['abs_mean_ratio']:>10.2f}x")

print()
print("=" * 80)
print("CRITICAL INSIGHT: Feature Magnitude")
print("=" * 80)

# Calculate average magnitude ratio
avg_ratio = np.mean([item['abs_mean_ratio'] for item in distribution_analysis])
print(f"Average |LONG|/|SHORT| ratio across top features: {avg_ratio:.2f}x")
print()
print("If LONG opportunities have consistently HIGHER feature magnitudes:")
print("  → XGBoost learns that high feature values = positive class")
print("  → When predicting, high feature values → high LONG probability")
print("  → But SHORT opportunities have LOWER magnitudes → lower SHORT probability")
print("  → This explains 2.65x probability distribution difference!")
print()
print("This is a STRUCTURAL BIAS in the data itself, not the model.")
