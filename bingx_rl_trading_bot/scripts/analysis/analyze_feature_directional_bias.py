"""
Feature Directional Bias Analysis

목적: 각 feature가 상승/하락 중 어느 방향을 더 잘 감지하는지 분석
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
print("FEATURE DIRECTIONAL BIAS ANALYSIS")
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

# Top features to analyze (2025-10-15: Include ALL advanced features + SHORT-specific)
top_features = [
    # Basic features
    'close_change_1', 'close_change_3',
    'macd', 'macd_signal', 'rsi',
    'bb_high', 'bb_low',

    # Support/Resistance
    'distance_to_resistance_pct', 'distance_to_support_pct',
    'num_resistance_touches', 'num_support_touches',

    # Trend Lines
    'price_vs_upper_trendline_pct', 'price_vs_lower_trendline_pct',
    'lower_trendline_slope', 'upper_trendline_slope',

    # Chart Patterns
    'double_top', 'double_bottom',
    'higher_highs_lows', 'lower_highs_lows',

    # Candlestick Patterns
    'hammer', 'shooting_star',
    'bullish_engulfing', 'bearish_engulfing',

    # SHORT-specific Features (2025-10-15: Feature Bias Fix)
    'distance_from_recent_high_pct',
    'bearish_candle_count',
    'red_candle_volume_ratio',
    'strong_selling_pressure',
    'price_momentum_near_resistance',
    'rsi_from_recent_peak',
    'consecutive_up_candles'
]

print("=" * 80)
print("FEATURE DIRECTIONAL CORRELATION")
print("=" * 80)
print(f"{'Feature':<40} {'LONG Corr':>12} {'SHORT Corr':>12} {'Bias':>10}")
print("-" * 80)

feature_bias_analysis = []

for feat in top_features:
    if feat in df_clean.columns:
        # Correlation with opportunities
        long_corr = df_clean[feat].corr(df_clean['is_long_opp'])
        short_corr = df_clean[feat].corr(df_clean['is_short_opp'])

        # Determine bias
        if abs(long_corr) > abs(short_corr) * 1.5:
            bias = "LONG >>"
        elif abs(short_corr) > abs(long_corr) * 1.5:
            bias = "SHORT >>"
        elif abs(long_corr) > abs(short_corr) * 1.2:
            bias = "LONG >"
        elif abs(short_corr) > abs(long_corr) * 1.2:
            bias = "SHORT >"
        else:
            bias = "Neutral"

        print(f"{feat:<40} {long_corr:>12.4f} {short_corr:>12.4f} {bias:>10}")

        feature_bias_analysis.append({
            'feature': feat,
            'long_corr': long_corr,
            'short_corr': short_corr,
            'bias': bias,
            'abs_diff': abs(long_corr) - abs(short_corr)
        })

# Summary
print()
print("=" * 80)
print("SUMMARY: Feature Bias Distribution")
print("=" * 80)

bias_counts = {}
for item in feature_bias_analysis:
    bias = item['bias']
    bias_counts[bias] = bias_counts.get(bias, 0) + 1

for bias, count in sorted(bias_counts.items()):
    print(f"  {bias:15s}: {count} features")

# Most biased features
print()
print("=" * 80)
print("TOP 5 LONG-BIASED FEATURES")
print("=" * 80)
long_biased = sorted(feature_bias_analysis, key=lambda x: x['abs_diff'], reverse=True)[:5]
for item in long_biased:
    print(f"  {item['feature']:40s} (LONG {item['long_corr']:+.4f} vs SHORT {item['short_corr']:+.4f})")

print()
print("=" * 80)
print("TOP 5 SHORT-BIASED FEATURES")
print("=" * 80)
short_biased = sorted(feature_bias_analysis, key=lambda x: x['abs_diff'])[:5]
for item in short_biased:
    print(f"  {item['feature']:40s} (SHORT {item['short_corr']:+.4f} vs LONG {item['long_corr']:+.4f})")

# CRITICAL: Check if LONG opportunities are simply more frequent
print()
print("=" * 80)
print("CRITICAL INSIGHT: Opportunity Frequency")
print("=" * 80)
print(f"LONG opportunities: {df_clean['is_long_opp'].sum()} samples")
print(f"SHORT opportunities: {df_clean['is_short_opp'].sum()} samples")
print(f"Ratio (LONG/SHORT): {df_clean['is_long_opp'].sum() / df_clean['is_short_opp'].sum():.2f}x")
print()
print("If LONG opportunities >> SHORT opportunities in training data,")
print("then model will naturally predict LONG more often (learned from distribution).")
