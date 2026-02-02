"""
Quick Threshold Sensitivity Analysis
긴급 해결: 현재 Bot의 0 trades 문제 해결

목적: 빠르게 (5-10분) threshold 최적값 찾기
방법: 최근 데이터로 threshold sweep (grid search)
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

DATA_DIR = PROJECT_ROOT / "data" / "historical"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

print("=" * 100)
print("QUICK THRESHOLD SENSITIVITY ANALYSIS")
print("목적: Bot의 0 trades 문제 즉각 해결")
print("=" * 100)
print()

# Load data
print("Loading data...")
df = pd.read_csv(DATA_DIR / "BTCUSDT_5m_max.csv")
df = calculate_features(df)
adv_features = AdvancedTechnicalFeatures()
df = adv_features.calculate_all_features(df)
df = df.ffill().dropna()

# Load models
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

print(f"✅ Data loaded: {len(df)} candles")
print()

# Get predictions
X = df[feature_columns].values
X_long_scaled = long_scaler.transform(X)
X_short_scaled = short_scaler.transform(X)

df['prob_long'] = long_model.predict_proba(X_long_scaled)[:, 1]
df['prob_short'] = short_model.predict_proba(X_short_scaled)[:, 1]

# Use recent data (last 2 weeks for speed)
two_weeks = 12 * 24 * 14
df_recent = df.iloc[-two_weeks:].copy().reset_index(drop=True)

print(f"✅ Analysis dataset: {len(df_recent)} candles ({len(df_recent)/(12*24):.1f} days)")
print()

# Threshold sweep
long_thresholds = np.arange(0.55, 0.81, 0.05)  # 0.55, 0.60, 0.65, 0.70, 0.75, 0.80
short_thresholds = np.arange(0.50, 0.76, 0.05)  # 0.50, 0.55, 0.60, 0.65, 0.70, 0.75

print("=" * 100)
print("THRESHOLD SENSITIVITY ANALYSIS")
print("=" * 100)
print()

results = []

print(f"Testing {len(long_thresholds)} LONG × {len(short_thresholds)} SHORT = {len(long_thresholds) * len(short_thresholds)} combinations")
print()

total_combos = len(long_thresholds) * len(short_thresholds)
combo_count = 0

for long_thresh in long_thresholds:
    for short_thresh in short_thresholds:
        combo_count += 1

        # Calculate signals
        long_signals = (df_recent['prob_long'] >= long_thresh).sum()
        short_signals = (df_recent['prob_short'] >= short_thresh).sum()
        total_signals = long_signals + short_signals

        signal_rate = (total_signals / len(df_recent) * 100) if len(df_recent) > 0 else 0
        weeks = len(df_recent) / (12 * 24 * 7)
        trades_per_week = total_signals / weeks if weeks > 0 else 0

        # Estimate quality (higher probability signals = better quality)
        if total_signals > 0:
            long_mask = df_recent['prob_long'] >= long_thresh
            short_mask = df_recent['prob_short'] >= short_thresh

            long_probs = df_recent.loc[long_mask, 'prob_long'].values if long_mask.sum() > 0 else []
            short_probs = df_recent.loc[short_mask, 'prob_short'].values if short_mask.sum() > 0 else []

            all_probs = np.concatenate([long_probs, short_probs]) if len(long_probs) > 0 or len(short_probs) > 0 else []
            avg_prob = np.mean(all_probs) if len(all_probs) > 0 else 0
        else:
            avg_prob = 0

        # Simple quality score: balance frequency and probability
        # Target: 20-60 trades/week with high probability
        if trades_per_week < 10:
            quality_score = trades_per_week * avg_prob * 0.1  # Heavy penalty
        elif trades_per_week > 60:
            quality_score = (120 - trades_per_week) * avg_prob  # Penalty for too many
        else:
            quality_score = trades_per_week * avg_prob  # Good range

        results.append({
            'long_threshold': long_thresh,
            'short_threshold': short_thresh,
            'long_signals': long_signals,
            'short_signals': short_signals,
            'total_signals': total_signals,
            'signal_rate': signal_rate,
            'trades_per_week': trades_per_week,
            'avg_probability': avg_prob,
            'quality_score': quality_score
        })

        print(f"Progress: {combo_count}/{total_combos} | "
              f"LONG={long_thresh:.2f} SHORT={short_thresh:.2f} | "
              f"Trades/W={trades_per_week:.1f} | "
              f"Score={quality_score:.2f}", end='\r')

print()
print()

# Sort by quality score
results_df = pd.DataFrame(results).sort_values('quality_score', ascending=False)

print("=" * 100)
print("TOP 10 THRESHOLD COMBINATIONS (Sorted by Quality Score)")
print("=" * 100)
print()
print(f"{'Rank':<5} {'LONG':<6} {'SHORT':<6} {'Trades/W':<10} {'Signal%':<8} {'AvgProb':<8} {'Score':<8}")
print("-" * 100)

for i, (idx, row) in enumerate(results_df.head(10).iterrows(), 1):
    marker = " ← BEST" if i == 1 else ""
    print(f"{i:<5} {row['long_threshold']:<6.2f} {row['short_threshold']:<6.2f} "
          f"{row['trades_per_week']:<10.1f} {row['signal_rate']:<8.2f} "
          f"{row['avg_probability']:<8.3f} {row['quality_score']:<8.2f}{marker}")

print()
print("=" * 100)
print("COMPARISON WITH CURRENT CONFIGURATION")
print("=" * 100)
print()

# Current config
current_long = 0.70
current_short = 0.65

current = results_df[
    (results_df['long_threshold'] == current_long) &
    (results_df['short_threshold'] == current_short)
]

best = results_df.iloc[0]

if len(current) > 0:
    current = current.iloc[0]
    current_rank = results_df.index.get_loc(current.name) + 1

    print(f"{'Metric':<25} {'Current (V3)':<20} {'Recommended (Best)':<20} {'Change':<15}")
    print("-" * 100)
    print(f"{'LONG Threshold':<25} {current_long:<20.2f} {best['long_threshold']:<20.2f} {f'{(best['long_threshold']-current_long)/current_long*100:+.1f}%':<15}")
    print(f"{'SHORT Threshold':<25} {current_short:<20.2f} {best['short_threshold']:<20.2f} {f'{(best['short_threshold']-current_short)/current_short*100:+.1f}%':<15}")
    print(f"{'Trades/Week':<25} {current['trades_per_week']:<20.1f} {best['trades_per_week']:<20.1f} {f'{(best['trades_per_week']-current['trades_per_week'])/current['trades_per_week']*100:+.1f}%':<15}")
    print(f"{'Signal Rate':<25} {f'{current['signal_rate']:.2f}%':<20} {f'{best['signal_rate']:.2f}%':<20} {f'{best['signal_rate']-current['signal_rate']:+.2f}pp':<15}")
    print(f"{'Avg Probability':<25} {current['avg_probability']:<20.3f} {best['avg_probability']:<20.3f} {f'{(best['avg_probability']-current['avg_probability'])/current['avg_probability']*100:+.1f}%':<15}")
    print(f"{'Quality Score':<25} {current['quality_score']:<20.2f} {best['quality_score']:<20.2f} {f'{(best['quality_score']-current['quality_score'])/current['quality_score']*100:+.1f}%':<15}")
    print(f"{'Rank':<25} {f'#{current_rank}':<20} {'#1':<20} {'':<15}")

    print()
    print("🎯 RECOMMENDATION:")
    print(f"  Change LONG threshold: {current_long:.2f} → {best['long_threshold']:.2f} ({(best['long_threshold']-current_long)/current_long*100:+.1f}%)")
    print(f"  Change SHORT threshold: {current_short:.2f} → {best['short_threshold']:.2f} ({(best['short_threshold']-current_short)/current_short*100:+.1f}%)")
    print()
    print(f"  Expected impact:")
    print(f"    - Trades/week: {current['trades_per_week']:.1f} → {best['trades_per_week']:.1f} ({(best['trades_per_week']-current['trades_per_week'])/current['trades_per_week']*100:+.1f}%)")
    print(f"    - Should fix 0 trades problem: {'✅ YES' if best['trades_per_week'] > 20 else '⚠️ PARTIAL'}")

# Save results
output_file = RESULTS_DIR / "quick_threshold_sensitivity_analysis.csv"
results_df.to_csv(output_file, index=False)
print()
print(f"✅ Results saved: {output_file}")

print()
print("=" * 100)
print("✅ QUICK ANALYSIS COMPLETE")
print("=" * 100)
print()
print("Next steps:")
print("  1. Review top 10 configurations above")
print("  2. Update production bot with recommended thresholds")
print("  3. Restart bot and monitor for 24 hours")
print("  4. If performance still poor, run full V4 Bayesian optimization")
