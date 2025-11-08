"""
Current Market Real-Time Analysis
현재 시장 상태 실시간 분석

목적: 왜 bot이 0 trades인지 현재 시장 분석
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import sys
from datetime import datetime, timedelta

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.production.train_xgboost_improved_v3_phase2 import calculate_features
from scripts.production.advanced_technical_features import AdvancedTechnicalFeatures

DATA_DIR = PROJECT_ROOT / "data" / "historical"
MODELS_DIR = PROJECT_ROOT / "models"

print("=" * 100)
print("CURRENT MARKET REAL-TIME ANALYSIS")
print("=" * 100)
print()

# Load data
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

# Get predictions
X = df[feature_columns].values
X_long_scaled = long_scaler.transform(X)
X_short_scaled = short_scaler.transform(X)

df['prob_long'] = long_model.predict_proba(X_long_scaled)[:, 1]
df['prob_short'] = short_model.predict_proba(X_short_scaled)[:, 1]

# Analyze different time periods
periods = [
    ('Last 24 hours', 12 * 24),
    ('Last 3 days', 12 * 24 * 3),
    ('Last 7 days', 12 * 24 * 7),
    ('Last 14 days', 12 * 24 * 14),
]

print("SIGNAL RATE ANALYSIS ACROSS TIME PERIODS")
print("=" * 100)
print()

# Current thresholds
LONG_THRESH = 0.70
SHORT_THRESH = 0.65

results = []

for period_name, candles in periods:
    df_period = df.iloc[-candles:].copy() if len(df) >= candles else df.copy()

    long_signals = (df_period['prob_long'] >= LONG_THRESH).sum()
    short_signals = (df_period['prob_short'] >= SHORT_THRESH).sum()
    total_signals = long_signals + short_signals

    signal_rate = (total_signals / len(df_period) * 100) if len(df_period) > 0 else 0
    weeks = len(df_period) / (12 * 24 * 7)
    trades_per_week = total_signals / weeks if weeks > 0 else 0

    # Probability distribution
    prob_long_max = df_period['prob_long'].max()
    prob_long_q95 = df_period['prob_long'].quantile(0.95)
    prob_long_median = df_period['prob_long'].median()

    prob_short_max = df_period['prob_short'].max()
    prob_short_q95 = df_period['prob_short'].quantile(0.95)
    prob_short_median = df_period['prob_short'].median()

    results.append({
        'period': period_name,
        'candles': len(df_period),
        'days': len(df_period) / (12 * 24),
        'long_signals': long_signals,
        'short_signals': short_signals,
        'total_signals': total_signals,
        'signal_rate': signal_rate,
        'trades_per_week': trades_per_week,
        'long_max': prob_long_max,
        'long_q95': prob_long_q95,
        'long_median': prob_long_median,
        'short_max': prob_short_max,
        'short_q95': prob_short_q95,
        'short_median': prob_short_median
    })

print(f"{'Period':<20} {'Days':>6} {'Signals':>8} {'Rate%':>8} {'Trades/W':>10}")
print("-" * 100)
for r in results:
    print(f"{r['period']:<20} {r['days']:>6.1f} {r['total_signals']:>8} {r['signal_rate']:>8.2f} {r['trades_per_week']:>10.1f}")

print()
print()

# Current market analysis (last 24h)
df_current = df.iloc[-288:].copy()  # Last 24 hours

print("=" * 100)
print("CURRENT MARKET DETAILED ANALYSIS (Last 24 Hours)")
print("=" * 100)
print()

print("Probability Distribution:")
print(f"  LONG Model:")
print(f"    Max:     {results[0]['long_max']:.4f} {'✅ ABOVE threshold' if results[0]['long_max'] >= LONG_THRESH else '❌ BELOW threshold'}")
print(f"    95th %:  {results[0]['long_q95']:.4f} {'✅ ABOVE threshold' if results[0]['long_q95'] >= LONG_THRESH else '❌ BELOW threshold'}")
print(f"    Median:  {results[0]['long_median']:.4f} {'✅ ABOVE threshold' if results[0]['long_median'] >= LONG_THRESH else '❌ BELOW threshold'}")
print(f"    Current threshold: {LONG_THRESH:.2f}")
print()
print(f"  SHORT Model:")
print(f"    Max:     {results[0]['short_max']:.4f} {'✅ ABOVE threshold' if results[0]['short_max'] >= SHORT_THRESH else '❌ BELOW threshold'}")
print(f"    95th %:  {results[0]['short_q95']:.4f} {'✅ ABOVE threshold' if results[0]['short_q95'] >= SHORT_THRESH else '❌ BELOW threshold'}")
print(f"    Median:  {results[0]['short_median']:.4f} {'✅ ABOVE threshold' if results[0]['short_median'] >= SHORT_THRESH else '❌ BELOW threshold'}")
print(f"    Current threshold: {SHORT_THRESH:.2f}")
print()

# Diagnosis
print("🔍 DIAGNOSIS:")
print()

if results[0]['long_max'] < LONG_THRESH and results[0]['short_max'] < SHORT_THRESH:
    print("  🚨 CRITICAL: BOTH models' max probability < thresholds!")
    print(f"     LONG: {results[0]['long_max']:.4f} < {LONG_THRESH:.2f}")
    print(f"     SHORT: {results[0]['short_max']:.4f} < {SHORT_THRESH:.2f}")
    print()
    print("  This explains 0 trades - IMPOSSIBLE to generate signals!")
    print()
    print("  🎯 SOLUTION: Lower thresholds immediately")

    # Suggest new thresholds
    suggested_long = max(results[0]['long_q95'] - 0.05, 0.50)
    suggested_short = max(results[0]['short_q95'] - 0.05, 0.45)

    print(f"     Suggested LONG: {LONG_THRESH:.2f} → {suggested_long:.2f} ({(suggested_long-LONG_THRESH)/LONG_THRESH*100:+.1f}%)")
    print(f"     Suggested SHORT: {SHORT_THRESH:.2f} → {suggested_short:.2f} ({(suggested_short-SHORT_THRESH)/SHORT_THRESH*100:+.1f}%)")
    print()
    print("  Expected impact:")

    # Calculate expected signals with new thresholds
    new_long_signals = (df_current['prob_long'] >= suggested_long).sum()
    new_short_signals = (df_current['prob_short'] >= suggested_short).sum()
    new_total = new_long_signals + new_short_signals
    new_trades_per_week = new_total / (len(df_current) / (12 * 24 * 7))

    print(f"     Expected signals (24h): {new_total}")
    print(f"     Expected trades/week: {new_trades_per_week:.1f}")
    print(f"     {'✅ Good range (20-60)' if 20 <= new_trades_per_week <= 60 else '⚠️ Outside target range'}")

elif results[0]['long_q95'] < LONG_THRESH or results[0]['short_q95'] < SHORT_THRESH:
    print("  ⚠️ WARNING: 95th percentile below threshold")
    print(f"     Only top 5% of signals can trigger trades")
    print(f"     This is TOO RESTRICTIVE for current market")
    print()
    print("  🎯 SOLUTION: Lower thresholds to capture more opportunities")

else:
    print("  ℹ️ Thresholds seem reasonable based on probability distribution")
    print(f"     But 0 trades indicates other issues (market timing, feature calculation, etc.)")

print()
print("=" * 100)
print("TIME SERIES ANALYSIS")
print("=" * 100)
print()

# Analyze hourly signal distribution (last 24h)
df_current['hour'] = (df_current.index // 12) % 24  # Hour of day

hourly_stats = []
for hour in range(24):
    hour_data = df_current[df_current['hour'] == hour]
    if len(hour_data) > 0:
        long_sig = (hour_data['prob_long'] >= LONG_THRESH).sum()
        short_sig = (hour_data['prob_short'] >= SHORT_THRESH).sum()
        total_sig = long_sig + short_sig

        hourly_stats.append({
            'hour': hour,
            'candles': len(hour_data),
            'signals': total_sig,
            'rate': (total_sig / len(hour_data) * 100) if len(hour_data) > 0 else 0
        })

hourly_df = pd.DataFrame(hourly_stats).sort_values('rate', ascending=False)

print("Top 5 Hours with Highest Signal Rate (Last 24h):")
print(f"{'Hour':<6} {'Candles':>8} {'Signals':>8} {'Rate%':>8}")
print("-" * 40)
for _, row in hourly_df.head(5).iterrows():
    print(f"{int(row['hour']):>2}:00 {row['candles']:>8} {row['signals']:>8} {row['rate']:>8.2f}")

print()
print("Bottom 5 Hours with Lowest Signal Rate (Last 24h):")
print(f"{'Hour':<6} {'Candles':>8} {'Signals':>8} {'Rate%':>8}")
print("-" * 40)
for _, row in hourly_df.tail(5).iterrows():
    print(f"{int(row['hour']):>2}:00 {row['candles']:>8} {row['signals']:>8} {row['rate']:>8.2f}")

print()
print("=" * 100)
print("✅ ANALYSIS COMPLETE")
print("=" * 100)
