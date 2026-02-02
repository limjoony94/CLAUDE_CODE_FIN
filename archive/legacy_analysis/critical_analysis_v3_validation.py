"""
V3 Optimization Critical Analysis & Validation
비판적 사고를 통한 논리적/수학적 모순점 검증

검증 항목:
1. Signal Rate Discrepancy Analysis
2. Walk-Forward Validation Integrity
3. Temporal Bias Re-examination
4. Production Performance Gap Root Cause
5. Statistical Significance Testing
6. Oct 10 "Outlier" vs "Extreme Event" Classification
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from datetime import datetime, timedelta
import sys

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data" / "historical"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

print("=" * 100)
print("V3 OPTIMIZATION CRITICAL ANALYSIS")
print("비판적 사고: 논리적/수학적 모순점 검증")
print("=" * 100)
print()

# Load data
print("Loading data...")
df = pd.read_csv(DATA_DIR / "BTCUSDT_5m_max.csv")
print(f"✅ Total data: {len(df)} candles ({len(df)/(12*24):.1f} days)")
print()

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

# Calculate features
from scripts.production.train_xgboost_improved_v3_phase2 import calculate_features
from scripts.production.advanced_technical_features import AdvancedTechnicalFeatures

df = calculate_features(df)
adv_features = AdvancedTechnicalFeatures()
df = adv_features.calculate_all_features(df)
df = df.ffill().dropna()

# Get predictions
X = df[feature_columns].values
X_long_scaled = long_scaler.transform(X)
X_short_scaled = short_scaler.transform(X)
df['prob_long'] = long_model.predict_proba(X_long_scaled)[:, 1]
df['prob_short'] = short_model.predict_proba(X_short_scaled)[:, 1]

print(f"✅ Features calculated: {len(df)} rows")
print()

# ============================================================================
# CRITICAL ANALYSIS 1: Signal Rate Distribution Analysis
# ============================================================================

print("=" * 100)
print("CRITICAL ANALYSIS 1: Signal Rate Distribution")
print("=" * 100)
print()

LONG_THRESH = 0.70
SHORT_THRESH = 0.65

# Analyze full dataset by time periods
def analyze_period(df_period, name):
    long_signals = (df_period['prob_long'] >= LONG_THRESH).sum()
    short_signals = (df_period['prob_short'] >= SHORT_THRESH).sum()
    total = long_signals + short_signals
    rate = (total / len(df_period) * 100) if len(df_period) > 0 else 0
    return {
        'period': name,
        'candles': len(df_period),
        'days': len(df_period) / (12*24),
        'long_signals': long_signals,
        'short_signals': short_signals,
        'total_signals': total,
        'signal_rate': rate,
        'expected_trades_per_week': (total / len(df_period)) * (12*24*7) if len(df_period) > 0 else 0
    }

# Analyze different periods
three_months = 12 * 24 * 90
df_3m = df.iloc[-three_months:].copy() if len(df) >= three_months else df.copy()

# Split for V3 analysis
train_end = int(len(df_3m) * 0.70)
val_end = int(len(df_3m) * 0.85)

df_train = df_3m.iloc[:train_end]
df_val = df_3m.iloc[train_end:val_end]
df_test = df_3m.iloc[val_end:]

# Also analyze V2 period (last 2 weeks)
two_weeks = 12 * 24 * 14
df_v2 = df.iloc[-two_weeks:].copy() if len(df) >= two_weeks else df.copy()

# Recent production period (last 24 hours)
one_day = 12 * 24
df_recent = df.iloc[-one_day:].copy() if len(df) >= one_day else df.copy()

results = []
results.append(analyze_period(df_train, "V3 Train (70%)"))
results.append(analyze_period(df_val, "V3 Val (15%)"))
results.append(analyze_period(df_test, "V3 Test (15%)"))
results.append(analyze_period(df_v2, "V2 Period (2 weeks)"))
results.append(analyze_period(df_recent, "Recent 24h (Production)"))

print(f"{'Period':<25} {'Days':>6} {'LONG':>6} {'SHORT':>6} {'Total':>6} {'Rate%':>7} {'Trades/Week':>12}")
print("-" * 100)
for r in results:
    print(f"{r['period']:<25} {r['days']:>6.1f} {r['long_signals']:>6} {r['short_signals']:>6} "
          f"{r['total_signals']:>6} {r['signal_rate']:>7.2f} {r['expected_trades_per_week']:>12.1f}")

print()
print("🔍 CRITICAL FINDING #1: Signal Rate Variance")
train_rate = results[0]['signal_rate']
val_rate = results[1]['signal_rate']
test_rate = results[2]['signal_rate']
recent_rate = results[4]['signal_rate']

print(f"  Train vs Val variance: {abs(train_rate - val_rate):.2f}% {'✅ OK' if abs(train_rate - val_rate) < 5 else '⚠️ HIGH'}")
print(f"  Train vs Test variance: {abs(train_rate - test_rate):.2f}% {'✅ OK' if abs(train_rate - test_rate) < 5 else '⚠️ HIGH'}")
print(f"  Test vs Recent variance: {abs(test_rate - recent_rate):.2f}%")
print()

if abs(train_rate - test_rate) > 5:
    print("⚠️ CONTRADICTION DETECTED:")
    print(f"   Walk-forward validation assumes Train/Val/Test come from same distribution")
    print(f"   But Test signal rate ({test_rate:.2f}%) is {test_rate/train_rate:.2f}x Train ({train_rate:.2f}%)")
    print(f"   This suggests NON-STATIONARY data")
    print(f"   → V3 optimization may have optimized on LOW signal rate period")
    print(f"   → Test happened during HIGH signal rate period")
    print(f"   → Parameters may not generalize!")
    print()

# ============================================================================
# CRITICAL ANALYSIS 2: Oct 10 "Outlier" Re-examination
# ============================================================================

print("=" * 100)
print("CRITICAL ANALYSIS 2: Oct 10 Analysis - Outlier or Extreme Event?")
print("=" * 100)
print()

# Find Oct 10 in data
df['date'] = pd.to_datetime(df['timestamp']) if 'timestamp' in df.columns else pd.to_datetime(df.index)
df['date_only'] = df['date'].dt.date

# Analyze daily signal rates
daily_stats = []
for date in df['date_only'].unique()[-30:]:  # Last 30 days
    day_data = df[df['date_only'] == date]
    long_sig = (day_data['prob_long'] >= LONG_THRESH).sum()
    short_sig = (day_data['prob_short'] >= SHORT_THRESH).sum()
    total_sig = long_sig + short_sig
    rate = (total_sig / len(day_data) * 100) if len(day_data) > 0 else 0

    # Calculate daily volatility
    daily_return = (day_data['close'].iloc[-1] / day_data['close'].iloc[0] - 1) * 100 if len(day_data) > 0 else 0
    intraday_vol = day_data['close'].pct_change().std() * 100 if len(day_data) > 1 else 0

    daily_stats.append({
        'date': date,
        'candles': len(day_data),
        'signal_rate': rate,
        'signals': total_sig,
        'daily_return': daily_return,
        'intraday_volatility': intraday_vol
    })

daily_df = pd.DataFrame(daily_stats).sort_values('signal_rate', ascending=False)

print("Top 10 Days by Signal Rate:")
print(f"{'Date':<12} {'Candles':>8} {'Signals':>8} {'Rate%':>7} {'Return%':>8} {'Vol%':>6}")
print("-" * 100)
for _, row in daily_df.head(10).iterrows():
    marker = " ← Oct 10?" if str(row['date']) == '2024-10-10' else ""
    print(f"{str(row['date']):<12} {row['candles']:>8} {row['signals']:>8} {row['signal_rate']:>7.2f} "
          f"{row['daily_return']:>8.2f} {row['intraday_volatility']:>6.2f}{marker}")

print()

# Statistical analysis
mean_rate = daily_df['signal_rate'].mean()
std_rate = daily_df['signal_rate'].std()
oct10_data = daily_df[daily_df['date'].astype(str) == '2024-10-10']

if len(oct10_data) > 0:
    oct10_rate = oct10_data.iloc[0]['signal_rate']
    z_score = (oct10_rate - mean_rate) / std_rate if std_rate > 0 else 0

    print(f"Statistical Analysis:")
    print(f"  Mean daily signal rate: {mean_rate:.2f}%")
    print(f"  Std deviation: {std_rate:.2f}%")
    print(f"  Oct 10 signal rate: {oct10_rate:.2f}%")
    print(f"  Oct 10 Z-score: {z_score:.2f} ({'⚠️ OUTLIER' if abs(z_score) > 2 else '✅ Normal variance'})")
    print()

    if abs(z_score) > 2:
        print("🔍 CRITICAL FINDING #2: Oct 10 is Statistical Outlier")
        print(f"   BUT: Is it an 'anomaly' or 'extreme market event'?")
        print(f"   - If anomaly: Should be excluded (won't repeat)")
        print(f"   - If extreme event: Should be INCLUDED (will repeat in volatile markets)")
        print(f"   → Current V3 approach 'dilutes' Oct 10 from 7% to 1.1% influence")
        print(f"   → This assumes Oct 10 won't happen again")
        print(f"   → But extreme events DO repeat in crypto markets!")
        print()
else:
    print("⚠️ Oct 10 not found in dataset (data may start after Oct 10)")
    print()

# ============================================================================
# CRITICAL ANALYSIS 3: Production Performance Gap
# ============================================================================

print("=" * 100)
print("CRITICAL ANALYSIS 3: Production Performance Gap Analysis")
print("=" * 100)
print()

# Calculate expected vs actual
recent_result = results[4]
expected_trades_24h = recent_result['expected_trades_per_week'] / 7
actual_trades_24h = 0  # From bot state (0 trades in 12 hours)

print(f"Production Performance (Last 24 hours):")
print(f"  Expected trades (from backtest): {expected_trades_24h:.2f} trades/day")
print(f"  Expected trades (12 hours): {expected_trades_24h/2:.2f} trades")
print(f"  Actual trades (12 hours): {actual_trades_24h} trades")
print(f"  Gap: {((actual_trades_24h - expected_trades_24h/2) / (expected_trades_24h/2) * 100):.1f}%")
print()

print("🔍 CRITICAL FINDING #3: Expectation-Reality Mismatch")
print(f"  Recent 24h signal rate: {recent_rate:.2f}%")
print(f"  V3 Train signal rate: {train_rate:.2f}%")
print(f"  V3 Test signal rate: {test_rate:.2f}%")
print()

if recent_rate < train_rate * 0.5:
    print("⚠️ CONTRADICTION:")
    print(f"   Current market signal rate ({recent_rate:.2f}%) is {recent_rate/train_rate:.2f}x LOWER than training")
    print(f"   V3 optimized on {train_rate:.2f}% signal rate environment")
    print(f"   Current production is {recent_rate:.2f}% signal rate environment")
    print(f"   → Parameters optimized for HIGH signal rate may not work in LOW signal rate")
    print()

# ============================================================================
# CRITICAL ANALYSIS 4: Parameter Robustness Test
# ============================================================================

print("=" * 100)
print("CRITICAL ANALYSIS 4: Parameters Unchanged Paradox")
print("=" * 100)
print()

# Load V2 and V3 results
v2_results = pd.read_csv(RESULTS_DIR / "position_sizing_comprehensive_final_results.csv")
v3_results = pd.read_csv(RESULTS_DIR / "position_sizing_v3_full_dataset_phase2_results.csv")

v2_best = v2_results.iloc[0]
v3_best = v3_results.iloc[0]

print("Optimal Parameters Comparison:")
print(f"{'Parameter':<20} {'V2 (2 weeks)':<15} {'V3 (3 months)':<15} {'Changed?':<10}")
print("-" * 100)
params = ['signal_weight', 'volatility_weight', 'regime_weight', 'streak_weight',
          'base_position', 'max_position', 'min_position']

all_same = True
for param in params:
    v2_val = v2_best[param]
    v3_val = v3_best[param]
    changed = "No" if abs(v2_val - v3_val) < 0.01 else "Yes"
    if changed == "Yes":
        all_same = False
    print(f"{param:<20} {v2_val:<15.2f} {v3_val:<15.2f} {changed:<10}")

print()
if all_same:
    print("🔍 CRITICAL FINDING #4: Parameters UNCHANGED")
    print("  V2 (2 weeks) vs V3 (3 months) → ALL parameters identical")
    print()
    print("  Possible Explanations:")
    print("  1. ✅ Parameters are genuinely robust across time periods")
    print("  2. ⚠️ Search space too narrow (only 162 combinations tested)")
    print("  3. ⚠️ Optimization stuck in local optimum")
    print("  4. ⚠️ Objective function (average return) not sensitive enough")
    print("  5. ⚠️ V2 and V3 datasets have similar underlying distribution")
    print()
    print("  Recommendation: Expand search space and test more combinations")
    print()

# ============================================================================
# CRITICAL ANALYSIS 5: Walk-Forward Validity Check
# ============================================================================

print("=" * 100)
print("CRITICAL ANALYSIS 5: Walk-Forward Validation Integrity")
print("=" * 100)
print()

# Check if Train/Val/Test are truly independent
train_start = df_train.index[0]
train_end_idx = df_train.index[-1]
val_start = df_val.index[0]
val_end_idx = df_val.index[-1]
test_start = df_test.index[0]

print(f"Dataset Split Verification:")
print(f"  Train: indices {train_start} to {train_end_idx}")
print(f"  Val:   indices {val_start} to {val_end_idx}")
print(f"  Test:  indices {test_start} to {df_test.index[-1]}")
print()

if val_start == train_end_idx + 1 and test_start == val_end_idx + 1:
    print("✅ Sequential split confirmed (no overlap)")
else:
    print("⚠️ Warning: Split may have gaps or overlaps")

print()

# Check temporal consistency assumption
print("Signal Rate Stationarity Test:")
print(f"  Train (70%): {train_rate:.2f}%")
print(f"  Val (15%):   {val_rate:.2f}%")
print(f"  Test (15%):  {test_rate:.2f}%")
print()

variance_ratio = max(train_rate, val_rate, test_rate) / min(train_rate, val_rate, test_rate)
print(f"  Max/Min ratio: {variance_ratio:.2f}x")

if variance_ratio > 2.0:
    print()
    print("⚠️ NON-STATIONARY DATA DETECTED:")
    print(f"   Signal rate varies by {variance_ratio:.2f}x across splits")
    print(f"   Walk-forward validation assumes stationary data")
    print(f"   → Optimization may have learned on WRONG regime")
    print(f"   → Test performance may not reflect true capability")
    print()
    print("  ROOT CAUSE: Crypto markets are non-stationary")
    print("  SOLUTION NEEDED:")
    print("   - Use rolling window optimization (re-optimize monthly)")
    print("   - Add regime detection and regime-specific parameters")
    print("   - Weight recent data more heavily")
    print()

# ============================================================================
# CRITICAL ANALYSIS 6: Mathematical Consistency Check
# ============================================================================

print("=" * 100)
print("CRITICAL ANALYSIS 6: Mathematical Consistency Verification")
print("=" * 100)
print()

# Check if reported metrics are mathematically consistent
print("Verifying V3 Test Results Consistency:")

# Load V3 best config
best_train_return = v3_best['train_return']
best_val_return = v3_best['val_return']
best_avg_return = v3_best['avg_return']

calculated_avg = (best_train_return + best_val_return) / 2
reported_avg = best_avg_return

print(f"  Train Return: {best_train_return:.2f}%")
print(f"  Val Return: {best_val_return:.2f}%")
print(f"  Reported Avg: {reported_avg:.2f}%")
print(f"  Calculated Avg: {calculated_avg:.2f}%")
print(f"  Difference: {abs(reported_avg - calculated_avg):.6f}%")

if abs(reported_avg - calculated_avg) < 0.01:
    print("  ✅ Average calculation is correct")
else:
    print(f"  ⚠️ Discrepancy detected: {abs(reported_avg - calculated_avg):.6f}%")

print()

# Check trades per week calculation
train_trades_per_week = v3_best['train_trades_per_week']
weeks_train = len(df_train) / (12 * 24 * 7)

print(f"  Reported Trades/Week (Train): {train_trades_per_week:.1f}")
print(f"  Training period: {weeks_train:.2f} weeks")
print(f"  Expected total trades: {train_trades_per_week * weeks_train:.0f} trades")
print()

# ============================================================================
# FINAL RECOMMENDATIONS
# ============================================================================

print("=" * 100)
print("FINAL RECOMMENDATIONS - 근본 원인 해결")
print("=" * 100)
print()

print("1. NON-STATIONARY DATA PROBLEM (ROOT CAUSE):")
print("   Issue: Signal rate varies 2-3x across different periods")
print("   Current approach: Static optimization on historical data")
print("   Solution: Implement ADAPTIVE OPTIMIZATION")
print("     - Re-optimize parameters monthly with latest 3-month data")
print("     - Use exponential weighting (recent data = more weight)")
print("     - Add regime detection (Bull/Bear/Sideways-specific params)")
print()

print("2. PARAMETER SEARCH SPACE TOO NARROW:")
print("   Issue: Only 162 combinations tested (V2 and V3 identical)")
print("   Solution: Expand search space")
print("     - Test more weight combinations (0.2-0.5 range, finer grid)")
print("     - Test position sizing 0.4-0.8 base range")
print("     - Use Bayesian optimization for efficiency")
print()

print("3. THRESHOLD OPTIMIZATION MISSING:")
print("   Issue: Entry thresholds (0.70 LONG, 0.65 SHORT) NOT optimized in V3")
print("   These came from V2 entry-only optimization")
print("   Solution: Re-optimize thresholds WITH position sizing")
print("     - Test threshold range: 0.60-0.80 for LONG")
print("     - Test threshold range: 0.55-0.75 for SHORT")
print("     - Optimize jointly with position sizing params")
print()

print("4. PRODUCTION-BACKTEST GAP:")
print("   Issue: Backtest uses completed candles, Production uses real-time")
print("   Solution: Add realism to backtest")
print("     - Simulate order slippage")
print("     - Add latency delays")
print("     - Test with different signal rate environments")
print()

print("5. OCT 10 TREATMENT:")
print("   Issue: Diluting extreme events may reduce robustness")
print("   Solution: Separate 'outlier removal' from 'extreme event handling'")
print("     - Keep extreme volatility events in training")
print("     - Remove true anomalies (exchange outages, flash crashes)")
print("     - Add volatility regime-specific parameters")
print()

print("=" * 100)
print("ANALYSIS COMPLETE")
print("=" * 100)
