"""
Critical Analysis: Signal Frequency Across Time
================================================
Compare signal frequency across different time periods to find root cause
"""
import pandas as pd
import numpy as np
import pickle
import joblib
from pathlib import Path
import sys
from datetime import datetime, timedelta

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiments.calculate_all_features import calculate_all_features

DATA_DIR = PROJECT_ROOT / "data" / "historical"
MODELS_DIR = PROJECT_ROOT / "models"

# Thresholds
LONG_THRESHOLD = 0.65
SHORT_THRESHOLD = 0.70

print("="*80)
print("🔍 CRITICAL ANALYSIS: Signal Frequency Across Time")
print("="*80)

# Load models
print("\nLoading models...")
long_model_path = MODELS_DIR / "xgboost_long_trade_outcome_full_20251018_233146.pkl"
with open(long_model_path, 'rb') as f:
    long_model = pickle.load(f)

long_scaler_path = MODELS_DIR / "xgboost_long_trade_outcome_full_20251018_233146_scaler.pkl"
long_scaler = joblib.load(long_scaler_path)

long_features_path = MODELS_DIR / "xgboost_long_trade_outcome_full_20251018_233146_features.txt"
with open(long_features_path, 'r') as f:
    long_feature_columns = [line.strip() for line in f.readlines()]

short_model_path = MODELS_DIR / "xgboost_short_trade_outcome_full_20251018_233146.pkl"
with open(short_model_path, 'rb') as f:
    short_model = pickle.load(f)

short_scaler_path = MODELS_DIR / "xgboost_short_trade_outcome_full_20251018_233146_scaler.pkl"
short_scaler = joblib.load(short_scaler_path)

short_features_path = MODELS_DIR / "xgboost_short_trade_outcome_full_20251018_233146_features.txt"
with open(short_features_path, 'r') as f:
    short_feature_columns = [line.strip() for line in f.readlines()]

print("✅ Models loaded")

# Load full data
print("\nLoading full historical data...")
data_path = DATA_DIR / "BTCUSDT_5m_max.csv"
df_full = pd.read_csv(data_path)
df_full['timestamp'] = pd.to_datetime(df_full['timestamp'])
print(f"✅ Loaded {len(df_full)} total candles")
print(f"   Date range: {df_full['timestamp'].min()} to {df_full['timestamp'].max()}")

# Define time periods for analysis
latest_date = df_full['timestamp'].max()
print(f"\n📅 Latest data point: {latest_date}")

periods = {
    "1. Most Recent (2 days)": (latest_date - timedelta(days=2), latest_date),
    "2. Last Week (7 days)": (latest_date - timedelta(days=7), latest_date - timedelta(days=5)),
    "3. Two Weeks Ago": (latest_date - timedelta(days=14), latest_date - timedelta(days=12)),
    "4. One Month Ago": (latest_date - timedelta(days=30), latest_date - timedelta(days=28)),
    "5. Two Months Ago": (latest_date - timedelta(days=60), latest_date - timedelta(days=58)),
}

results = []

for period_name, (start_date, end_date) in periods.items():
    print(f"\n{'='*80}")
    print(f"Analyzing: {period_name}")
    print(f"Date Range: {start_date} to {end_date}")
    print('='*80)

    # Filter data for this period
    df_period = df_full[(df_full['timestamp'] >= start_date) & (df_full['timestamp'] <= end_date)].copy()

    if len(df_period) < 100:
        print(f"⚠️  Insufficient data ({len(df_period)} candles), skipping...")
        continue

    print(f"Candles: {len(df_period)}")

    # Calculate features
    print("Calculating features...")
    df_period = calculate_all_features(df_period)

    # Generate signals
    long_signals = 0
    short_signals = 0
    long_probs = []
    short_probs = []

    for i in range(len(df_period)):
        try:
            # LONG signal
            long_features_values = df_period[long_feature_columns].iloc[i:i+1].values
            if not np.isnan(long_features_values).any():
                long_features_scaled = long_scaler.transform(long_features_values)
                long_prob = long_model.predict_proba(long_features_scaled)[0][1]
                long_probs.append(long_prob)

                if long_prob >= LONG_THRESHOLD:
                    long_signals += 1

            # SHORT signal
            short_features_values = df_period[short_feature_columns].iloc[i:i+1].values
            if not np.isnan(short_features_values).any():
                short_features_scaled = short_scaler.transform(short_features_values)
                short_prob = short_model.predict_proba(short_features_scaled)[0][1]
                short_probs.append(short_prob)

                if short_prob >= SHORT_THRESHOLD:
                    short_signals += 1
        except Exception as e:
            continue

    total_candles = len(df_period)
    long_pct = long_signals / total_candles * 100
    short_pct = short_signals / total_candles * 100

    # Calculate price statistics
    price_start = df_period['close'].iloc[0]
    price_end = df_period['close'].iloc[-1]
    price_change_pct = (price_end - price_start) / price_start * 100
    volatility = df_period['close'].pct_change().std() * 100

    print(f"\n📊 Results:")
    print(f"  LONG Signals: {long_signals}/{total_candles} ({long_pct:.1f}%)")
    print(f"  SHORT Signals: {short_signals}/{total_candles} ({short_pct:.1f}%)")
    print(f"  Avg LONG Prob: {np.mean(long_probs):.4f}")
    print(f"  Avg SHORT Prob: {np.mean(short_probs):.4f}")
    print(f"\n📈 Market Conditions:")
    print(f"  Price Change: {price_change_pct:+.2f}%")
    print(f"  Volatility: {volatility:.2f}%")

    results.append({
        'period': period_name,
        'start': start_date,
        'end': end_date,
        'candles': total_candles,
        'long_signals': long_signals,
        'long_pct': long_pct,
        'short_signals': short_signals,
        'short_pct': short_pct,
        'avg_long_prob': np.mean(long_probs) if long_probs else 0,
        'avg_short_prob': np.mean(short_probs) if short_probs else 0,
        'price_change_pct': price_change_pct,
        'volatility': volatility
    })

# Summary comparison
print("\n" + "="*80)
print("📊 COMPARATIVE SUMMARY")
print("="*80)

df_results = pd.DataFrame(results)
print("\n" + df_results.to_string(index=False))

# Critical analysis
print("\n" + "="*80)
print("🔍 CRITICAL ANALYSIS")
print("="*80)

print("\n1️⃣ LONG Signal Frequency Trend:")
long_pcts = df_results['long_pct'].values
if len(long_pcts) >= 2:
    recent = long_pcts[0]
    historical_avg = np.mean(long_pcts[1:])
    difference = recent - historical_avg

    print(f"   Most Recent: {recent:.1f}%")
    print(f"   Historical Avg: {historical_avg:.1f}%")
    print(f"   Difference: {difference:+.1f}%")

    if abs(difference) > 20:
        print(f"   ⚠️  WARNING: {abs(difference):.1f}% difference - ABNORMAL!")
    elif abs(difference) > 10:
        print(f"   ⚠️  CAUTION: {abs(difference):.1f}% difference - Notable change")
    else:
        print(f"   ✅ NORMAL: Consistent with historical patterns")

print("\n2️⃣ Market Conditions vs Signal Frequency:")
correlation_price_long = np.corrcoef(df_results['price_change_pct'], df_results['long_pct'])[0,1]
print(f"   Correlation (Price Change vs LONG Signals): {correlation_price_long:.3f}")

if abs(correlation_price_long) > 0.7:
    print(f"   ⚠️  Strong correlation - Model may be overfitting to recent trend!")
elif abs(correlation_price_long) > 0.5:
    print(f"   ⚠️  Moderate correlation - Check if model generalizes")
else:
    print(f"   ✅ Low correlation - Model appears robust")

print("\n3️⃣ Probability Distribution Analysis:")
prob_variance = np.var([r['avg_long_prob'] for r in results])
print(f"   Variance in avg LONG probability: {prob_variance:.6f}")

if prob_variance > 0.01:
    print(f"   ⚠️  High variance - Model behavior changing significantly over time")
else:
    print(f"   ✅ Low variance - Consistent model behavior")

print("\n4️⃣ Production vs Backtest Comparison:")
print(f"   Production (recent 4h): 52.7% LONG signals")
print(f"   Backtest recent 2d: {df_results.iloc[0]['long_pct']:.1f}% LONG signals")
print(f"   Backtest 1 month ago: {df_results.iloc[3]['long_pct']:.1f}% LONG signals" if len(df_results) > 3 else "")

print("\n" + "="*80)
print("💡 ROOT CAUSE HYPOTHESIS")
print("="*80)

if recent > 50 and historical_avg < 30:
    print("\n⚠️  HYPOTHESIS 1: Recent Market Regime Shift")
    print("   - Recent data shows unusually high LONG signal frequency")
    print("   - Model trained on historical data may not generalize")
    print("   - Recommendation: Monitor for mean reversion or retrain model")
elif correlation_price_long > 0.7:
    print("\n⚠️  HYPOTHESIS 2: Model Overfitting to Trend")
    print("   - Strong correlation between price movement and signals")
    print("   - Model may be trend-following rather than pattern-detecting")
    print("   - Recommendation: Add trend-neutral features or regularization")
elif prob_variance > 0.01:
    print("\n⚠️  HYPOTHESIS 3: Model Instability")
    print("   - Large variance in average probabilities across periods")
    print("   - Model behavior inconsistent over time")
    print("   - Recommendation: Check feature stability and model calibration")
else:
    print("\n✅ No Clear Abnormality Detected")
    print("   - Signal frequency appears consistent")
    print("   - Model behavior stable across time periods")
    print("   - System operating within expected parameters")

print("\n" + "="*80)
