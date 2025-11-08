"""
Compare backtest predictions vs actual production log probabilities
Verify if backtest reproduces the high probabilities seen in production
"""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import pickle
import joblib
import re

from scripts.experiments.calculate_all_features import calculate_all_features
from scripts.experiments.retrain_exit_models_opportunity_gating import prepare_exit_features

DATA_DIR = PROJECT_ROOT / "data" / "historical"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"

print("="*80)
print("BACKTEST vs PRODUCTION LOG COMPARISON")
print("="*80)

# ============================================================================
# STEP 1: Parse production log for actual probabilities
# ============================================================================
print("\n📋 Parsing production log...")
log_file = LOGS_DIR / "opportunity_gating_bot_4x_20251019.log"

# Pattern: "Final LONG probability: 0.9543 (95.43%)"
# Need to also capture timestamp from previous lines
log_data = []

with open(log_file, 'r', encoding='utf-8') as f:
    current_timestamp = None
    current_price = None

    for line in f:
        # Capture timestamp from data verification line
        # "LAST candle (index -1): 2025-10-18 20:40:00 | Close: $107,124.6"
        ts_match = re.search(r'LAST candle.*?(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*?Close: \$([0-9,]+\.\d+)', line)
        if ts_match:
            current_timestamp = ts_match.group(1)
            current_price = float(ts_match.group(2).replace(',', ''))

        # Capture LONG probability
        prob_match = re.search(r'Final LONG probability: ([\d.]+) \(([\d.]+)%\)', line)
        if prob_match and current_timestamp:
            prob = float(prob_match.group(1))
            log_data.append({
                'timestamp': current_timestamp,
                'price': current_price,
                'long_prob_production': prob
            })
            current_timestamp = None
            current_price = None

df_log = pd.DataFrame(log_data)
print(f"  Found {len(df_log)} production log entries")
if len(df_log) > 0:
    print(f"  Time range: {df_log['timestamp'].iloc[0]} to {df_log['timestamp'].iloc[-1]}")
    print(f"  LONG prob range: {df_log['long_prob_production'].min():.4f} - {df_log['long_prob_production'].max():.4f}")
    print(f"  LONG prob mean: {df_log['long_prob_production'].mean():.4f}")

# ============================================================================
# STEP 2: Load model and calculate backtest probabilities
# ============================================================================
print("\n📊 Loading data and model...")
data_file = DATA_DIR / "BTCUSDT_5m_max.csv"
df = pd.read_csv(data_file)
print(f"  Total candles: {len(df):,}")

# Load model
model_path = MODELS_DIR / "xgboost_long_trade_outcome_full_20251018_233146.pkl"
scaler_path = MODELS_DIR / "xgboost_long_trade_outcome_full_20251018_233146_scaler.pkl"
features_path = MODELS_DIR / "xgboost_long_trade_outcome_full_20251018_233146_features.txt"

with open(model_path, 'rb') as f:
    model = pickle.load(f)
scaler = joblib.load(scaler_path)
with open(features_path, 'r') as f:
    feature_names = [line.strip() for line in f.readlines() if line.strip()]

print(f"  Model features: {len(feature_names)}")

# Calculate features
print("\n🔧 Calculating features...")
df_features = calculate_all_features(df.copy())
df_features = prepare_exit_features(df_features)
print(f"  Features calculated for {len(df_features):,} candles")

# ============================================================================
# STEP 3: Calculate backtest probabilities for logged timestamps
# ============================================================================
print("\n🔬 Calculating backtest probabilities for production timestamps...")

results = []
for idx, row in df_log.iterrows():
    ts = row['timestamp']
    prod_prob = row['long_prob_production']
    prod_price = row['price']

    # Find in features DataFrame
    feature_row = df_features[df_features['timestamp'] == ts]

    if len(feature_row) == 0:
        print(f"  ⚠️  {ts} not found in backtest data")
        continue

    # Calculate backtest probability
    features = feature_row[feature_names].values
    scaled = scaler.transform(features)
    pred = model.predict_proba(scaled)
    backtest_prob = pred[0][1]
    backtest_price = feature_row['close'].iloc[0]

    results.append({
        'timestamp': ts,
        'backtest_prob': backtest_prob,
        'production_prob': prod_prob,
        'diff': abs(backtest_prob - prod_prob),
        'backtest_price': backtest_price,
        'production_price': prod_price,
        'price_diff': abs(backtest_price - prod_price)
    })

df_results = pd.DataFrame(results)

# ============================================================================
# STEP 4: Analyze results
# ============================================================================
print(f"\n📊 COMPARISON RESULTS ({len(df_results)} timestamps):")
print("="*80)

if len(df_results) == 0:
    print("❌ No matching timestamps found!")
    sys.exit(1)

print(f"\n📈 Probability Statistics:")
print(f"  Backtest mean: {df_results['backtest_prob'].mean():.4f} ({df_results['backtest_prob'].mean()*100:.2f}%)")
print(f"  Production mean: {df_results['production_prob'].mean():.4f} ({df_results['production_prob'].mean()*100:.2f}%)")
print(f"  Mean difference: {df_results['diff'].mean():.4f} ({df_results['diff'].mean()*100:.2f}%)")
print(f"  Max difference: {df_results['diff'].max():.4f} ({df_results['diff'].max()*100:.2f}%)")

print(f"\n💰 Price Statistics:")
print(f"  Backtest mean: ${df_results['backtest_price'].mean():,.2f}")
print(f"  Production mean: ${df_results['production_price'].mean():,.2f}")
print(f"  Mean price diff: ${df_results['price_diff'].mean():,.2f}")
print(f"  Max price diff: ${df_results['price_diff'].max():,.2f}")

print(f"\n📋 Sample Comparisons (first 20):")
print(f"{'Timestamp':<20} {'Backtest%':<12} {'Production%':<12} {'Diff%':<12} {'Price Diff':<12}")
print("-"*80)

for idx, row in df_results.head(20).iterrows():
    print(f"{row['timestamp']:<20} {row['backtest_prob']*100:<12.2f} {row['production_prob']*100:<12.2f} "
          f"{row['diff']*100:<12.2f} ${row['price_diff']:<11.2f}")

# Check if probabilities match
threshold = 0.02  # 2% tolerance
matching = df_results[df_results['diff'] < threshold]
print(f"\n✅ Matching probabilities (within {threshold*100:.0f}%): {len(matching)}/{len(df_results)} ({len(matching)/len(df_results)*100:.1f}%)")

if len(matching) / len(df_results) > 0.9:
    print(f"\n✅ CONCLUSION: Backtest REPRODUCES production probabilities!")
    print(f"  → Both show high LONG probabilities (70-98%)")
    print(f"  → The model itself produces these high probabilities")
    print(f"  → NOT a production-specific issue")
else:
    print(f"\n❌ CONCLUSION: Backtest DOES NOT match production!")
    print(f"  → Production and backtest show different probabilities")
    print(f"  → Possible production-specific issue")

# Distribution comparison
print(f"\n📊 Probability Distribution:")
print(f"\nBacktest:")
print(f"  ≥90%: {len(df_results[df_results['backtest_prob'] >= 0.9]):3d} ({len(df_results[df_results['backtest_prob'] >= 0.9])/len(df_results)*100:5.1f}%)")
print(f"  ≥80%: {len(df_results[df_results['backtest_prob'] >= 0.8]):3d} ({len(df_results[df_results['backtest_prob'] >= 0.8])/len(df_results)*100:5.1f}%)")
print(f"  ≥70%: {len(df_results[df_results['backtest_prob'] >= 0.7]):3d} ({len(df_results[df_results['backtest_prob'] >= 0.7])/len(df_results)*100:5.1f}%)")
print(f"  <70%: {len(df_results[df_results['backtest_prob'] < 0.7]):3d} ({len(df_results[df_results['backtest_prob'] < 0.7])/len(df_results)*100:5.1f}%)")

print(f"\nProduction:")
print(f"  ≥90%: {len(df_results[df_results['production_prob'] >= 0.9]):3d} ({len(df_results[df_results['production_prob'] >= 0.9])/len(df_results)*100:5.1f}%)")
print(f"  ≥80%: {len(df_results[df_results['production_prob'] >= 0.8]):3d} ({len(df_results[df_results['production_prob'] >= 0.8])/len(df_results)*100:5.1f}%)")
print(f"  ≥70%: {len(df_results[df_results['production_prob'] >= 0.7]):3d} ({len(df_results[df_results['production_prob'] >= 0.7])/len(df_results)*100:5.1f}%)")
print(f"  <70%: {len(df_results[df_results['production_prob'] < 0.7]):3d} ({len(df_results[df_results['production_prob'] < 0.7])/len(df_results)*100:5.1f}%)")

print("\n" + "="*80)
print("COMPARISON COMPLETE")
print("="*80)
