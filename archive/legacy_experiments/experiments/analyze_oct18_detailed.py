"""
Analyze Oct 18 in detail - 5 minute by 5 minute
Check if LONG probability really stays high all day
"""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import pickle
import joblib
from datetime import datetime

from scripts.experiments.calculate_all_features import calculate_all_features
from scripts.experiments.retrain_exit_models_opportunity_gating import prepare_exit_features

DATA_DIR = PROJECT_ROOT / "data" / "historical"
MODELS_DIR = PROJECT_ROOT / "models"

print("="*80)
print("DETAILED ANALYSIS: OCT 18, 2025 (5-minute by 5-minute)")
print("="*80)

# Load data
data_file = DATA_DIR / "BTCUSDT_5m_max.csv"
df = pd.read_csv(data_file)

# Calculate features
print("\nCalculating features...")
df = calculate_all_features(df)
df = prepare_exit_features(df)

# Load model
long_model_path = MODELS_DIR / "xgboost_long_trade_outcome_full_20251018_233146.pkl"
long_scaler_path = MODELS_DIR / "xgboost_long_trade_outcome_full_20251018_233146_scaler.pkl"
long_features_path = MODELS_DIR / "xgboost_long_trade_outcome_full_20251018_233146_features.txt"

with open(long_model_path, 'rb') as f:
    long_model = pickle.load(f)
long_scaler = joblib.load(long_scaler_path)
with open(long_features_path, 'r') as f:
    long_features = [line.strip() for line in f.readlines() if line.strip()]

# Calculate probabilities
probabilities = []
for i in range(len(df)):
    try:
        features = df[long_features].iloc[i:i+1].values
        scaled = long_scaler.transform(features)
        prob = long_model.predict_proba(scaled)[0][1]
        probabilities.append(prob)
    except:
        probabilities.append(np.nan)

df['long_prob'] = probabilities

# Filter Oct 18 data
df['date'] = pd.to_datetime(df['timestamp']).dt.date
oct18 = df[df['date'] == pd.to_datetime('2025-10-18').date()].copy()

print(f"\n✅ Oct 18 data: {len(oct18)} candles")
print(f"   Time range: {oct18['timestamp'].iloc[0]} to {oct18['timestamp'].iloc[-1]}")

# Expected: 24 hours × 12 candles/hour = 288 candles
print(f"   Expected candles in 24h: 288")
print(f"   Actual candles: {len(oct18)}")

# Hour by hour analysis
oct18['hour'] = pd.to_datetime(oct18['timestamp']).dt.hour

print("\n" + "="*80)
print("HOUR-BY-HOUR ANALYSIS (Oct 18)")
print("="*80)

hourly = oct18.groupby('hour').agg({
    'long_prob': ['count', 'mean', 'median', 'min', 'max', 'std'],
    'close': 'mean'
})

print("\n📊 LONG Probability by Hour:")
print(f"{'Hour':<6} {'Count':<8} {'Mean':<10} {'Median':<10} {'Min':<10} {'Max':<10} {'Std':<10} {'Price':<12}")
print("-" * 90)

for hour in range(24):
    if hour in hourly.index:
        row = hourly.loc[hour]
        count = row[('long_prob', 'count')]
        mean = row[('long_prob', 'mean')]
        median = row[('long_prob', 'median')]
        min_val = row[('long_prob', 'min')]
        max_val = row[('long_prob', 'max')]
        std = row[('long_prob', 'std')]
        price = row[('close', 'mean')]

        print(f"{hour:02d}:00  {count:<8.0f} {mean*100:<10.1f}% {median*100:<10.1f}% "
              f"{min_val*100:<10.1f}% {max_val*100:<10.1f}% {std:<10.4f} ${price:,.0f}")

# Check variation within each hour
print("\n" + "="*80)
print("VARIATION CHECK: Do probabilities change within hours?")
print("="*80)

# Sample a few hours
sample_hours = [0, 6, 12, 18, 20, 21]

for hour in sample_hours:
    hour_data = oct18[oct18['hour'] == hour]
    if len(hour_data) > 0:
        print(f"\n📌 Hour {hour:02d}:00 - {hour:02d}:59 ({len(hour_data)} candles):")
        print(f"   LONG prob range: {hour_data['long_prob'].min()*100:.1f}% - {hour_data['long_prob'].max()*100:.1f}%")
        print(f"   Price range: ${hour_data['close'].min():,.0f} - ${hour_data['close'].max():,.0f}")

        # Show first 5 candles
        if len(hour_data) >= 5:
            print(f"\n   First 5 candles:")
            for idx, row in hour_data.head(5).iterrows():
                print(f"      {row['timestamp']}: LONG {row['long_prob']*100:.1f}%, Price ${row['close']:,.1f}")

# Distribution analysis
print("\n" + "="*80)
print("PROBABILITY DISTRIBUTION (All of Oct 18)")
print("="*80)

print(f"\n📊 Statistics:")
print(f"  Mean: {oct18['long_prob'].mean()*100:.2f}%")
print(f"  Median: {oct18['long_prob'].median()*100:.2f}%")
print(f"  Std Dev: {oct18['long_prob'].std():.4f}")
print(f"  Min: {oct18['long_prob'].min()*100:.2f}%")
print(f"  Max: {oct18['long_prob'].max()*100:.2f}%")

print(f"\n📈 Distribution:")
ranges = [(0, 30), (30, 50), (50, 70), (70, 80), (80, 90), (90, 100)]
for low, high in ranges:
    count = len(oct18[(oct18['long_prob']*100 >= low) & (oct18['long_prob']*100 < high)])
    pct = count / len(oct18) * 100
    print(f"  {low:2d}-{high:2d}%: {count:4d} candles ({pct:5.1f}%)")

# Compare with production logs
print("\n" + "="*80)
print("COMPARE WITH PRODUCTION LOGS")
print("="*80)

# Production ran from 03:53 to 06:55 KST (18:53 to 21:55 UTC Oct 18)
production_start = pd.to_datetime("2025-10-18 18:53:00")
production_end = pd.to_datetime("2025-10-18 21:55:00")

production_period = oct18[(pd.to_datetime(oct18['timestamp']) >= production_start) &
                          (pd.to_datetime(oct18['timestamp']) <= production_end)]

print(f"\nProduction Period (18:53 - 21:55 UTC):")
print(f"  Candles: {len(production_period)}")
print(f"  LONG prob mean: {production_period['long_prob'].mean()*100:.2f}%")
print(f"  LONG prob median: {production_period['long_prob'].median()*100:.2f}%")
print(f"  Range: {production_period['long_prob'].min()*100:.1f}% - {production_period['long_prob'].max()*100:.1f}%")

# Show 20 sample points from production period
print(f"\n📋 Sample 20 points from production period:")
sample_indices = np.linspace(0, len(production_period)-1, 20, dtype=int)
for idx in sample_indices:
    row = production_period.iloc[idx]
    print(f"  {row['timestamp']}: LONG {row['long_prob']*100:.1f}%, Price ${row['close']:,.1f}")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
