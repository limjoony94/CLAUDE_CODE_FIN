"""
Analyze LONG/SHORT probability distribution in backtest
Compare with production probabilities
"""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import pickle
import joblib

from scripts.experiments.calculate_all_features import calculate_all_features
from scripts.experiments.retrain_exit_models_opportunity_gating import prepare_exit_features

DATA_DIR = PROJECT_ROOT / "data" / "historical"
MODELS_DIR = PROJECT_ROOT / "models"

print("="*80)
print("BACKTEST PROBABILITY DISTRIBUTION ANALYSIS")
print("="*80)

# Load data
data_file = DATA_DIR / "BTCUSDT_5m_max.csv"
df = pd.read_csv(data_file)
print(f"\n✅ Loaded {len(df):,} candles")
print(f"   Date range: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")

# Calculate features
print("\nCalculating features...")
df = calculate_all_features(df)
df = prepare_exit_features(df)
print(f"✅ Features calculated")

# Load models
print("\nLoading models...")
long_model_path = MODELS_DIR / "xgboost_long_trade_outcome_full_20251018_233146.pkl"
long_scaler_path = MODELS_DIR / "xgboost_long_trade_outcome_full_20251018_233146_scaler.pkl"
long_features_path = MODELS_DIR / "xgboost_long_trade_outcome_full_20251018_233146_features.txt"

with open(long_model_path, 'rb') as f:
    long_model = pickle.load(f)
long_scaler = joblib.load(long_scaler_path)
with open(long_features_path, 'r') as f:
    long_features = [line.strip() for line in f.readlines() if line.strip()]

print(f"✅ LONG model loaded: {len(long_features)} features")

# Calculate probabilities for ALL data points
print("\nCalculating LONG probabilities for all data points...")
probabilities = []

for i in range(len(df)):
    try:
        features = df[long_features].iloc[i:i+1].values
        scaled = long_scaler.transform(features)
        prob = long_model.predict_proba(scaled)[0][1]
        probabilities.append({
            'timestamp': df['timestamp'].iloc[i],
            'close': df['close'].iloc[i],
            'long_prob': prob
        })
    except:
        pass

prob_df = pd.DataFrame(probabilities)

print(f"\n✅ Calculated probabilities for {len(prob_df):,} points")

# Analyze distribution
print("\n" + "="*80)
print("PROBABILITY DISTRIBUTION ANALYSIS")
print("="*80)

print(f"\n📊 LONG Probability Statistics:")
print(f"  Mean: {prob_df['long_prob'].mean():.4f} ({prob_df['long_prob'].mean()*100:.2f}%)")
print(f"  Median: {prob_df['long_prob'].median():.4f} ({prob_df['long_prob'].median()*100:.2f}%)")
print(f"  Std Dev: {prob_df['long_prob'].std():.4f}")
print(f"  Min: {prob_df['long_prob'].min():.4f} ({prob_df['long_prob'].min()*100:.2f}%)")
print(f"  Max: {prob_df['long_prob'].max():.4f} ({prob_df['long_prob'].max()*100:.2f}%)")

print(f"\n📈 Distribution by Ranges:")
print(f"  < 50%: {len(prob_df[prob_df['long_prob'] < 0.50]):,} ({len(prob_df[prob_df['long_prob'] < 0.50])/len(prob_df)*100:.1f}%)")
print(f"  50-60%: {len(prob_df[(prob_df['long_prob'] >= 0.50) & (prob_df['long_prob'] < 0.60)]):,} ({len(prob_df[(prob_df['long_prob'] >= 0.50) & (prob_df['long_prob'] < 0.60)])/len(prob_df)*100:.1f}%)")
print(f"  60-70%: {len(prob_df[(prob_df['long_prob'] >= 0.60) & (prob_df['long_prob'] < 0.70)]):,} ({len(prob_df[(prob_df['long_prob'] >= 0.60) & (prob_df['long_prob'] < 0.70)])/len(prob_df)*100:.1f}%)")
print(f"  70-80%: {len(prob_df[(prob_df['long_prob'] >= 0.70) & (prob_df['long_prob'] < 0.80)]):,} ({len(prob_df[(prob_df['long_prob'] >= 0.70) & (prob_df['long_prob'] < 0.80)])/len(prob_df)*100:.1f}%)")
print(f"  80-90%: {len(prob_df[(prob_df['long_prob'] >= 0.80) & (prob_df['long_prob'] < 0.90)]):,} ({len(prob_df[(prob_df['long_prob'] >= 0.80) & (prob_df['long_prob'] < 0.90)])/len(prob_df)*100:.1f}%)")
print(f"  >= 90%: {len(prob_df[prob_df['long_prob'] >= 0.90]):,} ({len(prob_df[prob_df['long_prob'] >= 0.90])/len(prob_df)*100:.1f}%)")

# Above threshold (0.65)
above_threshold = prob_df[prob_df['long_prob'] >= 0.65]
print(f"\n🎯 Above Threshold (>= 0.65):")
print(f"  Count: {len(above_threshold):,} ({len(above_threshold)/len(prob_df)*100:.1f}%)")
print(f"  Mean prob: {above_threshold['long_prob'].mean():.4f} ({above_threshold['long_prob'].mean()*100:.2f}%)")

# Recent data (last 1000 points)
recent = prob_df.tail(1000)
print(f"\n📅 Recent Data (Last 1000 points):")
print(f"  Date range: {recent['timestamp'].iloc[0]} to {recent['timestamp'].iloc[-1]}")
print(f"  Mean prob: {recent['long_prob'].mean():.4f} ({recent['long_prob'].mean()*100:.2f}%)")
print(f"  >= 70%: {len(recent[recent['long_prob'] >= 0.70]):,} ({len(recent[recent['long_prob'] >= 0.70])/len(recent)*100:.1f}%)")
print(f"  >= 80%: {len(recent[recent['long_prob'] >= 0.80]):,} ({len(recent[recent['long_prob'] >= 0.80])/len(recent)*100:.1f}%)")
print(f"  >= 90%: {len(recent[recent['long_prob'] >= 0.90]):,} ({len(recent[recent['long_prob'] >= 0.90])/len(recent)*100:.1f}%)")

# Compare with production (Oct 19, 03:53 - 06:55)
print(f"\n🔴 Production (Oct 19, 03:53-06:55):")
print(f"  LONG probabilities observed: 70-98%")
print(f"  Most readings: 85-98%")
print(f"  Average: ~87%")

# Find similar price range in backtest
price_range = prob_df[(prob_df['close'] >= 103000) & (prob_df['close'] <= 108000)]
print(f"\n💰 Similar Price Range ($103k-$108k) in Backtest:")
print(f"  Count: {len(price_range):,}")
print(f"  Mean prob: {price_range['long_prob'].mean():.4f} ({price_range['long_prob'].mean()*100:.2f}%)")
print(f"  >= 80%: {len(price_range[price_range['long_prob'] >= 0.80]):,} ({len(price_range[price_range['long_prob'] >= 0.80])/len(price_range)*100:.1f}%)")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
