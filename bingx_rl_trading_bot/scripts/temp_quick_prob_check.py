"""Quick check of probability distribution on backtest data"""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import pickle
import joblib
from experiments.calculate_all_features import calculate_all_features

DATA_DIR = PROJECT_ROOT / "data" / "historical"
MODELS_DIR = PROJECT_ROOT / "models"

print("Loading data...")
df = pd.read_csv(DATA_DIR / "BTCUSDT_5m_max.csv")
print(f"Loaded {len(df):,} candles")

print("Calculating features...")
df = calculate_all_features(df)
print(f"Features calculated")

# Load production models
timestamp = "20251018_233146"
print(f"\nLoading models: {timestamp}")

# LONG
with open(MODELS_DIR / f"xgboost_long_trade_outcome_full_{timestamp}.pkl", 'rb') as f:
    long_model = pickle.load(f)
long_scaler = joblib.load(MODELS_DIR / f"xgboost_long_trade_outcome_full_{timestamp}_scaler.pkl")
with open(MODELS_DIR / f"xgboost_long_trade_outcome_full_{timestamp}_features.txt", 'r') as f:
    long_features = [line.strip() for line in f.readlines() if line.strip()]

# SHORT  
with open(MODELS_DIR / f"xgboost_short_trade_outcome_full_{timestamp}.pkl", 'rb') as f:
    short_model = pickle.load(f)
short_scaler = joblib.load(MODELS_DIR / f"xgboost_short_trade_outcome_full_{timestamp}_scaler.pkl")
with open(MODELS_DIR / f"xgboost_short_trade_outcome_full_{timestamp}_features.txt", 'r') as f:
    short_features = [line.strip() for line in f.readlines() if line.strip()]

print(f"Models loaded (LONG: {len(long_features)} features, SHORT: {len(short_features)} features)")

# Calculate probabilities for all candles
print("\nCalculating probabilities...")
long_probs = []
short_probs = []

for i in range(len(df)):
    try:
        # LONG
        long_feat = df[long_features].iloc[i:i+1].values
        long_scaled = long_scaler.transform(long_feat)
        long_prob = long_model.predict_proba(long_scaled)[0][1]
        long_probs.append(long_prob)
        
        # SHORT
        short_feat = df[short_features].iloc[i:i+1].values
        short_scaled = short_scaler.transform(short_feat)
        short_prob = short_model.predict_proba(short_scaled)[0][1]
        short_probs.append(short_prob)
        
        if (i + 1) % 5000 == 0:
            print(f"  Processed {i+1:,} candles...")
    except:
        long_probs.append(np.nan)
        short_probs.append(np.nan)

# Statistics
long_probs = np.array(long_probs)
short_probs = np.array(short_probs)

print("\n" + "="*80)
print("PROBABILITY DISTRIBUTION (Full Backtest Dataset)")
print("="*80)
print(f"\nTotal samples: {len(long_probs):,}")
print(f"\nLONG Probabilities:")
print(f"  Mean: {np.nanmean(long_probs):.4f} ({np.nanmean(long_probs)*100:.2f}%)")
print(f"  Median: {np.nanmedian(long_probs):.4f}")
print(f"  Min: {np.nanmin(long_probs):.4f}")
print(f"  Max: {np.nanmax(long_probs):.4f}")
print(f"  >= 0.65: {np.sum(long_probs >= 0.65)} ({np.sum(long_probs >= 0.65)/len(long_probs)*100:.1f}%)")

print(f"\nSHORT Probabilities:")
print(f"  Mean: {np.nanmean(short_probs):.4f} ({np.nanmean(short_probs)*100:.2f}%)")
print(f"  Median: {np.nanmedian(short_probs):.4f}")
print(f"  Min: {np.nanmin(short_probs):.4f}")
print(f"  Max: {np.nanmax(short_probs):.4f}")
print(f"  >= 0.70: {np.sum(short_probs >= 0.70)} ({np.sum(short_probs >= 0.70)/len(short_probs)*100:.1f}%)")

# Recent data (last 1000 candles)
print(f"\n" + "="*80)
print("RECENT DATA (Last 1000 candles)")
print("="*80)
recent_long = long_probs[-1000:]
recent_short = short_probs[-1000:]

print(f"\nLONG Probabilities (recent):")
print(f"  Mean: {np.nanmean(recent_long):.4f} ({np.nanmean(recent_long)*100:.2f}%)")
print(f"  >= 0.65: {np.sum(recent_long >= 0.65)} ({np.sum(recent_long >= 0.65)/len(recent_long)*100:.1f}%)")

print(f"\nSHORT Probabilities (recent):")
print(f"  Mean: {np.nanmean(recent_short):.4f} ({np.nanmean(recent_short)*100:.2f}%)")
print(f"  >= 0.70: {np.sum(recent_short >= 0.70)} ({np.sum(recent_short >= 0.70)/len(recent_short)*100:.1f}%)")

print("\n" + "="*80)
