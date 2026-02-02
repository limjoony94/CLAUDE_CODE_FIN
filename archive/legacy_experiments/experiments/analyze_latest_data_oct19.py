"""
Analyze latest data from BingX API (Oct 19)
Fetch fresh data and calculate probabilities
"""
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import pickle
import joblib
import yaml
from datetime import datetime, timedelta

# Import BingX API wrapper
from src.api.bingx_client import BingXClient

from scripts.experiments.calculate_all_features import calculate_all_features
from scripts.experiments.retrain_exit_models_opportunity_gating import prepare_exit_features

DATA_DIR = PROJECT_ROOT / "data" / "historical"
MODELS_DIR = PROJECT_ROOT / "models"
CONFIG_DIR = PROJECT_ROOT / "config"

print("="*80)
print("LATEST DATA ANALYSIS - OCT 19 (LIVE API DATA)")
print("="*80)

# ============================================================================
# STEP 1: Load API config
# ============================================================================
print("\n📁 Loading API config...")
with open(CONFIG_DIR / "api_keys.yaml", 'r') as f:
    config = yaml.safe_load(f)

testnet_config = config['bingx']['testnet']
client = BingXClient(
    api_key=testnet_config['api_key'],
    secret_key=testnet_config['secret_key'],
    testnet=True
)
print("  ✅ API initialized (Testnet)")

# ============================================================================
# STEP 2: Fetch latest data
# ============================================================================
print("\n📊 Fetching latest data from BingX...")

# Get last 1440 candles (5 days = enough for feature calculation)
klines = client.get_klines(
    symbol="BTC-USDT",
    interval="5m",
    limit=1440
)

print(f"  ✅ Fetched {len(klines)} candles")

# Convert to DataFrame
df_latest = pd.DataFrame(klines)
df_latest['timestamp'] = pd.to_datetime(df_latest['time'], unit='ms')
for col in ['open', 'high', 'low', 'close', 'volume']:
    df_latest[col] = df_latest[col].astype(float)
df_latest['timestamp'] = df_latest['timestamp'].astype(str)

print(f"  Time range: {df_latest['timestamp'].iloc[0]} to {df_latest['timestamp'].iloc[-1]}")
print(f"  Price range: ${df_latest['close'].min():,.2f} - ${df_latest['close'].max():,.2f}")
print(f"  Current price: ${df_latest['close'].iloc[-1]:,.2f}")

# ============================================================================
# STEP 3: Calculate features
# ============================================================================
print("\n🔧 Calculating features...")
df_features = calculate_all_features(df_latest.copy())
df_features = prepare_exit_features(df_features)
print(f"  ✅ Features calculated: {len(df_features):,} candles")

# ============================================================================
# STEP 4: Load Trade-Outcome models
# ============================================================================
print("\n📁 Loading Trade-Outcome models...")

long_model_path = MODELS_DIR / "xgboost_long_trade_outcome_full_20251018_233146.pkl"
with open(long_model_path, 'rb') as f:
    long_model = pickle.load(f)

long_scaler_path = MODELS_DIR / "xgboost_long_trade_outcome_full_20251018_233146_scaler.pkl"
long_scaler = joblib.load(long_scaler_path)

long_features_path = MODELS_DIR / "xgboost_long_trade_outcome_full_20251018_233146_features.txt"
with open(long_features_path, 'r') as f:
    long_feature_names = [line.strip() for line in f.readlines()]

print(f"  ✅ Trade-Outcome LONG model loaded")
print(f"     Features: {len(long_feature_names)}")

# ============================================================================
# STEP 5: Filter Oct 19 data (today)
# ============================================================================
print("\n📅 Filtering Oct 19 data...")

df_oct19 = df_features[df_features['timestamp'].str.startswith('2025-10-19')].copy()

if len(df_oct19) == 0:
    print("  ⚠️  No Oct 19 data in current fetch")
    print("  Using last 288 candles (1 day) as 'latest day'")
    df_oct19 = df_features.tail(288).copy()
    latest_date = df_oct19['timestamp'].iloc[0][:10]
    print(f"  Latest day: {latest_date}")
else:
    print(f"  Oct 19 candles: {len(df_oct19)}")

print(f"  Time range: {df_oct19['timestamp'].iloc[0]} to {df_oct19['timestamp'].iloc[-1]}")
print(f"  Price range: ${df_oct19['close'].min():,.2f} - ${df_oct19['close'].max():,.2f}")
print(f"  Avg price: ${df_oct19['close'].mean():,.2f}")

# ============================================================================
# STEP 6: Calculate probabilities
# ============================================================================
print("\n🔬 Calculating LONG probabilities for latest day...")

probs = []
timestamps = []
prices = []
macd_values = []
macd_diff_values = []

for idx in range(len(df_oct19)):
    row = df_oct19.iloc[idx:idx+1]
    features = row[long_feature_names].values
    scaled = long_scaler.transform(features)
    prob = long_model.predict_proba(scaled)[0][1]
    probs.append(prob)
    timestamps.append(row['timestamp'].iloc[0])
    prices.append(row['close'].iloc[0])
    macd_values.append(row['macd'].iloc[0])
    macd_diff_values.append(row['macd_diff'].iloc[0])

probs = np.array(probs)

# ============================================================================
# STEP 7: Analysis
# ============================================================================
print(f"\n" + "="*80)
print(f"LATEST DAY PROBABILITY DISTRIBUTION")
print(f"="*80)

print(f"\n📊 Statistics:")
print(f"  Candles: {len(probs)}")
print(f"  Mean: {probs.mean():.4f} ({probs.mean()*100:.2f}%)")
print(f"  Median: {np.median(probs):.4f} ({np.median(probs)*100:.2f}%)")
print(f"  Std Dev: {probs.std():.4f}")
print(f"  Min: {probs.min():.4f} ({probs.min()*100:.2f}%)")
print(f"  Max: {probs.max():.4f} ({probs.max()*100:.2f}%)")

print(f"\n📊 Distribution:")
print(f"  ≥90%: {(probs >= 0.9).sum():4d} ({(probs >= 0.9).sum()/len(probs)*100:5.1f}%)")
print(f"  ≥80%: {(probs >= 0.8).sum():4d} ({(probs >= 0.8).sum()/len(probs)*100:5.1f}%)")
print(f"  ≥70%: {(probs >= 0.7).sum():4d} ({(probs >= 0.7).sum()/len(probs)*100:5.1f}%)")
print(f"  ≥65%: {(probs >= 0.65).sum():4d} ({(probs >= 0.65).sum()/len(probs)*100:5.1f}%)")
print(f"  <65%: {(probs < 0.65).sum():4d} ({(probs < 0.65).sum()/len(probs)*100:5.1f}%)")

print(f"\n📈 MACD Analysis:")
print(f"  MACD Mean: {np.mean(macd_values):.2f}")
print(f"  MACD_diff Mean: {np.mean(macd_diff_values):.2f}")

# ============================================================================
# STEP 8: Sample output (last 30 candles)
# ============================================================================
print(f"\n📋 Sample LONG Probabilities (last 30 candles):")
print(f"{'Timestamp':<20} {'Price':<12} {'LONG Prob':<12} {'MACD':<10} {'MACD_diff':<12} {'Threshold':<12}")
print("-"*100)

start_idx = max(0, len(timestamps) - 30)
for i in range(start_idx, len(timestamps)):
    ts = timestamps[i]
    price = prices[i]
    prob = probs[i]
    macd_val = macd_values[i]
    macd_diff_val = macd_diff_values[i]
    threshold = "✅ ENTRY" if prob >= 0.65 else "⏸ WAIT"
    print(f"{ts:<20} ${price:<11,.2f} {prob*100:<11.2f}% {macd_val:<9.2f} {macd_diff_val:<11.2f} {threshold:<12}")

# ============================================================================
# STEP 9: Comparison with Oct 18
# ============================================================================
print(f"\n" + "="*80)
print(f"COMPARISON WITH OCT 18")
print(f"="*80)

print(f"\n📊 Oct 18 (from previous analysis):")
print(f"  Mean: 80.66%")
print(f"  ≥70%: 84.1%")
print(f"  MACD: +2.87")
print(f"  MACD_diff: +2.09")

print(f"\n📊 Latest Day:")
print(f"  Mean: {probs.mean()*100:.2f}%")
print(f"  ≥70%: {(probs >= 0.7).sum()/len(probs)*100:.1f}%")
print(f"  MACD: {np.mean(macd_values):.2f}")
print(f"  MACD_diff: {np.mean(macd_diff_values):.2f}")

diff = probs.mean()*100 - 80.66
print(f"\n🔍 Difference: {diff:+.2f}%")

# ============================================================================
# STEP 10: Final verdict
# ============================================================================
print(f"\n" + "="*80)
print("FINAL VERDICT - LATEST DAY")
print("="*80)

mean_prob = probs.mean() * 100

if mean_prob > 75:
    print(f"\n⚠️  LATEST DAY shows HIGH probabilities:")
    print(f"   Average: {mean_prob:.2f}%")
    print(f"   ≥70% candles: {(probs >= 0.7).sum()/len(probs)*100:.1f}%")
    print(f"\n   → Similar to Oct 18 pattern")
    print(f"   → MACD indicators likely showing positive trend")
    print(f"   → Model sees strong LONG opportunities")
    print(f"\n   ⚠️  CAUTION: Oct 18 showed similar pattern")
    print(f"   → Monitor actual market behavior")
    print(f"   → Consider if mean reversion is happening")
elif mean_prob > 50:
    print(f"\n⚖️  LATEST DAY shows MODERATE probabilities:")
    print(f"   Average: {mean_prob:.2f}%")
    print(f"   ≥65% candles: {(probs >= 0.65).sum()/len(probs)*100:.1f}%")
    print(f"\n   → Normal entry opportunities")
    print(f"   → Model showing balanced signals")
    print(f"   → Lower than Oct 18 (80.66%)")
else:
    print(f"\n✅ LATEST DAY shows NORMAL probabilities:")
    print(f"   Average: {mean_prob:.2f}%")
    print(f"   Similar to baseline (Sep 23: 13.43%)")
    print(f"\n   → Model operating in normal range")
    print(f"   → Significantly lower than Oct 18")

# Current status
print(f"\n📍 CURRENT STATUS:")
print(f"  Latest timestamp: {timestamps[-1]}")
print(f"  Latest price: ${prices[-1]:,.2f}")
print(f"  Latest LONG prob: {probs[-1]*100:.2f}%")
print(f"  Latest MACD: {macd_values[-1]:.2f}")
print(f"  Latest MACD_diff: {macd_diff_values[-1]:.2f}")

if probs[-1] >= 0.65:
    print(f"  → ✅ ENTRY SIGNAL (≥65%)")
else:
    print(f"  → ⏸ WAIT (< 65%)")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
