"""
Check bullish_engulfing feature values over recent candles
Created: 2025-11-03
Purpose: Determine if z=5.39 outlier is persistent or temporary
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import sys
import yaml

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.api.bingx_client import BingXClient
from scripts.production.production_features_v1 import calculate_all_features_enhanced_v2
from scripts.production.production_exit_features_v1 import prepare_exit_features

# Load API keys
config_path = PROJECT_ROOT / "config" / "api_keys.yaml"
with open(config_path, 'r') as f:
    config = yaml.safe_load(f)

bingx_config = config['bingx']['mainnet']
client = BingXClient(
    api_key=bingx_config['api_key'],
    secret_key=bingx_config['secret_key']
)

# Fetch data
print("Fetching 1000 candles...")
klines = client.get_klines("BTC-USDT", "5m", limit=1000)
df = pd.DataFrame(klines)

df['timestamp'] = pd.to_datetime(df['time'], unit='ms', utc=True).dt.tz_localize(None)
for col in ['open', 'high', 'low', 'close', 'volume']:
    df[col] = df[col].astype(float)

print(f"✅ Fetched {len(df)} candles")

# Calculate features
print("Calculating features...")
df_features = calculate_all_features_enhanced_v2(df.copy(), phase='phase1')
df_features = prepare_exit_features(df_features)

print(f"✅ Features: {len(df_features)} rows")

# Load LONG Entry model components
MODELS_DIR = PROJECT_ROOT / "models"
long_scaler = joblib.load(MODELS_DIR / "xgboost_long_entry_enhanced_20251024_012445_scaler.pkl")
with open(MODELS_DIR / "xgboost_long_entry_enhanced_20251024_012445_features.txt", 'r') as f:
    long_features = [line.strip() for line in f.readlines()]

# Get bullish_engulfing index
try:
    engulfing_idx = long_features.index('bullish_engulfing')
except ValueError:
    print("❌ bullish_engulfing not found in feature list!")
    sys.exit(1)

# Analyze recent 20 candles
print("\n" + "="*80)
print("BULLISH_ENGULFING ANALYSIS - RECENT 20 CANDLES")
print("="*80)

recent = df_features.iloc[-20:]

print(f"\nTime range: {recent.iloc[0]['timestamp']} to {recent.iloc[-1]['timestamp']}")
print(f"Price range: ${recent.iloc[0]['close']:,.1f} to ${recent.iloc[-1]['close']:,.1f}")

print("\nRaw values and normalized z-scores:")
print("-" * 80)

raw_values = []
z_scores = []

for i in range(len(recent)):
    candle = recent.iloc[i:i+1]
    timestamp = candle['timestamp'].iloc[0]
    price = candle['close'].iloc[0]

    # Get raw bullish_engulfing value
    raw_val = candle['bullish_engulfing'].iloc[0]

    # Get all features and normalize
    feat_df = candle[long_features]
    feat_normalized = long_scaler.transform(feat_df.values)
    z_score = feat_normalized[0][engulfing_idx]

    raw_values.append(raw_val)
    z_scores.append(z_score)

    status = "✅" if abs(z_score) <= 3 else "⚠️  OUTLIER" if abs(z_score) <= 5 else "🚨 EXTREME"

    print(f"{i+1:2d}. {timestamp} | Price: ${price:>10,.1f} | "
          f"Raw: {raw_val:4.1f} | Z-score: {z_score:7.2f}  {status}")

# Summary statistics
print("\n" + "="*80)
print("SUMMARY STATISTICS")
print("="*80)

raw_array = np.array(raw_values)
z_array = np.array(z_scores)

print(f"\nRaw Values:")
print(f"  Min: {raw_array.min():.1f}")
print(f"  Max: {raw_array.max():.1f}")
print(f"  Mean: {raw_array.mean():.1f}")
print(f"  Std: {raw_array.std():.1f}")
print(f"  Values = 1.0: {(raw_array == 1.0).sum()} / 20 ({(raw_array == 1.0).sum()/20*100:.1f}%)")
print(f"  Values = 0.0: {(raw_array == 0.0).sum()} / 20 ({(raw_array == 0.0).sum()/20*100:.1f}%)")

print(f"\nZ-scores (Normalized):")
print(f"  Min: {z_array.min():.2f}")
print(f"  Max: {z_array.max():.2f}")
print(f"  Mean: {z_array.mean():.2f}")
print(f"  Std: {z_array.std():.2f}")
print(f"  |Z| > 3: {(np.abs(z_array) > 3).sum()} / 20 ({(np.abs(z_array) > 3).sum()/20*100:.1f}%)")
print(f"  |Z| > 5: {(np.abs(z_array) > 5).sum()} / 20 ({(np.abs(z_array) > 5).sum()/20*100:.1f}%)")

# Scaler statistics
scaler_mean = long_scaler.mean_[engulfing_idx]
scaler_std = long_scaler.scale_[engulfing_idx]

print(f"\nScaler Parameters (Training Time):")
print(f"  Mean: {scaler_mean:.6f}")
print(f"  Std: {scaler_std:.6f}")

# Analysis
print("\n" + "="*80)
print("ANALYSIS")
print("="*80)

outlier_count = (np.abs(z_array) > 3).sum()
extreme_count = (np.abs(z_array) > 5).sum()

if extreme_count > 0:
    print(f"\n⚠️  PERSISTENT EXTREME OUTLIERS DETECTED")
    print(f"   - {extreme_count} / 20 candles have |Z| > 5 ({extreme_count/20*100:.1f}%)")
    print(f"   - This feature is CONSISTENTLY producing extreme values")
    print(f"   - Likely contributing to high LONG probabilities (0.80+)")
elif outlier_count > 5:
    print(f"\n⚠️  FREQUENT OUTLIERS DETECTED")
    print(f"   - {outlier_count} / 20 candles have |Z| > 3 ({outlier_count/20*100:.1f}%)")
    print(f"   - Feature frequently outside normal range")
    print(f"   - May be contributing to elevated probabilities")
else:
    print(f"\n✅ NORMAL BEHAVIOR")
    print(f"   - Only {outlier_count} / 20 candles outside normal range")
    print(f"   - Feature behaving as expected")

# Binary feature analysis
if raw_array.max() <= 1.0 and raw_array.min() >= 0.0:
    print(f"\n📊 BINARY FEATURE CHARACTERISTICS:")
    print(f"   - This is a binary (0/1) feature")
    print(f"   - Scaler mean: {scaler_mean:.6f} (training data had {scaler_mean*100:.2f}% occurrences)")
    print(f"   - Recent data: {raw_array.mean()*100:.1f}% occurrences (20 candles)")

    if raw_array.mean() > scaler_mean * 2:
        print(f"\n⚠️  REGIME CHANGE DETECTED:")
        print(f"   - Current occurrence rate: {raw_array.mean()*100:.1f}%")
        print(f"   - Training occurrence rate: {scaler_mean*100:.2f}%")
        print(f"   - {raw_array.mean()/scaler_mean:.1f}x higher than training!")
        print(f"   - Market showing more bullish engulfing patterns than training period")

print("\n" + "="*80)
