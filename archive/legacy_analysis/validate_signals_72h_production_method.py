"""
72-Hour Signal Validation - Production Method
==============================================

Compare backtest signals with actual production bot signals
using PRODUCTION'S EXACT DATA LOADING METHOD.

Key Change: Load 1000 candles from API (like production), not CSV.

Period: 2025-10-24 01:00 ~ 2025-10-27 01:00 (72 hours)
"""

import pandas as pd
import numpy as np
import pickle
import joblib
import yaml
from pathlib import Path
import sys
from datetime import datetime, timedelta

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiments.calculate_all_features_enhanced_v2 import calculate_all_features_enhanced_v2
from src.api.bingx_client import BingXClient

MODELS_DIR = PROJECT_ROOT / "models"
CONFIG_PATH = PROJECT_ROOT / "config" / "api_keys.yaml"

print("="*80)
print("72-HOUR SIGNAL VALIDATION - PRODUCTION METHOD")
print("="*80)
print()

# Load API keys
print("Loading API credentials...")
with open(CONFIG_PATH, 'r') as f:
    config = yaml.safe_load(f)
    api_key = config['bingx']['mainnet']['api_key']
    secret_key = config['bingx']['mainnet']['secret_key']

client = BingXClient(api_key, secret_key)
print("✅ BingX client initialized")
print()

# Load Entry Models (Same as production bot)
print("Loading Entry models...")
long_model_path = MODELS_DIR / "xgboost_long_entry_enhanced_20251024_012445.pkl"
with open(long_model_path, 'rb') as f:
    long_model = pickle.load(f)

long_scaler_path = MODELS_DIR / "xgboost_long_entry_enhanced_20251024_012445_scaler.pkl"
long_scaler = joblib.load(long_scaler_path)

long_features_path = MODELS_DIR / "xgboost_long_entry_enhanced_20251024_012445_features.txt"
with open(long_features_path, 'r') as f:
    long_feature_columns = [line.strip() for line in f.readlines()]

short_model_path = MODELS_DIR / "xgboost_short_entry_enhanced_20251024_012445.pkl"
with open(short_model_path, 'rb') as f:
    short_model = pickle.load(f)

short_scaler_path = MODELS_DIR / "xgboost_short_entry_enhanced_20251024_012445_scaler.pkl"
short_scaler = joblib.load(short_scaler_path)

short_features_path = MODELS_DIR / "xgboost_short_entry_enhanced_20251024_012445_features.txt"
with open(short_features_path, 'r') as f:
    short_feature_columns = [line.strip() for line in f.readlines()]

print(f"✅ LONG model loaded ({len(long_feature_columns)} features)")
print(f"✅ SHORT model loaded ({len(short_feature_columns)} features)")
print()

# Load data using PRODUCTION METHOD (API with 1000 candles)
print("="*80)
print("LOADING DATA - PRODUCTION METHOD (API, 1000 candles)")
print("="*80)
print()

print("Fetching 1000 candles from BingX API (exactly as production does)...")
try:
    candles = client.get_klines("BTC-USDT", "5m", limit=1000)
    print(f"✅ Fetched {len(candles)} candles from API")
    print()

    # Convert to DataFrame (exactly as production does)
    df_all = pd.DataFrame(candles)
    df_all.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    df_all['timestamp'] = pd.to_datetime(df_all['timestamp'], unit='ms')

    # Convert numeric columns
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df_all[col] = pd.to_numeric(df_all[col], errors='coerce')

    df_all = df_all.sort_values('timestamp').reset_index(drop=True)

    print(f"Loaded {len(df_all)} candles")
    print(f"Date range: {df_all['timestamp'].min()} to {df_all['timestamp'].max()}")
    print()

except Exception as e:
    print(f"❌ Error fetching data: {e}")
    sys.exit(1)

# Calculate features on ALL 1000 candles (as production does)
print("Calculating features on all 1000 candles...")
df_features = calculate_all_features_enhanced_v2(df_all)
print(f"✅ Features calculated: {len(df_features.columns)} columns")
print(f"   Valid rows after feature calculation: {len(df_features)}")
print(f"   Data range after features: {df_features['timestamp'].min()} to {df_features['timestamp'].max()}")
print()

# NOW filter to validation period (AFTER feature calculation)
validation_start_utc = pd.Timestamp('2025-10-24 00:55:00')  # 2025-10-24 09:55 KST
validation_end_utc = pd.Timestamp('2025-10-27 01:00:00')    # 2025-10-27 10:00 KST

print("Filtering to validation period (AFTER features)...")
df_72h = df_features[(df_features['timestamp'] >= validation_start_utc) &
                      (df_features['timestamp'] <= validation_end_utc)].copy()

print(f"Validation data: {len(df_72h)} candles")
print(f"Validation range: {df_72h['timestamp'].min()} to {df_72h['timestamp'].max()}")
print()

# Generate signals for validation period
print("Generating signals for validation period...")
signals = []

for idx, row in df_72h.iterrows():
    timestamp = row['timestamp']
    price = row['close']

    # LONG signal
    try:
        long_feat_df = pd.DataFrame([row])[long_feature_columns]
        if long_feat_df.shape[1] == len(long_feature_columns):
            long_feat_scaled = long_scaler.transform(long_feat_df.values)
            long_prob = long_model.predict_proba(long_feat_scaled)[0, 1]
        else:
            long_prob = 0.0
    except Exception as e:
        long_prob = 0.0

    # SHORT signal
    try:
        short_feat_df = pd.DataFrame([row])[short_feature_columns]
        if short_feat_df.shape[1] == len(short_feature_columns):
            short_feat_scaled = short_scaler.transform(short_feat_df.values)
            short_prob = short_model.predict_proba(short_feat_scaled)[0, 1]
        else:
            short_prob = 0.0
    except Exception as e:
        short_prob = 0.0

    signals.append({
        'timestamp': timestamp,
        'price': price,
        'long_prob': long_prob,
        'short_prob': short_prob
    })

df_signals = pd.DataFrame(signals)
print(f"✅ Signals generated for {len(df_signals)} candles")
print()

# Filter for SHORT >= 0.8 signals
df_short_high = df_signals[df_signals['short_prob'] >= 0.8].copy()

print("="*80)
print("SHORT SIGNALS >= 0.8 (PRODUCTION METHOD)")
print("="*80)
print()

if len(df_short_high) == 0:
    print("❌ NO SHORT signals >= 0.8 found")
else:
    print(f"✅ Found {len(df_short_high)} SHORT signals >= 0.8:")
    print()
    for idx, row in df_short_high.iterrows():
        ts = row['timestamp']
        # Convert to KST (UTC+9)
        ts_kst = ts + timedelta(hours=9)
        print(f"   {ts_kst.strftime('%Y-%m-%d %H:%M:%S')} KST")
        print(f"   Price: ${row['price']:,.1f}")
        print(f"   LONG: {row['long_prob']:.4f}")
        print(f"   SHORT: {row['short_prob']:.4f}")
        print()

print("="*80)
print("COMPARISON WITH LIVE BOT SIGNALS")
print("="*80)
print()

# Expected signals from live logs
live_signals = [
    {
        'time_kst': '2025-10-24 01:00:00',
        'price': 109966.0,
        'long_prob': 0.3707,
        'short_prob': 0.8122
    },
    {
        'time_kst': '2025-10-24 01:10:00',
        'price': 109958.0,
        'long_prob': 0.3128,
        'short_prob': 0.8886
    }
]

print("Expected signals from live bot:")
for sig in live_signals:
    print(f"   {sig['time_kst']} KST - SHORT: {sig['short_prob']:.4f} @ ${sig['price']:,.1f}")
print()

# Match backtest signals with live signals
print("Matching backtest with live signals:")
print()

matches_found = 0
for live_sig in live_signals:
    # Convert KST to UTC for matching
    live_ts_kst = pd.Timestamp(live_sig['time_kst'])
    live_ts_utc = live_ts_kst - timedelta(hours=9)

    # Find matching timestamp in backtest (within 5 minutes)
    matches = df_signals[
        (df_signals['timestamp'] >= live_ts_utc - timedelta(minutes=2.5)) &
        (df_signals['timestamp'] <= live_ts_utc + timedelta(minutes=2.5))
    ]

    if len(matches) > 0:
        match = matches.iloc[0]

        # Calculate differences
        price_diff = match['price'] - live_sig['price']
        long_diff = match['long_prob'] - live_sig['long_prob']
        short_diff = match['short_prob'] - live_sig['short_prob']

        print(f"✅ MATCH FOUND: {live_sig['time_kst']} KST")
        print(f"   Live:     LONG={live_sig['long_prob']:.4f}, SHORT={live_sig['short_prob']:.4f}, Price=${live_sig['price']:,.1f}")
        print(f"   Backtest: LONG={match['long_prob']:.4f}, SHORT={match['short_prob']:.4f}, Price=${match['price']:,.1f}")
        print(f"   Diff:     LONG={long_diff:+.4f}, SHORT={short_diff:+.4f}, Price=${price_diff:+.1f}")

        # Check if within acceptable tolerance
        if abs(short_diff) < 0.01 and abs(long_diff) < 0.01:
            print(f"   ✅ SIGNALS MATCH (within 1% tolerance)")
            matches_found += 1
        else:
            print(f"   ⚠️  SIGNALS DIFFER (outside 1% tolerance)")
        print()
    else:
        print(f"❌ NO MATCH: {live_sig['time_kst']} KST")
        print()

print("="*80)
print("SUMMARY")
print("="*80)
print()

print(f"Live signals (SHORT >= 0.8): {len(live_signals)}")
print(f"Backtest signals (SHORT >= 0.8): {len(df_short_high)}")
print(f"Matches found: {matches_found}/{len(live_signals)}")
print()

if len(df_short_high) == len(live_signals) and matches_found == len(live_signals):
    print("✅ PERFECT MATCH - Backtest reproduces production signals!")
    print()
    print("Key Success Factors:")
    print("  - Used production method: API with 1000 candles")
    print("  - Feature calculation on full dataset")
    print("  - Filtering AFTER features (not before)")
    print("  - Identical models and thresholds")
elif matches_found > 0:
    print(f"⚠️  PARTIAL MATCH - {matches_found}/{len(live_signals)} signals matched")
else:
    print(f"❌ NO MATCH - Backtest differs from production")
    print()
    print("Possible issues:")
    print("  - Data discrepancy between API and live bot")
    print("  - Feature calculation differences")
    print("  - Model/scaler version mismatch")

print()
print("Validation complete!")
