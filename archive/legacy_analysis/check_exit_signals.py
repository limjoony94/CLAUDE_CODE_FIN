#!/usr/bin/env python3
"""
Check ML Exit Signals for Last 4 Hours
직접 데이터를 입력해서 ML Exit 모델의 출력을 확인합니다.
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import pickle
import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.api.bingx_client import BingXClient

# Import feature calculation from production bot
sys.path.insert(0, str(PROJECT_ROOT / "scripts" / "experiments"))
from calculate_all_features_enhanced_v2 import calculate_all_features_enhanced_v2
from retrain_exit_models_opportunity_gating import prepare_exit_features

# Paths
MODELS_DIR = PROJECT_ROOT / "models"
CONFIG_DIR = PROJECT_ROOT / "config"

# ML Exit 모델 로드
print("=" * 80)
print("ML Exit Signal Analysis - Last 4 Hours")
print("=" * 80)

# Load SHORT Exit Model
short_exit_model_path = MODELS_DIR / "xgboost_short_exit_oppgating_improved_20251024_044510.pkl"
short_scaler_path = MODELS_DIR / "xgboost_short_exit_oppgating_improved_20251024_044510_scaler.pkl"
short_features_path = MODELS_DIR / "xgboost_short_exit_oppgating_improved_20251024_044510_features.txt"

print(f"\n📦 Loading SHORT Exit Model...")
print(f"   Model: {short_exit_model_path.name}")

with open(short_exit_model_path, 'rb') as f:
    model_data = pickle.load(f)
    # Check if it's a dict or direct model object
    if isinstance(model_data, dict):
        short_exit_model = model_data['model']
        short_feature_columns = model_data['feature_columns']
    else:
        # Direct XGBClassifier object - load feature columns from text file
        short_exit_model = model_data
        short_feature_columns = None

with open(short_scaler_path, 'rb') as f:
    short_scaler = pickle.load(f)

# Load feature column names from text file
if short_feature_columns is None:
    with open(short_features_path, 'r') as f:
        short_feature_columns = [line.strip() for line in f.readlines() if line.strip()]

print(f"   Features: {len(short_feature_columns)} features")
print(f"   ✅ Model loaded")

# Initialize API Client
print(f"\n🔌 Initializing BingX Client...")
api_keys_file = CONFIG_DIR / "api_keys.yaml"
with open(api_keys_file, 'r', encoding='utf-8') as f:
    config = yaml.safe_load(f)

api_config = config.get('bingx', {}).get('mainnet', {})
api_key = api_config.get('api_key')
api_secret = api_config.get('secret_key')

client = BingXClient(api_key, api_secret, testnet=False)
print(f"   ✅ Client initialized")

# Get last 4 hours of data (5-min candles = 48 candles)
print(f"\n📊 Fetching last 4 hours of data...")
symbol = "BTC-USDT"
interval = "5m"
limit = 48

klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)

# Convert to DataFrame (dict keys become columns automatically)
df = pd.DataFrame(klines)

# Convert timestamp from ms to datetime (UTC → KST)
df['timestamp'] = pd.to_datetime(df['time'], unit='ms', utc=True).dt.tz_convert('Asia/Seoul').dt.tz_localize(None)

# Convert price/volume columns from string to float
for col in ['open', 'high', 'low', 'close', 'volume']:
    df[col] = df[col].astype(float)

print(f"   ✅ Fetched {len(df)} candles")
print(f"   Time range: {df['timestamp'].iloc[0]} → {df['timestamp'].iloc[-1]}")

# Calculate features using production bot function
print(f"\n🔧 Using production bot feature calculation...")

# Check exit signal for each of the last 48 candles
print(f"\n🎯 Checking ML Exit signals for each candle...")
print(f"\n{'Time':<20} {'Close Price':>12} {'Exit Prob':>10} {'Status':>12}")
print("-" * 60)

exit_signals = []
for i in range(len(df)):
    # Use data up to this point
    df_slice = df.iloc[:i+1].copy()

    if len(df_slice) < 20:
        continue  # Need at least 20 candles for features

    # Calculate features
    try:
        # Calculate ALL features using production bot's method
        features_all = calculate_all_features_enhanced_v2(df_slice.copy(), phase='phase1')
        features_all = prepare_exit_features(features_all)  # Add EXIT-specific features

        # Extract SHORT Exit features
        # The model expects 27 exit features
        if short_feature_columns:
            # Use model's feature columns
            feature_df = features_all[short_feature_columns]
        else:
            # Use all numeric features (exclude timestamp columns)
            exclude_cols = ['timestamp', 'time']
            numeric_cols = [col for col in features_all.columns if col not in exclude_cols]
            feature_df = features_all[numeric_cols]

        # Select only the last row (most recent data)
        feature_row = feature_df.iloc[[-1]].copy()

        # Handle inf/nan values
        feature_row = feature_row.replace([np.inf, -np.inf], np.nan)
        feature_row = feature_row.fillna(0)

        # Scale features
        features_scaled = short_scaler.transform(feature_row)

        # Predict
        exit_prob = short_exit_model.predict_proba(features_scaled)[0][1]

        # Store result
        candle_time = df_slice['timestamp'].iloc[-1]
        close_price = df_slice['close'].iloc[-1]

        exit_signals.append({
            'time': candle_time,
            'close': close_price,
            'exit_prob': exit_prob
        })

        # Print if probability > 0.05 (5%)
        if exit_prob >= 0.05:
            status = "🔴 HIGH" if exit_prob >= 0.80 else "🟡 MEDIUM" if exit_prob >= 0.30 else "🟢 LOW"
            print(f"{candle_time:<20} ${close_price:>11,.2f} {exit_prob:>9.3f} {status:>12}")

    except Exception as e:
        print(f"Error at candle {i}: {e}")
        continue

# Summary
print("\n" + "=" * 80)
print("📊 SUMMARY")
print("=" * 80)

if exit_signals:
    exit_probs = [s['exit_prob'] for s in exit_signals]
    max_prob = max(exit_probs)
    max_idx = exit_probs.index(max_prob)
    max_signal = exit_signals[max_idx]

    print(f"\nTotal candles analyzed: {len(exit_signals)}")
    print(f"Max exit probability: {max_prob:.3f} ({max_prob*100:.1f}%)")
    print(f"   Time: {max_signal['time']}")
    print(f"   Price: ${max_signal['close']:,.2f}")
    print(f"   Threshold: 0.80 (80%)")
    print(f"   Distance from threshold: {(0.80 - max_prob)*100:.1f}% ({max_prob/0.80*100:.1f}% of threshold)")

    # Count signals above certain levels
    high_signals = [s for s in exit_signals if s['exit_prob'] >= 0.50]
    medium_signals = [s for s in exit_signals if 0.20 <= s['exit_prob'] < 0.50]

    print(f"\nSignal distribution:")
    print(f"   ≥ 50%: {len(high_signals)} candles")
    print(f"   20-50%: {len(medium_signals)} candles")
    print(f"   < 20%: {len(exit_signals) - len(high_signals) - len(medium_signals)} candles")

    # ML Exit would trigger?
    if max_prob >= 0.80:
        print(f"\n✅ ML Exit WOULD TRIGGER (probability >= 0.80)")
    else:
        print(f"\n❌ ML Exit would NOT trigger (max probability: {max_prob:.3f} < 0.80)")

print("\n" + "=" * 80)
