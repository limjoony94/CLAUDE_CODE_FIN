#!/usr/bin/env python3
"""
Test Lookback Window Hypothesis
================================
Goal: Prove that longer lookback periods make backtest converge to production

Method:
1. Run backtest with 1 day lookback (short)
2. Run backtest with 5 days lookback (current)
3. Run backtest with 10 days lookback (long)
4. Run backtest with 20 days lookback (very long)
5. Compare signal convergence to production

Expected Result:
- If hypothesis correct: Longer lookback → closer to production
- Signals should converge as lookback increases
"""

import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from datetime import datetime, timedelta
import pandas as pd
import numpy as np
import joblib
import yaml
from pathlib import Path
import pytz
from src.api.bingx_client import BingXClient
from scripts.experiments.calculate_all_features_enhanced_v2 import calculate_all_features_enhanced_v2
from scripts.production.production_exit_features_v1 import prepare_exit_features

# Configuration
KST = pytz.timezone('Asia/Seoul')
UTC = pytz.UTC

# Test single candle (worst mismatch)
TEST_TIME = datetime(2025, 11, 2, 21, 0, tzinfo=KST)  # Worst LONG case: prod 0.7736, backtest 0.3308
PRODUCTION_LONG = 0.7736
PRODUCTION_SHORT = 0.1210

# Lookback periods to test (skip 1 day - insufficient for VP/VWAP)
LOOKBACK_DAYS = [2, 5, 10, 15, 20, 25, 30]

# Models
MODELS_DIR = Path("models")
LONG_ENTRY_MODEL = MODELS_DIR / "xgboost_long_entry_enhanced_20251024_012445.pkl"
LONG_ENTRY_SCALER = MODELS_DIR / "xgboost_long_entry_enhanced_20251024_012445_scaler.pkl"
LONG_ENTRY_FEATURES = MODELS_DIR / "xgboost_long_entry_enhanced_20251024_012445_features.txt"

SHORT_ENTRY_MODEL = MODELS_DIR / "xgboost_short_entry_enhanced_20251024_012445.pkl"
SHORT_ENTRY_SCALER = MODELS_DIR / "xgboost_short_entry_enhanced_20251024_012445_scaler.pkl"
SHORT_ENTRY_FEATURES = MODELS_DIR / "xgboost_short_entry_enhanced_20251024_012445_features.txt"

def test_lookback(days):
    """Test backtest with specific lookback period"""
    print(f"\n{'='*60}")
    print(f"Testing {days}-day lookback")
    print(f"{'='*60}")

    # Load API credentials
    with open('config/api_keys.yaml', 'r') as f:
        api_config = yaml.safe_load(f)

    client = BingXClient(
        api_key=api_config['bingx']['mainnet']['api_key'],
        secret_key=api_config['bingx']['mainnet']['secret_key'],
        testnet=False
    )

    # Calculate time range
    test_utc = TEST_TIME.astimezone(UTC).replace(tzinfo=None)
    start_utc = test_utc - timedelta(days=days)

    # Fetch candles
    total_candles = days * 288  # 288 5-min candles per day
    print(f"Requesting {total_candles} candles (from {start_utc} to {test_utc})")

    raw_candles = client.get_klines(symbol="BTC-USDT", interval="5m", limit=total_candles)

    # BingX returns list of dicts with 'time' key (not 'timestamp')
    df = pd.DataFrame(raw_candles)
    df['timestamp'] = pd.to_datetime(df['time'], unit='ms', utc=True).dt.tz_localize(None)
    df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]

    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)

    print(f"Received {len(df)} candles")

    # Calculate features
    print("Calculating features...")
    df_features = calculate_all_features_enhanced_v2(df)
    df_features = prepare_exit_features(df_features)

    print(f"After feature calculation: {len(df_features)} rows")

    # Find test candle
    matching = df_features[df_features['timestamp'] == test_utc]

    if len(matching) == 0:
        print(f"❌ Test candle NOT FOUND (lost to lookback)")
        return None, None

    row = matching.iloc[0]

    # Load models
    long_entry_model = joblib.load(LONG_ENTRY_MODEL)
    long_entry_scaler = joblib.load(LONG_ENTRY_SCALER)
    with open(LONG_ENTRY_FEATURES, 'r') as f:
        long_entry_features = [line.strip() for line in f if line.strip()]

    short_entry_model = joblib.load(SHORT_ENTRY_MODEL)
    short_entry_scaler = joblib.load(SHORT_ENTRY_SCALER)
    with open(SHORT_ENTRY_FEATURES, 'r') as f:
        short_entry_features = [line.strip() for line in f if line.strip()]

    # Generate signals
    long_feat = row[long_entry_features].values.reshape(1, -1)
    long_feat_scaled = long_entry_scaler.transform(long_feat)
    long_prob = long_entry_model.predict_proba(long_feat_scaled)[0][1]

    short_feat = row[short_entry_features].values.reshape(1, -1)
    short_feat_scaled = short_entry_scaler.transform(short_feat)
    short_prob = short_entry_model.predict_proba(short_feat_scaled)[0][1]

    # Calculate differences
    long_diff = abs(long_prob - PRODUCTION_LONG)
    short_diff = abs(short_prob - PRODUCTION_SHORT)

    print(f"\nResults:")
    print(f"  LONG:  {long_prob:.4f} (prod: {PRODUCTION_LONG}, diff: {long_diff:.4f})")
    print(f"  SHORT: {short_prob:.4f} (prod: {PRODUCTION_SHORT}, diff: {short_diff:.4f})")

    return long_diff, short_diff

def main():
    print("="*60)
    print("LOOKBACK WINDOW HYPOTHESIS TEST")
    print("="*60)
    print(f"\nTest Candle: {TEST_TIME.strftime('%Y-%m-%d %H:%M:%S KST')}")
    print(f"Production Signals: LONG {PRODUCTION_LONG:.4f}, SHORT {PRODUCTION_SHORT:.4f}")

    results = []

    for days in LOOKBACK_DAYS:
        long_diff, short_diff = test_lookback(days)
        if long_diff is not None:
            results.append({
                'lookback_days': days,
                'long_diff': long_diff,
                'short_diff': short_diff
            })

    # Summary
    print("\n" + "="*60)
    print("CONVERGENCE ANALYSIS")
    print("="*60)

    df_results = pd.DataFrame(results)
    print("\nLONG Signal Differences:")
    for _, row in df_results.iterrows():
        trend = ""
        if row.name > 0:
            prev_diff = df_results.iloc[row.name - 1]['long_diff']
            if row['long_diff'] < prev_diff:
                trend = "✅ IMPROVING"
            else:
                trend = "❌ WORSE"
        print(f"  {int(row['lookback_days']):2d} days: {row['long_diff']:.4f} {trend}")

    print("\nSHORT Signal Differences:")
    for _, row in df_results.iterrows():
        trend = ""
        if row.name > 0:
            prev_diff = df_results.iloc[row.name - 1]['short_diff']
            if row['short_diff'] < prev_diff:
                trend = "✅ IMPROVING"
            else:
                trend = "❌ WORSE"
        print(f"  {int(row['lookback_days']):2d} days: {row['short_diff']:.4f} {trend}")

    # Conclusion
    print("\n" + "="*60)
    print("CONCLUSION")
    print("="*60)

    if len(df_results) >= 2:
        long_improving = df_results['long_diff'].iloc[-1] < df_results['long_diff'].iloc[0]
        short_improving = df_results['short_diff'].iloc[-1] < df_results['short_diff'].iloc[0]

        if long_improving and short_improving:
            print("✅ HYPOTHESIS CONFIRMED")
            print("   Longer lookback periods converge closer to production")
            print("   Root cause: Data window mismatch")
        else:
            print("❌ HYPOTHESIS REJECTED")
            print("   Lookback period NOT the primary cause")
            print("   Need alternative explanation")

    # Save results
    output_file = "results/lookback_hypothesis_test_20251103.csv"
    df_results.to_csv(output_file, index=False)
    print(f"\nResults saved: {output_file}")

if __name__ == "__main__":
    main()
