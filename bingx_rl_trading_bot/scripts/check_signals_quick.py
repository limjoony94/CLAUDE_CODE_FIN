#!/usr/bin/env python3
"""
Quick script to check if trading signals occur with recent real data
Based on phase4_dynamic_paper_trading.py logic
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pickle
import requests
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.production.train_xgboost_improved_v3_phase2 import calculate_features
from scripts.production.advanced_technical_features import AdvancedTechnicalFeatures

# Configuration
MODEL_PATH = PROJECT_ROOT / "models" / "xgboost_v4_phase4_advanced_lookahead3_thresh0.pkl"
FEATURE_PATH = PROJECT_ROOT / "models" / "xgboost_v4_phase4_advanced_lookahead3_thresh0_features.txt"
THRESHOLD = 0.7
LOOKBACK_CANDLES = 500

def get_live_data():
    """Get live data from BingX Testnet API"""
    print(f"\n{'='*80}")
    print(f"📡 Fetching live data from BingX Testnet API...")
    print(f"{'='*80}\n")

    try:
        base_url = "https://open-api-vst.bingx.com"
        url = f"{base_url}/openApi/swap/v3/quote/klines"
        params = {
            "symbol": "BTC-USDT",
            "interval": "5m",
            "limit": min(LOOKBACK_CANDLES, 500)
        }

        response = requests.get(url, params=params, timeout=10)

        if response.status_code == 200:
            data = response.json()

            if data.get('code') == 0 and 'data' in data:
                klines = data['data']
                df = pd.DataFrame(klines)
                df = df.rename(columns={'time': 'timestamp'})
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
                df[['open', 'high', 'low', 'close', 'volume']] = \
                    df[['open', 'high', 'low', 'close', 'volume']].astype(float)
                df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]

                # Sort by timestamp (BingX returns newest first)
                df = df.sort_values('timestamp').reset_index(drop=True)

                print(f"✅ Fetched {len(df)} candles")
                print(f"   Latest: ${df['close'].iloc[-1]:,.2f} @ {df['timestamp'].iloc[-1]}")
                print(f"   Range: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")

                return df
        else:
            print(f"❌ API error: Status {response.status_code}")
            return None

    except Exception as e:
        print(f"❌ Error fetching data: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Main function"""
    print(f"\n{'='*80}")
    print(f"🤖 Phase 4 Dynamic Trading Signal Checker")
    print(f"{'='*80}\n")
    print(f"Model: {MODEL_PATH.name}")
    print(f"Threshold: {THRESHOLD}")

    # Load model
    print(f"\n📥 Loading XGBoost model...")
    try:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        with open(FEATURE_PATH, 'r') as f:
            feature_columns = [line.strip() for line in f.readlines()]
        print(f"✅ Model loaded: {len(feature_columns)} features")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    # Get live data
    df = get_live_data()
    if df is None:
        return

    # Calculate features
    print(f"\n{'='*80}")
    print(f"🔧 Calculating features...")
    print(f"{'='*80}\n")

    df = calculate_features(df)

    atf = AdvancedTechnicalFeatures(lookback_sr=50, lookback_trend=20)
    df = atf.calculate_all_features(df)

    print(f"   Rows before NaN handling: {len(df)}")
    df = df.ffill()
    df = df.dropna()
    print(f"   Rows after NaN handling: {len(df)}")

    if len(df) < 50:
        print(f"❌ Too few rows after NaN handling ({len(df)} < 50)")
        return

    # Check features
    missing = [col for col in feature_columns if col not in df.columns]
    if missing:
        print(f"❌ Missing features ({len(missing)}): {missing[:10]}")
        return

    print(f"✅ All features present")

    # Predict probabilities
    print(f"\n{'='*80}")
    print(f"🎯 Predicting probabilities...")
    print(f"{'='*80}\n")

    try:
        X = df[feature_columns].values
        probabilities = model.predict_proba(X)[:, 1]
        print(f"✅ Predictions complete: {len(probabilities)} probabilities")
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # Analyze signals
    print(f"\n{'='*80}")
    print(f"📊 Signal Analysis (LONG + SHORT)")
    print(f"{'='*80}\n")

    # Statistics
    print(f"Probability Statistics:")
    print(f"  Mean:   {probabilities.mean():.4f}")
    print(f"  Median: {np.median(probabilities):.4f}")
    print(f"  Std:    {probabilities.std():.4f}")
    print(f"  Min:    {probabilities.min():.4f}")
    print(f"  Max:    {probabilities.max():.4f}")
    print()

    # Distribution
    print(f"Probability Distribution:")
    print(f"  >= 0.9 (Strong LONG):  {(probabilities >= 0.9).sum():4d} ({(probabilities >= 0.9).sum() / len(probabilities) * 100:5.2f}%)")
    print(f"  >= 0.8:                {(probabilities >= 0.8).sum():4d} ({(probabilities >= 0.8).sum() / len(probabilities) * 100:5.2f}%)")
    print(f"  >= 0.7 (LONG Signal):  {(probabilities >= 0.7).sum():4d} ({(probabilities >= 0.7).sum() / len(probabilities) * 100:5.2f}%)")
    print(f"  0.3 - 0.7 (Neutral):   {((probabilities > 0.3) & (probabilities < 0.7)).sum():4d} ({((probabilities > 0.3) & (probabilities < 0.7)).sum() / len(probabilities) * 100:5.2f}%)")
    print(f"  <= 0.3 (SHORT Signal): {(probabilities <= 0.3).sum():4d} ({(probabilities <= 0.3).sum() / len(probabilities) * 100:5.2f}%)")
    print(f"  <= 0.2:                {(probabilities <= 0.2).sum():4d} ({(probabilities <= 0.2).sum() / len(probabilities) * 100:5.2f}%)")
    print(f"  <= 0.1 (Strong SHORT): {(probabilities <= 0.1).sum():4d} ({(probabilities <= 0.1).sum() / len(probabilities) * 100:5.2f}%)")
    print()

    # High confidence signals - LONG
    long_signals_mask = probabilities >= THRESHOLD
    long_signals_indices = np.where(long_signals_mask)[0]

    # High confidence signals - SHORT
    short_threshold = 1 - THRESHOLD
    short_signals_mask = probabilities <= short_threshold
    short_signals_indices = np.where(short_signals_mask)[0]

    total_signals = len(long_signals_indices) + len(short_signals_indices)

    # Display LONG signals
    if len(long_signals_indices) > 0:
        print(f"✅ LONG Signals (>= {THRESHOLD}): {len(long_signals_indices)}\n")

        # Show last 5 LONG signals
        for idx in long_signals_indices[-5:]:
            timestamp = df.iloc[idx]['timestamp']
            price = df.iloc[idx]['close']
            prob = probabilities[idx]
            print(f"   {timestamp}: Prob={prob:.3f}, Price=${price:,.2f}")

        if len(long_signals_indices) > 5:
            print(f"   ... and {len(long_signals_indices) - 5} more")
    else:
        print(f"⚠️  No LONG signals found (>= {THRESHOLD})")
        print(f"   Highest probability: {probabilities.max():.3f}")

    print()

    # Display SHORT signals
    if len(short_signals_indices) > 0:
        print(f"✅ SHORT Signals (<= {short_threshold}): {len(short_signals_indices)}\n")

        # Show last 5 SHORT signals
        for idx in short_signals_indices[-5:]:
            timestamp = df.iloc[idx]['timestamp']
            price = df.iloc[idx]['close']
            prob = probabilities[idx]
            print(f"   {timestamp}: Prob={prob:.3f}, Price=${price:,.2f}")

        if len(short_signals_indices) > 5:
            print(f"   ... and {len(short_signals_indices) - 5} more")
    else:
        print(f"⚠️  No SHORT signals found (<= {short_threshold})")
        print(f"   Lowest probability: {probabilities.min():.3f}")

    print()

    # Signal rate
    if total_signals > 0:
        signal_rate = total_signals / len(df) * 100
        print(f"   Total Signals: {total_signals} (LONG: {len(long_signals_indices)}, SHORT: {len(short_signals_indices)})")
        print(f"   Signal rate: {signal_rate:.2f}%")
        print(f"   Expected trades per 48h: ~{total_signals / len(df) * 576:.1f}")

    # Summary
    print(f"\n{'='*80}")
    print(f"📝 Summary (LONG + SHORT)")
    print(f"{'='*80}\n")

    if total_signals > 0:
        print(f"✅ Trading signals ARE occurring in real data!")
        print(f"   LONG signals (>= {THRESHOLD}): {len(long_signals_indices)}")
        print(f"   SHORT signals (<= {short_threshold}): {len(short_signals_indices)}")
        print(f"   Total signals: {total_signals}")
        print(f"   Signal rate: {signal_rate:.2f}%")
        print(f"   Paper trading should work ✅")
        print(f"\n   ⚠️  Note: Signals may be infrequent during low volatility")
        print(f"   Expected wait time for first signal: Variable (could be minutes to hours)")
    else:
        print(f"⚠️  No signals in current market conditions")
        print(f"   Highest LONG probability: {probabilities.max():.3f}")
        print(f"   Lowest SHORT probability: {probabilities.min():.3f}")
        print(f"\n   Possible reasons:")
        print(f"   1. Market in sideways/low volatility phase")
        print(f"   2. Model waiting for high-confidence setups")
        print(f"   3. Need to wait for market conditions to improve")
        print(f"\n   Options:")
        print(f"   1. Wait for better market conditions")
        print(f"   2. Run longer (signals may appear soon)")
        print(f"   3. Lower threshold to 0.6 (reduces quality)")

    print(f"\n{'='*80}\n")

if __name__ == "__main__":
    main()
