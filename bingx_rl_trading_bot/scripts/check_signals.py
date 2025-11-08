#!/usr/bin/env python3
"""
Quick script to check if trading signals occur with recent real data
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pickle
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.api.bingx_api import BingXAPI
from src.indicators.advanced_technical_features import AdvancedTechnicalFeatures

# Configuration
MODEL_PATH = PROJECT_ROOT / "models" / "xgboost_v4_phase4_advanced_lookahead3_thresh0.pkl"
SYMBOL = "BTC-USDT"
INTERVAL = "5m"
LOOKBACK_CANDLES = 500
THRESHOLD = 0.7

def get_live_data():
    """Get live data from BingX API"""
    print(f"\n{'='*80}")
    print(f"Fetching live data from BingX Testnet API...")
    print(f"{'='*80}\n")

    api = BingXAPI(use_testnet=True)

    try:
        df = api.get_kline_data(
            symbol=SYMBOL,
            interval=INTERVAL,
            limit=LOOKBACK_CANDLES
        )

        print(f"✅ Fetched {len(df)} candles")
        print(f"   Latest: ${df['close'].iloc[-1]:,.2f} @ {df['timestamp'].iloc[-1]}")
        print(f"   Date range: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")

        return df

    except Exception as e:
        print(f"❌ Error fetching data: {e}")
        return None

def prepare_features(df):
    """Prepare features for XGBoost"""
    print(f"\n{'='*80}")
    print(f"Preparing features...")
    print(f"{'='*80}\n")

    # Initialize advanced technical features
    atf = AdvancedTechnicalFeatures()

    # Add advanced features
    df = atf.add_advanced_features(df)

    print(f"   Data rows before dropna: {len(df)}")

    # Drop NaN values
    df = df.dropna()

    print(f"   Data rows after dropna: {len(df)}")

    if len(df) == 0:
        print(f"❌ All data removed by dropna! Check feature calculations.")
        return None, None

    # Feature columns (37 features)
    feature_cols = [
        # Baseline features (10)
        'rsi', 'macd', 'macd_signal', 'bb_upper', 'bb_lower',
        'ema_21', 'ema_50', 'atr', 'volume_ratio', 'price_distance_ema',

        # Advanced technical features (27)
        'atr_ratio', 'bb_width', 'true_range', 'high_low_range',
        'stoch_rsi', 'williams_r', 'cci', 'cmo', 'ultimate_osc',
        'roc', 'mfi', 'tsi', 'kst',
        'macd_hist', 'adx', 'di_plus', 'di_minus', 'aroon_up', 'aroon_down',
        'vortex_pos', 'vortex_neg',
        'obv', 'cmf',
        'ema_distance', 'bb_position', 'price_momentum'
    ]

    # Check if all features exist
    missing = [col for col in feature_cols if col not in df.columns]
    if missing:
        print(f"⚠️  Missing features: {missing}")
        return None, None

    X = df[feature_cols].values

    print(f"✅ Features prepared: {X.shape}")

    return X, df

def analyze_signals(df, probabilities):
    """Analyze signal distribution"""
    print(f"\n{'='*80}")
    print(f"Signal Analysis")
    print(f"{'='*80}\n")

    # Add probabilities to dataframe
    df = df.copy()
    df['xgb_prob'] = probabilities

    # Statistics
    print(f"Probability Statistics:")
    print(f"  Mean: {probabilities.mean():.4f}")
    print(f"  Std:  {probabilities.std():.4f}")
    print(f"  Min:  {probabilities.min():.4f}")
    print(f"  Max:  {probabilities.max():.4f}")
    print(f"  Median: {np.median(probabilities):.4f}")
    print()

    # Distribution
    print(f"Probability Distribution:")
    print(f"  >= 0.9: {(probabilities >= 0.9).sum()} ({(probabilities >= 0.9).sum() / len(probabilities) * 100:.1f}%)")
    print(f"  >= 0.8: {(probabilities >= 0.8).sum()} ({(probabilities >= 0.8).sum() / len(probabilities) * 100:.1f}%)")
    print(f"  >= 0.7: {(probabilities >= 0.7).sum()} ({(probabilities >= 0.7).sum() / len(probabilities) * 100:.1f}%)")
    print(f"  >= 0.6: {(probabilities >= 0.6).sum()} ({(probabilities >= 0.6).sum() / len(probabilities) * 100:.1f}%)")
    print(f"  >= 0.5: {(probabilities >= 0.5).sum()} ({(probabilities >= 0.5).sum() / len(probabilities) * 100:.1f}%)")
    print(f"  < 0.5:  {(probabilities < 0.5).sum()} ({(probabilities < 0.5).sum() / len(probabilities) * 100:.1f}%)")
    print()

    # High confidence signals
    high_signals = df[df['xgb_prob'] >= THRESHOLD].copy()

    if len(high_signals) > 0:
        print(f"✅ Found {len(high_signals)} signals >= {THRESHOLD}:")
        print()

        for idx, row in high_signals.tail(10).iterrows():
            print(f"   {row['timestamp']}: Prob={row['xgb_prob']:.3f}, Price=${row['close']:,.2f}")

        if len(high_signals) > 10:
            print(f"   ... and {len(high_signals) - 10} more")

        # Recent signals (last 100 candles)
        recent_signals = high_signals.tail(100)
        print(f"\n   Recent signals (last 100 candles): {len(recent_signals)}")

        if len(recent_signals) > 0:
            latest = recent_signals.iloc[-1]
            print(f"   Latest signal: {latest['timestamp']}")
            print(f"   Probability: {latest['xgb_prob']:.3f}")
            print(f"   Price: ${latest['close']:,.2f}")

            # Time since latest signal
            latest_time = pd.to_datetime(latest['timestamp'])
            current_time = pd.to_datetime(df['timestamp'].iloc[-1])
            time_diff = (current_time - latest_time).total_seconds() / 60
            print(f"   Time since latest: {time_diff:.0f} minutes ago")
    else:
        print(f"❌ No signals found >= {THRESHOLD}")
        print(f"\n   Highest probability in dataset: {probabilities.max():.3f}")

        # Show top 10 probabilities
        top_10_idx = np.argsort(probabilities)[-10:][::-1]
        print(f"\n   Top 10 probabilities:")
        for idx in top_10_idx:
            prob = probabilities[idx]
            timestamp = df.iloc[idx]['timestamp']
            price = df.iloc[idx]['close']
            print(f"     {timestamp}: {prob:.3f} (${price:,.2f})")

    return high_signals

def main():
    """Main function"""
    print(f"\n{'='*80}")
    print(f"Phase 4 Dynamic Trading Signal Checker")
    print(f"{'='*80}\n")
    print(f"Model: {MODEL_PATH.name}")
    print(f"Symbol: {SYMBOL}")
    print(f"Interval: {INTERVAL}")
    print(f"Threshold: {THRESHOLD}")

    # Load model
    print(f"\nLoading XGBoost model...")
    try:
        with open(MODEL_PATH, 'rb') as f:
            model = pickle.load(f)
        print(f"✅ Model loaded successfully")
        print(f"   Features expected: {model.n_features_in_}")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return

    # Get live data
    df = get_live_data()
    if df is None:
        return

    # Prepare features
    X, df = prepare_features(df)
    if X is None or df is None:
        return

    # Predict probabilities
    print(f"\n{'='*80}")
    print(f"Predicting probabilities...")
    print(f"{'='*80}\n")

    try:
        probabilities = model.predict_proba(X)[:, 1]
        print(f"✅ Predictions complete: {len(probabilities)} probabilities")
    except Exception as e:
        print(f"❌ Prediction failed: {e}")
        return

    # Analyze signals
    high_signals = analyze_signals(df, probabilities)

    # Summary
    print(f"\n{'='*80}")
    print(f"Summary")
    print(f"{'='*80}\n")

    if len(high_signals) > 0:
        print(f"✅ Trading signals ARE occurring in real data")
        print(f"   Total signals >= {THRESHOLD}: {len(high_signals)}")
        print(f"   Signal rate: {len(high_signals) / len(df) * 100:.2f}%")
        print(f"\n   Recommendation: Paper trading should work ✅")
        print(f"   Expected trades per 48h window: ~{len(high_signals) / len(df) * 576:.1f}")
    else:
        print(f"⚠️  No signals >= {THRESHOLD} in current market conditions")
        print(f"   Highest probability: {probabilities.max():.3f}")
        print(f"\n   Possible reasons:")
        print(f"   1. Current market regime not favorable (sideways/low volatility)")
        print(f"   2. Model waiting for high-confidence setups")
        print(f"   3. May need to wait for market conditions to change")
        print(f"\n   Options:")
        print(f"   1. Wait for market conditions to improve")
        print(f"   2. Lower threshold to 0.6 (reduces quality, increases frequency)")
        print(f"   3. Continue monitoring - signals may appear soon")

    print(f"\n{'='*80}\n")

if __name__ == "__main__":
    main()
