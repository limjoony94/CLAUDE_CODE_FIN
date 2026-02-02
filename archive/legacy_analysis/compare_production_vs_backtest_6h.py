#!/usr/bin/env python3
"""
Production vs Backtest Signal Comparison - 6 Hours
===================================================
Purpose: Verify backtest signals exactly match production logs
Period: 2025-11-02 18:50 KST ~ 2025-11-03 00:45 KST (72 candles)
Preloading: 100 candles before test period for proper feature calculation
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

# Time Range (KST → UTC)
KST = pytz.timezone('Asia/Seoul')
UTC = pytz.UTC

# Test Period: 2025-11-02 18:50 ~ 2025-11-03 00:45 KST (72 candles = 6 hours)
# Need 100 candles before for feature calculation (preloading)
PRELOAD_START = datetime(2025, 11, 2, 10, 30, tzinfo=KST)  # 100 candles before
TEST_START = datetime(2025, 11, 2, 18, 50, tzinfo=KST)
TEST_END = datetime(2025, 11, 3, 0, 45, tzinfo=KST)

# Production Signals (from logs)
PRODUCTION_SIGNALS = {
    "2025-11-02 18:50:00": {"long": 0.8225, "short": 0.3477, "price": 110260.0},
    "2025-11-02 18:55:00": {"long": 0.8225, "short": 0.3215, "price": 110274.8},
    "2025-11-02 19:00:00": {"long": 0.7859, "short": 0.3564, "price": 110237.6},
    "2025-11-02 19:05:00": {"long": 0.7387, "short": 0.3238, "price": 110167.8},
    "2025-11-02 19:10:00": {"long": 0.7447, "short": 0.3110, "price": 110168.2},
    "2025-11-02 19:15:00": {"long": 0.7137, "short": 0.2723, "price": 110137.1},
    "2025-11-02 19:20:00": {"long": 0.7135, "short": 0.2760, "price": 110129.9},
    "2025-11-02 19:25:00": {"long": 0.8049, "short": 0.1935, "price": 109908.7},
    "2025-11-02 19:30:00": {"long": 0.7335, "short": 0.2098, "price": 110074.7},
    "2025-11-02 19:35:00": {"long": 0.6468, "short": 0.2104, "price": 110016.0},
    "2025-11-02 19:40:00": {"long": 0.6453, "short": 0.2146, "price": 110023.4},
    "2025-11-02 19:45:00": {"long": 0.7106, "short": 0.2152, "price": 109985.0},
    "2025-11-02 19:50:00": {"long": 0.7044, "short": 0.2121, "price": 109949.2},
    "2025-11-02 19:55:00": {"long": 0.7139, "short": 0.1939, "price": 109963.7},
    "2025-11-02 20:00:00": {"long": 0.6993, "short": 0.1857, "price": 109930.3},
    "2025-11-02 20:05:00": {"long": 0.7097, "short": 0.1772, "price": 109902.1},
    "2025-11-02 20:10:00": {"long": 0.6982, "short": 0.1728, "price": 109888.6},
    "2025-11-02 20:15:00": {"long": 0.7125, "short": 0.1685, "price": 109916.2},
    "2025-11-02 20:20:00": {"long": 0.7171, "short": 0.1769, "price": 109912.9},
    "2025-11-02 20:25:00": {"long": 0.7246, "short": 0.1861, "price": 109934.1},
    "2025-11-02 20:30:00": {"long": 0.7270, "short": 0.1781, "price": 109905.5},
    "2025-11-02 20:35:00": {"long": 0.7296, "short": 0.1748, "price": 109882.5},
    "2025-11-02 20:40:00": {"long": 0.7386, "short": 0.1682, "price": 109858.3},
    "2025-11-02 20:45:00": {"long": 0.7466, "short": 0.1572, "price": 109833.2},
    "2025-11-02 20:50:00": {"long": 0.7537, "short": 0.1482, "price": 109810.5},
    "2025-11-02 20:55:00": {"long": 0.7619, "short": 0.1336, "price": 109771.2},
    "2025-11-02 21:00:00": {"long": 0.7736, "short": 0.1205, "price": 109740.7},
    "2025-11-02 21:05:00": {"long": 0.7800, "short": 0.1196, "price": 109723.5},
    "2025-11-02 21:10:00": {"long": 0.7864, "short": 0.1213, "price": 109734.2},
    "2025-11-02 21:15:00": {"long": 0.7963, "short": 0.1247, "price": 109775.5},
    "2025-11-02 21:20:00": {"long": 0.8039, "short": 0.1246, "price": 109783.8},
    "2025-11-02 21:25:00": {"long": 0.8090, "short": 0.1283, "price": 109812.9},
    "2025-11-02 21:30:00": {"long": 0.8160, "short": 0.1203, "price": 109802.2},
    "2025-11-02 21:35:00": {"long": 0.8177, "short": 0.1204, "price": 109813.3},
    "2025-11-02 21:40:00": {"long": 0.8197, "short": 0.1176, "price": 109817.5},
    "2025-11-02 21:45:00": {"long": 0.8276, "short": 0.1156, "price": 109829.8},
    "2025-11-02 21:50:00": {"long": 0.8335, "short": 0.1128, "price": 109834.3},
    "2025-11-02 21:55:00": {"long": 0.8395, "short": 0.1126, "price": 109826.4},
    "2025-11-02 22:00:00": {"long": 0.8396, "short": 0.1118, "price": 109838.8},
    "2025-11-02 22:05:00": {"long": 0.8396, "short": 0.1178, "price": 109857.9},
    "2025-11-02 22:10:00": {"long": 0.8344, "short": 0.1150, "price": 109874.2},
    "2025-11-02 22:15:00": {"long": 0.8288, "short": 0.1155, "price": 109888.4},
    "2025-11-02 22:20:00": {"long": 0.8267, "short": 0.1241, "price": 109904.9},
    "2025-11-02 22:25:00": {"long": 0.8253, "short": 0.1210, "price": 109905.7},
    "2025-11-02 22:30:00": {"long": 0.8243, "short": 0.1160, "price": 109485.8},
    "2025-11-02 22:35:00": {"long": 0.8235, "short": 0.1190, "price": 109494.3},
    "2025-11-02 22:40:00": {"long": 0.8211, "short": 0.2132, "price": 109529.6},
    "2025-11-02 22:45:00": {"long": 0.8196, "short": 0.2348, "price": 109601.0},
    "2025-11-02 22:50:00": {"long": 0.8181, "short": 0.2440, "price": 109688.3},
    "2025-11-02 22:55:00": {"long": 0.8162, "short": 0.2570, "price": 109802.3},
    "2025-11-02 23:00:00": {"long": 0.8134, "short": 0.2693, "price": 109949.2},
    "2025-11-02 23:05:00": {"long": 0.8093, "short": 0.2834, "price": 110146.6},
    "2025-11-02 23:10:00": {"long": 0.8029, "short": 0.3002, "price": 110401.6},
    "2025-11-02 23:15:00": {"long": 0.7943, "short": 0.3165, "price": 110564.9},
    "2025-11-02 23:20:00": {"long": 0.7846, "short": 0.3317, "price": 110390.6},
    "2025-11-02 23:25:00": {"long": 0.7805, "short": 0.3382, "price": 110129.7},
    "2025-11-02 23:30:00": {"long": 0.8251, "short": 0.3455, "price": 110188.1},
    "2025-11-02 23:35:00": {"long": 0.8466, "short": 0.3457, "price": 110387.2},
    "2025-11-02 23:40:00": {"long": 0.8454, "short": 0.3826, "price": 110362.6},
    "2025-11-02 23:45:00": {"long": 0.8539, "short": 0.3635, "price": 110338.1},
    "2025-11-02 23:50:00": {"long": 0.8346, "short": 0.4429, "price": 110588.9},
    "2025-11-02 23:55:00": {"long": 0.7986, "short": 0.3962, "price": 110496.0},
    "2025-11-03 00:00:00": {"long": 0.8048, "short": 0.3667, "price": 110543.1},
    "2025-11-03 00:05:00": {"long": 0.7914, "short": 0.3960, "price": 110699.3},
    "2025-11-03 00:10:00": {"long": 0.8052, "short": 0.3709, "price": 110632.4},
    "2025-11-03 00:15:00": {"long": 0.7829, "short": 0.2993, "price": 110587.0},
    "2025-11-03 00:20:00": {"long": 0.8317, "short": 0.2383, "price": 110451.4},
    "2025-11-03 00:25:00": {"long": 0.8363, "short": 0.2341, "price": 110281.9},
    "2025-11-03 00:30:00": {"long": 0.8458, "short": 0.2223, "price": 110238.3},
    "2025-11-03 00:35:00": {"long": 0.8528, "short": 0.2459, "price": 110269.7},
    "2025-11-03 00:40:00": {"long": 0.8027, "short": 0.2562, "price": 110227.3},
    "2025-11-03 00:45:00": {"long": 0.7133, "short": 0.2324, "price": 110135.5},
}

# Production Models (Enhanced 5-Fold CV - 20251024_012445)
MODELS_DIR = Path(__file__).resolve().parent.parent.parent / "models"
LONG_ENTRY_MODEL = MODELS_DIR / "xgboost_long_entry_enhanced_20251024_012445.pkl"
SHORT_ENTRY_MODEL = MODELS_DIR / "xgboost_short_entry_enhanced_20251024_012445.pkl"
LONG_EXIT_MODEL = MODELS_DIR / "xgboost_long_exit_oppgating_improved_20251024_043527.pkl"
SHORT_EXIT_MODEL = MODELS_DIR / "xgboost_short_exit_oppgating_improved_20251024_044510.pkl"

# Load API keys
CONFIG_DIR = Path(__file__).resolve().parent.parent.parent / "config"

def load_api_keys():
    api_keys_file = CONFIG_DIR / "api_keys.yaml"
    if api_keys_file.exists():
        with open(api_keys_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            return config.get('bingx', {}).get('mainnet', {})
    return {}

_api_config = load_api_keys()
API_KEY = _api_config.get('api_key', "")
API_SECRET = _api_config.get('secret_key', "")

print("=" * 100)
print("PRODUCTION vs BACKTEST SIGNAL COMPARISON - 6 HOURS")
print("=" * 100)
print(f"Test Period: {TEST_START.strftime('%Y-%m-%d %H:%M')} ~ {TEST_END.strftime('%Y-%m-%d %H:%M')} KST")
print(f"Total Candles: 72 (6 hours)")
print(f"Preloading: 100 candles before test period")
print(f"Models: Enhanced 5-Fold CV (20251024_012445)")
print("=" * 100)
print()

# Fetch data with preloading
print("📡 Fetching data from BingX API...")
client = BingXClient(api_key=API_KEY, secret_key=API_SECRET, testnet=False)

# Request maximum available candles (1440 = 5 days) for proper feature calculation
# Feature calculation (especially VP and VWAP) requires significant lookback
total_candles_needed = 1440

print(f"   Requesting {total_candles_needed} candles (maximum available for proper lookback)")
klines = client.get_klines(symbol="BTC-USDT", interval="5m", limit=total_candles_needed)

df = pd.DataFrame(klines)
df['timestamp'] = pd.to_datetime(df['time'], unit='ms', utc=True).dt.tz_localize(None)
for col in ['open', 'high', 'low', 'close', 'volume']:
    df[col] = df[col].astype(float)

print(f"✅ Fetched {len(df)} candles")
print(f"   From: {df.iloc[0]['timestamp']} UTC")
print(f"   To:   {df.iloc[-1]['timestamp']} UTC")
print()

# Calculate features
print("🔧 Calculating features (two-stage pipeline)...")
print("   Stage 1: calculate_all_features_enhanced_v2...")
df_features = calculate_all_features_enhanced_v2(df.copy(), phase='phase1')
print(f"   ✅ Base features: {len(df_features)} rows (lost {len(df) - len(df_features)} due to lookback)")

print("   Stage 2: prepare_exit_features...")
df_features = prepare_exit_features(df_features)
print(f"   ✅ Enhanced exit features added")
print()

# Load models
print("🤖 Loading models...")
long_entry_model = joblib.load(LONG_ENTRY_MODEL)
short_entry_model = joblib.load(SHORT_ENTRY_MODEL)
long_exit_model = joblib.load(LONG_EXIT_MODEL)
short_exit_model = joblib.load(SHORT_EXIT_MODEL)

long_entry_scaler = joblib.load(str(LONG_ENTRY_MODEL).replace('.pkl', '_scaler.pkl'))
short_entry_scaler = joblib.load(str(SHORT_ENTRY_MODEL).replace('.pkl', '_scaler.pkl'))
long_exit_scaler = joblib.load(str(LONG_EXIT_MODEL).replace('.pkl', '_scaler.pkl'))
short_exit_scaler = joblib.load(str(SHORT_EXIT_MODEL).replace('.pkl', '_scaler.pkl'))

# Load feature lists
with open(str(LONG_ENTRY_MODEL).replace('.pkl', '_features.txt'), 'r') as f:
    long_entry_features = [line.strip() for line in f.readlines()]
with open(str(SHORT_ENTRY_MODEL).replace('.pkl', '_features.txt'), 'r') as f:
    short_entry_features = [line.strip() for line in f.readlines()]
with open(str(LONG_EXIT_MODEL).replace('.pkl', '_features.txt'), 'r') as f:
    long_exit_features = [line.strip() for line in f.readlines()]
with open(str(SHORT_EXIT_MODEL).replace('.pkl', '_features.txt'), 'r') as f:
    short_exit_features = [line.strip() for line in f.readlines()]

print(f"✅ Models loaded:")
print(f"   LONG Entry: {len(long_entry_features)} features")
print(f"   SHORT Entry: {len(short_entry_features)} features")
print(f"   LONG Exit: {len(long_exit_features)} features")
print(f"   SHORT Exit: {len(short_exit_features)} features")
print()

# Compare signals
print("🔍 Comparing backtest signals with production logs...")
print("=" * 100)

matches = 0
mismatches = 0
tolerance = 0.0001

comparison_results = []

for timestamp_str, prod_signals in PRODUCTION_SIGNALS.items():
    # Find corresponding row in backtest data
    candle_time = pd.to_datetime(timestamp_str).tz_localize(KST).tz_convert(UTC).tz_localize(None)

    # Find closest matching timestamp in features dataframe
    matching_rows = df_features[df_features['timestamp'] == candle_time]

    if len(matching_rows) == 0:
        print(f"⚠️  {timestamp_str} KST - NOT FOUND in backtest data")
        mismatches += 1
        continue

    row = matching_rows.iloc[0]

    # Generate LONG Entry signal
    long_feat = row[long_entry_features].values.reshape(1, -1)
    long_feat_scaled = long_entry_scaler.transform(long_feat)
    long_prob = long_entry_model.predict_proba(long_feat_scaled)[0][1]

    # Generate SHORT Entry signal
    short_feat = row[short_entry_features].values.reshape(1, -1)
    short_feat_scaled = short_entry_scaler.transform(short_feat)
    short_prob = short_entry_model.predict_proba(short_feat_scaled)[0][1]

    # Compare
    long_diff = abs(long_prob - prod_signals['long'])
    short_diff = abs(short_prob - prod_signals['short'])

    match_status = "✅ MATCH" if (long_diff <= tolerance and short_diff <= tolerance) else "❌ MISMATCH"

    if long_diff <= tolerance and short_diff <= tolerance:
        matches += 1
    else:
        mismatches += 1

    comparison_results.append({
        'timestamp': timestamp_str,
        'prod_long': prod_signals['long'],
        'backtest_long': long_prob,
        'long_diff': long_diff,
        'prod_short': prod_signals['short'],
        'backtest_short': short_prob,
        'short_diff': short_diff,
        'status': match_status
    })

    if long_diff > tolerance or short_diff > tolerance:
        print(f"{match_status} {timestamp_str} KST")
        print(f"   LONG:  Prod {prod_signals['long']:.4f} vs Backtest {long_prob:.4f} (diff: {long_diff:.6f})")
        print(f"   SHORT: Prod {prod_signals['short']:.4f} vs Backtest {short_prob:.4f} (diff: {short_diff:.6f})")

print()
print("=" * 100)
print("SUMMARY")
print("=" * 100)
print(f"Total Candles: {len(PRODUCTION_SIGNALS)}")
print(f"✅ Perfect Matches: {matches} ({matches/len(PRODUCTION_SIGNALS)*100:.1f}%)")
print(f"❌ Mismatches: {mismatches} ({mismatches/len(PRODUCTION_SIGNALS)*100:.1f}%)")
print(f"Tolerance: {tolerance}")
print("=" * 100)

if matches == len(PRODUCTION_SIGNALS):
    print()
    print("🎉 PERFECT! All 72 candles match production logs exactly!")
    print("✅ Backtest methodology is 100% accurate")
elif matches / len(PRODUCTION_SIGNALS) >= 0.95:
    print()
    print("✅ EXCELLENT! >95% match rate - backtest highly reliable")
elif matches / len(PRODUCTION_SIGNALS) >= 0.90:
    print()
    print("⚠️  GOOD: >90% match rate - minor discrepancies to investigate")
else:
    print()
    print("🚨 ATTENTION: <90% match rate - significant discrepancies found")
    print("   Review mismatches above for patterns")

# Save detailed comparison
results_df = pd.DataFrame(comparison_results)
output_file = "results/production_vs_backtest_signals_6h.csv"
results_df.to_csv(output_file, index=False)
print(f"\n📊 Detailed results saved to: {output_file}")
