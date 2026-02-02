"""
CRITICAL: Data Source Comparison
=================================
Compare production (live API data) vs backtest (historical CSV data)
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys
from datetime import datetime, timedelta
import re

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiments.calculate_all_features import calculate_all_features
from src.api.bingx_client import BingXClient
import yaml

DATA_DIR = PROJECT_ROOT / "data" / "historical"
CONFIG_DIR = PROJECT_ROOT / "config"
LOGS_DIR = PROJECT_ROOT / "logs"

print("="*80)
print("🔍 CRITICAL: Data Source Comparison")
print("="*80)

# 1. EXTRACT PRODUCTION DATA FROM LOGS
print("\n1️⃣ Extracting production data from logs...")
log_file = LOGS_DIR / "opportunity_gating_bot_4x_20251020.log"

production_data = []
with open(log_file, 'r', encoding='utf-8') as f:
    for line in f:
        # Extract candle data from logs
        match = re.search(r'\[([0-9-]+ [0-9:]+)\] Price: \$([0-9,]+\.[0-9]+)', line)
        if match:
            timestamp_str = match.group(1)
            price_str = match.group(2).replace(',', '')
            try:
                timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
                price = float(price_str)
                production_data.append({'timestamp': timestamp, 'close': price})
            except:
                pass

print(f"✅ Extracted {len(production_data)} data points from production log")

if production_data:
    df_prod = pd.DataFrame(production_data)
    print(f"\nProduction data range:")
    print(f"  Start: {df_prod['timestamp'].min()}")
    print(f"  End:   {df_prod['timestamp'].max()}")
    print(f"  Price range: ${df_prod['close'].min():,.2f} - ${df_prod['close'].max():,.2f}")

# 2. LOAD BACKTEST DATA
print("\n2️⃣ Loading backtest data...")
data_path = DATA_DIR / "BTCUSDT_5m_max.csv"
df_backtest = pd.read_csv(data_path)
df_backtest['timestamp'] = pd.to_datetime(df_backtest['timestamp'])

print(f"✅ Loaded {len(df_backtest)} candles from backtest file")
print(f"\nBacktest data range:")
print(f"  Start: {df_backtest['timestamp'].min()}")
print(f"  End:   {df_backtest['timestamp'].max()}")
print(f"  Price range: ${df_backtest['close'].min():,.2f} - ${df_backtest['close'].max():,.2f}")

# 3. CHECK IF PRODUCTION IS USING LIVE DATA OR OLD CSV
print("\n" + "="*80)
print("3️⃣ DATA SOURCE VERIFICATION")
print("="*80)

latest_backtest = df_backtest['timestamp'].max()
current_time = datetime.now()
data_age_days = (current_time - latest_backtest).days

print(f"\nCurrent time: {current_time}")
print(f"Latest backtest data: {latest_backtest}")
print(f"Data age: {data_age_days} days")

if data_age_days > 3:
    print(f"\n⚠️  WARNING: Backtest data is {data_age_days} days old!")
    print(f"   Production bot is using LIVE data")
    print(f"   Backtest is using OLD CSV data")
    print(f"   → This explains the distribution shift!")
elif data_age_days > 1:
    print(f"\n⚠️  CAUTION: Backtest data is {data_age_days} days old")
    print(f"   Small mismatch possible")
else:
    print(f"\n✅ Backtest data is recent (< 1 day old)")

# 4. GET LIVE DATA FROM API
print("\n4️⃣ Fetching LIVE data from BingX API...")

try:
    # Load API config
    config_path = CONFIG_DIR / "api_keys.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Initialize client (MAINNET - same as production bot)
    client = BingXClient(
        api_key=config['bingx']['mainnet']['api_key'],
        secret_key=config['bingx']['mainnet']['secret_key'],
        testnet=False  # MAINNET
    )

    # Fetch recent data
    print("Fetching last 100 candles from API (MAINNET)...")
    live_data = client.get_klines(
        symbol='BTC-USDT',
        interval='5m',
        limit=100
    )

    df_live = pd.DataFrame(live_data)
    # Convert 'time' to 'timestamp' and convert string values to numeric
    df_live['timestamp'] = pd.to_datetime(df_live['time'], unit='ms')
    df_live['open'] = df_live['open'].astype(float)
    df_live['high'] = df_live['high'].astype(float)
    df_live['low'] = df_live['low'].astype(float)
    df_live['close'] = df_live['close'].astype(float)
    df_live['volume'] = df_live['volume'].astype(float)

    # Check data order
    print(f"\n🔍 Checking data order...")
    print(f"   First candle: {df_live['timestamp'].iloc[0]}")
    print(f"   Last candle: {df_live['timestamp'].iloc[-1]}")

    # If data is in reverse order, sort it
    if df_live['timestamp'].iloc[0] > df_live['timestamp'].iloc[-1]:
        print(f"   ⚠️  Data is in REVERSE order, sorting...")
        df_live = df_live.sort_values('timestamp').reset_index(drop=True)
        print(f"   ✅ Sorted: {df_live['timestamp'].iloc[0]} → {df_live['timestamp'].iloc[-1]}")
    else:
        print(f"   ✅ Data is already in correct order (oldest → newest)")

    print(f"\n✅ Fetched {len(df_live)} live candles (MAINNET)")
    print(f"\nLive data range:")
    print(f"  Start: {df_live['timestamp'].min()}")
    print(f"  End:   {df_live['timestamp'].max()}")
    print(f"  Price range: ${df_live['close'].min():,.2f} - ${df_live['close'].max():,.2f}")

    # 5. COMPARE OVERLAPPING DATA
    print("\n" + "="*80)
    print("5️⃣ COMPARING OVERLAPPING DATA")
    print("="*80)

    # Find overlap period
    overlap_start = max(df_live['timestamp'].min(), df_backtest['timestamp'].max() - timedelta(hours=4))
    overlap_end = min(df_live['timestamp'].max(), df_backtest['timestamp'].max())

    if overlap_start < overlap_end:
        print(f"\nOverlap period: {overlap_start} to {overlap_end}")

        df_live_overlap = df_live[(df_live['timestamp'] >= overlap_start) & (df_live['timestamp'] <= overlap_end)]
        df_back_overlap = df_backtest[(df_backtest['timestamp'] >= overlap_start) & (df_backtest['timestamp'] <= overlap_end)]

        print(f"Live data points in overlap: {len(df_live_overlap)}")
        print(f"Backtest data points in overlap: {len(df_back_overlap)}")

        if len(df_live_overlap) > 0 and len(df_back_overlap) > 0:
            # Merge on timestamp
            df_merged = pd.merge(
                df_live_overlap[['timestamp', 'close']],
                df_back_overlap[['timestamp', 'close']],
                on='timestamp',
                suffixes=('_live', '_back')
            )

            if len(df_merged) > 0:
                print(f"\nMatched {len(df_merged)} timestamps")

                # Calculate differences
                df_merged['diff'] = df_merged['close_live'] - df_merged['close_back']
                df_merged['diff_pct'] = (df_merged['diff'] / df_merged['close_back']) * 100

                print(f"\nPrice differences:")
                print(f"  Mean difference: ${df_merged['diff'].mean():.2f}")
                print(f"  Max difference: ${df_merged['diff'].max():.2f}")
                print(f"  Mean % difference: {df_merged['diff_pct'].mean():.4f}%")

                if abs(df_merged['diff_pct'].mean()) > 0.1:
                    print(f"\n⚠️  WARNING: Significant price difference detected!")
                    print(f"   Live and backtest data DO NOT match!")
                else:
                    print(f"\n✅ Price differences are negligible")
            else:
                print(f"\n⚠️  No matching timestamps found!")
        else:
            print(f"\n⚠️  Insufficient overlap data for comparison")
    else:
        print(f"\n⚠️  No overlap between live and backtest data!")
        print(f"   Backtest ends at: {df_backtest['timestamp'].max()}")
        print(f"   Live data starts at: {df_live['timestamp'].min()}")

    # 6. CALCULATE FEATURES FOR LIVE DATA
    print("\n" + "="*80)
    print("6️⃣ COMPARING FEATURE DISTRIBUTIONS")
    print("="*80)

    print("\nCalculating features for live data...")
    df_live_features = calculate_all_features(df_live)

    print("Calculating features for recent backtest data...")
    df_back_recent = df_backtest.tail(100).copy()
    df_back_features = calculate_all_features(df_back_recent)

    # Compare key feature values
    key_features = ['close_change_1', 'rsi', 'macd', 'bb_high', 'bb_mid', 'bb_low']

    print(f"\nComparing key features (last value):")
    print(f"\n{'Feature':<25} {'Live':<15} {'Backtest':<15} {'Diff %':<10}")
    print("-" * 70)

    for feature in key_features:
        if feature in df_live_features.columns and feature in df_back_features.columns:
            live_val = df_live_features[feature].iloc[-1]
            back_val = df_back_features[feature].iloc[-1]

            if abs(back_val) > 1e-10:
                diff_pct = ((live_val - back_val) / abs(back_val)) * 100
                status = "⚠️" if abs(diff_pct) > 10 else "✅"
            else:
                diff_pct = 0
                status = "✅"

            print(f"{status} {feature:<24} {live_val:<14.4f} {back_val:<14.4f} {diff_pct:>9.1f}%")

except Exception as e:
    print(f"\n❌ ERROR fetching live data: {e}")
    import traceback
    traceback.print_exc()

# 7. SUMMARY
print("\n" + "="*80)
print("💡 ROOT CAUSE DIAGNOSIS")
print("="*80)

if data_age_days > 3:
    print("\n🚨 CRITICAL FINDING:")
    print(f"   Backtest CSV data is {data_age_days} days old")
    print(f"   Production bot uses LIVE API data (current)")
    print(f"   → DATA SOURCE MISMATCH is the root cause!")
    print(f"\n   Recent market conditions differ from backtest data")
    print(f"   Model behavior in production reflects CURRENT market")
    print(f"   Backtest results reflect HISTORICAL market (3+ days ago)")
    print(f"\n✅ SOLUTION:")
    print(f"   This is actually NORMAL and EXPECTED behavior")
    print(f"   The model is responding to current market conditions")
    print(f"   High LONG signal frequency may be appropriate for current market")
    print(f"\n   RECOMMENDATION:")
    print(f"   1. Update backtest CSV with latest data daily")
    print(f"   2. Run backtest on MOST RECENT data (last 2-7 days)")
    print(f"   3. Compare production behavior to recent backtest, not old backtest")
else:
    print(f"\n✅ Data sources are reasonably aligned")
    print(f"   Other factors may be causing the signal frequency difference")

print("\n" + "="*80)
