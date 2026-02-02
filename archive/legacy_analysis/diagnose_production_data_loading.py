"""
Diagnostic: Verify Production Data Loading
===========================================

Check if production bot is loading the LATEST data correctly,
or if timezone issues are causing it to use stale/old data.

User Concern: "프로덕션에서 데이터를 불러올 때 시간대를 제대로 입력해서
최신 데이터를 받아와야 하는데 그러지 않고 있어서 과거 데이터를 사용한다던가 하는 문제는 없나?"
"""

import pandas as pd
import yaml
from pathlib import Path
import sys
from datetime import datetime, timedelta, timezone

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.api.bingx_client import BingXClient

CONFIG_PATH = PROJECT_ROOT / "config" / "api_keys.yaml"

print("="*80)
print("PRODUCTION DATA LOADING DIAGNOSTIC")
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

# Get current time (system time)
now_system = datetime.now()
now_utc = datetime.now(timezone.utc)
now_kst = now_utc + timedelta(hours=9)

print("="*80)
print("CURRENT TIME CHECK")
print("="*80)
print(f"System Time: {now_system.strftime('%Y-%m-%d %H:%M:%S')}")
print(f"UTC Time:    {now_utc.strftime('%Y-%m-%d %H:%M:%S %Z')}")
print(f"KST Time:    {now_kst.strftime('%Y-%m-%d %H:%M:%S')} KST")
print()

# Test 1: Fetch data EXACTLY as production does (limit=1000, no time params)
print("="*80)
print("TEST 1: PRODUCTION METHOD (limit=1000)")
print("="*80)
print()

print("Fetching 1000 candles from BingX API (production method)...")
try:
    candles = client.get_klines("BTC-USDT", "5m", limit=1000)
    print(f"✅ Fetched {len(candles)} candles from API")
    print()

    # Convert to DataFrame
    df_prod = pd.DataFrame(candles)
    df_prod.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    df_prod['timestamp'] = pd.to_datetime(df_prod['timestamp'], unit='ms')

    # Convert numeric columns to float
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df_prod[col] = pd.to_numeric(df_prod[col], errors='coerce')

    # Check timestamps
    first_candle = df_prod.iloc[0]
    last_candle = df_prod.iloc[-1]

    print("First Candle (Oldest):")
    print(f"  Timestamp (UTC): {first_candle['timestamp']}")
    print(f"  Price: ${first_candle['close']:,.2f}")
    print()

    print("Last Candle (Newest):")
    print(f"  Timestamp (UTC): {last_candle['timestamp']}")
    print(f"  Price: ${last_candle['close']:,.2f}")
    print()

    # Calculate time difference from current time
    time_since_last_candle = now_utc.replace(tzinfo=None) - last_candle['timestamp']

    print("Data Freshness Check:")
    print(f"  Time since last candle: {time_since_last_candle}")
    print(f"  Hours ago: {time_since_last_candle.total_seconds() / 3600:.2f}h")

    if time_since_last_candle.total_seconds() < 600:  # Less than 10 minutes
        print(f"  ✅ DATA IS FRESH (within 10 minutes)")
    elif time_since_last_candle.total_seconds() < 3600:  # Less than 1 hour
        print(f"  ⚠️  DATA IS RECENT (within 1 hour)")
    else:
        print(f"  ❌ DATA IS STALE (more than 1 hour old)")
    print()

except Exception as e:
    print(f"❌ Error fetching data: {e}")
    sys.exit(1)

# Test 2: Check timezone conversion (as production does)
print("="*80)
print("TEST 2: TIMEZONE CONVERSION (UTC → KST)")
print("="*80)
print()

# Convert to KST (as production bot does at Line 799)
df_prod['timestamp_kst'] = df_prod['timestamp'] + timedelta(hours=9)

print("Sample of timezone conversion:")
print()
for i in [-5, -4, -3, -2, -1]:
    row = df_prod.iloc[i]
    print(f"  UTC: {row['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  KST: {row['timestamp_kst'].strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Price: ${row['close']:,.2f}")
    print()

# Test 3: Check if we can get the target timestamps from validation period
print("="*80)
print("TEST 3: TARGET TIMESTAMPS CHECK")
print("="*80)
print()

target_times_kst = [
    '2025-10-24 01:00:00',
    '2025-10-24 01:10:00'
]

print("Checking if target timestamps exist in production data:")
print()

for time_kst_str in target_times_kst:
    target_kst = pd.Timestamp(time_kst_str)
    target_utc = target_kst - timedelta(hours=9)

    # Check if exact timestamp exists
    exact_match = df_prod[df_prod['timestamp'] == target_utc]

    if len(exact_match) > 0:
        row = exact_match.iloc[0]
        print(f"✅ FOUND: {time_kst_str} KST ({target_utc} UTC)")
        print(f"   Price: ${row['close']:,.2f}")
    else:
        # Find nearest
        df_prod['time_diff'] = (df_prod['timestamp'] - target_utc).abs()
        nearest = df_prod.nsmallest(1, 'time_diff').iloc[0]
        print(f"❌ NOT FOUND: {time_kst_str} KST ({target_utc} UTC)")
        print(f"   Nearest: {nearest['timestamp']} (diff: {nearest['time_diff']})")
    print()

# Test 4: Compare data range with what backtest needs
print("="*80)
print("TEST 4: DATA RANGE COMPARISON")
print("="*80)
print()

# Backtest validation period
backtest_start_utc = pd.Timestamp('2025-10-24 00:55:00')  # Start of validation period
backtest_end_utc = pd.Timestamp('2025-10-27 01:00:00')    # End of validation period

# Check if production data covers this period
prod_start = df_prod['timestamp'].min()
prod_end = df_prod['timestamp'].max()

print("Backtest Validation Period:")
print(f"  Start: {backtest_start_utc} UTC (2025-10-24 09:55 KST)")
print(f"  End:   {backtest_end_utc} UTC (2025-10-27 10:00 KST)")
print()

print("Production Data Range:")
print(f"  Start: {prod_start} UTC")
print(f"  End:   {prod_end} UTC")
print()

print("Coverage Analysis:")
if prod_start <= backtest_start_utc:
    print(f"  ✅ Covers validation start (starts {backtest_start_utc - prod_start} earlier)")
else:
    print(f"  ❌ Missing validation start (starts {prod_start - backtest_start_utc} later)")

if prod_end >= backtest_end_utc:
    print(f"  ✅ Covers validation end (ends {prod_end - backtest_end_utc} later)")
else:
    print(f"  ❌ Missing validation end (ends {backtest_end_utc - prod_end} earlier)")
print()

# Test 5: Check if current approach could miss recent signals
print("="*80)
print("TEST 5: RECENT SIGNAL DETECTION CAPABILITY")
print("="*80)
print()

# Check last 20 candles (last 100 minutes)
recent_candles = df_prod.tail(20)

print("Recent Candles (Last 20 = 100 minutes):")
print()
print(f"  UTC Range: {recent_candles.iloc[0]['timestamp']} to {recent_candles.iloc[-1]['timestamp']}")
print(f"  KST Range: {recent_candles.iloc[0]['timestamp'] + timedelta(hours=9)} to {recent_candles.iloc[-1]['timestamp'] + timedelta(hours=9)}")
print(f"  Price Range: ${recent_candles['close'].min():,.2f} - ${recent_candles['close'].max():,.2f}")
print()

if time_since_last_candle.total_seconds() < 600:
    print("✅ Production can detect signals within 10 minutes of occurrence")
else:
    print(f"⚠️  Production may miss recent signals (last candle is {time_since_last_candle} old)")
print()

# Summary
print("="*80)
print("DIAGNOSTIC SUMMARY")
print("="*80)
print()

issues_found = []

# Check 1: Data freshness
if time_since_last_candle.total_seconds() > 3600:
    issues_found.append("❌ Data is more than 1 hour old (stale)")
elif time_since_last_candle.total_seconds() > 600:
    issues_found.append("⚠️  Data is more than 10 minutes old (slightly stale)")

# Check 2: Coverage
if not (prod_start <= backtest_start_utc and prod_end >= backtest_end_utc):
    issues_found.append("❌ Production data doesn't cover full validation period")

# Check 3: Target timestamps
target_found = 0
for time_kst_str in target_times_kst:
    target_kst = pd.Timestamp(time_kst_str)
    target_utc = target_kst - timedelta(hours=9)
    if len(df_prod[df_prod['timestamp'] == target_utc]) > 0:
        target_found += 1

if target_found == 0:
    issues_found.append("❌ Target timestamps (2025-10-24 01:00, 01:10 KST) not found")
elif target_found < len(target_times_kst):
    issues_found.append(f"⚠️  Only {target_found}/{len(target_times_kst)} target timestamps found")

if len(issues_found) == 0:
    print("✅ NO ISSUES FOUND")
    print()
    print("Production data loading appears to be working correctly:")
    print("  - Data is fresh and current")
    print("  - Timezone conversion is correct")
    print("  - Coverage is adequate")
    print("  - Can detect recent signals")
else:
    print("⚠️  ISSUES DETECTED:")
    print()
    for issue in issues_found:
        print(f"  {issue}")
    print()
    print("Recommendations:")
    print("  1. Check if API is returning latest data")
    print("  2. Verify timezone handling in production")
    print("  3. Ensure no parameter errors in get_klines()")

print()
print("Diagnostic complete!")
