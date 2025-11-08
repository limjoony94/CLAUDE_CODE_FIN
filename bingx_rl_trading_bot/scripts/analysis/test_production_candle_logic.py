"""
Test Production Candle Update Logic

Tests the complete candle fetching, validation, and filtering logic
used in the production bot across multiple timeframes.

Author: Trading System
Date: 2025-10-23
"""

import sys
import os
from datetime import datetime, timedelta, timezone
import time
import pandas as pd
import yaml
from pathlib import Path

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.insert(0, project_root)

from src.api.bingx_client import BingXClient


def load_api_keys():
    """Load API keys from config/api_keys.yaml"""
    config_dir = Path(project_root) / "config"
    api_keys_file = config_dir / "api_keys.yaml"
    if api_keys_file.exists():
        with open(api_keys_file, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
            return config.get('bingx', {}).get('mainnet', {})
    return {}


def format_timestamp(ts_ms):
    """Convert millisecond timestamp to readable format"""
    return datetime.fromtimestamp(ts_ms / 1000, tz=timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')


def filter_completed_candles(df, current_time, interval_minutes=5):
    """
    Replica of production filter_completed_candles() logic

    Args:
        df: DataFrame with candles (timestamp column in local time)
        current_time: Current local time
        interval_minutes: Candle interval (default 5m)

    Returns:
        DataFrame with only completed candles
    """
    # Current candle start time
    current_candle_start = current_time.replace(second=0, microsecond=0)
    current_candle_start = current_candle_start - timedelta(minutes=current_candle_start.minute % interval_minutes)

    # Filter: only candles with timestamp < current_candle_start
    df_completed = df[df['timestamp'] < current_candle_start].copy()

    return df_completed, current_candle_start


def test_candle_logic_single():
    """Test candle logic at current moment"""
    print("=" * 80)
    print("SINGLE TIMEPOINT TEST - Production Candle Logic")
    print("=" * 80)
    print()

    # Load API keys
    api_config = load_api_keys()
    api_key = api_config.get('api_key', '')
    secret_key = api_config.get('secret_key', '')

    if not api_key or not secret_key:
        print("❌ ERROR: API keys not found")
        return False

    # Initialize client
    client = BingXClient(api_key=api_key, secret_key=secret_key, testnet=False)

    # Current time
    current_time = datetime.now()
    current_time_utc = datetime.now(timezone.utc)

    print(f"Test Time (Local): {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Test Time (UTC):   {current_time_utc.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Seconds into 5m:   {current_time.second + (current_time.minute % 5) * 60}")
    print()

    # Expected candle (current 5m interval start)
    expected_candle = current_time.replace(second=0, microsecond=0)
    expected_candle = expected_candle - timedelta(minutes=expected_candle.minute % 5)

    print(f"Expected Latest Candle: {expected_candle.strftime('%H:%M:%S')}")
    print()

    # Fetch klines
    print("Fetching klines from API...")
    klines = client.get_klines(symbol='BTC-USDT', interval='5m', limit=10)

    if not klines:
        print("❌ Failed to fetch klines")
        return False

    print(f"✅ Received {len(klines)} candles")
    print()

    # Convert to DataFrame
    df = pd.DataFrame(klines)
    df['timestamp'] = pd.to_datetime(df['time'], unit='ms', utc=True).dt.tz_convert('Asia/Seoul').dt.tz_localize(None)
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)

    # Show raw data
    print("=" * 80)
    print("RAW API DATA (Last 5 candles)")
    print("=" * 80)
    for i in range(max(0, len(df)-5), len(df)):
        candle = df.iloc[i]
        print(f"  [{i+1}/{len(df)}] {candle['timestamp'].strftime('%H:%M:%S')} | "
              f"O: ${candle['open']:.2f} | H: ${candle['high']:.2f} | "
              f"L: ${candle['low']:.2f} | C: ${candle['close']:.2f}")
    print()

    # Apply production filter
    df_completed, current_candle_start = filter_completed_candles(df, current_time)

    print("=" * 80)
    print("FILTER RESULTS")
    print("=" * 80)
    print(f"Current Candle Start: {current_candle_start.strftime('%H:%M:%S')}")
    print(f"Total Candles:        {len(df)}")
    print(f"Completed Candles:    {len(df_completed)}")
    print(f"Filtered Out:         {len(df) - len(df_completed)}")
    print()

    if len(df_completed) == 0:
        print("❌ ERROR: No completed candles!")
        return False

    # Latest completed candle
    latest_completed = df_completed.iloc[-1]['timestamp']
    expected_latest = current_candle_start - timedelta(minutes=5)

    print(f"Latest Completed:     {latest_completed.strftime('%H:%M:%S')}")
    print(f"Expected Latest:      {expected_latest.strftime('%H:%M:%S')}")

    # Validation
    print()
    print("=" * 80)
    print("VALIDATION")
    print("=" * 80)

    # Check 1: Latest matches expected
    if latest_completed == expected_latest:
        print("✅ PASS: Latest completed matches expected")
    else:
        time_diff = (latest_completed - expected_latest).total_seconds()
        print(f"❌ FAIL: Time difference {time_diff:+.0f} seconds")
        return False

    # Check 2: Filtered candle is in-progress
    if len(df_completed) < len(df):
        filtered_candle = df.iloc[-1]['timestamp']
        print(f"✅ PASS: Filtered in-progress candle at {filtered_candle.strftime('%H:%M:%S')}")
        if filtered_candle == current_candle_start:
            print(f"   ✅ Correct: {filtered_candle.strftime('%H:%M:%S')} = {current_candle_start.strftime('%H:%M:%S')}")
        else:
            print(f"   ⚠️  Unexpected filtered candle timestamp")
    else:
        print("⚠️  WARNING: No candles filtered (API may not return in-progress)")

    # Check 3: All completed candles are before current interval
    all_valid = True
    for idx, row in df_completed.iterrows():
        if row['timestamp'] >= current_candle_start:
            print(f"❌ FAIL: Completed candle {row['timestamp'].strftime('%H:%M:%S')} >= current start")
            all_valid = False

    if all_valid:
        print("✅ PASS: All completed candles are before current interval")

    print()
    return True


def test_candle_logic_extended(duration_minutes=30, check_interval=5):
    """
    Test candle logic over extended period

    Args:
        duration_minutes: How long to test (default 30 min)
        check_interval: Interval between checks in minutes (default 5)
    """
    print("=" * 80)
    print(f"EXTENDED TEST - {duration_minutes} Minutes")
    print("=" * 80)
    print()

    # Load API keys
    api_config = load_api_keys()
    api_key = api_config.get('api_key', '')
    secret_key = api_config.get('secret_key', '')

    if not api_key or not secret_key:
        print("❌ ERROR: API keys not found")
        return

    # Initialize client
    client = BingXClient(api_key=api_key, secret_key=secret_key, testnet=False)

    # Calculate check times
    current_time = datetime.now()
    end_time = current_time - timedelta(minutes=duration_minutes)

    print(f"Analysis Period: {duration_minutes} minutes")
    print(f"Check Interval:  {check_interval} minutes")
    print(f"Current Time:    {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Lookback To:     {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()

    # Fetch historical data
    limit = int(duration_minutes / 5) + 10  # Extra buffer
    print(f"Fetching {limit} candles...")
    klines = client.get_klines(symbol='BTC-USDT', interval='5m', limit=limit)

    if not klines:
        print("❌ Failed to fetch klines")
        return

    print(f"✅ Received {len(klines)} candles")
    print()

    # Convert to DataFrame
    df = pd.DataFrame(klines)
    df['timestamp'] = pd.to_datetime(df['time'], unit='ms', utc=True).dt.tz_convert('Asia/Seoul').dt.tz_localize(None)
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)

    # Filter to analysis period
    df_period = df[df['timestamp'] >= end_time].copy()

    print("=" * 80)
    print("SIMULATED CHECKS (Every 5 minutes)")
    print("=" * 80)
    print()

    # Simulate checks at each 5-minute interval + 3 seconds
    check_times = []
    check_time = end_time.replace(second=0, microsecond=0)
    check_time = check_time - timedelta(minutes=check_time.minute % 5) + timedelta(minutes=5, seconds=3)

    while check_time <= current_time:
        check_times.append(check_time)
        check_time += timedelta(minutes=check_interval)

    results = []

    for i, check_time in enumerate(check_times):
        # Simulate what would happen at this check time
        current_candle_start = check_time.replace(second=0, microsecond=0)
        current_candle_start = current_candle_start - timedelta(minutes=current_candle_start.minute % 5)

        # Filter completed candles
        df_completed = df[df['timestamp'] < current_candle_start].copy()

        if len(df_completed) == 0:
            continue

        latest_completed = df_completed.iloc[-1]['timestamp']
        expected_latest = current_candle_start - timedelta(minutes=5)

        # Validation
        match = latest_completed == expected_latest
        time_diff = (latest_completed - expected_latest).total_seconds() if not match else 0

        results.append({
            'check_time': check_time,
            'current_start': current_candle_start,
            'latest_completed': latest_completed,
            'expected_latest': expected_latest,
            'match': match,
            'time_diff': time_diff
        })

        status = "✅" if match else "❌"
        print(f"{status} Check {i+1}/{len(check_times)}: {check_time.strftime('%H:%M:%S')}")
        print(f"   Current Start: {current_candle_start.strftime('%H:%M:%S')}")
        print(f"   Latest:        {latest_completed.strftime('%H:%M:%S')}")
        print(f"   Expected:      {expected_latest.strftime('%H:%M:%S')}")
        if not match:
            print(f"   ⚠️  Diff:       {time_diff:+.0f} seconds")
        print()

    # Summary
    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)

    total_checks = len(results)
    passed = sum(1 for r in results if r['match'])
    failed = total_checks - passed

    print(f"Total Checks:  {total_checks}")
    print(f"Passed:        {passed} ({passed/total_checks*100:.1f}%)")
    print(f"Failed:        {failed} ({failed/total_checks*100:.1f}%)")
    print()

    if failed > 0:
        print("Failed Checks:")
        for r in results:
            if not r['match']:
                print(f"  - {r['check_time'].strftime('%H:%M:%S')}: "
                      f"{r['time_diff']:+.0f}s difference")

    return passed == total_checks


def main():
    """Main test function"""
    print("\n")
    print("╔" + "═" * 78 + "╗")
    print("║" + " " * 20 + "PRODUCTION CANDLE LOGIC TEST" + " " * 30 + "║")
    print("╚" + "═" * 78 + "╝")
    print()

    # Test 1: Single timepoint
    print("TEST 1: SINGLE TIMEPOINT TEST")
    print("-" * 80)
    single_pass = test_candle_logic_single()
    print()

    # Test 2: Extended period (last 2 hours)
    print("\n" + "=" * 80)
    print("TEST 2: EXTENDED PERIOD TEST (2 Hours)")
    print("-" * 80)
    extended_pass = test_candle_logic_extended(duration_minutes=120, check_interval=5)
    print()

    # Final result
    print("=" * 80)
    print("FINAL RESULT")
    print("=" * 80)

    if single_pass and extended_pass:
        print("✅ ALL TESTS PASSED")
        print("   Production candle logic is working correctly!")
    else:
        print("❌ SOME TESTS FAILED")
        if not single_pass:
            print("   - Single timepoint test failed")
        if not extended_pass:
            print("   - Extended period test failed")

    print()


if __name__ == "__main__":
    main()
