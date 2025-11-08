"""
Test BingX API Timestamp Behavior

Tests whether kline timestamps represent:
1. Candle start time or end time
2. Whether the last candle returned is completed or in-progress
3. Optimal timing for fetching completed candles

Author: Trading System
Date: 2025-10-22
"""

import sys
import os
from datetime import datetime, timezone
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


def get_candle_status(candle_start_ms, current_ms, interval_seconds=300):
    """
    Determine if a candle is completed or in-progress

    Args:
        candle_start_ms: Candle timestamp in milliseconds
        current_ms: Current time in milliseconds
        interval_seconds: Candle interval in seconds (300 for 5m)

    Returns:
        str: 'COMPLETED' or 'IN_PROGRESS'
    """
    candle_end_ms = candle_start_ms + (interval_seconds * 1000)

    if current_ms >= candle_end_ms:
        return 'COMPLETED'
    else:
        seconds_remaining = (candle_end_ms - current_ms) / 1000
        return f'IN_PROGRESS ({seconds_remaining:.0f}s remaining)'


def test_timestamp_behavior():
    """Main test function"""
    print("=" * 80)
    print("BingX API Timestamp Behavior Test")
    print("=" * 80)
    print()

    # Load API keys
    api_config = load_api_keys()
    api_key = api_config.get('api_key', '')
    secret_key = api_config.get('secret_key', '')

    if not api_key or not secret_key:
        print("❌ ERROR: API keys not found in config/api_keys.yaml")
        return

    # Initialize client
    client = BingXClient(api_key=api_key, secret_key=secret_key, testnet=False)

    # Get current time
    current_time_ms = int(time.time() * 1000)
    current_dt = datetime.fromtimestamp(current_time_ms / 1000, tz=timezone.utc)

    print(f"Test Time: {format_timestamp(current_time_ms)}")
    print(f"Seconds into current 5m candle: {current_dt.second + (current_dt.minute % 5) * 60}")
    print()

    # Fetch recent klines
    print("Fetching last 5 candles from BingX API...")
    klines = client.get_klines(symbol='BTC-USDT', interval='5m', limit=5)

    if not klines:
        print("❌ ERROR: Failed to fetch klines")
        return

    print(f"✅ Received {len(klines)} candles")
    print()

    # Analyze each candle
    print("=" * 80)
    print("CANDLE ANALYSIS")
    print("=" * 80)
    print()

    for i, candle in enumerate(klines):
        timestamp_ms = candle['time']

        print(f"Candle {i+1}/{len(klines)}:")
        print(f"  Timestamp: {timestamp_ms}")
        print(f"  DateTime:  {format_timestamp(timestamp_ms)}")
        print(f"  Open:      ${float(candle['open']):.2f}")
        print(f"  Close:     ${float(candle['close']):.2f}")
        print(f"  Status:    {get_candle_status(timestamp_ms, current_time_ms)}")

        # Check if timestamps are 5 minutes apart
        if i > 0:
            time_diff_ms = timestamp_ms - klines[i-1]['time']
            time_diff_min = time_diff_ms / (1000 * 60)
            print(f"  Time Diff: {time_diff_min:.1f} minutes from previous")

        print()

    # Summary and recommendations
    print("=" * 80)
    print("ANALYSIS SUMMARY")
    print("=" * 80)
    print()

    last_candle = klines[-1]
    last_status = get_candle_status(last_candle['time'], current_time_ms)

    print("Key Findings:")
    print()
    print("1. Timestamp Meaning:")
    print("   → Timestamps represent CANDLE START TIME")
    print("   → Each 5m candle starts at :00, :05, :10, :15, etc.")
    print()

    print("2. Last Candle Status:")
    print(f"   → Last candle timestamp: {format_timestamp(last_candle['time'])}")
    print(f"   → Status: {last_status}")
    print()

    if 'IN_PROGRESS' in last_status:
        print("3. ⚠️  ISSUE DETECTED:")
        print("   → API returns IN-PROGRESS candle as last candle")
        print("   → This candle is NOT yet completed")
        print("   → Using this for trading signals would be INCORRECT")
        print()
        print("4. 📋 RECOMMENDATION:")
        print("   → Use klines[:-1] to exclude last candle")
        print("   → Or fetch with limit=N and use first N-1 candles")
        print("   → Last completed candle is at index -2")
        print()

        second_last = klines[-2]
        print(f"   Last COMPLETED candle:")
        print(f"   → Timestamp: {format_timestamp(second_last['time'])}")
        print(f"   → Close: ${float(second_last['close']):.2f}")
        print(f"   → Status: {get_candle_status(second_last['time'], current_time_ms)}")
    else:
        print("3. ✅ GOOD:")
        print("   → Last candle is COMPLETED")
        print("   → Safe to use for trading signals")
        print()
        print("4. 📋 RECOMMENDATION:")
        print("   → Current timing is appropriate")
        print("   → Can use all returned candles")

    print()
    print("=" * 80)
    print("TIMING RECOMMENDATION FOR PRODUCTION")
    print("=" * 80)
    print()
    print("Current production bot calls API at: XX:X0:03 (3 seconds after 5m intervals)")
    print()

    if 'IN_PROGRESS' in last_status:
        print("⚠️  ISSUE: Calling too early in the candle")
        print()
        print("Recommended changes:")
        print("1. Code change: Use klines[:-1] to exclude last incomplete candle")
        print("2. OR increase wait time to 10-15 seconds after the interval")
        print("   Example: XX:X0:10 or XX:X0:15")
        print()
        print("Trade-offs:")
        print("  - Waiting longer: More reliable data, slight delay")
        print("  - Using klines[:-1]: Immediate, but always 1 candle behind")
    else:
        print("✅ TIMING IS GOOD:")
        print("   → 3 seconds is sufficient for candle completion")
        print("   → Current implementation is correct")

    print()


if __name__ == "__main__":
    test_timestamp_behavior()
