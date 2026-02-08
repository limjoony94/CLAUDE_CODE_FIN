#!/usr/bin/env python3
"""
Fetch 270-Day 5-Minute Historical Data from Binance Public API
==============================================================

No API key required — uses Binance public klines endpoint via ccxt.
Downloads BTCUSDT perpetual futures 5m candles for 270 days.

Output: data/btc_5m_270days.csv

Usage:
    python scripts/data/fetch_historical.py [--days 270] [--symbol BTC/USDT] [--output data/btc_5m_270days.csv]
"""

import argparse
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import ccxt
import pandas as pd

# Defaults
DEFAULT_SYMBOL = "BTC/USDT"
DEFAULT_TIMEFRAME = "5m"
DEFAULT_DAYS = 270
DEFAULT_OUTPUT = "data/btc_5m_270days.csv"
CANDLES_PER_REQUEST = 1000  # Binance max
MS_PER_5MIN = 5 * 60 * 1000
RATE_LIMIT_SLEEP = 0.5  # seconds between requests


def parse_args():
    parser = argparse.ArgumentParser(description="Fetch historical 5m OHLCV data from Binance")
    parser.add_argument("--days", type=int, default=DEFAULT_DAYS, help=f"Days to fetch (default: {DEFAULT_DAYS})")
    parser.add_argument("--symbol", default=DEFAULT_SYMBOL, help=f"Trading pair (default: {DEFAULT_SYMBOL})")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help=f"Output CSV path (default: {DEFAULT_OUTPUT})")
    parser.add_argument("--exchange", default="binance", help="Exchange to use (default: binance)")
    return parser.parse_args()


def create_exchange(exchange_id: str) -> ccxt.Exchange:
    """Create exchange instance (no API keys needed for public data)."""
    exchange_class = getattr(ccxt, exchange_id, None)
    if exchange_class is None:
        print(f"❌ Unknown exchange: {exchange_id}")
        sys.exit(1)

    exchange = exchange_class({
        'enableRateLimit': True,
        'options': {'defaultType': 'swap'},  # perpetual futures
    })
    return exchange


def fetch_all_candles(exchange: ccxt.Exchange, symbol: str, timeframe: str, days: int) -> pd.DataFrame:
    """
    Fetch historical candles with pagination.

    Goes backward from now, fetching CANDLES_PER_REQUEST at a time.
    """
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)
    start_ms = now_ms - (days * 24 * 60 * 60 * 1000)
    target_candles = days * 24 * 60 // 5  # expected total

    all_candles = []
    since = start_ms
    request_count = 0
    last_progress = 0

    print(f"\n📥 Fetching {days} days of {timeframe} data for {symbol}...")
    print(f"   Target: ~{target_candles:,} candles")
    print(f"   Period: {datetime.fromtimestamp(start_ms/1000, tz=timezone.utc):%Y-%m-%d} → {datetime.fromtimestamp(now_ms/1000, tz=timezone.utc):%Y-%m-%d}")
    print()

    while since < now_ms:
        request_count += 1

        try:
            candles = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=CANDLES_PER_REQUEST)
        except ccxt.RateLimitExceeded:
            print("   ⚠️  Rate limited, waiting 10s...")
            time.sleep(10)
            continue
        except ccxt.NetworkError as e:
            print(f"   ⚠️  Network error: {e}, retrying in 5s...")
            time.sleep(5)
            continue
        except Exception as e:
            print(f"   ❌ Error: {e}")
            if all_candles:
                print(f"   Continuing with {len(all_candles)} candles collected so far")
                break
            raise

        if not candles:
            break

        all_candles.extend(candles)

        # Advance `since` past the last candle
        last_ts = candles[-1][0]
        since = last_ts + MS_PER_5MIN

        # Progress
        collected = len(all_candles)
        progress = int(collected / target_candles * 100)
        if progress >= last_progress + 5 or collected == target_candles:
            oldest = datetime.fromtimestamp(all_candles[0][0] / 1000, tz=timezone.utc)
            newest = datetime.fromtimestamp(last_ts / 1000, tz=timezone.utc)
            print(f"   [{progress:3d}%] {collected:,} candles | {oldest:%Y-%m-%d} → {newest:%Y-%m-%d} | {request_count} requests")
            last_progress = progress

        time.sleep(RATE_LIMIT_SLEEP)

    if not all_candles:
        print("❌ No data fetched!")
        sys.exit(1)

    # Convert to DataFrame
    df = pd.DataFrame(all_candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df = df.drop_duplicates(subset='timestamp').sort_values('timestamp').reset_index(drop=True)

    return df


def validate_data(df: pd.DataFrame, expected_days: int) -> None:
    """Print data quality summary."""
    actual_days = (df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]).days
    expected_candles = expected_days * 288
    completeness = len(df) / expected_candles * 100

    # Check for gaps (missing 5-min intervals)
    time_diffs = df['timestamp'].diff().dropna()
    expected_diff = pd.Timedelta(minutes=5)
    gaps = time_diffs[time_diffs > expected_diff * 1.5]

    print(f"\n📊 Data Quality:")
    print(f"   Candles: {len(df):,} (expected ~{expected_candles:,}, {completeness:.1f}%)")
    print(f"   Days: {actual_days} (requested {expected_days})")
    print(f"   Range: {df['timestamp'].iloc[0]} → {df['timestamp'].iloc[-1]}")
    print(f"   Price: ${df['low'].min():,.2f} - ${df['high'].max():,.2f}")
    print(f"   Gaps (>7.5min): {len(gaps)}")
    if len(gaps) > 0 and len(gaps) <= 10:
        for ts, diff in gaps.items():
            print(f"      {df['timestamp'].iloc[ts]}: {diff}")


def main():
    args = parse_args()

    print("=" * 70)
    print(f"HISTORICAL DATA FETCHER — {args.symbol} {DEFAULT_TIMEFRAME} × {args.days} days")
    print("=" * 70)

    # Resolve output path relative to project root
    project_root = Path(__file__).resolve().parent.parent.parent
    output_path = project_root / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create exchange
    exchange = create_exchange(args.exchange)
    print(f"🔗 Exchange: {exchange.id} (public API, no key required)")

    # Fetch
    df = fetch_all_candles(exchange, args.symbol, DEFAULT_TIMEFRAME, args.days)
    validate_data(df, args.days)

    # Save
    df.to_csv(output_path, index=False)
    size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"\n✅ Saved: {output_path}")
    print(f"   Size: {size_mb:.2f} MB | Rows: {len(df):,}")

    print(f"\n📁 Next: Run backtest/validation with this data")
    print(f"   python bingx_rl_trading_bot/scripts/analysis/full_270d_revalidation.py")


if __name__ == "__main__":
    main()
