#!/usr/bin/env python3
"""
Advanced Indicators Validation Test
====================================

Comprehensive testing and validation of advanced indicators:
- Statistical validation
- Signal quality checks
- Trading signal examples
- Performance benchmarking

Created: 2025-10-23
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
import time

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.indicators.advanced_indicators import (
    calculate_all_advanced_indicators,
    get_all_advanced_features
)


def validate_feature_quality(df, feature_name):
    """
    Validate feature quality and return metrics
    """
    series = df[feature_name]

    return {
        'name': feature_name,
        'missing_pct': (series.isna().sum() / len(series)) * 100,
        'unique_values': series.nunique(),
        'mean': series.mean(),
        'std': series.std(),
        'min': series.min(),
        'max': series.max(),
        'zero_pct': ((series == 0).sum() / len(series)) * 100
    }


def test_volume_profile_signals(df):
    """
    Test Volume Profile trading signals
    """
    print("Volume Profile Signal Tests:")
    print("-" * 80)

    # Test 1: Price near POC (neutral zone)
    near_poc = df[abs(df['vp_distance_to_poc_pct']) < 0.005]
    print(f"  Near POC (<0.5%): {len(near_poc):,} candles ({len(near_poc)/len(df)*100:.1f}%)")

    # Test 2: In Value Area
    in_va = df[df['vp_in_value_area'] == 1]
    print(f"  In Value Area: {len(in_va):,} candles ({len(in_va)/len(df)*100:.1f}%)")

    # Test 3: Above Value Area (potential SHORT)
    above_va = df[df['close'] > df['vp_value_area_high']]
    print(f"  Above Value Area: {len(above_va):,} candles ({len(above_va)/len(df)*100:.1f}%)")

    # Test 4: Below Value Area (potential LONG)
    below_va = df[df['close'] < df['vp_value_area_low']]
    print(f"  Below Value Area: {len(below_va):,} candles ({len(below_va)/len(df)*100:.1f}%)")

    # Test 5: Strong volume imbalance
    strong_buy = df[df['vp_volume_imbalance'] > 0.2]
    strong_sell = df[df['vp_volume_imbalance'] < -0.2]
    print(f"  Strong Buy Imbalance (>0.2): {len(strong_buy):,} candles ({len(strong_buy)/len(df)*100:.1f}%)")
    print(f"  Strong Sell Imbalance (<-0.2): {len(strong_sell):,} candles ({len(strong_sell)/len(df)*100:.1f}%)")

    print()


def test_vwap_signals(df):
    """
    Test VWAP trading signals
    """
    print("VWAP Signal Tests:")
    print("-" * 80)

    # Test 1: Price above VWAP
    above_vwap = df[df['vwap_above'] == 1]
    print(f"  Above VWAP: {len(above_vwap):,} candles ({len(above_vwap)/len(df)*100:.1f}%)")

    # Test 2: Price near VWAP (<0.5%)
    near_vwap = df[abs(df['vwap_distance_pct']) < 0.005]
    print(f"  Near VWAP (<0.5%): {len(near_vwap):,} candles ({len(near_vwap)/len(df)*100:.1f}%)")

    # Test 3: VWAP crossovers
    vwap_cross_up = df[(df['vwap_above'] == 1) & (df['vwap_above'].shift(1) == 0)]
    vwap_cross_down = df[(df['vwap_above'] == 0) & (df['vwap_above'].shift(1) == 1)]
    print(f"  VWAP Cross Up: {len(vwap_cross_up):,} signals")
    print(f"  VWAP Cross Down: {len(vwap_cross_down):,} signals")

    # Test 4: Band position extremes
    overbought = df[df['vwap_band_position'] > 0.8]
    oversold = df[df['vwap_band_position'] < 0.2]
    print(f"  Overbought (>0.8): {len(overbought):,} candles ({len(overbought)/len(df)*100:.1f}%)")
    print(f"  Oversold (<0.2): {len(oversold):,} candles ({len(oversold)/len(df)*100:.1f}%)")

    print()


def benchmark_performance(df):
    """
    Benchmark calculation performance
    """
    print("Performance Benchmark:")
    print("-" * 80)

    # Test data sizes
    test_sizes = [1000, 5000, 10000, 20000]

    for size in test_sizes:
        if size > len(df):
            continue

        test_df = df.head(size).copy()

        start = time.time()
        result = calculate_all_advanced_indicators(test_df, phase='phase1')
        elapsed = time.time() - start

        candles_per_sec = size / elapsed
        print(f"  {size:6,} candles: {elapsed:6.2f}s ({candles_per_sec:8.1f} candles/sec)")

    print()


def find_trading_opportunities(df, lookback=100):
    """
    Find recent trading opportunities based on signals
    """
    print("Recent Trading Opportunities (Last 100 candles):")
    print("-" * 80)

    recent = df.tail(lookback).copy()

    # LONG opportunities (below Value Area + below VWAP + buy imbalance)
    long_signals = recent[
        (recent['close'] < recent['vp_value_area_low']) &
        (recent['vwap_above'] == 0) &
        (recent['vp_volume_imbalance'] > 0.1)
    ]

    # SHORT opportunities (above Value Area + above VWAP + sell imbalance)
    short_signals = recent[
        (recent['close'] > recent['vp_value_area_high']) &
        (recent['vwap_above'] == 1) &
        (recent['vp_volume_imbalance'] < -0.1)
    ]

    print(f"  LONG Opportunities: {len(long_signals)}")
    if len(long_signals) > 0:
        print(f"    Latest: Index {long_signals.index[-1]}")
        print(f"    Price: {long_signals['close'].iloc[-1]:.2f}")
        print(f"    Below VA by: {(long_signals['vp_value_area_low'].iloc[-1] - long_signals['close'].iloc[-1]):.2f}")

    print(f"\n  SHORT Opportunities: {len(short_signals)}")
    if len(short_signals) > 0:
        print(f"    Latest: Index {short_signals.index[-1]}")
        print(f"    Price: {short_signals['close'].iloc[-1]:.2f}")
        print(f"    Above VA by: {(short_signals['close'].iloc[-1] - short_signals['vp_value_area_high'].iloc[-1]):.2f}")

    print()


def main():
    print("="*80)
    print("ADVANCED INDICATORS VALIDATION TEST")
    print("="*80)
    print()

    # Load data
    data_file = PROJECT_ROOT / "data" / "historical" / "BTCUSDT_5m_max.csv"
    df = pd.read_csv(data_file)
    print(f"✅ Loaded {len(df):,} candles")
    print()

    # Calculate indicators
    print("="*80)
    print("CALCULATING INDICATORS")
    print("="*80)
    print()

    start = time.time()
    df_enhanced = calculate_all_advanced_indicators(df, phase='phase1')
    elapsed = time.time() - start

    print(f"\n⏱️  Total calculation time: {elapsed:.2f}s")
    print(f"   Speed: {len(df_enhanced)/elapsed:.1f} candles/sec")
    print()

    # Feature quality validation
    print("="*80)
    print("FEATURE QUALITY VALIDATION")
    print("="*80)
    print()

    features = get_all_advanced_features('phase1')
    quality_results = []

    for feature in features:
        quality = validate_feature_quality(df_enhanced, feature)
        quality_results.append(quality)

    quality_df = pd.DataFrame(quality_results)

    print("Feature Statistics:")
    print("-" * 80)
    print(quality_df[['name', 'missing_pct', 'unique_values', 'mean', 'std']].to_string(index=False))
    print()

    # Check for issues
    issues = quality_df[
        (quality_df['missing_pct'] > 5) |  # >5% missing
        (quality_df['zero_pct'] > 90)       # >90% zeros
    ]

    if len(issues) > 0:
        print("⚠️  Quality Issues Detected:")
        print(issues[['name', 'missing_pct', 'zero_pct']].to_string(index=False))
    else:
        print("✅ All features passed quality checks")

    print()

    # Signal tests
    print("="*80)
    print("TRADING SIGNAL TESTS")
    print("="*80)
    print()

    test_volume_profile_signals(df_enhanced)
    test_vwap_signals(df_enhanced)

    # Find opportunities
    print("="*80)
    print("OPPORTUNITY DETECTION")
    print("="*80)
    print()

    find_trading_opportunities(df_enhanced, lookback=100)

    # Performance benchmark
    print("="*80)
    print("PERFORMANCE BENCHMARK")
    print("="*80)
    print()

    benchmark_performance(df)

    # Summary
    print("="*80)
    print("VALIDATION SUMMARY")
    print("="*80)
    print()

    print(f"✅ Feature Count: {len(features)}")
    print(f"✅ Data Coverage: {len(df_enhanced):,} / {len(df):,} candles ({len(df_enhanced)/len(df)*100:.1f}%)")
    print(f"✅ Calculation Speed: {len(df_enhanced)/elapsed:.1f} candles/sec")
    print(f"✅ Quality Issues: {len(issues)}")
    print()

    print("="*80)
    print("TEST COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
