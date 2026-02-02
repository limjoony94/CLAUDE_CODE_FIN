"""
Analyze Label Quality Differences: 52-Day vs 90-Day Periods
===========================================================

Purpose: Understand why 52-day period naturally achieves 6.6-12.5% labels
         with strict parameters, but 90-day period only gets 1.4-2.5%
         even with relaxed parameters.

52-Day Period (SUCCESS):
  Training: Aug 7 - Sep 28, 2025 (52 days)
  Labels: 3% in 30min (6 candles @ 5-min)
  Result: 6.6% LONG, 12.5% SHORT

90-Day Period (FAILURE):
  Training: Aug 8 - Nov 6, 2025 (89 days)
  Labels: 1.5% in 120min (24 candles @ 5-min)
  Result: 1.4% LONG, 2.5% SHORT

Question: What makes the 52-day period naturally more "label-friendly"?

Created: 2025-11-06 20:00 KST
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
from datetime import datetime

# Paths
DATA_DIR = PROJECT_ROOT / "data" / "features"
RESULTS_DIR = PROJECT_ROOT / "results"

# Input
RAW_90D = DATA_DIR / "BTCUSDT_5m_raw_90days_20251106_163815.csv"

print("=" * 80)
print("LABEL QUALITY ANALYSIS: 52-DAY vs 90-DAY PERIODS")
print("=" * 80)
print()

# ============================================================================
# LOAD DATA
# ============================================================================

print("=" * 80)
print("LOADING DATA")
print("=" * 80)
print()

print(f"📖 Loading 90-day raw data: {RAW_90D.name}")
df_90d = pd.read_csv(RAW_90D)
df_90d['timestamp'] = pd.to_datetime(df_90d['timestamp'])

print(f"   Rows: {len(df_90d):,}")
print(f"   Period: {df_90d['timestamp'].min()} to {df_90d['timestamp'].max()}")
print(f"   Days: {(df_90d['timestamp'].max() - df_90d['timestamp'].min()).days}")
print()

# Extract 52-day period (Aug 7 - Sep 28, 2025)
START_52D = "2025-08-07"
END_52D = "2025-09-28"

df_52d = df_90d[(df_90d['timestamp'] >= START_52D) & (df_90d['timestamp'] <= END_52D)].copy()

print(f"📖 Extracted 52-day period: {START_52D} to {END_52D}")
print(f"   Rows: {len(df_52d):,}")
print(f"   Period: {df_52d['timestamp'].min()} to {df_52d['timestamp'].max()}")
print(f"   Days: {(df_52d['timestamp'].max() - df_52d['timestamp'].min()).days}")
print()

# ============================================================================
# BASIC STATISTICS
# ============================================================================

print("=" * 80)
print("BASIC STATISTICS COMPARISON")
print("=" * 80)
print()

def calc_stats(df, label):
    """Calculate basic price statistics"""

    stats = {
        'Label': label,
        'Candles': len(df),
        'Price Mean': df['close'].mean(),
        'Price Std': df['close'].std(),
        'Price Min': df['close'].min(),
        'Price Max': df['close'].max(),
        'Price Range': df['close'].max() - df['close'].min(),
        'Price Range %': (df['close'].max() - df['close'].min()) / df['close'].mean() * 100
    }

    return stats

stats_52d = calc_stats(df_52d, "52-Day (Aug 7 - Sep 28)")
stats_90d = calc_stats(df_90d, "90-Day (Aug 8 - Nov 6)")

df_stats = pd.DataFrame([stats_52d, stats_90d])

print(df_stats.to_string(index=False))
print()

# ============================================================================
# VOLATILITY ANALYSIS
# ============================================================================

print("=" * 80)
print("VOLATILITY ANALYSIS")
print("=" * 80)
print()

def calc_volatility(df, label):
    """Calculate various volatility measures"""

    # Calculate returns
    df = df.copy()
    df['returns'] = df['close'].pct_change()
    df['abs_returns'] = df['returns'].abs()

    # 5-min returns
    ret_5m = df['returns'].dropna()

    # 30-min returns (6 candles)
    ret_30m = df['close'].pct_change(6)

    # 60-min returns (12 candles)
    ret_60m = df['close'].pct_change(12)

    # 120-min returns (24 candles)
    ret_120m = df['close'].pct_change(24)

    stats = {
        'Label': label,
        'Mean 5m Move': ret_5m.abs().mean() * 100,
        'Mean 30m Move': ret_30m.abs().mean() * 100,
        'Mean 60m Move': ret_60m.abs().mean() * 100,
        'Mean 120m Move': ret_120m.abs().mean() * 100,
        'Std 5m': ret_5m.std() * 100,
        'Std 30m': ret_30m.std() * 100,
        'Std 60m': ret_60m.std() * 100,
        'Std 120m': ret_120m.std() * 100,
        'Max 30m Up': ret_30m.max() * 100,
        'Max 30m Down': ret_30m.min() * 100,
        'Max 120m Up': ret_120m.max() * 100,
        'Max 120m Down': ret_120m.min() * 100
    }

    return stats, ret_30m, ret_120m

vol_52d, ret_30m_52d, ret_120m_52d = calc_volatility(df_52d, "52-Day")
vol_90d, ret_30m_90d, ret_120m_90d = calc_volatility(df_90d, "90-Day")

df_vol = pd.DataFrame([vol_52d, vol_90d])

print(df_vol.to_string(index=False))
print()

# ============================================================================
# LABEL POTENTIAL ANALYSIS
# ============================================================================

print("=" * 80)
print("LABEL POTENTIAL ANALYSIS (3% in 30min vs 1.5% in 120min)")
print("=" * 80)
print()

def calc_label_potential(df, threshold_pct, lookforward, label):
    """Calculate how many candles would generate labels with given parameters"""

    df = df.copy()
    n = len(df)

    long_count = 0
    short_count = 0

    for i in range(n - lookforward):
        current_price = df.iloc[i]['close']
        future_prices = df.iloc[i+1:i+1+lookforward]['close'].values

        if len(future_prices) == 0:
            continue

        # LONG: Price rises by threshold
        max_future = np.max(future_prices)
        if (max_future - current_price) / current_price >= threshold_pct:
            long_count += 1

        # SHORT: Price falls by threshold
        min_future = np.min(future_prices)
        if (current_price - min_future) / current_price >= threshold_pct:
            short_count += 1

    long_pct = long_count / n * 100
    short_pct = short_count / n * 100

    return {
        'Label': label,
        'LONG Count': long_count,
        'LONG %': long_pct,
        'SHORT Count': short_count,
        'SHORT %': short_pct,
        'Total %': long_pct + short_pct
    }

# Test with strict parameters (52-day successful approach)
print("📊 Strict Labels (3% in 30min - 52-day approach):")
print()

strict_52d = calc_label_potential(df_52d, 0.03, 6, "52-Day (Aug 7 - Sep 28)")
strict_90d = calc_label_potential(df_90d, 0.03, 6, "90-Day (Aug 8 - Nov 6)")

df_strict = pd.DataFrame([strict_52d, strict_90d])
print(df_strict.to_string(index=False))
print()

# Test with relaxed parameters (90-day attempted approach)
print("📊 Relaxed Labels (1.5% in 120min - 90-day attempt):")
print()

relaxed_52d = calc_label_potential(df_52d, 0.015, 24, "52-Day (Aug 7 - Sep 28)")
relaxed_90d = calc_label_potential(df_90d, 0.015, 24, "90-Day (Aug 8 - Nov 6)")

df_relaxed = pd.DataFrame([relaxed_52d, relaxed_90d])
print(df_relaxed.to_string(index=False))
print()

# ============================================================================
# DIRECTIONAL BIAS ANALYSIS
# ============================================================================

print("=" * 80)
print("DIRECTIONAL BIAS ANALYSIS")
print("=" * 80)
print()

def calc_directional_bias(df, label):
    """Analyze directional bias in price movements"""

    df = df.copy()

    # Overall trend
    first_close = df.iloc[0]['close']
    last_close = df.iloc[-1]['close']
    overall_change = (last_close - first_close) / first_close * 100

    # Count up/down candles
    df['is_up'] = df['close'] > df['open']
    up_candles = df['is_up'].sum()
    down_candles = (~df['is_up']).sum()

    # 30-min directional consistency
    df['ret_30m'] = df['close'].pct_change(6)
    up_30m = (df['ret_30m'] > 0).sum()
    down_30m = (df['ret_30m'] < 0).sum()

    # 120-min directional consistency
    df['ret_120m'] = df['close'].pct_change(24)
    up_120m = (df['ret_120m'] > 0).sum()
    down_120m = (df['ret_120m'] < 0).sum()

    return {
        'Label': label,
        'Overall Change %': overall_change,
        'Up Candles': up_candles,
        'Down Candles': down_candles,
        'Up %': up_candles / (up_candles + down_candles) * 100,
        'Up 30m Moves': up_30m,
        'Down 30m Moves': down_30m,
        'Up 30m %': up_30m / (up_30m + down_30m) * 100 if (up_30m + down_30m) > 0 else 0,
        'Up 120m Moves': up_120m,
        'Down 120m Moves': down_120m,
        'Up 120m %': up_120m / (up_120m + down_120m) * 100 if (up_120m + down_120m) > 0 else 0
    }

bias_52d = calc_directional_bias(df_52d, "52-Day (Aug 7 - Sep 28)")
bias_90d = calc_directional_bias(df_90d, "90-Day (Aug 8 - Nov 6)")

df_bias = pd.DataFrame([bias_52d, bias_90d])

print(df_bias.to_string(index=False))
print()

# ============================================================================
# MARKET REGIME ANALYSIS
# ============================================================================

print("=" * 80)
print("MARKET REGIME ANALYSIS (Weekly Breakdown)")
print("=" * 80)
print()

def weekly_breakdown(df, label):
    """Break down price action by week"""

    df = df.copy()
    df['week'] = df['timestamp'].dt.isocalendar().week
    df['year'] = df['timestamp'].dt.year

    weekly_stats = []

    for (year, week), group in df.groupby(['year', 'week']):
        if len(group) < 10:  # Skip incomplete weeks
            continue

        first_close = group.iloc[0]['close']
        last_close = group.iloc[-1]['close']
        weekly_change = (last_close - first_close) / first_close * 100

        # Volatility
        group['returns'] = group['close'].pct_change()
        volatility = group['returns'].std() * 100

        # Label potential (3% in 30min)
        long_labels = 0
        short_labels = 0

        for i in range(len(group) - 6):
            current = group.iloc[i]['close']
            future = group.iloc[i+1:i+7]['close'].values

            if len(future) > 0:
                max_fut = np.max(future)
                min_fut = np.min(future)

                if (max_fut - current) / current >= 0.03:
                    long_labels += 1
                if (current - min_fut) / current >= 0.03:
                    short_labels += 1

        weekly_stats.append({
            'Period': label,
            'Year': year,
            'Week': week,
            'Start Date': group.iloc[0]['timestamp'].strftime('%Y-%m-%d'),
            'Candles': len(group),
            'Weekly Change %': weekly_change,
            'Volatility %': volatility,
            'LONG Labels': long_labels,
            'SHORT Labels': short_labels,
            'Total Labels': long_labels + short_labels,
            'Label %': (long_labels + short_labels) / len(group) * 100
        })

    return pd.DataFrame(weekly_stats)

print("📅 52-Day Period Weekly Breakdown:")
df_weekly_52d = weekly_breakdown(df_52d, "52-Day")
print(df_weekly_52d.to_string(index=False))
print()

print("📅 90-Day Period Weekly Breakdown:")
df_weekly_90d = weekly_breakdown(df_90d, "90-Day")
print(df_weekly_90d.to_string(index=False))
print()

# ============================================================================
# KEY INSIGHTS
# ============================================================================

print("=" * 80)
print("KEY INSIGHTS")
print("=" * 80)
print()

print("🔍 Volatility Comparison:")
print(f"   52-Day Mean 30m Move: {vol_52d['Mean 30m Move']:.3f}%")
print(f"   90-Day Mean 30m Move: {vol_90d['Mean 30m Move']:.3f}%")
print(f"   Difference: {vol_52d['Mean 30m Move'] - vol_90d['Mean 30m Move']:+.3f}%")
print()

print("🔍 Label Potential (Strict 3% in 30min):")
print(f"   52-Day: {strict_52d['LONG %']:.2f}% LONG, {strict_52d['SHORT %']:.2f}% SHORT")
print(f"   90-Day: {strict_90d['LONG %']:.2f}% LONG, {strict_90d['SHORT %']:.2f}% SHORT")
print(f"   Difference: {strict_52d['Total %'] - strict_90d['Total %']:+.2f}%")
print()

print("🔍 Directional Bias:")
print(f"   52-Day Overall Change: {bias_52d['Overall Change %']:+.2f}%")
print(f"   90-Day Overall Change: {bias_90d['Overall Change %']:+.2f}%")
print(f"   52-Day Up/Down: {bias_52d['Up %']:.1f}% / {100-bias_52d['Up %']:.1f}%")
print(f"   90-Day Up/Down: {bias_90d['Up %']:.1f}% / {100-bias_90d['Up %']:.1f}%")
print()

# Calculate which weeks in 90-day are NOT in 52-day (the problematic extension)
print("🔍 Period Extension Analysis:")
print("   52-Day: Aug 7 - Sep 28 (weeks that worked)")
print("   90-Day Extension: Aug 8 - Sep 28 + Sep 29 - Nov 6 (added weeks)")
print()

# Weeks in 90-day that are NOT in 52-day
df_extension = df_weekly_90d[df_weekly_90d['Start Date'] > END_52D]

if len(df_extension) > 0:
    print("   📊 Extended Period Stats (Sep 29 - Nov 6):")
    print(f"      Weeks: {len(df_extension)}")
    print(f"      Avg Weekly Change: {df_extension['Weekly Change %'].mean():+.2f}%")
    print(f"      Avg Volatility: {df_extension['Volatility %'].mean():.3f}%")
    print(f"      Avg Label %: {df_extension['Label %'].mean():.2f}%")
    print()

    print("   📊 Original 52-Day Stats (Aug 7 - Sep 28):")
    df_original = df_weekly_90d[df_weekly_90d['Start Date'] <= END_52D]
    print(f"      Weeks: {len(df_original)}")
    print(f"      Avg Weekly Change: {df_original['Weekly Change %'].mean():+.2f}%")
    print(f"      Avg Volatility: {df_original['Volatility %'].mean():.3f}%")
    print(f"      Avg Label %: {df_original['Label %'].mean():.2f}%")
    print()

# ============================================================================
# CONCLUSIONS
# ============================================================================

print("=" * 80)
print("CONCLUSIONS")
print("=" * 80)
print()

print("❓ Why does 52-day achieve 6.6-12.5% labels with strict parameters,")
print("   but 90-day only gets 1.4-2.5% even with relaxed parameters?")
print()

print("💡 Hypothesis to test:")
print("   1. Higher volatility in 52-day period → more 3% moves")
print("   2. Specific market regime (bull/consolidation) in Aug-Sep")
print("   3. Extended period (Sep 29 - Nov 6) has lower volatility")
print("   4. Different price level affects percentage-based thresholds")
print()

print("✅ Analysis complete!")
print(f"   Results provide data to understand label quality differences")
print()
