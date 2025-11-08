"""
Data Quality Validation
=======================

Validate CSV data quality to identify potential issues causing strategy failures.

Created: 2025-10-30
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Data
DATA_DIR = PROJECT_ROOT / "data" / "features"
CSV_FILE = DATA_DIR / "BTCUSDT_5m_features_exit_ready.csv"

print("=" * 80)
print("DATA QUALITY VALIDATION")
print("=" * 80)
print()

# Load data
print("Loading data...")
df = pd.read_csv(CSV_FILE)
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values('timestamp').reset_index(drop=True)
print(f"✅ Loaded {len(df):,} candles")
print(f"   Period: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")
print()

# 1. Check for missing values
print("=" * 80)
print("1. MISSING VALUES CHECK")
print("=" * 80)
print()

missing_counts = df.isnull().sum()
missing_pct = (missing_counts / len(df)) * 100

if missing_counts.sum() == 0:
    print("✅ No missing values found")
else:
    print("⚠️ Missing values detected:")
    for col, count in missing_counts[missing_counts > 0].items():
        print(f"   {col}: {count} ({missing_pct[col]:.2f}%)")

print()

# 2. Check for duplicate timestamps
print("=" * 80)
print("2. DUPLICATE TIMESTAMPS CHECK")
print("=" * 80)
print()

duplicates = df['timestamp'].duplicated().sum()
if duplicates == 0:
    print("✅ No duplicate timestamps found")
else:
    print(f"⚠️ {duplicates} duplicate timestamps detected")
    dup_times = df[df['timestamp'].duplicated(keep=False)]['timestamp'].unique()
    print(f"   First 5 duplicates: {dup_times[:5]}")

print()

# 3. Check time intervals (should be 5 minutes)
print("=" * 80)
print("3. TIME INTERVAL CHECK")
print("=" * 80)
print()

time_diffs = df['timestamp'].diff()
expected_interval = pd.Timedelta(minutes=5)

# Count different intervals
interval_counts = time_diffs.value_counts().sort_index()
print(f"Time interval distribution:")
for interval, count in interval_counts.head(10).items():
    if pd.notna(interval):
        minutes = interval.total_seconds() / 60
        pct = (count / len(time_diffs)) * 100
        marker = "✅" if interval == expected_interval else "⚠️"
        print(f"   {marker} {minutes:.1f} minutes: {count} ({pct:.2f}%)")

# Check for gaps (> 5 minutes)
gaps = time_diffs[time_diffs > expected_interval].dropna()
if len(gaps) == 0:
    print("\n✅ No time gaps detected")
else:
    print(f"\n⚠️ {len(gaps)} time gaps detected (> 5 minutes)")
    print(f"   Largest gap: {gaps.max().total_seconds() / 60:.1f} minutes")
    print(f"   Total gap time: {gaps.sum().total_seconds() / 3600:.1f} hours")

print()

# 4. Check price spikes (abnormal changes)
print("=" * 80)
print("4. PRICE SPIKE CHECK")
print("=" * 80)
print()

price_changes = df['close'].pct_change()
price_changes_abs = price_changes.abs()

# Define spike threshold (> 5% in 5 minutes is suspicious)
spike_threshold = 0.05
spikes = price_changes_abs[price_changes_abs > spike_threshold]

if len(spikes) == 0:
    print(f"✅ No price spikes detected (> {spike_threshold*100}% in 5 min)")
else:
    print(f"⚠️ {len(spikes)} price spikes detected (> {spike_threshold*100}% in 5 min)")
    print(f"   Largest spike: {spikes.max()*100:.2f}%")
    print(f"\n   Top 5 spikes:")
    for idx in spikes.nlargest(5).index:
        time = df['timestamp'].iloc[idx]
        price_before = df['close'].iloc[idx-1]
        price_after = df['close'].iloc[idx]
        change = (price_after - price_before) / price_before * 100
        print(f"   {time}: ${price_before:,.2f} → ${price_after:,.2f} ({change:+.2f}%)")

print()

# 5. Check price range sanity
print("=" * 80)
print("5. PRICE RANGE SANITY CHECK")
print("=" * 80)
print()

print(f"Price statistics:")
print(f"   Min: ${df['close'].min():,.2f}")
print(f"   Max: ${df['close'].max():,.2f}")
print(f"   Mean: ${df['close'].mean():,.2f}")
print(f"   Median: ${df['close'].median():,.2f}")
print(f"   Std Dev: ${df['close'].std():,.2f}")

# Check for zeros or negative prices
zero_prices = (df['close'] <= 0).sum()
if zero_prices == 0:
    print("\n✅ No zero or negative prices found")
else:
    print(f"\n⚠️ {zero_prices} zero or negative prices detected")

print()

# 6. Check volume sanity
print("=" * 80)
print("6. VOLUME SANITY CHECK")
print("=" * 80)
print()

print(f"Volume statistics:")
print(f"   Min: {df['volume'].min():,.2f}")
print(f"   Max: {df['volume'].max():,.2f}")
print(f"   Mean: {df['volume'].mean():,.2f}")
print(f"   Median: {df['volume'].median():,.2f}")

zero_volume = (df['volume'] <= 0).sum()
if zero_volume == 0:
    print("\n✅ No zero or negative volume found")
else:
    print(f"\n⚠️ {zero_volume} zero or negative volume candles detected")

print()

# 7. Check indicator values
print("=" * 80)
print("7. INDICATOR VALUES CHECK")
print("=" * 80)
print()

indicators = ['rsi', 'macd', 'ema_9', 'ema_21']
for indicator in indicators:
    if indicator in df.columns:
        inf_count = np.isinf(df[indicator]).sum()
        nan_count = df[indicator].isnull().sum()

        if inf_count == 0 and nan_count == 0:
            print(f"✅ {indicator}: No inf/nan values")
        else:
            print(f"⚠️ {indicator}: {nan_count} NaN, {inf_count} Inf values")
    else:
        print(f"⚠️ {indicator}: Column not found")

print()

# 8. Visual inspection - price chart with gaps highlighted
print("=" * 80)
print("8. GENERATING VISUAL INSPECTION CHART")
print("=" * 80)
print()

fig, axes = plt.subplots(3, 1, figsize=(14, 10))

# Plot 1: Price with gaps
ax1 = axes[0]
ax1.plot(df['timestamp'], df['close'], linewidth=0.5, alpha=0.7)
if len(gaps) > 0:
    gap_indices = time_diffs[time_diffs > expected_interval].index
    for idx in gap_indices:
        ax1.axvline(df['timestamp'].iloc[idx], color='red', alpha=0.3, linewidth=1)
ax1.set_title('Price Chart with Time Gaps (red lines)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Price (USDT)', fontsize=10)
ax1.grid(True, alpha=0.3)

# Plot 2: Price changes
ax2 = axes[1]
ax2.plot(df['timestamp'].iloc[1:], price_changes.iloc[1:] * 100, linewidth=0.5, alpha=0.7)
ax2.axhline(spike_threshold * 100, color='red', linestyle='--', alpha=0.5, label='Spike threshold')
ax2.axhline(-spike_threshold * 100, color='red', linestyle='--', alpha=0.5)
ax2.set_title('Price Changes (%) - Spikes Detection', fontsize=12, fontweight='bold')
ax2.set_ylabel('Change (%)', fontsize=10)
ax2.legend()
ax2.grid(True, alpha=0.3)

# Plot 3: Volume
ax3 = axes[2]
ax3.bar(df['timestamp'], df['volume'], width=0.003, alpha=0.7)
ax3.set_title('Volume Distribution', fontsize=12, fontweight='bold')
ax3.set_ylabel('Volume', fontsize=10)
ax3.set_xlabel('Date', fontsize=10)
ax3.grid(True, alpha=0.3)

plt.tight_layout()

# Save chart
results_dir = PROJECT_ROOT / "results"
results_dir.mkdir(exist_ok=True)
chart_file = results_dir / "data_quality_validation.png"
plt.savefig(chart_file, dpi=150, bbox_inches='tight')
print(f"✅ Chart saved: {chart_file}")
print()

# 9. Summary
print("=" * 80)
print("DATA QUALITY SUMMARY")
print("=" * 80)
print()

issues = []
if missing_counts.sum() > 0:
    issues.append("Missing values detected")
if duplicates > 0:
    issues.append("Duplicate timestamps detected")
if len(gaps) > 0:
    issues.append(f"{len(gaps)} time gaps detected")
if len(spikes) > 0:
    issues.append(f"{len(spikes)} price spikes detected")
if zero_prices > 0:
    issues.append("Zero/negative prices detected")
if zero_volume > 0:
    issues.append("Zero/negative volume detected")

if len(issues) == 0:
    print("✅ DATA APPEARS CLEAN")
    print("   No major quality issues detected")
    print("   Strategy failures likely due to market conditions or strategy design")
else:
    print("⚠️ DATA QUALITY ISSUES DETECTED:")
    for i, issue in enumerate(issues, 1):
        print(f"   {i}. {issue}")
    print("\n   These issues may contribute to strategy failures")

print()
print("=" * 80)
print("VALIDATION COMPLETE")
print("=" * 80)
