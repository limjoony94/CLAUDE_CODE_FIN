"""
Signal Distribution Analysis
============================

Analyze LONG/SHORT signal probability distributions from bot logs.
"""

import re
from pathlib import Path
import numpy as np
from collections import Counter

PROJECT_ROOT = Path(__file__).parent.parent.parent
LOG_FILE = PROJECT_ROOT / "logs" / "bot_pipeline_debug_20251019.log"

print("="*80)
print("SIGNAL DISTRIBUTION ANALYSIS")
print("="*80)

# Extract signals from log
pattern = r"LONG: ([\d.]+) \| SHORT: ([\d.]+)"
long_probs = []
short_probs = []

if LOG_FILE.exists():
    with open(LOG_FILE, 'r', encoding='utf-8') as f:
        for line in f:
            match = re.search(pattern, line)
            if match:
                long_probs.append(float(match.group(1)))
                short_probs.append(float(match.group(2)))

if not long_probs:
    print("\n❌ No signals found in log file")
    print(f"   Checked: {LOG_FILE}")
    exit(1)

print(f"\n✅ Extracted {len(long_probs)} signals from log\n")

# LONG Distribution
print("="*80)
print("LONG SIGNAL DISTRIBUTION")
print("="*80)
long_arr = np.array(long_probs)
print(f"\nCount: {len(long_arr)}")
print(f"Min:   {long_arr.min():.4f} ({long_arr.min()*100:.2f}%)")
print(f"Max:   {long_arr.max():.4f} ({long_arr.max()*100:.2f}%)")
print(f"Mean:  {long_arr.mean():.4f} ({long_arr.mean()*100:.2f}%)")
print(f"Median: {np.median(long_arr):.4f} ({np.median(long_arr)*100:.2f}%)")
print(f"Std:   {long_arr.std():.4f}")

print("\nPercentiles:")
for p in [10, 25, 50, 75, 90, 95, 99]:
    val = np.percentile(long_arr, p)
    print(f"  {p:2d}th: {val:.4f} ({val*100:.2f}%)")

print("\nThreshold Analysis:")
for thresh in [0.50, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:
    count = np.sum(long_arr >= thresh)
    pct = count / len(long_arr) * 100
    print(f"  >= {thresh:.2f}: {count:3d} signals ({pct:5.1f}%)")

# SHORT Distribution
print("\n" + "="*80)
print("SHORT SIGNAL DISTRIBUTION")
print("="*80)
short_arr = np.array(short_probs)
print(f"\nCount: {len(short_arr)}")
print(f"Min:   {short_arr.min():.4f} ({short_arr.min()*100:.2f}%)")
print(f"Max:   {short_arr.max():.4f} ({short_arr.max()*100:.2f}%)")
print(f"Mean:  {short_arr.mean():.4f} ({short_arr.mean()*100:.2f}%)")
print(f"Median: {np.median(short_arr):.4f} ({np.median(short_arr)*100:.2f}%)")
print(f"Std:   {short_arr.std():.4f}")

print("\nPercentiles:")
for p in [10, 25, 50, 75, 90, 95, 99]:
    val = np.percentile(short_arr, p)
    print(f"  {p:2d}th: {val:.4f} ({val*100:.2f}%)")

print("\nThreshold Analysis:")
for thresh in [0.50, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]:
    count = np.sum(short_arr >= thresh)
    pct = count / len(short_arr) * 100
    print(f"  >= {thresh:.2f}: {count:3d} signals ({pct:5.1f}%)")

# Histogram
print("\n" + "="*80)
print("LONG PROBABILITY HISTOGRAM")
print("="*80)
bins = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
hist, _ = np.histogram(long_arr, bins=bins)
for i, count in enumerate(hist):
    bin_start = bins[i]
    bin_end = bins[i+1]
    pct = count / len(long_arr) * 100
    bar = "█" * int(pct / 2)  # Scale bar
    print(f"  {bin_start:.2f}-{bin_end:.2f}: {count:3d} ({pct:5.1f}%) {bar}")

print("\n" + "="*80)
print("SHORT PROBABILITY HISTOGRAM")
print("="*80)
bins = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
hist, _ = np.histogram(short_arr, bins=bins)
for i, count in enumerate(hist):
    bin_start = bins[i]
    bin_end = bins[i+1]
    pct = count / len(short_arr) * 100
    bar = "█" * int(pct / 2)  # Scale bar
    print(f"  {bin_start:.2f}-{bin_end:.2f}: {count:3d} ({pct:5.1f}%) {bar}")

# Current Thresholds
print("\n" + "="*80)
print("CURRENT THRESHOLD ANALYSIS")
print("="*80)
print(f"\nCurrent Thresholds:")
print(f"  LONG:  0.65 (65%)")
print(f"  SHORT: 0.70 (70%)")

long_above = np.sum(long_arr >= 0.65)
short_above = np.sum(short_arr >= 0.70)
print(f"\nSignals Above Threshold:")
print(f"  LONG  >= 0.65: {long_above:3d} / {len(long_arr)} ({long_above/len(long_arr)*100:.1f}%)")
print(f"  SHORT >= 0.70: {short_above:3d} / {len(short_arr)} ({short_above/len(short_arr)*100:.1f}%)")

# Recommendations
print("\n" + "="*80)
print("THRESHOLD RECOMMENDATIONS")
print("="*80)

# Find threshold that gives ~20% of signals (trade frequency control)
for target_pct in [10, 15, 20, 25, 30]:
    long_thresh = np.percentile(long_arr, 100 - target_pct)
    short_thresh = np.percentile(short_arr, 100 - target_pct)
    print(f"\nFor ~{target_pct}% trade frequency:")
    print(f"  LONG:  {long_thresh:.4f} ({long_thresh*100:.2f}%)")
    print(f"  SHORT: {short_thresh:.4f} ({short_thresh*100:.2f}%)")

print("\n" + "="*80)
