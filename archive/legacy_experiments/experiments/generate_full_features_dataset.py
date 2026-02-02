"""
Generate Full Features Dataset
================================

Generate complete 165-feature dataset using calculate_all_features_enhanced_v2.py

Pipeline:
1. Load raw OHLCV data
2. Calculate all 165 features
3. Save to data/features/BTCUSDT_5m_features.csv

Created: 2025-10-27
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from scripts.experiments.calculate_all_features_enhanced_v2 import calculate_all_features_enhanced_v2

print("="*80)
print("FULL FEATURES DATASET GENERATION")
print("="*80)
print()

# Paths
INPUT_FILE = PROJECT_ROOT / "data" / "historical" / "BTCUSDT_5m_max.csv"
OUTPUT_FILE = PROJECT_ROOT / "data" / "features" / "BTCUSDT_5m_features.csv"

# Step 1: Load raw data
print(f"Step 1: Loading raw data from {INPUT_FILE.name}...")
df_raw = pd.read_csv(INPUT_FILE)
print(f"  ✅ Loaded {len(df_raw):,} candles")
print()

# Step 2: Calculate all 165 features
print("Step 2: Calculating ALL 165 features...")
print()
df_features = calculate_all_features_enhanced_v2(df_raw, phase='phase1')
print()

# Step 3: Verify feature count
print("Step 3: Verifying feature count...")
feature_count = len(df_features.columns)
print(f"  Total columns: {feature_count}")
print(f"  Expected: ~165 (OHLCV + 160 features)")

if feature_count < 160:
    print(f"  ⚠️  WARNING: Only {feature_count} columns generated!")
else:
    print(f"  ✅ Feature count looks good!")
print()

# Step 4: Save to file
print(f"Step 4: Saving to {OUTPUT_FILE.name}...")
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
df_features.to_csv(OUTPUT_FILE, index=False)
print(f"  ✅ Saved {len(df_features):,} rows × {len(df_features.columns)} columns")
print()

# Summary
print("="*80)
print("DATASET GENERATION COMPLETE")
print("="*80)
print()
print(f"📁 Output file: {OUTPUT_FILE}")
print(f"📊 Data shape: {len(df_features):,} rows × {len(df_features.columns)} columns")
print(f"📈 Coverage: {len(df_features)/len(df_raw)*100:.1f}% of raw data")
print()

# Show sample features
print("Sample Features (last 3 rows):")
sample_cols = [
    'close', 'rsi', 'macd',
    'ma_200', 'ema_200', 'rsi_200', 'atr_200',
    'vp_poc', 'vp_value_area_high', 'vp_value_area_low',
    'vwap', 'vwap_distance_pct',
    'vp_value_area_width_pct', 'vwap_momentum'
]
available_cols = [c for c in sample_cols if c in df_features.columns]
print(df_features[available_cols].tail(3))
print()

print("="*80)
print("✅ READY FOR MODEL TRAINING")
print("="*80)
