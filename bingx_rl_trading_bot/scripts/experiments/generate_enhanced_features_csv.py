"""
Generate Enhanced Features CSV
================================

Reads existing features CSV and adds enhanced features needed for Exit models.

Created: 2025-10-29
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from calculate_all_features import calculate_all_features

DATA_DIR = PROJECT_ROOT / "data" / "features"
INPUT_FILE = DATA_DIR / "BTCUSDT_5m_features.csv"
OUTPUT_FILE = DATA_DIR / "BTCUSDT_5m_features_enhanced.csv"

print("="*80)
print("GENERATING ENHANCED FEATURES CSV")
print("="*80)
print()

# Load existing CSV
print(f"Loading: {INPUT_FILE}")
df = pd.read_csv(INPUT_FILE)
print(f"✅ Loaded {len(df):,} rows, {len(df.columns)} columns")
print()

# Calculate enhanced features
print("Calculating enhanced features...")
df = calculate_all_features(df)
print(f"✅ Enhanced features calculated")
print(f"   Total columns: {len(df.columns)}")
print()

# Save to new file
print(f"Saving to: {OUTPUT_FILE}")
df.to_csv(OUTPUT_FILE, index=False)
print(f"✅ Saved {len(df):,} rows, {len(df.columns)} columns")
print()

# Verify enhanced features present
enhanced_features = ['volume_surge', 'price_acceleration', 'rsi_slope', 'price_vs_ma20', 'price_vs_ma50']
missing = [f for f in enhanced_features if f not in df.columns]

if missing:
    print(f"⚠️  Missing enhanced features: {missing}")
else:
    print(f"✅ All enhanced features present!")

print()
print("="*80)
print("COMPLETE")
print("="*80)
