"""
Prepare Full Feature Dataset for Training
==========================================

Calculate all features for the updated dataset and save for training.
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from calculate_all_features import calculate_all_features

print("="*80)
print("PREPARING FULL FEATURE DATASET")
print("="*80)

# Load updated raw data
data_file = PROJECT_ROOT / "data" / "historical" / "BTCUSDT_5m_max.csv"
print(f"\n📥 Loading data from: {data_file}")
df = pd.read_csv(data_file)
print(f"✅ Loaded {len(df):,} candles")
print(f"   Date range: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")

# Calculate all features
print(f"\n🔄 Calculating all features...")
df = calculate_all_features(df)
print(f"✅ Features calculated: {len(df.columns)} total columns")

# Save with features
output_dir = PROJECT_ROOT / "data" / "features"
output_dir.mkdir(exist_ok=True, parents=True)
output_file = output_dir / "BTCUSDT_5m_features.csv"

print(f"\n💾 Saving to: {output_file}")
df.to_csv(output_file, index=False)
print(f"✅ Saved {len(df):,} rows with {len(df.columns)} columns")

print("\n" + "="*80)
print("FULL FEATURE DATASET READY")
print("="*80)
print(f"\n📊 Dataset Statistics:")
print(f"   Total Rows: {len(df):,}")
print(f"   Total Columns: {len(df.columns)}")
print(f"   Date Range: {df['timestamp'].iloc[0]} → {df['timestamp'].iloc[-1]}")
print(f"   File Size: {output_file.stat().st_size / 1024 / 1024:.2f} MB")
print("\n✅ Ready for model training!")
