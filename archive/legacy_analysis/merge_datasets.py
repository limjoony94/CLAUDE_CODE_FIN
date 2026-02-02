"""
Merge Historical and Recent Data
=================================
Combine BTCUSDT_5m_max.csv (until Oct 18) with recent data (Oct 15-20)
"""
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "historical"

print("=" * 80)
print("📊 Merging Historical and Recent Data")
print("=" * 80)

# Load historical data (until Oct 18)
print("\nLoading historical data...")
df_hist = pd.read_csv(DATA_DIR / "BTCUSDT_5m_max.csv")
df_hist['timestamp'] = pd.to_datetime(df_hist['timestamp'])
print(f"✅ Historical: {len(df_hist)} candles")
print(f"   Range: {df_hist['timestamp'].min()} to {df_hist['timestamp'].max()}")

# Load recent data (Oct 15-20)
print("\nLoading recent data...")
df_recent = pd.read_csv(DATA_DIR / "BTCUSDT_5m_recent.csv")
df_recent['timestamp'] = pd.to_datetime(df_recent['timestamp'])
print(f"✅ Recent: {len(df_recent)} candles")
print(f"   Range: {df_recent['timestamp'].min()} to {df_recent['timestamp'].max()}")

# Find overlap
overlap_start = max(df_hist['timestamp'].min(), df_recent['timestamp'].min())
overlap_end = min(df_hist['timestamp'].max(), df_recent['timestamp'].max())
print(f"\n🔍 Overlap period: {overlap_start} to {overlap_end}")

# Keep only data after historical cutoff
cutoff = df_hist['timestamp'].max()
df_new = df_recent[df_recent['timestamp'] > cutoff].copy()
print(f"\n✂️  Filtering new data (after {cutoff})...")
print(f"   New data points: {len(df_new)}")

# Combine
df_combined = pd.concat([df_hist, df_new], ignore_index=True)
df_combined = df_combined.sort_values('timestamp').reset_index(drop=True)

# Remove duplicates
df_combined = df_combined.drop_duplicates(subset=['timestamp'], keep='last').reset_index(drop=True)

print(f"\n✅ Combined dataset:")
print(f"   Total candles: {len(df_combined)}")
print(f"   Range: {df_combined['timestamp'].min()} to {df_combined['timestamp'].max()}")
print(f"   Days: {(df_combined['timestamp'].max() - df_combined['timestamp'].min()).days}")

# Save
output_path = DATA_DIR / "BTCUSDT_5m_updated.csv"
df_combined.to_csv(output_path, index=False)
print(f"\n💾 Saved to: {output_path}")

# Backup old file
backup_path = DATA_DIR / "BTCUSDT_5m_max_backup.csv"
if not backup_path.exists():
    df_hist.to_csv(backup_path, index=False)
    print(f"💾 Backed up old file to: {backup_path}")

print("\n" + "=" * 80)
print("✅ MERGE COMPLETE")
print("=" * 80)
