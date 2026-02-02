"""
Initialize Live Features CSV
=============================

Create initial BTCUSDT_5m_features_live.csv with 50K+ candles.
This provides proper context for production bot to match training conditions.

Problem:
- Training: 155K candles context
- Production: 1K candles only → Feature distribution shift → 0.0000 probabilities

Solution:
- Maintain 50K+ rolling context in production
- Load from this CSV + append new candles
- Ensures consistent feature distributions

Usage:
    python scripts/utils/initialize_live_features.py

Output:
    data/features/BTCUSDT_5m_features_live.csv (~50K candles with all features)

Created: 2025-10-28
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from scripts.experiments.calculate_all_features_enhanced_v2 import calculate_all_features_enhanced_v2

print("="*80)
print("INITIALIZE LIVE FEATURES CSV")
print("="*80)
print()

# Paths
INPUT_FILE = PROJECT_ROOT / "data" / "historical" / "BTCUSDT_5m_max.csv"
OUTPUT_FILE = PROJECT_ROOT / "data" / "features" / "BTCUSDT_5m_features_live.csv"

# Check input file exists
if not INPUT_FILE.exists():
    print(f"❌ ERROR: Input file not found: {INPUT_FILE}")
    print()
    print("Please ensure BTCUSDT_5m_max.csv exists in data/historical/")
    sys.exit(1)

# Load last 60K candles (will become ~50K after dropna)
print(f"📂 Loading from: {INPUT_FILE.name}")
df_all = pd.read_csv(INPUT_FILE)
print(f"   Total available: {len(df_all):,} candles")
print()

# Take last 60K candles
print("🔍 Extracting last 60K candles...")
df_raw = df_all.tail(60000).reset_index(drop=True)
print(f"   ✅ Extracted: {len(df_raw):,} candles")
print(f"      From: {df_raw['timestamp'].iloc[0]}")
print(f"      To:   {df_raw['timestamp'].iloc[-1]}")
print()

# Calculate features (will drop ~200 rows due to rolling windows)
print("🔧 Calculating features...")
print("   (This may take 1-2 minutes with 60K candles)")
print()
df_features = calculate_all_features_enhanced_v2(df_raw, phase='phase1')
print()
print(f"   ✅ Features calculated: {len(df_features):,} candles")
print(f"      Features: {len(df_features.columns)} columns")
print(f"      Lost: {len(df_raw) - len(df_features):,} rows (due to rolling windows)")
print()

# Save
print(f"💾 Saving to: {OUTPUT_FILE.name}")
OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
df_features.to_csv(OUTPUT_FILE, index=False)
print(f"   ✅ Saved: {len(df_features):,} rows × {len(df_features.columns)} columns")
print(f"   📁 Path: {OUTPUT_FILE}")
print()

print("="*80)
print("✅ LIVE FEATURES CSV INITIALIZED")
print("="*80)
print()
print("📊 Summary:")
print(f"   Context Size: {len(df_features):,} candles")
print(f"   Features: {len(df_features.columns)} columns")
print(f"   Ready for: Production bot with proper training-like context")
print()
print("🎯 Next Steps:")
print("   1. Update production bot to load this CSV")
print("   2. Bot will append new candles to maintain rolling context")
print("   3. Models will see consistent feature distributions")
print()
