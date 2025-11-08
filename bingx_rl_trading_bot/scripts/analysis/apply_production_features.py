"""
Apply Production Feature Pipeline to Downloaded Data
====================================================
Purpose: Use exact same feature calculation as production bot
Steps:
  1. calculate_all_features_enhanced_v2 (production phase1)
  2. prepare_exit_features (exit-specific features)
"""

import sys
from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))

from scripts.experiments.calculate_all_features_enhanced_v2 import calculate_all_features_enhanced_v2
from scripts.experiments.retrain_exit_models_opportunity_gating import prepare_exit_features

RESULTS_DIR = BASE_DIR / "results"

print("=" * 80)
print("APPLY PRODUCTION FEATURE PIPELINE")
print("=" * 80)
print()

# Load raw OHLCV data
print("Loading RAW OHLCV data...")
df = pd.read_csv(RESULTS_DIR / "BTCUSDT_5m_RAW_OHLCV_latest4weeks.csv")
df['timestamp'] = pd.to_datetime(df['timestamp'])
print(f"✓ Loaded: {len(df)} candles, {len(df.columns)} columns (raw OHLCV)")
print(f"  Period: {df['timestamp'].min()} to {df['timestamp'].max()}")
print()

# Step 1: calculate_all_features_enhanced_v2
print("Step 1: calculate_all_features_enhanced_v2 (production phase1)...")
try:
    df_enhanced = calculate_all_features_enhanced_v2(df.copy(), phase='phase1')
    print(f"✓ After phase1: {len(df_enhanced)} candles, {len(df_enhanced.columns)} columns")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# Step 2: prepare_exit_features
print("Step 2: prepare_exit_features (exit-specific features)...")
try:
    df_final = prepare_exit_features(df_enhanced)
    print(f"✓ After exit features: {len(df_final)} candles, {len(df_final.columns)} columns")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print()

# Save
output_file = RESULTS_DIR / "BTCUSDT_5m_latest4weeks_PRODUCTION_FEATURES.csv"
df_final.to_csv(output_file, index=False)
print(f"✓ Saved: {output_file.name}")
print()

# Check Oct 30 exit candle
print("=" * 80)
print("VERIFY OCT 30 EXIT CANDLE")
print("=" * 80)
print()

oct30_exit = df_final[df_final['timestamp'] == '2025-10-30 09:50:00']
if not oct30_exit.empty:
    candle = oct30_exit.iloc[0]
    print(f"Oct 30 09:50:00 (KST 18:50, Exit time):")
    print(f"  Close: ${candle['close']:,.2f}")
    print(f"  RSI: {candle.get('rsi', 'N/A')}")
    print(f"  MACD: {candle.get('macd', 'N/A')}")
    print(f"  Volume Surge: {candle.get('volume_surge', 'N/A')}")
    print(f"  Price Acceleration: {candle.get('price_acceleration', 'N/A')}")
else:
    print("❌ Oct 30 09:50 candle not found!")

print()
print("=" * 80)
print("DONE")
print("=" * 80)
