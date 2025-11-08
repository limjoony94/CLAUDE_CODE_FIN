"""
Debug Oct 30 Trade Replay
=========================
Check why ML Exit (0.755) doesn't trigger in replay
"""

import pandas as pd
import numpy as np
import pickle
import sys
from pathlib import Path
from datetime import datetime

# Paths
BASE_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(BASE_DIR))
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

from scripts.experiments.retrain_exit_models_opportunity_gating import prepare_exit_features

print("=" * 80)
print("DEBUG OCT 30 TRADE REPLAY")
print("=" * 80)
print()

# Load SHORT Exit model (PRODUCTION model used on Oct 30)
print("Loading SHORT Exit model (threshold_075 20251027_190512)...")
with open(MODELS_DIR / "xgboost_short_exit_threshold_075_20251027_190512.pkl", 'rb') as f:
    short_exit_model = pickle.load(f)
with open(MODELS_DIR / "xgboost_short_exit_threshold_075_20251027_190512_features.txt", 'r') as f:
    short_exit_features = [line.strip() for line in f.readlines() if line.strip()]

# Load SHORT Exit scaler
import joblib
print("Loading SHORT Exit scaler (threshold_075 20251027_190512)...")
with open(MODELS_DIR / "xgboost_short_exit_threshold_075_20251027_190512_scaler.pkl", 'rb') as f:
    short_exit_scaler = joblib.load(f)

print(f"✓ SHORT Exit: {len(short_exit_features)} features")
print(f"✓ SHORT Exit Scaler loaded")
print()

# Load market data (with PRODUCTION features)
print("Loading market data with PRODUCTION features...")
df = pd.read_csv(RESULTS_DIR / "BTCUSDT_5m_latest4weeks_PRODUCTION_FEATURES.csv")
df['timestamp'] = pd.to_datetime(df['timestamp'])
print(f"✓ Market data: {len(df):,} candles with PRODUCTION features")
print()

# Oct 30 trade info (KST times)
ENTRY_TIME_KST = "2025-10-30 15:08:55 KST"
EXIT_TIME_KST = "2025-10-30 18:50:10 KST"

# Convert to UTC for data matching
# CRITICAL: Production bot processes COMPLETED candles with 5-min offset
# At 18:50 KST (09:50 UTC), it processes the 18:45 KST (09:45 UTC) candle
ENTRY_TIME = pd.to_datetime("2025-10-30 06:10:00")  # 15:10 KST = 06:10 UTC (using candle close)
ENTRY_PRICE = 110357.7
EXIT_TIME = pd.to_datetime("2025-10-30 09:45:00")   # 18:50 KST processes 09:45 UTC candle (CORRECTED!)
EXIT_PRICE = 110168.2  # Production log: $110,168.2 at 09:45 UTC
SIDE = "SHORT"
LEVERAGE = 4

print("Production Trade Info (KST):")
print(f"  Entry: {ENTRY_TIME_KST} @ ${ENTRY_PRICE:,.1f}")
print(f"  Exit: {EXIT_TIME_KST} @ ${EXIT_PRICE:,.1f}")
print()
print("Converted to UTC for data matching:")
print(f"  Entry: {ENTRY_TIME} @ ${ENTRY_PRICE:,.1f}")
print(f"  Exit: {EXIT_TIME} @ ${EXIT_PRICE:,.1f}")
print(f"  Side: {SIDE}")
print(f"  Actual Exit Reason: ML Exit (0.755)")
print()

# Find entry candle
entry_candles = df[df['timestamp'] >= ENTRY_TIME].head(1)
if entry_candles.empty:
    print("❌ Entry candle not found in data!")
    sys.exit(1)

entry_idx = entry_candles.index[0]
entry_candle = df.iloc[entry_idx]

print(f"Entry Candle Found:")
print(f"  Index: {entry_idx}")
print(f"  Timestamp: {entry_candle['timestamp']}")
print(f"  Close: ${entry_candle['close']:,.1f}")
print()

# Find exit candle
exit_candles = df[df['timestamp'] >= EXIT_TIME].head(1)
if exit_candles.empty:
    print("❌ Exit candle not found in data!")
    sys.exit(1)

exit_idx = exit_candles.index[0]
exit_candle = df.iloc[exit_idx]

print(f"Exit Candle Found:")
print(f"  Index: {exit_idx}")
print(f"  Timestamp: {exit_candle['timestamp']}")
print(f"  Close: ${exit_candle['close']:,.1f}")
print()

# Simulate holding position
print("=" * 80)
print("REPLAY SIMULATION")
print("=" * 80)
print()

EMERGENCY_MAX_HOLD = 120
EXIT_THRESHOLD = 0.75

print(f"Exit Threshold: {EXIT_THRESHOLD}")
print(f"Emergency Max Hold: {EMERGENCY_MAX_HOLD} candles")
print()

found_ml_exit = False
hold_candles = exit_idx - entry_idx

print(f"Total hold time: {hold_candles} candles")
print()

# Check every candle
print("Checking candles for ML Exit signal...")
print()

for i in range(1, min(hold_candles + 1, EMERGENCY_MAX_HOLD + 1)):
    current_idx = entry_idx + i
    if current_idx >= len(df):
        break

    current_candle = df.iloc[current_idx]
    current_price = current_candle['close']
    current_time = current_candle['timestamp']

    # Calculate P&L
    pnl_pct = (ENTRY_PRICE - current_price) / ENTRY_PRICE  # SHORT
    leveraged_pnl_pct = pnl_pct * LEVERAGE

    # Check ML Exit
    try:
        exit_feat = current_candle[short_exit_features].values.reshape(1, -1)
        exit_feat_scaled = short_exit_scaler.transform(exit_feat)  # Apply scaler!
        exit_prob = short_exit_model.predict_proba(exit_feat_scaled)[0, 1]

        # Print ALL candles to see progression (don't stop early)
        if exit_prob >= 0.50 or current_idx >= exit_idx - 5 or i % 5 == 0:
            marker = "🎯" if current_idx == exit_idx else "  "
            print(f"{marker} Candle {i:3d} | {current_time} | Exit Prob: {exit_prob:.4f} | P&L: {leveraged_pnl_pct:+.2%}")

        if exit_prob >= EXIT_THRESHOLD and not found_ml_exit:
            print()
            print(f"✅ FIRST ML EXIT TRIGGER!")
            print(f"   Candle: {i}")
            print(f"   Time: {current_time}")
            print(f"   Exit Prob: {exit_prob:.4f}")
            print(f"   P&L: {leveraged_pnl_pct:+.2%}")
            print()
            print("   Continuing to check all candles...")
            print()
            found_ml_exit = True
            # Don't break! Continue to see all probabilities

    except KeyError as e:
        print(f"❌ Missing features at candle {i}: {e}")
        break

print()
if not found_ml_exit:
    print("❌ ML Exit NOT triggered in replay")
    print(f"   Max hold reached: {EMERGENCY_MAX_HOLD} candles")
    print()
    print("Checking exit candle features directly...")
    print()

    # Check exit candle directly
    try:
        exit_feat = exit_candle[short_exit_features].values.reshape(1, -1)
        exit_feat_scaled = short_exit_scaler.transform(exit_feat)  # Apply scaler!
        exit_prob = short_exit_model.predict_proba(exit_feat_scaled)[0, 1]
        print(f"Exit candle probability: {exit_prob:.4f}")
        print(f"Expected: 0.755 (from production logs)")
        print(f"Difference: {abs(exit_prob - 0.755):.4f}")
    except Exception as e:
        print(f"Error checking exit candle: {e}")
else:
    print("✅ ML Exit successfully replayed")

print()
print("=" * 80)
