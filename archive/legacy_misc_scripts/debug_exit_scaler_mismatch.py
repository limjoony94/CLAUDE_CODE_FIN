"""
Debug script to diagnose Exit Model scaler mismatch

Compares scaler's expected feature ranges with actual production feature ranges
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models"

# Load LONG Exit Model scaler
scaler_path = MODELS_DIR / "xgboost_v4_long_exit_scaler.pkl"
features_path = MODELS_DIR / "xgboost_v4_long_exit_features.txt"

print("=" * 80)
print("EXIT MODEL SCALER DIAGNOSTIC")
print("=" * 80)

# Load scaler
with open(scaler_path, 'rb') as f:
    scaler = pickle.load(f)

# Load feature names
with open(features_path, 'r') as f:
    feature_names = [line.strip() for line in f.readlines() if line.strip()]

print(f"\nScaler Info:")
print(f"  Type: {type(scaler).__name__}")
print(f"  N features: {scaler.n_features_in_}")
print(f"  Feature range: {scaler.feature_range}")

print(f"\n{'=' * 80}")
print(f"FEATURE RANGES EXPECTED BY SCALER (from training)")
print(f"{'=' * 80}")

print(f"\nFirst 10 features:")
for i in range(min(10, len(feature_names))):
    min_val = scaler.data_min_[i]
    max_val = scaler.data_max_[i]
    range_val = scaler.data_range_[i]
    print(f"{i+1:2d}. {feature_names[i]:30s}: [{min_val:12.6f}, {max_val:12.6f}]  range={range_val:12.6f}")

print(f"\nLast 8 features (position features):")
for i in range(len(feature_names) - 8, len(feature_names)):
    min_val = scaler.data_min_[i]
    max_val = scaler.data_max_[i]
    range_val = scaler.data_range_[i]
    print(f"{i+1:2d}. {feature_names[i]:30s}: [{min_val:12.6f}, {max_val:12.6f}]  range={range_val:12.6f}")

# Test with realistic production values
print(f"\n{'=' * 80}")
print(f"TEST 1: Realistic Production Features")
print(f"{'=' * 80}")

# Simulate realistic base features (36)
# These should match typical values from calculate_features()
base_features_test = np.array([
    0.001,    # close_change_1 (0.1%)
    0.003,    # close_change_3 (0.3%)
    0.5,      # volume_ma_ratio
    55.0,     # rsi
    50.0,     # macd
    0.002,    # signal
    0.01,     # macd_diff
    2.0,      # adx
    0.002,    # cci
    1.0,      # stoch_k
    1.0,      # stoch_d
    -0.001,   # williams_r (normalized to [-1, 1])
    0.01,     # bollinger_width
    0.0,      # price_position
    0.5,      # volume_ratio
    100.0,    # atr
    0.5,      # atr_ratio
    0.01,     # price_range
    0.5,      # momentum_5
    0.5,      # momentum_10
    1.0,      # trend_strength
    50.0,     # resistance_distance
    50.0,     # support_distance
    0.5,      # distance_ratio
    0.5,      # breakout_strength
    110000.0, # current_price
    0.0,      # above_resistance
    0.0,      # below_support
    0.0,      # near_resistance
    0.0,      # near_support
    0.0,      # consolidation
    0.0,      # strong_trend
    0.0,      # reversal_pattern
    0.0,      # breakout_signal
    0.0,      # doji
    0.0,      # engulfing
])

# Position features (8) - realistic values for 3.5h, -0.8% loss position
position_features_test = np.array([
    3.5,      # time_held (hours)
    -0.008,   # current_pnl_pct (-0.8%)
    -0.002,   # pnl_peak (-0.2%)
    -0.008,   # pnl_trough (-0.8%)
    -0.006,   # pnl_from_peak (-0.6% from peak)
    0.02,     # volatility_since_entry (2% std)
    0.1,      # volume_change (10% increase)
    -0.001    # momentum_shift (-0.1%)
])

# Combine
combined_features = np.concatenate([base_features_test, position_features_test]).reshape(1, -1)

print(f"\nInput features (first 10):")
for i in range(min(10, len(feature_names))):
    print(f"  {feature_names[i]:30s}: {combined_features[0][i]:12.6f}")

print(f"\nInput features (last 8 - position):")
for i in range(len(feature_names) - 8, len(feature_names)):
    print(f"  {feature_names[i]:30s}: {combined_features[0][i]:12.6f}")

# Apply scaler
scaled_features = scaler.transform(combined_features)

print(f"\nScaled features (first 10):")
for i in range(min(10, len(feature_names))):
    print(f"  {feature_names[i]:30s}: {scaled_features[0][i]:12.6f}")

print(f"\nScaled features (last 8 - position):")
for i in range(len(feature_names) - 8, len(feature_names)):
    print(f"  {feature_names[i]:30s}: {scaled_features[0][i]:12.6f}")

print(f"\nScaled features range: [{scaled_features.min():.3f}, {scaled_features.max():.3f}]")

# Check if any features are far outside [-1, 1]
outside_range = np.abs(scaled_features) > 1.5
if outside_range.any():
    print(f"\n⚠️ WARNING: {outside_range.sum()} features outside expected [-1, 1] range:")
    indices = np.where(outside_range[0])[0]
    for idx in indices:
        print(f"  {idx+1:2d}. {feature_names[idx]:30s}: {scaled_features[0][idx]:12.6f} (input: {combined_features[0][idx]:12.6f})")

print(f"\n{'=' * 80}")
