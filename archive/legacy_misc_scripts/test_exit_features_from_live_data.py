"""
Test Exit Model scaler with ACTUAL current market features
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import ccxt

# Project setup
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.production.train_xgboost_improved_v3_phase2 import calculate_features
from scripts.production.advanced_technical_features import AdvancedTechnicalFeatures

MODELS_DIR = PROJECT_ROOT / "models"

print("=" * 80)
print("TEST EXIT MODEL WITH LIVE MARKET DATA")
print("=" * 80)

# 1. Fetch current market data
print("\nFetching current 5m candles from BingX...")
exchange = ccxt.bingx({
    'enableRateLimit': True,
})

try:
    # Fetch last 200 candles (for feature calculation)
    ohlcv = exchange.fetch_ohlcv('BTC/USDT', '5m', limit=200)
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    print(f"✅ Fetched {len(df)} candles")
    print(f"   Latest candle time: {pd.to_datetime(df['timestamp'].iloc[-1], unit='ms')}")
    print(f"   Current price: ${df['close'].iloc[-1]:,.2f}")
except Exception as e:
    print(f"❌ Error fetching data: {e}")
    sys.exit(1)

# 2. Calculate features (same as training)
print("\nCalculating features...")
df = calculate_features(df)
adv_features = AdvancedTechnicalFeatures(lookback_sr=50, lookback_trend=20)
df = adv_features.calculate_all_features(df)
df = df.ffill().dropna()
print(f"✅ Features calculated: {len(df)} rows remaining")

# 3. Load Exit model components
print("\nLoading Exit model components...")
exit_model_path = MODELS_DIR / "xgboost_v4_long_exit.pkl"
exit_scaler_path = MODELS_DIR / "xgboost_v4_long_exit_scaler.pkl"
exit_features_path = MODELS_DIR / "xgboost_v4_long_exit_features.txt"

with open(exit_model_path, 'rb') as f:
    exit_model = pickle.load(f)

with open(exit_scaler_path, 'rb') as f:
    exit_scaler = pickle.load(f)

with open(exit_features_path, 'r') as f:
    exit_feature_columns = [line.strip() for line in f.readlines() if line.strip()]

print(f"✅ Model loaded: {exit_model.n_features_in_} features")
print(f"✅ Scaler loaded: {exit_scaler.n_features_in_} features, range {exit_scaler.feature_range}")

# 4. Extract base features from current data
exit_base_features = [f for f in exit_feature_columns if f not in [
    'time_held', 'current_pnl_pct', 'pnl_peak', 'pnl_trough',
    'pnl_from_peak', 'volatility_since_entry', 'volume_change', 'momentum_shift'
]]

print(f"\nBase features to extract: {len(exit_base_features)}")

# Get latest candle features
current_idx = len(df) - 1
base_features_values = df[exit_base_features].iloc[current_idx].values

print(f"\n{'=' * 80}")
print(f"ACTUAL BASE FEATURES FROM LIVE DATA")
print(f"{'=' * 80}")

print(f"\nFirst 10 base features:")
for i in range(min(10, len(exit_base_features))):
    print(f"  {exit_base_features[i]:30s}: {base_features_values[i]:12.6f}")

print(f"\nLast 5 base features:")
for i in range(len(exit_base_features) - 5, len(exit_base_features)):
    print(f"  {exit_base_features[i]:30s}: {base_features_values[i]:12.6f}")

# 5. Simulate position features (matching current open position)
print(f"\n{'=' * 80}")
print(f"SIMULATED POSITION FEATURES (current open position)")
print(f"{'=' * 80}")

# From state.json: LONG 0.6389 BTC @ $111,908.50, held 3.5h, -0.80% loss
position_features = np.array([
    3.5,      # time_held (hours)
    -0.008,   # current_pnl_pct (-0.8%)
    -0.002,   # pnl_peak (-0.2% - assuming small initial profit)
    -0.008,   # pnl_trough (-0.8% - current is worst)
    -0.006,   # pnl_from_peak (-0.6% from peak)
    0.02,     # volatility_since_entry (2% std)
    0.1,      # volume_change (10% increase)
    -0.001    # momentum_shift (-0.1%)
])

print(f"\nPosition features:")
position_feature_names = ['time_held', 'current_pnl_pct', 'pnl_peak', 'pnl_trough',
                          'pnl_from_peak', 'volatility_since_entry', 'volume_change', 'momentum_shift']
for i, (name, val) in enumerate(zip(position_feature_names, position_features)):
    print(f"  {name:30s}: {val:12.6f}")

# 6. Combine and scale
combined_features = np.concatenate([base_features_values, position_features]).reshape(1, -1)

print(f"\n{'=' * 80}")
print(f"SCALING TEST")
print(f"{'=' * 80}")

print(f"\nCombined features shape: {combined_features.shape}")
print(f"Expected by model: {exit_model.n_features_in_} features")
print(f"Expected by scaler: {exit_scaler.n_features_in_} features")

if combined_features.shape[1] != exit_model.n_features_in_:
    print(f"\n⚠️ WARNING: Feature count mismatch!")
    sys.exit(1)

# Apply scaler
scaled_features = exit_scaler.transform(combined_features)

print(f"\nScaled features range: [{scaled_features.min():.3f}, {scaled_features.max():.3f}]")

# Check for outliers
outliers = np.abs(scaled_features) > 2.0
if outliers.any():
    print(f"\n⚠️ {outliers.sum()} features scaled outside [-2, 2]:")
    indices = np.where(outliers[0])[0]
    for idx in indices[:10]:  # Show first 10 outliers
        feat_name = exit_feature_columns[idx]
        original_val = combined_features[0][idx]
        scaled_val = scaled_features[0][idx]
        expected_min = exit_scaler.data_min_[idx]
        expected_max = exit_scaler.data_max_[idx]
        print(f"  {idx+1:2d}. {feat_name:30s}: {original_val:12.2f} → {scaled_val:12.2f} (expected range: [{expected_min:.2f}, {expected_max:.2f}])")

# 7. Get exit prediction
exit_prob = exit_model.predict_proba(scaled_features)[0][1]

print(f"\n{'=' * 80}")
print(f"EXIT MODEL PREDICTION")
print(f"{'=' * 80}")

print(f"\nExit Probability: {exit_prob:.6f}")
print(f"Threshold: 0.603")
print(f"Signal: {'✅ EXIT' if exit_prob >= 0.603 else '❌ HOLD'}")

print(f"\n{'=' * 80}")
