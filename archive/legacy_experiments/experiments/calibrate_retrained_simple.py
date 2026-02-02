"""
Simplified Probability Calibration - Retrained Entry Models
===========================================================

Approach: Use holdout period for calibration
  Split: 11 days calibration, 3 days final validation
  Labels: Generated via trade-outcome simulation (same as training)
  Method: Isotonic Regression via CalibratedClassifierCV

Why Simplified:
  - Walk-Forward Decoupled creates labels on-the-fly (not saved)
  - Full calibration requires re-running trade simulations
  - Using holdout avoids regenerating training labels

Created: 2025-10-29
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from sklearn.calibration import CalibratedClassifierCV

# Configuration
LEVERAGE = 4
EMERGENCY_STOP_LOSS = -0.03
EMERGENCY_MAX_HOLD = 120
ML_EXIT_THRESHOLD = 0.75
CALIBRATION_DAYS = 11
VALIDATION_DAYS = 3

DATA_DIR = PROJECT_ROOT / "data" / "features"
MODELS_DIR = PROJECT_ROOT / "models"

print("="*80)
print("SIMPLIFIED PROBABILITY CALIBRATION")
print("="*80)
print()
print(f"Method: Isotonic Regression on {CALIBRATION_DAYS}-day calibration set")
print(f"Validation: {VALIDATION_DAYS}-day holdout for final testing")
print()

# Load data
print("-"*80)
print("Loading Data")
print("-"*80)

df = pd.read_csv(DATA_DIR / "BTCUSDT_5m_features.csv")
print(f"✅ Loaded {len(df):,} candles")

# Prepare exit features (needed for trade simulation)
def prepare_exit_features(df):
    df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
    df['volume_surge'] = (df['volume'] > df['volume'].rolling(20).mean() * 1.5).astype(float)
    if 'ma_50' in df.columns:
        df['price_vs_ma20'] = (df['close'] - df['ma_50']) / df['ma_50']
    else:
        ma_20 = df['close'].rolling(20).mean()
        df['price_vs_ma20'] = (df['close'] - ma_20) / ma_20
    if 'ma_200' in df.columns:
        df['price_vs_ma50'] = (df['close'] - df['ma_200']) / df['ma_200']
    else:
        ma_50 = df['close'].rolling(50).mean()
        df['price_vs_ma50'] = (df['close'] - ma_50) / ma_50
    df['returns'] = df['close'].pct_change()
    df['volatility_20'] = df['returns'].rolling(20).std()
    df['volatility_regime'] = (df['volatility_20'] > df['volatility_20'].rolling(100).median()).astype(float)
    df['rsi_slope'] = df['rsi'].diff(3) / 3
    df['rsi_overbought'] = (df['rsi'] > 70).astype(float)
    df['rsi_oversold'] = (df['rsi'] < 30).astype(float)
    df['rsi_divergence'] = 0
    df['macd_histogram_slope'] = df['macd_hist'].diff(3) / 3 if 'macd_hist' in df.columns else 0
    df['macd_crossover'] = ((df['macd'] > df['macd_signal']) &
                            (df['macd'].shift(1) <= df['macd_signal'].shift(1))).astype(float)
    df['macd_crossunder'] = ((df['macd'] < df['macd_signal']) &
                             (df['macd'].shift(1) >= df['macd_signal'].shift(1))).astype(float)
    df['higher_high'] = (df['high'] > df['high'].shift(1)).astype(float)
    df['lower_low'] = (df['low'] < df['low'].shift(1)).astype(float)
    df['price_acceleration'] = df['close'].diff().diff()
    df['near_resistance'] = 0
    df['near_support'] = 0
    if 'bb_upper' in df.columns and 'bb_lower' in df.columns:
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    else:
        df['bb_position'] = 0.5
    df = df.ffill().bfill()
    return df

df = prepare_exit_features(df)

# Split holdout into calibration + validation
total_holdout_candles = (CALIBRATION_DAYS + VALIDATION_DAYS) * 24 * 12
calib_candles = CALIBRATION_DAYS * 24 * 12
valid_candles = VALIDATION_DAYS * 24 * 12

calib_start_idx = len(df) - total_holdout_candles
calib_end_idx = calib_start_idx + calib_candles
valid_start_idx = calib_end_idx

df_calib = df.iloc[calib_start_idx:calib_end_idx].copy().reset_index(drop=True)
df_valid = df.iloc[valid_start_idx:].copy().reset_index(drop=True)

print(f"✅ Calibration: {len(df_calib):,} candles ({CALIBRATION_DAYS} days)")
print(f"✅ Validation: {len(df_valid):,} candles ({VALIDATION_DAYS} days)")
print()

# Load Exit models (for trade simulation)
print("-"*80)
print("Loading Exit Models")
print("-"*80)

long_exit_model = joblib.load(MODELS_DIR / "xgboost_long_exit_threshold_075_20251027_190512.pkl")
long_exit_scaler = joblib.load(MODELS_DIR / "xgboost_long_exit_threshold_075_20251027_190512_scaler.pkl")
with open(MODELS_DIR / "xgboost_long_exit_threshold_075_20251027_190512_features.txt", 'r') as f:
    long_exit_features = [line.strip() for line in f if line.strip()]

short_exit_model = joblib.load(MODELS_DIR / "xgboost_short_exit_threshold_075_20251027_190512.pkl")
short_exit_scaler = joblib.load(MODELS_DIR / "xgboost_short_exit_threshold_075_20251027_190512_scaler.pkl")
with open(MODELS_DIR / "xgboost_short_exit_threshold_075_20251027_190512_features.txt", 'r') as f:
    short_exit_features = [line.strip() for line in f if line.strip()]

print(f"✅ LONG Exit: {len(long_exit_features)} features")
print(f"✅ SHORT Exit: {len(short_exit_features)} features")
print()

# Load Entry models
print("-"*80)
print("Loading Entry Models")
print("-"*80)

long_entry_model = joblib.load(MODELS_DIR / "xgboost_long_entry_retrained_latest_20251029_081454.pkl")
long_entry_scaler = joblib.load(MODELS_DIR / "xgboost_long_entry_retrained_latest_20251029_081454_scaler.pkl")
with open(MODELS_DIR / "xgboost_long_entry_retrained_latest_20251029_081454_features.txt", 'r') as f:
    long_entry_features = [line.strip() for line in f if line.strip()]

short_entry_model = joblib.load(MODELS_DIR / "xgboost_short_entry_retrained_latest_20251029_081454.pkl")
short_entry_scaler = joblib.load(MODELS_DIR / "xgboost_short_entry_retrained_latest_20251029_081454_scaler.pkl")
with open(MODELS_DIR / "xgboost_short_entry_retrained_latest_20251029_081454_features.txt", 'r') as f:
    short_entry_features = [line.strip() for line in f if line.strip()]

print(f"✅ LONG Entry: {len(long_entry_features)} features")
print(f"✅ SHORT Entry: {len(short_entry_features)} features")
print()

# Generate calibration labels via trade simulation
print("-"*80)
print("Generating Calibration Labels (Trade Simulation)")
print("-"*80)

def simulate_trade(df, entry_idx, side, exit_model, exit_scaler, exit_features):
    """Simulate trade outcome"""
    entry_price = df['close'].iloc[entry_idx]

    max_idx = min(entry_idx + EMERGENCY_MAX_HOLD + 1, len(df))

    for i in range(1, max_idx - entry_idx):
        current_idx = entry_idx + i
        current_price = df['close'].iloc[current_idx]

        if side == 'LONG':
            pnl_pct = (current_price - entry_price) / entry_price
        else:
            pnl_pct = (entry_price - current_price) / entry_price

        leveraged_pnl = pnl_pct * LEVERAGE

        # Stop Loss
        if leveraged_pnl <= EMERGENCY_STOP_LOSS:
            return leveraged_pnl, i, 'SL'

        # ML Exit
        try:
            X_exit = df[exit_features].iloc[current_idx:current_idx+1].values
            if not np.isnan(X_exit).any():
                X_exit_scaled = exit_scaler.transform(X_exit)
                exit_prob = exit_model.predict_proba(X_exit_scaled)[0][1]

                if exit_prob >= ML_EXIT_THRESHOLD:
                    return leveraged_pnl, i, 'ML'
        except:
            pass

    # Max Hold
    final_price = df['close'].iloc[max_idx-1]
    if side == 'LONG':
        pnl_pct = (final_price - entry_price) / entry_price
    else:
        pnl_pct = (entry_price - final_price) / entry_price
    leveraged_pnl = pnl_pct * LEVERAGE

    return leveraged_pnl, EMERGENCY_MAX_HOLD, 'MAX'

# LONG calibration labels
long_labels = []
print("⏳ Simulating LONG trades...")
for i in range(100, len(df_calib) - EMERGENCY_MAX_HOLD):
    pnl, hold_time, exit_reason = simulate_trade(
        df_calib, i, 'LONG', long_exit_model, long_exit_scaler, long_exit_features
    )
    # Label: 1 if leveraged_pnl > 0.02, else 0
    long_labels.append(1 if pnl > 0.02 else 0)

df_calib_long = df_calib.iloc[100:100+len(long_labels)].copy()
df_calib_long['label'] = long_labels

print(f"✅ LONG labels: {len(long_labels):,}")
print(f"   Positive rate: {np.mean(long_labels)*100:.2f}%")

# SHORT calibration labels
short_labels = []
print("⏳ Simulating SHORT trades...")
for i in range(100, len(df_calib) - EMERGENCY_MAX_HOLD):
    pnl, hold_time, exit_reason = simulate_trade(
        df_calib, i, 'SHORT', short_exit_model, short_exit_scaler, short_exit_features
    )
    short_labels.append(1 if pnl > 0.02 else 0)

df_calib_short = df_calib.iloc[100:100+len(short_labels)].copy()
df_calib_short['label'] = short_labels

print(f"✅ SHORT labels: {len(short_labels):,}")
print(f"   Positive rate: {np.mean(short_labels)*100:.2f}%")
print()

# Calibrate LONG model
print("-"*80)
print("Calibrating LONG Entry Model")
print("-"*80)

X_calib_long = df_calib_long[long_entry_features].values
y_calib_long = df_calib_long['label'].values
X_calib_long_scaled = long_entry_scaler.transform(X_calib_long)

probs_before = long_entry_model.predict_proba(X_calib_long_scaled)[:, 1]
print(f"Before calibration: mean {probs_before.mean()*100:.2f}%, median {np.median(probs_before)*100:.2f}%")

calibrated_long = CalibratedClassifierCV(
    estimator=long_entry_model,
    method='isotonic',
    cv='prefit'
)
calibrated_long.fit(X_calib_long_scaled, y_calib_long)

probs_after = calibrated_long.predict_proba(X_calib_long_scaled)[:, 1]
print(f"After calibration:  mean {probs_after.mean()*100:.2f}%, median {np.median(probs_after)*100:.2f}%")
print(f"Actual positive rate: {y_calib_long.mean()*100:.2f}%")
print()

# Calibrate SHORT model
print("-"*80)
print("Calibrating SHORT Entry Model")
print("-"*80)

X_calib_short = df_calib_short[short_entry_features].values
y_calib_short = df_calib_short['label'].values
X_calib_short_scaled = short_entry_scaler.transform(X_calib_short)

probs_before = short_entry_model.predict_proba(X_calib_short_scaled)[:, 1]
print(f"Before calibration: mean {probs_before.mean()*100:.2f}%, median {np.median(probs_before)*100:.2f}%")

calibrated_short = CalibratedClassifierCV(
    estimator=short_entry_model,
    method='isotonic',
    cv='prefit'
)
calibrated_short.fit(X_calib_short_scaled, y_calib_short)

probs_after = calibrated_short.predict_proba(X_calib_short_scaled)[:, 1]
print(f"After calibration:  mean {probs_after.mean()*100:.2f}%, median {np.median(probs_after)*100:.2f}%")
print(f"Actual positive rate: {y_calib_short.mean()*100:.2f}%")
print()

# Save calibrated models
print("-"*80)
print("Saving Calibrated Models")
print("-"*80)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# LONG
long_path = MODELS_DIR / f"xgboost_long_entry_calibrated_{timestamp}.pkl"
joblib.dump(calibrated_long, long_path)
joblib.dump(long_entry_scaler, MODELS_DIR / f"xgboost_long_entry_calibrated_{timestamp}_scaler.pkl")
with open(MODELS_DIR / f"xgboost_long_entry_calibrated_{timestamp}_features.txt", 'w') as f:
    f.write('\n'.join(long_entry_features))

print(f"✅ LONG: {long_path.name}")

# SHORT
short_path = MODELS_DIR / f"xgboost_short_entry_calibrated_{timestamp}.pkl"
joblib.dump(calibrated_short, short_path)
joblib.dump(short_entry_scaler, MODELS_DIR / f"xgboost_short_entry_calibrated_{timestamp}_scaler.pkl")
with open(MODELS_DIR / f"xgboost_short_entry_calibrated_{timestamp}_features.txt", 'w') as f:
    f.write('\n'.join(short_entry_features))

print(f"✅ SHORT: {short_path.name}")
print()

print("="*80)
print("CALIBRATION COMPLETE")
print("="*80)
print()
print(f"Timestamp: {timestamp}")
print(f"Next: Backtest on {VALIDATION_DAYS}-day validation set with threshold 0.75")
print()
