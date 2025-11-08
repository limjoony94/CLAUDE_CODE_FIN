"""
Retrain Entry Models: Threshold 0.75 (LONG + SHORT)
====================================================

Train Entry models with:
1. Latest Exit models (2025-10-24)
2. ML Exit threshold: 0.75
3. Entry threshold: 0.75
4. Updated feature dataset (33,728 candles)
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import pickle
import joblib
from datetime import datetime
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report

# Use the optimized trade simulator
from scripts.experiments.trade_simulator_optimized import simulate_trades_optimized
from scripts.experiments.retrain_exit_models_opportunity_gating import prepare_exit_features

MODELS_DIR = PROJECT_ROOT / "models"
FEATURES_FILE = PROJECT_ROOT / "data" / "features" / "BTCUSDT_5m_features.csv"

print("="*80)
print("RETRAIN ENTRY MODELS: THRESHOLD 0.75 (LONG + SHORT)")
print("="*80)
print("\nConfiguration:")
print("  Entry Threshold: 0.75")
print("  ML Exit Threshold: 0.75")
print("  Exit Models: 2025-10-24 (latest)")
print("  Dataset: 33,728 candles")

# ============================================================================
# STEP 1: Load Data with Features
# ============================================================================

print("\n" + "-"*80)
print("STEP 1: Loading Feature Dataset")
print("-"*80)

df = pd.read_csv(FEATURES_FILE)
print(f"\n✅ Loaded {len(df):,} candles with {len(df.columns)} features")
print(f"   Date range: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")

# Prepare exit features (if not already present)
print("\nPreparing exit features...")
df = prepare_exit_features(df)
print(f"✅ Exit features ready ({len(df.columns)} total columns)")

# ============================================================================
# STEP 2: Load Latest Exit Models (2025-10-24)
# ============================================================================

print("\n" + "-"*80)
print("STEP 2: Loading Latest Exit Models (2025-10-24)")
print("-"*80)

# LONG Exit (NEW - Trained 2025-10-27 with Full 165 features)
long_exit_model_path = MODELS_DIR / "xgboost_long_exit_threshold_075_20251027_190512.pkl"
long_exit_scaler_path = MODELS_DIR / "xgboost_long_exit_threshold_075_20251027_190512_scaler.pkl"
long_exit_features_path = MODELS_DIR / "xgboost_long_exit_threshold_075_20251027_190512_features.txt"

with open(long_exit_model_path, 'rb') as f:
    long_exit_model = pickle.load(f)
long_exit_scaler = joblib.load(long_exit_scaler_path)
with open(long_exit_features_path, 'r') as f:
    long_exit_features = [line.strip() for line in f.readlines() if line.strip()]

print(f"  ✅ LONG Exit: {len(long_exit_features)} features")

# SHORT Exit (NEW - Trained 2025-10-27 with Full 165 features)
short_exit_model_path = MODELS_DIR / "xgboost_short_exit_threshold_075_20251027_190512.pkl"
short_exit_scaler_path = MODELS_DIR / "xgboost_short_exit_threshold_075_20251027_190512_scaler.pkl"
short_exit_features_path = MODELS_DIR / "xgboost_short_exit_threshold_075_20251027_190512_features.txt"

with open(short_exit_model_path, 'rb') as f:
    short_exit_model = pickle.load(f)
short_exit_scaler = joblib.load(short_exit_scaler_path)
with open(short_exit_features_path, 'r') as f:
    short_exit_features = [line.strip() for line in f.readlines() if line.strip()]

print(f"  ✅ SHORT Exit: {len(short_exit_features)} features")

# ============================================================================
# STEP 3: Trade-Outcome Labeling (Threshold 0.75)
# ============================================================================

print("\n" + "-"*80)
print("STEP 3: Trade-Outcome Labeling (ML Exit Threshold 0.75)")
print("-"*80)

# Criteria for positive label
PROFIT_THRESHOLD = 0.02  # 2% leveraged PnL
MAE_THRESHOLD = -0.04  # -4% MAE
MFE_THRESHOLD = 0.02  # 2% MFE
SCORING_THRESHOLD = 2  # 2 of 3 criteria

def evaluate_trade_quality(trade_result):
    """2-of-3 criteria scoring"""
    score = 0

    # Criterion 1: Profitable (>=2%)
    if trade_result['leveraged_pnl_pct'] >= PROFIT_THRESHOLD:
        score += 1

    # Criterion 2: Good Risk-Reward
    if (trade_result['mae'] >= MAE_THRESHOLD and
        trade_result['mfe'] >= MFE_THRESHOLD):
        score += 1

    # Criterion 3: ML Exit
    if trade_result['exit_reason'] == 'ml_exit':
        score += 1

    return score

# LONG Labeling
print("\n🔄 LONG Entry Labeling (Trade-Outcome Based)...")
import time
start_time = time.time()

long_trade_results = simulate_trades_optimized(
    df=df,
    exit_model=long_exit_model,
    exit_scaler=long_exit_scaler,
    exit_features=long_exit_features,
    side='LONG',
    ml_exit_threshold=0.75,  # ← THRESHOLD 0.75
    n_workers=4
)

# Create labels
long_labels = np.zeros(len(df))
for result in long_trade_results:
    if result is not None:
        entry_idx = result['entry_idx']
        score = evaluate_trade_quality(result)

        if score >= SCORING_THRESHOLD:
            long_labels[entry_idx] = 1

long_time = time.time() - start_time
print(f"  ✅ LONG completed in {long_time:.1f}s ({len(long_trade_results)} trades)")
print(f"     Positive labels: {long_labels.sum():,.0f} ({long_labels.mean()*100:.1f}%)")

# SHORT Labeling
print("\n🔄 SHORT Entry Labeling (Trade-Outcome Based)...")
start_time = time.time()

short_trade_results = simulate_trades_optimized(
    df=df,
    exit_model=short_exit_model,
    exit_scaler=short_exit_scaler,
    exit_features=short_exit_features,
    side='SHORT',
    ml_exit_threshold=0.75,  # ← THRESHOLD 0.75
    n_workers=4
)

# Create labels
short_labels = np.zeros(len(df))
for result in short_trade_results:
    if result is not None:
        entry_idx = result['entry_idx']
        score = evaluate_trade_quality(result)

        if score >= SCORING_THRESHOLD:
            short_labels[entry_idx] = 1

short_time = time.time() - start_time
print(f"  ✅ SHORT completed in {short_time:.1f}s ({len(short_trade_results)} trades)")
print(f"     Positive labels: {short_labels.sum():,.0f} ({short_labels.mean()*100:.1f}%)")

# ============================================================================
# STEP 4: Train LONG Entry Model
# ============================================================================

print("\n" + "-"*80)
print("STEP 4: Training LONG Entry Model")
print("-"*80)

# LONG feature columns (from Enhanced Entry model - 85 features)
LONG_FEATURE_COLUMNS = [
    'close_change_1', 'close_change_3', 'volume_ma_ratio', 'rsi', 'macd', 'macd_signal', 'macd_diff',
    'bb_high', 'bb_mid', 'bb_low', 'distance_to_support_pct', 'distance_to_resistance_pct',
    'num_support_touches', 'num_resistance_touches', 'upper_trendline_slope', 'lower_trendline_slope',
    'price_vs_upper_trendline_pct', 'price_vs_lower_trendline_pct', 'rsi_bullish_divergence',
    'rsi_bearish_divergence', 'macd_bullish_divergence', 'macd_bearish_divergence', 'double_top',
    'double_bottom', 'higher_highs_lows', 'lower_highs_lows', 'volume_ma_ratio', 'volume_price_correlation',
    'price_volume_trend', 'body_to_range_ratio', 'upper_shadow_ratio', 'lower_shadow_ratio',
    'bullish_engulfing', 'bearish_engulfing', 'hammer', 'shooting_star', 'doji',
    'distance_from_recent_high_pct', 'bearish_candle_count', 'red_candle_volume_ratio',
    'strong_selling_pressure', 'price_momentum_near_resistance', 'rsi_from_recent_peak',
    'consecutive_up_candles', 'ma_200', 'ema_200', 'rsi_200', 'atr_200',
    'vp_poc', 'vp_value_area_high', 'vp_value_area_low', 'vp_distance_to_poc_pct', 'vp_in_value_area',
    'vp_percentile', 'vp_volume_imbalance', 'vwap', 'vwap_distance_pct', 'vwap_above',
    'vwap_band_position', 'vp_value_area_width_pct', 'vp_price_in_va_position', 'vp_poc_momentum',
    'vp_va_midpoint_momentum', 'vp_strong_buy_pressure', 'vp_strong_sell_pressure',
    'vp_poc_distance_normalized', 'vp_above_va_breakout', 'vp_below_va_breakout', 'vwap_momentum',
    'vwap_vs_ma20', 'vwap_distance_normalized', 'vwap_cross_up', 'vwap_cross_down', 'vwap_overbought',
    'vwap_oversold', 'vwap_bullish_divergence', 'vwap_bearish_divergence', 'vp_vwap_bullish_confluence',
    'vp_vwap_bearish_confluence', 'vp_vwap_alignment', 'vwap_near_vp_support', 'vwap_near_vp_resistance',
    'institutional_activity_zone', 'vp_vwap_trend_alignment', 'vp_efficiency'
]

# Verify features
missing_long = [f for f in LONG_FEATURE_COLUMNS if f not in df.columns]
if missing_long:
    print(f"\n⚠️  Missing {len(missing_long)} LONG features!")
    print(f"   First 10: {missing_long[:10]}")
    print("\n🔄 Using available features only...")
    LONG_FEATURE_COLUMNS = [f for f in LONG_FEATURE_COLUMNS if f in df.columns]

print(f"\n  Using {len(LONG_FEATURE_COLUMNS)} features for LONG model")

# Prepare data
X_long = df[LONG_FEATURE_COLUMNS].fillna(0)
y_long = long_labels

# Train/Test split (80/20)
split_idx = int(len(X_long) * 0.8)
X_long_train, X_long_test = X_long[:split_idx], X_long[split_idx:]
y_long_train, y_long_test = y_long[:split_idx], y_long[split_idx:]

# Scale
long_scaler = StandardScaler()
X_long_train_scaled = long_scaler.fit_transform(X_long_train)
X_long_test_scaled = long_scaler.transform(X_long_test)

# Train XGBoost
print("\n🔄 Training LONG Entry model...")
long_model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    min_child_weight=3,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0.1,
    random_state=42,
    n_jobs=-1,
    scale_pos_weight=(y_long_train == 0).sum() / (y_long_train == 1).sum()  # Handle imbalance
)

long_model.fit(X_long_train_scaled, y_long_train)
print("  ✅ LONG model trained")

# Evaluate
y_long_pred = long_model.predict(X_long_test_scaled)
print("\n" + classification_report(y_long_test, y_long_pred, zero_division=0))

# Save
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
long_model_file = MODELS_DIR / f"xgboost_long_entry_threshold080_{timestamp}.pkl"
long_scaler_file = MODELS_DIR / f"xgboost_long_entry_threshold080_{timestamp}_scaler.pkl"
long_features_file = MODELS_DIR / f"xgboost_long_entry_threshold080_{timestamp}_features.txt"

with open(long_model_file, 'wb') as f:
    pickle.dump(long_model, f)
joblib.dump(long_scaler, long_scaler_file)
with open(long_features_file, 'w') as f:
    f.write('\n'.join(LONG_FEATURE_COLUMNS))

print(f"\n💾 LONG model saved:")
print(f"   Model: {long_model_file.name}")
print(f"   Scaler: {long_scaler_file.name}")
print(f"   Features: {long_features_file.name}")

# ============================================================================
# STEP 5: Train SHORT Entry Model
# ============================================================================

print("\n" + "-"*80)
print("STEP 5: Training SHORT Entry Model")
print("-"*80)

# SHORT feature columns (from Enhanced Entry model - 79 features)
SHORT_FEATURE_COLUMNS = [
    'rsi_deviation', 'rsi_direction', 'rsi_extreme', 'macd_strength', 'macd_direction', 'macd_divergence_abs',
    'price_distance_ma20', 'price_direction_ma20', 'price_distance_ma50', 'price_direction_ma50',
    'volatility', 'atr_pct', 'atr', 'negative_momentum', 'negative_acceleration', 'down_candle_ratio',
    'down_candle_body', 'lower_low_streak', 'resistance_rejection_count', 'bearish_divergence',
    'volume_decline_ratio', 'distribution_signal', 'down_candle', 'lower_low', 'near_resistance',
    'rejection_from_resistance', 'volume_on_decline', 'volume_on_advance', 'bear_market_strength',
    'trend_strength', 'downtrend_confirmed', 'volatility_asymmetry', 'below_support', 'support_breakdown',
    'panic_selling', 'downside_volatility', 'upside_volatility', 'ema_12',
    'ma_200', 'ema_200', 'rsi_200', 'atr_200',
    'vp_poc', 'vp_value_area_high', 'vp_value_area_low', 'vp_distance_to_poc_pct', 'vp_in_value_area',
    'vp_percentile', 'vp_volume_imbalance', 'vwap', 'vwap_distance_pct', 'vwap_above',
    'vwap_band_position', 'vp_value_area_width_pct', 'vp_price_in_va_position', 'vp_poc_momentum',
    'vp_va_midpoint_momentum', 'vp_strong_buy_pressure', 'vp_strong_sell_pressure',
    'vp_poc_distance_normalized', 'vp_above_va_breakout', 'vp_below_va_breakout', 'vwap_momentum',
    'vwap_vs_ma20', 'vwap_distance_normalized', 'vwap_cross_up', 'vwap_cross_down', 'vwap_overbought',
    'vwap_oversold', 'vwap_bullish_divergence', 'vwap_bearish_divergence', 'vp_vwap_bullish_confluence',
    'vp_vwap_bearish_confluence', 'vp_vwap_alignment', 'vwap_near_vp_support', 'vwap_near_vp_resistance',
    'institutional_activity_zone', 'vp_vwap_trend_alignment', 'vp_efficiency'
]

# Verify features
missing_short = [f for f in SHORT_FEATURE_COLUMNS if f not in df.columns]
if missing_short:
    print(f"\n⚠️  Missing {len(missing_short)} SHORT features!")
    print(f"   First 10: {missing_short[:10]}")
    print("\n🔄 Using available features only...")
    SHORT_FEATURE_COLUMNS = [f for f in SHORT_FEATURE_COLUMNS if f in df.columns]

print(f"\n  Using {len(SHORT_FEATURE_COLUMNS)} features for SHORT model")

# Prepare data
X_short = df[SHORT_FEATURE_COLUMNS].fillna(0)
y_short = short_labels

# Train/Test split
X_short_train, X_short_test = X_short[:split_idx], X_short[split_idx:]
y_short_train, y_short_test = y_short[:split_idx], y_short[split_idx:]

# Scale
short_scaler = StandardScaler()
X_short_train_scaled = short_scaler.fit_transform(X_short_train)
X_short_test_scaled = short_scaler.transform(X_short_test)

# Train XGBoost
print("\n🔄 Training SHORT Entry model...")
short_model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    min_child_weight=3,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0.1,
    random_state=42,
    n_jobs=-1,
    scale_pos_weight=(y_short_train == 0).sum() / (y_short_train == 1).sum()
)

short_model.fit(X_short_train_scaled, y_short_train)
print("  ✅ SHORT model trained")

# Evaluate
y_short_pred = short_model.predict(X_short_test_scaled)
print("\n" + classification_report(y_short_test, y_short_pred, zero_division=0))

# Save
short_model_file = MODELS_DIR / f"xgboost_short_entry_threshold080_{timestamp}.pkl"
short_scaler_file = MODELS_DIR / f"xgboost_short_entry_threshold080_{timestamp}_scaler.pkl"
short_features_file = MODELS_DIR / f"xgboost_short_entry_threshold080_{timestamp}_features.txt"

with open(short_model_file, 'wb') as f:
    pickle.dump(short_model, f)
joblib.dump(short_scaler, short_scaler_file)
with open(short_features_file, 'w') as f:
    f.write('\n'.join(SHORT_FEATURE_COLUMNS))

print(f"\n💾 SHORT model saved:")
print(f"   Model: {short_model_file.name}")
print(f"   Scaler: {short_scaler_file.name}")
print(f"   Features: {short_features_file.name}")

# ============================================================================
# COMPLETE
# ============================================================================

print("\n" + "="*80)
print("✅ ENTRY MODEL TRAINING COMPLETE (THRESHOLD 0.75)")
print("="*80)
print(f"\nModels trained with:")
print(f"  Entry Threshold: 0.75")
print(f"  ML Exit Threshold: 0.75")
print(f"  Exit Models: 2025-10-24")
print(f"  Dataset: {len(df):,} candles")
print(f"\n  LONG Model: {long_model_file.name}")
print(f"  SHORT Model: {short_model_file.name}")
print("\n✅ Ready for backtest validation!")
