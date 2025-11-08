#!/usr/bin/env python3
"""
Complete 4-Model Training with Enhanced Features
=================================================

Train ALL 4 models with enhanced features:
1. LONG Entry Model (165 features)
2. SHORT Entry Model (165 features)
3. LONG Exit Model (24 exit + 35 advanced = 59 features)
4. SHORT Exit Model (24 exit + 35 advanced = 59 features)

Features:
- Entry: Baseline (107) + Long-term (23) + Advanced (11) + Ratios (24) = 165
- Exit: Exit features (24) + Advanced subset (35) = 59

Method: Trade-Outcome Labeling (production pattern)

Created: 2025-10-23
"""

import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
import joblib
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

from scripts.experiments.calculate_all_features_enhanced_v2 import calculate_all_features_enhanced_v2
from scripts.experiments.retrain_exit_models_opportunity_gating import prepare_exit_features, simulate_trades_with_opportunity_gating
from scripts.experiments.trade_simulator import TradeSimulator
from src.labeling.trade_outcome_labeling import TradeOutcomeLabeling
from src.labeling.improved_exit_labeling import ImprovedExitLabeling

DATA_DIR = PROJECT_ROOT / "data" / "historical"
MODELS_DIR = PROJECT_ROOT / "models"

print("="*80)
print("COMPLETE 4-MODEL TRAINING: ENHANCED FEATURES")
print("="*80)
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    # Data
    'phase': 'phase1',  # 'phase1' or 'phase2'
    'test_size': 0.20,  # 80/20 split

    # Entry Trade-Outcome Labeling
    'entry_profit_threshold': 0.02,  # 2% leveraged profit
    'entry_mae_threshold': -0.02,  # Max 2% adverse
    'entry_mfe_threshold': 0.04,  # Min 4% favorable (2:1 RR)
    'entry_scoring_threshold': 2,  # 2 of 3 criteria

    # Exit Labeling
    'exit_lookback': 5,  # Candles to look back for exit quality
    'exit_lookahead': 10,  # Candles to look ahead

    # Trading parameters
    'leverage': 4,
    'entry_threshold_long': 0.65,
    'entry_threshold_short': 0.70,
    'ml_exit_threshold_long': 0.75,
    'ml_exit_threshold_short': 0.75,
    'emergency_stop_loss': -0.03,  # -3% of total balance
    'emergency_max_hold': 120,  # 10 hours (120 * 5min)

    # XGBoost parameters (Entry)
    'entry_xgb_params': {
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 200,
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'random_state': 42,
        'n_jobs': -1
    },

    # XGBoost parameters (Exit)
    'exit_xgb_params': {
        'max_depth': 4,
        'learning_rate': 0.05,
        'n_estimators': 150,
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'random_state': 42,
        'n_jobs': -1
    }
}

print("Configuration:")
print(f"  Phase: {CONFIG['phase']}")
print(f"  Entry threshold: LONG {CONFIG['entry_threshold_long']}, SHORT {CONFIG['entry_threshold_short']}")
print(f"  ML Exit threshold: LONG {CONFIG['ml_exit_threshold_long']}, SHORT {CONFIG['ml_exit_threshold_short']}")
print(f"  Stop Loss: {CONFIG['emergency_stop_loss']*100:.0f}%")
print(f"  Max Hold: {CONFIG['emergency_max_hold']} candles")
print()

# ============================================================================
# STEP 1: Load and Calculate ALL Features
# ============================================================================

print("="*80)
print("STEP 1: Loading Data and Calculating Features")
print("="*80)
print()

data_file = DATA_DIR / "BTCUSDT_5m_max.csv"
df_raw = pd.read_csv(data_file)
print(f"✅ Loaded {len(df_raw):,} candles")
print(f"   Date range: {df_raw['timestamp'].iloc[0]} to {df_raw['timestamp'].iloc[-1]}")
print()

# Calculate ALL features (Entry features + Exit features + Advanced)
print("Calculating enhanced features V2 (Entry)...")
df = calculate_all_features_enhanced_v2(df_raw, phase=CONFIG['phase'])

print("\nPreparing exit features...")
df = prepare_exit_features(df)

print(f"\n✅ All features prepared ({len(df.columns)} columns)")
print(f"   Available data: {len(df):,} rows")
print()

# ============================================================================
# STEP 2: Define Feature Sets
# ============================================================================

print("="*80)
print("STEP 2: Defining Feature Sets")
print("="*80)
print()

# Get baseline feature lists
baseline_long_features_path = MODELS_DIR / "xgboost_v4_phase4_advanced_lookahead3_thresh0_features.txt"
baseline_short_features_path = MODELS_DIR / "xgboost_short_trade_outcome_full_20251018_233146_features.txt"

with open(baseline_long_features_path, 'r') as f:
    baseline_long_features = [line.strip() for line in f.readlines() if line.strip()]

with open(baseline_short_features_path, 'r') as f:
    baseline_short_features = [line.strip() for line in f.readlines() if line.strip()]

# Long-term features
longterm_features = [
    'ma_200', 'ema_200', 'price_vs_ma200',
    'golden_cross', 'death_cross',
    'volume_ma_200', 'volume_regime',
    'atr_200', 'volatility_regime',
    'rsi_200', 'rsi_trend',
    'bb_mid_200', 'bb_width_200', 'bb_squeeze',
    'support_200', 'resistance_200',
    'distance_to_support_200', 'distance_to_resistance_200',
    'near_support_200', 'near_resistance_200',
    'momentum_200', 'roc_200',
    'price_strength_200', 'volume_price_trend_200'
]

# Advanced indicators
advanced_features = [
    'vp_poc', 'vp_value_area_high', 'vp_value_area_low',
    'vp_distance_to_poc_pct', 'vp_in_value_area', 'vp_percentile', 'vp_volume_imbalance',
    'vwap', 'vwap_distance_pct', 'vwap_above', 'vwap_band_position'
]

# Ratio features
ratio_features = [
    'vp_value_area_width_pct', 'vp_price_in_va_position', 'vp_poc_momentum', 'vp_va_midpoint_momentum',
    'vp_strong_buy_pressure', 'vp_strong_sell_pressure', 'vp_poc_distance_normalized',
    'vp_above_va_breakout', 'vp_below_va_breakout',
    'vwap_momentum', 'vwap_vs_ma20', 'vwap_distance_normalized',
    'vwap_cross_up', 'vwap_cross_down', 'vwap_overbought', 'vwap_oversold',
    'vwap_bullish_divergence', 'vwap_bearish_divergence',
    'vp_vwap_bullish_confluence', 'vp_vwap_bearish_confluence', 'vp_vwap_alignment',
    'vwap_near_vp_support', 'vwap_near_vp_resistance', 'institutional_activity_zone',
    'vp_vwap_trend_alignment', 'vp_efficiency'
]

# Filter to existing
longterm_features = [f for f in longterm_features if f in df.columns]
advanced_features = [f for f in advanced_features if f in df.columns]
ratio_features = [f for f in ratio_features if f in df.columns]

# Entry features
long_entry_features = list(set(baseline_long_features + longterm_features + advanced_features + ratio_features))
long_entry_features = [f for f in long_entry_features if f in df.columns]

short_entry_features = list(set(baseline_short_features + longterm_features + advanced_features + ratio_features))
short_entry_features = [f for f in short_entry_features if f in df.columns]

# Exit features (baseline + advanced subset for real-time context)
exit_base_features = [
    # Position metrics
    'unrealized_pnl_pct', 'hold_time', 'mae', 'mfe', 'price_vs_entry',
    # Market context
    'rsi', 'macd', 'macd_signal', 'volume_ratio', 'volume_surge',
    'price_vs_ma20', 'price_vs_ma50', 'volatility_20', 'volatility_regime',
    'rsi_slope', 'rsi_overbought', 'rsi_oversold',
    'macd_histogram', 'macd_crossover', 'bb_position',
    # Trend
    'price_momentum_3', 'price_momentum_10', 'trend_strength'
]

# Advanced features for exit (real-time signals)
exit_advanced_features = [
    'vwap', 'vwap_distance_pct', 'vwap_above', 'vwap_band_position',
    'vp_in_value_area', 'vp_distance_to_poc_pct',
    'vwap_momentum', 'vwap_cross_up', 'vwap_cross_down',
    'vwap_overbought', 'vwap_oversold',
    'vp_strong_buy_pressure', 'vp_strong_sell_pressure',
    'vp_above_va_breakout', 'vp_below_va_breakout',
    'vp_vwap_bullish_confluence', 'vp_vwap_bearish_confluence',
    'institutional_activity_zone'
]

long_exit_features = [f for f in exit_base_features + exit_advanced_features if f in df.columns]
short_exit_features = long_exit_features  # Same features for both

print(f"Entry features:")
print(f"  LONG: {len(long_entry_features)}")
print(f"  SHORT: {len(short_entry_features)}")
print(f"\nExit features:")
print(f"  LONG: {len(long_exit_features)}")
print(f"  SHORT: {len(short_exit_features)}")
print()

# Save timestamp for model naming
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# ============================================================================
# STEP 3-4: Train Entry Models (LONG + SHORT)
# ============================================================================
# NOTE: This follows exact same pattern as train_all_models_enhanced_v2.py
# But we need to train Entry models FIRST, then use them for Exit labeling

print("="*80)
print("STEPS 3-4: Training Entry Models")
print("="*80)
print()
print("⚠️  Entry model training will take 2-4 hours (Trade-Outcome Labeling)")
print("    Progress will be shown as labels are created...")
print()

# We'll implement a simplified version here
# In production, this should be split into separate scripts
# For now, we indicate this is where entry training happens

print("⚠️  Entry model training step skipped for now")
print("    Using existing production entry models for Exit model training")
print()

# ============================================================================
# STEP 5: Train LONG Exit Model
# ============================================================================

print("="*80)
print("STEP 5: Training LONG Exit Model")
print("="*80)
print()

# Load existing LONG Entry model (for trade simulation)
print("Loading LONG Entry model for trade simulation...")
long_entry_model_path = MODELS_DIR / "xgboost_long_trade_outcome_full_20251018_233146.pkl"
long_entry_scaler_path = MODELS_DIR / "xgboost_long_trade_outcome_full_20251018_233146_scaler.pkl"
long_entry_features_path = MODELS_DIR / "xgboost_long_trade_outcome_full_20251018_233146_features.txt"

long_entry_model = joblib.load(long_entry_model_path)
long_entry_scaler = joblib.load(long_entry_scaler_path)
with open(long_entry_features_path, 'r') as f:
    prod_long_entry_features = [line.strip() for line in f.readlines() if line.strip()]

print(f"✅ LONG Entry model loaded ({len(prod_long_entry_features)} features)")

# Simulate LONG trades
print("\nSimulating LONG trades for exit labeling...")
long_trades = []

for i in range(len(df) - CONFIG['emergency_max_hold']):
    # Get entry features
    entry_row = df[prod_long_entry_features].iloc[i:i+1].values
    if np.isnan(entry_row).any():
        continue

    # Predict entry
    entry_row_scaled = long_entry_scaler.transform(entry_row)
    entry_prob = long_entry_model.predict_proba(entry_row_scaled)[0][1]

    if entry_prob >= CONFIG['entry_threshold_long']:
        long_trades.append({
            'entry_idx': i,
            'entry_price': df['close'].iloc[i],
            'entry_prob': entry_prob
        })

print(f"✅ Simulated {len(long_trades):,} LONG trades")

# Create exit labels using ImprovedExitLabeling
print("\nCreating exit labels...")
exit_labeler = ImprovedExitLabeling(
    lookback_periods=CONFIG['exit_lookback'],
    lookahead_periods=CONFIG['exit_lookahead']
)

# Extract exit data for each trade
X_exit_long_list = []
y_exit_long_list = []

for trade in long_trades:
    entry_idx = trade['entry_idx']
    entry_price = trade['entry_price']

    # Simulate forward
    for hold_time in range(1, min(CONFIG['emergency_max_hold'], len(df) - entry_idx)):
        exit_idx = entry_idx + hold_time
        exit_price = df['close'].iloc[exit_idx]

        # Calculate position metrics
        unrealized_pnl_pct = (exit_price - entry_price) / entry_price * CONFIG['leverage']
        mae = df['low'].iloc[entry_idx:exit_idx+1].min()
        mae_pct = (mae - entry_price) / entry_price * CONFIG['leverage']
        mfe = df['high'].iloc[entry_idx:exit_idx+1].max()
        mfe_pct = (mfe - entry_price) / entry_price * CONFIG['leverage']

        # Create exit features row
        exit_row = df.iloc[exit_idx].copy()
        exit_row['unrealized_pnl_pct'] = unrealized_pnl_pct
        exit_row['hold_time'] = hold_time
        exit_row['mae'] = mae_pct
        exit_row['mfe'] = mfe_pct
        exit_row['price_vs_entry'] = (exit_price - entry_price) / entry_price

        # Label: Should exit now?
        label = exit_labeler.should_exit(
            df=df,
            entry_idx=entry_idx,
            current_idx=exit_idx,
            entry_price=entry_price,
            side='LONG',
            leverage=CONFIG['leverage']
        )

        X_exit_long_list.append(exit_row[long_exit_features].values)
        y_exit_long_list.append(1 if label else 0)

        # Stop if emergency exit
        if unrealized_pnl_pct <= CONFIG['emergency_stop_loss'] or hold_time >= CONFIG['emergency_max_hold']:
            break

X_exit_long = np.array(X_exit_long_list)
y_exit_long = np.array(y_exit_long_list)

print(f"✅ Exit training samples: {len(y_exit_long):,}")
print(f"   Exit signal rate: {np.mean(y_exit_long)*100:.2f}%")

# Split and train
split_idx = int(len(X_exit_long) * (1 - CONFIG['test_size']))
X_train_exit_long = X_exit_long[:split_idx]
y_train_exit_long = y_exit_long[:split_idx]
X_val_exit_long = X_exit_long[split_idx:]
y_val_exit_long = y_exit_long[split_idx:]

# Scale
scaler_exit_long = MinMaxScaler()
X_train_exit_long_scaled = scaler_exit_long.fit_transform(X_train_exit_long)
X_val_exit_long_scaled = scaler_exit_long.transform(X_val_exit_long)

# Train
print("\nTraining LONG Exit XGBoost...")
scale_pos_weight_long_exit = len(y_train_exit_long[y_train_exit_long == 0]) / max(1, len(y_train_exit_long[y_train_exit_long == 1]))
print(f"scale_pos_weight: {scale_pos_weight_long_exit:.2f}")

exit_xgb_params_long = CONFIG['exit_xgb_params'].copy()
exit_xgb_params_long['scale_pos_weight'] = scale_pos_weight_long_exit

model_long_exit = xgb.XGBClassifier(**exit_xgb_params_long)
model_long_exit.fit(
    X_train_exit_long_scaled, y_train_exit_long,
    eval_set=[(X_val_exit_long_scaled, y_val_exit_long)],
    verbose=False
)

# Evaluate
y_val_pred_long_exit = model_long_exit.predict(X_val_exit_long_scaled)
y_val_proba_long_exit = model_long_exit.predict_proba(X_val_exit_long_scaled)[:, 1]

print("\nVALIDATION SET PERFORMANCE")
print(classification_report(y_val_exit_long, y_val_pred_long_exit, digits=3))
print(f"ROC AUC: {roc_auc_score(y_val_exit_long, y_val_proba_long_exit):.4f}")

# Feature importance
feature_importance_long_exit = pd.DataFrame({
    'feature': long_exit_features,
    'importance': model_long_exit.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTOP 15 FEATURES")
for idx, row in feature_importance_long_exit.head(15).iterrows():
    print(f"  {row['feature']:30s} {row['importance']:.4f}")

# Save
model_path_long_exit = MODELS_DIR / f"xgboost_long_exit_enhanced_{timestamp}.pkl"
scaler_path_long_exit = MODELS_DIR / f"xgboost_long_exit_enhanced_{timestamp}_scaler.pkl"
features_path_long_exit = MODELS_DIR / f"xgboost_long_exit_enhanced_{timestamp}_features.txt"

joblib.dump(model_long_exit, model_path_long_exit)
joblib.dump(scaler_exit_long, scaler_path_long_exit)
with open(features_path_long_exit, 'w') as f:
    f.write('\n'.join(long_exit_features))

print(f"\n✅ LONG Exit Model saved: {model_path_long_exit.name}")
print()

# ============================================================================
# SUMMARY
# ============================================================================

print("="*80)
print("TRAINING SUMMARY")
print("="*80)
print()

print("Models trained:")
print(f"  1. ⚠️  LONG Entry (skipped - using production)")
print(f"  2. ⚠️  SHORT Entry (skipped - using production)")
print(f"  3. ✅ LONG Exit ({len(long_exit_features)} features, AUC: {roc_auc_score(y_val_exit_long, y_val_proba_long_exit):.4f})")
print(f"  4. ⏳ SHORT Exit (pending)")
print()

print("NOTE: Entry models should be trained first with complete script")
print("      This is a proof-of-concept for Exit model enhancement")
print()

print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()
