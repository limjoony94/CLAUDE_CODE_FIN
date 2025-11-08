#!/usr/bin/env python3
"""
Train All Models with Enhanced Features V2
===========================================

Train 4 models with complete enhanced features:
1. LONG Entry Model
2. SHORT Entry Model
3. LONG Exit Model
4. SHORT Exit Model

Features: Baseline (107) + Long-term (23) + Advanced (11) + Ratios (24) = 165 total

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
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score

from scripts.experiments.calculate_all_features_enhanced_v2 import calculate_all_features_enhanced_v2
from scripts.experiments.retrain_exit_models_opportunity_gating import prepare_exit_features
from scripts.experiments.trade_simulator import TradeSimulator, load_exit_models
from src.labeling.trade_outcome_labeling import TradeOutcomeLabeling

DATA_DIR = PROJECT_ROOT / "data" / "historical"
MODELS_DIR = PROJECT_ROOT / "models"

print("="*80)
print("TRAIN ALL MODELS: ENHANCED FEATURES V2")
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

    # Trade-Outcome Labeling
    'profit_threshold': 0.02,  # 2% leveraged profit
    'mae_threshold': -0.02,  # Max 2% adverse
    'mfe_threshold': 0.04,  # Min 4% favorable (2:1 RR)
    'scoring_threshold': 2,  # 2 of 3 criteria

    # Trading parameters
    'leverage': 4,
    'ml_exit_threshold_long': 0.75,
    'ml_exit_threshold_short': 0.75,
    'emergency_stop_loss': -0.03,  # -3% of total balance
    'emergency_max_hold': 120,  # 10 hours (120 * 5min)

    # XGBoost parameters
    'xgb_params': {
        'max_depth': 6,
        'learning_rate': 0.1,
        'n_estimators': 200,
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'random_state': 42,
        'n_jobs': -1
    }
}

print("Configuration:")
print(f"  Phase: {CONFIG['phase']}")
print(f"  Test split: {CONFIG['test_size']*100:.0f}%")
print(f"  Profit threshold: {CONFIG['profit_threshold']*100:.0f}%")
print(f"  ML Exit threshold: {CONFIG['ml_exit_threshold_long']*100:.0f}% (LONG), {CONFIG['ml_exit_threshold_short']*100:.0f}% (SHORT)")
print(f"  Stop Loss: {CONFIG['emergency_stop_loss']*100:.0f}%")
print(f"  Max Hold: {CONFIG['emergency_max_hold']} candles")
print()

# ============================================================================
# STEP 1: Load and Prepare Data
# ============================================================================

print("="*80)
print("STEP 1: Loading and Preparing Data")
print("="*80)
print()

data_file = DATA_DIR / "BTCUSDT_5m_max.csv"
df_raw = pd.read_csv(data_file)
print(f"✅ Loaded {len(df_raw):,} candles")
print(f"   Date range: {df_raw['timestamp'].iloc[0]} to {df_raw['timestamp'].iloc[-1]}")
print()

# Calculate ALL features (baseline + long-term + advanced + ratios)
print("Calculating enhanced features V2...")
print()
df = calculate_all_features_enhanced_v2(df_raw, phase=CONFIG['phase'])

# Also prepare exit features for trade simulation
print("\nPreparing exit features...")
df = prepare_exit_features(df)
print(f"✅ All features prepared ({len(df.columns)} columns)")
print()

# ============================================================================
# STEP 2: Define Feature Sets
# ============================================================================

print("="*80)
print("STEP 2: Defining Feature Sets")
print("="*80)
print()

# Get feature lists from production models
baseline_long_features_path = MODELS_DIR / "xgboost_v4_phase4_advanced_lookahead3_thresh0_features.txt"
baseline_short_features_path = MODELS_DIR / "xgboost_short_trade_outcome_full_20251018_233146_features.txt"

with open(baseline_long_features_path, 'r') as f:
    baseline_long_features = [line.strip() for line in f.readlines() if line.strip()]

with open(baseline_short_features_path, 'r') as f:
    baseline_short_features = [line.strip() for line in f.readlines() if line.strip()]

print(f"Baseline LONG features: {len(baseline_long_features)}")
print(f"Baseline SHORT features: {len(baseline_short_features)}")

# Add long-term features (23)
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

# Add advanced indicators (11)
advanced_features = [
    'vp_poc', 'vp_value_area_high', 'vp_value_area_low',
    'vp_distance_to_poc_pct', 'vp_in_value_area', 'vp_percentile', 'vp_volume_imbalance',
    'vwap', 'vwap_distance_pct', 'vwap_above', 'vwap_band_position'
]

# Add engineered ratio features (24)
ratio_features = [
    # Volume Profile ratios (8)
    'vp_value_area_width_pct', 'vp_price_in_va_position', 'vp_poc_momentum', 'vp_va_midpoint_momentum',
    'vp_strong_buy_pressure', 'vp_strong_sell_pressure', 'vp_poc_distance_normalized',
    'vp_above_va_breakout', 'vp_below_va_breakout',
    # VWAP ratios (8)
    'vwap_momentum', 'vwap_vs_ma20', 'vwap_distance_normalized',
    'vwap_cross_up', 'vwap_cross_down', 'vwap_overbought', 'vwap_oversold',
    'vwap_bullish_divergence', 'vwap_bearish_divergence',
    # Combined VP+VWAP (7)
    'vp_vwap_bullish_confluence', 'vp_vwap_bearish_confluence', 'vp_vwap_alignment',
    'vwap_near_vp_support', 'vwap_near_vp_resistance', 'institutional_activity_zone',
    'vp_vwap_trend_alignment', 'vp_efficiency'
]

# Filter to only features that exist in df
longterm_features = [f for f in longterm_features if f in df.columns]
advanced_features = [f for f in advanced_features if f in df.columns]
ratio_features = [f for f in ratio_features if f in df.columns]

print(f"\nEnhanced features available:")
print(f"  Long-term: {len(longterm_features)}")
print(f"  Advanced: {len(advanced_features)}")
print(f"  Ratios: {len(ratio_features)}")

# Combine for LONG Entry
long_entry_features = list(set(baseline_long_features + longterm_features + advanced_features + ratio_features))
long_entry_features = [f for f in long_entry_features if f in df.columns]

# Combine for SHORT Entry
short_entry_features = list(set(baseline_short_features + longterm_features + advanced_features + ratio_features))
short_entry_features = [f for f in short_entry_features if f in df.columns]

print(f"\nFinal feature counts:")
print(f"  LONG Entry: {len(long_entry_features)}")
print(f"  SHORT Entry: {len(short_entry_features)}")
print()

# ============================================================================
# STEP 3: Load Exit Models and Create Simulators
# ============================================================================

print("="*80)
print("STEP 3: Loading Exit Models for Trade Simulation")
print("="*80)
print()

exit_models = load_exit_models()

long_simulator = TradeSimulator(
    exit_model=exit_models['long'][0],
    exit_scaler=exit_models['long'][1],
    exit_features=exit_models['long'][2],
    leverage=CONFIG['leverage'],
    ml_exit_threshold=CONFIG['ml_exit_threshold_long'],
    emergency_stop_loss=CONFIG['emergency_stop_loss'],
    emergency_max_hold=CONFIG['emergency_max_hold']
)

short_simulator = TradeSimulator(
    exit_model=exit_models['short'][0],
    exit_scaler=exit_models['short'][1],
    exit_features=exit_models['short'][2],
    leverage=CONFIG['leverage'],
    ml_exit_threshold=CONFIG['ml_exit_threshold_short'],
    emergency_stop_loss=CONFIG['emergency_stop_loss'],
    emergency_max_hold=CONFIG['emergency_max_hold']
)

print("✅ Trade simulators ready")
print()

# ============================================================================
# STEP 4: Create Trade-Outcome Labels
# ============================================================================

print("="*80)
print("STEP 4: Creating Trade-Outcome Labels")
print("="*80)
print()

labeler = TradeOutcomeLabeling(
    profit_threshold=CONFIG['profit_threshold'],
    mae_threshold=CONFIG['mae_threshold'],
    mfe_threshold=CONFIG['mfe_threshold'],
    scoring_threshold=CONFIG['scoring_threshold']
)

print("Creating LONG Entry labels...")
long_labels = labeler.create_entry_labels(df, long_simulator, 'LONG', show_progress=True)

print("\nCreating SHORT Entry labels...")
short_labels = labeler.create_entry_labels(df, short_simulator, 'SHORT', show_progress=True)

print()
print(f"✅ Labels created")
print(f"   LONG positive rate: {np.sum(long_labels)/len(long_labels)*100:.2f}%")
print(f"   SHORT positive rate: {np.sum(short_labels)/len(short_labels)*100:.2f}%")
print()

# ============================================================================
# STEP 5: Train LONG Entry Model
# ============================================================================

print("="*80)
print("STEP 5: Training LONG Entry Model")
print("="*80)
print()

# Prepare data
X_long = df[long_entry_features].values
y_long = long_labels

# Split (80/20, time series - no shuffle)
split_idx = int(len(X_long) * (1 - CONFIG['test_size']))
X_train_long = X_long[:split_idx]
y_train_long = y_long[:split_idx]
X_val_long = X_long[split_idx:]
y_val_long = y_long[split_idx:]

print(f"Train samples: {len(X_train_long):,}")
print(f"Validation samples: {len(X_val_long):,}")
print(f"Positive rate (train): {np.sum(y_train_long)/len(y_train_long)*100:.2f}%")
print(f"Positive rate (val): {np.sum(y_val_long)/len(y_val_long)*100:.2f}%")

# Scale features
scaler_long_entry = StandardScaler()
X_train_long_scaled = scaler_long_entry.fit_transform(X_train_long)
X_val_long_scaled = scaler_long_entry.transform(X_val_long)

# Train XGBoost
print("\nTraining XGBoost...")
scale_pos_weight_long = len(y_train_long[y_train_long == 0]) / max(1, len(y_train_long[y_train_long == 1]))
print(f"scale_pos_weight: {scale_pos_weight_long:.2f}")

xgb_params_long = CONFIG['xgb_params'].copy()
xgb_params_long['scale_pos_weight'] = scale_pos_weight_long

model_long_entry = xgb.XGBClassifier(**xgb_params_long)
model_long_entry.fit(
    X_train_long_scaled, y_train_long,
    eval_set=[(X_val_long_scaled, y_val_long)],
    verbose=False
)

# Evaluate
y_val_pred_long = model_long_entry.predict(X_val_long_scaled)
y_val_proba_long = model_long_entry.predict_proba(X_val_long_scaled)[:, 1]

print("\nVALIDATION SET PERFORMANCE")
print(classification_report(y_val_long, y_val_pred_long, digits=3))
print(f"ROC AUC: {roc_auc_score(y_val_long, y_val_proba_long):.4f}")

# Feature importance
feature_importance_long = pd.DataFrame({
    'feature': long_entry_features,
    'importance': model_long_entry.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTOP 20 FEATURES")
for idx, row in feature_importance_long.head(20).iterrows():
    print(f"  {row['feature']:40s} {row['importance']:.4f}")

# Save model
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_path_long = MODELS_DIR / f"xgboost_long_entry_enhanced_v2_{timestamp}.pkl"
scaler_path_long = MODELS_DIR / f"xgboost_long_entry_enhanced_v2_{timestamp}_scaler.pkl"
features_path_long = MODELS_DIR / f"xgboost_long_entry_enhanced_v2_{timestamp}_features.txt"

joblib.dump(model_long_entry, model_path_long)
joblib.dump(scaler_long_entry, scaler_path_long)
with open(features_path_long, 'w') as f:
    f.write('\n'.join(long_entry_features))

print(f"\n✅ LONG Entry Model saved:")
print(f"   {model_path_long.name}")
print()

# ============================================================================
# STEP 6: Train SHORT Entry Model
# ============================================================================

print("="*80)
print("STEP 6: Training SHORT Entry Model")
print("="*80)
print()

# Prepare data
X_short = df[short_entry_features].values
y_short = short_labels

# Split (80/20, time series - no shuffle)
X_train_short = X_short[:split_idx]
y_train_short = y_short[:split_idx]
X_val_short = X_short[split_idx:]
y_val_short = y_short[split_idx:]

print(f"Train samples: {len(X_train_short):,}")
print(f"Validation samples: {len(X_val_short):,}")
print(f"Positive rate (train): {np.sum(y_train_short)/len(y_train_short)*100:.2f}%")
print(f"Positive rate (val): {np.sum(y_val_short)/len(y_val_short)*100:.2f}%")

# Scale features
scaler_short_entry = StandardScaler()
X_train_short_scaled = scaler_short_entry.fit_transform(X_train_short)
X_val_short_scaled = scaler_short_entry.transform(X_val_short)

# Train XGBoost
print("\nTraining XGBoost...")
scale_pos_weight_short = len(y_train_short[y_train_short == 0]) / max(1, len(y_train_short[y_train_short == 1]))
print(f"scale_pos_weight: {scale_pos_weight_short:.2f}")

xgb_params_short = CONFIG['xgb_params'].copy()
xgb_params_short['scale_pos_weight'] = scale_pos_weight_short

model_short_entry = xgb.XGBClassifier(**xgb_params_short)
model_short_entry.fit(
    X_train_short_scaled, y_train_short,
    eval_set=[(X_val_short_scaled, y_val_short)],
    verbose=False
)

# Evaluate
y_val_pred_short = model_short_entry.predict(X_val_short_scaled)
y_val_proba_short = model_short_entry.predict_proba(X_val_short_scaled)[:, 1]

print("\nVALIDATION SET PERFORMANCE")
print(classification_report(y_val_short, y_val_pred_short, digits=3))
print(f"ROC AUC: {roc_auc_score(y_val_short, y_val_proba_short):.4f}")

# Feature importance
feature_importance_short = pd.DataFrame({
    'feature': short_entry_features,
    'importance': model_short_entry.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTOP 20 FEATURES")
for idx, row in feature_importance_short.head(20).iterrows():
    print(f"  {row['feature']:40s} {row['importance']:.4f}")

# Save model
model_path_short = MODELS_DIR / f"xgboost_short_entry_enhanced_v2_{timestamp}.pkl"
scaler_path_short = MODELS_DIR / f"xgboost_short_entry_enhanced_v2_{timestamp}_scaler.pkl"
features_path_short = MODELS_DIR / f"xgboost_short_entry_enhanced_v2_{timestamp}_features.txt"

joblib.dump(model_short_entry, model_path_short)
joblib.dump(scaler_short_entry, scaler_path_short)
with open(features_path_short, 'w') as f:
    f.write('\n'.join(short_entry_features))

print(f"\n✅ SHORT Entry Model saved:")
print(f"   {model_path_short.name}")
print()

# ============================================================================
# SUMMARY
# ============================================================================

print("="*80)
print("TRAINING COMPLETE")
print("="*80)
print()

print("Models trained:")
print(f"  1. ✅ LONG Entry ({len(long_entry_features)} features, AUC: {roc_auc_score(y_val_long, y_val_proba_long):.4f})")
print(f"  2. ✅ SHORT Entry ({len(short_entry_features)} features, AUC: {roc_auc_score(y_val_short, y_val_proba_short):.4f})")
print()

print("Note: EXIT models NOT retrained (using existing production models)")
print("  - LONG Exit: xgboost_long_exit_oppgating_improved_20251017_151624.pkl")
print("  - SHORT Exit: xgboost_short_exit_oppgating_improved_20251017_152440.pkl")
print()

print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

print("="*80)
print("NEXT STEPS")
print("="*80)
print()
print("1. Review feature importance (advanced indicators in top 20?)")
print("2. Backtest validation with new models")
print("3. Compare performance vs production models")
print()
