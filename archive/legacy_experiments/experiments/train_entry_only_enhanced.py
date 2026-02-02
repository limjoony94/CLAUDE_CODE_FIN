"""
Complete 4-Model Training with Enhanced Features (CORRECT ORDER)
================================================================

CORRECT TRAINING SEQUENCE:
1. Exit models FIRST (LONG Exit, SHORT Exit)
   - Uses production Entry models for trade simulation
   - Creates exit labels with ImprovedExitLabeling

2. Entry models SECOND (LONG Entry, SHORT Entry)
   - Uses newly trained Exit models for Trade-Outcome labeling
   - Creates entry labels based on full trade outcomes

Enhanced Features (165 total):
- Baseline: 107 features (LONG/SHORT specific)
- Long-term: 23 features (200-period indicators)
- Advanced: 11 features (Volume Profile 7 + VWAP 4)
- Ratios: 24 features (engineered from advanced)

Created: 2025-10-23
"""

import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
import joblib
from pathlib import Path
import sys
from datetime import datetime
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import classification_report, roc_auc_score

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiments.calculate_all_features_enhanced_v2 import calculate_all_features_enhanced_v2
from scripts.experiments.retrain_exit_models_opportunity_gating import prepare_exit_features
from src.labeling.improved_exit_labeling import ImprovedExitLabeling
from src.labeling.trade_outcome_labeling import TradeOutcomeLabeling

DATA_DIR = PROJECT_ROOT / "data" / "historical"
MODELS_DIR = PROJECT_ROOT / "models"

# ============================================================================
# CONFIGURATION
# ============================================================================

CONFIG = {
    'phase': 'phase1',  # Volume Profile + VWAP
    'test_size': 0.20,

    # Entry parameters (for Exit training trade simulation)
    'entry_threshold_long': 0.65,
    'entry_threshold_short': 0.70,

    # Trade-Outcome parameters (for Entry training)
    'entry_profit_threshold': 0.02,
    'entry_mae_threshold': -0.02,
    'entry_mfe_threshold': 0.04,
    'entry_scoring_threshold': 2,

    # Exit parameters (OPTIMIZED 2025-10-22)
    'leverage': 4,
    'ml_exit_threshold_long': 0.75,  # Optimized from 0.70
    'ml_exit_threshold_short': 0.75,  # Optimized from 0.72
    'emergency_stop_loss': -0.03,  # Optimized from -0.04
    'emergency_max_hold': 120,  # Optimized from 96

    # XGBoost parameters
    'xgb_params': {
        'max_depth': 6,
        'learning_rate': 0.05,  # PRODUCTION: 0.05 (not 0.1)
        'n_estimators': 200,
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'random_state': 42,
        'n_jobs': -1,
        'subsample': 0.8,
        'colsample_bytree': 0.8
    }
}

print("="*80)
print("COMPLETE 4-MODEL TRAINING: ENHANCED FEATURES")
print("="*80)
print(f"\nConfiguration:")
print(f"  Phase: {CONFIG['phase']}")
print(f"  Entry Threshold (LONG/SHORT): {CONFIG['entry_threshold_long']}/{CONFIG['entry_threshold_short']}")
print(f"  ML Exit Threshold (LONG/SHORT): {CONFIG['ml_exit_threshold_long']}/{CONFIG['ml_exit_threshold_short']}")
print(f"  Stop Loss: {CONFIG['emergency_stop_loss']*100}%")
print(f"  Max Hold: {CONFIG['emergency_max_hold']} candles")
print()

# ============================================================================
# STEP 1: Load and Calculate Enhanced Features
# ============================================================================

print("="*80)
print("STEP 1: Loading Data and Calculating Enhanced Features")
print("="*80)

data_file = DATA_DIR / "BTCUSDT_5m_max.csv"
df_raw = pd.read_csv(data_file)
print(f"\n✅ Loaded {len(df_raw):,} candles")
print(f"   Date range: {df_raw['timestamp'].iloc[0]} to {df_raw['timestamp'].iloc[-1]}")

# Calculate all enhanced features (165 total)
print(f"\nCalculating enhanced features ({CONFIG['phase']})...")
df = calculate_all_features_enhanced_v2(df_raw, phase=CONFIG['phase'])

# Add Exit-specific features (returns, volatility, volume_surge, etc.)
df = prepare_exit_features(df)

print(f"\n✅ Enhanced features calculated (including Exit-specific features)")
print(f"   Total columns: {len(df.columns)}")
print(f"   Final rows: {len(df):,}")
print()

# ============================================================================
# STEP 2: Load Production Entry Models (for Exit training)
# ============================================================================

print("="*80)
print("STEP 2: Loading Production Entry Models")
print("="*80)

# Load baseline feature names
baseline_long_features_path = MODELS_DIR / "xgboost_long_trade_outcome_full_20251018_233146_features.txt"
baseline_short_features_path = MODELS_DIR / "xgboost_short_trade_outcome_full_20251018_233146_features.txt"

with open(baseline_long_features_path, 'r') as f:
    baseline_long_features = [line.strip() for line in f if line.strip()]

with open(baseline_short_features_path, 'r') as f:
    baseline_short_features = [line.strip() for line in f if line.strip()]

# Load production Entry models
prod_long_entry_model = joblib.load(MODELS_DIR / "xgboost_long_trade_outcome_full_20251018_233146.pkl")
prod_long_entry_scaler = joblib.load(MODELS_DIR / "xgboost_long_trade_outcome_full_20251018_233146_scaler.pkl")

prod_short_entry_model = joblib.load(MODELS_DIR / "xgboost_short_trade_outcome_full_20251018_233146.pkl")
prod_short_entry_scaler = joblib.load(MODELS_DIR / "xgboost_short_trade_outcome_full_20251018_233146_scaler.pkl")

print(f"\n✅ Production Entry models loaded")
print(f"   LONG Entry features: {len(baseline_long_features)}")
print(f"   SHORT Entry features: {len(baseline_short_features)}")
print()

# ============================================================================
# STEP 3: Prepare Enhanced Feature Sets
# ============================================================================

print("="*80)
print("STEP 3: Preparing Enhanced Feature Sets")
print("="*80)

# Long-term features (ONLY features that actually exist)
# Verified: Only ma_200, ema_200, rsi_200, atr_200 exist from long-term calculation
# Many features listed don't actually get calculated (bb_upper_200, obv_200, golden_cross, etc.)
longterm_features = [
    'ma_200', 'ema_200', 'rsi_200', 'atr_200'  # Only these 4 confirmed to exist
]

# Advanced indicators (11)
advanced_features = [
    'vp_poc', 'vp_value_area_high', 'vp_value_area_low',
    'vp_distance_to_poc_pct', 'vp_in_value_area', 'vp_percentile', 'vp_volume_imbalance',
    'vwap', 'vwap_distance_pct', 'vwap_above', 'vwap_band_position'
]

# Ratio features (24)
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

# Enhanced Entry features (baseline + additions)
long_entry_features = baseline_long_features + longterm_features + advanced_features + ratio_features
short_entry_features = baseline_short_features + longterm_features + advanced_features + ratio_features

# Exit features - start with base position metrics (will be added dynamically)
# PRODUCTION STYLE: Technical indicators only (NO position features!)
# Position features (unrealized_pnl_pct, hold_time, mae, mfe, price_vs_entry) are NOT used
# Reason: Exit model must work on market state alone, not position-specific information

# Technical features from baseline + prepare_exit_features()
exit_technical_candidates = [
    # Basic technical
    'rsi', 'macd', 'macd_signal', 'volume', 'atr',
    'close', 'high', 'low', 'open',

    # From prepare_exit_features() - Market context features
    'ema_12', 'trend_strength',
    'volatility_regime', 'volume_surge', 'price_acceleration', 'volume_ratio',
    'price_vs_ma20', 'price_vs_ma50', 'volatility_20',
    'rsi_slope', 'rsi_overbought', 'rsi_oversold', 'rsi_divergence',
    'macd_histogram_slope', 'macd_crossover', 'macd_crossunder',
    'higher_high', 'lower_low',
    'near_resistance', 'near_support',
    'bb_position',
    'returns'  # Also from prepare_exit_features()
]

# Advanced features that we added
exit_advanced_candidates = [
    'vwap', 'vwap_distance_pct', 'vwap_above', 'vwap_band_position',
    'vp_in_value_area', 'vp_distance_to_poc_pct',
    'vwap_momentum', 'vwap_cross_up', 'vwap_cross_down',
    'vp_vwap_bullish_confluence', 'vp_vwap_bearish_confluence', 'institutional_activity_zone'
]

# Long-term features (ONLY features that actually exist - verified!)
exit_longterm_candidates = [
    'ma_200', 'ema_200', 'rsi_200', 'atr_200'  # golden_cross, death_cross don't exist
]

# Filter to only use features that exist in DataFrame (NO position features!)
all_exit_candidates = exit_technical_candidates + exit_advanced_candidates + exit_longterm_candidates
long_exit_features = [f for f in all_exit_candidates if f in df.columns]
short_exit_features = long_exit_features.copy()

# DEBUG: Print filtered features
print(f"\n🔍 DEBUG: Exit Feature Filtering")
print(f"   Total candidates: {len(all_exit_candidates)}")
print(f"   Filtered features: {len(long_exit_features)}")
missing_features = [f for f in all_exit_candidates if f not in df.columns and f not in exit_base_features]
if missing_features:
    print(f"   ⚠️  Missing features (filtered out): {len(missing_features)}")
    print(f"      {missing_features[:10]}...")  # Show first 10

print(f"\n✅ Feature sets prepared:")
print(f"   LONG Entry: {len(long_entry_features)} features")
print(f"   SHORT Entry: {len(short_entry_features)} features")
print(f"   Exit (filtered): {len(long_exit_features)} features (from {len(all_exit_candidates)} candidates)")
print(f"   SHORT Exit: {len(short_exit_features)} features")
print()

# ============================================================================
# STEP 4: Train EXIT Models FIRST
# ============================================================================

# ============================================================================
# STEP 4: Load Pre-trained EXIT Models
# ============================================================================

print("="*80)
print("STEP 4: Loading Pre-trained EXIT Models")
print("="*80)
print("
Why load Exit models?")
print("  - Exit models already trained with production methodology")
print("  - Entry models need Trade-Outcome labels")
print("  - Trade-Outcome labels need Exit model for criterion 4 (efficient exit)")
print("  - Loading saves ~20-30 minutes of training time")
print()

# Model file paths (timestamp from previous training: 20251023_212712)
exit_model_timestamp = "20251023_212712"

# ----------------------------------------------------------------------------
# 4.1: Load LONG Exit Model
# ----------------------------------------------------------------------------

print("-"*80)
print("4.1: Loading LONG Exit Model")
print("-"*80)

model_path_long_exit = MODELS_DIR / f"xgboost_long_exit_enhanced_{exit_model_timestamp}.pkl"
scaler_path_long_exit = MODELS_DIR / f"xgboost_long_exit_enhanced_{exit_model_timestamp}_scaler.pkl"
features_path_long_exit = MODELS_DIR / f"xgboost_long_exit_enhanced_{exit_model_timestamp}_features.txt"

# Load model, scaler, and features
model_long_exit = joblib.load(model_path_long_exit)
scaler_exit_long = joblib.load(scaler_path_long_exit)
with open(features_path_long_exit, 'r') as f:
    long_exit_features = [line.strip() for line in f.readlines()]

print(f"
✅ LONG Exit model loaded:")
print(f"   Model: {model_path_long_exit.name}")
print(f"   Features: {len(long_exit_features)}")
print()

# ----------------------------------------------------------------------------
# 4.2: Load SHORT Exit Model
# ----------------------------------------------------------------------------

print("-"*80)
print("4.2: Loading SHORT Exit Model")
print("-"*80)

model_path_short_exit = MODELS_DIR / f"xgboost_short_exit_enhanced_{exit_model_timestamp}.pkl"
scaler_path_short_exit = MODELS_DIR / f"xgboost_short_exit_enhanced_{exit_model_timestamp}_scaler.pkl"
features_path_short_exit = MODELS_DIR / f"xgboost_short_exit_enhanced_{exit_model_timestamp}_features.txt"

# Load model, scaler, and features
model_short_exit = joblib.load(model_path_short_exit)
scaler_exit_short = joblib.load(scaler_path_short_exit)
with open(features_path_short_exit, 'r') as f:
    short_exit_features = [line.strip() for line in f.readlines()]

print(f"
✅ SHORT Exit model loaded:")
print(f"   Model: {model_path_short_exit.name}")
print(f"   Features: {len(short_exit_features)}")
print()


# ============================================================================
# STEP 5: Train ENTRY Models SECOND
# ============================================================================

print("="*80)
print("STEP 5: Training ENTRY Models (LONG + SHORT)")
print("="*80)
print("\nWhy Entry second?")
print("  - Entry labels use Trade-Outcome Labeling")
print("  - Trade-Outcome criterion 4 needs Exit model (efficient exit)")
print("  - So we use the newly trained Exit models above")
print()

# Create Trade-Outcome labeler
entry_labeler = TradeOutcomeLabeling(
    profit_threshold=CONFIG['entry_profit_threshold'],
    mae_threshold=CONFIG['entry_mae_threshold'],
    mfe_threshold=CONFIG['entry_mfe_threshold'],
    scoring_threshold=CONFIG['entry_scoring_threshold']
)

# ----------------------------------------------------------------------------
# 5.1: Train LONG Entry Model
# ----------------------------------------------------------------------------

print("-"*80)
print("5.1: Training LONG Entry Model")
print("-"*80)

# Create Trade-Outcome labels using new Exit model
print("\nCreating LONG entry labels with Trade-Outcome...")

# Build simple simulator using new Exit model
class SimpleExitSimulator:
    def __init__(self, model, scaler, features, threshold, max_hold, stop_loss, leverage):
        self.model = model
        self.scaler = scaler
        self.features = features
        self.threshold = threshold
        self.max_hold = max_hold
        self.emergency_max_hold = max_hold  # For TradeOutcomeLabeling compatibility
        self.stop_loss = stop_loss
        self.leverage = leverage

    def simulate_trade(self, df, entry_idx, side):
        """Simulate single trade and return exit info"""
        # Get entry price from DataFrame
        entry_price = df['close'].iloc[entry_idx]

        for hold_time in range(1, self.max_hold):
            exit_idx = entry_idx + hold_time
            if exit_idx >= len(df):
                return {'exit_reason': 'max_hold', 'hold_time': hold_time, 'exit_idx': exit_idx-1}

            exit_price = df['close'].iloc[exit_idx]

            # Calculate P&L
            if side == 'LONG':
                unrealized_pnl_pct = (exit_price - entry_price) / entry_price * self.leverage
            else:  # SHORT
                unrealized_pnl_pct = (entry_price - exit_price) / entry_price * self.leverage

            # Stop loss check
            if unrealized_pnl_pct <= self.stop_loss:
                return {'exit_reason': 'stop_loss', 'hold_time': hold_time, 'exit_idx': exit_idx}

            # MAE/MFE
            price_window = df['close'].iloc[entry_idx:exit_idx+1]
            if side == 'LONG':
                mae = (price_window.min() - entry_price) / entry_price * self.leverage
                mfe = (price_window.max() - entry_price) / entry_price * self.leverage
                price_vs_entry = (exit_price - entry_price) / entry_price
            else:
                mae = (entry_price - price_window.max()) / entry_price * self.leverage
                mfe = (entry_price - price_window.min()) / entry_price * self.leverage
                price_vs_entry = (entry_price - exit_price) / entry_price

            # Prepare exit features
            exit_row = df.iloc[exit_idx].copy()
            exit_row['unrealized_pnl_pct'] = unrealized_pnl_pct
            exit_row['hold_time'] = hold_time
            exit_row['mae'] = mae
            exit_row['mfe'] = mfe
            exit_row['price_vs_entry'] = price_vs_entry            # ML Exit check
            exit_features_values = exit_row[self.features]
            if exit_features_values.isna().any():
                continue
            exit_features_values = exit_features_values.values.reshape(1, -1)

            exit_features_scaled = self.scaler.transform(exit_features_values)
            exit_prob = self.model.predict_proba(exit_features_scaled)[0][1]

            if exit_prob >= self.threshold:
                return {'exit_reason': 'ml_exit', 'hold_time': hold_time, 'exit_idx': exit_idx}

        return {'exit_reason': 'max_hold', 'hold_time': self.max_hold, 'exit_idx': entry_idx + self.max_hold - 1}

long_simulator = SimpleExitSimulator(
    model_long_exit, scaler_exit_long, long_exit_features,
    CONFIG['ml_exit_threshold_long'], CONFIG['emergency_max_hold'],
    CONFIG['emergency_stop_loss'], CONFIG['leverage']
)

# Create labels
long_entry_labels = entry_labeler.create_entry_labels(
    df, long_simulator, 'LONG', show_progress=True
)

print(f"✅ Created {len(long_entry_labels)} LONG entry labels")
print(f"   Positive rate: {np.mean(long_entry_labels)*100:.2f}%")

# Prepare training data
X_entry_long = df[long_entry_features].values
y_entry_long = long_entry_labels

# PRODUCTION: Scale (StandardScaler for Entry)
scaler_entry_long = StandardScaler()
X_entry_long_scaled = scaler_entry_long.fit_transform(X_entry_long)

# PRODUCTION: Train on all data + 5-fold CV validation
print(f"\nTraining LONG Entry model (full data + 5-fold CV)...")
print(f"  Total samples: {len(X_entry_long):,}")

scale_pos_weight_long_entry = np.sum(y_entry_long == 0) / np.sum(y_entry_long == 1)

entry_xgb_params_long = CONFIG['xgb_params'].copy()
entry_xgb_params_long['scale_pos_weight'] = scale_pos_weight_long_entry

model_long_entry = xgb.XGBClassifier(**entry_xgb_params_long)

# 5-fold time series cross-validation
print("Running 5-fold time series cross-validation...")
cv_scores = cross_val_score(model_long_entry, X_entry_long_scaled, y_entry_long,
                            cv=tscv, scoring='precision', n_jobs=-1)
print(f"CV Precision: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
print(f"Folds: {', '.join([f'{s:.4f}' for s in cv_scores])}")

# Train final model on all data
print("\nTraining final model on all data...")
model_long_entry.fit(X_entry_long_scaled, y_entry_long)

# Evaluate on training data
y_pred_long_entry = model_long_entry.predict(X_entry_long_scaled)
y_proba_long_entry = model_long_entry.predict_proba(X_entry_long_scaled)[:, 1]

print("\n✅ LONG Entry Model Performance (training):")
print(classification_report(y_entry_long, y_pred_long_entry, digits=4))
print(f"ROC-AUC: {roc_auc_score(y_entry_long, y_proba_long_entry):.4f}")

# Feature importance
feature_importance_long_entry = pd.DataFrame({
    'feature': long_entry_features,
    'importance': model_long_entry.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 Features (LONG Entry):")
for idx, row in feature_importance_long_entry.head(10).iterrows():
    print(f"  {row['feature']:30s} {row['importance']:.4f}")

# Save
model_path_long_entry = MODELS_DIR / f"xgboost_long_entry_enhanced_{timestamp}.pkl"
scaler_path_long_entry = MODELS_DIR / f"xgboost_long_entry_enhanced_{timestamp}_scaler.pkl"
features_path_long_entry = MODELS_DIR / f"xgboost_long_entry_enhanced_{timestamp}_features.txt"

joblib.dump(model_long_entry, model_path_long_entry)
joblib.dump(scaler_entry_long, scaler_path_long_entry)
with open(features_path_long_entry, 'w') as f:
    f.write('\n'.join(long_entry_features))

print(f"\n✅ LONG Entry model saved:")
print(f"   {model_path_long_entry.name}")
print()

# ----------------------------------------------------------------------------
# 5.2: Train SHORT Entry Model
# ----------------------------------------------------------------------------

print("-"*80)
print("5.2: Training SHORT Entry Model")
print("-"*80)

# Create Trade-Outcome labels using new Exit model
print("\nCreating SHORT entry labels with Trade-Outcome...")

short_simulator = SimpleExitSimulator(
    model_short_exit, scaler_exit_short, short_exit_features,
    CONFIG['ml_exit_threshold_short'], CONFIG['emergency_max_hold'],
    CONFIG['emergency_stop_loss'], CONFIG['leverage']
)

# Create labels
short_entry_labels = entry_labeler.create_entry_labels(
    df, short_simulator, 'SHORT', show_progress=True
)

print(f"✅ Created {len(short_entry_labels)} SHORT entry labels")
print(f"   Positive rate: {np.mean(short_entry_labels)*100:.2f}%")

# Prepare training data
X_entry_short = df[short_entry_features].values
y_entry_short = short_entry_labels

# PRODUCTION: Scale (StandardScaler for Entry)
scaler_entry_short = StandardScaler()
X_entry_short_scaled = scaler_entry_short.fit_transform(X_entry_short)

# PRODUCTION: Train on all data + 5-fold CV validation
print(f"\nTraining SHORT Entry model (full data + 5-fold CV)...")
print(f"  Total samples: {len(X_entry_short):,}")

scale_pos_weight_short_entry = np.sum(y_entry_short == 0) / np.sum(y_entry_short == 1)

entry_xgb_params_short = CONFIG['xgb_params'].copy()
entry_xgb_params_short['scale_pos_weight'] = scale_pos_weight_short_entry

model_short_entry = xgb.XGBClassifier(**entry_xgb_params_short)

# 5-fold time series cross-validation
print("Running 5-fold time series cross-validation...")
cv_scores = cross_val_score(model_short_entry, X_entry_short_scaled, y_entry_short,
                            cv=tscv, scoring='precision', n_jobs=-1)
print(f"CV Precision: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
print(f"Folds: {', '.join([f'{s:.4f}' for s in cv_scores])}")

# Train final model on all data
print("\nTraining final model on all data...")
model_short_entry.fit(X_entry_short_scaled, y_entry_short)

# Evaluate on training data
y_pred_short_entry = model_short_entry.predict(X_entry_short_scaled)
y_proba_short_entry = model_short_entry.predict_proba(X_entry_short_scaled)[:, 1]

print("\n✅ SHORT Entry Model Performance (training):")
print(classification_report(y_entry_short, y_pred_short_entry, digits=4))
print(f"ROC-AUC: {roc_auc_score(y_entry_short, y_proba_short_entry):.4f}")

# Feature importance
feature_importance_short_entry = pd.DataFrame({
    'feature': short_entry_features,
    'importance': model_short_entry.feature_importances_
}).sort_values('importance', ascending=False)

print("\nTop 10 Features (SHORT Entry):")
for idx, row in feature_importance_short_entry.head(10).iterrows():
    print(f"  {row['feature']:30s} {row['importance']:.4f}")

# Save
model_path_short_entry = MODELS_DIR / f"xgboost_short_entry_enhanced_{timestamp}.pkl"
scaler_path_short_entry = MODELS_DIR / f"xgboost_short_entry_enhanced_{timestamp}_scaler.pkl"
features_path_short_entry = MODELS_DIR / f"xgboost_short_entry_enhanced_{timestamp}_features.txt"

joblib.dump(model_short_entry, model_path_short_entry)
joblib.dump(scaler_entry_short, scaler_path_short_entry)
with open(features_path_short_entry, 'w') as f:
    f.write('\n'.join(short_entry_features))

print(f"\n✅ SHORT Entry model saved:")
print(f"   {model_path_short_entry.name}")
print()

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("="*80)
print("TRAINING COMPLETE - ALL 4 MODELS")
print("="*80)

print(f"\n📊 Models Saved (timestamp: {timestamp}):")
print("\n1. LONG Exit:")
print(f"   {model_path_long_exit.name}")
print(f"   Features: {len(long_exit_features)}")
print(f"   Samples: {len(X_exit_long):,}")

print("\n2. SHORT Exit:")
print(f"   {model_path_short_exit.name}")
print(f"   Features: {len(short_exit_features)}")
print(f"   Samples: {len(X_exit_short):,}")

print("\n3. LONG Entry:")
print(f"   {model_path_long_entry.name}")
print(f"   Features: {len(long_entry_features)}")
print(f"   Samples: {len(X_entry_long):,}")

print("\n4. SHORT Entry:")
print(f"   {model_path_short_entry.name}")
print(f"   Features: {len(short_entry_features)}")
print(f"   Samples: {len(X_entry_short):,}")

print("\n✅ All models trained with enhanced features (165 total):")
print("   - Baseline: 107 features")
print("   - Long-term: 23 features")
print("   - Advanced: 11 features (VP + VWAP)")
print("   - Ratios: 24 features (engineered)")

print("\n🎯 Next Steps:")
print("   1. Backtest validation with new models")
print("   2. Performance comparison vs production models")
print("   3. Deploy if improvement confirmed")

print("\n" + "="*80)
