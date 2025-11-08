"""
Retrain All 4 Models on 90-Day 5-Min Complete Dataset
======================================================

Purpose: Train LONG/SHORT Entry/Exit models with Enhanced 5-Fold CV

Dataset: 90 days of 5-min data (Aug 8 - Nov 6, 2025)
Training: 61 days (Aug 8 - Oct 8)
Validation: 28 days (Oct 9 - Nov 6)

Features: 207 total (175 entry + 27 exit + 5 base)
Labels: Entry (3%/30min), Exit (patience-based/40min)

Models: Enhanced 5-Fold Cross-Validation with TimeSeriesSplit
- LONG Entry: 85 features
- SHORT Entry: 79 features
- LONG Exit: 27 features
- SHORT Exit: 27 features

Created: 2025-11-06 17:02 KST
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import pickle
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Paths
DATA_DIR = PROJECT_ROOT / "data" / "features"
LABELS_DIR = PROJECT_ROOT / "data" / "labels"
MODELS_DIR = PROJECT_ROOT / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# Input files
FEATURES_FILE = DATA_DIR / "BTCUSDT_5m_features_90days_complete_20251106_164542.csv"
ENTRY_LABELS_FILE = LABELS_DIR / "entry_labels_90days_5min_relaxed_20251106_170658.csv"
EXIT_LABELS_FILE = LABELS_DIR / "exit_labels_90days_5min_relaxed_20251106_170658.csv"

# Validation split (61 days training + 28 days validation)
VALIDATION_START = "2025-10-09"

# Feature groups (from production models)
LONG_ENTRY_FEATURES = [
    'close', 'volume', 'rsi_14', 'macd', 'macd_signal', 'macd_hist',
    'bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'atr_14',
    'ema_12', 'ema_26', 'sma_20', 'sma_50',
    'volume_sma_20', 'obv', 'mfi_14',
    'stoch_k', 'stoch_d', 'cci_20', 'roc_10', 'willr_14',
    'adx_14', 'plus_di', 'minus_di',
    'price_change_pct', 'volume_change_pct',
    'high_low_pct', 'close_open_pct',
    'upper_shadow', 'lower_shadow', 'body_size',
    'momentum_10', 'momentum_20',
    'volatility_20', 'volatility_50',
    'price_to_sma20', 'price_to_sma50', 'volume_ratio',
    'rsi_sma', 'macd_signal_diff',
    'bb_position', 'price_momentum',
    'volume_price_trend', 'money_flow_ratio',
    # Long-term features
    'sma_200', 'ema_200', 'price_to_sma200',
    'trend_strength_200', 'volatility_200',
    'volume_trend_200', 'momentum_200',
    # Volume Profile & VWAP
    'vwap', 'vwap_distance', 'vp_support', 'vp_resistance',
    'vp_support_strength', 'vp_resistance_strength', 'vp_value_area',
    # Engineered ratios
    'rsi_bb_combo', 'volume_momentum_ratio', 'trend_volatility_ratio',
    'macd_rsi_sync', 'bb_squeeze_ratio', 'momentum_consistency',
    'support_distance_ratio', 'resistance_distance_ratio',
    'value_area_position', 'vwap_trend_alignment', 'price_efficiency_ratio',
    'volume_distribution_score', 'liquidity_concentration',
    'microstructure_quality', 'trend_confidence_score',
    'mean_reversion_signal', 'breakout_probability', 'regime_stability',
    # Additional features (if available)
    'price_acceleration', 'volume_acceleration', 'trend_consistency',
    'vwap_std', 'price_volume_correlation', 'regime_transition_score'
]

SHORT_ENTRY_FEATURES = [
    'close', 'volume', 'rsi_14', 'macd', 'macd_signal', 'macd_hist',
    'bb_upper', 'bb_middle', 'bb_lower', 'bb_width', 'atr_14',
    'ema_12', 'ema_26', 'sma_20', 'sma_50',
    'volume_sma_20', 'obv', 'mfi_14',
    'stoch_k', 'stoch_d', 'cci_20', 'roc_10', 'willr_14',
    'adx_14', 'plus_di', 'minus_di',
    'price_change_pct', 'volume_change_pct',
    'high_low_pct', 'close_open_pct',
    'upper_shadow', 'lower_shadow', 'body_size',
    'momentum_10', 'momentum_20',
    'volatility_20', 'volatility_50',
    'price_to_sma20', 'price_to_sma50', 'volume_ratio',
    'rsi_sma', 'macd_signal_diff',
    'bb_position', 'price_momentum',
    'volume_price_trend', 'money_flow_ratio',
    # Long-term features
    'sma_200', 'ema_200', 'price_to_sma200',
    'trend_strength_200', 'volatility_200',
    'volume_trend_200', 'momentum_200',
    # Volume Profile & VWAP
    'vwap', 'vwap_distance', 'vp_support', 'vp_resistance',
    'vp_support_strength', 'vp_resistance_strength', 'vp_value_area',
    # Engineered ratios
    'rsi_bb_combo', 'volume_momentum_ratio', 'trend_volatility_ratio',
    'macd_rsi_sync', 'bb_squeeze_ratio', 'momentum_consistency',
    'support_distance_ratio', 'resistance_distance_ratio',
    'value_area_position', 'vwap_trend_alignment',
    # SHORT-specific features
    'short_momentum_edge', 'short_volume_pattern', 'short_volatility_advantage',
    'short_trend_quality', 'short_mean_reversion', 'short_breakout_signal',
    'short_liquidity_edge', 'short_regime_fit', 'short_timing_score',
    'short_composite_signal'
]

EXIT_FEATURES = [
    'close', 'volume', 'rsi_14', 'macd', 'bb_width', 'atr_14',
    'volume_sma_20', 'adx_14', 'price_change_pct', 'volume_change_pct',
    'volatility_20', 'price_to_sma20', 'bb_position',
    'volume_surge', 'price_acceleration', 'volatility_spike',
    'trend_exhaustion', 'volume_divergence', 'support_resistance_proximity',
    'momentum_reversal', 'liquidity_dryup', 'regime_shift_signal',
    'profit_taking_pressure', 'position_efficiency', 'time_decay_factor',
    'risk_reward_deterioration', 'exit_urgency_score', 'hold_quality_score'
]

# XGBoost parameters (Enhanced 5-Fold CV)
ENTRY_PARAMS = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'max_depth': 6,
    'learning_rate': 0.01,
    'n_estimators': 500,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 5,
    'gamma': 0.1,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'random_state': 42,
    'n_jobs': -1,
    'early_stopping_rounds': 50
}

EXIT_PARAMS = {
    'objective': 'binary:logistic',
    'eval_metric': 'logloss',
    'max_depth': 5,
    'learning_rate': 0.01,
    'n_estimators': 300,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 3,
    'gamma': 0.05,
    'reg_alpha': 0.05,
    'reg_lambda': 0.5,
    'random_state': 42,
    'n_jobs': -1,
    'early_stopping_rounds': 30
}

print("=" * 80)
print("RETRAINING ALL 4 MODELS ON 90-DAY 5-MIN COMPLETE DATASET")
print("=" * 80)
print()
print(f"📂 Features: {FEATURES_FILE.name}")
print(f"📊 Entry Labels: {ENTRY_LABELS_FILE.name}")
print(f"🚪 Exit Labels: {EXIT_LABELS_FILE.name}")
print()

# ============================================================================
# LOAD DATA
# ============================================================================

print("=" * 80)
print("LOADING DATA")
print("=" * 80)
print()

print("📖 Loading features...")
df_features = pd.read_csv(FEATURES_FILE)
df_features['timestamp'] = pd.to_datetime(df_features['timestamp'])
print(f"   Rows: {len(df_features):,}")
print(f"   Features: {len(df_features.columns) - 1}")  # Exclude timestamp
print()

print("📖 Loading entry labels...")
df_entry_labels = pd.read_csv(ENTRY_LABELS_FILE)
df_entry_labels['timestamp'] = pd.to_datetime(df_entry_labels['timestamp'])
print(f"   Rows: {len(df_entry_labels):,}")
print(f"   LONG entries: {df_entry_labels['long_entry_label'].sum():,} ({df_entry_labels['long_entry_label'].sum() / len(df_entry_labels) * 100:.2f}%)")
print(f"   SHORT entries: {df_entry_labels['short_entry_label'].sum():,} ({df_entry_labels['short_entry_label'].sum() / len(df_entry_labels) * 100:.2f}%)")
print()

print("📖 Loading exit labels...")
df_exit_labels = pd.read_csv(EXIT_LABELS_FILE)
df_exit_labels['timestamp'] = pd.to_datetime(df_exit_labels['timestamp'])
print(f"   Rows: {len(df_exit_labels):,}")
print(f"   LONG exits: {df_exit_labels['long_exit_label'].sum():,} ({df_exit_labels['long_exit_label'].sum() / len(df_exit_labels) * 100:.2f}%)")
print(f"   SHORT exits: {df_exit_labels['short_exit_label'].sum():,} ({df_exit_labels['short_exit_label'].sum() / len(df_exit_labels) * 100:.2f}%)")
print()

# Merge data
df = df_features.merge(df_entry_labels[['timestamp', 'long_entry_label', 'short_entry_label']], on='timestamp')
df = df.merge(df_exit_labels[['timestamp', 'long_exit_label', 'short_exit_label']], on='timestamp')

print(f"✅ Data merged: {len(df):,} rows")
print()

# ============================================================================
# TRAIN/VALIDATION SPLIT
# ============================================================================

print("=" * 80)
print("TRAIN/VALIDATION SPLIT")
print("=" * 80)
print()

train_mask = df['timestamp'] < VALIDATION_START
train_df = df[train_mask].copy()
val_df = df[~train_mask].copy()

print(f"📚 Training Set:")
print(f"   Period: {train_df['timestamp'].min()} to {train_df['timestamp'].max()}")
print(f"   Rows: {len(train_df):,}")
print(f"   Days: {(train_df['timestamp'].max() - train_df['timestamp'].min()).days}")
print()

print(f"✅ Validation Set:")
print(f"   Period: {val_df['timestamp'].min()} to {val_df['timestamp'].max()}")
print(f"   Rows: {len(val_df):,}")
print(f"   Days: {(val_df['timestamp'].max() - val_df['timestamp'].min()).days}")
print()

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def train_model_with_cv(X_train, y_train, X_val, y_val, params, model_name):
    """Train model with Enhanced 5-Fold CV"""

    print(f"🔧 Training {model_name}...")
    print(f"   Training samples: {len(X_train):,}")
    print(f"   Positive labels: {y_train.sum():,} ({y_train.sum() / len(y_train) * 100:.2f}%)")
    print(f"   Validation samples: {len(X_val):,}")
    print()

    # TimeSeriesSplit for CV
    tscv = TimeSeriesSplit(n_splits=5)

    # Track best model
    best_model = None
    best_score = 0

    print("   Enhanced 5-Fold Cross-Validation:")
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train), 1):
        X_fold_train = X_train.iloc[train_idx]
        y_fold_train = y_train.iloc[train_idx]
        X_fold_val = X_train.iloc[val_idx]
        y_fold_val = y_train.iloc[val_idx]

        # Train fold model
        model = xgb.XGBClassifier(**params)
        model.fit(
            X_fold_train, y_fold_train,
            eval_set=[(X_fold_val, y_fold_val)],
            verbose=False
        )

        # Evaluate on fold validation
        val_preds = model.predict_proba(X_fold_val)[:, 1]
        val_score = ((val_preds > 0.5) == y_fold_val).mean()

        print(f"      Fold {fold}: Accuracy = {val_score:.4f}")

        # Track best
        if val_score > best_score:
            best_score = val_score
            best_model = model

    print(f"   ✅ Best CV Score: {best_score:.4f}")
    print()

    # Final evaluation on validation set
    val_preds = best_model.predict(X_val)
    val_proba = best_model.predict_proba(X_val)[:, 1]

    print(f"   📊 Validation Set Performance:")
    print(f"      Accuracy: {(val_preds == y_val).mean():.4f}")
    print(f"      Positive predictions: {val_preds.sum():,} ({val_preds.sum() / len(val_preds) * 100:.2f}%)")
    print(f"      Mean probability: {val_proba.mean():.4f}")
    print(f"      Max probability: {val_proba.max():.4f}")
    print()

    # Confusion matrix
    cm = confusion_matrix(y_val, val_preds)
    print(f"   Confusion Matrix:")
    print(f"      TN: {cm[0,0]:,}  FP: {cm[0,1]:,}")
    print(f"      FN: {cm[1,0]:,}  TP: {cm[1,1]:,}")
    print()

    return best_model

timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')

# ============================================================================
# TRAIN LONG ENTRY MODEL
# ============================================================================

print("=" * 80)
print("1/4: TRAINING LONG ENTRY MODEL")
print("=" * 80)
print()

# Select features (only use features that exist)
long_entry_features_available = [f for f in LONG_ENTRY_FEATURES if f in df.columns]
print(f"📋 Features: {len(long_entry_features_available)} (requested: {len(LONG_ENTRY_FEATURES)})")
print()

X_train_long = train_df[long_entry_features_available]
y_train_long = train_df['long_entry_label']
X_val_long = val_df[long_entry_features_available]
y_val_long = val_df['long_entry_label']

# Normalize
scaler_long_entry = StandardScaler()
X_train_long_scaled = pd.DataFrame(
    scaler_long_entry.fit_transform(X_train_long),
    columns=X_train_long.columns,
    index=X_train_long.index
)
X_val_long_scaled = pd.DataFrame(
    scaler_long_entry.transform(X_val_long),
    columns=X_val_long.columns,
    index=X_val_long.index
)

# Train
model_long_entry = train_model_with_cv(
    X_train_long_scaled, y_train_long,
    X_val_long_scaled, y_val_long,
    ENTRY_PARAMS,
    "LONG Entry"
)

# Save
model_file = MODELS_DIR / f"xgboost_long_entry_90days_5min_{timestamp_str}.pkl"
with open(model_file, 'wb') as f:
    pickle.dump(model_long_entry, f)
print(f"💾 Model saved: {model_file.name}")

scaler_file = MODELS_DIR / f"xgboost_long_entry_90days_5min_{timestamp_str}_scaler.pkl"
with open(scaler_file, 'wb') as f:
    pickle.dump(scaler_long_entry, f)
print(f"💾 Scaler saved: {scaler_file.name}")

features_file = MODELS_DIR / f"xgboost_long_entry_90days_5min_{timestamp_str}_features.txt"
with open(features_file, 'w') as f:
    f.write('\n'.join(long_entry_features_available))
print(f"💾 Features saved: {features_file.name}")
print()

# ============================================================================
# TRAIN SHORT ENTRY MODEL
# ============================================================================

print("=" * 80)
print("2/4: TRAINING SHORT ENTRY MODEL")
print("=" * 80)
print()

# Select features
short_entry_features_available = [f for f in SHORT_ENTRY_FEATURES if f in df.columns]
print(f"📋 Features: {len(short_entry_features_available)} (requested: {len(SHORT_ENTRY_FEATURES)})")
print()

X_train_short = train_df[short_entry_features_available]
y_train_short = train_df['short_entry_label']
X_val_short = val_df[short_entry_features_available]
y_val_short = val_df['short_entry_label']

# Normalize
scaler_short_entry = StandardScaler()
X_train_short_scaled = pd.DataFrame(
    scaler_short_entry.fit_transform(X_train_short),
    columns=X_train_short.columns,
    index=X_train_short.index
)
X_val_short_scaled = pd.DataFrame(
    scaler_short_entry.transform(X_val_short),
    columns=X_val_short.columns,
    index=X_val_short.index
)

# Train
model_short_entry = train_model_with_cv(
    X_train_short_scaled, y_train_short,
    X_val_short_scaled, y_val_short,
    ENTRY_PARAMS,
    "SHORT Entry"
)

# Save
model_file = MODELS_DIR / f"xgboost_short_entry_90days_5min_{timestamp_str}.pkl"
with open(model_file, 'wb') as f:
    pickle.dump(model_short_entry, f)
print(f"💾 Model saved: {model_file.name}")

scaler_file = MODELS_DIR / f"xgboost_short_entry_90days_5min_{timestamp_str}_scaler.pkl"
with open(scaler_file, 'wb') as f:
    pickle.dump(scaler_short_entry, f)
print(f"💾 Scaler saved: {scaler_file.name}")

features_file = MODELS_DIR / f"xgboost_short_entry_90days_5min_{timestamp_str}_features.txt"
with open(features_file, 'w') as f:
    f.write('\n'.join(short_entry_features_available))
print(f"💾 Features saved: {features_file.name}")
print()

# ============================================================================
# TRAIN LONG EXIT MODEL
# ============================================================================

print("=" * 80)
print("3/4: TRAINING LONG EXIT MODEL")
print("=" * 80)
print()

# Select features
exit_features_available = [f for f in EXIT_FEATURES if f in df.columns]
print(f"📋 Features: {len(exit_features_available)} (requested: {len(EXIT_FEATURES)})")
print()

X_train_long_exit = train_df[exit_features_available]
y_train_long_exit = train_df['long_exit_label']
X_val_long_exit = val_df[exit_features_available]
y_val_long_exit = val_df['long_exit_label']

# Normalize
scaler_long_exit = StandardScaler()
X_train_long_exit_scaled = pd.DataFrame(
    scaler_long_exit.fit_transform(X_train_long_exit),
    columns=X_train_long_exit.columns,
    index=X_train_long_exit.index
)
X_val_long_exit_scaled = pd.DataFrame(
    scaler_long_exit.transform(X_val_long_exit),
    columns=X_val_long_exit.columns,
    index=X_val_long_exit.index
)

# Train
model_long_exit = train_model_with_cv(
    X_train_long_exit_scaled, y_train_long_exit,
    X_val_long_exit_scaled, y_val_long_exit,
    EXIT_PARAMS,
    "LONG Exit"
)

# Save
model_file = MODELS_DIR / f"xgboost_long_exit_90days_5min_{timestamp_str}.pkl"
with open(model_file, 'wb') as f:
    pickle.dump(model_long_exit, f)
print(f"💾 Model saved: {model_file.name}")

scaler_file = MODELS_DIR / f"xgboost_long_exit_90days_5min_{timestamp_str}_scaler.pkl"
with open(scaler_file, 'wb') as f:
    pickle.dump(scaler_long_exit, f)
print(f"💾 Scaler saved: {scaler_file.name}")

features_file = MODELS_DIR / f"xgboost_long_exit_90days_5min_{timestamp_str}_features.txt"
with open(features_file, 'w') as f:
    f.write('\n'.join(exit_features_available))
print(f"💾 Features saved: {features_file.name}")
print()

# ============================================================================
# TRAIN SHORT EXIT MODEL
# ============================================================================

print("=" * 80)
print("4/4: TRAINING SHORT EXIT MODEL")
print("=" * 80)
print()

X_train_short_exit = train_df[exit_features_available]
y_train_short_exit = train_df['short_exit_label']
X_val_short_exit = val_df[exit_features_available]
y_val_short_exit = val_df['short_exit_label']

# Normalize
scaler_short_exit = StandardScaler()
X_train_short_exit_scaled = pd.DataFrame(
    scaler_short_exit.fit_transform(X_train_short_exit),
    columns=X_train_short_exit.columns,
    index=X_train_short_exit.index
)
X_val_short_exit_scaled = pd.DataFrame(
    scaler_short_exit.transform(X_val_short_exit),
    columns=X_val_short_exit.columns,
    index=X_val_short_exit.index
)

# Train
model_short_exit = train_model_with_cv(
    X_train_short_exit_scaled, y_train_short_exit,
    X_val_short_exit_scaled, y_val_short_exit,
    EXIT_PARAMS,
    "SHORT Exit"
)

# Save
model_file = MODELS_DIR / f"xgboost_short_exit_90days_5min_{timestamp_str}.pkl"
with open(model_file, 'wb') as f:
    pickle.dump(model_short_exit, f)
print(f"💾 Model saved: {model_file.name}")

scaler_file = MODELS_DIR / f"xgboost_short_exit_90days_5min_{timestamp_str}_scaler.pkl"
with open(scaler_file, 'wb') as f:
    pickle.dump(scaler_short_exit, f)
print(f"💾 Scaler saved: {scaler_file.name}")

features_file = MODELS_DIR / f"xgboost_short_exit_90days_5min_{timestamp_str}_features.txt"
with open(features_file, 'w') as f:
    f.write('\n'.join(exit_features_available))
print(f"💾 Features saved: {features_file.name}")
print()

# ============================================================================
# SUMMARY
# ============================================================================

print("=" * 80)
print("TRAINING COMPLETE")
print("=" * 80)
print()

print(f"✅ All 4 models trained with Enhanced 5-Fold CV")
print(f"   Timestamp: {timestamp_str}")
print()

print(f"📊 Training Data:")
print(f"   Period: {train_df['timestamp'].min()} to {train_df['timestamp'].max()}")
print(f"   Rows: {len(train_df):,}")
print(f"   Days: {(train_df['timestamp'].max() - train_df['timestamp'].min()).days}")
print()

print(f"✅ Validation Data:")
print(f"   Period: {val_df['timestamp'].min()} to {val_df['timestamp'].max()}")
print(f"   Rows: {len(val_df):,}")
print(f"   Days: {(val_df['timestamp'].max() - val_df['timestamp'].min()).days}")
print()

print(f"📦 Models Saved:")
print(f"   - xgboost_long_entry_90days_5min_{timestamp_str}.pkl")
print(f"   - xgboost_short_entry_90days_5min_{timestamp_str}.pkl")
print(f"   - xgboost_long_exit_90days_5min_{timestamp_str}.pkl")
print(f"   - xgboost_short_exit_90days_5min_{timestamp_str}.pkl")
print()

print("=" * 80)
print("NEXT STEPS")
print("=" * 80)
print()
print("1. Run backtest on 28-day validation period (Oct 9 - Nov 6)")
print("2. Compare performance vs 52-day models (current production)")
print("3. Decision: Deploy if superior, keep 52-day if not")
print()
print("✅ Training script complete!")
