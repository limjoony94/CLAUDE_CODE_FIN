"""
Retrain Entry and Exit Models with Reduced Features
====================================================

Removes zero-importance features identified from feature importance analysis.

Zero-Importance Features Removed (22 total):
  - Candlestick patterns: doji, hammer, shooting_star
  - Divergences: macd_bullish/bearish_divergence, rsi_bullish/bearish/divergence
  - VWAP signals: vwap_above, vwap_overbought, vwap_oversold
  - Volume Profile: vp_strong_buy_pressure
  - Position indicators: bb_position, near_resistance, near_support
  - Trend: downtrend_confirmed, support_breakdown
  - Pressure: strong_selling_pressure
  - Ratios: volume_decline_ratio, volatility_asymmetry
  - Technical: macd_histogram_slope, rsi_overbought (SHORT Exit only)

Expected Impact:
  - Reduced overfitting (removed noise features)
  - Better generalization to new data
  - Faster training and prediction
  - More interpretable models
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime
import joblib
import xgboost as xgb
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiments.calculate_all_features_enhanced_v2 import calculate_all_features_enhanced_v2
from scripts.experiments.retrain_exit_models_opportunity_gating import prepare_exit_features

# =============================================================================
# ZERO-IMPORTANCE FEATURES TO REMOVE
# =============================================================================

ZERO_IMPORTANCE_FEATURES = {
    'LONG_Entry': [
        'macd_bullish_divergence',
        'rsi_bearish_divergence',
        'rsi_bullish_divergence',
        'macd_bearish_divergence',
        'strong_selling_pressure',
        'hammer',
        'shooting_star',
        'doji',
        'vp_strong_buy_pressure',
        'vwap_above',
        'vwap_overbought',
        'vwap_oversold'
    ],
    'SHORT_Entry': [
        'support_breakdown',
        'volume_decline_ratio',
        'volatility_asymmetry',
        'near_resistance',
        'downtrend_confirmed'
    ],
    'LONG_Exit': [
        'rsi_divergence',
        'macd_histogram_slope',
        'near_resistance',
        'near_support',
        'bb_position'
    ],
    'SHORT_Exit': [
        'rsi_divergence',
        'rsi_overbought',
        'macd_histogram_slope',
        'near_resistance',
        'near_support',
        'bb_position'
    ]
}

MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data" / "historical"

print("=" * 80)
print("RETRAINING MODELS WITH REDUCED FEATURES")
print("=" * 80)
print(f"\nZero-importance features to remove:")
for model_name, features in ZERO_IMPORTANCE_FEATURES.items():
    print(f"  {model_name}: {len(features)} features")
print()

# =============================================================================
# LOAD AND PREPARE DATA
# =============================================================================

print("Loading data...")
df = pd.read_csv(DATA_DIR / "BTCUSDT_5m_updated.csv")
df['timestamp'] = pd.to_datetime(df['timestamp'])
print(f"  Total candles: {len(df)}")

print("\nCalculating features...")
df_features = calculate_all_features_enhanced_v2(df)
print(f"  Features calculated: {len(df_features)} rows")

print("\nPreparing exit features...")
df_features = prepare_exit_features(df_features)
print(f"  Exit features ready: {len(df_features)} rows")

# =============================================================================
# ENTRY MODELS - TRADE OUTCOME LABELS
# =============================================================================

print("\n" + "=" * 80)
print("TRAINING ENTRY MODELS (REDUCED FEATURES)")
print("=" * 80)

# Load trade outcome labels
print("\nLoading trade outcome labels...")
labels_df = pd.read_csv(PROJECT_ROOT / "results" / "trade_outcome_labels_full_20251018_233120.csv")
labels_df['timestamp'] = pd.to_datetime(labels_df['timestamp'])
print(f"  Labels loaded: {len(labels_df)} rows")

# Merge with features
df_entry = df_features.merge(labels_df[['timestamp', 'long_good_entry', 'short_good_entry']],
                              on='timestamp', how='inner')
df_entry = df_entry.dropna()
print(f"  Merged data: {len(df_entry)} rows")

# Load original feature lists
long_entry_features_path = MODELS_DIR / "xgboost_long_entry_enhanced_20251024_012445_features.txt"
with open(long_entry_features_path, 'r') as f:
    long_entry_features_full = [line.strip() for line in f.readlines() if line.strip()]

short_entry_features_path = MODELS_DIR / "xgboost_short_entry_enhanced_20251024_012445_features.txt"
with open(short_entry_features_path, 'r') as f:
    short_entry_features_full = [line.strip() for line in f.readlines() if line.strip()]

# Remove zero-importance features
long_entry_features = [f for f in long_entry_features_full
                       if f not in ZERO_IMPORTANCE_FEATURES['LONG_Entry']]
short_entry_features = [f for f in short_entry_features_full
                        if f not in ZERO_IMPORTANCE_FEATURES['SHORT_Entry']]

print(f"\n📊 Feature Reduction:")
print(f"  LONG Entry: {len(long_entry_features_full)} → {len(long_entry_features)} features (-{len(long_entry_features_full) - len(long_entry_features)})")
print(f"  SHORT Entry: {len(short_entry_features_full)} → {len(short_entry_features)} features (-{len(short_entry_features_full) - len(short_entry_features)})")

# Train LONG Entry Model
print("\n" + "-" * 80)
print("LONG ENTRY MODEL")
print("-" * 80)

X_long = df_entry[long_entry_features].values
y_long = df_entry['long_good_entry'].values

scaler_long = StandardScaler()
X_long_scaled = scaler_long.fit_transform(X_long)

tscv = TimeSeriesSplit(n_splits=5)
long_scores = []

for fold, (train_idx, val_idx) in enumerate(tscv.split(X_long_scaled), 1):
    X_train, X_val = X_long_scaled[train_idx], X_long_scaled[val_idx]
    y_train, y_val = y_long[train_idx], y_long[val_idx]

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.01,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        gamma=0.1,
        random_state=42,
        eval_metric='logloss',
        early_stopping_rounds=50
    )

    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              verbose=False)

    y_pred = model.predict(X_val)
    y_pred_proba = model.predict_proba(X_val)[:, 1]

    from sklearn.metrics import accuracy_score, roc_auc_score
    acc = accuracy_score(y_val, y_pred)
    auc = roc_auc_score(y_val, y_pred_proba)

    long_scores.append({'fold': fold, 'accuracy': acc, 'auc': auc})
    print(f"  Fold {fold}: Accuracy={acc:.4f}, AUC={auc:.4f}")

# Train final model on all data
print("\nTraining final LONG Entry model...")
long_entry_model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.01,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    gamma=0.1,
    random_state=42
)
long_entry_model.fit(X_long_scaled, y_long)

# Train SHORT Entry Model
print("\n" + "-" * 80)
print("SHORT ENTRY MODEL")
print("-" * 80)

X_short = df_entry[short_entry_features].values
y_short = df_entry['short_good_entry'].values

scaler_short = StandardScaler()
X_short_scaled = scaler_short.fit_transform(X_short)

short_scores = []

for fold, (train_idx, val_idx) in enumerate(tscv.split(X_short_scaled), 1):
    X_train, X_val = X_short_scaled[train_idx], X_short_scaled[val_idx]
    y_train, y_val = y_short[train_idx], y_short[val_idx]

    model = xgb.XGBClassifier(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.01,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        gamma=0.1,
        random_state=42,
        eval_metric='logloss',
        early_stopping_rounds=50
    )

    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              verbose=False)

    y_pred = model.predict(X_val)
    y_pred_proba = model.predict_proba(X_val)[:, 1]

    acc = accuracy_score(y_val, y_pred)
    auc = roc_auc_score(y_val, y_pred_proba)

    short_scores.append({'fold': fold, 'accuracy': acc, 'auc': auc})
    print(f"  Fold {fold}: Accuracy={acc:.4f}, AUC={auc:.4f}")

# Train final model on all data
print("\nTraining final SHORT Entry model...")
short_entry_model = xgb.XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.01,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=3,
    gamma=0.1,
    random_state=42
)
short_entry_model.fit(X_short_scaled, y_short)

# =============================================================================
# EXIT MODELS - OPPORTUNITY GATING LABELS
# =============================================================================

print("\n" + "=" * 80)
print("TRAINING EXIT MODELS (REDUCED FEATURES)")
print("=" * 80)

# Load exit labels
print("\nLoading exit labels...")
exit_labels_df = pd.read_csv(PROJECT_ROOT / "results" / "exit_model_labels_oppgating_improved_20251017_151624.csv")
exit_labels_df['timestamp'] = pd.to_datetime(exit_labels_df['timestamp'])
print(f"  Exit labels loaded: {len(exit_labels_df)} rows")

# Merge with features
df_exit = df_features.merge(exit_labels_df[['timestamp', 'long_should_exit', 'short_should_exit']],
                            on='timestamp', how='inner')
df_exit = df_exit.dropna()
print(f"  Merged data: {len(df_exit)} rows")

# Load original feature lists
long_exit_features_path = MODELS_DIR / "xgboost_long_exit_oppgating_improved_20251024_043527_features.txt"
with open(long_exit_features_path, 'r') as f:
    long_exit_features_full = [line.strip() for line in f.readlines() if line.strip()]

short_exit_features_path = MODELS_DIR / "xgboost_short_exit_oppgating_improved_20251024_044510_features.txt"
with open(short_exit_features_path, 'r') as f:
    short_exit_features_full = [line.strip() for line in f.readlines() if line.strip()]

# Remove zero-importance features
long_exit_features = [f for f in long_exit_features_full
                      if f not in ZERO_IMPORTANCE_FEATURES['LONG_Exit']]
short_exit_features = [f for f in short_exit_features_full
                       if f not in ZERO_IMPORTANCE_FEATURES['SHORT_Exit']]

print(f"\n📊 Feature Reduction:")
print(f"  LONG Exit: {len(long_exit_features_full)} → {len(long_exit_features)} features (-{len(long_exit_features_full) - len(long_exit_features)})")
print(f"  SHORT Exit: {len(short_exit_features_full)} → {len(short_exit_features)} features (-{len(short_exit_features_full) - len(short_exit_features)})")

# Train LONG Exit Model
print("\n" + "-" * 80)
print("LONG EXIT MODEL")
print("-" * 80)

X_long_exit = df_exit[long_exit_features].values
y_long_exit = df_exit['long_should_exit'].values

scaler_long_exit = StandardScaler()
X_long_exit_scaled = scaler_long_exit.fit_transform(X_long_exit)

long_exit_scores = []

for fold, (train_idx, val_idx) in enumerate(tscv.split(X_long_exit_scaled), 1):
    X_train, X_val = X_long_exit_scaled[train_idx], X_long_exit_scaled[val_idx]
    y_train, y_val = y_long_exit[train_idx], y_long_exit[val_idx]

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.02,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=2,
        gamma=0.05,
        random_state=42,
        eval_metric='logloss',
        early_stopping_rounds=30
    )

    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              verbose=False)

    y_pred = model.predict(X_val)
    y_pred_proba = model.predict_proba(X_val)[:, 1]

    acc = accuracy_score(y_val, y_pred)
    auc = roc_auc_score(y_val, y_pred_proba)

    long_exit_scores.append({'fold': fold, 'accuracy': acc, 'auc': auc})
    print(f"  Fold {fold}: Accuracy={acc:.4f}, AUC={auc:.4f}")

# Train final model on all data
print("\nTraining final LONG Exit model...")
long_exit_model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.02,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=2,
    gamma=0.05,
    random_state=42
)
long_exit_model.fit(X_long_exit_scaled, y_long_exit)

# Train SHORT Exit Model
print("\n" + "-" * 80)
print("SHORT EXIT MODEL")
print("-" * 80)

X_short_exit = df_exit[short_exit_features].values
y_short_exit = df_exit['short_should_exit'].values

scaler_short_exit = StandardScaler()
X_short_exit_scaled = scaler_short_exit.fit_transform(X_short_exit)

short_exit_scores = []

for fold, (train_idx, val_idx) in enumerate(tscv.split(X_short_exit_scaled), 1):
    X_train, X_val = X_short_exit_scaled[train_idx], X_short_exit_scaled[val_idx]
    y_train, y_val = y_short_exit[train_idx], y_short_exit[val_idx]

    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.02,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=2,
        gamma=0.05,
        random_state=42,
        eval_metric='logloss',
        early_stopping_rounds=30
    )

    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              verbose=False)

    y_pred = model.predict(X_val)
    y_pred_proba = model.predict_proba(X_val)[:, 1]

    acc = accuracy_score(y_val, y_pred)
    auc = roc_auc_score(y_val, y_pred_proba)

    short_exit_scores.append({'fold': fold, 'accuracy': acc, 'auc': auc})
    print(f"  Fold {fold}: Accuracy={acc:.4f}, AUC={auc:.4f}")

# Train final model on all data
print("\nTraining final SHORT Exit model...")
short_exit_model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.02,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=2,
    gamma=0.05,
    random_state=42
)
short_exit_model.fit(X_short_exit_scaled, y_short_exit)

# =============================================================================
# SAVE MODELS
# =============================================================================

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

print("\n" + "=" * 80)
print("SAVING REDUCED-FEATURE MODELS")
print("=" * 80)

# Entry Models
joblib.dump(long_entry_model, MODELS_DIR / f"xgboost_long_entry_reduced_{timestamp}.pkl")
joblib.dump(scaler_long, MODELS_DIR / f"xgboost_long_entry_reduced_{timestamp}_scaler.pkl")
with open(MODELS_DIR / f"xgboost_long_entry_reduced_{timestamp}_features.txt", 'w') as f:
    f.write('\n'.join(long_entry_features))
print(f"✅ LONG Entry saved: {len(long_entry_features)} features")

joblib.dump(short_entry_model, MODELS_DIR / f"xgboost_short_entry_reduced_{timestamp}.pkl")
joblib.dump(scaler_short, MODELS_DIR / f"xgboost_short_entry_reduced_{timestamp}_scaler.pkl")
with open(MODELS_DIR / f"xgboost_short_entry_reduced_{timestamp}_features.txt", 'w') as f:
    f.write('\n'.join(short_entry_features))
print(f"✅ SHORT Entry saved: {len(short_entry_features)} features")

# Exit Models
joblib.dump(long_exit_model, MODELS_DIR / f"xgboost_long_exit_reduced_{timestamp}.pkl")
joblib.dump(scaler_long_exit, MODELS_DIR / f"xgboost_long_exit_reduced_{timestamp}_scaler.pkl")
with open(MODELS_DIR / f"xgboost_long_exit_reduced_{timestamp}_features.txt", 'w') as f:
    f.write('\n'.join(long_exit_features))
print(f"✅ LONG Exit saved: {len(long_exit_features)} features")

joblib.dump(short_exit_model, MODELS_DIR / f"xgboost_short_exit_reduced_{timestamp}.pkl")
joblib.dump(scaler_short_exit, MODELS_DIR / f"xgboost_short_exit_reduced_{timestamp}_scaler.pkl")
with open(MODELS_DIR / f"xgboost_short_exit_reduced_{timestamp}_features.txt", 'w') as f:
    f.write('\n'.join(short_exit_features))
print(f"✅ SHORT Exit saved: {len(short_exit_features)} features")

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "=" * 80)
print("TRAINING SUMMARY")
print("=" * 80)

print(f"\n📊 FEATURE REDUCTION:")
print(f"  LONG Entry: 85 → {len(long_entry_features)} features (-{85 - len(long_entry_features)})")
print(f"  SHORT Entry: 79 → {len(short_entry_features)} features (-{79 - len(short_entry_features)})")
print(f"  LONG Exit: 27 → {len(long_exit_features)} features (-{27 - len(long_exit_features)})")
print(f"  SHORT Exit: 27 → {len(short_exit_features)} features (-{27 - len(short_exit_features)})")
print(f"  Total: 218 → {len(long_entry_features) + len(short_entry_features) + len(long_exit_features) + len(short_exit_features)} features (-{218 - (len(long_entry_features) + len(short_entry_features) + len(long_exit_features) + len(short_exit_features))})")

print(f"\n📈 CROSS-VALIDATION PERFORMANCE:")

long_avg_acc = np.mean([s['accuracy'] for s in long_scores])
long_avg_auc = np.mean([s['auc'] for s in long_scores])
print(f"  LONG Entry: Accuracy={long_avg_acc:.4f}, AUC={long_avg_auc:.4f}")

short_avg_acc = np.mean([s['accuracy'] for s in short_scores])
short_avg_auc = np.mean([s['auc'] for s in short_scores])
print(f"  SHORT Entry: Accuracy={short_avg_acc:.4f}, AUC={short_avg_auc:.4f}")

long_exit_avg_acc = np.mean([s['accuracy'] for s in long_exit_scores])
long_exit_avg_auc = np.mean([s['auc'] for s in long_exit_scores])
print(f"  LONG Exit: Accuracy={long_exit_avg_acc:.4f}, AUC={long_exit_avg_auc:.4f}")

short_exit_avg_acc = np.mean([s['accuracy'] for s in short_exit_scores])
short_exit_avg_auc = np.mean([s['auc'] for s in short_exit_scores])
print(f"  SHORT Exit: Accuracy={short_exit_avg_acc:.4f}, AUC={short_exit_avg_auc:.4f}")

print(f"\n✅ Models saved with timestamp: {timestamp}")
print(f"\n💡 NEXT STEPS:")
print(f"  1. Run backtest with reduced-feature models")
print(f"  2. Compare performance: reduced vs full features")
print(f"  3. If performance maintained/improved: Deploy to production")
print(f"  4. Expected benefits:")
print(f"     - Reduced overfitting")
print(f"     - Better generalization")
print(f"     - Faster predictions")
print(f"     - More interpretable models")

print("\n" + "=" * 80)
