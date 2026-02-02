"""
Retrain Entry Models: Trade-Outcome Labeling (Full Dataset - Optimized)
========================================================================

Full dataset training with:
1. Improved Risk-Reward criteria (MAE >= -4%, MFE >= 2%)
2. Optimized trade simulator (vectorization + parallelization)
3. 30,517 candles

Expected time: ~20-30 minutes (vs >1.5 hours previously)
"""

import sys
import os
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

from scripts.experiments.calculate_all_features import calculate_all_features
from scripts.experiments.retrain_exit_models_opportunity_gating import prepare_exit_features
from scripts.experiments.trade_simulator_optimized import simulate_trades_optimized

DATA_DIR = PROJECT_ROOT / "data" / "historical"
MODELS_DIR = PROJECT_ROOT / "models"

print("="*80)
print("RETRAIN ENTRY MODELS: TRADE-OUTCOME LABELING (FULL DATASET - OPTIMIZED)")
print("="*80)

print("\n⚡ Using optimized trade simulator with parallelization")
print("   Expected time: 20-30 minutes")
print("   Improved Risk-Reward: MAE >= -4%, MFE >= 2%")

# ============================================================================
# STEP 1: Load and Prepare Data
# ============================================================================

print("\n" + "-"*80)
print("STEP 1: Loading and Preparing Data")
print("-"*80)

# Load data
data_file = DATA_DIR / "BTCUSDT_5m_max.csv"
df = pd.read_csv(data_file)
print(f"\n✅ Loaded {len(df):,} candles")
print(f"   Date range: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")

# Calculate features
print("\nCalculating features...")
df = calculate_all_features(df)
df = prepare_exit_features(df)
print(f"✅ All features calculated ({len(df.columns)} columns)")

# ============================================================================
# STEP 2: Load Exit Models
# ============================================================================

print("\n" + "-"*80)
print("STEP 2: Loading Exit Models for Trade Simulation")
print("-"*80)

# Load LONG Exit model
long_exit_model_path = MODELS_DIR / "xgboost_long_exit_oppgating_improved_20251017_151624.pkl"
with open(long_exit_model_path, 'rb') as f:
    long_exit_model = pickle.load(f)

long_exit_scaler_path = MODELS_DIR / "xgboost_long_exit_oppgating_improved_20251017_151624_scaler.pkl"
long_exit_scaler = joblib.load(long_exit_scaler_path)

long_exit_features_path = MODELS_DIR / "xgboost_long_exit_oppgating_improved_20251017_151624_features.txt"
with open(long_exit_features_path, 'r') as f:
    long_exit_features = [line.strip() for line in f.readlines() if line.strip()]

# Load SHORT Exit model
short_exit_model_path = MODELS_DIR / "xgboost_short_exit_oppgating_improved_20251017_152440.pkl"
with open(short_exit_model_path, 'rb') as f:
    short_exit_model = pickle.load(f)

short_exit_scaler_path = MODELS_DIR / "xgboost_short_exit_oppgating_improved_20251017_152440_scaler.pkl"
short_exit_scaler = joblib.load(short_exit_scaler_path)

short_exit_features_path = MODELS_DIR / "xgboost_short_exit_oppgating_improved_20251017_152440_features.txt"
with open(short_exit_features_path, 'r') as f:
    short_exit_features = [line.strip() for line in f.readlines() if line.strip()]

print(f"  ✅ LONG Exit: {len(long_exit_features)} features")
print(f"  ✅ SHORT Exit: {len(short_exit_features)} features")
print("✅ Trade simulators ready")

# ============================================================================
# STEP 3: Create Trade-Outcome Labels (Optimized)
# ============================================================================

print("\n" + "-"*80)
print("STEP 3: Creating Trade-Outcome Labels (Optimized Simulation)")
print("-"*80)

# Improved Risk-Reward criteria
PROFIT_THRESHOLD = 0.02  # 2% (leveraged)
MAE_THRESHOLD = -0.04  # -4% (leveraged) - More lenient
MFE_THRESHOLD = 0.02  # 2% (leveraged) - More lenient
SCORING_THRESHOLD = 2  # 2 of 3 criteria

def evaluate_trade_quality(trade_result):
    """Evaluate trade result against 2-of-3 criteria"""
    score = 0
    criteria = []

    # Criterion 1: Profitable Trade (>=2%)
    if trade_result['leveraged_pnl_pct'] >= PROFIT_THRESHOLD:
        score += 1
        criteria.append('profitable')

    # Criterion 2: Good Risk-Reward Ratio (IMPROVED)
    if (trade_result['mae'] >= MAE_THRESHOLD and
        trade_result['mfe'] >= MFE_THRESHOLD):
        score += 1
        criteria.append('good_rr')

    # Criterion 3: Efficient Exit (ML Exit works)
    if trade_result['exit_reason'] == 'ml_exit':
        score += 1
        criteria.append('ml_exit')

    return score, criteria

# LONG Entry Labeling
print("\nLONG Entry Labeling (Trade-Outcome Based - Optimized)")
print("  Simulating trades with 4 parallel workers...")
import time
start_time = time.time()

long_trade_results = simulate_trades_optimized(
    df=df,
    exit_model=long_exit_model,
    exit_scaler=long_exit_scaler,
    exit_features=long_exit_features,
    side='LONG',
    ml_exit_threshold=0.70,
    n_workers=4
)

long_elapsed = time.time() - start_time
print(f"  ✅ Simulated {len(long_trade_results):,} LONG entries in {long_elapsed:.1f}s")

# Create LONG labels
long_labels = np.zeros(len(df))
criterion_counts_long = {'profitable': 0, 'good_rr': 0, 'ml_exit': 0}

for i, result in enumerate(long_trade_results):
    entry_idx = 100 + i
    if entry_idx >= len(df):
        break

    score, criteria = evaluate_trade_quality(result)

    for c in criteria:
        criterion_counts_long[c] += 1

    if score >= SCORING_THRESHOLD:
        long_labels[entry_idx] = 1

print(f"\n  LONG Entry Label Statistics:")
print(f"  Total Candles Evaluated: {len(long_trade_results):,}")
print(f"\n  Criterion Met Rates:")
print(f"    1. Profitable (>={PROFIT_THRESHOLD*100:.1f}%): {criterion_counts_long['profitable']:,} ({criterion_counts_long['profitable']/len(long_trade_results)*100:.1f}%)")
print(f"    2. Good Risk-Reward (MAE>={MAE_THRESHOLD*100:.0f}%, MFE>={MFE_THRESHOLD*100:.0f}%): {criterion_counts_long['good_rr']:,} ({criterion_counts_long['good_rr']/len(long_trade_results)*100:.1f}%)")
print(f"    3. Efficient Exit (ML Exit): {criterion_counts_long['ml_exit']:,} ({criterion_counts_long['ml_exit']/len(long_trade_results)*100:.1f}%)")
print(f"\n  Positive Label Rate: {np.sum(long_labels):,.0f}/{len(df):,} ({np.sum(long_labels)/len(df)*100:.1f}%)")
print("  ✅ LONG Entry labels created")

# SHORT Entry Labeling
print("\n\nSHORT Entry Labeling (Trade-Outcome Based - Optimized)")
print("  Simulating trades with 4 parallel workers...")
start_time = time.time()

short_trade_results = simulate_trades_optimized(
    df=df,
    exit_model=short_exit_model,
    exit_scaler=short_exit_scaler,
    exit_features=short_exit_features,
    side='SHORT',
    ml_exit_threshold=0.72,
    n_workers=4
)

short_elapsed = time.time() - start_time
print(f"  ✅ Simulated {len(short_trade_results):,} SHORT entries in {short_elapsed:.1f}s")

# Create SHORT labels
short_labels = np.zeros(len(df))
criterion_counts_short = {'profitable': 0, 'good_rr': 0, 'ml_exit': 0}

for i, result in enumerate(short_trade_results):
    entry_idx = 100 + i
    if entry_idx >= len(df):
        break

    score, criteria = evaluate_trade_quality(result)

    for c in criteria:
        criterion_counts_short[c] += 1

    if score >= SCORING_THRESHOLD:
        short_labels[entry_idx] = 1

print(f"\n  SHORT Entry Label Statistics:")
print(f"  Total Candles Evaluated: {len(short_trade_results):,}")
print(f"\n  Criterion Met Rates:")
print(f"    1. Profitable (>={PROFIT_THRESHOLD*100:.1f}%): {criterion_counts_short['profitable']:,} ({criterion_counts_short['profitable']/len(short_trade_results)*100:.1f}%)")
print(f"    2. Good Risk-Reward (MAE>={MAE_THRESHOLD*100:.0f}%, MFE>={MFE_THRESHOLD*100:.0f}%): {criterion_counts_short['good_rr']:,} ({criterion_counts_short['good_rr']/len(short_trade_results)*100:.1f}%)")
print(f"    3. Efficient Exit (ML Exit): {criterion_counts_short['ml_exit']:,} ({criterion_counts_short['ml_exit']/len(short_trade_results)*100:.1f}%)")
print(f"\n  Positive Label Rate: {np.sum(short_labels):,.0f}/{len(df):,} ({np.sum(short_labels)/len(df)*100:.1f}%)")
print("  ✅ SHORT Entry labels created")

total_simulation_time = long_elapsed + short_elapsed
print(f"\n⚡ Total Simulation Time: {total_simulation_time:.1f}s ({total_simulation_time/60:.1f} minutes)")

# ============================================================================
# STEP 4: Retrain LONG Entry Model
# ============================================================================

print("\n" + "="*80)
print("STEP 4: Retraining LONG Entry Model")
print("="*80)

# Get LONG feature columns from baseline model
baseline_long_features_path = MODELS_DIR / "xgboost_v4_phase4_advanced_lookahead3_thresh0_features.txt"
with open(baseline_long_features_path, 'r') as f:
    long_feature_cols = [line.strip() for line in f.readlines() if line.strip()]

print(f"\nTotal LONG features: {len(long_feature_cols)}")

# Prepare data
X_long = df[long_feature_cols].values
y_long = long_labels

# Split train/test (80/20, no shuffle for time series)
split_idx = int(len(X_long) * 0.8)
X_train_long = X_long[:split_idx]
y_train_long = y_long[:split_idx]
X_test_long = X_long[split_idx:]
y_test_long = y_long[split_idx:]

print(f"Train samples: {len(X_train_long):,}")
print(f"Test samples: {len(X_test_long):,}")
print(f"Positive rate (train): {np.sum(y_train_long)/len(y_train_long)*100:.2f}%")
print(f"Positive rate (test): {np.sum(y_test_long)/len(y_test_long)*100:.2f}%")

# Scale features
scaler_long = StandardScaler()
X_train_long_scaled = scaler_long.fit_transform(X_train_long)
X_test_long_scaled = scaler_long.transform(X_test_long)

# Train XGBoost model
print("\nTraining LONG Entry Model...")
scale_pos_weight = len(y_train_long[y_train_long == 0]) / len(y_train_long[y_train_long == 1])
print(f"  Scale pos weight: {scale_pos_weight:.2f}")

model_long = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    eval_metric='logloss'
)

model_long.fit(X_train_long_scaled, y_train_long)

# Evaluate
y_pred_long = model_long.predict(X_test_long_scaled)
y_pred_proba_long = model_long.predict_proba(X_test_long_scaled)[:, 1]

print("\n" + "-"*80)
print("LONG Entry Model Performance (Test Set)")
print("-"*80)
print(classification_report(y_test_long, y_pred_long, digits=4))

# Get precision from classification report
from sklearn.metrics import precision_score
precision_long = precision_score(y_test_long, y_pred_long)

# Save model
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_long_path = MODELS_DIR / f"xgboost_long_trade_outcome_full_optimized_{timestamp}.pkl"
scaler_long_path = MODELS_DIR / f"xgboost_long_trade_outcome_full_optimized_{timestamp}_scaler.pkl"
features_long_path = MODELS_DIR / f"xgboost_long_trade_outcome_full_optimized_{timestamp}_features.txt"

with open(model_long_path, 'wb') as f:
    pickle.dump(model_long, f)

joblib.dump(scaler_long, scaler_long_path)

with open(features_long_path, 'w') as f:
    for feature in long_feature_cols:
        f.write(f"{feature}\n")

print(f"\n✅ LONG Entry Model saved:")
print(f"   {model_long_path.name}")
print(f"   Precision: {precision_long*100:.2f}%")

# ============================================================================
# STEP 5: Retrain SHORT Entry Model
# ============================================================================

print("\n" + "="*80)
print("STEP 5: Retraining SHORT Entry Model")
print("="*80)

# Get SHORT feature columns from baseline model
baseline_short_features_path = MODELS_DIR / "xgboost_short_redesigned_20251016_233322_features.txt"
with open(baseline_short_features_path, 'r') as f:
    short_feature_cols = [line.strip() for line in f.readlines() if line.strip()]

print(f"\nTotal SHORT features: {len(short_feature_cols)}")

# Prepare data
X_short = df[short_feature_cols].values
y_short = short_labels

# Split train/test
X_train_short = X_short[:split_idx]
y_train_short = y_short[:split_idx]
X_test_short = X_short[split_idx:]
y_test_short = y_short[split_idx:]

print(f"Train samples: {len(X_train_short):,}")
print(f"Test samples: {len(X_test_short):,}")
print(f"Positive rate (train): {np.sum(y_train_short)/len(y_train_short)*100:.2f}%")
print(f"Positive rate (test): {np.sum(y_test_short)/len(y_test_short)*100:.2f}%")

# Scale features
scaler_short = StandardScaler()
X_train_short_scaled = scaler_short.fit_transform(X_train_short)
X_test_short_scaled = scaler_short.transform(X_test_short)

# Train XGBoost model
print("\nTraining SHORT Entry Model...")
scale_pos_weight = len(y_train_short[y_train_short == 0]) / len(y_train_short[y_train_short == 1])
print(f"  Scale pos weight: {scale_pos_weight:.2f}")

model_short = XGBClassifier(
    n_estimators=300,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    eval_metric='logloss'
)

model_short.fit(X_train_short_scaled, y_train_short)

# Evaluate
y_pred_short = model_short.predict(X_test_short_scaled)
y_pred_proba_short = model_short.predict_proba(X_test_short_scaled)[:, 1]

print("\n" + "-"*80)
print("SHORT Entry Model Performance (Test Set)")
print("-"*80)
print(classification_report(y_test_short, y_pred_short, digits=4))

precision_short = precision_score(y_test_short, y_pred_short)

# Save model
model_short_path = MODELS_DIR / f"xgboost_short_trade_outcome_full_optimized_{timestamp}.pkl"
scaler_short_path = MODELS_DIR / f"xgboost_short_trade_outcome_full_optimized_{timestamp}_scaler.pkl"
features_short_path = MODELS_DIR / f"xgboost_short_trade_outcome_full_optimized_{timestamp}_features.txt"

with open(model_short_path, 'wb') as f:
    pickle.dump(model_short, f)

joblib.dump(scaler_short, scaler_short_path)

with open(features_short_path, 'w') as f:
    for feature in short_feature_cols:
        f.write(f"{feature}\n")

print(f"\n✅ SHORT Entry Model saved:")
print(f"   {model_short_path.name}")
print(f"   Precision: {precision_short*100:.2f}%")

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*80)
print("RETRAINING COMPLETE - TRADE-OUTCOME LABELING (FULL DATASET - OPTIMIZED)")
print("="*80)

print(f"\n✅ LONG Entry Model:")
print(f"   Precision: {precision_long*100:.2f}%")
print(f"   File: {model_long_path.name}")

print(f"\n✅ SHORT Entry Model:")
print(f"   Precision: {precision_short*100:.2f}%")
print(f"   File: {model_short_path.name}")

print(f"\n⚡ Performance:")
print(f"   Total Candles: {len(df):,}")
print(f"   Simulation Time: {total_simulation_time/60:.1f} minutes")
print(f"   Improved Risk-Reward: MAE >= -4%, MFE >= 2%")

print(f"\n🎯 Next Steps:")
print(f"   1. Backtest these full-dataset models")
print(f"   2. Compare vs Baseline and Sample models")
print(f"   3. Deploy best performer to production")

print("="*80)
