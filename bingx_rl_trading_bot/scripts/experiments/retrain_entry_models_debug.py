"""
Retrain Entry Models: Debug Version with Enhanced Monitoring
===========================================================

Changes:
- Progress updates every 100 entries (not 1000)
- sys.stdout.flush() after every progress update
- Detailed error logging
- Memory usage tracking
- Explicit exception handling with full traceback
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
from sklearn.metrics import classification_report, precision_score
import time
import traceback
import psutil

from scripts.experiments.calculate_all_features import calculate_all_features
from scripts.experiments.retrain_exit_models_opportunity_gating import prepare_exit_features
from scripts.experiments.trade_simulator import TradeSimulator

DATA_DIR = PROJECT_ROOT / "data" / "historical"
MODELS_DIR = PROJECT_ROOT / "models"

print("="*80)
print("RETRAIN ENTRY MODELS: DEBUG VERSION")
print("="*80)
print("\n⚡ Enhanced monitoring: Progress every 100 entries")
print("   Improved Risk-Reward: MAE >= -4%, MFE >= 2%")
print(f"   Process PID: {os.getpid()}")

# Check memory
process = psutil.Process(os.getpid())
print(f"   Initial Memory: {process.memory_info().rss / 1024 / 1024:.1f} MB")
print()
sys.stdout.flush()

# ============================================================================
# STEP 1: Load and Prepare Data
# ============================================================================

print("-"*80)
print("STEP 1: Loading and Preparing Data")
print("-"*80)

data_file = DATA_DIR / "BTCUSDT_5m_updated.csv"
df = pd.read_csv(data_file)
print(f"\n✅ Loaded {len(df):,} candles")
print(f"   Date range: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")
sys.stdout.flush()

print("\nCalculating features...")
sys.stdout.flush()

df = calculate_all_features(df)
df = prepare_exit_features(df)
print(f"✅ All features calculated ({len(df.columns)} columns)")
print(f"   Memory: {process.memory_info().rss / 1024 / 1024:.1f} MB")
sys.stdout.flush()

# ============================================================================
# STEP 2: Load Exit Models
# ============================================================================

print("\n" + "-"*80)
print("STEP 2: Loading Exit Models for Trade Simulation")
print("-"*80)
sys.stdout.flush()

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
sys.stdout.flush()

# Create simulators
long_simulator = TradeSimulator(
    exit_model=long_exit_model,
    exit_scaler=long_exit_scaler,
    exit_features=long_exit_features,
    leverage=4,
    ml_exit_threshold=0.70,
    emergency_stop_loss=-0.04,
    emergency_max_hold=96
)

short_simulator = TradeSimulator(
    exit_model=short_exit_model,
    exit_scaler=short_exit_scaler,
    exit_features=short_exit_features,
    leverage=4,
    ml_exit_threshold=0.72,
    emergency_stop_loss=-0.04,
    emergency_max_hold=96
)

# ============================================================================
# STEP 3: Create Trade-Outcome Labels (DEBUG MODE)
# ============================================================================

print("\n" + "-"*80)
print("STEP 3: Creating Trade-Outcome Labels (DEBUG MODE)")
print("-"*80)
sys.stdout.flush()

# Improved Risk-Reward criteria
PROFIT_THRESHOLD = 0.02
MAE_THRESHOLD = -0.04
MFE_THRESHOLD = 0.02
SCORING_THRESHOLD = 2

def evaluate_trade_quality(trade_result):
    """Evaluate trade result against 2-of-3 criteria"""
    score = 0
    criteria = []

    if trade_result['leveraged_pnl_pct'] >= PROFIT_THRESHOLD:
        score += 1
        criteria.append('profitable')

    if (trade_result['mae'] >= MAE_THRESHOLD and
        trade_result['mfe'] >= MFE_THRESHOLD):
        score += 1
        criteria.append('good_rr')

    if trade_result['exit_reason'] == 'ml_exit':
        score += 1
        criteria.append('ml_exit')

    return score, criteria

# LONG Entry Labeling
print("\nLONG Entry Labeling (Trade-Outcome Based - DEBUG)")
print("  Simulating trades with ENHANCED monitoring...")
sys.stdout.flush()

start_time = time.time()
long_trade_results = []

# Calculate valid entry indices
valid_indices = range(100, len(df) - 96)
total_entries = len(list(valid_indices))

print(f"  Total entries to simulate: {total_entries:,}")
print(f"  Memory before simulation: {process.memory_info().rss / 1024 / 1024:.1f} MB")
sys.stdout.flush()

# DEBUG: Much smaller batch size for frequent updates
BATCH_SIZE = 100
error_count = 0
last_success_idx = None

for batch_num, i in enumerate(valid_indices):
    try:
        # Simulate trade
        result = long_simulator.simulate_trade(df, i, 'LONG')

        if result is None:
            # Not enough data to simulate
            result = {
                'entry_idx': i,
                'leveraged_pnl_pct': 0,
                'mae': 0,
                'mfe': 0,
                'exit_reason': 'insufficient_data',
                'hold_periods': 0
            }

        long_trade_results.append(result)
        last_success_idx = i

    except Exception as e:
        error_count += 1

        # Log detailed error info
        print(f"\n    ❌ ERROR at entry {batch_num} (index {i}):")
        print(f"       Exception: {type(e).__name__}: {str(e)}")
        print(f"       Last success: {last_success_idx}")
        sys.stdout.flush()

        # Print full traceback for first 5 errors
        if error_count <= 5:
            print("       Traceback:")
            traceback.print_exc()
            sys.stdout.flush()

        # Add dummy result
        long_trade_results.append({
            'entry_idx': i,
            'leveraged_pnl_pct': 0,
            'mae': 0,
            'mfe': 0,
            'exit_reason': 'error',
            'hold_periods': 0
        })

        # Stop if too many errors
        if error_count >= 100:
            print(f"\n    ❌ Too many errors ({error_count}). Stopping.")
            sys.stdout.flush()
            break

    # Progress update every 100 entries (MUCH more frequent)
    if (batch_num + 1) % BATCH_SIZE == 0:
        elapsed = time.time() - start_time
        progress_pct = (batch_num + 1) / total_entries * 100
        rate = (batch_num + 1) / elapsed
        eta = (total_entries - batch_num - 1) / rate if rate > 0 else 0
        mem = process.memory_info().rss / 1024 / 1024

        print(f"    Progress: {batch_num+1:,}/{total_entries:,} ({progress_pct:.1f}%) | "
              f"Rate: {rate:.0f} e/s | ETA: {eta/60:.1f}min | "
              f"Err: {error_count} | Mem: {mem:.0f}MB")
        sys.stdout.flush()

long_elapsed = time.time() - start_time
print(f"\n  ✅ Simulated {len(long_trade_results):,} LONG entries in {long_elapsed:.1f}s ({long_elapsed/60:.1f} min)")
print(f"     Errors: {error_count}")
print(f"     Memory: {process.memory_info().rss / 1024 / 1024:.1f} MB")
sys.stdout.flush()

# Create LONG labels
long_labels = np.zeros(len(df))
criterion_counts_long = {'profitable': 0, 'good_rr': 0, 'ml_exit': 0}

print("\n  Creating LONG entry labels...")
sys.stdout.flush()

for idx, result in enumerate(long_trade_results):
    if idx >= len(valid_indices):
        break

    entry_idx = list(valid_indices)[idx]
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
sys.stdout.flush()

print("\n" + "="*80)
print("✅ LONG LABELING COMPLETE")
print("="*80)
print(f"\nResults saved in memory. Continue with SHORT labeling or model training.")
print(f"Process still running. PID: {os.getpid()}")
sys.stdout.flush()

# ============================================================================
# SHORT Entry Labeling (DEBUG MODE)
# ============================================================================

print("\n" + "-"*80)
print("SHORT Entry Labeling (Trade-Outcome Based - DEBUG)")
print("-"*80)
print("  Simulating trades with ENHANCED monitoring...")
sys.stdout.flush()

start_time = time.time()
short_trade_results = []

print(f"  Total entries to simulate: {total_entries:,}")
print(f"  Memory before simulation: {process.memory_info().rss / 1024 / 1024:.1f} MB")
sys.stdout.flush()

# DEBUG: Much smaller batch size for frequent updates
error_count = 0
last_success_idx = None

for batch_num, i in enumerate(valid_indices):
    try:
        # Simulate trade
        result = short_simulator.simulate_trade(df, i, 'SHORT')

        if result is None:
            # Not enough data to simulate
            result = {
                'entry_idx': i,
                'leveraged_pnl_pct': 0,
                'mae': 0,
                'mfe': 0,
                'exit_reason': 'insufficient_data',
                'hold_periods': 0
            }

        short_trade_results.append(result)
        last_success_idx = i

    except Exception as e:
        error_count += 1

        # Log detailed error info
        print(f"\n    ❌ ERROR at entry {batch_num} (index {i}):")
        print(f"       Exception: {type(e).__name__}: {str(e)}")
        print(f"       Last success: {last_success_idx}")
        sys.stdout.flush()

        # Print full traceback for first 5 errors
        if error_count <= 5:
            print("       Traceback:")
            traceback.print_exc()
            sys.stdout.flush()

        # Add dummy result
        short_trade_results.append({
            'entry_idx': i,
            'leveraged_pnl_pct': 0,
            'mae': 0,
            'mfe': 0,
            'exit_reason': 'error',
            'hold_periods': 0
        })

        # Stop if too many errors
        if error_count >= 100:
            print(f"\n    ❌ Too many errors ({error_count}). Stopping.")
            sys.stdout.flush()
            break

    # Progress update every 100 entries (MUCH more frequent)
    if (batch_num + 1) % BATCH_SIZE == 0:
        elapsed = time.time() - start_time
        progress_pct = (batch_num + 1) / total_entries * 100
        rate = (batch_num + 1) / elapsed
        eta = (total_entries - batch_num - 1) / rate if rate > 0 else 0
        mem = process.memory_info().rss / 1024 / 1024

        print(f"    Progress: {batch_num+1:,}/{total_entries:,} ({progress_pct:.1f}%) | "
              f"Rate: {rate:.0f} e/s | ETA: {eta/60:.1f}min | "
              f"Err: {error_count} | Mem: {mem:.0f}MB")
        sys.stdout.flush()

short_elapsed = time.time() - start_time
print(f"\n  ✅ Simulated {len(short_trade_results):,} SHORT entries in {short_elapsed:.1f}s ({short_elapsed/60:.1f} min)")
print(f"     Errors: {error_count}")
print(f"     Memory: {process.memory_info().rss / 1024 / 1024:.1f} MB")
sys.stdout.flush()

# Create SHORT labels
short_labels = np.zeros(len(df))
criterion_counts_short = {'profitable': 0, 'good_rr': 0, 'ml_exit': 0}

print("\n  Creating SHORT entry labels...")
sys.stdout.flush()

for idx, result in enumerate(short_trade_results):
    if idx >= len(valid_indices):
        break

    entry_idx = list(valid_indices)[idx]
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
sys.stdout.flush()

total_simulation_time = long_elapsed + short_elapsed
print(f"\n⚡ Total Simulation Time: {total_simulation_time:.1f}s ({total_simulation_time/60:.1f} minutes)")
sys.stdout.flush()

print("\n" + "="*80)
print("✅ LONG + SHORT LABELING COMPLETE")
print("="*80)
print(f"\nResults saved in memory. Ready for model training.")
print(f"Process still running. PID: {os.getpid()}")
sys.stdout.flush()


# ============================================================================
# STEP 4: Retrain LONG Entry Model
# ============================================================================

print("\n" + "="*80)
print("STEP 4: Retraining LONG Entry Model")
print("="*80)
sys.stdout.flush()

# Get LONG feature columns
baseline_long_features_path = MODELS_DIR / "xgboost_v4_phase4_advanced_lookahead3_thresh0_features.txt"
with open(baseline_long_features_path, 'r') as f:
    long_feature_cols = [line.strip() for line in f.readlines() if line.strip()]

print(f"\nTotal LONG features: {len(long_feature_cols)}")
sys.stdout.flush()

# Prepare data
X_long = df[long_feature_cols].values
y_long = long_labels

# Split train/test
split_idx = int(len(X_long) * 0.8)
X_train_long = X_long[:split_idx]
y_train_long = y_long[:split_idx]
X_test_long = X_long[split_idx:]
y_test_long = y_long[split_idx:]

print(f"Train samples: {len(X_train_long):,}")
print(f"Test samples: {len(X_test_long):,}")
print(f"Positive rate (train): {np.sum(y_train_long)/len(y_train_long)*100:.2f}%")
print(f"Positive rate (test): {np.sum(y_test_long)/len(y_test_long)*100:.2f}%")
sys.stdout.flush()

# Scale features
scaler_long = StandardScaler()
X_train_long_scaled = scaler_long.fit_transform(X_train_long)
X_test_long_scaled = scaler_long.transform(X_test_long)

# Train XGBoost model
print("\nTraining LONG Entry Model...")
sys.stdout.flush()

scale_pos_weight = len(y_train_long[y_train_long == 0]) / len(y_train_long[y_train_long == 1])
print(f"  Scale pos weight: {scale_pos_weight:.2f}")
sys.stdout.flush()

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

print("\n" + "-"*80)
print("LONG Entry Model Performance (Test Set)")
print("-"*80)
print(classification_report(y_test_long, y_pred_long, digits=4))
sys.stdout.flush()

precision_long = precision_score(y_test_long, y_pred_long)

# Save model
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_long_path = MODELS_DIR / f"xgboost_long_trade_outcome_full_{timestamp}.pkl"
scaler_long_path = MODELS_DIR / f"xgboost_long_trade_outcome_full_{timestamp}_scaler.pkl"
features_long_path = MODELS_DIR / f"xgboost_long_trade_outcome_full_{timestamp}_features.txt"

with open(model_long_path, 'wb') as f:
    pickle.dump(model_long, f)

joblib.dump(scaler_long, scaler_long_path)

with open(features_long_path, 'w') as f:
    for feature in long_feature_cols:
        f.write(f"{feature}\n")

print(f"\n✅ LONG Entry Model saved:")
print(f"   {model_long_path.name}")
print(f"   Precision: {precision_long*100:.2f}%")
sys.stdout.flush()

# ============================================================================
# STEP 5: Retrain SHORT Entry Model
# ============================================================================

print("\n" + "="*80)
print("STEP 5: Retraining SHORT Entry Model")
print("="*80)
sys.stdout.flush()

# Get SHORT feature columns
baseline_short_features_path = MODELS_DIR / "xgboost_short_redesigned_20251016_233322_features.txt"
with open(baseline_short_features_path, 'r') as f:
    short_feature_cols = [line.strip() for line in f.readlines() if line.strip()]

print(f"\nTotal SHORT features: {len(short_feature_cols)}")
sys.stdout.flush()

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
sys.stdout.flush()

# Scale features
scaler_short = StandardScaler()
X_train_short_scaled = scaler_short.fit_transform(X_train_short)
X_test_short_scaled = scaler_short.transform(X_test_short)

# Train XGBoost model
print("\nTraining SHORT Entry Model...")
sys.stdout.flush()

scale_pos_weight = len(y_train_short[y_train_short == 0]) / len(y_train_short[y_train_short == 1])
print(f"  Scale pos weight: {scale_pos_weight:.2f}")
sys.stdout.flush()

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

print("\n" + "-"*80)
print("SHORT Entry Model Performance (Test Set)")
print("-"*80)
print(classification_report(y_test_short, y_pred_short, digits=4))
sys.stdout.flush()

precision_short = precision_score(y_test_short, y_pred_short)

# Save model
model_short_path = MODELS_DIR / f"xgboost_short_trade_outcome_full_{timestamp}.pkl"
scaler_short_path = MODELS_DIR / f"xgboost_short_trade_outcome_full_{timestamp}_scaler.pkl"
features_short_path = MODELS_DIR / f"xgboost_short_trade_outcome_full_{timestamp}_features.txt"

with open(model_short_path, 'wb') as f:
    pickle.dump(model_short, f)

joblib.dump(scaler_short, scaler_short_path)

with open(features_short_path, 'w') as f:
    for feature in short_feature_cols:
        f.write(f"{feature}\n")

print(f"\n✅ SHORT Entry Model saved:")
print(f"   {model_short_path.name}")
print(f"   Precision: {precision_short*100:.2f}%")
sys.stdout.flush()

# ============================================================================
# SUMMARY
# ============================================================================

print("\n" + "="*80)
print("RETRAINING COMPLETE - TRADE-OUTCOME LABELING (FULL DATASET)")
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
sys.stdout.flush()

