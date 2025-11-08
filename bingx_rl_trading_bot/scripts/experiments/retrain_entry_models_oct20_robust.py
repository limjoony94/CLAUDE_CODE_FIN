"""
Retrain Entry Models (Oct 20, 2025) - ROBUST VERSION with Error Handling
==========================================================================
Retrain LONG and SHORT entry models with updated dataset (32,004 candles)
Includes: Error handling, checkpointing, progress saving
"""

import sys
import time
import pickle
import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import traceback
import json

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiments.calculate_all_features import calculate_all_features
from scripts.experiments.retrain_exit_models_opportunity_gating import prepare_exit_features
from scripts.experiments.trade_simulator import TradeSimulator
from xgboost import XGBClassifier

DATA_DIR = PROJECT_ROOT / "data" / "historical"
MODELS_DIR = PROJECT_ROOT / "models"
CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"
CHECKPOINT_DIR.mkdir(exist_ok=True)

# Improved Risk-Reward Thresholds
MAE_THRESHOLD = -0.04  # -4%
MFE_THRESHOLD = 0.02   # +2%

print("=" * 80)
print("RETRAIN ENTRY MODELS: TRADE-OUTCOME LABELING (ROBUST VERSION)")
print("=" * 80)
print()
print("⚡ Features: Error handling, checkpointing, resume capability")
print(f"   Improved Risk-Reward: MAE >= {MAE_THRESHOLD*100:.1f}%, MFE >= {MFE_THRESHOLD*100:.1f}%")
print()

# ============================================================================
# STEP 1: Load and Prepare Data
# ============================================================================
print("-" * 80)
print("STEP 1: Loading and Preparing Data")
print("-" * 80)

data_file = DATA_DIR / "BTCUSDT_5m_updated.csv"  # Updated with data until Oct 20, 2025
df = pd.read_csv(data_file)
df['timestamp'] = pd.to_datetime(df['timestamp'])

print(f"\n✅ Loaded {len(df):,} candles")
print(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")

# Calculate features
print("\nCalculating features...")
sys.stdout.flush()
df = calculate_all_features(df)
df = prepare_exit_features(df)
print(f"✅ All features calculated ({len(df.columns)} columns)")
sys.stdout.flush()

# ============================================================================
# STEP 2: Load Exit Models for Trade Simulation
# ============================================================================
print("\n" + "-" * 80)
print("STEP 2: Loading Exit Models for Trade Simulation")
print("-" * 80)

# Load LONG Exit Model
long_exit_model_path = MODELS_DIR / "xgboost_long_exit_oppgating_improved_20251017_151624.pkl"
with open(long_exit_model_path, 'rb') as f:
    long_exit_model = pickle.load(f)

long_exit_scaler_path = MODELS_DIR / "xgboost_long_exit_oppgating_improved_20251017_151624_scaler.pkl"
long_exit_scaler = joblib.load(long_exit_scaler_path)

long_exit_features_path = MODELS_DIR / "xgboost_long_exit_oppgating_improved_20251017_151624_features.txt"
with open(long_exit_features_path, 'r') as f:
    long_exit_feature_columns = [line.strip() for line in f.readlines() if line.strip()]

print(f"  ✅ LONG Exit: {len(long_exit_feature_columns)} features")

# Load SHORT Exit Model
short_exit_model_path = MODELS_DIR / "xgboost_short_exit_oppgating_improved_20251017_152440.pkl"
with open(short_exit_model_path, 'rb') as f:
    short_exit_model = pickle.load(f)

short_exit_scaler_path = MODELS_DIR / "xgboost_short_exit_oppgating_improved_20251017_152440_scaler.pkl"
short_exit_scaler = joblib.load(short_exit_scaler_path)

short_exit_features_path = MODELS_DIR / "xgboost_short_exit_oppgating_improved_20251017_152440_features.txt"
with open(short_exit_features_path, 'r') as f:
    short_exit_feature_columns = [line.strip() for line in f.readlines() if line.strip()]

print(f"  ✅ SHORT Exit: {len(short_exit_feature_columns)} features")

# Initialize Trade Simulators
long_simulator = TradeSimulator(
    exit_model=long_exit_model,
    exit_scaler=long_exit_scaler,
    exit_features=long_exit_feature_columns,
    leverage=4,
    ml_exit_threshold=0.70,
    emergency_stop_loss=-0.04,
    emergency_max_hold=96  # 8 hours at 5m candles
)

short_simulator = TradeSimulator(
    exit_model=short_exit_model,
    exit_scaler=short_exit_scaler,
    exit_features=short_exit_feature_columns,
    leverage=4,
    ml_exit_threshold=0.72,
    emergency_stop_loss=-0.04,
    emergency_max_hold=96
)

print("✅ Trade simulators ready")

# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def save_checkpoint(checkpoint_path, data):
    """Save checkpoint data"""
    with open(checkpoint_path, 'w') as f:
        json.dump(data, f)
    print(f"  💾 Checkpoint saved: {checkpoint_path.name}")

def load_checkpoint(checkpoint_path):
    """Load checkpoint data"""
    if checkpoint_path.exists():
        with open(checkpoint_path, 'r') as f:
            return json.load(f)
    return None

def calculate_trade_quality(trade_result, MAE_THRESHOLD, MFE_THRESHOLD):
    """Calculate trade quality score"""
    score = 0
    criteria = []

    # Criterion 1: Profitable
    if trade_result['profit_pct'] > 0:
        score += 1
        criteria.append('profitable')

    # Criterion 2: Good Risk-Reward Ratio
    if (trade_result['mae'] >= MAE_THRESHOLD and
        trade_result['mfe'] >= MFE_THRESHOLD):
        score += 1
        criteria.append('good_rr')

    # Criterion 3: Efficient Exit (ML Exit works)
    if trade_result['exit_reason'] == 'ml_exit':
        score += 1
        criteria.append('ml_exit')

    return score, criteria

# ============================================================================
# STEP 3: LONG Entry Labeling (WITH ERROR HANDLING)
# ============================================================================
print("\n" + "-" * 80)
print("STEP 3: LONG Entry Labeling (Trade-Outcome Based - ROBUST)")
print("-" * 80)

checkpoint_file = CHECKPOINT_DIR / "long_labeling_checkpoint.json"
checkpoint = load_checkpoint(checkpoint_file)

if checkpoint:
    print(f"\n🔄 Resuming from checkpoint:")
    print(f"   Last completed: {checkpoint['last_completed']:,}/{checkpoint['total_entries']:,}")
    long_trade_results = checkpoint['results']
    start_idx = checkpoint['last_completed']
else:
    print("\n▶️  Starting fresh labeling...")
    long_trade_results = []
    start_idx = 0

print("  Simulating trades with error handling...")
sys.stdout.flush()

start_time = time.time()
valid_indices = list(range(100, len(df) - 96))
total_entries = len(valid_indices)

print(f"  Total entries to simulate: {total_entries:,}")
print(f"  Starting from: {start_idx:,}")
sys.stdout.flush()

# Process with error handling and checkpointing
BATCH_SIZE = 1000
CHECKPOINT_INTERVAL = 5000
error_count = 0
MAX_ERRORS = 100  # Stop if too many errors

for batch_num in range(start_idx, total_entries):
    i = valid_indices[batch_num]

    try:
        result = long_simulator.simulate_trade(df, i, 'LONG')
        long_trade_results.append(result)

    except Exception as e:
        error_count += 1
        print(f"\n    ⚠️  Error at entry {batch_num} (index {i}): {str(e)}")

        # Add dummy result to maintain index alignment
        long_trade_results.append({
            'entry_idx': i,
            'profit_pct': 0,
            'mae': 0,
            'mfe': 0,
            'exit_reason': 'error',
            'hold_periods': 0
        })

        if error_count >= MAX_ERRORS:
            print(f"\n❌ Too many errors ({error_count}). Stopping.")
            break

        continue

    # Progress update
    if (batch_num + 1) % BATCH_SIZE == 0:
        elapsed = time.time() - start_time
        progress_pct = (batch_num + 1) / total_entries * 100
        rate = (batch_num + 1 - start_idx) / elapsed if elapsed > 0 else 0
        eta = (total_entries - batch_num - 1) / rate if rate > 0 else 0
        print(f"    Progress: {batch_num+1:,}/{total_entries:,} ({progress_pct:.1f}%) | "
              f"Rate: {rate:.0f} entries/s | ETA: {eta/60:.1f} min | Errors: {error_count}")
        sys.stdout.flush()

    # Save checkpoint
    if (batch_num + 1) % CHECKPOINT_INTERVAL == 0:
        checkpoint_data = {
            'last_completed': batch_num + 1,
            'total_entries': total_entries,
            'results': long_trade_results,
            'error_count': error_count
        }
        save_checkpoint(checkpoint_file, checkpoint_data)

long_elapsed = time.time() - start_time
print(f"\n  ✅ Completed {len(long_trade_results):,} LONG entries in {long_elapsed:.1f}s ({long_elapsed/60:.1f} min)")
print(f"     Errors encountered: {error_count}")
sys.stdout.flush()

# Create LONG labels
long_labels = np.zeros(len(df))
long_label_counts = {'positive': 0, 'negative': 0}

print("\n  Creating LONG entry labels...")
for idx, result in enumerate(long_trade_results):
    entry_idx = valid_indices[idx]
    quality_score, _ = calculate_trade_quality(result, MAE_THRESHOLD, MFE_THRESHOLD)

    # Require 2+ criteria for positive label
    if quality_score >= 2:
        long_labels[entry_idx] = 1
        long_label_counts['positive'] += 1
    else:
        long_label_counts['negative'] += 1

print(f"  ✅ LONG Labels: {long_label_counts['positive']:,} positive ({long_label_counts['positive']/len(long_trade_results)*100:.1f}%), "
      f"{long_label_counts['negative']:,} negative ({long_label_counts['negative']/len(long_trade_results)*100:.1f}%)")
sys.stdout.flush()

# ============================================================================
# STEP 4: SHORT Entry Labeling (WITH ERROR HANDLING)
# ============================================================================
print("\n" + "-" * 80)
print("STEP 4: SHORT Entry Labeling (Trade-Outcome Based - ROBUST)")
print("-" * 80)

checkpoint_file = CHECKPOINT_DIR / "short_labeling_checkpoint.json"
checkpoint = load_checkpoint(checkpoint_file)

if checkpoint:
    print(f"\n🔄 Resuming from checkpoint:")
    print(f"   Last completed: {checkpoint['last_completed']:,}/{checkpoint['total_entries']:,}")
    short_trade_results = checkpoint['results']
    start_idx = checkpoint['last_completed']
else:
    print("\n▶️  Starting fresh labeling...")
    short_trade_results = []
    start_idx = 0

print("  Simulating trades with error handling...")
sys.stdout.flush()

start_time = time.time()
error_count = 0

for batch_num in range(start_idx, total_entries):
    i = valid_indices[batch_num]

    try:
        result = short_simulator.simulate_trade(df, i, 'SHORT')
        short_trade_results.append(result)

    except Exception as e:
        error_count += 1
        print(f"\n    ⚠️  Error at entry {batch_num} (index {i}): {str(e)}")

        # Add dummy result
        short_trade_results.append({
            'entry_idx': i,
            'profit_pct': 0,
            'mae': 0,
            'mfe': 0,
            'exit_reason': 'error',
            'hold_periods': 0
        })

        if error_count >= MAX_ERRORS:
            print(f"\n❌ Too many errors ({error_count}). Stopping.")
            break

        continue

    # Progress update
    if (batch_num + 1) % BATCH_SIZE == 0:
        elapsed = time.time() - start_time
        progress_pct = (batch_num + 1) / total_entries * 100
        rate = (batch_num + 1 - start_idx) / elapsed if elapsed > 0 else 0
        eta = (total_entries - batch_num - 1) / rate if rate > 0 else 0
        print(f"    Progress: {batch_num+1:,}/{total_entries:,} ({progress_pct:.1f}%) | "
              f"Rate: {rate:.0f} entries/s | ETA: {eta/60:.1f} min | Errors: {error_count}")
        sys.stdout.flush()

    # Save checkpoint
    if (batch_num + 1) % CHECKPOINT_INTERVAL == 0:
        checkpoint_data = {
            'last_completed': batch_num + 1,
            'total_entries': total_entries,
            'results': short_trade_results,
            'error_count': error_count
        }
        save_checkpoint(checkpoint_file, checkpoint_data)

short_elapsed = time.time() - start_time
print(f"\n  ✅ Completed {len(short_trade_results):,} SHORT entries in {short_elapsed:.1f}s ({short_elapsed/60:.1f} min)")
print(f"     Errors encountered: {error_count}")
sys.stdout.flush()

# Create SHORT labels
short_labels = np.zeros(len(df))
short_label_counts = {'positive': 0, 'negative': 0}

print("\n  Creating SHORT entry labels...")
for idx, result in enumerate(short_trade_results):
    entry_idx = valid_indices[idx]
    quality_score, _ = calculate_trade_quality(result, MAE_THRESHOLD, MFE_THRESHOLD)

    if quality_score >= 2:
        short_labels[entry_idx] = 1
        short_label_counts['positive'] += 1
    else:
        short_label_counts['negative'] += 1

print(f"  ✅ SHORT Labels: {short_label_counts['positive']:,} positive ({short_label_counts['positive']/len(short_trade_results)*100:.1f}%), "
      f"{short_label_counts['negative']:,} negative ({short_label_counts['negative']/len(short_trade_results)*100:.1f}%)")
sys.stdout.flush()

# ============================================================================
# STEP 5: Train Models
# ============================================================================
print("\n" + "=" * 80)
print("STEP 5: Training XGBoost Models")
print("=" * 80)

# Add labels to dataframe
df['long_entry_label'] = long_labels
df['short_entry_label'] = short_labels

# Feature columns (exclude labels, timestamp, OHLCV)
exclude_cols = ['timestamp', 'open', 'high', 'low', 'close', 'volume',
                'long_entry_label', 'short_entry_label']
feature_cols = [col for col in df.columns if col not in exclude_cols]

print(f"\nUsing {len(feature_cols)} features")

# Train LONG Entry Model
print("\n" + "-" * 80)
print("Training LONG Entry Model")
print("-" * 80)

X_long = df[feature_cols].values
y_long = df['long_entry_label'].values

# Remove NaN
valid_mask = ~np.isnan(X_long).any(axis=1) & ~np.isnan(y_long)
X_long_clean = X_long[valid_mask]
y_long_clean = y_long[valid_mask]

print(f"  Training samples: {len(X_long_clean):,}")
print(f"  Positive rate: {y_long_clean.mean()*100:.2f}%")

# Train
long_entry_model = XGBClassifier(
    n_estimators=200,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='logloss'
)

long_entry_model.fit(X_long_clean, y_long_clean)
print("  ✅ LONG Entry model trained")

# Save
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
long_model_path = MODELS_DIR / f"xgboost_long_trade_outcome_full_{timestamp}.pkl"
with open(long_model_path, 'wb') as f:
    pickle.dump(long_entry_model, f)
print(f"  💾 Saved: {long_model_path.name}")

# Save feature names
long_features_path = MODELS_DIR / f"xgboost_long_trade_outcome_full_{timestamp}_features.txt"
with open(long_features_path, 'w') as f:
    f.write('\n'.join(feature_cols))
print(f"  💾 Saved features: {long_features_path.name}")

# Train SHORT Entry Model
print("\n" + "-" * 80)
print("Training SHORT Entry Model")
print("-" * 80)

X_short = df[feature_cols].values
y_short = df['short_entry_label'].values

valid_mask = ~np.isnan(X_short).any(axis=1) & ~np.isnan(y_short)
X_short_clean = X_short[valid_mask]
y_short_clean = y_short[valid_mask]

print(f"  Training samples: {len(X_short_clean):,}")
print(f"  Positive rate: {y_short_clean.mean()*100:.2f}%")

short_entry_model = XGBClassifier(
    n_estimators=200,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='logloss'
)

short_entry_model.fit(X_short_clean, y_short_clean)
print("  ✅ SHORT Entry model trained")

# Save
short_model_path = MODELS_DIR / f"xgboost_short_trade_outcome_full_{timestamp}.pkl"
with open(short_model_path, 'wb') as f:
    pickle.dump(short_entry_model, f)
print(f"  💾 Saved: {short_model_path.name}")

short_features_path = MODELS_DIR / f"xgboost_short_trade_outcome_full_{timestamp}_features.txt"
with open(short_features_path, 'w') as f:
    f.write('\n'.join(feature_cols))
print(f"  💾 Saved features: {short_features_path.name}")

# Clean up checkpoints
print("\n🧹 Cleaning up checkpoints...")
for checkpoint_file in CHECKPOINT_DIR.glob("*_labeling_checkpoint.json"):
    checkpoint_file.unlink()
    print(f"   Deleted: {checkpoint_file.name}")

print("\n" + "=" * 80)
print("✅ TRAINING COMPLETE")
print("=" * 80)
print(f"\nNew models saved with timestamp: {timestamp}")
print(f"  - {long_model_path.name}")
print(f"  - {short_model_path.name}")
print("\nNext steps:")
print("  1. Run backtest with new models")
print("  2. Compare performance vs old models")
print("  3. Deploy if performance is satisfactory")
