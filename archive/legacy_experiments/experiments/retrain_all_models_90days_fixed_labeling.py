"""
Retrain ALL 4 Models with 90-Day Dataset (62d train + 28d validation)
=====================================================================

Purpose: Full model retraining with longer historical data

Current Problem:
  - LONG Entry: Trained on 104 days (Jul-Oct) - Adequate but improvable
  - SHORT Entry: Trained on 35 days (Sep-Oct) - TOO SHORT!

New Approach:
  - Training: 62 days (Aug 7 - Oct 7, 2025)
  - Validation: 28 days (Oct 7 - Nov 4, 2025)
  - All 4 models: LONG Entry, SHORT Entry, LONG Exit, SHORT Exit

Expected Benefits:
  - SHORT gets 77% more training data (35d → 62d)
  - More consistent training across all models
  - Better generalization with validation holdout

Created: 2025-11-05 01:15 KST
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.metrics import accuracy_score, classification_report

# Paths
DATA_DIR = PROJECT_ROOT / "data" / "features"
MODELS_DIR = PROJECT_ROOT / "models"

# Labeling Configuration (FIXED as per production)
MIN_HOLD_TIME = 12  # 1 hour
MAX_HOLD_TIME = 144  # 12 hours
MIN_PNL = 0.003  # 0.3% profit (3× fees)

# Model Configuration
LEVERAGE = 4
STOP_LOSS = -0.03

print("="*80)
print("RETRAIN ALL 4 MODELS WITH 90-DAY DATASET")
print("="*80)
print()
print("📊 Dataset:")
print(f"   Total: 90 days (Aug 6 - Nov 4, 2025)")
print(f"   Training: 62 days (Aug 7 - Oct 7)")
print(f"   Validation: 28 days (Oct 7 - Nov 4)")
print()
print("🎯 Models to retrain:")
print("   1. LONG Entry (with NEW SHORT-specific features excluded)")
print("   2. SHORT Entry (with 10 NEW SHORT-specific features included)")
print("   3. LONG Exit")
print("   4. SHORT Exit")
print()
print("📝 Labeling Rules (FIXED):")
print(f"   Min Hold: {MIN_HOLD_TIME} candles ({MIN_HOLD_TIME/12:.1f}h)")
print(f"   Max Hold: {MAX_HOLD_TIME} candles ({MAX_HOLD_TIME/12:.1f}h)")
print(f"   Min PNL: {MIN_PNL*100}% (3× fees)")
print()

# ==============================================================================
# STEP 1: Load Feature Dataset
# ==============================================================================

print("-"*80)
print("STEP 1: Loading Feature Dataset")
print("-"*80)

# Find latest 90-day feature file
feature_files = list(DATA_DIR.glob("BTCUSDT_5m_features_90days_*.csv"))
if not feature_files:
    raise FileNotFoundError("No 90-day feature file found! Run calculate_features_90days.py first")

feature_file = sorted(feature_files, reverse=True)[0]
print(f"✅ Loading: {feature_file.name}")

df = pd.read_csv(feature_file)
df['timestamp'] = pd.to_datetime(df['timestamp'])

print(f"   Total rows: {len(df):,}")
print(f"   Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
print(f"   Total features: {len(df.columns) - 6} (excluding OHLCV + timestamp)")
print()

# ==============================================================================
# STEP 2: Train/Validation Split
# ==============================================================================

print("-"*80)
print("STEP 2: Train/Validation Split")
print("-"*80)

# Split: Last 28 days = validation, rest = training
train_end_timestamp = df['timestamp'].max() - timedelta(days=28)
train_df = df[df['timestamp'] <= train_end_timestamp].copy()
val_df = df[df['timestamp'] > train_end_timestamp].copy()

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

# ==============================================================================
# STEP 3: Define Feature Lists
# ==============================================================================

print("-"*80)
print("STEP 3: Defining Feature Lists")
print("-"*80)

# Load existing feature lists from current models
long_features_path = MODELS_DIR / "xgboost_long_entry_enhanced_20251024_012445_features.txt"
short_features_path = MODELS_DIR / "xgboost_short_entry_with_new_features_20251104_213043_features.txt"

with open(long_features_path, 'r') as f:
    long_entry_features = [line.strip() for line in f.readlines()]

with open(short_features_path, 'r') as f:
    short_entry_features = [line.strip() for line in f.readlines()]

# Use baseline exit features that EXIST in the dataset (phase1 calculation)
# Exit-specific features (volume_surge, price_acceleration, etc.) are NOT in phase1 datasets
exit_features = [
    # Core technical indicators (CORRECTED feature names from dataset)
    'rsi', 'macd', 'macd_signal', 'bb_width', 'atr',
    'ema_12', 'ema_200', 'sma_20', 'ma_20', 'ma_200',
    'vwap', 'volume_ratio',
    # Price/momentum features
    'price_change_5', 'momentum_200',
    # Volatility
    'volatility',
    # Support/Resistance
    'distance_to_support_pct', 'distance_to_resistance_pct',
    # Volume Profile
    'vp_value_area_low', 'vp_value_area_high', 'vp_poc'
]

print(f"✅ LONG Entry features: {len(long_entry_features)}")
print(f"✅ SHORT Entry features: {len(short_entry_features)} (includes 10 NEW)")
print(f"✅ Exit features (both): {len(exit_features)}")
print()

# Verify features exist in dataset
for feature_list, name in [(long_entry_features, "LONG Entry"),
                             (short_entry_features, "SHORT Entry"),
                             (exit_features, "Exit")]:
    missing = [f for f in feature_list if f not in df.columns]
    if missing:
        print(f"⚠️  WARNING: {name} missing features: {missing[:5]}...")
        print(f"   Total missing: {len(missing)}/{len(feature_list)}")
    else:
        print(f"✅ All {name} features present")

print()

# ==============================================================================
# STEP 4: Label Generation Function (FIXED)
# ==============================================================================

print("-"*80)
print("STEP 4: Labeling Data (FIXED Method)")
print("-"*80)

def label_entry_opportunities_fixed(df, side, min_hold=MIN_HOLD_TIME, max_hold=MAX_HOLD_TIME,
                                     min_pnl=MIN_PNL, leverage=LEVERAGE, stop_loss=STOP_LOSS):
    """
    FIXED labeling: Same as production (20251103 verified)

    Args:
        df: DataFrame with OHLCV
        side: 'LONG' or 'SHORT'
        min_hold: Minimum hold time (candles)
        max_hold: Maximum hold time (candles)
        min_pnl: Minimum profit threshold (0.003 = 0.3%)
        leverage: Position leverage
        stop_loss: Stop loss threshold (-0.03 = -3%)

    Returns:
        Series of binary labels (1 = good entry, 0 = bad entry)
    """
    labels = []
    total_candles = len(df) - max_hold
    print_interval = total_candles // 20  # Print 20 progress updates

    for i in range(total_candles):
        # Progress indicator
        if i % print_interval == 0:
            progress = (i / total_candles) * 100
            print(f"   Labeling progress: {progress:.1f}% ({i:,}/{total_candles:,})", end='\r')

        entry_price = df.iloc[i]['close']

        # Simulate position for max_hold candles
        best_pnl = -999
        best_hold = 0

        for hold in range(1, max_hold + 1):
            if i + hold >= len(df):
                break

            exit_price = df.iloc[i + hold]['close']

            # Calculate P&L based on side
            if side == 'LONG':
                pnl_pct = ((exit_price - entry_price) / entry_price) * leverage
            else:  # SHORT
                pnl_pct = ((entry_price - exit_price) / entry_price) * leverage

            # Check stop loss
            if pnl_pct <= stop_loss:
                best_pnl = pnl_pct
                best_hold = hold
                break

            # Track best P&L
            if pnl_pct > best_pnl:
                best_pnl = pnl_pct
                best_hold = hold

        # Label as 1 (good) if: hold time valid AND P&L exceeds threshold
        if best_hold >= min_hold and best_pnl >= min_pnl:
            labels.append(1)
        else:
            labels.append(0)

    print(f"   Labeling progress: 100.0% ({total_candles:,}/{total_candles:,})")  # Final update

    # Pad end with 0s (no opportunity at end of dataset)
    labels += [0] * max_hold

    return pd.Series(labels, index=df.index)

print("⏳ Labeling LONG opportunities...")
train_df['long_label'] = label_entry_opportunities_fixed(train_df, 'LONG')
print(f"   LONG labels: {train_df['long_label'].sum():,} / {len(train_df):,} ({train_df['long_label'].mean()*100:.2f}%)")

print("⏳ Labeling SHORT opportunities...")
train_df['short_label'] = label_entry_opportunities_fixed(train_df, 'SHORT')
print(f"   SHORT labels: {train_df['short_label'].sum():,} / {len(train_df):,} ({train_df['short_label'].mean()*100:.2f}%)")

print()

# ==============================================================================
# STEP 5: Train Models
# ==============================================================================

print("-"*80)
print("STEP 5: Training Models (5-Fold CV)")
print("-"*80)
print()

timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')

# Model 1: LONG Entry
print("🔵 Training LONG Entry Model...")
X_long = train_df[long_entry_features].fillna(0).replace([np.inf, -np.inf], 0)
y_long = train_df['long_label']

scaler_long = StandardScaler()
X_long_scaled = scaler_long.fit_transform(X_long)

model_long = xgb.XGBClassifier(
    max_depth=6,
    learning_rate=0.1,
    n_estimators=200,
    objective='binary:logistic',
    random_state=42,
    n_jobs=-1
)

# 5-Fold CV
cv_scores_long = cross_val_score(model_long, X_long_scaled, y_long, cv=5, scoring='accuracy')
print(f"   5-Fold CV Accuracy: {cv_scores_long.mean():.4f} ± {cv_scores_long.std():.4f}")

# Train on full training set
model_long.fit(X_long_scaled, y_long)

# Save
model_path_long = MODELS_DIR / f"xgboost_long_entry_90days_{timestamp_str}.pkl"
scaler_path_long = MODELS_DIR / f"xgboost_long_entry_90days_{timestamp_str}_scaler.pkl"
features_path_long = MODELS_DIR / f"xgboost_long_entry_90days_{timestamp_str}_features.txt"

joblib.dump(model_long, model_path_long)
joblib.dump(scaler_long, scaler_path_long)
with open(features_path_long, 'w') as f:
    f.write('\n'.join(long_entry_features))

print(f"   ✅ Saved: {model_path_long.name}")
print()

# Model 2: SHORT Entry
print("🔴 Training SHORT Entry Model...")
X_short = train_df[short_entry_features].fillna(0).replace([np.inf, -np.inf], 0)
y_short = train_df['short_label']

scaler_short = StandardScaler()
X_short_scaled = scaler_short.fit_transform(X_short)

model_short = xgb.XGBClassifier(
    max_depth=6,
    learning_rate=0.1,
    n_estimators=200,
    objective='binary:logistic',
    random_state=42,
    n_jobs=-1
)

# 5-Fold CV
cv_scores_short = cross_val_score(model_short, X_short_scaled, y_short, cv=5, scoring='accuracy')
print(f"   5-Fold CV Accuracy: {cv_scores_short.mean():.4f} ± {cv_scores_short.std():.4f}")

# Train on full training set
model_short.fit(X_short_scaled, y_short)

# Save
model_path_short = MODELS_DIR / f"xgboost_short_entry_90days_{timestamp_str}.pkl"
scaler_path_short = MODELS_DIR / f"xgboost_short_entry_90days_{timestamp_str}_scaler.pkl"
features_path_short = MODELS_DIR / f"xgboost_short_entry_90days_{timestamp_str}_features.txt"

joblib.dump(model_short, model_path_short)
joblib.dump(scaler_short, scaler_path_short)
with open(features_path_short, 'w') as f:
    f.write('\n'.join(short_entry_features))

print(f"   ✅ Saved: {model_path_short.name}")
print()

# ==============================================================================
# STEP 6: Label Exit Opportunities (In-Position Labeling)
# ==============================================================================

print("-"*80)
print("STEP 6: Labeling Exit Opportunities (In-Position)")
print("-"*80)

def label_exit_opportunities_in_position(df, entry_labels, side, max_hold=MAX_HOLD_TIME,
                                          leverage=LEVERAGE, stop_loss=STOP_LOSS):
    """
    Label exit opportunities while IN position (not from market-wide view)

    This matches production logic: We only consider exits AFTER entry signals
    """
    exit_labels = pd.Series([0] * len(df), index=df.index)

    # Find all entry points
    entry_indices = df[entry_labels == 1].index
    total_entries = len(entry_indices)
    print_interval = max(1, total_entries // 20)  # Print 20 progress updates

    for idx, entry_idx in enumerate(entry_indices):
        # Progress indicator
        if idx % print_interval == 0:
            progress = (idx / total_entries) * 100
            print(f"   Exit labeling progress: {progress:.1f}% ({idx:,}/{total_entries:,})", end='\r')

        entry_i = df.index.get_loc(entry_idx)
        if entry_i + max_hold >= len(df):
            continue

        entry_price = df.iloc[entry_i]['close']

        # Simulate position for max_hold candles
        for hold in range(1, max_hold + 1):
            if entry_i + hold >= len(df):
                break

            exit_idx = df.index[entry_i + hold]
            exit_price = df.iloc[entry_i + hold]['close']

            # Calculate P&L
            if side == 'LONG':
                pnl_pct = ((exit_price - entry_price) / entry_price) * leverage
            else:  # SHORT
                pnl_pct = ((entry_price - exit_price) / entry_price) * leverage

            # Check stop loss
            if pnl_pct <= stop_loss:
                exit_labels[exit_idx] = 1  # Exit immediately
                break

            # Label exits with positive P&L (even small gains)
            if pnl_pct >= 0.001:  # 0.1% profit
                exit_labels[exit_idx] = 1

    print(f"   Exit labeling progress: 100.0% ({total_entries:,}/{total_entries:,})")  # Final update

    return exit_labels

print("⏳ Labeling LONG exit opportunities...")
train_df['long_exit_label'] = label_exit_opportunities_in_position(
    train_df, train_df['long_label'], 'LONG'
)
print(f"   LONG exit labels: {train_df['long_exit_label'].sum():,} / {len(train_df):,} ({train_df['long_exit_label'].mean()*100:.2f}%)")

print("⏳ Labeling SHORT exit opportunities...")
train_df['short_exit_label'] = label_exit_opportunities_in_position(
    train_df, train_df['short_label'], 'SHORT'
)
print(f"   SHORT exit labels: {train_df['short_exit_label'].sum():,} / {len(train_df):,} ({train_df['short_exit_label'].mean()*100:.2f}%)")

print()

# ==============================================================================
# STEP 7: Train Exit Models
# ==============================================================================

print("-"*80)
print("STEP 7: Training Exit Models (5-Fold CV)")
print("-"*80)
print()

# Model 3: LONG Exit
print("🔵 Training LONG Exit Model...")
X_long_exit = train_df[exit_features].fillna(0).replace([np.inf, -np.inf], 0)
y_long_exit = train_df['long_exit_label']

scaler_long_exit = StandardScaler()
X_long_exit_scaled = scaler_long_exit.fit_transform(X_long_exit)

model_long_exit = xgb.XGBClassifier(
    max_depth=5,
    learning_rate=0.1,
    n_estimators=150,
    objective='binary:logistic',
    random_state=42,
    n_jobs=-1
)

# 5-Fold CV
cv_scores_long_exit = cross_val_score(model_long_exit, X_long_exit_scaled, y_long_exit, cv=5, scoring='accuracy')
print(f"   5-Fold CV Accuracy: {cv_scores_long_exit.mean():.4f} ± {cv_scores_long_exit.std():.4f}")

# Train on full training set
model_long_exit.fit(X_long_exit_scaled, y_long_exit)

# Save
model_path_long_exit = MODELS_DIR / f"xgboost_long_exit_90days_{timestamp_str}.pkl"
scaler_path_long_exit = MODELS_DIR / f"xgboost_long_exit_90days_{timestamp_str}_scaler.pkl"
features_path_long_exit = MODELS_DIR / f"xgboost_long_exit_90days_{timestamp_str}_features.txt"

joblib.dump(model_long_exit, model_path_long_exit)
joblib.dump(scaler_long_exit, scaler_path_long_exit)
with open(features_path_long_exit, 'w') as f:
    f.write('\n'.join(exit_features))

print(f"   ✅ Saved: {model_path_long_exit.name}")
print()

# Model 4: SHORT Exit
print("🔴 Training SHORT Exit Model...")
X_short_exit = train_df[exit_features].fillna(0).replace([np.inf, -np.inf], 0)
y_short_exit = train_df['short_exit_label']

scaler_short_exit = StandardScaler()
X_short_exit_scaled = scaler_short_exit.fit_transform(X_short_exit)

model_short_exit = xgb.XGBClassifier(
    max_depth=5,
    learning_rate=0.1,
    n_estimators=150,
    objective='binary:logistic',
    random_state=42,
    n_jobs=-1
)

# 5-Fold CV
cv_scores_short_exit = cross_val_score(model_short_exit, X_short_exit_scaled, y_short_exit, cv=5, scoring='accuracy')
print(f"   5-Fold CV Accuracy: {cv_scores_short_exit.mean():.4f} ± {cv_scores_short_exit.std():.4f}")

# Train on full training set
model_short_exit.fit(X_short_exit_scaled, y_short_exit)

# Save
model_path_short_exit = MODELS_DIR / f"xgboost_short_exit_90days_{timestamp_str}.pkl"
scaler_path_short_exit = MODELS_DIR / f"xgboost_short_exit_90days_{timestamp_str}_scaler.pkl"
features_path_short_exit = MODELS_DIR / f"xgboost_short_exit_90days_{timestamp_str}_features.txt"

joblib.dump(model_short_exit, model_path_short_exit)
joblib.dump(scaler_short_exit, scaler_path_short_exit)
with open(features_path_short_exit, 'w') as f:
    f.write('\n'.join(exit_features))

print(f"   ✅ Saved: {model_path_short_exit.name}")
print()

# ==============================================================================
# STEP 8: Summary
# ==============================================================================

print("="*80)
print("TRAINING COMPLETE - ALL 4 MODELS RETRAINED")
print("="*80)
print()
print("📊 Training Summary:")
print(f"   Training Period: {train_df['timestamp'].min()} to {train_df['timestamp'].max()}")
print(f"   Training Candles: {len(train_df):,}")
print(f"   Training Days: {(train_df['timestamp'].max() - train_df['timestamp'].min()).days}")
print()
print("✅ Models Saved:")
print(f"   1. LONG Entry:  {model_path_long.name}")
print(f"      Features: {len(long_entry_features)}, CV Acc: {cv_scores_long.mean():.4f}")
print(f"   2. SHORT Entry: {model_path_short.name}")
print(f"      Features: {len(short_entry_features)}, CV Acc: {cv_scores_short.mean():.4f}")
print(f"   3. LONG Exit:   {model_path_long_exit.name}")
print(f"      Features: {len(exit_features)}, CV Acc: {cv_scores_long_exit.mean():.4f}")
print(f"   4. SHORT Exit:  {model_path_short_exit.name}")
print(f"      Features: {len(exit_features)}, CV Acc: {cv_scores_short_exit.mean():.4f}")
print()
print("📈 Label Distribution:")
print(f"   LONG Entry: {train_df['long_label'].mean()*100:.2f}% positive")
print(f"   SHORT Entry: {train_df['short_label'].mean()*100:.2f}% positive")
print(f"   LONG Exit: {train_df['long_exit_label'].mean()*100:.2f}% positive")
print(f"   SHORT Exit: {train_df['short_exit_label'].mean()*100:.2f}% positive")
print()
print("="*80)
print("NEXT STEPS")
print("="*80)
print()
print("1. Backtest new models on validation period (28 days)")
print("   → Compare Entry 0.80, Exit 0.80 vs current production")
print()
print("2. Performance comparison:")
print("   → Return, Win Rate, Profit Factor, Trade Frequency")
print("   → LONG: Current (104d) vs New (62d)")
print("   → SHORT: Current (35d) vs New (62d)")
print()
print("3. Deployment decision:")
print("   → IF new models > current → Deploy new models")
print("   → IF new models ≈ current → Keep current (more data didn't help)")
print("   → IF new models < current → Investigate why (regime mismatch?)")
print()
print(f"✅ Retraining complete! Timestamp: {timestamp_str}")
print()
