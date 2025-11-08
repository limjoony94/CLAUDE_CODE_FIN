"""
Retrain SHORT Entry Model with New SHORT-Specific Features
============================================================

Problem: SHORT signals disappeared in Nov 2025 falling market
  - Oct 20-25: 37% avg SHORT prob, max 96%
  - Nov 3-4: 10% avg SHORT prob, max 30% (NEVER reaches 0.80 threshold)

Root Cause: Model over-dependent on vwap_near_vp_support (dropped 97% in Nov)

Solution: Add 10 NEW SHORT-specific features to reduce VWAP dependency
  1. downtrend_strength (composite score)
  2. ema12_slope (negative = downtrend)
  3. consecutive_red_candles
  4. price_distance_from_high_pct
  5. price_below_ma200_pct
  6. price_below_ema12_pct
  7. volatility_expansion_down
  8. volume_on_down_days_ratio
  9. lower_highs_pattern
  10. below_multiple_mas

Expected Outcome:
  - SHORT probabilities improve in falling markets
  - Nov 3-4 should show >0.80 SHORT probs
  - Model less dependent on single feature

Created: 2025-11-04 21:30 KST
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, classification_report

# Import production feature calculation
from scripts.production.production_features_v1 import calculate_all_features_enhanced_v2
from scripts.production.production_exit_features_v1 import prepare_exit_features

# Configuration
ENTRY_THRESHOLD = 0.80
ML_EXIT_THRESHOLD = 0.70
LEVERAGE = 4
EMERGENCY_STOP_LOSS = -0.03
MAX_HOLD = 120

# Labeling criteria (same as current production)
MIN_HOLD_TIME = 12  # 1 hour (12 × 5 min)
MAX_HOLD_TIME = 144  # 12 hours (144 × 5 min)
MIN_PNL = 0.003  # 0.3% profit minimum

DATA_DIR = PROJECT_ROOT / "data" / "features"
MODELS_DIR = PROJECT_ROOT / "models"

print("="*80)
print("RETRAIN SHORT ENTRY MODEL WITH NEW SHORT-SPECIFIC FEATURES")
print("="*80)
print()
print("🎯 Problem: SHORT signals disappeared in Nov 2025")
print("   - Oct 20-25: 37% avg SHORT prob (worked)")
print("   - Nov 3-4: 10% avg SHORT prob (failed)")
print()
print("🔧 Solution: Add 10 NEW SHORT-specific features")
print("   Reduces dependency on vwap_near_vp_support")
print()
print("Configuration:")
print(f"  Dataset: 35 days (Sep 30 - Nov 4, 2025)")
print(f"  Features: OLD (85) + NEW (10) = 95 features")
print(f"  Entry Threshold: {ENTRY_THRESHOLD}")
print(f"  ML Exit Threshold: {ML_EXIT_THRESHOLD}")
print(f"  Training Split: 80/20 (TimeSeriesSplit)")
print(f"  Min Hold: {MIN_HOLD_TIME} candles ({MIN_HOLD_TIME/12:.1f}h)")
print(f"  Max Hold: {MAX_HOLD_TIME} candles ({MAX_HOLD_TIME/12:.1f}h)")
print(f"  Min PNL: {MIN_PNL*100}% (3× fees)")
print()

# ==============================================================================
# STEP 1: Load and Calculate Features
# ==============================================================================

print("-"*80)
print("STEP 1: Loading Raw Data and Calculating Features")
print("-"*80)

# Find latest 35-day dataset
csv_files = list(DATA_DIR.glob("BTCUSDT_5m_raw_35days_*.csv"))
if not csv_files:
    raise FileNotFoundError("No 35-day dataset found! Run fetch_35days_for_retraining.py first")

csv_file = sorted(csv_files, reverse=True)[0]
print(f"✅ Loading: {csv_file.name}")

df_raw = pd.read_csv(csv_file)
df_raw['timestamp'] = pd.to_datetime(df_raw['timestamp'])
for col in ['open', 'high', 'low', 'close', 'volume']:
    df_raw[col] = df_raw[col].astype(float)

print(f"   {len(df_raw):,} candles loaded")
print(f"   Date range: {df_raw['timestamp'].iloc[0]} to {df_raw['timestamp'].iloc[-1]}")
print()

print("⏳ Calculating features (with NEW SHORT-specific features)...")
df = calculate_all_features_enhanced_v2(df_raw.copy(), phase='phase1')
df = prepare_exit_features(df)

print()
print(f"✅ Features calculated: {len(df)} candles (lost {len(df_raw) - len(df)} to lookback)")
print(f"   Total features: {len(df.columns) - 6} (OHLCV + timestamp excluded)")
print()

# Check if new SHORT features exist
new_short_features = [
    'downtrend_strength', 'ema12_slope', 'consecutive_red_candles',
    'price_distance_from_high_pct', 'price_below_ma200_pct', 'price_below_ema12_pct',
    'volatility_expansion_down', 'volume_on_down_days_ratio', 'lower_highs_pattern',
    'below_multiple_mas'
]

missing_features = [f for f in new_short_features if f not in df.columns]
if missing_features:
    print(f"⚠️  WARNING: Missing NEW features: {missing_features}")
    print("   Check if calculate_short_specific_features() was integrated correctly")
else:
    print(f"✅ All 10 NEW SHORT features present:")
    for feat in new_short_features:
        print(f"   - {feat}")
print()

# ==============================================================================
# STEP 2: Load OLD Feature List and Add NEW Features
# ==============================================================================

print("-"*80)
print("STEP 2: Defining Feature List (OLD + NEW)")
print("-"*80)

# Load existing SHORT entry features (85 features from Enhanced 5-Fold CV)
old_features_path = MODELS_DIR / "xgboost_short_entry_enhanced_20251024_012445_features.txt"

with open(old_features_path, 'r') as f:
    old_features = [line.strip() for line in f.readlines()]

print(f"✅ OLD SHORT features: {len(old_features)}")

# Add NEW SHORT-specific features
short_entry_features = old_features + new_short_features

print(f"✅ NEW SHORT features: {len(new_short_features)}")
print(f"✅ TOTAL SHORT features: {len(short_entry_features)}")
print()

# Verify all features exist in dataframe
missing = [f for f in short_entry_features if f not in df.columns]
if missing:
    print(f"❌ ERROR: Missing features in dataframe: {missing}")
    sys.exit(1)

print("✅ All features present in dataframe")
print()

# ==============================================================================
# STEP 3: Create Labels for SHORT Entry
# ==============================================================================

print("-"*80)
print("STEP 3: Creating Labels for SHORT Entry")
print("-"*80)

def create_short_entry_labels(df, min_hold=MIN_HOLD_TIME, max_hold=MAX_HOLD_TIME, min_pnl=MIN_PNL):
    """Label = 1 if profitable SHORT entry within time window"""
    labels = np.zeros(len(df), dtype=int)

    for i in range(len(df) - max_hold):
        entry_price = df.iloc[i]['close']

        # Look forward within time window
        for j in range(i + min_hold, min(i + max_hold + 1, len(df))):
            exit_price = df.iloc[j]['close']

            # SHORT profit = (entry - exit) / entry
            pnl = (entry_price - exit_price) / entry_price

            if pnl > min_pnl:
                labels[i] = 1
                break

    return labels

print("Creating SHORT labels...")
labels = create_short_entry_labels(df)

print(f"✅ Label distribution:")
print(f"   Positive (1): {labels.sum():,} ({labels.sum()/len(labels)*100:.1f}%)")
print(f"   Negative (0): {(labels == 0).sum():,} ({(labels == 0).sum()/len(labels)*100:.1f}%)")
print()

# ==============================================================================
# STEP 4: Train-Validation Split (80/20)
# ==============================================================================

print("-"*80)
print("STEP 4: Train-Validation Split (80/20)")
print("-"*80)

split_idx = int(len(df) * 0.8)

df_train = df.iloc[:split_idx].copy()
labels_train = labels[:split_idx]

df_val = df.iloc[split_idx:].copy()
labels_val = labels[split_idx:]

print(f"Training (80%): {len(df_train):,} candles")
print(f"  Date: {df_train['timestamp'].iloc[0]} to {df_train['timestamp'].iloc[-1]}")
print(f"  Labels: {labels_train.sum():,} positive ({labels_train.sum()/len(labels_train)*100:.1f}%)")
print()
print(f"Validation (20%): {len(df_val):,} candles")
print(f"  Date: {df_val['timestamp'].iloc[0]} to {df_val['timestamp'].iloc[-1]}")
print(f"  Labels: {labels_val.sum():,} positive ({labels_val.sum()/len(labels_val)*100:.1f}%)")
print()

# ==============================================================================
# STEP 5: Prepare Training Data
# ==============================================================================

print("-"*80)
print("STEP 5: Preparing Training Data")
print("-"*80)

X_train = df_train[short_entry_features].values
y_train = labels_train

X_val = df_val[short_entry_features].values
y_val = labels_val

print(f"X_train shape: {X_train.shape}")
print(f"X_val shape: {X_val.shape}")
print()

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

print("✅ Features standardized (StandardScaler)")
print()

# ==============================================================================
# STEP 6: Train Model with TimeSeriesSplit (5-Fold CV)
# ==============================================================================

print("-"*80)
print("STEP 6: Training SHORT Entry Model (5-Fold CV)")
print("-"*80)

tscv = TimeSeriesSplit(n_splits=5)

fold_scores = []
for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X_train_scaled), 1):
    X_fold_train = X_train_scaled[train_idx]
    y_fold_train = y_train[train_idx]
    X_fold_val = X_train_scaled[val_idx]
    y_fold_val = y_train[val_idx]

    # Train XGBoost
    model = xgb.XGBClassifier(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.1,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        eval_metric='logloss'
    )

    model.fit(
        X_fold_train, y_fold_train,
        eval_set=[(X_fold_val, y_fold_val)],
        verbose=False
    )

    # Evaluate
    y_pred = model.predict(X_fold_val)
    accuracy = accuracy_score(y_fold_val, y_pred)
    fold_scores.append(accuracy)

    print(f"Fold {fold_idx}/5: Accuracy = {accuracy:.4f} ({accuracy*100:.2f}%)")

print()
print(f"✅ Average CV Accuracy: {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f}")
print()

# ==============================================================================
# STEP 7: Train Final Model on Full Training Set
# ==============================================================================

print("-"*80)
print("STEP 7: Training Final SHORT Entry Model")
print("-"*80)

final_model = xgb.XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='logloss'
)

final_model.fit(
    X_train_scaled, y_train,
    eval_set=[(X_val_scaled, y_val)],
    verbose=True
)

print()

# Evaluate on validation set
y_val_pred = final_model.predict(X_val_scaled)
y_val_prob = final_model.predict_proba(X_val_scaled)[:, 1]

val_accuracy = accuracy_score(y_val, y_val_pred)
print()
print(f"✅ Validation Accuracy: {val_accuracy:.4f} ({val_accuracy*100:.2f}%)")
print()
print("Classification Report:")
print(classification_report(y_val, y_val_pred, target_names=['Not Entry', 'Entry']))
print()

# ==============================================================================
# STEP 8: Feature Importance Analysis
# ==============================================================================

print("-"*80)
print("STEP 8: Feature Importance Analysis")
print("-"*80)

feature_importance = final_model.feature_importances_
sorted_idx = np.argsort(feature_importance)[::-1]

print("Top 20 Most Important Features:")
for i, idx in enumerate(sorted_idx[:20], 1):
    feat_name = short_entry_features[idx]
    importance = feature_importance[idx]
    is_new = '🆕' if feat_name in new_short_features else '  '
    print(f"  {i:2d}. {is_new} {feat_name:40s} {importance:.6f}")

print()
print("NEW Feature Rankings:")
for feat in new_short_features:
    idx = short_entry_features.index(feat)
    importance = feature_importance[idx]
    rank = np.where(sorted_idx == idx)[0][0] + 1
    print(f"  Rank {rank:3d}: {feat:40s} {importance:.6f}")
print()

# ==============================================================================
# STEP 9: Test on Nov 3-4 (Falling Market Period)
# ==============================================================================

print("-"*80)
print("STEP 9: Testing on Nov 3-4 (Falling Market Period)")
print("-"*80)

# Filter Nov 3-4 data
nov_data = df[(df['timestamp'] >= '2025-11-03') & (df['timestamp'] <= '2025-11-04')]

if len(nov_data) > 0:
    X_nov = nov_data[short_entry_features].values
    X_nov_scaled = scaler.transform(X_nov)
    nov_probs = final_model.predict_proba(X_nov_scaled)[:, 1]

    print(f"📊 Nov 3-4 SHORT Signal Analysis:")
    print(f"   Total candles: {len(nov_data)}")
    print(f"   Avg SHORT prob: {nov_probs.mean():.4f} ({nov_probs.mean()*100:.2f}%)")
    print(f"   Max SHORT prob: {nov_probs.max():.4f} ({nov_probs.max()*100:.2f}%)")
    print(f"   Min SHORT prob: {nov_probs.min():.4f} ({nov_probs.min()*100:.2f}%)")
    print(f"   >0.80 threshold: {(nov_probs >= 0.80).sum()} ({(nov_probs >= 0.80).sum()/len(nov_data)*100:.1f}%)")
    print()

    if nov_probs.max() < 0.80:
        print("⚠️  WARNING: Max SHORT prob still <0.80 in Nov 3-4!")
        print("   Model may need more tuning or additional features")
    else:
        print("✅ SUCCESS: SHORT probabilities >0.80 achieved in Nov 3-4!")
        print(f"   {(nov_probs >= 0.80).sum()} candles exceed entry threshold")
else:
    print("⚠️  No Nov 3-4 data found in validation set")

print()

# ==============================================================================
# STEP 10: Save Model
# ==============================================================================

print("-"*80)
print("STEP 10: Saving Model and Scaler")
print("-"*80)

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
model_filename = f"xgboost_short_entry_with_new_features_{timestamp}.pkl"
scaler_filename = f"xgboost_short_entry_with_new_features_{timestamp}_scaler.pkl"
features_filename = f"xgboost_short_entry_with_new_features_{timestamp}_features.txt"

# Save model
model_path = MODELS_DIR / model_filename
joblib.dump(final_model, model_path)
print(f"✅ Model saved: {model_path}")

# Save scaler
scaler_path = MODELS_DIR / scaler_filename
joblib.dump(scaler, scaler_path)
print(f"✅ Scaler saved: {scaler_path}")

# Save feature list
features_path = MODELS_DIR / features_filename
with open(features_path, 'w') as f:
    for feat in short_entry_features:
        f.write(feat + '\n')
print(f"✅ Features saved: {features_path}")
print()

# ==============================================================================
# Summary
# ==============================================================================

print("="*80)
print("RETRAINING COMPLETE")
print("="*80)
print()
print("📊 Model Summary:")
print(f"   Features: {len(short_entry_features)} (85 OLD + 10 NEW)")
print(f"   Training: {len(X_train):,} candles")
print(f"   Validation: {len(X_val):,} candles")
print(f"   CV Accuracy: {np.mean(fold_scores):.4f} ± {np.std(fold_scores):.4f}")
print(f"   Val Accuracy: {val_accuracy:.4f}")
print()
print("📁 Files Created:")
print(f"   - {model_filename}")
print(f"   - {scaler_filename}")
print(f"   - {features_filename}")
print()
print("🎯 Next Steps:")
print("   1. Run backtest with new model on Nov 3-4 period")
print("   2. Compare SHORT signal quality (old vs new model)")
print("   3. If improved, deploy to production")
print()
print("="*80)
