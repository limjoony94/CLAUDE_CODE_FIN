"""
Retrain on Full Period - Walk-Forward Decoupled (No Holdout)
=============================================================

Addresses market regime mismatch by training on FULL period including both:
  - Calm market: Jul 14 - Sep 28 (2.90% volatility)
  - Volatile market: Sep 28 - Oct 26 (4.96% volatility, 1.70x higher)

Key Changes from Phase 2:
  ❌ NO validation holdout (train on all 30,004 candles)
  ✅ Same Walk-Forward Decoupled methodology
  ✅ Same 195 enhanced features
  ✅ Learn both market regimes in training

Rationale:
  - Phase 2 models failed on volatile period (-59% return)
  - Both calm and volatile regimes needed in training data
  - Walk-Forward still prevents look-ahead bias
  - Models will learn to handle volatility shifts

Created: 2025-11-02
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import pickle
import joblib
from datetime import datetime
import xgboost as xgb
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score

# Configuration
ENTRY_THRESHOLD = 0.75
ML_EXIT_THRESHOLD = 0.75
LEVERAGE = 4
EMERGENCY_STOP_LOSS = -0.03
EMERGENCY_MAX_HOLD = 120

DATA_DIR = PROJECT_ROOT / "data" / "features"
MODELS_DIR = PROJECT_ROOT / "models"

print("="*80)
print("FULL PERIOD WALK-FORWARD DECOUPLED TRAINING")
print("="*80)
print()
print("Training Configuration:")
print(f"  Dataset: BTCUSDT_5m_features_complete.csv (195 features)")
print(f"  Training: ALL 30,004 candles (NO validation holdout)")
print(f"  Entry Threshold: {ENTRY_THRESHOLD}")
print(f"  ML Exit Threshold: {ML_EXIT_THRESHOLD}")
print(f"  Emergency Stop Loss: {EMERGENCY_STOP_LOSS * 100}%")
print(f"  Emergency Max Hold: {EMERGENCY_MAX_HOLD} candles (10 hours)")
print(f"  Leverage: {LEVERAGE}x")
print()
print("Rationale:")
print("  ⚠️  Previous models failed on volatile period (-59% return)")
print("  ✅ Train on both calm (2.90% vol) and volatile (4.96% vol) regimes")
print("  ✅ Walk-Forward prevents look-ahead bias within full period")
print("  ✅ Models learn to handle volatility shifts")
print()
print("Methodology:")
print("  ✅ Walk-Forward Validation (5-fold TimeSeriesSplit)")
print("  ✅ Filtered Simulation (83% faster)")
print("  ✅ Decoupled Training (no circular dependency)")
print("  ✅ Enhanced Features (volatility_20, sma_50, ema_26)")
print()

# ==============================================================================
# STEP 1: Load Enhanced Features Dataset
# ==============================================================================

print("-"*80)
print("STEP 1: Loading Complete Features Dataset (Full Period)")
print("-"*80)

df = pd.read_csv(DATA_DIR / "BTCUSDT_5m_features_complete.csv")
df['timestamp'] = pd.to_datetime(df['timestamp'])

print(f"✅ Loaded {len(df):,} candles with {len(df.columns)} features")
print(f"   Date range: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")
print(f"   Period: {(df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]).days} days")
print()

# NO HOLDOUT - use all data for training
df_train_full = df.copy()

print(f"Training Data:")
print(f"  Total: {len(df_train_full):,} candles (100% - NO holdout)")
print(f"  Date range: {df_train_full['timestamp'].iloc[0]} to {df_train_full['timestamp'].iloc[-1]}")
print(f"  Includes: Both calm and volatile market regimes")
print()

# ==============================================================================
# STEP 2: Load Entry Feature Lists
# ==============================================================================

print("-"*80)
print("STEP 2: Loading Entry Feature Lists")
print("-"*80)

# Load from existing Phase 2 models (same features)
long_entry_features_path = MODELS_DIR / "xgboost_long_entry_phase2_enhanced_20251102_200202_features.txt"
short_entry_features_path = MODELS_DIR / "xgboost_short_entry_phase2_enhanced_20251102_200202_features.txt"

with open(long_entry_features_path, 'r') as f:
    long_entry_features = [line.strip() for line in f.readlines()]

with open(short_entry_features_path, 'r') as f:
    short_entry_features = [line.strip() for line in f.readlines()]

print(f"  ✅ LONG Entry: {len(long_entry_features)} features")
print(f"  ✅ SHORT Entry: {len(short_entry_features)} features")
print()

# ==============================================================================
# STEP 3: Define Exit Feature Lists (Enhanced)
# ==============================================================================

print("-"*80)
print("STEP 3: Defining Exit Feature Lists (Enhanced)")
print("-"*80)

EXIT_FEATURES = [
    # Core technical indicators (13 features)
    'rsi', 'macd', 'macd_signal',
    'bb_high', 'bb_mid', 'bb_low',
    'atr', 'sma_20', 'sma_50',  # COMPLETE MA features
    'ema_12', 'ema_26',  # EMA features
    'volume_sma',  # Volume indicator
    'volatility_20',  # EXACT volatility feature

    # Enhanced features (14 features)
    'volume_surge', 'price_acceleration',
    'price_vs_ma20', 'price_vs_ma50',
    'rsi_slope', 'rsi_overbought', 'rsi_oversold', 'rsi_divergence',
    'macd_histogram_slope', 'macd_crossover', 'macd_crossunder',
    'bb_position', 'higher_high', 'near_support'
]

print(f"  ✅ Exit Features: {len(EXIT_FEATURES)} features")
print(f"     Includes: volatility_20, sma_50, ema_26 (exact features)")
print()

# ==============================================================================
# STEP 4: Train Entry Models (Walk-Forward + Enhanced)
# ==============================================================================

print("="*80)
print("STEP 4: Training Entry Models (Walk-Forward + Full Period)")
print("="*80)
print()

def prepare_exit_features(df):
    """Prepare enhanced exit features matching exact column names"""
    exit_df = df[EXIT_FEATURES].copy()
    return exit_df

def create_rule_based_exit_labels(df, side='LONG'):
    """Create rule-based exit labels using 2-of-3 criteria"""
    exit_labels = []

    for i in range(len(df)):
        current_price = df['close'].iloc[i]

        # Scan forward for exit
        good_exit = False
        for j in range(i + 1, min(i + EMERGENCY_MAX_HOLD + 1, len(df))):
            future_price = df['close'].iloc[j]
            hold_time = j - i

            if side == 'LONG':
                pnl = (future_price - current_price) / current_price
            else:  # SHORT
                pnl = (current_price - future_price) / current_price

            leveraged_pnl = pnl * LEVERAGE

            # 2-of-3 criteria
            criteria_met = 0
            if leveraged_pnl > 0.02: criteria_met += 1
            if hold_time <= 60: criteria_met += 1
            if leveraged_pnl > 0.01: criteria_met += 1

            if criteria_met >= 2:
                good_exit = True
                break

            if leveraged_pnl < EMERGENCY_STOP_LOSS:
                break

        exit_labels.append(1 if good_exit else 0)

    return np.array(exit_labels)

def filter_entry_candidates(df, side='LONG'):
    """Pre-filter entry candidates using heuristics"""
    if side == 'LONG':
        mask = (
            (df['rsi'] < 60) &
            (df['macd'] > df['macd_signal']) &
            (df['volume'] > df['volume_sma'] * 0.8)
        )
    else:  # SHORT
        mask = (
            (df['rsi'] > 40) &
            (df['macd'] < df['macd_signal']) &
            (df['volume'] > df['volume_sma'] * 0.8)
        )

    return df[mask].copy()

def simulate_trade_outcomes(df, exit_model, scaler, side='LONG'):
    """Simulate trades to label entry points"""
    entry_labels = []

    for i in range(len(df)):
        current_price = df['close'].iloc[i]

        # Simulate trade
        good_entry = False
        for j in range(i + 1, min(i + EMERGENCY_MAX_HOLD + 1, len(df))):
            exit_features_raw = prepare_exit_features(df.iloc[[j]])
            exit_features_scaled = scaler.transform(exit_features_raw)

            exit_prob = exit_model.predict_proba(exit_features_scaled)[0][1]

            future_price = df['close'].iloc[j]
            hold_time = j - i

            if side == 'LONG':
                pnl = (future_price - current_price) / current_price
            else:
                pnl = (current_price - future_price) / current_price

            leveraged_pnl = pnl * LEVERAGE

            # 2-of-3 criteria
            criteria_met = 0
            if leveraged_pnl > 0.02: criteria_met += 1
            if hold_time <= 60: criteria_met += 1
            if exit_prob > ML_EXIT_THRESHOLD: criteria_met += 1

            if criteria_met >= 2:
                good_entry = True
                break

            if leveraged_pnl < EMERGENCY_STOP_LOSS or hold_time >= EMERGENCY_MAX_HOLD:
                break

        entry_labels.append(1 if good_entry else 0)

    return np.array(entry_labels)

def train_entry_model_walkforward(df, side='LONG', features=None):
    """Train entry model using Walk-Forward Decoupled"""
    print(f"\n{'-'*80}")
    print(f"Training {side} Entry Model")
    print(f"{'-'*80}\n")

    tscv = TimeSeriesSplit(n_splits=5)

    best_model = None
    best_scaler = None
    best_score = 0
    best_fold = 0

    print("="*80)
    print(f"Walk-Forward Training: {side} Entry Model")
    print("="*80)
    print()

    for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(df), 1):
        print("─"*80)
        print(f"FOLD {fold_idx}/5")
        print("─"*80)

        df_train_fold = df.iloc[train_idx].copy()
        df_val_fold = df.iloc[val_idx].copy()

        print(f"  Train: {len(df_train_fold):,} samples")
        print(f"  Val: {len(df_val_fold):,} samples")
        print()

        # Step 1: Train Exit model for this fold
        print(f"  Step 1: Training Exit model on Fold {fold_idx}...")
        exit_labels_train = create_rule_based_exit_labels(df_train_fold, side)
        print(f"  🔨 Creating Rule-Based Exit Labels for {side}...")
        print(f"     Exit Labels: {exit_labels_train.sum()} good exits ({exit_labels_train.mean()*100:.2f}%)")

        exit_features_train = prepare_exit_features(df_train_fold)

        exit_scaler = MinMaxScaler()
        exit_features_scaled = exit_scaler.fit_transform(exit_features_train)

        exit_model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            random_state=42
        )
        exit_model.fit(exit_features_scaled, exit_labels_train)

        # Step 2: Label Entry points using Exit model
        print(f"  🔍 Filtering {side} Entry Candidates...")
        df_filtered = filter_entry_candidates(df_train_fold, side)
        print(f"     Filtered: {len(df_filtered):,} candidates ({(1 - len(df_filtered)/len(df_train_fold))*100:.1f}% reduction)")

        print(f"  🎯 Labeling {side} Entries via Simulation...")
        entry_labels_train = simulate_trade_outcomes(df_filtered, exit_model, exit_scaler, side)
        print(f"     Entry Labels: {entry_labels_train.sum()} good entries ({entry_labels_train.mean()*100:.2f}%)")
        print()

        # Step 3: Train Entry model
        print(f"  Step 2: Training {side} Entry model on Fold {fold_idx}...")
        entry_features_train = df_filtered[features].copy()

        entry_scaler = MinMaxScaler()
        entry_features_scaled = entry_scaler.fit_transform(entry_features_train)

        entry_model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42
        )
        entry_model.fit(entry_features_scaled, entry_labels_train)

        # Validate
        df_filtered_val = filter_entry_candidates(df_val_fold, side)
        if len(df_filtered_val) > 0:
            entry_features_val = df_filtered_val[features].copy()
            entry_features_val_scaled = entry_scaler.transform(entry_features_val)

            pred_probs = entry_model.predict_proba(entry_features_val_scaled)[:, 1]
            pred_rate = (pred_probs >= ENTRY_THRESHOLD).mean()

            print(f"  ✅ Validation prediction rate: {pred_rate*100:.2f}%")
            print()

            if pred_rate > best_score:
                best_score = pred_rate
                best_model = entry_model
                best_scaler = entry_scaler
                best_fold = fold_idx

    print(f"✅ Best Model: Fold {best_fold} (pred rate: {best_score*100:.2f}%)")
    print()

    return best_model, best_scaler

# Train LONG Entry
long_entry_model, long_entry_scaler = train_entry_model_walkforward(
    df_train_full, 'LONG', long_entry_features
)

# Train SHORT Entry
short_entry_model, short_entry_scaler = train_entry_model_walkforward(
    df_train_full, 'SHORT', short_entry_features
)

# ==============================================================================
# STEP 5: Train Exit Models (Walk-Forward + Enhanced)
# ==============================================================================

print("="*80)
print("STEP 5: Training Exit Models (Walk-Forward + Full Period)")
print("="*80)
print()

def train_exit_model_walkforward(df, side='LONG'):
    """Train exit model using Walk-Forward"""
    print(f"\n{'-'*80}")
    print(f"Training {side} Exit Model")
    print(f"{'-'*80}\n")

    tscv = TimeSeriesSplit(n_splits=5)

    best_model = None
    best_scaler = None
    best_score = 0
    best_fold = 0

    print("="*80)
    print(f"Walk-Forward Training: {side} Exit Model")
    print("="*80)
    print()

    for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(df), 1):
        print("─"*80)
        print(f"FOLD {fold_idx}/5")
        print("─"*80)

        df_train_fold = df.iloc[train_idx].copy()
        df_val_fold = df.iloc[val_idx].copy()

        # Create labels
        exit_labels_train = create_rule_based_exit_labels(df_train_fold, side)
        exit_labels_val = create_rule_based_exit_labels(df_val_fold, side)

        print(f"  🔨 Creating Rule-Based Exit Labels for {side}...")
        print(f"     Exit Labels: {exit_labels_train.sum()} good exits ({exit_labels_train.mean()*100:.2f}%)")

        # Prepare features
        exit_features_train = prepare_exit_features(df_train_fold)
        exit_features_val = prepare_exit_features(df_val_fold)

        # Scale
        exit_scaler = MinMaxScaler()
        exit_features_train_scaled = exit_scaler.fit_transform(exit_features_train)
        exit_features_val_scaled = exit_scaler.transform(exit_features_val)

        # Train
        exit_model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            random_state=42
        )
        exit_model.fit(exit_features_train_scaled, exit_labels_train)

        # Validate
        val_acc = accuracy_score(exit_labels_val, exit_model.predict(exit_features_val_scaled))
        print(f"  🔨 Creating Rule-Based Exit Labels for {side}...")
        print(f"     Exit Labels: {exit_labels_val.sum()} good exits ({exit_labels_val.mean()*100:.2f}%)")
        print(f"  ✅ Validation accuracy: {val_acc*100:.2f}%")
        print()

        if val_acc > best_score:
            best_score = val_acc
            best_model = exit_model
            best_scaler = exit_scaler
            best_fold = fold_idx

    print(f"✅ Best Model: Fold {best_fold} (accuracy: {best_score*100:.2f}%)")
    print()

    return best_model, best_scaler

# Train LONG Exit
long_exit_model, long_exit_scaler = train_exit_model_walkforward(df_train_full, 'LONG')

# Train SHORT Exit
short_exit_model, short_exit_scaler = train_exit_model_walkforward(df_train_full, 'SHORT')

# ==============================================================================
# STEP 6: Save Models
# ==============================================================================

print("="*80)
print("STEP 6: Saving Models")
print("="*80)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# Save LONG Entry
long_entry_path = MODELS_DIR / f"xgboost_long_entry_full_period_{timestamp}.pkl"
joblib.dump(long_entry_model, long_entry_path)
joblib.dump(long_entry_scaler, str(long_entry_path).replace('.pkl', '_scaler.pkl'))
with open(str(long_entry_path).replace('.pkl', '_features.txt'), 'w') as f:
    f.write('\n'.join(long_entry_features))
print(f"✅ LONG Entry: {long_entry_path.name}")
print(f"   Features: {len(long_entry_features)}")

# Save SHORT Entry
short_entry_path = MODELS_DIR / f"xgboost_short_entry_full_period_{timestamp}.pkl"
joblib.dump(short_entry_model, short_entry_path)
joblib.dump(short_entry_scaler, str(short_entry_path).replace('.pkl', '_scaler.pkl'))
with open(str(short_entry_path).replace('.pkl', '_features.txt'), 'w') as f:
    f.write('\n'.join(short_entry_features))
print(f"✅ SHORT Entry: {short_entry_path.name}")
print(f"   Features: {len(short_entry_features)}")

# Save LONG Exit
long_exit_path = MODELS_DIR / f"xgboost_long_exit_full_period_{timestamp}.pkl"
joblib.dump(long_exit_model, long_exit_path)
joblib.dump(long_exit_scaler, str(long_exit_path).replace('.pkl', '_scaler.pkl'))
with open(str(long_exit_path).replace('.pkl', '_features.txt'), 'w') as f:
    f.write('\n'.join(EXIT_FEATURES))
print(f"✅ LONG Exit: {long_exit_path.name}")
print(f"   Features: {len(EXIT_FEATURES)}")

# Save SHORT Exit
short_exit_path = MODELS_DIR / f"xgboost_short_exit_full_period_{timestamp}.pkl"
joblib.dump(short_exit_model, short_exit_path)
joblib.dump(short_exit_scaler, str(short_exit_path).replace('.pkl', '_scaler.pkl'))
with open(str(short_exit_path).replace('.pkl', '_features.txt'), 'w') as f:
    f.write('\n'.join(EXIT_FEATURES))
print(f"✅ SHORT Exit: {short_exit_path.name}")
print(f"   Features: {len(EXIT_FEATURES)}")

print()
print("="*80)
print("FULL PERIOD TRAINING COMPLETE")
print("="*80)
print()
print("Models Trained:")
print("  ✅ LONG Entry (Walk-Forward Decoupled - Full Period)")
print("  ✅ SHORT Entry (Walk-Forward Decoupled - Full Period)")
print("  ✅ LONG Exit (Walk-Forward - Full Period)")
print("  ✅ SHORT Exit (Walk-Forward - Full Period)")
print()
print("Training Data:")
print(f"  Period: {(df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]).days} days")
print(f"  Candles: {len(df):,} (100% - NO holdout)")
print(f"  Volatility: Trained on both calm and volatile regimes")
print()
print("Key Improvements:")
print("  ✅ Includes volatile market regime (Sep 28 - Oct 26)")
print("  ✅ Models learn to handle volatility shifts")
print("  ✅ Walk-Forward prevents look-ahead bias")
print("  ✅ Same enhanced features (195 total)")
print()
print("Next Steps:")
print("  1. Test on NEW unseen data (Oct 27+) for validation")
print("  2. OR deploy for live testing with close monitoring")
print("  3. Compare live performance to Phase 2 models")
print()
print(f"Models saved with timestamp: {timestamp}")
