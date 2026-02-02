"""
PHASE 1 RETRAINING WITH LATEST DATA - 14-Day Holdout
====================================================

Issue: Original Phase 1 models trained on Jul 14 - Sep 26 (30-day gap to present)
Solution: Retrain with latest market data

Training Strategy:
  Training Set: Jul 14 - Oct 12 (89 days, ~24,480 candles)
  Holdout Set: Oct 13 - Oct 26 (14 days, ~4,032 candles)

  Features: Phase 1 reduction level (LONG 80, SHORT 79)
  Methodology: Walk-Forward Decoupled (proven methodology)

  Goal: Capture recent market regime for better trade frequency

Expected Improvements:
  - Trade frequency > 2.0/day (vs 1.0/day with old models)
  - Better adaptation to October market conditions
  - Statistical reliability (>20 trades in 14-day holdout)

Created: 2025-10-29
"""

import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
import joblib
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb

# Configuration
ENTRY_THRESHOLD = 0.75
ML_EXIT_THRESHOLD = 0.75
EMERGENCY_STOP_LOSS = -0.03
EMERGENCY_MAX_HOLD = 120
LEVERAGE = 4
HOLDOUT_DAYS = 14

DATA_DIR = PROJECT_ROOT / "data" / "features"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

print("=" * 80)
print("PHASE 1 RETRAINING - LATEST DATA WITH 14-DAY HOLDOUT")
print("=" * 80)
print()
print("Strategy:")
print("  Training: Jul 14 - Oct 12 (89 days)")
print("  Holdout: Oct 13 - Oct 26 (14 days)")
print("  Features: Phase 1 reduction (LONG 80, SHORT 79)")
print("  Methodology: Walk-Forward Decoupled")
print()
print(f"Configuration:")
print(f"  Entry Threshold: {ENTRY_THRESHOLD}")
print(f"  ML Exit Threshold: {ML_EXIT_THRESHOLD}")
print(f"  Emergency Stop Loss: {EMERGENCY_STOP_LOSS*100}%")
print(f"  Emergency Max Hold: {EMERGENCY_MAX_HOLD} candles")
print(f"  Leverage: {LEVERAGE}x")
print()

# ============================================================================
# STEP 1: Load Full Features Dataset
# ============================================================================
print("-" * 80)
print("STEP 1: Loading Full Features Dataset")
print("-" * 80)

df_full = pd.read_csv(DATA_DIR / "BTCUSDT_5m_features.csv")
print(f"✅ Loaded {len(df_full):,} candles with {len(df_full.columns)} features")
print(f"   Date range: {df_full['timestamp'].iloc[0]} to {df_full['timestamp'].iloc[-1]}")
print()

# ============================================================================
# STEP 1.5: Split Training and Holdout (14-day holdout)
# ============================================================================
print("-" * 80)
print("STEP 1.5: Splitting Data (Training + 14-Day Holdout)")
print("-" * 80)

holdout_candles = HOLDOUT_DAYS * 24 * 12
training_end_idx = len(df_full) - holdout_candles

df_training = df_full.iloc[:training_end_idx].copy()
df_holdout = df_full.iloc[training_end_idx:].copy()

print(f"Total Data: {len(df_full):,} candles")
print(f"Training Set: {len(df_training):,} candles ({len(df_training)/(24*12):.0f} days)")
print(f"  Date Range: {df_training['timestamp'].iloc[0]} to {df_training['timestamp'].iloc[-1]}")
print(f"Holdout Set: {len(df_holdout):,} candles ({HOLDOUT_DAYS} days)")
print(f"  Date Range: {df_holdout['timestamp'].iloc[0]} to {df_holdout['timestamp'].iloc[-1]}")
print()

# ============================================================================
# STEP 2: Prepare Exit Features
# ============================================================================
print("-" * 80)
print("STEP 2: Preparing Exit Features")
print("-" * 80)

def prepare_exit_features(df):
    """Prepare EXIT features with enhanced market context"""
    df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
    df['volume_surge'] = (df['volume'] > df['volume'].rolling(20).mean() * 1.5).astype(float)

    if 'ma_50' in df.columns:
        df['price_vs_ma20'] = (df['close'] - df['ma_50']) / df['ma_50']
    else:
        ma_20 = df['close'].rolling(20).mean()
        df['price_vs_ma20'] = (df['close'] - ma_20) / ma_20

    if 'ma_200' in df.columns:
        df['price_vs_ma50'] = (df['close'] - df['ma_200']) / df['ma_200']
    else:
        ma_50 = df['close'].rolling(50).mean()
        df['price_vs_ma50'] = (df['close'] - ma_50) / ma_50

    df['returns'] = df['close'].pct_change()
    df['volatility_20'] = df['returns'].rolling(20).std()
    df['volatility_regime'] = (df['volatility_20'] > df['volatility_20'].rolling(100).median()).astype(float)
    df['rsi_slope'] = df['rsi'].diff(3) / 3
    df['rsi_overbought'] = (df['rsi'] > 70).astype(float)
    df['rsi_oversold'] = (df['rsi'] < 30).astype(float)
    df['rsi_divergence'] = 0
    df['macd_histogram_slope'] = df['macd_hist'].diff(3) / 3 if 'macd_hist' in df.columns else 0
    df['macd_crossover'] = ((df['macd'] > df['macd_signal']) &
                            (df['macd'].shift(1) <= df['macd_signal'].shift(1))).astype(float)
    df['macd_crossunder'] = ((df['macd'] < df['macd_signal']) &
                             (df['macd'].shift(1) >= df['macd_signal'].shift(1))).astype(float)
    df['higher_high'] = (df['high'] > df['high'].shift(1)).astype(float)
    df['lower_low'] = (df['low'] < df['low'].shift(1)).astype(float)
    df['price_acceleration'] = df['close'].diff().diff()
    df['near_resistance'] = 0
    df['near_support'] = 0

    if 'bb_upper' in df.columns and 'bb_lower' in df.columns:
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    else:
        df['bb_position'] = 0.5

    df = df.ffill().bfill()
    return df

print("Calculating enhanced market context features...")
df_training = prepare_exit_features(df_training)
print(f"✅ Enhanced exit features calculated")
print(f"✅ Exit features added - now {len(df_training.columns)} total features")
print()

# ============================================================================
# STEP 3: Load Phase 1 Feature Lists
# ============================================================================
print("-" * 80)
print("STEP 3: Loading Phase 1 Feature Lists")
print("-" * 80)

# Phase 1 feature lists from original training
with open(MODELS_DIR / "xgboost_long_entry_walkforward_reduced_phase1_20251029_050448_features.txt", 'r') as f:
    long_entry_features = [line.strip() for line in f if line.strip()]

with open(MODELS_DIR / "xgboost_short_entry_walkforward_reduced_phase1_20251029_050448_features.txt", 'r') as f:
    short_entry_features = [line.strip() for line in f if line.strip()]

print(f"  LONG Entry: {len(long_entry_features)} features (Phase 1)")
print(f"  SHORT Entry: {len(short_entry_features)} features (Phase 1)")
print()

# ============================================================================
# STEP 4: Load Exit Models for Walk-Forward Validation
# ============================================================================
print("-" * 80)
print("STEP 4: Loading Exit Models for Walk-Forward Validation")
print("-" * 80)

long_exit_model = joblib.load(MODELS_DIR / "xgboost_long_exit_threshold_075_20251027_190512.pkl")
long_exit_scaler = joblib.load(MODELS_DIR / "xgboost_long_exit_threshold_075_20251027_190512_scaler.pkl")
with open(MODELS_DIR / "xgboost_long_exit_threshold_075_20251027_190512_features.txt", 'r') as f:
    long_exit_features = [line.strip() for line in f if line.strip()]

short_exit_model = joblib.load(MODELS_DIR / "xgboost_short_exit_threshold_075_20251027_190512.pkl")
short_exit_scaler = joblib.load(MODELS_DIR / "xgboost_short_exit_threshold_075_20251027_190512_scaler.pkl")
with open(MODELS_DIR / "xgboost_short_exit_threshold_075_20251027_190512_features.txt", 'r') as f:
    short_exit_features = [line.strip() for line in f if line.strip()]

print(f"✅ Exit models loaded (LONG: {len(long_exit_features)} features, SHORT: {len(short_exit_features)} features)")
print()

# ============================================================================
# WALK-FORWARD TRAINING FUNCTIONS
# ============================================================================

def filter_candidates(df, side='LONG'):
    """Filter entry candidates using heuristics"""
    candidates = []

    for i in range(100, len(df)):
        row = df.iloc[i]

        # Basic filters
        if pd.isna(row['rsi']) or pd.isna(row['macd']):
            continue

        if side == 'LONG':
            # LONG filters
            conditions = (
                (row['rsi'] < 45) or  # Not overbought
                (row['macd'] > row['macd_signal']) or  # MACD bullish
                (row['volume'] > row['volume_sma'] * 1.2) or  # Volume surge
                (row['close'] > row['sma_20'])  # Above SMA
            )
        else:
            # SHORT filters
            conditions = (
                (row['rsi'] > 55) or  # Not oversold
                (row['macd'] < row['macd_signal']) or  # MACD bearish
                (row['volume'] > row['volume_sma'] * 1.2) or  # Volume surge
                (row['close'] < row['sma_20'])  # Below SMA
            )

        if conditions:
            candidates.append(i)

    return candidates

def simulate_trade_outcome(df, entry_idx, side, exit_model, exit_scaler, exit_features):
    """Simulate trade outcome using Exit model"""
    entry_price = df.iloc[entry_idx]['close']

    for hold_time in range(1, min(EMERGENCY_MAX_HOLD + 1, len(df) - entry_idx)):
        current_idx = entry_idx + hold_time
        current_row = df.iloc[current_idx]
        current_price = current_row['close']

        # Calculate P&L
        if side == 'LONG':
            price_change_pct = (current_price - entry_price) / entry_price
        else:
            price_change_pct = (entry_price - current_price) / entry_price

        leveraged_pnl_pct = price_change_pct * LEVERAGE

        # Check emergency exits
        if leveraged_pnl_pct <= EMERGENCY_STOP_LOSS:
            return 0, hold_time  # Stop loss

        if hold_time >= EMERGENCY_MAX_HOLD:
            return 1 if leveraged_pnl_pct > 0.02 else 0, hold_time

        # Check ML exit
        try:
            X_exit = current_row[exit_features].values.reshape(1, -1)
            X_exit_scaled = exit_scaler.transform(X_exit)
            exit_prob = exit_model.predict_proba(X_exit_scaled)[0][1]

            if exit_prob >= ML_EXIT_THRESHOLD:
                return 1 if leveraged_pnl_pct > 0.02 else 0, hold_time
        except:
            continue

    # Max hold reached
    return 1 if leveraged_pnl_pct > 0.02 else 0, EMERGENCY_MAX_HOLD

def walk_forward_training(df, side, entry_features, exit_model, exit_scaler, exit_features):
    """Walk-Forward Decoupled Training"""
    print(f"\n🔄 Walk-Forward Simulation ({side})...")

    # Filter candidates
    print(f"\n🔍 Filtering {side} Entry Candidates...")
    all_candidates = filter_candidates(df, side)
    print(f"  Total possible: {len(df) - 100:,}")
    print(f"  After filtering: {len(all_candidates):,} ({(1 - len(all_candidates)/(len(df)-100))*100:.1f}% reduction)")

    # TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=5)
    best_fold_data = None
    best_positive_rate = 0

    for fold_num, (train_idx, val_idx) in enumerate(tscv.split(df), 1):
        print(f"\n  Fold {fold_num}/5:")
        print(f"    Train: {len(train_idx):,} candles")
        print(f"    Val: {len(val_idx):,} candles")

        # Get validation candidates
        val_candidates = [c for c in all_candidates if c in val_idx]

        if len(val_candidates) == 0:
            continue

        # Simulate outcomes for validation candidates
        labels = []
        valid_candidates = []

        for cand_idx in val_candidates:
            if cand_idx + EMERGENCY_MAX_HOLD >= len(df):
                continue

            label, hold_time = simulate_trade_outcome(
                df, cand_idx, side, exit_model, exit_scaler, exit_features
            )

            labels.append(label)
            valid_candidates.append(cand_idx)

        if len(valid_candidates) == 0:
            continue

        positive_rate = sum(labels) / len(labels) * 100
        print(f"    Candidates: {len(valid_candidates)}")
        print(f"    ✅ Positive: {sum(labels)} ({positive_rate:.2f}%)")

        if positive_rate > best_positive_rate:
            best_positive_rate = positive_rate
            best_fold_data = {
                'candidates': valid_candidates,
                'labels': labels,
                'fold': fold_num
            }

    return best_fold_data, best_positive_rate

# ============================================================================
# LONG ENTRY MODEL
# ============================================================================
print("\n" + "=" * 80)
print("LONG ENTRY - WALK-FORWARD TRAINING (PHASE 1: 80 FEATURES)")
print("=" * 80)

long_fold_data, long_positive_rate = walk_forward_training(
    df_training, 'LONG', long_entry_features,
    long_exit_model, long_exit_scaler, long_exit_features
)

print(f"\n✅ Best LONG Fold: {long_fold_data['fold']} ({long_positive_rate:.2f}% positive)")

# Train final LONG model
X_long = df_training.iloc[long_fold_data['candidates']][long_entry_features].values
y_long = np.array(long_fold_data['labels'])

long_scaler = StandardScaler()
X_long_scaled = long_scaler.fit_transform(X_long)

long_model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='logloss'
)
long_model.fit(X_long_scaled, y_long)

# ============================================================================
# SHORT ENTRY MODEL
# ============================================================================
print("\n" + "=" * 80)
print("SHORT ENTRY - WALK-FORWARD TRAINING (PHASE 1: 79 FEATURES)")
print("=" * 80)

short_fold_data, short_positive_rate = walk_forward_training(
    df_training, 'SHORT', short_entry_features,
    short_exit_model, short_exit_scaler, short_exit_features
)

print(f"\n✅ Best SHORT Fold: {short_fold_data['fold']} ({short_positive_rate:.2f}% positive)")

# Train final SHORT model
X_short = df_training.iloc[short_fold_data['candidates']][short_entry_features].values
y_short = np.array(short_fold_data['labels'])

short_scaler = StandardScaler()
X_short_scaled = short_scaler.fit_transform(X_short)

short_model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    eval_metric='logloss'
)
short_model.fit(X_short_scaled, y_short)

# ============================================================================
# SAVE MODELS
# ============================================================================
print("\n" + "=" * 80)
print("SAVING RETRAINED MODELS")
print("=" * 80)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# LONG Entry
joblib.dump(long_model, MODELS_DIR / f"xgboost_long_entry_retrained_latest_{timestamp}.pkl")
joblib.dump(long_scaler, MODELS_DIR / f"xgboost_long_entry_retrained_latest_{timestamp}_scaler.pkl")
with open(MODELS_DIR / f"xgboost_long_entry_retrained_latest_{timestamp}_features.txt", 'w') as f:
    f.write('\n'.join(long_entry_features))

# SHORT Entry
joblib.dump(short_model, MODELS_DIR / f"xgboost_short_entry_retrained_latest_{timestamp}.pkl")
joblib.dump(short_scaler, MODELS_DIR / f"xgboost_short_entry_retrained_latest_{timestamp}_scaler.pkl")
with open(MODELS_DIR / f"xgboost_short_entry_retrained_latest_{timestamp}_features.txt", 'w') as f:
    f.write('\n'.join(short_entry_features))

print(f"✅ LONG Entry saved (timestamp: {timestamp})")
print(f"✅ SHORT Entry saved (timestamp: {timestamp})")
print()

# ============================================================================
# SUMMARY
# ============================================================================
print("=" * 80)
print("RETRAINING SUMMARY")
print("=" * 80)
print()
print("Training Data:")
print(f"  Period: {df_training['timestamp'].iloc[0]} to {df_training['timestamp'].iloc[-1]}")
print(f"  Candles: {len(df_training):,} ({len(df_training)/(24*12):.0f} days)")
print()
print("Holdout Data:")
print(f"  Period: {df_holdout['timestamp'].iloc[0]} to {df_holdout['timestamp'].iloc[-1]}")
print(f"  Candles: {len(df_holdout):,} ({HOLDOUT_DAYS} days)")
print()
print("Features:")
print(f"  LONG Entry: {len(long_entry_features)} features (Phase 1 reduction)")
print(f"  SHORT Entry: {len(short_entry_features)} features (Phase 1 reduction)")
print()
print("Training Results:")
print(f"  LONG Entry: Best Fold {long_fold_data['fold']}, {long_positive_rate:.2f}% positive")
print(f"  SHORT Entry: Best Fold {short_fold_data['fold']}, {short_positive_rate:.2f}% positive")
print()
print(f"Models saved with timestamp: {timestamp}")
print()
print("Next Steps:")
print("  1. Run backtest on 14-day holdout with retrained models")
print("  2. Verify trade frequency > 2.0/day")
print("  3. Compare vs original Phase 1 models")
print("  4. If successful, deploy to production")
print()
print("=" * 80)
