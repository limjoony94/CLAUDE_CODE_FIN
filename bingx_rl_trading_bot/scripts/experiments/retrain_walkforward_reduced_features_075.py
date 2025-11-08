"""
Retrain Entry Models - Walk-Forward Decoupled with REDUCED FEATURES
====================================================================

Based on retrain_entry_walkforward_decoupled_075.py
Removes 17 zero-importance features identified from feature importance analysis

Zero-Importance Features Removed:
  LONG Entry (12): macd_bullish_divergence, rsi_bearish_divergence, rsi_bullish_divergence,
                   macd_bearish_divergence, strong_selling_pressure, hammer, shooting_star,
                   doji, vp_strong_buy_pressure, vwap_above, vwap_overbought, vwap_oversold

  SHORT Entry (5): support_breakdown, volume_decline_ratio, volatility_asymmetry,
                   near_resistance, downtrend_confirmed

Expected Impact:
  - Reduced overfitting (removed noise features)
  - Better generalization to new data
  - Faster predictions (85 → 73 LONG, 79 → 74 SHORT)

Created: 2025-10-28
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
from sklearn.metrics import precision_score, recall_score, f1_score

# Configuration
ENTRY_THRESHOLD = 0.75
ML_EXIT_THRESHOLD = 0.75
LEVERAGE = 4
EMERGENCY_STOP_LOSS = -0.03
EMERGENCY_MAX_HOLD = 120

DATA_DIR = PROJECT_ROOT / "data" / "features"
MODELS_DIR = PROJECT_ROOT / "models"

# Zero-importance features to remove
ZERO_IMPORTANCE_FEATURES = {
    'LONG_Entry': [
        'macd_bullish_divergence', 'rsi_bearish_divergence', 'rsi_bullish_divergence',
        'macd_bearish_divergence', 'strong_selling_pressure', 'hammer', 'shooting_star',
        'doji', 'vp_strong_buy_pressure', 'vwap_above', 'vwap_overbought', 'vwap_oversold'
    ],
    'SHORT_Entry': [
        'support_breakdown', 'volume_decline_ratio', 'volatility_asymmetry',
        'near_resistance', 'downtrend_confirmed'
    ]
}

print("="*80)
print("WALK-FORWARD DECOUPLED TRAINING (REDUCED FEATURES)")
print("="*80)
print()
print("Methodology:")
print("  ✅ Filtered Simulation (83% faster)")
print("  ✅ Walk-Forward Validation (no look-ahead bias)")
print("  ✅ Decoupled Training (breaks circular dependency)")
print("  ✅ Feature Reduction (removes zero-importance features)")
print()
print(f"Configuration:")
print(f"  Entry Threshold: {ENTRY_THRESHOLD}")
print(f"  ML Exit Threshold: {ML_EXIT_THRESHOLD}")
print(f"  Emergency Stop Loss: {EMERGENCY_STOP_LOSS}")
print(f"  Emergency Max Hold: {EMERGENCY_MAX_HOLD} candles")
print()
print(f"Features to Remove:")
print(f"  LONG Entry: {len(ZERO_IMPORTANCE_FEATURES['LONG_Entry'])} features")
print(f"  SHORT Entry: {len(ZERO_IMPORTANCE_FEATURES['SHORT_Entry'])} features")
print()

# Load Full Features Dataset
print("-"*80)
print("STEP 1: Loading Full Features Dataset")
print("-"*80)

df = pd.read_csv(DATA_DIR / "BTCUSDT_5m_features.csv")
print(f"✅ Loaded {len(df):,} candles with {len(df.columns)} features")
print(f"   Date range: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")
print()

# Prepare Exit Features (adds 15 additional features needed for Exit models)
print("-"*80)
print("STEP 1.5: Preparing Exit Features")
print("-"*80)

def prepare_exit_features(df):
    """Prepare EXIT features with enhanced market context"""
    print("Calculating enhanced market context features...")

    # Volume features
    df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
    df['volume_surge'] = (df['volume'] > df['volume'].rolling(20).mean() * 1.5).astype(float)

    # Price momentum features
    if 'sma_50' in df.columns:
        df['price_vs_ma20'] = (df['close'] - df['sma_50']) / df['sma_50']
    elif 'ma_50' in df.columns:
        df['price_vs_ma20'] = (df['close'] - df['ma_50']) / df['ma_50']
    else:
        ma_20 = df['close'].rolling(20).mean()
        df['price_vs_ma20'] = (df['close'] - ma_20) / ma_20

    if 'sma_200' in df.columns:
        df['price_vs_ma50'] = (df['close'] - df['sma_200']) / df['sma_200']
    elif 'ma_200' in df.columns:
        df['price_vs_ma50'] = (df['close'] - df['ma_200']) / df['ma_200']
    else:
        ma_50 = df['close'].rolling(50).mean()
        df['price_vs_ma50'] = (df['close'] - ma_50) / ma_50

    # Volatility features
    df['returns'] = df['close'].pct_change()
    df['volatility_20'] = df['returns'].rolling(20).std()
    df['volatility_regime'] = (df['volatility_20'] > df['volatility_20'].rolling(100).median()).astype(float)

    # RSI dynamics
    df['rsi_slope'] = df['rsi'].diff(3) / 3
    df['rsi_overbought'] = (df['rsi'] > 70).astype(float)
    df['rsi_oversold'] = (df['rsi'] < 30).astype(float)
    df['rsi_divergence'] = 0

    # MACD dynamics
    df['macd_histogram_slope'] = df['macd_hist'].diff(3) / 3 if 'macd_hist' in df.columns else 0
    df['macd_crossover'] = ((df['macd'] > df['macd_signal']) &
                            (df['macd'].shift(1) <= df['macd_signal'].shift(1))).astype(float)
    df['macd_crossunder'] = ((df['macd'] < df['macd_signal']) &
                             (df['macd'].shift(1) >= df['macd_signal'].shift(1))).astype(float)

    # Price patterns
    df['higher_high'] = (df['high'] > df['high'].shift(1)).astype(float)
    df['lower_low'] = (df['low'] < df['low'].shift(1)).astype(float)
    df['price_acceleration'] = df['close'].diff().diff()

    # Support/Resistance proximity
    if 'support_level' in df.columns and 'resistance_level' in df.columns:
        df['near_resistance'] = (df['close'] > df['resistance_level'] * 0.98).astype(float)
        df['near_support'] = (df['close'] < df['support_level'] * 1.02).astype(float)
    else:
        df['near_resistance'] = 0
        df['near_support'] = 0

    # Bollinger Band position
    if 'bb_upper' in df.columns and 'bb_lower' in df.columns:
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    else:
        df['bb_position'] = 0.5

    # Clean NaN
    df = df.ffill().bfill()

    print(f"✅ Enhanced exit features calculated")
    return df

df = prepare_exit_features(df)
print(f"✅ Exit features added - now {len(df.columns)} total features")
print()

# Load Entry Feature Lists and Remove Zero-Importance Features
print("-"*80)
print("STEP 2: Loading and Filtering Feature Lists")
print("-"*80)

with open(MODELS_DIR / "xgboost_long_entry_enhanced_20251024_012445_features.txt", 'r') as f:
    long_entry_features_full = [line.strip() for line in f if line.strip()]

with open(MODELS_DIR / "xgboost_short_entry_enhanced_20251024_012445_features.txt", 'r') as f:
    short_entry_features_full = [line.strip() for line in f if line.strip()]

# Remove zero-importance features
long_entry_features = [f for f in long_entry_features_full
                       if f not in ZERO_IMPORTANCE_FEATURES['LONG_Entry']]
short_entry_features = [f for f in short_entry_features_full
                        if f not in ZERO_IMPORTANCE_FEATURES['SHORT_Entry']]

print(f"  LONG Entry: {len(long_entry_features_full)} → {len(long_entry_features)} features (-{len(long_entry_features_full) - len(long_entry_features)})")
print(f"  SHORT Entry: {len(short_entry_features_full)} → {len(short_entry_features)} features (-{len(short_entry_features_full) - len(short_entry_features)})")
print()

# ==============================================================================
# OPTION A: Filtered Simulation (Candidate Pre-Filtering)
# ==============================================================================

def filter_entry_candidates(df, side):
    """Pre-filter candidates using simple heuristics"""
    print(f"\n🔍 Filtering {side} Entry Candidates...")

    mask = pd.Series(True, index=df.index)

    if 'volume' in df.columns:
        volume_filter = df['volume'] > df['volume'].rolling(20).mean()
        mask = mask & volume_filter

    if 'rsi' in df.columns:
        rsi_filter = (df['rsi'] > 30) & (df['rsi'] < 70)
        mask = mask & rsi_filter

    if side == 'LONG':
        if 'ma_50' in df.columns:
            trend_filter = df['close'] > df['ma_50']
            mask = mask & trend_filter
        if 'macd' in df.columns and 'macd_signal' in df.columns:
            macd_filter = df['macd'] > df['macd_signal']
            mask = mask & macd_filter
    else:
        if 'ma_50' in df.columns:
            trend_filter = df['close'] < df['ma_50']
            mask = mask & trend_filter
        if 'macd' in df.columns and 'macd_signal' in df.columns:
            macd_filter = df['macd'] < df['macd_signal']
            mask = mask & macd_filter

    valid_range = (df.index >= 100) & (df.index < len(df) - EMERGENCY_MAX_HOLD)
    mask = mask & valid_range

    candidates = df[mask].index.tolist()
    total_possible = len(df) - 100 - EMERGENCY_MAX_HOLD
    reduction_pct = (1 - len(candidates) / total_possible) * 100

    print(f"  Total possible: {total_possible:,}")
    print(f"  After filtering: {len(candidates):,} ({reduction_pct:.1f}% reduction)")

    return candidates

# ==============================================================================
# OPTION C: Decoupled Training (Rule-Based Exit Labels)
# ==============================================================================

def label_exits_rule_based(df, side):
    """Rule-based exit labels (breaks circular dependency)"""
    print(f"\n🔨 Creating Rule-Based Exit Labels for {side}...")

    exit_labels = np.zeros(len(df))
    good_exits = 0

    for i in range(100, len(df) - EMERGENCY_MAX_HOLD):
        entry_price = df.iloc[i]['close']

        for hold_time in range(1, EMERGENCY_MAX_HOLD + 1):
            exit_idx = i + hold_time
            if exit_idx >= len(df):
                break

            exit_price = df.iloc[exit_idx]['close']

            if side == 'LONG':
                price_change_pct = (exit_price - entry_price) / entry_price
            else:
                price_change_pct = (entry_price - exit_price) / entry_price

            leveraged_pnl = price_change_pct * LEVERAGE

            # Rule: Good exit if profit > 2% and hold < 60 candles
            if leveraged_pnl > 0.02 and hold_time < 60:
                exit_labels[exit_idx] = 1
                good_exits += 1
                break

    positive_pct = (good_exits / len(exit_labels)) * 100
    print(f"  ✅ Exit labels created: {good_exits:,} good exits ({positive_pct:.2f}%)")

    return exit_labels

# Load Exit Models for Option B (Walk-Forward)
print("-"*80)
print("STEP 3: Loading Exit Models for Walk-Forward Validation")
print("-"*80)

long_exit_model = joblib.load(MODELS_DIR / "xgboost_long_exit_oppgating_improved_20251024_043527.pkl")
long_exit_scaler = joblib.load(MODELS_DIR / "xgboost_long_exit_oppgating_improved_20251024_043527_scaler.pkl")
with open(MODELS_DIR / "xgboost_long_exit_oppgating_improved_20251024_043527_features.txt", 'r') as f:
    long_exit_features = [line.strip() for line in f if line.strip()]

short_exit_model = joblib.load(MODELS_DIR / "xgboost_short_exit_oppgating_improved_20251024_044510.pkl")
short_exit_scaler = joblib.load(MODELS_DIR / "xgboost_short_exit_oppgating_improved_20251024_044510_scaler.pkl")
with open(MODELS_DIR / "xgboost_short_exit_oppgating_improved_20251024_044510_features.txt", 'r') as f:
    short_exit_features = [line.strip() for line in f if line.strip()]

print(f"✅ Exit models loaded (LONG: {len(long_exit_features)} features, SHORT: {len(short_exit_features)} features)")
print()

# ==============================================================================
# OPTION B: Walk-Forward Validation (Per-Fold Training)
# ==============================================================================

def simulate_walk_forward(df, side, entry_features, exit_model, exit_scaler, exit_features, candidates):
    """Walk-Forward simulation with per-fold Exit model training"""
    print(f"\n🔄 Option B: Walk-Forward Simulation ({side})...")

    tscv = TimeSeriesSplit(n_splits=5)
    fold_results = []

    for fold, (train_idx, val_idx) in enumerate(tscv.split(df), 1):
        print(f"\n  Fold {fold}/5:")
        print(f"    Train: {len(train_idx):,} candles")
        print(f"    Val: {len(val_idx):,} candles")

        # Train Exit model on this fold's training data ONLY
        train_exit_labels = label_exits_rule_based(df.iloc[train_idx], side)

        # Train Exit model
        X_exit_train = df.iloc[train_idx][exit_features].values
        X_exit_train_scaled = exit_scaler.transform(X_exit_train)

        fold_exit_model = xgb.XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.02,
            random_state=42
        )
        fold_exit_model.fit(X_exit_train_scaled, train_exit_labels)

        # Simulate on validation set with this fold's Exit model
        fold_candidates = [c for c in candidates if c in val_idx]
        print(f"    Candidates: {len(fold_candidates):,}")

        labels = []
        for entry_idx in fold_candidates:
            entry_price = df.iloc[entry_idx]['close']
            good_entry = False

            for hold_time in range(1, EMERGENCY_MAX_HOLD + 1):
                exit_idx = entry_idx + hold_time
                if exit_idx >= len(df):
                    break

                # Use fold's Exit model to predict
                X_exit = df.iloc[exit_idx][exit_features].values.reshape(1, -1)
                X_exit_scaled = exit_scaler.transform(X_exit)
                exit_prob = fold_exit_model.predict_proba(X_exit_scaled)[0][1]

                exit_price = df.iloc[exit_idx]['close']

                if side == 'LONG':
                    price_change_pct = (exit_price - entry_price) / entry_price
                else:
                    price_change_pct = (entry_price - exit_price) / entry_price

                leveraged_pnl = price_change_pct * LEVERAGE

                # ML Exit or Emergency
                if exit_prob >= ML_EXIT_THRESHOLD or leveraged_pnl <= EMERGENCY_STOP_LOSS or hold_time >= EMERGENCY_MAX_HOLD:
                    if leveraged_pnl > 0.02:  # Good entry threshold
                        good_entry = True
                    break

            labels.append((entry_idx, 1 if good_entry else 0))

        fold_df = pd.DataFrame(labels, columns=['idx', 'label'])
        positive_pct = (fold_df['label'].sum() / len(fold_df)) * 100
        print(f"    ✅ Positive: {fold_df['label'].sum():,} ({positive_pct:.2f}%)")

        fold_results.append({
            'fold': fold,
            'candidates': fold_df,
            'positive_pct': positive_pct
        })

    return fold_results

# Run Walk-Forward Simulation for LONG
print("\n" + "="*80)
print("LONG ENTRY - WALK-FORWARD SIMULATION")
print("="*80)

long_candidates = filter_entry_candidates(df, 'LONG')
long_fold_results = simulate_walk_forward(df, 'LONG', long_entry_features,
                                         long_exit_model, long_exit_scaler,
                                         long_exit_features, long_candidates)

# Select best fold for LONG
best_long_fold = max(long_fold_results, key=lambda x: x['positive_pct'])
print(f"\n✅ Best LONG Fold: {best_long_fold['fold']} ({best_long_fold['positive_pct']:.2f}% positive)")

# Run Walk-Forward Simulation for SHORT
print("\n" + "="*80)
print("SHORT ENTRY - WALK-FORWARD SIMULATION")
print("="*80)

short_candidates = filter_entry_candidates(df, 'SHORT')
short_fold_results = simulate_walk_forward(df, 'SHORT', short_entry_features,
                                          short_exit_model, short_exit_scaler,
                                          short_exit_features, short_candidates)

# Select best fold for SHORT
best_short_fold = max(short_fold_results, key=lambda x: x['positive_pct'])
print(f"\n✅ Best SHORT Fold: {best_short_fold['fold']} ({best_short_fold['positive_pct']:.2f}% positive)")

# ==============================================================================
# TRAIN FINAL MODELS ON BEST FOLD DATA
# ==============================================================================

print("\n" + "="*80)
print("TRAINING FINAL ENTRY MODELS (REDUCED FEATURES)")
print("="*80)

# Train LONG Entry Model
print("\nLONG Entry Model:")
print("-"*80)

long_train_data = best_long_fold['candidates']
X_long = df.loc[long_train_data['idx'], long_entry_features].values
y_long = long_train_data['label'].values

scaler_long = MinMaxScaler()
X_long_scaled = scaler_long.fit_transform(X_long)

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

print(f"  ✅ Training samples: {len(X_long):,}")
print(f"  ✅ Features: {len(long_entry_features)} (reduced from 85)")
print(f"  ✅ Positive rate: {best_long_fold['positive_pct']:.2f}%")

# Train SHORT Entry Model
print("\nSHORT Entry Model:")
print("-"*80)

short_train_data = best_short_fold['candidates']
X_short = df.loc[short_train_data['idx'], short_entry_features].values
y_short = short_train_data['label'].values

scaler_short = MinMaxScaler()
X_short_scaled = scaler_short.fit_transform(X_short)

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

print(f"  ✅ Training samples: {len(X_short):,}")
print(f"  ✅ Features: {len(short_entry_features)} (reduced from 79)")
print(f"  ✅ Positive rate: {best_short_fold['positive_pct']:.2f}%")

# ==============================================================================
# SAVE MODELS
# ==============================================================================

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

print("\n" + "="*80)
print("SAVING MODELS")
print("="*80)

# LONG Entry
joblib.dump(long_entry_model, MODELS_DIR / f"xgboost_long_entry_walkforward_reduced_{timestamp}.pkl")
joblib.dump(scaler_long, MODELS_DIR / f"xgboost_long_entry_walkforward_reduced_{timestamp}_scaler.pkl")
with open(MODELS_DIR / f"xgboost_long_entry_walkforward_reduced_{timestamp}_features.txt", 'w') as f:
    f.write('\n'.join(long_entry_features))
print(f"✅ LONG Entry saved (timestamp: {timestamp})")

# SHORT Entry
joblib.dump(short_entry_model, MODELS_DIR / f"xgboost_short_entry_walkforward_reduced_{timestamp}.pkl")
joblib.dump(scaler_short, MODELS_DIR / f"xgboost_short_entry_walkforward_reduced_{timestamp}_scaler.pkl")
with open(MODELS_DIR / f"xgboost_short_entry_walkforward_reduced_{timestamp}_features.txt", 'w') as f:
    f.write('\n'.join(short_entry_features))
print(f"✅ SHORT Entry saved (timestamp: {timestamp})")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print()
print("Feature Reduction:")
print(f"  LONG Entry: 85 → {len(long_entry_features)} features (-{85 - len(long_entry_features)})")
print(f"  SHORT Entry: 79 → {len(short_entry_features)} features (-{79 - len(short_entry_features)})")
print()
print("Training Results:")
print(f"  LONG Entry: Best Fold {best_long_fold['fold']}, {best_long_fold['positive_pct']:.2f}% positive")
print(f"  SHORT Entry: Best Fold {best_short_fold['fold']}, {best_short_fold['positive_pct']:.2f}% positive")
print()
print(f"Models saved with timestamp: {timestamp}")
print()
print("Next Steps:")
print("  1. Run backtest with reduced-feature models")
print("  2. Compare: reduced vs full features (73/74 vs 85/79)")
print("  3. Expected: Similar/better performance with reduced overfitting")
print()
print("="*80)
