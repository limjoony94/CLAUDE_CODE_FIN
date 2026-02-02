"""
Retrain EXIT Models with 0.75/0.75 Threshold
==========================================

Trains EXIT models for 0.75 Entry + 0.75 ML Exit threshold configuration.
Uses Full 165-feature dataset.

Created: 2025-10-27
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

from scripts.experiments.retrain_exit_models_opportunity_gating import prepare_exit_features
from scripts.experiments.trade_simulator_optimized import simulate_trades_optimized

# Configuration
ENTRY_THRESHOLD = 0.75
ML_EXIT_THRESHOLD = 0.75
LEVERAGE = 4
EMERGENCY_STOP_LOSS = -0.03
EMERGENCY_MAX_HOLD = 120

DATA_DIR = PROJECT_ROOT / "data" / "features"
MODELS_DIR = PROJECT_ROOT / "models"

print("="*80)
print("RETRAIN EXIT MODELS: THRESHOLD 0.75 (LONG + SHORT)")
print("="*80)
print()
print(f"Configuration:")
print(f"  Entry Threshold: {ENTRY_THRESHOLD}")
print(f"  ML Exit Threshold: {ML_EXIT_THRESHOLD}")
print(f"  Emergency Stop Loss: {EMERGENCY_STOP_LOSS}")
print(f"  Emergency Max Hold: {EMERGENCY_MAX_HOLD} candles")
print()

# Step 1: Load Full Features Dataset
print("-"*80)
print("STEP 1: Loading Full Features Dataset")
print("-"*80)

df = pd.read_csv(DATA_DIR / "BTCUSDT_5m_features.csv")
print(f"✅ Loaded {len(df):,} candles with {len(df.columns)} features")
print(f"   Date range: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")
print()

# Prepare exit features (adds enhanced market context)
print("Preparing exit features...")
df = prepare_exit_features(df)
print(f"✅ Exit features ready ({len(df.columns)} total columns)")
print()

# Step 2: Load Latest Entry Models
print("-"*80)
print("STEP 2: Loading Latest Entry Models")
print("-"*80)

# LONG Entry
with open(MODELS_DIR / "xgboost_long_entry_enhanced_20251024_012445.pkl", 'rb') as f:
    long_entry_model = pickle.load(f)
long_entry_scaler = joblib.load(MODELS_DIR / "xgboost_long_entry_enhanced_20251024_012445_scaler.pkl")
with open(MODELS_DIR / "xgboost_long_entry_enhanced_20251024_012445_features.txt", 'r') as f:
    long_entry_features = [line.strip() for line in f]

# SHORT Entry
with open(MODELS_DIR / "xgboost_short_entry_enhanced_20251024_012445.pkl", 'rb') as f:
    short_entry_model = pickle.load(f)
short_entry_scaler = joblib.load(MODELS_DIR / "xgboost_short_entry_enhanced_20251024_012445_scaler.pkl")
with open(MODELS_DIR / "xgboost_short_entry_enhanced_20251024_012445_features.txt", 'r') as f:
    short_entry_features = [line.strip() for line in f]

print(f"  ✅ LONG Entry: {len(long_entry_features)} features")
print(f"  ✅ SHORT Entry: {len(short_entry_features)} features")
print()

# Step 3: Exit Labeling with Trade Simulation
print("-"*80)
print("STEP 3: Exit Labeling (Trade Simulation)")
print("-"*80)
print()

def label_exits_with_simulation(df, entry_model, entry_scaler, entry_features, side):
    """Label exit points using trade simulation"""

    print(f"🔄 {side} Exit Labeling...")

    # Get entry probabilities
    X_entry = df[entry_features].values
    X_entry_scaled = entry_scaler.transform(X_entry)
    entry_probs = entry_model.predict_proba(X_entry_scaled)[:, 1]

    # Find entries (threshold 0.75)
    entry_indices = np.where(entry_probs >= ENTRY_THRESHOLD)[0]
    entry_indices = [i for i in entry_indices if i < len(df) - EMERGENCY_MAX_HOLD]

    print(f"  Found {len(entry_indices):,} potential {side} entries (prob >= {ENTRY_THRESHOLD})")

    if len(entry_indices) == 0:
        print(f"  ⚠️  No entries found for {side}!")
        return None

    # Simulate trades (simple simulation without Exit model)
    trades = []
    for entry_idx in entry_indices:
        entry_price = df['close'].iloc[entry_idx]

        # Search for exit
        for hold_time in range(1, min(EMERGENCY_MAX_HOLD + 1, len(df) - entry_idx)):
            current_idx = entry_idx + hold_time
            current_price = df['close'].iloc[current_idx]

            # Calculate P&L
            if side == 'LONG':
                pnl_pct = (current_price - entry_price) / entry_price
            else:
                pnl_pct = (entry_price - current_price) / entry_price

            leveraged_pnl = pnl_pct * LEVERAGE

            # Exit conditions
            exit_reason = None

            # 1. Stop Loss
            if leveraged_pnl <= EMERGENCY_STOP_LOSS:
                exit_reason = 'stop_loss'

            # 2. Take Profit (simple: >+2% leveraged)
            elif leveraged_pnl >= 0.02:
                exit_reason = 'take_profit'

            # 3. Max Hold
            elif hold_time >= EMERGENCY_MAX_HOLD:
                exit_reason = 'max_hold'

            if exit_reason:
                trades.append({
                    'entry_idx': entry_idx,
                    'exit_idx': current_idx,
                    'hold_time': hold_time,
                    'pnl_pct': pnl_pct,
                    'leveraged_pnl': leveraged_pnl,
                    'exit_reason': exit_reason,
                    'good_exit': leveraged_pnl > 0
                })
                break

    print(f"  Simulated {len(trades):,} {side} trades")

    # Label exits
    df['exit_label'] = 0
    for trade in trades:
        if trade['good_exit']:
            df.loc[trade['exit_idx'], 'exit_label'] = 1

    positive = (df['exit_label'] == 1).sum()
    negative = (df['exit_label'] == 0).sum()

    print(f"  Exit Labels:")
    print(f"    Good (1): {positive:,} ({positive/len(df)*100:.2f}%)")
    print(f"    Bad (0): {negative:,} ({negative/len(df)*100:.2f}%)")
    print()

    return df

# Label LONG exits
df_long = label_exits_with_simulation(
    df.copy(), long_entry_model, long_entry_scaler, long_entry_features, 'LONG'
)

# Label SHORT exits
df_short = label_exits_with_simulation(
    df.copy(), short_entry_model, short_entry_scaler, short_entry_features, 'SHORT'
)

# Step 4: Train Exit Models
print("-"*80)
print("STEP 4: Training Exit Models")
print("-"*80)
print()

# Exit feature list (enhanced market context features)
EXIT_FEATURES = [
    # Price & Volume
    'close', 'volume', 'volume_surge', 'price_acceleration',
    # Moving Averages
    'ma_20', 'ma_50', 'price_vs_ma20', 'price_vs_ma50',
    # Volatility
    'volatility_20',
    # RSI
    'rsi', 'rsi_slope', 'rsi_overbought', 'rsi_oversold', 'rsi_divergence',
    # MACD
    'macd', 'macd_signal', 'macd_histogram', 'macd_histogram_slope',
    'macd_crossover', 'macd_crossunder',
    # Bollinger Bands
    'bb_upper', 'bb_lower', 'bb_position',
    # Position in Trade
    'higher_high', 'near_support'
]

def train_exit_model(df, side):
    """Train Exit model"""

    print(f"Training {side} Exit Model...")

    # Prepare data
    available_features = [f for f in EXIT_FEATURES if f in df.columns]
    print(f"  Using {len(available_features)} features")

    X = df[available_features].values
    y = df['exit_label'].values

    # Remove NaN
    mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
    X = X[mask]
    y = y[mask]

    print(f"  Training samples: {len(X):,}")
    print(f"  Positive ratio: {y.mean()*100:.2f}%")

    # Scale
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Train with 5-fold CV
    tscv = TimeSeriesSplit(n_splits=5)

    best_model = None
    best_score = 0

    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_scaled)):
        X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]

        # Train
        model = xgb.XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.01,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum(),
            random_state=42,
            eval_metric='logloss'
        )

        model.fit(X_train, y_train, verbose=False)

        # Evaluate
        score = f1_score(y_val, model.predict(X_val))

        if score > best_score:
            best_score = score
            best_model = model

        print(f"  Fold {fold+1}/5: F1={score:.4f}")

    print(f"  ✅ Best F1 Score: {best_score:.4f}")
    print()

    return best_model, scaler, available_features

# Train LONG Exit
long_exit_model, long_exit_scaler, long_exit_features = train_exit_model(df_long, 'LONG')

# Train SHORT Exit
short_exit_model, short_exit_scaler, short_exit_features = train_exit_model(df_short, 'SHORT')

# Step 5: Save Models
print("-"*80)
print("STEP 5: Saving Models")
print("-"*80)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

# LONG Exit
long_exit_path = MODELS_DIR / f"xgboost_long_exit_threshold_075_{timestamp}"
with open(f"{long_exit_path}.pkl", 'wb') as f:
    pickle.dump(long_exit_model, f)
joblib.dump(long_exit_scaler, f"{long_exit_path}_scaler.pkl")
with open(f"{long_exit_path}_features.txt", 'w') as f:
    f.write('\n'.join(long_exit_features))

print(f"✅ LONG Exit Model: {long_exit_path.name}.pkl")
print(f"   Features: {len(long_exit_features)}")

# SHORT Exit
short_exit_path = MODELS_DIR / f"xgboost_short_exit_threshold_075_{timestamp}"
with open(f"{short_exit_path}.pkl", 'wb') as f:
    pickle.dump(short_exit_model, f)
joblib.dump(short_exit_scaler, f"{short_exit_path}_scaler.pkl")
with open(f"{short_exit_path}_features.txt", 'w') as f:
    f.write('\n'.join(short_exit_features))

print(f"✅ SHORT Exit Model: {short_exit_path.name}.pkl")
print(f"   Features: {len(short_exit_features)}")
print()

print("="*80)
print("EXIT MODELS RETRAINING COMPLETE")
print("="*80)
print()
print("Next Steps:")
print("1. Retrain Entry models with these new Exit models")
print("2. Run 108-window backtest")
print("3. Deploy to production")
print()
