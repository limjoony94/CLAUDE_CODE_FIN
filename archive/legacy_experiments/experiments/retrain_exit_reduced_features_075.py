"""
Retrain Exit Models with REDUCED FEATURES
==========================================

Removes zero-importance features identified from feature importance analysis.

Zero-Importance Features Removed:
  LONG Exit (5): rsi_divergence, macd_histogram_slope, near_resistance, near_support, bb_position
  SHORT Exit (6): rsi_divergence, rsi_overbought, macd_histogram_slope, near_resistance, near_support, bb_position

Expected Impact:
  - Reduced overfitting (removed noise features)
  - Better generalization to new data
  - Faster predictions (27 → 22 LONG, 27 → 21 SHORT)

Created: 2025-10-28
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
from sklearn.metrics import accuracy_score, roc_auc_score

# Configuration
THRESHOLD = 0.75
DATA_DIR = PROJECT_ROOT / "data" / "features"
MODELS_DIR = PROJECT_ROOT / "models"

# Zero-importance features to remove
ZERO_IMPORTANCE_FEATURES = {
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

print("=" * 80)
print("RETRAIN EXIT MODELS (REDUCED FEATURES)")
print("=" * 80)
print()
print(f"Configuration:")
print(f"  Threshold: {THRESHOLD}")
print()
print(f"Features to Remove:")
print(f"  LONG Exit: {len(ZERO_IMPORTANCE_FEATURES['LONG_Exit'])} features")
print(f"  SHORT Exit: {len(ZERO_IMPORTANCE_FEATURES['SHORT_Exit'])} features")
print()

# Load Full Features Dataset
print("-" * 80)
print("STEP 1: Loading Full Features Dataset")
print("-" * 80)

df = pd.read_csv(DATA_DIR / "BTCUSDT_5m_features.csv")
print(f"✅ Loaded {len(df):,} candles with {len(df.columns)} features")
print(f"   Date range: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")
print()

# Prepare Exit Features
print("-" * 80)
print("STEP 1.5: Preparing Exit Features")
print("-" * 80)

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

# Load Exit Labels
print("-" * 80)
print("STEP 2: Loading Exit Labels")
print("-" * 80)

exit_labels_df = pd.read_csv(PROJECT_ROOT / "results" / "exit_model_labels_oppgating_improved_20251017_151624.csv")
exit_labels_df['timestamp'] = pd.to_datetime(exit_labels_df['timestamp'])
print(f"✅ Exit labels loaded: {len(exit_labels_df):,} rows")

# Merge with features
df_exit = df.merge(exit_labels_df[['timestamp', 'long_should_exit', 'short_should_exit']],
                   on='timestamp', how='inner')
df_exit = df_exit.dropna()
print(f"✅ Merged data: {len(df_exit):,} rows")
print()

# Load Original Feature Lists and Remove Zero-Importance Features
print("-" * 80)
print("STEP 3: Loading and Filtering Feature Lists")
print("-" * 80)

long_exit_features_path = MODELS_DIR / "xgboost_long_exit_oppgating_improved_20251024_043527_features.txt"
with open(long_exit_features_path, 'r') as f:
    long_exit_features_full = [line.strip() for line in f if line.strip()]

short_exit_features_path = MODELS_DIR / "xgboost_short_exit_oppgating_improved_20251024_044510_features.txt"
with open(short_exit_features_path, 'r') as f:
    short_exit_features_full = [line.strip() for line in f if line.strip()]

# Remove zero-importance features
long_exit_features = [f for f in long_exit_features_full
                      if f not in ZERO_IMPORTANCE_FEATURES['LONG_Exit']]
short_exit_features = [f for f in short_exit_features_full
                       if f not in ZERO_IMPORTANCE_FEATURES['SHORT_Exit']]

print(f"  LONG Exit: {len(long_exit_features_full)} → {len(long_exit_features)} features (-{len(long_exit_features_full) - len(long_exit_features)})")
print(f"  SHORT Exit: {len(short_exit_features_full)} → {len(short_exit_features)} features (-{len(short_exit_features_full) - len(short_exit_features)})")
print()

# Train LONG Exit Model
print("=" * 80)
print("TRAINING LONG EXIT MODEL (REDUCED FEATURES)")
print("=" * 80)

X_long_exit = df_exit[long_exit_features].values
y_long_exit = df_exit['long_should_exit'].values

scaler_long_exit = StandardScaler()
X_long_exit_scaled = scaler_long_exit.fit_transform(X_long_exit)

print(f"\nTraining data:")
print(f"  Samples: {len(X_long_exit):,}")
print(f"  Features: {len(long_exit_features)} (reduced from 27)")
print(f"  Positive rate: {y_long_exit.mean()*100:.2f}%")

# Cross-validation
tscv = TimeSeriesSplit(n_splits=5)
long_scores = []

print(f"\nCross-Validation:")
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

    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    y_pred = model.predict(X_val)
    y_pred_proba = model.predict_proba(X_val)[:, 1]

    acc = accuracy_score(y_val, y_pred)
    auc = roc_auc_score(y_val, y_pred_proba)

    long_scores.append({'fold': fold, 'accuracy': acc, 'auc': auc})
    print(f"  Fold {fold}: Accuracy={acc:.4f}, AUC={auc:.4f}")

# Train final model
print(f"\nTraining final LONG Exit model...")
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

long_avg_acc = np.mean([s['accuracy'] for s in long_scores])
long_avg_auc = np.mean([s['auc'] for s in long_scores])
print(f"✅ Average: Accuracy={long_avg_acc:.4f}, AUC={long_avg_auc:.4f}")

# Train SHORT Exit Model
print("\n" + "=" * 80)
print("TRAINING SHORT EXIT MODEL (REDUCED FEATURES)")
print("=" * 80)

X_short_exit = df_exit[short_exit_features].values
y_short_exit = df_exit['short_should_exit'].values

scaler_short_exit = StandardScaler()
X_short_exit_scaled = scaler_short_exit.fit_transform(X_short_exit)

print(f"\nTraining data:")
print(f"  Samples: {len(X_short_exit):,}")
print(f"  Features: {len(short_exit_features)} (reduced from 27)")
print(f"  Positive rate: {y_short_exit.mean()*100:.2f}%")

# Cross-validation
short_scores = []

print(f"\nCross-Validation:")
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

    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    y_pred = model.predict(X_val)
    y_pred_proba = model.predict_proba(X_val)[:, 1]

    acc = accuracy_score(y_val, y_pred)
    auc = roc_auc_score(y_val, y_pred_proba)

    short_scores.append({'fold': fold, 'accuracy': acc, 'auc': auc})
    print(f"  Fold {fold}: Accuracy={acc:.4f}, AUC={auc:.4f}")

# Train final model
print(f"\nTraining final SHORT Exit model...")
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

short_avg_acc = np.mean([s['accuracy'] for s in short_scores])
short_avg_auc = np.mean([s['auc'] for s in short_scores])
print(f"✅ Average: Accuracy={short_avg_acc:.4f}, AUC={short_avg_auc:.4f}")

# Save Models
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

print("\n" + "=" * 80)
print("SAVING MODELS")
print("=" * 80)

# LONG Exit
joblib.dump(long_exit_model, MODELS_DIR / f"xgboost_long_exit_reduced_{timestamp}.pkl")
joblib.dump(scaler_long_exit, MODELS_DIR / f"xgboost_long_exit_reduced_{timestamp}_scaler.pkl")
with open(MODELS_DIR / f"xgboost_long_exit_reduced_{timestamp}_features.txt", 'w') as f:
    f.write('\n'.join(long_exit_features))
print(f"✅ LONG Exit saved (timestamp: {timestamp})")

# SHORT Exit
joblib.dump(short_exit_model, MODELS_DIR / f"xgboost_short_exit_reduced_{timestamp}.pkl")
joblib.dump(scaler_short_exit, MODELS_DIR / f"xgboost_short_exit_reduced_{timestamp}_scaler.pkl")
with open(MODELS_DIR / f"xgboost_short_exit_reduced_{timestamp}_features.txt", 'w') as f:
    f.write('\n'.join(short_exit_features))
print(f"✅ SHORT Exit saved (timestamp: {timestamp})")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print()
print("Feature Reduction:")
print(f"  LONG Exit: 27 → {len(long_exit_features)} features (-{27 - len(long_exit_features)})")
print(f"  SHORT Exit: 27 → {len(short_exit_features)} features (-{27 - len(short_exit_features)})")
print()
print("Cross-Validation Performance:")
print(f"  LONG Exit: Accuracy={long_avg_acc:.4f}, AUC={long_avg_auc:.4f}")
print(f"  SHORT Exit: Accuracy={short_avg_acc:.4f}, AUC={short_avg_auc:.4f}")
print()
print(f"Models saved with timestamp: {timestamp}")
print()
print("Next Steps:")
print("  1. Run backtest with reduced-feature models")
print("  2. Compare: reduced vs full features")
print("     Entry: 73/74 vs 85/79")
print("     Exit: 22/21 vs 27/27")
print("  3. Expected: Similar/better performance with reduced overfitting")
print()
print("=" * 80)
