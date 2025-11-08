"""
Simple XGBoost Model for Paper Trading Bot

목적: Paper Trading Bot이 사용할 수 있는 간단한 binary classifier 생성
- Binary: 0 (not enter), 1 (enter long)
- Threshold 최적화 (0.002)
- pickle 형식 저장
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
from sklearn.model_selection import train_test_split
import xgboost as xgb
import ta

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

MODELS_DIR.mkdir(parents=True, exist_ok=True)

def calculate_features(df):
    """Calculate technical indicators"""
    df = df.copy()

    # Price changes
    for lag in range(1, 6):
        df[f'close_change_{lag}'] = df['close'].pct_change(lag)

    # Moving averages
    df['sma_10'] = ta.trend.sma_indicator(df['close'], window=10)
    df['sma_20'] = ta.trend.sma_indicator(df['close'], window=20)
    df['ema_10'] = ta.trend.ema_indicator(df['close'], window=10)

    # MACD
    macd = ta.trend.MACD(df['close'])
    df['macd'] = macd.macd()
    df['macd_signal'] = macd.macd_signal()
    df['macd_diff'] = macd.macd_diff()

    # RSI
    df['rsi'] = ta.momentum.rsi(df['close'], window=14)

    # Bollinger Bands
    bb = ta.volatility.BollingerBands(df['close'])
    df['bb_high'] = bb.bollinger_hband()
    df['bb_low'] = bb.bollinger_lband()
    df['bb_mid'] = bb.bollinger_mavg()

    # Volatility
    df['volatility'] = df['close'].pct_change().rolling(window=20).std()

    # Volume
    df['volume_sma'] = df['volume'].rolling(window=20).mean()
    df['volume_ratio'] = df['volume'] / df['volume_sma']

    return df

def create_target(df, lookahead=12, threshold=0.01):
    """
    Create binary target: 1 if price increases > threshold, 0 otherwise
    lookahead: 12 candles (1 hour for 5m data)
    """
    future_return = df['close'].pct_change(lookahead).shift(-lookahead)
    target = (future_return > threshold).astype(int)
    return target

print("=" * 80)
print("Simple XGBoost for Paper Trading Bot")
print("=" * 80)

# Load data
data_file = DATA_DIR / "historical" / "BTCUSDT_5m_max.csv"
df = pd.read_csv(data_file)
print(f"Loaded {len(df)} candles")

# Calculate features
df = calculate_features(df)
print(f"Calculated features: {len(df)} rows")

# Create target (1 hour lookahead, 1% threshold)
df['target'] = create_target(df, lookahead=12, threshold=0.01)

# Feature columns
feature_columns = [
    'close_change_1', 'close_change_2', 'close_change_3', 'close_change_4', 'close_change_5',
    'sma_10', 'sma_20', 'ema_10',
    'macd', 'macd_signal', 'macd_diff',
    'rsi',
    'bb_high', 'bb_low', 'bb_mid',
    'volatility',
    'volume_sma', 'volume_ratio'
]

# Drop NaN
df = df.dropna()
print(f"After dropna: {len(df)} rows")

# Prepare data
X = df[feature_columns].values
y = df['target'].values

print(f"\nTarget distribution:")
print(f"  Class 0 (not enter): {(y == 0).sum()} ({(y == 0).sum() / len(y) * 100:.1f}%)")
print(f"  Class 1 (enter): {(y == 1).sum()} ({(y == 1).sum() / len(y) * 100:.1f}%)")

# Train/val/test split
X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.2, shuffle=False)

print(f"\nData split:")
print(f"  Train: {len(X_train)}")
print(f"  Val: {len(X_val)}")
print(f"  Test: {len(X_test)}")

# Train XGBoost
print("\nTraining XGBoost...")
model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=5,
    learning_rate=0.1,
    random_state=42,
    eval_metric='logloss'
)

model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=False
)

# Evaluate
train_acc = model.score(X_train, y_train)
val_acc = model.score(X_val, y_val)
test_acc = model.score(X_test, y_test)

print(f"\nAccuracy:")
print(f"  Train: {train_acc:.4f}")
print(f"  Val: {val_acc:.4f}")
print(f"  Test: {test_acc:.4f}")

# Test predictions
y_pred_proba = model.predict_proba(X_test)[:, 1]
y_pred = model.predict(X_test)

# Metrics for test set
from sklearn.metrics import classification_report
print(f"\nTest Set Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Not Enter', 'Enter']))

# Save model
model_file = MODELS_DIR / "xgboost_model.pkl"
with open(model_file, 'wb') as f:
    pickle.dump(model, f)

print(f"\n✅ Model saved: {model_file}")

# Save feature columns for reference
feature_file = MODELS_DIR / "feature_columns.txt"
with open(feature_file, 'w') as f:
    f.write('\n'.join(feature_columns))

print(f"✅ Features saved: {feature_file}")

print("\n" + "=" * 80)
print("Simple XGBoost Model Ready for Paper Trading!")
print("=" * 80)
