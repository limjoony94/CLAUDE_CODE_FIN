"""
Enhanced SHORT Entry 재학습 - RSI/MACD 실제 계산 포함

목표: RSI/MACD features를 실제로 계산하여 완전한 22 features 활용
예상: 48.3% → 55-60% 승률 개선
"""

import pandas as pd
import numpy as np
import pickle
import talib
from pathlib import Path
from datetime import datetime
from sklearn.preprocessing import MinMaxScaler
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import sys

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data" / "historical"
MODELS_DIR = PROJECT_ROOT / "models"

# Parameters
LOOKAHEAD = 3  # 15 minutes (3 * 5min)
PROFIT_THRESHOLD = 0.005  # 0.5%
LEAD_TIME_WINDOW = 12  # 1 hour

def calculate_rsi_macd_features(df):
    """
    RSI/MACD features 실제 계산 (기본값 0.0 대체)

    Calculates:
    - rsi, rsi_slope, rsi_overbought, rsi_oversold, rsi_divergence
    - macd, macd_signal, macd_histogram_slope
    - macd_crossover, macd_crossunder
    """
    print("Calculating RSI features...")

    # RSI (14-period)
    df['rsi'] = talib.RSI(df['close'], timeperiod=14)
    df['rsi_slope'] = df['rsi'].diff(3)
    df['rsi_overbought'] = (df['rsi'] > 70).astype(float)
    df['rsi_oversold'] = (df['rsi'] < 30).astype(float)

    # RSI divergence (simplified: price vs RSI direction mismatch)
    price_change = df['close'].diff(5)
    rsi_change = df['rsi'].diff(5)
    df['rsi_divergence'] = (
        ((price_change > 0) & (rsi_change < 0)) |  # Bearish divergence
        ((price_change < 0) & (rsi_change > 0))    # Bullish divergence
    ).astype(float)

    print("Calculating MACD features...")

    # MACD (12, 26, 9)
    macd, macd_signal, macd_hist = talib.MACD(
        df['close'],
        fastperiod=12,
        slowperiod=26,
        signalperiod=9
    )

    df['macd'] = macd
    df['macd_signal'] = macd_signal
    df['macd_histogram_slope'] = pd.Series(macd_hist).diff(3)

    # MACD crossovers
    df['macd_crossover'] = (
        (macd > macd_signal) &
        (macd.shift(1) <= macd_signal.shift(1))
    ).astype(float)

    df['macd_crossunder'] = (
        (macd < macd_signal) &
        (macd.shift(1) >= macd_signal.shift(1))
    ).astype(float)

    print(f"  ✅ RSI/MACD calculated")
    print(f"     RSI range: {df['rsi'].min():.2f} - {df['rsi'].max():.2f}")
    print(f"     RSI overbought: {df['rsi_overbought'].sum()} candles")
    print(f"     RSI oversold: {df['rsi_oversold'].sum()} candles")
    print(f"     MACD crossovers: {df['macd_crossover'].sum()}")
    print(f"     MACD crossunders: {df['macd_crossunder'].sum()}")

    return df

def calculate_enhanced_short_features(df):
    """Calculate 22 SELL signal features"""

    print("\nCalculating Enhanced SHORT features...")

    # Volume features
    df['volume_ratio'] = df['volume'] / df['volume'].rolling(20).mean()
    df['volume_surge'] = (df['volume'] > df['volume'].rolling(20).mean() * 1.5).astype(float)

    # Price momentum
    ma_20 = df['close'].rolling(20).mean()
    ma_50 = df['close'].rolling(50).mean()
    df['price_vs_ma20'] = (df['close'] - ma_20) / ma_20
    df['price_vs_ma50'] = (df['close'] - ma_50) / ma_50
    df['price_acceleration'] = df['close'].diff().diff()

    # Volatility
    df['volatility_20'] = df['close'].pct_change().rolling(20).std()
    volatility_median = df['volatility_20'].median()
    df['volatility_regime'] = (df['volatility_20'] > volatility_median).astype(float)

    # RSI/MACD - NOW CALCULATED ABOVE (not defaults!)
    # These will be filled by calculate_rsi_macd_features()

    # Price patterns
    df['higher_high'] = (
        (df['high'] > df['high'].shift(1)) &
        (df['high'].shift(1) > df['high'].shift(2))
    ).astype(float)

    df['lower_low'] = (
        (df['low'] < df['low'].shift(1)) &
        (df['low'].shift(1) < df['low'].shift(2))
    ).astype(float)

    # Support/Resistance
    resistance = df['high'].rolling(50).max()
    support = df['low'].rolling(50).min()
    df['near_resistance'] = (df['close'] > resistance * 0.98).astype(float)
    df['near_support'] = (df['close'] < support * 1.02).astype(float)

    # Bollinger Bands
    bb_mid = df['close'].rolling(20).mean()
    bb_std = df['close'].rolling(20).std()
    bb_high = bb_mid + 2 * bb_std
    bb_low = bb_mid - 2 * bb_std
    bb_range = bb_high - bb_low
    df['bb_position'] = np.where(
        bb_range != 0,
        (df['close'] - bb_low) / bb_range,
        0.5
    )

    print("  ✅ Enhanced SHORT features calculated")

    return df

def create_2of3_scoring_labels(df, lookahead=LOOKAHEAD, profit_threshold=PROFIT_THRESHOLD, lead_time_window=LEAD_TIME_WINDOW):
    """
    2of3 Scoring Labeling for SHORT Entry

    Criteria:
    1. Profit potential: Price drops >= profit_threshold
    2. Lead-time quality: Drop happens within lead_time_window
    3. Beats delayed entry: Better than entering 1 candle later

    Label = 1 if 2 or more criteria are met
    """
    print("\nCreating 2of3 scoring labels...")

    labels = []

    for i in range(len(df) - lookahead - lead_time_window):
        current_price = df['close'].iloc[i]

        # Look ahead for SHORT profit (price drop)
        future_prices = df['close'].iloc[i+lookahead:i+lookahead+lead_time_window]

        if len(future_prices) == 0:
            labels.append(0)
            continue

        # Find minimum price (best SHORT exit)
        min_price = future_prices.min()
        min_idx = future_prices.idxmin()

        # Calculate profit potential
        profit_pct = (current_price - min_price) / current_price

        # Criteria scoring
        score = 0

        # Criterion 1: Profit potential
        if profit_pct >= profit_threshold:
            score += 1

        # Criterion 2: Lead-time quality (drop happens early)
        drop_time = min_idx - (i + lookahead)
        if drop_time <= lead_time_window / 2:  # Drop within first half
            score += 1

        # Criterion 3: Beats delayed entry
        if i + 1 < len(df):
            delayed_price = df['close'].iloc[i+1]
            delayed_profit = (delayed_price - min_price) / delayed_price
            if profit_pct > delayed_profit:
                score += 1

        # Label = 1 if 2 or more criteria met
        labels.append(1 if score >= 2 else 0)

    # Pad with 0s
    labels.extend([0] * (lookahead + lead_time_window))

    positive_rate = sum(labels) / len(labels) * 100
    print(f"  ✅ Labels created: {positive_rate:.2f}% positive (target: 10-20%)")

    return np.array(labels)

# Feature order (22 SELL features)
FEATURE_COLUMNS = [
    'rsi',
    'macd',
    'macd_signal',
    'volatility_regime',
    'volume_surge',
    'price_acceleration',
    'volume_ratio',
    'price_vs_ma20',
    'price_vs_ma50',
    'volatility_20',
    'rsi_slope',
    'rsi_overbought',
    'rsi_oversold',
    'rsi_divergence',
    'macd_histogram_slope',
    'macd_crossover',
    'macd_crossunder',
    'higher_high',
    'lower_low',
    'near_resistance',
    'near_support',
    'bb_position'
]

print("="*80)
print("Enhanced SHORT Entry Retraining - WITH RSI/MACD")
print("="*80)
print(f"\n🎯 Goal: Improve 48.3% → 55-60% win rate")
print(f"   Method: Calculate real RSI/MACD (not defaults)")
print()

# Load data
print("Loading data...")
data_file = DATA_DIR / "BTCUSDT_5m_max.csv"
df = pd.read_csv(data_file)
print(f"  ✅ Loaded: {len(df)} candles")

# Calculate RSI/MACD first (CRITICAL!)
df = calculate_rsi_macd_features(df)

# Then calculate other enhanced features
df = calculate_enhanced_short_features(df)

# Handle NaN
df = df.ffill().fillna(0)
print(f"  ✅ After fillna: {len(df)} rows")

# Create labels (2of3 scoring)
df['label'] = create_2of3_scoring_labels(df)

# Drop NaN rows
df = df.dropna()
print(f"  ✅ Final dataset: {len(df)} rows")

# Feature validation
print("\n" + "="*80)
print("Feature Validation")
print("="*80)

for col in FEATURE_COLUMNS:
    if col not in df.columns:
        print(f"  ❌ Missing: {col}")
    else:
        nan_count = df[col].isna().sum()
        if nan_count > 0:
            print(f"  ⚠️  {col}: {nan_count} NaN values")
        else:
            print(f"  ✅ {col}: min={df[col].min():.4f}, max={df[col].max():.4f}, mean={df[col].mean():.4f}")

# Prepare training data
print("\n" + "="*80)
print("Training Enhanced SHORT Entry Model")
print("="*80)

X = df[FEATURE_COLUMNS].values
y = df['label'].values

print(f"\nDataset:")
print(f"  Total samples: {len(X):,}")
print(f"  Positive (SELL): {y.sum():,} ({y.sum()/len(y)*100:.2f}%)")
print(f"  Negative: {len(y)-y.sum():,} ({(len(y)-y.sum())/len(y)*100:.2f}%)")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain set: {len(X_train):,} samples ({y_train.sum():,} positive)")
print(f"Test set: {len(X_test):,} samples ({y_test.sum():,} positive)")

# MinMax Scaler
print("\nApplying MinMaxScaler...")
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("  ✅ Scaler fitted")

# Train XGBoost
print("\nTraining XGBoost model...")
model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    scale_pos_weight=(len(y_train) - y_train.sum()) / y_train.sum(),
    random_state=42,
    n_jobs=-1
)

model.fit(X_train_scaled, y_train)
print("  ✅ Model trained")

# Evaluate
print("\n" + "="*80)
print("Model Evaluation")
print("="*80)

y_pred = model.predict(X_test_scaled)
y_proba = model.predict_proba(X_test_scaled)[:, 1]

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['No SELL', 'SELL']))

print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Probability distribution
print("\nProbability Distribution:")
for threshold in [0.5, 0.55, 0.6, 0.65, 0.7]:
    high_conf = (y_proba >= threshold).sum()
    if high_conf > 0:
        high_conf_correct = ((y_proba >= threshold) & (y_test == 1)).sum()
        precision = high_conf_correct / high_conf * 100
        print(f"  >= {threshold:.2f}: {high_conf:,} predictions ({high_conf/len(y_test)*100:.2f}%), Precision: {precision:.1f}%")

# Feature importance
print("\n" + "="*80)
print("Top 10 Feature Importance")
print("="*80)

feature_importance = pd.DataFrame({
    'feature': FEATURE_COLUMNS,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

for idx, row in feature_importance.head(10).iterrows():
    print(f"  {row['feature']:25s}: {row['importance']:.4f}")

# Save model
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_filename = f"xgboost_short_entry_enhanced_rsimacd_{timestamp}.pkl"
scaler_filename = f"xgboost_short_entry_enhanced_rsimacd_{timestamp}_scaler.pkl"
features_filename = f"xgboost_short_entry_enhanced_rsimacd_{timestamp}_features.txt"

model_path = MODELS_DIR / model_filename
scaler_path = MODELS_DIR / scaler_filename
features_path = MODELS_DIR / features_filename

print("\n" + "="*80)
print("Saving Model")
print("="*80)

with open(model_path, 'wb') as f:
    pickle.dump(model, f)
print(f"  ✅ Model saved: {model_filename}")

with open(scaler_path, 'wb') as f:
    pickle.dump(scaler, f)
print(f"  ✅ Scaler saved: {scaler_filename}")

with open(features_path, 'w') as f:
    f.write('\n'.join(FEATURE_COLUMNS))
print(f"  ✅ Features saved: {features_filename}")

print("\n" + "="*80)
print("Retraining Complete!")
print("="*80)

print(f"\n📊 Results Summary:")
print(f"   Total samples: {len(X):,}")
print(f"   Positive rate: {y.sum()/len(y)*100:.2f}%")
print(f"   Test accuracy: {(y_pred == y_test).sum() / len(y_test) * 100:.1f}%")
print(f"   Test recall: {cm[1,1] / (cm[1,0] + cm[1,1]) * 100:.1f}%")

print(f"\n🎯 Next Steps:")
print(f"   1. Run window backtest with new model")
print(f"   2. Compare: 48.3% (old) vs ??? (new)")
print(f"   3. Deploy if improved")

print(f"\n📁 Model Files:")
print(f"   Model: {model_path}")
print(f"   Scaler: {scaler_path}")
print(f"   Features: {features_path}")
print()
