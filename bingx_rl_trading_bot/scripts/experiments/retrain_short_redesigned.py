"""
SHORT Model Complete Redesign - Symmetric & Inverse Features

Problem: Current indicators are LONG-biased (0-100 scales favor bullish)
Solution:
1. Symmetric features (both directions equally weighted)
2. Inverse features (SHORT-specific decline detection)
3. Opportunity cost features (SHORT vs LONG comparison)

Goal: LONG+SHORT > LONG-only (+10.14%)
"""

import pandas as pd
import numpy as np
import pickle
import talib
from pathlib import Path
from datetime import datetime
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import sys

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

DATA_DIR = PROJECT_ROOT / "data" / "historical"
MODELS_DIR = PROJECT_ROOT / "models"

# Parameters
LOOKAHEAD = 3
PROFIT_THRESHOLD = 0.006  # 0.6%
LEAD_TIME_WINDOW = 12

def calculate_symmetric_features(df):
    """
    Symmetric features - LONG/SHORT equally weighted

    Key insight: Measure DISTANCE and DIRECTION separately
    """
    print("\n=== Symmetric Features (Unbiased) ===")

    # RSI: Deviation from neutral (50)
    df['rsi_raw'] = talib.RSI(df['close'], timeperiod=14)
    df['rsi_deviation'] = np.abs(df['rsi_raw'] - 50)  # Distance from neutral
    df['rsi_direction'] = np.sign(df['rsi_raw'] - 50)  # +1 bullish, -1 bearish
    df['rsi_extreme'] = ((df['rsi_raw'] > 70) | (df['rsi_raw'] < 30)).astype(float)  # Either extreme

    # MACD: Absolute strength + direction
    macd, macd_signal, macd_hist = talib.MACD(df['close'], fastperiod=12, slowperiod=26, signalperiod=9)
    df['macd_strength'] = np.abs(macd_hist)  # Strength regardless of direction
    df['macd_direction'] = np.sign(macd_hist)  # Direction separate
    df['macd_divergence_abs'] = np.abs(macd - macd_signal)

    # Price vs MA: Distance + direction
    ma_20 = df['close'].rolling(20).mean()
    ma_50 = df['close'].rolling(50).mean()
    df['price_distance_ma20'] = np.abs(df['close'] - ma_20) / ma_20  # Absolute distance
    df['price_direction_ma20'] = np.sign(df['close'] - ma_20)  # Direction
    df['price_distance_ma50'] = np.abs(df['close'] - ma_50) / ma_50
    df['price_direction_ma50'] = np.sign(df['close'] - ma_50)

    # Volatility (direction-agnostic)
    df['volatility'] = df['close'].pct_change().rolling(20).std()
    df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
    df['atr_pct'] = df['atr'] / df['close']

    print(f"  ✅ Symmetric features: 13 features")
    return df

def calculate_inverse_features(df):
    """
    Inverse features - SHORT-specific decline detection

    Focus on downward movement, bearish patterns, weakness
    """
    print("\n=== Inverse Features (SHORT-specific) ===")

    # Negative momentum
    df['negative_momentum'] = -df['close'].pct_change(5).clip(upper=0)  # Only negative changes
    df['negative_acceleration'] = -df['close'].diff().diff().clip(upper=0)

    # Down candle analysis
    df['down_candle'] = (df['close'] < df['open']).astype(float)
    df['down_candle_ratio'] = df['down_candle'].rolling(10).mean()  # Ratio of down candles
    df['down_candle_body'] = (df['open'] - df['close']).clip(lower=0)  # Down candle body size

    # Lower lows (weakness)
    df['lower_low'] = (df['low'] < df['low'].shift(1)).astype(float)
    df['lower_low_streak'] = df['lower_low'].rolling(5).sum()  # Consecutive lower lows

    # Resistance rejection (failed breakouts)
    resistance = df['high'].rolling(20).max()
    df['near_resistance'] = (df['high'] > resistance.shift(1) * 0.99).astype(float)
    df['rejection_from_resistance'] = (
        (df['near_resistance'] == 1) & (df['close'] < df['open'])
    ).astype(float)
    df['resistance_rejection_count'] = df['rejection_from_resistance'].rolling(10).sum()

    # Bearish divergence (price up but momentum down)
    price_change_5 = df['close'].diff(5)
    rsi_change_5 = df['rsi_raw'].diff(5) if 'rsi_raw' in df.columns else 0
    df['bearish_divergence'] = (
        (price_change_5 > 0) & (rsi_change_5 < 0)
    ).astype(float)

    # Volume on decline vs advance
    df['price_up'] = (df['close'] > df['close'].shift(1)).astype(float)
    df['volume_on_decline'] = df['volume'] * (1 - df['price_up'])
    df['volume_on_advance'] = df['volume'] * df['price_up']
    df['volume_decline_ratio'] = (
        df['volume_on_decline'].rolling(10).sum() /
        (df['volume_on_advance'].rolling(10).sum() + 1e-10)
    )

    # Distribution phase detection (high volume, price stagnant)
    price_range_20 = (df['high'].rolling(20).max() - df['low'].rolling(20).min()) / df['close']
    volume_surge = df['volume'] / df['volume'].rolling(20).mean()
    df['distribution_signal'] = (
        (price_range_20 < 0.05) &  # Narrow range
        (volume_surge > 1.5)  # High volume
    ).astype(float)

    print(f"  ✅ Inverse features: 15 features")
    return df

def calculate_opportunity_cost_features(df):
    """
    Opportunity cost features - SHORT vs LONG comparison

    Label SHORT only when it's BETTER than LONG
    """
    print("\n=== Opportunity Cost Features ===")

    # Bear market strength
    returns_20 = df['close'].pct_change(20)
    df['bear_market_strength'] = (-returns_20).clip(lower=0)  # Only negative returns

    # Trend strength (negative for downtrend)
    df['ema_12'] = df['close'].ewm(span=12).mean()
    df['ema_26'] = df['close'].ewm(span=26).mean()
    df['trend_strength'] = (df['ema_12'] - df['ema_26']) / df['close']  # Negative = downtrend
    df['downtrend_confirmed'] = (df['trend_strength'] < -0.01).astype(float)

    # Downside vs upside volatility asymmetry
    returns = df['close'].pct_change()
    upside_vol = returns[returns > 0].rolling(20).std()
    downside_vol = returns[returns < 0].rolling(20).std()
    df['downside_volatility'] = downside_vol.fillna(method='ffill')
    df['upside_volatility'] = upside_vol.fillna(method='ffill')
    df['volatility_asymmetry'] = df['downside_volatility'] / (df['upside_volatility'] + 1e-10)

    # Support breakdown (bearish)
    support = df['low'].rolling(50).min()
    df['below_support'] = (df['close'] < support.shift(1) * 1.01).astype(float)
    df['support_breakdown'] = (
        (df['close'].shift(1) >= support.shift(1)) &
        (df['close'] < support.shift(1))
    ).astype(float)

    # Weak hands exit (high volume down candles)
    df['panic_selling'] = (
        (df['down_candle'] == 1) &
        (df['volume'] > df['volume'].rolling(10).mean() * 1.5)
    ).astype(float)

    print(f"  ✅ Opportunity cost features: 10 features")
    return df

def create_opportunity_aware_labels(df, lookahead=LOOKAHEAD, profit_threshold=PROFIT_THRESHOLD, lead_time_window=LEAD_TIME_WINDOW):
    """
    Opportunity-aware labeling for SHORT

    Label = 1 ONLY if:
    1. SHORT has good profit potential (>= profit_threshold)
    2. SHORT happens within reasonable time (lead_time_window)
    3. SHORT is BETTER than waiting for LONG

    This ensures we only take SHORT when it's the BEST opportunity
    """
    print("\n=== Opportunity-Aware Labeling ===")

    labels = []

    for i in range(len(df) - lookahead - lead_time_window):
        current_price = df['close'].iloc[i]

        # SHORT opportunity
        future_prices = df['close'].iloc[i+lookahead:i+lookahead+lead_time_window]
        if len(future_prices) == 0:
            labels.append(0)
            continue

        min_price = future_prices.min()
        min_idx = future_prices.idxmin()
        short_profit = (current_price - min_price) / current_price

        # LONG opportunity (for comparison)
        max_price = future_prices.max()
        long_profit = (max_price - current_price) / current_price

        # Criteria
        criterion_1 = short_profit >= profit_threshold  # Good SHORT profit
        criterion_2 = (min_idx - (i + lookahead)) <= lead_time_window / 2  # Happens early
        criterion_3 = short_profit > long_profit * 1.2  # SHORT is 20% better than LONG

        # Label = 1 ONLY if ALL criteria met AND SHORT > LONG
        labels.append(1 if (criterion_1 and criterion_2 and criterion_3) else 0)

    # Pad
    labels.extend([0] * (lookahead + lead_time_window))

    positive_rate = sum(labels) / len(labels) * 100
    print(f"  ✅ Labels created: {positive_rate:.2f}% positive")
    print(f"     (Only when SHORT is BETTER than LONG)")

    return np.array(labels)

# Feature list (38 total)
FEATURE_COLUMNS = [
    # Symmetric (13)
    'rsi_deviation', 'rsi_direction', 'rsi_extreme',
    'macd_strength', 'macd_direction', 'macd_divergence_abs',
    'price_distance_ma20', 'price_direction_ma20',
    'price_distance_ma50', 'price_direction_ma50',
    'volatility', 'atr_pct', 'atr',

    # Inverse (15)
    'negative_momentum', 'negative_acceleration',
    'down_candle_ratio', 'down_candle_body',
    'lower_low_streak', 'resistance_rejection_count',
    'bearish_divergence', 'volume_decline_ratio',
    'distribution_signal', 'down_candle',
    'lower_low', 'near_resistance', 'rejection_from_resistance',
    'volume_on_decline', 'volume_on_advance',

    # Opportunity Cost (10)
    'bear_market_strength', 'trend_strength', 'downtrend_confirmed',
    'volatility_asymmetry', 'below_support', 'support_breakdown',
    'panic_selling', 'downside_volatility', 'upside_volatility', 'ema_12'
]

print("="*80)
print("SHORT Model Complete Redesign - Symmetric & Inverse Features")
print("="*80)
print(f"\n🎯 Goal: SHORT that BEATS LONG opportunities")
print(f"   Features: {len(FEATURE_COLUMNS)} (38 total)")
print(f"   - Symmetric: 13 (unbiased LONG/SHORT)")
print(f"   - Inverse: 15 (SHORT-specific)")
print(f"   - Opportunity Cost: 10 (SHORT vs LONG)")
print()

# Load data
print("Loading data...")
data_file = DATA_DIR / "BTCUSDT_5m_max.csv"
df = pd.read_csv(data_file)
print(f"  ✅ Loaded: {len(df)} candles")

# Calculate features
df = calculate_symmetric_features(df)
df = calculate_inverse_features(df)
df = calculate_opportunity_cost_features(df)

# Handle NaN
df = df.ffill().bfill().fillna(0)
print(f"\n  ✅ After NaN handling: {len(df)} rows")

# Create labels
df['label'] = create_opportunity_aware_labels(df)

# Drop remaining NaN
df = df.dropna()
print(f"  ✅ Final dataset: {len(df)} rows")

# Prepare training data
print("\n" + "="*80)
print("Training Redesigned SHORT Model")
print("="*80)

X = df[FEATURE_COLUMNS].values
y = df['label'].values

print(f"\nDataset:")
print(f"  Total samples: {len(X):,}")
print(f"  Positive (SELL): {y.sum():,} ({y.sum()/len(y)*100:.2f}%)")
print(f"  Negative: {len(y)-y.sum():,} ({(len(y)-y.sum())/len(y)*100:.2f}%)")

if y.sum() < 50:
    print(f"\n  ⚠️  WARNING: Very few positive samples!")
    print(f"     May need to adjust opportunity cost threshold")

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y if y.sum() >= 50 else None
)

print(f"\nTrain set: {len(X_train):,} samples ({y_train.sum():,} positive)")
print(f"Test set: {len(X_test):,} samples ({y_test.sum():,} positive)")

# StandardScaler (better for symmetric features)
print("\nApplying StandardScaler...")
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
print("  ✅ StandardScaler fitted (better for symmetric features)")

# Train XGBoost
print("\nTraining XGBoost model...")
model = XGBClassifier(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.1,
    scale_pos_weight=(len(y_train) - y_train.sum()) / max(y_train.sum(), 1),
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
for threshold in [0.5, 0.6, 0.7, 0.8, 0.9]:
    high_conf = (y_proba >= threshold).sum()
    if high_conf > 0:
        high_conf_correct = ((y_proba >= threshold) & (y_test == 1)).sum()
        precision = high_conf_correct / high_conf * 100
        print(f"  >= {threshold:.1f}: {high_conf:,} predictions ({high_conf/len(y_test)*100:.2f}%), Precision: {precision:.1f}%")

# Feature importance
print("\n" + "="*80)
print("Top 15 Feature Importance")
print("="*80)

feature_importance = pd.DataFrame({
    'feature': FEATURE_COLUMNS,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

for idx, row in feature_importance.head(15).iterrows():
    print(f"  {row['feature']:30s}: {row['importance']:.4f}")

# Save model
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_filename = f"xgboost_short_redesigned_{timestamp}.pkl"
scaler_filename = f"xgboost_short_redesigned_{timestamp}_scaler.pkl"
features_filename = f"xgboost_short_redesigned_{timestamp}_features.txt"

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
print("Redesign Complete!")
print("="*80)

print(f"\n📊 Model Summary:")
print(f"   Features: {len(FEATURE_COLUMNS)} (38 redesigned)")
print(f"   Positive rate: {y.sum()/len(y)*100:.2f}%")
print(f"   Test accuracy: {(y_pred == y_test).sum() / len(y_test) * 100:.1f}%")
if cm[1,0] + cm[1,1] > 0:
    print(f"   Test recall: {cm[1,1] / (cm[1,0] + cm[1,1]) * 100:.1f}%")

print(f"\n🎯 Next Steps:")
print(f"   1. Run window backtest (101 windows)")
print(f"   2. Test thresholds (0.7-0.9)")
print(f"   3. Compare with LONG-only (+10.14%)")
print(f"   4. Goal: LONG+SHORT > 10.14%")

print(f"\n📁 Model Files:")
print(f"   {model_path}")
print(f"   {scaler_path}")
print(f"   {features_path}")
print()
