"""
Debug: 왜 백테스트에서 거래가 0.1만 발생하는가?

비판적 사고:
- Training: Prob > 0.3이 77.7%
- Backtest: 0.1 trades
- → 무언가가 거래를 차단하고 있음

가설:
1. Volatility filter (min_volatility=0.0008)이 대부분 차단?
2. NaN features가 많아서 skip?
3. Backtest data의 probability가 training data와 다름?
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import ta

PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"

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

print("=" * 80)
print("Debug: Backtest Low Trades Problem")
print("=" * 80)

# Load model
model_file = MODELS_DIR / "xgboost_v2_lookahead3_thresh1.pkl"
with open(model_file, 'rb') as f:
    model = pickle.load(f)
print(f"✅ Model loaded: {model_file}")

# Load data
data_file = DATA_DIR / "historical" / "BTCUSDT_5m_max.csv"
df = pd.read_csv(data_file)
print(f"✅ Data loaded: {len(df)} candles")

# Calculate features
df = calculate_features(df)
df_full = df.copy()
df = df.dropna()
print(f"✅ Features calculated: {len(df)} rows after dropna")

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

# Predict on all data
print(f"\n{'=' * 80}")
print("PREDICTING ON ALL DATA")
print(f"{'=' * 80}")

X = df[feature_columns].values
probabilities = model.predict_proba(X)[:, 1]

print(f"\nProbability Distribution (Full Data):")
print(f"  Total samples: {len(probabilities)}")
print(f"  Mean: {probabilities.mean():.4f}")
print(f"  Std: {probabilities.std():.4f}")
print(f"  Min: {probabilities.min():.4f}")
print(f"  Max: {probabilities.max():.4f}")
print(f"  Median: {np.median(probabilities):.4f}")

print(f"\nProbability Thresholds:")
for threshold in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]:
    count = (probabilities > threshold).sum()
    pct = count / len(probabilities) * 100
    print(f"  Prob > {threshold:.1f}: {count} ({pct:.2f}%)")

# Check volatility filter
print(f"\n{'=' * 80}")
print("VOLATILITY ANALYSIS")
print(f"{'=' * 80}")

volatilities = df['volatility'].values
min_volatility_options = [0.0001, 0.0003, 0.0005, 0.0008, 0.001, 0.002]

print(f"\nVolatility Distribution:")
print(f"  Mean: {volatilities.mean():.6f}")
print(f"  Median: {np.median(volatilities):.6f}")
print(f"  Min: {volatilities.min():.6f}")
print(f"  Max: {volatilities.max():.6f}")

print(f"\nVolatility Filter Impact:")
for min_vol in min_volatility_options:
    passed = (volatilities >= min_vol).sum()
    blocked = (volatilities < min_vol).sum()
    pct_blocked = blocked / len(volatilities) * 100
    print(f"  min_vol={min_vol:.4f}: {passed} passed, {blocked} blocked ({pct_blocked:.1f}% blocked)")

# Simulate entry conditions
print(f"\n{'=' * 80}")
print("SIMULATING ENTRY CONDITIONS")
print(f"{'=' * 80}")

entry_threshold_options = [0.3, 0.4, 0.5]
min_volatility_options = [0.0, 0.0001, 0.0005, 0.0008]

for prob_threshold in entry_threshold_options:
    for min_vol in min_volatility_options:
        # Entry conditions:
        # 1. probability > threshold
        # 2. volatility >= min_volatility
        # 3. no NaN features (already filtered by dropna)

        prob_pass = probabilities > prob_threshold
        vol_pass = volatilities >= min_vol
        both_pass = prob_pass & vol_pass

        total_opportunities = both_pass.sum()
        pct_opportunities = total_opportunities / len(probabilities) * 100

        print(f"\n  Prob>{prob_threshold:.1f}, MinVol={min_vol:.4f}:")
        print(f"    Prob pass: {prob_pass.sum()} ({prob_pass.sum() / len(probabilities) * 100:.1f}%)")
        print(f"    Vol pass: {vol_pass.sum()} ({vol_pass.sum() / len(volatilities) * 100:.1f}%)")
        print(f"    Both pass: {total_opportunities} ({pct_opportunities:.1f}%)")
        print(f"    Expected trades in 60-day window: {total_opportunities / len(probabilities) * 60:.1f}")

# Check specific window
print(f"\n{'=' * 80}")
print("FIRST WINDOW ANALYSIS (0-60 days)")
print(f"{'=' * 80}")

window_size = 60
window_df = df.iloc[0:window_size].copy()

X_window = window_df[feature_columns].values
prob_window = model.predict_proba(X_window)[:, 1]
vol_window = window_df['volatility'].values

print(f"\nWindow Probability Distribution:")
print(f"  Mean: {prob_window.mean():.4f}")
print(f"  Median: {np.median(prob_window):.4f}")
print(f"  Min: {prob_window.min():.4f}")
print(f"  Max: {prob_window.max():.4f}")
print(f"  Prob > 0.3: {(prob_window > 0.3).sum()} ({(prob_window > 0.3).sum() / len(prob_window) * 100:.1f}%)")

print(f"\nWindow Volatility Distribution:")
print(f"  Mean: {vol_window.mean():.6f}")
print(f"  Median: {np.median(vol_window):.6f}")
print(f"  Min: {vol_window.min():.6f}")
print(f"  Max: {vol_window.max():.6f}")
print(f"  Vol >= 0.0008: {(vol_window >= 0.0008).sum()} ({(vol_window >= 0.0008).sum() / len(vol_window) * 100:.1f}%)")

# Final diagnosis
print(f"\n{'=' * 80}")
print("DIAGNOSIS")
print(f"{'=' * 80}")

prob_pass_rate = (probabilities > 0.3).sum() / len(probabilities) * 100
vol_pass_rate = (volatilities >= 0.0008).sum() / len(volatilities) * 100
both_pass_rate = ((probabilities > 0.3) & (volatilities >= 0.0008)).sum() / len(probabilities) * 100

print(f"\n전체 데이터 Entry 조건 만족률:")
print(f"  Probability > 0.3: {prob_pass_rate:.1f}%")
print(f"  Volatility >= 0.0008: {vol_pass_rate:.1f}%")
print(f"  Both: {both_pass_rate:.1f}%")

if both_pass_rate < 5:
    print(f"\n🔴 근본 원인 발견:")
    if vol_pass_rate < 20:
        print(f"  ❌ Volatility filter가 대부분 차단! ({100 - vol_pass_rate:.1f}% blocked)")
        print(f"  ✅ 해결: min_volatility 낮추기 (0.0008 → 0.0 or 0.0001)")
    elif prob_pass_rate < 20:
        print(f"  ❌ Probability가 너무 낮음! (Pass rate: {prob_pass_rate:.1f}%)")
        print(f"  ✅ 해결: Lookahead 더 줄이기 (3 → 2 candles) or Threshold 낮추기")
    else:
        print(f"  ❌ 두 조건의 조합 문제")
        print(f"  ✅ 해결: 둘 다 완화 필요")
else:
    print(f"\n✅ Entry 조건 만족률 충분: {both_pass_rate:.1f}%")
    print(f"  → 백테스트 로직에 다른 문제 있음 (포지션 관리, 타이밍 등)")

print(f"\n{'=' * 80}")
print("Debug Complete!")
print(f"{'=' * 80}")
