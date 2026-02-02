"""
Quick check: Signal frequency in backtest data
"""
import pandas as pd
import numpy as np
import pickle
import joblib
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiments.calculate_all_features import calculate_all_features

DATA_DIR = PROJECT_ROOT / "data" / "historical"
MODELS_DIR = PROJECT_ROOT / "models"

# Thresholds
LONG_THRESHOLD = 0.65
SHORT_THRESHOLD = 0.70

print("="*80)
print("📊 백테스트 데이터에서 신호 빈도 확인")
print("="*80)

# Load models
print("\nLoading models...")
long_model_path = MODELS_DIR / "xgboost_long_trade_outcome_full_20251018_233146.pkl"
with open(long_model_path, 'rb') as f:
    long_model = pickle.load(f)

long_scaler_path = MODELS_DIR / "xgboost_long_trade_outcome_full_20251018_233146_scaler.pkl"
long_scaler = joblib.load(long_scaler_path)

long_features_path = MODELS_DIR / "xgboost_long_trade_outcome_full_20251018_233146_features.txt"
with open(long_features_path, 'r') as f:
    long_feature_columns = [line.strip() for line in f.readlines()]

short_model_path = MODELS_DIR / "xgboost_short_trade_outcome_full_20251018_233146.pkl"
with open(short_model_path, 'rb') as f:
    short_model = pickle.load(f)

short_scaler_path = MODELS_DIR / "xgboost_short_trade_outcome_full_20251018_233146_scaler.pkl"
short_scaler = joblib.load(short_scaler_path)

short_features_path = MODELS_DIR / "xgboost_short_trade_outcome_full_20251018_233146_features.txt"
with open(short_features_path, 'r') as f:
    short_feature_columns = [line.strip() for line in f.readlines()]

print("✅ Models loaded")

# Load recent data (last 2 days only for speed)
print("\nLoading recent data (last 2 days)...")
data_path = DATA_DIR / "BTCUSDT_5m_max.csv"
df = pd.read_csv(data_path)
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Take last 576 candles (2 days)
df = df.tail(576).copy()
print(f"✅ Loaded {len(df)} candles")

# Calculate features
print("\nCalculating features...")
df = calculate_all_features(df)
print(f"✅ Features calculated")

# Generate signals
print("\nGenerating signals...")
long_signals = 0
short_signals = 0
total_candles = len(df)

for i in range(len(df)):
    try:
        # LONG signal
        long_features_values = df[long_feature_columns].iloc[i:i+1].values
        if not np.isnan(long_features_values).any():
            long_features_scaled = long_scaler.transform(long_features_values)
            long_prob = long_model.predict_proba(long_features_scaled)[0][1]

            if long_prob >= LONG_THRESHOLD:
                long_signals += 1

        # SHORT signal
        short_features_values = df[short_feature_columns].iloc[i:i+1].values
        if not np.isnan(short_features_values).any():
            short_features_scaled = short_scaler.transform(short_features_values)
            short_prob = short_model.predict_proba(short_features_scaled)[0][1]

            if short_prob >= SHORT_THRESHOLD:
                short_signals += 1

    except Exception as e:
        continue

print("\n" + "="*80)
print("📊 결과")
print("="*80)

print(f"\n총 캔들: {total_candles}개 (2일)")
print(f"LONG 신호: {long_signals}개 ({long_signals/total_candles*100:.1f}%)")
print(f"SHORT 신호: {short_signals}개 ({short_signals/total_candles*100:.1f}%)")
print(f"신호 없음: {total_candles - long_signals}개 ({(total_candles - long_signals)/total_candles*100:.1f}%)")

print("\n" + "="*80)
print("💡 프로덕션과 비교")
print("="*80)

print(f"\n백테스트 (최근 2일):")
print(f"  LONG: {long_signals/total_candles*100:.1f}%")
print(f"  SHORT: {short_signals/total_candles*100:.1f}%")

print(f"\n프로덕션 (최근 4시간):")
print(f"  LONG: 52.7%")
print(f"  SHORT: 0%")

if long_signals/total_candles*100 > 40:
    print(f"\n⚠️  백테스트도 높은 신호 빈도를 보임!")
    print(f"  → 모델이 원래 많은 신호를 생성하는 것일 수 있음")
    print(f"  → Backtest는 포지션 보유 중에 신호 무시하여 거래 수 적음")
else:
    print(f"\n⚠️  프로덕션이 비정상적으로 높은 신호 빈도")
    print(f"  → 데이터 차이 또는 모델 문제 가능성")

print("\n" + "="*80)
