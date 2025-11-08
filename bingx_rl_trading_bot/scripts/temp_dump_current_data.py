#!/usr/bin/env python
"""
현재 봇이 사용 중인 데이터를 덤프하여 분석

봇이 실제로 사용하는 1000개 캔들을 저장하고 확률 계산
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.api.bingx_client import BingXClient
import yaml
import pandas as pd
import numpy as np
import pickle
from datetime import datetime, timedelta

from scripts.experiments.calculate_all_features import calculate_all_features
import joblib

MODELS_DIR = Path(__file__).parent.parent / "models"

print("=" * 80)
print("현재 봇 데이터 덤프 및 분석")
print("=" * 80)
print()

# Load API keys (use testnet to avoid hitting mainnet rate limits)
print("1️⃣ API 연결 중...")
with open('config/api_keys.yaml', 'r') as f:
    config = yaml.safe_load(f)

api_key = config['bingx']['testnet']['api_key']
api_secret = config['bingx']['testnet']['secret_key']

client = BingXClient(api_key, api_secret, testnet=True)
print("   ✅ Testnet 연결 완료")
print()

# Load models (SAME AS PRODUCTION BOT - 10/18 UPGRADED)
print("2️⃣ 모델 로딩 중...")
print("   ⚠️  프로덕션 봇과 동일한 모델 사용 (10/18 업그레이드)")

with open(MODELS_DIR / "xgboost_long_trade_outcome_full_20251018_233146.pkl", 'rb') as f:
    long_model = pickle.load(f)

long_scaler = joblib.load(MODELS_DIR / "xgboost_long_trade_outcome_full_20251018_233146_scaler.pkl")

with open(MODELS_DIR / "xgboost_long_trade_outcome_full_20251018_233146_features.txt", 'r') as f:
    long_feature_columns = [line.strip() for line in f.readlines()]

with open(MODELS_DIR / "xgboost_short_trade_outcome_full_20251018_233146.pkl", 'rb') as f:
    short_model = pickle.load(f)

short_scaler = joblib.load(MODELS_DIR / "xgboost_short_trade_outcome_full_20251018_233146_scaler.pkl")

with open(MODELS_DIR / "xgboost_short_trade_outcome_full_20251018_233146_features.txt", 'r') as f:
    short_feature_columns = [line.strip() for line in f.readlines()]

print("   ✅ 모델 로딩 완료")
print()

# Get latest 1000 candles (same as bot)
print("3️⃣ 최신 1000개 캔들 가져오기...")
klines = client.get_klines('BTC-USDT', '5m', limit=1000)

if klines is None or len(klines) == 0:
    print("   ❌ 데이터 가져오기 실패!")
    sys.exit(1)

# Convert to DataFrame
df = pd.DataFrame(klines)
df['timestamp'] = pd.to_datetime(df['time'], unit='ms')
for col in ['open', 'high', 'low', 'close', 'volume']:
    df[col] = df[col].astype(float)

print(f"   ✅ {len(df)}개 캔들 가져옴")
print(f"   시간 범위: {df['timestamp'].min()} ~ {df['timestamp'].max()}")
print(f"   가격 범위: ${df['close'].min():,.1f} ~ ${df['close'].max():,.1f}")
print()

# Calculate features
print("4️⃣ Feature 계산 중...")
df_features = calculate_all_features(df.copy())
df_clean = df_features.dropna(subset=long_feature_columns + short_feature_columns)
print(f"   ✅ {len(df_clean)} 유효한 행 (NaN 제거 후)")
print()

# Calculate probabilities for all candles
print("5️⃣ 전체 캔들 확률 계산 중...")
X_long = df_clean[long_feature_columns].values
X_short = df_clean[short_feature_columns].values

X_long_scaled = long_scaler.transform(X_long)
X_short_scaled = short_scaler.transform(X_short)

long_probs = long_model.predict_proba(X_long_scaled)[:, 1]
short_probs = short_model.predict_proba(X_short_scaled)[:, 1]

print(f"   ✅ 확률 계산 완료")
print()

# Analyze by recent time periods
print("=" * 80)
print("📊 시간대별 확률 분포 분석")
print("=" * 80)
print()

# Define time periods to analyze
latest_time = df_clean['timestamp'].max()

time_periods = [
    ('최근 1시간', timedelta(hours=1)),
    ('최근 3시간', timedelta(hours=3)),
    ('최근 6시간', timedelta(hours=6)),
    ('최근 12시간', timedelta(hours=12)),
    ('최근 24시간', timedelta(hours=24)),
    ('전체 (1000 캔들)', None)
]

results = []

for period_name, delta in time_periods:
    if delta is not None:
        cutoff_time = latest_time - delta
        mask = df_clean['timestamp'] >= cutoff_time
        df_period = df_clean[mask]
        long_period = long_probs[mask.values]
        short_period = short_probs[mask.values]
    else:
        df_period = df_clean
        long_period = long_probs
        short_period = short_probs

    if len(df_period) == 0:
        continue

    print(f"{period_name}:")
    print(f"  캔들 수: {len(df_period)}")
    print(f"  시간: {df_period['timestamp'].min()} ~ {df_period['timestamp'].max()}")
    print(f"  LONG 평균: {long_period.mean():.4f}")
    print(f"  LONG 중앙값: {np.median(long_period):.4f}")
    print(f"  LONG >= 0.65: {(long_period >= 0.65).sum()} ({(long_period >= 0.65).sum()/len(long_period)*100:.1f}%)")
    print(f"  SHORT 평균: {short_period.mean():.4f}")
    print(f"  SHORT >= 0.70: {(short_period >= 0.70).sum()} ({(short_period >= 0.70).sum()/len(short_period)*100:.1f}%)")

    # Price change
    price_start = df_period['close'].iloc[0]
    price_end = df_period['close'].iloc[-1]
    price_change = ((price_end / price_start) - 1) * 100
    print(f"  가격 변화: ${price_start:,.1f} → ${price_end:,.1f} ({price_change:+.2f}%)")
    print()

    results.append({
        'period': period_name,
        'candles': len(df_period),
        'long_mean': long_period.mean(),
        'long_above_065': (long_period >= 0.65).sum() / len(long_period) * 100,
        'price_change': price_change
    })

# Latest candle details
print("=" * 80)
print("🔍 최신 캔들 상세 정보")
print("=" * 80)
print()

latest_idx = len(df_clean) - 1
latest_candle = df_clean.iloc[latest_idx]
latest_long_prob = long_probs[latest_idx]
latest_short_prob = short_probs[latest_idx]

print(f"시간: {latest_candle['timestamp']}")
print(f"가격: ${latest_candle['close']:,.2f}")
print(f"LONG 확률: {latest_long_prob:.4f}")
print(f"SHORT 확률: {latest_short_prob:.4f}")
print()

# Last 10 candles
print("최근 10개 캔들:")
print("시간                 | 가격        | LONG prob | SHORT prob")
print("-" * 70)
for i in range(max(0, len(df_clean) - 10), len(df_clean)):
    candle = df_clean.iloc[i]
    print(f"{candle['timestamp']} | ${candle['close']:9,.1f} | {long_probs[i]:.4f}    | {short_probs[i]:.4f}")

print()

# Comparison with backtest
print("=" * 80)
print("📊 백테스트 vs 현재 비교")
print("=" * 80)
print()

backtest_long_mean = 0.0972  # From previous analysis
current_long_mean = long_probs.mean()

print(f"LONG 확률 평균:")
print(f"  백테스트 (7-10월 전체): {backtest_long_mean:.4f}")
print(f"  현재 (1000 캔들):       {current_long_mean:.4f}")
print(f"  차이:                   {current_long_mean - backtest_long_mean:+.4f}")
print()

if current_long_mean > 0.5:
    print("⚠️  현재 LONG 확률이 백테스트보다 매우 높습니다!")
    print("   최근 시장이 강한 bullish 상태이거나")
    print("   모델이 과도하게 반응하고 있을 가능성")
elif current_long_mean > backtest_long_mean * 2:
    print("⚠️  현재 LONG 확률이 백테스트의 2배 이상입니다.")
    print("   시장 상황이 크게 변했을 가능성")
else:
    print("✅ 현재 LONG 확률이 백테스트와 비슷한 수준입니다.")

print()
