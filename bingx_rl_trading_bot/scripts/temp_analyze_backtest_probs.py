#!/usr/bin/env python
"""
백테스트 기간의 실제 LONG/SHORT 확률 분포 분석

백테스트에 사용한 데이터와 같은 모델로 확률 계산하여
실제 백테스트 때의 확률 분포를 확인
"""
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiments.calculate_all_features import calculate_all_features

DATA_DIR = PROJECT_ROOT / "data" / "historical"
MODELS_DIR = PROJECT_ROOT / "models"

print("=" * 80)
print("백테스트 기간 LONG/SHORT 확률 분포 분석")
print("=" * 80)
print()

# Load models (SAME AS PRODUCTION BOT - 10/18 UPGRADED)
print("1️⃣ 모델 로딩 중...")
print("   ⚠️  프로덕션 봇과 동일한 모델 사용 (10/18 업그레이드)")

with open(MODELS_DIR / "xgboost_long_trade_outcome_full_20251018_233146.pkl", 'rb') as f:
    long_model = pickle.load(f)

with open(MODELS_DIR / "xgboost_long_trade_outcome_full_20251018_233146_scaler.pkl", 'rb') as f:
    long_scaler = pickle.load(f)

with open(MODELS_DIR / "xgboost_long_trade_outcome_full_20251018_233146_features.txt", 'r') as f:
    long_feature_columns = [line.strip() for line in f.readlines()]

with open(MODELS_DIR / "xgboost_short_trade_outcome_full_20251018_233146.pkl", 'rb') as f:
    short_model = pickle.load(f)

with open(MODELS_DIR / "xgboost_short_trade_outcome_full_20251018_233146_scaler.pkl", 'rb') as f:
    short_scaler = pickle.load(f)

with open(MODELS_DIR / "xgboost_short_trade_outcome_full_20251018_233146_features.txt", 'r') as f:
    short_feature_columns = [line.strip() for line in f.readlines()]

print("   ✅ 모델 로딩 완료")
print()

# Load historical data (same as backtest)
print("2️⃣ 백테스트 데이터 로딩 중...")
csv_files = sorted(DATA_DIR.glob("btcusdt_5m_*.csv"))
print(f"   발견된 파일: {len(csv_files)}개")

dfs = []
for csv_file in csv_files:
    df = pd.read_csv(csv_file)
    dfs.append(df)

df_full = pd.concat(dfs, ignore_index=True)
df_full['timestamp'] = pd.to_datetime(df_full['timestamp'])
df_full = df_full.sort_values('timestamp').reset_index(drop=True)

print(f"   데이터 기간: {df_full['timestamp'].min()} ~ {df_full['timestamp'].max()}")
print(f"   총 캔들 수: {len(df_full):,}")
print()

# Calculate features
print("3️⃣ Feature 계산 중...")
df_features = calculate_all_features(df_full.copy())
print(f"   ✅ Feature 계산 완료 ({len(df_features):,} rows)")
print()

# Drop rows with NaN
df_clean = df_features.dropna(subset=long_feature_columns + short_feature_columns)
print(f"   유효한 행: {len(df_clean):,} (NaN 제거 후)")
print()

# Calculate probabilities
print("4️⃣ LONG/SHORT 확률 계산 중...")

# Prepare features
X_long = df_clean[long_feature_columns].values
X_short = df_clean[short_feature_columns].values

# Scale features
X_long_scaled = long_scaler.transform(X_long)
X_short_scaled = short_scaler.transform(X_short)

# Predict probabilities
long_probs = long_model.predict_proba(X_long_scaled)[:, 1]
short_probs = short_model.predict_proba(X_short_scaled)[:, 1]

print("   ✅ 확률 계산 완료")
print()

# Analyze distribution
print("=" * 80)
print("📊 백테스트 기간 확률 분포 (전체 데이터)")
print("=" * 80)
print()

print("LONG Probability 통계:")
print(f"  평균: {long_probs.mean():.4f}")
print(f"  중앙값: {np.median(long_probs):.4f}")
print(f"  표준편차: {long_probs.std():.4f}")
print(f"  최소: {long_probs.min():.4f}")
print(f"  최대: {long_probs.max():.4f}")
print()

print("LONG Probability 분포:")
print(f"  < 0.3: {(long_probs < 0.3).sum():,} ({(long_probs < 0.3).sum()/len(long_probs)*100:.1f}%)")
print(f"  0.3-0.5: {((long_probs >= 0.3) & (long_probs < 0.5)).sum():,} ({((long_probs >= 0.3) & (long_probs < 0.5)).sum()/len(long_probs)*100:.1f}%)")
print(f"  0.5-0.65: {((long_probs >= 0.5) & (long_probs < 0.65)).sum():,} ({((long_probs >= 0.5) & (long_probs < 0.65)).sum()/len(long_probs)*100:.1f}%)")
print(f"  0.65-0.8: {((long_probs >= 0.65) & (long_probs < 0.8)).sum():,} ({((long_probs >= 0.65) & (long_probs < 0.8)).sum()/len(long_probs)*100:.1f}%)")
print(f"  >= 0.8: {(long_probs >= 0.8).sum():,} ({(long_probs >= 0.8).sum()/len(long_probs)*100:.1f}%)")
print()

print("SHORT Probability 통계:")
print(f"  평균: {short_probs.mean():.4f}")
print(f"  중앙값: {np.median(short_probs):.4f}")
print(f"  표준편차: {short_probs.std():.4f}")
print(f"  최소: {short_probs.min():.4f}")
print(f"  최대: {short_probs.max():.4f}")
print()

print("SHORT Probability 분포:")
print(f"  < 0.1: {(short_probs < 0.1).sum():,} ({(short_probs < 0.1).sum()/len(short_probs)*100:.1f}%)")
print(f"  0.1-0.3: {((short_probs >= 0.1) & (short_probs < 0.3)).sum():,} ({((short_probs >= 0.1) & (short_probs < 0.3)).sum()/len(short_probs)*100:.1f}%)")
print(f"  0.3-0.5: {((short_probs >= 0.3) & (short_probs < 0.5)).sum():,} ({((short_probs >= 0.3) & (short_probs < 0.5)).sum()/len(short_probs)*100:.1f}%)")
print(f"  0.5-0.7: {((short_probs >= 0.5) & (short_probs < 0.7)).sum():,} ({((short_probs >= 0.5) & (short_probs < 0.7)).sum()/len(short_probs)*100:.1f}%)")
print(f"  >= 0.7: {(short_probs >= 0.7).sum():,} ({(short_probs >= 0.7).sum()/len(short_probs)*100:.1f}%)")
print()

print("=" * 80)
print("🎯 Threshold 초과 비율 (실제 진입 조건)")
print("=" * 80)
print(f"LONG >= 0.65 (진입): {(long_probs >= 0.65).sum():,} ({(long_probs >= 0.65).sum()/len(long_probs)*100:.1f}%)")
print(f"SHORT >= 0.70 (후보): {(short_probs >= 0.70).sum():,} ({(short_probs >= 0.70).sum()/len(short_probs)*100:.1f}%)")
print()
print(f"총 샘플 수: {len(long_probs):,}")
print()

# Compare with live data
print("=" * 80)
print("📊 실제 운영 vs 백테스트 비교")
print("=" * 80)
print()

# Live data (from previous analysis - 10/19)
live_long_mean = 0.8147
live_long_median = 0.8595
live_long_above_065 = 88.9
live_short_mean = 0.0030
live_short_median = 0.0012

backtest_long_mean = long_probs.mean()
backtest_long_median = np.median(long_probs)
backtest_long_above_065 = (long_probs >= 0.65).sum() / len(long_probs) * 100
backtest_short_mean = short_probs.mean()
backtest_short_median = np.median(short_probs)

print("LONG 확률:")
print(f"  평균:    실제 {live_long_mean:.4f} | 백테스트 {backtest_long_mean:.4f} | 차이: {abs(live_long_mean - backtest_long_mean):.4f}")
print(f"  중앙값:  실제 {live_long_median:.4f} | 백테스트 {backtest_long_median:.4f} | 차이: {abs(live_long_median - backtest_long_median):.4f}")
print(f"  >= 0.65: 실제 {live_long_above_065:.1f}% | 백테스트 {backtest_long_above_065:.1f}% | 차이: {abs(live_long_above_065 - backtest_long_above_065):.1f}%p")
print()

print("SHORT 확률:")
print(f"  평균:    실제 {live_short_mean:.4f} | 백테스트 {backtest_short_mean:.4f} | 차이: {abs(live_short_mean - backtest_short_mean):.4f}")
print(f"  중앙값:  실제 {live_short_median:.4f} | 백테스트 {backtest_short_median:.4f} | 차이: {abs(live_short_median - backtest_short_median):.4f}")
print()

# Conclusion
print("=" * 80)
print("🔍 분석 결론")
print("=" * 80)
print()

if abs(live_long_mean - backtest_long_mean) < 0.2:
    print("✅ 실제 운영과 백테스트의 LONG 확률 평균이 유사합니다.")
else:
    print("⚠️  실제 운영과 백테스트의 LONG 확률 평균에 차이가 있습니다.")
    print(f"   차이: {abs(live_long_mean - backtest_long_mean):.4f}")

print()

if abs(live_long_above_065 - backtest_long_above_065) < 20:
    print("✅ 진입 조건 만족 비율이 백테스트와 유사합니다.")
else:
    print("⚠️  진입 조건 만족 비율이 백테스트와 다릅니다.")
    print(f"   차이: {abs(live_long_above_065 - backtest_long_above_065):.1f}%p")

print()
print("💡 참고:")
print("   - 백테스트는 전체 기간 평균이므로 단기 변동과 다를 수 있습니다")
print("   - 시장 상황에 따라 확률 분포가 변하는 것은 정상입니다")
print("   - 중요한 것은 모델의 예측력(승률)이지 확률 분포 자체가 아닙니다")
print()
