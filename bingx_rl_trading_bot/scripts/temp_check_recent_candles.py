#!/usr/bin/env python
"""
최근 캔들 데이터를 직접 로딩하여 LONG/SHORT 확률 계산

목적: 백테스트 vs 실제 운영 차이가 데이터 문제인지 시장 변화인지 확인
"""
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import sys
from datetime import datetime, timedelta

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiments.calculate_all_features import calculate_all_features

MODELS_DIR = PROJECT_ROOT / "models"

print("=" * 80)
print("최근 캔들 데이터 직접 분석 (날짜별)")
print("=" * 80)
print()

# Load models
print("1️⃣ 모델 로딩 중...")
with open(MODELS_DIR / "xgboost_v4_phase4_advanced_lookahead3_thresh0.pkl", 'rb') as f:
    long_model = pickle.load(f)

with open(MODELS_DIR / "xgboost_v4_phase4_advanced_lookahead3_thresh0_scaler.pkl", 'rb') as f:
    long_scaler = pickle.load(f)

with open(MODELS_DIR / "xgboost_v4_phase4_advanced_lookahead3_thresh0_features.txt", 'r') as f:
    long_feature_columns = [line.strip() for line in f.readlines()]

with open(MODELS_DIR / "xgboost_short_redesigned_20251016_233322.pkl", 'rb') as f:
    short_model = pickle.load(f)

with open(MODELS_DIR / "xgboost_short_redesigned_20251016_233322_scaler.pkl", 'rb') as f:
    short_scaler = pickle.load(f)

with open(MODELS_DIR / "xgboost_short_redesigned_20251016_233322_features.txt", 'r') as f:
    short_feature_columns = [line.strip() for line in f.readlines()]

print("   ✅ 모델 로딩 완료")
print()

# Load historical data
print("2️⃣ 히스토리컬 데이터 로딩 중...")
DATA_DIR = PROJECT_ROOT / "data" / "historical"
csv_files = sorted(DATA_DIR.glob("btcusdt_5m_*.csv"))

dfs = []
for csv_file in csv_files:
    df = pd.read_csv(csv_file)
    dfs.append(df)

df_full = pd.concat(dfs, ignore_index=True)
df_full['timestamp'] = pd.to_datetime(df_full['timestamp'])
df_full = df_full.sort_values('timestamp').reset_index(drop=True)

print(f"   전체 데이터: {df_full['timestamp'].min()} ~ {df_full['timestamp'].max()}")
print(f"   총 캔들 수: {len(df_full):,}")
print()

# Filter to recent dates
print("3️⃣ 날짜별 데이터 필터링 및 분석...")
print()

# Dates to analyze
dates_to_check = [
    ('2025-10-15', '2025-10-16'),  # 10/15
    ('2025-10-16', '2025-10-17'),  # 10/16
    ('2025-10-17', '2025-10-18'),  # 10/17
    ('2025-10-18', '2025-10-19'),  # 10/18
    ('2025-10-19', '2025-10-20'),  # 10/19 (오늘)
]

results_summary = []

for start_date, end_date in dates_to_check:
    print("=" * 80)
    print(f"📅 {start_date} 분석")
    print("=" * 80)

    # Filter data for this date
    mask = (df_full['timestamp'] >= start_date) & (df_full['timestamp'] < end_date)
    df_date = df_full[mask].copy()

    if len(df_date) == 0:
        print(f"   ⚠️ {start_date} 데이터 없음")
        print()
        continue

    print(f"   캔들 수: {len(df_date)}")
    print(f"   시간 범위: {df_date['timestamp'].min()} ~ {df_date['timestamp'].max()}")
    print(f"   가격 범위: ${df_date['close'].min():,.1f} ~ ${df_date['close'].max():,.1f}")
    print()

    # Need enough history for features (minimum 200 candles)
    # Get data from 2 days before to have enough history
    start_with_history = pd.to_datetime(start_date) - timedelta(days=2)
    mask_with_history = (df_full['timestamp'] >= start_with_history) & (df_full['timestamp'] < end_date)
    df_with_history = df_full[mask_with_history].copy()

    print(f"   Feature 계산용 데이터: {len(df_with_history)} 캔들 (history 포함)")

    # Calculate features
    df_features = calculate_all_features(df_with_history.copy())

    # Filter back to target date
    df_features_date = df_features[
        (df_features['timestamp'] >= start_date) &
        (df_features['timestamp'] < end_date)
    ].copy()

    # Drop NaN
    df_clean = df_features_date.dropna(subset=long_feature_columns + short_feature_columns)

    if len(df_clean) == 0:
        print(f"   ⚠️ 유효한 데이터 없음 (NaN 제거 후)")
        print()
        continue

    print(f"   유효한 캔들: {len(df_clean)} (NaN 제거 후)")
    print()

    # Calculate probabilities
    X_long = df_clean[long_feature_columns].values
    X_short = df_clean[short_feature_columns].values

    X_long_scaled = long_scaler.transform(X_long)
    X_short_scaled = short_scaler.transform(X_short)

    long_probs = long_model.predict_proba(X_long_scaled)[:, 1]
    short_probs = short_model.predict_proba(X_short_scaled)[:, 1]

    # Statistics
    print("   LONG 확률 통계:")
    print(f"     평균: {long_probs.mean():.4f}")
    print(f"     중앙값: {np.median(long_probs):.4f}")
    print(f"     최소: {long_probs.min():.4f}")
    print(f"     최대: {long_probs.max():.4f}")
    print(f"     >= 0.65: {(long_probs >= 0.65).sum()} ({(long_probs >= 0.65).sum()/len(long_probs)*100:.1f}%)")
    print()

    print("   SHORT 확률 통계:")
    print(f"     평균: {short_probs.mean():.4f}")
    print(f"     중앙값: {np.median(short_probs):.4f}")
    print(f"     >= 0.70: {(short_probs >= 0.70).sum()} ({(short_probs >= 0.70).sum()/len(short_probs)*100:.1f}%)")
    print()

    # Price statistics
    price_change = ((df_clean['close'].iloc[-1] / df_clean['close'].iloc[0]) - 1) * 100
    print(f"   가격 변화: ${df_clean['close'].iloc[0]:,.1f} → ${df_clean['close'].iloc[-1]:,.1f} ({price_change:+.2f}%)")
    print()

    # Save summary
    results_summary.append({
        'date': start_date,
        'candles': len(df_clean),
        'long_mean': long_probs.mean(),
        'long_median': np.median(long_probs),
        'long_above_065': (long_probs >= 0.65).sum() / len(long_probs) * 100,
        'short_mean': short_probs.mean(),
        'short_above_070': (short_probs >= 0.70).sum() / len(short_probs) * 100,
        'price_start': df_clean['close'].iloc[0],
        'price_end': df_clean['close'].iloc[-1],
        'price_change_pct': price_change
    })

# Summary table
print("=" * 80)
print("📊 날짜별 요약")
print("=" * 80)
print()

df_summary = pd.DataFrame(results_summary)
print("날짜         | 캔들수 | LONG평균 | LONG>=0.65 | SHORT평균 | 가격변화")
print("-" * 80)
for _, row in df_summary.iterrows():
    print(f"{row['date']} | {row['candles']:4d}개 | {row['long_mean']:6.4f}  | {row['long_above_065']:6.1f}%  | {row['short_mean']:7.4f} | {row['price_change_pct']:+6.2f}%")

print()
print("=" * 80)
print("🔍 추세 분석")
print("=" * 80)
print()

# Trend analysis
if len(df_summary) >= 2:
    first_day = df_summary.iloc[0]
    last_day = df_summary.iloc[-1]

    print(f"LONG 확률 변화:")
    print(f"  {first_day['date']}: {first_day['long_mean']:.4f} (평균)")
    print(f"  {last_day['date']}: {last_day['long_mean']:.4f} (평균)")
    print(f"  변화: {last_day['long_mean'] - first_day['long_mean']:+.4f}")
    print()

    print(f"진입 기회 (LONG >= 0.65) 변화:")
    print(f"  {first_day['date']}: {first_day['long_above_065']:.1f}%")
    print(f"  {last_day['date']}: {last_day['long_above_065']:.1f}%")
    print(f"  변화: {last_day['long_above_065'] - first_day['long_above_065']:+.1f}%p")
    print()

print("=" * 80)
print("💡 결론")
print("=" * 80)
print()

if len(df_summary) > 0:
    # Check if recent day is unusually high
    recent_long_mean = df_summary.iloc[-1]['long_mean']

    if recent_long_mean > 0.5:
        print("⚠️  최근 LONG 확률이 매우 높습니다!")
        print("   - 백테스트 평균 (0.0972)보다 훨씬 높음")
        print("   - 시장이 강한 bullish 상태이거나")
        print("   - 모델이 과도하게 반응하고 있을 가능성")
        print()
        print("   권장: 실제 진입 후 승률 확인 필요")
    elif recent_long_mean < 0.1:
        print("✅ 최근 LONG 확률이 백테스트와 유사합니다.")
        print("   - 백테스트 평균: 0.0972")
        print("   - 정상적인 범위")
    else:
        print("📊 최근 LONG 확률이 백테스트보다 다소 높습니다.")
        print("   - 시장 상황 변화로 보임")
        print("   - 지속적인 모니터링 필요")

print()
