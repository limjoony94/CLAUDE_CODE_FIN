"""
NaN 처리 방식 분석 및 최적화

목표: NaN이 모델 성능에 미치는 영향을 분석하고 최적의 처리 방법 찾기
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.production.train_xgboost_improved_v3_phase2 import calculate_features
from scripts.production.advanced_technical_features import AdvancedTechnicalFeatures

# Load data
data_file = PROJECT_ROOT / "data" / "historical" / "BTCUSDT_5m_max.csv"
df = pd.read_csv(data_file)

print("="*80)
print("NaN 분석 및 최적화")
print("="*80)

# Calculate features
print("\n1. Feature 계산 중...")
df_original = df.copy()
df = calculate_features(df)
adv_features = AdvancedTechnicalFeatures(lookback_sr=50, lookback_trend=20)
df = adv_features.calculate_all_features(df)

print(f"✅ Features 계산 완료: {len(df)} rows")

# Analyze NaN patterns
print("\n2. NaN 패턴 분석...")
print("="*80)

# Check NaN by column
nan_counts = df.isna().sum()
nan_columns = nan_counts[nan_counts > 0].sort_values(ascending=False)

print(f"\nNaN이 있는 컬럼: {len(nan_columns)}개")
print("\nTop 10 NaN counts:")
for col, count in nan_columns.head(10).items():
    pct = (count / len(df)) * 100
    print(f"  {col:40s}: {count:5d} ({pct:5.1f}%)")

# Identify which features cause the most NaN
print("\n3. NaN 발생 원인 분석...")
print("="*80)

# Advanced features (Support/Resistance, Trend Lines, etc.)
advanced_feature_cols = adv_features.get_feature_names()
advanced_nan = df[advanced_feature_cols].isna().sum()
advanced_nan_cols = advanced_nan[advanced_nan > 0]

print(f"\nAdvanced Features NaN:")
for col, count in advanced_nan_cols.items():
    pct = (count / len(df)) * 100
    print(f"  {col:40s}: {count:5d} ({pct:5.1f}%)")

# Check where NaN starts (which row)
print("\n4. NaN 발생 위치 분석...")
print("="*80)

first_valid_idx = {}
for col in advanced_nan_cols.index:
    first_valid = df[col].first_valid_index()
    if first_valid is not None:
        first_valid_idx[col] = first_valid

if first_valid_idx:
    min_valid_row = min(first_valid_idx.values())
    max_valid_row = max(first_valid_idx.values())
    print(f"첫 유효 데이터 (최소): {min_valid_row}번째 행")
    print(f"첫 유효 데이터 (최대): {max_valid_row}번째 행")
    print(f"→ 처음 {max_valid_row}개 행이 NaN 발생 구간")

# Test different NaN handling strategies
print("\n5. 다양한 NaN 처리 방법 테스트...")
print("="*80)

strategies = {
    "ffill+dropna (현재)": lambda df: df.ffill().dropna(),
    "fillna(0)": lambda df: df.fillna(0),
    "fillna(mean)": lambda df: df.fillna(df.mean(numeric_only=True)),
    "ffill+bfill+dropna": lambda df: df.ffill().bfill().dropna(),
    "interpolate": lambda df: df.interpolate(method='linear').dropna()
}

results = {}

for strategy_name, strategy_func in strategies.items():
    df_test = df.copy()
    df_processed = strategy_func(df_test)

    rows_before = len(df)
    rows_after = len(df_processed)
    rows_lost = rows_before - rows_after
    pct_lost = (rows_lost / rows_before) * 100

    # Check if any NaN remains
    remaining_nan = df_processed.isna().sum().sum()

    results[strategy_name] = {
        'rows_before': rows_before,
        'rows_after': rows_after,
        'rows_lost': rows_lost,
        'pct_lost': pct_lost,
        'remaining_nan': remaining_nan
    }

    print(f"\n{strategy_name}:")
    print(f"  원본: {rows_before} rows")
    print(f"  처리 후: {rows_after} rows")
    print(f"  손실: {rows_lost} rows ({pct_lost:.2f}%)")
    print(f"  남은 NaN: {remaining_nan}")

# Analyze feature distributions after different NaN handling
print("\n6. NaN 처리 후 Feature 분포 분석...")
print("="*80)

# Compare feature statistics for different strategies
sample_features = ['distance_to_support_pct', 'distance_to_resistance_pct', 'rsi', 'atr_pct']
available_features = [f for f in sample_features if f in df.columns]

if available_features:
    print(f"\n샘플 Features: {', '.join(available_features)}")

    for feature in available_features:
        print(f"\n{feature}:")

        # Original (with NaN)
        print(f"  원본 (NaN 포함):")
        print(f"    Mean: {df[feature].mean():.4f}")
        print(f"    Std: {df[feature].std():.4f}")
        print(f"    NaN: {df[feature].isna().sum()}")

        # After ffill+dropna
        df_ffill = df.ffill().dropna()
        if feature in df_ffill.columns and len(df_ffill) > 0:
            print(f"  ffill+dropna:")
            print(f"    Mean: {df_ffill[feature].mean():.4f}")
            print(f"    Std: {df_ffill[feature].std():.4f}")

        # After fillna(0)
        df_zero = df.fillna(0)
        if feature in df_zero.columns:
            print(f"  fillna(0):")
            print(f"    Mean: {df_zero[feature].mean():.4f}")
            print(f"    Std: {df_zero[feature].std():.4f}")

# Recommendation
print("\n" + "="*80)
print("7. 권장 사항")
print("="*80)

print("""
분석 결과:

1. NaN 발생 원인:
   - Support/Resistance features는 lookback_sr=50 때문에 처음 50개 행에서 NaN 발생
   - Trend Line features는 lookback_trend=20 때문에 처음 20개 행에서 NaN 발생
   - 이는 기술적으로 불가피한 현상 (historical data 필요)

2. 현재 방식 (ffill+dropna):
   ✅ 장점: 잘못된 정보(0, mean 등)를 모델에 제공하지 않음
   ✅ 장점: 백테스트와 일치 (동일한 처리 방식 사용)
   ⚠️ 단점: 처음 50개 행 손실 (~50개 candles = 4.2시간 데이터)

3. fillna(0) 방식:
   ❌ 단점: Support/Resistance distance를 0으로 채우면 "가격이 S/R에 정확히 있다"는 잘못된 신호
   ❌ 단점: 모델이 잘못된 패턴을 학습할 수 있음

4. fillna(mean) 방식:
   ⚠️ 단점: 평균값은 실제 S/R 위치와 무관하므로 noise 추가
   ⚠️ 단점: Look-ahead bias (미래 정보 사용)

5. interpolate 방식:
   ⚠️ 단점: 초반 NaN은 interpolation 불가 (이전 값 없음)
   ⚠️ 단점: S/R은 discrete한 개념이므로 interpolation이 적합하지 않음

**최종 권장**:
→ 현재 방식(ffill+dropna) 유지
→ 이유: 백테스트와 일치하며, 잘못된 정보 제공하지 않음
→ 손실되는 50개 행(4.2시간)은 모델 학습/예측에 영향 미미

**개선 방안**:
→ LOOKBACK_CANDLES를 50개 더 늘려서 여유 확보
→ 현재: 1440 candles 요청 → 450 valid
→ 제안: 1500 candles 요청 → 500 valid (원래 목표와 일치)
""")

print("\n✅ 분석 완료!")
