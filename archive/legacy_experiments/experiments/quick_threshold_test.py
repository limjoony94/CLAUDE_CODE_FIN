"""
빠른 Threshold 테스트 - Phase 4 기술적 지표 사용
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from src.indicators.technical_indicators import TechnicalIndicators

print("="*80)
print("🔴 SHORT THRESHOLD 빠른 테스트")
print("="*80)

# 데이터 로드
df = pd.read_csv("data/historical/BTCUSDT_5m_max.csv")
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df.sort_values('timestamp')

# 최근 30일
cutoff = df['timestamp'].max() - pd.Timedelta(days=30)
df = df[df['timestamp'] >= cutoff].reset_index(drop=True)

print(f"\n📊 데이터: {len(df)} rows")
print(f"📅 기간: {df['timestamp'].min()} ~ {df['timestamp'].max()}")

# SHORT 모델 로드 (3-class model - V3 봇과 동일)
short_model_path = "models/xgboost_v4_phase4_3class_lookahead3_thresh3.pkl"
short_model = joblib.load(short_model_path)
print(f"\n✅ SHORT 모델 로드 완료: {os.path.basename(short_model_path)}")

# 3-class model인지 확인
is_3class = hasattr(short_model, 'classes_') and len(short_model.classes_) == 3
print(f"   Model type: {'3-class' if is_3class else 'Binary'}")

# Phase 4 기술적 지표 계산 (모델이 기대하는 features)
print(f"📊 기술적 지표 계산 중...")
ti = TechnicalIndicators()
df = ti.calculate_all_indicators(df)
print(f"✅ 지표 계산 완료: {len(df)} rows, {len(df.columns)} columns")

# 예측 가능한 feature만 선택
feature_cols = [col for col in df.columns if col not in
               ['timestamp', 'open', 'high', 'low', 'close', 'volume']]
X = df[feature_cols]
print(f"📊 Feature 개수: {len(feature_cols)}")

# SHORT 확률 계산
try:
    # 3-class model의 경우 class 2 (SHORT) 확률 사용
    if is_3class:
        short_probs = short_model.predict_proba(X)[:, 2]  # Class 0=HOLD, 1=LONG, 2=SHORT
        print(f"✅ SHORT 확률 계산 완료 (3-class model, using class 2)")
    else:
        short_probs = short_model.predict_proba(X)[:, 1]
        print(f"✅ SHORT 확률 계산 완료 (binary model)")

    df['short_prob'] = short_probs
    print(f"   확률 범위: {short_probs.min():.3f} ~ {short_probs.max():.3f}")
    print(f"   평균 확률: {short_probs.mean():.3f}")
except Exception as e:
    print(f"❌ 확률 계산 실패: {e}")
    print(f"   Feature 수: {len(feature_cols)}")
    print(f"   Expected: {short_model.n_features_in_ if hasattr(short_model, 'n_features_in_') else 'unknown'}")
    print(f"   Features: {feature_cols[:5]}...")
    sys.exit(1)

# 여러 threshold 테스트
thresholds = [0.3, 0.4, 0.5, 0.6, 0.7]
results = []

print(f"\n{'='*80}")
print(f"📊 Threshold 테스트 시작")
print(f"{'='*80}\n")

for thresh in thresholds:
    trades = []
    position = None
    capital = 10000

    for i in range(len(df)):
        row = df.iloc[i]

        # 포지션 보유 중
        if position:
            entry_time, entry_price, entry_idx = position
            current_price = row['close']
            hours_held = (row['timestamp'] - entry_time).total_seconds() / 3600

            # SHORT이므로 가격 하락이 수익
            pnl_pct = (entry_price - current_price) / entry_price * 100

            # 청산 조건
            exit_reason = None
            if pnl_pct >= 3.0:
                exit_reason = "TP"
            elif pnl_pct <= -1.5:
                exit_reason = "SL"
            elif hours_held >= 4:
                exit_reason = "Max Hold"

            if exit_reason:
                capital *= (1 + pnl_pct / 100)
                trades.append({
                    'pnl_pct': pnl_pct,
                    'hours': hours_held,
                    'reason': exit_reason
                })
                position = None

        # 새 포지션 진입
        elif row['short_prob'] >= thresh:
            position = (row['timestamp'], row['close'], i)

    # 통계
    if len(trades) > 0:
        trades_df = pd.DataFrame(trades)
        win_rate = (trades_df['pnl_pct'] > 0).mean() * 100
        avg_pnl = trades_df['pnl_pct'].mean()
        total_return = (capital - 10000) / 10000 * 100

        results.append({
            'threshold': thresh,
            'trades': len(trades),
            'win_rate': win_rate,
            'avg_pnl': avg_pnl,
            'total_return': total_return,
            'score': win_rate * total_return
        })

        print(f"Threshold {thresh}:")
        print(f"  거래 수: {len(trades)}")
        print(f"  승률: {win_rate:.1f}%")
        print(f"  평균 P&L: {avg_pnl:.2f}%")
        print(f"  총 수익률: {total_return:.2f}%")
        print(f"  Score: {win_rate * total_return:.1f}")
        print()
    else:
        print(f"Threshold {thresh}: ⚠️ 거래 없음\n")

# 최적값 선택
if results:
    results_df = pd.DataFrame(results)
    best_idx = results_df['score'].idxmax()
    best = results_df.loc[best_idx]

    print(f"{'='*80}")
    print(f"🏆 최적 SHORT Threshold: {best['threshold']}")
    print(f"{'='*80}")
    print(f"승률: {best['win_rate']:.1f}% (페이퍼: ~30%)")
    print(f"거래 수: {int(best['trades'])}")
    print(f"총 수익률: {best['total_return']:.2f}%/30일")
    print(f"Score: {best['score']:.1f}")
    print()

    # 결과 저장
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_df.to_csv(f"claudedocs/quick_threshold_test_{timestamp}.csv", index=False)
    print(f"💾 결과 저장: claudedocs/quick_threshold_test_{timestamp}.csv")

    print(f"\n{'='*80}")
    print(f"📋 다음 단계:")
    print(f"{'='*80}")
    print(f"1. 테스트넷 봇에 Threshold {best['threshold']} 적용")
    print(f"2. 실전 데이터 수집 (n=20-30)")
    print(f"3. 검증 완료 후 Mainnet 전환")
else:
    print("❌ 유효한 결과 없음")
