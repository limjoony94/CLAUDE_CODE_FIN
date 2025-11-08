"""
Compare signal frequency: Backtest vs Production
"""
import pandas as pd
import numpy as np

# Read latest backtest CSV
backtest_file = "results/full_backtest_opportunity_gating_4x_20251020_041825.csv"
df = pd.read_csv(backtest_file)

print("="*80)
print("📊 백테스트 vs 프로덕션 신호 빈도 비교")
print("="*80)

# Backtest analysis
total_windows = len(df)
avg_trades_per_window = df['total_trades'].mean()
total_trades = df['total_trades'].sum()

print("\n🔬 백테스트 데이터 (105일):")
print(f"  총 윈도우: {total_windows}개 (5일 간격)")
print(f"  평균 거래 수: {avg_trades_per_window:.1f} 거래/윈도우")
print(f"  총 거래 수: {total_trades:.0f}개")

# Calculate candles per window
candles_per_window = 5 * 24 * 60 / 5  # 5 days * 24 hours * 60 minutes / 5 min candles
print(f"\n  캔들 per 윈도우: {candles_per_window:.0f}개 (5일)")
print(f"  거래 빈도: {avg_trades_per_window / candles_per_window * 100:.2f}% (캔들당)")

# But this is TRADES, not SIGNALS!
# A trade only happens if:
# 1. Signal is strong enough (>= threshold)
# 2. No position is currently open
# 3. Opportunity gate passes (for SHORT)

print("\n⚠️  중요: 이것은 '실제 거래 수'입니다 (신호가 아님)")
print("  실제 거래는 다음 조건을 모두 만족해야 함:")
print("    1. 신호 강도 >= threshold")
print("    2. 현재 포지션 없음")
print("    3. Opportunity gate 통과 (SHORT의 경우)")

print("\n" + "="*80)
print("🤖 프로덕션 봇 (최근 4시간)")
print("="*80)

print("\n  총 체크포인트: 91개")
print("  LONG 신호: 48개 (52.7%)")
print("  SHORT 신호: 0개 (0%)")

print("\n⚠️  중요: 이것은 '모든 신호'입니다 (실제 거래 여부와 무관)")
print("  로그에는 포지션 보유 중에도 신호가 기록됨")

print("\n" + "="*80)
print("💡 분석")
print("="*80)

print("\n1. 백테스트 '거래 수' vs 프로덕션 '신호 수'")
print("   - 백테스트: 실제 진입한 거래만 카운트")
print("   - 프로덕션: threshold 넘는 모든 신호 로깅")
print("   → 직접 비교 불가능")

print("\n2. 프로덕션 봇이 거래하지 않는 이유:")
print("   - 이미 LONG 포지션 보유 중")
print("   - 48개 LONG 신호 중 단 1개만 실제 진입 (포지션 없을 때)")
print("   → 실제 거래 빈도는 백테스트와 유사할 것으로 예상")

print("\n3. 백테스트에서 신호 빈도 확인 필요:")
print("   - 백테스트 코드 수정하여 threshold 넘는 모든 신호 카운트")
print("   - 그래야 프로덕션과 공정한 비교 가능")

# Try to estimate signal frequency from backtest
# If we assume similar rejection rate (position already open, etc)
estimated_signal_rate = avg_trades_per_window / candles_per_window * 100
estimated_rejection_factor = 3  # Conservative estimate

print(f"\n4. 백테스트 신호 빈도 추정:")
print(f"   - 실제 거래: {estimated_signal_rate:.2f}% of candles")
print(f"   - 추정 신호 빈도 (거부율 3배 가정): {estimated_signal_rate * estimated_rejection_factor:.2f}%")
print(f"   - 프로덕션 신호 빈도: 52.7%")
print(f"   → 프로덕션이 {'높음' if 52.7 > estimated_signal_rate * estimated_rejection_factor else '비슷함'}")

print("\n" + "="*80)
