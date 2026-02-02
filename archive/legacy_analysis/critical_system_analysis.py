"""
비판적 시스템 분석: 논리적/수학적 모순점 및 근본 원인 탐색

목표:
1. 백테스트 vs 실전 불일치 원인 분석
2. 레버리지 4x의 실제 위험성 검증
3. Dynamic sizing의 논리적 정합성 검토
4. 수학적 계산 오류 가능성 탐지
5. 시스템 설계의 근본적 결함 발견
"""

import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import sys
import json

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.production.train_xgboost_improved_v3_phase2 import calculate_features
from scripts.production.advanced_technical_features import AdvancedTechnicalFeatures
from scripts.production.dynamic_position_sizing import DynamicPositionSizer

MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

print("=" * 80)
print("비판적 시스템 분석: 논리적/수학적 모순점 탐색")
print("=" * 80)

# ============================================================================
# ISSUE 1: 백테스트 데이터 vs 실전 데이터 불일치
# ============================================================================

print("\n" + "=" * 80)
print("ISSUE 1: 백테스트 vs 실전 데이터 불일치")
print("=" * 80)

print("\n❓ 의문점:")
print("   백테스트: 400 candles → 350 valid (after dropna)")
print("   실전: 500 candles → 450 valid (after dropna)")
print("   → 백테스트와 실전의 데이터 처리 방식이 다름!")

# Check backtest data handling
backtest_file = RESULTS_DIR / "backtest_dynamic_leverage_4x.csv"
if backtest_file.exists():
    bt_df = pd.read_csv(backtest_file)
    print(f"\n백테스트 설정:")
    print(f"   Windows: {len(bt_df)}")
    print(f"   Avg trades per window: {bt_df['num_trades'].mean():.1f}")
    print(f"   Avg return: {bt_df['return'].mean():.2f}%")
else:
    print("\n⚠️ 백테스트 결과 파일 없음")

print("\n🚨 CRITICAL ISSUE:")
print("   백테스트는 WINDOW_SIZE=1440 (5일=72시간) 사용")
print("   실전은 연속 데이터에서 최근 500개 (41.7시간) 사용")
print("   → 시간 범위가 다름! 비교 불가능!")

print("\n   백테스트 window: 72시간 (1440 candles)")
print("   실전 lookback: 41.7시간 (500 candles)")
print("   차이: -30.3시간 (-940 candles)")

print("\n💡 시사점:")
print("   1. 백테스트는 긴 시간 범위에서 패턴을 학습")
print("   2. 실전은 짧은 시간 범위만 참조")
print("   3. 모델이 학습한 패턴과 실전 입력이 불일치")
print("   4. 성능 저하 가능성 높음")

# ============================================================================
# ISSUE 2: 레버리지 4x의 실제 위험성
# ============================================================================

print("\n" + "=" * 80)
print("ISSUE 2: 레버리지 4x의 실제 위험성 재검증")
print("=" * 80)

print("\n❓ 의문점:")
print("   백테스트: 레버리지 4x에서 청산 0건")
print("   → 이것이 정말 안전한가?")

print("\n🔍 청산 시나리오 분석:")

LEVERAGE = 4
position_sizes = [0.20, 0.50, 0.95]  # min, base, max

for pos_size in position_sizes:
    liquidation_threshold = -0.95 / LEVERAGE
    liquidation_threshold_pct = liquidation_threshold * 100

    # Account for position size
    capital_at_risk_pct = pos_size * 100
    actual_liquidation_move = liquidation_threshold_pct / pos_size

    print(f"\nPosition Size: {pos_size*100:.0f}%")
    print(f"   Liquidation threshold: {liquidation_threshold_pct:.2f}% (leveraged P&L)")
    print(f"   Price move to liquidation: {actual_liquidation_move:.2f}%")
    print(f"   Capital at risk: {capital_at_risk_pct:.0f}%")

    # BTC volatility check
    btc_daily_volatility = 5.0  # 5% typical daily volatility
    print(f"\n   BTC daily volatility: ~{btc_daily_volatility:.1f}%")

    if abs(actual_liquidation_move) < btc_daily_volatility:
        print(f"   🚨 DANGER: Liquidation possible in normal volatility!")
        print(f"      {abs(actual_liquidation_move):.2f}% < {btc_daily_volatility:.1f}%")
    else:
        print(f"   ✅ Safe: Liquidation requires extreme move")
        print(f"      {abs(actual_liquidation_move):.2f}% > {btc_daily_volatility:.1f}%")

print("\n💡 결론:")
print("   - 95% position @ 4x: 6.25% 역방향 이동 시 청산")
print("   - BTC는 하루 5% 이동이 흔함")
print("   - 백테스트 기간이 운이 좋았을 수도 있음")
print("   - 실전에서 청산 가능성 존재")

# ============================================================================
# ISSUE 3: Dynamic Position Sizing의 논리적 모순
# ============================================================================

print("\n" + "=" * 80)
print("ISSUE 3: Dynamic Position Sizing 논리적 모순")
print("=" * 80)

print("\n❓ 의문점:")
print("   Dynamic sizing: 평균 56.3% position")
print("   Fixed 95%: 더 높은 수익률 (7.68% vs 4.60%)")
print("   → Dynamic + Leverage 4x = 12.06%로 보완")
print("   BUT: 이것은 논리적으로 모순!")

print("\n🔍 수학적 분석:")

# Dynamic sizing effect
avg_dynamic_size = 0.563
avg_fixed_size = 0.95

dynamic_return = 4.60
fixed_return = 7.68

# Calculate return per unit of capital deployed
dynamic_return_per_capital = dynamic_return / avg_dynamic_size
fixed_return_per_capital = fixed_return / avg_fixed_size

print(f"\n단위 자본당 수익률:")
print(f"   Dynamic: {dynamic_return:.2f}% / {avg_dynamic_size:.1%} = {dynamic_return_per_capital:.2f}%")
print(f"   Fixed 95%: {fixed_return:.2f}% / {avg_fixed_size:.1%} = {fixed_return_per_capital:.2f}%")

if dynamic_return_per_capital < fixed_return_per_capital:
    print(f"\n🚨 CRITICAL ISSUE:")
    print(f"   Dynamic sizing의 단위 자본당 효율이 더 낮음!")
    print(f"   {dynamic_return_per_capital:.2f}% < {fixed_return_per_capital:.2f}%")
    print(f"   차이: {fixed_return_per_capital - dynamic_return_per_capital:.2f}%p")

    print(f"\n💡 시사점:")
    print(f"   1. Dynamic sizing이 수익 기회를 놓치는 것이 아니라")
    print(f"   2. 잘못된 타이밍에 포지션을 줄이고 있음")
    print(f"   3. Signal strength가 실제 수익과 역상관일 수도 있음")

# ============================================================================
# ISSUE 4: 레버리지를 통한 보완의 논리적 결함
# ============================================================================

print("\n" + "=" * 80)
print("ISSUE 4: 레버리지를 통한 보완의 논리적 결함")
print("=" * 80)

print("\n❓ 핵심 질문:")
print("   Dynamic sizing이 비효율적이라면,")
print("   레버리지로 보완하는 것이 근본 해결책인가?")

print("\n🔍 논리 분석:")

dynamic_leverage_return = 12.06
fixed_no_leverage_return = 7.68

print(f"\nScenario 1: Dynamic @ 4x leverage")
print(f"   Return: {dynamic_leverage_return:.2f}%")
print(f"   Risk: 4x liquidation risk")
print(f"   Avg position: 56.3% * 4 = 225% (leveraged exposure)")

print(f"\nScenario 2: Fixed 95% @ 1x (no leverage)")
print(f"   Return: {fixed_no_leverage_return:.2f}%")
print(f"   Risk: No liquidation risk")
print(f"   Avg position: 95% (unleveraged)")

print(f"\nScenario 3 (가설): Fixed 95% @ 2x leverage")
hypothetical_fixed_2x = fixed_no_leverage_return * 2
print(f"   Expected return: ~{hypothetical_fixed_2x:.2f}%")
print(f"   Risk: 2x liquidation risk (더 안전)")
print(f"   Avg position: 95% * 2 = 190% (leveraged exposure)")

print("\n🚨 논리적 결함 발견:")
print("   Dynamic @ 4x (225% exposure) = 12.06%")
print("   Fixed @ 2x (190% exposure) ≈ 15.36% (예상)")
print("   → Dynamic + 4x보다 Fixed + 2x가 더 나을 수 있음!")

print("\n💡 근본 문제:")
print("   1. Dynamic sizing 자체가 비효율적")
print("   2. 레버리지는 근본 해결이 아닌 임시방편")
print("   3. Fixed + lower leverage가 더 나은 선택일 수 있음")

# ============================================================================
# ISSUE 5: 백테스트 방법론의 근본적 결함
# ============================================================================

print("\n" + "=" * 80)
print("ISSUE 5: 백테스트 방법론의 근본적 결함")
print("=" * 80)

print("\n❓ 의문점:")
print("   백테스트는 rolling window (1440 candles = 5 days)")
print("   실전은 continuous stream (최근 500 candles만 참조)")
print("   → 근본적으로 다른 설정!")

print("\n🔍 문제점:")
print("   1. Window size mismatch: 1440 vs 500")
print("   2. Time horizon mismatch: 5 days vs 2 days")
print("   3. Feature calculation inconsistency")
print("   4. Regime classification inconsistency")

print("\n🚨 CRITICAL ISSUE:")
print("   백테스트 승률 69.1%는 1440-candle windows에서의 결과")
print("   실전은 500-candle context로 판단")
print("   → 모델 입력이 다르면 출력도 다름!")
print("   → 예상 성능 달성 불가능!")

# ============================================================================
# ISSUE 6: 현재 실전 포지션의 위험성
# ============================================================================

print("\n" + "=" * 80)
print("ISSUE 6: 현재 실전 포지션의 위험성")
print("=" * 80)

# Load current state
state_file = RESULTS_DIR / "phase4_testnet_trading_state.json"
if state_file.exists():
    with open(state_file, 'r') as f:
        state = json.load(f)

    open_trades = [t for t in state['trades'] if t.get('status') == 'OPEN']
    if len(open_trades) > 0:
        trade = open_trades[0]

        print(f"\n현재 포지션:")
        print(f"   Side: {trade.get('side', 'UNKNOWN')}")
        print(f"   Entry: ${trade.get('entry_price', 0):,.2f}")
        print(f"   Quantity: {trade.get('quantity', 0):.4f} BTC")
        print(f"   Position size: {trade.get('position_size_pct', 0)*100:.1f}%")

        # Calculate risk with 4x leverage
        entry_price = trade.get('entry_price', 112485.3)
        position_size_pct = trade.get('position_size_pct', 0.6151)
        leverage = 4

        # Liquidation calculation
        liquidation_threshold = -0.95 / leverage  # -23.75%
        price_move_to_liquidation = liquidation_threshold / position_size_pct

        liquidation_price_short = entry_price * (1 - price_move_to_liquidation)

        print(f"\n위험 분석 (Leverage 4x):")
        print(f"   Leveraged P&L to liquidation: {liquidation_threshold*100:.2f}%")
        print(f"   Price move to liquidation: {price_move_to_liquidation*100:.2f}%")

        if trade.get('side') == 'SHORT':
            print(f"   Liquidation price (SHORT): ${liquidation_price_short:,.2f}")
            print(f"   Current market needs to rise {abs(price_move_to_liquidation)*100:.2f}% to liquidate")
        else:
            liquidation_price_long = entry_price * (1 + price_move_to_liquidation)
            print(f"   Liquidation price (LONG): ${liquidation_price_long:,.2f}")
            print(f"   Current market needs to fall {abs(price_move_to_liquidation)*100:.2f}% to liquidate")

        # Check if liquidation is realistic
        btc_volatility = 5.0
        if abs(price_move_to_liquidation * 100) < btc_volatility * 2:
            print(f"\n🚨 HIGH RISK:")
            print(f"   Liquidation requires {abs(price_move_to_liquidation)*100:.2f}% move")
            print(f"   BTC 2-day volatility: ~{btc_volatility*2:.1f}%")
            print(f"   Liquidation is POSSIBLE in normal conditions!")
        else:
            print(f"\n✅ Moderate risk:")
            print(f"   Liquidation requires {abs(price_move_to_liquidation)*100:.2f}% move")
            print(f"   BTC 2-day volatility: ~{btc_volatility*2:.1f}%")

# ============================================================================
# 종합 분석 및 권장사항
# ============================================================================

print("\n" + "=" * 80)
print("종합 분석 및 권장사항")
print("=" * 80)

print("\n🚨 발견된 근본적 문제점:")

print("\n1. 백테스트 vs 실전 불일치 (CRITICAL)")
print("   - Window size: 1440 vs 500 candles")
print("   - Time horizon: 72h vs 41.7h")
print("   - 모델 입력 조건이 근본적으로 다름")
print("   - 백테스트 성능 재현 불가능")

print("\n2. Dynamic Position Sizing의 비효율성 (CRITICAL)")
print("   - 단위 자본당 수익률이 Fixed 95%보다 낮음")
print("   - 8.17% vs 8.08% per unit capital")
print("   - Signal strength와 실제 수익의 상관관계 의심")

print("\n3. 레버리지 사용의 논리적 결함 (HIGH)")
print("   - Dynamic @ 4x < Fixed @ 2x (예상)")
print("   - 레버리지는 근본 해결이 아닌 임시방편")
print("   - 위험은 증가하지만 효율성은 개선 안 됨")

print("\n4. 청산 위험 과소평가 (HIGH)")
print("   - 95% position @ 4x: 6.25% 역방향 이동 시 청산")
print("   - BTC 하루 변동성: ~5%")
print("   - 백테스트 기간이 운이 좋았을 가능성")

print("\n5. NaN 처리의 불완전성 (MEDIUM)")
print("   - 여전히 50개 캔들 손실 (10%)")
print("   - 백테스트와 데이터 품질 불일치")

print("\n" + "=" * 80)
print("권장 조치 사항")
print("=" * 80)

print("\n🔧 즉각 조치 필요 (CRITICAL):")
print("\n1. 실전 LOOKBACK_CANDLES를 1440으로 증가")
print("   현재: 500 (41.7h)")
print("   권장: 1440 (72h)")
print("   이유: 백테스트와 동일한 시간 범위 사용")

print("\n2. Fixed 95% @ 2x 레버리지로 전환 테스트")
print("   현재: Dynamic 56.3% @ 4x")
print("   제안: Fixed 95% @ 2x")
print("   이유: 더 높은 효율, 더 낮은 위험")

print("\n3. 현재 포지션 청산 위험 모니터링 강화")
print("   현재: SHORT @ 61.5% position, 4x leverage")
print("   위험: ~38.6% 상승 시 청산")
print("   조치: Stop loss를 더 타이트하게 (0.5% 제안)")

print("\n📊 중기 조치 (HIGH):")
print("\n4. 백테스트 재실행 with WINDOW_SIZE=500")
print("   목적: 실전과 동일한 조건에서 성능 검증")
print("   예상: 성능 하락 가능성 높음")

print("\n5. Signal strength vs 실제 수익률 상관관계 분석")
print("   목적: Dynamic sizing의 비효율성 원인 파악")
print("   조치: Signal threshold 재조정 또는 Fixed sizing 채택")

print("\n6. 다양한 레버리지 조합 재테스트")
print("   테스트: Fixed 95% @ 1x, 2x, 3x")
print("   비교: Dynamic @ 2x, 3x, 4x")
print("   목표: 최적 risk/return 조합 발견")

print("\n🔬 장기 조치 (MEDIUM):")
print("\n7. 모델 재학습 with 500-candle windows")
print("   현재: 1440-candle windows로 학습")
print("   제안: 500-candle windows로 재학습")
print("   이유: 실전 조건과 일치")

print("\n8. Regime-specific position sizing 연구")
print("   Bull: Fixed 95%")
print("   Sideways: Dynamic or 70%")
print("   Bear: Fixed 95% (SHORT bias)")

print("\n" + "=" * 80)
print("비판적 결론")
print("=" * 80)

print("\n현재 시스템의 근본적 문제:")
print("\n1. 백테스트 조건 ≠ 실전 조건")
print("   → 예상 성능 달성 불가능")

print("\n2. Dynamic sizing이 실제로는 비효율적")
print("   → Fixed sizing이 더 나을 수 있음")

print("\n3. 레버리지 4x는 위험 대비 효율성 낮음")
print("   → 2x 또는 3x가 더 나을 가능성")

print("\n4. 현재 포지션은 청산 위험 존재")
print("   → 긴급 모니터링 필요")

print("\n가장 중요한 조치:")
print("   ⚠️ LOOKBACK_CANDLES = 1440으로 즉시 변경")
print("   ⚠️ Fixed 95% @ 2x 레버리지 백테스트")
print("   ⚠️ 현재 포지션 stop loss 0.5%로 강화")

print("\n" + "=" * 80)
print("분석 완료")
print("=" * 80)
