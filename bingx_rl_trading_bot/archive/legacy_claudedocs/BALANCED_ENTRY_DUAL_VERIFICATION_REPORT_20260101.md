# Balanced Entry Dual Verification Report

**검증일**: 2026-01-01
**목적**: LONG + SHORT 양방향 진입 가능한 균형 전략 이중 검증

---

## Executive Summary

| 결과 | 값 |
|------|-----|
| **테스트 전략 수** | 14개 (7 전략 × 2 TP/SL 조합) |
| **완전 통과 (3/3)** | **0개** |
| **부분 통과 (2/3)** | 2개 (RSI Extreme + Trend) |
| **결론** | ❌ 이중 검증 통과 전략 없음 |

---

## 검증 기준

### Type 1: Signal Quality (신호 품질)
- **통과 조건**: 승률(WR) ≥ 50% AND 기대값(EV) > 0
- 모든 신호에 대해 독립적으로 평가 (포지션 상태 무시)

### Type 2: Actual Trading (실제 거래)
- **통과 조건**: Total PnL > $0
- 포지션 있으면 진입 불가, 복리 효과 반영

### Walk-Forward (일관성)
- **통과 조건**: ≥ 50% 윈도우 수익 (3/6 이상)
- 6개 윈도우로 분할하여 Out-of-Sample 검증

---

## 테스트 전략 목록

### 1. Strict Trend Follow
- **LONG**: EMA20 > EMA50 > EMA100 + Close > EMA20 + RSI > 50
- **SHORT**: EMA20 < EMA50 < EMA100 + Close < EMA20 + RSI < 50
- **특징**: 강한 추세 확인 후 진입

### 2. EMA Pullback
- **LONG**: Close > EMA100 + Close crosses above EMA20 + RSI < 70
- **SHORT**: Close < EMA100 + Close crosses below EMA20 + RSI > 30
- **특징**: 추세 내 되돌림 후 진입

### 3. Breakout Momentum
- **LONG**: Close > BB Upper + ADX > 25 + RSI > 50
- **SHORT**: Close < BB Lower + ADX > 25 + RSI < 50
- **특징**: 볼린저 밴드 돌파 + 모멘텀

### 4. MACD Reversal Strict
- **LONG**: MACD Hist crossover 0 + Close > EMA100 + ADX > 20
- **SHORT**: MACD Hist crossunder 0 + Close < EMA100 + ADX > 20
- **특징**: MACD 히스토그램 제로선 교차 + 추세 필터

### 5. BB Bounce Filtered
- **LONG**: Close < BB Lower + RSI < 35 + Close > EMA200
- **SHORT**: Close > BB Upper + RSI > 65 + Close < EMA200
- **특징**: 볼린저 밴드 터치 + RSI 과매도/과매수

### 6. Multi Confirm Balanced
- **LONG**: RSI < 40 + MACD Hist > Signal + Close > EMA100 + ADX > 18
- **SHORT**: RSI > 60 + MACD Hist < Signal + Close < EMA100 + ADX > 18
- **특징**: 다중 조건 확인

### 7. RSI Extreme + Trend
- **LONG**: RSI < 30 + Close > EMA100 + ADX > 15
- **SHORT**: RSI > 70 + Close < EMA100 + ADX > 15
- **특징**: RSI 극단값 + 추세 필터

---

## 검증 결과

### 전체 결과표

| 전략 | TP/SL | Type1 | LWR | SWR | Type2 | WF | 결과 |
|------|-------|-------|-----|-----|-------|-----|------|
| Strict Trend Follow | 2.5/1.5 | ❌ | 23.2% | 39.0% | ❌ | ✅ 4/6 | ❌ FAIL |
| Strict Trend Follow | 3.0/2.0 | ❌ | 19.7% | 38.2% | ❌ | ✅ 3/6 | ❌ FAIL |
| EMA Pullback | 2.5/1.5 | ❌ | 28.8% | 40.3% | ❌ | ❌ 1/6 | ❌ FAIL |
| EMA Pullback | 3.0/2.0 | ❌ | 23.3% | 37.1% | ❌ | ❌ 2/6 | ❌ FAIL |
| Breakout Momentum | 2.5/1.5 | ❌ | 24.8% | 41.1% | ❌ | ✅ 3/6 | ❌ FAIL |
| Breakout Momentum | 3.0/2.0 | ❌ | 21.7% | 35.7% | ❌ | ❌ 2/6 | ❌ FAIL |
| MACD Reversal Strict | 2.5/1.5 | ❌ | 25.6% | 44.6% | ❌ | ❌ 2/6 | ❌ FAIL |
| MACD Reversal Strict | 3.0/2.0 | ❌ | 25.4% | 38.8% | ❌ | ❌ 2/6 | ❌ FAIL |
| BB Bounce Filtered | 2.0/1.5 | ❌ | 32.4% | 47.1% | ❌ | ✅ 4/6 | ❌ FAIL |
| BB Bounce Filtered | 2.5/1.5 | ❌ | 27.6% | 41.7% | ❌ | ❌ 2/6 | ❌ FAIL |
| Multi Confirm Balanced | 2.5/1.5 | ❌ | 27.7% | 41.0% | ❌ | ✅ 3/6 | ❌ FAIL |
| Multi Confirm Balanced | 3.0/2.0 | ❌ | 25.4% | 38.9% | ❌ | ✅ 3/6 | ❌ FAIL |
| **RSI Extreme + Trend** | **3.0/1.5** | ❌ | 25.3% | **60.0%** | **✅** | **✅ 4/6** | ❌ FAIL |
| **RSI Extreme + Trend** | **4.0/2.0** | ❌ | 0.0% | **62.1%** | **✅** | **✅ 4/6** | ❌ FAIL |

### 핵심 발견

#### 1. LONG 승률 일관적 저조
- **LONG WR 범위**: 0% ~ 32.4%
- **SHORT WR 범위**: 35.7% ~ 62.1%
- **평균 차이**: LONG WR이 SHORT WR 대비 약 15-20%p 낮음

#### 2. Type 1 전 전략 실패
- 14개 전략 모두 Type 1 (WR ≥ 50%, EV > 0) 미통과
- 가장 높은 전체 승률: 41.9% (RSI Extreme + Trend)
- 대부분 30-35% 범위

#### 3. RSI Extreme + Trend 부분 통과 (2/3)
| 조건 | 결과 | 값 |
|------|------|-----|
| Type 1 | ❌ | WR 41.9%, EV +0.387% (WR 미달) |
| Type 2 | ✅ | PnL +$10.64 |
| Walk-Forward | ✅ | 4/6 (66.7%) |

---

## 원인 분석

### 1. 시장 환경 편향 (Bearish Bias)
- **테스트 기간**: 90일 (5분봉 25,920개)
- **시장 특성**: 하락세 우세
- **영향**: SHORT 신호가 더 잘 작동, LONG 신호 성과 저조

### 2. LONG 조건 강화 효과 부족
- 추세 필터 (EMA100, EMA200) 추가에도 LONG WR 개선 미미
- RSI, ADX 등 추가 필터도 LONG WR 50% 달성 불가
- **결론**: 하락장에서는 LONG 진입 자체가 불리

### 3. 양방향 균형의 한계
- 동일한 전략 로직을 반전시키는 것만으로는 양방향 균형 달성 어려움
- LONG과 SHORT는 시장 특성상 비대칭적으로 작동

---

## 결론

### 이중 검증 통과 전략: **없음**

14개 균형 양방향 전략을 테스트한 결과, 이중 검증(Type1 + Type2 + WF)을 완전히 통과하는 전략이 없습니다.

### 가장 유망한 후보: RSI Extreme + Trend

| 지표 | 값 | 비고 |
|------|-----|------|
| 통과 조건 | 2/3 | Type2 ✅, WF ✅ |
| 전체 승률 | 41.9% | 50% 미달 (8.1%p 부족) |
| SHORT WR | **60.0%** | 양호 |
| LONG WR | **25.3%** | 치명적 저조 |
| Type 2 PnL | +$10.64 | 양수 |
| WF 일관성 | 4/6 (66.7%) | 양호 |

---

## 권고사항

### 단기 조치
1. **현재 프로덕션 봇 운영 재검토**
   - 4개 활성 봇 모두 이중 검증 실패
   - 신규 전략도 이중 검증 통과 불가

2. **RSI Extreme + Trend 추가 연구**
   - LONG WR 개선 방안 연구 (추가 필터, 다른 조건)
   - 또는 LONG 비중 축소 + SHORT 비중 확대

### 중기 연구
1. **시장 레짐 필터 적용**
   - 상승장에서만 LONG 진입
   - 하락장에서만 SHORT 진입
   - 횡보장에서는 진입 금지

2. **비대칭 전략 설계**
   - LONG과 SHORT에 다른 Entry 로직 적용
   - 각 방향에 최적화된 파라미터 사용

3. **더 긴 기간 데이터 테스트**
   - 상승장 + 하락장 혼합 데이터
   - 1년 이상 백테스트로 시장 사이클 반영

### 장기 관점
- **현실 인정**: 90일 하락장에서 LONG WR 50%+ 달성은 구조적으로 어려움
- **적응적 전략**: 시장 상황에 따라 방향 편향 조절
- **리스크 관리 우선**: 손실 최소화 후 수익 추구

---

## 관련 파일

| 파일 | 내용 |
|------|------|
| `scripts/analysis/balanced_entry_dual_verification.py` | 검증 스크립트 |
| `results/balanced_entry_dual_verification_20260101_100704.csv` | 상세 결과 CSV |
| `data/btc_5m_90days_v12research.csv` | 테스트 데이터 (25,920 candles) |
| `claudedocs/BACKTEST_VERIFICATION_METHODOLOGY_20251231.md` | 검증 방법론 |
| `claudedocs/PRODUCTION_DUAL_VERIFICATION_20260101.md` | 프로덕션 봇 검증 결과 |

---

**작성자**: Claude AI Assistant
**검토 상태**: 사용자 검토 필요
**버전**: 1.0
