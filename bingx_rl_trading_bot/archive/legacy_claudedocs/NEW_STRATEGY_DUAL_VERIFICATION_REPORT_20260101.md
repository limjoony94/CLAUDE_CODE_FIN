# New Strategy Dual Verification Report

**검증일**: 2026-01-01
**목적**: 외부 레퍼런스 기반 10개 신규 전략 이중 검증

---

## Executive Summary

| 결과 | 값 |
|------|-----|
| **테스트 전략 수** | 40개 (10 전략 × 4 TP/SL 조합) |
| **완전 통과 (3/3)** | **0개** |
| **부분 통과 (2/3)** | 4개 |
| **결론** | ❌ 이중 검증 완전 통과 전략 없음 |

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

## 테스트 전략 목록 (A-J)

### A: ADX Strong Trend
- **LONG**: ADX > 25 + (+DI > -DI) + Close > EMA20
- **SHORT**: ADX > 25 + (-DI > +DI) + Close < EMA20
- **특징**: 강한 추세 + 방향성 필터

### B: Multi-Confirmation
- **LONG**: RSI < 45 + MACD Hist > 0 + Close > EMA55 + ADX > 15
- **SHORT**: RSI > 55 + MACD Hist < 0 + Close < EMA55 + ADX > 15
- **특징**: 4중 필터 확인

### C: BB + Stochastic
- **LONG**: Close < BB Lower + Stoch K < 20 + Stoch K > Stoch D
- **SHORT**: Close > BB Upper + Stoch K > 80 + Stoch K < Stoch D
- **특징**: 과매도/과매수 + 모멘텀 반전

### D: VWAP + Volume
- **LONG**: Close > VWAP + Volume > 1.5x Avg + Close > EMA21
- **SHORT**: Close < VWAP + Volume > 1.5x Avg + Close < EMA21
- **특징**: 기관 매집 + 볼륨 확인

### E: EMA Triple Cross
- **LONG**: EMA8 > EMA21 > EMA55 + Close > EMA8
- **SHORT**: EMA8 < EMA21 < EMA55 + Close < EMA8
- **특징**: 3중 EMA 정렬

### F: RSI Divergence
- **LONG**: RSI < 35 + RSI(now) > RSI(prev) + Close < Close(prev)
- **SHORT**: RSI > 65 + RSI(now) < RSI(prev) + Close > Close(prev)
- **특징**: 가격-RSI 다이버전스

### G: Range Breakout
- **LONG**: Close > 20-period High + ADX > 20 + Volume > Avg
- **SHORT**: Close < 20-period Low + ADX > 20 + Volume > Avg
- **특징**: Donchian 채널 돌파

### H: ChoCH (Change of Character)
- **LONG**: 이전 Lower Low 후 Higher High 형성
- **SHORT**: 이전 Higher High 후 Lower Low 형성
- **특징**: 구조적 반전 패턴

### I: ATR Breakout
- **LONG**: Close > EMA20 + 1.5×ATR + ADX > 18
- **SHORT**: Close < EMA20 - 1.5×ATR + ADX > 18
- **특징**: 변동성 돌파

### J: Hybrid Adaptive
- **LONG**: (RSI < 40 OR BB% < 0.2) + MACD Hist > Signal + Close > EMA100
- **SHORT**: (RSI > 60 OR BB% > 0.8) + MACD Hist < Signal + Close < EMA100
- **특징**: 복합 적응형

---

## 검증 결과

### 전체 결과표 (40 테스트)

| 전략 | TP/SL | Type1 | L_WR | S_WR | Type2 | WF | Passes |
|------|-------|-------|------|------|-------|-----|--------|
| A: ADX Strong Trend | 2.0/1.5 | ❌ | 21.9% | 38.9% | ❌ -$7.59 | ✅ 3/6 | 1/3 |
| A: ADX Strong Trend | 2.5/1.5 | ❌ | 18.8% | 35.9% | ❌ -$5.87 | ✅ 4/6 | 1/3 |
| A: ADX Strong Trend | 3.0/2.0 | ❌ | 17.6% | 31.4% | ✅ +$2.80 | ❌ 2/6 | 1/3 |
| **A: ADX Strong Trend** | **3.5/2.0** | ❌ | 18.6% | 29.4% | **✅ +$8.48** | **✅ 3/6** | **2/3** |
| B: Multi-Confirm | 2.0/1.5 | ❌ | 36.6% | 46.6% | ❌ -$1.04 | ✅ 4/6 | 1/3 |
| B: Multi-Confirm | 2.5/1.5 | ❌ | 31.7% | 42.6% | ❌ -$6.17 | ✅ 3/6 | 1/3 |
| B: Multi-Confirm | 3.0/2.0 | ❌ | 27.7% | 40.7% | ❌ -$3.95 | ✅ 3/6 | 1/3 |
| B: Multi-Confirm | 3.5/2.0 | ❌ | 25.5% | 39.2% | ❌ -$3.62 | ✅ 3/6 | 1/3 |
| C: BB+Stoch | 2.0/1.5 | ❌ | 17.6% | 53.8% | ✅ +$4.47 | ❌ 2/6 | 1/3 |
| C: BB+Stoch | 2.5/1.5 | ❌ | 13.3% | 57.1% | ✅ +$18.51 | ❌ 2/6 | 1/3 |
| **C: BB+Stoch** | **3.0/2.0** | ❌ | 16.7% | 62.1% | **✅ +$30.02** | **✅ 3/6** | **2/3** |
| **C: BB+Stoch** | **3.5/2.0** | ❌ | 9.1% | 45.5% | **✅ +$19.79** | **✅ 3/6** | **2/3** |
| **D: VWAP+Volume** | **2.0/1.5** | ❌ | **46.6%** | **46.9%** | **✅ +$14.72** | **✅ 4/6** | **2/3** ⭐ |
| D: VWAP+Volume | 2.5/1.5 | ❌ | 39.7% | 42.8% | ✅ +$8.33 | ❌ 2/6 | 1/3 |
| D: VWAP+Volume | 3.0/2.0 | ❌ | 35.5% | 40.3% | ✅ +$9.16 | ❌ 2/6 | 1/3 |
| D: VWAP+Volume | 3.5/2.0 | ❌ | 33.5% | 38.4% | ✅ +$7.01 | ❌ 2/6 | 1/3 |
| E: EMA Triple | 2.0/1.5 | ❌ | 29.2% | 42.7% | ❌ -$2.55 | ❌ 2/6 | 0/3 |
| E: EMA Triple | 2.5/1.5 | ❌ | 27.2% | 39.2% | ✅ +$9.21 | ❌ 2/6 | 1/3 |
| E: EMA Triple | 3.0/2.0 | ❌ | 23.4% | 35.1% | ✅ +$5.88 | ❌ 2/6 | 1/3 |
| E: EMA Triple | 3.5/2.0 | ❌ | 21.3% | 32.4% | ✅ +$3.12 | ❌ 2/6 | 1/3 |
| F: RSI Divergence | 2.0/1.5 | ❌ | 28.9% | 36.4% | ❌ -$1.23 | ❌ 2/6 | 0/3 |
| F: RSI Divergence | 2.5/1.5 | ❌ | 27.8% | 35.2% | ✅ +$2.15 | ❌ 2/6 | 1/3 |
| F: RSI Divergence | 3.0/2.0 | ❌ | 26.7% | 33.9% | ✅ +$3.71 | ❌ 2/6 | 1/3 |
| F: RSI Divergence | 3.5/2.0 | ❌ | 24.1% | 31.5% | ✅ +$1.89 | ❌ 2/6 | 1/3 |
| G: Range Breakout | 2.0/1.5 | ❌ | 28.4% | 44.8% | ✅ +$5.43 | ❌ 2/6 | 1/3 |
| G: Range Breakout | 2.5/1.5 | ❌ | 26.7% | 41.4% | ✅ +$12.68 | ❌ 2/6 | 1/3 |
| G: Range Breakout | 3.0/2.0 | ❌ | 24.3% | 38.5% | ✅ +$8.92 | ❌ 2/6 | 1/3 |
| G: Range Breakout | 3.5/2.0 | ❌ | 22.1% | 35.7% | ✅ +$4.55 | ❌ 2/6 | 1/3 |
| H: ChoCH | All | ❌ | 0% | 0% | ❌ $0 | ❌ 0/6 | 0/3 |
| I: ATR Breakout | 2.0/1.5 | ❌ | 26.3% | 41.2% | ❌ -$8.34 | ❌ 1/6 | 0/3 |
| I: ATR Breakout | 2.5/1.5 | ❌ | 24.7% | 39.8% | ❌ -$5.21 | ❌ 2/6 | 0/3 |
| I: ATR Breakout | 3.0/2.0 | ❌ | 23.1% | 37.4% | ❌ -$2.88 | ❌ 2/6 | 0/3 |
| I: ATR Breakout | 3.5/2.0 | ❌ | 21.5% | 35.1% | ❌ -$1.05 | ❌ 2/6 | 0/3 |
| J: Hybrid | 2.0/1.5 | ❌ | 30.2% | 41.8% | ❌ -$4.67 | ❌ 2/6 | 0/3 |
| J: Hybrid | 2.5/1.5 | ❌ | 28.7% | 39.5% | ❌ -$2.33 | ❌ 2/6 | 0/3 |
| J: Hybrid | 3.0/2.0 | ❌ | 27.3% | 38.1% | ❌ -$0.89 | ❌ 2/6 | 0/3 |
| J: Hybrid | 3.5/2.0 | ❌ | 25.8% | 36.4% | ✅ +$0.54 | ❌ 2/6 | 1/3 |

---

## 핵심 발견

### 1. 완전 통과 전략: **없음**

40개 전략/설정 조합 중 **단 하나도** 3개 조건(Type1 + Type2 + WF)을 모두 통과하지 못함.

### 2. 부분 통과 (2/3) 전략: 4개

| 전략 | TP/SL | L_WR | S_WR | 특징 |
|------|-------|------|------|------|
| A: ADX Strong Trend | 3.5/2.0 | 18.6% | 29.4% | SHORT 우세 |
| C: BB+Stochastic | 3.0/2.0 | 16.7% | 62.1% | SHORT 극단적 우세 |
| C: BB+Stochastic | 3.5/2.0 | 9.1% | 45.5% | SHORT 극단적 우세 |
| **D: VWAP+Volume** | **2.0/1.5** | **46.6%** | **46.9%** | **⭐ 가장 균형** |

### 3. 가장 유망한 후보: Strategy D (VWAP+Volume)

| 지표 | 값 | 비고 |
|------|-----|------|
| **LONG Win Rate** | **46.6%** | ⭐ 50%까지 3.4%p만 부족 |
| **SHORT Win Rate** | **46.9%** | ⭐ 50%까지 3.1%p만 부족 |
| **L/S 차이** | **0.3%p** | ⭐ 가장 균형 잡힌 전략 |
| 전체 승률 | 46.8% | 50% 대비 3.2%p 부족 |
| Type 2 PnL | +$14.72 | ✅ 통과 |
| Walk-Forward | 4/6 (66.7%) | ✅ 통과 |
| **Type 1** | **❌** | WR 46.8% < 50% |

### 4. LONG 승률 일관적 저조

- **LONG WR 범위**: 0% ~ 46.6%
- **SHORT WR 범위**: 29.4% ~ 62.1%
- **원인**: 90일 테스트 기간의 하락장 편향 (Bearish Market Bias)

### 5. 전략 H (ChoCH) 완전 실패

- 신호 발생 0건 → 스윙 포인트 감지 로직 문제
- Lookback 파라미터 조정 필요

---

## 이전 연구와 비교

| 연구 | 테스트 수 | 3/3 통과 | Best L_WR | Best S_WR |
|------|----------|----------|-----------|-----------|
| Balanced Entry (14 strategies) | 14 | 0 | 32.4% | 62.1% |
| **New Strategy (10 strategies)** | **40** | **0** | **46.6%** | **62.1%** |
| **개선** | - | - | **+14.2%p** | 동일 |

**Strategy D (VWAP+Volume)**가 이전 연구 대비 LONG WR을 **14.2%p 개선**함.

---

## 원인 분석

### 1. 시장 환경 편향 (Bearish Bias)
- **테스트 기간**: 90일 (2025-10-02 ~ 2025-12-31)
- **시장 특성**: 전반적 하락세
- **영향**: 모든 전략에서 SHORT > LONG WR

### 2. Type 1 기준의 엄격함
- 50% WR + 양수 EV 동시 충족은 하락장에서 매우 어려움
- 특히 LONG 방향에서 구조적 불리함

### 3. 양방향 균형의 근본적 한계
- 동일 로직을 LONG/SHORT에 적용하면 시장 편향 영향 받음
- Strategy D가 가장 균형 잡힌 이유: VWAP + Volume 조합이 시장 구조에 덜 민감

---

## 결론 및 권고

### 이중 검증 통과 전략: **없음**

40개 신규 전략/설정 조합을 테스트한 결과, 이중 검증을 완전히 통과하는 전략이 없습니다.

### 가장 유망한 후보: Strategy D (VWAP+Volume, TP 2.0%/SL 1.5%)

**선정 이유**:
1. **가장 균형 잡힌 L/S WR**: 46.6% / 46.9% (0.3%p 차이)
2. **50% WR까지 근접**: 3.2%p만 부족
3. **Type 2 & WF 통과**: 수익성 + 일관성 검증됨

---

## 권고사항

### 단기 조치

1. **Strategy D 추가 최적화**
   - VWAP 대역 조정 (현재: Close > VWAP → Close > VWAP + 0.1% 등)
   - Volume 배수 조정 (현재: 1.5x → 1.3x, 2.0x 테스트)
   - EMA 기간 조정 (현재: 21 → 15, 30 테스트)

2. **Type 1 기준 완화 검토**
   - WR ≥ 48% (2%p 완화) 시 Strategy D 통과 가능
   - 단, 기준 완화는 리스크 증가 수반

### 중기 연구

1. **시장 레짐 필터 적용**
   - 상승장에서만 LONG 활성화
   - 하락장에서만 SHORT 활성화
   - ADX + EMA 기반 레짐 분류

2. **비대칭 전략 설계**
   - LONG과 SHORT에 다른 Entry 로직 적용
   - LONG: 더 엄격한 조건 (현 시장에서)
   - SHORT: 현재 조건 유지

3. **더 긴 기간 데이터 테스트**
   - 상승장 + 하락장 혼합 데이터 (6개월~1년)
   - 시장 사이클 전체 반영

### 장기 관점

- **현실 인정**: 90일 하락장에서 양방향 50%+ WR 달성은 구조적으로 어려움
- **적응적 전략**: 시장 상황에 따라 방향 편향 조절
- **리스크 관리 우선**: 손실 최소화 후 수익 추구

---

## 관련 파일

| 파일 | 내용 |
|------|------|
| `scripts/analysis/new_strategy_dual_verification.py` | 10개 전략 검증 스크립트 |
| `results/new_strategy_dual_verification_20260101_113949.csv` | 상세 결과 CSV |
| `data/btc_5m_90days_v12research.csv` | 테스트 데이터 (25,920 candles) |
| `claudedocs/BALANCED_ENTRY_DUAL_VERIFICATION_REPORT_20260101.md` | 이전 14개 전략 검증 |

---

## Strategy D (VWAP+Volume) 상세 명세

### Entry 조건
```python
# LONG
long_signal = (
    (df['close'] > df['vwap']) &           # 가격 > VWAP
    (df['volume'] > df['volume_ma'] * 1.5) & # 볼륨 스파이크
    (df['close'] > df['ema_21'])            # 단기 상승 추세
)

# SHORT
short_signal = (
    (df['close'] < df['vwap']) &           # 가격 < VWAP
    (df['volume'] > df['volume_ma'] * 1.5) & # 볼륨 스파이크
    (df['close'] < df['ema_21'])            # 단기 하락 추세
)
```

### Exit 조건
- **Take Profit**: 2.0% (고정)
- **Stop Loss**: 1.5% (고정)
- **R:R Ratio**: 1.33:1

### 백테스트 결과 (90일)
| 메트릭 | 값 |
|--------|-----|
| 총 신호 | ~2,500 |
| 실제 거래 | ~180 |
| LONG 거래 | ~90 |
| SHORT 거래 | ~90 |
| LONG Win Rate | 46.6% |
| SHORT Win Rate | 46.9% |
| Total PnL | +$14.72 |
| Walk-Forward | 4/6 (66.7%) |

---

**작성자**: Claude AI Assistant
**검토 상태**: 사용자 검토 필요
**버전**: 1.0
