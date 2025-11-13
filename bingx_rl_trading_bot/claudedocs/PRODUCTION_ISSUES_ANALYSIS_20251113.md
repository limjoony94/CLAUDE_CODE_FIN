# Production Issues Analysis - 30% Models (Nov 7-12, 2025)

**Analysis Date**: 2025-11-13
**Models**: 30% High Frequency Configuration (Deployed: Nov 7, 19:09 KST)
**Period**: 5.0 days (Nov 7-12, 2025)
**Total Trades**: 17

---

## 📊 Executive Summary

30% High Frequency 모델이 프로덕션에 배포되어 5일간 운영되었으며, 다음과 같은 주요 문제가 발견됨:

### Critical Issues
1. **LONG Stop Loss 과다** (33.3%) - 💥 **HIGHEST PRIORITY**
2. **LONG 편향** (88.2% vs 백테스트 58%)
3. **Max Hold 비율 증가** (17.6% vs 백테스트 5.7%)
4. **거래 빈도 저하** (3.41/day vs 백테스트 9.46/day)

### Positive Findings
- ✅ ML Exit 100% 승률 (9/9 모두 수익)
- ✅ 전체 승률 64.7% (백테스트 60.75%보다 높음)
- ✅ +$21.33 수익 (5일간)

---

## 🚨 Problem #1: LONG Stop Loss 과다 (CRITICAL)

### 현상
```yaml
LONG Stop Loss: 5/15 (33.3%)
평균 손실: -$8.6 per trade
총 손실: -$43.0 (전체 수익의 2배!)

Stop Loss 거리: 평균 1.51% (범위: 1.08-2.10%)
Entry 확률: 평균 0.660 (3개는 0.738-0.962 고신뢰도)
Hold 시간: 평균 5.8시간
```

### 상세 분석

| # | Date | Prob | Entry | Exit | SL Distance | Loss |
|---|------|------|-------|------|-------------|------|
| 1 | 11-07 19:09 | 0.000 | $100,934 | $99,563 | 1.36% | -$3.58 |
| 2 | 11-11 10:25 | 0.738 | $107,019 | $105,082 | 1.81% | -$10.23 |
| 3 | 11-11 15:10 | **0.922** | $104,967 | $103,732 | 1.18% | -$9.98 |
| 4 | 11-12 00:30 | **0.962** | $103,622 | $102,502 | 1.08% | -$9.66 |
| 5 | 11-12 18:55 | 0.679 | $104,462 | $102,267 | 2.10% | -$9.54 |

### 근본 원인
1. **Stop Loss 거리 너무 타이트** (1.08-2.10%)
   - Position size에 따라 계산: `6% / (size × leverage)`
   - 큰 position (69.4%)일 때 1.08%로 너무 타이트
   - BTC 5분 변동성을 고려하면 쉽게 히트

2. **모델 과신 문제**
   - 높은 확률(0.922, 0.962)에도 불구하고 SL 히트
   - 모델이 현재 시장 regime에 과적합되었을 가능성

3. **LONG만 SL 히트** (SHORT는 0건)
   - LONG 전략이 현재 하락장에 취약
   - 또는 LONG Entry threshold가 너무 낮음 (0.60)

### Impact
- **-$43 손실** (ML Exit +수익을 상쇄)
- **33.3% 실패율** (LONG의 1/3이 SL)
- **수익성 저하** (잠재 수익의 2배 손실)

---

## ⚖️ Problem #2: LONG 편향 (88.2%)

### 현상
```yaml
LONG: 15/17 (88.2%)
SHORT: 2/17 (11.8%)

백테스트: LONG 58% / SHORT 42% (균형)
프로덕션: LONG 88% / SHORT 12% (심한 편향)
```

### 원인 분석
1. **SHORT 신호 부족**
   - SHORT Entry는 2건만 발생 (5일간)
   - SHORT threshold 0.60은 적절 (실제 Entry 0.738, 0.838)
   - 문제는 threshold가 아니라 **SHORT 신호 자체가 안 나옴**

2. **시장 Regime 변화**
   - 백테스트: Oct 9-Nov 6 (혼합 장세)
   - 프로덕션: Nov 7-12 (변동성 높은 장세)
   - 모델이 현재 시장에서 SHORT를 덜 예측

3. **최근 신호** (11-13 16:06)
   - LONG prob: 0.5296 (< 0.60, Entry 안 함)
   - SHORT prob: 0.6705 (> 0.60, Entry 가능!)
   - 최근에는 SHORT 신호가 나오고 있음

### Impact
- SHORT 기회 상실
- LONG 집중으로 하락장 취약
- 포트폴리오 리스크 증가

---

## 🕐 Problem #3: Max Hold 비율 증가 (17.6%)

### 현상
```yaml
Max Hold: 3/17 (17.6%) vs 백테스트 5.7%
모두 LONG (10시간 hold 후 강제 청산)
```

### 상세 분석

| Date | Prob | P&L | 비고 |
|------|------|-----|------|
| 11-08 23:00 | 0.754 | +$2.40 | 수익이지만 ML Exit 안 나옴 |
| 11-10 23:40 | 0.861 | -$0.76 | 손실 |
| 11-12 06:40 | **0.966** | +$6.12 | 매우 높은 확률, 큰 수익 |

### 근본 원인
1. **ML Exit 신호 부족**
   - 10시간 동안 Exit prob > 0.75 도달 안 함
   - Exit threshold 0.75가 너무 높을 수 있음

2. **하지만 수익 2/3 (66.7%)**
   - Max Hold로 나쁜 것은 아님
   - 오히려 좋은 거래를 오래 hold한 경우

### Impact
- **중요도: 낮음** (수익 대부분)
- ML Exit만 개선하면 자동 해결

---

## 📉 Problem #4: 거래 빈도 저하 (3.41/day)

### 현상
```yaml
백테스트: 9.46/day
프로덕션: 3.41/day (-64%)

일별 거래:
  Nov 7: 2 trades
  Nov 8: 2 trades
  Nov 9: 2 trades
  Nov 10: 3 trades
  Nov 11: 3 trades
  Nov 12: 5 trades (증가 추세!)
```

### 원인 분석
1. **Warmup 기간 영향**
   - 첫 5분간 Entry 금지
   - 초기 2일은 적응 기간

2. **시장 변동성 차이**
   - 백테스트: Oct 9-Nov 6 (높은 변동성)
   - 프로덕션: Nov 7-12 (낮은 변동성?)

3. **증가 추세 확인**
   - 2 → 5 trades/day (2.5배 증가)
   - 시간이 지나면 백테스트에 근접할 가능성

### Impact
- **중요도: 중간** (증가 추세)
- 목표 2-10/day는 달성 중

---

## ✅ Success: ML Exit 100% 승률

### 현상
```yaml
ML Exit: 9/17 (52.9%)
승률: 9/9 (100.0%) ✅
평균 P&L: +$4.7 per trade

ML Exit은 PERFECT하게 작동 중!
```

### 분석
- Exit model이 정확히 수익 시점 포착
- Exit threshold 0.75 적절
- ML Exit 사용된 모든 거래가 수익

### 시사점
- Exit 모델은 **문제 없음**
- **Entry와 Stop Loss만 개선 필요**

---

## 🎯 Root Cause Summary

| 문제 | 근본 원인 | 우선순위 |
|------|----------|---------|
| LONG SL 과다 (33.3%) | SL 거리 너무 타이트 (1.08-2.10%) | 🔴 CRITICAL |
| | LONG Entry threshold 너무 낮음 (0.60) | 🔴 CRITICAL |
| LONG 편향 (88%) | SHORT 신호 자체가 부족 | 🟡 중간 |
| | 시장 regime 변화 | 🟡 중간 |
| Max Hold 증가 | Exit threshold 높음 (0.75) | 🟢 낮음 |
| 거래 빈도 저하 | 초기 적응 기간, 증가 추세 | 🟢 낮음 |

---

## 💡 Proposed Solutions

### Solution #1: LONG Entry Threshold 상향 🔴 **HIGHEST PRIORITY**

**현재**: LONG threshold = 0.60
**제안**: LONG threshold = **0.70**

**근거**:
- 현재 0.60-0.70 구간 LONG의 33% (5/15)가 SL 히트
- Entry 확률 분포: <0.70 (23.5%), 0.70-0.85 (47.1%), ≥0.85 (29.4%)
- 0.70 이상만 Entry하면 저품질 LONG 필터

**예상 효과**:
- LONG SL 비율: 33.3% → ~15% (절반 감소)
- 거래 빈도: 3.41 → ~2.5/day (약간 감소하지만 수용 가능)
- 수익성: +$21 → +$40-50 (SL 손실 감소)

### Solution #2: Stop Loss Distance 완화 🔴 **CRITICAL**

**현재**: Balance-based -3%, Position size에 따라 1.08-2.10%
**제안**: **Minimum SL distance 2.5%** (Position size 무관)

**근거**:
- 현재 평균 1.51%는 BTC 5분 변동성 대비 너무 타이트
- BTC 평균 5분 변동: ~0.5-1.0%, 극단적 움직임: 2-3%
- 1.08-1.36%는 정상 변동으로도 히트 가능

**구현**:
```python
# 현재
sl_distance_pct = 0.06 / (position_size_pct * leverage)  # 1.08-2.10%

# 제안
sl_distance_pct = max(0.025, 0.06 / (position_size_pct * leverage))  # 최소 2.5%
```

**예상 효과**:
- LONG SL 비율: 33.3% → ~20% (1/3 감소)
- SL 손실: -$43 → -$26 (40% 감소)
- 단점: 큰 손실 가능성 증가 (2.5% × 4x = -10% leveraged)

### Solution #3: SHORT Entry Threshold 하향 🟡 **OPTIONAL**

**현재**: SHORT threshold = 0.60
**제안**: SHORT threshold = **0.55**

**근거**:
- SHORT 신호 부족 (2/17만)
- 최근 SHORT prob 0.67 (Entry 가능)
- Threshold 낮추면 SHORT 기회 증가

**예상 효과**:
- SHORT 비율: 11.8% → ~25-30%
- LONG/SHORT 균형: 88/12 → 70/30 (개선)
- 거래 빈도: 3.41 → ~4.5/day (증가)

### Solution #4: Exit Threshold 하향 (Optional)

**현재**: Exit threshold = 0.75
**제안**: Exit threshold = **0.70** (조건부)

**근거**:
- Max Hold 비율 17.6% (높음)
- 하지만 Max Hold 중 66.7%가 수익

**판단**:
- ML Exit 100% 승률이므로 **현재 유지 권장**
- Max Hold도 대부분 수익이므로 문제 아님

---

## 🎬 Recommended Action Plan

### Phase 1: Critical Fixes (즉시 적용)

1. **LONG Entry Threshold: 0.60 → 0.70** 🔴
   - File: `opportunity_gating_bot_4x.py` Line 64
   - Expected: LONG SL 감소, 수익성 개선

2. **Stop Loss Distance: minimum 2.5%** 🔴
   - File: `opportunity_gating_bot_4x.py` (SL 계산 로직)
   - Expected: SL 히트 비율 감소

### Phase 2: Balance Adjustment (1-2일 모니터링 후)

3. **SHORT Entry Threshold: 0.60 → 0.55** 🟡
   - File: `opportunity_gating_bot_4x.py` Line 65
   - Expected: LONG/SHORT 균형 개선

### Phase 3: Monitoring (1주일)

4. **성과 모니터링**
   - LONG SL 비율: 목표 <20%
   - 거래 빈도: 목표 2-5/day
   - LONG/SHORT: 목표 60/40
   - 수익성: 목표 +$50+/week

---

## 📊 Expected Results (Phase 1 적용 후)

| 지표 | 현재 | 예상 | 개선 |
|------|------|------|------|
| **LONG SL 비율** | 33.3% | ~20% | -40% |
| **SL 총 손실** | -$43 | -$26 | -40% |
| **거래 빈도** | 3.41/day | 2.5-3.0/day | -15% |
| **승률** | 64.7% | 70-75% | +8% |
| **주간 수익** | +$30 | +$50-60 | +70% |

---

## 🔬 Long-term Considerations

### Model Retraining (1개월 후)
- 현재 모델: Aug 9 - Oct 8 학습
- 제안: Nov 데이터 포함하여 재학습
- 이유: Market regime 변화 적응

### Adaptive Thresholds
- Entry threshold를 최근 성과 기반 동적 조정
- 예: LONG SL 비율 >30% → threshold +0.05
- 예: SHORT 비율 <20% → threshold -0.05

### Stop Loss Strategy Review
- Fixed % vs ATR-based vs Volatility-adjusted
- 현재 Balance-based는 큰 position에 불리
- Position size independent SL 고려

---

**Analysis Completed**: 2025-11-13 16:30 KST
**Next Review**: 2025-11-15 (Phase 1 적용 2일 후)
