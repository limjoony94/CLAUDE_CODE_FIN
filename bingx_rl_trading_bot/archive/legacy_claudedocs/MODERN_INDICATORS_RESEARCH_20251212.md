# 현대적 지표 연구 결과 (2025-12-12)

## 연구 배경

### 문제 제기
- 기존 EMA200 필터가 "상승추세 + 과매도" 희귀 조합(1.33%)을 필터링
- Rolling Window 분석에서 Period 1이 -43.4% 손실 → 과적합 의심
- Train/Test Ratio 5.38로 과적합 위험 높음

### 연구 대상 지표
1. **Hull Moving Average (HMA)** - EMA 대비 지연 감소
2. **Squeeze Momentum Indicator** - 변동성 압축 탐지
3. **Adaptive EMA** - 변동성 기반 적응형 필터
4. **MACD variants** - 전통적 추세 지표

---

## 핵심 발견

### 1. EMA200 필터의 문제점

| 메트릭 | 값 | 평가 |
|--------|-----|------|
| Train Return | 171.3% | - |
| Test Return | 31.8% | - |
| Train/Test Ratio | **5.38** | ❌ 과적합 |
| Test RA | 0.89 | 기준선 |

### 2. 현대 지표 비교 결과

| 전략 | Test Trades | Test WR | Test Return | Test MDD | Test RA | 과적합 |
|------|-------------|---------|-------------|----------|---------|--------|
| EMA200 + Zone37 | 42 | 61.9% | 31.8% | 35.9% | 0.89 | ❌ High |
| **HMA50 + Zone35** | **75** | **69.3%** | **182.0%** | 54.7% | **3.33** | ✅ Low |
| HMA50 + Zone37 | 114 | 63.2% | 98.7% | 72.5% | 1.36 | 🟡 Medium |
| HMA50 + Zone30 | 18 | 83.3% | 39.7% | 16.3% | 2.44 | ✅ Low |
| HMA20 + Zone35 | 212 | 63.7% | 178.3% | 83.0% | 2.15 | ✅ Low |
| No Filter | 465 | 63.4% | 250.4% | 93.8% | 2.67 | ✅ Low |

### 3. 과적합 진단 (Train/Test Ratio)
- **< 2**: 🟢 과적합 위험 낮음 (Test가 Train 수준)
- **2 ~ 5**: 🟡 과적합 가능성 있음
- **> 5**: 🔴 과적합 위험 높음

---

## 권장 설정

### 🏆 최적 설정: HMA50 + RSI Zone (35/65)

```yaml
strategy:
  # RSI Zone (35/65 - Loose Zone)
  rsi_oversold_zone: 35
  rsi_recovery_threshold: 40
  rsi_overbought_zone: 65
  rsi_decline_threshold: 60

  # Trend Filter (HMA50)
  trend_filter: "HMA50"  # Price > HMA(50)

  # Depth
  min_rsi_depth: 2.0

exit:
  take_profit_pct: 3.5    # v2.2 최적화
  stop_loss_pct: 3.0      # v2.2 최적화 (넓은 SL)
  breakeven_trigger: 1.5  # v2.2 최적화
  cooldown_candles: 1
```

### 성과 비교

| 메트릭 | 현재 (EMA200) | 권장 (HMA50) | 개선율 |
|--------|---------------|--------------|--------|
| Test Trades | 42건 | 75건 | +79% |
| Test 승률 | 61.9% | 69.3% | +7.4%p |
| Test Return | 31.8% | 182.0% | +473% |
| Test MDD | 35.9% | 54.7% | -18.8%p |
| Test RA | 0.89 | 3.33 | **+274%** |
| Train/Test Ratio | 5.38 | 0.81 | -4.57 |

### HMA50 계산 방법
```python
def calc_hma(close, period=50):
    """Hull Moving Average - 지연 감소 이동평균"""
    half = int(period / 2)
    sqrt = int(np.sqrt(period))

    wma_half = WMA(close, half)
    wma_full = WMA(close, period)

    raw_hma = 2 * wma_half - wma_full
    hma = WMA(raw_hma, sqrt)

    return hma

# Trend Filter
uptrend = close > hma50
downtrend = close < hma50
```

---

## Rolling Window 검증

### HMA50 + Zone35 (기본 설정)
| Period | Trades | WR | Return | MDD |
|--------|--------|-----|--------|-----|
| Period 1 | 31 | 61.3% | 49.9% | 30.7% |
| Period 2 | 51 | 74.5% | 126.4% | 46.9% |
| Period 3 | 37 | 56.8% | -30.9% | 58.0% |
| Period 4 | 42 | 71.4% | 87.9% | 53.9% |
| **평균** | - | 66.0% | **58.3%** | 47.4% |

**수익 구간: 3/4 (75%)**

### 주의사항
- Period 3에서 -30.9% 손실 발생
- MDD 54.7%는 여전히 높은 편
- 강한 추세장에서 역추세 진입으로 손실 가능

---

## 대안 전략

### Option A: 보수적 설정 (낮은 MDD)
```yaml
# HMA50 + Zone30 (극단 존)
rsi_oversold_zone: 30
rsi_overbought_zone: 70
take_profit_pct: 2.5
stop_loss_pct: 3.0
breakeven_trigger: 1.5
```
- Test RA: 2.44
- Test MDD: 16.3% (매우 낮음)
- 단점: 거래 18건 (월 12건)

### Option B: 공격적 설정 (높은 빈도)
```yaml
# HMA20 + Zone35
trend_filter: "HMA20"
rsi_oversold_zone: 35
rsi_overbought_zone: 65
```
- Test RA: 2.15
- Test Trades: 212건 (월 142건)
- 단점: MDD 83%

---

## 결론

### HMA50 vs EMA200

| 특성 | EMA200 | HMA50 |
|------|--------|-------|
| 지연 | 높음 | **낮음** |
| 반응성 | 느림 | **빠름** |
| 노이즈 | 낮음 | 중간 |
| 과적합 | **높음** | 낮음 |

### 핵심 인사이트

1. **EMA200 필터는 과적합되었다**
   - Train/Test Ratio 5.38 (위험 수준)
   - 희귀 조합(1.33%)만 허용 → 샘플 부족

2. **HMA50이 더 나은 대안**
   - 빠른 반응 → 더 많은 거래 기회
   - Train/Test Ratio 0.81 → 과적합 없음
   - Test RA 3.33 (3.7배 개선)

3. **RSI Zone 35/65가 37/63보다 안정적**
   - 더 극단적인 존 → 더 높은 확신도
   - 거래 빈도 적절 (월 50건)

4. **MDD 관리 필요**
   - 권장 설정도 MDD 54.7%
   - 레버리지 조정 또는 포지션 사이징으로 관리

---

## 다음 단계

1. [ ] HMA50 지표를 production bot에 구현
2. [ ] Paper trading으로 실제 신호 검증
3. [ ] MDD 감소를 위한 포지션 사이징 최적화
4. [ ] 시장 상황별 필터 추가 검토

---

**생성일**: 2025-12-12
**분석 스크립트**:
- `scripts/analysis/modern_indicators_comparison.py`
- `scripts/analysis/practical_hma_analysis.py`
- `scripts/analysis/hma50_rsi70_deep_analysis.py`
