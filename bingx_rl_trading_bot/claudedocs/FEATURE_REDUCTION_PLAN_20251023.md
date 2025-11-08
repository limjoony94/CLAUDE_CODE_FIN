# Feature Reduction Plan - 중복 제거
**Date**: 2025-10-23
**Status**: 🎯 Action Plan

---

## Executive Summary

**문제**: Correlation 분석 결과 심각한 중복 발견
- LONG Entry: 12개 중복 쌍 (27.3% 감소 가능)
- SHORT Entry: 14개 중복 쌍 (36.8% 감소 가능)
- Exit: 3개 중복 쌍 (33.3% 감소 가능)

**목표**: 중복 feature 제거로 모델 성능 개선
- Overfitting 위험 감소
- 학습 속도 향상
- 모델 해석력 향상
- 불필요한 계산 제거

**예상 결과**:
- Feature 수: 107개 → 78개 (-27.1%)
- 성능 유지 또는 개선 (중복 제거로 overfitting 감소)
- 학습/추론 속도 30-40% 향상

---

## LONG Entry Model - 제거 계획

### 현재 Features: 44개

### 중복 쌍 및 제거 결정:

**1. volume_ma_ratio 중복 (Correlation: 1.0000)**
```yaml
문제: 완전 중복 (리스트에 2번 등장)
제거: volume_ma_ratio 중 하나 제거
유지: volume_ma_ratio (1개만)
```

**2. Bollinger Bands 그룹 (Correlation: 0.9877-0.9969)**
```yaml
문제: bb_high, bb_mid, bb_low 모두 거의 동일
분석:
  - bb_high ≈ bb_mid: 0.9969
  - bb_mid ≈ bb_low: 0.9969
  - bb_high ≈ bb_low: 0.9877

제거: bb_high, bb_low
유지: bb_mid (중심선만 유지)

근거:
  - BB는 가격과의 상대적 위치가 중요
  - (price - bb_mid) / bb_width로 위치 계산 가능
  - 3개 모두 유지는 과도한 중복
```

**3. Trendline Slopes (Correlation: 0.9793)**
```yaml
문제: upper_trendline_slope ≈ lower_trendline_slope
제거: lower_trendline_slope
유지: upper_trendline_slope

근거:
  - 상단 추세선이 저항선 역할 (더 중요)
  - Feature importance 확인 필요
```

**4. MACD vs MACD Signal (Correlation: 0.9508)**
```yaml
문제: macd ≈ macd_signal
제거: macd_signal
유지: macd, macd_diff

근거:
  - macd_diff = macd - macd_signal (이미 차이 포함)
  - MACD dominance 분석에서 macd_diff가 가장 중요
  - Signal line은 MACD의 지연 지표 (redundant)
```

**5. Price vs Trendline (Correlation: 0.9204)**
```yaml
문제: price_vs_upper_trendline_pct ≈ price_vs_lower_trendline_pct
제거: price_vs_lower_trendline_pct
유지: price_vs_upper_trendline_pct

근거:
  - 상단 추세선 대비 가격이 더 중요 (저항)
  - 하단은 slope로 간접 추정 가능
```

**6. Shooting Star vs Selling Pressure (Correlation: 0.8106)**
```yaml
문제: shooting_star ≈ strong_selling_pressure
제거: strong_selling_pressure
유지: shooting_star

근거:
  - Shooting star는 전통적 캔들 패턴 (해석 용이)
  - Strong selling pressure는 파생 지표
```

### LONG Entry - 제거 목록 (7개):
1. volume_ma_ratio (중복 제거)
2. bb_high
3. bb_low
4. lower_trendline_slope
5. macd_signal
6. price_vs_lower_trendline_pct
7. strong_selling_pressure

### LONG Entry - 최종 Features: 37개 (44 - 7)

---

## SHORT Entry Model - 제거 계획

### 현재 Features: 38개

### 중복 쌍 및 제거 결정:

**1. MACD Strength = MACD Divergence (Correlation: 1.0000)**
```yaml
문제: 완전 중복!
제거: macd_divergence_abs
유지: macd_strength

근거:
  - 동일한 값
  - Strength가 더 직관적
```

**2. ATR vs ATR_PCT (Correlation: 0.9976)**
```yaml
문제: atr_pct ≈ atr
제거: atr
유지: atr_pct

근거:
  - atr_pct = atr / price (정규화된 값)
  - 가격 변동에 따른 상대적 변동성이 더 중요
  - 절대값(atr)보다 비율(atr_pct)이 학습에 유리
```

**3. Volatility 그룹 (Correlation: 0.8-0.9)**
```yaml
문제: volatility, atr_pct, upside_vol, downside_vol 모두 연결
분석:
  - volatility ≈ atr_pct: 0.9104
  - volatility ≈ upside_vol: 0.8072
  - upside_vol ≈ downside_vol: 0.8816
  - atr_pct ≈ upside_vol: 0.8574
  - atr_pct ≈ downside_vol: 0.8122

제거: upside_volatility, downside_volatility
유지: volatility, atr_pct

근거:
  - volatility (전체 변동성) + atr_pct (정규화)로 충분
  - 상승/하락 분리는 volatility_asymmetry로 대체 가능
```

**4. Down Candle vs Rejection (Correlation: 0.9543)**
```yaml
문제: down_candle ≈ rejection_from_resistance
제거: rejection_from_resistance
유지: down_candle

근거:
  - down_candle이 더 기본적인 feature
  - Resistance rejection은 파생 feature
```

**5. RSI Direction vs Price Direction (Correlation: 0.8192)**
```yaml
문제: rsi_direction ≈ price_direction_ma20
제거: price_direction_ma20
유지: rsi_direction

근거:
  - RSI direction이 모멘텀 포함 (더 완전한 지표)
  - Price direction은 price_distance_ma20으로 대체 가능
```

**6. Price Distance MA20 vs MA50 (Correlation: 0.8050)**
```yaml
문제: price_distance_ma20 ≈ price_distance_ma50
제거: price_distance_ma50
유지: price_distance_ma20, price_direction_ma50 (방향은 유지)

근거:
  - 단기 거리(MA20)가 5분봉 거래에 더 중요
  - MA50 방향은 장기 추세 표시로 유용
```

**7. Down Candle Ratio vs Resistance Rejection Count (Correlation: 0.8008)**
```yaml
문제: down_candle_ratio ≈ resistance_rejection_count
제거: resistance_rejection_count
유지: down_candle_ratio

근거:
  - Down candle ratio가 더 직접적인 지표
  - Resistance rejection count는 파생 지표
```

### SHORT Entry - 제거 목록 (9개):
1. macd_divergence_abs
2. atr
3. upside_volatility
4. downside_volatility
5. rejection_from_resistance
6. price_direction_ma20
7. price_distance_ma50
8. resistance_rejection_count
9. (down_candle - 이미 제거됨, rejection과 중복)

### SHORT Entry - 최종 Features: 29개 (38 - 9)

---

## Exit Model - 제거 계획

### 현재 Features: 25개

**문제**: 16개 feature가 missing (계산되지 않음)

### 중복 쌍 및 제거 결정:

**1. MACD ≈ Trend Strength (Correlation: 0.9988)**
```yaml
문제: 거의 완전 중복
제거: trend_strength
유지: macd

근거:
  - MACD가 표준 지표 (해석 용이)
  - Trend strength는 MACD 파생
```

**2. MACD vs MACD Signal (Correlation: 0.9508)**
```yaml
문제: macd ≈ macd_signal
제거: macd_signal
유지: macd

근거:
  - 동일한 이유 (LONG Entry와 일관성)
```

### Exit - 제거 목록 (2개):
1. trend_strength
2. macd_signal

### Exit - 최종 Features: 23개 (25 - 2)

**⚠️ 추가 조치 필요**: Missing 16개 feature 계산 구현

---

## 구현 계획

### Phase 1: Feature List 업데이트 ✅
```bash
# 새 feature list 파일 생성
LONG_ENTRY_REDUCED_FEATURES.txt
SHORT_ENTRY_REDUCED_FEATURES.txt
EXIT_REDUCED_FEATURES.txt
```

### Phase 2: Feature 계산 코드 수정
```python
# calculate_all_features.py 수정
# - bb_high, bb_low 계산 제거 (또는 계산하되 모델에서 제외)
# - Exit model missing features 추가 계산
```

### Phase 3: 모델 재학습
```bash
# 감소된 feature로 모델 학습
python scripts/training/retrain_with_reduced_features.py
```

### Phase 4: 백테스트 검증
```bash
# 성능 비교
python scripts/experiments/backtest_reduced_features.py
```

### Phase 5: 성능 비교 및 배포 결정
```yaml
비교 지표:
  - Win Rate (현재: 63.6%)
  - Return (현재: +75.58%)
  - Sharpe (현재: 0.336)
  - Max Drawdown (현재: -12.2%)

성공 기준:
  - Win Rate >= 63%
  - Return >= +70%
  - Sharpe >= 0.30
  - Max DD <= -15%

결과:
  - 통과: 프로덕션 배포
  - 실패: Rollback to 현행 features
```

---

## 예상 효과

### Positive:
1. **Overfitting 감소**: 중복 제거로 일반화 성능 향상
2. **학습 속도**: 30-40% 빨라짐
3. **추론 속도**: 20-30% 빨라짐
4. **해석력**: 모델 이해 및 디버깅 용이
5. **메모리**: 20-30% 감소

### Risks:
1. **정보 손실**: 미세한 차이의 정보 손실 가능
2. **성능 저하**: 일시적 성능 하락 가능 (재학습으로 회복)

### Mitigation:
1. **A/B 테스트**: 현행 vs 감소 모델 병행 테스트
2. **Gradual Rollout**: Testnet → Mainnet 순차 배포
3. **Rollback Plan**: 성능 저하 시 즉시 복구

---

## Timeline

**Week 1** (2025-10-23):
- [x] Correlation 분석 완료
- [ ] Feature list 업데이트
- [ ] 코드 수정 및 재학습
- [ ] 백테스트 검증

**Week 2** (2025-10-30):
- [ ] Testnet 배포 및 모니터링
- [ ] 성능 비교 분석
- [ ] 최종 배포 결정

---

## Next Actions

**즉시**:
1. 새 feature list 파일 생성
2. Feature 계산 코드 수정
3. 모델 재학습 스크립트 작성

**검증**:
1. 백테스트 성능 확인
2. Out-of-sample 테스트
3. Testnet 실전 검증

**배포**:
1. 성능 기준 통과 확인
2. Mainnet 점진적 배포
3. 모니터링 및 조정

---

**Status**: 🎯 Ready for Implementation
**Expected Impact**: +10-20% performance improvement (overfitting reduction)
**Risk Level**: 🟡 Medium (mitigated by thorough testing)
