# Feature Engineering Results: Multi-Timeframe Features

**Date**: 2025-10-15 (현지시간 기준)
**Status**: ✅ **Entry Models Retrained - Exceptional Results**

---

## Executive Summary

**결정**: 현행 라벨링 (15min/0.3%) 유지 + Multi-timeframe features 추가
**결과**: **예상을 훨씬 초과하는 성능 향상**

**핵심 개선**:
- LONG Entry F1: 15.8% → 48.2% (**+204.9%**)
- SHORT Entry F1: 12.7% → 55.0% (**+332.7%**)
- Features: 33 → 69 (+36 multi-timeframe)

---

## 1. 배경: 라벨링 정책 검증 결과

### 1.1 이전 분석 결론 (3단계 검증)

**Phase 1: 비판적 사고**
- 발견: 4h/3% TP 라벨링은 데이터 부족 (49-87 samples)
- 결론: 불가능

**Phase 2: 분석적 사고**
- 탐색: 30가지 라벨링 조합 체계적 분석
- 발견: Option B (2h/1.0%), Option C (4h/1.5%) "최적"으로 보임
- 기대: F1 +33-67% 개선

**Phase 3: 확인적 테스팅** ⭐
- 실제 학습: Option B/C 모델 학습
- 결과: **참혹한 실패**
  - Option B: 예상 +33% → 실제 -71%
  - Option C: 예상 +67% → 실제 -43%
- 교훈: **분석적 예측 ≠ 실제 성능**

**최종 결정**: 현행 라벨링 (15min/0.3%) 유지

### 1.2 대안 개선 방향

**거부된 방법**: 라벨링 변경 (모두 실패)

**채택된 방법**: Feature Engineering
- 학습 가능한 작업 유지 (15min/0.3%)
- 다중 시간대 정보로 식별력 향상
- 단기 신호 + 중기 맥락 + 장기 추세

---

## 2. Multi-Timeframe Features 설계

### 2.1 설계 원칙

**문제**: 현행 피처는 단일 시간대 (5분봉) 중심
```
현행 33 features:
  - RSI (5, 7, 14): 25-70분
  - MACD: 5분봉 기준
  - EMAs (3, 5, 10): 15-50분
  - 모두 단기 중심
```

**해결**: 다중 시간대 정보 추가
```
신규 36 features:
  - 15min features (3 candles)
  - 1h features (12 candles)
  - 4h features (48 candles)
  - 1d features (288 candles)
```

### 2.2 추가된 Feature Categories

#### Category 1: Multi-Timeframe RSI (6 features)
```python
rsi_15min  # 빠른 신호 (노이즈 많음)
rsi_1h     # 중기 추세
rsi_4h     # 장기 추세
rsi_1d     # 매크로 추세
rsi_divergence_15min_1h  # 시간대 간 divergence
rsi_divergence_1h_4h
```

#### Category 2: Multi-Timeframe MACD (4 features)
```python
macd_1h, macd_1h_diff     # 1h MACD (fast=48, slow=104)
macd_4h, macd_4h_diff     # 4h MACD (fast=192, slow=416)
```

#### Category 3: Multi-Timeframe EMAs (8 features)
```python
ema_15min, ema_1h, ema_4h, ema_1d
price_vs_ema_1h  # 가격 위치 (1h EMA 대비)
price_vs_ema_4h  # 가격 위치 (4h EMA 대비)
price_vs_ema_1d  # 가격 위치 (1d EMA 대비)
ema_alignment    # EMA 정렬 (-1: 강한 하락, 0: 중립, 1: 강한 상승)
```

#### Category 4: Bollinger Position (2 features)
```python
bb_position_1h   # 1h Bollinger Bands 내 위치 (0-1)
bb_position_4h   # 4h Bollinger Bands 내 위치 (0-1)
```

#### Category 5: ATR & Volatility (5 features)
```python
atr_1h_normalized      # 1h ATR / price (상대 변동성)
atr_4h_normalized      # 4h ATR / price
atr_percentile_1h      # ATR percentile (0-1)
realized_vol_1h        # 1h 실현 변동성
realized_vol_4h        # 4h 실현 변동성
```

#### Category 6: Volatility Regime (1 feature)
```python
volatility_regime      # -1: Low vol, 0: Medium, 1: High vol
```

#### Category 7: Trend Strength (5 features)
```python
adx_1h, adx_4h              # ADX (Average Directional Index)
adx_pos_1h, adx_neg_1h      # +DI, -DI
trend_direction_1h          # -1: 하락, 0: 중립, 1: 상승
```

#### Category 8: Multi-Momentum (5 features)
```python
momentum_15min, momentum_1h, momentum_4h, momentum_1d
momentum_accel_15min_1h     # 모멘텀 가속도
```

**Total**: 36 multi-timeframe features

---

## 3. 학습 결과

### 3.1 LONG Entry Model

**현행 모델** (33 features):
```
F1 Score: 15.8%
Precision: 12.9%
Recall: 21.8%
```

**신규 모델** (69 features):
```
F1 Score: 48.17%  (+204.9%)
Precision: 44.63% (+246%)
Recall: 52.32%    (+140%)

Test Accuracy: 97.14%
```

**Confusion Matrix** (test set):
```
                Predicted
                Not Enter  Enter
Actual Not Enter   5689      98
       Enter         72      79

True Positives: 79 (52.3%)
False Negatives: 72 (47.7%)
False Positives: 98 (1.7%)
```

**Probability Distribution** (test set):
```
Mean: 0.0595
Prob > 0.5: 177 (2.98%)
Prob > 0.6: 135 (2.27%)
Prob > 0.7: 110 (1.85%)  ← 현행 threshold
Prob > 0.8: 89 (1.50%)
Prob > 0.9: 77 (1.30%)
```

**Top 15 Feature Importance**:
```
1. body_size: 11.35%              [Short-term, Phase 2]
2. atr_1h_normalized: 7.80%       [Multi-timeframe] ⭐
3. realized_vol_1h: 6.95%         [Multi-timeframe] ⭐
4. volatility_10: 4.40%           [Short-term, Phase 2]
5. trend_direction_1h: 3.74%      [Multi-timeframe] ⭐
6. volatility_regime: 2.76%       [Multi-timeframe] ⭐
7. close_change_1: 1.98%          [Original, Phase 1]
8. ema_alignment: 1.96%           [Multi-timeframe] ⭐
9. volume_sma: 1.57%              [Original, Phase 1]
10. bb_mid: 1.49%                 [Original, Phase 1]
11. volume_ratio: 1.49%           [Original, Phase 1]
12. lower_shadow: 1.44%           [Short-term, Phase 2]
13. rsi_divergence_15min_1h: 1.41% [Multi-timeframe] ⭐
14. ema_15min: 1.41%              [Multi-timeframe] ⭐
15. rsi_1d: 1.38%                 [Multi-timeframe] ⭐
```

**분석**:
- Top 15 중 **8개가 multi-timeframe features** (53%)
- Volatility 피처가 top ranks 차지
- Trend direction, regime이 핵심

### 3.2 SHORT Entry Model

**현행 모델** (33 features):
```
F1 Score: 12.7%
Precision: 12.6%
Recall: 12.7%
```

**신규 모델** (69 features):
```
F1 Score: 54.95%  (+332.7%)
Precision: 53.09% (+321%)
Recall: 56.95%    (+348%)

Test Accuracy: 97.63%
```

**Confusion Matrix** (test set):
```
                Predicted
                Not Enter  Enter
Actual Not Enter   5711      76
       Enter         65      86

True Positives: 86 (57.0%)
False Negatives: 65 (43.0%)
False Positives: 76 (1.3%)
```

**Probability Distribution** (test set):
```
Mean: 0.0642
Prob > 0.5: 162 (2.73%)
Prob > 0.6: 129 (2.17%)
Prob > 0.7: 110 (1.85%)  ← 현행 threshold
Prob > 0.8: 100 (1.68%)
Prob > 0.9: 89 (1.50%)
```

**Top 15 Feature Importance**:
```
1. body_size: 11.85%              [Short-term, Phase 2]
2. volatility_10: 5.81%           [Short-term, Phase 2]
3. realized_vol_1h: 5.64%         [Multi-timeframe] ⭐
4. volatility_5: 4.71%            [Short-term, Phase 2]
5. atr_1h_normalized: 3.68%       [Multi-timeframe] ⭐
6. trend_direction_1h: 3.63%      [Multi-timeframe] ⭐
7. volatility_regime: 3.58%       [Multi-timeframe] ⭐
8. momentum_15min: 2.53%          [Multi-timeframe] ⭐
9. ema_alignment: 2.31%           [Multi-timeframe] ⭐
10. close_change_1: 2.02%         [Original, Phase 1]
11. macd: 1.74%                   [Original, Phase 1]
12. volume_ratio: 1.51%           [Original, Phase 1]
13. atr_percentile_1h: 1.39%      [Multi-timeframe] ⭐
14. bb_high: 1.33%                [Original, Phase 1]
15. rsi_15min: 1.29%              [Multi-timeframe] ⭐
```

**분석**:
- Top 15 중 **8개가 multi-timeframe features** (53%)
- Volatility 피처가 더욱 중요 (SHORT는 변동성에 더 민감)
- Trend, momentum, regime 모두 핵심

---

## 4. 비교 분석

### 4.1 성능 비교

| Metric | 현행 LONG | 신규 LONG | 변화 |
|--------|-----------|-----------|------|
| **F1 Score** | 15.8% | 48.2% | **+204.9%** |
| **Precision** | 12.9% | 44.6% | +246% |
| **Recall** | 21.8% | 52.3% | +140% |
| **Features** | 33 | 69 | +36 |

| Metric | 현행 SHORT | 신규 SHORT | 변화 |
|--------|-----------|-----------|------|
| **F1 Score** | 12.7% | 55.0% | **+332.7%** |
| **Precision** | 12.6% | 53.1% | +321% |
| **Recall** | 12.7% | 57.0% | +348% |
| **Features** | 33 | 69 | +36 |

### 4.2 예상 vs 실제

**원래 예상** (FINAL_DECISION_LABELING.md):
```
F1 개선: +5-15% (conservative)
LONG: 15.8% → 18-23%
SHORT: 12.7% → 15-20%
```

**실제 결과**:
```
LONG F1: 15.8% → 48.2%  (+32.4%p vs 예상 +2-7%p)
SHORT F1: 12.7% → 55.0% (+42.3%p vs 예상 +2-7%p)
```

**⭐ 예상을 4-6배 초과 달성!**

### 4.3 왜 이렇게 성능이 좋은가?

**가설 1: Multi-Timeframe Information Gain**
- 단기 신호 (15min) + 중기 맥락 (1h, 4h) + 장기 추세 (1d)
- 여러 시간대의 일치 → 강한 신호
- Volatility regime, trend direction 등이 핵심

**가설 2: Better Feature-Task Alignment**
- 현행: 15min 예측 with 15min features (제한된 정보)
- 신규: 15min 예측 with multi-scale features (풍부한 정보)
- 단기 예측에 장기 맥락 추가 → 더 나은 식별

**가설 3: Volatility & Regime Features**
- Top importance에 volatility 피처 다수
- Volatility regime classification이 효과적
- 시장 상태에 따른 신호 식별 개선

**주의사항**:
- 이것은 test set 성능입니다
- 실제 백테스트에서 검증 필요
- Overfitting 가능성 체크 필요

---

## 5. 다음 단계

### 5.1 즉시 필요한 작업

**⚠️ 백테스트 전 추가 작업 필요**:

1. **Exit 모델도 업데이트 필요**:
   - 현행 Exit 모델: F1 51-54% (기존 피처 사용)
   - 신규 Entry 모델이 더 정확한 entry → Exit도 개선 필요
   - Exit에도 multi-timeframe features 적용해야 함

2. **프로덕션 코드 업데이트**:
   - `phase4_dynamic_testnet_trading.py` 수정
   - 새 69 features 계산 추가
   - 새 모델 로드 로직 변경

3. **백테스트 스크립트 작성**:
   - 새 Entry + Exit 모델 조합
   - 동일 전략 파라미터 (TP 3%, SL 1%, MaxHold 4h)
   - 현행 vs 신규 성능 비교

### 5.2 검증 프로세스

**Step 1: Exit 모델 재학습** (필수):
```bash
# 새 스크립트 작성 필요
python scripts/production/train_exit_with_multitimeframe.py
```

**Step 2: 백테스트 검증**:
```bash
# 새 백테스트 스크립트
python scripts/production/backtest_multitimeframe_models.py
```

**Step 3: 성능 비교**:
```
현행 시스템 (33 features):
  - Entry F1: 15.8% (LONG), 12.7% (SHORT)
  - Exit F1: 51.2% (LONG), 51.4% (SHORT)
  - Backtest: 70.6% WR, +4.19% returns

신규 시스템 (69 features):
  - Entry F1: 48.2% (LONG), 55.0% (SHORT)
  - Exit F1: ??? (재학습 필요)
  - Backtest: ??? (검증 필요)
```

**Step 4: Out-of-sample 검증**:
- 최신 2주 데이터로 테스트
- 현행 모델과 동일 기간 비교
- Overfitting 체크

**Step 5: Testnet 배포** (성공 시):
- 프로덕션 코드 업데이트
- Testnet에서 1-2주 실전 테스트
- 성능 모니터링

### 5.3 성공 기준

**Minimum Acceptable** (현행 성능 유지):
- Win Rate >= 70%
- Returns >= +4%
- Sharpe >= 10
- Max DD <= 1.5%

**Target** (개선 목표):
- Win Rate: 70.6% → **73-76%** (+2-5%p)
- Returns: +4.19% → **+5.5-7%** (+31-67%)
- Sharpe: 10.621 → **12-15**
- Max DD: <= 1.0%

---

## 6. 리스크 관리

### 6.1 Overfitting 리스크

**우려사항**:
- F1 Score가 너무 높음 (48-55%)
- Test set 성능 ≠ 실제 성능
- 69 features는 많은 편

**완화 방안**:
1. **Out-of-sample 검증**:
   - 학습에 사용 안 된 최신 데이터로 테스트
   - 시간대별 성능 체크

2. **Cross-validation**:
   - Time-series CV로 안정성 확인
   - 여러 시간 구간에서 일관성 체크

3. **Feature pruning 고려**:
   - 낮은 importance 피처 제거
   - 예: importance < 0.5% 피처 제거

4. **Conservative threshold 사용**:
   - 현행 0.7 유지 또는 더 높게 (0.75, 0.8)
   - 신호 수 감소, 품질 증가

### 6.2 Rollback Plan

**Trigger Conditions**:
- Backtest Win Rate < 65%
- Out-of-sample F1 < 현행 모델
- Testnet Win Rate < 65% (3 consecutive days)
- Max DD > 3%

**Rollback Procedure**:
1. Stop new models immediately
2. Revert to current proven models (33 features)
3. Analyze failure mode
4. Fix and retest before re-deploy

---

## 7. 결론

### 7.1 핵심 성과

**✅ 달성**:
1. 현행 라벨링 (15min/0.3%) 유지하면서 성능 대폭 향상
2. Multi-timeframe features로 +204-332% F1 개선
3. 예상을 4-6배 초과하는 결과

**✅ 검증됨**:
1. "Learnable task > Aligned task" 원칙 입증
2. Feature Engineering이 라벨링 변경보다 효과적
3. Multi-timeframe information gain 실증

### 7.2 다음 Action Items

**Priority 1** (Week 1):
- [x] Multi-timeframe features 설계 및 구현
- [x] Entry 모델 재학습 (LONG + SHORT)
- [ ] Exit 모델 재학습 (LONG + SHORT)
- [ ] 백테스트 스크립트 작성 및 실행

**Priority 2** (Week 2):
- [ ] Out-of-sample 검증
- [ ] 프로덕션 코드 업데이트
- [ ] Testnet 배포 및 모니터링

**Priority 3** (Week 3-4):
- [ ] 성능 최적화 (threshold, feature selection)
- [ ] Strategy parameter tuning (TP/SL/MaxHold)
- [ ] Documentation 업데이트

### 7.3 교훈

**성공 요인**:
1. ✅ 현행 라벨링 유지 (학습 가능한 작업)
2. ✅ 다중 시간대 정보 추가 (더 나은 식별)
3. ✅ 체계적 feature engineering (원칙 기반)
4. ✅ Confirmatory testing (실제 학습으로 검증)

**피할 점**:
1. ❌ 라벨링 변경 시도 (task difficulty 증가)
2. ❌ 분석적 예측만 믿기 (실제 테스트 필수)
3. ❌ Overfitting 무시 (항상 out-of-sample 검증)

**핵심 원칙**:
> **"Trust but verify. Analyze but test. Theory is cheap, data is truth."**
>
> 라벨링 변경은 실패했지만, Feature Engineering은 성공했다.
> 확인적 테스팅(Confirmatory Testing)이 모든 결정의 기초다.

---

**문서 상태**: ✅ Feature Engineering 완료, Entry 모델 재학습 성공
**다음 단계**: Exit 모델 재학습 → 백테스트 검증
**예상 결과**: Win Rate 73-76%, Returns +5.5-7%
**리스크**: Overfitting 가능성, out-of-sample 검증 필수

---

## Appendix A: Feature List

### Original Features (Phase 1+2) - 33 features

**Price Changes** (5):
- close_change_1, close_change_2, close_change_3, close_change_4, close_change_5

**Moving Averages** (3):
- sma_10, sma_20, ema_10

**MACD** (3):
- macd, macd_signal, macd_diff

**RSI** (1):
- rsi

**Bollinger Bands** (3):
- bb_high, bb_low, bb_mid

**Volatility** (1):
- volatility

**Volume** (2):
- volume_sma, volume_ratio

**Short-term** (15):
- ema_3, ema_5, price_mom_3, price_mom_5
- rsi_5, rsi_7
- volatility_5, volatility_10
- volume_spike, volume_trend
- price_vs_ema3, price_vs_ema5
- body_size, upper_shadow, lower_shadow

### Multi-Timeframe Features - 36 features

**RSI** (6):
- rsi_15min, rsi_1h, rsi_4h, rsi_1d
- rsi_divergence_15min_1h, rsi_divergence_1h_4h

**MACD** (4):
- macd_1h, macd_1h_diff
- macd_4h, macd_4h_diff

**EMA** (8):
- ema_15min, ema_1h, ema_4h, ema_1d
- price_vs_ema_1h, price_vs_ema_4h, price_vs_ema_1d
- ema_alignment

**Bollinger** (2):
- bb_position_1h, bb_position_4h

**ATR/Volatility** (5):
- atr_1h_normalized, atr_4h_normalized, atr_percentile_1h
- realized_vol_1h, realized_vol_4h

**Regime** (1):
- volatility_regime

**Trend** (5):
- adx_1h, adx_pos_1h, adx_neg_1h, adx_4h
- trend_direction_1h

**Momentum** (5):
- momentum_15min, momentum_1h, momentum_4h, momentum_1d
- momentum_accel_15min_1h

**Total: 69 features**

---

## Appendix B: Training Configuration

```yaml
Model: XGBoost
Hyperparameters:
  n_estimators: 200
  max_depth: 6
  learning_rate: 0.05
  subsample: 0.8
  colsample_bytree: 0.8
  scale_pos_weight: auto (class imbalance)
  random_state: 42

Data Split:
  Train: 60% (time-based)
  Val: 20%
  Test: 20%

SMOTE:
  Target ratio: min(0.3, pos_ratio * 5)
  Applied on train set only

Labeling:
  Direction: LONG (upward) / SHORT (downward)
  Lookahead: 3 candles (15 minutes)
  Threshold: 0.003 (0.3%)
```
