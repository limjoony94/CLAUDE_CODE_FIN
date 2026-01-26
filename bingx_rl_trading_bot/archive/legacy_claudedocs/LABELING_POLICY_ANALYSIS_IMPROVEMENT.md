# Labeling Policy Analysis & Improvement Plan

**Date**: 2025-10-15 01:00
**Status**: 🔍 Analysis Complete | 📋 Improvement Plan Ready
**Target**: All 4 Models (LONG Entry, SHORT Entry, LONG Exit, SHORT Exit)

---

## Executive Summary

현재 4개 모델의 라벨링 정책을 분석한 결과, **Entry 모델의 성능이 낮은 근본 원인**은 **라벨링 정책의 문제**로 확인됨.

**핵심 문제**:
- Entry 모델 F1 Score: 0.127-0.158 (매우 낮음)
- Exit 모델 F1 Score: 0.512-0.514 (중간)
- **백테스트 승률 70%임에도 Entry 모델 F1이 낮은 이유**: 라벨링과 실제 거래 전략 불일치

**해결 방향**:
1. Entry 라벨링을 실제 백테스트 승리 조건에 맞춤
2. Exit 라벨링을 더 보수적이고 실용적으로 개선
3. 전체 시스템의 라벨링 일관성 확보

---

## 1. 현재 라벨링 정책 분석

### 1.1 LONG Entry Model

**현재 정책**:
```python
lookahead = 3 candles (15 minutes)
threshold = 0.003 (0.3%)

Label = 1 if: max_price[t+1:t+3] >= current_price * 1.003
Label = 0 otherwise
```

**성능**:
```
F1 Score:  0.1577 ❌ 매우 낮음
Precision: 0.1295 (12.9%)
Recall:    0.2183 (21.8%)

Positive samples: 4.3%
Negative samples: 95.7%
```

**문제점 분석**:

1. **라벨링 vs 실제 전략 불일치**:
   ```
   라벨링: "향후 15분 내 +0.3% 도달"
   실제 전략:
     - Stop Loss: -1%
     - Take Profit: +3%
     - Max Hold: 4 hours (48 candles)

   → 라벨이 "15분 내 작은 상승"을 찾는데,
      실제론 "4시간 내 큰 상승 또는 작은 손절" 거래
   ```

2. **Lookahead가 너무 짧음**:
   - 15분 내 0.3% 상승은 노이즈가 많음
   - 실제 거래는 최대 4시간 보유
   - **백테스트 평균 보유**: 3.92시간 (47 candles)

3. **Threshold가 너무 낮음**:
   - 0.3%는 거래 수수료 (0.04%) 고려 시 실질 이익 0.26%
   - Stop Loss -1%와 비대칭 (Risk/Reward 1:0.3)
   - **실제 백테스트 TP**: +3%

4. **Positive 샘플 불균형**:
   - 4.3% positive → 심한 불균형
   - SMOTE로 보완해도 성능 한계

### 1.2 SHORT Entry Model

**현재 정책**:
```python
lookahead = 3 candles (15 minutes)
threshold = 0.003 (0.3%)

Label = 1 if: future_return[t+3] < -0.003
Label = 0 otherwise
```

**성능**:
```
F1 Score:  0.127 ❌ 매우 낮음
Precision: 0.126 (12.6%)
Recall:    0.127 (12.7%)

Positive samples: 4.4%
Probability > 0.7: 0.5% (매우 보수적)
```

**문제점 분석**:

1. **LONG 모델과 동일한 문제**:
   - Lookahead 15분 vs 실제 보유 4시간
   - Threshold 0.3% vs 실제 TP 3%

2. **하락 예측의 어려움**:
   - BTC는 장기 상승 자산
   - 15분 내 0.3% 하락은 더 드물고 노이즈가 많음
   - Positive 샘플 4.4% (LONG과 비슷하지만 더 어려움)

3. **확률 분포 문제**:
   - Mean probability: 0.1933 (낮음)
   - Prob > 0.7: 0.5% (거의 신호 없음)
   - **백테스트 SHORT 거래**: 1.1개/윈도우 (매우 희소)

### 1.3 LONG Exit Model

**현재 정책**:
```python
lookahead = 12 candles (60 minutes)

Label = 1 (Exit) if BOTH:
  1. Near Peak: current_pnl >= peak_pnl * 0.80
  2. Beats Holding: current_pnl > future_pnl[t+12]

Label = 0 (Hold) otherwise
```

**성능**:
```
F1 Score:  0.512 ⚠️ 중간
Precision: 0.349 (34.9%)
Recall:    0.963 (96.3%)

Accuracy: 86.9%
Data: 88,739 samples (1,933 LONG trades simulated)
```

**문제점 분석**:

1. **Precision이 낮음** (34.9%):
   - False Positive 많음 (불필요한 청산)
   - AND 조건이지만 여전히 노이즈 많음
   - **실제 백테스트**: Rule-based가 더 높은 수익

2. **Near Peak 80% 기준이 애매**:
   ```
   예시:
   - Peak P&L: +5%
   - Current P&L: +4% (80% 도달)
   - Future P&L (1h): +2%

   → Label = 1 (Exit)

   But 실제로 4% → 2%는 -2% 손실
   80% 기준이 너무 낮아서 조기 청산 유발
   ```

3. **Lookahead 1시간이 짧음**:
   - 평균 보유 시간: 3.92시간
   - 1시간 후 P&L로 판단은 불충분
   - 더 긴 lookahead 필요 (2-3시간)

4. **실제 거래와의 괴리**:
   ```
   라벨링: "80% 피크 + 1시간 후보다 나음"
   실제 전략: "TP +3%, SL -1%, Max Hold 4h"

   → 라벨이 TP/SL 기준을 전혀 고려 안 함
   ```

### 1.4 SHORT Exit Model

**현재 정책**:
```python
lookahead = 12 candles (60 minutes)

Label = 1 (Exit) if BOTH:
  1. Near Peak: current_pnl >= peak_pnl * 0.80
  2. Beats Holding: current_pnl > future_pnl[t+12]

Label = 0 (Hold) otherwise
```

**성능**:
```
F1 Score:  0.514 ⚠️ 중간
Precision: 0.352 (35.2%)
Recall:    0.956 (95.6%)

Accuracy: 88.0%
Data: 89,345 samples (1,936 SHORT trades simulated)
```

**문제점 분석**:
- LONG Exit과 동일한 문제
- SHORT는 더 빠른 청산이 필요할 수 있음 (하락이 빠름)
- 현재 정책은 LONG과 완전히 동일 (방향성 차이 미반영)

---

## 2. 문제점 종합 분석

### 2.1 근본 원인 (Root Cause)

**핵심 문제**: **라벨링이 실제 거래 전략을 반영하지 못함**

```
현재 상황:
┌─────────────────────────────────────────────────────┐
│ Entry Labeling:  15분 내 0.3% 예측                  │
│ Exit Labeling:   1시간 내 80% 피크 예측             │
│                                                     │
│ ≠                                                   │
│                                                     │
│ Actual Trading:  4시간 내 +3% TP / -1% SL 달성     │
│                                                     │
│ → Mismatch!                                         │
└─────────────────────────────────────────────────────┘
```

### 2.2 성능 영향 분석

**Entry 모델 F1 낮음 → 백테스트 승률 70%의 모순**:

```
Entry 모델이 F1 15.8%로 낮은데 왜 백테스트 승률이 70%인가?

원인:
1. 백테스트는 0.7 threshold 사용 → 매우 보수적 필터링
2. 실제 거래는 4시간 보유 + 3% TP 전략
3. 모델은 15분 0.3% 예측 (노이즈가 많음)
4. 하지만 높은 확률 신호만 선택 → 운 좋게 장기 수익

즉, 모델이 잘 예측해서가 아니라,
보수적 필터링과 장기 보유 전략 덕분!
```

**Exit 모델 F1 51% → Rule-based와 성능 유사**:

```
Exit 모델 F1이 51%인데 왜 Rule-based와 비슷한가?

원인:
1. Exit 라벨링이 TP/SL을 고려하지 않음
2. 80% 피크 기준이 너무 낮음 (조기 청산)
3. Precision 35% → False Positive 많음
4. Rule-based (TP +3%, SL -1%)가 더 명확하고 효과적

즉, ML이 복잡한 패턴을 못 잡고,
단순 규칙이 더 직관적이고 효과적!
```

### 2.3 백테스트 승률 70%의 진실

```
백테스트 승률 70.6% 달성 요인:

1. Threshold 0.7: 상위 1-2% 신호만 거래
   → 모델 F1 낮아도 극소수 고품질 신호 선택

2. TP +3%: 목표가 충분히 높음
   → 4시간 내 3% 상승 확률 높음 (BTC 변동성)

3. SL -1%: 손실 제한
   → 큰 손실 방지

4. Max Hold 4h: 충분한 대기
   → 15분 예측과 무관하게 장기 관찰

5. 수수료 0.02%: 낮음
   → 3% 수익의 0.67% 불과

결론:
모델이 정확해서가 아니라,
전략 파라미터와 보수적 필터링 덕분!
```

---

## 3. 개선 방안 (Improvement Plan)

### 3.1 Entry 모델 라벨링 개선

**목표**: 라벨링을 실제 거래 전략에 일치시키기

**새로운 정책 (Proposed)**:

```python
# === Option 1: Future Profitability (추천) ===
lookahead = 48 candles (4 hours, Max Hold 기간)
threshold_tp = 0.03 (3%, TP와 일치)
threshold_sl = -0.01 (-1%, SL과 일치)

Label = 1 if:
  max_profit[t+1:t+48] >= threshold_tp
  AND
  min_loss[t+1:t+48] > threshold_sl

Label = 0 otherwise

설명:
- 4시간 내 +3% 도달 가능하고
- 동시에 -1% 이하로 떨어지지 않으면 → LONG
- 실제 거래 승리 조건과 정확히 일치
```

```python
# === Option 2: Realistic Trade Simulation ===
lookahead = 48 candles (4 hours)
simulate_trade = True

For each candle:
  1. Simulate forward trade with actual rules:
     - TP: +3%
     - SL: -1%
     - Max Hold: 4h
     - Transaction cost: 0.04%

  2. Label = 1 if:
     - Trade exits with profit > 0
     - Exit reason: TP or positive MaxHold

  3. Label = 0 if:
     - Trade exits with loss
     - Exit reason: SL

설명:
- 실제 거래를 완전히 시뮬레이션
- 라벨 = 실제 거래 승패
- 가장 정확하지만 연산 비용 높음
```

**기대 효과**:
- F1 Score 15.8% → 40-50% (약 3배 향상)
- Precision 12.9% → 30-40%
- Positive 샘플 4.3% → 15-20% (더 균형)
- 백테스트 성능 유지 또는 향상

### 3.2 SHORT Entry 모델 라벨링 개선

**새로운 정책 (Proposed)**:

```python
# Option 1: Future Profitability for SHORT
lookahead = 48 candles (4 hours)
threshold_tp = 0.03 (3%, TP와 일치)
threshold_sl = -0.01 (-1%, SL과 일치)

Label = 1 if:
  # SHORT는 가격 하락이 수익
  max_profit[t+1:t+48] >= threshold_tp  # 하락으로 3% 수익
  AND
  max_loss[t+1:t+48] <= threshold_sl  # 상승으로 1% 손실 이내

Label = 0 otherwise
```

**추가 고려사항**:
- SHORT는 하락이 더 빠름 → Lookahead 짧게 (3-4시간)?
- 또는 Threshold 낮게 (2% TP)?
- 백테스트로 최적값 찾기

### 3.3 Exit 모델 라벨링 개선

**새로운 정책 (Proposed)**:

```python
# Option 1: TP/SL Aware Exit Labeling
lookahead = 24 candles (2 hours, 더 긴 시야)
near_peak_threshold = 0.90  # 90%로 상향 (더 보수적)

Label = 1 (Exit) if:
  1. TP Approaching: current_pnl >= 0.025 (TP 3%의 83%)
     OR
  2. SL Risk: future_min_pnl[t:t+24] < -0.008 (SL -1%의 80%)
     OR
  3. Near Peak + Beats Holding:
        current_pnl >= peak_pnl * 0.90
        AND
        current_pnl > future_pnl[t+24]

Label = 0 (Hold) otherwise

설명:
- TP 근접 시 청산 고려
- SL 위험 감지 시 미리 청산
- 피크 90% + 2시간 후보다 나음
- 더 실용적이고 TP/SL 반영
```

```python
# Option 2: Risk-Adjusted Exit
Label = 1 (Exit) if:
  1. Profit Secure: current_pnl >= 0.02 (2%)
     AND future_max_pnl[t:t+24] < current_pnl * 1.1
     # 현재 2% 이상이고, 향후 10% 이상 추가 상승 없으면 청산

  2. Loss Cut: current_pnl < -0.005 (-0.5%)
     AND future_min_pnl[t:t+24] < current_pnl
     # 손실이고 더 나빠지면 청산

  3. Momentum Loss:
     recent_momentum < 0
     AND current_pnl > 0.01
     # 모멘텀 상실하고 수익 있으면 청산

Label = 0 (Hold) otherwise
```

**기대 효과**:
- Precision 34.9% → 45-55%
- F1 Score 51.2% → 60-70%
- False Positive 감소
- Rule-based보다 우수한 수익

---

## 4. 구현 계획 (Implementation Plan)

### Phase 1: Entry 모델 재훈련 (우선순위 1)

**Task 1.1: LONG Entry 새 라벨링**
```python
Script: train_xgboost_phase4_advanced_v2.py

Changes:
1. create_labels() 함수 수정:
   - lookahead: 3 → 48
   - threshold: 0.003 → 0.03 (TP)
   - Add SL check: min_loss > -0.01

2. 라벨 로직:
   def create_labels_realistic(df, lookahead=48, tp=0.03, sl=0.01):
       labels = []
       for i in range(len(df)):
           if i >= len(df) - lookahead:
               labels.append(0)
               continue

           current_price = df['close'].iloc[i]
           future_prices = df['close'].iloc[i+1:i+1+lookahead]

           # Max profit
           max_profit_pct = (future_prices.max() - current_price) / current_price

           # Min loss
           min_loss_pct = (future_prices.min() - current_price) / current_price

           # Label = 1 if: TP reachable AND SL not hit
           if max_profit_pct >= tp and min_loss_pct > -sl:
               labels.append(1)
           else:
               labels.append(0)

       return np.array(labels)

3. 기대 positive 비율: 15-20%
```

**Task 1.2: SHORT Entry 새 라벨링**
```python
Script: train_xgboost_short_model_v2.py

Changes:
1. create_short_target() 함수 수정:
   - lookahead: 3 → 48
   - threshold: 0.003 → 0.03 (TP)
   - Add SL check for SHORT

2. 라벨 로직 (SHORT 방향):
   def create_short_target_realistic(df, lookahead=48, tp=0.03, sl=0.01):
       labels = []
       for i in range(len(df)):
           # ...
           # Max profit (하락으로 수익)
           max_profit_pct = (current_price - future_prices.min()) / current_price

           # Max loss (상승으로 손실)
           max_loss_pct = (future_prices.max() - current_price) / current_price

           # Label = 1 if: SHORT TP reachable AND SL not hit
           if max_profit_pct >= tp and max_loss_pct < sl:
               labels.append(1)
           else:
               labels.append(0)

       return np.array(labels)
```

**Task 1.3: 훈련 및 검증**
```bash
# LONG Entry 재훈련
python scripts/production/train_xgboost_phase4_advanced_v2.py

# SHORT Entry 재훈련
python scripts/production/train_xgboost_short_model_v2.py

# 모델 성능 확인
- F1 Score >= 0.40 목표
- Precision >= 0.30 목표
- Positive 샘플 15-20%
```

### Phase 2: Exit 모델 재훈련 (우선순위 2)

**Task 2.1: Exit 라벨링 개선**
```python
Script: train_exit_models_v2.py

Changes:
1. label_exit_point() 함수 수정:
   - near_peak_threshold: 0.80 → 0.90
   - lookahead: 12 → 24
   - Add TP/SL awareness

2. 새 라벨 로직:
   def label_exit_point_realistic(candle, trade, lookahead=24):
       current_pnl = candle['pnl_pct']

       # TP approaching
       if current_pnl >= 0.025:  # 2.5% (TP 3%의 83%)
           return 1

       # SL risk
       future_candles = [c for c in trade['candles']
                         if c['offset'] > candle['offset']
                         and c['offset'] <= candle['offset'] + lookahead]
       if future_candles:
           future_min = min([c['pnl_pct'] for c in future_candles])
           if future_min < -0.008:  # -0.8% (SL -1%의 80%)
               return 1

       # Near peak + Beats holding
       peak_pnl = trade['peak_pnl']
       near_peak = current_pnl >= (peak_pnl * 0.90)

       if near_peak and future_candles:
           future_pnl = future_candles[-1]['pnl_pct']
           if current_pnl > future_pnl:
               return 1

       return 0
```

**Task 2.2: 훈련 및 검증**
```bash
# Exit 모델 재훈련
python scripts/experiments/train_exit_models_v2.py

# 모델 성능 확인
- F1 Score >= 0.60 목표
- Precision >= 0.45 목표
- Recall 유지 (>= 0.90)
```

### Phase 3: 통합 백테스트 (우선순위 3)

**Task 3.1: 새 모델로 백테스트**
```bash
# 새 Entry 모델만
python scripts/experiments/backtest_new_entry_models.py

# 새 Entry + Exit 모델
python scripts/experiments/backtest_full_integrated_v2.py
```

**Task 3.2: 성능 비교**
```
비교 대상:
1. Old Entry + Rule Exit (현재)
2. New Entry + Rule Exit
3. New Entry + New Exit
4. Old Entry + New Exit

목표:
- 승률: 70%+ 유지
- 수익률: 현재 4.19% → 5-6% 향상
- Sharpe: 10.62 → 12+ 향상
```

---

## 5. 예상 성능 향상

### 5.1 Entry 모델

| 지표 | 현재 (Old) | 예상 (New) | 개선율 |
|------|-----------|-----------|--------|
| **F1 Score** | 0.158 | 0.40-0.50 | +153-216% |
| **Precision** | 0.129 | 0.30-0.40 | +133-210% |
| **Recall** | 0.218 | 0.50-0.60 | +129-175% |
| **Positive %** | 4.3% | 15-20% | +249-365% |

**이유**:
- 라벨이 실제 거래 승리 조건과 일치
- Lookahead 48 캔들로 충분한 시야
- TP/SL 기준으로 명확한 positive 샘플

### 5.2 Exit 모델

| 지표 | 현재 (Old) | 예상 (New) | 개선율 |
|------|-----------|-----------|--------|
| **F1 Score** | 0.512 | 0.60-0.70 | +17-37% |
| **Precision** | 0.349 | 0.45-0.55 | +29-58% |
| **Recall** | 0.963 | 0.90-0.95 | -7% to -1% |

**이유**:
- TP/SL awareness로 더 실용적
- Precision 향상 (False Positive 감소)
- Recall 약간 감소해도 실제 수익 향상

### 5.3 백테스트 성능

| 지표 | 현재 (Old) | 예상 (New) | 개선 |
|------|-----------|-----------|------|
| **Returns** | +4.19% | +5-7% | +0.8-2.8%p |
| **Win Rate** | 70.6% | 72-75% | +1.4-4.4%p |
| **Sharpe** | 10.621 | 12-14 | +13-32% |
| **Max DD** | 1.06% | 0.8-1.0% | -6% to -25% |

**이유**:
- 더 정확한 Entry 신호
- 더 효율적인 Exit 타이밍
- False Positive 감소 → 거래 품질 향상

---

## 6. 리스크 및 대응

### 6.1 라벨링 변경 리스크

**Risk 1: Positive 샘플 증가로 Overfitting**
```
현재: 4.3% positive
예상: 15-20% positive

대응:
- SMOTE 제거 또는 비율 낮춤
- Regularization 강화 (gamma, min_child_weight)
- Cross-validation 강화 (5-fold → 10-fold)
```

**Risk 2: Lookahead 증가로 Data Leakage**
```
Lookahead 48 캔들은 미래 4시간 정보 사용

대응:
- 백테스트에서 동일한 lookahead 적용
- 실시간 거래 시 문제없음 (예측 후 실제 대기)
- 과거 데이터로만 훈련 (Time Series Split 유지)
```

**Risk 3: 모델 성능 저하 가능성**
```
라벨링 변경이 항상 개선 보장 안 함

대응:
- Old 모델 백업 유지
- A/B 테스트 (Old vs New)
- 성능 저하 시 빠른 롤백
- 점진적 배포 (테스트넷 → 메인넷)
```

### 6.2 백테스트 성능 리스크

**Risk 1: 백테스트 성능 저하**
```
새 라벨링이 실제론 성능 저하 가능

대응:
- Phase별 백테스트 (Entry만, Exit만, 통합)
- 다양한 시나리오 테스트
- 시장 regime별 검증
```

**Risk 2: Overfitting to Recent Data**
```
Lookahead 48이 최근 시장 패턴에 과적합

대응:
- Walk-forward validation
- Out-of-sample 테스트 (최신 2주 데이터)
- 다양한 기간 백테스트
```

---

## 7. 성공 기준 (Success Criteria)

### 7.1 모델 성능 기준

**Entry 모델**:
- ✅ F1 Score >= 0.40 (현재 0.158)
- ✅ Precision >= 0.30 (현재 0.129)
- ✅ Recall >= 0.50 (현재 0.218)
- ✅ Positive 샘플 >= 15% (현재 4.3%)

**Exit 모델**:
- ✅ F1 Score >= 0.60 (현재 0.512)
- ✅ Precision >= 0.45 (현재 0.349)
- ✅ Recall >= 0.90 (현재 0.963)

### 7.2 백테스트 성능 기준

**필수 기준** (통과 필요):
- ✅ Returns >= +4.5% (현재 +4.19%)
- ✅ Win Rate >= 70% (현재 70.6%)
- ✅ Sharpe >= 10.0 (현재 10.621)
- ✅ Max DD <= 1.5% (현재 1.06%)

**목표 기준** (이상적):
- 🎯 Returns >= +6%
- 🎯 Win Rate >= 73%
- 🎯 Sharpe >= 13
- 🎯 Max DD <= 0.8%

### 7.3 검증 프로세스

```yaml
Phase 1 - Model Training:
  1. Train new Entry models
  2. Check F1 >= 0.40
  3. If fail → tune hyperparameters
  4. If pass → proceed to Phase 2

Phase 2 - Exit Training:
  1. Train new Exit models
  2. Check F1 >= 0.60
  3. If fail → tune labeling logic
  4. If pass → proceed to Phase 3

Phase 3 - Backtest:
  1. Run backtest with new models
  2. Check Returns >= +4.5%
  3. Check Win Rate >= 70%
  4. If all pass → Deploy to testnet
  5. If fail → rollback or iterate

Phase 4 - Testnet Validation:
  1. Deploy to testnet
  2. Monitor for 1 week
  3. Compare vs old models
  4. If successful → Deploy to mainnet
```

---

## 8. 타임라인 (Timeline)

### Week 1: Entry 모델 재훈련
```
Day 1-2: LONG Entry 새 라벨링 구현 및 훈련
Day 3-4: SHORT Entry 새 라벨링 구현 및 훈련
Day 5: Entry 모델 성능 검증
Day 6-7: Entry 모델만으로 백테스트
```

### Week 2: Exit 모델 재훈련
```
Day 8-9: Exit 라벨링 개선 구현
Day 10-11: LONG/SHORT Exit 훈련
Day 12: Exit 모델 성능 검증
Day 13-14: 통합 백테스트 (Entry + Exit)
```

### Week 3: 검증 및 배포
```
Day 15-17: 다양한 시나리오 백테스트
Day 18-19: 성능 비교 및 분석
Day 20-21: 테스트넷 배포 및 모니터링
```

---

## 9. 다음 단계 (Next Steps)

### 즉시 실행
1. ✅ 이 분석 문서 검토 및 승인
2. ⏳ Entry 라벨링 함수 구현 (create_labels_realistic)
3. ⏳ LONG Entry 모델 재훈련
4. ⏳ 성능 검증 (F1 >= 0.40)

### Week 1
- LONG Entry 재훈련 완료
- SHORT Entry 재훈련 완료
- Entry 모델 백테스트

### Week 2
- Exit 라벨링 개선
- Exit 모델 재훈련
- 통합 백테스트

### Week 3
- 성능 비교 분석
- 테스트넷 검증
- 최종 배포 결정

---

## 10. 결론

### 핵심 인사이트

1. **현재 모델의 낮은 F1은 라벨링 문제**:
   - Entry F1 15.8%: 15분/0.3% 예측 vs 4시간/3% 실제 거래
   - Exit F1 51.2%: TP/SL 무시한 라벨링

2. **백테스트 70% 승률의 비밀**:
   - 모델 정확도가 아닌 보수적 필터링 (threshold 0.7)
   - 전략 파라미터 (TP 3%, SL 1%, Max Hold 4h)
   - 수수료 0.02%의 낮은 영향

3. **개선 방향**:
   - 라벨링을 실제 거래 조건에 맞춤
   - Lookahead 15분 → 4시간
   - Threshold 0.3% → 3% (TP 일치)
   - Exit에 TP/SL 고려

### 기대 효과

**모델 성능**:
- Entry F1: 0.158 → 0.40+ (+153%)
- Exit F1: 0.512 → 0.60+ (+17%)

**백테스트 성능**:
- Returns: +4.19% → +5-7%
- Win Rate: 70.6% → 72-75%
- Sharpe: 10.621 → 12-14

### 최종 추천

**Phase 1: Entry 모델 재훈련** (우선순위 최상)
- 가장 큰 개선 여지
- F1 3배 향상 가능
- 백테스트 수익 1-2%p 향상 예상

**Phase 2: Exit 모델 개선** (우선순위 중간)
- 중간 개선 여지
- Precision 향상으로 False Positive 감소
- 백테스트 수익 0.5-1%p 향상 예상

**총 개선 예상**: Returns +1.5-3%p, Sharpe +1.5-3.5

---

**Document Version**: 1.0
**Last Updated**: 2025-10-15 01:00
**Next Review**: After Phase 1 completion (Entry model retraining)
