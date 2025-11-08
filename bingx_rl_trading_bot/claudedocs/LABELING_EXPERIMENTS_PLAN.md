# Labeling & Learning Methods Comparison Experiments

**Date**: 2025-10-14
**Goal**: 다양한 레이블링 및 학습 방법 비교 분석
**Hypothesis**: 현재 Supervised Learning의 레이블링 방식은 신뢰도가 낮아 개선 가능

---

## 🎯 실험 목표

**문제 인식:**
- 현재 방식: "다음 15분 내 0.3% 상승" = Label 1
- 문제: Stop Loss, Take Profit, 최종 P&L 무시
- 결과: "좋아 보이는 신호" ≠ "수익 나는 거래"

**실험 목표:**
1. 개선된 레이블링 방법 시도 (Realistic Labels)
2. Regression 방식 적용 (수익률 직접 예측)
3. Unsupervised Learning 활용 (Market Regime 분류)
4. 각 방법 비교 분석 및 최적 모델 선정

---

## 📊 Baseline 성능 (Phase 4 Base)

### 현재 모델
```yaml
Model: XGBoost Phase 4 Base (37 features)
Labeling: "lookahead=3, threshold=0.3%"
  - Label 1 if: 다음 3 candles(15분) 내 0.3%+ 상승
  - Label 0 otherwise

Training Performance:
  - F1 Score: 0.089
  - Precision/Recall: Balanced
  - Positive samples: ~5-10%

Backtest Performance (n=29 windows, 2-day):
  - Returns: +7.68% per 5 days (~46% monthly)
  - Win Rate: 69.1%
  - Sharpe Ratio: 11.88
  - Max Drawdown: 0.90%
  - Trade Frequency: ~15 per 5 days
  - Statistical Power: 88.3%
  - Effect Size: d=0.606 (large)

Live Performance (First Trade):
  - Trade: SHORT 0.4945 BTC @ $115,128
  - Status: Active (monitoring)
  - Entry Signal: 0.881 (88.1% confidence)
```

**Baseline 문제점:**
1. ⚠️ 레이블이 "가격 상승"만 봄 (최종 수익 무시)
2. ⚠️ SL/TP/Max Hold 시뮬레이션 없음
3. ⚠️ "0.3% 도달 → 즉시 폭락" 시나리오도 Label=1

---

## 🔬 실험 방법론

### 실험 1: Realistic Labels (P&L 기반)

**컨셉:**
```python
# 기존: 단순 가격 상승
if max_price_in_15min > entry_price * 1.003:
    label = 1

# 개선: 실제 거래 시뮬레이션
simulate_trade(entry_price, future_prices, SL=0.01, TP=0.03, max_hold=4h)
if final_pnl > 0:
    label = 1
```

**구현 계획:**
```yaml
File: scripts/experiments/train_xgboost_realistic_labels.py

Labeling Logic:
  - Lookahead: 48 candles (4 hours = Max Hold time)
  - 각 candle마다 SL/TP 체크
  - TP 도달(+3%): Label=1, break
  - SL 도달(-1%): Label=0, break
  - Max Hold 도달: Label=1 if final_pnl>0 else 0

Expected Changes:
  - More accurate labels (reflects actual trading outcome)
  - Potentially fewer positive samples (realistic outcomes)
  - Should improve win rate and reduce false positives

Expected Performance:
  - F1 Score: 0.10-0.12 (↑ 12-35%)
  - Returns: 9-11% per 5 days (↑ 17-43%)
  - Win Rate: 72-75% (↑ 3-6%)
```

**검증 방법:**
- 백테스트 (n=29 windows, 동일 조건)
- Statistical validation (Bootstrap CI, power analysis)
- Compare vs Baseline (7.68%)

---

### 실험 2: XGBoost Regression

**컨셉:**
```python
# 기존 Classification: 0 or 1
# 개선 Regression: 예상 수익률

target = simulate_trade_pnl(entry, future_prices, SL, TP, max_hold)
# target = -0.01 (SL), +0.03 (TP), +0.005 (small win), etc.

# 예측: +0.025 → LONG with high position
# 예측: -0.005 → HOLD
```

**구현 계획:**
```yaml
File: scripts/experiments/train_xgboost_regression_v2.py

Target Variable:
  - Simulate each trade outcome
  - Record final P&L percentage
  - Range: [-0.01, +0.03] (SL to TP)

Model:
  - XGBRegressor (not Classifier)
  - Same 37 features
  - Hyperparameters: adjusted for regression

Trading Logic:
  - If predicted_pnl > 0.01: LONG (강한 신호)
  - If predicted_pnl > 0.005: LONG (약한 신호)
  - Else: HOLD

Position Sizing:
  - Dynamic based on predicted_pnl
  - Higher prediction → Larger position

Expected Performance:
  - More nuanced signals (not binary)
  - Better position sizing
  - Returns: 8-10% per 5 days (↑ 4-30%)
```

**기존 코드 확인:**
- scripts/experiments/train_xgboost_regression.py (이미 존재)
- 디버깅 및 개선 필요

---

### 실험 3: Unsupervised Learning (Market Regime)

**컨셉:**
```python
# Market Regime 자동 분류
K-Means(n_clusters=4) on recent 20 candles

Cluster 0: High volatility + Bull
Cluster 1: High volatility + Bear
Cluster 2: Low volatility + Sideways
Cluster 3: Reversal patterns

# Supervised model에 regime feature 추가
feature_38 = current_market_regime
```

**구현 계획:**
```yaml
File: scripts/experiments/unsupervised_market_regime.py

Approach:
  - K-Means Clustering on rolling 20-candle windows
  - Features for clustering: returns, volatility, volume, trend
  - Identify 3-5 market regimes
  - Add regime as new feature to XGBoost

Integration:
  - Add "market_regime" feature (0-4)
  - Retrain Phase 4 Base with 38 features
  - Compare 37 vs 38 features

Expected Performance:
  - Regime-aware trading
  - Better performance in specific regimes
  - Returns: 8-9% per 5 days (↑ 4-17%)

Alternative Use:
  - Regime-specific models (separate model per regime)
  - Regime-specific thresholds
```

---

### 실험 4: RL Preparation (장기)

**컨셉:**
```python
# 현재 데이터로 RL 가능성 검증
# 실제 RL 훈련은 데이터 부족으로 연기

Validation Tasks:
  1. 데이터 충분성 검증 (60 days vs 180+ days needed)
  2. 환경 설정 검증 (Trading environment)
  3. Reward function 설계 및 시뮬레이션
  4. Supervised baseline 성능 기록 (RL 비교 대상)
```

**구현 계획:**
```yaml
File: scripts/experiments/rl_preparation_analysis.py

Tasks:
  1. RL 필요 데이터량 계산
     - Current: 60 days (17,280 candles)
     - PPO typical: 100K+ steps (190+ days)
     - Recommendation: Collect 180 days

  2. Trading Environment 검증
     - src/agent/rl_agent.py 코드 검증
     - Reward function 설계
     - State/Action space 정의

  3. Baseline 설정
     - Supervised model performance = RL 목표
     - Expected: 10-15% with 6 months data

Timeline:
  - Month 1-3: Data collection (60 → 180 days)
  - Month 4: RL training & validation
  - Month 5-6: RL fine-tuning & ensemble
```

---

## 📈 실험 실행 계획

### 순서 및 우선순위

**Phase 1: 즉시 실행 (Today - Day 1)**
```yaml
Priority: HIGH
Tasks:
  1. ✅ 실험 계획 문서화 (이 문서)
  2. 🔄 Realistic Labels 구현 및 훈련
  3. 🔄 백테스트 및 성능 비교

Expected Time: 2-3 hours
Expected Results: 9-11% returns (if successful)
```

**Phase 2: 단기 실험 (Day 2-3)**
```yaml
Priority: MEDIUM
Tasks:
  1. XGBoost Regression 구현 (기존 코드 개선)
  2. 백테스트 및 성능 비교
  3. Unsupervised Market Regime 구현
  4. 백테스트 및 성능 비교

Expected Time: 1 day
Expected Results: Identify best approach
```

**Phase 3: 종합 분석 (Day 4)**
```yaml
Priority: HIGH
Tasks:
  1. 전체 방법 비교 분석
  2. 최적 모델 선정
  3. Production 배포 계획
  4. 최종 리포트 작성

Expected Time: 3-4 hours
Deliverable: LABELING_EXPERIMENTS_RESULTS.md
```

**Phase 4: RL 준비 (Week 2+)**
```yaml
Priority: LOW (장기)
Tasks:
  1. RL 환경 검증
  2. 데이터 수집 계획
  3. Reward function 설계

Expected Time: Ongoing (background)
Timeline: Month 3-6
```

---

## 🎯 성공 기준

### 실험 성공 기준

**Realistic Labels:**
- ✅ F1 Score > 0.10 (↑ 12%+)
- ✅ Returns > 9% per 5 days (↑ 17%+)
- ✅ Win Rate > 72% (↑ 3%+)

**Regression:**
- ✅ Returns > 8% per 5 days (↑ 4%+)
- ✅ Position sizing improves returns
- ✅ Better risk management

**Unsupervised:**
- ✅ Returns > 8% per 5 days (↑ 4%+)
- ✅ Regime-specific performance clear
- ✅ Interpretable regimes

### 전체 실험 성공 기준

**Primary Goal:**
- At least ONE method beats baseline (7.68%)
- Improvement > 10% (≥8.45%)

**Secondary Goals:**
- Statistical validation (n≥29, power≥80%)
- Stable performance across windows
- Interpretable results

**Decision Criteria:**
```yaml
If best method > 9%:
  → Deploy immediately

If best method 8-9%:
  → Week 1 validation, then deploy

If best method < 8%:
  → Keep baseline, focus on LSTM/RL
```

---

## 📊 비교 메트릭

### 핵심 메트릭

**Performance Metrics:**
1. Returns per 5 days (primary)
2. Win Rate
3. Sharpe Ratio
4. Max Drawdown
5. Trade Frequency

**Model Metrics:**
1. F1 Score / R² (for regression)
2. Training stability
3. Feature importance changes
4. Prediction confidence

**Statistical Metrics:**
1. Bootstrap 95% CI
2. Effect size (Cohen's d)
3. Statistical power
4. Bonferroni p-value

### 비교 표 템플릿

| Method | Returns | Win Rate | Sharpe | F1/R² | Status |
|--------|---------|----------|--------|-------|--------|
| Baseline | 7.68% | 69.1% | 11.88 | 0.089 | ✅ Current |
| Realistic Labels | ?% | ?% | ? | ? | 🔄 Testing |
| Regression | ?% | ?% | ? | ? | ⏳ Pending |
| Unsupervised | ?% | ?% | ? | ? | ⏳ Pending |

---

## ⚠️ 리스크 및 고려사항

### 실험 리스크

**Overfitting 위험:**
- Multiple experiments → Multiple comparison problem
- Solution: Bonferroni correction, conservative thresholds

**데이터 부족:**
- 60 days only (limited)
- Solution: Bootstrap validation, conservative claims

**Implementation Bugs:**
- Complex labeling logic → bugs possible
- Solution: Unit tests, manual verification

### 프로덕션 리스크

**라이브 테스트 중단:**
- 현재 첫 거래 진행 중 (SHORT position)
- Don't interrupt for experiments
- Solution: Experiments in parallel, deploy after validation

**모델 교체 리스크:**
- New model → unknown live performance
- Solution: Week 1 paper trading validation

---

## 📝 문서화 계획

### 생성 문서

**1. LABELING_EXPERIMENTS_PLAN.md** (이 문서)
- 실험 계획 및 방법론
- Baseline 성능 기록
- 성공 기준 정의

**2. LABELING_EXPERIMENTS_RESULTS.md** (실험 후)
- 각 방법 상세 결과
- 비교 분석 및 통계
- 최종 권장사항

**3. LABELING_EXPERIMENTS_CODE_SUMMARY.md** (참고)
- 각 실험 코드 설명
- 실행 방법
- 재현 가능성 보장

---

## 🚀 Next Steps

### Immediate Actions

1. ✅ 실험 계획 문서 작성 (완료)
2. 🔄 Realistic Labels 구현
3. 🔄 훈련 및 백테스트
4. 📊 결과 비교 및 분석

### Timeline

```
Day 1 (Today):
  09:00-12:00: Realistic Labels implementation
  13:00-15:00: Training & backtesting
  15:00-17:00: Results analysis

Day 2-3:
  - Regression implementation
  - Unsupervised implementation
  - Comparative analysis

Day 4:
  - Final report
  - Model selection
  - Deployment decision
```

---

**Status**: 📋 Planning Complete → Ready for Implementation
**Next**: Implement Realistic Labels
**Expected Completion**: Day 4 (2025-10-18)
