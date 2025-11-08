# 실행 가능한 조치 사항 - 수익성 개선

**Date**: 2025-10-10
**Status**: 🚀 Ready for Action
**Based on**: PROFITABILITY_CRITICAL_ANALYSIS.md

---

## 🎯 현재 상태 요약

**거래 판단 모듈 수익성**: ❌ **없음**
- Conservative (최선): vs B&H **-0.66%** (p=0.41, 유의성 없음)
- VIP 계정 시: **+0.19%** (여전히 p=0.41)
- Bull 시장: **-4.45%** (시스템적 실패)

**근본 원인**:
1. Transaction Costs: 1.28% (가장 큰 장벽)
2. XGBoost F1-Score 0.34 (66% False signals)
3. Bull Market Detection 실패
4. Short-term Prediction 한계

---

## ✅ 즉시 실행 가능 (0-3일)

### Action 1: VIP 계정 전환 검토

**목표**: Transaction costs 0.12% → 0.04% 절감

**예상 효과**:
- Conservative: -0.66% → **+0.19%**
- 비용 절감: **+0.85%p**

**실행 단계**:

```bash
1. VIP 계정 조건 확인
   - BingX VIP/Pro 거래량 요구사항 확인
   - 월간 거래량 또는 보유량 기준 확인

2. Paper Trading 검증 (1-2주)
   - Conservative 설정으로 paper trading
   - VIP 비용 (0.04%)로 시뮬레이션
   - 실제 +0.19% 재현되는지 확인

3. 소량 실전 테스트
   - IF paper trading 성공 → 소액 (전체 자금의 5-10%)으로 실전
   - 1-2주 모니터링
   - 통계적 유의성 재평가 (더 많은 샘플)

4. 점진적 확대
   - IF 성공 → 점진적으로 자금 증액
   - ELSE → 중단하고 단기 개선안으로
```

**리스크**:
- ⚠️ p=0.41 (통계적 유의성 부족)
- ⚠️ +0.19%는 작은 차이 (변동성에 매몰될 수 있음)
- ⚠️ 11 windows 샘플로는 신뢰도 낮음

**판정**: ⚠️ **신중하게 진행** (paper trading 필수)

---

## 🔧 단기 실행 가능 (1-2주)

### Action 2: Multi-timeframe Features 추가

**목표**: Bull Market Detection 개선 (-4.45% → -1% ~ 0%)

**데이터 확인**:
- ✅ 15분 데이터: `data/historical/BTCUSDT_15m.csv` (사용 가능!)
- ❌ 1시간 데이터: 없음 (수집 필요)
- ❌ 4시간 데이터: 없음 (수집 필요)

**Phase 1: 15분 Features (즉시 가능)** ⭐⭐⭐⭐⭐

```python
# 구현 계획
1. 15분 데이터 로드
2. 15분 Long-term indicators 계산:
   - EMA(50), EMA(200) on 15m
   - RSI(14) on 15m
   - MACD on 15m
   - Trend strength (15m)

3. 5분 데이터와 merge:
   - 각 5분 candle에 해당 15분 features 추가
   - ~10-15개 새로운 features

4. XGBoost 재훈련:
   - 기존 33 features + 새 10-15 features = 43-48 features
   - SMOTE + 동일 하이퍼파라미터
   - Backtest

예상 시간: 3-5일
예상 개선: Bull -4.45% → -2% ~ -1% (+2-3%p)
```

**Phase 2: 1시간/4시간 Features (데이터 수집 후)**

```python
# 데이터 수집
1. BingX API로 1시간, 4시간 데이터 수집
   - scripts/data/collect_historical.py 수정
   - 1시간 × 3000 candles (~125일)
   - 4시간 × 1500 candles (~250일)

2. Long-term features:
   - 1시간: EMA(200), Trend strength
   - 4시간: Support/Resistance levels
   - Daily: Major trend direction

3. 전체 재훈련 및 backtest

예상 시간: 7-10일 (수집 포함)
예상 개선: Bull -4.45% → -1% ~ 0% (+3-4%p)
```

**리스크**:
- ⚠️ Overfitting 가능성 (features 너무 많으면)
- ⚠️ 추가 검증 필요 (walk-forward validation)

**판정**: ✅ **강력 권장** (Bull 성과 개선 critical)

---

### Action 3: Bull Market Adaptive Threshold

**목표**: Conservative threshold를 Bull regime에서 자동 완화

**구현**:

```python
# 현재
class HybridStrategy:
    def __init__(self, ...,
                 xgb_threshold_strong=0.6,
                 xgb_threshold_moderate=0.5,
                 tech_strength_threshold=0.7):
        # Fixed thresholds

# 개선안
class AdaptiveHybridStrategy:
    def __init__(self, ...,
                 base_thresholds={'strong': 0.6, 'moderate': 0.5, 'tech': 0.7},
                 bull_thresholds={'strong': 0.45, 'moderate': 0.35, 'tech': 0.55},
                 bear_thresholds={'strong': 0.7, 'moderate': 0.6, 'tech': 0.8}):

        self.base_thresholds = base_thresholds
        self.bull_thresholds = bull_thresholds
        self.bear_thresholds = bear_thresholds

    def get_thresholds(self, regime):
        if regime == 'Bull':
            return self.bull_thresholds
        elif regime == 'Bear':
            return self.bear_thresholds
        else:
            return self.base_thresholds

    def get_signal(self, df, idx):
        # Detect current regime
        regime = self.classify_regime(df, idx)

        # Get adaptive thresholds
        thresholds = self.get_thresholds(regime)

        # Use adaptive thresholds for decision
        ...
```

**구현 단계**:
1. Regime classification 개선 (더 빠른 감지)
2. Adaptive threshold logic 추가
3. Grid search로 최적 threshold 조합 찾기
4. Backtest 검증

**예상 시간**: 3-5일
**예상 개선**: Bull -4.45% → -2% ~ 0%

**판정**: ✅ **권장** (빠른 개선 가능)

---

## 🏗️ 중기 실행 (1-2개월)

### Action 4: Order Book Features (가장 효과적)

**목표**: F1-Score 0.34 → 0.40-0.45 (승률 +3-5%p)

**구현 계획**:

```python
# Phase 1: Data Collection (1-2주)
1. WebSocket으로 Order Book streaming
   - BingX WebSocket API
   - Real-time order book depth (Bid/Ask)
   - Update frequency: 100ms

2. Features 저장:
   - Bid-Ask spread
   - Order book imbalance (Bid vol / Ask vol)
   - Volume at price levels (Top 5, Top 10)
   - Large order detection (> 1 BTC)

# Phase 2: Feature Engineering (1주)
3. Order book features 계산:
   - Spread %
   - Imbalance ratio
   - Pressure (weighted bid vs ask)
   - Momentum (order flow direction)

   ~10-15 new features

# Phase 3: Integration (1주)
4. XGBoost 재훈련:
   - 기존 features + order book features
   - Real-time prediction system
   - Production 배포

예상 시간: 1-2개월
예상 개선: 승률 45.5% → 48-50% (+2.5-4.5%p)
             vs B&H -0.66% → +0.5% ~ +1.5%
```

**기술 요구사항**:
- WebSocket streaming (실시간 데이터)
- Database (order book 저장)
- Real-time feature calculation
- Infrastructure (서버, 네트워크)

**판정**: ⭐⭐⭐⭐⭐ **최고 우선순위** (장기적으로)

---

### Action 5: Ensemble Methods

**목표**: 모델 다양성으로 안정성 향상

**구현**:

```python
# 1. 추가 모델 훈련 (1주)
models = {
    'xgboost': XGBClassifier(...),      # Current
    'lightgbm': LGBMClassifier(...),    # Similar to XGBoost
    'random_forest': RandomForestClassifier(...),  # Baseline
    'lstm': build_lstm_model(...)       # Sequence learning (optional)
}

# 2. Voting System (3일)
class EnsembleStrategy:
    def __init__(self, models, voting_threshold):
        self.models = models
        self.voting_threshold = voting_threshold

    def predict(self, X):
        predictions = []
        for model in self.models:
            pred = model.predict_proba(X)[:, 1]
            predictions.append(pred)

        # Voting
        votes = (np.array(predictions) > 0.5).sum(axis=0)
        confidence = votes / len(self.models)

        # 2/4 = Moderate, 3/4 = Strong, 4/4 = Very Strong
        return confidence

    def get_signal_strength(self, confidence):
        if confidence >= 0.75:  # 3/4 or 4/4
            return 'strong'
        elif confidence >= 0.5:  # 2/4
            return 'moderate'
        else:
            return 'weak'

# 3. Backtest (2일)
예상 시간: 2-3주
예상 개선: Sharpe +0.5-1.0, 승률 +2-3%p
```

**판정**: ⭐⭐⭐ **중간 우선순위** (안정성 향상)

---

## 🌐 장기 검토 (2-3개월)

### Action 6: Alternative Strategy Pivot

**현실 인정**: Short-term trading으로 consistently Buy & Hold 이기기 **매우 어려움**

**Option A: Risk Management Focus (Bear 방어)** ⭐⭐⭐⭐

```python
# 전략
1. Bull/Sideways: Buy & Hold (거래 안 함)
2. Bear regime 감지 시: Active trading (손실 방어)

# 현재 성과
- Bull: -4.45% (vs B&H)
- Bear: +0.30% (vs B&H) ✅ 유일한 성공!
- Sideways: -1.28%

# 개선 목표
- Bear: +0.30% → +2-3% (더 공격적 방어)
- Bull/Sideways: 거래 안 함 (B&H 그대로)

# 예상 효과
- 전체 vs B&H: -0.66% → +1-2% (Bear만 잘해도 성공)
- Transaction costs 대폭 절감 (거래 감소)
```

**판정**: ✅ **매우 실용적** (검증된 성공 영역에 집중)

---

**Option B: Volatility Trading** ⭐⭐⭐

```python
# 전략
1. Low volatility: Hold (거래 안 함)
2. High volatility: Active trading

# 구현
volatility_threshold = df['volatility'].quantile(0.7)  # Top 30%

if current_volatility > volatility_threshold:
    # Active trading with hybrid strategy
else:
    # Hold position

# 장점
- Transaction costs 절감 (거래 빈도 감소)
- 변동성 높을 때만 기회 포착
- False signals 감소
```

**판정**: ⭐⭐⭐ **검토 가치 있음**

---

## 📋 우선순위별 실행 계획

### Week 1-2: 즉시 + 단기

```markdown
Day 1-3:
  ✅ VIP 계정 조건 확인
  ✅ Paper trading 시작 (Conservative + VIP cost)
  ✅ 15분 features 구현 시작

Day 4-7:
  ✅ 15분 features 완료
  ✅ XGBoost 재훈련 (Phase 2 with 15m features)
  ✅ Backtest 실행

Day 8-14:
  ✅ Adaptive threshold 구현
  ✅ Grid search 최적 조합
  ✅ Paper trading 모니터링 (VIP)
  ✅ 결과 분석 및 보고
```

### Week 3-4: 데이터 수집

```markdown
Week 3:
  ✅ 1시간, 4시간 데이터 수집
  ✅ Long-term features 엔지니어링
  ✅ XGBoost Phase 3 훈련

Week 4:
  ✅ Backtest 검증
  ✅ Walk-forward validation
  ✅ VIP 계정 실전 테스트 (IF paper trading 성공)
```

### Month 2-3: 중장기

```markdown
Month 2:
  ✅ Order book data collection 시작
  ✅ WebSocket streaming 구현
  ✅ Real-time features 계산

Month 3:
  ✅ Order book features 통합
  ✅ Ensemble methods 구현 (optional)
  ✅ Alternative strategy 검토
  ✅ Production 시스템 최종 배포
```

---

## 🎯 성공 기준

### Minimum Viable Success

1. **vs B&H**: +0.5% 이상 (유의미한 차이)
2. **p-value**: < 0.10 (near-significant)
3. **승률**: > 48%
4. **Bull 성과**: > -2% (현재 -4.45%)
5. **거래 빈도**: 8-12 trades/window (비용 관리 가능)

### Target Success

1. **vs B&H**: +1.0% 이상
2. **p-value**: < 0.05 (statistically significant)
3. **승률**: > 50%
4. **Bull 성과**: > 0% (상승장 포착)
5. **Sharpe**: > 3.0 (risk-adjusted 우수)

### Stretch Goal

1. **vs B&H**: +2.0% 이상
2. **p-value**: < 0.01 (highly significant)
3. **승률**: > 52%
4. **All regimes**: > 0% (모든 시장 조건에서 수익)
5. **Consistency**: CV < 2.0 (안정적 성과)

---

## ⚠️ Critical Warnings

### 1. 과최적화 (Overfitting) 위험

**문제**:
- Features 너무 많이 추가 → 과최적화
- 과거 데이터에만 잘 맞고 실전 실패

**방지책**:
- Walk-forward validation
- Out-of-sample testing
- Feature importance 분석 (불필요한 features 제거)

---

### 2. 통계적 유의성 부족

**문제**:
- 11 windows 샘플 너무 적음
- p-value 신뢰도 낮음

**해결책**:
- 더 많은 데이터 수집 (50+ windows)
- Walk-forward testing
- Paper trading 장기 검증 (1-2개월)

---

### 3. Transaction Costs 재확인

**문제**:
- VIP 계정 전환해도 실제 비용 다를 수 있음
- Slippage, Market impact 고려 필요

**해결책**:
- 실제 VIP 비용 정확히 확인
- Paper trading으로 실제 비용 측정
- Slippage 시뮬레이션 추가

---

## 📊 Progress Tracking

```markdown
### Immediate Actions (Week 1-2)
- [ ] VIP 계정 조건 확인
- [ ] Paper trading 시작 (Conservative + VIP)
- [ ] 15분 features 구현
- [ ] XGBoost Phase 2 재훈련
- [ ] Adaptive threshold 구현
- [ ] Backtest 검증

### Short-term Actions (Week 3-4)
- [ ] 1시간, 4시간 데이터 수집
- [ ] Long-term features 추가
- [ ] XGBoost Phase 3 훈련
- [ ] Walk-forward validation
- [ ] VIP 실전 테스트 (IF paper 성공)

### Mid-term Actions (Month 2)
- [ ] Order book data collection
- [ ] WebSocket streaming
- [ ] Order book features engineering
- [ ] Real-time prediction system

### Long-term Review (Month 3)
- [ ] Order book integration
- [ ] Ensemble methods (optional)
- [ ] Alternative strategy evaluation
- [ ] Production deployment decision
```

---

## 🚀 Final Recommendation

### Immediate (Week 1):
✅ **Start Paper Trading with VIP costs** (Conservative setting)
✅ **Implement 15m features** (Quick win for Bull performance)

### Short-term (Week 2-4):
✅ **Adaptive thresholds** (Regime-specific optimization)
✅ **Collect 1h/4h data** (Long-term trend capture)

### Mid-term (Month 2-3):
⚠️ **Order book features** (IF immediate actions show promise)
⚠️ **ELSE: Pivot to Alternative Strategy** (Risk management focus)

### Reality Check:
> **"Short-term trading으로 consistently Buy & Hold를 이기기는 매우 어렵습니다.
> 현실적 목표는 +0.5-1.0% (vs B&H)이며, 이것도 쉽지 않습니다.
> Bear 시장 방어 (+0.30% → +2-3%)에 집중하는 것이 더 실용적일 수 있습니다."**

---

**Date**: 2025-10-10
**Status**: 🚀 Ready for Implementation
**Priority**: VIP Paper Trading + 15m Features (Week 1)
**Critical**: 통계적 유의성 확보 필수 (p < 0.10 minimum)

**"비판적 사고 기반 실행 계획: 즉시 시작하되, 통계적 검증 없이는 실전 배포 금지"** 🎯
