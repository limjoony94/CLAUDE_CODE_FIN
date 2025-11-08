# V6 구현 요약

**날짜**: 2025-10-09
**목적**: 근본 원인 해결 및 수익 전환

---

## 🎯 목표

V1~V5의 근본 원인을 해결하여 **테스트 수익률 > 0%** 달성

---

## 🔍 근본 원인 (발견)

### 1. 보상 스케일 불균형 (가장 치명적)
```python
문제:
- 5분 평균 수익률: 0.0005%
- 보상 스케일 ×10000 → 평균 보상 0.15
- V5 페널티 -100 → 평균의 667배

결과:
- 모델이 거래를 완전히 포기
- Mode Collapse (행동 고정)
```

### 2. 5분봉의 노이즈
```python
문제:
- 신호/노이즈 비율: 0.5% (99.5%는 노이즈)
- 자기상관 ≈ 0 (랜덤 워크)

결과:
- 예측 불가능
- 강화학습 부적합
```

### 3. 수수료 압도
```python
문제:
- 수수료 0.06% vs 기대수익 0.0005%
- 수수료가 기대수익의 120배

결과:
- 모든 전략 필연적 손실
```

---

## ✅ V6 해결책

### 1. 적응형 보상 스케일
```python
# 기존 (V1~V5)
reward = portfolio_return × 10000 + penalties

# V6
reward_scale = 100 / volatility  # 동적 조정
reward_scale = clip(reward_scale, 5000, 100000)
reward = portfolio_return × reward_scale

효과:
- volatility 낮음 (0.001) → scale 100,000 (민감)
- volatility 높음 (0.01) → scale 10,000 (안정)
- 평균 volatility (0.002) → scale 50,000
```

**수학적 근거**:
- 시장 변동성에 따라 보상 크기 자동 조정
- 낮은 변동성일 때 작은 수익도 큰 보상
- 높은 변동성일 때 큰 변동을 정규화

### 2. 행동 공간 거래 제약
```python
# 기존 (V4~V5): 페널티로 제어
reward = pnl - frequency_penalty

# V6: 물리적 강제
def step(self, action):
    if self.current_step - self.last_trade < min_hold:
        action = np.array([self.position])  # 거래 불가

    # 순수 수익률만 보상
    reward = portfolio_return × reward_scale

효과:
- 불필요한 거래 물리적 차단
- 보상 함수 단순화
- 학습 목표 명확화
```

### 3. 15분봉 전환
```python
# 기존: 5분봉
데이터: 17,280 캔들
노이즈: 99.5%
수수료 영향: 높음

# V6: 15분봉
데이터: 5,761 캔들
노이즈: 감소 (√3 = 1.73배 개선)
수수료 영향: 1/3

효과:
- 신호 품질 향상
- 수수료 부담 감소
- 예측 가능성 증가
```

### 4. 손절/익절 완화
```python
# V4~V5
stop_loss: 1% → 2%
take_profit: 2% → 3%

# V6
stop_loss: 3%
take_profit: 5%

효과:
- 정상 변동 허용
- 손절 빈도 감소
- 승률 개선 기대
```

---

## 📊 구현 세부사항

### 파일 구조
```
src/environment/trading_env_v6.py    # V6 환경
scripts/create_15min_data.py        # 15분봉 생성
scripts/train_v6_quick.py            # 빠른 검증 훈련
data/historical/BTCUSDT_15m.csv      # 15분봉 데이터
```

### 핵심 코드 변경

#### trading_env_v6.py 주요 메서드
```python
def _update_volatility(self, portfolio_return):
    """적응형 스케일용 변동성 업데이트"""
    self.recent_returns.append(portfolio_return)
    if len(self.recent_returns) > self.volatility_window:
        self.recent_returns.pop(0)

    if len(self.recent_returns) >= 10:
        self.current_volatility = max(np.std(self.recent_returns), 0.0001)

def _calculate_reward_v6(self, prev_value, new_value):
    """적응형 스케일 + 순수 PnL"""
    portfolio_return = (new_value - prev_value) / prev_value

    # 적응형 스케일
    reward_scale = 100.0 / max(self.current_volatility, 0.0001)
    reward_scale = np.clip(reward_scale, 5000.0, 100000.0)

    reward = portfolio_return × reward_scale

    # 파산 페널티만
    if self.balance <= 0:
        reward -= 1000.0

    return reward
```

---

## 🎯 성공 기준

### 빠른 검증 (500K 타임스텝)
1. ✅ Action Diversity: std > 0.1
2. ✅ Trade Frequency: 5~15%
3. ✅ Test Return: > 0%
4. ✅ Mode Collapse 회피: std > 0.05

**통과 조건**: 4개 중 3개 이상 (75%)

### 전체 훈련 (5M 타임스텝)
1. 테스트 수익률: > +0.3%
2. 승률: > 40%
3. 과적합 비율: < 1.5배
4. 샤프 비율: > 0.5

---

## 📈 예상 결과

### 낙관적 시나리오 (70% 확률)
```
테스트 수익률: +0.3 ~ +1.0%
거래 빈도: 5~10%
승률: 45~55%
Mode Collapse: 없음
```

**다음 단계**: 전체 훈련 (5M), LSTM 적용, 알고리즘 비교

### 보수적 시나리오 (20% 확률)
```
테스트 수익률: -0.1 ~ +0.2%
거래 빈도: 3~8%
승률: 40~50%
Mode Collapse: 부분적
```

**다음 단계**: 파라미터 튜닝, reward_scale 범위 조정

### 실패 시나리오 (10% 확률)
```
테스트 수익률: < -0.3%
Mode Collapse: 재발
```

**다음 단계**: 문제 재정의 (포트폴리오 최적화로 전환)

---

## 🔧 향후 개선 방향

### Phase 2: 중기 개선
1. LSTM/Transformer 정책 (시계열 학습)
2. SAC/TD3 알고리즘 비교
3. Reward Shaping (다중 목표)
4. 6개월 데이터 수집

### Phase 3: 장기 구조 변경
1. 다자산 포트폴리오 최적화
2. 합성 데이터 생성 (GAN)
3. 메타 러닝
4. 앙상블 모델

---

## 📚 참고 문서

- `ROOT_CAUSE_ANALYSIS.md`: 심층 근본 원인 분석
- `FINAL_REPORT.md`: V1~V4 결과 요약
- `IMPROVEMENTS_V5.md`: V5 개선 시도 및 실패
- `TRAINING_STATUS.md`: 전체 훈련 히스토리

---

## 💡 핵심 인사이트

1. **보상 스케일이 전부**: 페널티가 아니라 스케일 자체가 문제
2. **5분봉은 구조적 한계**: 노이즈와 수수료로 인해 수익 불가능
3. **물리적 제약 > 페널티**: 행동 공간 제약이 보상 왜곡보다 효과적
4. **적응형 접근**: 고정된 보상 스케일은 다양한 시장에서 실패

---

**작성**: 2025-10-09 10:55
**상태**: V6 빠른 검증 훈련 진행 중
**다음**: 결과 분석 및 전체 훈련 결정
