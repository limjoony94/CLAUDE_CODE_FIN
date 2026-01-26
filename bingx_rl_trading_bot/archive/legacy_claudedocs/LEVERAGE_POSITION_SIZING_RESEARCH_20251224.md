# Leverage, Position Sizing, and Win Rate Research

**Date**: 2025-12-24
**Purpose**: 레버리지 트레이딩에서 마진, 포지션 사이즈, 승률 관계 연구

---

## 1. Position Sizing Formula (포지션 사이즈 공식)

### 기본 공식
```
Position Size = (Account Size × Risk %) ÷ Stop-Loss Distance
```

**예시**:
- Account: $5,000
- Risk: 1% = $50
- Stop-Loss: 2% ($100 거리)
- Position Size = $50 ÷ 0.02 = **$2,500**

### Effective Leverage 계산
```
Effective Leverage = Position Notional / Account Equity
```

**예시**:
- Account: $5,000
- Position: $20,000
- Effective Leverage = $20,000 / $5,000 = **4x**

### Risk Level별 포지션 사이즈

| Risk Level | Position Size (% of account) | Stop Loss Range | Use Case |
|------------|------------------------------|-----------------|----------|
| **Conservative** | 1-2% | 5-10 points | 초보자, 고변동성 |
| **Moderate** | 2-3% | 10-15 points | 경험자, 중변동성 |
| **Aggressive** | 3-5% | 15-25 points | 전문가, 저변동성 |

**Source**: [Investopedia](https://www.investopedia.com/articles/trading/09/determine-position-size.asp), [Altrady](https://www.altrady.com/crypto-trading/risk-management/determine-right-position-sizing)

---

## 2. Breakeven Win Rate (손익분기 승률)

### 핵심 공식
```
Breakeven Win Rate = 1 / (1 + Risk:Reward Ratio)
```

또는:
```
Breakeven Win Rate = SL% / (SL% + TP%)
```

### R:R Ratio별 필요 승률

| R:R Ratio | TP:SL | Breakeven Win Rate | Notes |
|-----------|-------|-------------------|-------|
| 1:1 | 1%:1% | **50.0%** | 높은 승률 필요 |
| 1.5:1 | 1.5%:1% | **40.0%** | 일반적 |
| **1.67:1** | **2.5%:1.5%** | **37.5%** | **MS_ChoCH Bot** |
| 2:1 | 2%:1% | **33.3%** | 권장 최소 |
| 2.5:1 | 2.5%:1% | **28.6%** | 좋음 |
| 3:1 | 3%:1% | **25.0%** | 최적 |
| 4:1 | 4%:1% | **20.0%** | 트렌드 추종 |
| 5:1 | 5%:1% | **16.7%** | 스윙 트레이딩 |

### MS_ChoCH Bot 분석
```
TP: 2.5%, SL: 1.5%
R:R Ratio = 2.5 / 1.5 = 1.67:1
Breakeven Win Rate = 1.5 / (1.5 + 2.5) = 37.5%
Actual Win Rate: 58% → Profitable ✅
Edge: 58% - 37.5% = +20.5%p 우위
```

**Source**: [Tradeciety](https://tradeciety.com/how-to-use-reward-risk-ratio-guide), [LuxAlgo](https://www.luxalgo.com/blog/win-rate-and-riskreward-connection-explained/)

---

## 3. Expectancy Formula (기대값 공식)

### 핵심 공식
```
Expectancy = (Win Rate × Average Win) - (Loss Rate × Average Loss)
```

### 예시 계산

**MS_ChoCH Bot**:
```
Win Rate: 58%
Loss Rate: 42%
Average Win: 2.5% (TP)
Average Loss: 1.5% (SL)

Expectancy = (0.58 × 2.5) - (0.42 × 1.5)
           = 1.45 - 0.63
           = +0.82% per trade
```

### Expectancy 해석

| Expectancy | 해석 | 권장 |
|------------|------|------|
| < 0 | 손실 시스템 | ❌ 거래 금지 |
| 0 - 0.2% | 약한 에지 | ⚠️ 개선 필요 |
| 0.2 - 0.5% | 적정 에지 | ✅ 운영 가능 |
| 0.5 - 1.0% | 좋은 에지 | ✅ 권장 |
| > 1.0% | 강한 에지 | ✅ 최적 |

**MS_ChoCH Bot: +0.82% per trade → 좋은 에지 ✅**

**Source**: [The Shmuts](https://theshmuts.substack.com/p/the-math-of-trading), [LinkedIn](https://www.linkedin.com/posts/valentin-nemesh-270ba5251_the-expectancy-formula-why-math-is-the-real-activity-7365407872173723648-H9PB)

---

## 4. Kelly Criterion (켈리 기준)

### 핵심 공식
```
f* = (bp - q) / b

where:
  f* = optimal fraction of capital to bet
  b  = net odds (win amount / loss amount)
  p  = probability of winning
  q  = probability of losing (1 - p)
```

### Alternative Formula
```
Kelly % = Win Rate - (1 - Win Rate) / Profit Factor
```

### MS_ChoCH Bot Kelly 계산
```
p = 0.58 (win rate)
q = 0.42 (loss rate)
b = 2.5 / 1.5 = 1.67 (profit ratio)

f* = (1.67 × 0.58 - 0.42) / 1.67
   = (0.969 - 0.42) / 1.67
   = 0.549 / 1.67
   = 0.329 or 32.9%

Full Kelly: 32.9% of capital per trade
```

### Fractional Kelly (실제 적용)

| Kelly Type | % of Full Kelly | 특성 | 권장 |
|------------|-----------------|------|------|
| Full Kelly | 100% | 최대 성장, 최대 변동성 | ❌ 위험 |
| Half Kelly | 50% | 71% 수익, 38% 변동성 | ✅ 권장 |
| Quarter Kelly | 25% | 50% 수익, 10% 변동성 | ✅ 보수적 |
| Tenth Kelly | 10% | 19% 수익, 1% 변동성 | 초보자 |

**실제 적용 (MS_ChoCH)**:
```
Full Kelly: 32.9%
Half Kelly: 16.5% → 실제 권장
Quarter Kelly: 8.2% → 보수적 권장

현재 설정: 2% risk per trade (Tenth Kelly 수준) → 매우 보수적 ✅
```

### Kelly Criterion 한계
1. **정확한 확률 필요**: 실제 승률/손익 추정 어려움
2. **수수료 미반영**: 실제 비용 고려 필요
3. **변동성 무시**: 크립토의 높은 변동성 미반영
4. **심리적 부담**: Full Kelly는 심리적으로 견디기 어려움

**권장**: 10-25% of Full Kelly 사용 (Pro traders)

**Source**: [LBank](https://www.lbank.com/explore/mastering-the-kelly-criterion-for-smarter-crypto-risk-management), [Investopedia](https://www.investopedia.com/articles/trading/04/091504.asp), [tastylive](https://www.tastylive.com/news-insights/kelly-criterion-explained-smarter-position-sizing-traders)

---

## 5. Leverage Rules (레버리지 규칙)

### 전문가 권장 레버리지

| Trader Level | Max Leverage | Risk per Trade | Notes |
|--------------|--------------|----------------|-------|
| **Beginner** | 2-3x | 0.5-1% | 학습 기간 |
| **Intermediate** | 3-5x | 1-2% | 검증된 전략 필요 |
| **Advanced** | 5-10x | 1-2% | 철저한 리스크 관리 |
| **Professional** | 5x max | 1-2% | 대부분 5x 초과하지 않음 |

### 레버리지별 청산 거리

| Leverage | Liquidation Distance | 안전 마진 (50%) |
|----------|---------------------|-----------------|
| 2x | 50% | 25% |
| 3x | 33.3% | 16.7% |
| **4x** | **25%** | **12.5%** (MS_ChoCH) |
| 5x | 20% | 10% |
| 10x | 10% | 5% |
| 20x | 5% | 2.5% |
| 50x | 2% | 1% |
| 100x | 1% | 0.5% |

### Margin Type 선택

| Type | 장점 | 단점 | 권장 |
|------|------|------|------|
| **Isolated** | 손실 제한, 포지션별 관리 | 청산 빈번 | ✅ 권장 |
| Cross | 청산 방지, 전체 활용 | 전 계좌 위험 | ⚠️ 주의 |

**Source**: [Debut Infotech](https://www.debutinfotech.com/blog/risk-management-in-crypto-derivatives), [IG](https://www.ig.com/ae/trading-need-to-knows/leverage-trading-crypto)

---

## 6. Loss Recovery Math (손실 복구 수학)

### 손실 복구 공식
```
Required Gain % = Loss % ÷ (1 - Loss %)
```

### 손실별 복구 필요율

| Loss % | Required Gain to Recover | Recovery Difficulty |
|--------|-------------------------|---------------------|
| 5% | 5.3% | Easy |
| 10% | 11.1% | Easy |
| 15% | 17.6% | Moderate |
| 20% | 25.0% | Moderate |
| **25%** | **33.3%** | **Moderate** (MS_ChoCH Max DD) |
| 30% | 42.9% | Hard |
| 40% | 66.7% | Very Hard |
| 50% | 100.0% | Extremely Hard |
| 75% | 300.0% | Nearly Impossible |
| 90% | 900.0% | Impossible |

### 시사점
- **Max Drawdown 20% 이하 유지 권장**
- MS_ChoCH Max DD 27.4% → 복구에 37.7% 필요 (관리 가능)
- 50% 이상 손실 시 사실상 복구 불가능

**Source**: [The Shmuts](https://theshmuts.substack.com/p/the-math-of-trading), [Pocket Option](https://pocketoption.com/blog/en/knowledge-base/learning/break-even-trading/)

---

## 7. Risk Management Framework

### Target Metrics

| Metric | Conservative | Moderate | Aggressive |
|--------|--------------|----------|------------|
| **Sharpe Ratio** | > 2.0 | > 1.5 | > 1.0 |
| **Max Drawdown** | < 15% | < 20% | < 30% |
| **Win Rate** | > 60% | > 50% | > 40% |
| **Profit Factor** | > 2.0 | > 1.5 | > 1.2 |
| **Risk per Trade** | 0.5-1% | 1-2% | 2-3% |

### MS_ChoCH Bot vs Targets

| Metric | MS_ChoCH | Target | Status |
|--------|----------|--------|--------|
| Sharpe Ratio | ~1.5 | > 1.5 | ⚠️ Borderline |
| Max Drawdown | 27.4% | < 30% | ⚠️ Borderline |
| Win Rate | 58% | > 50% | ✅ Pass |
| Profit Factor | 2.23 | > 1.5 | ✅ Pass |
| Risk per Trade | 2% | 1-2% | ✅ Pass |

---

## 8. Practical Application (MS_ChoCH Bot)

### 현재 설정 분석

```yaml
Leverage:
  Exchange: 10x
  Effective: 4x
  Status: ✅ 보수적 (Pro max 5x)

Position Sizing:
  Risk per Trade: 2%
  Method: Fixed %
  Kelly Comparison: ~6% of Full Kelly (매우 보수적)
  Status: ✅ 안전

Win Rate Requirements:
  Breakeven: 37.5%
  Actual: 58%
  Edge: +20.5%p
  Status: ✅ 강한 에지

Expectancy:
  Per Trade: +0.82%
  Daily (0.95 trades): +0.78%
  Monthly (20 days): +15.6%
  Status: ✅ 우수

Max Drawdown:
  Actual: 27.4%
  Recovery Needed: 37.7%
  Status: ⚠️ 경계선 (30% 이하 권장)
```

### 권장 조정사항

| 항목 | 현재 | 권장 | 이유 |
|------|------|------|------|
| Risk per Trade | 2% | 1.5% | DD 감소 |
| Effective Leverage | 4x | 3x | 안전 마진 확대 |
| Max Daily Loss | 15% | 10% | 조기 중단 |
| Position Mode | One-Way | Maintain | OK |

### 최적화된 Kelly 기반 포지션 사이즈

```
Half Kelly Position Size:
  Full Kelly: 32.9%
  Half Kelly: 16.5%

  Account: $1,000
  Max Position: $1,000 × 16.5% × 4x leverage = $660 × 4 = $2,640

Quarter Kelly Position Size (더 안전):
  Quarter Kelly: 8.2%
  Max Position: $1,000 × 8.2% × 4x = $328 × 4 = $1,312
```

---

## 9. Key Takeaways

### 핵심 원칙

1. **Risk 1-2% per Trade**: 전문가 표준, 초보자는 0.5-1%
2. **R:R 2:1 이상**: 33% 승률만으로도 수익 가능
3. **Fractional Kelly**: Full Kelly의 10-25% 사용
4. **Max Leverage 5x**: 경험 많은 트레이더도 초과하지 않음
5. **Isolated Margin**: Cross 대신 Isolated 사용
6. **Max Drawdown 20%**: 복구 가능한 수준 유지

### MS_ChoCH Bot 결론

| 항목 | 평가 |
|------|------|
| **Position Sizing** | ✅ 보수적 (2% risk, ~6% of Kelly) |
| **Leverage** | ✅ 적정 (4x effective) |
| **Win Rate Edge** | ✅ 강함 (+20.5%p over breakeven) |
| **Expectancy** | ✅ 양호 (+0.82% per trade) |
| **Max Drawdown** | ⚠️ 경계선 (27.4%, target <25%) |

**Overall**: 수학적으로 수익 가능한 시스템. Max DD 개선 시 더 안정적.

---

## Sources

1. [Investopedia - Position Sizing](https://www.investopedia.com/articles/trading/09/determine-position-size.asp)
2. [Investopedia - Kelly Criterion](https://www.investopedia.com/articles/trading/04/091504.asp)
3. [Tradeciety - Risk Reward Ratio](https://tradeciety.com/how-to-use-reward-risk-ratio-guide)
4. [LuxAlgo - Win Rate and Risk/Reward](https://www.luxalgo.com/blog/win-rate-and-riskreward-connection-explained/)
5. [Altrady - Position Sizing](https://www.altrady.com/crypto-trading/risk-management/determine-right-position-sizing)
6. [Debut Infotech - Crypto Derivatives Risk](https://www.debutinfotech.com/blog/risk-management-in-crypto-derivatives)
7. [LBank - Kelly Criterion Crypto](https://www.lbank.com/explore/mastering-the-kelly-criterion-for-smarter-crypto-risk-management)
8. [tastylive - Kelly Criterion](https://www.tastylive.com/news-insights/kelly-criterion-explained-smarter-position-sizing-traders)
9. [The Shmuts - Math of Trading](https://theshmuts.substack.com/p/the-math-of-trading)

---

**Last Updated**: 2025-12-24 KST
