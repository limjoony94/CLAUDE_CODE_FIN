# Threshold 0.80 모델 개량 계획
**Date**: 2025-10-30 16:00 KST
**Status**: 🎯 IMPROVEMENT PLAN - TARGETING 75%+ WIN RATE

---

## 📊 현재 성능 (Threshold 0.80 Baseline)

### 전체 성과 (540일, 108 windows)

```yaml
Total Return: +73.4%
Average Win Rate: 72.3%
Total Trades: 2,506
Trades per Day: 4.6

Distribution:
  LONG: 1,548 (61.8%)
  SHORT: 958 (38.2%)

Performance Tiers:
  Strong Windows (WR ≥80%): 49 (45.4%) - Avg WR 92.14%
  Weak Windows (WR <60%): 28 (25.9%) - Avg WR 40.57%
  Medium Windows: 31 (28.7%)
```

### 강점 분석

✅ **우수한 점**:
1. **Zero Loss Windows**: 108개 Windows 모두 플러스 수익
2. **High Win Rate**: 72.3% 평균 승률 (Enhanced Baseline 90.5%보다 낮지만 양호)
3. **Strong Performance in 45%**: 49개 Windows에서 80%+ 승률
4. **Balanced Distribution**: LONG 61.8%, SHORT 38.2% (적절한 균형)
5. **Consistent Profitability**: 손실 Windows 0개

### 약점 분석

⚠️ **개선 필요 영역**:

1. **Weak Windows - 28개 (25.9%)**:
   - 평균 승률: 40.57% (매우 낮음)
   - 평균 거래 수: 44회 (과다 거래)
   - 평균 수익: 14.67% (낮은 수익)
   - **근본 원인**: 낮은 승률에도 불구하고 과다 거래

2. **Win Rate Gap**:
   - 강한 Windows: 92.14%
   - 약한 Windows: 40.57%
   - **Gap**: 51.57%p (일관성 부족)

3. **Trading Frequency in Weak Periods**:
   - 약한 Windows에서 평균 44회 거래
   - 강한 Windows에서 평균 ?회 거래 (분석 필요)
   - **가설**: 약한 시장 상황에서도 과다 진입

---

## 🎯 개량 목표

### Primary Goal: 75%+ Win Rate

```yaml
Current Baseline: 72.3%
Target: 75.0%+
Improvement Needed: +2.7%p minimum

Key Strategy: Weak Windows 개선 (25.9% → 15% 이하)
  - Weak Windows 승률: 40.57% → 55%+
  - Weak Windows 거래 빈도: 44회 → 30회 이하
```

### Secondary Goals

1. **Trade Quality over Quantity**:
   - 약한 시장 상황 감지 → 거래 빈도 감소
   - 강한 시장 상황 유지 → 현재 수준 유지

2. **Consistent Performance**:
   - Win Rate Gap 감소: 51.57%p → 35%p 이하
   - 약한 Windows 비율: 25.9% → 15% 이하

3. **Maintain Strengths**:
   - Zero Loss Windows 유지
   - Strong Windows 성능 유지 (92.14%)
   - Trades/day 유지 (4.6/day 적정)

---

## 🔬 개선 전략

### Strategy 1: Market Regime Detection (시장 국면 감지)

**목적**: 약한 시장 상황 조기 감지 → 거래 빈도 감소

**구현 방법**:
1. **Volatility Regime**:
   - High Volatility (변동성 큰 시장): 거래 빈도 감소
   - Low Volatility (안정적 시장): 정상 거래
   - 지표: ATR, Bollinger Band Width

2. **Trend Strength**:
   - Weak Trend: 거래 빈도 감소
   - Strong Trend: 정상 거래
   - 지표: ADX, Trend Intensity

3. **Market Efficiency**:
   - Choppy Market (횡보장): 거래 빈도 감소
   - Trending Market (추세장): 정상 거래
   - 지표: Choppiness Index, R-squared

**Expected Impact**:
- 약한 Windows 거래 빈도: 44회 → 30회 (-32%)
- 약한 Windows 승률: 40.57% → 55%+ (+14%p)

---

### Strategy 2: Dynamic Threshold Adjustment

**목적**: 시장 상황에 따라 Entry Threshold 동적 조정

**구현 방법**:
1. **Base Threshold**: 0.80 (현재)
2. **Market-Adjusted Threshold**:
   ```python
   # Weak market conditions
   if market_regime == 'choppy' or volatility > threshold:
       entry_threshold = 0.85  # 더 높은 신호 요구
   else:
       entry_threshold = 0.80  # 정상
   ```

**Expected Impact**:
- 약한 시장: Entry 빈도 -20%
- 강한 시장: 변화 없음
- 전체 승률: +1.5%p

---

### Strategy 3: Exit Timing Improvement

**목적**: 약한 시장에서 조기 손절 → 큰 손실 방지

**구현 방법**:
1. **Market-Adjusted Exit**:
   ```python
   # Weak market: Lower ML Exit threshold (빠른 탈출)
   if market_regime == 'choppy':
       ml_exit_threshold = 0.70  # 더 빠른 Exit
   else:
       ml_exit_threshold = 0.80  # 정상
   ```

2. **Adaptive Stop Loss**:
   - Weak market: -2.5% (더 타이트)
   - Strong market: -3.0% (현재)

**Expected Impact**:
- 약한 Windows 평균 손실 감소: -15%
- 승률 개선: +0.5%p

---

### Strategy 4: Enhanced Entry Features

**목적**: Entry 모델의 판단 능력 향상

**새로운 Features 추가**:
1. **Market Regime Features** (3개):
   - `market_regime` (categorical: trending/choppy/volatile)
   - `trend_strength` (ADX-based)
   - `market_efficiency` (R-squared)

2. **Multi-Timeframe Features** (6개):
   - `price_vs_ma_15min` (15분봉 MA)
   - `price_vs_ma_1hour` (1시간봉 MA)
   - `trend_alignment` (5min/15min/1hour 정렬)

3. **Volume Profile Features** (4개):
   - `volume_profile_support` (거래량 프로파일 지지)
   - `volume_profile_resistance` (거래량 프로파일 저항)
   - `volume_imbalance` (거래량 불균형)

**Expected Impact**:
- Entry 정확도 향상: +2%p
- 약한 시장 진입 감소: -25%

---

## 📋 Implementation Roadmap

### Phase 1: Market Regime Detection (Week 1)

```yaml
Tasks:
  1. Feature Engineering:
     - Calculate ATR, Bollinger Width, ADX
     - Calculate Choppiness Index, R-squared
     - Create market_regime labels

  2. Regime Detection Logic:
     - Define thresholds for choppy/trending/volatile
     - Implement regime classification

  3. Validation:
     - Backtest with regime-aware threshold adjustment
     - Compare with Threshold 0.80 baseline
```

**Success Criteria**:
- Weak Windows 거래 빈도: 44회 → 35회 이하
- 승률: 72.3% → 73.5%+

---

### Phase 2: Dynamic Threshold System (Week 2)

```yaml
Tasks:
  1. Threshold Adjustment Logic:
     - Implement market-adjusted entry threshold
     - Implement market-adjusted exit threshold

  2. Testing:
     - Grid search optimal thresholds per regime
     - Validate on 108-window backtest

  3. Integration:
     - Add to production bot
     - Update monitoring metrics
```

**Success Criteria**:
- 승률: 73.5% → 74.5%+
- Weak Windows 비율: 25.9% → 20% 이하

---

### Phase 3: Enhanced Features + Retraining (Week 3)

```yaml
Tasks:
  1. Feature Engineering:
     - Add 13 new features (regime + multi-timeframe + volume profile)
     - Regenerate full dataset

  2. Model Retraining:
     - Retrain Entry models with enhanced features
     - 5-Fold Cross-Validation
     - Ensemble best fold

  3. Full Validation:
     - 108-window backtest with new models
     - Compare with Phase 2 results
```

**Success Criteria**:
- 승률: 74.5% → 75.0%+
- Weak Windows 비율: 20% → 15% 이하
- Win Rate Gap: 51.57%p → 35%p 이하

---

### Phase 4: Production Deployment (Week 4)

```yaml
Tasks:
  1. Deployment Preparation:
     - Update production bot with new models
     - Update monitoring thresholds
     - Create deployment documentation

  2. Week 1 Validation:
     - Monitor live performance
     - Compare actual vs expected metrics
     - Emergency rollback plan ready

  3. Performance Tracking:
     - Daily win rate tracking
     - Regime detection accuracy
     - Trade frequency by regime
```

**Success Criteria**:
- Live 승률: 73%+  (conservative -2%p from backtest)
- No catastrophic failures
- Regime detection working correctly

---

## 📊 Expected Final Performance

### Target Metrics (After All Improvements)

```yaml
Baseline (Threshold 0.80):
  Win Rate: 72.3%
  Return: +73.4% (540 days)
  Trades/day: 4.6
  Strong Windows: 45.4%
  Weak Windows: 25.9%

Target (Improved):
  Win Rate: 75.0%+ (+2.7%p)
  Return: +85%+ (540 days, +15% improvement)
  Trades/day: 4.2 (-9%, quality over quantity)
  Strong Windows: 50%+ (+4.6%p)
  Weak Windows: 15% (-10.9%p)
```

### Conservative Estimate (70% of target)

```yaml
Realistic Improvement:
  Win Rate: 74.0% (+1.7%p)
  Return: +80% (+9% improvement)
  Trades/day: 4.4 (-4%)
  Strong Windows: 48%
  Weak Windows: 18%
```

---

## 🚨 Risk Mitigation

### Overfitting Prevention

1. **Cross-Validation**: 5-Fold always
2. **Out-of-Sample Testing**: Last 20% holdout
3. **Walk-Forward Validation**: Ensure temporal validity
4. **Conservative Deployment**: Start with Phase 1 only

### Rollback Plan

```yaml
Trigger Conditions:
  - Live Win Rate < 68% (for 7 days)
  - Catastrophic losses > 10% in single day
  - System errors > 5% of trades

Rollback Steps:
  1. Stop bot immediately
  2. Revert to Threshold 0.80 baseline models
  3. Investigate failure cause
  4. Re-evaluate improvement strategy
```

---

## 📝 Summary

**Best Approach**: Gradual improvement with validation gates

1. **Phase 1**: Market Regime Detection → +1.2%p win rate
2. **Phase 2**: Dynamic Thresholds → +1.0%p win rate
3. **Phase 3**: Enhanced Features → +0.5%p win rate

**Total Expected**: +2.7%p win rate (72.3% → 75.0%)

**Timeline**: 4 weeks (1 week per phase + deployment)

**Next Step**: Implement Phase 1 (Market Regime Detection)
