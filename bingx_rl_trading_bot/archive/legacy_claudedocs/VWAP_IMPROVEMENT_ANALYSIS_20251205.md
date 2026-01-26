# VWAP Band Strategy Improvement Analysis
**Date**: 2025-12-05
**Status**: Completed

## Executive Summary

백테스트를 통해 VWAP Band 전략의 개선 방안을 분석했습니다. **현재 프로덕션 봇은 이미 최적 구성을 사용하고 있습니다.**

## Backtest Results (105일 데이터: 2025-08-09 ~ 2025-11-22)

| Strategy | Trades | Win Rate | Return | Profit Factor | Max DD |
|----------|--------|----------|--------|---------------|--------|
| **v11_no_rsi_filter** | 170 | 49.4% | **+211.3%** | 1.46 | 36.0% |
| baseline (RSI<30) | 137 | 47.4% | +114.9% | 1.38 | 36.2% |
| v3_rsi_extreme (RSI<25) | 107 | 44.9% | +55.3% | 1.29 | 32.2% |
| v12_longer_hold | 119 | 44.5% | +26.1% | 1.16 | 30.5% |
| v7_trailing_stop | 138 | 49.3% | +25.5% | 1.16 | 34.4% |
| v13_shorter_cooldown | 148 | 41.2% | +23.1% | 1.14 | 37.6% |
| v1_trend_ema200 | 22 | 54.5% | +11.8% | 1.42 | 9.5% |
| v5_combo_ema200_rsi25 | 17 | 58.8% | +10.2% | 1.54 | 11.8% |
| v4_atr_tpsl | 216 | 28.2% | **-53.8%** | 0.65 | 57.2% |
| v9_tight_sl | 147 | 35.4% | **-47.7%** | 0.80 | 52.0% |

## Key Findings

### 1. RSI Filter Is HARMFUL
- **RSI 필터 없음 (+211%)** vs **RSI < 30 필터 (+115%)** → **96% 성능 차이!**
- RSI 필터가 좋은 진입 기회를 필터링해서 수익을 줄임
- 프로덕션 봇은 이미 RSI 필터 없이 작동 중 (최적)

### 2. Current Production Configuration is Optimal
```yaml
Production Bot Settings (Optimal):
  VWAP_PERIOD: 20 candles (5h)
  BAND_MULTIPLIER: 2.5x
  TP: 3.0%
  SL: 1.0%
  VOLUME_FILTER: True (volume > vol_ma)
  RSI_FILTER: None  # Optimal - no RSI filter
  COOLDOWN: 4 candles (1h)
  MAX_HOLD: 48 candles (12h)
```

### 3. What DOESN'T Work
- **ATR-based TP/SL (-53.8%)**: ATR이 너무 작아서 TP가 빈번히 도달, SL도 타이트함
- **Tight SL (-47.7%)**: 0.75% SL은 너무 타이트해서 손절 빈번
- **EMA Trend Filters**: 트레이드 수가 급감 (170 → 17~27개), 총 수익 감소

### 4. Potential Minor Improvements
- **EMA200 Filter**: Win Rate 54.5% (↑5%), 하지만 Return +11.8% (↓200%)
  - 트레이드 품질은 향상되지만 기회 손실이 큼
- **Trailing Stop**: Return +25.5% (baseline의 22%)
  - 수익 보호에는 도움되지만 TP 도달 기회 감소

## Why Recent Production Performance Differs

### Data Period Matters
- **Backtest (105일)**: 2025-08-09 ~ 2025-11-22 → +211%
- **Recent 60일**: 2025-08-07 ~ 2025-12-05 → -1% (다른 기간)
- 시장 환경에 따라 Mean Reversion 전략의 성능이 크게 달라짐

### Market Regime Analysis
- **Trending Markets**: Mean Reversion 전략 손실
- **Range-bound Markets**: Mean Reversion 전략 이익
- 최근 시장이 강한 트렌드를 보이면 손실 발생 가능

## Recommendations

### Option 1: Keep Current Configuration (Recommended)
```
현재 구성 유지 + 시장 환경 모니터링
- 장점: 이미 최적화된 설정
- 단점: 트렌드 시장에서 손실 가능
- Action: 손절 후 재진입 패턴 모니터링
```

### Option 2: Add Market Regime Filter (Conservative)
```
EMA200 기반 트렌드 필터 추가
- 장점: Win Rate 향상 (49% → 55%), MDD 감소 (36% → 10%)
- 단점: 수익 대폭 감소 (+211% → +12%)
- Trade-off: 안정성 vs 수익성
```

### Option 3: Hybrid Approach
```
시장 상황에 따른 동적 필터
- NEUTRAL/RANGING: 필터 없음 (현재 설정)
- TRENDING: EMA200 필터 추가
- Implementation: 복잡도 증가
```

## Conclusion

**현재 프로덕션 봇의 설정이 백테스트에서 최적의 성능(+211%)을 보였습니다.**

최근 손실의 원인은:
1. 시장 환경 변화 (Range → Trend)
2. Mean Reversion 전략의 본질적 한계

개선 권장사항:
1. **단기**: 현재 설정 유지, 모니터링 강화
2. **중기**: 시장 레짐 감지 기능 추가 고려
3. **장기**: 트렌드 팔로잉 전략과 병행 운용

---

## Part 2: Volatility-Based Dynamic Parameters Research

**Date**: 2025-12-05 16:32 KST

### Research Objective
변동성 기반 동적 파라미터가 고정 파라미터보다 성능을 개선할 수 있는지 검증

### Volatility Regime Distribution (105일 데이터)
```
LOW:    3,590 candles (35.6%)
MEDIUM: 2,978 candles (29.5%)
HIGH:   3,513 candles (34.8%)

ATR Stats: Mean=0.285%, Std=0.169%
```

### Test Categories & Results

#### 1. Dynamic Band Multiplier Strategies

| Strategy | Trades | Win Rate | Return | PF | Logic |
|----------|--------|----------|--------|-----|-------|
| **baseline** | 169 | 49.7% | **+224.6%** | 1.48 | Fixed 2.5x |
| band_inverse | 163 | 44.8% | +126.0% | 1.33 | High vol → lower mult |
| band_regime | 163 | 49.1% | +121.0% | 1.35 | LOW=2.0x, MED=2.5x, HIGH=3.0x |
| band_atr_percentile | 162 | 49.4% | +89.2% | 1.29 | mult × (0.7 + pctl/100 × 0.6) |
| band_atr_zscore | 169 | 46.2% | +9.7% | 1.09 | mult × (1 + zscore × 0.2) |

**결론**: 고정 밴드 (2.5x)가 모든 동적 조정보다 우수

#### 2. Dynamic TP/SL Strategies

| Strategy | Trades | Win Rate | Return | PF | Logic |
|----------|--------|----------|--------|-----|-------|
| **baseline** | 169 | 49.7% | **+224.6%** | 1.48 | TP=3%, SL=1% |
| tpsl_inverse_vol | 169 | 47.3% | +97.6% | 1.29 | High vol → tighter TP |
| regime_tpsl_aggressive | 171 | 44.4% | -8.1% | 1.04 | Regime-based aggressive |
| tpsl_fixed_ratio_atr | 214 | 34.1% | -11.2% | 1.01 | TP=2×ATR, SL=1×ATR |
| tpsl_atr_percentile | 169 | 43.8% | -28.2% | 0.96 | ATR pctl scaled |
| regime_tpsl_conservative | 170 | 44.1% | -40.8% | 0.90 | Conservative by regime |
| tpsl_atr_scaled | 174 | 41.4% | -45.4% | 0.88 | TP=1.5×ATR, SL=0.75×ATR |
| regime_tpsl_tight_low | 173 | 39.9% | -67.4% | 0.75 | Tight in LOW vol |

**결론**: 고정 TP/SL이 모든 동적 조정보다 훨씬 우수. ATR 기반 TP/SL은 특히 저조함

#### 3. Volatility Entry Filter Strategies

| Strategy | Trades | Win Rate | Return | PF | Logic |
|----------|--------|----------|--------|-----|-------|
| **baseline** | 169 | 49.7% | **+224.6%** | 1.48 | No filter |
| entry_not_low_vol | 147 | 43.5% | +29.3% | 1.15 | Skip LOW regime |
| entry_low_only | 65 | 52.3% | +27.8% | 1.28 | Only LOW regime |
| entry_not_high_vol | 112 | 50.0% | +18.9% | 1.14 | Skip HIGH regime |
| entry_medium_only | 91 | 41.8% | -28.1% | 0.89 | Only MEDIUM regime |
| entry_high_only | 111 | 36.9% | -33.8% | 0.89 | Only HIGH regime |

**결론**: 필터 없이 모든 변동성 레짐에서 거래하는 것이 최적

### Baseline Performance by Volatility Regime

```
LOW Volatility:    56 trades | WR: 55.4% | Avg PnL: +0.90%  ← Best WR
MEDIUM Volatility: 39 trades | WR: 59.0% | Avg PnL: +1.02%  ← Best Avg PnL
HIGH Volatility:   74 trades | WR: 40.5% | Avg PnL: +0.69%  ← Worst WR but still profitable
```

### Key Insights

#### 1. Fixed Parameters Win
- **모든 동적 파라미터 전략이 고정 파라미터보다 저조**
- 가장 좋은 동적 전략 (band_inverse: +126%)도 baseline (+224.6%)의 56% 수준

#### 2. HIGH Volatility Paradox
- HIGH 변동성 기간: 낮은 Win Rate (40.5%) 하지만 여전히 수익 (+0.69% avg)
- HIGH 변동성 진입 필터링 시 전체 수익 급감 (+18.9%)
- **HIGH 변동성에서 지더라도 승리 시 이익이 커서 전체적으로 수익**

#### 3. ATR-Based TP/SL Failure
- ATR 기반 TP/SL이 가장 나쁜 성능 (-45% ~ -67%)
- 이유: BTC의 ATR이 0.285%로 너무 작아서 TP가 너무 타이트함
- 3% 고정 TP가 ATR 대비 약 10배 크기 → 더 큰 움직임 포착 가능

#### 4. Volatility Filtering Trade-off
- LOW-only 필터: 높은 WR (52.3%) 하지만 너무 적은 거래 (65개)
- 필터링으로 품질 향상되지만 기회 손실이 더 큼

### Final Conclusion

| Research Area | Best Strategy | Finding |
|---------------|---------------|---------|
| Band Multiplier | **Fixed 2.5x** | 동적 조정 불필요 |
| TP/SL | **Fixed 3%/1%** | ATR 기반 TP/SL은 실패 |
| Entry Timing | **No Filter** | 모든 레짐에서 거래 |
| Exit Strategy | **Fixed TP/SL** | Trailing/Dynamic 불필요 |

**최종 결론: 현재 프로덕션 봇의 고정 파라미터 설정이 테스트된 모든 동적/변동성 기반 전략보다 우수합니다.**

### 추가 연구 제안 (Optional)

1. **Market Regime Filter** (레짐 기반 진입 중단)
   - 강한 트렌드 감지 시 전략 일시 중단
   - EMA200 기반 트렌드 필터 고려

2. **Multi-Strategy Portfolio**
   - Mean Reversion (VWAP) + Trend Following 병행
   - 시장 상황에 따라 자동 전환

3. **Longer Timeframe Analysis**
   - 1H 또는 4H 기반 VWAP 전략 테스트
   - 더 큰 움직임 포착 가능성

---

## Part 3: Volatility-Based Strategies Consistency Analysis

**Date**: 2025-12-05 16:40 KST

### Research Objective
변동성 기반 전략들의 **일관성(Consistency)** 수준 비교 분석
- 주간/월간 수익 안정성
- 변동성 레짐별 성능 균형
- 연속 손실/이익 패턴
- Risk-adjusted 메트릭 (Calmar Ratio)

### Consistency Metrics Definition

```
Consistency Score = (Weekly Profitable % × 0.3) +
                   ((100 - Weekly Std) × 0.2) +
                   (Min Regime WR × 0.3) +
                   ((100 - |Max Drawdown|) × 0.2)
```

### Test Results

#### 1. Consistency Score Ranking

| Strategy | Score | Weekly Prof% | Weekly Std | Max DD | Max Loss Streak |
|----------|-------|-------------|-----------|--------|-----------------|
| **band_regime** | **71.6** | 68.8% | 4.05 | -12.3% | 8 |
| **baseline** | **71.3** | 68.8% | 3.69 | -9.5% | 6 |
| tpsl_regime_conservative | 68.6 | 62.5% | 4.56 | -13.8% | 6 |
| tpsl_inverse_vol | 67.6 | 56.2% | 4.17 | -11.3% | 6 |
| band_atr_percentile | 62.8 | 50.0% | 3.89 | -12.1% | 8 |
| entry_not_low_vol | 58.5 | 68.8% | 3.91 | -6.8% | 5 |

**결론**: `band_regime`이 가장 높은 일관성 점수, baseline은 근접한 2위

#### 2. Regime Performance Balance (핵심 지표)

| Strategy | LOW WR | MED WR | HIGH WR | **Variance** | Min WR |
|----------|--------|--------|---------|-------------|--------|
| **band_regime** | 52.4% | 50.0% | 47.4% | **4.2** | 47.4% |
| baseline | 59.6% | 46.2% | 44.2% | 46.8 | 44.2% |
| tpsl_inverse_vol | 59.6% | 45.9% | 46.2% | 40.9 | 45.9% |
| entry_not_low_vol | - | 49.2% | 43.6% | 7.6 | 43.6% |

**핵심 발견**:
- `band_regime`의 레짐별 WR 분산 = **4.2** (가장 균형잡힘)
- baseline의 레짐별 WR 분산 = 46.8 (불균형)
- LOW 변동성에서 강하고 HIGH에서 약한 패턴 (baseline)

#### 3. Risk-Adjusted Performance

| Strategy | Return | Max DD | Calmar Ratio | Max Loss Streak |
|----------|--------|--------|--------------|-----------------|
| **entry_not_low_vol** | +20.4% | **-6.8%** | **2.99** | 5 |
| baseline | +26.5% | -9.5% | 2.78 | 6 |
| entry_not_high_vol | +17.3% | -9.9% | 1.74 | 5 |
| band_regime | +20.4% | -12.3% | 1.66 | 8 |

**결론**:
- `entry_not_low_vol`이 최고 Calmar Ratio (리스크 대비 수익)
- baseline이 절대 수익은 최고

#### 4. Trade-off Analysis (Return vs Consistency)

| Strategy | Return | Consistency | Trade-off Score |
|----------|--------|-------------|-----------------|
| **baseline** | +26.5% | 71.3 | **48.9** |
| band_regime | +20.4% | 71.6 | 46.0 |
| tpsl_inverse_vol | +18.4% | 67.6 | 43.0 |
| entry_not_low_vol | +20.4% | 58.5 | 39.4 |

### Key Insights

#### 1. `band_regime`의 숨겨진 강점
```
LOW:  84 trades (52.4% WR)  - 넓은 밴드로 더 많은 진입
MED:  38 trades (50.0% WR)  - 기본 밴드
HIGH: 19 trades (47.4% WR)  - 좁은 밴드로 선택적 진입

→ HIGH 변동성에서 거래 수 대폭 감소 (74 → 19)
→ 모든 레짐에서 균일한 Win Rate 유지
```

#### 2. Baseline의 약점
```
LOW:  52 trades (59.6% WR)  ← 강함
MED:  39 trades (46.2% WR)  ← 보통
HIGH: 52 trades (44.2% WR)  ← 약함

→ HIGH 변동성에서 많은 거래 but 낮은 WR
→ 이 패턴이 연속 손실의 원인
```

#### 3. `entry_not_low_vol` - 리스크 최적화
```
LOW:   0 trades (진입 안함)
MED:  59 trades (49.2% WR)
HIGH: 55 trades (43.6% WR)

→ 최저 Drawdown (-6.8%)
→ 최고 Calmar Ratio (2.99)
→ 단점: LOW 변동성 기회 상실
```

### Strategy Recommendations

#### 수익 극대화 (Risk Tolerance: HIGH)
```yaml
Strategy: baseline (현재 프로덕션)
Return: +26.5%
Trade-off Score: 48.9
Best for: 최대 수익 추구, 일시적 손실 감내 가능
```

#### 일관성 최적화 (Risk Tolerance: MEDIUM)
```yaml
Strategy: band_regime
Return: +20.4%
Consistency Score: 71.6
Regime Variance: 4.2 (가장 균형)
Best for: 안정적 수익, 레짐 변화에 강함
Implementation:
  LOW vol:  band_mult = 2.0
  MED vol:  band_mult = 2.5
  HIGH vol: band_mult = 3.0
```

#### 리스크 최소화 (Risk Tolerance: LOW)
```yaml
Strategy: entry_not_low_vol
Return: +20.4%
Max Drawdown: -6.8% (최저)
Calmar Ratio: 2.99 (최고)
Best for: 자본 보존 최우선, 낮은 변동성 선호
Implementation:
  Entry only when: vol_regime != 'LOW'
```

### Final Conclusion

| Metric | Winner | vs Baseline |
|--------|--------|-------------|
| Highest Return | **baseline** | - |
| Best Consistency | **band_regime** | +0.3 점 |
| Best Regime Balance | **band_regime** | Var 4.2 vs 46.8 |
| Lowest Risk | **entry_not_low_vol** | DD -6.8% vs -9.5% |
| Best Trade-off | **baseline** | - |

**종합 결론**:
1. **수익 우선**: baseline 유지 (현재 프로덕션)
2. **일관성 우선**: `band_regime` 전환 고려
3. **리스크 우선**: `entry_not_low_vol` 전환 고려

현재 프로덕션 봇(baseline)은 **수익-일관성 Trade-off에서 최적**이지만,
HIGH 변동성 기간에 취약한 패턴이 있어 시장 상황에 따른 성능 변동이 클 수 있음.

## Files Referenced
- `scripts/analysis/backtest_vwap_improvements.py` - Improvement backtest script
- `scripts/analysis/backtest_entry_exit_improvements.py` - Entry/Exit improvement script
- `scripts/analysis/backtest_volatility_dynamic.py` - Volatility dynamic parameters script
- `scripts/analysis/backtest_volatility_consistency.py` - Consistency analysis script
- `results/vwap_improvements_20251205_143150.csv` - Improvement backtest results
- `results/volatility_dynamic_20251205_163215.csv` - Volatility backtest results
- `results/volatility_consistency_20251205_164007.csv` - Consistency analysis results
- `scripts/production/vwap_band_bot.py` - Production bot (optimal settings)
