# New Strategy Research - 이중 검증 통과를 위한 전략 연구

**연구일**: 2026-01-01
**목적**: 이중 검증(Type1 + Type2 + Walk-Forward)을 통과하는 양방향 전략 개발

---

## 1. 기존 연구 실패 원인 분석

### 1.1 14개 균형 전략 실패 요약

| 전략 | LONG WR | SHORT WR | 전체 WR | Type1 | Type2 | WF |
|------|---------|----------|---------|-------|-------|-----|
| RSI Extreme + Trend | 25.3% | 60.0% | 41.9% | ❌ | ✅ | ✅ 4/6 |
| BB Bounce Filtered | 32.4% | 47.1% | ~40% | ❌ | ❌ | ✅ 4/6 |
| Multi Confirm | 27.7% | 41.0% | ~34% | ❌ | ❌ | ✅ 3/6 |
| 기타 11개 | 19-28% | 35-44% | ~30% | ❌ | ❌ | ❌ |

**핵심 문제**:
- LONG WR: 0% ~ 32.4% (모두 50% 미달)
- SHORT WR: 35.7% ~ 62.1% (일부 50% 초과)
- 90일 하락장 시장 편향으로 LONG 구조적 불리

### 1.2 시장 환경 분석

- **테스트 기간**: 90일 (5분봉 25,920개)
- **시장 특성**: 하락세 우세 (Bearish bias)
- **영향**: SHORT 신호 유리, LONG 신호 불리

---

## 2. 외부 레퍼런스 수집 결과

### 2.1 높은 승률 전략 (인터넷 조사)

| 전략 | 보고된 WR | 출처 | 특징 |
|------|----------|------|------|
| **RSI Overbought + ADX** | **74.3%** | Backtest market | RSI > 70 + ADX > 25 |
| **EMA Bearish Crossover** | **79.5%** | Backtest market | EMA 단기 < 장기 + 하락 추세 |
| **MACD + ADX** | **69-73%** | Multiple sources | MACD 교차 + ADX > 25 |
| **BB + Stochastic** | **66-72%** | Trading blogs | BB 밴드 + 스토캐스틱 확인 |
| **BB Buy/Sell + ADX** | **72.4-72.5%** | Backtest market | BB 터치 + ADX 필터 |
| **Mean Reversion** | **60-65%** | Academic research | 주식에 효과적, 암호화폐 낮음 |
| **Market Structure (ChoCH)** | **60-70%** | SMC traders | Change of Character 패턴 |
| **4-Hour Range Strategy** | **65%+** | Scalping guides | 시간대 기반 레인지 돌파 |
| **Hyper Scalper** | 85-96% | Claims only | 검증 필요 |

### 2.2 핵심 인사이트

1. **ADX 필터의 중요성**: ADX > 25-30 조합 시 일관되게 높은 WR
2. **다중 확인 (Multi-Confirmation)**: 단일 지표보다 2-3개 조합이 효과적
3. **Mean Reversion 한계**: 암호화폐는 추세추종(trending)이 더 적합
4. **Market Structure**: SMC 기반 ChoCH/BOS가 60-70% 성공률
5. **시간대 필터**: 특정 시간대(아시아/유럽/미국) 세션 필터 효과적

---

## 3. 후보 전략 설계

### 3.1 Strategy A: ADX Strong Trend Filter

**가설**: 강한 추세(ADX ≥ 30)에서만 진입하면 신호 품질 향상

```python
# LONG Entry
conditions_long = (
    (adx >= 30) &                    # 강한 추세
    (di_plus > di_minus) &           # 상승 추세
    (close > ema_50) &               # 가격 > EMA50
    (rsi > 50) & (rsi < 70)          # RSI 모멘텀 확인
)

# SHORT Entry
conditions_short = (
    (adx >= 30) &                    # 강한 추세
    (di_minus > di_plus) &           # 하락 추세
    (close < ema_50) &               # 가격 < EMA50
    (rsi < 50) & (rsi > 30)          # RSI 모멘텀 확인
)
```

**파라미터**:
- TP: 2.0%, SL: 1.5% (R:R 1.33:1)
- ADX Period: 14
- EMA Period: 50

---

### 3.2 Strategy B: Multi-Confirmation Momentum

**가설**: 3개 이상 지표 확인 시 높은 신뢰도

```python
# LONG Entry (최소 3개 확인)
score_long = (
    (macd_hist > 0).astype(int) +           # MACD 양수
    (rsi > 50).astype(int) +                 # RSI > 50
    (close > ema_20).astype(int) +           # 가격 > EMA20
    (close > bb_middle).astype(int) +        # 가격 > BB 중간선
    (adx > 20).astype(int)                   # ADX > 20
)
conditions_long = (score_long >= 4) & (close > ema_100)

# SHORT Entry
score_short = (
    (macd_hist < 0).astype(int) +
    (rsi < 50).astype(int) +
    (close < ema_20).astype(int) +
    (close < bb_middle).astype(int) +
    (adx > 20).astype(int)
)
conditions_short = (score_short >= 4) & (close < ema_100)
```

**파라미터**:
- TP: 2.5%, SL: 1.5% (R:R 1.67:1)
- 최소 확인 수: 4/5

---

### 3.3 Strategy C: Bollinger + Stochastic Reversal

**가설**: BB 극단값 + Stochastic 확인으로 반전 포착

```python
# Stochastic 계산
stoch_k = ((close - low_14) / (high_14 - low_14)) * 100
stoch_d = stoch_k.rolling(3).mean()

# LONG Entry (과매도 반전)
conditions_long = (
    (close <= bb_lower) &            # BB 하단 터치
    (stoch_k < 20) &                 # Stochastic 과매도
    (stoch_k > stoch_d) &            # K가 D 상향 돌파
    (close > ema_200)                # 장기 상승 추세
)

# SHORT Entry (과매수 반전)
conditions_short = (
    (close >= bb_upper) &            # BB 상단 터치
    (stoch_k > 80) &                 # Stochastic 과매수
    (stoch_k < stoch_d) &            # K가 D 하향 돌파
    (close < ema_200)                # 장기 하락 추세
)
```

**파라미터**:
- TP: 1.5%, SL: 1.0% (R:R 1.5:1)
- BB Period: 20, StdDev: 2.0
- Stochastic: 14, 3, 3

---

### 3.4 Strategy D: VWAP + Volume Spike

**가설**: VWAP 교차 + 거래량 급증으로 기관 매집 포착

```python
# Volume Spike 감지
volume_ma = volume.rolling(20).mean()
volume_spike = volume > (volume_ma * 1.5)

# LONG Entry
conditions_long = (
    (close > vwap) &                 # VWAP 위
    (close.shift(1) <= vwap.shift(1)) &  # 이전에 VWAP 아래
    (volume_spike) &                 # 거래량 급증
    (close > ema_50)                 # 단기 상승 추세
)

# SHORT Entry
conditions_short = (
    (close < vwap) &                 # VWAP 아래
    (close.shift(1) >= vwap.shift(1)) &  # 이전에 VWAP 위
    (volume_spike) &                 # 거래량 급증
    (close < ema_50)                 # 단기 하락 추세
)
```

**파라미터**:
- TP: 2.0%, SL: 1.5% (R:R 1.33:1)
- Volume MA: 20
- Spike Threshold: 1.5x

---

### 3.5 Strategy E: EMA Triple Cross Momentum

**가설**: 3중 EMA 정렬 + MACD 확인으로 강한 추세 진입

```python
# LONG Entry (상승 정렬)
conditions_long = (
    (ema_8 > ema_21) &               # 단기 > 중기
    (ema_21 > ema_55) &              # 중기 > 장기
    (macd_hist > 0) &                # MACD 양수
    (macd_hist > macd_hist.shift(1)) &  # MACD 상승 중
    (adx > 20)                       # 추세 존재
)

# SHORT Entry (하락 정렬)
conditions_short = (
    (ema_8 < ema_21) &
    (ema_21 < ema_55) &
    (macd_hist < 0) &
    (macd_hist < macd_hist.shift(1)) &
    (adx > 20)
)
```

**파라미터**:
- TP: 3.0%, SL: 2.0% (R:R 1.5:1)
- EMA: 8, 21, 55
- MACD: 12, 26, 9

---

### 3.6 Strategy F: RSI Divergence + Trend

**가설**: RSI 다이버전스 + 추세 필터로 반전 포착

```python
# Price와 RSI 다이버전스 감지
def detect_bullish_divergence(price, rsi, lookback=10):
    price_low = price.rolling(lookback).min()
    rsi_low = rsi.rolling(lookback).min()
    # 가격은 낮아지는데 RSI는 높아지면 = Bullish Divergence
    return (price <= price_low) & (rsi > rsi_low)

# LONG Entry (Bullish Divergence)
conditions_long = (
    detect_bullish_divergence(close, rsi) &
    (close > ema_100) &              # 상승 추세
    (adx > 15)                       # 최소 추세 강도
)

# SHORT Entry (Bearish Divergence)
conditions_short = (
    detect_bearish_divergence(close, rsi) &
    (close < ema_100) &
    (adx > 15)
)
```

**파라미터**:
- TP: 3.0%, SL: 1.5% (R:R 2:1)
- Divergence Lookback: 10
- RSI Period: 14

---

### 3.7 Strategy G: Range Breakout + Time Filter

**가설**: 아시아 세션 레인지 돌파로 유럽/미국 세션 추세 진입

```python
# 아시아 세션 (UTC 0-8) 레인지 계산
def get_asian_range(df):
    asian_mask = (df.index.hour >= 0) & (df.index.hour < 8)
    asian_high = df.loc[asian_mask, 'high'].max()
    asian_low = df.loc[asian_mask, 'low'].min()
    return asian_high, asian_low

# LONG Entry (상단 돌파)
conditions_long = (
    (close > asian_high) &           # 아시아 고점 돌파
    (df.index.hour >= 8) &           # 유럽 세션 이후
    (df.index.hour < 20) &           # 미국 세션까지
    (adx > 20)                       # 추세 존재
)

# SHORT Entry (하단 돌파)
conditions_short = (
    (close < asian_low) &
    (df.index.hour >= 8) &
    (df.index.hour < 20) &
    (adx > 20)
)
```

**파라미터**:
- TP: 2.0%, SL: 1.5% (R:R 1.33:1)
- Asian Session: UTC 0-8
- Trading Session: UTC 8-20

---

### 3.8 Strategy H: Smart Money Concept (ChoCH)

**가설**: Market Structure 변화 감지로 추세 전환 포착

```python
# Swing High/Low 감지 (미래 데이터 사용 금지!)
def detect_swing_high(high, lookback=5):
    # 현재 high가 lookback 기간 중 최고점
    return high == high.rolling(lookback).max()

def detect_swing_low(low, lookback=5):
    return low == low.rolling(lookback).min()

# Change of Character 감지
# ChoCH = 이전 swing high/low 돌파

# LONG Entry (Bullish ChoCH)
conditions_long = (
    (close > prev_swing_high) &      # 이전 swing high 돌파
    (prev_trend == 'bearish') &      # 이전 하락 추세
    (volume_spike) &                 # 거래량 확인
    (adx > 20)
)

# SHORT Entry (Bearish ChoCH)
conditions_short = (
    (close < prev_swing_low) &
    (prev_trend == 'bullish') &
    (volume_spike) &
    (adx > 20)
)
```

**파라미터**:
- TP: 3.0%, SL: 2.0% (R:R 1.5:1)
- Swing Lookback: 5 (과거만)
- Volume Spike: 1.3x

---

### 3.9 Strategy I: ATR Volatility Breakout

**가설**: 변동성 돌파로 강한 모멘텀 진입

```python
# ATR 기반 돌파 레벨
atr = df['atr_14']
breakout_long = df['open'] + (atr * 1.5)
breakout_short = df['open'] - (atr * 1.5)

# LONG Entry
conditions_long = (
    (high >= breakout_long) &        # ATR 돌파
    (close > ema_50) &               # 상승 추세
    (adx > 25)                       # 강한 추세
)

# SHORT Entry
conditions_short = (
    (low <= breakout_short) &
    (close < ema_50) &
    (adx > 25)
)
```

**파라미터**:
- TP: 2.5%, SL: 1.5% (R:R 1.67:1)
- ATR Multiplier: 1.5
- ATR Period: 14

---

### 3.10 Strategy J: Hybrid Adaptive

**가설**: 시장 상황에 따라 Mean Reversion / Trend Following 전환

```python
# 시장 레짐 판단
def get_market_regime(df):
    volatility = df['atr_14'] / df['close']
    trend_strength = df['adx']

    if trend_strength > 30:
        return 'trending'
    elif volatility < 0.01:  # 1% 미만 변동성
        return 'ranging'
    else:
        return 'mixed'

# Trending 시장: 추세 추종
if regime == 'trending':
    # EMA 정렬 + MACD 확인
    conditions_long = (ema_20 > ema_50) & (macd_hist > 0)
    conditions_short = (ema_20 < ema_50) & (macd_hist < 0)

# Ranging 시장: Mean Reversion
elif regime == 'ranging':
    # BB 밴드 반전
    conditions_long = (close < bb_lower) & (rsi < 30)
    conditions_short = (close > bb_upper) & (rsi > 70)
```

**파라미터**:
- Trending: TP 3.0%, SL 2.0%
- Ranging: TP 1.5%, SL 1.0%
- Regime Threshold: ADX 30

---

## 4. 검증 계획

### 4.1 이중 검증 기준 (재확인)

| 검증 | 조건 | 기준 |
|------|------|------|
| **Type 1** | Signal Quality | WR ≥ 50% AND EV > 0 |
| **Type 2** | Actual Trading | Total PnL > 0 |
| **Walk-Forward** | Consistency | ≥ 50% 윈도우 수익 (3/6) |

### 4.2 추가 검증 기준 (선택)

- **양방향 수익**: LONG PnL > 0 AND SHORT PnL > 0
- **방향별 WR**: LONG WR ≥ 45% AND SHORT WR ≥ 45%
- **Max Drawdown**: < 30%
- **거래 빈도**: 월 10회 이상

### 4.3 검증 우선순위

1. **Strategy A** (ADX Strong Trend) - 기존 ADX 연구 기반
2. **Strategy B** (Multi-Confirmation) - 다중 확인 접근
3. **Strategy E** (EMA Triple Cross) - 추세 정렬 확인
4. **Strategy C** (BB + Stochastic) - 반전 전략
5. **Strategy H** (ChoCH) - Market Structure 기반
6. 나머지 5개 전략

---

## 5. 다음 단계

1. ✅ 레퍼런스 수집 완료
2. ✅ 후보 전략 설계 완료 (10개)
3. ⏳ 이중 검증 스크립트 작성
4. ⏳ 검증 실행 및 결과 분석
5. ⏳ 통과 전략 선정 및 봇 구현

---

## 6. 참고 자료

- Mean Reversion: Academic research on stock/crypto differences
- Market Structure: Smart Money Concepts (SMC) trading
- VWAP Analysis: Institutional trading patterns
- Multi-indicator: Backtest.market statistical analysis
- Time-based: Session-specific volatility patterns

---

**작성자**: Claude AI Assistant
**버전**: 1.0
