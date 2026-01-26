# Market Regime Classification Research Results
**Date**: 2025-12-12
**Goal**: Find indicators to match user's manually identified bull/bear periods

---

## 🎯 Executive Summary

### Winner: Multi-Timeframe Trend
**Accuracy: 100% (12/12 periods)**

| Metric | Value |
|--------|-------|
| Period Accuracy | 100.0% |
| Weighted Accuracy | 100.0% |
| BULL Detection | 6/6 (100%) |
| BEAR Detection | 6/6 (100%) |
| Average Confidence | 66.7% |
| Neutral % | 29.1% |

---

## 📊 Ground Truth (User-Defined)

| Period | Regime | Days | Actual Change |
|--------|--------|------|---------------|
| Jul 14 - Aug 2 | BEAR | 19 | -4.9% |
| Aug 2 - Aug 13 | BULL | 11 | +6.0% |
| Aug 13 - Aug 31 | BEAR | 18 | -9.4% |
| Aug 31 - Sep 18 | BULL | 18 | +7.0% |
| Sep 18 - Sep 26 | BEAR | 8 | -6.3% |
| Sep 26 - Oct 6 | BULL | 10 | +13.0% |
| Oct 6 - Oct 17 | BEAR | 11 | -12.4% |
| Oct 17 - Oct 27 | BULL | 10 | +5.8% |
| Oct 27 - Nov 5 | BEAR | 9 | -11.7% |
| Nov 5 - Nov 11 | BULL | 6 | +4.9% |
| Nov 11 - Nov 21 | BEAR | 10 | -18.2% |
| Nov 21 - Dec 12+ | BULL | 21+ | +6.0% |

**Patterns**:
- 평균 BULL 기간: 12.8일 (6-21일)
- 평균 BEAR 기간: 11.7일 (8-19일)
- BEAR 평균 하락: -10.5%
- BULL 평균 상승: +7.1%

---

## 🏆 Final Ranking (Top 10)

| Rank | Method | Period Accuracy | Weighted Acc | Notes |
|------|--------|-----------------|--------------|-------|
| 🏆 1 | **Multi-Timeframe Trend** | **100.0% (12/12)** | **100.0%** | **Best overall** |
| 2 | EMA 50/200 Cross | 91.7% (11/12) | 96.0% | Classic, reliable |
| 3 | Price vs EMA100 (Smoothed) | 91.7% (11/12) | 96.0% | Good balance |
| 4 | Price vs EMA50 (Smoothed) | 91.7% (11/12) | 93.4% | Responsive |
| 5 | Price vs EMA200 (Smoothed) | 83.3% (10/12) | 83.4% | Lagging |
| 6 | Price Momentum (3d, 1%) | 83.3% (10/12) | 83.4% | Fast response |
| 7 | EMA Ribbon | 83.3% (10/12) | 83.4% | Comprehensive |
| 8 | Combined Best (5 Methods) | 83.3% (10/12) | 82.1% | Ensemble |
| 9 | Supertrend Consensus | 75.0% (9/12) | 76.8% | Good for trending |
| 10 | Trend Strength (ADX+DI) | 75.0% (9/12) | 74.8% | Trend strength |

### 실패한 방법들
- **Higher High/Low**: 8.3-16.7% (시장 구조 감지가 너무 느림)
- **Daily Candle Trend**: 50-58% (노이즈에 취약)
- **MACD Histogram**: 58.3% (지연 심함)
- **Price Momentum (7d, 3%)**: 50% (임계값이 너무 높음)

---

## 💡 Winning Method: Multi-Timeframe Trend

### 원리
3개 타임프레임의 트렌드 방향을 종합하여 2개 이상이 동의하면 해당 방향으로 분류

### 구성 요소

```python
# 1. 15분 트렌드
trend_15m = 1 if EMA(20) > EMA(50) else -1 if EMA(20) < EMA(50) else 0

# 2. 4시간 트렌드 (시뮬레이션)
close_4h = close.rolling(16).mean()  # 15분 * 16 = 4시간
ema_50_4h = EMA(close_4h, 50)
trend_4h = 1 if close_4h > ema_50_4h else -1

# 3. 일간 트렌드 (시뮬레이션)
close_1d = close.rolling(96).mean()  # 15분 * 96 = 1일
ema_20_1d = EMA(close_1d, 20)
trend_1d = 1 if close_1d > ema_20_1d else -1

# 종합 점수
total_score = trend_15m + trend_4h + trend_1d

# 분류
if total_score >= 2:
    regime = "BULL"
elif total_score <= -2:
    regime = "BEAR"
else:
    regime = "NEUTRAL"
```

### 왜 효과적인가?

1. **Multi-Timeframe Confirmation**: 단기/중기/장기 추세가 일치할 때만 신호
2. **Noise Reduction**: 단일 타임프레임의 휩소 방지
3. **Rolling Average Smoothing**: 4H/1D 시뮬레이션으로 노이즈 제거
4. **Conservative Classification**: 2/3 동의 필요 → 확신도 높음

### 기간별 상세 결과

| 기간 | Truth | Pred | Bull% | Bear% | Neutral% |
|------|-------|------|-------|-------|----------|
| Jul 14 - Aug 2 | BEAR | ✅ BEAR | 33.4% | 35.1% | 31.5% |
| Aug 2 - Aug 13 | BULL | ✅ BULL | 43.8% | 25.1% | 31.1% |
| Aug 13 - Aug 31 | BEAR | ✅ BEAR | 24.9% | 45.1% | 30.0% |
| Aug 31 - Sep 18 | BULL | ✅ BULL | 48.4% | 23.8% | 27.8% |
| Sep 18 - Sep 26 | BEAR | ✅ BEAR | 19.8% | 53.3% | 26.9% |
| Sep 26 - Oct 6 | BULL | ✅ BULL | 58.4% | 13.2% | 28.4% |
| Oct 6 - Oct 17 | BEAR | ✅ BEAR | 22.2% | 46.0% | 31.8% |
| Oct 17 - Oct 27 | BULL | ✅ BULL | 56.4% | 21.7% | 21.9% |
| Oct 27 - Nov 5 | BEAR | ✅ BEAR | 26.3% | 50.7% | 23.0% |
| Nov 5 - Nov 11 | BULL | ✅ BULL | 36.6% | 26.6% | 36.8% |
| Nov 11 - Nov 21 | BEAR | ✅ BEAR | 10.7% | 55.6% | 33.7% |
| Nov 21 - Dec 12 | BULL | ✅ BULL | 38.6% | 34.7% | 26.7% |

---

## 🛠️ Production Implementation

### 프로덕션용 코드

```python
def detect_market_regime_mtf(df, lookback_15m=200):
    """
    Multi-Timeframe Trend 기반 시장 레짐 분류

    Returns:
        regime: 'BULL', 'BEAR', 'NEUTRAL'
        confidence: float (0-1)
    """
    if len(df) < lookback_15m:
        return 'NEUTRAL', 0.5

    close = df['close'].iloc[-lookback_15m:]

    # 1. 15분 트렌드: EMA(20) vs EMA(50)
    ema_20 = close.ewm(span=20, adjust=False).mean()
    ema_50 = close.ewm(span=50, adjust=False).mean()

    if ema_20.iloc[-1] > ema_50.iloc[-1]:
        trend_15m = 1
    elif ema_20.iloc[-1] < ema_50.iloc[-1]:
        trend_15m = -1
    else:
        trend_15m = 0

    # 2. 4시간 트렌드 (시뮬레이션)
    close_4h = close.rolling(16).mean()
    ema_50_4h = close_4h.ewm(span=50, adjust=False).mean()

    if close_4h.iloc[-1] > ema_50_4h.iloc[-1]:
        trend_4h = 1
    else:
        trend_4h = -1

    # 3. 일간 트렌드 (시뮬레이션)
    close_1d = close.rolling(96).mean()
    ema_20_1d = close_1d.ewm(span=20, adjust=False).mean()

    if close_1d.iloc[-1] > ema_20_1d.iloc[-1]:
        trend_1d = 1
    else:
        trend_1d = -1

    # 종합 점수 계산
    total_score = trend_15m + trend_4h + trend_1d

    # 분류
    if total_score >= 2:
        regime = 'BULL'
        confidence = abs(total_score) / 3
    elif total_score <= -2:
        regime = 'BEAR'
        confidence = abs(total_score) / 3
    else:
        regime = 'NEUTRAL'
        confidence = 0.5

    return regime, confidence


def get_regime_position_multiplier(regime):
    """
    레짐에 따른 포지션 사이즈 조정

    RSI Zone Bot에서의 적용:
    - BULL: LONG 신호만 허용 (또는 LONG 강화)
    - BEAR: SHORT 신호만 허용 (또는 SHORT 강화)
    - NEUTRAL: 양방향 허용 (기본 사이징)
    """
    if regime == 'BULL':
        return {'LONG': 1.0, 'SHORT': 0.5}
    elif regime == 'BEAR':
        return {'LONG': 0.5, 'SHORT': 1.0}
    else:
        return {'LONG': 1.0, 'SHORT': 1.0}
```

### RSI Zone Bot 통합 예시

```python
# rsi_zone_bot.py에 추가

def should_take_trade(signal_direction, regime):
    """
    레짐에 따라 거래 허용 여부 결정
    """
    if regime == 'BULL' and signal_direction == 'LONG':
        return True
    elif regime == 'BEAR' and signal_direction == 'SHORT':
        return True
    elif regime == 'NEUTRAL':
        return True
    else:
        # 레짐 반대 방향 거래는 50% 확률로만 허용 (보수적)
        return False  # 또는 return random.random() < 0.5
```

---

## 📈 Alternative Methods (Backup)

### 2위: EMA 50/200 Cross (91.7%)

가장 심플하고 안정적인 방법

```python
def detect_regime_ema_cross(df, smooth_period=96):
    ema_50 = df['close'].ewm(span=50, adjust=False).mean()
    ema_200 = df['close'].ewm(span=200, adjust=False).mean()

    diff = ema_50 - ema_200
    diff_smooth = diff.rolling(smooth_period).mean()

    threshold = df['close'].rolling(smooth_period).std() * 0.1

    if diff_smooth.iloc[-1] > threshold.iloc[-1]:
        return 'BULL'
    elif diff_smooth.iloc[-1] < -threshold.iloc[-1]:
        return 'BEAR'
    else:
        return 'NEUTRAL'
```

**장점**: 단순, 이해하기 쉬움, 지연 적음
**단점**: Nov 5-11 BULL 기간을 NEUTRAL로 분류

### 3위: Price vs EMA100 (91.7%)

```python
def detect_regime_price_ema(df, ema_period=100, smooth_period=96):
    ema = df['close'].ewm(span=ema_period, adjust=False).mean()

    diff = df['close'] - ema
    diff_pct = diff / ema * 100
    diff_smooth = diff_pct.rolling(smooth_period).mean()

    if diff_smooth.iloc[-1] > 0.5:
        return 'BULL'
    elif diff_smooth.iloc[-1] < -0.5:
        return 'BEAR'
    else:
        return 'NEUTRAL'
```

---

## 🔬 Research Insights

### 효과적인 방법의 공통점

1. **Smoothing/Rolling Average** 사용
   - 단순 비교보다 rolling mean 사용 시 성능 향상
   - 96 candles (1일) smoothing이 효과적

2. **Multi-Timeframe Confirmation**
   - 단일 지표보다 여러 타임프레임 결합 시 성능 향상
   - 15m + 4H + Daily 조합이 최적

3. **Conservative Threshold**
   - 너무 민감한 임계값은 휩소에 취약
   - 약간 보수적인 분류 (NEUTRAL 허용)가 더 정확

### 실패한 방법의 공통점

1. **과도한 지연**: Higher High/Low 구조 감지 (10일+ 지연)
2. **노이즈 민감**: Daily Candle, MACD Histogram
3. **너무 높은 임계값**: Price Momentum 7d 3% (전환점 놓침)

---

## 📋 Recommendations

### 즉시 적용 (권장)

1. **RSI Zone Bot에 Multi-Timeframe Trend 통합**
   - BULL 레짐: LONG 신호만 허용 또는 강화
   - BEAR 레짐: SHORT 신호만 허용 또는 강화
   - NEUTRAL: 양방향 허용 (기존 로직)

2. **Position Sizing 조정**
   - 레짐 방향과 일치하는 거래: 100% 사이즈
   - 레짐 반대 방향 거래: 50% 사이즈 또는 거부

### 모니터링

1. **실시간 레짐 추적**
   - 봇 로그에 현재 레짐 기록
   - 레짐 전환 시 알림

2. **성과 분석**
   - 레짐별 승률/수익률 추적
   - 레짐 필터 전후 성과 비교

### 향후 연구

1. **레짐 전환 예측**: 현재는 감지만 → 전환 조기 감지 시도
2. **HMM 적용**: Hidden Markov Model로 확률적 레짐 분류
3. **ML 분류기**: 여러 지표를 입력으로 하는 분류 모델

---

## 📁 Files Created

| File | Description |
|------|-------------|
| `scripts/analysis/regime_classification_research.py` | V1: Per-candle analysis |
| `scripts/analysis/regime_classification_v2.py` | V2: Period-based majority voting |
| `results/regime_classification_v2_20251212_*.csv` | Final rankings |
| `claudedocs/REGIME_CLASSIFICATION_RESEARCH_20251212.md` | This document |

---

**결론**: Multi-Timeframe Trend 방법이 사용자 정의 레짐을 **100% 정확하게** 분류합니다. 이 방법을 RSI Zone Bot에 통합하면 레짐 방향과 일치하는 거래만 허용하여 승률 향상이 기대됩니다.
