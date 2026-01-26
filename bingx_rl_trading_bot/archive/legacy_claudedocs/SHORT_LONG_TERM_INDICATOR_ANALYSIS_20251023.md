# Short-Term vs Long-Term Indicator Analysis

**Date**: 2025-10-23
**Purpose**: 단기/장기 조합이 필요한 지표들을 식별하고 현재 구현 상태 분석

---

## 📊 현재 Production Feature 분석

### ✅ 현재 구현된 Lookback Periods

```yaml
Technical Indicators:
  RSI: 14 (단기만)
  MACD: 12/26/9 (이미 단기/장기 조합 ✓)
  ATR: 14 (단기만)

Moving Averages:
  SMA: 20, 50 (단기/중기)
  EMA: 12, 26 (단기/중기)

Rolling Windows:
  5 candles: 단기 모멘텀
  10 candles: 단기 패턴
  20 candles: 중기 추세
  50 candles: 중기 추세
```

### ⚠️ 문제점: 장기 지표 부재

**현재 최장 lookback**: 50 candles (4.2시간, 5분봉 기준)

**누락된 관점**:
- 일봉급 추세 (288 candles = 24시간)
- 주봉급 추세 (2,016 candles = 7일)
- 장기 support/resistance 레벨

---

## 🎯 단기/장기 조합이 중요한 지표

### 1. 이동평균 (Moving Averages) ⭐ **최우선**

**현재 상태**:
```python
# 단기/중기만 존재
ma_20 = df['close'].rolling(20).mean()  # 1.7시간
ma_50 = df['close'].rolling(50).mean()  # 4.2시간
ema_12 = df['close'].ewm(span=12).mean()  # 1시간
ema_26 = df['close'].ewm(span=26).mean()  # 2.2시간
```

**권장 추가**:
```python
# 장기 추세 포착
ma_200 = df['close'].rolling(200).mean()  # 16.7시간 (주요 추세선)
ema_200 = df['close'].ewm(span=200).mean()  # 장기 지지/저항

# 골든크로스/데드크로스 강화
short_ma = ma_20   # 단기
long_ma = ma_200   # 장기

# 크로스오버 신호
golden_cross = (short_ma > long_ma) & (short_ma.shift(1) <= long_ma.shift(1))
death_cross = (short_ma < long_ma) & (short_ma.shift(1) >= long_ma.shift(1))

# 거리 측정 (상대적 위치)
ma_distance = (short_ma - long_ma) / long_ma  # 단기가 장기보다 얼마나 위/아래
```

**트레이딩 중요성**:
- **골든크로스**: 단기 MA가 장기 MA 상향 돌파 → 강력한 매수 신호
- **데드크로스**: 단기 MA가 장기 MA 하향 돌파 → 강력한 매도 신호
- **MA 200**: 전통적으로 가장 중요한 장기 추세선

---

### 2. RSI (Relative Strength Index)

**현재 상태**:
```python
rsi_14 = talib.RSI(df['close'], timeperiod=14)  # 단기만
```

**권장 추가**:
```python
rsi_14 = talib.RSI(df['close'], timeperiod=14)   # 단기 과매수/과매도
rsi_50 = talib.RSI(df['close'], timeperiod=50)   # 중기 추세
rsi_200 = talib.RSI(df['close'], timeperiod=200) # 장기 추세

# 단기/장기 RSI 다이버전스
rsi_divergence = rsi_14 - rsi_200  # 단기와 장기의 괴리

# 시그널 생성
short_term_oversold = rsi_14 < 30
long_term_uptrend = rsi_200 > 50
# → 장기 상승장에서 단기 과매도 = 강력한 매수 기회
```

**트레이딩 중요성**:
- **단기 RSI**: 진입/청산 타이밍
- **장기 RSI**: 전체 추세 방향 확인
- **조합**: 장기 상승장 + 단기 과매도 = 고승률 매수

---

### 3. Volume (거래량)

**현재 상태**:
```python
volume_ma_ratio = df['volume'] / df['volume'].rolling(20).mean()  # 단기만
```

**권장 추가**:
```python
# 단기/장기 volume 평균
volume_ma_20 = df['volume'].rolling(20).mean()    # 단기 평균
volume_ma_200 = df['volume'].rolling(200).mean()  # 장기 평균

# Volume surge detection
volume_spike_vs_short = df['volume'] / volume_ma_20   # 단기 대비 spike
volume_spike_vs_long = df['volume'] / volume_ma_200   # 장기 대비 spike

# Accumulation/Distribution phase
volume_trend = volume_ma_20 / volume_ma_200  # 최근 거래량이 증가 추세인가?
# > 1.5: Accumulation phase (강세)
# < 0.7: Distribution phase (약세)
```

**트레이딩 중요성**:
- **Volume spike**: 단기 평균 대비 2-3배 = 중요한 움직임
- **장기 volume 증가**: 새로운 참가자 유입 = 추세 강화
- **장기 volume 감소**: 관심 약화 = 추세 약화

---

### 4. Volatility (변동성)

**현재 상태**:
```python
atr_14 = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
bb_20 = talib.BBANDS(df['close'], timeperiod=20)
```

**권장 추가**:
```python
# 단기/장기 ATR
atr_14 = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
atr_50 = talib.ATR(df['high'], df['low'], df['close'], timeperiod=50)
atr_200 = talib.ATR(df['high'], df['low'], df['close'], timeperiod=200)

# Volatility regime change
volatility_ratio = atr_14 / atr_200  # 현재 변동성 vs 장기 baseline
# > 1.5: High volatility regime (위험 증가)
# < 0.7: Low volatility regime (조정 후 breakout 가능)

# Bollinger Bands
bb_20 = talib.BBANDS(df['close'], timeperiod=20, nbdevup=2, nbdevdn=2)
bb_200 = talib.BBANDS(df['close'], timeperiod=200, nbdevup=2, nbdevdn=2)

# BB squeeze detection
bb_width_short = (bb_20_upper - bb_20_lower) / bb_20_mid
bb_width_long = (bb_200_upper - bb_200_lower) / bb_200_mid
bb_squeeze = bb_width_short < bb_width_long * 0.5  # 단기 BB가 장기 대비 매우 좁음
```

**트레이딩 중요성**:
- **ATR ratio**: 변동성 체제 변화 감지 (조용한 시기 후 폭발)
- **BB squeeze**: 변동성 수축 후 방향성 있는 큰 움직임 예고
- **Stop loss 조정**: 높은 변동성 = 넓은 SL, 낮은 변동성 = 좁은 SL

---

### 5. Support/Resistance Levels

**현재 상태**:
```python
# advanced_technical_features.py
lookback_sr = 200  # 지지/저항 탐지 윈도우
```

**권장 추가**:
```python
# 단기 S/R (스윙 트레이딩)
support_short = df['low'].rolling(20).min()
resistance_short = df['high'].rolling(20).max()

# 장기 S/R (주요 레벨)
support_long = df['low'].rolling(200).min()
resistance_long = df['high'].rolling(200).max()

# S/R 레벨 강도
# 단기 레벨에 가까움 + 장기 레벨과 일치 = 강력한 레벨
distance_to_short_support = (df['close'] - support_short) / df['close']
distance_to_long_support = (df['close'] - support_long) / df['close']

major_support_confluence = (
    (abs(distance_to_short_support) < 0.01) &  # 단기 S/R 근처
    (abs(distance_to_long_support) < 0.01)      # 장기 S/R도 근처
)  # → 매우 강력한 지지선
```

**트레이딩 중요성**:
- **단기 S/R**: 일중 매매 진입/청산 포인트
- **장기 S/R**: 주요 추세 전환 레벨
- **Confluence**: 여러 timeframe의 S/R 겹침 = 강력한 레벨

---

### 6. Momentum (모멘텀)

**현재 상태**:
```python
close_change_1 = df['close'].pct_change(1)   # 단기만
close_change_3 = df['close'].pct_change(3)
negative_momentum = -df['close'].pct_change(5).clip(upper=0)
```

**권장 추가**:
```python
# 단기/중기/장기 모멘텀
momentum_short = df['close'].pct_change(5)    # 25분
momentum_mid = df['close'].pct_change(20)     # 1.7시간
momentum_long = df['close'].pct_change(200)   # 16.7시간

# Momentum divergence
momentum_divergence = momentum_short - momentum_long
# Positive: 단기가 장기보다 강함 (가속)
# Negative: 단기가 장기보다 약함 (감속)

# ROC (Rate of Change)
roc_short = talib.ROC(df['close'], timeperiod=5)
roc_long = talib.ROC(df['close'], timeperiod=200)
momentum_acceleration = roc_short > roc_long  # 모멘텀이 가속하는가?
```

**트레이딩 중요성**:
- **Momentum divergence**: 추세 약화 조기 감지
- **가속/감속**: 추세 지속 vs 조정 신호
- **장기 모멘텀 반전**: 주요 추세 전환

---

## 📈 권장 구현 우선순위

### Priority 1: 이동평균 (MA/EMA) ⭐⭐⭐⭐⭐
```python
# 즉시 추가 권장
ma_200 = df['close'].rolling(200).mean()
ema_200 = df['close'].ewm(span=200).mean()

# 골든크로스/데드크로스
ma_cross_signal = calculate_ma_cross(ma_20, ma_200)
```

**이유**:
- 전통적으로 가장 중요한 지표
- 골든크로스/데드크로스는 검증된 신호
- 구현 간단, 해석 명확

---

### Priority 2: Volume ⭐⭐⭐⭐
```python
volume_ma_200 = df['volume'].rolling(200).mean()
volume_regime = df['volume'] / volume_ma_200
```

**이유**:
- Volume은 가격 움직임의 신뢰도 판단
- 장기 평균 대비 비교로 accumulation/distribution 감지
- 중요한 움직임 필터링에 필수

---

### Priority 3: ATR/Volatility ⭐⭐⭐⭐
```python
atr_200 = talib.ATR(df['high'], df['low'], df['close'], timeperiod=200)
volatility_regime = atr_14 / atr_200
```

**이유**:
- Stop loss/Position sizing 최적화
- 변동성 regime 변화 감지 (조용한 시기 → 폭발)
- 리스크 관리에 직접 활용

---

### Priority 4: RSI ⭐⭐⭐
```python
rsi_200 = talib.RSI(df['close'], timeperiod=200)
rsi_trend = rsi_14 - rsi_200
```

**이유**:
- 장기 추세 확인
- 단기 과매수/과매도 신호의 신뢰도 향상
- False signal 필터링

---

### Priority 5: Support/Resistance ⭐⭐⭐
```python
support_long_term = df['low'].rolling(200).min()
resistance_long_term = df['high'].rolling(200).max()
```

**이유**:
- 주요 레벨 식별
- 단기 레벨과의 confluence 감지
- 진입/청산 타이밍 개선

---

## 🔬 검증 방법

### 1. Feature Importance 분석
```yaml
실험:
  1. 장기 지표 추가 (MA200, Volume200, ATR200, RSI200)
  2. 모델 재학습
  3. Feature importance 측정

기대:
  - 장기 지표가 top 20 features에 포함
  - 특히 ma_cross, volume_regime, volatility_regime
```

### 2. Backtest 비교
```yaml
Baseline (현재):
  - 단기/중기 지표만 사용
  - Return, Win Rate, Sharpe 측정

Enhanced (장기 추가):
  - 단기/중기/장기 지표 조합
  - 동일 metric 측정

목표 개선:
  - Win Rate: +3-5%
  - Sharpe: +10-20%
  - Max DD: -10-20% (개선)
```

### 3. Signal Quality 검증
```yaml
테스트:
  - 골든크로스 발생 시 LONG win rate
  - 데드크로스 발생 시 SHORT win rate
  - Volume surge + 장기 상승 추세 시 win rate

기준:
  - 단독 지표: 60%+ win rate
  - 조합 지표: 70%+ win rate
```

---

## 💡 구현 제안

### Phase 1: 핵심 장기 지표 추가 (1주)
```python
def calculate_long_term_features(df):
    """장기 지표 계산 (200 period)"""

    # Moving Averages
    df['ma_200'] = df['close'].rolling(200).mean()
    df['ema_200'] = df['close'].ewm(span=200).mean()

    # MA Cross signals
    df['ma_20_vs_200'] = (df['ma_20'] - df['ma_200']) / df['ma_200']
    df['golden_cross'] = (
        (df['ma_20'] > df['ma_200']) &
        (df['ma_20'].shift(1) <= df['ma_200'].shift(1))
    ).astype(float)
    df['death_cross'] = (
        (df['ma_20'] < df['ma_200']) &
        (df['ma_20'].shift(1) >= df['ma_200'].shift(1))
    ).astype(float)

    # Volume
    df['volume_ma_200'] = df['volume'].rolling(200).mean()
    df['volume_regime'] = df['volume'] / df['volume_ma_200']

    # Volatility
    df['atr_200'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=200)
    df['volatility_regime'] = df['atr'] / df['atr_200']

    # RSI
    df['rsi_200'] = talib.RSI(df['close'], timeperiod=200)
    df['rsi_trend'] = df['rsi'] - df['rsi_200']

    return df
```

### Phase 2: 모델 재학습 및 검증 (1주)
```yaml
1. Feature 추가 (10개 추가 → 117 features)
2. 모델 재학습 (LONG/SHORT Entry/Exit)
3. Backtest 검증 (30일 + 105일)
4. Feature importance 분석
```

### Phase 3: 최적화 및 배포 (1주)
```yaml
1. Lookback period 최적화 (50, 100, 150, 200, 250)
2. 성능 개선 확인
3. Testnet 검증
4. Mainnet 배포
```

---

## 📊 기대 효과

### 성능 개선 예상
```yaml
Win Rate: 63.6% → 67-70% (+3-6%p)
  - 골든크로스/데드크로스 필터링
  - Volume regime 확인
  - Volatility regime 적응

Sharpe Ratio: 0.336 → 0.37-0.40 (+10-20%)
  - 장기 추세 방향과 일치하는 거래만
  - 변동성 regime에 따른 position sizing

Max Drawdown: -12.2% → -8-10% (-20-30%)
  - 주요 추세 반전 조기 감지
  - 장기 지지/저항 존중
```

### 트레이딩 품질 개선
```yaml
False Signal 감소:
  - 단기 과매수/과매도 신호 중 장기 추세 역행 제거
  - 골든크로스 없는 LONG, 데드크로스 없는 SHORT 필터

Risk Management:
  - Volatility regime에 따른 동적 SL
  - Volume regime 확인으로 저품질 거래 제거

Entry Timing:
  - 장기 지지선 근처 LONG
  - 장기 저항선 근처 SHORT
```

---

## 🎯 결론

**현재 문제**:
- 단기/중기 지표만 사용 (최대 50 candles = 4.2시간)
- 장기 추세 무시 → 역추세 거래 多
- 골든크로스/데드크로스 같은 강력한 신호 미활용

**해결 방안**:
1. **MA 200 추가** (최우선) - 골든크로스/데드크로스
2. **Volume MA 200** - Accumulation/Distribution
3. **ATR 200** - Volatility regime
4. **RSI 200** - 장기 추세
5. **S/R 200** - 주요 레벨

**기대 효과**:
- Win Rate +3-6%p
- Sharpe +10-20%
- Max DD -20-30%
- False signal 감소

**다음 단계**:
1. 조합 테스트 완료 대기 (진행 중)
2. 장기 지표 추가 구현
3. 모델 재학습 및 검증
4. 성능 비교 backtest

---

**Created**: 2025-10-23
**Status**: Analysis Complete - Implementation Pending
