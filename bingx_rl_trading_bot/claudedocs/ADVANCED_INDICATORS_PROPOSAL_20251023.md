# Advanced Indicators Proposal - Beyond Traditional TA

**Created**: 2025-10-23
**Purpose**: 전통적 지표(RSI, MACD, BB)를 넘어 강력한 최신 지표 추가 계획

---

## 📊 Current Indicator Analysis

### Traditional Indicators (현재 사용 중)
```yaml
Price-based:
  - RSI (14)
  - MACD (12/26/9)
  - Bollinger Bands (20)
  - Moving Averages (20, 50, 200)
  - EMA (5, 8, 10, 12)

Volume-based:
  - Volume MA Ratio (단순 비율)
  - Volume Price Correlation
  - Red Candle Volume Ratio

Pattern Recognition:
  - Candlestick Patterns (Hammer, Doji, Engulfing)
  - Double Top/Bottom
  - Divergence (RSI, MACD)

Support/Resistance:
  - Distance to S/R
  - Trendlines
  - Recent High/Low

Limitations:
  ❌ Price만 보고 Volume의 가격대별 분포 무시
  ❌ 거래 방향성(매수/매도 압력) 정보 없음
  ❌ 기관 투자자 활동 영역 파악 어려움
  ❌ 주요 유동성 레벨 감지 미흡
```

---

## 🚀 Advanced Indicators (High Priority)

### Category 1: Volume Profile & Market Structure ⭐⭐⭐⭐⭐

#### 1.1 Volume Profile (VP)
**가격대별 거래량 분포 - 기관 투자자 활동 영역**

```python
def calculate_volume_profile(df, lookback=100, bins=20):
    """
    가격대별 거래량 프로파일

    Returns:
        - poc (Point of Control): 최대 거래량 가격대
        - value_area_high: 거래량 70% 구간 상단
        - value_area_low: 거래량 70% 구간 하단
        - current_price_percentile: 현재가의 VP 상 위치
    """
    recent_df = df.tail(lookback)

    # 가격대별 거래량 집계
    price_min = recent_df['low'].min()
    price_max = recent_df['high'].max()
    price_range = price_max - price_min
    bin_size = price_range / bins

    volume_profile = np.zeros(bins)

    for _, row in recent_df.iterrows():
        # 각 candle의 가격 범위를 bins에 분배
        low_bin = int((row['low'] - price_min) / bin_size)
        high_bin = int((row['high'] - price_min) / bin_size)

        # 해당 구간에 거래량 분배
        for b in range(max(0, low_bin), min(bins, high_bin + 1)):
            volume_profile[b] += row['volume'] / (high_bin - low_bin + 1)

    # POC (Point of Control) - 최대 거래량 가격대
    poc_bin = np.argmax(volume_profile)
    poc_price = price_min + (poc_bin + 0.5) * bin_size

    # Value Area (거래량 70% 구간)
    total_volume = volume_profile.sum()
    sorted_bins = np.argsort(volume_profile)[::-1]

    cumsum = 0
    value_area_bins = []
    for bin_idx in sorted_bins:
        cumsum += volume_profile[bin_idx]
        value_area_bins.append(bin_idx)
        if cumsum >= total_volume * 0.70:
            break

    value_area_high = price_min + (max(value_area_bins) + 1) * bin_size
    value_area_low = price_min + min(value_area_bins) * bin_size

    # 현재가의 VP 상 위치
    current_price = df.iloc[-1]['close']
    current_bin = int((current_price - price_min) / bin_size)
    current_percentile = np.sum(volume_profile[:current_bin]) / total_volume

    return {
        'poc': poc_price,
        'value_area_high': value_area_high,
        'value_area_low': value_area_low,
        'distance_to_poc_pct': (current_price - poc_price) / current_price,
        'in_value_area': 1 if value_area_low <= current_price <= value_area_high else 0,
        'vp_percentile': current_percentile,
        'vp_skew': (poc_price - (price_min + price_max) / 2) / price_range
    }

# Features (7):
# - vp_poc_distance: POC까지 거리 (%)
# - vp_in_value_area: Value Area 내부 여부 (0/1)
# - vp_percentile: VP 상 현재가 위치 (0-1)
# - vp_to_vah: Value Area High까지 거리
# - vp_to_val: Value Area Low까지 거리
# - vp_skew: VP 편향성 (위쪽/아래쪽 거래량 집중)
# - vp_narrow: Value Area 폭 (좁을수록 강한 컨센서스)
```

**Why Powerful**:
- 🎯 **기관 투자자 축적 영역**: POC = 가장 많이 거래된 가격
- 📊 **지지/저항 자동 감지**: Value Area = 70% 거래량 구간
- 💰 **Fair Value**: POC에서 멀어질수록 되돌림 가능성
- 🔥 **Breakout 신뢰도**: Value Area 이탈 = 강한 신호

---

#### 1.2 VWAP (Volume Weighted Average Price)
**거래량 가중 평균가 - 기관 벤치마크**

```python
def calculate_vwap(df, period='day'):
    """
    VWAP = Σ(Price × Volume) / Σ(Volume)

    Variations:
    - Daily VWAP: 매일 리셋
    - Rolling VWAP: 특정 기간 (e.g., 100 candles)
    - Anchored VWAP: 주요 이벤트부터 (고점, 저점, 골든크로스 등)
    """
    # Typical Price
    typical_price = (df['high'] + df['low'] + df['close']) / 3

    # Daily VWAP (5분봉이므로 288개 = 1일)
    df['vwap_daily'] = (typical_price * df['volume']).rolling(288).sum() / df['volume'].rolling(288).sum()

    # Rolling VWAP (100 candles = 8.3시간)
    df['vwap_100'] = (typical_price * df['volume']).rolling(100).sum() / df['volume'].rolling(100).sum()

    # VWAP 밴드 (std dev bands)
    vwap_std = (df['close'] - df['vwap_daily']).rolling(288).std()
    df['vwap_upper'] = df['vwap_daily'] + vwap_std * 2
    df['vwap_lower'] = df['vwap_daily'] - vwap_std * 2

    return {
        'distance_to_vwap_pct': (df['close'] - df['vwap_daily']) / df['close'],
        'above_vwap': (df['close'] > df['vwap_daily']).astype(int),
        'vwap_band_position': (df['close'] - df['vwap_lower']) / (df['vwap_upper'] - df['vwap_lower']),
        'vwap_slope': df['vwap_daily'].diff(5) / df['vwap_daily'],
    }

# Features (4):
# - distance_to_vwap: VWAP 대비 현재가 (%)
# - above_vwap: VWAP 위/아래 (0/1)
# - vwap_band_position: VWAP 밴드 내 위치 (0-1)
# - vwap_slope: VWAP 기울기 (추세)
```

**Why Powerful**:
- 🏦 **기관 투자자 벤치마크**: 대량 매매 시 VWAP 기준
- 📈 **Intraday Trend**: VWAP 위 = Bullish, 아래 = Bearish
- ⚖️ **Fair Value 기준**: VWAP 회귀 경향
- 🎯 **진입/청산 기준**: 기관들이 실제 사용하는 기준선

---

### Category 2: Volume Flow Indicators ⭐⭐⭐⭐

#### 2.1 On-Balance Volume (OBV)
**누적 거래량 - 매수/매도 압력**

```python
def calculate_obv(df):
    """
    OBV = 전일 OBV + (오늘 상승 시 Volume, 하락 시 -Volume)
    """
    obv = [0]

    for i in range(1, len(df)):
        if df.iloc[i]['close'] > df.iloc[i-1]['close']:
            obv.append(obv[-1] + df.iloc[i]['volume'])
        elif df.iloc[i]['close'] < df.iloc[i-1]['close']:
            obv.append(obv[-1] - df.iloc[i]['volume'])
        else:
            obv.append(obv[-1])

    df['obv'] = obv
    df['obv_ma'] = df['obv'].rolling(20).mean()
    df['obv_slope'] = df['obv'].diff(10) / df['obv'].rolling(10).mean()

    # OBV Divergence
    price_higher = df['close'] > df['close'].shift(20)
    obv_lower = df['obv'] < df['obv'].shift(20)
    df['obv_bearish_div'] = (price_higher & obv_lower).astype(int)

    price_lower = df['close'] < df['close'].shift(20)
    obv_higher = df['obv'] > df['obv'].shift(20)
    df['obv_bullish_div'] = (price_lower & obv_higher).astype(int)

    return df

# Features (5):
# - obv: On-Balance Volume 값
# - obv_slope: OBV 기울기 (누적 압력)
# - obv_vs_ma: OBV vs MA 비율
# - obv_bearish_div: 약세 다이버전스
# - obv_bullish_div: 강세 다이버전스
```

---

#### 2.2 Accumulation/Distribution (A/D)
**매집/분산 라인 - 가격 x 거래량**

```python
def calculate_accumulation_distribution(df):
    """
    Money Flow Multiplier = [(Close - Low) - (High - Close)] / (High - Low)
    A/D = 전일 A/D + (MFM × Volume)
    """
    clv = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
    clv = clv.fillna(0)  # High = Low인 경우

    ad = (clv * df['volume']).cumsum()
    df['ad_line'] = ad
    df['ad_slope'] = ad.diff(10) / ad.rolling(10).std()

    # A/D vs Price Divergence
    price_trend = df['close'].rolling(20).apply(lambda x: 1 if x[-1] > x[0] else -1)
    ad_trend = df['ad_line'].rolling(20).apply(lambda x: 1 if x[-1] > x[0] else -1)
    df['ad_price_divergence'] = (price_trend != ad_trend).astype(int)

    return df

# Features (3):
# - ad_slope: A/D 라인 기울기
# - ad_momentum: A/D 가속도
# - ad_price_divergence: 가격과 A/D 다이버전스
```

---

#### 2.3 Chaikin Money Flow (CMF)
**기간별 자금 흐름**

```python
def calculate_cmf(df, period=20):
    """
    CMF = Σ[(CLV × Volume)] / Σ(Volume)
    CLV = [(Close - Low) - (High - Close)] / (High - Low)
    """
    clv = ((df['close'] - df['low']) - (df['high'] - df['close'])) / (df['high'] - df['low'])
    clv = clv.fillna(0)

    cmf = (clv * df['volume']).rolling(period).sum() / df['volume'].rolling(period).sum()
    df[f'cmf_{period}'] = cmf

    # CMF 상태
    df['cmf_bullish'] = (cmf > 0.1).astype(int)
    df['cmf_bearish'] = (cmf < -0.1).astype(int)
    df['cmf_neutral'] = ((cmf >= -0.1) & (cmf <= 0.1)).astype(int)

    return df

# Features (4):
# - cmf_20: Chaikin Money Flow (20)
# - cmf_bullish: 강한 매수 압력 (>0.1)
# - cmf_bearish: 강한 매도 압력 (<-0.1)
# - cmf_slope: CMF 기울기
```

---

### Category 3: Volatility & Channels ⭐⭐⭐⭐

#### 3.1 Keltner Channels
**ATR 기반 채널 - BB의 대안**

```python
def calculate_keltner_channels(df, ema_period=20, atr_period=10, multiplier=2):
    """
    Middle Line = EMA(20)
    Upper = EMA + (ATR × multiplier)
    Lower = EMA - (ATR × multiplier)
    """
    df['kc_middle'] = df['close'].ewm(span=ema_period).mean()

    # ATR
    high_low = df['high'] - df['low']
    high_close = abs(df['high'] - df['close'].shift())
    low_close = abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.rolling(atr_period).mean()

    df['kc_upper'] = df['kc_middle'] + (atr * multiplier)
    df['kc_lower'] = df['kc_middle'] - (atr * multiplier)
    df['kc_width'] = (df['kc_upper'] - df['kc_lower']) / df['kc_middle']
    df['kc_position'] = (df['close'] - df['kc_lower']) / (df['kc_upper'] - df['kc_lower'])

    # Keltner Squeeze (Keltner vs Bollinger Bands)
    bb_upper, bb_mid, bb_lower = calculate_bollinger_bands(df, period=20)
    df['kc_squeeze'] = ((bb_upper < df['kc_upper']) & (bb_lower > df['kc_lower'])).astype(int)

    return df

# Features (5):
# - kc_width: Keltner 채널 폭
# - kc_position: 채널 내 가격 위치 (0-1)
# - kc_upper_breakout: 상단 돌파
# - kc_lower_breakout: 하단 이탈
# - kc_squeeze: BB-KC Squeeze (breakout 임박)
```

---

#### 3.2 Donchian Channels
**고점/저점 채널 - 브레이크아웃**

```python
def calculate_donchian_channels(df, period=20):
    """
    Upper = Highest(High, period)
    Lower = Lowest(Low, period)
    Middle = (Upper + Lower) / 2
    """
    df['dc_upper'] = df['high'].rolling(period).max()
    df['dc_lower'] = df['low'].rolling(period).min()
    df['dc_middle'] = (df['dc_upper'] + df['dc_lower']) / 2

    df['dc_width'] = (df['dc_upper'] - df['dc_lower']) / df['dc_middle']
    df['dc_position'] = (df['close'] - df['dc_lower']) / (df['dc_upper'] - df['dc_lower'])

    # Breakout detection
    df['dc_upper_breakout'] = (df['close'] > df['dc_upper'].shift()).astype(int)
    df['dc_lower_breakdown'] = (df['close'] < df['dc_lower'].shift()).astype(int)

    return df

# Features (5):
# - dc_width: Donchian 채널 폭 (변동성)
# - dc_position: 채널 내 위치
# - dc_upper_breakout: 상단 돌파
# - dc_lower_breakdown: 하단 이탈
# - dc_middle_distance: Middle 대비 거리
```

---

### Category 4: Momentum & Strength ⭐⭐⭐⭐

#### 4.1 Money Flow Index (MFI)
**Volume-Weighted RSI**

```python
def calculate_mfi(df, period=14):
    """
    MFI = RSI에 Volume 가중
    Typical Price = (High + Low + Close) / 3
    Money Flow = Typical Price × Volume
    """
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    money_flow = typical_price * df['volume']

    # Positive/Negative Money Flow
    positive_flow = money_flow.where(typical_price > typical_price.shift(), 0)
    negative_flow = money_flow.where(typical_price < typical_price.shift(), 0)

    positive_mf = positive_flow.rolling(period).sum()
    negative_mf = negative_flow.rolling(period).sum()

    mfi = 100 - (100 / (1 + positive_mf / negative_mf))
    df['mfi'] = mfi

    # MFI 상태
    df['mfi_overbought'] = (mfi > 80).astype(int)
    df['mfi_oversold'] = (mfi < 20).astype(int)
    df['mfi_divergence_bullish'] = detect_divergence(df['close'], mfi, 'bullish')
    df['mfi_divergence_bearish'] = detect_divergence(df['close'], mfi, 'bearish')

    return df

# Features (5):
# - mfi: Money Flow Index
# - mfi_overbought: 과매수 (>80)
# - mfi_oversold: 과매도 (<20)
# - mfi_divergence_bullish: 강세 다이버전스
# - mfi_divergence_bearish: 약세 다이버전스
```

---

#### 4.2 Elder Force Index
**가격 변동 × 거래량**

```python
def calculate_elder_force_index(df, period=13):
    """
    Force Index = (Close - Close[1]) × Volume
    Smoothed with EMA
    """
    force = (df['close'] - df['close'].shift()) * df['volume']
    df['force_index'] = force.ewm(span=period).mean()
    df['force_index_norm'] = df['force_index'] / df['force_index'].rolling(50).std()

    # Force 상태
    df['force_strong_bullish'] = (df['force_index_norm'] > 2).astype(int)
    df['force_strong_bearish'] = (df['force_index_norm'] < -2).astype(int)

    return df

# Features (3):
# - force_index_norm: 정규화된 Force Index
# - force_strong_bullish: 강한 매수 압력
# - force_strong_bearish: 강한 매도 압력
```

---

### Category 5: Ichimoku Cloud ⭐⭐⭐⭐

**종합 트렌드 시스템**

```python
def calculate_ichimoku(df, tenkan=9, kijun=26, senkou_b=52):
    """
    Ichimoku Kinko Hyo - 일목균형표
    """
    # Tenkan-sen (Conversion Line): (9-high + 9-low) / 2
    high_9 = df['high'].rolling(tenkan).max()
    low_9 = df['low'].rolling(tenkan).min()
    df['ichimoku_tenkan'] = (high_9 + low_9) / 2

    # Kijun-sen (Base Line): (26-high + 26-low) / 2
    high_26 = df['high'].rolling(kijun).max()
    low_26 = df['low'].rolling(kijun).min()
    df['ichimoku_kijun'] = (high_26 + low_26) / 2

    # Senkou Span A (Leading Span A): (Tenkan + Kijun) / 2, shifted +26
    df['ichimoku_senkou_a'] = ((df['ichimoku_tenkan'] + df['ichimoku_kijun']) / 2).shift(kijun)

    # Senkou Span B (Leading Span B): (52-high + 52-low) / 2, shifted +26
    high_52 = df['high'].rolling(senkou_b).max()
    low_52 = df['low'].rolling(senkou_b).min()
    df['ichimoku_senkou_b'] = ((high_52 + low_52) / 2).shift(kijun)

    # Chikou Span (Lagging Span): Close shifted -26
    df['ichimoku_chikou'] = df['close'].shift(-kijun)

    # Cloud thickness
    df['ichimoku_cloud_thickness'] = abs(df['ichimoku_senkou_a'] - df['ichimoku_senkou_b']) / df['close']

    # Price vs Cloud
    cloud_top = df[['ichimoku_senkou_a', 'ichimoku_senkou_b']].max(axis=1)
    cloud_bottom = df[['ichimoku_senkou_a', 'ichimoku_senkou_b']].min(axis=1)

    df['ichimoku_above_cloud'] = (df['close'] > cloud_top).astype(int)
    df['ichimoku_below_cloud'] = (df['close'] < cloud_bottom).astype(int)
    df['ichimoku_in_cloud'] = ((df['close'] >= cloud_bottom) & (df['close'] <= cloud_top)).astype(int)

    # TK Cross
    df['ichimoku_tk_cross_bullish'] = ((df['ichimoku_tenkan'] > df['ichimoku_kijun']) &
                                        (df['ichimoku_tenkan'].shift() <= df['ichimoku_kijun'].shift())).astype(int)
    df['ichimoku_tk_cross_bearish'] = ((df['ichimoku_tenkan'] < df['ichimoku_kijun']) &
                                        (df['ichimoku_tenkan'].shift() >= df['ichimoku_kijun'].shift())).astype(int)

    # Cloud color (Senkou A > B = Bullish cloud)
    df['ichimoku_cloud_bullish'] = (df['ichimoku_senkou_a'] > df['ichimoku_senkou_b']).astype(int)

    return df

# Features (10):
# - ichimoku_tenkan, kijun, senkou_a, senkou_b, chikou
# - ichimoku_cloud_thickness: 구름 두께 (지지/저항 강도)
# - ichimoku_above/below/in_cloud: 가격 위치
# - ichimoku_tk_cross_bullish/bearish: TK 크로스
# - ichimoku_cloud_bullish: 구름 색 (추세)
```

**Why Powerful**:
- 📊 **Multiple Timeframes**: 9/26/52 기간 동시 고려
- 🎯 **Support/Resistance**: Cloud = 동적 S/R
- 📈 **Trend Confirmation**: 5개 라인의 배열로 추세 확인
- ⚡ **Entry Signals**: TK Cross, Cloud Breakout

---

## 📋 Implementation Priority & Roadmap

### Phase 1: Volume Profile & VWAP (가장 강력) ⭐⭐⭐⭐⭐
```yaml
Priority: HIGHEST
Implementation Time: 2-3 hours
Features Added: 11 features

Features:
  Volume Profile (7):
    - vp_poc_distance
    - vp_in_value_area
    - vp_percentile
    - vp_to_vah, vp_to_val
    - vp_skew
    - vp_narrow

  VWAP (4):
    - distance_to_vwap
    - above_vwap
    - vwap_band_position
    - vwap_slope

Expected Impact:
  Win Rate: +2-4%p
  Sharpe: +0.03-0.05

Rationale:
  - 기관 투자자가 실제 사용하는 지표
  - POC/Value Area = 강력한 S/R
  - VWAP = Intraday 트렌드 벤치마크
```

---

### Phase 2: Volume Flow Indicators ⭐⭐⭐⭐
```yaml
Priority: HIGH
Implementation Time: 1-2 hours
Features Added: 13 features

Features:
  OBV (5):
    - obv, obv_slope, obv_vs_ma
    - obv_bullish/bearish_div

  A/D Line (3):
    - ad_slope, ad_momentum
    - ad_price_divergence

  CMF (4):
    - cmf_20
    - cmf_bullish/bearish/neutral

  MFI (5):
    - mfi, mfi_overbought/oversold
    - mfi_divergence_bullish/bearish

Expected Impact:
  Win Rate: +1-3%p
  Sharpe: +0.02-0.04

Rationale:
  - 매수/매도 압력 정량화
  - 가격-거래량 다이버전스 감지
  - 추세 전환 조기 포착
```

---

### Phase 3: Ichimoku Cloud ⭐⭐⭐⭐
```yaml
Priority: HIGH
Implementation Time: 1 hour
Features Added: 10 features

Expected Impact:
  Win Rate: +1-2%p
  Sharpe: +0.02-0.03

Rationale:
  - 종합적 트렌드 시스템
  - 동적 S/R (Cloud)
  - Multiple timeframe 고려
```

---

### Phase 4: Channels & Force Index ⭐⭐⭐
```yaml
Priority: MEDIUM
Implementation Time: 1 hour
Features Added: 13 features

Features:
  Keltner Channels (5)
  Donchian Channels (5)
  Elder Force Index (3)

Expected Impact:
  Win Rate: +0.5-1%p
  Sharpe: +0.01-0.02
```

---

## 📊 Total Feature Count Projection

### Current + Long-term + Advanced
```yaml
LONG Entry Model:
  Current: 44
  + Long-term: 23
  + Advanced Phase 1-4: 47
  Total: 114 features

SHORT Entry Model:
  Current: 38
  + Long-term: 23
  + Advanced Phase 1-4: 47
  Total: 108 features

Exit Models:
  Current: 24
  + Long-term: 23
  + Advanced Phase 1-4: 47
  Total: 94 features
```

---

## 🎯 Recommended Implementation Strategy

### Option A: All-in-One (Recommended)
```yaml
Approach:
  - Add all indicators at once
  - Long-term (23) + Advanced (47) = 70 new features
  - Train 4 models with enhanced feature set

Timeline:
  - Feature calculation: 4-5 hours
  - Model training: 30 minutes
  - Backtest validation: 1 hour
  - Total: 1 day

Pros:
  ✅ Maximum information available
  ✅ XGBoost handles feature selection
  ✅ One-time comprehensive upgrade

Cons:
  ⚠️ Cannot isolate individual indicator impact
  ⚠️ Longer initial development
```

---

### Option B: Phased Rollout (Conservative)
```yaml
Week 1: Long-term indicators (23 features)
  - Baseline performance measurement

Week 2: Volume Profile + VWAP (11 features)
  - Measure incremental impact

Week 3: Volume Flow (13 features)
  - Cumulative improvement tracking

Week 4: Ichimoku + Channels (23 features)
  - Final performance validation

Pros:
  ✅ Measure individual contributions
  ✅ Gradual complexity increase
  ✅ Easier debugging

Cons:
  ⚠️ Slower overall progress
  ⚠️ Multiple training cycles
```

---

## 🔬 Expected Performance Improvements

### Conservative Estimate
```yaml
Win Rate:
  Current: 63.6%
  + Long-term: 67-70% (+3-6%p)
  + Advanced: 69-73% (+2-3%p)
  Total: 69-73% (+5-9%p)

Sharpe Ratio:
  Current: 0.336
  + Long-term: 0.37-0.40 (+0.03-0.06)
  + Advanced: 0.40-0.45 (+0.03-0.05)
  Total: 0.40-0.45 (+0.06-0.11)

Max Drawdown:
  Current: -12.2%
  + Long-term: -8 to -10% (-20-30% improvement)
  + Advanced: -6 to -8% (-20-25% further)
  Total: -6 to -8% (-35-50% total improvement)
```

---

## 💡 Key Insights

### Why These Indicators Are Powerful

**Volume Profile**:
- 가격대별 거래량 = 기관 투자자 축적 영역
- POC = 자석 효과 (mean reversion)
- Value Area 이탈 = 강한 breakout

**VWAP**:
- 기관 투자자가 실제 사용하는 벤치마크
- Intraday fair value 기준
- 알고리즘 트레이딩의 표준 지표

**OBV/A/D/CMF/MFI**:
- 가격보다 먼저 움직이는 경향
- Divergence = 추세 전환 조기 신호
- 매수/매도 압력 정량화

**Ichimoku**:
- 5개 라인으로 종합 판단
- Cloud = 동적 support/resistance
- TK Cross = 명확한 진입 신호

---

## 🚀 Next Steps

1. **Review & Approve**: 사용자 피드백 반영
2. **Implementation**: Phase 1 시작 또는 All-in-One
3. **Testing**: 각 indicator 정확성 검증
4. **Training**: Enhanced 모델 훈련
5. **Validation**: Backtest 성능 확인

---

**Status**: Proposal Ready
**Recommendation**: Option A (All-in-One) for maximum impact
**Expected Development Time**: 1 day
**Expected Performance Gain**: +5-9%p Win Rate, +0.06-0.11 Sharpe
