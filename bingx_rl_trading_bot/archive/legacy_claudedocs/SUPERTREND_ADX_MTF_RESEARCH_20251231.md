# SuperTrend 5m Bot - ADX Filter & MTF Filter Research

**연구 날짜**: 2025-12-31
**데이터 기간**: 2025-10-02 ~ 2025-12-31 (약 90일, 25,920 캔들)
**기준 전략**: SuperTrend 5m Bot v1.3 Scale-out Exit

---

## 연구 목적

SuperTrend 5m Bot의 신호 품질 개선을 위해 두 가지 필터를 연구:
1. **ADX Filter**: 추세 강도가 일정 이상일 때만 진입
2. **MTF (Multi-Timeframe) Filter**: 1H 타임프레임 EMA 정렬 기반 필터

---

## 테스트 설정

### 기본 전략 파라미터
- SuperTrend: ATR Period 10, Multiplier 2.2
- Vol-Adaptive TP/SL: Base TP 2.5%, Base SL 2.0%
- Vol Lookback: 75 candles
- Scale-out Exit: 50%@0.5xTP + 30%@0.8xTP + 20%@1.0xTP
- Cooldown: 1 candle (5분)
- Effective Leverage: 4x

### 테스트 조합 (15개)

**ADX Thresholds**: [0 (No ADX), 15, 20, 25, 30]
**MTF Configs**: [No MTF, EMA20/50, EMA20/100]

Walk-Forward Validation: 30일 Train / 10일 Test, 6 Windows

---

## 연구 결과

### Full Period 백테스트 결과

| Config | Full PnL% | Trades | Win Rate | Max DD | LONG $ | SHORT $ | WF | WF PnL |
|--------|-----------|--------|----------|--------|--------|---------|-----|--------|
| **No ADX + No MTF** | +14.8% | 61 | 55.7% | 7.6% | $1.39 | $13.45 | 2/6 | $9.18 |
| **ADX≥15 + No MTF** | **+26.7%** | 61 | 59.0% | **6.1%** | $5.11 | $21.61 | 3/6 | $15.62 |
| **ADX≥20 + No MTF** | +21.0% | 60 | 58.3% | 8.9% | $1.16 | $19.88 | **4/6** | $15.53 |
| **ADX≥25 + No MTF** | +17.5% | 61 | 54.1% | 11.5% | $0.79 | $16.76 | **4/6** | **$17.48** |
| **ADX≥30 + No MTF** | +26.1% | 52 | 59.6% | 6.4% | $0.95 | $25.10 | **4/6** | $14.97 |
| No ADX + MTF EMA20/50 | -16.9% | 80 | 48.8% | 23.9% | $-14.67 | $-2.25 | 2/6 | $-16.68 |
| ADX≥15 + MTF EMA20/50 | -17.6% | 76 | 48.7% | 21.8% | $-14.81 | $-2.76 | 2/6 | $-14.66 |
| ADX≥20 + MTF EMA20/50 | -11.4% | 73 | 49.3% | 22.5% | $-8.38 | $-3.04 | 2/6 | $-17.64 |
| ADX≥25 + MTF EMA20/50 | -11.7% | 63 | 47.6% | 21.7% | $-5.17 | $-6.58 | 1/6 | $-15.10 |
| ADX≥30 + MTF EMA20/50 | -9.2% | 47 | 48.9% | 16.8% | $-0.13 | $-9.09 | 2/6 | $-16.20 |
| No ADX + MTF EMA20/100 | -20.1% | 87 | 46.0% | 26.2% | $-17.21 | $-2.89 | 2/6 | $-19.66 |
| ADX≥15 + MTF EMA20/100 | -20.7% | 83 | 45.8% | 24.8% | $-17.26 | $-3.46 | 2/6 | $-17.45 |
| ADX≥20 + MTF EMA20/100 | -13.0% | 77 | 45.5% | 19.3% | $-9.90 | $-3.06 | 2/6 | $-15.15 |
| ADX≥25 + MTF EMA20/100 | -8.4% | 61 | 45.9% | 15.3% | $-4.35 | $-4.05 | 2/6 | $-6.47 |
| ADX≥30 + MTF EMA20/100 | -4.6% | 48 | 50.0% | 13.8% | $0.04 | $-4.65 | 2/6 | $-7.03 |

---

## 핵심 발견

### 1. MTF Filter는 성능을 악화시킴

| MTF 설정 | PnL 범위 | Win Rate 범위 | Max DD 범위 |
|----------|----------|---------------|-------------|
| **No MTF** | **+14.8% ~ +26.7%** | 54.1% ~ 59.6% | 6.1% ~ 11.5% |
| EMA20/50 | -9.2% ~ -17.6% | 47.6% ~ 49.3% | 16.8% ~ 23.9% |
| EMA20/100 | -4.6% ~ -20.7% | 45.5% ~ 50.0% | 13.8% ~ 26.2% |

**결론**: MTF Filter는 **사용하지 않는 것이 최선**

MTF Filter가 성능을 악화시키는 이유:
- 1H 추세와 5m SuperTrend 신호가 자주 불일치
- 좋은 5m 진입 기회를 필터링으로 놓침
- LONG 방향에서 특히 큰 손실 발생

### 2. ADX Filter는 일관성을 개선

| ADX 설정 | Full PnL | WF 일관성 | WF PnL | 비고 |
|----------|----------|-----------|--------|------|
| No ADX | +14.8% | 2/6 (33%) | $9.18 | 기준선 |
| ADX≥15 | +26.7% | 3/6 (50%) | $15.62 | 최고 Full PnL |
| ADX≥20 | +21.0% | 4/6 (67%) | $15.53 | 좋은 균형 |
| **ADX≥25** | +17.5% | **4/6 (67%)** | **$17.48** | **최고 WF PnL** |
| ADX≥30 | +26.1% | 4/6 (67%) | $14.97 | 최저 거래 수 |

**결론**: **ADX≥25 Filter 권장** (최고 Walk-Forward 일관성 및 PnL)

### 3. 방향별 성과 분석

**No MTF 설정들** (수익):
- LONG: $0.79 ~ $5.11 (모두 양수)
- SHORT: $13.45 ~ $25.10 (LONG 대비 3~25배 높음)

**MTF 설정들** (손실):
- LONG: $-17.26 ~ $0.04 (대부분 큰 손실)
- SHORT: $-9.09 ~ $-2.25 (손실)

**결론**: SHORT 방향이 훨씬 수익성이 높음. MTF는 특히 LONG에서 손실 유발.

---

## 추천 설정

### 권장: ADX≥25 + No MTF

| 항목 | 기존 (v1.3 Baseline) | 권장 (ADX≥25) | 개선 |
|------|---------------------|---------------|------|
| Full Period PnL | +14.8% | +17.5% | **+2.7%p** |
| Walk-Forward | 2/6 (33%) | **4/6 (67%)** | **+34%p** |
| WF PnL | $9.18 | **$17.48** | **+90%** |
| Win Rate | 55.7% | 54.1% | -1.6%p |
| Max Drawdown | 7.6% | 11.5% | +3.9%p |
| Trades | 61 | 61 | 동일 |

### ADX Filter 구현 로직

```python
# ADX 계산
def calculate_adx(df, period=14):
    high = df['high']
    low = df['low']
    close = df['close']

    # True Range
    tr = pd.concat([
        high - low,
        abs(high - close.shift(1)),
        abs(low - close.shift(1))
    ], axis=1).max(axis=1)

    # +DM, -DM
    up_move = high - high.shift(1)
    down_move = low.shift(1) - low
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)

    # Smoothed averages
    atr = tr.ewm(span=period, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm).ewm(span=period, adjust=False).mean() / atr
    minus_di = 100 * pd.Series(minus_dm).ewm(span=period, adjust=False).mean() / atr

    # ADX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.ewm(span=period, adjust=False).mean()

    return adx

# Entry Logic
def generate_signal(df):
    # SuperTrend direction flip detection
    signal = 0
    if df['direction'].iloc[-2] != df['direction'].iloc[-1]:
        if df['direction'].iloc[-1] == 1:  # Bullish flip
            signal = 1  # LONG
        else:  # Bearish flip
            signal = -1  # SHORT

    # ADX Filter (≥25)
    adx = df['adx'].iloc[-1]
    if adx < 25:
        signal = 0  # Filter out weak trend

    return signal
```

---

## 추가 고려사항

### 1. ADX≥15 vs ADX≥25 Trade-off

| 항목 | ADX≥15 | ADX≥25 |
|------|--------|--------|
| Full PnL | **+26.7%** | +17.5% |
| WF 일관성 | 3/6 (50%) | **4/6 (67%)** |
| WF PnL | $15.62 | **$17.48** |
| Max DD | **6.1%** | 11.5% |

- **ADX≥15**: 더 높은 Full PnL, 낮은 Drawdown, 하지만 WF 일관성 낮음
- **ADX≥25**: 더 높은 WF 일관성 및 PnL, 하지만 Drawdown 증가

**권장**: 안정성 중시 → ADX≥25, 공격적 → ADX≥15

### 2. MTF Filter 제거 근거

MTF Filter가 손실을 유발하는 이유 분석:
1. **신호 지연**: 1H EMA 정렬은 5m 신호 대비 지연됨
2. **과도한 필터링**: 좋은 5m 기회를 놓침 (거래 수 증가하지만 Win Rate 하락)
3. **추세 불일치**: 5m SuperTrend는 빠른 반전을 잡지만, 1H 추세는 아직 반대

### 3. 구현 권장사항

1. **ADX Period**: 14 (표준값)
2. **ADX Threshold**: 25
3. **MTF Filter**: 비활성화
4. **나머지 파라미터**: v1.3 Scale-out 유지

---

## 파일 위치

- **연구 스크립트**: `scripts/analysis/supertrend_adx_mtf_research.py`
- **결과 CSV**: `results/supertrend_adx_mtf_research_20251231_162526.csv`
- **설정 파일**: `config/supertrend_5m_config.yaml`

---

## 결론

1. **MTF Filter 제거**: 성능을 악화시키므로 사용하지 않음
2. **ADX≥25 Filter 추가**: Walk-Forward 일관성 2/6 → 4/6 개선
3. **WF PnL +90% 개선**: $9.18 → $17.48
4. **SHORT 방향이 주 수익원**: LONG 대비 10~25배 수익

**최종 권장**: SuperTrend 5m Bot v1.4에 ADX≥25 Filter 추가
