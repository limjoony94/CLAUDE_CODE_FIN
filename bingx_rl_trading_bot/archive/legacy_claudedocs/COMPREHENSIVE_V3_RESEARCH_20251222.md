# Comprehensive V3 Research - Entry/Exit Optimization

**Date**: 2025-12-22
**Baseline**: RSI Trend Filter Bot v2.0 (PreBE Spike Protection)
**Period**: 90 days (2025-09-23 to 2025-12-22)

---

## 1. Research Objective

v2.0 PreBE Spike Protection 배포 후 추가 개선 가능성 탐색:
1. **Entry 개선**: PriceEff, Volume, RSI Divergence, Keltner Channel 필터
2. **Exit 최적화**: ATR Dynamic TP/SL, Partial Profit Taking, Keltner Trail
3. **조합 전략**: Entry + Exit 최적 조합 탐색

---

## 2. Tested Strategies (17 Total)

### Entry Filters

| # | Strategy | Entry Filter | Full PnL | WR | Trades |
|---|----------|-------------|----------|-----|--------|
| 0 | v2.0 Baseline | None | +17.59% | 67.7% | 130 |
| 1 | PriceEff 0.35 | efficiency > 0.35 | +0.25% | 63.8% | 47 |
| 2 | PriceEff 0.40 | efficiency > 0.40 | -0.90% | 62.5% | 32 |
| 3 | Volume 1.2 | vol > MA*1.2 | +0.90% | 63.9% | 83 |
| 4 | Volume 1.5 | vol > MA*1.5 | -1.42% | 61.8% | 68 |
| 5 | BodyRatio 0.5 | body/range > 0.5 | +15.30% | 69.6% | 115 |
| 6 | RSI Divergence | bullish/bearish div | +14.80% | 66.2% | 71 |
| 7 | Keltner | price near bands | +12.04% | 65.6% | 122 |
| 8 | PriceEff+Volume | combined | -2.78% | 63.6% | 22 |

### Exit Configurations

| # | Strategy | Exit Config | Full PnL | WR | Trades |
|---|----------|-------------|----------|-----|--------|
| 9 | ATR Dynamic 2.5/1.5 | TP=ATR*2.5, SL=ATR*1.5 | +2.94% | 38.3% | 183 |
| 10 | ATR Dynamic 3.0/1.5 | TP=ATR*3.0, SL=ATR*1.5 | +2.82% | 38.3% | 183 |
| 11 | Partial 50%@1.5% | 50% exit @1.5% | +20.02% | 75.1% | 169 |
| 12 | Partial 50%@2.0% | 50% exit @2.0% | +20.16% | 72.5% | 153 |
| 13 | Keltner Trail | trail at Keltner band | +13.00% | 65.9% | 132 |

### Combined Strategies

| # | Strategy | Entry + Exit | Full PnL | WR | Trades |
|---|----------|-------------|----------|-----|--------|
| 14 | PriceEff+ATR | PriceEff 0.35 + ATR Exit | +2.10% | 38.8% | 49 |
| 15 | Volume+Partial | Volume 1.2 + Partial | +1.31% | 71.2% | 104 |
| 16 | PriceEff+Vol+ATR | Both + ATR | +4.15% | 45.5% | 22 |

---

## 3. Walk-Forward Validation (Top 10)

| Strategy | WF PnL | Profitable | Windows | WF WR |
|----------|--------|------------|---------|-------|
| **12_Exit_Partial_50%@2.0%** | **+19.21%** | 4/6 | 6 | 66.7% |
| **11_Exit_Partial_50%@1.5%** | **+18.51%** | 4/6 | 6 | 66.7% |
| **6_Entry_RSI_Divergence** | **+17.97%** | **5/6** | 6 | **83.3%** |
| 0_v2.0_Baseline | +16.62% | 4/6 | 6 | 66.7% |
| 5_Entry_BodyRatio_0.5 | +14.26% | 4/6 | 6 | 66.7% |
| 13_Exit_Keltner_Trail | +12.01% | 4/6 | 6 | 66.7% |
| 7_Entry_Keltner | +11.57% | 4/6 | 6 | 66.7% |
| 16_PriceEff+Volume+ATR | +4.15% | 4/6 | 6 | 66.7% |
| 9_Exit_ATR_Dynamic_2.5_1.5 | +2.74% | 4/6 | 6 | 66.7% |
| 10_Exit_ATR_Dynamic_3.0_1.5 | +2.62% | 4/6 | 6 | 66.7% |

---

## 4. Key Findings

### Winners

#### 1. Partial Exit (50%@2.0%) - Best Absolute Returns
```
Full PnL:    +20.16% (+2.57% vs baseline)
WF PnL:      +19.21% (+2.59% vs baseline)
Win Rate:    72.5% (+4.8%p vs baseline)
Sharpe:      1.83 (+17% vs baseline)

LONG PnL:    +2.31% (baseline +2.96%)
SHORT PnL:   +17.85% (baseline +14.63%, +22% better)
```

**Mechanism**: 1.5~2% 수익 도달 시 50% 포지션 청산, 나머지 Trail 유지
- 조기 이익 실현으로 안정적 수익 확보
- SHORT 성능 대폭 개선 (+22%)

#### 2. RSI Divergence Entry - Best Consistency
```
Full PnL:    +14.80% (-2.79% vs baseline)
WF PnL:      +17.97% (+1.35% vs baseline)
Win Rate:    66.2% (-1.5%p vs baseline)
Sharpe:      2.18 (+40% vs baseline) - HIGHEST

WF Profitable: 5/6 (83.3%) - BEST
Avg PnL/Trade: 0.208 (+54% vs baseline)
```

**Mechanism**: RSI Divergence 감지 시에만 진입
- Bullish Div: 가격 lower low + RSI higher low
- Bearish Div: 가격 higher high + RSI lower high
- 거래 빈도 46% 감소 (130 → 71)하지만 품질 향상

### Losers

#### 1. Entry Filters (PriceEff, Volume)
```
PriceEff 0.35:     +0.25% (vs +17.59%)  - 97% 성능 하락!
Volume 1.2:        +0.90% (vs +17.59%)  - 94% 성능 하락!
PriceEff+Volume:   -2.78% (vs +17.59%)  - 손실 전환!
```

**원인**: 과도한 필터링으로 좋은 거래 기회 놓침
- 거래 수: 130 → 22~83 (46~83% 감소)
- 필터가 너무 엄격하여 유효 신호도 제거

#### 2. ATR Dynamic Exit
```
ATR Dynamic 2.5/1.5: +2.94% (vs +17.59%)
Win Rate:            38.3% (vs 67.7%)
```

**원인**: Whipsaw 다수 발생
- ATR 기반 TP/SL은 변동성 높은 구간에서 조기 청산 유발
- BE+Trail 방식이 더 효과적

---

## 5. LONG vs SHORT Performance

| Strategy | LONG PnL | SHORT PnL | Balance |
|----------|----------|-----------|---------|
| Baseline v2.0 | +2.96% | +14.63% | SHORT 우세 |
| Partial 50%@2.0% | +2.31% | +17.85% | SHORT 더 우세 |
| RSI Divergence | +2.31% | +12.49% | SHORT 우세 |
| ATR Dynamic | -5.85% | +8.78% | LONG 손실! |

**결론**: 현재 시장 조건에서 SHORT 전략이 더 효과적

---

## 6. Recommendations

### v2.0 유지 권장

현재 v2.0 PreBE Spike Protection이 이미 양호한 성능:
- WF PnL: +16.62%
- WF Profitable: 4/6 (66.7%)
- LONG 수익 전환 달성 (+2.96%)

### 업그레이드 옵션 (선택적)

| Option | 변경 | 예상 개선 | 위험 |
|--------|------|----------|------|
| **A. Partial Exit** | 50%@2.0% 조기 청산 | +2.57% PnL | 구현 복잡도 ↑ |
| **B. RSI Divergence** | Entry 필터 추가 | 83.3% WF 일관성 | 거래 빈도 ↓ |

**권장**: B. RSI Divergence를 선택적 필터로 추가 (일관성 우선)

---

## 7. Files

```
Scripts:
- scripts/analysis/comprehensive_v3_research.py

Results:
- results/comprehensive_v3_full_20251222_172833.csv
- results/comprehensive_v3_wf_20251222_172833.csv
```

---

## 8. Appendix: Indicator Calculations

### Price Efficiency
```python
direct_move = abs(close - close.shift(period))
total_move = abs(close.diff()).rolling(period).sum()
efficiency = direct_move / total_move
```

### RSI Divergence
```python
# Bullish: price makes lower low, RSI makes higher low
# Bearish: price makes higher high, RSI makes lower high
rsi_higher_low = (rsi > rsi.shift(lookback)) & (rsi.shift(lookback) < 50)
price_lower_low = close < close.shift(lookback).rolling(lookback).min()
bullish_div = rsi_higher_low & price_lower_low
```

### Keltner Channels
```python
mid = EMA(close, 20)
atr = ATR(high, low, close, 10)
upper = mid + atr * 2.0
lower = mid - atr * 2.0
```
