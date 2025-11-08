# Advanced Indicators Integration Guide

**Created**: 2025-10-23
**Status**: ✅ Phase 1 Complete (Volume Profile + VWAP)
**Next**: Phase 2 (Volume Flow Indicators)

---

## 📊 Overview

모듈형 고급 지표 시스템으로 전통적 RSI/MACD/BB를 넘어서는 강력한 지표들을 제공합니다.

### Phase 1: Volume Profile + VWAP ✅ COMPLETE
- **Volume Profile** (7 features): 기관 투자자 활동 영역
- **VWAP** (4 features): 기관 투자자 벤치마크
- **Total**: 11 features
- **Performance**: 417.3 candles/sec
- **Quality**: 0 issues, 99.1% data coverage

### Phase 2: Volume Flow Indicators (Next)
- **OBV** (3 features): 거래량 누적 분석
- **A/D Line** (3 features): 자금 분배 분석
- **CMF** (3 features): 차이킨 자금 흐름
- **MFI** (4 features): 거래량 포함 RSI
- **Total**: 13 features

---

## 🚀 Quick Start

### 1. Import Module
```python
from scripts.indicators.advanced_indicators import (
    calculate_all_advanced_indicators,
    get_all_advanced_features
)
```

### 2. Calculate Indicators
```python
# Load OHLCV data
df = pd.read_csv("BTCUSDT_5m.csv")

# Calculate Phase 1 indicators (11 features)
df_enhanced = calculate_all_advanced_indicators(df, phase='phase1')

# Get feature names
features = get_all_advanced_features('phase1')
print(f"Added {len(features)} features: {features}")
```

### 3. Use in Model Training
```python
# Example: Combine with existing features
from scripts.experiments.calculate_features_with_longterm import calculate_all_features_enhanced

# Calculate baseline + long-term features (140 total)
df = calculate_all_features_enhanced(df)

# Add advanced indicators (11 more)
df = calculate_all_advanced_indicators(df, phase='phase1')

# Total: 151 features
print(f"Total features: {len(df.columns)}")
```

---

## 📈 Phase 1 Features (11)

### Volume Profile (7 features)

**Concept**: 가격대별 거래량 분포 분석 - 기관 투자자들이 활발히 거래한 영역 파악

| Feature | Description | Trading Signal |
|---------|-------------|----------------|
| `vp_poc` | Point of Control (최대 거래량 가격대) | POC 근처 = 공정가치 영역 |
| `vp_value_area_high` | 거래량 70% 구간 상단 | 위 = 과매수 (SHORT 기회) |
| `vp_value_area_low` | 거래량 70% 구간 하단 | 아래 = 과매도 (LONG 기회) |
| `vp_distance_to_poc_pct` | POC까지 거리 (%) | 멀수록 극단적 |
| `vp_in_value_area` | Value Area 내부 여부 (1/0) | 1 = 정상 범위 |
| `vp_percentile` | VP 상 위치 (0~1) | 0 = 최저, 1 = 최고 |
| `vp_volume_imbalance` | POC 위/아래 거래량 불균형 | >0.2 = 매수세, <-0.2 = 매도세 |

**Example Trading Logic**:
```python
# LONG signal: Below Value Area + Buy imbalance
long_signal = (
    (df['close'] < df['vp_value_area_low']) &
    (df['vp_volume_imbalance'] > 0.1)
)

# SHORT signal: Above Value Area + Sell imbalance
short_signal = (
    (df['close'] > df['vp_value_area_high']) &
    (df['vp_volume_imbalance'] < -0.1)
)
```

**Validation Results** (31,488 candles):
- Near POC (<0.5%): 76.8% of time
- In Value Area: 63.6% of time
- Above Value Area: 18.7% (potential SHORT zones)
- Below Value Area: 17.6% (potential LONG zones)
- Strong Buy Imbalance (>0.2): 39.5%
- Strong Sell Imbalance (<-0.2): 26.9%

---

### VWAP (4 features)

**Concept**: 거래량 가중 평균가 - 기관 투자자 벤치마크

| Feature | Description | Trading Signal |
|---------|-------------|----------------|
| `vwap` | VWAP 가격 | VWAP = 공정가치 |
| `vwap_distance_pct` | VWAP까지 거리 (%) | 멀수록 극단적 |
| `vwap_above` | VWAP 위/아래 (1/0) | Cross = 추세 전환 |
| `vwap_band_position` | VWAP 밴드 내 위치 (0~1) | >0.8 = 과매수, <0.2 = 과매도 |

**Example Trading Logic**:
```python
# VWAP crossover (bullish)
vwap_cross_up = (
    (df['vwap_above'] == 1) &
    (df['vwap_above'].shift(1) == 0)
)

# VWAP extreme (overbought)
vwap_overbought = df['vwap_band_position'] > 0.8

# VWAP extreme (oversold)
vwap_oversold = df['vwap_band_position'] < 0.2
```

**Validation Results** (31,488 candles):
- Above VWAP: 53.9% (slightly bullish bias)
- Near VWAP (<0.5%): 49.9%
- VWAP Cross Up: 486 signals
- VWAP Cross Down: 486 signals (balanced)
- Overbought (>0.8): 24.4%
- Oversold (<0.2): 18.5%

---

## 🔧 Advanced Usage

### Custom Parameters

```python
from scripts.indicators.advanced_indicators import (
    calculate_volume_profile,
    calculate_vwap
)

# Custom Volume Profile (longer lookback, more bins)
df = calculate_volume_profile(
    df,
    lookback=200,      # 200 candles (default: 100)
    bins=50,           # 50 bins (default: 20)
    value_area_pct=0.80  # 80% VA (default: 0.70)
)

# Custom VWAP (shorter period)
df = calculate_vwap(
    df,
    period_candles=144  # 12 hours (default: 288 = 24h)
)
```

### Combining Multiple Signals

```python
# Strong LONG opportunity (3 confirmations)
strong_long = (
    # 1. Below Value Area (oversold by volume)
    (df['close'] < df['vp_value_area_low']) &

    # 2. Below VWAP (weak price)
    (df['vwap_above'] == 0) &

    # 3. Buy imbalance (volume accumulation)
    (df['vp_volume_imbalance'] > 0.1)
)

# Strong SHORT opportunity (3 confirmations)
strong_short = (
    # 1. Above Value Area (overbought by volume)
    (df['close'] > df['vp_value_area_high']) &

    # 2. Above VWAP (strong price)
    (df['vwap_above'] == 1) &

    # 3. Sell imbalance (volume distribution)
    (df['vp_volume_imbalance'] < -0.1)
)

print(f"Strong LONG opportunities: {strong_long.sum()}")
print(f"Strong SHORT opportunities: {strong_short.sum()}")
```

**Last 100 Candles**:
- Strong LONG: 8 opportunities
- Strong SHORT: 6 opportunities

---

## ✅ Validation Results

### Test Environment
- **Dataset**: BTCUSDT 5m, 31,488 candles
- **Test Date**: 2025-10-23
- **Script**: `scripts/indicators/test_advanced_indicators.py`

### Performance Metrics
```
✅ Feature Count: 11
✅ Data Coverage: 31,201 / 31,488 (99.1%)
✅ Calculation Speed: 417.3 candles/sec
✅ Quality Issues: 0

Performance Benchmark:
   1,000 candles:   2.19s (457.3 candles/sec)
   5,000 candles:  11.87s (421.3 candles/sec)
  10,000 candles:  23.56s (424.5 candles/sec)
  20,000 candles:  46.80s (427.3 candles/sec)
```

### Feature Quality
```
All 11 features passed quality checks:
- Missing values: 0%
- Extreme zeros: None (all < 90%)
- Unique values: Sufficient variance
- Mean/Std: Normal distributions
```

---

## 📁 File Structure

```
scripts/indicators/
├── __init__.py                      # Module exports
├── advanced_indicators.py           # Core implementation ✅
└── test_advanced_indicators.py      # Validation tests ✅

claudedocs/
├── ADVANCED_INDICATORS_PROPOSAL_20251023.md      # Research doc
└── ADVANCED_INDICATORS_INTEGRATION_GUIDE.md      # This file
```

---

## 🔄 Integration with Existing System

### Option 1: Add to Feature Calculator (Recommended)

**Modify**: `scripts/experiments/calculate_features_with_longterm.py`

```python
# Add after existing imports
from scripts.indicators.advanced_indicators import calculate_all_advanced_indicators

def calculate_all_features_enhanced(df):
    """
    Calculate ALL features: Baseline + Long-term + Advanced

    Total: 151 features
    - Baseline: 107 features (traditional TA)
    - Long-term: 23 features (200-period)
    - Advanced: 11 features (Volume Profile + VWAP)  ← NEW!
    """
    # Baseline features
    df_enhanced = calculate_all_features(df)

    # Long-term features (200-period)
    df_enhanced = calculate_long_term_features(df_enhanced)

    # Advanced indicators (Volume Profile + VWAP)
    df_enhanced = calculate_all_advanced_indicators(df_enhanced, phase='phase1')

    return df_enhanced
```

### Option 2: Standalone Usage

```python
# Use separately without modifying existing code
from scripts.indicators.advanced_indicators import calculate_all_advanced_indicators

# Your existing feature calculation
df = calculate_all_features(df)
df = calculate_long_term_features(df)

# Add advanced indicators
df = calculate_all_advanced_indicators(df, phase='phase1')
```

---

## 📊 Expected Model Impact

Based on research and industry benchmarks:

### Current Performance (Baseline + Long-term)
- Win Rate: 63.6%
- Sharpe Ratio: 0.336
- Max Drawdown: -12.2%

### Expected with Phase 1 (Volume Profile + VWAP)
- Win Rate: **65-67%** (+1.4-3.4%p)
- Sharpe Ratio: **0.37-0.40** (+10-19%)
- Max Drawdown: **-10% to -11%** (+10-18% improvement)

### Key Improvements
1. **Better Entry Timing**: Volume Profile identifies institutional activity zones
2. **Reduced False Signals**: VWAP filters low-quality setups
3. **Risk Management**: Value Area provides clear support/resistance
4. **Institutional Alignment**: Trade with smart money, not against

---

## 🧪 Testing & Validation

### Run Full Validation
```bash
cd bingx_rl_trading_bot
python scripts/indicators/test_advanced_indicators.py
```

### Quick Test
```python
from scripts.indicators.advanced_indicators import calculate_all_advanced_indicators
import pandas as pd

# Load data
df = pd.read_csv("data/historical/BTCUSDT_5m_max.csv")

# Calculate
df_enhanced = calculate_all_advanced_indicators(df, phase='phase1')

# Verify
print(f"Features added: {11}")
print(f"Rows: {len(df)} → {len(df_enhanced)} (lost {len(df) - len(df_enhanced)})")
print(f"Sample:\n{df_enhanced[['vp_poc', 'vwap']].tail()}")
```

---

## 🚀 Next Steps

### Phase 2: Volume Flow Indicators (Ready to Implement)
```python
# Already implemented in advanced_indicators.py
df = calculate_all_advanced_indicators(df, phase='phase2')
# Adds 13 more features: OBV, A/D Line, CMF, MFI
```

### Phase 3: Ichimoku Cloud (Planned)
- 10 features: Tenkan, Kijun, Senkou A/B, Chikou, Cloud analysis

### Phase 4: Channels & Force Index (Planned)
- 13 features: Keltner, Donchian, Elder Force Index

### Model Training
Once satisfied with indicator selection:
1. Update feature calculation script
2. Train new models with Trade-Outcome Labeling
3. Backtest validation (30-day)
4. Deploy to testnet

---

## 📝 Notes

### Design Principles
1. **Modularity**: Each indicator set is independent
2. **Composability**: Can combine any phase
3. **Testability**: Validation tests for quality assurance
4. **Documentation**: Clear trading signals and usage

### Known Limitations
1. **Lookback Cost**: Volume Profile requires 100 candles (lost 287 rows)
2. **Computation**: ~420 candles/sec (slower than basic TA)
3. **Memory**: Additional 11 columns per DataFrame

### Best Practices
1. Always run validation tests before production
2. Monitor feature quality metrics
3. Backtest thoroughly with new features
4. Start with Phase 1, add Phase 2 only if needed

---

## 📚 References

**Research Document**: `claudedocs/ADVANCED_INDICATORS_PROPOSAL_20251023.md`
- Complete theoretical background
- 47 total proposed indicators
- Expected performance impact
- Implementation roadmap

**Key Concepts**:
- Volume Profile: Market Profile theory (J. Peter Steidlmayer)
- VWAP: Institutional execution benchmark
- Volume Flow: Price-volume divergence analysis

---

**Last Updated**: 2025-10-23
**Status**: ✅ Phase 1 Complete & Validated
**Next Action**: Decide Phase 2 implementation or model training
