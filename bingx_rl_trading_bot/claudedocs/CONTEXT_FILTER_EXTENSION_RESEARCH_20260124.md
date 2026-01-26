# Context Filter Extension Research Report

**Date**: 2026-01-24
**Version**: v1.8 Candidate Research
**Author**: Claude (Trading Research)

---

## Executive Summary

v1.7에서 적용된 Context Filter의 효과를 확인하고, 나머지 5개 패턴에 대한 추가 필터 가능성을 분석했습니다.

### Key Findings

| 발견 | 상세 |
|------|------|
| ✅ v1.7 필터 효과 확인 | U-DN-DN + RSI<30 = **+223.1%** 개선 |
| 🔍 추가 필터 후보 발견 | 2개 패턴에 **회피 필터** 유효 |
| ⚠️ 대부분 패턴은 필터 불필요 | 5개 중 3개는 현상 유지 권장 |

---

## Research Methodology

### Data
- Source: `btc_5m_extended.csv` (105일, 30,232 candles)
- Period: 2025-10-08 ~ 2026-01-21

### Analysis Approach
1. **Positive Filter**: 특정 컨텍스트에서만 진입 (v1.7 방식)
2. **Exclusion Filter**: 특정 컨텍스트 회피 (신규 분석)

### Context Features
| Feature | Values | Description |
|---------|--------|-------------|
| `above_ema200` | True/False | EMA(200) 대비 가격 위치 |
| `vol` | L/M/H | ATR 기반 변동성 (33%/66% quantile) |
| `rsi_zone` | OS/N/OB | RSI(14) 구간 (<30 / 30-70 / >70) |
| `trend` | UP/DN | 20봉 기준 추세 |

---

## Results

### 1. v1.7 Filter Verification

**U-DN-DN + RSI Oversold (Required Filter)**

| Metric | Baseline | Filtered | Change |
|--------|----------|----------|--------|
| Trades | 321 | 36 | -89% |
| Win Rate | 52.3% | **72.2%** | +19.9% |
| Return | -52.5% | **+170.6%** | **+223.1%** |

✅ **결론**: v1.7 필터 효과 확인됨. 신호 빈도는 줄지만 품질이 크게 향상됨.

---

### 2. Remaining Patterns Analysis

**분석 대상** (v1.7에서 필터 없는 패턴):
- BD-BD-BD (SHORT)
- MU-ST-DN (SHORT)
- MU-ST-ST (SHORT)
- IH-DN-DN (SHORT)
- MU-U-DN (LONG)

#### Pattern-by-Pattern Results

| Pattern | Dir | Baseline | Best Context | Result |
|---------|-----|----------|--------------|--------|
| BD-BD-BD | SHORT | +60.4% | 모든 컨텍스트 하락 | ❌ 필터 불필요 |
| MU-ST-DN | SHORT | +175.2% | 모든 컨텍스트 하락 | ❌ 필터 불필요 |
| MU-ST-ST | SHORT | +56.8% | 모든 컨텍스트 하락 | ❌ 필터 불필요 |
| IH-DN-DN | SHORT | +38.6% | 모든 컨텍스트 하락 | ⚠️ 회피 필터 가능 |
| MU-U-DN | LONG | +152.2% | 모든 컨텍스트 하락 | ❌ 필터 불필요 |

**핵심 발견**: 나머지 5개 패턴은 추가 Positive Filter로 개선되지 않음.

---

### 3. Exclusion Filter Analysis (신규 발견)

**회피 필터**: 특정 컨텍스트를 피하면 성과 개선

| Pattern | EXCLUDE | Drop Trades | Drop Return | Keep Trades | Keep WR | Keep Return | Δ |
|---------|---------|-------------|-------------|-------------|---------|-------------|---|
| **IH-DN-DN** | vol=H | 12 | -14.3% | 21 | **76.2%** | **+61.8%** | **+23.2%** |
| BD-BD-BD | vol=L | 9 | -9.1% | 23 | 56.5% | +76.5% | +16.1% |

#### IH-DN-DN Exclusion Filter Detail

```
IH-DN-DN (SHORT) - High Volatility 회피 시:
├─ 제외: 12 trades (vol=H에서 -14.3% 손실)
├─ 유지: 21 trades (vol=L,M에서 76.2% WR, +61.8%)
└─ 개선: +23.2% (38.6% → 61.8%)
```

**Insight**: IH-DN-DN 패턴은 고변동성 환경에서 성과가 나쁨. 저/중변동성에서만 진입 권장.

---

## Recommendations

### v1.8 Candidate Implementation

```python
PATTERN_CONTEXT_FILTERS = {
    # Existing v1.7 filters
    'U-DN-DN': {
        'required': {'rsi_zone': ['OS']},  # RSI < 30 only
    },
    'DN-DN-BD': {
        'preferred': {'vol': ['H']},  # Bonus for high volatility
    },
    'U-BU-U': {
        'preferred': {'trend': ['DN']},  # Bonus for downtrend
    },

    # NEW v1.8 exclusion filters
    'IH-DN-DN': {
        'excluded': {'vol': ['H']},  # Avoid high volatility
    },
}
```

### Implementation Priority

| Priority | Pattern | Filter Type | Expected Improvement |
|----------|---------|-------------|----------------------|
| 🟢 High | IH-DN-DN | Exclude vol=H | +23.2% |
| 🟡 Medium | BD-BD-BD | Exclude vol=L | +16.1% |
| ⚪ Low | Others | No change | - |

---

## Trade-off Analysis

### IH-DN-DN Exclusion Filter

| Metric | Before | After | Trade-off |
|--------|--------|-------|-----------|
| Trades | 33 | 21 | -36% 신호 감소 |
| Win Rate | 66.7% | 76.2% | +9.5% 승률 증가 |
| Return | +38.6% | +61.8% | +23.2% 수익 증가 |

**결론**: 신호 빈도 감소를 감수할 만한 가치 있음.

---

## Statistical Confidence

| Filter | Sample Size | Confidence |
|--------|-------------|------------|
| U-DN-DN RSI_OS | 36 trades | ⚠️ Medium (더 많은 데이터 필요) |
| IH-DN-DN vol≠H | 21 trades | ⚠️ Medium (더 많은 데이터 필요) |

**Note**: 샘플 크기가 작아 실제 운영에서 추가 검증 필요.

---

## Next Steps

1. **v1.8 구현**: IH-DN-DN exclusion filter 추가
2. **모니터링**: v1.7 필터 실제 성과 추적
3. **데이터 축적**: 더 많은 샘플로 통계적 신뢰도 확보

---

## Appendix: Full Analysis Output

### Remaining Patterns Detailed Results

```
BD-BD-BD (SHORT) - Baseline: 32 trades, WR 46.9%, Return +60.4%
  vol=H: 12 trades, WR 58.3%, Return +37.6% (Δ-22.8%)
  vol=M: 11 trades, WR 54.5%, Return +28.3% (Δ-32.1%)
  vol=L: 9 trades, WR 22.2%, Return -9.1% (Δ-69.5%)

MU-ST-DN (SHORT) - Baseline: 60 trades, WR 56.7%, Return +175.2%
  All contexts perform worse than baseline

MU-ST-ST (SHORT) - Baseline: 29 trades, WR 75.9%, Return +56.8%
  All contexts perform worse than baseline

IH-DN-DN (SHORT) - Baseline: 33 trades, WR 66.7%, Return +38.6%
  vol=H: 12 trades, WR 50.0%, Return -14.3% (Δ-52.9%) ← AVOID
  vol=M: 9 trades, WR 77.8%, Return +25.3%
  vol=L: 12 trades, WR 75.0%, Return +36.5%

MU-U-DN (LONG) - Baseline: 62 trades, WR 37.1%, Return +152.2%
  All contexts perform worse than baseline
```

---

**Report Generated**: 2026-01-24 09:20 KST
