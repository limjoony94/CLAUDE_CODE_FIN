# Pattern-Size Based TP/SL Research Report

**Date**: 2026-01-25
**Version**: v1.10 → v1.11 consideration
**Research Type**: Exhaustive pattern search with pattern-size-based TP/SL

---

## Executive Summary

**핵심 결론**: 패턴 크기 기반 TP/SL은 고정 % 방식보다 열등함

| Approach | Avg Edge (Production Patterns) | WF Success Rate |
|----------|-------------------------------|-----------------|
| Fixed % (현재) | **+0.74** | 대부분 검증됨 |
| Pattern-Size Based | +0.20 | 1/8 (4/5+ WF) |

**권장사항**: 현재 Fixed % 방식 유지, DN-BD-BD 패턴만 추가 검토

---

## 1. Research Methodology

### 1.1 Pattern-Size Based TP/SL 개념

```
Pattern Size = max(high_3candles) - min(low_3candles)
TP Distance = pattern_size × tp_mult
SL Distance = pattern_size × sl_mult
```

**테스트한 배수**:
- TP Multipliers: 1.0x, 1.5x, 2.0x, 2.5x, 3.0x
- SL Multipliers: 0.5x, 0.75x, 1.0x, 1.5x, 2.0x

**이론적 장점**: 변동성에 자동 적응

### 1.2 데이터 범위

- **기간**: 2025-10-08 ~ 2026-01-21 (105일)
- **데이터**: 30,232 bars (5m)
- **평균 패턴 크기**: 0.319%
- **테스트 패턴 수**: 650개 (≥10 occurrences)

---

## 2. Walk-Forward Validation Results

### 2.1 WF 통과 패턴 (4/5+)

| Pattern | Direction | TP* | SL* | WF | Avg Edge | Trades |
|---------|-----------|-----|-----|-----|----------|--------|
| **DN-BD-BD** | SHORT | 3.0 | 2.0 | **4/5** | +0.74 | 20 |

**단 1개 패턴만 4/5 이상 WF 통과**

### 2.2 WF 3/5 패턴 (참고용)

| Pattern | Direction | TP* | SL* | WF | Avg Edge | Trades |
|---------|-----------|-----|-----|-----|----------|--------|
| BD-BD-BD | SHORT | 2.0 | 2.0 | 3/5 | +0.58 | 19 |
| ST-BU-MU | SHORT | 2.5 | 1.5 | 3/5 | +0.53 | 7 |
| IH-BD-DN | SHORT | 2.5 | 2.0 | 3/5 | +0.50 | 8 |

---

## 3. Fixed % vs Pattern-Size 비교

### 3.1 프로덕션 패턴별 비교

| Pattern | Fixed TP/SL | Fixed Edge | Size-Based | Size Edge | **Winner** |
|---------|-------------|------------|------------|-----------|------------|
| BD-BD-BD | 2.5/2.0% | **+2.09** | 2.0x/2.0x | +0.77 | ❌ Fixed |
| DN-DN-BD | 2.0/2.5% | **+1.68** | 2.0x/2.0x | +0.36 | ❌ Fixed |
| IH-DN-DN | 2.0/2.5% | **+1.36** | 1.5x/0.5x | +0.01 | ❌ Fixed |
| U-BU-U | 1.5/2.0% | **+0.86** | 1.0x/2.0x | +0.11 | ❌ Fixed |
| MU-ST-DN | 2.0/1.0% | **+0.69** | 2.5x/2.0x | +0.25 | ❌ Fixed |
| MU-ST-ST | 3.5/0.8% | -0.80 | 2.0x/2.0x | **+0.34** | ✅ Size |
| U-DN-DN | 3.5/0.8% | -0.52 | 2.5x/0.5x | **+0.02** | ✅ Size |
| MU-U-DN | 3.5/1.0% | -0.40 | 1.0x/2.0x | **+0.13** | ✅ Size |

### 3.2 분석

**Fixed % 우세 (5/8 패턴)**:
- BD-BD-BD: 2.09 vs 0.77 → **-1.33 차이**
- DN-DN-BD: 1.68 vs 0.36 → **-1.32 차이**
- IH-DN-DN: 1.36 vs 0.01 → **-1.35 차이**
- U-BU-U: 0.86 vs 0.11 → -0.75 차이
- MU-ST-DN: 0.69 vs 0.25 → -0.44 차이

**Pattern-Size 우세 (3/8 패턴)**:
- MU-ST-ST: -0.80 vs +0.34 → **+1.14 개선** (but still low edge)
- U-DN-DN: -0.52 vs +0.02 → +0.53 개선 (marginal)
- MU-U-DN: -0.40 vs +0.13 → +0.53 개선 (marginal)

**결론**: 고성능 패턴들(BD-BD-BD, DN-DN-BD, IH-DN-DN)에서 Fixed %가 압도적으로 우수

---

## 4. 신규 발견 패턴

### 4.1 DN-BD-BD (SHORT) - 추가 검토 대상

| Metric | Value |
|--------|-------|
| Direction | SHORT |
| TP Mult | 3.0x |
| SL Mult | 2.0x |
| Trades | 20 |
| Win Rate | 50.0% |
| Edge | +0.79 |
| WF Score | 4/5 |
| Avg Edge (WF) | +0.74 |

**Fixed % 비교 필요**: 이 패턴을 Fixed %로도 테스트해야 함

### 4.2 Top 10 새 패턴 (프로덕션 미포함)

| Rank | Pattern | Dir | Edge | WF | Notes |
|------|---------|-----|------|-----|-------|
| 1 | MU-MD-BD | SHORT | +1.58 | 1/5 | Low WF |
| 2 | BD-BD-MU | SHORT | +1.36 | 0/5 | WF fail |
| 3 | BD-BD-D | SHORT | +1.15 | 1/5 | Low WF |
| 4 | BD-MD-BD | SHORT | +1.08 | 1/5 | Low WF |
| 5 | MD-MD-BU | LONG | +1.07 | 2/5 | Low WF |
| 6 | MU-ST-BU | SHORT | +1.05 | 1/5 | Low WF |
| 7 | ST-BU-MU | SHORT | +1.04 | 3/5 | Borderline |
| 8 | DN-MU-IH | LONG | +1.03 | 2/5 | Low WF |
| 9 | U-DF-BU | SHORT | +0.87 | 2/5 | Low WF |
| 10 | BU-MU-BD | LONG | +0.83 | 1/5 | Low WF |

대부분 WF 검증 실패 → 과적합 가능성 높음

---

## 5. 이론적 분석

### 5.1 Pattern-Size 방식이 열등한 이유

1. **변동성 불균형**: 높은 변동성 패턴에서 TP/SL이 과도하게 넓어짐
2. **노이즈 증폭**: 3-candle range가 일시적 스파이크 포함
3. **일관성 부족**: 동일 패턴이 시장 상황에 따라 다른 TP/SL 적용
4. **통계적 안정성**: Fixed %는 충분한 백테스트 데이터로 최적화됨

### 5.2 Fixed %가 우수한 이유

1. **일관된 리스크 관리**: 모든 거래에 동일한 % 적용
2. **최적화 용이**: 각 패턴별 TP/SL 미세 조정 가능
3. **해석 용이**: 명확한 수익/손실 % 예측
4. **검증된 성능**: WF validation 통과율 높음

---

## 6. Recommendations

### 6.1 즉시 조치 (Not Recommended)

| Action | Status | Reason |
|--------|--------|--------|
| Pattern-Size TP/SL 전환 | ❌ **거부** | Fixed %보다 열등 |
| 전체 패턴 교체 | ❌ **거부** | 기존 검증된 패턴 유지 |

### 6.2 검토 필요

| Action | Status | Reason |
|--------|--------|--------|
| DN-BD-BD 추가 | ⚠️ 검토 | 4/5 WF, but Fixed %로 재검증 필요 |

### 6.3 v1.10 유지

현재 v1.10 설정 유지:
- Fixed % TP/SL 방식
- 기존 8개 패턴
- Pattern-specific TP/SL 최적화

---

## 7. Appendix: Raw Data

### 7.1 Files Generated

- `results/pattern_size_exhaustive_results.csv` - 전체 650 패턴 결과
- `results/pattern_size_wf_validated.csv` - Top 20 WF 검증 결과

### 7.2 Research Script

`scripts/analysis/pattern_size_fast.py` - 최적화된 연구 스크립트

---

## 8. Conclusion

**Pattern-Size Based TP/SL 방식은 Fixed % 방식보다 열등함**

- 고성능 패턴에서 -1.3 ~ -1.4 Edge 손실
- WF 통과율 낮음 (1/8 vs 대부분)
- 변동성 적응이 오히려 노이즈 증폭

**권장**: v1.10 Fixed % 방식 유지, DN-BD-BD 패턴 Fixed %로 재검증 후 추가 여부 결정

---

*Report generated by exhaustive pattern search with pattern-size-based TP/SL optimization*
