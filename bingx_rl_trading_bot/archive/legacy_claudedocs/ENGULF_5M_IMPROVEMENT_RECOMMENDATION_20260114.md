# Engulf 5m 개선 권장사항 - 2026-01-14

## Executive Summary

**90일 백테스트 기반 v1.9 vs Option A 비교 분석 완료**

| 항목 | Current v1.9 | Option A | 변화 |
|------|-------------|----------|------|
| Scale-out | 50/50 @ 0.8/1.0 | Disabled | - |
| Weighted TP | 90% | 100% | +10% |
| **Total PnL** | +74.6% | **+93.3%** | **+25.1%** |
| **Edge** | 61.65 | **78.02** | **+26.6%** |
| Max DD | 12.0% | 11.9% | -0.8% |
| Walk-Forward | 5/8 | **6/8** | +1 window |
| Final Balance | $1,746 | **$1,933** | +$187 |

---

## 결론: ✅ OPTION A 적용 권장

### 핵심 변경사항

```yaml
# 변경 전 (v1.9)
scale_out:
  enabled: true
  stages:
    - [0.50, 0.8]   # 50% @ 80% of TP
    - [0.50, 1.0]   # 50% @ 100% of TP

# 변경 후 (Option A)
scale_out:
  enabled: false
```

---

## 상세 분석

### 1. PnL 개선 (+25.1%)

| 구성 | PnL | 이유 |
|------|-----|------|
| v1.9 | +74.6% | 50%는 80% TP에서 조기 익절 → 90% 실효 TP |
| Option A | +93.3% | 100% 포지션이 100% TP 도달 |

**수학적 분석:**
- v1.9 Weighted TP: (0.5 × 0.8 + 0.5 × 1.0) × 2.5% = 2.25%
- Option A Full TP: 1.0 × 2.5% = 2.5%
- **차이: +11.1% per winning trade**

### 2. Edge 개선 (+26.6%)

```
Edge = (Final Balance - Initial) / (Max DD + 0.1)

v1.9:    (1746 - 1000) / (120 + 0.1) = 61.65
Option A: (1933 - 1000) / (119 + 0.1) = 78.02
```

**원인:**
- PnL 증가 (분자 ↑)
- DD 감소 (분모 ↓)

### 3. Walk-Forward 개선 (5/8 → 6/8)

| Window | v1.9 | Option A |
|--------|------|----------|
| 1 | ✅ | ✅ |
| 2 | ❌ | ✅ |
| 3 | ✅ | ✅ |
| 4 | ❌ | ❌ |
| 5 | ✅ | ✅ |
| 6 | ✅ | ✅ |
| 7 | ❌ | ❌ |
| 8 | ✅ | ✅ |

**의미:** Option A가 더 다양한 시장 조건에서 일관된 성과

### 4. Risk Analysis

| 지표 | v1.9 | Option A | 평가 |
|------|------|----------|------|
| Max Consecutive Losses | 3 | 3 | 동일 |
| Avg Win | $86.29 | $104.67 | +21% |
| Avg Loss | $82.79 | $98.67 | +19% |
| Risk:Reward | 1.04 | 1.06 | 개선 |
| EV per Trade | $18.66 | $23.34 | +25% |

**결론:** Option A는 승리 시 더 높은 수익 (RR 개선)

---

## 방향별 성과 (Option A)

| 방향 | Trades | Win Rate | 평가 |
|------|--------|----------|------|
| LONG | 18 | 61.1% | ✅ 양호 |
| SHORT | 22 | 59.1% | ✅ 양호 |
| **Total** | **40** | **60.0%** | ✅ 균형 |

---

## 심리적 고려사항

### Scale-out의 장점 (v1.9)
- 부분 익절로 심리적 안정감
- "일부라도 먹었다"는 만족감
- 변동성 높은 시장에서 조기 수익 확보

### No Scale-out의 장점 (Option A)
- 단순한 로직 (전량 TP 또는 SL)
- 더 높은 수익 잠재력
- 관리/모니터링 간소화

### 권장 접근법
1. **Option A 적용** - 백테스트 결과 기반 최적 선택
2. **30일 모니터링** - 실거래 성과 추적
3. **필요시 재평가** - 실거래에서 심리적 압박 과도 시 v1.9 복귀 고려

---

## 적용 방법

### 1. Config 파일 수정

```bash
# 파일 위치
bingx_rl_trading_bot/config/engulf_5m_config.yaml
```

```yaml
# 변경 사항
scale_out:
  enabled: false  # true → false
```

### 2. 봇 재시작

```bash
STOP_ENGULF_5M.bat
START_ENGULF_5M.bat
```

### 3. 모니터링

```bash
MONITOR_ENGULF_5M.bat
```

---

## 검증 요약

| 검증 항목 | v1.9 | Option A | 통과 기준 |
|----------|------|----------|----------|
| Type1 (WR≥50%, PnL>0) | ✅ PASS | ✅ PASS | WR≥50%, PnL>0 |
| Walk-Forward | ✅ 5/8 | ✅ 6/8 | ≥4/8 |
| Monte Carlo | ✅ 100% | ✅ 100% | 100% 수익 확률 |

---

## 최종 권장사항

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  ✅ OPTION A 적용 권장                                                  │
│                                                                         │
│  변경: scale_out.enabled = false                                        │
│                                                                         │
│  기대 효과:                                                             │
│  • PnL: +74.6% → +93.3% (+25.1% 개선)                                  │
│  • Edge: 61.65 → 78.02 (+26.6% 개선)                                   │
│  • WF: 5/8 → 6/8 (+1 window)                                           │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

**작성일**: 2026-01-14
**데이터 기간**: 2025-10-02 ~ 2025-12-31 (90일)
**검증 방법**: Type1, Walk-Forward, Monte Carlo

**관련 파일:**
- 비교 스크립트: `scripts/analysis/current_vs_optionA_comparison_20260114.py`
- 결과 CSV: `results/v19_vs_optionA_20260114_022723.csv`
- 종합 재검증: `claudedocs/COMPREHENSIVE_REVALIDATION_REPORT_20260114.md`
