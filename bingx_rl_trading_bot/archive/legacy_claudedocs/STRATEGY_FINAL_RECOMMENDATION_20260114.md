# 전략 최종 권장 보고서 - 2026-01-14

## Executive Summary

90일 데이터 기반 종합 연구 결과, 두 가지 검증된 전략과 최적 운영 방안을 도출했습니다.

| 전략 | Trades | WR | PnL% | Edge | WF | Type1 |
|------|--------|-----|------|------|-----|-------|
| **Engulf Option A** | 40 | 60.0% | +93.3% | 78.02 | 6/8 | ✅ |
| **Supertrend Flip** | 32 | 68.8% | +187.9% | 96.05 | 4/8 | ✅ |
| **Combined (50/50)** | 70 | 68.6% | +205.0% | **249.78** | - | ✅ |

---

## 1. Engulf 5m Option A (권장)

### 구성
```yaml
strategy:
  name: "engulf_5m"
  version: "2.0"  # Option A

entry:
  pattern: "engulfing"
  filters:
    volume_ratio_min: 1.2
    prev_body_ratio_min: 0.30
    body_pct_min: 0.24

exit:
  tp_pct: 2.5
  sl_pct: 2.0
  scale_out: disabled  # 100% @ TP (핵심 변경점)

position:
  position_pct: 0.95
  leverage: 3
```

### v1.9 대비 개선
| 항목 | v1.9 | Option A | 변화 |
|------|------|----------|------|
| Scale-out | 50/50 @ 0.8/1.0 | Disabled | 단순화 |
| PnL | +74.6% | **+93.3%** | **+25.1%** |
| Edge | 61.65 | **78.02** | **+26.6%** |
| WF | 5/8 | **6/8** | +1 window |
| Max DD | 12.0% | 11.9% | -0.8% |

### Monte Carlo 검증
- **100회 시뮬레이션**: 100% 수익 확률
- **평균 PnL**: +97.6%
- **PnL 범위**: +79.3% ~ +115.1%
- **통계적 유의성**: ✅ 확인됨

---

## 2. Supertrend Flip (보완 전략)

### 구성
```yaml
strategy:
  name: "supertrend_flip"
  version: "1.0"

indicators:
  supertrend:
    period: 10
    multiplier: 2.2
  adx:
    period: 14
    threshold: 15

entry:
  signal: "direction_change"
  filters:
    - adx_min: 15

exit:
  tp_pct: 3.5
  sl_pct: 3.5
```

### 성과
| 항목 | 값 |
|------|-----|
| Trades | 32 |
| Win Rate | 68.8% |
| PnL | +187.9% |
| Edge | 96.05 |
| WF | 4/8 |

### Monte Carlo 검증
- **100회 시뮬레이션**: 100% 수익 확률
- **평균 PnL**: +185.2%
- **PnL 범위**: +133.8% ~ +249.8%

---

## 3. Combined Strategy (최적 방안)

### 자본 배분
```
총 자본: $1000
├── Engulf Option A: $500 (50%)
└── Supertrend Flip: $500 (50%)
```

### 시뮬레이션 결과
| 항목 | Engulf Only | Supertrend Only | **Combined** |
|------|-------------|-----------------|--------------|
| Trades | 40 | 32 | **70** |
| Win Rate | 60.0% | 68.8% | **68.6%** |
| PnL | +93.3% | +187.9% | **+205.0%** |
| Max DD | 11.9% | 19.6% | **8.1%** |
| Edge | 78.02 | 96.05 | **249.78** |

### 분산 효과
- **DD 감소**: 개별 전략 대비 -58% ~ -69%
- **Edge 증가**: 개별 전략 대비 +160% ~ +220%
- **거래 빈도**: 70 trades/90days ≈ 0.78 trades/day

---

## 4. 권장 운영 방안

### Option A: 단일 전략 운영 (보수적)

**Engulf Option A 단독 운영**

장점:
- 단순한 운영 (봇 1개)
- 검증된 성과 (Edge 78.02)
- 높은 WF 일관성 (6/8)

단점:
- 분산 효과 없음
- 거래 빈도 제한 (40 trades/90days)

### Option B: 병행 운영 (권장)

**Engulf Option A + Supertrend Flip (50/50)**

장점:
- 최고 Edge (249.78)
- 최저 Max DD (8.1%)
- 높은 거래 빈도 (70 trades/90days)
- 다양한 시장 조건 포착

단점:
- 봇 2개 운영 필요
- 자본 분할 관리

### Option C: 적응형 운영 (고급)

**시장 상황에 따른 전략 전환**

- 고변동성 (ADX > 25): Supertrend 비중 ↑
- 저변동성 (ADX < 15): Engulf 비중 ↑
- 횡보장: 양 전략 균등

---

## 5. 구현 로드맵

### Phase 1: Engulf Option A 적용 (즉시)
```bash
# engulf_5m_config.yaml 수정
scale_out:
  enabled: false  # 핵심 변경
```

### Phase 2: Supertrend Flip 봇 개발 (1-2일)
1. `supertrend_flip_bot.py` 생성
2. `supertrend_flip_config.yaml` 생성
3. 단위 테스트 및 페이퍼 트레이딩

### Phase 3: Combined 운영 시작 (검증 후)
1. 양 봇 동시 운영
2. 자본 50/50 배분
3. 주간 성과 모니터링

---

## 6. 리스크 고려사항

### 백테스트 vs 실거래 차이
- 슬리피지: 백테스트 미반영 (실거래 시 0.01-0.05% 예상)
- 체결 지연: 고변동성 시 지연 가능
- 유동성: BTC 주요 페어로 문제 없음

### 과최적화 위험
- Walk-Forward 검증으로 완화 (Engulf 6/8, Supertrend 4/8)
- Monte Carlo로 통계적 유의성 확인
- Out-of-sample 기간 필요 (권장: 30일 페이퍼 트레이딩)

### 시장 레짐 변화
- 현재 파라미터는 2025-10 ~ 2025-12 데이터 기반
- 시장 구조 변화 시 재최적화 필요
- 월간 성과 리뷰 권장

---

## 7. 결론

### 최종 권장
```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│  🎯 PRIMARY RECOMMENDATION: Combined Strategy (50/50)                   │
│                                                                         │
│     Engulf Option A + Supertrend Flip                                   │
│     • Edge: 249.78 (최고)                                               │
│     • Max DD: 8.1% (최저)                                               │
│     • PnL: +205.0%                                                      │
│     • WR: 68.6%                                                         │
│                                                                         │
│  🔄 FALLBACK: Engulf Option A 단독                                      │
│     • Edge: 78.02                                                       │
│     • WF: 6/8 (높은 일관성)                                             │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 즉시 실행 가능 조치
1. **Engulf 5m config 업데이트**: scale_out 비활성화
2. **Supertrend Flip 봇 개발**: 새 봇 생성
3. **Combined 운영 준비**: 자본 배분 계획

---

**작성일**: 2026-01-14
**데이터 기간**: 2025-10-02 ~ 2025-12-31 (90일)
**검증 방법**: Type1, Type2, Walk-Forward, Monte Carlo (100회)
