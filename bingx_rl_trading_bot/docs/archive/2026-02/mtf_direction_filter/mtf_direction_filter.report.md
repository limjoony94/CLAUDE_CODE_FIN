# MTF Direction Filter — PDCA Research Report

> **Date**: 2026-02-24
> **Scope**: Multi-Timeframe Direction Filter + Tight TP 연구
> **Bot**: BTC 5m Pattern Trading, BingX, 3x Leverage, Hedge Mode, N=9
> **Baseline**: v1.33.0 (35 patterns, 9L+26S, Compact TP/SL, WR 68.1%, edge 0.221%)
> **Status**: STOP — 가설 기각, 현재 전략 유지
> **Type**: Research-only (코드 변경 없음)

---

## 1. Executive Summary

"1h 패턴이 방향을 결정하고, 5m 패턴은 진입 타이밍만 잡는다"는 MTF(Multi-Timeframe) 가설을 4-Phase 연구로 검증했다. **결론: MTF 방향 필터는 현재 전략 대비 유의미한 개선을 제공하지 않는다.**

1h 3-candle 패턴은 방향 예측력을 보유하지만 (best 70.3%), 이를 5m 진입 필터로 적용하면:
- **Permissive 모드**: WR +0.2~1.1pp (미미), 거래 66~92% 유지
- **Strict 모드**: WR +2.3~5.5pp, 거래 23~26% 잔존 (74~95% 손실)
- **3pp WR 개선 기준 미달** — 거래 손실이 WR 개선을 상쇄

| 지표 | Baseline | Best MTF (thr50 strict) | Delta |
|------|----------|-------------------------|-------|
| WR | 64.2% | 66.7% | +2.5pp |
| Edge/trade | 0.181% | 0.436% | +0.255% |
| Trades | 5,781 | 1,510 | -74% |
| Trades/day | 8.03 | 2.10 | -74% |
| Total edge (720d) | 1,046 | 658 | -37% |

**핵심 교훈**: 거래당 edge가 상승해도 거래 수가 74% 감소하면 총 edge가 37% 감소한다.

---

## 2. Plan (Phase 0)

### 가설

| ID | 가설 | 결과 |
|----|------|------|
| H1 | 1h 3-candle 패턴이 다음 N시간의 가격 방향을 50% baseline 대비 유의하게 예측 | **PASS** (15 patterns, best 70.3%) |
| H2 | 1h bias 방향과 일치하는 5m 진입이 전체 대비 높은 WR + edge | **PARTIAL** (strict +2.5pp, but -74% trades) |
| H3 | MTF 필터 + Tight TP 0.5~1.0%에서 수수료 차감 후 양의 edge | **FAIL** (best tight TP edge < baseline edge) |

### 연구 설계

6-Phase Go/No-Go 게이트 구조:
```
Phase 1 (Data) → Phase 2 (1h Direction) → GO? →
Phase 3 (MTF Filter) → WR +3pp? → Phase 4 (Tight TP) →
Phase 5 (WF) → Phase 6 (Production)
```

**실제 진행**: Phase 1 → Phase 2 (GO) → Phase 3 (STOP) → Phase 3X (확장 탐색, STOP) → Phase 4 (참고용 실행)

---

## 3. Research Execution

### Phase 1: Data Preparation

| 항목 | 값 |
|------|-----|
| 5m 데이터 | 207,360 rows (720일, 2024-02-23 ~ 2026-02-12) |
| 1h 집계 | 17,280 rows (5m 12봉 → 1봉) |
| 15m 집계 | 69,120 rows (참고용) |
| 1h unique 패턴 | 1,283 (min_occ >= 50: 65개) |
| 5m↔1h 매핑 | 207,324/207,360 (100.0%) |

**스크립트**: `scripts/analysis/mtf_direction_study.py` (Phase 1-2)

### Phase 2: 1h Directional Accuracy Study

**방법**: 각 1h 3-candle 패턴 출현 후 1h/3h/6h/12h/24h close-to-close 방향 기록, binomial test

| Horizon | 유의(p<0.01) | Strong(acc>55%) | Best accuracy |
|---------|------------|----------------|---------------|
| 1h | 3 | 3 | 68.9% |
| 3h | 5 | 5 | 67.6% |
| 6h | 3 | 3 | 70.3% |
| 12h | 3 | 3 | 65.6% |
| 24h | 4 | 4 | 68.8% |

**Strong 패턴** (acc>55%, p<0.01, N>=50): **15 unique** (10 LONG + 5 SHORT)

Top 5:
| Pattern | Horizon | Direction | Accuracy | N |
|---------|---------|-----------|----------|---|
| D-U-DN | 6h | LONG | 70.3% | 64 |
| U-BD-U | 1h | LONG | 68.9% | 74 |
| U-BU-U | 1h | SHORT | 68.8% | 64 |
| U-DN-D | 24h | LONG | 68.8% | 64 |
| ST-ST-DN | 3h | LONG | 67.6% | 139 |

**판정**: GO (15 strong patterns, best accuracy 70.3%)

### Phase 3: MTF Filter Effect

**설계**: 35개 프로덕션 5m 패턴 신호(8,233개)에 1h 방향 필터 적용

**Original (15 strong patterns, acc>55%, p<0.01, N>=50)**:

| Mode | Trades | WR | Edge | T/day | WR delta |
|------|--------|-----|------|-------|----------|
| Baseline | 5,781 | 64.2% | 0.181% | 8.03 | -- |
| Permissive | 5,362 | 64.4% | 0.211% | 7.45 | +0.2pp |
| Strict | 264 | 69.7% | 0.672% | 0.37 | +5.5pp |

**문제 식별**: 15 strong 패턴이 시간의 ~5%만 커버 → permissive에서 92.5%가 neutral 통과

### Phase 3X: Expanded Threshold Sweep

사용자 피드백으로 확장: min_occ=30, accuracy threshold [50%, 51%, ..., 55%]

| Threshold | 1h 패턴 수 | 커버리지 | Mode | Trades | WR delta | Edge delta | 잔존율 |
|-----------|----------|---------|------|--------|----------|-----------|-------|
| 50% | 142 | 60% | permissive | 3,835 | +0.9pp | +0.106% | 66% |
| 50% | 142 | 60% | strict | 1,510 | +2.5pp | +0.255% | 26% |
| 53% | 139 | 58% | permissive | 3,857 | +0.9pp | +0.105% | 67% |
| 53% | 139 | 58% | strict | 1,456 | +2.3pp | +0.245% | 25% |
| 55% | 132 | 53% | permissive | 4,058 | +1.1pp | +0.125% | 70% |
| 55% | 132 | 53% | strict | 1,328 | +2.4pp | +0.255% | 23% |

**발견**:
1. 50~55% 모든 threshold에서 결과 거의 동일 (패턴 수 142→132, 차이 미미)
2. Permissive: WR +1pp 미만, 거래 66~70% 유지 — 필터 효과 미미
3. Strict: WR +2.3~2.5pp로 일관, 거래 23~26% — 3pp 기준 미달
4. 커버리지 60%에서도 permissive 효과는 +1pp 수준 (자연적 방향 일치가 높기 때문)

### Phase 4: Tight TP Optimization (참고용)

Best sweep (thr50_strict) 위에서 Tight TP grid search 실행:

| TP/SL | WR | WR Excess | Edge | Trades | T/day | Hold |
|-------|-----|-----------|------|--------|-------|------|
| 0.90/2.00 | 76.4% | +7.4pp | 0.344% | 1,832 | 2.54 | 5.2h |
| 0.80/2.00 | 78.8% | +7.4pp | 0.323% | 1,891 | 2.63 | 4.6h |
| 1.00/2.00 | 73.8% | +7.1pp | 0.345% | 1,781 | 2.47 | 5.8h |

**주의**: MDD 86.9% — WF 미검증, 거래 2.54/day (baseline의 32%), IS-only 결과

---

## 4. Decision & Root Cause Analysis

### 최종 판정: **STOP — 현재 v1.33.0 유지**

### MTF 필터가 작동하지 않는 근본 원인

1. **자연적 방향 일치율이 높음**: 5m 패턴 방향과 1h 패턴 방향이 이미 상당 부분 자연적으로 일치. Permissive 필터에서 92.5%가 neutral(방향 정보 없는 1h 패턴) 경유로 통과.

2. **1h 방향 예측력의 한계**: Best accuracy 70.3%이지만 대부분 57~67% 수준. 5m 패턴이 이미 WR 64.2%의 edge를 가진 상태에서 1h 필터가 추가할 정보량이 제한적.

3. **Strict 모드의 거래 손실 문제**: 방향 필터를 엄격하게 적용하면 WR이 개선되지만, 거래의 74~95%를 잃음. 이는 **총 edge의 감소**를 초래:
   - Baseline: 5,781 trades × 0.181% = 1,046 edge points
   - thr50 strict: 1,510 trades × 0.436% = 658 edge points (-37%)

4. **R:R 구조의 영향**: 현재 전략은 R:R < 1.0 (avg_win 3.9% vs avg_loss 6.5%, leverage-included). MTF 필터는 이 R:R을 변경하지 않으므로 (avg_win/loss 유사) WR만의 미미한 개선은 구조적으로 한계.

5. **Threshold 민감도 부재**: 50~55% 모든 accuracy threshold에서 결과가 거의 동일 (패턴 수 142→132). 이는 1h 패턴 방향 예측이 binary(있다/없다)에 가깝고, accuracy 차이가 필터 효과로 전이되지 않음을 의미.

### Phase 5-6 미진행 사유

- Phase 3/3X에서 WR 개선이 3pp 기준에 미달
- Strict 모드는 기준 달성(+5.5pp) 하지만 거래 95% 손실 → 비현실적
- Phase 4 참고 결과도 IS-only + MDD 86.9% → WF 통과 가능성 낮음
- Plan의 Go/No-Go gate 설계에 따라 Phase 3 STOP → 전체 중단

---

## 5. Artifacts

### 연구 스크립트

| 파일 | 역할 | LOC |
|------|------|-----|
| `scripts/analysis/mtf_direction_study.py` | Phase 1-2: 데이터 집계 + 1h 방향 예측력 | ~575 |
| `scripts/analysis/mtf_filter_backtest.py` | Phase 3-4: MTF 필터 백테스트 + Tight TP | ~780 |

### 결과 JSON

| 파일 | 내용 |
|------|------|
| `results/mtf_direction_study.json` | Phase 2: 15 strong 1h patterns, GO 판정 |
| `results/mtf_filter_backtest.json` | Phase 3/3X/4: MTF 필터 효과, STOP 판정 |

### 생성 데이터

| 파일 | 내용 |
|------|------|
| `data/btc_1h_720days.csv` | 720일 1h OHLCV + classification |
| `data/btc_15m_720days.csv` | 720일 15m OHLCV + classification |

---

## 6. Learnings & Future Reference

### 연구 방법론 교훈

1. **Go/No-Go gate 설계의 중요성**: Phase 2 GO가 Phase 3 성공을 보장하지 않음. "1h 패턴에 방향 예측력이 있다" ≠ "1h 방향 필터가 5m 진입을 개선한다". 두 가설은 독립적.

2. **파라미터 공간 확장의 가치**: 최초 15 strong 패턴(커버리지 5%)으로 결론 내리기 전, min_occ=30 + 완화된 threshold(50~55%)로 확장 탐색. 결론은 동일하지만, "충분히 탐색했다"는 확신을 확보.

3. **총 edge 관점**: 거래당 edge 개선(+0.255%/trade)이 거래 수 감소(-74%)에 의해 상쇄될 수 있다. 전략 평가 시 반드시 total edge (trades × edge/trade)를 확인해야 함.

4. **R:R 고정 구조의 함의**: R:R < 1.0 전략에서 WR +2.5pp는 구조적 한계가 있다. MTF 필터가 R:R 자체를 개선하지 않는 한 (avg_win/loss 변경) 필터 효과는 제한적.

### 재사용 가능한 자산

- `btc_1h_720days.csv`: 1h 분류 데이터 (다른 1h 기반 연구에 활용)
- `measure_directional_accuracy()`: 패턴 방향 예측력 측정 함수 (다른 TF에 재사용)
- `build_expanded_1h_patterns()`: 다단계 threshold sweep 함수
- `bt_signals_from_list()`: 범용 signal-list 백테스트 엔진

### 연구하지 않은 대안 방향 (향후 참고)

| 대안 | 설명 | 가능성 |
|------|------|--------|
| 15m 방향 필터 | 1h 대신 15m 패턴 사용 (더 빈번, 더 적시) | 낮음 (1h도 실패) |
| Volume-weighted bias | 패턴 방향 대신 volume profile 사용 | 미지수 |
| 방향 비대칭 활용 | LONG/SHORT 별 다른 필터 강도 | 낮음 (SHORT WR 이미 높음) |
| Adaptive TP by 1h | 방향 필터 대신 1h에 따라 TP 크기 조정 | 중간 (Tight TP Phase 4 일부 유망) |

---

## 7. Impact on Production

**코드 변경: 없음** (연구 전용, production 미수정)

**현재 전략 유지**: v1.33.0 (35 patterns, Compact TP/SL, Direction Cap 6, WR 68.1%)

**권장 다음 행동**:

1. **v1.34.0 배포 안정화** — holdout/staleness/MDD sizing 운영 모니터링
2. **90일 re-scan 대기** (2026-05 예정) — scan staleness 체크에 의한 자동 트리거
3. **Live OOS 데이터 축적** — 현재 22일 → 90일 목표로 순수 forward 성과 축적
4. **다음 연구 후보**: ATR scaling 최적화 (이미 production 적용, 파라미터 fine-tuning) 또는 새로운 패턴 발굴 (1h 데이터 활용)

---

*Report generated: 2026-02-24 | Feature: mtf_direction_filter | Outcome: STOP*
