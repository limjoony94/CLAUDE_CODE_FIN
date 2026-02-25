# Plan: Multi-Timeframe Direction Filter + Tight TP

> Feature: `mtf_direction_filter`
> Date: 2026-02-24
> Status: Draft
> Type: Research → Production (conditional)

---

## 1. Problem Statement

현재 Pattern 5m 봇은 **5m 캔들 패턴만으로 방향 결정 + 진입**합니다.

**한계점**:
- 5m 노이즈에 의한 역방향 진입 (1h 추세와 반대)
- Compact TP 0.5~2.0%에서도 median 체결 16h — 더 빠른 회전 가능성
- Direction Cap 6으로 방향 집중 제한하지만, **큰 추세를 활용하지 못함**
- Live WR 61.8% (backtest 68.1%) — 방향 필터로 개선 여지

**핵심 아이디어**:
> 1h 패턴이 방향을 결정하고, 5m 패턴은 그 방향의 진입 타이밍만 잡는다.
> 추세 안에서의 스캘프이므로 TP를 0.5~1.0%로 타이트하게.

---

## 2. Hypothesis

### H1: 1h 방향 예측력
> 1h 3-candle 패턴은 다음 N시간의 가격 방향을 50% baseline 대비 유의하게 예측한다.

### H2: MTF 필터 효과
> 1h bias 방향과 일치하는 5m 진입은, 전체 5m 진입 대비 높은 WR + edge를 가진다.

### H3: Tight TP 최적화
> MTF 필터 적용 시, TP 0.5~1.0%에서도 수수료 차감 후 양의 edge가 존재한다.

---

## 3. Research Phases

### Phase 1: Data Preparation (데이터 집계)

**목적**: 5m 캔들을 15m/1h로 집계 + Ground Truth 분류 적용

**Input**: `data/btc_5m_720days_binance.csv` (207,361 rows, 720d)

**작업**:
1. 5m → 15m 집계 (3봉 = 1봉): `open:first, high:max, low:min, close:last, vol:sum`
2. 5m → 1h 집계 (12봉 = 1봉): 동일 방식
3. 각 TF에 `classify_candle()` 적용 → `type_code`, `pattern_3` 생성
4. `avg_body_20` 재계산 (각 TF의 20-period rolling)
5. 5m 데이터와 timestamp 정렬 (5m 각 봉에 대응하는 15m/1h 패턴 매핑)

**출력**:
- `data/btc_15m_720days.csv` (~69,120 rows)
- `data/btc_1h_720days.csv` (~17,280 rows)
- `data/btc_5m_720days_mtf.csv` (5m + `pattern_3_15m` + `pattern_3_1h` 컬럼 추가)

**검증**: 집계 후 OHLCV 정합성 — `high_1h == max(high_5m[12봉])` 등

---

### Phase 2: Higher TF Pattern Directional Study (1h 방향 예측력)

**목적**: 1h 3-candle 패턴이 미래 방향을 얼마나 예측하는지 측정

**방법**:
```
For each 1h pattern P:
  1. P 출현 후 다음 1h/3h/6h/12h/24h의 close-to-close 방향 기록
  2. directional_accuracy = (예측 방향 적중 수) / (총 출현 수)
  3. avg_move_pct = 평균 이동폭 (방향별)
```

**방향 정의 (1h 패턴)**:
- **Bullish 패턴**: 마지막 캔들이 BU/MU/H 또는 패턴이 반전 신호 (BD-BD-BU 등)
- **Bearish 패턴**: 마지막 캔들이 BD/MD/IH 또는 패턴이 반전 신호
- **중립**: ST, D 계열 마지막 캔들 → 필터 제외

**평가 기준**:
| 지표 | 기준 |
|------|------|
| Directional accuracy | > 55% (baseline 50%) |
| 통계적 유의성 | Binomial test p < 0.01 |
| 출현 빈도 | >= 50 occurrences (720d) |
| 예측 지속 시간 | 최소 3h 이상 유효 |

**Time horizons**: 1h, 3h, 6h, 12h, 24h (어느 기간에서 예측력이 최적인지)

**산출물**:
- 1h 패턴별 방향 예측 정확도 테이블
- 최적 예측 horizon 결정
- "Strong directional" 패턴 목록 (accuracy > 55%, n >= 50)

---

### Phase 3: MTF Filter Effect (5m 진입 필터링)

**목적**: 1h bias 일치 5m 진입 vs 전체 5m 진입 비교

**설계**:
```
Baseline: 현재 35패턴 5m 진입 (direction 무관)
Treatment: 1h Strong 패턴 방향 == 5m 패턴 방향일 때만 진입
```

**비교 지표**:
| 지표 | Baseline (현재) | MTF 필터 적용 |
|------|----------------|--------------|
| WR | 68.1% | ? |
| Edge (per-trade) | 0.221% | ? |
| Trades/day | 4.97 | ? (감소 예상) |
| PnL (270d) | +297.1% | ? |
| MDD | 10.3% | ? |

**Critical question**: 거래 수 감소가 WR 상승으로 보상되는가?

**변형 실험**:
| 변형 | 설명 |
|------|------|
| A | 1h 패턴 방향 필터만 (현재 TP/SL) |
| B | 1h 패턴 방향 필터 + 15m 확인 (이중 필터) |
| C | 1h 패턴 방향 필터 + Tight TP (0.5~1.0%) |
| D | 1h 패턴 방향 필터 + Tight TP + 현재 TP/SL 비교 |

**1h bias 적용 방식**:
- **현재 1h 캔들 패턴** (최근 완성된 3-candle)
- bias 유효 기간: 1h 패턴 완성 후 다음 1h 봉 시작까지 (최대 1h)
- 중립 패턴 시: 양방향 진입 허용 (기존과 동일)

---

### Phase 4: Tight TP Optimization (타이트 TP 최적화)

**목적**: MTF 필터 적용 시 최적 TP/SL 도출

**Conditioned on**: Phase 3에서 MTF 필터가 WR을 유의하게 개선한 경우만 진행

**TP Grid** (MTF 필터 + 추세 방향 진입):
```
TP: [0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.00]  (%)
SL: [0.50, 0.75, 1.00, 1.25, 1.50, 2.00]                (%)
```

**수수료 제약**: TP >= 0.40% 필수 (0.30% 수수료 + 0.06% 슬리피지 = 0.36%)

**최적화 기준**: WR Excess 극대화 (Phase 4 of v1.31.1 동일 방법론)

**산출물**:
- MTF 필터 + Tight TP 최적 조합
- 패턴별 또는 Universal TP/SL (거래 수에 따라 결정)
- Expected trades/day, WR, edge

---

### Phase 5: Portfolio Construction + WF Validation

**목적**: 최종 전략 포트폴리오 구성 + 통계 검증

**Portfolio options** (Phase 3-4 결과에 따라):
| 옵션 | 구성 |
|------|------|
| A | MTF 필터 + 현재 35패턴 + Tight TP |
| B | MTF 필터 + 새 패턴 발굴 (1h 방향별) + Tight TP |
| C | Hybrid: 강한 1h 추세 시 Tight TP, 약한 시 기존 TP |

**WF 검증**:
- 720d 데이터, 3-fold Expanding Window
- 기준: 3/3 PASS (OOS WR > baseline, OOS PnL > 0)
- MC test: 3 seeds, p < 0.01

**비교 대상**: 현재 v1.33.0 (35pat Compact, no MTF filter)

---

### Phase 6 (Conditional): Production Implementation

**Conditioned on**: Phase 5 WF 3/3 PASS + MTF가 현재 대비 개선

**구현 범위**:

| 변경 | 파일 | 설명 |
|------|------|------|
| 1h 캔들 수집 | `exchange.py` | `fetch_ohlcv(timeframe='1h')` 추가 |
| 1h 분류 | `indicators.py` | 기존 `classify_candle()` 재사용 |
| 1h 패턴 추적 | `signals.py` | `current_1h_pattern`, `1h_bias` 상태 |
| 진입 필터 | `signals.py` | `check_entry_signal()` 내 bias 일치 조건 |
| TP/SL 조정 | `position_open.py` | MTF 기반 Tight TP/SL |
| Config | `config.yaml` | `mtf_filter` 섹션 |
| Scanner | `pattern_scanner.py` | MTF-aware 스캔 모드 |

---

## 4. Data Requirements

| 데이터 | 소스 | 용량 |
|--------|------|------|
| 5m 720d | `btc_5m_720days_binance.csv` | 207,361 rows |
| 5m 270d (Ground Truth) | `btc_5m_270days_reclassified.csv` | 77,761 rows |
| 15m (생성) | 5m 집계 | ~69,120 rows |
| 1h (생성) | 5m 집계 | ~17,280 rows |

**Primary dataset**: 720d (WF 검증용 충분한 길이)
**Ground Truth**: 270d (5m 분류 검증 기준)

---

## 5. Success Criteria

| 지표 | 최소 기준 | 목표 |
|------|----------|------|
| 1h 방향 예측 accuracy | > 55% | > 60% |
| MTF 필터 WR 개선 | > +3pp vs baseline | > +5pp |
| Tight TP edge (after fees) | > 0.10%/trade | > 0.20%/trade |
| WF OOS | 3/3 PASS | 3/3 PASS |
| PnL vs current v1.33.0 | >= 동등 | > +20% |
| Trades/day | >= 3.0 | >= 5.0 |

---

## 6. Risks & Mitigations

| 리스크 | 확률 | 완화 |
|--------|------|------|
| 1h 패턴 방향 예측력 없음 (<53%) | 중간 | Phase 2에서 조기 중단, 15m 대안 테스트 |
| MTF 필터로 거래 수 과다 감소 | 높음 | 중립 패턴 시 양방향 허용, 약한 bias도 적용 |
| Tight TP가 수수료에 잠식 | 중간 | TP >= 0.40% 하한, R:R 최적화 |
| 5m/1h 패턴 상관관계 없음 | 낮음 | 동일 분류 체계, 물리적으로 5m이 1h 구성 |
| Overfit (다중 TF 파라미터 폭발) | 높음 | 단순 방향 필터만 (파라미터 최소화), WF 필수 |

---

## 7. Phase Dependencies & Go/No-Go Gates

```
Phase 1 (Data) ──→ Phase 2 (1h Direction Study)
                      │
                      ├── accuracy > 55%? ──YES──→ Phase 3 (MTF Filter)
                      │                              │
                      └── NO → 15m 대안 or STOP       ├── WR improvement > 3pp?
                                                     │     │
                                                     │    YES → Phase 4 (Tight TP)
                                                     │              │
                                                     │             Phase 5 (WF)
                                                     │              │
                                                     │            3/3 PASS? → Phase 6 (Prod)
                                                     │
                                                     └── NO → STOP (current strategy 유지)
```

---

## 8. Research Script Structure

```
scripts/analysis/mtf_direction_study.py
├── Phase 1: aggregate_to_higher_tf()
├── Phase 2: measure_directional_accuracy()
├── Phase 3: backtest_with_mtf_filter()
├── Phase 4: optimize_tight_tp()
├── Phase 5: walk_forward_mtf()
└── Output: results/mtf_direction_study.json
```

**Standard Research Protocol 준수**:
- Production `classify_candle` import
- LEVERAGE=3 PnL
- FEE_PCT * LEVERAGE = 0.30%
- Entry at next bar open
- Expanding Window WF
- MC test 3 seeds

---

## 9. Estimated Output

### Best Case (모든 가설 성립)
| 지표 | 현재 v1.33.0 | MTF Tight TP |
|------|-------------|-------------|
| WR | 68.1% | ~75% |
| TP range | 0.5~2.0% | 0.5~1.0% |
| Trades/day | 4.97 | ~6-8 |
| Hold time | 16h median | ~4-6h |
| PnL/MDD | 28.84 | ~35+ |

### Worst Case (가설 기각)
> 1h 패턴 방향 예측력 없음 → Phase 2 STOP → 현재 v1.33.0 유지 (리스크 없음)

---

## 10. Timeline

| Phase | 예상 소요 |
|-------|----------|
| Phase 1: Data | 연구 스크립트 ~200 LOC |
| Phase 2: 1h Study | 핵심 — Go/No-Go 결정 |
| Phase 3: MTF Filter | Phase 2 PASS 시 |
| Phase 4: Tight TP | Phase 3 개선 확인 시 |
| Phase 5: WF | 최종 검증 |
| Phase 6: Production | WF PASS 시 |

---

*Plan created: 2026-02-24 | Feature: mtf_direction_filter*
