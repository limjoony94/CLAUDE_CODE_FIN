# SL/Trail 파라미터 튜닝 PDCA 완료 보고서

> **Feature**: sl_trail_tuning
> **Type**: Research-only (production 코드 변경 없음)
> **Duration**: 2026-04-18 ~ 2026-04-19
> **Status**: COMPLETED (STOP 판정)
> **Match Rate**: 94% (Design ↔ 구현)

---

## 1. 사업 요약

C1 Breakout v2.6의 **3가지 파라미터 최적화 연구**: `max_sl_atr`(현 3.3), `trail_K`(현 2.5), `max_hold_bars`(현 192). 3D grid 120 combos 실행 후 train/val/test 계층적 선정, WF/MC/Bootstrap/Neighborhood 종합 검증. 

**결과**: 최적 후보 `(4.5, 2.2, 144)` 선정됨. 6/7 GO 조건 통과, **PnL/MDD 비율 기준 1개 실패** → **baseline 유지 결정** (production 변경 없음).

**교훈**: 1D grid는 MDD 트레이드오프를 은폐. 절대 PnL ≥ 계약된 risk-adjusted 기준이 필요. 사전 선언된 protocol의 가치 검증됨.

---

## 2. PDCA 사이클 개요

### Plan (계획)
- **목표**: 3D grid로 상호작용 고려한 최적 조합 탐색
- **범위**: 6×5×4=120 combos, train/val/test split + 7개 GO 조건
- **기간**: 2026-04-18 계획 수립
- **담당자**: automated research script

**문서**: `docs/01-plan/features/sl_trail_tuning.plan.md`

### Design (설계)
- **아키텍처**: research-only, 기존 validation 인프라 재사용 + grid 신규 스크립트
- **Grid**: `max_sl_atr ∈ [2.8, 4.5]`, `trail_K ∈ [2.0, 3.0]`, `max_hold_bars ∈ [96, 288]`
- **Selection**: S1(train top-10) → S2(val top-3) → S3(test 검증) → S4-5(WF/MC/Bootstrap/Neighborhood)
- **GO 판정**: 7개 조건 모두 충족 필요

**문서**: `docs/02-design/features/sl_trail_tuning.design.md`

### Do (구현)
- **스크립트**: `scripts/analysis/sl_trail_grid.py` 신규 작성
- **실행**: 2026-04-18 22:11 시작 → 2026-04-18 22:11 완료 (2.8초)
- **결과 저장**: `results/sl_trail_grid_full_20260418_221157.json`

**인프라**: c1_refined_validation, c1_refined_bootstrap_mdd 재사용

### Check (분석)
- **Match Rate**: 94% (12 matched + 4 partial + 0 critical)
- **Baseline 회귀**: drift = 0.00 (완벽)
- **GO 조건 평가**: 6/7 통과 (ratio_ok 1개 실패)

**문서**: `docs/03-analysis/sl_trail_tuning.analysis.md`

### Act (결론)
- **판정**: STOP (baseline 유지)
- **근거**: PnL/MDD 비율이 기준(34.86) 미달(27.92)
- **후속**: 교훈 기록 후 종료, 별도 feature로 향후 검토

---

## 3. 연구 범위 및 방법론

### 3.1 그리드 설계

| 축 | 값 | 카드너리 | 의미 |
|-----|-----------|---------|--------|
| `max_sl_atr` | [2.8, 3.0, 3.3, 3.6, 4.0, 4.5] | 6 | 프랙탈 SL 거리 상한 (×ATR) |
| `trail_K` | [2.0, 2.2, 2.5, 2.8, 3.0] | 5 | Trail TP 콜백 폭 (best − K×ATR) |
| `max_hold_bars` | [96, 144, 192, 288] | 4 | 강제 청산 타임아웃 (현재 미발동) |
| **합계** | | **120** | |

### 3.2 데이터 및 기간

- **데이터**: BTC/USDT 15m, 333일 (2025-06-30 ~ 2026-04-03)
- **분할**: warmup(50봉) 제외 후 train 60% / val 20% / test 20%
- **수수료**: 0.10% RT (taker 0.05% × 2)
- **PnL 계산**: additive 1x (compound 왜곡 제거)

### 3.3 선정 프로토콜 (Selection-After-Peek 방지)

| 단계 | 대상 | 평가 데이터 | 필터 | 출력 |
|------|------|-----------|------|-------|
| S1 | 120 combos | train 60% | trades ≥ 300 | 상위 10 |
| S2 | top-10 | val 20% | trades ≥ 100 | 상위 3 |
| S3 | top-3 | test 20% | — | 사후 기록 |
| S4 | top-3 | full (WF) | expanding window | PASS/FAIL |
| S5 | top-3 | full (MC/BS/NBR) | 통계 검정 | flag |

**Ranking metric**: Primary = PnL/MDD, Tiebreaker = PnL

### 3.4 Robustness 검증 (top-3 대상)

| 검증 | 방법 | 기준 |
|------|------|------|
| **WF 5-fold** | expanding window | 5/5 OOS PnL > 0 |
| **MC Direction** | sign randomization 999 sims | p < 0.01 |
| **Bootstrap PnL CI** | stationary block 1000 sims | 95% 하한 > 0 |
| **Neighborhood** | ±1 grid step 축방향 이웃 | 75% 이상 양수 |
| **3-way Split** | train/val/test | 모두 > 0 + test ≥ 49.2% |

---

## 4. 결과 및 GO 조건 평가

### 4.1 베이스라인 기준점

| 지표 | 값 | 비고 |
|------|-----|------|
| Full PnL | +170.49% | 1027 trades, 333일 |
| Full MDD | 5.38% | |
| **PnL/MDD (ratio)** | **31.69** | risk-adjusted 기준 |
| Test PnL | +54.20% | 20% 구간 |
| WF 5/5 | PASS | OOS 모두 양수 |
| MC p-value | 0.000 | DISC (significant) |

### 4.2 선정 결과 (120 → 10 → 3)

**S1 Train top-10** (상위 3개):
1. `(3.6, 2.2, 96)`: PnL +91.73%, MDD 3.61, ratio 25.41
2. `(3.6, 2.2, 144)`: PnL +91.73%, MDD 3.61, ratio 25.41 (동일)
3. `(3.6, 2.2, 192)`: PnL +91.73%, MDD 3.61, ratio 25.41 (동일)

→ `max_sl_atr=3.6` 또는 `max_sl_atr=4.5` 근처 집중

**S2 Val Rerank top-3**:
1. `(4.5, 2.2, 144)`: val PnL +33.88%, ratio 11.56
2. `(4.5, 2.2, 192)`: val PnL +33.88%, ratio 11.56 (동일)
3. `(4.5, 2.2, 288)`: val PnL +33.88%, ratio 11.56 (동일)

→ val에서 `(4.5, 2.2)` 조합 선정, `max_hold_bars` 변종 (미발동)

**최종 후보**: `(4.5, 2.2, 144)` (및 동등 2개 variant)

### 4.3 최종 후보 vs Baseline 상세 비교

#### 절대 성과

| 지표 | Baseline | Candidate `(4.5,2.2,144)` | Δ | 평가 |
|------|----------|---------------------------|---|------|
| **Full PnL** | +170.49% | **+183.17%** | **+12.68pp** | 절대 수익 개선 |
| **Full MDD** | 5.38 | 6.56 | +1.18pp | MDD 악화 |
| **Ratio (PnL/MDD)** | 31.69 | 27.92 | **-11.9%** | risk-adjusted 악화 |
| Test PnL | +54.20% | +57.64% | +3.44pp | test 구간 개선 |
| Trades | 1027 | 1101 | +74 | 거래량 증가 |
| WR | 36.6% | 36.4% | -0.2pp | 승률 유지 |

#### 통계적 검증

| 검증 | 결과 | Pass |
|------|------|------|
| **WF 5-fold OOS** | [12.96, 26.09, 47.84, 37.02, 37.76]% | ✅ (5/5) |
| **3-way split** | train 91.65%, val 33.88%, test 57.64% | ✅ (all > 0) |
| **MC p-value** | 0.001 | ✅ (< 0.01) |
| **Bootstrap PnL 95% CI** | [+132.24, +242.81]% | ✅ (lower > 0) |
| **Neighborhood** | 5/5 이웃 양수 | ✅ (100%) |

### 4.4 GO 조건 7개 최종 평가

| # | 조건 | 기준값 | 실제값 | Pass |
|----|------|--------|--------|------|
| 1 | **ratio_ok** | PnL/MDD ≥ 34.86 (31.69×1.10) | 27.92 | ❌ **FAIL** |
| 2 | wf_pass | 5/5 OOS PnL > 0 | ✅ 5/5 | ✅ |
| 3 | tw_pass | train/val/test 모두 > 0 | ✅ all > 0 | ✅ |
| 4 | test_ok | test ≥ 49.20% (54.20−5.0) | 57.64% | ✅ |
| 5 | mc_pass | MC p < 0.01 | 0.001 | ✅ |
| 6 | ci_pass | Bootstrap PnL CI 하한 > 0 | 132.24% | ✅ |
| 7 | nbr_pass | Neighborhood ≥ 75% | 5/5 (100%) | ✅ |

**결론**: 6/7 통과, **ratio_ok 1개만 실패** → **STOP**

---

## 5. 부수적 발견

### 5.1 max_hold_bars는 Dead Parameter

top-3 모두 `max_hold_bars ∈ [96, 144, 192, 288]`에서 **완전히 동일 결과**:

- Full PnL: 183.17% (동일)
- WF/test 결과: 동일
- Timeout exit 비율: 0.0% (미발동)

**해석**: 192봉(48h) timeout이 현재 데이터에서 한 번도 발동되지 않음. timeout은 **구조적 안전장치일 뿐 활동 파라미터가 아님**.

### 5.2 trail_K와 max_sl_atr의 상관관계

기존 1D grid (`extended_param_grid.json`)에서:
- `trail_K=2.5` 시 sharp peak (2.5→169% PnL)
- 변경 시 악화 (1.8→136%, 2.8→129%)

3D grid에서:
- `(4.5, 2.2)` 조합이 `(3.3, 2.5)` 동등-이상
- `max_sl_atr` 증가 시 최적 `trail_K`가 감소 추세

**해석**: `max_sl_atr` 확장 후 `trail_K` 콜백을 더 타이트하게 해서 whipsaw 방지.

### 5.3 전략의 Robustness

6개 non-ratio 조건 통과 + neighborhood 5/5 양수:
- 선명한 overfit 신호 없음
- 근처 파라미터도 강건
- **Strategy 자체가 SL/Trail 변화에 대해 robust**

---

## 6. 판정 근거

### 왜 STOP인가

1. **Plan §2에서 사전 선언한 7개 GO 조건 중 1개 실패** — ratio_ok 부족
2. **절대 PnL ≠ risk-adjusted 최적**
   - PnL: +12.68pp 개선
   - MDD: +1.18pp 악화 (22% 증가)
   - Ratio: -11.9% 악화 (31.69→27.92, 기준은 34.86)
3. **Selection-After-Peek Fallacy 위험**
   - 사후 임계값 완화는 과거 교훈(메모리 `direction_switching_20260418.md`) 위반
   - Protocol 준수의 가치: objective 판정 유지
4. **깊은 의미**: 넓은 SL이 whipsaw 탈출만이 아닌 역추세 loss 증가도 초래 (MDD 비례 이상)

---

## 7. 코드 품질 및 인프라 검증

### 7.1 설계 준수율 (Match Rate)

| 항목 | 상태 | 세부 |
|------|------|------|
| **12 Matched** | ✅ | Grid 120 combos, 3-stage selection, filtering, ranking, WF/MC/Bootstrap/Neighborhood, verdict logic, baseline regression 0.00, smoke/full test |
| **4 Partial** | ⚠️ | 파일 위치 정리(archive), 키 naming 미세차이, JSON 구조 변형, 성능(예상 수분 vs 실제 2.8초) |
| **0 Critical** | ✅ | 블로킹 이슈 없음 |

**Match Rate: 94%** (12+4/17) → **Report 단계 통과**

### 7.2 성능 및 신뢰성

| 항목 | 결과 | 평가 |
|------|------|------|
| **Total runtime** | 2.8초 | 예상(수분) 대비 훨씬 빠름, precompute 캐싱 효과 |
| **Baseline regression** | drift = 0.00 | 완벽, 인프라 버그 없음 |
| **120-combo execution** | 완료 | trade count filter 적용, 최소 통계량 확보 |
| **Top-3 validation** | 완료 | WF/MC/Bootstrap/NBR 모두 실행 |

### 7.3 구현 검증

- ✅ Grid 설계 정확함 (6×5×4=120)
- ✅ Selection protocol 준수 (train/val/test 계층 준수)
- ✅ Ranking metric 정확함 (ratio, PnL)
- ✅ GO 조건 7개 모두 평가됨
- ✅ Output JSON 완전하고 구조적

---

## 8. 교훈 및 향후 적용

### 8.1 핵심 교훈

#### 1. 1D Grid의 함정: MDD 은폐

**발견**: `max_sl_atr` 단조증가가 1D에서 보였으나, 3D에서 MDD 비례 이상 증가.

**원인**: 각 축의 상호작용 무시. `max_sl_atr`↑는 whipsaw 탈출 이점도 있지만, 역추세 포지션의 손실도 증대.

**적용**: 단일 축 그리드는 초차 탐색용, 최종 판정은 risk-adjusted 메트릭(Sharpe, Calmar, Sortino) 필수.

#### 2. 절대 PnL ≠ Risk-Adjusted 최적

**발견**: `(4.5, 2.2)`가 +12.68pp PnL 개선도 ratio 기준 실패.

**의미**: 높은 절대 수익만으로는 부족, **위험 대비 수익률(risk-adjusted return)** 동시 충족 필요.

**적용**: GO/STOP 판정에 절대, 상대, risk-adjusted 지표 모두 포함 권장.

#### 8.3 Protocol 선언의 가치

**발견**: Plan §2에서 7개 조건 사전 선언 → objective 판정 가능.

**장점**:
- selection-after-peek fallacy 방지
- 감정적 판정 배제
- 학습 기록 신뢰성 향상

**적용**: 향후 연구는 모두 **사전 선언 protocol** 필수.

#### 8.4 max_hold_bars Dead Parameter 확정

**발견**: 96/144/192/288 모두 동일 (timeout 미발동).

**의미**: 48h 강제 청산이 BTC 5m 추세 특성상 불필요, 구조적 안전장치일 뿐.

**적용**: 추후 파라미터 튜닝 시 `max_hold_bars` 제외 가능.

### 8.2 향후 PDCA 후보 (별도 feature)

1. **MDD Tolerance 시나리오** (feature: `sl_trail_mdd_tolerance`)
   - operational balance/leverage 축소하면서 `(4.5, 2.2)` 시도
   - risk-adjusted 동등 유지 → 절대 ROI 증가 검증

2. **Regime-Conditional Parameters** (feature: `sl_trail_regime_adaptive`)
   - 저변동성 레짐에서만 `max_sl_atr=4.5` 허용
   - 고변동성 시 보수적 3.3 유지

3. **Emergency SL 축소** (feature: `emergency_sl_hardening`)
   - 3.0% → 2.5% 축소로 hard SL 상한 제어
   - MDD 상한 제어

---

## 9. 최종 결론

### 9.1 판정

**OUTCOME: STOP (baseline 유지)**

- Production 코드 변경 없음
- `config/c1_breakout_config.yaml` 그대로 유지
- 파라미터: `max_sl_atr=3.3`, `trail_K=2.5`, `max_hold_bars=192`

### 9.2 가치

**"실패"한 연구지만 높은 학습 가치**:

1. **전략 robustness 검증**: 6/7 GO 조건 통과로 C1 전략의 기초 견고함 확인
2. **인프라 검증**: 120 combos 3D grid 안정적 실행, baseline drift 0.00
3. **교훈 수립**: 1D grid 한계, risk-adjusted 필수성, protocol 가치
4. **향후 방향 제시**: MDD tolerance, regime adaptive, emergency SL 축소 등 3개 후보 제시

### 9.3 요약 메트릭

| 항목 | 결과 |
|------|------|
| Grid combos | 120 (6×5×4) |
| Runtime | 2.8초 |
| Selection | 120 → 10 → 3 |
| GO 조건 pass | 6/7 |
| Match rate | 94% |
| Verdict | **STOP** |
| Production 변경 | 0건 |
| Lessons learned | 4건 |
| Future candidates | 3개 |

---

## 10. 참조

### PDCA 문서

| Phase | 문서 | 경로 |
|-------|------|------|
| **Plan** | 목표 및 범위 정의 | `docs/01-plan/features/sl_trail_tuning.plan.md` |
| **Design** | 기술 설계 및 방법론 | `docs/02-design/features/sl_trail_tuning.design.md` |
| **Analysis** | Gap analysis 및 발견 | `docs/03-analysis/sl_trail_tuning.analysis.md` |
| **Report** | 완료 보고서 (이 파일) | `docs/04-report/sl_trail_tuning.report.md` |

### 데이터 및 코드

| 항목 | 경로 |
|------|------|
| 결과 JSON | `results/sl_trail_grid_full_20260418_221157.json` |
| 구현 (archived) | `archive/cleanup_20260418/analysis/sl_trail_grid.py` |
| 기존 인프라 | `scripts/analysis/c1_refined_validation.py` |
| 기존 인프라 | `scripts/analysis/c1_refined_bootstrap_mdd.py` |

### 메모리 및 교훈

| 항목 | 경로 | 주제 |
|------|------|------|
| Selection 교훈 | `MEMORY.md: direction_switching_20260418.md` | selection-after-peek fallacy |
| Variant 기각 | `MEMORY.md: refined_decision_20260418.md` | variants 시도 사례 |
| Protocol | `claudedocs/STANDARD_RESEARCH_PROTOCOL.md` | 표준 연구 프로토콜 |

---

## 11. 승인 및 이력

| 단계 | 일시 | 상태 | 비고 |
|------|------|------|------|
| Plan | 2026-04-18 | ✅ | 7개 GO 조건 사전 선언 |
| Design | 2026-04-18 | ✅ | Grid 120, selection 3-stage, verdict logic |
| Do | 2026-04-18 22:11 | ✅ | 2.8초 완료, regression 0.00 |
| Check | 2026-04-18 | ✅ | Match 94%, 모든 검증 완료 |
| Act | 2026-04-18 | ✅ | **STOP 판정, baseline 유지** |

**보고서 작성**: 2026-04-19
**Project**: CLAUDE_CODE_FIN / bingx_rl_trading_bot
**Strategy**: C1 Breakout v2.6

---

*이 보고서는 사전 선언 protocol에 따라 objective하게 작성되었으며, 향후 PDCA 개선 시 본 교훈과 후보들을 참고하기를 권장합니다.*
