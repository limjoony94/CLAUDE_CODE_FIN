# Plan: SL/Trail 파라미터 튜닝 (C1 Breakout v2.6)

> **Feature**: sl_trail_tuning
> **Date**: 2026-04-18
> **Phase**: Plan
> **Target**: `max_sl_atr`, `trail_K`, `max_hold_bars` 최적화
> **Status**: RESEARCH (production 적용은 WF/MC 통과 시에만)

---

## 1. Background

### 현재 파라미터 (v2.6 production)
| 파라미터 | 현재값 | 역할 |
|----------|--------|------|
| `max_sl_atr` | 3.3 | 프랙탈 SL 거리 상한 (×ATR) |
| `trail_K` | 2.5 | Trail TP 콜백 폭 (best − K×ATR) |
| `max_hold_bars` | 192 (48h) | 강제 청산 (현재 0회 발동) |
| `emergency_sl_pct` | 3.0% | Hard SL (범위 외) |

### Baseline 성과 (1027 trades, 333일)
| 지표 | 값 |
|------|-----|
| PnL (add 1x) | +170.5% |
| MDD | 5.4% |
| WR | 36.6% |
| R:R | 3.36 |
| Trail-TP exit | 85.5% |
| SL exit | 14.5% |
| Timeout exit | **0.0%** |
| Max 연속손실 | 13 |

### 기존 1D Grid (`results/extended_param_grid.json`) 시사점
| 파라미터 | 관찰 | 해석 |
|----------|------|------|
| `max_sl_atr` | 3.3→4.0 시 PnL 169.5→192.8 (**단조증가**) | 현재값이 과보수적일 가능성 |
| `trail_K` | 2.5에서 sharp peak (1.8→136, 2.5→169, 2.8→129) | 현재값이 최적에 위치, 취약 |
| `max_hold_bars` | grid 없음 | 검증 안됨 (exit 0회 = 무의미 or 안전장치) |

### 문제 의식
1. **1D grid는 상호작용 무시** — `max_sl_atr`↑ 시 `trail_K` 변화 가능
2. **IS peak ≠ OOS best** — overfit 위험
3. **PnL 최대화 ≠ Risk-adjusted 최적** — MDD/Calmar/Sortino 병행 필요
4. **Timeout 역할 불명** — 0회 발동은 "불필요"인지 "안전망"인지 미확인

---

## 2. Goal

3개 파라미터의 **상호작용을 고려한** 최적 조합을 탐색하되, **OOS 일반화 증거**를 확보한 경우에만 production 변경.

### 성공 기준 (GO 조건, 7개)
모두 충족해야 변경 진행:

1. **IS 개선**: PnL/MDD 비율이 baseline 대비 ≥ +10% (ratio_ok)
2. **WF 5/5 PASS**: expanding window OOS 전부 양수 (wf_pass)
3. **3-way split 양수**: train/val/test 모두 PnL > 0 (tw_pass)
4. **Test 하락 제한**: test PnL이 baseline test 대비 하락 ≤ 5%p (test_ok, tw_pass와 분리 가시화)
5. **MC Direction p < 0.01**: 여전히 DISC (mc_pass)
6. **Bootstrap 95% CI**: PnL 하한 > 0 (ci_pass)
7. **Param neighborhood**: 최적점 ±1 step 축방향 이웃(최대 6개) 중 ≥75% 양수 (nbr_pass)

한 항목이라도 실패 시 **baseline 유지** + 학습 내용 기록 후 종료.

> **Note**: Design §9의 `decide_verdict()`와 1:1 대응.

---

## 3. Hypotheses

| 가설 | 내용 | 예상 |
|------|------|------|
| **H1** | `max_sl_atr` ∈ [3.5, 4.0]이 3.3 대비 PnL/MDD 개선 | GO 가능 |
| **H2** | `trail_K` 2.5는 sharp peak이라 변경 시 악화 | STOP |
| **H3** | `max_hold_bars` 축소(96=24h)는 SL 치환량 증가로 WR 상승 vs PnL 감소 tradeoff | 중립 |
| **H4** | (`max_sl_atr`, `trail_K`) 상호작용 — SL↑ 시 Trail 최적이 이동 | 탐색 필요 |

---

## 4. Methodology

### 4.1 Grid 설계 (3D)
| 축 | 값 | 카드너리 |
|----|----|----|
| `max_sl_atr` | 2.8, 3.0, 3.3, 3.6, 4.0, 4.5 | 6 |
| `trail_K` | 2.0, 2.2, 2.5, 2.8, 3.0 | 5 |
| `max_hold_bars` | 96, 144, 192, 288 | 4 |

총 **120 combos**. 기존 스크립트(`scripts/analysis/c1_param_grid.py` 등) 재사용.

### 4.2 데이터
- **전체**: BTC/USDT 15m, 333일 (기존과 동일 데이터)
- **Split**: train 60% / val 20% / test 20% (시간 순차)
- **Fee**: 0.10% RT, additive 1x PnL

### 4.3 평가 지표 (per combo)
```
PnL (add 1x)
MDD
PnL/MDD (Calmar proxy)
Trades
WR
R:R
Max consecutive loss
Timeout exit 비율
```

### 4.4 Selection Protocol (selection-after-peek 방지)
1. **Train 데이터만**으로 top-10 combo 선정 (val/test 미사용)
2. Top-10에 대해 **val**에서 re-rank → top-3
3. Top-3 중 **test** 결과는 **사후 검증용**만 (재선정 금지)
4. 최종 선택은 test PnL이 아닌 **train+val 종합**으로 확정

### 4.5 Robustness 검증 (선택된 combo에 한정)
| 검증 | 방법 |
|------|------|
| WF 5-fold | expanding window, 기존 C1 WF 재사용 |
| MC Direction | sign randomization ≥999 sims |
| Bootstrap CI | stationary bootstrap 1000회 (block=20) |
| Parameter neighborhood | 최적점 ±1 grid step 주변 8개 이웃 positive rate |
| Top-N trade 제거 | 상위 10/20/50 trade 제거 후 PnL 확인 |

---

## 5. Implementation Plan

### Change 1: 연구 스크립트 `scripts/analysis/sl_trail_grid.py` 신규
- 기존 `c1_param_grid.py` 재사용 가능하면 확장, 아니면 유사 구조로 신규 작성
- 3D grid + train/val/test split + selection protocol 내장
- 출력: `results/sl_trail_grid_{timestamp}.json`

### Change 2: WF 검증 스크립트 재사용
- 기존 `scripts/analysis/c1_wf_validation.py` 혹은 `wf-validate` command 사용
- 선정된 top-3 파라미터에 대해서만 실행

### Change 3: production 적용 (GO 판정 시에만)
- `config/c1_breakout_config.yaml`의 3개 값만 수정
- 코드 변경 **없음** — 파라미터는 이미 config-driven
- 봇 재시작 + state.json 보존

### Change 4: 문서 갱신 (변경 시)
- `CLAUDE.md` Version History + 빠른 참조
- `claudedocs/c1_breakout_v2_design.md` 검증 수치

---

## 6. Non-Changes

1. `emergency_sl_pct` (3.0%) — 발동 0회, 이번 범위 제외
2. `channel_period`, `atr_period`, `body_min_ratio` — 1D grid에서 이미 안정
3. `min_bars_between`, `trail_activation_pct` — 구조적 파라미터
4. 엔트리 로직 — 변경 없음
5. Exchange SL/TP 배치 로직 — 변경 없음
6. **STOP 조건 시 변경 없음** — baseline 유지

---

## 7. Implementation Order

1. Grid 스크립트 작성 (재사용 최대화)
2. 120-combo 실행 → train top-10 선정
3. Val re-rank → top-3 확정
4. Top-3 WF 5-fold 검증
5. Top-3 MC + Bootstrap + Neighborhood 검증
6. GO/STOP 판정
7. GO 시: config 변경 + 문서 갱신 + 봇 재시작
8. STOP 시: 학습 기록 후 종료

---

## 8. Rollback

```yaml
# config/c1_breakout_config.yaml 원복
strategy:
  max_sl_atr: 3.3
  trail_K: 2.5
  max_hold_bars: 192
```

봇 재시작으로 즉시 반영. 코드 변경 없으므로 git revert 불필요.

---

## 9. Risks & Mitigations

| 리스크 | 완화 |
|--------|------|
| **Overfit to train** | train/val/test split + WF + neighborhood positive rate |
| **Selection-after-peek fallacy** | val 재선정 금지, test는 사후 검증만 |
| **Sharp peak 함정** | ±1 step 이웃 ≥6/8 positive 요구 |
| **Live ≠ Backtest** | GO 시에도 소규모 기간(≥30일) 라이브 모니터링 |
| **MDD 증가** | `max_sl_atr`↑ 시 per-trade max loss↑ 자동 계산 + 8개 기준 중 하나 |

---

## 10. Reference

- Grid 1D: `results/extended_param_grid.json`
- Variants: `results/c1_refined_variants.json` (BASELINE 기록)
- WF 기존: `results/c1_last_30days_backtest.json`, `c1_reverse_full_backtest.json`
- 연구 프로토콜: `claudedocs/STANDARD_RESEARCH_PROTOCOL.md`
- 교훈 메모리: `direction_switching_20260418.md` (selection-after-peek), `refined_decision_20260418.md` (variants 기각 사례)
