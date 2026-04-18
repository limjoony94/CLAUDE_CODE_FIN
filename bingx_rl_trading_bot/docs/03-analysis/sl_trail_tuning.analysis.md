# Analysis: SL/Trail 파라미터 튜닝

> **Feature**: sl_trail_tuning
> **Date**: 2026-04-18
> **Phase**: Check
> **Match Rate**: **94%** (Design ↔ 구현)
> **Research Outcome**: **STOP** — baseline 유지

---

## 1. Executive Summary

C1 Breakout v2.6의 `max_sl_atr`, `trail_K`, `max_hold_bars` 3 파라미터 최적화 연구. 120 combos 3D grid + train/val/test 선정 + WF/MC/Bootstrap/Neighborhood 검증.

**결과**: 7개 GO 조건 중 6개 통과, **ratio_ok (PnL/MDD ≥ 1.10×baseline)** 1개만 실패 → **STOP**. production 파라미터 변경 없음.

---

## 2. Gap Analysis (Design vs 구현)

### Match Rate: 94% (17/17 항목 중 12 matched + 4 partial + 0 critical)

#### ✅ Matched (12)
- GRID 6×5×4=120 combos, 재사용 인프라 import, Selection 3-stage, trade count filters (300/100/100), ranking metric(ratio,PnL), WF/MC/Bootstrap PnL+MDD/Neighborhood 구현, decide_verdict 7 flags, output schema, baseline regression drift=0.00, smoke/full modes, STOP verdict 작동.

#### ⚠ Partial Gap (4, 영향도 low)
1. **파일 위치**: 실행 후 `archive/cleanup_20260418/analysis/`로 자동 정리됨. 결과 JSON은 `results/`에 보존.
2. **decide_verdict 키 naming**: Design `reasons` (복수) vs 구현 `reason` (단수). 기능 영향 없음.
3. **Output JSON 구조**: Design `verdict.go_conditions` vs 구현 `verdict.last_flags` + per-combo `verdict_flags`. 동일 정보, 키 이름만 다름.
4. **Performance drift**: 예상 "수 분" vs 실제 **2.8초**. precompute 캐싱 효과로 훨씬 빠름 (긍정적 편차).

#### ❌ Critical Gap: 0건

**권고**: ≥90% Match, Blocker 없음 → **Report 단계로 직행**.

---

## 3. Research Findings

### 3.1 베이스라인 회귀 확인
```
baseline combo (3.3, 2.5, 192) full PnL = +170.49%
expected from c1_refined_variants.json BASELINE = +170.49%
drift = 0.000  (Design §11 "±0.5% 이내" 완벽 만족)
```
→ 신규 grid 스크립트의 backtest 로직이 기존 인프라와 비트 단위로 일치. 오버레이 버그 없음.

### 3.2 Selection 결과 (120 → 3)

| 단계 | 필터 | 남은 combo | 최상위 |
|------|------|-----------|--------|
| S1 Train | trades ≥ 300 | 10/120 | `(3.6, 2.2, 96)` PnL+91.7, ratio 25.41 |
| S2 Val | trades ≥ 100 | 3/10 | `(4.5, 2.2, 144)` val+33.9, ratio 11.56 |
| S3 Test | 사후검증 | 3/3 (재선정X) | +57.6% (baseline +54.2% 대비 +3.4pp) |

Top-3 모두 `max_sl_atr=4.5, trail_K=2.2`의 `max_hold_bars` 변종(144/192/288). 이들의 full 기간 성과는 완전히 동일 — **timeout 미발동**.

### 3.3 최종 후보 vs Baseline 비교 (`max_sl_atr=4.5, trail_K=2.2, max_hold_bars=144`)

| 지표 | Baseline | Candidate | Δ | 평가 |
|------|----------|-----------|----|------|
| Full PnL | +170.49% | **+183.17%** | +12.68pp | 절대 수익 개선 |
| Full MDD | 5.38 | 6.56 | +1.18pp | MDD 증가 |
| PnL/MDD (ratio) | 31.69 | 27.92 | **-11.9%** | risk-adjusted 악화 |
| Test PnL | +54.20% | +57.64% | +3.44pp | test 구간 개선 |
| WF 5-fold | 5/5 | **5/5** (12.96/26.09/47.84/37.02/37.76) | 동등 | 모두 양수 |
| 3-way split | train+val+test>0 | 91.65/33.88/57.64 | 모두 양수 | PASS |
| MC p-value | 0.000 | 0.001 | 동등 | DISC 유지 |
| Bootstrap PnL 95% CI | - | [+132.2, +242.8] | 하한>0 | 견고 |
| Neighborhood | - | 5/5 positive | 100% | 평탄 |

### 3.4 GO Condition 7개 통과 여부

| # | Flag | Threshold | Actual | Pass |
|---|------|-----------|--------|------|
| 1 | `ratio_ok` | PnL/MDD ≥ 34.86 (31.69×1.10) | 27.92 | **❌** |
| 2 | `wf_pass` | 5/5 folds OOS > 0 | 5/5 | ✅ |
| 3 | `tw_pass` | train/val/test 모두 > 0 | 91.65/33.88/57.64 | ✅ |
| 4 | `test_ok` | test ≥ 54.20 − 5.0 = 49.20 | 57.64 | ✅ |
| 5 | `mc_pass` | MC p < 0.01 | 0.001 | ✅ |
| 6 | `ci_pass` | Bootstrap PnL CI 하한 > 0 | +132.2 | ✅ |
| 7 | `nbr_pass` | Neighborhood ≥ 75% positive | 5/5 (100%) | ✅ |

**6/7 통과**. ratio_ok 1개 실패로 **STOP**.

---

## 4. Interpretation

### 4.1 왜 STOP인가
`(4.5, 2.2)` 변종은 **더 넓은 SL을 허용**하여 whipsaw 탈출 후 추세 포착 기회를 증가시킨다(+12.7pp PnL). 그러나 넓어진 SL이 역추세 포지션에서는 손실을 키우는 면이 있어 **MDD가 비례 이상으로 증가**(+22%). Plan §2에서 사전 확정한 risk-adjusted 기준(+10% PnL/MDD 향상)을 넘지 못한다.

Post-hoc으로 임계값을 완화하는 것은 selection-after-peek fallacy(메모리 `direction_switching_20260418.md` 교훈) 재현. **규칙대로 STOP** 채택.

### 4.2 부수적 발견
1. **`max_hold_bars`는 dead parameter**: 96/144/192/288이 전부 동일 결과 → timeout이 한 번도 발동되지 않음. 192 유지가 안전. Plan H3 가설 확인.
2. **`trail_K`는 sharp peak가 아님**: 1D grid에서는 2.5에서 날카로운 봉우리였으나, 3D 상호작용 고려 시 `(4.5, 2.2)`가 `(3.3, 2.5)`와 동등-이상 — `max_sl_atr`이 넓어지면 `trail_K`도 더 타이트해지는 상관관계(Plan H4 확인).
3. **strategy robustness**: 6개 non-ratio GO 조건 통과는 C1 전략이 SL/trail 변화에 대해 전반적으로 robust하다는 증거. 어느 축으로도 선명한 overfit 신호 없음.

### 4.3 교훈
- **1D grid의 함정**: `max_sl_atr` 단조증가형은 MDD 증가를 은폐. 2D+risk 지표 필수.
- **절대 PnL ≠ risk-adjusted 최적**: baseline을 이길 수 있어도 risk-adjusted로 져야 기각.
- **Protocol 선언의 가치**: Plan §2에서 7개 조건을 사전 확정했기에 "아쉽지만 STOP" 판단이 가능했음.

---

## 5. Recommended Action

### 즉시
1. **production 변경 없음** — `config/c1_breakout_config.yaml` 그대로.
2. **Report 작성** (`/pdca report sl_trail_tuning`).
3. **MEMORY.md에 교훈 기록**: `sl_trail_tuning_20260418.md`로 "STOP ratio, max_hold_bars dead parameter, 1D grid 한계".

### 향후 (별도 PDCA)
- MDD 허용 시나리오: 운영 balance/leverage 축소로 실질 리스크를 유지한 채 `(4.5, 2.2)` 적용 ROI 재평가 (별개 feature).
- Regime-conditional 파라미터: 저변동성 레짐에서만 넓은 SL 허용하는 조건부 튜닝.
- Emergency SL 축소: 3.0% → 2.5%로 축소하여 MDD 상한 제어 (별개 feature).

---

## 6. Reference

- Plan: `docs/01-plan/features/sl_trail_tuning.plan.md`
- Design: `docs/02-design/features/sl_trail_tuning.design.md`
- 구현 (archived): `archive/cleanup_20260418/analysis/sl_trail_grid.py`
- 결과 JSON: `results/sl_trail_grid_full_20260418_221157.json`
- 사용 인프라: `scripts/analysis/c1_refined_validation.py`, `scripts/analysis/c1_refined_bootstrap_mdd.py`
