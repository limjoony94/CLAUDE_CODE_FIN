# Plan: Progressive Trail (수익 확보 후 Aggressive Tighten)

> **Feature**: progressive_trail
> **Date**: 2026-04-21
> **Phase**: Plan
> **Trigger**: 4-path F6 dead-end 후 exit 재설계에서 session 최초 F6+WF+Sharpe 동시 해결 후보 발견
> **Target**: thr=0.8~1.0%, tkT=0.5 family를 full validation하여 cand_C+trend의 next-generation 후보 확정

---

## 1. Background

### 발견 경로
1. 4-path exploration (MR/Scalping/Relaxed F6) 전부 daily 음수 → 전략 교체로 F6 해결 불가 확정
2. Exit 재설계 시도 — **profit-based progressive trail** 탐색
3. Extended grid (113 combos: fine/tk_base/3-tier/time/ATR) 중 **thr 0.8~1.0 × tkT 0.5** family가 단독 지배

### 핵심 발견
**Baseline (tk=2.5 고정)** vs **Progressive (thr=0.9, tkT=0.5)**:

| 지표 | Baseline | Progressive | Δ |
|------|----------|-------------|---|
| PnL | +94.00 | **+122.49** | **+30%** |
| Daily | +0.282% | **+0.368%** | +30% |
| MDD | 7.91 | **4.97** | **-37%** |
| ex_top5 (F6) | **-22.01 ❌** | **+40.23 ✅** | 구조적 해결 |
| pos% (bootstrap) | 59.9% | **68.0%** | +8.1pp |
| Sharpe | 0.319 | **0.533** | +67% |
| WF 5-fold | 5/5 | **5/5** | stability 유지 |

### 검증 완료 사항 (Extended study)
- ✅ Fine grid sweet spot 확인 (0.8~1.0% × 0.5)
- ✅ tk_base 변화 불필요 (2.5 유지 최적)
- ✅ 3-tier/time-based/ATR-unit 전부 열세
- ✅ WF 5/5 (top 8 전부 fold 5/5)

### 검증 필요 사항 (이 PDCA)
- ❓ Strict Expanding WF (partition 외)
- ❓ OOS Split 5 boundaries
- ❓ 3-way split (train/val/test)
- ❓ Bootstrap 3-day CI
- ❓ Slippage 3 scenarios (low/med/high)
- ❓ Parameter consistency (2-half)
- ❓ MC direction test
- ❓ Neighborhood (25 grid) overfit check

---

## 2. Hypotheses

| H | 내용 |
|---|------|
| H1 | Progressive trail(thr=0.9, tkT=0.5)은 전체 기간 PnL/MDD/Sharpe/F6 모두 baseline 초과 |
| H2 | Strict Expanding WF 5/5 PASS |
| H3 | 3-way split (train/val/test) 전구간 양수 |
| H4 | Slippage 3 scenario(low/med/high) 모두 baseline 대비 우위 |
| H5 | Bootstrap 3-day: pos>=55%, p5>=-3.5, sharpe>0.3 |
| H6 | Neighborhood 25 combos 중 GO candidate 8+ (sharp peak 배제) |
| H7 | Structural Δ (progressive vs baseline) 2-half 모두 유지 |
| H8 | MC direction p < 0.01 |

---

## 3. Method

### 3.1 Fixed Base Strategy
```yaml
max_sl_atr: 4.0
trail_K_initial: 2.5          # pre-threshold
trail_K_post: 0.5             # post-threshold
max_hold_bars: 192
body_min_ratio: 0.60
trend_filter: abs(rolling_trend_192bars) > 1.0%
progress_threshold_pct: 0.9   # best_pnl 기준
```

### 3.2 Test Battery

| # | Test | Script/Engine |
|---|------|---------------|
| 1 | Full period clean/slip3 (low/med/high) | `progressive_trail_full_validation.py` |
| 2 | Strict Expanding WF 5-fold | 기존 `c1_refined_validation.py` 적용 |
| 3 | OOS Split 5 boundaries | Custom (60/20/20, 50/25/25, 70/15/15, 55/15/30, 60/15/25) |
| 4 | 3-way split (train/val/test) | 60/20/20 기본 |
| 5 | Bootstrap 3-day × 1000 | `stability_gate_validation.py` 패턴 |
| 6 | Neighborhood 5×5 (thr±2, tkT±2) | 25 combos |
| 7 | Parameter consistency 2-half | Spearman ρ 검사 |
| 8 | MC direction test 999 sims | Sign randomization |
| 9 | Fold 2 regime re-check | fold2_regime_analysis 재실행 |

### 3.3 Slippage Sensitivity
`c1_intrabar_parity.py`의 SLIPPAGE (low 0.5×, med 1.0×, high 2.0×) 3 scenario 비교.
각각에서 progressive vs baseline delta 측정.

### 3.4 Neighborhood Grid
```python
# Sharp peak 검사
thr_grid = [0.7, 0.8, 0.9, 1.0, 1.1]
tkT_grid = [0.3, 0.4, 0.5, 0.6, 0.7]
# 5×5 = 25 combos
```
GO combo 비율 ≥ 8/25 (32%) → non-sharp-peak.

### 3.5 Parameter Consistency
1st half (bars 26..15973) vs 2nd half (bars 15973..31945) 독립 BT.
각 half에서 progressive vs baseline delta 측정.
"Structural Δ는 유지되는가" 검증 (deep_validation_20260420 패턴).

---

## 4. Success Criteria (10-flag GO)

### Core (9 Required Hard)
1. **full_clean_pnl_gain** — 전체 기간 clean PnL: progressive ≥ baseline + 10pp
2. **full_slip_med_pnl_gain** — slip MED PnL: progressive ≥ baseline + 10pp
3. **strict_wf_5of5** — Strict Expanding WF 5-fold 전부 양수
4. **oos_split_5of5** — OOS Split 5 boundaries 전부 진행(train/val/test 양수)
5. **3way_all_positive** — 3-way train/val/test 모두 양수
6. **bootstrap_core_3of3** — mean>0, pos>=55%, p5>=-3.5
7. **bootstrap_relative** — progressive per-window PnL > baseline 확률 ≥ 55%
8. **neighborhood_8plus** — 25 combos 중 GO ≥ 8 (non-sharp-peak)
9. **structural_delta_both_halves** — 2-half 모두 delta ≥ +5pp

### Warnings (Log only)
10. **mc_direction_p<0.01** — MC direction p < 0.01
11. **f6_pass** — ex_top5 > 0 (이번 전략의 독특한 기준)
12. **mdd_improved** — MDD ≤ baseline - 1.0pp
13. **sharpe_improved** — Sharpe ≥ 0.45

### Deploy 조건
9/9 Core PASS → production config 변경 후보 (enabled=false로 deploy)
Warnings 4/4 → 30일 LIVE 관찰 후 enabled=true 검토

---

## 5. Risks & Mitigation

| Risk | Mitigation |
|------|-----------|
| tkT=0.5 whipsaw 증가 (매우 tight) | bootstrap 3-day pos% + p5 체크, live 30일 후 재평가 |
| "Profit locked too early" — trending day 수익 축소 | Fold별 trending 강도 분석 (fold 3/5에서 여전히 우세인지) |
| Parameter overfit (fine grid sweet spot) | Neighborhood 25 combo 검사로 sharp peak 식별 |
| LIVE 구현 복잡도 (condition branch) | signals.py `check_exit` conditional 추가 — 단순 1-branch |
| Backtest-Live parity 재검증 필요 | BUG#61 baton-touch 로직에 tkT=0.5 변동 대응 확인 |
| Progressive trail의 structural change로 기존 validation artifact 무효 | 전체 test battery 재실행 |

---

## 6. Non-Goals

- tk_base 변경 (Phase 2에서 2.5 최적 확인됨)
- 3-tier/time/ATR variant (전부 열세 확인됨)
- trend filter parameter 변경 (이전 PDCA에서 1.0/192 확정)
- body_filter/SL/hold 변경
- regime filter 추가 (trend + progressive 단일 layer만 검증)

---

## 7. Deliverables

1. **Script**: `scripts/analysis/progressive_trail_full_validation.py` (9 tests)
2. **Design**: `docs/02-design/features/progressive_trail.design.md`
3. **Production code**:
   - `scripts/production/c1_breakout/signals.py` — `check_exit` conditional trail_K
   - `config/c1_breakout_config.yaml` — `progressive_trail` section (enabled=false)
4. **Tests**: `scripts/tests/test_progressive_trail.py` (≥6 cases)
5. **Memory**: `progressive_trail_20260421.md`
6. **Report**: `docs/04-report/features/progressive_trail.report.md`

---

## 8. Reference

- Extended study: `scripts/analysis/progressive_trail_extended.py`
- Baseline study: `scripts/analysis/progressive_trail_study.py`
- Stability gate: `scripts/analysis/stability_gate_validation.py`
- Bootstrap protocol: `memory/research_protocol_3day_bootstrap.md`
- Parameter consistency: `memory/deep_validation_20260420.md`
- trend filter base: `docs/04-report/features/regime_filter_trend.report.md` (if exists) / `memory/regime_filter_trend_20260419.md`
- Overfit guards: `memory/research_protocol_overfit_guards.md`
- F6 relaxation parallel plan: `docs/01-plan/features/f6_relaxation.plan.md`

---

## 9. Timeline

- Day 1: Script scaffolding (progressive_trail_full_validation.py)
- Day 1: Tests 1~4 (full, WF, OOS, 3-way)
- Day 2: Tests 5~9 (bootstrap, neighborhood, consistency, MC, fold2)
- Day 2: Design 문서 작성
- Day 3: Production code 반영 (enabled=false) + unit tests
- Day 3: Final validation + Report

**Total: ~3 days compact PDCA cycle**.
