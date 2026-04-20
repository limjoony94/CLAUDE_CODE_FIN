# Report: Progressive Trail PDCA 완료

> **Feature**: progressive_trail
> **Date**: 2026-04-21
> **Phase**: Completed
> **Match Rate**: 98%
> **Verdict**: ⭐⭐ 세션 최초 **9/9 CORE + 5/5 WARNINGS FULL GO**
> **Version**: v4.8.0 (`3a669f1`)

---

## Executive Summary

`cand_C_b0.60 + trend_filter(1.0%/192)`에 **progressive trail**(`thr=0.9%, tk_post=0.5`) 적용.
수익 확보(best_pnl ≥ 0.9%) 시점부터 trail K를 aggressive하게 tighten하여 **F6 구조적 문제 해결** +
Sharpe +67%, MDD -37%, PnL +30% 동시 달성.

---

## 1. Plan (2026-04-21)

### Trigger
4-path F6 challenge (MR/Scalping/Relaxed) 전부 실패 → exit 재설계 탐색.
Extended grid (113 combos)에서 profit-based 2-step trail이 단독 지배적 발견.

### Hypothesis
`best_pnl > threshold` 시 trail K를 극단적으로 tighten(0.5) → 소수 대박 의존 ↓, 꾸준한 수익 분포 ↑.

### Artifact
`docs/01-plan/features/progressive_trail.plan.md`

---

## 2. Design (2026-04-21)

### 9-Test Validation Battery
1. Full Period (Clean + Slip LOW/MED/HIGH)
2. Strict Expanding WF 5-fold
3. OOS Split 5 Boundaries
4. 3-Way Split (60/20/20)
5. Bootstrap 3-day × 1000
5b. Bootstrap Relative (per-window P(tgt>base))
6. Neighborhood 5×5 (sharp peak check)
7. Parameter Consistency 2-half
8. MC Direction 999 sims
9. Fold 2 Regime Re-check

### Production Code Spec
- `signals.py`: `get_effective_trail_k(best_pnl)` helper + check_exit 분기
- `bot.py`: `_calc_trail_trigger_price` baton-touch에 dynamic K
- `config.yaml`: `progressive_trail` 섹션

### Artifact
`docs/02-design/features/progressive_trail.design.md`

---

## 3. Do — Validation Results (332.8일, SLIP_MED)

### 핵심 수치 (vs cand_C+trend baseline)

| 지표 | Baseline | Target | Δ |
|------|----------|--------|---|
| PnL | +94.00 | **+122.49** | **+30%** |
| Daily | +0.282% | **+0.368%** | +30% |
| MDD | 7.91 | **4.97** | **-37%** |
| WR | 35.4% | **40.6%** | +5.2pp |
| RR | 2.96 | 2.85 | -0.11 |
| F6 ex_top5 | **-22.01 ❌** | **+40.23 ✅** | **구조적 해결** |
| Sharpe (boot) | 0.319 | **0.533** | **+67%** |
| pos% (boot) | 59.9% | **68.0%** | +8.1pp |

### 9 Core Flags (ALL PASS)
1. ✅ full_clean_gain (Δ +38.20pp)
2. ✅ full_slip_med_gain (Δ +28.49pp)
3. ✅ strict_wf_5of5
4. ✅ oos_split_5of5 (5 boundaries 전부 val&test 양수)
5. ✅ 3way_all_positive (train +52, val +35, test +35)
6. ✅ bootstrap_core (mean +1.22, pos 68%, p5 -2.48)
7. ✅ bootstrap_relative (P(tgt>base) = 64.7%)
8. ✅ neighborhood_8plus (**22/25**, non-sharp-peak)
9. ✅ structural_halves (h1 Δ+7.17, h2 Δ+21.32)

### 5 Warnings (ALL PASS)
- ✓ MC p=0.0000 (beat 999/999 sims)
- ✓ F6 ex_top5 > 0
- ✓ MDD improved -2.94pp
- ✓ Sharpe ≥ 0.45 (0.533)
- ✓ Fold 2 target +5.07 vs base +1.24 (Δ+3.84)

### Slippage Robustness (4 scenarios)
| Scenario | baseline PnL | target PnL | Δ |
|----------|--------------|------------|---|
| Clean | +168.18 | +206.38 | +38.20 |
| Low | +131.09 | +164.43 | +33.34 |
| Med | +94.00 | +122.49 | +28.49 |
| High | +19.82 | +38.61 | +18.79 |

모든 시나리오 우위 유지.

### Artifacts
- Validation: `scripts/analysis/progressive_trail_full_validation.py`
- Result JSON: `results/progressive_trail_validation_20260421_003533.json`
- Production code: `signals.py`, `bot.py`, `config.yaml`
- Tests: `scripts/tests/test_progressive_trail.py` (8 cases, 8/8 PASS)
- Regression: 127/127 pytest PASS

---

## 4. Check — Gap Analysis

### Match Rate: **98%**

| 영역 | Match |
|------|-------|
| signals.py (helper + check_exit) | 100% |
| config.yaml | 100% |
| bot.py (baton-touch dynamic K) | 100% |
| Unit tests (6+ required → 8 actual) | 100% |
| Doc/Memory deliverables | 90% |

### Design vs Implementation 차이
Design §5.1의 inline `prog_cfg` lookup을 **helper-based refactor**로 구현 — DRY + hot-path 최적화.
`signals.py`/`bot.py` 양쪽이 **동일 helper**(`get_effective_trail_k`)로 접근 → Design §7 Risk
"BUG#61 재발 가능성" **구조적 차단**.

### Artifact
`docs/03-analysis/progressive_trail.analysis.md`

---

## 5. Lessons Learned

### 전략적 교훈
1. **Exit 재설계가 진짜 edge였음**: Exit 3축(trail/SL/emergency) 고정 K dead end 이후
   entry selectivity(body)와 regime(trend)를 통해 9/9 GO 도달, 다시 exit에서 **profit-conditional K**
   발견 → BTC 변동성 수확기 전략은 "수익 확보 직후 빠른 lock-in"이 본질.

2. **"Loosen" 직관은 역효과**: Profit 확보 후 더 풀어주면 반전 손실이 이익을 갉아먹음.
   Extended study에서 tkT > 2.5 전 조합 baseline 악화 확인.

3. **F6 해법은 전략 교체 아닌 exit 재설계에 있었음**: 4-path F6 challenge에서
   MR/Scalping/Relaxed 전부 실패한 이유 = F6가 R:R 3:1 trend-following 구조적 상한.
   Progressive trail은 R:R을 3.36 → 2.85로 의도적 축소하되 WR을 35→41%로 상승시켜
   "소수 대박 의존" 제거.

### 프로세스 교훈
4. **Monkey-patch + 기존 엔진 재사용**으로 9-test battery를 단일 script(~500라인)에 압축.
   새 engine 작성 대비 시간 80% 절약, 검증 일관성 보장.

5. **Neighborhood sharp peak check가 과적합 판별의 가장 정확한 gauge**:
   regime_filter_trend는 2/25였으나 progressive_trail은 22/25 → 완전히 다른 robustness 레벨.

6. **단일 helper 추상화로 multi-file 수식 불일치 위험 차단**: signals.py와 bot.py에서
   동일한 `get_effective_trail_k`를 호출 → BUG#61 유형의 재발 방지.

### 정합성 교훈
7. **Pre-activation TRAILING_STOP_MARKET은 구조적 한계 유지**: best_pnl<0.05% 단계에서
   progressive는 의미 없고 (threshold 0.9%와 무관), 기존 baton-touch 구조 그대로 사용 가능.

---

## 6. Production Deploy Plan

### 현재 상태 (deploy-ready)
- ✅ config `enabled: false` (기본 비활성)
- ✅ signals.py/bot.py/config.yaml 반영 완료
- ✅ 127/127 pytest PASS
- ✅ commit `3a669f1` + push

### 30일 LIVE 관찰 조건 (enabled=true 검토 기준)
1. **50+ trades 수집**
2. **Actual slippage ≤ SLIP_MED 범위**
3. **best_pnl 0.9% 돌파 빈도** 모니터링 (BT에서 약 70% trades가 threshold 도달)
4. **tk_post=0.5 baton-touch replace**가 실제 거래소에서 whipsaw 증가 없는지
5. **Baseline vs progressive delta +5pp 이상** (structural Δ 유지)

### Rollback 기준
- 30일 중 baseline 대비 -5pp 이상 악화
- Baton-touch replace 빈도 BT 예측 대비 +50%
- Exchange API error 증가

---

## 7. Next Steps

### Immediate (완료)
- [x] Validation 9/9 CORE + 5/5 WARN PASS
- [x] Production code 반영
- [x] Unit tests 추가
- [x] Memory / CLAUDE.md / commit / push

### Short-term (1-2주)
- [ ] `/pdca archive progressive_trail` (문서 아카이브)
- [ ] BACKTEST_LIVE_PARITY.md에 progressive_trail 항목 추가 (22/22 유지 확인)
- [ ] BUG_HISTORY.md에 dynamic K 관련 방어 패턴 기록

### Medium-term (30일)
- [ ] LIVE 관찰 → 활성화 조건 5개 검증
- [ ] LIVE BT 괴리 재측정 (기존 BT-LIVE parity fix-후 재현)
- [ ] 활성화 PDCA (`progressive_trail_activation`) 생성 또는 직접 enabled=true

---

## 8. Metadata

- **PDCA Duration**: 1일 (동일 session에서 Plan → Design → Do → Check → Report)
- **Iterations**: 0 (1st pass에 98% match)
- **Commit**: `3a669f1`
- **Memory**: `~/.claude/projects/.../memory/progressive_trail_20260421.md`
- **Related**:
  - `regime_filter_trend_20260419.md` — base of base
  - `four_path_exploration_20260420.md` — F6 대안 탐색 dead-end
  - `deep_validation_20260420.md` — structural Δ 개념 정립

---

## 9. Verdict

⭐⭐ **Session 최초 9/9 CORE + 5/5 WARN FULL GO**. 
Production code deploy-ready (enabled=false). 30일 LIVE 관찰 후 활성화 단계 진입.

**이번 세션은 C1 Breakout v2.6의 next-generation 후보를 확정하는 major milestone**.
