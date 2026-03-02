# PDCA Completion Report: Adaptive Leverage (v1.39.0)

> **Feature**: adaptive-leverage
> **Version**: v1.39.0
> **Date**: 2026-03-02
> **Phase**: Plan → Design → Do → Check → Report ✅
> **Verdict**: GO (risk-superior)

---

## 1. Plan Summary

### 목표
고정 3x 레버리지를 동적 1-3x로 전환하여 리스크 관리 개선.
- Rolling WR이 백테스트 기대값(73.2%)에 가까울수록 → full leverage
- Rolling WR이 낮을수록 → 낮은 leverage (edge 불확실)
- R:R 구조를 반영한 5개 방법론 비교

### 가설 (9개)
| ID | 방법 | 설명 |
|----|------|------|
| H1 | StepTiers | WR 구간별 계단식 레버리지 (v1 best) |
| H2 | LinearWR | WR 비례 선형 레버리지 |
| H3 | StepWR_High | 높은 WR 구간 집중 |
| H4 | Inverse Kelly | Kelly fraction 기반 |
| **H5** | **WR Confidence** | **WR/expected × edge/ref_edge (핵심)** |
| H6 | Half-Kelly | Kelly fraction × 0.5 |
| H7 | Breakeven-Aware | breakeven WR 대비 마진 기반 |
| H8 | Combined | H5(60%) + H7(40%) 혼합 |
| H9 | Exp Decay | WR 갭에 대한 지수 감소 |

---

## 2. Design Summary

### 아키텍처
- **Config-driven**: `strategy.adaptive_leverage` 섹션으로 활성화/비활성화
- **State-tracked**: `rolling_wr_tracker` (최근 50거래 이력)
- **Per-slot leverage**: 각 슬롯에 `effective_leverage` 저장
- **Rollback**: `enabled: false`로 즉시 고정 3x 복귀

### 변경 파일 (7개)
| 파일 | 변경 내용 |
|------|----------|
| `bot.py` | `_compute_adaptive_leverage()` + 5 헬퍼 + tracker + entry 통합 |
| `position_open.py` | `leverage_override` 파라미터, per-slot 저장 |
| `position_close.py` | per-slot leverage PnL 계산 |
| `config.py` | `_npos_portfolio_wr`, `_npos_ref_edge` 로딩 |
| `state.py` | `rolling_wr_tracker` 기본값 |
| `models.py` | BotState 타입 확장 |
| `config.yaml` | `adaptive_leverage` 섹션 |

---

## 3. Implementation (Do)

### 연구 3단계

**Phase A: v1 Study** (`adaptive_leverage_study.py`)
- H1-H4 × window [8,10,12,15,20] = 20 시나리오
- **H1_StepTiers_w10**: IS PnL/MDD 33.98 (best), WF 3/3 PASS
- v1 결론: StepTiers가 Fixed 3x 대비 PnL/MDD +20.6%, MDD -16%

**Phase B: v2 Study** (`adaptive_leverage_v2_study.py`)
- H5-H9 × window [8,10,12,15,20] = 25 시나리오 + 3 baseline = 28개
- IS 순위: H1_Step(33.98) > H5_w8(30.63) > H8_w10(34.10) > H5_w12(36.93)
- **PnL/MDD Verdict: STOP** — H1이 여전히 최우수

**Phase C: Risk Study** (`adaptive_leverage_risk_study.py`) — 6-Phase
- 사용자 관찰: "H5_w12 OOS min fold +24.8% vs baseline +18.4% — 리스크 관리 우수"
- 6개 리스크 차원 심층 분석 → **Verdict: GO (7/9 metrics superior)**

### Production 구현
- 7개 파일 수정, `enabled: false` 상태로 구현 후 연구 완료 후 `enabled: true`
- `_compute_adaptive_leverage()` — 5개 method 지원 (wr_confidence 선택)
- `_update_rolling_wr_tracker()` — 거래 종료 시 자동 기록
- aggregate risk cap에서 per-slot `effective_leverage` 사용

---

## 4. Check (연구 결과)

### Phase 1: IS Risk Metrics (Neutral Window 74,722 bars)

| 지표 | Fixed_3x | H5_WRConf_w12 | 변화 |
|------|----------|---------------|------|
| Total PnL% | 126.41 | 119.73 | **-5.3%** |
| Max DD% | 4.75 | **3.24** | **-31.8%** |
| PnL/MDD | 26.60 | **36.93** | **+38.8%** |
| Calmar Ratio | 37.4 | **52.0** | **+39.0%** |
| Ulcer Index | 1.12 | **1.05** | **-6.3%** |
| Worst Daily PnL | -2.45% | -2.41% | +1.6% |
| Daily Sharpe | 0.384 | 0.387 | +0.8% |
| Worst Single Trade | -24.57% | -22.73% | +7.5% |
| Avg Leverage | 3.00x | 2.55x | -15.0% |
| Avg Loss Leverage | 3.00x | 2.65x | -11.7% |

### Phase 2: Leverage Stress Response

DD 구간별 레버리지 자동 감축:

| DD Threshold | Calm Lev | Stress Lev | Reduction |
|-------------|----------|------------|-----------|
| ≥ 0.5% | 2.93x | 1.97x | **-32.9%** |
| ≥ 1.0% | 2.81x | 1.70x | **-39.4%** |
| ≥ 2.0% | 2.63x | 1.39x | **-47.4%** |

→ 드로다운이 깊어질수록 자동으로 레버리지를 더 줄임 (antifragile)

### Phase 3: WF OOS Fold Stability (3-fold)

| 시나리오 | F1 PnL | F2 PnL | F3 PnL | Min | Std | Consistency |
|----------|--------|--------|--------|-----|-----|-------------|
| Fixed_3x | +18.4% | +23.5% | +43.6% | +18.4% | 10.9 | 0.645 |
| **H5_w12** | **+24.8%** | **+30.5%** | **+35.1%** | **+24.8%** | **4.2** | **0.823** |
| H8_w10 | +22.5% | +31.4% | +30.4% | +22.5% | 4.0 | 0.801 |

- **H5_w12 Consistency 0.823** — 전 시나리오 중 최고 (fold 간 분산 최소)
- Min fold +24.8% vs baseline +18.4% (**+35% 개선**)
- 모두 WF **3/3 PASS**

### Phase 4: Monte Carlo Robustness (1,000 sims)

| 시나리오 | Actual MDD | MC Mean | MC 95% | MC 99% | MC Max |
|----------|-----------|---------|--------|--------|--------|
| Fixed_3x | 4.75% | 6.59% | 10.41% | 13.05% | 19.38% |
| H5_w12 | 3.24% | 5.67% | 8.95% | 10.79% | 17.71% |
| H8_w10 | 3.72% | 4.76% | **7.15%** | 9.06% | 11.48% |

### Phase 5: Worst 30-Day Window

| 시나리오 | Worst 30d PnL |
|----------|--------------|
| Fixed_3x | -0.60% |
| **H5_w12** | **-0.42%** |
| H8_w10 | -0.47% |

### Phase 6: Risk Verdict Score Card

**vs Fixed_3x baseline (9 metrics)**:

| 시나리오 | Superior Metrics | Score |
|----------|-----------------|-------|
| H1_StepTiers_w10 | 6/9 | 67% |
| **H5_WRConf_w12** | **7/9** | **78%** |
| H5_WRConf_w8 | 6/9 | 67% |
| **H8_Combined_w10** | **7/9** | **78%** |
| H9_ExpDecay_w10 | 6/9 | 67% |

**GO 기준: ≥60% (6/9+) → 모두 PASS**
**Best: H5_WRConf_w12 & H8_Combined_w10 (7/9)**

H5_w12 선택 이유 (vs H8):
- OOS Consistency 0.823 > 0.801 (더 안정적)
- PnL/MDD 36.93 > 34.10 (더 높은 위험 대비 수익)
- 단일 method로 구현이 단순 (H8은 두 method의 blend)

---

## 5. Production Config

```yaml
strategy:
  adaptive_leverage:
    enabled: true           # v1.39.0
    method: "wr_confidence"
    min_leverage: 1.0
    max_leverage: 3.0
    window: 12
    expected_wr: 73.2       # npos_portfolio IS stats (%)
    ref_edge: 0.126         # npos_portfolio pnl_per_trade (%)
```

### 동작 원리
1. 최초 3거래까지: `min_leverage = 1.0x` (데이터 부족)
2. 이후: `rolling_WR / expected_WR × rolling_edge / ref_edge` → 1.0~3.0x
3. 드로다운 발생 시: WR 하락 → 자동으로 leverage 감소
4. WR 회복 시: 자동으로 leverage 증가

### Rollback
```yaml
adaptive_leverage:
  enabled: false  # 즉시 고정 3x 복귀
```

---

## 6. Test Results

```
1061 tests passed (0 failed, 0 errors)
```

---

## 7. Key Learnings

### 발견
1. **PnL 최적화 ≠ 리스크 최적화**: v2 PnL/MDD verdict STOP이었지만, 리스크 관점에서는 명확한 GO
2. **Antifragile 특성**: DD 구간에서 레버리지 자동 감축(-47%)이 MDD 제한에 핵심 기여
3. **OOS Consistency가 핵심 지표**: fold 간 분산(std 4.2 vs 10.9)이 전략 안정성의 가장 좋은 지표
4. **Kelly는 R:R 불리 구조에서 실패**: H6_Kelly는 leverage ~1.05x → 이 전략의 R:R 구조(mean 0.478)에서 Kelly가 과소 평가
5. **연구 순서가 중요**: PnL 연구 → STOP → 리스크 연구 → GO. 단일 차원 평가로는 불완전

### 위험 요소
- 처음 3거래까지 1x로 시작 → 초기 수익 기회 일부 미스
- Rolling window 12가 충분히 반응적인지 실전 검증 필요
- Live에서 WR이 기대치 대비 크게 하락하면 장기간 1x 운영 가능

---

## 8. Metrics Summary

| 항목 | 값 |
|------|-----|
| 연구 스크립트 | 3개 (v1 + v2 + risk) |
| 가설 검증 | 9개 (H1-H4 + H5-H9) |
| Window sweep | 5개 (8, 10, 12, 15, 20) |
| 총 시나리오 | 48+ (baselines + hypotheses × windows) |
| WF folds | 3-fold expanding window |
| MC simulations | 1,000회 |
| Production 파일 수정 | 7개 |
| 테스트 | 1,061 passed |
| 커밋 | `30015fe` |

---

## 9. PDCA Cycle Status

```
[Plan] ✅ → [Design] ✅ → [Do] ✅ → [Check] ✅ → [Report] ✅
```

**Phase**: completed
**Match Rate**: 100% (설계 대비 구현 완전 일치)
**Iteration**: 0 (first pass 완료)
