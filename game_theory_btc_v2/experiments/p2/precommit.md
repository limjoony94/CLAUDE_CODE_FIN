# P2 Pre-Commit — Force-Flow Reversal Hypothesis Testing

**Pre-commit date**: 2026-05-01
**Priority**: P2
**Mandate basis**: § P2 (force-flow reversal H3-H5 + H7-basis Phase A subset)
**Authority**: Pre-committed BEFORE any P2 strategy code. **Mutability**: changes require new amendment.

---

## Hypotheses Tested (Phase A only)

| H | Mechanism | Direction | Sample window |
|---|-----------|-----------|---------------|
| H3 | M001 force_flow_long_proxy | long | 540d 1h (free) |
| H4 | M003 cascade_window_5m_long | long | 185d 1m (free intersect 365d sealed-180d) |
| H5 | M002 force_flow_short_proxy | short | 540d 1h |
| H7-basis | M004 basis_fade_perp | bidirectional | 540d 1h |

Phase A active set 14 mechanisms 중 force-flow 관련 4개. 나머지 10개 (momentum/breakout-control/pattern/regime/cross-section)는 P3에서 MAP-Elites archive 평가.

---

## Selection Rule = Option α (Pre-Registered Single Config)

**Advisor 2026-05-01 lock**: Option α 채택. Items 1 (sweep methodology) + 2 (multiple comparisons) 동시 해결.

Each mechanism은 **단일 parameter config**으로 평가. 후속 sweep 금지. 결과 PASS 못하면 FAIL closure (post-hoc tuning 금지).

### Locked Configs (P2 single evaluation per mechanism)

#### M001 force_flow_long_proxy (H3)
```yaml
threshold_high: 5.5         # median of [4.0, 5.5, 7.0]
threshold_low_quantile: 0.30  # less restrictive of [0.20, 0.30]
forward_window_hours: 4
direction: long
data: btc_perp_1h_720d (free, 540d)
```

#### M002 force_flow_short_proxy (H5)
```yaml
threshold_high: 5.5
threshold_high_quantile: 0.70  # mirror of M001
forward_window_min: 60
direction: short
data: btc_perp_1h_720d (free, 540d)
```

#### M003 cascade_window_5m_long (H4)
```yaml
threshold_high: 5.5
forward_window_min: 15  # mid of [5, 15, 30]
direction: long
data: btc_perp_1m_365d (free first 185d) + btc_funding_binance_720d
```

#### M004 basis_fade_perp (H7-basis)
```yaml
basis_z_threshold: 2.0  # mid of [1.5, 2.0, 2.5]
forward_h: 4
direction: bidirectional (sign of basis_z)
data: btc_perp_1h_720d + btc_spot_1h_720d (free, 540d)
```

### Single-Config Rationale
- Multiple comparisons FWER 1.0 회피 (sweep 시 100+ evals × α=0.05)
- "Best params" optimization 환상 회피
- 명확한 PASS/FAIL boundary
- Post-hoc tuning 차단

---

## 6-Criteria Gate Application

각 mechanism × scenario (realistic + stress) 별도 평가. 8 evaluations total (4 mech × 2 scenario):

```python
# For each mechanism × scenario:
strategy_pnl = run_mechanism(M00X, locked_config, scenario)
result = bootstrap_six_criteria(
    daily_pnl=strategy_pnl,
    baseline_pnl=buy_and_hold_same_window,  # mandatory P2+
    priority="P2",
    B=10000,
    block_size=3,
    seed=42,
)
```

P2 thresholds (six_criteria_thresholds.md v2):
- target_daily=0.10%, max_dd_floor=-3%, min_pos_rate=0.5
- min_p_beats=0.70, min_sharpe=1.5, p5 (bs lower)≥0
- baseline_pnl mandatory

### PASS Definition
**PASS** = realistic 6/6 PASS AND stress 6/6 PASS (둘 다)
**PARTIAL** = realistic 6/6 PASS, stress < 6/6 (deploy 금지)
**FAIL** = realistic < 6/6

### Closure Decision
- Per-mechanism PASS/PARTIAL/FAIL 보고
- 종합 P2 closure:
  - ≥1 mechanism PASS → P3 entry candidate
  - 모두 FAIL → P2 envelope falsified, Phase B 활성화 ROI 재평가
  - PARTIAL only → defer + paper trade plan

---

## PASS/FAIL Criteria Rule (Strict Pre-Commitment)

| Outcome | Criteria | Action |
|---------|----------|--------|
| PASS | 8/8 (4 mech × 2 scenario) all PASS | escalate to P3 immediately |
| PARTIAL | ≥1 mech realistic PASS + stress FAIL | deploy 금지, P3 deferred candidate |
| FAIL | All 8 evaluations FAIL | P2 closure, document + Phase B re-eval |

---

## Anti-Fishing Locks

1. ❌ **No parameter sweep**: 위 locked config만 사용. 다른 값 실험 금지
2. ❌ **No threshold tuning**: 6-criteria thresholds (six_criteria_thresholds.md v2) 변경 금지
3. ❌ **No selection bias**: PASS mechanism만 보고 금지. **8/8 evaluations 모두 보고**
4. ❌ **No baseline cherry-pick**: buy_and_hold_same_window MANDATORY (P0.4 baselines_result.md 사용)
5. ❌ **No silent re-scope**: H3-H5/H7-basis만 P2. H6/H1/H7-full는 Phase B
6. ❌ **No "borderline PASS" rationalization**: 4/6 또는 5/6 = FAIL
7. ✅ FAIL closure는 expected outcome (friction-floor prior). FAIL을 honest closure로 보고
8. ✅ Sealed window assert via `assert_no_sealed_data()` 의무 호출 (모든 P2 코드 시작 시)

---

## Pre-Sweep Frequency Validation (advisor non-blocking)

P2 strategy code 실행 전:
1. Each mechanism × locked config 별 event frequency 측정 (full free window)
2. `min_event_freq_gate` (registry's default 0.5/day) 미달 시 **vacuous** 라벨 (R38 lesson)
3. Vacuous mechanism은 6-criteria 평가 skip, FAIL로 카운트 (mandate § 0.7 honest closure)

`scripts/analysis/p2_frequency_scan.py` 실행 → 결과 `experiments/p2/frequency_scan.md`.

---

## Stopping Rule

- **Hard limit**: 5 days from P2 entry (mandate § P2)
- **Per-mechanism timeout**: B=10000 bootstrap should complete <5 min per mechanism. If >30 min → optimize or escalate.
- **Honest closure**: 결과 PASS/PARTIAL/FAIL 모두 `experiments/p2/result.md`에 보고.

---

## Closure Output (P2 Day-N)

`experiments/p2/result.md`에 의무 포함:
- 8 evaluations (4 mech × 2 scenario) 6-criteria table
- Frequency scan results
- Sealed boundary assertion log (no leak occurred)
- PASS/PARTIAL/FAIL per mechanism + 종합
- GO/NOGO for P3
- Honest discussion of friction-floor prior 정합 여부

---

**Pre-commit signed**: Claude Code agent, 2026-05-01.
P2 entry 시점 IMMUTABLE. 변경 시 새 amendment 의무.
